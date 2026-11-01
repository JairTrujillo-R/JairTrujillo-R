from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from lottery_pipeline.collector import LotteryDataCollector
from lottery_pipeline.config import DataSource, load_sources
from lottery_pipeline.db import LotteryRepository
from lottery_pipeline.ensemble import LotteryEnsembleForecaster
from lottery_pipeline.modeling import LotteryForecaster
from lottery_pipeline.notifier import EmailNotifier, SMTPSettings, render_hit_email


def build_training_data(rows: int = 16) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "draw_date": dates,
            "strategy": ["chance"] * rows,
            "game": ["chance_nacional"] * rows,
            "frequency": ["daily"] * rows,
            "bet_type": ["numero_directo"] * rows,
            "min_number": [0] * rows,
            "max_number": [9] * rows,
            "number": [index % 10 for index in range(rows)],
        }
    )


def test_collector_normalizes_official_record() -> None:
    collector = LotteryDataCollector(
        sources=[
            DataSource(
                name="test",
                url="https://example.test",
                default_strategy="loteria",
                default_game="loteria_medellin",
            )
        ]
    )

    result = collector._normalize_records(
        [
            {
                "fecha_sorteo": "2026-01-02",
                "modalidad": "Chance",
                "juego": "Chance Nacional",
                "numero_ganador": "0042",
            }
        ],
        collector.sources[0],
    )

    assert result.iloc[0]["draw_date"] == pd.Timestamp("2026-01-02")
    assert result.iloc[0]["strategy"] == "chance"
    assert result.iloc[0]["game"] == "chance_nacional"
    assert result.iloc[0]["number"] == 42


def test_default_source_is_official_lottery_results_dataset() -> None:
    source = load_sources()[0]

    assert source.url.endswith("/resource/i3kx-3zps.json")


def test_collector_normalizes_open_data_lottery_fields() -> None:
    source = DataSource(
        name="datos_abiertos_colombia_loterias",
        url="https://www.datos.gov.co/resource/i3kx-3zps.json",
        default_strategy="loteria",
    )
    collector = LotteryDataCollector(sources=[source])

    result = collector._normalize_records(
        [
            {
                "fecha": "2026-10-09",
                "loter_a": "Lotería de Medellín",
                "numero_billete_ganador": "1234",
                "tipo_de_premio": "Mayor",
            }
        ],
        source,
    )

    assert len(result) == 1
    assert result.iloc[0]["strategy"] == "loteria"
    assert result.iloc[0]["game"] == "loteria_de_medellin"
    assert result.iloc[0]["number"] == 1234
    assert result.iloc[0]["prize_type"] == "mayor"


def test_collector_preserves_secondary_prizes() -> None:
    source = DataSource(
        name="datos_abiertos_colombia_loterias",
        url="https://www.datos.gov.co/resource/i3kx-3zps.json",
        default_strategy="loteria",
    )
    collector = LotteryDataCollector(sources=[source])

    result = collector._normalize_records(
        [
            {
                "fecha_del_sorteo": "2026-10-09",
                "loter_a": "Loteria del Quindio",
                "numero_billete_ganador": "9876",
                "tipo_de_premio": "Seco",
            }
        ],
        source,
    )

    assert len(result) == 1
    assert result.iloc[0]["prize_type"] == "seco"


def test_forecaster_trains_and_returns_bounded_prediction() -> None:
    forecaster = LotteryForecaster(min_rows=8, random_state=7)
    models = forecaster.train_models(build_training_data())

    predictions = forecaster.predict_next_day(
        models, prediction_date=pd.Timestamp("2026-02-01"), top_k=3
    )

    assert ("chance", "chance_nacional") in models
    assert len(predictions) == 1
    row = predictions.iloc[0]
    assert 0 <= row["predicted_number"] <= 9
    assert len(row["top_candidates"]) == 3
    assert sum(item["probability"] for item in row["top_candidates"]) <= 1


def test_forecaster_uses_major_number_once_when_secondary_rows_exist() -> None:
    data = build_training_data(16)
    secondary = data.copy()
    secondary["prize_type"] = "seco"
    secondary["number"] = (secondary["number"] + 1) % 10
    data["prize_type"] = "mayor"
    combined = pd.concat([data, secondary], ignore_index=True)

    models = LotteryForecaster(min_rows=8).train_models(combined)

    assert models[("chance", "chance_nacional")].total_observations == 16


def test_forecaster_skips_games_not_scheduled_for_target_date() -> None:
    data = build_training_data(16)
    data["frequency"] = "weekly"
    data["draw_date"] = pd.date_range("2026-01-03", periods=16, freq="7D")
    forecaster = LotteryForecaster(min_rows=8)
    models = forecaster.train_models(data)

    predictions = forecaster.predict_next_day(
        models, prediction_date=pd.Timestamp("2026-01-04"), top_k=3
    )

    assert predictions.empty


def test_ensemble_trains_parallel_candidates_and_normalizes_weights() -> None:
    data = build_training_data(40)
    forecaster = LotteryEnsembleForecaster(min_rows=8, enable_neural=False, max_workers=3)

    models = forecaster.train_models(data)
    model = models[("chance", "chance_nacional")]
    predictions = forecaster.predict_next_day(
        models, prediction_date=pd.Timestamp("2026-02-15"), top_k=5
    )

    assert set(model.candidates) == {"random_forest", "extra_trees", "hist_gradient_boosting"}
    assert math.isclose(sum(model.weights.values()), 1.0)
    assert len(predictions) == 1
    assert math.isclose(
        sum(item["probability"] for item in predictions.iloc[0]["top_candidates"]),
        sum(item["probability"] for item in predictions.iloc[0]["top_candidates"]),
    )


def test_ensemble_walk_forward_uses_expanding_history() -> None:
    data = build_training_data(24)
    forecaster = LotteryEnsembleForecaster(min_rows=8, enable_neural=False)

    results = forecaster.walk_forward_evaluate(data, min_train_rows=8)

    assert len(results) > 0
    assert results["draw_date"].min() >= pd.Timestamp("2026-01-09").date()
    assert results["actual_probability"].between(0, 1).all()


def test_repository_evaluates_one_result_per_draw(tmp_path: Path) -> None:
    repository = LotteryRepository(tmp_path / "lottery.db")
    try:
        official = pd.DataFrame(
            [
                {
                    "draw_date": "2026-02-01",
                    "strategy": "chance",
                    "game": "chance_nacional",
                    "number": 7,
                    "prize_type": "mayor",
                    "source": "test",
                    "raw_payload": "{}",
                },
                {
                    "draw_date": "2026-02-01",
                    "strategy": "chance",
                    "game": "chance_nacional",
                    "number": 8,
                    "prize_type": "seco",
                    "source": "test",
                    "raw_payload": "{}",
                },
            ]
        )
        assert repository.save_official_results(official) == 2

        predictions = pd.DataFrame(
            [
                {
                    "prediction_date": "2026-02-01",
                    "strategy": "chance",
                    "game": "chance_nacional",
                    "predicted_number": 8,
                    "predicted_probability": 0.6,
                    "top_candidates": [
                        {"number": 8, "probability": 0.6},
                        {"number": 7, "probability": 0.2},
                    ],
                }
            ]
        )
        repository.save_predictions(predictions, {})

        evaluations = repository.evaluate_due_predictions("2026-02-01")

        assert len(evaluations) == 1
        assert evaluations.iloc[0]["actual_number"] == 7
        assert evaluations.iloc[0]["is_hit"]
        assert evaluations.iloc[0]["secondary_prize_hit"]
        assert evaluations.iloc[0]["matched_prize_types"] == ["seco"]
        assert math.isclose(evaluations.iloc[0]["brier_score"], 0.16)
    finally:
        repository.close()


def test_end_to_end_training_persistence_and_evaluation(tmp_path: Path) -> None:
    data = build_training_data(16)
    data["draw_date"] = data["draw_date"].dt.strftime("%Y-%m-%d")
    input_path = tmp_path / "results.csv"
    data.to_csv(input_path, index=False)

    forecaster = LotteryForecaster(min_rows=8)
    models = forecaster.train_models(pd.read_csv(input_path))
    predictions = forecaster.predict_next_day(
        models, prediction_date=pd.Timestamp("2026-01-17"), top_k=5
    )
    model_path = tmp_path / "models.pkl"
    LotteryForecaster.save_models(models, model_path)

    repository = LotteryRepository(tmp_path / "lottery.db")
    try:
        model_version = repository.create_model_version(
            "chance", "chance_nacional", model_path, len(data), "test", []
        )
        repository.save_predictions(predictions, {("chance", "chance_nacional"): model_version})
        actual = predictions.iloc[0]["predicted_number"]
        repository.save_official_results(
            pd.DataFrame(
                [
                    {
                        "draw_date": "2026-01-17",
                        "strategy": "chance",
                        "game": "chance_nacional",
                        "number": actual,
                    }
                ]
            )
        )
        evaluations = repository.evaluate_due_predictions("2026-01-17")
    finally:
        repository.close()

    assert model_path.exists()
    assert predictions.iloc[0]["top_candidates"]
    assert len(evaluations) == 1
    assert bool(evaluations.iloc[0]["is_hit"])


def test_email_notifier_sends_message_without_network(monkeypatch) -> None:
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            sent["connection"] = (host, port, timeout)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def starttls(self):
            sent["tls"] = True

        def login(self, username, password):
            sent["login"] = (username, password)

        def send_message(self, message):
            sent["message"] = message

    monkeypatch.setattr("lottery_pipeline.notifier.smtplib.SMTP", FakeSMTP)
    settings = SMTPSettings(
        "smtp.example.com", 587, "user", "secret", "from@example.com", "to@example.com"
    )

    assert EmailNotifier(settings).send_hit_alert("Subject", "Body") == "sent"
    assert sent["connection"] == ("smtp.example.com", 587, 30)
    assert sent["message"]["To"] == "to@example.com"


def test_email_notifier_adds_html_alternative(monkeypatch) -> None:
    class FakeSMTP:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def starttls(self):
            pass

        def login(self, *_args):
            pass

        def send_message(self, message):
            assert message.get_body("html") is not None

    monkeypatch.setattr("lottery_pipeline.notifier.smtplib.SMTP", FakeSMTP)
    settings = SMTPSettings(
        "smtp.example.com", 587, "user", "secret", "from@example.com", "to@example.com"
    )
    EmailNotifier(settings).send_hit_alert("Subject", "Body", "<p>Body</p>")


def test_hit_email_renders_one_and_multiple_hits() -> None:
    rows = [
        {
            "prediction_date": "2026-11-01",
            "strategy": "loteria",
            "game": "loteria_medellin",
            "predicted_number": 1234,
            "matched_prize_types": ["seco"],
        },
        {
            "prediction_date": "2026-11-01",
            "strategy": "loteria",
            "game": "loteria_bogota",
            "predicted_number": 5678,
            "matched_prize_types": ["mayor"],
        },
    ]
    plain, html = render_hit_email(rows)
    assert "2 aciertos de lotería" in plain
    assert "Premios mayores: 1" in plain
    assert "Premios secundarios: 1" in plain
    assert "<table" in html
    assert "loteria_medellin" in html
