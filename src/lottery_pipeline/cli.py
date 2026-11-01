from __future__ import annotations

import argparse
import logging
import smtplib
from pathlib import Path

import pandas as pd

from .collector import LotteryDataCollector, build_metadata_snapshot
from .db import LotteryRepository
from .modeling import LotteryForecaster
from .notifier import EmailNotifier, SMTPSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline de loterías para Colombia.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db_parser = subparsers.add_parser("init-db", help="Inicializa la base de datos.")
    init_db_parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/lottery.db"),
        help="Ruta de la base de datos SQLite.",
    )

    collect_parser = subparsers.add_parser("collect", help="Extrae datos y guarda CSV.")
    collect_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/lottery_results.csv"),
        help="Ruta de salida para datos normalizados.",
    )
    collect_parser.add_argument(
        "--limit",
        type=int,
        default=50000,
        help="Límite de registros por fuente.",
    )
    collect_parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("data/raw/metadata.json"),
        help="Ruta de salida para metadatos de fuentes.",
    )
    collect_parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/lottery.db"),
        help="Ruta de la base de datos SQLite.",
    )
    collect_parser.add_argument(
        "--frequency",
        type=str,
        default=None,
        choices=["daily", "weekly", "monthly", "all"],
        help="Filtra por frecuencia de juego.",
    )

    train_parser = subparsers.add_parser(
        "train", help="Entrena modelos por estrategia y genera predicción."
    )
    train_parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/lottery_results.csv"),
        help="CSV de entrada con datos normalizados.",
    )
    train_parser.add_argument(
        "--models-output",
        type=Path,
        default=Path("models/lottery_models.pkl"),
        help="Archivo de salida para modelos serializados.",
    )
    train_parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("predictions/daily_predictions.csv"),
        help="CSV de salida para predicciones diarias.",
    )
    train_parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/lottery.db"),
        help="Ruta de la base de datos SQLite.",
    )
    train_parser.add_argument(
        "--prediction-date",
        type=str,
        default=None,
        help="Fecha a predecir (YYYY-MM-DD). Por defecto usa hoy UTC.",
    )
    train_parser.add_argument(
        "--min-rows",
        type=int,
        default=30,
        help="Mínimo de filas por estrategia/juego para entrenar.",
    )
    train_parser.add_argument(
        "--frequency",
        type=str,
        default=None,
        choices=["daily", "weekly", "monthly", "all"],
        help="Filtra por frecuencia de juego.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evalúa predicciones pendientes contra resultados oficiales."
    )
    evaluate_parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/lottery.db"),
        help="Ruta de la base de datos SQLite.",
    )
    evaluate_parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Fecha máxima de evaluación (YYYY-MM-DD).",
    )

    notify_parser = subparsers.add_parser(
        "notify", help="Envía correo cuando haya aciertos sin alerta previa."
    )
    notify_parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/lottery.db"),
        help="Ruta de la base de datos SQLite.",
    )

    test_email_parser = subparsers.add_parser(
        "test-email", help="Envía un correo de prueba usando la configuración SMTP."
    )
    test_email_parser.add_argument(
        "--recipient",
        type=str,
        default=None,
        help="Destinatario de la prueba. Por defecto usa LOTTERY_ALERT_RECIPIENT.",
    )

    cycle_parser = subparsers.add_parser(
        "run-cycle", help="Ejecuta ciclo completo: collect -> train -> evaluate -> notify."
    )
    cycle_parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/lottery.db"),
        help="Ruta de la base de datos SQLite.",
    )
    cycle_parser.add_argument(
        "--limit",
        type=int,
        default=50000,
        help="Límite de registros por fuente.",
    )
    cycle_parser.add_argument(
        "--frequency",
        type=str,
        default="all",
        choices=["daily", "weekly", "monthly", "all"],
        help="Frecuencia del ciclo.",
    )
    cycle_parser.add_argument(
        "--prediction-date",
        type=str,
        default=None,
        help="Fecha objetivo para predicción (YYYY-MM-DD).",
    )
    cycle_parser.add_argument(
        "--min-rows",
        type=int,
        default=30,
        help="Mínimo de filas por estrategia/juego para entrenar.",
    )

    return parser


def init_db_command(db_path: Path) -> None:
    repository = LotteryRepository(db_path)
    repository.close()
    print(f"Base de datos inicializada en: {db_path}")


def collect_command(
    output: Path, metadata_output: Path, limit: int, db_path: Path, frequency: str | None
) -> pd.DataFrame:
    collector = LotteryDataCollector()
    df = collector.collect(limit=limit)
    if df.empty:
        raise RuntimeError("No se obtuvieron datos válidos desde las fuentes configuradas.")
    if frequency and frequency != "all":
        df = df[df["frequency"] == frequency]
    if df.empty:
        raise RuntimeError("No hay datos para la frecuencia solicitada.")
    collector.save_to_csv(df, output)
    build_metadata_snapshot(metadata_output)
    repository = LotteryRepository(db_path)
    inserted = repository.save_official_results(df)
    repository.close()
    print(f"Datos guardados en: {output} | resultados oficiales insertados: {inserted}")
    return df


def train_command(
    input_path: Path,
    models_output: Path,
    predictions_output: Path,
    db_path: Path,
    prediction_date: str | None,
    min_rows: int,
    frequency: str | None,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise RuntimeError("El archivo de entrada no contiene datos.")

    df["draw_date"] = pd.to_datetime(df["draw_date"], errors="coerce")
    df["number"] = pd.to_numeric(df["number"], errors="coerce")
    if "frequency" not in df.columns:
        df["frequency"] = "daily"
    if "min_number" not in df.columns:
        df["min_number"] = 0
    if "max_number" not in df.columns:
        df["max_number"] = 9999
    df = df.dropna(subset=["draw_date", "number", "strategy", "game"])
    if frequency and frequency != "all":
        df = df[df["frequency"] == frequency]
    if df.empty:
        raise RuntimeError("No hay datos válidos para entrenar.")

    forecaster = LotteryForecaster(min_rows=min_rows)
    models = forecaster.train_models(df)
    if not models:
        raise RuntimeError(
            "No se entrenaron modelos. Verifica cantidad de datos y columnas de estrategia/juego."
        )

    forecast_date = pd.to_datetime(prediction_date) if prediction_date else None
    predictions = forecaster.predict_next_day(models, prediction_date=forecast_date)

    LotteryForecaster.save_models(models, models_output)
    repository = LotteryRepository(db_path)
    model_version_map: dict[tuple[str, str], int] = {}
    features = ["day_of_week", "day_of_month", "month", "lag1", "lag7", "mean7"]
    for (strategy, game), trained in models.items():
        group_rows = int(len(df[(df["strategy"] == strategy) & (df["game"] == game)]))
        model_version_map[(strategy, game)] = repository.create_model_version(
            strategy=strategy,
            game=game,
            model_path=models_output,
            data_rows=group_rows,
            algorithm="RandomForestRegressor+BayesianSmoothing",
            features=features,
        )
    repository.save_predictions(predictions, model_version_map=model_version_map)
    repository.close()
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_output, index=False)
    print(f"Modelos guardados en: {models_output}")
    print(f"Predicciones guardadas en: {predictions_output}")
    return predictions


def evaluate_command(db_path: Path, evaluation_date: str | None) -> pd.DataFrame:
    repository = LotteryRepository(db_path)
    evaluations = repository.evaluate_due_predictions(evaluation_date=evaluation_date)
    repository.close()
    if evaluations.empty:
        print("No hubo predicciones pendientes para evaluar.")
    else:
        hits = int(evaluations["is_hit"].sum())
        print(f"Evaluaciones realizadas: {len(evaluations)} | aciertos: {hits}")
    return evaluations


def notify_command(db_path: Path) -> int:
    settings = SMTPSettings.from_env()
    if not settings.is_complete():
        print("Credenciales SMTP incompletas. Se omite envío de correo.")
        return 0

    repository = LotteryRepository(db_path)
    notifier = EmailNotifier(settings)
    sent = 0
    try:
        for hit in repository.pending_hit_alerts():
            subject = f"🎯 Acierto de lotería: {hit['strategy']} - {hit['game']}"
            body = (
                f"Predicción acertada para {hit['prediction_date']}.\n"
                f"Juego: {hit['strategy']} / {hit['game']}\n"
                f"Número predicho: {hit['predicted_number']}\n"
                f"Número real: {hit['actual_number']}\n"
            )
            try:
                notifier.send_hit_alert(subject=subject, body=body)
                repository.mark_alert(int(hit["prediction_id"]), settings.recipient, "sent", "ok")
                sent += 1
            except (OSError, smtplib.SMTPException) as exc:
                repository.mark_alert(
                    int(hit["prediction_id"]), settings.recipient, "failed", str(exc)
                )
    finally:
        repository.close()
    print(f"Alertas enviadas: {sent}")
    return sent


def test_email_command(recipient: str | None) -> None:
    settings = SMTPSettings.from_env()
    target = recipient or settings.recipient
    if not settings.is_complete():
        raise RuntimeError("Credenciales SMTP incompletas; no se puede enviar el correo de prueba.")
    if not target:
        raise RuntimeError("Debe indicar un destinatario para el correo de prueba.")

    notifier = EmailNotifier(
        SMTPSettings(
            host=settings.host,
            port=settings.port,
            username=settings.username,
            password=settings.password,
            sender=settings.sender,
            recipient=target,
            use_tls=settings.use_tls,
        )
    )
    notifier.send_hit_alert(
        subject="Prueba de alertas - Lottery Pipeline",
        body=(
            "Este es un correo de prueba del pipeline de loterías.\n"
            "La configuración SMTP de producción respondió correctamente."
        ),
    )
    print(f"Correo de prueba enviado a: {target}")


def run_cycle_command(
    db_path: Path, limit: int, frequency: str, prediction_date: str | None, min_rows: int
) -> None:
    output = Path(f"data/raw/lottery_results_{frequency}.csv")
    collect_command(
        output=output,
        metadata_output=Path("data/raw/metadata.json"),
        limit=limit,
        db_path=db_path,
        frequency=frequency,
    )
    train_command(
        input_path=output,
        models_output=Path(f"models/lottery_models_{frequency}.pkl"),
        predictions_output=Path(f"predictions/{frequency}_predictions.csv"),
        db_path=db_path,
        prediction_date=prediction_date,
        min_rows=min_rows,
        frequency=frequency,
    )
    evaluate_command(db_path=db_path, evaluation_date=None)
    notify_command(db_path=db_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init-db":
        init_db_command(db_path=args.db_path)
        return

    if args.command == "collect":
        collect_command(
            output=args.output,
            metadata_output=args.metadata_output,
            limit=args.limit,
            db_path=args.db_path,
            frequency=args.frequency,
        )
        return

    if args.command == "train":
        train_command(
            input_path=args.input,
            models_output=args.models_output,
            predictions_output=args.predictions_output,
            db_path=args.db_path,
            prediction_date=args.prediction_date,
            min_rows=args.min_rows,
            frequency=args.frequency,
        )
        return

    if args.command == "evaluate":
        evaluate_command(db_path=args.db_path, evaluation_date=args.date)
        return

    if args.command == "notify":
        notify_command(db_path=args.db_path)
        return

    if args.command == "test-email":
        test_email_command(recipient=args.recipient)
        return

    run_cycle_command(
        db_path=args.db_path,
        limit=args.limit,
        frequency=args.frequency,
        prediction_date=args.prediction_date,
        min_rows=args.min_rows,
    )


if __name__ == "__main__":
    main()
