from __future__ import annotations

import math
import pickle
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPRegressor

from .modeling import LotteryForecaster

FEATURES = ["day_of_week", "day_of_month", "month", "lag1", "lag7", "mean7"]


@dataclass
class CandidateModel:
    name: str
    estimator: Any


@dataclass
class EnsembleModel:
    strategy: str
    game: str
    candidates: dict[str, CandidateModel]
    weights: dict[str, float]
    min_number: int
    max_number: int
    frequency_counts: dict[int, int]
    total_observations: int
    dispersion: float
    latest_number: int
    latest_lag7: float
    latest_mean7: float
    frequency: str
    draw_weekdays: tuple[int, ...]
    draw_month_days: tuple[int, ...]
    validation_scores: dict[str, float]


class LotteryEnsembleForecaster:
    """Train and evaluate several independent forecasters per game.

    The ensemble weights are learned only from an expanding temporal validation
    split. This is supervised model selection, not reinforcement learning.
    """

    def __init__(
        self,
        min_rows: int = 30,
        random_state: int = 42,
        max_workers: int | None = None,
        enable_neural: bool = True,
        neural_min_rows: int = 100,
    ) -> None:
        self.min_rows = min_rows
        self.random_state = random_state
        self.max_workers = max_workers
        self.enable_neural = enable_neural
        self.neural_min_rows = neural_min_rows

    def train_models(self, df: pd.DataFrame) -> dict[tuple[str, str], EnsembleModel]:
        if df.empty:
            return {}
        models: dict[tuple[str, str], EnsembleModel] = {}
        for key, raw_group in df.groupby(["strategy", "game"], dropna=False):
            group = LotteryForecaster._training_rows(raw_group).sort_values("draw_date")
            features = LotteryForecaster._build_features(group)
            if len(features) < self.min_rows:
                continue
            models[(str(key[0]), str(key[1]))] = self._fit_group(str(key[0]), str(key[1]), group)
        return models

    def predict_next_day(
        self,
        models: dict[tuple[str, str], EnsembleModel],
        prediction_date: pd.Timestamp | None = None,
        top_k: int = 10,
    ) -> pd.DataFrame:
        target_date = pd.to_datetime(prediction_date or pd.Timestamp.utcnow().normalize())
        rows: list[dict[str, Any]] = []
        for model in models.values():
            if not self._is_scheduled_date(model, target_date):
                continue
            features = pd.DataFrame(
                [
                    {
                        "day_of_week": target_date.day_of_week,
                        "day_of_month": target_date.day,
                        "month": target_date.month,
                        "lag1": model.latest_number,
                        "lag7": model.latest_lag7,
                        "mean7": model.latest_mean7,
                    }
                ]
            )
            distribution = self._ensemble_distribution(model, features)
            top = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:top_k]
            rows.append(
                {
                    "prediction_date": target_date.date(),
                    "strategy": model.strategy,
                    "game": model.game,
                    "predicted_number": int(top[0][0]),
                    "predicted_probability": float(top[0][1]),
                    "top_candidates": [
                        {"number": int(number), "probability": float(probability)}
                        for number, probability in top
                    ],
                    "model_weights": model.weights,
                }
            )
        return pd.DataFrame(rows)

    def walk_forward_evaluate(
        self, df: pd.DataFrame, min_train_rows: int | None = None
    ) -> pd.DataFrame:
        """Evaluate future observations using only data available before each date."""
        required = min_train_rows or self.min_rows
        results: list[dict[str, Any]] = []
        for key, raw_group in df.groupby(["strategy", "game"], dropna=False):
            group = LotteryForecaster._training_rows(raw_group).sort_values("draw_date")
            group["draw_date"] = pd.to_datetime(group["draw_date"])
            for index in range(required, len(group)):
                train = group.iloc[:index]
                test = group.iloc[index]
                models = self.train_models(train)
                model = models.get((str(key[0]), str(key[1])))
                if model is None:
                    continue
                target = pd.Timestamp(test["draw_date"])
                if not self._is_scheduled_date(model, target):
                    continue
                features = pd.DataFrame(
                    [
                        {
                            "day_of_week": target.day_of_week,
                            "day_of_month": target.day,
                            "month": target.month,
                            "lag1": model.latest_number,
                            "lag7": model.latest_lag7,
                            "mean7": model.latest_mean7,
                        }
                    ]
                )
                distribution = self._ensemble_distribution(model, features)
                actual = int(test["number"])
                probability = float(distribution.get(actual, 0.0))
                results.append(
                    {
                        "strategy": str(key[0]),
                        "game": str(key[1]),
                        "draw_date": target.date(),
                        "actual_number": actual,
                        "predicted_number": max(distribution, key=distribution.get),
                        "actual_probability": probability,
                        "brier_score": (1.0 - probability) ** 2,
                        "weights": model.weights,
                    }
                )
        return pd.DataFrame(results)

    def save_models(
        self, models: dict[tuple[str, str], EnsembleModel], output_path: str | Path
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(models, file)
        return path

    def _fit_group(self, strategy: str, game: str, group: pd.DataFrame) -> EnsembleModel:
        features = LotteryForecaster._build_features(group)
        X = features[FEATURES]
        y = features["number"]
        factories = self._factories(len(features))
        candidates: dict[str, CandidateModel] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers or len(factories)) as executor:
            jobs = {executor.submit(factory, X, y): name for name, factory in factories.items()}
            for job in as_completed(jobs):
                name = jobs[job]
                try:
                    candidates[name] = CandidateModel(name, job.result())
                except ValueError:
                    continue
                except RuntimeError:
                    continue
        if not candidates:
            raise RuntimeError(f"No se pudo entrenar ningún modelo para {strategy}/{game}")
        validation_scores = self._validation_scores(group, factories)
        weights = self._scores_to_weights(validation_scores, candidates)
        last_row = features.iloc[-1]
        min_number = int(pd.to_numeric(group.get("min_number", 0), errors="coerce").min())
        max_number = int(pd.to_numeric(group.get("max_number", 9999), errors="coerce").max())
        counts = (
            pd.to_numeric(group["number"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
            .to_dict()
        )
        dispersion = float(pd.to_numeric(group["number"], errors="coerce").std())
        if math.isnan(dispersion) or dispersion <= 0:
            dispersion = max(1.0, (max_number - min_number) / 20)
        return EnsembleModel(
            strategy=strategy,
            game=game,
            candidates=candidates,
            weights=weights,
            min_number=max(0, min_number),
            max_number=max(min_number, max_number),
            frequency_counts=counts,
            total_observations=int(sum(counts.values())),
            dispersion=dispersion,
            latest_number=int(last_row["number"]),
            latest_lag7=float(last_row["lag7"]),
            latest_mean7=float(last_row["mean7"]),
            frequency=str(group["frequency"].mode().iloc[0])
            if "frequency" in group and not group["frequency"].mode().empty
            else "daily",
            draw_weekdays=tuple(sorted(pd.to_datetime(group["draw_date"]).dt.day_of_week.unique())),
            draw_month_days=tuple(sorted(pd.to_datetime(group["draw_date"]).dt.day.unique())),
            validation_scores=validation_scores,
        )

    def _factories(self, rows: int) -> dict[str, Callable[[pd.DataFrame, pd.Series], Any]]:
        factories: dict[str, Callable[[pd.DataFrame, pd.Series], Any]] = {
            "random_forest": lambda X, y: RandomForestRegressor(
                n_estimators=200, min_samples_leaf=2, random_state=self.random_state, n_jobs=1
            ).fit(X, y),
            "extra_trees": lambda X, y: ExtraTreesRegressor(
                n_estimators=200, min_samples_leaf=2, random_state=self.random_state, n_jobs=1
            ).fit(X, y),
            "hist_gradient_boosting": lambda X, y: HistGradientBoostingRegressor(
                max_iter=150, l2_regularization=1.0, random_state=self.random_state
            ).fit(X, y),
        }
        if self.enable_neural and rows >= self.neural_min_rows:
            factories["mlp"] = lambda X, y: MLPRegressor(
                hidden_layer_sizes=(32, 16),
                early_stopping=True,
                max_iter=400,
                random_state=self.random_state,
            ).fit(X, y)
        return factories

    def _validation_scores(
        self, group: pd.DataFrame, factories: dict[str, Callable[[pd.DataFrame, pd.Series], Any]]
    ) -> dict[str, float]:
        features = LotteryForecaster._build_features(group)
        split = max(self.min_rows, int(len(features) * 0.7))
        if split >= len(features):
            return {name: 1.0 for name in factories}
        train, valid = features.iloc[:split], features.iloc[split:]
        X_train, y_train = train[FEATURES], train["number"]
        counts = y_train.astype(int).value_counts().to_dict()
        low = int(pd.to_numeric(group.get("min_number", 0), errors="coerce").min())
        high = int(pd.to_numeric(group.get("max_number", 9999), errors="coerce").max())
        dispersion = float(y_train.std())
        if math.isnan(dispersion) or dispersion <= 0:
            dispersion = max(1.0, (high - low) / 20)
        scores: dict[str, float] = {}
        for name, factory in factories.items():
            try:
                estimator = factory(X_train, y_train)
                predictions = estimator.predict(valid[FEATURES])
                brier_scores = []
                for point, actual in zip(predictions, valid["number"], strict=True):
                    distribution = self._distribution_from_history(
                        counts, len(y_train), low, high, dispersion, float(point)
                    )
                    brier_scores.append((1.0 - distribution.get(int(actual), 0.0)) ** 2)
                scores[name] = float(sum(brier_scores) / len(brier_scores))
            except ValueError:
                scores[name] = float("inf")
            except RuntimeError:
                scores[name] = float("inf")
        return scores

    @staticmethod
    def _scores_to_weights(
        scores: dict[str, float], candidates: dict[str, CandidateModel]
    ) -> dict[str, float]:
        finite = {
            name: score
            for name, score in scores.items()
            if name in candidates and math.isfinite(score)
        }
        if not finite:
            finite = {name: 1.0 for name in candidates}
        inverse = {name: 1.0 / max(score, 1e-9) for name, score in finite.items()}
        total = sum(inverse.values())
        return {name: value / total for name, value in inverse.items()}

    def _ensemble_distribution(
        self, model: EnsembleModel, features: pd.DataFrame
    ) -> dict[int, float]:
        distributions: dict[str, dict[int, float]] = {}
        for name, candidate in model.candidates.items():
            point = float(candidate.estimator.predict(features[FEATURES])[0])
            distributions[name] = self._distribution(model, point)
        result = {number: 0.0 for number in range(model.min_number, model.max_number + 1)}
        for name, distribution in distributions.items():
            for number, probability in distribution.items():
                result[number] += model.weights.get(name, 0.0) * probability
        total = sum(result.values())
        return {number: value / total for number, value in result.items()} if total else result

    @staticmethod
    def _distribution(model: EnsembleModel, point: float) -> dict[int, float]:
        return LotteryEnsembleForecaster._distribution_from_history(
            model.frequency_counts,
            model.total_observations,
            model.min_number,
            model.max_number,
            model.dispersion,
            point,
        )

    @staticmethod
    def _distribution_from_history(
        counts: dict[int, int],
        observations: int,
        low: int,
        high: int,
        dispersion: float,
        point: float,
    ) -> dict[int, float]:
        span = max(1, high - low + 1)
        scores: dict[int, float] = {}
        for number in range(low, high + 1):
            historical = (counts.get(number, 0) + 1.0) / (observations + span)
            scores[number] = historical * math.exp(-abs(number - point) / max(dispersion, 1.0))
        total = sum(scores.values())
        return {number: score / total for number, score in scores.items()}

    @staticmethod
    def _is_scheduled_date(model: EnsembleModel, target: pd.Timestamp) -> bool:
        if model.frequency == "weekly":
            return target.day_of_week in model.draw_weekdays
        if model.frequency == "monthly":
            return target.day in model.draw_month_days
        return True
