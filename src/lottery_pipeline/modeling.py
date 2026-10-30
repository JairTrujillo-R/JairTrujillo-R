from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


@dataclass
class TrainedModel:
    strategy: str
    game: str
    model: RandomForestRegressor
    min_number: int
    max_number: int
    frequency_counts: dict[int, int]
    total_observations: int
    dispersion: float
    latest_date: pd.Timestamp
    latest_number: int
    latest_lag7: float
    latest_mean7: float
    frequency: str
    draw_weekdays: tuple[int, ...]
    draw_month_days: tuple[int, ...]


class LotteryForecaster:
    """Entrena modelos por estrategia y juego para predecir el siguiente sorteo."""

    def __init__(self, min_rows: int = 30, random_state: int = 42) -> None:
        self.min_rows = min_rows
        self.random_state = random_state

    def train_models(self, df: pd.DataFrame) -> dict[tuple[str, str], TrainedModel]:
        if df.empty:
            return {}

        models: dict[tuple[str, str], TrainedModel] = {}
        for (strategy, game), group in df.groupby(["strategy", "game"], dropna=False):
            features_df = self._build_features(group.sort_values("draw_date"))
            if len(features_df) < self.min_rows:
                continue

            X = features_df[["day_of_week", "day_of_month", "month", "lag1", "lag7", "mean7"]]
            y = features_df["number"]
            model = RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=self.random_state,
            )
            model.fit(X, y)

            last_row = features_df.iloc[-1]
            min_number = int(
                pd.to_numeric(group.get("min_number", pd.Series([0])), errors="coerce").min()
            )
            max_number = int(
                pd.to_numeric(group.get("max_number", pd.Series([9999])), errors="coerce").max()
            )
            min_number = max(0, min_number)
            max_number = max(min_number, max_number)
            frequency_counts = (
                pd.to_numeric(group["number"], errors="coerce")
                .dropna()
                .astype(int)
                .value_counts()
                .to_dict()
            )
            total_observations = int(sum(frequency_counts.values()))
            dispersion = float(pd.to_numeric(group["number"], errors="coerce").std())
            if math.isnan(dispersion) or dispersion <= 0:
                dispersion = max(1.0, (max_number - min_number) / 20)
            models[(str(strategy), str(game))] = TrainedModel(
                strategy=str(strategy),
                game=str(game),
                model=model,
                min_number=min_number,
                max_number=max_number,
                frequency_counts=frequency_counts,
                total_observations=total_observations,
                dispersion=dispersion,
                latest_date=pd.to_datetime(last_row["draw_date"]),
                latest_number=int(last_row["number"]),
                latest_lag7=float(last_row["lag7"]),
                latest_mean7=float(last_row["mean7"]),
                frequency=str(group["frequency"].mode().iloc[0])
                if "frequency" in group and not group["frequency"].mode().empty
                else "daily",
                draw_weekdays=tuple(
                    sorted(pd.to_datetime(group["draw_date"]).dt.day_of_week.unique())
                ),
                draw_month_days=tuple(sorted(pd.to_datetime(group["draw_date"]).dt.day.unique())),
            )

        return models

    def predict_next_day(
        self,
        models: dict[tuple[str, str], TrainedModel],
        prediction_date: pd.Timestamp | None = None,
        top_k: int = 10,
    ) -> pd.DataFrame:
        if not models:
            return pd.DataFrame(
                columns=[
                    "prediction_date",
                    "strategy",
                    "game",
                    "predicted_number",
                    "predicted_probability",
                    "top_candidates",
                ]
            )

        target_date = pd.to_datetime(prediction_date or pd.Timestamp.utcnow().normalize())
        rows: list[dict[str, Any]] = []
        for trained in models.values():
            if not self._is_scheduled_date(trained, target_date):
                continue
            features = pd.DataFrame(
                [
                    {
                        "day_of_week": target_date.day_of_week,
                        "day_of_month": target_date.day,
                        "month": target_date.month,
                        "lag1": trained.latest_number,
                        "lag7": trained.latest_lag7,
                        "mean7": trained.latest_mean7,
                    }
                ]
            )
            prediction = float(trained.model.predict(features)[0])
            probabilities = self._build_probability_distribution(trained, prediction)
            top_candidates = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[
                :top_k
            ]
            best_number, best_probability = top_candidates[0]
            rows.append(
                {
                    "prediction_date": target_date.date(),
                    "strategy": trained.strategy,
                    "game": trained.game,
                    "predicted_number": int(best_number),
                    "predicted_probability": float(best_probability),
                    "top_candidates": [
                        {"number": int(number), "probability": float(probability)}
                        for number, probability in top_candidates
                    ],
                }
            )
        predictions = pd.DataFrame(rows)
        if predictions.empty:
            return pd.DataFrame(
                columns=[
                    "prediction_date",
                    "strategy",
                    "game",
                    "predicted_number",
                    "predicted_probability",
                    "top_candidates",
                ]
            )
        return predictions.sort_values(["strategy", "game"])

    @staticmethod
    def _is_scheduled_date(trained: TrainedModel, target_date: pd.Timestamp) -> bool:
        if trained.frequency == "daily":
            return True
        if trained.frequency == "weekly":
            return target_date.day_of_week in trained.draw_weekdays
        if trained.frequency == "monthly":
            return target_date.day in trained.draw_month_days
        return True

    @staticmethod
    def save_models(models: dict[tuple[str, str], TrainedModel], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(models, file)
        return path

    @staticmethod
    def _build_features(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["draw_date"] = pd.to_datetime(data["draw_date"])
        data = data.sort_values("draw_date")
        data["min_number"] = pd.to_numeric(data.get("min_number", 0), errors="coerce").fillna(0)
        data["max_number"] = pd.to_numeric(data.get("max_number", 9999), errors="coerce").fillna(
            9999
        )
        data["day_of_week"] = data["draw_date"].dt.day_of_week
        data["day_of_month"] = data["draw_date"].dt.day
        data["month"] = data["draw_date"].dt.month
        data["lag1"] = data["number"].shift(1)
        data["lag7"] = data["number"].shift(7)
        data["mean7"] = data["number"].rolling(7).mean().shift(1)
        data = data.dropna(subset=["lag1", "lag7", "mean7", "number"])
        return data

    def _build_probability_distribution(
        self, trained: TrainedModel, point_prediction: float
    ) -> dict[int, float]:
        low = trained.min_number
        high = trained.max_number
        span = max(1, high - low + 1)
        smoothing = 1.0
        dispersion = max(1.0, trained.dispersion)
        score_map: dict[int, float] = {}
        total_score = 0.0
        for number in range(low, high + 1):
            count = trained.frequency_counts.get(number, 0)
            historical_prob = (count + smoothing) / (trained.total_observations + smoothing * span)
            distance = abs(number - point_prediction)
            distance_factor = math.exp(-distance / dispersion)
            score = historical_prob * distance_factor
            score_map[number] = score
            total_score += score
        if total_score <= 0:
            return {number: 1.0 / span for number in range(low, high + 1)}
        return {number: score / total_score for number, score in score_map.items()}
