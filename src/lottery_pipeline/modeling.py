from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


@dataclass
class TrainedModel:
    strategy: str
    game: str
    model: RandomForestRegressor
    latest_date: pd.Timestamp
    latest_number: int
    latest_lag7: float
    latest_mean7: float


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
            models[(str(strategy), str(game))] = TrainedModel(
                strategy=str(strategy),
                game=str(game),
                model=model,
                latest_date=pd.to_datetime(last_row["draw_date"]),
                latest_number=int(last_row["number"]),
                latest_lag7=float(last_row["lag7"]),
                latest_mean7=float(last_row["mean7"]),
            )

        return models

    def predict_next_day(
        self,
        models: dict[tuple[str, str], TrainedModel],
        prediction_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if not models:
            return pd.DataFrame(columns=["prediction_date", "strategy", "game", "predicted_number"])

        target_date = pd.to_datetime(prediction_date or pd.Timestamp.utcnow().normalize())
        rows: list[dict[str, Any]] = []
        for trained in models.values():
            features = pd.DataFrame(
                [
                    {
                        "day_of_week": target_date.dayofweek,
                        "day_of_month": target_date.day,
                        "month": target_date.month,
                        "lag1": trained.latest_number,
                        "lag7": trained.latest_lag7,
                        "mean7": trained.latest_mean7,
                    }
                ]
            )
            prediction = trained.model.predict(features)[0]
            rows.append(
                {
                    "prediction_date": target_date.date(),
                    "strategy": trained.strategy,
                    "game": trained.game,
                    "predicted_number": max(0, int(round(prediction))),
                }
            )
        return pd.DataFrame(rows).sort_values(["strategy", "game"])

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
        data["day_of_week"] = data["draw_date"].dt.dayofweek
        data["day_of_month"] = data["draw_date"].dt.day
        data["month"] = data["draw_date"].dt.month
        data["lag1"] = data["number"].shift(1)
        data["lag7"] = data["number"].shift(7)
        data["mean7"] = data["number"].rolling(7).mean().shift(1)
        data = data.dropna(subset=["lag1", "lag7", "mean7", "number"])
        return data

