from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .game_catalog import GAME_CATALOG


class LotteryRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.init_db()
        self.seed_catalog()

    def close(self) -> None:
        self.conn.close()

    def init_db(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS game_catalog (
                strategy TEXT NOT NULL,
                game TEXT NOT NULL,
                frequency TEXT NOT NULL,
                draw_time_local TEXT NOT NULL,
                timezone TEXT NOT NULL,
                bet_type TEXT NOT NULL,
                min_number INTEGER NOT NULL,
                max_number INTEGER NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (strategy, game)
            );

            CREATE TABLE IF NOT EXISTS official_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                draw_date TEXT NOT NULL,
                strategy TEXT NOT NULL,
                game TEXT NOT NULL,
                actual_number INTEGER NOT NULL,
                source TEXT,
                raw_payload TEXT,
                loaded_at_utc TEXT NOT NULL,
                UNIQUE(draw_date, strategy, game, actual_number)
            );

            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                game TEXT NOT NULL,
                trained_at_utc TEXT NOT NULL,
                model_path TEXT NOT NULL,
                data_rows INTEGER NOT NULL,
                algorithm TEXT NOT NULL,
                features TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TEXT NOT NULL,
                strategy TEXT NOT NULL,
                game TEXT NOT NULL,
                predicted_number INTEGER NOT NULL,
                predicted_probability REAL NOT NULL,
                top_candidates_json TEXT NOT NULL,
                model_version_id INTEGER,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at_utc TEXT NOT NULL,
                UNIQUE(prediction_date, strategy, game),
                FOREIGN KEY(model_version_id) REFERENCES model_versions(id)
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL UNIQUE,
                draw_date TEXT NOT NULL,
                actual_number INTEGER NOT NULL,
                is_hit INTEGER NOT NULL,
                brier_score REAL NOT NULL,
                hit_at_3 INTEGER NOT NULL,
                hit_at_5 INTEGER NOT NULL,
                evaluated_at_utc TEXT NOT NULL,
                FOREIGN KEY(prediction_id) REFERENCES predictions(id)
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                alert_type TEXT NOT NULL,
                channel TEXT NOT NULL,
                recipient TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                sent_at_utc TEXT NOT NULL,
                UNIQUE(prediction_id, alert_type, channel, recipient),
                FOREIGN KEY(prediction_id) REFERENCES predictions(id)
            );
            """
        )
        self.conn.commit()

    def seed_catalog(self) -> None:
        for item in GAME_CATALOG:
            row = asdict(item)
            self.conn.execute(
                """
                INSERT INTO game_catalog(
                    strategy, game, frequency, draw_time_local, timezone, bet_type, min_number, max_number
                ) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(strategy, game) DO UPDATE SET
                    frequency=excluded.frequency,
                    draw_time_local=excluded.draw_time_local,
                    timezone=excluded.timezone,
                    bet_type=excluded.bet_type,
                    min_number=excluded.min_number,
                    max_number=excluded.max_number
                """,
                (
                    row["strategy"],
                    row["game"],
                    row["frequency"],
                    row["draw_time_local"],
                    row["timezone"],
                    row["bet_type"],
                    row["min_number"],
                    row["max_number"],
                ),
            )
        self.conn.commit()

    def save_official_results(self, df: pd.DataFrame) -> int:
        inserted = 0
        utc_now = datetime.now(timezone.utc).isoformat()
        for _, row in df.iterrows():
            cursor = self.conn.execute(
                """
                INSERT OR IGNORE INTO official_results(
                    draw_date, strategy, game, actual_number, source, raw_payload, loaded_at_utc
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    pd.to_datetime(row["draw_date"]).date().isoformat(),
                    str(row["strategy"]),
                    str(row["game"]),
                    int(row["number"]),
                    str(row.get("source", "")),
                    str(row.get("raw_payload", "")),
                    utc_now,
                ),
            )
            inserted += cursor.rowcount
        self.conn.commit()
        return inserted

    def create_model_version(
        self,
        strategy: str,
        game: str,
        model_path: str | Path,
        data_rows: int,
        algorithm: str,
        features: list[str],
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO model_versions(strategy, game, trained_at_utc, model_path, data_rows, algorithm, features)
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                strategy,
                game,
                datetime.now(timezone.utc).isoformat(),
                str(model_path),
                int(data_rows),
                algorithm,
                json.dumps(features),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def save_predictions(self, predictions: pd.DataFrame, model_version_map: dict[tuple[str, str], int]) -> int:
        inserted = 0
        utc_now = datetime.now(timezone.utc).isoformat()
        for _, row in predictions.iterrows():
            model_version_id = model_version_map.get((str(row["strategy"]), str(row["game"])))
            top_candidates_json = json.dumps(row["top_candidates"])
            self.conn.execute(
                """
                INSERT INTO predictions(
                    prediction_date, strategy, game, predicted_number, predicted_probability,
                    top_candidates_json, model_version_id, status, created_at_utc
                ) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(prediction_date, strategy, game) DO UPDATE SET
                    predicted_number=excluded.predicted_number,
                    predicted_probability=excluded.predicted_probability,
                    top_candidates_json=excluded.top_candidates_json,
                    model_version_id=excluded.model_version_id,
                    status='pending',
                    created_at_utc=excluded.created_at_utc
                """,
                (
                    pd.to_datetime(row["prediction_date"]).date().isoformat(),
                    str(row["strategy"]),
                    str(row["game"]),
                    int(row["predicted_number"]),
                    float(row["predicted_probability"]),
                    top_candidates_json,
                    model_version_id,
                    "pending",
                    utc_now,
                ),
            )
            inserted += 1
        self.conn.commit()
        return inserted

    def evaluate_due_predictions(self, evaluation_date: str | None = None) -> pd.DataFrame:
        date_filter = evaluation_date or datetime.now(timezone.utc).date().isoformat()
        rows = self.conn.execute(
            """
            SELECT p.id AS prediction_id, p.prediction_date, p.strategy, p.game, p.predicted_number,
                   p.predicted_probability, p.top_candidates_json, r.actual_number
            FROM predictions p
            JOIN official_results r
              ON r.draw_date = p.prediction_date
             AND r.strategy = p.strategy
             AND r.game = p.game
            WHERE p.status = 'pending'
              AND p.prediction_date <= ?
            """,
            (date_filter,),
        ).fetchall()

        evaluations: list[dict[str, Any]] = []
        utc_now = datetime.now(timezone.utc).isoformat()
        for row in rows:
            top_candidates = json.loads(row["top_candidates_json"])
            predicted_number = int(row["predicted_number"])
            actual_number = int(row["actual_number"])
            prob = float(row["predicted_probability"])
            is_hit = int(predicted_number == actual_number)
            top_numbers = [int(item["number"]) for item in top_candidates]
            hit_at_3 = int(actual_number in top_numbers[:3])
            hit_at_5 = int(actual_number in top_numbers[:5])
            brier = (1.0 - prob) ** 2 if is_hit else (prob**2)
            self.conn.execute(
                """
                INSERT OR REPLACE INTO evaluations(
                    prediction_id, draw_date, actual_number, is_hit, brier_score, hit_at_3, hit_at_5, evaluated_at_utc
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    int(row["prediction_id"]),
                    str(row["prediction_date"]),
                    actual_number,
                    is_hit,
                    float(brier),
                    hit_at_3,
                    hit_at_5,
                    utc_now,
                ),
            )
            self.conn.execute("UPDATE predictions SET status='evaluated' WHERE id=?", (int(row["prediction_id"]),))
            evaluations.append(
                {
                    "prediction_id": int(row["prediction_id"]),
                    "prediction_date": row["prediction_date"],
                    "strategy": row["strategy"],
                    "game": row["game"],
                    "predicted_number": predicted_number,
                    "actual_number": actual_number,
                    "is_hit": bool(is_hit),
                    "brier_score": float(brier),
                    "hit_at_3": bool(hit_at_3),
                    "hit_at_5": bool(hit_at_5),
                }
            )

        self.conn.commit()
        return pd.DataFrame(evaluations)

    def pending_hit_alerts(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT p.id AS prediction_id, p.prediction_date, p.strategy, p.game, p.predicted_number, e.actual_number
            FROM predictions p
            JOIN evaluations e ON e.prediction_id = p.id
            LEFT JOIN alerts a
              ON a.prediction_id = p.id
             AND a.alert_type = 'hit'
             AND a.channel = 'email'
            WHERE e.is_hit = 1
              AND a.id IS NULL
            """
        ).fetchall()

    def mark_alert(self, prediction_id: int, recipient: str, status: str, message: str) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO alerts(
                prediction_id, alert_type, channel, recipient, status, message, sent_at_utc
            ) VALUES(?,?,?,?,?,?,?)
            """,
            (
                prediction_id,
                "hit",
                "email",
                recipient,
                status,
                message,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

