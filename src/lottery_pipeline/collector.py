from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import DataSource, load_sources
from .game_catalog import catalog_map


class LotteryDataCollector:
    """Extrae datos de fuentes oficiales y los normaliza."""

    def __init__(self, sources: list[DataSource] | None = None, timeout: int = 30) -> None:
        self.sources = sources or load_sources()
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.7, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self._catalog = catalog_map()

    def collect(self, limit: int = 50000) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for source in self.sources:
            try:
                raw_records = self._fetch_source(source, limit=limit)
                normalized = self._normalize_records(raw_records, source.name)
                if not normalized.empty:
                    frames.append(normalized)
                self.logger.info(
                    "source_processed", extra={"source": source.name, "rows": len(normalized)}
                )
            except requests.RequestException:
                self.logger.exception("source_request_error", extra={"source": source.name})
                continue

        if not frames:
            return pd.DataFrame(
                columns=[
                    "draw_date",
                    "strategy",
                    "game",
                    "frequency",
                    "bet_type",
                    "min_number",
                    "max_number",
                    "number",
                    "source",
                    "raw_payload",
                ]
            )
        return pd.concat(frames, ignore_index=True).sort_values("draw_date")

    def save_to_csv(self, df: pd.DataFrame, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        return output

    def _fetch_source(self, source: DataSource, limit: int) -> list[dict[str, Any]]:
        params = {"$limit": limit}
        response = self.session.get(source.url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        if source.records_key:
            records = payload.get(source.records_key, [])
        elif isinstance(payload, list):
            records = payload
        else:
            records = payload.get("results", [])

        if not isinstance(records, list):
            raise ValueError(f"Formato inesperado en {source.name}")
        return records

    @staticmethod
    def _pick_value(record: dict[str, Any], keys: list[str], default: Any = None) -> Any:
        for key in keys:
            if key in record and record[key] not in (None, ""):
                return record[key]
        return default

    def _normalize_records(self, records: list[dict[str, Any]], source_name: str) -> pd.DataFrame:
        normalized_rows = []
        for record in records:
            draw_date = self._pick_value(
                record,
                ["fecha", "fecha_sorteo", "draw_date", "fecharesultado", "f_sorteo", "date"],
            )
            game = self._pick_value(
                record, ["juego", "loteria", "nombre_juego", "game"], "desconocido"
            )
            strategy = self._pick_value(record, ["modalidad", "tipo_juego", "strategy"], game)
            number = self._pick_value(
                record,
                ["numero", "numero_ganador", "resultado", "winning_number", "cifras", "ganador"],
            )

            if draw_date is None or number is None:
                continue

            parsed_date = pd.to_datetime(draw_date, errors="coerce")
            parsed_number = pd.to_numeric(str(number).strip(), errors="coerce")
            if pd.isna(parsed_date) or pd.isna(parsed_number):
                continue

            strategy_clean = str(strategy).strip().lower().replace(" ", "_")
            game_clean = str(game).strip().lower().replace(" ", "_")
            catalog_item = self._catalog.get((strategy_clean, game_clean))
            normalized_rows.append(
                {
                    "draw_date": parsed_date.date(),
                    "strategy": strategy_clean,
                    "game": game_clean,
                    "frequency": catalog_item.frequency if catalog_item else "daily",
                    "bet_type": catalog_item.bet_type if catalog_item else "numero_directo",
                    "min_number": catalog_item.min_number if catalog_item else 0,
                    "max_number": catalog_item.max_number if catalog_item else 9999,
                    "number": int(parsed_number),
                    "source": source_name,
                    "raw_payload": str(record),
                }
            )

        df = pd.DataFrame(normalized_rows)
        if df.empty:
            return df

        df["draw_date"] = pd.to_datetime(df["draw_date"])
        df = df.drop_duplicates(subset=["draw_date", "strategy", "game", "number"])
        return df


def build_metadata_snapshot(output_path: str | Path) -> Path:
    snapshot = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "sources": [asdict(source) for source in load_sources()],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([snapshot]).to_json(path, orient="records", indent=2)
    return path
