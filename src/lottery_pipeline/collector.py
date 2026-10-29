from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import DataSource, OFFICIAL_SOURCES


class LotteryDataCollector:
    """Extrae datos de fuentes oficiales y los normaliza."""

    def __init__(self, sources: list[DataSource] | None = None, timeout: int = 30) -> None:
        self.sources = sources or OFFICIAL_SOURCES
        self.timeout = timeout

    def collect(self, limit: int = 50000) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for source in self.sources:
            try:
                raw_records = self._fetch_source(source, limit=limit)
                normalized = self._normalize_records(raw_records, source.name)
                if not normalized.empty:
                    frames.append(normalized)
            except requests.RequestException:
                continue

        if not frames:
            return pd.DataFrame(
                columns=["draw_date", "strategy", "game", "number", "source", "raw_payload"]
            )
        return pd.concat(frames, ignore_index=True).sort_values("draw_date")

    def save_to_csv(self, df: pd.DataFrame, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        return output

    def _fetch_source(self, source: DataSource, limit: int) -> list[dict[str, Any]]:
        params = {"$limit": limit}
        response = requests.get(source.url, params=params, timeout=self.timeout)
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
            game = self._pick_value(record, ["juego", "loteria", "nombre_juego", "game"], "desconocido")
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

            normalized_rows.append(
                {
                    "draw_date": parsed_date.date(),
                    "strategy": str(strategy).strip(),
                    "game": str(game).strip(),
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
        "created_at_utc": datetime.utcnow().isoformat(),
        "sources": [asdict(source) for source in OFFICIAL_SOURCES],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([snapshot]).to_json(path, orient="records", indent=2)
    return path

