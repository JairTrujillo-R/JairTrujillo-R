from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataSource:
    name: str
    url: str
    records_key: str | None = None
    enabled: bool = True
    default_strategy: str | None = None
    default_game: str | None = None


OFFICIAL_SOURCES: list[DataSource] = [
    DataSource(
        name="datos_abiertos_colombia_loterias",
        url="https://www.datos.gov.co/resource/i3kx-3zps.json",
        records_key=None,
    ),
    DataSource(
        name="datos_abiertos_colombia_chance",
        url="https://www.datos.gov.co/resource/8hn7-rpp8.json",
        records_key=None,
        enabled=False,
    ),
    DataSource(
        name="datos_abiertos_colombia_baloto",
        url="https://www.datos.gov.co/resource/2fgt-r8bt.json",
        records_key=None,
        enabled=False,
    ),
]


def load_sources() -> list[DataSource]:
    sources = [source for source in OFFICIAL_SOURCES if source.enabled]
    extra_sources_path = os.getenv("LOTTERY_SOURCES_JSON")
    if not extra_sources_path:
        return sources

    path = Path(extra_sources_path)
    if not path.exists():
        return sources

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        return sources

    for item in payload:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        url = item.get("url")
        if not name or not url:
            continue
        sources.append(
            DataSource(
                name=str(name).strip(),
                url=str(url).strip(),
                records_key=item.get("records_key"),
                enabled=True,
                default_strategy=item.get("default_strategy"),
                default_game=item.get("default_game"),
            )
        )
    return sources
