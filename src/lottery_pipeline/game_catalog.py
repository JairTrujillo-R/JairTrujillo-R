from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameDefinition:
    strategy: str
    game: str
    frequency: str
    draw_time_local: str
    timezone: str
    bet_type: str
    min_number: int
    max_number: int


GAME_CATALOG: list[GameDefinition] = [
    GameDefinition(
        strategy="chance",
        game="chance_nacional",
        frequency="daily",
        draw_time_local="21:00",
        timezone="America/Bogota",
        bet_type="numero_directo",
        min_number=0,
        max_number=9999,
    ),
    GameDefinition(
        strategy="loteria",
        game="loteria_bogota",
        frequency="weekly",
        draw_time_local="22:25",
        timezone="America/Bogota",
        bet_type="numero_directo",
        min_number=0,
        max_number=9999,
    ),
    GameDefinition(
        strategy="loteria",
        game="loteria_medellin",
        frequency="weekly",
        draw_time_local="23:00",
        timezone="America/Bogota",
        bet_type="numero_directo",
        min_number=0,
        max_number=9999,
    ),
    GameDefinition(
        strategy="baloto",
        game="baloto_tradicional",
        frequency="weekly",
        draw_time_local="23:00",
        timezone="America/Bogota",
        bet_type="multibola",
        min_number=1,
        max_number=45,
    ),
    GameDefinition(
        strategy="astro",
        game="super_astro",
        frequency="daily",
        draw_time_local="22:00",
        timezone="America/Bogota",
        bet_type="numero_y_signo",
        min_number=0,
        max_number=9999,
    ),
    GameDefinition(
        strategy="extraordinario",
        game="sorteo_extraordinario",
        frequency="monthly",
        draw_time_local="20:00",
        timezone="America/Bogota",
        bet_type="numero_directo",
        min_number=0,
        max_number=9999,
    ),
]


def catalog_map() -> dict[tuple[str, str], GameDefinition]:
    return {(item.strategy.lower(), item.game.lower()): item for item in GAME_CATALOG}
