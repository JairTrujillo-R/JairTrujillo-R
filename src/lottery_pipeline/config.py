from dataclasses import dataclass


@dataclass(frozen=True)
class DataSource:
    name: str
    url: str
    records_key: str | None = None


OFFICIAL_SOURCES: list[DataSource] = [
    DataSource(
        name="datos_abiertos_colombia",
        url="https://www.datos.gov.co/resource/xjzv-xx6x.json",
        records_key=None,
    ),
]

