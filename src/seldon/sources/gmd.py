"""Global Macro Database — годовые макроданные по странам с 1870-х. Скачивание: `download gmd`."""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW

GMD_VERSION = "2026_09"  # релиз Global Macro Database; список версий — helpers/versions.csv
GMD_FILE = DATA_RAW / "gmd" / f"GMD_{GMD_VERSION}.dta"


def load_gmd(columns: list[str] | None = None) -> pd.DataFrame:
    """Таблица GMD как есть (страна — `ISO3`, `countryname`; год — `year`)."""
    return pd.read_stata(GMD_FILE, columns=columns)
