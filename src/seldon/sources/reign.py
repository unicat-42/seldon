"""REIGN: лидеры, выборы, перевороты помесячно, 1950–2021. Скачивание: `download reign`."""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW
from seldon.core.entities import ccode_to_gw

REIGN_FILE = DATA_RAW / "reign" / "REIGN_2021_8.csv"


def load_reign() -> pd.DataFrame:
    """Лидер-месяцы REIGN в кодах Gleditsch–Ward, по порядку (страна, год, месяц)."""
    r = pd.read_csv(REIGN_FILE)
    r = r[r.year.notna()].copy()
    r["year"] = r.year.astype(int)
    r["month"] = r.month.astype(int)
    r["gw"] = ccode_to_gw(r.ccode)
    return r.sort_values(["gw", "year", "month"])
