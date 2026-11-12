"""Mass Mobilization: протесты по событиям, 1990–2020. Скачивание: `download massmob`."""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW
from seldon.core.entities import ccode_to_gw

MM_FILE = DATA_RAW / "massmob" / "mmALL_073120_csv.tab"


def load_massmob() -> pd.DataFrame:
    """Протесты (строка — событие) в кодах Gleditsch–Ward."""
    m = pd.read_csv(MM_FILE, sep=",", low_memory=False)
    m = m[m.protest == 1].copy()
    m["gw"] = ccode_to_gw(m.ccode)
    m["year"] = m.year.astype(int)
    return m
