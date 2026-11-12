"""UCDP: события GED (Georeferenced Event Dataset) и список сторон конфликтов ACD (UCDP/PRIO
Armed Conflict Dataset). Скачивание: `download ucdp`."""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW

UCDP_VERSION = "26.1"  # релиз 26.1 покрывает события по 2025 год включительно
_V = UCDP_VERSION.replace(".", "_")
YEAR_FROM, YEAR_TO = 1989, 2025  # GED покрывает 1989–2025
GED_FILE = DATA_RAW / "ucdp" / f"GEDEvent_v{_V}.csv"
ACD_FILE = DATA_RAW / "ucdp" / f"UcdpPrioConflict_v{_V}.csv"


def load_ged() -> pd.DataFrame:
    """События GED: страна (код GW), год, месяц, тип насилия, погибшие (`best`), конфликт."""
    cols = ["country", "country_id", "year", "date_start", "type_of_violence", "best", "conflict_new_id"]
    g = pd.read_csv(GED_FILE, usecols=cols)
    g["date_start"] = pd.to_datetime(g["date_start"])
    g["month"] = g["date_start"].dt.month
    g["deaths_state"] = g["best"].where(g["type_of_violence"] == 1, 0)
    return g.rename(columns={"country_id": "gw"})


def load_acd(year_from: int = YEAR_FROM, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Конфликт-годы ACD: коды сторон (основные `gwno_a`/`gwno_b`, помогающие войсками
    `gwno_a_2nd`/`gwno_b_2nd`) и место `gwno_loc` — строками, как в файле. `extra` —
    дополнительные колонки (например, `incompatibility`: 1 — территория, 2 — власть, 3 — обе)."""
    cols = ["conflict_id", "year", "gwno_a", "gwno_b", "gwno_a_2nd", "gwno_b_2nd", "gwno_loc", *extra]
    a = pd.read_csv(ACD_FILE, usecols=cols)
    return a[a.year >= year_from]
