"""Polity5 (режимы, до 2018 года). Скачивание: `download polity`."""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW
from seldon.core.entities import POLITY_TO_GW

POLITY_FILE = DATA_RAW / "polity" / "p5v2018.xls"


def load_polity() -> pd.DataFrame:
    """Страна-годы Polity5 в кодах Gleditsch–Ward: ccode, year, exrec, parcomp, polity2, gw."""
    p = pd.read_excel(POLITY_FILE, usecols=["ccode", "year", "exrec", "parcomp", "polity2"])
    p["gw"] = p["ccode"].replace(POLITY_TO_GW)
    return p
