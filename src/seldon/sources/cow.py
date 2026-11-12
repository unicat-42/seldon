"""Correlates of War, Direct Contiguity 3.2: кто с кем граничит. Скачивание: `download cow`."""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW
from seldon.core.entities import COW_TO_GW, SERBIA_COW, SERBIA_GW
from seldon.sources.ucdp import YEAR_FROM, YEAR_TO

CONTIGUITY_FILE = DATA_RAW / "cow" / "DirectContiguity320" / "contdird.csv"
CONTIGUITY_MAX_TYPE = 2  # 1 — сухопутная граница, 2 — через море до 12 миль
CONTIGUITY_END_YEAR = 2016  # последний год в данных COW; дальше считаем границы неизменными


def load_neighbors(year_from: int = YEAR_FROM, year_to: int = YEAR_TO) -> set[tuple[int, int, int]]:
    """Множество троек (страна, сосед, год) в кодах GW: кто с кем граничит в каждом году."""
    c = pd.read_csv(CONTIGUITY_FILE, usecols=["state1no", "state2no", "year", "conttype"])
    c = c[(c.conttype <= CONTIGUITY_MAX_TYPE) & (c.year >= year_from)]
    serbia = c[(c.state1no == SERBIA_COW) | (c.state2no == SERBIA_COW)].replace(
        {"state1no": {SERBIA_COW: SERBIA_GW}, "state2no": {SERBIA_COW: SERBIA_GW}})
    c[["state1no", "state2no"]] = c[["state1no", "state2no"]].replace(COW_TO_GW)
    c = pd.concat([c, serbia])  # Сербия: и как 345 (Югославия), и как 340
    tail = c[c.year == CONTIGUITY_END_YEAR]
    c = pd.concat([c, *(tail.assign(year=y) for y in range(CONTIGUITY_END_YEAR + 1, year_to + 1))])
    return set(zip(c.state1no, c.state2no, c.year))
