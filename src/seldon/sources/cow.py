"""Correlates of War, Direct Contiguity 3.2: кто с кем граничит. Скачивание: `download cow`."""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW
from seldon.core.entities import COW_TO_GW, SERBIA_COW, SERBIA_GW
from seldon.sources.ucdp import YEAR_FROM, YEAR_TO

CONTIGUITY_FILE = DATA_RAW / "cow" / "DirectContiguity320" / "contdird.csv"
CONTIGUITY_MAX_TYPE = 2  # 1 — сухопутная граница, 2 — через море до 12 миль
CONTIGUITY_END_YEAR = 2016  # последний год в данных COW; дальше считаем границы неизменными


CONTIGUITY_WIDE_TYPE = 5  # 3…5 — через море до 24 / 150 / 400 миль (вселенная диадной панели)


def load_contiguity(year_from: int = YEAR_FROM, year_to: int = YEAR_TO, max_type: int = CONTIGUITY_WIDE_TYPE) -> pd.DataFrame:
    """Направленные пары соседей по годам в кодах GW: a, b, year, conttype (1 — суша … 5 — море
    до 400 миль). После `CONTIGUITY_END_YEAR` география границ продлевается как в последнем
    году данных (границы меняются редко; новых государств после 2016 в списке GW нет)."""
    c = pd.read_csv(CONTIGUITY_FILE, usecols=["state1no", "state2no", "year", "conttype"])
    c = c[(c.conttype <= max_type) & (c.year >= year_from)]
    serbia = c[(c.state1no == SERBIA_COW) | (c.state2no == SERBIA_COW)].replace(
        {"state1no": {SERBIA_COW: SERBIA_GW}, "state2no": {SERBIA_COW: SERBIA_GW}})
    c[["state1no", "state2no"]] = c[["state1no", "state2no"]].replace(COW_TO_GW)
    c = pd.concat([c, serbia])  # Сербия: и как 345 (Югославия), и как 340
    tail = c[c.year == CONTIGUITY_END_YEAR]
    c = pd.concat([c, *(tail.assign(year=y) for y in range(CONTIGUITY_END_YEAR + 1, year_to + 1))])
    c = c.rename(columns={"state1no": "a", "state2no": "b"})
    return c.groupby(["a", "b", "year"], as_index=False).conttype.min()


def load_neighbors(year_from: int = YEAR_FROM, year_to: int = YEAR_TO) -> set[tuple[int, int, int]]:
    """Множество троек (страна, сосед, год) в кодах GW: кто с кем граничит в каждом году
    (суша или море до 12 миль)."""
    c = load_contiguity(year_from, year_to, CONTIGUITY_MAX_TYPE)
    return set(zip(c.a, c.b, c.year))
