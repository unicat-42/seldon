"""Correlates of War, диадные милитаризованные межгосударственные споры (Dyadic MID 4.02).
Скачивание: `download mid`.

Спор (MID) — эпизод, когда одно государство угрожает другому силой, демонстрирует или
применяет её. Строка файла — пара государств × год спора. Берём:
    hihost    высший уровень враждебности в паре: 2 — угроза, 3 — демонстрация силы,
              4 — применение силы, 5 — война;
    revtypea / revtypeb   чего добивалась сторона: 1 — территория, 2 — политика, 3 — режим.

Хроника кончается 2014 годом (`MID_LAST_YEAR`): после него в файле не «нули», а ничего
(инвариант 6). Выходит с опозданием в годы — см. `cow_mid` в `core/temporal.py`.
"""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW
from seldon.core.entities import cow_year_to_gw

MID_FILE = DATA_RAW / "mid" / "dyadic_mid_4.02.csv"
MID_LAST_YEAR = 2014
FORCE = 4  # hihost ≥ 4 — применение силы или война
TERRITORY = 1  # revtype == 1 — спор о территории


def load_mid(year_from: int = 1946) -> pd.DataFrame:
    """Пара × год спора в кодах GW (обе записи направления, как в файле): a, b, year,
    force (1 — сила применялась), terr (1 — хотя бы одна сторона добивалась территории)."""
    m = pd.read_csv(MID_FILE, usecols=["statea", "stateb", "year", "hihost", "revtypea", "revtypeb"])
    m = m[m.year >= year_from].copy()
    m["a"], m["b"] = cow_year_to_gw(m.statea, m.year), cow_year_to_gw(m.stateb, m.year)
    m["force"] = (m.hihost >= FORCE).astype(int)
    m["terr"] = ((m.revtypea == TERRITORY) | (m.revtypeb == TERRITORY)).astype(int)
    return m.groupby(["a", "b", "year"], as_index=False)[["force", "terr"]].max()
