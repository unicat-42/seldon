"""Годовой валютный блок, ЦЕЛЕВЫЕ: девальвация декабрь к декабрю, хроники кризисов, метка.

    deval25  — декабрьский курс к доллару вырос ≥ 25 % за год после ≥ 2 спокойных лет;
    gmd_cc   — валютный кризис по хронике GMD (обрезана 2016 годом);
    lv_cc    — валютный кризис по Laeven–Valencia 2018 (1970–2017).
Метка `y(t)` = 1, если кризис в t+1 или t+2; `evaluable` — строка, где такой кризис может
быть засчитан (инвариант 3). Подробности — в docstring `econ_long.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FIRST_YEAR, LAST_YEAR = 1957, 2025  # декабрьский курс МВФ: 1957-12 … 2025-12
H = 2
POST_CRISIS_DROP = 2
DEVAL = 0.25
HYPER = 100.0  # инфляция, %
MIN_POS_TRAIN = 30
# последний год, за который хроника кризисов реально размечена (дальше — нули вместо пропусков)
TARGET_LAST_YEAR = {"gmd_cc": 2016, "lv_cc": 2017}


def add_events(p: pd.DataFrame) -> pd.DataFrame:
    """События по страна-годам: `deval` (декабрь к декабрю), `big` (≥ 25 %), целевые
    `deval25` / `gmd_cc` (`lv_cc` уже в панели), `hyper_prev`. Только данные ≤ t."""
    p = p.copy()
    gb = p.groupby("iso3")
    # девальвация: декабрь к декабрю; у общей валюты своего курса нет
    p["deval"] = (p.fx_dec / gb.fx_dec.shift(1) - 1).where(p.own_currency)
    p["big"] = (p.deval >= DEVAL).astype(int)
    prev = gb.big.shift(1).fillna(0) + gb.big.shift(2).fillna(0)
    p["deval25"] = ((p.big == 1) & (prev == 0)).astype(float)
    p.loc[p.deval.isna(), "deval25"] = np.nan
    p["gmd_cc"] = p.CurrencyCrisis
    for t, last in TARGET_LAST_YEAR.items():
        p.loc[p.year > last, t] = np.nan
    gb = p.groupby("iso3")
    p["hyper_prev"] = gb.infl.shift(1) > HYPER
    return p


def add_target(p: pd.DataFrame, target: str) -> pd.DataFrame:
    p = p.copy()
    gb = p.groupby("iso3")[target]
    nxt, nxt2 = gb.shift(-1), gb.shift(-2)
    # метка известна, если кризис уже случился в t+1 или известны оба года
    y = np.where(nxt == 1, 1.0, np.where(nxt.isna() | nxt2.isna(), np.nan, (nxt2 == 1).astype(float)))
    p["y"] = y
    drop = p[target].fillna(0).astype(bool)
    for j in range(1, POST_CRISIS_DROP + 1):
        drop |= gb.shift(j).fillna(0).astype(bool)
    if target == "deval25":
        # большая девальвация идёт в t или t−1 → в t+1 «начаться» она не может: y был бы занулён правилом
        drop |= (p.big == 1) | (p.groupby("iso3").big.shift(1).fillna(0) == 1)
    # своя валюта нужна и в t, и в t+1: если с t+1 страна в валютном союзе (Эстония-2011,
    # Болгария-2026), своего риска девальвации в окне прогноза уже нет
    p["usable"] = (p.deval.notna() & p.own_currency & p.own_currency_next
                   & ~p.hyper_prev.fillna(False) & (p.year >= FIRST_YEAR))
    p["evaluable"] = ~drop & p.y.notna() & p.usable
    return p
