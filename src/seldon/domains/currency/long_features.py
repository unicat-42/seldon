"""Годовой валютный блок, ПРИЗНАКИ (год t; курсовые — по декабрю t, остальные — GMD за t).

Строятся поверх событий из `long_targets.add_events` (`deval`, `big`). Список и смысл
признаков — в docstring `econ_long.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE = ["infl_log", "infl_d2", "deval_prev", "deval_5", "gdp_growth_2", "debt_d2", "ca_level",
        "money_growth_2", "world_deval", "peg_years_log"]
MISSING = ["infl_missing", "gdp_missing", "debt_missing", "ca_missing", "money_missing"]
FEATURES = [*BASE, *MISSING]


def add_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    gb = p.groupby("iso3")
    p["infl_log"] = np.log1p(p.infl.clip(lower=-50, upper=10_000) / 100.0)
    p["infl_d2"] = (p.infl - gb.infl.shift(2)).clip(-500, 500) / 100.0
    p["deval_prev"] = p.deval.clip(-0.9, 5)
    p["deval_5"] = gb.big.transform(lambda s: s.rolling(5, min_periods=1).sum())
    p["gdp_growth_2"] = ((p.rGDP / gb.rGDP.shift(2)) ** 0.5 - 1).clip(-0.5, 0.5)
    p["debt_d2"] = (gb.govdebt_GDP.diff(2) / 100.0).clip(-2, 2)
    p["ca_level"] = (p.CA_GDP / 100.0).clip(-0.5, 0.5)
    money = p.M2.fillna(p.M3).fillna(p.M0)
    p["money_growth_2"] = (((money / money.groupby(p.iso3).shift(2)) ** 0.5 - 1) - (p.nGDP / gb.nGDP.shift(2)) ** 0.5 + 1).clip(-1, 2)
    world = p[p.deval.notna()].groupby("year").big.mean()
    p["world_deval"] = p.year.map(world)
    stable = (p.deval.abs() <= 0.02).astype(int)
    run = stable.groupby(p.iso3).transform(lambda s: s.groupby((s == 0).cumsum()).cumsum())
    p["peg_years_log"] = np.log1p(run)
    for src, flag in [("infl_log", "infl_missing"), ("gdp_growth_2", "gdp_missing"), ("debt_d2", "debt_missing"),
                      ("ca_level", "ca_missing"), ("money_growth_2", "money_missing")]:
        p[flag] = p[src].isna().astype(int)
        p[src] = p[src].fillna(0.0)
    for c in ("infl_d2", "deval_prev", "world_deval"):
        p[c] = p[c].fillna(0.0)
    return p
