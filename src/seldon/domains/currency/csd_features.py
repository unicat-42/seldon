"""Признаки критического замедления (CSD) для месячного валютного контура и компактные
живые наборы `live_fx` (чекпоинты 4.3b и 4.3c).

Гипотеза (Scheffer et al.):
  - при плавающем курсе предвестники критического замедления (рост дисперсии и AR(1))
    появляются в месячных изменениях самого курса (`fx_ret1`);
  - при жёсткой привязке колебания курса задавлены и «перетекают» в резервы (`res_ret1`):
    сигнал проявляется как `csd_peg_res = peg_share12 * max(res_var_surge12, 0)` и
    `csd_peg_res_ar1 = peg_share12 * max(res_ar1_12, 0)`.

Все признаки вычисляются причинно на строке месяца `m` только по данным ≤ `m`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon.core.csd import ar1_delta, by_entity, rolling_ar1, var_surge

CSD_RAW = [
    "fx_var_surge12",
    "fx_ar1_12",
    "fx_ar1_d12",
    "res_var_surge12",
    "res_ar1_12",
    "res_ar1_d12",
    "csd_fx_missing",
    "csd_res_missing",
]
CSD_REGIME = [
    *CSD_RAW,
    "csd_peg_res",
    "csd_peg_res_ar1",
    "csd_float_fx",
    "csd_float_fx_ar1",
]
# 100 % живые на текущем крае признаки: курс/резервы МВФ + REER Bruegel + война UCDP.
# Без замороженных архивов IRR, без gg_missing (ВБ IDS) и без торговли IMTS.
LIVE_FX_ADD = [
    "peg_share12",
    "reer_gap",
    "reer_peg",
    "reer_missing",
    "peg_resfall",
    "peg_resfall_missing",
]
LIVE_FX_CTOT_ADD = [
    *LIVE_FX_ADD,
    "ctot_d12",
    "ctot_missing",
    "ctot_loss12",
    "ctot_loss24",
    "peg_loss12",
    "peg_loss24",
    "peg_vol",
    "brk_missing",
]


def _step_return(df: pd.DataFrame, col: str, lo: float, hi: float) -> pd.Series:
    """Месячная доходность `col` только на соседних календарных месяцах (без перескока через дыры)."""
    ord_m = pd.Series(pd.PeriodIndex(df.month, freq="M").asi8, index=df.index)
    is_next = ord_m.groupby(df.iso3).diff() == 1
    prev = df.groupby("iso3")[col].shift(1)
    ret = (df[col] / prev - 1.0).where(is_next & (prev > 0) & (df[col] > 0))
    return ret.clip(lower=lo, upper=hi)


def build_fx_csd(panel: pd.DataFrame) -> pd.DataFrame:
    """Добавить признаки критического замедления и `peg_resfall_missing` к панели `econ_contagion`."""
    d = panel.sort_values(["iso3", "month"]).copy()
    d["fx_ret1"] = _step_return(d, "fx", -0.5, 2.0)
    d["res_ret1"] = _step_return(d, "reserves", -0.8, 2.0)

    d["fx_var_surge12"] = by_entity(d, "iso3", "fx_ret1", var_surge, short=12, long=48, min_short=6, min_long=24, eps=1e-6)
    d["fx_ar1_12"] = by_entity(d, "iso3", "fx_ret1", rolling_ar1, window=12, min_periods=8)
    d["fx_ar1_d12"] = by_entity(d, "iso3", "fx_ret1", ar1_delta, short=12, long=48, min_short=8, min_long=24)

    d["res_var_surge12"] = by_entity(d, "iso3", "res_ret1", var_surge, short=12, long=48, min_short=6, min_long=24, eps=1e-4)
    d["res_ar1_12"] = by_entity(d, "iso3", "res_ret1", rolling_ar1, window=12, min_periods=8)
    d["res_ar1_d12"] = by_entity(d, "iso3", "res_ret1", ar1_delta, short=12, long=48, min_short=8, min_long=24)

    d["csd_fx_missing"] = d.fx_var_surge12.isna().astype("int8")
    d["csd_res_missing"] = d.res_var_surge12.isna().astype("int8")
    d["peg_resfall_missing"] = d.reserves_g12.isna().astype("int8")

    num_cols = [c for c in CSD_RAW if not c.endswith("_missing")]
    d[num_cols] = d[num_cols].fillna(0.0)

    peg = d.peg_share12.fillna(0.0)
    d["csd_peg_res"] = peg * d.res_var_surge12.clip(lower=0.0)
    d["csd_peg_res_ar1"] = peg * d.res_ar1_12.clip(lower=0.0)
    d["csd_float_fx"] = (1.0 - peg) * d.fx_var_surge12.clip(lower=0.0)
    d["csd_float_fx_ar1"] = (1.0 - peg) * d.fx_ar1_12.clip(lower=0.0)

    d = d.drop(columns=["fx_ret1", "res_ret1"])
    return d.loc[panel.index].copy()
