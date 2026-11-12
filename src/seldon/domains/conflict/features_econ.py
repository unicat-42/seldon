"""Экономические признаки для модели войн: GMD (годовые) + давление на валюту из блока KR.

Запуск:
    uv run python -m seldon features_econ

Читает features_pitf_country_year.parquet (история + PITF), добавляет колонки ниже и пишет
features_econ_country_year.parquet. Признаки строки T используются для прогноза войны в T+1
(так устроен model_hist.py); годовые GMD и кредит ВБ берутся за T−1 — лаг публикации. GMD обрезан 2024 годом — дальше
в нём прогнозы МВФ, их не берём.

    gdp_growth        рост реального ВВП за год, доля (обрезан ±50 %)
    gdp_growth_3      средний рост за 3 года
    log_gdp_pc        ln(реальный ВВП на душу, долл.)
    infl_log          ln(1 + инфляция/100)
    deval             рост курса к доллару за год, доля (обрезан 0…5)
    credit_diff2      изменение кредит/ВВП за 2 года, доля (ВБ, как в econ_panel)
    emp_max           максимум индекса давления на валютный рынок за год, в σ (econ_kr)
    emp_crisis_now    EMP-кризис в году T
    emp_crisis_prev   EMP-кризис в году T−1
    econ_missing      1, если GMD нет для страны-года (тогда признаки GMD = 0)
    emp_missing       1, если месячных данных МВФ нет (тогда emp_* = 0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED
from seldon.core.entities import GW_TO_ISO3
from seldon.core.temporal import publication_lag
from seldon.sources.gmd import load_gmd
from seldon.sources.wdi import load_wdi

GMD_LAST_YEAR = 2024
MACRO_LAG = publication_lag("gmd_annual", "jan1")  # годовые GMD и ВБ за T к 1 января T+1 ещё не вышли
GMD_FEATURES = ["gdp_growth", "gdp_growth_3", "log_gdp_pc", "infl_log", "deval", "credit_diff2"]
EMP_FEATURES = ["emp_max", "emp_crisis_now", "emp_crisis_prev"]
ECON_FEATURES = [*GMD_FEATURES, *EMP_FEATURES, "econ_missing", "emp_missing"]


def iso_to_gw_table() -> pd.DataFrame:
    rows = [(gw, iso) for gw, iso in GW_TO_ISO3.items() if iso]  # 340 и 345 (Югославия) → SRB
    return pd.DataFrame(rows, columns=["gw", "iso3"])


def gmd_features() -> pd.DataFrame:
    g = load_gmd()
    g = g[(g.year >= 1985) & (g.year <= GMD_LAST_YEAR)].rename(columns={"ISO3": "iso3"})
    g["year"] = g.year.astype(int)
    g = g.sort_values(["iso3", "year"])
    gb = g.groupby("iso3")
    out = pd.DataFrame({"iso3": g.iso3, "year": g.year})
    out["gdp_growth"] = gb.rGDP.pct_change().clip(-0.5, 0.5)
    out["gdp_growth_3"] = (gb.rGDP.transform(lambda s: s / s.shift(3)) ** (1 / 3) - 1).clip(-0.5, 0.5)
    out["log_gdp_pc"] = np.log(g.rGDP_pc_USD.where(g.rGDP_pc_USD > 0))
    out["infl_log"] = np.log1p(g.infl.clip(lower=-50, upper=10_000) / 100.0)
    out["deval"] = gb.USDfx.pct_change().clip(0, 5)
    return out


def load_emp() -> pd.DataFrame:
    p = pd.read_parquet(DATA_PROCESSED / "econ_kr_panel.parquet", columns=["iso3", "month", "emp_z", "crisis"])
    p["year"] = p.month.str[:4].astype(int)
    a = p.groupby(["iso3", "year"]).agg(emp_max=("emp_z", "max"), emp_crisis_now=("crisis", "max")).reset_index()
    a["emp_max"] = a.emp_max.clip(-5, 20)
    a = a.sort_values(["iso3", "year"])
    a["emp_crisis_prev"] = a.groupby("iso3").emp_crisis_now.shift(1)
    return a


def build(macro_lag: int = MACRO_LAG) -> pd.DataFrame:
    """Лаг публикации (чекпоинт 2.2): признаки GMD и кредит ВБ на строке T — за год
    T − `macro_lag`; EMP собран из месячных данных года T и лага не требует."""
    f = pd.read_parquet(DATA_PROCESSED / "features_pitf_country_year.parquet")
    key = iso_to_gw_table()
    gmd = gmd_features().merge(key, on="iso3").drop(columns="iso3")
    gmd["year"] = gmd.year + macro_lag
    credit = load_wdi("credit_gdp").sort_values(["gw", "year"])
    credit["credit_diff2"] = credit.groupby("gw").credit_gdp.diff(2) / 100.0
    credit["year"] = credit.year + macro_lag
    emp = load_emp().merge(key, on="iso3").drop(columns="iso3")
    f = (f.merge(gmd, on=["gw", "year"], how="left")
          .merge(credit[["gw", "year", "credit_diff2"]], on=["gw", "year"], how="left")
          .merge(emp, on=["gw", "year"], how="left"))
    f["econ_missing"] = f.gdp_growth.isna().astype(int)
    f["emp_missing"] = f.emp_max.isna().astype(int)
    for c in [*GMD_FEATURES, "credit_diff2"]:
        f[c] = f[c].fillna(0.0)
    f["log_gdp_pc"] = f.log_gdp_pc.where(f.econ_missing == 0, f.groupby("year").log_gdp_pc.transform("median")).fillna(0)
    for c in EMP_FEATURES:
        f[c] = f[c].fillna(0.0)
    return f


def main(argv: list[str]) -> int:
    f = build()
    f.to_parquet(DATA_PROCESSED / "features_econ_country_year.parquet", index=False)
    print(f"{len(f)} строк, {f.gw.nunique()} стран, {f.year.min()}–{f.year.max()}")
    print("доля строк без GMD:", round(f.econ_missing.mean(), 3), " без EMP:", round(f.emp_missing.mean(), 3))
    print(f[ECON_FEATURES].describe().T.round(3).to_string())
    return 0

