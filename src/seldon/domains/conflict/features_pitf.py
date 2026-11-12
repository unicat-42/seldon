"""Признаки в духе PITF: тип режима (Polity5), детская смертность и население (Всемирный банк).

Запуск:
    uv run python -m seldon features_pitf

Читает features_country_year.parquet (история, см. features.py), добавляет колонки ниже
и пишет features_pitf_country_year.parquet.

Режим — по правилу Goldstone et al. (2010) из компонентов Polity `exrec` (как выбирается
исполнительная власть, 1–8) и `parcomp` (конкуренция участия, 0–5):
    full_autoc         exrec ≤ 5 и parcomp ∈ {1, 2}                         (база, без дамми)
    part_autoc         exrec ≤ 5 и parcomp ∈ {0, 3, 4, 5}, или exrec ≥ 6 и parcomp ∈ {1, 2}
    part_dem_fact      exrec ≥ 6 и parcomp == 3    — «частичная демократия с фракционностью»
    part_dem_nofact    exrec ≥ 6 и parcomp ∈ {0, 4, 5}, кроме полной демократии
    full_dem           exrec == 8 и parcomp == 5
    transition         коды −66/−77/−88 (оккупация, анархия, переход)
Проверено на replication data PITF: с лагом 2 года совпадает с их `sftptv2a` в 91% строк.

    polity2            сводный индекс Polity от −10 (автократия) до +10 (демократия)
    regime_missing     1, если Polity нет для страны-года (микрогосударства, Косово)
    regime_extended    1, если значения продлены из последнего года Polity (2018–2020 → 2024)
    log_imr_rel        ln(детская смертность / медиана по миру в том же году), как у PITF;
                       смертность и население — за год T−1 (лаг публикации ВБ, см. `core/temporal.py`)
    imr_missing        1, если смертности нет (тогда log_imr_rel = 0)
    log_pop            ln(население)
    pop_missing        1, если населения нет (тогда log_pop = медиана года)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED
from seldon.core.temporal import publication_lag
from seldon.sources.polity import load_polity
from seldon.sources.wdi import load_wdi

REGIMES = ["part_autoc", "part_dem_fact", "part_dem_nofact", "full_dem", "transition"]


def regime_type(exrec: float, parcomp: float) -> str:
    if pd.isna(exrec) or pd.isna(parcomp):
        return "missing"
    if exrec < 0 or parcomp < 0:
        return "transition"
    if exrec == 8 and parcomp == 5:
        return "full_dem"
    if exrec >= 6:
        return {3: "part_dem_fact", 1: "part_autoc", 2: "part_autoc"}.get(int(parcomp), "part_dem_nofact")
    return "full_autoc" if parcomp in (1, 2) else "part_autoc"


def regime_table(years: range) -> pd.DataFrame:
    """Режим по страна-годам: тип по правилу Goldstone, `polity2`; последний год Polity
    продлён до конца панели с флагом `regime_extended`."""
    p = load_polity()
    p = p[p.year >= years.start - 1].sort_values(["gw", "year"]).drop_duplicates(["gw", "year"], keep="last")
    p["regime"] = [regime_type(e, c) for e, c in zip(p.exrec, p.parcomp)]
    p["regime_extended"] = 0
    # продлеваем последний год каждой страны до конца панели
    last = p.groupby("gw").tail(1)
    ext = pd.concat(
        [last[last.year < y].assign(year=y, regime_extended=1) for y in years], ignore_index=True
    )
    return pd.concat([p, ext], ignore_index=True)[["gw", "year", "regime", "polity2", "regime_extended"]]


MACRO_LAG = publication_lag("wdi_annual", "jan1")  # 1 год: прогноз войны на T+1 делается «на 1 января T+1»


def add_pitf_features(f: pd.DataFrame, pol: pd.DataFrame, imr: pd.DataFrame, pop: pd.DataFrame,
                      macro_lag: int = MACRO_LAG) -> pd.DataFrame:
    """Признаки PITF поверх истории. Чистая функция. Лаг публикации (чекпоинт 2.2): детская
    смертность и население ВБ за год T к 1 января T+1 ещё не вышли, поэтому на строке T
    стоят значения за T − `macro_lag`. `imr`, `pop` — таблицы gw, year, <показатель>."""
    imr = imr.assign(year=imr.year + macro_lag)
    pop = pop.assign(year=pop.year + macro_lag)
    f = f.merge(pol, on=["gw", "year"], how="left")
    f["regime_missing"] = (f["regime"].isna() | (f["regime"] == "missing")).astype("int8")
    f["regime"] = f["regime"].fillna("missing")
    for r in REGIMES:
        f[r] = (f["regime"] == r).astype("int8")
    f["polity2"] = f["polity2"].fillna(0)
    f["regime_extended"] = f["regime_extended"].fillna(0).astype("int8")

    f = f.merge(imr, on=["gw", "year"], how="left")
    f["infant_mortality"] = f.groupby("gw")["infant_mortality"].ffill()
    median = f.groupby("year")["infant_mortality"].transform("median")
    f["imr_missing"] = f["infant_mortality"].isna().astype("int8")
    f["log_imr_rel"] = np.log(f["infant_mortality"] / median).fillna(0)

    f = f.merge(pop, on=["gw", "year"], how="left")
    f["population"] = f.groupby("gw")["population"].ffill()
    f["pop_missing"] = f["population"].isna().astype("int8")
    f["log_pop"] = np.log(f["population"])
    f["log_pop"] = f["log_pop"].fillna(f.groupby("year")["log_pop"].transform("median"))
    return f.drop(columns=["regime", "infant_mortality", "population"])


def build() -> pd.DataFrame:
    f = pd.read_parquet(DATA_PROCESSED / "features_country_year.parquet")
    years = range(int(f.year.min()), int(f.year.max()) + 1)
    return add_pitf_features(f, regime_table(years), load_wdi("infant_mortality"), load_wdi("population"))


PITF_FEATURES = [*REGIMES, "polity2", "regime_missing", "regime_extended", "log_imr_rel", "imr_missing", "log_pop", "pop_missing"]


def main(argv: list[str]) -> int:
    f = build()
    f.to_parquet(DATA_PROCESSED / "features_pitf_country_year.parquet", index=False)
    print(f"признаки PITF добавлены: {len(f)} строк → data/processed/features_pitf_country_year.parquet")
    print(f"режим нет: {f.regime_missing.mean():.1%} строк, продлён: {f.regime_extended.mean():.1%}; "
          f"смертности нет: {f.imr_missing.mean():.1%}; населения нет: {f.pop_missing.mean():.1%}")
    print(f[REGIMES].mean().round(3).to_string())
    return 0

