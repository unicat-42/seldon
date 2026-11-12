"""Институты по V-Dem — замена замороженного в 2018 году Polity5 (чекпоинт 3.1).

Запуск:
    uv run python -m seldon features_vdem

Читает features_pitf_country_year.parquet, добавляет колонки ниже и пишет
features_vdem_country_year.parquet. Лаг публикации: релиз V-Dem с данными за год T выходит
в марте T+1, поэтому при прогнозе «на 1 января T+1» на строке T стоят значения за T − 1
(`core/temporal.py`), а двухлетний сдвиг — разница между T − 1 и T − 3.

    vd_polyarchy       электоральная демократия, 0…1 (v2x_polyarchy)
    vd_anoc            4·p·(1 − p) от неё же: 1 — ровно посередине между автократией и
                       демократией, 0 — на краях. Аналог «анократии» PITF: рискованнее всего
                       переходные режимы, а не диктатуры и не демократии
    vd_libdem          либеральная демократия, 0…1 (v2x_libdem)
    vd_polar           политическая поляризация (v2cacamps), больше — глубже раскол
    vd_faction         vd_anoc × поляризация выше нуля — аналог «частичной демократии с
                       фракционностью», главного признака PITF
    vd_repress         1 − свобода от политических убийств и пыток (v2x_clphy)
    vd_*_d2            сдвиг за два года: polyarchy, libdem, polar, repress
    vd_missing         1, если V-Dem для страны-года нет (микрогосударства); уровни тогда —
                       медиана года, сдвиги — 0
"""

from __future__ import annotations

import pandas as pd

from seldon import DATA_PROCESSED
from seldon.core.temporal import publication_lag
from seldon.sources.vdem import load_vdem

VDEM_LAG = publication_lag("vdem_annual", "jan1")  # 1 год
CHANGE_YEARS = 2
LEVELS = ["vd_polyarchy", "vd_anoc", "vd_libdem", "vd_polar", "vd_faction", "vd_repress"]
CHANGES = ["vd_polyarchy_d2", "vd_libdem_d2", "vd_polar_d2", "vd_repress_d2"]
VDEM_FEATURES = [*LEVELS, *CHANGES, "vd_missing"]


def vdem_table(v: pd.DataFrame) -> pd.DataFrame:
    """Признаки по собственным годам V-Dem (без лага). Чистая функция: строка года Y
    зависит только от значений V-Dem за Y и Y − 2."""
    t = pd.DataFrame({"gw": v.gw, "year": v.year, "vd_polyarchy": v.v2x_polyarchy, "vd_libdem": v.v2x_libdem,
                      "vd_polar": v.v2cacamps, "vd_repress": 1 - v.v2x_clphy})
    t["vd_anoc"] = 4 * t.vd_polyarchy * (1 - t.vd_polyarchy)
    t["vd_faction"] = t.vd_anoc * t.vd_polar.clip(lower=0)
    base = ["vd_polyarchy", "vd_libdem", "vd_polar", "vd_repress"]
    prev = t[["gw", "year", *base]].assign(year=t.year + CHANGE_YEARS)  # сдвиг по календарю, не по строкам
    t = t.merge(prev, on=["gw", "year"], how="left", suffixes=("", "_prev"))
    for c in base:
        t[f"{c}_d2"] = t[c] - t.pop(f"{c}_prev")
    return t


def add_vdem_features(f: pd.DataFrame, v: pd.DataFrame, lag: int = VDEM_LAG) -> pd.DataFrame:
    """Признаки V-Dem поверх панели. На строке T — значения V-Dem за T − `lag`."""
    t = vdem_table(v)
    f = f.merge(t.assign(year=t.year + lag), on=["gw", "year"], how="left")
    f["vd_missing"] = f.vd_polyarchy.isna().astype("int8")
    for c in LEVELS:
        f[c] = f[c].fillna(f.groupby("year")[c].transform("median")).fillna(0)
    f[CHANGES] = f[CHANGES].fillna(0)
    return f


def build() -> pd.DataFrame:
    return add_vdem_features(pd.read_parquet(DATA_PROCESSED / "features_pitf_country_year.parquet"), load_vdem())


def main(argv: list[str]) -> int:
    f = build()
    f.to_parquet(DATA_PROCESSED / "features_vdem_country_year.parquet", index=False)
    print(f"признаки V-Dem добавлены: {len(f)} строк → data/processed/features_vdem_country_year.parquet")
    print(f"V-Dem нет: {f.vd_missing.mean():.1%} строк; в последнем году ({f.year.max()}): {f[f.year == f.year.max()].vd_missing.mean():.1%}")
    print(f[VDEM_FEATURES].describe().T[["mean", "min", "max"]].round(3).to_string())
    return 0
