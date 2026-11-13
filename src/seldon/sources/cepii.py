"""CEPII GeoDist (Mayer & Zignago, 2011): матрица языковой, колониальной и исторической близости пар стран.

Скачивание: `uv run python -m seldon download cepii`.
Таблица статическая (выпущена в 2011 году, без временной динамики); существование государств по годам
и месяцам ограничивается сеткой Gleditsch–Ward снаружи (`sources/gw.py`). Коды стран переводятся
из ISO3 в Gleditsch–Ward (`core/entities.py`), включая исторические коды `ROM` (ROU), `ZAR` (COD),
`TMP` (TLS) и `YUG` (Сербия 340, Черногория 341, Югославия 345). Южный Судан (626, независим с 2011)
в файле CEPII отсутствует — его этноязыковые связи покрываются через `EPR TEK`.
"""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW
from seldon.core.entities import CEPII_ISO3_ALIASES, ISO3_TO_GW

CEPII_FILE = DATA_RAW / "cepii" / "dist_cepii.xls"
CEPII_COLS = ["comlang_off", "comlang_ethno", "colony", "smctry"]


def _iso_to_gws(iso: str) -> tuple[int, ...]:
    if iso in CEPII_ISO3_ALIASES:
        return CEPII_ISO3_ALIASES[iso]
    gw = ISO3_TO_GW.get(iso)
    return (gw,) if gw is not None else ()


def parse_cepii(raw: pd.DataFrame) -> pd.DataFrame:
    """Преобразовать сырую таблицу CEPII в направленные пары (a, b) в кодах GW без дублей и диагонали."""
    df = raw[["iso_o", "iso_d", *CEPII_COLS]].copy()
    codes = sorted(set(df.iso_o.dropna()).union(set(df.iso_d.dropna())))
    iso_map = {iso: _iso_to_gws(iso) for iso in codes}
    df["a_list"] = df.iso_o.map(iso_map)
    df["b_list"] = df.iso_d.map(iso_map)
    df = df[df.a_list.map(bool) & df.b_list.map(bool)]
    df = df.explode("a_list").explode("b_list").rename(columns={"a_list": "a", "b_list": "b"})
    df = df[df.a != df.b].copy()
    for c in CEPII_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")

    # Внутренние пары бывшей Югославии (Сербия 340, Черногория 341, Югославия до 2006 — 345)
    yug = CEPII_ISO3_ALIASES["YUG"]
    if set(yug) & set(df.a):
        inner = pd.DataFrame([
            {"a": x, "b": y, "comlang_off": 1, "comlang_ethno": 1, "colony": 0, "smctry": 1}
            for x in yug for y in yug if x != y
        ])
        df = pd.concat([df[["a", "b", *CEPII_COLS]], inner], ignore_index=True)

    out = df.groupby(["a", "b"], as_index=False)[CEPII_COLS].max()
    out["a"] = out["a"].astype("int16")
    out["b"] = out["b"].astype("int16")
    for c in CEPII_COLS:
        out[c] = out[c].astype("int8")
    return out.sort_values(["a", "b"]).reset_index(drop=True)


def load_cepii() -> pd.DataFrame:
    """Прочитать `dist_cepii.xls` из `data/raw/cepii/` и вернуть пары GW: a, b, comlang_off, comlang_ethno, colony, smctry."""
    raw = pd.read_excel(CEPII_FILE, usecols=["iso_o", "iso_d", *CEPII_COLS])
    return parse_cepii(raw)
