"""Всемирный банк (World Development Indicators): по файлу на показатель, колонки
iso3, year, value. Скачивание: `download wdi`."""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW
from seldon.core.entities import GW_TO_ISO3

WDI_DIR = DATA_RAW / "wdi"


def read_wdi(name: str) -> pd.DataFrame:
    """Показатель ВБ как в файле: iso3, year, value."""
    return pd.read_csv(WDI_DIR / f"{name}.csv")


def load_wdi(name: str) -> pd.DataFrame:
    """Показатель ВБ в кодах Gleditsch–Ward: gw, year, <name>."""
    w = read_wdi(name)
    iso_to_gw = {iso: gw for gw, iso in GW_TO_ISO3.items() if iso}
    iso_to_gw["SRB"] = 340  # Югославия (345) до 2006 возьмёт SRB ниже отдельно
    w["gw"] = w["iso3"].map(iso_to_gw)
    w = w.dropna(subset=["gw"])
    yug = w[w.iso3 == "SRB"].assign(gw=345)
    w = pd.concat([w, yug])
    w["gw"] = w["gw"].astype(int)
    return w[["gw", "year", "value"]].rename(columns={"value": name})


def load_annual_cpi() -> pd.DataFrame:
    """Годовой ИПЦ Всемирного банка: iso3, year, value."""
    return read_wdi("cpi_annual")[["iso3", "year", "value"]]
