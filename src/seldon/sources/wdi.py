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


def load_external_buffers() -> pd.DataFrame:
    """Запас прочности по внешним платежам (чекпоинт 3.3): iso3, year,
    `std_res`    краткосрочный внешний долг в % резервов (`DT.DOD.DSTC.IR.ZS`, статистика
                 внешнего долга ВБ — только развивающиеся страны; у развитых пропуск);
    `res_months` резервы в месяцах импорта (`FI.RES.TOTL.MO`)."""
    a = read_wdi("short_debt_reserves")[["iso3", "year", "value"]].rename(columns={"value": "std_res"})
    b = read_wdi("reserves_months")[["iso3", "year", "value"]].rename(columns={"value": "res_months"})
    out = a.merge(b, on=["iso3", "year"], how="outer")
    return out[out.iso3.notna() & (out.iso3.str.len() == 3)].sort_values(["iso3", "year"]).reset_index(drop=True)
