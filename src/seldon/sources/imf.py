"""МВФ, месячные ряды (SDMX 3.0): курс, резервы, ИПЦ, ставки, деньги, экспорт.
Скачивание: `download imf`."""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW

IMF = DATA_RAW / "imf"


def read_imf(name: str) -> pd.DataFrame:
    """Один поток МВФ в длинном формате: iso3, month (`pd.Period`), value и коды ряда."""
    d = pd.read_csv(IMF / f"{name}.csv", usecols=lambda c: c in {
        "COUNTRY", "INDICATOR", "UNIT", "TYPE_OF_TRANSFORMATION", "TIME_PERIOD", "OBS_VALUE"})
    d = d[d.OBS_VALUE.notna()].copy()
    d["month"] = pd.PeriodIndex(d.TIME_PERIOD.str.replace("-M", "-"), freq="M")
    return d.rename(columns={"COUNTRY": "iso3", "OBS_VALUE": "value"})


def load_dec_fx() -> pd.DataFrame:
    """Курс к доллару на конец декабря (МВФ, EOP_RT): iso3, year, fx_dec."""
    er = read_imf("exchange_rates")
    er = er[(er.TYPE_OF_TRANSFORMATION == "EOP_RT") & (er.month.dt.month == 12) & (er.value > 0)]
    er = er.assign(year=er.month.dt.year).groupby(["iso3", "year"], as_index=False).value.first()
    return er.rename(columns={"value": "fx_dec"})
