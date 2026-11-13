"""МВФ, месячные ряды (SDMX 3.0): курс, резервы, ИПЦ, ставки, деньги, экспорт, сырьевые
условия торговли. Скачивание: `download imf`."""

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


# Сырьевые условия торговли (Commodity Terms of Trade, Gruss & Kebhaj 2019), поток IMF.RES/CTOT.
# Индекс «чистый экспорт сырья, взвешенный по доле в ВВП»: изменение его логарифма ≈ выигрыш
# или потеря дохода страны от мировых цен, в долях ВВП. Вариант весов — СКОЛЬЗЯЩИЕ (`R_RW_IX`,
# месячный, с 1980): вес товара в году t — средняя торговля трёх ПРЕДЫДУЩИХ лет. У варианта
# с фиксированными весами (`R_FW_IX`) вес — средняя торговля за десятилетия, включая будущие
# для прошлых лет: это скрытая утечка структуры экспорта, его не берём.
CTOT_INDICATOR = "CEMPI_CTOTXM_GDP"
CTOT_WEIGHTS = "R_RW_IX"


def load_ctot() -> pd.DataFrame:
    """Месячный индекс сырьевых условий торговли: iso3, month (`pd.Period`), ctot."""
    d = pd.read_csv(IMF / "ctot.csv", usecols=["COUNTRY", "INDICATOR", "WGT_TYPE", "TIME_PERIOD", "OBS_VALUE"])
    d = d[(d.INDICATOR == CTOT_INDICATOR) & (d.WGT_TYPE == CTOT_WEIGHTS) & (d.OBS_VALUE > 0)]
    month = pd.PeriodIndex(d.TIME_PERIOD.str.replace("-M", "-"), freq="M")
    return pd.DataFrame({"iso3": d.COUNTRY.to_numpy(), "month": month, "ctot": d.OBS_VALUE.to_numpy()}) \
        .sort_values(["iso3", "month"]).reset_index(drop=True)


def load_monthly_fx() -> pd.DataFrame:
    """Курс к доллару на конец месяца (МВФ, EOP_RT), вся история: iso3, month, fx."""
    er = read_imf("exchange_rates")
    er = er[(er.TYPE_OF_TRANSFORMATION == "EOP_RT") & (er.value > 0)]
    return er.groupby(["iso3", "month"], as_index=False).value.first().rename(columns={"value": "fx"})


EURO_AREA = "G163"   # код еврозоны в потоке ER
EURO_FROM = pd.Period("1999-01", freq="M")
DEM_PER_EUR = 1.95583


def load_eur_anchor(fx: pd.DataFrame | None = None) -> pd.DataFrame:
    """Второй якорь для привязок: евро за доллар по месяцам, до 1999 года — немецкая марка,
    пересчитанная в евро по курсу фиксации. Колонки: month, eur_usd."""
    fx = load_monthly_fx() if fx is None else fx
    dem = fx[(fx.iso3 == "DEU") & (fx.month < EURO_FROM)].assign(eur_usd=lambda d: d.fx / DEM_PER_EUR)
    eur = fx[(fx.iso3 == EURO_AREA) & (fx.month >= EURO_FROM)].assign(eur_usd=lambda d: d.fx)
    return pd.concat([dem, eur])[["month", "eur_usd"]].sort_values("month").reset_index(drop=True)


# Двусторонняя торговля товарами (IMTS — бывшая Direction of Trade Statistics), годовая,
# 1948–2025: экспорт FOB и импорт CIF страны-отчётчика по партнёрам, доллары. Постсоветские
# страны — с 1992–1994 годов. Агрегаты (коды `G…`: мир, еврозона, регионы) выбрасываются.
# Ревизии не покрыты: читается сегодняшняя редакция (у МВФ есть помесячные «винтажи» набора,
# но только с декабря 2025).
TRADE_EXPORTS, TRADE_IMPORTS = "XG_FOB_USD", "MG_CIF_USD"


def load_trade() -> pd.DataFrame:
    """Годовая торговля пар: iso3 (отчётчик), partner, year, exports, imports (доллары;
    пропуск — отчётчик потока не сообщал)."""
    d = pd.read_csv(IMF / "trade_partners.csv", usecols=["COUNTRY", "INDICATOR", "COUNTERPART_COUNTRY", "TIME_PERIOD", "OBS_VALUE"])
    iso = r"[A-Z]{3}"
    d = d[d.COUNTRY.str.fullmatch(iso, na=False) & d.COUNTERPART_COUNTRY.str.fullmatch(iso, na=False) & (d.OBS_VALUE > 0)]
    t = d.pivot_table(index=["COUNTRY", "COUNTERPART_COUNTRY", "TIME_PERIOD"], columns="INDICATOR", values="OBS_VALUE", aggfunc="first")
    t = t.reindex(columns=[TRADE_EXPORTS, TRADE_IMPORTS]).reset_index()
    t.columns = ["iso3", "partner", "year", "exports", "imports"]
    return t.astype({"year": int})
