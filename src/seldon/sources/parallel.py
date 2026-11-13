"""Параллельный (чёрный) валютный курс: премия к официальному курсу, по месяцам.
Скачивание: `download parallel` (архив Рейнхарт) и `download imf` (криптопремия).

Единого открытого ряда нет — два куска с дырой между ними (разведка 13.11.2026):

1. Архив Рейнхарт–Рогофф («The Modern History of Exchange Rate Arrangements», 2004):
   месячные официальный и параллельный курсы, 93 страны, январь 1946 — декабрь 1998
   (источник — Pick's / World Currency Yearbook). На сайте автора ссылка на файл мёртвая;
   берём снимок Архива интернета (`web.archive.org`) с контрольной суммой. Постсоветских
   стран в архиве нет. Хроника кончается декабрём 1998 (`REINHART_LAST_MONTH`, инвариант 6).
2. МВФ, «Crypto-based Parallel Exchange Rates» (поток `IMF.STA/WPCPER`; Graf von Luckner,
   Koepke, Sgherri, IMF WP 2024/133): премия курса, вычисленного из цен биткойна в местной
   валюте и в долларах, к официальному. Январь 2019 — октябрь 2025, 33 страны с рядом премии
   (из постсоветских — Украина целиком, Грузия и Киргизия по 10–11 месяцев).

С января 1999 по декабрь 2018 открытых месячных рядов параллельного курса нет: у IRR на это
время — только флаг «двойной рынок» (0/1) без самого курса; AREAER МВФ закрыт логином.
Для России, Беларуси, Казахстана, Азербайджана, Узбекистана, Египта и Турции после 1998 года
открытых рядов нет вовсе.

Премия = (параллельный − официальный) / официальный, в долях. Оба куска — рыночные цены,
но собраны они задним числом; считаем значение месяца m известным к концу m+1
(`parallel_premium` в `core/temporal.py`) — это допущение.
"""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW

REINHART_FILE = "33_data.xls"
REINHART_URL = "https://web.archive.org/web/20160411223943id_/http://www.carmenreinhart.com/user_uploads/data/33_data.xls"
REINHART_SHA256 = "8c720930a24ab89a74b61e31fb3f6c0f141409a6509ab09d4e1c46b48d20692f"
REINHART_LAST_MONTH = "1998-12"
CRYPTO_INDICATOR = "BIT_SHD_PT"  # премия «теневого» курса из цен биткойна к официальному, %

REINHART_NAME_TO_ISO3 = {
    "ALGERIA": "DZA", "ARGENTINA": "ARG", "AUSTRALIA": "AUS", "AUSTRIA": "AUT", "BELGIUM": "BEL", "BENIN": "BEN",
    "BOLIVIA": "BOL", "BOTSWANA": "BWA", "BRAZIL": "BRA", "BURUNDI": "BDI", "CANADA": "CAN", "CHILE": "CHL",
    "CHINA,:HONG KONG": "HKG", "CHINA,P.R.: MAINLAND": "CHN", "COLOMBIA": "COL", "CONGO, DEM. REP. OF": "COD",
    "COSTA RICA": "CRI", "CYPRUS": "CYP", "DENMARK": "DNK", "DOMINICAN REPUBLIC": "DOM", "ECUADOR": "ECU",
    "EGYPT": "EGY", "EL SALVADOR": "SLV", "FINLAND": "FIN", "FRANCE": "FRA", "GAMBIA, THE": "GMB", "GERMANY": "DEU",
    "GHANA": "GHA", "GREECE": "GRC", "GUATEMALA": "GTM", "GUINEA": "GIN", "GUYANA": "GUY", "HAITI": "HTI",
    "HONDURAS": "HND", "ICELAND": "ISL", "INDIA": "IND", "INDONESIA": "IDN", "IRAN, I.R. OF": "IRN", "IRAQ": "IRQ",
    "IRELAND": "IRL", "ISRAEL": "ISR", "ITALY": "ITA", "JAMAICA": "JAM", "JAPAN": "JPN", "JORDAN": "JOR",
    "KENYA": "KEN", "KOREA": "KOR", "KUWAIT": "KWT", "LAO PEOPLE'S DEM.REP": "LAO", "LEBANON": "LBN",
    "LESOTHO": "LSO", "LIBERIA": "LBR", "LIBYA": "LBY", "MADAGASCAR": "MDG", "MALAWI": "MWI", "MALAYSIA": "MYS",
    "MALTA": "MLT", "MAURITANIA": "MRT", "MAURITIUS": "MUS", "MEXICO": "MEX", "MOROCCO": "MAR", "MYANMAR": "MMR",
    "NEPAL": "NPL", "NETHERLANDS": "NLD", "NEW ZEALAND": "NZL", "NICARAGUA": "NIC", "NIGERIA": "NGA",
    "NORWAY": "NOR", "PAKISTAN": "PAK", "PARAGUAY": "PRY", "PERU": "PER", "PHILIPPINES": "PHL", "PORTUGAL": "PRT",
    "SAUDI ARABIA": "SAU", "SINGAPORE": "SGP", "SOUTH AFRICA": "ZAF", "SPAIN": "ESP", "SRI LANKA": "LKA",
    "SURINAME": "SUR", "SWAZILAND": "SWZ", "SWEDEN": "SWE", "SWITZERLAND": "CHE", "SYRIAN ARAB REPUBLIC": "SYR",
    "TANZANIA": "TZA", "THAILAND": "THA", "TUNISIA": "TUN", "TURKEY": "TUR", "UGANDA": "UGA",
    "UNITED KINGDOM": "GBR", "URUGUAY": "URY", "VENEZUELA, REP. BOL.": "VEN", "ZAMBIA": "ZMB", "ZIMBABWE": "ZWE",
}


def _reinhart_sheet(name: str) -> pd.DataFrame:
    d = pd.read_excel(DATA_RAW / "parallel" / REINHART_FILE, sheet_name=name, header=None, engine="xlrd")
    names = d.iloc[3, 1:].astype(str).str.strip()
    body = d.iloc[7:]
    body = body[body.iloc[:, 0].astype(str).str.fullmatch(r"\d{4}M\d{1,2}")]
    month = pd.PeriodIndex(body.iloc[:, 0].astype(str).str.replace("M", "-"), freq="M")
    vals = body.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    vals.columns = names.map(REINHART_NAME_TO_ISO3).to_numpy()
    vals.index = month
    long = vals.loc[:, vals.columns.notna()].rename_axis("month").reset_index().melt(id_vars="month", var_name="iso3", value_name=name)
    return long[long[name] > 0]


def load_reinhart() -> pd.DataFrame:
    """Архив Рейнхарт–Рогофф: iso3, month, premium (доли), по декабрь 1998."""
    d = _reinhart_sheet("black").merge(_reinhart_sheet("official"), on=["iso3", "month"])
    d["premium"] = d.black / d.official - 1
    return d[d.month <= REINHART_LAST_MONTH][["iso3", "month", "premium"]]


def load_crypto() -> pd.DataFrame:
    """Криптопремия МВФ: iso3, month, premium (доли), с января 2019."""
    d = pd.read_csv(DATA_RAW / "imf" / "crypto_parallel.csv", usecols=["COUNTRY", "INDICATOR", "TIME_PERIOD", "OBS_VALUE"])
    d = d[(d.INDICATOR == CRYPTO_INDICATOR) & d.OBS_VALUE.notna()]
    month = pd.PeriodIndex(d.TIME_PERIOD.str.replace("-M", "-"), freq="M")
    return pd.DataFrame({"iso3": d.COUNTRY.to_numpy(), "month": month, "premium": d.OBS_VALUE.to_numpy() / 100.0})


def load_parallel() -> pd.DataFrame:
    """Оба куска одной таблицей: iso3, month, premium, source (`reinhart` / `crypto`)."""
    out = pd.concat([load_reinhart().assign(source="reinhart"), load_crypto().assign(source="crypto")], ignore_index=True)
    return out.sort_values(["iso3", "month"]).reset_index(drop=True)
