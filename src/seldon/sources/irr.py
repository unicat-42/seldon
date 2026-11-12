"""Фактический режим курса по Ilzetzki–Reinhart–Rogoff (IRR). Скачивание: `download irr`.

Источник — сайт автора (ilzetzki.com/irr-data), стабильный релиз ноября 2021: месячная
классификация 1940–2019, 194 страны. Берём «тонкую» шкалу (лист `Fine`, коды 1–15):
    1–4    нет своей валюты, валютный совет, объявленная или фактическая привязка;
    5–8    ползущая привязка и узкий коридор (до ±2 %);
    9–12   широкий коридор, управляемое плавание;
    13     свободное плавание;   14 — «свободное падение» (инфляция > 40 % или обвал);
    15     двойной рынок без данных о параллельном курсе.

Хроника кончается 2019 годом (`IRR_LAST_YEAR`, инвариант 6): дальше режим не «плавающий», а
неизвестный. Признак после этого года заморожен на последнем известном значении с флагом
(`struct_features.py`). В октябре 2026 автор выложил бета-версию по 2025 год — она не
подключена: в ней пропуски («n.a.») у Украины 2016–19 и Нигерии 2020–25, релиз не финальный.

Классификация ретроспективная («свободное падение» ставится по факту обвала), поэтому год Y
используется с большим лагом — см. `irr_regime` в `core/temporal.py`.
"""

from __future__ import annotations

import re

import pandas as pd

from seldon import DATA_RAW

IRR_FILE = "ERA_Classification_Monthly_1940-2019.xlsx"
IRR_URL = "https://www.ilzetzki.com/_files/ugd/b3763a_242513d0fba24aa1a64be41c8f73d887.xlsx"
IRR_LAST_YEAR = 2019
PEG_MAX_CODE = 8  # коды 1–8: привязка, ползущая привязка, узкий коридор

# название страны в файле (без лишних пробелов) → ISO3
IRR_NAME_TO_ISO3 = {
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Angola": "AGO", "Anguilla": "AIA",
    "Antigua and Barbuda": "ATG", "Argentina": "ARG", "Armenia": "ARM", "Aruba": "ABW", "Australia": "AUS",
    "Austria": "AUT", "Azerbaijan Rep. of": "AZE", "The Bahamas": "BHS", "Bahrain Kingdom of": "BHR",
    "Bangladesh": "BGD", "Barbados": "BRB", "Belarus": "BLR", "Belgium": "BEL", "Belize": "BLZ", "Benin": "BEN",
    "Bermuda": "BMU", "Bhutan": "BTN", "Bolivia": "BOL", "Bosnia & Herzegovina": "BIH", "Botswana": "BWA",
    "Brazil": "BRA", "Brunei Darussalam": "BRN", "Bulgaria": "BGR", "Burkina Faso": "BFA", "Burundi": "BDI",
    "Cabo Verde": "CPV", "Cambodia": "KHM", "Cameroon": "CMR", "Canada": "CAN", "Central African Rep.": "CAF",
    "Chad": "TCD", "Chile": "CHL", "China, PR": "CHN", "Colombia": "COL", "Comoros": "COM",
    "Congo Dem. Rep. of": "COD", "Congo Rep. of": "COG", "Costa Rica": "CRI", 'Cote D"Ivoire': "CIV",
    "Croatia": "HRV", "Curacao": "CUW", "Cyprus": "CYP", "Czech Rep.": "CZE", "Denmark": "DNK", "Djibouti": "DJI",
    "Dominica": "DMA", "Dominican Republic": "DOM", "Ecuador": "ECU", "Egypt": "EGY", "El Salvador": "SLV",
    "Equatorial Guinea": "GNQ", "Eritrea": "ERI", "Estonia": "EST", "Ethiopia": "ETH", "Fiji": "FJI",
    "Finland": "FIN", "France": "FRA", "Gabon": "GAB", "The Gambia": "GMB", "Georgia": "GEO", "Germany": "DEU",
    "Ghana": "GHA", "Greece": "GRC", "Grenada": "GRD", "Guatemala": "GTM", "Guinea": "GIN", "Guinea Bissau": "GNB",
    "Guyana": "GUY", "Haiti": "HTI", "Honduras": "HND", "Hong Kong": "HKG", "Hungary": "HUN", "Iceland": "ISL",
    "India": "IND", "Indonesia": "IDN", "Iran": "IRN", "Iraq": "IRQ", "Ireland": "IRL", "Israel": "ISR",
    "Italy": "ITA", "Jamaica": "JAM", "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ", "Kenya": "KEN",
    "Kiribati": "KIR", "Korea": "KOR", "Kuwait": "KWT", "Kyrgyz Rep.": "KGZ", "Lao Dem. Rep.": "LAO",
    "Latvia": "LVA", "Lebanon": "LBN", "Lesotho": "LSO", "Liberia": "LBR", "Libya": "LBY", "Liechtesntein": "LIE",
    "Lithuania": "LTU", "Luxembourg": "LUX", "Macao": "MAC", "Macedonia FYR": "MKD", "Madagascar": "MDG",
    "Malawi": "MWI", "Malaysia": "MYS", "Maldives": "MDV", "Mali": "MLI", "Malta": "MLT", "Marshall Islands": "MHL",
    "Mauritania": "MRT", "Mauritius": "MUS", "Mexico": "MEX", "Micronesia": "FSM", "Moldova": "MDA",
    "Monaco": "MCO", "Mongolia": "MNG", "Montenegro": "MNE", "Morocco": "MAR", "Mozambique": "MOZ",
    "Myanmar": "MMR", "Namibia": "NAM", "Nepal": "NPL", "Netherlands": "NLD", "Netherlands Antilles": "ANT",
    "New Zealand": "NZL", "Nicaragua": "NIC", "Niger": "NER", "Nigeria": "NGA", "Norway": "NOR", "Oman": "OMN",
    "Pakistan": "PAK", "Palau": "PLW", "Panama": "PAN", "PNG": "PNG", "Paraguay": "PRY", "Peru": "PER",
    "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT", "Qatar": "QAT", "Romania": "ROU", "Russia": "RUS",
    "Rwanda": "RWA", "Samoa": "WSM", "San Marino": "SMR", "Sao Tome & Principe": "STP", "Saudi Arabia": "SAU",
    "Senegal": "SEN", "Serbia, Rep. of": "SRB", "Seychelles": "SYC", "Sierra Leone": "SLE", "Singapore": "SGP",
    "Slovak Republic": "SVK", "Slovenia": "SVN", "Solomon Islands": "SLB", "Somalia": "SOM", "South Africa": "ZAF",
    "Spain": "ESP", "Sri Lanka": "LKA", "St. Kitts and Nevis": "KNA", "St. Lucia": "LCA",
    "St Vincent & Grenadines": "VCT", "Sudan": "SDN", "Suriname": "SUR", "Swaziland": "SWZ", "Sweden": "SWE",
    "Switzerland": "CHE", "Syrian Arab Rep.": "SYR", "Tajikistan": "TJK", "Tanzania": "TZA", "Thailand": "THA",
    "Togo": "TGO", "Tonga": "TON", "Trinidad Tobago": "TTO", "Tunisia": "TUN", "Turkey": "TUR",
    "Turkmenistan": "TKM", "Uganda": "UGA", "Ukraine": "UKR", "UAE": "ARE", "United Kingdom": "GBR",
    "United States": "USA", "Uruguay": "URY", "Uzbekistan": "UZB", "Vanuatu": "VUT", "Venezuela": "VEN",
    "Vietnam": "VNM", "West Bank and Gaza": "PSE", "Yemen Rep. of": "YEM", "Zambia": "ZMB", "Zimbabwe": "ZWE",
}


def load_irr() -> pd.DataFrame:
    """Режим курса на декабрь каждого года: iso3, year, irr_fine (1–15; пропуск — «n.a.»)."""
    raw = pd.read_excel(DATA_RAW / "irr" / IRR_FILE, sheet_name="Fine", header=None)
    names = [" ".join(f"{a} {b}".split()) for a, b in
             zip(raw.iloc[4, 2:].fillna("").astype(str), raw.iloc[5, 2:].fillna("").astype(str))]
    unknown = [n for n in names if n not in IRR_NAME_TO_ISO3]
    if unknown:
        raise ValueError(f"IRR: неизвестные названия стран {unknown} — дополни IRR_NAME_TO_ISO3")
    body = raw.iloc[7:, 1:].copy()
    body.columns = ["period", *[IRR_NAME_TO_ISO3[n] for n in names]]
    body = body[body.period.astype(str).str.fullmatch(r"\d{4}M12")]
    body["year"] = body.period.map(lambda s: int(re.match(r"\d{4}", s).group()))
    long = body.drop(columns="period").melt(id_vars="year", var_name="iso3", value_name="irr_fine")
    long["irr_fine"] = pd.to_numeric(long.irr_fine, errors="coerce")
    long = long[long.irr_fine.notna() & (long.year <= IRR_LAST_YEAR)]
    return long[["iso3", "year", "irr_fine"]].sort_values(["iso3", "year"]).reset_index(drop=True)
