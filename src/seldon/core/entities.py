"""Страны и их коды в разных базах (ISO3, Gleditsch–Ward, COW, Polity, FIPS), валютные союзы.

UCDP использует коды Gleditsch–Ward (GW) — система нумерации стран, принятая
в исследованиях конфликтов. Она похожа на коды Correlates of War (COW), которые
использует PITF, но не везде совпадает.
"""

# name — как страна называется в UCDP; gw — код Gleditsch–Ward; iso3 — трёхбуквенный ISO
POST_SOVIET = [
    {"name": "Russia (Soviet Union)", "short": "Россия", "gw": 365, "iso3": "RUS"},
    {"name": "Ukraine", "short": "Украина", "gw": 369, "iso3": "UKR"},
    {"name": "Belarus", "short": "Беларусь", "gw": 370, "iso3": "BLR"},
    {"name": "Moldova", "short": "Молдова", "gw": 359, "iso3": "MDA"},
    {"name": "Georgia", "short": "Грузия", "gw": 372, "iso3": "GEO"},
    {"name": "Armenia", "short": "Армения", "gw": 371, "iso3": "ARM"},
    {"name": "Azerbaijan", "short": "Азербайджан", "gw": 373, "iso3": "AZE"},
    {"name": "Kazakhstan", "short": "Казахстан", "gw": 705, "iso3": "KAZ"},
    {"name": "Kyrgyzstan", "short": "Кыргызстан", "gw": 703, "iso3": "KGZ"},
    {"name": "Tajikistan", "short": "Таджикистан", "gw": 702, "iso3": "TJK"},
    {"name": "Turkmenistan", "short": "Туркменистан", "gw": 701, "iso3": "TKM"},
    {"name": "Uzbekistan", "short": "Узбекистан", "gw": 704, "iso3": "UZB"},
    {"name": "Estonia", "short": "Эстония", "gw": 366, "iso3": "EST"},
    {"name": "Latvia", "short": "Латвия", "gw": 367, "iso3": "LVA"},
    {"name": "Lithuania", "short": "Литва", "gw": 368, "iso3": "LTU"},
]

POST_SOVIET_GW = {c["gw"] for c in POST_SOVIET}
POST_SOVIET_ISO3 = {c["iso3"] for c in POST_SOVIET}
GW_TO_SHORT = {c["gw"]: c["short"] for c in POST_SOVIET}
ISO3_TO_SHORT = {c["iso3"]: c["short"] for c in POST_SOVIET}


# Коды Gleditsch–Ward → ISO3 (для Всемирного банка и других источников с ISO).
# None — государства, которых нет в современных ISO/ВБ (ГДР, Чехословакия, Южный Йемен).
# Сербия (340) и Югославия (345) обе → SRB; Косово → XKX (код ВБ, не ISO).
GW_TO_ISO3: dict[int, str | None] = {
    2: "USA", 20: "CAN", 31: "BHS", 40: "CUB", 41: "HTI", 42: "DOM", 51: "JAM", 52: "TTO",
    53: "BRB", 70: "MEX", 80: "BLZ", 90: "GTM", 91: "HND", 92: "SLV", 93: "NIC", 94: "CRI",
    95: "PAN", 100: "COL", 101: "VEN", 110: "GUY", 115: "SUR", 130: "ECU", 135: "PER",
    140: "BRA", 145: "BOL", 150: "PRY", 155: "CHL", 160: "ARG", 165: "URY", 200: "GBR",
    205: "IRL", 210: "NLD", 211: "BEL", 212: "LUX", 220: "FRA", 225: "CHE", 230: "ESP",
    235: "PRT", 260: "DEU", 265: None, 290: "POL", 305: "AUT", 310: "HUN", 315: None,
    316: "CZE", 317: "SVK", 325: "ITA", 338: "MLT", 339: "ALB", 340: "SRB", 341: "MNE",
    343: "MKD", 344: "HRV", 345: "SRB", 346: "BIH", 347: "XKX", 349: "SVN", 350: "GRC",
    352: "CYP", 355: "BGR", 359: "MDA", 360: "ROU", 365: "RUS", 366: "EST", 367: "LVA",
    368: "LTU", 369: "UKR", 370: "BLR", 371: "ARM", 372: "GEO", 373: "AZE", 375: "FIN",
    380: "SWE", 385: "NOR", 390: "DNK", 395: "ISL", 402: "CPV", 404: "GNB", 411: "GNQ",
    420: "GMB", 432: "MLI", 433: "SEN", 434: "BEN", 435: "MRT", 436: "NER", 437: "CIV",
    438: "GIN", 439: "BFA", 450: "LBR", 451: "SLE", 452: "GHA", 461: "TGO", 471: "CMR",
    475: "NGA", 481: "GAB", 482: "CAF", 483: "TCD", 484: "COG", 490: "COD", 500: "UGA",
    501: "KEN", 510: "TZA", 516: "BDI", 517: "RWA", 520: "SOM", 522: "DJI", 530: "ETH",
    531: "ERI", 540: "AGO", 541: "MOZ", 551: "ZMB", 552: "ZWE", 553: "MWI", 560: "ZAF",
    565: "NAM", 570: "LSO", 571: "BWA", 572: "SWZ", 580: "MDG", 581: "COM", 590: "MUS",
    600: "MAR", 615: "DZA", 616: "TUN", 620: "LBY", 625: "SDN", 626: "SSD", 630: "IRN",
    640: "TUR", 645: "IRQ", 651: "EGY", 652: "SYR", 660: "LBN", 663: "JOR", 666: "ISR",
    670: "SAU", 678: "YEM", 680: None, 690: "KWT", 692: "BHR", 694: "QAT", 696: "ARE",
    698: "OMN", 700: "AFG", 701: "TKM", 702: "TJK", 703: "KGZ", 704: "UZB", 705: "KAZ",
    710: "CHN", 712: "MNG", 713: "TWN", 731: "PRK", 732: "KOR", 740: "JPN", 750: "IND",
    760: "BTN", 770: "PAK", 771: "BGD", 775: "MMR", 780: "LKA", 781: "MDV", 790: "NPL",
    800: "THA", 811: "KHM", 812: "LAO", 816: "VNM", 820: "MYS", 830: "SGP", 835: "BRN",
    840: "PHL", 850: "IDN", 860: "TLS", 900: "AUS", 910: "PNG", 920: "NZL", 940: "SLB",
    950: "FJI",
}

# ---------------------------------------------------------------------------
# Единые таблицы кодов (аудит 12.11.2026): все модули берут маппинги отсюда.
# ---------------------------------------------------------------------------

# ISO3 → GW для ныне существующих государств (SRB → 340: Сербия; 345 — Югославия до 2006).
ISO3_TO_GW: dict[str, int] = {iso: gw for gw, iso in GW_TO_ISO3.items() if iso and gw != 345}

# Вселенная валютных моделей — государства из списка Gleditsch–Ward (решение Ангелины 14.11.2026).
# Потоки МВФ содержат ещё зависимые территории (Гонконг, Аруба, Джерси, Гренландия…), агрегаты
# (`G163` — еврозона) и территории с чужой валютой (Косово): они почти никогда не девальвируют, и
# пока были в обучении и оценке, модель получала прибавку к AUC за то, что их узнавала. По этому
# же списку публикуется таблица риска — учимся и оцениваем на том, что публикуем.
STATE_ISO3: frozenset[str] = frozenset(v for v in GW_TO_ISO3.values() if v) | frozenset(c["iso3"] for c in POST_SOVIET)


def states_only(df, col: str = "iso3"):
    """Строки таблицы, где `col` — государство из `STATE_ISO3`."""
    return df[df[col].isin(STATE_ISO3)]

# Исторические коды ISO3 в CEPII GeoDist (Mayer & Zignago 2011) → коды GW.
# YUG в CEPII кодирует Сербию и Черногорию до распада 2006 года; при разборе CEPII
# строки YUG раскрываются на 340 (Сербия), 341 (Черногория) и 345 (Югославия до 2006).
CEPII_ISO3_ALIASES: dict[str, tuple[int, ...]] = {
    "ROM": (360,),           # Румыния (ROU)
    "ZAR": (490,),           # ДР Конго (COD)
    "TMP": (860,),           # Восточный Тимор (TLS)
    "YUG": (340, 341, 345),  # Сербия, Черногория, Югославия
}

# Correlates of War → Gleditsch–Ward, где коды расходятся (1989+). Сербия: COW 345 = GW 340
# с 2006 и GW 345 до того — в данных о соседях дублируется под обоими кодами (SERBIA_COW).
COW_TO_GW: dict[int, int] = {
    255: 260,  # Германия
    679: 678,  # Йемен
    946: 970, 970: 971, 955: 972, 947: 973,  # Кирибати, Науру, Тонга, Тувалу
}
SERBIA_COW, SERBIA_GW = 345, 340

# Polity5 (≈ COW, но со своими кодами для государств-преемников) → Gleditsch–Ward.
POLITY_TO_GW: dict[int, int] = {
    **COW_TO_GW,
    364: 365,  # СССР → Россия
    342: 340,  # Сербия
    348: 341,  # Черногория
    525: 626,  # Южный Судан
    529: 530,  # Эфиопия после 1993
    818: 816,  # Вьетнам
}

# ISO3 → FIPS 10-4 (так GDELT кодирует страну источника); только страны, которые использовались.
ISO3_TO_FIPS: dict[str, str] = {"TUR": "TU", "ARG": "AR", "EGY": "EG", "GHA": "GH", "NGA": "NI", "PAK": "PK", "LKA": "CE", "LBN": "LE",
    "KAZ": "KZ", "RUS": "RS", "UKR": "UP", "BRA": "BR", "COL": "CO", "ZAF": "SF", "MEX": "MX", "IDN": "ID",
    "KOR": "KS", "GBR": "UK", "JPN": "JA", "HUN": "HU", "CHL": "CI", "BLR": "BO", "UZB": "UZ", "ZMB": "ZA",
    "ETH": "ET", "AGO": "AO", "KGZ": "KG", "GEO": "GG", "VEN": "VE", "ZWE": "ZI", "MWI": "MI", "SDN": "SU",
    "IRQ": "IZ", "LAO": "LA", "SLE": "SL", "TJK": "TI", "NZL": "NZ", "BWA": "BC", "SUR": "NS"}


def to_gw(code: int, system: str) -> int:
    """Код COW или Polity → код Gleditsch–Ward."""
    table = {"cow": COW_TO_GW, "polity": POLITY_TO_GW}[system]
    return table.get(int(code), int(code))


SERBIA_SINCE = 2006  # COW 345 — Югославия до 2006 (GW 345) и Сербия после (GW 340)


def cow_year_to_gw(ccode, year):
    """Колонки кодов COW и годов → коды Gleditsch–Ward с учётом Сербии: COW 345 — это
    GW 345 (Югославия) до 2006 и GW 340 (Сербия) с 2006."""
    gw = ccode.astype(int).map(lambda c: COW_TO_GW.get(c, c))
    return gw.where(~((ccode.astype(int) == SERBIA_COW) & (year >= SERBIA_SINCE)), SERBIA_GW)


def ccode_to_gw(ccode):
    """Колонка кодов `ccode` из REIGN и Mass Mobilization (COW со своими кодами
    государств-преемников, как в Polity) → коды Gleditsch–Ward."""
    return ccode.astype(int).map(lambda c: POLITY_TO_GW.get(c, c))


# ---------------------------------------------------------------------------
# ISO 3166: двухбуквенный код → трёхбуквенный (список стран Всемирного банка + Тайвань).
# Нужен источникам, которые подписывают страны двумя буквами (REER Bruegel).
# ---------------------------------------------------------------------------

ISO2_TO_ISO3: dict[str, str] = {
    "AD": "AND", "AE": "ARE", "AF": "AFG", "AG": "ATG", "AL": "ALB", "AM": "ARM", "AO": "AGO", "AR": "ARG", "AS":
    "ASM", "AT": "AUT", "AU": "AUS", "AW": "ABW", "AZ": "AZE", "BA": "BIH", "BB": "BRB", "BD": "BGD", "BE": "BEL",
    "BF": "BFA", "BG": "BGR", "BH": "BHR", "BI": "BDI", "BJ": "BEN", "BM": "BMU", "BN": "BRN", "BO": "BOL", "BR":
    "BRA", "BS": "BHS", "BT": "BTN", "BW": "BWA", "BY": "BLR", "BZ": "BLZ", "CA": "CAN", "CD": "COD", "CF": "CAF",
    "CG": "COG", "CH": "CHE", "CI": "CIV", "CL": "CHL", "CM": "CMR", "CN": "CHN", "CO": "COL", "CR": "CRI", "CU":
    "CUB", "CV": "CPV", "CW": "CUW", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DJ": "DJI", "DK": "DNK", "DM": "DMA",
    "DO": "DOM", "DZ": "DZA", "EC": "ECU", "EE": "EST", "EG": "EGY", "ER": "ERI", "ES": "ESP", "ET": "ETH", "FI":
    "FIN", "FJ": "FJI", "FM": "FSM", "FO": "FRO", "FR": "FRA", "GA": "GAB", "GB": "GBR", "GD": "GRD", "GE": "GEO",
    "GH": "GHA", "GI": "GIB", "GL": "GRL", "GM": "GMB", "GN": "GIN", "GQ": "GNQ", "GR": "GRC", "GT": "GTM", "GU":
    "GUM", "GW": "GNB", "GY": "GUY", "HK": "HKG", "HN": "HND", "HR": "HRV", "HT": "HTI", "HU": "HUN", "ID": "IDN",
    "IE": "IRL", "IL": "ISR", "IM": "IMN", "IN": "IND", "IQ": "IRQ", "IR": "IRN", "IS": "ISL", "IT": "ITA", "JG":
    "CHI", "JM": "JAM", "JO": "JOR", "JP": "JPN", "KE": "KEN", "KG": "KGZ", "KH": "KHM", "KI": "KIR", "KM": "COM",
    "KN": "KNA", "KP": "PRK", "KR": "KOR", "KW": "KWT", "KY": "CYM", "KZ": "KAZ", "LA": "LAO", "LB": "LBN", "LC":
    "LCA", "LI": "LIE", "LK": "LKA", "LR": "LBR", "LS": "LSO", "LT": "LTU", "LU": "LUX", "LV": "LVA", "LY": "LBY",
    "MA": "MAR", "MC": "MCO", "MD": "MDA", "ME": "MNE", "MF": "MAF", "MG": "MDG", "MH": "MHL", "MK": "MKD", "ML":
    "MLI", "MM": "MMR", "MN": "MNG", "MO": "MAC", "MP": "MNP", "MR": "MRT", "MT": "MLT", "MU": "MUS", "MV": "MDV",
    "MW": "MWI", "MX": "MEX", "MY": "MYS", "MZ": "MOZ", "NA": "NAM", "NC": "NCL", "NE": "NER", "NG": "NGA", "NI":
    "NIC", "NL": "NLD", "NO": "NOR", "NP": "NPL", "NR": "NRU", "NZ": "NZL", "OM": "OMN", "PA": "PAN", "PE": "PER",
    "PF": "PYF", "PG": "PNG", "PH": "PHL", "PK": "PAK", "PL": "POL", "PR": "PRI", "PS": "PSE", "PT": "PRT", "PW":
    "PLW", "PY": "PRY", "QA": "QAT", "RO": "ROU", "RS": "SRB", "RU": "RUS", "RW": "RWA", "SA": "SAU", "SB": "SLB",
    "SC": "SYC", "SD": "SDN", "SE": "SWE", "SG": "SGP", "SI": "SVN", "SK": "SVK", "SL": "SLE", "SM": "SMR", "SN":
    "SEN", "SO": "SOM", "SR": "SUR", "SS": "SSD", "ST": "STP", "SV": "SLV", "SX": "SXM", "SY": "SYR", "SZ": "SWZ",
    "TC": "TCA", "TD": "TCD", "TG": "TGO", "TH": "THA", "TJ": "TJK", "TL": "TLS", "TM": "TKM", "TN": "TUN", "TO":
    "TON", "TR": "TUR", "TT": "TTO", "TV": "TUV", "TW": "TWN", "TZ": "TZA", "UA": "UKR", "UG": "UGA", "US": "USA",
    "UY": "URY", "UZ": "UZB", "VC": "VCT", "VE": "VEN", "VG": "VGB", "VI": "VIR", "VN": "VNM", "VU": "VUT", "WS":
    "WSM", "XK": "XKX", "YE": "YEM", "ZA": "ZAF", "ZM": "ZMB", "ZW": "ZWE",
}


# ---------------------------------------------------------------------------
# Общие и чужие валюты: (iso3, валюта, первый год, последний год или None = по сей день).
# У такой страны нет «своего» риска девальвации: курс к доллару — это курс евро / франка
# КФА / сам доллар. В валютном блоке такие страна-годы исключаются из обучения, оценки и
# индивидуального прогноза — строго с года перехода (до него валюта своя и риск настоящий).
# ---------------------------------------------------------------------------

_EURO = {
    1999: ["AUT", "BEL", "FIN", "FRA", "DEU", "IRL", "ITA", "LUX", "NLD", "PRT", "ESP"],
    2001: ["GRC"], 2007: ["SVN"], 2008: ["CYP", "MLT"], 2009: ["SVK"], 2011: ["EST"],
    2014: ["LVA"], 2015: ["LTU"], 2023: ["HRV"], 2026: ["BGR"],
    2002: ["MNE", "XKX", "AND", "MCO", "SMR"],  # евро без членства в еврозоне
}
_XOF = {1945: ["BEN", "BFA", "CIV", "NER", "SEN", "TGO"], 1984: ["MLI"], 1997: ["GNB"]}
_XAF = {1945: ["CMR", "CAF", "TCD", "COG", "GAB"], 1985: ["GNQ"]}
_XCD = {1965: ["ATG", "DMA", "GRD", "KNA", "LCA", "VCT"]}

SHARED_CURRENCY: list[tuple[str, str, int, int | None]] = [
    *((iso, "EUR", y, None) for y, isos in _EURO.items() for iso in isos),
    *((iso, "XOF", y, None) for y, isos in _XOF.items() for iso in isos),
    *((iso, "XAF", y, None) for y, isos in _XAF.items() for iso in isos),
    *((iso, "XCD", y, None) for y, isos in _XCD.items() for iso in isos),
    # долларизация
    ("PAN", "USD", 1904, None), ("ECU", "USD", 2000, None), ("SLV", "USD", 2001, None),
    ("TLS", "USD", 2000, None), ("ZWE", "USD", 2009, 2018),
    ("FSM", "USD", 1986, None), ("MHL", "USD", 1986, None), ("PLW", "USD", 1994, None),
    # чужая валюта у микрогосударств
    ("KIR", "AUD", 1979, None), ("NRU", "AUD", 1968, None), ("TUV", "AUD", 1978, None),
    ("LIE", "CHF", 1924, None),
]


def shared_currency(iso3: str, year: int) -> str | None:
    """Код общей/чужой валюты страны в данном году или None, если валюта своя."""
    for iso, cur, y0, y1 in SHARED_CURRENCY:
        if iso == iso3 and year >= y0 and (y1 is None or year <= y1):
            return cur
    return None


def shared_currency_table(first_year: int, last_year: int):
    """Таблица (iso3, year, fx_regime) для страна-лет с общей/чужой валютой — для merge."""
    import pandas as pd

    rows = [(iso, y, cur) for iso, cur, y0, y1 in SHARED_CURRENCY
            for y in range(max(y0, first_year), min(y1 or last_year, last_year) + 1)]
    return pd.DataFrame(rows, columns=["iso3", "year", "fx_regime"])


# ---------------------------------------------------------------------------
# Названия государств витрины для веб-интерфейса (Этап 5, чекпоинт 5.1): ISO3 → (русское,
# английское). 174 государства системы Gleditsch–Ward из `reports/risk_2025_26.csv`.
# Русские названия — краткие общеупотребительные; у постсоветских стран — те же, что в
# `POST_SOVIET` (Беларусь, Молдова, Кыргызстан). `ISO3_TO_SHORT` и колонка `country` витрины
# не меняются: таблицы конвейера остаются байт в байт прежними.
# ---------------------------------------------------------------------------

COUNTRY_NAMES: dict[str, tuple[str, str]] = {
    "AFG": ("Афганистан", "Afghanistan"), "AGO": ("Ангола", "Angola"), "ALB": ("Албания", "Albania"),
    "ARE": ("ОАЭ", "United Arab Emirates"), "ARG": ("Аргентина", "Argentina"), "ARM": ("Армения", "Armenia"),
    "AUS": ("Австралия", "Australia"), "AUT": ("Австрия", "Austria"), "AZE": ("Азербайджан", "Azerbaijan"),
    "BDI": ("Бурунди", "Burundi"), "BEL": ("Бельгия", "Belgium"), "BEN": ("Бенин", "Benin"),
    "BFA": ("Буркина-Фасо", "Burkina Faso"), "BGD": ("Бангладеш", "Bangladesh"), "BGR": ("Болгария", "Bulgaria"),
    "BHR": ("Бахрейн", "Bahrain"), "BHS": ("Багамы", "Bahamas"), "BIH": ("Босния и Герцеговина", "Bosnia and Herzegovina"),
    "BLR": ("Беларусь", "Belarus"), "BLZ": ("Белиз", "Belize"), "BOL": ("Боливия", "Bolivia"),
    "BRA": ("Бразилия", "Brazil"), "BRB": ("Барбадос", "Barbados"), "BRN": ("Бруней", "Brunei"),
    "BTN": ("Бутан", "Bhutan"), "BWA": ("Ботсвана", "Botswana"), "CAF": ("ЦАР", "Central African Republic"),
    "CAN": ("Канада", "Canada"), "CHE": ("Швейцария", "Switzerland"), "CHL": ("Чили", "Chile"),
    "CHN": ("Китай", "China"), "CIV": ("Кот-д’Ивуар", "Côte d’Ivoire"), "CMR": ("Камерун", "Cameroon"),
    "COD": ("ДР Конго", "DR Congo"), "COG": ("Республика Конго", "Republic of the Congo"), "COL": ("Колумбия", "Colombia"),
    "COM": ("Коморы", "Comoros"), "CPV": ("Кабо-Верде", "Cabo Verde"), "CRI": ("Коста-Рика", "Costa Rica"),
    "CUB": ("Куба", "Cuba"), "CYP": ("Кипр", "Cyprus"), "CZE": ("Чехия", "Czechia"),
    "DEU": ("Германия", "Germany"), "DJI": ("Джибути", "Djibouti"), "DNK": ("Дания", "Denmark"),
    "DOM": ("Доминиканская Республика", "Dominican Republic"), "DZA": ("Алжир", "Algeria"), "ECU": ("Эквадор", "Ecuador"),
    "EGY": ("Египет", "Egypt"), "ERI": ("Эритрея", "Eritrea"), "ESP": ("Испания", "Spain"),
    "EST": ("Эстония", "Estonia"), "ETH": ("Эфиопия", "Ethiopia"), "FIN": ("Финляндия", "Finland"),
    "FJI": ("Фиджи", "Fiji"), "FRA": ("Франция", "France"), "GAB": ("Габон", "Gabon"),
    "GBR": ("Великобритания", "United Kingdom"), "GEO": ("Грузия", "Georgia"), "GHA": ("Гана", "Ghana"),
    "GIN": ("Гвинея", "Guinea"), "GMB": ("Гамбия", "Gambia"), "GNB": ("Гвинея-Бисау", "Guinea-Bissau"),
    "GNQ": ("Экваториальная Гвинея", "Equatorial Guinea"), "GRC": ("Греция", "Greece"), "GTM": ("Гватемала", "Guatemala"),
    "GUY": ("Гайана", "Guyana"), "HND": ("Гондурас", "Honduras"), "HRV": ("Хорватия", "Croatia"),
    "HTI": ("Гаити", "Haiti"), "HUN": ("Венгрия", "Hungary"), "IDN": ("Индонезия", "Indonesia"),
    "IND": ("Индия", "India"), "IRL": ("Ирландия", "Ireland"), "IRN": ("Иран", "Iran"),
    "IRQ": ("Ирак", "Iraq"), "ISL": ("Исландия", "Iceland"), "ISR": ("Израиль", "Israel"),
    "ITA": ("Италия", "Italy"), "JAM": ("Ямайка", "Jamaica"), "JOR": ("Иордания", "Jordan"),
    "JPN": ("Япония", "Japan"), "KAZ": ("Казахстан", "Kazakhstan"), "KEN": ("Кения", "Kenya"),
    "KGZ": ("Кыргызстан", "Kyrgyzstan"), "KHM": ("Камбоджа", "Cambodia"), "KOR": ("Южная Корея", "South Korea"),
    "KWT": ("Кувейт", "Kuwait"), "LAO": ("Лаос", "Laos"), "LBN": ("Ливан", "Lebanon"),
    "LBR": ("Либерия", "Liberia"), "LBY": ("Ливия", "Libya"), "LKA": ("Шри-Ланка", "Sri Lanka"),
    "LSO": ("Лесото", "Lesotho"), "LTU": ("Литва", "Lithuania"), "LUX": ("Люксембург", "Luxembourg"),
    "LVA": ("Латвия", "Latvia"), "MAR": ("Марокко", "Morocco"), "MDA": ("Молдова", "Moldova"),
    "MDG": ("Мадагаскар", "Madagascar"), "MDV": ("Мальдивы", "Maldives"), "MEX": ("Мексика", "Mexico"),
    "MKD": ("Северная Македония", "North Macedonia"), "MLI": ("Мали", "Mali"), "MLT": ("Мальта", "Malta"),
    "MMR": ("Мьянма", "Myanmar"), "MNE": ("Черногория", "Montenegro"), "MNG": ("Монголия", "Mongolia"),
    "MOZ": ("Мозамбик", "Mozambique"), "MRT": ("Мавритания", "Mauritania"), "MUS": ("Маврикий", "Mauritius"),
    "MWI": ("Малави", "Malawi"), "MYS": ("Малайзия", "Malaysia"), "NAM": ("Намибия", "Namibia"),
    "NER": ("Нигер", "Niger"), "NGA": ("Нигерия", "Nigeria"), "NIC": ("Никарагуа", "Nicaragua"),
    "NLD": ("Нидерланды", "Netherlands"), "NOR": ("Норвегия", "Norway"), "NPL": ("Непал", "Nepal"),
    "NZL": ("Новая Зеландия", "New Zealand"), "OMN": ("Оман", "Oman"), "PAK": ("Пакистан", "Pakistan"),
    "PAN": ("Панама", "Panama"), "PER": ("Перу", "Peru"), "PHL": ("Филиппины", "Philippines"),
    "PNG": ("Папуа — Новая Гвинея", "Papua New Guinea"), "POL": ("Польша", "Poland"), "PRK": ("КНДР", "North Korea"),
    "PRT": ("Португалия", "Portugal"), "PRY": ("Парагвай", "Paraguay"), "QAT": ("Катар", "Qatar"),
    "ROU": ("Румыния", "Romania"), "RUS": ("Россия", "Russia"), "RWA": ("Руанда", "Rwanda"),
    "SAU": ("Саудовская Аравия", "Saudi Arabia"), "SDN": ("Судан", "Sudan"), "SEN": ("Сенегал", "Senegal"),
    "SGP": ("Сингапур", "Singapore"), "SLB": ("Соломоновы Острова", "Solomon Islands"), "SLE": ("Сьерра-Леоне", "Sierra Leone"),
    "SLV": ("Сальвадор", "El Salvador"), "SOM": ("Сомали", "Somalia"), "SRB": ("Сербия", "Serbia"),
    "SSD": ("Южный Судан", "South Sudan"), "SUR": ("Суринам", "Suriname"), "SVK": ("Словакия", "Slovakia"),
    "SVN": ("Словения", "Slovenia"), "SWE": ("Швеция", "Sweden"), "SWZ": ("Эсватини", "Eswatini"),
    "SYR": ("Сирия", "Syria"), "TCD": ("Чад", "Chad"), "TGO": ("Того", "Togo"),
    "THA": ("Таиланд", "Thailand"), "TJK": ("Таджикистан", "Tajikistan"), "TKM": ("Туркменистан", "Turkmenistan"),
    "TLS": ("Восточный Тимор", "Timor-Leste"), "TTO": ("Тринидад и Тобаго", "Trinidad and Tobago"), "TUN": ("Тунис", "Tunisia"),
    "TUR": ("Турция", "Turkey"), "TWN": ("Тайвань", "Taiwan"), "TZA": ("Танзания", "Tanzania"),
    "UGA": ("Уганда", "Uganda"), "UKR": ("Украина", "Ukraine"), "URY": ("Уругвай", "Uruguay"),
    "USA": ("США", "United States"), "UZB": ("Узбекистан", "Uzbekistan"), "VEN": ("Венесуэла", "Venezuela"),
    "VNM": ("Вьетнам", "Vietnam"), "XKX": ("Косово", "Kosovo"), "YEM": ("Йемен", "Yemen"),
    "ZAF": ("ЮАР", "South Africa"), "ZMB": ("Замбия", "Zambia"), "ZWE": ("Зимбабве", "Zimbabwe"),
}
# Как страну ещё ищут: полные формы сокращений и распространённые варианты (для поиска на сайте)
COUNTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "ARE": ("Объединённые Арабские Эмираты", "Эмираты", "UAE"), "BLR": ("Белоруссия",), "CAF": ("Центральноафриканская Республика",),
    "CIV": ("Берег Слоновой Кости", "Ivory Coast", "Кот-д'Ивуар", "Cote d'Ivoire"), "COD": ("Демократическая Республика Конго", "Заир", "Congo-Kinshasa"),
    "COG": ("Конго", "Congo-Brazzaville"), "CPV": ("Острова Зелёного Мыса", "Cape Verde"), "CZE": ("Чешская Республика", "Czech Republic"),
    "GBR": ("Соединённое Королевство", "Англия", "UK", "Great Britain"), "KGZ": ("Киргизия",), "KOR": ("Республика Корея", "Korea"),
    "MDA": ("Молдавия",), "MKD": ("Македония", "Macedonia"), "MMR": ("Бирма", "Burma"), "NLD": ("Голландия", "Holland"),
    "PRK": ("Северная Корея", "Korea DPR"), "SWZ": ("Свазиленд", "Swaziland"), "TKM": ("Туркмения",),
    "TLS": ("Тимор-Лешти", "East Timor"), "TUR": ("Türkiye",), "USA": ("Соединённые Штаты", "Америка", "USA", "United States of America"),
    "ZAF": ("Южно-Африканская Республика", "Южная Африка"),
}
ISO3_TO_RU: dict[str, str] = {iso: ru for iso, (ru, _) in COUNTRY_NAMES.items()}
ISO3_TO_EN: dict[str, str] = {iso: en for iso, (_, en) in COUNTRY_NAMES.items()}
