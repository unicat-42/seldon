"""Постсоветские страны и их коды в разных базах.

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
