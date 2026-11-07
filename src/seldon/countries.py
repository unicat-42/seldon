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
