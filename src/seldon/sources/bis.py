"""BIS (Банк международных расчётов): кредит нефинансовому сектору. Скачивание: `download bis`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_RAW

BIS_CREDIT_FILE = DATA_RAW / "bis" / "WS_TC_csv_flat.csv"

# BIS: ISO2 заёмщика → ISO3; агрегаты (4T, 5A, 5R, G2, XM) пропускаем
BIS_ISO = {
    "AR": "ARG", "AT": "AUT", "AU": "AUS", "BE": "BEL", "BR": "BRA", "CA": "CAN", "CH": "CHE",
    "CL": "CHL", "CN": "CHN", "CO": "COL", "CZ": "CZE", "DE": "DEU", "DK": "DNK", "ES": "ESP",
    "FI": "FIN", "FR": "FRA", "GB": "GBR", "GR": "GRC", "HK": "HKG", "HU": "HUN", "ID": "IDN",
    "IE": "IRL", "IL": "ISR", "IN": "IND", "IT": "ITA", "JP": "JPN", "KR": "KOR", "LU": "LUX",
    "MX": "MEX", "MY": "MYS", "NL": "NLD", "NO": "NOR", "NZ": "NZL", "PL": "POL", "PT": "PRT",
    "RU": "RUS", "SA": "SAU", "SE": "SWE", "SG": "SGP", "TH": "THA", "TR": "TUR", "US": "USA",
    "ZA": "ZAF",
}


def load_bis_credit() -> pd.DataFrame:
    """Кредит банков частному нефинансовому сектору, % ВВП, значение 4-го квартала → год.
    Где банковского ряда нет — кредит всех секторов (помечено `credit_src`)."""
    d = pd.read_csv(BIS_CREDIT_FILE, low_memory=False)
    d.columns = [c.split(":")[0] for c in d.columns]
    for c in ["BORROWERS_CTY", "TC_BORROWERS", "TC_LENDERS", "VALUATION", "UNIT_TYPE", "TC_ADJUST"]:
        d[c] = d[c].astype(str).str.split(":").str[0]
    d = d[(d.TC_BORROWERS == "P") & (d.UNIT_TYPE == "770") & (d.TC_ADJUST == "A")
          & d.TIME_PERIOD.astype(str).str.endswith("Q4") & d.OBS_VALUE.notna()]
    d["iso3"] = d.BORROWERS_CTY.map(BIS_ISO)
    d = d[d.iso3.notna()].copy()
    d["year"] = d.TIME_PERIOD.str[:4].astype(int)
    banks = d[d.TC_LENDERS == "B"].set_index(["iso3", "year"]).OBS_VALUE.rename("credit_banks")
    allsec = d[(d.TC_LENDERS == "A") & (d.VALUATION == "M")].set_index(["iso3", "year"]).OBS_VALUE.rename("credit_all")
    out = pd.concat([banks, allsec], axis=1).reset_index()
    out["credit_gdp"] = out.credit_banks.fillna(out.credit_all)
    out["credit_src"] = np.where(out.credit_banks.notna(), "banks", "all")
    return out
