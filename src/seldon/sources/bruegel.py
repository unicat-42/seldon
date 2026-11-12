"""REER — реальный эффективный курс (Bruegel, Zsolt Darvas, «Real effective exchange rates
for 178 countries»). Скачивание: `download bruegel` — архив релиза 11 октября 2026.

REER — курс валюты к корзине валют торговых партнёров с поправкой на разницу инфляции.
Рост индекса = реальное УКРЕПЛЕНИЕ своей валюты (у МВФ и в нашем `rer_dev` знак обратный).
Листы файла:
    REER_MONTHLY_120   корзина из 120 партнёров, с 1993 года — основной ряд: в корзине есть
                       Россия, Казахстан, Китай (для постсоветских стран это главное);
    REER_MONTHLY_51    корзина из 51 партнёра, с 1960 года — только для истории до 1993
                       (России в корзине нет).
Покрытие шире, чем у МВФ (поток EER — 94 страны, из постсоветских 6) и ВБ (`PX.REX.REER` —
те же 94): здесь 178 стран, из постсоветских нет только Туркмении (Узбекистан — с 2007-12,
Таджикистан — с 2000).

Ограничения: веса корзины фиксированные (торговля 1998–2003 годов и позже), то есть ряд до
1998 года взвешен «будущей» торговлей; ИПЦ и курсы — в редакции 2026 года (ревизии не
покрыты). Свежие месяцы автор досчитывает по прогнозу ИПЦ; мы берём ряд с лагом
(`bruegel_reer` в `core/temporal.py`).
"""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW
from seldon.core.entities import ISO2_TO_ISO3

BRUEGEL_VERSION = "ver11Oct2026"
BRUEGEL_ZIP = f"REER_database_{BRUEGEL_VERSION}.zip"
BRUEGEL_FILE = f"REER_database_{BRUEGEL_VERSION}.xls"
BRUEGEL_URL = f"https://www.bruegel.org/sites/default/files/2026-10/{BRUEGEL_ZIP}"
# NEER — номинальный эффективный курс (та же корзина, без цен): нужны только курсы валют,
# поэтому он известен сразу; по нему REER «дотягивается» до текущего месяца (`struct_features.py`)
SHEETS = {"reer120": "REER_MONTHLY_120", "reer51": "REER_MONTHLY_51",
          "neer120": "NEER_MONTHLY_120", "neer51": "NEER_MONTHLY_51"}


def _sheet(name: str, col: str) -> pd.DataFrame:
    d = pd.read_excel(DATA_RAW / "bruegel" / BRUEGEL_FILE, sheet_name=name, header=0)
    d = d.rename(columns={d.columns[0]: "period"})
    d = d[d.period.astype(str).str.fullmatch(r"\d{4}M\d{2}")]
    long = d.melt(id_vars="period", var_name="code", value_name=col)
    long["iso3"] = long.code.str.rsplit("_", n=1).str[-1].map(ISO2_TO_ISO3)  # агрегаты (EA12) и Антилы выпадают
    long[col] = pd.to_numeric(long[col], errors="coerce")
    long = long[long.iso3.notna() & (long[col] > 0)]
    long["month"] = pd.PeriodIndex(long.period.str.replace("M", "-"), freq="M")
    return long[["iso3", "month", col]]


def load_reer() -> pd.DataFrame:
    """Месячные REER и NEER: iso3, month (`pd.Period`), reer120, reer51, neer120, neer51
    (индексы; рост = укрепление)."""
    out = None
    for col, sheet in SHEETS.items():
        d = _sheet(sheet, col)
        out = d if out is None else out.merge(d, on=["iso3", "month"], how="outer")
    return out.sort_values(["iso3", "month"]).reset_index(drop=True)
