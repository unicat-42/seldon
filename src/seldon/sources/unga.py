"""Идеальные точки стран по голосованиям в Генеральной Ассамблее ООН (Bailey, Strezhnev,
Voeten 2017, «Estimating Dynamic State Preferences from United Nations Voting Data»).
Скачивание: `download unga` — Harvard Dataverse, hdl:1902.1/12379, файл релиза июля 2026.

Одна шкала на всю историю: чем больше значение, тем ближе голосования страны к США и
западным демократиям. Год — календарный год сессии (сессия открывается в сентябре).

Ограничение (как у V-Dem): оценки пересчитываются целиком при каждом релизе; мы читаем
релиз 2026 года, то есть точку за 2005 год — в редакции 2026-го. Лаг публикации
(`core/temporal.py`, `unga_ideal`) закрывает дату выхода, но не ревизии.
"""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW
from seldon.core.entities import cow_year_to_gw

UNGA_FILE = "IdealpointestimatesFP_2026FP.csv"
UNGA_DATAVERSE_ID = 14098429
UNGA_LAST_YEAR = 2025  # последняя сессия в релизе


def load_unga(year_from: int = 1980) -> pd.DataFrame:
    """Страна-годы: gw, year, ideal (апостериорное среднее идеальной точки)."""
    u = pd.read_csv(DATA_RAW / "unga" / UNGA_FILE, usecols=["ccode", "year", "IdealPointFP"])
    u = u[u.year >= year_from].copy()
    u["gw"] = cow_year_to_gw(u.ccode, u.year)
    u = u.rename(columns={"IdealPointFP": "ideal"})
    return u[["gw", "year", "ideal"]].drop_duplicates(["gw", "year"]).reset_index(drop=True)
