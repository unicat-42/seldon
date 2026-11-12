"""V-Dem (Varieties of Democracy), Country-Year Full+Others, релиз v16 (март 2026, данные
по 2025 год). Скачивание: `download vdem` — файл `vdem.RData` из официального R-пакета
института на GitHub (vdeminstitute/vdemdata), ссылка закреплена на коммите релиза.

Берём только индексы (все — от экспертов, шкала 0…1, кроме `v2cacamps`):
    v2x_polyarchy  электоральная демократия;
    v2x_libdem     либеральная демократия;
    v2cacamps      политическая поляризация («общество расколото на враждебные лагеря»),
                   модельная шкала примерно −3…+4, больше — сильнее раскол;
    v2x_clphy      свобода от политических убийств и пыток (1 — насилия нет).

Ограничение: V-Dem каждый год пересчитывает ВСЮ историю своей измерительной моделью. Мы
читаем релиз v16 целиком, то есть значения за, скажем, 2005 год — в редакции 2026 года,
а не той, что была известна в 2006. Лаг публикации (`core/temporal.py`) закрывает дату
выхода, но не ревизии; прошлые релизы (V10…V15) лежат в том же репозитории по тегам.
"""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW
from seldon.core.entities import POLITY_TO_GW, SERBIA_COW, SERBIA_GW

VDEM_VERSION = "16"
VDEM_FILE = DATA_RAW / "vdem" / "vdem.RData"
VDEM_COLS = ["v2x_polyarchy", "v2x_libdem", "v2cacamps", "v2x_clphy"]
SERBIA_SINCE = 2006  # COW 345 — Югославия до 2006 (GW 345) и Сербия после (GW 340)


def load_vdem(year_from: int = 1980) -> pd.DataFrame:
    """Страна-годы V-Dem в кодах Gleditsch–Ward: gw, year и колонки `VDEM_COLS`.
    Территории без кода COW (Палестина, Сомалиленд, Гонконг, Занзибар) выпадают."""
    import pyreadr  # тяжёлый импорт; нужен только здесь

    v = pyreadr.read_r(str(VDEM_FILE))["vdem"][["COWcode", "year", *VDEM_COLS]]
    v = v[v.COWcode.notna() & (v.year >= year_from)].copy()
    v["year"] = v.year.astype(int)
    cow = v.pop("COWcode").astype(int)
    v["gw"] = cow.map(lambda c: POLITY_TO_GW.get(c, c))
    v.loc[(cow == SERBIA_COW) & (v.year >= SERBIA_SINCE), "gw"] = SERBIA_GW
    v = v.sort_values(["gw", "year"]).drop_duplicates(["gw", "year"], keep="last")
    return v[["gw", "year", *VDEM_COLS]].reset_index(drop=True)
