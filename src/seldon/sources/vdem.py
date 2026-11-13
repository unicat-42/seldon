"""V-Dem (Varieties of Democracy), Country-Year Full+Others, релиз v16 (март 2026, данные
по 2025 год). Скачивание: `download vdem` — файл `vdem.RData` из официального R-пакета
института на GitHub (vdeminstitute/vdemdata), ссылка закреплена на коммите релиза.

Берём только индексы (все — от экспертов, шкала 0…1, кроме `v2cacamps`):
    v2x_polyarchy  электоральная демократия;
    v2x_libdem     либеральная демократия;
    v2cacamps      политическая поляризация («общество расколото на враждебные лагеря»),
                   модельная шкала примерно −3…+4, больше — сильнее раскол;
    v2x_clphy      свобода от политических убийств и пыток (1 — насилия нет).

Для исследования «идеология и партия у руля» (`research/party_features.py`) — ещё `PARTY_COLS`:
    v2exl_legitideol       насколько власть оправдывает себя идеологией (модельная шкала ≈ −3…+3);
    v2exl_legitideolcr_0…4 какая это идеология — доля экспертов, отметивших: 0 националистическая,
                           1 социалистическая / коммунистическая, 2 консервативно-реставрационная,
                           3 сепаратистская / автономистская, 4 религиозная;
    v2exl_legitlead        опора на личность лидера (персонализм);
    v2exl_legitperf        опора на результаты (рост, порядок, безопасность);
    v2exnamhos, v2exparhos имя главы государства на 31 декабря и его партия;
    v2regidnr              номер политического режима (меняется при смене режима);
    v2xps_party            индекс институционализации партий, 0…1.
ВНИМАНИЕ: блок легитимации `v2exl_legit*` в релизе v16 размечен только по 2023 год
(`LEGIT_LAST_YEAR`): за 2024 и 2025 — пропуски. Имя лидера, партия и номер режима — по 2025.

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
VDEM_COMMIT = "f4dd26922e658442524dfd954bf14f7ebe622d5d"  # «v16 release», 17.03.2026 (vdeminstitute/vdemdata)
VDEM_FILE = DATA_RAW / "vdem" / "vdem.RData"
VDEM_COLS = ["v2x_polyarchy", "v2x_libdem", "v2cacamps", "v2x_clphy"]
SERBIA_SINCE = 2006  # COW 345 — Югославия до 2006 (GW 345) и Сербия после (GW 340)
LEGIT = ["v2exl_legitideol", "v2exl_legitlead", "v2exl_legitperf"]
LEGIT_CHARACTER = [f"v2exl_legitideolcr_{i}" for i in range(5)]
PARTY_COLS = [*LEGIT, *LEGIT_CHARACTER, "v2exnamhos", "v2exparhos", "v2regidnr", "v2xps_party", "v2x_polyarchy"]
LEGIT_LAST_YEAR = 2023  # последний год, за который в v16 размечен блок легитимации
VDEM_LAST_YEAR = 2025


def cow_to_gw(cow: pd.Series, year: pd.Series) -> pd.Series:
    """Коды COW в редакции V-Dem (со своими кодами государств-преемников, как в Polity) →
    Gleditsch–Ward; COW 345 — Югославия до 2006 года и Сербия (GW 340) после."""
    cow = cow.astype(int)
    gw = cow.map(lambda c: POLITY_TO_GW.get(c, c))
    return gw.where(~((cow == SERBIA_COW) & (year >= SERBIA_SINCE)), SERBIA_GW)


def load_vdem(year_from: int = 1980, cols: list[str] = VDEM_COLS) -> pd.DataFrame:
    """Страна-годы V-Dem в кодах Gleditsch–Ward: gw, year и колонки `cols` (по умолчанию
    `VDEM_COLS`). Территории без кода COW (Палестина, Сомалиленд, Гонконг, Занзибар) выпадают."""
    import pyreadr  # тяжёлый импорт; нужен только здесь

    v = pyreadr.read_r(str(VDEM_FILE))["vdem"][["COWcode", "year", *cols]]
    v = v[v.COWcode.notna() & (v.year >= year_from)].copy()
    v["year"] = v.year.astype(int)
    v["gw"] = cow_to_gw(v.pop("COWcode"), v.year)
    v = v.sort_values(["gw", "year"]).drop_duplicates(["gw", "year"], keep="last")
    return v[["gw", "year", *cols]].reset_index(drop=True)
