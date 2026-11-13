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
# Прошлые релизы (исследование 14.11.2026, `research/democ_fx.py`): год выхода → (версия, коммит
# файла data/vdem.RData в vdeminstitute/vdemdata, sha256). Релиз года R выходит в марте (v11.1 —
# 1 апреля) и несёт данные по R − 1. Нужны, чтобы проверить признак в той редакции, которая
# существовала на дату прогноза, а не в сегодняшней. `download vdem_vintages`.
VDEM_RELEASES: dict[int, tuple[str, str, str | None]] = {
    2020: ("10", "6dd1327c632f999019c7a547e906f91fac453613", "2fa46c56154f981833626e0569ecafbb03b5265f3e8478426c3a5b06a242fbb5"),
    2021: ("11.1", "cbaa6d59d74ff2e6f3f3efff8597ee9fbe9e2ae2", "4a171101b830952df7e27675a3a99c682187f6fd094eeb29576c3a7366cb653c"),
    2022: ("12", "cf87496460cd6eff8ef534b9d7f39e20dc558d23", "0b6da7f13970ef882596d7fb88262a2c1a0c4cb7da06e3c3a7026202f2cd3dc0"),
    2023: ("13", "8b8fd298a5352e57bac113cdcf1ca9c616958c3d", "1e8749eed57bf4ce34356123dcc074c1e5be27887abf1803ff605bbe187e68a8"),
    2024: ("14", "93f2c195505274842dfabd966447ef8c41340328", "946a92b4002112e26e6ab0429d3b2476843c86f3666bc3b05ab9ec756fd1720a"),
    2025: ("15", "c7836967c508fd463ad4ef866529933938e30cb0", "4b715190a1e351a81daf11cbe6f9f77e67d8914f0ed612b93e2d8fba5523e540"),
}
VINTAGE_DIR = DATA_RAW / "vdem_vintages"
LAST_RELEASE_YEAR = 2026  # v16 — боевой файл `VDEM_FILE`


def vintage_file(release_year: int) -> str:
    return f"vdem_v{VDEM_RELEASES[release_year][0]}.RData"


def vintage_url(release_year: int) -> str:
    return f"https://raw.githubusercontent.com/vdeminstitute/vdemdata/{VDEM_RELEASES[release_year][1]}/data/vdem.RData"


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


def load_vdem(year_from: int = 1980, cols: list[str] = VDEM_COLS, release_year: int | None = None) -> pd.DataFrame:
    """Страна-годы V-Dem в кодах Gleditsch–Ward: gw, year и колонки `cols` (по умолчанию
    `VDEM_COLS`). Территории без кода COW (Палестина, Сомалиленд, Гонконг, Занзибар) выпадают.
    `release_year` — прошлый релиз из `VDEM_RELEASES` (по умолчанию — боевой v16)."""
    import pyreadr  # тяжёлый импорт; нужен только здесь

    path = VDEM_FILE if release_year in (None, LAST_RELEASE_YEAR) else VINTAGE_DIR / vintage_file(release_year)
    v = pyreadr.read_r(str(path))["vdem"][["COWcode", "year", *cols]]
    v = v[v.COWcode.notna() & (v.year >= year_from)].copy()
    v["year"] = v.year.astype(int)
    v["gw"] = cow_to_gw(v.pop("COWcode"), v.year)
    v = v.sort_values(["gw", "year"]).drop_duplicates(["gw", "year"], keep="last")
    return v[["gw", "year", *cols]].reset_index(drop=True)
