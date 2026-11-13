"""V-Party (Varieties of Party Identity and Organization, институт V-Dem), релиз V2: экспертная
разметка партий в годы парламентских выборов, 1900–2019 (мы берём с 1970). Скачивание:
`download vparty` — файл `vparty.RData` официального R-пакета института на GitHub
(vdeminstitute/vdemdata), тот же закреплённый коммит, что и у V-Dem v16.

Строка исходника — партия × год выборов. Берём:
    v2pagovsup        отношение к правительству ПОСЛЕ этих выборов: 0 — старший партнёр (партия
                      главы правительства), 1 — младший партнёр, 2 — поддерживает, не входя,
                      3 — оппозиция;
    v2paseatshare     доля мест в нижней палате, %;
    v2xpa_popul       индекс популизма, 0…1 (антиэлитизм + «народ как единое целое»);
    v2xpa_antiplural  индекс антиплюрализма, 0…1 (неприятие оппонентов, прав меньшинств, выборов);
    v2pariglef        экономическая шкала левые–правые, модельная, примерно −4…+4 (больше — правее).

`load_ruling()` сворачивает это в одну правящую партию на страна-год выборов: старший партнёр
правительства с наибольшей долей мест; если старшего нет — младший; если нет и его — пропуск
(Россия-1993: «партии власти» в Думе нет).

Ограничения: база кончается 2019 годом (`VPARTY_LAST_YEAR`), после него — пропуск, а не «нет
популизма» (инвариант 6); данные есть только в годы выборов — между выборами правящая партия
«тянется» в признаках (`research/party_features.py`), но не дальше конца базы.
"""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW
from seldon.sources.vdem import VDEM_COMMIT, cow_to_gw

VPARTY_FILE = DATA_RAW / "vparty" / "vparty.RData"
VPARTY_URL = f"https://raw.githubusercontent.com/vdeminstitute/vdemdata/{VDEM_COMMIT}/data/vparty.RData"
VPARTY_SHA256 = "a4bde75e84c01ddfe5fc1fe8b32a87df2c13158eb658769f8a4ceb8f3ca5b964"
VPARTY_LAST_YEAR = 2019
VPARTY_FIRST_YEAR = 1970
SCORES = ["v2xpa_popul", "v2xpa_antiplural", "v2pariglef"]


def load_vparty() -> pd.DataFrame:
    """Партия × год выборов: gw, year, party, govsup, seatshare и колонки `SCORES`."""
    import pyreadr  # тяжёлый импорт; нужен только здесь

    v = pyreadr.read_r(str(VPARTY_FILE))["vparty"][["COWcode", "year", "v2paenname", "v2pagovsup", "v2paseatshare", *SCORES]]
    v = v[v.COWcode.notna() & (v.year >= VPARTY_FIRST_YEAR)].copy()
    v["year"] = v.year.astype(int)
    v["gw"] = cow_to_gw(v.pop("COWcode"), v.year)
    return v.rename(columns={"v2paenname": "party", "v2pagovsup": "govsup", "v2paseatshare": "seatshare"}).reset_index(drop=True)


def ruling_party(v: pd.DataFrame) -> pd.DataFrame:
    """Одна правящая партия на страна-год выборов (чистая функция от таблицы `load_vparty`):
    gw, year, party и `SCORES`. Страна-год выборов без партии в правительстве — в таблице
    остаётся, с пропусками (выборы были, правящей партии нет)."""
    gov = v[v.govsup.isin([0, 1]) & v[SCORES].notna().all(axis=1)]
    gov = gov.sort_values(["gw", "year", "govsup", "seatshare"], ascending=[True, True, True, False], na_position="last")
    top = gov.drop_duplicates(["gw", "year"], keep="first")[["gw", "year", "party", *SCORES]]
    return v[["gw", "year"]].drop_duplicates().merge(top, on=["gw", "year"], how="left").sort_values(["gw", "year"]).reset_index(drop=True)


def load_ruling() -> pd.DataFrame:
    return ruling_party(load_vparty())
