"""EPR (Ethnic Power Relations, ETH Zürich): доступ этнических групп к государственной
власти по годам, и TEK (Transborder Ethnic Kin) — какие группы в разных странах родственны.
Скачивание: `download epr` (прямые CSV с icr.ethz.ch, без регистрации).

EPR Core: строка — группа × период [`from`, `to`] × статус. Статус кодируется на 1 января:
    MONOPOLY, DOMINANT             группа правит одна;
    SENIOR PARTNER, JUNIOR PARTNER делит власть;
    POWERLESS                      представителей во власти нет, но и гонений нет;
    DISCRIMINATED                  целенаправленная дискриминация государством;
    SELF-EXCLUSION                 группа контролирует территорию и не признаёт центр
                                   (Абхазия, Донбасс после 2014);
    IRRELEVANT, STATE COLLAPSE     этничность политически не значима / государства нет.
`size` — доля группы в населении страны. Коды стран — Gleditsch–Ward, как в UCDP.

Версия: официально на странице объявлен релиз 2021 (вышел 8.06.2021, данные по 2021);
файлы 2023 года (данные по 2023) лежат на том же сервере по той же схеме имён, но на
странице не объявлены. Берём 2023, контрольные суммы закреплены в `download.py`.

Ограничение: периоды размечены задним числом (переход статуса русских в Украине записан
одной датой 2015 в релизе, вышедшем годы спустя). Лаг публикации закрывает дату выхода
года, но не то, что эксперты знали исход.
"""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW

EPR_VERSION = "2023"
EPR_FILE = DATA_RAW / "epr" / f"EPR-{EPR_VERSION}.csv"
TEK_FILE = DATA_RAW / "epr" / f"TEK-{EPR_VERSION}.csv"
EPR_LAST_YEAR = int(EPR_VERSION)  # хроника кончается здесь: дальше — пропуск, а не «без изменений»

# ступени доступа к власти; самоисключение — на одной ступени с «бесправными» (не во власти)
POWER_RANK = {"MONOPOLY": 6, "DOMINANT": 5, "SENIOR PARTNER": 4, "JUNIOR PARTNER": 3,
              "POWERLESS": 2, "SELF-EXCLUSION": 2, "DISCRIMINATED": 1}
EXCLUDED = {"POWERLESS", "SELF-EXCLUSION", "DISCRIMINATED"}
INCLUDED_MIN_RANK = 3


def load_epr_group_years(year_from: int = 1980, year_to: int = EPR_LAST_YEAR) -> pd.DataFrame:
    """Группа × год: gw, year, gwgroupid, size, status, rank (NaN у IRRELEVANT / STATE COLLAPSE)."""
    e = pd.read_csv(EPR_FILE, usecols=["gwid", "from", "to", "gwgroupid", "size", "status"])
    e["year"] = [range(max(a, year_from), min(b, year_to) + 1) for a, b in zip(e["from"], e["to"])]
    e = e.explode("year").dropna(subset=["year"])
    e["year"] = e.year.astype(int)
    e["rank"] = e.status.map(POWER_RANK)
    return e.rename(columns={"gwid": "gw"})[["gw", "year", "gwgroupid", "size", "status", "rank"]].reset_index(drop=True)


def load_tek() -> pd.DataFrame:
    """Трансграничная родня: tekid (одна «семья» групп), gw, gwgroupid."""
    t = pd.read_csv(TEK_FILE, usecols=["tekid", "gwid", "gwgroupid"])
    return t.rename(columns={"gwid": "gw"}).drop_duplicates()
