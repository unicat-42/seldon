"""Global Leader Ideologies (Herre 2023, «Identifying Ideologues: A Global Dataset on Political
Leaders, 1945–2020», British Journal of Political Science). Скачивание: `download gli` — CSV из
репозитория автора на GitHub (bastianherre/global-leader-ideologies), ссылка закреплена на коммите.

Экономическая идеология главы исполнительной власти по страна-годам: `leftist` / `centrist` /
`rightist` / `none` (неидеологический лидер); `no information` и `not applicable` — пропуски.
Берём `leader` — того из пары «глава государства / глава правительства», у кого реальная власть
(так его определяет автор), а не формального главу правительства: в России 2012–2019 это Путин,
а не Медведев.

Ограничения: база собрана один раз задним числом и кончается 2020 годом (`GLI_LAST_YEAR`) —
дальше не ноль и не «нет идеологии», а пропуск (инвариант 6). Значение за год — лидер на
31 декабря.
"""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW
from seldon.sources.vdem import cow_to_gw

GLI_COMMIT = "7d0d5bb189706ecfbf90346529713a912cd519d2"  # main на 02.08.2023, последний коммит репозитория
GLI_FILE = "global_leader_ideologies.csv"
GLI_URL = f"https://raw.githubusercontent.com/bastianherre/global-leader-ideologies/{GLI_COMMIT}/{GLI_FILE}"
GLI_SHA256 = "c9e2752bd304bb59666ef8fd7a56ea600d51e4c38e5cbaf0cd07757b5f7f99c3"
GLI_LAST_YEAR = 2020
CAMPS = ("leftist", "centrist", "rightist", "none")


def load_gli() -> pd.DataFrame:
    """Страна-годы 1945–2020: gw, year, leader (имя), camp (один из `CAMPS` или пропуск)."""
    g = pd.read_csv(DATA_RAW / "gli" / GLI_FILE, usecols=["country_code_cow", "year", "leader", "leader_ideology"])
    g = g[g.country_code_cow.notna()].copy()
    g["year"] = g.year.astype(int)
    g["gw"] = cow_to_gw(g.pop("country_code_cow"), g.year)
    g["camp"] = g.pop("leader_ideology").where(lambda s: s.isin(CAMPS))
    g = g.sort_values(["gw", "year"]).drop_duplicates(["gw", "year"], keep="last")
    return g[["gw", "year", "leader", "camp"]].reset_index(drop=True)
