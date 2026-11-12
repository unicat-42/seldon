"""Список независимых государств Gleditsch & Ward (файл iisystem.dat). Скачивание: `download gw`."""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW
from seldon.sources.ucdp import YEAR_FROM, YEAR_TO

GW_FILE = DATA_RAW / "gw" / "iisystem.dat"
GW_LIST_END_YEAR = 2017  # последняя дата в iisystem.dat — «ещё существует»


def load_states(year_from: int = YEAR_FROM, year_to: int = YEAR_TO) -> pd.DataFrame:
    """Государства Gleditsch–Ward: gw, название, год начала и конца существования."""
    rows = []
    for line in GW_FILE.read_text(encoding="latin-1").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        gw, _abbr, name, start, end = parts[:5]
        start_y, end_y = int(start[-4:]), int(end[-4:])
        if end_y >= GW_LIST_END_YEAR:
            end_y = year_to
        rows.append({"gw": int(gw), "gw_name": name, "start_year": start_y, "end_year": end_y})
    st = pd.DataFrame(rows)
    st = st[(st.end_year >= year_from) & (st.start_year <= year_to)]
    # Одна страна может входить несколько раз (Балтия: 1918–1940 и с 1991) — берём последний отрезок
    return st.sort_values(["gw", "start_year"]).groupby("gw", as_index=False).last()
