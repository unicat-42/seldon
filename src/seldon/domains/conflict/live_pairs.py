"""Живая хроника столкновений между государствами — из событий UCDP (чекпоинт 4.2).

Замена хронике споров COW MID, которая кончается 2014 годом. Берём государственные события
(`type_of_violence == 1`) GED и Candidate Events, включая субпороговые (1–24 погибших), и
раскладываем по направленным парам A ← B. Четыре вида (`KINDS`):

    direct    обе стороны события — правительства (коды `gwnoa` / `gwnob`): Индия–Пакистан,
              Киргизия–Таджикистан, Иран–Израиль, Россия–Украина (в GED — с 2014 года);
    proxy     событие конфликта, в котором A и B стоят по разные стороны по списку сторон ACD
              (основные или помогающие войсками), но само событие — не «правительство против
              правительства»: Армения ↔ Азербайджан (Карабах), Россия ↔ Украина в Донбассе
              2014–2021. Список сторон берётся с лагом `acd_lag` лет (список за год Y выходит
              после его окончания);
    inc_in    на территории A воюет правительство B, а правительство A в событии не участвует:
              Израиль в Ливане («Хезболла»), Турция в Ираке (РПК);
    inc_out   зеркально: правительство A воюет на территории B.

Одно событие для одной пары считается один раз (`direct` важнее `proxy`).

`pair_events`     пара × месяц × вид → погибшие и число событий;
`annual_features` признаки пары на строке года T (прогноз «на 1 января T+1»): события по
                  месяц `декабрь T − lag` включительно (`ucdp_events`: декабрьский файл выходит
                  20 января);
`country_month`   свёртка на страну × месяц для месячной модели войн.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon.core.temporal import monthly_lag

KINDS = ("direct", "proxy", "inc_in", "inc_out")
EVENT_LAG = monthly_lag("ucdp_events")
WINDOW_YEARS = 15  # как окно `mid_force`
LIVE_FEATURES = ["live_years15", "live_ld3", "live_ev12"]
COUNTRY_COLS = ["lv_dir_d", "lv_px_d", "lv_in_d", "lv_out_d", "lv_ev"]


def _codes(cell) -> list[int]:
    return [int(float(x)) for x in str(cell).replace(";", ",").split(",") if x.strip() not in ("", "nan", "None")]


def month_ordinal(year, month):
    """Номер месяца по порядку (как у `pd.Period(..., freq="M").ordinal`)."""
    return (np.asarray(year, dtype=int) - 1970) * 12 + np.asarray(month, dtype=int) - 1


def pair_events(events: pd.DataFrame, opp: pd.DataFrame, acd_lag: int = 1) -> pd.DataFrame:
    """Пара × месяц × вид: a, b, o (номер месяца), kind, deaths, events.
    `events` — колонки gw (место), year, month, type_of_violence, best, conflict_new_id и, если
    есть, gwnoa / gwnob. `opp` — противники по ACD (`dyads.opponents`): conflict_id, year, a, b."""
    e = events[events.type_of_violence == 1].reset_index(drop=True)
    e = e.assign(eid=np.arange(len(e)), o=month_ordinal(e.year, e.month))
    rows = []
    if "gwnoa" in e and "gwnob" in e:
        st = e[e.gwnoa.notna() | e.gwnob.notna()]
        for eid, loc, ga, gb in zip(st.eid, st.gw, st.gwnoa, st.gwnob):
            sa, sb = _codes(ga), _codes(gb)
            for x in sa:
                for y in sb:
                    if x != y:
                        rows += [(eid, x, y, "direct"), (eid, y, x, "direct")]
            sides = set(sa) | set(sb)
            if loc not in sides:
                for s in sides:
                    rows += [(eid, loc, s, "inc_in"), (eid, s, loc, "inc_out")]
    d = pd.DataFrame(rows, columns=["eid", "a", "b", "kind"])
    o = opp[["conflict_id", "year", "a", "b"]].drop_duplicates().assign(year=lambda x: x.year + acd_lag)
    px = e[["eid", "conflict_new_id", "year"]].merge(o, left_on=["conflict_new_id", "year"], right_on=["conflict_id", "year"])
    px = px[["eid", "a", "b"]].drop_duplicates().assign(kind="proxy")
    px = px.merge(d.loc[d.kind == "direct", ["eid", "a", "b"]].assign(_d=1), on=["eid", "a", "b"], how="left")
    d = pd.concat([d, px[px._d.isna()].drop(columns="_d")], ignore_index=True).drop_duplicates()
    d = d.merge(e[["eid", "o", "best"]], on="eid")
    out = d.groupby(["a", "b", "o", "kind"], as_index=False).agg(deaths=("best", "sum"), events=("best", "size"))
    return out.astype({"a": int, "b": int})


def annual_features(pe: pd.DataFrame, years, lag: int = EVENT_LAG) -> pd.DataFrame:
    """Признаки пары на строке года T — только по событиям не позже месяца `декабрь T − lag`:
        live_years15  log(1 + число 12-месячных отрезков из последних 15, где у пары было хоть
                      одно событие любого вида) — живой аналог `mid_force`;
        live_ld3      log(1 + погибшие в событиях пары за последние 36 месяцев);
        live_ev12     log(1 + число событий пары за последние 12 месяцев).
    Колонки: a, b, year и `LIVE_FEATURES`; пары без событий в окне в таблицу не попадают."""
    out = []
    for T in years:
        c = month_ordinal(T, 12) - lag
        x = pe[(pe.o <= c) & (pe.o > c - 12 * WINDOW_YEARS)]
        if x.empty:
            continue
        k = (c - x.o) // 12
        g = x.assign(k=k, d3=x.deaths.where(k < 3, 0), e12=x.events.where(k < 1, 0)).groupby(["a", "b"])
        f = g.agg(n=("k", "nunique"), d3=("d3", "sum"), e12=("e12", "sum")).reset_index()
        out.append(pd.DataFrame({"a": f.a, "b": f.b, "year": T, "live_years15": np.log1p(f.n),
                                 "live_ld3": np.log1p(f.d3), "live_ev12": np.log1p(f.e12)}))
    cols = ["a", "b", "year", *LIVE_FEATURES]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def country_month(pe: pd.DataFrame) -> pd.DataFrame:
    """Страна × месяц: gw, o и `COUNTRY_COLS` — погибшие в прямых столкновениях с другими
    государствами, в прокси-столкновениях, от чужой армии на своей территории, от своей армии
    на чужой и число событий всех видов."""
    w = pe.pivot_table(index=["a", "o"], columns="kind", values="deaths", aggfunc="sum", fill_value=0)
    for k in KINDS:
        if k not in w:
            w[k] = 0
    w["lv_ev"] = pe.groupby(["a", "o"]).events.sum()
    w = w.rename(columns={"direct": "lv_dir_d", "proxy": "lv_px_d", "inc_in": "lv_in_d", "inc_out": "lv_out_d"}).reset_index()
    return w.rename(columns={"a": "gw"})[["gw", "o", *COUNTRY_COLS]]
