"""Панель «страна × период» по UCDP GED: погибшие, события и целевые переменные.

Запуск:
    uv run python -m seldon.panel

Результат:
    data/processed/panel_country_year.parquet   — страна × год, 1989–2024
    data/processed/panel_country_month.parquet  — страна × месяц, 1989-01 … 2024-12

Колонки (в месячной версии вместо `year` — `year`, `month` и `period` = первое число месяца):
    gw             код страны Gleditsch–Ward (как в UCDP)
    country        название (как в UCDP GED, иначе как у Gleditsch–Ward)
    deaths_best    сумма «лучшей оценки» погибших по всем типам насилия
    deaths_state   то же, но только государственные конфликты (type_of_violence == 1)
    events         число событий GED
    conflict25     1, если за период погибло ≥ 25 человек (порог UCDP для «вооружённого конфликта»)
    conflict1000   1, если ≥ 1000 (порог UCDP для «войны»)
    onset25        1, если conflict25 == 1 в этом периоде и == 0 в предыдущем («начало»)
    deaths_party   погибшие в государственных конфликтах, где страна — СТОРОНА (см. ниже),
                   где бы бои ни шли
    party25        1, если deaths_party ≥ 25
    war25          1, если conflict25 == 1 ИЛИ party25 == 1 — «страна воюет»: либо на её
                   территории, либо она сторона конфликта
    onset_war25    начало по war25
    war1000        1, если ≥ 1000 погибших на территории ИЛИ в конфликтах, где страна сторона
                   («война» по порогу UCDP)
    onset_war1000  начало по war1000

Как считаем:
  - `conflict25` — по стране, на территории которой произошли события (`country_id` в GED).
    Так Грузия 2008 и Карабах 2020 попадают в целевую (территория), хотя UCDP классифицирует
    их не как межгосударственные войны. Но Армения 2020 и Россия 2008/2014/2022 — нет.
  - `party25` закрывает этот пробел. Страна считается стороной конфликта-года (UCDP/PRIO
    ACD), если она (а) основная сторона (`gwno_a`/`gwno_b`) — где угодно, или (б) помогает
    войсками (`gwno_a_2nd`/`gwno_b_2nd`) И граничит со страной, где идёт конфликт
    (`gwno_loc`). Погибшие события (GED, `conflict_new_id` → ACD) приписываются всем таким
    сторонам. Правило (б) ловит Армению 2020, Россию 2008 и 2014, Узбекистан в Таджикистане
    1993, но не Балтию в Афганистане и не Россию в Сирии. Побочный эффект: Турция 2020
    (помогала Азербайджану, граничит с ним) тоже «воюет».
  - Соседи — Correlates of War Direct Contiguity 3.2: суша или море до 12 миль
    (`conttype` ≤ 2), по годам до 2016, дальше — как в 2016. Коды COW → Gleditsch–Ward:
    Германия 255 → 260, Йемен 679 → 678, Сербия 345 продублирована как 340.
  - Список стран — все независимые государства по Gleditsch & Ward (файл iisystem.dat),
    существовавшие хотя бы один день в 1989–2024. Строка «страна-период» есть только за
    периоды, когда государство существовало (Украина — с 1991, Южный Судан — с 2011).
    Записи с датой конца 31.12.2017 (последний год списка) считаем существующими по сей день.
  - В первый период существования страны `onset25` считаем от нуля (предыдущего периода
    нет, значит конфликта в нём «не было»).
"""

from __future__ import annotations

import sys

import pandas as pd

from seldon import DATA_PROCESSED, DATA_RAW

YEAR_FROM, YEAR_TO = 1989, 2024  # GED покрывает 1989–2024
GW_LIST_END_YEAR = 2017  # последняя дата в iisystem.dat — «ещё существует»

CONTIGUITY_FILE = DATA_RAW / "cow" / "DirectContiguity320" / "contdird.csv"
CONTIGUITY_MAX_TYPE = 2  # 1 — сухопутная граница, 2 — через море до 12 миль
CONTIGUITY_END_YEAR = 2016  # последний год в данных COW; дальше считаем границы неизменными
COW_TO_GW = {255: 260, 679: 678}  # Германия, Йемен


def load_neighbors() -> set[tuple[int, int, int]]:
    """Множество троек (страна, сосед, год): кто с кем граничит в каждом году 1989–2024."""
    c = pd.read_csv(CONTIGUITY_FILE, usecols=["state1no", "state2no", "year", "conttype"])
    c = c[(c.conttype <= CONTIGUITY_MAX_TYPE) & (c.year >= YEAR_FROM)]
    c[["state1no", "state2no"]] = c[["state1no", "state2no"]].replace(COW_TO_GW)
    serbia = c[(c.state1no == 345) | (c.state2no == 345)].replace({"state1no": {345: 340}, "state2no": {345: 340}})
    c = pd.concat([c, serbia])
    tail = c[c.year == CONTIGUITY_END_YEAR]
    c = pd.concat([c, *(tail.assign(year=y) for y in range(CONTIGUITY_END_YEAR + 1, YEAR_TO + 1))])
    return set(zip(c.state1no, c.state2no, c.year))


def _gw_list(cell) -> list[int]:
    return [int(float(x)) for x in str(cell).split(",") if x.strip() not in ("", "nan")]


def load_conflict_parties(neighbors: set[tuple[int, int, int]]) -> pd.DataFrame:
    """Для каждого конфликта-года (ACD): список стран-сторон по правилу из docstring."""
    cols = ["conflict_id", "year", "gwno_a", "gwno_b", "gwno_a_2nd", "gwno_b_2nd", "gwno_loc"]
    a = pd.read_csv(DATA_RAW / "ucdp" / "UcdpPrioConflict_v25_1.csv", usecols=cols)
    a = a[a.year >= YEAR_FROM]
    parties = []
    for r in a.itertuples(index=False):
        primary = set(_gw_list(r.gwno_a)) | set(_gw_list(r.gwno_b))
        secondary = set(_gw_list(r.gwno_a_2nd)) | set(_gw_list(r.gwno_b_2nd))
        locs = _gw_list(r.gwno_loc)
        helping_neighbors = {s for s in secondary if any((s, loc, r.year) in neighbors for loc in locs)}
        parties.append(sorted(primary | helping_neighbors))
    return pd.DataFrame({"conflict_id": a.conflict_id.values, "year": a.year.values, "parties": parties})


def party_deaths(ged: pd.DataFrame, parties: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Погибшие по странам-сторонам: событие приписывается каждой стороне своего конфликта-года."""
    g = ged.loc[ged.type_of_violence == 1, ["conflict_new_id", *keys, "best"]]
    g = g.merge(parties, left_on=["conflict_new_id", "year"], right_on=["conflict_id", "year"], how="inner")
    g = g.explode("parties").dropna(subset=["parties"])
    g["gw"] = g["parties"].astype(int)
    return g.groupby(["gw", *keys])["best"].sum().rename("deaths_party").reset_index()


def load_states() -> pd.DataFrame:
    """Государства Gleditsch–Ward: gw, название, год начала и конца существования."""
    rows = []
    for line in (DATA_RAW / "gw" / "iisystem.dat").read_text(encoding="latin-1").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        gw, _abbr, name, start, end = parts[:5]
        start_y, end_y = int(start[-4:]), int(end[-4:])
        if end_y >= GW_LIST_END_YEAR:
            end_y = YEAR_TO
        rows.append({"gw": int(gw), "gw_name": name, "start_year": start_y, "end_year": end_y})
    st = pd.DataFrame(rows)
    st = st[(st.end_year >= YEAR_FROM) & (st.start_year <= YEAR_TO)]
    # Одна страна может входить несколько раз (Балтия: 1918–1940 и с 1991) — берём последний отрезок
    return st.sort_values(["gw", "start_year"]).groupby("gw", as_index=False).last()


def load_ged() -> pd.DataFrame:
    cols = ["country", "country_id", "year", "date_start", "type_of_violence", "best", "conflict_new_id"]
    g = pd.read_csv(DATA_RAW / "ucdp" / "GEDEvent_v25_1.csv", usecols=cols)
    g["date_start"] = pd.to_datetime(g["date_start"])
    g["month"] = g["date_start"].dt.month
    g["deaths_state"] = g["best"].where(g["type_of_violence"] == 1, 0)
    return g.rename(columns={"country_id": "gw"})


def aggregate(ged: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        ged.groupby(keys)
        .agg(deaths_best=("best", "sum"), deaths_state=("deaths_state", "sum"), events=("best", "size"))
        .reset_index()
    )


def add_targets(panel: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    panel = panel.sort_values(["gw", *keys]).reset_index(drop=True)
    panel["conflict25"] = (panel["deaths_best"] >= 25).astype("int8")
    panel["conflict1000"] = (panel["deaths_best"] >= 1000).astype("int8")
    panel["party25"] = (panel["deaths_party"] >= 25).astype("int8")
    panel["war25"] = panel[["conflict25", "party25"]].max(axis=1).astype("int8")
    # «война» по UCDP: ≥1000 погибших — на территории или в конфликтах, где страна сторона
    panel["war1000"] = ((panel["deaths_best"] >= 1000) | (panel["deaths_party"] >= 1000)).astype("int8")
    for target, onset in (("conflict25", "onset25"), ("war25", "onset_war25"), ("war1000", "onset_war1000")):
        prev = panel.groupby("gw")[target].shift(1).fillna(0).astype("int8")
        panel[onset] = ((panel[target] == 1) & (prev == 0)).astype("int8")
    return panel


def build(ged: pd.DataFrame, states: pd.DataFrame, parties: pd.DataFrame, monthly: bool) -> pd.DataFrame:
    names = ged.drop_duplicates("gw").set_index("gw")["country"]
    states = states.assign(country=states["gw"].map(names).fillna(states["gw_name"]))

    if monthly:
        periods = pd.DataFrame({"period": pd.date_range(f"{YEAR_FROM}-01-01", f"{YEAR_TO}-12-01", freq="MS")})
        periods["year"], periods["month"] = periods.period.dt.year, periods.period.dt.month
        keys = ["year", "month"]
    else:
        periods = pd.DataFrame({"year": range(YEAR_FROM, YEAR_TO + 1)})
        keys = ["year"]

    grid = states[["gw", "country", "start_year", "end_year"]].merge(periods, how="cross")
    grid = grid[(grid.year >= grid.start_year) & (grid.year <= grid.end_year)].drop(columns=["start_year", "end_year"])

    agg = aggregate(ged, ["gw", *keys])
    panel = grid.merge(agg, on=["gw", *keys], how="left")
    panel = panel.merge(party_deaths(ged, parties, keys), on=["gw", *keys], how="left")
    for c in ["deaths_best", "deaths_state", "events", "deaths_party"]:
        panel[c] = panel[c].fillna(0).astype("int64")
    return add_targets(panel, keys)


def main() -> int:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    states, ged = load_states(), load_ged()
    parties = load_conflict_parties(load_neighbors())

    missing = sorted(set(ged.gw) - set(states.gw))
    if missing:
        lost = ged[ged.gw.isin(missing)]
        print(
            f"Внимание: {len(missing)} кодов GED нет в списке государств, их события выпадут "
            f"({len(lost)} событий, {lost.best.sum()} погибших): "
            + ", ".join(f"{n} ({g})" for g, n in lost.drop_duplicates('gw')[['gw','country']].values)
        )

    year = build(ged, states, parties, monthly=False)
    year.to_parquet(DATA_PROCESSED / "panel_country_year.parquet", index=False)
    print(f"панель страна×год: {len(year)} строк, {year.gw.nunique()} стран, "
          f"conflict25 в {year.conflict25.mean():.1%} строк (онсетов {year.onset25.sum()}), "
          f"war25 в {year.war25.mean():.1%} (онсетов {year.onset_war25.sum()})")

    month = build(ged, states, parties, monthly=True)
    month.to_parquet(DATA_PROCESSED / "panel_country_month.parquet", index=False)
    print(f"панель страна×месяц: {len(month)} строк, conflict25 в {month.conflict25.mean():.1%} строк "
          f"(онсетов {month.onset25.sum()}), war25 в {month.war25.mean():.1%} (онсетов {month.onset_war25.sum()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
