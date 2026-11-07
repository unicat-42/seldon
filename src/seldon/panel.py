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

Как считаем:
  - События берём по стране, на территории которой они произошли (`country_id` в GED),
    а не по сторонам конфликта. Поэтому Грузия 2008 и Карабах 2020 попадают в целевую
    переменную, хотя UCDP классифицирует их не как межгосударственные войны.
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
    cols = ["country", "country_id", "year", "date_start", "type_of_violence", "best"]
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
    prev = panel.groupby("gw")["conflict25"].shift(1).fillna(0).astype("int8")
    panel["onset25"] = ((panel["conflict25"] == 1) & (prev == 0)).astype("int8")
    return panel


def build(ged: pd.DataFrame, states: pd.DataFrame, monthly: bool) -> pd.DataFrame:
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
    for c in ["deaths_best", "deaths_state", "events"]:
        panel[c] = panel[c].fillna(0).astype("int64")
    return add_targets(panel, keys)


def main() -> int:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    states, ged = load_states(), load_ged()

    missing = sorted(set(ged.gw) - set(states.gw))
    if missing:
        lost = ged[ged.gw.isin(missing)]
        print(
            f"Внимание: {len(missing)} кодов GED нет в списке государств, их события выпадут "
            f"({len(lost)} событий, {lost.best.sum()} погибших): "
            + ", ".join(f"{n} ({g})" for g, n in lost.drop_duplicates('gw')[['gw','country']].values)
        )

    year = build(ged, states, monthly=False)
    year.to_parquet(DATA_PROCESSED / "panel_country_year.parquet", index=False)
    print(f"панель страна×год: {len(year)} строк, {year.gw.nunique()} стран, "
          f"conflict25 в {year.conflict25.mean():.1%} строк, онсетов {year.onset25.sum()}")

    month = build(ged, states, monthly=True)
    month.to_parquet(DATA_PROCESSED / "panel_country_month.parquet", index=False)
    print(f"панель страна×месяц: {len(month)} строк, conflict25 в {month.conflict25.mean():.1%} строк, "
          f"онсетов {month.onset25.sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
