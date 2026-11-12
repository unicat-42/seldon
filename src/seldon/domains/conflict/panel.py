"""Панель «страна × период» по UCDP GED: погибшие, события и целевые переменные.

Запуск:
    uv run python -m seldon panel

Результат:
    data/processed/panel_country_year.parquet   — страна × год, 1989–2024
    data/processed/panel_country_month.parquet  — страна × месяц, 1989-01 … 2024-12

Колонки (в месячной версии вместо `year` — `year`, `month` и `period` = первое число месяца):
    gw               код страны Gleditsch–Ward (как в UCDP)
    country          название (как в UCDP GED, иначе как у Gleditsch–Ward)
    deaths_best      погибшие на территории, все типы насилия
    deaths_state     то же, только государственные конфликты (type_of_violence == 1)
    deaths_nonstate  то же, негосударственное и одностороннее насилие (типы 2 и 3:
                     банды, картели, теракты, расправы над мирными)
    deaths_party_nb  погибшие в государственных конфликтах, где страна — СТОРОНА, на
                     территории её СОСЕДЕЙ (своя территория сюда не входит — без двойного счёта)
    deaths_war       deaths_state + deaths_party_nb — «погибшие в войнах страны»
    events           число событий GED

  Основные целевые (после аудита 12.11.2026):
    state_war25        1, если deaths_state ≥ 25 (война на своей территории) ИЛИ
                       deaths_party_nb ≥ 25 (страна — сторона войны у соседа)
    state_war1000      1, если deaths_war ≥ 1000
    nonstate25         1, если deaths_nonstate ≥ 25 — отдельная метка, НЕ война
    onset_state_war25, onset_state_war1000, onset_nonstate25 — начала (см. ниже)
    state_war1000_12m  (только месячная) ≥ 1000 погибших за последние 12 месяцев

  Старые целевые — оставлены для преемственности с отчётами до аудита:
    conflict25 / conflict1000 / onset25   по deaths_best (все типы насилия)
    deaths_party, party25                 старое правило сторон (основная где угодно,
                                          помогающая — сосед места по ACD), с двойным счётом
    war25 / war1000 и их онсеты           conflict25 ИЛИ party25

Как считаем:
  - Сторона конфликта-года — по UCDP/PRIO ACD: основная (`gwno_a`/`gwno_b`) или помогающая
    войсками (`gwno_a_2nd`/`gwno_b_2nd`). В `state_war25` сторона «воюет» вне своей
    территории, только если граничит со страной, где по событиям GED реально шли бои этого
    конфликта. Так Армения 2020, Россия 2008/2014/2022 — воюют; США в Ираке, Россия в Сирии,
    Британия и Канада против хуситов 2024 — нет (далёкие экспедиции — другое явление).
    До аудита место бралось из `gwno_loc` ACD (у конфликта с хуситами там США — и Канада
    «воевала» как сосед США), а основная сторона считалась воюющей где угодно.
  - Теракты и банды (типы 2, 3) в `state_war25` не входят: Франция 2015, Бельгия 2016,
    Канада 1998, Гондурас — это `nonstate25`.
  - Начало (онсет). Годовая панель: целевая = 1 в этом году и 0 в прошлом. Месячная:
    целевая = 1 в этом месяце и 0 во ВСЕХ 12 предыдущих (иначе один тихий месяц посреди
    вялотекущего конфликта порождал «новую войну» — так у России весной 2008 «горел»
    онсет из-за Кавказа).
  - Что известно в месяце m. Погибшие месяца m — только события месяца m. Список сторон
    берётся из строки ACD за этот же год: это знание «кто участвует в году Y», и для
    стороны, вступившей в уже идущую войну посреди года (Турция в Сирии, август 2016),
    бои с января приписываются ей заранее. Событийных данных о помогающих сторонах в GED
    нет. Поэтому в месячной панели есть вторые, «в реальном времени», версии колонок
    (чекпоинт 2.2): `deaths_party_nb_rt`, `state_war25_rt`, `onset_state_war25_rt`,
    `state_war1000_12m_rt` — в них стороны берутся из ACD за ПРОШЛЫЙ год (список за год Y
    выходит после его окончания, `core/temporal.py`). Новый участник (Армения, сентябрь 2020)
    в них до конца года «воюет» только на своей территории. Колонки без `_rt` — целевые
    (что произошло на самом деле), с `_rt` — признаки (что было известно в тот месяц).
  - Соседи — Correlates of War Direct Contiguity 3.2: суша или море до 12 миль
    (`conttype` ≤ 2), по годам до 2016, дальше — как в 2016. Коды COW → GW — в core/entities.py.
  - Список стран — все независимые государства по Gleditsch & Ward (файл iisystem.dat),
    существовавшие хотя бы один день в 1989–2024. Строка «страна-период» есть только за
    периоды, когда государство существовало (Украина — с 1991, Южный Судан — с 2011).
    Записи с датой конца 31.12.2017 (последний год списка) считаем существующими по сей день.
  - В первый период существования страны предыдущих периодов нет — считаем их тихими.
"""

from __future__ import annotations

import pandas as pd

from seldon import DATA_PROCESSED
from seldon.core.temporal import last_available_year
from seldon.sources.cow import load_neighbors
from seldon.sources.gw import load_states
from seldon.sources.ucdp import YEAR_FROM, YEAR_TO, load_acd, load_ged

ONSET_QUIET_MONTHS = 12  # месячный онсет: столько месяцев подряд ниже порога перед началом
# список сторон ACD за год Y выходит после окончания Y: внутри года Y известен список за Y−1
ACD_LAG = 2000 - last_available_year("ucdp_acd", pd.Timestamp("2000-12-31"))

# целевая → её онсет
ONSET_OF = {
    "state_war25": "onset_state_war25", "state_war1000": "onset_state_war1000", "nonstate25": "onset_nonstate25",
    "conflict25": "onset25", "war25": "onset_war25", "war1000": "onset_war1000",
}


def _gw_list(cell) -> list[int]:
    return [int(float(x)) for x in str(cell).split(",") if x.strip() not in ("", "nan")]


def load_conflict_parties(neighbors: set[tuple[int, int, int]]) -> pd.DataFrame:
    """Для каждого конфликта-года (ACD): `parties` — все стороны (основные и помогающие
    войсками); `parties_old` — старое правило (основные + помогающие-соседи места по ACD)."""
    a = load_acd(YEAR_FROM)
    parties, old = [], []
    for r in a.itertuples(index=False):
        primary = set(_gw_list(r.gwno_a)) | set(_gw_list(r.gwno_b))
        secondary = set(_gw_list(r.gwno_a_2nd)) | set(_gw_list(r.gwno_b_2nd))
        locs = _gw_list(r.gwno_loc)
        helping_neighbors = {s for s in secondary if any((s, loc, r.year) in neighbors for loc in locs)}
        parties.append(sorted(primary | secondary))
        old.append(sorted(primary | helping_neighbors))
    return pd.DataFrame({"conflict_id": a.conflict_id.values, "year": a.year.values, "parties": parties, "parties_old": old})


def _state_events_by_party(ged: pd.DataFrame, parties: pd.DataFrame, keys: list[str], col: str) -> pd.DataFrame:
    """Государственные события, размноженные по сторонам своего конфликта-года:
    колонки gw (сторона), loc (где событие), ключи периода, best."""
    g = ged.loc[ged.type_of_violence == 1].groupby(["conflict_new_id", "gw", *dict.fromkeys(["year", *keys])], as_index=False).best.sum()
    g = g.rename(columns={"gw": "loc"})
    g = g.merge(parties[["conflict_id", "year", col]], left_on=["conflict_new_id", "year"], right_on=["conflict_id", "year"], how="inner")
    g = g.explode(col).dropna(subset=[col])
    g["gw"] = g[col].astype(int)
    return g


def party_deaths_old(ged: pd.DataFrame, parties: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Старое правило: событие приписывается каждой стороне своего конфликта-года, где бы ни было."""
    col = "parties_old" if "parties_old" in parties else "parties"
    g = _state_events_by_party(ged, parties, keys, col)
    return g.groupby(["gw", *keys]).best.sum().rename("deaths_party").reset_index()


def party_deaths_neighbors(ged: pd.DataFrame, parties: pd.DataFrame, neighbors: set, keys: list[str]) -> pd.DataFrame:
    """Погибшие в конфликтах, где страна — сторона, на территории её соседей (не на своей)."""
    g = _state_events_by_party(ged, parties, keys, "parties")
    nb = pd.DataFrame(list(neighbors), columns=["gw", "loc", "year"])
    g = g[g.gw != g["loc"]].merge(nb, on=["gw", "loc", "year"], how="inner")
    return g.groupby(["gw", *keys]).best.sum().rename("deaths_party_nb").reset_index()


def aggregate(ged: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        ged.groupby(keys)
        .agg(deaths_best=("best", "sum"), deaths_state=("deaths_state", "sum"), events=("best", "size"))
        .reset_index()
    )


def add_targets(panel: pd.DataFrame, keys: list[str], monthly: bool) -> pd.DataFrame:
    panel = panel.sort_values(["gw", *keys]).reset_index(drop=True)
    panel["deaths_nonstate"] = panel.deaths_best - panel.deaths_state
    panel["deaths_war"] = panel.deaths_state + panel.deaths_party_nb
    panel["state_war25"] = ((panel.deaths_state >= 25) | (panel.deaths_party_nb >= 25)).astype("int8")
    panel["state_war1000"] = (panel.deaths_war >= 1000).astype("int8")
    panel["nonstate25"] = (panel.deaths_nonstate >= 25).astype("int8")
    # старые целевые (до аудита)
    panel["conflict25"] = (panel["deaths_best"] >= 25).astype("int8")
    panel["conflict1000"] = (panel["deaths_best"] >= 1000).astype("int8")
    panel["party25"] = (panel["deaths_party"] >= 25).astype("int8")
    panel["war25"] = panel[["conflict25", "party25"]].max(axis=1).astype("int8")
    panel["war1000"] = ((panel["deaths_best"] >= 1000) | (panel["deaths_party"] >= 1000)).astype("int8")
    quiet = ONSET_QUIET_MONTHS if monthly else 1
    for target, onset in ONSET_OF.items():
        # была ли целевая = 1 хоть раз за `quiet` предыдущих периодов
        recent = panel.groupby("gw")[target].transform(lambda s: s.shift(1).rolling(quiet, min_periods=1).max()).fillna(0)
        panel[onset] = ((panel[target] == 1) & (recent == 0)).astype("int8")
    if monthly:
        d12 = panel.groupby("gw").deaths_war.transform(lambda s: s.rolling(12, min_periods=1).sum())
        panel["state_war1000_12m"] = (d12 >= 1000).astype("int8")
        # то же «в реальном времени» (для ПРИЗНАКОВ): стороны — по списку ACD за прошлый год
        g = panel.groupby("gw")
        panel["state_war25_rt"] = ((panel.deaths_state >= 25) | (panel.deaths_party_nb_rt >= 25)).astype("int8")
        recent = g.state_war25_rt.transform(lambda s: s.shift(1).rolling(quiet, min_periods=1).max()).fillna(0)
        panel["onset_state_war25_rt"] = ((panel.state_war25_rt == 1) & (recent == 0)).astype("int8")
        d12 = (panel.deaths_state + panel.deaths_party_nb_rt).groupby(panel.gw).transform(lambda s: s.rolling(12, min_periods=1).sum())
        panel["state_war1000_12m_rt"] = (d12 >= 1000).astype("int8")
    return panel


def build(ged: pd.DataFrame, states: pd.DataFrame, parties: pd.DataFrame, neighbors: set, monthly: bool,
          year_from: int = YEAR_FROM, year_to: int = YEAR_TO) -> pd.DataFrame:
    """Панель страна×период. Значения на строке периода зависят только от событий этого
    периода и раньше и от списка сторон ACD за этот же год (целевые — см. docstring модуля);
    месячные колонки `*_rt` — только от списка сторон за прошлый год."""
    names = ged.drop_duplicates("gw").set_index("gw")["country"]
    states = states.assign(country=states["gw"].map(names).fillna(states["gw_name"]))

    if monthly:
        periods = pd.DataFrame({"period": pd.date_range(f"{year_from}-01-01", f"{year_to}-12-01", freq="MS")})
        periods["year"], periods["month"] = periods.period.dt.year, periods.period.dt.month
        keys = ["year", "month"]
    else:
        periods = pd.DataFrame({"year": range(year_from, year_to + 1)})
        keys = ["year"]

    grid = states[["gw", "country", "start_year", "end_year"]].merge(periods, how="cross")
    grid = grid[(grid.year >= grid.start_year) & (grid.year <= grid.end_year)].drop(columns=["start_year", "end_year"])

    panel = grid.merge(aggregate(ged, ["gw", *keys]), on=["gw", *keys], how="left")
    panel = panel.merge(party_deaths_old(ged, parties, keys), on=["gw", *keys], how="left")
    panel = panel.merge(party_deaths_neighbors(ged, parties, neighbors, keys), on=["gw", *keys], how="left")
    cols = ["deaths_best", "deaths_state", "events", "deaths_party", "deaths_party_nb"]
    if monthly:
        known = parties.assign(year=parties.year + ACD_LAG)  # в году Y известен список сторон за Y−1
        rt = party_deaths_neighbors(ged, known, neighbors, keys).rename(columns={"deaths_party_nb": "deaths_party_nb_rt"})
        panel = panel.merge(rt, on=["gw", *keys], how="left")
        cols.append("deaths_party_nb_rt")
    for c in cols:
        panel[c] = panel[c].fillna(0).astype("int64")
    return add_targets(panel, keys, monthly)


def main(argv: list[str]) -> int:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    states, ged = load_states(), load_ged()
    neighbors = load_neighbors()
    parties = load_conflict_parties(neighbors)

    missing = sorted(set(ged.gw) - set(states.gw))
    if missing:
        lost = ged[ged.gw.isin(missing)]
        print(
            f"Внимание: {len(missing)} кодов GED нет в списке государств, их события выпадут "
            f"({len(lost)} событий, {lost.best.sum()} погибших): "
            + ", ".join(f"{n} ({g})" for g, n in lost.drop_duplicates('gw')[['gw','country']].values)
        )

    for monthly, name in ((False, "year"), (True, "month")):
        p = build(ged, states, parties, neighbors, monthly=monthly)
        p.to_parquet(DATA_PROCESSED / f"panel_country_{name}.parquet", index=False)
        print(f"панель страна×{'месяц' if monthly else 'год'}: {len(p)} строк, {p.gw.nunique()} стран")
        for t in ("state_war25", "state_war1000", "nonstate25", "war25", "war1000", "conflict25"):
            print(f"    {t:14s} в {p[t].mean():6.1%} строк, онсетов {int(p[ONSET_OF[t]].sum())}")
    return 0

