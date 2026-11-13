"""Месячная панель войн с предвестниками «дым до огня» (чекпоинт 4.2).

Запуск:
    uv run python -m seldon month_features

Панель страна × месяц — та же, что в `panel.py` (правила `state_war25`: бои на своей
территории или страна — сторона войны у соседа; порог — 25 погибших В МЕСЯЦ), но продлённая
оперативными Candidate Events за месяцы после выверенного GED (`sources/ucdp.load_events`).
Боевой файл `panel_country_month.parquet` не трогаем — пишем свой.

Что известно на строке месяца m (прогноз «на конец месяца m», окно m+1 … m+h):
  - события UCDP — по месяц m − lag включительно, lag = 1 (`core/temporal.py`, `ucdp_events`:
    файл Candidate за месяц m выходит 20-го числа m+1). Вариант `lag = 0` — только для
    сравнения с conflictforecast, у которых признаки строки включают её собственный месяц;
  - стороны войн — по списку ACD за прошлый год (колонки `*_rt` панели);
  - режим, детская смертность, население (признаки PITF) — строка годовой таблицы за год Y−1,
    она собрана «на 1 января Y» (`features_pitf.py`);
  - соседи — COW, суша или море до 12 миль (`conttype` ≤ 2); у соседа считаются только бои
    на ЕГО территории (без «телепортации» через вторые руки).

Группы признаков:
  HIST       память о своей войне и война у соседей (месячный аналог `features.py`);
  HIST_DRY   то же, но погибшие считаются только в месяцы войны (без субпороговых стычек);
  PITF       структурные признаки эталона `pitf`;
  SMOKE      свои субвоенные предвестники: государственные стычки ниже порога (погибшие за
             3 / 12 / 24 месяца, число событий, месяцы с 1–24 погибшими, ускорение),
             негосударственное (тип 2) и одностороннее (тип 3) насилие, малые потери как
             стороны войны у соседа;
  SMOKE_NB   то же у непосредственных соседей;
  LIVE       столкновения с другими государствами (`live_pairs.py`): прямые, прокси, чужая
             армия на своей территории, своя — на чужой;
  CFX        только для целевой conflictforecast: память о любом насилии и их пороге
             «0,5 погибшего на миллион жителей в месяц».

Состояния и выборки:
  quiet_true    в месяцах m−11 … m не было `state_war25` — страна «мирная» по факту (на этих
                строках оценивается НАЧАЛО войны);
  quiet_known   то же по тому, что известно: `state_war25_rt` в месяцах m−lag−11 … m−lag
                (по нему строка получает модель начала или продолжения);
  qmonths       сколько месяцев прошло с последнего месяца войны по факту (для срезов
                «тихие ≥ 5 лет» и «≥ 10 лет»; не воевавшие с 1989 — месяцы с начала панели);
  ac, ac_quiet_true, ac_quiet_known — то же для целевой conflictforecast (порог на миллион
                жителей, «мирная» = нет конфликта в одном текущем месяце).

Выход: data/processed/war_month_panel.parquet, war_month_features.parquet (lag 1),
war_month_features_lag0.parquet, features_smoke_country_year.parquet (годовой срез: строка
декабря T → набор `pitf_smoke` / `pitf_live` для `model_hist`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED
from seldon.core.temporal import monthly_lag, publication_lag
from seldon.domains.conflict import dyads, live_pairs, panel
from seldon.domains.conflict.features_pitf import PITF_FEATURES
from seldon.sources.cow import load_neighbors
from seldon.sources.gw import load_states
from seldon.sources.ucdp import CANDIDATE_LAST_MONTH, YEAR_FROM, YEAR_TO, load_acd, load_events
from seldon.sources.wdi import load_wdi

EVENT_LAG = monthly_lag("ucdp_events")  # 1
WAR = 25  # порог месяца
QUIET_MONTHS = panel.ONSET_QUIET_MONTHS  # 12
POP_LAG = publication_lag("wdi_annual", "jan1") + 1  # население за Y−2: известно весь год Y
AC_PER_MILLION = 0.5  # порог conflictforecast: погибших на миллион жителей в месяц

HIST = ["war_prev", "m_since_war", "never_war", "war_m12", "war_m60", "ld_war_1", "ld_war_3", "ld_war_12", "ld_war_60",
        "decay_6", "decay_36", "nb_war_share3", "nb_war_any12", "nb_onset12", "nb_ld_3"]
BASE = ["m_since_war", "never_war", "decay_6", "decay_36"]
# «сухая» история: погибшие считаются только в месяцы войны (≥ 25). В HIST (как и в годовом
# `features.py`) `ld_war_*` у мирной страны — это и есть её субпороговые стычки, то есть часть
# «дыма» уже сидит в эталоне; HIST_DRY нужен, чтобы измерить вклад дыма целиком
DRY = ["ldw_1", "ldw_3", "ldw_12", "ldw_60"]
HIST_DRY = [c for c in HIST if not c.startswith("ld_war_")] + DRY
SMOKE = ["sb_ld_3", "sb_ld_12", "sb_ld_24", "sb_ev_3", "sb_ev_12", "sb_months12", "sb_low_months24", "sb_accel",
         "pnb_ld_12", "ns_ld_3", "ns_ld_12", "ns_ev_12", "os_ld_3", "os_ld_12", "os_ev_12"]
SMOKE_NB = ["nb_low_ld12", "nb_low_share12", "nb_ns_ld12", "nb_os_ld12", "nb_accel"]
LIVE = ["lv_dir_ld12", "lv_px_ld12", "lv_in_ld12", "lv_out_ld12", "lv_ev36"]
CFX = ["ac_prev", "ac_m12", "m_since_ac", "m_since_any", "ld_best_1", "ld_best_3", "ld_best_12", "ld_best_60"]
ALL_FEATURES = [*HIST, *DRY, *PITF_FEATURES, *SMOKE, *SMOKE_NB, *LIVE, *CFX]


def build_panel(events: pd.DataFrame, states: pd.DataFrame, parties: pd.DataFrame, neighbors: set,
                year_from: int, last_month: pd.Period) -> pd.DataFrame:
    """Месячная панель `panel.build` по `last_month` включительно плюс разбивка по типам насилия:
    sb_events (число государственных событий), ns_* (тип 2), os_* (тип 3), `ym`, `o`."""
    p = panel.build(events, states, parties, neighbors, monthly=True, year_from=year_from, year_to=last_month.year)
    e = events.assign(sb=(events.type_of_violence == 1).astype(int), ns=(events.type_of_violence == 2).astype(int),
                      os=(events.type_of_violence == 3).astype(int))
    for k in ("ns", "os"):
        e[f"{k}_d"] = e.best * e[k]
    x = e.groupby(["gw", "year", "month"], as_index=False).agg(
        sb_events=("sb", "sum"), ns_deaths=("ns_d", "sum"), ns_events=("ns", "sum"), os_deaths=("os_d", "sum"), os_events=("os", "sum"))
    p = p.merge(x, on=["gw", "year", "month"], how="left")
    cols = ["sb_events", "ns_deaths", "ns_events", "os_deaths", "os_events"]
    p[cols] = p[cols].fillna(0).astype("int64")
    p["o"] = live_pairs.month_ordinal(p.year, p.month)
    p["ym"] = pd.PeriodIndex.from_ordinals(p.o, freq="M")
    return p[p.ym <= last_month].sort_values(["gw", "o"]).reset_index(drop=True)


def _months_since(flag: pd.Series, gw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Месяцев с последнего месяца, где `flag` == 1 (0 — в этом месяце); если такого не было —
    месяцев с начала ряда страны. Второй результат — флаг «не было ни разу»."""
    pos = flag.groupby(gw).cumcount()
    last = pos.where(flag == 1).groupby(gw).ffill()
    return (pos - last).fillna(pos + 1).astype(int), last.isna().astype("int8")


def build_features(p: pd.DataFrame, neighbors: set, struct: pd.DataFrame | None = None, live: pd.DataFrame | None = None,
                   pop: pd.DataFrame | None = None, lag: int = EVENT_LAG) -> pd.DataFrame:
    """Признаки и состояния месячной панели. Чистая функция: всё, что зависит от событий, на
    строке месяца m считается по месяцам ≤ m − lag (кроме `quiet_true`, `qmonths`, `ac`,
    `ac_quiet_true` — это факт, а не признак: по ним отбираются строки обучения и оценки).
    `struct` — gw, year и признаки PITF (строка года Y−1 ложится на месяцы года Y);
    `live` — `live_pairs.country_month`; `pop` — gw, year, population."""
    p = p.sort_values(["gw", "o"]).reset_index(drop=True)
    gw = p.gw
    f = p[["gw", "country", "year", "month", "ym", "o", "state_war25", "onset_state_war25", "deaths_state", "deaths_best"]].copy()

    def lagged(s: pd.Series) -> pd.Series:
        return s.groupby(gw).shift(lag).fillna(0)

    def roll(s: pd.Series, w: int, how: str = "sum") -> pd.Series:
        """Окно в w месяцев, кончающееся месяцем m − lag."""
        return lagged(s).groupby(gw).transform(lambda x: x.rolling(w, min_periods=1).agg(how))

    war_rt, d_war = p.state_war25_rt, p.deaths_state + p.deaths_party_nb_rt
    # --- факт (не признаки) ---
    f["quiet_true"] = (p.state_war25.groupby(gw).transform(lambda x: x.rolling(QUIET_MONTHS, min_periods=1).max()) == 0).astype("int8")
    f["qmonths"], _ = _months_since(p.state_war25, gw)
    # --- HIST ---
    f["quiet_known"] = (roll(war_rt, QUIET_MONTHS, "max") == 0).astype("int8")
    f["war_prev"] = lagged(war_rt)
    since, never = _months_since(lagged(war_rt), gw)
    f["m_since_war"], f["never_war"] = np.log1p(since), never
    f["war_m12"], f["war_m60"] = roll(war_rt, 12, "mean"), roll(war_rt, 60, "mean")
    for w in (1, 3, 12, 60):
        f[f"ld_war_{w}"] = np.log1p(roll(d_war, w))
        f[f"ldw_{w}"] = np.log1p(roll(d_war * war_rt, w))
    for h in (6, 36):
        f[f"decay_{h}"] = lagged(war_rt).groupby(gw).transform(lambda x: x.ewm(alpha=1 - 0.5 ** (1 / h), adjust=False).mean())
    # --- SMOKE ---
    ds = p.deaths_state
    for w in (3, 12, 24):
        f[f"sb_ld_{w}"] = np.log1p(roll(ds, w))
    f["sb_ev_3"], f["sb_ev_12"] = np.log1p(roll(p.sb_events, 3)), np.log1p(roll(p.sb_events, 12))
    f["sb_months12"] = roll((ds > 0).astype(int), 12, "mean")
    f["sb_low_months24"] = roll(((ds > 0) & (ds < WAR)).astype(int), 24)
    d3, d12 = roll(ds, 3), roll(ds, 12)
    f["sb_accel"] = np.log1p(d3) - np.log1p((d12 - d3) / 3)
    f["pnb_ld_12"] = np.log1p(roll(p.deaths_party_nb_rt, 12))
    for k in ("ns", "os"):
        f[f"{k}_ld_3"], f[f"{k}_ld_12"] = np.log1p(roll(p[f"{k}_deaths"], 3)), np.log1p(roll(p[f"{k}_deaths"], 12))
        f[f"{k}_ev_12"] = np.log1p(roll(p[f"{k}_events"], 12))
    # --- соседи: только бои на территории самого соседа ---
    terr = (ds >= WAR).astype(int)
    terr_prev = terr.groupby(gw).transform(lambda x: x.shift(1).rolling(QUIET_MONTHS, min_periods=1).max()).fillna(0)
    low = ds.where(ds < WAR, 0)
    own = pd.DataFrame({"nb": gw, "o": p.o, "t_war3": roll(terr, 3, "max"), "t_war12": roll(terr, 12, "max"),
                        "t_onset12": roll(((terr == 1) & (terr_prev == 0)).astype(int), 12, "max"), "t_d3": d3, "t_d9": d12 - d3,
                        "t_low12": roll(low, 12), "t_ns12": roll(p.ns_deaths, 12), "t_os12": roll(p.os_deaths, 12)})
    own["t_lowany"] = (own.t_low12 > 0).astype(int)
    nb = pd.DataFrame(list(neighbors), columns=["gw", "nb", "year"]).merge(p[["gw", "year", "o"]], on=["gw", "year"])
    nb = nb.merge(own, on=["nb", "o"], how="inner")
    a = nb.groupby(["gw", "o"]).agg(nb_war_share3=("t_war3", "mean"), nb_war_any12=("t_war12", "max"), nb_onset12=("t_onset12", "max"),
                                    d3=("t_d3", "sum"), d9=("t_d9", "sum"), low=("t_low12", "sum"), nb_low_share12=("t_lowany", "mean"),
                                    ns=("t_ns12", "sum"), os=("t_os12", "sum")).reset_index()
    a["nb_ld_3"], a["nb_low_ld12"] = np.log1p(a.d3), np.log1p(a.low)
    a["nb_ns_ld12"], a["nb_os_ld12"] = np.log1p(a.ns), np.log1p(a.os)
    a["nb_accel"] = np.log1p(a.d3) - np.log1p(a.d9 / 3)
    nb_cols = ["nb_war_share3", "nb_war_any12", "nb_onset12", "nb_ld_3", *SMOKE_NB]
    f = f.merge(a[["gw", "o", *nb_cols]], on=["gw", "o"], how="left")
    f[nb_cols] = f[nb_cols].fillna(0.0)
    # --- LIVE ---
    lv = p[["gw", "o"]].merge(live, on=["gw", "o"], how="left").fillna(0) if live is not None else None
    for col, src, w in (("lv_dir_ld12", "lv_dir_d", 12), ("lv_px_ld12", "lv_px_d", 12), ("lv_in_ld12", "lv_in_d", 12),
                        ("lv_out_ld12", "lv_out_d", 12), ("lv_ev36", "lv_ev", 36)):
        f[col] = np.log1p(roll(lv[src], w)) if lv is not None else 0.0
    # --- PITF ---
    if struct is not None:
        s = struct[["gw", "year", *PITF_FEATURES]].assign(year=struct.year + 1)
        f = f.merge(s, on=["gw", "year"], how="left")
        f["pitf_row_missing"] = f[PITF_FEATURES[0]].isna().astype("int8")
        f[PITF_FEATURES] = f[PITF_FEATURES].fillna(0.0)
    else:
        f[PITF_FEATURES], f["pitf_row_missing"] = 0.0, 1
    # --- целевая conflictforecast и память о любом насилии ---
    db = p.deaths_best
    if pop is not None:
        years = range(int(pop.year.min()), int(p.year.max()) + 1)
        idx = pd.MultiIndex.from_product([sorted(pop.gw.unique()), years], names=["gw", "year"])
        t = pop.drop_duplicates(["gw", "year"]).set_index(["gw", "year"]).population.reindex(idx).groupby(level="gw").ffill().reset_index()
        pm = f[["gw", "year"]].merge(t.assign(year=t.year + POP_LAG), on=["gw", "year"], how="left").population.to_numpy(float)
    else:
        pm = np.full(len(f), np.nan)
    f["ac"] = (np.nan_to_num(db.to_numpy(float) / pm * 1e6, nan=0.0) >= AC_PER_MILLION).astype("int8")
    f["ac_quiet_true"] = (f.ac == 0).astype("int8")
    f["ac_prev"] = lagged(f.ac)
    f["ac_quiet_known"] = (f.ac_prev == 0).astype("int8")
    f["ac_m12"] = roll(f.ac, 12, "mean")
    f["m_since_ac"] = np.log1p(_months_since(lagged(f.ac), gw)[0])
    f["m_since_any"] = np.log1p(_months_since(lagged((db > 0).astype(int)), gw)[0])
    for w in (1, 3, 12, 60):
        f[f"ld_best_{w}"] = np.log1p(roll(db, w))
    return f


def annual_slice(f: pd.DataFrame) -> pd.DataFrame:
    """Годовой срез: признаки строки декабря T (события по ноябрь T при lag = 1) → строка
    (страна, T) годовой панели. gw, year и `SMOKE`, `SMOKE_NB`, `LIVE`."""
    d = f[f.month == 12]
    return d[["gw", "year", *SMOKE, *SMOKE_NB, *LIVE]].reset_index(drop=True)


def load_inputs(last_month: pd.Period = CANDIDATE_LAST_MONTH) -> dict:
    events = load_events()
    events = events[(events.year < last_month.year) | (events.month <= last_month.month)]
    neighbors = load_neighbors(YEAR_FROM, last_month.year)
    return {"events": events, "states": load_states(YEAR_FROM, last_month.year), "neighbors": neighbors,
            "parties": panel.load_conflict_parties(neighbors),
            "opp": dyads.opponents(load_acd(dyads.ACD_FROM, extra=("incompatibility",)))}


def main(argv: list[str]) -> int:
    last = CANDIDATE_LAST_MONTH
    s = load_inputs(last)
    p = build_panel(s["events"], s["states"], s["parties"], s["neighbors"], YEAR_FROM, last)
    p.to_parquet(DATA_PROCESSED / "war_month_panel.parquet", index=False)
    live = live_pairs.country_month(live_pairs.pair_events(s["events"], s["opp"], acd_lag=panel.ACD_LAG))
    struct = pd.read_parquet(DATA_PROCESSED / "features_pitf_country_year.parquet")
    pop = load_wdi("population")
    for lag, name in ((EVENT_LAG, "war_month_features.parquet"), (0, "war_month_features_lag0.parquet")):
        f = build_features(p, s["neighbors"], struct, live, pop, lag=lag)
        f.to_parquet(DATA_PROCESSED / name, index=False)
        if lag == EVENT_LAG:
            base = struct.merge(annual_slice(f), on=["gw", "year"], how="left")
            base[[*SMOKE, *SMOKE_NB, *LIVE]] = base[[*SMOKE, *SMOKE_NB, *LIVE]].fillna(0.0)
            base.to_parquet(DATA_PROCESSED / "features_smoke_country_year.parquet", index=False)
            fin = f[f.year <= YEAR_TO]
            print(f"месячная панель: {len(f)} строк, {f.gw.nunique()} стран, {f.ym.min()} … {f.ym.max()} "
                  f"(выверенный GED по {YEAR_TO}-12, дальше Candidate)")
            print(f"  месяцев войны (`state_war25`, ≥ {WAR} погибших в месяц) по {YEAR_TO}: {int(fin.state_war25.sum())} "
                  f"({fin.state_war25.mean():.1%}); начал после {QUIET_MONTHS} тихих месяцев: {int(fin.onset_state_war25.sum())}")
            q = fin[fin.quiet_true == 1]
            print(f"  мирных страна-месяцев: {len(q)}; из них со стычками 1–24 погибших за 12 месяцев: {(q.sb_ld_12 > 0).mean():.1%}, "
                  f"с негосударственным насилием: {(q.ns_ld_12 > 0).mean():.1%}, с односторонним: {(q.os_ld_12 > 0).mean():.1%}, "
                  f"со столкновениями с другим государством: {(q.lv_ev36 > 0).mean():.1%}")
            print(f"  расхождение известного и фактического «мирного» состояния: {(fin.quiet_true != fin.quiet_known).mean():.2%} строк")
    return 0
