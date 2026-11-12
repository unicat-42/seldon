"""Направленная диадная панель «страна A ← страна B × год» (чекпоинт 3.2).

Запуск:
    uv run python -m seldon dyads

A — страна, чей риск считаем (бои на её территории или она — сторона войны у соседа);
B — возможный противник или интервент. Строка (A, B, T) → вопрос «окажутся ли A и B по
разные стороны государственной войны в году T+1».

Вселенная пар (политически значимые диады):
  - все соседи по COW Direct Contiguity, `conttype` 1…5 (суша и море до 400 миль); после
    2016 года география границ продлена из последнего года данных (`sources/cow.py`);
  - плюс пары, где B — держава с глобальной проекцией силы (`POWERS`: США, Россия, Китай,
    Великобритания, Франция), даже без общей границы.

Целевая `dyad_war` (A ← B, год Y) — прямо из UCDP 26.1: A и B стоят по разные стороны
конфликта ACD (основные стороны `gwno_a` / `gwno_b` или помогающие войсками `*_2nd`), и в
этом конфликте за год погибло ≥ 25 человек на территории A ИЛИ ≥ 25 на территории её
близких соседей (суша / море до 12 миль). Это ровно та часть `state_war25` страны A, где
противник — государство B: `dyad_war = 1` ⇒ `state_war25(A) = 1` (сторожит тест). Далёкая
экспедиция — война для A, на чьей территории бои, но не для далёкого B: Ирак ← США 2003 = 1,
США ← Ирак 2003 = 0.

Что известно на строке T (прогноз «на 1 января T+1», `core/temporal.py`):
  - события GED и стороны ACD за год T (годовой список сторон выходит после окончания года);
  - военные расходы, ВВП, население — за T−1 (`wdi_annual`, `gmd_annual`);
  - идеальные точки ГА ООН — за T−1 (`unga_ideal`);
  - споры MID — по T−3 (`cow_mid`), и не позже последнего года хроники (2014): дальше
    значения заморожены с флагом `mid_extended` (инвариант 6).

Признаки модели (`FEATURES`, 8 штук — набор зафиксирован до взгляда на тест):
    contig_close     1 — общая граница по суше или море до 12 миль (иначе — дальний сосед
                     по морю или далёкая держава)
    pair_recency     память пары: 1 / (1 + лет с последнего года, когда A и B были по разные
                     стороны конфликта ACD, с 1946); 0 — не были никогда
    terr_dispute     1, если в истории пары есть спор о территории: конфликт UCDP с
                     территориальной несовместимостью, где они были противниками, или
                     территориальный спор MID
    mid_force        log(1 + число лет с применением силы в паре по MID за 15 доступных лет)
    log_power_ratio  логарифм отношения потенциалов B / A: среднее из доступных отношений
                     военных расходов, ВВП и населения
    b_milex_surge    рывок реальных военных расходов B за два года (логарифм отношения;
                     реальные расходы = доля в ВВП × реальный ВВП)
    unga_dist        внешнеполитическая дистанция |идеальная точка A − идеальная точка B|
    unga_shift       изменение этой дистанции за три года (> 0 — пара расходится)
Простая база (`BASE_FEATURES`): память пары + общая граница.

Не признаки, для отчёта: `pair_years_since`, `pair_episodes` (число прошлых эпизодов
вражды), `pair_terr_ucdp`, `mid_terr`, `mid_extended`, `cap_missing`, `unga_missing`.

Выход:
    data/processed/dyad_features.parquet             панель пар с признаками и целевой
    data/processed/dyad_ext_country_year.parquet     страна × год: `ext_war` — в `state_war25`
                                                     страны есть государство-противник (все
                                                     противники, не только из вселенной пар)
    data/processed/features_dyad_country_year.parquet   признаки `pitf` + свёрнутые диадные
                                                     (`COUNTRY_FEATURES`, набор `pitf_dyad`)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED
from seldon.core.entities import ISO3_TO_GW
from seldon.core.temporal import publication_lag
from seldon.sources.cow import CONTIGUITY_MAX_TYPE, load_contiguity
from seldon.sources.gmd import load_gmd
from seldon.sources.gw import load_states
from seldon.sources.mid import MID_LAST_YEAR, load_mid
from seldon.sources.ucdp import YEAR_FROM, YEAR_TO, load_acd, load_ged
from seldon.sources.unga import load_unga
from seldon.sources.wdi import load_wdi

POWERS = (2, 365, 710, 200, 220)  # США, Россия, Китай, Великобритания, Франция (коды GW)
WAR_DEATHS = 25
ACD_FROM = 1946  # память пары — по всей хронике ACD
TERRITORIAL = (1, 3)  # incompatibility: 1 — территория, 3 — территория и власть
MID_WINDOW = 15
UNGA_SHIFT_YEARS = 3
SURGE_YEARS = 2
SURGE_CLIP = 1.5
FILL_YEARS = 3  # пропуск в годовом ряду закрывается последним известным значением, не старше

CAP_LAG = publication_lag("wdi_annual", "jan1")  # 1
UNGA_LAG = publication_lag("unga_ideal", "jan1")  # 1
MID_LAG = publication_lag("cow_mid", "jan1")  # 3

BASE_FEATURES = ["contig_close", "pair_recency"]
FEATURES = [*BASE_FEATURES, "terr_dispute", "mid_force", "log_power_ratio", "b_milex_surge", "unga_dist", "unga_shift"]
COUNTRY_FEATURES = ["dy_terr_any", "dy_unga_shift_max", "dy_power_max"]
CAP_COLS = ["milex_usd", "milex_gdp", "gdp_usd", "rgdp", "pop"]


def _gw_list(cell) -> list[int]:
    return [int(float(x)) for x in str(cell).split(",") if x.strip() not in ("", "nan")]


# ---------------------------------------------------------------------------
# Вселенная пар и целевая
# ---------------------------------------------------------------------------


def build_grid(states: pd.DataFrame, contig: pd.DataFrame, year_from: int = YEAR_FROM, year_to: int = YEAR_TO,
               powers: tuple[int, ...] = POWERS) -> pd.DataFrame:
    """Сетка A ← B × год: соседи (`contig`: a, b, year, conttype) плюс пары с державой-B.
    Строка есть, только если оба государства существуют в этом году."""
    years = pd.DataFrame({"year": range(year_from, year_to + 1)})
    ex = states[["gw", "start_year", "end_year"]].merge(years, how="cross")
    ex = ex[(ex.year >= ex.start_year) & (ex.year <= ex.end_year)][["gw", "year"]]
    alive = set(zip(ex.gw, ex.year))
    nb = contig[["a", "b", "year", "conttype"]]
    pw = ex.rename(columns={"gw": "a"}).merge(pd.DataFrame({"b": list(powers)}), how="cross")
    g = pd.concat([nb, pw[pw.a != pw.b]], ignore_index=True)
    g = g.groupby(["a", "b", "year"], as_index=False).conttype.min()  # NaN — границы нет
    ok = np.array([(a, y) in alive and (b, y) in alive for a, b, y in zip(g.a, g.b, g.year)], dtype=bool)
    g = g[ok & (g.year >= year_from) & (g.year <= year_to)].copy()
    g["conttype"] = g.conttype.fillna(0).astype("int8")
    g["contig_any"] = (g.conttype > 0).astype("int8")
    g["contig_close"] = ((g.conttype > 0) & (g.conttype <= CONTIGUITY_MAX_TYPE)).astype("int8")
    g["b_power"] = g.b.isin(powers).astype("int8")
    return g.sort_values(["a", "b", "year"]).reset_index(drop=True)


def opponents(acd: pd.DataFrame) -> pd.DataFrame:
    """Пары государств по разные стороны конфликта-года ACD, в обоих направлениях:
    conflict_id, year, a, b, terr (1 — несовместимость территориальная)."""
    rows = []
    for r in acd.itertuples(index=False):
        side_a = set(_gw_list(r.gwno_a)) | set(_gw_list(r.gwno_a_2nd))
        side_b = set(_gw_list(r.gwno_b)) | set(_gw_list(r.gwno_b_2nd))
        terr = int(r.incompatibility in TERRITORIAL)
        for x in side_a:
            for y in side_b:
                if x != y:
                    rows.append((r.conflict_id, r.year, x, y, terr))
                    rows.append((r.conflict_id, r.year, y, x, terr))
    out = pd.DataFrame(rows, columns=["conflict_id", "year", "a", "b", "terr"])
    return out.drop_duplicates().reset_index(drop=True)


def dyad_wars(ged: pd.DataFrame, opp: pd.DataFrame, neighbors: set) -> pd.DataFrame:
    """Пары-годы с `dyad_war = 1` (A ← B): в конфликтах, где A и B — противники, за год
    ≥ 25 погибших на территории A или ≥ 25 на территории близких соседей A."""
    g = ged.loc[ged.type_of_violence == 1].groupby(["conflict_new_id", "year", "gw"], as_index=False).best.sum()
    g = g.rename(columns={"conflict_new_id": "conflict_id", "gw": "loc"})
    d = opp.merge(g, on=["conflict_id", "year"], how="inner")
    nb = pd.DataFrame(list(neighbors), columns=["a", "loc", "year"]).assign(near=1)
    d = d.merge(nb, on=["a", "loc", "year"], how="left")
    d["own"] = d.best.where(d["loc"] == d.a, 0)
    d["nb"] = d.best.where((d["loc"] != d.a) & (d.near == 1), 0)
    s = d.groupby(["a", "b", "year"], as_index=False)[["own", "nb"]].sum()
    s = s[(s.own >= WAR_DEATHS) | (s.nb >= WAR_DEATHS)]
    return s[["a", "b", "year"]].assign(dyad_war=1).reset_index(drop=True)


def external_wars(dw: pd.DataFrame) -> pd.DataFrame:
    """Страна × год, где в `state_war25` страны есть государство-противник: gw, year, ext_war."""
    return dw.groupby(["a", "year"], as_index=False).dyad_war.max().rename(columns={"a": "gw", "dyad_war": "ext_war"})


# ---------------------------------------------------------------------------
# Признаки
# ---------------------------------------------------------------------------


def _asof(left: pd.DataFrame, right: pd.DataFrame, key: str) -> pd.DataFrame:
    """К каждой строке `left` — последняя строка `right` той же пары с годом ≤ `left[key]`."""
    r = right.rename(columns={"year": "_hyear"}).sort_values("_hyear")
    l = left.reset_index().sort_values(key)
    out = pd.merge_asof(l, r, left_on=key, right_on="_hyear", by=["a", "b"])
    return out.set_index("index").loc[left.index].drop(columns="_hyear")


def add_pair_history(d: pd.DataFrame, opp: pd.DataFrame) -> pd.DataFrame:
    """Память пары по ACD (с 1946): строка T зависит только от конфликт-лет ≤ T."""
    h = opp.groupby(["a", "b", "year"], as_index=False).terr.max().sort_values(["a", "b", "year"])
    gap = h.groupby(["a", "b"]).year.diff()
    h["pair_episodes"] = (gap.isna() | (gap > 1)).astype(int).groupby([h.a, h.b]).cumsum()
    h["pair_terr_ucdp"] = h.groupby(["a", "b"]).terr.cummax()
    h["pair_last"] = h.year
    out = _asof(d, h[["a", "b", "year", "pair_last", "pair_episodes", "pair_terr_ucdp"]], "year")
    never = out.pair_last.isna()
    out["pair_years_since"] = (out.year - out.pair_last).where(~never)
    out["pair_recency"] = (1.0 / (1.0 + out.pair_years_since)).fillna(0.0)
    out[["pair_episodes", "pair_terr_ucdp"]] = out[["pair_episodes", "pair_terr_ucdp"]].fillna(0).astype(int)
    return out.drop(columns="pair_last")


def add_mid(d: pd.DataFrame, mid: pd.DataFrame, lag: int = MID_LAG, last_year: int = MID_LAST_YEAR) -> pd.DataFrame:
    """Споры MID: на строке T — хроника по `min(T − lag, last_year)`. После конца хроники
    значения заморожены (`mid_extended = 1`), всё записанное позже `last_year` игнорируется."""
    m = mid[mid.year <= last_year].sort_values(["a", "b", "year"]).copy()
    m["cum_force"] = m.groupby(["a", "b"]).force.cumsum()
    m["mid_terr"] = m.groupby(["a", "b"]).terr.cummax()
    out = d.copy()
    out["_av"] = np.minimum(out.year - lag, last_year)
    out["_av0"] = out._av - MID_WINDOW
    now = _asof(out, m[["a", "b", "year", "cum_force", "mid_terr"]], "_av")
    old = _asof(out, m[["a", "b", "year", "cum_force"]], "_av0")
    out["mid_force"] = np.log1p(now.cum_force.fillna(0) - old.cum_force.fillna(0))
    out["mid_terr"] = now.mid_terr.fillna(0).astype(int)
    out["mid_extended"] = (out.year - lag > last_year).astype("int8")
    return out.drop(columns=["_av", "_av0"])


def _lagged(table: pd.DataFrame, cols: list[str], lag: int, years: range) -> pd.DataFrame:
    """Страна × год на полной сетке лет: пропуск закрыт последним значением не старше
    `FILL_YEARS`, затем сдвиг на `lag` (на строке T — значение за T − lag)."""
    idx = pd.MultiIndex.from_product([sorted(table.gw.unique()), years], names=["gw", "year"])
    t = table.drop_duplicates(["gw", "year"]).set_index(["gw", "year"])[cols].reindex(idx)
    t = t.groupby(level="gw").ffill(limit=FILL_YEARS).reset_index()
    return t.assign(year=t.year + lag)


def _side(d: pd.DataFrame, t: pd.DataFrame, who: str) -> pd.DataFrame:
    """Значения таблицы страна × год для стороны `who` ('a' или 'b') каждой строки `d`."""
    r = t.rename(columns={"gw": who})
    return d[[who, "year"]].merge(r, on=[who, "year"], how="left").drop(columns=[who, "year"]).set_index(d.index)


def add_capabilities(d: pd.DataFrame, cap: pd.DataFrame, lag: int = CAP_LAG) -> pd.DataFrame:
    """Асимметрия потенциалов и рывок военных расходов B. `cap`: gw, year и `CAP_COLS`."""
    years = range(int(cap.year.min()), int(d.year.max()) + 1)
    c = cap.copy()
    c["real_milex"] = c.milex_gdp * c.rgdp
    cols = ["milex_usd", "gdp_usd", "pop", "real_milex"]
    t = _lagged(c, cols, lag, years)
    old = _lagged(c, ["real_milex"], lag + SURGE_YEARS, years).rename(columns={"real_milex": "real_milex_old"})
    a, b = _side(d, t, "a"), _side(d, t.merge(old, on=["gw", "year"], how="left"), "b")
    out = d.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.column_stack([np.log(b[k].to_numpy(float) / a[k].to_numpy(float)) for k in ("milex_usd", "gdp_usd", "pop")])
        surge = np.log(b.real_milex.to_numpy(float) / b.real_milex_old.to_numpy(float))
    ratios[~np.isfinite(ratios)] = np.nan
    n = (~np.isnan(ratios)).sum(axis=1)
    out["log_power_ratio"] = np.where(n > 0, np.nansum(ratios, axis=1) / np.maximum(n, 1), np.nan)
    out["cap_missing"] = (n == 0).astype("int8")
    out["log_power_ratio"] = out.log_power_ratio.fillna(out.groupby("year").log_power_ratio.transform("median")).fillna(0.0)
    out["b_milex_surge"] = np.clip(np.where(np.isfinite(surge), surge, 0.0), -SURGE_CLIP, SURGE_CLIP)
    return out


def add_unga(d: pd.DataFrame, unga: pd.DataFrame, lag: int = UNGA_LAG) -> pd.DataFrame:
    """Внешнеполитическая дистанция пары и её сдвиг за `UNGA_SHIFT_YEARS` лет."""
    years = range(int(unga.year.min()), int(d.year.max()) + 1)
    now = _lagged(unga, ["ideal"], lag, years)
    old = _lagged(unga, ["ideal"], lag + UNGA_SHIFT_YEARS, years)
    dist = (_side(d, now, "a").ideal - _side(d, now, "b").ideal).abs()
    dist_old = (_side(d, old, "a").ideal - _side(d, old, "b").ideal).abs()
    out = d.copy()
    out["unga_missing"] = dist.isna().astype("int8")
    out["unga_dist"] = dist.fillna(dist.groupby(out.year).transform("median")).fillna(0.0)
    out["unga_shift"] = (dist - dist_old).fillna(0.0)
    return out


def build_features(grid: pd.DataFrame, opp: pd.DataFrame, dw: pd.DataFrame, unga: pd.DataFrame, mid: pd.DataFrame,
                   cap: pd.DataFrame, unga_lag: int = UNGA_LAG, mid_lag: int = MID_LAG, cap_lag: int = CAP_LAG,
                   mid_last_year: int = MID_LAST_YEAR) -> pd.DataFrame:
    """Панель пар с целевой и признаками. Чистая функция: строка (A, B, T) зависит только от
    того, что известно к 1 января T+1 (см. docstring модуля)."""
    d = grid.merge(dw, on=["a", "b", "year"], how="left")
    d["dyad_war"] = d.dyad_war.fillna(0).astype("int8")
    d = add_pair_history(d, opp)
    d = add_mid(d, mid, mid_lag, mid_last_year)
    d["terr_dispute"] = ((d.pair_terr_ucdp == 1) | (d.mid_terr == 1)).astype("int8")
    d = add_capabilities(d, cap, cap_lag)
    d = add_unga(d, unga, unga_lag)
    return d.sort_values(["a", "b", "year"]).reset_index(drop=True)


def country_features(d: pd.DataFrame) -> pd.DataFrame:
    """Свёртка пар на страну A (только соседи): gw, year и `COUNTRY_FEATURES`."""
    n = d[d.contig_any == 1]
    out = n.groupby(["a", "year"], as_index=False).agg(
        dy_terr_any=("terr_dispute", "max"), dy_unga_shift_max=("unga_shift", "max"), dy_power_max=("log_power_ratio", "max"))
    return out.rename(columns={"a": "gw"})


# ---------------------------------------------------------------------------
# Чтение источников и запуск
# ---------------------------------------------------------------------------


def load_capabilities() -> pd.DataFrame:
    """Страна × год: военные расходы (ВБ / SIPRI), ВВП (GMD), население (ВБ)."""
    g = load_gmd(["ISO3", "year", "nGDP_USD", "rGDP"]).rename(columns={"nGDP_USD": "gdp_usd", "rGDP": "rgdp"})
    g["gw"] = g.ISO3.map(ISO3_TO_GW)
    g = pd.concat([g, g[g.ISO3 == "SRB"].assign(gw=345)]).dropna(subset=["gw"])
    g["gw"], g["year"] = g.gw.astype(int), g.year.astype(int)
    out = g[["gw", "year", "gdp_usd", "rgdp"]]
    for name, col in (("milex_usd", "milex_usd"), ("milex_gdp", "milex_gdp"), ("population", "pop")):
        out = out.merge(load_wdi(name).rename(columns={name: col}), on=["gw", "year"], how="outer")
    return out[["gw", "year", *CAP_COLS]]


def load_neighbor_set(contig: pd.DataFrame) -> set:
    c = contig[contig.conttype <= CONTIGUITY_MAX_TYPE]
    return set(zip(c.a, c.b, c.year))


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    contig = load_contiguity()
    grid = build_grid(load_states(), contig)
    opp = opponents(load_acd(ACD_FROM, extra=("incompatibility",)))
    dw = dyad_wars(load_ged(), opp, load_neighbor_set(contig))
    d = build_features(grid, opp, dw, load_unga(), load_mid(), load_capabilities())
    return d, external_wars(dw)


def main(argv: list[str]) -> int:
    d, ext = build()
    d.to_parquet(DATA_PROCESSED / "dyad_features.parquet", index=False)
    ext.to_parquet(DATA_PROCESSED / "dyad_ext_country_year.parquet", index=False)
    f = pd.read_parquet(DATA_PROCESSED / "features_pitf_country_year.parquet")
    f = f.merge(country_features(d), on=["gw", "year"], how="left")
    f[COUNTRY_FEATURES] = f[COUNTRY_FEATURES].fillna(0.0)
    f.to_parquet(DATA_PROCESSED / "features_dyad_country_year.parquet", index=False)

    last = d[d.year == d.year.max()]
    print(f"диадная панель: {len(d)} строк, {d.groupby(['a', 'b']).ngroups} направленных пар, {d.year.min()}–{d.year.max()}; "
          f"в {d.year.max()} году: {len(last)} пар, из них близких соседей {int(last.contig_close.sum())}, "
          f"дальних по морю {int((last.contig_any - last.contig_close).sum())}, держава без границы {int((last.contig_any == 0).sum())}")
    prev = d[["a", "b", "year", "dyad_war"]].assign(year=d.year + 1).rename(columns={"dyad_war": "prev"})
    w = d.merge(prev, on=["a", "b", "year"], how="left")
    print(f"dyad_war = 1 в {int(d.dyad_war.sum())} пара-годах ({d.dyad_war.mean():.2%}); начал (после года без войны пары): "
          f"{int(((w.dyad_war == 1) & (w.prev == 0)).sum())}")
    print(f"нет данных о потенциале: {d.cap_missing.mean():.1%} строк; нет идеальных точек ООН: {d.unga_missing.mean():.1%}; "
          f"MID заморожен (после {MID_LAST_YEAR}): {d.mid_extended.mean():.1%}")
    print(d[FEATURES].describe().T[["mean", "min", "max"]].round(3).to_string())
    return 0
