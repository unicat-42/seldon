"""Этническое исключение по EPR — четвёртый столп PITF (чекпоинт 3.1).

Запуск:
    uv run python -m seldon features_epr

Читает features_vdem_country_year.parquet, добавляет колонки ниже и пишет
features_struct_country_year.parquet (история + PITF + V-Dem + EPR — все структурные наборы).
Лаг публикации: на строке T стоит статус групп за T − 1 (`core/temporal.py`); война у
соседа — за T (события UCDP известны к 1 января T+1).

    epr_excl_share        доля населения в группах без доступа к власти (POWERLESS,
                          DISCRIMINATED, SELF-EXCLUSION)
    epr_discr_share       из них — целенаправленно дискриминируемые государством
    epr_downgraded_2y     1, если за последние два года хоть одна группа спустилась по
                          ступеням власти (например, из младших партнёров в бесправные)
    epr_down_size         доля населения в таких группах
    epr_tek_kin_excl_nb   1, если группа, которая у нас во власти, в СОСЕДНЕЙ стране
                          исключена из неё (мотив «защиты своих»: Россия ↔ русские Донбасса)
    epr_tek_target_nb     доля населения в самой крупной исключённой у нас группе, чья родня
                          у соседа во власти (обратная сторона: Украина, Азербайджан, Грузия)
    epr_tek_kin_war_nb    1, если у любой нашей группы есть родня в соседней стране, на
                          территории которой в году T идёт государственная война
    epr_missing           1, если страны-года в EPR нет (значения тогда — 0)
    epr_extended          (не признак) 1, если статус продлён из последнего года EPR
"""

from __future__ import annotations

import pandas as pd

from seldon import DATA_PROCESSED
from seldon.core.temporal import publication_lag
from seldon.sources.cow import load_neighbors
from seldon.sources.epr import EPR_LAST_YEAR, EXCLUDED, INCLUDED_MIN_RANK, load_epr_group_years, load_tek

EPR_LAG = publication_lag("epr_annual", "jan1")  # 1 год
DOWNGRADE_YEARS = 2
WAR_DEATHS = 25
EPR_FEATURES = ["epr_excl_share", "epr_discr_share", "epr_downgraded_2y", "epr_down_size",
                "epr_tek_kin_excl_nb", "epr_tek_target_nb", "epr_tek_kin_war_nb", "epr_missing"]


def extend_groups(groups: pd.DataFrame, last_year: int, upto: int) -> pd.DataFrame:
    """Хроника EPR кончается в `last_year`; дальше состав и статусы групп продлеваются из
    последнего года с флагом `extended` (как Polity5 в `features_pitf`)."""
    g = groups[groups.year <= last_year].assign(extended=0)
    tail = g[g.year == last_year]
    return pd.concat([g, *(tail.assign(year=y, extended=1) for y in range(last_year + 1, upto + 1))], ignore_index=True)


def country_table(g: pd.DataFrame) -> pd.DataFrame:
    """Страна × год EPR (без лага): исключение и понижения. Строка года Y зависит только от
    статусов за Y − 2…Y."""
    # страна-годы, где этничность политически не значима (IRRELEVANT) или государства нет
    # (STATE COLLAPSE), в EPR есть: исключённых групп там ноль — это не пропуск
    present = g.groupby(["gw", "year"], as_index=False).extended.max()
    g = g[g["rank"].notna()].copy()
    g["excl"] = g["size"].where(g.status.isin(EXCLUDED), 0.0)
    g["discr"] = g["size"].where(g.status == "DISCRIMINATED", 0.0)
    prev = g[["gwgroupid", "year", "rank"]].assign(year=g.year + 1).rename(columns={"rank": "rank_prev"})
    g = g.merge(prev, on=["gwgroupid", "year"], how="left")
    g["down"] = (g["rank"] < g.rank_prev).astype(int)  # группы не было год назад → не понижение
    recent = g[["gwgroupid", "year", "down"]]
    for k in range(1, DOWNGRADE_YEARS):
        g = g.merge(recent.assign(year=recent.year + k).rename(columns={"down": f"down_{k}"}), on=["gwgroupid", "year"], how="left")
        g["down"] = g[["down", f"down_{k}"]].max(axis=1).astype(int)
    g["down_size"] = g["size"] * g.down
    agg = g.groupby(["gw", "year"], as_index=False).agg(
        epr_excl_share=("excl", "sum"), epr_discr_share=("discr", "sum"),
        epr_downgraded_2y=("down", "max"), epr_down_size=("down_size", "sum"))
    out = present.rename(columns={"extended": "epr_extended"}).merge(agg, on=["gw", "year"], how="left")
    return out.fillna({c: 0.0 for c in agg.columns if c.startswith("epr_")})


def kin_pairs(g: pd.DataFrame, tek: pd.DataFrame) -> pd.DataFrame:
    """Пары «наша группа — её родня в другой стране» по годам EPR (без лага): gw, other, year,
    patron (мы во власти, родня там исключена), target_size (мы исключены, родня там во власти)."""
    k = tek.merge(g[g["rank"].notna()], on=["gw", "gwgroupid"], how="inner")
    p = k.merge(k, on=["tekid", "year"], suffixes=("", "_o"))
    p = p[p.gw != p.gw_o]
    ours_in, theirs_in = p["rank"] >= INCLUDED_MIN_RANK, p.rank_o >= INCLUDED_MIN_RANK
    p["patron"] = (ours_in & p.status_o.isin(EXCLUDED)).astype(int)
    p["target_size"] = p["size"].where(p.status.isin(EXCLUDED) & theirs_in, 0.0)
    return p.rename(columns={"gw_o": "other"})[["gw", "other", "year", "patron", "target_size"]]


def add_epr_features(f: pd.DataFrame, groups: pd.DataFrame, tek: pd.DataFrame, neighbors: set, war_terr: pd.DataFrame,
                     lag: int = EPR_LAG, last_year: int = EPR_LAST_YEAR) -> pd.DataFrame:
    """Признаки EPR поверх панели. На строке T — статусы групп за T − `lag`; `war_terr`
    (gw, year, deaths_state) — бои на территории соседа в самом году T."""
    g = extend_groups(groups, last_year, int(f.year.max()))
    c = country_table(g)
    f = f.merge(c.assign(year=c.year + lag), on=["gw", "year"], how="left")
    f["epr_missing"] = f.epr_excl_share.isna().astype("int8")

    pairs = kin_pairs(g, tek)
    pairs = pairs.assign(year=pairs.year + lag)
    nb = pd.DataFrame(list(neighbors), columns=["gw", "other", "year"])
    pairs = pairs.merge(nb, on=["gw", "other", "year"], how="inner")  # только соседи в году T
    war = war_terr.rename(columns={"gw": "other"}).assign(kin_war=lambda d: (d.deaths_state >= WAR_DEATHS).astype(int))
    pairs = pairs.merge(war[["other", "year", "kin_war"]], on=["other", "year"], how="left").fillna({"kin_war": 0})
    agg = pairs.groupby(["gw", "year"], as_index=False).agg(
        epr_tek_kin_excl_nb=("patron", "max"), epr_tek_target_nb=("target_size", "max"), epr_tek_kin_war_nb=("kin_war", "max"))
    f = f.merge(agg, on=["gw", "year"], how="left")
    cols = [c for c in EPR_FEATURES if c != "epr_missing"]
    f[cols] = f[cols].fillna(0.0)
    f["epr_extended"] = f.epr_extended.fillna(0).astype("int8")
    return f


def build() -> pd.DataFrame:
    f = pd.read_parquet(DATA_PROCESSED / "features_vdem_country_year.parquet")
    panel = pd.read_parquet(DATA_PROCESSED / "panel_country_year.parquet", columns=["gw", "year", "deaths_state"])
    return add_epr_features(f, load_epr_group_years(), load_tek(), load_neighbors(), panel)


def main(argv: list[str]) -> int:
    f = build()
    f.to_parquet(DATA_PROCESSED / "features_struct_country_year.parquet", index=False)
    print(f"признаки EPR добавлены: {len(f)} строк → data/processed/features_struct_country_year.parquet")
    print(f"EPR нет: {f.epr_missing.mean():.1%} строк; продлён из {EPR_LAST_YEAR}: {f.epr_extended.mean():.1%}")
    print(f[EPR_FEATURES].describe().T[["mean", "min", "max"]].round(3).to_string())
    return 0
