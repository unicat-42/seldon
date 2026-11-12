"""Признаки истории конфликтов на годовой панели — аналог модели `hist` conflictforecast.

Запуск:
    uv run python -m seldon.features [целевая]     # по умолчанию state_war25

Читает data/processed/panel_country_year.parquet, пишет features_country_year.parquet
(для целевой не по умолчанию — features_country_year_<целевая>.parquet).
Все признаки на строке (страна, год T) считаются ТОЛЬКО по данным до года T включительно —
их можно использовать для прогноза на T+1 без утечки будущего.

Признаки (история считается по целевой, по умолчанию state_war25 — см. panel.py):
    war_now            целевая в году T (0/1)
    years_since_war    лет с последнего года с целевой == 1 (0 — воюет сейчас; если не воевала
                       с 1989 — лет с начала панели, помечено флагом never_war)
    never_war          1, если с 1989 (или с появления страны) не воевала ни разу
    log_deaths_1/5/10  log(1 + погибшие за последний год / 5 лет / 10 лет). Погибшие —
                       `deaths_war`: государственный конфликт на своей территории плюс у
                       соседей, где страна — сторона. Каждое событие считается один раз
                       (до аудита складывались «территория» и «участие», и свои войны
                       Украины, Грузии, Азербайджана считались дважды). Для старых
                       целевых (war25…) — как раньше, deaths_best + deaths_party
    war_years_10       сколько из последних 10 лет страна воевала
    decay_1/3          затухающая память о целевой с полураспадом 1 и 3 года (как в baseline)
    nb_war_share       доля соседей с войной НА СВОЕЙ территории в году T (нет соседей → 0)
    nb_war_any         1, если хотя бы один сосед воюет
    nb_onset_any       1, если у кого-то из соседей в году T началась война
    nb_log_deaths_1    log(1 + погибшие на территории всех соседей за год T)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED
from seldon.panel import ONSET_OF, load_neighbors

DEFAULT_TARGET = "state_war25"
OLD_TARGETS = {"conflict25", "war25", "war1000"}  # история «как до аудита»: с двойным счётом погибших


def features_file(target: str = DEFAULT_TARGET) -> str:
    return "features_country_year.parquet" if target == DEFAULT_TARGET else f"features_country_year_{target}.parquet"


def _rolling_sum(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).sum()


def own_history(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    p = panel.sort_values(["gw", "year"]).copy()
    p["deaths_any"] = (p.deaths_best + p.deaths_party) if target in OLD_TARGETS else p.deaths_war
    g = p.groupby("gw")
    p["war_now"] = p[target].astype("int8")
    # лет с последней войны: номер года минус последний год с войной (forward fill)
    last_war_year = p["year"].where(p[target] == 1)
    p["last_war_year"] = last_war_year.groupby(p["gw"]).ffill()
    first_year = g["year"].transform("min")
    p["never_war"] = p["last_war_year"].isna().astype("int8")
    p["years_since_war"] = (p["year"] - p["last_war_year"].fillna(first_year - 1)).astype(int)
    for w in (1, 5, 10):
        p[f"log_deaths_{w}"] = np.log1p(g["deaths_any"].transform(lambda s: _rolling_sum(s, w)))
    p["war_years_10"] = g[target].transform(lambda s: _rolling_sum(s, 10)).astype(int)
    for h in (1, 3):
        alpha = 1 - 0.5 ** (1 / h)
        p[f"decay_{h}"] = g[target].transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    return p.drop(columns=["last_war_year"])


def neighbor_features(p: pd.DataFrame, neighbors: set, target: str) -> pd.DataFrame:
    """Для каждой (страна, год): что происходит у соседей в этом же году.

    Соседская война считается строго по боям НА ТЕРРИТОРИИ самого соседа: погибшие —
    `deaths_state`, «сосед воюет» — `deaths_state ≥ 25`, начало — то же после тихого года.
    До правки 12.11.2026 брались `deaths_war` и `state_war25` соседа, куда входят его потери
    как стороны на ЧУЖОЙ территории: бои в Украине через Россию «перетекали» вторым шагом
    ко всем соседям России — Китаю по Амуру и США через Берингов пролив.
    Для старых целевых (war25…) — как раньше, по целевой соседа и deaths_any."""
    nb = pd.DataFrame(list(neighbors), columns=["gw", "nb", "year"])
    if target in OLD_TARGETS:
        state = p[["gw", "year", target, ONSET_OF[target], "deaths_any"]].rename(
            columns={"gw": "nb", target: "nb_war", ONSET_OF[target]: "nb_onset", "deaths_any": "nb_deaths"})
    else:
        q = p.sort_values(["gw", "year"])
        terr = (q.deaths_state >= 25).astype(int)
        prev = terr.groupby(q.gw).shift(1).fillna(0)
        state = pd.DataFrame({"nb": q.gw, "year": q.year, "nb_war": terr,
                              "nb_onset": ((terr == 1) & (prev == 0)).astype(int), "nb_deaths": q.deaths_state})
    nb = nb.merge(state, on=["nb", "year"], how="inner")  # соседи, которых нет в панели, выпадают
    agg = nb.groupby(["gw", "year"]).agg(
        nb_war_share=("nb_war", "mean"),
        nb_war_any=("nb_war", "max"),
        nb_onset_any=("nb_onset", "max"),
        nb_deaths=("nb_deaths", "sum"),
    ).reset_index()
    out = p.merge(agg, on=["gw", "year"], how="left")
    for c in ["nb_war_share", "nb_war_any", "nb_onset_any", "nb_deaths"]:
        out[c] = out[c].fillna(0)
    out["nb_log_deaths_1"] = np.log1p(out.pop("nb_deaths"))
    return out


FEATURES = [
    "war_now", "years_since_war", "never_war", "log_deaths_1", "log_deaths_5", "log_deaths_10",
    "war_years_10", "decay_1", "decay_3", "nb_war_share", "nb_war_any", "nb_onset_any", "nb_log_deaths_1",
]


def build_features(panel: pd.DataFrame, neighbors: set, target: str = DEFAULT_TARGET) -> pd.DataFrame:
    """Признаки истории. Чистая функция: строка года T зависит только от панели ≤ T."""
    p = neighbor_features(own_history(panel, target), neighbors, target)
    targets = [c for pair in ONSET_OF.items() for c in pair]
    keep = ["gw", "country", "year", *targets, *FEATURES]
    return p[keep].sort_values(["gw", "year"]).reset_index(drop=True)


def build(target: str = DEFAULT_TARGET) -> pd.DataFrame:
    panel = pd.read_parquet(DATA_PROCESSED / "panel_country_year.parquet")
    return build_features(panel, load_neighbors(), target)


def main(argv: list[str]) -> int:
    target = argv[0] if argv else DEFAULT_TARGET
    f = build(target)
    f.to_parquet(DATA_PROCESSED / features_file(target), index=False)
    print(f"признаки ({target}): {len(f)} строк × {len(FEATURES)} признаков → data/processed/{features_file(target)}")
    print(f[FEATURES].describe().T[["mean", "min", "max"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
