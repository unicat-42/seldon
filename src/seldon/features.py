"""Признаки истории конфликтов на годовой панели — аналог модели `hist` conflictforecast.

Запуск:
    uv run python -m seldon.features

Читает data/processed/panel_country_year.parquet, пишет features_country_year.parquet.
Все признаки на строке (страна, год T) считаются ТОЛЬКО по данным до года T включительно —
их можно использовать для прогноза на T+1 без утечки будущего.

Признаки (TARGET = war25 — «страна воюет», см. panel.py):
    war_now            war25 в году T (0/1)
    years_since_war    лет с последнего года с war25 == 1 (0 — воюет сейчас; если не воевала
                       с 1989 — лет с начала панели, помечено флагом never_war)
    never_war          1, если с 1989 (или с появления страны) не воевала ни разу
    log_deaths_1/5/10  log(1 + погибшие за последний год / 5 лет / 10 лет), погибшие —
                       deaths_best + deaths_party (территория + участие)
    war_years_10       сколько из последних 10 лет страна воевала
    decay_1/3          затухающая память о war25 с полураспадом 1 и 3 года (как в baseline)
    nb_war_share       доля соседей, воюющих в году T (нет соседей → 0)
    nb_war_any         1, если хотя бы один сосед воюет
    nb_onset_any       1, если у кого-то из соседей в году T началась война
    nb_log_deaths_1    log(1 + погибшие у всех соседей за год T)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED
from seldon.panel import load_neighbors

TARGET = "war25"
DEATHS = ["deaths_best", "deaths_party"]


def _rolling_sum(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).sum()


def own_history(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["gw", "year"]).copy()
    p["deaths_any"] = p[DEATHS].sum(axis=1)
    g = p.groupby("gw")
    p["war_now"] = p[TARGET].astype("int8")
    # лет с последней войны: номер года минус последний год с войной (forward fill)
    last_war_year = p["year"].where(p[TARGET] == 1)
    p["last_war_year"] = last_war_year.groupby(p["gw"]).ffill()
    first_year = g["year"].transform("min")
    p["never_war"] = p["last_war_year"].isna().astype("int8")
    p["years_since_war"] = (p["year"] - p["last_war_year"].fillna(first_year - 1)).astype(int)
    for w in (1, 5, 10):
        p[f"log_deaths_{w}"] = np.log1p(g["deaths_any"].transform(lambda s: _rolling_sum(s, w)))
    p["war_years_10"] = g[TARGET].transform(lambda s: _rolling_sum(s, 10)).astype(int)
    for h in (1, 3):
        alpha = 1 - 0.5 ** (1 / h)
        p[f"decay_{h}"] = g[TARGET].transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    return p.drop(columns=["last_war_year"])


def neighbor_features(p: pd.DataFrame) -> pd.DataFrame:
    """Для каждой (страна, год): что происходит у соседей в этом же году."""
    nb = pd.DataFrame(list(load_neighbors()), columns=["gw", "nb", "year"])
    onset_col = "onset_war25" if TARGET == "war25" else "onset25"
    state = p[["gw", "year", TARGET, onset_col, "deaths_any"]].rename(
        columns={"gw": "nb", TARGET: "nb_war", onset_col: "nb_onset", "deaths_any": "nb_deaths"}
    )
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


def build() -> pd.DataFrame:
    panel = pd.read_parquet(DATA_PROCESSED / "panel_country_year.parquet")
    p = neighbor_features(own_history(panel))
    keep = ["gw", "country", "year", "conflict25", "onset25", "war25", "onset_war25", "war1000", "onset_war1000", *FEATURES]
    return p[keep].sort_values(["gw", "year"]).reset_index(drop=True)


def main() -> None:
    f = build()
    f.to_parquet(DATA_PROCESSED / "features_country_year.parquet", index=False)
    print(f"признаки: {len(f)} строк × {len(FEATURES)} признаков → data/processed/features_country_year.parquet")
    print(f[FEATURES].describe().T[["mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
