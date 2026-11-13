"""Признаки критического замедления (CSD) в месячном контуре войн (чекпоинт 4.3b).

Запуск:
    uv run python -m seldon war_csd [3] [6] [12]

Проверяет гипотезу: растущая автокорреляция AR(1) и дисперсия мелких вспышек насилия
предвещают переход к войне `state_war25`, — и фиксирует честное ограничение: в отличие от
непрерывных валютных рядов, месячная статистика насилия `UCDP` сильно разрежена («избыток
нулей» среди мирных стран).

Все признаки строятся по `np.log1p(погибшие)` с честным лагом `EVENT_LAG = 1` месяц
(`core/temporal.py`).

Выход:
    data/processed/war_csd_predictions_h{3,6,12}.parquet
    reports/war_csd_comparison.csv
    reports/war_csd_univariate.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, FLOAT_FORMAT, REPORTS
from seldon.core.csd import ar1_delta, by_entity, rolling_ar1, var_surge
from seldon.core.entities import POST_SOVIET_GW
from seldon.core.metrics import auc, paired_comparison
from seldon.domains.conflict import model_month
from seldon.domains.conflict.live_pairs import month_ordinal
from seldon.domains.conflict.model_month import HORIZONS, SETS
from seldon.domains.conflict.month_features import EVENT_LAG, load_inputs
from seldon.sources.ucdp import CANDIDATE_LAST_MONTH, YEAR_TO

EVAL_FROM = 1996
N_BOOT = 500
WAR_CSD = [
    "war_var_surge12",
    "war_ar1_12",
    "war_ar1_d12",
    "nb_var_surge12",
    "nb_ar1_12",
    "war_csd_missing",
]


def _neighbor_deaths_series(
    p: pd.DataFrame,
    gws: list[int],
    ord_from: int,
    ord_to: int,
    own_mat: np.ndarray,
    neighbors: set[tuple[int, int, int]],
) -> np.ndarray:
    n_gw, n_t = own_mat.shape
    gw_idx = {g: i for i, g in enumerate(gws)}
    ym_grid = pd.PeriodIndex.from_ordinals(np.arange(ord_from, ord_to + 1), freq="M")
    years = sorted(set(ym_grid.year.tolist()))
    by_y: dict[int, list[tuple[int, int]]] = {y: [] for y in years}
    for a, b, y in neighbors:
        if y in by_y and a in gw_idx and b in gw_idx and a != b:
            by_y[y].append((gw_idx[a], gw_idx[b]))
    nb_adj: dict[int, np.ndarray] = {}
    for y, pairs in by_y.items():
        mat = np.zeros((n_gw, n_gw), dtype=float)
        if pairs:
            arr = np.asarray(pairs, dtype=int)
            mat[arr[:, 0], arr[:, 1]] = 1.0
        nb_adj[y] = mat

    nb_mat = np.zeros((n_gw, n_t), dtype=float)
    for t in range(n_t):
        nb_mat[:, t] = nb_adj[int(ym_grid[t].year)] @ own_mat[:, t]
    return nb_mat


def build_war_csd(
    panel_df: pd.DataFrame,
    events: pd.DataFrame,
    neighbors: set[tuple[int, int, int]],
    lag: int = EVENT_LAG,
) -> pd.DataFrame:
    """Посчитать месячные признаки критического замедления на сетке `panel_df` с лагом `lag`."""
    p = panel_df[["gw", "year", "month", "ym", "o"]].sort_values(["gw", "o"]).reset_index(drop=True)
    gws = sorted(int(g) for g in p.gw.unique())
    gw_idx = {g: i for i, g in enumerate(gws)}
    ord_from, ord_to = int(p.o.min()), int(p.o.max())
    n_gw, n_t = len(gws), ord_to - ord_from + 1

    ev = events.copy()
    ev["o"] = month_ordinal(ev.year, ev.month)
    ev_m = ev.groupby(["gw", "o"], as_index=False)["best"].sum()
    ev_m = ev_m[ev_m.gw.isin(gw_idx) & (ev_m.o >= ord_from) & (ev_m.o <= ord_to)]

    own_mat = np.zeros((n_gw, n_t), dtype=float)
    if len(ev_m):
        ii = ev_m.gw.map(gw_idx).to_numpy(dtype=int)
        tt = (ev_m.o - ord_from).to_numpy(dtype=int)
        own_mat[ii, tt] = np.maximum(ev_m.best.to_numpy(dtype=float), 0.0)

    nb_mat = _neighbor_deaths_series(p, gws, ord_from, ord_to, own_mat, neighbors)

    ii = p.gw.map(gw_idx).to_numpy(dtype=int)
    tt = (p.o - ord_from).to_numpy(dtype=int)
    p["own_v_raw"] = np.log1p(own_mat[ii, tt])
    p["nb_v_raw"] = np.log1p(nb_mat[ii, tt])
    # Строгий календарный сдвиг на `lag` месяцев внутри каждой страны
    p["own_v"] = p.groupby("gw")["own_v_raw"].shift(lag)
    p["nb_v"] = p.groupby("gw")["nb_v_raw"].shift(lag)

    p["war_var_surge12"] = by_entity(p, "gw", "own_v", var_surge, short=12, long=36, min_short=6, min_long=18, eps=1e-4)
    p["war_ar1_12"] = by_entity(p, "gw", "own_v", rolling_ar1, window=12, min_periods=8)
    p["war_ar1_d12"] = by_entity(p, "gw", "own_v", ar1_delta, short=12, long=36, min_short=8, min_long=18)
    p["nb_var_surge12"] = by_entity(p, "gw", "nb_v", var_surge, short=12, long=36, min_short=6, min_long=18, eps=1e-4)
    p["nb_ar1_12"] = by_entity(p, "gw", "nb_v", rolling_ar1, window=12, min_periods=8)
    p["own_zero12"] = p.groupby("gw")["own_v"].transform(lambda s: s.rolling(12, min_periods=6).max() == 0.0).astype("int8")

    p["war_csd_missing"] = p.war_var_surge12.isna().astype("int8")
    num = [c for c in WAR_CSD if not c.endswith("_missing")]
    p[num] = p[num].fillna(0.0)
    return p[["gw", "year", "month", "ym", "o", *WAR_CSD, "own_zero12"]].copy()


def main(argv: list[str]) -> int:
    hs = tuple(int(a) for a in argv if a in ("3", "6", "12")) or HORIZONS
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    p = pd.read_parquet(DATA_PROCESSED / "war_month_panel.parquet")
    s = load_inputs(CANDIDATE_LAST_MONTH)
    print("Расчёт CSD для месячного контура войн…")
    csd = build_war_csd(p, s["events"], s["neighbors"])
    mf = pd.read_parquet(DATA_PROCESSED / "war_month_features.parquet")
    f = mf.merge(csd[["gw", "o", *WAR_CSD, "own_zero12"]], on=["gw", "o"], how="inner")

    e_eval = f[(f.year >= EVAL_FROM) & (f.year <= YEAR_TO) & (f.quiet_true == 1)]
    print(f"Диагностика разреженности (мирные строки 1996–{YEAR_TO}, n={len(e_eval)}): "
          f"12 мес подряд 0 погибших = {e_eval.own_zero12.mean():.1%}, war_ar1_12 == 0 = {(e_eval.war_ar1_12 == 0).mean():.1%}")

    csd_sets = {
        "pitf_csd": [*SETS["pitf"], *WAR_CSD],
        "smoke_nb_csd": [*SETS["smoke_nb"], *WAR_CSD],
    }
    comp_rows, uni_rows = [], []
    for h in hs:
        print(f"\nБэктест war_csd h={h} мес…")
        pr, _ = model_month.run(f, h, csd_sets)
        base = pd.read_parquet(DATA_PROCESSED / f"war_month_predictions_h{h}.parquet", columns=["gw", "o", "p__pitf", "p__smoke_nb"])
        m = pr.merge(base, on=["gw", "o"], how="inner")
        m.to_parquet(DATA_PROCESSED / f"war_csd_predictions_h{h}.parquet", index=False)

        e = m[m.y.notna() & (m.year >= EVAL_FROM) & (m.year <= YEAR_TO)]
        cols = {n: f"p__{n}" for n in ("pitf", "pitf_csd", "smoke_nb", "smoke_nb_csd")}
        for sname, mask in (("мир", lambda d: d.quiet_true == 1), ("постсоветские", lambda d: (d.quiet_true == 1) & d.gw.isin(POST_SOVIET_GW)),
                            ("тихие ≥ 5 лет", lambda d: (d.quiet_true == 1) & (d.qmonths >= 60)), ("все строки", lambda d: d.year > 0)):
            d = e[mask(e)]
            for ref in ("pitf", "smoke_nb"):
                t = paired_comparison(d, cols, ref, n_boot=N_BOOT)
                comp_rows.append(t.assign(h=h, sample=sname, ref=ref, zero12_share=float(e_eval.own_zero12.mean())))

        u_df = e[e.quiet_true == 1][["gw", "o", "y"]].merge(csd[["gw", "o", *WAR_CSD]], on=["gw", "o"], how="inner")
        for c in [col for col in WAR_CSD if not col.endswith("_missing")]:
            uni_rows.append({"h": h, "feature": c, "share_nonzero": float((u_df[c] != 0).mean()), "auc_world": auc(u_df.y, u_df[c])})

    comp = pd.concat(comp_rows, ignore_index=True)
    comp.to_csv(REPORTS / "war_csd_comparison.csv", index=False, float_format=FLOAT_FORMAT)
    uni = pd.DataFrame(uni_rows)
    uni.to_csv(REPORTS / "war_csd_univariate.csv", index=False, float_format=FLOAT_FORMAT)

    print("\nСводка war_csd (мирные строки):")
    sub = comp[(comp.sample == "мир") & (comp.ref == "pitf")]
    for h, g in sub.groupby("h"):
        print(f"  h={h}:")
        for r in g.itertuples():
            d = "" if pd.isna(r.d_auc) else f" | ΔAUC {r.d_auc:+.3f} [{r.d_auc_lo:+.3f}; {r.d_auc_hi:+.3f}]"
            print(f"    {r.model:14s} AUC {r.auc:.3f} [{r.auc_lo:.3f}; {r.auc_hi:.3f}]  wAUC {r.auc_within_year:.3f}{d}")
    print("\nОднофакторный AUC CSD войн (h=12):")
    print(uni[uni.h == 12].round(3).to_string(index=False))
    return 0
