"""Оценка сетевого заражения конфликтов: графы (COW, EPR TEK, CEPII, V-Dem), живые и
исторические политические шоки (чекпоинт 4.3a).

Запуск:
    uv run python -m seldon compare_network
    uv run python -m seldon compare_network quick   # не пересчитывать месячные модели, взять готовые прогнозы

Собирает четыре таблицы без утечек (только прогнозы вслепую):
  1. Годовой контур (1996–2025 и срез 1996–2020, где MassMob/REIGN ещё живы):
     `pitf` (эталон) vs `pitf_net_el`, `pitf_net_cl`, `vdem_net_cl`, `pitf_polit_net` (+ `vdem`).
  2. Месячный контур (h = 3, 6, 12; 1996–2025 и живой срез 1996–2019):
     `pitf` и `smoke_nb` (эталоны из `war_month_predictions_h{h}.parquet`) vs
     `net_el`, `net_cl`, `smoke_nb_net`, `polit_own`, `polit_net`, `smoke_nb_polit_net`.
  3. Однофакторный AUC каждого сетевого/политического признака на мирных строках.
  4. Контрольные кейсы (ранг и вероятность среди мирных стран мира): Арабская весна 2011
     (Ливия, Сирия, Йемен), Грузия-2008, Украина-2014 и 2022, Карабах-2020, Судан-2023.

Выход:
    data/processed/war_network_predictions_h{3,6,12}.parquet
    reports/war_network_annual_comparison.csv
    reports/war_network_month_comparison.csv
    reports/war_network_univariate.csv
    reports/war_network_cases.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, FLOAT_FORMAT, REPORTS
from seldon.core.entities import GW_TO_SHORT, POST_SOVIET_GW
from seldon.core.metrics import auc, paired_comparison
from seldon.domains.conflict import model_month
from seldon.domains.conflict.model_month import HORIZONS, SETS
from seldon.domains.conflict.network_features import NET_CL, NET_EL, NETWORK_COLS, POLIT_NET, POLIT_OWN
from seldon.evaluation import compare_war
from seldon.sources.ucdp import YEAR_TO

EVAL_FROM = 1996
N_BOOT = 500
ANNUAL_NAMES = ["pitf", "pitf_net_el", "pitf_net_cl", "vdem", "vdem_net_cl", "pitf_polit_net"]
MONTH_NEW_SETS = {
    "net_el": [*SETS["pitf"], *NET_EL],
    "net_cl": [*SETS["pitf"], *NET_CL],
    "smoke_nb_net": [*SETS["smoke_nb"], *NET_CL],
    "polit_own": [*SETS["pitf"], *POLIT_OWN],
    "polit_net": [*SETS["pitf"], *POLIT_OWN, *POLIT_NET],
    "smoke_nb_polit_net": [*SETS["smoke_nb"], *NET_CL, *POLIT_OWN, *POLIT_NET],
}
MONTH_ALL_NAMES = ["pitf", "smoke_nb", *MONTH_NEW_SETS]
MONTH_SAMPLES = {
    "мир (1996–2025)": lambda d: d.quiet_true == 1,
    "тихие ≥ 5 лет (1996–2025)": lambda d: (d.quiet_true == 1) & (d.qmonths >= 60),
    "постсоветские (1996–2025)": lambda d: (d.quiet_true == 1) & d.gw.isin(POST_SOVIET_GW),
    "после 2014": lambda d: (d.quiet_true == 1) & (d.year >= 2015),
    "мир (1996–2019, живой MassMob/REIGN)": lambda d: (d.quiet_true == 1) & (d.year <= 2019),
    "постсоветские (1996–2019)": lambda d: (d.quiet_true == 1) & (d.year <= 2019) & d.gw.isin(POST_SOVIET_GW),
    "все строки (с продолжением)": lambda d: d.year > 0,
}
ANNUAL_SAMPLES = {
    "мир (1996–2025)": lambda d: d.at_risk == 1,
    "тихие ≥ 5 (1996–2025)": lambda d: (d.at_risk == 1) & (d.years_since_war >= 5),
    "постсоветские (1996–2025)": lambda d: (d.at_risk == 1) & d.gw.isin(POST_SOVIET_GW),
    "мир (1996–2020, живой MassMob/REIGN)": lambda d: (d.at_risk == 1) & (d.year <= 2020),
    "постсоветские (1996–2020)": lambda d: (d.at_risk == 1) & (d.year <= 2020) & d.gw.isin(POST_SOVIET_GW),
}
CASE_ORIGINS = [
    ("Ливия", 620, "2011-01"),
    ("Ливия", 620, "2011-02"),
    ("Сирия", 652, "2011-02"),
    ("Сирия", 652, "2011-03"),
    ("Сирия", 652, "2011-04"),
    ("Йемен", 678, "2011-02"),
    ("Йемен", 678, "2011-04"),
    ("Йемен", 678, "2014-08"),
    ("Грузия", 372, "2008-05"),
    ("Грузия", 372, "2008-07"),
    ("Украина", 369, "2014-01"),
    ("Украина", 369, "2014-02"),
    ("Украина", 369, "2014-03"),
    ("Украина", 369, "2021-11"),
    ("Украина", 369, "2022-01"),
    ("Армения", 371, "2020-08"),
    ("Азербайджан", 373, "2020-08"),
    ("Судан", 625, "2023-01"),
    ("Судан", 625, "2023-03"),
]
CASE_MODELS = ["pitf", "net_el", "net_cl", "polit_net", "smoke_nb", "smoke_nb_net", "smoke_nb_polit_net"]


def annual_compare(n_boot: int = N_BOOT) -> pd.DataFrame:
    pred = compare_war.load(ANNUAL_NAMES)
    t = compare_war.compare(pred, ANNUAL_NAMES, "pitf", n_boot=n_boot, samples=ANNUAL_SAMPLES)
    return t


def _load_month_inputs() -> pd.DataFrame:
    f = pd.read_parquet(DATA_PROCESSED / "war_month_features.parquet")
    net = pd.read_parquet(DATA_PROCESSED / "war_network_month.parquet", columns=["gw", "o", *NETWORK_COLS])
    return f.merge(net, on=["gw", "o"], how="inner")


def run_month_models(quick: bool = False) -> dict[int, pd.DataFrame]:
    f = None if quick else _load_month_inputs()
    out = {}
    for h in HORIZONS:
        path = DATA_PROCESSED / f"war_network_predictions_h{h}.parquet"
        if quick and path.exists():
            out[h] = pd.read_parquet(path)
            continue
        if f is None:
            f = _load_month_inputs()
        pred, _ = model_month.run(f, h, MONTH_NEW_SETS)
        base = pd.read_parquet(
            DATA_PROCESSED / f"war_month_predictions_h{h}.parquet",
            columns=["gw", "o", "p__pitf", "p__smoke_nb", "po__pitf", "po__smoke_nb"],
        )
        m = pred.merge(base, on=["gw", "o"], how="inner")
        m.to_parquet(path, index=False)
        out[h] = m
    return out


def month_compare(preds: dict[int, pd.DataFrame], n_boot: int = N_BOOT) -> pd.DataFrame:
    out = []
    cols = {n: f"p__{n}" for n in MONTH_ALL_NAMES}
    for h, p in preds.items():
        e = p[p.y.notna() & (p.year >= EVAL_FROM) & (p.year <= YEAR_TO)]
        for sample, mask in MONTH_SAMPLES.items():
            d = e[mask(e)]
            for ref in ("pitf", "smoke_nb"):
                t = paired_comparison(d, cols, ref, n_boot=n_boot)
                out.append(t.assign(h=h, sample=sample, ref=ref))
    t = pd.concat(out, ignore_index=True)
    lead = ["h", "sample", "ref", "model", "n", "positives"]
    return t[[*lead, *[c for c in t if c not in lead]]]


def univariate_table(preds: dict[int, pd.DataFrame]) -> pd.DataFrame:
    net = pd.read_parquet(DATA_PROCESSED / "war_network_month.parquet", columns=["gw", "o", *NETWORK_COLS])
    rows = []
    for h, p in preds.items():
        e = p[p.y.notna() & (p.year >= EVAL_FROM) & (p.year <= YEAR_TO) & (p.quiet_true == 1)]
        d = e[["gw", "o", "year", "y"]].merge(net, on=["gw", "o"], how="inner")
        d19 = d[d.year <= 2019]
        for c in NETWORK_COLS:
            if c.endswith("_missing"):
                continue
            rows.append({
                "h": h, "feature": c,
                "share_nonzero_25": float((d[c] != 0).mean()),
                "auc_world_25": auc(d.y, d[c]),
                "share_nonzero_19": float((d19[c] != 0).mean()),
                "auc_world_19": auc(d19.y, d19[c]),
            })
    return pd.DataFrame(rows)


def cases_table(preds: dict[int, pd.DataFrame]) -> pd.DataFrame:
    net = pd.read_parquet(DATA_PROCESSED / "war_network_month.parquet", columns=["gw", "o", "net_el_war12", "net_cl_shock3", "net_cl_prot12", "own_prot3"])
    rows = []
    for label, gw, ym_str in CASE_ORIGINS:
        M = pd.Period(ym_str, freq="M")
        for h in (3, 12):
            p = preds[h].merge(net, on=["gw", "o"], how="left")
            r = p[(p.gw == gw) & (p.ym == M)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            quiet = r.quiet_true == 1
            peers = p[(p.ym == M) & ((p.quiet_true == 1) if quiet else True)]
            row = {
                "case": label, "gw": gw, "country": GW_TO_SHORT.get(gw, r.country),
                "origin": ym_str, "h": h,
                "status": "мирная" if quiet else f"уже воюет (q={int(r.qmonths)}м)",
                "n_peers": len(peers),
                "war_in_window": r.y,
                "net_el_war12": float(r.net_el_war12),
                "net_cl_shock3": float(r.net_cl_shock3),
                "net_cl_prot12": float(r.net_cl_prot12),
                "own_prot3": float(r.own_prot3),
            }
            for m in CASE_MODELS:
                v = float(r[f"p__{m}"])
                row[f"p_{m}"] = v
                row[f"rank_{m}"] = int((peers[f"p__{m}"] > v).sum()) + 1
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    quick = "quick" in argv
    REPORTS.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("1/4 Годовой контур: сетевые наборы против `pitf`…")
    ann = annual_compare()
    ann.to_csv(REPORTS / "war_network_annual_comparison.csv", index=False, float_format=FLOAT_FORMAT)
    print(compare_war.show(ann))

    print("\n2/4 Месячный контур (h = 3, 6, 12)…")
    preds = run_month_models(quick=quick)
    mon = month_compare(preds)
    mon.to_csv(REPORTS / "war_network_month_comparison.csv", index=False, float_format=FLOAT_FORMAT)
    for (h, sample, ref), g in mon[mon.ref == "pitf"].groupby(["h", "sample", "ref"], sort=False):
        if "после" in sample or "тихие" in sample:
            continue
        print(f"  h={h} | {sample} (ref={ref}, n={int(g.n.iloc[0])}, pos={int(g.positives.iloc[0])}):")
        for r in g.itertuples():
            d = "" if pd.isna(r.d_auc) else f" | ΔAUC {r.d_auc:+.3f} [{r.d_auc_lo:+.3f}; {r.d_auc_hi:+.3f}]"
            print(f"    {r.model:20s} AUC {r.auc:.3f} [{r.auc_lo:.3f}; {r.auc_hi:.3f}]  wAUC {r.auc_within_year:.3f}  calib {r.calib:.2f}{d}")

    print("\n3/4 Однофакторный AUC сетевых признаков…")
    uni = univariate_table(preds)
    uni.to_csv(REPORTS / "war_network_univariate.csv", index=False, float_format=FLOAT_FORMAT)
    print(uni[uni.h == 12][["feature", "share_nonzero_25", "auc_world_25", "share_nonzero_19", "auc_world_19"]].round(3).to_string(index=False))

    print("\n4/4 Контрольные кейсы (место среди мирных стран мира, h=12)…")
    cs = cases_table(preds)
    cs.to_csv(REPORTS / "war_network_cases.csv", index=False, float_format=FLOAT_FORMAT)
    cols = ["case", "origin", "status", "war_in_window", "rank_pitf", "rank_net_cl", "rank_polit_net", "rank_smoke_nb", "rank_smoke_nb_net", "p_pitf", "p_net_cl", "p_smoke_nb_net"]
    print(cs[cs.h == 12][cols].round(3).to_string(index=False))
    return 0
