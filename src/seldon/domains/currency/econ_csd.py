"""Критическое замедление (CSD) и компактные живые наборы `live_fx` в месячном валютном
контуре `deval15` (чекпоинты 4.3b и 4.3c).

Запуск:
    uv run python -m seldon econ_csd [3] [6] [12]

Проверяет:
  1. Добавляет ли CSD (`csd`, `csd_regime`) сигнал к боевому набору `ref`/`base` и к
     кандидату `live_fx`, и работает ли перенос флуктуаций с зажатого курса в резервы
     на привязках (`peg_share12 >= 0.8`) по сравнению со свободным плаванием (`< 0.3`).
  2. Насколько компактный 100 % живой набор `live_fx` (= KLR + `peg_share12` + `reer_gap` +
     `reer_peg` + `peg_resfall`) держит качество `all_brk` без замороженных архивов (`IRR`),
     без артефакта `gg_missing` (ВБ IDS) и без задержек торговли (`IMTS`); нужен ли линейный
     `reserves_g12` рядом с `peg_resfall` (`live_fx` vs `live_fx_swap`).

Выход:
    data/processed/econ_csd_predictions_h{3,6,12}.parquet
    reports/fx_csd_comparison.csv
    reports/fx_csd_univariate.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, FLOAT_FORMAT, REPORTS
from seldon.core.entities import POST_SOVIET_ISO3
from seldon.core.metrics import auc, paired_comparison
from seldon.core.temporal import label_known
from seldon.domains.currency import contagion_features as cf
from seldon.domains.currency import csd_features, econ_contagion, econ_kr, struct_features
from seldon.domains.currency.csd_features import CSD_RAW, CSD_REGIME, LIVE_FX_ADD, LIVE_FX_CTOT_ADD
from seldon.domains.currency.econ_struct import KR_C_REF, KR_MIN_POS, KR_TARGET, _kr_design, _kr_fit, choose_C_kr
from seldon.domains.currency.kr_features import INDICATORS
from seldon.evaluation.backtest import backtest
from seldon.sources.gmd import load_gmd

HORIZONS = (3, 6, 12)
N_BOOT = 500
SETS: dict[str, list[str]] = {
    "ref": [],
    "base": [],
    "csd": CSD_RAW,
    "csd_regime": CSD_REGIME,
    "live_fx": LIVE_FX_ADD,
    "live_fx_swap": LIVE_FX_ADD,
    "live_fx_ctot": LIVE_FX_CTOT_ADD,
    "live_fx_csd": [*LIVE_FX_ADD, *CSD_REGIME],
    "all_brk_csd": [*econ_contagion.SETS["all_brk"], *CSD_REGIME],
}
DROP: dict[str, tuple[str, ...]] = {
    "live_fx_swap": ("reserves_g12",),
}
NAMES = list(SETS)


def features(name: str) -> list[str]:
    drop = DROP.get(name, ())
    return [*[i for i in INDICATORS if i not in drop], *SETS[name]]


def build_panel_from(raw, annual, war, s: dict, c: dict) -> pd.DataFrame:
    """Сборка панели из переданных таблиц (для чистых тестов без обращения к диску)."""
    st = struct_features.build_struct(s["fx"], s["eur"], s["reer"], s["ctot"], s["irr"], s["buffers"])
    ct = cf.build_contagion(s["fx"], s["eur"], c["trade"], st, c["parallel"])
    p = econ_contagion.build_panel(st, ct, c["gdp"], raw=raw, annual_cpi=annual, war=war, ctot=s["ctot"])
    return csd_features.build_fx_csd(p)


def load_full_panel(last_month: str = struct_features.LAST_MONTH) -> pd.DataFrame:
    """Собрать полную панель на реальных данных по `last_month`."""
    if last_month == struct_features.LAST_MONTH and (DATA_PROCESSED / "fx_struct_monthly.parquet").exists():
        st = pd.read_parquet(DATA_PROCESSED / "fx_struct_monthly.parquet")
        st["month"] = pd.PeriodIndex(st.month, freq="M")
        ct = pd.read_parquet(DATA_PROCESSED / "fx_contagion_monthly.parquet")
        ct["month"] = pd.PeriodIndex(ct.month, freq="M")
        gdp = econ_contagion.gdp_per_capita(load_gmd())
        p = econ_contagion.build_panel(st, ct, gdp)
        return csd_features.build_fx_csd(p)

    # Расширенный срез на текущий край (например `2026-10`)
    st = struct_features.load_struct(last_month=last_month, filled=True)
    # `econ_kr.build_panel` + `peg_resfall` без тяжелого паркета IMTS 1948–2025, если `live_fx`
    raw = econ_kr.load_panel(last_month)
    p = econ_kr.build_panel(raw, econ_contagion.load_annual_cpi(), econ_contagion.load_war(), target=KR_TARGET)
    p = struct_features.attach(p, st, ["iso3", "month"])
    p["peg_resfall"] = p.peg_share12.fillna(0.0) * (-p.reserves_g12).clip(lower=0.0).fillna(0.0)
    return csd_features.build_fx_csd(p)


def _predict(p: pd.DataFrame, as_of: pd.Period, test: pd.DataFrame, name: str, h: int) -> tuple[np.ndarray, float] | None:
    feats = features(name)
    known = p[label_known(p.month, as_of, h) & p.evaluable]
    if known.y.sum() < KR_MIN_POS:
        return None
    C = KR_C_REF if name == "ref" else choose_C_kr(known, feats, h=h)
    m, lo, hi, med = _kr_fit(known, feats, C)
    return m.predict_proba(_kr_design(test, feats, lo, hi, med))[:, 1], C


def forecast_asof(p: pd.DataFrame, as_of: pd.Period, name: str = "live_fx", h: int = 12) -> pd.DataFrame:
    test = p[(p.month == as_of) & p.own_currency & p.fx.notna()].copy()
    out = _predict(p, as_of, test, name, h)
    test["logit"], test["C"] = (np.nan, np.nan) if out is None else out
    return test


def run(p: pd.DataFrame, h: int, names: list[str] = NAMES) -> pd.DataFrame:
    def step(T: int) -> pd.DataFrame | None:
        as_of = pd.Period(f"{T - 1}-12", freq="M")
        test = p[(p.month.dt.year == T) & p.own_currency & p.fx.notna()].copy()
        if not test.evaluable.any():
            return None
        for name in names:
            out = _predict(p, as_of, test, name, h)
            if out is None:
                return None
            test[f"logit__{name}"], test[f"C__{name}"] = out
        print(f"  CSD h={h} {T}: строк {int(test.evaluable.sum())}, кризисных окон {int(test.y[test.evaluable].sum())}")
        return test

    return backtest(range(econ_kr.FIRST_EVAL_YEAR, econ_kr.LAST_EVAL_YEAR + 1), step)


def univariate_csd(p: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    regimes = {
        "мир": lambda d: d.iso3.notna(),
        "постсоветские": lambda d: d.iso3.isin(POST_SOVIET_ISO3),
        "привязки (peg ≥ 0.8)": lambda d: d.peg_share12 >= 0.8,
        "ползущие (0.3–0.8)": lambda d: (d.peg_share12 >= 0.3) & (d.peg_share12 < 0.8),
        "плавающие (peg < 0.3)": lambda d: d.peg_share12 < 0.3,
    }
    for h in horizons:
        ph = econ_contagion.with_horizon(p, h)
        d = ph[ph.evaluable & (ph.month.dt.year >= econ_kr.FIRST_EVAL_YEAR)]
        for f in [c for c in CSD_REGIME if not c.endswith("_missing")]:
            flag = "csd_res_missing" if "res" in f else "csd_fx_missing"
            sub_valid = d[d[flag] == 0]
            for sname, mask in regimes.items():
                s = sub_valid[mask(sub_valid)]
                if s.y.nunique() < 2:
                    continue
                rows.append({
                    "h": h, "feature": f, "sample": sname,
                    "n": len(s), "positives": int(s.y.sum()),
                    "auc": auc(s.y, s[f]),
                })
    return pd.DataFrame(rows)


def compare_predictions(preds: dict[int, pd.DataFrame], n_boot: int = N_BOOT) -> pd.DataFrame:
    out = []
    regimes = {
        "мир": lambda d: d.iso3.notna(),
        "постсоветские": lambda d: d.post_soviet.astype(bool),
        "привязки (peg ≥ 0.8)": lambda d: d.peg_share12 >= 0.8,
        "плавающие (peg < 0.3)": lambda d: d.peg_share12 < 0.3,
    }
    for h, pr in preds.items():
        d0 = pr[pr.evaluable & (pr.month.dt.year >= econ_kr.FIRST_EVAL_YEAR)].copy()
        d0["year"] = d0.month.dt.year
        avail = [c.removeprefix("logit__") for c in d0.columns if c.startswith("logit__")]
        cols = {n: f"logit__{n}" for n in avail}
        for sname, mask in regimes.items():
            d = d0[mask(d0)]
            if d.y.nunique() < 2:
                continue
            for ref in ("ref", "live_fx"):
                if ref not in cols:
                    continue
                t = paired_comparison(d, cols, ref, n_boot=n_boot)
                out.append(t.assign(h=h, sample=sname, ref=ref))
    t = pd.concat(out, ignore_index=True)
    lead = ["h", "sample", "ref", "model", "n", "positives"]
    return t[[*lead, *[c for c in t if c not in lead]]]


def main(argv: list[str]) -> int:
    hs = tuple(int(a) for a in argv if a in ("3", "6", "12")) or HORIZONS
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("Сборка панели FX + CSD…")
    p = load_full_panel()

    preds = {}
    for h in hs:
        print(f"\nБэктест CSD / live_fx на h={h} мес…")
        ph = econ_contagion.with_horizon(p, h)
        pr = run(ph, h)
        # Подклеиваем эталонные наборы 4.1 (`struct`, `res_both`, `res_swap`, `all_brk`, `all_brk_swap`)
        cpath = DATA_PROCESSED / f"econ_contagion_predictions_h{h}.parquet"
        if cpath.exists():
            extra_cols = ["iso3", "month", "logit__struct", "logit__res_both", "logit__res_swap", "logit__all_brk", "logit__all_brk_swap"]
            cpred = pd.read_parquet(cpath, columns=extra_cols)
            cpred["month"] = pd.PeriodIndex(cpred.month, freq="M")
            pr = pr.merge(cpred, on=["iso3", "month"], how="left")
        pr_save = pr.copy()
        pr_save["month"] = pr_save["month"].astype(str)
        pr_save.to_parquet(DATA_PROCESSED / f"econ_csd_predictions_h{h}.parquet", index=False)
        preds[h] = pr

    print("\nПарные сравнения…")
    comp = compare_predictions(preds)
    comp.to_csv(REPORTS / "fx_csd_comparison.csv", index=False, float_format=FLOAT_FORMAT)
    for (h, sample), g in comp[(comp["ref"] == "ref") & (comp["sample"].isin(["мир", "постсоветские"]))].groupby(["h", "sample"], sort=False):
        print(f"  h={h} | {sample} (n={int(g.n.iloc[0])}, pos={int(g.positives.iloc[0])}):")
        for r in g.itertuples():
            d = "" if pd.isna(r.d_auc) else f" | ΔAUC {r.d_auc:+.3f} [{r.d_auc_lo:+.3f}; {r.d_auc_hi:+.3f}]"
            print(f"    {r.model:16s} AUC {r.auc:.3f} [{r.auc_lo:.3f}; {r.auc_hi:.3f}]  wAUC {r.auc_within_year:.3f}  calib {r.calib:.2f}{d}")

    print("\nОднофакторный AUC признаков CSD (проверка переноса флуктуаций):")
    uni = univariate_csd(p, hs)
    uni.to_csv(REPORTS / "fx_csd_univariate.csv", index=False, float_format=FLOAT_FORMAT)
    piv = uni[uni.h == 12].pivot(index="feature", columns="sample", values="auc").round(3)
    print(piv.to_string())
    return 0
