"""Таблица риска на 2025–26: война, валютный кризис, банковский кризис — по странам.

Запуск:
    uv run python -m seldon.risk_table

Три модели проекта, обученные на всём, что известно, и применённые к последним данным:
  1. Война (`model_hist pitf`, логрегрессия): признаки 2024 → онсет войны ≥ 25 / ≥ 1000
     погибших в 2025. UCDP есть по 2024, поэтому 2026 не считаем.
  2. Валютный кризис (`econ_kr`): признаки последнего месяца с данными МВФ по стране
     (обычно 2026-09) → EMP-кризис в ближайшие 12 месяцев. Три модели: `pressure`
     (персистентность, лучшая вслепую), `signals` (KLR), `logit`.
  3. Банковский кризис (`econ_boe_em`): признаки 2024 → кризис в 2025–26 (надёжно) и
     признаки 2025 → 2026–27 (предварительно: в GMD за 2025 могут быть оценки). Деревья
     BoE с JST (`jst_extree`) и объединённые (`pooled_extree`), наборы `full`, `core` и
     `nomoney` (последний — чтобы были Россия и еврозона, у которых нет M3).

Обучение «на всём известном»: валютная модель — на окнах, закрывшихся к последнему
месяцу; банковская — своя часть на годах ≤ 2023 (кризисы 2024–25 в GMD уже проставлены);
война — на парах по 2023 → 2024.

Выход:
    reports/risk_2025_26.csv — все страны;
    печать: постсоветские и топ-15 мира по каждой модели.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, PROJECT_ROOT
from seldon import econ_boe_em, econ_kr
from seldon.countries import GW_TO_ISO3, ISO3_TO_SHORT, POST_SOVIET_ISO3
from seldon.econ_boe import JST_R6, build_boe_dataset, load_jst
from seldon.features import FEATURES as HIST_FEATURES
from seldon.features_pitf import PITF_FEATURES

REPORTS = PROJECT_ROOT / "reports"
FX_LAST_MONTH = "2026-10"
FX_MIN_AS_OF = "2026-04"  # страны, у которых курс кончается раньше, в таблицу не берём
BANK_YEARS = (2024, 2025)


# ---------------------------------------------------------------------------
# 1. Война
# ---------------------------------------------------------------------------


def war_risk() -> pd.DataFrame:
    f = pd.read_parquet(DATA_PROCESSED / "features_pitf_country_year.parquet")
    feats = [*HIST_FEATURES, *PITF_FEATURES]
    last = f.year.max()
    out = f[f.year == last][["gw", "country", "war25", "war1000"]].copy()
    for target in ("war25", "war1000"):
        nxt = f[["gw", "year", target]].assign(year=lambda d: d.year - 1).rename(columns={target: "y"})
        d = f.merge(nxt, on=["gw", "year"], how="inner")
        train = d[d.y.notna() & (d.year <= last - 1)]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(train[feats], train.y.astype(int))
        test = f[f.year == last]
        out[f"p_{target}_{last + 1}"] = m.predict_proba(test[feats])[:, 1]
        at_risk = out[out[target] == 0]
        out[f"rank_{target}_at_risk"] = np.nan
        out.loc[at_risk.index, f"rank_{target}_at_risk"] = at_risk[f"p_{target}_{last + 1}"].rank(ascending=False, method="min")
    out["iso3"] = out.gw.map(GW_TO_ISO3)
    out = out.rename(columns={"war25": "war25_now", "war1000": "war1000_now", "country": "country_ucdp"})
    out["war_features_year"] = last
    return out[out.iso3.notna()].drop(columns="gw")


# ---------------------------------------------------------------------------
# 2. Валютный кризис
# ---------------------------------------------------------------------------


def fx_risk() -> pd.DataFrame:
    econ_kr.LAST_MONTH = FX_LAST_MONTH
    p = econ_kr.load_panel()
    p = econ_kr.fill_cpi(p)
    p = econ_kr.add_emp_and_crises(p)
    p = econ_kr.add_indicators(p)
    p = econ_kr.add_target(p, "crisis")
    feats = list(econ_kr.INDICATORS)
    known = p[p.evaluable & p.y.notna()]
    # прогноз — последний месяц с курсом по каждой стране
    last = p[p.fx.notna()].groupby("iso3").month.max()
    test = p.merge(last.rename("as_of"), on="iso3")
    test = test[(test.month == test.as_of) & (test.as_of >= FX_MIN_AS_OF)].copy()
    # для стран в «тишине после кризиса» прогноз тоже считаем, но помечаем
    recent = p.groupby("iso3").crisis.apply(lambda s: s.tail(12).max())
    test["crisis_last_12m"] = test.iso3.map(recent).fillna(0).astype(int)

    sig = econ_kr.fit_signals(known)
    s_tr = econ_kr.composite(known, sig, p)
    s_te = econ_kr.composite(test, sig, p)
    test["fx_signals"] = econ_kr.calibrate(s_tr, known.y, s_te).to_numpy()
    test["fx_pressure"] = econ_kr.calibrate(known.emp_z_max3.fillna(0), known.y, test.emp_z_max3.fillna(0)).to_numpy()
    tr = known[feats]
    lo, hi = tr.quantile(0.01), tr.quantile(0.99)
    med = tr.median().fillna(0.0)
    logit = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    logit.fit(tr.clip(lo, hi, axis=1).fillna(med), known.y)
    test["fx_logit"] = logit.predict_proba(test[feats].clip(lo, hi, axis=1).fillna(med))[:, 1]
    for m in ("fx_pressure", "fx_signals", "fx_logit"):
        test[f"rank_{m}"] = test[m].rank(ascending=False, method="min")
    test["fx_base_rate"] = known.y.mean()
    test["signals_fired"] = [
        ",".join(ind for ind in sig if econ_kr._signal(test.loc[[i]], ind, econ_kr.INDICATORS[ind], sig[ind]["pct"], p).iloc[0] == 1)
        for i in test.index
    ]
    cols = ["iso3", "as_of", "fx_pressure", "fx_signals", "fx_logit", "rank_fx_pressure", "rank_fx_signals",
            "rank_fx_logit", "fx_base_rate", "emp_z_max3", "deval12", "reserves_g12", "infl12", "crisis_last_12m",
            "signals_fired"]
    out = test[cols].copy()
    out["as_of"] = out.as_of.astype(str)
    return out


# ---------------------------------------------------------------------------
# 3. Банковский кризис
# ---------------------------------------------------------------------------


def bank_risk() -> pd.DataFrame:
    econ_boe_em.LAST_YEAR = BANK_YEARS[-1]
    panel = econ_boe_em.load_panel()
    jst = load_jst(JST_R6, "Sheet1")
    frames: dict[str, list[pd.DataFrame]] = {}
    for tag, feats in econ_boe_em.FEATURE_SETS.items():
        data = econ_boe_em.build(panel, feats)
        jst_d = build_boe_dataset(jst, predictors=feats)
        train_own = data[data.year <= 2023]  # кризисы 2024–25 в GMD известны
        lo, hi = jst_d[feats].min(), jst_d[feats].max()
        q1, q99 = train_own[feats].quantile(0.01), train_own[feats].quantile(0.99)
        models = econ_boe_em.make_models()
        jst_et = models["extree"].fit(jst_d[feats].values, jst_d.crisis.values)
        pooled = pd.concat([jst_d, train_own])
        pooled_et = econ_boe_em.make_models()["extree"].fit(pooled[feats].clip(q1, q99, axis=1).values, pooled.crisis.values)
        own_lr = econ_boe_em.make_models()["logreg"].fit(train_own[feats].clip(q1, q99, axis=1).values, train_own.crisis.values)
        for y in BANK_YEARS:
            t = data[data.year == y].copy()
            if t.empty:
                continue
            t[f"bank_jst_extree_{tag}"] = jst_et.predict_proba(t[feats].clip(lo, hi, axis=1).values)[:, 1]
            t[f"bank_pooled_extree_{tag}"] = pooled_et.predict_proba(t[feats].clip(q1, q99, axis=1).values)[:, 1]
            t[f"bank_own_logreg_{tag}"] = own_lr.predict_proba(t[feats].clip(q1, q99, axis=1).values)[:, 1]
            t["bank_features_year"] = y
            keep = [f"bank_jst_extree_{tag}", f"bank_pooled_extree_{tag}", f"bank_own_logreg_{tag}"]
            if tag == "nomoney":  # признаки кредита одни и те же во всех наборах, берём из самого широкого
                keep += ["tloan_gdp_rdiff2", "global_loan2"]
            frames.setdefault(tag, []).append(t.set_index(["iso", "bank_features_year"])[keep])
    out = pd.concat([pd.concat(v) for v in frames.values()], axis=1).reset_index().rename(columns={"iso": "iso3"})
    for c in [c for c in out.columns if c.startswith("bank_") and c != "bank_features_year"]:
        out[f"rank_{c}"] = out.groupby("bank_features_year")[c].rank(ascending=False, method="min")
    return out


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    print("1/3 война…")
    war = war_risk()
    print("2/3 валютный кризис…")
    fx = fx_risk()
    print("3/3 банковский кризис…")
    bank = bank_risk()
    bank24 = bank[bank.bank_features_year == 2024].drop(columns="bank_features_year").add_suffix("_2025_26").rename(columns={"iso3_2025_26": "iso3"})
    bank25 = bank[bank.bank_features_year == 2025].drop(columns="bank_features_year").add_suffix("_2026_27").rename(columns={"iso3_2026_27": "iso3"})

    t = war.merge(fx, on="iso3", how="outer").merge(bank24, on="iso3", how="left").merge(bank25, on="iso3", how="left")
    t["post_soviet"] = t.iso3.isin(POST_SOVIET_ISO3)
    t["country"] = t.iso3.map(ISO3_TO_SHORT).fillna(t.country_ucdp).fillna(t.iso3)
    t = t.sort_values("iso3").reset_index(drop=True)
    t.to_csv(REPORTS / "risk_2025_26.csv", index=False, float_format="%.4f")

    war_col = [c for c in t.columns if c.startswith("p_war25_")][0]
    war1000_col = [c for c in t.columns if c.startswith("p_war1000_")][0]
    pd.set_option("display.width", 250)
    ps = t[t.post_soviet].copy()
    print(f"\n=== Постсоветские страны ({len(ps)}) ===")
    print("Война (признаки 2024 → 2025), место среди стран под риском (не воевавших в 2024):")
    print(ps[["country", "war25_now", war_col, "rank_war25_at_risk", war1000_col, "rank_war1000_at_risk"]].round(3).to_string(index=False))
    print(f"\nВалютный кризис в ближайшие 12 мес (на дату as_of), место среди {fx.iso3.nunique()} стран; база {fx.fx_base_rate.iloc[0]:.3f}:")
    print(ps[["country", "as_of", "fx_pressure", "rank_fx_pressure", "fx_signals", "rank_fx_signals", "fx_logit", "rank_fx_logit",
              "emp_z_max3", "deval12", "reserves_g12", "crisis_last_12m", "signals_fired"]].round(3).to_string(index=False))
    print("\nБанковский кризис (только страны BIS): признаки 2024 → 2025–26 и 2025 → 2026–27:")
    bcols = ["country", "bank_jst_extree_nomoney_2025_26", "rank_bank_jst_extree_nomoney_2025_26", "bank_pooled_extree_nomoney_2025_26",
             "bank_jst_extree_core_2025_26", "bank_jst_extree_full_2025_26", "bank_jst_extree_nomoney_2026_27",
             "bank_jst_extree_full_2026_27", "tloan_gdp_rdiff2_2025_26", "tloan_gdp_rdiff2_2026_27"]
    print(ps[ps.bank_jst_extree_nomoney_2025_26.notna() | ps.bank_jst_extree_nomoney_2026_27.notna()][bcols].round(3).to_string(index=False))

    print("\n=== Топ-15 мира ===")
    print("Война ≥ 25 в 2025, страны под риском:")
    print(t[t.war25_now == 0].nlargest(15, war_col)[["country", war_col, "rank_war25_at_risk"]].round(3).to_string(index=False))
    print("\nВалютный кризис, `pressure`:")
    print(t.nlargest(15, "fx_pressure")[["country", "as_of", "fx_pressure", "fx_signals", "fx_logit", "emp_z_max3", "deval12", "crisis_last_12m"]].round(3).to_string(index=False))
    print("\nВалютный кризис, `signals`:")
    print(t.nlargest(15, "fx_signals")[["country", "as_of", "fx_signals", "fx_pressure", "signals_fired"]].round(3).to_string(index=False))
    n26 = t.bank_jst_extree_nomoney_2025_26.notna().sum()
    print(f"\nБанковский кризис 2025–26, `jst_extree` (nomoney, {n26} стран; core и full — где есть):")
    print(t.nlargest(15, "bank_jst_extree_nomoney_2025_26")[["country", "bank_jst_extree_nomoney_2025_26", "bank_jst_extree_core_2025_26",
                                                              "bank_jst_extree_full_2025_26", "bank_pooled_extree_nomoney_2025_26", "tloan_gdp_rdiff2_2025_26"]].round(3).to_string(index=False))
    print("\nБанковский кризис 2026–27 (предварительно), `jst_extree` (nomoney):")
    print(t.nlargest(15, "bank_jst_extree_nomoney_2026_27")[["country", "bank_jst_extree_nomoney_2026_27", "bank_jst_extree_core_2026_27",
                                                              "bank_jst_extree_full_2026_27", "tloan_gdp_rdiff2_2026_27"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
