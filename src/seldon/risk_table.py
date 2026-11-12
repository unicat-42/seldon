"""Таблица риска на 2025–27: война, валютный кризис, банковский кризис — по странам.

Запуск:
    uv run python -m seldon.risk_table

Три модели проекта, обученные на всём, что известно, и применённые к последним данным
(после аудита 12.11.2026 — без утечек будущего, см. `tests/test_no_leakage.py`):
  1. Война (`model_hist pitf`, две логрегрессии — начала и продолжения): признаки 2024 →
     государственная война (`state_war25`, `state_war1000`) в 2025. UCDP есть по 2024,
     поэтому 2026 не считаем. Теракты и банды — отдельная колонка `nonstate25_now`.
  2. Валютный кризис, две шкалы:
     - годовая модель (`econ_long`: декабрьский курс МВФ + GMD, 1957–2025): признаки 2025 →
       девальвация ≥ 25 % (декабрь к декабрю) в 2026–27 — основной прогноз; признаки 2024 →
       2025–26 — для сверки: 2025 уже известен (`fxa_deval_2025`);
     - месячная (`econ_kr`, целевая `deval15`, логрегрессия): признаки последнего месяца МВФ →
       скачок курса ≥ 15 % за месяц в ближайшие 12 месяцев (`fx_deval15_12m`). Рядом —
       индикаторы текущего стресса, это НЕ вероятности: `emp_z_max3`, `reserves_g12`,
       `deval12`, `infl12`, `signals_fired`.
     Колонка `fx_regime`: `own` — своя валюта; код валюты (EUR, XOF, USD…) — общая или чужая,
     своего валютного риска нет, вероятность пустая; `no_imf_fx` — валюта своя, но курса
     в данных МВФ нет (Туркменистан), прогноз посчитать не на чем.
  3. Банковский кризис (`econ_boe_em`): признаки 2024 → кризис в 2025–26 и признаки 2025 →
     2026–27 (предварительно: в GMD за 2025 могут быть оценки). Деревья BoE с JST
     (`jst_extree`) и объединённые (`pooled_extree`), наборы `full`, `core`, `nomoney`.
     Свои данные — только годы с размеченной хроникой кризисов (≤ 2015), JST — ≤ 2018.

Выход:
    reports/risk_2025_26.csv — все страны;
    печать: постсоветские и топ-15 мира по каждой модели.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, PROJECT_ROOT
from seldon import econ_boe_em, econ_kr, econ_long, model_hist
from seldon.countries import GW_TO_ISO3, ISO3_TO_SHORT, POST_SOVIET_ISO3, shared_currency
from seldon.econ_boe import JST_R6, build_boe_dataset, load_jst
from seldon.features import FEATURES as HIST_FEATURES
from seldon.features_pitf import PITF_FEATURES

REPORTS = PROJECT_ROOT / "reports"
FX_LAST_MONTH = "2026-10"
FX_MIN_AS_OF = "2026-04"  # страны, у которых курс кончается раньше, в таблицу не берём
FX_REGIME_YEAR = 2026
FX_ANNUAL_YEARS = (2024, 2025)
FX_ANNUAL_TRAIN_THROUGH = 2023  # метка строки t смотрит в t+2; декабрь 2025 известен
BANK_YEARS = (2024, 2025)
# только государства системы Gleditsch–Ward: у территорий (Гернси, Гибралтар, Гренландия…)
# чужая валюта и пустые признаки, флаги пропусков модель читает как риск
GW_ISO = {v for v in GW_TO_ISO3.values() if v} | POST_SOVIET_ISO3


# ---------------------------------------------------------------------------
# 1. Война
# ---------------------------------------------------------------------------


def war_risk() -> pd.DataFrame:
    f = pd.read_parquet(DATA_PROCESSED / "features_pitf_country_year.parquet")
    feats = [*HIST_FEATURES, *PITF_FEATURES]
    last = int(f.year.max())
    out = f[f.year == last][["gw", "country", "state_war25", "state_war1000", "nonstate25"]].copy()
    for target in ("state_war25", "state_war1000"):
        p = model_hist.forecast_asof(f, last, feats, target).set_index("gw")
        out[f"p_{target}_{last + 1}"] = out.gw.map(p.p)
        at_risk = out[out[target] == 0]
        out[f"rank_{target}_at_risk"] = np.nan
        out.loc[at_risk.index, f"rank_{target}_at_risk"] = at_risk[f"p_{target}_{last + 1}"].rank(ascending=False, method="min")
    out["iso3"] = out.gw.map(GW_TO_ISO3)
    out = out.rename(columns={"state_war25": "state_war25_now", "state_war1000": "state_war1000_now",
                              "nonstate25": "nonstate25_now", "country": "country_ucdp"})
    out["war_features_year"] = last
    return out[out.iso3.notna()].drop(columns="gw")


# ---------------------------------------------------------------------------
# 2. Валютный кризис, месячный термометр
# ---------------------------------------------------------------------------


def fx_risk() -> pd.DataFrame:
    """Месячная шкала. Вероятность — только одна: `fx_deval15_12m`, логрегрессия по целевой
    `deval15` (курс за месяц вырос ≥ 15 % в ближайшие 12 мес; вслепую AUC 0,65 сквозной,
    0,70 внутри года). Остальное — описательные индикаторы текущего стресса, НЕ вероятности:
    `emp_z_max3` (давление за 3 мес, в σ), `reserves_g12`, `deval12`, `infl12`,
    `signals_fired` (какие индикаторы KLR сейчас за порогом). Модели `pressure` и `signals`
    по EMP-кризису в витрину не идут: сквозной AUC 0,51 — монетка (исследовательский режим
    остался в `econ_kr`)."""
    as_of = pd.Period(FX_LAST_MONTH, freq="M")
    p = econ_kr.build_panel(econ_kr.load_panel(FX_LAST_MONTH), econ_kr.load_annual_cpi(), econ_kr.load_war(),
                            target="deval15")
    m = econ_kr.fit(p, as_of)  # метки — только окна, закрывшиеся к as_of; пороги — история ≤ as_of
    # прогноз — последний месяц с курсом по каждой стране (признаки строки — только данные ≤ её месяца)
    last = p[p.fx.notna()].groupby("iso3").month.max()
    test = p[p.iso3.isin(GW_ISO) & p.own_currency].merge(last.rename("as_of"), on="iso3")
    test = test[(test.month == test.as_of) & (test.as_of >= FX_MIN_AS_OF)]
    test = econ_kr.predict(m, test).rename(columns={"logit": "fx_deval15_12m", "base_rate": "fx_deval15_base_rate"})
    test["rank_fx_deval15_12m"] = test.fx_deval15_12m.rank(ascending=False, method="min")
    # страны в «тишине» после недавнего скачка: прогноз считаем, но помечаем
    test["deval15_last_12m"] = (test.quiet_deval15 < econ_kr.POST_CRISIS_DROP).astype(int)
    sig = m.sig_model
    fired = {ind: econ_kr._signal(test, ind, econ_kr.INDICATORS[ind], sig[ind]["pct"], m.ref) for ind in sig}
    test["signals_fired"] = [",".join(ind for ind in sig if fired[ind].loc[i] == 1) for i in test.index]
    cols = ["iso3", "as_of", "fx_deval15_12m", "rank_fx_deval15_12m", "fx_deval15_base_rate", "deval15_last_12m",
            "emp_z_max3", "reserves_g12", "deval12", "infl12", "signals_fired"]
    out = test[cols].copy()
    out["as_of"] = out.as_of.astype(str)
    return out


# ---------------------------------------------------------------------------
# 2б. Валютный кризис, годовая модель (econ_long)
# ---------------------------------------------------------------------------


def fx_annual_risk() -> pd.DataFrame:
    """Признаки года t → девальвация ≥ 25 % (декабрь к декабрю) в t+1 или t+2."""
    p = econ_long.build_panel(econ_long.load_gmd(), econ_long.load_dec_fx(), econ_long.load_lv(), "deval25")
    p = p[p.iso3.isin(GW_ISO)]
    frames = []
    base_rate = np.nan
    for y in FX_ANNUAL_YEARS:
        t = econ_long.forecast_asof(p, y, train_through=FX_ANNUAL_TRAIN_THROUGH)
        if t.empty:
            continue
        base_rate = t.base_rate.iloc[0]
        sfx = "_2025_26" if y == 2024 else "_2026_27"
        t[f"fxa_extree{sfx}"], t[f"fxa_logit{sfx}"] = t.extree, t.logit
        t[f"rank_fxa_extree{sfx}"] = t.extree.rank(ascending=False, method="min")
        t[f"rank_fxa_logit{sfx}"] = t.logit.rank(ascending=False, method="min")
        t[f"fxa_infl{sfx}"] = t.infl
        t[f"fxa_deval_prev{sfx}"] = 100 * t.deval_prev
        t[f"fxa_ca{sfx}"] = 100 * t.ca_level
        t[f"fxa_debt_d2{sfx}"] = 100 * t.debt_d2
        frames.append(t[["iso3"] + [c for c in t.columns if c.endswith(sfx)]].set_index("iso3"))
    out = pd.concat(frames, axis=1).reset_index()
    out["fxa_base_rate"] = base_rate
    fact = p[p.year == 2025].set_index("iso3").deval
    out["fxa_deval_2025"] = 100 * out.iso3.map(fact)  # что уже случилось: декабрь 2025 к декабрю 2024, %
    return out


# ---------------------------------------------------------------------------
# 3. Банковский кризис
# ---------------------------------------------------------------------------


def bank_risk() -> pd.DataFrame:
    cfg = econ_boe_em.Config(last_year=BANK_YEARS[-1])
    panel = econ_boe_em.load_panel(cfg)
    jst = load_jst(JST_R6, "Sheet1")
    frames: dict[str, list[pd.DataFrame]] = {}
    for tag, feats in econ_boe_em.FEATURE_SETS.items():
        data = econ_boe_em.build(panel, feats)
        jst_d = build_boe_dataset(jst, predictors=feats)
        jst_d = jst_d[jst_d.year <= econ_boe_em.JST_LAST_LABEL_YEAR - 2]
        train_own = data[data.year <= cfg.last_label_year - 2]  # дальше хроника не размечена
        lo, hi = jst_d[feats].min(), jst_d[feats].max()
        q1, q99 = train_own[feats].quantile(0.01), train_own[feats].quantile(0.99)
        jst_et = econ_boe_em.make_models()["extree"].fit(jst_d[feats].values, jst_d.crisis.values)
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
    print("1/4 война…")
    war = war_risk()
    print("2/4 валютный кризис, месячный термометр…")
    fx = fx_risk()
    print("2б/4 валютный кризис, годовая модель…")
    fxa = fx_annual_risk()
    print("3/4 банковский кризис…")
    bank = bank_risk()
    bank24 = bank[bank.bank_features_year == 2024].drop(columns="bank_features_year").add_suffix("_2025_26").rename(columns={"iso3_2025_26": "iso3"})
    bank25 = bank[bank.bank_features_year == 2025].drop(columns="bank_features_year").add_suffix("_2026_27").rename(columns={"iso3_2026_27": "iso3"})

    t = (war.merge(fx, on="iso3", how="outer").merge(fxa, on="iso3", how="outer")
            .merge(bank24, on="iso3", how="left").merge(bank25, on="iso3", how="left"))
    t["post_soviet"] = t.iso3.isin(POST_SOVIET_ISO3)
    t["country"] = t.iso3.map(ISO3_TO_SHORT).fillna(t.country_ucdp).fillna(t.iso3)
    # режим валюты: own — своя (свой риск девальвации); иначе код общей/чужой валюты, прогноза нет
    t["fx_regime"] = [shared_currency(i, FX_REGIME_YEAR) or "own" for i in t.iso3]
    # у страны своя валюта, но в данных МВФ нет её курса (Туркменистан): явный статус вместо молчаливых NaN
    no_fx = (t.fx_regime == "own") & t.as_of.isna() & t.fxa_deval_2025.isna()
    t.loc[no_fx, "fx_regime"] = "no_imf_fx"
    t = t.sort_values("iso3").reset_index(drop=True)
    t.to_csv(REPORTS / "risk_2025_26.csv", index=False, float_format="%.4f")

    w25, w1000 = "p_state_war25_2025", "p_state_war1000_2025"
    pd.set_option("display.width", 250)
    ps = t[t.post_soviet].copy()
    print(f"\n=== Постсоветские страны ({len(ps)}) ===")
    print("Война (признаки 2024 → 2025), место среди стран под риском (без государственной войны в 2024):")
    print(ps[["country", "state_war25_now", w25, "rank_state_war25_at_risk", w1000, "rank_state_war1000_at_risk"]].round(3).to_string(index=False))
    nfa = fxa.fxa_extree_2026_27.notna().sum()
    print(f"\nВалютный кризис, годовая модель: признаки 2025 → девальвация ≥ 25 % в 2026–27 (место среди {nfa} стран; "
          f"база {fxa.fxa_base_rate.iloc[0]:.3f}); 2024 → 2025–26 для сверки с фактом 2025:")
    print(ps[["country", "fx_regime", "fxa_logit_2026_27", "rank_fxa_logit_2026_27", "fxa_extree_2026_27", "rank_fxa_extree_2026_27",
              "fxa_logit_2025_26", "fxa_extree_2025_26", "fxa_deval_2025", "fxa_infl_2026_27", "fxa_deval_prev_2026_27", "fxa_ca_2026_27"]].round(3).to_string(index=False))
    print(f"\nМесячная шкала на дату as_of. Вероятность скачка курса ≥ 15 % за месяц в ближайшие 12 мес (`deval15`, логрегрессия; "
          f"место среди {fx.iso3.nunique()} стран; база {fx.fx_deval15_base_rate.iloc[0]:.3f}) и индикаторы текущего стресса (не вероятности):")
    print(ps[["country", "fx_regime", "as_of", "fx_deval15_12m", "rank_fx_deval15_12m", "deval15_last_12m",
              "emp_z_max3", "reserves_g12", "deval12", "infl12", "signals_fired"]].round(3).to_string(index=False))
    print("\nБанковский кризис (только страны BIS): признаки 2024 → 2025–26 и 2025 → 2026–27:")
    bcols = ["country", "bank_jst_extree_nomoney_2025_26", "rank_bank_jst_extree_nomoney_2025_26", "bank_pooled_extree_nomoney_2025_26",
             "bank_jst_extree_core_2025_26", "bank_jst_extree_full_2025_26", "bank_jst_extree_nomoney_2026_27",
             "bank_jst_extree_full_2026_27", "tloan_gdp_rdiff2_2025_26", "tloan_gdp_rdiff2_2026_27"]
    print(ps[ps.bank_jst_extree_nomoney_2025_26.notna() | ps.bank_jst_extree_nomoney_2026_27.notna()][bcols].round(3).to_string(index=False))

    print("\n=== Топ-15 мира ===")
    print("Государственная война ≥ 25 в 2025, страны под риском:")
    print(t[t.state_war25_now == 0].nlargest(15, w25)[["country", w25, "rank_state_war25_at_risk", "nonstate25_now"]].round(3).to_string(index=False))
    print("\nВалютный кризис 2026–27, годовая модель (логит):")
    print(t.nlargest(15, "fxa_logit_2026_27")[["country", "fxa_logit_2026_27", "fxa_extree_2026_27", "fxa_logit_2025_26", "fxa_deval_2025",
                                                "fxa_infl_2026_27", "fxa_deval_prev_2026_27", "fxa_ca_2026_27", "fx_deval15_12m"]].round(3).to_string(index=False))
    print("\nСкачок курса ≥ 15 % в ближайшие 12 мес (`deval15`, логрегрессия):")
    print(t.nlargest(15, "fx_deval15_12m")[["country", "as_of", "fx_deval15_12m", "deval15_last_12m", "emp_z_max3", "deval12",
                                             "infl12", "signals_fired"]].round(3).to_string(index=False))
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
