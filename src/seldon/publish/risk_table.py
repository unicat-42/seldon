"""Таблица риска на 2026–27: война, валютный кризис, банковский кризис — по странам.

Запуск:
    uv run python -m seldon risk_table

Три модели проекта, обученные на всём, что известно, и применённые к последним данным
(после аудита 12.11.2026 — без утечек будущего, см. `tests/test_no_leakage.py`).
Годовые прогнозы — в той же постановке, в какой модели проверены вслепую: «на 1 января»
(чекпоинт 2.2, `core/temporal.py`). «Признаки года T» значит: курс и история войн — за T,
а годовые макроданные (GMD, ВБ, BIS) — за T−1, потому что за T к 1 января ещё не вышли.
  1. Война (`model_hist pitf`, две логрегрессии — начала и продолжения): признаки 2025 →
     государственная война (`state_war25`, `state_war1000`) в 2026, прогноз «на 1 января
     2026» (UCDP 26.1 покрывает 2025 год; чекпоинт 3.1). Теракты и банды — отдельная
     колонка `nonstate25_now`.
  2. Валютный кризис, две шкалы:
     - годовая модель (`econ_long`: декабрьский курс МВФ + GMD, 1957–2025): признаки 2025
       (курс — декабрь 2025, макро — 2024) → девальвация ≥ 25 % (декабрь к декабрю) в 2026–27 —
       основной прогноз «на 1 января 2026»; признаки 2024 → 2025–26 — для сверки: 2025 уже
       известен (`fxa_deval_2025`). Колонки `fxa_infl_*`, `fxa_ca_*`, `fxa_debt_d2_*` — те
       самые макроданные с лагом (для `_2026_27` это 2024 год);
     - месячная (`econ_kr`, целевая `deval15`, логрегрессия): признаки последнего месяца МВФ →
       скачок курса ≥ 15 % за месяц в ближайшие 12 месяцев (`fx_deval15_12m`). Рядом —
       претендент `fx_deval15_12m_live` (набор `live_fx_swap` из `econ_csd`: те же целевая и
       окно, плюс привязка, REER и привязка × падение резервов; боевая модель не заменена —
       обе копятся в снимках для проспективного сравнения). Месячные резервы МВФ на строке
       месяца m — за m − 2 (лаг публикации `imf_reserves`). Ещё рядом —
       индикаторы текущего стресса, это НЕ вероятности: `emp_z_max3`, `reserves_g12`,
       `deval12`, `infl12`, `signals_fired` и два индикатора уязвимости режима курса
       («сжатая пружина», чекпоинт 3.3; в модели витрины не входят): `peg_share12` — доля
       спокойных месяцев из последних 12 (1 — жёсткая привязка к доллару или евро),
       `reer_peg` — реальное укрепление сверх собственного среднего за 5 лет, умноженное на
       `peg_share12` (0 — пружина не натянута); `peg_resfall` — привязка × падение резервов
       за 12 месяцев (0 — резервы не уходят; пусто — резервов нет в данных). Все три считаются
       по живым рядам (курс МВФ, резервы МВФ, REER Bruegel), без замороженной классификации IRR.
       REER выходит реже курса: `reer_peg` — на последний месяц с REER, не старше
       `REER_MAX_STALE` месяцев от `as_of`.
     Колонка `fx_regime`: `own` — своя валюта; код валюты (EUR, XOF, USD…) — общая или чужая,
     своего валютного риска нет, вероятность пустая; `no_imf_fx` — валюта своя, но курса
     в данных МВФ нет (Туркменистан), прогноз посчитать не на чем.
  3. Банковский кризис (`econ_boe_em`): прогноз «на 1 января 2025» (данные по 2023) → кризис
     в 2025–26 и «на 1 января 2026» (данные по 2024) → 2026–27. В витрину идёт ТОЛЬКО набор
     `full` (с кривой доходности): вслепую на странах вне JST сквозной AUC 0,67 [0,52; 0,80],
     внутри года 0,75, всего 9 предкризисных лет — покрытие узкое. Колонка `bank_status`
     объясняет, почему прогноза нет (см. `bank_risk`). Наборы без ставок (`core`, `nomoney`)
     в витрину не идут: после лага публикации они не отличаются от монетки.

Выход:
    reports/risk_2025_26.csv — все страны;
    печать: постсоветские и топ-15 мира по каждой модели.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, REPORTS
from seldon.core.entities import GW_TO_ISO3, ISO3_TO_SHORT, POST_SOVIET_ISO3, STATE_ISO3, shared_currency
from seldon.domains.banking import econ_boe_em
from seldon.domains.conflict import model_hist
from seldon.domains.conflict.features import FEATURES as HIST_FEATURES
from seldon.domains.conflict.features_pitf import PITF_FEATURES
from seldon.domains.currency import econ_csd, econ_kr, econ_long, kr_features, kr_targets
from seldon.domains.currency.struct_features import load_struct
from seldon.sources.gmd import load_gmd
from seldon.sources.imf import load_dec_fx
from seldon.sources.jst import JST_R6, load_jst
from seldon.sources.wdi import load_annual_cpi

FX_LAST_MONTH = "2026-10"
FX_MIN_AS_OF = "2026-04"  # страны, у которых курс кончается раньше, в таблицу не берём
FX_REGIME_YEAR = 2026
REER_MAX_STALE = 3  # `reer_peg` берётся на последний месяц с REER, если он не старше стольких месяцев от `as_of`
RES_MAX_STALE = 3   # то же для `peg_resfall`: месячные резервы МВФ выходят на 1–2 месяца позже курса
FX_ANNUAL_YEARS = (2024, 2025)
FX_ANNUAL_TRAIN_THROUGH = 2023  # метка строки t смотрит в t+2; декабрь 2025 известен
BANK_YEARS = (2024, 2025)
# только государства системы Gleditsch–Ward: у территорий (Гернси, Гибралтар, Гренландия…)
# чужая валюта и пустые признаки, флаги пропусков модель читает как риск
GW_ISO = STATE_ISO3  # тот же список, на котором обучаются и оцениваются валютные модели


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
    `emp_z_max3` (давление за 3 мес, в σ), `reserves_g12` (по месяц `as_of` − 2: лаг публикации), `deval12`, `infl12`,
    `signals_fired` (какие индикаторы KLR сейчас за порогом), `peg_share12`, `reer_peg` и
    `peg_resfall` (уязвимость режима курса, см. `regime_stress`). Модели `pressure` и `signals`
    по EMP-кризису в витрину не идут: сквозной AUC 0,51 — монетка (исследовательский режим
    остался в `econ_kr`)."""
    as_of = pd.Period(FX_LAST_MONTH, freq="M")
    p = econ_kr.build_panel(econ_kr.load_panel(FX_LAST_MONTH), load_annual_cpi(), kr_features.load_war(),
                            target="deval15")
    m = econ_kr.fit(p, as_of)  # метки — только окна, закрывшиеся к as_of; пороги — история ≤ as_of
    # прогноз — последний месяц с курсом по каждой стране (признаки строки — только данные ≤ её месяца)
    last = p[p.fx.notna()].groupby("iso3").month.max()
    test = p[p.iso3.isin(GW_ISO) & p.own_currency].merge(last.rename("as_of"), on="iso3")
    test = test[(test.month == test.as_of) & (test.as_of >= FX_MIN_AS_OF)]
    test = econ_kr.predict(m, test).rename(columns={"logit": "fx_deval15_12m", "base_rate": "fx_deval15_base_rate"})
    test["rank_fx_deval15_12m"] = test.fx_deval15_12m.rank(ascending=False, method="min")
    # страны в «тишине» после недавнего скачка: прогноз считаем, но помечаем
    test["deval15_last_12m"] = (test.quiet_deval15 < kr_targets.POST_CRISIS_DROP).astype(int)
    sig = m.sig_model
    fired = {ind: econ_kr._signal(test, ind, kr_features.INDICATORS[ind], sig[ind]["pct"], m.ref) for ind in sig}
    test["signals_fired"] = [",".join(ind for ind in sig if fired[ind].loc[i] == 1) for i in test.index]
    cols = ["iso3", "as_of", "fx_deval15_12m", "rank_fx_deval15_12m", "fx_deval15_base_rate", "deval15_last_12m",
            "emp_z_max3", "reserves_g12", "deval12", "infl12", "signals_fired"]
    out = test[cols].merge(regime_stress(test[["iso3", "as_of"]]), on="iso3", how="left")
    out["peg_resfall"] = peg_resfall(p, test[["iso3", "as_of"]], out.peg_share12)
    # претендент рядом с боевой моделью (развилка B1): та же целевая и то же окно, набор `live_fx_swap`
    live = econ_csd.live_edge(econ_csd.light_panel(p, load_struct(last_month=FX_LAST_MONTH)),
                              load_struct(last_month=FX_LAST_MONTH, filled=False), test[["iso3", "as_of"]], as_of,
                              reer_max_stale=REER_MAX_STALE, res_max_stale=RES_MAX_STALE).set_index("iso3")
    out["fx_deval15_12m_live"] = out.iso3.map(live.p)
    out["rank_fx_deval15_12m_live"] = out.fx_deval15_12m_live.rank(ascending=False, method="min")
    out["as_of"] = out.as_of.astype(str)
    return out


def peg_resfall(panel: pd.DataFrame, as_of: pd.DataFrame, peg: pd.Series) -> pd.Series:
    """`peg_resfall` — привязка × падение резервов за 12 месяцев (лучший признак чекпоинта 4.1,
    решение Ангелины 13.11.2026): 0 — резервы не уходят, пусто — резервов нет в данных.
    Описательный индикатор, НЕ вероятность и не признак боевой модели.

    Месячные резервы МВФ выходят позже курса: `reserves_g12` на строке месяца m — прирост по
    месяц m − 2 (лаг `imf_reserves`). Если и его ещё нет, берётся последнее значение не старше
    `RES_MAX_STALE` месяцев от `as_of` страны — как у `reer_peg`. Поэтому индикатор может быть
    посчитан не на тот месяц, что колонка `reserves_g12` рядом."""
    h = panel[panel.reserves_g12.notna()].merge(as_of, on="iso3")
    h = h[(h.month <= h.as_of) & ((h.as_of - h.month).map(lambda d: d.n) <= RES_MAX_STALE)]
    res = h.sort_values("month").groupby("iso3").reserves_g12.last()
    return peg * (-as_of.iso3.map(res).to_numpy()).clip(min=0)


def regime_stress(as_of: pd.DataFrame) -> pd.DataFrame:
    """Индикаторы уязвимости режима курса на дату `as_of` страны (iso3, as_of): `peg_share12`
    и `reer_peg` из `struct_features` (определения — там). Описательные, НЕ вероятности и не
    признаки боевых моделей. Пропуск остаётся пропуском (нет REER у страны — пусто, а не 0)."""
    s = load_struct(last_month=FX_LAST_MONTH, filled=False).merge(as_of, on="iso3")
    s = s[s.month <= s.as_of]
    peg = s[s.month == s.as_of].set_index("iso3").peg_share12
    r = s[s.reer_gap.notna() & ((s.as_of - s.month).map(lambda d: d.n) <= REER_MAX_STALE)]
    reer = r.sort_values("month").groupby("iso3").reer_peg.last()
    return pd.DataFrame({"peg_share12": peg, "reer_peg": reer}).rename_axis("iso3").reset_index()


# ---------------------------------------------------------------------------
# 2б. Валютный кризис, годовая модель (econ_long)
# ---------------------------------------------------------------------------


def fx_annual_risk() -> pd.DataFrame:
    """Признаки года t → девальвация ≥ 25 % (декабрь к декабрю) в t+1 или t+2."""
    p = econ_long.build_panel(load_gmd(), load_dec_fx(), econ_long.load_lv(), "deval25")
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


BANK_SET = "full"  # единственный набор признаков, который идёт в витрину (см. docstring)


def bank_risk() -> pd.DataFrame:
    """Банковский прогноз для витрины: только деревья BoE с переносом с JST на наборе `full`
    (с кривой доходности). Наборы без ставок (`nomoney`, `core`) после лага публикации не
    отличаются от монетки (сквозной AUC 0,55 [0,41; 0,65] и 0,52) — их вероятности и ранги
    не публикуем; цифры остаются в `econ_boe_em` и `scorecard`.
    Колонки: iso3, bank_features_year, bank_jst_extree_full, rank_bank_jst_extree_full,
    bank_status — почему прогноз есть или почему его нет:
        ok              прогноз есть;
        no_yield_curve  страна есть в кредитной статистике BIS, но нет длинной ставки (Россия);
        no_broad_money  длинная ставка есть, но нет своей денежной массы (еврозона);
        no_bis_credit   нет кредитного ряда BIS (большинство стран мира)."""
    cfg = econ_boe_em.Config(last_year=BANK_YEARS[-1])
    panel = econ_boe_em.load_panel(cfg)
    feats = econ_boe_em.FEATURE_SETS[BANK_SET]
    data = econ_boe_em.build(panel, feats, cfg.macro_lag)
    jst_d = econ_boe_em.boe_dataset(load_jst(JST_R6, "Sheet1"), feats, cfg.macro_lag)
    jst_d = jst_d[jst_d.year <= econ_boe_em.JST_LAST_LABEL_YEAR - 2]
    lo, hi = jst_d[feats].min(), jst_d[feats].max()
    jst_et = econ_boe_em.make_models()["extree"].fit(jst_d[feats].values, jst_d.crisis.values)
    col = f"bank_jst_extree_{BANK_SET}"
    frames = []
    for y in BANK_YEARS:
        rows = panel[panel.year == y].set_index("iso")
        t = data[data.year == y].set_index("iso")
        out = pd.DataFrame({"iso3": rows.index, "bank_features_year": y})
        p = pd.Series(jst_et.predict_proba(t[feats].clip(lo, hi, axis=1).values)[:, 1], index=t.index) if len(t) else pd.Series(dtype=float)
        out[col] = out.iso3.map(p)
        out[f"rank_{col}"] = out[col].rank(ascending=False, method="min")
        # ставки — рыночные, берутся за год y; денежная масса — макроряд с лагом
        no_curve = rows.ltrate.isna() | rows.stir.isna()
        out["bank_status"] = np.where(out[col].notna(), "ok",
                                      np.where(out.iso3.map(no_curve), "no_yield_curve", "no_broad_money"))
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


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
    # статус покрытия — по основному прогнозу (на 2026–27); прогноз на 2025–26 — для сверки
    bank24 = bank[bank.bank_features_year == 2024].drop(columns=["bank_features_year", "bank_status"]).add_suffix("_2025_26").rename(columns={"iso3_2025_26": "iso3"})
    bank25 = bank[bank.bank_features_year == 2025].drop(columns="bank_features_year").add_suffix("_2026_27").rename(columns={"iso3_2026_27": "iso3", "bank_status_2026_27": "bank_status"})

    t = (war.merge(fx, on="iso3", how="outer").merge(fxa, on="iso3", how="outer")
            .merge(bank24, on="iso3", how="left").merge(bank25, on="iso3", how="left"))
    t["bank_status"] = t.bank_status.fillna("no_bis_credit")
    t["post_soviet"] = t.iso3.isin(POST_SOVIET_ISO3)
    t["country"] = t.iso3.map(ISO3_TO_SHORT).fillna(t.country_ucdp).fillna(t.iso3)
    # режим валюты: own — своя (свой риск девальвации); иначе код общей/чужой валюты, прогноза нет
    t["fx_regime"] = [shared_currency(i, FX_REGIME_YEAR) or "own" for i in t.iso3]
    # у страны своя валюта, но в данных МВФ нет её курса (Туркменистан): явный статус вместо молчаливых NaN
    no_fx = (t.fx_regime == "own") & t.as_of.isna() & t.fxa_deval_2025.isna()
    t.loc[no_fx, "fx_regime"] = "no_imf_fx"
    t = t.sort_values("iso3").reset_index(drop=True)
    t.to_csv(REPORTS / "risk_2025_26.csv", index=False, float_format="%.4f")

    wy = int(war.war_features_year.iloc[0])  # год признаков войны; прогноз — на wy + 1
    w25, w1000 = f"p_state_war25_{wy + 1}", f"p_state_war1000_{wy + 1}"
    pd.set_option("display.width", 250)
    ps = t[t.post_soviet].copy()
    print(f"\n=== Постсоветские страны ({len(ps)}) ===")
    print(f"Война (признаки {wy} → {wy + 1}), место среди стран под риском (без государственной войны в {wy}):")
    print(ps[["country", "state_war25_now", w25, "rank_state_war25_at_risk", w1000, "rank_state_war1000_at_risk"]].round(3).to_string(index=False))
    nfa = fxa.fxa_extree_2026_27.notna().sum()
    print(f"\nВалютный кризис, годовая модель: признаки 2025 → девальвация ≥ 25 % в 2026–27 (место среди {nfa} стран; "
          f"база {fxa.fxa_base_rate.iloc[0]:.3f}); 2024 → 2025–26 для сверки с фактом 2025:")
    print(ps[["country", "fx_regime", "fxa_logit_2026_27", "rank_fxa_logit_2026_27", "fxa_extree_2026_27", "rank_fxa_extree_2026_27",
              "fxa_logit_2025_26", "fxa_extree_2025_26", "fxa_deval_2025", "fxa_infl_2026_27", "fxa_deval_prev_2026_27", "fxa_ca_2026_27"]].round(3).to_string(index=False))
    print(f"\nМесячная шкала на дату as_of. Вероятность скачка курса ≥ 15 % за месяц в ближайшие 12 мес (`deval15`, логрегрессия; "
          f"место среди {fx.iso3.nunique()} стран; база {fx.fx_deval15_base_rate.iloc[0]:.3f}) и индикаторы текущего стресса (не вероятности):")
    print(ps[["country", "fx_regime", "as_of", "fx_deval15_12m", "rank_fx_deval15_12m", "deval15_last_12m",
              "emp_z_max3", "reserves_g12", "deval12", "infl12", "signals_fired", "peg_share12", "reer_peg", "peg_resfall"]].round(3).to_string(index=False))
    print("\nБанковский кризис (деревья BoE, набор `full`; есть только у стран BIS с длинной ставкой и своей денежной массой):")
    print(ps[["country", "bank_status", "bank_jst_extree_full_2026_27", "rank_bank_jst_extree_full_2026_27"]].round(3).to_string(index=False))
    print("  покрытие по миру:", t.bank_status.value_counts().to_dict())

    print("\n=== Топ-15 мира ===")
    print(f"Государственная война ≥ 25 в {wy + 1}, страны под риском:")
    print(t[t.state_war25_now == 0].nlargest(15, w25)[["country", w25, "rank_state_war25_at_risk", "nonstate25_now"]].round(3).to_string(index=False))
    print("\nВалютный кризис 2026–27, годовая модель (логит):")
    print(t.nlargest(15, "fxa_logit_2026_27")[["country", "fxa_logit_2026_27", "fxa_extree_2026_27", "fxa_logit_2025_26", "fxa_deval_2025",
                                                "fxa_infl_2026_27", "fxa_deval_prev_2026_27", "fxa_ca_2026_27", "fx_deval15_12m"]].round(3).to_string(index=False))
    print("\nСкачок курса ≥ 15 % в ближайшие 12 мес (`deval15`, логрегрессия):")
    print(t.nlargest(15, "fx_deval15_12m")[["country", "as_of", "fx_deval15_12m", "deval15_last_12m", "emp_z_max3", "deval12",
                                             "infl12", "signals_fired"]].round(3).to_string(index=False))
    n27 = t.bank_jst_extree_full_2026_27.notna().sum()
    print(f"\nБанковский кризис 2026–27, `jst_extree` (набор full, {n27} стран); рядом — прогноз на 2025–26:")
    print(t.nlargest(15, "bank_jst_extree_full_2026_27")[["country", "bank_jst_extree_full_2026_27", "rank_bank_jst_extree_full_2026_27",
                                                           "bank_jst_extree_full_2025_26"]].round(3).to_string(index=False))
    return 0

