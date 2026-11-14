"""Валютные кризисы по Kaminsky–Lizondo–Reinhart (1998) на месячных данных МВФ.

Запуск:
    uv run python -m seldon econ_kr [crisis|deval15] [all_currencies]

Что делает:
  1. Месячная панель страна×месяц по всем странам из данных МВФ (`download imf`): курс
     к доллару (конец месяца), резервы без золота (долл.), ставки, ИПЦ. Дыры в месячном
     ИПЦ затыкаем годовым ИПЦ ВБ (`download wdi`) экстраполяцией из последнего
     завершённого года — без утечки (см. `fill_cpi`).
     Страны с общей или чужой валютой (`core.entities.SHARED_CURRENCY`) исключаются с года
     перехода; `all_currencies` — срез для сравнения без исключения (суффикс `_allcur`).
  2. Индекс давления на валютный рынок (EMP — exchange market pressure):
         EMP = %Δe − (σ_e / σ_R) · %ΔR,
     где e — курс (рост = ослабление), R — резервы, σ — стандартные отклонения по стране.
     Все параметры (σ, медиана EMP) в месяце m считаются только по месяцам ≤ m —
     расширяющимся окном: кризис определяется так, как его увидели бы в реальном времени
     (после аудита 12.11.2026; раньше — один раз по всей выборке 1993–2025).
     Кризис = месяц, когда EMP выше медианы по стране на 3σ (как у KLR, но σ устойчивая:
     через медианное абсолютное отклонение, иначе 1990-е раздувают масштаб) и при этом
     EMP ≥ 10% (иначе у привязанных валют σ ≈ 0 и сдвиг в 1% — «кризис»). Месяцы
     гиперинфляции (6-месячная инфляция > 150%) из расчёта σ выброшены, как у KLR.
     Повторные всплески внутри 12 месяцев после кризиса — не новый кризис.
     Для сравнения — `deval15`: курс за месяц вырос ≥ 15% (чисто по курсу, без резервов).
  3. Индикаторы KLR, доступные помесячно: рост резервов за 1 и 12 мес, отклонение реального
     курса от тренда, прошлая девальвация за 12 мес, инфляция, реальная ставка, разница
     ставок с США, спред кредит/депозит, рост M2/резервы за 12 мес, рост экспорта за 12 мес,
     плюс два «не из KLR»: доля стран мира с кризисом за последние 3 мес (заражение) и
     максимум EMP самой страны за последние 3 мес (давление уже идёт); плюс война из блока
     войн (UCDP по месяцам, `war_features`): государственная война ≥ 25 за месяц, ≥ 1000
     за последние 12 мес, начало войны за 12 мес.
  4. Целевая: кризис в ближайшие 12 месяцев. Строка оценивается, только если с последнего
     ВСПЛЕСКА давления прошло ≥ 12 месяцев (то же правило, по которому повторный всплеск
     не считается новым кризисом) и у страны накоплено ≥ 36 спокойных месяцев истории.
  5. Модели вслепую по годам (учимся на метках, известных к 1 января года T,
     предсказываем все месяцы T):
     - `base_rate` — доля кризисных окон в обучающей выборке;
     - `pressure` — персистентность: доля кризисных окон в обучении при таком же максимуме
       EMP за последние 3 месяца (аналог `decay` в блоке войн);
     - `signals` — метод сигналов KLR: каждый индикатор даёт сигнал, если выходит за
       порог-процентиль своей страны; процентиль подбирается на обучении по минимуму
       отношения шум/сигнал; сводный индекс = сумма сигналов с весом 1/(шум/сигнал);
       вероятность — доля кризисов в обучении при таком же значении индекса;
     - `logit` — логрегрессия на тех же индикаторах.
  6. Таблица постсоветских кризисов: что модели давали за 12 месяцев до каждого,
     и место страны среди всех стран мира в тот месяц.

Выход:
    data/processed/econ_kr_panel[_deval15].parquet, econ_kr_predictions[_deval15].parquet
    reports/econ_kr_coverage.csv, econ_kr_crises_post_soviet.csv, econ_kr_metrics.csv,
    econ_kr_metrics_by_year.csv, econ_kr_pre_crisis_table.csv, econ_kr_thresholds.csv
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, FLOAT_FORMAT, REPORTS
from seldon.core.calibration import binned_rate
from seldon.core import entities
from seldon.core.entities import ISO3_TO_SHORT, POST_SOVIET_ISO3
from seldon.core.metrics import auc, auc_within, brier
from seldon.core.panel import full_grid
from seldon.core.temporal import label_known
from seldon.domains.currency.kr_features import BINARY, INDICATORS, add_indicators, fill_cpi, load_war
from seldon.domains.currency.kr_targets import H, MIN_HISTORY, add_emp_and_crises, add_target
from seldon.evaluation.backtest import backtest
from seldon.sources.imf import read_imf
from seldon.sources.wdi import load_annual_cpi

FIRST_MONTH = "1993-01"
LAST_MONTH = "2025-12"
FIRST_EVAL_YEAR = 1996
LAST_EVAL_YEAR = 2025
PERCENTILES = [5, 10, 15, 20, 25, 30]  # хвост распределения, кандидаты порога (KLR: 10–20)

MMRT = "MMRT_RT_PT_A_PT"
RATE_PRIORITY = ["MMRT_RT_PT_A_PT", "IRTBILL_RT_PT_A_PT", "MFS166_RT_PT_A_PT",
                 "DISR_RT_PT_A_PT", "MFS171_RT_PT_A_PT"]


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------


def load_panel(last_month: str = LAST_MONTH, states_only: bool = True) -> pd.DataFrame:
    """Месячные ряды МВФ по странам. `states_only` — только государства списка GW
    (`core.entities.STATE_ISO3`): без зависимых территорий и агрегатов; `False` — всё, что есть
    в потоках (так было до 14.11.2026; нужно только для сравнения «было → стало»)."""
    er = read_imf("exchange_rates")
    er = er[er.TYPE_OF_TRANSFORMATION == "EOP_RT"].pivot_table(
        index=["iso3", "month"], values="value", aggfunc="first").rename(columns={"value": "fx"})
    il = read_imf("reserves")
    il = il[(il.INDICATOR == "RXF11_REVS") & (il.UNIT == "USD")].pivot_table(
        index=["iso3", "month"], values="value", aggfunc="first").rename(columns={"value": "reserves"})
    cpi = read_imf("cpi").pivot_table(index=["iso3", "month"], values="value", aggfunc="first") \
        .rename(columns={"value": "cpi"})
    ir = read_imf("interest_rates").pivot_table(index=["iso3", "month"], columns="INDICATOR",
                                                 values="value", aggfunc="first")
    # ставка: денежный рынок, если нет — векселя, ключевая, учётная, рефинансирования
    ir["rate"] = np.nan
    for col in RATE_PRIORITY:
        if col in ir:
            ir["rate"] = ir["rate"].fillna(ir[col])
    ir = ir.rename(columns={"MFS162_RT_PT_A_PT": "lending", "MFS135_RT_PT_A_PT": "deposit"})
    ir = ir[[c for c in ("rate", "lending", "deposit") if c in ir]]
    bm = read_imf("broad_money").pivot_table(index=["iso3", "month"], values="value", aggfunc="first") \
        .rename(columns={"value": "broad_money"})
    xg = read_imf("exports").pivot_table(index=["iso3", "month"], values="value", aggfunc="first") \
        .rename(columns={"value": "exports"})

    p = er.join(il, how="outer").join(cpi, how="outer").join(ir, how="outer") \
        .join(bm, how="outer").join(xg, how="outer").reset_index()
    p = p[(p.month >= FIRST_MONTH) & (p.month <= last_month)]
    if states_only:
        p = entities.states_only(p)
    # полная сетка месяцев внутри страны, чтобы лаги считались по календарю, а не по строкам
    idx = full_grid(p.groupby("iso3").month.agg(["min", "max"]), names=["iso3", "month"], freq="M")
    p = p.set_index(["iso3", "month"]).reindex(idx).reset_index()
    # 0 в курсе/резервах — пропуск, не значение
    for c in ("fx", "reserves", "cpi", "broad_money", "exports"):
        p.loc[p[c] <= 0, c] = np.nan
    p["post_soviet"] = p.iso3.isin(POST_SOVIET_ISO3)
    return p.sort_values(["iso3", "month"]).reset_index(drop=True)


def build_panel(raw: pd.DataFrame, annual_cpi: pd.DataFrame, war: pd.DataFrame | None = None,
                target: str = "crisis", drop_shared: bool = True, reserves_lag: int | None = None) -> pd.DataFrame:
    """Сырые ряды → панель с EMP, кризисами, индикаторами и целевой. Чистая функция:
    значения на строке месяца m зависят только от данных ≤ m (кроме `y` — это будущее);
    признаки с резервами — от резервов ≤ m − лаг публикации (`reserves_lag`, см. `add_indicators`)."""
    p = fill_cpi(raw, annual_cpi)
    p = add_emp_and_crises(p, drop_shared=drop_shared)
    p = add_indicators(p, war, reserves_lag)
    return add_target(p, target)


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------


def fit_signals(train: pd.DataFrame) -> dict:
    """Для каждого индикатора — процентиль-порог (общий для всех стран, применяется
    к распределению каждой страны) с минимальным отношением шум/сигнал на обучении."""
    model = {}
    for ind, side in INDICATORS.items():
        best = None
        for pct in PERCENTILES:
            sig = _signal(train, ind, side, pct, train)
            ok = sig.notna() & train.y.notna()
            if ok.sum() == 0:
                continue
            s, y = sig[ok].astype(bool), train.y[ok].astype(bool)
            hit = (s & y).sum() / max(y.sum(), 1)          # доля пойманных кризисных окон
            false = (s & ~y).sum() / max((~y).sum(), 1)    # доля ложных тревог в тишине
            nsr = false / hit if hit > 0 else np.inf
            if best is None or nsr < best["nsr"]:
                best = {"pct": pct, "nsr": nsr, "hit": hit, "false": false}
        if best is not None and np.isfinite(best["nsr"]) and best["nsr"] < 1:
            model[ind] = best
    return model


def _thresholds(ref: pd.DataFrame, ind: str, pct: int, side: str) -> pd.Series:
    q = pct / 100 if side == "low" else 1 - pct / 100
    cnt = ref.groupby("iso3")[ind].count()
    thr = ref.groupby("iso3")[ind].quantile(q)
    thr[cnt < MIN_HISTORY] = np.nan
    pooled = ref[ind].quantile(q)
    return thr, pooled


def _signal(df: pd.DataFrame, ind: str, side: str, pct: int, ref: pd.DataFrame) -> pd.Series:
    if ind in BINARY:
        return (df[ind] >= 1).astype(float).where(df[ind].notna())
    thr, pooled = _thresholds(ref, ind, pct, side)
    t = df.iso3.map(thr).fillna(pooled)
    x = df[ind]
    sig = (x <= t) if side == "low" else (x >= t)
    # Порог-процентиль страны может оказаться ровно нулём (жёсткая привязка: курс не менялся
    # большую часть истории, и 80-й процентиль `deval12` = 0). Тогда «изменение = 0» сигналом
    # не считаем — иначе у привязанных валют «девальвация за 12 мес» горела бы постоянно.
    sig &= ~((t == 0) & (x == 0))
    return sig.astype(float).where(x.notna())


def composite(df: pd.DataFrame, model: dict, ref: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    for ind, m in model.items():
        sig = _signal(df, ind, INDICATORS[ind], m["pct"], ref).fillna(0)
        score += sig / m["nsr"]
    return score


@dataclass(frozen=True)
class Fitted:
    """Модели, обученные на том, что известно к концу месяца `as_of`."""
    as_of: pd.Period
    known: pd.DataFrame      # строки с закрывшимся окном: month + H ≤ as_of
    ref: pd.DataFrame        # вся история ≤ as_of — для порогов-процентилей по стране
    sig_model: dict
    logit: object
    lo: pd.Series
    hi: pd.Series
    med: pd.Series


def fit(p: pd.DataFrame, as_of: pd.Period) -> Fitted | None:
    """Обучение в точке as_of: метки — только окна, целиком закрывшиеся к as_of."""
    feats = list(INDICATORS)
    known = p[label_known(p.month, as_of, H) & p.evaluable]
    ref = p[p.month <= as_of]
    if known.y.sum() < 5:
        return None
    tr = known[feats]
    lo, hi = tr.quantile(0.01), tr.quantile(0.99)
    med = tr.median().fillna(0.0)  # ряд пуст во всём обучении (экспорт в 1990-е) → 0
    logit = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    logit.fit(tr.clip(lo, hi, axis=1).fillna(med), known.y)
    return Fitted(as_of, known, ref, fit_signals(known), logit, lo, hi, med)


def predict(m: Fitted, test: pd.DataFrame) -> pd.DataFrame:
    feats = list(INDICATORS)
    test = test.copy()
    known = m.known
    test["base_rate"] = known.y.mean()
    # персистентность: вероятность кризиса в обучении при таком же давлении за 3 мес
    # (как `decay` в блоке войн: «где трясёт — там тряхнёт ещё»)
    test["pressure"] = binned_rate(known.emp_z_max3.fillna(0), known.y, test.emp_z_max3.fillna(0)).to_numpy()
    s_tr, s_te = composite(known, m.sig_model, m.ref), composite(test, m.sig_model, m.ref)
    test["signals_score"] = s_te
    test["signals"] = binned_rate(s_tr, known.y, s_te).to_numpy()
    te = test[feats].clip(m.lo, m.hi, axis=1).fillna(m.med)
    test["logit"] = m.logit.predict_proba(te)[:, 1]
    return test


def forecast_asof(p: pd.DataFrame, as_of: pd.Period) -> pd.DataFrame:
    """Прогноз в точке as_of для всех стран со своей валютой: всё — по данным ≤ as_of."""
    m = fit(p, as_of)
    test = p[(p.month == as_of) & p.own_currency & p.fx.notna()]
    return predict(m, test)


def run_models(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = []

    def step(T: int) -> pd.DataFrame | None:
        # учимся на том, что известно к концу декабря T−1; предсказываем все месяцы T
        m = fit(p, pd.Period(f"{T - 1}-12", freq="M"))
        test = p[(p.month.dt.year == T) & p.evaluable]
        if m is None or test.empty:
            return None
        test = predict(m, test)
        for ind, par in m.sig_model.items():
            thresholds.append({"year": T, "indicator": ind, **par})
        print(f"  {T}: обучение {len(m.known)} мес ({int(m.known.y.sum())} кризисных окон), "
              f"тест {len(test)} мес ({int(test.y.sum())}), сигналов в модели {len(m.sig_model)}")
        return test

    pred = backtest(range(FIRST_EVAL_YEAR, LAST_EVAL_YEAR + 1), step)
    return pred, pd.DataFrame(thresholds)


# ---------------------------------------------------------------------------
# Оценка и таблицы
# ---------------------------------------------------------------------------

MODELS = ["base_rate", "pressure", "signals", "logit"]


def metrics(pred: pd.DataFrame, mask: pd.Series | None = None) -> pd.DataFrame:
    d = pred if mask is None else pred[mask]
    rows = []
    for m in MODELS:
        ok = d[m].notna()
        if d.y[ok].nunique() < 2:
            continue
        # AUC внутри года: умеет ли модель отличать страны друг от друга в один и тот же год
        # (сквозной AUC ещё и штрафует за то, что общий уровень риска по годам не угадан)
        rows.append({"model": m, "n": int(ok.sum()), "crisis_windows": int(d.y[ok].sum()),
                     "brier": brier(d.y[ok], d[m][ok]),
                     "auc": auc(d.y[ok], d[m][ok]),
                     "auc_within_year": auc_within(d.y[ok], d[m][ok], d.month[ok].dt.year)})
    return pd.DataFrame(rows)


def metrics_by_year(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for T, d in pred.groupby(pred.month.dt.year):
        for m in MODELS:
            if d.y.nunique() < 2:
                continue
            rows.append({"year": T, "model": m, "n": len(d), "crisis_windows": int(d.y.sum()),
                         "brier": brier(d.y, d[m]), "auc": auc(d.y, d[m], constant_ok=True)})
    return pd.DataFrame(rows)


def pre_crisis_table(p: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Для каждого постсоветского кризиса: прогноз за 12, 6 и 3 месяца до него и место
    страны среди всех стран в тот месяц (1 = самый высокий риск)."""
    rows = []
    crises = p[(p.crisis == 1) & p.post_soviet]
    for _, c in crises.iterrows():
        row = {"country": ISO3_TO_SHORT.get(c.iso3, c.iso3), "iso3": c.iso3, "crisis_month": str(c.month),
               "de": round(c.de, 3), "dr": round(c.dr, 3) if pd.notna(c.dr) else np.nan,
               "emp_z": round(c.emp_z, 1)}
        for lead in (12, 6, 3):
            m = c.month - lead
            snap = pred[pred.month == m]
            me = snap[snap.iso3 == c.iso3]
            for model in ("pressure", "signals", "logit"):
                if me.empty or me[model].isna().all():
                    row[f"{model}_{lead}m"] = np.nan
                    row[f"{model}_rank_{lead}m"] = np.nan
                    continue
                v = me[model].iloc[0]
                row[f"{model}_{lead}m"] = round(v, 3)
                row[f"{model}_rank_{lead}m"] = int((snap[model] > v).sum() + 1)
            row[f"n_countries_{lead}m"] = len(snap)
        rows.append(row)
    return pd.DataFrame(rows)


def coverage(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iso, g in p[p.post_soviet].groupby("iso3"):
        r = {"country": ISO3_TO_SHORT[iso], "iso3": iso}
        for c in ("fx", "reserves", "cpi", "rate", "lending", "deposit", "broad_money", "exports"):
            s = g[g[c].notna()]
            r[f"{c}_from"] = str(s.month.min()) if len(s) else ""
            r[f"{c}_to"] = str(s.month.max()) if len(s) else ""
        r["cpi_filled_months"] = int(g.cpi_filled.sum())
        r["n_crises"] = int(g.crisis.sum())
        r["n_deval15"] = int(g.deval15.sum())
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    target = next((a for a in argv if a in ("crisis", "deval15")), "crisis")
    drop_shared = "all_currencies" not in argv  # `all_currencies` — срез «как до аудита», с еврозоной и т.п.
    suffix = ("" if target == "crisis" else f"_{target}") + ("" if drop_shared else "_allcur")
    print("Панель МВФ…")
    p = build_panel(load_panel(), load_annual_cpi(), load_war(), target=target, drop_shared=drop_shared)
    print(f"  {p.iso3.nunique()} стран, {len(p)} страна-месяцев, {int(p.crisis.sum())} кризисов "
          f"по EMP (3σ), {int(p.deval15.sum())} по deval15")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    p.assign(month=p.month.astype(str)).to_parquet(DATA_PROCESSED / f"econ_kr_panel{suffix}.parquet", index=False)

    cov = coverage(p)
    cov.to_csv(REPORTS / f"econ_kr_coverage{suffix}.csv", index=False, float_format=FLOAT_FORMAT)
    print("\nПокрытие постсоветских стран:")
    print(cov[["country", "fx_from", "reserves_from", "cpi_from", "cpi_filled_months", "rate_from", "n_crises", "n_deval15"]].to_string(index=False))

    cr = p[(p.crisis == 1) & p.post_soviet][["iso3", "month", "de", "dr", "emp_z"]].copy()
    cr["country"] = cr.iso3.map(ISO3_TO_SHORT)
    cr["month"] = cr.month.astype(str)
    cr.to_csv(REPORTS / f"econ_kr_crises_post_soviet{suffix}.csv", index=False, float_format=FLOAT_FORMAT)
    print("\nПостсоветские кризисы по EMP:")
    print(cr[["country", "month", "de", "dr", "emp_z"]].round(2).to_string(index=False))

    print("\nМодели вслепую…")
    pred, thr = run_models(p)
    pred.assign(month=pred.month.astype(str)).to_parquet(DATA_PROCESSED / f"econ_kr_predictions{suffix}.parquet", index=False)
    thr.to_csv(REPORTS / f"econ_kr_thresholds{suffix}.csv", index=False, float_format=FLOAT_FORMAT)

    m_all = metrics(pred).assign(sample="world")
    m_ps = metrics(pred, pred.post_soviet).assign(sample="post_soviet")
    m = pd.concat([m_all, m_ps])
    m.to_csv(REPORTS / f"econ_kr_metrics{suffix}.csv", index=False, float_format=FLOAT_FORMAT)
    print(f"\nМетрики вслепую {FIRST_EVAL_YEAR}–{LAST_EVAL_YEAR}, целевая {target} в ближайшие {H} мес:")
    print(m.round(4).to_string(index=False))
    by = metrics_by_year(pred)
    by.to_csv(REPORTS / f"econ_kr_metrics_by_year{suffix}.csv", index=False, float_format=FLOAT_FORMAT)
    wins = by.pivot(index="year", columns="model", values="auc")
    print(f"\n  по AUC лучше base_rate: pressure в {int((wins.pressure > wins.base_rate).sum())} годах из {len(wins)}, "
          f"signals в {int((wins.signals > wins.base_rate).sum())}, logit в {int((wins.logit > wins.base_rate).sum())}; "
          f"signals лучше pressure в {int((wins.signals > wins.pressure).sum())}, "
          f"logit лучше pressure в {int((wins.logit > wins.pressure).sum())}")

    if not thr.empty:
        t = thr.groupby("indicator").agg(pct=("pct", "median"), nsr=("nsr", "median"),
                                         years_used=("year", "count")).sort_values("nsr")
        print("\nИндикаторы в методе сигналов (медианы по годам): порог-процентиль, шум/сигнал, лет в модели")
        print(t.round(2).to_string())

    tab = pre_crisis_table(p, pred)
    tab.to_csv(REPORTS / f"econ_kr_pre_crisis_table{suffix}.csv", index=False, float_format=FLOAT_FORMAT)
    print("\nПостсоветские кризисы: место среди стран мира (1 = самый рискованный) за 12 / 6 / 3 месяца до:")
    cols = ["country", "crisis_month", "pressure_rank_12m", "signals_rank_12m", "logit_rank_12m",
            "pressure_rank_6m", "signals_rank_6m", "logit_rank_6m",
            "pressure_rank_3m", "signals_rank_3m", "logit_rank_3m", "n_countries_12m"]
    print(tab[cols].to_string(index=False))
    return 0

