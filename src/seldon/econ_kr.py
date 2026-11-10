"""Валютные кризисы по Kaminsky–Lizondo–Reinhart (1998) на месячных данных МВФ.

Запуск:
    uv run python -m seldon.econ_kr

Что делает:
  1. Месячная панель страна×месяц по всем странам из данных МВФ (`download imf`): курс
     к доллару (конец месяца), резервы без золота (долл.), ставки, ИПЦ.
  2. Индекс давления на валютный рынок (EMP — exchange market pressure):
         EMP = %Δe − (σ_e / σ_R) · %ΔR,
     где e — курс (рост = ослабление), R — резервы, σ — стандартные отклонения по стране.
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
     максимум EMP самой страны за последние 3 мес (давление уже идёт).
  4. Целевая: кризис в ближайшие 12 месяцев. 12 месяцев после кризиса из оценки выброшены.
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

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, DATA_RAW, PROJECT_ROOT
from seldon.countries import ISO3_TO_SHORT, POST_SOVIET_ISO3

REPORTS = PROJECT_ROOT / "reports"
IMF = DATA_RAW / "imf"

FIRST_MONTH = "1993-01"
LAST_MONTH = "2025-12"
FIRST_EVAL_YEAR = 1996
LAST_EVAL_YEAR = 2025
H = 12  # горизонт: кризис в ближайшие 12 месяцев
POST_CRISIS_DROP = 12  # месяцев после кризиса, которые не оцениваем
EMP_SIGMA = 3.0
EMP_FLOOR = 0.10  # и не меньше 10% давления: у жёстких привязок σ≈0, любой сдвиг — «3σ»
HYPERINFL_6M = 1.5  # 6-месячная инфляция > 150% — гиперинфляция (KLR)
DEVAL15 = 0.15
MIN_HISTORY = 36  # месяцев истории страны для расчёта σ и процентилей
PERCENTILES = [5, 10, 15, 20, 25, 30]  # хвост распределения, кандидаты порога (KLR: 10–20)

# индикатор → в каком хвосте сигнал: "high" — сигнал при больших значениях, "low" — при малых
INDICATORS = {
    "reserves_g12": "low",   # резервы упали за 12 мес
    "rer_dev": "low",        # реальный курс крепче тренда (переоценка): e·P*/P ниже тренда
    "deval12": "high",       # уже девальвировались за 12 мес
    "infl12": "high",        # инфляция за 12 мес
    "real_rate": "high",     # реальная ставка (защита курса ставкой)
    "rate_diff_us": "high",  # разница ставок с США
    "spread": "high",        # кредитная минус депозитная
    "reserves_g1": "low",    # резервы за последний месяц
    "m2_res_g12": "high",    # M2 (в долл.) к резервам: рост за 12 мес — лучший индикатор KLR
    "exports_g12": "low",    # экспорт упал за 12 мес
    "world_crisis3": "high", # заражение: доля стран мира с кризисом за последние 3 мес
    "emp_z_max3": "high",    # давление уже идёт: максимум EMP (в σ) за последние 3 мес
}

MMRT = "MMRT_RT_PT_A_PT"
RATE_PRIORITY = ["MMRT_RT_PT_A_PT", "IRTBILL_RT_PT_A_PT", "MFS166_RT_PT_A_PT",
                 "DISR_RT_PT_A_PT", "MFS171_RT_PT_A_PT"]


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------


def read_imf(name: str) -> pd.DataFrame:
    d = pd.read_csv(IMF / f"{name}.csv", usecols=lambda c: c in {
        "COUNTRY", "INDICATOR", "UNIT", "TYPE_OF_TRANSFORMATION", "TIME_PERIOD", "OBS_VALUE"})
    d = d[d.OBS_VALUE.notna()].copy()
    d["month"] = pd.PeriodIndex(d.TIME_PERIOD.str.replace("-M", "-"), freq="M")
    return d.rename(columns={"COUNTRY": "iso3", "OBS_VALUE": "value"})


def load_panel() -> pd.DataFrame:
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
    p = p[(p.month >= FIRST_MONTH) & (p.month <= LAST_MONTH)]
    # полная сетка месяцев внутри страны, чтобы лаги считались по календарю, а не по строкам
    full = p.groupby("iso3").month.agg(["min", "max"])
    idx = pd.MultiIndex.from_tuples(
        [(c, m) for c, r in full.iterrows() for m in pd.period_range(r["min"], r["max"], freq="M")],
        names=["iso3", "month"])
    p = p.set_index(["iso3", "month"]).reindex(idx).reset_index()
    # 0 в курсе/резервах — пропуск, не значение
    for c in ("fx", "reserves", "cpi", "broad_money", "exports"):
        p.loc[p[c] <= 0, c] = np.nan
    p["post_soviet"] = p.iso3.isin(POST_SOVIET_ISO3)
    return p.sort_values(["iso3", "month"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# EMP, кризисы, индикаторы
# ---------------------------------------------------------------------------


def _pct_change(s: pd.Series, k: int) -> pd.Series:
    return s / s.shift(k) - 1


def add_emp_and_crises(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    gb = p.groupby("iso3")
    p["de"] = gb.fx.transform(lambda s: _pct_change(s, 1))
    p["dr"] = gb.reserves.transform(lambda s: _pct_change(s, 1))
    p["infl6"] = gb.cpi.transform(lambda s: _pct_change(s, 6))
    # гиперинфляция: по ИПЦ, а где ИПЦ ещё нет (у Украины месячный ИПЦ в МВФ только с 2005)
    # — по самому курсу, но только в переходные 1992–1995 (скачок > 50% за месяц)
    early = p.month < pd.Period("1996-01", freq="M")
    p["hyper"] = (p.infl6 > HYPERINFL_6M) | (p.infl6.isna() & early & (p.de.abs() > 0.5))
    # экстремальные сдвиги в резервах (переоценка, смена методики) режем, чтобы не ломать σ
    p["dr"] = p.dr.clip(-0.9, 2.0)
    p["de"] = p.de.clip(-0.9, 5.0)

    def sigmas(g: pd.DataFrame) -> pd.Series:
        calm = g[~g.hyper]
        return pd.Series({"sd_e": _robust_sd(calm.de), "sd_r": _robust_sd(calm.dr),
                          "n_calm": calm.de.notna().sum()})

    sd = p.groupby("iso3").apply(sigmas, include_groups=False)
    p = p.merge(sd, left_on="iso3", right_index=True, how="left")
    w = (p.sd_e / p.sd_r).replace([np.inf, -np.inf], np.nan)
    # если резервов нет — EMP только по курсу
    p["emp"] = np.where(p.dr.notna(), p.de - w * p.dr, p.de)
    p.loc[p.de.isna(), "emp"] = np.nan
    calm = p[~p.hyper]
    stats = calm.groupby("iso3").emp.agg(emp_mean="median", emp_sd=_robust_sd)
    p = p.merge(stats, left_on="iso3", right_index=True, how="left")
    p["emp_z"] = (p.emp - p.emp_mean) / p.emp_sd
    spike = (p.emp_z > EMP_SIGMA) & (p.emp >= EMP_FLOOR) & (p.n_calm >= MIN_HISTORY) & ~p.hyper
    p["crisis"] = _onsets(spike, p.iso3)
    p["deval15"] = _onsets((p.de >= DEVAL15) & ~p.hyper, p.iso3)
    return p


def _robust_sd(s: pd.Series) -> float:
    """σ через медианное абсолютное отклонение (MAD·1,4826): редкие скачки курса —
    сами кризисы — не раздувают масштаб «спокойных» месяцев. Классическая σ у России
    из-за 1993–95 и 1998 такая, что 2009 и 2014 не дотягивают до 3σ."""
    s = s.dropna()
    if len(s) < 12:
        return np.nan
    mad = (s - s.median()).abs().median() * 1.4826
    return max(mad, 1e-4)  # фиксированный курс: MAD = 0, тогда любой сдвиг — всплеск


def _onsets(spike: pd.Series, iso3: pd.Series) -> pd.Series:
    """Первый всплеск после ≥ POST_CRISIS_DROP спокойных месяцев = кризис."""
    out = np.zeros(len(spike), dtype=int)
    last = {}
    for i, (c, s) in enumerate(zip(iso3.to_numpy(), spike.to_numpy())):
        if s and (c not in last or i - last[c] > POST_CRISIS_DROP):
            out[i] = 1
            last[c] = i
        elif s:
            last[c] = i  # продолжение того же кризиса продлевает «тишину»
    return pd.Series(out, index=spike.index)


def add_indicators(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    us = p[p.iso3 == "USA"].set_index("month")[["cpi", "rate"]].rename(
        columns={"cpi": "cpi_us", "rate": "rate_us"})
    p = p.merge(us, left_on="month", right_index=True, how="left")
    gb = p.groupby("iso3")
    p["reserves_g12"] = gb.reserves.transform(lambda s: _pct_change(s, 12)).clip(-1, 3)
    p["reserves_g1"] = p.dr
    p["deval12"] = gb.fx.transform(lambda s: _pct_change(s, 12)).clip(-0.9, 5)
    p["infl12"] = gb.cpi.transform(lambda s: _pct_change(s, 12)).clip(-0.5, 5)
    # реальный курс: e · P_US / P. Рост = реальное ослабление. Отклонение от тренда —
    # отношение к скользящему среднему за 36 мес (только прошлое: без утечки будущего)
    p["rer"] = p.fx * p.cpi_us / p.cpi
    trend = gb.rer.transform(lambda s: s.rolling(36, min_periods=24).mean())
    p["rer_dev"] = (p.rer / trend - 1).clip(-0.9, 3)
    p["real_rate"] = (p.rate - p.infl12 * 100).clip(-100, 200)
    p["rate_diff_us"] = (p.rate - p.rate_us).clip(-50, 200)
    p["spread"] = (p.lending - p.deposit).clip(-20, 100)
    p["m2_res"] = p.broad_money / p.fx / p.reserves
    p["m2_res_g12"] = gb.m2_res.transform(lambda s: _pct_change(s, 12)).clip(-0.9, 5)
    # экспорт помесячно шумный и сезонный: сравниваем суммы за 12 мес
    x12 = gb.exports.transform(lambda s: s.rolling(12, min_periods=12).sum())
    p["exports_g12"] = (x12 / x12.shift(12) - 1).where(p.iso3 == p.iso3.shift(12)).clip(-0.9, 3)
    # всё ниже известно на момент прогноза: кризисы прошлых месяцев и собственный EMP
    world = p.groupby("month").agg(n=("crisis", "count"), k=("crisis", "sum"))
    share = (world.k / world.n).rolling(3, min_periods=1).sum()
    p["world_crisis3"] = p.month.map(share)
    p["emp_z_max3"] = gb.emp_z.transform(lambda s: s.rolling(3, min_periods=1).max()).clip(-5, 50)
    return p


def add_target(p: pd.DataFrame, target: str = "crisis") -> pd.DataFrame:
    p = p.copy()
    gb = p.groupby("iso3")[target]
    fwd = sum(gb.shift(-k) for k in range(1, H + 1))
    p["y"] = (fwd > 0).astype(float)
    p.loc[fwd.isna(), "y"] = np.nan  # конец ряда: окно не полностью известно
    # 12 месяцев после кризиса (включая его месяц) не оцениваем
    recent = sum(gb.shift(k).fillna(0) for k in range(0, POST_CRISIS_DROP))
    p["evaluable"] = (recent == 0) & p.y.notna()
    return p


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
    thr, pooled = _thresholds(ref, ind, pct, side)
    t = df.iso3.map(thr).fillna(pooled)
    x = df[ind]
    sig = (x <= t) if side == "low" else (x >= t)
    return sig.astype(float).where(x.notna())


def composite(df: pd.DataFrame, model: dict, ref: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    for ind, m in model.items():
        sig = _signal(df, ind, INDICATORS[ind], m["pct"], ref).fillna(0)
        score += sig / m["nsr"]
    return score


def calibrate(score_train: pd.Series, y_train: pd.Series, score_test: pd.Series) -> pd.Series:
    """Вероятность = доля кризисных окон в обучении среди наблюдений с таким же сводным
    индексом (по децилям ненулевых значений, ноль — отдельно)."""
    nz = score_train[score_train > 0]
    edges = np.unique(np.quantile(nz, np.linspace(0, 1, 11))) if len(nz) > 50 else np.array([0, np.inf])
    edges = np.concatenate([[-np.inf, 0], edges[1:-1], [np.inf]])
    b_tr = pd.cut(score_train, edges, labels=False, include_lowest=True)
    b_te = pd.cut(score_test, edges, labels=False, include_lowest=True)
    rate = y_train.groupby(b_tr).mean()
    # монотонно не убывающая по бину, чтобы редкие верхние бины не проваливались
    rate = rate.cummax()
    return b_te.map(rate).fillna(y_train.mean())


def run_models(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = list(INDICATORS)
    preds, thresholds = [], []
    for T in range(FIRST_EVAL_YEAR, LAST_EVAL_YEAR + 1):
        origin = pd.Period(f"{T}-01", freq="M")
        # к 1 января T известны метки для месяцев m с m + H ≤ декабрь T−1
        known = p[(p.month + H < origin) & p.evaluable]
        test = p[(p.month.dt.year == T) & p.evaluable].copy()
        ref = p[p.month < origin]  # для порогов: вся история до T (метки не нужны)
        if known.y.sum() < 5 or test.empty:
            continue
        test["base_rate"] = known.y.mean()
        # персистентность: вероятность кризиса в обучении при таком же давлении за 3 мес
        # (как `decay` в блоке войн: «где трясёт — там тряхнёт ещё»)
        test["pressure"] = calibrate(known.emp_z_max3.fillna(0), known.y, test.emp_z_max3.fillna(0)).to_numpy()
        sig_model = fit_signals(known)
        s_tr, s_te = composite(known, sig_model, ref), composite(test, sig_model, ref)
        test["signals_score"] = s_te
        test["signals"] = calibrate(s_tr, known.y, s_te).to_numpy()
        for ind, m in sig_model.items():
            thresholds.append({"year": T, "indicator": ind, **m})
        # логрегрессия: пропуски → медиана обучения, обрезка по 1/99 процентилям
        tr = known[feats].copy()
        lo, hi = tr.quantile(0.01), tr.quantile(0.99)
        med = tr.median().fillna(0.0)  # ряд пуст во всём обучении (экспорт в 1990-е) → 0
        tr = tr.clip(lo, hi, axis=1).fillna(med)
        te = test[feats].clip(lo, hi, axis=1).fillna(med)
        logit = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
        logit.fit(tr, known.y)
        test["logit"] = logit.predict_proba(te)[:, 1]
        preds.append(test)
        print(f"  {T}: обучение {len(known)} мес ({int(known.y.sum())} кризисных окон), "
              f"тест {len(test)} мес ({int(test.y.sum())}), сигналов в модели {len(sig_model)}")
    return pd.concat(preds, ignore_index=True), pd.DataFrame(thresholds)


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
        rows.append({"model": m, "n": int(ok.sum()), "crisis_windows": int(d.y[ok].sum()),
                     "brier": brier_score_loss(d.y[ok], d[m][ok]),
                     "auc": roc_auc_score(d.y[ok], d[m][ok])})
    return pd.DataFrame(rows)


def metrics_by_year(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for T, d in pred.groupby(pred.month.dt.year):
        for m in MODELS:
            if d.y.nunique() < 2:
                continue
            rows.append({"year": T, "model": m, "n": len(d), "crisis_windows": int(d.y.sum()),
                         "brier": brier_score_loss(d.y, d[m]), "auc": roc_auc_score(d.y, d[m])})
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
        r["n_crises"] = int(g.crisis.sum())
        r["n_deval15"] = int(g.deval15.sum())
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    target = argv[0] if argv else "crisis"
    suffix = "" if target == "crisis" else f"_{target}"
    print("Панель МВФ…")
    p = load_panel()
    p = add_emp_and_crises(p)
    p = add_indicators(p)
    p = add_target(p, target)
    print(f"  {p.iso3.nunique()} стран, {len(p)} страна-месяцев, {int(p.crisis.sum())} кризисов "
          f"по EMP (3σ), {int(p.deval15.sum())} по deval15")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    p.assign(month=p.month.astype(str)).to_parquet(DATA_PROCESSED / f"econ_kr_panel{suffix}.parquet", index=False)

    cov = coverage(p)
    cov.to_csv(REPORTS / "econ_kr_coverage.csv", index=False)
    print("\nПокрытие постсоветских стран:")
    print(cov[["country", "fx_from", "reserves_from", "cpi_from", "rate_from", "n_crises", "n_deval15"]].to_string(index=False))

    cr = p[(p.crisis == 1) & p.post_soviet][["iso3", "month", "de", "dr", "emp_z"]].copy()
    cr["country"] = cr.iso3.map(ISO3_TO_SHORT)
    cr["month"] = cr.month.astype(str)
    cr.to_csv(REPORTS / "econ_kr_crises_post_soviet.csv", index=False)
    print("\nПостсоветские кризисы по EMP:")
    print(cr[["country", "month", "de", "dr", "emp_z"]].round(2).to_string(index=False))

    print("\nМодели вслепую…")
    pred, thr = run_models(p)
    pred.assign(month=pred.month.astype(str)).to_parquet(DATA_PROCESSED / f"econ_kr_predictions{suffix}.parquet", index=False)
    thr.to_csv(REPORTS / f"econ_kr_thresholds{suffix}.csv", index=False)

    m_all = metrics(pred).assign(sample="world")
    m_ps = metrics(pred, pred.post_soviet).assign(sample="post_soviet")
    m = pd.concat([m_all, m_ps])
    m.to_csv(REPORTS / f"econ_kr_metrics{suffix}.csv", index=False)
    print(f"\nМетрики вслепую {FIRST_EVAL_YEAR}–{LAST_EVAL_YEAR}, целевая {target} в ближайшие {H} мес:")
    print(m.round(4).to_string(index=False))
    by = metrics_by_year(pred)
    by.to_csv(REPORTS / f"econ_kr_metrics_by_year{suffix}.csv", index=False)
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
    tab.to_csv(REPORTS / f"econ_kr_pre_crisis_table{suffix}.csv", index=False)
    print("\nПостсоветские кризисы: место среди стран мира (1 = самый рискованный) за 12 / 6 / 3 месяца до:")
    cols = ["country", "crisis_month", "pressure_rank_12m", "signals_rank_12m", "logit_rank_12m",
            "pressure_rank_6m", "signals_rank_6m", "logit_rank_6m",
            "pressure_rank_3m", "signals_rank_3m", "logit_rank_3m", "n_countries_12m"]
    print(tab[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
