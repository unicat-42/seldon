"""Валютные кризисы по Kaminsky–Lizondo–Reinhart (1998) на месячных данных МВФ.

Запуск:
    uv run python -m seldon.econ_kr [crisis|deval15] [all_currencies]

Что делает:
  1. Месячная панель страна×месяц по всем странам из данных МВФ (`download imf`): курс
     к доллару (конец месяца), резервы без золота (долл.), ставки, ИПЦ. Дыры в месячном
     ИПЦ затыкаем годовым ИПЦ ВБ (`download wdi`) экстраполяцией из последнего
     завершённого года — без утечки (см. `fill_cpi`).
     Страны с общей или чужой валютой (`countries.SHARED_CURRENCY`) исключаются с года
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

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, DATA_RAW, PROJECT_ROOT
from seldon.countries import GW_TO_ISO3, ISO3_TO_SHORT, POST_SOVIET_ISO3, shared_currency_table

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
    # война (UCDP, panel_country_month): связка блоков «война → кризис»
    "war25_now": "high",     # государственная война ≥ 25 погибших в этом месяце (у себя или у соседа-стороной)
    "war1000_now": "high",   # ≥ 1000 погибших за последние 12 месяцев
    "war_onset12": "high",   # начало войны (после ≥ 12 тихих месяцев) за последние 12 мес
}
BINARY = {"war25_now", "war1000_now", "war_onset12"}  # сигнал = 1, без порога-процентиля

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


def load_panel(last_month: str = LAST_MONTH) -> pd.DataFrame:
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


def load_annual_cpi() -> pd.DataFrame:
    """Годовой ИПЦ Всемирного банка: iso3, year, value."""
    return pd.read_csv(DATA_RAW / "wdi" / "cpi_annual.csv")[["iso3", "year", "value"]]


def fill_cpi(p: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Дыры в месячном ИПЦ МВФ затыкаем годовым ИПЦ Всемирного банка — без утечки будущего.

    Индекс цен собирается цепочкой из месячных ТЕМПОВ, а не уровней:
      - если ИПЦ МВФ есть и в этом месяце, и в прошлом — темп из МВФ;
      - иначе для месяца года t — темп последнего завершённого года:
        ln(A(t−1)/A(t−2)) / 12, где A(y) — годовое среднее ВБ (известно к началу t).
    Уровень индекса произвольный (везде используются только темпы и отношение к своему
    тренду). До аудита уровень заполнения подгонялся к первому БУДУЩЕМУ наблюдению МВФ
    (утечка), а темп внутри года был занижен в 12 раз и давал скачок в январе.
    Флаг `cpi_filled` отмечает месяцы, где темп взят из годовых данных."""
    p = p.sort_values(["iso3", "month"]).reset_index(drop=True)
    a = annual[annual.value > 0].sort_values(["iso3", "year"]).copy()
    a["prev"] = a.groupby("iso3").value.shift(1)
    a["prev_year"] = a.groupby("iso3").year.shift(1)
    a = a[a.prev_year == a.year - 1]  # нужны два подряд идущих года
    a["g_annual"] = np.log(a.value / a.prev) / 12  # месячный лог-темп последнего завершённого года
    a["year"] = a.year + 1  # применяем к следующему году
    p["year"] = p.month.dt.year
    p = p.merge(a[["iso3", "year", "g_annual"]], on=["iso3", "year"], how="left")
    same = p.iso3 == p.iso3.shift(1)
    g_imf = np.log(p.cpi / p.cpi.shift(1)).where(same)
    g = g_imf.fillna(p.g_annual)
    p["cpi_filled"] = g_imf.isna() & p.g_annual.notna()
    # цепочка: отрезки подряд идущих известных темпов; месяц перед отрезком — база (уровень 1)
    seg = g.isna().groupby(p.iso3).cumsum()
    level = np.exp(g.fillna(0.0).groupby([p.iso3, seg]).cumsum())
    isolated = g.isna() & (g.shift(-1).isna() | (p.iso3 != p.iso3.shift(-1)))
    p["cpi"] = level.where(~isolated)
    return p.drop(columns=["year", "g_annual"])


# ---------------------------------------------------------------------------
# EMP, кризисы, индикаторы
# ---------------------------------------------------------------------------


def _pct_change(s: pd.Series, k: int) -> pd.Series:
    return s / s.shift(k) - 1


def add_emp_and_crises(p: pd.DataFrame, drop_shared: bool = True) -> pd.DataFrame:
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

    # Все параметры EMP — расширяющимся окном: в месяце m используются только месяцы ≤ m.
    # (До аудита σ, медиана и число спокойных месяцев считались один раз по всей выборке,
    # так что индекс 1998 года «знал» норму 2020-х.)
    p["own_currency"] = True
    if drop_shared:
        reg = shared_currency_table(int(p.month.dt.year.min()), int(p.month.dt.year.max()))
        yr = pd.DataFrame({"iso3": p.iso3, "year": p.month.dt.year}).merge(reg, on=["iso3", "year"], how="left")
        p["fx_regime"] = yr.fx_regime.to_numpy()
        p["own_currency"] = p.fx_regime.isna()
    parts = []
    for _, g in p.groupby("iso3", sort=False):
        de, dr, calm = g.de.to_numpy(), g.dr.to_numpy(), ~g.hyper.to_numpy()
        _, sd_e, n_calm = _expanding_robust(de, calm)
        _, sd_r, _ = _expanding_robust(dr, calm)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = sd_e / sd_r
        w[~np.isfinite(w)] = np.nan
        emp = np.where(np.isnan(dr), de, de - w * dr)  # если резервов нет — EMP только по курсу
        emp[np.isnan(de)] = np.nan
        emp_mean, emp_sd, _ = _expanding_robust(emp, calm)
        parts.append(pd.DataFrame({"sd_e": sd_e, "sd_r": sd_r, "n_calm": n_calm, "emp": emp,
                                   "emp_mean": emp_mean, "emp_sd": emp_sd}, index=g.index))
    p = p.join(pd.concat(parts))
    p["emp_z"] = (p.emp - p.emp_mean) / p.emp_sd
    spike = (p.emp_z > EMP_SIGMA) & (p.emp >= EMP_FLOOR) & (p.n_calm >= MIN_HISTORY) & ~p.hyper & p.own_currency
    p["spike_crisis"] = spike.fillna(False)
    p["crisis"], p["quiet_crisis"] = onsets(p.spike_crisis, p.iso3)
    p["spike_deval15"] = ((p.de >= DEVAL15) & ~p.hyper & p.own_currency).fillna(False)
    p["deval15"], p["quiet_deval15"] = onsets(p.spike_deval15, p.iso3)
    return p


def _expanding_robust(x: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Медиана, устойчивая σ и число наблюдений по значениям x[s], s ≤ t (только где ok и не NaN)."""
    n = len(x)
    med, sd, cnt = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    buf, k = np.empty(n), 0
    cur_med = cur_sd = np.nan
    for t in range(n):
        if ok[t] and not np.isnan(x[t]):
            buf[k] = x[t]
            k += 1
            if k >= 12:
                cur_med = np.median(buf[:k])
                cur_sd = max(np.median(np.abs(buf[:k] - cur_med)) * 1.4826, 1e-4)
        med[t], sd[t], cnt[t] = cur_med, cur_sd, k
    return med, sd, cnt


def _robust_sd(s: pd.Series) -> float:
    """σ через медианное абсолютное отклонение (MAD·1,4826): редкие скачки курса —
    сами кризисы — не раздувают масштаб «спокойных» месяцев. Классическая σ у России
    из-за 1993–95 и 1998 такая, что 2009 и 2014 не дотягивают до 3σ."""
    s = s.dropna()
    if len(s) < 12:
        return np.nan
    mad = (s - s.median()).abs().median() * 1.4826
    return max(mad, 1e-4)  # фиксированный курс: MAD = 0, тогда любой сдвиг — всплеск


def onsets(spike: pd.Series, iso3: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Первый всплеск после ≥ POST_CRISIS_DROP спокойных месяцев = кризис.
    Возвращает (кризис 0/1, `quiet` — месяцев с последнего всплеска, NaN если не было).
    Повторный всплеск продлевает «тишину»; по `quiet` строится `evaluable`, чтобы оценка
    и блокировка считались от одного и того же — от последнего ВСПЛЕСКА."""
    out = np.zeros(len(spike), dtype=int)
    quiet = np.full(len(spike), np.nan)
    last = {}
    for i, (c, s) in enumerate(zip(iso3.to_numpy(), spike.to_numpy())):
        if s:
            if c not in last or i - last[c] > POST_CRISIS_DROP:
                out[i] = 1
            last[c] = i
        if c in last:
            quiet[i] = i - last[c]
    return pd.Series(out, index=spike.index), pd.Series(quiet, index=spike.index)


def add_indicators(p: pd.DataFrame, war: pd.DataFrame | None = None) -> pd.DataFrame:
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
    live = p[p.own_currency & p.de.notna()]
    world = live.groupby("month").agg(n=("crisis", "count"), k=("crisis", "sum"))
    share = (world.k / world.n).rolling(3, min_periods=1).sum()
    p["world_crisis3"] = p.month.map(share)
    p["emp_z_max3"] = gb.emp_z.transform(lambda s: s.rolling(3, min_periods=1).max()).clip(-5, 50)
    if war is None:
        war = load_war()
    return p.merge(war[["iso3", "month", *WAR_COLS]], on=["iso3", "month"], how="left")


WAR_COLS = ["war25_now", "war1000_now", "war_onset12"]


def war_features(month_panel: pd.DataFrame, gw_to_iso3: dict | None = None) -> pd.DataFrame:
    """Война по месяцам из блока войн (panel.py, коды GW) → iso3, month и три признака.
    Всё на строке месяца m посчитано по событиям ≤ m (см. panel.py):
      war25_now    — государственная война: ≥ 25 погибших за месяц у себя или у соседа,
                     где страна — сторона (`state_war25`);
      war1000_now  — ≥ 1000 погибших за последние 12 месяцев (скользящая сумма);
      war_onset12  — начало войны (после ≥ 12 тихих месяцев) за последние 12 мес."""
    w = month_panel[["gw", "year", "month", "state_war25", "state_war1000_12m", "onset_state_war25"]].copy()
    w["iso3"] = w.gw.map(GW_TO_ISO3 if gw_to_iso3 is None else gw_to_iso3)
    w = w[w.iso3.notna()].copy()
    w["month"] = pd.PeriodIndex(pd.to_datetime(dict(year=w.year, month=w.month, day=1)), freq="M")
    w = w.sort_values(["iso3", "gw", "month"])
    w["war_onset12"] = w.groupby("gw").onset_state_war25.transform(lambda s: s.rolling(12, min_periods=1).max())
    w = w.drop_duplicates(["iso3", "month"], keep="first")  # 340/345 → SRB
    w = w.rename(columns={"state_war25": "war25_now", "state_war1000_12m": "war1000_now"})
    return w[["iso3", "month", *WAR_COLS]].reset_index(drop=True)


def load_war() -> pd.DataFrame:
    return war_features(pd.read_parquet(DATA_PROCESSED / "panel_country_month.parquet"))


def add_target(p: pd.DataFrame, target: str = "crisis") -> pd.DataFrame:
    p = p.copy()
    gb = p.groupby("iso3")[target]
    fwd = sum(gb.shift(-k) for k in range(1, H + 1))
    p["y"] = (fwd > 0).astype(float)
    p.loc[fwd.isna(), "y"] = np.nan  # конец ряда: окно не полностью известно
    # Строку оцениваем, только если кризис в её окне МОЖЕТ быть засчитан: с последнего
    # всплеска прошло ≥ 12 месяцев (то же правило, что в `onsets`), валюта своя, а для EMP
    # ещё и накоплена история для σ. До аудита «тишина» считалась от начала кризиса, и
    # строки посреди затяжного кризиса возвращались в оценку с принудительным y = 0.
    quiet = p[f"quiet_{target}"]
    ok = (quiet.isna() | (quiet >= POST_CRISIS_DROP)) & p.own_currency & p.de.notna()
    if target == "crisis":
        ok &= p.n_calm >= MIN_HISTORY
    p["evaluable"] = ok & p.y.notna()
    return p


def build_panel(raw: pd.DataFrame, annual_cpi: pd.DataFrame, war: pd.DataFrame | None = None,
                target: str = "crisis", drop_shared: bool = True) -> pd.DataFrame:
    """Сырые ряды → панель с EMP, кризисами, индикаторами и целевой. Чистая функция:
    значения на строке месяца m зависят только от данных ≤ m (кроме `y` — это будущее)."""
    p = fill_cpi(raw, annual_cpi)
    p = add_emp_and_crises(p, drop_shared=drop_shared)
    p = add_indicators(p, war)
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
    known = p[(p.month + H <= as_of) & p.evaluable]
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
    test["pressure"] = calibrate(known.emp_z_max3.fillna(0), known.y, test.emp_z_max3.fillna(0)).to_numpy()
    s_tr, s_te = composite(known, m.sig_model, m.ref), composite(test, m.sig_model, m.ref)
    test["signals_score"] = s_te
    test["signals"] = calibrate(s_tr, known.y, s_te).to_numpy()
    te = test[feats].clip(m.lo, m.hi, axis=1).fillna(m.med)
    test["logit"] = m.logit.predict_proba(te)[:, 1]
    return test


def forecast_asof(p: pd.DataFrame, as_of: pd.Period) -> pd.DataFrame:
    """Прогноз в точке as_of для всех стран со своей валютой: всё — по данным ≤ as_of."""
    m = fit(p, as_of)
    test = p[(p.month == as_of) & p.own_currency & p.fx.notna()]
    return predict(m, test)


def run_models(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds, thresholds = [], []
    for T in range(FIRST_EVAL_YEAR, LAST_EVAL_YEAR + 1):
        # учимся на том, что известно к концу декабря T−1; предсказываем все месяцы T
        m = fit(p, pd.Period(f"{T - 1}-12", freq="M"))
        test = p[(p.month.dt.year == T) & p.evaluable]
        if m is None or test.empty:
            continue
        test = predict(m, test)
        for ind, par in m.sig_model.items():
            thresholds.append({"year": T, "indicator": ind, **par})
        preds.append(test)
        print(f"  {T}: обучение {len(m.known)} мес ({int(m.known.y.sum())} кризисных окон), "
              f"тест {len(test)} мес ({int(test.y.sum())}), сигналов в модели {len(m.sig_model)}")
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
        # AUC внутри года: умеет ли модель отличать страны друг от друга в один и тот же год
        # (сквозной AUC ещё и штрафует за то, что общий уровень риска по годам не угадан)
        yearly = [roc_auc_score(g.y, g[m]) for _, g in d[ok].groupby(d.month[ok].dt.year)
                  if g.y.nunique() == 2 and g[m].nunique() > 1]
        rows.append({"model": m, "n": int(ok.sum()), "crisis_windows": int(d.y[ok].sum()),
                     "brier": brier_score_loss(d.y[ok], d[m][ok]),
                     "auc": roc_auc_score(d.y[ok], d[m][ok]),
                     "auc_within_year": float(np.mean(yearly)) if yearly else np.nan})
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
    cov.to_csv(REPORTS / f"econ_kr_coverage{suffix}.csv", index=False)
    print("\nПокрытие постсоветских стран:")
    print(cov[["country", "fx_from", "reserves_from", "cpi_from", "cpi_filled_months", "rate_from", "n_crises", "n_deval15"]].to_string(index=False))

    cr = p[(p.crisis == 1) & p.post_soviet][["iso3", "month", "de", "dr", "emp_z"]].copy()
    cr["country"] = cr.iso3.map(ISO3_TO_SHORT)
    cr["month"] = cr.month.astype(str)
    cr.to_csv(REPORTS / f"econ_kr_crises_post_soviet{suffix}.csv", index=False)
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
