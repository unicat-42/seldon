"""Месячный валютный блок, ЦЕЛЕВЫЕ: индекс давления на валютный рынок (EMP), кризисы, метка.

    EMP = %Δe − (σ_e / σ_R) · %ΔR  (EMP — exchange market pressure; e — курс, R — резервы).
    Все параметры (σ, медиана EMP) в месяце m считаются только по месяцам ≤ m —
    расширяющимся окном. Кризис (`crisis`) = месяц, когда EMP выше медианы страны на 3σ
    (σ устойчивая, через медианное абсолютное отклонение) и EMP ≥ 10 %; `deval15` — курс за
    месяц вырос ≥ 15 %. Повторные всплески внутри 12 месяцев — не новый кризис.
    Метка `y`: кризис в ближайшие 12 месяцев; `evaluable` — строка, где такой кризис может
    быть засчитан (правило тишины одно и для метки, и для оценки — инвариант 3).

Подробности и история правок — в docstring `econ_kr.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon.core.entities import shared_currency_table
from seldon.core.panel import pct_change

H = 12  # горизонт: кризис в ближайшие 12 месяцев
POST_CRISIS_DROP = 12  # месяцев после кризиса, которые не оцениваем
EMP_SIGMA = 3.0
EMP_FLOOR = 0.10  # и не меньше 10% давления: у жёстких привязок σ≈0, любой сдвиг — «3σ»
HYPERINFL_6M = 1.5  # 6-месячная инфляция > 150% — гиперинфляция (KLR)
DEVAL15 = 0.15
MIN_HISTORY = 36  # месяцев истории страны для расчёта σ и процентилей


def add_emp_and_crises(p: pd.DataFrame, drop_shared: bool = True) -> pd.DataFrame:
    p = p.copy()
    gb = p.groupby("iso3")
    p["de"] = gb.fx.transform(lambda s: pct_change(s, 1))
    p["dr"] = gb.reserves.transform(lambda s: pct_change(s, 1))
    p["infl6"] = gb.cpi.transform(lambda s: pct_change(s, 6))
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
        med_e, sd_e, n_calm = _expanding_robust(de, calm)
        _, sd_r, _ = _expanding_robust(dr, calm)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = sd_e / sd_r
        w[~np.isfinite(w)] = np.nan
        emp = np.where(np.isnan(dr), de, de - w * dr)  # если резервов нет — EMP только по курсу
        emp[np.isnan(de)] = np.nan
        emp_mean, emp_sd, _ = _expanding_robust(emp, calm)
        parts.append(pd.DataFrame({"med_e": med_e, "sd_e": sd_e, "sd_r": sd_r, "n_calm": n_calm, "emp": emp,
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


def add_target(p: pd.DataFrame, target: str = "crisis", h: int = H) -> pd.DataFrame:
    """Метка `y`: кризис в ближайшие `h` месяцев (по умолчанию 12; чекпоинт 4.1 — ещё 3 и 6)."""
    p = p.copy()
    gb = p.groupby("iso3")[target]
    fwd = sum(gb.shift(-k) for k in range(1, h + 1))
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
