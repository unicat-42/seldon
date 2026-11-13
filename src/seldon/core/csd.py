"""Примитивы критического замедления (Critical Slowing Down, CSD — Scheffer et al.):
скользящая дисперсия, автокорреляция первого порядка AR(1) и их сдвиги к непересекающемуся
базовому окну в прошлом (чекпоинт 4.3b).

Все функции строго причинны: значение в точке `t` зависит только от значений ≤ `t`.
На нулевых или константных окнах (при наличии не менее `min_periods` наблюдений) возвращают
`0.0`, а не `NaN` или `inf`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_var(s: pd.Series, window: int, min_periods: int = 6) -> pd.Series:
    """Причинная скользящая несмещённая дисперсия на окне `[t - window + 1 .. t]`."""
    return s.rolling(window, min_periods=min_periods).var(ddof=1)


def rolling_ar1(s: pd.Series, window: int, min_periods: int = 8, tol: float = 1e-12) -> pd.Series:
    """Причинная автокорреляция лага 1 на окне `[t - window + 1 .. t]`.

    Считается по парам `(x_i, x_{i-1})` внутри окна, где оба конца конечны. Если конечных
    пар меньше `max(2, min_periods - 1)` — `NaN`; если дисперсия одного из концов ≤ `tol`
    (плоский / нулевой ряд) — `0.0`.
    """
    prev = s.shift(1)
    valid = s.notna() & prev.notna() & np.isfinite(s) & np.isfinite(prev)
    x = s.where(valid, 0.0).astype(float)
    y = prev.where(valid, 0.0).astype(float)
    w = max(2, window - 1)
    mp = max(2, min_periods - 1)

    n = valid.astype(float).rolling(w, min_periods=1).sum()
    sx = x.rolling(w, min_periods=1).sum()
    sy = y.rolling(w, min_periods=1).sum()
    sxx = (x * x).rolling(w, min_periods=1).sum()
    syy = (y * y).rolling(w, min_periods=1).sum()
    sxy = (x * y).rolling(w, min_periods=1).sum()

    n_arr = n.to_numpy(dtype=float)
    safe_n = np.where(n_arr > 0, n_arr, 1.0)
    vx = np.maximum(sxx.to_numpy(dtype=float) - sx.to_numpy(dtype=float) ** 2 / safe_n, 0.0)
    vy = np.maximum(syy.to_numpy(dtype=float) - sy.to_numpy(dtype=float) ** 2 / safe_n, 0.0)
    cov = sxy.to_numpy(dtype=float) - (sx.to_numpy(dtype=float) * sy.to_numpy(dtype=float)) / safe_n
    den = np.sqrt(vx * vy)

    out = np.full(len(s), np.nan, dtype=float)
    enough = n_arr >= mp
    flat = enough & (den <= tol)
    ok = enough & (den > tol)
    out[flat] = 0.0
    out[ok] = np.clip(cov[ok] / den[ok], -1.0, 1.0)
    return pd.Series(out, index=s.index, dtype=float)


def var_surge(
    s: pd.Series,
    short: int = 12,
    long: int = 48,
    min_short: int = 6,
    min_long: int = 24,
    eps: float = 1e-8,
) -> pd.Series:
    """Лог-отношение недавней дисперсии `[t - short + 1 .. t]` к предшествующему базовому окну
    `[t - short - long + 1 .. t - short]` (строго не пересекается с коротким окном)."""
    vs = rolling_var(s, short, min_periods=min_short)
    vl = rolling_var(s, long, min_periods=min_long).shift(short)
    both = vs.notna() & vl.notna()
    out = np.full(len(s), np.nan, dtype=float)
    vs_a, vl_a = vs.to_numpy(dtype=float), vl.to_numpy(dtype=float)
    m = both.to_numpy()
    out[m] = np.log((np.maximum(vs_a[m], 0.0) + eps) / (np.maximum(vl_a[m], 0.0) + eps))
    return pd.Series(out, index=s.index, dtype=float)


def ar1_delta(
    s: pd.Series,
    short: int = 12,
    long: int = 48,
    min_short: int = 8,
    min_long: int = 24,
) -> pd.Series:
    """Сдвиг автокорреляции AR(1): недавнее окно `[t - short + 1 .. t]` минус предшествующее
    базовое окно `[t - short - long + 1 .. t - short]`."""
    a_short = rolling_ar1(s, short, min_periods=min_short)
    a_long = rolling_ar1(s, long, min_periods=min_long).shift(short)
    return a_short - a_long


def by_entity(df: pd.DataFrame, group: str, col: str, fn, *args, **kwargs) -> pd.Series:
    """Применить причинную функцию `fn` строго внутри каждой сущности без перетока через границы."""
    return df.groupby(group, group_keys=False)[col].apply(lambda s: fn(s, *args, **kwargs))
