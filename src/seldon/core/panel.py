"""Общие операции с панелями «страна × период».

Главное правило: лаги и приросты считаются по КАЛЕНДАРЮ, а не по соседним строкам.
Для этого панель сначала достраивается до полной сетки периодов внутри каждой страны
(`full_grid`), и только потом берутся `shift` / `rolling`.
"""

from __future__ import annotations

import pandas as pd


def full_grid(spans: pd.DataFrame, names: list[str], freq: str | None = None) -> pd.MultiIndex:
    """Полная сетка периодов для каждой страны: от её первого до последнего периода.

    `spans` — таблица с индексом-страной и колонками `min`, `max` (первый и последний период).
    `freq="M"` — периоды-месяцы (`pd.Period`); без `freq` — целые годы."""
    def steps(lo, hi):
        return pd.period_range(lo, hi, freq=freq) if freq else range(lo, hi + 1)

    return pd.MultiIndex.from_tuples(
        [(c, t) for c, r in spans.iterrows() for t in steps(r["min"], r["max"])], names=names)


def pct_change(s: pd.Series, k: int) -> pd.Series:
    """Прирост за k периодов назад по строкам ряда одной страны (ряд должен быть на полной сетке)."""
    return s / s.shift(k) - 1


def lag_columns(df: pd.DataFrame, entity: str, period: str, cols: list[str], lag: int) -> pd.DataFrame:
    """Сдвиг колонок `cols` на `lag` периодов назад ПО КАЛЕНДАРЮ: на строке периода t
    оказывается значение периода t − lag той же страны (нет такой строки — пропуск).
    Так вносится лаг публикации: «в момент прогноза известны только данные за t − lag».
    Пары (страна, период) должны быть уникальны; порядок строк и колонок сохраняется."""
    if lag == 0:
        return df
    src = df[[entity, period, *cols]].copy()
    src[period] = src[period] + lag
    out = df.drop(columns=cols).merge(src, on=[entity, period], how="left")
    out.index = df.index
    return out[list(df.columns)]
