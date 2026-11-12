"""Единый цикл проверки вслепую (out-of-sample) расширяющимся окном.

Все контуры проверяются одинаково: идём по точкам отсчёта T, в каждой зовём
`forecast_asof` своего контура (он учится только на том, что известно к T, — см.
`core/temporal.py`) и склеиваем прогнозы в одну таблицу. Отличается только шаг.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd


def backtest(origins: Iterable, step: Callable[[object], pd.DataFrame | None]) -> pd.DataFrame:
    """Прогнозы по всем точкам отсчёта. `step(T)` возвращает таблицу прогнозов из точки T
    или None / пустую таблицу, если в этой точке прогноза нет (мало истории)."""
    frames = []
    for T in origins:
        out = step(T)
        if out is None or out.empty:
            continue
        frames.append(out)
    return pd.concat(frames, ignore_index=True)
