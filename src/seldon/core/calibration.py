"""Калибровка: перевод балла модели в вероятность — только по прошлым данным.

Два способа, оба монотонные (порядок стран не меняют):
  - `binned_rate` — как в методе сигналов Kaminsky–Lizondo–Reinhart: доля событий в
    обучении среди наблюдений с таким же баллом (по децилям);
  - `expanding_isotonic` — изотоническая регрессия расширяющимся окном: балл периода t
    калибруется по периодам строго раньше t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def binned_rate(score_train: pd.Series, y_train: pd.Series, score_test: pd.Series) -> pd.Series:
    """Вероятность = доля событий в обучении среди наблюдений с таким же баллом
    (по децилям ненулевых значений, ноль — отдельно)."""
    nz = score_train[score_train > 0]
    edges = np.unique(np.quantile(nz, np.linspace(0, 1, 11))) if len(nz) > 50 else np.array([0, np.inf])
    edges = np.concatenate([[-np.inf, 0], edges[1:-1], [np.inf]])
    b_tr = pd.cut(score_train, edges, labels=False, include_lowest=True)
    b_te = pd.cut(score_test, edges, labels=False, include_lowest=True)
    rate = y_train.groupby(b_tr).mean()
    # монотонно не убывающая по бину, чтобы редкие верхние бины не проваливались
    rate = rate.cummax()
    return b_te.map(rate).fillna(y_train.mean())


def expanding_isotonic(df: pd.DataFrame, score: str, target: str, period: str = "year",
                       group: str | None = None, min_events: int = 5) -> pd.Series:
    """Балл `score` → вероятность `target`: для каждого периода t изотоническая регрессия
    учится на периодах < t (отдельно в каждой группе `group`) и применяется к t.
    Пока в прошлом нет обоих классов или событий меньше `min_events` — остаётся сырой балл."""
    out = df[score].copy()
    groups = [None] if group is None else sorted(df[group].unique())
    for t in sorted(df[period].unique()):
        for g in groups:
            in_group = True if g is None else (df[group] == g)
            past = df[(df[period] < t) & in_group]
            now = (df[period] == t) & in_group
            if now.sum() == 0 or past[target].nunique() < 2 or past[target].sum() < min_events:
                continue
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(past[score], past[target])
            out.loc[now] = iso.predict(df.loc[now, score])
    return out
