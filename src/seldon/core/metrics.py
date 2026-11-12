"""Единый расчёт метрик качества прогноза.

    auc          AUC (area under ROC curve) — доля пар «событие / не событие», которые модель
                 упорядочила правильно: 0,5 — монетка, 1 — идеально. Сквозной (pooled) — по
                 всем страна-периодам сразу.
    auc_within   среднее AUC по группам (обычно по годам): умеет ли модель отличать страны
                 друг от друга внутри одного года. Сквозной AUC ещё и штрафует за то, что
                 общий уровень риска по годам не угадан.
    brier        Brier score — средний квадрат ошибки вероятности (0 — идеально).
    year_block_bootstrap, interval
                 95 % интервал блочным бутстрэпом по годам: годы тасуются с возвращением,
                 страна-годы одного года остаются вместе (они зависимы — общий мировой фон).
                 Сравнение моделей — парное: разница считается на одних и тех же выборках лет.

Инвариант 10 (CLAUDE.md): метрика без интервала — не результат.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def auc(y, p, constant_ok: bool = False) -> float:
    """AUC; NaN, если в `y` не два класса или модель выдала всем одно и то же.
    `constant_ok=True` — для одинакового у всех прогноза вернуть 0,5 (так считает sklearn):
    нужно в погодовых таблицах, где `base_rate` внутри года — константа."""
    yv, pv = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(np.unique(yv[~np.isnan(yv)])) != 2:
        return np.nan
    if len(np.unique(pv[~np.isnan(pv)])) < 2 and not constant_ok:
        return np.nan
    return roc_auc_score(y, p)


def brier(y, p) -> float:
    """Brier score по строкам, где есть и факт, и прогноз; NaN, если таких нет."""
    yv, pv = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    ok = ~(np.isnan(yv) | np.isnan(pv))
    if not ok.any():
        return np.nan
    return brier_score_loss(y, p) if ok.all() else brier_score_loss(yv[ok], pv[ok])


def auc_by_group(y, p, groups) -> list[float]:
    """AUC в каждой группе (по возрастанию ключа); NaN там, где он не определён."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "g": np.asarray(groups)})
    return [auc(g.y, g.p) for _, g in d.groupby("g")]


def mean_defined(values, compact: bool = False) -> float:
    """Среднее по определённым (не NaN) значениям; NaN, если таких нет.

    `compact` выбирает порядок суммирования и на смысл не влияет: False — как `np.nanmean`
    (пропуски заменяются нулями на своих местах), True — сначала выбросить пропуски, потом
    сложить. Результаты расходятся в 16-м знаке. Оба режима оставлены на чекпоинте 2.1,
    чтобы таблицы совпали с эталоном до последнего знака (месячный валютный блок исторически
    считал вторым способом, остальные — первым); свести к одному — при первом пересчёте,
    который и так меняет цифры (чекпоинт 2.2)."""
    a = np.asarray(values, dtype=float)
    valid = a[~np.isnan(a)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid)) if compact else float(np.nanmean(a))


def auc_within(y, p, groups, compact: bool = False) -> float:
    """Среднее AUC по группам (по годам), где он определён."""
    return mean_defined(auc_by_group(y, p, groups), compact=compact)


def year_block_bootstrap(pred: pd.DataFrame, stat: Callable[[pd.DataFrame], object], n_boot: int = 500,
                         seed: int = 0, year: str = "year") -> tuple[object, list]:
    """Блочный бутстрэп по годам. `stat(d)` считает метрики одной выборки; у `d` есть колонка
    `boot_year` — номер блока (один и тот же год может войти в реплику дважды, это разные
    блоки). Возвращает (метрики исходной выборки, список метрик реплик)."""
    years = np.sort(pred[year].unique())
    blocks = {y: g for y, g in pred.groupby(year)}
    point = stat(pred.assign(boot_year=pred[year]))
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        sample = rng.choice(years, len(years), replace=True)
        d = pd.concat([blocks[y].assign(boot_year=i) for i, y in enumerate(sample)], ignore_index=True)
        reps.append(stat(d))
    return point, reps


def interval(values, level: float = 95.0) -> tuple[float, float]:
    """Процентильный интервал по репликам бутстрэпа (NaN-реплики пропускаются)."""
    b = np.asarray(values, dtype=float)
    tail = (100.0 - level) / 2
    return np.nanpercentile(b, tail), np.nanpercentile(b, 100.0 - tail)
