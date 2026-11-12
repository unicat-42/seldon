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

    paired_comparison
                 несколько прогнозов против эталона на одних и тех же строках: метрики, `calib`
                 (средняя предсказанная вероятность / фактическая частота: 1 — в среднем не
                 врёт, > 1 — завышает риск) и разницы с эталоном, всё с интервалами.

Инвариант 10 (CLAUDE.md): метрика без интервала — не результат.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
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


def mean_defined(values) -> float:
    """Среднее по определённым (не NaN) значениям; NaN, если таких нет."""
    a = np.asarray(values, dtype=float)
    valid = a[~np.isnan(a)]
    return float(np.mean(valid)) if len(valid) else np.nan


def auc_within(y, p, groups) -> float:
    """Среднее AUC по группам (по годам), где он определён."""
    return mean_defined(auc_by_group(y, p, groups))


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


COMPARE_METRICS = ("auc", "auc_within_year", "brier", "calib")


def _rank_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC через ранги (то же число, что `auc`, но без накладных расходов sklearn)."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0 or p.min() == p.max():
        return np.nan
    return float((rankdata(p)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paired_comparison(pred: pd.DataFrame, cols: dict[str, str], ref: str, y: str = "y", year: str = "year",
                      n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Парное сравнение прогнозов `cols` (имя → колонка) с эталоном `ref` на строках `pred`.
    На каждую модель — строка: метрики `COMPARE_METRICS` с 95 % интервалом и разницы `d_*`
    с эталоном. Интервалы — блочный бутстрэп по годам (та же схема и тот же генератор, что в
    `year_block_bootstrap`); разница считается внутри каждой реплики — обе модели на одной и
    той же выборке лет. Годовые блоки посчитаны заранее, поэтому на месячных панелях в
    десятки тысяч строк это секунды, а не часы."""
    years = np.sort(pred[year].unique())
    blocks = {t: g for t, g in pred.groupby(year)}
    yb = {t: blocks[t][y].to_numpy(dtype=float) == 1 for t in years}
    n = {t: len(yb[t]) for t in years}

    def stats(sample) -> dict:
        ys = np.concatenate([yb[t] for t in sample])
        size, rate = sum(n[t] for t in sample), ys.mean()
        out = {}
        for name, c in cols.items():
            within = np.array([per_year[name][t][0] for t in sample], dtype=float)
            ok = ~np.isnan(within)
            out[name] = {
                "auc": _rank_auc(ys, np.concatenate([pb[name][t] for t in sample])),
                "auc_within_year": float(within[ok].mean()) if ok.any() else np.nan,
                "brier": sum(per_year[name][t][1] for t in sample) / size,
                "calib": sum(per_year[name][t][2] for t in sample) / size / rate if rate > 0 else np.nan,
            }
        return out

    pb = {name: {t: blocks[t][c].to_numpy(dtype=float) for t in years} for name, c in cols.items()}
    # на год: (AUC внутри года, сумма квадратов ошибок, сумма вероятностей)
    per_year = {name: {t: (_rank_auc(yb[t], pb[name][t]), float(((pb[name][t] - yb[t]) ** 2).sum()), float(pb[name][t].sum()))
                       for t in years} for name in cols}
    point = stats(years)
    rng = np.random.default_rng(seed)
    reps = [stats(rng.choice(years, len(years), replace=True)) for _ in range(n_boot)]
    rows = []
    for name in cols:
        row = {"model": name, "n": len(pred), "positives": int(pred[y].sum())}
        for k in COMPARE_METRICS:
            b = np.array([r[name][k] for r in reps], dtype=float)
            row[k], (row[f"{k}_lo"], row[f"{k}_hi"]) = point[name][k], interval(b)
            if name != ref:
                diff = b - np.array([r[ref][k] for r in reps], dtype=float)
                row[f"d_{k}"] = point[name][k] - point[ref][k]
                row[f"d_{k}_lo"], row[f"d_{k}_hi"] = interval(diff)
        rows.append(row)
    return pd.DataFrame(rows)
