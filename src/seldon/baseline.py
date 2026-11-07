"""Базовые модели «следующий год похож на прошлые» и их проверка вслепую.

Запуск:
    uv run python -m seldon.baseline            # целевая conflict25 (по территории)
    uv run python -m seldon.baseline war25      # целевая war25 (территория ИЛИ сторона конфликта)

Для целевой, отличной от conflict25, выходные файлы получают суффикс `_war25`.

Модели (все — без признаков, только история целевой переменной):
    last       вероятность = был ли конфликт (≥25 погибших) в стране в прошлом году (0 или 1)
    decay      экспоненциально затухающая память: m_t = (1 − α)·m_{t−1} + α·conflict_t,
               где α задаётся полураспадом h (через h лет без конфликта память падает вдвое).
               h выбираем из {1, 3, 5} по обучающим данным (меньший Brier), отдельно для
               каждого года проверки — будущее не подглядываем.
    base_rate  одна и та же вероятность для всех стран: доля стран с конфликтом в последнем
               обучающем году.

Проверка вслепую (out-of-sample): для каждого T от 1995 до 2023 используем данные
по год T включительно и предсказываем год T+1. Метрики за каждый год T+1:
    brier      Brier score — средний квадрат ошибки вероятности (0 — идеально, 0.25 — монетка
               на событии с частотой 50%). Чем меньше, тем лучше.
    auc        AUC (area under ROC curve) — вероятность того, что случайной стране с конфликтом
               модель дала балл выше, чем случайной стране без конфликта. 0.5 — наугад, 1 — идеал.
               Не определён, если в году только один класс или модель выдала всем одно и то же.
    ..._onset  то же, но только для стран, где в году T конфликта не было, а целевая — onset25
               (началось ли). Это главный честный тест: «помнить вчерашнее» тут не помогает.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from seldon import DATA_PROCESSED, PROJECT_ROOT

REPORTS = PROJECT_ROOT / "reports"
TRAIN_YEARS = range(1995, 2024)  # T: учимся по T, предсказываем T+1 (1996…2024)
HALF_LIVES = (1, 3, 5)
MODELS = ("last", "decay", "base_rate")
TARGET, ONSET = "conflict25", "onset25"  # переопределяются в main() аргументом командной строки
ONSET_OF = {"conflict25": "onset25", "war25": "onset_war25"}


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / "panel_country_year.parquet")


def decay_memory(panel: pd.DataFrame, half_life: float) -> pd.Series:
    """Память о конфликтах на конец каждого года (включая этот год), по странам."""
    alpha = 1 - 0.5 ** (1 / half_life)
    return panel.groupby("gw")[TARGET].transform(
        lambda s: s.ewm(alpha=alpha, adjust=False).mean()
    )


def choose_half_life(panel: pd.DataFrame, upto: int) -> int:
    """Полураспад с наименьшим Brier на обучающих годах (≤ upto): память до t−1 → факт t."""
    train = panel[panel.year <= upto]
    best_h, best_brier = None, np.inf
    for h in HALF_LIVES:
        mem = decay_memory(train, h)
        pred = mem.groupby(train["gw"]).shift(1)
        mask = pred.notna()
        brier = ((pred[mask] - train.loc[mask, TARGET]) ** 2).mean()
        if brier < best_brier:
            best_h, best_brier = h, brier
    return best_h


def predict_year(panel: pd.DataFrame, T: int) -> pd.DataFrame:
    """Прогнозы всех моделей на год T+1 по данным до T включительно."""
    train = panel[panel.year <= T]
    last = train[train.year == T].set_index("gw")[TARGET]
    h = choose_half_life(panel, T)
    mem = decay_memory(train, h)
    decay = pd.Series(mem[train.year == T].to_numpy(), index=train.loc[train.year == T, "gw"].to_numpy())
    base = float(last.mean())

    test = panel[panel.year == T + 1][["gw", "country", "year", TARGET, ONSET]].copy()
    test = test[test.gw.isin(last.index)]  # страны, которые существовали в году T
    test["at_risk"] = (test.gw.map(last) == 0).astype("int8")  # не было конфликта в T → кандидат на онсет
    test["last"] = test.gw.map(last).astype(float)
    test["decay"] = test.gw.map(decay).astype(float)
    test["base_rate"] = base
    test["decay_half_life"] = h
    return test


def predict_all(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([predict_year(panel, T) for T in TRAIN_YEARS], ignore_index=True)


def _auc(y: pd.Series, p: pd.Series) -> float:
    if y.nunique() < 2 or p.nunique() < 2:
        return np.nan
    return roc_auc_score(y, p)


def evaluate(pred: pd.DataFrame) -> pd.DataFrame:
    """Метрики по годам проверки: строки — год, колонки — модель × метрика."""
    rows = []
    for year, g in pred.groupby("year"):
        row = {"year": year, "n_countries": len(g), "n_conflict": int(g[TARGET].sum()),
               "n_at_risk": int(g.at_risk.sum()), "n_onset": int(g.loc[g.at_risk == 1, ONSET].sum()),
               "decay_half_life": int(g.decay_half_life.iloc[0])}
        r = g[g.at_risk == 1]
        for m in MODELS:
            row[f"{m}_brier"] = ((g[m] - g[TARGET]) ** 2).mean()
            row[f"{m}_auc"] = _auc(g[TARGET], g[m])
            row[f"{m}_brier_onset"] = ((r[m] - r[ONSET]) ** 2).mean()
            row[f"{m}_auc_onset"] = _auc(r[ONSET], r[m])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Среднее по годам: строки — модель, колонки — метрики."""
    out = {}
    for m in MODELS:
        out[m] = {k: metrics[f"{m}_{k}"].mean() for k in ("brier", "auc", "brier_onset", "auc_onset")}
    return pd.DataFrame(out).T


def main(argv: list[str]) -> int:
    global TARGET, ONSET
    if argv:
        if argv[0] not in ONSET_OF:
            print(f"неизвестная целевая {argv[0]!r}; доступны: {', '.join(ONSET_OF)}")
            return 2
        TARGET, ONSET = argv[0], ONSET_OF[argv[0]]
    suffix = "" if TARGET == "conflict25" else f"_{TARGET}"

    panel = load_panel()
    pred = predict_all(panel)
    metrics = evaluate(pred)
    summary = summarize(metrics)

    REPORTS.mkdir(exist_ok=True)
    pred.to_parquet(DATA_PROCESSED / f"baseline_predictions{suffix}.parquet", index=False)
    metrics.to_csv(REPORTS / f"baseline_metrics_by_year{suffix}.csv", index=False, float_format="%.4f")
    summary.to_csv(REPORTS / f"baseline_metrics_summary{suffix}.csv", float_format="%.4f")

    print(f"целевая: {TARGET}; прогнозов: {len(pred)} строк, годы {pred.year.min()}–{pred.year.max()}")
    print("полураспад decay по годам:", metrics.decay_half_life.value_counts().to_dict())
    print("\nСреднее по годам проверки:")
    print(summary.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
