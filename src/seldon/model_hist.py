"""Первая модель с признаками: логистическая регрессия на истории конфликтов.

Запуск:
    uv run python -m seldon.model_hist

Что делает:
  - берёт признаки из features.py (все считаются по данным до года T включительно);
  - целевая — war25 в году T+1;
  - проверка вслепую как в baseline.py: для каждого T = 1995…2023 обучаемся на парах
    (признаки года t, целевая года t+1) с t+1 ≤ T, предсказываем год T+1;
  - сравнивает с `decay` из baseline (файл baseline_predictions_war25.parquet) на тех же строках.

Логистическая регрессия — простейшая модель «признаки → вероятность». Признаки
стандартизируются (вычитаем среднее, делим на разброс — по обучающим данным), небольшая
L2-регуляризация (штраф за большие коэффициенты, чтобы модель не переобучалась на шуме).

Выход:
    data/processed/hist_predictions.parquet
    reports/hist_metrics_by_year.csv, reports/hist_metrics_summary.csv, reports/hist_coefficients.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, PROJECT_ROOT
from seldon.features import FEATURES

REPORTS = PROJECT_ROOT / "reports"
TARGET, ONSET = "war25", "onset_war25"
TRAIN_YEARS = range(1995, 2024)
MODELS = ("hist", "decay")


def load() -> pd.DataFrame:
    f = pd.read_parquet(DATA_PROCESSED / "features_country_year.parquet")
    # целевая следующего года на строку текущего
    nxt = f[["gw", "year", TARGET, ONSET]].assign(year=lambda d: d.year - 1)
    nxt = nxt.rename(columns={TARGET: "y_next", ONSET: "onset_next"})
    return f.merge(nxt, on=["gw", "year"], how="left")


def make_model() -> make_pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def predict_year(f: pd.DataFrame, T: int) -> tuple[pd.DataFrame, pd.Series]:
    train = f[(f.year <= T - 1) & f.y_next.notna()]
    test = f[(f.year == T) & f.y_next.notna()].copy()
    m = make_model().fit(train[FEATURES], train.y_next.astype(int))
    test["hist"] = m.predict_proba(test[FEATURES])[:, 1]
    coef = pd.Series(m[-1].coef_[0], index=FEATURES, name=T + 1)
    out = test[["gw", "country", "y_next", "onset_next", "war_now", "hist"]].rename(
        columns={"y_next": TARGET, "onset_next": ONSET}
    )
    out["year"] = T + 1
    out["at_risk"] = (out.war_now == 0).astype("int8")
    return out.drop(columns="war_now"), coef


def _auc(y, p):
    return roc_auc_score(y, p) if (y.nunique() == 2 and p.nunique() > 1) else np.nan


def evaluate(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in pred.groupby("year"):
        r = g[g.at_risk == 1]
        row = {"year": year, "n": len(g), "n_at_risk": len(r), "n_onset": int(r[ONSET].sum())}
        for m in MODELS:
            row[f"{m}_brier"] = ((g[m] - g[TARGET]) ** 2).mean()
            row[f"{m}_auc"] = _auc(g[TARGET], g[m])
            row[f"{m}_brier_onset"] = ((r[m] - r[ONSET]) ** 2).mean()
            row[f"{m}_auc_onset"] = _auc(r[ONSET], r[m])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    f = load()
    preds, coefs = [], []
    for T in TRAIN_YEARS:
        p, c = predict_year(f, T)
        preds.append(p)
        coefs.append(c)
    pred = pd.concat(preds, ignore_index=True)
    coefs = pd.concat(coefs, axis=1)

    base = pd.read_parquet(DATA_PROCESSED / "baseline_predictions_war25.parquet")[["gw", "year", "decay"]]
    pred = pred.merge(base, on=["gw", "year"], how="inner")

    metrics = evaluate(pred)
    summary = pd.DataFrame(
        {m: {k: metrics[f"{m}_{k}"].mean() for k in ("brier", "auc", "brier_onset", "auc_onset")} for m in MODELS}
    ).T

    pred.to_parquet(DATA_PROCESSED / "hist_predictions.parquet", index=False)
    metrics.to_csv(REPORTS / "hist_metrics_by_year.csv", index=False, float_format="%.4f")
    summary.to_csv(REPORTS / "hist_metrics_summary.csv", float_format="%.4f")
    coefs.round(3).to_csv(REPORTS / "hist_coefficients.csv")

    print(f"прогнозов: {len(pred)} строк, годы {pred.year.min()}–{pred.year.max()}")
    print("\nСреднее по годам проверки (hist — логрегрессия на признаках, decay — базовая):")
    print(summary.round(4).to_string())
    print("\nКоэффициенты (стандартизованные; последняя модель, обучена по 2023):")
    print(coefs.iloc[:, -1].sort_values(key=abs, ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
