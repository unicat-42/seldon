"""Первая модель с признаками: логистическая регрессия на истории конфликтов.

Запуск:
    uv run python -m seldon.model_hist              # признаки истории (features.py), логрегрессия
    uv run python -m seldon.model_hist pitf         # история + режим/смертность/население (features_pitf.py)
    uv run python -m seldon.model_hist pitf gbm     # то же, но градиентный бустинг вместо логрегрессии
    uv run python -m seldon.model_hist pitf logit war1000   # третий аргумент — целевая (по умолчанию war25)
    uv run python -m seldon.model_hist econ [logit|gbm] [war25|war1000]   # + экономика (features_econ.py)
    uv run python -m seldon.model_hist histecon              # история + экономика, без PITF
    uv run python -m seldon.model_hist polit [logit|gbm] [war25|war1000]  # + политика (features_polit.py)

Признаки истории всегда считаются по war25 (features.py); целевая может быть другой —
тогда имя прогноза получает суффикс `_<целевая>`, а для сравнения берётся `decay` из
`baseline_predictions_<целевая>.parquet` (сначала запусти `baseline <целевая>`).

Градиентный бустинг (`HistGradientBoostingClassifier` из scikit-learn) — ансамбль
неглубоких деревьев; в отличие от логрегрессии умеет находить взаимодействия признаков
(«фракционность опасна, только если рядом война») и пороги. Параметры скромные (глубина 3,
200 деревьев, learning_rate 0.05), без подбора — «сначала простое». Имя модели в выходных
файлах: `<набор>_gbm`.

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, PROJECT_ROOT
from seldon.features import FEATURES as HIST_FEATURES
from seldon.features_econ import ECON_FEATURES
from seldon.features_pitf import PITF_FEATURES
from seldon.features_polit import POLIT_FEATURES

REPORTS = PROJECT_ROOT / "reports"
TARGET, ONSET = "war25", "onset_war25"
ONSET_OF = {"war25": "onset_war25", "war1000": "onset_war1000"}
TRAIN_YEARS = range(1995, 2024)
FEATURE_SETS = {
    "hist": (HIST_FEATURES, "features_country_year.parquet"),
    "pitf": ([*HIST_FEATURES, *PITF_FEATURES], "features_pitf_country_year.parquet"),
    # экономика (features_econ.py): GMD + давление на валюту из econ_kr
    "econ": ([*HIST_FEATURES, *PITF_FEATURES, *ECON_FEATURES], "features_econ_country_year.parquet"),
    "histecon": ([*HIST_FEATURES, *ECON_FEATURES], "features_econ_country_year.parquet"),
    # политика (features_polit.py): лидеры, выборы, перевороты (REIGN), протесты (Mass Mobilization)
    "polit": ([*HIST_FEATURES, *PITF_FEATURES, *POLIT_FEATURES], "features_polit_country_year.parquet"),
}
NAME = "hist"  # имя прогноза в файлах; переопределяется в main()
FEATURE_SET = "hist"
KIND = "logit"  # или "gbm"
FEATURES = HIST_FEATURES
MODELS = ("hist", "decay")


def load() -> pd.DataFrame:
    f = pd.read_parquet(DATA_PROCESSED / FEATURE_SETS[FEATURE_SET][1])
    # целевая следующего года на строку текущего
    nxt = f[["gw", "year", TARGET, ONSET]].assign(year=lambda d: d.year - 1)
    nxt = nxt.rename(columns={TARGET: "y_next", ONSET: "onset_next"})
    return f.merge(nxt, on=["gw", "year"], how="left")


def make_model():
    if KIND == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def predict_year(f: pd.DataFrame, T: int) -> tuple[pd.DataFrame, pd.Series]:
    train = f[(f.year <= T - 1) & f.y_next.notna()]
    test = f[(f.year == T) & f.y_next.notna()].copy()
    m = make_model().fit(train[FEATURES], train.y_next.astype(int))
    test[NAME] = m.predict_proba(test[FEATURES])[:, 1]
    if KIND == "gbm":  # у бустинга нет коэффициентов — пишем нули, важность смотрим отдельно
        coef = pd.Series(0.0, index=FEATURES, name=T + 1)
    else:
        coef = pd.Series(m[-1].coef_[0], index=FEATURES, name=T + 1)
    out = test[["gw", "country", "y_next", "onset_next", TARGET, NAME]].rename(
        columns={TARGET: "target_now"}
    ).rename(columns={"y_next": TARGET, "onset_next": ONSET})
    out["year"] = T + 1
    out["at_risk"] = (out.target_now == 0).astype("int8")  # не воевала (по целевой) в году T
    return out.drop(columns="target_now"), coef


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


def main(argv: list[str]) -> int:
    global NAME, FEATURE_SET, KIND, FEATURES, MODELS, TARGET, ONSET
    if len(argv) > 2:
        if argv[2] not in ONSET_OF:
            print(f"неизвестная целевая {argv[2]!r}; доступны: {', '.join(ONSET_OF)}")
            return 2
        TARGET, ONSET = argv[2], ONSET_OF[argv[2]]
    if argv:
        if argv[0] not in FEATURE_SETS:
            print(f"неизвестный набор признаков {argv[0]!r}; доступны: {', '.join(FEATURE_SETS)}")
            return 2
        FEATURE_SET = argv[0]
    if len(argv) > 1:
        if argv[1] not in ("logit", "gbm"):
            print(f"неизвестный тип модели {argv[1]!r}; доступны: logit, gbm")
            return 2
        KIND = argv[1]
    NAME = FEATURE_SET if KIND == "logit" else f"{FEATURE_SET}_gbm"
    if TARGET != "war25":
        NAME = f"{NAME}_{TARGET}"
    FEATURES = FEATURE_SETS[FEATURE_SET][0]

    f = load()
    preds, coefs = [], []
    for T in TRAIN_YEARS:
        p, c = predict_year(f, T)
        preds.append(p)
        coefs.append(c)
    pred = pd.concat(preds, ignore_index=True)
    coefs = pd.concat(coefs, axis=1)

    base = pd.read_parquet(DATA_PROCESSED / f"baseline_predictions_{TARGET}.parquet")[["gw", "year", "decay"]]
    pred = pred.merge(base, on=["gw", "year"], how="inner")
    MODELS = (NAME, "decay")
    if NAME != "hist" and TARGET == "war25":  # для сравнения подкладываем прогнозы модели только на истории
        hist = pd.read_parquet(DATA_PROCESSED / "hist_predictions.parquet")[["gw", "year", "hist"]]
        pred = pred.merge(hist, on=["gw", "year"], how="inner")
        MODELS = (NAME, "hist", "decay")

    metrics = evaluate(pred)
    summary = pd.DataFrame(
        {m: {k: metrics[f"{m}_{k}"].mean() for k in ("brier", "auc", "brier_onset", "auc_onset")} for m in MODELS}
    ).T

    pred.to_parquet(DATA_PROCESSED / f"{NAME}_predictions.parquet", index=False)
    metrics.to_csv(REPORTS / f"{NAME}_metrics_by_year.csv", index=False, float_format="%.4f")
    summary.to_csv(REPORTS / f"{NAME}_metrics_summary.csv", float_format="%.4f")
    coefs.round(3).to_csv(REPORTS / f"{NAME}_coefficients.csv")

    print(f"набор признаков: {NAME} ({len(FEATURES)}); прогнозов: {len(pred)} строк, годы {pred.year.min()}–{pred.year.max()}")
    print("\nСреднее по годам проверки:")
    print(summary.round(4).to_string())
    if KIND == "logit":
        print("\nКоэффициенты (стандартизованные; последняя модель, обучена по 2023):")
        print(coefs.iloc[:, -1].sort_values(key=abs, ascending=False).round(3).to_string())
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
