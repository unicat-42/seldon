"""Месячная модель войн: начнётся ли (или продолжится ли) война в ближайшие 3 / 6 / 12 месяцев.

Запуск:
    uv run python -m seldon model_month          # целевая `state_war25`, горизонты 3 / 6 / 12
    uv run python -m seldon model_month cf       # целевая conflictforecast, горизонты 3 / 12

Строка (страна, месяц m) → метка: была ли война хотя бы в одном из месяцев m+1 … m+h.
Две модели, как в годовой `model_hist`:
  - НАЧАЛО — учится на «мирных» строках (`quiet_true`: 12 месяцев без войны по факту);
  - ПРОДОЛЖЕНИЕ — на остальных.
  Строка теста получает модель по тому, что о стране ИЗВЕСТНО (`quiet_known`: события по
  m − lag, стороны — по прошлогоднему списку ACD), а оценивается начало на строках, мирных по
  факту (`quiet_true`) — целевая и отбор строк согласованы (инвариант 3).

Проверка вслепую, как в `econ_contagion`: модель для месяцев года T обучается один раз, на
метках, известных к концу декабря T−1. Метка строки t известна, когда закрылось её окно И
вышли события последнего месяца окна: t + h + lag ≤ декабрь T−1. T = 1996 … 2025; оценка —
по строкам, чьё окно целиком лежит в выверенном GED (m + h ≤ декабрь 2025). Строки 2026 года
(Candidate Events) получают прогноз, но в оценку не входят.

Логрегрессия со стандартизацией; сила регуляризации `C` выбирается ВНУТРИ обучения по
времени: последние `VAL_YEARS` лет обучения — проверочные, критерий — логарифмическая потеря
(отдельно для каждого набора и горизонта, в тест не подглядываем).

Наборы (`SETS`; признаки — `month_features.py`):
    base       только память о своей войне (4 признака) — простая база;
    hist       история войны и война у соседей;
    dry        «сухой» эталон: как pitf, но погибшие считаются только в месяцы войны — в
               обычной истории (`ld_war_*`) у мирной страны уже сидят её субпороговые стычки;
    pitf       hist + режим, детская смертность, население — месячный аналог боевого `pitf`,
               ЭТАЛОН для сравнения;
    smoke      pitf + свои предвестники (стычки 1–24 погибших, типы насилия 2 и 3);
    smoke_nb   smoke + предвестники у соседей;
    live       pitf + столкновения с другими государствами;
    full       всё вместе;
    pitf_l0, full_l0   те же наборы без лага событий (`lag = 0`) — чувствительность и честное
               сравнение с conflictforecast, чьи признаки включают текущий месяц.
Для целевой conflictforecast (`cf`): cf_hist (история + память о любом насилии + население),
cf_full (всё), и они же с `_l0`.

Выход: data/processed/war_month_predictions_h{3,6,12}.parquet,
war_month_cf_predictions_h{3,12}.parquet, reports/war_month_coefficients.csv
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, REPORTS
from seldon.core.temporal import label_known
from seldon.domains.conflict.features_pitf import PITF_FEATURES
from seldon.domains.conflict.month_features import BASE, CFX, EVENT_LAG, HIST, HIST_DRY, LIVE, SMOKE, SMOKE_NB
from seldon.sources.ucdp import YEAR_TO

HORIZONS = (3, 6, 12)
CF_HORIZONS = (3, 12)
TEST_YEARS = range(1996, YEAR_TO + 2)  # 1996 … 2026 (2026 — только прогноз, без оценки)
CF_TEST_YEARS = range(2010, YEAR_TO + 2)
TRAIN_FROM = pd.Period("1990-01", freq="M")  # 1989 год — разгон окон
LABEL_END = pd.Period(f"{YEAR_TO}-12", freq="M")  # последний месяц выверенного GED
C_GRID = (0.01, 0.1, 1.0)
DEFAULT_C = 0.1
VAL_YEARS = 5
MIN_EVENTS_FIT, MIN_EVENTS_VAL = 20, 5

SETS: dict[str, list[str]] = {
    "base": BASE,
    "hist": HIST,
    "dry": [*HIST_DRY, *PITF_FEATURES],
    "pitf": [*HIST, *PITF_FEATURES],
    "smoke": [*HIST, *PITF_FEATURES, *SMOKE],
    "smoke_nb": [*HIST, *PITF_FEATURES, *SMOKE, *SMOKE_NB],
    "live": [*HIST, *PITF_FEATURES, *LIVE],
    "full": [*HIST, *PITF_FEATURES, *SMOKE, *SMOKE_NB, *LIVE],
}
CF_SETS: dict[str, list[str]] = {
    "cf_hist": [*HIST, *CFX, "log_pop", "pop_missing"],
    "cf_full": [*HIST, *PITF_FEATURES, *SMOKE, *SMOKE_NB, *LIVE, *CFX],
}
LAG0 = ("pitf", "full")  # какие наборы считаем ещё и без лага событий


@dataclass(frozen=True)
class Spec:
    """Целевая: колонка состояния месяца, «мирная» по факту, «мирная» по известному."""
    name: str
    state: str
    quiet_true: str
    quiet_known: str


WAR = Spec("war", "state_war25", "quiet_true", "quiet_known")
CF = Spec("cf", "ac", "ac_quiet_true", "ac_quiet_known")


def make_model(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def with_label(f: pd.DataFrame, h: int, state: str = WAR.state, label_end: pd.Period = LABEL_END) -> pd.DataFrame:
    """`y` — максимум состояния за месяцы m+1 … m+h; NaN, если окно выходит за `label_end`
    или за конец ряда страны (инвариант 6: чего нет в хронике — то пропуск, а не ноль)."""
    f = f.sort_values(["gw", "o"]).reset_index(drop=True)
    g = f.groupby("gw")[state]
    fut = pd.concat([g.shift(-k) for k in range(1, h + 1)], axis=1)
    y = fut.max(axis=1).where(fut.notna().all(axis=1))
    return f.assign(y=y.where(f.ym + h <= label_end))


def choose_C(train: pd.DataFrame, features: list[str], grid: tuple[float, ...] = C_GRID) -> float:
    """Регуляризация по времени внутри обучения: последние `VAL_YEARS` лет — проверка."""
    cut = train.o.max() - 12 * VAL_YEARS
    fit, val = train[train.o <= cut], train[train.o > cut]
    if fit.y.sum() < MIN_EVENTS_FIT or val.y.sum() < MIN_EVENTS_VAL or fit.y.nunique() < 2:
        return DEFAULT_C
    yf, yv = fit.y.astype(int), val.y.astype(int)
    loss = {C: log_loss(yv, make_model(C).fit(fit[features], yf).predict_proba(val[features])[:, 1], labels=[0, 1]) for C in grid}
    return min(loss, key=loss.get)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], C: float | None):
    """(вероятности, C, модель | None); если в обучении один класс — его доля."""
    if len(train) == 0:
        return np.full(len(test), np.nan), np.nan, None
    if train.y.nunique() < 2:
        return np.full(len(test), float(train.y.mean())), np.nan, None
    c = choose_C(train, features) if C is None else C
    m = make_model(c).fit(train[features], train.y.astype(int))
    return m.predict_proba(test[features])[:, 1], c, m


def forecast_asof(f: pd.DataFrame, as_of: pd.Period, features: list[str], h: int, spec: Spec = WAR, lag: int = EVENT_LAG,
                  test: pd.DataFrame | None = None, C: float | None = None, return_model: bool = False):
    """Прогноз на окно m+1 … m+h. Обучение — строки, чья метка известна к концу `as_of`
    (t + h + lag ≤ as_of). `test` — строки, для которых нужен прогноз (по умолчанию — месяц
    `as_of`). Колонки: p_onset, p_cont, p (по известному состоянию), C."""
    d = f if "y" in f else with_label(f, h, spec.state)
    known = d[label_known(d.ym, as_of - lag, h) & d.y.notna() & (d.ym >= TRAIN_FROM)]
    test = (d[d.ym == as_of] if test is None else test).copy()
    quiet = known[spec.quiet_true] == 1
    test["p_onset"], c, m = _fit_predict(known[quiet], test, features, C)
    test["p_cont"], _, _ = _fit_predict(known[~quiet], test, features, DEFAULT_C if C is None else C)
    test["p"] = np.where(test[spec.quiet_known] == 1, test.p_onset, test.p_cont)
    test["C"] = c
    return (test, m) if return_model else test


def _year(d: pd.DataFrame, T: int, sets: dict[str, list[str]], h: int, spec: Spec, lag: int, suffix: str):
    as_of = pd.Period(f"{T - 1}-12", freq="M")
    test = d[d.year == T]
    out = test[["gw", "o"]].copy()
    coefs = {}
    for name, feats in sets.items():
        t, m = forecast_asof(d, as_of, feats, h, spec, lag, test=test, return_model=True)
        out[f"p__{name}{suffix}"], out[f"po__{name}{suffix}"], out[f"C__{name}{suffix}"] = t.p.to_numpy(), t.p_onset.to_numpy(), t.C.to_numpy()
        if m is not None:
            coefs[name] = pd.Series(m[-1].coef_[0], index=feats)
    return out, coefs


def run(f: pd.DataFrame, h: int, sets: dict[str, list[str]], spec: Spec = WAR, lag: int = EVENT_LAG, years=TEST_YEARS,
        suffix: str = "", n_jobs: int = -1) -> tuple[pd.DataFrame, dict]:
    """Прогнозы вслепую по всем годам `years` для наборов `sets`. Возвращает таблицу
    (gw, o, p__<набор>, po__<набор> — модель начала для всех строк, C__<набор>) и
    коэффициенты модели начала последнего года."""
    d = with_label(f, h, spec.state)
    res = Parallel(n_jobs=n_jobs)(delayed(_year)(d, T, sets, h, spec, lag, suffix) for T in years)
    pred = pd.concat([r[0] for r in res], ignore_index=True)
    keep = ["gw", "country", "ym", "o", "year", "month", "y", spec.state, spec.quiet_true, spec.quiet_known, "qmonths"]
    return d[keep].merge(pred, on=["gw", "o"], how="inner"), res[-1][1]


def main(argv: list[str]) -> int:
    f1 = pd.read_parquet(DATA_PROCESSED / "war_month_features.parquet")
    f0 = pd.read_parquet(DATA_PROCESSED / "war_month_features_lag0.parquet")
    if argv[:1] == ["cf"]:
        for h in CF_HORIZONS:
            a, _ = run(f1, h, CF_SETS, CF, EVENT_LAG, CF_TEST_YEARS)
            b, _ = run(f0, h, CF_SETS, CF, 0, CF_TEST_YEARS, suffix="_l0")
            # «мирная по известному» при lag = 0 — это и есть их отбор строк
            b = b.rename(columns={CF.quiet_known: "ac_quiet_known_l0"})
            out = a.merge(b[["gw", "o", "ac_quiet_known_l0", *[c for c in b if "__" in c]]], on=["gw", "o"])
            out.to_parquet(DATA_PROCESSED / f"war_month_cf_predictions_h{h}.parquet", index=False)
            e = out[out.y.notna() & (out.ac_quiet_true == 1)]
            print(f"целевая conflictforecast, h = {h:2d}: строк оценки {len(e)}, положительных {int(e.y.sum())} ({e.y.mean():.2%})")
        return 0
    coefs = []
    for h in HORIZONS:
        a, c = run(f1, h, SETS, WAR, EVENT_LAG)
        b, _ = run(f0, h, {k: SETS[k] for k in LAG0}, WAR, 0, suffix="_l0")
        out = a.merge(b[["gw", "o", *[x for x in b if "__" in x]]], on=["gw", "o"])
        out.to_parquet(DATA_PROCESSED / f"war_month_predictions_h{h}.parquet", index=False)
        coefs += [s.rename(f"{name}_h{h}") for name, s in c.items()]
        e = out[out.y.notna() & (out.quiet_true == 1)]
        cs = {n: out[out.year == YEAR_TO][f"C__{n}"].iloc[0] for n in SETS}
        print(f"h = {h:2d}: мирных страна-месяцев в оценке {len(e)}, с началом войны в окне {int(e.y.sum())} ({e.y.mean():.2%}); "
              f"C в последнем году: {cs}")
    pd.concat(coefs, axis=1).rename_axis("feature").to_csv(REPORTS / "war_month_coefficients.csv", float_format="%.4f")
    return 0
