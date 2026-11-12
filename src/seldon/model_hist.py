"""Модель войн на годовой панели: признаки → вероятность войны в следующем году.

Запуск:
    uv run python -m seldon.model_hist [набор] [logit|gbm] [целевая]
        набор:    hist (история, features.py) | pitf (+ режим, смертность, население) |
                  econ (+ экономика) | histecon | polit (+ лидеры, перевороты, протесты)
        целевая:  state_war25 (по умолчанию) | state_war1000 | war25 | war1000 (старые)

История (признаки `features.py`) считается по state_war25; для старой целевой war25 —
по war25 (файл features_country_year_war25.parquet, только набор hist).

Схема после аудита 12.11.2026 — две модели вместо одной (`split-population`):
  - модель НАЧАЛА учится только на страна-годах без войны (`at_risk`) и отвечает на
    вопрос «начнётся ли»;
  - модель ПРОДОЛЖЕНИЯ учится только на воюющих и отвечает «продолжится ли».
  Прогноз страны берётся из той модели, к которой она относится в году T.
  Раньше одна модель училась на всех строках сразу: её коэффициенты подстраивались под
  главный сигнал «воюет сейчас → воюет завтра», а проверялась она на онсетах, где этого
  сигнала нет. Старая схема оставлена как `<имя>_single` для сравнения.

Проверка вслепую: для каждого T = 1995…2023 учимся на парах (признаки года t, целевая
года t+1) с t+1 ≤ T, предсказываем год T+1.

Что считаем (главное — онсеты: страны, не воевавшие в году T):
  auc_onset_pooled   сквозной AUC по всем страна-годам сразу (для редких событий устойчивее);
  auc_onset_by_year  среднее AUC по годам (как в отчётах до аудита);
  brier_onset        средний квадрат ошибки вероятности;
  quiet10            то же на странах с тишиной ≥ 10 лет;
  95 % интервалы — блочный бутстрэп по годам (годы тасуются с возвращением, 500 раз);
  разница с базой (`decay`) — парная, на тех же выборках лет.
База: `decay` из baseline.py (для AUC) и `decay_cal` — она же, переведённая в вероятность
монотонной калибровкой на прошлых годах, отдельно для стран под риском и воюющих (для
Brier: сырая память — не вероятность, сравнивать с ней Brier нечестно).

AUC (area under ROC curve) — доля пар «началась / не началась», которые модель упорядочила
правильно: 0,5 — монетка, 1 — идеально.

Выход:
    data/processed/<имя>_predictions.parquet
    reports/<имя>_metrics_by_year.csv, <имя>_metrics_summary.csv, <имя>_coefficients.csv
    где <имя> = <набор>[_gbm]_<целевая>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, PROJECT_ROOT
from seldon.features import DEFAULT_TARGET, features_file
from seldon.features import FEATURES as HIST_FEATURES
from seldon.features_econ import ECON_FEATURES
from seldon.features_pitf import PITF_FEATURES
from seldon.features_polit import POLIT_FEATURES
from seldon.panel import ONSET_OF

REPORTS = PROJECT_ROOT / "reports"
TRAIN_YEARS = range(1995, 2024)
N_BOOT = 500
QUIET_YEARS = 10
FEATURE_SETS = {
    "hist": (HIST_FEATURES, "features_country_year.parquet"),
    "pitf": ([*HIST_FEATURES, *PITF_FEATURES], "features_pitf_country_year.parquet"),
    "econ": ([*HIST_FEATURES, *PITF_FEATURES, *ECON_FEATURES], "features_econ_country_year.parquet"),
    "histecon": ([*HIST_FEATURES, *ECON_FEATURES], "features_econ_country_year.parquet"),
    "polit": ([*HIST_FEATURES, *PITF_FEATURES, *POLIT_FEATURES], "features_polit_country_year.parquet"),
}
# история считается по этой целевой (state_war1000 — редкая, её историю ведём по state_war25)
HISTORY_OF = {"state_war25": "state_war25", "state_war1000": "state_war25", "war25": "war25", "war1000": "war25"}


@dataclass(frozen=True)
class Config:
    feature_set: str = "hist"
    kind: str = "logit"  # или "gbm"
    target: str = DEFAULT_TARGET

    @property
    def name(self) -> str:
        return f"{self.feature_set}{'_gbm' if self.kind == 'gbm' else ''}_{self.target}"

    @property
    def features(self) -> list[str]:
        return list(FEATURE_SETS[self.feature_set][0])

    @property
    def file(self) -> str:
        hist = HISTORY_OF[self.target]
        if hist == DEFAULT_TARGET:
            return FEATURE_SETS[self.feature_set][1]
        if self.feature_set != "hist":
            raise ValueError("для старых целевых (war25, war1000) есть только набор hist")
        return features_file(hist)


def make_model(kind: str = "logit"):
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def with_next(f: pd.DataFrame, target: str) -> pd.DataFrame:
    """Целевая следующего года на строку текущего."""
    onset = ONSET_OF[target]
    nxt = f[["gw", "year", target, onset]].assign(year=lambda d: d.year - 1)
    return f.merge(nxt.rename(columns={target: "y_next", onset: "onset_next"}), on=["gw", "year"], how="left")


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], kind: str):
    """Вероятности для test; если в обучении один класс — его доля. Возвращает (p, модель | None)."""
    if len(train) == 0:
        return np.full(len(test), np.nan), None
    if train.y_next.nunique() < 2:
        return np.full(len(test), float(train.y_next.mean())), None
    m = make_model(kind).fit(train[features], train.y_next.astype(int))
    return m.predict_proba(test[features])[:, 1], m


def forecast_asof(f: pd.DataFrame, T: int, features: list[str], target: str = DEFAULT_TARGET,
                  kind: str = "logit", return_models: bool = False):
    """Прогноз на год T+1 по признакам года T. Обучение — пары (t, t+1) с t+1 ≤ T.
    Колонки: p_onset (модель начала), p_cont (модель продолжения), p (своя модель для
    каждой страны), p_single (одна модель на всех — схема до аудита)."""
    d = f if "y_next" in f else with_next(f, target)
    train = d[(d.year <= T - 1) & d.y_next.notna()]
    test = d[d.year == T].copy()
    quiet_tr, quiet_te = train[target] == 0, test[target] == 0
    test["p_onset"], m_onset = _fit_predict(train[quiet_tr], test, features, kind)
    test["p_cont"], _ = _fit_predict(train[~quiet_tr], test, features, kind)
    test["p_single"], _ = _fit_predict(train, test, features, kind)
    test["p"] = np.where(quiet_te, test.p_onset, test.p_cont)
    test["at_risk"] = quiet_te.astype("int8")
    return (test, m_onset) if return_models else test


def run(f: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    target, onset = cfg.target, ONSET_OF[cfg.target]
    d = with_next(f, target)
    preds, coefs = [], []
    for T in TRAIN_YEARS:
        test, m = forecast_asof(d, T, cfg.features, target, cfg.kind, return_models=True)
        test = test[test.y_next.notna()]
        out = test[["gw", "country", "at_risk", "years_since_war", "p", "p_onset", "p_single"]].copy()
        out[target], out[onset] = test.y_next.astype(int), test.onset_next.astype(int)
        out["year"] = T + 1
        preds.append(out)
        if m is not None and cfg.kind == "logit":
            coefs.append(pd.Series(m[-1].coef_[0], index=cfg.features, name=T + 1))
    pred = pd.concat(preds, ignore_index=True).rename(columns={"p": cfg.name, "p_single": f"{cfg.name}_single"})
    return pred, (pd.concat(coefs, axis=1) if coefs else pd.DataFrame())


def add_baseline(pred: pd.DataFrame, target: str) -> pd.DataFrame:
    """`decay` из baseline и `decay_cal` — она же, переведённая в вероятность: монотонная
    (изотоническая) калибровка на прошлых годах, отдельно для стран под риском и воюющих.
    Порядок стран внутри года калибровка не меняет. Первые годы, пока прошлого мало, —
    сырая `decay`."""
    base = pd.read_parquet(DATA_PROCESSED / f"baseline_predictions_{target}.parquet")[["gw", "year", "decay"]]
    pred = pred.merge(base, on=["gw", "year"], how="inner")
    pred["decay_cal"] = pred.decay
    for Y in sorted(pred.year.unique()):
        for risk in (0, 1):
            past = pred[(pred.year < Y) & (pred.at_risk == risk)]
            now = (pred.year == Y) & (pred.at_risk == risk)
            if now.sum() == 0 or past[target].nunique() < 2 or past[target].sum() < 5:
                continue
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(past.decay, past[target])
            pred.loc[now, "decay_cal"] = iso.predict(pred.loc[now, "decay"])
    return pred


def _auc(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return roc_auc_score(y, p) if (len(np.unique(y)) == 2 and len(np.unique(p)) > 1) else np.nan


def evaluate(pred: pd.DataFrame, models: list[str], target: str) -> pd.DataFrame:
    onset = ONSET_OF[target]
    rows = []
    for year, g in pred.groupby("year"):
        r = g[g.at_risk == 1]
        row = {"year": year, "n": len(g), "n_at_risk": len(r), "n_onset": int(r[onset].sum())}
        for m in models:
            row[f"{m}_brier"] = ((g[m] - g[target]) ** 2).mean()
            row[f"{m}_auc"] = _auc(g[target], g[m])
            row[f"{m}_brier_onset"] = ((r[m] - r[onset]) ** 2).mean()
            row[f"{m}_auc_onset"] = _auc(r[onset], r[m])
        rows.append(row)
    return pd.DataFrame(rows)


def _stats(d: pd.DataFrame, models: list[str], target: str) -> dict:
    """Метрики одной выборки (исходной или бутстрэп-реплики)."""
    onset = ONSET_OF[target]
    r = d[d.at_risk == 1]
    q = r[r.years_since_war >= QUIET_YEARS]
    out = {}
    for m in models:
        by_year = [_auc(g[onset], g[m]) for _, g in r.groupby("boot_year")]
        out[m] = {
            "auc_all_pooled": _auc(d[target], d[m]),
            "brier_all": float(((d[m] - d[target]) ** 2).mean()),
            "auc_onset_pooled": _auc(r[onset], r[m]),
            "auc_onset_by_year": float(np.nanmean(by_year)) if len(by_year) else np.nan,
            "brier_onset": float(((r[m] - r[onset]) ** 2).mean()),
            "auc_onset_quiet10": _auc(q[onset], q[m]),
        }
    return out


def summarize(pred: pd.DataFrame, models: list[str], target: str, ref: str = "decay",
              n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Метрики с 95 % интервалами (блочный бутстрэп по годам) и парной разницей с `ref`."""
    onset = ONSET_OF[target]
    years = np.sort(pred.year.unique())
    blocks = {y: g for y, g in pred.groupby("year")}
    point = _stats(pred.assign(boot_year=pred.year), models, target)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        sample = rng.choice(years, len(years), replace=True)
        d = pd.concat([blocks[y].assign(boot_year=i) for i, y in enumerate(sample)], ignore_index=True)
        reps.append(_stats(d, models, target))
    r = pred[pred.at_risk == 1]
    rows = []
    for m in models:
        row = {"model": m, "n": len(pred), "n_at_risk": len(r), "n_onset": int(r[onset].sum()),
               "n_onset_quiet10": int(r[r.years_since_war >= QUIET_YEARS][onset].sum())}
        for k, v in point[m].items():
            b = np.array([rep[m][k] for rep in reps], dtype=float)
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = v, np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)
            if m != ref and ref in point:
                diff = b - np.array([rep[ref][k] for rep in reps], dtype=float)
                row[f"{k}_vs_{ref}"] = v - point[ref][k]
                row[f"{k}_vs_{ref}_lo"], row[f"{k}_vs_{ref}_hi"] = np.nanpercentile(diff, 2.5), np.nanpercentile(diff, 97.5)
        rows.append(row)
    return pd.DataFrame(rows)


def show(summary: pd.DataFrame, ref: str = "decay") -> str:
    lines = []
    for _, r in summary.iterrows():
        s = (f"  {r.model:28s} онсеты: AUC сквозной {r.auc_onset_pooled:.3f} [{r.auc_onset_pooled_lo:.3f}; {r.auc_onset_pooled_hi:.3f}]"
             f", по годам {r.auc_onset_by_year:.3f}, тихие≥10 {r.auc_onset_quiet10:.3f}, Brier {r.brier_onset:.4f}")
        k = f"auc_onset_pooled_vs_{ref}"
        if k in r and pd.notna(r[k]):
            s += f" | Δ к {ref}: {r[k]:+.3f} [{r[k + '_lo']:+.3f}; {r[k + '_hi']:+.3f}]"
        lines.append(s)
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    cfg = Config(*(argv[:3]))
    if cfg.feature_set not in FEATURE_SETS or cfg.kind not in ("logit", "gbm") or cfg.target not in HISTORY_OF:
        print(__doc__)
        return 2
    f = pd.read_parquet(DATA_PROCESSED / cfg.file)
    pred, coefs = run(f, cfg)
    pred = add_baseline(pred, cfg.target)
    models = [cfg.name, f"{cfg.name}_single", "decay_cal", "decay"]
    by_year = evaluate(pred, models, cfg.target)
    summary = summarize(pred, models, cfg.target)

    pred.to_parquet(DATA_PROCESSED / f"{cfg.name}_predictions.parquet", index=False)
    by_year.to_csv(REPORTS / f"{cfg.name}_metrics_by_year.csv", index=False, float_format="%.4f")
    summary.to_csv(REPORTS / f"{cfg.name}_metrics_summary.csv", index=False, float_format="%.4f")
    if not coefs.empty:
        coefs.round(3).to_csv(REPORTS / f"{cfg.name}_coefficients.csv")

    s0 = summary.iloc[0]
    print(f"{cfg.name}: {len(cfg.features)} признаков; прогнозов {len(pred)}, годы {pred.year.min()}–{pred.year.max()}; "
          f"под риском {int(s0.n_at_risk)} страна-лет, онсетов {int(s0.n_onset)} (из них после тишины ≥ {QUIET_YEARS} лет: {int(s0.n_onset_quiet10)})")
    print(show(summary))
    if not coefs.empty:
        print("\nКоэффициенты модели начала (стандартизованные; последняя, обучена по 2023):")
        print(coefs.iloc[:, -1].sort_values(key=abs, ascending=False).round(3).head(12).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
