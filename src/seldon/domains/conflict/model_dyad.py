"""Диадная модель: какая пара стран начнёт воевать в T+1 — и что это даёт страновому прогнозу.

Запуск:
    uv run python -m seldon model_dyad

Уровень 1 — сама диадная модель. Строка (A, B, T) с признаками `dyads.FEATURES` → вероятность
того, что в T+1 у пары начнётся `dyad_war` (A ← B). Под риском — пары, не воевавшие в T.
Логрегрессия; сила регуляризации `C` выбирается ВНУТРИ обучения: последние `VAL_YEARS` лет
обучения — проверочные, остальное — учебное, критерий — логарифмическая потеря; в тест не
подглядываем. Проверка вслепую: T = 1995…2024 → прогноз на 1996…2025, обучение — пары
(t, t+1) с t+1 ≤ T. База — та же логрегрессия на `dyads.BASE_FEATURES` (память пары + общая
граница). Метрики — `core/metrics.py`, 95 % интервалы и разница с базой — парный блочный
бутстрэп по годам.

Уровень 2 — два канала риска для страны A (начало `state_war25` в T+1, страны без войны в T):
    p_ext(A) = 1 − ∏_B (1 − p_{A←B})          внешний канал: свёртка диадных прогнозов
    dy_indep   1 − (1 − p_pitf)·(1 − p_ext)     формула независимых каналов поверх `pitf` как есть
    dy_intext  1 − (1 − p_int)·(1 − p_ext)      конкурирующие риски: p_int — та же модель `pitf`,
                                               обученная только на ВНУТРЕННИХ началах (без
                                               государства-противника), чтобы внешние начала
                                               не считались дважды
    dy_cal2    логрегрессия на (logit p_pitf, logit p_ext), обученная на прогнозах вслепую
               прошлых лет (пока лет меньше `CAL_MIN_YEARS` — dy_indep)
Четвёртый вариант — набор `pitf_dyad` (свёрнутые диадные признаки прямо в `pitf`) — считает
`model_hist pitf_dyad`. Сравнение всех с эталоном `pitf` — `compare_war dyad`.

Выход:
    data/processed/dyad_predictions.parquet        прогнозы пар вслепую (year — год прогноза)
    data/processed/dyad_country_predictions.parquet  p_ext по странам, включая прогноз на YEAR_TO+1
    data/processed/dy_<вариант>_state_war25_predictions.parquet   для `compare_war`
    reports/dyad_metrics_summary.csv, dyad_coefficients.csv, dyad_cases.csv, dyad_top_<год>.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, REPORTS
from seldon.core.metrics import auc, auc_within, brier, interval, year_block_bootstrap
from seldon.core.temporal import label_known
from seldon.domains.conflict import model_hist
from seldon.domains.conflict.dyads import BASE_FEATURES, FEATURES
from seldon.evaluation.backtest import backtest
from seldon.sources.ucdp import YEAR_TO

HORIZON = 1
TRAIN_YEARS = range(1995, YEAR_TO)  # T = 1995…2024 → прогноз на 1996…2025
C_GRID = (0.01, 0.1, 1.0)
DEFAULT_C = 0.1  # пока в обучении слишком мало событий, чтобы выбирать
VAL_YEARS = 5
MIN_EVENTS_FIT, MIN_EVENTS_VAL = 10, 3
CAL_MIN_YEARS, CAL_MIN_EVENTS = 8, 20
N_BOOT = 500
TARGET = "state_war25"
EPS = 1e-6

# (подпись, A, B, год начала) — где стояла пара перед известными войнами
CASES = [
    ("Грузия ← Россия", 372, 365, 2008), ("Россия ← Грузия", 365, 372, 2008),
    ("Украина ← Россия", 369, 365, 2014), ("Россия ← Украина", 365, 369, 2014),
    ("Украина ← Россия", 369, 365, 2022), ("Россия ← Украина", 365, 369, 2022),
    ("Армения ← Азербайджан", 371, 373, 2020), ("Азербайджан ← Армения", 373, 371, 2020),
    ("Кыргызстан ← Таджикистан", 703, 702, 2021), ("Таджикистан ← Кыргызстан", 702, 703, 2021),
    ("Таиланд ← Камбоджа", 800, 811, 2025), ("Камбоджа ← Таиланд", 811, 800, 2025),
    ("Ливан ← Израиль", 660, 666, 2023), ("Израиль ← Ливан", 666, 660, 2023),
    ("Ливан ← Израиль", 660, 666, 2024), ("Израиль ← Ливан", 666, 660, 2024),
    ("Иран ← Израиль", 630, 666, 2024), ("Израиль ← Иран", 666, 630, 2024),
]


def make_model(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))


def with_next(d: pd.DataFrame) -> pd.DataFrame:
    """Целевая пары в следующем году на строку текущего (NaN, если пары в T+1 нет)."""
    nxt = d[["a", "b", "year", "dyad_war"]].assign(year=d.year - 1).rename(columns={"dyad_war": "y_next"})
    return d.merge(nxt, on=["a", "b", "year"], how="left")


def choose_C(train: pd.DataFrame, features: list[str], grid: tuple[float, ...] = C_GRID) -> float:
    """Сила регуляризации по времени внутри обучения: последние `VAL_YEARS` лет — проверка."""
    cut = train.year.max() - VAL_YEARS
    fit, val = train[train.year <= cut], train[train.year > cut]
    if fit.y_next.sum() < MIN_EVENTS_FIT or val.y_next.sum() < MIN_EVENTS_VAL:
        return DEFAULT_C
    loss = {C: log_loss(val.y_next.astype(int), make_model(C).fit(fit[features], fit.y_next.astype(int)).predict_proba(val[features])[:, 1],
                        labels=[0, 1]) for C in grid}
    return min(loss, key=loss.get)


def forecast_asof(d: pd.DataFrame, T: int, features: list[str] = FEATURES, base_features: list[str] = BASE_FEATURES,
                  C: float | None = None, return_model: bool = False):
    """Прогноз начала войны пары в T+1 по признакам года T. Обучение — пары под риском с
    известной к T меткой (t+1 ≤ T). Колонки: p_dyad (модель), p_base (память + граница), C."""
    d = d if "y_next" in d else with_next(d)
    train = d[label_known(d.year, T, HORIZON) & d.y_next.notna() & (d.dyad_war == 0)]
    test = d[(d.year == T) & (d.dyad_war == 0)].copy()
    if len(test) == 0 or len(train) == 0 or train.y_next.nunique() < 2:
        rate = float(train.y_next.mean()) if len(train) else np.nan
        test["p_dyad"], test["p_base"], test["C"] = rate, rate, np.nan
        return (test, None) if return_model else test
    y = train.y_next.astype(int)
    c = choose_C(train, features) if C is None else C
    m = make_model(c).fit(train[features], y)
    test["p_dyad"] = m.predict_proba(test[features])[:, 1]
    test["p_base"] = make_model(1.0).fit(train[base_features], y).predict_proba(test[base_features])[:, 1]
    test["C"] = c
    return (test, m) if return_model else test


def run(d: pd.DataFrame, origins=TRAIN_YEARS) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = with_next(d)
    coefs = []

    def step(T: int) -> pd.DataFrame:
        test, m = forecast_asof(d, T, return_model=True)
        if m is not None:
            coefs.append(pd.Series(m[-1].coef_[0], index=FEATURES, name=T + 1))
        out = test[test.y_next.notna()][["a", "b", "contig_close", "b_power", "p_dyad", "p_base", "C"]].copy()
        out["onset"] = test.loc[out.index, "y_next"].astype(int)
        out["year"] = T + 1
        return out

    return backtest(origins, step), (pd.concat(coefs, axis=1) if coefs else pd.DataFrame())


def _stats(x: pd.DataFrame) -> dict:
    return {m: {"auc": auc(x.onset, x[m]), "auc_within_year": auc_within(x.onset, x[m], x.boot_year),
                "brier": float(brier(x.onset, x[m]))} for m in ("p_dyad", "p_base")}


def summarize(pred: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Метрики диадной модели и базы с 95 % интервалами; разница — парная."""
    samples = {"все пары": pred, "близкие соседи": pred[pred.contig_close == 1], "без общей границы": pred[pred.contig_close == 0]}
    rows = []
    for name, x in samples.items():
        point, reps = year_block_bootstrap(x, _stats, n_boot=n_boot, seed=seed)
        for m in ("p_dyad", "p_base"):
            row = {"sample": name, "model": m, "n": len(x), "n_onset": int(x.onset.sum()),
                   "calib": x[m].mean() / x.onset.mean() if x.onset.mean() > 0 else np.nan}
            for k in ("auc", "auc_within_year", "brier"):
                b = np.array([r[m][k] for r in reps], dtype=float)
                row[k], (row[f"{k}_lo"], row[f"{k}_hi"]) = point[m][k], interval(b)
                if m == "p_dyad":
                    diff = b - np.array([r["p_base"][k] for r in reps], dtype=float)
                    row[f"d_{k}"] = point[m][k] - point["p_base"][k]
                    row[f"d_{k}_lo"], row[f"d_{k}_hi"] = interval(diff)
            rows.append(row)
    return pd.DataFrame(rows)


def cases_table(pred: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """Место пары среди всех пар мира под риском в прогнозе на год начала войны."""
    rows = []
    for label, a, b, year in CASES:
        row = {"case": label, "a": a, "b": b, "year": year}
        x = pred[(pred.a == a) & (pred.b == b) & (pred.year == year)]
        prev = d[(d.a == a) & (d.b == b) & (d.year == year - 1)]
        if len(x):
            peers = pred[pred.year == year]
            r = x.iloc[0]
            row |= {"status": "начало войны" if r.onset == 1 else "под риском, войны пары по UCDP не было", "n_pairs": len(peers),
                    "rank": int((peers.p_dyad > r.p_dyad).sum()) + 1, "rank_base": int((peers.p_base > r.p_base).sum()) + 1,
                    "p_dyad": r.p_dyad, "p_base": r.p_base}
        elif len(prev) == 0:
            row["status"] = "пары нет во вселенной (не соседи, B не держава)"
        else:
            row["status"] = "пара уже воевала в предыдущем году — не начало"
        rows.append(row)
    return pd.DataFrame(rows)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def internal_channel(f: pd.DataFrame, ext: pd.DataFrame, features: list[str], origins=TRAIN_YEARS) -> pd.DataFrame:
    """`p_int`: модель начала `pitf`, обученная только на внутренних началах (в год начала у
    страны нет государства-противника). gw, year (год прогноза), p_int."""
    d = model_hist.with_next(f, TARGET)
    e = ext.assign(year=ext.year - 1).rename(columns={"ext_war": "ext_next"})
    d = d.merge(e, on=["gw", "year"], how="left").fillna({"ext_next": 0})

    def step(T: int) -> pd.DataFrame:
        train = d[label_known(d.year, T, HORIZON) & d.y_next.notna() & (d[TARGET] == 0)]
        test = d[(d.year == T) & (d[TARGET] == 0)]
        y = ((train.y_next == 1) & (train.ext_next == 0)).astype(int)
        if y.nunique() < 2:
            return None
        p = model_hist.make_model("logit").fit(train[features], y).predict_proba(test[features])[:, 1]
        return pd.DataFrame({"gw": test.gw.to_numpy(), "year": T + 1, "p_int": p})

    return backtest(origins, step)


def combine(country: pd.DataFrame) -> pd.DataFrame:
    """Варианты двух каналов. `country`: год, at_risk, onset, p_pitf, p_ext, p_int — строки
    стран под риском. Калибровка `dy_cal2` для года Y учится только на годах < Y."""
    c = country.copy()
    c["dy_indep"] = 1 - (1 - c.p_pitf) * (1 - c.p_ext)
    c["dy_intext"] = 1 - (1 - c.p_int) * (1 - c.p_ext)
    c["dy_cal2"] = c.dy_indep
    X = np.column_stack([_logit(c.p_pitf), _logit(c.p_ext)])
    for year in sorted(c.year.unique()):
        past, now = (c.year < year).to_numpy(), (c.year == year).to_numpy()
        if c.year[past].nunique() < CAL_MIN_YEARS or c.onset[past].sum() < CAL_MIN_EVENTS:
            continue
        m = LogisticRegression(C=1000.0, max_iter=1000).fit(X[past], c.onset[past].astype(int))
        c.loc[now, "dy_cal2"] = m.predict_proba(X[now])[:, 1]
    return c


def external_channel(pred: pd.DataFrame) -> pd.DataFrame:
    """Свёртка диадных прогнозов на страну A: gw, year, p_ext, top_b (самый опасный противник)."""
    g = pred.assign(q=1 - pred.p_dyad).groupby(["a", "year"])
    out = (1 - g.q.prod()).rename("p_ext").reset_index()
    top = pred.loc[pred.groupby(["a", "year"]).p_dyad.idxmax(), ["a", "year", "b"]].rename(columns={"b": "top_b"})
    return out.merge(top, on=["a", "year"]).rename(columns={"a": "gw"})


def main(argv: list[str]) -> int:
    d = pd.read_parquet(DATA_PROCESSED / "dyad_features.parquet")
    names = pd.read_parquet(DATA_PROCESSED / "panel_country_year.parquet", columns=["gw", "country"]).drop_duplicates("gw").set_index("gw").country

    # --- уровень 1 ---
    pred, coefs = run(d)
    summary = summarize(pred)
    pred.to_parquet(DATA_PROCESSED / "dyad_predictions.parquet", index=False)
    summary.to_csv(REPORTS / "dyad_metrics_summary.csv", index=False, float_format="%.5f")
    coefs.round(3).to_csv(REPORTS / "dyad_coefficients.csv")
    cases = cases_table(pred, d)
    cases.to_csv(REPORTS / "dyad_cases.csv", index=False, float_format="%.5f")

    print(f"диадная модель: {len(FEATURES)} признаков; прогнозов {len(pred)}, годы {pred.year.min()}–{pred.year.max()}; "
          f"начал войны пары {int(pred.onset.sum())} ({pred.onset.mean():.3%})")
    print("выбранное C по годам прогноза:", pred.groupby("year").C.first().value_counts().to_dict())
    for r in summary.itertuples():
        s = (f"  {r.sample:18s} {r.model:7s} n {r.n:6d}, начал {r.n_onset:3d}: AUC {r.auc:.3f} [{r.auc_lo:.3f}; {r.auc_hi:.3f}], "
             f"внутри года {r.auc_within_year:.3f}, Brier {r.brier:.5f}, calib {r.calib:.2f}")
        if r.model == "p_dyad":
            s += (f" | Δ к базе: AUC {r.d_auc:+.3f} [{r.d_auc_lo:+.3f}; {r.d_auc_hi:+.3f}], "
                  f"Brier {r.d_brier:+.5f} [{r.d_brier_lo:+.5f}; {r.d_brier_hi:+.5f}]")
        print(s)
    print("\nКоэффициенты (стандартизованные; последняя модель, обучена по 2024):")
    print(coefs.iloc[:, -1].round(3).to_string())
    print("\nГде стояла пара перед началом (место среди пар мира под риском; база — в скобках):")
    for r in cases.itertuples():
        place = f"{int(r.rank)} из {int(r.n_pairs)} (база {int(r.rank_base)}), p = {r.p_dyad:.3f}" if pd.notna(getattr(r, "rank", np.nan)) else ""
        print(f"  {r.year} {r.case:26s} {r.status}{': ' + place if place else ''}")

    # --- прогноз на следующий год ---
    nxt = forecast_asof(d, YEAR_TO)
    top = nxt.sort_values("p_dyad", ascending=False).head(15)[["a", "b", "p_dyad", "p_base", *FEATURES]].copy()
    top.insert(2, "pair", top.a.map(names) + " ← " + top.b.map(names))
    top.to_csv(REPORTS / f"dyad_top_{YEAR_TO + 1}.csv", index=False, float_format="%.4f")
    print(f"\nТоп-15 пар под риском на {YEAR_TO + 1} год (пары, уже воюющие в {YEAR_TO}, сюда не входят):")
    print(top[["pair", "p_dyad", "p_base"]].round(3).to_string(index=False))

    # --- уровень 2 ---
    ext = pd.read_parquet(DATA_PROCESSED / "dyad_ext_country_year.parquet")
    pitf_col = f"pitf_{TARGET}"
    base = pd.read_parquet(DATA_PROCESSED / f"{pitf_col}_predictions.parquet")
    f = pd.read_parquet(DATA_PROCESSED / "features_pitf_country_year.parquet")
    p_int = internal_channel(f, ext, model_hist.FEATURE_SETS["pitf"][0])
    p_ext = external_channel(pd.concat([pred, nxt.assign(year=YEAR_TO + 1)], ignore_index=True))
    p_ext.assign(country=p_ext.gw.map(names), top_b_name=p_ext.top_b.map(names)).to_parquet(
        DATA_PROCESSED / "dyad_country_predictions.parquet", index=False)
    onset = model_hist.ONSET_OF[TARGET]
    c = base[base.at_risk == 1][["gw", "year", "at_risk", onset, pitf_col]].rename(columns={onset: "onset", pitf_col: "p_pitf"})
    c = c.merge(p_ext[["gw", "year", "p_ext"]], on=["gw", "year"], how="left").merge(p_int, on=["gw", "year"], how="left")
    c["p_ext"] = c.p_ext.fillna(0.0)  # у страны нет ни одной пары под риском
    c["p_int"] = c.p_int.fillna(c.p_pitf)
    c = combine(c)
    for v in ("dy_indep", "dy_intext", "dy_cal2"):
        col = f"{v}_{TARGET}"
        out = base[["gw", "year", "at_risk", pitf_col]].merge(c[["gw", "year", v]], on=["gw", "year"], how="left")
        out[col] = out[v].fillna(out[pitf_col])  # воюющие страны — прогноз `pitf` как есть
        out[["gw", "year", "at_risk", col]].to_parquet(DATA_PROCESSED / f"{col}_predictions.parquet", index=False)
    print(f"\nуровень 2: варианты двух каналов записаны для {len(c)} страна-лет под риском; сравнение — `compare_war dyad`")
    return 0
