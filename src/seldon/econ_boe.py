"""Повторение модели Банка Англии: Bluwstein, Buckmann, Joseph, Kapadia, Şimşek (2020/2023),
«Credit growth, the yield curve and financial crisis prediction» (BoE SWP 848, JIE 2023).

Запуск:
    uv run python -m seldon.econ_boe            # на их данных (JST R3) + свежих (JST R6)

Что делает:
  1. Собирает таблицу «страна×год → признаки, целевая» ровно по их скрипту make_data.py
     (порт на наш код, без их окружения Python 3.6 / R):
       - 12 признаков: наклон кривой доходности (`drate` = долгая ставка − короткая) и
         его среднее по другим странам (`global_drate`); 2-летние изменения отношений к ВВП
         (кредит, широкие деньги, госдолг, инвестиции, счёт текущих операций, обслуживание
         долга) и 2-летние темпы роста (ИПЦ, акции, потребление на душу); среднее по другим
         странам 2-летнего роста кредит/ВВП (`global_loan`);
       - целевая: «1» для двух лет перед системным финансовым кризисом (crisisJST),
         сам год кризиса и 4 года после — выброшены; мировые войны и Великая депрессия
         тоже выброшены.
  2. Сверяет полученную таблицу с той, что лежит в их репозитории
     (results/baseline/data_all_CV_2_crisisSHAP_.txt) — строка в строку.
  3. Повторяет их основной эксперимент: 5-кратная кросс-валидация с группировкой по
     кризису (оба года одного кризиса — в одном блоке), 10 повторов, логистическая
     регрессия без штрафа и extremely randomised trees (1000 деревьев). В статье:
     AUC логрег 0,816, деревья 0,868.
  4. Их «прогнозный» эксперимент: учимся на годах ≤ T, предсказываем все годы > T,
     T идёт с момента, когда в обучении набирается 20 кризисных наблюдений.
     Плюс наш вариант «только следующий год»: предсказываем T+1, склеиваем по всем T.
  5. То же на JST R6 (18 стран, 1870–2020).

AUC (area under ROC curve) — доля пар «кризис/не кризис», которые модель упорядочила
правильно: 0,5 — монетка, 1 — идеально. Brier — средний квадрат ошибки вероятности.

Выход:
    data/processed/boe_dataset_r3.parquet, boe_dataset_r6.parquet
    reports/boe_cv_results.csv, reports/boe_forecast_by_year.csv, reports/boe_forecast_summary.csv,
    reports/boe_coefficients.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, DATA_RAW, PROJECT_ROOT

REPORTS = PROJECT_ROOT / "reports"
BOE_DIR = DATA_RAW / "boe" / "MachineLearningCrisisPrediction-master"
JST_R3 = BOE_DIR / "data" / "JSTdatasetR3.xlsx"
JST_R6 = DATA_RAW / "jst" / "JSTdatasetR6.xlsx"
BOE_DATA_FILE = BOE_DIR / "results" / "baseline" / "data_all_CV_2_crisisSHAP_.txt"

HORIZON = 2
PREDICTORS_BASE = [
    "drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff", "cons_pdiff",
    "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff", "tloan_gdp_rdiff",
    "tdbtserv_gdp_rdiff", "global_loan", "global_drate",
]
NO_CHANGE = {"drate", "global_drate", "lrate", "srate"}
PREDICTORS = [p + str(HORIZON) for p in PREDICTORS_BASE if p not in NO_CHANGE] + [
    p for p in PREDICTORS_BASE if p in NO_CHANGE
]
RENAME = {
    "crisisJST": "crisis", "stir": "srate", "ltrate": "lrate", "iy": "inv_gdp",
    "debtgdp": "pdebt_gdp", "money": "bmon", "narrowm": "nmon", "tloans": "tloan",
    "tbus": "bloan", "thh": "hloan", "tmort": "mort", "stocks": "stock", "hpnom": "hp",
    "rconpc": "cons",
}
YEARS_PRE_CRISIS = 2
POST_CRISIS_DROP = 4
MIN_CRISIS_TRAIN = 20


# ---------------------------------------------------------------------------
# Сборка данных (порт make_data.py из репозитория BoE)
# ---------------------------------------------------------------------------


def load_jst(path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    if "rconpc" not in df.columns and "rconsbarro" in df.columns:
        df = df.rename(columns={"rconsbarro": "rconpc"})  # R6: реальное потребление на душу (Barro)
    if "stocks" not in df.columns and "eq_capgain" in df.columns:
        # R6 убрал индекс акций `stocks`; восстанавливаем уровень из прироста цен акций
        # eq_capgain (кумулятивно внутри страны) — для 2-летнего темпа роста это эквивалентно.
        df["stocks"] = df.groupby("iso")["eq_capgain"].transform(lambda s: (1 + s.fillna(0)).cumprod())
        df.loc[df.eq_capgain.isna(), "stocks"] = np.nan
    return df.sort_values(["iso", "year"]).reset_index(drop=True)


def excluded_periods(year: pd.Series) -> pd.Series:
    """Великая депрессия (1934–1938), Первая (1914–1918) и Вторая (1939–1945) мировые."""
    y = year.values
    return pd.Series(
        ((y > 1933) & (y < 1939)) | ((y > 1913) & (y < 1919)) | ((y > 1938) & (y < 1946)),
        index=year.index,
    )


def build_boe_dataset(jst: pd.DataFrame, predictors: list[str] = PREDICTORS) -> pd.DataFrame:
    """Строка — страна×год; признаки за год t, `crisis` = 1, если в t+1 или t+2 кризис."""
    df = jst.rename(columns=RENAME).copy()
    crisis_in = df["crisis"].fillna(0).values == 1
    excl = excluded_periods(df.year)
    feature_cols = [c for c in df.columns if c not in ("year", "country", "iso")]
    df.loc[excl, feature_cols] = np.nan

    df["drate"] = df.lrate - df.srate
    df["pdebt"] = df.pdebt_gdp * df.gdp
    df["inv"] = df.inv_gdp * df.gdp
    df["tdbtserv"] = df.tloan * df.lrate / 100.0
    pre_gdp = ["bmon", "nmon", "tloan", "bloan", "hloan", "mort", "ca", "cpi", "tdbtserv", "inv", "pdebt"]
    ratios = []
    for v in pre_gdp:
        df[f"{v}_gdp"] = df[v] / df.gdp
        ratios.append(f"{v}_gdp")
    g = df.groupby("iso")
    for v in ["lrate", "srate", "drate", *ratios]:
        df[f"{v}_rdiff{HORIZON}"] = g[v].diff(HORIZON)
    for v in ["stock", "cpi", "hp", "cons", "gdp", *pre_gdp]:
        lag = g[v].shift(HORIZON)
        df[f"{v}_pdiff{HORIZON}"] = (df[v] - lag) / lag
    # глобальные признаки: среднее по всем остальным странам того же года
    for src, dst in [(f"tloan_gdp_rdiff{HORIZON}", f"global_loan{HORIZON}"), ("drate", "global_drate")]:
        tot = df.groupby("year")[src].transform("sum")
        cnt = df.groupby("year")[src].transform("count")
        own = df[src].notna()
        df[dst] = (tot - df[src].fillna(0)) / (cnt - own.astype(int))
        df.loc[(cnt - own.astype(int)) == 0, dst] = np.nan

    # целевая: два года перед кризисом (как у них — по позиции в таблице)
    years = df.year.values
    isos = df.iso.values
    crisis = np.zeros(len(df), dtype=int)
    keep = np.ones(len(df), dtype=bool)
    for i in np.where(crisis_in)[0]:
        for lag in range(1, YEARS_PRE_CRISIS + 1):
            if years[i] > years.min() + lag - 1 and i - lag >= 0:
                crisis[i - lag] = 1
        keep[i] = False
        for j in range(1, POST_CRISIS_DROP + 1):
            if i + j < len(df) and isos[i + j] == isos[i]:
                keep[i + j] = False
    crisis_id = np.zeros(len(df), dtype=int)
    count = 1
    for i in range(2, len(df)):
        if crisis[i] == 1:
            if not (crisis[i - 1] == 1 and isos[i] == isos[i - 1]):
                count += 1
            crisis_id[i] = count
    free = crisis_id == 0
    crisis_id[free] = np.arange(free.sum()) + 2 + crisis_id.max()

    out = df[predictors].copy()
    out["crisis"] = crisis
    out["crisis_id"] = crisis_id
    out["year"] = years.astype(int)
    out["iso"] = isos
    out = out[~excl.values & keep].dropna().reset_index(drop=True)
    return out


def check_against_boe(ours: pd.DataFrame) -> dict:
    """Сравнить с таблицей из репозитория BoE: те же строки, те же значения (у них 3 знака)."""
    theirs = pd.read_csv(BOE_DATA_FILE, sep="\t", index_col=0)
    key = ["iso", "year"]
    m = theirs.merge(ours, on=key, how="outer", suffixes=("_boe", "_our"), indicator=True)
    res = {
        "rows_boe": len(theirs), "rows_ours": len(ours),
        "only_boe": int((m._merge == "left_only").sum()),
        "only_ours": int((m._merge == "right_only").sum()),
    }
    both = m[m._merge == "both"]
    res["crisis_mismatch"] = int((both.crisis_boe != both.crisis_our).sum())
    diffs = {p: float((both[f"{p}_boe"] - both[f"{p}_our"]).abs().max()) for p in PREDICTORS}
    res["max_abs_diff"] = max(diffs.values())
    res["max_abs_diff_by_predictor"] = diffs
    return res


# ---------------------------------------------------------------------------
# Модели и эксперименты
# ---------------------------------------------------------------------------


def make_models(seed: int = 0) -> dict:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=5000)),
        "extree": ExtraTreesClassifier(n_estimators=1000, random_state=seed, n_jobs=-1),
    }


def cross_validate(data: pd.DataFrame, predictors: list[str], n_rep: int = 10, n_folds: int = 5) -> pd.DataFrame:
    """5 блоков, оба года одного кризиса — в одном блоке; AUC считается по всем блокам повтора."""
    X, y, groups = data[predictors].values, data.crisis.values, data.crisis_id.values
    rows = []
    for rep in range(n_rep):
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=rep)
        pred = {name: np.zeros(len(y)) for name in make_models()}
        for tr, te in cv.split(X, y, groups):
            for name, m in make_models(rep).items():
                m.fit(X[tr], y[tr])
                pred[name][te] = m.predict_proba(X[te])[:, 1]
        for name, p in pred.items():
            rows.append({"rep": rep, "model": name, "auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p)})
    return pd.DataFrame(rows)


def forecast(data: pd.DataFrame, predictors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Их схема: учимся на year ≤ T, тестируем на year > T (все будущие годы).
    Наша: тестируем только year == T+1, предсказания склеиваем."""
    years = np.sort(data.year.unique())
    cum = data.groupby("year").crisis.sum().cumsum()
    splits = [T for T in years[:-1] if cum.loc[T] >= MIN_CRISIS_TRAIN]
    by_year, next_rows, coefs = [], [], []
    for T in splits:
        tr = data[data.year <= T]
        te_all = data[data.year > T]
        te_next = data[data.year == T + 1]
        for name, m in make_models(0).items():
            m.fit(tr[predictors].values, tr.crisis.values)
            p_all = m.predict_proba(te_all[predictors].values)[:, 1]
            row = {"train_end": T, "model": name, "n_train": len(tr), "n_test": len(te_all),
                   "crises_test": int(te_all.crisis.sum())}
            row["auc_all_future"] = roc_auc_score(te_all.crisis, p_all) if te_all.crisis.nunique() == 2 else np.nan
            by_year.append(row)
            if len(te_next):
                p = m.predict_proba(te_next[predictors].values)[:, 1]
                next_rows.append(te_next[["iso", "year", "crisis"]].assign(model=name, p=p, train_end=T))
            if name == "logreg":
                coefs.append(pd.Series(m[-1].coef_[0], index=predictors, name=T))
    nxt = pd.concat(next_rows)
    summary = []
    for name, g in nxt.groupby("model"):
        summary.append({"model": name, "scheme": "next_year_pooled", "n": len(g), "crises": int(g.crisis.sum()),
                        "auc": roc_auc_score(g.crisis, g.p), "brier": brier_score_loss(g.crisis, g.p)})
    by = pd.DataFrame(by_year)
    for name, g in by.groupby("model"):
        summary.append({"model": name, "scheme": "all_future_mean_over_T", "n": int(g.n_test.mean()),
                        "crises": int(g.crises_test.mean()), "auc": g.auc_all_future.mean(), "brier": np.nan})
    return by, pd.DataFrame(summary), pd.DataFrame(coefs), nxt


def main(argv: list[str]) -> int:
    results = {}
    for tag, path, sheet in [("r3", JST_R3, "Data"), ("r6", JST_R6, "Sheet1")]:
        print(f"\n=== JST {tag.upper()} ===")
        jst = load_jst(path, sheet)
        data = build_boe_dataset(jst)
        data.to_parquet(DATA_PROCESSED / f"boe_dataset_{tag}.parquet", index=False)
        print(f"строк {len(data)}, кризисных {int(data.crisis.sum())}, стран {data.iso.nunique()}, "
              f"годы {data.year.min()}–{data.year.max()}")
        if tag == "r3":
            chk = check_against_boe(data)
            print("сверка с таблицей BoE:", {k: v for k, v in chk.items() if k != "max_abs_diff_by_predictor"})
            results["check"] = chk
        cv = cross_validate(data, PREDICTORS)
        cv["dataset"] = tag
        print("кросс-валидация (среднее по 10 повторам):")
        print(cv.groupby("model")[["auc", "brier"]].agg(["mean", "std"]).round(3))
        by, summ, coefs, nxt = forecast(data, PREDICTORS)
        by["dataset"] = tag
        summ["dataset"] = tag
        print("прогноз вслепую:")
        print(summ.round(3).to_string(index=False))
        nxt.to_parquet(DATA_PROCESSED / f"boe_forecast_next_year_{tag}.parquet", index=False)
        results[tag] = (cv, by, summ, coefs)

    pd.concat([results[t][0] for t in ("r3", "r6")]).to_csv(REPORTS / "boe_cv_results.csv", index=False)
    pd.concat([results[t][1] for t in ("r3", "r6")]).to_csv(REPORTS / "boe_forecast_by_year.csv", index=False)
    pd.concat([results[t][2] for t in ("r3", "r6")]).to_csv(REPORTS / "boe_forecast_summary.csv", index=False)
    results["r6"][3].round(4).to_csv(REPORTS / "boe_coefficients.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
