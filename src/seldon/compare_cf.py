"""Сравнение наших базовых моделей с эталоном conflictforecast.org (Mueller & Rauh).

Запуск:
    uv run python -m seldon.compare_cf

Берём их панель страна-месяц `conflictforecast_ons_armedconf_12.csv` (182 страны,
2010-01 … 2026-10). В ней:
    armedconf                 идёт ли вооружённый конфликт в этом месяце (их кодировка по UCDP)
    ons_armedconf_12_target   целевая: только для месяцев без конфликта — начнётся ли он
                              в ближайшие 12 месяцев (1/0). В месяцах с конфликтом — пусто.
    ons_armedconf_12_{naive,hist,text,all}  их четыре прогноза этой целевой.

Чтобы сравнивать модели, а не данные, наши `last` и `decay` строим на ИХ ряде `armedconf`
и оцениваем на ИХ целевой — на тех же строках, что и их прогнозы.

Наши модели (без признаков, только история конфликта):
    last12   был ли конфликт хоть раз за последние 12 месяцев (0/1)
    decay    затухающая память о конфликте по месяцам, полураспад h ∈ {6, 12, 36} месяцев;
             h выбирается по Brier на годах до года оценки (без подглядывания в будущее),
             затем память масштабируется в вероятность логистической регрессией с одним
             признаком, обученной на тех же прошлых годах (иначе Brier несравним: память —
             это не вероятность онсета).

Метрики — Brier и AUC (см. baseline.py), по годам и в сумме за 2011–2025 (2010 — разгон).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from seldon import DATA_RAW, PROJECT_ROOT
from seldon.countries import ISO3_TO_SHORT, POST_SOVIET_ISO3

CF_FILE = DATA_RAW / "conflictforecast" / "latest" / "conflictforecast_ons_armedconf_12.csv"
REPORTS = PROJECT_ROOT / "reports"
TARGET = "ons_armedconf_12_target"
THEIRS = {f"cf_{m}": f"ons_armedconf_12_{m}" for m in ("naive", "hist", "text", "all")}
HALF_LIVES = (6, 12, 36)
EVAL_YEARS = range(2011, 2026)


def load() -> pd.DataFrame:
    cf = pd.read_csv(CF_FILE, usecols=["isocode", "period", "armedconf", TARGET, *THEIRS.values()])
    cf = cf.rename(columns={v: k for k, v in THEIRS.items()})
    cf["year"] = cf["period"] // 100
    cf = cf.sort_values(["isocode", "period"]).reset_index(drop=True)
    # контроль: ряды без дыр внутри (начинаться могут позже — Южный Судан с 2011-07)
    t = (cf["period"] // 100) * 12 + cf["period"] % 100
    gaps = t.groupby(cf["isocode"]).diff().dropna()
    assert (gaps == 1).all(), "в рядах стран есть пропущенные месяцы"
    return cf


def add_history_features(cf: pd.DataFrame) -> pd.DataFrame:
    g = cf.groupby("isocode")["armedconf"]
    cf["last12"] = g.transform(lambda s: s.rolling(12, min_periods=1).max())
    for h in HALF_LIVES:
        alpha = 1 - 0.5 ** (1 / h)
        cf[f"mem_{h}"] = g.transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    return cf


def fit_decay_for_year(cf: pd.DataFrame, year: int) -> pd.Series:
    """Для года `year`: выбрать h и откалибровать на годах < year, предсказать год `year`."""
    train = cf[(cf.year < year) & cf[TARGET].notna()]
    test = cf[(cf.year == year) & cf[TARGET].notna()]
    best = None
    for h in HALF_LIVES:
        lr = LogisticRegression().fit(train[[f"mem_{h}"]], train[TARGET])
        brier = np.mean((lr.predict_proba(train[[f"mem_{h}"]])[:, 1] - train[TARGET]) ** 2)
        if best is None or brier < best[0]:
            best = (brier, h, lr)
    _, h, lr = best
    return pd.Series(lr.predict_proba(test[[f"mem_{h}"]])[:, 1], index=test.index), h


def metrics(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    y = df[TARGET]
    out = {"n": len(df), "onsets": int(y.sum())}
    for c in cols:
        out[f"brier_{c}"] = np.mean((df[c] - y) ** 2)
        out[f"auc_{c}"] = roc_auc_score(y, df[c]) if y.nunique() == 2 else np.nan
    return pd.Series(out)


def main() -> None:
    cf = add_history_features(load())
    cf["decay"] = np.nan
    chosen = {}
    for year in EVAL_YEARS:
        pred, h = fit_decay_for_year(cf, year)
        cf.loc[pred.index, "decay"] = pred
        chosen[year] = h
    print("Полураспад decay по годам (мес.):", chosen)

    cols = ["last12", "decay", *THEIRS]
    ev = cf[cf.year.isin(EVAL_YEARS) & cf[TARGET].notna()].copy()
    ev["post_soviet"] = ev.isocode.isin(POST_SOVIET_ISO3)

    by_year = ev.groupby("year").apply(metrics, cols=cols, include_groups=False)
    summary = pd.DataFrame(
        {"all": metrics(ev, cols), "post_soviet": metrics(ev[ev.post_soviet], cols)}
    ).T
    by_year.round(4).to_csv(REPORTS / "cf_comparison_by_year.csv")
    summary.round(4).to_csv(REPORTS / "cf_comparison_summary.csv")

    pd.set_option("display.width", 200)
    print("\nИтог 2011–2025 (строки страна-месяц без конфликта сейчас):")
    print(summary.round(3).to_string())
    print("\nПо годам (Brier):")
    print(by_year[[f"brier_{c}" for c in cols]].round(3).to_string())
    print("\nПо годам (AUC):")
    print(by_year[["onsets", *[f"auc_{c}" for c in cols]]].round(3).to_string())

    # Ключевые постсоветские кейсы: что давали модели за 12 месяцев до начала
    print("\nПостсоветские онсеты: месяц, с которого целевая стала 1, и прогнозы в этот месяц")
    ps = ev[ev.post_soviet].copy()
    ps["short"] = ps.isocode.map(ISO3_TO_SHORT)
    ps["prev_target"] = ps.groupby("isocode")[TARGET].shift(1)
    first = ps[(ps[TARGET] == 1) & (ps.prev_target != 1)]
    print(first[["short", "period", *cols]].round(3).to_string(index=False))
    ev.to_parquet(DATA_RAW.parent / "processed" / "cf_comparison.parquet", index=False)
    print("\nЗаписано: reports/cf_comparison_summary.csv, reports/cf_comparison_by_year.csv, "
          "data/processed/cf_comparison.parquet")


if __name__ == "__main__":
    main()
