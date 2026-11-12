"""Повторение глобальной модели PITF (Goldstone et al. 2010, AJPS) на их replication data.

Запуск:
    uv run python -m seldon pitf

Что делает:
  1. Читает Stata-файл из data/raw/pitf/ (см. `python -m seldon download pitf`).
  2. Строит условную логистическую регрессию (`clogit` в Stata) — ту же, что в их
     скрипте `.do`: страты «год × регион × group», признаки — тип режима (5 дамми
     относительно полной автократии), детская смертность (лог), вооружённый конфликт
     в 4+ соседних странах, дискриминация со стороны государства.
  3. Сверяет свои вероятности с опубликованными `Global model score` из Excel
     «Hits and misses» — по каждой стране-году, во всех трёх выборках A/B/C.
  4. Пишет data/processed/pitf_replication.parquet и reports/pitf_odds_ratios.csv.

Важно: условная регрессия даёт вероятности только *внутри страты* (кто из 4 стран
данного года и региона — жертва). «Абсолютную» вероятность для одной страны из неё
получить нельзя — см. отчёт сессии 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyreadstat
from statsmodels.discrete.conditional_models import ConditionalLogit

from seldon import DATA_PROCESSED, DATA_RAW, REPORTS

PITF_DIR = DATA_RAW / "pitf"
DTA = PITF_DIR / "PITF-Global-Model-Replication-Data-121709.dta"
HITS_XLS = PITF_DIR / "Table-1-Hits-and-misses-Global-model.xls"

REGIME_DUMMIES = {
    "Partial autocracy": "part_autoc",
    "Partial democracy with factionalism": "part_dem_fact",
    "Partial democracy without factionalism": "part_dem_nofact",
    "Full democracy": "full_dem",
    "Transition": "transition",
}
FEATURES = [*REGIME_DUMMIES.values(), "logim", "maccat", "disp4cat"]
SAMPLES = {1: "A2", 2: "B2", 3: "C2"}


def load_replication_data() -> pd.DataFrame:
    df, _ = pyreadstat.read_dta(DTA)
    df["sftgcode"] = df["sftgcode"].str.strip()
    for label, col in REGIME_DUMMIES.items():
        df[col] = (df["sftptv2a"] == label).astype(float)
    df["stratum"] = (
        df["year"].astype(int).astype(str)
        + "_" + df["byregn2"].astype(int).astype(str)
        + "_" + df["group"].astype(int).astype(str)
    )
    return df


def load_published_scores(sheet_suffix: str) -> pd.DataFrame:
    """Опубликованные вероятности из листа T1-Full Problem Set-<A2|B2|C2>."""
    x = pd.read_excel(HITS_XLS, sheet_name=f"T1-Full Problem Set-{sheet_suffix}", header=None)
    x = x.iloc[4:, [0, 2, 3, 4]]
    x.columns = ["sftgcode", "year", "sftpcons", "score_published"]
    x["score_published"] = pd.to_numeric(x["score_published"], errors="coerce")
    x = x.dropna(subset=["score_published"])
    x["sftgcode"] = x["sftgcode"].astype(str).str.strip()
    x["year"] = x["year"].astype(int)
    x["sftpcons"] = x["sftpcons"].astype(int)
    return x


def fit_sample(d: pd.DataFrame):
    """clogit sftpcons <features>, group(stratum)."""
    return ConditionalLogit(
        d["sftpcons"].values, d[FEATURES].astype(float), groups=d["stratum"].values
    ).fit(disp=0)


def replicate() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_replication_data()
    odds, rows = {}, []
    for s, sheet in SAMPLES.items():
        d = df[df["sample"] == s].copy()
        m = fit_sample(d)
        odds[f"sample_{sheet}"] = np.exp(m.params)
        d["xb"] = d[FEATURES].astype(float).values @ m.params.values
        # условная вероятность внутри страты = softmax(xb)
        d["score_ours"] = d.groupby("stratum")["xb"].transform(lambda v: np.exp(v) / np.exp(v).sum())
        d = d.merge(load_published_scores(sheet), on=["sftgcode", "year", "sftpcons"], how="left")
        d["sample_letter"] = sheet
        rows.append(d)
    odds_df = pd.DataFrame(odds)
    odds_df.index = FEATURES
    return odds_df, pd.concat(rows, ignore_index=True)


def main(argv: list[str]) -> int:
    odds, pred = replicate()
    print("Отношения шансов (odds ratios), база — полная автократия:")
    print(odds.round(2).to_string())
    print()
    for sheet, g in pred.groupby("sample_letter"):
        diff = (g["score_ours"] - g["score_published"]).abs()
        acc = ((g["score_ours"] > 0.5) == (g["sftpcons"] == 1)).mean()
        print(
            f"выборка {sheet}: {len(g)} строк, сверено {g['score_published'].notna().sum()}, "
            f"max|разница| = {diff.max():.4f}, точность (порог 0.5) = {acc:.3f}"
        )
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(DATA_PROCESSED / "pitf_replication.parquet", index=False)
    odds.round(4).to_csv(REPORTS / "pitf_odds_ratios.csv")
    print("\nЗаписано: data/processed/pitf_replication.parquet, reports/pitf_odds_ratios.csv")
    return 0

