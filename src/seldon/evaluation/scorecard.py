"""Сводная таблица качества основных моделей проекта — одним способом и с интервалами.

Запуск:
    uv run python -m seldon scorecard

Ничего не обучает: читает сохранённые прогнозы вслепую (их пишут `model_hist`, `econ_long`,
`econ_kr`, `econ_csd live`, `econ_boe_em`) и считает по ним метрики из `core/metrics.py`:
    auc              сквозной AUC по всем страна-периодам (0,5 — монетка, 1 — идеально);
    auc_lo, auc_hi   95 % интервал — блочный бутстрэп по годам (500 реплик);
    auc_within_year  среднее AUC по годам (умеет ли модель различать страны внутри года);
    brier            средний квадрат ошибки вероятности (у `decay` и `persist` балл — не
                     вероятность, их Brier с остальными не сравнивать);
    auc_vs_ref…      парная разница сквозного AUC с базовой моделью блока на тех же
                     выборках лет, с интервалом.

Что в таблице (строка — блок × выборка × модель):
    война, начало (`state_war25`, страны без войны в году T)      `pitf` против `decay`
    валютный кризис, год (`deval25` в t+1…t+2)                     `logit_long`, `extree_long` против `persist`
    валютный кризис, месяц (`deval15` в ближайшие 12 мес)          `logit` против `pressure`
    то же, претендент витрины (`econ_csd live`)                    `live_fx_swap` против боевого `logit`
    банковский кризис (t+1…t+2), страны вне JST                    `jst_extree` (наборы `full`, `nomoney`)

Выход: reports/scorecard.csv
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, REPORTS
from seldon.core.metrics import auc, auc_within, brier, interval, year_block_bootstrap
from seldon.domains.banking.econ_boe_em import JST_ISO

N_BOOT = 500


@dataclass(frozen=True)
class Entry:
    block: str               # блок и целевая, по-человечески
    sample: str              # выборка
    file: str                # прогнозы вслепую в data/processed
    y: str                   # колонка факта
    models: tuple[str, ...]  # колонки прогнозов; первая из `ref`, если задана, — база для сравнения
    ref: str | None = None
    query: str | None = None  # отбор строк (pandas query)


# годовой валютный блок: те же строки, что в `econ_long.metrics`, — где есть прогноз и длинного, и короткого окна
BOTH_WINDOWS = "extree_long == extree_long and extree_short == extree_short"

ENTRIES = (
    Entry("война: начало state_war25 в T+1", "мир, страны без войны в T", "pitf_state_war25_predictions.parquet",
          "onset_state_war25", ("pitf_state_war25", "decay"), ref="decay", query="at_risk == 1"),
    Entry("валютный кризис, год: deval25 в t+1…t+2", "мир", "econ_long_predictions.parquet",
          "y", ("logit_long", "extree_long", "persist"), ref="persist", query=BOTH_WINDOWS),
    Entry("валютный кризис, год: deval25 в t+1…t+2", "постсоветские", "econ_long_predictions.parquet",
          "y", ("logit_long", "extree_long", "persist"), ref="persist", query=f"post_soviet and {BOTH_WINDOWS}"),
    Entry("валютный кризис, месяц: deval15 за 12 мес", "мир", "econ_kr_predictions_deval15.parquet",
          "y", ("logit", "pressure"), ref="pressure"),
    Entry("валютный кризис, месяц: deval15 за 12 мес", "постсоветские", "econ_kr_predictions_deval15.parquet",
          "y", ("logit", "pressure"), ref="pressure", query="post_soviet"),
    # претендент витрины рядом с боевой моделью (развилка B1): те же строки и целевая, база — боевой `logit`
    Entry("валютный кризис, месяц: deval15 за 12 мес, претендент", "мир", "econ_live_fx_predictions.parquet",
          "y", ("live_fx_swap", "logit"), ref="logit"),
    Entry("валютный кризис, месяц: deval15 за 12 мес, претендент", "постсоветские", "econ_live_fx_predictions.parquet",
          "y", ("live_fx_swap", "logit"), ref="logit", query="post_soviet"),
    Entry("банковский кризис в t+1…t+2, набор full", "страны BIS вне JST", "boe_em_predictions.parquet",
          "crisis", ("jst_extree",), query="feature_set == 'full' and not jst"),
    Entry("банковский кризис в t+1…t+2, набор nomoney", "страны BIS вне JST", "boe_em_predictions.parquet",
          "crisis", ("jst_extree",), query="feature_set == 'nomoney' and not jst"),
)


def load(entry: Entry) -> pd.DataFrame:
    d = pd.read_parquet(DATA_PROCESSED / entry.file)
    if "year" not in d:  # месячные прогнозы: год из месяца «ГГГГ-ММ»
        d["year"] = d.month.astype(str).str[:4].astype(int)
    if "iso" in d:
        d["jst"] = d.iso.isin(JST_ISO)
    if entry.query:
        d = d.query(entry.query)
    return d[["year", entry.y, *entry.models]].reset_index(drop=True)


def _stat(d: pd.DataFrame, entry: Entry) -> dict:
    out = {}
    for m in entry.models:
        s = d[d[m].notna()]
        out[m] = {"auc": auc(s[entry.y], s[m]), "auc_within_year": auc_within(s[entry.y], s[m], s.boot_year),
                  "brier": brier(s[entry.y], s[m])}
    return out


def score(entry: Entry, n_boot: int = N_BOOT, seed: int = 0) -> list[dict]:
    d = load(entry)
    point, reps = year_block_bootstrap(d, lambda x: _stat(x, entry), n_boot=n_boot, seed=seed)
    rows = []
    for m in entry.models:
        s = d[d[m].notna()]
        row = {"block": entry.block, "sample": entry.sample, "model": m, "n": len(s),
               "positives": int(s[entry.y].sum()), "years": f"{int(s.year.min())}–{int(s.year.max())}", **point[m]}
        b = np.array([r[m]["auc"] for r in reps], dtype=float)
        row["auc_lo"], row["auc_hi"] = interval(b)
        if entry.ref and m != entry.ref:
            diff = b - np.array([r[entry.ref]["auc"] for r in reps], dtype=float)
            row["ref"] = entry.ref
            row["auc_vs_ref"] = point[m]["auc"] - point[entry.ref]["auc"]
            row["auc_vs_ref_lo"], row["auc_vs_ref_hi"] = interval(diff)
        rows.append(row)
    return rows


def build(n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = [r for e in ENTRIES for r in score(e, n_boot=n_boot)]
    cols = ["block", "sample", "model", "n", "positives", "years", "auc", "auc_lo", "auc_hi", "auc_within_year",
            "brier", "ref", "auc_vs_ref", "auc_vs_ref_lo", "auc_vs_ref_hi"]
    return pd.DataFrame(rows).reindex(columns=cols)


def main(argv: list[str]) -> int:
    t = build()
    REPORTS.mkdir(parents=True, exist_ok=True)
    t.to_csv(REPORTS / "scorecard.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print("Качество вслепую, 95 % интервалы — бутстрэп по годам:")
    print(t.round(3).fillna("").to_string(index=False))
    print("\n→ reports/scorecard.csv")
    return 0
