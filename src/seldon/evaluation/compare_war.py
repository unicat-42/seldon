"""Парное сравнение наборов признаков модели войн с эталоном (чекпоинт 3.1).

Запуск:
    uv run python -m seldon compare_war [эталон] [набор ...]
        по умолчанию: эталон pitf, наборы vdem epr vdem_epr (целевая state_war25, логрегрессия)

Ничего не обучает: читает прогнозы вслепую `<набор>_state_war25_predictions.parquet`
(их пишет `model_hist`) и считает на ОДНИХ И ТЕХ ЖЕ строках начало войны в T+1 для стран
без войны в T, на трёх выборках:
    мир          все страны без государственной войны в году T;
    тихие ≥ 10   из них — те, у кого с последней войны прошло ≥ 10 лет (или её не было);
    постсоветские.
Метрики (`core/metrics.py`): сквозной AUC, AUC внутри года, Brier; `calib` — средняя
предсказанная вероятность, делённая на фактическую частоту (1 — в среднем не врёт, > 1 —
завышает риск). Δ — разница с эталоном; 95 % интервал — парный блочный бутстрэп по годам
(500 реплик: обе модели считаются на одной и той же выборке лет).

Выход: reports/war_struct_comparison.csv

Чекпоинт 3.2 — `compare_war dyad`: эталон `pitf` против двух каналов риска (`dy_indep`,
`dy_intext`, `dy_cal2` из `model_dyad`) и набора `pitf_dyad`, на пяти выборках — к трём выше
добавлены:
    внешние начала     положительные — начала войны, где в год начала у страны есть
                       государство-противник (`ext_war`, `dyads.py`); внутренние начала из
                       выборки убраны;
    внутренние начала  наоборот: положительные — начала без государства-противника.
Выход: reports/war_dyad_comparison.csv

Чекпоинт 4.2 — `compare_war live`: эталон `pitf` против доводки диад (`dy_cal` — только
калибровка `p_ext`, `dy_live_raw` — живые стычки и раздельные модели без калибровки,
`dy_live_pooled`, `dy_live` — все правки; `dy_intext` — вариант 3.2) и годового среза
предвестников (`pitf_smoke`, `pitf_live`). К выборкам добавлены «тихие ≥ 5», «после 2014».
Выход: reports/war_live_comparison.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, REPORTS
from seldon.core.entities import POST_SOVIET_GW
from seldon.core.metrics import auc, auc_within, brier, interval, year_block_bootstrap
from seldon.domains.conflict.model_hist import QUIET_YEARS
from seldon.domains.conflict.panel import ONSET_OF

TARGET = "state_war25"
N_BOOT = 500
SAMPLES = {
    "мир": lambda d: d.at_risk == 1,
    f"тихие ≥ {QUIET_YEARS}": lambda d: (d.at_risk == 1) & (d.years_since_war >= QUIET_YEARS),
    "постсоветские": lambda d: (d.at_risk == 1) & d.gw.isin(POST_SOVIET_GW),
}
# выборки чекпоинта 3.2: нужна колонка `ext_war` (есть ли в год прогноза государство-противник)
DYAD_SAMPLES = {
    **SAMPLES,
    "внешние начала": lambda d: (d.at_risk == 1) & ~((d[ONSET_OF[TARGET]] == 1) & (d.ext_war == 0)),
    "внутренние начала": lambda d: (d.at_risk == 1) & ~((d[ONSET_OF[TARGET]] == 1) & (d.ext_war == 1)),
}
DYAD_NAMES = ["pitf", "dy_indep", "dy_intext", "dy_cal2", "pitf_dyad"]
# чекпоинт 4.2: доводка диад (`dyads_live`) и годовой срез предвестников
LIVE_NAMES = ["pitf", "dy_intext", "dy_cal", "dy_live_raw", "dy_live_pooled", "dy_live", "pitf_smoke", "pitf_live"]
QUIET_SHORT = 5
LIVE_SAMPLES = {
    "мир": SAMPLES["мир"],
    f"тихие ≥ {QUIET_SHORT}": lambda d: (d.at_risk == 1) & (d.years_since_war >= QUIET_SHORT),
    f"тихие ≥ {QUIET_YEARS}": SAMPLES[f"тихие ≥ {QUIET_YEARS}"],
    "постсоветские": SAMPLES["постсоветские"],
    "после 2014": lambda d: (d.at_risk == 1) & (d.year >= 2015),
    "после 2014, тихие ≥ 5": lambda d: (d.at_risk == 1) & (d.year >= 2015) & (d.years_since_war >= QUIET_SHORT),
    "внешние начала": DYAD_SAMPLES["внешние начала"],
    "внешние начала после 2014": lambda d: DYAD_SAMPLES["внешние начала"](d) & (d.year >= 2015),
    "внутренние начала": DYAD_SAMPLES["внутренние начала"],
}
METRICS = ("auc", "auc_within_year", "brier")


def load(names: list[str], target: str = TARGET) -> pd.DataFrame:
    """Прогнозы всех наборов на общих строках (страна, год)."""
    onset = ONSET_OF[target]
    out = None
    for n in names:
        col = f"{n}_{target}"
        d = pd.read_parquet(DATA_PROCESSED / f"{col}_predictions.parquet")
        d = d[["gw", "country", "year", "at_risk", "years_since_war", onset, col]] if out is None else d[["gw", "year", col]]
        out = d if out is None else out.merge(d, on=["gw", "year"], how="inner")
    return out


def _stats(d: pd.DataFrame, cols: list[str], onset: str) -> dict:
    return {c: {"auc": auc(d[onset], d[c]), "auc_within_year": auc_within(d[onset], d[c], d.boot_year),
                "brier": float(brier(d[onset], d[c]))} for c in cols}


def compare(pred: pd.DataFrame, names: list[str], ref: str, target: str = TARGET, n_boot: int = N_BOOT, seed: int = 0,
            samples: dict | None = None) -> pd.DataFrame:
    onset = ONSET_OF[target]
    samples = SAMPLES if samples is None else samples
    cols = [f"{n}_{target}" for n in names]
    ref_col = f"{ref}_{target}"
    rows = []
    for sample, mask in samples.items():
        d = pred[mask(pred)]
        point, reps = year_block_bootstrap(d, lambda x: _stats(x, cols, onset), n_boot=n_boot, seed=seed)
        for n, c in zip(names, cols):
            row = {"sample": sample, "model": n, "n": len(d), "n_onset": int(d[onset].sum()),
                   "calib": d[c].mean() / d[onset].mean() if d[onset].mean() > 0 else np.nan}
            for k in METRICS:
                b = np.array([r[c][k] for r in reps], dtype=float)
                row[k], (row[f"{k}_lo"], row[f"{k}_hi"]) = point[c][k], interval(b)
                if c != ref_col:
                    diff = b - np.array([r[ref_col][k] for r in reps], dtype=float)
                    row[f"d_{k}"] = point[c][k] - point[ref_col][k]
                    row[f"d_{k}_lo"], row[f"d_{k}_hi"] = interval(diff)
            rows.append(row)
    return pd.DataFrame(rows)


def show(t: pd.DataFrame) -> str:
    lines = []
    for sample, g in t.groupby("sample", sort=False):
        lines.append(f"{sample}: {int(g.n.iloc[0])} страна-лет, начал войны {int(g.n_onset.iloc[0])}")
        for r in g.itertuples():
            s = (f"  {r.model:14s} AUC {r.auc:.3f} [{r.auc_lo:.3f}; {r.auc_hi:.3f}], внутри года {r.auc_within_year:.3f}, "
                 f"Brier {r.brier:.4f}, calib {r.calib:.2f}")
            if pd.notna(getattr(r, "d_auc", np.nan)):
                s += (f" | ΔAUC {r.d_auc:+.3f} [{r.d_auc_lo:+.3f}; {r.d_auc_hi:+.3f}], внутри года {r.d_auc_within_year:+.3f} "
                      f"[{r.d_auc_within_year_lo:+.3f}; {r.d_auc_within_year_hi:+.3f}], ΔBrier {r.d_brier:+.4f} [{r.d_brier_lo:+.4f}; {r.d_brier_hi:+.4f}]")
            lines.append(s)
    return "\n".join(lines)


def main_dyad(names: list[str] = DYAD_NAMES, samples: dict = DYAD_SAMPLES, out: str = "war_dyad_comparison.csv") -> int:
    pred = load(names)
    ext = pd.read_parquet(DATA_PROCESSED / "dyad_ext_country_year.parquet")
    pred = pred.merge(ext, on=["gw", "year"], how="left").fillna({"ext_war": 0})
    t = compare(pred, names, "pitf", samples=samples)
    t.to_csv(REPORTS / out, index=False, float_format="%.4f")
    print(f"эталон: pitf; целевая: начало {TARGET} в T+1; Δ — к эталону, парный бутстрэп по годам ({N_BOOT} реплик)")
    print(show(t))
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["dyad"]:
        return main_dyad()
    if argv[:1] == ["live"]:
        return main_dyad(LIVE_NAMES, LIVE_SAMPLES, "war_live_comparison.csv")
    ref, *others = argv or ["pitf", "vdem", "epr", "vdem_epr"]
    names = [ref, *others]
    t = compare(load(names), names, ref)
    t.to_csv(REPORTS / "war_struct_comparison.csv", index=False, float_format="%.4f")
    print(f"эталон: {ref}; целевая: начало {TARGET} в T+1; Δ — к эталону, парный бутстрэп по годам ({N_BOOT} реплик)")
    print(show(t))
    return 0
