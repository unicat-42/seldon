"""Парное сравнение наборов чекпоинта 4.1 (заражение по торговым связям, параллельный курс,
чистка `gg_missing`) с боевой месячной валютной моделью и набором `struct` чекпоинта 3.3.

Запуск:
    uv run python -m seldon compare_contagion

Ничего не обучает: читает прогнозы вслепую `econ_contagion` на горизонтах 3, 6 и 12 месяцев
и считает на ОДНИХ И ТЕХ ЖЕ оцениваемых строках метрики каждого набора и разницу:
    к `ref`     боевой логрегрессии `econ_kr` (C = 0,3);
    к `base`    тому же набору с выбранной C — вклад признаков без вклада настройки;
    к `struct`  набору чекпоинта 3.3 — что добавлено именно в 4.1.
Выборки: мир; постсоветские; привязки (`peg_share12` ≥ 0,8); мир до 2010 / с 2010; для
набора `par` — ещё строки, где премия параллельного курса есть, по двум эпохам покрытия.
Метрики (`core/metrics.py`): сквозной AUC, AUC внутри года, Brier, `calib`; 95 % интервалы —
парный блочный бутстрэп по годам (500 реплик).

Кейсы: место страны по риску (1 = самый рискованный) среди всех стран со своей валютой за
3, 6 и 12 месяцев до скачка курса — у модели с горизонтом, равным сроку до события.

Выход:
    reports/fx_contagion_comparison.csv   все сравнения (колонка `vs` — с чем сравнивается)
    reports/fx_contagion_cases.csv        контрольные кейсы
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, REPORTS
from seldon.core.entities import ISO3_TO_SHORT
from seldon.core.metrics import paired_comparison
from seldon.domains.currency.econ_contagion import HORIZONS, NAMES
from seldon.evaluation.compare_fx import N_BOOT, PEG_MIN, SPLIT_YEAR, _rank, show

# контрольные кейсы: страна, месяц скачка курса
CASES = [("RUS", "1998-08"), ("BLR", "2011-05"), ("RUS", "2014-12"), ("BLR", "2015-01"), ("KAZ", "2015-08"),
         ("AZE", "2015-02"), ("AZE", "2015-12"), ("UZB", "2017-09"), ("TUR", "2018-08"), ("EGY", "2022-03"),
         ("EGY", "2022-10"), ("NGA", "2016-06"), ("NGA", "2023-06")]
CASE_NAMES = {**ISO3_TO_SHORT, "TUR": "Турция", "EGY": "Египет", "NGA": "Нигерия"}
CASE_SETS = ["ref", "struct", "clean", "contagion", "full"]


def load(h: int) -> pd.DataFrame:
    pred = pd.read_parquet(DATA_PROCESSED / f"econ_contagion_predictions_h{h}.parquet")
    pred["month"] = pd.PeriodIndex(pred.month, freq="M")
    pred["year"] = pred.month.dt.year
    return pred


def samples(d: pd.DataFrame) -> dict[str, pd.Series]:
    return {"мир": pd.Series(True, index=d.index), "постсоветские": d.post_soviet.astype(bool),
            "привязки": d.peg_share12 >= PEG_MIN, f"мир, до {SPLIT_YEAR}": d.year < SPLIT_YEAR, f"мир, с {SPLIT_YEAR}": d.year >= SPLIT_YEAR,
            "есть параллельный курс, 1996–1999": (d.par_missing == 0) & (d.year < 2010),
            "есть параллельный курс, 2019–2025": (d.par_missing == 0) & (d.year >= 2010)}


def compare(pred: pd.DataFrame, h: int, ref: str, n_boot: int = N_BOOT) -> pd.DataFrame:
    order = [ref, *[n for n in NAMES if n != ref]]
    cols = {n: f"logit__{n}" for n in order}
    pred = pred[pred.evaluable & pred[list(cols.values())].notna().all(axis=1)]
    out = []
    for sample, mask in samples(pred).items():
        d = pred[mask]
        if d.y.nunique() < 2 or d.year.nunique() < 2:
            continue
        t = paired_comparison(d, cols, ref, n_boot=n_boot)
        t.insert(0, "sample", sample)
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    t.insert(0, "contour", f"h={h}")
    t.insert(1, "algo", f"к {ref}")
    return t


def cases() -> pd.DataFrame:
    rows = []
    for iso, month in CASES:
        e = pd.Period(month, freq="M")
        for lead in HORIZONS:
            pred = load(lead)
            snap = pred[pred.month == e - lead]
            me = snap[snap.iso3 == iso]
            row = {"country": CASE_NAMES[iso], "crisis_month": month, "lead_months": lead, "n_countries": len(snap),
                   "evaluable": bool(me.evaluable.iloc[0]) if len(me) else np.nan}
            for n in CASE_SETS:
                row[n] = _rank(snap, iso, f"logit__{n}") if len(snap) else np.nan
            for f in ("peg_share12", "pshock3", "pshock12", "pmax6", "pshock_peg6"):
                row[f] = float(me[f].iloc[0]) if len(me) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    out = []
    for h in HORIZONS:
        pred = load(h)
        for ref in ("ref", "base", "struct"):
            out.append(compare(pred, h, ref))
    t = pd.concat(out, ignore_index=True).rename(columns={"algo": "vs"})
    t.to_csv(REPORTS / "fx_contagion_comparison.csv", index=False, float_format="%.4f")
    print(f"Δ — к указанному эталону, парный блочный бутстрэп по годам ({N_BOOT} реплик)")
    print(show(t.rename(columns={"vs": "algo"})))
    c = cases()
    c.to_csv(REPORTS / "fx_contagion_cases.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    print("\nКонтрольные кейсы: место страны по риску за 3 / 6 / 12 месяцев до скачка курса (модель с тем же горизонтом)")
    print(c.round(3).to_string(index=False))
    return 0
