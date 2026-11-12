"""Парное сравнение структурных наборов с боевыми валютными моделями (чекпоинт 3.3).

Запуск:
    uv run python -m seldon compare_fx [ctot_rt]
        `ctot_rt` — то же для проверки чувствительности (`econ_struct ctot_rt`), файлы с суффиксом.

Ничего не обучает: читает прогнозы вслепую, которые пишет `econ_struct`, и считает на ОДНИХ И
ТЕХ ЖЕ строках метрики эталона (`ref`) и каждого набора (`base`, `regime`, `reserves`, `reer`,
`ctot`, `struct`) — для трёх моделей:
    long / logit     годовая `deval25`, логрегрессия (эталон — `logit_long`);
    long / extree    годовая `deval25`, деревья (эталон — `extree_long`);
    kr / logit       месячная `deval15`, логрегрессия (эталон — `logit` из `econ_kr`).
Выборки:
    мир              все оцениваемые строки;
    постсоветские;
    тихие ≥ 10 лет   страны без большой девальвации (≥ 25 % за год / ≥ 15 % за месяц) в
                     последние 10 лет — «гром среди ясного неба»;
    привязки         живой индикатор привязки `peg_share12` ≥ 0,8 — там, где «пружина» возможна;
    мир, до 2010 / мир, с 2010   две половины периода — проверка, что прирост не из одной эпохи.
Метрики (`core/metrics.py`): сквозной AUC, AUC внутри года, Brier, `calib` (средняя
предсказанная вероятность / фактическая частота). Δ — разница с эталоном; 95 % интервалы —
парный блочный бутстрэп по годам (500 реплик).

Кейсы: место страны по риску (1 = самый рискованный) перед известными девальвациями — у
эталона и у наборов. Годовой контур — за 1 и 2 года до; месячный — за 12, 6 и 3 месяца.

Выход:
    reports/fx_struct_comparison.csv     все сравнения с эталоном `ref`
    reports/fx_struct_vs_base.csv        логрегрессии: наборы против `base` (вклад признаков без вклада C)
    reports/fx_struct_cases_long.csv     кейсы годового контура
    reports/fx_struct_cases_kr.csv       кейсы месячного контура
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, REPORTS
from seldon.core.entities import ISO3_TO_SHORT
from seldon.core.metrics import paired_comparison
from seldon.domains.currency.econ_struct import VARIANTS

N_BOOT = 500
QUIET_YEARS = 10
PEG_MIN = 0.8
SPLIT_YEAR = 2010  # устойчивость: прирост должен быть в обеих половинах периода
NAMES = ["ref", *VARIANTS]
# кейсы вне постсоветских стран
CASE_EXTRA = {"EGY": "Египет", "ARG": "Аргентина", "NGA": "Нигерия"}
CASE_FROM = 1997


def load_long(sfx: str = "") -> pd.DataFrame:
    pred = pd.read_parquet(DATA_PROCESSED / f"econ_long_struct_predictions{sfx}.parquet")
    panel = pd.read_parquet(DATA_PROCESSED / "econ_long_panel.parquet", columns=["iso3", "year", "big"])
    panel = panel.sort_values(["iso3", "year"])
    # лет без большой девальвации, включая год t (всё — прошлое относительно прогноза)
    recent = panel.groupby("iso3").big.transform(lambda s: s.rolling(QUIET_YEARS, min_periods=1).sum())
    panel["quiet"] = recent == 0
    return pred.merge(panel[["iso3", "year", "quiet"]], on=["iso3", "year"], how="left")


def load_kr(sfx: str = "") -> pd.DataFrame:
    pred = pd.read_parquet(DATA_PROCESSED / f"econ_kr_struct_predictions{sfx}.parquet")
    pred["month"] = pd.PeriodIndex(pred.month, freq="M")
    pred["year"] = pred.month.dt.year
    q = pred["quiet_deval15"]
    pred["quiet"] = q.isna() | (q >= 12 * QUIET_YEARS)
    return pred


def samples(d: pd.DataFrame) -> dict[str, pd.Series]:
    return {"мир": pd.Series(True, index=d.index), "постсоветские": d.post_soviet.astype(bool),
            f"тихие ≥ {QUIET_YEARS} лет": d.quiet.astype(bool), "привязки": d.peg_share12 >= PEG_MIN,
            f"мир, до {SPLIT_YEAR}": d.year < SPLIT_YEAR, f"мир, с {SPLIT_YEAR}": d.year >= SPLIT_YEAR}


def compare(pred: pd.DataFrame, contour: str, model: str, n_boot: int = N_BOOT, ref: str = "ref") -> pd.DataFrame:
    """Все наборы против эталона `ref` на всех выборках. `ref="base"` — против того же боевого
    набора с выбранной C: чистый вклад признаков, без вклада настройки регуляризации."""
    cols = {n: f"{model}__{n}" for n in NAMES}
    pred = pred[pred[list(cols.values())].notna().all(axis=1)]
    if ref != "ref":
        cols.pop("ref")
    out = []
    for sample, mask in samples(pred).items():
        d = pred[mask]
        if d.y.nunique() < 2:
            continue
        t = paired_comparison(d, cols, ref, n_boot=n_boot)
        t.insert(0, "sample", sample)
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    t.insert(0, "contour", contour)
    t.insert(1, "algo", model)
    return t


def show(t: pd.DataFrame) -> str:
    lines = []
    for (contour, algo, sample), g in t.groupby(["contour", "algo", "sample"], sort=False):
        lines.append(f"{contour} / {algo} / {sample}: строк {int(g.n.iloc[0])}, кризисных окон {int(g.positives.iloc[0])}")
        for r in g.itertuples():
            s = (f"  {r.model:9s} AUC {r.auc:.3f} [{r.auc_lo:.3f}; {r.auc_hi:.3f}], внутри года {r.auc_within_year:.3f}, "
                 f"Brier {r.brier:.4f}, calib {r.calib:.2f}")
            if pd.notna(r.d_auc):
                s += (f" | ΔAUC {r.d_auc:+.3f} [{r.d_auc_lo:+.3f}; {r.d_auc_hi:+.3f}], внутри года {r.d_auc_within_year:+.3f} "
                      f"[{r.d_auc_within_year_lo:+.3f}; {r.d_auc_within_year_hi:+.3f}], ΔBrier {r.d_brier:+.4f} "
                      f"[{r.d_brier_lo:+.4f}; {r.d_brier_hi:+.4f}]")
            lines.append(s)
    return "\n".join(lines)


def _rank(snap: pd.DataFrame, iso3: str, col: str) -> float:
    me = snap[snap.iso3 == iso3]
    if me.empty or pd.isna(me[col].iloc[0]):
        return np.nan
    return int((snap[col] > me[col].iloc[0]).sum() + 1)


def cases_long(pred: pd.DataFrame) -> pd.DataFrame:
    """Перед каждой девальвацией ≥ 25 % (`deval25`) у стран из списка кейсов: место страны по
    риску за 1 и 2 года до события среди всех оцениваемых стран того года."""
    panel = pd.read_parquet(DATA_PROCESSED / "econ_long_panel.parquet", columns=["iso3", "year", "deval25", "deval"])
    names = {**ISO3_TO_SHORT, **CASE_EXTRA}
    ev = panel[(panel.deval25 == 1) & panel.iso3.isin(names) & (panel.year >= CASE_FROM)].sort_values(["year", "iso3"])
    rows = []
    for e in ev.itertuples():
        for lead in (1, 2):
            snap = pred[pred.year == e.year - lead]
            if snap.empty or e.iso3 not in set(snap.iso3):
                continue
            row = {"country": names[e.iso3], "crisis_year": int(e.year), "deval_pct": round(100 * e.deval), "lead_years": lead,
                   "n_countries": len(snap)}
            for algo in ("logit", "extree"):
                for n in ("ref", "regime", "reserves", "reer", "ctot", "struct"):
                    row[f"{algo}_{n}"] = _rank(snap, e.iso3, f"{algo}__{n}")
            rows.append(row)
    return pd.DataFrame(rows)


def cases_kr(pred: pd.DataFrame) -> pd.DataFrame:
    """Перед каждой месячной девальвацией ≥ 15 % (`deval15`): место страны за 12, 6 и 3 месяца."""
    panel = pd.read_parquet(DATA_PROCESSED / "econ_kr_panel_deval15.parquet", columns=["iso3", "month", "deval15", "de"])
    panel["month"] = pd.PeriodIndex(panel.month, freq="M")
    names = {**ISO3_TO_SHORT, **CASE_EXTRA}
    ev = panel[(panel.deval15 == 1) & panel.iso3.isin(names) & (panel.month >= f"{CASE_FROM}-01")].sort_values(["month", "iso3"])
    rows = []
    for e in ev.itertuples():
        row = {"country": names[e.iso3], "crisis_month": str(e.month), "deval_pct": round(100 * e.de)}
        for lead in (12, 6, 3):
            snap = pred[pred.month == e.month - lead]
            row[f"n_{lead}m"] = len(snap)
            for n in ("ref", "reer", "struct"):
                row[f"{n}_{lead}m"] = _rank(snap, e.iso3, f"logit__{n}") if len(snap) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    sfx = "_ctot_rt" if "ctot_rt" in argv else ""
    long, kr = load_long(sfx), load_kr(sfx)
    t = pd.concat([compare(long, "long", "logit"), compare(long, "long", "extree"), compare(kr, "kr", "logit")], ignore_index=True)
    t.to_csv(REPORTS / f"fx_struct_comparison{sfx}.csv", index=False, float_format="%.4f")
    print(f"эталон: ref (боевые модели Этапа 2); Δ — к эталону, парный блочный бутстрэп по годам ({N_BOOT} реплик)")
    print(show(t))
    if not sfx:
        # логрегрессии: наборы против `base` (тот же боевой набор, C выбрана) — чистый вклад признаков
        vb = pd.concat([compare(long, "long", "logit", ref="base"), compare(kr, "kr", "logit", ref="base")], ignore_index=True)
        vb.to_csv(REPORTS / "fx_struct_vs_base.csv", index=False, float_format="%.4f")
        print("\nТо же против `base` (боевой набор с выбранной C): чистый вклад признаков")
        print(show(vb))
        cl, ck = cases_long(long), cases_kr(kr)
        cl.to_csv(REPORTS / "fx_struct_cases_long.csv", index=False, float_format="%.0f")
        ck.to_csv(REPORTS / "fx_struct_cases_kr.csv", index=False, float_format="%.0f")
        pd.set_option("display.width", 250)
        print("\nГодовой контур: место страны по риску перед девальвацией ≥ 25 % (1 = самый рискованный)")
        print(cl.to_string(index=False))
        print("\nМесячный контур: место страны перед девальвацией ≥ 15 % за месяц")
        print(ck.to_string(index=False))
    return 0
