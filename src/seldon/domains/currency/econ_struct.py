"""Структурный валютный блок (чекпоинт 3.3): проверка вслепую четырёх семейств признаков
против боевых валютных моделей — в годовом (`deval25`) и месячном (`deval15`) контурах.

Запуск:
    uv run python -m seldon econ_struct [long] [kr] [ctot_rt]
        без аргументов — оба контура; `ctot_rt` — проверка чувствительности: сырьевые условия
        торговли с лагом 1 месяц вместо 6 («цены видны сразу»), файлы с суффиксом `_ctot_rt`.

Признаки — `struct_features.py` (режим курса, резервы, REER, условия торговли; 7 признаков и
5 флагов пропусков). Боевые модели не меняются: их прогнозы здесь пересчитываются теми же
функциями (`econ_long.forecast_asof`, логрегрессия `econ_kr`) как эталон `ref`, а рядом — те
же модели с добавленными наборами:
    ref        эталон Этапа 2: боевой набор, C фиксирована (годовая 1,0; месячная 0,3);
    base       тот же набор, но C выбирается внутри обучения — чтобы отделить вклад
               признаков от вклада настройки регуляризации;
    regime / reserves / reer / ctot   эталонный набор + одно семейство (C выбирается);
    struct     эталонный набор + все четыре семейства (C выбирается).
C выбирается по времени внутри обучающего окна (`choose_C`): последние 5 лет обучения —
проверка, тест в выборе не участвует.

Годовой контур: логрегрессия и деревья (`ExtraTrees`), обучение с 1957 года, прогноз «на
1 января», T = 1996…2023. Месячный: логрегрессия на индикаторах KLR, обучение — метки,
известные к концу декабря T−1, тест — все месяцы T, T = 1996…2025.

Выход:
    data/processed/fx_struct_monthly.parquet            таблица структурных признаков
    data/processed/econ_long_struct_predictions.parquet прогнозы годового контура по наборам
    data/processed/econ_kr_struct_predictions.parquet   прогнозы месячного контура по наборам
    reports/fx_struct_coverage.csv                      покрытие признаков по постсоветским странам
    reports/fx_struct_C.csv                             какая C выбиралась по годам
    reports/fx_struct_coefficients.csv                  коэффициенты набора `struct` в обоих контурах
Парное сравнение с интервалами и кейсы — команда `compare_fx`.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, FLOAT_FORMAT, REPORTS
from seldon.core.entities import ISO3_TO_SHORT
from seldon.core.temporal import label_known
from seldon.domains.currency import econ_kr, econ_long
from seldon.domains.currency.kr_features import INDICATORS, load_war
from seldon.domains.currency.long_features import BASE, MISSING
from seldon.domains.currency.struct_features import COLUMNS, SETS, StructLags, annual_view, attach, load_struct
from seldon.evaluation.backtest import backtest
from seldon.sources.gmd import load_gmd
from seldon.sources.imf import load_dec_fx
from seldon.sources.wdi import load_annual_cpi

VARIANTS = ["base", *SETS]  # всё, что сравнивается с эталоном `ref`
KR_TARGET = "deval15"
KR_C_REF = 0.3  # боевая логрегрессия месячного контура
KR_C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
KR_VAL_MONTHS = 60
KR_MIN_POS = 5


def _extra(name: str) -> tuple[list[str], list[str]]:
    """Добавочные признаки набора: (числовые, флаги). `ref` и `base` — без добавок."""
    return SETS.get(name, ([], []))


# ---------------------------------------------------------------------------
# Годовой контур
# ---------------------------------------------------------------------------


def build_long_panel(struct: pd.DataFrame, gmd=None, dec_fx=None, lv=None) -> pd.DataFrame:
    return econ_long.build_panel(load_gmd() if gmd is None else gmd, load_dec_fx() if dec_fx is None else dec_fx,
                                 econ_long.load_lv() if lv is None else lv, "deval25", struct=annual_view(struct))


def forecast_asof_long(p: pd.DataFrame, T: int, name: str = "struct", n_estimators: int = 400) -> pd.DataFrame:
    """Прогноз годового контура из точки T с набором `name` (`ref` — боевая модель как есть).
    Колонки `logit`, `extree`, для наборов с выбором C — ещё `C`."""
    if name == "ref":
        return econ_long.forecast_asof(p, T, n_estimators=n_estimators)
    cont, flags = _extra(name)
    return econ_long.forecast_asof(p, T, n_estimators=n_estimators, cont=[*BASE, *cont], flags=[*MISSING, *flags], C=None)


def run_long(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    chosen = []

    def step(T: int) -> pd.DataFrame | None:
        ref = forecast_asof_long(p, T, "ref")
        out = ref[ref.evaluable].rename(columns={"logit": "logit__ref", "extree": "extree__ref"})
        if out.empty or out.logit__ref.isna().all():
            return None
        for name in VARIANTS:
            f = forecast_asof_long(p, T, name)
            out = out.join(f[["logit", "extree"]].rename(columns={"logit": f"logit__{name}", "extree": f"extree__{name}"}))
            chosen.append({"contour": "long", "year": T, "set": name, "C": float(f.C.iloc[0])})
        print(f"  {T}: строк {len(out)}, кризисных окон {int(out.y.sum())}")
        return out

    last_eval = int(p[p.evaluable].year.max())
    return backtest(range(econ_long.EVAL_FROM, last_eval + 1), step), pd.DataFrame(chosen)


# ---------------------------------------------------------------------------
# Месячный контур
# ---------------------------------------------------------------------------


def build_kr_panel(struct: pd.DataFrame, raw=None, annual_cpi=None, war=None, target: str = KR_TARGET) -> pd.DataFrame:
    p = econ_kr.build_panel(econ_kr.load_panel() if raw is None else raw, load_annual_cpi() if annual_cpi is None else annual_cpi,
                            load_war() if war is None else war, target=target)
    return attach(p, struct, ["iso3", "month"])


def _kr_design(d: pd.DataFrame, feats: list[str], lo: pd.Series, hi: pd.Series, med: pd.Series) -> pd.DataFrame:
    return d[feats].clip(lo, hi, axis=1).fillna(med)


def _kr_fit(train: pd.DataFrame, feats: list[str], C: float):
    tr = train[feats]
    lo, hi, med = tr.quantile(0.01), tr.quantile(0.99), tr.median().fillna(0.0)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(_kr_design(train, feats, lo, hi, med), train.y)
    return m, lo, hi, med


def choose_C_kr(known: pd.DataFrame, feats: list[str], grid: tuple[float, ...] = KR_C_GRID, h: int = econ_kr.H) -> float:
    """C по времени внутри обучения: последние `KR_VAL_MONTHS` месяцев обучения — проверка,
    учимся на окнах, закрывшихся до её начала (месяц + h ≤ граница)."""
    cut = known.month.max() - KR_VAL_MONTHS
    fit, val = known[label_known(known.month, cut, h)], known[known.month > cut]
    if fit.y.sum() < KR_MIN_POS or val.y.nunique() < 2:
        return KR_C_REF
    loss = {}
    for C in grid:
        m, lo, hi, med = _kr_fit(fit, feats, C)
        loss[C] = log_loss(val.y, m.predict_proba(_kr_design(val, feats, lo, hi, med))[:, 1], labels=[0, 1])
    return min(loss, key=loss.get)


def _kr_predict(p: pd.DataFrame, as_of: pd.Period, test: pd.DataFrame, name: str) -> tuple[np.ndarray, float] | None:
    cont, flags = _extra(name)
    feats = [*INDICATORS, *cont, *flags]
    known = p[label_known(p.month, as_of, econ_kr.H) & p.evaluable]
    if known.y.sum() < KR_MIN_POS:
        return None
    C = KR_C_REF if name == "ref" else choose_C_kr(known, feats)
    m, lo, hi, med = _kr_fit(known, feats, C)
    return m.predict_proba(_kr_design(test, feats, lo, hi, med))[:, 1], C


def forecast_asof_kr(p: pd.DataFrame, as_of: pd.Period, name: str = "struct") -> pd.DataFrame:
    """Прогноз месячного контура в точке `as_of` с набором `name`: всё — по данным ≤ as_of.
    Колонка `logit`; `ref` совпадает с логрегрессией `econ_kr.forecast_asof`."""
    test = p[(p.month == as_of) & p.own_currency & p.fx.notna()].copy()
    out = _kr_predict(p, as_of, test, name)
    test["logit"], test["C"] = (np.nan, np.nan) if out is None else out
    return test


def run_kr(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    chosen = []

    def step(T: int) -> pd.DataFrame | None:
        as_of = pd.Period(f"{T - 1}-12", freq="M")
        test = p[(p.month.dt.year == T) & p.evaluable].copy()
        if test.empty:
            return None
        for name in ("ref", *VARIANTS):
            out = _kr_predict(p, as_of, test, name)
            if out is None:
                return None
            test[f"logit__{name}"] = out[0]
            if name != "ref":
                chosen.append({"contour": "kr", "year": T, "set": name, "C": out[1]})
        print(f"  {T}: строк {len(test)}, кризисных окон {int(test.y.sum())}")
        return test

    return backtest(range(econ_kr.FIRST_EVAL_YEAR, econ_kr.LAST_EVAL_YEAR + 1), step), pd.DataFrame(chosen)


# ---------------------------------------------------------------------------


def coefficients(p_long: pd.DataFrame, p_kr: pd.DataFrame) -> pd.DataFrame:
    """Стандартизованные коэффициенты логрегрессии набора `struct` в последней точке обучения
    каждого контура — чтобы видеть знак и вес каждого признака."""
    cont, flags = SETS["struct"]
    T = int(p_long[p_long.evaluable].year.max())
    _, models = econ_long.forecast_asof(p_long, T, cont=[*BASE, *cont], flags=[*MISSING, *flags], C=None, return_models=True, n_estimators=10)
    long = pd.Series(models["logit"][-1].coef_[0], index=[*BASE, *cont, *MISSING, *flags], name="long")
    feats = [*INDICATORS, *cont, *flags]
    as_of = pd.Period(f"{econ_kr.LAST_EVAL_YEAR - 1}-12", freq="M")
    known = p_kr[label_known(p_kr.month, as_of, econ_kr.H) & p_kr.evaluable]
    m, *_ = _kr_fit(known, feats, choose_C_kr(known, feats))
    kr = pd.Series(m[-1].coef_[0], index=feats, name="kr")
    out = pd.concat([long, kr], axis=1)
    out["struct"] = out.index.isin(COLUMNS)
    return out.rename_axis("feature").reset_index()


def coverage(struct: pd.DataFrame) -> pd.DataFrame:
    """С какого месяца у постсоветской страны есть каждый признак (и доля покрытия с 1996)."""
    rows = []
    for iso, short in ISO3_TO_SHORT.items():
        g = struct[struct.iso3 == iso]
        r = {"country": short, "iso3": iso}
        for src, flag in (("irr_peg", "irr_missing"), ("gg_log", "gg_missing"), ("res_months_log", "resm_missing"),
                          ("reer_gap", "reer_missing"), ("ctot_d12", "ctot_missing")):
            have = g[g[flag] == 0]
            r[f"{src}_from"] = str(have.month.min()) if len(have) else ""
            r[f"{src}_to"] = str(have.month.max()) if len(have) else ""
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    contours = [a for a in argv if a in ("long", "kr")] or ["long", "kr"]
    rt = "ctot_rt" in argv
    lags = replace(StructLags(), ctot_months=1) if rt else StructLags()
    sfx = "_ctot_rt" if rt else ""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"Структурные признаки (лаги: REER {lags.reer_months} мес, условия торговли {lags.ctot_months} мес)…")
    struct = load_struct(lags)
    struct.assign(month=struct.month.astype(str)).to_parquet(DATA_PROCESSED / f"fx_struct_monthly{sfx}.parquet", index=False)
    d = struct[struct.month >= "1996-01"]
    print(f"  {struct.iso3.nunique()} стран, {len(struct)} страна-месяцев; доля строк с 1996 года, где признак есть:")
    print("  " + ", ".join(f"{f.replace('_missing', '')} {1 - d[f].mean():.0%}" for f in COLUMNS if f.endswith("_missing")))
    chosen, panels = [], {}
    if not rt:
        cov = coverage(struct)
        cov.to_csv(REPORTS / "fx_struct_coverage.csv", index=False)
        print(cov[["country", "irr_peg_from", "gg_log_from", "gg_log_to", "reer_gap_from", "ctot_d12_from"]].to_string(index=False))
    if "long" in contours:
        print("\nГодовой контур (deval25), вслепую…")
        panels["long"] = build_long_panel(struct)
        pred, c = run_long(panels["long"])
        pred.to_parquet(DATA_PROCESSED / f"econ_long_struct_predictions{sfx}.parquet", index=False)
        chosen.append(c)
    if "kr" in contours:
        print("\nМесячный контур (deval15), вслепую…")
        panels["kr"] = build_kr_panel(struct)
        pred, c = run_kr(panels["kr"])
        pred.assign(month=pred.month.astype(str)).to_parquet(DATA_PROCESSED / f"econ_kr_struct_predictions{sfx}.parquet", index=False)
        chosen.append(c)
    c = pd.concat(chosen, ignore_index=True)
    if not rt and len(contours) == 2:
        c.to_csv(REPORTS / "fx_struct_C.csv", index=False, float_format=FLOAT_FORMAT)
        coef = coefficients(panels["long"], panels["kr"])
        coef.to_csv(REPORTS / "fx_struct_coefficients.csv", index=False, float_format=FLOAT_FORMAT)
        print("\nКоэффициенты набора struct (стандартизованные, последняя точка обучения):")
        print(coef[coef.struct].drop(columns="struct").round(3).to_string(index=False))
    print("\nКакая C выбиралась (число лет):")
    print(c.groupby(["contour", "set"]).C.value_counts().unstack(fill_value=0).to_string())
    return 0
