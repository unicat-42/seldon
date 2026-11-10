"""Валютные кризисы на длинной истории: GMD 1870–2024, все страны, по годам.

Запуск:
    uv run python -m seldon.econ_long [deval25|gmd_cc]

Идея: за 1996–2024 девальваций ≥ 25 % около 110, за 1870–2024 — около 630. Модель на
150 годах видит сотни эпизодов вместо десятков (так сработала модель BoE на JST). Проверка
честная: учимся на годах ≤ T−2, предсказываем T; и главный тест — то же самое, но обучение
только на 1970+ (`short`): если длинная история не помогает, разница будет нулевой.

Целевые:
    deval25  — курс к доллару (среднегодовой) вырос ≥ 25 % за год после ≥ 2 спокойных лет
               (как в econ_panel); годы с инфляцией > 100 % в прошлом году не оцениваем
               (гиперинфляция: девальвация там — не событие, а фон);
    gmd_cc   — валютный кризис по хронике GMD (Reinhart–Rogoff, Laeven–Valencia; до 2017).
Метка: y(t) = 1, если кризис в t+1 или t+2 (как у BoE). Год кризиса и 2 после — не оцениваем.

Признаки (год t, всё с лагом — известно к концу t):
    infl_log        ln(1 + инфляция/100)
    infl_d2         изменение инфляции за 2 года (п.п./100)
    deval_prev      девальвация за год t (доля, обрезана)
    deval_5         число девальваций ≥ 25 % за t−4…t
    gdp_growth_2    средний рост реального ВВП за 2 года
    debt_d2         изменение госдолг/ВВП за 2 года
    ca_level        счёт текущих операций / ВВП
    money_growth_2  рост денежной массы (M2, иначе M3, M0) за 2 года минус рост ВВП
    world_deval     доля стран мира с девальвацией ≥ 25 % в году t (заражение)
    peg_years_log   ln(1 + лет без изменения курса > 2 %) — привязка копит давление
    *_missing       флаги пропусков (признак тогда = 0)

Модели вслепую по годам T:
    base_rate       доля положительных в обучении
    persist         девальвация была в t или t−1 → доля положительных в обучении при этом
    logit_long / extree_long     обучение на всех годах ≤ T−2 (с 1870)
    logit_short / extree_short   обучение только на годах 1970…T−2
Оценка: T = 1950–2024 (длинное окно) и 1996–2024 (как в других блоках), мир и постсовет.

Выход:
    data/processed/econ_long_panel.parquet, econ_long_predictions[_gmd_cc].parquet
    reports/econ_long_metrics[_gmd_cc].csv, econ_long_metrics_by_year[_gmd_cc].csv,
    econ_long_pre_crisis_table[_gmd_cc].csv, econ_long_coefficients[_gmd_cc].csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, DATA_RAW, PROJECT_ROOT
from seldon.countries import ISO3_TO_SHORT, POST_SOVIET_ISO3
from seldon.download import GMD_VERSION

REPORTS = PROJECT_ROOT / "reports"
FIRST_YEAR, LAST_YEAR = 1870, 2024
SHORT_FROM = 1970
EVAL_FROM = 1950
H = 2
POST_CRISIS_DROP = 2
DEVAL = 0.25
HYPER = 100.0  # инфляция, %
MIN_POS_TRAIN = 30

BASE = ["infl_log", "infl_d2", "deval_prev", "deval_5", "gdp_growth_2", "debt_d2", "ca_level",
        "money_growth_2", "world_deval", "peg_years_log"]
MISSING = ["infl_missing", "gdp_missing", "debt_missing", "ca_missing", "money_missing"]
FEATURES = [*BASE, *MISSING]


def load_panel() -> pd.DataFrame:
    g = pd.read_stata(DATA_RAW / "gmd" / f"GMD_{GMD_VERSION}.dta")
    g = g[(g.year >= FIRST_YEAR - 10) & (g.year <= LAST_YEAR)].rename(columns={"ISO3": "iso3", "countryname": "country"})
    g["year"] = g.year.astype(int)
    g = g[g.USDfx.notna() & (g.USDfx > 0)].sort_values(["iso3", "year"]).reset_index(drop=True)
    # полная сетка лет внутри страны, чтобы лаги шли по календарю
    idx = pd.MultiIndex.from_tuples(
        [(c, y) for c, r in g.groupby("iso3").year.agg(["min", "max"]).iterrows() for y in range(r["min"], r["max"] + 1)],
        names=["iso3", "year"])
    g = g.set_index(["iso3", "year"]).reindex(idx).reset_index()
    g["country"] = g.groupby("iso3").country.transform(lambda s: s.ffill().bfill())
    g["post_soviet"] = g.iso3.isin(POST_SOVIET_ISO3)
    return g


def add_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    gb = p.groupby("iso3")
    p["deval"] = gb.USDfx.pct_change()
    p["big"] = (p.deval >= DEVAL).astype(int)
    prev = gb.big.shift(1).fillna(0) + gb.big.shift(2).fillna(0)
    p["deval25"] = ((p.big == 1) & (prev == 0)).astype(int)
    p.loc[p.deval.isna(), "deval25"] = np.nan
    p["gmd_cc"] = p.CurrencyCrisis
    p["hyper_prev"] = gb.infl.shift(1) > HYPER

    p["infl_log"] = np.log1p(p.infl.clip(lower=-50, upper=10_000) / 100.0)
    p["infl_d2"] = (p.infl - gb.infl.shift(2)).clip(-500, 500) / 100.0
    p["deval_prev"] = p.deval.clip(-0.9, 5)
    p["deval_5"] = gb.big.transform(lambda s: s.rolling(5, min_periods=1).sum())
    p["gdp_growth_2"] = ((p.rGDP / gb.rGDP.shift(2)) ** 0.5 - 1).clip(-0.5, 0.5)
    p["debt_d2"] = (gb.govdebt_GDP.diff(2) / 100.0).clip(-2, 2)
    p["ca_level"] = (p.CA_GDP / 100.0).clip(-0.5, 0.5)
    money = p.M2.fillna(p.M3).fillna(p.M0)
    p["money_growth_2"] = (((money / money.groupby(p.iso3).shift(2)) ** 0.5 - 1) - (p.nGDP / gb.nGDP.shift(2)) ** 0.5 + 1).clip(-1, 2)
    world = p[p.deval.notna()].groupby("year").big.mean()
    p["world_deval"] = p.year.map(world)
    stable = (p.deval.abs() <= 0.02).astype(int)
    run = stable.groupby(p.iso3).transform(lambda s: s.groupby((s == 0).cumsum()).cumsum())
    p["peg_years_log"] = np.log1p(run)
    for src, flag in [("infl_log", "infl_missing"), ("gdp_growth_2", "gdp_missing"), ("debt_d2", "debt_missing"),
                      ("ca_level", "ca_missing"), ("money_growth_2", "money_missing")]:
        p[flag] = p[src].isna().astype(int)
        p[src] = p[src].fillna(0.0)
    for c in ("infl_d2", "deval_prev", "world_deval"):
        p[c] = p[c].fillna(0.0)
    return p


def add_target(p: pd.DataFrame, target: str) -> pd.DataFrame:
    p = p.copy()
    gb = p.groupby("iso3")[target]
    nxt, nxt2 = gb.shift(-1), gb.shift(-2)
    p["y"] = np.where(nxt.isna() & nxt2.isna(), np.nan, ((nxt == 1) | (nxt2 == 1)).astype(float))
    drop = p[target].fillna(0).astype(bool)
    for j in range(1, POST_CRISIS_DROP + 1):
        drop |= gb.shift(j).fillna(0).astype(bool)
    p["evaluable"] = ~drop & p.y.notna() & ~p.hyper_prev.fillna(False) & (p.year >= FIRST_YEAR) & p.deval.notna()
    return p


def make_models(seed: int = 0) -> dict:
    return {
        "logit": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
        "extree": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=10, random_state=seed, n_jobs=-1),
    }


def run_models(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = p[p.evaluable]
    preds, coefs = [], []
    for T in range(EVAL_FROM, LAST_YEAR + 1):
        test = base[base.year == T].copy()
        if test.empty:
            continue
        for tag, start in (("long", FIRST_YEAR), ("short", SHORT_FROM)):
            train = base[(base.year <= T - H) & (base.year >= start)]
            if train.y.sum() < MIN_POS_TRAIN:
                for k in ("logit", "extree"):
                    test[f"{k}_{tag}"] = np.nan
                continue
            lo, hi = train[BASE].quantile(0.01), train[BASE].quantile(0.99)
            Xtr = pd.concat([train[BASE].clip(lo, hi, axis=1), train[MISSING]], axis=1).values
            Xte = pd.concat([test[BASE].clip(lo, hi, axis=1), test[MISSING]], axis=1).values
            for k, m in make_models().items():
                m.fit(Xtr, train.y.astype(int))
                test[f"{k}_{tag}"] = m.predict_proba(Xte)[:, 1]
                if k == "logit":
                    coefs.append(pd.Series(m[-1].coef_[0], index=FEATURES, name=f"{tag}_{T}"))
            if tag == "long":
                test["base_rate"] = train.y.mean()
                recent_tr = (train.deval_5 > 0).astype(int)
                rate = train.groupby(recent_tr).y.mean()
                test["persist"] = (test.deval_5 > 0).astype(int).map(rate).fillna(train.y.mean()).to_numpy()
        preds.append(test)
        if T % 10 == 0:
            print(f"  {T}: обучение long {int(base[(base.year <= T - H)].y.sum())} полож., "
                  f"short {int(base[(base.year <= T - H) & (base.year >= SHORT_FROM)].y.sum())}, тест {len(test)} стран")
    return pd.concat(preds, ignore_index=True), pd.DataFrame(coefs)


MODELS = ["base_rate", "persist", "logit_short", "extree_short", "logit_long", "extree_long"]


def _auc(y, s):
    return roc_auc_score(y, s) if (y.nunique() == 2 and s.nunique() > 1) else np.nan


def metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, lo in (("1950–2024", 1950), ("1996–2024", 1996)):
        d = pred[pred.year >= lo]
        for scope, sub in (("world", d), ("post_soviet", d[d.post_soviet])):
            for m in MODELS:
                s = sub[sub[m].notna()]
                if s.empty or s.y.nunique() < 2:
                    continue
                yearly = [_auc(g.y, g[m]) for _, g in s.groupby("year")]
                rows.append({"window": window, "scope": scope, "model": m, "n": len(s), "positives": int(s.y.sum()),
                             "auc": _auc(s.y, s[m]), "auc_within_year": np.nanmean(yearly),
                             "brier": brier_score_loss(s.y, s[m])})
    return pd.DataFrame(rows)


def metrics_by_year(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for T, d in pred.groupby("year"):
        row = {"year": T, "n": len(d), "positives": int(d.y.sum())}
        for m in MODELS:
            s = d[d[m].notna()]
            row[f"{m}_auc"] = _auc(s.y, s[m]) if len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pre_crisis_table(p: pd.DataFrame, pred: pd.DataFrame, target: str) -> pd.DataFrame:
    """Постсоветские кризисы с 1996: прогноз за год до (признаки t−1) и место среди стран."""
    rows = []
    ev = p[(p[target] == 1) & p.post_soviet & (p.year >= 1996)]
    for _, e in ev.iterrows():
        snap = pred[pred.year == e.year - 1]
        me = snap[snap.iso3 == e.iso3]
        row = {"country": ISO3_TO_SHORT[e.iso3], "crisis_year": int(e.year), "deval_pct": round(100 * e.deval, 0) if pd.notna(e.deval) else np.nan,
               "n_countries": len(snap)}
        for m in ("persist", "logit_short", "logit_long", "extree_short", "extree_long"):
            if me.empty or pd.isna(me[m].iloc[0]):
                row[f"{m}_rank"] = np.nan
            else:
                v = me[m].iloc[0]
                row[f"{m}_rank"] = int((snap[m] > v).sum() + 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    target = argv[0] if argv else "deval25"
    suffix = "" if target == "deval25" else f"_{target}"
    p = add_target(add_features(load_panel()), target)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    p.to_parquet(DATA_PROCESSED / "econ_long_panel.parquet", index=False)
    ev = p[p.evaluable]
    print(f"панель: {p.iso3.nunique()} стран, {p.year.min()}–{p.year.max()}; оцениваемых страна-лет {len(ev)}, "
          f"положительных {int(ev.y.sum())} ({target}); до 1970: {int(ev[ev.year < 1970].y.sum())}, с 1970: {int(ev[ev.year >= 1970].y.sum())}")
    print("\nМодели вслепую…")
    pred, coefs = run_models(p)
    pred.to_parquet(DATA_PROCESSED / f"econ_long_predictions{suffix}.parquet", index=False)
    coefs.to_csv(REPORTS / f"econ_long_coefficients{suffix}.csv")
    met = metrics(pred)
    met.to_csv(REPORTS / f"econ_long_metrics{suffix}.csv", index=False)
    print(f"\nМетрики, целевая {target} в t+1/t+2:")
    print(met.round(3).to_string(index=False))
    by = metrics_by_year(pred)
    by.to_csv(REPORTS / f"econ_long_metrics_by_year{suffix}.csv", index=False)
    ok = by[by.logit_long_auc.notna() & by.logit_short_auc.notna()]
    print(f"\n  long лучше short по AUC внутри года: логит в {int((ok.logit_long_auc > ok.logit_short_auc).sum())} из {len(ok)} лет, "
          f"деревья в {int((ok.extree_long_auc > ok.extree_short_auc).sum())}")
    last = coefs[coefs.index.str.startswith("long_")].iloc[-1]
    print("\nКоэффициенты logit_long (последняя модель, стандартизованные):")
    print(last.sort_values(key=abs, ascending=False).round(3).to_string())
    tab = pre_crisis_table(p, pred, target)
    tab.to_csv(REPORTS / f"econ_long_pre_crisis_table{suffix}.csv", index=False)
    print("\nПостсоветские кризисы: место среди стран за год до (1 = самый рискованный):")
    print(tab.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
