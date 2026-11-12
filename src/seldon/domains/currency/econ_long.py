"""Валютные кризисы по годам: декабрьский курс МВФ 1957–2025 + макроданные GMD, все страны.

Запуск:
    uv run python -m seldon econ_long [deval25|gmd_cc|lv_cc] [all_currencies]

После аудита 12.11.2026 всё, что касается курса, считается по курсу НА КОНЕЦ ДЕКАБРЯ
(МВФ, `EOP_RT`): девальвация года t = Dec(t)/Dec(t−1) − 1. Раньше брался среднегодовой курс
GMD — и обвал в конце года T (Россия-2014: среднее +21 %, декабрь к декабрю +72 %) попадал
в «кризис T+1», который модель «предсказывала», уже видя его в признаках года T.
Декабрьский курс есть с 1957 года, поэтому история 1870–1956 в блок больше не входит.

Страны с общей или чужой валютой (еврозона, франк КФА, доллар — `core.entities.SHARED_CURRENCY`)
исключаются из обучения, оценки и прогноза строго с года перехода: у Эстонии с 2011 года нет
своего риска девальвации, а до 2011 — есть. Аргумент `all_currencies` отключает исключение
(срез для сравнения, файлы с суффиксом `_allcur`).

Целевые:
    deval25  — декабрьский курс к доллару вырос ≥ 25 % за год после ≥ 2 спокойных лет;
               годы с инфляцией > 100 % в прошлом году не оцениваем (гиперинфляция);
    gmd_cc   — валютный кризис по хронике GMD (Reinhart–Rogoff, Laeven–Valencia); хроника
               обрезана 2016 годом — дальше в GMD стоят нули вместо пропусков;
    lv_cc    — валютный кризис по Laeven–Valencia 2018 напрямую (`sources/lv.py`; 1970–2017).
Метка: y(t) = 1, если кризис в t+1 или t+2 (как у BoE); известна, только когда известны
оба года (или кризис уже случился в t+1). Год кризиса и 2 после — не оцениваем; для deval25
ещё и годы, когда большая девальвация идёт в t или t−1 (в t+1 она не может «начаться»).

Признаки (год t; курсовые — по декабрю t, остальные — годовые данные GMD за t):
    infl_log        ln(1 + инфляция/100)
    infl_d2         изменение инфляции за 2 года (п.п./100)
    deval_prev      девальвация за год t, декабрь к декабрю (доля, обрезана)
    deval_5         число девальваций ≥ 25 % за t−4…t
    gdp_growth_2    средний рост реального ВВП за 2 года
    debt_d2         изменение госдолг/ВВП за 2 года
    ca_level        счёт текущих операций / ВВП
    money_growth_2  рост денежной массы (M2, иначе M3, M0) за 2 года минус рост ВВП
    world_deval     доля стран мира (со своей валютой) с девальвацией ≥ 25 % в году t
    peg_years_log   ln(1 + лет без изменения декабрьского курса > 2 %) — привязка копит давление
    *_missing       флаги пропусков (признак тогда = 0)
Оговорка: годовые данные GMD за год t считаются известными к концу t. На деле ВВП и счёт
текущих операций публикуются с лагом в несколько месяцев — это закрывается на Этапе 2
(`available_from`).

Модели вслепую по годам T (обучение — строки с известной меткой, год ≤ T−2):
    base_rate       доля положительных в обучении
    persist         девальвация была в последние 5 лет → доля положительных в обучении при этом
    logit_long / extree_long     обучение на всех годах с 1957
    logit_short / extree_short   обучение только на годах с 1980
Оценка: T = 1996–2023, одни и те же строки для long и short, мир и постсовет.

Выход:
    data/processed/econ_long_panel.parquet, econ_long_predictions[_<целевая>][_allcur].parquet
    reports/econ_long_metrics*.csv, econ_long_metrics_by_year*.csv,
    econ_long_pre_crisis_table*.csv, econ_long_coefficients*.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, REPORTS
from seldon.core.entities import ISO3_TO_SHORT, POST_SOVIET_ISO3, shared_currency_table
from seldon.core.metrics import auc, auc_within, brier
from seldon.core.panel import full_grid
from seldon.core.temporal import label_known
from seldon.domains.currency.long_features import BASE, FEATURES, MISSING, add_features
from seldon.domains.currency.long_targets import FIRST_YEAR, H, LAST_YEAR, MIN_POS_TRAIN, add_events, add_target
from seldon.evaluation.backtest import backtest
from seldon.sources.gmd import load_gmd
from seldon.sources.imf import load_dec_fx
from seldon.sources.lv import load_lv_country_year

SHORT_FROM = 1980
EVAL_FROM = 1996


def load_lv() -> pd.DataFrame:
    return load_lv_country_year()[["iso3", "year", "lv_currency"]]


def build_panel(gmd: pd.DataFrame, dec_fx: pd.DataFrame, lv: pd.DataFrame, target: str = "deval25",
                last_year: int = LAST_YEAR, drop_shared: bool = True) -> pd.DataFrame:
    """Панель страна×год с признаками и целевой. Чистая функция: признаки года t зависят
    только от данных ≤ t (кроме `y` — это будущее)."""
    g = gmd.rename(columns={"ISO3": "iso3", "countryname": "country"})
    g = g[(g.year >= FIRST_YEAR - 10) & (g.year <= last_year)].copy()
    g["year"] = g.year.astype(int)
    d = dec_fx[(dec_fx.year <= last_year) & dec_fx.fx_dec.notna()]
    # полная сетка лет внутри страны (от первого до последнего декабрьского курса) — лаги по календарю
    idx = full_grid(d.groupby("iso3").year.agg(["min", "max"]), names=["iso3", "year"])
    p = pd.DataFrame(index=idx).reset_index().merge(d, on=["iso3", "year"], how="left").merge(g, on=["iso3", "year"], how="left")
    p["country"] = p.groupby("iso3").country.transform(lambda s: s.ffill().bfill()).fillna(p.iso3)
    p["post_soviet"] = p.iso3.isin(POST_SOVIET_ISO3)
    p["own_currency"] = True
    if drop_shared:
        p = p.merge(shared_currency_table(int(p.year.min()), int(p.year.max())), on=["iso3", "year"], how="left")
        p["own_currency"] = p.fx_regime.isna()
    p = p.merge(lv, on=["iso3", "year"], how="left").rename(columns={"lv_currency": "lv_cc"})
    return add_target(add_features(add_events(p.sort_values(["iso3", "year"]).reset_index(drop=True))), target)


def make_models(seed: int = 0, n_estimators: int = 400) -> dict:
    return {
        "logit": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
        "extree": ExtraTreesClassifier(n_estimators=n_estimators, min_samples_leaf=10, random_state=seed, n_jobs=-1),
    }


def _design(d: pd.DataFrame, lo: pd.Series, hi: pd.Series) -> np.ndarray:
    return pd.concat([d[BASE].clip(lo, hi, axis=1), d[MISSING]], axis=1).values


def forecast_asof(p: pd.DataFrame, T: int, start: int = FIRST_YEAR, train_through: int | None = None,
                  n_estimators: int = 400, return_models: bool = False):
    """Прогноз с признаков года T на T+1…T+2 для всех стран со своей валютой.
    Обучение — оцениваемые строки `start` ≤ год ≤ `train_through` (по умолчанию T−2: метка
    строки t смотрит в t+2, позже T−2 она в момент T ещё не известна)."""
    known = label_known(p.year, T, H) if train_through is None else (p.year <= train_through)
    train = p[p.evaluable & known & (p.year >= start)]
    test = p[(p.year == T) & p.usable].copy()
    models = {}
    for k in ("base_rate", "persist", "logit", "extree"):
        test[k] = np.nan
    if train.y.sum() >= MIN_POS_TRAIN and not test.empty:
        lo, hi = train[BASE].quantile(0.01), train[BASE].quantile(0.99)
        models = make_models(n_estimators=n_estimators)
        for k, m in models.items():
            m.fit(_design(train, lo, hi), train.y.astype(int))
            test[k] = m.predict_proba(_design(test, lo, hi))[:, 1]
        test["base_rate"] = train.y.mean()
        rate = train.groupby((train.deval_5 > 0).astype(int)).y.mean()
        test["persist"] = (test.deval_5 > 0).astype(int).map(rate).fillna(train.y.mean()).to_numpy()
    return (test, models) if return_models else test


def run_models(p: pd.DataFrame, last_eval: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefs = []

    def step(T: int) -> pd.DataFrame | None:
        long, m_long = forecast_asof(p, T, FIRST_YEAR, return_models=True)
        short, m_short = forecast_asof(p, T, SHORT_FROM, return_models=True)
        test = long[long.evaluable].rename(columns={"logit": "logit_long", "extree": "extree_long"})
        if test.empty:
            return None
        test = test.join(short[["logit", "extree"]].rename(columns={"logit": "logit_short", "extree": "extree_short"}))
        for tag, m in (("long", m_long), ("short", m_short)):
            if m:
                coefs.append(pd.Series(m["logit"][-1].coef_[0], index=FEATURES, name=f"{tag}_{T}"))
        return test

    pred = backtest(range(EVAL_FROM, last_eval + 1), step)
    return pred, pd.DataFrame(coefs)


MODELS = ["base_rate", "persist", "logit_short", "extree_short", "logit_long", "extree_long"]


def metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # одни и те же строки для всех моделей: только там, где есть и long, и short
    pred = pred[pred.extree_long.notna() & pred.extree_short.notna()]
    for window, lo in ((f"{EVAL_FROM}–{int(pred.year.max())}", EVAL_FROM), (f"2008–{int(pred.year.max())}", 2008)):
        d = pred[pred.year >= lo]
        for scope, sub in (("world", d), ("post_soviet", d[d.post_soviet])):
            for m in MODELS:
                s = sub[sub[m].notna()]
                if s.empty or s.y.nunique() < 2:
                    continue
                rows.append({"window": window, "scope": scope, "model": m, "n": len(s), "positives": int(s.y.sum()),
                             "auc": auc(s.y, s[m]), "auc_within_year": auc_within(s.y, s[m], s.year),
                             "brier": brier(s.y, s[m])})
    return pd.DataFrame(rows)


def metrics_by_year(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for T, d in pred.groupby("year"):
        row = {"year": T, "n": len(d), "positives": int(d.y.sum())}
        for m in MODELS:
            s = d[d[m].notna()]
            row[f"{m}_auc"] = auc(s.y, s[m]) if len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pre_crisis_table(p: pd.DataFrame, pred: pd.DataFrame, target: str) -> pd.DataFrame:
    """Постсоветские кризисы с 1996: прогноз за год до (признаки t−1) и место среди стран."""
    rows = []
    ev = p[(p[target] == 1) & p.post_soviet & (p.year > EVAL_FROM)]
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
    target = next((a for a in argv if a in ("deval25", "gmd_cc", "lv_cc")), "deval25")
    drop_shared = "all_currencies" not in argv
    suffix = ("" if target == "deval25" else f"_{target}") + ("" if drop_shared else "_allcur")
    p = build_panel(load_gmd(), load_dec_fx(), load_lv(), target, drop_shared=drop_shared)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    p.to_parquet(DATA_PROCESSED / f"econ_long_panel{suffix}.parquet", index=False)
    ev = p[p.evaluable]
    last_eval = int(ev.year.max())
    print(f"панель: {p.iso3.nunique()} стран, {p.year.min()}–{p.year.max()}; оцениваемых страна-лет {len(ev)}, "
          f"положительных {int(ev.y.sum())} ({target}); до {SHORT_FROM}: {int(ev[ev.year < SHORT_FROM].y.sum())}, "
          f"с {SHORT_FROM}: {int(ev[ev.year >= SHORT_FROM].y.sum())}; метка известна по {last_eval}")
    print("\nМодели вслепую…")
    pred, coefs = run_models(p, last_eval)
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

