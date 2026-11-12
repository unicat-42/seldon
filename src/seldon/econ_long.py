"""Валютные кризисы по годам: декабрьский курс МВФ 1957–2025 + макроданные GMD, все страны.

Запуск:
    uv run python -m seldon.econ_long [deval25|gmd_cc|lv_cc] [all_currencies]

После аудита 12.11.2026 всё, что касается курса, считается по курсу НА КОНЕЦ ДЕКАБРЯ
(МВФ, `EOP_RT`): девальвация года t = Dec(t)/Dec(t−1) − 1. Раньше брался среднегодовой курс
GMD — и обвал в конце года T (Россия-2014: среднее +21 %, декабрь к декабрю +72 %) попадал
в «кризис T+1», который модель «предсказывала», уже видя его в признаках года T.
Декабрьский курс есть с 1957 года, поэтому история 1870–1956 в блок больше не входит.

Страны с общей или чужой валютой (еврозона, франк КФА, доллар — `countries.SHARED_CURRENCY`)
исключаются из обучения, оценки и прогноза строго с года перехода: у Эстонии с 2011 года нет
своего риска девальвации, а до 2011 — есть. Аргумент `all_currencies` отключает исключение
(срез для сравнения, файлы с суффиксом `_allcur`).

Целевые:
    deval25  — декабрьский курс к доллару вырос ≥ 25 % за год после ≥ 2 спокойных лет;
               годы с инфляцией > 100 % в прошлом году не оцениваем (гиперинфляция);
    gmd_cc   — валютный кризис по хронике GMD (Reinhart–Rogoff, Laeven–Valencia); хроника
               обрезана 2016 годом — дальше в GMD стоят нули вместо пропусков;
    lv_cc    — валютный кризис по Laeven–Valencia 2018 напрямую (lv.py; 1970–2017).
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

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, DATA_RAW, PROJECT_ROOT
from seldon.countries import ISO3_TO_SHORT, POST_SOVIET_ISO3, shared_currency_table
from seldon.download import GMD_VERSION

REPORTS = PROJECT_ROOT / "reports"
FIRST_YEAR, LAST_YEAR = 1957, 2025  # декабрьский курс МВФ: 1957-12 … 2025-12
SHORT_FROM = 1980
EVAL_FROM = 1996
H = 2
POST_CRISIS_DROP = 2
DEVAL = 0.25
HYPER = 100.0  # инфляция, %
MIN_POS_TRAIN = 30
# последний год, за который хроника кризисов реально размечена (дальше — нули вместо пропусков)
TARGET_LAST_YEAR = {"gmd_cc": 2016, "lv_cc": 2017}

BASE = ["infl_log", "infl_d2", "deval_prev", "deval_5", "gdp_growth_2", "debt_d2", "ca_level",
        "money_growth_2", "world_deval", "peg_years_log"]
MISSING = ["infl_missing", "gdp_missing", "debt_missing", "ca_missing", "money_missing"]
FEATURES = [*BASE, *MISSING]


def load_gmd() -> pd.DataFrame:
    return pd.read_stata(DATA_RAW / "gmd" / f"GMD_{GMD_VERSION}.dta")


def load_dec_fx() -> pd.DataFrame:
    """Курс к доллару на конец декабря (МВФ, EOP_RT): iso3, year, fx_dec."""
    from seldon.econ_kr import read_imf

    er = read_imf("exchange_rates")
    er = er[(er.TYPE_OF_TRANSFORMATION == "EOP_RT") & (er.month.dt.month == 12) & (er.value > 0)]
    er = er.assign(year=er.month.dt.year).groupby(["iso3", "year"], as_index=False).value.first()
    return er.rename(columns={"value": "fx_dec"})


def load_lv() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / "lv_country_year.parquet")[["iso3", "year", "lv_currency"]]


def build_panel(gmd: pd.DataFrame, dec_fx: pd.DataFrame, lv: pd.DataFrame, target: str = "deval25",
                last_year: int = LAST_YEAR, drop_shared: bool = True) -> pd.DataFrame:
    """Панель страна×год с признаками и целевой. Чистая функция: признаки года t зависят
    только от данных ≤ t (кроме `y` — это будущее)."""
    g = gmd.rename(columns={"ISO3": "iso3", "countryname": "country"})
    g = g[(g.year >= FIRST_YEAR - 10) & (g.year <= last_year)].copy()
    g["year"] = g.year.astype(int)
    d = dec_fx[(dec_fx.year <= last_year) & dec_fx.fx_dec.notna()]
    span = d.groupby("iso3").year.agg(["min", "max"])
    # полная сетка лет внутри страны (от первого до последнего декабрьского курса) — лаги по календарю
    idx = pd.MultiIndex.from_tuples([(c, y) for c, r in span.iterrows() for y in range(r["min"], r["max"] + 1)],
                                    names=["iso3", "year"])
    p = pd.DataFrame(index=idx).reset_index().merge(d, on=["iso3", "year"], how="left").merge(g, on=["iso3", "year"], how="left")
    p["country"] = p.groupby("iso3").country.transform(lambda s: s.ffill().bfill()).fillna(p.iso3)
    p["post_soviet"] = p.iso3.isin(POST_SOVIET_ISO3)
    p["own_currency"] = True
    if drop_shared:
        p = p.merge(shared_currency_table(int(p.year.min()), int(p.year.max())), on=["iso3", "year"], how="left")
        p["own_currency"] = p.fx_regime.isna()
    p = p.merge(lv, on=["iso3", "year"], how="left").rename(columns={"lv_currency": "lv_cc"})
    return add_target(add_features(p.sort_values(["iso3", "year"]).reset_index(drop=True)), target)


def add_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    gb = p.groupby("iso3")
    # девальвация: декабрь к декабрю; у общей валюты своего курса нет
    p["deval"] = (p.fx_dec / gb.fx_dec.shift(1) - 1).where(p.own_currency)
    p["big"] = (p.deval >= DEVAL).astype(int)
    prev = gb.big.shift(1).fillna(0) + gb.big.shift(2).fillna(0)
    p["deval25"] = ((p.big == 1) & (prev == 0)).astype(float)
    p.loc[p.deval.isna(), "deval25"] = np.nan
    p["gmd_cc"] = p.CurrencyCrisis
    for t, last in TARGET_LAST_YEAR.items():
        p.loc[p.year > last, t] = np.nan
    gb = p.groupby("iso3")
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
    # метка известна, если кризис уже случился в t+1 или известны оба года
    y = np.where(nxt == 1, 1.0, np.where(nxt.isna() | nxt2.isna(), np.nan, (nxt2 == 1).astype(float)))
    p["y"] = y
    drop = p[target].fillna(0).astype(bool)
    for j in range(1, POST_CRISIS_DROP + 1):
        drop |= gb.shift(j).fillna(0).astype(bool)
    if target == "deval25":
        # большая девальвация идёт в t или t−1 → в t+1 «начаться» она не может: y был бы занулён правилом
        drop |= (p.big == 1) | (p.groupby("iso3").big.shift(1).fillna(0) == 1)
    p["usable"] = p.deval.notna() & p.own_currency & ~p.hyper_prev.fillna(False) & (p.year >= FIRST_YEAR)
    p["evaluable"] = ~drop & p.y.notna() & p.usable
    return p


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
    through = T - H if train_through is None else train_through
    train = p[p.evaluable & (p.year <= through) & (p.year >= start)]
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
    preds, coefs = [], []
    for T in range(EVAL_FROM, last_eval + 1):
        long, m_long = forecast_asof(p, T, FIRST_YEAR, return_models=True)
        short, m_short = forecast_asof(p, T, SHORT_FROM, return_models=True)
        test = long[long.evaluable].rename(columns={"logit": "logit_long", "extree": "extree_long"})
        if test.empty:
            continue
        test = test.join(short[["logit", "extree"]].rename(columns={"logit": "logit_short", "extree": "extree_short"}))
        for tag, m in (("long", m_long), ("short", m_short)):
            if m:
                coefs.append(pd.Series(m["logit"][-1].coef_[0], index=FEATURES, name=f"{tag}_{T}"))
        preds.append(test)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(coefs)


MODELS = ["base_rate", "persist", "logit_short", "extree_short", "logit_long", "extree_long"]


def _auc(y, s):
    return roc_auc_score(y, s) if (y.nunique() == 2 and s.nunique() > 1) else np.nan


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


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
