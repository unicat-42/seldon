"""Модель Банка Англии (Bluwstein и др.) на развивающихся странах: GMD + кредит BIS.

Запуск:
    uv run python -m seldon econ_boe_em [lv] [apr1]
    без аргументов                             # метка — хроника GMD
    lv                                         # метка — Laeven–Valencia 2018 (`sources/lv.py`), оценка до 2015
    apr1                                       # прогноз «на 1 апреля» (макроданные без лага), суффикс `_apr1`

Вопрос: держится ли модель BoE по банковским кризисам (AUC 0,65 при переносе на GMD в
`research/econ_panel.py`) на выборке побольше — 44 страны BIS, из них ~20 развивающихся (Россия,
Турция, Бразилия, Мексика, Аргентина, Корея, Таиланд, Индонезия, Китай...), где есть
длинный кредит частному сектору и хроника банковских кризисов.

Что делает:
  1. Панель страна×год: из GMD (`download gmd`) ВВП, ИПЦ, потребление, население, госдолг,
     инвестиции, СТО, M3 (или M2), короткая ставка (`strate`, или ставка ЦБ) и длинная
     (`ltrate`), дамми банковского кризиса; из BIS (`download bis`) кредит банков частному
     нефинансовому сектору, % ВВП, 4-й квартал. Всё это переводится в формат JST и
     прогоняется через тот же `build_boe_dataset`, что и в повторении статьи (`econ_boe.py`).
  2. Два набора признаков BoE (без цен акций — их у GMD нет):
     `full` — 11 признаков, включая наклон кривой доходности и обслуживание долга
     (нужна длинная ставка; есть у 18 из 21 развивающихся);
     `core` — 8 признаков без ставок;
     `nomoney` — 7, ещё и без денежной массы (у России нет M3 с 2023, у еврозоны — вовсе).
     Лаг публикации (чекпоинт 2.2): прогноз делается «на 1 января T+1», когда годовые
     макроданные за T ещё не вышли, — поэтому макроряды строки T берутся за T−1 (и у JST
     тоже); ставки — рыночные данные, они за T. `apr1` — без сдвига, как раньше.
  3. Целевая как у BoE: 1 для двух лет перед банковским кризисом; год кризиса и 4 после —
     выброшены.
  4. Модели вслепую по годам T (учимся на строках с t + 2 ≤ T, предсказываем T):
     - `base_rate`;
     - `jst_*` — логрег/деревья BoE, обученные на JST R6 (18 развитых, 1870–2020),
       применены без переобучения (перенос);
     - `own_*` — то же, обучено на панели GMD+BIS (все 44 страны) с расширяющимся окном;
     - `pooled_*` — обучено на JST + GMD+BIS вместе.
     Оценка: все страны BIS, развивающиеся, развитые; AUC общий и внутри года.
  5. Таблица: кризисы развивающихся стран с 1990 — прогноз за год до и место среди стран.

Выход:
    data/processed/boe_em_panel.parquet, boe_em_predictions.parquet
    reports/boe_em_coverage.csv, boe_em_crises.csv, boe_em_metrics.csv,
    boe_em_metrics_by_year.csv, boe_em_pre_crisis_table.csv, boe_em_coefficients.csv
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, REPORTS
from seldon.core.metrics import auc, auc_within, brier
from seldon.core.panel import lag_columns
from seldon.core.temporal import label_known, publication_lag
from seldon.domains.banking.econ_boe import HORIZON, PREDICTORS, build_boe_dataset
from seldon.evaluation.backtest import backtest
from seldon.sources.bis import BIS_ISO, load_bis_credit
from seldon.sources.gmd import load_gmd
from seldon.sources.jst import JST_R6, load_jst
from seldon.sources.lv import load_lv_country_year

FIRST_EVAL_YEAR = 1990
# Последний год, за который хроника банковских кризисов реально размечена. В GMD колонка
# BankingCrisis после ~2017 заполнена нулями вместо пропусков (у развивающихся стран BIS
# последний кризис — 2008): учиться и оценивать на этих «нулях» нельзя.
LAST_LABEL_YEAR = {"gmd": 2017, "lv": 2017}
JST_LAST_LABEL_YEAR = 2020  # хроника JST R6


@dataclass(frozen=True)
class Config:
    label: str = "gmd"      # или "lv": хроника Laeven–Valencia 2018 (lv.py), 1970–2017, где есть; иначе GMD
    last_year: int = 2024   # последний год признаков в панели
    origin: str = "jan1"    # точка прогноза: "jan1" (макро за T−1) или "apr1" (макро за T)

    @property
    def macro_lag(self) -> int:
        return publication_lag("gmd_annual", self.origin)

    @property
    def last_label_year(self) -> int:
        return LAST_LABEL_YEAR[self.label]

    @property
    def eval_years(self) -> range:
        return range(FIRST_EVAL_YEAR, self.last_label_year - 1)  # метка строки T смотрит в T+2

    @property
    def suffix(self) -> str:
        return ("" if self.label == "gmd" else f"_{self.label}") + ("" if self.origin == "jan1" else f"_{self.origin}")


MIN_CRISIS_TRAIN = 10

EM = {"ARG", "BRA", "CHL", "CHN", "COL", "CZE", "HKG", "HUN", "IDN", "IND", "ISR", "KOR", "MEX",
      "MYS", "POL", "RUS", "SAU", "SGP", "THA", "TUR", "ZAF"}
JST_ISO = {"AUS", "BEL", "CAN", "CHE", "DEU", "DNK", "ESP", "FIN", "FRA", "GBR", "IRL", "ITA",
           "JPN", "NLD", "NOR", "PRT", "SWE", "USA"}

FULL = [p for p in PREDICTORS if p != "stock_pdiff2"]
CORE = [p for p in FULL if p not in ("drate", "global_drate", "tdbtserv_gdp_rdiff2")]
NOMONEY = [p for p in CORE if p != "bmon_gdp_rdiff2"]  # без M3: Россия с 2023, еврозона
FEATURE_SETS = {"full": FULL, "core": CORE, "nomoney": NOMONEY}


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------


def load_panel(cfg: Config = Config()) -> pd.DataFrame:
    """GMD + BIS в формате JST (имена колонок как в JST R6, чтобы работал build_boe_dataset)."""
    g = load_gmd()
    g = g[(g.year >= 1950) & (g.year <= cfg.last_year)].copy()
    g["year"] = g.year.astype(int)
    g = g.rename(columns={"ISO3": "iso3"})
    g = g[g.iso3.isin(set(BIS_ISO.values()))]
    g = g.merge(load_bis_credit(), on=["iso3", "year"], how="left")
    crisis = g.BankingCrisis
    if cfg.label == "lv":
        lv = load_lv_country_year()[["iso3", "year", "lv_banking"]]
        g = g.merge(lv, on=["iso3", "year"], how="left")
        crisis = g.lv_banking.where(g.lv_banking.notna(), g.BankingCrisis)
    j = pd.DataFrame({
        "iso": g.iso3, "year": g.year, "country": g.countryname,
        "gdp": g.nGDP,
        "tloans": g.credit_gdp / 100.0 * g.nGDP,
        "money": g.M3.fillna(g.M2),
        "stir": g.strate.fillna(g.cbrate),
        "ltrate": g.ltrate,
        "cpi": g.CPI,
        "rconpc": g.cons / g.CPI / g["pop"],
        "iy": g.inv_GDP / 100.0,
        "debtgdp": g.govdebt_GDP / 100.0,
        "ca": g.CA_GDP / 100.0 * g.nGDP,
        "crisisJST": crisis,
        "credit_src": g.credit_src,
    })
    # build_boe_dataset ждёт и эти колонки (не используются в наших наборах признаков)
    for c in ("narrowm", "tbus", "thh", "tmort", "stocks", "hpnom"):
        j[c] = np.nan
    j = j[j.gdp.notna()].sort_values(["iso", "year"]).reset_index(drop=True)
    j["em"] = j.iso.isin(EM)
    return j


MACRO_LAG = publication_lag("gmd_annual", "jan1")  # 1 год: прогноз «на 1 января»
# Что НЕ сдвигается: ключи; хроника кризисов (это метка, не признак); ставки и цены акций —
# рыночные данные, они известны в конце года T без лага публикации. Сдвигаются годовые
# макроданные: ВВП, кредит, деньги, ИПЦ, потребление, инвестиции, госдолг, счёт текущих операций.
NOT_LAGGED = ("iso", "year", "country", "crisisJST", "stir", "ltrate", "stocks")


def boe_dataset(raw: pd.DataFrame, feats: list[str], macro_lag: int = MACRO_LAG) -> pd.DataFrame:
    """Таблица признаков BoE с лагом публикации: годовые макроряды сдвигаются на `macro_lag`
    лет по календарю (на строке T — данные за T − `macro_lag`), ставки остаются за T; метка
    (кризис в T+1…T+2) остаётся на своём месте. Наклон кривой доходности считается по ставкам
    года T, обслуживание долга — последний известный кредит × ставка года T.
    Годится и для панели GMD+BIS, и для JST."""
    raw = lag_columns(raw, "iso", "year", [c for c in raw.columns if c not in NOT_LAGGED], macro_lag)
    return build_boe_dataset(raw, predictors=feats)


def build(panel: pd.DataFrame, feats: list[str], macro_lag: int = MACRO_LAG) -> pd.DataFrame:
    d = boe_dataset(panel.drop(columns=["em", "credit_src", "country"]), feats, macro_lag)
    d["em"] = d.iso.isin(EM)
    return d


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------


def make_models(seed: int = 0, n_estimators: int = 500) -> dict:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=5000)),
        "extree": ExtraTreesClassifier(n_estimators=n_estimators, min_samples_leaf=5, random_state=seed, n_jobs=-1),
    }


MODELS = ["base_rate", "jst_logreg", "jst_extree", "own_logreg", "own_extree", "pooled_logreg", "pooled_extree"]


def forecast_asof(data: pd.DataFrame, jst_d: pd.DataFrame, feats: list[str], T: int,
                  n_estimators: int = 500, return_coefs: bool = False):
    """Прогноз с признаков года T (кризис в T+1…T+2). ВСЕ модели, включая перенос с JST,
    учатся только на строках года ≤ T−2 — тех, чья метка к концу T уже известна.
    (До аудита `jst_*` обучались один раз на всей JST 1870–2020 и «знали» кризисы 1990-х
    и 2008 года, а 18 развитых стран BIS — это те же 18 стран JST.)"""
    test = data[data.year == T].copy()
    for m in MODELS:
        test[m] = np.nan
    coefs = {}
    jst_tr = jst_d[label_known(jst_d.year, T, HORIZON)]
    if jst_tr.crisis.sum() >= MIN_CRISIS_TRAIN and not test.empty:
        lo, hi = jst_tr[feats].min(), jst_tr[feats].max()
        for k, m in make_models(n_estimators=n_estimators).items():
            m.fit(jst_tr[feats].values, jst_tr.crisis.values)
            test[f"jst_{k}"] = m.predict_proba(test[feats].clip(lo, hi, axis=1).values)[:, 1]
            if k == "logreg":
                coefs["jst"] = pd.Series(m[-1].coef_[0], index=feats)
    train = data[label_known(data.year, T, HORIZON)]
    if train.crisis.sum() >= MIN_CRISIS_TRAIN and not test.empty:
        test["base_rate"] = train.crisis.mean()
        q1, q99 = train[feats].quantile(0.01), train[feats].quantile(0.99)
        Xtr = train[feats].clip(q1, q99, axis=1).values
        Xte = test[feats].clip(q1, q99, axis=1).values
        for k, m in make_models(n_estimators=n_estimators).items():
            m.fit(Xtr, train.crisis.values)
            test[f"own_{k}"] = m.predict_proba(Xte)[:, 1]
            if k == "logreg":
                coefs["own"] = pd.Series(m[-1].coef_[0], index=feats)
        pooled = pd.concat([jst_tr, train])
        Xp = pooled[feats].clip(q1, q99, axis=1).values
        for k, m in make_models(n_estimators=n_estimators).items():
            m.fit(Xp, pooled.crisis.values)
            test[f"pooled_{k}"] = m.predict_proba(Xte)[:, 1]
    return (test, coefs) if return_coefs else test


def run(panel: pd.DataFrame, jst: pd.DataFrame, feats: list[str], tag: str, cfg: Config
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = build(panel, feats, cfg.macro_lag)
    jst_d = boe_dataset(jst, feats, cfg.macro_lag)
    jst_coefs, own_coefs = [], []

    def step(T: int) -> pd.DataFrame | None:
        test, c = forecast_asof(data, jst_d, feats, T, return_coefs=True)
        if test.empty:
            return None
        if "jst" in c:
            jst_coefs.append(c["jst"].rename(T))
        if "own" in c:
            own_coefs.append(c["own"].rename(T))
        return test

    out = backtest(cfg.eval_years, step)
    out["feature_set"] = tag
    coefs = [pd.DataFrame(x).mean().rename(f"{tag}_{name}_logreg_mean")
             for name, x in (("jst", jst_coefs), ("own", own_coefs)) if x]
    return out, pd.concat(coefs, axis=1)


def metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tag, d in pred.groupby("feature_set"):
        jst_c = d.iso.isin(JST_ISO)
        # non_jst_adv / em — честный перенос: стран нет в обучающей выборке JST
        for scope, sub in [("all", d), ("em", d[d.em]), ("advanced", d[~d.em]),
                           ("jst_adv", d[jst_c]), ("non_jst_adv", d[~d.em & ~jst_c]), ("non_jst", d[~jst_c])]:
            for m in MODELS:
                s = sub[sub[m].notna()]
                if s.empty:
                    continue
                rows.append({"feature_set": tag, "scope": scope, "model": m, "n": len(s),
                             "countries": s.iso.nunique(), "positives": int(s.crisis.sum()),
                             "auc": auc(s.crisis, s[m]), "auc_within_year": auc_within(s.crisis, s[m], s.year),
                             "brier": brier(s.crisis, s[m])})
    return pd.DataFrame(rows)


def metrics_by_year(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tag, T), d in pred.groupby(["feature_set", "year"]):
        for m in MODELS:
            s = d[d[m].notna()]
            rows.append({"feature_set": tag, "year": T, "model": m, "n": len(s), "positives": int(s.crisis.sum()),
                         "auc": auc(s.crisis, s[m]) if len(s) else np.nan})
    return pd.DataFrame(rows)


def pre_crisis_table(panel: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Кризисы развивающихся стран с FIRST_EVAL_YEAR: прогноз за год до (признаки года t−1)
    и место среди всех стран с прогнозом в том году."""
    rows = []
    crises = panel[(panel.crisisJST == 1) & panel.em & (panel.year > FIRST_EVAL_YEAR)]
    for _, c in crises.iterrows():
        for tag, d in pred.groupby("feature_set"):
            snap = d[d.year == c.year - 1]
            me = snap[snap.iso == c.iso]
            row = {"iso3": c.iso, "country": c.country, "crisis_year": int(c.year), "feature_set": tag,
                   "in_sample": not me.empty, "n_countries": len(snap)}
            for m in ("jst_logreg", "jst_extree", "own_logreg", "own_extree", "pooled_extree"):
                if me.empty or pd.isna(me[m].iloc[0]):
                    row[m], row[f"{m}_rank"] = np.nan, np.nan
                else:
                    v = me[m].iloc[0]
                    row[m], row[f"{m}_rank"] = round(v, 3), int((snap[m] > v).sum() + 1)
            rows.append(row)
    return pd.DataFrame(rows)


def coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iso, g in panel[panel.em].groupby("iso"):
        row = {"iso3": iso, "country": g.country.iloc[0]}
        for c in ["tloans", "money", "stir", "ltrate", "cpi", "rconpc", "iy", "debtgdp", "ca"]:
            yrs = g.loc[g[c].notna(), "year"]
            row[c] = f"{yrs.min()}–{yrs.max()}" if len(yrs) else "—"
        row["credit_src"] = ",".join(sorted(g.credit_src.dropna().unique()))
        row["crises"] = ",".join(str(int(y)) for y in g.loc[g.crisisJST == 1, "year"])
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    cfg = Config(label="lv" if "lv" in argv else "gmd", origin="apr1" if "apr1" in argv else "jan1")
    print(f"прогноз «на {'1 января' if cfg.origin == 'jan1' else '1 апреля'}»: макроданные с лагом {cfg.macro_lag} г.")
    sfx = cfg.suffix
    panel = load_panel(cfg)
    panel.to_parquet(DATA_PROCESSED / f"boe_em_panel{sfx}.parquet", index=False)
    cov = coverage(panel)
    cov.to_csv(REPORTS / f"boe_em_coverage{sfx}.csv", index=False)
    print("Покрытие развивающихся стран (GMD + BIS):\n", cov.to_string(index=False))

    jst = load_jst(JST_R6, "Sheet1")
    preds, coefs = [], []
    for tag, feats in FEATURE_SETS.items():
        d = build(panel, feats, cfg.macro_lag)
        print(f"\n[{tag}] {len(feats)} признаков: {len(d)} страна-лет, {d.iso.nunique()} стран "
              f"({d[d.em].iso.nunique()} развивающихся), {int(d.crisis.sum())} предкризисных лет "
              f"({int(d[d.em].crisis.sum())} у развивающихся)")
        p, c = run(panel, jst, feats, tag, cfg)
        preds.append(p)
        coefs.append(c)
    pred = pd.concat(preds, ignore_index=True)
    pred.to_parquet(DATA_PROCESSED / f"boe_em_predictions{sfx}.parquet", index=False)
    pd.concat(coefs, axis=1).round(3).to_csv(REPORTS / f"boe_em_coefficients{sfx}.csv")

    met = metrics(pred)
    met.to_csv(REPORTS / f"boe_em_metrics{sfx}.csv", index=False)
    print(f"\nМетрики вслепую {FIRST_EVAL_YEAR}–{cfg.last_label_year - 2}, метка {cfg.label}, целевая: банковский кризис в t+1 или t+2:")
    print(met.round(3).to_string(index=False))
    metrics_by_year(pred).to_csv(REPORTS / f"boe_em_metrics_by_year{sfx}.csv", index=False)

    cr = panel[(panel.crisisJST == 1) & panel.em][["iso", "country", "year"]]
    cr.to_csv(REPORTS / f"boe_em_crises{sfx}.csv", index=False)
    tab = pre_crisis_table(panel, pred)
    tab.to_csv(REPORTS / f"boe_em_pre_crisis_table{sfx}.csv", index=False)
    show = ["country", "crisis_year", "feature_set", "n_countries", "jst_logreg_rank", "jst_extree_rank",
            "own_logreg_rank", "own_extree_rank", "pooled_extree_rank"]
    print("\nКризисы развивающихся стран: место среди стран за год до (1 = самый рискованный):")
    print(tab[show].to_string(index=False))
    return 0

