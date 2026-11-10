"""Модель Банка Англии (Bluwstein и др.) на развивающихся странах: GMD + кредит BIS.

Запуск:
    uv run python -m seldon.econ_boe_em

Вопрос: держится ли модель BoE по банковским кризисам (AUC 0,65 при переносе на GMD в
`econ_panel.py`) на выборке побольше — 44 страны BIS, из них ~20 развивающихся (Россия,
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

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, DATA_RAW, PROJECT_ROOT
from seldon.download import GMD_VERSION
from seldon.econ_boe import JST_R6, PREDICTORS, build_boe_dataset, load_jst

REPORTS = PROJECT_ROOT / "reports"
LAST_YEAR = 2024
FIRST_EVAL_YEAR = 1990
EVAL_YEARS = range(FIRST_EVAL_YEAR, LAST_YEAR - 1)  # метки для T нужны до T+2
MIN_CRISIS_TRAIN = 10

# BIS: ISO2 заёмщика → ISO3; агрегаты (4T, 5A, 5R, G2, XM) пропускаем
BIS_ISO = {
    "AR": "ARG", "AT": "AUT", "AU": "AUS", "BE": "BEL", "BR": "BRA", "CA": "CAN", "CH": "CHE",
    "CL": "CHL", "CN": "CHN", "CO": "COL", "CZ": "CZE", "DE": "DEU", "DK": "DNK", "ES": "ESP",
    "FI": "FIN", "FR": "FRA", "GB": "GBR", "GR": "GRC", "HK": "HKG", "HU": "HUN", "ID": "IDN",
    "IE": "IRL", "IL": "ISR", "IN": "IND", "IT": "ITA", "JP": "JPN", "KR": "KOR", "LU": "LUX",
    "MX": "MEX", "MY": "MYS", "NL": "NLD", "NO": "NOR", "NZ": "NZL", "PL": "POL", "PT": "PRT",
    "RU": "RUS", "SA": "SAU", "SE": "SWE", "SG": "SGP", "TH": "THA", "TR": "TUR", "US": "USA",
    "ZA": "ZAF",
}
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


def load_bis_credit() -> pd.DataFrame:
    """Кредит банков частному нефинансовому сектору, % ВВП, значение 4-го квартала → год.
    Где банковского ряда нет — кредит всех секторов (помечено `credit_src`)."""
    d = pd.read_csv(DATA_RAW / "bis" / "WS_TC_csv_flat.csv", low_memory=False)
    d.columns = [c.split(":")[0] for c in d.columns]
    for c in ["BORROWERS_CTY", "TC_BORROWERS", "TC_LENDERS", "VALUATION", "UNIT_TYPE", "TC_ADJUST"]:
        d[c] = d[c].astype(str).str.split(":").str[0]
    d = d[(d.TC_BORROWERS == "P") & (d.UNIT_TYPE == "770") & (d.TC_ADJUST == "A")
          & d.TIME_PERIOD.astype(str).str.endswith("Q4") & d.OBS_VALUE.notna()]
    d["iso3"] = d.BORROWERS_CTY.map(BIS_ISO)
    d = d[d.iso3.notna()].copy()
    d["year"] = d.TIME_PERIOD.str[:4].astype(int)
    banks = d[d.TC_LENDERS == "B"].set_index(["iso3", "year"]).OBS_VALUE.rename("credit_banks")
    allsec = d[(d.TC_LENDERS == "A") & (d.VALUATION == "M")].set_index(["iso3", "year"]).OBS_VALUE.rename("credit_all")
    out = pd.concat([banks, allsec], axis=1).reset_index()
    out["credit_gdp"] = out.credit_banks.fillna(out.credit_all)
    out["credit_src"] = np.where(out.credit_banks.notna(), "banks", "all")
    return out


def load_panel() -> pd.DataFrame:
    """GMD + BIS в формате JST (имена колонок как в JST R6, чтобы работал build_boe_dataset)."""
    g = pd.read_stata(DATA_RAW / "gmd" / f"GMD_{GMD_VERSION}.dta")
    g = g[(g.year >= 1950) & (g.year <= LAST_YEAR)].copy()
    g["year"] = g.year.astype(int)
    g = g.rename(columns={"ISO3": "iso3"})
    g = g[g.iso3.isin(set(BIS_ISO.values()))]
    g = g.merge(load_bis_credit(), on=["iso3", "year"], how="left")
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
        "crisisJST": g.BankingCrisis,
        "credit_src": g.credit_src,
    })
    # build_boe_dataset ждёт и эти колонки (не используются в наших наборах признаков)
    for c in ("narrowm", "tbus", "thh", "tmort", "stocks", "hpnom"):
        j[c] = np.nan
    j = j[j.gdp.notna()].sort_values(["iso", "year"]).reset_index(drop=True)
    j["em"] = j.iso.isin(EM)
    return j


def build(panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    d = build_boe_dataset(panel.drop(columns=["em", "credit_src", "country"]), predictors=feats)
    d["em"] = d.iso.isin(EM)
    return d


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------


def make_models(seed: int = 0) -> dict:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=5000)),
        "extree": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=5, random_state=seed, n_jobs=-1),
    }


def run(panel: pd.DataFrame, jst: pd.DataFrame, feats: list[str], tag: str
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = build(panel, feats)
    jst_d = build_boe_dataset(jst, predictors=feats)
    preds, coefs = [], []
    # перенос: обучено на JST один раз
    jst_models = {k: m.fit(jst_d[feats].values, jst_d.crisis.values) for k, m in make_models().items()}
    coefs.append(pd.Series(jst_models["logreg"][-1].coef_[0], index=feats, name=f"{tag}_jst_logreg"))
    lo, hi = jst_d[feats].min(), jst_d[feats].max()
    own_coefs = []
    for T in EVAL_YEARS:
        test = data[data.year == T].copy()
        if test.empty:
            continue
        test["base_rate"] = np.nan
        Xte_j = test[feats].clip(lo, hi, axis=1).values
        for k, m in jst_models.items():
            test[f"jst_{k}"] = m.predict_proba(Xte_j)[:, 1]
        train = data[data.year <= T - 2]
        if train.crisis.sum() >= MIN_CRISIS_TRAIN:
            test["base_rate"] = train.crisis.mean()
            q1, q99 = train[feats].quantile(0.01), train[feats].quantile(0.99)
            Xtr = train[feats].clip(q1, q99, axis=1).values
            Xte = test[feats].clip(q1, q99, axis=1).values
            for k, m in make_models().items():
                m.fit(Xtr, train.crisis.values)
                test[f"own_{k}"] = m.predict_proba(Xte)[:, 1]
                if k == "logreg":
                    own_coefs.append(pd.Series(m[-1].coef_[0], index=feats, name=T))
            pooled = pd.concat([jst_d[jst_d.year <= T - 2], train])
            Xp = pooled[feats].clip(q1, q99, axis=1).values
            for k, m in make_models().items():
                m.fit(Xp, pooled.crisis.values)
                test[f"pooled_{k}"] = m.predict_proba(Xte)[:, 1]
        preds.append(test)
    out = pd.concat(preds, ignore_index=True)
    out["feature_set"] = tag
    if own_coefs:
        coefs.append(pd.DataFrame(own_coefs).mean().rename(f"{tag}_own_logreg_mean"))
    return out, pd.concat(coefs, axis=1)


MODELS = ["base_rate", "jst_logreg", "jst_extree", "own_logreg", "own_extree", "pooled_logreg", "pooled_extree"]


def _auc(y, p):
    return roc_auc_score(y, p) if (y.nunique() == 2 and p.nunique() > 1) else np.nan


def metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tag, d in pred.groupby("feature_set"):
        for scope, sub in [("all", d), ("em", d[d.em]), ("advanced", d[~d.em])]:
            for m in MODELS:
                s = sub[sub[m].notna()]
                if s.empty:
                    continue
                yearly = [_auc(g.crisis, g[m]) for _, g in s.groupby("year")]
                rows.append({"feature_set": tag, "scope": scope, "model": m, "n": len(s),
                             "countries": s.iso.nunique(), "positives": int(s.crisis.sum()),
                             "auc": _auc(s.crisis, s[m]), "auc_within_year": np.nanmean(yearly),
                             "brier": brier_score_loss(s.crisis, s[m])})
    return pd.DataFrame(rows)


def metrics_by_year(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tag, T), d in pred.groupby(["feature_set", "year"]):
        for m in MODELS:
            s = d[d[m].notna()]
            rows.append({"feature_set": tag, "year": T, "model": m, "n": len(s), "positives": int(s.crisis.sum()),
                         "auc": _auc(s.crisis, s[m]) if len(s) else np.nan})
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
    panel = load_panel()
    panel.to_parquet(DATA_PROCESSED / "boe_em_panel.parquet", index=False)
    cov = coverage(panel)
    cov.to_csv(REPORTS / "boe_em_coverage.csv", index=False)
    print("Покрытие развивающихся стран (GMD + BIS):\n", cov.to_string(index=False))

    jst = load_jst(JST_R6, "Sheet1")
    preds, coefs = [], []
    for tag, feats in FEATURE_SETS.items():
        d = build(panel, feats)
        print(f"\n[{tag}] {len(feats)} признаков: {len(d)} страна-лет, {d.iso.nunique()} стран "
              f"({d[d.em].iso.nunique()} развивающихся), {int(d.crisis.sum())} предкризисных лет "
              f"({int(d[d.em].crisis.sum())} у развивающихся)")
        p, c = run(panel, jst, feats, tag)
        preds.append(p)
        coefs.append(c)
    pred = pd.concat(preds, ignore_index=True)
    pred.to_parquet(DATA_PROCESSED / "boe_em_predictions.parquet", index=False)
    pd.concat(coefs, axis=1).round(3).to_csv(REPORTS / "boe_em_coefficients.csv")

    met = metrics(pred)
    met.to_csv(REPORTS / "boe_em_metrics.csv", index=False)
    print(f"\nМетрики вслепую {FIRST_EVAL_YEAR}–{LAST_YEAR - 2}, целевая: банковский кризис в t+1 или t+2:")
    print(met.round(3).to_string(index=False))
    metrics_by_year(pred).to_csv(REPORTS / "boe_em_metrics_by_year.csv", index=False)

    cr = panel[(panel.crisisJST == 1) & panel.em][["iso", "country", "year"]]
    cr.to_csv(REPORTS / "boe_em_crises.csv", index=False)
    tab = pre_crisis_table(panel, pred)
    tab.to_csv(REPORTS / "boe_em_pre_crisis_table.csv", index=False)
    show = ["country", "crisis_year", "feature_set", "n_countries", "jst_logreg_rank", "jst_extree_rank",
            "own_logreg_rank", "own_extree_rank", "pooled_extree_rank"]
    print("\nКризисы развивающихся стран: место среди стран за год до (1 = самый рискованный):")
    print(tab[show].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
