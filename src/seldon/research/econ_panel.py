"""Экономическая панель по всем странам (Global Macro Database + Всемирный банк),
целевая «валютный кризис», модель Банка Англии на постсоветских странах.

Запуск:
    uv run python -m seldon econ_panel

Что делает:
  1. Панель страна×год из GMD (nGDP, CPI, курс к доллару, счёт текущих операций, госдолг,
     инвестиции, потребление, дамми кризисов) и кредита частному сектору (% ВВП) из ВБ.
     Годы ≤ 2024 (дальше в GMD прогнозы МВФ).
  2. Признаки в духе BoE (2-летние изменения): рост ИПЦ, рост реального потребления на душу,
     изменения госдолг/ВВП, инвестиции/ВВП, СТО/ВВП, кредит/ВВП, плюс «мировой» рост кредита
     (среднее по 18 странам JST). Наклона кривой доходности, акций и обслуживания долга
     для постсоветских стран нет — их не берём.
  3. Целевая `deval25`: курс к доллару (среднегодовой) вырос ≥ 25% за год, и это первый
     такой год после ≥ 2 спокойных. Для сравнения — `gmd_crisis` = валютный или банковский
     кризис по хронике GMD (Laeven–Valencia и др., до 2017).
     Метка BoE: y(t) = 1, если кризис в t+1 или t+2.
  4. Две модели:
     - «перенос»: логрегрессия BoE, обученная на JST R6 на тех признаках, что есть у нас,
       применённая к постсоветским странам без переобучения;
     - «своя»: логрегрессия и extremely randomised trees на панели GMD по всем странам,
       проверка вслепую по годам (учимся на метках, известных к году T, предсказываем T+1).
  5. Таблица: что модели давали за год до каждого постсоветского кризиса (1998, 2008–09,
     2014–15 и другие), с местом страны среди всех стран мира в тот год.

Выход:
    data/processed/econ_panel.parquet, econ_predictions.parquet
    reports/econ_post_soviet_coverage.csv, econ_crises_post_soviet.csv,
    reports/econ_metrics.csv, econ_pre_crisis_table.csv, econ_coefficients.csv
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import DATA_PROCESSED, REPORTS
from seldon.core.entities import ISO3_TO_SHORT, POST_SOVIET_ISO3
from seldon.domains.banking.econ_boe import build_boe_dataset
from seldon.sources.gmd import load_gmd
from seldon.sources.jst import JST_R6, load_jst
from seldon.sources.wdi import read_wdi

H = 2
LAST_YEAR = 2024
FIRST_EVAL_YEAR = 1996  # 1992–1995 — гиперинфляция переходного периода, кризис у всех подряд
JST_ISO = ["AUS", "BEL", "CAN", "CHE", "DEU", "DNK", "ESP", "FIN", "FRA", "GBR", "IRL", "ITA",
           "JPN", "NLD", "NOR", "PRT", "SWE", "USA"]
FEATURES = ["cpi_pdiff2", "cons_pdiff2", "pdebt_gdp_rdiff2", "inv_gdp_rdiff2", "ca_gdp_rdiff2",
            "tloan_gdp_rdiff2", "global_loan2"]
FEATURES_NOCREDIT = [f for f in FEATURES if f not in ("tloan_gdp_rdiff2", "global_loan2")]
# признаки из литературы о валютных кризисах (Kaminsky–Reinhart): уровень инфляции, СТО/ВВП,
# падение резервов, рост реального курса (переоценка), прошлогодняя девальвация
FEATURES_KR = [*FEATURES, "infl_log", "ca_gdp_level", "reserves_pdiff2", "reer_pdiff2", "deval_prev"]
DEVAL = 0.25
POST_CRISIS_DROP = 2  # у BoE 4; у нас история короткая, 4 года съели бы 2008 после 2003 и т.п.
TRAIN_YEARS = range(1994, LAST_YEAR)  # T: признаки года T → кризис в T+1/T+2
TARGETS = ("deval25", "gmd_crisis", "banking")


# ---------------------------------------------------------------------------
# Панель
# ---------------------------------------------------------------------------


def load_panel() -> pd.DataFrame:
    g = load_gmd()
    g = g[(g.year >= 1960) & (g.year <= LAST_YEAR)].copy()
    g["year"] = g.year.astype(int)
    g = g.rename(columns={"ISO3": "iso3", "countryname": "country"})
    cols = ["iso3", "country", "year", "nGDP", "rGDP", "CPI", "infl", "USDfx", "REER", "CA_GDP", "govdebt_GDP",
            "inv_GDP", "cons", "pop", "M3", "CurrencyCrisis", "BankingCrisis", "SovDebtCrisis"]
    g = g[cols]
    for name in ("credit_gdp", "reserves_usd"):
        w = read_wdi(name).rename(columns={"value": name})
        g = g.merge(w[["iso3", "year", name]], on=["iso3", "year"], how="left")
    g = g[g.nGDP.notna()].sort_values(["iso3", "year"]).reset_index(drop=True)
    g["post_soviet"] = g.iso3.isin(POST_SOVIET_ISO3)
    return g


def add_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    gb = p.groupby("iso3")
    p["rcons_pc"] = p.cons / p.CPI / p["pop"]
    for src, dst in [("CPI", "cpi_pdiff2"), ("rcons_pc", "cons_pdiff2")]:
        lag = gb[src].shift(H)
        p[dst] = (p[src] - lag) / lag
    for src, dst in [("govdebt_GDP", "pdebt_gdp_rdiff2"), ("inv_GDP", "inv_gdp_rdiff2"),
                     ("CA_GDP", "ca_gdp_rdiff2"), ("credit_gdp", "tloan_gdp_rdiff2")]:
        p[dst] = gb[src].diff(H) / 100.0
    # «мировой» рост кредита — среднее по 18 странам JST (как global_loan у BoE, но без
    # исключения собственной страны: постсоветских стран там всё равно нет)
    jst_mean = p[p.iso3.isin(JST_ISO)].groupby("year").tloan_gdp_rdiff2.mean().rename("global_loan2")
    p = p.merge(jst_mean, on="year", how="left")
    p["infl_log"] = np.log1p(p.infl.clip(lower=-50) / 100.0)
    p["ca_gdp_level"] = p.CA_GDP / 100.0
    for src, dst in [("reserves_usd", "reserves_pdiff2"), ("REER", "reer_pdiff2")]:
        lag = gb[src].shift(H)
        p[dst] = (p[src] - lag) / lag
    # целевая
    p["deval"] = gb.USDfx.pct_change()
    p["deval_prev"] = p.groupby("iso3").deval.shift(1)
    big = (p.deval >= DEVAL).astype(int)
    prev = big.groupby(p.iso3).shift(1).fillna(0) + big.groupby(p.iso3).shift(2).fillna(0)
    p["deval25"] = ((big == 1) & (prev == 0)).astype(int)
    p["gmd_crisis"] = ((p.CurrencyCrisis == 1) | (p.BankingCrisis == 1)).astype(int)
    p.loc[p.CurrencyCrisis.isna() & p.BankingCrisis.isna(), "gmd_crisis"] = np.nan
    p["banking"] = p.BankingCrisis  # ближе всего к целевой BoE (системный банковский кризис)
    gb = p.groupby("iso3")
    for tgt in TARGETS:
        nxt = gb[tgt].shift(-1)
        nxt2 = gb[tgt].shift(-2)
        p[f"y_{tgt}"] = np.where(nxt.isna() & nxt2.isna(), np.nan, ((nxt == 1) | (nxt2 == 1)).astype(float))
        # сам год кризиса и POST_CRISIS_DROP после — не в обучении/оценке (как у BoE)
        drop = p[tgt].fillna(0).astype(bool)
        for j in range(1, POST_CRISIS_DROP + 1):
            drop |= gb[tgt].shift(j).fillna(0).astype(bool)
        p[f"drop_{tgt}"] = drop
    return p


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------


def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=5000))


def transfer_from_jst(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Логрегрессия BoE на JST R6, признаки — только те, что есть у нас; применяем ко всем."""
    jst = build_boe_dataset(load_jst(JST_R6, "Sheet1"), predictors=FEATURES)
    out = panel.copy()
    coefs = {}
    for name, feats in [("jst_transfer", FEATURES), ("jst_transfer_nocredit", FEATURES_NOCREDIT)]:
        m = make_logreg().fit(jst[feats].values, jst.crisis.values)
        coefs[name] = pd.Series(m[-1].coef_[0], index=feats)
        X = out[feats].copy()
        # переходная гиперинфляция даёт значения вне диапазона JST — обрезаем по обучающей выборке
        X = X.clip(lower=jst[feats].min(), upper=jst[feats].max(), axis=1)
        ok = X.notna().all(axis=1)
        out[name] = np.nan
        out.loc[ok, name] = m.predict_proba(X[ok].values)[:, 1]
    return out, pd.DataFrame(coefs)


def own_model(panel: pd.DataFrame, feats: list[str], name: str, target: str = "deval25"
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Проверка вслепую по годам на всей панели GMD: обучение на строках с t + 2 ≤ T.
    Признаки обрезаются по 1-му и 99-му процентилю обучающей выборки (гиперинфляции
    и прочие выбросы иначе ломают логрегрессию)."""
    y = f"y_{target}"
    base = panel[panel[y].notna() & ~panel[f"drop_{target}"] & panel[feats].notna().all(axis=1)]
    preds, coefs = [], []
    for T in TRAIN_YEARS:
        train = base[base.year <= T - 2]
        test = base[base.year == T].copy()
        if len(test) == 0 or train[y].sum() < 20:
            continue
        lo, hi = train[feats].quantile(0.01), train[feats].quantile(0.99)
        Xtr = train[feats].clip(lo, hi, axis=1).values
        Xte = test[feats].clip(lo, hi, axis=1).values
        lr = make_logreg().fit(Xtr, train[y].astype(int))
        et = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1).fit(
            Xtr, train[y].astype(int))
        test[f"{name}_logreg"] = lr.predict_proba(Xte)[:, 1]
        test[f"{name}_extree"] = et.predict_proba(Xte)[:, 1]
        preds.append(test[["iso3", "year", f"{name}_logreg", f"{name}_extree"]])
        coefs.append(pd.Series(lr[-1].coef_[0], index=feats, name=T))
    return pd.concat(preds), pd.DataFrame(coefs)


def _auc(y, p):
    return roc_auc_score(y, p) if (y.nunique() == 2 and p.nunique() > 1) else np.nan


def metrics(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        y = f"y_{target}"
        d = df[df[y].notna() & ~df[f"drop_{target}"] & (df.year >= FIRST_EVAL_YEAR)]
        for scope, sub in [("world", d), ("post_soviet", d[d.post_soviet])]:
            for m in models:
                s = sub[sub[m].notna()]
                # auc — по всем строкам сразу; auc_within_year — среднее по годам (умеет ли
                # модель ранжировать страны внутри одного года, без «волн» 1998/2009/2015)
                yearly = [_auc(g[y], g[m]) for _, g in s.groupby("year")]
                rows.append({"target": target, "scope": scope, "model": m, "n": len(s),
                             "positives": int(s[y].sum()), "auc": _auc(s[y], s[m]),
                             "auc_within_year": np.nanmean(yearly) if len(yearly) else np.nan,
                             "brier": brier_score_loss(s[y], s[m]) if len(s) else np.nan})
    return pd.DataFrame(rows)


def pre_crisis_table(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Для каждого постсоветского кризиса (deval25 или хроника GMD, с 1996): значения моделей
    за год до него и место страны среди всех стран мира с прогнозом в тот год."""
    ev = df[df.post_soviet & (df.year >= FIRST_EVAL_YEAR) & ((df.deval25 == 1) | (df.gmd_crisis == 1))]
    rows = []
    for _, e in ev.iterrows():
        prev = df[(df.iso3 == e.iso3) & (df.year == e.year - 1)]
        row = {"country": ISO3_TO_SHORT[e.iso3], "iso3": e.iso3, "crisis_year": int(e.year),
               "deval_pct": round(100 * e.deval, 0) if pd.notna(e.deval) else np.nan,
               "deval25": int(e.deval25), "gmd_currency": e.CurrencyCrisis, "gmd_banking": e.BankingCrisis}
        if len(prev):
            prev = prev.iloc[0]
            world = df[df.year == e.year - 1]
            for m in models:
                row[m] = prev[m]
                if pd.notna(prev[m]):
                    row[f"{m}_rank"] = int((world[m] > prev[m]).sum() + 1)
                    row[f"{m}_of"] = int(world[m].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def coverage(panel: pd.DataFrame) -> pd.DataFrame:
    ps = panel[panel.post_soviet]
    rows = []
    for iso, g in ps.groupby("iso3"):
        row = {"country": ISO3_TO_SHORT[iso], "iso3": iso}
        for c in ["nGDP", "rGDP", "CPI", "USDfx", "CA_GDP", "govdebt_GDP", "inv_GDP", "cons", "M3", "credit_gdp"]:
            yrs = g.loc[g[c].notna(), "year"]
            row[c] = f"{yrs.min()}–{yrs.max()}" if len(yrs) else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    panel = add_features(load_panel())
    panel.to_parquet(DATA_PROCESSED / "econ_panel.parquet", index=False)
    cov = coverage(panel)
    cov.to_csv(REPORTS / "econ_post_soviet_coverage.csv", index=False)
    print("Покрытие рядов по постсоветским странам:\n", cov.to_string(index=False))

    crises = panel[panel.post_soviet & (panel.year >= 1992) & ((panel.deval25 == 1) | (panel.gmd_crisis == 1))]
    crises = crises[["iso3", "year", "deval", "deval25", "CurrencyCrisis", "BankingCrisis"]].assign(
        country=lambda d: d.iso3.map(ISO3_TO_SHORT))
    crises.to_csv(REPORTS / "econ_crises_post_soviet.csv", index=False)
    print("\nКризисы по постсоветским странам (deval25 или хроника GMD):\n", crises.round(2).to_string(index=False))

    df, coef_t = transfer_from_jst(panel)
    own, coef_o = own_model(panel, FEATURES, "own")
    kr, coef_k = own_model(panel, FEATURES_KR, "kr")
    df = df.merge(own, on=["iso3", "year"], how="left").merge(kr, on=["iso3", "year"], how="left")
    models = ["jst_transfer", "jst_transfer_nocredit", "own_logreg", "own_extree", "kr_logreg", "kr_extree"]
    df.to_parquet(DATA_PROCESSED / "econ_predictions.parquet", index=False)

    met = metrics(df, models)
    met.to_csv(REPORTS / "econ_metrics.csv", index=False)
    print("\nМетрики (вслепую, с 1996):\n", met.round(3).to_string(index=False))

    tab = pre_crisis_table(df, models)
    tab.to_csv(REPORTS / "econ_pre_crisis_table.csv", index=False)
    show = ["country", "crisis_year", "deval_pct", "gmd_currency", "gmd_banking", "jst_transfer", "jst_transfer_rank",
            "jst_transfer_nocredit", "own_logreg", "own_logreg_rank", "own_extree", "own_extree_rank",
            "kr_logreg", "kr_logreg_rank", "kr_extree", "kr_extree_rank", "kr_extree_of"]
    print("\nЗа год до кризиса:\n", tab[[c for c in show if c in tab]].round(3).to_string(index=False))

    coefs = pd.concat([coef_t, coef_o.mean().rename("own_logreg_mean"), coef_k.mean().rename("kr_logreg_mean")], axis=1)
    coefs.round(3).to_csv(REPORTS / "econ_coefficients.csv")
    print("\nКоэффициенты (стандартизованные признаки):\n", coefs.round(2))
    return 0

