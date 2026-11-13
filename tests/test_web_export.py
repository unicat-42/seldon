"""Контракт веб-экспорта (Этап 5, чекпоинт 5.1).

Две группы тестов:
  1. всегда — на синтетике: математика вкладов и интервала (`publish/explain.py`), совпадение
     пересчёта экспортера с боевыми функциями прогноза, словарь названий стран;
  2. на собранном бандле `web/data/` — что каждая цифра сайта совпадает с `reports/risk_2025_26.csv`,
     `reports/scorecard.csv` и снимком `forecasts/2026-11/`, что на пропусках и общих валютах нет
     нулей, а даты наблюдений соответствуют лагам `core/temporal.py`. Бандл лежит в git: если он
     собран из других таблиц (sha256 в манифесте не совпадает) — тесты падают с подсказкой
     `make web`; пропускаются они только там, где бандла нет вовсе.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seldon import PROJECT_ROOT, REPORTS
from seldon.core import temporal
from seldon.core.entities import COUNTRY_ALIASES, COUNTRY_NAMES, ISO3_TO_RU, POST_SOVIET
from seldon.publish import web_export
from seldon.publish.explain import explain_logit, sigmoid

DATA = PROJECT_ROOT / "web" / "data"
VINTAGE = PROJECT_ROOT / "forecasts" / "2026-11"


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Синтетика
# ---------------------------------------------------------------------------


def _synth(n_groups=40, per=30, k=4, seed=0, group_effect=0.0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    X = rng.normal(size=(len(g), k))
    X[:, 0] += group_effect * rng.normal(size=n_groups)[g]  # признак, общий для группы → наблюдения зависимы
    eta = -1.5 + X @ np.array([1.0, -0.5, 0.0, 0.3])[:k] + group_effect * rng.normal(size=n_groups)[g]
    y = (rng.random(len(g)) < sigmoid(eta)).astype(int)
    return X, y, g, np.tile(np.arange(per), n_groups)


@pytest.mark.parametrize("C", [0.3, 1.0, np.inf])
def test_explain_contributions_sum_to_log_odds(C):
    """Вклады β·z в сумме со свободным членом дают лог-шансы, а сигмоида от них — ровно ту
    вероятность, которую выдаёт модель. Интервал накрывает оценку."""
    X, y, g, t = _synth()
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X, y)
    ex = explain_logit(m, list("abcd"), X, y, X[:50], [g, t])
    np.testing.assert_allclose(ex.intercept + ex.contrib.sum(axis=1), ex.eta, atol=1e-12)
    np.testing.assert_allclose(ex.p, m.predict_proba(X[:50])[:, 1], atol=1e-10)
    assert (ex.lo < ex.p).all() and (ex.p < ex.hi).all() and (ex.se > 0).all()
    assert ex.n_train == len(y) and ex.base_rate == pytest.approx(y.mean())


def test_explain_interval_is_wider_for_clustered_data():
    """Если наблюдения внутри страны зависимы, интервал с кластерами по странам шире, чем
    «наивный» (каждое наблюдение — свой кластер). И точка вдали от данных неопределённее средней."""
    X, y, g, _ = _synth(group_effect=1.5, seed=3)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X, y)
    naive = explain_logit(m, list("abcd"), X, y, X[:200], [np.arange(len(y))])
    clustered = explain_logit(m, list("abcd"), X, y, X[:200], [g])
    both = explain_logit(m, list("abcd"), X, y, X[:200], [np.arange(len(y)), g])
    assert clustered.se.mean() > 1.2 * naive.se.mean()
    np.testing.assert_allclose(both.se, np.maximum(naive.se, clustered.se))  # берётся большая из дисперсий
    far = explain_logit(m, list("abcd"), X, y, np.vstack([X.mean(axis=0), X.mean(axis=0) + 6]), [g])
    assert far.se[1] > 3 * far.se[0]


def test_explain_survives_constant_feature():
    X, y, g, t = _synth()
    X[:, 2] = 0.0  # флаг пропуска, который в обучении ни разу не горел
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X, y)
    ex = explain_logit(m, list("abcd"), X, y, X[:10], [g, t])
    assert np.isfinite(ex.se).all() and (ex.contrib[:, 2] == 0).all()


def _synth_war(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for gw in range(1, 61):
        war, risk = 0, rng.normal()
        for year in range(1989, 2016):
            x1, x2 = risk + rng.normal(), rng.normal()
            p = sigmoid(-2.2 + 0.9 * x1 + (2.5 if war else 0))
            nxt = int(rng.random() < p)
            rows.append({"gw": gw, "country": f"S{gw}", "year": year, "state_war25": war, "x1": x1, "x2": x2})
            war = nxt
    f = pd.DataFrame(rows)
    prev = f.groupby("gw").state_war25.shift(1).fillna(0)
    f["onset_state_war25"] = ((f.state_war25 == 1) & (prev == 0)).astype(int)
    return f


def test_war_export_matches_battle_forecast():
    """Пересчёт экспортера — та же модель, что `model_hist.forecast_asof`: вероятности совпадают
    у всех стран, и у мирных (модель начала), и у воюющих (модель продолжения)."""
    from seldon.domains.conflict import model_hist

    f, T, feats = _synth_war(), 2012, ["x1", "x2"]
    ref = model_hist.forecast_asof(f, T, feats, "state_war25").set_index("gw")
    recs = web_export.war_explain(f, T, "state_war25", feats)
    assert set(recs) == set(ref.index)
    for gw, r in recs.items():
        assert r["p"] == pytest.approx(ref.at[gw, "p"], abs=1e-10)
        assert r["part"] == ("onset" if ref.at[gw, "at_risk"] == 1 else "cont")
        assert r["lo"] < r["p"] < r["hi"]
    # будущее не влияет: испортили всё после T — разбор тот же
    bad = f.copy()
    bad.loc[bad.year > T, ["x1", "x2"]] = 99.0
    bad.loc[bad.year > T + 1, "state_war25"] = 1
    again = web_export.war_explain(bad, T, "state_war25", feats)
    for gw, r in recs.items():
        assert again[gw]["p"] == pytest.approx(r["p"], abs=1e-12) and again[gw]["hi"] == pytest.approx(r["hi"], abs=1e-12)


def test_fxa_export_matches_battle_forecast():
    """То же для годовой валютной логрегрессии: `fxa_explain` = `econ_long.forecast_asof`."""
    from seldon.domains.currency import econ_long
    from seldon.domains.currency.long_features import BASE, MISSING

    rng = np.random.default_rng(5)
    n_c, years = 70, range(1960, 2011)
    p = pd.DataFrame([{"iso3": f"C{c:02d}", "year": y} for c in range(n_c) for y in years])
    for b in BASE:
        p[b] = rng.normal(size=len(p))
    for m in MISSING:
        p[m] = (rng.random(len(p)) < 0.1).astype(int)
    p["deval_5"] = rng.integers(0, 3, len(p))
    p["y"] = (rng.random(len(p)) < sigmoid(-2.5 + 0.8 * p.infl_log + 0.5 * p.deval_prev)).astype(float)
    p["usable"] = True
    p["evaluable"] = rng.random(len(p)) < 0.9
    T = 2008
    ref = econ_long.forecast_asof(p, T, n_estimators=5).set_index("iso3")
    recs = web_export.fxa_explain(p, T)
    assert set(recs) == set(ref.index)
    for iso, r in recs.items():
        assert r["p"] == pytest.approx(ref.at[iso, "logit"], abs=1e-10)
    assert recs["C00"]["n_train"] == int((p.evaluable & (p.year <= T - 2)).sum())  # метка известна только по T − 2


def test_country_names_cover_showcase():
    risk = pd.read_csv(REPORTS / "risk_2025_26.csv")
    frozen = pd.read_csv(VINTAGE / "predictions.csv")
    assert set(COUNTRY_NAMES) == set(risk.iso3) == set(frozen.iso3) and len(COUNTRY_NAMES) == 174
    ru, en = [v[0] for v in COUNTRY_NAMES.values()], [v[1] for v in COUNTRY_NAMES.values()]
    assert len(set(ru)) == 174 and len(set(en)) == 174
    assert all(any("а" <= ch.lower() <= "я" or ch == "ё" for ch in name) for name in ru)  # русское — кириллицей
    assert all(name.isascii() or name == "Côte d’Ivoire" for name in en)
    assert all(ISO3_TO_RU[c["iso3"]] == c["short"] for c in POST_SOVIET)
    assert set(COUNTRY_ALIASES) <= set(COUNTRY_NAMES)


def test_feature_dictionary_is_complete():
    """У каждого признака четырёх логит-моделей и двух моделей на деревьях есть подпись,
    единица, источник и правило даты."""
    from seldon.domains.banking.econ_boe_em import FEATURE_SETS
    from seldon.domains.conflict.features import FEATURES as HIST
    from seldon.domains.conflict.features_pitf import PITF_FEATURES
    from seldon.domains.currency import econ_csd
    from seldon.domains.currency.kr_features import INDICATORS
    from seldon.domains.currency.long_features import FEATURES as FXA

    used = {*HIST, *PITF_FEATURES, *FXA, *INDICATORS, *econ_csd.features(econ_csd.CHALLENGER), *FEATURE_SETS["full"]}
    assert used <= set(web_export.FEATURE_META), sorted(used - set(web_export.FEATURE_META))
    for name in used:
        meta = web_export.FEATURE_META[name]
        assert meta["label"] and meta["unit"] and meta["src"] in web_export.SOURCES
    ctx = {"T": 2025, "m": pd.Period("2026-09", freq="M"), "macro_lag": 1, "res_lag": 2, "polity_year": 2018,
           "wdi:infant_mortality": 2024, "wdi:population": 2024, "reer_month": pd.Period("2026-08", freq="M"),
           "res_month": pd.Period("2026-07", freq="M")}
    for name in used:
        d, _ = web_export._obs_date(web_export.FEATURE_META[name]["when"], ctx)
        assert d is not None and d <= "2026-09"  # ни одно наблюдение не датировано будущим


def test_num_keeps_missing_as_null():
    assert web_export.num(float("nan")) is None and web_export.num(None) is None
    assert web_export.num(0.0) == 0.0 and web_export.num(0.00004999) == 0.0 and web_export.num(0.12345) == 0.1235


# ---------------------------------------------------------------------------
# 2. Собранный бандл
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bundle():
    manifest = DATA / "manifest.json"
    if not manifest.exists():
        pytest.skip("нет web/data — сначала `make web`")
    man = json.loads(manifest.read_text())
    for rel, sha in man["inputs"].items():
        # бандл лежит в git (решение Ангелины 13.11.2026): устаревший бандл — ошибка, а не пропуск
        assert _sha(PROJECT_ROOT / rel) == sha, f"web/data собран из другой версии {rel} — пересобери: `make web`"
    load = lambda name: json.loads((DATA / name).read_text())  # noqa: E731
    cur = load("current.json")
    return {"manifest": man, "current": cur, "rows": {r["iso3"]: r for r in cur["rows"]}, "load": load,
            "risk": pd.read_csv(REPORTS / "risk_2025_26.csv").set_index("iso3")}


# ключ оценки на сайте → колонки таблицы риска (вероятность, место)
EST = {
    "fxa": ("fxa_logit_2026_27", "rank_fxa_logit_2026_27"),
    "fxa_tree": ("fxa_extree_2026_27", "rank_fxa_extree_2026_27"),
    "fxa_check": ("fxa_logit_2025_26", "rank_fxa_logit_2025_26"),
    "fxm": ("fx_deval15_12m", "rank_fx_deval15_12m"),
    "fxm_live": ("fx_deval15_12m_live", "rank_fx_deval15_12m_live"),
    "bank": ("bank_jst_extree_full_2026_27", "rank_bank_jst_extree_full_2026_27"),
    "bank_check": ("bank_jst_extree_full_2025_26", "rank_bank_jst_extree_full_2025_26"),
}
LOGIT = {"war25_onset", "war25_cont", "war1000_onset", "war1000_cont", "fxa", "fxa_check", "fxm", "fxm_live"}
TREES = {"fxa_tree", "bank", "bank_check"}


def test_bundle_numbers_equal_risk_table(bundle):
    """Каждая вероятность и каждое место на сайте — ровно из `reports/risk_2025_26.csv`;
    где в таблице пусто — на сайте нет числа, а есть код причины."""
    risk, rows = bundle["risk"], bundle["rows"]
    assert set(rows) == set(risk.index) and len(rows) == 174
    for iso, r in risk.iterrows():
        e = rows[iso]["est"]
        for key, (col, rank) in EST.items():
            if pd.isna(r[col]):
                assert "p" not in e[key] and e[key]["status"], (iso, key)
            else:
                assert e[key]["p"] == pytest.approx(r[col], abs=1e-9), (iso, key)
                assert e[key]["rank"] == int(r[rank]) and e[key]["n"] == int(risk[col].notna().sum()), (iso, key)
        for short, target in (("war25", "state_war25"), ("war1000", "state_war1000")):
            now = int(r[f"{target}_now"])
            mine, other = (f"{short}_cont", f"{short}_onset") if now else (f"{short}_onset", f"{short}_cont")
            assert e[mine]["p"] == pytest.approx(r[f"p_{target}_2026"], abs=1e-9), (iso, short)
            assert e[other] == {"status": "other_group" if now else "no_conflict"}, (iso, short)
            if not now:  # место начала — опубликованное
                assert e[mine]["rank"] == int(r[f"rank_{target}_at_risk"]) and e[mine]["n"] == int((risk[f"{target}_now"] == 0).sum())
        assert rows[iso]["bank_status"] == r.bank_status and rows[iso]["fx_regime"] == r.fx_regime
        assert rows[iso]["war_now"]["state_war25"] == int(r.state_war25_now)


def test_bundle_has_no_false_zeros(bundle):
    """Общая валюта, отсутствие ряда и отсутствие банковской оценки — это статус, а не ноль."""
    risk, rows, cur = bundle["risk"], bundle["rows"], bundle["current"]
    shared = risk[~risk.fx_regime.isin(["own", "no_imf_fx"])]
    assert len(shared) == cur["coverage"]["fx"]["not_applicable"] > 0
    for iso, r in shared.iterrows():
        for key in ("fxa", "fxa_tree", "fxm", "fxm_live", "stress"):
            assert rows[iso]["est"][key] == {"status": f"shared:{r.fx_regime}"}, (iso, key)
        assert "indicators" not in rows[iso]
    for iso in risk[risk.fx_regime == "no_imf_fx"].index:
        assert rows[iso]["est"]["fxa"] == {"status": "no_imf_fx"} and rows[iso]["est"]["fxm"] == {"status": "no_imf_fx"}
    for iso, r in risk[risk.bank_status != "ok"].iterrows():
        assert rows[iso]["est"]["bank"] == {"status": r.bank_status}
    assert rows["EST"]["est"]["fxa"] == {"status": "shared:EUR"} and rows["TKM"]["est"]["fxa"] == {"status": "no_imf_fx"}
    assert rows["RUS"]["est"]["bank"] == {"status": "no_yield_curve"} and rows["KAZ"]["est"]["bank"] == {"status": "no_bis_credit"}
    assert rows["RUS"]["est"]["war25_onset"] == {"status": "other_group"}
    cov = cur["coverage"]
    for dom in ("war", "fx", "bank"):
        assert cov[dom]["estimated"] + cov[dom]["not_applicable"] + cov[dom]["no_estimate"] == 174
    assert cov["bank"]["estimated"] == int((risk.bank_status == "ok").sum())
    assert cov["bank"]["by_status"] == risk.bank_status.value_counts().to_dict()
    # все коды причин, встретившиеся в данных, есть в словаре состояний
    used = {e["status"] for r in rows.values() for e in r["est"].values() if "status" in e}
    assert {u for u in used if not u.startswith("shared:")} <= set(cur["status_text"])
    # у описательных индикаторов пропуск — null вместе с датой
    for r in rows.values():
        for k, v in r.get("indicators", {}).items():
            if isinstance(v, dict) and v["value"] is None:
                assert v["date"] is None, (r["iso3"], k)


def test_bundle_intervals_only_for_logit_models(bundle):
    """Интервал есть у логит-моделей и накрывает оценку; у деревьев его нет вовсе."""
    for iso, row in bundle["rows"].items():
        for key, e in row["est"].items():
            if "p" not in e:
                assert "lo" not in e and "hi" not in e
            elif key in TREES:
                assert "lo" not in e and "hi" not in e, (iso, key)
            elif key in LOGIT:
                assert e["lo"] <= e["p"] <= e["hi"] and e["hi"] - e["lo"] < 1, (iso, key)
    ctx = bundle["current"]["contexts"]
    assert all(ctx[k]["interval"] == "none" for k in ("fxa_tree", "bank", "bank_check", "stress"))
    assert ctx["stress"]["observed"] and "не вероятность" in ctx["stress"]["event"]
    assert ctx["fxm_live"]["challenger"]


def test_bundle_factors_are_exact_log_odds(bundle):
    """Вклады в досье складываются в лог-шансы прогноза и дают опубликованную вероятность;
    у моделей на деревьях вкладов нет — только входные наблюдения."""
    risk = bundle["risk"]
    col = {"war25": "p_state_war25_2026", "war1000": "p_state_war1000_2026", "fxa": "fxa_logit_2026_27",
           "fxm": "fx_deval15_12m", "fxm_live": "fx_deval15_12m_live"}
    for iso in ("KAZ", "RUS", "UKR", "EST", "TKM", "USA", "TUR", "ARG", "LBN", "DEU"):
        d = bundle["load"](f"country/{iso}.json")
        for key, c in col.items():
            if pd.isna(risk.at[iso, c]):
                assert key not in d["factors"], (iso, key)
                continue
            b = d["factors"][key]
            assert b["kind"] == "logit" and "лог-шанс" in b["scale"]
            eta = b["intercept"] + sum(f["contrib"] for f in b["features"])
            assert eta == pytest.approx(b["eta"], abs=2e-3), (iso, key)
            assert 1 / (1 + math.exp(-b["eta"])) == pytest.approx(risk.at[iso, c], abs=2e-4), (iso, key)
            for f in b["features"]:
                # все три числа округлены до 4 знаков — допуск на округление сомножителей
                tol = 1e-4 + 6e-5 * (abs(f["beta"]) + abs(f["z"]))
                assert f["contrib"] == pytest.approx(f["beta"] * f["z"], abs=tol)
                assert (f["value"] is None) == ("missing" in f), (iso, key, f["key"])
                if f["value"] is None:
                    assert f["date"] is None and f["natural"] is None, (iso, key, f["key"])
                elif not f["key"].endswith("_missing"):  # у флага пропуска даты наблюдения может не быть
                    assert f["date"] is not None, (iso, key, f["key"])
        for key in ("fxa_tree", "bank"):
            if key in d["inputs"]:
                assert d["inputs"][key]["kind"] == "trees" and all("contrib" not in f for f in d["inputs"][key]["inputs"])
        assert ("bank" in d["inputs"]) == (risk.at[iso, "bank_status"] == "ok")


def test_bundle_feature_dates_follow_publication_lags(bundle):
    """Даты наблюдений — по `core/temporal.py`: годовые макроданные — за T−1, декабрьский курс —
    за T, месячные резервы МВФ — за as_of − 2 (или дотянутое значение — тогда ещё раньше)."""
    cur, risk = bundle["current"], bundle["risk"]
    T_war = int(risk.war_features_year.iloc[0])
    T_fx = cur["contexts"]["fxa"]["features_year"]
    macro = str(T_fx - temporal.publication_lag("gmd_annual", "jan1"))
    res_lag = temporal.monthly_lag("imf_reserves")
    assert (T_fx, macro, res_lag) == (2025, "2024", 2)
    for iso in ("KAZ", "RUS", "UKR", "TUR", "USA", "GEO", "UZB"):
        d = bundle["load"](f"country/{iso}.json")
        if "fxa" in d["factors"]:
            dates = {f["key"]: f["date"] for f in d["factors"]["fxa"]["features"] if f["date"]}
            assert {dates[k] for k in ("deval_prev", "deval_5", "world_deval", "peg_years_log")} == {f"{T_fx}-12"}
            assert {v for k, v in dates.items() if k in ("infl_log", "infl_d2", "gdp_growth_2", "debt_d2", "ca_level", "money_growth_2")} <= {macro}
            tree = {f["key"]: f["date"] for f in d["inputs"]["fxa_tree"]["inputs"] if f["date"]}
            assert tree.get("ca_level", macro) == macro and tree["deval_prev"] == f"{T_fx}-12"
        w = {f["key"]: f["date"] for f in d["factors"]["war25"]["features"]}
        assert w["log_deaths_1"] == str(T_war) and w["nb_war_share"] == str(T_war)
        assert w["log_pop"] is None or w["log_pop"] <= str(T_war - temporal.publication_lag("wdi_annual", "jan1"))
        assert w["polity2"] is None or w["polity2"] <= "2020"  # Polity5 кончается раньше — значение продлено и это видно
        if "fxm" in d["factors"]:
            as_of = pd.Period(risk.at[iso, "as_of"], freq="M")
            m = {f["key"]: f for f in d["factors"]["fxm"]["features"]}
            assert m["deval12"]["date"] == str(as_of)
            for k in ("reserves_g12", "reserves_g1", "m2_res_g12"):
                assert m[k]["date"] in (None, str(as_of - res_lag)), (iso, k)
                assert (m[k]["date"] is None) == ("missing" in m[k])
            live = {f["key"]: f for f in d["factors"]["fxm_live"]["features"]}
            if live["peg_resfall"]["date"]:
                assert live["peg_resfall"]["date"] <= str(as_of - res_lag)
            if live["reer_gap"]["date"]:
                assert live["reer_gap"]["date"] <= str(as_of)
        ind = bundle["rows"][iso].get("indicators")
        if ind and ind["reserves_g12"]["value"] is not None:
            assert ind["reserves_g12"]["date"] == str(pd.Period(risk.at[iso, "as_of"], freq="M") - res_lag)


def test_bundle_scorecard_equals_report(bundle):
    score = pd.read_csv(REPORTS / "scorecard.csv")
    rows = bundle["load"]("scorecard.json")["rows"]
    assert len(rows) == len(score)
    for a, (_, b) in zip(rows, score.iterrows()):
        assert (a["block"], a["sample"], a["model"], a["n"], a["positives"]) == (b.block, b["sample"], b.model, b.n, b.positives)
        for k in ("auc", "auc_lo", "auc_hi", "auc_within_year", "brier", "auc_vs_ref"):
            assert (a[k] is None and pd.isna(b[k])) or a[k] == pytest.approx(b[k], abs=1e-12)


def test_bundle_ribbon_matches_saved_blind_predictions(bundle):
    """Лента рангов — те же прогнозы вслепую, по которым считается `scorecard`."""
    score = pd.read_csv(REPORTS / "scorecard.csv")
    war = bundle["load"]("oos/war25.json")
    ref = score[(score.model == "pitf_state_war25")].iloc[0]
    assert sum(y["n"] for y in war["years"]) == ref.n and sum(y["events"] for y in war["years"]) == ref.positives
    fxa = bundle["load"]("oos/fxa.json")
    ref = score[(score.model == "logit_long") & (score["sample"] == "мир")].iloc[0]
    assert sum(y["n"] for y in fxa["years"]) == ref.n and sum(y["events"] for y in fxa["years"]) == ref.positives
    for rib in (war, fxa):
        for y in rib["years"]:
            ps = [r[1] for r in y["rows"]]
            assert ps == sorted(ps, reverse=True) and y["rows"][0][4] == 1 and len(y["rows"]) == y["n"]
            assert all(r[2] <= r[1] <= r[3] for r in y["rows"]) and {r[5] for r in y["rows"]} <= {0, 1}
            assert int(y["as_of"][:4]) == y["release"] + 1  # прогноз «на 1 января» года после года признаков
    for c in bundle["current"]["cases"]:
        yr = next(y for y in (war if c["contour"] == "war25" else fxa)["years"] if y["release"] == c["release"])
        row = next(r for r in yr["rows"] if r[0] == c["iso3"])
        assert (row[1], row[4], row[5], yr["n"]) == (c["p"], c["rank"], c["outcome"], c["n"])
    det = bundle["load"]("oos/war25/2019.json")
    aze = det["rows"]["AZE"]
    assert det["intercept"] + sum(aze["c"]) == pytest.approx(aze["eta"], abs=2e-3) and len(aze["c"]) == len(det["features"])


def test_bundle_archive_reproduces_frozen_vintage(bundle):
    """Архив показывает снимок `forecasts/2026-11/` как он был опубликован; сам снимок не тронут.
    Исход стоит только там, где целевой период закончился и есть наблюдения; иначе — статус."""
    meta = json.loads((VINTAGE / "metadata.json").read_text())
    assert _sha(VINTAGE / "predictions.csv") == meta["predictions"]["sha256"]  # снимок неизменяем
    pred = pd.read_csv(VINTAGE / "predictions.csv").set_index("iso3")
    v = next(x for x in bundle["load"]("archive.json")["vintages"] if x["vintage"] == "2026-11")
    assert v["predictions_sha256"] == meta["predictions"]["sha256"] and v["commit"] == meta["code"]["commit"]
    today = bundle["manifest"]["today"]
    cols = {m["column"] for m in v["models"]}
    assert {"p_state_war25_2025", "fxa_logit_2026_27", "fx_deval15_12m", "bank_jst_extree_full_2026_27"} <= cols
    assert "p_state_war25_2026" not in cols  # окна разных выпусков не склеиваются
    for c in cols:
        recs = v["records"][c]
        assert set(recs) == set(pred.index[pred[c].notna()]), c
        for iso, r in recs.items():
            assert r["p"] == pytest.approx(pred.at[iso, c], abs=1e-9)
            assert ("outcome" in r) == (r["status"] == "resolved")
            if c.startswith(("fxa_", "bank_")):
                assert r["status"] == ("ongoing" if today <= "2027-12-31" else r["status"])
    war = next(m for m in v["models"] if m["column"] == "p_state_war25_2025")
    assert war["interval"] == "no_interval" and "retrospective_note" in war and war["outcome_source"]["source"].startswith("UCDP")
    assert any("p_state_war25_2025" in str(n) for n in v["schema_notes"])
    risk = bundle["risk"]
    for iso, r in v["records"]["p_state_war25_2025"].items():
        assert r["outcome"] == int(risk.at[iso, "state_war25_now"])


# ---------------------------------------------------------------------------
# 3. Статический сайт (чекпоинт 5.2)
# ---------------------------------------------------------------------------

WEB = PROJECT_ROOT / "web"
JS = sorted((WEB / "js").rglob("*.js"))


def test_site_is_static_and_self_contained():
    """Сайт — статические файлы без сборки и без сервера: страница, стили, ES-модули. Модули
    импортируют только друг друга, данные берут только из `data/` (относительные пути — сайт
    работает и из подпапки GitHub Pages)."""
    import re

    html = (WEB / "index.html").read_text()
    assert 'src="js/app.js"' in html and 'href="css/styles.css"' in html and 'lang="ru"' in html
    for label in ("Прогнозы", "Проверка", "Архив", "Метод и данные", "Найти страну"):
        assert label in html
    assert "Страны</a>" not in html  # отдельного пункта «Страны» в шапке нет
    names = {p.relative_to(WEB / "js").as_posix() for p in JS}
    assert {"app.js", "lib.js", "views/home.js", "views/forecasts.js", "views/country.js", "views/archive.js"} <= names
    for p in JS:
        src = p.read_text()
        for target in re.findall(r'from "([^"]+)"', src):
            assert target.startswith("."), (p.name, target)  # никаких внешних библиотек
            assert (p.parent / target).resolve().exists(), (p.name, target)
        assert "http://" not in src.replace("http://www.w3.org/2000/svg", "") and "localhost" not in src
        assert 'fetch("/' not in src and "fetch(`/" not in src  # только относительные пути


def test_site_state_dictionary_covers_export():
    """Единый словарь состояний: у каждого кода причины из экспортера есть вид оформления на
    сайте, а обязательные тексты раздела 6 ТЗ совпадают дословно."""
    lib = (WEB / "js" / "lib.js").read_text()
    for code in web_export.STATUS_TEXT:
        assert f"{code}:" in lib, code
    required = ["Нет ряда BIS", "Нет валютного ряда", "Нет рыночного ряда", "Нет ряда денежной массы",
                "Идёт конфликт · посмотреть продолжение", "Интервал не рассчитан", "Целевой период ещё идёт",
                "Ожидаются данные об исходе"]
    assert set(required) <= set(web_export.STATUS_TEXT.values())
    assert "· неприменимо" in lib and "страна использует" in lib
    assert set(web_export.STATUS_DETAIL) <= set(web_export.STATUS_TEXT)


def test_site_wording_keeps_honesty_rules():
    """Формулировки, без которых экран вводил бы в заблуждение."""
    text = "\n".join(p.read_text() for p in JS)
    for phrase in ("Ретроспективная OOS-проверка", "Положение в рейтинге не равно вероятности события",
                   "Отсутствие оценки не означает низкий риск", "не устанавливают причины будущего события",
                   "Единого индекса кризиса здесь нет", "модель-претендент", "Это не вероятность кризиса",
                   "Они не были публичными прогнозами того времени"):
        assert phrase in text, phrase
    assert "лог-шанс" in text and "процентных пункт" in text  # вклады подписаны как лог-шансы
    assert "Math.random" not in text  # никаких случайных или «учебных» чисел


def test_site_javascript_parses():
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("нет node — синтаксис модулей не проверен")
    for p in JS:
        r = subprocess.run([node, "--check", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, f"{p.name}: {r.stderr[-400:]}"


def test_site_reads_only_exported_files(bundle):
    """Каждый файл данных, к которому обращается сайт, есть в бандле."""
    import re

    files = set(bundle["manifest"]["files"]) | {"manifest.json"}
    text = "\n".join(p.read_text() for p in JS)
    fixed = set(re.findall(r'load\("([a-z_/]+\.json)"\)', text)) | set(re.findall(r'"([a-z]+\.json)"', text))
    assert {"current.json", "countries.json", "archive.json", "manifest.json", "scorecard.json", "meta.json"} <= fixed
    assert fixed <= files, fixed - files
    assert {f"country/{iso}.json" for iso in bundle["rows"]} <= files
    for rib in ("war25", "fxa"):
        assert f"oos/{rib}.json" in files
        for y in bundle["load"](f"oos/{rib}.json")["years"]:
            assert f"oos/{rib}/{y['release']}.json" in files


# ---------------------------------------------------------------------------
# 4. «Проверка», вид P4 и мобильная версия (чекпоинт 5.3)
# ---------------------------------------------------------------------------


def test_check_helpers_on_synthetic():
    """Калибровка, разность Brier и верхние k — на синтетике с известным ответом."""
    rng = np.random.default_rng(7)
    n = 4000
    p = rng.uniform(0.01, 0.6, n)
    y = (rng.random(n) < p).astype(float)  # идеально откалиброванная модель
    bins = web_export._reliability(p, y)
    assert sum(b["n"] for b in bins) == n and sum(b["events"] for b in bins) == int(y.sum())
    assert all(b["lo"] <= b["freq"] <= b["hi"] for b in bins)
    assert sum(b["lo"] <= b["p"] <= b["hi"] for b in bins) >= len(bins) - 1  # частота накрывает оценку
    assert [b["p"] for b in bins] == sorted(b["p"] for b in bins)
    d = pd.DataFrame({"y": y, "model": p, "base": y.mean(), "year": rng.integers(2000, 2020, n), "release": rng.integers(0, 40, n)})
    b = web_export._brier_vs(d, "model", "base")
    assert b["diff"] > 0 and b["lo"] > 0 and b["brier_model"] < b["brier_base"]  # модель лучше константы
    worse = web_export._brier_vs(d.assign(noise=rng.random(n)), "noise", "base")
    assert worse["diff"] < 0 and worse["hi"] < 0
    top = web_export._top_k(d, "model")
    shares = [t["share"] for t in top]
    assert shares == sorted(shares) and all(t["share"] > t["random"] for t in top)
    assert web_export._wilson(0, 50)[0] == 0 and web_export._wilson(50, 50)[1] == pytest.approx(1)


def test_bundle_check_matches_scorecard(bundle):
    """Экран «Проверка» считает по тем же прогнозам вслепую, что `scorecard`: число оценок,
    событий и AUC совпадают; калибровочные группы покрывают все оценки."""
    ck = bundle["load"]("check.json")
    score = pd.read_csv(REPORTS / "scorecard.csv")
    ref = {"war25": ("pitf_state_war25", "мир, страны без войны в T"), "fxa": ("logit_long", "мир"), "fxm": ("logit", "мир"),
           "fxm_live": ("live_fx_swap", "мир"), "bank": ("jst_extree", "страны BIS вне JST")}
    for key, (model, sample) in ref.items():
        cell = ck["contours"][key]["cells"]["world|all"]
        r = score[(score.model == model) & (score["sample"] == sample)].iloc[0]
        assert (cell["n"], cell["events"]) == (r.n, r.positives), key
        assert cell["auc"]["pooled"] == pytest.approx(r.auc, abs=1e-4) and cell["auc"]["lo"] == pytest.approx(r.auc_lo, abs=1e-4)
        assert cell["auc"]["within_year"] == pytest.approx(r.auc_within_year, abs=1e-4)
    for key, c in ck["contours"].items():
        for name, cell in c["cells"].items():
            assert sum(b["n"] for b in cell["reliability"]) == cell["n"], (key, name)
            assert sum(b["events"] for b in cell["reliability"]) == cell["events"], (key, name)
            for b in cell["brier"]:
                assert b["lo"] <= b["diff"] <= b["hi"] and b["diff"] == pytest.approx(b["brier_base"] - b["brier_model"], abs=2e-5)
            assert [t["k"] for t in cell["top_k"]] == ck["k"]
            assert all(0 <= t["share"] <= 1 and t["caught"] <= t["events"] for t in cell["top_k"])
    # слабое место не спрятано: месячная модель на постсоветских не лучше базы
    ps = ck["contours"]["fxm"]["cells"]["ps|all"]
    assert ps["auc"]["pooled"] < 0.5 and ps["brier"][0]["diff"] < 0
    for rib in ("fxm", "bank"):
        r = bundle["load"](f"oos/{rib}.json")
        assert r["details"] is False and all(row[2] is None for y in r["years"] for row in y["rows"])  # интервала в этих лентах нет


def test_site_check_p4_and_mobile():
    """Формулировки и поведение чекпоинта 5.3 на месте."""
    text = "\n".join(p.read_text() for p in JS)
    for phrase in ("Посмотреть выпуск без знания исходов", "Показать исходы", "Соответствуют ли вероятности наблюдаемым частотам?",
                   "Лучше ли модель наивной базы?", "Сколько событий попадает в поле внимания?", "Диагональ — совпадение вероятности и частоты",
                   "Выше нуля — меньшая ошибка у модели", "Исследовательский порог. Он не меняет исторически опубликованные сигналы",
                   "Дополнительные метрики", "Публичные выпуски", "Структурная уязвимость и текущее давление",
                   "Текущее валютное давление, σ", "Вероятность годового обвала курса, %"):
        assert phrase in text, phrase
    check = (WEB / "js" / "views" / "check.js").read_text()
    assert "blind" in check and "Слой исходов скрыт" in check
    lib = (WEB / "js" / "lib.js").read_text()
    assert "r[5] && !blind" in lib  # в режиме без исходов отметки событий на ленте не рисуются
    css = (WEB / "css" / "styles.css").read_text()
    assert "@media (max-width: 720px)" in css and ".ctx-col { display: none; }" in css and "tr.exp { display: table-row; }" in css
