"""Золотой тест проекта: ни один прогноз не видит будущего.

Запуск:
    uv run pytest

Три проверки (все — на маленьких синтетических панелях, секунды):

1. `test_future_blindness_*` — для каждого контура (войны годовые и месячные, месячный
   валютный EMP/KLR, годовой валютный, банковский BoE) считаем прогноз в точке `as_of`,
   затем портим ВСЕ входные данные строго после `as_of` (шум в тысячи раз и пропуски),
   заново строим признаки и прогноз — он обязан совпасть до 1e-12.
   `test_harness_catches_leak` — контроль самого теста: заведомо «текущая» функция
   (нормировка по всей выборке) обязана этим же способом ловиться.
2. `test_target_evaluable_sync_*` — у строки с `evaluable == True` целевая не может быть
   принудительно занулена внутренним правилом «тишины после кризиса».
3. `test_entity_mapping` — коды стран и валютные союзы в `core/entities.py`.
4. `test_available_from_*` — лаги публикации (чекпоинт 2.2): ни один признак не использует
   значение раньше его `available_from`. Портим всё, что к дате прогноза ещё НЕ ВЫШЛО
   (годовые макроданные за текущий год, список сторон войны за текущий год), — прогноз
   обязан совпасть. Контроль: без лага та же порча прогноз меняет.

Глобальные константы модулей тесты не трогают: всё передаётся аргументами.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seldon.core import entities, temporal
from seldon.domains.banking import econ_boe, econ_boe_em
from seldon.domains.conflict import features, features_pitf, model_hist, panel
from seldon.domains.currency import econ_kr, econ_long, kr_features, kr_targets, long_features

ATOL = 1e-12


def _noise(rng, n):
    """Экстремальный шум: множители от 1e-3 до 1e3."""
    return np.exp(rng.normal(0, 3, n))


def _corrupt(df: pd.DataFrame, future: pd.Series, cols: list[str], seed: int = 99) -> pd.DataFrame:
    """Строки `future`: числовые колонки × шум, каждая пятая ячейка — пропуск."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    idx = out.index[future.to_numpy()]
    for c in cols:
        v = out.loc[idx, c].astype(float).to_numpy() * _noise(rng, len(idx))
        v[rng.random(len(idx)) < 0.2] = np.nan
        out[c] = out[c].astype(float)
        out.loc[idx, c] = v
    return out


def _same(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> None:
    assert len(a) == len(b) and len(a) > 0
    for c in cols:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        assert np.allclose(x, y, atol=ATOL, rtol=0, equal_nan=True), f"прогноз `{c}` изменился от будущих данных"


# ---------------------------------------------------------------------------
# Синтетические данные
# ---------------------------------------------------------------------------

KR_COUNTRIES = ["USA", "AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "KKK"]
KR_COLS = ["fx", "reserves", "cpi", "rate", "lending", "deposit", "broad_money", "exports"]


def synth_monthly(seed: int = 1):
    rng = np.random.default_rng(seed)
    months = pd.period_range("1993-01", "2014-12", freq="M")
    rows = []
    for c in KR_COUNTRIES:
        n = len(months)
        jumps = (rng.random(n) < 0.012) * rng.uniform(0.15, 0.6, n)
        fx = 10 * np.exp(np.cumsum(rng.normal(0.002, 0.012, n) + jumps))
        res = 1e3 * np.exp(np.cumsum(rng.normal(0.003, 0.04, n) - 0.5 * jumps))
        cpi = 100 * np.exp(np.cumsum(rng.normal(0.004, 0.004, n) + 0.3 * jumps))
        if c not in ("USA", "AAA"):  # у части стран месячного ИПЦ нет в начале — заполняется годовым
            cpi[: int(rng.integers(40, 120))] = np.nan
        rate = 8 + rng.normal(0, 2, n).cumsum() * 0.1
        rows.append(pd.DataFrame({
            "iso3": c, "month": months, "fx": fx, "reserves": res, "cpi": cpi, "rate": rate,
            "lending": rate + 4 + rng.normal(0, 0.5, n), "deposit": rate - 1 + rng.normal(0, 0.5, n),
            "broad_money": fx * res * np.exp(rng.normal(0, 0.05, n)), "exports": 100 * np.exp(rng.normal(0, 0.1, n)),
        }))
    raw = pd.concat(rows, ignore_index=True)
    raw["post_soviet"] = False
    years = range(1990, 2015)
    annual = pd.DataFrame([(c, y, 50 * 1.07 ** (y - 1990) * (1 + 0.1 * rng.random())) for c in KR_COUNTRIES for y in years],
                          columns=["iso3", "year", "value"])
    war = raw[["iso3", "month"]].copy()
    war["war25_now"] = (rng.random(len(war)) < 0.03).astype(int)
    war["war1000_now"] = (rng.random(len(war)) < 0.01).astype(int)
    war["war_onset12"] = (rng.random(len(war)) < 0.05).astype(int)
    return raw, annual, war


def synth_annual_fx(seed: int = 2):
    rng = np.random.default_rng(seed)
    years = np.arange(1950, 2016)
    g, d = [], []
    for i in range(40):
        iso = f"C{i:02d}"
        n = len(years)
        big = (rng.random(n) < 0.07) * rng.uniform(0.3, 1.0, n)
        fx = np.exp(np.cumsum(rng.normal(0.01, 0.03, n) + big))
        g.append(pd.DataFrame({
            "ISO3": iso, "countryname": iso, "year": years.astype(float), "USDfx": fx,
            "infl": 5 + 40 * big + rng.normal(0, 2, n), "rGDP": np.exp(np.cumsum(rng.normal(0.03, 0.03, n))),
            "govdebt_GDP": 40 + rng.normal(0, 3, n).cumsum(), "CA_GDP": rng.normal(-2, 4, n),
            "M2": np.exp(np.cumsum(rng.normal(0.08, 0.05, n))), "M3": np.nan, "M0": np.nan,
            "nGDP": np.exp(np.cumsum(rng.normal(0.07, 0.04, n))),
            "CurrencyCrisis": (big > 0).astype(float),
        }))
        d.append(pd.DataFrame({"iso3": iso, "year": years, "fx_dec": fx * np.exp(rng.normal(0, 0.01, n))}))
    lv = pd.DataFrame({"iso3": "C00", "year": years, "lv_currency": 0})
    return pd.concat(g, ignore_index=True), pd.concat(d, ignore_index=True), lv


def synth_jst(n_countries: int, seed: int, prefix: str):
    rng = np.random.default_rng(seed)
    years = np.arange(1950, 2016)
    rows = []
    for i in range(n_countries):
        n = len(years)
        gdp = 100 * np.exp(np.cumsum(rng.normal(0.05, 0.03, n)))
        boom = rng.normal(0, 0.03, n).cumsum()
        crisis = ((np.r_[0, 0, np.diff(boom, 2)] > 0.03) & (rng.random(n) < 0.5)).astype(float)
        cpi = 100 * np.exp(np.cumsum(rng.normal(0.03, 0.02, n)))
        stir = 5 + rng.normal(0, 1, n)
        df = pd.DataFrame({
            "iso": f"{prefix}{i:02d}", "year": years, "country": f"{prefix}{i:02d}", "gdp": gdp,
            "tloans": gdp * (0.6 + boom), "money": gdp * (0.7 + rng.normal(0, 0.02, n).cumsum()),
            "stir": stir, "ltrate": stir + rng.normal(1, 1, n), "cpi": cpi,
            "rconpc": np.exp(np.cumsum(rng.normal(0.02, 0.02, n))), "iy": 0.2 + rng.normal(0, 0.02, n),
            "debtgdp": 0.5 + rng.normal(0, 0.03, n).cumsum(), "ca": gdp * rng.normal(0, 0.03, n),
            "crisisJST": crisis,
        })
        for c in ("narrowm", "tbus", "thh", "tmort", "stocks", "hpnom"):
            df[c] = np.nan
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


WAR_STATES = list(range(1, 21))


def synth_war(seed: int = 4):
    rng = np.random.default_rng(seed)
    y0, y1 = 1989, 2013
    states = pd.DataFrame({"gw": WAR_STATES, "gw_name": [f"S{g}" for g in WAR_STATES], "start_year": y0, "end_year": y1})
    neighbors = {(a, b, y) for a in WAR_STATES for b in WAR_STATES if abs(a - b) == 1 for y in range(y0, y1 + 1)}
    ev, parties = [], []
    for cid in range(1, 15):  # конфликт cid идёт на территории страны loc, эпизодами
        loc = int(rng.integers(1, 21))
        helper = loc + 1 if loc < 20 else loc - 1
        active = False
        for y in range(y0, y1 + 1):
            active = (rng.random() < 0.85) if active else (rng.random() < 0.15)
            if not active:
                continue
            start = int(rng.integers(1, 13))  # в первый год эпизода бои начинаются не с января
            parties.append({"conflict_id": cid, "year": y, "parties": [loc, helper] if rng.random() < 0.5 else [loc]})
            for m in range(start, 13):
                for _ in range(int(rng.integers(0, 4))):
                    ev.append({"gw": loc, "country": f"S{loc}", "year": y, "month": m, "type_of_violence": 1,
                               "best": int(rng.integers(1, 80)), "conflict_new_id": cid})
    for _ in range(300):  # негосударственное насилие
        ev.append({"gw": int(rng.integers(1, 21)), "country": "x", "year": int(rng.integers(y0, y1 + 1)),
                   "month": int(rng.integers(1, 13)), "type_of_violence": int(rng.integers(2, 4)),
                   "best": int(rng.integers(1, 60)), "conflict_new_id": 1000 + int(rng.integers(0, 30))})
    ged = pd.DataFrame(ev)
    ged["deaths_state"] = ged.best.where(ged.type_of_violence == 1, 0)
    return ged, states, pd.DataFrame(parties), neighbors, (y0, y1)


# ---------------------------------------------------------------------------
# 1. Слепота к будущему
# ---------------------------------------------------------------------------


def test_harness_catches_leak():
    """Контроль: нормировка по ВСЕЙ выборке (так было до аудита) обязана ловиться."""
    raw, _, _ = synth_monthly()
    as_of = pd.Period("2010-06", freq="M")

    def leaky(d):
        z = d.groupby("iso3").fx.transform(lambda s: (s - s.mean()) / s.std())
        return z[d.month == as_of].to_numpy()

    bad = _corrupt(raw, raw.month > as_of, KR_COLS)
    assert not np.allclose(leaky(raw), leaky(bad), atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("target", ["crisis", "deval15"])
def test_future_blindness_fx_monthly(target):
    raw, annual, war = synth_monthly()
    as_of = pd.Period("2010-06", freq="M")
    models = ["base_rate", "pressure", "signals", "logit"]

    def run(raw, annual, war):
        p = econ_kr.build_panel(raw, annual, war, target=target)
        return econ_kr.forecast_asof(p, as_of).sort_values("iso3")

    clean = run(raw, annual, war)
    bad = run(_corrupt(raw, raw.month > as_of, KR_COLS),
              _corrupt(annual, annual.year >= as_of.year, ["value"]),
              _corrupt(war, war.month > as_of, ["war25_now", "war1000_now", "war_onset12"]))
    assert clean[models].notna().all().all()
    assert clean.logit.nunique() > 1 and clean.emp_z_max3.nunique() > 1
    _same(clean, bad, models + list(kr_features.INDICATORS))


@pytest.mark.parametrize("target", ["deval25", "gmd_cc"])
def test_future_blindness_fx_annual(target):
    gmd, dec, lv = synth_annual_fx()
    T = 2000
    models = ["base_rate", "persist", "logit", "extree"]

    def run(gmd, dec):
        p = econ_long.build_panel(gmd, dec, lv, target=target, last_year=2015)
        return econ_long.forecast_asof(p, T, n_estimators=40).sort_values("iso3")

    clean = run(gmd, dec)
    num = [c for c in gmd.columns if c not in ("ISO3", "countryname", "year")]
    bad = run(_corrupt(gmd, gmd.year > T, num), _corrupt(dec, dec.year > T, ["fx_dec"]))
    assert clean[models].notna().all().all() and clean.extree.nunique() > 1
    _same(clean, bad, models + long_features.FEATURES)


def test_future_blindness_banking():
    jst, own = synth_jst(10, 3, "J"), synth_jst(12, 5, "E")
    T = 2000
    feats = econ_boe_em.NOMONEY
    num = [c for c in jst.columns if c not in ("iso", "year", "country")]
    models = ["base_rate", "jst_logreg", "jst_extree", "own_logreg", "own_extree", "pooled_logreg", "pooled_extree"]

    def run(jst, own):
        data = econ_boe.build_boe_dataset(own, predictors=feats)
        jst_d = econ_boe.build_boe_dataset(jst, predictors=feats)
        return econ_boe_em.forecast_asof(data, jst_d, feats, T, n_estimators=40).sort_values("iso")

    clean = run(jst, own)
    bad = run(_corrupt(jst, jst.year > T, num), _corrupt(own, own.year > T, num, seed=7))
    assert clean[models].notna().all().all() and clean.jst_extree.nunique() > 1
    _same(clean, bad, models)


def test_future_blindness_boe_forecast_window():
    """`econ_boe.forecast`: прогноз года T+1 учится только на строках, чья метка (t+1, t+2) уже известна."""
    jst = synth_jst(10, 3, "J")
    data = econ_boe.build_boe_dataset(jst, predictors=econ_boe_em.NOMONEY)
    used = econ_boe.next_year_training_window(data, 2000)
    assert len(used) > 100 and used.year.max() == 2000 - 1


def _war_panels(ged, states, parties, neighbors, span):
    year = panel.build(ged, states, parties, neighbors, monthly=False, year_from=span[0], year_to=span[1])
    month = panel.build(ged, states, parties, neighbors, monthly=True, year_from=span[0], year_to=span[1])
    return year, month


def test_future_blindness_war_annual():
    ged, states, parties, neighbors, span = synth_war()
    T = 2008

    def run(ged, parties):
        year, _ = _war_panels(ged, states, parties, neighbors, span)
        f = features.build_features(year, neighbors, target="state_war25")
        return model_hist.forecast_asof(f, T, features.FEATURES, target="state_war25").sort_values("gw")

    clean = run(ged, parties)
    bad_ged = _corrupt(ged, ged.year > T, ["best", "deaths_state"])
    bad_ged[["best", "deaths_state"]] = bad_ged[["best", "deaths_state"]].fillna(5000).round()
    bad_parties = parties.copy()
    bad_parties.loc[bad_parties.year > T, "parties"] = bad_parties.loc[bad_parties.year > T, "parties"].map(lambda _: WAR_STATES)
    bad = run(bad_ged, bad_parties)
    assert clean.p.nunique() > 1
    _same(clean, bad, ["p", "p_onset"])


def test_future_blindness_war_monthly():
    """Месячные признаки войны в точке as_of не зависят от событий после as_of
    (в т.ч. от боёв того же года, которые начнутся позже) и от состава сторон будущих лет."""
    ged, states, parties, neighbors, span = synth_war()
    as_of = pd.Period("2008-06", freq="M")
    cols = ["deaths_state", "deaths_party_nb", "state_war25", "onset_state_war25", "nonstate25", "war25", "state_war1000_12m"]

    def run(ged, parties):
        _, month = _war_panels(ged, states, parties, neighbors, span)
        m = month[(month.year == as_of.year) & (month.month == as_of.month)].sort_values("gw")
        w = kr_features.war_features(month, gw_to_iso3={g: f"S{g:02d}" for g in WAR_STATES})
        return m, w[w.month == as_of].sort_values("iso3")

    m0, w0 = run(ged, parties)
    after = (ged.year > as_of.year) | ((ged.year == as_of.year) & (ged.month > as_of.month))
    bad_ged = _corrupt(ged, after, ["best", "deaths_state"])
    bad_ged[["best", "deaths_state"]] = bad_ged[["best", "deaths_state"]].fillna(5000).round()
    bad_parties = parties.copy()
    fut = bad_parties.year > as_of.year
    bad_parties.loc[fut, "parties"] = bad_parties.loc[fut, "parties"].map(lambda _: WAR_STATES)
    m1, w1 = run(bad_ged, bad_parties)
    assert m0.state_war25.sum() > 0
    _same(m0, m1, cols)
    _same(w0, w1, ["war25_now", "war1000_now", "war_onset12"])


def test_war_party_not_open_before_first_event():
    """Сторона нового эпизода не «воюет» раньше месяца первого события конфликта;
    месячный онсет — только после 12 месяцев ниже порога."""
    states = pd.DataFrame({"gw": [1, 2, 3], "gw_name": ["A", "B", "C"], "start_year": 1989, "end_year": 1995})
    neighbors = {(a, b, y) for a, b in [(1, 2), (2, 1)] for y in range(1989, 1996)}
    ev = [{"gw": 1, "country": "A", "year": 1992, "month": m, "type_of_violence": 1, "best": b, "conflict_new_id": 7}
          for m, b in [(8, 400), (9, 30), (10, 10), (11, 40)]]
    ged = pd.DataFrame(ev)
    ged["deaths_state"] = ged.best
    parties = pd.DataFrame([{"conflict_id": 7, "year": 1992, "parties": [1, 2, 3]}])
    month = panel.build(ged, states, parties, neighbors, monthly=True, year_from=1989, year_to=1995)
    b = month[(month.gw == 2) & (month.year == 1992)].set_index("month")
    assert b.state_war25.loc[:7].sum() == 0 and b.state_war25.loc[8] == 1  # сосед-сторона: с августа
    assert b.onset_state_war25.sum() == 1                                    # ноябрь после октября — не новый онсет
    assert b.state_war25_rt.sum() == 0                                       # в реальном времени: списка сторон за 1992 ещё нет
    a_rt = month[(month.gw == 1) & (month.year == 1992)].set_index("month")
    assert a_rt.state_war25_rt.loc[8] == 1 and a_rt.onset_state_war25_rt.sum() == 1  # своя территория видна сразу
    c = month[(month.gw == 3) & (month.year == 1992)]
    assert c.state_war25.sum() == 0                                          # сторона без общей границы не «воюет»
    year = panel.build(ged, states, parties, neighbors, monthly=False, year_from=1989, year_to=1995)
    a = year[(year.gw == 1) & (year.year == 1992)].iloc[0]
    assert a.deaths_war == 480                                               # без двойного счёта «территория + сторона»


def test_neighbor_war_does_not_teleport():
    """Бои только в стране C; B — сосед C и сторона конфликта; A — мирный сосед B (не C).
    Война B на чужой территории не должна доходить до A через соседские признаки."""
    A, B, C = 1, 2, 3
    states = pd.DataFrame({"gw": [A, B, C], "gw_name": ["A", "B", "C"], "start_year": 1989, "end_year": 1995})
    neighbors = {(a, b, y) for a, b in [(A, B), (B, A), (B, C), (C, B)] for y in range(1989, 1996)}
    ged = pd.DataFrame([{"gw": C, "country": "C", "year": 1992, "month": 5, "type_of_violence": 1,
                         "best": 5000, "conflict_new_id": 7}])
    ged["deaths_state"] = ged.best
    parties = pd.DataFrame([{"conflict_id": 7, "year": 1992, "parties": [B, C]}])
    year = panel.build(ged, states, parties, neighbors, monthly=False, year_from=1989, year_to=1995)
    f = features.build_features(year, neighbors, target="state_war25").set_index(["gw", "year"])
    assert f.loc[(B, 1992), "state_war25"] == 1                    # B воюет (сторона у соседа)
    assert f.loc[(B, 1992), "nb_log_deaths_1"] > 8                 # и видит бои у соседа C
    a = f.loc[(A, 1992)]
    assert a.nb_log_deaths_1 == 0 and a.nb_war_any == 0 and a.nb_onset_any == 0 and a.nb_war_share == 0


# ---------------------------------------------------------------------------
# 4. Лаги публикации: признак не использует значение раньше его available_from
# ---------------------------------------------------------------------------


def _differs(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> bool:
    if len(a) != len(b):  # испорченные строки выпали целиком — тоже «изменилось»
        return True
    return any(not np.allclose(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), atol=ATOL, rtol=0, equal_nan=True)
               for c in cols)


def test_publication_lags():
    t = temporal
    assert t.available_from("gmd_annual", 2024) == pd.Timestamp("2025-04-01")
    assert t.available_from("imf_fx_december", 2024) == pd.Timestamp("2025-01-01")
    assert t.available_from("ucdp_acd", 2020) == pd.Timestamp("2021-01-01")
    # прогноз с признаков года T «на 1 января T+1»: курс за T есть, макро — только за T−1
    assert t.publication_lag("imf_fx_december", "jan1") == 0
    assert t.publication_lag("gmd_annual", "jan1") == 1 == t.publication_lag("wdi_annual", "jan1")
    assert t.publication_lag("gmd_annual", "apr1") == 0
    # внутри года Y известен список сторон войны только за Y−1
    assert t.last_available_year("ucdp_acd", pd.Timestamp("2020-09-30")) == 2019 and panel.ACD_LAG == 1
    months = pd.Series(pd.period_range("2010-01", "2010-06", freq="M"))
    assert t.last_available_year_in_month("wdi_annual", months).tolist() == [2008, 2008, 2008, 2009, 2009, 2009]


def test_available_from_fx_annual():
    """Прогноз «на 1 января T+1»: макроданные GMD за год T ещё не вышли и влиять не могут."""
    gmd, dec, lv = synth_annual_fx()
    T = 2000
    models = ["base_rate", "persist", "logit", "extree"]

    def run(gmd, lag):
        p = econ_long.build_panel(gmd, dec, lv, target="deval25", last_year=2015, macro_lag=lag)
        return econ_long.forecast_asof(p, T, n_estimators=40).sort_values("iso3")

    bad = _corrupt(gmd, gmd.year >= T, econ_long.MACRO_COLS)  # всё, что на 1 января T+1 не опубликовано
    lag = temporal.publication_lag("gmd_annual", "jan1")
    assert lag == econ_long.MACRO_LAG == 1
    clean = run(gmd, lag)
    assert clean[models].notna().all().all() and clean.extree.nunique() > 1
    _same(clean, run(bad, lag), models + long_features.FEATURES)
    assert _differs(run(gmd, 0), run(bad, 0), long_features.FEATURES), "контроль: без лага порча года T обязана быть видна"


def test_available_from_banking():
    """Банковская модель «на 1 января T+1»: входные ряды за год T (свои и JST) ещё не вышли."""
    jst, own = synth_jst(10, 3, "J"), synth_jst(12, 5, "E")
    T = 2000
    feats = econ_boe_em.NOMONEY
    inputs = [c for c in jst.columns if c not in econ_boe_em.NOT_LAGGED]
    models = ["base_rate", "jst_logreg", "jst_extree", "own_logreg", "own_extree", "pooled_logreg", "pooled_extree"]

    def run(jst, own, lag):
        data, jst_d = econ_boe_em.boe_dataset(own, feats, lag), econ_boe_em.boe_dataset(jst, feats, lag)
        return econ_boe_em.forecast_asof(data, jst_d, feats, T, n_estimators=40).sort_values("iso")

    bad_jst, bad_own = _corrupt(jst, jst.year >= T, inputs), _corrupt(own, own.year >= T, inputs, seed=7)
    clean = run(jst, own, econ_boe_em.MACRO_LAG)
    assert econ_boe_em.MACRO_LAG == 1 and clean[models].notna().all().all() and clean.jst_extree.nunique() > 1
    _same(clean, run(bad_jst, bad_own, econ_boe_em.MACRO_LAG), models + feats)
    assert _differs(run(jst, own, 0), run(jst, bad_own, 0), feats), "контроль: без лага порча года T обязана быть видна"


def test_available_from_cpi_fill():
    """Январь–март года t: годовой ИПЦ ВБ за t−1 ещё не вышел (выйдет 1 апреля) — заполнение
    месячного ИПЦ и прогноз не должны от него зависеть."""
    raw, annual, war = synth_monthly()
    as_of = pd.Period("2010-02", freq="M")

    def run(raw, annual):
        p = econ_kr.build_panel(raw, annual, war, target="deval15")
        return econ_kr.forecast_asof(p, as_of).sort_values("iso3")

    clean = run(raw, annual)
    bad = run(_corrupt(raw, raw.month > as_of, KR_COLS), _corrupt(annual, annual.year >= as_of.year - 1, ["value"]))
    assert clean.infl12.nunique() > 1
    _same(clean, bad, ["logit", "signals", "pressure", *kr_features.INDICATORS])


def test_available_from_war_monthly():
    """Месячные ПРИЗНАКИ войны (`*_rt`): список сторон ACD за текущий год ещё не вышел."""
    ged, states, parties, neighbors, span = synth_war()
    as_of = pd.Period("2008-06", freq="M")
    rt = ["deaths_party_nb_rt", "state_war25_rt", "onset_state_war25_rt", "state_war1000_12m_rt"]
    wcols = ["war25_now", "war1000_now", "war_onset12"]
    iso = {g: f"S{g:02d}" for g in WAR_STATES}

    def run(parties, realtime=True):
        _, month = _war_panels(ged, states, parties, neighbors, span)
        w = kr_features.war_features(month, gw_to_iso3=iso, realtime=realtime)
        return month[month.year == as_of.year].sort_values(["gw", "month"]), w[w.month.dt.year == as_of.year].sort_values(["iso3", "month"])

    bad_parties = parties.copy()
    now = bad_parties.year >= as_of.year  # список за текущий год и позже
    bad_parties.loc[now, "parties"] = bad_parties.loc[now, "parties"].map(lambda _: WAR_STATES)
    m0, w0 = run(parties)
    m1, w1 = run(bad_parties)
    assert m0.deaths_party_nb_rt.sum() > 0
    _same(m0, m1, rt)
    _same(w0, w1, wcols)
    assert _differs(run(parties, realtime=False)[1], run(bad_parties, realtime=False)[1], wcols), \
        "контроль: целевые (не `_rt`) от списка сторон текущего года зависят"


def test_available_from_war_annual():
    """Годовая модель войн: детская смертность и население ВБ за год T к 1 января T+1 не вышли."""
    ged, states, parties, neighbors, span = synth_war()
    year, _ = _war_panels(ged, states, parties, neighbors, span)
    f = features.build_features(year, neighbors, target="state_war25")
    rng = np.random.default_rng(6)
    grid = f[["gw", "year"]].copy()
    imr = grid.assign(infant_mortality=rng.uniform(5, 120, len(grid)))
    pop = grid.assign(population=rng.uniform(1e5, 1e8, len(grid)))
    pol = pd.DataFrame({"gw": pd.Series(dtype=int), "year": pd.Series(dtype=int), "regime": pd.Series(dtype=str),
                        "polity2": pd.Series(dtype=float), "regime_extended": pd.Series(dtype=float)})
    T = 2008
    cols = ["log_imr_rel", "imr_missing", "log_pop", "pop_missing"]

    def run(imr, pop, lag):
        out = features_pitf.add_pitf_features(f, pol, imr, pop, macro_lag=lag)
        return out[out.year == T].sort_values("gw")

    bad_imr, bad_pop = _corrupt(imr, imr.year >= T, ["infant_mortality"]), _corrupt(pop, pop.year >= T, ["population"], seed=5)
    clean = run(imr, pop, features_pitf.MACRO_LAG)
    assert features_pitf.MACRO_LAG == 1 and clean.log_imr_rel.nunique() > 1
    _same(clean, run(bad_imr, bad_pop, features_pitf.MACRO_LAG), cols)
    assert _differs(run(imr, pop, 0), run(bad_imr, bad_pop, 0), cols), "контроль: без лага порча года T обязана быть видна"


# ---------------------------------------------------------------------------
# 2. Целевая и evaluable согласованы
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["crisis", "deval15"])
def test_target_evaluable_sync_monthly(target):
    raw, annual, war = synth_monthly(seed=11)
    p = econ_kr.build_panel(raw, annual, war, target=target).reset_index(drop=True)
    spike = p[f"spike_{target}"].to_numpy().copy()
    ev = np.flatnonzero(p.evaluable.to_numpy())
    assert len(ev) > 500
    same_country = p.iso3.to_numpy()[ev] == p.iso3.to_numpy()[np.minimum(ev + 1, len(p) - 1)]
    rng = np.random.default_rng(0)
    for i in rng.choice(ev[same_country], 200, replace=False):
        s = spike.copy()
        s[i + 1] = True  # всплеск в первом же месяце окна
        onset, _ = kr_targets.onsets(pd.Series(s), p.iso3)
        assert onset.iloc[i + 1] == 1, "строка evaluable, но кризис в её окне заблокирован правилом тишины"


def test_target_evaluable_sync_annual():
    gmd, dec, lv = synth_annual_fx(seed=12)
    p = econ_long.build_panel(gmd, dec, lv, target="deval25", last_year=2015)
    ev = p[p.evaluable]
    assert len(ev) > 500
    prev_big = p.groupby("iso3").big.shift(1)
    assert (ev.big == 0).all() and (prev_big[ev.index].fillna(0) == 0).all(), \
        "строка evaluable, но девальвация в t+1 не может стать «началом» (уже идёт)"


# ---------------------------------------------------------------------------
# 3. Коды стран и валютные режимы
# ---------------------------------------------------------------------------


def test_entity_mapping():
    c = entities
    # постсоветские: полная и взаимно однозначная разметка
    assert len(c.POST_SOVIET) == 15 == len(c.POST_SOVIET_GW) == len(c.POST_SOVIET_ISO3)
    for s in c.POST_SOVIET:
        assert c.GW_TO_ISO3[s["gw"]] == s["iso3"] and c.ISO3_TO_GW[s["iso3"]] == s["gw"]
    # GW ↔ ISO3: взаимно однозначно, кроме Сербии/Югославии (340 и 345 → SRB)
    iso = [v for v in c.GW_TO_ISO3.values() if v]
    dup = {v for v in iso if iso.count(v) > 1}
    assert dup == {"SRB"} and c.ISO3_TO_GW["SRB"] == 340
    assert all(c.GW_TO_ISO3[gw] == iso3 for iso3, gw in c.ISO3_TO_GW.items())
    # COW и Polity → GW: всё приводится к кодам из списка GW
    assert c.to_gw(255, "cow") == 260 and c.to_gw(679, "cow") == 678 and c.to_gw(2, "cow") == 2
    assert set(c.COW_TO_GW.items()) <= set(c.POLITY_TO_GW.items())
    for code in (364, 342, 348, 525, 529, 818):
        assert c.to_gw(code, "polity") in c.GW_TO_ISO3
    # валютные режимы: строго по году перехода
    sc = c.shared_currency
    assert sc("EST", 2010) is None and sc("EST", 2011) == "EUR"
    assert sc("LVA", 2013) is None and sc("LVA", 2014) == "EUR"
    assert sc("LTU", 2014) is None and sc("LTU", 2015) == "EUR"
    assert sc("HRV", 2022) is None and sc("HRV", 2023) == "EUR"
    assert sc("ITA", 1998) is None and sc("ITA", 1999) == "EUR" and sc("FIN", 1992) is None
    assert sc("ECU", 1999) is None and sc("ECU", 2000) == "USD" and sc("SLV", 2001) == "USD"
    assert sc("ZWE", 2015) == "USD" and sc("ZWE", 2020) is None
    assert sc("SEN", 1994) == "XOF" and sc("CMR", 2020) == "XAF" and sc("GNB", 1996) is None
    for iso3 in ("RUS", "UKR", "KAZ", "USA", "GBR", "CHE", "DNK"):
        assert sc(iso3, 2025) is None
    t = c.shared_currency_table(1990, 2025)
    assert not t.duplicated(["iso3", "year"]).any()
    assert set(t[t.year == 2025].fx_regime) >= {"EUR", "USD", "XOF", "XAF"}
    # FIPS для GDELT: без повторов
    assert len(set(c.ISO3_TO_FIPS.values())) == len(c.ISO3_TO_FIPS)
