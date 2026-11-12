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
3. `test_entity_mapping` — коды стран и валютные союзы в `countries.py`.

Глобальные константы модулей тесты не трогают: всё передаётся аргументами.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seldon import countries, econ_boe, econ_boe_em, econ_kr, econ_long, features, model_hist, panel

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
    _same(clean, bad, models + list(econ_kr.INDICATORS))


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
    _same(clean, bad, models + econ_long.FEATURES)


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
        w = econ_kr.war_features(month, gw_to_iso3={g: f"S{g:02d}" for g in WAR_STATES})
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
    c = month[(month.gw == 3) & (month.year == 1992)]
    assert c.state_war25.sum() == 0                                          # сторона без общей границы не «воюет»
    year = panel.build(ged, states, parties, neighbors, monthly=False, year_from=1989, year_to=1995)
    a = year[(year.gw == 1) & (year.year == 1992)].iloc[0]
    assert a.deaths_war == 480                                               # без двойного счёта «территория + сторона»


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
        onset, _ = econ_kr.onsets(pd.Series(s), p.iso3)
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
    c = countries
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
