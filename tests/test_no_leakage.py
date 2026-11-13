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

5. Структурный валютный блок (чекпоинт 3.3): те же две проверки (слепота к будущему и лаги
   публикации) для таблицы признаков `struct_features` и обоих контуров `econ_struct`;
   конец хроники IRR; привязка к евро; вариант весов условий торговли.

6. Валютное заражение (чекпоинт 4.1): торговые веса не смотрят в текущий и будущие годы,
   курсы партнёров — за месяц m, премия параллельного курса соблюдает лаг публикации;
   прогноз `econ_contagion` на горизонтах 3, 6 и 12 месяцев слеп к будущему.

Глобальные константы модулей тесты не трогают: всё передаётся аргументами.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seldon.core import entities, temporal
from seldon.domains.banking import econ_boe, econ_boe_em
from seldon.domains.conflict import (dyads, features, features_epr, features_pitf, features_vdem, live_pairs, model_dyad, model_dyad_live,
                                     model_hist, model_month, month_features, panel)
from seldon.sources import epr as epr_source
from seldon.sources import vdem as vdem_source
from seldon.domains.currency import (break_features, contagion_features, econ_contagion, econ_kr, econ_long, econ_struct, kr_features, kr_targets,
                                     long_features, struct_features)
from seldon.domains.currency.contagion_features import ContagionLags
from seldon.domains.currency.struct_features import StructLags
from seldon.sources import imf as imf_source
from seldon.sources import irr as irr_source

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
    # политологические базы (чекпоинт 3.1): к 1 января T+1 год T не опубликован ни в V-Dem, ни в EPR
    assert t.available_from("vdem_annual", 2024) == pd.Timestamp("2025-04-01")
    assert t.publication_lag("vdem_annual", "jan1") == 1 == features_vdem.VDEM_LAG and t.publication_lag("vdem_annual", "apr1") == 0
    assert t.publication_lag("epr_annual", "jan1") == 1 == features_epr.EPR_LAG and t.publication_lag("epr_annual", "apr1") == 1
    # диадный контур (чекпоинт 3.2): сессия ГА ООН года T к 1 января T+1 не оценена; хроника
    # MID выходит с опозданием в годы
    assert t.available_from("unga_ideal", 2024) == pd.Timestamp("2025-07-01")
    assert t.publication_lag("unga_ideal", "jan1") == 1 == dyads.UNGA_LAG and t.publication_lag("unga_ideal", "apr1") == 1
    assert t.publication_lag("cow_mid", "jan1") == 3 == dyads.MID_LAG and dyads.CAP_LAG == 1
    months = pd.Series(pd.period_range("2010-01", "2010-06", freq="M"))
    assert t.last_available_year_in_month("wdi_annual", months).tolist() == [2008, 2008, 2008, 2009, 2009, 2009]
    # события UCDP (чекпоинт 4.2): файл Candidate за месяц m выходит 20-го числа m+1
    assert t.monthly_lag("ucdp_events") == 1


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


def synth_struct(grid: pd.DataFrame, seed: int = 7):
    """Синтетические V-Dem (страна × год) и EPR (группа × год, родня — у соседа по номеру)."""
    rng = np.random.default_rng(seed)
    v = grid[["gw", "year"]].copy()
    for c in ("v2x_polyarchy", "v2x_libdem", "v2x_clphy"):
        v[c] = rng.uniform(0.05, 0.95, len(v))
    v["v2cacamps"] = rng.normal(0, 1.5, len(v))
    statuses = list(epr_source.POWER_RANK)
    rows = []
    for gw in sorted(grid.gw.unique()):
        for k in range(3):
            status = statuses[int(rng.integers(len(statuses)))]
            for year in sorted(grid.year.unique()):
                if rng.random() < 0.12:
                    status = statuses[int(rng.integers(len(statuses)))]
                rows.append({"gw": gw, "year": year, "gwgroupid": gw * 100 + k, "size": [0.6, 0.25, 0.1][k], "status": status})
    g = pd.DataFrame(rows)
    g["rank"] = g.status.map(epr_source.POWER_RANK)
    tek = pd.DataFrame([{"tekid": gw * 10 + k, "gw": c, "gwgroupid": c * 100 + k}
                        for gw in sorted(grid.gw.unique())[:-1] for k in (1, 2) for c in (gw, gw + 1) if (gw + k) % 2])
    return v, g, tek


def test_available_from_war_struct():
    """V-Dem и EPR за год T к 1 января T+1 не опубликованы: ни признаки, ни прогноз набора
    `vdem_epr` в точке T не зависят от их значений за T и позже."""
    ged, states, parties, neighbors, span = synth_war()
    year, _ = _war_panels(ged, states, parties, neighbors, span)
    f = features.build_features(year, neighbors, target="state_war25")
    for c in model_hist.PITF_MACRO:
        f[c] = 0.0
    v, g, tek = synth_struct(f)
    war_terr = year[["gw", "year", "deaths_state"]]
    T = 2008
    feats = model_hist.FEATURE_SETS["vdem_epr"][0]
    cols = [*features_vdem.VDEM_FEATURES, *features_epr.EPR_FEATURES]

    def run(v, g, lag):
        out = features_vdem.add_vdem_features(f, v, lag=lag)
        out = features_epr.add_epr_features(out, g, tek, neighbors, war_terr, lag=lag, last_year=span[1])
        return model_hist.forecast_asof(out, T, feats, target="state_war25").sort_values("gw")

    bad_v = _corrupt(v, v.year >= T, vdem_source.VDEM_COLS)
    bad_v = bad_v.fillna(0.5)
    bad_g = g.copy()
    late = bad_g.year >= T
    rng = np.random.default_rng(8)
    bad_g.loc[late, "status"] = rng.choice(list(epr_source.POWER_RANK), int(late.sum()))
    bad_g.loc[late, "size"] = rng.uniform(0, 1, int(late.sum()))
    bad_g["rank"] = bad_g.status.map(epr_source.POWER_RANK)
    clean = run(v, g, 1)
    assert clean.p.nunique() > 1 and clean.epr_downgraded_2y.sum() > 0 and clean.epr_tek_kin_war_nb.sum() > 0
    _same(clean, run(bad_v, bad_g, 1), ["p", "p_onset", *cols])
    assert _differs(run(v, g, 0), run(bad_v, g, 0), features_vdem.VDEM_FEATURES), "контроль: без лага порча V-Dem года T видна"
    assert _differs(run(v, g, 0), run(v, bad_g, 0), features_epr.EPR_FEATURES), "контроль: без лага порча EPR года T видна"


def test_epr_chronicle_end_is_flagged():
    """Хроника EPR кончается в своём последнем году: дальше статусы продлены с флагом
    `epr_extended`, а не выдаются за свежие данные (инвариант 6)."""
    ged, states, parties, neighbors, span = synth_war()
    year, _ = _war_panels(ged, states, parties, neighbors, span)
    f = features.build_features(year, neighbors, target="state_war25")
    _, g, tek = synth_struct(f)
    last = span[1] - 4
    out = features_epr.add_epr_features(f, g, tek, neighbors, year[["gw", "year", "deaths_state"]], lag=1, last_year=last)
    assert (out[out.year <= last + 1].epr_extended == 0).all() and (out[out.year > last + 1].epr_extended == 1).all()
    assert (out[out.year > last + 1].epr_missing == 0).all()
    # всё, что в хронике записано после её последнего года, игнорируется
    g2 = g.copy()
    g2.loc[g2.year > last, "status"] = "DISCRIMINATED"
    g2["rank"] = g2.status.map(epr_source.POWER_RANK)
    out2 = features_epr.add_epr_features(f, g2, tek, neighbors, year[["gw", "year", "deaths_state"]], lag=1, last_year=last)
    _same(out, out2, features_epr.EPR_FEATURES)


# ---------------------------------------------------------------------------
# 4б. Диадный контур (чекпоинт 3.2): слепота к будущему, лаги публикации, конец хроники MID
# ---------------------------------------------------------------------------

DYAD_POWERS = (1, 2)


def synth_dyad(seed: int = 11):
    """Синтетика для пар: ACD со сторонами (страна места — сторона A, помощник — на стороне
    противника), идеальные точки ООН, потенциалы, споры MID."""
    ged, states, parties, neighbors, span = synth_war()
    y0, y1 = span
    rng = np.random.default_rng(seed)
    contig = pd.DataFrame(list(neighbors), columns=["a", "b", "year"]).assign(conttype=1)
    acd = pd.DataFrame({
        "conflict_id": parties.conflict_id, "year": parties.year,
        "gwno_a": parties.parties.map(lambda p: str(p[0])), "gwno_a_2nd": "", "gwno_b": "",
        "gwno_b_2nd": parties.parties.map(lambda p: str(p[1]) if len(p) > 1 else ""),
        "gwno_loc": parties.parties.map(lambda p: str(p[0])), "incompatibility": rng.integers(1, 3, len(parties))})
    cy = pd.DataFrame([(g, y) for g in WAR_STATES for y in range(y0 - 6, y1 + 1)], columns=["gw", "year"])
    unga = cy.assign(ideal=rng.normal(0, 1, len(cy)))
    cap = cy.copy()
    for c in dyads.CAP_COLS:
        cap[c] = np.exp(rng.normal(3, 1, len(cap)))
    pairs = [(a, b) for a in WAR_STATES for b in WAR_STATES if a != b and (abs(a - b) == 1 or b in DYAD_POWERS)]
    mid = pd.DataFrame([(a, b, y, int(rng.random() < 0.6), int(rng.random() < 0.3))
                        for a, b in pairs for y in range(1975, y1 + 1) if rng.random() < 0.15],
                       columns=["a", "b", "year", "force", "terr"])
    return {"ged": ged, "states": states, "parties": parties, "neighbors": neighbors, "span": span,
            "contig": contig, "acd": acd, "unga": unga, "cap": cap, "mid": mid}


def _dyad_features(s: dict, lags=(1, 3, 1), mid_last_year: int | None = None, **over) -> pd.DataFrame:
    """Панель пар из синтетики; `over` подменяет источники, `lags` = (ООН, MID, потенциалы)."""
    s = {**s, **over}
    y0, y1 = s["span"]
    grid = dyads.build_grid(s["states"], s["contig"], y0, y1, powers=DYAD_POWERS)
    opp = dyads.opponents(s["acd"])
    dw = dyads.dyad_wars(s["ged"], opp, s["neighbors"])
    return dyads.build_features(grid, opp, dw, s["unga"], s["mid"], s["cap"], unga_lag=lags[0], mid_lag=lags[1],
                                cap_lag=lags[2], mid_last_year=y1 - 2 if mid_last_year is None else mid_last_year)


def test_future_blindness_dyad():
    """Диадный прогноз в точке T не зависит ни от боёв и сторон войн после T, ни от того, что
    к 1 января T+1 не опубликовано: сессия ООН и потенциалы за T, споры MID позже T−3."""
    s = synth_dyad()
    T = 2008
    cols = ["p_dyad", "p_base", *dyads.FEATURES]

    def run(lags=(1, 3, 1), **over):
        return model_dyad.forecast_asof(_dyad_features(s, lags, **over), T).sort_values(["a", "b"])

    ged, acd = s["ged"], s["acd"]
    bad_ged = _corrupt(ged, ged.year > T, ["best", "deaths_state"])
    bad_ged[["best", "deaths_state"]] = bad_ged[["best", "deaths_state"]].fillna(5000).round()
    bad_acd = acd.copy()
    bad_acd.loc[bad_acd.year > T, "gwno_b_2nd"] = ", ".join(map(str, WAR_STATES))
    bad_acd.loc[bad_acd.year > T, "incompatibility"] = 1
    bad_unga = _corrupt(s["unga"], s["unga"].year >= T, ["ideal"])
    bad_cap = _corrupt(s["cap"], s["cap"].year >= T, dyads.CAP_COLS)
    bad_mid = s["mid"].copy()
    late = bad_mid.year > T - dyads.MID_LAG
    bad_mid.loc[late, ["force", "terr"]] = 1 - bad_mid.loc[late, ["force", "terr"]]

    clean = run()
    assert clean.p_dyad.nunique() > 1 and clean.pair_recency.max() > 0 and clean.mid_force.max() > 0 and len(clean) > 50
    _same(clean, run(ged=bad_ged, acd=bad_acd, unga=bad_unga, cap=bad_cap, mid=bad_mid), cols)
    # контроль: без лага порча тех же лет обязана быть видна
    assert _differs(run((0, 3, 1)), run((0, 3, 1), unga=bad_unga), ["unga_dist", "unga_shift"]), "ООН: без лага порча года T видна"
    assert _differs(run((1, 3, 0)), run((1, 3, 0), cap=bad_cap), ["log_power_ratio", "b_milex_surge"]), "потенциалы: без лага порча года T видна"
    assert _differs(run((1, 0, 1)), run((1, 0, 1), mid=bad_mid), ["mid_force", "terr_dispute"]), "MID: без лага порча лет после T−3 видна"
    # контроль самой проверки: порча ПРОШЛОГО обязана менять прогноз
    past_unga = _corrupt(s["unga"], s["unga"].year < T, ["ideal"]).fillna(3.0)
    assert _differs(clean, run(unga=past_unga), ["p_dyad"])


def test_mid_chronicle_end_is_flagged():
    """Хроника MID кончается в своём последнем году: дальше значения заморожены с флагом
    `mid_extended`, записанное после конца хроники игнорируется (инвариант 6)."""
    s = synth_dyad()
    last = s["span"][1] - 8
    d = _dyad_features(s, mid_last_year=last)
    assert (d[d.year <= last + dyads.MID_LAG].mid_extended == 0).all() and (d[d.year > last + dyads.MID_LAG].mid_extended == 1).all()
    frozen = d[d.year >= last + dyads.MID_LAG].groupby(["a", "b"]).mid_force.nunique()
    assert (frozen == 1).all() and d.mid_force.max() > 0
    bad_mid = s["mid"].copy()
    bad_mid.loc[bad_mid.year > last, ["force", "terr"]] = 1
    _same(d, _dyad_features(s, mid_last_year=last, mid=bad_mid), ["mid_force", "mid_terr", "terr_dispute"])


def test_dyad_war_is_part_of_state_war():
    """Диадная целевая — часть страновой: `dyad_war(A ← B) = 1` ⇒ `state_war25(A) = 1`. Далёкая
    экспедиция — война для страны, где идут бои, но не для далёкой державы."""
    s = synth_dyad()
    year, _ = _war_panels(s["ged"], s["states"], s["parties"], s["neighbors"], s["span"])
    d = _dyad_features(s)
    w = d[d.dyad_war == 1].merge(year[["gw", "year", "state_war25"]], left_on=["a", "year"], right_on=["gw", "year"], how="left")
    assert len(w) > 20 and (w.state_war25 == 1).all()
    # страна 10 воюет на своей территории против далёкой державы 1 и соседа 11
    ged = pd.DataFrame({"gw": [10], "year": [2000], "type_of_violence": [1], "best": [300], "conflict_new_id": [77]})
    acd = pd.DataFrame({"conflict_id": [77], "year": [2000], "gwno_a": ["10"], "gwno_a_2nd": [""], "gwno_b": ["1"],
                        "gwno_b_2nd": ["11"], "incompatibility": [2]})
    dw = dyads.dyad_wars(ged, dyads.opponents(acd), s["neighbors"])
    assert set(zip(dw.a, dw.b)) == {(10, 1), (10, 11), (11, 10)}


# ---------------------------------------------------------------------------
# 5. Структурный валютный блок (чекпоинт 3.3)
# ---------------------------------------------------------------------------

STRUCT_NO_LAG = StructLags(reer_months=0, ctot_months=0, wdi_source=None, irr_source=None)
STRUCT_IRR_LAST = 2014  # в синтетике хроника IRR «кончается» здесь, если не сказано иное


def synth_fx_struct(countries: list[str], months: pd.PeriodIndex, seed: int = 11, fx: pd.DataFrame | None = None) -> dict:
    """Входы `struct_features.build_struct`: курс, евро-якорь, REER/NEER, условия торговли,
    режим IRR и резервы ВБ — случайные ряды."""
    rng = np.random.default_rng(seed)
    n = len(months)
    years = range(int(months.year.min()) - 1, int(months.year.max()) + 1)
    walk = lambda scale, drift=0.0: np.exp(np.cumsum(rng.normal(drift, scale, n)))  # noqa: E731
    if fx is None:
        fx = pd.concat([pd.DataFrame({"iso3": c, "month": months, "fx": 10 * walk(0.02, 0.002)}) for c in countries], ignore_index=True)
    reer = pd.concat([pd.DataFrame({"iso3": c, "month": months, "reer120": 100 * walk(0.02), "reer51": 100 * walk(0.02),
                                    "neer120": 100 * walk(0.02), "neer51": 100 * walk(0.02)}) for c in countries], ignore_index=True)
    ctot = pd.concat([pd.DataFrame({"iso3": c, "month": months, "ctot": 100 * walk(0.01)}) for c in countries], ignore_index=True)
    return {
        "fx": fx[["iso3", "month", "fx"]], "eur": pd.DataFrame({"month": months, "eur_usd": walk(0.02)}), "reer": reer, "ctot": ctot,
        "irr": pd.DataFrame([(c, y, int(rng.integers(1, 15))) for c in countries for y in years], columns=["iso3", "year", "irr_fine"]),
        "buffers": pd.DataFrame([(c, y, float(rng.uniform(5, 300)), float(rng.uniform(0.5, 12))) for c in countries for y in years],
                                columns=["iso3", "year", "std_res", "res_months"]),
    }


def _struct(s: dict, lags: StructLags = StructLags(), irr_last_year: int = STRUCT_IRR_LAST) -> pd.DataFrame:
    return struct_features.build_struct(s["fx"], s["eur"], s["reer"], s["ctot"], s["irr"], s["buffers"], lags, irr_last_year)


def _corrupt_struct(s: dict, fx_after=None, eur_after=None, reer_after=None, neer_after=None, ctot_after=None,
                    wdi_from=None, irr_from=None) -> dict:
    """Испортить входы: месячные — строго после указанного месяца, годовые — начиная с года."""
    out = dict(s)
    if fx_after is not None:
        out["fx"] = _corrupt(s["fx"], s["fx"].month > fx_after, ["fx"], seed=21)
    if eur_after is not None:
        out["eur"] = _corrupt(s["eur"], s["eur"].month > eur_after, ["eur_usd"], seed=22)
    reer = s["reer"]
    if reer_after is not None:
        reer = _corrupt(reer, reer.month > reer_after, ["reer120", "reer51"], seed=23)
    if neer_after is not None:
        reer = _corrupt(reer, reer.month > neer_after, ["neer120", "neer51"], seed=24)
    out["reer"] = reer
    if ctot_after is not None:
        out["ctot"] = _corrupt(s["ctot"], s["ctot"].month > ctot_after, ["ctot"], seed=25)
    if wdi_from is not None:
        out["buffers"] = _corrupt(s["buffers"], s["buffers"].year >= wdi_from, ["std_res", "res_months"], seed=26)
    if irr_from is not None:
        out["irr"] = _corrupt(s["irr"], s["irr"].year >= irr_from, ["irr_fine"], seed=27)
    return out


def _future_struct(s: dict, as_of: pd.Period) -> dict:
    """Всё, чего в месяце `as_of` ещё не случилось."""
    return _corrupt_struct(s, fx_after=as_of, eur_after=as_of, reer_after=as_of, neer_after=as_of, ctot_after=as_of,
                           wdi_from=as_of.year, irr_from=as_of.year)


def _unpublished_struct(s: dict, as_of: pd.Period) -> dict:
    """Всё, что к концу месяца `as_of` случилось, но ещё НЕ ОПУБЛИКОВАНО (плюс будущее)."""
    month = pd.Series([as_of])
    return _corrupt_struct(
        s, fx_after=as_of, eur_after=as_of, neer_after=as_of,
        reer_after=as_of - temporal.monthly_lag("bruegel_reer"), ctot_after=as_of - temporal.monthly_lag("imf_ctot"),
        wdi_from=int(temporal.last_available_year_in_month("wdi_annual", month).iloc[0]) + 1,
        irr_from=int(temporal.last_available_year_in_month("irr_regime", month).iloc[0]) + 1)


def test_future_blindness_fx_struct_features():
    months = pd.period_range("1993-01", "2014-12", freq="M")
    s = synth_fx_struct(KR_COUNTRIES, months)
    as_of = pd.Period("2010-06", freq="M")
    cols = struct_features.COLUMNS + ["irr_extended"]
    clean, bad = _struct(s), _struct(_future_struct(s, as_of))
    past = lambda d: d[d.month <= as_of].sort_values(["iso3", "month"])  # noqa: E731
    assert all(clean[clean.month == as_of][c].nunique() > 1 for c in struct_features.CONT)
    _same(past(clean), past(bad), cols)
    assert _differs(clean[clean.month > as_of], bad[bad.month > as_of], cols), "контроль: порча будущего обязана быть видна в будущем"


def test_available_from_fx_struct_features():
    """Строка месяца m не использует REER позже m−2, условия торговли позже m−6, годовые
    данные ВБ и режим IRR — раньше их `available_from`. Контроль: без лага порча видна."""
    t = temporal
    assert t.monthly_lag("bruegel_reer") == 2 and t.monthly_lag("imf_ctot") == 6 and t.monthly_lag("bruegel_neer") == 0
    assert StructLags() == StructLags(reer_months=2, ctot_months=6, wdi_source="wdi_annual", irr_source="irr_regime")
    assert t.available_from("irr_regime", 2019) == pd.Timestamp("2021-12-01")  # стабильный релиз IRR — ноябрь 2021
    assert t.publication_lag("irr_regime", "jan1") == 2 == t.publication_lag("irr_regime", "apr1")
    nov_dec = pd.Series(pd.period_range("2010-11", "2010-12", freq="M"))
    assert t.last_available_year_in_month("irr_regime", nov_dec).tolist() == [2007, 2008]
    months = pd.period_range("1993-01", "2014-12", freq="M")
    s = synth_fx_struct(KR_COUNTRIES, months)
    cols = struct_features.COLUMNS
    for as_of in (pd.Period("2010-02", freq="M"), pd.Period("2010-12", freq="M")):  # декабрь — срез годового контура
        row = lambda d: d[d.month == as_of].sort_values("iso3")  # noqa: E731
        bad = _unpublished_struct(s, as_of)
        _same(row(_struct(s)), row(_struct(bad)), cols)
        for lags, feats in ((StructLags(reer_months=0), ["reer_gap", "reer_peg"]), (StructLags(ctot_months=0), ["ctot_d12"]),
                            (StructLags(wdi_source=None), ["gg_log", "res_months_log"]), (StructLags(irr_source=None), ["irr_peg"])):
            assert _differs(row(_struct(s, lags)), row(_struct(bad, lags)), feats), f"контроль: без лага порча обязана быть видна ({feats})"
    # годовой срез: декабрьская строка года T = то, что известно «на 1 января T+1»
    annual = struct_features.annual_view(_struct(s))
    assert annual.year.max() == 2014 and not annual.duplicated(["iso3", "year"]).any()


def test_irr_chronicle_end_is_flagged():
    """После последнего размеченного года IRR режим не «плавающий», а замороженный с флагом
    (инвариант 6); данные «после конца хроники» на признак не влияют."""
    assert irr_source.IRR_LAST_YEAR == 2019 and irr_source.PEG_MAX_CODE == 8
    months = pd.period_range("1993-01", "2014-12", freq="M")
    s = synth_fx_struct(KR_COUNTRIES, months)
    last = 2005
    clean = _struct(s, irr_last_year=last)
    known = temporal.last_available_year_in_month("irr_regime", clean.month)
    assert (clean.irr_extended == (known > last).astype(int)).all() and clean.irr_extended.sum() > 0
    frozen = clean[clean.irr_extended == 1].merge(s["irr"][s["irr"].year == last], on="iso3")
    assert (frozen.irr_peg == (frozen.irr_fine <= irr_source.PEG_MAX_CODE).astype(float)).all()
    bad = _struct(_corrupt_struct(s, irr_from=last + 1), irr_last_year=last)
    _same(clean, bad, ["irr_peg", "irr_missing", "irr_extended"])


def test_peg_sees_euro_anchor():
    """Валюта, привязанная к евро, качается к доллару вместе с евро — живой индикатор обязан
    видеть привязку (иначе она выглядит как плавание)."""
    months = pd.period_range("1999-01", "2014-12", freq="M")
    s = synth_fx_struct(["PEG", "FLT", "USD"], months, seed=5)
    rng = np.random.default_rng(6)
    eur = s["eur"].eur_usd.to_numpy()
    fx = pd.concat([
        pd.DataFrame({"iso3": "PEG", "month": months, "fx": 1.95583 * eur}),                                    # привязка к евро
        pd.DataFrame({"iso3": "FLT", "month": months, "fx": np.exp(np.cumsum(rng.normal(0, 0.04, len(months))))}),  # плавание
        pd.DataFrame({"iso3": "USD", "month": months, "fx": 3.75}),                                             # привязка к доллару
    ], ignore_index=True)
    out = _struct({**s, "fx": fx})
    last = out[out.month == months[-1]].set_index("iso3").peg_share12
    assert last["PEG"] == 1.0 and last["USD"] == 1.0 and last["FLT"] < 0.5
    usd_only = (np.abs(np.diff(np.log(1.95583 * eur))) <= struct_features.PEG_STEP)[-12:].mean()
    assert usd_only < 0.8, "контроль: к доллару привязанная к евро валюта привязкой не выглядит"


def test_ctot_uses_rolling_weights():
    """Сырьевые условия торговли — только со скользящими весами ПРОШЛЫХ лет: у фиксированных
    весов структура экспорта будущих лет попадает в прошлое."""
    from seldon.sources.download import SOURCES

    assert imf_source.CTOT_WEIGHTS == "R_RW_IX"
    flow, filters = SOURCES["imf"]["flows"]["ctot"]
    assert flow == "CTOT" and filters["WGT_TYPE"] == imf_source.CTOT_WEIGHTS and filters["INDICATOR"] == imf_source.CTOT_INDICATOR


def test_future_blindness_fx_struct_monthly():
    raw, annual, war = synth_monthly()
    months = pd.period_range("1993-01", "2014-12", freq="M")
    s = synth_fx_struct(KR_COUNTRIES, months, fx=raw)
    as_of = pd.Period("2010-06", freq="M")

    def run(raw, annual, war, s, name):
        p = econ_struct.build_kr_panel(_struct(s), raw, annual, war)
        return econ_struct.forecast_asof_kr(p, as_of, name).sort_values("iso3")

    bad_raw = _corrupt(raw, raw.month > as_of, KR_COLS)
    bad = (bad_raw, _corrupt(annual, annual.year >= as_of.year, ["value"]),
           _corrupt(war, war.month > as_of, ["war25_now", "war1000_now", "war_onset12"]),
           _future_struct({**s, "fx": bad_raw[["iso3", "month", "fx"]]}, as_of))
    for name in ("struct", "reer"):
        clean = run(raw, annual, war, s, name)
        assert clean.logit.notna().all() and clean.logit.nunique() > 1
        _same(clean, run(*bad, name), ["logit", "C", *struct_features.COLUMNS])
    # эталон `ref` — та же логрегрессия, что в боевом контуре
    ref = run(raw, annual, war, s, "ref")
    battle = econ_kr.forecast_asof(econ_kr.build_panel(raw, annual, war, target="deval15"), as_of).sort_values("iso3")
    _same(ref, battle, ["logit"])
    assert _differs(ref, run(raw, annual, war, s, "struct"), ["logit"]), "контроль: набор обязан менять прогноз"


def _annual_struct_inputs():
    gmd, dec, lv = synth_annual_fx()
    months = pd.period_range("1950-01", "2015-12", freq="M")
    return gmd, dec, lv, synth_fx_struct(sorted(dec.iso3.unique()), months, seed=12)


def test_future_blindness_fx_struct_annual():
    gmd, dec, lv, s = _annual_struct_inputs()
    T = 2000
    models = ["logit", "extree"]

    def run(gmd, dec, s, name="struct"):
        p = econ_struct.build_long_panel(_struct(s), gmd, dec, lv)
        return econ_struct.forecast_asof_long(p, T, name, n_estimators=40).sort_values("iso3")

    num = [c for c in gmd.columns if c not in ("ISO3", "countryname", "year")]
    bad_gmd, bad_dec = _corrupt(gmd, gmd.year > T, num), _corrupt(dec, dec.year > T, ["fx_dec"])
    clean = run(gmd, dec, s)
    assert clean[models].notna().all().all() and clean.extree.nunique() > 1 and clean.C.notna().all()
    _same(clean, run(bad_gmd, bad_dec, _future_struct(s, pd.Period(f"{T}-12", freq="M"))), models + ["C", *struct_features.COLUMNS])
    # эталон `ref` — боевая модель как есть
    ref = run(gmd, dec, s, "ref")
    battle = econ_long.forecast_asof(econ_long.build_panel(gmd, dec, lv, target="deval25", last_year=2025), T, n_estimators=40)
    _same(ref, battle.sort_values("iso3"), ["logit", "extree"])
    assert _differs(ref, clean, ["logit"]), "контроль: набор обязан менять прогноз"


def test_available_from_fx_struct_annual():
    """Годовой прогноз «на 1 января T+1»: REER позже октября T, условия торговли позже июня T,
    данные ВБ за T и режим IRR за T−1 ещё не вышли и влиять не могут."""
    gmd, dec, lv, s = _annual_struct_inputs()
    T = 2000

    def run(s, lags=StructLags()):
        p = econ_struct.build_long_panel(_struct(s, lags), gmd, dec, lv)
        return econ_struct.forecast_asof_long(p, T, "struct", n_estimators=40).sort_values("iso3")

    bad = _unpublished_struct(s, pd.Period(f"{T}-12", freq="M"))
    cols = ["logit", "extree", *struct_features.COLUMNS]
    _same(run(s), run(bad), cols)
    assert _differs(run(s, STRUCT_NO_LAG), run(bad, STRUCT_NO_LAG), struct_features.CONT), "контроль: без лага порча обязана быть видна"


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


def test_shared_currency_entry_year_boundary():
    """Стык года вступления в валютный союз (чекпоинт 3.1): Эстония переходит на евро в 2011.
    Строка 2010 года смотрит в окно 2011…2012, где своей валюты уже нет: она не может быть
    ни оцениваемой, ни получить индивидуальный прогноз. Строка 2009 (окно начинается в 2010,
    ещё крона) прогноз получает."""
    gmd, dec, lv = synth_annual_fx(seed=13)
    gmd, dec = gmd.replace({"ISO3": {"C05": "EST"}}), dec.replace({"iso3": {"C05": "EST"}})
    p = econ_long.build_panel(gmd, dec, lv, target="deval25", last_year=2015)
    est = p[p.iso3 == "EST"].set_index("year")
    assert est.own_currency[2010] and not est.own_currency[2011]
    assert est.usable[2009], "за два года до перехода валюта своя и окно начинается со своей валюты"
    assert not est.usable[2010] and not est.evaluable[2010], \
        "строка года t получила валютный прогноз, хотя с t+1 у страны общая валюта"
    assert "EST" in set(econ_long.forecast_asof(p, 2009, n_estimators=40).iso3)
    assert "EST" not in set(econ_long.forecast_asof(p, 2010, n_estimators=40).iso3)
    # вариант «все валюты» (без исключения союзов) правило не трогает
    p_all = econ_long.build_panel(gmd, dec, lv, target="deval25", last_year=2015, drop_shared=False)
    assert p_all[(p_all.iso3 == "EST") & (p_all.year == 2010)].usable.all()


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


# ---------------------------------------------------------------------------
# Чекпоинт 4.1: валютное заражение по торговым связям и параллельный курс
# ---------------------------------------------------------------------------


def synth_contagion(countries: list[str], months: pd.PeriodIndex, seed: int = 31) -> dict:
    """Торговля пар по годам, премия параллельного курса и ВВП на душу — случайные."""
    rng = np.random.default_rng(seed)
    years = range(int(months.year.min()) - 4, int(months.year.max()) + 1)
    # лог-нормальные объёмы: у стран появляются «ключевые» партнёры с долей ≥ 15 %
    trade = pd.DataFrame([(a, b, y, float(rng.lognormal(0, 2)), float(rng.lognormal(0, 2)))
                          for a in countries for b in countries if a != b for y in years],
                         columns=["iso3", "partner", "year", "exports", "imports"])
    parallel = pd.DataFrame([(c, m, float(rng.uniform(0, 1))) for c in countries[1:7] for m in months], columns=["iso3", "month", "premium"])
    gdp = pd.DataFrame([(c, y, float(rng.uniform(500, 50000))) for c in countries for y in years], columns=["iso3", "year", "gdp_pc"])
    return {"trade": trade, "parallel": parallel, "gdp": gdp}


def _contagion(s: dict, c: dict, lags: ContagionLags = ContagionLags()) -> pd.DataFrame:
    return contagion_features.build_contagion(s["fx"], s["eur"], c["trade"], _struct(s), c["parallel"], lags)


def _corrupt_contagion(c: dict, trade_from=None, parallel_after=None, gdp_from=None) -> dict:
    out = dict(c)
    if trade_from is not None:
        out["trade"] = _corrupt(c["trade"], c["trade"].year >= trade_from, ["exports", "imports"], seed=41).fillna({"exports": 1e9, "imports": 1e9})
    if parallel_after is not None:
        out["parallel"] = _corrupt(c["parallel"], c["parallel"].month > parallel_after, ["premium"], seed=42)
    if gdp_from is not None:
        out["gdp"] = _corrupt(c["gdp"], c["gdp"].year >= gdp_from, ["gdp_pc"], seed=43)
    return out


def test_future_blindness_fx_contagion_features():
    """Строки ≤ as_of не меняются, если испортить курсы (свои и партнёров) после as_of,
    торговлю текущего и будущих лет и параллельный курс после as_of."""
    months = pd.period_range("1993-01", "2014-12", freq="M")
    s, c = synth_fx_struct(KR_COUNTRIES, months), synth_contagion(KR_COUNTRIES, months)
    as_of = pd.Period("2010-09", freq="M")
    clean = _contagion(s, c)
    bad = _contagion(_future_struct(s, as_of), _corrupt_contagion(c, trade_from=as_of.year, parallel_after=as_of))
    past = lambda d: d[d.month <= as_of].sort_values(["iso3", "month"])  # noqa: E731
    now = clean[clean.month == as_of]
    assert all(now[f].nunique() > 1 for f in ("pshock3", "pshock12", "pmax6", "pshock_peg6", "par_prem", "pgap24", "pkey24", "pkey_peg24"))
    _same(past(clean), past(bad), contagion_features.COLUMNS)
    assert _differs(clean[clean.month > as_of], bad[bad.month > as_of], contagion_features.COLUMNS), "контроль: порча будущего обязана быть видна"


@pytest.mark.parametrize("as_of", ["2010-03", "2010-09"])
def test_available_from_fx_contagion_features(as_of):
    """Торговые веса на строке месяца m — по годам, вышедшим к m (`imts_annual`: в марте 2010 —
    по 2008-й, в сентябре — по 2009-й); премия параллельного курса — не позже m − 1."""
    months = pd.period_range("1993-01", "2014-12", freq="M")
    s, c = synth_fx_struct(KR_COUNTRIES, months), synth_contagion(KR_COUNTRIES, months)
    as_of = pd.Period(as_of, freq="M")
    last_year = int(temporal.last_available_year_in_month("imts_annual", pd.Series([as_of])).iloc[0])
    assert last_year == (2008 if as_of.month < 7 else 2009)
    bad_c = _corrupt_contagion(c, trade_from=last_year + 1, parallel_after=as_of - temporal.monthly_lag("parallel_premium"))
    row = lambda d: d[d.month == as_of].sort_values("iso3")  # noqa: E731
    _same(row(_contagion(s, c)), row(_contagion(s, bad_c)), contagion_features.COLUMNS)
    # контроль: без лагов та же порча признаки меняет
    no_lag = ContagionLags(trade_source=None, parallel_months=0)
    assert _differs(row(_contagion(s, c, no_lag)), row(_contagion(s, bad_c, no_lag)), ["pshock3", "pmax6"])
    assert _differs(row(_contagion(s, c, no_lag)), row(_contagion(s, bad_c, no_lag)), ["pgap24", "pkey24"])
    assert _differs(row(_contagion(s, c, no_lag)), row(_contagion(s, bad_c, no_lag)), ["par_prem"])


def test_contagion_sees_partner_crash_same_month_not_earlier():
    """Смысл признака: главный партнёр обесценился на 50 % в месяце m — у страны с неподвижным
    курсом `pshock3` и `pmax3` положительны в m (и только с m), у самого партнёра — отрицательны."""
    months = pd.period_range("2000-01", "2008-12", freq="M")
    crash = pd.Period("2005-06", freq="M")
    fx = pd.concat([pd.DataFrame({"iso3": "AAA", "month": months, "fx": 10.0}),
                    pd.DataFrame({"iso3": "BBB", "month": months, "fx": np.where(months >= crash, 15.0, 10.0)}),
                    pd.DataFrame({"iso3": "CCC", "month": months, "fx": 5.0})], ignore_index=True)
    eur = pd.DataFrame({"month": months, "eur_usd": 1.0})
    trade = pd.DataFrame([("AAA", "BBB", y, 80.0, 0.0) for y in range(1996, 2009)] + [("AAA", "CCC", y, 20.0, 0.0) for y in range(1996, 2009)]
                         + [("BBB", "AAA", y, 80.0, 0.0) for y in range(1996, 2009)], columns=["iso3", "partner", "year", "exports", "imports"])
    struct = pd.DataFrame({"iso3": "AAA", "month": months, "peg_share12": 1.0, "ctot_d12": 0.0, "ctot_missing": 0})
    t = contagion_features.build_contagion(fx, eur, trade, struct).set_index(["iso3", "month"])
    assert t.loc[("AAA", crash - 1), "pshock3"] == 0 and t.loc[("AAA", crash - 1), "pmax3"] == 0
    assert np.isclose(t.loc[("AAA", crash), "pshock3"], 0.8 * np.log(1.5))
    assert np.isclose(t.loc[("AAA", crash), "pmax3"], np.log(1.5)) and np.isclose(t.loc[("AAA", crash), "pshock_peg3"], 0.8 * np.log(1.5))
    assert np.isclose(t.loc[("AAA", crash + 3), "pshock3"], 0) and np.isclose(t.loc[("AAA", crash + 3), "pshock6"], 0.8 * np.log(1.5))
    assert t.loc[("BBB", crash), "pshock3"] < 0 and t.loc[("BBB", crash), "pmax3"] == 0
    # накопленный шок: до обвала ноль; после — держится, когда окна изменений его уже забыли
    assert t.loc[("AAA", crash - 1), "pgap24"] == 0 and t.loc[("AAA", crash - 1), "pkey24"] == 0
    late = crash + 15
    assert np.isclose(t.loc[("AAA", late), "pshock12"], 0)
    assert np.isclose(t.loc[("AAA", late), "pgap24"], 0.8 * np.log(1.5)) and np.isclose(t.loc[("AAA", late), "pkey24"], np.log(1.5))
    assert np.isclose(t.loc[("AAA", late), "pkey_peg24"], np.log(1.5))
    assert np.isclose(t.loc[("AAA", crash + 30), "pgap24"], 0), "через 24 месяца нижняя точка окна — уже новый уровень"
    assert t.loc[("BBB", late), "pgap24"] == 0 and t.loc[("BBB", late), "pkey24"] == 0


@pytest.mark.parametrize("h", [3, 12])
def test_future_blindness_fx_contagion_monthly(h):
    """Прогноз `econ_contagion.forecast_asof` на горизонте h не меняется от порчи будущего
    (курсы, макро, война, торговля, параллельный курс, ВВП на душу)."""
    raw, annual, war = synth_monthly()
    months = pd.period_range("1993-01", "2014-12", freq="M")
    s, c = synth_fx_struct(KR_COUNTRIES, months, fx=raw), synth_contagion(KR_COUNTRIES, months)
    as_of = pd.Period("2010-06", freq="M")

    def run(raw, annual, war, s, c, name):
        p = econ_contagion.build_panel(_struct(s), _contagion(s, c), c["gdp"], raw, annual, war, ctot=s["ctot"])
        return econ_contagion.forecast_asof(econ_contagion.with_horizon(p, h), as_of, name, h).sort_values("iso3")

    bad_raw = _corrupt(raw, raw.month > as_of, KR_COLS)
    bad = (bad_raw, _corrupt(annual, annual.year >= as_of.year, ["value"]),
           _corrupt(war, war.month > as_of, ["war25_now", "war1000_now", "war_onset12"]),
           _future_struct({**s, "fx": bad_raw[["iso3", "month", "fx"]]}, as_of),
           _corrupt_contagion(c, trade_from=as_of.year, parallel_after=as_of, gdp_from=as_of.year - 1))
    for name in ("full", "par", "struct_level", "all_brk", "struct_brk_rt", "res_swap"):
        clean = run(raw, annual, war, s, c, name)
        assert clean.logit.notna().all() and clean.logit.nunique() > 1
        _same(clean, run(*bad, name), ["logit", "C", *contagion_features.COLUMNS, "gdp_pc_log", *break_features.BRK, *break_features.BRK_RT])
    ref = run(raw, annual, war, s, c, "ref")
    assert _differs(ref, run(raw, annual, war, s, c, "full"), ["logit"]), "контроль: набор обязан менять прогноз"
    if h == 12:  # на горизонте 12 месяцев эталон — та же логрегрессия, что в боевом контуре
        battle = econ_kr.forecast_asof(econ_kr.build_panel(raw, annual, war, target="deval15"), as_of).sort_values("iso3")
        _same(ref, battle, ["logit"])


def test_horizon_label_looks_h_months_ahead():
    """Метка горизонта h смотрит ровно в m+1…m+h; по умолчанию — прежние 12 месяцев."""
    raw, annual, war = synth_monthly()
    p = econ_kr.build_panel(raw, annual, war, target="deval15")
    assert kr_targets.add_target(p, "deval15").y.equals(p.y)
    for h in (3, 6):
        q = kr_targets.add_target(p, "deval15", h=h)
        fwd = sum(q.groupby("iso3").deval15.shift(-k) for k in range(1, h + 1))
        assert ((q.y == 1) == (fwd > 0))[fwd.notna()].all() and q.y[fwd.isna()].isna().all()
        assert (q.y.fillna(0) <= p.y.fillna(1)).all(), "кризис за h месяцев — подмножество кризиса за 12"


def test_available_from_break_features():
    """Сырьевые признаки разрыва привязки на строке месяца m не используют индекс позже m − 6
    (`imf_ctot`); вариант чувствительности `_rt` — позже m − 1. Контроль: без лага порча видна."""
    months = pd.period_range("1993-01", "2014-12", freq="M")
    ctot = synth_fx_struct(KR_COUNTRIES, months)["ctot"]
    as_of = pd.Period("2010-06", freq="M")
    row = lambda d: d[d.month == as_of].sort_values("iso3")  # noqa: E731
    for lag in (temporal.monthly_lag("imf_ctot"), break_features.RT_LAG):
        bad = _corrupt(ctot, ctot.month > as_of - lag, ["ctot"], seed=51)
        clean_t = break_features.ctot_table(ctot, lag)
        assert all(row(clean_t)[c].nunique() > 1 for c in break_features.CTOT_COLS)
        _same(row(clean_t), row(break_features.ctot_table(bad, lag)), break_features.CTOT_COLS)
        assert _differs(row(break_features.ctot_table(ctot, 0)), row(break_features.ctot_table(bad, 0)), break_features.CTOT_COLS)


def test_feature_sets_drop_and_add():
    """`DROP` действительно выбрасывает боевой индикатор, а `SETS` добавляет свои: наборы
    проверки «нужен ли линейный член резервов» отличаются ровно этим."""
    ref = econ_contagion.features("ref")
    assert ref == list(kr_features.INDICATORS) and "reserves_g12" in ref
    assert econ_contagion.features("res_drop") == [i for i in ref if i != "reserves_g12"]
    assert econ_contagion.features("res_both") == [*ref, "peg_resfall"]
    assert econ_contagion.features("res_swap") == [*[i for i in ref if i != "reserves_g12"], "peg_resfall"]
    swap, both = econ_contagion.features("all_brk_swap"), econ_contagion.features("all_brk")
    assert set(both) - set(swap) == {"reserves_g12"} and len(swap) == len(both) - 1


# ---------------------------------------------------------------------------
# 8. Месячная модель войн, живые стычки, доводка диад (чекпоинт 4.2)
# ---------------------------------------------------------------------------

MONTH_LAST = pd.Period("2013-12", freq="M")


def synth_month(seed: int = 41) -> dict:
    """Синтетика месячного контура войн: события с кодами государств-сторон (`gwnoa` / `gwnob`:
    часть — «правительство против правительства», часть — чужая армия на территории), годовые
    структурные признаки и население."""
    s = synth_dyad()
    rng = np.random.default_rng(seed)
    ged = s["ged"].copy()
    n, st = len(ged), (s["ged"].type_of_violence == 1).to_numpy()
    u = rng.random(n)
    other = np.where(ged.gw < 20, ged.gw + 1, ged.gw - 1)
    ged["gwnoa"] = np.where(st, np.where(u < 0.15, other, ged.gw).astype(str), None)  # 15 % — вторжение соседа
    ged["gwnob"] = np.where(st & (u > 0.7), other.astype(float), np.nan)  # 30 % — прямое столкновение
    y0, y1 = s["span"]
    cy = pd.DataFrame([(g, y) for g in WAR_STATES for y in range(y0 - 3, y1 + 1)], columns=["gw", "year"])
    struct = cy.copy()
    for c in features_pitf.PITF_FEATURES:
        struct[c] = rng.normal(0, 1, len(cy))
    pop = cy.assign(population=np.exp(rng.normal(15, 1, len(cy))))
    return {**s, "ged": ged, "struct": struct, "pop": pop}


def _month_features(s: dict, lag: int = 1, **over) -> pd.DataFrame:
    s = {**s, **over}
    p = month_features.build_panel(s["ged"], s["states"], s["parties"], s["neighbors"], s["span"][0], MONTH_LAST)
    live = live_pairs.country_month(live_pairs.pair_events(s["ged"], dyads.opponents(s["acd"]), acd_lag=panel.ACD_LAG))
    return month_features.build_features(p, s["neighbors"], s["struct"], live, s["pop"], lag=lag)


def _bad_events(ged: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    bad = _corrupt(ged, mask, ["best", "deaths_state"])
    bad[["best", "deaths_state"]] = bad[["best", "deaths_state"]].fillna(5000).round()
    return bad


def _ym(ged: pd.DataFrame) -> pd.Series:
    return pd.Series(pd.PeriodIndex.from_ordinals(live_pairs.month_ordinal(ged.year, ged.month), freq="M"), index=ged.index)


MONTH_COLS = [*month_features.ALL_FEATURES, "quiet_known", "ac_quiet_known"]


def test_available_from_war_month_features():
    """Лаг событий UCDP: признаки строки месяца m не зависят от событий месяца m и позже
    (файл Candidate за месяц m выходит в m+1), от списка сторон ACD за текущий год, от
    структурных признаков текущего года и населения двух последних лет."""
    assert temporal.monthly_lag("ucdp_events") == 1 == month_features.EVENT_LAG == live_pairs.EVENT_LAG
    s = synth_month()
    as_of = pd.Period("2008-06", freq="M")
    row = lambda f: f[f.ym == as_of].sort_values("gw")  # noqa: E731
    ged = s["ged"]
    bad_ged = _bad_events(ged, _ym(ged) >= as_of)
    bad_parties, bad_acd = s["parties"].copy(), s["acd"].copy()
    now = bad_parties.year >= as_of.year
    bad_parties.loc[now, "parties"] = bad_parties.loc[now, "parties"].map(lambda _: WAR_STATES)
    bad_acd.loc[bad_acd.year >= as_of.year, "gwno_b_2nd"] = ", ".join(map(str, WAR_STATES))
    bad_struct = _corrupt(s["struct"], s["struct"].year >= as_of.year, features_pitf.PITF_FEATURES)
    bad_pop = _corrupt(s["pop"], s["pop"].year >= as_of.year - 1, ["population"])
    clean = row(_month_features(s))
    assert all(clean[c].nunique() > 1 for c in ("sb_ld_12", "ns_ld_12", "os_ld_12", "nb_low_ld12", "lv_dir_ld12", "lv_in_ld12", "lv_px_ld12", "ac_m12"))
    _same(clean, row(_month_features(s, ged=bad_ged, parties=bad_parties, acd=bad_acd, struct=bad_struct, pop=bad_pop)), MONTH_COLS)
    # контроль: без лага события самого месяца m обязаны быть видны
    only_m = _bad_events(ged, _ym(ged) == as_of)
    assert _differs(row(_month_features(s, lag=0)), row(_month_features(s, lag=0, ged=only_m)), ["sb_ld_3", "ld_war_1", "ld_best_1"])
    _same(clean, row(_month_features(s, ged=only_m)), MONTH_COLS)
    # контроль самой проверки: порча прошлого меняет признаки
    assert _differs(clean, row(_month_features(s, ged=_bad_events(ged, _ym(ged) < as_of))), ["sb_ld_12", "ld_war_60"])


@pytest.mark.parametrize("h", [3, 12])
def test_future_blindness_war_monthly_model(h):
    """Месячный прогноз войны в точке as_of не меняется, если испортить все события месяца
    as_of и позже: обучение берёт только строки с закрывшимся окном и вышедшими событиями
    (t + h + lag ≤ as_of), признаки — события по as_of − lag."""
    s = synth_month()
    as_of = pd.Period("2009-06", freq="M")
    feats = model_month.SETS["full"]

    def run(lag_label: int = 1, **over):
        f = _month_features(s, **over)
        return model_month.forecast_asof(f, as_of, feats, h, lag=lag_label, C=0.1).sort_values("gw")

    ged = s["ged"]
    bad_ged = _bad_events(ged, _ym(ged) >= as_of)
    bad_parties = s["parties"].copy()
    fut = bad_parties.year >= as_of.year
    bad_parties.loc[fut, "parties"] = bad_parties.loc[fut, "parties"].map(lambda _: WAR_STATES)
    clean = run()
    assert clean.p_onset.nunique() > 1 and clean.p_cont.nunique() > 1 and len(clean) == len(WAR_STATES)
    _same(clean, run(ged=bad_ged, parties=bad_parties), ["p", "p_onset", "p_cont"])
    # контроль: порча последнего месяца, чьи события уже вышли (as_of − 1), прогноз меняет;
    # если «забыть» лаг в правиле известности метки — в обучение попадает окно с испорченным месяцем
    assert _differs(clean, run(ged=_bad_events(ged, _ym(ged) == as_of - 1)), ["p_onset", "p_cont"])
    only_m = _bad_events(ged, _ym(ged) == as_of)
    assert _differs(run(lag_label=0), run(lag_label=0, ged=only_m), ["p_onset", "p_cont"])


@pytest.mark.parametrize("h", [3, 6, 12])
def test_month_label_and_evaluable_sync(h):
    """Метка горизонта h — война в месяцы m+1 … m+h и ничего больше; за концом хроники —
    пропуск, а не ноль; на «мирных» строках положительная метка — всегда начало войны по
    правилу 12 тихих месяцев (целевая и отбор строк согласованы)."""
    f = _month_features(synth_month())
    end = MONTH_LAST - 6
    d = model_month.with_label(f, h, label_end=end)
    assert d[d.ym + h > end].y.isna().all() and d[d.ym + h <= end].y.notna().all()
    w = d.pivot(index="o", columns="gw", values="state_war25")
    fut = pd.concat([w.shift(-k) for k in range(1, h + 1)]).groupby(level=0).max().stack().rename("want").reset_index()
    m = d[d.y.notna()].merge(fut, on=["o", "gw"])
    assert len(m) > 1000 and (m.y == m.want).all() and 0 < m.y.mean() < 1
    on = d.pivot(index="o", columns="gw", values="onset_state_war25")
    fon = pd.concat([on.shift(-k) for k in range(1, h + 1)]).groupby(level=0).max().stack().rename("onset_in_window").reset_index()
    q = d[d.y.notna() & (d.quiet_true == 1)].merge(fon, on=["o", "gw"])
    assert q.y.sum() > 5 and (q.y == q.onset_in_window).all()


def test_live_pairs_kinds_and_lag():
    """Живая хроника пар: прямое столкновение, прокси и вторжение не смешиваются и не
    считаются дважды; годовой признак строки T не видит событий декабря T (лаг 1 месяц)."""
    ev = pd.DataFrame({
        "gw": [10, 10, 12, 10], "year": [2000, 2000, 2000, 2000], "month": [3, 5, 7, 12], "type_of_violence": [1, 1, 1, 1],
        "best": [5, 7, 3, 9], "conflict_new_id": [1, 2, 3, 1], "gwnoa": ["10", "10", "11", "10"], "gwnob": [11.0, np.nan, np.nan, 11.0]})
    opp = pd.DataFrame({"conflict_id": [2], "year": [2000], "a": [10], "b": [13]})
    opp = pd.concat([opp, opp.rename(columns={"a": "b", "b": "a"})])
    pe = live_pairs.pair_events(ev, opp, acd_lag=0)
    got = {(r.a, r.b, r.kind): r.deaths for r in pe.groupby(["a", "b", "kind"], as_index=False).deaths.sum().itertuples()}
    assert got == {(10, 11, "direct"): 14, (11, 10, "direct"): 14, (10, 13, "proxy"): 7, (13, 10, "proxy"): 7,
                   (12, 11, "inc_in"): 3, (11, 12, "inc_out"): 3}
    assert live_pairs.pair_events(ev, opp, acd_lag=1).query("kind == 'proxy'").empty  # список сторон 2000 года в 2000-м ещё не вышел
    f1 = live_pairs.annual_features(pe, [2000], lag=1).set_index(["a", "b"])
    f0 = live_pairs.annual_features(pe, [2000], lag=0).set_index(["a", "b"])
    assert np.isclose(f1.loc[(10, 11), "live_ld3"], np.log1p(5)) and np.isclose(f0.loc[(10, 11), "live_ld3"], np.log1p(14))
    cm = live_pairs.country_month(pe)
    assert cm[cm.gw == 12].lv_in_d.sum() == 3 and cm[cm.gw == 11].lv_out_d.sum() == 3 and cm[cm.gw == 10].lv_dir_d.sum() == 14


def _dyad_live(s: dict, event_lag: int = 1, **over) -> pd.DataFrame:
    s = {**s, **over}
    y0, y1 = s["span"]
    grid = dyads.build_grid(s["states"], s["contig"], y0, y1, powers=DYAD_POWERS)
    opp = dyads.opponents(s["acd"])
    dw = dyads.dyad_wars(s["ged"], opp, s["neighbors"])
    return model_dyad_live.build_features(grid, s["states"], opp, dw, s["ged"], s["unga"], s["mid"], s["cap"], event_lag=event_lag,
                                          mid_last_year=y1 - 2)


def test_future_blindness_dyad_live():
    """Доводка диад: прогноз пары, сырая свёртка и откалиброванный `p_ext` в точке T не зависят
    от событий после ноября T (декабрьский файл к 1 января не вышел), от сторон войн после T
    и от страновых меток внешнего нападения, ещё не известных к T."""
    s = synth_month()
    T = 2008
    year, _ = _war_panels(s["ged"], s["states"], s["parties"], s["neighbors"], s["span"])

    def run(event_lag: int = 1, labels_bad: bool = False, **over):
        d = _dyad_live(s, event_lag, **over)
        src = {**s, **over}
        ext = dyads.external_wars(dyads.dyad_wars(src["ged"], dyads.opponents(src["acd"]), s["neighbors"]))
        labels = model_dyad_live.ext_onset_labels(year, ext)
        if labels_bad:
            labels.loc[labels.year >= T, "ext_onset_next"] = 1 - labels.loc[labels.year >= T, "ext_onset_next"].fillna(0)
        c, pairs = model_dyad_live.external_asof(d, labels, T, C=0.1, return_pairs=True)
        return c.sort_values("gw"), pairs.sort_values(["a", "b"])

    ged, acd = s["ged"], s["acd"]
    late = (ged.year > T) | ((ged.year == T) & (ged.month == 12) & (ged.type_of_violence == 1))
    live_only = late & (ged.year == T)  # декабрь T: годовая целевая пары его видит, живые признаки — нет
    bad_acd = acd.copy()
    bad_acd.loc[bad_acd.year > T, "gwno_b_2nd"] = ", ".join(map(str, WAR_STATES))
    c0, p0 = run()
    assert c0.s.nunique() > 1 and (c0.s != c0.p_ext).any() and p0[live_pairs.LIVE_FEATURES[0]].max() > 0 and len(p0) > 50
    c1, p1 = run(ged=_bad_events(ged, ged.year > T), acd=bad_acd, labels_bad=True)
    _same(c0, c1, ["s", "p_ext"])
    _same(p0, p1, ["p_dyad", *model_dyad_live.LIVE_SET])
    # декабрь T: живые признаки строки T его не видят; без лага — видят
    dec = ged.copy()
    dec.loc[live_only, "gwnob"] = np.where(dec.loc[live_only, "gw"] < 20, dec.loc[live_only, "gw"] + 1, dec.loc[live_only, "gw"] - 1).astype(float)
    def feats(lag, g):
        d = _dyad_live(s, lag, ged=g)
        return d[d.year == T].sort_values(["a", "b"])

    _same(feats(1, ged), feats(1, dec), live_pairs.LIVE_FEATURES)
    assert _differs(feats(0, ged), feats(0, dec), live_pairs.LIVE_FEATURES), "без лага декабрьские события обязаны быть видны"
    # раздельные модели: пара без границы не получает модель соседей
    test = model_dyad_live.forecast_asof(_dyad_live(s), T, C=0.1)
    assert set(test.model) <= {"near", "far", "pooled"} and not ((test.near == 0) & (test.model == "near")).any()
