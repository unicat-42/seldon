"""Исследование «идеология и партия у руля» (`research/party_*`): утечки будущего и лаги публикации.

Запуск:
    uv run pytest tests/test_party_study.py

Все проверки — на маленьких синтетических таблицах, без обращения к диску:

1. `test_party_publication_lags` — лаги `gli_annual` и `vparty_annual` записаны в
   `core/temporal.py`: год Y известен с 1 апреля Y+1, при прогнозе «на 1 января» на строке T
   стоят данные за T−1.
2. `test_party_features_future_blind` — портим ВСЕ исходные данные за годы ≥ Y: строки T ≤ Y не
   меняются (причина раньше следствия: кризис года T не может «вызвать» смену партии на строке T).
   Контроль: с нулевым лагом та же порча строки меняет.
3. `test_change_is_calendar_based` — «смена за 2 года» считается по календарю, а не по соседним
   строкам: дыра в данных даёт пропуск, а не сравнение с позапрошлым десятилетием.
4. `test_archive_ends_without_zeros_or_flags` — после конца архивной базы признаки — пропуск,
   строка вне чистого окна (`*_ok` = False); нулей и флага «пропуск = 1» там нет.
5. `test_vparty_carry_forward` — партия, пришедшая на выборах года Y, появляется на строке Y+1,
   тянется до следующих выборов и не дальше конца базы.
6. `test_ruling_party_is_senior_partner` — правящая партия — старший партнёр правительства,
   а не самая большая фракция.
7. `test_monthly_attach_respects_lag` — в месячной панели январь–март года t видят год t−2,
   апрель–декабрь — t−1.
8. `test_legitimation_block_ends_2023` — блок легитимации V-Dem v16 кончается 2023 годом: строка
   2025 года не получает «протянутых» значений.
9. `test_war_forecast_with_party_features_is_future_blind` — прогноз модели войн с партийными
   признаками в точке T не меняется, если испортить все строки после T.
10. `test_clean_window_drops_rows_after_archive_end` — чистое окно архивных наборов.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon.core import temporal
from seldon.research import party_features as pf
from seldon.research import party_study as ps
from seldon.sources import vparty as vparty_source

YEARS = range(1990, 2026)
GWS = (1, 2, 3, 4)


def _vdem(rng) -> pd.DataFrame:
    rows = []
    for gw in GWS:
        name, regime = f"leader{gw}_0", 100 * gw
        for y in YEARS:
            if rng.random() < 0.15:
                name = f"leader{gw}_{y}"
            if rng.random() < 0.05:
                regime += 1
            r = {"gw": gw, "year": y, "v2exnamhos": name, "v2exnamhog": None, "v2ex_hosw": 1.0, "v2regidnr": float(regime),
                 "v2xps_party": rng.random(), "v2x_polyarchy": rng.random(), "v2exparhos": "p",
                 "v2elturnhos": float(rng.integers(0, 3)) if y % 4 == gw % 4 else np.nan, "v2elturnhog": np.nan}
            for c in ("v2exl_legitideol", "v2exl_legitlead", "v2exl_legitperf"):
                r[c] = rng.normal() if y <= 2023 else np.nan
            for i in range(5):
                r[f"v2exl_legitideolcr_{i}"] = rng.random() if y <= 2023 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def _gli(rng) -> pd.DataFrame:
    camps = ["leftist", "centrist", "rightist", "none"]
    rows = []
    for gw in GWS:
        camp, leader = camps[gw % 4], "a"
        for y in range(1990, 2021):
            if rng.random() < 0.2:
                camp, leader = camps[int(rng.integers(0, 4))], f"l{y}"
            rows.append({"gw": gw, "year": y, "leader": leader, "camp": camp})
    return pd.DataFrame(rows)


def _ruling(rng) -> pd.DataFrame:
    rows = []
    for gw in GWS:
        for y in range(1990 + gw % 4, 2020, 4):
            rows.append({"gw": gw, "year": y, "party": f"party{int(rng.integers(0, 3))}", "v2xpa_popul": rng.random(),
                         "v2xpa_antiplural": rng.random(), "v2pariglef": rng.normal()})
    return pd.DataFrame(rows)


def _sources(seed: int = 0):
    rng = np.random.default_rng(seed)
    return _vdem(rng), _gli(rng), _ruling(rng)


def _spoil(v, g, r, year_from: int, seed: int = 7):
    """Испортить все исходные данные за годы ≥ `year_from`: числа — шумом, имена и лагеря — заменой."""
    rng = np.random.default_rng(seed)
    v, g, r = v.copy(), g.copy(), r.copy()
    fv = v.year >= year_from
    for c in v.columns:
        if c in ("gw", "year"):
            continue
        if not pd.api.types.is_numeric_dtype(v[c]):
            v.loc[fv, c] = [f"x{i}" for i in range(int(fv.sum()))]
        else:
            v.loc[fv, c] = rng.normal(0, 50, int(fv.sum()))
    fg = g.year >= year_from
    g.loc[fg, "camp"] = rng.choice(["leftist", "rightist", "none"], int(fg.sum()))
    g.loc[fg, "leader"] = [f"y{i}" for i in range(int(fg.sum()))]
    fr = r.year >= year_from
    r.loc[fr, "party"] = [f"z{i}" for i in range(int(fr.sum()))]
    for c in vparty_source.SCORES:
        r.loc[fr, c] = rng.normal(0, 50, int(fr.sum()))
    return v, g, r


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    if a.shape != b.shape:
        return False
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) or pd.api.types.is_bool_dtype(a[c]):
            if not np.allclose(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), atol=1e-12, equal_nan=True):
                return False
        elif not (a[c].fillna("") == b[c].fillna("")).all():
            return False
    return True


def test_party_publication_lags():
    for src in ("gli_annual", "vparty_annual", "vdem_annual"):
        assert temporal.available_from(src, 2019) == pd.Timestamp("2020-04-01")
        assert temporal.publication_lag(src, "jan1") == 1
    lags = pf.PartyLags()
    assert (lags.vdem, lags.gli, lags.vparty) == (1, 1, 1)


def test_party_features_future_blind():
    v, g, r = _sources()
    base = pf.build_annual(v, g, r)
    Y = 2008
    bad = pf.build_annual(*_spoil(v, g, r, Y))
    assert _same(base[base.year <= Y], bad[bad.year <= Y])
    # контроль: без лага публикации те же строки от порчи года Y зависят
    lag0 = pf.PartyLags(vdem=0, gli=0, vparty=0)
    a, b = pf.build_annual(v, g, r, lag0), pf.build_annual(*_spoil(v, g, r, Y), lag0)
    assert not _same(a[a.year <= Y], b[b.year <= Y])


def test_change_is_calendar_based():
    v, g, r = _sources(1)
    v = v[~((v.gw == 1) & v.year.between(2000, 2009))]  # дыра в десять лет
    t = pf.live_table(v)
    row = t[(t.gw == 1) & (t.year == 2010)].iloc[0]
    assert np.isnan(row.pl_ideol_d2) and np.isnan(row.pl_turn) and np.isnan(row.pl_leader_new)
    g = g[~((g.gw == 2) & g.year.between(2000, 2009))]
    row = pf.gli_table(g).query("gw == 2 and year == 2010").iloc[0]
    assert np.isnan(row.gl_switch)


def test_archive_ends_without_zeros_or_flags():
    v, g, r = _sources(2)
    f = pf.build_annual(v, g, r)
    gli_last = int(g.year.max()) + 1          # последняя строка с данными GLI (лаг 1 год)
    vp_last = vparty_source.VPARTY_LAST_YEAR + 1
    after = f[f.year > gli_last]
    assert len(after) and not after.gli_ok.any() and after[pf.ARCH_GLI].isna().all().all()
    after = f[f.year > vp_last]
    assert len(after) and not after.vp_ok.any() and after[pf.ARCH_VP].isna().all().all()
    assert f[f.year <= gli_last].gli_ok.any() and f[f.year <= vp_last].vp_ok.any()
    assert not any(c.endswith("_missing") for c in f.columns)  # флаги пропуска ставит только чистое окно


def test_vparty_carry_forward():
    r = pd.DataFrame({"gw": [1, 1, 1], "year": [2000, 2004, 2016], "party": ["a", "b", "b"],
                      "v2xpa_popul": [0.1, 0.9, 0.8], "v2xpa_antiplural": [0.2, 0.3, 0.4], "v2pariglef": [1.0, -1.0, -1.0]})
    t = pf.vparty_table(r).set_index("year")
    assert t.loc[2000, "vp_popul"] == 0.1 and t.loc[2003, "vp_popul"] == 0.1 and t.loc[2004, "vp_popul"] == 0.9
    assert t.loc[2004, "vp_new"] == 1 and t.loc[2005, "vp_new"] == 1 and t.loc[2006, "vp_new"] == 0
    assert t.loc[2016, "vp_new"] == 0 and t.index.max() == vparty_source.VPARTY_LAST_YEAR
    v, g, _ = _sources(3)
    f = pf.build_annual(v[v.gw == 1], g[g.gw == 1], r).set_index("year")
    assert f.loc[2004, "vp_popul"] == 0.1 and f.loc[2005, "vp_popul"] == 0.9  # выборы-2004 видны со строки 2005
    assert np.isnan(f.loc[vparty_source.VPARTY_LAST_YEAR + 2, "vp_popul"])


def test_ruling_party_is_senior_partner():
    v = pd.DataFrame({"gw": [1, 1, 1, 2], "year": [2000, 2000, 2000, 2000], "party": ["big_opposition", "junior", "senior", "only_opposition"],
                      "govsup": [3.0, 1.0, 0.0, 3.0], "seatshare": [60.0, 30.0, 10.0, 100.0],
                      "v2xpa_popul": [0.9, 0.5, 0.1, 0.7], "v2xpa_antiplural": [0.9, 0.5, 0.1, 0.7], "v2pariglef": [0.0, 0.0, 0.0, 0.0]})
    t = vparty_source.ruling_party(v).set_index("gw")
    assert t.loc[1, "party"] == "senior" and t.loc[1, "v2xpa_popul"] == 0.1
    assert pd.isna(t.loc[2, "party"]) and np.isnan(t.loc[2, "v2xpa_popul"])


def test_monthly_attach_respects_lag():
    v, g, r = _sources(4)
    tables = pf.own_year_tables(v, g, r)
    months = pd.DataFrame({"gw": 1, "month": pd.period_range("2010-01", "2010-12", freq="M")})
    m = pf.attach_monthly(months, tables)
    live = tables["live"].set_index(["gw", "year"])
    assert np.isclose(m.pl_ideol.iloc[0], live.loc[(1, 2008), "pl_ideol"]) and np.isclose(m.pl_ideol.iloc[2], live.loc[(1, 2008), "pl_ideol"])
    assert np.isclose(m.pl_ideol.iloc[3], live.loc[(1, 2009), "pl_ideol"]) and np.isclose(m.pl_ideol.iloc[11], live.loc[(1, 2009), "pl_ideol"])
    # порча данных за 2010 год и позже ни один месяц 2010 года не меняет
    v2, g2, r2 = _spoil(v, g, r, 2010)
    m2 = pf.attach_monthly(months, pf.own_year_tables(v2, g2, r2))
    cols = [*pf.LIVE_LEGIT, *pf.LIVE_TURN, *pf.ARCH_GLI, *pf.ARCH_VP]
    assert _same(m[cols], m2[cols])


def test_legitimation_block_ends_2023():
    v, g, r = _sources(5)
    f = pf.build_annual(v, g, r)
    assert f[f.year == 2024].legit_ok.all() and not f[f.year == 2025].legit_ok.any()
    assert f[f.year == 2025][pf.LIVE_LEGIT].isna().all().all()  # не протянуто и не занулено
    assert f[f.year == 2025].pl_leader_new.notna().all()        # имя лидера в V-Dem есть по 2025


def _war_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    v, g, r = _sources(seed)
    rows = [{"gw": gw, "year": y, "country": str(gw), "x1": rng.normal(), "x2": rng.normal(), "years_since_war": 5.0,
             "state_war25": int(rng.random() < 0.3)} for gw in range(1, 41) for y in range(1990, 2021)]
    f = pd.DataFrame(rows)
    f["onset_state_war25"] = f.state_war25
    big_v = pd.concat([v.assign(gw=v.gw + 4 * k) for k in range(10)], ignore_index=True)
    big_g = pd.concat([g.assign(gw=g.gw + 4 * k) for k in range(10)], ignore_index=True)
    big_r = pd.concat([r.assign(gw=r.gw + 4 * k) for k in range(10)], ignore_index=True)
    return f, (big_v, big_g, big_r)


def test_war_forecast_with_party_features_is_future_blind():
    f, src = _war_panel()
    T = 2010
    feats = ["x1", "x2"]

    def forecast(f, src):
        p = ps.war_panel(f, pf.build_annual(*src))
        return ps.war_forecast_asof(p, T, feats, "live")[["gw", "p"]]

    base = forecast(f, src)
    bad_f = f.copy()
    fut = bad_f.year > T
    bad_f.loc[fut, ["x1", "x2"]] = 1e6
    bad_f.loc[fut, ["state_war25", "onset_state_war25"]] = 1 - bad_f.loc[fut, ["state_war25", "onset_state_war25"]]
    bad = forecast(bad_f, _spoil(*src, T))  # данные V-Dem за T к 1 января T+1 ещё не вышли
    assert _same(base, bad) and base.p.notna().all()


def test_clean_window_drops_rows_after_archive_end():
    v, g, r = _sources(6)
    f = pf.build_annual(v, g, r)
    w = pf.clean_window(f, "gli")
    assert w.year.max() == int(g.year.max()) + 1 and w.gli_ok.all()
    w = pf.clean_window(f, "vparty")
    assert w.year.max() <= vparty_source.VPARTY_LAST_YEAR + 1
    assert pf.clean_window(f, "live").year.max() == 2024  # блок легитимации: данные по 2023
