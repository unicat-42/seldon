"""Сетевое заражение конфликтов по соседским, этноязыковым и режимным графам (чекпоинт 4.3a).

Запуск:
    uv run python -m seldon network_features

Строит три графа связей стран `A ← B` (`A ≠ B`, оба государства существуют в периоде) строго по
данным, известным к моменту прогноза на строке месяца `m`:
  1. Соседи (`COW Direct Contiguity`, суша / море до 12 миль).
  2. Этноязыковой граф (`w_el = max(w_tek, w_lang)`):
     - `EPR TEK` за последний вышедший год (`epr_annual`, с флагом `epr_extended` после `2023`):
       сумма долей населения групп `A`, имеющих родню в `B` (`size_B ≥ 0.05`);
     - `CEPII GeoDist`: `0.5 * comlang_off + 0.5 * comlang_ethno`.
     Дальняя (не-соседская) компонента `w_el_far = w_el * (1 - is_neighbor)` проверяет, даёт ли
     этноязыковая сеть прирост сверх уже учтённых географических соседей.
  3. Культурно-режимный кластер (`w_cluster = max(w_el, is_neighbor) * w_aut_sim`), где
     `w_aut_sim = (1 - poly_A) * (1 - poly_B) * clip(1 - 2 * |poly_A - poly_B|, 0, 1)` по `V-Dem`
     за последний вышедший год (`vdem_annual`): высока для близких недемократий одной
     культурно-региональной сферы («Арабская весна», «цветные революции») и близка к 0 для
     устойчивых демократий.

Импульсы `B` и собственные шоки `A` берутся с честным лагом в 1 месяц (`core/temporal.py`):
  - живые импульсы `UCDP`: государственные бои (`type=1`), одностороннее насилие (`type=3`),
    новые начала войн и институциональные сдвиги `V-Dem` (`NET_EL`, `NET_CL`);
  - исторический срез протестов (`Mass Mobilization`, 1990–2020-03) и переворотов/смен лидера
    (`REIGN`, 1950–2021-08) с явными флагами конца хроники `mm_net_missing` и `reign_net_missing`
    (`POLIT_OWN`, `POLIT_NET`) — проверка, нужен ли живой протестный поток (`ACLED` / `GDELT`).

Выход:
    data/processed/war_network_month.parquet            месячные сетевые признаки (1989–2026-09)
    data/processed/features_network_country_year.parquet годовой срез (декабрь T) для `model_hist`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED
from seldon.core.temporal import last_available_year_in_month, monthly_lag
from seldon.domains.conflict.features_epr import extend_groups
from seldon.domains.conflict.features_vdem import vdem_table
from seldon.domains.conflict.live_pairs import month_ordinal
from seldon.domains.conflict.month_features import WAR, load_inputs
from seldon.sources.cepii import load_cepii
from seldon.sources.epr import EPR_LAST_YEAR, load_epr_group_years, load_tek
from seldon.sources.massmob import load_massmob
from seldon.sources.reign import load_reign
from seldon.sources.ucdp import CANDIDATE_LAST_MONTH, YEAR_FROM, YEAR_TO
from seldon.sources.vdem import load_vdem

MM_FIRST_MONTH = pd.Period("1990-01", freq="M")
MM_LAST_MONTH = pd.Period("2020-03", freq="M")
REIGN_LAST_MONTH = pd.Period("2021-08", freq="M")
TEK_MIN_B_SIZE = 0.05
REPRESSIVE_RESPONSES = frozenset({"shootings", "killings", "beatings"})

NET_EL = ["net_el_war12", "net_el_onset12", "net_el_os12", "net_el_vdem_d2"]
NET_CL = [*NET_EL, "net_cl_shock3", "net_cl_shock12", "net_vuln_cl", "vd_net_missing"]
POLIT_OWN = ["own_prot3", "own_prot_rep12", "mm_net_missing", "own_coup12", "reign_net_missing"]
POLIT_NET = ["net_cl_prot3", "net_cl_prot12", "net_cl_coup12"]
NETWORK_COLS = [*NET_CL, *POLIT_OWN, *POLIT_NET]


@dataclass(frozen=True)
class NetworkLags:
    """Лаги публикации источников для сетевых признаков (`core/temporal.py`)."""
    event_lag: int = monthly_lag("ucdp_events")
    vdem_source: str | None = "vdem_annual"
    epr_source: str | None = "epr_annual"
    mm_lag: int = monthly_lag("massmob_events")
    reign_lag: int = monthly_lag("reign_monthly")


def _known_year(ym: pd.Series, source: str | None) -> pd.Series:
    return ym.dt.year if source is None else last_available_year_in_month(source, ym)


def _country_month_matrix(df: pd.DataFrame, gws: list[int], ord_from: int, ord_to: int, cols: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Перевести таблицу `(gw, o, ...)` на плотную решётку `(N_gw, N_months)`."""
    n_gw, n_t = len(gws), ord_to - ord_from + 1
    gw_idx = {int(g): i for i, g in enumerate(gws)}
    alive = np.zeros((n_gw, n_t), dtype=bool)
    mats = {c: np.zeros((n_gw, n_t), dtype=float) for c in cols}
    g_i = df.gw.map(gw_idx)
    ok = g_i.notna() & (df.o >= ord_from) & (df.o <= ord_to)
    sub = df[ok]
    ii = g_i[ok].to_numpy(dtype=int)
    tt = (sub.o - ord_from).to_numpy(dtype=int)
    alive[ii, tt] = True
    for c in cols:
        mats[c][ii, tt] = sub[c].to_numpy(dtype=float)
    return mats, alive


def _lagged_roll_sum(mat: np.ndarray, lag: int, window: int) -> np.ndarray:
    """Скользящая сумма по оси времени (axis=1) за `window` месяцев, заканчивающаяся на `t - lag`."""
    n_gw, n_t = mat.shape
    out = np.zeros((n_gw, n_t), dtype=float)
    for k in range(lag, lag + window):
        if k == 0:
            out += mat
        elif k < n_t:
            out[:, k:] += mat[:, :-k]
    return np.maximum(out, 0.0)


def _lagged_roll_max(mat: np.ndarray, lag: int, window: int) -> np.ndarray:
    """Скользящий максимум по оси времени за `window` месяцев, заканчивающийся на `t - lag`."""
    shifted = pd.DataFrame(mat.T).shift(lag).fillna(0.0)
    return shifted.rolling(window, min_periods=1).max().to_numpy().T


def _ucdp_impulses(panel_df: pd.DataFrame, events: pd.DataFrame, gws: list[int], ord_from: int, ord_to: int, lag: int) -> dict[str, np.ndarray]:
    """Подготовить месячные импульсы насилия UCDP (с лагом `lag` месяцев)."""
    ev = events.copy()
    ev["o"] = month_ordinal(ev.year, ev.month)
    ev["ds"] = np.where(ev.type_of_violence == 1, ev.best, 0.0)
    ev["dos"] = np.where(ev.type_of_violence == 3, ev.best, 0.0)
    ev_m = ev.groupby(["gw", "o"], as_index=False)[["ds", "dos"]].sum()

    base = panel_df[["gw", "o", "deaths_state"]].merge(ev_m, on=["gw", "o"], how="left").fillna(0.0)
    # Начало боёв на своей территории (≥ 25 погибших в месяц после 12 месяцев < 25)
    base = base.sort_values(["gw", "o"])
    terr = (base.deaths_state >= WAR).astype(int)
    prev12 = terr.groupby(base.gw).transform(lambda s: s.shift(1).rolling(12, min_periods=1).max()).fillna(0)
    base["terr_onset"] = ((terr == 1) & (prev12 == 0)).astype(float)
    base["ds_use"] = np.maximum(base.deaths_state, base.ds)

    mats, _ = _country_month_matrix(base, gws, ord_from, ord_to, ["ds_use", "dos", "terr_onset"])
    return {
        "ds3": _lagged_roll_sum(mats["ds_use"], lag, 3),
        "ds12": _lagged_roll_sum(mats["ds_use"], lag, 12),
        "dos3": _lagged_roll_sum(mats["dos"], lag, 3),
        "dos12": _lagged_roll_sum(mats["dos"], lag, 12),
        "onset12": _lagged_roll_max(mats["terr_onset"], lag, 12),
    }


def _massmob_impulses(massmob: pd.DataFrame, gws: list[int], ord_from: int, ord_to: int, lag: int,
                      last_month: pd.Period) -> tuple[dict[str, np.ndarray], dict[int, int]]:
    """Подготовить месячные импульсы протестов Mass Mobilization (`≤ last_month`) с лагом `lag`.
    Второе значение — первый месяц (ординал), когда страна появилась в архиве (для `_coverage`)."""
    if len(massmob) == 0:
        z = np.zeros((len(gws), ord_to - ord_from + 1), dtype=float)
        return {"prot3": z, "prot_rep3": z, "prot_rep12": z}, {}
    m = massmob.dropna(subset=["startyear", "startmonth", "gw"]).copy()
    m["o"] = month_ordinal(m.startyear.astype(int), m.startmonth.astype(int))
    m = m[(m.o >= MM_FIRST_MONTH.ordinal) & (m.o <= last_month.ordinal)].copy()
    resp_cols = [c for c in m.columns if c.startswith("stateresponse")]
    rep_resp = np.zeros(len(m), dtype=bool)
    for c in resp_cols:
        rep_resp |= m[c].astype(str).str.lower().isin(REPRESSIVE_RESPONSES).to_numpy()
    part = pd.to_numeric(m.get("participants", np.nan), errors="coerce") >= 10_000
    viol = pd.to_numeric(m.get("protesterviolence", 0), errors="coerce").fillna(0) == 1
    m["prot_n"] = 1.0
    m["prot_rep"] = (rep_resp | part.to_numpy() | viol.to_numpy()).astype(float)
    agg = m.groupby(["gw", "o"], as_index=False)[["prot_n", "prot_rep"]].sum()
    first_seen = {int(g): int(o) for g, o in agg.groupby("gw").o.min().items()}
    mats, _ = _country_month_matrix(agg, gws, ord_from, ord_to, ["prot_n", "prot_rep"])
    return {
        "prot3": _lagged_roll_sum(mats["prot_n"], lag, 3),
        "prot_rep3": _lagged_roll_sum(mats["prot_rep"], lag, 3),
        "prot_rep12": _lagged_roll_sum(mats["prot_rep"], lag, 12),
    }, first_seen


def _reign_impulses(reign: pd.DataFrame, gws: list[int], ord_from: int, ord_to: int, lag: int,
                    last_month: pd.Period) -> tuple[dict[str, np.ndarray], dict[int, int]]:
    """Подготовить месячные шоки власти REIGN (перевороты / попытки и смены лидера) `≤ last_month`.
    Второе значение — первый месяц (ординал), когда страна появилась в архиве (для `_coverage`)."""
    if len(reign) == 0:
        z = np.zeros((len(gws), ord_to - ord_from + 1), dtype=float)
        return {"coup12": z}, {}
    r = reign.dropna(subset=["gw", "year", "month"]).copy()
    r["o"] = month_ordinal(r.year.astype(int), r.month.astype(int))
    r = r[r.o <= last_month.ordinal].sort_values(["gw", "o"]).copy()
    r = r.groupby(["gw", "o"], as_index=False).agg(leader=("leader", "last"), pt_attempt=("pt_attempt", "max"))
    prev_leader = r.groupby("gw").leader.shift(1)
    lchange = (prev_leader.notna() & (r.leader != prev_leader)).astype(float)
    r["shock"] = ((pd.to_numeric(r.pt_attempt, errors="coerce").fillna(0) > 0) | (lchange > 0)).astype(float)
    first_seen = {int(g): int(o) for g, o in r.groupby("gw").o.min().items()}
    mats, _ = _country_month_matrix(r[["gw", "o", "shock"]], gws, ord_from, ord_to, ["shock"])
    return {"coup12": _lagged_roll_sum(mats["shock"], lag, 12)}, first_seen


def _coverage(first_seen: dict[int, int], gws: list[int], ord_from: int, ord_to: int, lag: int) -> np.ndarray:
    """Покрыта ли страна архивом на строке месяца t: 1, если её первое наблюдение в архиве —
    не позже t − `lag`. Только прошлое: до первого появления страны о её покрытии знать нельзя
    (раньше покрытие считалось по архиву целиком — страна с первым протестом в 2015 году была
    «покрыта» уже в 1995-м)."""
    first = np.array([first_seen.get(g, np.inf) for g in gws], dtype=float)
    t = np.arange(ord_from, ord_to + 1, dtype=float) - lag
    return (first[:, None] <= t[None, :]).astype(float)


def _tek_matrices(epr_groups: pd.DataFrame, tek: pd.DataFrame, gws: list[int], years: list[int], epr_last_year: int) -> dict[int, np.ndarray]:
    """Для каждого года `y` (с продлением после `epr_last_year`) — матрица `w_tek(A ← B)`."""
    n_gw = len(gws)
    gw_idx = {int(g): i for i, g in enumerate(gws)}
    g = extend_groups(epr_groups, epr_last_year, max(years))
    g = g[g["rank"].notna() & g.year.isin(years)][["gw", "year", "gwgroupid", "size"]].copy()
    k = tek.merge(g, on=["gw", "gwgroupid"], how="inner")
    k = k[k.gw.isin(gw_idx)]

    out = {}
    for y in years:
        mat = np.zeros((n_gw, n_gw), dtype=float)
        ky = k[k.year == y]
        if len(ky):
            p = ky.merge(ky[ky["size"] >= TEK_MIN_B_SIZE][["tekid", "gw"]], on="tekid", suffixes=("", "_b"))
            p = p[p.gw != p.gw_b]
            if len(p):
                # Дедупликация по (gw, gw_b, gwgroupid), чтобы одна группа A не считалась дважды через разные tekid
                p = p.groupby(["gw", "gw_b", "gwgroupid"], as_index=False)["size"].first()
                s = p.groupby(["gw", "gw_b"], as_index=False)["size"].sum()
                ii = s.gw.map(gw_idx).to_numpy(dtype=int)
                jj = s.gw_b.map(gw_idx).to_numpy(dtype=int)
                mat[ii, jj] = np.clip(s["size"].to_numpy(dtype=float), 0.0, 1.0)
        np.fill_diagonal(mat, 0.0)
        out[y] = mat
    return out


def _cepii_matrix(cepii: pd.DataFrame, gws: list[int]) -> np.ndarray:
    n_gw = len(gws)
    gw_idx = {int(g): i for i, g in enumerate(gws)}
    mat = np.zeros((n_gw, n_gw), dtype=float)
    if len(cepii):
        c = cepii[cepii.a.isin(gw_idx) & cepii.b.isin(gw_idx)].copy()
        w = 0.5 * c.comlang_off.to_numpy(dtype=float) + 0.5 * c.comlang_ethno.to_numpy(dtype=float)
        ii = c.a.map(gw_idx).to_numpy(dtype=int)
        jj = c.b.map(gw_idx).to_numpy(dtype=int)
        mat[ii, jj] = w
    np.fill_diagonal(mat, 0.0)
    return mat


def _neighbor_matrices(neighbors: set[tuple[int, int, int]], gws: list[int], years: list[int]) -> dict[int, np.ndarray]:
    n_gw = len(gws)
    gw_idx = {int(g): i for i, g in enumerate(gws)}
    by_y: dict[int, list[tuple[int, int]]] = {y: [] for y in years}
    for a, b, y in neighbors:
        if y in by_y and a in gw_idx and b in gw_idx and a != b:
            by_y[y].append((gw_idx[a], gw_idx[b]))
    out = {}
    for y, pairs in by_y.items():
        mat = np.zeros((n_gw, n_gw), dtype=float)
        if pairs:
            arr = np.asarray(pairs, dtype=int)
            mat[arr[:, 0], arr[:, 1]] = 1.0
        out[y] = mat
    return out


def _vdem_vectors(vdem: pd.DataFrame, gws: list[int], years: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """На каждый год V-Dem → (polyarchy, anoc, d2_shock, missing_mask) длины N_gw."""
    n_gw = len(gws)
    gw_idx = {int(g): i for i, g in enumerate(gws)}
    vt = vdem_table(vdem) if len(vdem) else pd.DataFrame(columns=["gw", "year", "vd_polyarchy", "vd_anoc", "vd_polyarchy_d2", "vd_polar_d2"])
    vt = vt[vt.gw.isin(gw_idx) & vt.year.isin(years)].copy()
    vt["shock"] = vt.vd_polyarchy_d2.abs().fillna(0.0) + 0.25 * vt.vd_polar_d2.clip(lower=0.0).fillna(0.0)
    out = {}
    for y in years:
        poly = np.zeros(n_gw, dtype=float)
        anoc = np.zeros(n_gw, dtype=float)
        shock = np.zeros(n_gw, dtype=float)
        miss = np.ones(n_gw, dtype=float)
        sub = vt[vt.year == y]
        if len(sub):
            ii = sub.gw.map(gw_idx).to_numpy(dtype=int)
            ok = sub.vd_polyarchy.notna().to_numpy()
            ii_ok = ii[ok]
            poly[ii_ok] = np.clip(sub.vd_polyarchy.to_numpy(dtype=float)[ok], 0.0, 1.0)
            anoc[ii_ok] = np.clip(sub.vd_anoc.to_numpy(dtype=float)[ok], 0.0, 1.0)
            shock[ii_ok] = np.maximum(sub.shock.to_numpy(dtype=float)[ok], 0.0)
            miss[ii_ok] = 0.0
        out[y] = (poly, anoc, shock, miss)
    return out


def build_network_features(
    panel_df: pd.DataFrame,
    events: pd.DataFrame,
    vdem: pd.DataFrame,
    epr_groups: pd.DataFrame,
    tek: pd.DataFrame,
    cepii: pd.DataFrame,
    neighbors: set[tuple[int, int, int]],
    massmob: pd.DataFrame,
    reign: pd.DataFrame,
    lags: NetworkLags = NetworkLags(),
    epr_last_year: int = EPR_LAST_YEAR,
    mm_last_month: pd.Period = MM_LAST_MONTH,
    reign_last_month: pd.Period = REIGN_LAST_MONTH,
) -> pd.DataFrame:
    """Собрать сетевые и политические признаки на сетке `panel_df` (`gw, year, month, ym, o`).
    Чистая функция: строка месяца `m` использует только данные, известные к концу `m`."""
    p = panel_df[["gw", "year", "month", "ym", "o", "deaths_state", "state_war25"]].sort_values(["gw", "o"]).reset_index(drop=True)
    gws = sorted(int(g) for g in p.gw.unique())
    gw_idx = {g: i for i, g in enumerate(gws)}
    ord_from, ord_to = int(p.o.min()), int(p.o.max())
    n_gw, n_t = len(gws), ord_to - ord_from + 1

    # Какие страны живы в каждом месяце решётки
    _, alive = _country_month_matrix(p, gws, ord_from, ord_to, [])

    # Импульсы на плотной решётке (N_gw × N_months)
    ucdp = _ucdp_impulses(p, events, gws, ord_from, ord_to, lags.event_lag)
    mm, mm_first = _massmob_impulses(massmob, gws, ord_from, ord_to, lags.mm_lag, mm_last_month)
    rg, rg_first = _reign_impulses(reign, gws, ord_from, ord_to, lags.reign_lag, reign_last_month)
    mm_cov = _coverage(mm_first, gws, ord_from, ord_to, lags.mm_lag)
    rg_cov = _coverage(rg_first, gws, ord_from, ord_to, lags.reign_lag)

    # Расписание доступных лет по месяцам
    ym_grid = pd.Series(pd.PeriodIndex.from_ordinals(np.arange(ord_from, ord_to + 1), freq="M"))
    y_cow_arr = ym_grid.dt.year.to_numpy(dtype=int)
    y_vd_arr = _known_year(ym_grid, lags.vdem_source).to_numpy(dtype=int)
    y_epr_arr = _known_year(ym_grid, lags.epr_source).to_numpy(dtype=int)

    lang_mat = _cepii_matrix(cepii, gws)
    nb_mats = _neighbor_matrices(neighbors, gws, sorted(set(y_cow_arr)))
    tek_mats = _tek_matrices(epr_groups, tek, gws, sorted(set(y_epr_arr)), epr_last_year)
    vdem_vecs = _vdem_vectors(vdem, gws, sorted(set(y_vd_arr)))

    # Результаты на плотной решётке (N_gw × N_months)
    res = {c: np.zeros((n_gw, n_t), dtype=float) for c in NETWORK_COLS}
    res["own_prot3"] = np.log1p(mm["prot3"])
    res["own_prot_rep12"] = np.log1p(mm["prot_rep12"])
    res["own_coup12"] = np.log1p(rg["coup12"])

    for t in range(n_t):
        ord_t = ord_from + t
        al = alive[:, t].astype(float)
        # Матрицы связей на текущем (y_cow, y_vd, y_epr)
        nb = nb_mats[int(y_cow_arr[t])]
        w_el = np.maximum(tek_mats[int(y_epr_arr[t])], lang_mat)
        # Обнуляем несуществующие в этом месяце страны-источники B
        w_el_alive = w_el * al[None, :]
        w_el_far = w_el_alive * (1.0 - nb)

        poly, anoc, vd_shock, vd_miss = vdem_vecs[int(y_vd_arr[t])]
        w_aut = (1.0 - poly[:, None]) * (1.0 - poly[None, :]) * np.clip(1.0 - 2.0 * np.abs(poly[:, None] - poly[None, :]), 0.0, 1.0)
        w_aut *= (1.0 - vd_miss[:, None]) * (1.0 - vd_miss[None, :])
        w_cl = np.maximum(w_el, nb) * w_aut * al[None, :]
        np.fill_diagonal(w_cl, 0.0)

        # NET_EL
        ds3_t, ds12_t = ucdp["ds3"][:, t], ucdp["ds12"][:, t]
        dos3_t, dos12_t = ucdp["dos3"][:, t], ucdp["dos12"][:, t]
        res["net_el_war12"][:, t] = (w_el_far * np.log1p(ds12_t)[None, :]).max(axis=1)
        res["net_el_onset12"][:, t] = (w_el_alive * ucdp["onset12"][:, t][None, :]).max(axis=1)
        res["net_el_os12"][:, t] = np.log1p(w_el_far @ dos12_t)
        res["net_el_vdem_d2"][:, t] = (w_el_alive * (vd_shock * (1.0 - vd_miss))[None, :]).max(axis=1)

        # NET_CL
        res["net_cl_shock3"][:, t] = (w_cl * np.log1p(ds3_t + dos3_t)[None, :]).max(axis=1)
        shock12 = np.log1p(w_cl @ (ds12_t + dos12_t))
        res["net_cl_shock12"][:, t] = shock12
        res["net_vuln_cl"][:, t] = anoc * (1.0 - vd_miss) * shock12
        res["vd_net_missing"][:, t] = vd_miss

        # POLIT_OWN / POLIT_NET с контролем конца хроник (инвариант 6)
        mm_cov_vec, rg_cov_vec = mm_cov[:, t], rg_cov[:, t]
        mm_active = (ord_t - lags.mm_lag >= MM_FIRST_MONTH.ordinal) and (ord_t - lags.mm_lag <= mm_last_month.ordinal)
        if mm_active:
            res["mm_net_missing"][:, t] = 1.0 - mm_cov_vec
            res["own_prot3"][:, t] *= mm_cov_vec
            res["own_prot_rep12"][:, t] *= mm_cov_vec
            res["net_cl_prot3"][:, t] = np.log1p(w_cl @ mm["prot_rep3"][:, t])
            res["net_cl_prot12"][:, t] = np.log1p(w_cl @ mm["prot_rep12"][:, t])
        else:
            res["mm_net_missing"][:, t] = 1.0
            res["own_prot3"][:, t] = 0.0
            res["own_prot_rep12"][:, t] = 0.0

        rg_active = ord_t - lags.reign_lag <= reign_last_month.ordinal
        if rg_active:
            res["reign_net_missing"][:, t] = 1.0 - rg_cov_vec
            res["own_coup12"][:, t] *= rg_cov_vec
            res["net_cl_coup12"][:, t] = (w_cl * (rg["coup12"][:, t] > 0).astype(float)[None, :]).max(axis=1)
        else:
            res["reign_net_missing"][:, t] = 1.0
            res["own_coup12"][:, t] = 0.0

    # Выгрузка обратно на строки `panel_df`
    ii = p.gw.map(gw_idx).to_numpy(dtype=int)
    tt = (p.o - ord_from).to_numpy(dtype=int)
    out = p[["gw", "year", "month", "ym", "o"]].copy()
    for c in NETWORK_COLS:
        vals = res[c][ii, tt]
        out[c] = vals.astype("int8") if c.endswith("_missing") else vals
    return out


def annual_slice(net_month: pd.DataFrame) -> pd.DataFrame:
    """Годовой срез сетевых признаков: строка декабря T (события по ноябрь T при lag = 1)."""
    d = net_month[net_month.month == 12]
    return d[["gw", "year", *NETWORK_COLS]].reset_index(drop=True)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Собрать месячную таблицу сетевых признаков и годовую панель для `model_hist`."""
    p = pd.read_parquet(DATA_PROCESSED / "war_month_panel.parquet")
    s = load_inputs(CANDIDATE_LAST_MONTH)
    net = build_network_features(
        p, s["events"], load_vdem(), load_epr_group_years(), load_tek(), load_cepii(),
        s["neighbors"], load_massmob(), load_reign(),
    )
    pitf = pd.read_parquet(DATA_PROCESSED / "features_pitf_country_year.parquet")
    vdem = pd.read_parquet(DATA_PROCESSED / "features_vdem_country_year.parquet")
    ann = annual_slice(net)
    f = pitf.merge(ann, on=["gw", "year"], how="left")
    # Годы до начала месячной панели (1989): заполняем флагами пропуска и нулями
    for flag in ("vd_net_missing", "mm_net_missing", "reign_net_missing"):
        f[flag] = f[flag].fillna(1).astype("int8")
    num_cols = [c for c in NETWORK_COLS if not c.endswith("_missing")]
    f[num_cols] = f[num_cols].fillna(0.0)
    # Добавим V-Dem колонки (без дублирования PITF) для набора `vdem_net_cl`
    v_cols = [c for c in vdem.columns if c not in pitf.columns]
    f = f.merge(vdem[["gw", "year", *v_cols]], on=["gw", "year"], how="left")
    for c in v_cols:
        f[c] = f[c].fillna(0.0)
    return net, f


def main(argv: list[str]) -> int:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    net, f = build()
    net.to_parquet(DATA_PROCESSED / "war_network_month.parquet", index=False)
    f.to_parquet(DATA_PROCESSED / "features_network_country_year.parquet", index=False)
    fin = net[net.year.between(1996, YEAR_TO)]
    print(f"сетевые признаки войн: {len(net)} страна-месяцев, {net.gw.nunique()} стран, "
          f"{net.ym.min()} … {net.ym.max()}")
    print(f"  покрытие 1996–{YEAR_TO}: V-Dem {1 - fin.vd_net_missing.mean():.1%}, "
          f"MassMob {1 - fin.mm_net_missing.mean():.1%}, REIGN {1 - fin.reign_net_missing.mean():.1%}")
    print(net[net.ym == CANDIDATE_LAST_MONTH][["net_el_war12", "net_cl_shock12", "mm_net_missing", "reign_net_missing"]].describe().T[["mean", "max"]].round(3).to_string())
    return 0
