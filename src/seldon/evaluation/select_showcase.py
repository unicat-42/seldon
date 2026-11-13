"""Отбор кандидатов в боевую витрину (чекпоинт 4.3c): сравнение `res_both` vs `res_swap`,
траектории России-2014 и Турции-2018, годовой `long_live_fx`, единая сводная таблица
по всем 4 контурам и проверка живого среза на текущем крае (2026-10 / 2026).

Запуск:
    uv run python -m seldon select_showcase

Боевую витрину `reports/risk_2025_26.csv`, `reports/scorecard.csv` и замороженные снимки
`forecasts/` НЕ изменяет — пишет только диагностические таблицы:
    reports/fx_showcase_regime_slices.csv
    reports/fx_crawling_cases.csv
    reports/fx_annual_live_comparison.csv
    reports/stage4_showcase_candidates.csv
    reports/stage4_showcase_today_fx.csv
    reports/stage4_showcase_today_war.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, FLOAT_FORMAT, REPORTS
from seldon.core.entities import GW_TO_ISO3, ISO3_TO_SHORT, POST_SOVIET_ISO3
from seldon.core.metrics import paired_comparison
from seldon.core.panel import full_grid, pct_change
from seldon.core.temporal import label_known, monthly_lag
from seldon.domains.conflict import model_hist
from seldon.domains.conflict.features import FEATURES as HIST_FEATURES
from seldon.domains.conflict.features_pitf import PITF_FEATURES
from seldon.domains.conflict.network_features import NET_CL
from seldon.domains.currency import econ_contagion, econ_csd, econ_kr, econ_long, econ_struct, struct_features
from seldon.domains.currency.econ_long import EVAL_FROM as LONG_EVAL_FROM
from seldon.domains.currency.econ_struct import KR_C_REF, KR_TARGET, _kr_design, _kr_fit, choose_C_kr
from seldon.domains.currency.long_features import BASE as LONG_BASE
from seldon.domains.currency.long_features import MISSING as LONG_MISSING
from seldon.evaluation.backtest import backtest
from seldon.publish.risk_table import FX_ANNUAL_TRAIN_THROUGH, FX_LAST_MONTH, FX_MIN_AS_OF, GW_ISO, REER_MAX_STALE, RES_MAX_STALE
from seldon.sources.imf import read_imf

N_BOOT = 500
FX_REGIME_SLICES = {
    "мир": lambda d: d.iso3.notna(),
    "постсоветские": lambda d: d.post_soviet.astype(bool),
    "привязка (peg ≥ 0.8)": lambda d: d.peg_share12 >= 0.8,
    "ползущая (0.3 ≤ peg < 0.8)": lambda d: (d.peg_share12 >= 0.3) & (d.peg_share12 < 0.8),
    "плавание (peg < 0.3)": lambda d: d.peg_share12 < 0.3,
}
FX_SLICE_MODELS = ["ref", "base", "struct", "res_both", "res_swap", "all_brk", "all_brk_swap", "live_fx", "live_fx_swap", "live_fx_ctot"]
CRAWLING_TRAJ = [
    ("Россия-2014", "RUS", pd.period_range("2014-06", "2014-11", freq="M")),
    ("Турция-2018", "TUR", pd.period_range("2018-01", "2018-07", freq="M")),
    ("Казахстан-2014 (привязка)", "KAZ", [pd.Period("2013-11", freq="M"), pd.Period("2014-01", freq="M")]),
    ("Казахстан-2015 (отпустили)", "KAZ", [pd.Period("2015-05", freq="M"), pd.Period("2015-07", freq="M")]),
    ("Азербайджан-2015/02", "AZE", [pd.Period("2014-11", freq="M"), pd.Period("2015-01", freq="M")]),
    ("Азербайджан-2015/12", "AZE", [pd.Period("2015-09", freq="M"), pd.Period("2015-11", freq="M")]),
    ("Беларусь-2015/01", "BLR", [pd.Period("2014-10", freq="M"), pd.Period("2014-12", freq="M")]),
]


def _load_fx_preds(h: int) -> pd.DataFrame:
    d = pd.read_parquet(DATA_PROCESSED / f"econ_csd_predictions_h{h}.parquet")
    d["month"] = pd.PeriodIndex(d.month, freq="M")
    d["year"] = d.month.dt.year
    return d


def fx_regime_slices(n_boot: int = N_BOOT) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    preds = {h: _load_fx_preds(h) for h in (3, 6, 12)}
    out = []
    for h, pr in preds.items():
        d0 = pr[pr.evaluable & (pr.year >= econ_kr.FIRST_EVAL_YEAR)].copy()
        avail = [m for m in FX_SLICE_MODELS if f"logit__{m}" in d0.columns]
        cols = {m: f"logit__{m}" for m in avail}
        for sname, mask in FX_REGIME_SLICES.items():
            d = d0[mask(d0)]
            if d.y.nunique() < 2:
                continue
            for ref in ("ref", "live_fx_swap", "all_brk_swap"):
                if ref not in cols:
                    continue
                t = paired_comparison(d, cols, ref, n_boot=n_boot)
                out.append(t.assign(h=h, sample=sname, ref=ref))
    t = pd.concat(out, ignore_index=True)
    lead = ["h", "sample", "ref", "model", "n", "positives"]
    return t[[*lead, *[c for c in t if c not in lead]]], preds


def crawling_cases(preds: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    models = ["ref", "res_both", "res_swap", "live_fx", "live_fx_swap", "all_brk", "all_brk_swap"]
    for h in (3, 12):
        pr = preds[h]
        for label, iso, months in CRAWLING_TRAJ:
            for m in months:
                snap = pr[pr.month == m]
                me = snap[snap.iso3 == iso]
                if me.empty:
                    continue
                r = me.iloc[0]
                row = {
                    "case": label, "iso3": iso, "month": str(m), "h": h,
                    "n_peers": len(snap), "y": r.y,
                    "peg_share12": float(r.peg_share12),
                    "reserves_g12": float(r.reserves_g12) if pd.notna(r.reserves_g12) else np.nan,
                    "peg_resfall": float(r.peg_resfall),
                    "deval12": float(r.deval12),
                }
                for mod in models:
                    col = f"logit__{mod}"
                    if col in snap.columns and pd.notna(r[col]):
                        v = float(r[col])
                        row[f"p_{mod}"] = v
                        row[f"rank_{mod}"] = int((snap[col] > v).sum()) + 1
                rows.append(row)
    return pd.DataFrame(rows)


# Годовой прогноз с признаков года T делается «на 1 января T+1»: к этой дате вышли резервы МВФ
# по месяц 12 − лаг `imf_reserves` (октябрь) — его и берём на строку года T
ANNUAL_RES_MONTH = 12 - monthly_lag("imf_reserves")


def annual_reserves_g12(il: pd.DataFrame, res_month: int = ANNUAL_RES_MONTH) -> pd.DataFrame:
    """Месячные резервы (iso3, month, value) → iso3, year, reserves_g12: прирост резервов за
    12 месяцев по месяц `res_month` года. Чистая функция: строка года T зависит только от
    резервов ≤ `res_month` этого года."""
    il = il[il.value > 0].groupby(["iso3", "month"], as_index=False)["value"].first()
    idx = full_grid(il.groupby("iso3").month.agg(["min", "max"]), names=["iso3", "month"], freq="M")
    il = pd.DataFrame(index=idx).reset_index().merge(il, on=["iso3", "month"], how="left")
    il["reserves_g12"] = il.groupby("iso3")["value"].transform(lambda s: pct_change(s, 12)).clip(-1, 3)
    il = il[il.month.dt.month == res_month]
    return il.assign(year=il.month.dt.year)[["iso3", "year", "reserves_g12"]].reset_index(drop=True)


def _build_annual_live_panel() -> pd.DataFrame:
    st = struct_features.load_struct(last_month="2025-12")
    out = econ_struct.build_long_panel(st)

    il = read_imf("reserves")
    il = il[(il.INDICATOR == "RXF11_REVS") & (il.UNIT == "USD") & (il.month <= FX_LAST_MONTH)]
    out = out.merge(annual_reserves_g12(il[["iso3", "month", "value"]]), on=["iso3", "year"], how="left")
    out["peg_resfall_missing"] = out.reserves_g12.isna().astype(int)
    out["peg_resfall"] = out.peg_share12 * (-out.reserves_g12).clip(lower=0.0).fillna(0.0)
    return out


def annual_fx_live(n_boot: int = N_BOOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = _build_annual_live_panel()
    live_cont = [*LONG_BASE, "peg_share12", "reer_gap", "reer_peg", "peg_resfall"]
    live_flags = [*LONG_MISSING, "reer_missing", "peg_resfall_missing"]

    def step(T: int) -> pd.DataFrame | None:
        r_ref = econ_long.forecast_asof(p, T, n_estimators=10, C=1.0)
        ev = r_ref[r_ref.evaluable].copy()
        if ev.empty:
            return None
        ev = ev.rename(columns={"logit": "logit__ref"})
        ev["logit__base"] = econ_long.forecast_asof(p, T, n_estimators=10, C=None).loc[ev.index, "logit"]
        ev["logit__long_live_fx"] = econ_long.forecast_asof(p, T, n_estimators=10, cont=live_cont, flags=live_flags, C=None).loc[ev.index, "logit"]
        ev["logit__long_live_fx_c1"] = econ_long.forecast_asof(p, T, n_estimators=10, cont=live_cont, flags=live_flags, C=1.0).loc[ev.index, "logit"]
        return ev[["iso3", "year", "post_soviet", "peg_share12", "y", "logit__ref", "logit__base", "logit__long_live_fx", "logit__long_live_fx_c1"]]

    last_eval = int(p[p.evaluable].year.max())
    pred = backtest(range(LONG_EVAL_FROM, last_eval + 1), step)
    spath = DATA_PROCESSED / "econ_long_struct_predictions.parquet"
    if spath.exists():
        sp = pd.read_parquet(spath, columns=["iso3", "year", "logit__struct"])
        pred = pred.merge(sp, on=["iso3", "year"], how="left")

    avail = [c.removeprefix("logit__") for c in pred.columns if c.startswith("logit__")]
    cols = {m: f"logit__{m}" for m in avail}
    rows = []
    for sname, mask in (
        ("мир", pred.iso3.notna()),
        ("постсоветские", pred.post_soviet.astype(bool)),
        ("привязки (peg ≥ 0.8)", pred.peg_share12 >= 0.8),
    ):
        d = pred[mask & pred[list(cols.values())].notna().all(axis=1)]
        if d.y.nunique() < 2:
            continue
        for ref in ("ref", "base"):
            t = paired_comparison(d, cols, ref, n_boot=n_boot)
            rows.append(t.assign(sample=sname, ref=ref))
    return pd.concat(rows, ignore_index=True), p


def _candidate_scorecard() -> pd.DataFrame:
    rows = []

    def add(contour: str, scope: str, model: str, row: pd.Series, live_only: str, note: str):
        rows.append({
            "contour": contour, "scope": scope, "model": model,
            "n": int(row["n"]), "positives": int(row["positives"]),
            "auc": row["auc"], "auc_lo": row["auc_lo"], "auc_hi": row["auc_hi"],
            "d_auc": row.get("d_auc", np.nan), "d_auc_lo": row.get("d_auc_lo", np.nan), "d_auc_hi": row.get("d_auc_hi", np.nan),
            "auc_within_year": row["auc_within_year"], "brier": row["brier"], "calib": row["calib"],
            "live_only": live_only, "notes": note,
        })

    # 1. Годовая война
    for path, contour_picks in [
        (REPORTS / "war_network_annual_comparison.csv", [
            ("мир (1996–2025)", "pitf", "да", "боевой эталон"),
            ("мир (1996–2025)", "vdem", "да", "V-Dem вместо Polity5"),
            ("мир (1996–2025)", "pitf_net_cl", "да", "сеть соседей/языка/режима"),
            ("мир (1996–2025)", "vdem_net_cl", "да", "V-Dem + сеть"),
            ("мир (1996–2025)", "pitf_polit_net", "нет (MassMob 2020-03, REIGN 2021-08)", "протесты и перевороты"),
        ]),
        (REPORTS / "war_live_comparison.csv", [
            ("мир", "dy_live", "нет (EPR/ACD доработка)", "живые диады 4.2"),
            ("мир", "pitf_live", "да", "годовой срез дыма 4.2"),
        ]),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            for sname, m, live, note in contour_picks:
                sub = df[(df["sample"] == sname) & (df["model"] == m)]
                if len(sub):
                    r = sub.iloc[0].rename({"n_onset": "positives"})
                    add("Война, год (state_war25)", "мир", m, r, live, note)

    # 2. Месячная война
    wm_path = REPORTS / "war_month_comparison.csv"
    wn_path = REPORTS / "war_network_month_comparison.csv"
    wc_path = REPORTS / "war_csd_comparison.csv"
    for h in (12, 3):
        if wm_path.exists():
            df = pd.read_csv(wm_path)
            for m, note in [("pitf", "месячный PITF"), ("smoke_nb", "дым + соседи 4.2")]:
                sub = df[(df.h == h) & (df["sample"] == "мир") & (df.ref == "pitf") & (df.model == m)]
                if len(sub):
                    add(f"Война, месяц h={h}", "мир", m, sub.iloc[0], "да", note)
        if wn_path.exists():
            df = pd.read_csv(wn_path)
            for m, live, note in [
                ("net_cl", "да", "сетевой кластер 4.3a"),
                ("smoke_nb_net", "да", "дым + сетевой кластер 4.3a"),
                ("polit_net", "нет (MassMob/REIGN)", "протесты и перевороты"),
            ]:
                sub = df[(df.h == h) & (df["sample"] == "мир (1996–2025)") & (df.ref == "pitf") & (df.model == m)]
                if len(sub):
                    add(f"Война, месяц h={h}", "мир", m, sub.iloc[0], live, note)
        if wc_path.exists():
            df = pd.read_csv(wc_path)
            sub = df[(df.h == h) & (df["sample"] == "мир") & (df.ref == "pitf") & (df.model == "pitf_csd")]
            if len(sub):
                add(f"Война, месяц h={h}", "мир", "pitf_csd", sub.iloc[0], "да", "CSD поверх pitf 4.3b")

    # 3. Годовой FX
    fxa = pd.read_csv(REPORTS / "fx_annual_live_comparison.csv")
    for scope in ("мир", "постсоветские"):
        for m, live, note in [
            ("ref", "да", "боевой logit_long"),
            ("struct", "нет (IRR 2021)", "структурный набор 3.3"),
            ("long_live_fx", "да", "100 % живой: +peg, reer_peg, peg_resfall"),
        ]:
            sub = fxa[(fxa["sample"] == scope) & (fxa.ref == "ref") & (fxa.model == m)]
            if len(sub):
                add("Валюта, год (deval25)", scope, m, sub.iloc[0], live, note)

    # 4. Месячный FX
    fxm = pd.read_csv(REPORTS / "fx_csd_comparison.csv")
    for h in (12, 3):
        for scope in ("мир", "постсоветские"):
            for m, live, note in [
                ("ref", "да", "боевой econ_kr"),
                ("struct", "нет (IRR, gg_missing)", "структурный 3.3"),
                ("all_brk", "нет (IRR, gg_missing, IMTS)", "полный 4.1 с линейным резервом"),
                ("all_brk_swap", "нет (IRR, gg_missing, IMTS)", "полный 4.1 swap"),
                ("live_fx", "да", "живой компактный + линейный резерв"),
                ("live_fx_swap", "да", "живой компактный без линейного резерва"),
                ("live_fx_ctot", "да (лаги ctot 6 мес)", "живой + условия торговли"),
                ("live_fx_csd", "да", "живой + CSD"),
            ]:
                sub = fxm[(fxm.h == h) & (fxm["sample"] == scope) & (fxm.ref == "ref") & (fxm.model == m)]
                if len(sub):
                    add(f"Валюта, месяц h={h}", scope, m, sub.iloc[0], live, note)

    return pd.DataFrame(rows)


def _apply_stale_fill(p: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """REER и резервы на строках `test` — последним известным значением (`econ_csd.stale_fill`,
    те же допуски, что в `risk_table.py`)."""
    st_raw = struct_features.load_struct(last_month=FX_LAST_MONTH, filled=False)
    return econ_csd.stale_fill(p, test, st_raw, REER_MAX_STALE, RES_MAX_STALE)


def live_today_check(annual_live_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Проверка кандидатов на живом срезе `2026-10` (FX) и `2026` (война), НЕ меняя витрину."""
    as_of = pd.Period(FX_LAST_MONTH, freq="M")
    p = econ_csd.load_full_panel(last_month=FX_LAST_MONTH)
    last = p[p.fx.notna()].groupby("iso3").month.max()
    test = p[p.iso3.isin(GW_ISO) & p.own_currency].merge(last.rename("as_of"), on="iso3")
    test = test[(test.month == test.as_of) & (test.as_of >= FX_MIN_AS_OF)].sort_values("iso3").reset_index(drop=True)
    test_stale = _apply_stale_fill(p, test)

    # Обязательная сверка `ref` с боевым `econ_kr` и `reports/risk_2025_26.csv`
    p_kr = econ_kr.build_panel(econ_kr.load_panel(FX_LAST_MONTH), econ_contagion.load_annual_cpi(), econ_contagion.load_war(), target=KR_TARGET)
    m_kr = econ_kr.fit(p_kr, as_of)
    battle = econ_kr.predict(m_kr, test)
    known = p[label_known(p.month, as_of, 12) & p.evaluable]

    def fit_pred(feats: list[str], C_val: float | None, eval_df: pd.DataFrame) -> np.ndarray:
        c = KR_C_REF if C_val is not None else choose_C_kr(known, feats, h=12)
        mod, lo, hi, med = _kr_fit(known, feats, c)
        return mod.predict_proba(_kr_design(eval_df, feats, lo, hi, med))[:, 1]

    ref_prob = fit_pred(econ_csd.features("ref"), KR_C_REF, test)
    diff_mem = float(np.max(np.abs(ref_prob - battle.logit.to_numpy())))
    assert np.allclose(ref_prob, battle.logit.to_numpy(), atol=1e-12), f"ref ≠ econ_kr: {diff_mem}"

    risk = pd.read_csv(REPORTS / "risk_2025_26.csv")
    r_chk = test[["iso3"]].assign(ref=ref_prob).merge(risk[["iso3", "fx_deval15_12m"]], on="iso3")
    diff_csv = float(np.max(np.abs(r_chk.ref - r_chk.fx_deval15_12m)))
    if diff_csv > 1e-4:  # витрина заморожена до лага публикации резервов (`imf_reserves`) — пока не пересобрана, расходится
        print(f"ВНИМАНИЕ: ref ≠ reports/risk_2025_26.csv (max|Δ| = {diff_csv:.2e}): витрина посчитана другой версией признаков")
    print(f"Сверка эталона ref на {FX_LAST_MONTH}: max|Δ| в памяти = {diff_mem:.2e}, к risk_2025_26.csv = {diff_csv:.2e}")

    fx_out = pd.DataFrame({
        "iso3": test.iso3,
        "country": test.iso3.map(ISO3_TO_SHORT).fillna(test.iso3),
        "post_soviet": test.iso3.isin(POST_SOVIET_ISO3),
        "as_of": test.as_of.astype(str),
        "p_ref": ref_prob,
        "p_live_fx": fit_pred(econ_csd.features("live_fx"), None, test),
        "p_live_fx_swap": fit_pred(econ_csd.features("live_fx_swap"), None, test),
        "p_live_fx_stale": fit_pred(econ_csd.features("live_fx"), None, test_stale),
        "p_live_fx_swap_stale": fit_pred(econ_csd.features("live_fx_swap"), None, test_stale),
        "peg_share12": test.peg_share12.to_numpy(),
        "reer_missing_raw": test.reer_missing.to_numpy(),
        "reer_missing_stale": test_stale.reer_missing.to_numpy(),
        "res_missing_raw": test.peg_resfall_missing.to_numpy(),
        "res_missing_stale": test_stale.peg_resfall_missing.to_numpy(),
    })

    # Годовой FX на 2025 -> 2026-27
    al = annual_live_panel[annual_live_panel.iso3.isin(GW_ISO)]
    a_ref = econ_long.forecast_asof(al, 2025, train_through=FX_ANNUAL_TRAIN_THROUGH, n_estimators=10, C=1.0).set_index("iso3").logit
    live_cont = [*LONG_BASE, "peg_share12", "reer_gap", "reer_peg", "peg_resfall"]
    live_flags = [*LONG_MISSING, "reer_missing", "peg_resfall_missing"]
    a_live = econ_long.forecast_asof(al, 2025, train_through=FX_ANNUAL_TRAIN_THROUGH, n_estimators=10, cont=live_cont, flags=live_flags, C=None).set_index("iso3").logit
    fx_out["p_long_ref_2026_27"] = fx_out.iso3.map(a_ref)
    fx_out["p_long_live_fx_2026_27"] = fx_out.iso3.map(a_live)
    for c in ("p_ref", "p_live_fx", "p_live_fx_swap", "p_live_fx_stale", "p_live_fx_swap_stale", "p_long_ref_2026_27", "p_long_live_fx_2026_27"):
        fx_out[f"rank_{c[2:]}"] = fx_out[c].rank(ascending=False, method="min")

    # Война 2025 -> 2026
    fn = pd.read_parquet(DATA_PROCESSED / "features_network_country_year.parquet")
    last_w = int(fn.year.max())
    w_row = fn[fn.year == last_w][["gw", "country", "state_war25"]].copy()
    w_pitf = model_hist.forecast_asof(fn, last_w, [*HIST_FEATURES, *PITF_FEATURES], "state_war25").set_index("gw").p
    w_net = model_hist.forecast_asof(fn, last_w, [*HIST_FEATURES, *PITF_FEATURES, *NET_CL], "state_war25").set_index("gw").p
    w_row["iso3"] = w_row.gw.map(GW_TO_ISO3)
    w_row = w_row[w_row.iso3.notna()].copy()
    w_row["post_soviet"] = w_row.iso3.isin(POST_SOVIET_ISO3)
    w_row["p_pitf"] = w_row.gw.map(w_pitf)
    w_row["p_pitf_net_cl"] = w_row.gw.map(w_net)
    at_risk = w_row.state_war25 == 0
    for c in ("p_pitf", "p_pitf_net_cl"):
        w_row[f"rank_{c[2:]}_at_risk"] = np.nan
        w_row.loc[at_risk, f"rank_{c[2:]}_at_risk"] = w_row.loc[at_risk, c].rank(ascending=False, method="min")
    return fx_out.sort_values("rank_live_fx_stale").reset_index(drop=True), w_row[at_risk].sort_values("rank_pitf_net_cl_at_risk").reset_index(drop=True)


def main(argv: list[str]) -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)

    print("1/5 Срезы режимов курса (`res_both` vs `res_swap`)…")
    sl, preds = fx_regime_slices()
    sl.to_csv(REPORTS / "fx_showcase_regime_slices.csv", index=False, float_format=FLOAT_FORMAT)
    sub = sl[(sl.h.isin([3, 12])) & (sl.ref == "ref") & (sl.model.isin(["ref", "live_fx", "live_fx_swap", "all_brk", "all_brk_swap"]))]
    for (h, sname), g in sub.groupby(["h", "sample"], sort=False):
        print(f"  h={h} | {sname} (n={int(g.n.iloc[0])}, pos={int(g.positives.iloc[0])}):")
        for r in g.itertuples():
            d = "" if pd.isna(r.d_auc) else f" | ΔAUC {r.d_auc:+.3f} [{r.d_auc_lo:+.3f}; {r.d_auc_hi:+.3f}]"
            print(f"    {r.model:14s} AUC {r.auc:.3f} [{r.auc_lo:.3f}; {r.auc_hi:.3f}]  wAUC {r.auc_within_year:.3f}{d}")

    print("\n2/5 Траектории ползущих валют (Россия-2014 и Турция-2018) и привязок…")
    cr = crawling_cases(preds)
    cr.to_csv(REPORTS / "fx_crawling_cases.csv", index=False, float_format=FLOAT_FORMAT)
    print(cr[cr.h == 3][["case", "month", "y", "peg_share12", "reserves_g12", "rank_ref", "rank_live_fx", "rank_live_fx_swap", "rank_all_brk", "p_ref", "p_live_fx", "p_live_fx_swap"]].round(3).to_string(index=False))

    print("\n3/5 Годовой FX: компактный живой `long_live_fx`…")
    fxa, al_panel = annual_fx_live()
    fxa.to_csv(REPORTS / "fx_annual_live_comparison.csv", index=False, float_format=FLOAT_FORMAT)
    print(fxa[fxa.ref == "ref"][["sample", "model", "n", "positives", "auc", "auc_lo", "auc_hi", "d_auc", "d_auc_lo", "d_auc_hi", "auc_within_year", "brier", "calib"]].round(3).to_string(index=False))

    print("\n4/5 Единая сводная таблица кандидатов этапа 4…")
    cand = _candidate_scorecard()
    cand.to_csv(REPORTS / "stage4_showcase_candidates.csv", index=False, float_format=FLOAT_FORMAT)
    print(cand[["contour", "scope", "model", "auc", "d_auc", "auc_within_year", "calib", "live_only"]].round(3).to_string(index=False))

    print("\n5/5 Проверка на живом срезе 2026-10 (FX) и 2026 (война)…")
    tfx, twar = live_today_check(al_panel)
    tfx.to_csv(REPORTS / "stage4_showcase_today_fx.csv", index=False, float_format=FLOAT_FORMAT)
    twar.to_csv(REPORTS / "stage4_showcase_today_war.csv", index=False, float_format=FLOAT_FORMAT)
    print(f"Покрытие на крае 2026-10 (n={len(tfx)}): REER raw={1 - tfx.reer_missing_raw.mean():.1%} -> stale={1 - tfx.reer_missing_stale.mean():.1%}; "
          f"резервы raw={1 - tfx.res_missing_raw.mean():.1%} -> stale={1 - tfx.res_missing_stale.mean():.1%}")
    print("Топ-15 мира FX по `live_fx_stale` (2026-10):")
    print(tfx.head(15)[["country", "as_of", "p_ref", "rank_ref", "p_live_fx_stale", "rank_live_fx_stale", "p_live_fx_swap_stale", "rank_live_fx_swap_stale", "peg_share12"]].round(3).to_string(index=False))
    print("Постсоветские страны FX (2026-10):")
    print(tfx[tfx.post_soviet][["country", "as_of", "p_ref", "rank_ref", "p_live_fx_stale", "rank_live_fx_stale", "p_live_fx_swap_stale", "rank_live_fx_swap_stale", "p_long_ref_2026_27", "p_long_live_fx_2026_27"]].round(3).to_string(index=False))
    return 0
