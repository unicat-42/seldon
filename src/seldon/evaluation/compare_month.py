"""Месячная модель войн: срезы, контрольные кейсы, сравнение с conflictforecast (чекпоинт 4.2).

Запуск:
    uv run python -m seldon compare_month      # нужны `model_month`, `model_month cf`, `dyads_live`
    uv run python -m seldon compare_month quick   # то же без пересчёта таблицы срезов

Ничего не обучает — читает прогнозы вслепую `war_month_predictions_h{3,6,12}.parquet`.

1. Срезы. Оценивается НАЧАЛО войны: строки, мирные по факту (12 месяцев без `state_war25`),
   метка — была ли война в месяцы m+1 … m+h; годы 1996–2025. Выборки: мир, тихие ≥ 5 лет,
   тихие ≥ 10 лет, постсоветские, после 2014 года (и «все строки» — вместе с продолжением
   идущих войн). Парное сравнение с эталоном (`core.metrics.paired_comparison`: сквозной AUC,
   AUC внутри года, Brier, `calib`; 95 % интервалы — блочный бутстрэп по годам). Два эталона:
   `pitf` (месячный аналог боевой модели) и `dry` (он же без субпороговых погибших в истории).
   Одно начало войны попадает в окно до h соседних строк одной страны — число самих начал
   (`n_onsets`) приведено отдельно; блоки бутстрэпа — годы, зависимость соседних месяцев
   внутри года учтена.
2. Признаки поодиночке: AUC каждого предвестника на мирных строках.
3. Контрольные кейсы: место страны среди мирных стран мира за 3, 6 и 12 месяцев до начала
   (прогноз из месяца M − k моделью с горизонтом k). Если страна к этому месяцу уже воевала
   по месячному порогу — показана вероятность продолжения и место среди всех стран; для всех
   кейсов — внешний канал годовой диадной модели (`dyads_live`) на год начала.
4. conflictforecast.org (Mueller & Rauh), файлы `latest`, горизонты 3 и 12 месяцев:
   а) на ИХ целевой (`armedconf`: погибших от любого насилия ≥ 0,5 на миллион жителей в
      месяц; строки — месяцы без такого конфликта; начало в m+1 … m+h) — наши модели
      `cf_hist` / `cf_full`, обученные вслепую на этой же целевой, против их `naive`, `hist`,
      `text`, `all` на одних и тех же строках, 2010–2025;
   б) на НАШЕЙ целевой (`state_war25`, мирные строки) — их прогнозы как ранжирование против
      наших; сравнивать здесь можно только AUC (их вероятности — про другое событие).
   Наши прогнозы — в двух вариантах: с честным лагом событий в 1 месяц и без лага (`_l0`),
   как у них (признаки строки включают её собственный месяц).
5. Прогноз из последнего месяца с данными Candidate Events.

Выход: reports/war_month_comparison.csv, war_month_single_auc.csv, war_month_cases.csv,
war_month_cf_comparison.csv, war_month_top_<месяц>.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, DATA_RAW, REPORTS
from seldon.core.entities import GW_TO_SHORT, ISO3_TO_GW, POST_SOVIET_GW
from seldon.core.metrics import auc, paired_comparison
from seldon.domains.conflict.model_month import CF_HORIZONS, HORIZONS
from seldon.domains.conflict.month_features import LIVE, SMOKE, SMOKE_NB
from seldon.sources.ucdp import CANDIDATE_LAST_MONTH, YEAR_TO

N_BOOT = 500
EVAL_FROM = 1996
NAMES = ["pitf", "base", "hist", "dry", "smoke", "smoke_nb", "live", "full", "pitf_l0", "full_l0"]
SAMPLES = {
    "мир": lambda d: d.quiet_true == 1,
    "тихие ≥ 5 лет": lambda d: (d.quiet_true == 1) & (d.qmonths >= 60),
    "тихие ≥ 10 лет": lambda d: (d.quiet_true == 1) & (d.qmonths >= 120),
    "постсоветские": lambda d: (d.quiet_true == 1) & d.gw.isin(POST_SOVIET_GW),
    "после 2014": lambda d: (d.quiet_true == 1) & (d.year >= 2015),
    "после 2014, тихие ≥ 5 лет": lambda d: (d.quiet_true == 1) & (d.year >= 2015) & (d.qmonths >= 60),
    "все строки (с продолжением)": lambda d: d.year > 0,
}
# (подпись, код GW, месяц начала)
CASES = [("Грузия", 372, "2008-08"), ("Украина", 369, "2014-02"), ("Украина", 369, "2022-02"), ("Армения", 371, "2020-09"),
         ("Азербайджан", 373, "2020-09"), ("Кыргызстан", 703, "2021-04"), ("Таджикистан", 702, "2021-04"),
         ("Кыргызстан", 703, "2022-09"), ("Таджикистан", 702, "2022-09"), ("Эфиопия", 530, "2020-11"), ("Судан", 625, "2023-04"),
         ("Израиль", 666, "2023-10"), ("Ливан", 660, "2023-10")]
CASE_SETS = ["base", "pitf", "smoke_nb", "full"]
CF_DIR = DATA_RAW / "conflictforecast" / "latest"
CF_MODELS = ("naive", "hist", "text", "all")


def load(h: int) -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / f"war_month_predictions_h{h}.parquet")


def evaluable(p: pd.DataFrame) -> pd.DataFrame:
    return p[p.y.notna() & (p.year >= EVAL_FROM) & (p.year <= YEAR_TO)]


def n_onsets(d: pd.DataFrame, onsets: pd.DataFrame, h: int) -> int:
    """Сколько разных начал войны попадает в окна строк выборки."""
    x = d[["gw", "o"]].merge(onsets, on="gw")
    return int(x[(x.o_on > x.o) & (x.o_on <= x.o + h)].drop_duplicates(["gw", "o_on"]).shape[0])


def compare(n_boot: int = N_BOOT) -> pd.DataFrame:
    out = []
    for h in HORIZONS:
        p = load(h)
        onsets = p.loc[(p.state_war25 == 1), ["gw", "o"]]  # месяцы войны; начало — первый в окне мирной строки
        onsets = onsets.rename(columns={"o": "o_on"})
        e = evaluable(p)
        cols = {n: f"p__{n}" for n in NAMES}
        for sample, mask in SAMPLES.items():
            d = e[mask(e)]
            first = d[["gw", "o"]].merge(onsets, on="gw")
            first = first[(first.o_on > first.o) & (first.o_on <= first.o + h)].groupby(["gw", "o"]).o_on.min().reset_index()
            k = first.drop_duplicates(["gw", "o_on"]).shape[0] if "все" not in sample else np.nan
            for ref in ("pitf", "dry"):
                t = paired_comparison(d, cols, ref, n_boot=n_boot)
                out.append(t.assign(h=h, sample=sample, ref=ref, n_onsets=k))
    t = pd.concat(out, ignore_index=True)
    return t[["h", "sample", "ref", "model", "n", "positives", "n_onsets", *[c for c in t if c not in ("h", "sample", "ref", "model", "n", "positives", "n_onsets")]]]


def single_auc() -> pd.DataFrame:
    """AUC каждого предвестника в одиночку на мирных строках (признаки — из таблицы lag = 1)."""
    f = pd.read_parquet(DATA_PROCESSED / "war_month_features.parquet", columns=["gw", "o", "ld_war_12", *SMOKE, *SMOKE_NB, *LIVE])
    rows = []
    for h in HORIZONS:
        e = evaluable(load(h))
        e = e[e.quiet_true == 1][["gw", "o", "y", "qmonths", "year"]].merge(f, on=["gw", "o"])
        for c in ["ld_war_12", *SMOKE, *SMOKE_NB, *LIVE]:
            rows.append({"h": h, "feature": c, "share_nonzero": float((e[c] != 0).mean()), "auc_world": auc(e.y, e[c]),
                         "auc_quiet5": auc(e[e.qmonths >= 60].y, e[e.qmonths >= 60][c]),
                         "auc_after2014": auc(e[e.year >= 2015].y, e[e.year >= 2015][c])})
    return pd.DataFrame(rows)


def cases() -> pd.DataFrame:
    preds = {h: load(h) for h in HORIZONS}
    ext = pd.read_parquet(DATA_PROCESSED / "dyad_live_country.parquet")
    rows = []
    for label, gw, month in CASES:
        M = pd.Period(month, freq="M")
        now = preds[3][(preds[3].gw == gw) & (preds[3].ym == M)]
        prev12 = preds[3][(preds[3].gw == gw) & (preds[3].ym < M) & (preds[3].ym >= M - 12)]
        is_onset = bool(len(now) and now.state_war25.iloc[0] == 1 and prev12.state_war25.sum() == 0)
        x = ext[(ext.gw == gw) & (ext.year == M.year)]
        peers = ext[ext.year == M.year]
        ext_p = float(x["p_ext__live_split"].iloc[0]) if len(x) else np.nan
        ext_rank = int((peers["p_ext__live_split"] > ext_p).sum()) + 1 if len(x) else np.nan
        for k in HORIZONS:
            p = preds[k]
            origin = M - k
            r = p[(p.gw == gw) & (p.ym == origin)]
            row = {"case": label, "onset_month": month, "onset_by_rule": is_onset, "months_before": k, "origin": str(origin),
                   "p_ext_year": ext_p, "p_ext_rank": ext_rank}
            if len(r) == 0:
                rows.append(row | {"status": "нет строки"})
                continue
            r = r.iloc[0]
            quiet = r.quiet_true == 1
            peers_m = p[(p.ym == origin) & ((p.quiet_true == 1) if quiet else True)]
            row |= {"status": "мирная" if quiet else f"уже воюет (месяцев с войны: {int(r.qmonths)})", "n_peers": len(peers_m), "war_in_window": r.y}
            for s in CASE_SETS:
                row[f"rank_{s}"], row[f"p_{s}"] = int((peers_m[f"p__{s}"] > r[f"p__{s}"]).sum()) + 1, r[f"p__{s}"]
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# conflictforecast
# ---------------------------------------------------------------------------


def load_cf(kind: str, h: int) -> pd.DataFrame:
    """Их файл для целевой `kind` (`armedconf` / `anyviolence`) и горизонта h: gw, o, target,
    их четыре прогноза, `since` (месяцев с последнего конфликта по их счёту)."""
    stem = f"ons_{kind}_{h:02d}"
    c = pd.read_csv(CF_DIR / f"conflictforecast_{stem}.csv",
                    usecols=["isocode", "period", f"{stem}_target", *[f"{stem}_{m}" for m in CF_MODELS], kind, f"since_{kind}"])
    c["gw"] = c.isocode.map(ISO3_TO_GW)
    c = c.dropna(subset=["gw"]).astype({"gw": int})
    c["o"] = (c.period // 100 - 1970) * 12 + c.period % 100 - 1
    c = c.rename(columns={f"{stem}_target": f"{kind}_target", f"since_{kind}": f"{kind}_since", **{f"{stem}_{m}": f"cf_{kind}_{m}" for m in CF_MODELS}})
    return c.drop(columns=["isocode", "period"])


def compare_cf(n_boot: int = N_BOOT) -> pd.DataFrame:
    out = []
    for h in CF_HORIZONS:
        cf = load_cf("armedconf", h).merge(load_cf("anyviolence", h)[["gw", "o", "cf_anyviolence_all"]], on=["gw", "o"])
        # --- а) их целевая ---
        ours = pd.read_parquet(DATA_PROCESSED / f"war_month_cf_predictions_h{h}.parquet")
        d = cf[cf.armedconf_target.notna()].merge(ours, on=["gw", "o"], how="inner")
        d = d[(d.year <= YEAR_TO) & d.po__cf_full.notna()]
        agree = float((d.armedconf_target == d.y).mean())
        cols = {**{f"cf_{m}": f"cf_armedconf_{m}" for m in CF_MODELS},
                **{f"ours_{n}": f"po__{n}" for n in ("cf_hist", "cf_full", "cf_hist_l0", "cf_full_l0")}}
        samples = {"мир": d, "тихие ≥ 5 лет (их счёт)": d[d.armedconf_since >= 60], "тихие ≥ 10 лет (их счёт)": d[d.armedconf_since >= 120],
                   "постсоветские": d[d.gw.isin(POST_SOVIET_GW)], "после 2014": d[d.year >= 2015]}
        for sample, x in samples.items():
            t = paired_comparison(x, cols, "cf_all", y="armedconf_target", n_boot=n_boot)
            out.append(t.assign(h=h, target="их armedconf", sample=sample, target_agreement=agree))
        # --- б) наша целевая ---
        e = evaluable(load(h))
        e = e[e.quiet_true == 1].merge(cf, on=["gw", "o"], how="inner")
        cols = {**{f"cf_{m}": f"cf_armedconf_{m}" for m in CF_MODELS}, "cf_anyviolence_all": "cf_anyviolence_all",
                **{f"ours_{n}": f"p__{n}" for n in ("base", "pitf", "smoke_nb", "full", "pitf_l0", "full_l0")}}
        samples = {"мир": e, "тихие ≥ 5 лет": e[e.qmonths >= 60], "тихие ≥ 10 лет": e[e.qmonths >= 120],
                   "постсоветские": e[e.gw.isin(POST_SOVIET_GW)], "после 2014": e[e.year >= 2015]}
        for sample, x in samples.items():
            t = paired_comparison(x, cols, "cf_all", n_boot=n_boot)
            out.append(t.assign(h=h, target="наша state_war25", sample=sample, target_agreement=np.nan))
    t = pd.concat(out, ignore_index=True)
    lead = ["h", "target", "sample", "model", "n", "positives"]
    return t[[*lead, *[c for c in t if c not in lead]]]


def top_now(month: pd.Period = CANDIDATE_LAST_MONTH, n: int = 20) -> pd.DataFrame:
    """Прогноз из последнего месяца с данными: мирные (по известному) страны, набор `smoke_nb`
    и эталон `pitf`, горизонты 3 / 6 / 12; плюс все постсоветские."""
    out = None
    for h in HORIZONS:
        p = load(h)
        x = p[p.ym == month][["gw", "country", "quiet_known", "p__pitf", "p__smoke_nb", "p__full"]]
        x = x.rename(columns={c: f"{c[3:]}_h{h}" for c in x if c.startswith("p__")})
        out = x if out is None else out.merge(x.drop(columns=["country", "quiet_known"]), on="gw")
    q = out[out.quiet_known == 1].sort_values("smoke_nb_h3", ascending=False)
    q = q.assign(rank_h3=np.arange(1, len(q) + 1))
    keep = q[(q.rank_h3 <= n) | q.gw.isin(POST_SOVIET_GW)]
    war = out[(out.quiet_known == 0) & out.gw.isin(POST_SOVIET_GW)].assign(rank_h3=np.nan)
    return pd.concat([keep, war], ignore_index=True)


def _show(t: pd.DataFrame, by: list[str], width: int = 18) -> None:
    for key, g in t.groupby(by, sort=False):
        k = "" if "n_onsets" not in g or pd.isna(g.n_onsets.iloc[0]) else f", разных начал {int(g.n_onsets.iloc[0])}"
        print(f"{key}: строк {int(g.n.iloc[0])}, положительных {int(g.positives.iloc[0])}{k}")
        for r in g.itertuples():
            s = (f"  {r.model:{width}s} AUC {r.auc:.3f} [{r.auc_lo:.3f}; {r.auc_hi:.3f}], внутри года {r.auc_within_year:.3f}, "
                 f"Brier {r.brier:.4f}, calib {r.calib:.2f}")
            if pd.notna(getattr(r, "d_auc", np.nan)):
                s += (f" | ΔAUC {r.d_auc:+.3f} [{r.d_auc_lo:+.3f}; {r.d_auc_hi:+.3f}], внутри года {r.d_auc_within_year:+.3f} "
                      f"[{r.d_auc_within_year_lo:+.3f}; {r.d_auc_within_year_hi:+.3f}]")
            print(s)


def main(argv: list[str]) -> int:
    pd.set_option("display.width", 250)
    if argv[:1] == ["quick"]:  # без пересчёта срезов (бутстрэп срезов — самая долгая часть)
        t = pd.read_csv(REPORTS / "war_month_comparison.csv")
    else:
        t = compare()
        t.to_csv(REPORTS / "war_month_comparison.csv", index=False, float_format="%.4f")
    print(f"Начало `state_war25` в окне m+1 … m+h, мирные строки, {EVAL_FROM}–{YEAR_TO}; Δ — к эталону `pitf`")
    _show(t[t.ref == "pitf"], ["h", "sample"], 9)
    print("\nΔ к «сухому» эталону `dry` (история без субпороговых погибших):")
    d = t[(t.ref == "dry") & t.model.isin(["pitf", "smoke_nb", "full"]) & t["sample"].isin(["мир", "тихие ≥ 5 лет", "после 2014"])]
    for r in d.itertuples():
        print(f"  h {r.h:2d}, {r.sample:14s} {r.model:9s} ΔAUC {r.d_auc:+.3f} [{r.d_auc_lo:+.3f}; {r.d_auc_hi:+.3f}]")

    sa = single_auc()
    sa.to_csv(REPORTS / "war_month_single_auc.csv", index=False, float_format="%.4f")
    print("\nПредвестники поодиночке (AUC на мирных строках, h = 3 и 12):")
    w = sa[sa.h.isin([3, 12])].pivot(index="feature", columns="h", values=["auc_world", "auc_quiet5"])
    w.columns = [f"{a}_h{b}" for a, b in w.columns]
    w["share_nonzero"] = sa[sa.h == 3].set_index("feature").share_nonzero
    print(w.round(3).sort_values("auc_world_h3", ascending=False).to_string())

    c = cases()
    c.to_csv(REPORTS / "war_month_cases.csv", index=False, float_format="%.5f")
    print("\nКонтрольные кейсы (место среди мирных стран мира; у воюющих — среди всех стран):")
    print(c[["case", "onset_month", "onset_by_rule", "months_before", "status", "n_peers", "rank_base", "rank_pitf", "rank_smoke_nb",
             "rank_full", "p_pitf", "p_full", "p_ext_year", "p_ext_rank"]].round(3).to_string(index=False))

    cf = compare_cf()
    cf.to_csv(REPORTS / "war_month_cf_comparison.csv", index=False, float_format="%.4f")
    print("\nconflictforecast.org; Δ — к их модели `all`")
    for target, g in cf.groupby("target", sort=False):
        if g.target_agreement.notna().any():
            print(f"[{target}] совпадение нашей реконструкции их целевой с их колонкой: {g.target_agreement.dropna().iloc[0]:.2%}")
        _show(g, ["target", "h", "sample"], 20)

    top = top_now()
    top.to_csv(REPORTS / f"war_month_top_{CANDIDATE_LAST_MONTH}.csv", index=False, float_format="%.4f")
    print(f"\nПрогноз из {CANDIDATE_LAST_MONTH} (события по {CANDIDATE_LAST_MONTH - 1}); мирные страны, порядок — `smoke_nb`, h = 3:")
    print(top.assign(short=top.gw.map(GW_TO_SHORT)).round(3).to_string(index=False))
    return 0
