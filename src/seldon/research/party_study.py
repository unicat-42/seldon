"""[research] Идеология правящей партии / режима и смена партии у руля: влияют ли они на риск
начала войны, валютного и банковского кризиса (гипотеза Ангелины, 14.11.2026).

Запуск:
    uv run python -m seldon party_study [describe] [war] [dyad] [fx_long] [fx_month] [ablation] [bank] [cases]
        без аргументов — всё по порядку (≈ 25 минут). Нужны готовые таблицы `make all` и
        `make variants` (`dyads`, `econ_struct`, `econ_contagion`) и `download gli`, `download vparty`.

Боевой конвейер, витрина и снимки прогнозов не затрагиваются: модуль только читает их таблицы.

Признаки — `research/party_features.py`: живой слой (V-Dem: идеологическая, персоналистская и
результативная легитимация, характер идеологии, смена лидера / режима / власти на выборах) и
архивный (Global Leader Ideologies 1945–2020, V-Party 1970–2019). Лаг публикации — 1 год: на
строке T стоят данные за T − 1. Архивные базы оцениваются только на своём «чистом окне» — до
последней строки, где база ещё есть; после неё нет ни нулей, ни флагов пропуска.

Шаги:
    describe  разведка: частота событий в окне прогноза по типам власти, со срезами (богатые /
              бедные, демократии / автократии, постсоветские) — БЕЗ моделей;
    war       годовая модель войн: эталон `pitf` против `pitf` + партийные наборы;
    dyad      диадная модель: идеология потенциального противника B и идеологическая дистанция пары;
    fx_long   годовой валютный контур (`deval25`, окно 2 года): логрегрессия и деревья;
    fx_month  месячный (`deval15`, 12 и 3 месяца): боевая `econ_kr` и претендент `live_fx_swap`;
    ablation  разбор прироста живого слоя в месячном контуре: флаг пропуска, тип режима, персонализм;
    bank      банковские кризисы: JST (обучение с 1946 года) и перенос на страны BIS;
    cases     места стран в контрольных кейсах до и после добавления признаков.
Во всех контурах эталон пересчитывается теми же функциями боевого кода (`forecast_asof` своего
контура), а набор с партийными признаками — ими же с расширенным списком колонок. Сравнение —
`core.metrics.paired_comparison`: сквозной AUC, AUC внутри года, Brier, `calib` и разницы с
эталоном с 95 % интервалом (парный блочный бутстрэп по годам, 500 реплик).

Выход:
    data/processed/party_features_country_year.parquet, party_<контур>_predictions*.parquet
    reports/party_coverage.csv            покрытие слоёв по 15 постсоветским странам
    reports/party_post_soviet_coding.csv  как закодированы постсоветские страны в ключевые годы
    reports/party_base_rates.csv          разведка базовых частот
    reports/party_oos_comparison.csv      проверка вслепую, все контуры
    reports/party_coefficients.csv        стандартизованные коэффициенты партийных признаков
    reports/party_fx_month_ablation.csv   разбор прироста в месячном валютном контуре
    reports/party_cases.csv               контрольные кейсы
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, REPORTS
from seldon.core.entities import GW_TO_ISO3, GW_TO_SHORT, ISO3_TO_GW, POST_SOVIET_GW, POST_SOVIET_ISO3
from seldon.core.metrics import interval, paired_comparison, year_block_bootstrap
from seldon.core.temporal import label_known, publication_lag
from seldon.research import party_features as pf

FMT = "%.4f"
N_BOOT = 500
WAR_TARGET = "state_war25"
VDEM_FROM = 1940

# набор → (слои, признаки). Флаги пропуска слоёв добавляются автоматически (`set_features`)
WAR_SETS: dict[str, tuple[tuple[str, ...], list[str]]] = {
    "legit": (("live",), pf.LIVE_LEGIT),
    "turn": (("live",), pf.LIVE_TURN),
    "live": (("live",), [*pf.LIVE_LEGIT, *pf.LIVE_TURN]),
    "gli": (("gli",), pf.ARCH_GLI),
    "vparty": (("vparty",), pf.ARCH_VP),
    "arch": (("gli", "vparty"), [*pf.ARCH_GLI, *pf.ARCH_VP]),
}
# взаимодействия для валютных контуров: «держат курс до срыва» и «смена власти × натянутая пружина»
FX_INTER = {
    "live": ["pl_soc_peg", "pl_lead_peg", "pl_new_spring"],
    "gli": ["gl_left_peg", "gl_left_def", "gl_switch_spring"],
    "vparty": ["vp_popul_peg", "vp_popul_def", "vp_new_spring"],
}
FX_SETS: dict[str, tuple[tuple[str, ...], list[str]]] = {
    "live": (("live",), [*pf.LIVE_LEGIT, *pf.LIVE_TURN, *FX_INTER["live"]]),
    "gli": (("gli",), [*pf.ARCH_GLI, *FX_INTER["gli"]]),
    "vparty": (("vparty",), [*pf.ARCH_VP, *FX_INTER["vparty"]]),
}
BANK_SETS: dict[str, tuple[tuple[str, ...], list[str]]] = {
    "legit": (("live",), pf.LIVE_LEGIT),
    "live": (("live",), [*pf.LIVE_LEGIT, *pf.LIVE_TURN]),
    "gli": (("gli",), [c for c in pf.ARCH_GLI if c != "gl_none"]),  # в 18 странах JST неидеологических лидеров нет
    "vparty": (("vparty",), pf.ARCH_VP),
}
DY_LIVE = ["b_nat", "b_lead", "b_ideol", "b_leader_new", "b_regime_new", "ab_ideol_dist", "ab_char_dist"]
DY_ARCH = ["b_gl_right", "b_gl_left", "ab_camp_diff", "b_popul", "b_antiplural"]
DYAD_SETS: dict[str, tuple[str, list[str]]] = {  # набор → (слой пары, признаки)
    "b_ideol": ("live", ["b_nat", "b_lead", "b_ideol"]),
    "dist": ("live", ["ab_ideol_dist", "ab_char_dist"]),
    "dyad_live": ("live", DY_LIVE),
    "dyad_arch": ("arch", DY_ARCH),
}
DY_FLAG = {"live": "dyl_missing", "arch": "dya_missing"}
FX_CASE_ISO = ["ARG", "VEN", "TUR", "RUS", "UKR", "BLR", "KAZ", "GEO", "AZE"]
WAR_CASES = [("Украина-2014", 369, 2014), ("Грузия-2008", 372, 2008), ("Азербайджан-2020 (Карабах)", 373, 2020),
             ("Армения-2020 (Карабах)", 371, 2020), ("Кыргызстан-2021", 703, 2021), ("Таджикистан-2021", 702, 2021)]


# ---------------------------------------------------------------------------
# Признаки
# ---------------------------------------------------------------------------


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from seldon.sources.gli import load_gli
    from seldon.sources.vdem import PARTY_COLS, load_vdem
    from seldon.sources.vparty import load_ruling

    v = load_vdem(VDEM_FROM, [*PARTY_COLS, "v2exnamhog", "v2ex_hosw", "v2elturnhos", "v2elturnhog"])
    return v, load_gli(), load_ruling()


def load_features(src=None) -> pd.DataFrame:
    """Годовая таблица на строках прогноза (gw, year = T; данные за T − 1), с колонкой iso3."""
    f = pf.build_annual(*(load_sources() if src is None else src))
    f["iso3"] = f.gw.map(GW_TO_ISO3)
    return f


def by_iso3(t: pd.DataFrame) -> pd.DataFrame:
    """Таблица с ключом gw → с ключом iso3 (Югославия и Сербия — один код; берём позднюю)."""
    t = t.assign(iso3=t.gw.map(GW_TO_ISO3)).dropna(subset=["iso3"]).sort_values(["iso3", "year", "gw"])
    return t.drop_duplicates(["iso3", "year"], keep="first").drop(columns="gw")


def attach_annual(p: pd.DataFrame, feats: pd.DataFrame, key: str = "gw") -> pd.DataFrame:
    """Партийные признаки на годовую панель контура по (`key`, year); `*_ok` без пары — False."""
    cols = [*pf.ALL, *pf.OK.values(), *pf.EXTRA]
    t = feats if key == "gw" else by_iso3(feats.drop(columns="iso3", errors="ignore"))
    out = p.merge(t[[key, "year", *cols]], on=[key, "year"], how="left")
    for ok in pf.OK.values():
        out[ok] = out[ok].fillna(False).astype(bool)
    return out


def prepare(p: pd.DataFrame, layers: tuple[str, ...], period: str = "year", lower: bool = False) -> pd.DataFrame:
    """Панель набора: чистое окно всех его слоёв, пропуски внутри окна заполнены (`pf.fill`).
    `lower=True` — отрезать и годы ДО появления слоя (банковский контур: JST с 1870 года)."""
    for layer in layers:
        p = pf.clean_window(p, layer, period)
        if lower:
            p = p[p[period] >= p.loc[p[pf.OK[layer]], period].min()]
        p = pf.fill(p, layer, period)
    return p


def set_features(sets: dict, name: str) -> list[str]:
    layers, cols = sets[name]
    return [*cols, *[pf.LAYERS[layer][1] for layer in layers]]


def add_fx_interactions(p: pd.DataFrame, peg: str, deficit: pd.Series | None, resfall: str | None = None) -> pd.DataFrame:
    """Взаимодействия валютных контуров (после `prepare`). `peg` — мера привязки курса контура
    (годовой: `peg_years_log`, месячный: `peg_share12`); `deficit` — дефицит счёта текущих операций
    (годовой) или `resfall` — привязка × сжигание резервов (месячный, `peg_resfall`)."""
    p = p.copy()
    k = p[peg].fillna(0.0)
    stress = deficit if resfall is None else p[resfall].fillna(0.0)
    spring = p.reer_peg.fillna(0.0)
    if "pl_soc" in p:
        p["pl_soc_peg"], p["pl_lead_peg"] = p.pl_soc * k, p.pl_lead.clip(lower=0) * k
        p["pl_new_spring"] = p.pl_leader_new * spring
    if "gl_left" in p:
        p["gl_left_peg"], p["gl_left_def"], p["gl_switch_spring"] = p.gl_left * k, p.gl_left * stress, p.gl_switch * spring
    if "vp_popul" in p:
        p["vp_popul_peg"], p["vp_popul_def"], p["vp_new_spring"] = p.vp_popul * k, p.vp_popul * stress, p.vp_new * spring
    return p


# ---------------------------------------------------------------------------
# Сравнение с эталоном
# ---------------------------------------------------------------------------


def compare(pred: pd.DataFrame, ref: str, cols: dict[str, str], y: str, samples: dict, year: str = "year",
            n_boot: int = N_BOOT, **tags) -> pd.DataFrame:
    """Парное сравнение каждого набора из `cols` (имя → колонка) с эталоном `ref` на строках,
    где у набора есть прогноз (его чистое окно). Строка на (набор, выборка): метрики набора и
    эталона на тех же строках и Δ с 95 % интервалом."""
    rows = []
    for name, col in cols.items():
        d = pred[pred[col].notna() & pred[ref].notna()]
        for sample, mask in samples.items():
            s = d[mask(d)]
            if len(s) == 0 or s[y].nunique() < 2:
                continue
            t = paired_comparison(s, {"ref": ref, name: col}, ref="ref", y=y, year=year, n_boot=n_boot).set_index("model")
            r, m = t.loc["ref"], t.loc[name]
            row = {**tags, "set": name, "sample": sample, "years": f"{int(s[year].min())}–{int(s[year].max())}", "n": int(m.n),
                   "positives": int(m.positives), "ref_auc": r.auc, "ref_auc_within_year": r.auc_within_year, "ref_brier": r.brier,
                   "ref_calib": r.calib}
            for k in ("auc", "auc_within_year", "brier", "calib"):
                row[k] = m[k]
                row[f"d_{k}"], row[f"d_{k}_lo"], row[f"d_{k}_hi"] = m[f"d_{k}"], m[f"d_{k}_lo"], m[f"d_{k}_hi"]
            row["auc_lo"], row["auc_hi"] = m.auc_lo, m.auc_hi
            rows.append(row)
    return pd.DataFrame(rows)


def show(t: pd.DataFrame) -> str:
    lines = []
    for r in t.itertuples():
        tag = " ".join(str(getattr(r, k)) for k in ("contour", "model", "horizon") if hasattr(r, k) and pd.notna(getattr(r, k)))
        lines.append(f"  {tag:28s} {r.set:11s} {r.sample:16s} {r.years} n {r.n:6d}, событий {r.positives:4d}: AUC {r.ref_auc:.3f} → {r.auc:.3f} "
                     f"Δ {r.d_auc:+.3f} [{r.d_auc_lo:+.3f}; {r.d_auc_hi:+.3f}], внутри года Δ {r.d_auc_within_year:+.3f} "
                     f"[{r.d_auc_within_year_lo:+.3f}; {r.d_auc_within_year_hi:+.3f}], ΔBrier {r.d_brier:+.4f} "
                     f"[{r.d_brier_lo:+.4f}; {r.d_brier_hi:+.4f}], calib {r.ref_calib:.2f} → {r.calib:.2f}")
    return "\n".join(lines)


def _rank(peers: pd.Series, value: float) -> int:
    return int((peers > value).sum()) + 1


def _save_oos(part: pd.DataFrame, contour: str) -> None:
    """Обновить в общей таблице сравнений строки одного контура (остальные не трогаем)."""
    path = REPORTS / "party_oos_comparison.csv"
    part = part.assign(contour=contour) if "contour" not in part else part
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[~old.contour.isin(part.contour.unique())]
    out = pd.concat([old, part], ignore_index=True)
    lead = ["contour", "model", "horizon", "set", "sample", "years", "n", "positives"]
    out = out[[c for c in lead if c in out] + [c for c in out.columns if c not in lead]]
    out.to_csv(path, index=False, float_format=FMT)


def _save_coefs(part: pd.DataFrame) -> None:
    path = REPORTS / "party_coefficients.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[~old.contour.isin(part.contour.unique())]
    pd.concat([old, part], ignore_index=True).to_csv(path, index=False, float_format=FMT)


def _coef_rows(coef: pd.Series, contour: str, set_name: str, model: str = "logit") -> pd.DataFrame:
    """Стандартизованные коэффициенты партийных признаков (всё, что не из эталона)."""
    prefixes = ("pl_", "gl_", "vp_", "b_", "ab_", "dyl_", "dya_")
    c = coef[[i for i in coef.index if i.startswith(prefixes)]]
    return pd.DataFrame({"contour": contour, "set": set_name, "model": model, "feature": c.index, "coef": c.to_numpy()})


# ---------------------------------------------------------------------------
# Шаг 2. Разведка базовых частот
# ---------------------------------------------------------------------------

# группа → (слой, условие). Частота событий в группе сравнивается с остальными строками, где слой есть
GROUPS: dict[str, tuple[str, object]] = {
    "лидер: левый": ("gli", lambda d: d.gl_camp == "leftist"),
    "лидер: центрист": ("gli", lambda d: d.gl_camp == "centrist"),
    "лидер: правый": ("gli", lambda d: d.gl_camp == "rightist"),
    "лидер: без идеологии": ("gli", lambda d: d.gl_camp == "none"),
    "смена идеологического лагеря за 2 года": ("gli", lambda d: d.gl_switch == 1),
    "смена лидера за 2 года (GLI)": ("gli", lambda d: d.gl_leader_new == 1),
    "правящая партия: популизм ≥ 0,5": ("vparty", lambda d: d.vp_popul >= 0.5),
    "правящая партия: антиплюрализм ≥ 0,5": ("vparty", lambda d: d.vp_antiplural >= 0.5),
    "правящая партия: левая (шкала < −1)": ("vparty", lambda d: d.vp_lr < -1),
    "правящая партия: правая (шкала > +1)": ("vparty", lambda d: d.vp_lr > 1),
    "новая правящая партия (выборы ≤ 2 лет назад)": ("vparty", lambda d: d.vp_new == 1),
    "режим: националистическая идеология": ("live", lambda d: d.pl_nat >= 0.4),
    "режим: социалистическая идеология": ("live", lambda d: d.pl_soc >= 0.4),
    "режим: консервативно-реставрационная": ("live", lambda d: d.pl_cons >= 0.4),
    "режим: религиозная идеология": ("live", lambda d: d.pl_rel >= 0.4),
    "режим: персоналистский (опора на лидера ≥ 1)": ("live", lambda d: d.pl_lead >= 1),
    "режим: идеологический (опора на идеологию ≥ 1)": ("live", lambda d: d.pl_ideol >= 1),
    "идеологический разворот за 2 года (≥ 0,25)": ("live", lambda d: d.pl_turn >= 0.25),
    "смена лидера за 2 года (V-Dem)": ("live", lambda d: d.pl_leader_new == 1),
    "смена власти на выборах за 2 года": ("live", lambda d: d.pl_turnover == 1),
    "смена режима за 2 года": ("live", lambda d: d.pl_regime_new == 1),
}
STRATA = {
    "все": lambda d: pd.Series(True, index=d.index),
    "богатые (ВВП на душу выше медианы года)": lambda d: d.rich == 1,
    "бедные": lambda d: d.rich == 0,
    "демократии (V-Dem ≥ 0,5)": lambda d: d.democ >= 0.5,
    "автократии": lambda d: d.democ < 0.5,
    "постсоветские": lambda d: d.gw.isin(POST_SOVIET_GW),
}
BASE_FROM = {"война": 1989, "валютный кризис": 1970, "банковский кризис": 1970}


def income_flag() -> pd.DataFrame:
    """gw, year = T, rich: ВВП на душу (GMD) за T − 2 выше медианы стран того же года.
    Лаг — `gdp_pc_level` (как в чекпоинте 4.1)."""
    from seldon.domains.currency.econ_contagion import gdp_per_capita
    from seldon.sources.gmd import load_gmd

    g = gdp_per_capita(load_gmd())
    g = g.assign(year=g.year + publication_lag("gdp_pc_level", "jan1"), gw=g.iso3.map(ISO3_TO_GW)).dropna(subset=["gw"])
    g["rich"] = (g.gdp_pc > g.groupby("year").gdp_pc.transform("median")).astype(int)
    return g.assign(gw=g.gw.astype(int))[["gw", "year", "rich"]].drop_duplicates(["gw", "year"])


def event_frames() -> dict[str, pd.DataFrame]:
    """Три таблицы (gw, year = T, event): строки, с которых событие в окне прогноза может
    начаться, и случилось ли оно. Определения — те же, что у контуров проекта."""
    from seldon.domains.conflict import model_hist
    from seldon.domains.conflict.panel import ONSET_OF
    from seldon.sources.lv import load_lv_country_year

    w = model_hist.with_next(pd.read_parquet(DATA_PROCESSED / "features_pitf_country_year.parquet"), WAR_TARGET)
    w = w[(w[WAR_TARGET] == 0) & w.y_next.notna()]
    war = pd.DataFrame({"gw": w.gw, "year": w.year, "event": w.onset_next.astype(int)})
    assert ONSET_OF[WAR_TARGET] in w

    x = pd.read_parquet(DATA_PROCESSED / "econ_long_panel.parquet", columns=["iso3", "year", "y", "evaluable"])
    x = x[x.evaluable].assign(gw=lambda d: d.iso3.map(ISO3_TO_GW)).dropna(subset=["gw"])
    fx = pd.DataFrame({"gw": x.gw.astype(int), "year": x.year, "event": x.y.astype(int)})

    lv = load_lv_country_year()[["iso3", "year", "lv_banking"]]
    b = lv.copy()
    for j in (1, 2):
        b = b.merge(lv.assign(year=lv.year - j).rename(columns={"lv_banking": f"n{j}"}), on=["iso3", "year"], how="left")
    for j in (1, 2):  # идёт кризис или только что был — «начаться» в окне он не может
        b = b.merge(lv.assign(year=lv.year + j).rename(columns={"lv_banking": f"p{j}"}), on=["iso3", "year"], how="left")
    b = b[b.n1.notna() & b.n2.notna() & (b.lv_banking == 0) & (b.p1.fillna(0) == 0) & (b.p2.fillna(0) == 0)]
    b = b.assign(gw=b.iso3.map(ISO3_TO_GW)).dropna(subset=["gw"])
    bank = pd.DataFrame({"gw": b.gw.astype(int), "year": b.year, "event": ((b.n1 == 1) | (b.n2 == 1)).astype(int)})
    return {"война": war, "валютный кризис": fx, "банковский кризис": bank}


def base_rates(feats: pd.DataFrame, n_boot: int = 300) -> pd.DataFrame:
    """Частота событий в окне прогноза по типам власти. Признаки — с лагом публикации (строка T:
    данные за T − 1), событие — в T+1 (война) или T+1…T+2 (кризисы). Интервал разницы частот —
    блочный бутстрэп по годам."""
    rich = income_flag()
    rows = []
    for event, e in event_frames().items():
        d = attach_annual(e[e.year >= BASE_FROM[event]], feats).merge(rich, on=["gw", "year"], how="left")
        for stratum, smask in STRATA.items():
            s = d[smask(d).fillna(False)]
            for group, (layer, gmask) in GROUPS.items():
                x = s[s[pf.OK[layer]]]
                if x.empty:
                    continue
                x = x.assign(g=gmask(x).fillna(False).astype(int))
                tab = x.groupby("year").apply(lambda t: pd.Series({"n1": t.g.sum(), "e1": (t.g * t.event).sum(), "n0": (1 - t.g).sum(),
                                                                   "e0": ((1 - t.g) * t.event).sum()}), include_groups=False).reset_index()

                def stat(t: pd.DataFrame) -> float:
                    return t.e1.sum() / t.n1.sum() - t.e0.sum() / t.n0.sum() if t.n1.sum() and t.n0.sum() else np.nan

                n1, e1, n0, e0 = (int(tab[c].sum()) for c in ("n1", "e1", "n0", "e0"))
                row = {"event": event, "stratum": stratum, "group": group, "layer": layer, "years": f"{int(x.year.min())}–{int(x.year.max())}",
                       "n_group": n1, "events_group": e1, "rate_group": e1 / n1 if n1 else np.nan, "n_rest": n0, "events_rest": e0,
                       "rate_rest": e0 / n0 if n0 else np.nan}
                row["ratio"] = row["rate_group"] / row["rate_rest"] if n1 and e0 else np.nan
                if n1 >= 30 and n0 >= 30:
                    point, reps = year_block_bootstrap(tab, stat, n_boot=n_boot)
                    row["diff"], (row["diff_lo"], row["diff_hi"]) = point, interval(reps)
                rows.append(row)
    return pd.DataFrame(rows)


def coverage(feats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gw in sorted(POST_SOVIET_GW):
        g = feats[(feats.gw == gw) & (feats.year >= 1992)]
        r = {"country": GW_TO_SHORT[gw], "gw": gw}
        for layer, ok in pf.OK.items():
            have = g[g[ok]]
            r[f"{layer}_from"], r[f"{layer}_to"] = (int(have.year.min()), int(have.year.max())) if len(have) else (np.nan, np.nan)
            r[f"{layer}_years"] = len(have)
        rows.append(r)
    return pd.DataFrame(rows)


def post_soviet_coding(feats: pd.DataFrame, years=(1997, 2004, 2008, 2013, 2014, 2015, 2020, 2021, 2024)) -> pd.DataFrame:
    """Как закодированы постсоветские страны на строках ключевых лет (данные за год − 1)."""
    g = feats[feats.gw.isin(POST_SOVIET_GW) & feats.year.isin(years)].copy()
    g.insert(0, "country", g.gw.map(GW_TO_SHORT))
    cols = ["country", "year", "gl_leader", "gl_camp", "gl_switch", "vp_party", "vp_popul", "vp_antiplural", "vp_lr", "vp_new",
            "pl_ideol", "pl_lead", "pl_perf", "pl_nat", "pl_soc", "pl_cons", "pl_turn", "pl_leader_new", "pl_turnover", "pl_regime_new"]
    return g[cols].sort_values(["country", "year"]).reset_index(drop=True)


def main_describe(feats: pd.DataFrame) -> None:
    cov = coverage(feats)
    cov.to_csv(REPORTS / "party_coverage.csv", index=False, float_format="%.0f")
    print("Покрытие слоёв по постсоветским странам (строки прогноза; данные — за год − 1):")
    print(cov.to_string(index=False))
    post_soviet_coding(feats).to_csv(REPORTS / "party_post_soviet_coding.csv", index=False, float_format="%.2f")
    t = base_rates(feats)
    t.to_csv(REPORTS / "party_base_rates.csv", index=False, float_format=FMT)
    pd.set_option("display.width", 250)
    for event, g in t[t.stratum == "все"].groupby("event", sort=False):
        print(f"\n{event}: частота в окне прогноза, группа против остальных (все страны)")
        print(g[["group", "n_group", "events_group", "rate_group", "rate_rest", "ratio", "diff_lo", "diff_hi"]].round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Шаг 3.1. Войны: годовой контур
# ---------------------------------------------------------------------------


def war_panel(f: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Панель признаков модели войн + партийные признаки + целевая следующего года."""
    from seldon.domains.conflict import model_hist

    return model_hist.with_next(attach_annual(f, feats), WAR_TARGET)


def war_forecast_asof(p: pd.DataFrame, T: int, base_feats: list[str], name: str, return_models: bool = False):
    """Прогноз начала войны в T+1 набором `name` (эталонные признаки + партийные). Та же
    `model_hist.forecast_asof`; None, если T вне чистого окна набора."""
    from seldon.domains.conflict import model_hist

    q = prepare(p, WAR_SETS[name][0])
    if T > q.year.max():
        return None
    return model_hist.forecast_asof(q, T, [*base_feats, *set_features(WAR_SETS, name)], WAR_TARGET, return_models=return_models)


def run_war(feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from seldon.domains.conflict import model_hist

    base = model_hist.FEATURE_SETS["pitf"][0]
    p = war_panel(pd.read_parquet(DATA_PROCESSED / "features_pitf_country_year.parquet"), feats)
    frames, coefs = [], []
    for T in model_hist.TRAIN_YEARS:
        ref = model_hist.forecast_asof(p, T, base, WAR_TARGET)
        ref = ref[ref.y_next.notna()]
        out = ref[["gw", "country", "at_risk", "years_since_war"]].assign(year=T + 1, onset=ref.onset_next.astype(int), p_ref=ref.p)
        for name in WAR_SETS:
            res = war_forecast_asof(p, T, base, name, return_models=True)
            if res is None:
                out[f"p_{name}"] = np.nan
                continue
            test, m = res
            out[f"p_{name}"] = out.gw.map(test.set_index("gw").p)
            if m is not None:
                c = pd.Series(m[-1].coef_[0], index=[*base, *set_features(WAR_SETS, name)])
                coefs.append(_coef_rows(c, "война", name).assign(year=T + 1))
        frames.append(out)
    c = pd.concat(coefs, ignore_index=True)
    last = c[c.year == c.groupby("set").year.transform("max")].drop(columns="year")
    mean = c.groupby(["set", "feature"]).coef.mean().rename("coef_mean_over_years").reset_index()
    return pd.concat(frames, ignore_index=True), last.merge(mean, on=["set", "feature"])


WAR_SAMPLES = {
    "мир": lambda d: d.at_risk == 1,
    "тихие ≥ 10 лет": lambda d: (d.at_risk == 1) & (d.years_since_war >= 10),
    "постсоветские": lambda d: (d.at_risk == 1) & d.gw.isin(POST_SOVIET_GW),
}


def main_war(feats: pd.DataFrame) -> None:
    pred, coefs = run_war(feats)
    pred.to_parquet(DATA_PROCESSED / "party_war_predictions.parquet", index=False)
    t = compare(pred, "p_ref", {n: f"p_{n}" for n in WAR_SETS}, "onset", WAR_SAMPLES, contour="война", model="logit")
    _save_oos(t, "война")
    _save_coefs(coefs)
    print(f"\nВойны, начало {WAR_TARGET} в T+1, эталон pitf:")
    print(show(t))
    print(coefs[coefs.set.isin(["live", "arch"])].round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Шаг 3.1б. Диады
# ---------------------------------------------------------------------------


def dyad_panel(d: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Диадная панель A ← B + идеология потенциального противника B и дистанция пары.
    Пропуски не заполнены; `dyl_ok` / `dya_ok` — есть ли слой у ОБЕИХ стран пары."""
    live = ["pl_ideol", "pl_lead", "pl_nat", "pl_soc", "pl_cons", "pl_rel", "pl_leader_new", "pl_regime_new", "legit_ok"]
    arch = ["gl_left", "gl_right", "gl_camp", "vp_popul", "vp_antiplural", "gli_ok", "vp_ok"]
    t = feats[["gw", "year", *live, *arch]]
    out = d.merge(t.rename(columns={c: f"a__{c}" for c in t.columns if c != "year"}).rename(columns={"a__gw": "a"}), on=["a", "year"], how="left")
    out = out.merge(t.rename(columns={c: f"b__{c}" for c in t.columns if c != "year"}).rename(columns={"b__gw": "b"}), on=["b", "year"], how="left")
    ok = lambda c: out[f"a__{c}"].fillna(False).astype(bool) & out[f"b__{c}"].fillna(False).astype(bool)  # noqa: E731
    out["dyl_ok"], out["dya_ok"] = ok("legit_ok"), ok("gli_ok") & ok("vp_ok")
    out["b_nat"], out["b_lead"], out["b_ideol"] = out.b__pl_nat, out.b__pl_lead, out.b__pl_ideol
    out["b_leader_new"], out["b_regime_new"] = out.b__pl_leader_new, out.b__pl_regime_new
    out["ab_ideol_dist"] = (out.a__pl_ideol - out.b__pl_ideol).abs()
    chars = ["pl_nat", "pl_soc", "pl_cons", "pl_rel"]
    out["ab_char_dist"] = 0.5 * sum((out[f"a__{c}"] - out[f"b__{c}"]).abs() for c in chars)
    out["b_gl_right"], out["b_gl_left"] = out.b__gl_right, out.b__gl_left
    out["ab_camp_diff"] = (out.a__gl_camp != out.b__gl_camp).astype(float).where(out.a__gl_camp.notna() & out.b__gl_camp.notna())
    out["b_popul"], out["b_antiplural"] = out.b__vp_popul, out.b__vp_antiplural
    for layer, cols in (("live", DY_LIVE), ("arch", DY_ARCH)):
        out.loc[~out[f"dy{layer[0]}_ok"], cols] = np.nan
    return out.drop(columns=[c for c in out.columns if c.startswith(("a__", "b__"))])


def dyad_prepare(d: pd.DataFrame, layer: str) -> pd.DataFrame:
    """Чистое окно слоя пары и заполнение пропусков внутри него (медиана года / ноль) с флагом."""
    ok, cols, flag = f"dy{layer[0]}_ok", (DY_LIVE if layer == "live" else DY_ARCH), DY_FLAG[layer]
    d = d[d.year <= d.loc[d[ok], "year"].max()].copy()
    d[flag] = (~d[ok]).astype(int)
    for c in cols:
        binary = c in ("b_leader_new", "b_regime_new", "b_gl_right", "b_gl_left", "ab_camp_diff")
        d[c] = d[c].fillna(0.0) if binary else d[c].fillna(d.groupby("year")[c].transform("median")).fillna(0.0)
    return d


def dyad_forecast_asof(d: pd.DataFrame, T: int, name: str, return_model: bool = False):
    """Прогноз начала войны пары в T+1: признаки `dyads.FEATURES` + набор `name`. `d` — панель
    `dyad_panel` с колонкой `y_next`. None, если T вне чистого окна."""
    from seldon.domains.conflict import model_dyad
    from seldon.domains.conflict.dyads import FEATURES

    layer, cols = DYAD_SETS[name]
    q = dyad_prepare(d, layer)
    if T > q.year.max():
        return None
    return model_dyad.forecast_asof(q, T, features=[*FEATURES, *cols, DY_FLAG[layer]], return_model=return_model)


def run_dyad(feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from seldon.domains.conflict import model_dyad
    from seldon.domains.conflict.dyads import FEATURES

    d = model_dyad.with_next(dyad_panel(pd.read_parquet(DATA_PROCESSED / "dyad_features.parquet"), feats))
    frames, coefs = [], []
    for T in model_dyad.TRAIN_YEARS:
        ref = model_dyad.forecast_asof(d, T)
        ref = ref[ref.y_next.notna()]
        out = ref[["a", "b", "contig_close", "b_power"]].assign(year=T + 1, onset=ref.y_next.astype(int), p_ref=ref.p_dyad)
        for name, (layer, cols) in DYAD_SETS.items():
            res = dyad_forecast_asof(d, T, name, return_model=True)
            if res is None or res[1] is None:
                out[f"p_{name}"] = np.nan
                continue
            test, m = res
            out[f"p_{name}"] = test.p_dyad.reindex(out.index)
            c = pd.Series(m[-1].coef_[0], index=[*FEATURES, *cols, DY_FLAG[layer]])
            coefs.append(_coef_rows(c, "диады", name).assign(year=T + 1))
        frames.append(out)
    c = pd.concat(coefs, ignore_index=True)
    last = c[c.year == c.groupby("set").year.transform("max")].drop(columns="year")
    mean = c.groupby(["set", "feature"]).coef.mean().rename("coef_mean_over_years").reset_index()
    return pd.concat(frames, ignore_index=True), last.merge(mean, on=["set", "feature"])


DYAD_SAMPLES = {
    "все пары": lambda d: d.a.notna(),
    "близкие соседи": lambda d: d.contig_close == 1,
    "B — держава без общей границы": lambda d: d.contig_close == 0,
    "A — постсоветская": lambda d: d.a.isin(POST_SOVIET_GW),
}


def main_dyad(feats: pd.DataFrame) -> None:
    pred, coefs = run_dyad(feats)
    pred.to_parquet(DATA_PROCESSED / "party_dyad_predictions.parquet", index=False)
    t = compare(pred, "p_ref", {n: f"p_{n}" for n in DYAD_SETS}, "onset", DYAD_SAMPLES, contour="диады", model="logit")
    _save_oos(t, "диады")
    _save_coefs(coefs)
    print("\nДиады, начало войны пары A ← B в T+1, эталон — диадная модель чекпоинта 3.2:")
    print(show(t))
    print(coefs.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Шаг 3.2а. Валюта, годовой контур
# ---------------------------------------------------------------------------


def _load_struct() -> pd.DataFrame:
    st = pd.read_parquet(DATA_PROCESSED / "fx_struct_monthly.parquet")
    st["month"] = pd.PeriodIndex(st.month, freq="M")
    return st


def fx_long_panel(feats: pd.DataFrame, struct: pd.DataFrame | None = None, **inputs) -> pd.DataFrame:
    """Годовая валютная панель (как в `econ_struct`: боевая + структурные колонки) + партийные."""
    from seldon.domains.currency import econ_struct

    p = econ_struct.build_long_panel(_load_struct() if struct is None else struct, **inputs)
    return attach_annual(p, feats, key="iso3")


def fx_long_forecast_asof(p: pd.DataFrame, T: int, name: str, n_estimators: int = 400, return_models: bool = False):
    """Прогноз `deval25` на T+1…T+2: боевой набор + партийный набор `name` (C выбирается внутри
    обучения, как у наборов чекпоинта 3.3). None, если T вне чистого окна."""
    from seldon.domains.currency import econ_long
    from seldon.domains.currency.long_features import BASE, MISSING

    layers, cols = FX_SETS[name]
    q = prepare(p, layers)
    if T > q.year.max():
        return None
    q = add_fx_interactions(q, "peg_years_log", (-q.ca_level).clip(lower=0))
    flags = [pf.LAYERS[layer][1] for layer in layers]
    return econ_long.forecast_asof(q, T, n_estimators=n_estimators, cont=[*BASE, *cols], flags=[*MISSING, *flags], C=None,
                                   return_models=return_models)


def run_fx_long(feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from seldon.domains.currency import econ_long
    from seldon.domains.currency.long_features import BASE, MISSING

    p = fx_long_panel(feats)
    last_eval = int(p[p.evaluable].year.max())
    frames, coefs = [], []
    for T in range(econ_long.EVAL_FROM, last_eval + 1):
        ref = econ_long.forecast_asof(p, T)
        out = ref[ref.evaluable][["iso3", "year", "post_soviet", "y", "logit", "extree"]].rename(columns={"logit": "logit__ref", "extree": "extree__ref"})
        if out.empty or out.logit__ref.isna().all():
            continue
        base = econ_long.forecast_asof(p, T, C=None)
        out["logit__base"] = base.logit.reindex(out.index)
        for name, (layers, cols) in FX_SETS.items():
            res = fx_long_forecast_asof(p, T, name, return_models=True)
            if res is None:
                out[f"logit__{name}"] = out[f"extree__{name}"] = np.nan
                continue
            test, models = res
            m = test.set_index("iso3")
            out[f"logit__{name}"], out[f"extree__{name}"] = out.iso3.map(m.logit), out.iso3.map(m.extree)
            if models:
                idx = [*BASE, *cols, *MISSING, *[pf.LAYERS[layer][1] for layer in layers]]
                coefs.append(_coef_rows(pd.Series(models["logit"][-1].coef_[0], index=idx), "валюта годовая", name).assign(year=T))
        print(f"  fx_long {T}: строк {len(out)}, кризисных окон {int(out.y.sum())}")
        frames.append(out)
    c = pd.concat(coefs, ignore_index=True)
    last = c[c.year == c.groupby("set").year.transform("max")].drop(columns="year")
    mean = c.groupby(["set", "feature"]).coef.mean().rename("coef_mean_over_years").reset_index()
    return pd.concat(frames, ignore_index=True), last.merge(mean, on=["set", "feature"])


FX_SAMPLES = {"мир": lambda d: d.iso3.notna(), "постсоветские": lambda d: d.iso3.isin(POST_SOVIET_ISO3)}


def main_fx_long(feats: pd.DataFrame) -> None:
    pred, coefs = run_fx_long(feats)
    pred.to_parquet(DATA_PROCESSED / "party_fx_long_predictions.parquet", index=False)
    parts = []
    for kind in ("logit", "extree"):
        cols = {n: f"{kind}__{n}" for n in FX_SETS}
        if kind == "logit":
            cols = {"base (тот же набор, C подобрана)": "logit__base", **cols}
        parts.append(compare(pred, f"{kind}__ref", cols, "y", FX_SAMPLES, contour="валюта годовая", model=kind))
        if kind == "logit":  # тот же вопрос против эталона с подобранной C: вклад признаков, а не настройки
            parts.append(compare(pred, "logit__base", {n: f"logit__{n}" for n in FX_SETS}, "y", FX_SAMPLES,
                                 contour="валюта годовая", model="logit к base"))
    t = pd.concat(parts, ignore_index=True)
    _save_oos(t, "валюта годовая")
    _save_coefs(coefs)
    print("\nВалюта, годовой контур: deval25 в T+1…T+2, эталон — боевая модель econ_long:")
    print(show(t))
    print(coefs.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Шаг 3.2б. Валюта, месячный контур
# ---------------------------------------------------------------------------

MONTH_HORIZONS = (12, 3)
PEG_SHARE = 0.8


def fx_month_panel(tabs: dict[str, pd.DataFrame], p: pd.DataFrame | None = None) -> pd.DataFrame:
    """Полная месячная панель (`econ_csd.load_full_panel`) + партийные признаки с месячным лагом."""
    from seldon.domains.currency import econ_csd

    p = econ_csd.load_full_panel() if p is None else p
    return pf.attach_monthly(p, {k: by_iso3(t) for k, t in tabs.items()}, key="iso3")


def fx_month_predict(p: pd.DataFrame, as_of: pd.Period, test: pd.DataFrame, feats: list[str], h: int, C: float | None = None):
    """Логрегрессия месячного контура на признаках `feats`: обучение — окна, закрывшиеся к
    `as_of` (месяц + h ≤ as_of); C — фиксированная или выбранная внутри обучения. Те же
    `_kr_fit` / `choose_C_kr`, что у `econ_struct`, `econ_contagion`, `econ_csd`."""
    from seldon.domains.currency.econ_struct import KR_MIN_POS, _kr_design, _kr_fit, choose_C_kr

    known = p[label_known(p.month, as_of, h) & p.evaluable]
    if known.y.sum() < KR_MIN_POS:
        return None
    m, lo, hi, med = _kr_fit(known, feats, choose_C_kr(known, feats, h=h) if C is None else C)
    return m.predict_proba(_kr_design(test, feats, lo, hi, med))[:, 1], pd.Series(m[-1].coef_[0], index=feats)


def fx_month_set_panel(p: pd.DataFrame, name: str, sets: dict | None = None) -> pd.DataFrame:
    q = prepare(p, (FX_SETS if sets is None else sets)[name][0], period="month")
    q["democ"] = q.democ.fillna(q.groupby("month").democ.transform("median")).fillna(0.5)  # только для разбора прироста
    return add_fx_interactions(q, "peg_share12", None, resfall="peg_resfall")


# Разбор прироста живого слоя в месячном контуре: что именно его даёт. Все наборы — слой `live`
# (флаг `pl_missing` добавляется к каждому): только флаг; только уровень демократии V-Dem
# (контроль «это просто тип режима»); только персонализм; он же плюс демократия; уровни
# легитимации; смены; всё без взаимодействий
ABLATION_SETS: dict[str, tuple[tuple[str, ...], list[str]]] = {
    "flag": (("live",), []),
    "democ": (("live",), ["democ"]),
    "lead": (("live",), ["pl_lead"]),
    "lead_democ": (("live",), ["pl_lead", "democ"]),
    "legit": (("live",), pf.LIVE_LEGIT),
    "turn": (("live",), pf.LIVE_TURN),
    "no_inter": (("live",), [*pf.LIVE_LEGIT, *pf.LIVE_TURN]),
    "live": FX_SETS["live"],
}


def run_fx_month(p: pd.DataFrame, h: int, sets: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from seldon.domains.currency import econ_contagion, econ_csd, econ_kr
    from seldon.domains.currency.econ_struct import KR_C_REF
    from seldon.domains.currency.kr_features import INDICATORS

    p = econ_contagion.with_horizon(p, h)
    families = {"klr": list(INDICATORS), "swap": econ_csd.features(econ_csd.CHALLENGER)}
    sets = FX_SETS if sets is None else sets
    panels = {name: fx_month_set_panel(p, name, sets) for name in sets}
    frames, coefs = [], []
    for T in range(econ_kr.FIRST_EVAL_YEAR, econ_kr.LAST_EVAL_YEAR + 1):
        as_of = pd.Period(f"{T - 1}-12", freq="M")
        test = p[(p.month.dt.year == T) & p.own_currency & p.fx.notna()]
        if not test.evaluable.any():
            continue
        out = test[["iso3", "month", "post_soviet", "y", "evaluable", "peg_share12"]].copy()
        res = {"klr__ref": fx_month_predict(p, as_of, test, families["klr"], h, C=KR_C_REF),
               "klr__base": fx_month_predict(p, as_of, test, families["klr"], h),
               "swap__ref": fx_month_predict(p, as_of, test, families["swap"], h)}
        if any(v is None for v in res.values()):
            continue
        for k, v in res.items():
            out[k] = v[0]
        for name, (layers, cols) in sets.items():
            q = panels[name]
            qt = q[(q.month.dt.year == T) & q.own_currency & q.fx.notna()]
            extra = [*cols, *[pf.LAYERS[layer][1] for layer in layers]]
            for fam, feats in families.items():
                r = fx_month_predict(q, as_of, qt, [*feats, *extra], h) if len(qt) else None
                out[f"{fam}__{name}"] = np.nan if r is None else pd.Series(r[0], index=qt.index).reindex(out.index)
                if r is not None:
                    coefs.append(_coef_rows(r[1], f"валюта месячная h={h}", name, model=fam).assign(year=T))
        print(f"  fx_month h={h} {T}: строк {int(out.evaluable.sum())}, кризисных окон {int(out.y[out.evaluable].sum())}")
        frames.append(out)
    c = pd.concat(coefs, ignore_index=True)
    last = c[c.year == c.groupby(["set", "model"]).year.transform("max")].drop(columns="year")
    mean = c.groupby(["set", "model", "feature"]).coef.mean().rename("coef_mean_over_years").reset_index()
    return pd.concat(frames, ignore_index=True), last.merge(mean, on=["set", "model", "feature"])


MONTH_SAMPLES = {
    "мир": lambda d: d.iso3.notna(),
    "постсоветские": lambda d: d.iso3.isin(POST_SOVIET_ISO3),
    "привязки": lambda d: d.peg_share12 >= PEG_SHARE,
}


def main_fx_month(tabs: dict[str, pd.DataFrame]) -> None:
    p = fx_month_panel(tabs)
    parts, coefs = [], []
    for h in MONTH_HORIZONS:
        print(f"\nВалюта, месячный контур, горизонт {h} мес…")
        pred, c = run_fx_month(p, h)
        pred.assign(month=pred.month.astype(str)).to_parquet(DATA_PROCESSED / f"party_fx_month_predictions_h{h}.parquet", index=False)
        coefs.append(c)
        d = pred[pred.evaluable].assign(year=lambda x: x.month.dt.year)
        tag = {"contour": "валюта месячная", "horizon": h}
        parts.append(compare(d, "klr__ref", {"base (тот же набор, C подобрана)": "klr__base", **{n: f"klr__{n}" for n in FX_SETS}}, "y",
                             MONTH_SAMPLES, model="боевая econ_kr", **tag))
        parts.append(compare(d, "klr__base", {n: f"klr__{n}" for n in FX_SETS}, "y", MONTH_SAMPLES, model="econ_kr к base", **tag))
        parts.append(compare(d, "swap__ref", {n: f"swap__{n}" for n in FX_SETS}, "y", MONTH_SAMPLES, model="претендент live_fx_swap", **tag))
    t = pd.concat(parts, ignore_index=True)
    _save_oos(t, "валюта месячная")
    _save_coefs(pd.concat(coefs, ignore_index=True))
    print("\nВалюта, месячный контур: deval15, эталоны — боевая econ_kr и претендент live_fx_swap:")
    print(show(t))


def main_ablation(tabs: dict[str, pd.DataFrame]) -> None:
    """Что даёт прирост живого слоя в месячном контуре (`reports/party_fx_month_ablation.csv`)."""
    p = fx_month_panel(tabs)
    parts = []
    for h in MONTH_HORIZONS:
        pred, _ = run_fx_month(p, h, ABLATION_SETS)
        d = pred[pred.evaluable].assign(year=lambda x: x.month.dt.year)
        samples = {**MONTH_SAMPLES, "мир, до 2010": lambda x: x.year < 2010, "мир, с 2010": lambda x: x.year >= 2010,
                   "постсоветские, до 2010": lambda x: x.iso3.isin(POST_SOVIET_ISO3) & (x.year < 2010),
                   "постсоветские, с 2010": lambda x: x.iso3.isin(POST_SOVIET_ISO3) & (x.year >= 2010)}
        for fam, title in (("klr", "econ_kr к base"), ("swap", "претендент live_fx_swap")):
            parts.append(compare(d, f"{fam}__{'base' if fam == 'klr' else 'ref'}", {n: f"{fam}__{n}" for n in ABLATION_SETS}, "y", samples,
                                 contour="валюта месячная", model=title, horizon=h))
    t = pd.concat(parts, ignore_index=True)
    t.to_csv(REPORTS / "party_fx_month_ablation.csv", index=False, float_format=FMT)
    print("\nРазбор прироста живого слоя в месячном контуре:")
    print(show(t))


# ---------------------------------------------------------------------------
# Шаг 3.3. Банковские кризисы
# ---------------------------------------------------------------------------

JST_FROM = 1946          # обучение: послевоенный период (до него нет GLI, а режимы другие)
BANK_MIN_POS = 10
BANK_TREES = 500


def bank_frames(feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(JST R6, панель BIS) в формате BoE с лагом публикации макроданных + партийные признаки."""
    from seldon.domains.banking import econ_boe_em as em
    from seldon.sources.jst import JST_R6, load_jst

    jst = em.boe_dataset(load_jst(JST_R6, "Sheet1"), em.FULL)
    data = em.build(em.load_panel(em.Config()), em.FULL)
    return _bank_attach(jst[jst.year >= JST_FROM], feats), _bank_attach(data, feats)


def _bank_attach(d: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    return attach_annual(d.rename(columns={"iso": "iso3"}), feats, key="iso3").rename(columns={"iso3": "iso"})


def bank_jst_forecast_asof(d: pd.DataFrame, T: int, feats: list[str], n_estimators: int = BANK_TREES) -> pd.DataFrame | None:
    """Прогноз банковского кризиса в T+1…T+2 на JST: обучение — строки с известной к T меткой
    (год ≤ T − 2). Модели BoE: логрегрессия без штрафа и деревья."""
    from seldon.domains.banking import econ_boe_em as em

    train, test = d[label_known(d.year, T, em.HORIZON)], d[d.year == T].copy()
    if train.crisis.sum() < BANK_MIN_POS or test.empty:
        return None
    for k, m in em.make_models(n_estimators=n_estimators).items():
        m.fit(train[feats].values, train.crisis.values)
        test[k] = m.predict_proba(test[feats].values)[:, 1]
    return test


def run_bank(feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from seldon.domains.banking import econ_boe_em as em

    jst, data = bank_frames(feats)
    cfg = em.Config()
    jst_frames, bis_frames = [], []
    for name, (layers, cols) in BANK_SETS.items():
        extra = [*cols, *[pf.LAYERS[layer][1] for layer in layers]]
        jq, dq = prepare(jst, layers, lower=True), prepare(data, layers, lower=True)
        for T in range(int(jq.year.min()) + 5, em.JST_LAST_LABEL_YEAR - em.HORIZON + 1):
            if T > jq.year.max():
                break
            ref, new = bank_jst_forecast_asof(jq, T, em.FULL), bank_jst_forecast_asof(jq, T, [*em.FULL, *extra])
            if ref is None:
                continue
            out = ref[["iso", "year", "crisis"]].assign(set=name)
            for k in ("logreg", "extree"):
                out[f"{k}__ref"], out[f"{k}__set"] = ref[k], new[k]
            jst_frames.append(out)
        for T in cfg.eval_years:
            if T > dq.year.max():
                break
            ref = em.forecast_asof(dq, jq, em.FULL, T, n_estimators=BANK_TREES)
            new = em.forecast_asof(dq, jq, [*em.FULL, *extra], T, n_estimators=BANK_TREES)
            if ref.empty:
                continue
            out = ref[["iso", "year", "crisis", "em"]].assign(set=name)
            for k in ("jst_logreg", "jst_extree", "pooled_extree"):
                out[f"{k}__ref"], out[f"{k}__set"] = ref[k], new[k]
            bis_frames.append(out)
        print(f"  bank {name}: JST и BIS посчитаны")
    return pd.concat(jst_frames, ignore_index=True), pd.concat(bis_frames, ignore_index=True)


def main_bank(feats: pd.DataFrame) -> None:
    from seldon.domains.banking.econ_boe_em import JST_ISO

    jst, bis = run_bank(feats)
    jst.to_parquet(DATA_PROCESSED / "party_bank_jst_predictions.parquet", index=False)
    bis.to_parquet(DATA_PROCESSED / "party_bank_bis_predictions.parquet", index=False)
    parts = []
    for name in BANK_SETS:
        j, b = jst[jst.set == name], bis[bis.set == name]
        for k in ("logreg", "extree"):
            parts.append(compare(j, f"{k}__ref", {name: f"{k}__set"}, "crisis", {"JST, 18 развитых": lambda d: d.iso.notna()},
                                 contour="банки", model=f"JST {k}"))
        scopes = {"BIS, все": lambda d: d.iso.notna(), "BIS, развивающиеся": lambda d: d.em, "BIS, вне JST": lambda d: ~d.iso.isin(JST_ISO)}
        for k in ("jst_logreg", "jst_extree", "pooled_extree"):
            parts.append(compare(b, f"{k}__ref", {name: f"{k}__set"}, "crisis", scopes, contour="банки", model=f"BIS {k}"))
    t = pd.concat(parts, ignore_index=True)
    _save_oos(t, "банки")
    print(f"\nБанки: кризис в T+1…T+2; JST — обучение с {JST_FROM} года; BIS — перенос с JST (`jst_*`) и общая модель (`pooled`):")
    print(show(t))


# ---------------------------------------------------------------------------
# Шаг 4. Контрольные кейсы
# ---------------------------------------------------------------------------


def cases() -> pd.DataFrame:
    """Место страны (пары) среди всех под риском в выпуске перед событием: эталон и наборы.
    1 = самый высокий риск. Пусто — у набора нет прогноза (кейс вне его чистого окна)."""
    from seldon.domains.conflict.model_dyad import CASES as DYAD_CASES

    rows = []

    def add(contour, case, peers, me, ref, cols, status=""):
        row = {"contour": contour, "case": case, "n_peers": len(peers), "status": status}
        if len(me):
            row["rank_ref"], row["p_ref"] = _rank(peers[ref], me[ref].iloc[0]), me[ref].iloc[0]
            for name, col in cols.items():
                ok = peers[col].notna()
                if pd.notna(me[col].iloc[0]):
                    row[f"rank_{name}"] = _rank(peers.loc[ok, col], me[col].iloc[0])
                    row[f"rank_ref_same_rows_{name}"] = _rank(peers.loc[ok, ref], me[ref].iloc[0])
        rows.append(row)

    w = pd.read_parquet(DATA_PROCESSED / "party_war_predictions.parquet")
    w = w[w.at_risk == 1]
    for label, gw, year in WAR_CASES:
        peers = w[w.year == year]
        me = peers[peers.gw == gw]
        add("война", label, peers, me, "p_ref", {n: f"p_{n}" for n in WAR_SETS},
            "" if len(me) and me.onset.iloc[0] == 1 else "по UCDP не начало (страна уже воевала или порог не пройден)")

    d = pd.read_parquet(DATA_PROCESSED / "party_dyad_predictions.parquet")
    for label, a, b, year in DYAD_CASES:
        peers = d[d.year == year]
        me = peers[(peers.a == a) & (peers.b == b)]
        add("диады", f"{label}, {year}", peers, me, "p_ref", {n: f"p_{n}" for n in DYAD_SETS},
            "" if len(me) and me.onset.iloc[0] == 1 else ("пара уже воевала или вне вселенной" if me.empty else "под риском, войны пары по UCDP не было"))

    panel = pd.read_parquet(DATA_PROCESSED / "econ_long_panel.parquet", columns=["iso3", "year", "deval25", "deval"])
    x = pd.read_parquet(DATA_PROCESSED / "party_fx_long_predictions.parquet")
    ev = panel[(panel.deval25 == 1) & panel.iso3.isin(FX_CASE_ISO) & (panel.year >= 1997)]
    for e in ev.sort_values(["iso3", "year"]).itertuples():
        peers = x[x.year == e.year - 1]
        me = peers[peers.iso3 == e.iso3]
        for kind in ("logit", "extree"):
            add(f"валюта годовая, {kind}", f"{e.iso3}-{e.year} (девальвация {e.deval:+.0%}), за год до", peers, me, f"{kind}__ref",
                {n: f"{kind}__{n}" for n in FX_SETS}, "" if len(me) else "строка за год до не оценивается (идёт прошлый кризис)")

    from seldon.domains.currency import econ_kr
    mp = pd.read_parquet(DATA_PROCESSED / "econ_kr_panel_deval15.parquet", columns=["iso3", "month", "deval15"])
    mp["month"] = pd.PeriodIndex(mp.month, freq="M")
    ev = mp[(mp.deval15 == 1) & mp.iso3.isin(FX_CASE_ISO) & (mp.month.dt.year > econ_kr.FIRST_EVAL_YEAR)]
    for h in MONTH_HORIZONS:
        m = pd.read_parquet(DATA_PROCESSED / f"party_fx_month_predictions_h{h}.parquet")
        m["month"] = pd.PeriodIndex(m.month, freq="M")
        for e in ev.sort_values(["iso3", "month"]).itertuples():
            peers = m[m.month == e.month - h]
            me = peers[peers.iso3 == e.iso3]
            for fam, title in (("klr", "боевая econ_kr"), ("swap", "претендент")):
                add(f"валюта месячная, {title}", f"{e.iso3}-{e.month}, за {h} мес", peers, me, f"{fam}__ref", {n: f"{fam}__{n}" for n in FX_SETS})

    b = pd.read_parquet(DATA_PROCESSED / "party_bank_bis_predictions.parquet")
    lv = pd.read_parquet(DATA_PROCESSED / "lv_country_year.parquet")
    for e in lv[(lv.lv_banking == 1) & lv.iso3.isin(["RUS", "TUR", "ARG"]) & (lv.year >= 1992)].itertuples():
        for name in BANK_SETS:
            peers = b[(b.year == e.year - 1) & (b.set == name)]
            me = peers[peers.iso == e.iso3]
            add("банки, BIS jst_extree", f"{e.iso3}-{e.year} (LV), за год до, набор {name}", peers, me, "jst_extree__ref", {"set": "jst_extree__set"},
                "" if len(me) else "страны нет в панели BIS на этот год")
    return pd.DataFrame(rows)


def main_cases() -> None:
    t = cases()
    t.to_csv(REPORTS / "party_cases.csv", index=False, float_format=FMT)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print("\nКонтрольные кейсы (место среди стран / пар под риском; 1 — самый высокий риск):")
    print(t.drop(columns=[c for c in t.columns if c.startswith("rank_ref_same_rows")]).round(3).to_string(index=False))


# ---------------------------------------------------------------------------

STEPS = ("describe", "war", "dyad", "fx_long", "fx_month", "ablation", "bank", "cases")


def main(argv: list[str]) -> int:
    if any(a not in STEPS for a in argv):
        print(__doc__)
        return 2
    steps = argv or list(STEPS)
    src = load_sources()
    feats = load_features(src)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(DATA_PROCESSED / "party_features_country_year.parquet", index=False)
    print(f"партийные признаки: {feats.gw.nunique()} стран, строки {feats.year.min()}–{feats.year.max()}; чистые окна (последняя строка): "
          + ", ".join(f"{layer} — {int(pf.clean_window(feats, layer).year.max())}" for layer in pf.OK))
    if "describe" in steps:
        main_describe(feats)
    if "war" in steps:
        main_war(feats)
    if "dyad" in steps:
        main_dyad(feats)
    if "fx_long" in steps:
        main_fx_long(feats)
    if "fx_month" in steps:
        main_fx_month(pf.own_year_tables(*src))
    if "ablation" in steps:
        main_ablation(pf.own_year_tables(*src))
    if "bank" in steps:
        main_bank(feats)
    if "cases" in steps:
        main_cases()
    return 0
