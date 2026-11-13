"""Доводка диадного канала (чекпоинт 4.2): живые стычки вместо MID, раздельные модели для
соседей и далёких держав, калибровка внешнего канала `p_ext`.

Запуск:
    uv run python -m seldon dyads_live      # нужны `dyads`, `model_dyad`, `model_hist pitf`

Что меняется относительно чекпоинта 3.2 (`dyads.py`, `model_dyad.py` — не тронуты):

1. Живая хроника. `mid_force` (споры COW MID, заморожены на 2014 годе) заменён тремя
   признаками из событий UCDP (`live_pairs.annual_features`): `live_years15`, `live_ld3`,
   `live_ev12` — прямые столкновения правительств, прокси-столкновения и вторжения, включая
   субпороговые (1–24 погибших). События — по ноябрь года T (декабрьский файл Candidate
   выходит 20 января, а прогноз делается «на 1 января T+1»).
2. Вселенная пар расширена «проявленной враждой»: к соседям и парам с державой добавлены
   пары, у которых за последние 15 лет было хоть одно живое событие (Израиль ↔ Иран,
   Израиль ↔ Сирия). Решение о включении опирается только на прошлое.
3. Раздельные параметры. Пары делятся на `near` (есть общая граница, суша или море до
   400 миль) и `far` (границы нет: далёкая держава или проявленная вражда). У каждой группы
   своя логрегрессия и своя `C`; если в обучении группы меньше `MIN_GROUP_EVENTS` начал —
   группа берёт общую модель.
4. Калибровка `p_ext`. Сырая свёртка `s = 1 − ∏(1 − p_пары)` считает коалиционную войну
   столько раз, сколько в ней противников. Внутри обучения (метки ≤ T) те же диадные прогнозы
   сворачиваются по странам, и на страновой целевой «начало войны с государством-противником
   в t+1» учится одномерная логистическая поправка (Platt) от logit(s). В тест не подглядываем.

Варианты диадной модели (все — на расширенной вселенной):
    mid_pooled    признаки 3.2 (с `mid_force`), одна модель — ЭТАЛОН;
    live_pooled   живые стычки вместо `mid_force`, одна модель;
    mid_split     признаки 3.2, раздельные модели;
    live_split    живые стычки + раздельные модели — основной вариант;
    both_split    `mid_force` и живые стычки вместе, раздельные модели.
Страновой уровень (для `compare_war live`), p_int — внутренний канал из
`model_dyad.internal_channel`, p_ext — откалиброванный внешний:
    dy_live_raw     1 − (1 − p_int)(1 − s), диадная модель `live_split`, БЕЗ калибровки;
    dy_live         1 − (1 − p_int)(1 − p_ext), `live_split` — все три правки вместе;
    dy_live_pooled  то же на `live_pooled` (живые стычки, одна модель);
    dy_cal          то же на `mid_pooled` — признаки 3.2, из правок только калибровка.

Выход: data/processed/dyad_live_features.parquet, dyad_live_predictions.parquet,
dyad_live_country.parquet, dy_{live,live_raw,live_pooled,cal}_state_war25_predictions.parquet;
reports/dyad_live_metrics.csv, dyad_live_coefficients.csv, dyad_live_cases.csv,
dyad_live_ext_channel.csv, dyad_live_top_<год>.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from seldon import DATA_PROCESSED, REPORTS
from seldon.core.metrics import auc, paired_comparison
from seldon.core.temporal import label_known
from seldon.domains.conflict import dyads, live_pairs, model_dyad, model_hist
from seldon.domains.conflict.live_pairs import LIVE_FEATURES
from seldon.evaluation.backtest import backtest
from seldon.sources.cow import load_contiguity
from seldon.sources.gw import load_states
from seldon.sources.mid import load_mid
from seldon.sources.ucdp import YEAR_FROM, YEAR_TO, load_acd, load_ged
from seldon.sources.unga import load_unga

HORIZON = 1
TRAIN_YEARS = model_dyad.TRAIN_YEARS
TARGET = model_dyad.TARGET
MIN_GROUP_EVENTS = 10
CAL_MIN_EVENTS = 10
N_BOOT = 500
MID_FEATURES = list(dyads.FEATURES)
LIVE_SET = [c for c in dyads.FEATURES if c != "mid_force"] + LIVE_FEATURES
VARIANTS: dict[str, tuple[list[str], bool]] = {  # имя → (признаки, раздельные ли модели)
    "mid_pooled": (MID_FEATURES, False),
    "live_pooled": (LIVE_SET, False),
    "mid_split": (MID_FEATURES, True),
    "live_split": (LIVE_SET, True),
    "both_split": ([*MID_FEATURES, *LIVE_FEATURES], True),
}
MAIN = "live_split"
# страновой вариант → диадная модель под ним (у всех — откалиброванный p_ext)
COUNTRY_VARIANTS = {"live_split": "dy_live", "mid_pooled": "dy_cal", "live_pooled": "dy_live_pooled"}
CASES = [*model_dyad.CASES,
         ("Пакистан ← Индия", 770, 750, 2019), ("Индия ← Пакистан", 750, 770, 2019),
         ("Пакистан ← Индия", 770, 750, 2025), ("Индия ← Пакистан", 750, 770, 2025),
         ("Армения ← Азербайджан", 371, 373, 2016), ("Армения ← Азербайджан", 371, 373, 2022),
         ("Кыргызстан ← Таджикистан", 703, 702, 2022), ("Иран ← Израиль", 630, 666, 2025)]


# ---------------------------------------------------------------------------
# Панель пар
# ---------------------------------------------------------------------------


def extend_grid(grid: pd.DataFrame, live: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """К вселенной 3.2 добавляются пары с живыми событиями в окне (строки `live`), если оба
    государства существуют в этом году. Новые строки: границы нет, `revealed = 1`."""
    alive = {(g, y) for g, a, b in zip(states.gw, states.start_year, states.end_year) for y in range(a, b + 1)}
    new = live[["a", "b", "year"]].merge(grid[["a", "b", "year"]].assign(_in=1), on=["a", "b", "year"], how="left")
    new = new[new._in.isna() & (new.a != new.b)].drop(columns="_in")
    ok = np.array([(a, y) in alive and (b, y) in alive for a, b, y in zip(new.a, new.b, new.year)], dtype=bool)
    new = new[ok].assign(conttype=0, contig_any=0, contig_close=0, revealed=1)
    new["b_power"] = new.b.isin(dyads.POWERS).astype("int8")
    out = pd.concat([grid.assign(revealed=0), new], ignore_index=True)
    return out.astype({"conttype": "int8", "contig_any": "int8", "contig_close": "int8", "revealed": "int8"}).sort_values(["a", "b", "year"]).reset_index(drop=True)


def build_features(grid: pd.DataFrame, states: pd.DataFrame, opp: pd.DataFrame, dw: pd.DataFrame, events: pd.DataFrame,
                   unga: pd.DataFrame, mid: pd.DataFrame, cap: pd.DataFrame, event_lag: int = live_pairs.EVENT_LAG,
                   acd_lag: int = 0, **kw) -> pd.DataFrame:
    """Панель пар 3.2 на расширенной вселенной плюс `LIVE_FEATURES`. Чистая функция: строка
    (A, B, T) зависит от событий не позже месяца `декабрь T − event_lag` (живые стычки) и от
    того же, что в `dyads.build_features`. `acd_lag = 0`: список сторон за год T при прогнозе
    «на 1 января T+1» уже известен."""
    pe = live_pairs.pair_events(events, opp, acd_lag=acd_lag)
    live = live_pairs.annual_features(pe, range(int(grid.year.min()), int(grid.year.max()) + 1), lag=event_lag)
    d = dyads.build_features(extend_grid(grid, live, states), opp, dw, unga, mid, cap, **kw)
    d = d.merge(live, on=["a", "b", "year"], how="left")
    d[LIVE_FEATURES] = d[LIVE_FEATURES].fillna(0.0)
    d["near"] = d.contig_any.astype("int8")
    return d


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    contig, states = load_contiguity(), load_states()
    grid = dyads.build_grid(states, contig)
    opp = dyads.opponents(load_acd(dyads.ACD_FROM, extra=("incompatibility",)))
    from seldon.sources.ucdp import load_events
    events = load_events(candidate=False)
    dw = dyads.dyad_wars(load_ged(), opp, dyads.load_neighbor_set(contig))
    d = build_features(grid, states, opp, dw, events, load_unga(), load_mid(), dyads.load_capabilities())
    return d, dyads.external_wars(dw)


# ---------------------------------------------------------------------------
# Диадная модель: общая или раздельная
# ---------------------------------------------------------------------------


def _fit(train: pd.DataFrame, features: list[str], C: float | None):
    c = model_dyad.choose_C(train, features) if C is None else C
    return model_dyad.make_model(c).fit(train[features], train.y_next.astype(int)), c


def forecast_asof(d: pd.DataFrame, T: int, features: list[str] = LIVE_SET, split: bool = True, C: float | None = None,
                  return_fit: bool = False):
    """Прогноз начала войны пары в T+1. Обучение — пары под риском с меткой, известной к T.
    `split`: отдельные модели для `near` (есть граница) и `far`; группа, где в обучении меньше
    `MIN_GROUP_EVENTS` начал, берёт общую модель. Колонки теста: p_dyad, C, model (`near` /
    `far` / `pooled`). `return_fit` — ещё и обучение с прогнозом «на себе» (для калибровки
    `p_ext` внутри обучения) и словарь моделей."""
    d = d if "y_next" in d else model_dyad.with_next(d)
    train = d[label_known(d.year, T, HORIZON) & d.y_next.notna() & (d.dyad_war == 0)].copy()
    test = d[(d.year == T) & (d.dyad_war == 0)].copy()
    models: dict = {}
    if len(test) == 0 or len(train) == 0 or train.y_next.nunique() < 2:
        rate = float(train.y_next.mean()) if len(train) else np.nan
        for x in (train, test):
            x["p_dyad"], x["C"], x["model"] = rate, np.nan, "rate"
        return (test, train, models) if return_fit else test
    pooled, c0 = _fit(train, features, C)
    for x in (train, test):
        x["p_dyad"], x["C"], x["model"] = pooled.predict_proba(x[features])[:, 1], c0, "pooled"
    models["pooled"] = pooled
    if split:
        for name, flag in (("near", 1), ("far", 0)):
            tr = train[train.near == flag]
            if tr.y_next.sum() < MIN_GROUP_EVENTS or tr.y_next.nunique() < 2:
                continue
            m, c = _fit(tr, features, C)
            models[name] = m
            for x in (train, test):
                k = x.near == flag
                if k.any():
                    x.loc[k, "p_dyad"], x.loc[k, "C"], x.loc[k, "model"] = m.predict_proba(x.loc[k, features])[:, 1], c, name
    return (test, train, models) if return_fit else test


# ---------------------------------------------------------------------------
# Внешний канал: свёртка на страну и калибровка внутри обучения
# ---------------------------------------------------------------------------


def collapse(pairs: pd.DataFrame) -> pd.DataFrame:
    """Сырая свёртка по стране A и году строки: gw, year, s = 1 − ∏(1 − p_dyad), top_b."""
    g = pairs.assign(q=1 - pairs.p_dyad).groupby(["a", "year"])
    out = (1 - g.q.prod()).rename("s").reset_index()
    top = pairs.loc[pairs.groupby(["a", "year"]).p_dyad.idxmax(), ["a", "year", "b"]].rename(columns={"b": "top_b"})
    return out.merge(top, on=["a", "year"]).rename(columns={"a": "gw"})


def ext_onset_labels(country: pd.DataFrame, ext: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Страновая целевая внешнего канала на строке года t: страна без войны в t, а в t+1 —
    начало войны, где у неё есть государство-противник. gw, year, at_risk, ext_onset_next
    (NaN, если года t+1 в панели нет)."""
    c = country[["gw", "year", target]].copy()
    nxt = c.assign(year=c.year - 1).rename(columns={target: "war_next"})
    e = ext.assign(year=ext.year - 1).rename(columns={"ext_war": "ext_next"})
    c = c.merge(nxt, on=["gw", "year"], how="left").merge(e, on=["gw", "year"], how="left")
    c["at_risk"] = (c[target] == 0).astype("int8")
    c["ext_onset_next"] = ((c.war_next == 1) & (c.ext_next.fillna(0) == 1)).astype(float).where(c.war_next.notna())
    return c[["gw", "year", "at_risk", "ext_onset_next"]]


def external_asof(d: pd.DataFrame, labels: pd.DataFrame, T: int, features: list[str] = LIVE_SET, split: bool = True,
                  C: float | None = None, return_pairs: bool = False):
    """Внешний канал страны в точке T: gw, year (= T), s (сырая свёртка), p_ext (после
    калибровки), top_b. Калибровка — логистическая поправка от logit(s), обученная на странах
    под риском в годах t с t+1 ≤ T (диадные прогнозы тех лет — той же модели, что и прогноз)."""
    test, train, _ = forecast_asof(d, T, features, split, C, return_fit=True)
    now = collapse(test)
    past = collapse(train).merge(labels, on=["gw", "year"], how="inner")
    past = past[(past.at_risk == 1) & past.ext_onset_next.notna() & label_known(past.year, T, HORIZON)]
    now["p_ext"] = now.s
    if past.ext_onset_next.sum() >= CAL_MIN_EVENTS and past.ext_onset_next.nunique() == 2:
        cal = LogisticRegression(C=1000.0, max_iter=1000).fit(model_dyad._logit(past.s).reshape(-1, 1), past.ext_onset_next.astype(int))
        now["p_ext"] = cal.predict_proba(model_dyad._logit(now.s).reshape(-1, 1))[:, 1]
    return (now, test) if return_pairs else now


def run(d: pd.DataFrame, labels: pd.DataFrame, origins=TRAIN_YEARS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Вслепую по годам: (прогнозы пар всех вариантов, внешний канал стран для основного
    варианта и эталона, коэффициенты последних моделей)."""
    d = model_dyad.with_next(d)
    pairs, country = [], []
    for name, (feats, split) in VARIANTS.items():
        def step(T: int, feats=feats, split=split, name=name) -> pd.DataFrame:
            test = forecast_asof(d, T, feats, split)
            out = test[test.y_next.notna()][["a", "b", "near", "contig_close", "b_power", "revealed", "p_dyad"]].copy()
            out["onset"], out["year"] = test.loc[out.index, "y_next"].astype(int), T + 1
            return out.rename(columns={"p_dyad": name})
        p = backtest(origins, step)
        pairs.append(p if not pairs else p[["a", "b", "year", name]])
    pred = pairs[0]
    for p in pairs[1:]:
        pred = pred.merge(p, on=["a", "b", "year"], how="inner")
    base = backtest(origins, lambda T: model_dyad.forecast_asof(d, T)[["a", "b", "year", "p_base"]].assign(year=T + 1))
    pred = pred.merge(base, on=["a", "b", "year"], how="left")
    for name in COUNTRY_VARIANTS:
        feats, split = VARIANTS[name]
        c = pd.concat([external_asof(d, labels, T, feats, split).assign(year=T + 1) for T in [*origins, YEAR_TO]], ignore_index=True)
        country.append(c.rename(columns={"s": f"s__{name}", "p_ext": f"p_ext__{name}", "top_b": f"top_b__{name}"}))
    _, _, models = forecast_asof(d, YEAR_TO - 1, LIVE_SET, True, return_fit=True)
    coefs = pd.concat([pd.Series(m[-1].coef_[0], index=LIVE_SET, name=k) for k, m in models.items()], axis=1)
    cext = country[0]
    for c in country[1:]:
        cext = cext.merge(c, on=["gw", "year"], how="outer")
    return pred, cext, coefs


SAMPLES = {
    "все пары": lambda p: p.year > 0,
    "соседи": lambda p: p.near == 1,
    "без общей границы": lambda p: p.near == 0,
    "после 2014": lambda p: p.year >= 2015,
    "после 2014, соседи": lambda p: (p.year >= 2015) & (p.near == 1),
}


def summarize(pred: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    cols = {**{n: n for n in VARIANTS}, "base": "p_base"}
    out = []
    for name, mask in SAMPLES.items():
        x = pred[mask(pred)]
        if x.onset.sum() < 3:
            continue
        out.append(paired_comparison(x, cols, "mid_pooled", y="onset", n_boot=n_boot).assign(sample=name))
    return pd.concat(out, ignore_index=True)


def cases_table(pred: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, a, b, year in CASES:
        row = {"case": label, "a": a, "b": b, "year": year}
        x = pred[(pred.a == a) & (pred.b == b) & (pred.year == year)]
        prev = d[(d.a == a) & (d.b == b) & (d.year == year - 1)]
        if len(x):
            peers, r = pred[pred.year == year], x.iloc[0]
            row |= {"status": "начало войны" if r.onset == 1 else "под риском, войны пары по UCDP не было", "n_pairs": len(peers)}
            for n in ("mid_pooled", "live_pooled", MAIN):
                row[f"rank_{n}"], row[f"p_{n}"] = int((peers[n] > r[n]).sum()) + 1, r[n]
            row |= {k: float(prev[k].iloc[0]) for k in ("mid_force", *LIVE_FEATURES)}
        elif len(prev) == 0:
            row["status"] = "пары нет во вселенной"
        else:
            row["status"] = "пара уже воевала в предыдущем году — не начало"
        rows.append(row)
    return pd.DataFrame(rows)


def ext_channel_table(c: pd.DataFrame) -> pd.DataFrame:
    """Внешний канал как самостоятельный прогноз: страны под риском, целевая — начало войны с
    государством-противником. calib = средняя вероятность / фактическая частота."""
    rows = []
    for sample, x in (("1996–2025", c), ("после 2014", c[c.year >= 2015])):
        for col in [k for k in c if k.startswith(("s__", "p_ext__", "p_ext_32"))]:
            v = x[x[col].notna()]
            rows.append({"sample": sample, "model": col, "n": len(v), "ext_onsets": int(v.ext_onset.sum()),
                         "auc": auc(v.ext_onset, v[col]), "mean_p": v[col].mean(), "calib": v[col].mean() / v.ext_onset.mean()})
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    d, ext = build()
    d.to_parquet(DATA_PROCESSED / "dyad_live_features.parquet", index=False)
    names = pd.read_parquet(DATA_PROCESSED / "panel_country_year.parquet", columns=["gw", "country"]).drop_duplicates("gw").set_index("gw").country
    country_panel = pd.read_parquet(DATA_PROCESSED / "panel_country_year.parquet", columns=["gw", "year", TARGET])
    labels = ext_onset_labels(country_panel, ext)

    pred, cext, coefs = run(d, labels)
    pred.to_parquet(DATA_PROCESSED / "dyad_live_predictions.parquet", index=False)
    summary = summarize(pred)
    summary.to_csv(REPORTS / "dyad_live_metrics.csv", index=False, float_format="%.5f")
    coefs.round(3).to_csv(REPORTS / "dyad_live_coefficients.csv")
    cases = cases_table(pred, d)
    cases.to_csv(REPORTS / "dyad_live_cases.csv", index=False, float_format="%.5f")

    last = d[d.year == YEAR_TO]
    print(f"диадная панель: {len(d)} строк; в {YEAR_TO} году пар {len(last)}: соседей {int(last.near.sum())}, держава без границы "
          f"{int(((last.near == 0) & (last.revealed == 0)).sum())}, добавлено по проявленной вражде {int(last.revealed.sum())}")
    print(f"живые стычки: у {(d.live_years15 > 0).mean():.1%} строк есть события за 15 лет; после 2014: {(d[d.year >= 2015].live_years15 > 0).mean():.1%}")
    for sample, g in summary.groupby("sample", sort=False):
        print(f"{sample}: {int(g.n.iloc[0])} пара-лет, начал {int(g.positives.iloc[0])}")
        for r in g.itertuples():
            s = f"  {r.model:12s} AUC {r.auc:.3f} [{r.auc_lo:.3f}; {r.auc_hi:.3f}], внутри года {r.auc_within_year:.3f}, calib {r.calib:.2f}"
            if pd.notna(getattr(r, "d_auc", np.nan)):
                s += f" | ΔAUC к mid_pooled {r.d_auc:+.3f} [{r.d_auc_lo:+.3f}; {r.d_auc_hi:+.3f}]"
            print(s)
    print("\nКоэффициенты (стандартизованные; модели обучены по 2024):")
    print(coefs.round(3).to_string())
    pd.set_option("display.width", 250)
    print("\nКонтрольные пары (место среди пар мира под риском):")
    print(cases.drop(columns=["a", "b"]).round(3).to_string(index=False))

    # --- страновой уровень ---
    pitf_col = f"pitf_{TARGET}"
    base = pd.read_parquet(DATA_PROCESSED / f"{pitf_col}_predictions.parquet")
    f = pd.read_parquet(DATA_PROCESSED / "features_pitf_country_year.parquet")
    p_int = model_dyad.internal_channel(f, ext, model_hist.FEATURE_SETS["pitf"][0])
    onset = model_hist.ONSET_OF[TARGET]
    c = base[base.at_risk == 1][["gw", "year", onset, pitf_col]].rename(columns={onset: "onset", pitf_col: "p_pitf"})
    c = c.merge(cext, on=["gw", "year"], how="left").merge(p_int, on=["gw", "year"], how="left").merge(ext, on=["gw", "year"], how="left")
    c["ext_onset"] = ((c.onset == 1) & (c.ext_war.fillna(0) == 1)).astype(int)
    old = pd.read_parquet(DATA_PROCESSED / "dyad_country_predictions.parquet")[["gw", "year", "p_ext"]].rename(columns={"p_ext": "p_ext_32"})
    c = c.merge(old, on=["gw", "year"], how="left")
    pcols = [k for k in c if k.startswith(("s__", "p_ext"))]
    c[pcols] = c[pcols].fillna(0.0)
    c["p_int"] = c.p_int.fillna(c.p_pitf)
    c["dy_live_raw"] = 1 - (1 - c.p_int) * (1 - c[f"s__{MAIN}"])
    for dyad_name, v in COUNTRY_VARIANTS.items():
        c[v] = 1 - (1 - c.p_int) * (1 - c[f"p_ext__{dyad_name}"])
    for v in ("dy_live_raw", *COUNTRY_VARIANTS.values()):
        col = f"{v}_{TARGET}"
        out = base[["gw", "country", "year", "at_risk", "years_since_war", TARGET, onset, pitf_col]].merge(c[["gw", "year", v]], on=["gw", "year"], how="left")
        out[col] = out[v].fillna(out[pitf_col])  # воюющие страны — прогноз `pitf` как есть
        out.drop(columns=[v, pitf_col]).to_parquet(DATA_PROCESSED / f"{col}_predictions.parquet", index=False)
    et = ext_channel_table(c)
    et.to_csv(REPORTS / "dyad_live_ext_channel.csv", index=False, float_format="%.5f")
    print("\nВнешний канал отдельно (страны под риском; целевая — начало войны с государством-противником):")
    print(et.round(4).to_string(index=False))

    nxt = cext[cext.year == YEAR_TO + 1].sort_values(f"p_ext__{MAIN}", ascending=False).head(15)
    top = pd.DataFrame({"country": nxt.gw.map(names), "p_ext": nxt[f"p_ext__{MAIN}"], "s_raw": nxt[f"s__{MAIN}"],
                        "top_b": nxt[f"top_b__{MAIN}"].map(names)})
    top.to_csv(REPORTS / f"dyad_live_top_{YEAR_TO + 1}.csv", index=False, float_format="%.4f")
    cext.assign(country=cext.gw.map(names)).to_parquet(DATA_PROCESSED / "dyad_live_country.parquet", index=False)
    print(f"\nВнешний канал на {YEAR_TO + 1} год, топ-15 стран (включая уже воюющие):")
    print(top.round(3).to_string(index=False))
    return 0
