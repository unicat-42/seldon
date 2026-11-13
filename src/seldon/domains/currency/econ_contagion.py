"""Валютное заражение по торговым связям и параллельный курс (чекпоинт 4.1): проверка
вслепую в месячном контуре `deval15` на горизонтах 3, 6 и 12 месяцев.

Запуск:
    uv run python -m seldon econ_contagion [3] [6] [12]
        без аргументов — все три горизонта.

Признаки — `contagion_features.py` (острый курсовой шок торговых партнёров на окнах 3, 6 и
12 месяцев, его взаимодействие с привязкой и сырьевым шоком, премия параллельного курса) и
`struct_features.py` (чекпоинт 3.3). Боевая модель не меняется; наборы:
    ref         боевая логрегрессия `econ_kr`: индикаторы KLR, C = 0,3;
    base        тот же набор, C выбирается внутри обучения;
    struct      + структурный набор чекпоинта 3.3 как есть;
    struct_gdp  struct + ln(ВВП на душу, доллары) с лагом T−2 — проверка гипотезы, что флаг
                `gg_missing` (нет данных ВБ о краткосрочном долге) — скрытый признак богатой
                страны: база долга ВБ (IDS) не включает страны с высоким доходом;
    clean       struct без флага `gg_missing`: пропуск `gg_log` заполняется медианой обучения
                (не поощряется и не штрафуется), доход учтён явно через ВВП на душу;
    contagion   ref + заражение по торговым связям;
    full        clean + заражение;
    par         ref + премия параллельного курса (покрытие — см. `sources/parallel.py`);
    level       ref + накопленный шок партнёров (уровень от нижней точки за 24 месяца);
    contagion_level   contagion + накопленный шок;
    struct_level      struct + накопленный шок.
Для горизонта h метка — `deval15` в месяцы m+1…m+h; правило «тишины» то же, что у боевой
модели. Обучение — метки, известные к концу декабря T−1 (m + h ≤ декабрь T−1), тест — все
месяцы года T, T = 1996…2025; C — по времени внутри обучения (`econ_struct.choose_C_kr`).
Прогноз пишется для всех строк со своей валютой и курсом; в метрики идут только `evaluable`.

Выход:
    data/processed/fx_contagion_monthly.parquet            таблица признаков
    data/processed/econ_contagion_predictions_h{h}.parquet прогнозы по наборам
    reports/fx_contagion_coverage.csv                      покрытие по постсоветским странам
    reports/fx_contagion_coefficients.csv                  коэффициенты наборов `full` и `struct_gdp`
    reports/fx_contagion_univariate.csv                    AUC каждого нового признака в одиночку
    reports/fx_contagion_gg_missing.csv                    кто попадает под флаг `gg_missing`
Парное сравнение и кейсы — команда `compare_contagion`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, FLOAT_FORMAT, REPORTS
from seldon.core.entities import ISO3_TO_SHORT, POST_SOVIET_ISO3
from seldon.core.metrics import auc, auc_within
from seldon.core.temporal import label_known, last_available_year_in_month
from seldon.domains.currency import contagion_features as cf
from seldon.domains.currency import econ_kr, struct_features
from seldon.domains.currency.econ_struct import KR_C_REF, KR_MIN_POS, KR_TARGET, _kr_design, _kr_fit, choose_C_kr
from seldon.domains.currency.kr_features import INDICATORS, load_war
from seldon.domains.currency.kr_targets import add_target
from seldon.evaluation.backtest import backtest
from seldon.sources.wdi import load_annual_cpi

HORIZONS = (3, 6, 12)
S_CONT, S_FLAGS = struct_features.SETS["struct"]
GDP = ["gdp_pc_log", "gdp_pc_missing"]
CLEAN = [*[c for c in S_CONT if c != "gg_log"], "gg_log_nan", *[f for f in S_FLAGS if f != "gg_missing"], *GDP]
SETS: dict[str, list[str]] = {
    "ref": [],
    "base": [],
    "struct": [*S_CONT, *S_FLAGS],
    "struct_gdp": [*S_CONT, *S_FLAGS, *GDP],
    "clean": CLEAN,
    "contagion": [*cf.CONT, *cf.FLAGS],
    "full": [*CLEAN, *cf.CONT, *cf.FLAGS],
    "par": [*cf.PAR_CONT, *cf.PAR_FLAGS],
    # накопленный шок (уровень от нижней точки за 24 месяца) — развилка 3 отчёта 4.1
    "level": [*cf.LEVEL, *cf.FLAGS],
    "contagion_level": [*cf.CONT, *cf.LEVEL, *cf.FLAGS],
    "struct_level": [*S_CONT, *S_FLAGS, *cf.LEVEL, *cf.FLAGS],
}
NAMES = list(SETS)


def gdp_per_capita(gmd: pd.DataFrame) -> pd.DataFrame:
    """ВВП на душу в долларах по годам (GMD): iso3, year, gdp_pc."""
    g = gmd[["ISO3", "year", "nGDP_USD", "pop"]].rename(columns={"ISO3": "iso3"})
    g = g[(g.nGDP_USD > 0) & (g["pop"] > 0)]
    return g.assign(year=g.year.astype(int), gdp_pc=g.nGDP_USD / g["pop"])[["iso3", "year", "gdp_pc"]]


def add_income(p: pd.DataFrame, gdp_pc: pd.DataFrame, source: str | None = "gdp_pc_level") -> pd.DataFrame:
    """`gdp_pc_log` — ln ВВП на душу за последний «вышедший» год (`gdp_pc_level`: T−2);
    `gg_log_nan` — `gg_log` с пропуском вместо нуля там, где данных ВБ о долге нет."""
    p = p.copy()
    p["year"] = p.month.dt.year if source is None else last_available_year_in_month(source, p.month)
    p = p.merge(gdp_pc, on=["iso3", "year"], how="left").drop(columns="year")
    p["gdp_pc_missing"] = p.gdp_pc.isna().astype(int)
    p["gdp_pc_log"] = np.log(p.gdp_pc).fillna(0.0)
    p["gg_log_nan"] = p.gg_log.where(p.gg_missing == 0)
    return p.drop(columns="gdp_pc")


def build_panel(struct: pd.DataFrame, contagion: pd.DataFrame, gdp_pc: pd.DataFrame, raw=None, annual_cpi=None, war=None) -> pd.DataFrame:
    """Месячная панель `econ_kr` (целевая `deval15`) + структурные признаки + заражение + доход."""
    p = econ_kr.build_panel(econ_kr.load_panel() if raw is None else raw, load_annual_cpi() if annual_cpi is None else annual_cpi,
                            load_war() if war is None else war, target=KR_TARGET)
    p = struct_features.attach(p, struct, ["iso3", "month"])
    return add_income(cf.attach(p, contagion), gdp_pc)


def with_horizon(p: pd.DataFrame, h: int) -> pd.DataFrame:
    """Та же панель с меткой на горизонте h месяцев (`y`, `evaluable` пересчитаны)."""
    return add_target(p, KR_TARGET, h=h)


def _predict(p: pd.DataFrame, as_of: pd.Period, test: pd.DataFrame, name: str, h: int) -> tuple[np.ndarray, float] | None:
    feats = [*INDICATORS, *SETS[name]]
    known = p[label_known(p.month, as_of, h) & p.evaluable]
    if known.y.sum() < KR_MIN_POS:
        return None
    C = KR_C_REF if name == "ref" else choose_C_kr(known, feats, h=h)
    m, lo, hi, med = _kr_fit(known, feats, C)
    return m.predict_proba(_kr_design(test, feats, lo, hi, med))[:, 1], C


def forecast_asof(p: pd.DataFrame, as_of: pd.Period, name: str = "full", h: int = 12) -> pd.DataFrame:
    """Прогноз в точке `as_of` набором `name` на горизонте h: всё — по данным ≤ as_of.
    `p` — панель `with_horizon(build_panel(...), h)`."""
    test = p[(p.month == as_of) & p.own_currency & p.fx.notna()].copy()
    out = _predict(p, as_of, test, name, h)
    test["logit"], test["C"] = (np.nan, np.nan) if out is None else out
    return test


def run(p: pd.DataFrame, h: int, names: list[str] = NAMES) -> pd.DataFrame:
    def step(T: int) -> pd.DataFrame | None:
        as_of = pd.Period(f"{T - 1}-12", freq="M")
        test = p[(p.month.dt.year == T) & p.own_currency & p.fx.notna()].copy()
        if not test.evaluable.any():
            return None
        for name in names:
            out = _predict(p, as_of, test, name, h)
            if out is None:
                return None
            test[f"logit__{name}"], test[f"C__{name}"] = out
        print(f"  h={h} {T}: строк {int(test.evaluable.sum())}, кризисных окон {int(test.y[test.evaluable].sum())}")
        return test

    return backtest(range(econ_kr.FIRST_EVAL_YEAR, econ_kr.LAST_EVAL_YEAR + 1), step)


# ---------------------------------------------------------------------------


def coefficients(p: pd.DataFrame, h: int, names=("full", "struct", "struct_gdp", "level", "struct_level")) -> pd.DataFrame:
    """Стандартизованные коэффициенты в последней точке обучения."""
    as_of = pd.Period(f"{econ_kr.LAST_EVAL_YEAR - 1}-12", freq="M")
    known = p[label_known(p.month, as_of, h) & p.evaluable]
    out = []
    for name in names:
        feats = [*INDICATORS, *SETS[name]]
        m, *_ = _kr_fit(known, feats, choose_C_kr(known, feats, h=h))
        out.append(pd.Series(m[-1].coef_[0], index=feats, name=f"{name}_h{h}"))
    return pd.concat(out, axis=1)


def univariate(p: pd.DataFrame, h: int) -> pd.DataFrame:
    """AUC каждого нового признака в одиночку на оцениваемых строках 1996–2025 (без обучения:
    сам признак как оценка риска) — мир, постсоветские, привязки. Для параллельной премии —
    только строки, где она есть."""
    d = p[p.evaluable & (p.month.dt.year >= econ_kr.FIRST_EVAL_YEAR)]
    rows = []
    for f in [*cf.CONT, *cf.LEVEL, *cf.PAR_CONT, "reer_peg", "peg_share12"]:
        flag = "par_missing" if f in cf.PAR_CONT else ("reer_missing" if f == "reer_peg" else "cont_missing")
        x = d[d[flag] == 0] if flag in d else d
        for sample, mask in (("мир", x.iso3.notna()), ("постсоветские", x.iso3.isin(POST_SOVIET_ISO3)), ("привязки", x.peg_share12 >= 0.8)):
            s = x[mask]
            if s.y.nunique() < 2:
                continue
            rows.append({"h": h, "feature": f, "sample": sample, "n": len(s), "positives": int(s.y.sum()),
                         "auc": auc(s.y, s[f]), "auc_within_year": auc_within(s.y, s[f], s.month.dt.year)})
    return pd.DataFrame(rows)


def coverage(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iso, short in ISO3_TO_SHORT.items():
        g = table[table.iso3 == iso]
        r = {"country": short, "iso3": iso}
        for name, flag in (("contagion", "cont_missing"), ("parallel", "par_missing")):
            have = g[g[flag] == 0]
            r[f"{name}_from"], r[f"{name}_to"] = (str(have.month.min()), str(have.month.max())) if len(have) else ("", "")
        rows.append(r)
    return pd.DataFrame(rows)


def gg_missing_profile(p: pd.DataFrame) -> pd.DataFrame:
    """Кто попадает под `gg_missing`: доля строк, средний ВВП на душу и частота кризисных окон —
    по группам дохода (терцили ВВП на душу в своём году), на оцениваемых строках 1996–2025."""
    d = p[p.evaluable & (p.month.dt.year >= econ_kr.FIRST_EVAL_YEAR) & (p.gdp_pc_missing == 0)].copy()
    d["income"] = d.groupby(d.month.dt.year).gdp_pc_log.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["бедные", "средние", "богатые"]))
    g = d.groupby(["income", "gg_missing"], observed=True).agg(n=("y", "size"), crisis_rate=("y", "mean"), gdp_pc=("gdp_pc_log", lambda s: float(np.exp(s.mean()))))
    g["share_in_income_group"] = g.n / g.groupby(level=0).n.transform("sum")
    return g.reset_index()


def main(argv: list[str]) -> int:
    from seldon.sources.gmd import load_gmd

    horizons = [int(a) for a in argv if a.isdigit()] or list(HORIZONS)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    print("Признаки: структурные (3.3) и заражение по торговым связям…")
    struct = struct_features.load_struct()
    table = cf.load_contagion(struct, struct_features.LAST_MONTH)
    table.assign(month=table.month.astype(str)).to_parquet(DATA_PROCESSED / "fx_contagion_monthly.parquet", index=False)
    d = table[table.month >= "1996-01"]
    print(f"  {table.iso3.nunique()} стран; доля строк с 1996 года: заражение {1 - d.cont_missing.mean():.0%}, параллельный курс {1 - d.par_missing.mean():.1%}")
    cov = coverage(table)
    cov.to_csv(REPORTS / "fx_contagion_coverage.csv", index=False)
    print(cov.to_string(index=False))
    base = build_panel(struct, table, gdp_per_capita(load_gmd()))
    coefs, uni = [], []
    for h in horizons:
        print(f"\nГоризонт {h} мес, вслепую…")
        p = with_horizon(base, h)
        pred = run(p, h)
        pred.assign(month=pred.month.astype(str)).to_parquet(DATA_PROCESSED / f"econ_contagion_predictions_h{h}.parquet", index=False)
        coefs.append(coefficients(p, h))
        uni.append(univariate(p, h))
    if len(horizons) == len(HORIZONS):
        c = pd.concat(coefs, axis=1).rename_axis("feature").reset_index()
        c.to_csv(REPORTS / "fx_contagion_coefficients.csv", index=False, float_format=FLOAT_FORMAT)
        u = pd.concat(uni, ignore_index=True)
        u.to_csv(REPORTS / "fx_contagion_univariate.csv", index=False, float_format="%.4f")
        gg = gg_missing_profile(with_horizon(base, 12))
        gg.to_csv(REPORTS / "fx_contagion_gg_missing.csv", index=False, float_format="%.4f")
        pd.set_option("display.width", 250)
        print("\nКто попадает под gg_missing (h = 12):")
        print(gg.round(3).to_string(index=False))
        new = c[c.feature.isin([*cf.COLUMNS, *S_CONT, *S_FLAGS, *GDP, "gg_log_nan"])]
        print("\nКоэффициенты (стандартизованные, последняя точка обучения):")
        print(new.round(3).to_string(index=False))
        print("\nAUC признаков в одиночку:")
        print(u.pivot_table(index=["feature", "sample"], columns="h", values="auc").round(3).to_string())
    return 0
