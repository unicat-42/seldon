"""[research] Уровень демократии как признак месячной валютной модели — и не знание ли это
задним числом (продолжение исследования `party_study`, развилка Р1, 14.11.2026).

Запуск:
    uv run python -m seldon democ_fx
        ≈ 30 минут. Нужны готовые таблицы `make all` и `make variants` (`econ_struct`,
        `econ_contagion`) и `download vdem_vintages` (прошлые релизы V-Dem, ≈ 175 МБ).

Боевой конвейер, витрина и снимки прогнозов не затрагиваются.

Вопрос. В `party_study` одна переменная V-Dem (персонализм режима) улучшала месячную валютную
модель, но она размечена задним числом и кончается 2023 годом. Живой заменитель — индекс
электоральной демократии `v2x_polyarchy` (0…1, есть по 2025 год). Даёт ли он прирост поверх
боевой `econ_kr` и претендента `live_fx_swap`, и держится ли прирост, если брать индекс в той
редакции, которая существовала на дату прогноза?

Редакции. V-Dem каждый год пересчитывает всю историю. Релиз года R выходит в марте (v11.1 —
1 апреля) и несёт данные по R − 1; в проекте год Y считается известным с 1 апреля Y+1
(`core/temporal.py`, `vdem_annual`). Поэтому при прогнозе на месяцы года T (обучение — на
конец декабря T − 1):
    обучающие строки и январь–март года T   релиз года T − 1 (вся история — в его редакции);
    апрель–декабрь года T                   релиз года T (он вышел в марте T).
Сохранённые релизы — v10 (2020) … v16 (2026). Для T ≤ 2020 более ранних редакций нет, берётся
v10: это не «реальное время», но хотя бы редакция, не знающая ничего после 2019 года. Честное
реальное время — прогнозы на 2021–2025 годы (`REALTIME_FROM`).
Режим `latest` — всё из сегодняшнего v16 (так считался `party_study`); сравнение двух режимов и
показывает, сколько прироста — знание задним числом.

Наборы (к признакам эталона добавляется):
    flag        только флаг «страны нет в V-Dem» — контроль артефакта покрытия;
    democ       индекс демократии + флаг;
    anoc        демократия + «анократия» 4·p·(1 − p) (риск выше всего посередине) + флаг;
    gdp         ln ВВП на душу с лагом T − 2 — контроль «это просто бедность»;
    gdp_democ   демократия поверх ВВП на душу (сравнивается и с эталоном, и с `gdp`).
Эталоны: `klr` — боевой набор `econ_kr` с подобранной внутри обучения C; `swap` — претендент
`live_fx_swap` как есть. Целевая `deval15`, горизонты 12 и 3 месяца, T = 1996…2025.

Критерий, объявленный до расчёта: демократия — кандидат в претендента, только если на
постсоветских (а) `gdp_democ` значимо лучше `gdp` и (б) на релизах своего времени знак и
величина прироста не разваливаются.

Выход:
    data/processed/democ_fx_predictions_<режим>_h<h>.parquet
    reports/democ_fx_comparison.csv     парные сравнения по режимам, выборкам и горизонтам
    reports/democ_fx_coefficients.csv   коэффициент демократии по годам обучения
    reports/democ_fx_revisions.csv      насколько релизы расходятся с сегодняшним
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, REPORTS
from seldon.core.entities import POST_SOVIET_ISO3
from seldon.core.temporal import last_available_year_in_month
from seldon.research.party_study import FMT, by_iso3, compare, fx_month_predict, show

FIRST_RELEASE, LAST_RELEASE = 2020, 2026   # v10 … v16
REALTIME_FROM = FIRST_RELEASE + 1          # первый год прогноза, где обучение — на релизе своего времени
HORIZONS = (12, 3)
SETS: dict[str, list[str]] = {
    "flag": ["democ_missing"],
    "democ": ["democ", "democ_missing"],
    "anoc": ["democ", "anoc", "democ_missing"],
    "gdp": ["gdp_pc_log", "gdp_pc_missing"],
    "gdp_democ": ["gdp_pc_log", "gdp_pc_missing", "democ", "democ_missing"],
}
PEG_SHARE = 0.8


def release_year(T: int, late: bool) -> int:
    """Год релиза V-Dem, известного при прогнозе на год T: `late=False` — обучение и
    январь–март (релиз года T − 1), `late=True` — апрель–декабрь (релиз года T). Вне
    сохранённых релизов — ближайший."""
    return int(min(max(T if late else T - 1, FIRST_RELEASE), LAST_RELEASE))


def attach_democ(p: pd.DataFrame, tables: dict[int, pd.DataFrame], T: int, vintage: bool = True) -> pd.DataFrame:
    """Индекс демократии на месячную панель (iso3, month) для прогноза на год T. Месяц видит
    последний год, вышедший к его концу (`vdem_annual`), в редакции своего релиза (см. docstring
    модуля); `vintage=False` — всё из последнего релиза. Колонки: `democ` (пропуск — медиана
    месяца), `anoc`, `democ_missing`. `tables` — год релиза → таблица (iso3, year, democ)."""
    out = p.copy()
    out["_y"] = last_available_year_in_month("vdem_annual", out.month)
    early, late = (release_year(T, False), release_year(T, True)) if vintage else (max(tables), max(tables))
    out = out.merge(tables[early].rename(columns={"year": "_y", "democ": "_a"}), on=["iso3", "_y"], how="left")
    out = out.merge(tables[late].rename(columns={"year": "_y", "democ": "_b"}), on=["iso3", "_y"], how="left")
    is_late = (out.month.dt.year == T) & (out.month.dt.month >= 4)
    raw = out._b.where(is_late, out._a)
    out["democ_missing"] = raw.isna().astype(int)
    out["democ"] = raw.fillna(raw.groupby(out.month).transform("median")).fillna(0.5)
    out["anoc"] = 4 * out.democ * (1 - out.democ)
    out.index = p.index
    return out.drop(columns=["_y", "_a", "_b"])


def default_families() -> dict[str, list[str]]:
    from seldon.domains.currency import econ_csd
    from seldon.domains.currency.kr_features import INDICATORS

    return {"klr": list(INDICATORS), "swap": econ_csd.features(econ_csd.CHALLENGER)}


def predict_year(p: pd.DataFrame, tables: dict[int, pd.DataFrame], T: int, h: int, vintage: bool = True,
                 families: dict[str, list[str]] | None = None, coefs: list | None = None) -> pd.DataFrame | None:
    """Прогнозы на все месяцы года T: обучение — окна, закрывшиеся к декабрю T − 1. `p` — панель
    с меткой `y` на горизонте h. Колонки `<эталон>__ref` и `<эталон>__<набор>`."""
    families = default_families() if families is None else families
    q = attach_democ(p, tables, T, vintage)
    as_of = pd.Period(f"{T - 1}-12", freq="M")
    test = q[(q.month.dt.year == T) & q.own_currency & q.fx.notna()]
    if test.empty or not test.evaluable.any():
        return None
    keep = [c for c in ("iso3", "month", "post_soviet", "y", "evaluable", "peg_share12") if c in test]
    out = test[keep].copy()
    for fam, feats in families.items():
        for name, extra in {"ref": [], **SETS}.items():
            r = fx_month_predict(q, as_of, test, [*feats, *extra], h)
            if r is None:
                return None
            out[f"{fam}__{name}"] = r[0]
            if coefs is not None and "democ" in extra:
                coefs.append({"family": fam, "set": name, "h": h, "vintage": vintage, "year": T, "coef_democ": r[1]["democ"],
                              "coef_anoc": r[1].get("anoc", np.nan)})
    return out


def load_tables() -> dict[int, pd.DataFrame]:
    from seldon.sources.vdem import load_vdem

    out = {}
    for r in range(FIRST_RELEASE, LAST_RELEASE + 1):
        v = load_vdem(1980, ["v2x_polyarchy"], release_year=r).rename(columns={"v2x_polyarchy": "democ"})
        out[r] = by_iso3(v)[["iso3", "year", "democ"]]
    return out


def revisions(tables: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Насколько каждая прошлая редакция расходится с сегодняшней на общих страна-годах 1990+."""
    last = tables[LAST_RELEASE].set_index(["iso3", "year"]).democ
    rows = []
    for r, t in tables.items():
        if r == LAST_RELEASE:
            continue
        d = pd.concat([t.set_index(["iso3", "year"]).democ.rename("old"), last.rename("new")], axis=1).dropna()
        d = d[d.index.get_level_values("year") >= 1990]
        ps = d[d.index.get_level_values("iso3").isin(POST_SOVIET_ISO3)]
        newest = d[d.index.get_level_values("year") == r - 1]
        rows.append({"release_year": r, "n": len(d), "mean_abs_diff": (d.old - d.new).abs().mean(), "max_abs_diff": (d.old - d.new).abs().max(),
                     "corr": d.old.corr(d.new), "mean_abs_diff_post_soviet": (ps.old - ps.new).abs().mean(),
                     "mean_abs_diff_newest_year": (newest.old - newest.new).abs().mean()})
    return pd.DataFrame(rows)


def samples() -> dict:
    ps = lambda d: d.iso3.isin(POST_SOVIET_ISO3)  # noqa: E731
    return {
        "мир": lambda d: d.iso3.notna(),
        "постсоветские": ps,
        "привязки": lambda d: d.peg_share12 >= PEG_SHARE,
        "мир, до 2010": lambda d: d.year < 2010,
        "мир, с 2010": lambda d: d.year >= 2010,
        "постсоветские, до 2010": lambda d: ps(d) & (d.year < 2010),
        "постсоветские, с 2010": lambda d: ps(d) & (d.year >= 2010),
        f"мир, {REALTIME_FROM}–2025": lambda d: d.year >= REALTIME_FROM,
        f"постсоветские, {REALTIME_FROM}–2025": lambda d: ps(d) & (d.year >= REALTIME_FROM),
    }


def main(argv: list[str]) -> int:
    from seldon.domains.currency import econ_contagion, econ_csd, econ_kr

    tables = load_tables()
    rev = revisions(tables)
    rev.to_csv(REPORTS / "democ_fx_revisions.csv", index=False, float_format=FMT)
    print("Расхождение прошлых редакций индекса демократии с сегодняшней (v16):")
    print(rev.round(4).to_string(index=False))
    base = econ_csd.load_full_panel()
    parts, coefs = [], []
    titles = {"klr": "econ_kr (C подобрана)", "swap": "претендент live_fx_swap"}
    for h in HORIZONS:
        p = econ_contagion.with_horizon(base, h)
        for vintage in (True, False):
            mode = "релизы своего времени" if vintage else "сегодняшний v16"
            frames = [predict_year(p, tables, T, h, vintage, coefs=coefs) for T in range(econ_kr.FIRST_EVAL_YEAR, econ_kr.LAST_EVAL_YEAR + 1)]
            pred = pd.concat([f for f in frames if f is not None], ignore_index=True)
            pred.assign(month=pred.month.astype(str)).to_parquet(
                DATA_PROCESSED / f"democ_fx_predictions_{'vintage' if vintage else 'latest'}_h{h}.parquet", index=False)
            d = pred[pred.evaluable].assign(year=lambda x: x.month.dt.year)
            for fam, title in titles.items():
                tag = {"contour": "валюта месячная", "horizon": h, "edition": mode}
                parts.append(compare(d, f"{fam}__ref", {n: f"{fam}__{n}" for n in SETS}, "y", samples(), model=title, **tag))
                parts.append(compare(d, f"{fam}__gdp", {"gdp_democ": f"{fam}__gdp_democ"}, "y", samples(), model=f"{title} + ВВП на душу", **tag))
            print(f"  h={h}, {mode}: готово")
    t = pd.concat(parts, ignore_index=True)
    lead = ["horizon", "edition", "model", "set", "sample"]
    t = t[lead + [c for c in t.columns if c not in lead]]
    t.to_csv(REPORTS / "democ_fx_comparison.csv", index=False, float_format=FMT)
    pd.DataFrame(coefs).to_csv(REPORTS / "democ_fx_coefficients.csv", index=False, float_format=FMT)
    key = t[t.set.isin(["democ", "gdp_democ"]) & t["sample"].isin(["мир", "постсоветские", f"постсоветские, {REALTIME_FROM}–2025"])]
    for (h, mode), g in key.groupby(["horizon", "edition"], sort=False):
        print(f"\nГоризонт {h} мес, {mode}:")
        print(show(g.drop(columns=["horizon"])))
    return 0
