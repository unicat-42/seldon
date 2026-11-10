"""Поведение людей в действиях: вклады и бегство в валюту в России против давления на рубль.

Запуск:
    uv run python -m seldon.behavior_ru

Газеты (news_ru.py) оказались термометром. Проверяем действия:
  1. Вклады физлиц по валютам, помесячно (ЦБ РФ, `download cbr`): 2007–2018 из годовых
     HTML-таблиц, с 2012 из funds_all.xlsx. Считаем:
        rub_dep_g     рост рублёвых вкладов за месяц, % (отток = набег на банки)
        fx_dep_usd_g  рост валютных вкладов в долларовом выражении за месяц, % — только он
                      значит, что люди несут валюту в банк; в рублёвой оценке валютные
                      вклады растут сами собой при падении рубля
        fx_share      доля валютных вкладов, % (для справки; механически растёт с курсом)
  2. Платёжный баланс (МВФ, `download imf`, поток `bop_cash_deposits`), поквартально
     с 1994: «наличная валюта и депозиты» домохозяйств и нефинансовых компаний, чистое
     приобретение, млрд долл. — покупка наличной валюты и вывод на счета за рубеж.
  3. EMP рубля из econ_kr (в σ) и курс на конец месяца.

Окна −12…+3 мес вокруг 2008-10, 2014-10, 2018-04 (контроль) и корреляции с опережением.
Google Trends («курс доллара», 2004+) — отдельно, CSV скачивает Ангелина руками.

Выход:
    data/processed/behavior_ru_monthly.parquet
    reports/behavior_ru_monthly.csv, behavior_ru_episodes.csv, behavior_ru_leadlag.csv,
    behavior_ru_bop_quarterly.csv
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, DATA_RAW, PROJECT_ROOT

REPORTS = PROJECT_ROOT / "reports"
CBR = DATA_RAW / "cbr"
EPISODES = {"2008-10": "мировой кризис; EMP-кризис 2008-10, LV банковский 2008",
            "2014-10": "Крым/санкции/нефть; EMP-кризис 2014-10, LV валютный 2014",
            "2018-04": "контроль: санкции, рубль −10 %, кризисом не стал"}
WINDOW = (-12, 3)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[\s\xa0]", "", regex=True).str.replace(",", "."), errors="coerce")


def load_deposits_yearly() -> pd.DataFrame:
    """Годовые таблицы ЦБ 2007–2018: вклады физлиц всего / в рублях / в валюте, млн руб.,
    на 1-е число месяца → относим к концу предыдущего месяца."""
    rows = []
    for f in sorted(CBR.glob("deposits_*.html")):
        year = int(re.search(r"(\d{4})", f.name).group(1))
        t = pd.read_html(f)[0]
        t.columns = [c[1] if isinstance(c, tuple) else c for c in t.columns]
        # первая колонка — названия строк; остальные — даты вида «01.02» (или «1.02»)
        labels = t.iloc[:, 0].astype(str).str.replace("\xa0", " ").str.strip().str.lower()
        header = t.iloc[0, 1:].astype(str) if not re.match(r"\d{1,2}\.\d{2}", str(t.columns[1])) else pd.Series(t.columns[1:], index=t.columns[1:])
        dates = [str(h).strip() for h in header]
        def row(pattern):
            idx = labels[labels.str.match(pattern)].index
            return _num(t.loc[idx[0], t.columns[1:]]) if len(idx) else None
        total, rub, fx = row(r"^всего"), row(r"^в рублях"), row(r"^в иностранной валюте")
        if total is None or rub is None or fx is None:
            print(f"  {f.name}: не нашёл строки, пропускаю"); continue
        for d, a, b, c in zip(dates, total, rub, fx):
            m = re.match(r"(\d{1,2})\.(\d{2})", d)
            if not m or pd.isna(a):
                continue
            as_of = pd.Period(f"{year}-{int(m.group(2)):02d}", freq="M")
            rows.append({"month": as_of - 1, "dep_total": a, "dep_rub": b, "dep_fx": c, "src": "yearly"})
    return pd.DataFrame(rows)


def load_deposits_xlsx() -> pd.DataFrame:
    """funds_all.xlsx: лист «итого», «в рублях», «в инвалюте»; строка «вклады … физических лиц
    (с учетом счетов эскроу)»; колонки — даты на 1-е число, млн руб."""
    out = {}
    for sheet, key in (("итого", "dep_total"), ("в рублях", "dep_rub"), ("в инвалюте", "dep_fx")):
        t = pd.read_excel(CBR / "funds_all.xlsx", sheet_name=sheet, header=None)
        dates = pd.to_datetime(t.iloc[1, 1:], format="%d.%m.%Y", errors="coerce")
        lab = t.iloc[:, 0].astype(str).str.lower()
        i = lab[lab.str.contains("физических лиц") & lab.str.contains("эскроу") & ~lab.str.startswith("\\*")].index[0]
        vals = _num(t.iloc[i, 1:])
        s = pd.Series(vals.values, index=pd.PeriodIndex(dates, freq="M") - 1)
        out[key] = s[s.index.notna()]
    d = pd.DataFrame(out).rename_axis("month").reset_index()
    d["src"] = "xlsx"
    return d.dropna(subset=["dep_total"])


def load_deposits() -> pd.DataFrame:
    y, x = load_deposits_yearly(), load_deposits_xlsx()
    both = y.merge(x, on="month", suffixes=("_y", "_x"))
    if len(both):
        diff = ((both.dep_total_y - both.dep_total_x) / both.dep_total_x).abs()
        print(f"  стык годовых таблиц и xlsx: {len(both)} общих месяцев, расхождение по сумме вкладов "
              f"медиана {diff.median():.1%}, макс {diff.max():.1%}")
    d = pd.concat([y, x[~x.month.isin(y.month)]]).sort_values("month").reset_index(drop=True)
    return d


def load_fx_emp() -> pd.DataFrame:
    p = pd.read_parquet(DATA_PROCESSED / "econ_kr_panel.parquet", columns=["iso3", "month", "fx", "emp_z", "crisis", "de"])
    p = p[p.iso3 == "RUS"].drop(columns="iso3")
    p["month"] = pd.PeriodIndex(p.month, freq="M")
    return p


def load_bop() -> pd.DataFrame:
    b = pd.read_csv(DATA_RAW / "imf" / "bop_cash_deposits.csv", usecols=["COUNTRY", "BOP_ACCOUNTING_ENTRY", "INDICATOR", "UNIT", "TIME_PERIOD", "OBS_VALUE"])
    b = b[(b.COUNTRY == "RUS") & (b.INDICATOR == "O_F2_S1V") & (b.UNIT == "USD") & b.BOP_ACCOUNTING_ENTRY.str.startswith("A") & b.OBS_VALUE.notna()]
    b["quarter"] = pd.PeriodIndex(b.TIME_PERIOD.str.replace("-", ""), freq="Q")
    return b[["quarter", "OBS_VALUE"]].rename(columns={"OBS_VALUE": "bop_cash_dep_usd_bn"}).assign(
        bop_cash_dep_usd_bn=lambda d: d.bop_cash_dep_usd_bn / 1e9).sort_values("quarter")


def build() -> pd.DataFrame:
    d = load_deposits().merge(load_fx_emp(), on="month", how="left")
    d["fx_share"] = 100 * d.dep_fx / d.dep_total
    d["dep_fx_usd_bn"] = d.dep_fx / d.fx / 1e3  # млн руб / (руб за долл) → млн долл → млрд
    d["dep_rub_bn"] = d.dep_rub / 1e3
    d["rub_dep_g"] = 100 * d.dep_rub.pct_change()
    d["fx_dep_usd_g"] = 100 * d.dep_fx_usd_bn.pct_change()
    d["fx_share_d"] = d.fx_share.diff()
    return d


def episodes_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ep, note in EPISODES.items():
        t0 = pd.Period(ep, freq="M")
        for k in range(WINDOW[0], WINDOW[1] + 1):
            r = d[d.month == t0 + k]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({"episode": ep, "offset": k, "month": str(t0 + k), "emp_z": round(r.emp_z, 2) if pd.notna(r.emp_z) else np.nan,
                         "fx_rub_per_usd": round(r.fx, 1) if pd.notna(r.fx) else np.nan,
                         "rub_dep_g": round(r.rub_dep_g, 1), "fx_dep_usd_g": round(r.fx_dep_usd_g, 1),
                         "fx_share": round(r.fx_share, 1), "dep_fx_usd_bn": round(r.dep_fx_usd_bn, 1)})
    return pd.DataFrame(rows)


def leadlag(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x = d.set_index("month")
    for key in ("rub_dep_g", "fx_dep_usd_g", "fx_share_d"):
        s = x[key]
        for k in range(-6, 7):
            e = x.emp_z.shift(-k)
            ok = s.notna() & e.notna()
            rows.append({"series": key, "lead_k": k, "corr": s[ok].corr(e[ok]), "n": int(ok.sum())})
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    print("Вклады ЦБ РФ…")
    d = build()
    d.assign(month=d.month.astype(str)).to_parquet(DATA_PROCESSED / "behavior_ru_monthly.parquet", index=False)
    d.assign(month=d.month.astype(str)).to_csv(REPORTS / "behavior_ru_monthly.csv", index=False, float_format="%.3f")
    print(f"  {d.month.min()}–{d.month.max()}, {len(d)} месяцев; медианы: рост руб. вкладов {d.rub_dep_g.median():.1f} %/мес, "
          f"рост валютных в долл. {d.fx_dep_usd_g.median():.1f} %/мес, доля валютных {d.fx_share.median():.0f} %")
    ep = episodes_table(d)
    ep.to_csv(REPORTS / "behavior_ru_episodes.csv", index=False)
    pd.set_option("display.width", 200)
    for name, note in EPISODES.items():
        print(f"\n== {name}: {note} ==")
        print(ep[ep.episode == name].drop(columns="episode").to_string(index=False))
    ll = leadlag(d)
    ll.to_csv(REPORTS / "behavior_ru_leadlag.csv", index=False)
    print("\nКорреляция ряда в месяце t с EMP в месяце t+k (k > 0: действия раньше давления):")
    print(ll.pivot(index="series", columns="lead_k", values="corr").round(2).to_string())

    print("\nПлатёжный баланс: чистое приобретение наличной валюты и депозитов за рубежом домохозяйствами и компаниями, млрд долл. за квартал")
    b = load_bop()
    b.assign(quarter=b.quarter.astype(str)).to_csv(REPORTS / "behavior_ru_bop_quarterly.csv", index=False)
    med = b.bop_cash_dep_usd_bn.median()
    for ep in ("2008-10", "2014-10", "2018-04"):
        q0 = pd.Period(ep, freq="M").asfreq("Q")
        w = b[(b.quarter >= q0 - 5) & (b.quarter <= q0 + 1)]
        print(f"  {ep} (медиана по всем кварталам {med:.1f}):", ", ".join(f"{str(q)}: {v:+.1f}" for q, v in zip(w.quarter, w.bop_cash_dep_usd_bn)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
