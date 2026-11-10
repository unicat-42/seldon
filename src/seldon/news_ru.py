"""Газеты как поведенческий слой: о чём писала Lenta.ru за месяцы до валютных кризисов в России.

Запуск:
    uv run python -m seldon.news_ru

Вопрос: растут ли «тревожные» слова в новостях раньше, чем давление на рубль (EMP), —
барометр, или одновременно с ним — термометр? Статистики на 2–3 эпизодах не построить,
это описательная проверка перед тем, как тащить газеты в модель.

Данные: открытый корпус Lenta.ru 1999-09…2019-12 (`download lenta`), ~800 тыс. статей.
Считаем помесячно долю статей, в которых встречается хотя бы одно слово словаря
(заголовок + текст, регистр не важен). Словари:
    fx        курс рубля, обменники, доллар/евро, девальвация, падение рубля
    banks     вклады, отзыв лицензии, банкротство банка, очереди в банках, АСВ
    prices    инфляция, подорожание, дефицит, цены на продукты
    crisis    слова «кризис», «дефолт», «обвал», «паника»
    protest   митинг, протест, акция протеста, задержания
    power     отставка, премьер, кремль, санкции
    oil       нефть, Brent, баррель
Сами тексты никуда не выводятся — только счётчики.

EMP — из econ_kr (econ_kr_panel.parquet, страна RUS, индекс давления в σ по месяцам).
Эпизоды (датировка LV/EMP): 2008-10 (мировой кризис, EMP-кризис 2008-10, LV банковский
2008), 2014-10/12 (EMP-кризис 2014-10, LV валютный 2014), плюс 2018-04/08 (санкции,
рубль −10 %, кризисом не стал) как контроль.

Выход:
    data/processed/news_ru_monthly.parquet — доли словарей по месяцам
    reports/news_ru_monthly.csv, news_ru_episodes.csv (окна −12…+3 мес вокруг эпизодов),
    news_ru_leadlag.csv (корреляция доли словаря в месяце t с EMP в месяце t+k, k = −6…+6)
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, DATA_RAW, PROJECT_ROOT

REPORTS = PROJECT_ROOT / "reports"
CORPUS = DATA_RAW / "lenta" / "lenta-ru-news.csv.bz2"

DICTS = {
    "fx": r"курс[а-я]* (рубл|доллар|евро)|обменник|обменн[а-я]+ пункт|девальвац|падени[а-я]+ рубл|ослаблени[а-я]+ рубл|рубль (упал|рухнул|подешевел)|скупа[а-я]+ валют",
    "banks": r"вклад[а-я]*|депозит[а-я]*|отозва[а-я]+ лиценз|лишил[а-я]* лиценз|банкротств[а-я]+ банк|очеред[а-я]+ (в|у) банк|агентств[а-я]+ по страхованию вкладов|асв\b|снят[а-я]+ (денег|наличн)",
    "prices": r"инфляци|подорожа|рост[а-я]* цен|дефицит[а-я]* (товар|продукт)|цен[а-я]* на (продукт|продовольств|бензин)",
    "crisis": r"кризис|дефолт|обвал|паник",
    "protest": r"митинг|протест|акци[а-я]+ протест|задержан[а-я]+ (участник|активист)|беспорядк",
    "power": r"отставк|премьер-министр|кремл|санкци",
    "oil": r"нефт|brent|баррел",
}
EPISODES = {"2008-10": "мировой кризис; EMP-кризис 2008-10, LV банковский 2008",
            "2014-10": "Крым/санкции/нефть; EMP-кризис 2014-10, LV валютный 2014",
            "2018-04": "контроль: санкции, рубль −10 %, кризисом не стал"}
WINDOW = (-12, 3)


def monthly_shares() -> pd.DataFrame:
    """Читаем корпус кусками, считаем по месяцам число статей и число статей с совпадением."""
    pats = {k: re.compile(v, re.IGNORECASE) for k, v in DICTS.items()}
    acc: dict[pd.Period, dict[str, int]] = {}
    for chunk in pd.read_csv(CORPUS, usecols=["title", "text", "date", "topic"], chunksize=50_000, compression="bz2"):
        # в корпусе есть рубрика «Библиотека» — перепечатки газет 1914 года и т.п.; выкидываем
        chunk = chunk[(chunk.topic != "Библиотека")]
        month = pd.to_datetime(chunk.date, format="%Y/%m/%d", errors="coerce").dt.to_period("M")
        chunk = chunk[month.notna() & (month >= pd.Period("1999-01", freq="M"))]
        month = month[chunk.index]
        txt = (chunk.title.fillna("") + " " + chunk.text.fillna("")).str.lower()
        flags = pd.DataFrame({k: txt.str.contains(p) for k, p in pats.items()})
        flags["n"] = 1
        flags["month"] = month.values
        g = flags.dropna(subset=["month"]).groupby("month").sum()
        for m, row in g.iterrows():
            a = acc.setdefault(m, {k: 0 for k in [*DICTS, "n"]})
            for k in a:
                a[k] += int(row[k])
        print(f"  …{len(acc)} месяцев, последний {max(acc)}", end="\r")
    print()
    out = pd.DataFrame.from_dict(acc, orient="index").sort_index()
    out.index.name = "month"
    for k in DICTS:
        out[f"share_{k}"] = out[k] / out.n
    return out.reset_index()


def load_emp() -> pd.DataFrame:
    p = pd.read_parquet(DATA_PROCESSED / "econ_kr_panel.parquet", columns=["iso3", "month", "emp_z", "de", "dr", "crisis"])
    p = p[p.iso3 == "RUS"].copy()
    p["month"] = pd.PeriodIndex(p.month, freq="M")
    return p.drop(columns="iso3")


def episodes_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = {k: d[f"share_{k}"].median() for k in DICTS}
    for ep, note in EPISODES.items():
        t0 = pd.Period(ep, freq="M")
        for k in range(WINDOW[0], WINDOW[1] + 1):
            m = t0 + k
            r = d[d.month == m]
            if r.empty:
                continue
            r = r.iloc[0]
            row = {"episode": ep, "offset": k, "month": str(m), "n_articles": int(r.n), "emp_z": round(r.emp_z, 2) if pd.notna(r.emp_z) else np.nan}
            for key in DICTS:  # доля относительно медианы за всё время: 1 = как обычно
                row[key] = round(r[f"share_{key}"] / base[key], 2)
            rows.append(row)
    return pd.DataFrame(rows)


def leadlag(d: pd.DataFrame) -> pd.DataFrame:
    """Корреляция доли словаря в месяце t с EMP в месяце t+k. k > 0 — слова опережают EMP."""
    rows = []
    x = d.set_index("month")
    for key in DICTS:
        s = x[f"share_{key}"]
        for k in range(-6, 7):
            e = x.emp_z.shift(-k)  # EMP через k месяцев
            ok = s.notna() & e.notna()
            rows.append({"dict": key, "lead_k": k, "corr": s[ok].corr(e[ok]), "n": int(ok.sum())})
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    if not CORPUS.exists():
        print(f"нет корпуса {CORPUS}: запусти `uv run python -m seldon.download lenta`")
        return 1
    cache = DATA_PROCESSED / "news_ru_monthly.parquet"
    if cache.exists() and "--recount" not in argv:
        shares = pd.read_parquet(cache)
        shares["month"] = pd.PeriodIndex(shares.month, freq="M")
    else:
        print("Считаю словари по корпусу (несколько минут)…")
        shares = monthly_shares()
        shares.assign(month=shares.month.astype(str)).to_parquet(cache, index=False)
    d = shares.merge(load_emp(), on="month", how="left")
    d.assign(month=d.month.astype(str)).to_csv(REPORTS / "news_ru_monthly.csv", index=False)
    print(f"{d.month.min()}–{d.month.max()}, статей {int(d.n.sum())}, в среднем {int(d.n.mean())} в месяц")
    print("медианные доли словарей:", {k: round(d[f'share_{k}'].median(), 3) for k in DICTS})

    ep = episodes_table(d)
    ep.to_csv(REPORTS / "news_ru_episodes.csv", index=False)
    for name, note in EPISODES.items():
        print(f"\n== {name}: {note} ==  (доли словарей к медиане за 1999–2019; 1 = как обычно)")
        print(ep[ep.episode == name][["offset", "month", "emp_z", *DICTS]].to_string(index=False))

    ll = leadlag(d)
    ll.to_csv(REPORTS / "news_ru_leadlag.csv", index=False)
    piv = ll.pivot(index="dict", columns="lead_k", values="corr").round(2)
    print("\nКорреляция доли словаря в месяце t с EMP в месяце t+k (k > 0: слова раньше давления):")
    print(piv.to_string())
    best = ll[ll.lead_k > 0].sort_values("corr", ascending=False).head(5)
    print("\nлучшие опережения:", best[["dict", "lead_k", "corr"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
