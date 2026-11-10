"""Поведение масс перед валютными кризисами — на всех странах, а не на двух российских эпизодах.

Запуск:
    uv run python -m seldon.behavior_world bop      # платёжный баланс, ~400 кризисов 1996–2021
    uv run python -m seldon.behavior_world gdelt    # новости GDELT, кризисы 2017–2025
    uv run python -m seldon.behavior_world gdelt-cached   # только из кэша data/raw/gdelt
    uv run python -m seldon.behavior_world          # оба

Вопрос тот же, что в behavior_ru/news_ru: опережает ли поведение (бегство в валюту, тревога
в новостях) давление на курс — или идёт вместе с ним? Здесь — событийный анализ: для каждого
EMP-кризиса (econ_kr, 3σ + 10 %) берём ряд в окне до и после, нормируем по истории самой
страны и усредняем по всем кризисам. Отдельно — кризисы вне мировых волн (1998, 2008, 2020,
2022), т.е. зревшие внутри страны: если поведение что-то предсказывает, то именно их.

1. Платёжный баланс МВФ (`download imf`, поток `bop_cash_deposits`), поквартально с 1994:
   «наличная валюта и депозиты» домохозяйств и нефинансовых компаний (O_F2_S1V; где нет —
   прочих секторов O_F2_S1Z), чистое приобретение, долл. Нормировка: z по стране
   (медиана/MAD). Окно −4…+2 квартала. 121 страна.
2. GDELT DOC API (с 2017, без ключа, кэш в data/raw/gdelt/): доля новостей страны о курсе
   и девальвации (`timelinevol`) и их тон (`timelinetone`), помесячно. Нормировка: к медиане
   страны. Окно −6…+2 месяца. ~30 стран с EMP-кризисами после 2017-07.

Выход:
    reports/behavior_world_bop_event.csv, behavior_world_bop_by_crisis.csv
    reports/behavior_world_gdelt_event.csv, behavior_world_gdelt_by_crisis.csv
"""

from __future__ import annotations

import json
import sys
import time
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from seldon import DATA_PROCESSED, DATA_RAW, PROJECT_ROOT

REPORTS = PROJECT_ROOT / "reports"
GDELT_DIR = DATA_RAW / "gdelt"
WAVE_SHARE = 0.10  # квартал/месяц считаем «мировой волной», если в кризисе ≥ 10 % стран с данными

# ISO3 → FIPS (так GDELT кодирует страну источника)
FIPS = {"TUR": "TU", "ARG": "AR", "EGY": "EG", "GHA": "GH", "NGA": "NI", "PAK": "PK", "LKA": "CE", "LBN": "LE",
        "KAZ": "KZ", "RUS": "RS", "UKR": "UP", "BRA": "BR", "COL": "CO", "ZAF": "SF", "MEX": "MX", "IDN": "ID",
        "KOR": "KS", "GBR": "UK", "JPN": "JA", "HUN": "HU", "CHL": "CI", "BLR": "BO", "UZB": "UZ", "ZMB": "ZA",
        "ETH": "ET", "AGO": "AO", "KGZ": "KG", "GEO": "GG", "VEN": "VE", "ZWE": "ZI", "MWI": "MI", "SDN": "SU",
        "IRQ": "IZ", "LAO": "LA", "SLE": "SL", "TJK": "TI", "NZL": "NZ", "BWA": "BC", "SUR": "NS"}
GDELT_QUERY = '("exchange rate" OR devaluation OR depreciation OR "currency crisis" OR "dollar rate" OR "capital controls")'
GDELT_FROM, GDELT_TO = 2017, 2025
# тон отключён: GDELT считает каждый запрос 30–60 с, с тоном 39 стран — больше двух часов
MODES = (("timelinevol", "vol"),)
OFFLINE = False  # `gdelt-cached`: только то, что уже в data/raw/gdelt (GDELT режет по частоте)
# страны с заметным англоязычным медиапокрытием; мелкие (BWA, SUR, SLE, LAO…) пропускаем
GDELT_COUNTRIES = {"AGO", "ARG", "BLR", "BRA", "COL", "EGY", "ETH", "GHA", "NGA", "PAK", "LKA", "LBN", "KAZ", "RUS",
                   "UKR", "ZAF", "MEX", "IDN", "KOR", "GBR", "JPN", "HUN", "UZB", "ZMB", "VEN", "TUR", "CHL", "ZWE"}


# ---------------------------------------------------------------------------
# Кризисы
# ---------------------------------------------------------------------------


def load_crises() -> pd.DataFrame:
    p = pd.read_parquet(DATA_PROCESSED / "econ_kr_panel.parquet", columns=["iso3", "month", "crisis", "de"])
    p["month"] = pd.PeriodIndex(p.month, freq="M")
    share = p[p.de.notna()].groupby("month").crisis.mean()
    c = p[p.crisis == 1].copy()
    c["quarter"] = c.month.dt.asfreq("Q")
    # волна: доля стран в кризисе в этом месяце ±1 ≥ WAVE_SHARE
    roll = share.rolling(3, center=True, min_periods=1).max()
    c["wave"] = c.month.map(roll) >= WAVE_SHARE
    return c


# ---------------------------------------------------------------------------
# 1. Платёжный баланс
# ---------------------------------------------------------------------------


def load_bop() -> pd.DataFrame:
    b = pd.read_csv(DATA_RAW / "imf" / "bop_cash_deposits.csv",
                    usecols=["COUNTRY", "BOP_ACCOUNTING_ENTRY", "INDICATOR", "UNIT", "TIME_PERIOD", "OBS_VALUE"])
    b = b[b.OBS_VALUE.notna() & (b.UNIT == "USD") & b.BOP_ACCOUNTING_ENTRY.str.startswith("A")]
    b["quarter"] = pd.PeriodIndex(b.TIME_PERIOD.str.replace("-", ""), freq="Q")
    out = []
    for iso, g in b.groupby("COUNTRY"):
        ind = "O_F2_S1V" if (g.INDICATOR == "O_F2_S1V").sum() >= 40 else "O_F2_S1Z"
        s = g[g.INDICATOR == ind].drop_duplicates("quarter").set_index("quarter").OBS_VALUE.sort_index()
        if len(s) < 40:
            continue
        mad = (s - s.median()).abs().median() * 1.4826
        if mad <= 0:
            continue
        out.append(pd.DataFrame({"iso3": iso, "quarter": s.index, "value": s.values / 1e9,
                                 "z": ((s - s.median()) / mad).clip(-10, 10).values, "indicator": ind}))
    return pd.concat(out, ignore_index=True)


def event_study(series: pd.DataFrame, key: str, crises: pd.DataFrame, offsets: range, cols: list[str]
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """series: iso3, <key>, cols…; crises: iso3, <key>, wave. Возвращает средние по смещениям
    (все / вне волн / волны) и таблицу по кризисам."""
    idx = series.set_index(["iso3", key])
    rows = []
    for _, c in crises.iterrows():
        rec = {"iso3": c.iso3, "crisis": str(c[key]), "wave": bool(c.wave)}
        found = False
        for k in offsets:
            t = c[key] + k
            if (c.iso3, t) in idx.index:
                r = idx.loc[(c.iso3, t)]
                for col in cols:
                    rec[f"{col}_{k:+d}"] = float(r[col])
                found = True
        if found:
            rows.append(rec)
    by = pd.DataFrame(rows)
    out = []
    for scope, mask in (("all", by.wave | ~by.wave), ("internal", ~by.wave), ("wave", by.wave)):
        d = by[mask]
        for k in offsets:
            row = {"scope": scope, "offset": k, "n_crises": len(d)}
            for col in cols:
                v = d[f"{col}_{k:+d}"].dropna()
                row[f"{col}_mean"] = v.mean()
                row[f"{col}_share_gt1"] = (v > 1).mean() if len(v) else np.nan
                row[f"{col}_n"] = len(v)
            out.append(row)
    return pd.DataFrame(out), by


def run_bop() -> None:
    bop = load_bop()
    crises = load_crises()
    crises = crises[(crises.quarter >= "1996Q1") & (crises.quarter <= "2021Q4")]
    crises = crises.drop_duplicates(["iso3", "quarter"])
    ev, by = event_study(bop, "quarter", crises, range(-4, 3), ["z"])
    ev.to_csv(REPORTS / "behavior_world_bop_event.csv", index=False)
    by.to_csv(REPORTS / "behavior_world_bop_by_crisis.csv", index=False)
    print(f"Платёжный баланс: {bop.iso3.nunique()} стран, кризисов с данными {len(by)} "
          f"(вне волн {int((~by.wave).sum())}, в волнах {int(by.wave.sum())})")
    print("z приобретения валюты и зарубежных депозитов (по стране: медиана/MAD), среднее по кризисам;"
          " share_gt1 — доля кризисов, где z > 1 (заметное бегство в валюту):")
    piv = ev.pivot(index="offset", columns="scope", values=["z_mean", "z_share_gt1"]).round(2)
    print(piv.to_string())
    base = (bop.z > 1).mean()
    print(f"  фон: доля всех кварталов с z > 1 = {base:.2f}")


# ---------------------------------------------------------------------------
# 2. GDELT
# ---------------------------------------------------------------------------


def gdelt_fetch(fips: str, mode: str, y0: int, y1: int) -> pd.Series:
    GDELT_DIR.mkdir(parents=True, exist_ok=True)
    f = GDELT_DIR / f"{fips}_{mode}_{y0}_{y1}.json"
    if not f.exists() and OFFLINE:
        return pd.Series(dtype=float)
    if not f.exists():
        url = ("https://api.gdeltproject.org/api/v2/doc/doc?query=" + quote(f"{GDELT_QUERY} sourcecountry:{fips}")
               + f"&mode={mode}&format=json&startdatetime={y0}0101000000&enddatetime={y1}1231235959")
        # GDELT просит не чаще одного запроса в 5 секунд — соблюдаем
        status, text = None, ""
        for attempt in range(4):
            try:
                r = requests.get(url, timeout=120, headers={"User-Agent": "seldon-research/0.1 (personal research project)"})
                status, text = r.status_code, r.text
            except requests.RequestException as e:
                status, text = "conn", str(e)[:80]
            if status == 200 and text.strip().startswith("{"):
                f.write_text(text)
                break
            time.sleep(30 * (attempt + 1))
        else:
            print(f"  GDELT {fips} {mode} {y0}–{y1}: не удалось ({status}): {text[:80]}")
            return pd.Series(dtype=float)
        time.sleep(15)
    d = json.loads(f.read_text())
    tl = d.get("timeline", [])
    if not tl:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(x["date"]): x["value"] for x in tl[0]["data"]})
    return s


def load_gdelt(isos: list[str]) -> pd.DataFrame:
    out = []
    for iso in isos:
        fips = FIPS.get(iso)
        if not fips:
            continue
        parts = {}
        for mode, name in MODES:
            chunks = [gdelt_fetch(fips, mode, y, min(y + 2, GDELT_TO)) for y in range(GDELT_FROM, GDELT_TO + 1, 3)]
            s = pd.concat([c for c in chunks if len(c)]) if any(len(c) for c in chunks) else pd.Series(dtype=float)
            if len(s):
                parts[name] = s.groupby(s.index.to_period("M")).mean()
        if "vol" not in parts:
            continue
        d = pd.DataFrame(parts).rename_axis("month").reset_index()
        if len(d) < 60:  # нужен почти весь период, иначе медиана страны не та
            print(f"  {iso} ({fips}): только {len(d)} месяцев в кэше, пропускаю")
            continue
        d["iso3"] = iso
        d["vol_rel"] = d.vol / d.vol.median()  # к медиане страны: 1 = как обычно
        if "tone" in d:
            d["tone_d"] = d.tone - d.tone.median()  # тон ниже обычного — тревожнее
        out.append(d)
        print(f"  {iso} ({fips}): {len(d)} месяцев, медиана доли {d.vol.median():.3f} %")
    return pd.concat(out, ignore_index=True)


def run_gdelt() -> None:
    crises = load_crises()
    crises = crises[crises.month >= pd.Period(f"{GDELT_FROM}-07", freq="M")]
    isos = sorted(set(crises.iso3) & set(FIPS) & GDELT_COUNTRIES)
    print(f"GDELT: {len(isos)} стран с EMP-кризисами после {GDELT_FROM}-07 и кодом FIPS")
    g = load_gdelt(isos)
    cols = ["vol_rel"] + (["tone_d"] if "tone_d" in g else [])
    cr = crises[crises.iso3.isin(g.iso3.unique())]
    ev, by = event_study(g, "month", cr, range(-6, 3), cols)
    ev.to_csv(REPORTS / "behavior_world_gdelt_event.csv", index=False)
    by.to_csv(REPORTS / "behavior_world_gdelt_by_crisis.csv", index=False)
    print(f"кризисов с данными GDELT: {len(by)} (вне волн {int((~by.wave).sum())}, в волнах {int(by.wave.sum())})")
    print("доля новостей о курсе к медиане страны (1 = как обычно) и тон минус медиана, среднее по кризисам:")
    vals = ["vol_rel_mean"] + (["tone_d_mean"] if "tone_d" in g else [])
    print(ev.pivot(index="offset", columns="scope", values=vals).round(2).to_string())
    print("\nпо кризисам вне волн, доля новостей за −6…0:")
    show = [c for c in by.columns if c.startswith("vol_rel_") and int(c.rsplit("_", 1)[1]) <= 0]
    print(by[~by.wave][["iso3", "crisis", *show]].round(2).to_string(index=False))


def main(argv: list[str]) -> int:
    global OFFLINE
    what = argv[0] if argv else "all"
    if what in ("bop", "all"):
        run_bop()
    if what == "gdelt-cached":
        OFFLINE = True
        run_gdelt()
    elif what in ("gdelt", "all"):
        run_gdelt()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
