"""Месячный валютный блок, ПРИЗНАКИ: индикаторы Kaminsky–Lizondo–Reinhart и война.

Всё на строке месяца m посчитано по данным ≤ m. Список индикаторов и хвост, в котором
индикатор подаёт сигнал, — `INDICATORS`. Описание — в docstring `econ_kr.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED
from seldon.core.entities import GW_TO_ISO3
from seldon.core.panel import pct_change
from seldon.core.temporal import last_available_year_in_month

# индикатор → в каком хвосте сигнал: "high" — сигнал при больших значениях, "low" — при малых
INDICATORS = {
    "reserves_g12": "low",   # резервы упали за 12 мес
    "rer_dev": "low",        # реальный курс крепче тренда (переоценка): e·P*/P ниже тренда
    "deval12": "high",       # уже девальвировались за 12 мес
    "infl12": "high",        # инфляция за 12 мес
    "real_rate": "high",     # реальная ставка (защита курса ставкой)
    "rate_diff_us": "high",  # разница ставок с США
    "spread": "high",        # кредитная минус депозитная
    "reserves_g1": "low",    # резервы за последний месяц
    "m2_res_g12": "high",    # M2 (в долл.) к резервам: рост за 12 мес — лучший индикатор KLR
    "exports_g12": "low",    # экспорт упал за 12 мес
    "world_crisis3": "high", # заражение: доля стран мира с кризисом за последние 3 мес
    "emp_z_max3": "high",    # давление уже идёт: максимум EMP (в σ) за последние 3 мес
    # война (UCDP, panel_country_month): связка блоков «война → кризис»
    "war25_now": "high",     # государственная война ≥ 25 погибших в этом месяце (у себя или у соседа-стороной)
    "war1000_now": "high",   # ≥ 1000 погибших за последние 12 месяцев
    "war_onset12": "high",   # начало войны (после ≥ 12 тихих месяцев) за последние 12 мес
}
BINARY = {"war25_now", "war1000_now", "war_onset12"}  # сигнал = 1, без порога-процентиля
WAR_COLS = ["war25_now", "war1000_now", "war_onset12"]


def fill_cpi(p: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Дыры в месячном ИПЦ МВФ затыкаем годовым ИПЦ Всемирного банка — без утечки будущего.

    Индекс цен собирается цепочкой из месячных ТЕМПОВ, а не уровней:
      - если ИПЦ МВФ есть и в этом месяце, и в прошлом — темп из МВФ;
      - иначе — темп последнего года, за который годовое среднее ВБ уже ВЫШЛО:
        ln(A(y)/A(y−1)) / 12. Годовые данные ВБ за y доступны с 1 апреля y+1
        (`core/temporal.py`), поэтому в январе–марте года t это y = t−2, с апреля — y = t−1.
        (До чекпоинта 2.2 A(t−1) считалось известным уже в январе t.)
    Уровень индекса произвольный (везде используются только темпы и отношение к своему
    тренду). До аудита уровень заполнения подгонялся к первому БУДУЩЕМУ наблюдению МВФ
    (утечка), а темп внутри года был занижен в 12 раз и давал скачок в январе.
    Флаг `cpi_filled` отмечает месяцы, где темп взят из годовых данных."""
    p = p.sort_values(["iso3", "month"]).reset_index(drop=True)
    a = annual[annual.value > 0].sort_values(["iso3", "year"]).copy()
    a["prev"] = a.groupby("iso3").value.shift(1)
    a["prev_year"] = a.groupby("iso3").year.shift(1)
    a = a[a.prev_year == a.year - 1]  # нужны два подряд идущих года
    a["g_annual"] = np.log(a.value / a.prev) / 12  # месячный лог-темп года a.year
    p["year"] = last_available_year_in_month("wdi_annual", p.month)  # последний вышедший год
    p = p.merge(a[["iso3", "year", "g_annual"]], on=["iso3", "year"], how="left")
    same = p.iso3 == p.iso3.shift(1)
    g_imf = np.log(p.cpi / p.cpi.shift(1)).where(same)
    g = g_imf.fillna(p.g_annual)
    p["cpi_filled"] = g_imf.isna() & p.g_annual.notna()
    # цепочка: отрезки подряд идущих известных темпов; месяц перед отрезком — база (уровень 1)
    seg = g.isna().groupby(p.iso3).cumsum()
    level = np.exp(g.fillna(0.0).groupby([p.iso3, seg]).cumsum())
    isolated = g.isna() & (g.shift(-1).isna() | (p.iso3 != p.iso3.shift(-1)))
    p["cpi"] = level.where(~isolated)
    return p.drop(columns=["year", "g_annual"])


def add_indicators(p: pd.DataFrame, war: pd.DataFrame | None = None) -> pd.DataFrame:
    p = p.copy()
    us = p[p.iso3 == "USA"].set_index("month")[["cpi", "rate"]].rename(
        columns={"cpi": "cpi_us", "rate": "rate_us"})
    p = p.merge(us, left_on="month", right_index=True, how="left")
    gb = p.groupby("iso3")
    p["reserves_g12"] = gb.reserves.transform(lambda s: pct_change(s, 12)).clip(-1, 3)
    p["reserves_g1"] = p.dr
    p["deval12"] = gb.fx.transform(lambda s: pct_change(s, 12)).clip(-0.9, 5)
    p["infl12"] = gb.cpi.transform(lambda s: pct_change(s, 12)).clip(-0.5, 5)
    # реальный курс: e · P_US / P. Рост = реальное ослабление. Отклонение от тренда —
    # отношение к скользящему среднему за 36 мес (только прошлое: без утечки будущего)
    p["rer"] = p.fx * p.cpi_us / p.cpi
    trend = gb.rer.transform(lambda s: s.rolling(36, min_periods=24).mean())
    p["rer_dev"] = (p.rer / trend - 1).clip(-0.9, 3)
    p["real_rate"] = (p.rate - p.infl12 * 100).clip(-100, 200)
    p["rate_diff_us"] = (p.rate - p.rate_us).clip(-50, 200)
    p["spread"] = (p.lending - p.deposit).clip(-20, 100)
    p["m2_res"] = p.broad_money / p.fx / p.reserves
    p["m2_res_g12"] = gb.m2_res.transform(lambda s: pct_change(s, 12)).clip(-0.9, 5)
    # экспорт помесячно шумный и сезонный: сравниваем суммы за 12 мес
    x12 = gb.exports.transform(lambda s: s.rolling(12, min_periods=12).sum())
    p["exports_g12"] = (x12 / x12.shift(12) - 1).where(p.iso3 == p.iso3.shift(12)).clip(-0.9, 3)
    # всё ниже известно на момент прогноза: кризисы прошлых месяцев и собственный EMP
    live = p[p.own_currency & p.de.notna()]
    world = live.groupby("month").agg(n=("crisis", "count"), k=("crisis", "sum"))
    share = (world.k / world.n).rolling(3, min_periods=1).sum()
    p["world_crisis3"] = p.month.map(share)
    p["emp_z_max3"] = gb.emp_z.transform(lambda s: s.rolling(3, min_periods=1).max()).clip(-5, 50)
    if war is None:
        war = load_war()
    return p.merge(war[["iso3", "month", *WAR_COLS]], on=["iso3", "month"], how="left")


def war_features(month_panel: pd.DataFrame, gw_to_iso3: dict | None = None, realtime: bool = True) -> pd.DataFrame:
    """Война по месяцам из блока войн (`domains/conflict/panel.py`, коды GW) → iso3, month и три признака.
    Всё на строке месяца m посчитано по событиям ≤ m:
      war25_now    — государственная война: ≥ 25 погибших за месяц у себя или у соседа,
                     где страна — сторона (`state_war25`);
      war1000_now  — ≥ 1000 погибших за последние 12 месяцев (скользящая сумма);
      war_onset12  — начало войны (после ≥ 12 тихих месяцев) за последние 12 мес.
    `realtime=True` (по умолчанию) — колонки `*_rt` месячной панели: кто сторона войны,
    берётся из списка ACD за ПРОШЛЫЙ год (за текущий он выйдет после его окончания)."""
    sfx = "_rt" if realtime else ""  # `_rt` — как было видно в реальном времени: стороны по ACD за прошлый год
    w = month_panel[["gw", "year", "month", f"state_war25{sfx}", f"state_war1000_12m{sfx}", f"onset_state_war25{sfx}"]].copy()
    w.columns = ["gw", "year", "month", "state_war25", "state_war1000_12m", "onset_state_war25"]
    w["iso3"] = w.gw.map(GW_TO_ISO3 if gw_to_iso3 is None else gw_to_iso3)
    w = w[w.iso3.notna()].copy()
    w["month"] = pd.PeriodIndex(pd.to_datetime(dict(year=w.year, month=w.month, day=1)), freq="M")
    w = w.sort_values(["iso3", "gw", "month"])
    w["war_onset12"] = w.groupby("gw").onset_state_war25.transform(lambda s: s.rolling(12, min_periods=1).max())
    w = w.drop_duplicates(["iso3", "month"], keep="first")  # 340/345 → SRB
    w = w.rename(columns={"state_war25": "war25_now", "state_war1000_12m": "war1000_now"})
    return w[["iso3", "month", *WAR_COLS]].reset_index(drop=True)


def load_war() -> pd.DataFrame:
    return war_features(pd.read_parquet(DATA_PROCESSED / "panel_country_month.parquet"))
