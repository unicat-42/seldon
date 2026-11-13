"""Структурные валютные признаки (чекпоинт 3.3): режим курса, запас резервов, REER,
сырьевые условия торговли. Одна месячная таблица на оба валютных контура.

Строка (страна, месяц m) содержит только то, что известно к концу месяца m:

Режим курса
    peg_share12     живой индикатор привязки: доля месяцев из последних 12, когда курс
                    сдвинулся не больше чем на 1 %, — к доллару или к евро (до 1999 года —
                    к немецкой марке), берётся больший из двух. 1 — жёсткая привязка, ~0,2–0,4
                    — плавание. Разовая девальвация «ступенькой» привязку не отменяет
                    (Казахстан-2014: 11 месяцев из 12). Считается по курсу МВФ ≤ m.
    irr_peg         1, если по классификации Ilzetzki–Reinhart–Rogoff режим — привязка,
                    ползущая привязка или узкий коридор (коды 1–8). Год классификации — последний
                    вышедший (`irr_regime` в `core/temporal.py`: T−2) и не позже `IRR_LAST_YEAR`;
                    дальше значение заморожено, `irr_extended` = 1 (инвариант 6).
Запас резервов (правило Гринспена–Гвидотти: резервы должны покрывать краткосрочный долг)
    gg_log          ln(резервы / краткосрочный внешний долг), ВБ; у развитых стран пропуск.
    res_months_log  ln(1 + резервы в месяцах импорта), ВБ.
                    Оба — за последний вышедший год (`wdi_annual`: с 1 апреля Y+1).
REER (реальный эффективный курс к корзине партнёров, Bruegel; рост = укрепление)
    reer_gap        ln(REER) минус среднее ln(REER) за 60 месяцев — насколько валюта крепче
                    собственного недавнего прошлого. Только прошлое окно, никакого тренда по
                    всей выборке. REER зависит от ИПЦ и известен по месяц m − 2
                    (`bruegel_reer`); последние два месяца «дотягиваются» номинальным курсом к
                    корзине (NEER — одни курсы валют, известен сразу): REER(m) ≈ REER(m−2) ×
                    NEER(m) / NEER(m−2). Так обвал валюты соседа виден в том же месяце.
    reer_peg        «натянутая пружина»: max(reer_gap, 0) × peg_share12 — реальное укрепление
                    при привязанном курсе (соседи уже девальвировались, привязка ещё держит).
Сырьевые условия торговли (МВФ, скользящие веса прошлых лет)
    ctot_d12        изменение ln(индекса) за 12 месяцев ≈ выигрыш (+) или потеря (−) дохода от
                    мировых цен в долях ВВП. Значение — месяца m − 6 (`imf_ctot`).
*_missing           флаги пропусков (сам признак тогда = 0).

Годовой контур берёт декабрьскую строку года T: это ровно то, что известно «на 1 января T+1»
(курс по декабрь T, REER по октябрь T плюс NEER по декабрь T, условия торговли по июнь T, ВБ за T−1, IRR за T−2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from seldon.core.panel import full_grid
from seldon.core.temporal import last_available_year_in_month, monthly_lag
from seldon.sources.irr import IRR_LAST_YEAR, PEG_MAX_CODE

PEG_STEP = 0.01      # месяц «спокойный», если курс к якорю сдвинулся не больше чем на 1 %
PEG_WINDOW, PEG_MIN_MONTHS = 12, 9
REER_WINDOW, REER_MIN_MONTHS = 60, 36
LAST_MONTH = "2025-12"  # как у обоих валютных контуров: декабрьский курс МВФ по 2025-12

# наборы признаков: имя → (числовые, флаги пропусков)
SETS: dict[str, tuple[list[str], list[str]]] = {
    "regime": (["peg_share12", "irr_peg"], ["irr_missing"]),
    "reserves": (["gg_log", "res_months_log"], ["gg_missing", "resm_missing"]),
    "reer": (["reer_gap", "reer_peg"], ["reer_missing"]),
    "ctot": (["ctot_d12"], ["ctot_missing"]),
}
SETS["struct"] = ([c for cont, _ in SETS.values() for c in cont], [f for _, flags in SETS.values() for f in flags])
CONT, FLAGS = SETS["struct"]
COLUMNS = [*CONT, *FLAGS]


@dataclass(frozen=True)
class StructLags:
    """Лаги публикации источников. Значения по умолчанию — из `core/temporal.py`; другие
    нужны только тестам (контроль «без лага») и проверке чувствительности."""
    reer_months: int = monthly_lag("bruegel_reer")  # NEER — без лага (`bruegel_neer` = 0)
    ctot_months: int = monthly_lag("imf_ctot")
    wdi_source: str | None = "wdi_annual"   # None — годовое значение ВБ за текущий год (без лага)
    irr_source: str | None = "irr_regime"   # None — режим IRR за текущий год (без лага)


def _known_year(month: pd.Series, source: str | None) -> pd.Series:
    return month.dt.year if source is None else last_available_year_in_month(source, month)


def build_struct(fx: pd.DataFrame, eur: pd.DataFrame, reer: pd.DataFrame, ctot: pd.DataFrame, irr: pd.DataFrame,
                 buffers: pd.DataFrame, lags: StructLags = StructLags(), irr_last_year: int = IRR_LAST_YEAR,
                 filled: bool = True) -> pd.DataFrame:
    """Месячная таблица структурных признаков: iso3, month, `COLUMNS`, irr_extended.
    Чистая функция: строка месяца m зависит только от данных, известных к концу m.
    `filled=False` — без флагов и нулей вместо пропусков (пропуск остаётся пропуском): для
    описательных индикаторов витрины, где ноль и «нет данных» путать нельзя.

    Вход: `fx` (iso3, month, fx — курс к доллару на конец месяца), `eur` (month, eur_usd),
    `reer` (iso3, month, reer120, reer51, neer120, neer51), `ctot` (iso3, month, ctot), `irr` (iso3, year,
    irr_fine), `buffers` (iso3, year, std_res, res_months)."""
    idx = full_grid(fx.groupby("iso3").month.agg(["min", "max"]), names=["iso3", "month"], freq="M")
    p = pd.DataFrame(index=idx).reset_index().merge(fx[["iso3", "month", "fx"]], on=["iso3", "month"], how="left")
    p = p.sort_values(["iso3", "month"]).reset_index(drop=True)

    # --- живой индикатор привязки: к доллару и к евро
    e = eur.sort_values("month")
    e_grid = pd.DataFrame({"month": pd.period_range(e.month.min(), e.month.max(), freq="M")}).merge(e, on="month", how="left")
    e_grid["dl_eur_usd"] = np.log(e_grid.eur_usd).diff()
    p = p.merge(e_grid[["month", "dl_eur_usd"]], on="month", how="left")
    dl = np.log(p.fx).groupby(p.iso3).diff()
    calm = {"usd": (dl.abs() <= PEG_STEP).astype(float).where(dl.notna()),
            "eur": ((dl - p.dl_eur_usd).abs() <= PEG_STEP).astype(float).where(dl.notna() & p.dl_eur_usd.notna())}
    share = {k: v.groupby(p.iso3).transform(lambda s: s.rolling(PEG_WINDOW, min_periods=PEG_MIN_MONTHS).mean())
             for k, v in calm.items()}
    p["peg_share12"] = np.fmax(share["usd"], share["eur"])

    # --- REER: отклонение от собственного среднего за 60 месяцев, с лагом публикации
    p = p.merge(reer[["iso3", "month", "reer120", "reer51", "neer120", "neer51"]], on=["iso3", "month"], how="left")
    gap, L = None, lags.reer_months
    for basket in ("120", "51"):  # основная корзина — 120 партнёров; 51 — история до 1993
        lr, ln = np.log(p[f"reer{basket}"]), np.log(p[f"neer{basket}"])
        trend = lr.groupby(p.iso3).transform(lambda s: s.rolling(REER_WINDOW, min_periods=REER_MIN_MONTHS).mean())
        # REER и его среднее — по месяц m − L; сдвиг номинального курса корзины за последние L месяцев
        g = (lr - trend).groupby(p.iso3).shift(L) + (ln - ln.groupby(p.iso3).shift(L))
        gap = g if gap is None else gap.fillna(g)
    p["reer_gap"] = gap.clip(-1, 1)

    # --- сырьевые условия торговли: изменение за 12 месяцев, с лагом публикации
    p = p.merge(ctot[["iso3", "month", "ctot"]], on=["iso3", "month"], how="left")
    lc = np.log(p.ctot)
    p["ctot_d12"] = (lc - lc.groupby(p.iso3).shift(12)).groupby(p.iso3).shift(lags.ctot_months).clip(-0.5, 0.5)

    # --- годовые источники: значение за последний вышедший год
    p["year"] = _known_year(p.month, lags.wdi_source)
    p = p.merge(buffers[["iso3", "year", "std_res", "res_months"]], on=["iso3", "year"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        p["gg_log"] = (-np.log(p.std_res.where(p.std_res >= 0) / 100.0)).clip(-3, 5)  # долга нет → потолок
    p["res_months_log"] = np.log1p(p.res_months.clip(0, 60))
    irr_year = _known_year(p.month, lags.irr_source)
    p["irr_extended"] = (irr_year > irr_last_year).astype(int)  # хроника кончилась: значение заморожено
    p["year"] = irr_year.clip(upper=irr_last_year)
    p = p.merge(irr[irr.year <= irr_last_year][["iso3", "year", "irr_fine"]], on=["iso3", "year"], how="left")
    p["irr_peg"] = (p.irr_fine <= PEG_MAX_CODE).astype(float).where(p.irr_fine.notna())

    p["reer_peg"] = p.reer_gap.clip(lower=0) * p.peg_share12
    raw = p[["iso3", "month", "peg_share12", "irr_peg", "gg_log", "res_months_log", "reer_gap", "reer_peg",
             "ctot_d12", "irr_extended"]]
    if not filled:
        return raw.reset_index(drop=True)
    out = fill(raw)
    return out[["iso3", "month", *COLUMNS, "irr_extended"]]


def fill(p: pd.DataFrame) -> pd.DataFrame:
    """Флаги пропусков и нули вместо пропусков. Идемпотентна: годится и для таблицы признаков,
    и для панели после `merge` (страна-периоды без строки в таблице — сплошные пропуски)."""
    p = p.copy()
    for src, flag in (("irr_peg", "irr_missing"), ("gg_log", "gg_missing"), ("res_months_log", "resm_missing"),
                      ("reer_gap", "reer_missing"), ("ctot_d12", "ctot_missing")):
        known_missing = p[flag].eq(1) if flag in p else False
        p[flag] = (p[src].isna() | known_missing).astype(int)
        p[src] = p[src].fillna(0.0)
    for c in ("peg_share12", "reer_peg"):
        p[c] = p[c].fillna(0.0)
    p["irr_extended"] = p.irr_extended.fillna(0).astype(int)
    return p


def annual_view(struct: pd.DataFrame) -> pd.DataFrame:
    """Декабрьские строки → iso3, year: то, что известно «на 1 января» следующего года."""
    d = struct[struct.month.dt.month == 12]
    return d.assign(year=d.month.dt.year).drop(columns="month").reset_index(drop=True)


def attach(p: pd.DataFrame, table: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """Приклеить структурные признаки к панели контура (по `on`), пропуски — флагами."""
    return fill(p.merge(table[[*on, *COLUMNS, "irr_extended"]], on=on, how="left"))


def load_struct(lags: StructLags = StructLags(), last_month: str = LAST_MONTH, filled: bool = True) -> pd.DataFrame:
    """Собрать таблицу из сырых источников (`sources/`). `last_month` — последний месяц курса
    (по умолчанию как у проверки вслепую; витрина берёт свой, более свежий)."""
    from seldon.sources.bruegel import load_reer
    from seldon.sources.imf import load_ctot, load_eur_anchor, load_monthly_fx
    from seldon.sources.irr import load_irr
    from seldon.sources.wdi import load_external_buffers

    raw = load_monthly_fx()
    fx = raw[raw.iso3.str.fullmatch(r"[A-Z]{3}") & (raw.month <= last_month)]  # без агрегатов (еврозона и т. п.)
    return build_struct(fx, load_eur_anchor(raw), load_reer(), load_ctot(), load_irr(), load_external_buffers(), lags,
                        filled=filled)
