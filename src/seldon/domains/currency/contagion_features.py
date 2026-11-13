"""Валютное заражение по торговым связям и параллельный курс (чекпоинт 4.1).

Зачем. В чекпоинте 3.3 «пружина» мерилась отклонением REER от среднего за 60 месяцев: в
длинном окне смешаны медленная инфляция и курс, и острый обвал валюты главного партнёра
(рубль в ноябре–декабре 2014) в нём тонет. Здесь — короткие окна и только курсы: они
известны в реальном времени на конец месяца m.

Обозначения: dl_c(k) = ln fx_c(m) − ln fx_c(m − k) — обесценение валюты страны c к доллару
за k месяцев (плюс = подешевела); w_ij — доля партнёра j в торговле страны i.

Торговые веса (`trade_weights`)
    w_ij(Y) = торговля i с j (экспорт + импорт) за три года Y−2…Y / вся торговля i за те же
    годы. Берётся отчёт самой страны i; где его нет — зеркальный отчёт партнёра j.
    На строке месяца m год Y — последний вышедший (`imts_annual`: год Y — с 1 июля Y+1):
    в январе–июне года t это Y = t−2, с июля — Y = t−1. Текущий и будущие годы в веса не входят.

Курс партнёра
    свой курс МВФ; у стран с общей валютой — курс якоря (евро; до 1999 — марка) или 0 для
    долларизованных и самих США. Изменение за k месяцев обрезано до [−1; +1,5] в логарифмах,
    чтобы партнёр в гиперинфляции не задавал весь индекс. Партнёры без курса из суммы
    выпадают, веса остальных перенормируются; если покрыто меньше `MIN_COVER` торговли —
    пропуск.

Признаки, k ∈ {3, 6, 12} (`K_WINDOWS`):
    pshock{k}       Σ_j w_ij · dl_j(k) − dl_i(k): накопленное «недообесценение» к упавшим
                    партнёрам (плюс — партнёры подешевели сильнее меня, мой товар подорожал);
    pmax{k}         max по ключевым партнёрам (w_ij ≥ `KEY_SHARE`) величины dl_j(k) − dl_i(k),
                    не меньше нуля: главный партнёр уже рухнул, своя валюта ещё стоит;
    pshock_peg{k}   max(pshock{k}, 0) × peg_share12 — то же при живой привязке.
    pshock_ctot     max(pshock6, 0) × max(−ctot_d12, 0): партнёр упал И цены на свой экспорт
                    упали (общий сырьевой шок); `ctot_d12` — с лагом набора МВФ, как в 3.3.
    cont_missing    флаг пропуска (признаки тогда = 0).

Параллельный курс (`sources/parallel.py`; покрытие — 1946–1998 и 2019–2025, между ними дыра):
    par_prem        ln(1 + премия параллельного курса к официальному), обрезана до [−0,5; 3];
                    значение месяца m − 1 (`parallel_premium`);
    par_d6          изменение par_prem за 6 месяцев;
    par_missing     флаг пропуска.

Строка (страна, месяц m) зависит только от данных, известных к концу m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from seldon.core.entities import shared_currency_table
from seldon.core.temporal import last_available_year_in_month, monthly_lag

K_WINDOWS = (3, 6, 12)
WEIGHT_YEARS = 3     # окно усреднения торговых весов, лет
KEY_SHARE = 0.15     # «ключевой» партнёр: не меньше 15 % торговли
MIN_COVER = 0.5      # минимум покрытой курсами доли торговли
DL_CLIP = (-1.0, 1.5)
PAR_CLIP = (-0.5, 3.0)

CONT = [f"{n}{k}" for k in K_WINDOWS for n in ("pshock", "pmax", "pshock_peg")] + ["pshock_ctot"]
FLAGS = ["cont_missing"]
PAR_CONT, PAR_FLAGS = ["par_prem", "par_d6"], ["par_missing"]
COLUMNS = [*CONT, *FLAGS, *PAR_CONT, *PAR_FLAGS]


@dataclass(frozen=True)
class ContagionLags:
    """Лаги публикации. Значения по умолчанию — из `core/temporal.py`; другие — только для
    тестов (контроль «без лага»)."""
    trade_source: str | None = "imts_annual"   # None — веса по текущему году (утечка; для контроля)
    parallel_months: int = monthly_lag("parallel_premium")


def trade_weights(trade: pd.DataFrame, years: int = WEIGHT_YEARS) -> pd.DataFrame:
    """Веса партнёров по годам: iso3, partner, year, w. Вес года Y — по торговле Y−years+1…Y.
    Отчёт самой страны; где его нет за весь год — зеркальный отчёт партнёров."""
    own = trade.assign(total=trade.exports.fillna(0) + trade.imports.fillna(0))[["iso3", "partner", "year", "total"]]
    mirror = own.rename(columns={"iso3": "partner", "partner": "iso3"})
    reported = own[["iso3", "year"]].drop_duplicates().assign(has_own=True)
    mirror = mirror.merge(reported, on=["iso3", "year"], how="left")
    t = pd.concat([own, mirror[mirror.has_own.isna()][["iso3", "partner", "year", "total"]]], ignore_index=True)
    t = t[(t.total > 0) & (t.iso3 != t.partner)].groupby(["iso3", "partner", "year"], as_index=False).total.sum()
    # сумма за окно: строка (i, j, y) даёт вклад в веса лет y … y + years − 1
    parts = [t.assign(year=t.year + s) for s in range(years)]
    w = pd.concat(parts, ignore_index=True).groupby(["iso3", "partner", "year"], as_index=False).total.sum()
    w["w"] = w.total / w.groupby(["iso3", "year"]).total.transform("sum")
    return w[w.year <= int(trade.year.max())][["iso3", "partner", "year", "w"]]


def currency_moves(fx: pd.DataFrame, eur: pd.DataFrame, k: int) -> pd.DataFrame:
    """dl_c(k) по всем странам и месяцам: таблица month × iso3. У стран с общей валютой —
    движение якоря (евро / 0 для долларовых); США — 0."""
    wide = np.log(fx.pivot(index="month", columns="iso3", values="fx"))
    months = pd.period_range(wide.index.min(), wide.index.max(), freq="M")
    wide = wide.reindex(months)
    dl = wide - wide.shift(k)
    e = np.log(eur.set_index("month").eur_usd).reindex(months)
    d_eur = e - e.shift(k)
    reg = shared_currency_table(months.min().year, months.max().year)
    for iso, g in reg.groupby("iso3"):
        col = dl[iso] if iso in dl else pd.Series(np.nan, index=months)
        for cur, gy in g.groupby("fx_regime"):
            mask = np.isin(months.year, gy.year.to_numpy())
            if cur == "USD":
                col = col.mask(mask, 0.0)
            elif cur in ("EUR", "XOF", "XAF"):  # франк КФА привязан к евро (до 1999 — к франку, идём за маркой)
                col = col.mask(mask, d_eur)
        dl[iso] = col
    dl["USA"] = 0.0
    return dl.clip(*DL_CLIP).rename_axis(index="month", columns="iso3")


def build_contagion(fx: pd.DataFrame, eur: pd.DataFrame, trade: pd.DataFrame, struct: pd.DataFrame,
                    parallel: pd.DataFrame | None = None, lags: ContagionLags = ContagionLags(),
                    filled: bool = True) -> pd.DataFrame:
    """Месячная таблица: iso3, month, `COLUMNS`. Чистая функция.

    Вход: `fx` (iso3, month, fx — курс к доллару на конец месяца), `eur` (month, eur_usd),
    `trade` (iso3, partner, year, exports, imports), `struct` (iso3, month, peg_share12,
    ctot_d12, ctot_missing — таблица чекпоинта 3.3, уже с лагами), `parallel` (iso3, month, premium)."""
    w = trade_weights(trade)
    wide = fx.pivot(index="month", columns="iso3", values="fx")
    months = pd.period_range(wide.index.min(), wide.index.max(), freq="M")
    have_fx = wide.reindex(months).notna()
    known = pd.Series(months, index=months)
    w_year = (known.dt.year if lags.trade_source is None else last_available_year_in_month(lags.trade_source, known)).to_numpy()
    own = [c for c in wide.columns if c != "USA"]
    out = pd.DataFrame(index=pd.MultiIndex.from_product([own, months], names=["iso3", "month"])).reset_index()
    wy = {y: g.pivot(index="iso3", columns="partner", values="w") for y, g in w.groupby("year")}
    for k in K_WINDOWS:
        dl = currency_moves(fx, eur, k)
        cols = dl.columns
        shock = np.full((len(months), len(own)), np.nan)
        pmax = np.full((len(months), len(own)), np.nan)
        dl_own = dl.reindex(columns=own).to_numpy()
        for y in np.unique(w_year):
            if y not in wy:
                continue
            rows = np.where(w_year == y)[0]
            W = wy[y].reindex(index=own, columns=cols).fillna(0.0).to_numpy()   # страна × партнёр
            D = dl.to_numpy()[rows]                                             # месяц × партнёр
            ok = ~np.isnan(D)
            cover = ok.astype(float) @ W.T                                      # месяц × страна
            num = np.where(ok, D, 0.0) @ W.T
            with np.errstate(divide="ignore", invalid="ignore"):
                avg = np.where(cover >= MIN_COVER, num / cover, np.nan)
            shock[rows] = avg - dl_own[rows]
            key = W >= KEY_SHARE                                                # страна × партнёр
            for i in np.where(key.any(axis=1))[0]:
                best = np.nanmax(np.where(ok[:, key[i]], D[:, key[i]], -np.inf), axis=1)
                pmax[rows, i] = np.where(np.isfinite(best), best, np.nan) - dl_own[rows, i]
            no_key = ~key.any(axis=1)
            pmax[np.ix_(rows, np.where(no_key)[0])] = np.where(np.isnan(shock[np.ix_(rows, np.where(no_key)[0])]), np.nan, 0.0)
        for name, arr in ((f"pshock{k}", shock), (f"pmax{k}", np.clip(pmax, 0, None))):
            out[name] = pd.DataFrame(arr, index=months, columns=own).where(have_fx[own]).T.stack(future_stack=True).to_numpy()
        out.loc[out[f"pshock{k}"].isna(), f"pmax{k}"] = np.nan
    s = struct[["iso3", "month", "peg_share12", "ctot_d12", "ctot_missing"]]
    out = out.merge(s, on=["iso3", "month"], how="left")
    for k in K_WINDOWS:
        out[f"pshock_peg{k}"] = out[f"pshock{k}"].clip(lower=0) * out.peg_share12.fillna(0)
    loss = (-out.ctot_d12.where(out.ctot_missing != 1)).clip(lower=0).fillna(0)
    out["pshock_ctot"] = out.pshock6.clip(lower=0) * loss
    # параллельный курс
    out["par_prem"], out["par_d6"] = np.nan, np.nan
    if parallel is not None and len(parallel):
        p = parallel.assign(par=np.log1p(parallel.premium.clip(lower=-0.9)).clip(*PAR_CLIP))[["iso3", "month", "par"]]
        p = p.drop_duplicates(["iso3", "month"])
        L = lags.parallel_months
        a = p.assign(month=p.month + L).rename(columns={"par": "par_prem"})
        b = p.assign(month=p.month + L + 6).rename(columns={"par": "par_prev"})
        out = out.drop(columns=["par_prem", "par_d6"]).merge(a, on=["iso3", "month"], how="left").merge(b, on=["iso3", "month"], how="left")
        out["par_d6"] = out.par_prem - out.par_prev
    out = out[["iso3", "month", *CONT, *PAR_CONT]]
    return fill(out) if filled else out.reset_index(drop=True)


def fill(p: pd.DataFrame) -> pd.DataFrame:
    """Флаги пропусков и нули вместо пропусков. Идемпотентна."""
    p = p.copy()
    for cols, flag, lead in ((CONT, "cont_missing", "pshock3"), (PAR_CONT, "par_missing", "par_prem")):
        known_missing = p[flag].eq(1) if flag in p else False
        p[flag] = (p[lead].isna() | known_missing).astype(int)
        p[cols] = p[cols].fillna(0.0)
    return p[[c for c in p.columns if c not in COLUMNS] + COLUMNS]


def attach(p: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Приклеить признаки заражения к месячной панели; пропуски — флагами."""
    return fill(p.merge(table[["iso3", "month", *COLUMNS]], on=["iso3", "month"], how="left"))


def load_contagion(struct: pd.DataFrame, last_month: str, lags: ContagionLags = ContagionLags(), filled: bool = True) -> pd.DataFrame:
    """Собрать таблицу из сырых источников (`sources/`)."""
    from seldon.sources.imf import load_eur_anchor, load_monthly_fx, load_trade
    from seldon.sources.parallel import load_parallel

    raw = load_monthly_fx()
    fx = raw[raw.iso3.str.fullmatch(r"[A-Z]{3}") & (raw.month <= last_month)]
    return build_contagion(fx, load_eur_anchor(raw), load_trade(), struct, load_parallel(), lags, filled)
