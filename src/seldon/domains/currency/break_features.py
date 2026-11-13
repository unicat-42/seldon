"""Разрыв привязки: «кто отпустит курс разом» (чекпоинт 4.1, раздел 12 отчёта).

Зачем. Накопленный шок партнёров (раздел 11) у Казахстана-2015 держится, но не отличает его
от Киргизии или Армении с тем же разрывом к рублю: те сползали плавно. Гипотеза: разом
отпускают те, кто держит привязку, теряя сырьевой доход и резервы, — привязка × сырьё × резервы.

Сырьевой индекс МВФ (`sources/imf.py: load_ctot`, скользящие веса прошлых лет) — «чистый
экспорт сырья, взвешенный по доле в ВВП»: изменение его логарифма ≈ потеря или выигрыш дохода
в долях ВВП. Поэтому размах его колебаний — мера сырьевой зависимости страны.

`ctot_table` (месяц m — значение месяца m − лаг, `imf_ctot`):
    ctot_loss12   max(−Δln индекса за 12 месяцев, 0) — потеря дохода за год;
    ctot_loss24   ln максимума индекса за 24 месяца − ln индекса: накопленная потеря от пика,
                  которая ещё не отыграна (уровень, а не изменение);
    ctot_vol60    стандартное отклонение месячных Δln за 60 месяцев × √12 — сырьевая зависимость.
`add_break` — взаимодействия на месячной панели (всё на строке m — по данным ≤ m):
    peg_loss12, peg_loss24   peg_share12 × потеря дохода;
    peg_vol                  peg_share12 × ctot_vol60: сырьевой экспортёр/импортёр на привязке;
    peg_resfall              peg_share12 × max(−рост резервов за 12 месяцев, 0): привязку
                             защищают резервами;
    peg_gap_loss             pgap_peg24 × ctot_loss24: партнёры подешевели И сырьё упало И привязка;
    brk_missing              флаг: сырьевого индекса нет (признаки тогда = 0).
Суффикс `_rt` — проверка чувствительности: сырьевой индекс с лагом 1 месяц вместо 6 (биржевые
цены видны сразу, задерживается только набор МВФ). В основной постановке не используется.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon.core.temporal import monthly_lag

LOSS_WINDOW, VOL_WINDOW, VOL_MIN = 24, 60, 36
RT_LAG = 1
CTOT_COLS = ["ctot_loss12", "ctot_loss24", "ctot_vol60"]
BRK = ["ctot_loss12", "ctot_loss24", "ctot_vol60", "peg_loss12", "peg_loss24", "peg_vol", "peg_resfall", "peg_gap_loss"]
BRK_FLAGS = ["brk_missing"]
BRK_RT = [f"{c}_rt" for c in BRK]
BRK_RT_FLAGS = ["brk_missing_rt"]


def ctot_table(ctot: pd.DataFrame, lag: int = monthly_lag("imf_ctot")) -> pd.DataFrame:
    """iso3, month, `CTOT_COLS` — с лагом публикации `lag` месяцев. Чистая функция."""
    parts = []
    for iso, g in ctot[ctot.ctot > 0].groupby("iso3"):
        s = np.log(g.set_index("month").ctot.sort_index())
        s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
        d = s.diff()
        t = pd.DataFrame({
            "ctot_loss12": (-(s - s.shift(12))).clip(0, 0.5),
            "ctot_loss24": (s.rolling(LOSS_WINDOW + 1, min_periods=12).max() - s).clip(0, 0.5),
            "ctot_vol60": (d.rolling(VOL_WINDOW, min_periods=VOL_MIN).std() * np.sqrt(12)).clip(0, 0.5),
        })
        t.index = t.index + lag   # значение месяца m − lag стоит на строке m
        parts.append(t.rename_axis("month").reset_index().assign(iso3=iso))
    out = pd.concat(parts, ignore_index=True)
    return out[["iso3", "month", *CTOT_COLS]]


def add_break(p: pd.DataFrame, table: pd.DataFrame, sfx: str = "") -> pd.DataFrame:
    """Приклеить `ctot_table` к месячной панели и посчитать взаимодействия. Нужны колонки
    `peg_share12`, `reserves_g12`, `pgap_peg24`. `sfx="_rt"` — вариант чувствительности."""
    t = table.rename(columns={c: f"{c}{sfx}" for c in CTOT_COLS})
    p = p.merge(t, on=["iso3", "month"], how="left")
    loss12, loss24, vol = (p[f"{c}{sfx}"] for c in CTOT_COLS)
    p[f"brk_missing{sfx}"] = loss12.isna().astype(int)
    peg = p.peg_share12.fillna(0)
    p[f"peg_loss12{sfx}"], p[f"peg_loss24{sfx}"], p[f"peg_vol{sfx}"] = peg * loss12, peg * loss24, peg * vol
    p[f"peg_resfall{sfx}"] = peg * (-p.reserves_g12).clip(lower=0).fillna(0)
    p[f"peg_gap_loss{sfx}"] = p.pgap_peg24.fillna(0) * loss24
    cols = [f"{c}{sfx}" for c in BRK]
    p[cols] = p[cols].fillna(0.0)
    return p
