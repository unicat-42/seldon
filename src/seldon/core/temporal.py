"""Время: контракт `as_of`, расширяющееся окно проверки вслепую, лаги публикации (`available_from`).

Контракт `as_of` (инвариант 1 в CLAUDE.md). У каждого контура есть функция
`forecast_asof(...)`: прогноз в точке `as_of` считается только по данным ≤ `as_of` и не
меняется, если испортить все данные после неё. Это сторожит `tests/test_no_leakage.py`.

Метка известна не сразу (инвариант 2). Если метка строки t смотрит в t+1…t+h, то в момент
`as_of` учиться можно только на строках t + h ≤ `as_of` — это `label_known`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pandas as pd


def label_known(period, as_of, horizon: int):
    """Маска строк, чья метка (окно t+1…t+horizon) к моменту `as_of` уже целиком известна.
    Работает и для целых лет, и для месяцев (`pd.Period`)."""
    return period + horizon <= as_of


def expanding_origin_split(period: pd.Series, origins: Iterable, horizon: int) -> Iterator[tuple]:
    """Проверка вслепую расширяющимся окном. Для каждой точки отсчёта T отдаёт
    (T, маска обучения, маска теста): обучение — строки с известной к T меткой
    (t + horizon ≤ T), тест — строки периода T (их признаки → прогноз на T+1…T+horizon)."""
    for T in origins:
        yield T, label_known(period, T, horizon), period == T


# ---------------------------------------------------------------------------
# Лаги публикации (чекпоинт 2.2)
#
# Данные за период становятся известны не в его конце, а позже. Таблица ниже — когда
# значение за год Y считается доступным: (сколько лет прибавить к Y, месяц, день).
# Лаги консервативные и одинаковые для всех стран; это допущение, а не расписание релизов.
# ---------------------------------------------------------------------------

AVAILABLE_FROM: dict[str, tuple[int, int, int]] = {
    # годовые макроданные (ВВП, счёт текущих операций, госдолг, инфляция, деньги, кредит)
    # за год Y — с 1 апреля Y+1
    "gmd_annual": (1, 4, 1),
    "wdi_annual": (1, 4, 1),
    "bis_credit_annual": (1, 4, 1),
    "jst_annual": (1, 4, 1),
    # курс на конец декабря Y — с 1 января Y+1
    "imf_fx_december": (1, 1, 1),
    # список сторон конфликтов UCDP ACD за год Y — после окончания Y
    "ucdp_acd": (1, 1, 1),
}

# Точки, в которых делается годовой прогноз с признаков года T: (месяц, день) года T+1.
# Основная — 1 января; 1 апреля — проверка чувствительности (макроданные за T уже вышли).
ORIGINS: dict[str, tuple[int, int]] = {"jan1": (1, 1), "apr1": (4, 1)}


def available_from(source: str, year: int) -> pd.Timestamp:
    """Дата, начиная с которой значение источника `source` за год `year` считается известным."""
    dy, month, day = AVAILABLE_FROM[source]
    return pd.Timestamp(year=int(year) + dy, month=month, day=day)


def last_available_year(source: str, as_of: pd.Timestamp) -> int:
    """Последний год, за который значение источника известно на дату `as_of`."""
    year = as_of.year
    while available_from(source, year) > as_of:
        year -= 1
    return year


def publication_lag(source: str, origin: str = "jan1") -> int:
    """На сколько лет назад от T лежит самое свежее доступное значение источника, если
    прогноз с признаков года T делается в точке `origin` года T+1. Пример: макроданные GMD
    при прогнозе «на 1 января» — лаг 1 (известен только год T−1), «на 1 апреля» — 0."""
    month, day = ORIGINS[origin]
    T = 2000  # лаг от конкретного года не зависит
    return T - last_available_year(source, pd.Timestamp(year=T + 1, month=month, day=day))


def last_available_year_in_month(source: str, month: pd.Series) -> pd.Series:
    """Для каждого месяца (колонка `pd.Period`) — последний год, за который годовое значение
    источника известно к КОНЦУ этого месяца. Пример: годовой ИПЦ ВБ в январе–марте года t —
    это год t−2, с апреля — t−1."""
    dy, avail_month, _ = AVAILABLE_FROM[source]
    return month.dt.year - dy - (month.dt.month < avail_month).astype(int)
