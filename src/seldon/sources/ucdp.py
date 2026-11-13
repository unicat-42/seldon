"""UCDP: события GED (Georeferenced Event Dataset), оперативные Candidate Events и список сторон
конфликтов ACD (UCDP/PRIO Armed Conflict Dataset). Скачивание: `download ucdp`,
`download ucdp_candidate`.

GED — годовой выверенный релиз (26.1 — события по декабрь 2025). Candidate Events — тот же
формат, но предварительный: файл за месяц m выкладывается 20-го числа месяца m+1 (проверено
по датам файлов 2020–2026: `GEDEvent_v26_0_9.csv`, сентябрь 2026, — 20 октября 2026), раз в
квартал — накопленный файл с начала года (`GEDEvent_v26_01_26_09.csv`). Лаг публикации
месячных событий — `core/temporal.py`, `ucdp_events` (1 месяц)."""

from __future__ import annotations

import pandas as pd

from seldon import DATA_RAW

UCDP_VERSION = "26.1"  # релиз 26.1 покрывает события по 2025 год включительно
_V = UCDP_VERSION.replace(".", "_")
YEAR_FROM, YEAR_TO = 1989, 2025  # GED покрывает 1989–2025
GED_FILE = DATA_RAW / "ucdp" / f"GEDEvent_v{_V}.csv"
ACD_FILE = DATA_RAW / "ucdp" / f"UcdpPrioConflict_v{_V}.csv"

# Candidate Events: накопленные квартальные файлы «январь … месяц» (имя — как на сайте UCDP).
# Текущий год — продолжение GED; прошлый — только для сверки «предварительное против выверенного».
CANDIDATE_DIR = DATA_RAW / "ucdp_candidate"
CANDIDATE_LIVE = "GEDEvent_v26_01_26_09.csv"  # январь–сентябрь 2026 (выложен 20.10.2026)
CANDIDATE_PREV = "GEDEvent_v25_01_25_12.csv"  # январь–декабрь 2025 (выложен 20.01.2026)
CANDIDATE_LAST_MONTH = pd.Period("2026-09", freq="M")  # последний месяц, покрытый событиями
EVENT_COLS = ["country", "country_id", "year", "date_start", "type_of_violence", "best", "conflict_new_id",
              "side_a", "side_b", "gwnoa", "gwnob"]


def load_ged() -> pd.DataFrame:
    """События GED: страна (код GW), год, месяц, тип насилия, погибшие (`best`), конфликт."""
    cols = ["country", "country_id", "year", "date_start", "type_of_violence", "best", "conflict_new_id"]
    g = pd.read_csv(GED_FILE, usecols=cols)
    g["date_start"] = pd.to_datetime(g["date_start"])
    g["month"] = g["date_start"].dt.month
    g["deaths_state"] = g["best"].where(g["type_of_violence"] == 1, 0)
    return g.rename(columns={"country_id": "gw"})


def _read_events(path, cols: list[str]) -> pd.DataFrame:
    g = pd.read_csv(path, usecols=cols, low_memory=False)
    g["date_start"] = pd.to_datetime(g["date_start"])
    g["month"] = g["date_start"].dt.month
    g["deaths_state"] = g["best"].where(g["type_of_violence"] == 1, 0)
    return g.rename(columns={"country_id": "gw"})


def load_candidate(name: str = CANDIDATE_LIVE) -> pd.DataFrame:
    """Candidate Events (предварительные события) — те же колонки, что у `load_events`.
    Год и месяц берутся из даты начала события; события вне года файла отбрасываются."""
    c = _read_events(CANDIDATE_DIR / name, EVENT_COLS)
    c["year"] = c.date_start.dt.year
    file_year = 2000 + int(name.split("_v")[1][:2])
    return c[c.year == file_year].assign(source="candidate").reset_index(drop=True)


def load_events(candidate: bool = True) -> pd.DataFrame:
    """События для месячного контура войн: выверенный GED по `YEAR_TO` включительно плюс
    Candidate Events за следующие месяцы (по `CANDIDATE_LAST_MONTH`). Колонка `source`:
    `ged` / `candidate`. Кроме колонок `load_ged` — стороны события: `side_a`, `side_b` и коды
    государств-сторон `gwnoa`, `gwnob` (пусто, если сторона — не государство)."""
    g = _read_events(GED_FILE, EVENT_COLS).assign(source="ged")
    if not candidate:
        return g
    c = load_candidate()
    c = c[c.year > YEAR_TO]
    return pd.concat([g, c], ignore_index=True)


def load_acd(year_from: int = YEAR_FROM, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Конфликт-годы ACD: коды сторон (основные `gwno_a`/`gwno_b`, помогающие войсками
    `gwno_a_2nd`/`gwno_b_2nd`) и место `gwno_loc` — строками, как в файле. `extra` —
    дополнительные колонки (например, `incompatibility`: 1 — территория, 2 — власть, 3 — обе)."""
    cols = ["conflict_id", "year", "gwno_a", "gwno_b", "gwno_a_2nd", "gwno_b_2nd", "gwno_loc", *extra]
    a = pd.read_csv(ACD_FILE, usecols=cols)
    return a[a.year >= year_from]
