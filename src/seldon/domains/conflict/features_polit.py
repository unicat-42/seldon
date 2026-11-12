"""Политические признаки для модели войн: лидеры, выборы, перевороты (REIGN) и протесты
(Mass Mobilization).

Запуск:
    uv run python -m seldon features_polit

Читает features_pitf_country_year.parquet, добавляет колонки ниже и пишет
features_polit_country_year.parquet. Признаки года T известны к концу T (прогноз на T+1).

REIGN (помесячно, 1950–2021; берём состояние на декабрь T и суммы за год):
    leader_tenure_log  ln(1 + месяцев у власти текущего лидера)
    leader_age         возраст лидера
    leader_military    1, если лидер — военный по карьере
    leader_changes     сколько раз сменился лидер за год T
    coup_attempts      попыток переворота за год T (Powell–Thyne, вкл. успешные)
    coup_success       успешных переворотов за год T
    coup_attempts_5    попыток за T−4…T
    election_year      1, если в году T были выборы (любые)
    election_ahead     1, если в декабре T ожидаются выборы в ближайшие 6 месяцев (REIGN
                       `anticipation` — по расписанию, известно заранее)
    months_since_election_log  ln(1 + месяцев с последних выборов) на декабрь T
    reign_missing      1, если REIGN нет для страны-года (после 2021 — у всех)

Mass Mobilization (события, 1990–2020):
    protests_log       ln(1 + число протестов за год T)
    protests_violent   число протестов с насилием протестующих за год T
    protests_3_log     ln(1 + протестов за T−2…T)
    protests_large     протестов с ≥ 10 000 участников за год T
    mm_missing         1, если данных нет (до 1990, после 2020, страны вне базы)

Коды стран в обоих источниках — COW; перевод в Gleditsch–Ward — `core.entities.ccode_to_gw`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED
from seldon.sources.massmob import load_massmob
from seldon.sources.reign import load_reign

REIGN_FEATURES = ["leader_tenure_log", "leader_age", "leader_military", "leader_changes", "coup_attempts",
                  "coup_success", "coup_attempts_5", "election_year", "election_ahead",
                  "months_since_election_log", "reign_missing"]
MM_FEATURES = ["protests_log", "protests_violent", "protests_3_log", "protests_large", "mm_missing"]
POLIT_FEATURES = [*REIGN_FEATURES, *MM_FEATURES]


def reign_features() -> pd.DataFrame:
    r = load_reign()
    # состояние на декабрь: последняя строка года (при нескольких лидерах в месяце — последний)
    dec = r.groupby(["gw", "year"]).tail(1).set_index(["gw", "year"])
    out = pd.DataFrame(index=dec.index)
    out["leader_tenure_log"] = np.log1p(dec.tenure_months)
    out["leader_age"] = dec.age
    out["leader_military"] = dec.militarycareer
    out["election_ahead"] = dec.anticipation
    out["months_since_election_log"] = dec.lastelection  # в REIGN уже ln(1 + месяцев)
    g = r.groupby(["gw", "year"])
    out["leader_changes"] = g.leader.nunique() - 1
    out["coup_attempts"] = g.pt_attempt.sum()
    out["coup_success"] = g.pt_suc.sum()
    out["election_year"] = (g.election_now.sum() > 0).astype(int)
    out = out.reset_index().sort_values(["gw", "year"])
    out["coup_attempts_5"] = out.groupby("gw").coup_attempts.transform(lambda s: s.rolling(5, min_periods=1).sum())
    return out


def massmob_features() -> pd.DataFrame:
    m = load_massmob()
    m["violent"] = (m.protesterviolence == 1).astype(int)
    m["large"] = (pd.to_numeric(m.participants, errors="coerce") >= 10_000).astype(int)
    a = m.groupby(["gw", "year"]).agg(n=("protest", "size"), protests_violent=("violent", "sum"),
                                      protests_large=("large", "sum")).reset_index()
    # страны, которые в базе есть, но в этом году протестов не было — нули, не пропуски
    years = range(int(m.year.min()), int(m.year.max()) + 1)
    full = pd.MultiIndex.from_product([a.gw.unique(), years], names=["gw", "year"]).to_frame(index=False)
    a = full.merge(a, on=["gw", "year"], how="left").fillna(0).sort_values(["gw", "year"])
    a["protests_log"] = np.log1p(a.n)
    a["protests_3_log"] = np.log1p(a.groupby("gw").n.transform(lambda s: s.rolling(3, min_periods=1).sum()))
    return a.drop(columns="n")


def build() -> pd.DataFrame:
    f = pd.read_parquet(DATA_PROCESSED / "features_pitf_country_year.parquet")
    f = f.merge(reign_features(), on=["gw", "year"], how="left").merge(massmob_features(), on=["gw", "year"], how="left")
    f["reign_missing"] = f.leader_tenure_log.isna().astype(int)
    f["mm_missing"] = f.protests_log.isna().astype(int)
    for c in REIGN_FEATURES + MM_FEATURES:
        f[c] = f[c].fillna(0.0)
    return f


def main(argv: list[str]) -> int:
    f = build()
    f.to_parquet(DATA_PROCESSED / "features_polit_country_year.parquet", index=False)
    print(f"{len(f)} строк, {f.gw.nunique()} стран, {f.year.min()}–{f.year.max()}")
    print("без REIGN:", round(f.reign_missing.mean(), 3), "(до 2021:", round(f[f.year <= 2021].reign_missing.mean(), 3),
          ") без Mass Mob:", round(f.mm_missing.mean(), 3), "(1990–2020:", round(f[f.year.between(1990, 2020)].mm_missing.mean(), 3), ")")
    print(f[POLIT_FEATURES].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    return 0

