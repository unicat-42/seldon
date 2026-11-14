"""Вселенная валютных моделей — только государства (решение Ангелины 14.11.2026).

Потоки МВФ содержат не только государства: зависимые территории (Гонконг, Аруба, Джерси,
Гренландия…), агрегаты (`G163` — еврозона) и территории с чужой валютой (Косово). Они почти
никогда не девальвируют; пока они были в обучении и оценке, модель получала прибавку к AUC
просто за то, что узнавала их. Правило: валютные панели читаются только по государствам из
списка Gleditsch–Ward (`core.entities.STATE_ISO3`) — тем же, по которым публикуется таблица риска.

Проверки (потоки МВФ подменены синтетическими, диск не нужен):
1. `test_state_list` — кто в списке, кто нет.
2. `test_monthly_panel_reads_only_states` — `econ_kr.load_panel` по умолчанию отдаёт только
   государства; `states_only=False` — всё, что есть в потоке.
3. `test_december_fx_reads_only_states` — то же для декабрьского курса годового контура.
4. `test_full_monthly_fx_keeps_aggregates` — полный месячный курс (нужен для якоря евро и курсов
   торговых партнёров) агрегаты и территории НЕ теряет.
5. `test_showcase_uses_the_same_list` — таблица риска и модели смотрят на один список.
"""

from __future__ import annotations

import pandas as pd

from seldon.core import entities
from seldon.domains.currency import econ_kr
from seldon.publish import risk_table
from seldon.sources import imf

CODES = ["RUS", "USA", "BHS", "HKG", "G163", "KOS", "VUT"]
STATES = {"RUS", "USA", "BHS"}


def _flow(name: str) -> pd.DataFrame:
    months = pd.period_range("2019-01", "2021-12", freq="M")
    d = pd.DataFrame([(c, m, 1.0 + i + 0.01 * j) for i, c in enumerate(CODES) for j, m in enumerate(months)],
                     columns=["iso3", "month", "value"])
    d["TYPE_OF_TRANSFORMATION"], d["UNIT"] = "EOP_RT", "USD"
    d["INDICATOR"] = {"reserves": "RXF11_REVS", "interest_rates": econ_kr.MMRT}.get(name, "X")
    return d


def test_state_list():
    s = entities.STATE_ISO3
    assert {"RUS", "USA", "BHS", "BLZ", "BRN", "KAZ", "SRB"} <= s
    assert not s & {"HKG", "MAC", "ABW", "JEY", "GRL", "G163", "G309", "KOS", "CWX", "VUT", "SMR", None}
    assert entities.POST_SOVIET_ISO3 <= s
    d = pd.DataFrame({"iso3": CODES, "x": range(len(CODES))})
    assert set(entities.states_only(d).iso3) == STATES and list(entities.states_only(d).x) == [0, 1, 2]


def test_monthly_panel_reads_only_states(monkeypatch):
    monkeypatch.setattr(econ_kr, "read_imf", _flow)
    assert set(econ_kr.load_panel("2021-12").iso3) == STATES
    assert set(econ_kr.load_panel("2021-12", states_only=False).iso3) == set(CODES)


def test_december_fx_reads_only_states(monkeypatch):
    monkeypatch.setattr(imf, "read_imf", _flow)
    assert set(imf.load_dec_fx().iso3) == STATES
    assert set(imf.load_dec_fx(states_only=False).iso3) == set(CODES)


def test_full_monthly_fx_keeps_aggregates(monkeypatch):
    monkeypatch.setattr(imf, "read_imf", _flow)
    assert {"G163", "HKG"} <= set(imf.load_monthly_fx().iso3)


def test_showcase_uses_the_same_list():
    assert risk_table.GW_ISO is entities.STATE_ISO3
