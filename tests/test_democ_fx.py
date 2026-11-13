"""Исследование «уровень демократии в месячной валютной модели» (`research/democ_fx.py`).

Запуск:
    uv run pytest tests/test_democ_fx.py

Главный вопрос исследования — не знание ли это задним числом: V-Dem каждый год пересчитывает
всю историю. Поэтому признак берётся из того релиза, который существовал на дату прогноза.
Проверки (синтетические таблицы, без диска):

1. `test_release_schedule` — какой релиз известен: обучение в декабре T−1 и январь–март года T
   видят релиз года T−1, апрель–декабрь года T — релиз года T; до первого сохранённого релиза
   (2020) — он же, после последнего — последний.
2. `test_attach_uses_only_releases_known_at_the_time` — порча всех релизов, вышедших ПОСЛЕ года
   T, панель года T не меняет; порча релиза года T меняет только апрель–декабрь года T, а
   обучающие строки и январь–март — нет.
3. `test_attach_respects_publication_lag` — январь–март года T получают данные за T−2,
   апрель–декабрь — за T−1; значения берутся из нужного релиза.
4. `test_forecast_is_future_blind` — прогноз на месяцы года T не меняется, если испортить
   панель после декабря T−1 (кроме самих признаков тестовых строк) и все более поздние релизы.
5. `test_latest_mode_ignores_vintages` — режим «сегодняшняя редакция» читает только последний релиз.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon.research import democ_fx as dx

RELEASES = range(2020, 2027)
ISO = [f"C{i:02d}" for i in range(12)]


def _tables(seed: int = 0) -> dict[int, pd.DataFrame]:
    """Релиз года R: данные по R − 1; каждая редакция чуть отличается от других."""
    rng = np.random.default_rng(seed)
    base = {(i, y): rng.random() for i in ISO for y in range(1990, 2026)}
    out = {}
    for r in RELEASES:
        rows = [{"iso3": i, "year": y, "democ": float(np.clip(v + rng.normal(0, 0.02), 0, 1))} for (i, y), v in base.items() if y <= r - 1]
        out[r] = pd.DataFrame(rows)
    return out


def _spoil(tables: dict[int, pd.DataFrame], releases) -> dict[int, pd.DataFrame]:
    return {r: (t.assign(democ=t.democ * 0 + 0.123 * (r % 7 + 1) % 1) if r in releases else t) for r, t in tables.items()}


def _months(first: str = "2015-01", last: str = "2024-12") -> pd.DataFrame:
    m = pd.period_range(first, last, freq="M")
    return pd.DataFrame([(i, x) for i in ISO for x in m], columns=["iso3", "month"])


def test_release_schedule():
    assert dx.release_year(2023, late=False) == 2022 and dx.release_year(2023, late=True) == 2023
    assert dx.release_year(2010, late=False) == 2020 and dx.release_year(2010, late=True) == 2020  # раньше релизов нет
    assert dx.release_year(2021, late=False) == 2020
    assert dx.release_year(2030, late=True) == 2026
    assert dx.REALTIME_FROM == 2021  # первый год, где и обучение, и тест — на релизах своего времени


def test_attach_uses_only_releases_known_at_the_time():
    t, p, T = _tables(), _months(), 2022
    base = dx.attach_democ(p, t, T)
    later = dx.attach_democ(p, _spoil(t, [r for r in RELEASES if r > T]), T)
    upto = base.month.dt.year <= T
    assert np.allclose(base.democ[upto], later.democ[upto], equal_nan=True)
    bad = dx.attach_democ(p, _spoil(t, [T]), T)
    late = (base.month.dt.year == T) & (base.month.dt.month >= 4)
    before = base.month < pd.Period(f"{T}-04", freq="M")
    assert np.allclose(base.democ[before], bad.democ[before], equal_nan=True)
    assert not np.allclose(base.democ[late], bad.democ[late], equal_nan=True)


def test_attach_respects_publication_lag():
    t, p, T = _tables(1), _months(), 2023
    a = dx.attach_democ(p, t, T).set_index(["iso3", "month"]).democ
    old, new = t[2022].set_index(["iso3", "year"]).democ, t[2023].set_index(["iso3", "year"]).democ
    assert np.isclose(a[("C03", pd.Period("2023-02", freq="M"))], old[("C03", 2021)])
    assert np.isclose(a[("C03", pd.Period("2023-06", freq="M"))], new[("C03", 2022)])
    assert np.isclose(a[("C03", pd.Period("2019-06", freq="M"))], old[("C03", 2018)])  # обучение — редакция декабря T−1


def _panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = _months("2005-01", "2024-12")
    n = len(p)
    p["x1"], p["x2"] = rng.normal(size=n), rng.normal(size=n)
    p["y"] = (rng.random(n) < 0.08).astype(float)
    p["evaluable"], p["own_currency"], p["fx"] = True, True, 1.0
    p["gdp_pc_log"], p["gdp_pc_missing"] = rng.normal(size=n), 0
    return p


def test_forecast_is_future_blind():
    t, p, T, h = _tables(2), _panel(), 2022, 12
    fam = {"klr": ["x1", "x2"]}
    base = dx.predict_year(p, t, T, h, families=fam)
    bad = p.copy()
    fut = bad.month > pd.Period(f"{T - 1}-12", freq="M")
    bad.loc[fut, "y"] = 1 - bad.loc[fut, "y"]
    after = bad.month.dt.year > T
    bad.loc[after, ["x1", "x2", "gdp_pc_log"]] = 1e6
    again = dx.predict_year(bad, _spoil(t, [r for r in RELEASES if r > T]), T, h, families=fam)
    cols = [c for c in base.columns if "__" in c]
    assert cols and base[cols].notna().all().all()
    assert np.allclose(base[cols].to_numpy(), again[cols].to_numpy(), atol=1e-12)
    # контроль: порча релиза, вышедшего в году T, прогноз апреля–декабря меняет
    ctrl = dx.predict_year(p, _spoil(t, [T]), T, h, families=fam)
    late = (base.month.dt.month >= 4).to_numpy()
    assert not np.allclose(base.loc[late, "klr__democ"].to_numpy(), ctrl.loc[late, "klr__democ"].to_numpy())
    assert np.allclose(base.loc[~late, "klr__democ"].to_numpy(), ctrl.loc[~late, "klr__democ"].to_numpy(), atol=1e-12)


def test_latest_mode_ignores_vintages():
    t, p, T = _tables(3), _months(), 2022
    a = dx.attach_democ(p, t, T, vintage=False)
    b = dx.attach_democ(p, _spoil(t, [r for r in RELEASES if r < max(RELEASES)]), T, vintage=False)
    assert np.allclose(a.democ, b.democ, equal_nan=True)
    last = t[max(RELEASES)].set_index(["iso3", "year"]).democ
    assert np.isclose(a.set_index(["iso3", "month"]).democ[("C05", pd.Period("2022-06", freq="M"))], last[("C05", 2021)])
