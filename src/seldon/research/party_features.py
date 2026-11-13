"""Признаки «идеология и партия у руля» — два слоя (исследование 14.11.2026, вне боевого конвейера).

Живой слой — V-Dem v16 (`sources/vdem.py`, `PARTY_COLS`):
    pl_ideol        насколько власть оправдывает себя идеологией (v2exl_legitideol, ≈ −3…+3)
    pl_lead         опора на личность лидера — персонализм (v2exl_legitlead)
    pl_perf         опора на результаты: рост, порядок (v2exl_legitperf)
    pl_nat / pl_soc / pl_cons / pl_rel
                    характер идеологии — доля экспертов, назвавших её националистической /
                    социалистической-коммунистической / консервативно-реставрационной / религиозной,
                    умноженная на «вес идеологии» — pl_ideol, переведённый в 0…1 логистической
                    функцией (у власти без идеологии её характер почти не считается)
    pl_ideol_d2, pl_lead_d2   сдвиг за два года
    pl_turn         идеологический разворот: половина суммы модулей изменений пяти долей
                    характера идеологии за два года (0 — та же идеология, 1 — полная смена)
    pl_leader_new   сменился ли за последние два года глава исполнительной власти (тот из пары
                    «глава государства / глава правительства», у кого больше полномочий, `v2ex_hosw`)
    pl_turnover     была ли за последние два года смена власти на выборах (v2elturnhos / hog ≥ 1)
    pl_regime_new   сменился ли за последние два года политический режим (v2regidnr)
  Блок легитимации в v16 размечен по 2023 год; имя лидера, выборы и режим — по 2025.

Архивный слой:
  Global Leader Ideologies (`sources/gli.py`, 1945–2020):
    gl_left / gl_right / gl_none   экономическая идеология лидера (база сравнения — центрист)
    gl_switch       сменился ли идеологический лагерь у руля за последние два года
    gl_leader_new   сменился ли сам лидер за последние два года
  V-Party (`sources/vparty.py`, выборы 1970–2019), правящая партия:
    vp_popul / vp_antiplural / vp_lr   популизм, антиплюрализм, шкала левые–правые (больше — правее)
    vp_new          правящая партия сменилась на выборах не раньше двух лет назад

Три правила, которые сторожит `tests/test_party_study.py`:
  1. Лаг публикации (`core/temporal.py`): на строке года T стоят данные за T − 1, «смена за два
     года» — это сравнение T − 1 с T − 3 (или события T − 1 и T − 2). Кризис года T не может
     «вызвать» смену партии в признаках строки T. В месячной панели январь–март года t видят
     год t − 2, апрель–декабрь — t − 1.
  2. Все «изменения» считаются по календарю (нет строки за нужный год — пропуск).
  3. Архивная база кончается там, где кончается: после её последнего года признаки — пропуск,
     а строка — вне «чистого окна» (`*_ok` = False, `clean_window`). Нули и флаги пропуска
     ставятся только ВНУТРИ окна (`fill`), иначе флаг стал бы дамми «эпохи после 2020 года».
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from seldon.core.temporal import last_available_year_in_month, publication_lag
from seldon.sources.vdem import LEGIT_CHARACTER, LEGIT_LAST_YEAR
from seldon.sources.vparty import VPARTY_LAST_YEAR

CHANGE_YEARS = 2
LIVE_LEGIT = ["pl_ideol", "pl_lead", "pl_perf", "pl_nat", "pl_soc", "pl_cons", "pl_rel"]
LIVE_TURN = ["pl_ideol_d2", "pl_lead_d2", "pl_turn", "pl_leader_new", "pl_turnover", "pl_regime_new"]
LIVE_EDGE = ["pl_leader_new", "pl_turnover", "pl_regime_new"]  # размечены в V-Dem по 2025 год
ARCH_GLI = ["gl_left", "gl_right", "gl_none", "gl_switch", "gl_leader_new"]
ARCH_VP = ["vp_popul", "vp_antiplural", "vp_lr", "vp_new"]
ALL = [*LIVE_LEGIT, *LIVE_TURN, *ARCH_GLI, *ARCH_VP]
OK = {"live": "legit_ok", "gli": "gli_ok", "vparty": "vp_ok"}
# признаки набора и флаг пропуска, который `fill` ставит внутри чистого окна
LAYERS: dict[str, tuple[list[str], str]] = {
    "live": ([*LIVE_LEGIT, *LIVE_TURN], "pl_missing"),
    "gli": (ARCH_GLI, "gl_missing"),
    "vparty": (ARCH_VP, "vp_missing"),
}
EXTRA = ["democ", "pl_party_inst", "gl_camp", "gl_leader", "vp_party"]  # для описательных таблиц, в модели не идут


@dataclass(frozen=True)
class PartyLags:
    """Сколько лет назад от строки T лежит самое свежее известное значение (прогноз «на 1 января»)."""
    vdem: int = publication_lag("vdem_annual", "jan1")
    gli: int = publication_lag("gli_annual", "jan1")
    vparty: int = publication_lag("vparty_annual", "jan1")


def _prev(t: pd.DataFrame, cols: list[str], k: int, key: str = "gw") -> pd.DataFrame:
    """Значения `cols` за год − k той же страны, по календарю (нет такого года — пропуск)."""
    src = t[[key, "year", *cols]].assign(year=t.year + k)
    return t[[key, "year"]].merge(src, on=[key, "year"], how="left")[cols].set_axis(t.index)


def _changed(t: pd.DataFrame, col: str, years: int = CHANGE_YEARS) -> pd.Series:
    """1, если значение `col` менялось хотя бы раз за последние `years` лет (сравниваются соседние
    годы Y−j и Y−j−1); пропуск, если хоть одного из нужных лет нет."""
    out = pd.Series(0.0, index=t.index)
    cur = t[col]
    for j in range(1, years + 1):
        prev = _prev(t, [col], j)[col]
        out = out.where(cur.notna() & prev.notna()) if j == 1 else out.where(prev.notna())
        out = out.where(~(out.notna() & (cur != prev)), 1.0)
        cur = prev
    return out


def live_table(v: pd.DataFrame) -> pd.DataFrame:
    """Живой слой по собственным годам V-Dem (без лага). Чистая функция: строка года Y зависит
    только от значений за Y, Y − 1 и Y − 2."""
    v = v.sort_values(["gw", "year"]).reset_index(drop=True)
    t = v[["gw", "year"]].copy()
    t["pl_ideol"], t["pl_lead"], t["pl_perf"] = v.v2exl_legitideol, v.v2exl_legitlead, v.v2exl_legitperf
    weight = 1 / (1 + np.exp(-v.v2exl_legitideol))
    for name, col in (("pl_nat", 0), ("pl_soc", 1), ("pl_cons", 2), ("pl_rel", 4)):
        t[name] = v[LEGIT_CHARACTER[col]] * weight
    prev = _prev(v, ["v2exl_legitideol", "v2exl_legitlead", *LEGIT_CHARACTER], CHANGE_YEARS)
    t["pl_ideol_d2"] = v.v2exl_legitideol - prev.v2exl_legitideol
    t["pl_lead_d2"] = v.v2exl_legitlead - prev.v2exl_legitlead
    t["pl_turn"] = 0.5 * (v[LEGIT_CHARACTER] - prev[LEGIT_CHARACTER]).abs().sum(axis=1, min_count=len(LEGIT_CHARACTER))
    hos_rules = v.v2exnamhog.isna() | (v.v2ex_hosw.fillna(1.0) >= 0.5)
    v = v.assign(_leader=v.v2exnamhos.where(hos_rules, v.v2exnamhog),
                 _turn=(v.v2elturnhos.where(hos_rules, v.v2elturnhog) >= 1).astype(float))
    t["pl_leader_new"] = _changed(v, "_leader")
    t["pl_regime_new"] = _changed(v, "v2regidnr")
    t["pl_turnover"] = np.maximum(v._turn, _prev(v, ["_turn"], 1)._turn)  # пропуск, если прошлого года нет
    t["pl_party_inst"], t["democ"] = v.v2xps_party, v.v2x_polyarchy
    t["legit_ok"] = t[LIVE_LEGIT].notna().all(axis=1) & (t.year <= LEGIT_LAST_YEAR)
    t.loc[~t.legit_ok, [*LIVE_LEGIT, "pl_ideol_d2", "pl_lead_d2", "pl_turn"]] = np.nan
    return t


def gli_table(g: pd.DataFrame) -> pd.DataFrame:
    """Global Leader Ideologies по собственным годам (без лага)."""
    g = g.sort_values(["gw", "year"]).reset_index(drop=True)
    t = g[["gw", "year"]].copy()
    known = g.camp.notna()
    for name, camp in (("gl_left", "leftist"), ("gl_right", "rightist"), ("gl_none", "none")):
        t[name] = (g.camp == camp).astype(float).where(known)
    t["gl_switch"] = _changed(g, "camp")
    t["gl_leader_new"] = _changed(g, "leader")
    t["gl_camp"], t["gl_leader"] = g.camp, g.leader
    t["gli_ok"] = known
    t.loc[~known, ARCH_GLI] = np.nan
    return t


def vparty_table(r: pd.DataFrame, last_year: int = VPARTY_LAST_YEAR) -> pd.DataFrame:
    """Правящая партия на каждый год: значения года выборов тянутся до следующих выборов, но не
    дальше `last_year` (конец базы). `r` — таблица `sources.vparty.ruling_party`."""
    frames = []
    for gw, e in r.sort_values(["gw", "year"]).groupby("gw"):
        e = e[e.year <= last_year]
        if e.empty:
            continue
        changed = (e.party != e.party.shift(1)) & e.party.notna() & e.party.shift(1).notna()
        e = e.assign(_changed_at=e.year.where(changed))
        grid = pd.DataFrame({"gw": gw, "year": range(int(e.year.min()), last_year + 1)}).merge(e, on=["gw", "year"], how="left")
        grid["_election"] = grid.year.where(grid.year.isin(e.year))
        cols = ["party", "v2xpa_popul", "v2xpa_antiplural", "v2pariglef", "_changed_at"]
        last_election = grid._election.ffill()
        for c in cols:  # тянем значение ПОСЛЕДНИХ выборов (если там правящей партии нет — пропуск, а не старая партия)
            grid[c] = grid.year.map(e.set_index("year")[c]).where(grid._election.notna())
            grid[c] = grid.groupby(last_election)[c].transform("first")
        frames.append(grid)
    g = pd.concat(frames, ignore_index=True)
    t = g[["gw", "year"]].copy()
    t["vp_popul"], t["vp_antiplural"], t["vp_lr"] = g.v2xpa_popul, g.v2xpa_antiplural, g.v2pariglef
    t["vp_new"] = (g.year - g._changed_at < CHANGE_YEARS).astype(float)
    t["vp_party"] = g.party
    t["vp_ok"] = t[["vp_popul", "vp_antiplural", "vp_lr"]].notna().all(axis=1)
    t.loc[~t.vp_ok, ARCH_VP] = np.nan
    return t


def own_year_tables(v: pd.DataFrame, g: pd.DataFrame, r: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Три таблицы признаков по собственным годам источников (без лага публикации)."""
    return {"live": live_table(v), "gli": gli_table(g), "vparty": vparty_table(r)}


def _finish(f: pd.DataFrame) -> pd.DataFrame:
    for ok in OK.values():
        f[ok] = f[ok].fillna(False).astype(bool)
    return f


def build_annual(v: pd.DataFrame, g: pd.DataFrame, r: pd.DataFrame, lags: PartyLags = PartyLags()) -> pd.DataFrame:
    """Годовая таблица признаков на СТРОКАХ прогноза: gw, year = T, значения источников за
    T − лаг. Пропуски не заполнены; `legit_ok`, `gli_ok`, `vp_ok` — есть ли слой на строке."""
    tabs = own_year_tables(v, g, r)
    out = None
    for name, lag in (("live", lags.vdem), ("gli", lags.gli), ("vparty", lags.vparty)):
        t = tabs[name].assign(year=tabs[name].year + lag)
        out = t if out is None else out.merge(t, on=["gw", "year"], how="outer")
    return _finish(out.sort_values(["gw", "year"]).reset_index(drop=True))


def attach_monthly(p: pd.DataFrame, tabs: dict[str, pd.DataFrame], key: str = "gw") -> pd.DataFrame:
    """Признаки на месячную панель (`key`, month): месяц видит последний год, вышедший к его
    концу (`last_available_year_in_month`). `tabs` — `own_year_tables` (ключ `gw`) или они же с
    ключом `iso3`."""
    out = p.copy()
    for name, source in (("live", "vdem_annual"), ("gli", "gli_annual"), ("vparty", "vparty_annual")):
        out["_y"] = last_available_year_in_month(source, out.month)
        t = tabs[name].rename(columns={"year": "_y"})
        out = out.merge(t, on=[key, "_y"], how="left").drop(columns="_y")
    out.index = p.index
    return _finish(out)


def clean_window(f: pd.DataFrame, layer: str, period: str = "year") -> pd.DataFrame:
    """Чистое окно слоя: строки не позже последнего периода, в котором слой вообще есть.
    Внутри окна пропуски остаются (это страны без разметки — их закрывает `fill`), после конца
    базы строк нет совсем."""
    ok = OK[layer]
    return f[f[period] <= f.loc[f[ok], period].max()]


def fill(f: pd.DataFrame, layer: str, period: str = "year") -> pd.DataFrame:
    """Заполнить пропуски слоя ВНУТРИ чистого окна: числовые — медианой своего периода (её
    считаем только по строкам того же года или месяца — будущего в ней нет), признаки-события —
    нулём; флаг `<слой>_missing` = 1. Вызывать только после `clean_window`."""
    cols, flag = LAYERS[layer]
    f = f.copy()
    f[flag] = (~f[OK[layer]]).astype(int)
    for c in cols:
        binary = c in ("pl_leader_new", "pl_turnover", "pl_regime_new", "gl_left", "gl_right", "gl_none", "gl_switch", "gl_leader_new", "vp_new")
        f[c] = f[c].fillna(0.0) if binary else f[c].fillna(f.groupby(period)[c].transform("median")).fillna(0.0)
    return f
