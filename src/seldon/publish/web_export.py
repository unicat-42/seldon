"""Экспорт данных для веб-витрины: статические JSON в `web/data/` (Этап 5, чекпоинт 5.1).

Запуск (после `make all`):
    uv run python -m seldon web_export        # или `make web`

Сервера у сайта нет: фронтенд в `web/` читает только эти файлы. Экспортер ничего не
придумывает и не меняет боевые модели — он
  1. читает готовые таблицы: `reports/risk_2025_26.csv` (текущий срез), `reports/scorecard.csv`
     (качество вслепую), прогнозы вслепую из `data/processed/`, снимки `forecasts/*`;
  2. заново обучает ТЕ ЖЕ четыре логит-модели теми же функциями (`model_hist`, `econ_long`,
     `econ_kr`, `econ_csd`), сверяет вероятности с витриной (расхождение — ошибка, а не
     предупреждение) и снимает с них то, чего в таблицах нет: вклады признаков `β·z` в
     лог-шансах и 95 % интервал оценки (`publish/explain.py`);
  3. у каждого наблюдения ставит фактическую дату с учётом лагов публикации (`core/temporal.py`).

Правила честности (тесты — `tests/test_web_export.py`):
  - пропуск остаётся `null` и получает код причины (`status`), нулей вместо пропусков нет;
  - у деревьев (`fxa_extree`, `bank_jst_extree_full`) нет ни вкладов, ни интервала — только
    входные наблюдения с датами;
  - вклады — в лог-шансах; в процентные пункты вероятности они не переводятся.

Файлы:
    manifest.json        когда и из чего собрано (sha256 входных таблиц), коммит кода
    countries.json       174 государства: ISO3, русское и английское названия, варианты для поиска
    current.json         текущий срез: оценки, интервалы, ранги, статусы покрытия, контексты событий
    country/<ISO3>.json  досье: вклады признаков, входы деревьев, история проверок вслепую
    oos/<контур>.json    лента рангов проверки вслепую (война — начало, годовой валютный)
    oos/<контур>/<год>.json  вклады признаков исторических прогнозов этого выпуска
    scorecard.json       сводка качества с интервалами
    archive.json         снимки `forecasts/*`: опубликованные оценки и состояние проверки
    meta.json            источники, лаги публикации, описание метода интервалов
"""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from seldon import DATA_PROCESSED, PROJECT_ROOT, REPORTS
from seldon.core.entities import COUNTRY_ALIASES, COUNTRY_NAMES, GW_TO_ISO3, ISO3_TO_GW, POST_SOVIET_ISO3
from seldon.core.temporal import AVAILABLE_FROM, AVAILABLE_FROM_MONTHS, label_known, monthly_lag, publication_lag
from seldon.publish import vintage
from seldon.publish.explain import Explained, explain_logit

WEB = PROJECT_ROOT / "web"
OUT = WEB / "data"
TOL = 6e-5          # таблица риска округлена до 4 знаков
EMP_MONTHS = 18     # сколько последних месяцев давления отдаём в досье
CASES = [           # исторические кейсы главной: контур, страна, год выпуска (год признаков T)
    {"id": "blr-2015", "contour": "fxa", "iso3": "BLR", "release": 2014, "title": "Беларусь‑2015"},
    {"id": "aze-2020", "contour": "war25", "iso3": "AZE", "release": 2019, "title": "Азербайджан‑2020"},
    {"id": "rus-2014", "contour": "fxa", "iso3": "RUS", "release": 2013, "title": "Россия‑2014"},
]

# территории и агрегаты из данных МВФ, которые встречаются в исторической проверке валютных
# моделей, но не входят в 174 государства витрины
TERRITORY_RU = {
    "ABW": "Аруба", "AIA": "Ангилья", "ANT": "Нидерландские Антилы", "BMU": "Бермуды", "CUW": "Кюрасао",
    "CWX": "Кюрасао и Синт-Мартен", "CYM": "Каймановы острова", "FRO": "Фарерские острова", "GGY": "Гернси",
    "GIB": "Гибралтар", "GLP": "Гваделупа", "GRL": "Гренландия", "GUF": "Французская Гвиана", "HKG": "Гонконг",
    "IMN": "Остров Мэн", "JEY": "Джерси", "KOS": "Косово", "MAC": "Макао", "MSR": "Монтсеррат", "MTQ": "Мартиника",
    "NCL": "Новая Каледония", "PYF": "Французская Полинезия", "REU": "Реюньон", "SMR": "Сан-Марино",
    "SPM": "Сен-Пьер и Микелон", "STP": "Сан-Томе и Принсипи", "SXM": "Синт-Мартен", "SYC": "Сейшелы",
    "TON": "Тонга", "VUT": "Вануату", "WSM": "Самоа",
}

# ---------------------------------------------------------------------------
# Словарь признаков: подпись, единица, источник и правило даты наблюдения.
# `when` — имя правила из `_obs_date`; `nat` — как показать значение «по-человечески».
# ---------------------------------------------------------------------------

F = dict
FEATURE_META: dict[str, dict] = {
    # --- война: история (UCDP), год T ---
    "war_now": F(label="Государственная война в году наблюдения", unit="флаг", src="ucdp", when="T"),
    "years_since_war": F(label="Лет с последней войны", unit="лет", src="ucdp", when="T"),
    "never_war": F(label="Не воевала с 1989 года", unit="флаг", src="ucdp", when="T"),
    "log_deaths_1": F(label="Погибшие в государственном конфликте за год", unit="ln(1 + чел.)", src="ucdp", when="T", nat="expm1:чел."),
    "log_deaths_5": F(label="Погибшие за 5 лет", unit="ln(1 + чел.)", src="ucdp", when="T", nat="expm1:чел."),
    "log_deaths_10": F(label="Погибшие за 10 лет", unit="ln(1 + чел.)", src="ucdp", when="T", nat="expm1:чел."),
    "war_years_10": F(label="Лет с войной из последних 10", unit="лет", src="ucdp", when="T"),
    "decay_1": F(label="Память о войне, полураспад 1 год", unit="индекс 0–1", src="ucdp", when="T"),
    "decay_3": F(label="Память о войне, полураспад 3 года", unit="индекс 0–1", src="ucdp", when="T"),
    "nb_war_share": F(label="Доля соседей с войной на своей территории", unit="доля", src="ucdp", when="T"),
    "nb_war_any": F(label="Хотя бы один сосед воюет", unit="флаг", src="ucdp", when="T"),
    "nb_onset_any": F(label="У соседа началась война", unit="флаг", src="ucdp", when="T"),
    "nb_log_deaths_1": F(label="Погибшие на территории соседей за год", unit="ln(1 + чел.)", src="ucdp", when="T", nat="expm1:чел."),
    # --- война: режим (Polity5), смертность и население (ВБ) ---
    "part_autoc": F(label="Режим: частичная автократия", unit="флаг", src="polity", when="polity"),
    "part_dem_fact": F(label="Режим: частичная демократия с фракционностью", unit="флаг", src="polity", when="polity"),
    "part_dem_nofact": F(label="Режим: частичная демократия без фракционности", unit="флаг", src="polity", when="polity"),
    "full_dem": F(label="Режим: полная демократия", unit="флаг", src="polity", when="polity"),
    "transition": F(label="Режим: переход, оккупация или безвластие", unit="флаг", src="polity", when="polity"),
    "polity2": F(label="Индекс Polity (−10 автократия … +10 демократия)", unit="баллы", src="polity", when="polity"),
    "regime_missing": F(label="Флаг: режима нет в Polity", unit="флаг", src="polity", when="polity"),
    "regime_extended": F(label="Флаг: режим продлён с последнего года Polity", unit="флаг", src="polity", when="polity"),
    "log_imr_rel": F(label="Детская смертность к мировой медиане", unit="ln(отношения)", src="wdi", when="wdi:infant_mortality", nat="exp:× к медиане"),
    "imr_missing": F(label="Флаг: нет детской смертности", unit="флаг", src="wdi", when="wdi:infant_mortality"),
    "log_pop": F(label="Население", unit="ln(чел.)", src="wdi", when="wdi:population", nat="exp:чел."),
    "pop_missing": F(label="Флаг: нет населения", unit="флаг", src="wdi", when="wdi:population"),
    # --- годовой валютный: курс — декабрь T, макро — T − лаг ---
    "infl_log": F(label="Инфляция за год", unit="ln(1 + доля)", src="gmd", when="macro", nat="infl:%"),
    "infl_d2": F(label="Изменение инфляции за 2 года", unit="п.п. / 100", src="gmd", when="macro"),
    "deval_prev": F(label="Девальвация за год, декабрь к декабрю", unit="доля", src="imf_fx", when="dec", nat="pct:%"),
    "deval_5": F(label="Девальваций ≥ 25 % за последние 5 лет", unit="число", src="imf_fx", when="dec"),
    "gdp_growth_2": F(label="Средний рост реального ВВП за 2 года", unit="доля", src="gmd", when="macro", nat="pct:%"),
    "debt_d2": F(label="Изменение госдолга к ВВП за 2 года", unit="доля ВВП", src="gmd", when="macro", nat="pct:п.п. ВВП"),
    "ca_level": F(label="Счёт текущих операций к ВВП", unit="доля ВВП", src="gmd", when="macro", nat="pct:% ВВП"),
    "money_growth_2": F(label="Рост денежной массы сверх роста ВВП, 2 года", unit="доля", src="gmd", when="macro", nat="pct:%"),
    "world_deval": F(label="Доля стран мира с девальвацией ≥ 25 % за год", unit="доля", src="imf_fx", when="dec", nat="pct:%"),
    "peg_years_log": F(label="Лет без изменения декабрьского курса больше 2 %", unit="ln(1 + лет)", src="imf_fx", when="dec", nat="expm1:лет"),
    "infl_missing": F(label="Флаг: нет инфляции", unit="флаг", src="gmd", when="macro"),
    "gdp_missing": F(label="Флаг: нет роста ВВП", unit="флаг", src="gmd", when="macro"),
    "debt_missing": F(label="Флаг: нет госдолга", unit="флаг", src="gmd", when="macro"),
    "ca_missing": F(label="Флаг: нет счёта текущих операций", unit="флаг", src="gmd", when="macro"),
    "money_missing": F(label="Флаг: нет денежной массы", unit="флаг", src="gmd", when="macro"),
    # --- месячный валютный: курс — месяц m, резервы — m − лаг ---
    "reserves_g12": F(label="Изменение резервов за 12 месяцев", unit="доля", src="imf_res", when="res", nat="pct:%"),
    "rer_dev": F(label="Реальный курс к собственному тренду за 3 года", unit="доля", src="imf_fx_cpi", when="m", nat="pct:%"),
    "deval12": F(label="Изменение курса за 12 месяцев (рост — ослабление)", unit="доля", src="imf_fx", when="m", nat="pct:%"),
    "infl12": F(label="Инфляция за 12 месяцев", unit="доля", src="imf_cpi", when="m", nat="pct:%"),
    "real_rate": F(label="Реальная ставка", unit="п.п.", src="imf_rates", when="m"),
    "rate_diff_us": F(label="Разница ставок с США", unit="п.п.", src="imf_rates", when="m"),
    "spread": F(label="Кредитная ставка минус депозитная", unit="п.п.", src="imf_rates", when="m"),
    "reserves_g1": F(label="Изменение резервов за месяц", unit="доля", src="imf_res", when="res", nat="pct:%"),
    "m2_res_g12": F(label="Рост отношения денежной массы к резервам за 12 месяцев", unit="доля", src="imf_res", when="res", nat="pct:%"),
    "exports_g12": F(label="Изменение экспорта за 12 месяцев", unit="доля", src="imf_exports", when="m", nat="pct:%"),
    "world_crisis3": F(label="Доля стран мира с валютным кризисом за 3 месяца", unit="сумма долей", src="imf_res", when="res"),
    "emp_z_max3": F(label="Максимум валютного давления за 3 месяца", unit="σ", src="imf_fx", when="emp"),
    "war25_now": F(label="Государственная война в этом месяце", unit="флаг", src="ucdp", when="m"),
    "war1000_now": F(label="≥ 1000 погибших за последние 12 месяцев", unit="флаг", src="ucdp", when="m"),
    "war_onset12": F(label="Начало войны за последние 12 месяцев", unit="флаг", src="ucdp", when="m"),
    # --- претендент: привязка, REER, привязка × падение резервов ---
    "peg_share12": F(label="Доля спокойных месяцев курса из последних 12 (привязка)", unit="доля", src="imf_fx", when="m"),
    "reer_gap": F(label="Реальный эффективный курс к своему среднему за 5 лет", unit="ln(отношения)", src="bruegel", when="reer", nat="exp1:%"),
    "reer_peg": F(label="Реальное укрепление × привязка («пружина»)", unit="индекс", src="bruegel", when="reer"),
    "reer_missing": F(label="Флаг: нет реального эффективного курса", unit="флаг", src="bruegel", when="reer"),
    "peg_resfall": F(label="Привязка × падение резервов за 12 месяцев", unit="индекс", src="imf_res", when="res_live"),
    "peg_resfall_missing": F(label="Флаг: нет резервов", unit="флаг", src="imf_res", when="res_live"),
    # --- банковский (деревья BoE): макро — T − лаг, ставки — T ---
    "cpi_pdiff2": F(label="Рост потребительских цен за 2 года", unit="доля", src="gmd", when="macro", nat="pct:%"),
    "bmon_gdp_rdiff2": F(label="Изменение денежной массы к ВВП за 2 года", unit="доля ВВП", src="gmd", when="macro", nat="pct:п.п. ВВП"),
    "cons_pdiff2": F(label="Рост реального потребления на душу за 2 года", unit="доля", src="gmd", when="macro", nat="pct:%"),
    "pdebt_gdp_rdiff2": F(label="Изменение госдолга к ВВП за 2 года", unit="доля ВВП", src="gmd", when="macro", nat="pct:п.п. ВВП"),
    "inv_gdp_rdiff2": F(label="Изменение инвестиций к ВВП за 2 года", unit="доля ВВП", src="gmd", when="macro", nat="pct:п.п. ВВП"),
    "ca_gdp_rdiff2": F(label="Изменение счёта текущих операций к ВВП за 2 года", unit="доля ВВП", src="gmd", when="macro", nat="pct:п.п. ВВП"),
    "tloan_gdp_rdiff2": F(label="Изменение кредита к ВВП за 2 года", unit="доля ВВП", src="bis", when="macro", nat="pct:п.п. ВВП"),
    "tdbtserv_gdp_rdiff2": F(label="Изменение обслуживания долга к ВВП за 2 года", unit="доля ВВП", src="bis", when="macro_rate", nat="pct:п.п. ВВП"),
    "global_loan2": F(label="Мировой кредитный рост за 2 года", unit="доля ВВП", src="bis", when="macro", nat="pct:п.п. ВВП"),
    "drate": F(label="Наклон кривой доходности (длинная ставка минус короткая)", unit="п.п.", src="gmd_rates", when="T"),
    "global_drate": F(label="Мировой наклон кривой доходности", unit="п.п.", src="gmd_rates", when="T"),
}

SOURCES = {
    "ucdp": "UCDP GED / ACD", "polity": "Polity5", "wdi": "Всемирный банк (WDI)", "gmd": "Global Macro Database",
    "imf_fx": "МВФ, курс к доллару", "imf_fx_cpi": "МВФ, курс и ИПЦ", "imf_cpi": "МВФ, ИПЦ", "imf_rates": "МВФ, ставки",
    "imf_res": "МВФ, резервы", "imf_exports": "МВФ, экспорт", "bruegel": "Bruegel, REER", "bis": "BIS, кредит",
    "gmd_rates": "Global Macro Database, ставки",
}

STATUS_TEXT = {  # единый словарь состояний (раздел 6 ТЗ): код → текст на экране
    "no_bis_credit": "Нет ряда BIS",
    "no_yield_curve": "Нет рыночного ряда",
    "no_broad_money": "Нет ряда денежной массы",
    "no_imf_fx": "Нет валютного ряда",
    "no_recent_fx": "Нет свежего курса",
    "out_of_sample": "Вне выборки модели",
    "no_estimate_period": "Нет оценки на этот период",
    "other_group": "Идёт конфликт · посмотреть продолжение",
    "no_conflict": "Конфликта нет · посмотреть начало",
    "no_interval": "Интервал не рассчитан",
    "ongoing": "Целевой период ещё идёт",
    "awaiting": "Ожидаются данные об исходе",
}
STATUS_DETAIL = {
    "no_bis_credit": "У страны нет кредитного ряда в статистике Банка международных расчётов (BIS), на котором построена банковская модель.",
    "no_yield_curve": "Страна есть в кредитной статистике BIS, но в данных нет длинной или короткой рыночной ставки: наклон кривой доходности посчитать не на чем.",
    "no_broad_money": "Длинная ставка есть, но у страны нет собственного ряда денежной массы (общая валюта).",
    "no_imf_fx": "Валюта у страны своя, но её курса нет в данных МВФ — прогноз посчитать не на чем.",
    "no_recent_fx": "Месячный курс МВФ по стране кончается раньше допустимой даты среза — свежего прогноза нет.",
    "out_of_sample": "Модель не оценивает годы сразу после инфляции выше 100 % и годы, когда большая девальвация уже идёт.",
    "no_estimate_period": "На этот целевой период у страны не хватило входных данных.",
    "other_group": "Страна уже воюет: вопрос «начнётся ли» к ней не относится. Для неё оценивается продолжение конфликта.",
    "no_conflict": "В стране нет государственной войны: для неё оценивается начало, а не продолжение.",
}


# ---------------------------------------------------------------------------
# Мелочи
# ---------------------------------------------------------------------------


def num(x, nd: int = 4):
    """Число для JSON: NaN / None → null, иначе округление. Нулей вместо пропусков не бывает."""
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def _rank(s: pd.Series) -> pd.Series:
    return s.rank(ascending=False, method="min")


def _natural(spec: str | None, v: float):
    """Значение признака «по-человечески»: (число, единица) или None."""
    if spec is None or v is None or not np.isfinite(v):
        return None
    kind, unit = spec.split(":", 1)
    val = {"expm1": np.expm1(v), "exp": np.exp(v), "exp1": 100 * np.expm1(v), "pct": 100 * v,
           "infl": 100 * np.expm1(v)}[kind]
    return {"value": num(val, 3 if abs(val) < 100 else 0), "unit": unit}


def _write(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def country_name(code: str, fallback: str | None = None) -> str:
    return COUNTRY_NAMES[code][0] if code in COUNTRY_NAMES else TERRITORY_RU.get(code, fallback or code)


# ---------------------------------------------------------------------------
# Разбор прогнозов: одна запись на строку теста
# ---------------------------------------------------------------------------


def _records(ex: Explained, ids: list, raw: np.ndarray, model_x: np.ndarray, part: str = "") -> dict:
    """id → запись с вероятностью, интервалом и вкладами. `raw` — значения признаков до
    обрезки и заполнения (NaN = наблюдения нет), `model_x` — что получила модель."""
    out = {}
    for i, key in enumerate(ids):
        out[key] = {"p": float(ex.p[i]), "lo": float(ex.lo[i]), "hi": float(ex.hi[i]), "eta": float(ex.eta[i]),
                    "intercept": ex.intercept, "base_rate": ex.base_rate, "n_train": ex.n_train, "part": part,
                    "features": ex.features, "raw": raw[i], "x": model_x[i], "z": ex.z[i], "beta": ex.beta,
                    "contrib": ex.contrib[i]}
    return out


def _constant(ids: list, p: float, n_train: int, part: str) -> dict:
    """В обучении один класс: прогноз — его доля, модели нет, интервала и вкладов тоже."""
    return {k: {"p": p, "lo": None, "hi": None, "eta": None, "base_rate": p, "n_train": n_train, "part": part,
                "features": [], "no_model": True} for k in ids}


def war_explain(f: pd.DataFrame, T: int, target: str, feats: list[str]) -> dict:
    """Прогноз войны на T+1 по признакам года T — то же, что `model_hist.forecast_asof`, плюс
    разбор. Ключ — код GW. Страна берёт модель начала (`onset`) или продолжения (`cont`)."""
    from seldon.domains.conflict import model_hist

    d = f if "y_next" in f else model_hist.with_next(f, target)
    train = d[label_known(d.year, T, model_hist.HORIZON) & d.y_next.notna()]
    test = d[d.year == T]
    out = {}
    for part, flag in (("onset", 0), ("cont", 1)):
        tr, te = train[train[target] == flag], test[test[target] == flag]
        ids = te.gw.astype(int).tolist()
        if not ids:
            continue
        if len(tr) == 0:
            continue
        if tr.y_next.nunique() < 2:
            out.update(_constant(ids, float(tr.y_next.mean()), len(tr), part))
            continue
        m = model_hist.make_model("logit").fit(tr[feats], tr.y_next.astype(int))
        X = te[feats].to_numpy(dtype=float)
        ex = explain_logit(m, feats, tr[feats].to_numpy(dtype=float), tr.y_next, X, [tr.gw, tr.year])
        out.update(_records(ex, ids, X, X, part))
    return out


def fxa_explain(p: pd.DataFrame, T: int, train_through: int | None = None) -> dict:
    """Годовой валютный прогноз (логрегрессия `logit_long`) с признаков года T — как
    `econ_long.forecast_asof`, плюс разбор. Ключ — ISO3."""
    from seldon.domains.currency import econ_long
    from seldon.domains.currency.long_features import BASE, FEATURES, MISSING
    from seldon.domains.currency.long_targets import FIRST_YEAR, H, MIN_POS_TRAIN

    known = label_known(p.year, T, H) if train_through is None else (p.year <= train_through)
    train = p[p.evaluable & known & (p.year >= FIRST_YEAR)]
    test = p[(p.year == T) & p.usable]
    if train.y.sum() < MIN_POS_TRAIN or test.empty:
        return {}
    lo, hi = train[BASE].quantile(0.01), train[BASE].quantile(0.99)
    Xtr, Xte = econ_long._design(train, lo, hi), econ_long._design(test, lo, hi)
    m = econ_long.make_models()["logit"].fit(Xtr, train.y.astype(int))
    ex = explain_logit(m, FEATURES, Xtr, train.y, Xte, [train.iso3, train.year])
    return _records(ex, test.iso3.tolist(), test[[*BASE, *MISSING]].to_numpy(dtype=float), Xte)


def fxm_explain(p: pd.DataFrame, as_of: pd.Period, test: pd.DataFrame) -> dict:
    """Месячный валютный прогноз (боевая логрегрессия `econ_kr`). Ключ — ISO3."""
    from seldon.domains.currency import econ_kr
    from seldon.domains.currency.kr_features import INDICATORS

    feats = list(INDICATORS)
    m = econ_kr.fit(p, as_of)
    Xtr = m.known[feats].clip(m.lo, m.hi, axis=1).fillna(m.med)
    Xte = test[feats].clip(m.lo, m.hi, axis=1).fillna(m.med)
    ex = explain_logit(m.logit, feats, Xtr.to_numpy(dtype=float), m.known.y, Xte.to_numpy(dtype=float),
                       [m.known.iso3, m.known.month.dt.year])
    return _records(ex, test.iso3.tolist(), test[feats].to_numpy(dtype=float), Xte.to_numpy(dtype=float))


def live_explain(light: pd.DataFrame, as_of: pd.Period, test: pd.DataFrame) -> tuple[dict, float]:
    """Претендент `live_fx_swap` на живом крае — как `econ_csd.live_edge`. `test` — строки после
    `stale_fill`. Возвращает (записи, выбранная сила регуляризации C)."""
    from seldon.domains.currency import econ_csd
    from seldon.domains.currency.econ_struct import _kr_design, _kr_fit, choose_C_kr

    feats = econ_csd.features(econ_csd.CHALLENGER)
    known = light[label_known(light.month, as_of, 12) & light.evaluable]
    C = choose_C_kr(known, feats, h=12)
    m, lo, hi, med = _kr_fit(known, feats, C)
    Xtr, Xte = _kr_design(known, feats, lo, hi, med), _kr_design(test, feats, lo, hi, med)
    ex = explain_logit(m, feats, Xtr.to_numpy(dtype=float), known.y, Xte.to_numpy(dtype=float),
                       [known.iso3, known.month.dt.year])
    return _records(ex, test.iso3.tolist(), test[feats].to_numpy(dtype=float), Xte.to_numpy(dtype=float)), float(C)


def _check(name: str, ours: pd.Series, published: pd.Series) -> None:
    """Пересчитанные вероятности обязаны совпасть с опубликованными в таблице риска."""
    both = pd.concat([ours.rename("a"), published.rename("b")], axis=1)
    bad = both[both.a.notna() != both.b.notna()]
    if len(bad):
        raise RuntimeError(f"{name}: покрытие не совпало с reports/risk_2025_26.csv: {bad.index.tolist()[:8]}")
    diff = (both.a - both.b).abs().max()
    if diff > TOL:
        raise RuntimeError(f"{name}: пересчёт расходится с reports/risk_2025_26.csv на {diff:.2e} — сначала `make all`")
    print(f"  сверка {name}: {int(both.a.notna().sum())} стран, max|Δ| = {diff:.1e}")


# ---------------------------------------------------------------------------
# Даты наблюдений
# ---------------------------------------------------------------------------


def _obs_date(when: str, ctx: dict) -> tuple[str | None, str | None]:
    """Дата наблюдения признака по правилу `when` и примечание. Год — «ГГГГ», месяц — «ГГГГ-ММ»."""
    T, m = ctx.get("T"), ctx.get("m")
    if when == "T":
        return str(T), None
    if when == "dec":
        return f"{T}-12", "курс на конец декабря"
    if when == "macro":
        return str(T - ctx["macro_lag"]), "годовые макроданные выходят с лагом публикации"
    if when == "macro_rate":
        return str(T - ctx["macro_lag"]), f"кредит — за {T - ctx['macro_lag']}, ставка — за {T}"
    if when == "polity":
        y = ctx.get("polity_year")
        if y is None:
            return None, "страны нет в Polity"
        return str(min(int(y), T)), ("последний год Polity5, значение продлено" if y < T else None)
    if when.startswith("wdi:"):
        y = ctx.get(when)
        return (str(int(y)), None) if y is not None else (None, "нет ряда Всемирного банка")
    if when == "m":
        return str(m), None
    if when == "res":
        return str(m - ctx["res_lag"]), f"резервы МВФ выходят на {ctx['res_lag']} месяца позже курса"
    if when == "emp":
        return str(m), f"курс — по {m}, резервы — по {m - ctx['res_lag']}"
    if when == "reer":
        r = ctx.get("reer_month")
        if r is None:
            return None, "нет ряда REER не старше допуска"
        return str(r), ("последнее известное значение (дотянуто)" if r < m else "последние два месяца досчитаны по номинальному курсу корзины")
    if when == "res_live":
        r = ctx.get("res_month")
        if r is None:
            return None, "нет резервов не старше допуска"
        note = "последнее известное значение (дотянуто)" if ctx.get("res_stale") else None
        return str(r), note
    raise KeyError(when)


def factor_list(rec: dict, ctx: dict, binary_missing: dict[str, str] | None = None) -> list[dict]:
    """Вклады признаков записи `rec` в формате сайта. `binary_missing` — признак → его флаг
    пропуска (годовые модели: при пропуске в модели стоит 0 и горит флаг)."""
    binary_missing = binary_missing or {}
    out = []
    for j, name in enumerate(rec["features"]):
        meta = FEATURE_META[name]
        raw, x = float(rec["raw"][j]), float(rec["x"][j])
        flag = binary_missing.get(name)
        missing = (not np.isfinite(raw)) or (flag is not None and rec["raw"][rec["features"].index(flag)] == 1)
        obs, note = _obs_date(meta["when"], ctx)
        item = {"key": name, "label": meta["label"], "unit": meta["unit"], "source": SOURCES[meta["src"]],
                "value": None if missing else num(raw, 4), "natural": None if missing else _natural(meta.get("nat"), raw),
                "date": None if missing else obs, "date_note": note,
                "z": num(rec["z"][j], 4), "beta": num(rec["beta"][j], 4), "contrib": num(rec["contrib"][j], 4)}
        if missing:
            item["missing"] = ("в модели стоит 0 и отдельный флаг пропуска" if flag is not None
                               else "наблюдения нет: подставлена медиана обучающей выборки")
        elif abs(raw - x) > 1e-12:
            item["clipped"] = num(x, 4)  # модель получила значение, обрезанное по 1–99-му процентилю обучения
        out.append(item)
    return out


def factor_block(rec: dict | None, ctx: dict, model: str, binary_missing: dict | None = None) -> dict | None:
    if rec is None:
        return None
    if rec.get("no_model"):
        return {"model": model, "kind": "constant", "p": num(rec["p"]),
                "note": "В обучающей выборке события одного класса: оценка равна их доле, модели и вкладов нет."}
    return {"model": model, "kind": "logit", "scale": "лог-шансы (β·z); не процентные пункты вероятности",
            "p": num(rec["p"]), "lo": num(rec["lo"]), "hi": num(rec["hi"]), "eta": num(rec["eta"]),
            "intercept": num(rec["intercept"]), "base_rate": num(rec["base_rate"]), "n_train": rec["n_train"],
            "part": rec.get("part") or None, "features": factor_list(rec, ctx, binary_missing)}


def input_list(names: list[str], values, ctx: dict) -> list[dict]:
    """Входные наблюдения модели на деревьях: значения и даты, без вкладов."""
    out = []
    for name, v in zip(names, values):
        meta = FEATURE_META[name]
        obs, note = _obs_date(meta["when"], ctx)
        ok = v is not None and np.isfinite(v)
        out.append({"key": name, "label": meta["label"], "unit": meta["unit"], "source": SOURCES[meta["src"]],
                    "value": num(v, 4) if ok else None, "natural": _natural(meta.get("nat"), float(v)) if ok else None,
                    "date": obs if ok else None, "date_note": note})
    return out


# ---------------------------------------------------------------------------
# Сборка
# ---------------------------------------------------------------------------


def _sha(path) -> str:
    return vintage.sha256_of(path)


def _month_ru_ctx(as_of_min: str, as_of_max: str) -> str:
    return as_of_min if as_of_min == as_of_max else f"{as_of_min} … {as_of_max}"


def build_current(risk: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Текущий срез. Возвращает (current.json, досье-заготовки по ISO3, служебное для meta)."""
    from seldon.domains.banking import econ_boe_em
    from seldon.domains.conflict.features import FEATURES as HIST
    from seldon.domains.conflict.features_pitf import PITF_FEATURES
    from seldon.domains.currency import econ_csd, econ_kr, econ_long, kr_features
    from seldon.domains.currency.long_features import BASE, FEATURES as FXA_FEATS, MISSING
    from seldon.domains.currency.struct_features import load_struct
    from seldon.publish import risk_table as rt
    from seldon.sources.gmd import load_gmd
    from seldon.sources.imf import load_dec_fx
    from seldon.sources.jst import JST_R6, load_jst
    from seldon.sources.polity import load_polity
    from seldon.sources.wdi import load_annual_cpi, load_wdi

    risk = risk.set_index("iso3")
    dossier: dict[str, dict] = {iso: {"iso3": iso, "factors": {}, "inputs": {}} for iso in risk.index}

    # ---------------- война ----------------
    print("война: обучение и разбор…")
    f = pd.read_parquet(DATA_PROCESSED / "features_pitf_country_year.parquet")
    feats = [*HIST, *PITF_FEATURES]
    T = int(f.year.max())
    assert T == int(risk.war_features_year.iloc[0])
    wdi_lag = publication_lag("wdi_annual", "jan1")
    polity_last = load_polity().groupby("gw").year.max()
    wdi_last = {}
    for ind in ("infant_mortality", "population"):
        w = load_wdi(ind)
        w = w[w[ind].notna() & (w.year <= T - wdi_lag)]
        wdi_last[ind] = w.groupby("gw").year.max()
    war = {}
    for target in ("state_war25", "state_war1000"):
        recs = war_explain(f, T, target, feats)
        p = pd.Series({GW_TO_ISO3[g]: r["p"] for g, r in recs.items() if GW_TO_ISO3.get(g) in risk.index})
        _check(f"война {target}", p.reindex(risk.index), risk[f"p_{target}_{T + 1}"])
        war[target] = recs
        short = "war25" if target == "state_war25" else "war1000"
        for g, r in recs.items():
            iso = GW_TO_ISO3.get(g)
            if iso not in dossier:
                continue
            ctx = {"T": T, "polity_year": int(polity_last[g]) if g in polity_last.index else None,
                   **{f"wdi:{k}": (int(v[g]) if g in v.index else None) for k, v in wdi_last.items()}}
            name = "логрегрессия начала конфликта" if r["part"] == "onset" else "логрегрессия продолжения конфликта"
            dossier[iso]["factors"][short] = factor_block(r, ctx, f"model_hist pitf · {name}")

    # ---------------- годовой валютный ----------------
    print("годовой валютный: обучение и разбор…")
    lp = econ_long.build_panel(load_gmd(), load_dec_fx(), econ_long.load_lv(), "deval25")
    lp = lp[lp.iso3.isin(rt.GW_ISO)]
    fxa = {}
    miss_of = dict(zip(["infl_log", "gdp_growth_2", "debt_d2", "ca_level", "money_growth_2"], MISSING))
    for y, sfx in zip(rt.FX_ANNUAL_YEARS, ("_2025_26", "_2026_27")):
        recs = fxa_explain(lp, y, train_through=rt.FX_ANNUAL_TRAIN_THROUGH)
        _check(f"валютный годовой{sfx}", pd.Series({k: r["p"] for k, r in recs.items()}).reindex(risk.index), risk[f"fxa_logit{sfx}"])
        fxa[sfx] = recs
    macro_lag = econ_long.MACRO_LAG
    Ty = rt.FX_ANNUAL_YEARS[-1]
    for iso, r in fxa["_2026_27"].items():
        ctx = {"T": Ty, "macro_lag": macro_lag}
        dossier[iso]["factors"]["fxa"] = factor_block(r, ctx, "econ_long logit_long · логрегрессия", miss_of)
        vals = [None if (n in miss_of and r["raw"][FXA_FEATS.index(miss_of[n])] == 1) else r["raw"][FXA_FEATS.index(n)] for n in BASE]
        dossier[iso]["inputs"]["fxa_tree"] = {"model": "econ_long extree_long · ExtraTrees (400 деревьев)", "kind": "trees",
                                             "note": "Модель на деревьях: линейных вкладов признаков у неё нет. Показаны входные наблюдения.",
                                             "p": num(risk.at[iso, "fxa_extree_2026_27"]), "inputs": input_list(BASE, vals, ctx)}

    # ---------------- месячный валютный ----------------
    print("месячный валютный: панель, обучение и разбор…")
    as_of = pd.Period(rt.FX_LAST_MONTH, freq="M")
    mp = econ_kr.build_panel(econ_kr.load_panel(rt.FX_LAST_MONTH), load_annual_cpi(), kr_features.load_war(), target="deval15")
    last = mp[mp.fx.notna()].groupby("iso3").month.max()
    test = mp[mp.iso3.isin(rt.GW_ISO) & mp.own_currency].merge(last.rename("as_of"), on="iso3")
    test = test[(test.month == test.as_of) & (test.as_of >= rt.FX_MIN_AS_OF)]
    fxm = fxm_explain(mp, as_of, test)
    _check("валютный месячный", pd.Series({k: r["p"] for k, r in fxm.items()}).reindex(risk.index), risk.fx_deval15_12m)
    res_lag = monthly_lag("imf_reserves")
    as_of_by = test.set_index("iso3").as_of
    for iso, r in fxm.items():
        dossier[iso]["factors"]["fxm"] = factor_block(r, {"m": as_of_by[iso], "res_lag": res_lag}, "econ_kr logit · логрегрессия")

    print("претендент: панель, обучение и разбор…")
    struct_raw = load_struct(last_month=rt.FX_LAST_MONTH, filled=False)
    light = econ_csd.light_panel(mp, load_struct(last_month=rt.FX_LAST_MONTH))
    lt = light.merge(test[["iso3", "as_of"]], on="iso3")
    lt = econ_csd.stale_fill(light, lt[lt.month == lt.as_of], struct_raw, rt.REER_MAX_STALE, rt.RES_MAX_STALE)
    live, live_C = live_explain(light, as_of, lt)
    _check("претендент", pd.Series({k: r["p"] for k, r in live.items()}).reindex(risk.index), risk.fx_deval15_12m_live)
    # откуда на самом деле взяты REER и резервы на живом крае (то же правило, что в `stale_fill`)
    age = lambda d: (d.as_of - d.month).map(lambda x: x.n)  # noqa: E731
    st = struct_raw.merge(test[["iso3", "as_of"]], on="iso3")
    st = st[(st.month <= st.as_of) & st.reer_gap.notna()]
    reer_month = st[age(st) <= rt.REER_MAX_STALE].groupby("iso3").month.max()
    rs = light[light.reserves_g12.notna()].merge(test[["iso3", "as_of"]], on="iso3")
    rs = rs[rs.month <= rs.as_of]
    res_row = rs[age(rs) <= rt.RES_MAX_STALE].groupby("iso3").month.max()
    for iso, r in live.items():
        rr = res_row.get(iso)
        ctx = {"m": as_of_by[iso], "res_lag": res_lag, "reer_month": reer_month.get(iso),
               "res_month": None if rr is None else rr - res_lag, "res_stale": rr is not None and rr < as_of_by[iso]}
        dossier[iso]["factors"]["fxm_live"] = factor_block(r, ctx, f"econ_csd live_fx_swap · логрегрессия, C = {live_C:g} (модель-претендент)",
                                                           {"peg_resfall": "peg_resfall_missing", "reer_gap": "reer_missing", "reer_peg": "reer_missing"})
    # давление за последние месяцы — для мини-графика досье
    tail = mp[mp.iso3.isin(test.iso3)].merge(test[["iso3", "as_of"]], on="iso3")
    tail = tail[(tail.month <= tail.as_of) & (tail.month > tail.as_of - EMP_MONTHS)]
    for iso, g in tail.groupby("iso3"):
        dossier[iso]["emp"] = [[str(m), num(v, 3)] for m, v in zip(g.month, g.emp_z_max3)]

    # ---------------- банковский: входы деревьев ----------------
    print("банковский: входные наблюдения…")
    cfg = econ_boe_em.Config(last_year=rt.BANK_YEARS[-1])
    bfeats = econ_boe_em.FEATURE_SETS[rt.BANK_SET]
    bdata = econ_boe_em.build(econ_boe_em.load_panel(cfg), bfeats, cfg.macro_lag)
    jst_d = econ_boe_em.boe_dataset(load_jst(JST_R6, "Sheet1"), bfeats, cfg.macro_lag)
    jst_d = jst_d[jst_d.year <= econ_boe_em.JST_LAST_LABEL_YEAR - 2]
    bank_base, bank_n = float(jst_d.crisis.mean()), int(len(jst_d))
    Tb = rt.BANK_YEARS[-1]
    for _, b in bdata[bdata.year == Tb].iterrows():
        if b.iso in dossier and pd.notna(risk.at[b.iso, "bank_jst_extree_full_2026_27"]):
            dossier[b.iso]["inputs"]["bank"] = {
                "model": "econ_boe_em jst_extree, набор full · ExtraTrees (500 деревьев), обучение на JST", "kind": "trees",
                "note": "Модель на деревьях: линейных вкладов признаков у неё нет. Показаны входные наблюдения.",
                "p": num(risk.at[b.iso, "bank_jst_extree_full_2026_27"]),
                "inputs": input_list(bfeats, [b[c] for c in bfeats], {"T": Tb, "macro_lag": cfg.macro_lag})}

    # ---------------- строки стран ----------------
    def est(p, lo=None, hi=None, rank=None, n=None, **extra):
        e = {"p": num(p), "rank": None if rank is None or pd.isna(rank) else int(rank), "n": n}
        if lo is not None and np.isfinite(lo):
            e["lo"], e["hi"] = num(lo), num(hi)
        e.update(extra)
        return e

    w25, w1000 = f"p_state_war25_{T + 1}", f"p_state_war1000_{T + 1}"
    groups = {}
    for short, col, now in (("war25", w25, "state_war25_now"), ("war1000", w1000, "state_war1000_now")):
        for part, flag in (("onset", 0), ("cont", 1)):
            g = risk[risk[now] == flag]
            # места — по неокруглённым вероятностям пересчёта (в таблице риска они округлены до 4 знаков)
            exact = pd.Series({iso: war["state_" + short][ISO3_TO_GW[iso]]["p"] for iso in g.index})
            groups[f"{short}_{part}"] = (_rank(exact), len(g))
    for short in ("war25", "war1000"):  # места начала обязаны совпасть с опубликованными
        np.testing.assert_array_equal(groups[f"{short}_onset"][0].reindex(risk.index).to_numpy(), risk[f"rank_state_{short}_at_risk"].to_numpy())
    counts = {c: int(risk[c].notna().sum()) for c in ("fxa_logit_2026_27", "fxa_extree_2026_27", "fxa_logit_2025_26", "fx_deval15_12m",
                                                      "fx_deval15_12m_live", "bank_jst_extree_full_2026_27", "bank_jst_extree_full_2025_26")}
    gw_of = {iso: ISO3_TO_GW.get(iso) for iso in risk.index}
    rows = []
    for iso, r in risk.iterrows():
        g = gw_of[iso]
        e: dict = {}
        for short, target, col, now in (("war25", "state_war25", w25, "state_war25_now"), ("war1000", "state_war1000", w1000, "state_war1000_now")):
            rec = war[target].get(g)
            part = "cont" if r[now] == 1 else "onset"
            rk, n = groups[f"{short}_{part}"]
            item = est(r[col], rec and rec["lo"], rec and rec["hi"], rk[iso], n, base_rate=num(rec["base_rate"]) if rec else None)
            e[f"{short}_{part}"] = item
            e[f"{short}_{'onset' if part == 'cont' else 'cont'}"] = {"status": "other_group" if part == "cont" else "no_conflict"}
        # валюта
        regime = r.fx_regime
        fx_block = {"shared": f"shared:{regime}"} if regime not in ("own", "no_imf_fx") else {}

        def fx_status(default: str) -> dict:
            if "shared" in fx_block:
                return {"status": fx_block["shared"]}
            return {"status": "no_imf_fx" if regime == "no_imf_fx" else default}

        ra = fxa["_2026_27"].get(iso)
        if pd.notna(r.fxa_logit_2026_27):
            e["fxa"] = est(r.fxa_logit_2026_27, ra["lo"], ra["hi"], r.rank_fxa_logit_2026_27, counts["fxa_logit_2026_27"], base_rate=num(r.fxa_base_rate))
            e["fxa_tree"] = est(r.fxa_extree_2026_27, rank=r.rank_fxa_extree_2026_27, n=counts["fxa_extree_2026_27"], base_rate=num(r.fxa_base_rate))
        else:
            e["fxa"] = e["fxa_tree"] = fx_status("out_of_sample")
        rc = fxa["_2025_26"].get(iso)
        e["fxa_check"] = (est(r.fxa_logit_2025_26, rc["lo"], rc["hi"], r.rank_fxa_logit_2025_26, counts["fxa_logit_2025_26"])
                          if pd.notna(r.fxa_logit_2025_26) else fx_status("out_of_sample"))
        if pd.notna(r.fx_deval15_12m):
            rm, rl = fxm[iso], live[iso]
            e["fxm"] = est(r.fx_deval15_12m, rm["lo"], rm["hi"], r.rank_fx_deval15_12m, counts["fx_deval15_12m"],
                           base_rate=num(r.fx_deval15_base_rate), recent_jump=int(r.deval15_last_12m))
            e["fxm_live"] = est(r.fx_deval15_12m_live, rl["lo"], rl["hi"], r.rank_fx_deval15_12m_live, counts["fx_deval15_12m_live"],
                                base_rate=num(rl["base_rate"]))
            e["stress"] = {"value": num(r.emp_z_max3, 3), "unit": "σ"}
        else:
            e["fxm"] = e["fxm_live"] = e["stress"] = fx_status("no_recent_fx")
        # банки
        if pd.notna(r.bank_jst_extree_full_2026_27):
            e["bank"] = est(r.bank_jst_extree_full_2026_27, rank=r.rank_bank_jst_extree_full_2026_27, n=counts["bank_jst_extree_full_2026_27"], base_rate=num(bank_base))
        else:
            e["bank"] = {"status": r.bank_status}
        if pd.notna(r.bank_jst_extree_full_2025_26):
            e["bank_check"] = est(r.bank_jst_extree_full_2025_26, rank=r.rank_bank_jst_extree_full_2025_26, n=counts["bank_jst_extree_full_2025_26"], base_rate=num(bank_base))
        else:
            e["bank_check"] = {"status": r.bank_status if r.bank_status != "ok" else "no_estimate_period"}
        row = {"iso3": iso, "post_soviet": bool(r.post_soviet), "est": e,
               "war_now": {"state_war25": int(r.state_war25_now), "state_war1000": int(r.state_war1000_now), "nonstate25": int(r.nonstate25_now)},
               "fx_regime": regime, "fx_as_of": None if pd.isna(r.as_of) else str(r.as_of), "bank_status": r.bank_status}
        if pd.notna(r.as_of):
            m = pd.Period(r.as_of, freq="M")
            rr = res_row.get(iso)
            row["indicators"] = {  # наблюдения, НЕ вероятности
                "emp_z_max3": {"value": num(r.emp_z_max3, 3), "date": str(m), "note": f"курс — по {m}, резервы — по {m - res_lag}"},
                "reserves_g12": {"value": num(r.reserves_g12), "date": None if pd.isna(r.reserves_g12) else str(m - res_lag)},
                "deval12": {"value": num(r.deval12), "date": str(m)},
                "infl12": {"value": num(r.infl12), "date": None if pd.isna(r.infl12) else str(m)},
                "peg_share12": {"value": num(r.peg_share12), "date": None if pd.isna(r.peg_share12) else str(m)},
                "reer_peg": {"value": num(r.reer_peg), "date": None if pd.isna(r.reer_peg) or iso not in reer_month.index else str(reer_month[iso])},
                "peg_resfall": {"value": num(r.peg_resfall), "date": None if pd.isna(r.peg_resfall) or rr is None else str(rr - res_lag)},
                "signals_fired": [] if pd.isna(r.signals_fired) else str(r.signals_fired).split(","),
            }
        if pd.notna(r.fxa_deval_2025):
            row["fx_fact"] = {"deval_2025": num(r.fxa_deval_2025, 2)}  # декабрь 2025 к декабрю 2024, %
        rows.append(row)

    # ---------------- контексты событий ----------------
    as_of_war = f"{T + 1}-01-01"
    amin, amax = str(risk.as_of.dropna().min()), str(risk.as_of.dropna().max())
    n_on25, n_on1000 = groups["war25_onset"][1], groups["war1000_onset"][1]
    ctx_war = dict(domain="war", as_of=as_of_war, features_year=T, target_period=str(T + 1), horizon="1 год",
                   source=f"UCDP GED {vintage.UCDP_VERSION}", interval="logit")
    contexts = {
        "war25_onset": dict(ctx_war, title="Начало конфликта, ≥ 25 погибших", short="начало, ≥ 25",
                            event=f"Начало государственного вооружённого конфликта: ≥ 25 погибших за {T + 1} год на своей территории или у соседа, где страна — сторона",
                            sample=f"страны без государственной войны в {T} году", n=n_on25, model="model_hist pitf · логрегрессия начала"),
        "war25_cont": dict(ctx_war, title="Продолжение конфликта, ≥ 25 погибших", short="продолжение, ≥ 25",
                           event=f"Продолжение государственного вооружённого конфликта: ≥ 25 погибших за {T + 1} год",
                           sample=f"страны с государственной войной в {T} году", n=groups["war25_cont"][1], model="model_hist pitf · логрегрессия продолжения"),
        "war1000_onset": dict(ctx_war, title="Начало конфликта, ≥ 1000 погибших", short="начало, ≥ 1000",
                              event=f"Начало государственного вооружённого конфликта высокой интенсивности: ≥ 1000 погибших за {T + 1} год",
                              sample=f"страны без войны ≥ 1000 погибших в {T} году", n=n_on1000, model="model_hist pitf · логрегрессия начала"),
        "war1000_cont": dict(ctx_war, title="Продолжение конфликта, ≥ 1000 погибших", short="продолжение, ≥ 1000",
                             event=f"Продолжение конфликта высокой интенсивности: ≥ 1000 погибших за {T + 1} год",
                             sample=f"страны с войной ≥ 1000 погибших в {T} году", n=groups["war1000_cont"][1], model="model_hist pitf · логрегрессия продолжения"),
        "fxa": dict(domain="fx", title="Структурный прогноз", short="девальвация ≥ 25 %", as_of=f"{Ty + 1}-01-01", features_year=Ty,
                    target_period=f"{Ty + 1}–{Ty + 2}", horizon="2 года", interval="logit", n=counts["fxa_logit_2026_27"],
                    event=f"Девальвация ≥ 25 % за год (курс к доллару, декабрь к декабрю) в {Ty + 1} или {Ty + 2} году после ≥ 2 спокойных лет",
                    sample="страны со своей валютой и курсом МВФ", model="econ_long logit_long · логрегрессия",
                    source=f"МВФ (курс — декабрь {Ty}), Global Macro Database (макроданные — {Ty - macro_lag})"),
        "fxa_tree": dict(domain="fx", title="Структурный прогноз, деревья", short="девальвация ≥ 25 %, деревья", as_of=f"{Ty + 1}-01-01", features_year=Ty,
                         target_period=f"{Ty + 1}–{Ty + 2}", horizon="2 года", interval="none", n=counts["fxa_extree_2026_27"],
                         event=f"Девальвация ≥ 25 % за год (декабрь к декабрю) в {Ty + 1} или {Ty + 2} году",
                         sample="страны со своей валютой и курсом МВФ", model="econ_long extree_long · ExtraTrees",
                         source=f"МВФ (курс — декабрь {Ty}), Global Macro Database (макроданные — {Ty - macro_lag})"),
        "fxa_check": dict(domain="fx", title="Структурный прогноз, проверочный период", short="девальвация ≥ 25 %", as_of=f"{Ty}-01-01", features_year=Ty - 1,
                          target_period=f"{Ty}–{Ty + 1}", horizon="2 года", interval="logit", n=counts["fxa_logit_2025_26"],
                          event=f"Девальвация ≥ 25 % за год (декабрь к декабрю) в {Ty} или {Ty + 1} году", sample="страны со своей валютой и курсом МВФ",
                          model="econ_long logit_long · логрегрессия", source="МВФ, Global Macro Database"),
        "fxm": dict(domain="fx", title="Месячный обвал", short="скачок курса ≥ 15 % за месяц", as_of=_month_ru_ctx(amin, amax), as_of_per_country=True,
                    target_period="12 месяцев после даты среза страны", horizon="12 месяцев", interval="logit", n=counts["fx_deval15_12m"],
                    event="Скачок курса к доллару ≥ 15 % за один месяц (первый после ≥ 12 спокойных месяцев) в течение 12 месяцев после даты среза",
                    sample="страны со своей валютой и свежим месячным курсом МВФ", model="econ_kr logit · логрегрессия", source="МВФ, месячные ряды"),
        "fxm_live": dict(domain="fx", title="Месячный обвал, модель-претендент", short="скачок курса ≥ 15 %, претендент", as_of=_month_ru_ctx(amin, amax),
                         as_of_per_country=True, challenger=True, target_period="12 месяцев после даты среза страны", horizon="12 месяцев",
                         interval="logit", n=counts["fx_deval15_12m_live"],
                         event="То же событие и то же окно, что у месячного обвала; другой набор признаков (привязка, REER, привязка × падение резервов)",
                         sample="страны со своей валютой и свежим месячным курсом МВФ", model=f"econ_csd live_fx_swap · логрегрессия, C = {live_C:g}",
                         source="МВФ, Bruegel", note="Модель-претендент: публикуется рядом с основной для сравнения на будущих данных и её не заменяет."),
        "stress": dict(domain="fx", title="Текущий стресс", short="давление, σ", as_of=_month_ru_ctx(amin, amax), as_of_per_country=True, observed=True,
                       target_period=None, horizon=None, interval="none", n=counts["fx_deval15_12m"],
                       event="Текущее валютное давление — наблюдаемый показатель в стандартных отклонениях. Это не вероятность кризиса.",
                       sample="страны со своей валютой и свежим месячным курсом МВФ", model="индекс давления EMP, максимум за 3 месяца", source="МВФ, курс и резервы"),
        "bank": dict(domain="bank", title="Банковский кризис", short="системный банковский кризис", as_of=f"{Tb + 1}-01-01", features_year=Tb,
                     target_period=f"{Tb + 1}–{Tb + 2}", horizon="2 года", interval="none", n=counts["bank_jst_extree_full_2026_27"],
                     event=f"Системный банковский кризис в {Tb + 1} или {Tb + 2} году (по метке Bluwstein и др.)",
                     sample="страны BIS с длинной ставкой и собственной денежной массой", model="econ_boe_em jst_extree, набор full · ExtraTrees",
                     source=f"BIS, Global Macro Database (макроданные — {Tb - cfg.macro_lag}, ставки — {Tb})", base_rate=num(bank_base), n_train=bank_n),
        "bank_check": dict(domain="bank", title="Банковский кризис, проверочный период", short="системный банковский кризис", as_of=f"{Tb}-01-01",
                           features_year=Tb - 1, target_period=f"{Tb}–{Tb + 1}", horizon="2 года", interval="none", n=counts["bank_jst_extree_full_2025_26"],
                           event=f"Системный банковский кризис в {Tb} или {Tb + 1} году (по метке Bluwstein и др.)",
                           sample="страны BIS с длинной ставкой и собственной денежной массой", model="econ_boe_em jst_extree, набор full · ExtraTrees",
                           source="BIS, Global Macro Database", base_rate=num(bank_base), n_train=bank_n),
    }
    shared = int((~risk.fx_regime.isin(["own", "no_imf_fx"])).sum())
    coverage = {
        "war": {"estimated": len(risk), "not_applicable": 0, "no_estimate": 0, "at_risk": n_on25, "at_war": groups["war25_cont"][1]},
        "fx": {"estimated": counts["fxa_logit_2026_27"], "not_applicable": shared, "no_estimate": len(risk) - counts["fxa_logit_2026_27"] - shared,
               "monthly": counts["fx_deval15_12m"]},
        "bank": {"estimated": counts["bank_jst_extree_full_2026_27"], "not_applicable": 0,
                 "no_estimate": len(risk) - counts["bank_jst_extree_full_2026_27"], "by_status": risk.bank_status.value_counts().to_dict()},
    }
    current = {"rows": rows, "contexts": contexts, "coverage": coverage,
               "status_text": STATUS_TEXT, "status_detail": STATUS_DETAIL,
               "fx_monthly": {"last_month": rt.FX_LAST_MONTH, "as_of_min": amin, "as_of_max": amax, "reserves_lag": res_lag}}
    service = {"T_war": T, "T_fxa": Ty, "T_bank": Tb, "macro_lag": macro_lag, "wdi_lag": wdi_lag, "res_lag": res_lag,
               "bank_macro_lag": cfg.macro_lag, "live_C": live_C, "features": f, "war_feats": feats, "long_panel": lp,
               "polity_last": polity_last, "wdi_last": wdi_last}
    return current, dossier, service


# ---------------------------------------------------------------------------
# История проверки вслепую
# ---------------------------------------------------------------------------


def build_oos(service: dict) -> tuple[dict, dict]:
    """Лента рангов (`oos/<контур>.json`) и вклады по выпускам (`oos/<контур>/<год>.json`).
    Прогнозы заново обучаются по годам и сверяются с сохранёнными прогнозами вслепую."""
    from seldon.domains.conflict import model_hist
    from seldon.domains.currency import econ_long
    from seldon.domains.currency.long_features import MISSING

    ribbons, details = {}, {}

    # --- война: начало state_war25 ---
    print("лента проверки: война, начало…")
    saved = pd.read_parquet(DATA_PROCESSED / "pitf_state_war25_predictions.parquet")
    saved = saved[saved.at_risk == 1]
    f, feats = service["features"], service["war_feats"]
    d = model_hist.with_next(f, "state_war25")
    years, worst = [], 0.0
    for T in model_hist.TRAIN_YEARS:
        s = saved[saved.year == T + 1].set_index("gw")
        if s.empty:
            continue
        recs = {g: r for g, r in war_explain(d, T, "state_war25", feats).items() if r["part"] == "onset" and g in s.index}
        ours = pd.Series({g: r["p"] for g, r in recs.items()}).reindex(s.index)
        worst = max(worst, float((ours - s.pitf_state_war25).abs().max()))
        rk = _rank(s.pitf_state_war25)
        rows, det = [], {}
        for g in s.sort_values("pitf_state_war25", ascending=False).index:
            iso = GW_TO_ISO3.get(g) or f"GW{g}"
            r = recs[g]
            rows.append([iso, num(s.at[g, "pitf_state_war25"]), num(r["lo"]), num(r["hi"]), int(rk[g]), int(s.at[g, "onset_state_war25"])])
            ctx = {"T": T, "polity_year": int(service["polity_last"][g]) if g in service["polity_last"].index else None,
                   **{f"wdi:{k}": _last_before(v, g, T - service["wdi_lag"]) for k, v in service["wdi_raw"].items()}}
            det[iso] = factor_block(r, ctx, "model_hist pitf · логрегрессия начала конфликта")
        years.append({"release": T, "target": str(T + 1), "as_of": f"{T + 1}-01-01", "n": len(rows), "events": int(s.onset_state_war25.sum()), "rows": rows})
        details[("war25", T)] = det
    if worst > 1e-6:
        raise RuntimeError(f"лента войны: пересчёт расходится с pitf_state_war25_predictions.parquet на {worst:.1e}")
    print(f"  сверка с сохранёнными прогнозами вслепую: max|Δ| = {worst:.1e}")
    ribbons["war25"] = {"contour": "war25", "domain": "war", "title": "Начало государственного вооружённого конфликта",
                        "event": "Начало государственного вооружённого конфликта: ≥ 25 погибших за год; страны без войны в году выпуска",
                        "horizon": "1 год", "columns": ["id", "p", "lo", "hi", "rank", "outcome"], "years": years,
                        "source": f"UCDP GED {vintage.UCDP_VERSION}", "model": "model_hist pitf · логрегрессия начала"}

    # --- годовой валютный ---
    print("лента проверки: годовой валютный…")
    saved = pd.read_parquet(DATA_PROCESSED / "econ_long_predictions.parquet", columns=["iso3", "year", "country", "logit_long", "extree_long", "extree_short", "y"])
    saved = saved[saved.extree_long.notna() & saved.extree_short.notna()]  # те же строки, что в scorecard
    lp_all = service["long_panel_all"]
    miss_of = dict(zip(["infl_log", "gdp_growth_2", "debt_d2", "ca_level", "money_growth_2"], MISSING))
    years, worst, names = [], 0.0, {}
    for T in sorted(saved.year.unique()):
        s = saved[saved.year == T].set_index("iso3")
        recs = fxa_explain(lp_all, int(T))
        ours = pd.Series({k: r["p"] for k, r in recs.items()}).reindex(s.index)
        worst = max(worst, float((ours - s.logit_long).abs().max()))
        rk = _rank(s.logit_long)
        rows, det = [], {}
        for iso in s.sort_values("logit_long", ascending=False).index:
            r = recs[iso]
            rows.append([iso, num(s.at[iso, "logit_long"]), num(r["lo"]), num(r["hi"]), int(rk[iso]), int(s.at[iso, "y"])])
            det[iso] = factor_block(r, {"T": int(T), "macro_lag": econ_long.MACRO_LAG}, "econ_long logit_long · логрегрессия", miss_of)
            if iso not in COUNTRY_NAMES:
                names[iso] = country_name(iso, s.at[iso, "country"])
        years.append({"release": int(T), "target": f"{T + 1}–{T + 2}", "as_of": f"{T + 1}-01-01", "n": len(rows), "events": int(s.y.sum()), "rows": rows})
        details[("fxa", int(T))] = det
    if worst > 1e-6:
        raise RuntimeError(f"лента валютная: пересчёт расходится с econ_long_predictions.parquet на {worst:.1e}")
    print(f"  сверка с сохранёнными прогнозами вслепую: max|Δ| = {worst:.1e}")
    ribbons["fxa"] = {"contour": "fxa", "domain": "fx", "title": "Девальвация ≥ 25 % за год",
                      "event": "Девальвация ≥ 25 % за год (курс к доллару, декабрь к декабрю) в одном из двух лет после выпуска; страны и территории со своей валютой",
                      "horizon": "2 года", "columns": ["id", "p", "lo", "hi", "rank", "outcome"], "years": years, "extra_names": names,
                      "source": "МВФ, Global Macro Database", "model": "econ_long logit_long · логрегрессия"}
    return ribbons, details


def _compact(blocks: dict) -> dict:
    """Вклады одного исторического выпуска в сжатом виде: описание признаков — один раз,
    по строке — массивы в том же порядке (`v` значение, `c` вклад, `z`, `d` дата;
    `miss` — номера признаков без наблюдения, `clip` — номер → обрезанное значение)."""
    first = next(b for b in blocks.values() if b and b.get("kind") == "logit")
    out = {"model": first["model"], "scale": first["scale"], "intercept": first["intercept"], "base_rate": first["base_rate"],
           "n_train": first["n_train"], "beta": [x["beta"] for x in first["features"]],
           "features": [{**{k: x[k] for k in ("key", "label", "unit", "source")}, "nat": FEATURE_META[x["key"]].get("nat")}
                        for x in first["features"]], "rows": {}}
    for key, b in blocks.items():
        fs = b["features"]
        out["rows"][key] = {"p": b["p"], "lo": b["lo"], "hi": b["hi"], "eta": b["eta"],
                            "v": [x["value"] for x in fs], "c": [x["contrib"] for x in fs], "z": [x["z"] for x in fs],
                            "d": [x["date"] for x in fs], "miss": [i for i, x in enumerate(fs) if "missing" in x],
                            "clip": {str(i): x["clipped"] for i, x in enumerate(fs) if "clipped" in x}}
    return out


def _last_before(table: pd.DataFrame, gw: int, year: int):
    s = table[(table.gw == gw) & (table.year <= year)]
    return None if s.empty else int(s.year.max())


def build_history(dossier: dict) -> None:
    """История оценок вслепую и зарегистрированные события — в досье каждой страны."""
    iso_of = lambda g: GW_TO_ISO3.get(int(g))  # noqa: E731
    for short, target in (("war25", "state_war25"), ("war1000", "state_war1000")):
        w = pd.read_parquet(DATA_PROCESSED / f"pitf_{target}_predictions.parquet")
        w["iso3"] = w.gw.map(iso_of)
        for iso, g in w[w.iso3.isin(dossier)].sort_values("year").groupby("iso3"):
            # [целевой год, оценка, 1 — страна была без войны в году выпуска (начало) / 0 — воевала (продолжение), факт]
            dossier[iso].setdefault("history", {})[short] = [[int(y), num(p), int(a), int(t)] for y, p, a, t in zip(g.year, g[f"pitf_{target}"], g.at_risk, g[target])]
    e = pd.read_parquet(DATA_PROCESSED / "econ_long_predictions.parquet", columns=["iso3", "year", "logit_long", "y"])
    for iso, g in e[e.iso3.isin(dossier)].sort_values("year").groupby("iso3"):
        dossier[iso].setdefault("history", {})["fxa"] = [[int(y), num(p), int(t)] for y, p, t in zip(g.year, g.logit_long, g.y)]
    lp = pd.read_parquet(DATA_PROCESSED / "econ_long_panel.parquet", columns=["iso3", "year", "deval25", "deval"])
    for iso, g in lp[lp.iso3.isin(dossier) & (lp.deval25 == 1) & (lp.year >= 1996)].groupby("iso3"):
        dossier[iso].setdefault("history", {})["fxa_events"] = [[int(y), num(100 * v, 1)] for y, v in zip(g.year, g.deval)]
    k = pd.read_parquet(DATA_PROCESSED / "econ_kr_predictions_deval15.parquet", columns=["iso3", "month", "logit", "y"])
    for iso, g in k[k.iso3.isin(dossier)].sort_values("month").groupby("iso3"):
        dossier[iso].setdefault("history", {})["fxm"] = [[str(m), num(p), int(t)] for m, p, t in zip(g.month, g.logit, g.y)]
    kp = pd.read_parquet(DATA_PROCESSED / "econ_kr_panel_deval15.parquet", columns=["iso3", "month", "deval15"])
    kp["month"] = kp.month.astype(str)
    for iso, g in kp[kp.iso3.isin(dossier) & (kp.deval15 == 1) & (kp.month >= "1996-01")].groupby("iso3"):
        dossier[iso].setdefault("history", {})["fxm_events"] = sorted(g.month.tolist())
    b = pd.read_parquet(DATA_PROCESSED / "boe_em_predictions.parquet")
    b = b[(b.feature_set == "full") & b.jst_extree.notna()]
    for iso, g in b[b.iso.isin(dossier)].sort_values("year").groupby("iso"):
        dossier[iso].setdefault("history", {})["bank"] = [[int(y), num(p), int(c)] for y, p, c in zip(g.year, g.jst_extree, g.crisis)]


# ---------------------------------------------------------------------------
# Архив
# ---------------------------------------------------------------------------


def _window(column: str, row: pd.Series) -> tuple[str, date | None, str]:
    """Целевой период опубликованной оценки: (подпись, последний день окна, семейство)."""
    parts = column.split("_")
    if column.startswith("p_state_war"):
        y = int(parts[-1])
        return str(y), date(y, 12, 31), "war"
    if column.startswith(("fxa_", "bank_")):
        y1, y2 = 2000 + int(parts[-2]) if len(parts[-2]) == 2 else int(parts[-2]), 2000 + int(parts[-1])
        return f"{y1}–{y2}", date(y2, 12, 31), "fxa" if column.startswith("fxa_") else "bank"
    if column.startswith("fx_deval15_12m"):
        if pd.isna(row.get("as_of")):
            return "12 месяцев после даты среза", None, "fxm"
        end = pd.Period(row.as_of, freq="M") + 12
        return f"{pd.Period(row.as_of, freq='M') + 1} … {end}", end.end_time.date(), "fxm"
    raise KeyError(column)


def build_archive(risk: pd.DataFrame, today: date, T_war: int) -> dict:
    """Снимки `forecasts/*`: опубликованные оценки как есть и состояние проверки каждой.
    Исход ставится только там, где целевой период закончился И есть данные наблюдений;
    иначе — статус ожидания, никогда не ноль."""
    fact_war = risk.set_index("iso3")  # факт UCDP за год T_war — в текущей таблице риска
    out = []
    for folder in sorted(p for p in vintage.FORECASTS.iterdir() if (p / "metadata.json").exists()):
        meta = json.loads((folder / "metadata.json").read_text())
        pred = pd.read_csv(folder / "predictions.csv")
        published = date.fromisoformat(meta["generated_at"][:10])
        prob_cols = [m["column"] for m in meta["models"]] + [c for c in pred.columns if c.startswith("p_state_war1000")]
        prob_cols = list(dict.fromkeys(prob_cols))
        models, records = [], {c: {} for c in prob_cols}
        for c in prob_cols:
            m = next((x for x in meta["models"] if x["column"] == c), None)
            label, end, family = _window(c, pred.iloc[0])
            entry = {"column": c, "family": family, "model": m["model"] if m else "model_hist pitf (логрегрессия начала + продолжения)",
                     "target": m["target"] if m else "state_war1000: ≥ 1000 погибших в государственном конфликте за год",
                     "role": (m or {}).get("role", "champion"), "as_of": m["as_of"] if m else None, "window": m["window"] if m else label,
                     "n_countries": int(pred[c].notna().sum()), "out_of_sample": (m or {}).get("out_of_sample"),
                     "interval": "no_interval", "factors": "В этом выпуске вклады признаков не публиковались."}
            statuses = set()
            for _, r in pred.iterrows():
                if pd.isna(r[c]):
                    continue
                label, end, _ = _window(c, r)
                rec = {"p": num(r[c], 6), "window": label}
                rank_col = {"p_state_war25": "rank_state_war25_at_risk"}.get(c[:13], f"rank_{c}")
                if rank_col in pred and pd.notna(r[rank_col]):
                    rec["rank"] = int(r[rank_col])
                if end is not None and end >= today:
                    rec["status"] = "ongoing"
                elif family == "war" and int(c.split("_")[-1]) <= T_war and r.iso3 in fact_war.index:
                    target = "state_war25" if "war25" in c else "state_war1000"
                    rec["status"] = "resolved"
                    rec["outcome"] = int(fact_war.at[r.iso3, f"{target}_now"])
                    if target == "state_war25" and "state_war25_now" in r:
                        rec["group"] = "cont" if r.state_war25_now == 1 else "onset"
                else:
                    rec["status"] = "awaiting"
                statuses.add(rec["status"])
                records[c][r.iso3] = rec
            entry["check_status"] = sorted(statuses)
            if family == "war" and "resolved" in statuses:
                y = int(c.split("_")[-1])
                entry["outcome_source"] = {"source": f"UCDP GED {vintage.UCDP_VERSION}", "observed": f"факт {y} года", "checked_at": today.isoformat()}
                if date(y, 12, 31) < published:
                    entry["retrospective_note"] = (f"Выпуск опубликован {published.isoformat()} — после окончания целевого периода ({y}). "
                                                   "Это не проспективная проверка: исход показан для справки и в публичный трек-рекорд не входит.")
            models.append(entry)
        changes = [c for c in vintage.SCHEMA_CHANGES if meta["vintage"] in c.get("previous", {}) or c.get("since") == meta["vintage"]]
        statuses = pred[["iso3", "fx_regime", "bank_status", "as_of"]].copy()
        if "state_war25_now" in pred:
            statuses["state_war25_now"] = pred.state_war25_now
        out.append({
            "vintage": meta["vintage"], "published": meta["generated_at"], "schema_version": meta.get("schema_version", 1),
            "commit": meta["code"]["commit"], "uncommitted_changes": meta["code"].get("uncommitted_changes"), "src_sha256": meta["code"]["src_sha256"],
            "predictions_sha256": meta["predictions"]["sha256"], "rows": int(meta["predictions"]["rows"]), "origin": meta.get("origin"),
            "raw_versions": {k: v.get("version") for k, v in meta.get("raw_data", {}).items() if v.get("version")},
            "models": models, "records": records, "schema_notes": changes,
            "countries": {r.iso3: {"fx_regime": r.fx_regime, "bank_status": r.bank_status, "fx_as_of": None if pd.isna(r.as_of) else str(r.as_of),
                                   **({"state_war25_now": int(r.state_war25_now)} if "state_war25_now" in statuses else {})}
                          for r in statuses.itertuples()},
            "files": {"predictions": f"forecasts/{meta['vintage']}/predictions.csv", "metadata": f"forecasts/{meta['vintage']}/metadata.json"},
        })
    return {"vintages": out}


# ---------------------------------------------------------------------------
# Метод и данные
# ---------------------------------------------------------------------------

LAG_NOTES = {
    "gmd_annual": "Годовые макроданные Global Macro Database", "wdi_annual": "Годовые ряды Всемирного банка",
    "bis_credit_annual": "Кредит BIS, годовой", "jst_annual": "Макроистория JST", "imf_fx_december": "Курс МВФ на конец декабря",
    "ucdp_acd": "Список сторон конфликтов UCDP ACD", "imf_reserves": "Месячные резервы МВФ", "bruegel_reer": "REER Bruegel",
    "bruegel_neer": "NEER Bruegel (номинальный курс корзины)", "ucdp_events": "События UCDP (Candidate Events)",
}


def build_meta(service: dict) -> dict:
    annual = [{"source": k, "label": LAG_NOTES[k], "rule": f"значение за год Y известно с {d:02d}.{m:02d} года Y+{dy}"}
              for k, (dy, m, d) in AVAILABLE_FROM.items() if k in LAG_NOTES]
    monthly = [{"source": k, "label": LAG_NOTES[k], "rule": "известно сразу" if v == 0 else f"значение за месяц m известно к концу m+{v}"}
               for k, v in AVAILABLE_FROM_MONTHS.items() if k in LAG_NOTES]
    return {
        "lags": {"annual": annual, "monthly": monthly, "file": "src/seldon/core/temporal.py"},
        "origin": "Годовые прогнозы делаются «на 1 января»: курс и история войн — за год T, годовые макроданные — за T−1.",
        "interval_method": {
            "title": "95 % интервал оценки",
            "text": ("Для логит-моделей интервал переносит на прогноз неопределённость коэффициентов: ковариация коэффициентов — "
                     "«сэндвич»-оценка с кластерами по странам и по периодам (берётся большая из двух дисперсий), границы считаются "
                     "на шкале лог-шансов и переводятся в вероятность. Интервал не учитывает ошибку спецификации модели и выбор "
                     "регуляризации; это не интервал предсказания события."),
            "no_interval": "Для моделей на деревьях (ExtraTrees) такой интервал не определён — показывается «Интервал не рассчитан».",
            "file": "src/seldon/publish/explain.py"},
        "factors_method": ("Вклад признака — произведение коэффициента на стандартизованное значение признака (β·z), в лог-шансах. "
                           "Ноль — значение равно среднему по обучающей выборке. Вклады объясняют расчёт модели и не устанавливают причин."),
        "sources": [{"key": k, "version": vintage.VERSIONS.get(k)} for k in vintage.RAW_FILES],
        "lags_used": {"macro_annual": service["macro_lag"], "wdi": service["wdi_lag"], "imf_reserves_months": service["res_lag"]},
        "downloads": [{"label": "Таблица риска (CSV)", "path": "reports/risk_2025_26.csv"},
                      {"label": "Сводка качества (CSV)", "path": "reports/scorecard.csv"},
                      {"label": "Опубликованные выпуски", "path": "forecasts/"}],
    }


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    from seldon.domains.currency import econ_long
    from seldon.sources.gmd import load_gmd
    from seldon.sources.imf import load_dec_fx
    from seldon.sources.wdi import load_wdi

    risk_file, score_file = REPORTS / "risk_2025_26.csv", REPORTS / "scorecard.csv"
    risk, score = pd.read_csv(risk_file), pd.read_csv(score_file)
    missing = set(risk.iso3) ^ set(COUNTRY_NAMES)
    if missing:
        raise RuntimeError(f"названия стран не покрывают витрину: {sorted(missing)}")
    today = date.today()

    current, dossier, service = build_current(risk)
    service["wdi_raw"] = {k: (lambda w, k=k: w[w[k].notna()])(load_wdi(k)) for k in ("infant_mortality", "population")}
    service["long_panel_all"] = econ_long.build_panel(load_gmd(), load_dec_fx(), econ_long.load_lv(), "deval25")
    ribbons, details = build_oos(service)
    build_history(dossier)
    archive = build_archive(risk, today, service["T_war"])

    cases = []
    for c in CASES:
        yr = next(y for y in ribbons[c["contour"]]["years"] if y["release"] == c["release"])
        row = next(r for r in yr["rows"] if r[0] == c["iso3"])
        cases.append({**c, "target": yr["target"], "p": row[1], "rank": row[4], "n": yr["n"], "outcome": row[5]})

    if OUT.exists():
        shutil.rmtree(OUT)
    calc_date = datetime.fromtimestamp(risk_file.stat().st_mtime).date().isoformat()
    last_vintage = archive["vintages"][-1]["vintage"] if archive["vintages"] else None
    current["release"] = {"id": "current", "kind": "current", "calculated": calc_date, "data_cut": current["fx_monthly"]["last_month"],
                          "frozen": False, "last_public_vintage": last_vintage,
                          "note": "Текущий расчёт боевого конвейера. Он ещё не заморожен как публичный выпуск: следующий снимок — forecasts/2026-12."}
    current["cases"] = cases
    _write(OUT / "current.json", current)
    _write(OUT / "countries.json", [{"iso3": iso, "ru": ru, "en": en, "aliases": list(COUNTRY_ALIASES.get(iso, ())),
                                    "post_soviet": iso in POST_SOVIET_ISO3} for iso, (ru, en) in sorted(COUNTRY_NAMES.items(), key=lambda kv: kv[1][0])])
    for iso, d in dossier.items():
        _write(OUT / "country" / f"{iso}.json", d)
    for name, rib in ribbons.items():
        _write(OUT / "oos" / f"{name}.json", rib)
    for (name, T), det in details.items():
        _write(OUT / "oos" / name / f"{T}.json", _compact(det))
    rows = json.loads(score.to_json(orient="records", force_ascii=False))
    _write(OUT / "scorecard.json", {"rows": rows, "method": "Проверка вслепую расширяющимся окном; AUC сквозной и средний по годам; 95 % интервал — блочный бутстрэп по годам, 500 реплик.",
                                     "file": "reports/scorecard.csv"})
    _write(OUT / "archive.json", archive)
    _write(OUT / "meta.json", build_meta(service))
    inputs = {"reports/risk_2025_26.csv": _sha(risk_file), "reports/scorecard.csv": _sha(score_file)}
    for folder in sorted(vintage.FORECASTS.iterdir()):
        for n in ("predictions.csv", "metadata.json"):
            if (folder / n).exists():
                inputs[f"forecasts/{folder.name}/{n}"] = _sha(folder / n)
    _write(OUT / "manifest.json", {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "today": today.isoformat(),
                                   "inputs": inputs, "code": vintage.code_state(), "countries": len(dossier),
                                   "files": sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*.json"))})
    size = sum(p.stat().st_size for p in OUT.rglob("*.json"))
    print(f"веб-экспорт: {len(list(OUT.rglob('*.json')))} файлов, {size / 1e6:.1f} МБ → {OUT.relative_to(PROJECT_ROOT)}/")
    return 0
