"""Команды проекта: `uv run python -m seldon <команда> [аргументы]`.

Каждая команда — функция `main(argv) -> int` своего модуля; модуль импортируется только
при вызове (чтобы `download` не тянул sklearn, а справка печаталась мгновенно).
Порядок полного пересчёта — в `Makefile` (цель `all`).
"""

from __future__ import annotations

import importlib

# команда → (модуль, что делает)
COMMANDS: dict[str, tuple[str, str]] = {
    # sources
    "download": ("seldon.sources.download", "скачать сырые данные: download <источник>|all"),
    "lv": ("seldon.sources.lv", "хроника кризисов Laeven–Valencia из PDF статьи → data/processed"),
    # domains.conflict
    "panel": ("seldon.domains.conflict.panel", "панель страна×год/месяц по UCDP, целевые войны"),
    "features": ("seldon.domains.conflict.features", "признаки истории конфликтов: features [целевая]"),
    "features_pitf": ("seldon.domains.conflict.features_pitf", "+ режим, детская смертность, население"),
    "features_vdem": ("seldon.domains.conflict.features_vdem", "+ институты V-Dem (нужен features_pitf)"),
    "features_epr": ("seldon.domains.conflict.features_epr", "+ этническое исключение EPR (нужен features_vdem)"),
    "features_econ": ("seldon.domains.conflict.features_econ", "+ экономика (нужен econ_kr)"),
    "features_polit": ("seldon.domains.conflict.features_polit", "+ лидеры, перевороты, протесты"),
    "dyads": ("seldon.domains.conflict.dyads", "диадная панель A ← B × год: целевая и признаки пар (нужен features_pitf)"),
    "model_dyad": ("seldon.domains.conflict.model_dyad", "диадная модель вслепую + два канала риска поверх pitf (нужны dyads, model_hist pitf)"),
    "month_features": ("seldon.domains.conflict.month_features", "месячная панель войн с Candidate Events и предвестниками «дым до огня» (нужен features_pitf)"),
    "model_month": ("seldon.domains.conflict.model_month", "месячная модель войн на 3 / 6 / 12 мес: model_month [war|cf] (нужен month_features)"),
    "dyads_live": ("seldon.domains.conflict.model_dyad_live", "диады: живые стычки вместо MID, раздельные модели соседей и держав, калибровка p_ext (нужны dyads, model_dyad)"),
    "baseline": ("seldon.domains.conflict.baseline", "базовые модели войн: baseline [целевая]"),
    "model_hist": ("seldon.domains.conflict.model_hist", "модель войн: model_hist [набор] [logit|gbm] [целевая]"),
    # domains.currency
    "econ_kr": ("seldon.domains.currency.econ_kr", "месячный валютный блок: econ_kr [crisis|deval15] [all_currencies]"),
    "econ_long": ("seldon.domains.currency.econ_long", "годовой валютный блок: econ_long [deval25|gmd_cc|lv_cc] [all_currencies]"),
    "econ_struct": ("seldon.domains.currency.econ_struct", "структурные признаки против боевых валютных моделей: econ_struct [long] [kr] [ctot_rt]"),
    "econ_contagion": ("seldon.domains.currency.econ_contagion", "валютное заражение по торговым связям и параллельный курс, горизонты 3 / 6 / 12 мес: econ_contagion [3] [6] [12]"),
    # domains.banking
    "econ_boe": ("seldon.domains.banking.econ_boe", "повторение модели Банка Англии на JST (≈ 10 минут)"),
    "econ_boe_em": ("seldon.domains.banking.econ_boe_em", "модель BoE на странах BIS: econ_boe_em [lv]"),
    # evaluation, publish
    "compare_war": ("seldon.evaluation.compare_war", "парное сравнение наборов модели войн с эталоном: compare_war [эталон] [набор ...]"),
    "compare_fx": ("seldon.evaluation.compare_fx", "парное сравнение структурных наборов с валютными эталонами + кейсы (нужен econ_struct)"),
    "compare_contagion": ("seldon.evaluation.compare_contagion", "парное сравнение наборов заражения с боевой моделью и struct + контрольные кейсы (нужен econ_contagion)"),
    "compare_month": ("seldon.evaluation.compare_month", "месячная модель войн: срезы, контрольные кейсы, сравнение с conflictforecast (нужен model_month)"),
    "scorecard": ("seldon.evaluation.scorecard", "сводная таблица качества основных моделей с интервалами"),
    "risk_table": ("seldon.publish.risk_table", "таблица риска 2026–27 → reports/risk_2025_26.csv"),
    "vintage": ("seldon.publish.vintage", "заморозить прогнозы: vintage [ГГГГ-ММ] → forecasts/ГГГГ-ММ/"),
    # research — вне боевого конвейера
    "pitf": ("seldon.research.pitf", "[research] повторение модели PITF на их данных"),
    "compare_cf": ("seldon.research.compare_cf", "[research] сравнение с conflictforecast"),
    "econ_panel": ("seldon.research.econ_panel", "[research] постсоветская панель GMD + ВБ"),
    "news_ru": ("seldon.research.news_ru", "[research] газеты: Lenta.ru против EMP рубля"),
    "behavior_ru": ("seldon.research.behavior_ru", "[research] вклады и покупка валюты, Россия"),
    "behavior_world": ("seldon.research.behavior_world", "[research] бегство в валюту и GDELT, все страны"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = ["Запуск: uv run python -m seldon <команда> [аргументы]", "", "Команды:"]
    lines += [f"  {name:{width}s}  {about}" for name, (_, about) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    name, args = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"неизвестная команда {name!r}\n\n{usage()}")
        return 2
    module = importlib.import_module(COMMANDS[name][0])
    return int(module.main(args) or 0)
