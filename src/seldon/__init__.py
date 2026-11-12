"""Селдон: прогноз войн и экономических кризисов.

Слои (после Этапа 2, чекпоинт 2.1):
    core/        общие кирпичи: коды стран, панели, контракт `as_of`, калибровка, метрики
    sources/     загрузка и чтение сырых данных (единственное место, где читается data/raw)
    domains/     признаки, целевые и модели по блокам: conflict, currency, banking
    evaluation/  единый бэктест и сводная таблица качества (`scorecard`)
    publish/     витрина: таблица риска
    research/    разовые исследовательские блоки вне боевого конвейера

Запуск — одна точка входа: `uv run python -m seldon <команда>` (список — без аргументов).
"""

from pathlib import Path

# Корень проекта: .../seldon (два уровня вверх от src/seldon/__init__.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"

# Все таблицы в reports/ и forecasts/ пишутся с 6 знаками после запятой: многопоточные
# деревья (`ExtraTrees`, `n_jobs=-1`) дают шум в 17-м знаке от запуска к запуску, и без
# округления повторный `make all` менял бы файлы в git.
FLOAT_FORMAT = "%.6f"
