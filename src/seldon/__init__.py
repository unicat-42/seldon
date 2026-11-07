"""Селдон: прогноз войн и экономических кризисов."""

from pathlib import Path

# Корень проекта: .../seldon (два уровня вверх от src/seldon/__init__.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
