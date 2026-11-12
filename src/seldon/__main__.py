"""Единая точка входа: `uv run python -m seldon <команда> [аргументы]`."""

import sys

from seldon.cli import main

sys.exit(main(sys.argv[1:]))
