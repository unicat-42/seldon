"""Jordà–Schularick–Taylor Macrohistory Database (18 развитых стран с 1870) и репозиторий
Банка Англии с их версией данных (JST R3) и готовой таблицей признаков.
Скачивание: `download jst`, `download boe`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from seldon import DATA_RAW

BOE_DIR = DATA_RAW / "boe" / "MachineLearningCrisisPrediction-master"
JST_R3 = BOE_DIR / "data" / "JSTdatasetR3.xlsx"
JST_R6 = DATA_RAW / "jst" / "JSTdatasetR6.xlsx"
BOE_DATA_FILE = BOE_DIR / "results" / "baseline" / "data_all_CV_2_crisisSHAP_.txt"


def load_jst(path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    if "rconpc" not in df.columns and "rconsbarro" in df.columns:
        df = df.rename(columns={"rconsbarro": "rconpc"})  # R6: реальное потребление на душу (Barro)
    if "stocks" not in df.columns and "eq_capgain" in df.columns:
        # R6 убрал индекс акций `stocks`; восстанавливаем уровень из прироста цен акций
        # eq_capgain (кумулятивно внутри страны) — для 2-летнего темпа роста это эквивалентно.
        df["stocks"] = df.groupby("iso")["eq_capgain"].transform(lambda s: (1 + s.fillna(0)).cumprod())
        df.loc[df.eq_capgain.isna(), "stocks"] = np.nan
    return df.sort_values(["iso", "year"]).reset_index(drop=True)


def load_boe_table() -> pd.DataFrame:
    """Готовая таблица признаков из репозитория BoE — эталон для сверки нашей сборки."""
    return pd.read_csv(BOE_DATA_FILE, sep="\t", index_col=0)
