"""Винтаж прогнозов: замороженный снимок витрины для проспективной проверки.

Запуск (после `make all`):
    uv run python -m seldon vintage [ГГГГ-ММ] [--force]

Создаёт папку `forecasts/ГГГГ-ММ/` (по умолчанию — текущий месяц):
    predictions.csv  прогнозы боевых моделей по странам, ранги и статусы покрытия
                     (выжимка из `reports/risk_2025_26.csv`);
    metadata.json    когда и чем посчитано: точки отсчёта (`as_of`) каждого контура, коммит
                     и контрольная сумма кода, версии и sha256 сырых файлов из `data/raw`,
                     честные метрики вслепую каждой публикуемой модели с 95 % интервалами
                     (из `reports/scorecard.csv`).

Зачем. Проверка вслепую на истории — это всё ещё оценка задним числом. Настоящая проверка —
проспективная: записать прогноз ДО того, как наступит будущее, и потом сверить. Винтаж
неизменяем: существующую папку команда не перезаписывает (нужен `--force`), а дата фиксируется
коммитом Git, который её добавляет. Ничего не обучает — только собирает готовые таблицы.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from seldon import DATA_RAW, FLOAT_FORMAT, PROJECT_ROOT, REPORTS
from seldon.publish import risk_table
from seldon.sources import bis, cow, gmd, gw, imf, jst, lv, polity, ucdp, wdi
from seldon.sources.download import UCDP_VERSION

FORECASTS = PROJECT_ROOT / "forecasts"

# Колонки снимка: что публикуем (всё остальное в таблице риска — описательные индикаторы)
COLUMNS = {
    "iso3": "код страны ISO3",
    "country": "название",
    "post_soviet": "постсоветская страна",
    "state_war25_now": "государственная война (≥ 25 погибших) в 2025 году, факт UCDP",
    "p_state_war25_2026": "вероятность государственной войны ≥ 25 погибших в 2026 году",
    "rank_state_war25_at_risk": "место среди стран без войны в 2025 (1 = самый высокий риск начала)",
    "p_state_war1000_2026": "вероятность войны ≥ 1000 погибших в 2026 году",
    "fx_regime": "валюта: own — своя; код (EUR, USD…) — общая/чужая, прогноза нет; no_imf_fx — нет курса в данных МВФ",
    "fxa_logit_2026_27": "вероятность девальвации ≥ 25 % (декабрь к декабрю) в 2026 или 2027, логрегрессия",
    "rank_fxa_logit_2026_27": "место по этой вероятности (1 = самый высокий риск)",
    "fxa_extree_2026_27": "то же, деревья (ExtraTrees)",
    "rank_fxa_extree_2026_27": "место по деревьям",
    "as_of": "последний месяц с курсом МВФ — точка отсчёта месячного прогноза",
    "fx_deval15_12m": "вероятность скачка курса ≥ 15 % за месяц в ближайшие 12 месяцев, логрегрессия",
    "rank_fx_deval15_12m": "место по этой вероятности",
    "fx_deval15_12m_live": "то же событие и окно, модель-претендент live_fx_swap (боевая не заменена; копится для сравнения)",
    "rank_fx_deval15_12m_live": "место по вероятности претендента",
    "bank_status": "банковский контур: ok | no_yield_curve | no_broad_money | no_bis_credit",
    "bank_jst_extree_full_2026_27": "вероятность системного банковского кризиса в 2026 или 2027 году (деревья BoE, набор full)",
    "rank_bank_jst_extree_full_2026_27": "место по этой вероятности",
}

# Публикуемые модели: колонка снимка → строка в scorecard (block, sample, model) и постановка
MODELS = [
    {"column": "p_state_war25_2026", "contour": "война, год", "model": "model_hist pitf (логрегрессия начала + продолжения)",
     "target": "state_war25: ≥ 25 погибших в государственном конфликте на своей территории или у соседа, где страна — сторона",
     "as_of": "2026-01-01", "features_year": 2025, "window": "2026",
     "scorecard": ("война: начало state_war25 в T+1", "мир, страны без войны в T", "pitf_state_war25"),
     "note": "UCDP GED 26.1 — по 2025 год. Оценка вслепую — для стран без войны в году T (начало войны)."},
    {"column": "fxa_logit_2026_27", "contour": "валютный кризис, год", "model": "econ_long logit_long",
     "target": "deval25: курс к доллару на конец декабря вырос ≥ 25 % за год после ≥ 2 спокойных лет",
     "as_of": "2026-01-01", "features_year": 2025, "window": "2026–2027",
     "scorecard": ("валютный кризис, год: deval25 в t+1…t+2", "мир", "logit_long"),
     "note": "Курс — декабрь 2025, макроданные GMD — 2024 (лаг публикации)."},
    {"column": "fxa_extree_2026_27", "contour": "валютный кризис, год", "model": "econ_long extree_long",
     "target": "deval25", "as_of": "2026-01-01", "features_year": 2025, "window": "2026–2027",
     "scorecard": ("валютный кризис, год: deval25 в t+1…t+2", "мир", "extree_long"), "note": ""},
    {"column": "fx_deval15_12m", "contour": "валютный кризис, месяц", "model": "econ_kr logit, целевая deval15",
     "target": "deval15: курс к доллару за месяц вырос ≥ 15 % (первый скачок после ≥ 12 спокойных месяцев)",
     "as_of": "колонка as_of (последний месяц с курсом МВФ по стране)", "features_year": None, "window": "12 месяцев после as_of",
     "scorecard": ("валютный кризис, месяц: deval15 за 12 мес", "мир", "logit"),
     "note": "На постсоветской подвыборке модель не лучше монетки — см. вторую строку метрик."},
    {"column": "fx_deval15_12m_live", "role": "challenger", "contour": "валютный кризис, месяц",
     "model": "econ_csd live_fx_swap: индикаторы econ_kr без линейного reserves_g12 + peg_share12, reer_gap, reer_peg, peg_resfall",
     "target": "deval15 — как у fx_deval15_12m",
     "as_of": "колонка as_of (последний месяц с курсом МВФ по стране)", "features_year": None, "window": "12 месяцев после as_of",
     "scorecard": ("валютный кризис, месяц: deval15 за 12 мес, претендент", "мир", "live_fx_swap"),
     "note": "Претендент, не замена: публикуется рядом с боевой моделью, чтобы сравнить их на будущих данных. "
             "На живом крае REER и резервы дотянуты последним значением не старше 3 месяцев; в проверке вслепую такого "
             "дотягивания нет. На постсоветской подвыборке хуже монетки — см. weak_spots."},
    {"column": "bank_jst_extree_full_2026_27", "contour": "банковский кризис", "model": "econ_boe_em jst_extree, набор full",
     "target": "как у Bluwstein и др.: строка T = 1, если T — один из двух лет перед системным банковским кризисом; "
               "признаки в строке T — макроданные за T−1, ставки за T",
     "as_of": "2026-01-01", "features_year": 2025, "window": "кризис в 2026–2027 по метке BoE (год T+1 или T+2)",
     "scorecard": ("банковский кризис в t+1…t+2, набор full", "страны BIS вне JST", "jst_extree"),
     "note": "Узкое покрытие: только страны BIS с длинной ставкой и своей денежной массой; в оценке вслепую вне JST — 9 предкризисных лет."},
]
EXTRA_SCORECARD = [  # строки, которые обязаны быть рядом: где модель слаба
    ("валютный кризис, год: deval25 в t+1…t+2", "постсоветские", "logit_long"),
    ("валютный кризис, месяц: deval15 за 12 мес", "постсоветские", "logit"),
    ("валютный кризис, месяц: deval15 за 12 мес, претендент", "постсоветские", "live_fx_swap"),
    ("банковский кризис в t+1…t+2, набор nomoney", "страны BIS вне JST", "jst_extree"),
]

# Схема снимка (решение Ангелины 13.11.2026, развилка B4): колонки только добавляются,
# существующие не переименовываются и не меняют смысл молча. Всё, что отличает снимок от
# предыдущих, записывается сюда и попадает в `metadata.json` — чтобы сравнивать по месяцам.
SCHEMA_VERSION = 2
SCHEMA_CHANGES = [
    {"since": "2026-12", "kind": "new_window", "columns": ["p_state_war25_2026", "p_state_war1000_2026"],
     "previous": {"2026-11": ["p_state_war25_2025", "p_state_war1000_2025"]},
     "note": "Не переименование: в снимке 2026-11 прогноз войны был на 2025 год (признаки 2024, UCDP 25.1), "
             "с чекпоинта 3.1 — на 2026 (признаки 2025, UCDP 26.1). Значения между снимками не сравнимы напрямую."},
    {"since": "2026-12", "kind": "method", "columns": ["fx_deval15_12m", "rank_fx_deval15_12m"],
     "note": "Лаг публикации месячных резервов МВФ (imf_reserves = 2 месяца, коммит e3f065d): reserves_g12, "
             "reserves_g1, m2_res_g12 на строке месяца m — по m−2; emp_z_max3 для двух свежих месяцев — только курсовая "
             "часть; world_crisis3 кончается в m−2. Модель и целевая те же; вслепую сквозной AUC 0,640 → 0,643."},
    {"since": "2026-12", "kind": "new_column", "columns": ["fx_deval15_12m_live", "rank_fx_deval15_12m_live"],
     "note": "Модель-претендент рядом с боевой (см. models, role = challenger)."},
]

RAW_FILES = {
    "ucdp": [ucdp.GED_FILE, ucdp.ACD_FILE],
    "gw": [gw.GW_FILE],
    "cow": [cow.CONTIGUITY_FILE],
    "polity": [polity.POLITY_FILE],
    "wdi": [wdi.WDI_DIR / f"{n}.csv" for n in ("infant_mortality", "population", "cpi_annual")],
    "imf": [imf.IMF / f"{n}.csv" for n in ("exchange_rates", "reserves", "cpi", "interest_rates", "broad_money", "exports")],
    "gmd": [gmd.GMD_FILE],
    "bis": [bis.BIS_CREDIT_FILE],
    "jst": [jst.JST_R6],
    "lv": [lv.PDF],
}
VERSIONS = {"ucdp": UCDP_VERSION, "gmd": gmd.GMD_VERSION, "jst": "R6", "polity": "Polity5 (2018)",
            "cow": "Direct Contiguity 3.2", "lv": "IMF WP/18/206"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def code_state() -> dict:
    """Чем посчитано: коммит, на котором стояло дерево, и контрольная сумма самого кода.
    Снимок кладётся в тот же коммит, что и код, поэтому «свой» хеш коммита записать нельзя:
    `commit` — предыдущий коммит, а код однозначно опознаётся по `src_sha256`."""
    h = hashlib.sha256()
    for f in sorted((PROJECT_ROOT / "src" / "seldon").rglob("*.py")):
        h.update(f.relative_to(PROJECT_ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return {"commit": _git("rev-parse", "HEAD"), "uncommitted_changes": bool(_git("status", "--porcelain", "--", "src", "tests")),
            "src_sha256": h.hexdigest(),
            "note": "винтаж фиксируется коммитом Git, который добавляет эту папку; commit — коммит, поверх которого он собран"}


def raw_inventory() -> dict:
    out = {}
    for source, files in RAW_FILES.items():
        manifest_file = DATA_RAW / source / "manifest.json"
        manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
        entries = []
        for f in files:
            m = manifest.get(f.name, {})
            entries.append({"file": f.relative_to(DATA_RAW).as_posix(), "bytes": f.stat().st_size, "sha256": sha256_of(f),
                            "downloaded_at": m.get("downloaded_at"), "url": m.get("url")})
        out[source] = {"version": VERSIONS.get(source), "files": entries}
    return out


def _score_row(score: pd.DataFrame, key: tuple[str, str, str]) -> dict:
    r = score[(score.block == key[0]) & (score["sample"] == key[1]) & (score.model == key[2])]
    if len(r) != 1:
        raise RuntimeError(f"в reports/scorecard.csv нет строки {key}: сначала `make all`")
    r = r.iloc[0]
    out = {"block": key[0], "sample": key[1], "model": key[2], "n": int(r.n), "positives": int(r.positives),
           "base_rate": round(float(r.positives) / float(r.n), 4), "years": r.years,
           "auc_pooled": float(r.auc), "auc_pooled_ci95": [float(r.auc_lo), float(r.auc_hi)],
           "auc_within_year": float(r.auc_within_year), "brier": float(r.brier)}
    if pd.notna(r.get("auc_vs_ref")):
        out["vs_baseline"] = {"baseline": r.ref, "auc_diff": float(r.auc_vs_ref), "ci95": [float(r.auc_vs_ref_lo), float(r.auc_vs_ref_hi)]}
    return out


def build(name: str) -> tuple[pd.DataFrame, dict]:
    risk_file, score_file = REPORTS / "risk_2025_26.csv", REPORTS / "scorecard.csv"
    risk, score = pd.read_csv(risk_file), pd.read_csv(score_file)
    pred = risk[list(COLUMNS)].copy()
    for c in [c for c in pred.columns if c.startswith("rank_")]:
        pred[c] = pred[c].astype("Int64")  # места — целые; пусто, где прогноза нет
    models = []
    for m in MODELS:
        e = {"role": "champion", **{k: v for k, v in m.items() if k != "scorecard"}}
        e["n_countries"] = int(pred[m["column"]].notna().sum())
        e["out_of_sample"] = _score_row(score, m["scorecard"])
        models.append(e)
    meta = {
        "vintage": name,
        "schema_version": SCHEMA_VERSION,
        "schema_changes": SCHEMA_CHANGES,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "what": "Замороженные прогнозы проекта «Селдон». Вероятности, а не «да/нет»; события редкие.",
        "origin": "годовые прогнозы — «на 1 января»: курс и история войн за год T, годовые макроданные за T−1 (лаги публикации, src/seldon/core/temporal.py)",
        "code": code_state(),
        "inputs": {"risk_table": {"file": "reports/risk_2025_26.csv", "sha256": sha256_of(risk_file)},
                   "scorecard": {"file": "reports/scorecard.csv", "sha256": sha256_of(score_file)}},
        "predictions": {"file": "predictions.csv", "rows": len(pred), "columns": COLUMNS,
                        "coverage": {"fx_regime": pred.fx_regime.value_counts().to_dict(),
                                     "bank_status": pred.bank_status.value_counts().to_dict()}},
        "fx_monthly_as_of": {"last_month_in_data": risk_table.FX_LAST_MONTH, "by_country_min": str(pred.as_of.dropna().min()),
                             "by_country_max": str(pred.as_of.dropna().max())},
        "models": models,
        "weak_spots": [_score_row(score, k) for k in EXTRA_SCORECARD],
        "metrics_method": "проверка вслепую расширяющимся окном; AUC сквозной (по всем страна-периодам) и средний по годам; "
                          "95 % интервал — блочный бутстрэп по годам, 500 реплик (src/seldon/core/metrics.py)",
        "raw_data": raw_inventory(),
    }
    return pred, meta


def main(argv: list[str]) -> int:
    names = [a for a in argv if not a.startswith("--")]
    return freeze(names[0] if names else datetime.now().strftime("%Y-%m"), force="--force" in argv)


def freeze(name: str, force: bool = False, root: Path = FORECASTS) -> int:
    """Заморозить снимок `name` в папке `root / name`. Существующую непустую папку не трогает
    (код возврата 2), если не задан `force`."""
    folder = root / name
    if folder.exists() and any(folder.iterdir()) and not force:
        print(f"винтаж {name} уже существует ({folder}) — он неизменяем. Для нового месяца укажи другое имя; "
              "перезапись только с `--force` и только пока винтаж не закоммичен.")
        return 2
    pred, meta = build(name)
    folder.mkdir(parents=True, exist_ok=True)
    pred.to_csv(folder / "predictions.csv", index=False, float_format=FLOAT_FORMAT)
    meta["predictions"]["sha256"] = sha256_of(folder / "predictions.csv")
    (folder / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"винтаж {name}: {len(pred)} стран → {folder}/predictions.csv, metadata.json")
    for m in meta["models"]:
        o = m["out_of_sample"]
        print(f"  {m['column']:32s} стран {m['n_countries']:3d}; вслепую AUC {o['auc_pooled']:.3f} "
              f"[{o['auc_pooled_ci95'][0]:.3f}; {o['auc_pooled_ci95'][1]:.3f}], внутри года {o['auc_within_year']:.3f}, база {o['base_rate']:.3f}")
    return 0
