"""Хроника кризисов Laeven–Valencia (2018), «Systemic Banking Crises Revisited», IMF WP/18/206.

Запуск:
    uv run python -m seldon.lv

Сайт МВФ не отдаёт базу скриптам (403), поэтому берём таблицу дат из самой статьи:
Appendix Table 1 «Crisis Dates» (страницы 30–33), файл `data/raw/lv/wp18206.pdf`
(скачан Ангелиной руками). Текст вытаскивается `pdftotext -layout`, колонки — по позиции
года в строке, переносы названий склеиваются. Покрытие 1970–2017: банковские (системные),
валютные (девальвация ≥ 30 % и ускорение ≥ 10 п.п.), суверенные дефолты и реструктуризации.

Выход:
    data/processed/lv_crises.csv — длинный формат: iso3, country_lv, type, year
    data/processed/lv_country_year.parquet — дамми по страна×год 1970–2017:
        lv_banking, lv_currency, lv_sovereign, lv_restructuring
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from seldon import DATA_PROCESSED, DATA_RAW
from seldon.download import GMD_VERSION

PDF = DATA_RAW / "lv" / "wp18206.pdf"
PAGES = (31, 34)  # страницы PDF (в статье 30–33)
YEARS = range(1970, 2018)
TYPES = ["banking", "currency", "sovereign", "restructuring"]

# название в LV → название в GMD (где отличается)
ALIASES = {
    "China, P.R.": "China", "Congo, Dem. Rep. of": "Congo DR", "Congo, Rep. of": "Republic of the Congo",
    "Côte d’Ivoire": "Ivory Coast", "Côte d'Ivoire": "Ivory Coast", "China, P.R.: Hong Kong": "Hong Kong",
    "Iran, I.R. of": "Iran", "Korea": "South Korea", "Kyrgyz Republic": "Kyrgyzstan",
    "Lao People’s Dem. Rep.": "Laos", "Lao People's Dem. Rep.": "Laos", "Gambia, The": "Gambia",
    "Russia": "Russian Federation", "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "São Tomé and Principe": "Sao Tome and Principe", "Serbia, Republic of": "Serbia",
    "Slovak Republic": "Slovakia", "Swaziland": "Eswatini", "Syrian Arab Republic": "Syria",
    "Yugoslavia, SFR": "Yugoslavia", "Central African Rep.": "Central African Republic",
}


def _bounds(page: str) -> tuple[int, int]:
    """Границы колонок по заголовку страницы: начало первого «Sovereign» и «(Restructuring)».
    Годы в колонках выровнены вправо; банковские начинаются ≤ 33-й позиции, валютные — с 37-й
    (на всех страницах), остальное зависит от страницы."""
    s1 = r1 = -1
    for line in page.split("\n")[:12]:
        if s1 < 0 and "Sovereign" in line:
            s1 = line.find("Sovereign")
        if r1 < 0 and "(Restructuring)" in line:
            r1 = line.find("(Restructuring)")
    assert s1 > 0 and r1 > 0, (s1, r1)
    return s1 - 2, r1 - 2


def _column(start: int, bounds: tuple[int, int]) -> str:
    sov, restr = bounds
    if start <= 33:
        return "banking"
    if start < sov:
        return "currency"
    if start < restr:
        return "sovereign"
    return "restructuring"


def extract_text(pdf: Path = PDF) -> str:
    """Текст страниц с таблицей, с сохранением колонок (`pdftotext -layout` из poppler).
    Парсер ниже опирается на позиции колонок именно в этой раскладке; `pdfplumber`
    (чистый Python) проверен 12.11.2026 — см. отчёт сессии аудита."""
    if not pdf.exists():
        raise FileNotFoundError(f"нет файла {pdf}: запусти `uv run python -m seldon.download lv` "
                                "или положи туда PDF статьи IMF WP/18/206 вручную")
    if shutil.which("pdftotext") is None:
        raise RuntimeError("не найдена программа `pdftotext` (пакет poppler). Установи: `brew install poppler` "
                           "(macOS) или `apt install poppler-utils` (Linux) и запусти ещё раз")
    return subprocess.run(["pdftotext", "-layout", "-f", str(PAGES[0]), "-l", str(PAGES[1]), str(pdf), "-"],
                          check=True, capture_output=True, text=True).stdout


def parse(known_names: set[str], text: str | None = None) -> pd.DataFrame:
    text = extract_text() if text is None else text
    pat = re.compile(r"\d{4}")
    records: list[dict] = []
    pending_years: list[tuple[str, int]] = []
    pending_name = ""
    lines: list[tuple[str, dict]] = []
    for page in text.split("\f"):
        if "Banking" not in page:
            continue
        bounds = _bounds(page)
        lines += [(l, bounds) for l in page.split("\n")]
    for raw, bounds in lines:
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith(("Country", "Sovereign", "(Restructuring)", "Appendix", "Table 1")):
            continue
        if re.fullmatch(r"\s*\d+\s*", line):  # номер страницы
            continue
        years = [(_column(m.start(), bounds), int(m.group())) for m in pat.finditer(line)]
        first = min((m.start() for m in pat.finditer(line)), default=len(line))
        name = line[:first].strip()
        if not name:  # только годы — относятся к следующей строке с названием
            pending_years += years
            continue
        if name[0].islower() and records:  # «of» — хвост предыдущего названия
            records[-1]["name"] += " " + name
            records[-1]["years"] += years
            continue
        if pending_name:
            name = pending_name + " " + name
            pending_name = ""
        if not years and not pending_years and name not in known_names and name not in ALIASES:
            pending_name = name  # «Bosnia and», «Dominican», «Trinidad and»
            continue
        records.append({"name": name, "years": pending_years + years})
        pending_years = []
    rows = [{"country_lv": r["name"], "type": t, "year": y} for r in records for t, y in r["years"]]
    out = pd.DataFrame(rows)
    names = pd.DataFrame({"country_lv": [r["name"] for r in records]})
    return out, names


def main(argv: list[str]) -> int:
    gmd = pd.read_stata(DATA_RAW / "gmd" / f"GMD_{GMD_VERSION}.dta", columns=["ISO3", "countryname"]).drop_duplicates()
    name_to_iso = dict(zip(gmd.countryname, gmd.ISO3))
    known = set(name_to_iso) | set(ALIASES)
    crises, names = parse(known)
    names["gmd_name"] = names.country_lv.map(lambda n: ALIASES.get(n, n))
    names["iso3"] = names.gmd_name.map(name_to_iso)
    unmatched = names[names.iso3.isna()].country_lv.tolist()
    print(f"стран в таблице: {len(names)}, не сопоставлено с GMD: {unmatched}")
    crises = crises.merge(names[["country_lv", "iso3"]], on="country_lv", how="left")
    print("кризисов по типам:", crises.groupby("type").size().to_dict(),
          "(в статье: 151 системный банковский)")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    crises.sort_values(["type", "country_lv", "year"]).to_csv(DATA_PROCESSED / "lv_crises.csv", index=False)
    # дамми страна×год для стран, которые в таблице есть (в т.ч. без кризисов)
    idx = pd.MultiIndex.from_product([names.iso3.dropna().unique(), YEARS], names=["iso3", "year"]).to_frame(index=False)
    for t in TYPES:
        d = crises[(crises.type == t) & crises.iso3.notna()][["iso3", "year"]].drop_duplicates().assign(**{f"lv_{t}": 1})
        idx = idx.merge(d, on=["iso3", "year"], how="left")
    idx = idx.fillna(0)
    for t in TYPES:
        idx[f"lv_{t}"] = idx[f"lv_{t}"].astype(int)
    idx.to_parquet(DATA_PROCESSED / "lv_country_year.parquet", index=False)
    ps = crises[crises.iso3.isin({"RUS", "UKR", "BLR", "KAZ", "GEO", "ARM", "AZE", "MDA", "KGZ", "TJK", "TKM", "UZB", "EST", "LVA", "LTU"})]
    print("\nпостсоветские:\n", ps.pivot_table(index="iso3", columns="type", values="year", aggfunc=lambda s: ",".join(map(str, sorted(s)))).fillna("").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
