"""Скачивание сырых данных первого этапа.

Запуск:
    uv run python -m seldon download ucdp
    uv run python -m seldon download pitf
    uv run python -m seldon download conflictforecast
    uv run python -m seldon download gw
    uv run python -m seldon download cow
    uv run python -m seldon download polity
    uv run python -m seldon download wdi
    uv run python -m seldon download jst
    uv run python -m seldon download boe
    uv run python -m seldon download gmd
    uv run python -m seldon download imf
    uv run python -m seldon download bis
    uv run python -m seldon download reign
    uv run python -m seldon download massmob
    uv run python -m seldon download lenta
    uv run python -m seldon download cbr
    uv run python -m seldon download lv
    uv run python -m seldon download vdem
    uv run python -m seldon download epr
    uv run python -m seldon download all

Файлы кладутся в data/raw/<источник>/. Уже скачанное не качается повторно
(удали файл, чтобы перекачать). Рядом с файлами пишется manifest.json:
откуда, когда и какого размера — чтобы потом можно было понять, какая версия данных
лежала в момент эксперимента.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

from seldon import DATA_RAW
from seldon.sources.gmd import GMD_VERSION
from seldon.sources.epr import EPR_VERSION
from seldon.sources.ucdp import UCDP_VERSION

# ---------------------------------------------------------------------------
# Источники
# ---------------------------------------------------------------------------

UCDP_V = UCDP_VERSION.replace(".", "")  # "251" — так в именах файлов на сайте

SOURCES: dict[str, dict] = {
    # UCDP — Uppsala Conflict Data Program. Три набора:
    #  - GED: Georeferenced Event Dataset, события с погибшими, с 1989 года, с координатами;
    #  - UCDP/PRIO ACD: Armed Conflict Dataset, конфликты по годам, с 1946 года;
    #  - BRD: Battle-Related Deaths, число погибших в боях по конфликтам и годам, с 1989 года.
    "ucdp": {
        "files": {
            f"ged{UCDP_V}-csv.zip": f"https://ucdp.uu.se/downloads/ged/ged{UCDP_V}-csv.zip",
            f"ucdp-prio-acd-{UCDP_V}-csv.zip": f"https://ucdp.uu.se/downloads/ucdpprio/ucdp-prio-acd-{UCDP_V}-csv.zip",
            f"ucdp-brd-conf-{UCDP_V}-csv.zip": f"https://ucdp.uu.se/downloads/brd/ucdp-brd-conf-{UCDP_V}-csv.zip",
        },
        "unzip": True,
    },
    # PITF — Political Instability Task Force. Replication data к статье
    # Goldstone et al. (2010) "A Global Model for Forecasting Political Instability",
    # AJPS. Основной файл — Stata .dta, к нему скрипт .do и исходные таблицы .xls.
    "pitf": {
        "files": {
            "PITF-Global-Model-Replication-Data-121709.dta": "https://scip.gmu.edu/wp-content/uploads/2023/12/PITF-Global-Model-Replication-Data-121709.dta",
            "PITF-Global-Forecasting-Model-Replication-Script.do": "https://scip.gmu.edu/wp-content/uploads/2023/12/PITF-Global-Forecasting-Model-Replication-Script.do",
            "AJPS-Table-1-and-3-input-data-Set-A2.xls": "https://scip.gmu.edu/wp-content/uploads/2023/11/AJPS-Table-1-and-3-input-data-Set-A2.xls",
            "AJPS-Table-1-and-3-input-data-Set-B2.xls": "https://scip.gmu.edu/wp-content/uploads/2023/11/AJPS-Table-1-and-3-input-data-Set-B2.xls",
            "AJPS-Table-1-and-3-input-data-Set-C2.xls": "https://scip.gmu.edu/wp-content/uploads/2023/11/AJPS-Table-1-and-3-input-data-Set-C2.xls",
            "Table-1-Hits-and-misses-Global-model.xls": "https://scip.gmu.edu/wp-content/uploads/2023/11/Table-1-Hits-and-misses-tables_Global-foresting-model.xls",
            "Table-3-Hits-and-misses-Fearon-Laitin-model.xls": "https://scip.gmu.edu/wp-content/uploads/2023/11/Table-3-Hits-and-misses-tables_Fearon_Laitin-model.xls",
            "Table-3-Hits-and-misses-Global-model-FL-data.xls": "https://scip.gmu.edu/wp-content/uploads/2023/11/Table-3-Hits-and-misses-tables_Global-foresting-model-with-the-Same-Data-Set-as-Fearon-and-Laitin-Model.xls",
        },
        "unzip": False,
    },
    # conflictforecast.org — Mueller & Rauh. Список свежих файлов отдаёт открытый API
    # (без ключа), сами файлы — CSV. Список запрашивается при каждом запуске.
    "conflictforecast": {
        "listing_api": (
            "https://api.backendless.com/C177D0DC-B3D5-818C-FF1E-1CC11BC69600/"
            "C5F2917E-C2F6-4F7D-9063-69555274134E/services/fileService/get-latest-file-listing"
        ),
        "unzip": False,
    },
    # Gleditsch & Ward — список независимых государств с датами существования.
    # Нужен, чтобы в панели были все страны мира, включая «тихие» без единого
    # события в UCDP (Беларусь, Казахстан, Балтия). Коды те же, что в UCDP.
    "gw": {
        "files": {"iisystem.dat": "http://ksgleditsch.com/data/iisystem.dat"},
        "unzip": False,
    },
    # Correlates of War — Direct Contiguity 3.2: кто с кем граничит (суша / море до N миль),
    # по годам 1816–2016. Нужен для правила «помогает соседу» в целевой war25 (panel.py)
    # и для признака «конфликт у соседей». Коды COW ≈ Gleditsch–Ward, расхождения — в panel.py.
    "cow": {
        "files": {
            "DirectContiguity320.zip": "https://correlatesofwar.org/wp-content/uploads/DirectContiguity320.zip",
        },
        "unzip": True,
    },
    # Polity5 (Center for Systemic Peace) — тип политического режима по странам и годам,
    # 1800–2018. Нужен для признаков PITF: polity2, parcomp (фракционность), exrec.
    "polity": {
        "files": {"p5v2018.xls": "https://www.systemicpeace.org/inscr/p5v2018.xls"},
        "unzip": False,
    },
    # Всемирный банк, World Development Indicators — через открытый API, без ключа.
    # Детская смертность (признак PITF) и население (для «погибших на душу»).
    "wdi": {
        "indicators": {
            "SP.DYN.IMRT.IN": "infant_mortality",  # смертность до 1 года на 1000 рождений
            "SP.POP.TOTL": "population",
            # для блока «экономика»: кредит частному сектору (% ВВП) из Global Financial
            # Development (FD.*, с 1990-х для постсоветских; у FS.AST.PRVT.GD.ZS ряды короче) —
            # замена JST tloans/gdp для стран вне JST; резервы в долларах
            "FD.AST.PRVT.GD.ZS": "credit_gdp",
            "FI.RES.TOTL.CD": "reserves_usd",
            # годовой ИПЦ (2010 = 100) — затыкать дыры в месячном ИПЦ МВФ (econ_kr.py):
            # у Украины, Беларуси, Казахстана месячный ИПЦ у МВФ только с 2005
            "FP.CPI.TOTL": "cpi_annual",
        },
        "api": "https://api.worldbank.org/v2/country/all/indicator/{ind}?format=json&per_page=20000&date=1960:2025",
    },
    # Jordà–Schularick–Taylor Macrohistory Database (macrohistory.net), релиз R6:
    # 18 развитых стран с 1870 года, по годам. Кредит, ВВП, цены, ставки, счёт текущих
    # операций, цены на жильё и дамми системного финансового кризиса (crisisJST).
    # Данные модели Банка Англии (Bluwstein и др.). Без регистрации, CC BY-NC-SA.
    "jst": {
        "files": {
            "JSTdatasetR6.xlsx": "https://www.macrohistory.net/app/download/9834512569/JSTdatasetR6.xlsx?t=1763503850",
            "JST_documentationR6.pdf": "https://www.macrohistory.net/app/download/9834516169/JST_documentationR6.pdf?t=1676279836",
            "JSTcrisis_chronology.pdf": "https://www.macrohistory.net/app/download/9844625569/JSTcrisis_chronology.pdf",
        },
        "unzip": False,
    },
    # Банк Англии — код к Staff Working Paper 848 (Bluwstein, Buckmann, Joseph, Kapadia,
    # Şimşek, 2020; позже Journal of International Economics 2023): Python + R. Берём
    # архив репозитория целиком: в нём лежит JST R3 (data/JSTdatasetR3.xlsx) — ровно тот
    # файл, на котором считалась статья, — и скрипты подготовки признаков (make_data.py).
    "boe": {
        "files": {
            "MachineLearningCrisisPrediction-master.zip": "https://github.com/bank-of-england/MachineLearningCrisisPrediction/archive/refs/heads/master.zip",
        },
        "unzip": True,
    },
    # Global Macro Database (Müller, Xu, Lehbib, Chen; globalmacrodata.com) — 46 рядов по
    # 239 странам, в т.ч. постсоветским. Файл Stata лежит на S3 авторов, тот же, что
    # качает их пакет `global_macro_data`; регистрация не нужна. CC BY-NC-SA.
    "gmd": {
        "files": {
            "versions.csv": "https://gmd-releases.s3.ap-southeast-2.amazonaws.com/data/helpers/versions.csv",
            "varlist.csv": "https://raw.githubusercontent.com/KMueller-Lab/Global-Macro-Database/refs/heads/main/data/helpers/varlist.csv",
            f"GMD_{GMD_VERSION}.dta": f"https://gmd-releases.s3.ap-southeast-2.amazonaws.com/data/distribute/GMD_{GMD_VERSION}.dta",
        },
        "unzip": False,
    },
    # REIGN (Rulers, Elections, and Irregular Governance; One Earth Future) — лидеры, выборы,
    # перевороты (вкл. Powell–Thyne) по странам и месяцам, 1950–2021. Проект закрыт в 2021,
    # последний файл лежит на GitHub. Коды стран — COW (≈ Gleditsch–Ward).
    "reign": {
        "files": {"REIGN_2021_8.csv": "https://raw.githubusercontent.com/OEFDataScience/REIGN.github.io/gh-pages/data_sets/REIGN_2021_8.csv"},
        "unzip": False,
    },
    # Mass Mobilization (Clark & Regan) — антиправительственные протесты по событиям,
    # 1990–2020, Harvard Dataverse (открытый файл, без логина). Коды стран — COW.
    "massmob": {
        "files": {"mmALL_073120_csv.tab": "https://dataverse.harvard.edu/api/access/datafile/4291456?format=original"},
        "unzip": False,
    },
    # Lenta.ru — открытый корпус новостей 1999-09…2019-12 (~800 тыс. статей; Yutkin, GitHub).
    # Поведенческий слой: о чём писали газеты за месяцы до валютных кризисов в России.
    # Тексты в git не кладём и не воспроизводим — только месячные частоты словарей.
    "lenta": {
        "files": {"lenta-ru-news.csv.bz2": "https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2"},
        "unzip": False,
    },
    # ЦБ РФ — вклады физических лиц по валютам: помесячно с 2012 (xlsx) и по годам 2007–2018
    # (HTML-таблицы «Данные об объемах привлеченных кредитными организациями вкладов»).
    # Долларизация вкладов населения — поведенческий слой (behavior_ru.py). Без ключа.
    "cbr": {
        "files": {
            "funds_all.xlsx": "https://www.cbr.ru/vfs/statistics/BankSector/Borrowings/02_01_Funds_all.xlsx",
            **{f"deposits_20{y:02d}.html": f"https://www.cbr.ru/statistics/bank_system/4-2-1a_{y:02d}/" for y in range(7, 19)},
        },
        "unzip": False,
    },
    # EPR (Ethnic Power Relations, ETH Zürich) — доступ этнических групп к власти и
    # трансграничная родня (TEK). Прямые CSV, без регистрации; см. `sources/epr.py`.
    "epr": {
        "files": {
            f"EPR-{EPR_VERSION}.csv": f"https://icr.ethz.ch/data/epr/core/EPR-{EPR_VERSION}.csv",
            f"TEK-{EPR_VERSION}.csv": f"https://icr.ethz.ch/data/epr/tek/TEK-{EPR_VERSION}.csv",
        },
        "unzip": False,
        "sha256": {
            "EPR-2023.csv": "8b4637484fb0963e6fc72306139deab655145589c9c851f312e8b231ca66cbd7",
            "TEK-2023.csv": "26b2689eb49673629a773d95c380266cf79514beb54647d5233c6a18f54a1c96",
        },
    },
    # BIS — кредит частному нефинансовому сектору (total credit, WS_TC): ~45 стран, в т.ч.
    # развивающиеся (Россия, Турция, Бразилия, Китай...), поквартально с 1940–1990-х, % ВВП.
    # Длиннее рядов ВБ на 10–20 лет у развитых и с поправками на разрывы. Открытый bulk-архив.
    "bis": {
        "files": {"WS_TC_csv_flat.zip": "https://data.bis.org/static/bulk/WS_TC_csv_flat.zip"},
        "unzip": True,
    },
    # МВФ — новый портал данных (data.imf.org, SDMX 3.0 API, без ключа; старый IFS-сервис
    # dataservices.imf.org закрыт в 2025). Бывший IFS разложен по наборам: ER — курсы,
    # IL — резервы, MFS_IR — ставки, CPI — цены. Месячные ряды по всем странам для модели
    # валютных кризисов Kaminsky–Reinhart. Фильтры — через параметры c[...]: ключ после
    # первого измерения этот API игнорирует (проверено 10.11.2026).
    "imf": {
        "api": "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/{flow}/+/*",
        "flows": {
            # имя файла: (dataflow, фильтры)
            "exchange_rates": ("ER", {"INDICATOR": "XDC_USD", "FREQUENCY": "M"}),
            "reserves": ("IL", {"INDICATOR": "RXF11_REVS,TRGMV_REVS", "UNIT": "USD", "FREQUENCY": "M"}),
            "interest_rates": ("MFS_IR", {
                "INDICATOR": ",".join([
                    "MMRT_RT_PT_A_PT",     # ставка денежного рынка — главная у KR
                    "DISR_RT_PT_A_PT",     # учётная ставка ЦБ
                    "MFS166_RT_PT_A_PT",   # ключевая (monetary policy-related)
                    "MFS171_RT_PT_A_PT",   # ставка рефинансирования
                    "IRTBILL_RT_PT_A_PT",  # казначейские векселя
                    "MFS135_RT_PT_A_PT",   # депозитная
                    "MFS162_RT_PT_A_PT",   # кредитная
                ]),
                "FREQUENCY": "M",
            }),
            "cpi": ("CPI", {"INDEX_TYPE": "CPI", "COICOP_1999": "_T",
                            "TYPE_OF_TRANSFORMATION": "IX", "FREQUENCY": "M"}),
            # широкая денежная масса в нацвалюте (для M2/резервы) и экспорт товаров FOB в долл.
            "broad_money": ("MFS_MA", {"INDICATOR": "BM_MAI", "UNIT": "XDC", "FREQUENCY": "M"}),
            "exports": ("ITG", {"INDICATOR": "XG", "TYPE_OF_TRANSFORMATION": "FOB_USD", "FREQUENCY": "M"}),
            # платёжный баланс, поквартально: «наличная валюта и депозиты» прочих секторов
            # (домохозяйства + нефинансовые компании) — покупка валюты и вывод на счета за рубеж.
            # Поведенческий слой: бегство в валюту (behavior_ru.py).
            "bop_cash_deposits": ("BOP", {"INDICATOR": "O_F2_S1V,O_F2_S1Z", "FREQUENCY": "Q"}),
        },
        "accept": "application/vnd.sdmx.data+csv;version=2.0.0",
    },
}

HEADERS = {"User-Agent": "seldon-research/0.1 (personal research project)"}


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------


def download_file(url: str, dest: Path) -> dict:
    """Скачать url в dest с полосой прогресса. Возвращает запись для манифеста."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  есть, пропускаю: {dest.name}")
        return {"url": url, "file": dest.name, "bytes": dest.stat().st_size, "skipped": True}

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"  {dest.name}", leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    print(f"  скачано: {dest.name} ({dest.stat().st_size / 1e6:.1f} МБ)")
    return {
        "url": url,
        "file": dest.name,
        "bytes": dest.stat().st_size,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def unzip(path: Path) -> list[str]:
    """Распаковать zip рядом с собой. Возвращает список файлов внутри."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        for n in names:
            if not (path.parent / n).exists():
                z.extract(n, path.parent)
    return names


def write_manifest(folder: Path, entries: list[dict]) -> None:
    manifest = folder / "manifest.json"
    old = json.loads(manifest.read_text()) if manifest.exists() else {}
    for e in entries:
        if e.get("skipped") and e["file"] in old:
            continue  # не затираем дату скачивания
        old[e["file"]] = {k: v for k, v in e.items() if k != "skipped"}
    manifest.write_text(json.dumps(old, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Источники
# ---------------------------------------------------------------------------


def fetch_static(name: str) -> None:
    """Источник с фиксированным списком файлов (ucdp, pitf)."""
    spec = SOURCES[name]
    folder = DATA_RAW / name
    print(f"[{name}] → {folder}")
    entries = []
    for fname, url in spec["files"].items():
        entry = download_file(url, folder / fname)
        if spec["unzip"] and fname.endswith(".zip"):
            entry["contents"] = unzip(folder / fname)
        want = spec.get("sha256", {}).get(fname)
        if want:
            entry["sha256"] = sha256_of(folder / fname)
            if entry["sha256"] != want:
                raise RuntimeError(f"контрольная сумма {fname} не совпала: ожидалось {want}, получено {entry['sha256']} — "
                                   "файл на сервере изменился, проверь версию")
        entries.append(entry)
    write_manifest(folder, entries)


def fetch_conflictforecast() -> None:
    spec = SOURCES["conflictforecast"]
    folder = DATA_RAW / "conflictforecast"
    print(f"[conflictforecast] → {folder}")
    r = requests.get(spec["listing_api"], headers=HEADERS, timeout=60)
    r.raise_for_status()
    listing = r.json()
    (folder / "latest").mkdir(parents=True, exist_ok=True)
    (folder / "listing.json").write_text(json.dumps(listing, indent=2))

    # У publicUrl (backendlessappcontent.com) на 07.11.2026 просрочен SSL-сертификат.
    # Те же файлы отдаёт основной хост api.backendless.com по относительному пути `url`.
    base = spec["listing_api"].split("/services/")[0] + "/files/"
    entries = []
    for item in listing:
        rel = item.get("url")
        if not rel:
            continue
        fname = item.get("name") or rel.rsplit("/", 1)[-1]
        entries.append(download_file(base + rel, folder / "latest" / fname))
    write_manifest(folder, entries)


def fetch_wdi() -> None:
    """Всемирный банк: каждый индикатор — в CSV страна-год (iso3, country, year, value)."""
    spec = SOURCES["wdi"]
    folder = DATA_RAW / "wdi"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"[wdi] → {folder}")
    entries = []
    for ind, name in spec["indicators"].items():
        dest = folder / f"{name}.csv"
        url = spec["api"].format(ind=ind)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  есть, пропускаю: {dest.name}")
            entries.append({"url": url, "file": dest.name, "bytes": dest.stat().st_size, "skipped": True})
            continue
        r = requests.get(url, headers=HEADERS, timeout=120)
        r.raise_for_status()
        meta, rows = r.json()
        if meta.get("pages", 1) != 1:
            raise RuntimeError(f"WDI {ind}: ответ на {meta['pages']} страниц, ожидалась одна")
        with open(dest, "w") as f:
            f.write("iso3,country,year,value\n")
            for row in rows:
                if row["value"] is None:
                    continue
                f.write(f'{row["countryiso3code"]},"{row["country"]["value"]}",{row["date"]},{row["value"]}\n')
        print(f"  скачано: {dest.name} ({len(rows)} строк)")
        entries.append({
            "url": url, "file": dest.name, "bytes": dest.stat().st_size,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wdi_lastupdated": meta.get("lastupdated"),
        })
    write_manifest(folder, entries)


def fetch_imf() -> None:
    """МВФ: каждый набор — в CSV формата SDMX-CSV 2.0 (COUNTRY, INDICATOR, ..., TIME_PERIOD,
    OBS_VALUE). Все страны, месячная частота. Фильтр по индикаторам — в параметрах запроса."""
    spec = SOURCES["imf"]
    folder = DATA_RAW / "imf"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"[imf] → {folder}")
    entries = []
    for name, (flow, filters) in spec["flows"].items():
        dest = folder / f"{name}.csv"
        params = {f"c[{k}]": v for k, v in filters.items()}
        url = requests.Request("GET", spec["api"].format(flow=flow), params=params).prepare().url
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  есть, пропускаю: {dest.name}")
            entries.append({"url": url, "file": dest.name, "bytes": dest.stat().st_size, "skipped": True})
            continue
        tmp = dest.with_suffix(".part")
        with requests.get(url, headers={**HEADERS, "Accept": spec["accept"]}, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
        tmp.rename(dest)
        n_rows = sum(1 for _ in open(dest)) - 1
        print(f"  скачано: {dest.name} ({n_rows} строк, {dest.stat().st_size / 1e6:.1f} МБ)")
        entries.append({
            "url": url, "file": dest.name, "bytes": dest.stat().st_size,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    write_manifest(folder, entries)


def fetch_acled() -> None:
    """ACLED — события насилия и протестов. Нужна регистрация на acleddata.com.

    Доступ через их API: логин (email) и пароль из личного кабинета. Положи их в файл
    `.env` в корне проекта (он в .gitignore):
        ACLED_EMAIL=...
        ACLED_PASSWORD=...
    Пока что скрипт только проверяет, что ключи на месте, и получает токен —
    выгрузка событий добавится, когда решим, какие страны и годы нужны.
    """
    import os

    env_file = DATA_RAW.parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    email, password = os.environ.get("ACLED_EMAIL"), os.environ.get("ACLED_PASSWORD")
    if not (email and password):
        raise RuntimeError(
            "нет ACLED_EMAIL / ACLED_PASSWORD. Создай файл .env в корне проекта "
            "с логином и паролем от acleddata.com (см. docstring fetch_acled)."
        )
    r = requests.post(
        "https://acleddata.com/oauth/token",
        data={"username": email, "password": password, "grant_type": "password", "client_id": "acled"},
        headers=HEADERS,
        timeout=60,
    )
    r.raise_for_status()
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError(f"ACLED не выдал токен: {r.text[:200]}")
    print("  ACLED: токен получен, доступ работает. Выгрузка событий — следующим шагом.")


LV_URL = "https://www.imf.org/-/media/Files/Publications/WP/2018/wp18206.ashx"
LV_FILE = "wp18206.pdf"
LV_SHA256 = "e2c008d0cd9e12b336441f4eb9a10d0787dcb702e466437686dc21a70216f1e9"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_lv() -> None:
    """Laeven–Valencia (2018), IMF WP/18/206 — PDF статьи с таблицей дат кризисов (`sources/lv.py`).
    Сайт МВФ часто отвечает скриптам 403: тогда файл нужно скачать браузером и положить
    по указанному пути; при повторном запуске проверяется его контрольная сумма."""
    folder = DATA_RAW / "lv"
    dest = folder / LV_FILE
    print(f"[lv] → {folder}")
    if not dest.exists():
        try:
            entry = download_file(LV_URL, dest)
            write_manifest(folder, [entry])
        except requests.RequestException as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            folder.mkdir(parents=True, exist_ok=True)
            raise RuntimeError(
                f"сайт МВФ не отдал файл скрипту ({code or type(e).__name__}). Скачай вручную в браузере:\n"
                f"    {LV_URL}\n"
                f"  (страница статьи: https://www.imf.org/en/Publications/WP/Issues/2018/09/14/Systemic-Banking-Crises-Revisited-46232)\n"
                f"  и положи файл сюда:\n    {dest}\n"
                f"  затем запусти `uv run python -m seldon download lv` ещё раз — проверю контрольную сумму."
            ) from e
    got = sha256_of(dest)
    if got != LV_SHA256:
        raise RuntimeError(
            f"контрольная сумма {dest} не совпала:\n    ожидалось {LV_SHA256}\n    получено  {got}\n"
            "  Это другой файл или другая редакция PDF: таблица дат в sources/lv.py разбирается по позициям\n"
            "  колонок и на другом файле может разъехаться. Проверь, что скачана статья WP/18/206.")
    print(f"  {LV_FILE}: sha256 совпадает ({got[:12]}…)")
    write_manifest(folder, [{"url": LV_URL, "file": LV_FILE, "bytes": dest.stat().st_size, "sha256": got, "skipped": False}])


VDEM_COMMIT = "f4dd26922e658442524dfd954bf14f7ebe622d5d"  # «v16 release», 17.03.2026
VDEM_URL = f"https://raw.githubusercontent.com/vdeminstitute/vdemdata/{VDEM_COMMIT}/data/vdem.RData"
VDEM_SHA256 = "39b412d39a061c18f20c98e4ad4d6355b05a0441df31be4ee9aec420dc3d95ea"


def fetch_vdem() -> None:
    """V-Dem v16 — файл данных официального R-пакета института (GitHub vdeminstitute/vdemdata),
    ссылка закреплена на коммите релиза. Сайт v-dem.net отдаёт архив только через веб-форму —
    её не трогаем. Если GitHub недоступен, файл можно положить руками; сумма проверяется."""
    folder = DATA_RAW / "vdem"
    dest = folder / "vdem.RData"
    print(f"[vdem] → {folder}")
    if not dest.exists():
        try:
            write_manifest(folder, [download_file(VDEM_URL, dest)])
        except requests.RequestException as e:
            folder.mkdir(parents=True, exist_ok=True)
            raise RuntimeError(
                f"GitHub не отдал файл ({type(e).__name__}). Скачай вручную в браузере:\n    {VDEM_URL}\n"
                f"  (репозиторий: https://github.com/vdeminstitute/vdemdata, релиз V16, файл data/vdem.RData)\n"
                f"  и положи сюда:\n    {dest}\n  затем запусти `uv run python -m seldon download vdem` ещё раз."
            ) from e
    got = sha256_of(dest)
    if got != VDEM_SHA256:
        raise RuntimeError(f"контрольная сумма {dest} не совпала:\n    ожидалось {VDEM_SHA256}\n    получено  {got}\n"
                           "  Это другой релиз V-Dem: проверь версию (нужен v16) или обнови VDEM_COMMIT и VDEM_SHA256.")
    print(f"  vdem.RData: sha256 совпадает ({got[:12]}…)")
    write_manifest(folder, [{"url": VDEM_URL, "file": dest.name, "bytes": dest.stat().st_size, "sha256": got, "skipped": False}])


FETCHERS = {
    "lv": fetch_lv,
    "vdem": fetch_vdem,
    "epr": lambda: fetch_static("epr"),
    "ucdp": lambda: fetch_static("ucdp"),
    "pitf": lambda: fetch_static("pitf"),
    "conflictforecast": fetch_conflictforecast,
    "gw": lambda: fetch_static("gw"),
    "cow": lambda: fetch_static("cow"),
    "polity": lambda: fetch_static("polity"),
    "wdi": fetch_wdi,
    "jst": lambda: fetch_static("jst"),
    "boe": lambda: fetch_static("boe"),
    "gmd": lambda: fetch_static("gmd"),
    "imf": fetch_imf,
    "bis": lambda: fetch_static("bis"),
    "reign": lambda: fetch_static("reign"),
    "massmob": lambda: fetch_static("massmob"),
    "lenta": lambda: fetch_static("lenta"),
    "cbr": lambda: fetch_static("cbr"),
    "acled": fetch_acled,  # не входит в `all`: требует логин/пароль в .env
}
DEFAULT_SOURCES = ["ucdp", "pitf", "conflictforecast", "gw", "cow", "polity", "wdi", "jst", "boe", "gmd", "imf", "bis", "reign", "massmob", "lenta", "cbr", "lv", "vdem", "epr"]


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in (*FETCHERS, "all"):
        print(__doc__)
        return 2
    names = DEFAULT_SOURCES if argv[0] == "all" else [argv[0]]
    failed = []
    for n in names:
        try:
            FETCHERS[n]()
        except Exception as e:  # noqa: BLE001 — хотим дойти до конца списка
            print(f"  ОШИБКА [{n}]: {e}")
            failed.append(n)
    if failed:
        print(f"\nНе скачалось: {', '.join(failed)}")
        return 1
    print("\nГотово.")
    return 0

