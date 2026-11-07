"""Скачивание сырых данных первого этапа.

Запуск:
    uv run python -m seldon.download ucdp
    uv run python -m seldon.download pitf
    uv run python -m seldon.download conflictforecast
    uv run python -m seldon.download gw
    uv run python -m seldon.download cow
    uv run python -m seldon.download all

Файлы кладутся в data/raw/<источник>/. Уже скачанное не качается повторно
(удали файл, чтобы перекачать). Рядом с файлами пишется manifest.json:
откуда, когда и какого размера — чтобы потом можно было понять, какая версия данных
лежала в момент эксперимента.
"""

from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

from seldon import DATA_RAW

# ---------------------------------------------------------------------------
# Источники
# ---------------------------------------------------------------------------

UCDP_VERSION = "25.1"
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


FETCHERS = {
    "ucdp": lambda: fetch_static("ucdp"),
    "pitf": lambda: fetch_static("pitf"),
    "conflictforecast": fetch_conflictforecast,
    "gw": lambda: fetch_static("gw"),
    "cow": lambda: fetch_static("cow"),
    "acled": fetch_acled,  # не входит в `all`: требует логин/пароль в .env
}
DEFAULT_SOURCES = ["ucdp", "pitf", "conflictforecast", "gw", "cow"]


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


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
