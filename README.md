# Селдон

Модель, которая прогнозирует войны и экономические кризисы. Глобальная, с углублением
в постсоветское пространство. Название — в честь Гэри Селдона из «Основания» Азимова,
но цель приземлённая: работающий прогноз, проверенный «вслепую» (`out-of-sample`).

Ресёрч по существующим системам и данным: https://claude.ai/artifact/NUKDLvVMxefGmZnL8GnzQy

## Принципы

1. Сначала самое простое: базовая модель → повторение чужих → только потом своё.
2. Честная проверка: модель учится на прошлом и предсказывает период, которого не видела.
3. Прогноз — вероятность, а не «да/нет».
4. Всё воспроизводимо: данные качаются скриптом, в git лежит только код.

## Как запустить

Нужен [`uv`](https://docs.astral.sh/uv/). Дальше:

```bash
uv sync
```

Скачать все данные (UCDP, PITF, conflictforecast, списки государств и соседей, Polity, ВБ, JST, код BoE, GMD) в `data/raw/`:

```bash
uv run python -m seldon download all
```

Открыть ноутбуки:

```bash
uv run jupyter lab
```

## Структура

```
data/raw/        сырые скачанные файлы (не в git)
data/processed/  обработанные таблицы (не в git)
notebooks/       ноутбуки разведки и экспериментов
src/seldon/      код: загрузчики данных, признаки, модели
reports/         отчёты по сессиям
```

## Данные

| Источник | Что внутри | Зачем |
| --- | --- | --- |
| [UCDP](https://ucdp.uu.se) (Уппсала), версия 25.1 | события с погибшими (GED), конфликты по годам (UCDP/PRIO ACD), погибшие в боях (BRD) | целевая переменная по войнам |
| [PITF](https://scip.gmu.edu/a-global-model-for-forecasting-political-instability/) (Goldstone и др., 2010) | replication data: таблица стран-лет и Stata-скрипт | повторение модели на 4 переменных |
| [conflictforecast.org](https://conflictforecast.org) | ежемесячные прогнозы по странам и ячейкам | эталон для сравнения |
| [Gleditsch & Ward](http://ksgleditsch.com/data-4.html) | список независимых государств с датами | «тихие» страны в панели |
| [COW Direct Contiguity 3.2](https://correlatesofwar.org/data-sets/direct-contiguity/) | кто с кем граничит, по годам | целевая «сторона конфликта у соседа», признак «война у соседей» |
| [Polity5](https://www.systemicpeace.org/inscrdata.html) | тип политического режима, 1800–2018 | признаки PITF (режим, фракционность) |
| [Всемирный банк, WDI/GFD](https://data.worldbank.org) (открытый API) | детская смертность, население, кредит частному сектору (% ВВП), резервы | признаки PITF; кредит для стран вне JST |
| [JST Macrohistory Database](https://www.macrohistory.net/database/), R6 | 18 развитых стран с 1870: кредит, ВВП, ставки, цены, кризисы | данные модели Банка Англии |
| [Bank of England SWP 848](https://github.com/bank-of-england/MachineLearningCrisisPrediction) | код статьи Bluwstein и др. + JST R3 | повторение модели экономических кризисов |
| [Global Macro Database](https://www.globalmacrodata.com), 2026_09 | 46 рядов по 239 странам, вкл. постсоветские; дамми кризисов | панель для постсоветских стран |
