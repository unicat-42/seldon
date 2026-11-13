**Русская версия** · [English](README.md)

# Что лежит в `reports/`

Здесь два вида файлов: отчёты по сессиям (`ГГГГ-ММ-ДД-*.md`) и таблицы (`*.csv`).
Таблицы копились с первого дня, и не все они актуальны. Ниже — что чем является.

Правило чтения: цифрам из файлов раздела «Архив до аудита» верить нельзя — в них утечки
будущего. Актуальное качество моделей — `scorecard.csv`.

## Актуальное: пересчитывается `make all`

Постановка — после аудита, с лагами публикации, прогноз «на 1 января».

| Файлы | Что это | Команда |
| --- | --- | --- |
| `scorecard.csv` | сводка качества основных моделей с 95 % интервалами | `scorecard` |
| `risk_2025_26.csv` | таблица риска по странам (её выжимка — `forecasts/ГГГГ-ММ/predictions.csv`). Кроме вероятностей — описательные индикаторы, это НЕ прогнозы: текущий стресс (`emp_z_max3`, `reserves_g12`, `deval12`, `infl12`, `signals_fired`) и уязвимость режима курса (`peg_share12` — доля спокойных месяцев из 12, `reer_peg` — реальное укрепление сверх 5-летнего среднего × привязка, `peg_resfall` — привязка × падение резервов за 12 месяцев; по живым рядам, без IRR) | `risk_table` |
| `baseline_*_state_war25.csv`, `baseline_*_state_war1000.csv`, `baseline_*_nonstate25.csv` | базовые модели войн | `baseline` |
| `hist_state_war25_*`, `hist_state_war1000_*`, `pitf_state_war25_*`, `pitf_state_war1000_*` | модель войн: метрики по годам, сводка с интервалами, коэффициенты | `model_hist` |
| `econ_kr_*.csv`, `econ_kr_*_deval15.csv` | месячный валютный блок: покрытие, кризисы, метрики, пороги сигналов, таблица «за 12 / 6 / 3 месяца до кризиса» | `econ_kr` |
| `econ_long_*.csv` (без суффикса) | годовой валютный блок, целевая `deval25` | `econ_long` |
| `boe_em_*.csv` (без суффикса) | банковская модель на странах BIS, все три набора признаков | `econ_boe_em` |

## Актуальное: срезы для сравнения, `make variants`

| Файлы | Что это |
| --- | --- |
| `*_apr1.csv` | прогноз «на 1 апреля»: макроданные без лага публикации (проверка чувствительности) |
| `*_allcur.csv` | без исключения стран с общей валютой |
| `econ_long_*_gmd_cc.csv`, `econ_long_*_lv_cc.csv` | другие хроники валютных кризисов как целевая |
| `boe_em_*_lv.csv` | банковская модель с меткой Laeven–Valencia |
| `baseline_*_war25.csv`, `baseline_*_war1000.csv`, `hist_war25_*`, `hist_war1000_*` | старые целевые войн (до аудита), посчитанные нынешним кодом, — для преемственности |
| `pitf_gbm_state_war25_*`, `econ_state_war25_*`, `polit_state_war25_*` | модель войн: бустинг, + экономика, + политика |
| `vdem_state_war25_*`, `epr_state_war25_*`, `vdem_epr_state_war25_*` | модель войн со структурными признаками: V-Dem вместо Polity5, EPR, оба вместе (чекпоинт 3.1) |
| `war_struct_comparison.csv` | парное сравнение этих наборов с эталоном `pitf`: мир, тихие ≥ 10 лет, постсоветские (`compare_war`) |
| `dyad_metrics_summary.csv`, `dyad_coefficients.csv` | диадная модель (пары стран, чекпоинт 3.2): качество вслепую против базы «память пары + граница», коэффициенты по годам (`model_dyad`) |
| `dyad_cases.csv`, `dyad_top_2026.csv` | место известных пар перед началом войны; 15 самых опасных пар на 2026 год |
| `pitf_dyad_state_war25_*` | модель войн со свёрнутыми диадными признаками (контроль) |
| `war_dyad_comparison.csv` | два канала риска (`dy_indep`, `dy_intext`, `dy_cal2`) и `pitf_dyad` против `pitf`: мир, тихие ≥ 10 лет, постсоветские, внешние и внутренние начала (`compare_war dyad`) |
| `war_month_comparison.csv`, `war_month_single_auc.csv`, `war_month_coefficients.csv` | месячная модель войн (чекпоинт 4.2): начало `state_war25` в ближайшие 3 / 6 / 12 месяцев, наборы предвестников против эталонов `pitf` и `dry`; мир, тихие ≥ 5 и ≥ 10 лет, постсоветские, после 2014 (`model_month`, `compare_month`) |
| `war_month_cases.csv`, `war_month_top_2026-09.csv` | место стран за 3 / 6 / 12 месяцев до известных начал; прогноз из последнего месяца Candidate Events (не витрина) |
| `war_month_cf_comparison.csv` | сравнение с conflictforecast.org на их целевой и на нашей, горизонты 3 и 12 месяцев |
| `dyad_live_metrics.csv`, `dyad_live_coefficients.csv`, `dyad_live_cases.csv`, `dyad_live_ext_channel.csv`, `dyad_live_top_2026.csv` | доводка диад (чекпоинт 4.2): живые стычки вместо MID, раздельные модели соседей и держав, калибровка `p_ext` (`dyads_live`) |
| `war_live_comparison.csv`, `pitf_smoke_state_war25_*`, `pitf_live_state_war25_*` | годовой срез: доводка диад и годовые предвестники против `pitf` (`compare_war live`) |
| `fx_struct_comparison.csv`, `fx_struct_comparison_ctot_rt.csv` | структурный валютный блок (чекпоинт 3.3): режим курса, резервы, REER, условия торговли против боевых валютных моделей — годовой и месячный контуры; мир, постсоветские, тихие ≥ 10 лет, привязки, две половины периода (`compare_fx`). `_ctot_rt` — условия торговли с лагом 1 месяц вместо 6 |
| `fx_struct_vs_base.csv` | то же для логрегрессий, но против `base` — боевого набора с выбранной регуляризацией: чистый вклад признаков |
| `fx_struct_cases_long.csv`, `fx_struct_cases_kr.csv` | место страны по риску перед известными девальвациями: эталон против наборов |
| `fx_contagion_comparison.csv` | чекпоинт 4.1: заражение по торговым связям, параллельный курс и чистка `gg_missing` против боевой месячной модели, её же с выбранной регуляризацией и набора `struct` — горизонты 3 / 6 / 12 месяцев; мир, постсоветские, привязки, две половины периода, окна покрытия параллельного курса (`compare_contagion`) |
| `fx_contagion_cases.csv` | место страны по риску за 3 / 6 / 12 месяцев до контрольных скачков курса |
| `fx_contagion_coefficients.csv`, `fx_contagion_univariate.csv`, `fx_contagion_coverage.csv`, `fx_contagion_gg_missing.csv` | коэффициенты наборов; AUC каждого нового признака в одиночку; покрытие по постсоветским странам; кто попадает под флаг `gg_missing` (`econ_contagion`) |
| `fx_struct_coverage.csv`, `fx_struct_C.csv`, `fx_struct_coefficients.csv` | покрытие структурных признаков по постсоветским странам; какая сила регуляризации выбиралась; коэффициенты набора `struct` (`econ_struct`) |
| `war_dyad_comparison.csv`, `war_month_comparison.csv`, `war_month_cases.csv`, `war_month_calibration.csv`, `war_month_univariate.csv`, `war_month_coef_2025.csv`, `war_month_ext_diag.csv` | чекпоинт 4.2: двусторонний диадный и месячный контуры войн, кейсы, калибровка и диагностика редких/архивных признаков (`compare_war`, `compare_month`) |
| `war_network_annual_comparison.csv`, `war_network_month_comparison.csv`, `war_network_univariate.csv`, `war_network_cases.csv`, `pitf_net_cl_state_war25_coefficients.csv`, `pitf_polit_net_state_war25_coefficients.csv` | чекпоинт 4.3a: сетевое заражение войн в годовом и месячном контурах, одиночные признаки, кейсы и коэффициенты артефакта пропуска архивов (`compare_network`, `model_hist`) |
| `war_csd_comparison.csv`, `war_csd_univariate.csv`, `fx_csd_comparison.csv`, `fx_csd_univariate.csv` | чекпоинт 4.3b: критическое замедление (`CSD`) в месячных войнах и валютных кризисах против эталонов (`csd_month`, `econ_csd`) |
| `fx_showcase_regime_slices.csv`, `fx_crawling_cases.csv`, `fx_annual_live_comparison.csv`, `stage4_showcase_candidates.csv`, `stage4_showcase_today_fx.csv`, `stage4_showcase_today_war.csv` | чекпоинт 4.3c: отбор кандидатов по режимам курса, кейсы ползущих привязок, годовой живой блок, сводка 4 контуров и сухая проверка живого края `2026-10` / `2026` (`select_showcase`) |

## Актуальный код, но не пересчитывалось после этапа 1

| Файлы | Что это | Команда |
| --- | --- | --- |
| `boe_cv_results.csv`, `boe_forecast_*.csv`, `boe_coefficients.csv` | повторение статьи Банка Англии на данных JST (сверка с опубликованными цифрами) | `make replicate` |

## Исследовательские блоки (`research/`)

Выводы от утечек не зависят, но индекс давления на валютный рынок (EMP) в них посчитан
по-старому («всезнающий» — параметры по всей выборке). После аудита не пересчитывались.

| Файлы | Что это |
| --- | --- |
| `pitf_odds_ratios.csv` | повторение модели PITF на её данных |
| `cf_comparison_*.csv` | сравнение с conflictforecast.org |
| `econ_metrics.csv`, `econ_pre_crisis_table.csv`, `econ_post_soviet_coverage.csv`, `econ_crises_post_soviet.csv` | постсоветская панель GMD + ВБ (`econ_panel`) |
| `news_ru_*.csv` | газеты: Lenta.ru против давления на рубль |
| `behavior_ru_*.csv`, `behavior_world_*.csv` | вклады, покупка валюты, GDELT |

## Архив до аудита

Посчитаны 7–10 ноября 2026 года старым кодом, со старыми определениями войны (`war25`,
`war1000`, `conflict25`) и без защиты от утечек. Оставлены, чтобы отчёты тех дней можно
было сверить. Узнать их просто: в имени НЕТ названия целевой после набора признаков.

- `baseline_metrics_by_year.csv`, `baseline_metrics_summary.csv`;
- `hist_*`, `hist_gbm_*`, `histecon_*`, `pitf_*`, `pitf_gbm_*`, `econ_*`, `econ_gbm_*`,
  `polit_*`, `polit_gbm_*` с окончаниями `_metrics_by_year.csv`, `_metrics_summary.csv`,
  `_coefficients.csv` — сразу или после `_war1000` (например, `pitf_metrics_summary.csv`,
  `polit_gbm_war1000_coefficients.csv`).

Актуальные аналоги содержат целевую в имени: `pitf_state_war25_metrics_summary.csv`.

## Отчёты по сессиям

- `2026-11-07-*`, `2026-11-10-*` — до аудита; метрики валютного, банковского и связочного
  блоков в них завышены.
- `2026-11-12-brief-audit.md` — архив: входной бриф для внешнего аудита (состояние проекта до аудита).
- `2026-11-12-session-audit-stage1.md` — аудит, «было → стало».
- `2026-11-12-session-stage2-layers.md`, `-lags.md`, `-vintage.md` — Этап 2: слои, лаги
  публикации, первый винтаж прогнозов.
- `2026-11-12-session-stage3-fx-struct.md`, `2026-11-13-session-stage3-war.md`,
  `2026-11-13-session-stage3-fx-live-edge.md` — Этап 3: структурные признаки, живые источники,
  свежесть данных на живом крае.
- `2026-11-13-session-stage4-1.md`, `2026-11-13-session-stage4-2.md`,
  `2026-11-13-session-stage4-3.md` — Этап 4: валютное заражение и параллельный курс (`4.1`),
  диады и месячные войны (`4.2`), сетевое заражение, `CSD` и отбор кандидатов в витрину (`4.3`).
