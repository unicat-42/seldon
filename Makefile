# Селдон — пересчёт всего с нуля одной командой.
#
#   make all        боевой конвейер в правильном порядке (≈ 8 минут)
#   make variants   срезы для сравнения: старые целевые, бустинг, all_currencies, другие хроники,
#                   структурные наборы войн (V-Dem, EPR) против эталона `pitf` (`compare_war`),
#                   диадная панель и два канала риска (`dyads`, `model_dyad`, `compare_war dyad`),
#                   прогноз «на 1 апреля» (`apr1` — макроданные без лага публикации),
#                   структурный валютный блок против боевых валютных моделей (`econ_struct`,
#                   `compare_fx`; `ctot_rt` — условия торговли без задержки набора МВФ),
#                   валютное заражение по торговым связям и параллельный курс на горизонтах
#                   3 / 6 / 12 месяцев (`econ_contagion`, `compare_contagion`; ≈ 6 минут),
#                   месячная модель войн на 3 / 6 / 12 месяцев с предвестниками и Candidate
#                   Events (`month_features`, `model_month`, `compare_month`), доводка диад
#                   (`dyads_live`, `compare_war live`; вместе ≈ 25 минут)
#   make replicate  повторение статьи Банка Англии на JST (≈ 10 минут, 1000 деревьев)
#   make risk       пересобрать только таблицу риска (нужны готовые таблицы `make all`)
#   make test       автотесты на утечки будущего (секунды)
#   make vintage    заморозить прогнозы в forecasts/ГГГГ-ММ/ (после `make all`; раз в месяц)
#   make data       скачать сырые данные (Laeven–Valencia — PDF руками, см. подсказку команды)
#
# Все команды — `uv run python -m seldon <команда>`; список: `uv run python -m seldon`.
# Исследовательские блоки (`research/`) сюда не входят — они запускаются по одному.

RUN = uv run python -m seldon

.PHONY: all variants replicate risk test data vintage

# Порядок важен: панель войн → признаки → база → модель войн → хроника LV → месячный
# валютный блок (ему нужна месячная панель войн) → годовой валютный (нужна LV) →
# банковский → таблица риска → сводка качества.
all:
	$(RUN) panel
	$(RUN) features
	$(RUN) features_pitf
	$(RUN) baseline state_war25
	$(RUN) baseline state_war1000
	$(RUN) baseline nonstate25
	$(RUN) model_hist hist logit state_war25
	$(RUN) model_hist pitf logit state_war25
	$(RUN) model_hist hist logit state_war1000
	$(RUN) model_hist pitf logit state_war1000
	$(RUN) lv
	$(RUN) econ_kr crisis
	$(RUN) econ_kr deval15
	$(RUN) econ_long deval25
	$(RUN) econ_boe_em
	$(RUN) risk_table
	$(RUN) scorecard

# Срезы для сравнения (нужен готовый `make all`: признаки экономики берут EMP из econ_kr).
variants:
	$(RUN) features war25
	$(RUN) baseline war25
	$(RUN) baseline war1000
	$(RUN) model_hist hist logit war25
	$(RUN) model_hist hist logit war1000
	$(RUN) model_hist pitf gbm state_war25
	$(RUN) features_econ
	$(RUN) features_polit
	$(RUN) model_hist econ logit state_war25
	$(RUN) model_hist polit logit state_war25
	$(RUN) features_vdem
	$(RUN) features_epr
	$(RUN) model_hist vdem logit state_war25
	$(RUN) model_hist epr logit state_war25
	$(RUN) model_hist vdem_epr logit state_war25
	$(RUN) compare_war
	$(RUN) dyads
	$(RUN) model_dyad
	$(RUN) model_hist pitf_dyad logit state_war25
	$(RUN) compare_war dyad
	$(RUN) month_features
	$(RUN) model_month
	$(RUN) model_month cf
	$(RUN) dyads_live
	$(RUN) model_hist pitf_smoke logit state_war25
	$(RUN) model_hist pitf_live logit state_war25
	$(RUN) compare_war live
	$(RUN) compare_month
	$(RUN) econ_kr crisis all_currencies
	$(RUN) econ_long deval25 all_currencies
	$(RUN) econ_long gmd_cc
	$(RUN) econ_long lv_cc
	$(RUN) econ_boe_em lv
	$(RUN) econ_long deval25 apr1
	$(RUN) econ_boe_em apr1
	$(RUN) econ_struct
	$(RUN) compare_fx
	$(RUN) econ_struct ctot_rt
	$(RUN) compare_fx ctot_rt
	$(RUN) econ_contagion
	$(RUN) compare_contagion

replicate:
	$(RUN) econ_boe

risk:
	$(RUN) risk_table

test:
	uv run pytest

data:
	$(RUN) download all

vintage:
	$(RUN) vintage
