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
#                   (`dyads_live`, `compare_war live`), сетевое заражение конфликтов
#                   (`network_features`, `compare_network`), критическое замедление (`econ_csd`,
#                   `war_csd`) и сводный отбор кандидатов в витрину (`select_showcase`; ≈ 30 минут)
#   make replicate  повторение статьи Банка Англии на JST (≈ 10 минут, 1000 деревьев)
#   make risk       пересобрать только таблицу риска (нужны готовые таблицы `make all`)
#   make test       автотесты на утечки будущего (секунды)
#   make vintage    заморозить прогнозы в forecasts/ГГГГ-ММ/ (после `make all`; раз в месяц)
#   make web        данные для веб-витрины: JSON в web/data/ (после `make all`; ≈ 30 секунд)
#   make serve      открыть веб-витрину локально: http://localhost:8791 (после `make web`)
#   make data       скачать сырые данные (Laeven–Valencia — PDF руками, см. подсказку команды)
#
# Все команды — `uv run python -m seldon <команда>`; список: `uv run python -m seldon`.
# Исследовательские блоки (`research/`) сюда не входят — они запускаются по одному.

RUN = uv run python -m seldon

.PHONY: all variants replicate risk test data vintage web serve

# Порядок важен: панель войн → признаки → база → модель войн → хроника LV → месячный
# валютный блок (ему нужна месячная панель войн) → годовой валютный (нужна LV) →
# банковский → таблица риска → сводка качества → данные веб-витрины (`web/data/` лежит в git
# и обязан соответствовать таблицам: это сторожит `tests/test_web_export.py`).
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
	$(RUN) econ_csd live
	$(RUN) econ_long deval25
	$(RUN) econ_boe_em
	$(RUN) risk_table
	$(RUN) scorecard
	$(RUN) web_export

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
	$(RUN) network_features
	$(RUN) model_hist pitf_net_el logit state_war25
	$(RUN) model_hist pitf_net_cl logit state_war25
	$(RUN) model_hist vdem_net_cl logit state_war25
	$(RUN) model_hist pitf_polit_net logit state_war25
	$(RUN) compare_network
	$(RUN) econ_csd
	$(RUN) war_csd
	$(RUN) select_showcase

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

# Данные для веб-витрины (Этап 5): читает готовые таблицы, сверяет пересчёт с витриной.
web:
	$(RUN) web_export

# Веб-витрина — статические файлы в web/, сервер нужен только чтобы браузер отдал JSON.
serve:
	python3 -m http.server 8791 --bind 127.0.0.1 --directory web
