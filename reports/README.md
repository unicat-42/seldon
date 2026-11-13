**English** · [Русская версия](README.ru.md)

# What Lives in `reports/`

This directory holds two kinds of artifacts: session research logs (`YYYY-MM-DD-*.md`, written in
Russian) and generated CSV tables (`*.csv`). Because tables accumulated from day one of the project,
the index below separates current leak-free outputs from pre-audit historical artifacts.

> **Important:** Do **not** cite metrics from files listed under *Pre-Audit Archive* — those runs
> predated the November 12, 2026 look-ahead audit. Authoritative post-audit benchmarks live in
> `scorecard.csv`.

## Current Production Outputs (`make all`)

Post-audit expanding-window setup with publication lags enforced ("as of January 1" for annual models).

| Files | Description | CLI Command |
| --- | --- | --- |
| `scorecard.csv` | Master out-of-sample evaluation table across all core models with 95% block-bootstrap CIs | `scorecard` |
| `risk_2025_26.csv` | Master country risk table (frozen snapshots in `forecasts/YYYY-MM/predictions.csv` are distilled from it). Alongside probabilities, includes **descriptive stress & peg-vulnerability indicators** (not probabilities): `emp_z_max3`, `reserves_g12`, `deval12`, `infl12`, `signals_fired`, `peg_share12`, `reer_peg`, `peg_resfall` | `risk_table` |
| `baseline_*_state_war25.csv`, `baseline_*_state_war1000.csv`, `baseline_*_nonstate25.csv` | Conflict-history decay baselines | `baseline` |
| `hist_state_war25_*`, `hist_state_war1000_*`, `pitf_state_war25_*`, `pitf_state_war1000_*` | Armed conflict onset/continuation models: yearly metrics, summary with CIs, coefficients | `model_hist` |
| `econ_kr_*.csv`, `econ_kr_*_deval15.csv`, `econ_csd_live_*.csv` | Monthly currency block (`crisis` EMP and `deval15` jumps) plus live challenger `live_fx_swap`: coverage, crises, metrics, KLR signal thresholds, pre-crisis rankings at 12 / 6 / 3 months | `econ_kr`, `econ_csd live` |
| `econ_long_*.csv` (unsuffixed) | Annual currency crisis block (`deval25` target) | `econ_long` |
| `boe_em_*.csv` (unsuffixed) | Systemic banking crisis model on BIS countries across all three feature sets (`full`, `noyield`, `nomoney`) | `econ_boe_em` |

## Current Sensitivity Slices & Stage 3–4 Candidates (`make variants`)

| Files | Description |
| --- | --- |
| `*_apr1.csv` | "As of April 1" sensitivity runs (annual macro fundamentals without publication lag) |
| `*_allcur.csv` | Without excluding currency-union / dollarized states |
| `econ_long_*_gmd_cc.csv`, `econ_long_*_lv_cc.csv` | Alternative currency crisis chronologies (GMD, Laeven–Valencia) as target |
| `boe_em_*_lv.csv` | Banking crisis model evaluated on Laeven–Valencia crisis dates |
| `baseline_*_war25.csv`, `baseline_*_war1000.csv`, `hist_war25_*`, `hist_war1000_*` | Legacy conflict target definitions recomputed with leak-free code for continuity |
| `pitf_gbm_state_war25_*`, `econ_state_war25_*`, `polit_state_war25_*` | Armed conflict variants: gradient boosting, + macro features, + political events |
| `vdem_state_war25_*`, `epr_state_war25_*`, `vdem_epr_state_war25_*`, `war_struct_comparison.csv` | Stage 3.1 structural conflict features (V-Dem v16, ETH EPR 2023) paired vs `pitf` (`compare_war`) |
| `dyad_metrics_summary.csv`, `dyad_coefficients.csv`, `dyad_cases.csv`, `dyad_top_2026.csv`, `pitf_dyad_state_war25_*`, `war_dyad_comparison.csv` | Stage 3.2 directed country-dyad conflict model & two-channel country risk (`model_dyad`, `compare_war dyad`) |
| `fx_struct_comparison*.csv`, `fx_struct_vs_base.csv`, `fx_struct_cases_*.csv`, `fx_struct_coverage.csv`, `fx_struct_C.csv`, `fx_struct_coefficients.csv` | Stage 3.3 structural currency features (IRR regimes, Greenspan–Guidotti, Bruegel REER, IMF commodity terms of trade) (`econ_struct`, `compare_fx`) |
| `fx_contagion_*.csv` | Stage 4.1 bilateral trade FX contagion, parallel-rate coverage, `gg_missing` audit, and `peg_resfall` (`econ_contagion`, `compare_contagion`) |
| `war_month_*.csv`, `dyad_live_*.csv`, `war_live_comparison.csv`, `pitf_smoke_state_war25_*`, `pitf_live_state_war25_*` | Stage 4.2 monthly conflict escalation (3 / 6 / 12m), live UCDP sub-war clashes, `conflictforecast.org` benchmark (`model_month`, `compare_month`, `dyads_live`) |
| `war_network_*.csv`, `pitf_net_*`, `vdem_net_*`, `pitf_polit_net_*` | Stage 4.3a conflict network contagion across linguistic, regime, ethnic-kin & geographic graphs (`compare_network`) |
| `war_csd_*.csv`, `fx_csd_*.csv` | Stage 4.3b critical slowing down (`CSD`) variance & `AR(1)` tests (`war_csd`, `econ_csd`) |
| `fx_showcase_regime_slices.csv`, `fx_crawling_cases.csv`, `fx_annual_live_comparison.csv`, `stage4_showcase_*.csv` | Stage 4.3c candidate selection across FX regimes & live-edge parity checks (`select_showcase`) |

## Current Code, Not Re-Run Since Stage 1

| Files | Description | Command |
| --- | --- | --- |
| `boe_cv_results.csv`, `boe_forecast_*.csv`, `boe_coefficients.csv` | Replication of Bluwstein et al. (Bank of England) on JST R6 (1,000 trees) | `make replicate` |

## Standalone Research Studies (`research/`)

Qualitative conclusions are unaffected by the audit, though legacy scripts (`econ_panel`, `news_ru`,
`behavior_*`) used full-sample EMP scaling when originally run.

| Files | Description |
| --- | --- |
| `pitf_odds_ratios.csv` | Replication of Goldstone et al. (2010) PITF odds ratios |
| `cf_comparison_*.csv` | Benchmark comparison against `conflictforecast.org` |
| `econ_metrics.csv`, `econ_pre_crisis_table.csv`, `econ_post_soviet_coverage.csv`, `econ_crises_post_soviet.csv` | Exploratory post-Soviet GMD + WDI currency panel (`econ_panel`) |
| `news_ru_*.csv` | Russian news corpus (`Lenta.ru`) vs ruble exchange-market pressure |
| `behavior_ru_*.csv`, `behavior_world_*.csv` | Household deposit runs, FX purchases & GDELT protest series |
| `party_*.csv` | Ruling-party ideology & turnover study (`party_study`, `make party`) |

## Pre-Audit Archive (Do Not Cite)

Generated Nov 7–10, 2026 prior to the anti-leakage audit with legacy conflict definitions (`war25`,
`war1000`, `conflict25`). Retained solely for audit traceability. Easy to spot: filenames omit the
target name between feature set and suffix (e.g., `pitf_metrics_summary.csv` instead of
`pitf_state_war25_metrics_summary.csv`):

- `baseline_metrics_by_year.csv`, `baseline_metrics_summary.csv`;
- `hist_*`, `hist_gbm_*`, `histecon_*`, `pitf_*`, `pitf_gbm_*`, `econ_*`, `econ_gbm_*`, `polit_*`,
  `polit_gbm_*` ending directly in `_metrics_by_year.csv`, `_metrics_summary.csv`, or
  `_coefficients.csv` (or immediately after `_war1000`).

## Session Reports (Russian)

- `2026-11-07-*`, `2026-11-10-*` — pre-audit exploratory sessions (FX/banking metrics inflated by leakage).
- `2026-11-12-brief-audit.md` — archive: input brief prepared for the external audit.
- `2026-11-12-session-audit-stage1.md` — Stage 1 audit report (before vs. after leak removal).
- `2026-11-12-session-stage2-layers.md`, `-lags.md`, `-vintage.md` — Stage 2 architecture, publication lags, first vintage.
- `2026-11-12-session-stage3-1.md`, `-2.md`, `-3.md` — Stage 3 structural conflict, dyads, and FX blocks.
- `2026-11-13-session-stage4-1.md`, `-2.md`, `-3.md` — Stage 4 FX contagion, monthly conflict escalation, network contagion & CSD.
- `2026-11-13-session-stage5-1.md`, `-2.md`, `-3.md` — Stage 5 static web showcase export, UI & verification views.
