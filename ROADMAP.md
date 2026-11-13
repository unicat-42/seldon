**English** · [Русская версия](ROADMAP.ru.md)

# Seldon — Post-Audit Roadmap

An external audit on 2026-11-12 uncovered look-ahead leakage across early prototype blocks: at
forecast time, models had access to observations or sample statistics not yet available. This
roadmap tracks five post-audit stages. Pre-audit session history is preserved in `CLAUDE.md` and
`reports/`.

**Ground rule across all stages:** write a failing anti-leakage test first, then fix the code. All
metrics in reports carry an explicit "pre-audit" or "post-audit" label; pre-audit numbers are never
cited as findings.

## Stage 1. Leak Audit & Honest Baselines — Completed 2026-11-12

Report: `reports/2026-11-12-session-audit-stage1.md`.

- Architectural invariants codified in `CLAUDE.md`.
- `tests/test_no_leakage.py` (`uv run pytest`): future blindness across every forecasting pipeline,
  target–`evaluable` alignment, unified country codes, and currency-union exclusions.
- Fixes shipped:
  1. **Annual currency block (`econ_long.py`):** switched from annual-average GMD exchange rates to
     end-of-December IMF rates; currency-union members excluded starting from the year of adoption.
  2. **Banking crisis block (`econ_boe_em.py`, `econ_boe.py`):** JST training moved inside the
     annual expanding-window loop; out-of-JST BIS evaluation separated; crisis chronologies
     truncated at their actual end year rather than zero-filled.
  3. **Armed conflict (`panel.py`, `features*.py`, `entities.py`):** separated `state_war25` /
     `state_war1000` (state-based conflict at home or in a neighbor where the state is a belligerent)
     from `nonstate25`; enforced 12 quiet months before monthly onset; eliminated fatality
     double-counting; unified country identifiers.
  4. **Monthly currency block (`econ_kr.py`):** Exchange Market Pressure (EMP) parameters estimated
     strictly on expanding windows; CPI future multiplier removed; post-crisis quiet window
     synchronized with `evaluable`.
  5. **Conflict model (`model_hist.py`):** separate onset vs. continuation fits, year-blocked
     bootstrap CIs, pooled AUC reported alongside within-year AUC.
- All core tables and `reports/risk_2025_26.csv` recomputed from scratch.

Deferred to Stage 2:
- explicit publication lags (annual GMD/WDI treated as known at year-end, UCDP ACD belligerents
  within the same calendar year);
- frozen standalone research scripts (`econ_panel.py`, `behavior_*.py`, `news_ru.py`, `compare_cf.py`)
  left untouched as historical archive.

## Stage 2. Clean Architecture, Publication Lags & First Forecast Vintage — Completed 2026-11-12

Goal: make look-ahead leaks structurally hard to write and start a verifiable prospective forecast
record.

### Checkpoint 2.1 — Five-Layer Architecture (No Logic Changes) — Completed 2026-11-12

Report: `reports/2026-11-12-session-stage2-layers.md`.

Refactored flat modules in `src/seldon/` into five clean layers plus `research/`:

- `core/` — `entities.py` (country codes & currency unions), `panel.py` (country×period grids &
  calendar lags), `temporal.py` (`as_of` contract, `available_from` lags, `expanding_origin_split`),
  `calibration.py` (KLR binning, isotonic), `metrics.py` (pooled & within-year AUC, Brier, paired
  year-blocked bootstrap).
- `sources/` — deterministic downloaders and parsers (`ucdp`, `imf`, `gmd`, `wdi`, `bis`, `jst`,
  `lv`, `polity`, `cow`, `reign`, `massmob`, etc.); sole reader of `data/raw/`.
- `domains/` — separated targets, features, and models across `conflict/`, `currency/`, and `banking/`.
- `evaluation/` — unified backtester and cross-domain `scorecard`.
- `publish/` — `risk_table` and frozen `vintage` exports.
- `research/` — standalone empirical studies outside the production pipeline.

Single CLI entrypoint: `uv run python -m seldon <command>` + `Makefile`. Verified bit-for-bit parity
with pre-refactor commit `5b42125`.

### Checkpoint 2.2 — Explicit Publication Lags (`available_from`) — Completed 2026-11-12

Report: `reports/2026-11-12-session-stage2-lags.md`.

- Conservative release schedules encoded in `core/temporal.py`: annual macro fundamentals (GMD/WDI/BIS)
  for year $t$ become available April 1 of $t+1$; December IMF FX rates on January 1 of $t+1$; UCDP
  ACD belligerent lists after calendar year $Y$ ends.
- Production annual forecasts run **as of January 1**: FX & conflict history through $t$, annual
  macro through $t-1$. An **April 1** sensitivity slice (`apr1`, macro through $t$) is tracked in
  `make variants`.
- Monthly conflict features within an incomplete year $Y$ use ACD belligerents from $Y-1$.
- Pytest invariant ensures no feature accesses any observation prior to its `available_from` date.

### Checkpoint 2.3 — Forecast Vintage & Public Repository — Completed 2026-11-12

Report: `reports/2026-11-12-session-stage2-vintage.md`.

- `forecasts/2026-11/`: immutable `predictions.csv` and `metadata.json` (Git commit hash, `as_of`
  dates, raw-file `sha256` checksums, post-audit out-of-sample metrics with 95% CIs).
- Public `README.md`, `.gitignore` rules keeping `data/raw/` and `data/processed/` out of Git, and
  GitHub Actions CI (`.github/workflows/ci.yml`).

## Stage 3. Structural Features on a 1–2 Year Horizon — Completed 2026-11-13

Goal: test whether richer institutional, ethnic, dyadic, and external-balance indicators catch
"quiet" countries and interstate wars missed by conflict history alone (Georgia 2008, Ukraine 2014 &
2022, Karabakh 2020) and regional FX vulnerabilities.

### Checkpoint 3.1 — Live Conflict Horizon & Domestic Structural Features — Completed 2026-11-12

Report: `reports/2026-11-12-session-stage3-1.md`.

- Upgraded to **UCDP 26.1** (GED & ACD through 2025), shifting live conflict forecasts to 2026
  (`as_of = 2026-01-01`).
- Tested **V-Dem v16** (levels and 2-year shifts in electoral/liberal democracy, polarization,
  repression, anocracy) in place of Polity5 (frozen in 2018), plus **ETH EPR 2023** (excluded/discriminated
  population shares, status downgrades, transborder ethnic kin `TEK`).
- **Result:** Neither V-Dem nor EPR improved out-of-sample AUC over parsimonious `pitf` globally, on
  peaceful states ($\ge 10$ quiet years), or on post-Soviet countries (all $\Delta\text{AUC}$ CIs
  span zero). Production model kept on `pitf`.
- Fixed currency-union edge case: year-$t$ rows omit 2-year FX forecasts if the country adopts a
  shared currency in $t+1$ (e.g., Bulgaria-2026).

### Checkpoint 3.2 — Directed Country-Dyad Conflict Panel — Completed 2026-11-12

Report: `reports/2026-11-12-session-stage3-2.md`.

- Built directed dyadic panel (`country A ← country B × year`) for contiguous neighbors (land/sea
  $\le 400$ miles) and five major powers, integrating UCDP 26.1 dyadic targets, World Bank/SIPRI
  military spending, UNGA Ideal Points (2026 release), and COW Dyadic MID 4.02 (with explicit lags).
- **Result:** At the dyad level, the model achieves `0.848` pooled AUC vs `0.794` for pair-memory +
  contiguity (`+0.054` [`-0.005`, `+0.109`]; `+0.117` [`+0.012`, `+0.224`] among close neighbors),
  driven by MID history and borders rather than UN voting or capability ratios. Combining internal
  and external risk channels (`dy_intext`) yields a borderline country-level gain (`+0.021`
  [`-0.001`, `+0.043`] globally, `+0.032` post-Soviet) with 19% probability overprediction, while
  still missing Georgia-2008 and Ukraine-2014. Production showcase left unchanged.

### Checkpoint 3.3 — Structural Currency Block — Completed 2026-11-12

Report: `reports/2026-11-12-session-stage3-3.md`.

- Tested four structural families across monthly (`deval15`) and annual (`deval25`) FX contours:
  Ilzetzki–Reinhart–Rogoff (`IRR`) de facto regimes + live USD/EUR peg share, Greenspan–Guidotti
  reserves-to-short-term-debt, **Bruegel REER** (178 countries, real overvaluation $\times$ peg),
  and **IMF Commodity Terms of Trade** (rolling historical weights).
- **Result:** Significant global gain on monthly `deval15` (`+0.054` [`+0.014`, `+0.099`] vs
  production logit), but concentrated pre-2010 (`+0.092` pre-2010 vs `+0.025` post-2010). No
  reliable gain on annual `deval25` or post-Soviet states (`0.446 → 0.506`).
- **Stage 3 wrap-up (2026-11-13):** Production probability columns unchanged; two live descriptive
  "coiled spring" indicators (`peg_share12`, `reer_peg`) added to `reports/risk_2025_26.csv`.

## Stage 4. Monthly Horizon (3–12 Months), Contagion & Challenger Selection

### Checkpoint 4.1 — Trade-Network FX Contagion & Reserve Drain — Completed 2026-11-13

Report: `reports/2026-11-13-session-stage4-1.md`.

- Built lagged bilateral trade matrices (`A ↔ B × year`) and partner currency-shock features at 3,
  6, and 12-month horizons; audited World Bank `IDS` missingness (`gg_missing`) and historical
  parallel-market FX coverage.
- **Result:** Trade contagion yields `+0.045 / +0.037 / +0.040` over baseline monthly logit at
  3 / 6 / 12m (lifting Kazakhstan-2015 from rank 157 to 21 six months ahead), though redundant on
  top of `struct`. Best single discovery: **peg defense via reserve drain (`peg_resfall` = peg
  share $\times$ 12m reserve drop)**, making `all_brk` the first specification to significantly beat
  `struct` on post-Soviet countries. Added `peg_resfall` as a descriptive risk-table indicator.

### Checkpoint 4.2 — Monthly Armed Conflict Escalation — Completed 2026-11-13

Report: `reports/2026-11-13-session-stage4-2.md`.

- Built monthly `state_war25` onset models at 3, 6, and 12-month horizons (1996–2025, 1-month UCDP
  event lag, automated UCDP Candidate Events ingestion).
- **Result:** Monthly `pitf` reaches `0.871 / 0.867 / 0.864` pooled AUC. Sub-threshold "smoke before
  fire" clashes (1–24 deaths, state/non-state/one-sided violence at home and in neighbors) add a
  modest significant gain at short horizons (`+0.012` [`+0.005`, `+0.021`] globally at 3m; `+0.029`
  on states peaceful $\ge 5$ years). On the `conflictforecast.org` target, Seldon's non-text models
  nearly match their text-augmented `all` benchmark (`0.930` vs `0.934` at 3m, tied on quiet states).

### Checkpoint 4.3 — Network Contagion, Critical Slowing Down & Showcase Selection

Report: `reports/2026-11-13-session-stage4-3.md`.

- [x] **Conflict network contagion** across linguistic, regime, geographic, and ethnic-kin graphs:
  null in annual `pitf_net_cl` (`0.8240` vs `0.8247`), positive in 3-month `smoke_nb_net` (`0.8875`,
  `+0.0424` on quiet states); archival `MassMob`/`REIGN` add nothing beyond `smoke_nb_net`.
- [x] **Critical slowing down (`CSD`)** — rolling variance and `AR(1)` autocorrelation in monthly
  stress series: hypothesis rejected (`80.6%` zeros in monthly conflict series; `-0.023…-0.042` AUC
  drag in FX).
- [x] **Publication-lag audit & challenger selection (2026-11-13):** enforced 2-month publication
  lag on `imf_reserves`; kept production `fx_deval15_12m` intact while promoting compact live-data
  challenger `fx_deval15_12m_live` (`live_fx_swap`: `0.669` pooled AUC globally, `0.744` within-year)
  side-by-side in `risk_2025_26.csv` and `scorecard.csv`.
- [ ] **December vintage `forecasts/2026-12/`** — scheduled for December 2026 (`make data` →
  `make all` → `make test` → `uv run python -m seldon vintage 2026-12`).

## Stage 5. Static Public Web Showcase (`web/`) — Completed 2026-11-13

Reports: `reports/2026-11-13-session-stage5-1.md`, `-2.md`, `-3.md`.

- **Checkpoint 5.1 — Deterministic Web Export (`publish/web_export.py`, `make web`):** exports 240
  static JSON files into `web/data/`, computing exact log-odds feature attributions ($\beta \cdot z$)
  and two-way clustered 95% confidence intervals (`publish/explain.py`) for all four logistic models,
  bilingual country search (`ISO3`, EN, RU), and SHA-256 contract tests (`tests/test_web_export.py`).
- **Checkpoint 5.2 — Design System & Core Views:** zero-build ES-module frontend (`web/`) with
  unified missing-data state components, Home (`H1`–`H5` historical rank ribbons & regional lens),
  URL-stateful Forecasts screener (`P1`–`P3`), 174 Country Dossiers (`C1`–`C6`), and Vintage Archive.
- **Checkpoint 5.3 — Verification View, Structure×Pressure Scatter & Mobile Layout:** calibration
  curves, Brier skill breakdown, interactive blind-vs-realized outcome toggle, `P4` scatter, and
  responsive mobile drawers (91 automated tests passing).
