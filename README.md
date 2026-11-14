**English** · [Русская версия](README.ru.md)

# Seldon — Armed Conflict, Currency & Banking Crisis Forecasting

Country-level early-warning models that estimate the probability of **state-based armed conflict**,
**currency devaluations**, and **systemic banking crises** across 174 countries, with a dedicated
regional lens on the 15 post-Soviet states. Named after Hari Seldon from Isaac Asimov's
*Foundation*. The goal is pragmatic: probabilistic forecasts you can trust strictly to the extent
they have been verified out-of-sample — and an explicit record of where they are blind.

This is an open research project, not a commercial product or investment advice. Forecasts are
probabilities, not binary calls: crisis onsets are rare events, false alarms are expected, and
calibration and ranking matter more than point cutoffs.

## Core Principle: Strictly Out-of-Sample & Point-in-Time

Macroeconomic and conflict models easily "predict" the past when subtle look-ahead leakage creeps
in. In November 2026, an external audit uncovered look-ahead leaks across nearly every block of the
early prototype; once fixed, several flattering numbers disappeared. Everything in Seldon is now
built around three invariants:

1. **Expanding-window out-of-sample backtesting (`forecast_asof`).** Models train strictly on past
   observations and predict unseen future windows, stepping forward year by year (or month by
   month). Every pipeline exposes a `forecast_asof` contract, and automated tests verify that
   corrupting or deleting all data after the forecast origin leaves predictions unchanged to $10^{-12}$.
2. **Explicit publication lags (`available_from`).** Macroeconomic statistics arrive late: annual
   GDP for year $t$ is published in spring $t+1$. Annual forecasts are issued "as of January 1" and
   only use series already published by that date: exchange rates and conflict history through $t$,
   annual macro fundamentals through $t-1$.
3. **Immutable forecast vintages.** Historical backtests are still hindsight evaluations. Once a
   month, live forecasts are frozen into `forecasts/YYYY-MM/` alongside SHA-256 checksums of code
   and raw data, committed to Git, and scored prospectively against realized events.

## Out-of-Sample Performance (Post-Audit)

Expanding-window out-of-sample evaluation with publication lags enforced. **ROC AUC** is the share
of (event, non-event) pairs ranked in the right order: `0.5` is a coin flip, `1.0` is ideal.

- **Pooled AUC** is computed across all country-periods jointly: to score high, a model must tell
  both *which countries* are riskier and *which years* are riskier.
- **Within-year AUC** averages cross-sectional AUC year by year: whether the model ranks countries
  correctly in the same year. When within-year AUC noticeably exceeds pooled AUC, the model
  separates countries well but misses synchronized global or regional crisis waves.
- **95% CI** uses year-blocked bootstrap (500 replicates). If the interval covers `0.5`, the model
  is indistinguishable from a coin flip.
- **Base rate** is the empirical event frequency in the evaluation sample.

| Target Event | Sample | Base Rate | Model | Pooled AUC [95% CI] | Within-Year AUC |
| --- | --- | --- | --- | --- | --- |
| Onset of state-based armed conflict ($\ge$ 25 battle deaths) next year | Global, countries at peace; 151 onsets (1996–2025) | 3.7% | `pitf` (logistic regression) | **0.825** [0.780, 0.867] | 0.859 |
| | | | baseline: conflict-history decay | 0.761 [0.707, 0.816] | 0.788 |
| Currency devaluation $\ge$ 25%/yr over the next 2 years (Dec-to-Dec vs USD) | Global, own-currency states; 246 events (1996–2023) | 8.0% | logistic regression (`logit_long`) | **0.677** [0.612, 0.740] | 0.710 |
| | | | ExtraTrees (`extree_long`) | 0.673 [0.599, 0.753] | 0.758 |
| | Post-Soviet; 37 events | 14.5% | logistic regression | 0.585 [0.466, 0.744] | 0.654 |
| Monthly exchange-rate jump $\ge$ 15% within the next 12 months | Global, states only; 2,103 windows (1996–2024) | 4.9% | logistic regression (`econ_kr`) | **0.619** [0.563, 0.680] | 0.669 |
| | Post-Soviet; 370 windows | 9.7% | logistic regression (`econ_kr`) | 0.425 [0.312, 0.534] | 0.541 |
| Monthly exchange-rate jump $\ge$ 15% within 12 months (challenger) | Global, states only; 2,103 windows (1996–2024) | 4.9% | `live_fx_swap` (live peg & reserve drain) | **0.658** [0.606, 0.734] | 0.737 |
| | Post-Soviet; 370 windows | 9.7% | `live_fx_swap` | 0.471 [0.373, 0.579] | 0.562 |
| Systemic banking crisis over the next 2 years | BIS countries outside JST training sample; 9 pre-crisis years (1990–2015) | 3.3% | Bank of England ExtraTrees (`full`, with yield curve) | 0.671 [0.517, 0.798] | 0.745 |
| | Same, without interest rates & broad money; 30 pre-crisis years | 6.4% | ExtraTrees (`nomoney`) | 0.546 [0.414, 0.654] | 0.496 |

Full paired comparisons are in `reports/scorecard.csv` (`uv run python -m seldon scorecard`).
The `pitf` conflict onset model beats the conflict-memory baseline by `+0.064` [`+0.019`, `+0.104`].

### Where the Models Work

- **Armed conflict onset.** Structural and neighborhood features (infant mortality, regime type,
  neighboring wars, and own conflict history) reach `0.825` pooled AUC (`0.859` within-year). Ahead
  of the 2020 Karabakh war, Azerbaijan ranked **3rd** out of 131 peaceful countries and Armenia **11th**.
- **Annual currency crises.** Currency pegs, past devaluations, inflation, and current-account
  deficits reach `0.677` globally. A year before devaluation, Belarus-2015 ranked **1st** in the
  world, Ukraine and Russia ahead of 1998 ranked **1st** and **3rd**, and Uzbekistan-2017 ranked **7th**.
- **Systemic banking crises** — where a market yield curve exists: an inverted yield slope paired
  with credit booms transfers reasonably well outside the historical JST sample.

### Known Blind Spots & Limitations

- **"Out-of-the-blue" shocks & interstate wars.** Ukraine 2014 & 2022, Georgia 2008, and Syria &
  Libya 2011 do not stand out in domestic structural data a year ahead. The same holds for the 2014
  FX shocks: a year prior, Ukraine ranked 101st of 149 and Russia 60th.
- **Regional contagion waves.** On post-Soviet countries, the 12-month monthly FX model is no
  better than a coin flip (`0.439` pooled AUC; `0.471` for the challenger, `0.550` within-year).
  The 1998 and 2014–2015 crises hit the region simultaneously from external shocks (oil collapse,
  ruble spillover), whereas the baseline uses domestic indicators. `fx_deval15_12m` is reported for
  all countries for completeness, **but should not be relied upon for post-Soviet states**.
- **Banking crisis prediction fails without a yield curve.** Once publication lags are respected
  (credit and GDP known only through $t-2$), dropping interest rates and money reduces AUC to
  `0.546` [`0.414`, `0.654`]. We therefore publish banking probabilities **only** for the 26 BIS
  countries with long-term rates and national broad money; Russia, eurozone members, and most EMs
  carry an explicit `bank_status` reason instead of a number.
- **Residual real-time caveats.** IMF international reserves carry a 2-month lag and monthly UCDP
  events carry a 1-month lag, while remaining monthly IMF series (CPI, rates, money, exports) are
  treated as contemporaneous. Historical data revisions (vintage macro releases) are not replayed.
- **Small-sample uncertainty.** 37 post-Soviet annual devaluations and 9 out-of-JST banking
  pre-crisis country-years produce wide confidence intervals by necessity.

## Forecast Vintages & Risk Table

`forecasts/YYYY-MM/` stores immutable monthly forecast snapshots:

- `predictions.csv` — country probabilities, cross-sectional ranks, and coverage status codes;
- `metadata.json` — `as_of` origins, Git commit & raw-data SHA-256 hashes, and out-of-sample
  metrics for every published model.

Columns in the inaugural snapshot `forecasts/2026-11/`:

| Column | Meaning | Forecast Origin (`as_of`) |
| --- | --- | --- |
| `p_state_war25_2025` | Probability of state-based armed conflict ($\ge$ 25 deaths) in 2025 | Jan 1, 2025 (in Nov vintage UCDP ended at 2024; live risk table now outputs `p_state_war25_2026` on UCDP 26.1) |
| `fxa_logit_2026_27`, `fxa_extree_2026_27` | Probability of $\ge$ 25% annual USD devaluation in 2026 or 2027 | Jan 1, 2026 |
| `fx_deval15_12m` | Probability of $\ge$ 15% single-month FX jump within 12 months | Latest IMF FX month per country (`as_of`) |
| `bank_jst_extree_full_2026_27` | Probability of systemic banking crisis in 2026 or 2027 | Jan 1, 2026 |
| `fx_regime` | `own` — national currency; ISO code (`EUR`, `USD`, `XOF`…) — currency union / dollarized (no own FX risk); `no_imf_fx` — missing IMF rate | |
| `bank_status` | `ok`; `no_yield_curve`; `no_broad_money`; `no_bis_credit` | |

The full analytical table `reports/risk_2025_26.csv` additionally includes the challenger column
`fx_deval15_12m_live` and **descriptive stress indicators** (not probabilities): Exchange Market
Pressure (`emp_z_max3`), `reserves_g12`, `deval12`, `infl12`, `signals_fired`, and live "coiled
spring" peg vulnerability metrics (`peg_share12`, `reer_peg` — real appreciation $\times$ peg, and
`peg_resfall` — peg $\times$ 12-month reserve drain).

## Static Web Showcase (`web/`)

A zero-backend static web app serving as a public forecast journal, cross-country screener, and
post-Soviet regional lens. Precomputed JSON bundles in `web/data/` are read directly by vanilla ES
modules in `web/` (ready for GitHub Pages; interface currently in Russian, searchable by English
country names, Russian names, and `ISO3` codes).

```bash
make web      # reports/ + forecasts/ → web/data/*.json (~30s; included in make all)
make serve    # http://localhost:8791
```

Contract tests in `tests/test_web_export.py` enforce that:
- every displayed number matches `reports/risk_2025_26.csv`, `reports/scorecard.csv`, or `forecasts/*`;
- missing coverage is rendered as an explicit reason (`no_bis_credit`, `shared:EUR`), never `0.0%`;
- uncertainty intervals appear only for logistic models (trees explicitly state no interval);
- feature attributions are reported in log-odds ($\beta \cdot z$) with true publication-lagged observation dates.

## Architecture (`src/seldon/`)

Five domain-clean layers plus a sandboxed research shelf:

```text
core/        domain-agnostic primitives:
             entities (ISO3 / GW / COW / Polity country codes, currency unions),
             panel (country×period grids, calendar lags),
             temporal (as_of contract, available_from publication lags),
             calibration, metrics (pooled & within-year AUC, block bootstrap)
sources/     data downloaders and parsers; the ONLY layer reading data/raw/
domains/     features, targets, and forecasting models by crisis domain:
  conflict/    armed conflict: panel, features*, baseline, model_hist; dyads, model_dyad
  currency/    currency crises: monthly EMP/jump (kr_*, econ_kr), annual (long_*, econ_long)
  banking/     banking crises: econ_boe (Bank of England JST replication), econ_boe_em
evaluation/  unified expanding-window backtest & scorecard
publish/     risk_table, vintage snapshots, web_export, explain (log-odds & clustered CIs)
research/    standalone studies outside the production pipeline (news, bank runs, ViEWS/CF, parties)
```

### Anti-Leakage Test Suite

`tests/test_no_leakage.py` and `tests/test_web_export.py` (90+ tests running in seconds on compact
synthetic panels and the checked-in web bundle — no raw data needed, executed in GitHub Actions CI):

- **Future blindness** — corrupting all rows after `as_of` with extreme noise or `NaN`s leaves
  forecasts identical to $10^{-12}$ (plus a meta-test proving a deliberately leaky function fails).
- **Publication lags** — corrupting data not yet released by `as_of` (current-year GDP, current-year
  UCDP ACD belligerent lists, unlagged IMF reserves) leaves forecasts unchanged.
- **Target–evaluable alignment**, **currency-union exclusions**, and **web bundle SHA-256 parity**.

## Reproducing From Scratch

Requires Python $\ge 3.12$, [`uv`](https://docs.astral.sh/uv/), and `pdftotext` (`poppler`:
`brew install poppler` on macOS or `apt install poppler-utils` on Ubuntu/Debian).

```bash
uv sync          # install locked dependencies
make test        # run anti-leakage & contract test suite (no raw data required)
make data        # download raw public datasets into data/raw (~1 GB)
make all         # full pipeline: panels → models → risk table → scorecard → web/data
make vintage     # freeze monthly snapshot into forecasts/YYYY-MM/
```

> **Note on `make data`:** The IMF website blocks automated downloads of the Laeven & Valencia
> (2018) working paper PDF (`IMF WP/18/206`). The command prints the exact URL and target path
> (`data/raw/lv/wp18206.pdf`) and verifies its SHA-256 checksum on re-run.

Run individual steps via `uv run python -m seldon <command>` (run without arguments to list all CLI
commands). `make variants` runs sensitivity slices and Stage 3–4 candidate models; `make replicate`
reproduces Bluwstein et al. (Bank of England) on JST. Re-running `make all` on identical inputs
reproduces CSV tables byte-for-byte (6-decimal formatting eliminates multi-threaded tree float noise).

## Data Sources

| Source | Usage in Seldon |
| --- | --- |
| [UCDP](https://ucdp.uu.se) 26.1 | Georeferenced Event Dataset (GED), Candidate Events, UCDP/PRIO Armed Conflict Dataset (ACD) |
| [Gleditsch & Ward](http://ksgleditsch.com/data-4.html), [COW Direct Contiguity 3.2](https://correlatesofwar.org/data-sets/direct-contiguity/) | Independent state system membership and land/sea contiguity |
| [Polity5](https://www.systemicpeace.org/inscrdata.html), [World Bank WDI](https://data.worldbank.org) | Regime authority characteristics, infant mortality, population, CPI, external debt |
| [V-Dem](https://github.com/vdeminstitute/vdemdata) v16, [ETH EPR](https://icr.ethz.ch/data/epr/) 2023 | Institutional indices, polarization, ethnic power exclusion & transborder kin (tested in Stage 3.1) |
| [UNGA Ideal Points](https://dataverse.harvard.edu/dataset.xhtml?persistentId=hdl:1902.1/12379), [COW Dyadic MID 4.02](https://correlatesofwar.org/data-sets/mids/), World Bank military expenditure | Directed dyadic conflict model: foreign-policy alignment, militarized disputes, capability asymmetry |
| [Ilzetzki–Reinhart–Rogoff](https://www.ilzetzki.com/irr-data), [Bruegel REER](https://www.bruegel.org/publications/datasets/real-effective-exchange-rates-for-178-countries-a-new-database), IMF Commodity Terms of Trade | Exchange rate regimes, real effective exchange rates (178 countries), Greenspan–Guidotti reserve adequacy |
| [IMF SDMX API](https://data.imf.org) | Monthly exchange rates, international reserves, CPI, interest rates, broad money, exports, bilateral trade |
| [Global Macro Database (GMD)](https://www.globalmacrodata.com) 2026_09 | Annual macroeconomic fundamentals and historical crisis chronologies |
| [BIS](https://data.bis.org) | Total credit to the private non-financial sector |
| [JST Macrohistory](https://www.macrohistory.net/database/) R6 & [Bank of England code](https://github.com/bank-of-england/MachineLearningCrisisPrediction) | Historical macro-financial panel and Bluwstein et al. banking crisis benchmark |
| Laeven & Valencia (2018), IMF WP/18/206 | Systemic banking and currency crisis dates |
| REIGN, Mass Mobilization, CEPII GeoDist, [conflictforecast.org](https://conflictforecast.org), PITF | Research benchmarks and comparison baselines |

Raw data are never committed to Git — only deterministic downloaders and SHA-256 verification.

## Roadmap & Project Status

Detailed milestone history and benchmark logs are in [ROADMAP.md](ROADMAP.md) ([на русском](ROADMAP.ru.md)),
and an index of all generated tables and session reports is in [reports/README.md](reports/README.md)
([на русском](reports/README.ru.md)).

- **Stages 1–2 (Completed):** Post-audit leak elimination, 5-layer architecture, publication-lag
  enforcement (`available_from`), and inaugural frozen forecast vintage `forecasts/2026-11/`.
- **Stage 3 (Completed):** 1–2 year structural features (V-Dem, EPR, directed country dyads, IRR
  regimes, Bruegel REER, Greenspan–Guidotti, IMF commodity terms of trade). Production probabilities
  kept parsimonious; live peg-vulnerability indicators added to the risk table.
- **Stage 4 (Wrapping up):** 3–12 month horizon — trade-network FX contagion & peg-defense reserve
  drain (`peg_resfall`, 4.1); monthly conflict escalation with live UCDP sub-war clashes (4.2);
  conflict network contagion, critical slowing down tests (rejected), and selection of challenger
  `fx_deval15_12m_live` alongside production models (4.3). Next: `forecasts/2026-12/` vintage in December.
- **Stage 5 (Completed):** Zero-backend static web showcase (`web/`) with full data-contract tests.

## Citation & License

Code is released under the **MIT License** (see [LICENSE](LICENSE)). Analytical reports and frozen
forecast vintages (`reports/`, `forecasts/`) are licensed under **CC BY 4.0** — please attribute
**Seldon** (machine-readable citation metadata in [CITATION.cff](CITATION.cff)). Underlying raw
datasets remain subject to the terms of their respective providers.
