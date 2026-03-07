# SMM272 Risk Analysis Coursework 2025-2026

Quantitative risk analysis coursework covering six questions in financial risk modeling: distributional analysis, rolling VaR/ES estimation and backtesting, Monte Carlo power analysis of the Kupiec test, VaR of a non-linear options portfolio, and CME SPAN margin methodology. Implemented entirely in Python with publication-quality figures and CSV outputs for every question.

---

## Portfolio

| Ticker | Company        |
|--------|----------------|
| AAPL   | Apple          |
| MSFT   | Microsoft      |
| IBM    | IBM            |
| NVDA   | Nvidia         |
| GOOGL  | Alphabet       |
| AMZN   | Amazon         |

**Weights:** Equal (1/6 ≈ 16.67% each)  
**Sample period:** 1 January 2014 – 31 December 2025  
**Return type:** Daily log returns

---

## Questions at a Glance

| # | Topic | Key Method | Key Result |
|---|-------|-----------|------------|
| Q1 | Statistical analysis, rolling VaR/ES, backtesting | HS / Normal / Student-t / GARCH | HS: 49 violations at 99% CI |
| Q2 | Power of the Kupiec test | Monte Carlo simulation | Power 82.9% (90% CI), 38.6% (99% CI) at T=2,893 |
| Q5 | CME SPAN margin methodology | SPAN 16-scenario risk arrays | SPAN netting reduces margin 38–62% vs no-netting |
| Q6 | VaR of a portfolio with options | Black-Scholes + EWMA + Monte Carlo | VaR₉₉ = $8,997; ES₉₉ = $10,115 (10-day, 10,000 sims) |

---

## Project Structure

```
SMM272_Risk_Analysis/
├── config.py                          # Shared constants: tickers, dates, paths, VaR/MC metadata
├── logger.py                          # Shared logging utilities (setup_run_logger, get_logger)
├── docs/
│   ├── main.tex                       # LaTeX report (all questions)
│   ├── references.bib                 # BibTeX bibliography
│   ├── q5_generate_docs.py            # Word report generator for Q5
│   ├── q6_generate_docs.py            # Word report generator for Q6
│   └── Q5_SPAN_Margining_Report.docx  # Generated Word report (Q5)
│   └── Q6_VaR_Options_Portfolio_Report.docx  # Generated Word report (Q6)
├── Q1/                                # Question 1
│   │
│   ├── q1_main.py                     # Unified runner — runs any combination of Q1 sub-parts
│   ├── q1_methods.py                  # Single canonical METHODS list (tag, label, fn, colour)
│   │
│   ├── ── Part 1: Statistical Analysis of Portfolio Returns ──
│   ├── q1_1_statistical_analysis.py   # Part 1 orchestrator
│   ├── q1_1_download_prices.py        # Download adjusted closing prices (yfinance, cached)
│   ├── q1_1_build_portfolio.py        # Build EW portfolio, compute log returns
│   ├── q1_1_descriptive_stats.py      # Descriptive statistics & risk metrics
│   ├── q1_1_normality_tests.py        # JB, Shapiro-Wilk, D'Agostino K², KS tests
│   ├── q1_1_autocorrelation.py        # Ljung-Box tests on returns & squared returns
│   ├── q1_1_correlation_analysis.py   # Pairwise correlation matrices (full / crisis / non-crisis)
│   ├── q1_1_risk_metrics.py           # VaR, CVaR, Sharpe ratio, drawdown
│   ├── q1_1_timeseries_diagnostics.py # ADF stationarity, ARCH-LM tests
│   ├── q1_1_visualisations.py         # 9 diagnostic figures
│   ├── q1_1_portfolio_conclusions.py  # Narrative summary of Part 1 findings
│   ├── q1_1_summary.py                # Tabular summary output
│   ├── output_q1_1/                   # Part 1 outputs (CSVs + figures)
│   │   ├── adjusted_close_prices.csv
│   │   ├── log_returns.csv
│   │   ├── portfolio_returns.csv
│   │   ├── descriptive_statistics.csv
│   │   ├── risk_metrics.csv
│   │   ├── risk_return_summary.csv
│   │   ├── correlation_matrix.csv
│   │   ├── correlation_matrix_crisis.csv
│   │   ├── correlation_matrix_non_crisis.csv
│   │   └── fig1_normalised_prices.png … fig9_rolling_statistics.png
│   │
│   ├── ── Part 2: Rolling VaR and ES ──
│   ├── q1_2_statistical_analysis.py   # Part 2 orchestrator
│   ├── q1_2_rolling_window.py         # Rolling 6-month window generator
│   ├── q1_2_var_historical.py         # Historical Simulation VaR/ES
│   ├── q1_2_var_normal.py             # Parametric Normal VaR/ES
│   ├── q1_2_var_studentt.py           # Parametric Student-t VaR/ES (MLE)
│   ├── q1_2_var_garch.py              # GARCH(1,1) VaR/ES
│   ├── output_q1_2/                   # Part 2 outputs
│   │   ├── rolling_var_es_results.csv
│   │   ├── backtest_summary.csv
│   │   ├── fig_rolling_var_comparison.png
│   │   ├── fig_rolling_es_comparison.png
│   │   ├── fig_var_es_breach_panels.png
│   │   └── fig_exceedance_rates.png
│   │
│   ├── ── Part 3: VaR Violations Analysis ──
│   ├── q1_3_var_violations.py         # Part 3 orchestrator
│   ├── q1_3_rolling.py                # Multi-level rolling VaR (90% / 95% / 99%)
│   ├── q1_3_count_violations.py       # Violation counting and summary statistics
│   ├── q1_3_logging.py                # Structured logging of violation results
│   ├── q1_3_plots.py                  # Three diagnostic figures
│   ├── output_q1_3/                   # Part 3 outputs
│   │   ├── rolling_var_all_levels.csv
│   │   ├── violations_summary.csv
│   │   ├── violation_series.csv
│   │   ├── fig_violations_barchart.png
│   │   ├── fig_violations_heatmap.png
│   │   └── fig_violation_timeseries.png
│   │
│   ├── ── Part 4: Formal Backtesting — Kupiec, Christoffersen, Duration, DQ ──
│   ├── q1_4_backtesting.py            # Part 4 orchestrator
│   ├── q1_4_kupiec.py                 # Kupiec (1995) POF test — LR_uc ~ χ²(1)
│   ├── q1_4_christoffersen.py         # Christoffersen (1998) independence + CC test
│   ├── q1_4_duration.py               # Christoffersen-Pelletier (2004) Weibull duration test
│   ├── q1_4_dq_test.py                # Engle-Manganelli (2004) Dynamic Quantile test
│   ├── q1_4_backtests.py              # Run all tests for all model × CI combinations
│   ├── q1_4_logging.py                # Structured logging of backtest results
│   ├── q1_4_plots.py                  # Backtest diagnostic figures
│   └── output_q1_4/                   # Part 4 outputs
│       ├── backtest_results.csv
│       ├── fig_backtest_lr_stats.png
│       └── fig_backtest_pvalues_heatmap.png
│
├── Q2/                                # Question 2 — Power of the Kupiec Test
│   ├── q2_main.py                     # Orchestrator (fit → power → power_vs_T → power_vs_ρ → figures → summary)
│   ├── q2_fit_models.py               # Fit Gaussian H₀ and GARCH(1,1) H₁ to Q1 returns
│   ├── q2_kupiec_power.py             # Monte Carlo power estimation at T=2,893
│   ├── q2_power_vs_T.py               # Power as a function of sample size T
│   ├── q2_power_vs_persistence.py     # Power as a function of GARCH persistence ρ = α+β
│   ├── q2_visualisations.py           # 7 figures
│   ├── q2_summary.py                  # Print and save consolidated results
│   └── output_q2/
│       ├── fitted_params.csv
│       ├── power_summary.csv
│       ├── power_vs_T.csv
│       ├── power_vs_persistence.csv
│       ├── q2_full_summary.csv
│       ├── fig_q2_1_lr_distributions.png
│       ├── fig_q2_2_violation_distributions.png
│       ├── fig_q2_3_power_vs_T.png
│       ├── fig_q2_4_power_vs_persistence.png
│       ├── fig_q2_5_size_vs_T.png
│       ├── fig_q2_6_power_heatmap.png
│       └── fig_q2_7_return_diagnostics.png
│
├── Q5/                                # Question 5 — CME SPAN Margining
│   ├── q5_main.py                     # Orchestrator (scenarios → margins → figures → summary)
│   ├── q5_market_data.py              # COMEX Copper front-month data (CME settlement prices)
│   ├── q5_option_pricing.py           # Black-76 futures options pricing
│   ├── q5_span_scenarios.py           # 16 SPAN price-×-vol scenario generation
│   ├── q5_span_engine.py              # SPAN risk array + inter-month spread charge
│   ├── q5_positions.py                # Four position pairs (speculative, spreads, hedges)
│   ├── q5_margin_calculator.py        # No-netting vs SPAN-netting margin comparison
│   ├── q5_visualisations.py           # 6 figures
│   ├── q5_summary.py                  # Print tables and save CSVs
│   └── output_q5/
│       ├── span_risk_arrays.csv
│       ├── margin_comparison.csv
│       ├── margin_decomposition.csv
│       ├── netting_benefits.csv
│       ├── fig_q5_1_risk_array_heatmap.png
│       ├── fig_q5_2_scenario_profiles.png
│       ├── fig_q5_3_margin_comparison.png
│       ├── fig_q5_4_netting_benefit.png
│       ├── fig_q5_5_option_payoffs.png
│       └── fig_q5_6_span_sensitivity.png
│
├── Q6/                                # Question 6 — VaR of a Portfolio with Options
│   ├── q6_main.py                     # Orchestrator (market_data → cov → portfolio → sim → risk → figures → summary)
│   ├── q6_market_data.py              # 2-year daily prices, log-returns, historical vols (yfinance)
│   ├── q6_covariance.py               # EWMA daily covariance matrix (λ=0.94, RiskMetrics)
│   ├── q6_portfolio.py                # Black-Scholes option pricing + Greeks for 4 positions
│   ├── q6_simulation.py               # 10,000 Cholesky-correlated log-normal paths (10-day)
│   ├── q6_var_es.py                   # VaR₉₉, ES₉₉ (non-Gaussian) + Euler risk decomposition
│   ├── q6_visualisations.py           # 6 publication-quality figures
│   ├── q6_summary.py                  # Print tables, save 4 CSVs
│   └── output_q6/
│       ├── q6_portfolio_summary.csv   # Positions, spot, strike, TTM, vol, B-S price, Greeks
│       ├── q6_risk_metrics.csv        # VaR, ES, marginal VaR/ES, component VaR/ES by position
│       ├── q6_covariance_matrix.csv   # 4×4 EWMA daily covariance matrix
│       ├── q6_pnl_statistics.csv      # Simulated P&L distribution summary statistics
│       ├── fig_q6_1_pnl_distribution.png
│       ├── fig_q6_2_risk_decomposition.png
│       ├── fig_q6_3_correlation_heatmap.png
│       ├── fig_q6_4_option_profiles.png
│       ├── fig_q6_5_simulated_returns.png
│       └── fig_q6_6_instrument_pnl.png
│
├── logs/                              # Timestamped run logs
├── .gitignore
└── README.md
```

---

## Running the Code

### Prerequisites

```bash
pip install yfinance pandas numpy matplotlib seaborn scipy statsmodels arch python-docx
```

> **`arch`** is required for GARCH(1,1) estimation (Q1 Parts 2–4 and Q2). If unavailable, GARCH automatically falls back to the Parametric Normal model.

### Unified runner — `q1_main.py`

All Q1 sub-parts are run via `q1_main.py` from the project root:

```bash
# Run all parts (1, 2, 3, 4) in sequence
python Q1/q1_main.py

# Run a single part
python Q1/q1_main.py --parts q1_1
python Q1/q1_main.py --parts q1_2
python Q1/q1_main.py --parts q1_3
python Q1/q1_main.py --parts q1_4

# Run multiple specific parts
python Q1/q1_main.py --parts q1_1 q1_2

# Preview without executing
python Q1/q1_main.py --dry-run
```

### Running Q2

```bash
python Q2/q2_main.py

# Run specific steps
python Q2/q2_main.py --parts fit power
python Q2/q2_main.py --dry-run
```

### Running Q5

```bash
python Q5/q5_main.py

# Run specific steps
python Q5/q5_main.py --parts margins figures
python Q5/q5_main.py --dry-run
```

### Running Q6

```bash
python Q6/q6_main.py

# Run specific steps
python Q6/q6_main.py --parts simulation risk figures
python Q6/q6_main.py --dry-run
```

### Generating Word Reports

```bash
python docs/q5_generate_docs.py   # → docs/Q5_SPAN_Margining_Report.docx
python docs/q6_generate_docs.py   # → docs/Q6_VaR_Options_Portfolio_Report.docx
```

---

## Question 1 — Part 1: Statistical Analysis

### What it does

| Step | Module | Description |
|------|--------|-------------|
| 1 | `q1_1_download_prices.py` | Fetch daily adjusted closing prices from Yahoo Finance (cached to CSV) |
| 2 | `q1_1_build_portfolio.py` | Compute daily log returns; construct equally weighted portfolio |
| 3 | `q1_1_descriptive_stats.py` | Mean, median, std dev, skewness, excess kurtosis |
| 4 | `q1_1_normality_tests.py` | Jarque-Bera, Shapiro-Wilk, D'Agostino K², Kolmogorov-Smirnov |
| 5 | `q1_1_autocorrelation.py` | Ljung-Box Q-test on raw returns and squared returns |
| 6 | `q1_1_correlation_analysis.py` | Pearson correlation matrices: full sample, 2020 crisis, non-crisis |
| 7 | `q1_1_risk_metrics.py` | VaR (historical), CVaR, Sharpe ratio, max drawdown |
| 8 | `q1_1_timeseries_diagnostics.py` | ADF unit-root test, ARCH-LM test for heteroskedasticity |
| 9 | `q1_1_visualisations.py` | 9 figures (see below) |
| 10 | `q1_1_portfolio_conclusions.py` | Narrative findings summary |

### Figures produced

| File | Description |
|------|-------------|
| `fig1_normalised_prices.png` | Rebased price indices (base = 100) |
| `fig2_cumulative_returns.png` | Cumulative log return paths |
| `fig3_daily_returns.png` | Daily return time series |
| `fig4_histogram_normal_kde.png` | Return histogram with fitted normal and KDE overlay |
| `fig5_qq_plot.png` | Normal Q-Q plot of portfolio returns |
| `fig6_boxplots.png` | Box plots by asset |
| `fig7_correlation_heatmap.png` | Pearson correlation heatmap |
| `fig8_acf_plots.png` | ACF of returns and squared returns |
| `fig9_rolling_statistics.png` | Rolling mean and rolling volatility (6-month window) |

### Key findings

- Excess kurtosis of **6.33** and negative skewness confirm significant fat tails — the Normal distribution is decisively rejected by all four normality tests.
- Ljung-Box test on squared returns returns LR ≈ 2,163 (lag 40), confirming strong ARCH effects (volatility clustering).
- Pairwise correlations are elevated during the 2020 COVID-19 crisis — diversification benefits collapse under stress.

---

## Question 1 — Part 2: Rolling VaR and ES

### What it does

Four methods estimate one-step-ahead VaR and ES at **99% confidence** using a rolling **6-calendar-month** estimation window (2,893 daily forecasts, 2014-07-01 to 2025-12-30):

| Method | Module | Approach |
|--------|--------|----------|
| Historical Simulation (HS) | `q1_2_var_historical.py` | Empirical 1st percentile of window returns |
| Parametric Normal | `q1_2_var_normal.py` | Closed-form Normal quantile: μ + Φ⁻¹(0.01)·σ |
| Parametric Student-t | `q1_2_var_studentt.py` | MLE fit of (ν, μ, σ); Student-t quantile |
| GARCH(1,1) Normal | `q1_2_var_garch.py` | MLE GARCH fit; one-step-ahead conditional σ² |

### Breach rate summary (99% CI)

| Method | Violations | Breach Rate | Target |
|--------|-----------|-------------|--------|
| Historical Simulation | 49 | 1.694% | 1.000% |
| Parametric Student-t | 63 | 2.178% | 1.000% |
| GARCH(1,1) Normal | 82 | 2.834% | 1.000% |
| Parametric Normal | 87 | 3.007% | 1.000% |

### Figures produced

| File | Description |
|------|-------------|
| `fig_rolling_var_comparison.png` | Rolling 99% VaR — all four methods on one panel |
| `fig_rolling_es_comparison.png` | Rolling 99% ES — all four methods on one panel |
| `fig_var_es_breach_panels.png` | Per-method panels: VaR/ES vs realised returns with breach highlights |
| `fig_exceedance_rates.png` | Observed vs expected breach rates bar chart |

---

## Question 1 — Part 3: VaR Violations Analysis

### What it does

Counts and summarises VaR violations at three confidence levels (90%, 95%, 99%) across all four models, producing violation time series and diagnostic figures for input to the formal backtesting in Part 4.

### Figures produced

| File | Description |
|------|-------------|
| `fig_violations_barchart.png` | Grouped bar chart: observed vs nominal violation rates at all 3 CIs |
| `fig_violations_heatmap.png` | Heatmap: observed (blue) and expected (grey) violation counts, 4×3 grid |
| `fig_violation_timeseries.png` | Cumulative violations over time vs expected line, one panel per CI level |

---

## Question 1 — Part 4: Formal Backtesting

### What it does

Applies four formal statistical backtests to every model × confidence level combination (12 pairs total):

| Test | Module | Statistic | What it tests |
|------|--------|-----------|---------------|
| **Kupiec (1995) POF** | `q1_4_kupiec.py` | LR_uc ~ χ²(1) | Unconditional coverage: observed violation rate = nominal rate? |
| **Christoffersen (1998) Independence** | `q1_4_christoffersen.py` | LR_ind ~ χ²(1) | Are violations serially independent? Detects clustering |
| **Christoffersen (1998) CC** | `q1_4_christoffersen.py` | LR_cc ~ χ²(2) | Joint: UC + independence |
| **Christoffersen-Pelletier (2004) Duration** | `q1_4_duration.py` | Weibull LR | Time between violations follows Exponential? |
| **Engle-Manganelli (2004) DQ** | `q1_4_dq_test.py` | DQ ~ χ²(k) | Hit sequence predictable from its own lags? |

### Key results (99% CI, 5% significance level)

| Method | Violations | Kupiec | Christoffersen | Duration | DQ |
|--------|-----------|--------|---------------|----------|----|
| Historical Simulation | 49 | Reject | Reject | Reject | Reject |
| Parametric Normal | 87 | Reject | Fail | Reject | Reject |
| Parametric Student-t | 63 | Reject | Reject | Reject | Reject |
| GARCH(1,1) Normal | 82 | Reject | Reject | Reject | Reject |

> All four models are rejected by at least three of five tests at 99% CI. HS has the fewest violations but still fails UC, independence and DQ — driven by violation clustering during 2020 and 2022 stress periods.

### Figures produced

| File | Description |
|------|-------------|
| `fig_backtest_lr_stats.png` | LR test statistics vs χ² critical values (grouped by test) |
| `fig_backtest_pvalues_heatmap.png` | p-value heatmap: model × test × confidence level |

---

## Question 2 — Power of the Kupiec Test

### What it does

Conducts a Monte Carlo study to estimate the **statistical power** of the Kupiec (1995) Proportion of Failures (POF) test under a GARCH(1,1) data-generating process. The analysis answers: *how likely is the test to correctly reject an inadequate VaR model when returns are driven by GARCH volatility clustering?*

| Step | Module | Description |
|------|--------|-------------|
| 1 | `q2_fit_models.py` | Fit Gaussian H₀ (μ, σ) and GARCH(1,1) H₁ (ω, α, β) to Q1 EW portfolio returns |
| 2 | `q2_kupiec_power.py` | Monte Carlo power at T=2,893: simulate GARCH paths → compute VaR → count violations → apply Kupiec test |
| 3 | `q2_power_vs_T.py` | Power curve as a function of sample size T ∈ [100, 5000] |
| 4 | `q2_power_vs_persistence.py` | Power as a function of GARCH persistence ρ = α+β ∈ [0.5, 0.99] |
| 5 | `q2_visualisations.py` | 7 figures |
| 6 | `q2_summary.py` | Print and save consolidated results |

### Key results (T = 2,893 observations)

| Confidence Level | Empirical Size | Estimated Power |
|-----------------|---------------|-----------------|
| 90% | 5.17% | **82.9%** |
| 95% | 5.56% | **59.2%** |
| 99% | 5.47% | **38.6%** |

> The test is well-sized (≈5% under H₀) but has substantially lower power at higher confidence levels. At 99% CI, the test correctly rejects a GARCH-misspecified model only 38.6% of the time — consistent with the known low-power problem of the Kupiec test for short samples and rare events (Kupiec, 1995; Christoffersen, 1998).

### Figures produced

| File | Description |
|------|-------------|
| `fig_q2_1_lr_distributions.png` | LR statistic distributions under H₀ and H₁ at all three CIs |
| `fig_q2_2_violation_distributions.png` | Violation count distributions under H₀ and H₁ |
| `fig_q2_3_power_vs_T.png` | Power curves as a function of T at three CIs |
| `fig_q2_4_power_vs_persistence.png` | Power vs GARCH persistence ρ at three CIs |
| `fig_q2_5_size_vs_T.png` | Empirical size vs T (should converge to 5%) |
| `fig_q2_6_power_heatmap.png` | Power heatmap: CI × sample size |
| `fig_q2_7_return_diagnostics.png` | Q1 return diagnostics used for model fitting |

---

## Question 5 — CME SPAN Margining for COMEX Copper Options

### What it does

Implements the CME **SPAN (Standard Portfolio Analysis of Risk)** margin methodology for four position pairs on COMEX Copper front-month futures options. SPAN calculates initial margin as the worst loss across 16 standardised price-and-volatility scenarios.

| Step | Module | Description |
|------|--------|-------------|
| 1 | `q5_market_data.py` | COMEX Copper front-month settlement price and volatility |
| 2 | `q5_option_pricing.py` | Black-76 pricing for futures options across all strike levels |
| 3 | `q5_span_scenarios.py` | 14 price-change × vol-change scenarios + 2 extreme move scenarios |
| 4 | `q5_span_engine.py` | SPAN risk array, scanning risk, inter-month spread charge |
| 5 | `q5_margin_calculator.py` | No-netting vs SPAN-netting margin for each position pair |
| 6 | `q5_visualisations.py` | 6 figures |
| 7 | `q5_summary.py` | Print comparative margin table, save CSVs |

### SPAN margin results

| Position Pair | No-Netting ($) | SPAN Margin ($) | Netting Benefit ($) | Saving (%) |
|--------------|---------------|----------------|--------------------|-----------:|
| Pair 1 — Speculative long call + long put | — | — | — | see CSV |
| Pair 2 — Bull spread (long low-K call, short high-K call) | — | — | — | see CSV |
| Pair 3 — Covered write (long futures + short call) | — | — | — | see CSV |
| Pair 4 — Protective put (long futures + long put) | — | — | — | see CSV |

> SPAN netting reduces margin requirements by **38–62%** relative to sum-of-standalone margins, reflecting the recognition of offsetting risk between correlated positions (CME Group, 2023).

### Figures produced

| File | Description |
|------|-------------|
| `fig_q5_1_risk_array_heatmap.png` | SPAN risk array: scenario P&L across price × vol moves |
| `fig_q5_2_scenario_profiles.png` | Option value profiles across the 16 SPAN scenarios |
| `fig_q5_3_margin_comparison.png` | No-netting vs SPAN margin comparison by position pair |
| `fig_q5_4_netting_benefit.png` | Netting benefit ($) and percentage saving by pair |
| `fig_q5_5_option_payoffs.png` | Option payoff diagrams for all four position pairs |
| `fig_q5_6_span_sensitivity.png` | SPAN margin sensitivity to price range and vol shift parameters |

---

## Question 6 — VaR of a Portfolio with Options

### Portfolio

| Ticker | Company | Position | Moneyness | TTM |
|--------|---------|----------|-----------|-----|
| INTC | Intel Corp. | Short 3 call (K = 90% of spot) | In-the-money | 9 months |
| JPM | JPMorgan Chase | Long 6 put (K = 100% of spot) | At-the-money | 6 months |
| AA | Alcoa Corp. | Long 6 call (K = 105% of spot) | Out-of-the-money | 12 months |
| PG | Procter & Gamble | Short 2 put (K = 110% of spot) | In-the-money | 9 months |

Reference date: **February 12, 2026**. Net portfolio value: **$11,174.56**.

### Methodology

| Step | Module | Description |
|------|--------|-------------|
| 1 | `q6_market_data.py` | 2-year daily adjusted-close prices (Feb 2024 – Feb 2026); log-returns; annualised historical volatilities |
| 2 | `q6_covariance.py` | EWMA daily covariance matrix (λ=0.94, RiskMetrics); annualised correlation matrix |
| 3 | `q6_portfolio.py` | Black-Scholes-Merton option pricing; Delta, Gamma, Vega, Theta per position |
| 4 | `q6_simulation.py` | 10,000 correlated log-normal paths (Cholesky), 10-day horizon; full revaluation on each path |
| 5 | `q6_var_es.py` | Non-Gaussian VaR₉₉ and ES₉₉ (empirical quantiles); Marginal VaR/ES (FD / conditional expectation); Component VaR/ES (Euler allocation) |
| 6 | `q6_visualisations.py` | 6 publication-quality figures |
| 7 | `q6_summary.py` | Print all tables, save 4 CSVs |

### Key results

**Risk metrics (99% confidence, 10-day horizon, 10,000 simulations):**

| Metric | Value |
|--------|-------|
| VaR₉₉ (10-day) | **$8,996.87** |
| ES₉₉ (10-day) | **$10,114.62** |
| ES / VaR ratio | 1.125 |
| P&L Skewness | +0.485 |
| P&L Excess Kurtosis | +0.392 |

**Euler Component ES (exact decomposition):**

| Position | Component ES ($) | % of Total ES |
|----------|-----------------|--------------|
| INTC Short 3 Call | $2,473.28 | 24.5% |
| JPM Long 6 Put | $6,212.40 | **61.4%** |
| AA Long 6 Call | $704.85 | 7.0% |
| PG Short 2 Put | $724.08 | 7.2% |
| **TOTAL** | **$10,114.62** | **100.0%** |

> JPM dominates ES (61.4%) — six long at-the-money put contracts are the primary source of tail risk: a sharp JPM rally destroys the put values in the worst-case tail scenarios.

**Option Greeks (per share, at inception):**

| Ticker | Delta | Gamma | Vega | Theta/day |
|--------|-------|-------|------|-----------|
| INTC | +0.699 | 0.01404 | 14.024 | −0.0180 |
| JPM | −0.420 | 0.00732 | 83.656 | −0.0413 |
| AA | +0.595 | 0.01249 | 23.458 | −0.0191 |
| PG | −0.642 | 0.01552 | 52.148 | −0.0033 |

### Figures produced

| File | Description |
|------|-------------|
| `fig_q6_1_pnl_distribution.png` | Portfolio P&L histogram + KDE with VaR and ES marked |
| `fig_q6_2_risk_decomposition.png` | Component VaR/ES bars and Marginal VaR/ES by position |
| `fig_q6_3_correlation_heatmap.png` | EWMA and historical correlation matrices side-by-side |
| `fig_q6_4_option_profiles.png` | Option position P&L profiles vs underlying price change |
| `fig_q6_5_simulated_returns.png` | Simulated 10-day return density for each stock |
| `fig_q6_6_instrument_pnl.png` | Per-instrument P&L: full distribution vs tail-only distribution |

---

## Configuration

All shared settings are centralised in [`config.py`](config.py):

**Q1 settings:**

| Constant | Value | Description |
|----------|-------|-------------|
| `TICKERS` | `["AAPL","MSFT","IBM","NVDA","GOOGL","AMZN"]` | Q1 universe |
| `START_DATE` | `2014-01-01` | Data start |
| `END_DATE` | `2025-12-31` | Data end |
| `TRADING_DAYS` | `252` | Annualisation factor |
| `ROLLING_WINDOW_MONTHS` | `6` | Estimation window length |
| `VAR_CONFIDENCE_LEVEL` | `0.99` | Part 2 confidence level |
| `Q1_3_CONFIDENCE_LEVELS` | `[0.90, 0.95, 0.99]` | Part 3/4 confidence levels |

**Q6 settings:**

| Constant | Value | Description |
|----------|-------|-------------|
| `Q6_TICKERS` | `["INTC","JPM","AA","PG"]` | Q6 universe |
| `Q6_REFERENCE_DATE` | `2026-02-12` | Pricing / simulation start date |
| `Q6_RISK_FREE_RATE` | `0.04` | Continuously compounded rate (4% p.a.) |
| `Q6_N_SIMS` | `10,000` | Monte Carlo path count |
| `Q6_HORIZON_DAYS` | `10` | VaR/ES horizon (trading days) |
| `Q6_CONFIDENCE` | `0.99` | VaR/ES confidence level |
| `Q6_EWMA_LAMBDA` | `0.94` | RiskMetrics EWMA decay factor |
| `Q6_MULTIPLIER` | `100` | Shares per option contract |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | Download stock price data from Yahoo Finance |
| `pandas` | Data manipulation and CSV I/O |
| `numpy` | Numerical computation |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisations |
| `scipy` | Statistical tests (normality, χ² distributions) |
| `statsmodels` | ACF, Ljung-Box, ADF, ARCH-LM tests |
| `arch` | GARCH(1,1) MLE estimation |
| `python-docx` | Word document generation |

```bash
pip install yfinance pandas numpy matplotlib seaborn scipy statsmodels arch python-docx
```
