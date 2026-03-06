# SMM272 Risk Analysis Coursework 2025-2026

Quantitative risk analysis of an equally weighted portfolio of six US technology stocks, implemented in Python. The project covers distributional analysis, rolling Value-at-Risk (VaR) and Expected Shortfall (ES) estimation across four models, and formal VaR violations backtesting using Kupiec and Christoffersen tests at multiple confidence levels.

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

## Project Structure

```
SMM272_Risk_Analysis/
├── config.py                          # Shared constants: tickers, dates, paths, VaR metadata
├── logger.py                          # Shared logging utilities (setup_run_logger, get_logger)
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
│   ├── ── Part 3: VaR Violations Backtesting ──
│   ├── q1_3_var_violations.py         # Part 3 orchestrator
│   ├── q1_3_rolling.py                # Multi-level rolling VaR (90% / 95% / 99%)
│   ├── q1_3_kupiec.py                 # Kupiec (1995) POF test
│   ├── q1_3_christoffersen.py         # Christoffersen (1998) independence + CC test
│   ├── q1_3_backtests.py              # Run all tests for all model × CI combinations
│   ├── q1_3_logging.py                # Structured logging of violation results
│   ├── q1_3_plots.py                  # Three diagnostic figures
│   └── output_q1_3/                   # Part 3 outputs
│       ├── rolling_var_all_levels.csv
│       ├── violations_summary.csv
│       ├── kupiec_results.csv
│       ├── christoffersen_results.csv
│       ├── violation_series.csv
│       ├── fig_violations_barchart.png
│       ├── fig_violations_heatmap.png
│       └── fig_violation_timeseries.png
│
├── logs/                              # Timestamped run logs
├── .gitignore
└── README.md
```

---

## Running the Code

### Prerequisites

```bash
pip install yfinance pandas numpy matplotlib seaborn scipy statsmodels arch
```

> **`arch`** is required for GARCH(1,1) estimation (Part 2 & 3). If unavailable, GARCH automatically falls back to the Parametric Normal model.

### Unified runner — `q1_main.py`

All Q1 sub-parts are run via `q1_main.py` from the `Q1/` directory:

```bash
cd Q1

# Run all parts (1, 2, 3) in sequence
python q1_main.py

# Run a single part
python q1_main.py --parts q1_1
python q1_main.py --parts q1_2
python q1_main.py --parts q1_3

# Run multiple specific parts
python q1_main.py --parts q1_1 q1_2

# Run a range
python q1_main.py --from q1_1 --to q1_3

# Preview without executing
python q1_main.py --dry-run
python q1_main.py --parts q1_3 --dry-run
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

## Question 1 — Part 3: VaR Violations Backtesting

### What it does

Extends Part 2 to three confidence levels (90%, 95%, 99%) and applies two formal statistical backtests to every model × confidence level combination (12 pairs total):

| Test | Module | What it tests |
|------|--------|---------------|
| **Kupiec (1995) POF** | `q1_3_kupiec.py` | Unconditional coverage: is the observed violation rate statistically equal to the nominal rate? LR ~ χ²(1) |
| **Christoffersen (1998)** | `q1_3_christoffersen.py` | Independence: are violations serially independent? Detects clustering. LR_ind ~ χ²(1) |
| **Joint CC test** | `q1_3_backtests.py` | Combined test: LR_cc = LR_uc + LR_ind ~ χ²(2) |

### Backtest results summary (99% CI, 5% significance)

| Method | Violations | Kupiec reject? | Christoffersen reject? | CC reject? |
|--------|-----------|---------------|----------------------|------------|
| Historical Simulation | 49 | No | —* | —* |
| Parametric Student-t | 63 | Yes | — | Yes |
| GARCH(1,1) Normal | 82 | Yes | Yes | Yes |
| Parametric Normal | 87 | Yes | — | Yes |

*Depends on the exact clustering pattern; see `christoffersen_results.csv` for precise statistics.

### Figures produced

| File | Description |
|------|-------------|
| `fig_violations_barchart.png` | Grouped bar chart: observed vs nominal violation rates at all 3 CIs |
| `fig_violations_heatmap.png` | Heatmap: observed (blue) and expected (grey) violation counts, 4×3 grid |
| `fig_violation_timeseries.png` | Cumulative violations over time vs expected line, one panel per CI level |

---

## Configuration

All shared settings are centralised in [`config.py`](config.py):

| Constant | Value | Description |
|----------|-------|-------------|
| `TICKERS` | `["AAPL","MSFT","IBM","NVDA","GOOGL","AMZN"]` | Universe |
| `START_DATE` | `2014-01-01` | Data start |
| `END_DATE` | `2025-12-31` | Data end |
| `TRADING_DAYS` | `252` | Annualisation factor |
| `ROLLING_WINDOW_MONTHS` | `6` | Estimation window length |
| `ROLLING_START_DATE` | `2014-07-01` | First forecast date |
| `VAR_CONFIDENCE_LEVEL` | `0.99` | Part 2 confidence level |
| `Q1_3_CONFIDENCE_LEVELS` | `[0.90, 0.95, 0.99]` | Part 3 confidence levels |
| `VAR_METHODS_META` | tag / label / colour tuples | Shared method metadata |

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
