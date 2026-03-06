# SMM272 Risk Analysis Coursework 2025-2026

Statistical analysis of an equally weighted portfolio of six US tech stocks using Python.

## Stocks

| Ticker | Company |
|--------|---------|
| AAPL | Apple |
| MSFT | Microsoft |
| IBM | IBM |
| NVDA | Nvidia |
| GOOGL | Alphabet |
| AMZN | Amazon |

**Sample period:** 1 January 2014 – 31 December 2025

---

## Project Structure

```
├── config.py                        # Shared constants (tickers, dates, paths)
├── Q1/                              # Question 1 — Statistical Analysis
│   ├── q1_statistical_analysis.py   # Master script (runs Steps 1-7)
│   ├── q1_download_prices.py        # Step 1: Download adjusted closing prices
│   ├── q1_build_portfolio.py        # Step 2: Build equally weighted portfolio
│   ├── q1_descriptive_stats.py      # Step 3: Descriptive statistics
│   ├── q1_normality_tests.py        # Step 4: Normality tests
│   ├── q1_autocorrelation.py        # Step 5: Autocorrelation & correlation matrix
│   ├── q1_visualisations.py         # Step 6: All visualisations (9 figures)
│   ├── q1_summary.py               # Step 7: Summary of findings
│   └── output/                      # Generated CSVs and figures
├── .gitignore
└── README.md
```

## Question 1 — Statistical Analysis of Portfolio Returns

### What it does

1. **Download prices** — fetches adjusted closing prices from Yahoo Finance via `yfinance`.
2. **Build portfolio** — computes daily log returns and constructs an equally weighted (1/6) portfolio.
3. **Descriptive statistics** — mean, median, standard deviation, skewness, excess kurtosis, VaR, CVaR, Sharpe ratio (annualised).
4. **Normality tests** — Jarque-Bera, Shapiro-Wilk, D'Agostino-Pearson K², Kolmogorov-Smirnov.
5. **Autocorrelation analysis** — Ljung-Box tests on raw and squared returns; pairwise correlation matrix.
6. **Visualisations** — 9 figures including normalised prices, cumulative returns, histogram vs fitted normal, Q-Q plot, box plots, correlation heatmap, ACF plots, and rolling statistics.
7. **Summary** — consolidated narrative of key findings.

### How to run

Each module can be executed independently or all at once via the master script:

```bash
# Run everything
python Q1/q1_statistical_analysis.py

# Or run individual steps
python Q1/q1_download_prices.py
python Q1/q1_normality_tests.py
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | Download stock price data |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisations |
| `scipy` | Statistical tests |
| `statsmodels` | ACF plots & Ljung-Box tests |

Install all dependencies:

```bash
pip install yfinance pandas numpy matplotlib seaborn scipy statsmodels
```

---

## Output

All generated figures and CSV files are saved to `Q1/output/`.
