"""
Q1 : All visualisations (9 figures).
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves files, never opens windows
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from config import TICKERS, TRADING_DAYS, Q1_1_OUTPUT_DIR
from q1_1_build_portfolio import build_portfolio
from logger import setup_run_logger, get_logger, log_start, log_end

logger = get_logger("q1_1_visualisations")


def generate_visualisations(prices=None, log_returns=None,
                            portfolio_returns=None, corr_matrix=None):
    """Generate and save all Q1 figures."""
    log_start(logger, "q1_1_visualisations.py")
    if prices is None or log_returns is None or portfolio_returns is None:
        prices, log_returns, portfolio_returns = build_portfolio()
    if corr_matrix is None:
        corr_matrix = log_returns.corr()

    sns.set_style("whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10})

    # ── FIGURE 1: Normalised price paths ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    normalised = prices / prices.iloc[0] * 100
    for ticker in TICKERS:
        ax.plot(normalised.index, normalised[ticker], label=ticker, linewidth=0.9)
    ax.set_title("Normalised Adjusted Closing Prices (Base = 100)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalised Price")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(Q1_1_OUTPUT_DIR, "fig1_normalised_prices.png"))
    plt.close()

    # ── FIGURE 2: Portfolio cumulative returns ─────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    cum_returns = portfolio_returns.cumsum().apply(np.exp) - 1
    ax.plot(cum_returns.index, cum_returns * 100, color="navy", linewidth=0.9)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_title("Equally Weighted Portfolio — Cumulative Returns", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(Q1_1_OUTPUT_DIR, "fig2_cumulative_returns.png"))
    plt.close()

    # ── FIGURE 3: Daily portfolio returns ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(portfolio_returns.index, portfolio_returns * 100, width=1,
           color="steelblue", edgecolor="steelblue", linewidth=0.2, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_title("Equally Weighted Portfolio — Daily Log Returns", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(Q1_1_OUTPUT_DIR, "fig3_daily_returns.png"))
    plt.close()

    # ── FIGURE 4: Histogram with fitted Normal & KDE ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(portfolio_returns, bins=100, density=True, alpha=0.55,
            color="steelblue", edgecolor="white", label="Empirical")

    x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 500)
    mu, sigma = portfolio_returns.mean(), portfolio_returns.std()
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=1.5,
            label=f"Normal (μ={mu:.5f}, σ={sigma:.5f})")
    portfolio_returns.plot.kde(ax=ax, color="darkgreen", linewidth=1.5, label="KDE")

    ax.set_title("Portfolio Returns Distribution vs Normal Distribution", fontsize=13)
    ax.set_xlabel("Daily Log Return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(Q1_1_OUTPUT_DIR, "fig4_histogram_normal_kde.png"))
    plt.close()

    # ── FIGURE 5: QQ-Plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    stats.probplot(portfolio_returns, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot — Portfolio Returns vs Normal Distribution", fontsize=13)
    ax.get_lines()[0].set(markerfacecolor="steelblue", markeredgecolor="steelblue",
                           markersize=2.5)
    ax.get_lines()[1].set(color="red", linewidth=1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(Q1_1_OUTPUT_DIR, "fig5_qq_plot.png"))
    plt.close()

    # ── FIGURE 6: Box plots ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    combined = pd.concat([log_returns,
                          portfolio_returns.rename("EW_Portfolio")], axis=1)
    combined.boxplot(ax=ax, showfliers=True, patch_artist=True,
                     boxprops=dict(facecolor="lightblue", color="navy"),
                     medianprops=dict(color="red"),
                     flierprops=dict(marker="o", markersize=2, alpha=0.3))
    ax.set_title("Box Plots — Daily Log Returns by Asset & Portfolio", fontsize=13)
    ax.set_ylabel("Daily Log Return")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(Q1_1_OUTPUT_DIR, "fig6_boxplots.png"))
    plt.close()

    # ── FIGURE 7: Correlation heatmap ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdYlBu_r",
                vmin=-1, vmax=1, center=0, square=True,
                mask=mask, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix — Daily Log Returns", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(Q1_1_OUTPUT_DIR, "fig7_correlation_heatmap.png"))
    plt.close()

    # ── FIGURE 8: ACF of returns and squared returns ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(portfolio_returns, lags=40, ax=axes[0],
             title="ACF — Portfolio Returns", alpha=0.05, zero=False)
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")

    plot_acf(portfolio_returns ** 2, lags=40, ax=axes[1],
             title="ACF — Squared Portfolio Returns", alpha=0.05, zero=False)
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Autocorrelation")
    plt.tight_layout()
    plt.savefig(os.path.join(Q1_1_OUTPUT_DIR, "fig8_acf_plots.png"))
    plt.close()

    # ── FIGURE 9: Rolling statistics (252-day) ────────────────────────────
    window = TRADING_DAYS
    rolling_mean = portfolio_returns.rolling(window).mean() * TRADING_DAYS
    rolling_std = portfolio_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    rolling_sharpe = rolling_mean / rolling_std

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    axes[0].plot(rolling_mean.index, rolling_mean, color="navy", linewidth=0.9)
    axes[0].axhline(0, color="grey", linestyle="--", linewidth=0.5)
    axes[0].set_title("252-Day Rolling Annualised\nMean Return", fontsize=11)
    axes[0].set_ylabel("Annualised Return")

    axes[1].plot(rolling_std.index, rolling_std, color="darkred", linewidth=0.9)
    axes[1].set_title("252-Day Rolling Annualised\nVolatility", fontsize=11)
    axes[1].set_ylabel("Annualised Volatility")

    axes[2].plot(rolling_sharpe.index, rolling_sharpe, color="darkgreen", linewidth=0.9)
    axes[2].axhline(0, color="grey", linestyle="--", linewidth=0.5)
    axes[2].set_title("252-Day Rolling Sharpe\nRatio (Rf = 0)", fontsize=11)
    axes[2].set_ylabel("Sharpe Ratio")

    for ax in axes:
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(Q1_1_OUTPUT_DIR, "fig9_rolling_statistics.png"))
    plt.close()

    logger.info(f"All 9 figures saved to: {Q1_1_OUTPUT_DIR}")
    log_end(logger, "q1_1_visualisations.py")


if __name__ == "__main__":
    setup_run_logger("smm272_q1_visualisations")
    generate_visualisations()
