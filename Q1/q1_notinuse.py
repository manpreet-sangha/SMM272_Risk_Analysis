"""
Q1: VaR Modelling — Full Implementation
========================================
Equally-weighted portfolio of AAPL, MSFT, IBM, NVDA, GOOGL, AMZN
January 2014 – December 2025.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import yfinance as yf
from scipy import stats
from scipy.stats import norm, t as student_t, jarque_bera, shapiro, chi2
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────
TICKERS     = ["AAPL", "MSFT", "IBM", "NVDA", "GOOGL", "AMZN"]
START_DATE  = "2014-01-01"
END_DATE    = "2025-12-31"
ROLL_START  = "2014-07-01"
WINDOW      = 126          # ~6 months
CONF_LEVELS = [0.90, 0.95, 0.99]
ALPHA       = 0.05
TRADING_DAYS = 252
OUTPUT_DIR  = "output_q1/"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# PART 1 — DATA & STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════

def download_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"][tickers]
    prices.dropna(how="all", inplace=True)
    return prices


def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def build_ew_portfolio(log_returns):
    return log_returns.mean(axis=1).rename("EW_Portfolio")


def descriptive_statistics(portfolio, log_returns):
    combined = pd.concat([log_returns, portfolio], axis=1)
    rows = {}
    for col in combined.columns:
        s = combined[col].dropna()
        jb_stat, jb_p   = jarque_bera(s)
        sw_stat, sw_p   = shapiro(s)
        rows[col] = {
            "Mean (%)":        s.mean() * 100,
            "Std (%)":         s.std(ddof=1) * 100,
            "Skewness":        s.skew(),
            "Exc. Kurtosis":   s.kurtosis(),
            "Min (%)":         s.min() * 100,
            "Max (%)":         s.max() * 100,
            "JB stat":         jb_stat,
            "JB p-value":      jb_p,
            "SW stat":         sw_stat,
            "SW p-value":      sw_p,
        }
    return pd.DataFrame(rows).T


def annualised_risk_return(portfolio, log_returns):
    combined = pd.concat([log_returns, portfolio], axis=1)
    rows = {}
    for col in combined.columns:
        s = combined[col].dropna()
        ann_ret = s.mean() * TRADING_DAYS
        ann_std = s.std(ddof=1) * np.sqrt(TRADING_DAYS)
        sharpe  = ann_ret / ann_std
        downside = s[s < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)
        sortino  = ann_ret / downside if downside > 0 else np.nan
        rows[col] = {"Return": ann_ret, "Std": ann_std,
                     "Sharpe": sharpe, "Sortino": sortino}
    return pd.DataFrame(rows).T


def plot_prices_and_returns(prices, portfolio):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    norm_prices = prices / prices.iloc[0]
    norm_prices.plot(ax=axes[0], linewidth=0.8)
    axes[0].set_title("Normalised Adjusted Closing Prices (base = 1)")
    axes[0].set_ylabel("Normalised Price")
    axes[0].legend(fontsize=7)

    portfolio.plot(ax=axes[1], color="black", linewidth=0.6, alpha=0.8)
    axes[1].set_title("EW Portfolio — Daily Log Returns")
    axes[1].set_ylabel("Log Return")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig1_prices_returns.png", dpi=150)
    plt.close()


def plot_distributional_properties(portfolio):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mu, sigma = portfolio.mean(), portfolio.std()
    x = np.linspace(portfolio.min(), portfolio.max(), 300)

    axes[0].hist(portfolio, bins=100, density=True, alpha=0.5,
                 color="steelblue", label="Empirical")
    axes[0].plot(x, norm.pdf(x, mu, sigma), "r-", label=f"Normal(μ={mu:.4f}, σ={sigma:.4f})")
    portfolio.plot.kde(ax=axes[0], color="green", label="KDE")
    axes[0].set_title("Return Distribution vs Normal")
    axes[0].legend()

    stats.probplot(portfolio, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q–Q Plot")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig2_distributions.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(log_returns):
    corr = log_returns.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Pearson Correlation Matrix — Daily Log Returns")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig3_correlation.png", dpi=150)
    plt.close()
    return corr


def plot_acf_returns(portfolio, lags=40):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(portfolio, lags=lags, ax=axes[0], title="ACF — Portfolio Returns")
    plot_acf(portfolio**2, lags=lags, ax=axes[1], title="ACF — Squared Returns")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig4_acf.png", dpi=150)
    plt.close()

    lb = acorr_ljungbox(portfolio**2, lags=20, return_df=True)
    print("Ljung-Box on squared returns (lags 1-20):")
    print(lb)
    return lb


def plot_rolling_stats(log_returns, portfolio, window=63):
    combined = pd.concat([log_returns, portfolio], axis=1)
    roll_mean = combined.rolling(window).mean() * TRADING_DAYS
    roll_std  = combined.rolling(window).std()  * np.sqrt(TRADING_DAYS)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    roll_mean.plot(ax=axes[0], linewidth=0.7)
    axes[0].set_title(f"{window}-Day Rolling Annualised Mean Return")
    roll_std.plot(ax=axes[1], linewidth=0.7)
    axes[1].set_title(f"{window}-Day Rolling Annualised Volatility")
    (roll_mean / roll_std).plot(ax=axes[2], linewidth=0.7)
    axes[2].set_title(f"{window}-Day Rolling Sharpe Ratio (rf=0)")
    for ax in axes:
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig5_rolling_stats.png", dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# PART 2 — ROLLING VAR & ES (4 METHODS)
# ═══════════════════════════════════════════════════════════════

def var_es_historical_simulation(window_returns, conf):
    sorted_r = np.sort(window_returns)
    n = len(sorted_r)
    if n == 0:
        return np.nan, np.nan
    k = int(np.floor((1 - conf) * n))
    k = max(k, 1)
    k = min(k, n)
    var = -sorted_r[k - 1]
    es  = -sorted_r[:k].mean()
    return var, es


def var_es_parametric_normal(window_returns, conf):
    mu    = window_returns.mean()
    sigma = window_returns.std(ddof=1)
    z     = norm.ppf(1 - conf)
    var   = -(mu + z * sigma)
    es    = -(mu - sigma * norm.pdf(z) / (1 - conf))
    return var, es


def var_es_parametric_student_t(window_returns, conf):
    try:
        nu, loc, scale = student_t.fit(window_returns, floc=0)
        nu = max(nu, 2.01)   # ensure ES is defined (nu > 2)
    except Exception:
        return var_es_parametric_normal(window_returns, conf) + (np.nan,)

    q_alpha = student_t.ppf(1 - conf, df=nu, loc=loc, scale=scale)
    var = -q_alpha

    # Analytical ES for scaled Student-t
    t_q   = student_t.ppf(1 - conf, df=nu)
    es_std = (student_t.pdf(t_q, nu) / (1 - conf)) * (nu + t_q**2) / (nu - 1)
    es    = -(loc - scale * es_std)
    return var, es, nu


def var_es_garch(window_returns, conf):
    var, es, _ = _fit_garch(window_returns, conf)
    return var, es


def rolling_var_es(portfolio, window, conf, roll_start):
    """
    Master rolling loop — returns one row per forecast day.
    GARCH is re-fitted every GARCH_REFIT_FREQ days and the previous
    parameters are used as starting values (warm start) to dramatically
    speed up convergence on intermediate steps.
    """
    GARCH_REFIT_FREQ = 5   # refit GARCH every N days; increase to 10 for speed

    all_dates    = portfolio.index
    start_idx    = all_dates.get_loc(all_dates[all_dates >= roll_start][0])
    forecast_idx = list(range(start_idx, len(all_dates) - 1))

    # Filter to valid indices upfront
    valid_idx = [i for i in forecast_idx
                 if i - window >= 0 and i + 1 < len(all_dates)]

    total     = len(valid_idx)
    records   = []
    last_garch_params = None   # warm-start cache
    steps_since_refit = 0

    print(f"  Rolling VaR ({int(conf*100)}% CI): {total} forecasts to compute...")

    for count, i in enumerate(valid_idx):
        win = portfolio.iloc[i - window: i].values
        win = win[~np.isnan(win)]
        if len(win) < 20:
            continue

        next_r = portfolio.iloc[i + 1]
        date   = all_dates[i + 1]

        hs_v,  hs_e        = var_es_historical_simulation(win, conf)
        nm_v,  nm_e        = var_es_parametric_normal(win, conf)
        st_v,  st_e, st_nu = var_es_parametric_student_t(win, conf)

        # GARCH: only refit every GARCH_REFIT_FREQ steps
        if steps_since_refit == 0 or last_garch_params is None:
            gc_v, gc_e, last_garch_params = _fit_garch(
                win, conf, starting_vals=last_garch_params)
            steps_since_refit = GARCH_REFIT_FREQ
        else:
            gc_v, gc_e, _ = _fit_garch(
                win, conf, starting_vals=last_garch_params)
            steps_since_refit -= 1

        records.append({
            "date":       date,
            "actual":     next_r,
            "HS_VaR":     hs_v,  "HS_ES":     hs_e,
            "Normal_VaR": nm_v,  "Normal_ES": nm_e,
            "StudT_VaR":  st_v,  "StudT_ES":  st_e,  "StudT_nu": st_nu,
            "GARCH_VaR":  gc_v,  "GARCH_ES":  gc_e,
        })

        # Progress every 250 steps
        if (count + 1) % 250 == 0 or (count + 1) == total:
            print(f"    {count + 1}/{total} ({(count+1)/total*100:.0f}%)", flush=True)

    df = pd.DataFrame(records).set_index("date")
    df.to_csv(OUTPUT_DIR + f"rolling_var_conf{int(conf*100)}.csv")
    return df


def _fit_garch(window_returns, conf, starting_vals=None):
    """
    Fit GARCH(1,1)-Normal and return (VaR, ES, fitted_params).
    Accepts optional starting_vals dict for warm-starting convergence.
    Falls back to parametric Normal on failure.
    """
    try:
        am  = arch_model(window_returns * 100, vol="Garch", p=1, q=1,
                         mean="Constant", dist="Normal")
        fit_kwargs = {"disp": "off", "show_warning": False,
                      "options": {"maxiter": 200}}
        if starting_vals is not None:
            fit_kwargs["starting_values"] = starting_vals
        res = am.fit(**fit_kwargs)

        fcast      = res.forecast(horizon=1, reindex=False)
        sigma_next = np.sqrt(fcast.variance.iloc[-1, 0]) / 100
        mu_next    = res.params["mu"] / 100
        fitted_params = res.params.values   # cache for next warm start

    except Exception:
        var, es = var_es_parametric_normal(window_returns, conf)
        return var, es, starting_vals   # keep old params on failure

    z   = norm.ppf(1 - conf)
    var = -(mu_next + z * sigma_next)
    es  = -(mu_next - sigma_next * norm.pdf(z) / (1 - conf))
    return var, es, fitted_params


def plot_rolling_var(results, conf=0.99):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    cols_var = ["HS_VaR", "Normal_VaR", "StudT_VaR", "GARCH_VaR"]
    cols_es  = ["HS_ES",  "Normal_ES",  "StudT_ES",  "GARCH_ES"]
    labels   = ["Historical Simulation", "Parametric Normal",
                "Parametric Student-t", "GARCH(1,1) Normal"]

    for col, lbl in zip(cols_var, labels):
        axes[0].plot(-results[col], label=lbl, linewidth=0.7, alpha=0.8)
    axes[0].plot(results["actual"], color="black", linewidth=0.4,
                 alpha=0.5, label="Daily Return")
    axes[0].set_title(f"Rolling {int(conf*100)}% VaR")
    axes[0].legend(fontsize=7)

    for col, lbl in zip(cols_es, labels):
        axes[1].plot(-results[col], label=lbl, linewidth=0.7, alpha=0.8)
    axes[1].plot(results["actual"], color="black", linewidth=0.4,
                 alpha=0.5, label="Daily Return")
    axes[1].set_title(f"Rolling {int(conf*100)}% ES")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + f"fig6_rolling_var_es_{int(conf*100)}.png", dpi=150)
    plt.close()


def plot_breach_scatter(results, conf=0.99):
    models = {
        "Historical Simulation": "HS_VaR",
        "Parametric Normal":     "Normal_VaR",
        "Parametric Student-t":  "StudT_VaR",
        "GARCH(1,1) Normal":     "GARCH_VaR",
    }
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    for ax, (name, var_col) in zip(axes, models.items()):
        breach = results["actual"] < -results[var_col]
        ax.scatter(results.index[~breach], results["actual"][~breach],
                   s=1, color="steelblue", alpha=0.4, label="No breach")
        ax.scatter(results.index[breach], results["actual"][breach],
                   s=4, color="red", alpha=0.8, label="Breach")
        ax.plot(results.index, -results[var_col], color="orange",
                linewidth=0.6, label="VaR")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=6)
    plt.suptitle(f"Breach Scatter — {int(conf*100)}% VaR", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + f"fig7_breach_scatter_{int(conf*100)}.png", dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# PART 3 — VIOLATION COUNTS
# ═══════════════════════════════════════════════════════════════

def hit_sequence(actual, var_forecast):
    """I_t = 1 if actual < -VaR (loss exceeds VaR), else 0."""
    return (actual < -var_forecast).astype(int)


def violation_summary(portfolio, window, conf_levels, roll_start):
    model_map = {
        "HS":        "HS_VaR",
        "Normal":    "Normal_VaR",
        "Student-t": "StudT_VaR",
        "GARCH":     "GARCH_VaR",
    }
    rows = []
    results_cache = {}
    for conf in conf_levels:
        res = rolling_var_es(portfolio, window, conf, roll_start)
        results_cache[conf] = res
        T = len(res)
        for model, var_col in model_map.items():
            hits = hit_sequence(res["actual"], res[var_col])
            k    = hits.sum()
            rows.append({
                "Model": model, "CI": conf,
                "k":     k,
                "E[k]":  round((1 - conf) * T, 1),
                "T":     T,
                "Obs%":  round(k / T * 100, 2),
                "Nom%":  round((1 - conf) * 100, 2),
                "Exc pp": round((k / T - (1 - conf)) * 100, 2),
            })
    return pd.DataFrame(rows), results_cache


def plot_violation_rates(summary_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = summary_df["Model"].unique()
    x = np.arange(len(models))
    width = 0.15

    for idx, (conf, ax) in enumerate(zip([0.90, 0.95, 0.99], axes)):
        sub = summary_df[summary_df["CI"] == conf]
        expected = sub["Nom%"].iloc[0]
        bars = ax.bar(x, sub["Obs%"].values, color=["steelblue","tomato","green","gold"])
        ax.axhline(expected, color="black", linestyle="--", linewidth=1,
                   label=f"Expected {expected}%")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(f"Violation Rates — {int(conf*100)}% CI")
        ax.set_ylabel("Violation Rate (%)")
        ax.legend(fontsize=8)
        for bar, val in zip(bars, sub["Obs%"].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig9_violation_rates.png", dpi=150)
    plt.close()


def plot_cumulative_violations(results_cache, conf_levels):
    model_map = {"HS": "HS_VaR", "Normal": "Normal_VaR",
                 "Student-t": "StudT_VaR", "GARCH": "GARCH_VaR"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["steelblue", "tomato", "green", "gold"]

    for ax, conf in zip(axes, conf_levels):
        res = results_cache[conf]
        T   = len(res)
        expected_line = np.linspace(0, (1 - conf) * T, T)
        ax.plot(res.index, expected_line, "k--", linewidth=1, label="Expected")

        for (model, var_col), color in zip(model_map.items(), colors):
            hits     = hit_sequence(res["actual"], res[var_col])
            cum_hits = hits.cumsum()
            ax.plot(res.index, cum_hits, color=color, linewidth=0.8,
                    label=model, alpha=0.9)

        ax.set_title(f"Cumulative Violations — {int(conf*100)}% CI")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig10_cumulative_violations.png", dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# PART 4 — BACKTESTING
# ═══════════════════════════════════════════════════════════════

def kupiec_test(hits, conf):
    T     = len(hits)
    k     = int(hits.sum())
    p     = 1 - conf
    p_hat = k / T if T > 0 else p
    if p_hat == 0:
        LR_uc = -2 * T * np.log(1 - p)
    elif p_hat == 1:
        LR_uc = -2 * T * np.log(p)
    else:
        LR_uc = -2 * (k * np.log(p / p_hat) + (T - k) * np.log((1 - p) / (1 - p_hat)))
    p_val = 1 - chi2.cdf(LR_uc, df=1)
    return {"LR_uc": LR_uc, "p_uc": p_val, "Rej_uc": p_val < ALPHA, "k": k, "T": T}


def christoffersen_independence_test(hits):
    h   = hits.values
    n00 = np.sum((h[:-1] == 0) & (h[1:] == 0))
    n01 = np.sum((h[:-1] == 0) & (h[1:] == 1))
    n10 = np.sum((h[:-1] == 1) & (h[1:] == 0))
    n11 = np.sum((h[:-1] == 1) & (h[1:] == 1))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi   = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def safe_log(x):
        return np.log(x) if x > 0 else 0.0

    L_A1 = (n00 * safe_log(1 - pi01) + n01 * safe_log(pi01) +
             n10 * safe_log(1 - pi11) + n11 * safe_log(pi11))
    L_A0 = ((n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi))
    LR_ind = -2 * (L_A0 - L_A1)
    LR_ind = max(LR_ind, 0)
    p_val  = 1 - chi2.cdf(LR_ind, df=1)
    return {"LR_ind": LR_ind, "p_ind": p_val, "Rej_ind": p_val < ALPHA}


def christoffersen_cc_test(hits, conf):
    kup = kupiec_test(hits, conf)
    ind = christoffersen_independence_test(hits)
    LR_cc = kup["LR_uc"] + ind["LR_ind"]
    p_val = 1 - chi2.cdf(LR_cc, df=2)
    return {**kup, **ind,
            "LR_cc": LR_cc, "p_cc": p_val, "Rej_cc": p_val < ALPHA}


def duration_test(hits):
    """
    Christoffersen & Pelletier (2004).
    Tests whether inter-violation durations are geometrically distributed (memoryless).
    H1: Weibull alternative.
    """
    hit_idx   = np.where(hits.values == 1)[0]
    if len(hit_idx) < 3:
        return {"LR_dur": np.nan, "p_dur": np.nan, "Rej_dur": False}

    durations = np.diff(hit_idx).astype(float)
    n         = len(durations)

    # Geometric H0 MLE: p_hat = 1 / mean_duration
    mean_dur = durations.mean()
    p_hat    = 1.0 / mean_dur if mean_dur > 0 else 1e-9

    def geom_loglik(p, d):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.sum(np.log(p) + (d - 1) * np.log(1 - p))

    # Weibull H1 MLE via scipy
    from scipy.stats import weibull_min
    try:
        a_hat, loc_hat, b_hat = weibull_min.fit(durations, floc=0)
        weib_loglik = np.sum(weibull_min.logpdf(durations, a_hat, loc=0, scale=b_hat))
    except Exception:
        return {"LR_dur": np.nan, "p_dur": np.nan, "Rej_dur": False}

    geom_loglik_val = geom_loglik(p_hat, durations)
    LR_dur = -2 * (geom_loglik_val - weib_loglik)
    LR_dur = max(LR_dur, 0)
    p_val  = 1 - chi2.cdf(LR_dur, df=2)
    return {"LR_dur": LR_dur, "p_dur": p_val, "Rej_dur": p_val < ALPHA}


def dq_test(hits, var_forecast, conf, lags=4):
    """
    Engle & Manganelli (2004) Dynamic Quantile test.
    Regresses de-meaned hits on lagged hits and VaR forecast.
    DQ ~ chi2(lags + 1) under H0.
    """
    p        = 1 - conf
    demeaned = hits - p
    df       = pd.DataFrame({"demeaned": demeaned, "VaR": var_forecast})
    for lag in range(1, lags + 1):
        df[f"lag{lag}"] = demeaned.shift(lag)
    df.dropna(inplace=True)

    y  = df["demeaned"].values
    X  = df[[f"lag{i}" for i in range(1, lags + 1)] + ["VaR"]].values
    X  = np.column_stack([np.ones(len(X)), X])   # add intercept

    try:
        beta  = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat
        # Wald statistic: b^T (X^T X)^{-1} b * n / sigma^2
        sigma2 = np.var(resid, ddof=X.shape[1])
        XtX_inv = np.linalg.inv(X.T @ X)
        DQ = (beta[1:].T @ np.linalg.inv(XtX_inv[1:, 1:]) @ beta[1:]) / sigma2
        DQ = float(abs(DQ))
    except np.linalg.LinAlgError:
        return {"DQ": np.nan, "p_dq": np.nan, "Rej_dq": False}

    p_val = 1 - chi2.cdf(DQ, df=lags + 1)
    return {"DQ": DQ, "p_dq": p_val, "Rej_dq": p_val < ALPHA}


def pit_distributional_test(actual, results_cache, conf_levels):
    """
    PIT (Probability Integral Transform) tests for all four models at all conf levels.

    For a correctly-specified model, z_t = F_t(r_{t+1}) ~ i.i.d. Uniform(0,1).
    Equivalently, x_t = Phi^{-1}(z_t) ~ i.i.d. N(0,1).

    Tests applied per model × conf level:
      1. KS test       : z_t ~ Uniform(0,1)  [Kolmogorov-Smirnov]
      2. AD test       : z_t ~ Uniform(0,1)  [Anderson-Darling, more powerful in tails]
      3. Berkowitz LR  : x_t = Phi^{-1}(z_t) tested via LR for N(0,1) vs N(mu,sig^2)
                         with AR(1) serial dependence, df=3, chi2 distribution
      4. LB on z       : Ljung-Box(10) on z_t  — tests level autocorrelation
      5. LB on (z-0.5)^2: Ljung-Box(10) on squared centred PITs — tests vol clustering

    PITs are computed using the rolling window parameters already stored in results_cache.
    HS uses the empirical CDF of the window; GARCH uses Normal CDF with GARCH sigma_next.
    """
    from scipy.stats import kstest, anderson, chi2 as _chi2

    def _berkowitz_lr(z_arr):
        """
        Berkowitz (2001) LR test.
        Transform z -> x = Phi^{-1}(z). Under H0: x ~ i.i.d. N(0,1).
        Fit AR(1)-Normal: x_t = mu + rho*(x_{t-1} - mu) + eps, eps ~ N(0,sig^2).
        LR = -2*(logL_H0 - logL_H1) ~ chi2(3).
        """
        z_safe = np.clip(z_arr, 1e-7, 1 - 1e-7)
        x = norm.ppf(z_safe)
        n = len(x)
        if n < 10:
            return np.nan, np.nan

        # Log-likelihood under H0: N(0,1) i.i.d.
        ll_h0 = np.sum(norm.logpdf(x))

        # Log-likelihood under H1: AR(1) with N(mu, sig^2)
        # MLE via OLS on x_t = mu(1-rho) + rho*x_{t-1} + eps
        x1 = x[:-1]   # x_{t-1}
        x2 = x[1:]    # x_t
        X  = np.column_stack([np.ones(len(x1)), x1])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, x2, rcond=None)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        c, rho = beta
        mu_hat  = c / (1 - rho) if abs(1 - rho) > 1e-9 else 0.0
        resid   = x2 - (c + rho * x1)
        sig2    = np.var(resid, ddof=2)
        sig2    = max(sig2, 1e-9)

        # First observation contribution under H1
        sig2_1  = sig2 / max(1 - rho**2, 1e-9)
        ll_h1   = norm.logpdf(x[0], mu_hat, np.sqrt(sig2_1))
        ll_h1  += np.sum(norm.logpdf(resid, 0, np.sqrt(sig2)))

        lr_stat = -2 * (ll_h0 - ll_h1)
        lr_stat = max(lr_stat, 0.0)
        p_val   = 1 - _chi2.cdf(lr_stat, df=3)
        return round(lr_stat, 4), round(p_val, 4)

    def _ad_uniform(z_arr):
        """Anderson-Darling test against Uniform(0,1)."""
        z_s = np.sort(np.clip(z_arr, 1e-9, 1 - 1e-9))
        n   = len(z_s)
        i   = np.arange(1, n + 1)
        A2  = -n - np.sum((2*i - 1) / n * (np.log(z_s) + np.log(1 - z_s[::-1])))
        # p-value approximation (Marsaglia & Marsaglia 2004)
        z_adj = A2 * (1 + 0.75/n + 2.25/n**2)
        if   z_adj < 0.200: p = 1 - np.exp(-13.436 + 101.14*z_adj - 223.73*z_adj**2)
        elif z_adj < 0.340: p = 1 - np.exp(-8.318  + 42.796*z_adj - 59.938*z_adj**2)
        elif z_adj < 0.600: p = np.exp(0.9177 - 4.279*z_adj - 1.38*z_adj**2)
        elif z_adj < 13.00: p = np.exp(1.2937 - 5.709*z_adj + 0.0186*z_adj**2)
        else:                p = 0.0
        return round(float(A2), 4), round(float(np.clip(p, 0, 1)), 4)

    rows = []
    model_map = {
        "HS":        "HS_VaR",
        "Normal":    "Normal_VaR",
        "Student-t": "StudT_VaR",
        "GARCH":     "GARCH_VaR",
    }

    for conf in conf_levels:
        res = results_cache[conf]
        print(f"  Computing PITs at {int(conf*100)}% CI...")

        for model in ["HS", "Normal", "Student-t", "GARCH"]:
            z_series = []

            for date in res.index:
                r = res.loc[date, "actual"]

                if model == "HS":
                    # Empirical CDF: rank of r in its own window
                    hs_var = res.loc[date, "HS_VaR"]
                    # Use stored HS_ES as proxy for tail; reconstruct empirical CDF
                    # We re-derive z from Normal as approximation — better: use stored
                    # Normal_VaR sigma to back out window sigma, then use HS rank
                    # Cleanest: use Normal CDF with same mu/sigma as a proxy baseline
                    # but TRUE HS PIT requires window data. We approximate via:
                    # z_HS = rank(r in window) / (n+1) — we reconstruct window here.
                    pos = actual.index.get_loc(date)
                    win = actual.iloc[max(0, pos - WINDOW - 1): pos - 1].values
                    win = win[~np.isnan(win)]
                    if len(win) < 10:
                        continue
                    n_w = len(win)
                    rank = np.sum(win < r)
                    z = (rank + 0.5) / n_w   # continuity-corrected empirical CDF

                elif model == "Normal":
                    nm_var = res.loc[date, "Normal_VaR"]
                    nm_es  = res.loc[date, "Normal_ES"]
                    # Back out sigma: ES = -(mu - sigma * phi(z_a) / (1-alpha))
                    # VaR = -(mu + z_a * sigma), where z_a = norm.ppf(1-conf)
                    # Use VaR directly: sigma = (mu - (-VaR)) / z_a
                    z_a   = norm.ppf(1 - conf)
                    # Approximate mu ≈ 0 (rolling mean is tiny vs sigma)
                    sigma = abs(-nm_var / z_a) if abs(z_a) > 1e-9 else 1e-4
                    sigma = max(sigma, 1e-6)
                    z = norm.cdf(r, loc=0.0, scale=sigma)

                elif model == "Student-t":
                    nu = res.loc[date, "StudT_nu"]
                    st_var = res.loc[date, "StudT_VaR"]
                    if np.isnan(nu) or nu < 2.01:
                        continue
                    # Back out scale: VaR = -(loc + t.ppf(1-conf,nu)*scale), loc≈0
                    t_q   = student_t.ppf(1 - conf, df=nu)
                    scale = abs(-st_var / t_q) if abs(t_q) > 1e-9 else 1e-4
                    scale = max(scale, 1e-6)
                    z = student_t.cdf(r, df=nu, loc=0.0, scale=scale)

                elif model == "GARCH":
                    gc_var = res.loc[date, "GARCH_VaR"]
                    # GARCH uses Normal CDF with GARCH sigma_next
                    # Back out sigma_next from VaR: VaR = -(mu + z_a*sigma), mu≈0
                    z_a    = norm.ppf(1 - conf)
                    sigma  = abs(-gc_var / z_a) if abs(z_a) > 1e-9 else 1e-4
                    sigma  = max(sigma, 1e-6)
                    z = norm.cdf(r, loc=0.0, scale=sigma)

                z_series.append(np.clip(z, 1e-7, 1 - 1e-7))

            if len(z_series) < 30:
                continue

            z_arr = np.array(z_series)

            # 1. KS test
            ks_stat, ks_p = kstest(z_arr, "uniform")

            # 2. AD test
            ad_stat, ad_p = _ad_uniform(z_arr)

            # 3. Berkowitz LR test
            bk_stat, bk_p = _berkowitz_lr(z_arr)

            # 4. Ljung-Box on z (levels)
            lb_z  = acorr_ljungbox(z_arr, lags=10, return_df=True)
            lb_z_p = round(lb_z["lb_pvalue"].min(), 4)

            # 5. Ljung-Box on (z - 0.5)^2 (variance clustering)
            lb_z2  = acorr_ljungbox((z_arr - 0.5)**2, lags=10, return_df=True)
            lb_z2_p = round(lb_z2["lb_pvalue"].min(), 4)

            rows.append({
                "Model":         model,
                "CI":            conf,
                "N":             len(z_arr),
                "Mean_z":        round(float(np.mean(z_arr)), 4),
                "Std_z":         round(float(np.std(z_arr)), 4),
                "KS_stat":       round(float(ks_stat), 4),
                "KS_p":          round(float(ks_p), 4),
                "Rej_KS":        "Y" if ks_p < 0.05 else "N",
                "AD_stat":       ad_stat,
                "AD_p":          ad_p,
                "Rej_AD":        "Y" if ad_p < 0.05 else "N",
                "Berk_LR":       bk_stat,
                "Berk_p":        bk_p,
                "Rej_Berk":      "Y" if (bk_p is not None and bk_p < 0.05) else "N",
                "LB_z_minp":     lb_z_p,
                "Rej_LBz":       "Y" if lb_z_p < 0.05 else "N",
                "LB_z2_minp":    lb_z2_p,
                "Rej_LBz2":      "Y" if lb_z2_p < 0.05 else "N",
            })

    return pd.DataFrame(rows)


def _build_z_arrays(results_cache, actual, conf=0.99):
    """
    Shared helper: reconstruct PIT z_t series for all four models at a given CI.
    Returns dict {model_name: z_array} with z clipped to (1e-7, 1-1e-7).
    Used by both plot_pit_histograms and plot_pit_qqplots to avoid code duplication.
    """
    res    = results_cache[conf]
    z_a    = norm.ppf(1 - conf)
    output = {}

    for model in ["HS", "Normal", "Student-t", "GARCH"]:
        z_series = []
        for date in res.index:
            r = res.loc[date, "actual"]
            try:
                if model == "HS":
                    pos = actual.index.get_loc(date)
                    win = actual.iloc[max(0, pos - WINDOW - 1): pos - 1].values
                    win = win[~np.isnan(win)]
                    if len(win) < 10:
                        continue
                    z = (np.sum(win < r) + 0.5) / len(win)

                elif model == "Normal":
                    sigma = max(abs(-res.loc[date, "Normal_VaR"] / z_a), 1e-6)
                    z = norm.cdf(r, 0.0, sigma)

                elif model == "Student-t":
                    nu = res.loc[date, "StudT_nu"]
                    if np.isnan(nu) or nu < 2.01:
                        continue
                    t_q   = student_t.ppf(1 - conf, df=nu)
                    scale = max(abs(-res.loc[date, "StudT_VaR"] / t_q), 1e-6)
                    z = student_t.cdf(r, df=nu, loc=0.0, scale=scale)

                elif model == "GARCH":
                    sigma = max(abs(-res.loc[date, "GARCH_VaR"] / z_a), 1e-6)
                    z = norm.cdf(r, 0.0, sigma)

                z_series.append(np.clip(z, 1e-7, 1 - 1e-7))
            except Exception:
                continue

        if len(z_series) >= 30:
            output[model] = np.array(z_series)

    return output


def plot_pit_histograms(pit_df, results_cache, actual, conf_levels):
    """
    3-row × 4-col figure at 99% CI:
      Row 1: z_t histogram vs U(0,1) flat density
      Row 2: Φ⁻¹(z_t) normal-score histogram vs N(0,1)
      Row 3: U(0,1) Q-Q plot  — ordered z_t vs theoretical uniform quantiles
    """
    models = ["HS", "Normal", "Student-t", "GARCH"]
    colors = ["steelblue", "tomato", "green", "gold"]
    conf   = 0.99

    z_arrays = _build_z_arrays(results_cache, actual, conf)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))

    for col_idx, (model, color) in enumerate(zip(models, colors)):
        if model not in z_arrays:
            continue

        z_arr = z_arrays[model]
        x_arr = norm.ppf(z_arr)
        n     = len(z_arr)

        sub  = pit_df[(pit_df["Model"] == model) & (pit_df["CI"] == conf)]
        ks_p = sub["KS_p"].values[0]
        ad_p = sub["AD_p"].values[0]
        bk_p = sub["Berk_p"].values[0]

        # ── Row 0: PIT histogram vs U(0,1) ──────────────────────
        ax0 = axes[0, col_idx]
        ax0.hist(z_arr, bins=25, density=True, color=color, alpha=0.7, edgecolor="white")
        ax0.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="U(0,1)")
        ax0.set_title(f"{model} — PIT Histogram\nKS p={ks_p:.3f}  AD p={ad_p:.3f}",
                      fontsize=9)
        ax0.set_xlabel("z = F_t(r_{t+1})", fontsize=8)
        ax0.set_ylabel("Density", fontsize=8)
        ax0.legend(fontsize=7)
        ax0.set_xlim(0, 1)

        # ── Row 1: Normal-score histogram vs N(0,1) ─────────────
        ax1 = axes[1, col_idx]
        xg  = np.linspace(-4.5, 4.5, 300)
        ax1.hist(x_arr, bins=35, density=True, color=color, alpha=0.7, edgecolor="white")
        ax1.plot(xg, norm.pdf(xg), "k--", linewidth=1.5, label="N(0,1)")
        ax1.set_title(f"{model} — Normal Score\nBerkowitz LR p={bk_p:.3f}", fontsize=9)
        ax1.set_xlabel("x = Φ⁻¹(z_t)", fontsize=8)
        ax1.set_ylabel("Density", fontsize=8)
        ax1.legend(fontsize=7)
        ax1.set_xlim(-5, 5)

        # ── Row 2: U(0,1) Q-Q plot ───────────────────────────────
        ax2 = axes[2, col_idx]
        z_sorted    = np.sort(z_arr)
        # Theoretical quantiles of U(0,1): (i - 0.5) / n
        theoretical = (np.arange(1, n + 1) - 0.5) / n
        ax2.scatter(theoretical, z_sorted, s=2, color=color, alpha=0.5, label="Observed")
        ax2.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="45° line")
        # 95% Kolmogorov-Smirnov confidence band
        ks_band = 1.36 / np.sqrt(n)   # KS critical value at 5%
        ax2.fill_between([0, 1],
                         [max(0, q - ks_band) for q in [0, 1]],
                         [min(1, q + ks_band) for q in [0, 1]],
                         alpha=0.12, color="grey", label="95% KS band")
        ax2.set_title(f"{model} — U(0,1) Q-Q Plot\nKS p={ks_p:.3f}  AD p={ad_p:.3f}",
                      fontsize=9)
        ax2.set_xlabel("Theoretical U(0,1) quantile", fontsize=8)
        ax2.set_ylabel("Empirical quantile", fontsize=8)
        ax2.legend(fontsize=7)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

    plt.suptitle(
        "PIT Tests — 99% CI\n"
        "Row 1: z_t ~ U(0,1) histogram  |  "
        "Row 2: Φ⁻¹(z_t) ~ N(0,1) histogram  |  "
        "Row 3: U(0,1) Q-Q plot",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig_pit_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig_pit_histograms.png")


def plot_pit_qqplots(results_cache, actual, pit_df):
    """
    Standalone 2-row × 4-col figure focused purely on Q-Q plots at 99% CI:
      Row 1: U(0,1) Q-Q  — z_t vs theoretical uniform quantiles
      Row 2: N(0,1) Q-Q  — x_t = Φ⁻¹(z_t) vs theoretical normal quantiles
    Gives a cleaner, larger view of the same information for the report.
    """
    from scipy.stats import probplot

    models = ["HS", "Normal", "Student-t", "GARCH"]
    colors = ["steelblue", "tomato", "green", "gold"]
    conf   = 0.99

    z_arrays = _build_z_arrays(results_cache, actual, conf)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    for col_idx, (model, color) in enumerate(zip(models, colors)):
        if model not in z_arrays:
            continue

        z_arr = z_arrays[model]
        x_arr = norm.ppf(z_arr)
        n     = len(z_arr)

        sub  = pit_df[(pit_df["Model"] == model) & (pit_df["CI"] == conf)]
        ks_p = sub["KS_p"].values[0]
        ad_p = sub["AD_p"].values[0]
        bk_p = sub["Berk_p"].values[0]

        # ── Row 0: U(0,1) Q-Q ────────────────────────────────────
        ax0 = axes[0, col_idx]
        z_sorted    = np.sort(z_arr)
        theoretical = (np.arange(1, n + 1) - 0.5) / n
        ks_band     = 1.36 / np.sqrt(n)

        ax0.scatter(theoretical, z_sorted, s=3, color=color, alpha=0.5)
        ax0.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="45° line")
        ax0.fill_between(
            theoretical,
            np.clip(theoretical - ks_band, 0, 1),
            np.clip(theoretical + ks_band, 0, 1),
            alpha=0.15, color="grey", label="95% KS band"
        )
        # Annotate key deviations: left and right tail
        left_dev  = z_sorted[:int(0.05*n)] - theoretical[:int(0.05*n)]
        right_dev = z_sorted[int(0.95*n):] - theoretical[int(0.95*n):]
        ax0.annotate(f"Left tail dev\n{left_dev.mean():+.3f}",
                     xy=(0.04, z_sorted[int(0.05*n)-1]),
                     fontsize=7, color="darkred",
                     xytext=(0.12, z_sorted[int(0.05*n)-1] - 0.08),
                     arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))
        ax0.set_title(f"{model} — U(0,1) Q-Q\nKS p={ks_p:.3f}  AD p={ad_p:.3f}",
                      fontsize=9)
        ax0.set_xlabel("Theoretical U(0,1)", fontsize=8)
        ax0.set_ylabel("Empirical", fontsize=8)
        ax0.legend(fontsize=7)
        ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)

        # ── Row 1: N(0,1) Q-Q ────────────────────────────────────
        ax1 = axes[1, col_idx]
        (osm, osr), (slope, intercept, _) = probplot(x_arr, dist="norm")
        ax1.scatter(osm, osr, s=3, color=color, alpha=0.5)
        fit_line = slope * np.array([osm[0], osm[-1]]) + intercept
        ax1.plot([osm[0], osm[-1]], fit_line, "k--", linewidth=1.5, label="N(0,1) fit")
        ax1.set_title(f"{model} — N(0,1) Q-Q  [Φ⁻¹(z_t)]\nBerkowitz LR p={bk_p:.3f}",
                      fontsize=9)
        ax1.set_xlabel("Theoretical N(0,1)", fontsize=8)
        ax1.set_ylabel("Empirical Φ⁻¹(z_t)", fontsize=8)
        ax1.legend(fontsize=7)

    plt.suptitle(
        "PIT Q-Q Plots — 99% CI\n"
        "Row 1: z_t vs U(0,1)  |  Row 2: Φ⁻¹(z_t) vs N(0,1)\n"
        "Deviations from the 45° line indicate distributional mis-specification",
        fontsize=10, y=1.02
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig_pit_qqplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig_pit_qqplots.png")


def backtest_all_models(results_cache):
    model_map = {"HS":        "HS_VaR",
                 "Normal":    "Normal_VaR",
                 "Student-t": "StudT_VaR",
                 "GARCH":     "GARCH_VaR"}
    rows = []
    for conf in CONF_LEVELS:
        res = results_cache[conf]
        for model, var_col in model_map.items():
            hits = hit_sequence(res["actual"], res[var_col])
            cc   = christoffersen_cc_test(hits, conf)
            dur  = duration_test(hits)
            dq   = dq_test(hits, res[var_col], conf)
            rows.append({
                "Model": model, "CI": conf,
                "k":      cc["k"],
                "LR_uc":  round(cc["LR_uc"], 2),
                "p_uc":   round(cc["p_uc"], 4),
                "Rej_uc": "Y" if cc["Rej_uc"] else "N",
                "LR_ind": round(cc["LR_ind"], 2),
                "p_ind":  round(cc["p_ind"], 4),
                "Rej_ind":"Y" if cc["Rej_ind"] else "N",
                "LR_cc":  round(cc["LR_cc"], 2),
                "p_cc":   round(cc["p_cc"], 4),
                "Rej_cc": "Y" if cc["Rej_cc"] else "N",
                "LR_dur": round(dur["LR_dur"], 2) if not np.isnan(dur["LR_dur"]) else np.nan,
                "p_dur":  round(dur["p_dur"], 4) if not np.isnan(dur["p_dur"]) else np.nan,
                "Rej_dur":"Y" if dur["Rej_dur"] else "N",
                "DQ":     round(dq["DQ"], 2) if not np.isnan(dq["DQ"]) else np.nan,
                "p_dq":   round(dq["p_dq"], 4) if not np.isnan(dq["p_dq"]) else np.nan,
                "Rej_dq": "Y" if dq["Rej_dq"] else "N",
            })
    return pd.DataFrame(rows)


def plot_backtest_bar_chart(backtest_df):
    conf_levels = [0.90, 0.95, 0.99]
    stat_cols   = ["LR_uc", "LR_ind", "LR_cc", "LR_dur", "DQ"]
    crit_vals   = {
        "LR_uc":  chi2.ppf(0.95, df=1),
        "LR_ind": chi2.ppf(0.95, df=1),
        "LR_cc":  chi2.ppf(0.95, df=2),
        "LR_dur": chi2.ppf(0.95, df=2),
        "DQ":     chi2.ppf(0.95, df=5),
    }
    models  = backtest_df["Model"].unique()
    colors  = ["steelblue", "tomato", "green", "gold"]
    x       = np.arange(len(stat_cols))
    width   = 0.18

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, conf in zip(axes, conf_levels):
        sub = backtest_df[backtest_df["CI"] == conf]
        for idx, (model, color) in enumerate(zip(models, colors)):
            row  = sub[sub["Model"] == model].iloc[0]
            vals = [row.get(c, np.nan) for c in stat_cols]
            vals = [v if not (isinstance(v, float) and np.isnan(v)) else 0 for v in vals]
            ax.bar(x + idx * width, vals, width, label=model, color=color, alpha=0.8)

        for stat, cv in crit_vals.items():
            xi = stat_cols.index(stat)
            ax.plot([xi - width/2, xi + (len(models)-0.5)*width],
                    [cv, cv], "k--", linewidth=0.8)

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(["Kupiec", "Ind", "CC", "Duration", "DQ"], fontsize=8)
        ax.set_title(f"LR Statistics — {int(conf*100)}% CI")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig11_backtest_bar.png", dpi=150)
    plt.close()


def plot_pvalue_heatmap(backtest_df):
    p_cols    = ["p_uc", "p_ind", "p_cc", "p_dur", "p_dq"]
    col_names = ["Kupiec (UC)", "Independence", "CC", "Duration", "DQ (K=4)"]
    models    = backtest_df["Model"].unique()
    conf_labels = [f"{int(c*100)}%" for c in CONF_LEVELS]

    fig, axes = plt.subplots(1, len(p_cols), figsize=(20, 5))
    for ax, pcol, cname in zip(axes, p_cols, col_names):
        mat = pd.DataFrame(index=models, columns=conf_labels, dtype=float)
        for conf, clbl in zip(CONF_LEVELS, conf_labels):
            sub = backtest_df[backtest_df["CI"] == conf]
            for model in models:
                row = sub[sub["Model"] == model]
                if len(row):
                    mat.loc[model, clbl] = row.iloc[0][pcol]

        mat = mat.astype(float)
        colors_map = sns.diverging_palette(10, 145, as_cmap=True)
        sns.heatmap(mat, annot=True, fmt=".3f", cmap=colors_map,
                    center=0.05, vmin=0, vmax=1,
                    linewidths=0.5, ax=ax)
        ax.set_title(cname, fontsize=9)

    plt.suptitle("p-value Heatmap (teal = pass ≥ 0.05, red = fail)", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig12_pvalue_heatmap.png", dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("── Part 1: Data & Statistical Analysis ──")
    prices      = download_prices(TICKERS, START_DATE, END_DATE)
    log_returns = compute_log_returns(prices)
    portfolio   = build_ew_portfolio(log_returns)

    desc   = descriptive_statistics(portfolio, log_returns)
    ann    = annualised_risk_return(portfolio, log_returns)
    corr   = plot_correlation_heatmap(log_returns)

    print("Descriptive Statistics:\n", desc.to_string())
    print("\nAnnualised Risk/Return:\n", ann.to_string())
    print("\nCorrelation Matrix:\n", corr.round(4).to_string())

    plot_prices_and_returns(prices, portfolio)
    plot_distributional_properties(portfolio)
    lb = plot_acf_returns(portfolio)
    plot_rolling_stats(log_returns, portfolio)

    desc.to_csv(OUTPUT_DIR + "table1_desc_stats.csv")
    ann.to_csv(OUTPUT_DIR + "table2_ann_metrics.csv")
    corr.to_csv(OUTPUT_DIR + "table3_correlation.csv")

    print("\n── Part 2 & 3: Rolling VaR/ES & Violations ──")
    summary_df, results_cache = violation_summary(
        portfolio, WINDOW, CONF_LEVELS, ROLL_START)
    print(summary_df.to_string())
    summary_df.to_csv(OUTPUT_DIR + "table5_violations.csv", index=False)

    res_99 = results_cache[0.99]
    plot_rolling_var(res_99, conf=0.99)
    plot_breach_scatter(res_99, conf=0.99)
    plot_violation_rates(summary_df)
    plot_cumulative_violations(results_cache, CONF_LEVELS)

    print("\n── Part 4: Backtesting ──")
    backtest_df = backtest_all_models(results_cache)
    print(backtest_df.to_string())
    backtest_df.to_csv(OUTPUT_DIR + "table6_backtests.csv", index=False)

    plot_backtest_bar_chart(backtest_df)
    plot_pvalue_heatmap(backtest_df)

    # PIT distributional tests — all 4 models, all 3 CI levels
    print("\n── Part 5: PIT Distributional Tests ──")
    pit_df = pit_distributional_test(portfolio, results_cache, CONF_LEVELS)
    print(pit_df.to_string())
    pit_df.to_csv(OUTPUT_DIR + "table_pit.csv", index=False)
    plot_pit_histograms(pit_df, results_cache, portfolio, CONF_LEVELS)
    plot_pit_qqplots(results_cache, portfolio, pit_df)

    # Rolling Student-t nu estimates
    nu_series = res_99["StudT_nu"].dropna()
    fig, ax = plt.subplots(figsize=(12, 4))
    nu_series.plot(ax=ax, color="purple", linewidth=0.7)
    ax.set_title("Rolling MLE Degrees of Freedom (Student-t) — 6-month Window")
    ax.set_ylabel("ν̂")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig_rolling_nu.png", dpi=150)
    plt.close()

    print("\nDone. All outputs saved to", OUTPUT_DIR)