"""
Q5 — Summary & CSV Output
==========================
Prints formatted tables to the console and saves four CSV files to
Q5_OUTPUT_DIR:

  span_risk_arrays.csv      16-scenario P&L breakdown for all 4 pairs
  margin_comparison.csv     No-net vs SPAN-net comparison table
  netting_benefits.csv      Netting benefit ranked by % benefit
  margin_decomposition.csv  Scanning risk / SOM / NOV breakdown
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from q5_market_data import (
    SNAPSHOT_DATE, CONTRACT_LABEL, EXCHANGE, F, ATM_STRIKE,
    IV_CALL_ATM, IV_PUT_ATM, CALL_PRICE_ATM, PUT_PRICE_ATM,
    T, RISK_FREE_RATE, CONTRACT_SIZE,
    PSR, VSR, SOM_RATE,
)
from q5_positions import ALL_PAIRS, single_positions
from q5_span_engine import span_margin, no_netting_margin
from q5_span_scenarios import build_scenario_table, scenario_breakdown_df
from q5_margin_calculator import (
    run_all_margins,
    decomposition_df,
)


# ── Console helpers ───────────────────────────────────────────────────────────

SEP  = "=" * 72
SEP2 = "-" * 72


def _print_header(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _fmt_usd(x): return f"${x:>10,.2f}"


# ── Market snapshot ───────────────────────────────────────────────────────────

def print_market_snapshot():
    _print_header("Q5  MARKET DATA SNAPSHOT")
    print(f"  Snapshot date    : {SNAPSHOT_DATE}")
    print(f"  Contract         : {CONTRACT_LABEL}  ({EXCHANGE})")
    print(f"  Futures price    : ${F:.4f}/lb")
    print(f"  ATM strike       : ${ATM_STRIKE:.2f}/lb")
    print(f"  Contract size    : {CONTRACT_SIZE:,} lb")
    print(f"  Time to expiry   : {T:.4f} years  ({round(T*365)} days)")
    print(f"  Risk-free rate   : {RISK_FREE_RATE*100:.2f}%  (SOFR)")
    print(f"  IV (ATM call)    : {IV_CALL_ATM*100:.2f}%")
    print(f"  IV (ATM put)     : {IV_PUT_ATM*100:.2f}%")
    print(f"  ATM call price   : ${CALL_PRICE_ATM:.4f}/lb  (${CALL_PRICE_ATM*CONTRACT_SIZE:,.2f}/contract)")
    print(f"  ATM put  price   : ${PUT_PRICE_ATM:.4f}/lb  (${PUT_PRICE_ATM*CONTRACT_SIZE:,.2f}/contract)")
    print(SEP2)
    print(f"  SPAN PSR         : ${PSR:.4f}/lb  (${PSR*CONTRACT_SIZE:,.0f}/contract)")
    print(f"  SPAN VSR         : {VSR*100:.0f} vol-pts")
    print(f"  Short option min : ${SOM_RATE:.2f}/contract")


# ── Risk array table ──────────────────────────────────────────────────────────

def print_risk_arrays():
    scenario_df = build_scenario_table()
    for pair in ALL_PAIRS:
        _print_header(f"16-SCENARIO RISK ARRAY — {pair['name']}")
        df = scenario_breakdown_df(pair["legs"], scenario_df)
        leg_cols = [pos.get("label", pos["type"]) for pos in pair["legs"]]
        show_cols = ["scenario_no", "label", "delta_F", "delta_sigma", "weight"] + \
                    leg_cols + ["portfolio_pnl", "weighted_loss"]
        out = df[show_cols].copy()
        # Format monetary columns
        for col in leg_cols + ["portfolio_pnl", "weighted_loss"]:
            out[col] = out[col].map(lambda x: f"${x:>8,.1f}")
        out["delta_F"] = out["delta_F"].map(lambda x: f"{x:+.4f}")
        out["delta_sigma"] = out["delta_sigma"].map(lambda x: f"{x:+.3f}")
        print(out.to_string(index=False))


# ── Margin comparison ─────────────────────────────────────────────────────────

def print_margin_comparison():
    _print_header("SPAN MARGIN COMPARISON — No-Netting vs Portfolio Netting")
    df = run_all_margins()
    for _, row in df.iterrows():
        print(f"\n  {row['pair_name']}")
        print(f"    Leg 1 standalone margin  : {_fmt_usd(row['leg1_margin'])}")
        print(f"    Leg 2 standalone margin  : {_fmt_usd(row['leg2_margin'])}")
        print(f"    No-netting total         : {_fmt_usd(row['no_net_total'])}")
        print(f"    SPAN portfolio margin    : {_fmt_usd(row['span_net_margin'])}")
        print(f"    Netting benefit          : {_fmt_usd(row['netting_benefit'])}  ({row['netting_pct']:.1f}%)")


# ── Decomposition table ───────────────────────────────────────────────────────

def print_decomposition():
    _print_header("SPAN MARGIN DECOMPOSITION  (Scanning / SOM / NOV / Final)")
    df = decomposition_df()
    for pair_name in df["pair_name"].unique():
        sub = df[df["pair_name"] == pair_name]
        print(f"\n  {pair_name}")
        header = f"  {'Method':<14} {'ScanRisk':>10} {'SOM':>8} {'ActiveRsk':>10} {'NOV-Cred':>10} {'Margin':>10}"
        print(header)
        print(f"  {'-'*62}")
        for _, row in sub.iterrows():
            print(
                f"  {row['method']:<14}"
                f" {_fmt_usd(row['scanning_risk'])}"
                f" {_fmt_usd(row['som'])}"
                f" {_fmt_usd(row['active_risk'])}"
                f" {_fmt_usd(row['nov_credit'])}"
                f" {_fmt_usd(row['span_margin'])}"
            )


# ── CSV savers ────────────────────────────────────────────────────────────────

def save_risk_arrays_csv():
    scenario_df = build_scenario_table()
    frames = []
    for pair in ALL_PAIRS:
        df = scenario_breakdown_df(pair["legs"], scenario_df).copy()
        df.insert(0, "pair", pair["short"])
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    path = os.path.join(config.Q5_OUTPUT_DIR, "span_risk_arrays.csv")
    combined.to_csv(path, index=False, float_format="%.2f")
    print(f"  Saved: {os.path.basename(path)}")
    return path


def save_margin_comparison_csv():
    df = run_all_margins()
    path = os.path.join(config.Q5_OUTPUT_DIR, "margin_comparison.csv")
    df.to_csv(path, index=False, float_format="%.2f")
    print(f"  Saved: {os.path.basename(path)}")
    return path


def save_netting_benefits_csv():
    df = run_all_margins().sort_values("netting_pct", ascending=False)
    path = os.path.join(config.Q5_OUTPUT_DIR, "netting_benefits.csv")
    df[["pair_name", "no_net_total", "span_net_margin",
        "netting_benefit", "netting_pct"]].to_csv(path, index=False, float_format="%.2f")
    print(f"  Saved: {os.path.basename(path)}")
    return path


def save_decomposition_csv():
    df = decomposition_df()
    path = os.path.join(config.Q5_OUTPUT_DIR, "margin_decomposition.csv")
    df.to_csv(path, index=False, float_format="%.2f")
    print(f"  Saved: {os.path.basename(path)}")
    return path


# ── Run all ───────────────────────────────────────────────────────────────────

def run_all_summary():
    print_market_snapshot()
    print_risk_arrays()
    print_margin_comparison()
    print_decomposition()

    print(f"\n{SEP}")
    print("  Saving CSVs...")
    print(SEP2)
    save_risk_arrays_csv()
    save_margin_comparison_csv()
    save_netting_benefits_csv()
    save_decomposition_csv()
    print("Done.")


if __name__ == "__main__":
    run_all_summary()
