"""
Backtester for Project 1: 3:2:1 Crack Spread → Refiner Stocks.

Strategy: Confidence-scaled daily trading.
  - Chronos AI predicts P(UP) each day using crack spread signal.
  - Position size = |P(UP) - 0.5| × 2 × $100. Held for 1 day.
  - Notional: $100 (multiply by 100K for real $10M trading).
"""

import os
import json
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\styu0\Energy-Strategy\backtest1"
NOTIONAL = 100


def calc_metrics(pnl_series):
    if len(pnl_series) == 0:
        return {"sharpe": 0, "total_pnl": 0, "max_dd": 0, "hit_rate": 0, "days": 0}
    daily_ret = pnl_series / NOTIONAL
    sharpe = np.sqrt(252) * daily_ret.mean() / (daily_ret.std() + 1e-8)
    cum_pnl = pnl_series.cumsum()
    max_dd = (cum_pnl - cum_pnl.cummax()).min()
    hit_rate = (pnl_series > 0).sum() / len(pnl_series) * 100
    return {"sharpe": round(float(sharpe), 2), "total_pnl": round(float(cum_pnl.iloc[-1]), 2),
            "max_dd": round(float(max_dd), 2), "hit_rate": round(float(hit_rate), 1), "days": len(pnl_series)}


def pos_size(p_up):
    return (1 if p_up > 0.5 else -1) * abs(p_up - 0.5) * 2 * NOTIONAL


def run_backtest():
    inf_df = pd.read_csv(os.path.join(DATA_DIR, "inference_results.csv"), parse_dates=['date'])
    with open(os.path.join(DATA_DIR, "moirai_discovery.json")) as f:
        moirai = json.load(f)
    master = pd.read_csv(os.path.join(DATA_DIR, "master_dataset.csv"), index_col=0, parse_dates=True).sort_index()

    stocks = inf_df['stock'].unique().tolist()
    stock_analysis = moirai.get("stock_analysis", {})

    # ── Compute all strategies ──
    individual = {}
    for s in stocks:
        sdf = inf_df[inf_df['stock'] == s].sort_values('date')
        pnl = sdf['p_up'].apply(pos_size) * sdf['actual_return']
        sdf_acc = ((sdf['p_up'] > 0.5) == (sdf['actual_return'] > 0)).sum()
        individual[s] = {"metrics": calc_metrics(pnl.reset_index(drop=True)),
                         "accuracy": f"{sdf_acc}/{len(sdf)} ({sdf_acc/len(sdf)*100:.0f}%)"}

    dates = sorted(inf_df['date'].unique())
    eq_pnl = pd.Series([sum(pos_size(r['p_up']) / len(stocks) * r['actual_return']
                             for _, r in inf_df[inf_df['date'] == d].iterrows()) for d in dates])

    raw_w = {s: stock_analysis.get(s, {}).get("stock_to_crack", 1.0) for s in stocks}
    tw = sum(raw_w.values())
    nw = {s: w / tw for s, w in raw_w.items()}
    attn_pnl = pd.Series([sum(pos_size(r['p_up']) * nw.get(r['stock'], 1/len(stocks)) * r['actual_return']
                               for _, r in inf_df[inf_df['date'] == d].iterrows()) for d in dates])

    eq_m = calc_metrics(eq_pnl)
    at_m = calc_metrics(attn_pnl)

    # ══════════════════════════════════════════════════════════════
    # OUTPUT
    # ══════════════════════════════════════════════════════════════
    print("=" * 72)
    print(" BACKTEST RESULTS: Crack Spread → Refiner Stocks ($100 notional)")
    print("=" * 72)
    print(f"\n  Strategy: Chronos predicts P(UP) daily using crack spread signal.")
    print(f"  Sizing: Bet = |P(UP)-0.5| × 2 × $100.  Hold: 1 day.\n")

    print(f"  {'STRATEGY':<30} {'Sharpe':>7} {'P&L':>8} {'Hit%':>6} {'MaxDD':>8} {'Accuracy'}")
    print(f"  {'─' * 72}")
    for s, d in sorted(individual.items(), key=lambda x: x[1]['metrics']['sharpe'], reverse=True):
        m = d['metrics']
        print(f"  {s:<30} {m['sharpe']:>7.2f} ${m['total_pnl']:>6.2f} {m['hit_rate']:>5.1f}% ${m['max_dd']:>6.2f}  {d['accuracy']}")
    print(f"  {'─' * 72}")
    print(f"  {'Equal-Weight Basket':<30} {eq_m['sharpe']:>7.2f} ${eq_m['total_pnl']:>6.2f} {eq_m['hit_rate']:>5.1f}% ${eq_m['max_dd']:>6.2f}")
    print(f"  {'Attn-Weighted Basket':<30} {at_m['sharpe']:>7.2f} ${at_m['total_pnl']:>6.2f} {at_m['hit_rate']:>5.1f}% ${at_m['max_dd']:>6.2f}")

    best = max({**{s: d['metrics'] for s, d in individual.items()},
                "EQ-Basket": eq_m, "Attn-Basket": at_m}.items(), key=lambda x: x[1]['sharpe'])
    print(f"\n  🏆 BEST: {best[0]} (Sharpe {best[1]['sharpe']})")
    print("=" * 72)


if __name__ == "__main__":
    run_backtest()

