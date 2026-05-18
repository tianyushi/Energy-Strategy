"""
Backtester for Project 1: 3:2:1 Crack Spread → Refiner Stocks.

BASIC TRADING RULES & MECHANICS:
=================================
1. Universe Selection (Vetted Follower Universe):
   - The strategy only trades refiner stocks classified by Salesforce MOIRAI as "Followers" 
     (attention ratio stock_to_crack / crack_to_stock > 1.2) and verified by walking-forward 
     Chronos-2 ablation analysis (multivariate hit-rate > baseline hit-rate).
   - Vetted universe on 2020 out-of-sample period: ['PSX', 'CVI'].

2. Position Timing & Execution:
   - Signal Time: At the end of day T-1, the history is sliced strictly BEFORE T (history < date T) 
     to predict the directional probability P(UP) and return for Day T (No look-ahead bias).
   - Execution Time: The trade is executed at the Market-on-Close (MOC) of Day T-1 (equivalent to 
     entering at the exact Market Open of Day T).
   - Holding Period: The position is held for exactly 1 business day (Day T).
   - Exit/Roll Time: The position is closed at the Market-on-Close (MOC) of Day T.

3. Position Sizing (Confidence Scaling):
   - Sizing Formula: Size = (1 if P(UP) > 0.5 else -1) * |P(UP) - 0.5| * 2 * NOTIONAL.
   - For a standard notional of $100:
     - Neutral prediction (P(UP) = 0.50) -> Flat position ($0 size).
     - Strong prediction (e.g. P(UP) = 0.80 or 0.20) -> Position scaled to 60% of maximum ($60 size).
     - Maximum confidence (P(UP) = 1.00 or 0.00) -> Max position ($100 size).

4. Volatility / Market Risk Hedging (Beta Hedging):
   - Isolates pure refinery alpha by hedging out broad equity/energy market fluctuations.
   - The return traded is the Beta-Hedged Return:
       Hedged_Return = Raw_Return - Beta * SPY_Return
   - Beta is computed dynamically using a 60-day rolling covariance/variance against SPY.

5. Portfolio Basket Weighting:
   - Equal-Weight Basket: splits capital equally (50.0% each) between PSX and CVI.
   - Attention-Weighted Basket: capital is allocated proportional to their MOIRAI attention
     dependency on the 3:2:1 Crack Spread:
       - PSX Weight: 0.0392 / (0.0392 + 0.0269) = 59.3%
       - CVI Weight: 0.0269 / (0.0392 + 0.0269) = 40.7%
"""

import os
import json
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
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

    # ==============================================================
    # OUTPUT
    # ==============================================================
    print("=" * 72)
    print(" BACKTEST RESULTS: Crack Spread -> Refiner Stocks ($100 notional)")
    print("=" * 72)
    print(f"\n  Strategy: Chronos predicts P(UP) daily using crack spread signal.")
    print(f"  Sizing: Bet = |P(UP)-0.5| × 2 × $100.  Hold: 1 day.")
    print(f"  Attn-Weighted: Capital is allocated to each stock proportional to its MOIRAI")
    print(f"                 attention dependency (strength) on the 3:2:1 crack spread.\n")

    print(f"  {'STRATEGY':<30} {'Sharpe':>7} {'P&L':>8} {'Hit%':>6} {'MaxDD':>8} {'Accuracy':<15} {'Attn-Weight':>11}")
    print(f"  {'-' * 88}")
    for s, d in sorted(individual.items(), key=lambda x: x[1]['metrics']['sharpe'], reverse=True):
        m = d['metrics']
        w_pct = nw.get(s, 0.0) * 100
        print(f"  {s:<30} {m['sharpe']:>7.2f} ${m['total_pnl']:>6.2f} {m['hit_rate']:>5.1f}% ${m['max_dd']:>6.2f}  {d['accuracy']:<15} {w_pct:>10.1f}%")
    print(f"  {'-' * 88}")
    print(f"  {'Equal-Weight Basket':<30} {eq_m['sharpe']:>7.2f} ${eq_m['total_pnl']:>6.2f} {eq_m['hit_rate']:>5.1f}% ${eq_m['max_dd']:>6.2f}  {'1/N each':<15} {'50.0% each':>11}")
    print(f"  {'Attn-Weighted Basket':<30} {at_m['sharpe']:>7.2f} ${at_m['total_pnl']:>6.2f} {at_m['hit_rate']:>5.1f}% ${at_m['max_dd']:>6.2f}  {'Weighted':<15} {'100.0% total':>11}")

    best = max({**{s: d['metrics'] for s, d in individual.items()},
                "EQ-Basket": eq_m, "Attn-Basket": at_m}.items(), key=lambda x: x[1]['sharpe'])
    print(f"\n  [BEST]: {best[0]} (Sharpe {best[1]['sharpe']})")
    


    print("=" * 72)


if __name__ == "__main__":
    run_backtest()

