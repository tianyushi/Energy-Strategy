"""
Chronos-2 Pair Contribution & Ablation Test.

Experiments:
  A) BASELINE: Chronos-2 forecasts DM003EL using ONLY its own history (univariate).
  B) FULL MV:  Chronos-2 forecasts DM003EL using its history + 9 MOIRAI-discovered pairs.
  C) ABLATION: For each pair, run DM003EL + that single pair to measure
               the marginal contribution of each feature.

Output: Standard Chronos quantile forecast format + accuracy comparison.
Hardware: RTX 4060 (8GB VRAM)
"""

import sys
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.data_preprocessing import fetch_data, pivot_and_filter

# ======================================================================
# CONFIGURATION
# ======================================================================
TARGET_SYMBOL = "DM003EL"
PAIRS = [
    "DM0043X", "DM0033X", "DM0043Z", "DM0093Y",
    "DM0033Z", "DM0093X", "DM0033Y", "DM0043Y", "DM003AS"
]
PREDICTION_LENGTH = 10
CONTEXT_LENGTH = 2048
MODEL_PATH = "amazon/chronos-2"


# ======================================================================
# Helper: Run a single Chronos-2 forecast with given variates
# ======================================================================
def run_forecast(pipeline, train_df, variates, prediction_length):
    """
    Run Chronos-2 with given variates. Returns (median, q10, q90) for the
    FIRST variate (the target).
    """
    context = train_df[variates].values.T  # (n_variates, n_days)
    if context.shape[1] > CONTEXT_LENGTH:
        context = context[:, -CONTEXT_LENGTH:]

    tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)  # (1, n_var, ctx)

    with torch.inference_mode():
        q_list, m_list = pipeline.predict_quantiles(
            tensor,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )

    q_tensor = q_list[0]  # (n_variates, pred_len, n_quantiles)
    m_tensor = m_list[0]  # (n_variates, pred_len)

    return {
        "mean": m_tensor[0].numpy(),             # target variate median
        "q10": q_tensor[0, :, 0].numpy(),         # 10th pctile
        "q20": q_tensor[0, :, 1].numpy(),
        "q30": q_tensor[0, :, 2].numpy(),
        "q40": q_tensor[0, :, 3].numpy(),
        "q50": q_tensor[0, :, 4].numpy(),         # median
        "q60": q_tensor[0, :, 5].numpy(),
        "q70": q_tensor[0, :, 6].numpy(),
        "q80": q_tensor[0, :, 7].numpy(),
        "q90": q_tensor[0, :, 8].numpy(),         # 90th pctile
    }


def calc_metrics(forecast, actual, last_train):
    """Calculate MAE, direction accuracy, and per-day hit array."""
    median = forecast["mean"]
    mae = np.mean(np.abs(median - actual))

    hits = 0
    hit_details = []
    for d in range(len(actual)):
        pred_dir = 1 if (median[d] - last_train) > 0 else -1
        act_dir = 1 if (actual[d] - last_train) > 0 else -1
        hit = pred_dir == act_dir
        if hit:
            hits += 1
        hit_details.append(hit)

    hit_rate = hits / len(actual) * 100
    return {"mae": float(mae), "hit_rate": float(hit_rate), "hits": hits, "total": len(actual), "hit_details": hit_details}


# ======================================================================
# Main Pipeline
# ======================================================================
def run_contribution_test():
    print("=" * 70)
    print("  CHRONOS-2 PAIR CONTRIBUTION & ABLATION TEST")
    print("=" * 70)

    # ── 1. DATA ──
    df_raw = fetch_data()
    pivot_df = pivot_and_filter(df_raw, liquidity_threshold=0.80)

    all_symbols = list(pivot_df.columns)
    available_pairs = [p for p in PAIRS if p in all_symbols]
    if TARGET_SYMBOL not in all_symbols:
        print(f"[ERROR] Target {TARGET_SYMBOL} not found!"); sys.exit(1)

    mv_df = pivot_df[[TARGET_SYMBOL] + available_pairs].copy()
    mv_df = mv_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    train_df = mv_df.iloc[:-PREDICTION_LENGTH]
    test_df = mv_df.iloc[-PREDICTION_LENGTH:]
    last_train_z = train_df[TARGET_SYMBOL].iloc[-1]
    actual_future = test_df[TARGET_SYMBOL].values

    print(f"[DATA] Target: {TARGET_SYMBOL} | Pairs: {len(available_pairs)}")
    print(f"[DATA] Train: {len(train_df)} days | Test: {PREDICTION_LENGTH} days")
    print(f"[DATA] Last Train Z: {last_train_z:+.4f}\n")

    # ── 2. LOAD MODEL ──
    print(f"[MODEL] Loading {MODEL_PATH}...")
    from chronos import BaseChronosPipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_PATH, device_map="cuda", dtype=torch.float32,
    )

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT A: BASELINE (Univariate — Target Only)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print(f"  EXPERIMENT A: BASELINE (DM003EL only — Univariate)")
    print(f"{'─' * 70}")
    fc_baseline = run_forecast(pipeline, train_df, [TARGET_SYMBOL], PREDICTION_LENGTH)
    m_baseline = calc_metrics(fc_baseline, actual_future, last_train_z)
    print(f"  MAE: {m_baseline['mae']:.4f} | Hit Rate: {m_baseline['hit_rate']:.0f}% ({m_baseline['hits']}/{m_baseline['total']})")

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT B: FULL MULTIVARIATE (Target + All 9 Pairs)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print(f"  EXPERIMENT B: FULL MULTIVARIATE (DM003EL + {len(available_pairs)} pairs)")
    print(f"{'─' * 70}")
    fc_full = run_forecast(pipeline, train_df, [TARGET_SYMBOL] + available_pairs, PREDICTION_LENGTH)
    m_full = calc_metrics(fc_full, actual_future, last_train_z)
    print(f"  MAE: {m_full['mae']:.4f} | Hit Rate: {m_full['hit_rate']:.0f}% ({m_full['hits']}/{m_full['total']})")

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT C: PER-PAIR ABLATION
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print(f"  EXPERIMENT C: PER-PAIR MARGINAL CONTRIBUTION")
    print(f"{'─' * 70}")

    pair_results = {}
    for pair in available_pairs:
        fc_pair = run_forecast(pipeline, train_df, [TARGET_SYMBOL, pair], PREDICTION_LENGTH)
        m_pair = calc_metrics(fc_pair, actual_future, last_train_z)
        mae_delta = m_baseline["mae"] - m_pair["mae"]  # positive = improvement
        pair_results[pair] = {
            "mae": m_pair["mae"],
            "hit_rate": m_pair["hit_rate"],
            "mae_improvement": mae_delta,
        }
        sign = "✅ +" if mae_delta > 0 else "❌ "
        print(f"  {pair:<12} MAE: {m_pair['mae']:.4f}  Hit: {m_pair['hit_rate']:>3.0f}%  Δ MAE: {sign}{abs(mae_delta):.4f}")

    # ══════════════════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ══════════════════════════════════════════════════════════════════
    total_improvement = m_baseline["mae"] - m_full["mae"]

    print(f"\n{'=' * 70}")
    print(f"  📊 CONTRIBUTION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'CONFIGURATION':<35} {'MAE':>8} {'HIT RATE':>10} {'Δ vs BASE':>10}")
    print(f"  {'─' * 65}")
    print(f"  {'A) Baseline (Target Only)':<35} {m_baseline['mae']:>8.4f} {m_baseline['hit_rate']:>9.0f}% {'—':>10}")
    print(f"  {'B) Full MV (Target + 9 Pairs)':<35} {m_full['mae']:>8.4f} {m_full['hit_rate']:>9.0f}% {total_improvement:>+10.4f}")
    print(f"  {'─' * 65}")

    # Sort pairs by contribution
    sorted_pairs = sorted(pair_results.items(), key=lambda x: x[1]["mae_improvement"], reverse=True)
    for rank, (sym, res) in enumerate(sorted_pairs, 1):
        sign = "✅" if res["mae_improvement"] > 0 else "❌"
        print(f"  {rank:>2}. {sym:<12} (solo)             {res['mae']:>8.4f} {res['hit_rate']:>9.0f}% {res['mae_improvement']:>+10.4f} {sign}")

    print(f"\n  ── Verdict ──")
    if total_improvement > 0:
        pct = (total_improvement / m_baseline["mae"]) * 100
        print(f"  ✅ MOIRAI pairs IMPROVED forecast by {total_improvement:.4f} MAE ({pct:.1f}% reduction)")
    else:
        pct = (abs(total_improvement) / m_baseline["mae"]) * 100
        print(f"  ❌ MOIRAI pairs DEGRADED forecast by {abs(total_improvement):.4f} MAE ({pct:.1f}% increase)")

    # ══════════════════════════════════════════════════════════════════
    # STANDARD CHRONOS OUTPUT: Day-by-Day Forecast Table
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  🎯 CHRONOS-2 FORECAST OUTPUT: {TARGET_SYMBOL}")
    print(f"{'=' * 70}")
    print(f"  {'DAY':<5} {'MEAN':>8} {'Q10':>8} {'Q50':>8} {'Q90':>8} {'ACTUAL':>8} {'DIR':>6} {'HIT':>4}")
    print(f"  {'─' * 60}")

    for d in range(PREDICTION_LENGTH):
        pred_z = fc_full["mean"][d]
        q10 = fc_full["q10"][d]
        q50 = fc_full["q50"][d]
        q90 = fc_full["q90"][d]
        act = actual_future[d]
        diff = pred_z - last_train_z
        pred_dir = "▲" if diff > 0 else "▼"
        act_diff = act - last_train_z
        act_dir = "▲" if act_diff > 0 else "▼"
        hit = "✅" if pred_dir == act_dir else "❌"
        print(f"  D{d+1:<3} {pred_z:>+8.4f} {q10:>+8.4f} {q50:>+8.4f} {q90:>+8.4f} {act:>+8.4f} {pred_dir:>6} {hit:>4}")

    print(f"{'=' * 70}\n")

    # ── SAVE ──
    results = {
        "target": TARGET_SYMBOL,
        "pairs": available_pairs,
        "baseline_mae": m_baseline["mae"],
        "baseline_hit_rate": m_baseline["hit_rate"],
        "full_mv_mae": m_full["mae"],
        "full_mv_hit_rate": m_full["hit_rate"],
        "total_improvement": float(total_improvement),
        "per_pair": {sym: res for sym, res in sorted_pairs},
    }
    out_path = ROOT / "data" / "pair_contribution_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"  Results saved to: {out_path}")

    # ── PLOT ──
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: Forecast comparison
    ax = axes[0]
    hist = train_df[TARGET_SYMBOL].tail(60)
    ax.plot(range(len(hist)), hist.values, color="black", label="History", linewidth=1.5)
    fc_x = range(len(hist), len(hist) + PREDICTION_LENGTH)
    ax.plot(fc_x, actual_future, color="blue", linewidth=2, label="Actual")
    ax.plot(fc_x, fc_baseline["mean"], color="gray", linestyle="--", label=f"Baseline (MAE={m_baseline['mae']:.3f})")
    ax.plot(fc_x, fc_full["mean"], color="red", linestyle="--", linewidth=2, label=f"+ Pairs (MAE={m_full['mae']:.3f})")
    ax.fill_between(fc_x, fc_full["q10"], fc_full["q90"], color="red", alpha=0.1, label="80% PI")
    ax.set_title(f"Chronos-2: {TARGET_SYMBOL} Forecast Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Z-Score")

    # Right: Feature contribution bar chart
    ax2 = axes[1]
    names = [s[0] for s in sorted_pairs]
    vals = [s[1]["mae_improvement"] for s in sorted_pairs]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]
    ax2.barh(names, vals, color=colors)
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("MAE Improvement vs Baseline (positive = better)")
    ax2.set_title("Per-Pair Marginal Contribution")
    ax2.invert_yaxis()

    plt.tight_layout()
    plot_path = ROOT / "data" / "chronos2_contribution_test.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to: {plot_path}")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    try:
        run_contribution_test()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()
