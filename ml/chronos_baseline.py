"""
Chronos-T5 Zero-Shot Baseline Forecast.

Ingests the ML-Ready dataset directly from Snowflake,
formats it for AutoGluon TimeSeries, and runs a zero-shot
forecast using Amazon's Chronos-T5 architecture.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json

# Attempt to import autogluon, but provide a graceful warning if it's missing
try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
except ImportError:
    print("[ERROR] autogluon.timeseries is not installed in this environment.")
    print("Please run: pip install autogluon.timeseries")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utility.snowflake_client import SnowflakeClient

SOURCE_TABLE = "CMDTYA.PUBLIC.PRICEDATA_ML_DAILY_SUMMARY"


def fetch_prototype_data(product: str = 'Unleaded Gasoline') -> pd.DataFrame:
    """
    Ingest the aggregated daily summary data from Snowflake.
    """
    print(f"[1/3] Ingesting aggregated Daily Summary (Product: {product})...")
    
    query = f"""
        SELECT 
            SYMBOL, ASSESSDATE, Z_SCORE,
            PRODUCT, GRADE, GEOGRAPHY, DELIVERY, TIMING
        FROM {SOURCE_TABLE}
        WHERE PRODUCT = '{product}'
        ORDER BY SYMBOL, ASSESSDATE
    """
    
    with SnowflakeClient() as sf:
        sf.connect()
        df = sf.read_sql(query)
        
    print(f"      Successfully pulled {len(df):,} rows from {SOURCE_TABLE}.")
    
    # Ensure datetime parsing and correct sorting
    df['ASSESSDATE'] = pd.to_datetime(df['ASSESSDATE'])
    df = df.sort_values(by=['SYMBOL', 'ASSESSDATE'])
    
    return df


def format_autogluon_data(df: pd.DataFrame) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
    """
    Transform the Pandas DataFrame into an AutoGluon TimeSeriesDataFrame
    and perform a 14-day chronological Train/Test split.
    """
    print("[2/3] Formatting data and performing 14-day chronological split...")
    
    # Safety: Drop any symbols with fewer than 15 observations (required for 14-day forecast)
    counts = df.groupby('SYMBOL').size()
    valid_symbols = counts[counts >= 15].index
    if len(valid_symbols) < len(counts):
        print(f"      [CLEANUP] Dropped {len(counts) - len(valid_symbols)} symbols with <15 days of history.")
        df = df[df['SYMBOL'].isin(valid_symbols)]

    static_cols = ['PRODUCT', 'GRADE', 'GEOGRAPHY', 'DELIVERY', 'TIMING']
    static_features = df[['SYMBOL'] + static_cols].drop_duplicates(subset=['SYMBOL']).set_index('SYMBOL')
    time_varying_df = df.drop(columns=static_cols)
    
    ts_df = TimeSeriesDataFrame.from_data_frame(
        time_varying_df,
        id_column="SYMBOL",
        timestamp_column="ASSESSDATE"
    )
    
    # Force regular daily frequency (filling gaps with ffill)
    ts_df = ts_df.convert_frequency(freq='D')
    ts_df = ts_df.fillna(method='ffill')
    
    ts_df.static_features = static_features
    
    # Split: Train is everything except the last 14 days per item
    # Test is the full dataframe (AutoGluon evaluates on the last windows)
    train_data = ts_df.slice_by_timestep(None, -14)
    test_data = ts_df
    
    print(f"      Train set: {len(train_data):,} rows | Test set (Full): {len(test_data):,} rows.")
    return train_data, test_data


def calculate_directional_accuracy(train_data: TimeSeriesDataFrame, test_data: TimeSeriesDataFrame, predictions: pd.DataFrame) -> float:
    """
    Calculates the percentage of assets where the model correctly predicted 
    the direction of the price move over the 14-day window.
    """
    hits = 0
    total = 0
    
    for symbol in train_data.item_ids:
        try:
            # Last known point in training
            last_train_val = train_data.loc[symbol]['Z_SCORE'].iloc[-1]
            
            # Actual value at day 14 of test
            actual_val = test_data.loc[symbol]['Z_SCORE'].iloc[-1]
            
            # Predicted value at day 14
            pred_val = predictions.loc[symbol]['mean'].iloc[-1]
            
            # Directional moves
            actual_dir = 1 if actual_val > last_train_val else -1
            pred_dir = 1 if pred_val > last_train_val else -1
            
            if actual_dir == pred_dir:
                hits += 1
            total += 1
        except Exception:
            continue
            
    return (hits / total) * 100 if total > 0 else 0.0


def run_baseline_forecast(train_data: TimeSeriesDataFrame, test_data: TimeSeriesDataFrame, prediction_length: int = 14):
    """
    Initialize the AutoGluon TimeSeriesPredictor, fit the Chronos model,
    evaluate on the held-out test set, and plot the baseline predictions.
    """
    print(f"\n[3/3] Running Zero-Shot Baseline Forecast (amazon/chronos-t5-large)...")
    
    # Initialize Predictor with explicit Daily frequency
    predictor = TimeSeriesPredictor(
        target="Z_SCORE",
        prediction_length=prediction_length,
        eval_metric="MASE",
        freq="D"
    )
    
    # Fit the model using the train data
    predictor.fit(
        train_data,
        enable_ensemble=False,
        hyperparameters={
            "Chronos": {"model_path": "amazon/chronos-t5-large"}
        }
    )
    
    # Evaluate model performance on the HELD-OUT test set
    print("\n" + "="*50)
    print("  MODEL PERFORMANCE REPORT")
    print("="*50)
    
    leaderboard = predictor.leaderboard(test_data)
    print(leaderboard)
    
    # Save Leaderboard
    lb_path = ROOT / "data" / "chronos_leaderboard.csv"
    leaderboard.to_csv(lb_path, index=False)
    
    # Generate predictions for all assets
    print(f"\nGenerating {prediction_length}-day forecast for all assets...")
    predictions = predictor.predict(train_data)
    
    # Calculate Directional Accuracy
    dir_acc = calculate_directional_accuracy(train_data, test_data, predictions)
    print(f"\n[METRIC] Directional Accuracy (Hit Rate): {dir_acc:.2f}%")
    print("="*50)

    # Save Metrics
    metrics = {
        "directional_accuracy_hit_rate": dir_acc,
        "prediction_length": prediction_length,
        "model_path": "amazon/chronos-t5-large",
        "rows_processed": len(test_data),
        "unique_symbols": len(train_data.item_ids)
    }
    metrics_path = ROOT / "data" / "forecast_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Results saved to: {lb_path} and {metrics_path}")    
    # Plotting the 14-day prediction for the first available SYMBOL
    first_symbol = train_data.item_ids[0]
    
    plt.figure(figsize=(15, 6))
    
    # Plot history (last 100 days)
    history = train_data.loc[first_symbol][-100:]
    plt.plot(history.index, history['Z_SCORE'], label="Historical Z-Score (Train)")
    
    # Plot actuals (the held-out 14 days)
    actuals = test_data.loc[first_symbol][-14:]
    plt.plot(actuals.index, actuals['Z_SCORE'], label="Actual Z-Score (Held-out)", color='black', linewidth=2)
    
    # Plot prediction bounds
    pred = predictions.loc[first_symbol]
    plt.plot(pred.index, pred['mean'], label="Chronos Forecast (Mean)", color='red', linestyle='--')
    plt.fill_between(
        pred.index,
        pred['0.1'],
        pred['0.9'],
        color='red',
        alpha=0.2,
        label="80% Prediction Interval"
    )
    
    plt.title(f"Chronos-T5-Large Zero-Shot Forecast vs Actuals | Asset: {first_symbol}")
    plt.ylabel("Z-Score (Volatility)")
    plt.xlabel("Date")
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = ROOT / "data" / "chronos_baseline_plot.png"
    plt.savefig(plot_path)
    print(f"\nForecast plot saved successfully to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    # Prevent multi-processing errors on Windows
    # 1. Fetch
    df_raw = fetch_prototype_data(product='Unleaded Gasoline')
    
    # 2. Format & Split
    train_data, test_data = format_autogluon_data(df_raw)
    
    # 3. Forecast
    run_baseline_forecast(train_data, test_data, prediction_length=14)
