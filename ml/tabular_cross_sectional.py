"""
Tabular Cross-Sectional Forecast using AutoGluon TabularPredictor.

This script demonstrates how to predict the directional probability 
of a target symbol using the cross-section of all other symbols.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    print("[ERROR] autogluon.tabular is not installed.")
    print("Please run: pip install autogluon.tabular")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utility.snowflake_client import SnowflakeClient

SOURCE_TABLE = "CMDTYA.PUBLIC.PRICEDATA_ML_DAILY_SUMMARY"

def fetch_and_pivot_data(product: str = 'Unleaded Gasoline') -> pd.DataFrame:
    print(f"[1/4] Fetching data for {product}...")
    
    query = f"""
        SELECT SYMBOL, ASSESSDATE, Z_SCORE
        FROM {SOURCE_TABLE}
        WHERE PRODUCT = '{product}'
        ORDER BY SYMBOL, ASSESSDATE
        LIMIT 3000000
    """
    
    with SnowflakeClient() as sf:
        sf.connect()
        df = sf.read_sql(query)
        
    df['ASSESSDATE'] = pd.to_datetime(df['ASSESSDATE'])
    
    print("[2/4] Pivoting data to cross-sectional (wide) format...")
    # Pivot so rows are Dates, columns are Symbols
    pivot_df = df.pivot(index='ASSESSDATE', columns='SYMBOL', values='Z_SCORE')
    
    # Forward fill to handle days where some symbols traded but others didn't
    pivot_df = pivot_df.ffill()
    
    # Drop symbols with too much missing data (e.g. > 20% missing)
    thresh = len(pivot_df) * 0.8
    pivot_df = pivot_df.dropna(axis=1, thresh=thresh)
    
    # Fill remaining NaNs with 0 (mean since Z-score)
    pivot_df = pivot_df.fillna(0)
    
    print(f"      Wide DataFrame shape: {pivot_df.shape}")
    return pivot_df

def create_target_and_features(pivot_df: pd.DataFrame, target_symbol: str = None, horizon: int = 14) -> pd.DataFrame:
    print(f"[3/4] Engineering target and features...")
    
    # If no target provided, pick the symbol with the most non-zero variance
    if not target_symbol or target_symbol not in pivot_df.columns:
        split_idx = int(len(pivot_df) * 0.8)
        recent_var = pivot_df.iloc[split_idx:].var().sort_values(ascending=False)
        for potential_target in recent_var.index:
            fz = pivot_df[potential_target].shift(-horizon)
            cz = pivot_df[potential_target]
            direction = (fz > cz).astype(int)
            test_dir = direction.iloc[split_idx:-horizon]
            if len(test_dir) > 0 and test_dir.nunique() > 1:
                target_symbol = potential_target
                break
        print(f"      No valid target provided. Auto-selected active symbol: {target_symbol}")
    else:
        print(f"      Target symbol: {target_symbol}")
        
    print("      Eliminating highly correlated 'easy' pairs (Correlation > 0.95)...")
    corr_matrix = pivot_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    
    if target_symbol in to_drop:
        to_drop.remove(target_symbol)
        
    if to_drop:
        print(f"      Dropping {len(to_drop)} redundant/highly-correlated symbols.")
        pivot_df = pivot_df.drop(columns=to_drop)
        
    # Calculate target: 1 if Z_SCORE at t+14 > Z_SCORE at t, else 0
    # We shift the future value backwards by 'horizon' to align with today's features
    future_z = pivot_df[target_symbol].shift(-horizon)
    current_z = pivot_df[target_symbol]
    
    # Create the binary target
    pivot_df['TARGET_DIRECTION'] = (future_z > current_z).astype(int)
    
    # The last 'horizon' rows will have NaN for future_z, so we drop them for training
    ml_df = pivot_df.iloc[:-horizon].copy()
    
    return ml_df, target_symbol

def train_tabular_model(ml_df: pd.DataFrame, target_col: str = 'TARGET_DIRECTION'):
    print(f"[4/4] Training AutoGluon TabularPredictor...")
    
    # Chronological Split (last 20% for testing)
    split_idx = int(len(ml_df) * 0.8)
    train_data = ml_df.iloc[:split_idx]
    test_data = ml_df.iloc[split_idx:]
    
    print(f"      Train set: {len(train_data)} rows | Test set: {len(test_data)} rows")
    
    # We restrict to fast tree models for the POC to save memory/time
    predictor = TabularPredictor(
        label=target_col,
        eval_metric='roc_auc',
        path='AutogluonModels/tabular_poc'
    ).fit(
        train_data,
        hyperparameters={
            'GBM': {},      # LightGBM
            'XGB': {},      # XGBoost
        },
        time_limit=300,     # 5 minute limit for POC
        verbosity=2
    )
    
    print("\n" + "="*50)
    print("  TABULAR PERFORMANCE REPORT")
    print("="*50)
    
    leaderboard = predictor.leaderboard(test_data)
    print(leaderboard)
    
    print(f"\n[METRIC] ROC AUC on Test Set: {predictor.evaluate(test_data)['roc_auc']:.4f}")
    
    print("\nGenerating Directional Probabilities for Test Set...")
    probabilities = predictor.predict_proba(test_data)
    prob_output = pd.DataFrame({
        'Date': test_data.index,
        'Actual_Direction': test_data[target_col].values,
        'Probability_DOWN': probabilities[0].values,
        'Probability_UP': probabilities[1].values
    })
    prob_path = ROOT / "data" / "tabular_directional_probabilities.csv"
    prob_output.to_csv(prob_path, index=False)
    print(f"Directional probabilities saved to: {prob_path}")
    
    print("\nExtracting Feature Importances (Top 10 Drivers)...")
    importance = predictor.feature_importance(test_data)
    print(importance.head(10))
    
    # Save outputs
    imp_path = ROOT / "data" / "tabular_feature_importance.csv"
    imp_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(imp_path)
    print(f"Feature importance saved to: {imp_path}")

if __name__ == "__main__":
    df_wide = fetch_and_pivot_data()
    ml_df, chosen_target = create_target_and_features(df_wide, horizon=14)
    train_tabular_model(ml_df)
