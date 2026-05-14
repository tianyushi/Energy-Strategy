"""
Phase 1: The "Finder" (Temporal Graph Neural Network)

This script implements an automated pipeline using a TGNN to mathematically discover 
hidden market signals (influencing contracts) for a target Unleaded Gasoline contract.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utility.snowflake_client import SnowflakeClient

SOURCE_TABLE = "CMDTYA.PUBLIC.PRICEDATA_ML_DAILY_SUMMARY"

def fetch_and_pivot_data(product: str = 'Unleaded Gasoline') -> pd.DataFrame:
    print(f"[1/4] Fetching Z-Score data in Snowflake for {product}...")
    
    query = f"""
        SELECT 
            SYMBOL, ASSESSDATE, Z_SCORE
        FROM {SOURCE_TABLE}
        WHERE PRODUCT = '{product}' AND ASSESSDATE >= '2020-01-01'
        ORDER BY SYMBOL, ASSESSDATE
    """
    
    with SnowflakeClient() as sf:
        sf.connect()
        df = sf.read_sql(query)
        
    df['ASSESSDATE'] = pd.to_datetime(df['ASSESSDATE'])
    
    print("[2/4] Pivoting data to cross-sectional (wide) format...")
    pivot_df = df.pivot(index='ASSESSDATE', columns='SYMBOL', values='Z_SCORE')
    
    # The Python Liquidity Filter
    # Drop any symbol that contains more than 20% missing values (NaNs) or 0s
    valid_mask = (pivot_df.notna()) & (pivot_df != 0)
    active_symbols = valid_mask.mean()[valid_mask.mean() >= 0.8].index.tolist()
    pivot_df = pivot_df[active_symbols]
    print(f"      Reduced graph to {len(active_symbols)} highly active symbols.")
    
    pivot_df = pivot_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pivot_df

def construct_graph_and_target(pivot_df: pd.DataFrame, horizon: int = 10, corr_threshold: float = 0.90):
    print(f"[3/4] Constructing Graph and Masking Highly Correlated Pairs...")
    
    # Select the most volatile contract as the target 'y'
    symbol_variances = pivot_df.var()
    target_symbol = symbol_variances.idxmax()
    symbols = pivot_df.columns.tolist()
    target_idx = symbols.index(target_symbol)
    
    print(f"      Target Sync: Locked onto Most Volatile Symbol: {target_symbol}")
    
    # Calculate the adjacency correlation mask using the first-difference of the Z-scores
    corr_matrix = pivot_df.diff().corr(method='pearson').abs()
    
    # Create adjacency mask: 1 if connected, 0 if unconnected
    # We disconnect pairs with correlation > threshold (no economic logic to find same thing)
    adjacency = (corr_matrix <= corr_threshold).astype(float).values
    # Ensure self connections
    np.fill_diagonal(adjacency, 1.0)
    
    # Create Target Label for regression (Raw Z-Score of next day)
    pivot_df['TARGET_VALUE'] = pivot_df[target_symbol].shift(-1)
    
    # Prepare Sequences
    sequence_length = 10 # 10 days of history
    features = pivot_df[symbols].values # (Time, Nodes)
    labels = pivot_df['TARGET_VALUE'].values
    
    X, y = [], []
    for i in range(len(features) - sequence_length - 1):
        X.append(features[i:i+sequence_length, :])
        y.append(labels[i+sequence_length-1])
        
    X = np.array(X) # (Samples, Seq_Len, Nodes)
    y = np.array(y)
    
    return X, y, adjacency, target_idx, symbols

class TemporalGraphAttentionNetwork(nn.Module):
    def __init__(self, num_nodes, seq_len, hidden_dim=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Temporal Encoder: process each node's sequence independently
        # Input to GRU: (Batch * Num_Nodes, Seq_Len, 1)
        self.temporal_encoder = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Graph Attention Weights
        self.W_query = nn.Linear(hidden_dim, hidden_dim)
        self.W_key = nn.Linear(hidden_dim, hidden_dim)
        self.W_value = nn.Linear(hidden_dim, hidden_dim)
        
        # Predictor Head (takes Target Node's embedded state)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x, adjacency_mask):
        # x shape: (Batch, Seq_Len, Num_Nodes)
        batch_size = x.size(0)
        
        # Reshape for Temporal Encoder -> (Batch * Num_Nodes, Seq_Len, 1)
        x_reshaped = x.permute(0, 2, 1).contiguous().view(batch_size * self.num_nodes, -1, 1)
        
        # Encode Temporally
        _, h_n = self.temporal_encoder(x_reshaped)
        # h_n shape: (1, Batch * Num_Nodes, Hidden_Dim)
        
        # Reshape back to (Batch, Num_Nodes, Hidden_Dim)
        node_embeddings = h_n.squeeze(0).view(batch_size, self.num_nodes, self.hidden_dim)
        node_embeddings = self.layer_norm(node_embeddings)
        
        # --- Graph Attention Layer ---
        Q = self.W_query(node_embeddings) # (B, N, H)
        K = self.W_key(node_embeddings)   # (B, N, H)
        V = self.W_value(node_embeddings) # (B, N, H)
        
        # Attention scores: Q * K^T
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5) # (B, N, N)
        
        # Apply Adjacency Mask (0s in adjacency become -10000.0, 1s stay 0)
        # adjacency_mask is (N, N)
        mask = (1.0 - adjacency_mask) * -10000.0
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1) # (B, N, N)
        
        scores = scores + mask
        attention_weights = torch.softmax(scores, dim=-1) # (B, N, N)
        
        # Aggregate information
        updated_nodes = torch.bmm(attention_weights, V) # (B, N, H)
        
        return updated_nodes, attention_weights

def train_and_extract_signals(X, y, adjacency, target_idx, symbols):
    print(f"[4/4] Training TGNN and Extracting Signals...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      Using device: {device}")
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    adjacency_tensor = torch.FloatTensor(adjacency).to(device)
    
    model = TemporalGraphAttentionNetwork(num_nodes=len(symbols), seq_len=X.shape[1], hidden_dim=32).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scaler = torch.cuda.amp.GradScaler()
    
    model_path = ROOT / "data" / "tgnn_model.pth"
    if model_path.exists():
        print(f"      [MODEL] Loading pre-trained model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        epochs = 4
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # Guard against inf/nan propagation from float16 overflow
                batch_x = torch.nan_to_num(batch_x, nan=0.0, posinf=0.0, neginf=0.0)
                batch_y = torch.nan_to_num(batch_y, nan=0.0, posinf=0.0, neginf=0.0)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    updated_nodes, _ = model(batch_x, adjacency_tensor)
                    
                    # Predict only for target node
                    target_node_state = updated_nodes[:, target_idx, :]
                    preds = model.predictor(target_node_state).squeeze(-1)
                    
                    loss = criterion(preds, batch_y)
                    
                scaler.scale(loss).backward()
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                
                del loss, updated_nodes
                gc.collect()
                torch.cuda.empty_cache()
                
            print(f"      Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
            
        print(f"      [MODEL] Saving trained model to {model_path}...")
        torch.save(model.state_dict(), model_path)
        
    print("\nExtracting 'Finder' Signals (Attention Weights) on Test Data...")
    model.eval()
    
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    test_preds_list = []
    attention_weights_sum = None
    
    with torch.no_grad():
        for test_batch_x, _ in test_loader:
            test_batch_x = test_batch_x.to(device)
            with torch.cuda.amp.autocast():
                updated_nodes, attention_weights = model(test_batch_x, adjacency_tensor)
                
                # Predict regression target for test set
                target_node_state = updated_nodes[:, target_idx, :]
                preds = model.predictor(target_node_state).squeeze(-1).float().cpu().numpy()
                test_preds_list.extend(preds)
                
                # Accumulate attention weights for the Target Node
                att_weights = attention_weights[:, target_idx, :].sum(dim=0).float().cpu().numpy()
                if attention_weights_sum is None:
                    attention_weights_sum = att_weights
                else:
                    attention_weights_sum += att_weights
        
    test_preds = np.array(test_preds_list)
    avg_attention = attention_weights_sum / len(X_test)
    
    influence_df = pd.DataFrame({
        'Symbol': symbols,
        'Influence_Weight': avg_attention,
        'Correlation_with_Target': adjacency[target_idx] # 1 if connected, 0 if masked
    })
    
    # Filter out the target itself and zeroed out (highly correlated) ones
    influence_df = influence_df[(influence_df['Symbol'] != symbols[target_idx]) & (influence_df['Influence_Weight'] > 0)]
    influence_df = influence_df.sort_values(by='Influence_Weight', ascending=False)
    
    print(f"\n========================================================")
    print(f" MATHEMATICAL DISCOVERY: TOP INFLUENCERS FOR {symbols[target_idx]}")
    print(f"========================================================")
    print(influence_df.head(15).to_string(index=False))
    
    print(f"\n========================================================")
    print(f" REGRESSION FORECAST (Last 5 days of Test Set)")
    print(f"========================================================")
    for i in range(1, 6):
        pred_z = test_preds[-i]
        actual_z = y_test[-i]
        print(f"Day -{i}: Predicted Z-Score {pred_z:.2f} | Actual Z-Score: {actual_z:.2f}")

if __name__ == "__main__":
    cache_file = ROOT / "data" / "tgnn_cache.npz"
    if cache_file.exists():
        print("\n[CACHE] Loading Step 3 data from cache...")
        data = np.load(cache_file, allow_pickle=True)
        X = data['X']
        y = data['y']
        adjacency = data['adjacency']
        target_idx = int(data['target_idx'])
        symbols = data['symbols'].tolist()
    else:
        df_wide = fetch_and_pivot_data()
        X, y, adjacency, target_idx, symbols = construct_graph_and_target(df_wide)
        print("\n[CACHE] Saving Step 3 data to cache...")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, X=X, y=y, adjacency=adjacency, target_idx=target_idx, symbols=symbols)
        
    train_and_extract_signals(X, y, adjacency, target_idx, symbols)
