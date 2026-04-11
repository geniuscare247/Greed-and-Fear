import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random

# Fix the random seed so we get the exact same results every time we run this.
# This eliminates variations from random weight initializations and data shuffling!
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # We also need these cudnn backend settings to ensure PyTorch relies on deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------------------------------------------------
# 1. Positional Encoding & Transformer Model (from the paper)
# ---------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Fix: using math.log instead of np.log to avoid numpy dependency in tensor building
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is expected to be of shape (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_length, max_len=24):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.encoder = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = nn.Linear(d_model, output_length)
        self.d_model = d_model
        self.activation = nn.Sigmoid()
        self.input_dim = input_dim

    def forward(self, src):
        # src: (sequence_length, batch_size, input_dim)
        PAD = -1.5
        
        # Calculate padding mask
        # src == PAD -> (seq_len, batch_size, input_dim)
        # .any(dim=-1) -> (seq_len, batch_size)
        # .t() -> (batch_size, seq_len)   (Matches Transformer mask expectation)
        padding_mask = (src == PAD).any(dim=-1).t() 
        
        # Ensure causality: apply upper triangular mask to prevent looking ahead
        seq_len = src.size(0)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        
        src = self.encoder(src) * math.sqrt(self.d_model) # using math.sqrt
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        # Average pooling representation across sequence length
        output = output.mean(dim=0)
        
        output = self.decoder(output)
        # Scales via user-specified logic
        output = self.activation(output) * 100
        return output

# ---------------------------------------------------------
# 2. Data Loading & Sequence Preparation
# ---------------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X_seq, y_seq):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y_seq, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(X_df, y_series, time_steps=5):
    """
    Ronald Note
    Given a dataframe X_df and target series y_series, 
    creates sliding window sequences of length `time_steps`.
    Because y_series at index `i` is forward-looking (calculates forward_vol_5d starting from t),
    a sequence using data [i - time_steps + 1 : i] maps perfectly to target at 'i', 
    ensuring strictly trailing and thus causal mapping from past data to target.
    """
    Xs, ys = [], []
    for i in range(time_steps - 1, len(X_df)):
        v = X_df.iloc[(i - time_steps + 1) : (i + 1)].values
        Xs.append(v)
        ys.append(y_series.iloc[i])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

def load_and_preprocess_split(file_path, features, target):
    df = pd.read_csv(file_path, low_memory=False)
    df = df.dropna(subset=features + [target])
    
    # Optional sorting by date and ticker, assuming df has 'date' or we just proceed chronologically.
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['ticker', 'date'])
        
    return df

# ---------------------------------------------------------
# 3. Main Routine
# ---------------------------------------------------------
if __name__ == "__main__":
    # Base configuration based on V2 target and features
    BASE_DIR = r"C:\Users\X\GearAndFear\Greed-and-Fear\data_splits"
    
    TRAIN_FILE = os.path.join(BASE_DIR, "vol_dataset_train_20150102_20221230.csv")
    VAL_FILE   = os.path.join(BASE_DIR, "vol_dataset_validation_20230103_20241231.csv")
    TEST_FILE  = os.path.join(BASE_DIR, "vol_dataset_test_20250102_20251231.csv")
    
    TARGET = 'forward_vol_5d_annual_decimel_calculated'
    
    # 8 features corresponding to Elastic_Net V2 logic
    FEATURES = [
        'y_known_at_t',
        'trailing_vol_annual_decimel_20d_calculated',
        'volume',
        'NYGOLDS',
        'OIL_WTI_S',
        'US_10Y_BOND_YLD',
        'US_3M_TB_YLD',
        'VIX'
    ]

    V1_FEATURES = [
        'y_known_at_t',
        'trailing_vol_annual_decimel_20d_calculated'
    ]

    print("Loading datasets...")
    train_df = load_and_preprocess_split(TRAIN_FILE, FEATURES, TARGET)
    val_df   = load_and_preprocess_split(VAL_FILE, FEATURES, TARGET)
    test_df  = load_and_preprocess_split(TEST_FILE, FEATURES, TARGET)
    
    print(f"Train Rows: {len(train_df)} | Val Rows: {len(val_df)} | Test Rows: {len(test_df)}")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    
    # Fit scaler only on train to prevent leakage
    train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
    val_df[FEATURES] = scaler.transform(val_df[FEATURES])
    test_df[FEATURES] = scaler.transform(test_df[FEATURES])
    
    # Make sequences
    TIME_STEPS = 5 #5, 10, 15

    print(f"Creating sequences (Time Steps = {TIME_STEPS})...")
    
    # Note: creating sequences for all tickers indiscriminately may mix tickers at boundaries.
    # We group by ticker to prevent boundary leakage.
    def build_dataset_by_ticker(df):
        X_all, y_all = [], []
        for ticker, group in df.groupby('ticker'):
            X_grp, y_grp = create_sequences(group[FEATURES], group[TARGET], time_steps=TIME_STEPS)
            if len(X_grp) > 0:
                X_all.append(X_grp)
                y_all.append(y_grp)
        if not X_all:
            return np.array([]), np.array([])
        return np.vstack(X_all), np.vstack(y_all)

    X_train_seq, y_train_seq = build_dataset_by_ticker(train_df)
    X_val_seq, y_val_seq = build_dataset_by_ticker(val_df)
    X_test_seq, y_test_seq = build_dataset_by_ticker(test_df)
    
    print(f"Train Seq: {X_train_seq.shape} | Val Seq: {X_val_seq.shape} | Test Seq: {X_test_seq.shape}")

    # DataLoaders
    batch_size = 128
    train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
    val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq)
    test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize PyTorch Transformer Architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TransformerModel(
        input_dim=len(FEATURES), 
        d_model=64, 
        nhead=4, 
        num_encoder_layers=2, 
        dim_feedforward=128, 
        output_length=1, 
        max_len=TIME_STEPS
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 11

    print("Beginning Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # DataLoader outputs (batch, seq_len, input_dim). 
            # Snippet expects (seq_len, batch, input_dim), so we transpose:
            batch_X = batch_X.transpose(0, 1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(1) # size(1) is batch here
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                batch_X = batch_X.transpose(0, 1)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(1)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Testing
    print("Evaluating on Test Set...")
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X = batch_X.transpose(0, 1)
            
            outputs = model(batch_X)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    all_preds = np.array(all_preds).reshape(-1)
    all_targets = np.array(all_targets).reshape(-1)
    
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = math.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    print(f"Test MAE: {mae:.6f} | Test RMSE: {rmse:.6f} | Test R^2: {r2:.6f}")
    
    print("Code Execution Completed.")
