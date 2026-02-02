import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.preprocessing import StandardScaler

# Define paths
DATA_PATH = '/home/ubuntu/clawd/datasets/ETF_data/Greed-and-Fear/data/volatility_dataset_013026.csv'
OUTPUT_DIR = '/home/ubuntu/clawd/datasets/ETF_data/Greed-and-Fear/Code'

def load_data(filepath):
    """Loads and preprocesses the ETF dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    
    # Convert date to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date')
    
    return df

def prepare_features(df):
    """
    Prepares features and target for the volatility prediction task.
    Target: Next 5-day realized volatility (annualized).
    Features: Trailing realized volatility (5, 20 days), log returns, volume.
    """
    # Filter for a specific ticker to simplify the baseline demonstration (e.g., SPY)
    # In a full pipeline, we would process all tickers.
    ticker = 'SPY'
    print(f"Filtering data for {ticker} for baseline demonstration...")
    df_ticker = df[df['ticker'] == ticker].copy()
    
    # Ensure data is sorted by date
    df_ticker = df_ticker.sort_values('date')
    
    # --- Feature Engineering ---
    # 1. Calculate Daily Log Returns if not present or rely on 'daily_log_return'
    if 'daily_log_return' not in df_ticker.columns:
        df_ticker['daily_log_return'] = np.log(df_ticker['adj_close'] / df_ticker['adj_close'].shift(1))
    
    # 2. Target: Next-5-trading-day realized volatility (annualized)
    # Formula: sqrt(sum(r_t+j^2)) * sqrt(252/5)
    # We use a rolling window shifted backwards to create the future target
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
    df_ticker['target_vol_5d'] = df_ticker['daily_log_return'].pow(2).rolling(window=indexer).sum().pow(0.5) * np.sqrt(252)
    
    # 3. Feature: Trailing 20-day realized volatility (Persistence Baseline Feature)
    # Formula: sqrt(sum(r_t-j^2)) * sqrt(252/20)
    df_ticker['trailing_vol_20d'] = df_ticker['daily_log_return'].pow(2).rolling(window=20).sum().pow(0.5) * np.sqrt(252)
    
    # 4. Feature: Trailing 5-day realized volatility
    df_ticker['trailing_vol_5d'] = df_ticker['daily_log_return'].pow(2).rolling(window=5).sum().pow(0.5) * np.sqrt(252)

    # Drop NaN values created by rolling windows
    df_clean = df_ticker.dropna().copy()
    
    # Define Input Features (X) and Target (y)
    # Baseline uses 'trailing_vol_20d' as the primary predictor
    feature_cols = ['trailing_vol_20d', 'trailing_vol_5d']
    target_col = 'target_vol_5d'
    
    return df_clean, feature_cols, target_col

def create_dataset(X, y, batch_size=32):
    """Creates a tf.data.Dataset from numpy arrays."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_linear_model(input_dim):
    """
    Builds a Linear Regression model in TensorFlow.
    Corresponds to 'Model 1' in the proposal.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(input_dim,), activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model

def main():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    df = load_data(DATA_PATH)
    
    # 2. Prepare Features
    df_clean, feature_cols, target_col = prepare_features(df)
    
    # 3. Split Data (Time-based split as per proposal)
    # Train (2015–2021), Validation (2022–2023), Test (2024–2025)
    train_mask = (df_clean['date'].dt.year >= 2015) & (df_clean['date'].dt.year <= 2021)
    val_mask = (df_clean['date'].dt.year >= 2022) & (df_clean['date'].dt.year <= 2023)
    test_mask = (df_clean['date'].dt.year >= 2024) & (df_clean['date'].dt.year <= 2025)
    
    X_train = df_clean.loc[train_mask, feature_cols].values
    y_train = df_clean.loc[train_mask, target_col].values
    
    X_val = df_clean.loc[val_mask, feature_cols].values
    y_val = df_clean.loc[val_mask, target_col].values
    
    X_test = df_clean.loc[test_mask, feature_cols].values
    y_test = df_clean.loc[test_mask, target_col].values
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shapes:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test shapes:  X={X_test.shape}, y={y_test.shape}")
    
    # 4. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Create TF Datasets
    train_ds = create_dataset(X_train_scaled, y_train)
    val_ds = create_dataset(X_val_scaled, y_val)
    
    # 6. Build and Train Linear Model (Model 1)
    print("\n--- Training Model 1: Linear Regression (TensorFlow) ---")
    model = build_linear_model(input_dim=len(feature_cols))
    model.summary()
    
    history = model.fit(train_ds, epochs=50, validation_data=val_ds, verbose=1)
    
    # 7. Evaluation
    print("\n--- Evaluation on Test Set ---")
    # A. Heuristic Baseline: Persistence (Predict next 5d vol = current 20d vol)
    # Note: In the features, 'trailing_vol_20d' is at index 0
    y_pred_heuristic = X_test[:, 0] 
    mae_heuristic = np.mean(np.abs(y_test - y_pred_heuristic))
    rmse_heuristic = np.sqrt(np.mean((y_test - y_pred_heuristic)**2))
    
    print(f"Baseline (Persistence Heuristic):")
    print(f"  MAE:  {mae_heuristic:.4f}")
    print(f"  RMSE: {rmse_heuristic:.4f}")
    
    # B. Model 1 (Linear Regression)
    loss, mae_model1, mse_model1 = model.evaluate(X_test_scaled, y_test, verbose=0)
    rmse_model1 = np.sqrt(mse_model1)
    
    print(f"Model 1 (TF Linear Regression):")
    print(f"  MAE:  {mae_model1:.4f}")
    print(f"  RMSE: {rmse_model1:.4f}")
    
    # Save Model
    model_save_path = os.path.join(OUTPUT_DIR, 'baseline_model_tf.keras')
    model.save(model_save_path)
    print(f"\nModel saved to {model_save_path}")

if __name__ == "__main__":
    main()
