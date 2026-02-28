#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
    df_ticker['target_vol_5d'] = df_ticker['daily_log_return'].pow(2).rolling(window=indexer).sum().pow(0.5) * np.sqrt(252)
    
    # 3. Feature: Trailing 20-day realized volatility
    df_ticker['trailing_vol_20d'] = df_ticker['daily_log_return'].pow(2).rolling(window=20).sum().pow(0.5) * np.sqrt(252)
    
    # 4. Feature: Trailing 5-day realized volatility
    df_ticker['trailing_vol_5d'] = df_ticker['daily_log_return'].pow(2).rolling(window=5).sum().pow(0.5) * np.sqrt(252)

    # Define Input Features (X) and Target (y)
    feature_cols = ['trailing_vol_20d', 'trailing_vol_5d']
    target_col = 'target_vol_5d'

    # Drop NaN values created by rolling windows
    df_clean = df_ticker.dropna(subset=feature_cols + [target_col]).copy()
    
    print(f"Data range after filtering: {df_clean['date'].min()} to {df_clean['date'].max()}")
    print(f"Total rows: {len(df_clean)}")

    return df_clean, feature_cols, target_col

def create_sequences(X, y, time_steps=1):
    """Create sequences for LSTM and Transformer models."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_linear_model(input_dim):
    """Build a simple linear regression model (Fully Connected Network baseline)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(input_dim,), activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model

def build_lstm_model(input_shape):
    """Build an LSTM model for sequence prediction."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Build a transformer encoder block."""
    # Multi-head attention
    x = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + inputs)

    # Feed forward network
    outputs = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    outputs = tf.keras.layers.Dropout(dropout)(outputs)
    outputs = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + x)

    return outputs

def build_transformer_model(
    input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, 
    dropout=0, mlp_dropout=0
):
    """Build a Transformer model for sequence prediction."""
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    # Stack transformer encoder blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Global average pooling and dense layers
    x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)
    for dim in mlp_units:
        x = tf.keras.layers.Dense(dim, activation='relu')(x)
        x = tf.keras.layers.Dropout(mlp_dropout)(x)
    
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model

def main():
    """Main execution function."""
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    df = load_data(DATA_PATH)
    
    # 2. Prepare Features
    df_clean, feature_cols, target_col = prepare_features(df)
    
    # 3. Split Data (Time-based split)
    train_mask = (df_clean['date'].dt.year >= 2015) & (df_clean['date'].dt.year <= 2021)
    val_mask = (df_clean['date'].dt.year >= 2022) & (df_clean['date'].dt.year <= 2023)
    test_mask = (df_clean['date'].dt.year >= 2024) & (df_clean['date'].dt.year <= 2025)
    
    X_train_df = df_clean.loc[train_mask, feature_cols]
    y_train_df = df_clean.loc[train_mask, target_col]
    X_val_df = df_clean.loc[val_mask, feature_cols]
    y_val_df = df_clean.loc[val_mask, target_col]
    X_test_df = df_clean.loc[test_mask, feature_cols]
    y_test_df = df_clean.loc[test_mask, target_col]

    print(f"\nTrain shapes: X={X_train_df.shape}, y={y_train_df.shape}")
    print(f"Val shapes:   X={X_val_df.shape}, y={y_val_df.shape}")
    print(f"Test shapes:  X={X_test_df.shape}, y={y_test_df.shape}")

    # 4. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)
    X_test_scaled = scaler.transform(X_test_df)

    y_train = y_train_df.values
    y_val = y_val_df.values
    y_test = y_test_df.values

    # --- Model Training and Evaluation ---
    results = {}

    # ========== Model 1: Baseline Fully Connected Network ==========
    print("\n" + "="*60)
    print("Training Model 1: Baseline Fully Connected Network")
    print("="*60)
    baseline_model = build_linear_model(X_train_scaled.shape[1])
    baseline_model.summary()
    baseline_model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), 
                       epochs=50, batch_size=32, verbose=0)
    y_pred_baseline = baseline_model.predict(X_test_scaled, verbose=0).flatten()
    mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
    rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    results['Baseline FCN'] = {'MAE': mae_baseline, 'RMSE': rmse_baseline}
    print(f"Baseline FCN - MAE: {mae_baseline:.6f}, RMSE: {rmse_baseline:.6f}")

    # ========== Model 2: Random Forest ==========
    print("\n" + "="*60)
    print("Training Model 2: Random Forest")
    print("="*60)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    results['Random Forest'] = {'MAE': mae_rf, 'RMSE': rmse_rf}
    print(f"Random Forest - MAE: {mae_rf:.6f}, RMSE: {rmse_rf:.6f}")

    # ========== Model 3: XGBoost ==========
    print("\n" + "="*60)
    print("Training Model 3: XGBoost")
    print("="*60)
    xgb_model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=1)
    xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    results['XGBoost'] = {'MAE': mae_xgb, 'RMSE': rmse_xgb}
    print(f"XGBoost - MAE: {mae_xgb:.6f}, RMSE: {rmse_xgb:.6f}")

    # ========== Prepare Sequence Data for LSTM and Transformer ==========
    print("\n" + "="*60)
    print("Preparing Sequence Data for LSTM and Transformer")
    print("="*60)
    time_steps = 5  # Use 5-day sequences
    X_train_seq, y_train_seq = create_sequences(X_train_df, y_train_df, time_steps)
    X_val_seq, y_val_seq = create_sequences(X_val_df, y_val_df, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test_df, y_test_df, time_steps)

    print(f"Sequence shapes:")
    print(f"  Train: X={X_train_seq.shape}, y={y_train_seq.shape}")
    print(f"  Val:   X={X_val_seq.shape}, y={y_val_seq.shape}")
    print(f"  Test:  X={X_test_seq.shape}, y={y_test_seq.shape}")

    # Scale sequence data
    scaler_seq = StandardScaler()
    X_train_seq_flat = X_train_seq.reshape(-1, X_train_seq.shape[-1])
    X_train_seq_scaled = scaler_seq.fit_transform(X_train_seq_flat).reshape(X_train_seq.shape)
    
    X_val_seq_flat = X_val_seq.reshape(-1, X_val_seq.shape[-1])
    X_val_seq_scaled = scaler_seq.transform(X_val_seq_flat).reshape(X_val_seq.shape)
    
    X_test_seq_flat = X_test_seq.reshape(-1, X_test_seq.shape[-1])
    X_test_seq_scaled = scaler_seq.transform(X_test_seq_flat).reshape(X_test_seq.shape)

    # ========== Model 4: LSTM ==========
    print("\n" + "="*60)
    print("Training Model 4: LSTM")
    print("="*60)
    lstm_model = build_lstm_model((X_train_seq_scaled.shape[1], X_train_seq_scaled.shape[2]))
    lstm_model.summary()
    lstm_model.fit(X_train_seq_scaled, y_train_seq, 
                   validation_data=(X_val_seq_scaled, y_val_seq), 
                   epochs=50, batch_size=32, verbose=0)
    y_pred_lstm = lstm_model.predict(X_test_seq_scaled, verbose=0).flatten()
    mae_lstm = mean_absolute_error(y_test_seq, y_pred_lstm)
    rmse_lstm = np.sqrt(mean_squared_error(y_test_seq, y_pred_lstm))
    results['LSTM'] = {'MAE': mae_lstm, 'RMSE': rmse_lstm}
    print(f"LSTM - MAE: {mae_lstm:.6f}, RMSE: {rmse_lstm:.6f}")

    # ========== Model 5: Transformer ==========
    print("\n" + "="*60)
    print("Training Model 5: Transformer")
    print("="*60)
    transformer_model = build_transformer_model(
        input_shape=(X_train_seq_scaled.shape[1], X_train_seq_scaled.shape[2]),
        head_size=256,
        num_heads=4,
        ff_dim=128,
        num_transformer_blocks=2,
        mlp_units=[64],
        dropout=0.1,
        mlp_dropout=0.1
    )
    transformer_model.summary()
    transformer_model.fit(X_train_seq_scaled, y_train_seq, 
                          validation_data=(X_val_seq_scaled, y_val_seq), 
                          epochs=50, batch_size=32, verbose=0)
    y_pred_transformer = transformer_model.predict(X_test_seq_scaled, verbose=0).flatten()
    mae_transformer = mean_absolute_error(y_test_seq, y_pred_transformer)
    rmse_transformer = np.sqrt(mean_squared_error(y_test_seq, y_pred_transformer))
    results['Transformer'] = {'MAE': mae_transformer, 'RMSE': rmse_transformer}
    print(f"Transformer - MAE: {mae_transformer:.6f}, RMSE: {rmse_transformer:.6f}")

    # ========== Model Comparison ==========
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON RESULTS")
    print("="*60)
    results_df = pd.DataFrame(results).T
    print(results_df)
    print("\n" + "="*60)

    # Save results to CSV
    results_csv_path = os.path.join(OUTPUT_DIR, 'model_comparison_results.csv')
    results_df.to_csv(results_csv_path)
    print(f"Results saved to {results_csv_path}")

    # ========== Visualization ==========
    print("\nGenerating comparison visualizations...")
    
    # Plot 1: Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE comparison
    results_df['MAE'].plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Mean Absolute Error (MAE) Comparison', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('MAE', fontsize=11)
    axes[0].set_xlabel('Model', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # RMSE comparison
    results_df['RMSE'].plot(kind='bar', ax=axes[1], color='coral')
    axes[1].set_title('Root Mean Squared Error (RMSE) Comparison', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('RMSE', fontsize=11)
    axes[1].set_xlabel('Model', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    comparison_plot_path = os.path.join(OUTPUT_DIR, 'model_performance_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {comparison_plot_path}")
    plt.close()

    # Plot 2: Prediction scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Predicted vs Actual Volatility (Test Set)', fontsize=14, fontweight='bold')
    
    # Baseline FCN
    axes[0, 0].scatter(y_test, y_pred_baseline, alpha=0.5, s=20)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('Baseline FCN')
    axes[0, 0].grid(alpha=0.3)
    
    # Random Forest
    axes[0, 1].scatter(y_test, y_pred_rf, alpha=0.5, s=20, color='green')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].set_title('Random Forest')
    axes[0, 1].grid(alpha=0.3)
    
    # XGBoost
    axes[0, 2].scatter(y_test, y_pred_xgb, alpha=0.5, s=20, color='purple')
    axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 2].set_xlabel('Actual')
    axes[0, 2].set_ylabel('Predicted')
    axes[0, 2].set_title('XGBoost')
    axes[0, 2].grid(alpha=0.3)
    
    # LSTM
    axes[1, 0].scatter(y_test_seq, y_pred_lstm, alpha=0.5, s=20, color='orange')
    axes[1, 0].plot([y_test_seq.min(), y_test_seq.max()], [y_test_seq.min(), y_test_seq.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual')
    axes[1, 0].set_ylabel('Predicted')
    axes[1, 0].set_title('LSTM')
    axes[1, 0].grid(alpha=0.3)
    
    # Transformer
    axes[1, 1].scatter(y_test_seq, y_pred_transformer, alpha=0.5, s=20, color='brown')
    axes[1, 1].plot([y_test_seq.min(), y_test_seq.max()], [y_test_seq.min(), y_test_seq.max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual')
    axes[1, 1].set_ylabel('Predicted')
    axes[1, 1].set_title('Transformer')
    axes[1, 1].grid(alpha=0.3)
    
    # Hide the last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    scatter_plot_path = os.path.join(OUTPUT_DIR, 'prediction_scatter_comparison.png')
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to {scatter_plot_path}")
    plt.close()

    print("\n" + "="*60)
    print("Model comparison completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
