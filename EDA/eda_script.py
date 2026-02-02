import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Settings
INPUT_FILE = '/home/ubuntu/clawd/datasets/ETF_data/Greed-and-Fear/data/volatility_dataset_013026.csv'
OUTPUT_DIR = '/home/ubuntu/clawd/datasets/ETF_data/Greed-and-Fear/EDA'
sns.set_theme(style="whitegrid")

print(f"Loading data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

# Convert date to datetime with flexible parsing
print("Converting dates...")
df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
# Drop rows with invalid dates if any
if df['date'].isnull().any():
    print(f"Warning: Dropped {df['date'].isnull().sum()} rows with invalid dates")
    df = df.dropna(subset=['date'])
print(f"Data loaded. Shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# 1. Overview of Tickers
ticker_counts = df['ticker'].value_counts()
print(f"Top 5 tickers by row count:\n{ticker_counts.head()}")
top_tickers = ticker_counts.head(4).index.tolist()

# 2. Time Series: Adjusted Close Price for Top Tickers
plt.figure(figsize=(12, 6))
for ticker in top_tickers:
    subset = df[df['ticker'] == ticker].sort_values('date')
    plt.plot(subset['date'], subset['adj_close'], label=ticker)

plt.title('Adjusted Close Price Over Time (Top Tickers)')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1_price_history.png'))
print("Saved 1_price_history.png")
plt.close()

# 3. Time Series: Volatility Comparison (SPY only for clarity)
# comparing 20d Factset vs Calculated
if 'SPY' in df['ticker'].unique():
    spy_data = df[df['ticker'] == 'SPY'].sort_values('date')
    plt.figure(figsize=(12, 6))
    
    # Plotting trailing annual volatility (20d)
    plt.plot(spy_data['date'], spy_data['trailing_vol_annual_decimel_20d_factset'], 
             label='FactSet 20d Annual Vol', alpha=0.7)
    plt.plot(spy_data['date'], spy_data['trailing_vol__annual_decimel_20d_calculated'], 
             label='Calculated 20d Annual Vol', alpha=0.7, linestyle='--')
    
    plt.title('SPY: 20-Day Trailing Annualized Volatility Comparison')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Decimal)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_spy_volatility_comparison.png'))
    print("Saved 2_spy_volatility_comparison.png")
    plt.close()

# 4. Distribution of Daily Log Returns
plt.figure(figsize=(10, 6))
# Filter out extreme outliers for better visualization if needed, but histogram usually handles it
sns.histplot(data=df[df['ticker'].isin(top_tickers)], x='daily_log_return', hue='ticker', kde=True, bins=50)
plt.title('Distribution of Daily Log Returns (Top Tickers)')
plt.xlabel('Daily Log Return')
plt.xlim(-0.1, 0.1) # Limiting x-axis to focus on the main distribution
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '3_returns_distribution.png'))
print("Saved 3_returns_distribution.png")
plt.close()

# 5. Correlation Heatmap (Numerical columns)
# Select relevant numerical columns for correlation
cols_to_corr = [
    'adj_close', 'volume', 
    'trailing_vol_daily_pct_20d_factset', 
    'trailing_vol__annual_decimel_20d_calculated',
    'daily_log_return'
]
corr_matrix = df[cols_to_corr].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Key Metrics')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4_correlation_heatmap.png'))
print("Saved 4_correlation_heatmap.png")
plt.close()

# 6. Volatility vs Returns Scatter (Risk/Return profile daily)
plt.figure(figsize=(10, 6))
# Sample a subset if data is too huge to prevent overplotting issues
subset_scatter = df[df['ticker'].isin(top_tickers)].sample(frac=0.5, random_state=42)
sns.scatterplot(data=subset_scatter, x='trailing_vol_annual_decimel_20d_factset', y='daily_log_return', hue='ticker', alpha=0.5)
plt.title('Daily Return vs. 20d Volatility (Scatter)')
plt.xlabel('20d Trailing Annual Volatility (FactSet)')
plt.ylabel('Daily Log Return')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5_vol_vs_return_scatter.png'))
print("Saved 5_vol_vs_return_scatter.png")
plt.close()

print("EDA analysis complete.")
