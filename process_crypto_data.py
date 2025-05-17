import pandas as pd
import numpy as np
from datetime import datetime
import os

def calculate_technical_indicators(df):
    """Calculate technical indicators for the dataframe."""
    # Ensure date is datetime, coerce errors
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop rows where date could not be parsed
    n_before = len(df)
    df = df.dropna(subset=['date'])
    n_after = len(df)
    if n_after < n_before:
        print(f"  [!] Dropped {n_before - n_after} rows due to unparseable dates.")
    
    # Convert relevant columns to numeric, coerce errors
    numeric_cols = ['open', 'high', 'low', 'close', 'Volume USD']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Drop rows where any of these columns are NaN
    n_before = len(df)
    df = df.dropna(subset=numeric_cols)
    n_after = len(df)
    if n_after < n_before:
        print(f"  [!] Dropped {n_before - n_after} rows due to non-numeric values in price/volume columns.")
    
    # Sort by date
    df = df.sort_values('date')
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Calculate volatility (20-day rolling standard deviation of returns)
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Calculate volume change (using Volume USD)
    df['volume_change'] = df['Volume USD'].pct_change()
    
    # Calculate moving averages
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_30'] = df['close'].rolling(window=30).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['ma_200'] = df['close'].rolling(window=200).mean()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Calculate Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    return df

def process_crypto_data(file_path):
    """Process a single cryptocurrency data file."""
    # Read the CSV file, skip blank lines
    df = pd.read_csv(file_path, skip_blank_lines=True)
    
    # Basic data cleaning
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Drop rows with NaN values
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    if n_after < n_before:
        print(f"  [!] Dropped {n_before - n_after} rows due to NaNs in features.")
    
    return df

def main():
    # Create processed_data directory if it doesn't exist
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    # List of cryptocurrencies to process
    cryptos = ['BTC', 'ETH', 'LTC', 'TRX', 'EOC_coin']
    
    # Process each cryptocurrency
    for crypto in cryptos:
        print(f"Processing {crypto}...")
        try:
            df = process_crypto_data(f"{crypto}.csv")
            output_path = os.path.join('processed_data', f"{crypto}.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved processed data for {crypto} to {output_path}")
            print(f"Shape: {df.shape}")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        except Exception as e:
            print(f"  [ERROR] Failed to process {crypto}: {e}")
        print("---")

if __name__ == "__main__":
    main() 