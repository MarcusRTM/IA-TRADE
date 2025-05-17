import pandas as pd
import numpy as np
from datetime import datetime
import os

def process_crypto_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df = df.dropna(subset=['date'])
    
    # Filter data up to December 31, 2024
    cutoff_date = pd.Timestamp('2024-12-31')
    date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
    df = df[df[date_col] <= cutoff_date]
    
    # Explicitly convert relevant columns to numeric
    if 'close' in df.columns:
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
    if 'Volume USD' in df.columns:
        df['Volume USD'] = pd.to_numeric(df['Volume USD'], errors='coerce')
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Basic feature engineering
    if 'close' in df.columns:
        # Calculate price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_24h'] = df['close'].pct_change(periods=24)
        
        # Calculate moving averages
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_30'] = df['close'].rolling(window=30).mean()
        
        # Calculate volatility (standard deviation of price changes)
        df['volatility'] = df['price_change'].rolling(window=7).std()
    
    if 'Volume USD' in df.columns:
        # Calculate volume changes
        df['volume_change'] = df['Volume USD'].pct_change()
        
        # Calculate volume moving average
        df['volume_ma_7'] = df['Volume USD'].rolling(window=7).mean()
    
    # Drop any remaining NaN values
    df = df.dropna()
    
    return df

def save_processed_data(df, base_filename):
    # Create processed_data directory if it doesn't exist
    os.makedirs('processed_data', exist_ok=True)
    
    # Save in different formats
    base_path = os.path.join('processed_data', base_filename)
    
    # CSV
    df.to_csv(f"{base_path}.csv", index=False)
    
    # JSON
    df.to_json(f"{base_path}.json", orient='records', date_format='iso')
    
    # Parquet
    df.to_parquet(f"{base_path}.parquet", index=False)

def main():
    # List of cryptocurrency files to process
    crypto_files = ['BTC.csv', 'ETH.csv', 'LTC.csv', 'TRX.csv', 'EOC_coin.csv']
    
    for file in crypto_files:
        print(f"Processing {file}...")
        try:
            # Process the data
            processed_df = process_crypto_data(file)
            
            # Get base filename without extension
            base_filename = os.path.splitext(file)[0]
            
            # Save processed data in multiple formats
            save_processed_data(processed_df, base_filename)
            
            print(f"Successfully processed {file}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    main() 