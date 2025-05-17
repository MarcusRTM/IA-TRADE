import os
import pandas as pd

def split_and_save_train_val(df, symbol):
    """Divide o dataset em treino (até 2024-12-31) e validação (2025-01-01 a 2025-03-31) e salva como CSV na pasta processed_data."""
    df['date'] = pd.to_datetime(df['date'])
    train_df = df[df['date'] <= '2024-12-31'].copy()
    val_df = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-03-31')].copy()
    DATA_DIR = 'processed_data'
    train_path = os.path.join(DATA_DIR, f'{symbol}_train.csv')
    val_path = os.path.join(DATA_DIR, f'{symbol}_val.csv')
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    return train_path, val_path 