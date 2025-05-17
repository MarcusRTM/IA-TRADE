import pandas as pd
from utils_data import split_and_save_train_val
import os

DATA_DIR = 'processed_data'
symbols = ['BTC', 'ETH', 'LTC', 'TRX', 'EOC_coin']

for symbol in symbols:
    full_path = os.path.join(DATA_DIR, f'{symbol}.csv')
    if os.path.exists(full_path):
        print(f'Processando {full_path}...')
        df = pd.read_csv(full_path)
        split_and_save_train_val(df, symbol)
        print(f'Arquivos de treino e validação gerados para {symbol}.')
    else:
        print(f'Arquivo não encontrado: {full_path}') 