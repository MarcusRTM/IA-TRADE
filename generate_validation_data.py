import os
import pandas as pd

# Configurações
DATA_DIR = 'processed_data'
SYMBOL = 'BTC'

# Carrega o dataset completo
data_path = os.path.join(DATA_DIR, f'{SYMBOL}.csv')
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_path}")
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])

# Filtra para o período de validação (2025-01-01 a 2025-03-31)
val_df = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-03-31')].copy()

# Salva o arquivo de validação
val_path = os.path.join(DATA_DIR, f'{SYMBOL}_val.csv')
val_df.to_csv(val_path, index=False)
print(f"Arquivo de validação gerado: {val_path}") 