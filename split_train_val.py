import os
import pandas as pd
from quanttrade_streamlit import prepare_data  # Importa a função de feature engineering

ASSETS = ["BTC", "ETH", "LTC", "TRX", "EOC_coin"]
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_data")
os.makedirs(PROCESSED_DIR, exist_ok=True)

for asset in ASSETS:
    file_path = os.path.join(DATA_DIR, f"{asset}.csv")
    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        continue
    df = pd.read_csv(file_path)
    # Padroniza nomes das colunas para minúsculas e remove espaços
    df.columns = [col.lower().replace(' ', '') for col in df.columns]
    # Renomeia volumeusd para volume, se existir
    if 'volumeusd' in df.columns:
        df = df.rename(columns={'volumeusd': 'volume'})
    print(f"{asset} - colunas disponíveis:", df.columns.tolist())
    print(df.head())
    # Detecta coluna de data
    date_col = None
    for col in df.columns:
        if col in ["date", "timestamp"]:
            date_col = col
            break
    if date_col is None:
        print(f"Coluna de data não encontrada em {file_path}")
        continue
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    # Checa colunas base
    required_cols = ['close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"{asset}: Colunas base faltando no CSV original: {missing}")
        continue
    # Checa se são numéricas
    if not pd.api.types.is_numeric_dtype(df['close']):
        print(f"{asset}: Coluna 'close' não é numérica!")
        continue
    if not pd.api.types.is_numeric_dtype(df['volume']):
        print(f"{asset}: Coluna 'volume' não é numérica!")
        continue
    # Checa valores nulos
    if df['close'].isnull().any() or df['volume'].isnull().any():
        print(f"{asset}: Existem valores nulos em 'close' ou 'volume'.")
        continue
    # Diagnóstico extra
    print(f"{asset} - tipos das colunas:\n", df.dtypes)
    print(f"{asset} - quantidade de linhas: {len(df)}")
    print(f"{asset} - valores únicos em 'close':", df['close'].unique()[:10])
    print(f"{asset} - valores únicos em 'volume':", df['volume'].unique()[:10])
    if len(df) < 30:
        print(f"{asset}: Poucos dados para calcular médias móveis.")
        continue
    # Feature engineering antes do split
    _, _, df = prepare_data(df)
    # Split
    train_df = df[df[date_col] <= "2024-12-31"].copy()
    val_df = df[(df[date_col] >= "2025-01-01") & (df[date_col] <= "2025-03-31")].copy()
    # Salva
    train_path = os.path.join(PROCESSED_DIR, f"{asset}_train.csv")
    val_path = os.path.join(PROCESSED_DIR, f"{asset}_val.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    # Prints de confirmação
    if not train_df.empty:
        print(f"{asset} train.csv: {len(train_df)} registros, de {train_df[date_col].min().date()} até {train_df[date_col].max().date()}")
    else:
        print(f"{asset} train.csv: VAZIO")
    if not val_df.empty:
        print(f"{asset} val.csv: {len(val_df)} registros, de {val_df[date_col].min().date()} até {val_df[date_col].max().date()}")
    else:
        print(f"{asset} val.csv: VAZIO") 