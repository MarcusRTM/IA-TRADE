import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

SEED = 42  # Semente fixa para reprodutibilidade
np.random.seed(SEED)

DATA_DIR = 'processed_data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

cryptos = ['BTC', 'ETH', 'LTC', 'TRX', 'EOC_coin']

# Features para o modelo
features = [
    'close', 'ma_7', 'ma_30', 'volatility', 'volume_change',
    'rsi', 'macd', 'macd_signal', 'bb_middle', 'bb_upper', 'bb_lower'
]

def prepare_labels(df):
    # Classificação: 1=buy, -1=sell, 0=hold
    df['next_close'] = df['close'].shift(-1)
    df['signal'] = np.where(df['next_close'] > df['close'], 1, np.where(df['next_close'] < df['close'], -1, 0))
    # Regressão: retorno percentual futuro
    df['future_return'] = (df['next_close'] - df['close']) / df['close']
    df = df.dropna(subset=['next_close'])
    return df

def split_train_val(df):
    df['date'] = pd.to_datetime(df['date'])
    train_df = df[df['date'] <= '2024-12-31'].copy()
    val_df = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-03-31')].copy()
    return train_df, val_df

def train_and_eval_models(symbol):
    print(f'\n=== {symbol} ===')
    df = pd.read_csv(os.path.join(DATA_DIR, f'{symbol}.csv'))
    df = prepare_labels(df)
    train_df, val_df = split_train_val(df)
    print(f"Treino: {train_df.shape}, Validação: {val_df.shape}")
    X_train, y_train_cls, y_train_reg = train_df[features], train_df['signal'], train_df['future_return']
    X_val, y_val_cls, y_val_reg = val_df[features], val_df['signal'], val_df['future_return']

    # Limpeza de NaN e infinitos nas features
    n_before = len(X_train)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
    y_train_cls = y_train_cls.loc[X_train.index]
    y_train_reg = y_train_reg.loc[X_train.index]
    if len(X_train) < n_before:
        print(f"  [!] Removidas {n_before - len(X_train)} linhas com NaN/inf nos dados de treino.")
    n_before = len(X_val)
    X_val = X_val.replace([np.inf, -np.inf], np.nan).dropna()
    y_val_cls = y_val_cls.loc[X_val.index]
    y_val_reg = y_val_reg.loc[X_val.index]
    if len(X_val) < n_before:
        print(f"  [!] Removidas {n_before - len(X_val)} linhas com NaN/inf nos dados de validação.")

    # Padronização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, os.path.join(MODEL_DIR, f'{symbol}_scaler.joblib'))

    # Random Forest Classifier com GridSearch
    rf_params = {'n_estimators': [100, 200], 'max_depth': [3, 5, None]}
    rf_cls = GridSearchCV(RandomForestClassifier(random_state=SEED), rf_params, cv=3, n_jobs=-1)
    rf_cls.fit(X_train_scaled, y_train_cls)
    best_rf = rf_cls.best_estimator_
    joblib.dump(best_rf, os.path.join(MODEL_DIR, f'{symbol}_rf_cls.joblib'))
    y_pred_cls = best_rf.predict(X_val_scaled)
    print("Random Forest Classifier (Buy/Sell/Hold):")
    print(classification_report(y_val_cls, y_pred_cls, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_val_cls, y_pred_cls))

    # XGBoost Classifier com GridSearch
    xgb_map = {-1: 0, 0: 1, 1: 2}
    y_train_cls_xgb = y_train_cls.map(xgb_map)
    y_val_cls_xgb = y_val_cls.map(xgb_map)
    xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7]}
    xgb_cls = GridSearchCV(xgb.XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='mlogloss'), xgb_params, cv=3, n_jobs=-1)
    xgb_cls.fit(X_train_scaled, y_train_cls_xgb)
    best_xgb = xgb_cls.best_estimator_
    joblib.dump(best_xgb, os.path.join(MODEL_DIR, f'{symbol}_xgb_cls.joblib'))
    y_pred_xgb_cls = best_xgb.predict(X_val_scaled)
    print("XGBoost Classifier (Buy/Sell/Hold):")
    print(classification_report(y_val_cls_xgb, y_pred_xgb_cls, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_val_cls_xgb, y_pred_xgb_cls))

    # LightGBM Classifier com GridSearch
    if lgb is not None:
        lgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7]}
        lgb_cls = GridSearchCV(lgb.LGBMClassifier(random_state=SEED), lgb_params, cv=3, n_jobs=-1)
        lgb_cls.fit(X_train_scaled, y_train_cls_xgb)
        best_lgb = lgb_cls.best_estimator_
        joblib.dump(best_lgb, os.path.join(MODEL_DIR, f'{symbol}_lgb_cls.joblib'))
        y_pred_lgb_cls = best_lgb.predict(X_val_scaled)
        print("LightGBM Classifier (Buy/Sell/Hold):")
        print(classification_report(y_val_cls_xgb, y_pred_lgb_cls, digits=3))
        print("Confusion Matrix:\n", confusion_matrix(y_val_cls_xgb, y_pred_lgb_cls))
    else:
        print("[!] LightGBM não instalado. Pulei este modelo.")

    # Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=SEED)
    rf_reg.fit(X_train_scaled, y_train_reg)
    joblib.dump(rf_reg, os.path.join(MODEL_DIR, f'{symbol}_rf_reg.joblib'))
    y_pred_reg = rf_reg.predict(X_val_scaled)
    print("Random Forest Regressor (Future Return):")
    print(f"MSE: {mean_squared_error(y_val_reg, y_pred_reg):.6f} | R2: {r2_score(y_val_reg, y_pred_reg):.4f}")

    # XGBoost Regressor
    xgb_reg = xgb.XGBRegressor(n_estimators=100, random_state=SEED)
    xgb_reg.fit(X_train_scaled, y_train_reg)
    joblib.dump(xgb_reg, os.path.join(MODEL_DIR, f'{symbol}_xgb_reg.joblib'))
    y_pred_xgb_reg = xgb_reg.predict(X_val_scaled)
    print("XGBoost Regressor (Future Return):")
    print(f"MSE: {mean_squared_error(y_val_reg, y_pred_xgb_reg):.6f} | R2: {r2_score(y_val_reg, y_pred_xgb_reg):.4f}")

    # LightGBM Regressor
    if lgb is not None:
        lgb_reg = lgb.LGBMRegressor(n_estimators=100, random_state=SEED)
        lgb_reg.fit(X_train_scaled, y_train_reg)
        joblib.dump(lgb_reg, os.path.join(MODEL_DIR, f'{symbol}_lgb_reg.joblib'))
        y_pred_lgb_reg = lgb_reg.predict(X_val_scaled)
        print("LightGBM Regressor (Future Return):")
        print(f"MSE: {mean_squared_error(y_val_reg, y_pred_lgb_reg):.6f} | R2: {r2_score(y_val_reg, y_pred_lgb_reg):.4f}")

if __name__ == "__main__":
    for symbol in cryptos:
        train_and_eval_models(symbol) 