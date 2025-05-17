import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import math
import random

app = Flask(__name__)
CORS(app)

# SEED GLOBAL DOCUMENTADA
SEED = 42  # Semente global para reprodutibilidade de experimentos (documentada)

random.seed(SEED)
np.random.seed(SEED)

# Configuração para permitir deploy futuro
# Em produção, use variáveis de ambiente ou configurações externas
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'btc_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'btc_scaler.joblib')

# Função para obter caminhos de modelo e scaler por ativo
def get_paths(symbol):
    symbol = symbol.upper()
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{symbol}_model.joblib')
    scaler_path = os.path.join(model_dir, f'{symbol}_scaler.joblib')
    data_path = os.path.join(os.path.dirname(__file__), 'processed_data', f'{symbol}.csv')
    return model_path, scaler_path, data_path

# Carrega os dados processados do BTC
def load_data(symbol):
    _, _, data_path = get_paths(symbol)
    df = pd.read_csv(data_path)
    print(df.head())
    print(df.columns)
    return df

# Prepara features e labels para o modelo
def prepare_data(df):
    # Features: usar colunas relevantes
    features = ['close', 'ma_7', 'ma_30', 'volatility', 'volume_change']
    # Remove linhas com NaN ou inf nas features
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features)
    # Label: criar um sinal simples
    df['next_close'] = df['close'].shift(-1)
    df['signal'] = np.where(df['next_close'] > df['close'], 1, np.where(df['next_close'] < df['close'], -1, 0))
    df = df.dropna(subset=['next_close'])  # Remove última linha sem next_close
    X = df[features].values
    y = df['signal'].values
    return X, y

# Treina o modelo Random Forest
def train_model(X, y, model_path, scaler_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model, scaler, X_test_scaled, y_test

# Carrega o modelo e scaler salvos
def load_model(symbol):
    model_path, scaler_path, _ = get_paths(symbol)
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

# Endpoint para servir o home.html
@app.route('/')
def index():
    return send_from_directory('.', 'home.html')

# Endpoint para treinar o modelo
@app.route('/train', methods=['POST'])
def train():
    try:
        symbol = request.json.get('symbol', 'BTC')
        df = load_data(symbol)
        X, y = prepare_data(df)
        model_path, scaler_path, _ = get_paths(symbol)
        model, scaler, X_test, y_test = train_model(X, y, model_path, scaler_path)
        accuracy = model.score(X_test, y_test)
        if accuracy is not None and not math.isnan(accuracy):
            accuracy = round(float(accuracy) * 100, 2)
        else:
            accuracy = 0.0
        return jsonify({'message': f'Model trained successfully for {symbol}', 'accuracy': accuracy})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print('Erro detalhado no /train:', tb)
        return jsonify({'error': str(e), 'traceback': tb}), 500

# Endpoint para predizer sinal (buy/sell/hold)
@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.json.get('symbol', 'BTC')
    model, scaler = load_model(symbol)
    if model is None or scaler is None:
        return jsonify({'error': f'Model for {symbol} not trained'}), 400
    data = request.json
    features = np.array([
        data['close'], data['ma_7'], data['ma_30'], data['volatility'], data['volume_change']
    ]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    signal = {1: 'buy', -1: 'sell', 0: 'hold'}[prediction]
    return jsonify({'signal': signal})

# Endpoint para obter histórico de sinais (simulado)
@app.route('/history', methods=['GET'])
def history():
    # Simulação de histórico (em produção, use dados reais)
    history = [
        {'date': '2024-01-01', 'signal': 'buy'},
        {'date': '2024-01-02', 'signal': 'hold'},
        {'date': '2024-01-03', 'signal': 'sell'}
    ]
    return jsonify(history)

# Função de backtesting
def backtest(df, model, scaler, initial_balance=10000, fee=0.001, slippage=0.0005):
    actions = []
    balance = initial_balance
    position = 0  # 0: fora, 1: comprado
    entry_price = 0
    for i in range(len(df)):
        row = df.iloc[i]
        features = np.array([
            row['close'], row['ma_7'], row['ma_30'], row['volatility'], row['volume_change']
        ]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        signal = model.predict(features_scaled)[0]
        price = row['close']
        # Simula as ações
        if signal == 1 and position == 0:  # Buy
            entry_price = price * (1 + slippage)
            position = 1
            balance -= entry_price * (1 + fee)
            actions.append({'date': str(row['date']), 'action': 'buy', 'price': float(entry_price), 'balance': float(balance)})
        elif signal == -1 and position == 1:  # Sell
            exit_price = price * (1 - slippage)
            balance += exit_price * (1 - fee)
            profit = exit_price - entry_price
            actions.append({'date': str(row['date']), 'action': 'sell', 'price': float(exit_price), 'balance': float(balance), 'profit': float(profit)})
            position = 0
        else:
            actions.append({'date': str(row['date']), 'action': 'hold', 'price': float(price), 'balance': float(balance)})
    return actions

def calc_backtest_metrics(actions):
    total = len(actions)
    buy = sell = hold = 0
    buy_win = buy_loss = sell_win = sell_loss = hold_count = 0
    win = loss = 0
    for a in actions:
        if a['action'] == 'buy':
            buy += 1
        elif a['action'] == 'sell':
            sell += 1
            if 'profit' in a:
                if a['profit'] > 0:
                    win += 1
                    sell_win += 1
                else:
                    loss += 1
                    sell_loss += 1
        elif a['action'] == 'hold':
            hold += 1
            hold_count += 1
    lucro_total = actions[-1]['balance'] - actions[0]['balance'] if actions else 0
    pct_win = (win / sell) * 100 if sell else 0
    pct_loss = (loss / sell) * 100 if sell else 0
    taxa_buy = (buy / total) * 100 if total else 0
    taxa_sell = (sell / total) * 100 if total else 0
    taxa_hold = (hold / total) * 100 if total else 0
    return {
        'lucro_total': lucro_total,
        'pct_win': pct_win,
        'pct_loss': pct_loss,
        'taxa_buy': taxa_buy,
        'taxa_sell': taxa_sell,
        'taxa_hold': taxa_hold,
        'total_trades': total,
        'total_buy': buy,
        'total_sell': sell,
        'total_hold': hold
    }

# Endpoint para rodar o backtest
@app.route('/backtest', methods=['POST'])
def run_backtest():
    try:
        symbol = request.json.get('symbol', 'BTC')
        initial_balance = float(request.json.get('initialBalance', 10000))
        fee = float(request.json.get('fee', 0.001))
        slippage = float(request.json.get('slippage', 0.0005))
        df = load_data(symbol)
        model, scaler = load_model(symbol)
        features = ['close', 'ma_7', 'ma_30', 'volatility', 'volume_change']
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=features)
        actions = backtest(df, model, scaler, initial_balance, fee, slippage)
        metrics = calc_backtest_metrics(actions)
        # Calcular maxDrawdown
        balances = [a['balance'] for a in actions]
        peak = balances[0] if balances else 0
        max_drawdown = 0
        for b in balances:
            if b > peak:
                peak = b
            drawdown = peak - b
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        metrics['maxDrawdown'] = max_drawdown
        return jsonify({'actions': actions, 'metrics': metrics})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print('Erro detalhado no /backtest:', tb)
        return jsonify({'error': str(e), 'traceback': tb}), 500

# Endpoint para expor a seed para o frontend
@app.route('/seed', methods=['GET'])
def get_seed():
    return jsonify({'seed': SEED})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5555) 