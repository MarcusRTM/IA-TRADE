import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import random
import plotly.graph_objects as go
import plotly.express as px
import altair as alt
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
from utils_ai import ask_ai
import requests
from trading_bot import TradingBot
from run_trading_bot import run_backtest, calculate_metrics, plot_results

# Configuração do tema e estilo
st.set_page_config(
    page_title="QuantTrade AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stMetric:hover {
        background-color: #e6e9ef;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .kpi-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 2rem 1rem 1.5rem 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        text-align: center;
    }
    .kpi-title {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #222;
    }
    .config-card {
        background: #e6f0fa;
        border-radius: 10px;
        padding: 1.5rem 1rem;
        margin: 1.5rem auto 2rem auto;
        width: 100%;
        max-width: 350px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .block-space {
        margin-bottom: 2.5rem;
    }
    </style>
""", unsafe_allow_html=True)

symbols = ["BTC", "ETH", "LTC", "TRX", "EOC_coin"]

# Sidebar de navegação
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/robot-2.png", width=100)
    menu = st.radio("Navegação", [
        "Dashboard", "Model Training", "Backtesting", "Simulação em Tempo Real",
        "Settings", "Performance", "Assistente IA", "Sentimento de Mercado"
    ])
    st.markdown("---")
    st.markdown("### Sobre")
    st.info("""
    QuantTrade AI é um bot de trading que utiliza machine learning para prever sinais de compra/venda.
    - Modelo: Random Forest
    - Features: Preço, Médias Móveis, Volatilidade
    - Seed: 42 (reprodutível)
    """)

# SEED GLOBAL DOCUMENTADA
SEED = 42  # Semente global para reprodutibilidade de experimentos (documentada)
random.seed(SEED)
np.random.seed(SEED)

# Inicialização do session_state para persistência
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'fav_symbol': 'BTC',
        'initial_balance': 10000.0,
        'fee': 0.1,
        'slippage': 0.05,
        'theme': 'light'
    }
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ''

if 'last_backtest' not in st.session_state:
    st.session_state.last_backtest = None

# Diretórios
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'processed_data')
os.makedirs(DATA_DIR, exist_ok=True)

# Carregar automaticamente o arquivo de candles BTC.csv ao iniciar o app
if 'price_df' not in st.session_state:
    try:
        price_df = pd.read_csv('processed_data/BTC.csv')
        st.session_state['price_df'] = price_df
    except Exception as e:
        st.warning(f'Não foi possível carregar o arquivo BTC.csv: {e}')

# Cache para melhor performance
@st.cache_data
def load_data(symbol):
    """Carrega dados do ativo com cache para melhor performance."""
    _, _, data_path = get_paths(symbol)
    if not os.path.exists(data_path):
        st.warning(f"Arquivo de dados não encontrado para {symbol}: {data_path}")
        return None
    df = pd.read_csv(data_path)
    return df

@st.cache_resource
def load_model(symbol):
    """Carrega modelo e scaler com cache para melhor performance."""
    model_path, scaler_path, _ = get_paths(symbol)
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

def get_paths(symbol):
    """Retorna caminhos dos arquivos para um símbolo."""
    symbol = symbol.upper()
    model_path = os.path.join(MODEL_DIR, f'{symbol}_model.joblib')
    scaler_path = os.path.join(MODEL_DIR, f'{symbol}_scaler.joblib')
    data_path = os.path.join(DATA_DIR, f'{symbol}.csv')
    return model_path, scaler_path, data_path

def validate_data(df):
    """Valida dados e retorna mensagens de erro se necessário."""
    if df is None:
        return False, "Dados não encontrados"
    
    required_columns = ['close', 'ma_7', 'ma_30', 'volatility', 'volume_change']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Colunas obrigatórias faltando: {', '.join(missing_columns)}"
    
    if df.isnull().values.any():
        return False, "Dados contêm valores nulos"
    
    return True, "Dados válidos"

def prepare_data(df):
    """Prepara dados para treinamento do modelo."""
    features = ['close', 'ma_7', 'ma_30', 'volatility', 'volume_change']
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features)
    df['next_close'] = df['close'].shift(-1)
    df['signal'] = np.where(df['next_close'] > df['close'], 1, np.where(df['next_close'] < df['close'], -1, 0))
    df = df.dropna(subset=['next_close'])
    X = df[features].values
    y = df['signal'].values
    return X, y, df

def temporal_train_test_split(df, features):
    """Separa o dataset em treino (até 2024-12-31) e teste (2025-01-01 a 2025-03-31)"""
    df['date'] = pd.to_datetime(df['date'])
    train_df = df[df['date'] <= '2024-12-31'].copy()
    test_df = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-03-31')].copy()
    X_train, y_train = train_df[features], train_df['signal']
    X_test, y_test = test_df[features], test_df['signal']
    return X_train, y_train, X_test, y_test, train_df, test_df

def train_model(X, y, model_path, scaler_path):
    """Treina modelo Random Forest e salva resultados. Usa apenas dados até 31/12/2024."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_scaled, y)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model, scaler

def predict_signal(model, scaler, features):
    """Prediz sinal de trading para features dadas."""
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    signal = {1: 'buy', -1: 'sell', 0: 'hold'}[prediction]
    return signal

def backtest(df, model, scaler, initial_balance=10000, fee=0.001, slippage=0.0005):
    """Executa backtest com modelo treinado."""
    actions = []
    balance = initial_balance
    position = 0
    entry_price = 0
    for i in range(len(df)):
        row = df.iloc[i]
        features = np.array([
            row['close'], row['ma_7'], row['ma_30'], row['volatility'], row['volume_change']
        ]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        signal = model.predict(features_scaled)[0]
        price = row['close']
        if signal == 1 and position == 0:
            entry_price = price * (1 + slippage)
            position = 1
            balance -= entry_price * (1 + fee)
            actions.append({'date': str(row['date']), 'action': 'buy', 'price': float(entry_price), 'balance': float(balance)})
        elif signal == -1 and position == 1:
            exit_price = price * (1 - slippage)
            balance += exit_price * (1 - fee)
            profit = exit_price - entry_price
            actions.append({'date': str(row['date']), 'action': 'sell', 'price': float(exit_price), 'balance': float(balance), 'profit': float(profit)})
            position = 0
        else:
            actions.append({'date': str(row['date']), 'action': 'hold', 'price': float(price), 'balance': float(balance)})
    return actions

def calc_backtest_metrics(actions):
    """Calcula métricas do backtest."""
    total = len(actions)
    buy = sell = hold = 0
    win = loss = 0
    profits = []
    for a in actions:
        if a['action'] == 'buy':
            buy += 1
        elif a['action'] == 'sell':
            sell += 1
            if 'profit' in a:
                profits.append(a['profit'])
                if a['profit'] > 0:
                    win += 1
                else:
                    loss += 1
        elif a['action'] == 'hold':
            hold += 1
    
    lucro_total = actions[-1]['balance'] - actions[0]['balance'] if actions else 0
    pct_win = (win / sell) * 100 if sell else 0
    pct_loss = (loss / sell) * 100 if sell else 0
    taxa_buy = (buy / total) * 100 if total else 0
    taxa_sell = (sell / total) * 100 if total else 0
    taxa_hold = (hold / total) * 100 if total else 0
    
    # Métricas adicionais
    avg_profit = np.mean(profits) if profits else 0
    max_profit = max(profits) if profits else 0
    min_profit = min(profits) if profits else 0
    profit_factor = abs(sum([p for p in profits if p > 0]) / sum([abs(p) for p in profits if p < 0])) if sum([abs(p) for p in profits if p < 0]) != 0 else float('inf')
    
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
        'total_hold': hold,
        'avg_profit': avg_profit,
        'max_profit': max_profit,
        'min_profit': min_profit,
        'profit_factor': profit_factor
    }

def plot_price_signals(df, actions):
    """Plota gráfico de preço com sinais de trading."""
    fig = go.Figure()
    
    # Preço
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        mode='lines',
        name='Preço',
        line=dict(color='blue')
    ))
    
    # Médias móveis
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['ma_7'],
        mode='lines',
        name='MA7',
        line=dict(color='orange', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['ma_30'],
        mode='lines',
        name='MA30',
        line=dict(color='purple', dash='dash')
    ))
    
    # Sinais de compra
    buy_signals = [a for a in actions if a['action'] == 'buy']
    if buy_signals:
        fig.add_trace(go.Scatter(
            x=[a['date'] for a in buy_signals],
            y=[a['price'] for a in buy_signals],
            mode='markers',
            name='Compra',
            marker=dict(
                symbol='triangle-up',
                size=15,
                color='green'
            )
        ))
    
    # Sinais de venda
    sell_signals = [a for a in actions if a['action'] == 'sell']
    if sell_signals:
        fig.add_trace(go.Scatter(
            x=[a['date'] for a in sell_signals],
            y=[a['price'] for a in sell_signals],
            mode='markers',
            name='Venda',
            marker=dict(
                symbol='triangle-down',
                size=15,
                color='red'
            )
        ))
    
    fig.update_layout(
        title='Preço e Sinais de Trading',
        xaxis_title='Data',
        yaxis_title='Preço',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def plot_operations_pie(metrics):
    """Plota gráfico de pizza com distribuição de operações."""
    fig = go.Figure(data=[go.Pie(
        labels=['Compra', 'Venda', 'Hold'],
        values=[metrics['total_buy'], metrics['total_sell'], metrics['total_hold']],
        hole=.3,
        marker_colors=['green', 'red', 'gray']
    )])
    
    fig.update_layout(
        title='Distribuição de Operações',
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def get_download_link(df, filename, text):
    """Gera link para download de arquivo."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def compute_rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(df, fast=12, slow=26, signal=9):
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# --- Funções para análise de sentimento de mercado ---
def fetch_crypto_news(api_key, days=7):
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": api_key,
        "public": "true",
        "currencies": "BTC,ETH,LTC,TRX,EOC_coin",
        "filter": "news"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
    except Exception as e:
        return [], f"Erro ao buscar notícias: {e}"
    # Filtrar por últimos X dias
    filtered = []
    date_limit = (datetime.utcnow() - timedelta(days=days)).date()
    for item in data.get("results", []):
        published = item["published_at"][:10]
        if datetime.strptime(published, '%Y-%m-%d').date() >= date_limit:
            filtered.append({
                "title": item["title"],
                "url": item["url"],
                "published_at": published
            })
    return filtered, None

def analyze_sentiment_openai(news_list, api_key):
    client = openai.OpenAI(api_key=api_key)
    results = []
    for news in news_list:
        prompt = f"Classifique o sentimento desta manchete de notícia sobre criptomoedas como positivo, negativo ou neutro:\n\n\"{news['title']}\""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            sentiment = response.choices[0].message.content.strip().lower()
        except Exception as e:
            sentiment = "erro"
        results.append({**news, "sentiment": sentiment})
    return results

def fetch_crypto_news_period(api_key, start_date, end_date, max_pages=10):
    import requests
    all_news = []
    page = 1
    while page <= max_pages:
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": api_key,
            "public": "true",
            "currencies": "BTC,ETH,LTC,TRX,EOC_coin",
            "filter": "news",
            "since": start_date,
            "until": end_date,
            "page": page
        }
        try:
            response = requests.get(url, params=params)
            data = response.json()
        except Exception as e:
            return all_news, f"Erro ao buscar notícias: {e}"
        results = data.get("results", [])
        if not results:
            break
        for item in results:
            published = item["published_at"][:10]
            all_news.append({
                "title": item["title"],
                "url": item["url"],
                "published_at": published
            })
        page += 1
    return all_news, None

# --- DASHBOARD ---
if menu == "Dashboard":
    st.header("Dashboard")
    # --- PAINEL DE ALERTAS DA IA ---
    if 'ia_explicacoes' in st.session_state and st.session_state['ia_explicacoes']:
        st.markdown('---')
        st.subheader('Painel de Alertas da IA')
        for exp in st.session_state['ia_explicacoes'][-10:][::-1]:
            cor = '#ffe6e6' if exp.get('alerta') else '#e6f7ff'
            st.markdown(f"<div style='background:{cor}; border-radius:8px; padding:10px 16px; margin-bottom:8px;'><b>{exp['data']}</b> | <b>Ação:</b> {exp['acao']} | <b>Confiança:</b> {exp['confianca']:.2f}<br><b>Explicação IA:</b> {exp['explicacao']}</div>", unsafe_allow_html=True)
    # --- DASHBOARD INSPIRADO NO KAGGLE: GRÁFICOS INTERATIVOS ---
    st.markdown('---')
    st.header('Dashboard de Preço e Indicadores')
    # Usar price_df carregado automaticamente
    main_df_hist = st.session_state.get('price_df', None)
    df_pred = st.session_state.get('df_pred', None)
    if main_df_hist is not None and 'date' in main_df_hist.columns:
        main_df_hist['date'] = pd.to_datetime(main_df_hist['date'])
        hist_df = main_df_hist[main_df_hist['date'] <= pd.to_datetime('2024-12-31')].copy()
        # Se houver df_pred, usar para datas >= 2025-01-01
        if df_pred is not None and 'date' in df_pred.columns:
            df_pred['date'] = pd.to_datetime(df_pred['date'])
            pred_df = df_pred[df_pred['date'] >= pd.to_datetime('2025-01-01')].copy()
            # Padronizar colunas para merge (usar close/price, volume, etc)
            for col in ['open','high','low','close','Volume USD','rsi']:
                if col not in pred_df.columns and col in hist_df.columns:
                    pred_df[col] = None
            # Se não houver 'close', usar 'price' como 'close'
            if 'close' not in pred_df.columns and 'price' in pred_df.columns:
                pred_df['close'] = pred_df['price']
            # Se não houver 'Volume USD', usar 'volume' se existir
            if 'Volume USD' not in pred_df.columns and 'volume' in pred_df.columns:
                pred_df['Volume USD'] = pred_df['volume']
            # Concatenar
            main_df = pd.concat([hist_df, pred_df], ignore_index=True, sort=False)
        else:
            main_df = hist_df
        main_df = main_df.sort_values('date').reset_index(drop=True)
        # Candlestick Chart
        st.subheader('Candlestick Chart')
        if all(col in main_df.columns for col in ['open', 'high', 'low', 'close']):
            import plotly.graph_objects as go
            fig_candle = go.Figure()
            # Histórico
            hist_mask = main_df['date'] <= pd.to_datetime('2024-12-31')
            pred_mask = main_df['date'] >= pd.to_datetime('2025-01-01')
            fig_candle.add_trace(go.Candlestick(
                x=main_df.loc[hist_mask, 'date'],
                open=main_df.loc[hist_mask, 'open'],
                high=main_df.loc[hist_mask, 'high'],
                low=main_df.loc[hist_mask, 'low'],
                close=main_df.loc[hist_mask, 'close'],
                name='Histórico',
                increasing_line_color='green', decreasing_line_color='red',
                opacity=0.8
            ))
            # Predição
            if pred_mask.sum() > 0:
                fig_candle.add_trace(go.Candlestick(
                    x=main_df.loc[pred_mask, 'date'],
                    open=main_df.loc[pred_mask, 'open'],
                    high=main_df.loc[pred_mask, 'high'],
                    low=main_df.loc[pred_mask, 'low'],
                    close=main_df.loc[pred_mask, 'close'],
                    name='Predição',
                    increasing_line_color='blue', decreasing_line_color='orange',
                    opacity=0.8,
                    showlegend=True
                ))
                # Linha vertical
                fig_candle.add_vline(x=pd.to_datetime('2025-01-01'), line_width=2, line_dash='dash', line_color='blue', annotation_text='Início Predição', annotation_position='top right')
            st.plotly_chart(fig_candle, use_container_width=True)
        # Price Chart + Volume + RSI em colunas
        st.subheader('Indicadores de Mercado')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('**Price Chart**')
            price_col = 'close' if 'close' in main_df.columns else 'price'
            import plotly.graph_objects as go
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=main_df.loc[hist_mask, 'date'],
                y=main_df.loc[hist_mask, price_col],
                mode='lines',
                name='Histórico',
                line=dict(color='green')
            ))
            if pred_mask.sum() > 0:
                fig_price.add_trace(go.Scatter(
                    x=main_df.loc[pred_mask, 'date'],
                    y=main_df.loc[pred_mask, price_col],
                    mode='lines',
                    name='Predição',
                    line=dict(color='blue', dash='dash')
                ))
                fig_price.add_vline(x=pd.to_datetime('2025-01-01'), line_width=2, line_dash='dash', line_color='blue', annotation_text='Início Predição', annotation_position='top right')
            st.plotly_chart(fig_price, use_container_width=True)
            if 'Volume USD' in main_df.columns:
                st.markdown('**Volume Traded**')
                import plotly.graph_objects as go
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=main_df.loc[hist_mask, 'date'],
                    y=main_df.loc[hist_mask, 'Volume USD'],
                    name='Histórico',
                    marker_color='green',
                    opacity=0.7
                ))
                if pred_mask.sum() > 0:
                    fig_vol.add_trace(go.Bar(
                        x=main_df.loc[pred_mask, 'date'],
                        y=main_df.loc[pred_mask, 'Volume USD'],
                        name='Predição',
                        marker_color='blue',
                        opacity=0.7
                    ))
                    fig_vol.add_vline(x=pd.to_datetime('2025-01-01'), line_width=2, line_dash='dash', line_color='blue', annotation_text='Início Predição', annotation_position='top right')
                st.plotly_chart(fig_vol, use_container_width=True)
        with col2:
            st.markdown('**Relative Strength Index (RSI)**')
            # Calcular RSI se não existir
            if 'rsi' in main_df.columns:
                rsi = main_df['rsi']
            elif 'close' in main_df.columns:
                close = main_df['close']
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = None
            if rsi is not None:
                import plotly.graph_objects as go
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=main_df.loc[hist_mask, 'date'],
                    y=rsi.loc[hist_mask] if hasattr(rsi, 'loc') else rsi[hist_mask],
                    mode='lines',
                    name='Histórico',
                    line=dict(color='green')
                ))
                if pred_mask.sum() > 0:
                    fig_rsi.add_trace(go.Scatter(
                        x=main_df.loc[pred_mask, 'date'],
                        y=rsi.loc[pred_mask] if hasattr(rsi, 'loc') else rsi[pred_mask],
                        mode='lines',
                        name='Predição',
                        line=dict(color='blue', dash='dash')
                    ))
                    fig_rsi.add_vline(x=pd.to_datetime('2025-01-01'), line_width=2, line_dash='dash', line_color='blue', annotation_text='Início Predição', annotation_position='top right')
                st.plotly_chart(fig_rsi, use_container_width=True)
    if 'metrics' in st.session_state:
        metrics = st.session_state['metrics']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de Trades", metrics['total_trades'])
        col2.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
        col3.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        col4.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        st.line_chart(st.session_state['actions_df'].set_index('date')['balance'])
        st.dataframe(st.session_state['trades_df'])
    else:
        st.info("Rode um backtest para ver o dashboard.")

# --- MODEL TRAINING ---
elif menu == "Model Training":
    st.header("Model Training")
    symbol = st.selectbox("Ativo para treinar", symbols, key="train_symbol")
    if st.button("Treinar Modelo", key="train_btn"):
        train_path = os.path.join('processed_data', f'{symbol}_train.csv')
        if os.path.exists(train_path):
            df_train = pd.read_csv(train_path)
            features = ['close', 'ma_7', 'ma_30', 'volatility', 'volume_change']
            X_train = df_train[features]
            y_train = df_train['signal'] if 'signal' in df_train else (df_train['close'].shift(-1) > df_train['close']).astype(int)
            # Limpeza dos dados
            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            mask = X_train.notnull().all(axis=1)
            X_train = X_train[mask]
            y_train = y_train[mask]
            if X_train.empty:
                st.error("Todos os dados de treino foram removidos por conterem NaN ou infinito. Verifique o dataset.")
                st.stop()
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            joblib.dump(model, f'models/{symbol}_model.joblib')
            joblib.dump(scaler, f'models/{symbol}_scaler.joblib')
            st.success(f"Modelo treinado e salvo para {symbol}!")
            st.bar_chart(pd.Series(model.feature_importances_, index=features))
        else:
            st.warning(f"Arquivo de treino não encontrado: {train_path}")

# --- BACKTESTING ---
elif menu == "Backtesting":
    st.header("Backtesting")
    symbol = st.selectbox("Ativo", symbols, key="bt_symbol")
    # Modo agressivo
    modo_agressivo = st.checkbox("Modo Agressivo (mais operações)", value=False, key="bt_aggressive")
    if modo_agressivo:
        st.markdown('<div style="background:#ffe6e6; color:#b30000; border-radius:8px; padding:10px 16px; margin-bottom:12px;"><b>Modo Agressivo Ativo:</b> O bot fará mais operações, aceitando sinais menos confiáveis e sem filtro de tendência.</div>', unsafe_allow_html=True)
        confidence_threshold = 0.3
        trend_filter = False
        stop_loss = 0.01
        take_profit = 0.02
    else:
        confidence_threshold = st.slider("Limiar de confiança (%)", 0, 100, 60, key="bt_conf") / 100
        trend_filter = st.checkbox("Trend Filter", value=True, key="bt_trend")
        stop_loss = st.slider("Stop Loss (%)", 0, 20, 2, key="bt_stop") / 100
        take_profit = st.slider("Take Profit (%)", 0, 20, 4, key="bt_take") / 100
    initial_balance = st.number_input("Saldo inicial", value=10000.0, min_value=100.0, step=100.0, key="bt_balance")
    position_size = st.slider("Tamanho da posição (%)", 1, 100, 20, key="bt_pos_size") / 100
    if st.button("Rodar Backtest", key="bt_run"):
        features = ['close', 'ma_7', 'ma_30', 'volatility', 'volume_change']
        bot = TradingBot(
            symbol=symbol,
            model_type="rf",
            use_rl=False,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence_threshold=confidence_threshold,
            dynamic_confidence=False,
            confidence_window=20,
            features=features,
            trend_filter=trend_filter
        )
        actions_df, trades_df = run_backtest(
            bot, symbol, initial_balance, 0.001, 0.0005, confidence_threshold
        )
        metrics = calculate_metrics(actions_df, trades_df)
        st.session_state['actions_df'] = actions_df
        st.session_state['trades_df'] = trades_df
        st.session_state['metrics'] = metrics
        st.success("Backtest concluído!")
        st.plotly_chart(plot_results(actions_df, trades_df, symbol), use_container_width=True)
        st.dataframe(trades_df)

# --- SIMULAÇÃO EM TEMPO REAL ---
elif menu == "Simulação em Tempo Real":
    st.header("Simulação em Tempo Real")
    st.write("(Em desenvolvimento: simulação candle a candle, controles de play/pause, gráficos dinâmicos)")
    # Aqui você pode integrar a lógica de simulação já existente

# --- SETTINGS ---
elif menu == "Settings":
    st.header("Settings")
    st.write("Configurações globais do bot e preferências do usuário.")
    # Campos para chaves de API
    openai_api_key = st.text_input("Chave da API OpenAI", type="password", value=st.session_state.get('openai_api_key', ''))
    cryptopanic_api_key = st.text_input("Chave da API CryptoPanic", type="password", value=st.session_state.get('cryptopanic_api_key', ''))
    if st.button("Salvar Configurações", key="save_settings"):
        st.session_state['openai_api_key'] = openai_api_key
        st.session_state['cryptopanic_api_key'] = cryptopanic_api_key
        st.success("Configurações salvas!")

# --- PERFORMANCE ---
elif menu == "Performance":
    st.header("Performance Reports")
    if 'metrics' in st.session_state:
        metrics = st.session_state['metrics']
        st.table(pd.DataFrame([metrics]))
        st.dataframe(st.session_state['trades_df'])
        st.download_button("Exportar Trades CSV", st.session_state['trades_df'].to_csv(index=False), file_name="trades.csv")
    else:
        st.info("Rode um backtest para ver relatórios de performance.")

# --- ASSISTENTE IA ---
elif menu == "Assistente IA":
    st.header("Assistente QuantTrade AI")
    st.write("Chat com IA para dúvidas, sugestões e explicações.")
    # Inicializar histórico de chat na sessão
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'clear_input' not in st.session_state:
        st.session_state.clear_input = False

    # Botão para limpar conversa
    if st.button("🧹 Limpar conversa"):
        st.session_state.chat_history = []
        st.session_state.clear_input = True
        st.rerun()

    # UI do chat
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"<div style='background:#222; color:#fff; border-radius:12px; padding:10px 16px; margin:8px 0 8px auto; max-width:70%; text-align:right;'><b>Você:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background:#e6f0fa; color:#222; border-radius:12px; padding:10px 16px; margin:8px auto 8px 0; max-width:70%; text-align:left;'><b>IA:</b><br>{msg['content']}</div>", unsafe_allow_html=True)

    # Campo de input e botão de envio lado a lado
    col_input, col_button = st.columns([6,1])
    with col_input:
        user_input = st.text_input(
            "",
            placeholder="Digite sua mensagem...",
            key="chat_input",
            value="" if st.session_state.clear_input else st.session_state.get("chat_input", "")
        )
    with col_button:
        send_clicked = st.button("Enviar", use_container_width=True)

    # Montar contexto do projeto
    contexto = """
    Projeto: QuantTrade AI
    Ativos disponíveis: BTC, ETH, LTC, TRX, EOC_coin
    O usuário pode treinar modelos, rodar backtests, simular operações e analisar sentimento de mercado.
    O bot utiliza machine learning e reinforcement learning para gerar sinais de trading.
    """
    if 'metrics' in st.session_state:
        contexto += f"\nÚltimo backtest: {st.session_state['metrics']}"
    if 'openai_api_key' in st.session_state:
        openai_api_key = st.session_state['openai_api_key']
    else:
        openai_api_key = ''

    # Lógica de envio de mensagem
    if send_clicked and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        prompt = (
            "Você é um assistente do QuantTrade AI. Use o contexto abaixo para responder de forma personalizada e útil.\n"
            f"Contexto do projeto:\n{contexto}\n"
            "Histórico do chat:\n" +
            "\n".join([f"Usuário: {m['content']}" if m['role']=='user' else f"IA: {m['content']}" for m in st.session_state.chat_history]) +
            f"\nPergunta do usuário: {user_input}"
        )
        with st.spinner("Consultando IA..."):
            resposta = ask_ai(prompt, openai_api_key)
        st.session_state.chat_history.append({"role": "assistant", "content": resposta})
        st.session_state.clear_input = True
        st.rerun()
    else:
        st.session_state.clear_input = False

# --- SENTIMENTO DE MERCADO ---
elif menu == "Sentimento de Mercado":
    import datetime
    st.header("Análise de Sentimento de Mercado")
    st.write("Veja o sentimento do mercado de criptomoedas em qualquer período histórico usando sua conta paga do CryptoPanic.")
    cryptopanic_api_key = st.session_state.get('cryptopanic_api_key', '')
    openai_api_key = st.session_state.get('openai_api_key', '')
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Data inicial", value=datetime.date(2024, 1, 1), min_value=datetime.date(2018, 1, 1))
    with col2:
        end_date = st.date_input("Data final", value=datetime.date(2024, 12, 31), min_value=datetime.date(2018, 1, 1))
    max_pages = st.number_input("Máx. páginas a buscar (20 notícias por página)", min_value=1, max_value=50, value=5)
    if st.button("Buscar e Analisar Notícias"):
        if not cryptopanic_api_key:
            st.warning("Informe a chave da API CryptoPanic em Settings.")
        elif not openai_api_key:
            st.warning("Configure sua chave OpenAI em Settings.")
        elif start_date > end_date:
            st.warning("A data inicial deve ser anterior à data final.")
        else:
            with st.spinner("Buscando notícias históricas..."):
                def fetch_crypto_news_period(api_key, start_date, end_date, max_pages=10):
                    import requests
                    all_news = []
                    page = 1
                    while page <= max_pages:
                        url = "https://cryptopanic.com/api/v1/posts/"
                        params = {
                            "auth_token": api_key,
                            "public": "true",
                            "currencies": "BTC,ETH,LTC,TRX,EOC_coin",
                            "filter": "news",
                            "since": start_date,
                            "until": end_date,
                            "page": page
                        }
                        try:
                            response = requests.get(url, params=params)
                            data = response.json()
                        except Exception as e:
                            return all_news, f"Erro ao buscar notícias: {e}"
                        results = data.get("results", [])
                        if not results:
                            break
                        for item in results:
                            published = item["published_at"][:10]
                            all_news.append({
                                "title": item["title"],
                                "url": item["url"],
                                "published_at": published
                            })
                        page += 1
                    return all_news, None
                news, err = fetch_crypto_news_period(
                    cryptopanic_api_key,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    max_pages=int(max_pages)
                )
            if err:
                st.error(err)
            elif not news:
                st.info("Nenhuma notícia encontrada para o período selecionado.")
            else:
                with st.spinner("Analisando sentimento das manchetes..."):
                    def analyze_sentiment_openai(news_list, api_key):
                        import openai
                        client = openai.OpenAI(api_key=api_key)
                        results = []
                        for news in news_list:
                            prompt = f"Classifique o sentimento desta manchete de notícia sobre criptomoedas como positivo, negativo ou neutro:\n\n\"{news['title']}\""
                            try:
                                response = client.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=[{"role": "user", "content": prompt}]
                                )
                                sentiment = response.choices[0].message.content.strip().lower()
                            except Exception as e:
                                sentiment = "erro"
                            results.append({**news, "sentiment": sentiment})
                        return results
                    news_with_sentiment = analyze_sentiment_openai(news, openai_api_key)
                df = pd.DataFrame(news_with_sentiment)
                st.dataframe(df[["published_at", "title", "sentiment"]])
                # Gráfico de pizza
                if not df.empty:
                    st.plotly_chart(
                        px.pie(df, names="sentiment", title="Distribuição de Sentimento das Notícias"),
                        use_container_width=True
                    )

# --- ANÁLISE DE PREVISÕES: ORGANIZAÇÃO VISUAL ---
st.markdown('---')
st.header('Análise de Previsões')

# Seção 1: Upload de dados e modelo
st.subheader('1. Upload de Dados e Modelo')
uploaded_file = st.file_uploader('Carregue um arquivo CSV com colunas: date, price, profit, real_action, predicted_action', type='csv', key='uploader_pred')
if uploaded_file is not None:
    df_pred = pd.read_csv(uploaded_file)
    st.session_state['df_pred'] = df_pred
    st.success('Arquivo carregado com sucesso!')

df_pred = st.session_state.get('df_pred', None)
if df_pred is not None:
    # Seção 2: Gráfico de decisões e lucro
    st.subheader('2. Gráfico de Decisões e Lucro')
    if all(col in df_pred.columns for col in ['date', 'price', 'real_action', 'predicted_action']):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_pred['date'], y=df_pred['price'], mode='lines', name='Preço'))
        for action, color, symbol in [('buy', 'green', 'triangle-up'), ('sell', 'red', 'triangle-down'), ('hold', 'gray', 'circle')]:
            mask = df_pred['predicted_action'] == action
            fig.add_trace(go.Scatter(
                x=df_pred.loc[mask, 'date'],
                y=df_pred.loc[mask, 'price'],
                mode='markers',
                name=f'Pred: {action}',
                marker=dict(color=color, symbol=symbol, size=10)
            ))
        st.plotly_chart(fig, use_container_width=True)
    # Seção 3: Métricas principais
    if all(col in df_pred.columns for col in ['real_action', 'predicted_action', 'profit']):
        st.subheader('3. Métricas Principais')
        total_profit = df_pred['profit'].sum()
        total_ops = len(df_pred)
        win_ops = (df_pred['profit'] > 0).sum()
        loss_ops = (df_pred['profit'] <= 0).sum()
        pct_win = 100 * win_ops / total_ops if total_ops else 0
        pct_loss = 100 * loss_ops / total_ops if total_ops else 0
        col1, col2, col3 = st.columns(3)
        col1.metric('Lucro Total', f'{total_profit:.2f}')
        col2.metric('% Operações Vencedoras', f'{pct_win:.1f}%')
        col3.metric('% Operações Perdedoras', f'{pct_loss:.1f}%')
        # Seção 4: Acurácia por tipo de ação
        st.subheader('4. Acurácia por Tipo de Ação')
        types = ['buy', 'sell', 'hold']
        accuracy_by_type = {}
        for t in types:
            mask = df_pred['predicted_action'] == t
            total_type = mask.sum()
            correct = ((df_pred['predicted_action'] == df_pred['real_action']) & mask).sum()
            acc = 100 * correct / total_type if total_type else 0
            accuracy_by_type[t] = acc
        acc_df = pd.DataFrame({
            'Tipo': list(accuracy_by_type.keys()),
            'Taxa de Acerto (%)': list(accuracy_by_type.values())
        })
        st.table(acc_df)
        # Botão de exportação de relatório em CSV
        import io
        relatorio_df = pd.DataFrame({
            'Métrica': ['Lucro Total', '% Operações Vencedoras', '% Operações Perdedoras'] + list(acc_df['Tipo']),
            'Valor': [f'{total_profit:.2f}', f'{pct_win:.1f}%', f'{pct_loss:.1f}%'] + [f'{v:.1f}%' for v in acc_df['Taxa de Acerto (%)']]
        })
        csv = relatorio_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Exportar Relatório CSV',
            data=csv,
            file_name='relatorio_metricas.csv',
            mime='text/csv'
        )
    # Seção 5: Comparativo com buy & hold
    if all(col in df_pred.columns for col in ['date', 'price', 'profit']):
        st.subheader('5. Comparativo: Bot vs Buy & Hold')
        df_sorted = df_pred.sort_values('date').reset_index(drop=True)
        bot_cum_profit = df_sorted['profit'].cumsum()
        bot_cum_profit_pct = 100 * (bot_cum_profit / (df_sorted['price'].iloc[0] if df_sorted['price'].iloc[0] != 0 else 1))
        buy_hold_pct = 100 * (df_sorted['price'] - df_sorted['price'].iloc[0]) / (df_sorted['price'].iloc[0] if df_sorted['price'].iloc[0] != 0 else 1)
        plot_df = pd.DataFrame({
            'Data': df_sorted['date'],
            'Bot (Lucro Acumulado %)': bot_cum_profit_pct,
            'Buy & Hold (%)': buy_hold_pct
        })
        plot_df = plot_df.set_index('Data')
        st.line_chart(plot_df)

# --- GRÁFICO COMPARATIVO: BOT vs BUY & HOLD ---
if ('df' in locals() or 'df' in globals()) and all(col in df.columns for col in ['date', 'price', 'profit']):
    df_sorted = df.sort_values('date').reset_index(drop=True)
    # Lucro acumulado do bot
    bot_cum_profit = df_sorted['profit'].cumsum()
    # Lucro acumulado percentual do bot
    bot_cum_profit_pct = 100 * (bot_cum_profit / (df_sorted['price'].iloc[0] if df_sorted['price'].iloc[0] != 0 else 1))
    # Buy & Hold: variação percentual do preço
    buy_hold_pct = 100 * (df_sorted['price'] - df_sorted['price'].iloc[0]) / (df_sorted['price'].iloc[0] if df_sorted['price'].iloc[0] != 0 else 1)
    # DataFrame para plot
    plot_df = pd.DataFrame({
        'Data': df_sorted['date'],
        'Bot (Lucro Acumulado %)': bot_cum_profit_pct,
        'Buy & Hold (%)': buy_hold_pct
    })
    plot_df = plot_df.set_index('Data')
    st.markdown('### Comparativo: Bot vs Buy & Hold')
    st.line_chart(plot_df)

# --- MÉTRICAS FINAIS PERSONALIZADAS ---
if 'df' in locals() or 'df' in globals():
    # Garante que as colunas necessárias existem
    if all(col in df.columns for col in ['real_action', 'predicted_action', 'profit']):
        # Lucro total
        total_profit = df['profit'].sum()
        # Percentual de operações vencedoras e perdedoras
        total_ops = len(df)
        win_ops = (df['profit'] > 0).sum()
        loss_ops = (df['profit'] <= 0).sum()
        pct_win = 100 * win_ops / total_ops if total_ops else 0
        pct_loss = 100 * loss_ops / total_ops if total_ops else 0
        # Taxa de acerto por tipo de decisão
        types = ['buy', 'sell', 'hold']
        accuracy_by_type = {}
        for t in types:
            mask = df['predicted_action'] == t
            total_type = mask.sum()
            correct = ((df['predicted_action'] == df['real_action']) & mask).sum()
            acc = 100 * correct / total_type if total_type else 0
            accuracy_by_type[t] = acc
        # Exibição
        st.markdown("## Métricas Finais do Modelo")
        col1, col2, col3 = st.columns(3)
        col1.metric("Lucro Total", f"{total_profit:.2f}")
        col2.metric("% Operações Vencedoras", f"{pct_win:.1f}%")
        col3.metric("% Operações Perdedoras", f"{pct_loss:.1f}%")
        # Tabela de acurácia por tipo
        acc_df = pd.DataFrame({
            'Tipo': list(accuracy_by_type.keys()),
            'Taxa de Acerto (%)': list(accuracy_by_type.values())
        })
        st.table(acc_df)
        # Botão de exportação de relatório em CSV
        import io
        relatorio_df = pd.DataFrame({
            'Métrica': ['Lucro Total', '% Operações Vencedoras', '% Operações Perdedoras'] + list(acc_df['Tipo']),
            'Valor': [f'{total_profit:.2f}', f'{pct_win:.1f}%', f'{pct_loss:.1f}%'] + [f'{v:.1f}%' for v in acc_df['Taxa de Acerto (%)']]
        })
        csv = relatorio_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Exportar Relatório CSV',
            data=csv,
            file_name='relatorio_metricas.csv',
            mime='text/csv'
        ) 