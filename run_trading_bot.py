import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trading_bot import TradingBot
import os

# Configurações globais
DATA_DIR = 'processed_data'
MODEL_DIR = 'models'
SYMBOL = 'BTC'

def run_backtest(bot, symbol, initial_balance=10000.0, fee=0.001, slippage=0.0005, confidence_threshold=0.6):
    """
    Executa backtest com o bot de trading.
    
    Args:
        bot (TradingBot): Instância do bot de trading
        symbol (str): Símbolo da criptomoeda
        initial_balance (float): Saldo inicial
        fee (float): Taxa de transação
        slippage (float): Slippage esperado
        confidence_threshold (float): Limiar de confiança para executar ações
    """
    # Carrega dados de validação
    val_path = os.path.join(DATA_DIR, f'{symbol}_val.csv')
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Arquivo de validação não encontrado: {val_path}")
    df = pd.read_csv(val_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Simula o backtest
    balance = initial_balance
    position = 0
    entry_price = None
    actions = []
    trades = []
    
    for i, row in df.iterrows():
        action, confidence, position_size_usd, explicacao = bot.get_action(
            current_state=row,
            available_balance=balance,
            in_position=(position == 1),
            entry_price=entry_price,
            explain_with_ai=True,
            openai_api_key=None  # ou passe a chave se disponível
        )
        price = row['close']
        if action == 1 and position == 0:
            # Compra
            entry_price = price
            position = 1
            balance -= position_size_usd
            trades.append({
                'date': row['date'],
                'action': 'BUY',
                'price': price,
                'confidence': confidence,
                'balance': balance,
                'explicacao': explicacao
            })
        elif action == -1 and position == 1:
            # Venda
            if position_size_usd is None:
                position_size_usd = balance
            balance += position_size_usd * (price / entry_price)
            position = 0
            entry_price = None
            trades.append({
                'date': row['date'],
                'action': 'SELL',
                'price': price,
                'confidence': confidence,
                'balance': balance,
                'explicacao': explicacao
            })
        actions.append({
            'date': row['date'],
            'action': action,
            'confidence': confidence,
            'balance': balance,
            'position': position,
            'price': price,
            'explicacao': explicacao
        })
    
    actions_df = pd.DataFrame(actions)
    trades_df = pd.DataFrame(trades)
    # Garante colunas mesmo se trades_df estiver vazio
    for col in ['date', 'action', 'price', 'confidence', 'balance', 'explicacao']:
        if col not in trades_df.columns:
            trades_df[col] = []
    return actions_df, trades_df

def calculate_metrics(actions_df, trades_df):
    """Calcula métricas de performance do backtest."""
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'avg_confidence': 0.0
        }
    
    # Calculate profit for each trade
    trades_df['profit'] = trades_df['balance'].diff().where(trades_df['action'] == 'SELL', 0)
    
    winning_trades = len(trades_df[(trades_df['action'] == 'SELL') & (trades_df['profit'] > 0)])
    total_trades = len(trades_df[trades_df['action'] == 'SELL'])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    total_profit = trades_df['profit'].sum()
    total_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Calculate drawdown
    cumulative_returns = (1 + trades_df['profit'] / trades_df['balance'].shift(1)).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = abs(drawdowns.min())
    
    # Calculate Sharpe ratio
    returns = trades_df['profit'] / trades_df['balance'].shift(1)
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0.0
    
    # Calculate average confidence
    avg_confidence = trades_df['confidence'].mean()
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_profit': total_profit,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'avg_confidence': avg_confidence
    }

def plot_results(actions_df, trades_df, symbol):
    """Plota resultados do backtest."""
    fig = make_subplots(rows=3, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=('Preço e Sinais', 'Equity Curve', 'Drawdown'),
                       row_heights=[0.5, 0.3, 0.2])
    
    # Preço e sinais
    fig.add_trace(
        go.Scatter(
            x=actions_df['date'],
            y=actions_df['price'],
            mode='lines',
            name='Preço',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Sinais de compra/venda
    buy_signals = trades_df[trades_df['action'] == 'BUY']
    sell_signals = trades_df[trades_df['action'] == 'SELL']
    
    fig.add_trace(
        go.Scatter(
            x=buy_signals['date'],
            y=buy_signals['price'],
            mode='markers',
            name='Compra',
            marker=dict(
                symbol='triangle-up',
                size=10,
                color='green'
            )
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=sell_signals['date'],
            y=sell_signals['price'],
            mode='markers',
            name='Venda',
            marker=dict(
                symbol='triangle-down',
                size=10,
                color='red'
            )
        ),
        row=1, col=1
    )
    
    # Equity Curve
    fig.add_trace(
        go.Scatter(
            x=actions_df['date'],
            y=actions_df['balance'],
            name='Equity',
            line=dict(color='green')
        ),
        row=2, col=1
    )
    
    # Drawdown
    equity_curve = actions_df['balance'].values
    rolling_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    fig.add_trace(
        go.Scatter(
            x=actions_df['date'],
            y=drawdown * 100,
            name='Drawdown',
            line=dict(color='red')
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title=f'Backtest Results - {symbol}',
        height=900,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def main():
    # Configurações
    symbol = 'BTC'
    initial_balance = 10000.0
    fee = 0.001
    slippage = 0.0005
    confidence_threshold = 0.6
    
    print(f"\n=== Backtest do Bot de Trading para {symbol} ===")
    print(f"Saldo inicial: ${initial_balance:.2f}")
    print(f"Taxa: {fee*100:.1f}%")
    print(f"Slippage: {slippage*100:.2f}%")
    print(f"Limiar de confiança: {confidence_threshold*100:.1f}%")
    
    # Cria e executa o bot
    bot = TradingBot(
        symbol=symbol,
        model_type='ensemble',
        use_rl=True,
        position_size=0.2,
        stop_loss=0.02,
        take_profit=0.04,
        confidence_threshold=0.55,
        dynamic_confidence=True,
        confidence_window=20,
        features=None,  # Usa o conjunto padrão de features
        trend_filter=True
    )
    actions_df, trades_df = run_backtest(bot, symbol, initial_balance, fee, slippage, confidence_threshold)
    
    # Calcula e exibe métricas
    metrics = calculate_metrics(actions_df, trades_df)
    print("\nMétricas de Performance:")
    print(f"Total de trades: {metrics['total_trades']}")
    print(f"Taxa de acerto: {metrics['win_rate']:.1f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Lucro total: ${metrics['total_profit']:.2f}")
    print(f"Drawdown máximo: {metrics['max_drawdown']:.1f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Confiança média: {metrics['avg_confidence']:.1f}%")
    
    # Plota resultados
    fig = plot_results(actions_df, trades_df, symbol)
    fig.show()
    
    # Salva resultados
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Salva métricas
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(results_dir, f'{symbol}_metrics_{timestamp}.csv'), index=False)
    
    # Salva trades
    trades_df.to_csv(os.path.join(results_dir, f'{symbol}_trades_{timestamp}.csv'), index=False)
    
    print(f"\nResultados salvos em: {results_dir}")

if __name__ == "__main__":
    main() 