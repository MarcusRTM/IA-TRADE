import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import joblib
warnings.filterwarnings('ignore')

class BacktestSystem:
    def __init__(self, 
                 initial_balance=10000.0,
                 transaction_fee=0.001,  # 0.1%
                 slippage=0.0005,       # 0.05%
                 execution_delay=1,     # 1 candle de delay
                 position_size=1.0,     # Tamanho da posição em % do capital
                 stop_loss=None,        # Stop loss em % (ex: 0.02 para 2%)
                 take_profit=None       # Take profit em % (ex: 0.04 para 4%)
                 ):
        """
        Sistema de backtesting com simulação realista de ordens.
        
        Args:
            initial_balance (float): Capital inicial
            transaction_fee (float): Taxa de transação (ex: 0.001 = 0.1%)
            slippage (float): Slippage esperado (ex: 0.0005 = 0.05%)
            execution_delay (int): Delay de execução em candles
            position_size (float): Tamanho da posição em % do capital
            stop_loss (float): Stop loss em % (ex: 0.02 para 2%)
            take_profit (float): Take profit em % (ex: 0.04 para 4%)
        """
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.execution_delay = execution_delay
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Estado do backtest
        self.reset()
    
    def reset(self):
        """Reseta o estado do backtest."""
        self.balance = self.initial_balance
        self.position = 0  # 0: sem posição, 1: comprado
        self.entry_price = 0
        self.trades = []
        self.equity_curve = []
        self.current_step = 0
        self.stop_triggered = False
    
    def execute_order(self, action, price, timestamp):
        """
        Executa uma ordem com delay e custos realistas.
        
        Args:
            action (int): 1 (comprar), -1 (vender), 0 (manter)
            price (float): Preço atual
            timestamp (datetime): Timestamp da ordem
        """
        # Aplica slippage
        if action == 1:  # Compra
            execution_price = price * (1 + self.slippage)
        elif action == -1:  # Venda
            execution_price = price * (1 - self.slippage)
        else:
            execution_price = price
            
        # Calcula custos
        cost = abs(action) * self.transaction_fee
        
        # Executa a ordem
        if action == 1 and self.position == 0:  # Compra
            self.position = 1
            self.entry_price = execution_price
            self.balance -= execution_price * (1 + cost)
            self.trades.append({
                'timestamp': timestamp,
                'action': 'buy',
                'price': execution_price,
                'balance': self.balance,
                'cost': cost * execution_price
            })
        elif action == -1 and self.position == 1:  # Venda
            self.position = 0
            profit = execution_price - self.entry_price
            self.balance += execution_price * (1 - cost)
            self.trades.append({
                'timestamp': timestamp,
                'action': 'sell',
                'price': execution_price,
                'balance': self.balance,
                'profit': profit,
                'cost': cost * execution_price
            })
            self.stop_triggered = False
            
        # Registra curva de equity
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.balance + (self.position * execution_price)
        })
    
    def run_backtest(self, df, signals, confidences=None, min_confidence=0.6):
        """
        Executa o backtest com os sinais fornecidos.
        
        Args:
            df (pd.DataFrame): DataFrame com dados de preço
            signals (pd.Series): Série com sinais (1: compra, -1: venda, 0: manter)
            confidences (pd.Series): Série com confianças (0-1)
            min_confidence (float): Filtro de confiança mínima
        """
        self.reset()
        
        # Adiciona delay aos sinais
        delayed_signals = signals.shift(self.execution_delay).fillna(0)
        if confidences is not None:
            delayed_confidences = confidences.shift(self.execution_delay).fillna(0)
        else:
            delayed_confidences = pd.Series([1.0]*len(df), index=df.index)
        
        for i in range(len(df)):
            self.current_step = i
            price = df.iloc[i]['close']
            timestamp = df.iloc[i]['date']
            action = delayed_signals.iloc[i]
            confidence = delayed_confidences.iloc[i]
            
            # Filtro de confiança
            if abs(action) > 0 and confidence < min_confidence:
                action = 0
            
            # Stop loss/take profit
            if self.position == 1:
                change = (price - self.entry_price) / self.entry_price
                if self.stop_loss is not None and change <= -self.stop_loss:
                    action = -1
                    self.stop_triggered = True
                elif self.take_profit is not None and change >= self.take_profit:
                    action = -1
                    self.stop_triggered = True
            
            self.execute_order(action, price, timestamp)
    
    def calculate_metrics(self):
        """Calcula métricas de performance do backtest."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_profit': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Converte trades para DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # Filtra apenas trades de venda (completos)
        sell_trades = trades_df[trades_df['action'] == 'sell']
        
        # Métricas básicas
        total_trades = len(sell_trades)
        winning_trades = len(sell_trades[sell_trades['profit'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Lucro/Prejuízo
        total_profit = sell_trades['profit'].sum()
        total_costs = sell_trades['cost'].sum()
        net_profit = total_profit - total_costs
        
        # Profit Factor
        gross_profit = sell_trades[sell_trades['profit'] > 0]['profit'].sum()
        gross_loss = abs(sell_trades[sell_trades['profit'] < 0]['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Drawdown
        equity_curve = pd.DataFrame(self.equity_curve)
        equity_curve['equity'] = equity_curve['equity'].astype(float)
        rolling_max = equity_curve['equity'].expanding().max()
        drawdown = (equity_curve['equity'] - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Sharpe Ratio (assumindo retorno livre de risco = 0)
        returns = equity_curve['equity'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_costs': total_costs,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def plot_results(self, df):
        """
        Gera gráficos interativos com os resultados do backtest.
        
        Args:
            df (pd.DataFrame): DataFrame com dados de preço
        """
        # Cria subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Preço e Sinais', 'Equity Curve', 'Drawdown'),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Preço e Sinais
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['close'],
                name='Preço',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Adiciona sinais de compra/venda
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            buy_signals = trades_df[trades_df['action'] == 'buy']
            sell_signals = trades_df[trades_df['action'] == 'sell']
            
            fig.add_trace(
                go.Scatter(
                    x=buy_signals['timestamp'],
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
                    x=sell_signals['timestamp'],
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
        equity_df = pd.DataFrame(self.equity_curve)
        fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['equity'],
                name='Equity',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        # Drawdown
        equity_curve = equity_df['equity'].astype(float)
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=drawdown,
                name='Drawdown',
                line=dict(color='red')
            ),
            row=3, col=1
        )
        
        # Layout
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text='Resultados do Backtest',
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def generate_report(self, df):
        """
        Gera relatório completo do backtest.
        
        Args:
            df (pd.DataFrame): DataFrame com dados de preço
        """
        metrics = self.calculate_metrics()
        
        print("\n=== Relatório de Backtest ===")
        print(f"\nCapital Inicial: ${self.initial_balance:,.2f}")
        print(f"Capital Final: ${metrics['net_profit'] + self.initial_balance:,.2f}")
        print(f"Lucro Líquido: ${metrics['net_profit']:,.2f}")
        print(f"Retorno Total: {(metrics['net_profit'] / self.initial_balance * 100):,.2f}%")
        print(f"\nTotal de Trades: {metrics['total_trades']}")
        print(f"Trades Vencedores: {metrics['winning_trades']}")
        print(f"Taxa de Acerto: {metrics['win_rate']*100:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"\nMáximo Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"\nCustos Totais: ${metrics['total_costs']:,.2f}")
        
        # Plota resultados
        fig = self.plot_results(df)
        fig.show()

def main():
    """Exemplo de uso do sistema de backtest com filtro de confiança e stop loss/take profit."""
    # Carrega dados
    df = pd.read_csv('processed_data/BTC.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Carrega modelo e scaler
    model = joblib.load('models/BTC_xgb_cls.joblib')
    scaler = joblib.load('models/BTC_scaler.joblib')
    
    # Prepara features
    features = [
        'close', 'ma_7', 'ma_30', 'volatility', 'volume_change',
        'rsi', 'macd', 'macd_signal', 'bb_middle', 'bb_upper', 'bb_lower'
    ]
    # Remove inf, -inf e NaN
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features)
    X = df[features].values
    X_scaled = scaler.transform(X)
    
    # Probabilidades do modelo
    proba = model.predict_proba(X_scaled)
    preds = model.predict(X_scaled)
    # Confiança: maior probabilidade entre as classes
    confidences = pd.Series(proba.max(axis=1), index=df.index)
    signals = pd.Series(preds, index=df.index).map({0: -1, 1: 0, 2: 1})
    
    # Backtest com filtro de confiança e stop loss/take profit
    backtest = BacktestSystem(
        initial_balance=10000.0,
        transaction_fee=0.001,
        slippage=0.0005,
        execution_delay=1,
        stop_loss=0.02,      # 2% stop loss
        take_profit=0.04     # 4% take profit
    )
    
    backtest.run_backtest(df, signals, confidences=confidences, min_confidence=0.6)
    backtest.generate_report(df)

if __name__ == "__main__":
    main() 