import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import joblib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from utils_ai import ask_ai

# Configurações
MODEL_DIR = 'models'
DATA_DIR = 'processed_data'
SEED = 42

# Features usadas no modelo
DEFAULT_FEATURES = [
    'close', 'ma_7', 'ma_30', 'volatility', 'volume_change',
    'rsi', 'macd', 'macd_signal', 'bb_middle', 'bb_upper', 'bb_lower'
]

class TradingBot:
    def __init__(self, symbol, model_type='ensemble', use_rl=False, position_size=0.2, stop_loss=0.02, take_profit=0.04, confidence_threshold=0.55, dynamic_confidence=False, confidence_window=20, features=None, trend_filter=False):
        """
        Inicializa o bot de trading.
        
        Args:
            symbol (str): Símbolo da criptomoeda (BTC, ETH, etc.)
            model_type (str): Tipo de modelo ('rf', 'xgb', 'ensemble')
            use_rl (bool): Se deve usar o modelo de RL junto com o tradicional
            position_size (float): Fração do saldo a ser usada por trade (ex: 0.2 = 20%)
            stop_loss (float): Stop loss em % (ex: 0.02 para 2%)
            take_profit (float): Take profit em % (ex: 0.04 para 4%)
            confidence_threshold (float): Limiar de confiança estático
            dynamic_confidence (bool): Se True, usa limiar dinâmico baseado em rolling window
            confidence_window (int): Tamanho da janela para limiar dinâmico
            features (list): Lista de features a serem usadas pelo modelo
            trend_filter (bool): Se True, só permite trades na direção da tendência (MA30)
        """
        self.symbol = symbol
        self.model_type = model_type
        self.use_rl = use_rl
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence_threshold = confidence_threshold
        self.dynamic_confidence = dynamic_confidence
        self.confidence_window = confidence_window
        self.recent_confidences = []
        self.features = features if features is not None else DEFAULT_FEATURES
        self.scaler = joblib.load(os.path.join(MODEL_DIR, f'{symbol}_scaler.joblib'))
        self.trend_filter = trend_filter
        
        # Carrega os modelos tradicionais
        if model_type == 'ensemble':
            self.rf_model = joblib.load(os.path.join(MODEL_DIR, f'{symbol}_rf_cls.joblib'))
            self.xgb_model = joblib.load(os.path.join(MODEL_DIR, f'{symbol}_xgb_cls.joblib'))
        elif model_type == 'rf':
            self.model = joblib.load(os.path.join(MODEL_DIR, f'{symbol}_rf_cls.joblib'))
        else:  # xgb
            self.model = joblib.load(os.path.join(MODEL_DIR, f'{symbol}_xgb_cls.joblib'))
            self.xgb_map = {-1: 0, 0: 1, 1: 2}  # Mapeamento para XGBoost
        
        # Carrega o modelo RL se solicitado
        if use_rl:
            self.rl_model = PPO.load(os.path.join(MODEL_DIR, f"{symbol}_ppo"))
            self.rl_env = CryptoTradingEnv(symbol)
        
        # Carrega os dados
        self.data = pd.read_csv(os.path.join(DATA_DIR, f'{symbol}.csv'))
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        self.entry_price = None  # Para controle de stop
        
    def get_action(self, current_state, available_balance=None, in_position=False, entry_price=None, explain_with_ai=False, openai_api_key=None):
        """
        Decide a ação (comprar, vender, manter) baseado no estado atual.
        Se explain_with_ai=True, retorna também uma explicação da IA.
        
        Args:
            current_state (pd.Series): Estado atual com as features
            available_balance (float): Saldo disponível para calcular o tamanho da posição
            in_position (bool): Se está em posição comprada
            entry_price (float): Preço de entrada da posição atual
            explain_with_ai (bool): Se True, retorna também uma explicação da IA
            openai_api_key (str): Chave da API da IA
            
        Returns:
            tuple: (action, confidence, position_size_usd, explicacao)
                - action: 1 (comprar), -1 (vender), 0 (manter)
                - confidence: confiança da decisão (0-1)
                - position_size_usd: valor em USD a ser usado na operação (ou None se não comprar)
                - explicacao: explicação da IA (ou None se não for solicitado)
        """
        # Se está em posição, verifica stop-loss/take-profit
        if in_position and entry_price is not None:
            current_price = current_state['close']
            change = (current_price - entry_price) / entry_price
            if self.stop_loss is not None and change <= -self.stop_loss:
                return -1, 1.0, None, None  # Sinal de venda por stop-loss
            if self.take_profit is not None and change >= self.take_profit:
                return -1, 1.0, None, None  # Sinal de venda por take-profit
        
        # Prepara o estado para o modelo
        state_scaled = self.scaler.transform(current_state[self.features].values.reshape(1, -1))
        
        # Obtém predição do modelo tradicional
        if self.model_type == 'ensemble':
            rf_pred = self.rf_model.predict_proba(state_scaled)[0]
            xgb_pred = self.xgb_model.predict_proba(state_scaled)[0]
            # Combina as predições (média ponderada)
            combined_pred = (rf_pred + xgb_pred) / 2
            action_idx = np.argmax(combined_pred)
            confidence = combined_pred[action_idx]
            # Mapeia para {-1, 0, 1}
            action = action_idx - 1
        else:
            if self.model_type == 'rf':
                pred_proba = self.model.predict_proba(state_scaled)[0]
                action_idx = np.argmax(pred_proba)
                confidence = pred_proba[action_idx]
                action = action_idx - 1
            else:  # xgb
                pred_proba = self.model.predict_proba(state_scaled)[0]
                action_idx = np.argmax(pred_proba)
                confidence = pred_proba[action_idx]
                # Mapeia de volta para {-1, 0, 1}
                action = list(self.xgb_map.keys())[list(self.xgb_map.values()).index(action_idx)]
        
        # Se estiver usando RL, combina com a predição do RL
        if self.use_rl:
            rl_action, _ = self.rl_model.predict(self.rl_env._get_observation())
            # Combina as predições (média ponderada)
            if rl_action == 1:  # RL sugere comprar
                action = 1 if action >= 0 else action
            elif rl_action == 2:  # RL sugere vender
                action = -1 if action <= 0 else action
            # Ajusta a confiança baseado no RL
            confidence = (confidence + 0.5) / 2
        
        # Atualiza lista de confianças recentes
        self.recent_confidences.append(confidence)
        if len(self.recent_confidences) > self.confidence_window:
            self.recent_confidences.pop(0)
        
        # Determina limiar de confiança
        if self.dynamic_confidence and len(self.recent_confidences) >= self.confidence_window:
            threshold = np.mean(self.recent_confidences)
        else:
            threshold = self.confidence_threshold
        
        # Aplica o limiar de confiança
        if confidence < threshold:
            action = 0
            confidence = 1.0 - confidence
        
        # Trend filter: só permite trades na direção da tendência
        if self.trend_filter:
            if action == 1 and current_state['close'] <= current_state['ma_30']:
                action = 0
            if action == -1 and current_state['close'] >= current_state['ma_30']:
                action = 0
        
        # Calcula o tamanho da posição em USD
        position_size_usd = None
        if action == 1 and available_balance is not None:
            position_size_usd = available_balance * self.position_size
        
        explicacao = None
        if explain_with_ai and openai_api_key:
            # Monta prompt para IA
            prompt = (
                f"Explique de forma simples e curta por que o bot tomou a decisão '{['hold','buy','sell'][action]}', "
                f"com confiança {confidence:.2f}, para o ativo {self.symbol}.\n"
                f"Features: {dict(zip(self.features, current_state[self.features].values))}\n"
                f"Em posição: {in_position}, Preço de entrada: {entry_price}, Stop: {self.stop_loss}, Take: {self.take_profit}."
            )
            try:
                explicacao = ask_ai(prompt, openai_api_key)
            except Exception as e:
                explicacao = f"Erro ao consultar IA: {e}"
        if explain_with_ai:
            return action, confidence, position_size_usd, explicacao
        return action, confidence, position_size_usd

class CryptoTradingEnv(gym.Env):
    """
    Ambiente de trading para criptomoedas usando Gymnasium.
    """
    def __init__(self, symbol, initial_balance=10000.0, transaction_fee=0.001):
        super(CryptoTradingEnv, self).__init__()
        
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # Carrega os dados e limpa NaN/infinito nas features
        self.data = pd.read_csv(os.path.join(DATA_DIR, f'{symbol}.csv'))
        self.data['date'] = pd.to_datetime(self.data['date'])
        n_before = len(self.data)
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data = self.data.dropna(subset=DEFAULT_FEATURES)
        n_after = len(self.data)
        if n_after < n_before:
            print(f"[RL ENV] Removidas {n_before - n_after} linhas com NaN/infinito nas features do ambiente RL.")
        self.data = self.data.reset_index(drop=True)
        
        # Define o espaço de observação (features + posição atual)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(DEFAULT_FEATURES) + 2,), dtype=np.float32
        )
        
        # Define o espaço de ação (0: manter, 1: comprar, 2: vender)
        self.action_space = spaces.Discrete(3)
        
        # Inicializa o ambiente
        self.reset()
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0: sem posição, 1: comprado
        self.total_value = self.balance
        self.returns = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Retorna o estado atual do ambiente."""
        # Garante que não acessa índice inválido
        idx = min(self.current_step, len(self.data) - 1)
        current_data = self.data.iloc[idx]
        obs = current_data[DEFAULT_FEATURES].values
        # Adiciona informações de posição e saldo
        obs = np.append(obs, [self.position, self.balance])
        # Converte para float antes dos checks
        obs = obs.astype(np.float32)
        # Sanitiza para evitar NaN/infinito
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"[RL ENV] Observação com NaN/infinito detectada no step {self.current_step}. Substituindo por zeros.")
            obs = np.zeros_like(obs)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return obs
    
    def step(self, action):
        """
        Executa uma ação no ambiente.
        
        Args:
            action (int): 0 (manter), 1 (comprar), 2 (vender)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        idx = min(self.current_step, len(self.data) - 1)
        current_price = self.data.iloc[idx]['close']
        next_idx = min(self.current_step + 1, len(self.data) - 1)
        next_price = self.data.iloc[next_idx]['close']
        
        # Executa a ação
        reward = 0
        if action == 1 and self.position == 0:  # Comprar
            self.position = 1
            self.balance *= (1 - self.transaction_fee)
        elif action == 2 and self.position == 1:  # Vender
            self.position = 0
            self.balance *= (1 - self.transaction_fee)
        
        # Calcula o retorno
        if self.position == 1:
            price_change = (next_price - current_price) / current_price
            reward = price_change
        
        # Atualiza o valor total
        self.total_value = self.balance * (1 + reward if self.position == 1 else 1)
        self.returns.append(reward)
        
        # Avança para o próximo passo
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Calcula o Sharpe Ratio como recompensa adicional
        if len(self.returns) > 1:
            sharpe = np.mean(self.returns) / (np.std(self.returns) + 1e-6)
            reward += sharpe * 0.1  # Peso para o Sharpe Ratio
        
        # Sanitiza recompensa
        if np.isnan(reward) or np.isinf(reward):
            print(f"[RL ENV] Recompensa com NaN/infinito detectada no step {self.current_step}. Substituindo por zero.")
            reward = 0.0
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=1e6, neginf=-1e6))
        
        return self._get_observation(), reward, done, False, {}

def main():
    """Exemplo de uso do bot de trading."""
    # Testa com BTC
    symbol = 'BTC'
    print(f"\nTestando bot de trading para {symbol}")
    
    # Cria o bot com ensemble e RL
    bot = TradingBot(symbol, model_type='ensemble', use_rl=True)
    
    # Testa em alguns estados
    test_data = pd.read_csv(os.path.join(DATA_DIR, f'{symbol}.csv'))
    for i in range(5):
        state = test_data.iloc[i]
        action, confidence, position_size_usd = bot.get_action(state)
        action_str = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[action]
        print(f"Estado {i}: Ação = {action_str} (Confiança: {confidence:.2f}), Tamanho da Posição: {position_size_usd} USD")

if __name__ == "__main__":
    main() 