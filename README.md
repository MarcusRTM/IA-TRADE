# QuantTrade AI

QuantTrade AI é uma plataforma de trading quantitativo com interface web, que utiliza machine learning e inteligência artificial para prever sinais de compra e venda de criptomoedas, realizar backtests, simulações e análises de performance.

## Sumário
- [Funcionalidades](#funcionalidades)
- [Como rodar o projeto](#como-rodar-o-projeto)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Como o projeto foi feito](#como-o-projeto-foi-feito)
- [Personalização e Extensões](#personalização-e-extensões)

---

## Funcionalidades

- **Dashboard interativo** com gráficos de preço, candles, volume, RSI e área de predição destacada.
- **Treinamento de modelos** de machine learning (Random Forest, XGBoost, Ensemble).
- **Backtesting** com controle de parâmetros, modo agressivo e métricas detalhadas.
- **Simulação em tempo real** (em desenvolvimento).
- **Assistente IA** para explicação automática das decisões do bot.
- **Análise de sentimento de mercado** via integração com CryptoPanic e OpenAI.
- **Exportação de relatórios** em CSV.
- **Upload de previsões externas** para análise.

---

## Como rodar o projeto

### 1. Pré-requisitos

- Python 3.8+
- (Opcional) Conta OpenAI para explicações automáticas
- (Opcional) Conta CryptoPanic para análise de sentimento

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Prepare os dados

- Os dados históricos já estão em `processed_data/` (ex: `BTC.csv`, `ETH.csv`).
- Se quiser processar novos dados, utilize os scripts da pasta.

### 4. Treine os modelos (opcional)

Se quiser treinar modelos do zero:

```bash
python train_models.py
```

### 5. Execute a interface web

```bash
streamlit run quanttrade_streamlit.py
```

Acesse o app em [http://localhost:8501](http://localhost:8501).

---

## Estrutura do Projeto

```
.
├── quanttrade_streamlit.py      # Interface principal (Streamlit)
├── trading_bot.py               # Lógica do bot de trading e IA
├── run_trading_bot.py           # Funções de backtest e métricas
├── train_models.py              # Script para treinar modelos
├── utils_ai.py                  # Utilitários de IA (OpenAI)
├── processed_data/              # Dados históricos prontos para uso
├── models/                      # Modelos treinados (joblib)
├── requirements.txt             # Dependências do projeto
└── ...                          # Outros scripts utilitários
```

---

## Como o projeto foi feito

### 1. **Aquisição e Processamento de Dados**
- Os dados históricos de criptomoedas são processados e salvos em `processed_data/`.
- Features extraídas: preço de fechamento, médias móveis, volatilidade, volume, RSI, MACD, bandas de Bollinger, etc.

### 2. **Treinamento de Modelos**
- Modelos de machine learning (Random Forest, XGBoost, Ensemble) são treinados para prever sinais de compra/venda/hold.
- O treinamento utiliza apenas dados até 31/12/2024 para evitar lookahead bias.
- Os modelos e scalers são salvos em `models/`.

### 3. **Bot de Trading**
- O `TradingBot` carrega o modelo e faz previsões a cada candle.
- Parâmetros ajustáveis: limiar de confiança, stop loss, take profit, tamanho da posição, filtro de tendência, modo agressivo.
- O bot pode usar reinforcement learning (PPO) opcionalmente.

### 4. **Backtesting**
- O backtest simula operações com base nos sinais do modelo, aplicando taxas, slippage e stops.
- Métricas calculadas: lucro total, win rate, profit factor, drawdown, Sharpe ratio, etc.
- O modo agressivo reduz o limiar de confiança e desativa filtros para mais operações.

### 5. **Interface Web (Streamlit)**
- Navegação por abas: Dashboard, Model Training, Backtesting, Simulação, Performance, Assistente IA, Sentimento de Mercado.
- Gráficos interativos com Plotly: candles, preço, volume, RSI, comparativo com buy & hold.
- Upload de CSV para análise de previsões externas.
- Exportação de relatórios em CSV.
- Painel de alertas da IA com explicações automáticas das decisões do bot.

### 6. **Explicação Automática das Decisões**
- O bot pode consultar a OpenAI para explicar, em linguagem natural, o motivo de cada decisão de trade, com base nas features e contexto.

### 7. **Análise de Sentimento**
- Integração com CryptoPanic para buscar notícias e classificar o sentimento via OpenAI.

---

## Personalização e Extensões

- **Adicionar novos ativos:** basta colocar o CSV em `processed_data/` e treinar o modelo.
- **Ajustar parâmetros do bot:** use o painel de backtesting para experimentar diferentes estratégias.
- **Integrar novas fontes de dados ou modelos:** a arquitetura é modular e permite fácil extensão.
- **Explicações IA:** configure sua chave OpenAI em Settings para ativar explicações automáticas.

---

## Dúvidas?

Use a aba "Assistente IA" no app para tirar dúvidas! 
