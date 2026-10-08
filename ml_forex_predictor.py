from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools
from config import config
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class MLForexPredictor:
    """Preditor de Forex usando ML com dados coletados pelo Agno"""
    
    def __init__(self):
        # Agente especializado em coleta e análise de dados
        self.data_agent = Agent(
            name="Data Collection Agent",
            model=Claude(id=config.DEFAULT_MODEL),
            tools=[
                ReasoningTools(add_instructions=True),
                YFinanceTools(
                    stock_price=True,
                    analyst_recommendations=True,
                    company_info=True,
                    company_news=True
                )
            ],
            instructions=[
                "Você é especialista em coleta e análise de dados financeiros.",
                "Colete dados históricos de forex de forma estruturada.",
                "Identifique padrões e tendências nos dados.",
                "Forneça dados em formato JSON quando solicitado.",
                "Foque em dados quantitativos para análise ML."
            ],
            markdown=True,
            show_tool_calls=True
        )
        
        # Agente especializado em ML e predições
        self.ml_agent = Agent(
            name="ML Prediction Agent",
            model=Claude(id=config.DEFAULT_MODEL),
            tools=[ReasoningTools(add_instructions=True)],
            instructions=[
                "Você é especialista em machine learning para forex.",
                "Analise dados históricos para identificar padrões.",
                "Sugira features relevantes para modelos ML.",
                "Forneça insights sobre correlações e tendências.",
                "Explique metodologias de ML aplicáveis ao forex."
            ],
            markdown=True
        )
    
    def collect_forex_data(self, currency_pair="EUR/USD", period="1mo"):
        """Coleta dados históricos de forex"""
        prompt = f"""
        Colete e analise dados históricos para {currency_pair} no período de {period}:
        
        1. **Dados de Preço:**
           - Preços de abertura, máxima, mínima e fechamento
           - Volume de negociação
           - Volatilidade histórica
        
        2. **Análise Técnica:**
           - Calcule médias móveis (SMA 20, 50, 200)
           - RSI e MACD
           - Bollinger Bands
           - Níveis de suporte e resistência
        
        3. **Padrões Identificados:**
           - Tendências de alta/baixa
           - Padrões de reversão
           - Breakouts importantes
        
        4. **Estrutura dos Dados:**
           - Organize os dados de forma tabular
           - Identifique missing values ou anomalias
           - Sugira features para ML
        
        Forneça uma análise detalhada dos dados coletados.
        """
        
        return self.data_agent.print_response(
            prompt,
            stream=True,
            show_full_reasoning=True
        )
    
    def feature_engineering(self, currency_pair="EUR/USD"):
        """Engenharia de features para ML"""
        prompt = f"""
        Desenvolva features para um modelo de ML que prediz {currency_pair}:
        
        1. **Features Técnicas:**
           - Indicadores técnicos (RSI, MACD, Stochastic)
           - Médias móveis e crossovers
           - Volatilidade realizada
           - Price action patterns
        
        2. **Features Fundamentais:**
           - Diferenciais de taxa de juros
           - Indicadores econômicos relativos
           - Sentiment de mercado
           - Correlações com commodities
        
        3. **Features Temporais:**
           - Hora do dia
           - Dia da semana
           - Sessões de trading
           - Sazonalidade
        
        4. **Features de Lag:**
           - Retornos passados (1h, 4h, 1d)
           - Volatilidade passada
           - Volume passado
        
        5. **Feature Selection:**
           - Métodos de seleção de features
           - Análise de correlação
           - Importância das features
        
        Explique como cada feature pode contribuir para a predição.
        """
        
        return self.ml_agent.print_response(
            prompt,
            stream=True,
            show_full_reasoning=True
        )
    
    def model_selection(self, prediction_horizon="1h"):
        """Seleção e comparação de modelos ML"""
        prompt = f"""
        Analise diferentes modelos de ML para predição de forex com horizonte de {prediction_horizon}:
        
        1. **Modelos Tradicionais:**
           - Linear Regression
           - Random Forest
           - Support Vector Machines
           - XGBoost/LightGBM
        
        2. **Modelos de Deep Learning:**
           - LSTM (Long Short-Term Memory)
           - GRU (Gated Recurrent Unit)
           - Transformer models
           - CNN para time series
        
        3. **Modelos Específicos para Finance:**
           - ARIMA/GARCH
           - Vector Autoregression (VAR)
           - Kalman Filters
           - Hidden Markov Models
        
        4. **Ensemble Methods:**
           - Voting classifiers
           - Stacking
           - Blending
        
        5. **Critérios de Avaliação:**
           - Métricas apropriadas (Sharpe ratio, Max drawdown)
           - Backtesting methodology
           - Walk-forward analysis
           - Out-of-sample testing
        
        Para cada modelo, explique:
        - Vantagens e desvantagens
        - Quando usar cada um
        - Hiperparâmetros importantes
        - Complexidade computacional
        """
        
        return self.ml_agent.print_response(
            prompt,
            stream=True,
            show_full_reasoning=True
        )
    
    def risk_management_ml(self):
        """Gestão de risco usando ML"""
        prompt = """
        Desenvolva estratégias de gestão de risco usando machine learning:
        
        1. **Position Sizing com ML:**
           - Kelly Criterion adaptativo
           - Volatility targeting
           - Risk parity dinâmico
           - ML-based position sizing
        
        2. **Stop Loss Dinâmico:**
           - ATR-based stops
           - Volatility-adjusted stops
           - ML-predicted support/resistance
           - Trailing stops inteligentes
        
        3. **Portfolio Risk:**
           - Correlação dinâmica entre pares
           - VaR (Value at Risk) com ML
           - Expected Shortfall
           - Stress testing
        
        4. **Detecção de Anomalias:**
           - Outlier detection
           - Regime change detection
           - Market crash prediction
           - Liquidity risk assessment
        
        5. **Risk Metrics:**
           - Sharpe ratio
           - Maximum drawdown
           - Calmar ratio
           - Sortino ratio
        
        Explique como implementar cada estratégia e suas vantagens.
        """
        
        return self.ml_agent.print_response(
            prompt,
            stream=True,
            show_full_reasoning=True
        )
    
    def backtesting_strategy(self):
        """Estratégia de backtesting para modelos ML"""
        prompt = """
        Desenvolva uma metodologia robusta de backtesting para modelos ML de forex:
        
        1. **Data Preparation:**
           - Train/Validation/Test splits
           - Time series cross-validation
           - Purged cross-validation
           - Embargo periods
        
        2. **Backtesting Framework:**
           - Walk-forward analysis
           - Expanding window vs Rolling window
           - Out-of-sample testing
           - Paper trading simulation
        
        3. **Performance Metrics:**
           - Return metrics (CAGR, volatility)
           - Risk metrics (Sharpe, Sortino, Calmar)
           - Drawdown analysis
           - Win rate e profit factor
        
        4. **Bias Prevention:**
           - Look-ahead bias
           - Survivorship bias
           - Data snooping
           - Overfitting detection
        
        5. **Stress Testing:**
           - Different market regimes
           - Crisis periods
           - High volatility periods
           - Low liquidity conditions
        
        6. **Implementation Details:**
           - Transaction costs
           - Slippage modeling
           - Latency considerations
           - Market impact
        
        Forneça código Python conceitual para implementação.
        """
        
        return self.ml_agent.print_response(
            prompt,
            stream=True,
            show_full_reasoning=True
        )
    
    def generate_trading_signals(self, currency_pair="EUR/USD"):
        """Gera sinais de trading usando análise ML"""
        prompt = f"""
        Gere sinais de trading para {currency_pair} usando abordagem de machine learning:
        
        1. **Análise de Dados Atuais:**
           - Colete dados recentes do par
           - Calcule indicadores técnicos
           - Avalie sentiment de mercado
        
        2. **Feature Analysis:**
           - Identifique features mais relevantes
           - Analise correlações
           - Detecte padrões emergentes
        
        3. **Signal Generation:**
           - Probabilidade de movimento de alta/baixa
           - Força do sinal (confidence score)
           - Horizonte temporal da predição
           - Níveis de entrada sugeridos
        
        4. **Risk Assessment:**
           - Stop loss recomendado
           - Take profit targets
           - Position size sugerido
           - Risk/reward ratio
        
        5. **Market Context:**
           - Condições atuais de mercado
           - Eventos econômicos próximos
           - Correlações com outros ativos
        
        Forneça sinais específicos e acionáveis com justificativas.
        """
        
        return self.ml_agent.print_response(
            prompt,
            stream=True,
            show_full_reasoning=True
        )

def main():
    """Função principal para demonstrar o uso do ML Forex Predictor"""
    
    if not config.validate_config():
        return
    
    print("Iniciando ML Forex Predictor com Agno...")
    
    predictor = MLForexPredictor()
    
    print("\n" + "="*60)
    print("COLETA E ANÁLISE DE DADOS")
    print("="*60)
    
    # Coletar dados
    predictor.collect_forex_data("EUR/USD", "3mo")
    
    print("\n" + "="*60)
    print("ENGENHARIA DE FEATURES")
    print("="*60)
    
    # Feature engineering
    predictor.feature_engineering("EUR/USD")
    
    print("\n" + "="*60)
    print("GERAÇÃO DE SINAIS DE TRADING")
    print("="*60)
    
    # Gerar sinais
    predictor.generate_trading_signals("EUR/USD")
    
    print("\nAnálise ML completa!")
    print("\nPróximos passos:")
    print("- Implemente os modelos sugeridos")
    print("- Execute backtesting rigoroso")
    print("- Teste em paper trading")
    print("- Monitore performance em tempo real")

if __name__ == "__main__":
    main()