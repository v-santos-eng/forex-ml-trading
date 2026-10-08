import os
from typing import List, Dict, Any

class ForexConfig:
    """Configurações centralizadas para o projeto Forex ML"""
    
    # Configurações da API
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Modelo padrão
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    
    # Pares de moedas principais
    MAJOR_PAIRS = [
        "EUR/USD",  # Euro/Dólar
        "GBP/USD",  # Libra/Dólar
        "USD/JPY",  # Dólar/Iene
        "USD/CHF",  # Dólar/Franco Suíço
        "AUD/USD",  # Dólar Australiano/Dólar
        "USD/CAD",  # Dólar/Dólar Canadense
        "NZD/USD"   # Dólar Neozelandês/Dólar
    ]
    
    # Pares de moedas menores
    MINOR_PAIRS = [
        "EUR/GBP",  # Euro/Libra
        "EUR/JPY",  # Euro/Iene
        "GBP/JPY",  # Libra/Iene
        "EUR/CHF",  # Euro/Franco
        "GBP/CHF",  # Libra/Franco
        "AUD/JPY",  # Dólar Australiano/Iene
        "CAD/JPY"   # Dólar Canadense/Iene
    ]
    
    # Timeframes para análise
    TIMEFRAMES = {
        "M1": "1 minuto",
        "M5": "5 minutos",
        "M15": "15 minutos",
        "M30": "30 minutos",
        "H1": "1 hora",
        "H4": "4 horas",
        "D1": "1 dia",
        "W1": "1 semana",
        "MN1": "1 mês"
    }
    
    # Indicadores técnicos
    TECHNICAL_INDICATORS = [
        "RSI",
        "MACD",
        "Bollinger Bands",
        "Moving Averages (SMA, EMA)",
        "Stochastic",
        "Williams %R",
        "ADX",
        "Fibonacci Retracements"
    ]
    
    # Instruções padrão para agentes
    DEFAULT_INSTRUCTIONS = [
        "Você é um especialista em análise de forex.",
        "Use dados atualizados e análise fundamentada.",
        "Apresente informações de forma clara e organizada.",
        "Sempre inclua disclaimers sobre riscos de investimento.",
        "Use tabelas para exibir dados quando apropriado."
    ]
    
    # Instruções específicas por tipo de agente
    AGENT_INSTRUCTIONS = {
        "technical": [
            "Foque em análise técnica e indicadores.",
            "Identifique padrões de preço e formações gráficas.",
            "Determine níveis de suporte e resistência.",
            "Use múltiplos timeframes para confirmação."
        ],
        "fundamental": [
            "Analise dados econômicos e políticas monetárias.",
            "Monitore eventos geopolíticos relevantes.",
            "Avalie sentiment de mercado e fluxos de capital.",
            "Considere correlações com outros ativos."
        ],
        "risk": [
            "Priorize gestão de risco em todas as análises.",
            "Calcule position sizing apropriado.",
            "Determine stop-loss e take-profit otimizados.",
            "Avalie correlações e diversificação."
        ]
    }
    
    # Configurações de risco
    RISK_SETTINGS = {
        "max_risk_per_trade": 0.02,  # 2% do capital por trade
        "max_daily_risk": 0.06,     # 6% do capital por dia
        "max_correlation": 0.7,      # Correlação máxima entre posições
        "min_risk_reward": 1.5       # Risk/Reward mínimo
    }
    
    # Horários de mercado (UTC)
    MARKET_SESSIONS = {
        "sydney": {"open": "21:00", "close": "06:00"},
        "tokyo": {"open": "00:00", "close": "09:00"},
        "london": {"open": "08:00", "close": "17:00"},
        "new_york": {"open": "13:00", "close": "22:00"}
    }
    
    # Eventos econômicos importantes
    ECONOMIC_EVENTS = [
        "NFP (Non-Farm Payrolls)",
        "CPI (Consumer Price Index)",
        "GDP (Gross Domestic Product)",
        "Interest Rate Decisions",
        "Employment Data",
        "Retail Sales",
        "Manufacturing PMI",
        "Services PMI"
    ]
    
    # Bancos centrais
    CENTRAL_BANKS = {
        "USD": "Federal Reserve (Fed)",
        "EUR": "European Central Bank (ECB)",
        "GBP": "Bank of England (BoE)",
        "JPY": "Bank of Japan (BoJ)",
        "CHF": "Swiss National Bank (SNB)",
        "AUD": "Reserve Bank of Australia (RBA)",
        "CAD": "Bank of Canada (BoC)",
        "NZD": "Reserve Bank of New Zealand (RBNZ)"
    }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Valida se as configurações estão corretas"""
        valid = True
        
        if not cls.ANTHROPIC_API_KEY:
            print("ANTHROPIC_API_KEY não configurada!")
            valid = False
        elif not cls.ANTHROPIC_API_KEY.startswith("sk-ant-"):
            print("ANTHROPIC_API_KEY parece inválida!")
            valid = False
        
        if not cls.OPENAI_API_KEY:
            print("OPENAI_API_KEY não configurada!")
            valid = False
        elif not cls.OPENAI_API_KEY.startswith("sk-"):
            print("OPENAI_API_KEY parece inválida!")
            valid = False
        else:
            print("OPENAI_API_KEY configurada com sucesso!")
        
        if valid:
            print("Configurações validadas com sucesso!")
        
        return valid
    
    @classmethod
    def get_pair_info(cls, pair: str) -> Dict[str, Any]:
        """Retorna informações sobre um par de moedas"""
        base_currency = pair.split("/")[0]
        quote_currency = pair.split("/")[1]
        
        return {
            "pair": pair,
            "base_currency": base_currency,
            "quote_currency": quote_currency,
            "base_bank": cls.CENTRAL_BANKS.get(base_currency, "Unknown"),
            "quote_bank": cls.CENTRAL_BANKS.get(quote_currency, "Unknown"),
            "is_major": pair in cls.MAJOR_PAIRS,
            "is_minor": pair in cls.MINOR_PAIRS
        }
    
    @classmethod
    def get_trading_session_info(cls) -> Dict[str, str]:
        """Retorna informações sobre as sessões de trading"""
        from datetime import datetime
        
        current_hour = datetime.utcnow().hour
        active_sessions = []
        
        for session, times in cls.MARKET_SESSIONS.items():
            open_hour = int(times["open"].split(":")[0])
            close_hour = int(times["close"].split(":")[0])
            
            # Ajustar para sessões que cruzam meia-noite
            if open_hour > close_hour:
                if current_hour >= open_hour or current_hour < close_hour:
                    active_sessions.append(session)
            else:
                if open_hour <= current_hour < close_hour:
                    active_sessions.append(session)
        
        return {
            "current_hour_utc": current_hour,
            "active_sessions": active_sessions,
            "all_sessions": cls.MARKET_SESSIONS
        }

# Instância global de configuração
config = ForexConfig()

# Função de conveniência para validar configuração
def check_config() -> bool:
    """Verifica se a configuração está válida"""
    return config.validate_config()

if __name__ == "__main__":
    # Teste das configurações
    print("Testando configurações...")
    
    if check_config():
        print(f"Pares principais: {len(config.MAJOR_PAIRS)}")
        print(f"Indicadores técnicos: {len(config.TECHNICAL_INDICATORS)}")
        print(f"Bancos centrais: {len(config.CENTRAL_BANKS)}")
        
        # Exemplo de uso
        pair_info = config.get_pair_info("EUR/USD")
        print(f"\nInformações do EUR/USD:")
        for key, value in pair_info.items():
            print(f"  {key}: {value}")
        
        session_info = config.get_trading_session_info()
        print(f"\nSessões ativas: {session_info['active_sessions']}")
    else:
        print("Configuração inválida!")