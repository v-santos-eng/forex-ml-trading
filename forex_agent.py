from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
import os

# Configurar a chave da API do Anthropic
# Certifique-se de definir sua ANTHROPIC_API_KEY como variável de ambiente

class ForexMLAgent:
    def __init__(self):
        self.agent = Agent(
            name="Forex ML Agent",
            model=Claude(id="claude-3-5-sonnet-20241022"),
            tools=[
                ReasoningTools(add_instructions=True),
                YFinanceTools(
                    stock_price=True, 
                    analyst_recommendations=True, 
                    company_info=True, 
                    company_news=True
                ),
                DuckDuckGoTools()
            ],
            instructions=[
                "Você é um especialista em análise de forex e machine learning.",
                "Analise dados de moedas e pares de forex com foco em EUR/USD, GBP/USD, USD/JPY.",
                "Use tabelas para exibir dados de forma clara.",
                "Forneça insights baseados em análise técnica e fundamental.",
                "Considere fatores econômicos globais em suas análises.",
                "Sempre inclua disclaimers sobre riscos de investimento."
            ],
            markdown=True,
            show_tool_calls=True
        )
    
    def analyze_currency_pair(self, pair="EUR/USD"):
        """Analisa um par de moedas específico"""
        prompt = f"""
        Faça uma análise completa do par de moedas {pair}:
        
        1. Obtenha dados históricos recentes
        2. Analise tendências e padrões
        3. Considere fatores econômicos relevantes
        4. Forneça uma perspectiva de curto e médio prazo
        5. Identifique níveis de suporte e resistência
        6. Sugira estratégias de trading (com disclaimers apropriados)
        
        Use dados atualizados e reasoning para fundamentar sua análise.
        """
        
        return self.agent.print_response(
            prompt,
            stream=True,
            show_full_reasoning=True,
            stream_intermediate_steps=True
        )
    
    def market_overview(self):
        """Fornece uma visão geral do mercado forex"""
        prompt = """
        Forneça uma visão geral completa do mercado forex atual:
        
        1. Análise dos principais pares de moedas (EUR/USD, GBP/USD, USD/JPY, USD/CHF)
        2. Fatores econômicos globais influenciando o mercado
        3. Eventos econômicos importantes da semana
        4. Sentiment geral do mercado
        5. Oportunidades e riscos identificados
        
        Use dados em tempo real e reasoning para sua análise.
        """
        
        return self.agent.print_response(
            prompt,
            stream=True,
            show_full_reasoning=True,
            stream_intermediate_steps=True
        )
    
    def predict_trend(self, pair="EUR/USD", timeframe="1 semana"):
        """Faz previsões de tendência para um par de moedas"""
        prompt = f"""
        Faça uma previsão de tendência para {pair} no período de {timeframe}:
        
        1. Analise dados históricos e padrões
        2. Considere indicadores técnicos relevantes
        3. Avalie fatores fundamentais
        4. Use reasoning para justificar sua previsão
        5. Forneça cenários otimista, realista e pessimista
        6. Inclua níveis de entrada, stop-loss e take-profit sugeridos
        
        IMPORTANTE: Inclua disclaimers sobre riscos e que esta não é uma recomendação de investimento.
        """
        
        return self.agent.print_response(
            prompt,
            stream=True,
            show_full_reasoning=True,
            stream_intermediate_steps=True
        )

if __name__ == "__main__":
    # Verificar se a chave da API está configurada
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ATENÇÃO: Configure sua ANTHROPIC_API_KEY como variável de ambiente")
        print("Exemplo: export ANTHROPIC_API_KEY=sk-ant-api03-xxxx")
        exit(1)
    
    # Criar e usar o agente
    forex_agent = ForexMLAgent()
    
    print("Forex ML Agent iniciado!")
    print("\n=== Análise de Mercado Forex ===")
    
    # Exemplo de uso
    forex_agent.market_overview()