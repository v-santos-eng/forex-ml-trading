from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.team import Team
import os
from datetime import datetime

class ForexAnalysisTeam:
    """Team de agentes especializados em análise forex"""
    
    def __init__(self):
        # Agente especializado em análise técnica
        self.technical_agent = Agent(
            name="Technical Analysis Agent",
            role="Especialista em análise técnica de forex",
            model=Claude(id="claude-3-5-sonnet-20241022"),
            tools=[
                ReasoningTools(add_instructions=True),
                YFinanceTools(stock_price=True, analyst_recommendations=True)
            ],
            instructions=[
                "Você é um especialista em análise técnica de forex.",
                "Foque em indicadores técnicos: RSI, MACD, Bollinger Bands, Moving Averages.",
                "Identifique padrões de candlestick e formações gráficas.",
                "Determine níveis de suporte e resistência precisos.",
                "Use timeframes múltiplos para análise completa."
            ],
            markdown=True,
            show_tool_calls=True
        )
        
        # Agente especializado em análise fundamental
        self.fundamental_agent = Agent(
            name="Fundamental Analysis Agent",
            role="Especialista em análise fundamental e econômica",
            model=Claude(id="claude-3-5-sonnet-20241022"),
            tools=[
                ReasoningTools(add_instructions=True),
                DuckDuckGoTools(),
                YFinanceTools(company_news=True, company_info=True)
            ],
            instructions=[
                "Você é um especialista em análise fundamental de moedas.",
                "Analise dados econômicos: PIB, inflação, taxas de juros, emprego.",
                "Monitore políticas de bancos centrais (Fed, ECB, BoE, BoJ).",
                "Avalie eventos geopolíticos e seu impacto nas moedas.",
                "Considere fluxos de capital e sentiment de mercado."
            ],
            markdown=True,
            show_tool_calls=True
        )
        
        # Agente de risk management
        self.risk_agent = Agent(
            name="Risk Management Agent",
            role="Especialista em gestão de risco e money management",
            model=Claude(id="claude-3-5-sonnet-20241022"),
            tools=[ReasoningTools(add_instructions=True)],
            instructions=[
                "Você é um especialista em gestão de risco para forex.",
                "Calcule position sizing adequado baseado em volatilidade.",
                "Determine stop-loss e take-profit otimizados.",
                "Avalie correlações entre pares de moedas.",
                "Forneça estratégias de hedging quando apropriado.",
                "Sempre enfatize a importância do risk management."
            ],
            markdown=True
        )
        
        # Team de análise forex
        self.forex_team = Team(
            agents=[self.technical_agent, self.fundamental_agent, self.risk_agent],
            instructions=[
                "Trabalhem juntos para fornecer análise completa de forex.",
                "Combinem análise técnica, fundamental e gestão de risco.",
                "Forneçam recomendações consensuais quando possível.",
                "Destaquem divergências de opinião quando existirem."
            ],
            show_tool_calls=True
        )
    
    def comprehensive_analysis(self, currency_pair="EUR/USD"):
        """Análise completa de um par de moedas"""
        prompt = f"""
        Realizem uma análise completa e colaborativa do par {currency_pair}:
        
        **Technical Analysis Agent:**
        - Analise gráficos em timeframes H1, H4 e D1
        - Identifique tendências, suportes e resistências
        - Avalie indicadores técnicos principais
        - Procure por padrões de reversão ou continuação
        
        **Fundamental Analysis Agent:**
        - Analise dados econômicos recentes das moedas envolvidas
        - Monitore decisões de política monetária
        - Avalie eventos econômicos programados
        - Considere fatores geopolíticos relevantes
        
        **Risk Management Agent:**
        - Calcule volatilidade atual do par
        - Sugira position sizing apropriado
        - Determine níveis de stop-loss e take-profit
        - Avalie correlações com outros pares
        
        **Conclusão da Equipe:**
        - Forneçam uma recomendação consensual
        - Destaquem pontos de concordância e divergência
        - Incluam cenários alternativos
        - Enfatizem disclaimers de risco
        """
        
        return self.forex_team.print_response(
            prompt,
            stream=True,
            show_full_reasoning=True
        )
    
    def daily_market_briefing(self):
        """Briefing diário do mercado forex"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""
        Preparem um briefing diário do mercado forex para {today}:
        
        **Technical Analysis Agent:**
        - Analise os principais pares: EUR/USD, GBP/USD, USD/JPY, AUD/USD
        - Identifique breakouts ou reversões importantes
        - Destaque níveis técnicos chave para hoje
        
        **Fundamental Analysis Agent:**
        - Resumo dos eventos econômicos de hoje
        - Impacto esperado de releases econômicos
        - Notícias relevantes dos bancos centrais
        - Desenvolvimentos geopolíticos importantes
        
        **Risk Management Agent:**
        - Avalie volatilidade geral do mercado
        - Identifique pares com maior/menor risco
        - Sugira estratégias de diversificação
        - Alertas de correlação entre pares
        
        **Resumo Executivo:**
        - Top 3 oportunidades do dia
        - Principais riscos a monitorar
        - Bias geral do mercado (risk-on/risk-off)
        """
        
        return self.forex_team.print_response(
            prompt,
            stream=True,
            show_full_reasoning=True
        )
    
    def strategy_development(self, strategy_type="scalping"):
        """Desenvolve estratégias de trading específicas"""
        prompt = f"""
        Desenvolvam uma estratégia de {strategy_type} para forex:
        
        **Technical Analysis Agent:**
        - Defina indicadores técnicos apropriados para {strategy_type}
        - Estabeleça critérios de entrada e saída
        - Determine timeframes ideais
        - Identifique condições de mercado favoráveis
        
        **Fundamental Analysis Agent:**
        - Avalie como eventos fundamentais afetam a estratégia
        - Identifique horários de maior volatilidade
        - Considere impacto de news releases
        - Sugira pares mais adequados para a estratégia
        
        **Risk Management Agent:**
        - Calcule risk/reward ratio apropriado
        - Defina regras de position sizing
        - Estabeleça critérios de stop-loss dinâmico
        - Crie regras de money management
        
        **Estratégia Final:**
        - Documento completo da estratégia
        - Regras claras de entrada e saída
        - Backtesting sugerido
        - Métricas de performance a monitorar
        """
        
        return self.forex_team.print_response(
            prompt,
            stream=True,
            show_full_reasoning=True
        )

if __name__ == "__main__":
    # Verificar configuração da API
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Configure sua ANTHROPIC_API_KEY como variável de ambiente")
        print("Exemplo: export ANTHROPIC_API_KEY=sk-ant-api03-xxxx")
        exit(1)
    
    # Criar team de análise
    forex_team = ForexAnalysisTeam()
    
    print("Forex Analysis Team iniciado!")
    print("\n=== Briefing Diário do Mercado ===")
    
    # Executar briefing diário
    forex_team.daily_market_briefing()