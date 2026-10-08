from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools
from agno_ml_tool import MLForexTool
from config import ForexConfig
from typing import Dict, Any, List
import os

class ForexAgentTeam:
    """
    Equipe de agentes especializados para trading de Forex
    """
    
    def __init__(self, model_id: str = "gpt-4o-mini"):
        self.model_id = model_id
        self.ml_tool = MLForexTool()
        
        # Configurar API OpenAI
        if ForexConfig.OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = ForexConfig.OPENAI_API_KEY
        
        self._create_agents()
    
    def _create_agents(self):
        """
        Criar agentes especializados
        """
        
        # Agente de Análise Técnica (ML)
        self.technical_agent = Agent(
            name="Technical Analyst",
            role="Especialista em análise técnica usando Machine Learning",
            model=OpenAIChat(id=self.model_id),
            tools=[self.ml_tool],
            instructions=[
                "Use machine learning para analisar padrões técnicos",
                "Forneça predições com níveis de confiança",
                "Compare múltiplos modelos de ML",
                "Explique o raciocínio por trás das predições",
                "Sempre inclua métricas de performance"
            ],
            show_tool_calls=True,
            markdown=True
        )
        
        # Agente de Análise Fundamental
        self.fundamental_agent = Agent(
            name="Fundamental Analyst",
            role="Especialista em análise fundamental e notícias econômicas",
            model=OpenAIChat(id=self.model_id),
            tools=[
                DuckDuckGoTools(),
                YFinanceTools(company_news=True, company_info=True)
            ],
            instructions=[
                "Analise notícias econômicas relevantes",
                "Considere indicadores macroeconômicos",
                "Avalie eventos geopolíticos",
                "Forneça contexto fundamental para decisões",
                "Sempre cite fontes das informações"
            ],
            show_tool_calls=True,
            markdown=True
        )
        
        # Agente de Gestão de Risco
        self.risk_agent = Agent(
            name="Risk Manager",
            role="Especialista em gestão de risco e controle de posições",
            model=OpenAIChat(id=self.model_id),
            tools=[ReasoningTools(add_instructions=True)],
            instructions=[
                "Avalie riscos de cada operação",
                "Calcule tamanhos de posição apropriados",
                "Defina stop-loss e take-profit",
                "Considere correlações entre pares",
                "Mantenha exposição total controlada",
                "Use raciocínio estruturado para decisões"
            ],
            show_tool_calls=True,
            markdown=True
        )
        
        # Agente Coordenador
        self.coordinator_agent = Agent(
            name="Trading Coordinator",
            role="Coordenador que integra análises e toma decisões finais",
            model=OpenAIChat(id=self.model_id),
            tools=[ReasoningTools(add_instructions=True)],
            instructions=[
                "Integre análises técnica e fundamental",
                "Considere recomendações de gestão de risco",
                "Tome decisões finais de trading",
                "Forneça justificativas claras",
                "Priorize preservação de capital",
                "Use raciocínio passo-a-passo"
            ],
            show_tool_calls=True,
            markdown=True
        )
    
    def analyze_pair(self, pair: str, period: str = "6mo") -> Dict[str, Any]:
        """
        Análise completa de um par de moedas
        
        Args:
            pair: Par de moedas (ex: EURUSD=X)
            period: Período de análise
            
        Returns:
            Dict com análises de todos os agentes
        """
        results = {}
        
        # Análise Técnica (ML)
        print(f"\nAnálise Técnica (ML) para {pair}...")
        technical_query = f"Analise tecnicamente o par {pair} usando machine learning. Inclua predição, confiança, backtest e comparação de modelos."
        technical_analysis = self.technical_agent.run(technical_query)
        results["technical"] = technical_analysis.content
        
        # Análise Fundamental
        print(f"\nAnálise Fundamental para {pair}...")
        base_currency = pair.split("USD")[0] if "USD" in pair else pair[:3]
        quote_currency = "USD" if "USD" in pair else pair[3:6]
        
        fundamental_query = f"Analise as condições fundamentais para {base_currency} e {quote_currency}. Busque notícias econômicas recentes, indicadores e eventos que possam afetar {pair}."
        fundamental_analysis = self.fundamental_agent.run(fundamental_query)
        results["fundamental"] = fundamental_analysis.content
        
        # Gestão de Risco
        print(f"\nAnálise de Risco para {pair}...")
        risk_query = f"Com base nas análises técnica e fundamental de {pair}, avalie os riscos e defina estratégia de gestão de risco incluindo tamanho de posição, stop-loss e take-profit."
        risk_analysis = self.risk_agent.run(risk_query)
        results["risk"] = risk_analysis.content
        
        # Decisão Final
        print(f"\nDecisão Final para {pair}...")
        coordinator_query = f"""Integre as seguintes análises para {pair} e tome uma decisão final de trading:
        
        ANÁLISE TÉCNICA (ML):
        {results['technical']}
        
        ANÁLISE FUNDAMENTAL:
        {results['fundamental']}
        
        GESTÃO DE RISCO:
        {results['risk']}
        
        Forneça uma recomendação final: COMPRA, VENDA ou NEUTRO, com justificativa completa."""
        
        final_decision = self.coordinator_agent.run(coordinator_query)
        results["decision"] = final_decision.content
        
        return results
    
    def multi_pair_analysis(self, pairs: List[str] = None, period: str = "6mo") -> Dict[str, Any]:
        """
        Análise de múltiplos pares para identificar melhores oportunidades
        
        Args:
            pairs: Lista de pares de moedas
            period: Período de análise
            
        Returns:
            Dict com análises e ranking de oportunidades
        """
        if pairs is None:
            pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
        
        print(f"\nAnálise Técnica Comparativa...")
        
        # Análise técnica comparativa
        comparison_query = f"Compare os pares {', '.join(pairs)} usando machine learning. Identifique qual par tem melhor potencial de trading baseado em confiança das predições, performance dos modelos e qualidade dos dados."
        comparison_analysis = self.technical_agent.run(comparison_query)
        
        # Análise de correlações e risco de portfólio
        print(f"\nAnálise de Correlações e Risco de Portfólio...")
        portfolio_query = f"Analise as correlações entre os pares {', '.join(pairs)} e recomende uma estratégia de portfólio que minimize riscos e maximize oportunidades. Considere diversificação e exposição total."
        portfolio_analysis = self.risk_agent.run(portfolio_query)
        
        # Decisão de portfólio
        print(f"\nEstratégia de Portfólio Final...")
        portfolio_decision_query = f"""Com base nas análises:
        
        COMPARAÇÃO TÉCNICA:
        {comparison_analysis.content}
        
        ANÁLISE DE PORTFÓLIO:
        {portfolio_analysis.content}
        
        Crie uma estratégia de portfólio ranqueando os pares por prioridade e definindo alocação de capital."""
        
        portfolio_decision = self.coordinator_agent.run(portfolio_decision_query)
        
        return {
            "pairs_analyzed": pairs,
            "period": period,
            "technical_comparison": comparison_analysis.content,
            "portfolio_analysis": portfolio_analysis.content,
            "portfolio_strategy": portfolio_decision.content
        }
    
    def get_team_status(self) -> Dict[str, str]:
        """
        Status da equipe de agentes
        
        Returns:
            Dict com status de cada agente
        """
        return {
            "technical_agent": f"Ativo - {self.technical_agent.name}",
            "fundamental_agent": f"Ativo - {self.fundamental_agent.name}", 
            "risk_agent": f"Ativo - {self.risk_agent.name}",
            "coordinator_agent": f"Ativo - {self.coordinator_agent.name}",
            "ml_tool": "Integrado - ML Forex Predictor",
            "model": self.model_id
        }

# Função de conveniência para uso rápido
def quick_forex_analysis(pair: str = "EURUSD=X", period: str = "6mo") -> Dict[str, Any]:
    """
    Análise rápida de um par usando a equipe de agentes
    
    Args:
        pair: Par de moedas
        period: Período de análise
        
    Returns:
        Análise completa do par
    """
    team = ForexAgentTeam()
    return team.analyze_pair(pair, period)