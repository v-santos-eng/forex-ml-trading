from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools
from agno_ml_tool import MLForexTool
from config import ForexConfig
from typing import Dict, Any, List
import os

class OllamaForexAgentTeam:
    """
    Equipe de agentes especializados para trading de Forex usando Ollama (modelos locais gratuitos)
    """
    
    def __init__(self, model_id: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model_id = model_id
        self.host = host
        self.ml_tool = MLForexTool()
        self._create_agents()
    
    def _create_agents(self):
        """
        Criar agentes especializados usando Ollama
        """
        
        # Agente de Análise Técnica (ML)
        self.technical_agent = Agent(
            name="Technical Analyst",
            role="Especialista em análise técnica usando Machine Learning",
            model=Ollama(id=self.model_id, host=self.host),
            tools=[self.ml_tool],
            instructions=[
                "Use machine learning para analisar padrões técnicos",
                "Forneça predições com níveis de confiança",
                "Compare múltiplos modelos de ML",
                "Explique o raciocínio por trás das predições",
                "Sempre inclua métricas de performance",
                "Seja conciso e objetivo nas respostas"
            ],
            show_tool_calls=True,
            markdown=True
        )
        
        # Agente de Análise Fundamental
        self.fundamental_agent = Agent(
            name="Fundamental Analyst",
            role="Especialista em análise fundamental e notícias econômicas",
            model=Ollama(id=self.model_id, host=self.host),
            tools=[
                DuckDuckGoTools(),
                YFinanceTools(company_news=True, company_info=True)
            ],
            instructions=[
                "Analise notícias econômicas relevantes",
                "Considere indicadores macroeconômicos",
                "Avalie eventos geopolíticos",
                "Forneça contexto fundamental para decisões",
                "Sempre cite fontes das informações",
                "Seja objetivo e factual"
            ],
            show_tool_calls=True,
            markdown=True
        )
        
        # Agente de Gestão de Risco
        self.risk_agent = Agent(
            name="Risk Manager",
            role="Especialista em gestão de risco e controle de posições",
            model=Ollama(id=self.model_id, host=self.host),
            tools=[ReasoningTools(add_instructions=True)],
            instructions=[
                "Avalie riscos de cada operação",
                "Calcule tamanhos de posição apropriados",
                "Defina stop-loss e take-profit",
                "Considere correlações entre pares",
                "Mantenha exposição total controlada",
                "Use raciocínio estruturado para decisões",
                "Priorize preservação de capital"
            ],
            show_tool_calls=True,
            markdown=True
        )
        
        # Agente Coordenador
        self.coordinator_agent = Agent(
            name="Trading Coordinator",
            role="Coordenador que integra análises e toma decisões finais",
            model=Ollama(id=self.model_id, host=self.host),
            tools=[ReasoningTools(add_instructions=True)],
            instructions=[
                "Integre análises técnica e fundamental",
                "Considere recomendações de gestão de risco",
                "Tome decisões finais de trading",
                "Forneça justificativas claras",
                "Priorize preservação de capital",
                "Use raciocínio passo-a-passo",
                "Seja decisivo mas cauteloso"
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
        
        Forneça uma recomendação final clara: BUY, SELL ou HOLD com justificativa."""
        
        coordinator_analysis = self.coordinator_agent.run(coordinator_query)
        results["decision"] = coordinator_analysis.content
        
        return results
    
    def quick_analysis(self, pair: str) -> str:
        """
        Análise rápida usando apenas o agente técnico
        
        Args:
            pair: Par de moedas
            
        Returns:
            Análise técnica rápida
        """
        query = f"Faça uma análise técnica rápida de {pair} usando machine learning. Seja conciso."
        response = self.technical_agent.run(query)
        return response.content
    
    def risk_assessment(self, pair: str, position_size: float = 10000) -> str:
        """
        Avaliação de risco para uma posição específica
        
        Args:
            pair: Par de moedas
            position_size: Tamanho da posição
            
        Returns:
            Avaliação de risco
        """
        query = f"Avalie o risco de uma posição de {position_size} em {pair}. Calcule stop-loss, take-profit e tamanho ideal da posição."
        response = self.risk_agent.run(query)
        return response.content
    
    def get_market_sentiment(self, pair: str) -> str:
        """
        Análise de sentimento do mercado
        
        Args:
            pair: Par de moedas
            
        Returns:
            Análise de sentimento
        """
        base_currency = pair.split("USD")[0] if "USD" in pair else pair[:3]
        query = f"Analise o sentimento atual do mercado para {base_currency} baseado em notícias recentes e indicadores econômicos."
        response = self.fundamental_agent.run(query)
        return response.content
    
    def check_ollama_status(self) -> Dict[str, Any]:
        """
        Verifica se o Ollama está funcionando
        
        Returns:
            Status do Ollama
        """
        try:
            import requests
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return {
                    "status": "online",
                    "host": self.host,
                    "models_available": len(models),
                    "current_model": self.model_id,
                    "models": [model['name'] for model in models]
                }
            else:
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "offline",
                "error": str(e),
                "suggestion": "Execute 'ollama serve' para iniciar o Ollama"
            }

# Função de conveniência
def create_ollama_team(model: str = "llama3.2") -> OllamaForexAgentTeam:
    """
    Cria uma equipe de agentes Ollama
    
    Args:
        model: Modelo Ollama a usar
        
    Returns:
        Equipe de agentes configurada
    """
    return OllamaForexAgentTeam(model_id=model)

if __name__ == "__main__":
    # Teste básico
    print("Testando integração com Ollama...")
    
    team = create_ollama_team()
    status = team.check_ollama_status()
    
    print(f"Status do Ollama: {status}")
    
    if status["status"] == "online":
        print("Ollama está funcionando!")
        print(f"Modelos disponíveis: {status['models_available']}")
    else:
        print("Ollama não está disponível")
        print("Execute: ollama serve")
        print("Instale um modelo: ollama pull llama3.2")