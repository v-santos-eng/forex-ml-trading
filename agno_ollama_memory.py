from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.reasoning import ReasoningTools
from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime

class OllamaForexMemorySystem:
    """
    Sistema de memória e raciocínio para Forex usando Ollama (modelos locais gratuitos)
    """
    
    def __init__(self, model_id: str = "llama3.2", host: str = "http://localhost:11434", db_path: str = "forex_memory.db"):
        self.model_id = model_id
        self.host = host
        self.db_path = db_path
        
        # Inicializar storage
        self.storage = SqliteAgentStorage(
            table_name="forex_agent_sessions",
            db_file=db_path
        )
        
        self._create_agents()
        
    def _create_agents(self):
        """
        Criar agentes de memória e aprendizado
        """
        
        # Agente de Análise de Memória
        self.memory_agent = Agent(
            name="Memory Analyst",
            role="Especialista em análise de histórico de trading e padrões",
            model=Ollama(id=self.model_id, host=self.host),
            tools=[ReasoningTools(add_instructions=True)],
            instructions=[
                "Analise histórico de decisões de trading",
                "Identifique padrões de sucesso e falha",
                "Extraia insights de performance passada",
                "Forneça recomendações baseadas em experiência",
                "Use raciocínio estruturado para análises",
                "Seja objetivo e baseado em dados"
            ],
            storage=self.storage,
            show_tool_calls=True,
            markdown=True
        )
        
        # Agente de Aprendizado
        self.learning_agent = Agent(
            name="Learning Agent",
            role="Especialista em aprendizado contínuo e otimização de estratégias",
            model=Ollama(id=self.model_id, host=self.host),
            tools=[ReasoningTools(add_instructions=True)],
            instructions=[
                "Aprenda com resultados de trading anteriores",
                "Otimize estratégias baseado em performance",
                "Identifique melhorias nos processos",
                "Sugira ajustes em parâmetros de trading",
                "Use raciocínio para evolução contínua",
                "Foque em melhoria incremental"
            ],
            storage=self.storage,
            show_tool_calls=True,
            markdown=True
        )
    
    def store_trading_decision(self, pair: str, decision: Dict[str, Any]) -> str:
        """
        Armazena uma decisão de trading na memória
        
        Args:
            pair: Par de moedas
            decision: Dicionário com a decisão completa
            
        Returns:
            ID da sessão armazenada
        """
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "pair": pair,
            "decision": decision,
            "type": "trading_decision"
        }
        
        # Criar uma sessão para armazenar
        session_id = f"trading_{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Usar o agente de memória para processar e armazenar
        memory_query = f"""Processe e armazene esta decisão de trading:
        
        Par: {pair}
        Timestamp: {session_data['timestamp']}
        Decisão: {json.dumps(decision, indent=2)}
        
        Extraia insights importantes e padrões desta decisão."""
        
        self.memory_agent.run(memory_query, session_id=session_id)
        
        return session_id
    
    def analyze_trading_history(self, pair: str = None, days: int = 30) -> str:
        """
        Analisa histórico de trading
        
        Args:
            pair: Par específico (opcional)
            days: Número de dias para analisar
            
        Returns:
            Análise do histórico
        """
        # Recuperar sessões relevantes
        sessions = self.storage.get_all_sessions()
        
        if pair:
            # Filtrar por par específico
            relevant_sessions = [s for s in sessions if pair.lower() in str(s.session_id).lower()]
            query = f"Analise o histórico de trading para {pair} dos últimos {days} dias. Identifique padrões, sucessos e falhas."
        else:
            relevant_sessions = sessions
            query = f"Analise o histórico geral de trading dos últimos {days} dias. Identifique padrões globais e tendências."
        
        if not relevant_sessions:
            return "Nenhum histórico de trading encontrado."
        
        # Usar agente de memória para análise
        analysis_query = f"""{query}
        
        Sessões disponíveis: {len(relevant_sessions)}
        
        Forneça insights sobre:
        1. Padrões de sucesso
        2. Erros comuns
        3. Tendências de performance
        4. Recomendações de melhoria"""
        
        response = self.memory_agent.run(analysis_query)
        return response.content
    
    def learn_from_outcomes(self, pair: str, actual_outcome: Dict[str, Any]) -> str:
        """
        Aprende com resultados reais de trading
        
        Args:
            pair: Par de moedas
            actual_outcome: Resultado real da operação
            
        Returns:
            Insights de aprendizado
        """
        learning_query = f"""Analise este resultado real de trading e extraia aprendizados:
        
        Par: {pair}
        Resultado: {json.dumps(actual_outcome, indent=2)}
        
        Compare com decisões anteriores e identifique:
        1. O que funcionou bem
        2. O que pode ser melhorado
        3. Ajustes recomendados para futuras decisões
        4. Padrões emergentes"""
        
        response = self.learning_agent.run(learning_query)
        return response.content
    
    def get_recommendations(self, pair: str, current_analysis: Dict[str, Any]) -> str:
        """
        Obtém recomendações baseadas em memória e aprendizado
        
        Args:
            pair: Par de moedas
            current_analysis: Análise atual
            
        Returns:
            Recomendações baseadas em experiência
        """
        recommendation_query = f"""Baseado no histórico de trading e aprendizados anteriores, 
        forneça recomendações para esta situação atual:
        
        Par: {pair}
        Análise Atual: {json.dumps(current_analysis, indent=2)}
        
        Considere:
        1. Experiências similares passadas
        2. Padrões de sucesso identificados
        3. Riscos baseados em histórico
        4. Ajustes recomendados
        
        Forneça recomendações específicas e acionáveis."""
        
        response = self.memory_agent.run(recommendation_query)
        return response.content
    
    def optimize_strategy(self, strategy_name: str, performance_data: List[Dict]) -> str:
        """
        Otimiza estratégia baseada em dados de performance
        
        Args:
            strategy_name: Nome da estratégia
            performance_data: Dados de performance
            
        Returns:
            Estratégia otimizada
        """
        optimization_query = f"""Otimize a estratégia '{strategy_name}' baseado nestes dados de performance:
        
        Dados: {json.dumps(performance_data, indent=2)}
        
        Analise:
        1. Métricas de performance
        2. Pontos fracos da estratégia
        3. Oportunidades de melhoria
        4. Parâmetros a ajustar
        
        Forneça uma versão otimizada da estratégia."""
        
        response = self.learning_agent.run(optimization_query)
        return response.content
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Obtém estatísticas da memória
        
        Returns:
            Estatísticas do sistema de memória
        """
        try:
            sessions = self.storage.get_all_sessions()
            
            return {
                "total_sessions": len(sessions),
                "db_path": self.db_path,
                "model": self.model_id,
                "host": self.host,
                "recent_sessions": sessions[-5:] if sessions else []
            }
        except Exception as e:
            return {
                "error": str(e),
                "total_sessions": 0
            }
    
    def clear_old_memories(self, days: int = 90) -> str:
        """
        Remove memórias antigas
        
        Args:
            days: Dias para manter
            
        Returns:
            Resultado da limpeza
        """
        try:
            # Esta é uma implementação simplificada
            # Em uma versão completa, filtraria por data
            sessions_before = len(self.storage.get_all_sessions())
            
            # Por enquanto, apenas retorna estatísticas
            return f"Sistema de memória tem {sessions_before} sessões. Limpeza automática não implementada ainda."
        except Exception as e:
            return f"Erro na limpeza: {str(e)}"
    
    def check_ollama_status(self) -> Dict[str, Any]:
        """
        Verifica status do Ollama
        
        Returns:
            Status do Ollama
        """
        try:
            import requests
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                return {
                    "status": "online",
                    "host": self.host,
                    "model": self.model_id
                }
            else:
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "offline",
                "error": str(e)
            }

# Função de conveniência
def create_ollama_memory_system(model: str = "llama3.2") -> OllamaForexMemorySystem:
    """
    Cria sistema de memória Ollama
    
    Args:
        model: Modelo Ollama a usar
        
    Returns:
        Sistema de memória configurado
    """
    return OllamaForexMemorySystem(model_id=model)

if __name__ == "__main__":
    # Teste básico
    print("Testando sistema de memória Ollama...")
    
    memory_system = create_ollama_memory_system()
    status = memory_system.check_ollama_status()
    stats = memory_system.get_memory_stats()
    
    print(f"Status Ollama: {status}")
    print(f"Estatísticas de Memória: {stats}")
    
    if status["status"] == "online":
        print("Sistema de memória Ollama está funcionando!")
    else:
        print("Ollama não está disponível para o sistema de memória")