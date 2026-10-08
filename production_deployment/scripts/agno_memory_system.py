from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.memory import AgentMemory
from agno.tools.reasoning import ReasoningTools
from agno_ml_tool import MLForexTool
from agno_forex_agents import ForexAgentTeam
from config import ForexConfig
from typing import Dict, Any, List, Optional
import json
import sqlite3
from datetime import datetime, timedelta
import os

class ForexMemorySystem:
    """
    Sistema de memória e raciocínio para agentes de Forex
    """
    
    def __init__(self, db_path: str = "forex_memory.db", model_id: str = "gpt-4o-mini"):
        self.db_path = db_path
        self.model_id = model_id
        
        # Configurar API OpenAI
        if ForexConfig.OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = ForexConfig.OPENAI_API_KEY
        
        self._init_database()
        self._create_memory_agents()
    
    def _init_database(self):
        """
        Inicializar banco de dados SQLite para memória
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de predições históricas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                pair TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL,
                model_used TEXT,
                actual_outcome TEXT,
                accuracy REAL,
                notes TEXT
            )
        ''')
        
        # Tabela de performance dos modelos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                pair TEXT NOT NULL,
                accuracy REAL,
                cv_score REAL,
                backtest_return REAL,
                sharpe_ratio REAL
            )
        ''')
        
        # Tabela de padrões de mercado
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                pair TEXT NOT NULL,
                pattern_type TEXT,
                description TEXT,
                success_rate REAL,
                conditions TEXT
            )
        ''')
        
        # Tabela de decisões de trading
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                pair TEXT NOT NULL,
                decision TEXT NOT NULL,
                reasoning TEXT,
                technical_score REAL,
                fundamental_score REAL,
                risk_score REAL,
                outcome TEXT,
                profit_loss REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _create_memory_agents(self):
        """
        Criar agentes com capacidades de memória e raciocínio
        """
        
        # Storage para agentes
        storage = SqliteAgentStorage(
            table_name="agent_sessions",
            db_file=self.db_path
        )
        
        # Agente de Análise com Memória
        self.memory_analyst = Agent(
            name="Memory Analyst",
            role="Analista com memória de padrões históricos",
            model=OpenAIChat(id=self.model_id),
            tools=[
                MLForexTool(),
                ReasoningTools(add_instructions=True)
            ],
            storage=storage,
            instructions=[
                "Use memória de análises anteriores para melhorar predições",
                "Identifique padrões recorrentes no mercado",
                "Aprenda com sucessos e falhas passadas",
                "Mantenha contexto de sessões anteriores",
                "Use raciocínio estruturado para decisões"
            ],
            show_tool_calls=True,
            markdown=True
        )
        
        # Agente de Aprendizado Contínuo
        self.learning_agent = Agent(
            name="Learning Agent",
            role="Especialista em aprendizado contínuo e otimização",
            model=OpenAIChat(id=self.model_id),
            tools=[ReasoningTools(add_instructions=True)],
            storage=storage,
            instructions=[
                "Analise performance histórica dos modelos",
                "Identifique quando retraining é necessário",
                "Otimize estratégias baseado em resultados",
                "Mantenha registro de lições aprendidas",
                "Sugira melhorias nos processos"
            ],
            show_tool_calls=True,
            markdown=True
        )
    
    def store_prediction(self, pair: str, prediction: str, confidence: float, 
                        model_used: str, notes: str = "") -> int:
        """
        Armazenar predição na memória
        
        Args:
            pair: Par de moedas
            prediction: Predição (ALTA/BAIXA)
            confidence: Nível de confiança
            model_used: Modelo utilizado
            notes: Notas adicionais
            
        Returns:
            ID da predição armazenada
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (pair, prediction, confidence, model_used, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (pair, prediction, confidence, model_used, notes))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def update_prediction_outcome(self, prediction_id: int, actual_outcome: str, accuracy: float):
        """
        Atualizar resultado real de uma predição
        
        Args:
            prediction_id: ID da predição
            actual_outcome: Resultado real
            accuracy: Acurácia da predição
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE predictions 
            SET actual_outcome = ?, accuracy = ?
            WHERE id = ?
        ''', (actual_outcome, accuracy, prediction_id))
        
        conn.commit()
        conn.close()
    
    def store_model_performance(self, model_name: str, pair: str, accuracy: float,
                               cv_score: float, backtest_return: float = None,
                               sharpe_ratio: float = None):
        """
        Armazenar performance de modelo
        
        Args:
            model_name: Nome do modelo
            pair: Par de moedas
            accuracy: Acurácia
            cv_score: Score de validação cruzada
            backtest_return: Retorno do backtest
            sharpe_ratio: Sharpe ratio
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_performance 
            (model_name, pair, accuracy, cv_score, backtest_return, sharpe_ratio)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_name, pair, accuracy, cv_score, backtest_return, sharpe_ratio))
        
        conn.commit()
        conn.close()
    
    def get_historical_performance(self, pair: str = None, days: int = 30) -> List[Dict]:
        """
        Obter performance histórica
        
        Args:
            pair: Par específico (opcional)
            days: Número de dias para buscar
            
        Returns:
            Lista de registros de performance
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        date_limit = datetime.now() - timedelta(days=days)
        
        if pair:
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE pair = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', (pair, date_limit))
        else:
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', (date_limit,))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def analyze_with_memory(self, pair: str, period: str = "6mo") -> Dict[str, Any]:
        """
        Análise usando memória histórica
        
        Args:
            pair: Par de moedas
            period: Período de análise
            
        Returns:
            Análise com contexto histórico
        """
        # Obter histórico
        historical_data = self.get_historical_performance(pair, days=90)
        
        # Preparar contexto histórico
        context = f"""Histórico de predições para {pair} (últimos 90 dias):
        Total de predições: {len(historical_data)}
        """
        
        if historical_data:
            successful_predictions = [p for p in historical_data if p['accuracy'] and p['accuracy'] > 0.5]
            context += f"Predições bem-sucedidas: {len(successful_predictions)}/{len(historical_data)}\n"
            
            # Padrões identificados
            high_predictions = [p for p in historical_data if p['prediction'] == 'ALTA']
            low_predictions = [p for p in historical_data if p['prediction'] == 'BAIXA']
            
            context += f"Predições ALTA: {len(high_predictions)}\n"
            context += f"Predições BAIXA: {len(low_predictions)}\n"
        
        # Análise com contexto
        query = f"""Analise {pair} considerando o seguinte contexto histórico:
        
        {context}
        
        Faça uma nova análise técnica usando machine learning e identifique:
        1. Padrões que se repetem
        2. Lições aprendidas de predições anteriores
        3. Ajustes necessários na estratégia
        4. Nova predição com justificativa baseada na memória
        
        Use raciocínio estruturado para sua análise."""
        
        analysis = self.memory_analyst.run(query)
        
        return {
            "pair": pair,
            "period": period,
            "historical_context": context,
            "memory_analysis": analysis.content,
            "historical_predictions": len(historical_data)
        }
    
    def continuous_learning_session(self) -> Dict[str, Any]:
        """
        Sessão de aprendizado contínuo
        
        Returns:
            Insights e recomendações de melhoria
        """
        # Obter dados de performance geral
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Performance por modelo
        cursor.execute('''
            SELECT model_name, AVG(accuracy) as avg_accuracy, COUNT(*) as count
            FROM model_performance
            WHERE timestamp >= datetime('now', '-30 days')
            GROUP BY model_name
            ORDER BY avg_accuracy DESC
        ''')
        
        model_performance = cursor.fetchall()
        
        # Performance por par
        cursor.execute('''
            SELECT pair, AVG(accuracy) as avg_accuracy, COUNT(*) as count
            FROM predictions
            WHERE timestamp >= datetime('now', '-30 days') AND accuracy IS NOT NULL
            GROUP BY pair
            ORDER BY avg_accuracy DESC
        ''')
        
        pair_performance = cursor.fetchall()
        
        conn.close()
        
        # Preparar contexto para aprendizado
        learning_context = f"""Dados de Performance (últimos 30 dias):
        
        PERFORMANCE POR MODELO:
        {chr(10).join([f'{model}: {acc:.3f} ({count} predições)' for model, acc, count in model_performance])}
        
        PERFORMANCE POR PAR:
        {chr(10).join([f'{pair}: {acc:.3f} ({count} predições)' for pair, acc, count in pair_performance])}
        """
        
        query = f"""Analise os dados de performance e forneça insights para melhoria contínua:
        
        {learning_context}
        
        Identifique:
        1. Modelos com melhor performance
        2. Pares mais difíceis de prever
        3. Padrões de sucesso e falha
        4. Recomendações para otimização
        5. Necessidade de retraining
        
        Use raciocínio estruturado para sua análise."""
        
        learning_analysis = self.learning_agent.run(query)
        
        return {
            "learning_period": "30 days",
            "model_performance": model_performance,
            "pair_performance": pair_performance,
            "learning_insights": learning_analysis.content
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Estatísticas do sistema de memória
        
        Returns:
            Estatísticas da base de dados
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Contar registros
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM model_performance")
        total_model_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM trading_decisions")
        total_decisions = cursor.fetchone()[0]
        
        # Últimas atividades
        cursor.execute("SELECT MAX(timestamp) FROM predictions")
        last_prediction = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "database_path": self.db_path,
            "total_predictions": total_predictions,
            "total_model_records": total_model_records,
            "total_decisions": total_decisions,
            "last_prediction": last_prediction,
            "memory_agents": {
                "memory_analyst": self.memory_analyst.name,
                "learning_agent": self.learning_agent.name
            }
        }

# Função de conveniência
def create_memory_enhanced_analysis(pair: str = "EURUSD=X") -> Dict[str, Any]:
    """
    Análise completa com sistema de memória
    
    Args:
        pair: Par de moedas
        
    Returns:
        Análise com memória e aprendizado
    """
    memory_system = ForexMemorySystem()
    return memory_system.analyze_with_memory(pair)