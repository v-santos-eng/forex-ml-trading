#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Logging Detalhado para Decisoes de Trading
Captura todas as informacoes do processo de tomada de decisao do bot
Autor: Sistema Forex ML
Data: 2024
"""

import json
import sqlite3
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid
import os

@dataclass
class DecisionData:
    """Dados completos de uma decisao de trading"""
    symbol: str
    action: str  # 'GENERATE_SIGNAL', 'EXECUTE_TRADE', 'NO_ACTION'
    reasoning: 'DecisionReasoning'
    market_data: Optional[Dict[str, Any]] = None
    technical_analysis: Optional[Dict[str, Any]] = None
    ml_predictions: Optional[Dict[str, Any]] = None
    bot_action: Optional['BotAction'] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class DecisionReasoning:
    """Raciocinio por tras de uma decisao"""
    factors: List[str]
    confidence_score: float
    risk_assessment: str

@dataclass
class BotAction:
    """Acao realizada pelo bot"""
    action_type: str
    parameters: Dict[str, Any]
    result: str
    details: Optional[str] = None
    errors: Optional[List[str]] = None

@dataclass
class AdvancedRiskData:
    """Dados específicos do sistema avançado de risco"""
    atr_value: float
    atr_method: str
    volatility_regime: str
    sl_multiplier: float
    tp_multiplier: float
    risk_reward_ratio: float
    gap_protection_active: bool
    technical_levels_considered: bool
    trailing_stop_enabled: bool
    calculation_metadata: Optional[Dict[str, Any]] = None

class DetailedDecisionLogger:
    """Sistema de logging detalhado para decisoes de trading"""
    
    def __init__(self, db_path: str = "detailed_decisions.db"):
        self.db_path = db_path
        self._init_database()
        print(f"DetailedDecisionLogger inicializado com banco: {db_path}")
    
    def _init_database(self):
        """Inicializa o banco de dados"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabela principal de decisoes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detailed_decisions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning_factors TEXT,
                    risk_assessment TEXT,
                    market_data TEXT,
                    technical_analysis TEXT,
                    ml_predictions TEXT,
                    bot_action TEXT,
                    advanced_risk_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Adicionar coluna advanced_risk_data se não existir (para compatibilidade)
            try:
                cursor.execute('ALTER TABLE detailed_decisions ADD COLUMN advanced_risk_data TEXT')
            except sqlite3.OperationalError:
                pass  # Coluna já existe
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Erro ao inicializar banco de dados: {e}")
    
    def log_decision(self, decision_data: DecisionData, advanced_risk_data: Optional[AdvancedRiskData] = None) -> str:
        """Registra uma decisao detalhada"""
        try:
            decision_id = str(uuid.uuid4())
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO detailed_decisions (
                    id, timestamp, symbol, action, confidence,
                    reasoning_factors, risk_assessment, market_data,
                    technical_analysis, ml_predictions, bot_action, advanced_risk_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                decision_id,
                decision_data.timestamp.isoformat(),
                decision_data.symbol,
                decision_data.action,
                decision_data.reasoning.confidence_score,
                json.dumps(decision_data.reasoning.factors),
                decision_data.reasoning.risk_assessment,
                json.dumps(decision_data.market_data, default=str) if decision_data.market_data else None,
                json.dumps(decision_data.technical_analysis, default=str) if decision_data.technical_analysis else None,
                json.dumps(decision_data.ml_predictions, default=str) if decision_data.ml_predictions else None,
                json.dumps({
                    'action_type': decision_data.bot_action.action_type,
                    'parameters': decision_data.bot_action.parameters,
                    'result': decision_data.bot_action.result,
                    'details': decision_data.bot_action.details,
                    'errors': decision_data.bot_action.errors
                }, default=str) if decision_data.bot_action else None,
                json.dumps(asdict(advanced_risk_data), default=str) if advanced_risk_data else None
            ))
            
            conn.commit()
            conn.close()
            
            return decision_id
            
        except Exception as e:
            print(f"Erro ao registrar decisao: {e}")
            return ""
    
    def log_advanced_risk_decision(self, symbol: str, action: str, confidence: float, 
                                 atr_data: Dict[str, Any], sl_tp_data: Dict[str, Any],
                                 market_data: Optional[Dict] = None) -> str:
        """Método conveniente para registrar decisões do sistema avançado de risco"""
        try:
            # Criar dados de decisão
            reasoning = DecisionReasoning(
                factors=[
                    f"ATR {atr_data.get('method', 'unknown')}: {atr_data.get('value', 0):.5f}",
                    f"Volatilidade: {atr_data.get('volatility_regime', 'normal')}",
                    f"SL Multiplier: {sl_tp_data.get('sl_multiplier', 0):.2f}",
                    f"TP Multiplier: {sl_tp_data.get('tp_multiplier', 0):.2f}",
                    f"Risk/Reward: {sl_tp_data.get('risk_reward_ratio', 0):.2f}"
                ],
                confidence_score=confidence,
                risk_assessment=f"ATR-based dynamic risk management active"
            )
            
            bot_action = BotAction(
                action_type="ADVANCED_RISK_CALCULATION",
                parameters={
                    "atr_method": atr_data.get('method'),
                    "atr_value": atr_data.get('value'),
                    "sl_price": sl_tp_data.get('sl_price'),
                    "tp_price": sl_tp_data.get('tp_price'),
                    "entry_price": sl_tp_data.get('entry_price')
                },
                result="SUCCESS",
                details=f"Dynamic SL/TP calculated using {atr_data.get('method')} ATR method"
            )
            
            decision_data = DecisionData(
                symbol=symbol,
                action=action,
                reasoning=reasoning,
                market_data=market_data,
                bot_action=bot_action
            )
            
            # Criar dados avançados de risco
            advanced_risk_data = AdvancedRiskData(
                atr_value=atr_data.get('value', 0.0),
                atr_method=atr_data.get('method', 'unknown'),
                volatility_regime=atr_data.get('volatility_regime', 'normal'),
                sl_multiplier=sl_tp_data.get('sl_multiplier', 0.0),
                tp_multiplier=sl_tp_data.get('tp_multiplier', 0.0),
                risk_reward_ratio=sl_tp_data.get('risk_reward_ratio', 0.0),
                gap_protection_active=atr_data.get('gap_protection', False),
                technical_levels_considered=atr_data.get('technical_levels', False),
                trailing_stop_enabled=atr_data.get('trailing_stop', False),
                calculation_metadata=atr_data.get('metadata', {})
            )
            
            return self.log_decision(decision_data, advanced_risk_data)
            
        except Exception as e:
            print(f"Erro ao registrar decisão avançada de risco: {e}")
            return ""
    
    def get_recent_decisions(self, limit: int = 50) -> List[Dict]:
        """Recupera decisoes recentes"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, timestamp, symbol, action, confidence,
                       reasoning_factors, risk_assessment
                FROM detailed_decisions 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            results = []
            for row in cursor.fetchall():
                # Converter timestamp de string para datetime
                timestamp_str = row[1]
                try:
                    if isinstance(timestamp_str, str):
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = timestamp_str
                except (ValueError, AttributeError):
                    timestamp = datetime.now()
                
                results.append({
                    'id': row[0],
                    'timestamp': timestamp,
                    'symbol': row[2],
                    'action': row[3],
                    'confidence': row[4],
                    'reasoning_factors': json.loads(row[5]) if row[5] else [],
                    'risk_assessment': row[6]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"Erro ao recuperar decisoes: {e}")
            return []
    
    def get_decision_by_id(self, decision_id: str) -> Optional[Dict]:
        """Recupera decisao especifica por ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM detailed_decisions WHERE id = ?
            ''', (decision_id,))
            
            row = cursor.fetchone()
            if not row:
                conn.close()
                return None
            
            result = {
                'id': row[0],
                'timestamp': row[1],
                'symbol': row[2],
                'action': row[3],
                'confidence': row[4],
                'reasoning_factors': json.loads(row[5]) if row[5] else [],
                'risk_assessment': row[6],
                'market_data': json.loads(row[7]) if row[7] else None,
                'technical_analysis': json.loads(row[8]) if row[8] else None,
                'ml_predictions': json.loads(row[9]) if row[9] else None,
                'bot_action': json.loads(row[10]) if row[10] else None
            }
            
            conn.close()
            return result
            
        except Exception as e:
            print(f"Erro ao recuperar decisao {decision_id}: {e}")
            return None
    
    def get_performance_analysis(self, days: int = 7) -> Dict[str, Any]:
        """Analisa performance das decisoes"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Data limite
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Total de decisoes
            cursor.execute('''
                SELECT COUNT(*) FROM detailed_decisions 
                WHERE timestamp >= ?
            ''', (since_date,))
            total_decisions = cursor.fetchone()[0]
            
            # Confianca media
            cursor.execute('''
                SELECT AVG(confidence) FROM detailed_decisions 
                WHERE timestamp >= ?
            ''', (since_date,))
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            # Distribuicao por acao
            cursor.execute('''
                SELECT action, COUNT(*) FROM detailed_decisions 
                WHERE timestamp >= ?
                GROUP BY action
            ''', (since_date,))
            action_breakdown = dict(cursor.fetchall())
            
            # Simbolos mais negociados
            cursor.execute('''
                SELECT symbol, COUNT(*) FROM detailed_decisions 
                WHERE timestamp >= ?
                GROUP BY symbol
                ORDER BY COUNT(*) DESC
                LIMIT 5
            ''', (since_date,))
            top_symbols = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'total_decisions': total_decisions,
                'average_confidence': round(avg_confidence, 2),
                'action_breakdown': action_breakdown,
                'top_symbols': top_symbols,
                'analysis_period_days': days
            }
            
        except Exception as e:
            print(f"Erro na analise de performance: {e}")
            return {}
    
    def export_decisions_to_csv(self, filename: str, days: int = 30):
        """Exporta decisoes para CSV"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute('''
                SELECT id, timestamp, symbol, action, confidence,
                       reasoning_factors, risk_assessment
                FROM detailed_decisions 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', (since_date,))
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'ID', 'Timestamp', 'Symbol', 'Action', 'Confidence',
                    'Reasoning_Factors', 'Risk_Assessment'
                ])
                
                for row in cursor.fetchall():
                    writer.writerow(row)
            
            conn.close()
            print(f"Decisoes exportadas para {filename}")
            
        except Exception as e:
            print(f"Erro ao exportar decisoes: {e}")
    
    def generate_decision_summary(self, decision_id: str) -> str:
        """Gera resumo de uma decisao"""
        decision = self.get_decision_by_id(decision_id)
        if not decision:
            return "Decisao nao encontrada"
        
        summary = f"""RESUMO DA DECISAO {decision_id[:8]}...

Simbolo: {decision['symbol']}
Acao: {decision['action']}
Confianca: {decision['confidence']:.2f}
Timestamp: {decision['timestamp']}

Raciocinio:
- Fatores: {', '.join(decision['reasoning_factors'])}
- Avaliacao de Risco: {decision['risk_assessment']}
"""
        
        if decision.get('bot_action'):
            bot_action = decision['bot_action']
            summary += f"\nAcao do Bot:\n- Tipo: {bot_action.get('action_type', 'N/A')}\n- Resultado: {bot_action.get('result', 'N/A')}"
        
        return summary

# Exemplo de uso
if __name__ == "__main__":
    # Criar logger
    logger = DetailedDecisionLogger()
    
    # Exemplo de dados de mercado
    market_data = {
        'symbol': 'EURUSD',
        'bid': 1.0850,
        'ask': 1.0852,
        'spread': 0.0002,
        'volume': 1000,
        'timestamp': datetime.now()
    }
    
    # Exemplo de raciocinio
    reasoning = DecisionReasoning(
        factors=["RSI em zona neutra", "MACD positivo", "Modelo ML confiante"],
        confidence_score=0.75,
        risk_assessment="Risco medio - condicoes favoraveis"
    )
    
    # Exemplo de acao
    bot_action = BotAction(
        action_type="EXECUTE_TRADE",
        parameters={"symbol": "EURUSD", "volume": 0.01, "type": "BUY"},
        result="SUCCESS",
        details="Trade executado com sucesso"
    )
    
    # Criar decisao
    decision_data = DecisionData(
        symbol='EURUSD',
        action='EXECUTE_TRADE',
        reasoning=reasoning,
        market_data=market_data,
        bot_action=bot_action
    )
    
    # Salvar log
    log_id = logger.log_decision(decision_data)
    print(f"Log salvo com ID: {log_id}")
    
    # Gerar resumo
    summary = logger.generate_decision_summary(log_id)
    print(summary)
    
    # Analise de performance
    performance = logger.get_performance_analysis(days=7)
    print(f"\nAnalise de Performance: {performance}")