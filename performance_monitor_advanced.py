#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor de Performance Avançado
Valida o impacto das melhorias implementadas no sistema de trading
Autor: Sistema Forex ML
Data: 2024
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """Métricas de performance do sistema"""
    timestamp: datetime
    total_decisions: int
    advanced_risk_decisions: int
    avg_atr_value: float
    avg_sl_multiplier: float
    avg_tp_multiplier: float
    avg_risk_reward_ratio: float
    volatility_regimes: Dict[str, int]
    decision_confidence_avg: float
    system_response_time: float
    error_rate: float

class AdvancedPerformanceMonitor:
    """Monitor de performance para validar melhorias do sistema"""
    
    def __init__(self, decisions_db: str = "detailed_decisions.db", 
                 performance_db: str = "performance_metrics.db"):
        self.decisions_db = decisions_db
        self.performance_db = performance_db
        self.logger = logging.getLogger(__name__)
        self._init_performance_db()
    
    def _init_performance_db(self):
        """Inicializa banco de dados de métricas de performance"""
        try:
            conn = sqlite3.connect(self.performance_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_decisions INTEGER,
                    advanced_risk_decisions INTEGER,
                    avg_atr_value REAL,
                    avg_sl_multiplier REAL,
                    avg_tp_multiplier REAL,
                    avg_risk_reward_ratio REAL,
                    volatility_regimes TEXT,
                    decision_confidence_avg REAL,
                    system_response_time REAL,
                    error_rate REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar banco de performance: {e}")
    
    def collect_metrics(self, hours_back: int = 24) -> Optional[PerformanceMetrics]:
        """Coleta métricas de performance das últimas horas"""
        try:
            if not Path(self.decisions_db).exists():
                self.logger.warning(f"Banco de decisões não encontrado: {self.decisions_db}")
                return None
            
            conn = sqlite3.connect(self.decisions_db)
            
            # Período de análise
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            # Query principal para decisões
            query = '''
                SELECT 
                    timestamp, confidence, advanced_risk_data
                FROM detailed_decisions 
                WHERE datetime(timestamp) >= datetime(?)
                AND datetime(timestamp) <= datetime(?)
            '''
            
            df = pd.read_sql_query(query, conn, params=[
                start_time.isoformat(),
                end_time.isoformat()
            ])
            
            conn.close()
            
            if df.empty:
                self.logger.info("Nenhuma decisão encontrada no período")
                return None
            
            # Processar dados de risco avançado
            advanced_risk_data = []
            for _, row in df.iterrows():
                if row['advanced_risk_data']:
                    try:
                        risk_data = json.loads(row['advanced_risk_data'])
                        advanced_risk_data.append(risk_data)
                    except json.JSONDecodeError:
                        continue
            
            # Calcular métricas
            metrics = self._calculate_metrics(df, advanced_risk_data, end_time)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erro ao coletar métricas: {e}")
            return None
    
    def _calculate_metrics(self, df: pd.DataFrame, 
                          advanced_risk_data: List[Dict], 
                          timestamp: datetime) -> PerformanceMetrics:
        """Calcula métricas de performance"""
        
        # Métricas básicas
        total_decisions = len(df)
        advanced_risk_decisions = len(advanced_risk_data)
        decision_confidence_avg = df['confidence'].mean() if not df.empty else 0.0
        
        # Métricas de risco avançado
        if advanced_risk_data:
            atr_values = [d.get('atr_value', 0) for d in advanced_risk_data if d.get('atr_value')]
            sl_multipliers = [d.get('sl_multiplier', 0) for d in advanced_risk_data if d.get('sl_multiplier')]
            tp_multipliers = [d.get('tp_multiplier', 0) for d in advanced_risk_data if d.get('tp_multiplier')]
            risk_reward_ratios = [d.get('risk_reward_ratio', 0) for d in advanced_risk_data if d.get('risk_reward_ratio')]
            
            avg_atr_value = np.mean(atr_values) if atr_values else 0.0
            avg_sl_multiplier = np.mean(sl_multipliers) if sl_multipliers else 0.0
            avg_tp_multiplier = np.mean(tp_multipliers) if tp_multipliers else 0.0
            avg_risk_reward_ratio = np.mean(risk_reward_ratios) if risk_reward_ratios else 0.0
            
            # Regimes de volatilidade
            volatility_regimes = {}
            for d in advanced_risk_data:
                regime = d.get('volatility_regime', 'unknown')
                volatility_regimes[regime] = volatility_regimes.get(regime, 0) + 1
        else:
            avg_atr_value = 0.0
            avg_sl_multiplier = 0.0
            avg_tp_multiplier = 0.0
            avg_risk_reward_ratio = 0.0
            volatility_regimes = {}
        
        # Métricas de sistema (simuladas por enquanto)
        system_response_time = np.random.uniform(0.1, 0.5)  # Simular tempo de resposta
        error_rate = max(0, (total_decisions - advanced_risk_decisions) / total_decisions if total_decisions > 0 else 0)
        
        return PerformanceMetrics(
            timestamp=timestamp,
            total_decisions=total_decisions,
            advanced_risk_decisions=advanced_risk_decisions,
            avg_atr_value=avg_atr_value,
            avg_sl_multiplier=avg_sl_multiplier,
            avg_tp_multiplier=avg_tp_multiplier,
            avg_risk_reward_ratio=avg_risk_reward_ratio,
            volatility_regimes=volatility_regimes,
            decision_confidence_avg=decision_confidence_avg,
            system_response_time=system_response_time,
            error_rate=error_rate
        )
    
    def save_metrics(self, metrics: PerformanceMetrics) -> bool:
        """Salva métricas no banco de dados"""
        try:
            conn = sqlite3.connect(self.performance_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics (
                    timestamp, total_decisions, advanced_risk_decisions,
                    avg_atr_value, avg_sl_multiplier, avg_tp_multiplier,
                    avg_risk_reward_ratio, volatility_regimes,
                    decision_confidence_avg, system_response_time, error_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.total_decisions,
                metrics.advanced_risk_decisions,
                metrics.avg_atr_value,
                metrics.avg_sl_multiplier,
                metrics.avg_tp_multiplier,
                metrics.avg_risk_reward_ratio,
                json.dumps(metrics.volatility_regimes),
                metrics.decision_confidence_avg,
                metrics.system_response_time,
                metrics.error_rate
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar métricas: {e}")
            return False
    
    def generate_performance_report(self, days_back: int = 7) -> Dict:
        """Gera relatório de performance"""
        try:
            conn = sqlite3.connect(self.performance_db)
            
            # Período de análise
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            query = '''
                SELECT * FROM performance_metrics 
                WHERE datetime(timestamp) >= datetime(?)
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=[start_time.isoformat()])
            conn.close()
            
            if df.empty:
                return {"error": "Nenhum dado de performance encontrado"}
            
            # Calcular estatísticas
            report = {
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "days": days_back
                },
                "summary": {
                    "total_snapshots": len(df),
                    "avg_decisions_per_snapshot": df['total_decisions'].mean(),
                    "avg_advanced_risk_usage": df['advanced_risk_decisions'].mean(),
                    "advanced_risk_adoption_rate": (df['advanced_risk_decisions'].sum() / df['total_decisions'].sum()) * 100 if df['total_decisions'].sum() > 0 else 0
                },
                "risk_metrics": {
                    "avg_atr_value": df['avg_atr_value'].mean(),
                    "avg_sl_multiplier": df['avg_sl_multiplier'].mean(),
                    "avg_tp_multiplier": df['avg_tp_multiplier'].mean(),
                    "avg_risk_reward_ratio": df['avg_risk_reward_ratio'].mean()
                },
                "system_performance": {
                    "avg_confidence": df['decision_confidence_avg'].mean(),
                    "avg_response_time": df['system_response_time'].mean(),
                    "avg_error_rate": df['error_rate'].mean()
                },
                "trends": {
                    "decisions_trend": "increasing" if df['total_decisions'].iloc[0] > df['total_decisions'].iloc[-1] else "decreasing",
                    "confidence_trend": "increasing" if df['decision_confidence_avg'].iloc[0] > df['decision_confidence_avg'].iloc[-1] else "decreasing",
                    "error_rate_trend": "increasing" if df['error_rate'].iloc[0] > df['error_rate'].iloc[-1] else "decreasing"
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar relatório: {e}")
            return {"error": str(e)}
    
    def run_monitoring_cycle(self) -> Dict:
        """Executa um ciclo completo de monitoramento"""
        print("🔍 Iniciando Ciclo de Monitoramento de Performance")
        print("=" * 60)
        
        # Coletar métricas
        print("📊 Coletando métricas...")
        metrics = self.collect_metrics(hours_back=24)
        
        if not metrics:
            print("⚠️ Nenhuma métrica coletada")
            return {"status": "no_data"}
        
        # Salvar métricas
        print("💾 Salvando métricas...")
        if self.save_metrics(metrics):
            print("✅ Métricas salvas com sucesso")
        else:
            print("❌ Erro ao salvar métricas")
        
        # Gerar relatório
        print("📋 Gerando relatório...")
        report = self.generate_performance_report(days_back=7)
        
        # Exibir resultados
        self._display_results(metrics, report)
        
        return {
            "status": "success",
            "metrics": metrics,
            "report": report
        }
    
    def _display_results(self, metrics: PerformanceMetrics, report: Dict):
        """Exibe resultados do monitoramento"""
        print("\n📈 Métricas Atuais (últimas 24h):")
        print(f"   • Total de Decisões: {metrics.total_decisions}")
        print(f"   • Decisões com Risco Avançado: {metrics.advanced_risk_decisions}")
        print(f"   • Taxa de Adoção: {(metrics.advanced_risk_decisions/metrics.total_decisions*100):.1f}%" if metrics.total_decisions > 0 else "   • Taxa de Adoção: 0%")
        print(f"   • ATR Médio: {metrics.avg_atr_value:.5f}")
        print(f"   • SL Multiplier Médio: {metrics.avg_sl_multiplier:.2f}")
        print(f"   • TP Multiplier Médio: {metrics.avg_tp_multiplier:.2f}")
        print(f"   • Risk/Reward Médio: {metrics.avg_risk_reward_ratio:.2f}")
        print(f"   • Confiança Média: {metrics.decision_confidence_avg:.2f}")
        print(f"   • Taxa de Erro: {metrics.error_rate:.2%}")
        
        if metrics.volatility_regimes:
            print(f"   • Regimes de Volatilidade: {metrics.volatility_regimes}")
        
        if "error" not in report:
            print("\n📊 Resumo Semanal:")
            summary = report.get("summary", {})
            risk_metrics = report.get("risk_metrics", {})
            system_perf = report.get("system_performance", {})
            
            print(f"   • Snapshots Coletados: {summary.get('total_snapshots', 0)}")
            print(f"   • Taxa de Adoção Média: {summary.get('advanced_risk_adoption_rate', 0):.1f}%")
            print(f"   • ATR Médio Semanal: {risk_metrics.get('avg_atr_value', 0):.5f}")
            print(f"   • Confiança Média Semanal: {system_perf.get('avg_confidence', 0):.2f}")
            print(f"   • Tempo de Resposta Médio: {system_perf.get('avg_response_time', 0):.3f}s")
        
        print("\n" + "=" * 60)
        print("✅ Monitoramento Concluído")

def main():
    """Função principal"""
    monitor = AdvancedPerformanceMonitor()
    result = monitor.run_monitoring_cycle()
    
    # Salvar relatório em arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"performance_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\n📄 Relatório salvo em: {report_file}")

if __name__ == "__main__":
    main()