#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Consolidado do Sistema de Trading
Consolida monitoramento, performance e status geral
Autor: Sistema Forex ML
Data: 2024
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from pathlib import Path
import time

class SystemDashboard:
    """Dashboard consolidado para monitoramento do sistema"""
    
    def __init__(self):
        self.decisions_db = "detailed_decisions.db"
        self.performance_db = "performance_metrics.db"
        self.config_file = "auto_trader_config.json"
    
    def get_system_status(self) -> Dict:
        """Obtém status geral do sistema"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "databases": {},
            "config": {},
            "processes": {}
        }
        
        # Status dos bancos de dados
        status["databases"]["decisions"] = {
            "exists": Path(self.decisions_db).exists(),
            "size_mb": round(Path(self.decisions_db).stat().st_size / 1024 / 1024, 2) if Path(self.decisions_db).exists() else 0
        }
        
        status["databases"]["performance"] = {
            "exists": Path(self.performance_db).exists(),
            "size_mb": round(Path(self.performance_db).stat().st_size / 1024 / 1024, 2) if Path(self.performance_db).exists() else 0
        }
        
        # Status da configuração
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                status["config"] = {
                    "loaded": True,
                    "advanced_risk_enabled": config.get("advanced_risk", {}).get("enabled", False),
                    "symbols_count": len(config.get("symbols", [])),
                    "last_modified": datetime.fromtimestamp(Path(self.config_file).stat().st_mtime).isoformat()
                }
            except Exception as e:
                status["config"] = {"loaded": False, "error": str(e)}
        else:
            status["config"] = {"loaded": False, "error": "Config file not found"}
        
        return status
    
    def get_recent_activity(self, hours: int = 24) -> Dict:
        """Obtém atividade recente do sistema"""
        if not Path(self.decisions_db).exists():
            return {"error": "Database not found"}
        
        try:
            conn = sqlite3.connect(self.decisions_db)
            
            # Período de análise
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Decisões por hora
            query_hourly = '''
                SELECT 
                    strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                    COUNT(*) as decisions,
                    SUM(CASE WHEN advanced_risk_data IS NOT NULL AND advanced_risk_data != '' THEN 1 ELSE 0 END) as advanced_decisions
                FROM detailed_decisions 
                WHERE datetime(timestamp) >= datetime(?)
                GROUP BY hour
                ORDER BY hour DESC
                LIMIT 24
            '''
            
            df_hourly = pd.read_sql_query(query_hourly, conn, params=[start_time.isoformat()])
            
            # Decisões por símbolo
            query_symbols = '''
                SELECT 
                    symbol,
                    COUNT(*) as decisions,
                    SUM(CASE WHEN advanced_risk_data IS NOT NULL AND advanced_risk_data != '' THEN 1 ELSE 0 END) as advanced_decisions,
                    AVG(confidence) as avg_confidence
                FROM detailed_decisions 
                WHERE datetime(timestamp) >= datetime(?)
                GROUP BY symbol
                ORDER BY decisions DESC
            '''
            
            df_symbols = pd.read_sql_query(query_symbols, conn, params=[start_time.isoformat()])
            
            # Decisões por ação
            query_actions = '''
                SELECT 
                    action,
                    COUNT(*) as decisions,
                    AVG(confidence) as avg_confidence
                FROM detailed_decisions 
                WHERE datetime(timestamp) >= datetime(?)
                GROUP BY action
                ORDER BY decisions DESC
            '''
            
            df_actions = pd.read_sql_query(query_actions, conn, params=[start_time.isoformat()])
            
            conn.close()
            
            return {
                "period_hours": hours,
                "hourly_activity": df_hourly.to_dict('records') if not df_hourly.empty else [],
                "symbol_breakdown": df_symbols.to_dict('records') if not df_symbols.empty else [],
                "action_breakdown": df_actions.to_dict('records') if not df_actions.empty else [],
                "total_decisions": df_hourly['decisions'].sum() if not df_hourly.empty else 0,
                "total_advanced": df_hourly['advanced_decisions'].sum() if not df_hourly.empty else 0
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_performance_summary(self) -> Dict:
        """Obtém resumo de performance"""
        if not Path(self.performance_db).exists():
            return {"error": "Performance database not found"}
        
        try:
            conn = sqlite3.connect(self.performance_db)
            
            # Últimas métricas
            query_latest = '''
                SELECT * FROM performance_metrics 
                ORDER BY timestamp DESC 
                LIMIT 1
            '''
            
            df_latest = pd.read_sql_query(query_latest, conn)
            
            # Tendências (últimos 7 registros)
            query_trends = '''
                SELECT 
                    timestamp,
                    total_decisions,
                    advanced_risk_decisions,
                    avg_atr_value,
                    decision_confidence_avg,
                    error_rate
                FROM performance_metrics 
                ORDER BY timestamp DESC 
                LIMIT 7
            '''
            
            df_trends = pd.read_sql_query(query_trends, conn)
            
            conn.close()
            
            result = {}
            
            if not df_latest.empty:
                latest = df_latest.iloc[0]
                result["latest_metrics"] = {
                    "timestamp": latest['timestamp'],
                    "total_decisions": latest['total_decisions'],
                    "advanced_decisions": latest['advanced_risk_decisions'],
                    "adoption_rate": round((latest['advanced_risk_decisions'] / latest['total_decisions'] * 100) if latest['total_decisions'] > 0 else 0, 1),
                    "avg_atr": round(latest['avg_atr_value'], 5),
                    "avg_confidence": round(latest['decision_confidence_avg'], 2),
                    "error_rate": round(latest['error_rate'] * 100, 1)
                }
            
            if not df_trends.empty and len(df_trends) > 1:
                # Calcular tendências
                first = df_trends.iloc[-1]  # Mais antigo
                last = df_trends.iloc[0]    # Mais recente
                
                result["trends"] = {
                    "decisions_change": last['total_decisions'] - first['total_decisions'],
                    "adoption_change": round(
                        (last['advanced_risk_decisions'] / last['total_decisions'] * 100 if last['total_decisions'] > 0 else 0) -
                        (first['advanced_risk_decisions'] / first['total_decisions'] * 100 if first['total_decisions'] > 0 else 0), 1
                    ),
                    "confidence_change": round(last['decision_confidence_avg'] - first['decision_confidence_avg'], 2),
                    "error_rate_change": round((last['error_rate'] - first['error_rate']) * 100, 1)
                }
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def display_dashboard(self):
        """Exibe dashboard completo"""
        print("\n" + "=" * 80)
        print("🚀 DASHBOARD DO SISTEMA DE TRADING FOREX ML")
        print("=" * 80)
        
        # Status do sistema
        print("\n📊 STATUS DO SISTEMA")
        print("-" * 40)
        status = self.get_system_status()
        
        print(f"⏰ Timestamp: {status['timestamp']}")
        print(f"💾 DB Decisões: {'✅' if status['databases']['decisions']['exists'] else '❌'} ({status['databases']['decisions']['size_mb']} MB)")
        print(f"📈 DB Performance: {'✅' if status['databases']['performance']['exists'] else '❌'} ({status['databases']['performance']['size_mb']} MB)")
        print(f"⚙️ Configuração: {'✅' if status['config']['loaded'] else '❌'}")
        
        if status['config']['loaded']:
            print(f"🔧 Risco Avançado: {'✅ Ativo' if status['config']['advanced_risk_enabled'] else '❌ Inativo'}")
            print(f"💱 Símbolos: {status['config']['symbols_count']}")
        
        # Atividade recente
        print("\n📈 ATIVIDADE RECENTE (24H)")
        print("-" * 40)
        activity = self.get_recent_activity(24)
        
        if "error" not in activity:
            print(f"📊 Total de Decisões: {activity['total_decisions']}")
            print(f"🎯 Com Risco Avançado: {activity['total_advanced']}")
            print(f"📊 Taxa de Adoção: {(activity['total_advanced']/activity['total_decisions']*100):.1f}%" if activity['total_decisions'] > 0 else "📊 Taxa de Adoção: 0%")
            
            if activity['symbol_breakdown']:
                print("\n💱 Por Símbolo:")
                for symbol in activity['symbol_breakdown'][:5]:  # Top 5
                    print(f"   {symbol['symbol']}: {symbol['decisions']} decisões (conf: {symbol['avg_confidence']:.2f})")
            
            if activity['action_breakdown']:
                print("\n🎯 Por Ação:")
                for action in activity['action_breakdown']:
                    print(f"   {action['action']}: {action['decisions']} decisões (conf: {action['avg_confidence']:.2f})")
        else:
            print(f"❌ Erro: {activity['error']}")
        
        # Performance
        print("\n🚀 MÉTRICAS DE PERFORMANCE")
        print("-" * 40)
        performance = self.get_performance_summary()
        
        if "error" not in performance:
            if "latest_metrics" in performance:
                metrics = performance["latest_metrics"]
                print(f"📊 Última Coleta: {metrics['timestamp']}")
                print(f"🎯 Taxa de Adoção: {metrics['adoption_rate']}%")
                print(f"📈 ATR Médio: {metrics['avg_atr']}")
                print(f"🎯 Confiança Média: {metrics['avg_confidence']}")
                print(f"⚠️ Taxa de Erro: {metrics['error_rate']}%")
            
            if "trends" in performance:
                trends = performance["trends"]
                print("\n📈 Tendências:")
                print(f"   Decisões: {'+' if trends['decisions_change'] >= 0 else ''}{trends['decisions_change']}")
                print(f"   Adoção: {'+' if trends['adoption_change'] >= 0 else ''}{trends['adoption_change']}%")
                print(f"   Confiança: {'+' if trends['confidence_change'] >= 0 else ''}{trends['confidence_change']}")
                print(f"   Taxa Erro: {'+' if trends['error_rate_change'] >= 0 else ''}{trends['error_rate_change']}%")
        else:
            print(f"❌ Erro: {performance['error']}")
        
        # Recomendações
        print("\n💡 RECOMENDAÇÕES")
        print("-" * 40)
        self._show_recommendations(status, activity, performance)
        
        print("\n" + "=" * 80)
        print("✅ Dashboard atualizado com sucesso")
        print("=" * 80)
    
    def _show_recommendations(self, status: Dict, activity: Dict, performance: Dict):
        """Mostra recomendações baseadas no status atual"""
        recommendations = []
        
        # Verificar configuração
        if not status['config']['loaded']:
            recommendations.append("⚠️ Verificar arquivo de configuração")
        elif not status['config']['advanced_risk_enabled']:
            recommendations.append("🔧 Ativar sistema de risco avançado na configuração")
        
        # Verificar atividade
        if "error" not in activity:
            if activity['total_decisions'] == 0:
                recommendations.append("📊 Sistema sem atividade recente - verificar conexão")
            elif activity['total_advanced'] == 0:
                recommendations.append("🎯 Nenhuma decisão com risco avançado - verificar integração")
            elif activity['total_decisions'] > 0:
                adoption_rate = activity['total_advanced'] / activity['total_decisions'] * 100
                if adoption_rate < 50:
                    recommendations.append(f"📈 Taxa de adoção baixa ({adoption_rate:.1f}%) - otimizar parâmetros")
        
        # Verificar performance
        if "error" not in performance and "latest_metrics" in performance:
            metrics = performance["latest_metrics"]
            if metrics['error_rate'] > 50:
                recommendations.append(f"⚠️ Alta taxa de erro ({metrics['error_rate']}%) - revisar sistema")
            if metrics['avg_confidence'] < 0.6:
                recommendations.append(f"🎯 Baixa confiança média ({metrics['avg_confidence']}) - ajustar modelos")
        
        if not recommendations:
            recommendations.append("✅ Sistema operando dentro dos parâmetros normais")
        
        for rec in recommendations:
            print(f"   {rec}")
    
    def run_continuous_monitoring(self, interval_minutes: int = 30):
        """Executa monitoramento contínuo"""
        print(f"🔄 Iniciando monitoramento contínuo (intervalo: {interval_minutes} min)")
        print("Pressione Ctrl+C para parar")
        
        try:
            while True:
                self.display_dashboard()
                print(f"\n⏳ Próxima atualização em {interval_minutes} minutos...")
                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n🛑 Monitoramento interrompido pelo usuário")

def main():
    """Função principal"""
    dashboard = SystemDashboard()
    
    # Exibir dashboard uma vez
    dashboard.display_dashboard()
    
    # Perguntar se quer monitoramento contínuo
    print("\n🔄 Deseja iniciar monitoramento contínuo? (s/n): ", end="")
    # Para automação, vamos apenas mostrar o dashboard uma vez
    print("n")
    print("\n📊 Dashboard executado com sucesso!")

if __name__ == "__main__":
    main()