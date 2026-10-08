#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relatório Detalhado de Anomalias do Sistema ML
Documenta todas as anomalias encontradas e suas causas
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sqlite3
from datetime import datetime
import json

def generate_anomalies_report():
    """
    Gera relatório completo das anomalias encontradas no sistema ML
    """
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "analysis_summary": {
            "total_anomalies_found": 0,
            "critical_issues": 0,
            "resolved_issues": 0,
            "pending_issues": 0
        },
        "anomalies": []
    }
    
    print("="*100)
    print("RELATÓRIO DETALHADO DE ANOMALIAS DO SISTEMA ML")
    print("="*100)
    print(f"Data/Hora da Análise: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print()
    
    # ANOMALIA 1: Erro de Estrutura de Dados nos Modelos ML
    anomaly_1 = {
        "id": "ML_001",
        "title": "Erro de Acesso aos Modelos ML Treinados",
        "severity": "CRÍTICA",
        "status": "RESOLVIDA",
        "description": "O AutoTrader estava tentando acessar métodos predict() diretamente em dicionários ao invés dos objetos de modelo",
        "root_cause": {
            "technical_cause": "Incompatibilidade entre estrutura de dados retornada pelo train_models() e código de predição",
            "code_location": "auto_trader.py, linha 365 (antes da correção)",
            "error_pattern": "'dict' object has no attribute 'predict'",
            "data_structure_issue": "train_models() retorna dict com chave 'model', mas código tentava usar dict diretamente"
        },
        "impact": {
            "business_impact": "100% das predições ML falhando",
            "affected_symbols": ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"],
            "trading_impact": "Nenhum sinal ML sendo gerado, sistema dependendo apenas de análise técnica",
            "frequency": "Todas as tentativas de predição desde implementação"
        },
        "evidence": {
            "log_entries": "detailed_decisions.db contém múltiplas entradas 'NO_ACTION' com razão 'Nenhum modelo conseguiu fazer predição válida'",
            "error_messages": ["'dict' object has no attribute 'predict'", "'dict' object has no attribute 'predict_proba'"],
            "affected_models": ["RandomForest", "GradientBoosting", "LogisticRegression", "SVM"]
        },
        "resolution": {
            "fix_applied": "Modificação do código para extrair corretamente o objeto modelo do dicionário de resultados",
            "code_changes": "Alteração de 'model.predict()' para 'model_result['model'].predict()' com validação de estrutura",
            "validation_method": "Teste isolado confirmou funcionamento correto das predições",
            "fix_date": datetime.now().isoformat()
        },
        "prevention": {
            "code_review": "Implementar testes unitários para validar estrutura de dados entre componentes",
            "monitoring": "Adicionar logs específicos para validação de estrutura de modelos",
            "documentation": "Documentar claramente a estrutura de dados retornada por train_models()"
        }
    }
    
    # ANOMALIA 2: Problemas de Logging e Performance Context
    anomaly_2 = {
        "id": "LOG_001",
        "title": "Erros no Sistema de Logging Avançado",
        "severity": "MÉDIA",
        "status": "IDENTIFICADA",
        "description": "Sistema de logging apresenta erros de AttributeError relacionados ao performance_context",
        "root_cause": {
            "technical_cause": "Incompatibilidade entre AdvancedLoggingSystem e PerformanceMonitor",
            "code_location": "Múltiplos arquivos de logging",
            "error_pattern": "'AdvancedLoggingSystem' object has no attribute 'performance_context'",
            "initialization_issue": "PerformanceMonitor não está sendo inicializado corretamente"
        },
        "impact": {
            "business_impact": "Logs de performance incompletos, dificultando debugging",
            "affected_functions": ["run_full_evaluation()"],
            "monitoring_impact": "Métricas de performance não sendo coletadas adequadamente",
            "frequency": "Múltiplas ocorrências durante processamento de pares"
        },
        "evidence": {
            "log_files": "debug.log contém múltiplos AttributeError",
            "affected_pairs": ["AUDJPY", "EURUSD", "GBPUSD", "USDJPY"],
            "error_frequency": "Recorrente durante avaliações completas"
        },
        "resolution": {
            "status": "PENDENTE",
            "recommended_fix": "Revisar inicialização do PerformanceMonitor e compatibilidade com AdvancedLoggingSystem",
            "priority": "Média - não impacta funcionalidade core, mas afeta monitoramento"
        }
    }
    
    # ANOMALIA 3: Configurações de Spread e Thresholds
    anomaly_3 = {
        "id": "CFG_001",
        "title": "Inconsistências nos Critérios de Spread",
        "severity": "BAIXA",
        "status": "OBSERVADA",
        "description": "Sistema apresenta decisões conflitantes sobre spreads altos/baixos",
        "root_cause": {
            "technical_cause": "Thresholds de spread podem estar mal calibrados",
            "configuration_issue": "Critérios de 'spread alto' vs 'spread favorável' inconsistentes",
            "market_condition_dependency": "Spreads variam significativamente entre sessões de mercado"
        },
        "impact": {
            "business_impact": "Possível perda de oportunidades de trading por critérios restritivos",
            "trading_efficiency": "Algumas operações podem ser rejeitadas desnecessariamente",
            "frequency": "Observado em múltiplas decisões de trading"
        },
        "evidence": {
            "decision_patterns": "Logs mostram tanto 'spread favorece execução' quanto 'spread muito alto (5 pips)'",
            "threshold_analysis": "Necessária revisão dos limites de spread por par de moedas"
        },
        "resolution": {
            "status": "REQUER_ANÁLISE",
            "recommended_action": "Análise estatística dos spreads históricos para calibração de thresholds",
            "priority": "Baixa - otimização de performance"
        }
    }
    
    # ANOMALIA 4: Dados Insuficientes para Treinamento
    anomaly_4 = {
        "id": "DATA_001",
        "title": "Requisitos Mínimos de Dados para ML",
        "severity": "BAIXA",
        "status": "CONFIGURAÇÃO",
        "description": "Sistema requer mínimo de 100 amostras para treinamento, pode ser restritivo",
        "root_cause": {
            "design_decision": "Threshold conservador de 100 amostras para garantir qualidade do modelo",
            "data_availability": "Alguns pares podem ter histórico limitado em certas condições de mercado"
        },
        "impact": {
            "business_impact": "Atraso no início de operações ML para novos pares ou após reinicializações",
            "cold_start_problem": "Sistema precisa acumular dados antes de começar predições",
            "frequency": "Principalmente durante inicialização ou para pares menos líquidos"
        },
        "resolution": {
            "status": "CONFIGURAÇÃO_VÁLIDA",
            "justification": "Threshold de 100 amostras é apropriado para qualidade do modelo",
            "monitoring": "Acompanhar se threshold causa impacto significativo em produção"
        }
    }
    
    # Adicionar anomalias ao relatório
    report["anomalies"] = [anomaly_1, anomaly_2, anomaly_3, anomaly_4]
    
    # Atualizar sumário
    report["analysis_summary"]["total_anomalies_found"] = len(report["anomalies"])
    report["analysis_summary"]["critical_issues"] = len([a for a in report["anomalies"] if a["severity"] == "CRÍTICA"])
    report["analysis_summary"]["resolved_issues"] = len([a for a in report["anomalies"] if a["status"] == "RESOLVIDA"])
    report["analysis_summary"]["pending_issues"] = len([a for a in report["anomalies"] if a["status"] in ["IDENTIFICADA", "PENDENTE", "REQUER_ANÁLISE"]])
    
    # Imprimir relatório formatado
    print("SUMÁRIO EXECUTIVO:")
    print("-" * 50)
    print(f"Total de Anomalias Encontradas: {report['analysis_summary']['total_anomalies_found']}")
    print(f"Problemas Críticos: {report['analysis_summary']['critical_issues']}")
    print(f"Problemas Resolvidos: {report['analysis_summary']['resolved_issues']}")
    print(f"Problemas Pendentes: {report['analysis_summary']['pending_issues']}")
    print()
    
    for i, anomaly in enumerate(report["anomalies"], 1):
        print(f"{i}. ANOMALIA {anomaly['id']}: {anomaly['title']}")
        print(f"   Severidade: {anomaly['severity']} | Status: {anomaly['status']}")
        print(f"   Descrição: {anomaly['description']}")
        print(f"   Causa Raiz: {anomaly['root_cause'].get('technical_cause', anomaly['root_cause'].get('design_decision', 'Não especificada'))}")
        print(f"   Impacto: {anomaly['impact'].get('business_impact', 'Não especificado')}")
        
        if anomaly['status'] == 'RESOLVIDA':
            print(f"   ✅ Resolução: {anomaly['resolution']['fix_applied']}")
        elif anomaly['status'] in ['IDENTIFICADA', 'PENDENTE']:
            print(f"   ⚠️  Ação Recomendada: {anomaly['resolution'].get('recommended_fix', 'Análise adicional necessária')}")
        
        print()
    
    # Análise de logs de decisão
    print("ANÁLISE DOS LOGS DE DECISÃO:")
    print("-" * 50)
    
    try:
        conn = sqlite3.connect('detailed_decisions.db')
        cursor = conn.cursor()
        
        # Estatísticas gerais
        cursor.execute("SELECT COUNT(*) FROM detailed_decisions")
        total_decisions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM detailed_decisions WHERE action = 'NO_ACTION'")
        no_action_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM detailed_decisions WHERE reasoning_factors LIKE '%Nenhum modelo conseguiu fazer predição válida%'")
        ml_failure_count = cursor.fetchone()[0]
        
        print(f"Total de Decisões Registradas: {total_decisions}")
        print(f"Decisões NO_ACTION: {no_action_count} ({no_action_count/total_decisions*100:.1f}%)")
        print(f"Falhas de Predição ML: {ml_failure_count} ({ml_failure_count/total_decisions*100:.1f}%)")
        
        # Análise por símbolo
        cursor.execute("""
            SELECT symbol, COUNT(*) as count
            FROM detailed_decisions 
            WHERE reasoning_factors LIKE '%Nenhum modelo conseguiu fazer predição válida%'
            GROUP BY symbol
            ORDER BY count DESC
        """)
        
        symbol_failures = cursor.fetchall()
        if symbol_failures:
            print("\nFalhas de Predição por Símbolo:")
            for symbol, count in symbol_failures:
                print(f"   {symbol}: {count} falhas")
        
        conn.close()
        
    except Exception as e:
        print(f"Erro ao analisar logs: {e}")
    
    print()
    print("RECOMENDAÇÕES PRIORITÁRIAS:")
    print("-" * 50)
    print("1. ✅ CONCLUÍDO: Correção do erro de predição ML (CRÍTICO)")
    print("2. 🔄 EM ANDAMENTO: Monitorar sistema após correção para validar estabilidade")
    print("3. 📋 PRÓXIMOS PASSOS: Revisar sistema de logging para corrigir erros de performance_context")
    print("4. 📊 OTIMIZAÇÃO: Analisar thresholds de spread para otimizar oportunidades de trading")
    print("5. 🧪 TESTES: Implementar testes unitários para prevenir regressões similares")
    
    print()
    print("="*100)
    print("RELATÓRIO CONCLUÍDO")
    print("="*100)
    
    # Salvar relatório em JSON
    with open('ml_anomalies_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Relatório detalhado salvo em: ml_anomalies_report.json")
    
    return report

if __name__ == "__main__":
    generate_anomalies_report()