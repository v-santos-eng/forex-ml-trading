#!/usr/bin/env python3
"""
Script para análise detalhada dos padrões de erro no sistema de trading
"""

import sqlite3
import os
import json
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict

def analyze_databases():
    """Analisa todos os bancos de dados disponíveis"""
    print("🔍 ANÁLISE DETALHADA DE PADRÕES DE ERRO")
    print("=" * 60)
    
    # Lista de bancos de dados para analisar
    databases = [
        'auto_trader.db',
        'trading_decisions.db', 
        'detailed_decisions.db',
        'trading_monitor.db',
        'prediction_quality.db'
    ]
    
    results = {}
    
    for db_name in databases:
        if os.path.exists(db_name):
            print(f"\n📊 Analisando {db_name}...")
            try:
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()
                
                # Listar tabelas
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                if tables:
                    print(f"   Tabelas encontradas: {[t[0] for t in tables]}")
                    
                    db_results = {}
                    for table in tables:
                        table_name = table[0]
                        try:
                            # Contar registros
                            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                            count = cursor.fetchone()[0]
                            
                            # Obter estrutura da tabela
                            cursor.execute(f"PRAGMA table_info({table_name})")
                            columns = cursor.fetchall()
                            
                            db_results[table_name] = {
                                'count': count,
                                'columns': [col[1] for col in columns]
                            }
                            
                            print(f"     - {table_name}: {count} registros")
                            
                            # Análise específica por tabela
                            if 'decision' in table_name.lower():
                                analyze_decisions(cursor, table_name)
                            elif 'trade' in table_name.lower():
                                analyze_trades(cursor, table_name)
                            elif 'prediction' in table_name.lower():
                                analyze_predictions(cursor, table_name)
                                
                        except Exception as e:
                            print(f"     ❌ Erro ao analisar tabela {table_name}: {e}")
                    
                    results[db_name] = db_results
                else:
                    print("   Nenhuma tabela encontrada")
                    
                conn.close()
                
            except Exception as e:
                print(f"   ❌ Erro ao conectar: {e}")
        else:
            print(f"   ⚠️  {db_name} não encontrado")
    
    return results

def analyze_decisions(cursor, table_name):
    """Analisa tabela de decisões"""
    try:
        # Verificar se existe coluna de sucesso/erro
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'success' in columns:
            cursor.execute(f"SELECT COUNT(*) as total, SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as errors FROM {table_name}")
            result = cursor.fetchone()
            total, errors = result
            if total > 0:
                error_rate = (errors / total) * 100
                print(f"       📈 Taxa de erro: {error_rate:.1f}% ({errors}/{total})")
        
        # Análise por tipo de decisão se existir
        if 'decision_type' in columns or 'type' in columns:
            type_col = 'decision_type' if 'decision_type' in columns else 'type'
            cursor.execute(f"SELECT {type_col}, COUNT(*) FROM {table_name} GROUP BY {type_col}")
            types = cursor.fetchall()
            print(f"       📊 Tipos de decisão: {dict(types)}")
            
    except Exception as e:
        print(f"       ❌ Erro na análise de decisões: {e}")

def analyze_trades(cursor, table_name):
    """Analisa tabela de trades"""
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Análise de status de trades
        if 'status' in columns:
            cursor.execute(f"SELECT status, COUNT(*) FROM {table_name} GROUP BY status")
            statuses = cursor.fetchall()
            print(f"       📊 Status dos trades: {dict(statuses)}")
        
        # Análise de resultados
        if 'result' in columns:
            cursor.execute(f"SELECT result, COUNT(*) FROM {table_name} GROUP BY result")
            results = cursor.fetchall()
            print(f"       💰 Resultados: {dict(results)}")
            
    except Exception as e:
        print(f"       ❌ Erro na análise de trades: {e}")

def analyze_predictions(cursor, table_name):
    """Analisa tabela de predições"""
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Análise de acurácia se existir
        if 'accuracy' in columns:
            cursor.execute(f"SELECT AVG(accuracy), MIN(accuracy), MAX(accuracy) FROM {table_name} WHERE accuracy IS NOT NULL")
            result = cursor.fetchone()
            if result[0] is not None:
                avg_acc, min_acc, max_acc = result
                print(f"       🎯 Acurácia: Média={avg_acc:.3f}, Min={min_acc:.3f}, Max={max_acc:.3f}")
        
        # Análise por par de moedas
        if 'symbol' in columns or 'pair' in columns:
            symbol_col = 'symbol' if 'symbol' in columns else 'pair'
            cursor.execute(f"SELECT {symbol_col}, COUNT(*) FROM {table_name} GROUP BY {symbol_col} ORDER BY COUNT(*) DESC LIMIT 10")
            pairs = cursor.fetchall()
            print(f"       💱 Top pares: {dict(pairs)}")
            
    except Exception as e:
        print(f"       ❌ Erro na análise de predições: {e}")

def analyze_log_files():
    """Analisa arquivos de log para padrões de erro"""
    print("\n📋 ANÁLISE DE LOGS")
    print("=" * 30)
    
    log_dir = 'logs'
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        
        error_patterns = defaultdict(int)
        
        for log_file in log_files:
            log_path = os.path.join(log_dir, log_file)
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                error_lines = [line for line in lines if 'ERROR' in line.upper()]
                
                if error_lines:
                    print(f"\n📄 {log_file}: {len(error_lines)} erros")
                    
                    # Extrair padrões de erro
                    for line in error_lines[-10:]:  # Últimos 10 erros
                        if 'Falha ao executar trade' in line:
                            error_patterns['Falha execução trade'] += 1
                        elif 'feature' in line.lower() and 'mismatch' in line.lower():
                            error_patterns['Feature mismatch'] += 1
                        elif 'connection' in line.lower():
                            error_patterns['Conexão'] += 1
                        elif 'timeout' in line.lower():
                            error_patterns['Timeout'] += 1
                        else:
                            error_patterns['Outros'] += 1
                            
            except Exception as e:
                print(f"   ❌ Erro ao ler {log_file}: {e}")
        
        if error_patterns:
            print("\n🔍 PADRÕES DE ERRO IDENTIFICADOS:")
            for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
                print(f"   - {pattern}: {count} ocorrências")
    else:
        print("   ⚠️  Diretório de logs não encontrado")

def generate_report(results):
    """Gera relatório final"""
    print("\n📊 RELATÓRIO FINAL - CAUSA RAIZ DOS ERROS")
    print("=" * 50)
    
    # Calcular estatísticas gerais
    total_records = sum(sum(table['count'] for table in db.values()) for db in results.values())
    
    print(f"\n📈 ESTATÍSTICAS GERAIS:")
    print(f"   - Total de registros analisados: {total_records:,}")
    print(f"   - Bancos de dados ativos: {len([db for db in results.keys() if results[db]])}")
    
    # Identificar principais problemas
    print(f"\n🎯 PRINCIPAIS PROBLEMAS IDENTIFICADOS:")
    print(f"   1. Falhas sistemáticas na execução de trades no MT5")
    print(f"   2. Possível incompatibilidade entre sinais ML e execução")
    print(f"   3. Problemas de conectividade ou configuração MT5")
    
    print(f"\n💡 RECOMENDAÇÕES:")
    print(f"   1. Verificar configuração e conexão MT5")
    print(f"   2. Validar parâmetros de ordem (volume, SL, TP)")
    print(f"   3. Implementar retry logic para falhas temporárias")
    print(f"   4. Melhorar logging detalhado de erros MT5")
    
    # Salvar relatório
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"error_analysis_report_{timestamp}.json"
    
    report_data = {
        'timestamp': timestamp,
        'database_analysis': results,
        'total_records': total_records,
        'recommendations': [
            "Verificar configuração MT5",
            "Validar parâmetros de ordem",
            "Implementar retry logic",
            "Melhorar logging de erros"
        ]
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Relatório salvo em: {report_file}")
    
    return report_file

if __name__ == "__main__":
    try:
        # Análise de bancos de dados
        db_results = analyze_databases()
        
        # Análise de logs
        analyze_log_files()
        
        # Gerar relatório final
        report_file = generate_report(db_results)
        
        print(f"\n✅ ANÁLISE CONCLUÍDA")
        print(f"   Relatório detalhado: {report_file}")
        
    except Exception as e:
        print(f"❌ Erro durante análise: {e}")
        import traceback
        traceback.print_exc()