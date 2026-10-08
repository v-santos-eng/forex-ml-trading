#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise de Predições e Performance dos Modelos ML
"""

import sqlite3
import pandas as pd
from datetime import datetime
import numpy as np

def analyze_predictions():
    """Analisa as predições dos modelos ML"""
    try:
        conn = sqlite3.connect('prediction_quality.db')
        
        # Últimas predições
        print("=== ÚLTIMAS 10 PREDIÇÕES ===")
        query = "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10"
        df_predictions = pd.read_sql_query(query, conn)
        
        if not df_predictions.empty:
            for _, row in df_predictions.iterrows():
                print(f"ID: {row['id']}, Par: {row['pair']}, Modelo: {row['model_name']}")
                print(f"  Timestamp: {row['timestamp']}")
                print(f"  Predito: {row['predicted_value']}, Classe: {row['predicted_class']}")
                print(f"  Confiança: {row['confidence']:.4f}")
                print(f"  Real: {row['actual_value']}, Classe Real: {row['actual_class']}")
                print(f"  Tempo Execução: {row['execution_time']:.4f}s")
                print("---")
        else:
            print("Nenhuma predição encontrada")
        
        # Estatísticas por modelo
        print("\n=== ESTATÍSTICAS POR MODELO ===")
        if not df_predictions.empty:
            stats = df_predictions.groupby('model_name').agg({
                'confidence': ['mean', 'std', 'min', 'max'],
                'execution_time': ['mean', 'std'],
                'id': 'count'
            }).round(4)
            print(stats)
        
        # Acurácia por modelo (quando temos valores reais)
        print("\n=== ACURÁCIA POR MODELO ===")
        if not df_predictions.empty:
            df_with_actual = df_predictions.dropna(subset=['actual_class'])
            if not df_with_actual.empty:
                accuracy_stats = df_with_actual.groupby('model_name').apply(
                    lambda x: (x['predicted_class'] == x['actual_class']).mean()
                ).round(4)
                print(accuracy_stats)
            else:
                print("Nenhuma predição com valor real encontrada")
        
        conn.close()
        
    except Exception as e:
        print(f"Erro na análise de predições: {e}")

def analyze_trading_decisions():
    """Analisa as decisões de trading"""
    try:
        conn = sqlite3.connect('detailed_decisions.db')
        
        print("\n=== ANÁLISE DE DECISÕES DE TRADING ===")
        query = "SELECT * FROM detailed_decisions ORDER BY timestamp DESC LIMIT 20"
        df_decisions = pd.read_sql_query(query, conn)
        
        if not df_decisions.empty:
            print(f"Total de decisões: {len(df_decisions)}")
            
            # Distribuição de ações
            action_dist = df_decisions['action'].value_counts()
            print("\nDistribuição de ações:")
            for action, count in action_dist.items():
                print(f"  {action}: {count} ({count/len(df_decisions)*100:.1f}%)")
            
            # Estatísticas de confiança
            print(f"\nConfiança média: {df_decisions['confidence'].mean():.4f}")
            print(f"Confiança std: {df_decisions['confidence'].std():.4f}")
            print(f"Confiança min: {df_decisions['confidence'].min():.4f}")
            print(f"Confiança max: {df_decisions['confidence'].max():.4f}")
            
            # Últimas decisões
            print("\n=== ÚLTIMAS 5 DECISÕES ===")
            for _, row in df_decisions.head(5).iterrows():
                print(f"Timestamp: {row['timestamp']}")
                print(f"  Par: {row['symbol']}, Ação: {row['action']}, Confiança: {row['confidence']:.4f}")
                print(f"  Bot Action: {row['bot_action']}")
                print("---")
        else:
            print("Nenhuma decisão encontrada")
        
        conn.close()
        
    except Exception as e:
        print(f"Erro na análise de decisões: {e}")

def analyze_performance_data():
    """Analisa dados de performance"""
    try:
        conn = sqlite3.connect('trading_monitor.db')
        
        print("\n=== ANÁLISE DE PERFORMANCE ===")
        query = "SELECT * FROM performance ORDER BY timestamp DESC LIMIT 100"
        df_performance = pd.read_sql_query(query, conn)
        
        if not df_performance.empty:
            print(f"Registros de performance: {len(df_performance)}")
            print(f"Balance atual: {df_performance.iloc[0]['balance']:.2f}")
            print(f"Equity atual: {df_performance.iloc[0]['equity']:.2f}")
            print(f"Profit atual: {df_performance.iloc[0]['profit']:.2f}")
            print(f"Drawdown atual: {df_performance.iloc[0]['drawdown']:.2f}%")
            
            # Estatísticas históricas
            print(f"\nBalance médio (últimos 100): {df_performance['balance'].mean():.2f}")
            print(f"Profit médio: {df_performance['profit'].mean():.2f}")
            print(f"Drawdown médio: {df_performance['drawdown'].mean():.2f}%")
            print(f"Drawdown máximo: {df_performance['drawdown'].max():.2f}%")
        else:
            print("Nenhum dado de performance encontrado")
        
        conn.close()
        
    except Exception as e:
        print(f"Erro na análise de performance: {e}")

if __name__ == "__main__":
    print("🔍 ANÁLISE DETALHADA DE PREDIÇÕES E PERFORMANCE")
    print("=" * 60)
    
    analyze_predictions()
    analyze_trading_decisions()
    analyze_performance_data()
    
    print("\n✅ ANÁLISE CONCLUÍDA")