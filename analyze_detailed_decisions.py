#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para analisar o banco de dados detailed_decisions.db
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

def analyze_detailed_decisions():
    """Analisa as decisões detalhadas no banco de dados"""
    
    db_path = 'detailed_decisions.db'
    
    if not os.path.exists(db_path):
        print(f"Banco de dados não encontrado: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verificar estrutura da tabela
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tabelas encontradas: {tables}")
        
        if not tables:
            print("Nenhuma tabela encontrada no banco de dados")
            conn.close()
            return
        
        # Analisar cada tabela
        for table in tables:
            table_name = table[0]
            print(f"\n=== Analisando tabela '{table_name}' ===")
            
            # Estrutura da tabela
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print(f"\nEstrutura da tabela '{table_name}':")
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")
            
            # Total de registros
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_records = cursor.fetchone()[0]
            print(f"\nTotal de registros: {total_records}")
            
            if total_records == 0:
                print(f"Nenhum registro encontrado na tabela '{table_name}'")
                continue
            
            # Últimas 10 entradas
            print(f"\nÚltimas 10 entradas em '{table_name}':")
            cursor.execute(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT 10")
            recent_entries = cursor.fetchall()
            
            for i, entry in enumerate(recent_entries, 1):
                print(f"  {i}: {entry}")
            
            # Se a tabela tem coluna timestamp, analisar por data
            column_names = [col[1] for col in columns]
            if 'timestamp' in column_names:
                print(f"\nAnálise por data para '{table_name}':")
                cursor.execute(f"""
                    SELECT DATE(timestamp) as date, COUNT(*) as count 
                    FROM {table_name} 
                    GROUP BY DATE(timestamp) 
                    ORDER BY date DESC 
                    LIMIT 5
                """)
                date_analysis = cursor.fetchall()
                
                for date_info in date_analysis:
                    print(f"  {date_info[0]}: {date_info[1]} registros")
                
                # Verificar última atualização
                cursor.execute(f"SELECT MAX(timestamp) FROM {table_name}")
                last_update = cursor.fetchone()[0]
                print(f"\nÚltima atualização em '{table_name}': {last_update}")
            
            # Se a tabela tem coluna decision, analisar decisões NA
            if 'decision' in column_names:
                print(f"\nAnálise de decisões NA em '{table_name}':")
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE decision = 'NA' OR decision IS NULL")
                na_count = cursor.fetchone()[0]
                print(f"  Total de decisões NA: {na_count}")
                
                if na_count > 0:
                    cursor.execute(f"""
                        SELECT timestamp, symbol, decision, confidence 
                        FROM {table_name} 
                        WHERE decision = 'NA' OR decision IS NULL 
                        ORDER BY timestamp DESC 
                        LIMIT 5
                    """)
                    na_decisions = cursor.fetchall()
                    print("\nÚltimas 5 decisões NA:")
                    for decision in na_decisions:
                        print(f"  {decision}")
            
            # Análise de hoje
            if 'timestamp' in column_names:
                today = datetime.now().strftime('%Y-%m-%d')
                print(f"\nAnálise de hoje ({today}) para '{table_name}':")
                cursor.execute(f"""
                    SELECT COUNT(*) FROM {table_name} 
                    WHERE DATE(timestamp) = ?
                """, (today,))
                today_count = cursor.fetchone()[0]
                print(f"  Registros de hoje: {today_count}")
                
                if today_count > 0:
                    cursor.execute(f"""
                        SELECT * FROM {table_name} 
                        WHERE DATE(timestamp) = ? 
                        ORDER BY timestamp DESC
                        LIMIT 5
                    """, (today,))
                    today_entries = cursor.fetchall()
                    print("\nÚltimos 5 registros de hoje:")
                    for entry in today_entries:
                        print(f"  {entry}")
        
        conn.close()
        
    except Exception as e:
        print(f"Erro ao analisar banco de dados: {e}")
        import traceback
        traceback.print_exc()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("Iniciando análise do banco de dados detailed_decisions...")
    analyze_detailed_decisions()
    print("\nAnálise concluída")