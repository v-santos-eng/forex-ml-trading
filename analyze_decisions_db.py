#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para analisar o banco de dados de decisões de trading
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

def analyze_trading_decisions():
    """Analisa as decisões de trading no banco de dados"""
    
    db_path = 'trading_decisions.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Banco de dados não encontrado: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Verificar estrutura da tabela
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📊 Tabelas encontradas: {tables}")
        
        # Verificar se existe tabela decisions
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='decisions';")
        if not cursor.fetchone():
            print("❌ Tabela 'decisions' não encontrada")
            
            # Listar todas as tabelas e suas estruturas
            for table in tables:
                table_name = table[0]
                print(f"\n📋 Estrutura da tabela '{table_name}':")
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                for col in columns:
                    print(f"  - {col[1]} ({col[2]})")
                
                # Mostrar algumas linhas de exemplo
                cursor.execute(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT 5")
                rows = cursor.fetchall()
                print(f"\n📝 Últimas 5 entradas em '{table_name}':")
                for i, row in enumerate(rows, 1):
                    print(f"  {i}: {row}")
            
            conn.close()
            return
        
        # Analisar tabela decisions
        print("\n🔍 Analisando tabela 'decisions'...")
        
        # Estrutura da tabela
        cursor.execute("PRAGMA table_info(decisions)")
        columns = cursor.fetchall()
        print("\n📋 Estrutura da tabela 'decisions':")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        # Total de registros
        cursor.execute("SELECT COUNT(*) FROM decisions")
        total_records = cursor.fetchone()[0]
        print(f"\n📊 Total de registros: {total_records}")
        
        if total_records == 0:
            print("❌ Nenhuma decisão encontrada no banco de dados")
            conn.close()
            return
        
        # Últimas 20 decisões
        print("\n📝 Últimas 20 decisões:")
        cursor.execute("SELECT * FROM decisions ORDER BY timestamp DESC LIMIT 20")
        recent_decisions = cursor.fetchall()
        
        for i, decision in enumerate(recent_decisions, 1):
            print(f"  {i}: {decision}")
        
        # Análise por data
        print("\n📅 Análise por data:")
        cursor.execute("""
            SELECT DATE(timestamp) as date, COUNT(*) as count 
            FROM decisions 
            GROUP BY DATE(timestamp) 
            ORDER BY date DESC 
            LIMIT 10
        """)
        date_analysis = cursor.fetchall()
        
        for date_info in date_analysis:
            print(f"  {date_info[0]}: {date_info[1]} decisões")
        
        # Análise de decisões NA
        print("\n🔍 Análise de decisões NA:")
        cursor.execute("SELECT COUNT(*) FROM decisions WHERE decision = 'NA' OR decision IS NULL")
        na_count = cursor.fetchone()[0]
        print(f"  Total de decisões NA: {na_count}")
        
        if na_count > 0:
            cursor.execute("""
                SELECT timestamp, symbol, decision, confidence 
                FROM decisions 
                WHERE decision = 'NA' OR decision IS NULL 
                ORDER BY timestamp DESC 
                LIMIT 10
            """)
            na_decisions = cursor.fetchall()
            print("\n📝 Últimas 10 decisões NA:")
            for decision in na_decisions:
                print(f"  {decision}")
        
        # Análise de hoje
        today = datetime.now().strftime('%Y-%m-%d')
        print(f"\n📅 Análise de hoje ({today}):")
        cursor.execute("""
            SELECT COUNT(*) FROM decisions 
            WHERE DATE(timestamp) = ?
        """, (today,))
        today_count = cursor.fetchone()[0]
        print(f"  Decisões de hoje: {today_count}")
        
        if today_count > 0:
            cursor.execute("""
                SELECT timestamp, symbol, decision, confidence 
                FROM decisions 
                WHERE DATE(timestamp) = ? 
                ORDER BY timestamp DESC
            """, (today,))
            today_decisions = cursor.fetchall()
            print("\n📝 Decisões de hoje:")
            for decision in today_decisions:
                print(f"  {decision}")
        
        # Verificar última atualização
        cursor.execute("SELECT MAX(timestamp) FROM decisions")
        last_update = cursor.fetchone()[0]
        print(f"\n⏰ Última atualização: {last_update}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Erro ao analisar banco de dados: {e}")
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("🔍 Iniciando análise do banco de dados de decisões...")
    analyze_trading_decisions()
    print("\n✅ Análise concluída")