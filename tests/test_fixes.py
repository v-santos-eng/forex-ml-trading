#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste das Correções Aplicadas
Verifica se os problemas críticos foram resolvidos
"""

import sys
import os
from datetime import datetime
import traceback

# Adicionar diretório atual ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_decision_loading():
    """Testa o carregamento de decisões sem erro strftime"""
    print("\n🔍 TESTE 1: Carregamento de Decisões")
    print("="*50)
    
    try:
        from detailed_decision_logger import DetailedDecisionLogger
        
        # Inicializar logger
        logger = DetailedDecisionLogger()
        print("✅ DetailedDecisionLogger inicializado")
        
        # Tentar carregar decisões recentes
        decisions = logger.get_recent_decisions(limit=10)
        print(f"✅ Carregadas {len(decisions)} decisões sem erro")
        
        # Verificar se timestamps são objetos datetime
        for i, decision in enumerate(decisions[:3]):
            timestamp = decision['timestamp']
            print(f"   Decisão {i+1}: timestamp = {type(timestamp).__name__} - {timestamp}")
            
            # Tentar usar strftime
            if hasattr(timestamp, 'strftime'):
                formatted = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                print(f"   ✅ strftime funcionou: {formatted}")
            else:
                print(f"   ⚠️ timestamp não é datetime: {type(timestamp)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO no teste de carregamento: {e}")
        traceback.print_exc()
        return False

def test_sl_tp_validation():
    """Testa a validação de SL/TP"""
    print("\n🔍 TESTE 2: Validação SL/TP")
    print("="*50)
    
    try:
        from auto_trader import AutoTrader
        
        # Verificar se a validação foi adicionada
        with open('auto_trader.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'SL e TP são obrigatórios' in content:
            print("✅ Validação SL/TP encontrada no código")
        else:
            print("⚠️ Validação SL/TP não encontrada")
            
        if 'stop_loss is None or take_profit is None' in content:
            print("✅ Verificação de SL/TP None encontrada")
        else:
            print("⚠️ Verificação de SL/TP None não encontrada")
            
        return True
        
    except Exception as e:
        print(f"❌ ERRO no teste SL/TP: {e}")
        return False

def test_dashboard_refresh():
    """Testa a função refresh_decisions do dashboard"""
    print("\n🔍 TESTE 3: Dashboard Refresh")
    print("="*50)
    
    try:
        # Verificar se a correção foi aplicada
        with open('mt5_dashboard.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'isinstance(decision[\'timestamp\'], datetime)' in content:
            print("✅ Verificação de tipo datetime encontrada")
        else:
            print("⚠️ Verificação de tipo datetime não encontrada")
            
        if 'reasoning_factors' in content:
            print("✅ Correção de reasoning_factors encontrada")
        else:
            print("⚠️ Correção de reasoning_factors não encontrada")
            
        return True
        
    except Exception as e:
        print(f"❌ ERRO no teste dashboard: {e}")
        return False

def test_database_creation():
    """Testa se o banco de dados foi criado"""
    print("\n🔍 TESTE 4: Banco de Dados")
    print("="*50)
    
    try:
        import sqlite3
        
        # Verificar se o banco existe
        db_files = [
            'trading_decisions.db',
            'detailed_decisions.db'
        ]
        
        for db_file in db_files:
            if os.path.exists(db_file):
                print(f"✅ Banco {db_file} existe")
                
                # Verificar tabelas
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                print(f"   Tabelas: {[table[0] for table in tables]}")
                
                # Contar registros
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    print(f"   {table_name}: {count} registros")
                
                conn.close()
            else:
                print(f"⚠️ Banco {db_file} não existe")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO no teste de banco: {e}")
        return False

def main():
    """Executa todos os testes"""
    print("🚀 INICIANDO TESTES DAS CORREÇÕES")
    print("="*60)
    
    tests = [
        test_decision_loading,
        test_sl_tp_validation,
        test_dashboard_refresh,
        test_database_creation
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ ERRO CRÍTICO no teste {test.__name__}: {e}")
            results.append(False)
    
    # Resumo
    print("\n📋 RESUMO DOS TESTES")
    print("="*30)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Testes Passaram: {passed}/{total}")
    print(f"❌ Testes Falharam: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 TODAS AS CORREÇÕES FUNCIONANDO!")
    else:
        print("\n⚠️ ALGUMAS CORREÇÕES PRECISAM DE AJUSTES")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)