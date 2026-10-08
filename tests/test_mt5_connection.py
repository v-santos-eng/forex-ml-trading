#!/usr/bin/env python3
"""
Teste rápido de conexão MT5
"""

import MetaTrader5 as mt5
import sys
import os

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt5_integration import MT5Integration, MT5Config

def test_direct_mt5():
    """Teste direto da API MT5"""
    print("=== Teste Direto MT5 ===")
    
    # Verificar se MT5 está disponível
    if not mt5.initialize():
        print(f"Erro ao inicializar MT5: {mt5.last_error()}")
        return False
    
    # Verificar informações da conta
    account_info = mt5.account_info()
    if account_info:
        print(f"Conta: {account_info.login}")
        print(f"Servidor: {account_info.server}")
        print(f"Saldo: ${account_info.balance:,.2f}")
        print("✓ Conexão MT5 direta OK")
        mt5.shutdown()
        return True
    else:
        print(f"Erro ao obter info da conta: {mt5.last_error()}")
        mt5.shutdown()
        return False

def test_mt5_integration():
    """Teste da classe MT5Integration"""
    print("\n=== Teste MT5Integration ===")
    
    try:
        # Usar configuração vazia para reutilizar conexão existente
        config = MT5Config(
            login=0,  # Usar conexão existente
            password="",
            server="",
            path=""
        )
        
        mt5_int = MT5Integration(config)
        
        if mt5_int.connect():
            print("✓ MT5Integration conectado")
            
            # Testar método básico
            account_info = mt5_int.get_account_info()
            if account_info:
                print(f"Conta via Integration: {account_info['login']}")
                print(f"Saldo via Integration: ${account_info['balance']:,.2f}")
            
            return True
        else:
            print("✗ Falha na conexão MT5Integration")
            return False
            
    except Exception as e:
        print(f"Erro no teste MT5Integration: {e}")
        return False

if __name__ == "__main__":
    print("Diagnóstico de Conexão MT5")
    print("=" * 30)
    
    # Teste 1: API direta
    direct_ok = test_direct_mt5()
    
    # Teste 2: Classe Integration
    if direct_ok:
        integration_ok = test_mt5_integration()
        
        if integration_ok:
            print("\n✓ Todos os testes passaram")
        else:
            print("\n✗ Problema na classe MT5Integration")
    else:
        print("\n✗ Problema na conexão MT5 básica")