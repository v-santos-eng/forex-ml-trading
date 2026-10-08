#!/usr/bin/env python3
"""
Verificação específica do servidor MT5
"""

import os
import MetaTrader5 as mt5
import time

def check_mt5_server():
    """Verificar configuração do servidor MT5"""
    print("=== VERIFICAÇÃO DO SERVIDOR MT5 ===")
    
    # Lista de servidores demo comuns
    demo_servers = [
        "MetaQuotes-Demo",
        "MetaQuotes Software Corp.-Demo",
        "Demo01-MetaQuotes",
        "Demo02-MetaQuotes",
        "Demo03-MetaQuotes"
    ]
    
    # Contas demo comuns
    demo_accounts = [
        int(os.getenv("MT5_LOGIN", "0")),  # Conta atual
        0,        # Conta padrão
    ]
    
    print(f"Testando {len(demo_servers)} servidores com {len(demo_accounts)} contas...\n")
    
    for server in demo_servers:
        print(f"--- Testando servidor: {server} ---")
        
        for account in demo_accounts:
            try:
                print(f"  Tentando conta {account}...")
                
                # Tentar inicializar com servidor específico
                if account == 0:
                    # Sem login específico
                    result = mt5.initialize()
                else:
                    # Com login específico
                    result = mt5.initialize(
                        login=account,
                        server=server,
                        password=""  # Senha vazia para demo
                    )
                
                if result:
                    # Sucesso - obter informações
                    account_info = mt5.account_info()
                    terminal_info = mt5.terminal_info()
                    
                    if account_info:
                        print(f"  ✓ SUCESSO!")
                        print(f"    Conta: {account_info.login}")
                        print(f"    Servidor: {account_info.server}")
                        print(f"    Nome: {account_info.name}")
                        print(f"    Saldo: ${account_info.balance:,.2f}")
                        print(f"    Equity: ${account_info.equity:,.2f}")
                        print(f"    Moeda: {account_info.currency}")
                        print(f"    Alavancagem: 1:{account_info.leverage}")
                        
                        if terminal_info:
                            print(f"    Terminal: {terminal_info.name}")
                            print(f"    Build: {terminal_info.build}")
                            print(f"    Caminho: {terminal_info.path}")
                        
                        # Testar alguns símbolos
                        symbols = mt5.symbols_get()
                        if symbols:
                            print(f"    Símbolos disponíveis: {len(symbols)}")
                            
                            # Testar EURUSD
                            if mt5.symbol_select("EURUSD", True):
                                tick = mt5.symbol_info_tick("EURUSD")
                                if tick:
                                    print(f"    EURUSD: Bid={tick.bid}, Ask={tick.ask}")
                        
                        mt5.shutdown()
                        return True, server, account
                    
                else:
                    error = mt5.last_error()
                    print(f"  ✗ Falha: {error}")
                    
            except Exception as e:
                print(f"  ✗ Erro: {e}")
            
            # Pequena pausa entre tentativas
            time.sleep(0.5)
        
        print()
    
    return False, None, None

def check_current_connection():
    """Verificar se já existe uma conexão ativa"""
    print("=== VERIFICAÇÃO DE CONEXÃO EXISTENTE ===")
    
    try:
        # Tentar obter informações sem inicializar
        account_info = mt5.account_info()
        if account_info:
            print("✓ Conexão MT5 já ativa!")
            print(f"Conta: {account_info.login}")
            print(f"Servidor: {account_info.server}")
            print(f"Saldo: ${account_info.balance:,.2f}")
            return True
        else:
            print("✗ Nenhuma conexão ativa")
            return False
            
    except Exception as e:
        print(f"✗ Erro ao verificar conexão: {e}")
        return False

if __name__ == "__main__":
    print("VERIFICAÇÃO COMPLETA DO SERVIDOR MT5")
    print("=" * 50)
    
    # Verificar conexão existente primeiro
    has_connection = check_current_connection()
    print()
    
    if not has_connection:
        # Tentar estabelecer nova conexão
        success, server, account = check_mt5_server()
        
        if success:
            print(f"\n✓ CONFIGURAÇÃO CORRETA ENCONTRADA:")
            print(f"  Servidor: {server}")
            print(f"  Conta: {account}")
            print(f"\nUse estas configurações no sistema de trading.")
        else:
            print(f"\n✗ NENHUMA CONFIGURAÇÃO FUNCIONOU")
            print(f"\nVerifique:")
            print(f"1. Se o MetaTrader 5 está instalado e funcionando")
            print(f"2. Se há conexão com a internet")
            print(f"3. Se o firewall não está bloqueando")
            print(f"4. Tente reiniciar o MT5")
    else:
        print("\n✓ Sistema já conectado corretamente!")