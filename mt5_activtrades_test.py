#!/usr/bin/env python3
"""
Teste específico para servidor ActivTradesMarkets-Server
"""

import os
import MetaTrader5 as mt5
import time

def test_activtrades_server():
    """Testar especificamente o servidor ActivTradesMarkets-Server"""
    print("=== TESTE SERVIDOR ACTIVTRADES ===")
    print("Servidor: ActivTradesMarkets-Server")
    print("Conta: <MT5_LOGIN>")
    print()
    
    try:
        print("[1/3] Verificando conexão existente...")
        account_info = mt5.account_info()
        if account_info:
            print(f"✓ Conexão ativa encontrada!")
            print(f"  Conta atual: {account_info.login}")
            print(f"  Servidor atual: {account_info.server}")
            print(f"  Saldo: ${account_info.balance:,.2f}")
            
            if account_info.server == "ActivTradesMarkets-Server":
                print("✓ Servidor correto já conectado!")
                return True
            else:
                print(f"⚠ Servidor diferente. Tentando reconectar...")
                mt5.shutdown()
        else:
            print("✗ Nenhuma conexão ativa")
        
        print("\n[2/3] Tentando conectar ao ActivTradesMarkets-Server...")
        
        # Tentar com diferentes configurações
        configs = [
            {
                "login": int(os.getenv("MT5_LOGIN", "0")),
                "server": "ActivTradesMarkets-Server",
                "password": ""
            },
            {
                "login": int(os.getenv("MT5_LOGIN", "0")),
                "server": "ActivTradesMarkets-Server",
                "password": None
            },
            {
                "server": "ActivTradesMarkets-Server"
            }
        ]
        
        for i, config in enumerate(configs, 1):
            print(f"  Tentativa {i}: {config}")
            
            try:
                if "login" in config and "password" in config:
                    result = mt5.initialize(
                        login=config["login"],
                        server=config["server"],
                        password=config["password"]
                    )
                else:
                    result = mt5.initialize(server=config["server"])
                
                if result:
                    account_info = mt5.account_info()
                    if account_info:
                        print(f"  ✓ SUCESSO!")
                        print(f"    Conta: {account_info.login}")
                        print(f"    Servidor: {account_info.server}")
                        print(f"    Saldo: ${account_info.balance:,.2f}")
                        print(f"    Moeda: {account_info.currency}")
                        
                        # Testar funcionalidades básicas
                        print("\n[3/3] Testando funcionalidades...")
                        
                        # Testar símbolos
                        symbols = mt5.symbols_get()
                        if symbols:
                            print(f"  ✓ Símbolos disponíveis: {len(symbols)}")
                            
                            # Testar alguns pares principais
                            test_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
                            working_symbols = []
                            
                            for symbol in test_symbols:
                                if mt5.symbol_select(symbol, True):
                                    tick = mt5.symbol_info_tick(symbol)
                                    if tick:
                                        working_symbols.append(symbol)
                                        print(f"    {symbol}: Bid={tick.bid}, Ask={tick.ask}")
                            
                            if working_symbols:
                                print(f"  ✓ {len(working_symbols)} símbolos funcionando")
                            else:
                                print(f"  ⚠ Nenhum símbolo de teste funcionando")
                        
                        # Testar informações do terminal
                        terminal_info = mt5.terminal_info()
                        if terminal_info:
                            print(f"  ✓ Terminal: {terminal_info.name} Build {terminal_info.build}")
                        
                        return True
                    else:
                        print(f"  ✗ Falha ao obter informações da conta")
                else:
                    error = mt5.last_error()
                    print(f"  ✗ Falha na inicialização: {error}")
                    
            except Exception as e:
                print(f"  ✗ Erro: {e}")
            
            time.sleep(1)
        
        print("\n✗ Todas as tentativas falharam")
        return False
        
    except Exception as e:
        print(f"✗ Erro geral: {e}")
        return False
    
    finally:
        # Não fazer shutdown se conectou com sucesso
        pass

def check_server_alternatives():
    """Verificar servidores alternativos similares"""
    print("\n=== SERVIDORES ALTERNATIVOS ===")
    
    alternatives = [
        "ActivTrades-Server",
        "ActivTrades-Demo",
        "ActivTradesMarkets-Demo",
        "ActivTrades-Live",
        "ActivTradesMarkets-Live"
    ]
    
    for server in alternatives:
        print(f"\nTestando: {server}")
        try:
            result = mt5.initialize(server=server)
            if result:
                account_info = mt5.account_info()
                if account_info:
                    print(f"  ✓ Conectado: Conta {account_info.login}, Servidor {account_info.server}")
                    mt5.shutdown()
                    return server
                else:
                    print(f"  ✗ Sem informações da conta")
            else:
                error = mt5.last_error()
                print(f"  ✗ Falha: {error}")
        except Exception as e:
            print(f"  ✗ Erro: {e}")
        
        time.sleep(0.5)
    
    return None

if __name__ == "__main__":
    print("TESTE SERVIDOR ACTIVTRADES")
    print("=" * 40)
    
    success = test_activtrades_server()
    
    if not success:
        print("\nTentando servidores alternativos...")
        alternative = check_server_alternatives()
        
        if alternative:
            print(f"\n✓ SERVIDOR ALTERNATIVO ENCONTRADO: {alternative}")
            print(f"Use este servidor no lugar de 'ActivTradesMarkets-Server'")
        else:
            print(f"\n✗ NENHUM SERVIDOR ACTIVTRADES FUNCIONOU")
            print(f"\nVerificações necessárias:")
            print(f"1. Confirme o nome exato do servidor")
            print(f"2. Verifique se a conta está ativa")
            print(f"3. Confirme as credenciais")
            print(f"4. Verifique a conexão com a internet")
    else:
        print(f"\n✓ SERVIDOR ACTIVTRADES FUNCIONANDO CORRETAMENTE!")