#!/usr/bin/env python3
"""
Teste básico MT5 - Diagnóstico de problemas IPC
"""

import os
import MetaTrader5 as mt5
import time
import sys

def test_mt5_basic():
    """Teste básico da conexão MT5"""
    print("=== DIAGNÓSTICO MT5 BÁSICO ===")
    print(f"Versão MT5 Python: {mt5.__version__}")
    
    # Teste 1: Verificar se MT5 está disponível
    print("\n[1/4] Verificando disponibilidade MT5...")
    try:
        # Tentar inicializar sem parâmetros
        result = mt5.initialize()
        if result:
            print("✓ MT5 inicializado com sucesso")
            
            # Obter informações básicas
            terminal_info = mt5.terminal_info()
            if terminal_info:
                print(f"Terminal: {terminal_info.name}")
                print(f"Versão: {terminal_info.build}")
                print(f"Caminho: {terminal_info.path}")
            
            account_info = mt5.account_info()
            if account_info:
                print(f"Conta: {account_info.login}")
                print(f"Servidor: {account_info.server}")
                print(f"Saldo: ${account_info.balance:,.2f}")
                print(f"Equity: ${account_info.equity:,.2f}")
            
            # Testar símbolos
            symbols = mt5.symbols_get()
            if symbols:
                print(f"Símbolos disponíveis: {len(symbols)}")
                
                # Testar EURUSD
                eurusd_tick = mt5.symbol_info_tick("EURUSD")
                if eurusd_tick:
                    print(f"EURUSD - Bid: {eurusd_tick.bid}, Ask: {eurusd_tick.ask}")
            
            mt5.shutdown()
            return True
            
        else:
            error = mt5.last_error()
            print(f"✗ Falha na inicialização: {error}")
            return False
            
    except Exception as e:
        print(f"✗ Erro na inicialização: {e}")
        return False

def test_mt5_with_path():
    """Teste com caminho específico do MT5"""
    print("\n[2/4] Testando com caminhos específicos...")
    
    # Caminhos comuns do MT5
    paths = [
        "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
        "C:\\Program Files (x86)\\MetaTrader 5\\terminal64.exe",
        "C:\\Users\\%USERNAME%\\AppData\\Roaming\\MetaQuotes\\Terminal\\*\\terminal64.exe"
    ]
    
    for path in paths:
        try:
            print(f"Tentando: {path}")
            result = mt5.initialize(path=path)
            if result:
                print(f"✓ Sucesso com: {path}")
                mt5.shutdown()
                return True
            else:
                error = mt5.last_error()
                print(f"✗ Falha: {error}")
        except Exception as e:
            print(f"✗ Erro: {e}")
    
    return False

def test_mt5_login():
    """Teste de login específico"""
    print("\n[3/4] Testando login específico...")
    
    try:
        # Tentar com login específico
        result = mt5.initialize(
            login=int(os.getenv("MT5_LOGIN", "0")),
            server="MetaQuotes-Demo",
            password=""  # Senha vazia para conta demo
        )
        
        if result:
            print("✓ Login específico bem-sucedido")
            account_info = mt5.account_info()
            if account_info:
                print(f"Conta logada: {account_info.login}")
            mt5.shutdown()
            return True
        else:
            error = mt5.last_error()
            print(f"✗ Falha no login: {error}")
            return False
            
    except Exception as e:
        print(f"✗ Erro no login: {e}")
        return False

def check_processes():
    """Verificar processos MT5 em execução"""
    print("\n[4/4] Verificando processos MT5...")
    
    import subprocess
    try:
        # Verificar processos terminal64.exe
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq terminal64.exe'],
            capture_output=True,
            text=True
        )
        
        if 'terminal64.exe' in result.stdout:
            print("✓ Processo MT5 encontrado em execução")
            print(result.stdout)
            return True
        else:
            print("✗ Nenhum processo MT5 encontrado")
            return False
            
    except Exception as e:
        print(f"✗ Erro ao verificar processos: {e}")
        return False

if __name__ == "__main__":
    print("DIAGNÓSTICO COMPLETO MT5")
    print("=" * 40)
    
    # Executar todos os testes
    test1 = test_mt5_basic()
    test2 = test_mt5_with_path() if not test1 else True
    test3 = test_mt5_login() if not test1 else True
    test4 = check_processes()
    
    print("\n=== RESUMO ===")
    print(f"Teste básico: {'✓' if test1 else '✗'}")
    print(f"Teste com path: {'✓' if test2 else '✗'}")
    print(f"Teste de login: {'✓' if test3 else '✗'}")
    print(f"Processos MT5: {'✓' if test4 else '✗'}")
    
    if test1 or test2 or test3:
        print("\n✓ MT5 está funcionando")
        sys.exit(0)
    else:
        print("\n✗ Problemas com MT5 detectados")
        print("\nSoluções sugeridas:")
        print("1. Reiniciar o MetaTrader 5")
        print("2. Verificar se está logado na conta")
        print("3. Verificar conexão com internet")
        print("4. Reinstalar MT5 se necessário")
        sys.exit(1)