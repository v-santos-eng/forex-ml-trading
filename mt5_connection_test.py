#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste de Conexão MT5 - Diagnóstico de Interface
Testa conexão com credenciais específicas e diagnostica problemas de interface
"""

import os
import MetaTrader5 as mt5
import time
import sys
from datetime import datetime

def test_mt5_connection():
    """
    Testa conexão MT5 com diagnóstico detalhado
    """
    print("=== TESTE DE CONEXÃO MT5 ===")
    print(f"Hora: {datetime.now()}")
    print()
    
    # Credenciais fornecidas
    login = int(os.getenv("MT5_LOGIN", "0"))
    password = os.getenv("MT5_PASSWORD", "")
    server = "ActivTradesMarkets-Server"
    
    print(f"Login: {login}")
    print(f"Servidor: {server}")
    print(f"Senha: {'*' * len(password)}")
    print()
    
    # Verificar se MT5 está instalado
    print("[1/6] Verificando instalação do MT5...")
    try:
        version = mt5.version()
        if version:
            print(f"✓ MT5 versão: {version}")
        else:
            print("✗ MT5 não encontrado ou não instalado")
            return False
    except Exception as e:
        print(f"✗ Erro ao verificar MT5: {e}")
        return False
    
    # Inicializar MT5
    print("\n[2/6] Inicializando MT5...")
    try:
        if not mt5.initialize():
            error = mt5.last_error()
            print(f"✗ Falha ao inicializar MT5: {error}")
            return False
        print("✓ MT5 inicializado com sucesso")
    except Exception as e:
        print(f"✗ Erro na inicialização: {e}")
        return False
    
    # Verificar terminal info
    print("\n[3/6] Verificando informações do terminal...")
    try:
        terminal_info = mt5.terminal_info()
        if terminal_info:
            print(f"✓ Terminal: {terminal_info.name}")
            print(f"  Caminho: {terminal_info.path}")
            print(f"  Empresa: {terminal_info.company}")
            print(f"  Conectado: {terminal_info.connected}")
            print(f"  Trade permitido: {terminal_info.trade_allowed}")
        else:
            print("⚠ Informações do terminal não disponíveis")
    except Exception as e:
        print(f"⚠ Erro ao obter info do terminal: {e}")
    
    # Tentar login
    print("\n[4/6] Tentando login...")
    try:
        result = mt5.login(login, password=password, server=server)
        if result:
            print("✓ Login realizado com sucesso")
        else:
            error = mt5.last_error()
            print(f"✗ Falha no login: {error}")
            
            # Tentar diagnóstico adicional
            print("\n[DIAGNÓSTICO] Tentando soluções alternativas...")
            
            # Verificar se o servidor está na lista
            print("Verificando servidores disponíveis...")
            try:
                # Aguardar um pouco para o terminal carregar
                time.sleep(3)
                
                # Tentar novamente
                print("Tentativa 2 de login...")
                result = mt5.login(login, password=password, server=server)
                if result:
                    print("✓ Login realizado na segunda tentativa")
                else:
                    error = mt5.last_error()
                    print(f"✗ Falha na segunda tentativa: {error}")
                    
                    # Sugestões de solução
                    print("\n[SOLUÇÕES SUGERIDAS]")
                    print("1. Verificar se o MetaTrader 5 está completamente carregado")
                    print("2. Tentar fazer login manual primeiro no terminal")
                    print("3. Verificar se o servidor está correto")
                    print("4. Verificar conectividade com a internet")
                    print("5. Reiniciar o MetaTrader 5")
                    
            except Exception as e:
                print(f"Erro no diagnóstico: {e}")
    except Exception as e:
        print(f"✗ Erro durante login: {e}")
        return False
    
    # Verificar conta
    print("\n[5/6] Verificando informações da conta...")
    try:
        account_info = mt5.account_info()
        if account_info:
            print(f"✓ Conta: {account_info.login}")
            print(f"  Nome: {account_info.name}")
            print(f"  Servidor: {account_info.server}")
            print(f"  Moeda: {account_info.currency}")
            print(f"  Saldo: {account_info.balance}")
            print(f"  Equity: {account_info.equity}")
            print(f"  Margem: {account_info.margin}")
        else:
            print("⚠ Informações da conta não disponíveis")
    except Exception as e:
        print(f"⚠ Erro ao obter info da conta: {e}")
    
    # Teste de símbolos
    print("\n[6/6] Testando acesso a símbolos...")
    try:
        symbols = mt5.symbols_get()
        if symbols:
            print(f"✓ {len(symbols)} símbolos disponíveis")
            # Mostrar alguns símbolos principais
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
            for pair in major_pairs:
                symbol_info = mt5.symbol_info(pair)
                if symbol_info:
                    print(f"  {pair}: Disponível")
        else:
            print("⚠ Nenhum símbolo disponível")
    except Exception as e:
        print(f"⚠ Erro ao obter símbolos: {e}")
    
    # Finalizar
    print("\n[FINALIZANDO] Desconectando...")
    mt5.shutdown()
    print("✓ Desconectado do MT5")
    
    print("\n=== TESTE CONCLUÍDO ===")
    return True

def check_mt5_process():
    """
    Verifica se o processo MT5 está rodando
    """
    import psutil
    
    print("\n=== VERIFICAÇÃO DE PROCESSOS MT5 ===")
    mt5_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            if proc.info['name'] and 'terminal' in proc.info['name'].lower():
                mt5_processes.append(proc.info)
            elif proc.info['exe'] and 'metatrader' in proc.info['exe'].lower():
                mt5_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if mt5_processes:
        print(f"✓ Encontrados {len(mt5_processes)} processos MT5:")
        for proc in mt5_processes:
            print(f"  PID: {proc['pid']}, Nome: {proc['name']}")
            if proc['exe']:
                print(f"    Caminho: {proc['exe']}")
    else:
        print("⚠ Nenhum processo MT5 encontrado")
        print("Sugestão: Iniciar o MetaTrader 5 manualmente")

if __name__ == "__main__":
    print("DIAGNÓSTICO DE CONEXÃO MT5")
    print("=" * 50)
    
    # Verificar processos primeiro
    try:
        check_mt5_process()
    except ImportError:
        print("⚠ psutil não disponível - pulando verificação de processos")
    except Exception as e:
        print(f"⚠ Erro na verificação de processos: {e}")
    
    # Aguardar um pouco
    print("\nAguardando 2 segundos...")
    time.sleep(2)
    
    # Executar teste de conexão
    try:
        test_mt5_connection()
    except KeyboardInterrupt:
        print("\n\nTeste interrompido pelo usuário")
    except Exception as e:
        print(f"\nErro durante o teste: {e}")
    
    print("\nPressione Enter para sair...")
    input()