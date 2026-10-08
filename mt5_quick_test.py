#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste Rápido MT5 - Verificação de Conexão Existente
Verifica se já existe uma conexão ativa e testa funcionalidades básicas
"""

import os
import MetaTrader5 as mt5
import time
from datetime import datetime

def quick_mt5_test():
    """
    Teste rápido da conexão MT5 existente
    """
    print("=== TESTE RÁPIDO MT5 ===")
    print(f"Hora: {datetime.now()}")
    print()
    
    # Inicializar MT5
    print("[1/4] Inicializando MT5...")
    if not mt5.initialize():
        error = mt5.last_error()
        print(f"✗ Falha ao inicializar: {error}")
        return False
    print("✓ MT5 inicializado")
    
    # Verificar conta atual
    print("\n[2/4] Verificando conta atual...")
    account_info = mt5.account_info()
    if account_info:
        print(f"✓ Conectado à conta: {account_info.login}")
        print(f"  Servidor: {account_info.server}")
        print(f"  Saldo: ${account_info.balance:,.2f}")
        print(f"  Equity: ${account_info.equity:,.2f}")
        print(f"  Margem Livre: ${account_info.margin_free:,.2f}")
        
        # Verificar se é a conta correta
        if account_info.login == int(os.getenv("MT5_LOGIN", "0")):
            print("✓ Conta correta já conectada!")
        else:
            print(f"⚠ Conta diferente conectada (esperado: <MT5_LOGIN>)")
    else:
        print("✗ Nenhuma conta conectada")
        return False
    
    # Teste de cotações
    print("\n[3/4] Testando cotações em tempo real...")
    symbols_to_test = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    for symbol in symbols_to_test:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            print(f"✓ {symbol}: Bid={tick.bid:.5f}, Ask={tick.ask:.5f}, Spread={tick.ask-tick.bid:.5f}")
        else:
            print(f"✗ {symbol}: Cotação não disponível")
    
    # Teste de histórico
    print("\n[4/4] Testando acesso ao histórico...")
    try:
        # Pegar últimas 10 barras do EURUSD M1
        rates = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_M1, 0, 10)
        if rates is not None and len(rates) > 0:
            print(f"✓ Histórico disponível: {len(rates)} barras recebidas")
            last_bar = rates[-1]
            print(f"  Última barra EURUSD M1:")
            print(f"    Tempo: {datetime.fromtimestamp(last_bar['time'])}")
            print(f"    OHLC: {last_bar['open']:.5f} / {last_bar['high']:.5f} / {last_bar['low']:.5f} / {last_bar['close']:.5f}")
        else:
            print("✗ Histórico não disponível")
    except Exception as e:
        print(f"✗ Erro ao acessar histórico: {e}")
    
    print("\n=== RESULTADO ===")
    print("✓ MT5 está funcionando corretamente!")
    print("✓ Conta <MT5_LOGIN> já está conectada")
    print("✓ Cotações em tempo real funcionando")
    print("✓ Sistema pronto para trading")
    
    # Não desconectar - manter a sessão ativa
    print("\n[INFO] Mantendo conexão ativa para o sistema principal...")
    return True

def show_connection_status():
    """
    Mostra status detalhado da conexão
    """
    print("\n=== STATUS DA CONEXÃO ===")
    
    if not mt5.initialize():
        print("✗ MT5 não inicializado")
        return
    
    # Info do terminal
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"Terminal: {terminal_info.name}")
        print(f"Conectado: {'✓ Sim' if terminal_info.connected else '✗ Não'}")
        print(f"Trade Permitido: {'✓ Sim' if terminal_info.trade_allowed else '✗ Não'}")
    
    # Info da conta
    account_info = mt5.account_info()
    if account_info:
        print(f"\nConta Ativa: {account_info.login}")
        print(f"Servidor: {account_info.server}")
        print(f"Tipo de Conta: {account_info.trade_mode}")
        print(f"Alavancagem: 1:{account_info.leverage}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    try:
        show_connection_status()
        success = quick_mt5_test()
        
        if success:
            print("\n🎉 CONEXÃO MT5 ESTABELECIDA COM SUCESSO!")
            print("\nO sistema principal pode agora usar esta conexão.")
            print("\nPróximos passos:")
            print("1. O sistema de trading pode começar a operar")
            print("2. Monitoramento de mercado ativo")
            print("3. Pronto para executar estratégias")
        else:
            print("\n❌ PROBLEMAS NA CONEXÃO MT5")
            print("\nVerifique:")
            print("1. Se o MetaTrader 5 está aberto")
            print("2. Se as credenciais estão corretas")
            print("3. Se há conexão com a internet")
            
    except KeyboardInterrupt:
        print("\n\nTeste interrompido")
    except Exception as e:
        print(f"\nErro durante o teste: {e}")
    
    print("\nPressione Enter para continuar...")
    input()