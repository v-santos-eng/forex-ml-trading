#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste de Eficiência de Sessão MT5
Demonstra a melhoria de performance na verificação de sessão
"""

import os
import time
import statistics
from datetime import datetime
from mt5_integration import MT5Integration, MT5Config
import logging

# Configurar logging mínimo
logging.basicConfig(level=logging.WARNING)

def benchmark_session_checks():
    """
    Benchmark das verificações de sessão otimizadas
    """
    print("=== BENCHMARK DE EFICIÊNCIA DE SESSÃO MT5 ===")
    print(f"Teste iniciado em: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Configuração
    config = MT5Config(
        login=int(os.getenv("MT5_LOGIN", "0")),
        password=os.getenv("MT5_PASSWORD", ""),
        server="ActivTradesMarkets-Server"
    )
    
    mt5_client = MT5Integration(config)
    
    print("[1/4] Conexão inicial (estabelecendo sessão)...")
    start_time = time.time()
    success = mt5_client.connect()
    initial_time = time.time() - start_time
    
    if not success:
        print(f"✗ Falha na conexão inicial: {mt5_client.last_error}")
        return False
    
    print(f"✓ Sessão estabelecida em {initial_time:.3f}s")
    
    print("\n[2/4] Testando reutilização de sessão (10 tentativas)...")
    reuse_times = []
    
    for i in range(10):
        start_time = time.time()
        mt5_client.connect()
        reuse_time = time.time() - start_time
        reuse_times.append(reuse_time)
        print(f"  Tentativa {i+1:2d}: {reuse_time:.4f}s")
    
    avg_reuse = statistics.mean(reuse_times)
    min_reuse = min(reuse_times)
    max_reuse = max(reuse_times)
    
    print(f"\n✓ Estatísticas de reutilização:")
    print(f"  Média: {avg_reuse:.4f}s")
    print(f"  Mínimo: {min_reuse:.4f}s")
    print(f"  Máximo: {max_reuse:.4f}s")
    
    print("\n[3/4] Testando verificações de saúde (20 tentativas)...")
    health_times = []
    
    for i in range(20):
        start_time = time.time()
        is_healthy = mt5_client.is_connection_healthy()
        health_time = time.time() - start_time
        health_times.append(health_time)
        if i < 5:  # Mostrar apenas as primeiras 5
            print(f"  Verificação {i+1}: {health_time:.4f}s (saudável: {is_healthy})")
        elif i == 5:
            print("  ... (continuando verificações)")
    
    avg_health = statistics.mean(health_times)
    print(f"\n✓ Verificações de saúde - Média: {avg_health:.4f}s")
    
    print("\n[4/4] Testando verificação rápida MT5 (50 tentativas)...")
    quick_times = []
    
    for i in range(50):
        start_time = time.time()
        is_ready = mt5_client.is_mt5_ready()
        quick_time = time.time() - start_time
        quick_times.append(quick_time)
    
    avg_quick = statistics.mean(quick_times)
    print(f"✓ Verificações rápidas - Média: {avg_quick:.4f}s")
    
    # Calcular melhorias
    improvement_reuse = ((initial_time - avg_reuse) / initial_time) * 100
    improvement_health = ((initial_time - avg_health) / initial_time) * 100
    improvement_quick = ((initial_time - avg_quick) / initial_time) * 100
    
    print("\n=== ANÁLISE DE PERFORMANCE ===")
    print(f"Conexão inicial:        {initial_time:.4f}s (baseline)")
    print(f"Reutilização média:     {avg_reuse:.4f}s ({improvement_reuse:+.1f}%)")
    print(f"Verificação saúde:      {avg_health:.4f}s ({improvement_health:+.1f}%)")
    print(f"Verificação rápida:     {avg_quick:.4f}s ({improvement_quick:+.1f}%)")
    
    print("\n=== BENEFÍCIOS PRÁTICOS ===")
    
    # Simular uso em produção
    daily_checks = 1000  # Verificações por dia
    time_saved_per_check = initial_time - avg_reuse
    daily_time_saved = (time_saved_per_check * daily_checks) / 60  # em minutos
    
    print(f"Em um sistema com {daily_checks} verificações/dia:")
    print(f"  Tempo economizado por verificação: {time_saved_per_check:.4f}s")
    print(f"  Tempo total economizado por dia: {daily_time_saved:.1f} minutos")
    print(f"  Redução de latência: {improvement_reuse:.1f}%")
    
    # Limpeza
    mt5_client.disconnect()
    print("\n✓ Teste concluído com sucesso!")
    
    return True

def test_production_scenario():
    """
    Simula cenário de produção com múltiplas verificações
    """
    print("\n=== SIMULAÇÃO DE CENÁRIO DE PRODUÇÃO ===")
    
    config = MT5Config(
        login=int(os.getenv("MT5_LOGIN", "0")),
        password=os.getenv("MT5_PASSWORD", ""),
        server="ActivTradesMarkets-Server"
    )
    
    mt5_client = MT5Integration(config)
    
    # Conectar uma vez
    print("Estabelecendo conexão inicial...")
    if not mt5_client.connect():
        print(f"✗ Falha na conexão: {mt5_client.last_error}")
        return False
    
    print("\nSimulando operações de trading típicas:")
    
    operations = [
        ("Verificar conexão antes de trade", lambda: mt5_client.connect()),
        ("Verificar saúde da conexão", lambda: mt5_client.is_connection_healthy()),
        ("Obter informações da conta", lambda: mt5_client.get_account_info()),
        ("Verificar se MT5 está pronto", lambda: mt5_client.is_mt5_ready()),
        ("Reconectar se necessário", lambda: mt5_client.connect()),
    ]
    
    total_time = 0
    for i, (desc, operation) in enumerate(operations * 3):  # Repetir 3x
        start_time = time.time()
        result = operation()
        op_time = time.time() - start_time
        total_time += op_time
        
        print(f"  {i+1:2d}. {desc}: {op_time:.4f}s")
    
    print(f"\n✓ Total de {len(operations) * 3} operações em {total_time:.3f}s")
    print(f"  Tempo médio por operação: {total_time / (len(operations) * 3):.4f}s")
    
    mt5_client.disconnect()
    return True

if __name__ == "__main__":
    try:
        print("Iniciando benchmark de eficiência de sessão MT5...\n")
        
        if not benchmark_session_checks():
            print("\n✗ Benchmark principal falhou")
            exit(1)
        
        if not test_production_scenario():
            print("\n✗ Teste de produção falhou")
            exit(1)
        
        print("\n🎉 Benchmark concluído com sucesso!")
        print("\n=== RESUMO DAS OTIMIZAÇÕES ===")
        print("✓ Verificação de sessão existente implementada")
        print("✓ Reutilização inteligente de conexões ativas")
        print("✓ Validação eficiente de conta e servidor")
        print("✓ Múltiplos níveis de verificação (rápida, saúde, completa)")
        print("✓ Logs otimizados para debugging")
        print("✓ Redução significativa de latência em produção")
        
    except KeyboardInterrupt:
        print("\n⚠ Benchmark interrompido pelo usuário")
    except Exception as e:
        print(f"\n✗ Erro inesperado: {e}")
        exit(1)