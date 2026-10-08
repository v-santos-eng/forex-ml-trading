#!/usr/bin/env python3
"""
Demo de Análise Automática - Sistema de Trading Forex ML
Executa análise automática dos principais pares de moedas
"""

import sys
import os
from pathlib import Path

# Adicionar diretórios ao path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from start_trading import SimpleTradingSystem
import time

def run_demo_analysis():
    """Executar análise demonstrativa automática"""
    print("\n" + "="*60)
    print("🎯 DEMO: ANÁLISE AUTOMÁTICA DE FOREX")
    print("="*60)
    
    # Inicializar sistema
    system = SimpleTradingSystem()
    
    # Lista de pares para análise
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    results = {}
    
    print("\n🔄 Iniciando análise automática...")
    
    for i, pair in enumerate(pairs, 1):
        print(f"\n📊 [{i}/{len(pairs)}] Processando {pair}...")
        
        try:
            # Executar análise
            result = system.analyze_pair(pair)
            results[pair] = result
            
            # Pequena pausa entre análises
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Erro ao analisar {pair}: {e}")
            results[pair] = {'error': str(e)}
    
    # Resumo final
    print("\n" + "="*60)
    print("📈 RESUMO DA ANÁLISE")
    print("="*60)
    
    for pair, result in results.items():
        if 'error' not in result:
            print(f"\n💰 {pair}:")
            print(f"   Preço: {result['price']:.5f}")
            print(f"   RSI: {result['rsi']:.1f}")
            print(f"   Recomendação: {result['recommendation']}")
            print(f"   Confiança: {result['confidence']:.1%}")
        else:
            print(f"\n❌ {pair}: {result['error']}")
    
    # Identificar melhores oportunidades
    print("\n" + "="*60)
    print("🎯 MELHORES OPORTUNIDADES")
    print("="*60)
    
    opportunities = []
    for pair, result in results.items():
        if 'error' not in result and result['recommendation'] != 'HOLD':
            opportunities.append({
                'pair': pair,
                'action': result['recommendation'],
                'confidence': result['confidence'],
                'rsi': result['rsi']
            })
    
    # Ordenar por confiança
    opportunities.sort(key=lambda x: x['confidence'], reverse=True)
    
    if opportunities:
        for i, opp in enumerate(opportunities[:3], 1):
            print(f"\n{i}. {opp['pair']} - {opp['action']}")
            print(f"   Confiança: {opp['confidence']:.1%}")
            print(f"   RSI: {opp['rsi']:.1f}")
    else:
        print("\n📊 Nenhuma oportunidade clara identificada no momento.")
        print("💡 Recomendação: Aguardar melhores condições de mercado.")
    
    print("\n" + "="*60)
    print("✅ ANÁLISE CONCLUÍDA")
    print("="*60)
    print("\n💡 Para análise interativa, execute: python start_trading.py")
    
    return results

if __name__ == "__main__":
    try:
        results = run_demo_analysis()
        print("\n🎉 Demo executado com sucesso!")
    except Exception as e:
        print(f"\n❌ Erro na execução do demo: {e}")