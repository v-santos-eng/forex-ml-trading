#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para simular decisões com dados de risco avançado
Para demonstrar o funcionamento do sistema de monitoramento
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
import random
from typing import Dict, Any

def generate_advanced_risk_data() -> Dict[str, Any]:
    """Gera dados simulados de risco avançado"""
    volatility_regimes = ['low', 'medium', 'high', 'extreme']
    
    return {
        'atr_value': round(random.uniform(0.0001, 0.0050), 5),
        'sl_multiplier': round(random.uniform(1.5, 3.0), 2),
        'tp_multiplier': round(random.uniform(2.0, 4.0), 2),
        'risk_reward_ratio': round(random.uniform(1.5, 3.5), 2),
        'volatility_regime': random.choice(volatility_regimes),
        'gap_protection_active': random.choice([True, False]),
        'trailing_stop_active': random.choice([True, False]),
        'technical_levels_detected': random.randint(0, 5),
        'scaled_tp_levels': random.randint(1, 3),
        'confidence_score': round(random.uniform(0.6, 0.95), 2)
    }

def simulate_decisions(num_decisions: int = 20):
    """Simula decisões com dados de risco avançado"""
    print(f"🎯 Simulando {num_decisions} decisões com risco avançado...")
    
    try:
        conn = sqlite3.connect('detailed_decisions.db')
        cursor = conn.cursor()
        
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        actions = ['BUY', 'SELL', 'HOLD']
        
        for i in range(num_decisions):
            decision_id = str(uuid.uuid4())
            timestamp = (datetime.now() - timedelta(hours=random.randint(0, 23))).isoformat()
            symbol = random.choice(symbols)
            action = random.choice(actions)
            confidence = round(random.uniform(0.6, 0.95), 2)
            
            # Gerar dados de risco avançado
            advanced_risk_data = generate_advanced_risk_data()
            
            # Dados básicos simulados
            reasoning_factors = json.dumps({
                'technical_score': round(random.uniform(0.5, 0.9), 2),
                'fundamental_score': round(random.uniform(0.4, 0.8), 2),
                'sentiment_score': round(random.uniform(0.3, 0.7), 2)
            })
            
            risk_assessment = json.dumps({
                'risk_level': random.choice(['low', 'medium', 'high']),
                'max_drawdown': round(random.uniform(0.01, 0.05), 3),
                'position_size': round(random.uniform(0.1, 1.0), 2)
            })
            
            market_data = json.dumps({
                'price': round(random.uniform(1.0500, 1.2000), 5),
                'spread': round(random.uniform(0.0001, 0.0005), 5),
                'volume': random.randint(1000, 10000)
            })
            
            cursor.execute('''
                INSERT INTO detailed_decisions (
                    id, timestamp, symbol, action, confidence,
                    reasoning_factors, risk_assessment, market_data,
                    technical_analysis, ml_predictions, bot_action,
                    advanced_risk_data, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                decision_id,
                timestamp,
                symbol,
                action,
                confidence,
                reasoning_factors,
                risk_assessment,
                market_data,
                json.dumps({'rsi': random.randint(20, 80), 'macd': round(random.uniform(-0.01, 0.01), 4)}),
                json.dumps({'prediction': round(random.uniform(0.4, 0.9), 2), 'model': 'ensemble'}),
                json.dumps({'executed': True, 'execution_time': round(random.uniform(0.1, 0.5), 3)}),
                json.dumps(advanced_risk_data),
                datetime.now().isoformat()
            ))
            
            if (i + 1) % 5 == 0:
                print(f"   ✅ {i + 1}/{num_decisions} decisões simuladas")
        
        conn.commit()
        conn.close()
        
        print(f"\n🎉 Simulação concluída! {num_decisions} decisões adicionadas com dados de risco avançado.")
        
    except Exception as e:
        print(f"❌ Erro na simulação: {e}")

def show_simulation_summary():
    """Mostra resumo da simulação"""
    try:
        conn = sqlite3.connect('detailed_decisions.db')
        cursor = conn.cursor()
        
        # Total de decisões
        cursor.execute("SELECT COUNT(*) FROM detailed_decisions")
        total = cursor.fetchone()[0]
        
        # Decisões com risco avançado
        cursor.execute("SELECT COUNT(*) FROM detailed_decisions WHERE advanced_risk_data IS NOT NULL AND advanced_risk_data != ''")
        with_advanced = cursor.fetchone()[0]
        
        # Decisões recentes (últimas 2 horas)
        recent_time = (datetime.now() - timedelta(hours=2)).isoformat()
        cursor.execute("SELECT COUNT(*) FROM detailed_decisions WHERE datetime(timestamp) >= datetime(?) AND advanced_risk_data IS NOT NULL AND advanced_risk_data != ''", [recent_time])
        recent_advanced = cursor.fetchone()[0]
        
        conn.close()
        
        print("\n📊 Resumo do Banco de Dados:")
        print(f"   • Total de Decisões: {total}")
        print(f"   • Com Risco Avançado: {with_advanced}")
        print(f"   • Taxa de Adoção: {(with_advanced/total*100):.1f}%")
        print(f"   • Recentes com Risco Avançado (2h): {recent_advanced}")
        
    except Exception as e:
        print(f"❌ Erro ao gerar resumo: {e}")

def main():
    """Função principal"""
    print("🚀 Simulador de Decisões com Risco Avançado")
    print("=" * 50)
    
    # Mostrar estado atual
    show_simulation_summary()
    
    # Simular decisões
    simulate_decisions(20)
    
    # Mostrar estado após simulação
    show_simulation_summary()
    
    print("\n✨ Agora você pode executar o monitor de performance para ver os resultados!")
    print("   Comando: python performance_monitor_advanced.py")

if __name__ == "__main__":
    main()