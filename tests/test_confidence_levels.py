#!/usr/bin/env python3
"""
Teste de Níveis de Confiança dos Modelos ML
Verifica qual confiança os modelos estão realmente atingindo
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ml_predictor_free import MLForexPredictorFree
import MetaTrader5 as mt5
import json
from datetime import datetime

def test_model_confidence():
    """Testa a confiança real dos modelos ML"""
    print("=" * 60)
    print("TESTE DE CONFIANÇA DOS MODELOS ML")
    print("=" * 60)
    
    # Inicializar MT5
    if not mt5.initialize():
        print("❌ Falha ao inicializar MT5")
        return
    
    try:
        ml_predictor = MLForexPredictorFree()
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        confidence_results = {}
        
        for symbol in symbols:
            print(f"\n📊 Testando {symbol}...")
            
            # Adicionar sufixo =X para Yahoo Finance
            yahoo_symbol = f"{symbol}=X"
            
            # Preparar dados
            X, y = ml_predictor.prepare_data(yahoo_symbol)
            
            if X is None or len(X) < 100:
                print(f"   ❌ Dados insuficientes: {len(X) if X is not None else 0} amostras")
                continue
            
            print(f"   ✅ Dados preparados: {len(X)} amostras")
            
            # Treinar modelos
            models = ml_predictor.train_models(X, y)
            
            if not models:
                print(f"   ❌ Falha no treinamento dos modelos")
                continue
            
            print(f"   ✅ Modelos treinados: {list(models.keys())}")
            
            # Testar predições
            symbol_confidences = []
            
            # Fazer múltiplas predições para obter estatísticas
            for i in range(5):
                prediction = ml_predictor.predict_direction(yahoo_symbol)
                if prediction and 'confidence' in prediction:
                    confidence = prediction['confidence'] / 100.0  # Converter para 0-1
                    symbol_confidences.append(confidence)
                    print(f"   Predição {i+1}: Confiança = {confidence:.3f} ({prediction['direction']})")
            
            if symbol_confidences:
                avg_symbol_confidence = np.mean(symbol_confidences)
                max_confidence = max(symbol_confidences)
                min_confidence = min(symbol_confidences)
                
                confidence_results[symbol] = {
                    'avg_confidence': avg_symbol_confidence,
                    'max_confidence': max_confidence,
                    'min_confidence': min_confidence,
                    'samples': len(symbol_confidences)
                }
                
                print(f"   📈 Resumo {symbol}:")
                print(f"      Confiança Média: {avg_symbol_confidence:.3f}")
                print(f"      Confiança Máxima: {max_confidence:.3f}")
                print(f"      Confiança Mínima: {min_confidence:.3f}")
            else:
                print(f"   ❌ Não foi possível obter predições para {symbol}")
        
        # Análise geral
        print("\n" + "=" * 60)
        print("ANÁLISE GERAL DE CONFIANÇA")
        print("=" * 60)
        
        if confidence_results:
            all_avg_confidences = [result['avg_confidence'] for result in confidence_results.values()]
            overall_avg = np.mean(all_avg_confidences)
            overall_max = max([result['max_confidence'] for result in confidence_results.values()])
            overall_min = min([result['min_confidence'] for result in confidence_results.values()])
            
            print(f"\n📊 Estatísticas Gerais:")
            print(f"   Confiança Média Geral: {overall_avg:.3f}")
            print(f"   Confiança Máxima Geral: {overall_max:.3f}")
            print(f"   Confiança Mínima Geral: {overall_min:.3f}")
            
            # Recomendações
            print(f"\n💡 Recomendações:")
            current_min_confidence = 0.65
            print(f"   Configuração atual: min_confidence = {current_min_confidence}")
            
            if overall_avg < current_min_confidence:
                recommended_confidence = max(0.55, overall_avg - 0.05)
                print(f"   ⚠️ PROBLEMA: Confiança média ({overall_avg:.3f}) < mínimo configurado ({current_min_confidence})")
                print(f"   ✅ SOLUÇÃO: Reduzir min_confidence para {recommended_confidence:.2f}")
                
                # Criar nova configuração
                new_config = {
                    "max_risk_per_trade": 0.02,
                    "max_daily_trades": 10,
                    "min_confidence": recommended_confidence,
                    "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
                    "ollama_url": "http://localhost:11434",
                    "prediction_interval": 300,
                    "risk_reward_ratio": 2.0,
                    "max_spread_pips": 3.0,
                    "trading_hours": {
                        "start": "08:00",
                        "end": "22:00"
                    }
                }
                
                print(f"\n🔧 Salvando nova configuração...")
                with open('auto_trader_config_optimized.json', 'w') as f:
                    json.dump(new_config, f, indent=2)
                print(f"   ✅ Nova configuração salva em 'auto_trader_config_optimized.json'")
                
            else:
                print(f"   ✅ Configuração atual está adequada")
            
            # Análise por símbolo
            print(f"\n📋 Análise por Símbolo:")
            for symbol, result in confidence_results.items():
                status = "✅ OK" if result['avg_confidence'] >= current_min_confidence else "❌ BAIXA"
                print(f"   {symbol}: {result['avg_confidence']:.3f} {status}")
        
        else:
            print("❌ Nenhum resultado de confiança obtido")
    
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    test_model_confidence()