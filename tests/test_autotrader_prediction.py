#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste específico do método predict_direction no contexto do AutoTrader
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_predictor_local import MLForexPredictorLocal
import MetaTrader5 as mt5
from datetime import datetime

def test_autotrader_prediction():
    print("🧪 TESTE DE PREDIÇÃO NO CONTEXTO DO AUTOTRADER")
    print("=" * 60)
    
    # Conectar ao MT5
    if not mt5.initialize():
        print("❌ Falha na inicialização do MT5")
        return
    
    try:
        # Inicializar preditor
        predictor = MLForexPredictorLocal()
        
        # Símbolos para testar
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        
        for symbol in symbols:
            print(f"\n📊 Testando {symbol}...")
            
            # Verificar se símbolo está disponível
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Símbolo {symbol} não disponível")
                continue
                
            # Forçar uso de amostra pequena para teste rápido
            original_prepare_data = predictor.prepare_data
            
            def prepare_data_fast(pair):
                X, y = original_prepare_data(pair)
                if X is not None and y is not None:
                    # Usar apenas 500 amostras para teste ultra-rápido
                    sample_size = min(500, len(X))
                    print(f"   Usando {sample_size} amostras de {len(X)} disponíveis")
                    return X.iloc[:sample_size], y.iloc[:sample_size]
                return X, y
            
            predictor.prepare_data = prepare_data_fast
            
            # Testar predição
            start_time = datetime.now()
            result = predictor.predict_direction(symbol)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            if result:
                direction, confidence, prob_up, prob_down = result
                print(f"   ✅ SUCESSO em {duration:.1f}s:")
                print(f"      Direção: {direction}")
                print(f"      Confiança: {confidence:.3f}")
                print(f"      Prob. Alta: {prob_up:.3f}")
                print(f"      Prob. Baixa: {prob_down:.3f}")
                
                # Verificar se atende critério de confiança mínima (65%)
                min_confidence = 0.65
                meets_criteria = confidence >= min_confidence
                print(f"      Atende critério (≥{min_confidence:.0%}): {'✅' if meets_criteria else '❌'}")
                
            else:
                print(f"   ❌ FALHA em {duration:.1f}s")
            
            # Restaurar método original
            predictor.prepare_data = original_prepare_data
            
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        mt5.shutdown()
        print(f"\n🔚 Teste concluído")

if __name__ == "__main__":
    test_autotrader_prediction()