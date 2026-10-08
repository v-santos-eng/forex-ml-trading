#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste rápido do método predict_direction corrigido
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_predictor_local import MLForexPredictorLocal

def test_quick_prediction():
    print("🧪 TESTE RÁPIDO DE PREDIÇÃO")
    print("=" * 50)
    
    try:
        # Inicializar preditor
        predictor = MLForexPredictorLocal()
        
        # Testar predição para EURUSD
        pair = "EURUSD"
        print(f"\n📊 Testando predição para {pair}...")
        
        # Forçar uso de amostra pequena modificando temporariamente o método
        original_prepare_data = predictor.prepare_data
        
        def prepare_data_small(pair):
            X, y = original_prepare_data(pair)
            if X is not None and y is not None:
                # Usar apenas 1000 amostras para teste rápido
                sample_size = min(1000, len(X))
                return X.iloc[:sample_size], y.iloc[:sample_size]
            return X, y
        
        predictor.prepare_data = prepare_data_small
        
        # Fazer predição
        result = predictor.predict_direction(pair)
        
        if result:
            direction, confidence, prob_up, prob_down = result
            print(f"\n✅ PREDIÇÃO BEM-SUCEDIDA:")
            print(f"   Direção: {direction}")
            print(f"   Confiança: {confidence:.3f}")
            print(f"   Prob. Alta: {prob_up:.3f}")
            print(f"   Prob. Baixa: {prob_down:.3f}")
            print(f"   Tipo do resultado: {type(result)}")
            print(f"   Número de valores: {len(result)}")
            
            # Verificar se pode ser desempacotado corretamente
            try:
                dir_test, conf_test, up_test, down_test = result
                print(f"\n✅ DESEMPACOTAMENTO OK:")
                print(f"   {dir_test}, {conf_test:.3f}, {up_test:.3f}, {down_test:.3f}")
            except Exception as e:
                print(f"\n❌ ERRO NO DESEMPACOTAMENTO: {e}")
                
        else:
            print(f"\n❌ PREDIÇÃO FALHOU: resultado é None")
            
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quick_prediction()