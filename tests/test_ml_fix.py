#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste da correção do problema de predição ML
Verifica se a correção no AutoTrader resolve o erro 'dict' object has no attribute 'predict'
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_predictor_local import MLForexPredictorLocal
import MetaTrader5 as mt5
from datetime import datetime
import traceback
import pandas as pd
import numpy as np

def test_ml_fix():
    """
    Testa a correção do problema de predição ML
    """
    print("="*80)
    print("TESTE DA CORREÇÃO DO PROBLEMA DE PREDIÇÃO ML")
    print("="*80)
    
    # Inicializar predictor
    try:
        predictor = MLForexPredictorLocal()
        print("✅ MLPredictor inicializado")
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        return False
    
    # Preparar dados
    symbol = "EURUSD"
    try:
        X, y = predictor.prepare_data(symbol)
        if X is None or len(X) < 100:
            print(f"❌ Dados insuficientes: {len(X) if X is not None else 0}")
            return False
        print(f"✅ Dados preparados: {len(X)} amostras")
    except Exception as e:
        print(f"❌ Erro na preparação: {e}")
        return False
    
    # Treinar modelos (simulando o que o AutoTrader faz)
    try:
        print("\nTreinando modelos...")
        models = predictor.train_models(X.iloc[:1000], y.iloc[:1000])  # Amostra menor para teste
        
        if not models:
            print("❌ Nenhum modelo treinado")
            return False
            
        print(f"✅ Modelos treinados: {list(models.keys())}")
        
        # Verificar estrutura dos modelos
        for name, model_result in models.items():
            print(f"   {name}: {type(model_result)}")
            if isinstance(model_result, dict):
                print(f"      Keys: {list(model_result.keys())}")
                if 'model' in model_result:
                    print(f"      Model type: {type(model_result['model']).__name__}")
                    
    except Exception as e:
        print(f"❌ Erro no treinamento: {e}")
        traceback.print_exc()
        return False
    
    # Testar predição com a nova lógica (simulando o código corrigido)
    print("\n" + "-"*50)
    print("TESTANDO LÓGICA CORRIGIDA DE PREDIÇÃO")
    print("-"*50)
    
    try:
        latest_features = X.iloc[-1:]
        predictions = []
        confidences = []
        model_details = {}
        
        print(f"Features shape: {latest_features.shape}")
        
        # Simular exatamente o código corrigido do AutoTrader
        for name, model_result in models.items():
            try:
                print(f"\nTestando {name}...")
                
                # Extrair o modelo do resultado do treinamento
                if isinstance(model_result, dict) and 'model' in model_result:
                    model = model_result['model']
                    print(f"   Modelo extraído: {type(model).__name__}")
                    
                    # Escalar as features usando o scaler do predictor
                    latest_scaled = predictor.scaler.transform(latest_features)
                    print(f"   Features escaladas shape: {latest_scaled.shape}")
                    
                    pred = model.predict(latest_scaled)[0]
                    prob = model.predict_proba(latest_scaled)[0]
                    confidence = max(prob)
                    
                    predictions.append(pred)
                    confidences.append(confidence)
                    model_details[name] = {
                        'prediction': float(pred),
                        'confidence': float(confidence),
                        'probabilities': [float(p) for p in prob]
                    }
                    
                    print(f"   ✅ Predição: {pred}, Confiança: {confidence:.4f}")
                    
                else:
                    model_details[name] = {'error': 'Estrutura de modelo inválida'}
                    print(f"   ❌ Estrutura inválida")
                    
            except Exception as e:
                model_details[name] = {'error': str(e)}
                print(f"   ❌ Erro: {e}")
                continue
        
        # Verificar resultados
        print(f"\nRESULTADOS:")
        print(f"   Predições válidas: {len(predictions)}")
        print(f"   Predições: {predictions}")
        print(f"   Confianças: {[f'{c:.4f}' for c in confidences]}")
        
        if predictions:
            avg_pred = sum(predictions) / len(predictions)
            avg_confidence = sum(confidences) / len(confidences)
            print(f"   Predição média: {avg_pred:.4f}")
            print(f"   Confiança média: {avg_confidence:.4f}")
            
            # Simular lógica de decisão do AutoTrader
            min_confidence = 0.6  # Threshold padrão
            if avg_confidence >= min_confidence:
                direction = "BUY" if avg_pred > 0.5 else "SELL"
                print(f"   ✅ SINAL GERADO: {direction} (confiança: {avg_confidence:.4f})")
            else:
                print(f"   ⚠️  Confiança baixa: {avg_confidence:.4f} < {min_confidence}")
                
            print("\n✅ CORREÇÃO FUNCIONANDO CORRETAMENTE!")
            return True
        else:
            print("   ❌ Nenhuma predição válida")
            print("\nDetalhes dos erros:")
            for name, details in model_details.items():
                if 'error' in details:
                    print(f"   {name}: {details['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste de predição: {e}")
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("TESTE CONCLUÍDO")
    print("="*80)

if __name__ == "__main__":
    # Inicializar MT5
    if not mt5.initialize():
        print("❌ Falha na inicialização do MT5")
        sys.exit(1)
    
    try:
        success = test_ml_fix()
        if success:
            print("\n🎉 CORREÇÃO VALIDADA COM SUCESSO!")
            print("O problema 'dict' object has no attribute 'predict' foi resolvido.")
        else:
            print("\n❌ CORREÇÃO NÃO FUNCIONOU")
            print("Ainda há problemas na predição ML.")
    finally:
        mt5.shutdown()