#!/usr/bin/env python3
"""
Teste rápido do método predict_direction do MLForexPredictorLocal
"""

import pandas as pd
from ml_predictor_local import MLForexPredictorLocal
import numpy as np

def test_predict_direction_quick():
    """
    Teste rápido da predição com dados limitados
    """
    print("=== TESTE RÁPIDO PREDICT_DIRECTION ===")
    
    # Inicializar o preditor
    predictor = MLForexPredictorLocal()
    
    pair = 'EURUSD'
    period = '1d'  # Período menor para teste rápido
    
    print(f"\n1. Preparando dados para {pair} com período {period}...")
    
    # Preparar dados com período menor
    X, y = predictor.prepare_data(pair, period=period)
    
    if X is None or y is None:
        print("❌ Falha na preparação de dados")
        return
    
    print(f"✅ Dados preparados: {len(X)} amostras, {X.shape[1]} features")
    
    # Usar apenas uma amostra pequena para treinamento rápido
    sample_size = min(1000, len(X))
    X_sample = X.iloc[:sample_size]
    y_sample = y.iloc[:sample_size]
    
    print(f"\n2. Treinando com amostra reduzida: {len(X_sample)} registros...")
    
    # Treinar modelos com amostra pequena
    predictor.train_models(X_sample, y_sample)
    print("✅ Modelos treinados")
    
    print(f"\n3. Testando predict_direction...")
    
    try:
        # Fazer predição
        result = predictor.predict_direction(pair)
        
        print(f"Resultado da predição: {type(result)}")
        print(f"Conteúdo: {result}")
        
        # Verificar o tipo de retorno
        if result is None:
            print("❌ predict_direction retornou None")
        elif isinstance(result, bool):
            print(f"❌ predict_direction retornou bool: {result}")
        elif isinstance(result, tuple):
            print(f"✅ predict_direction retornou tuple com {len(result)} elementos")
            if len(result) == 4:
                direction, confidence, prob_up, prob_down = result
                print(f"   Direção: {direction}")
                print(f"   Confiança: {confidence}")
                print(f"   Prob. Alta: {prob_up}")
                print(f"   Prob. Baixa: {prob_down}")
            else:
                print(f"   Elementos: {result}")
        else:
            print(f"❌ Tipo inesperado: {type(result)}")
            
    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        import traceback
        traceback.print_exc()

def test_predict_direction_method():
    """
    Examina diretamente o método predict_direction
    """
    print("\n=== EXAMINANDO MÉTODO PREDICT_DIRECTION ===")
    
    predictor = MLForexPredictorLocal()
    
    # Verificar se o método existe
    if hasattr(predictor, 'predict_direction'):
        print("✅ Método predict_direction existe")
        
        # Verificar o código do método
        import inspect
        source = inspect.getsource(predictor.predict_direction)
        print("\nCódigo do método:")
        print("-" * 50)
        print(source)
        print("-" * 50)
    else:
        print("❌ Método predict_direction não encontrado")

if __name__ == "__main__":
    test_predict_direction_quick()
    test_predict_direction_method()