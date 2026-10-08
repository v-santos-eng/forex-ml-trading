#!/usr/bin/env python3
"""
Teste das Melhorias do ML Forex Predictor
Demonstra Ridge Regression, KNN, XGBoost e Ensemble Methods
"""

import numpy as np
import pandas as pd
from ml_predictor_free import MLForexPredictorFree
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_synthetic_forex_data(n_samples=1000):
    """
    Cria dados sintéticos de forex para teste
    """
    print("Criando dados sintéticos para demonstração...")
    
    # Gerar série temporal sintética
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Simular preços com tendência e ruído
    trend = np.linspace(1.1000, 1.2000, n_samples)
    noise = np.random.normal(0, 0.01, n_samples)
    seasonal = 0.005 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
    
    close_prices = trend + noise + seasonal
    
    # Criar OHLV
    high_prices = close_prices + np.abs(np.random.normal(0, 0.002, n_samples))
    low_prices = close_prices - np.abs(np.random.normal(0, 0.002, n_samples))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    volume = np.random.randint(100000, 1000000, n_samples)
    
    data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)
    
    return data

def test_enhanced_models():
    """
    Testa os modelos aprimorados
    """
    print("="*70)
    print("TESTE DAS MELHORIAS DO ML FOREX PREDICTOR")
    print("Ridge Regression, KNN, XGBoost e Ensemble Methods")
    print("="*70)
    
    # Criar instância do preditor
    predictor = MLForexPredictorFree()
    
    # Criar dados sintéticos
    synthetic_data = create_synthetic_forex_data(1000)
    
    # Criar features
    print("\nCriando features...")
    features = predictor.calculate_technical_indicators(synthetic_data)
    X = predictor.create_features(synthetic_data)
    y = predictor.create_target(synthetic_data, 1)
    
    # Remover NaN
    valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X[valid_idx]
    y_clean = y[valid_idx]
    
    print(f"Dados preparados: {len(X_clean)} amostras, {len(predictor.feature_names)} features")
    
    if len(X_clean) < 50:
        print("Dados insuficientes para teste")
        return
    
    # Teste 1: Modelos sem otimização
    print("\n" + "="*50)
    print("TESTE 1: MODELOS SEM OTIMIZAÇÃO")
    print("="*50)
    
    results_basic = predictor.train_models(X_clean, y_clean, optimize_hyperparams=False)
    
    if results_basic:
        print("\nResultados (sem otimização):")
        for name, result in results_basic.items():
            print(f"{name:20}: Acurácia={result['accuracy']:.3f}, CV={result['cv_mean']:.3f}")
    
    # Teste 2: Modelos com otimização
    print("\n" + "="*50)
    print("TESTE 2: MODELOS COM OTIMIZAÇÃO")
    print("="*50)
    
    predictor_opt = MLForexPredictorFree()
    results_opt = predictor_opt.train_models(X_clean, y_clean, optimize_hyperparams=True)
    
    if results_opt:
        print("\nResultados (com otimização):")
        for name, result in results_opt.items():
            print(f"{name:20}: Acurácia={result['accuracy']:.3f}, CV={result['cv_mean']:.3f}")
    
    # Comparação de performance
    if results_basic and results_opt:
        print("\n" + "="*50)
        print("COMPARAÇÃO DE PERFORMANCE")
        print("="*50)
        
        print(f"{'Modelo':<20} {'Básico':<10} {'Otimizado':<10} {'Melhoria':<10}")
        print("-" * 50)
        
        for name in results_basic.keys():
            if name in results_opt:
                basic_score = results_basic[name]['cv_mean']
                opt_score = results_opt[name]['cv_mean']
                improvement = ((opt_score - basic_score) / basic_score) * 100
                
                print(f"{name:<20} {basic_score:<10.3f} {opt_score:<10.3f} {improvement:+6.1f}%")
    
    # Teste de predição
    print("\n" + "="*50)
    print("TESTE DE PREDIÇÃO")
    print("="*50)
    
    if results_opt:
        # Simular predição com dados mais recentes
        latest_features = X_clean.iloc[-10:]
        
        for i, (idx, row) in enumerate(latest_features.iterrows()):
            if predictor_opt.best_model is not None:
                # Fazer predição
                features_scaled = predictor_opt.scaler.transform(row.values.reshape(1, -1))
                
                if hasattr(predictor_opt.best_model, 'predict_proba'):
                    prediction = predictor_opt.best_model.predict(features_scaled)[0]
                    probability = predictor_opt.best_model.predict_proba(features_scaled)[0]
                    
                    direction = "ALTA" if prediction == 1 else "BAIXA"
                    confidence = max(probability) * 100
                    
                    print(f"Predição {i+1}: {direction} (Confiança: {confidence:.1f}%)")
                
                if i >= 4:  # Mostrar apenas 5 predições
                    break
    
    # Mostrar feature importance
    print("\n" + "="*50)
    print("IMPORTÂNCIA DAS FEATURES")
    print("="*50)
    
    if predictor_opt.best_model is not None:
        predictor_opt.feature_importance()
    
    print("\n" + "="*70)
    print("RESUMO DAS MELHORIAS IMPLEMENTADAS")
    print("="*70)
    
    print("\nModelos Adicionados:")
    print("   • Ridge Classifier (regularização L2)")
    print("   • K-Nearest Neighbors (padrões locais)")
    print("   • XGBoost (gradient boosting avançado)")
    print("   • Ensemble Voting (combinação de modelos)")
    
    print("\nFuncionalidades Novas:")
    print("   • Otimização automática de hiperparâmetros")
    print("   • Grid Search para melhores parâmetros")
    print("   • Ensemble com os 3 melhores modelos")
    print("   • Validação cruzada robusta")
    
    print("\nMelhorias de Performance:")
    print("   • Regularização para evitar overfitting")
    print("   • Múltiplos algoritmos para diversidade")
    print("   • Combinação inteligente de predições")
    print("   • Otimização automática de parâmetros")
    
    print("\n" + "="*70)
    print("TESTE CONCLUÍDO COM SUCESSO!")
    print("="*70)

if __name__ == "__main__":
    test_enhanced_models()