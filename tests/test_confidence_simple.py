#!/usr/bin/env python3
"""
Teste Simples de Confiança - Verificar se min_confidence está muito alto
"""

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def create_synthetic_forex_data(n_samples=1000):
    """
    Cria dados sintéticos de forex para teste
    """
    np.random.seed(42)
    
    # Criar features sintéticas (simulando indicadores técnicos)
    features = {
        'sma_5': np.random.normal(1.1, 0.1, n_samples),
        'sma_20': np.random.normal(1.1, 0.1, n_samples),
        'rsi': np.random.uniform(20, 80, n_samples),
        'macd': np.random.normal(0, 0.01, n_samples),
        'volatility': np.random.uniform(0.001, 0.05, n_samples),
        'momentum': np.random.normal(1, 0.02, n_samples)
    }
    
    df = pd.DataFrame(features)
    
    # Criar target baseado em algumas regras simples
    # (simulando padrões que um modelo ML poderia aprender)
    target = np.zeros(n_samples)
    for i in range(n_samples):
        score = 0
        if df.iloc[i]['rsi'] < 30:  # Oversold
            score += 1
        if df.iloc[i]['rsi'] > 70:  # Overbought
            score -= 1
        if df.iloc[i]['sma_5'] > df.iloc[i]['sma_20']:  # Uptrend
            score += 1
        if df.iloc[i]['macd'] > 0:  # Positive momentum
            score += 1
        
        # Adicionar ruído
        score += np.random.normal(0, 0.5)
        
        target[i] = 1 if score > 0 else 0
    
    return df.values, target

def test_model_confidence():
    """
    Testa a confiança de modelos ML com dados sintéticos
    """
    print("="*60)
    print("TESTE DE CONFIANÇA COM DADOS SINTÉTICOS")
    print("="*60)
    
    # Criar dados sintéticos
    X, y = create_synthetic_forex_data(1000)
    print(f"Dados criados: {len(X)} amostras, {X.shape[1]} features")
    print(f"Distribuição do target: {np.bincount(y.astype(int))}")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Treinar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Fazer predições
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Calcular estatísticas de confiança
    confidences = np.max(probabilities, axis=1)
    
    print(f"\nEstatísticas de Confiança:")
    print(f"  Confiança Média: {np.mean(confidences):.3f}")
    print(f"  Confiança Mediana: {np.median(confidences):.3f}")
    print(f"  Confiança Mínima: {np.min(confidences):.3f}")
    print(f"  Confiança Máxima: {np.max(confidences):.3f}")
    print(f"  Desvio Padrão: {np.std(confidences):.3f}")
    
    # Verificar configuração atual
    try:
        with open('auto_trader_config.json', 'r') as f:
            config = json.load(f)
            current_min_confidence = config.get('min_confidence', 0.65)
    except:
        current_min_confidence = 0.65
    
    print(f"\nConfiguração Atual:")
    print(f"  min_confidence: {current_min_confidence}")
    
    # Analisar quantas predições passariam no filtro
    passing_predictions = confidences >= current_min_confidence
    pass_rate = np.mean(passing_predictions)
    
    print(f"\nAnálise de Filtro:")
    print(f"  Predições que passam no filtro: {np.sum(passing_predictions)}/{len(confidences)} ({pass_rate:.1%})")
    
    if pass_rate < 0.1:  # Menos de 10% passam
        print(f"  ⚠️  PROBLEMA: Muito poucas predições passam no filtro!")
        
        # Sugerir novo valor
        suggested_confidence = np.percentile(confidences, 70)  # 70% das predições passariam
        print(f"  💡 Sugestão: Reduzir min_confidence para {suggested_confidence:.3f}")
        
        # Criar configuração otimizada
        optimized_config = {
            "min_confidence": round(suggested_confidence, 3),
            "max_risk_per_trade": 0.02,
            "max_daily_trades": 10,
            "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
            "ollama_url": "http://localhost:11434",
            "prediction_interval": 300,
            "risk_reward_ratio": 2.0,
            "max_spread_pips": 3.0,
            "trading_hours": {
                "start": "08:00",
                "end": "18:00"
            }
        }
        
        with open('auto_trader_config_optimized.json', 'w') as f:
            json.dump(optimized_config, f, indent=2)
        
        print(f"  ✅ Configuração otimizada salva em 'auto_trader_config_optimized.json'")
    
    elif pass_rate > 0.8:  # Mais de 80% passam
        print(f"  ✅ Configuração atual parece adequada")
    else:
        print(f"  ℹ️  Configuração atual é razoável ({pass_rate:.1%} das predições passam)")
    
    # Testar diferentes thresholds
    print(f"\nAnálise de Diferentes Thresholds:")
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    for threshold in thresholds:
        pass_rate = np.mean(confidences >= threshold)
        print(f"  Threshold {threshold:.2f}: {pass_rate:.1%} das predições passam")

if __name__ == "__main__":
    test_model_confidence()