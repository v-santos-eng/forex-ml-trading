#!/usr/bin/env python3
"""
Demo do Sistema ML Forex Melhorado

Este script demonstra como usar o sistema ML melhorado com todas as funcionalidades:
- Ensemble Methods (XGBoost, Random Forest, Gradient Boosting)
- Walk-Forward Optimization
- PCA para Redução de Dimensionalidade
- Seleção Automática de Features
- Prevenção de Overfitting
- Integração com Agno/Ollama
"""

from agno_ml_enhanced import AgnoMLEnhanced
import json
from datetime import datetime

def demo_basic_usage():
    """
    Demonstração básica do sistema melhorado
    """
    print("DEMO - Sistema ML Forex Melhorado")
    print("="*60)
    
    # Criar sistema com configurações otimizadas
    system = AgnoMLEnhanced(
        use_pca=True,
        n_components=0.95,  # Manter 95% da variância
        feature_selection_method='auto'
    )
    
    # Treinar para um par específico
    pairs = ['EURUSD=X']
    print(f"\nTreinando sistema para {pairs[0]}...")
    
    training_results = system.train_enhanced_system(pairs, period='6mo')
    
    if training_results and pairs[0] in training_results:
        result = training_results[pairs[0]]
        if 'error' not in result:
            print(f"Treinamento concluído com sucesso!")
            print(f"   Features totais: {result['features_count']}")
            print(f"   Features selecionadas: {result['selected_features_count']}")
            print(f"   Amostras de treino: {result['training_samples']}")
            
            # Fazer predição
            print(f"\nGerando predição...")
            prediction = system.enhanced_prediction(pairs[0])
            
            if prediction:
                ml_pred = prediction['ml_prediction']
                print(f"\nRESULTADO:")
                print(f"   Par: {ml_pred['pair']}")
                print(f"   Direção: {ml_pred['direction']}")
                print(f"   Confiança: {ml_pred['confidence']}%")
                print(f"   Modelo: {ml_pred['model_type']}")
                
                return prediction
        else:
            print(f"Erro no treinamento: {result['error']}")
    
    return None

def demo_advanced_features():
    """
    Demonstração de funcionalidades avançadas
    """
    print("\nDEMO - Funcionalidades Avançadas")
    print("="*60)
    
    # Sistema com configurações diferentes
    system = AgnoMLEnhanced(
        use_pca=True,
        n_components=10,  # Número fixo de componentes
        feature_selection_method='rfe'  # Recursive Feature Elimination
    )
    
    # Treinar múltiplos pares
    pairs = ['EURUSD=X', 'GBPUSD=X']
    print(f"\nTreinamento avançado para múltiplos pares...")
    
    training_results = system.train_enhanced_system(pairs, period='1y')
    
    # Comparar com sistema básico
    print(f"\nComparando com sistema básico...")
    comparison = system.compare_with_basic_system('EURUSD=X')
    
    # Gerar relatório completo
    print(f"\nGerando relatório completo...")
    report = system.generate_enhanced_report(pairs)
    
    return report

def demo_custom_configuration():
    """
    Demonstração de configuração personalizada
    """
    print("\nDEMO - Configuração Personalizada")
    print("="*60)
    
    # Sistema sem PCA
    system_no_pca = AgnoMLEnhanced(
        use_pca=False,
        feature_selection_method='mutual_info'
    )
    
    print("\nSistema configurado sem PCA, usando Mutual Information")
    
    # Treinar
    pairs = ['EURUSD=X']
    training_results = system_no_pca.train_enhanced_system(pairs, period='3mo')
    
    if training_results and pairs[0] in training_results:
        result = training_results[pairs[0]]
        if 'error' not in result:
            print(f"Sistema alternativo treinado!")
            
            # Predição
            prediction = system_no_pca.enhanced_prediction(pairs[0])
            if prediction:
                ml_pred = prediction['ml_prediction']
                print(f"\nPredição sem PCA:")
                print(f"   Direção: {ml_pred['direction']}")
                print(f"   Confiança: {ml_pred['confidence']}%")
                
                return prediction
    
    return None

def main():
    """
    Executa todas as demonstrações
    """
    print("DEMONSTRAÇÃO COMPLETA - Sistema ML Forex Melhorado")
    print("="*80)
    print("\nFuncionalidades implementadas:")
    print("1. Ensemble Methods (XGBoost, Random Forest, Gradient Boosting)")
    print("2. Walk-Forward Optimization")
    print("3. PCA para Redução de Dimensionalidade")
    print("4. Seleção Automática de Features")
    print("5. Prevenção de Overfitting")
    print("6. Integração com Agno/Ollama")
    print("\n" + "="*80)
    
    # Demo básico
    basic_result = demo_basic_usage()
    
    # Demo avançado
    advanced_result = demo_advanced_features()
    
    # Demo personalizado
    custom_result = demo_custom_configuration()
    
    # Resumo final
    print("\nRESUMO FINAL")
    print("="*60)
    
    if basic_result:
        print("Demo básico: Sucesso")
    else:
        print("Demo básico: Falhou")
    
    if advanced_result:
        print("Demo avançado: Sucesso")
        print(f"   Relatório salvo: enhanced_ml_report_*.json")
    else:
        print("Demo avançado: Falhou")
    
    if custom_result:
        print("Demo personalizado: Sucesso")
    else:
        print("Demo personalizado: Falhou")
    
    print("\nSistema ML Forex Melhorado está pronto para uso!")
    print("\nComo usar:")
    print("   1. from agno_ml_enhanced import AgnoMLEnhanced")
    print("   2. system = AgnoMLEnhanced(use_pca=True, n_components=0.95)")
    print("   3. system.train_enhanced_system(['EURUSD=X'])")
    print("   4. prediction = system.enhanced_prediction('EURUSD=X')")

if __name__ == "__main__":
    main()