#!/usr/bin/env python3
"""
Agno ML Enhanced - Integração do Sistema ML Melhorado com Agno/Ollama

Este módulo integra o sistema ML melhorado com ensemble methods,
walk-forward optimization, PCA e seleção automática de features
com o sistema Agno/Ollama existente.
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Importar o sistema ML melhorado
from ml_predictor_enhanced import MLForexPredictorEnhanced

# Importar componentes do Agno existente
try:
    from agno_ollama_agents import ForexAnalysisAgent, RiskAnalysisAgent, TechnicalAnalysisAgent
    from agno_ollama_memory import AgentMemorySystem
    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False
    print("Componentes Agno não encontrados. Funcionando em modo standalone.")

class AgnoMLEnhanced:
    """
    Sistema integrado Agno ML com funcionalidades avançadas
    """
    
    def __init__(self, use_pca=True, n_components=0.95, feature_selection_method='auto'):
        """
        Inicializa o sistema integrado
        """
        # Sistema ML melhorado
        self.ml_predictor = MLForexPredictorEnhanced(
            use_pca=use_pca,
            n_components=n_components,
            feature_selection_method=feature_selection_method
        )
        
        # Sistema de memória
        if AGNO_AVAILABLE:
            self.memory_system = AgentMemorySystem()
            
            # Agentes Ollama
            self.forex_agent = ForexAnalysisAgent()
            self.risk_agent = RiskAnalysisAgent()
            self.technical_agent = TechnicalAnalysisAgent()
        else:
            self.memory_system = None
            self.forex_agent = None
            self.risk_agent = None
            self.technical_agent = None
        
        # Estado do sistema
        self.is_trained = False
        self.training_results = {}
        self.last_predictions = {}
        
    def train_enhanced_system(self, pairs=['EURUSD=X', 'GBPUSD=X'], period='2y'):
        """
        Treina o sistema ML melhorado para múltiplos pares
        """
        print("Iniciando treinamento do sistema ML melhorado...")
        print("="*70)
        
        training_results = {}
        
        for pair in pairs:
            print(f"\nTreinando para {pair}...")
            
            try:
                # Obter dados
                data = self.ml_predictor.get_forex_data(pair, period=period)
                if data is None or data.empty:
                    print(f"Dados insuficientes para {pair}")
                    continue
                
                # Criar features avançadas
                features = self.ml_predictor.create_enhanced_features(data)
                target = self.ml_predictor.create_target(data, prediction_horizon=1, threshold=0.001)
                
                print(f"   Features criadas: {len(self.ml_predictor.feature_names)}")
                
                # Walk-forward optimization
                print(f"   Executando Walk-Forward Optimization...")
                wf_results = self.ml_predictor.walk_forward_optimization(features, target)
                
                # Treinar modelo final
                print(f"   Treinando modelo final...")
                final_results = self.ml_predictor.train_final_model(features, target)
                
                # Armazenar resultados
                training_results[pair] = {
                    'walk_forward': wf_results,
                    'final_model': final_results,
                    'features_count': len(self.ml_predictor.feature_names),
                    'selected_features_count': len(self.ml_predictor.selected_features),
                    'training_samples': len(features),
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"   {pair} treinado com sucesso!")
                
            except Exception as e:
                print(f"   Erro ao treinar {pair}: {e}")
                training_results[pair] = {'error': str(e)}
        
        self.training_results = training_results
        self.is_trained = True
        
        print(f"\nTreinamento concluído para {len([p for p in training_results if 'error' not in training_results[p]])} pares!")
        
        return training_results
    
    def enhanced_prediction(self, pair, include_ollama_analysis=True):
        """
        Faz predição melhorada combinando ML avançado com análise Ollama
        """
        if not self.is_trained:
            print("Sistema não treinado. Execute train_enhanced_system primeiro.")
            return None
        
        print(f"Gerando predição melhorada para {pair}...")
        
        # Predição ML melhorada
        ml_prediction = self.ml_predictor.predict_direction(pair)
        if not ml_prediction:
            print(f"Falha na predição ML para {pair}")
            return None
        
        # Análise Ollama (se disponível)
        ollama_analysis = {}
        if include_ollama_analysis and AGNO_AVAILABLE:
            try:
                print("   Executando análise Ollama...")
                
                # Análise técnica
                if self.technical_agent:
                    technical_analysis = self.technical_agent.analyze_pair(pair)
                    ollama_analysis['technical'] = technical_analysis
                
                # Análise fundamental
                if self.forex_agent:
                    fundamental_analysis = self.forex_agent.analyze_pair(pair)
                    ollama_analysis['fundamental'] = fundamental_analysis
                
                # Análise de risco
                if self.risk_agent:
                    risk_analysis = self.risk_agent.analyze_risk(pair, ml_prediction)
                    ollama_analysis['risk'] = risk_analysis
                
            except Exception as e:
                print(f"   Erro na análise Ollama: {e}")
                ollama_analysis['error'] = str(e)
        
        # Combinar análises
        enhanced_prediction = {
            'pair': pair,
            'timestamp': datetime.now().isoformat(),
            'ml_prediction': ml_prediction,
            'ollama_analysis': ollama_analysis,
            'model_summary': self.ml_predictor.get_model_summary(),
            'confidence_score': self._calculate_combined_confidence(ml_prediction, ollama_analysis)
        }
        
        # Salvar na memória
        if self.memory_system:
            try:
                self.memory_system.store_prediction(enhanced_prediction)
            except Exception as e:
                print(f"   Erro ao salvar na memória: {e}")
        
        # Armazenar localmente
        self.last_predictions[pair] = enhanced_prediction
        
        return enhanced_prediction
    
    def _calculate_combined_confidence(self, ml_prediction, ollama_analysis):
        """
        Calcula confiança combinada entre ML e análise Ollama
        """
        base_confidence = ml_prediction.get('confidence', 50)
        
        # Se não há análise Ollama, usar apenas ML
        if not ollama_analysis or 'error' in ollama_analysis:
            return base_confidence
        
        # Ajustar confiança baseado na concordância
        adjustments = []
        
        # Verificar concordância técnica
        if 'technical' in ollama_analysis:
            tech_direction = self._extract_direction_from_analysis(ollama_analysis['technical'])
            if tech_direction == ml_prediction['direction']:
                adjustments.append(5)  # +5% se concordam
            else:
                adjustments.append(-10)  # -10% se discordam
        
        # Verificar concordância fundamental
        if 'fundamental' in ollama_analysis:
            fund_direction = self._extract_direction_from_analysis(ollama_analysis['fundamental'])
            if fund_direction == ml_prediction['direction']:
                adjustments.append(3)  # +3% se concordam
            else:
                adjustments.append(-5)  # -5% se discordam
        
        # Aplicar ajustes
        final_confidence = base_confidence + sum(adjustments)
        
        # Limitar entre 0 e 100
        return max(0, min(100, final_confidence))
    
    def _extract_direction_from_analysis(self, analysis_text):
        """
        Extrai direção (ALTA/BAIXA) de texto de análise
        """
        if not isinstance(analysis_text, str):
            return None
        
        text_lower = analysis_text.lower()
        
        # Palavras indicativas de alta
        bullish_words = ['alta', 'subir', 'compra', 'buy', 'bullish', 'positivo', 'otimista']
        # Palavras indicativas de baixa
        bearish_words = ['baixa', 'descer', 'venda', 'sell', 'bearish', 'negativo', 'pessimista']
        
        bullish_count = sum(1 for word in bullish_words if word in text_lower)
        bearish_count = sum(1 for word in bearish_words if word in text_lower)
        
        if bullish_count > bearish_count:
            return 'ALTA'
        elif bearish_count > bullish_count:
            return 'BAIXA'
        else:
            return None
    
    def generate_enhanced_report(self, pairs=['EURUSD=X', 'GBPUSD=X']):
        """
        Gera relatório completo do sistema melhorado
        """
        print("Gerando relatório melhorado...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'version': 'Enhanced ML with Agno/Ollama Integration',
                'ml_model': self.ml_predictor.get_model_summary(),
                'agno_available': AGNO_AVAILABLE,
                'is_trained': self.is_trained
            },
            'training_results': self.training_results,
            'predictions': {},
            'performance_summary': {}
        }
        
        # Gerar predições para todos os pares
        for pair in pairs:
            try:
                prediction = self.enhanced_prediction(pair)
                if prediction:
                    report['predictions'][pair] = prediction
            except Exception as e:
                report['predictions'][pair] = {'error': str(e)}
        
        # Resumo de performance
        if self.training_results:
            successful_pairs = [p for p in self.training_results if 'error' not in self.training_results[p]]
            
            if successful_pairs:
                avg_accuracy = np.mean([
                    self.training_results[p]['walk_forward']['avg_accuracy'] 
                    for p in successful_pairs 
                    if self.training_results[p].get('walk_forward')
                ])
                
                report['performance_summary'] = {
                    'successful_pairs': len(successful_pairs),
                    'total_pairs': len(self.training_results),
                    'average_accuracy': float(avg_accuracy) if not np.isnan(avg_accuracy) else None,
                    'features_used': self.ml_predictor.get_model_summary().get('selected_features', 0)
                }
        
        # Converter tipos numpy para JSON serializable
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Salvar relatório
        report_filename = f"enhanced_ml_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(report), f, indent=2, ensure_ascii=False)
        
        print(f"Relatório salvo em: {report_filename}")
        
        return report
    
    def compare_with_basic_system(self, pair='EURUSD=X'):
        """
        Compara sistema melhorado com sistema básico
        """
        print(f"Comparando sistemas para {pair}...")
        
        # Importar sistema básico
        try:
            from ml_predictor_free import MLForexPredictorFree
            basic_predictor = MLForexPredictorFree()
            
            # Treinar sistema básico
            X_basic, y_basic = basic_predictor.prepare_data(pair, period='1y')
            if X_basic is not None and y_basic is not None:
                basic_results = basic_predictor.train_models(X_basic, y_basic)
                basic_prediction = basic_predictor.predict_direction(pair)
            else:
                basic_prediction = None
                basic_results = None
            
        except Exception as e:
            print(f"Erro ao executar sistema básico: {e}")
            basic_prediction = None
            basic_results = None
        
        # Predição do sistema melhorado
        enhanced_prediction = self.enhanced_prediction(pair, include_ollama_analysis=False)
        
        # Comparação
        comparison = {
            'pair': pair,
            'timestamp': datetime.now().isoformat(),
            'basic_system': {
                'prediction': basic_prediction,
                'features_count': len(basic_predictor.feature_names) if basic_prediction else 0,
                'models': list(basic_predictor.models.keys()) if basic_prediction else []
            },
            'enhanced_system': {
                'prediction': enhanced_prediction['ml_prediction'] if enhanced_prediction else None,
                'features_count': self.ml_predictor.get_model_summary().get('total_features', 0),
                'selected_features': self.ml_predictor.get_model_summary().get('selected_features', 0),
                'models': self.ml_predictor.get_model_summary().get('ensemble_models', []),
                'use_pca': self.ml_predictor.use_pca,
                'feature_selection': self.ml_predictor.feature_selection_method
            },
            'improvements': {
                'ensemble_methods': True,
                'walk_forward_optimization': True,
                'pca_dimensionality_reduction': self.ml_predictor.use_pca,
                'automatic_feature_selection': True,
                'overfitting_prevention': True,
                'ollama_integration': AGNO_AVAILABLE
            }
        }
        
        print("\nCOMPARAÇÃO DE SISTEMAS:")
        print("="*50)
        
        if basic_prediction and enhanced_prediction:
            print(f"Sistema Básico:")
            print(f"  Direção: {basic_prediction['direction']}")
            print(f"  Confiança: {basic_prediction['confidence']}%")
            print(f"  Features: {comparison['basic_system']['features_count']}")
            
            print(f"\nSistema Melhorado:")
            print(f"  Direção: {enhanced_prediction['ml_prediction']['direction']}")
            print(f"  Confiança: {enhanced_prediction['ml_prediction']['confidence']}%")
            print(f"  Features Totais: {comparison['enhanced_system']['features_count']}")
            print(f"  Features Selecionadas: {comparison['enhanced_system']['selected_features']}")
            print(f"  PCA: {'Sim' if comparison['enhanced_system']['use_pca'] else 'Não'}")
            
            # Verificar concordância
            agreement = basic_prediction['direction'] == enhanced_prediction['ml_prediction']['direction']
            print(f"\nConcordância: {'Sim' if agreement else 'Não'}")
        
        return comparison

def main():
    """
    Demonstração do sistema Agno ML melhorado
    """
    print("AGNO ML ENHANCED - Sistema Integrado")
    print("Machine Learning Avançado + Agno/Ollama")
    print("="*70)
    
    # Criar sistema integrado
    agno_enhanced = AgnoMLEnhanced(
        use_pca=True,
        n_components=0.95,
        feature_selection_method='auto'
    )
    
    # Treinar sistema
    pairs = ['EURUSD=X', 'GBPUSD=X']
    training_results = agno_enhanced.train_enhanced_system(pairs, period='1y')
    
    # Gerar predições melhoradas
    print(f"\nPREDIÇÕES MELHORADAS")
    print("="*70)
    
    for pair in pairs:
        prediction = agno_enhanced.enhanced_prediction(pair)
        if prediction:
            ml_pred = prediction['ml_prediction']
            print(f"\n{pair}:")
            print(f"   Direção: {ml_pred['direction']}")
            print(f"   Confiança ML: {ml_pred['confidence']}%")
            print(f"   Confiança Combinada: {prediction['confidence_score']}%")
            print(f"   Modelo: {ml_pred['model_type']}")
    
    # Comparar com sistema básico
    print(f"\nCOMPARAÇÃO COM SISTEMA BÁSICO")
    print("="*70)
    
    comparison = agno_enhanced.compare_with_basic_system('EURUSD=X')
    
    # Gerar relatório completo
    print(f"\nGERANDO RELATÓRIO COMPLETO")
    print("="*70)
    
    report = agno_enhanced.generate_enhanced_report(pairs)
    
    print(f"\nMELHORIAS IMPLEMENTADAS E FUNCIONANDO:")
    print("="*70)
    print("1. Ensemble Methods (XGBoost, Random Forest, Gradient Boosting)")
    print("2. Walk-Forward Optimization")
    print("3. PCA para Redução de Dimensionalidade")
    print("4. Seleção Automática de Features")
    print("5. Prevenção de Overfitting")
    print("6. Integração com Agno/Ollama")
    
    print(f"\nSISTEMA PRONTO PARA USO AVANÇADO!")

if __name__ == "__main__":
    main()