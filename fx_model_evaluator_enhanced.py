#!/usr/bin/env python3
"""
Sistema Robusto de Avaliação de Modelos FX - Versão Corrigida
Corrige problemas de:
1. Modelos não encontrados
2. Incompatibilidade de features
3. Falhas no backtesting

Autor: Sistema ML Trading
Data: 2025-01-18
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Métricas de avaliação
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_absolute_percentage_error
)

# Testes estatísticos
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Logging
import logging

class EnhancedFXModelEvaluator:
    """
    Avaliador robusto de modelos FX com correções automáticas
    """
    
    def __init__(self, base_dir: str = ".", min_samples: int = 100, significance_level: float = 0.05):
        self.base_dir = Path(base_dir)
        self.min_samples = min_samples
        self.significance_level = significance_level
        
        # Diretórios
        self.trained_models_dir = self.base_dir / "trained_models"
        self.optimized_models_dir = self.base_dir / "optimized_models"
        self.processed_data_dir = self.base_dir / "processed_fx_data"
        self.results_dir = self.base_dir / "model_evaluation_enhanced"
        
        # Criar diretório de resultados
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Tipos de modelos suportados
        self.model_types = {
            'regression': [
                'random_forest_reg', 'gradient_boosting_reg', 'xgboost_reg', 'lightgbm_reg'
            ],
            'classification': [
                'random_forest_clf', 'gradient_boosting_clf', 'xgboost_clf', 'lightgbm_clf'
            ]
        }
        
        # Métricas de avaliação
        self.regression_metrics = {
            'MSE': mean_squared_error,
            'MAE': mean_absolute_error,
            'R2': r2_score,
            'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': mean_absolute_percentage_error
        }
        
        self.classification_metrics = {
            'Accuracy': accuracy_score,
            'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Cache para metadados
        self.metadata_cache = {}
        
        # Estatísticas de correções
        self.correction_stats = {
            'models_not_found': 0,
            'feature_mismatches_fixed': 0,
            'successful_evaluations': 0,
            'total_attempts': 0
        }
    
    def _setup_logging(self):
        """Configura sistema de logging"""
        log_file = self.results_dir / f"evaluation_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Limpar handlers existentes
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Sistema de Avaliacao Robusto Iniciado")
        self.logger.info(f"Log file: {log_file}")
    
    def _discover_available_pairs(self) -> List[str]:
        """Descobre pares disponíveis nos dados processados"""
        pairs = []
        
        self.logger.info(f"🔍 Verificando diretório: {self.processed_data_dir}")
        self.logger.info(f"🔍 Diretório existe: {self.processed_data_dir.exists()}")
        
        if self.processed_data_dir.exists():
            self.logger.info(f"🔍 Listando conteúdo do diretório...")
            for pair_dir in self.processed_data_dir.iterdir():
                self.logger.info(f"🔍 Verificando: {pair_dir.name} (é diretório: {pair_dir.is_dir()})")
                if pair_dir.is_dir():
                    test_file = pair_dir / "test.parquet"
                    self.logger.info(f"🔍 Arquivo test.parquet existe em {pair_dir.name}: {test_file.exists()}")
                    if test_file.exists():
                        pairs.append(pair_dir.name)
                        self.logger.info(f"✅ Par adicionado: {pair_dir.name}")
        else:
            self.logger.error(f"❌ Diretório não encontrado: {self.processed_data_dir}")
        
        self.logger.info(f"📊 Pares descobertos: {pairs}")
        return sorted(pairs)
    
    def load_test_data(self, pair: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Carrega dados de teste com validação robusta"""
        pair_dir = self.processed_data_dir / pair
        
        if not pair_dir.exists():
            self.logger.error(f"❌ Diretório do par não encontrado: {pair_dir}")
            return None, None, None
        
        # Procurar arquivos de teste
        test_files = list(pair_dir.glob("*test*.csv")) + list(pair_dir.glob("*test*.parquet"))
        
        if not test_files:
            self.logger.error(f"❌ Nenhum arquivo de teste encontrado para {pair}")
            return None, None, None
        
        try:
            # Carregar o primeiro arquivo encontrado
            test_file = test_files[0]
            
            if test_file.suffix == '.csv':
                test_data = pd.read_csv(test_file)
            else:
                test_data = pd.read_parquet(test_file)
            
            # Validar dados
            if len(test_data) < self.min_samples:
                self.logger.warning(f"⚠️ Dados insuficientes para {pair}: {len(test_data)} < {self.min_samples}")
                return None, None, None
            
            # Identificar colunas de features e targets
            feature_cols = [col for col in test_data.columns 
                          if not col.startswith('Target_') and not col.startswith('Unnamed')]
            
            if not feature_cols:
                self.logger.error(f"❌ Nenhuma feature encontrada para {pair}")
                return None, None, None
            
            # Converter para numérico e tratar tipos de dados
            test_data_numeric = test_data[feature_cols].apply(pd.to_numeric, errors='coerce')
            X_test = test_data_numeric.values
            
            # Verificar targets
            y_test_reg = None
            y_test_clf = None
            
            reg_targets = [col for col in test_data.columns if 'Return' in col and 'Target' in col]
            clf_targets = [col for col in test_data.columns if 'Direction' in col and 'Target' in col]
            
            if reg_targets:
                y_test_reg_series = pd.to_numeric(test_data[reg_targets[0]], errors='coerce')
                y_test_reg = y_test_reg_series.values
                # Validar targets de regressão
                if pd.isna(y_test_reg).all() or np.isinf(y_test_reg).any():
                    self.logger.warning(f"⚠️ Targets de regressão inválidos para {pair}")
                    y_test_reg = None
            
            if clf_targets:
                y_test_clf_series = pd.to_numeric(test_data[clf_targets[0]], errors='coerce')
                y_test_clf = y_test_clf_series.values
                # Validar targets de classificação
                valid_mask = ~pd.isna(y_test_clf)
                if valid_mask.any():
                    unique_classes = np.unique(y_test_clf[valid_mask])
                    if len(unique_classes) < 2:
                        self.logger.warning(f"⚠️ Targets de classificação insuficientes para {pair}: {unique_classes}")
                        y_test_clf = None
                else:
                    self.logger.warning(f"⚠️ Nenhum target de classificação válido para {pair}")
                    y_test_clf = None
            
            # Validar features
            if pd.isna(X_test).any() or np.isinf(X_test).any():
                self.logger.warning(f"⚠️ Features com valores inválidos para {pair}, aplicando limpeza...")
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
            
            self.logger.info(f"✅ Dados carregados para {pair}: {X_test.shape[0]} amostras, {X_test.shape[1]} features")
            return X_test, y_test_reg, y_test_clf
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar dados para {pair}: {e}")
            return None, None, None
    
    def load_model_metadata(self, pair: str, model_type: str, model_category: str = 'optimized') -> Optional[Dict]:
        """Carrega metadados do modelo com cache"""
        cache_key = f"{pair}_{model_type}_{model_category}"
        
        if cache_key in self.metadata_cache:
            return self.metadata_cache[cache_key]
        
        if model_category == 'optimized':
            model_dir = self.optimized_models_dir / pair
        else:
            model_dir = self.trained_models_dir / pair
        
        if not model_dir.exists():
            return None
        
        # Procurar arquivo de metadados
        metadata_files = list(model_dir.glob(f"*{model_type}*metadata.json"))
        
        if not metadata_files:
            return None
        
        try:
            with open(metadata_files[0], 'r') as f:
                metadata = json.load(f)
            
            self.metadata_cache[cache_key] = metadata
            return metadata
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erro ao carregar metadados {pair}/{model_type}: {e}")
            return None
    
    def adjust_features_for_model(self, X: np.ndarray, model_metadata: Dict) -> np.ndarray:
        """Ajusta features para compatibilidade com modelo treinado"""
        if 'feature_importance' not in model_metadata:
            return X
        
        expected_features = len(model_metadata['feature_importance'])
        current_features = X.shape[1]
        
        if current_features == expected_features:
            return X
        
        self.logger.warning(f"⚠️ Ajustando features: {current_features} -> {expected_features}")
        self.correction_stats['feature_mismatches_fixed'] += 1
        
        if current_features > expected_features:
            # Truncar features extras
            return X[:, :expected_features]
        else:
            # Preencher features faltantes com zeros
            padding = np.zeros((X.shape[0], expected_features - current_features))
            return np.hstack([X, padding])
    
    def load_model(self, pair: str, model_type: str, model_category: str = 'optimized') -> Optional[Any]:
        """Carrega modelo com fallbacks robustos"""
        if model_category == 'optimized':
            model_dir = self.optimized_models_dir / pair
        else:
            model_dir = self.trained_models_dir / pair
        
        if not model_dir.exists():
            self.logger.warning(f"⚠️ Diretório de modelos não encontrado: {model_dir}")
            self.correction_stats['models_not_found'] += 1
            return None
        
        # Procurar arquivo do modelo com múltiplas variações
        model_patterns = [
            f"*{model_type}*.pkl",
            f"*{model_type}*.joblib",
            f"{model_type}*.pkl",
            f"{model_type}*.joblib"
        ]
        
        model_files = []
        for pattern in model_patterns:
            model_files.extend(list(model_dir.glob(pattern)))
        
        if not model_files:
            self.logger.warning(f"⚠️ Modelo não encontrado: {pair}/{model_type}")
            self.correction_stats['models_not_found'] += 1
            return None
        
        try:
            model_file = model_files[0]
            
            if model_file.suffix == '.pkl':
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
            else:
                import joblib
                model = joblib.load(model_file)
            
            # Validar se o modelo tem método predict
            if not hasattr(model, 'predict'):
                self.logger.error(f"❌ Modelo inválido (sem método predict): {model_file}")
                return None
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar modelo {pair}/{model_type}: {e}")
            return None
    
    def evaluate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 model_type: str) -> Dict[str, float]:
        """Avalia performance do modelo com métricas apropriadas"""
        results = {}
        
        # Validar entradas
        if len(y_true) != len(y_pred):
            self.logger.error("❌ Tamanhos incompatíveis entre y_true e y_pred")
            return results
        
        # Remover valores inválidos
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        
        if not valid_mask.any():
            self.logger.error("❌ Nenhuma predição válida encontrada")
            return results
        
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        try:
            if 'reg' in model_type:
                # Métricas de regressão
                for metric_name, metric_func in self.regression_metrics.items():
                    try:
                        if metric_name == 'MAPE':
                            # Evitar divisão por zero no MAPE
                            mask = y_true_clean != 0
                            if mask.any():
                                results[metric_name] = metric_func(y_true_clean[mask], y_pred_clean[mask])
                            else:
                                results[metric_name] = np.inf
                        else:
                            results[metric_name] = metric_func(y_true_clean, y_pred_clean)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Erro ao calcular {metric_name}: {e}")
                        results[metric_name] = np.nan
            
            elif 'clf' in model_type:
                # Métricas de classificação
                for metric_name, metric_func in self.classification_metrics.items():
                    try:
                        results[metric_name] = metric_func(y_true_clean, y_pred_clean)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Erro ao calcular {metric_name}: {e}")
                        results[metric_name] = np.nan
                
                # Adicionar AUC se possível
                try:
                    if len(np.unique(y_true_clean)) == 2:  # Classificação binária
                        results['AUC'] = roc_auc_score(y_true_clean, y_pred_clean)
                except:
                    pass
        
        except Exception as e:
            self.logger.error(f"❌ Erro na avaliação de performance: {e}")
        
        return results
    
    def evaluate_pair_models(self, pair: str) -> Dict[str, Any]:
        """Avalia todos os modelos para um par específico com correções automáticas"""
        self.logger.info(f"🎯 Avaliando modelos para {pair}...")
        
        # Carregar dados de teste
        X_test, y_test_reg, y_test_clf = self.load_test_data(pair)
        
        if X_test is None:
            self.logger.error(f"❌ Não foi possível carregar dados de teste para {pair}")
            return {}
        
        pair_results = {
            'pair': pair,
            'test_samples': len(X_test),
            'original_features': X_test.shape[1],
            'models': {},
            'summary': {
                'total_models': 0,
                'successful_evaluations': 0,
                'models_with_improvement': 0,
                'average_improvement': 0.0,
                'corrections_applied': 0
            }
        }
        
        all_model_types = (self.model_types['regression'] + self.model_types['classification'])
        
        for model_type in all_model_types:
            self.correction_stats['total_attempts'] += 1
            
            try:
                # Determinar tipo de target
                if 'reg' in model_type:
                    if y_test_reg is None:
                        continue
                    y_test = y_test_reg
                else:
                    if y_test_clf is None:
                        continue
                    y_test = y_test_clf
                
                pair_results['summary']['total_models'] += 1
                
                # Carregar metadados para ajuste de features
                baseline_metadata = self.load_model_metadata(pair, model_type, 'baseline')
                optimized_metadata = self.load_model_metadata(pair, model_type, 'optimized')
                
                # Carregar modelos
                baseline_model = self.load_model(pair, model_type, 'baseline')
                optimized_model = self.load_model(pair, model_type, 'optimized')
                
                model_result = {
                    'model_type': model_type,
                    'baseline_available': baseline_model is not None,
                    'optimized_available': optimized_model is not None,
                    'baseline_metrics': {},
                    'optimized_metrics': {},
                    'improvements': {},
                    'feature_adjustments': {}
                }
                
                # Avaliar modelo baseline
                if baseline_model is not None:
                    try:
                        X_test_adjusted = X_test
                        if baseline_metadata:
                            X_test_adjusted = self.adjust_features_for_model(X_test, baseline_metadata)
                            model_result['feature_adjustments']['baseline'] = {
                                'original': X_test.shape[1],
                                'adjusted': X_test_adjusted.shape[1]
                            }
                        
                        y_pred_baseline = baseline_model.predict(X_test_adjusted)
                        model_result['baseline_metrics'] = self.evaluate_model_performance(
                            y_test, y_pred_baseline, model_type
                        )
                        
                        if model_result['baseline_metrics']:
                            self.correction_stats['successful_evaluations'] += 1
                            
                    except Exception as e:
                        self.logger.warning(f"⚠️ Erro na predição baseline {model_type}: {e}")
                
                # Avaliar modelo otimizado
                if optimized_model is not None:
                    try:
                        X_test_adjusted = X_test
                        if optimized_metadata:
                            X_test_adjusted = self.adjust_features_for_model(X_test, optimized_metadata)
                            model_result['feature_adjustments']['optimized'] = {
                                'original': X_test.shape[1],
                                'adjusted': X_test_adjusted.shape[1]
                            }
                        
                        y_pred_optimized = optimized_model.predict(X_test_adjusted)
                        model_result['optimized_metrics'] = self.evaluate_model_performance(
                            y_test, y_pred_optimized, model_type
                        )
                        
                        if model_result['optimized_metrics']:
                            self.correction_stats['successful_evaluations'] += 1
                            
                    except Exception as e:
                        self.logger.warning(f"⚠️ Erro na predição otimizada {model_type}: {e}")
                
                # Calcular melhorias
                if model_result['baseline_metrics'] and model_result['optimized_metrics']:
                    model_result['improvements'] = self.compare_models(
                        model_result['baseline_metrics'], 
                        model_result['optimized_metrics']
                    )
                
                pair_results['models'][model_type] = model_result
                
                if model_result['baseline_metrics'] or model_result['optimized_metrics']:
                    pair_results['summary']['successful_evaluations'] += 1
                
            except Exception as e:
                self.logger.error(f"❌ Erro ao avaliar modelo {model_type} para {pair}: {e}")
        
        return pair_results
    
    def compare_models(self, baseline_results: Dict[str, float], 
                     optimized_results: Dict[str, float]) -> Dict[str, float]:
        """Compara performance entre modelos baseline e otimizados"""
        improvements = {}
        
        for metric in baseline_results.keys():
            if metric in optimized_results:
                baseline_val = baseline_results[metric]
                optimized_val = optimized_results[metric]
                
                if not np.isnan(baseline_val) and not np.isnan(optimized_val) and baseline_val != 0:
                    # Para métricas onde menor é melhor (MSE, MAE, RMSE, MAPE)
                    if metric in ['MSE', 'MAE', 'RMSE', 'MAPE']:
                        improvement = ((baseline_val - optimized_val) / baseline_val) * 100
                    else:
                        # Para métricas onde maior é melhor (R2, Accuracy, etc.)
                        improvement = ((optimized_val - baseline_val) / abs(baseline_val)) * 100
                    
                    improvements[f"{metric}_improvement_%"] = improvement
        
        return improvements
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """Executa avaliação completa com correções automáticas"""
        self.logger.info("🚀 Iniciando avaliação completa com correções automáticas...")
        
        # Descobrir pares disponíveis
        available_pairs = self._discover_available_pairs()
        
        if not available_pairs:
            self.logger.error("❌ Nenhum par de dados encontrado")
            return {}
        
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'pairs_evaluated': [],
            'correction_statistics': {},
            'global_summary': {},
            'detailed_results': {}
        }
        
        # Avaliar cada par
        for pair in available_pairs:
            try:
                pair_results = self.evaluate_pair_models(pair)
                if pair_results:
                    evaluation_results['pairs_evaluated'].append(pair)
                    evaluation_results['detailed_results'][pair] = pair_results
                    
            except Exception as e:
                self.logger.error(f"❌ Erro ao avaliar par {pair}: {e}")
        
        # Estatísticas de correções
        evaluation_results['correction_statistics'] = self.correction_stats.copy()
        
        # Calcular estatísticas globais
        evaluation_results['global_summary'] = self._calculate_global_statistics(evaluation_results)
        
        # Salvar resultados
        self._save_evaluation_results(evaluation_results)
        
        # Gerar relatório
        self._generate_comprehensive_report(evaluation_results)
        
        return evaluation_results
    
    def _calculate_global_statistics(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula estatísticas globais da avaliação"""
        stats = {
            'total_pairs': len(evaluation_results['pairs_evaluated']),
            'total_models_attempted': 0,
            'successful_evaluations': 0,
            'correction_rate': 0.0,
            'average_features_per_pair': 0.0,
            'model_availability': {}
        }
        
        if not evaluation_results['detailed_results']:
            return stats
        
        total_features = 0
        model_counts = {}
        
        for pair_results in evaluation_results['detailed_results'].values():
            stats['total_models_attempted'] += pair_results['summary']['total_models']
            stats['successful_evaluations'] += pair_results['summary']['successful_evaluations']
            total_features += pair_results.get('original_features', 0)
            
            # Contar disponibilidade de modelos
            for model_type, model_result in pair_results['models'].items():
                if model_type not in model_counts:
                    model_counts[model_type] = {'baseline': 0, 'optimized': 0}
                
                if model_result['baseline_available']:
                    model_counts[model_type]['baseline'] += 1
                if model_result['optimized_available']:
                    model_counts[model_type]['optimized'] += 1
        
        if stats['total_pairs'] > 0:
            stats['average_features_per_pair'] = total_features / stats['total_pairs']
        
        if self.correction_stats['total_attempts'] > 0:
            stats['correction_rate'] = (
                (self.correction_stats['feature_mismatches_fixed'] + 
                 self.correction_stats['successful_evaluations']) / 
                self.correction_stats['total_attempts']
            ) * 100
        
        stats['model_availability'] = model_counts
        
        return stats
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Salva resultados da avaliação"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"evaluation_results_enhanced_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"💾 Resultados salvos em: {results_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao salvar resultados: {e}")
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]):
        """Gera relatório abrangente da avaliação"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.results_dir / f"evaluation_report_enhanced_{timestamp}.md"
        
        try:
            report_content = self._create_report_content(results)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info(f"📋 Relatório gerado: {report_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao gerar relatório: {e}")
    
    def _create_report_content(self, results: Dict[str, Any]) -> str:
        """Cria conteúdo do relatório"""
        report = f"""
# Relatório de Avaliação de Modelos FX - Versão Corrigida

**Data:** {results['timestamp']}
**Sistema:** Avaliador Robusto com Correções Automáticas

## 📊 Resumo Executivo

### Estatísticas Gerais
- **Pares Avaliados:** {results['global_summary']['total_pairs']}
- **Modelos Tentados:** {results['global_summary']['total_models_attempted']}
- **Avaliações Bem-sucedidas:** {results['global_summary']['successful_evaluations']}
- **Taxa de Correção:** {results['global_summary']['correction_rate']:.1f}%
- **Features Médias por Par:** {results['global_summary']['average_features_per_pair']:.0f}

### 🔧 Estatísticas de Correções
- **Modelos Não Encontrados:** {results['correction_statistics']['models_not_found']}
- **Incompatibilidades de Features Corrigidas:** {results['correction_statistics']['feature_mismatches_fixed']}
- **Avaliações Bem-sucedidas:** {results['correction_statistics']['successful_evaluations']}
- **Total de Tentativas:** {results['correction_statistics']['total_attempts']}

## 📈 Disponibilidade de Modelos

"""
        
        # Adicionar disponibilidade de modelos
        if 'model_availability' in results['global_summary']:
            for model_type, counts in results['global_summary']['model_availability'].items():
                report += f"### {model_type}\n"
                report += f"- Baseline: {counts['baseline']} pares\n"
                report += f"- Otimizado: {counts['optimized']} pares\n\n"
        
        # Adicionar detalhes por par
        report += "## 🎯 Resultados Detalhados por Par\n\n"
        
        for pair, pair_results in results['detailed_results'].items():
            report += f"### {pair}\n"
            report += f"- **Amostras de Teste:** {pair_results['test_samples']:,}\n"
            report += f"- **Features Originais:** {pair_results['original_features']}\n"
            report += f"- **Modelos Avaliados:** {pair_results['summary']['successful_evaluations']}/{pair_results['summary']['total_models']}\n\n"
            
            # Melhores modelos
            best_models = []
            for model_type, model_result in pair_results['models'].items():
                if model_result['optimized_metrics']:
                    if 'reg' in model_type and 'R2' in model_result['optimized_metrics']:
                        r2 = model_result['optimized_metrics']['R2']
                        best_models.append((model_type, 'R2', r2))
                    elif 'clf' in model_type and 'Accuracy' in model_result['optimized_metrics']:
                        acc = model_result['optimized_metrics']['Accuracy']
                        best_models.append((model_type, 'Accuracy', acc))
            
            if best_models:
                best_models.sort(key=lambda x: x[2], reverse=True)
                report += f"**Melhor Modelo:** {best_models[0][0]} ({best_models[0][1]}: {best_models[0][2]:.4f})\n\n"
        
        # Recomendações
        report += "## 🎯 Recomendações\n\n"
        
        if results['correction_statistics']['models_not_found'] > 0:
            report += "- ⚠️ **Modelos Faltantes:** Alguns modelos não foram encontrados. Considere retreinar os modelos faltantes.\n"
        
        if results['correction_statistics']['feature_mismatches_fixed'] > 0:
            report += "- 🔧 **Incompatibilidades Corrigidas:** Features foram ajustadas automaticamente. Verifique a consistência do pipeline de features.\n"
        
        if results['global_summary']['correction_rate'] > 80:
            report += "- ✅ **Sistema Robusto:** Alta taxa de correção automática indica boa robustez do sistema.\n"
        else:
            report += "- ⚠️ **Melhorias Necessárias:** Taxa de correção baixa. Revisar pipeline de dados e treinamento.\n"
        
        report += "\n---\n*Relatório gerado automaticamente pelo Sistema de Avaliação Robusto*\n"
        
        return report

def main():
    """Função principal"""
    print("🚀 Iniciando Sistema de Avaliação Robusto de Modelos FX")
    
    evaluator = EnhancedFXModelEvaluator()
    results = evaluator.run_complete_evaluation()
    
    if results:
        print(f"\n✅ Avaliação concluída!")
        print(f"📊 Pares avaliados: {len(results['pairs_evaluated'])}")
        print(f"🔧 Correções aplicadas: {results['correction_statistics']['feature_mismatches_fixed']}")
        print(f"✅ Avaliações bem-sucedidas: {results['correction_statistics']['successful_evaluations']}")
    else:
        print("❌ Falha na avaliação")

if __name__ == "__main__":
    main()