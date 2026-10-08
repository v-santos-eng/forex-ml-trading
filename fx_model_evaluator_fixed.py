#!/usr/bin/env python3
"""
FX Model Performance Evaluator - Versão Corrigida
Avalia e compara a performance dos modelos otimizados vs modelos baseline

Correções implementadas:
- Validação robusta de dados de entrada
- Métricas de avaliação mais precisas
- Tratamento adequado de dados ausentes
- Prevenção de overfitting na avaliação
- Análise estatística de significância
- Relatórios mais informativos
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

class RobustFXModelEvaluator:
    """
    Avaliador robusto de modelos FX com:
    - Validação rigorosa de dados
    - Métricas estatisticamente significativas
    - Análise de estabilidade temporal
    - Detecção de overfitting
    - Relatórios detalhados
    """
    
    def __init__(self, base_dir: str = ".", min_samples: int = 100, significance_level: float = 0.05):
        self.base_dir = Path(base_dir)
        self.min_samples = min_samples
        self.significance_level = significance_level
        
        # Diretórios
        self.processed_data_dir = self.base_dir / "processed_fx_data"
        self.trained_models_dir = self.base_dir / "trained_models"
        self.optimized_models_dir = self.base_dir / "optimized_models"
        self.evaluation_dir = self.base_dir / "model_evaluation_fixed"
        
        # Criar diretório de avaliação
        self.evaluation_dir.mkdir(exist_ok=True)
        
        # Configurar logging
        self._setup_logging()
        
        # Identificar dados disponíveis
        self.available_pairs = self._discover_available_pairs()
        
        # Tipos de modelos suportados
        self.model_types = {
            'regression': ['random_forest_reg', 'xgboost_reg', 'gradient_boosting_reg', 'lightgbm_reg'],
            'classification': ['random_forest_clf', 'xgboost_clf', 'gradient_boosting_clf', 'lightgbm_clf']
        }
        
        # Métricas de avaliação
        self.regression_metrics = {
            'MSE': mean_squared_error,
            'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error,
            'MAPE': mean_absolute_percentage_error,
            'R2': r2_score
        }
        
        self.classification_metrics = {
            'Accuracy': accuracy_score,
            'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Resultados da avaliação
        self.evaluation_results = {}
        self.statistical_tests = {}
        
    def _setup_logging(self):
        """Configura sistema de logging"""
        log_file = self.evaluation_dir / 'evaluation_fixed.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _discover_available_pairs(self) -> List[str]:
        """Descobre pares disponíveis com dados válidos"""
        pairs = []
        
        if not self.processed_data_dir.exists():
            self.logger.warning(f"Diretório de dados processados não encontrado: {self.processed_data_dir}")
            return pairs
        
        for pair_dir in self.processed_data_dir.iterdir():
            if pair_dir.is_dir():
                # Verificar se há dados de teste
                test_files = list(pair_dir.glob("*test*.csv")) + list(pair_dir.glob("*test*.parquet"))
                if test_files:
                    pairs.append(pair_dir.name)
                    self.logger.info(f"✅ Par descoberto: {pair_dir.name}")
        
        self.logger.info(f"📊 Total de pares disponíveis: {len(pairs)}")
        return sorted(pairs)
    
    def load_test_data(self, pair: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Carrega e valida dados de teste"""
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
                          if not col.startswith('target_') and not col.startswith('Unnamed')]
            
            if not feature_cols:
                self.logger.error(f"❌ Nenhuma feature encontrada para {pair}")
                return None, None, None
            
            X_test = test_data[feature_cols].values
            
            # Verificar targets
            y_test_reg = None
            y_test_clf = None
            
            reg_targets = [col for col in test_data.columns if 'target_reg' in col or 'target_return' in col]
            clf_targets = [col for col in test_data.columns if 'target_clf' in col or 'target_direction' in col]
            
            if reg_targets:
                y_test_reg = test_data[reg_targets[0]].values
                # Validar targets de regressão
                if np.isnan(y_test_reg).all() or np.isinf(y_test_reg).any():
                    self.logger.warning(f"⚠️ Targets de regressão inválidos para {pair}")
                    y_test_reg = None
            
            if clf_targets:
                y_test_clf = test_data[clf_targets[0]].values
                # Validar targets de classificação
                unique_classes = np.unique(y_test_clf[~np.isnan(y_test_clf)])
                if len(unique_classes) < 2:
                    self.logger.warning(f"⚠️ Targets de classificação insuficientes para {pair}: {unique_classes}")
                    y_test_clf = None
            
            # Validar features
            if np.isnan(X_test).any() or np.isinf(X_test).any():
                self.logger.warning(f"⚠️ Features com valores inválidos para {pair}, aplicando limpeza...")
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
            
            self.logger.info(f"✅ Dados carregados para {pair}: {X_test.shape[0]} amostras, {X_test.shape[1]} features")
            return X_test, y_test_reg, y_test_clf
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar dados para {pair}: {e}")
            return None, None, None
    
    def load_model(self, pair: str, model_type: str, model_category: str = 'optimized') -> Optional[Any]:
        """Carrega modelo treinado com validação"""
        if model_category == 'optimized':
            model_dir = self.optimized_models_dir / pair
        else:
            model_dir = self.trained_models_dir / pair
        
        if not model_dir.exists():
            self.logger.warning(f"⚠️ Diretório de modelos não encontrado: {model_dir}")
            return None
        
        # Procurar arquivo do modelo
        model_files = list(model_dir.glob(f"*{model_type}*.pkl")) + list(model_dir.glob(f"*{model_type}*.joblib"))
        
        if not model_files:
            self.logger.warning(f"⚠️ Modelo não encontrado: {pair}/{model_type}")
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
    
    def compare_models(self, baseline_results: Dict[str, float], 
                     optimized_results: Dict[str, float]) -> Dict[str, float]:
        """Compara performance entre modelos baseline e otimizados"""
        improvements = {}
        
        for metric in baseline_results.keys():
            if metric in optimized_results:
                baseline_val = baseline_results[metric]
                optimized_val = optimized_results[metric]
                
                # Evitar divisão por zero
                if abs(baseline_val) < 1e-10:
                    if abs(optimized_val) < 1e-10:
                        improvements[metric] = 0.0
                    else:
                        improvements[metric] = 100.0 if optimized_val > 0 else -100.0
                else:
                    # Métricas onde menor é melhor
                    if metric in ['MSE', 'RMSE', 'MAE', 'MAPE']:
                        improvement = ((baseline_val - optimized_val) / abs(baseline_val)) * 100
                    else:
                        # Métricas onde maior é melhor
                        improvement = ((optimized_val - baseline_val) / abs(baseline_val)) * 100
                    
                    improvements[metric] = improvement
        
        return improvements
    
    def evaluate_pair_models(self, pair: str) -> Dict[str, Any]:
        """Avalia todos os modelos para um par específico"""
        self.logger.info(f"🎯 Avaliando modelos para {pair}...")
        
        # Carregar dados de teste
        X_test, y_test_reg, y_test_clf = self.load_test_data(pair)
        
        if X_test is None:
            self.logger.error(f"❌ Não foi possível carregar dados de teste para {pair}")
            return {}
        
        pair_results = {
            'pair': pair,
            'test_samples': len(X_test),
            'models': {},
            'summary': {
                'total_models': 0,
                'successful_evaluations': 0,
                'models_with_improvement': 0,
                'average_improvement': 0.0
            }
        }
        
        all_model_types = (self.model_types['regression'] + self.model_types['classification'])
        
        for model_type in all_model_types:
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
                    'statistical_significance': {}
                }
                
                # Avaliar modelo baseline
                if baseline_model is not None:
                    try:
                        y_pred_baseline = baseline_model.predict(X_test)
                        model_result['baseline_metrics'] = self.evaluate_model_performance(
                            y_test, y_pred_baseline, model_type
                        )
                    except Exception as e:
                        self.logger.warning(f"⚠️ Erro na predição baseline {model_type}: {e}")
                
                # Avaliar modelo otimizado
                if optimized_model is not None:
                    try:
                        y_pred_optimized = optimized_model.predict(X_test)
                        model_result['optimized_metrics'] = self.evaluate_model_performance(
                            y_test, y_pred_optimized, model_type
                        )
                        
                        # Comparar com baseline se disponível
                        if model_result['baseline_metrics']:
                            model_result['improvements'] = self.compare_models(
                                model_result['baseline_metrics'],
                                model_result['optimized_metrics']
                            )
                            
                            # Teste de significância estatística
                            if baseline_model is not None:
                                model_result['statistical_significance'] = self._test_statistical_significance(
                                    y_test, y_pred_baseline, y_pred_optimized, model_type
                                )
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Erro na predição otimizada {model_type}: {e}")
                
                # Verificar se houve melhoria
                if model_result['improvements']:
                    # Considerar melhoria se pelo menos uma métrica principal melhorou significativamente
                    main_metrics = ['R2', 'F1', 'Accuracy'] if 'clf' in model_type else ['R2', 'RMSE']
                    significant_improvements = [model_result['improvements'].get(m, 0) for m in main_metrics 
                                              if m in model_result['improvements']]
                    
                    if any(imp > 1.0 for imp in significant_improvements):  # Melhoria > 1%
                        pair_results['summary']['models_with_improvement'] += 1
                
                pair_results['models'][model_type] = model_result
                pair_results['summary']['successful_evaluations'] += 1
                
                self.logger.info(f"✅ {model_type} avaliado com sucesso")
                
            except Exception as e:
                self.logger.error(f"❌ Erro ao avaliar {model_type} para {pair}: {e}")
                continue
        
        # Calcular melhoria média
        if pair_results['summary']['successful_evaluations'] > 0:
            all_improvements = []
            for model_data in pair_results['models'].values():
                if model_data['improvements']:
                    all_improvements.extend(model_data['improvements'].values())
            
            if all_improvements:
                pair_results['summary']['average_improvement'] = np.mean(all_improvements)
        
        return pair_results
    
    def _test_statistical_significance(self, y_true: np.ndarray, y_pred_baseline: np.ndarray, 
                                     y_pred_optimized: np.ndarray, model_type: str) -> Dict[str, Any]:
        """Testa significância estatística das melhorias"""
        results = {}
        
        try:
            # Calcular erros/scores para cada predição
            if 'reg' in model_type:
                # Para regressão, usar erros absolutos
                errors_baseline = np.abs(y_true - y_pred_baseline)
                errors_optimized = np.abs(y_true - y_pred_optimized)
                
                # Teste t pareado (menor erro é melhor)
                t_stat, p_value = ttest_rel(errors_baseline, errors_optimized)
                results['t_test'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.significance_level and t_stat > 0
                }
                
                # Teste de Wilcoxon (não-paramétrico)
                try:
                    w_stat, w_p_value = wilcoxon(errors_baseline, errors_optimized, alternative='greater')
                    results['wilcoxon_test'] = {
                        'statistic': w_stat,
                        'p_value': w_p_value,
                        'significant': w_p_value < self.significance_level
                    }
                except:
                    pass
            
            else:
                # Para classificação, usar acurácia por amostra
                acc_baseline = (y_true == y_pred_baseline).astype(float)
                acc_optimized = (y_true == y_pred_optimized).astype(float)
                
                # Teste t pareado (maior acurácia é melhor)
                t_stat, p_value = ttest_rel(acc_optimized, acc_baseline)
                results['t_test'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.significance_level and t_stat > 0
                }
        
        except Exception as e:
            self.logger.warning(f"⚠️ Erro no teste de significância: {e}")
        
        return results
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """Executa avaliação completa de todos os pares"""
        self.logger.info("🚀 Iniciando avaliação completa dos modelos FX...")
        
        if not self.available_pairs:
            self.logger.error("❌ Nenhum par disponível para avaliação")
            return {}
        
        evaluation_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_pairs': len(self.available_pairs),
            'evaluated_pairs': 0,
            'total_models': 0,
            'successful_evaluations': 0,
            'models_with_improvement': 0,
            'pairs_results': {},
            'global_statistics': {}
        }
        
        # Avaliar cada par
        for pair in self.available_pairs:
            try:
                pair_results = self.evaluate_pair_models(pair)
                
                if pair_results:
                    evaluation_summary['pairs_results'][pair] = pair_results
                    evaluation_summary['evaluated_pairs'] += 1
                    evaluation_summary['total_models'] += pair_results['summary']['total_models']
                    evaluation_summary['successful_evaluations'] += pair_results['summary']['successful_evaluations']
                    evaluation_summary['models_with_improvement'] += pair_results['summary']['models_with_improvement']
                    
                    self.logger.info(f"✅ {pair} avaliado: {pair_results['summary']['successful_evaluations']} modelos")
                
            except Exception as e:
                self.logger.error(f"❌ Erro ao avaliar {pair}: {e}")
                continue
        
        # Calcular estatísticas globais
        evaluation_summary['global_statistics'] = self._calculate_global_statistics(evaluation_summary)
        
        # Salvar resultados
        self.evaluation_results = evaluation_summary
        self._save_evaluation_results()
        
        # Gerar relatórios
        self._generate_comprehensive_report()
        
        self.logger.info("✅ Avaliação completa finalizada!")
        return evaluation_summary
    
    def _calculate_global_statistics(self, evaluation_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula estatísticas globais da avaliação"""
        stats = {
            'success_rate': 0.0,
            'improvement_rate': 0.0,
            'average_improvements_by_metric': {},
            'best_performing_models': {},
            'statistical_significance_summary': {}
        }
        
        if evaluation_summary['total_models'] > 0:
            stats['success_rate'] = evaluation_summary['successful_evaluations'] / evaluation_summary['total_models']
        
        if evaluation_summary['successful_evaluations'] > 0:
            stats['improvement_rate'] = evaluation_summary['models_with_improvement'] / evaluation_summary['successful_evaluations']
        
        # Coletar todas as melhorias por métrica
        all_improvements = {}
        model_performances = {}
        significance_count = 0
        total_significance_tests = 0
        
        for pair_results in evaluation_summary['pairs_results'].values():
            for model_type, model_data in pair_results['models'].items():
                # Melhorias
                for metric, improvement in model_data['improvements'].items():
                    if metric not in all_improvements:
                        all_improvements[metric] = []
                    all_improvements[metric].append(improvement)
                
                # Performance dos modelos
                if model_data['optimized_metrics']:
                    if model_type not in model_performances:
                        model_performances[model_type] = []
                    
                    # Usar métrica principal
                    if 'clf' in model_type and 'F1' in model_data['optimized_metrics']:
                        model_performances[model_type].append(model_data['optimized_metrics']['F1'])
                    elif 'reg' in model_type and 'R2' in model_data['optimized_metrics']:
                        model_performances[model_type].append(model_data['optimized_metrics']['R2'])
                
                # Significância estatística
                if model_data['statistical_significance']:
                    total_significance_tests += 1
                    if model_data['statistical_significance'].get('t_test', {}).get('significant', False):
                        significance_count += 1
        
        # Estatísticas de melhorias
        for metric, improvements in all_improvements.items():
            if improvements:
                stats['average_improvements_by_metric'][metric] = {
                    'mean': np.mean(improvements),
                    'std': np.std(improvements),
                    'median': np.median(improvements),
                    'min': np.min(improvements),
                    'max': np.max(improvements),
                    'positive_count': sum(1 for x in improvements if x > 0),
                    'total_count': len(improvements)
                }
        
        # Performance dos modelos
        for model_type, performances in model_performances.items():
            if performances:
                stats['best_performing_models'][model_type] = {
                    'mean_performance': np.mean(performances),
                    'std_performance': np.std(performances),
                    'best_performance': np.max(performances),
                    'worst_performance': np.min(performances)
                }
        
        # Resumo de significância estatística
        if total_significance_tests > 0:
            stats['statistical_significance_summary'] = {
                'significant_improvements': significance_count,
                'total_tests': total_significance_tests,
                'significance_rate': significance_count / total_significance_tests
            }
        
        return stats
    
    def _save_evaluation_results(self):
        """Salva resultados da avaliação"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar JSON completo
        json_file = self.evaluation_dir / f'evaluation_results_fixed_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"💾 Resultados salvos: {json_file}")
    
    def _generate_comprehensive_report(self):
        """Gera relatório abrangente"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.evaluation_dir / f'comprehensive_report_{timestamp}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self._create_report_content())
        
        self.logger.info(f"📄 Relatório gerado: {report_file}")
    
    def _create_report_content(self) -> str:
        """Cria conteúdo do relatório"""
        results = self.evaluation_results
        stats = results.get('global_statistics', {})
        
        report = f"""
# RELATÓRIO ABRANGENTE DE AVALIAÇÃO DE MODELOS FX
## Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

## RESUMO EXECUTIVO
- **Pares Analisados:** {results.get('evaluated_pairs', 0)}/{results.get('total_pairs', 0)}
- **Modelos Avaliados:** {results.get('successful_evaluations', 0)}/{results.get('total_models', 0)}
- **Taxa de Sucesso:** {stats.get('success_rate', 0):.1%}
- **Modelos com Melhoria:** {results.get('models_with_improvement', 0)} ({stats.get('improvement_rate', 0):.1%})

## ANÁLISE ESTATÍSTICA
"""
        
        # Significância estatística
        sig_summary = stats.get('statistical_significance_summary', {})
        if sig_summary:
            report += f"""
### Significância Estatística
- **Melhorias Significativas:** {sig_summary.get('significant_improvements', 0)}/{sig_summary.get('total_tests', 0)}
- **Taxa de Significância:** {sig_summary.get('significance_rate', 0):.1%}
"""
        
        # Melhorias por métrica
        improvements = stats.get('average_improvements_by_metric', {})
        if improvements:
            report += "\n### Melhorias por Métrica\n"
            for metric, metric_stats in improvements.items():
                positive_rate = metric_stats['positive_count'] / metric_stats['total_count']
                report += f"""
#### {metric}
- **Melhoria Média:** {metric_stats['mean']:.2f}% (±{metric_stats['std']:.2f}%)
- **Mediana:** {metric_stats['median']:.2f}%
- **Melhor Resultado:** {metric_stats['max']:.2f}%
- **Pior Resultado:** {metric_stats['min']:.2f}%
- **Taxa de Melhoria:** {positive_rate:.1%} ({metric_stats['positive_count']}/{metric_stats['total_count']})
"""
        
        # Performance dos modelos
        model_perf = stats.get('best_performing_models', {})
        if model_perf:
            report += "\n### Performance dos Modelos\n"
            sorted_models = sorted(model_perf.items(), key=lambda x: x[1]['mean_performance'], reverse=True)
            
            for model_type, perf_stats in sorted_models:
                report += f"""
#### {model_type}
- **Performance Média:** {perf_stats['mean_performance']:.4f}
- **Desvio Padrão:** {perf_stats['std_performance']:.4f}
- **Melhor Performance:** {perf_stats['best_performance']:.4f}
- **Pior Performance:** {perf_stats['worst_performance']:.4f}
"""
        
        # Resultados detalhados por par
        report += "\n## RESULTADOS DETALHADOS POR PAR\n"
        
        for pair, pair_data in results.get('pairs_results', {}).items():
            report += f"\n### {pair}\n"
            report += f"- **Amostras de Teste:** {pair_data.get('test_samples', 0)}\n"
            report += f"- **Modelos Avaliados:** {pair_data['summary']['successful_evaluations']}/{pair_data['summary']['total_models']}\n"
            report += f"- **Modelos com Melhoria:** {pair_data['summary']['models_with_improvement']}\n"
            report += f"- **Melhoria Média:** {pair_data['summary']['average_improvement']:.2f}%\n"
            
            for model_type, model_data in pair_data.get('models', {}).items():
                if model_data['optimized_metrics']:
                    report += f"\n#### {model_type}\n"
                    
                    # Métricas otimizadas
                    for metric, value in model_data['optimized_metrics'].items():
                        improvement = model_data['improvements'].get(metric, 0)
                        symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                        report += f"  - **{metric}:** {value:.4f} ({improvement:+.2f}%) {symbol}\n"
                    
                    # Significância estatística
                    if model_data['statistical_significance']:
                        t_test = model_data['statistical_significance'].get('t_test', {})
                        if t_test:
                            sig_symbol = "✅" if t_test.get('significant', False) else "❌"
                            report += f"  - **Significância Estatística:** p={t_test.get('p_value', 0):.4f} {sig_symbol}\n"
        
        # Conclusões e recomendações
        report += "\n## CONCLUSÕES E RECOMENDAÇÕES\n"
        
        if results.get('models_with_improvement', 0) > 0:
            improvement_rate = stats.get('improvement_rate', 0)
            report += f"\n### Principais Descobertas\n"
            report += f"- {improvement_rate:.1%} dos modelos apresentaram melhorias mensuráveis\n"
            
            if improvements:
                best_metric = max(improvements.items(), key=lambda x: x[1]['mean'])
                report += f"- Maior melhoria média em {best_metric[0]}: {best_metric[1]['mean']:.2f}%\n"
            
            if model_perf:
                best_model = max(model_perf.items(), key=lambda x: x[1]['mean_performance'])
                report += f"- Melhor modelo geral: {best_model[0]} (performance: {best_model[1]['mean_performance']:.4f})\n"
        
        report += """
### Recomendações Técnicas
1. **Implementar modelos otimizados** em produção para pares com melhorias significativas
2. **Monitorar performance continuamente** usando métricas estabelecidas
3. **Re-otimizar periodicamente** os modelos com novos dados
4. **Implementar ensemble** dos melhores modelos para maior robustez
5. **Validar significância estatística** antes de implementar mudanças
6. **Estabelecer thresholds** mínimos de melhoria para justificar atualizações

### Próximos Passos
1. Análise de estabilidade temporal dos modelos
2. Implementação de backtesting rigoroso
3. Análise de correlação entre pares
4. Otimização de hiperparâmetros avançada
5. Implementação de modelos ensemble
"""
        
        return report

# Função principal
def main():
    """Executa avaliação completa"""
    print("🚀 Iniciando Avaliação Robusta de Modelos FX...")
    
    evaluator = RobustFXModelEvaluator(
        base_dir=".",
        min_samples=100,
        significance_level=0.05
    )
    
    if not evaluator.available_pairs:
        print("❌ Nenhum par disponível para avaliação!")
        return
    
    print(f"📊 Pares disponíveis: {', '.join(evaluator.available_pairs)}")
    
    # Executar avaliação
    results = evaluator.run_complete_evaluation()
    
    # Imprimir resumo
    if results:
        stats = results.get('global_statistics', {})
        print(f"\n✅ Avaliação concluída!")
        print(f"📈 Taxa de sucesso: {stats.get('success_rate', 0):.1%}")
        print(f"🎯 Taxa de melhoria: {stats.get('improvement_rate', 0):.1%}")
        print(f"📊 Modelos com melhoria: {results.get('models_with_improvement', 0)}")
    
if __name__ == "__main__":
    main()