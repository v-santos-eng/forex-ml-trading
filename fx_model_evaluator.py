#!/usr/bin/env python3
"""
FX Model Performance Evaluator
Avalia e compara a performance dos modelos otimizados vs modelos de linha de base
"""

import os
import json
import pickle
import pandas as pd
import re
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class FXModelEvaluator:
    def __init__(self, base_dir: str = ".", timeframe: str = 'M30'):
        self.base_dir = base_dir
        self.timeframe = timeframe
        self.processed_data_dir = os.path.join(base_dir, f"processed_fx_data_{timeframe}")
        self.trained_models_dir = os.path.join(base_dir, f"trained_models_{timeframe}")
        self.optimized_models_dir = os.path.join(base_dir, f"optimized_models_{timeframe}")
        self.evaluation_dir = os.path.join(base_dir, f"model_evaluation_{timeframe}")
        
        # Criar diretório de avaliação
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
        # Configurar logging
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.evaluation_dir, 'evaluation.log'), encoding='utf-8'),
                logging.StreamHandler()
            ],
            encoding='utf-8'
        )
        self.logger = logging.getLogger(__name__)
        
        # Pares disponíveis para avaliação
        self.available_pairs = self._get_available_pairs()
        
        # Modelos disponíveis
        self.model_types = [
            'random_forest_reg', 'random_forest_clf',
            'xgboost_reg', 'xgboost_clf',
            'gradient_boosting_reg', 'gradient_boosting_clf'
        ]
        
        # Métricas de avaliação
        self.regression_metrics = ['MSE', 'MAE', 'R2']
        self.classification_metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        
        # Resultados da avaliação
        self.evaluation_results = {}
        
    def _get_available_pairs(self) -> List[str]:
        """Identifica pares disponíveis para avaliação.
        Retorna pares que possuam modelos otimizados ou, caso não existam,
        retorna pares com modelos baseline treinados, permitindo geração de
        relatórios somente-baseline (ex.: timeframe M15 antes da otimização)."""
        pairs: set[str] = set()

        # Procurar pares com modelos otimizados
        if os.path.exists(self.optimized_models_dir):
            pairs.update([
                d for d in os.listdir(self.optimized_models_dir)
                if os.path.isdir(os.path.join(self.optimized_models_dir, d))
            ])

        # Se não houver pares otimizados (ou para complementar), usar baseline treinado
        if os.path.exists(self.trained_models_dir):
            pairs.update([
                d for d in os.listdir(self.trained_models_dir)
                if os.path.isdir(os.path.join(self.trained_models_dir, d))
            ])

        return sorted(pairs)
    
    def load_test_data(self, pair: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Carrega dados de teste para um par específico"""
        try:
            # Carregar dados de teste do formato parquet
            test_file = os.path.join(self.processed_data_dir, pair, "test.parquet")
            
            if not os.path.exists(test_file):
                self.logger.warning(f"⚠️ Arquivo de teste não encontrado: {test_file}")
                return None, None, None
            
            df = pd.read_parquet(test_file)
            
            # Excluir colunas não-numéricas e targets das features
            # Excluir: DateTime, colunas que começam com 'Target_' (maiúsculas/minúsculas)
            exclude_cols = ['DateTime'] + [col for col in df.columns if col.lower().startswith('target_')]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            X_full = df[feature_cols]
            
            # Aplicar seleção de features para compatibilidade com modelos treinados (64 features)
            target_for_selection = None
            
            # Primeiro, identificar uma coluna de target para seleção de features
            for col in df.columns:
                if re.match(r'(?i)target_(return|price)_1$', col):
                    target_for_selection = df[col]
                    break
            
            # Se não encontrou target de regressão, usar target de classificação
            if target_for_selection is None:
                for col in df.columns:
                    if re.match(r'(?i)target_direction_1$', col):
                        target_for_selection = df[col]
                        break
            
            # Aplicar seleção de features se temos um target
            if target_for_selection is not None and len(feature_cols) > 64:
                # Calcular correlação absoluta com target
                correlations = X_full.corrwith(target_for_selection).abs()
                # Selecionar top 64 features
                top_features = correlations.nlargest(64).index.tolist()
                X_test = X_full[top_features].values
                feature_cols = top_features
                self.logger.info(f"Selecionadas {len(top_features)} features baseadas em correlação")
            else:
                X_test = X_full.values
            
            # Inicializar variáveis de target
            y_test_reg = None
            y_test_clf = None
            
            # Tentativa 1: colunas padrão em minúsculas
            if 'target_reg' in df.columns:
                y_test_reg = df['target_reg'].values
            if 'target_clf' in df.columns:
                y_test_clf = df['target_clf'].values
            
            # Tentativa 2: detectar padrões utilizados no pipeline de feature engineering
            if y_test_reg is None:
                for col in df.columns:
                    if re.match(r'(?i)target_(return|price)_1$', col):
                        y_test_reg = df[col].values
                        break
            if y_test_clf is None:
                for col in df.columns:
                    if re.match(r'(?i)target_direction_1$', col):
                        y_test_clf = df[col].values
                        break
            
            self.logger.info(f"📊 Dados de teste carregados para {pair}: {X_test.shape[0]} amostras, {X_test.shape[1]} features")
            return X_test, y_test_reg, y_test_clf
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar dados de teste para {pair}: {e}")
            return None, None, None
    
    def load_model(self, pair: str, model_type: str, optimized: bool = True) -> Any:
        """Carrega um modelo específico"""
        try:
            if optimized:
                model_dir = self.optimized_models_dir
                model_file = f"{model_type}_randomized_target_1.pkl"
            else:
                model_dir = self.trained_models_dir
                model_file = f"{model_type}_target_1.pkl"
            
            model_path = os.path.join(model_dir, pair, model_file)
            
            if not os.path.exists(model_path):
                return None
                
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar modelo {model_type} para {pair}: {e}")
            return None
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de regressão"""
        return {
            'MSE': mean_squared_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de classificação"""
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def evaluate_model(self, pair: str, model_type: str) -> Dict[str, Any]:
        """Avalia um modelo específico (baseline vs otimizado)"""
        self.logger.info(f"🔍 Avaliando {model_type} para {pair}...")
        
        # Carregar dados de teste
        X_test, y_test_reg, y_test_clf = self.load_test_data(pair)
        if X_test is None:
            return None
        
        # Carregar modelos
        baseline_model = self.load_model(pair, model_type, optimized=False)
        optimized_model = self.load_model(pair, model_type, optimized=True)
        
        if baseline_model is None and optimized_model is None:
            self.logger.warning(f"⚠️ Nenhum modelo encontrado para {model_type} - {pair}")
            return None
        
        results = {
            'pair': pair,
            'model_type': model_type,
            'baseline': {},
            'optimized': {},
            'improvement': {}
        }
        
        # Determinar se é regressão ou classificação
        is_regression = 'reg' in model_type
        y_test = y_test_reg if is_regression else y_test_clf
        
        # Avaliar modelo baseline
        if baseline_model is not None:
            try:
                y_pred_baseline = baseline_model.predict(X_test)
                if is_regression:
                    results['baseline'] = self.calculate_regression_metrics(y_test, y_pred_baseline)
                else:
                    results['baseline'] = self.calculate_classification_metrics(y_test, y_pred_baseline)
                self.logger.info(f"✅ Modelo baseline avaliado para {model_type} - {pair}")
            except Exception as e:
                self.logger.error(f"❌ Erro ao avaliar modelo baseline: {e}")
        
        # Avaliar modelo otimizado
        if optimized_model is not None:
            try:
                y_pred_optimized = optimized_model.predict(X_test)
                if is_regression:
                    results['optimized'] = self.calculate_regression_metrics(y_test, y_pred_optimized)
                else:
                    results['optimized'] = self.calculate_classification_metrics(y_test, y_pred_optimized)
                self.logger.info(f"✅ Modelo otimizado avaliado para {model_type} - {pair}")
            except Exception as e:
                self.logger.error(f"❌ Erro ao avaliar modelo otimizado: {e}")
        
        # Calcular melhorias
        if results['baseline'] and results['optimized']:
            for metric in results['baseline'].keys():
                baseline_val = results['baseline'][metric]
                optimized_val = results['optimized'][metric]
                
                if baseline_val != 0:
                    if metric in ['MSE', 'MAE']:  # Menor é melhor
                        improvement = ((baseline_val - optimized_val) / baseline_val) * 100
                    else:  # Maior é melhor
                        improvement = ((optimized_val - baseline_val) / baseline_val) * 100
                else:
                    improvement = 0
                
                results['improvement'][metric] = improvement
        
        return results
    
    def evaluate_all_models(self) -> None:
        """Avalia todos os modelos disponíveis"""
        self.logger.info(f"🚀 Iniciando avaliação de modelos para {len(self.available_pairs)} pares no timeframe {self.timeframe}...")
        
        total_evaluations = 0
        successful_evaluations = 0
        
        for pair in self.available_pairs:
            self.logger.info(f"🎯 Avaliando modelos para {pair}...")
            
            pair_results = []
            
            for model_type in self.model_types:
                total_evaluations += 1
                result = self.evaluate_model(pair, model_type)
                
                if result is not None:
                    pair_results.append(result)
                    successful_evaluations += 1
            
            if pair_results:
                self.evaluation_results[pair] = pair_results
                self.logger.info(f"✅ {pair}: {len(pair_results)} modelos avaliados")
        
        self.logger.info(f"📊 Avaliação concluída: {successful_evaluations}/{total_evaluations} modelos avaliados")
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Gera estatísticas resumidas da avaliação"""
        summary = {
            'total_pairs': len(self.available_pairs),
            'total_models_evaluated': 0,
            'models_with_improvement': 0,
            'average_improvements': {},
            'best_performing_pairs': {},
            'best_performing_models': {}
        }
        
        all_improvements = {}
        model_performance = {}
        
        for pair, results in self.evaluation_results.items():
            for result in results:
                summary['total_models_evaluated'] += 1
                
                if result['improvement']:
                    summary['models_with_improvement'] += 1
                    
                    # Coletar melhorias por métrica
                    for metric, improvement in result['improvement'].items():
                        if metric not in all_improvements:
                            all_improvements[metric] = []
                        all_improvements[metric].append(improvement)
                    
                    # Coletar performance por modelo
                    model_key = result['model_type']
                    if model_key not in model_performance:
                        model_performance[model_key] = []
                    
                    # Usar F1 para classificação, R2 para regressão como métrica principal
                    if 'clf' in model_key and 'F1' in result['optimized']:
                        model_performance[model_key].append(result['optimized']['F1'])
                    elif 'reg' in model_key and 'R2' in result['optimized']:
                        model_performance[model_key].append(result['optimized']['R2'])
        
        # Calcular melhorias médias
        for metric, improvements in all_improvements.items():
            summary['average_improvements'][metric] = {
                'mean': np.mean(improvements),
                'std': np.std(improvements),
                'min': np.min(improvements),
                'max': np.max(improvements)
            }
        
        # Identificar melhores modelos
        for model_type, performances in model_performance.items():
            if performances:
                summary['best_performing_models'][model_type] = {
                    'mean_performance': np.mean(performances),
                    'std_performance': np.std(performances),
                    'best_performance': np.max(performances)
                }
        
        return summary
    
    def create_visualizations(self) -> None:
        """Cria visualizações dos resultados"""
        self.logger.info("📈 Gerando visualizações...")
        
        # Verificar se há dados para visualizar
        if not self.evaluation_results:
            self.logger.warning("⚠️ Nenhum resultado de avaliação disponível para visualização")
            return
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        try:
            # 1. Gráfico de melhorias por métrica
            self._plot_improvements_by_metric()
            
            # 2. Gráfico de performance por modelo
            self._plot_model_performance()
            
            # 3. Heatmap de performance por par
            self._plot_performance_heatmap()
            
            # 4. Distribuição de melhorias
            self._plot_improvement_distribution()
            
            self.logger.info("✅ Visualizações salvas em model_evaluation/")
        except Exception as e:
            self.logger.error(f"❌ Erro ao criar visualizações: {e}")
            self.logger.info("📊 Continuando sem visualizações...")
    
    def _plot_improvements_by_metric(self) -> None:
        """Plota melhorias por métrica"""
        improvements_data = []
        
        for pair, results in self.evaluation_results.items():
            for result in results:
                if result['improvement']:
                    for metric, improvement in result['improvement'].items():
                        improvements_data.append({
                            'Pair': pair,
                            'Model': result['model_type'],
                            'Metric': metric,
                            'Improvement': improvement
                        })
        
        if improvements_data:
            df = pd.DataFrame(improvements_data)
            
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=df, x='Metric', y='Improvement')
            plt.title('Distribuição de Melhorias por Métrica', fontsize=16, fontweight='bold')
            plt.ylabel('Melhoria (%)', fontsize=12)
            plt.xlabel('Métrica', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.evaluation_dir, 'improvements_by_metric.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_model_performance(self) -> None:
        """Plota performance por tipo de modelo"""
        performance_data = []
        
        for pair, results in self.evaluation_results.items():
            for result in results:
                if result['optimized']:
                    # Usar métrica principal baseada no tipo de modelo
                    if 'clf' in result['model_type'] and 'F1' in result['optimized']:
                        performance_data.append({
                            'Pair': pair,
                            'Model': result['model_type'],
                            'Performance': result['optimized']['F1'],
                            'Type': 'Classification'
                        })
                    elif 'reg' in result['model_type'] and 'R2' in result['optimized']:
                        performance_data.append({
                            'Pair': pair,
                            'Model': result['model_type'],
                            'Performance': result['optimized']['R2'],
                            'Type': 'Regression'
                        })
        
        if performance_data:
            df = pd.DataFrame(performance_data)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Classificação
            clf_data = df[df['Type'] == 'Classification']
            if not clf_data.empty:
                sns.boxplot(data=clf_data, x='Model', y='Performance', ax=ax1)
                ax1.set_title('Performance dos Modelos de Classificação (F1 Score)', fontweight='bold')
                ax1.set_ylabel('F1 Score')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
            
            # Regressão
            reg_data = df[df['Type'] == 'Regression']
            if not reg_data.empty:
                sns.boxplot(data=reg_data, x='Model', y='Performance', ax=ax2)
                ax2.set_title('Performance dos Modelos de Regressão (R² Score)', fontweight='bold')
                ax2.set_ylabel('R² Score')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.evaluation_dir, 'model_performance.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_performance_heatmap(self) -> None:
        """Cria heatmap de performance por par e modelo"""
        # Preparar dados para heatmap
        heatmap_data = {}
        
        for pair, results in self.evaluation_results.items():
            heatmap_data[pair] = {}
            for result in results:
                if result['optimized']:
                    model_type = result['model_type']
                    # Usar métrica principal
                    if 'clf' in model_type and 'F1' in result['optimized']:
                        heatmap_data[pair][model_type] = result['optimized']['F1']
                    elif 'reg' in model_type and 'R2' in result['optimized']:
                        heatmap_data[pair][model_type] = result['optimized']['R2']
        
        if heatmap_data:
            df_heatmap = pd.DataFrame(heatmap_data).T
            
            # Verificar se há dados suficientes
            if df_heatmap.empty or df_heatmap.isna().all().all():
                self.logger.warning("⚠️ Dados insuficientes para criar heatmap")
                return
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(df_heatmap, annot=True, cmap='RdYlGn', center=0.5, 
                       fmt='.3f', cbar_kws={'label': 'Performance Score'})
            plt.title('Heatmap de Performance por Par e Modelo', fontsize=16, fontweight='bold')
            plt.ylabel('Par de Moedas', fontsize=12)
            plt.xlabel('Tipo de Modelo', fontsize=12)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(self.evaluation_dir, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            self.logger.warning("⚠️ Nenhum dado de performance disponível para heatmap")
    
    def _plot_improvement_distribution(self) -> None:
        """Plota distribuição de melhorias"""
        all_improvements = []
        
        for pair, results in self.evaluation_results.items():
            for result in results:
                if result['improvement']:
                    for metric, improvement in result['improvement'].items():
                        all_improvements.append(improvement)
        
        if all_improvements:
            plt.figure(figsize=(10, 6))
            plt.hist(all_improvements, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(np.mean(all_improvements), color='red', linestyle='--', 
                       label=f'Média: {np.mean(all_improvements):.2f}%')
            plt.axvline(0, color='black', linestyle='-', alpha=0.5, label='Sem melhoria')
            plt.title('Distribuição de Melhorias dos Modelos Otimizados', fontsize=16, fontweight='bold')
            plt.xlabel('Melhoria (%)', fontsize=12)
            plt.ylabel('Frequência', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.evaluation_dir, 'improvement_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_detailed_report(self) -> None:
        """Gera relatório detalhado em formato JSON e texto"""
        self.logger.info("📝 Gerando relatório detalhado...")
        
        # Gerar estatísticas resumidas
        summary = self.generate_summary_statistics()
        
        # Relatório completo
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'summary': summary,
            'detailed_results': self.evaluation_results,
            'methodology': {
                'test_data': 'Conjunto de teste separado (20% dos dados)',
                'metrics_regression': self.regression_metrics,
                'metrics_classification': self.classification_metrics,
                'comparison': 'Modelos otimizados vs modelos baseline'
            }
        }
        
        # Salvar relatório JSON
        report_file = os.path.join(self.evaluation_dir, f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Gerar relatório em texto
        self._generate_text_report(report, summary)
        
        self.logger.info(f"✅ Relatório salvo: {report_file}")
    
    def _generate_text_report(self, report: Dict, summary: Dict) -> None:
        """Gera relatório em formato texto legível"""
        improvement_percentage = 0
        if summary['total_models_evaluated'] > 0:
            improvement_percentage = summary['models_with_improvement']/summary['total_models_evaluated']*100
        
        report_text = f"""
# RELATÓRIO DE AVALIAÇÃO DE MODELOS FX
## Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

## RESUMO EXECUTIVO
- **Total de Pares Avaliados:** {summary['total_pairs']}
- **Total de Modelos Avaliados:** {summary['total_models_evaluated']}
- **Modelos com Melhoria:** {summary['models_with_improvement']} ({improvement_percentage:.1f}%)

## MELHORIAS MÉDIAS POR MÉTRICA
"""
        
        for metric, stats in summary['average_improvements'].items():
            report_text += f"\n### {metric}\n"
            report_text += f"- Melhoria Média: {stats['mean']:.2f}%\n"
            report_text += f"- Desvio Padrão: {stats['std']:.2f}%\n"
            report_text += f"- Melhor Melhoria: {stats['max']:.2f}%\n"
            report_text += f"- Pior Resultado: {stats['min']:.2f}%\n"
        
        report_text += "\n## PERFORMANCE DOS MODELOS\n"
        
        for model_type, stats in summary['best_performing_models'].items():
            report_text += f"\n### {model_type}\n"
            report_text += f"- Performance Média: {stats['mean_performance']:.4f}\n"
            report_text += f"- Desvio Padrão: {stats['std_performance']:.4f}\n"
            report_text += f"- Melhor Performance: {stats['best_performance']:.4f}\n"
        
        report_text += "\n## RESULTADOS DETALHADOS POR PAR\n"
        
        for pair, results in self.evaluation_results.items():
            report_text += f"\n### {pair}\n"
            
            for result in results:
                model_type = result['model_type']
                report_text += f"\n#### {model_type}\n"
                
                if result['baseline']:
                    report_text += "**Baseline:**\n"
                    for metric, value in result['baseline'].items():
                        report_text += f"  - {metric}: {value:.4f}\n"
                
                if result['optimized']:
                    report_text += "**Otimizado:**\n"
                    for metric, value in result['optimized'].items():
                        report_text += f"  - {metric}: {value:.4f}\n"
                
                if result['improvement']:
                    report_text += "**Melhorias:**\n"
                    for metric, improvement in result['improvement'].items():
                        symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                        report_text += f"  - {metric}: {improvement:.2f}% {symbol}\n"
        
        report_text += "\n## CONCLUSÕES\n"
        report_text += "\n### Principais Descobertas:\n"
        
        if summary['models_with_improvement'] > 0:
            improvement_rate = summary['models_with_improvement'] / summary['total_models_evaluated'] * 100
            report_text += f"- {improvement_rate:.1f}% dos modelos apresentaram melhorias após otimização\n"
            
            # Encontrar melhor métrica
            best_metric = max(summary['average_improvements'].items(), key=lambda x: x[1]['mean'])
            report_text += f"- Maior melhoria média em {best_metric[0]}: {best_metric[1]['mean']:.2f}%\n"
            
            # Encontrar melhor modelo
            if summary['best_performing_models']:
                best_model = max(summary['best_performing_models'].items(), key=lambda x: x[1]['mean_performance'])
                report_text += f"- Melhor modelo geral: {best_model[0]} (performance: {best_model[1]['mean_performance']:.4f})\n"
        
        report_text += "\n### Recomendações:\n"
        report_text += "- Utilizar modelos otimizados em produção\n"
        report_text += "- Monitorar performance continuamente\n"
        report_text += "- Considerar re-otimização periódica\n"
        report_text += "- Implementar ensemble dos melhores modelos\n"
        
        # Salvar relatório texto
        text_report_file = os.path.join(self.evaluation_dir, f'evaluation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    def run_complete_evaluation(self) -> None:
        """Executa avaliação completa"""
        self.logger.info("🚀 Iniciando avaliação completa dos modelos...")
        
        # 1. Avaliar todos os modelos
        self.evaluate_all_models()
        
        # 2. Gerar visualizações
        self.create_visualizations()
        
        # 3. Gerar relatório detalhado
        self.generate_detailed_report()
        
        # 4. Imprimir resumo
        self.print_summary()
        
        self.logger.info("✅ Avaliação completa finalizada!")
    
    def print_summary(self) -> None:
        """Imprime resumo da avaliação"""
        summary = self.generate_summary_statistics()
        
        print("\n" + "="*80)
        print("📊 RESUMO DA AVALIAÇÃO DE MODELOS FX")
        print("="*80)
        print(f"📈 Pares Avaliados: {summary['total_pairs']}")
        print(f"🤖 Modelos Avaliados: {summary['total_models_evaluated']}")
        
        improvement_percentage = 0
        if summary['total_models_evaluated'] > 0:
            improvement_percentage = summary['models_with_improvement']/summary['total_models_evaluated']*100
        
        print(f"✅ Modelos com Melhoria: {summary['models_with_improvement']} ({improvement_percentage:.1f}%)")
        
        if summary['average_improvements']:
            print("\n📈 MELHORIAS MÉDIAS:")
            for metric, stats in summary['average_improvements'].items():
                print(f"  {metric}: {stats['mean']:.2f}% (±{stats['std']:.2f}%)")
        
        if summary['best_performing_models']:
            print("\n🏆 MELHORES MODELOS:")
            sorted_models = sorted(summary['best_performing_models'].items(), 
                                 key=lambda x: x[1]['mean_performance'], reverse=True)
            for model_type, stats in sorted_models[:3]:
                print(f"  {model_type}: {stats['mean_performance']:.4f}")
        
        print("\n📁 Arquivos Gerados:")
        print(f"  - Relatórios: {self.evaluation_dir}/")
        print(f"  - Visualizações: {self.evaluation_dir}/*.png")
        print("="*80)

def main():
    """Função principal"""
    import argparse
    parser = argparse.ArgumentParser(description='FX Model Performance Evaluator')
    parser.add_argument('--timeframe', type=str, default='M30', choices=['M5', 'M15', 'M30'],
                        help='Timeframe para avaliação (default: M30)')
    args = parser.parse_args()

    print(f"🚀 Iniciando Avaliação de Modelos FX para o timeframe {args.timeframe}...")
    
    # Criar avaliador
    evaluator = FXModelEvaluator(timeframe=args.timeframe)
    
    # Verificar se há modelos para avaliar
    if not evaluator.available_pairs:
        print(f"❌ Nenhum modelo otimizado encontrado para avaliação no timeframe {args.timeframe}!")
        return
    
    print(f"📊 Encontrados {len(evaluator.available_pairs)} pares para avaliação: {', '.join(evaluator.available_pairs)}")
    
    # Executar avaliação completa
    evaluator.run_complete_evaluation()
    
    print("\n✅ Avaliação concluída com sucesso!")

if __name__ == "__main__":
    main()