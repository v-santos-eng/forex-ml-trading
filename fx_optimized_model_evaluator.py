#!/usr/bin/env python3
"""
FX Optimized Model Performance Evaluator
Avalia e compara a performance dos modelos otimizados vs modelos de linha de base
Apenas para os modelos que foram realmente otimizados
"""

import os
import json
import glob
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class FXOptimizedModelEvaluator:
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.processed_data_dir = os.path.join(base_dir, "processed_fx_data")
        self.trained_models_dir = os.path.join(base_dir, "trained_models")
        self.optimized_models_dir = os.path.join(base_dir, "optimized_models")
        self.evaluation_dir = os.path.join(base_dir, "model_evaluation")
        
        # Criar diretório de avaliação
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
        # Pares com modelos otimizados
        self.optimized_pairs = ['AUDJPY', 'EURJPY', 'GBPJPY', 'USDJPY', 'USDNOK', 'USDSEK']
        
        # Modelos que foram otimizados
        self.optimized_model_types = [
            'random_forest_reg', 'random_forest_clf',
            'xgboost_reg', 'xgboost_clf'
        ]
        
        # Resultados da avaliação
        self.evaluation_results = {}
        
        print(f"🎯 Avaliando {len(self.optimized_pairs)} pares com modelos otimizados")
        print(f"🤖 Tipos de modelos: {', '.join(self.optimized_model_types)}")
        
    def load_test_data(self, pair: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Carrega dados de teste para um par específico"""
        try:
            test_file = os.path.join(self.processed_data_dir, pair, "test.parquet")
            
            if not os.path.exists(test_file):
                print(f"⚠️ Arquivo de teste não encontrado: {test_file}")
                return None, None, None
            
            test_data = pd.read_parquet(test_file)
            
            # Separar features e targets (mesma lógica do treinamento)
            exclude_cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'] + \
                          [col for col in test_data.columns if col.startswith('Target_')]
            
            feature_cols = [col for col in test_data.columns if col not in exclude_cols]
            X_test = test_data[feature_cols].values
            
            y_test_reg = None
            y_test_clf = None
            
            # Usar Target_Return_1 para regressão e Target_Direction_1 para classificação
            if 'Target_Return_1' in test_data.columns:
                y_test_reg = test_data['Target_Return_1'].values
            
            if 'Target_Direction_1' in test_data.columns:
                y_test_clf = test_data['Target_Direction_1'].values
            
            print(f"📊 {pair}: {X_test.shape[0]} amostras de teste, {X_test.shape[1]} features")
            return X_test, y_test_reg, y_test_clf
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados de teste para {pair}: {e}")
            return None, None, None
    
    def load_baseline_model(self, pair: str, model_type: str) -> Any:
        """Carrega modelo baseline"""
        try:
            model_path = os.path.join(self.trained_models_dir, pair, f"{model_type}_target_1.pkl")
            
            if not os.path.exists(model_path):
                return None
                
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            return model
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo baseline {model_type} para {pair}: {e}")
            return None
    
    def load_optimized_model(self, pair: str, model_type: str):
        """Carrega modelo otimizado, detectando dinamicamente o método de otimização no nome do arquivo"""
        try:
            model_dir = os.path.join(self.optimized_models_dir, pair)
            pattern = f"{model_type}_*_target_1.pkl"
            candidates = glob.glob(os.path.join(model_dir, pattern))
            if not candidates:
                return None
            # Caso haja múltiplos, escolher o mais recente
            candidates.sort(key=os.path.getmtime, reverse=True)
            model_path = candidates[0]
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"❌ Erro ao carregar modelo otimizado {model_type} para {pair}: {e}")
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
    
    def evaluate_model_pair(self, pair: str, model_type: str) -> Dict[str, Any]:
        """Avalia um par de modelos (baseline vs otimizado) com alinhamento exato de features por modelo"""
        print(f"🔍 Avaliando {model_type} para {pair}...")

        # Helpers locais para obter lista de features e preparar X
        def _load_json(path: str):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None

        def _get_feature_list(model_path: str, pair: str, df: pd.DataFrame) -> List[str]:
            # 1) Tenta metadados específicos do modelo
            meta_path = os.path.splitext(model_path)[0] + "_metadata.json"
            feature_keys = ["feature_columns", "features", "feature_names", "features_used"]
            if os.path.exists(meta_path):
                meta = _load_json(meta_path)
                if isinstance(meta, dict):
                    for k in feature_keys:
                        if k in meta and isinstance(meta[k], list) and meta[k]:
                            return list(meta[k])
            # 2) Fallback: metadados do processamento
            proc_meta_path = os.path.join(self.processed_data_dir, pair, "metadata.json")
            if os.path.exists(proc_meta_path):
                pmeta = _load_json(proc_meta_path)
                if isinstance(pmeta, dict):
                    cols = pmeta.get("feature_columns")
                    if isinstance(cols, list) and cols:
                        return list(cols)
            # 3) Heurística: todas numéricas não-Target_ e não OHLCV
            exclude_cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
            candidates = [
                c for c in df.columns
                if (c not in exclude_cols) and (not c.startswith('Target_')) and pd.api.types.is_numeric_dtype(df[c])
            ]
            return candidates

        def _ensure_columns_and_order(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
            # Garante colunas ausentes com zeros e ordena
            for c in feature_list:
                if c not in df.columns:
                    df[c] = 0.0
            return df[feature_list].copy()

        try:
            # Carregar dados de teste bruto
            test_file = os.path.join(self.processed_data_dir, pair, "test.parquet")
            if not os.path.exists(test_file):
                print(f"⚠️ Arquivo de teste não encontrado: {test_file}")
                return None
            df = pd.read_parquet(test_file)

            # Definir alvo conforme tipo do modelo
            is_regression = 'reg' in model_type
            target_col = 'Target_Return_1' if is_regression else 'Target_Direction_1'
            if target_col not in df.columns:
                print(f"⚠️ Target {target_col} não encontrado para {model_type} - {pair}")
                return None
            y_series = df[target_col].copy()

            # Carregar modelos e caminhos (para localizar metadados)
            baseline_model_path = os.path.join(self.trained_models_dir, pair, f"{model_type}_target_1.pkl")
            # Localiza qualquer modelo otimizado independentemente do método
            opt_pattern = os.path.join(self.optimized_models_dir, pair, f"{model_type}_*_target_1.pkl")
            opt_files = glob.glob(opt_pattern)
            optimized_model_path = opt_files[0] if opt_files else None
            if optimized_model_path is None:
                print(f"⚠️ Modelo otimizado não encontrado para {model_type} - {pair}")
                return None

            baseline_model = self.load_baseline_model(pair, model_type)
            optimized_model = self.load_optimized_model(pair, model_type)
            if baseline_model is None or optimized_model is None:
                print(f"⚠️ Modelos não encontrados para {model_type} - {pair}")
                return None

            # Obter listas de features específicas de cada modelo
            feat_baseline = _get_feature_list(baseline_model_path, pair, df)
            feat_optimized = _get_feature_list(optimized_model_path, pair, df)

            if not feat_baseline or not feat_optimized:
                print(f"⚠️ Não foi possível determinar feature set para {model_type} - {pair}")
                return None

            # Garantir que todas as colunas necessárias existam antes de filtrar NA
            for c in set(feat_baseline) | set(feat_optimized):
                if c not in df.columns:
                    df[c] = 0.0

            # Construir máscara de linhas válidas (sem NA nas features necessárias e no alvo)
            union_feats = list(dict.fromkeys(list(feat_baseline) + list(feat_optimized)))
            valid_mask = (~df[union_feats].isna().any(axis=1)) & (~y_series.isna())
            df_valid = df.loc[valid_mask].copy()
            y_valid = y_series.loc[valid_mask].values

            # Preparar matrizes X exatamente com as features esperadas por cada modelo
            X_baseline = _ensure_columns_and_order(df_valid, feat_baseline).values
            X_optimized = _ensure_columns_and_order(df_valid, feat_optimized).values

            results = {
                'pair': pair,
                'model_type': model_type,
                'baseline': {},
                'optimized': {},
                'improvement': {},
                'test_samples': len(y_valid)
            }

            # Predição e métricas
            y_pred_baseline = baseline_model.predict(X_baseline)
            y_pred_optimized = optimized_model.predict(X_optimized)

            if is_regression:
                results['baseline'] = self.calculate_regression_metrics(y_valid, y_pred_baseline)
                results['optimized'] = self.calculate_regression_metrics(y_valid, y_pred_optimized)
            else:
                results['baseline'] = self.calculate_classification_metrics(y_valid, y_pred_baseline)
                results['optimized'] = self.calculate_classification_metrics(y_valid, y_pred_optimized)

            # Calcular melhorias
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

            print(f"✅ {pair} - {model_type}: Avaliação concluída")
            return results

        except Exception as e:
            print(f"❌ Erro na avaliação de {model_type} - {pair}: {e}")
            return None
    
    def evaluate_all_models(self) -> None:
        """Avalia todos os modelos otimizados"""
        print(f"\n🚀 Iniciando avaliação completa...")
        print("="*60)
        
        total_evaluations = 0
        successful_evaluations = 0
        
        for pair in self.optimized_pairs:
            print(f"\n🎯 Avaliando modelos para {pair}...")
            
            pair_results = []
            
            for model_type in self.optimized_model_types:
                total_evaluations += 1
                result = self.evaluate_model_pair(pair, model_type)
                
                if result is not None:
                    pair_results.append(result)
                    successful_evaluations += 1
            
            if pair_results:
                self.evaluation_results[pair] = pair_results
                print(f"✅ {pair}: {len(pair_results)}/{len(self.optimized_model_types)} modelos avaliados")
        
        print(f"\n📊 Avaliação concluída: {successful_evaluations}/{total_evaluations} modelos avaliados")
        print("="*60)
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Gera estatísticas resumidas"""
        summary = {
            'total_pairs': len(self.optimized_pairs),
            'total_models_evaluated': 0,
            'models_with_improvement': 0,
            'average_improvements': {},
            'best_improvements': {},
            'model_performance': {},
            'pair_performance': {}
        }
        
        all_improvements = {}
        model_scores = {}
        pair_scores = {}
        
        for pair, results in self.evaluation_results.items():
            pair_improvements = []
            
            for result in results:
                summary['total_models_evaluated'] += 1
                
                model_type = result['model_type']
                
                # Coletar scores dos modelos
                if model_type not in model_scores:
                    model_scores[model_type] = {'baseline': [], 'optimized': []}
                
                # Usar métrica principal
                main_metric = 'R2' if 'reg' in model_type else 'F1'
                
                if main_metric in result['baseline']:
                    model_scores[model_type]['baseline'].append(result['baseline'][main_metric])
                if main_metric in result['optimized']:
                    model_scores[model_type]['optimized'].append(result['optimized'][main_metric])
                
                # Processar melhorias
                if result['improvement']:
                    has_improvement = any(imp > 0 for imp in result['improvement'].values())
                    if has_improvement:
                        summary['models_with_improvement'] += 1
                    
                    for metric, improvement in result['improvement'].items():
                        if metric not in all_improvements:
                            all_improvements[metric] = []
                        all_improvements[metric].append(improvement)
                        pair_improvements.append(improvement)
            
            # Performance por par
            if pair_improvements:
                pair_scores[pair] = {
                    'avg_improvement': np.mean(pair_improvements),
                    'best_improvement': np.max(pair_improvements),
                    'models_evaluated': len(results)
                }
        
        # Calcular estatísticas de melhoria
        for metric, improvements in all_improvements.items():
            summary['average_improvements'][metric] = {
                'mean': np.mean(improvements),
                'std': np.std(improvements),
                'min': np.min(improvements),
                'max': np.max(improvements),
                'positive_count': sum(1 for x in improvements if x > 0)
            }
        
        # Performance dos modelos
        for model_type, scores in model_scores.items():
            if scores['baseline'] and scores['optimized']:
                summary['model_performance'][model_type] = {
                    'baseline_avg': np.mean(scores['baseline']),
                    'optimized_avg': np.mean(scores['optimized']),
                    'improvement_avg': np.mean(scores['optimized']) - np.mean(scores['baseline'])
                }
        
        summary['pair_performance'] = pair_scores
        
        return summary
    
    def create_visualizations(self) -> None:
        """Cria visualizações dos resultados"""
        print("\n📈 Gerando visualizações...")
        
        if not self.evaluation_results:
            print("⚠️ Nenhum resultado disponível para visualização")
            return
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        try:
            # 1. Comparação Baseline vs Otimizado
            self._plot_baseline_vs_optimized()
            
            # 2. Melhorias por modelo
            self._plot_improvements_by_model()
            
            # 3. Performance por par
            self._plot_performance_by_pair()
            
            # 4. Distribuição de melhorias
            self._plot_improvement_distribution()
            
            print("✅ Visualizações salvas em model_evaluation/")
            
        except Exception as e:
            print(f"❌ Erro ao criar visualizações: {e}")
    
    def _plot_baseline_vs_optimized(self) -> None:
        """Compara performance baseline vs otimizada"""
        comparison_data = []
        
        for pair, results in self.evaluation_results.items():
            for result in results:
                model_type = result['model_type']
                main_metric = 'R2' if 'reg' in model_type else 'F1'
                
                if main_metric in result['baseline'] and main_metric in result['optimized']:
                    comparison_data.extend([
                        {'Pair': pair, 'Model': model_type, 'Type': 'Baseline', 'Score': result['baseline'][main_metric]},
                        {'Pair': pair, 'Model': model_type, 'Type': 'Optimized', 'Score': result['optimized'][main_metric]}
                    ])
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Comparação: Modelos Baseline vs Otimizados', fontsize=16, fontweight='bold')
            
            models = df['Model'].unique()
            
            for i, model in enumerate(models):
                ax = axes[i//2, i%2]
                model_data = df[df['Model'] == model]
                
                sns.barplot(data=model_data, x='Pair', y='Score', hue='Type', ax=ax)
                ax.set_title(f'{model}', fontweight='bold')
                ax.set_xlabel('Par de Moedas')
                ax.set_ylabel('Score')
                ax.tick_params(axis='x', rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.evaluation_dir, 'baseline_vs_optimized.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_improvements_by_model(self) -> None:
        """Plota melhorias por tipo de modelo"""
        improvement_data = []
        
        for pair, results in self.evaluation_results.items():
            for result in results:
                if result['improvement']:
                    for metric, improvement in result['improvement'].items():
                        improvement_data.append({
                            'Pair': pair,
                            'Model': result['model_type'],
                            'Metric': metric,
                            'Improvement': improvement
                        })
        
        if improvement_data:
            df = pd.DataFrame(improvement_data)
            
            plt.figure(figsize=(14, 8))
            sns.boxplot(data=df, x='Model', y='Improvement', hue='Metric')
            plt.title('Distribuição de Melhorias por Tipo de Modelo', fontsize=16, fontweight='bold')
            plt.ylabel('Melhoria (%)', fontsize=12)
            plt.xlabel('Tipo de Modelo', fontsize=12)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Sem melhoria')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.evaluation_dir, 'improvements_by_model.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_performance_by_pair(self) -> None:
        """Plota performance por par de moedas"""
        pair_data = []
        
        for pair, results in self.evaluation_results.items():
            improvements = []
            for result in results:
                if result['improvement']:
                    improvements.extend(result['improvement'].values())
            
            if improvements:
                pair_data.append({
                    'Pair': pair,
                    'Avg_Improvement': np.mean(improvements),
                    'Max_Improvement': np.max(improvements),
                    'Models_Count': len(results)
                })
        
        if pair_data:
            df = pd.DataFrame(pair_data)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Melhoria média por par
            sns.barplot(data=df, x='Pair', y='Avg_Improvement', ax=ax1)
            ax1.set_title('Melhoria Média por Par de Moedas', fontweight='bold')
            ax1.set_ylabel('Melhoria Média (%)')
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Melhor melhoria por par
            sns.barplot(data=df, x='Pair', y='Max_Improvement', ax=ax2)
            ax2.set_title('Melhor Melhoria por Par de Moedas', fontweight='bold')
            ax2.set_ylabel('Melhor Melhoria (%)')
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.evaluation_dir, 'performance_by_pair.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_improvement_distribution(self) -> None:
        """Plota distribuição geral de melhorias"""
        all_improvements = []
        
        for pair, results in self.evaluation_results.items():
            for result in results:
                if result['improvement']:
                    all_improvements.extend(result['improvement'].values())
        
        if all_improvements:
            plt.figure(figsize=(12, 6))
            
            plt.hist(all_improvements, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(np.mean(all_improvements), color='red', linestyle='--', linewidth=2,
                       label=f'Média: {np.mean(all_improvements):.2f}%')
            plt.axvline(0, color='black', linestyle='-', alpha=0.5, label='Sem melhoria')
            
            positive_count = sum(1 for x in all_improvements if x > 0)
            total_count = len(all_improvements)
            
            plt.title(f'Distribuição de Melhorias dos Modelos Otimizados\n'
                     f'{positive_count}/{total_count} ({positive_count/total_count*100:.1f}%) com melhorias positivas', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Melhoria (%)', fontsize=12)
            plt.ylabel('Frequência', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.evaluation_dir, 'improvement_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_detailed_report(self) -> None:
        """Gera relatório detalhado"""
        print("\n📝 Gerando relatório detalhado...")
        
        summary = self.generate_summary_statistics()
        
        # Relatório completo
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'summary': summary,
            'detailed_results': self.evaluation_results,
            'methodology': {
                'pairs_evaluated': self.optimized_pairs,
                'models_evaluated': self.optimized_model_types,
                'test_data_source': 'Conjunto de teste separado (formato parquet)',
                'comparison_method': 'Modelos otimizados vs modelos baseline'
            }
        }
        
        # Salvar relatório JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.evaluation_dir, f'optimized_evaluation_report_{timestamp}.json')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Gerar relatório em texto
        self._generate_text_report(summary, timestamp)
        
        print(f"✅ Relatório salvo: {report_file}")
    
    def _generate_text_report(self, summary: Dict, timestamp: str) -> None:
        """Gera relatório em formato texto"""
        improvement_rate = 0
        if summary['total_models_evaluated'] > 0:
            improvement_rate = summary['models_with_improvement'] / summary['total_models_evaluated'] * 100
        
        report_text = f"""
# RELATÓRIO DE AVALIAÇÃO - MODELOS FX OTIMIZADOS
## Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

## RESUMO EXECUTIVO
- **Pares Avaliados:** {summary['total_pairs']} ({', '.join(self.optimized_pairs)})
- **Modelos Avaliados:** {summary['total_models_evaluated']}
- **Modelos com Melhoria:** {summary['models_with_improvement']} ({improvement_rate:.1f}%)
- **Tipos de Modelos:** {', '.join(self.optimized_model_types)}

## MELHORIAS POR MÉTRICA
"""
        
        if summary['average_improvements']:
            for metric, stats in summary['average_improvements'].items():
                positive_pct = (stats['positive_count'] / summary['total_models_evaluated']) * 100
                report_text += f"""
### {metric}
- **Melhoria Média:** {stats['mean']:.2f}% (±{stats['std']:.2f}%)
- **Melhor Melhoria:** {stats['max']:.2f}%
- **Pior Resultado:** {stats['min']:.2f}%
- **Modelos com Melhoria:** {stats['positive_count']}/{summary['total_models_evaluated']} ({positive_pct:.1f}%)
"""
        
        report_text += "\n## PERFORMANCE DOS MODELOS\n"
        
        if summary['model_performance']:
            for model_type, perf in summary['model_performance'].items():
                report_text += f"""
### {model_type}
- **Baseline Médio:** {perf['baseline_avg']:.4f}
- **Otimizado Médio:** {perf['optimized_avg']:.4f}
- **Melhoria Absoluta:** {perf['improvement_avg']:.4f}
"""
        
        report_text += "\n## PERFORMANCE POR PAR\n"
        
        if summary['pair_performance']:
            for pair, perf in summary['pair_performance'].items():
                report_text += f"""
### {pair}
- **Melhoria Média:** {perf['avg_improvement']:.2f}%
- **Melhor Melhoria:** {perf['best_improvement']:.2f}%
- **Modelos Avaliados:** {perf['models_evaluated']}
"""
        
        report_text += "\n## RESULTADOS DETALHADOS\n"
        
        for pair, results in self.evaluation_results.items():
            report_text += f"\n### {pair}\n"
            
            for result in results:
                model_type = result['model_type']
                report_text += f"\n#### {model_type} ({result['test_samples']} amostras de teste)\n"
                
                if result['baseline'] and result['optimized']:
                    report_text += "| Métrica | Baseline | Otimizado | Melhoria |\n"
                    report_text += "|---------|----------|-----------|----------|\n"
                    
                    for metric in result['baseline'].keys():
                        baseline_val = result['baseline'][metric]
                        optimized_val = result['optimized'][metric]
                        improvement = result['improvement'].get(metric, 0)
                        
                        symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                        report_text += f"| {metric} | {baseline_val:.4f} | {optimized_val:.4f} | {improvement:.2f}% {symbol} |\n"
        
        report_text += "\n## CONCLUSÕES E RECOMENDAÇÕES\n"
        
        if summary['models_with_improvement'] > 0:
            report_text += f"""
### Principais Descobertas:
- **{improvement_rate:.1f}%** dos modelos apresentaram melhorias após otimização
- **{len(self.optimized_pairs)}** pares de moedas foram avaliados com sucesso
- **{len(self.optimized_model_types)}** tipos de modelos foram comparados
"""
            
            if summary['average_improvements']:
                best_metric = max(summary['average_improvements'].items(), key=lambda x: x[1]['mean'])
                report_text += f"- Maior melhoria média em **{best_metric[0]}**: {best_metric[1]['mean']:.2f}%\n"
            
            if summary['model_performance']:
                best_model = max(summary['model_performance'].items(), key=lambda x: x[1]['improvement_avg'])
                report_text += f"- Melhor modelo geral: **{best_model[0]}** (melhoria: {best_model[1]['improvement_avg']:.4f})\n"
        
        report_text += """

### Recomendações:
1. **Implementar modelos otimizados** em produção para os pares avaliados
2. **Monitorar performance** continuamente em ambiente de produção
3. **Expandir otimização** para outros tipos de modelos (Gradient Boosting)
4. **Considerar ensemble** dos melhores modelos por par
5. **Re-otimizar periodicamente** com novos dados
6. **Implementar A/B testing** para validação em produção

### Próximos Passos:
- Otimizar modelos Gradient Boosting que falharam
- Expandir para outros pares de moedas
- Implementar pipeline de produção
- Configurar monitoramento de drift
"""
        
        # Salvar relatório texto
        text_file = os.path.join(self.evaluation_dir, f'optimized_evaluation_summary_{timestamp}.txt')
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    def print_summary(self) -> None:
        """Imprime resumo da avaliação"""
        summary = self.generate_summary_statistics()
        
        print("\n" + "="*80)
        print("📊 RESUMO DA AVALIAÇÃO - MODELOS FX OTIMIZADOS")
        print("="*80)
        print(f"🎯 Pares Avaliados: {summary['total_pairs']} ({', '.join(self.optimized_pairs)})")
        print(f"🤖 Modelos Avaliados: {summary['total_models_evaluated']}")
        
        improvement_rate = 0
        if summary['total_models_evaluated'] > 0:
            improvement_rate = summary['models_with_improvement'] / summary['total_models_evaluated'] * 100
        
        print(f"✅ Modelos com Melhoria: {summary['models_with_improvement']} ({improvement_rate:.1f}%)")
        
        if summary['average_improvements']:
            print("\n📈 MELHORIAS MÉDIAS POR MÉTRICA:")
            for metric, stats in summary['average_improvements'].items():
                positive_pct = (stats['positive_count'] / summary['total_models_evaluated']) * 100
                print(f"  {metric}: {stats['mean']:.2f}% (±{stats['std']:.2f}%) - {stats['positive_count']} modelos melhoraram ({positive_pct:.1f}%)")
        
        if summary['model_performance']:
            print("\n🏆 PERFORMANCE DOS MODELOS:")
            sorted_models = sorted(summary['model_performance'].items(), 
                                 key=lambda x: x[1]['improvement_avg'], reverse=True)
            for model_type, perf in sorted_models:
                print(f"  {model_type}: {perf['baseline_avg']:.4f} → {perf['optimized_avg']:.4f} (Δ{perf['improvement_avg']:.4f})")
        
        if summary['pair_performance']:
            print("\n🌍 MELHORES PARES:")
            sorted_pairs = sorted(summary['pair_performance'].items(), 
                                key=lambda x: x[1]['avg_improvement'], reverse=True)
            for pair, perf in sorted_pairs[:3]:
                print(f"  {pair}: {perf['avg_improvement']:.2f}% (melhor: {perf['best_improvement']:.2f}%)")
        
        print("\n📁 Arquivos Gerados:")
        print(f"  - Relatórios: {self.evaluation_dir}/optimized_evaluation_*")
        print(f"  - Visualizações: {self.evaluation_dir}/*.png")
        print("="*80)
    
    def run_complete_evaluation(self) -> None:
        """Executa avaliação completa"""
        print("🚀 Iniciando Avaliação Completa dos Modelos FX Otimizados")
        print("="*80)
        
        # 1. Avaliar todos os modelos
        self.evaluate_all_models()
        
        # 2. Gerar visualizações
        self.create_visualizations()
        
        # 3. Gerar relatório detalhado
        self.generate_detailed_report()
        
        # 4. Imprimir resumo
        self.print_summary()
        
        print("\n✅ Avaliação completa finalizada com sucesso!")

def main():
    """Função principal"""
    evaluator = FXOptimizedModelEvaluator()
    evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main()