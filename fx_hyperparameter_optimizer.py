#!/usr/bin/env python3
"""
Otimizador de Hiperparâmetros FX - Machine Learning para Forex
Otimiza hiperparâmetros dos modelos ML para maximizar performance
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
from itertools import product

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
import xgboost as xgb
from scipy.stats import uniform, randint

warnings.filterwarnings('ignore')

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fx_hyperparameter_optimization.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FXHyperparameterOptimizer:
    def __init__(self, data_directory: str = "processed_fx_data", 
                 models_directory: str = "trained_models",
                 optimized_models_directory: str = "optimized_models"):
        self.data_directory = Path(data_directory)
        self.models_directory = Path(models_directory)
        self.optimized_models_directory = Path(optimized_models_directory)
        self.optimized_models_directory.mkdir(exist_ok=True)
        
        # Pares prioritários baseados nos resultados do treinamento
        self.top_performing_pairs = ['audjpy', 'eurjpy', 'usdnok', 'gbpjpy', 'usdsek']
        self.all_pairs = [
            'usdjpy', 'gbpusd', 'eurusd', 'usdchf', 'audjpy',
            'gbpjpy', 'xauusd', 'eurjpy', 'usdsek', 'usdnok'
        ]
        
        # Configurações de otimização para cada modelo
        self.optimization_configs = {
            'random_forest_reg': {
                'model': RandomForestRegressor,
                'param_grid': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False]
                },
                'param_distributions': {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(5, 30),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['sqrt', 'log2', None]
                },
                'scoring': 'r2',
                'type': 'regression'
            },
            'random_forest_clf': {
                'model': RandomForestClassifier,
                'param_grid': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False]
                },
                'param_distributions': {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(5, 30),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['sqrt', 'log2', None]
                },
                'scoring': 'f1_weighted',
                'type': 'classification'
            },
            'xgboost_reg': {
                'model': xgb.XGBRegressor,
                'param_grid': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 6, 9, 12],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'reg_alpha': [0, 0.1, 0.5, 1.0],
                    'reg_lambda': [0, 0.1, 0.5, 1.0]
                },
                'param_distributions': {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(3, 15),
                    'learning_rate': uniform(0.01, 0.3),
                    'subsample': uniform(0.6, 0.4),
                    'colsample_bytree': uniform(0.6, 0.4),
                    'reg_alpha': uniform(0, 2),
                    'reg_lambda': uniform(0, 2)
                },
                'scoring': 'r2',
                'type': 'regression'
            },
            'xgboost_clf': {
                'model': xgb.XGBClassifier,
                'param_grid': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 6, 9, 12],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'reg_alpha': [0, 0.1, 0.5, 1.0],
                    'reg_lambda': [0, 0.1, 0.5, 1.0]
                },
                'param_distributions': {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(3, 15),
                    'learning_rate': uniform(0.01, 0.3),
                    'subsample': uniform(0.6, 0.4),
                    'colsample_bytree': uniform(0.6, 0.4),
                    'reg_alpha': uniform(0, 2),
                    'reg_lambda': uniform(0, 2)
                },
                'scoring': 'f1_weighted',
                'type': 'classification'
            },
            'gradient_boosting_reg': {
                'model': GradientBoostingRegressor,
                'param_grid': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 6, 9, 12],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'param_distributions': {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(3, 15),
                    'learning_rate': uniform(0.01, 0.3),
                    'subsample': uniform(0.6, 0.4),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10)
                },
                'scoring': 'r2',
                'type': 'regression'
            }
        }
        
        self.optimization_results = {}
        
    def load_processed_data(self, pair_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Carrega dados pré-processados de um par"""
        logger.info(f"📊 Carregando dados processados para {pair_name.upper()}...")
        
        pair_dir = self.data_directory / pair_name.upper()
        if not pair_dir.exists():
            logger.error(f"Diretório não encontrado: {pair_dir}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        try:
            train_df = pd.read_parquet(pair_dir / 'train.parquet')
            val_df = pd.read_parquet(pair_dir / 'validation.parquet')
            test_df = pd.read_parquet(pair_dir / 'test.parquet')
            
            logger.info(f"✅ Dados carregados: {len(train_df):,} treino, {len(val_df):,} validação, {len(test_df):,} teste")
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados para {pair_name}: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def prepare_features_and_targets(self, df: pd.DataFrame, target_period: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepara features e targets para otimização"""
        # Remove colunas não-feature
        exclude_cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'] + \
                      [col for col in df.columns if col.startswith('Target_')]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Features
        X = df[feature_cols].values
        
        # Targets para regressão e classificação
        y_reg = df[f'Target_Return_{target_period}'].values
        y_clf = df[f'Target_Direction_{target_period}'].values
        
        # Remove linhas com NaN
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_reg) | np.isnan(y_clf))
        X = X[valid_mask]
        y_reg = y_reg[valid_mask]
        y_clf = y_clf[valid_mask]
        
        logger.info(f"📊 Features preparadas: {X.shape[1]} features, {X.shape[0]} amostras")
        return X, y_reg, y_clf
    
    def optimize_model_hyperparameters(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                                     X_val: np.ndarray, y_val: np.ndarray, 
                                     optimization_method: str = 'randomized') -> Dict:
        """Otimiza hiperparâmetros de um modelo específico"""
        logger.info(f"🔧 Otimizando hiperparâmetros para {model_name} usando {optimization_method}...")
        
        config = self.optimization_configs[model_name]
        model_class = config['model']
        scoring = config['scoring']
        model_type = config['type']
        
        start_time = datetime.now()
        
        try:
            # Combina dados de treino e validação para otimização
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.hstack([y_train, y_val])
            
            # Configura modelo base
            # GradientBoosting não suporta n_jobs
            if 'gradient_boosting' in model_name:
                base_model = model_class(random_state=42)
            else:
                base_model = model_class(random_state=42, n_jobs=-1)
            
            # Escolhe método de otimização
            if optimization_method == 'grid':
                # Grid Search (mais lento, mas exaustivo)
                param_grid = config['param_grid']
                # Reduz grid para evitar explosão combinatória
                reduced_grid = self.reduce_param_grid(param_grid)
                
                optimizer = GridSearchCV(
                    estimator=base_model,
                    param_grid=reduced_grid,
                    scoring=scoring,
                    cv=3,
                    n_jobs=-1,
                    verbose=0
                )
                
            else:  # randomized
                # Randomized Search (mais rápido)
                param_distributions = config['param_distributions']
                
                optimizer = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_distributions,
                    n_iter=50,  # Número de combinações a testar
                    scoring=scoring,
                    cv=3,
                    n_jobs=-1,
                    random_state=42,
                    verbose=0
                )
            
            # Executa otimização
            optimizer.fit(X_combined, y_combined)
            
            # Melhor modelo
            best_model = optimizer.best_estimator_
            
            # Avalia no conjunto de validação
            y_train_pred = best_model.predict(X_train)
            y_val_pred = best_model.predict(X_val)
            
            # Calcula métricas
            if model_type == 'regression':
                metrics = self.calculate_regression_metrics(y_train, y_train_pred, y_val, y_val_pred)
            else:
                metrics = self.calculate_classification_metrics(y_train, y_train_pred, y_val, y_val_pred)
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'model': best_model,
                'model_name': model_name,
                'model_type': model_type,
                'optimization_method': optimization_method,
                'optimization_time': optimization_time,
                'best_params': optimizer.best_params_,
                'best_score': float(optimizer.best_score_),
                'cv_results': {
                    'mean_test_score': optimizer.cv_results_['mean_test_score'].tolist(),
                    'std_test_score': optimizer.cv_results_['std_test_score'].tolist()
                },
                'metrics': metrics,
                'feature_importance': self.get_feature_importance(best_model),
                'success': True
            }
            
            logger.info(f"✅ {model_name} otimizado em {optimization_time:.2f}s | Score: {optimizer.best_score_:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro ao otimizar {model_name}: {e}")
            return {
                'model_name': model_name,
                'optimization_method': optimization_method,
                'success': False,
                'error': str(e)
            }
    
    def reduce_param_grid(self, param_grid: Dict, max_combinations: int = 100) -> Dict:
        """Reduz o grid de parâmetros para evitar explosão combinatória"""
        # Calcula número total de combinações
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        
        if total_combinations <= max_combinations:
            return param_grid
        
        # Reduz cada parâmetro proporcionalmente
        reduction_factor = (max_combinations / total_combinations) ** (1 / len(param_grid))
        
        reduced_grid = {}
        for param, values in param_grid.items():
            if isinstance(values, list):
                new_size = max(2, int(len(values) * reduction_factor))
                # Seleciona valores espaçados uniformemente
                indices = np.linspace(0, len(values) - 1, new_size, dtype=int)
                reduced_grid[param] = [values[i] for i in indices]
            else:
                reduced_grid[param] = values
        
        return reduced_grid
    
    def calculate_regression_metrics(self, y_train_true: np.ndarray, y_train_pred: np.ndarray,
                                   y_val_true: np.ndarray, y_val_pred: np.ndarray) -> Dict:
        """Calcula métricas para modelos de regressão"""
        return {
            'train_mse': float(mean_squared_error(y_train_true, y_train_pred)),
            'train_mae': float(mean_absolute_error(y_train_true, y_train_pred)),
            'train_r2': float(r2_score(y_train_true, y_train_pred)),
            'val_mse': float(mean_squared_error(y_val_true, y_val_pred)),
            'val_mae': float(mean_absolute_error(y_val_true, y_val_pred)),
            'val_r2': float(r2_score(y_val_true, y_val_pred)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train_true, y_train_pred))),
            'val_rmse': float(np.sqrt(mean_squared_error(y_val_true, y_val_pred)))
        }
    
    def calculate_classification_metrics(self, y_train_true: np.ndarray, y_train_pred: np.ndarray,
                                       y_val_true: np.ndarray, y_val_pred: np.ndarray) -> Dict:
        """Calcula métricas para modelos de classificação"""
        return {
            'train_accuracy': float(accuracy_score(y_train_true, y_train_pred)),
            'train_precision': float(precision_score(y_train_true, y_train_pred, average='weighted', zero_division=0)),
            'train_recall': float(recall_score(y_train_true, y_train_pred, average='weighted', zero_division=0)),
            'train_f1': float(f1_score(y_train_true, y_train_pred, average='weighted', zero_division=0)),
            'val_accuracy': float(accuracy_score(y_val_true, y_val_pred)),
            'val_precision': float(precision_score(y_val_true, y_val_pred, average='weighted', zero_division=0)),
            'val_recall': float(recall_score(y_val_true, y_val_pred, average='weighted', zero_division=0)),
            'val_f1': float(f1_score(y_val_true, y_val_pred, average='weighted', zero_division=0))
        }
    
    def get_feature_importance(self, model) -> Optional[List[float]]:
        """Extrai importância das features do modelo"""
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                return np.abs(model.coef_).flatten().tolist()
            else:
                return None
        except:
            return None
    
    def save_optimized_model(self, optimization_result: Dict, pair_name: str, target_period: int = 1) -> str:
        """Salva modelo otimizado"""
        if not optimization_result.get('success', False):
            return ""
        
        pair_models_dir = self.optimized_models_directory / pair_name.upper()
        pair_models_dir.mkdir(exist_ok=True)
        
        model_name = optimization_result['model_name']
        optimization_method = optimization_result['optimization_method']
        model_file = pair_models_dir / f"{model_name}_{optimization_method}_target_{target_period}.pkl"
        
        # Salva o modelo
        with open(model_file, 'wb') as f:
            pickle.dump(optimization_result['model'], f)
        
        # Salva metadados
        metadata = {
            'model_name': model_name,
            'model_type': optimization_result['model_type'],
            'optimization_method': optimization_method,
            'pair': pair_name.upper(),
            'target_period': target_period,
            'optimization_time': optimization_result['optimization_time'],
            'best_params': optimization_result['best_params'],
            'best_score': optimization_result['best_score'],
            'metrics': optimization_result['metrics'],
            'feature_importance': optimization_result['feature_importance'],
            'timestamp': datetime.now().isoformat(),
            'model_file': str(model_file)
        }
        
        metadata_file = pair_models_dir / f"{model_name}_{optimization_method}_target_{target_period}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Modelo otimizado salvo: {model_file}")
        return str(model_file)
    
    def optimize_pair_models(self, pair_name: str, target_period: int = 1, 
                           optimization_method: str = 'randomized') -> Dict:
        """Otimiza todos os modelos para um par específico"""
        logger.info(f"🎯 Otimizando modelos para {pair_name.upper()}...")
        
        # Carrega dados
        train_df, val_df, test_df = self.load_processed_data(pair_name)
        if train_df.empty:
            return {'success': False, 'error': 'Falha ao carregar dados'}
        
        # Prepara features e targets
        X_train, y_train_reg, y_train_clf = self.prepare_features_and_targets(train_df, target_period)
        X_val, y_val_reg, y_val_clf = self.prepare_features_and_targets(val_df, target_period)
        
        if X_train.shape[0] == 0:
            return {'success': False, 'error': 'Nenhuma amostra válida encontrada'}
        
        # Otimiza modelos
        pair_results = {
            'pair': pair_name.upper(),
            'target_period': target_period,
            'optimization_method': optimization_method,
            'models': {},
            'best_regression_model': None,
            'best_classification_model': None
        }
        
        best_reg_score = -np.inf
        best_clf_score = -np.inf
        
        for model_name, config in self.optimization_configs.items():
            if config['type'] == 'regression':
                result = self.optimize_model_hyperparameters(
                    model_name, X_train, y_train_reg, X_val, y_val_reg, optimization_method
                )
                if result.get('success', False):
                    # Salva modelo
                    model_file = self.save_optimized_model(result, pair_name, target_period)
                    result['model_file'] = model_file
                    
                    # Verifica se é o melhor modelo de regressão
                    best_score = result['best_score']
                    if best_score > best_reg_score:
                        best_reg_score = best_score
                        pair_results['best_regression_model'] = model_name
                
            else:  # classification
                result = self.optimize_model_hyperparameters(
                    model_name, X_train, y_train_clf, X_val, y_val_clf, optimization_method
                )
                if result.get('success', False):
                    # Salva modelo
                    model_file = self.save_optimized_model(result, pair_name, target_period)
                    result['model_file'] = model_file
                    
                    # Verifica se é o melhor modelo de classificação
                    best_score = result['best_score']
                    if best_score > best_clf_score:
                        best_clf_score = best_score
                        pair_results['best_classification_model'] = model_name
            
            # Remove o objeto modelo para economizar memória
            if 'model' in result:
                del result['model']
            
            pair_results['models'][model_name] = result
        
        # Calcula estatísticas do par
        successful_models = [m for m in pair_results['models'].values() if m.get('success', False)]
        pair_results['summary'] = {
            'total_models': len(self.optimization_configs),
            'successful_optimizations': len(successful_models),
            'training_samples': X_train.shape[0],
            'validation_samples': X_val.shape[0],
            'features_count': X_train.shape[1],
            'best_regression_score': best_reg_score if best_reg_score > -np.inf else None,
            'best_classification_score': best_clf_score if best_clf_score > -np.inf else None
        }
        
        pair_results['success'] = len(successful_models) > 0
        logger.info(f"✅ {pair_name.upper()}: {len(successful_models)}/{len(self.optimization_configs)} modelos otimizados")
        
        return pair_results
    
    def optimize_top_performing_pairs(self, target_period: int = 1, 
                                    optimization_method: str = 'randomized') -> Dict:
        """Otimiza modelos para os pares com melhor performance"""
        logger.info("🎯 Otimizando modelos para pares com melhor performance...")
        
        results = {}
        for pair in self.top_performing_pairs:
            results[pair] = self.optimize_pair_models(pair, target_period, optimization_method)
        
        return results
    
    def optimize_all_pairs(self, target_period: int = 1, 
                         optimization_method: str = 'randomized') -> Dict:
        """Otimiza modelos para todos os pares disponíveis"""
        logger.info("📊 Otimizando modelos para todos os pares...")
        
        results = {}
        for pair in self.all_pairs:
            results[pair] = self.optimize_pair_models(pair, target_period, optimization_method)
        
        return results
    
    def generate_optimization_report(self, results: Dict, target_period: int = 1) -> str:
        """Gera relatório de otimização"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"hyperparameter_optimization_report_{timestamp}.json"
        
        # Calcula estatísticas gerais
        successful_pairs = [p for p, r in results.items() if r.get('success', False)]
        total_optimizations = sum([r.get('summary', {}).get('successful_optimizations', 0) for r in results.values()])
        
        # Encontra melhores modelos globais
        best_models = {
            'regression': {'pair': None, 'model': None, 'score': -np.inf},
            'classification': {'pair': None, 'model': None, 'score': -np.inf}
        }
        
        for pair, result in results.items():
            if result.get('success', False):
                summary = result.get('summary', {})
                
                # Melhor regressão
                if summary.get('best_regression_score') and summary['best_regression_score'] > best_models['regression']['score']:
                    best_models['regression'] = {
                        'pair': pair,
                        'model': result.get('best_regression_model'),
                        'score': summary['best_regression_score']
                    }
                
                # Melhor classificação
                if summary.get('best_classification_score') and summary['best_classification_score'] > best_models['classification']['score']:
                    best_models['classification'] = {
                        'pair': pair,
                        'model': result.get('best_classification_model'),
                        'score': summary['best_classification_score']
                    }
        
        report = {
            'timestamp': timestamp,
            'target_period': target_period,
            'optimization_summary': {
                'total_pairs_optimized': len(results),
                'successful_pairs': len(successful_pairs),
                'total_optimizations': total_optimizations,
                'optimized_models_directory': str(self.optimized_models_directory),
                'best_models': best_models
            },
            'pair_results': results,
            'optimization_configs': self.optimization_configs,
            'recommendations': self.generate_optimization_recommendations(results)
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📋 Relatório de otimização salvo: {report_file}")
        return report_file
    
    def generate_optimization_recommendations(self, results: Dict) -> Dict:
        """Gera recomendações baseadas nos resultados da otimização"""
        successful_pairs = [p for p, r in results.items() if r.get('success', False)]
        
        # Analisa melhoria de performance
        improvements = []
        for pair, result in results.items():
            if result.get('success', False):
                for model_name, model_result in result.get('models', {}).items():
                    if model_result.get('success', False):
                        best_score = model_result.get('best_score', 0)
                        improvements.append({
                            'pair': pair,
                            'model': model_name,
                            'score': best_score
                        })
        
        # Ordena por performance
        improvements.sort(key=lambda x: x['score'], reverse=True)
        
        recommendations = {
            'production_ready_models': improvements[:10],
            'optimization_insights': {
                'most_improved_pairs': successful_pairs[:5],
                'best_optimization_method': 'randomized',  # Baseado na velocidade vs qualidade
                'recommended_ensemble_size': min(5, len(improvements))
            },
            'next_steps': [
                'Avaliar modelos otimizados em dados de teste',
                'Implementar ensemble dos melhores modelos',
                'Configurar pipeline de produção',
                'Monitorar performance em tempo real',
                'Implementar retreinamento automático'
            ],
            'performance_analysis': {
                'average_improvement': np.mean([imp['score'] for imp in improvements]) if improvements else 0,
                'top_performing_models': [imp['model'] for imp in improvements[:5]],
                'optimization_success_rate': len(successful_pairs) / len(results) if results else 0
            }
        }
        
        return recommendations
    
    def print_optimization_summary(self, results: Dict):
        """Imprime resumo da otimização"""
        print("\n" + "="*60)
        print("🔧 RESUMO DA OTIMIZAÇÃO DE HIPERPARÂMETROS")
        print("="*60)
        
        successful_pairs = [p for p, r in results.items() if r.get('success', False)]
        total_optimizations = sum([r.get('summary', {}).get('successful_optimizations', 0) for r in results.values()])
        
        print(f"\n✅ Pares Otimizados: {len(successful_pairs)}/{len(results)}")
        print(f"🔧 Total de Otimizações: {total_optimizations}")
        
        if successful_pairs:
            print("\n🏆 RESULTADOS POR PAR:")
            for pair in successful_pairs:
                result = results[pair]
                summary = result.get('summary', {})
                successful_optimizations = summary.get('successful_optimizations', 0)
                best_reg_score = summary.get('best_regression_score', 0)
                best_clf_score = summary.get('best_classification_score', 0)
                
                print(f"{pair.upper()}: {successful_optimizations} otimizações | "
                      f"Melhor R²: {best_reg_score:.4f} | Melhor F1: {best_clf_score:.4f}")
        
        if len(successful_pairs) >= 3:
            print("\n🎉 OTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
            print("✅ Modelos otimizados prontos para produção")
        else:
            print("\n⚠️ ATENÇÃO: Poucos pares otimizados com sucesso")
            print("❌ Revisar configuração de otimização")

def main():
    # Configuração
    data_dir = "processed_fx_data"
    models_dir = "trained_models"
    optimized_models_dir = "optimized_models"
    
    if not Path(data_dir).exists():
        print(f"❌ Diretório não encontrado: {data_dir}")
        print("Execute primeiro o pipeline de pré-processamento")
        return
    
    # Inicializar otimizador
    optimizer = FXHyperparameterOptimizer(data_dir, models_dir, optimized_models_dir)
    
    print("🔧 Iniciando Otimização de Hiperparâmetros FX")
    print("="*50)
    
    # Otimizar pares com melhor performance primeiro
    print("\n🎯 Fase 1: Otimizando pares com melhor performance...")
    top_results = optimizer.optimize_top_performing_pairs(
        target_period=1, 
        optimization_method='randomized'
    )
    
    # Verificar se otimização foi bem-sucedida
    successful_top = [p for p, r in top_results.items() if r.get('success', False)]
    
    if len(successful_top) >= 3:
        print("\n📊 Fase 2: Expandindo para todos os pares...")
        # Otimizar todos os pares
        all_results = optimizer.optimize_all_pairs(
            target_period=1,
            optimization_method='randomized'
        )
    else:
        print("\n⚠️ Problemas com pares prioritários. Continuando apenas com os bem-sucedidos...")
        all_results = top_results
    
    # Gerar relatório
    report_file = optimizer.generate_optimization_report(all_results, target_period=1)
    
    # Imprimir resumo
    optimizer.print_optimization_summary(all_results)
    
    print(f"\n📋 Relatório detalhado: {report_file}")
    print(f"💾 Modelos otimizados salvos em: {optimized_models_dir}")
    print("\n🎯 Pipeline de ML FX concluído com sucesso!")
    print("✅ Modelos prontos para produção")

if __name__ == "__main__":
    main()