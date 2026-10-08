#!/usr/bin/env python3
"""
Otimizador específico para modelos Gradient Boosting
Corrige problemas identificados na otimização anterior
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import randint, uniform

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fx_gradient_boosting_optimization.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GradientBoostingOptimizer:
    """Otimizador específico para modelos Gradient Boosting"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.processed_data_dir = self.base_dir / "processed_fx_data"
        self.optimized_models_dir = self.base_dir / "optimized_models"
        self.optimized_models_dir.mkdir(exist_ok=True)
        
        # Configurações específicas para Gradient Boosting
        self.gb_configs = {
            'gradient_boosting_reg': {
                'model': GradientBoostingRegressor,
                'param_distributions': {
                    'n_estimators': randint(100, 500),
                    'max_depth': randint(3, 12),
                    'learning_rate': uniform(0.01, 0.19),  # 0.01 to 0.2
                    'subsample': uniform(0.7, 0.3),        # 0.7 to 1.0
                    'min_samples_split': randint(2, 15),
                    'min_samples_leaf': randint(1, 8),
                    'max_features': ['sqrt', 'log2', None]
                },
                'scoring': 'r2',
                'type': 'regression'
            },
            'gradient_boosting_clf': {
                'model': GradientBoostingClassifier,
                'param_distributions': {
                    'n_estimators': randint(100, 500),
                    'max_depth': randint(3, 12),
                    'learning_rate': uniform(0.01, 0.19),
                    'subsample': uniform(0.7, 0.3),
                    'min_samples_split': randint(2, 15),
                    'min_samples_leaf': randint(1, 8),
                    'max_features': ['sqrt', 'log2', None]
                },
                'scoring': 'f1_weighted',
                'type': 'classification'
            }
        }
        
        # Pares disponíveis para otimização (com dados processados)
        self.currency_pairs = [
            'AUDJPY', 'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 
            'USDCHF', 'USDJPY', 'USDNOK', 'USDSEK', 'XAUUSD'
        ]
    
    def load_processed_data(self, pair_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Carrega dados processados para um par"""
        try:
            pair_dir = self.processed_data_dir / pair_name.upper()
            
            train_file = pair_dir / "train.parquet"
            val_file = pair_dir / "validation.parquet"
            test_file = pair_dir / "test.parquet"
            
            if not all([train_file.exists(), val_file.exists(), test_file.exists()]):
                logger.error(f"❌ Arquivos de dados não encontrados para {pair_name}")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            train_df = pd.read_parquet(train_file)
            val_df = pd.read_parquet(val_file)
            test_df = pd.read_parquet(test_file)
            
            logger.info(f"📊 Dados carregados para {pair_name}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados para {pair_name}: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def prepare_features_and_targets(self, df: pd.DataFrame, target_period: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepara features e targets (mesma lógica do treinamento original)"""
        # Remove colunas não-feature (mesma lógica do fx_model_trainer.py)
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
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calcula métricas de regressão"""
        return {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred)))
        }
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calcula métricas de classificação"""
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
    
    def optimize_gradient_boosting_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                                       X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Otimiza um modelo Gradient Boosting específico"""
        logger.info(f"🔧 Otimizando {model_name}...")
        
        config = self.gb_configs[model_name]
        model_class = config['model']
        param_distributions = config['param_distributions']
        scoring = config['scoring']
        model_type = config['type']
        
        start_time = datetime.now()
        
        try:
            # Combina dados de treino e validação
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.hstack([y_train, y_val])
            
            # Modelo base (SEM n_jobs para Gradient Boosting)
            base_model = model_class(random_state=42)
            
            # Randomized Search otimizado
            optimizer = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=20,   # Reduzido para acelerar otimização
                scoring=scoring,
                cv=3,        # Reduzido para acelerar validação
                random_state=42,
                verbose=1,
                n_jobs=-1    # n_jobs no RandomizedSearchCV, não no modelo
            )
            
            # Executa otimização
            logger.info(f"🚀 Iniciando otimização com {optimizer.n_iter} iterações...")
            optimizer.fit(X_combined, y_combined)
            
            # Melhor modelo
            best_model = optimizer.best_estimator_
            
            # Avalia no conjunto de validação
            y_val_pred = best_model.predict(X_val)
            
            # Calcula métricas
            if model_type == 'regression':
                metrics = self.calculate_regression_metrics(y_val, y_val_pred)
            else:
                metrics = self.calculate_classification_metrics(y_val, y_val_pred)
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            # Feature importance
            feature_importance = None
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = best_model.feature_importances_.tolist()
            
            result = {
                'model': best_model,
                'model_name': model_name,
                'model_type': model_type,
                'optimization_time': optimization_time,
                'best_params': optimizer.best_params_,
                'best_score': float(optimizer.best_score_),
                'metrics': metrics,
                'feature_importance': feature_importance,
                'cv_scores': {
                    'mean': float(np.mean(optimizer.cv_results_['mean_test_score'])),
                    'std': float(np.std(optimizer.cv_results_['mean_test_score']))
                }
            }
            
            logger.info(f"✅ {model_name} otimizado em {optimization_time:.2f}s")
            logger.info(f"📊 Melhor score: {optimizer.best_score_:.4f}")
            logger.info(f"🎯 Melhores parâmetros: {optimizer.best_params_}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro ao otimizar {model_name}: {e}")
            return {'error': str(e), 'model_name': model_name}
    
    def save_optimized_model(self, optimization_result: Dict, pair_name: str, target_period: int = 1) -> str:
        """Salva modelo otimizado"""
        if 'error' in optimization_result:
            return None
        
        # Diretório do par
        pair_models_dir = self.optimized_models_dir / pair_name.upper()
        pair_models_dir.mkdir(exist_ok=True)
        
        model_name = optimization_result['model_name']
        model = optimization_result['model']
        
        # Salva modelo
        model_file = pair_models_dir / f"{model_name}_target_{target_period}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Salva metadata
        metadata = {
            'model_name': model_name,
            'model_type': optimization_result['model_type'],
            'pair': pair_name.upper(),
            'target_period': target_period,
            'optimization_time': optimization_result['optimization_time'],
            'best_params': optimization_result['best_params'],
            'best_score': optimization_result['best_score'],
            'metrics': optimization_result['metrics'],
            'feature_importance': optimization_result['feature_importance'],
            'cv_scores': optimization_result['cv_scores'],
            'timestamp': datetime.now().isoformat(),
            'model_file': str(model_file)
        }
        
        metadata_file = pair_models_dir / f"{model_name}_target_{target_period}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Modelo otimizado salvo: {model_file}")
        return str(model_file)
    
    def optimize_pair_gradient_boosting(self, pair_name: str, target_period: int = 1) -> Dict:
        """Otimiza modelos Gradient Boosting para um par específico"""
        logger.info(f"🎯 Otimizando Gradient Boosting para {pair_name.upper()}...")
        
        # Carrega dados
        train_df, val_df, test_df = self.load_processed_data(pair_name)
        if train_df.empty:
            return {'success': False, 'error': 'Falha ao carregar dados'}
        
        # Prepara features e targets
        X_train, y_train_reg, y_train_clf = self.prepare_features_and_targets(train_df, target_period)
        X_val, y_val_reg, y_val_clf = self.prepare_features_and_targets(val_df, target_period)
        
        if X_train.shape[0] == 0:
            return {'success': False, 'error': 'Nenhuma amostra válida encontrada'}
        
        # Resultados do par
        pair_results = {
            'pair': pair_name.upper(),
            'target_period': target_period,
            'models': {},
            'success': True
        }
        
        # Otimiza modelos Gradient Boosting
        for model_name in ['gradient_boosting_reg', 'gradient_boosting_clf']:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔧 Otimizando {model_name} para {pair_name.upper()}")
            logger.info(f"{'='*60}")
            
            # Seleciona target apropriado
            if 'reg' in model_name:
                y_train, y_val = y_train_reg, y_val_reg
            else:
                y_train, y_val = y_train_clf, y_val_clf
            
            # Otimiza modelo
            result = self.optimize_gradient_boosting_model(
                model_name, X_train, y_train, X_val, y_val
            )
            
            if 'error' not in result:
                # Salva modelo otimizado
                model_file = self.save_optimized_model(result, pair_name, target_period)
                result['model_file'] = model_file
                
                # Remove modelo do resultado para economizar memória
                result_copy = result.copy()
                del result_copy['model']
                pair_results['models'][model_name] = result_copy
                
                logger.info(f"✅ {model_name} otimizado com sucesso!")
            else:
                pair_results['models'][model_name] = result
                logger.error(f"❌ Falha na otimização de {model_name}: {result['error']}")
        
        return pair_results
    
    def optimize_all_pairs(self, target_period: int = 1) -> Dict:
        """Otimiza modelos Gradient Boosting para todos os pares"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 INICIANDO OTIMIZAÇÃO GRADIENT BOOSTING - {len(self.currency_pairs)} PARES")
        logger.info(f"{'='*80}")
        
        start_time = datetime.now()
        all_results = {
            'optimization_start': start_time.isoformat(),
            'target_period': target_period,
            'pairs': {},
            'summary': {
                'total_pairs': len(self.currency_pairs),
                'successful_pairs': 0,
                'failed_pairs': 0,
                'total_models_optimized': 0,
                'failed_models': 0
            }
        }
        
        for i, pair in enumerate(self.currency_pairs, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"📈 PROCESSANDO PAR {i}/{len(self.currency_pairs)}: {pair}")
            logger.info(f"{'='*60}")
            
            pair_result = self.optimize_pair_gradient_boosting(pair, target_period)
            all_results['pairs'][pair] = pair_result
            
            if pair_result['success']:
                all_results['summary']['successful_pairs'] += 1
                # Conta modelos otimizados com sucesso
                for model_name, model_result in pair_result['models'].items():
                    if 'error' not in model_result:
                        all_results['summary']['total_models_optimized'] += 1
                    else:
                        all_results['summary']['failed_models'] += 1
            else:
                all_results['summary']['failed_pairs'] += 1
        
        # Finaliza
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        all_results['optimization_end'] = end_time.isoformat()
        all_results['total_optimization_time'] = total_time
        
        # Salva relatório
        report_file = self.base_dir / f"gradient_boosting_optimization_report_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Log final
        logger.info(f"\n{'='*80}")
        logger.info(f"🏁 OTIMIZAÇÃO GRADIENT BOOSTING CONCLUÍDA")
        logger.info(f"{'='*80}")
        logger.info(f"⏱️  Tempo total: {total_time:.2f}s ({total_time/60:.1f} min)")
        logger.info(f"📊 Pares processados: {all_results['summary']['successful_pairs']}/{all_results['summary']['total_pairs']}")
        logger.info(f"🤖 Modelos otimizados: {all_results['summary']['total_models_optimized']}")
        logger.info(f"❌ Modelos falharam: {all_results['summary']['failed_models']}")
        logger.info(f"📁 Relatório salvo: {report_file}")
        
        return all_results

def main():
    """Função principal"""
    logger.info("🚀 Iniciando otimização específica para modelos Gradient Boosting...")
    
    optimizer = GradientBoostingOptimizer()
    results = optimizer.optimize_all_pairs(target_period=1)
    
    # Resumo final
    summary = results['summary']
    logger.info(f"\n📊 RESUMO FINAL:")
    logger.info(f"✅ Modelos otimizados com sucesso: {summary['total_models_optimized']}")
    logger.info(f"❌ Modelos que falharam: {summary['failed_models']}")
    logger.info(f"⏱️  Tempo total: {results['total_optimization_time']:.2f}s")
    
    if summary['total_models_optimized'] > 0:
        logger.info(f"\n🎉 Otimização Gradient Boosting concluída com sucesso!")
        return True
    else:
        logger.error(f"\n❌ Nenhum modelo foi otimizado com sucesso.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)