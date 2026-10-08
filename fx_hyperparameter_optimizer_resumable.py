#!/usr/bin/env python3
"""
Otimizador de Hiperparâmetros FX com Suporte a Checkpoints
Versão modificada que permite pausar e continuar de onde parou
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
import pickle
import argparse
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
        logging.FileHandler('fx_hyperparameter_optimizer.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FXHyperparameterOptimizerResumable:
    def __init__(self, data_directory: str = "processed_fx_data", 
                 models_directory: str = "trained_models",
                 optimized_models_directory: str = "optimized_models",
                 timeframe: str = "M1",
                 optimization_method: str = "randomized"):
        
        # Configuração de diretórios (assumindo que já vêm com o timeframe correto)
        self.data_directory = Path(data_directory)
        self.models_directory = Path(models_directory)
        self.optimized_models_directory = Path(optimized_models_directory)
        self.optimized_models_directory.mkdir(exist_ok=True)
        self.timeframe = timeframe
        self.optimization_method = optimization_method  # novo atributo
        
        # Arquivo de checkpoint
        self.checkpoint_file = Path(f"optimization_checkpoint_{timeframe}.json")
        
        # Pares prioritários baseados nos resultados do treinamento
        self.top_performing_pairs = ['audjpy', 'eurjpy', 'usdnok', 'gbpjpy', 'usdsek']
        self.all_pairs = [
            'usdjpy', 'gbpusd', 'eurusd', 'usdchf', 'audjpy',
            'gbpjpy', 'xauusd', 'eurjpy', 'usdsek', 'usdnok'
        ]
        
        # Modelos a otimizar
        self.models_to_optimize = [
            'random_forest_reg', 'random_forest_clf', 
            'xgboost_reg', 'xgboost_clf', 
            'gradient_boosting_reg', 'gradient_boosting_clf'
        ]
        
        # Configurações de otimização para cada modelo
        self.optimization_configs = {
            'random_forest_reg': {
                'model': RandomForestRegressor,
                'param_distributions': {
                    'n_estimators': randint(50, 200),  # Reduzido para evitar travamentos
                    'max_depth': randint(5, 15),       # Reduzido para ser mais rápido
                    'min_samples_split': randint(2, 10),
                    'min_samples_leaf': randint(1, 5),
                    'max_features': ['sqrt', 'log2']   # Removido None para ser mais rápido
                },
                'scoring': 'r2',
                'type': 'regression'
            },
            'random_forest_clf': {
                'model': RandomForestClassifier,
                'param_distributions': {
                    'n_estimators': randint(50, 200),  # Reduzido para evitar travamentos
                    'max_depth': randint(5, 15),       # Reduzido para ser mais rápido
                    'min_samples_split': randint(2, 10),
                    'min_samples_leaf': randint(1, 5),
                    'max_features': ['sqrt', 'log2']   # Removido None para ser mais rápido
                },
                'scoring': 'f1_weighted',
                'type': 'classification'
            },
            'xgboost_reg': {
                'model': xgb.XGBRegressor,
                'param_distributions': {
                    'n_estimators': randint(50, 300),
                    'max_depth': randint(3, 10),
                    'learning_rate': uniform(0.01, 0.3),
                    'subsample': uniform(0.6, 0.4),
                    'colsample_bytree': uniform(0.6, 0.4)
                },
                'scoring': 'r2',
                'type': 'regression'
            },
            'xgboost_clf': {
                'model': xgb.XGBClassifier,
                'param_distributions': {
                    'n_estimators': randint(50, 300),
                    'max_depth': randint(3, 10),
                    'learning_rate': uniform(0.01, 0.3),
                    'subsample': uniform(0.6, 0.4),
                    'colsample_bytree': uniform(0.6, 0.4)
                },
                'scoring': 'f1_weighted',
                'type': 'classification'
            },
            'gradient_boosting_reg': {
                'model': GradientBoostingRegressor,
                'param_distributions': {
                    'n_estimators': randint(50, 300),
                    'max_depth': randint(3, 10),
                    'learning_rate': uniform(0.01, 0.3),
                    'subsample': uniform(0.6, 0.4)
                },
                'scoring': 'r2',
                'type': 'regression'
            },
            'gradient_boosting_clf': {
                'model': GradientBoostingClassifier,
                'param_distributions': {
                    'n_estimators': randint(50, 300),
                    'max_depth': randint(3, 10),
                    'learning_rate': uniform(0.01, 0.3),
                    'subsample': uniform(0.6, 0.4)
                },
                'scoring': 'f1_weighted',
                'type': 'classification'
            }
        }
    
    def save_checkpoint(self, checkpoint_data: Dict):
        """Salva o estado atual da otimização"""
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Checkpoint salvo: {self.checkpoint_file}")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar checkpoint: {e}")
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Carrega o estado salvo da otimização"""
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            logger.info(f"📂 Checkpoint carregado: {self.checkpoint_file}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"❌ Erro ao carregar checkpoint: {e}")
            return None
    
    def is_model_already_optimized(self, pair_name: str, model_name: str) -> bool:
        """Verifica se um modelo já foi otimizado"""
        pair_dir = self.optimized_models_directory / pair_name.upper()
        model_file = pair_dir / f"{model_name}_{self.optimization_method}_target_1.pkl"
        metadata_file = pair_dir / f"{model_name}_{self.optimization_method}_target_1_metadata.json"
        
        return model_file.exists() and metadata_file.exists()
    
    def get_optimization_progress(self) -> Dict:
        """Retorna o progresso atual da otimização"""
        progress = {
            'completed_pairs': [],
            'partial_pairs': {},
            'remaining_pairs': []
        }
        
        for pair in self.all_pairs:
            completed_models = []
            for model in self.models_to_optimize:
                if self.is_model_already_optimized(pair, model):
                    completed_models.append(model)
            
            if len(completed_models) == len(self.models_to_optimize):
                progress['completed_pairs'].append(pair)
            elif len(completed_models) > 0:
                progress['partial_pairs'][pair] = {
                    'completed': completed_models,
                    'remaining': [m for m in self.models_to_optimize if m not in completed_models]
                }
            else:
                progress['remaining_pairs'].append(pair)
        
        return progress
    
    def print_progress_summary(self):
        """Imprime um resumo do progresso atual"""
        progress = self.get_optimization_progress()
        
        print("\n📊 RESUMO DO PROGRESSO DA OTIMIZAÇÃO")
        print("=" * 50)
        
        print(f"✅ Pares completamente otimizados: {len(progress['completed_pairs'])}")
        for pair in progress['completed_pairs']:
            print(f"   - {pair.upper()}")
        
        print(f"\n🔄 Pares parcialmente otimizados: {len(progress['partial_pairs'])}")
        for pair, info in progress['partial_pairs'].items():
            print(f"   - {pair.upper()}: {len(info['completed'])}/{len(self.models_to_optimize)} modelos")
            print(f"     Restantes: {', '.join(info['remaining'])}")
        
        print(f"\n⏳ Pares não iniciados: {len(progress['remaining_pairs'])}")
        for pair in progress['remaining_pairs']:
            print(f"   - {pair.upper()}")
        
        total_models = len(self.all_pairs) * len(self.models_to_optimize)
        completed_models = len(progress['completed_pairs']) * len(self.models_to_optimize)
        for info in progress['partial_pairs'].values():
            completed_models += len(info['completed'])
        
        completion_rate = (completed_models / total_models) * 100
        print(f"\n📈 Progresso geral: {completed_models}/{total_models} modelos ({completion_rate:.1f}%)")
        print("=" * 50)
    
    def continue_optimization_from_checkpoint(self, start_pair: str = None, start_model: str = None):
        """Continua a otimização de onde parou"""
        logger.info("🔄 Continuando otimização de onde parou...")
        
        # Mostrar progresso atual
        self.print_progress_summary()
        
        progress = self.get_optimization_progress()
        
        # Determinar de onde continuar
        pairs_to_process = []
        
        # Adicionar pares parcialmente completos
        for pair, info in progress['partial_pairs'].items():
            pairs_to_process.append((pair, info['remaining']))
        
        # Adicionar pares não iniciados
        for pair in progress['remaining_pairs']:
            pairs_to_process.append((pair, self.models_to_optimize.copy()))
        
        if not pairs_to_process:
            logger.info("✅ Todos os modelos já foram otimizados!")
            return
        
        logger.info(f"🎯 Continuando otimização para {len(pairs_to_process)} pares...")
        
        # Processar pares restantes
        for pair, models_to_do in pairs_to_process:
            logger.info(f"\n🎯 Processando par: {pair.upper()}")
            
            try:
                # Carregar dados
                train_df, val_df, test_df = self.load_processed_data(pair)
                X_train, y_train_reg, y_train_clf = self.prepare_features_and_targets(train_df)
                X_val, y_val_reg, y_val_clf = self.prepare_features_and_targets(val_df)
                
                # Otimizar modelos restantes
                for model_name in models_to_do:
                    if self.is_model_already_optimized(pair, model_name):
                        logger.info(f"⏭️ {model_name} já otimizado para {pair.upper()}, pulando...")
                        continue
                    
                    logger.info(f"🔧 Otimizando {model_name} para {pair.upper()}...")
                    
                    # Selecionar targets corretos
                    if 'reg' in model_name:
                        y_train, y_val = y_train_reg, y_val_reg
                    else:
                        y_train, y_val = y_train_clf, y_val_clf
                    
                    # Otimizar modelo
                    result = self.optimize_model_hyperparameters(
                        model_name, X_train, y_train, X_val, y_val,
                        optimization_method=self.optimization_method
                    )
                    
                    if result.get('success', False):
                        # Salvar modelo otimizado
                        model_path = self.save_optimized_model(result, pair)
                        logger.info(f"✅ {model_name} otimizado e salvo: {model_path}")
                    else:
                        logger.error(f"❌ Falha na otimização de {model_name} para {pair.upper()}")
                    
                    # Salvar checkpoint após cada modelo
                    checkpoint_data = {
                        'last_pair': pair,
                        'last_model': model_name,
                        'timestamp': datetime.now().isoformat(),
                        'timeframe': self.timeframe
                    }
                    self.save_checkpoint(checkpoint_data)
                
                logger.info(f"✅ Par {pair.upper()} concluído!")
                
            except Exception as e:
                logger.error(f"❌ Erro ao processar par {pair.upper()}: {e}")
                continue
        
        logger.info("🎉 Otimização completa!")
        self.print_progress_summary()
    
    # Métodos auxiliares (copiados do original)
    def load_processed_data(self, pair_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Carrega dados processados para um par específico"""
        logger.info(f"📊 Carregando dados processados para {pair_name.upper()}...")
        
        pair_dir = self.data_directory / pair_name.upper()
        logger.debug(f"Diretório do par: {pair_dir}")
        logger.debug(f"Diretório existe: {pair_dir.exists()}")
        
        # Tentar primeiro arquivos .parquet (formato mais recente)
        train_file = pair_dir / "train.parquet"
        val_file = pair_dir / "validation.parquet"
        test_file = pair_dir / "test.parquet"
        
        logger.debug(f"Arquivos parquet - Train: {train_file.exists()}, Val: {val_file.exists()}, Test: {test_file.exists()}")
        
        # Se não encontrar .parquet, tentar .csv (formato antigo)
        if not all([train_file.exists(), val_file.exists(), test_file.exists()]):
            logger.debug("Arquivos .parquet não encontrados, tentando .csv")
            train_file = pair_dir / "train_data.csv"
            val_file = pair_dir / "validation_data.csv"
            test_file = pair_dir / "test_data.csv"
            logger.debug(f"Arquivos csv - Train: {train_file.exists()}, Val: {val_file.exists()}, Test: {test_file.exists()}")
        
        if not all([train_file.exists(), val_file.exists(), test_file.exists()]):
            logger.error(f"Arquivos não encontrados:")
            logger.error(f"  Train: {train_file} (existe: {train_file.exists()})")
            logger.error(f"  Val: {val_file} (existe: {val_file.exists()})")
            logger.error(f"  Test: {test_file} (existe: {test_file.exists()})")
            raise FileNotFoundError(f"Dados processados não encontrados para {pair_name.upper()}")
        
        # Carregar dados baseado na extensão
        if train_file.suffix == '.parquet':
            train_df = pd.read_parquet(train_file)
            val_df = pd.read_parquet(val_file)
            test_df = pd.read_parquet(test_file)
        else:
            train_df = pd.read_csv(train_file)
            val_df = pd.read_csv(val_file)
            test_df = pd.read_csv(test_file)
        
        # Para evitar travamentos com datasets grandes, usar amostra menor para treino
        if len(train_df) > 5000:
            logger.info(f"Dataset grande detectado ({len(train_df)} amostras). Usando amostra de 5000 para otimização.")
            train_df = train_df.sample(n=5000, random_state=42)
        
        logger.info(f"✅ Dados carregados: {len(train_df):,} treino, {len(val_df):,} validação, {len(test_df):,} teste")
        
        return train_df, val_df, test_df
    
    def prepare_features_and_targets(self, df: pd.DataFrame, target_period: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepara features e targets para treinamento"""
        # Targets para regressão e classificação (com capitalização correta)
        target_reg_col = f'Target_Return_{target_period}'
        target_clf_col = f'Target_Direction_{target_period}'
        
        if target_reg_col not in df.columns or target_clf_col not in df.columns:
            raise ValueError(f"Colunas de target não encontradas: {target_reg_col}, {target_clf_col}")
        
        # Separar features e targets (excluir colunas não-numéricas)
        exclude_cols = ['DateTime', 'Date', 'Time', 'Timestamp', 'timestamp', 'datetime'] + [col for col in df.columns if col.startswith(('target_', 'Target_'))]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y_reg = df[target_reg_col].values
        y_clf = df[target_clf_col].values
        
        logger.info(f"📊 Features preparadas: {X.shape[1]} features, {X.shape[0]} amostras")
        
        return X, y_reg, y_clf
    
    def optimize_model_hyperparameters(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                                     X_val: np.ndarray, y_val: np.ndarray, 
                                     optimization_method: str = 'randomized') -> Dict:
        """Otimiza hiperparâmetros de um modelo específico"""
        
        if model_name not in self.optimization_configs:
            return {'success': False, 'error': f'Modelo {model_name} não configurado'}
        
        config = self.optimization_configs[model_name]
        
        try:
            start_time = datetime.now()
            
            # Configurar otimização
            if optimization_method == 'randomized':
                search = RandomizedSearchCV(
                    estimator=config['model'](random_state=42),
                    param_distributions=config['param_distributions'],
                    n_iter=50,  # Reduzido para ser mais rápido
                    cv=3,
                    scoring=config['scoring'],
                    n_jobs=-1,
                    random_state=42,
                    verbose=0
                )
            elif optimization_method == 'hybrid':
                # Método híbrido: Usando RandomizedSearchCV otimizado (mais estável que BayesSearchCV)
                logger.info(f"🔧 Configurando otimização híbrida para {model_name}...")
                search = RandomizedSearchCV(
                    estimator=config['model'](random_state=42),
                    param_distributions=config['param_distributions'],
                    n_iter=10,  # Reduzido para evitar travamentos com datasets grandes
                    cv=3,
                    scoring=config['scoring'],
                    n_jobs=1,  # Sem paralelização para estabilidade
                    random_state=42,
                    verbose=1
                )
                logger.info(f"✅ Otimização híbrida configurada com sucesso")
            else:
                raise ValueError(f"Método de otimização não suportado: {optimization_method}")
            
            # Executar otimização
            search.fit(X_train, y_train)
            
            # Calcular métricas
            best_model = search.best_estimator_
            train_pred = best_model.predict(X_train)
            val_pred = best_model.predict(X_val)
            
            if config['type'] == 'regression':
                metrics = self.calculate_regression_metrics(y_train, train_pred, y_val, val_pred)
            else:
                metrics = self.calculate_classification_metrics(y_train, train_pred, y_val, val_pred)
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ {model_name} otimizado em {optimization_time:.2f}s | Score: {search.best_score_:.4f}")
            
            return {
                 'success': True,
                 'model_name': model_name,
                 'best_model': best_model,
                 'best_params': search.best_params_,
                 'best_score': search.best_score_,
                 'metrics': metrics,
                 'optimization_method': optimization_method,
                 'optimization_time': optimization_time,
                 'feature_importance': self.get_feature_importance(best_model)
             }
        except Exception as e:
            logger.error(f"❌ Erro na otimização de {model_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    def calculate_regression_metrics(self, y_train_true: np.ndarray, y_train_pred: np.ndarray,
                                   y_val_true: np.ndarray, y_val_pred: np.ndarray) -> Dict:
        """Calcula métricas para modelos de regressão"""
        return {
            'train_r2': r2_score(y_train_true, y_train_pred),
            'val_r2': r2_score(y_val_true, y_val_pred),
            'train_mse': mean_squared_error(y_train_true, y_train_pred),
            'val_mse': mean_squared_error(y_val_true, y_val_pred),
            'train_mae': mean_absolute_error(y_train_true, y_train_pred),
            'val_mae': mean_absolute_error(y_val_true, y_val_pred)
        }
    
    def calculate_classification_metrics(self, y_train_true: np.ndarray, y_train_pred: np.ndarray,
                                       y_val_true: np.ndarray, y_val_pred: np.ndarray) -> Dict:
        """Calcula métricas para modelos de classificação"""
        return {
            'train_accuracy': accuracy_score(y_train_true, y_train_pred),
            'val_accuracy': accuracy_score(y_val_true, y_val_pred),
            'train_f1': f1_score(y_train_true, y_train_pred, average='weighted'),
            'val_f1': f1_score(y_val_true, y_val_pred, average='weighted'),
            'train_precision': precision_score(y_train_true, y_train_pred, average='weighted', zero_division=0),
            'val_precision': precision_score(y_val_true, y_val_pred, average='weighted', zero_division=0),
            'train_recall': recall_score(y_train_true, y_train_pred, average='weighted', zero_division=0),
            'val_recall': recall_score(y_val_true, y_val_pred, average='weighted', zero_division=0)
        }
    
    def get_feature_importance(self, model) -> Optional[List[float]]:
        """Extrai importância das features do modelo"""
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                return np.abs(model.coef_).tolist()
            else:
                return None
        except:
            return None
    
    def save_optimized_model(self, optimization_result: Dict, pair_name: str, target_period: int = 1) -> str:
        """Salva modelo otimizado e metadados"""
        pair_dir = self.optimized_models_directory / pair_name.upper()
        pair_dir.mkdir(exist_ok=True)
        
        model_name = optimization_result['model_name']
        
        # Salvar modelo
        model_file = pair_dir / f"{model_name}_{self.optimization_method}_target_{target_period}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(optimization_result['best_model'], f)
        
        # Salvar metadados
        metadata = {
            'optimization_method': self.optimization_method,
            'model_name': model_name,
            'pair': pair_name.upper(),
            'target_period': target_period,
            'best_params': optimization_result['best_params'],
            'best_score': optimization_result['best_score'],
            'metrics': optimization_result['metrics'],
            'optimization_time': optimization_result['optimization_time'],
            'feature_importance': optimization_result['feature_importance'],
            'timestamp': datetime.now().isoformat(),
            'timeframe': self.timeframe
        }
        
        metadata_file = pair_dir / f"{model_name}_{self.optimization_method}_target_{target_period}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Modelo otimizado salvo: {model_file}")
        
        return str(model_file)

def main():
    parser = argparse.ArgumentParser(description='Otimizador de Hiperparâmetros FX com Checkpoints')
    parser.add_argument('--timeframe', type=str, default='M1', 
                       help='Timeframe dos dados (M1, M5, M15, M30)')
    parser.add_argument('--continue', action='store_true', dest='continue_opt',
                       help='Continuar otimização de onde parou')
    parser.add_argument('--status', action='store_true',
                       help='Mostrar apenas o status atual')
    parser.add_argument('--method', type=str, default='randomized', choices=['randomized', 'hybrid'],
                       help='Método de otimização dos hiperparâmetros (randomized ou hybrid)')
    
    args = parser.parse_args()
    
    # Configuração de diretórios
    data_dir = "processed_fx_data"
    models_dir = "trained_models"
    optimized_models_dir = "optimized_models"
    
    if args.timeframe != "M1":
        data_dir = f"{data_dir}_{args.timeframe}"
        models_dir = f"{models_dir}_{args.timeframe}"
        optimized_models_dir = f"{optimized_models_dir}_{args.timeframe}"
    
    if not Path(data_dir).exists():
        print(f"❌ Diretório não encontrado: {data_dir}")
        print("Execute primeiro o pipeline de pré-processamento")
        return
    
    # Inicializar otimizador
    optimizer = FXHyperparameterOptimizerResumable(
        data_dir, models_dir, optimized_models_dir, args.timeframe, args.method
    )
    
    print(f"🔧 Otimizador de Hiperparâmetros FX - Timeframe: {args.timeframe}")
    print("=" * 60)
    
    if args.status:
        # Apenas mostrar status
        optimizer.print_progress_summary()
        return
    
    if args.continue_opt:
        # Continuar de onde parou
        optimizer.continue_optimization_from_checkpoint()
    else:
        # Iniciar nova otimização (mas pular modelos já otimizados)
        print("\n🚀 Iniciando otimização (pulando modelos já otimizados)...")
        optimizer.continue_optimization_from_checkpoint()
    
    print(f"\n💾 Modelos otimizados salvos em: {optimized_models_dir}")
    print("✅ Otimização concluída!")

if __name__ == "__main__":
    main()