#!/usr/bin/env python3
"""
Treinador de Modelos FX - Machine Learning para Forex
Treina múltiplos modelos ML para previsão de movimentos de preços FX
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

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb

warnings.filterwarnings('ignore')

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fx_model_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FXModelTrainer:
    def __init__(self, data_directory: str = "processed_fx_data", models_directory: str = "trained_models"):
        self.data_directory = Path(data_directory)
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(exist_ok=True)
        
        # Pares prioritários baseados no pré-processamento
        self.top_5_pairs = ['usdjpy', 'gbpusd', 'eurusd', 'usdchf', 'audjpy']
        self.all_10_pairs = [
            'usdjpy', 'gbpusd', 'eurusd', 'usdchf', 'audjpy',
            'gbpjpy', 'xauusd', 'eurjpy', 'usdsek', 'usdnok'
        ]
        
        # Configuração de modelos
        self.model_configs = {
            'random_forest_reg': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'type': 'regression'
            },
            'random_forest_clf': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'type': 'classification'
            },
            'xgboost_reg': {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'type': 'regression'
            },
            'xgboost_clf': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'type': 'classification'
            },
            'gradient_boosting_reg': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'random_state': 42
                },
                'type': 'regression'
            }
        }
        
        self.training_results = {}
        
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
        """Prepara features e targets para treinamento"""
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
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Treina um modelo específico"""
        logger.info(f"🤖 Treinando modelo {model_name}...")
        
        config = self.model_configs[model_name]
        model_class = config['model']
        params = config['params']
        model_type = config['type']
        
        start_time = datetime.now()
        
        try:
            # Inicializa e treina o modelo
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            # Faz previsões
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Calcula métricas
            if model_type == 'regression':
                metrics = self.calculate_regression_metrics(y_train, y_train_pred, y_val, y_val_pred)
            else:
                metrics = self.calculate_classification_metrics(y_train, y_train_pred, y_val, y_val_pred)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'model': model,
                'model_name': model_name,
                'model_type': model_type,
                'training_time': training_time,
                'metrics': metrics,
                'feature_importance': self.get_feature_importance(model),
                'success': True
            }
            
            logger.info(f"✅ {model_name} treinado em {training_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro ao treinar {model_name}: {e}")
            return {
                'model_name': model_name,
                'success': False,
                'error': str(e)
            }
    
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
    
    def save_model(self, model_result: Dict, pair_name: str, target_period: int = 1) -> str:
        """Salva modelo treinado"""
        if not model_result.get('success', False):
            return ""
        
        pair_models_dir = self.models_directory / pair_name.upper()
        pair_models_dir.mkdir(exist_ok=True)
        
        model_name = model_result['model_name']
        model_file = pair_models_dir / f"{model_name}_target_{target_period}.pkl"
        
        # Salva o modelo
        with open(model_file, 'wb') as f:
            pickle.dump(model_result['model'], f)
        
        # Salva metadados
        metadata = {
            'model_name': model_name,
            'model_type': model_result['model_type'],
            'pair': pair_name.upper(),
            'target_period': target_period,
            'training_time': model_result['training_time'],
            'metrics': model_result['metrics'],
            'feature_importance': model_result['feature_importance'],
            'timestamp': datetime.now().isoformat(),
            'model_file': str(model_file)
        }
        
        metadata_file = pair_models_dir / f"{model_name}_target_{target_period}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Modelo salvo: {model_file}")
        return str(model_file)
    
    def train_pair_models(self, pair_name: str, target_period: int = 1) -> Dict:
        """Treina todos os modelos para um par específico"""
        logger.info(f"🎯 Treinando modelos para {pair_name.upper()}...")
        
        # Carrega dados
        train_df, val_df, test_df = self.load_processed_data(pair_name)
        if train_df.empty:
            return {'success': False, 'error': 'Falha ao carregar dados'}
        
        # Prepara features e targets
        X_train, y_train_reg, y_train_clf = self.prepare_features_and_targets(train_df, target_period)
        X_val, y_val_reg, y_val_clf = self.prepare_features_and_targets(val_df, target_period)
        
        if X_train.shape[0] == 0:
            return {'success': False, 'error': 'Nenhuma amostra válida encontrada'}
        
        # Treina modelos
        pair_results = {
            'pair': pair_name.upper(),
            'target_period': target_period,
            'models': {},
            'best_regression_model': None,
            'best_classification_model': None
        }
        
        best_reg_score = -np.inf
        best_clf_score = -np.inf
        
        for model_name, config in self.model_configs.items():
            if config['type'] == 'regression':
                result = self.train_model(model_name, X_train, y_train_reg, X_val, y_val_reg)
                if result.get('success', False):
                    # Salva modelo
                    model_file = self.save_model(result, pair_name, target_period)
                    result['model_file'] = model_file
                    
                    # Verifica se é o melhor modelo de regressão
                    val_r2 = result['metrics']['val_r2']
                    if val_r2 > best_reg_score:
                        best_reg_score = val_r2
                        pair_results['best_regression_model'] = model_name
                
            else:  # classification
                result = self.train_model(model_name, X_train, y_train_clf, X_val, y_val_clf)
                if result.get('success', False):
                    # Salva modelo
                    model_file = self.save_model(result, pair_name, target_period)
                    result['model_file'] = model_file
                    
                    # Verifica se é o melhor modelo de classificação
                    val_f1 = result['metrics']['val_f1']
                    if val_f1 > best_clf_score:
                        best_clf_score = val_f1
                        pair_results['best_classification_model'] = model_name
            
            # Remove o objeto modelo para economizar memória
            if 'model' in result:
                del result['model']
            
            pair_results['models'][model_name] = result
        
        # Calcula estatísticas do par
        successful_models = [m for m in pair_results['models'].values() if m.get('success', False)]
        pair_results['summary'] = {
            'total_models': len(self.model_configs),
            'successful_models': len(successful_models),
            'training_samples': X_train.shape[0],
            'validation_samples': X_val.shape[0],
            'features_count': X_train.shape[1],
            'best_regression_r2': best_reg_score if best_reg_score > -np.inf else None,
            'best_classification_f1': best_clf_score if best_clf_score > -np.inf else None
        }
        
        pair_results['success'] = len(successful_models) > 0
        logger.info(f"✅ {pair_name.upper()}: {len(successful_models)}/{len(self.model_configs)} modelos treinados")
        
        return pair_results
    
    def train_top_5_pairs(self, target_period: int = 1) -> Dict:
        """Treina modelos para os 5 principais pares"""
        logger.info("🎯 Treinando modelos para TOP 5 pares...")
        
        results = {}
        for pair in self.top_5_pairs:
            results[pair] = self.train_pair_models(pair, target_period)
        
        return results
    
    def train_all_pairs(self, target_period: int = 1) -> Dict:
        """Treina modelos para todos os pares disponíveis"""
        logger.info("📊 Treinando modelos para todos os pares...")
        
        results = {}
        for pair in self.all_10_pairs:
            results[pair] = self.train_pair_models(pair, target_period)
        
        return results
    
    def generate_training_report(self, results: Dict, target_period: int = 1) -> str:
        """Gera relatório de treinamento"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"model_training_report_{timestamp}.json"
        
        # Calcula estatísticas gerais
        successful_pairs = [p for p, r in results.items() if r.get('success', False)]
        total_models = sum([r.get('summary', {}).get('successful_models', 0) for r in results.values()])
        
        # Encontra melhores modelos globais
        best_models = {
            'regression': {'pair': None, 'model': None, 'r2': -np.inf},
            'classification': {'pair': None, 'model': None, 'f1': -np.inf}
        }
        
        for pair, result in results.items():
            if result.get('success', False):
                summary = result.get('summary', {})
                
                # Melhor regressão
                if summary.get('best_regression_r2') and summary['best_regression_r2'] > best_models['regression']['r2']:
                    best_models['regression'] = {
                        'pair': pair,
                        'model': result.get('best_regression_model'),
                        'r2': summary['best_regression_r2']
                    }
                
                # Melhor classificação
                if summary.get('best_classification_f1') and summary['best_classification_f1'] > best_models['classification']['f1']:
                    best_models['classification'] = {
                        'pair': pair,
                        'model': result.get('best_classification_model'),
                        'f1': summary['best_classification_f1']
                    }
        
        report = {
            'timestamp': timestamp,
            'target_period': target_period,
            'training_summary': {
                'total_pairs_trained': len(results),
                'successful_pairs': len(successful_pairs),
                'total_models_trained': total_models,
                'models_directory': str(self.models_directory),
                'best_models': best_models
            },
            'pair_results': results,
            'model_configs': self.model_configs,
            'recommendations': self.generate_model_recommendations(results)
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📋 Relatório de treinamento salvo: {report_file}")
        return report_file
    
    def generate_model_recommendations(self, results: Dict) -> Dict:
        """Gera recomendações baseadas nos resultados"""
        successful_pairs = [p for p, r in results.items() if r.get('success', False)]
        
        # Analisa performance dos modelos
        model_performance = {}
        for pair, result in results.items():
            if result.get('success', False):
                for model_name, model_result in result.get('models', {}).items():
                    if model_result.get('success', False):
                        if model_name not in model_performance:
                            model_performance[model_name] = []
                        
                        metrics = model_result.get('metrics', {})
                        if 'val_r2' in metrics:
                            model_performance[model_name].append(metrics['val_r2'])
                        elif 'val_f1' in metrics:
                            model_performance[model_name].append(metrics['val_f1'])
        
        # Calcula performance média
        avg_performance = {}
        for model_name, scores in model_performance.items():
            if scores:
                avg_performance[model_name] = np.mean(scores)
        
        # Ordena modelos por performance
        best_models = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = {
            'production_ready_pairs': successful_pairs[:5],
            'best_performing_models': best_models[:3],
            'next_steps': [
                'Avaliar modelos em dados de teste',
                'Implementar ensemble de melhores modelos',
                'Configurar pipeline de previsão em tempo real',
                'Otimizar hiperparâmetros dos melhores modelos'
            ],
            'model_selection_criteria': {
                'regression': 'R² Score (explicação da variância)',
                'classification': 'F1 Score (balanceamento precisão/recall)'
            }
        }
        
        return recommendations
    
    def print_training_summary(self, results: Dict):
        """Imprime resumo do treinamento"""
        print("\n" + "="*60)
        print("🤖 RESUMO DO TREINAMENTO DE MODELOS")
        print("="*60)
        
        successful_pairs = [p for p, r in results.items() if r.get('success', False)]
        total_models = sum([r.get('summary', {}).get('successful_models', 0) for r in results.values()])
        
        print(f"\n✅ Pares Treinados: {len(successful_pairs)}/{len(results)}")
        print(f"🤖 Total de Modelos: {total_models}")
        
        if successful_pairs:
            print("\n🏆 RESULTADOS POR PAR:")
            for pair in successful_pairs:
                result = results[pair]
                summary = result.get('summary', {})
                successful_models = summary.get('successful_models', 0)
                best_reg_r2 = summary.get('best_regression_r2', 0)
                best_clf_f1 = summary.get('best_classification_f1', 0)
                
                print(f"{pair.upper()}: {successful_models} modelos | "
                      f"Melhor R²: {best_reg_r2:.3f} | Melhor F1: {best_clf_f1:.3f}")
        
        if len(successful_pairs) >= 3:
            print("\n🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
            print("✅ Modelos prontos para avaliação e produção")
        else:
            print("\n⚠️ ATENÇÃO: Poucos pares treinados com sucesso")
            print("❌ Revisar configuração antes de prosseguir")

def main():
    # Parser de argumentos
    import argparse
    parser = argparse.ArgumentParser(description="Treinamento de modelos FX")
    parser.add_argument("--data_dir", default="processed_fx_data", help="Diretório dos dados processados (ex.: processed_fx_data_M5)")
    parser.add_argument("--models_dir", default="trained_models", help="Diretório para salvar modelos")
    parser.add_argument("--pairs", default="", help="Lista de pares separados por vírgula para treinar (ex.: EURUSD,USDJPY)")
    parser.add_argument("--target_period", type=int, default=1, help="Horizonte de target (ex.: 1)")
    args = parser.parse_args()

    # Configuração a partir dos argumentos
    data_dir = args.data_dir
    models_dir = args.models_dir
    target_period = args.target_period
    selected_pairs = [p.strip().lower() for p in args.pairs.split(',') if p.strip()] if args.pairs else []
    
    if not Path(data_dir).exists():
        print(f"❌ Diretório não encontrado: {data_dir}")
        print("Execute primeiro o pipeline de pré-processamento")
        return
    
    # Inicializar trainer
    trainer = FXModelTrainer(data_dir, models_dir)
    
    # Se pares foram informados, restringe o treinamento a eles
    if selected_pairs:
        trainer.top_5_pairs = selected_pairs
        trainer.all_10_pairs = selected_pairs
    
    print("🚀 Iniciando Treinamento de Modelos FX")
    print("="*50)
    print(f"📁 Dados: {data_dir} | 💾 Modelos: {models_dir} | 🎯 Target: {target_period}")
    if selected_pairs:
        print(f"🎯 Pares selecionados: {', '.join([p.upper() for p in selected_pairs])}")
    
    # Treinar TOP 5 (ou pares selecionados)
    print("\n🎯 Fase 1: Treinando TOP 5 pares...")
    top5_results = trainer.train_top_5_pairs(target_period=target_period)
    
    # Verificar se TOP 5 foi bem-sucedido
    successful_top5 = [p for p, r in top5_results.items() if r.get('success', False)]
    
    # Só expande se não houver restrição explícita de pares e houver sucesso suficiente
    if not selected_pairs and len(successful_top5) >= 3:
        print("\n📊 Fase 2: Expandindo para pares adicionais...")
        # Treinar pares restantes
        all_results = trainer.train_all_pairs(target_period=target_period)
    else:
        if not selected_pairs:
            print("\n⚠️ Problemas com TOP 5 pares. Continuando apenas com os bem-sucedidos...")
        all_results = top5_results
    
    # Gerar relatório
    report_file = trainer.generate_training_report(all_results, target_period=target_period)
    
    # Imprimir resumo
    trainer.print_training_summary(all_results)
    
    print(f"\n📋 Relatório detalhado: {report_file}")
    print(f"💾 Modelos salvos em: {models_dir}")
    print("\n🎯 Próximo passo: Otimização de hiperparâmetros")

if __name__ == "__main__":
    main()