#!/usr/bin/env python3
"""
Avaliador de Modelos FX Ultra-Robusto
Versão que trata adequadamente todos os tipos de dados e erros
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Métricas
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_percentage_error
)
from scipy import stats

@dataclass
class EvaluationResult:
    """Resultado de avaliação de um modelo"""
    pair: str
    model_type: str
    model_category: str
    metrics: Dict[str, float]
    sample_size: int
    success: bool
    error_message: Optional[str] = None

class RobustFXModelEvaluator:
    """
    Avaliador ultra-robusto para modelos FX
    Trata adequadamente todos os tipos de dados e possíveis erros
    """
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        
        # Diretórios
        self.processed_data_dir = self.base_dir / "processed_fx_data"
        self.trained_models_dir = self.base_dir / "trained_models"
        self.optimized_models_dir = self.base_dir / "optimized_models"
        self.evaluation_dir = self.base_dir / "model_evaluation_robust"
        
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
            'MSE': self._safe_mse,
            'RMSE': self._safe_rmse,
            'MAE': self._safe_mae,
            'MAPE': self._safe_mape,
            'R2': self._safe_r2
        }
        
        self.classification_metrics = {
            'Accuracy': self._safe_accuracy,
            'Precision': self._safe_precision,
            'Recall': self._safe_recall,
            'F1': self._safe_f1
        }
        
        # Resultados
        self.evaluation_results: List[EvaluationResult] = []
    
    def _setup_logging(self):
        """Configura sistema de logging"""
        log_file = self.evaluation_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _discover_available_pairs(self) -> List[str]:
        """Descobre pares de moedas disponíveis"""
        if not self.processed_data_dir.exists():
            self.logger.warning(f"Diretório de dados não encontrado: {self.processed_data_dir}")
            return []
        
        pairs = []
        for pair_dir in self.processed_data_dir.iterdir():
            if pair_dir.is_dir():
                # Verificar se tem arquivos de teste
                test_files = list(pair_dir.glob("test.*"))
                if test_files:
                    pairs.append(pair_dir.name)
        
        self.logger.info(f"📊 Pares disponíveis: {len(pairs)} - {pairs}")
        return pairs
    
    def _safe_convert_to_numeric(self, data: Any) -> np.ndarray:
        """Converte dados para array numérico de forma segura"""
        try:
            # Se já é numpy array
            if isinstance(data, np.ndarray):
                # Verificar se é numérico
                if data.dtype.kind in 'biufc':  # bool, int, uint, float, complex
                    return data.astype(np.float64)
                else:
                    # Tentar converter
                    return pd.to_numeric(data, errors='coerce').astype(np.float64)
            
            # Se é pandas Series/DataFrame
            elif isinstance(data, pd.DataFrame):
                # Para DataFrame, converter cada coluna e manter estrutura 2D
                numeric_df = data.apply(pd.to_numeric, errors='coerce')
                return numeric_df.values.astype(np.float64)
            elif isinstance(data, pd.Series):
                return pd.to_numeric(data, errors='coerce').values.astype(np.float64)
            
            # Se é lista ou outro tipo
            else:
                return pd.to_numeric(pd.Series(data), errors='coerce').values.astype(np.float64)
                
        except Exception as e:
            self.logger.warning(f"Erro na conversão numérica: {e}")
            # Fallback: criar array de zeros
            if hasattr(data, '__len__'):
                return np.zeros(len(data), dtype=np.float64)
            else:
                return np.array([0.0], dtype=np.float64)
    
    def _clean_data(self, data: np.ndarray) -> np.ndarray:
        """Limpa dados removendo valores inválidos"""
        try:
            # Converter para float64 se necessário
            if data.dtype != np.float64:
                data = data.astype(np.float64)
            
            # Substituir inf por valores finitos
            data = np.where(np.isinf(data), np.nan, data)
            
            # Substituir NaN por mediana (ou 0 se todos forem NaN)
            if np.isnan(data).any():
                median_val = np.nanmedian(data)
                if np.isnan(median_val):
                    median_val = 0.0
                data = np.where(np.isnan(data), median_val, data)
            
            return data
            
        except Exception as e:
            self.logger.warning(f"Erro na limpeza de dados: {e}")
            # Fallback: array de zeros
            return np.zeros_like(data, dtype=np.float64)
    
    def load_test_data(self, pair: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Carrega dados de teste de forma ultra-robusta"""
        try:
            pair_dir = self.processed_data_dir / pair
            
            if not pair_dir.exists():
                self.logger.warning(f"⚠️ Diretório não encontrado: {pair_dir}")
                return None, None, None
            
            # Procurar arquivos de teste
            test_files = list(pair_dir.glob("test.*"))
            
            if not test_files:
                self.logger.warning(f"⚠️ Arquivo de teste não encontrado para {pair}")
                return None, None, None
            
            # Carregar dados
            test_file = test_files[0]
            
            try:
                if test_file.suffix == '.csv':
                    test_data = pd.read_csv(test_file)
                elif test_file.suffix == '.parquet':
                    test_data = pd.read_parquet(test_file)
                else:
                    self.logger.error(f"❌ Formato de arquivo não suportado: {test_file.suffix}")
                    return None, None, None
            except Exception as e:
                self.logger.error(f"❌ Erro ao carregar arquivo {test_file}: {e}")
                return None, None, None
            
            # Validar dados básicos
            if test_data.empty:
                self.logger.warning(f"⚠️ Dados vazios para {pair}")
                return None, None, None
            
            # Identificar colunas de features (excluir targets e DateTime)
            feature_cols = [col for col in test_data.columns 
                          if not col.startswith(('target_', 'Target_', 'Unnamed')) and col != 'DateTime']
            
            if not feature_cols:
                self.logger.error(f"❌ Nenhuma feature encontrada para {pair}")
                return None, None, None
            
            # Extrair features
            try:
                X_test_raw = test_data[feature_cols]
                X_test = self._safe_convert_to_numeric(X_test_raw)
                X_test = self._clean_data(X_test)
                
                # Verificar dimensões
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                
            except Exception as e:
                self.logger.error(f"❌ Erro ao processar features para {pair}: {e}")
                return None, None, None
            
            # Extrair targets de regressão
            y_test_reg = None
            reg_targets = [col for col in test_data.columns if 'Target_Return_' in col or 'Target_Price_' in col or 'Target_Magnitude_' in col]
            
            if reg_targets:
                try:
                    y_reg_raw = test_data[reg_targets[0]]
                    y_test_reg = self._safe_convert_to_numeric(y_reg_raw)
                    y_test_reg = self._clean_data(y_test_reg)
                except Exception as e:
                    self.logger.warning(f"⚠️ Erro ao processar targets de regressão para {pair}: {e}")
                    y_test_reg = None
            
            # Extrair targets de classificação
            y_test_clf = None
            clf_targets = [col for col in test_data.columns if 'Target_Direction_' in col]
            
            if clf_targets:
                try:
                    y_clf_raw = test_data[clf_targets[0]]
                    y_test_clf = self._safe_convert_to_numeric(y_clf_raw)
                    y_test_clf = self._clean_data(y_test_clf)
                    
                    # Converter para inteiros para classificação
                    y_test_clf = np.round(y_test_clf).astype(int)
                    
                    # Verificar se tem pelo menos 2 classes
                    unique_classes = np.unique(y_test_clf)
                    if len(unique_classes) < 2:
                        self.logger.warning(f"⚠️ Insuficientes classes para classificação em {pair}: {unique_classes}")
                        y_test_clf = None
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Erro ao processar targets de classificação para {pair}: {e}")
                    y_test_clf = None
            
            # Validar tamanhos
            if len(X_test) < 10:
                self.logger.warning(f"⚠️ Dados insuficientes para {pair}: {len(X_test)} amostras")
                return None, None, None
            
            self.logger.info(f"✅ Dados carregados para {pair}: {X_test.shape[0]} amostras, {X_test.shape[1]} features")
            return X_test, y_test_reg, y_test_clf
            
        except Exception as e:
            self.logger.error(f"❌ Erro geral ao carregar dados para {pair}: {e}")
            return None, None, None
    
    def load_model(self, pair: str, model_type: str, model_category: str = 'optimized') -> Optional[Any]:
        """Carrega modelo de forma robusta"""
        try:
            if model_category == 'optimized':
                model_dir = self.optimized_models_dir / pair
            else:
                model_dir = self.trained_models_dir / pair
            
            if not model_dir.exists():
                return None
            
            # Procurar arquivo do modelo
            model_files = (list(model_dir.glob(f"*{model_type}*.pkl")) + 
                          list(model_dir.glob(f"*{model_type}*.joblib")))
            
            if not model_files:
                return None
            
            model_file = model_files[0]
            
            # Carregar modelo
            if model_file.suffix == '.pkl':
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
            else:
                model = joblib.load(model_file)
            
            # Validar modelo
            if not hasattr(model, 'predict'):
                self.logger.error(f"❌ Modelo sem método predict: {model_file}")
                return None
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar modelo {pair}/{model_type}: {e}")
            return None
    
    # Métricas seguras
    def _safe_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            y_true = self._clean_data(self._safe_convert_to_numeric(y_true))
            y_pred = self._clean_data(self._safe_convert_to_numeric(y_pred))
            return float(mean_squared_error(y_true, y_pred))
        except:
            return float('inf')
    
    def _safe_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            mse = self._safe_mse(y_true, y_pred)
            return float(np.sqrt(mse)) if mse != float('inf') else float('inf')
        except:
            return float('inf')
    
    def _safe_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            y_true = self._clean_data(self._safe_convert_to_numeric(y_true))
            y_pred = self._clean_data(self._safe_convert_to_numeric(y_pred))
            return float(mean_absolute_error(y_true, y_pred))
        except:
            return float('inf')
    
    def _safe_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            y_true = self._clean_data(self._safe_convert_to_numeric(y_true))
            y_pred = self._clean_data(self._safe_convert_to_numeric(y_pred))
            # Evitar divisão por zero
            mask = np.abs(y_true) > 1e-8
            if not mask.any():
                return float('inf')
            return float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]))
        except:
            return float('inf')
    
    def _safe_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            y_true = self._clean_data(self._safe_convert_to_numeric(y_true))
            y_pred = self._clean_data(self._safe_convert_to_numeric(y_pred))
            return float(r2_score(y_true, y_pred))
        except:
            return -float('inf')
    
    def _safe_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            y_true = np.round(self._clean_data(self._safe_convert_to_numeric(y_true))).astype(int)
            y_pred = np.round(self._clean_data(self._safe_convert_to_numeric(y_pred))).astype(int)
            return float(accuracy_score(y_true, y_pred))
        except:
            return 0.0
    
    def _safe_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            y_true = np.round(self._clean_data(self._safe_convert_to_numeric(y_true))).astype(int)
            y_pred = np.round(self._clean_data(self._safe_convert_to_numeric(y_pred))).astype(int)
            return float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        except:
            return 0.0
    
    def _safe_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            y_true = np.round(self._clean_data(self._safe_convert_to_numeric(y_true))).astype(int)
            y_pred = np.round(self._clean_data(self._safe_convert_to_numeric(y_pred))).astype(int)
            return float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        except:
            return 0.0
    
    def _safe_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            y_true = np.round(self._clean_data(self._safe_convert_to_numeric(y_true))).astype(int)
            y_pred = np.round(self._clean_data(self._safe_convert_to_numeric(y_pred))).astype(int)
            return float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        except:
            return 0.0
    
    def evaluate_model(self, pair: str, model_type: str, task_type: str) -> Optional[EvaluationResult]:
        """Avalia um modelo específico"""
        try:
            # Carregar dados
            X_test, y_test_reg, y_test_clf = self.load_test_data(pair)
            
            if X_test is None:
                return EvaluationResult(
                    pair=pair, model_type=model_type, model_category='optimized',
                    metrics={}, sample_size=0, success=False,
                    error_message="Dados de teste não disponíveis"
                )
            
            # Selecionar target apropriado
            if task_type == 'regression':
                y_test = y_test_reg
                metrics_dict = self.regression_metrics
            else:
                y_test = y_test_clf
                metrics_dict = self.classification_metrics
            
            if y_test is None:
                return EvaluationResult(
                    pair=pair, model_type=model_type, model_category='optimized',
                    metrics={}, sample_size=0, success=False,
                    error_message=f"Target para {task_type} não disponível"
                )
            
            # Carregar modelo otimizado
            optimized_model = self.load_model(pair, model_type, 'optimized')
            
            if optimized_model is None:
                return EvaluationResult(
                    pair=pair, model_type=model_type, model_category='optimized',
                    metrics={}, sample_size=0, success=False,
                    error_message="Modelo otimizado não encontrado"
                )
            
            # Fazer predições
            try:
                y_pred = optimized_model.predict(X_test)
                y_pred = self._clean_data(self._safe_convert_to_numeric(y_pred))
            except Exception as e:
                return EvaluationResult(
                    pair=pair, model_type=model_type, model_category='optimized',
                    metrics={}, sample_size=len(X_test), success=False,
                    error_message=f"Erro na predição: {e}"
                )
            
            # Calcular métricas
            metrics = {}
            for metric_name, metric_func in metrics_dict.items():
                try:
                    metrics[metric_name] = metric_func(y_test, y_pred)
                except Exception as e:
                    self.logger.warning(f"Erro ao calcular {metric_name} para {pair}/{model_type}: {e}")
                    metrics[metric_name] = float('inf') if task_type == 'regression' else 0.0
            
            return EvaluationResult(
                pair=pair, model_type=model_type, model_category='optimized',
                metrics=metrics, sample_size=len(X_test), success=True
            )
            
        except Exception as e:
            return EvaluationResult(
                pair=pair, model_type=model_type, model_category='optimized',
                metrics={}, sample_size=0, success=False,
                error_message=f"Erro geral: {e}"
            )
    
    def run_evaluation(self):
        """Executa avaliação completa"""
        self.logger.info("🚀 Iniciando avaliação robusta de modelos...")
        
        total_evaluations = 0
        successful_evaluations = 0
        
        for pair in self.available_pairs:
            self.logger.info(f"\n🎯 Avaliando modelos para {pair}...")
            
            # Avaliar modelos de regressão
            for model_type in self.model_types['regression']:
                result = self.evaluate_model(pair, model_type, 'regression')
                if result:
                    self.evaluation_results.append(result)
                    total_evaluations += 1
                    if result.success:
                        successful_evaluations += 1
                        self.logger.info(f"✅ {pair}/{model_type} (reg): {result.metrics}")
                    else:
                        self.logger.warning(f"⚠️ {pair}/{model_type} (reg): {result.error_message}")
            
            # Avaliar modelos de classificação
            for model_type in self.model_types['classification']:
                result = self.evaluate_model(pair, model_type, 'classification')
                if result:
                    self.evaluation_results.append(result)
                    total_evaluations += 1
                    if result.success:
                        successful_evaluations += 1
                        self.logger.info(f"✅ {pair}/{model_type} (clf): {result.metrics}")
                    else:
                        self.logger.warning(f"⚠️ {pair}/{model_type} (clf): {result.error_message}")
        
        # Salvar resultados
        self._save_results()
        
        # Resumo final
        success_rate = (successful_evaluations / total_evaluations * 100) if total_evaluations > 0 else 0
        
        self.logger.info(f"\n📊 RESUMO DA AVALIAÇÃO:")
        self.logger.info(f"   Total de avaliações: {total_evaluations}")
        self.logger.info(f"   Avaliações bem-sucedidas: {successful_evaluations}")
        self.logger.info(f"   Taxa de sucesso: {success_rate:.1f}%")
        
        print(f"\n✅ Avaliação concluída!")
        print(f"📈 Taxa de sucesso: {success_rate:.1f}%")
        print(f"📊 Modelos avaliados: {successful_evaluations}/{total_evaluations}")
    
    def _save_results(self):
        """Salva resultados da avaliação"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Converter resultados para dicionário
        results_dict = {
            'timestamp': timestamp,
            'summary': {
                'total_evaluations': len(self.evaluation_results),
                'successful_evaluations': sum(1 for r in self.evaluation_results if r.success),
                'pairs_evaluated': list(set(r.pair for r in self.evaluation_results)),
                'model_types_evaluated': list(set(r.model_type for r in self.evaluation_results))
            },
            'results': []
        }
        
        for result in self.evaluation_results:
            results_dict['results'].append({
                'pair': result.pair,
                'model_type': result.model_type,
                'model_category': result.model_category,
                'metrics': result.metrics,
                'sample_size': result.sample_size,
                'success': result.success,
                'error_message': result.error_message
            })
        
        # Salvar JSON
        results_file = self.evaluation_dir / f"evaluation_results_robust_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Resultados salvos: {results_file}")

def main():
    evaluator = RobustFXModelEvaluator()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()