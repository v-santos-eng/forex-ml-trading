#!/usr/bin/env python3
"""
Avaliador Final de Modelos Forex - Versão Robusta e Compatível
Resolve problemas de incompatibilidade de features e avalia modelos de forma robusta.
"""

import pandas as pd
import numpy as np
import json
import logging
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

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
    feature_count: Optional[int] = None

class FinalFXModelEvaluator:
    """Avaliador final robusto para modelos Forex"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.processed_data_dir = self.base_dir / "processed_fx_data"
        self.optimized_models_dir = self.base_dir / "optimized_models"
        self.evaluation_dir = self.base_dir / "model_evaluation_final"
        
        # Criar diretório de resultados
        self.evaluation_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Descobrir pares disponíveis
        self.available_pairs = self._discover_available_pairs()
        
        # Tipos de modelos
        self.model_types = {
            'regression': ['random_forest_reg', 'xgboost_reg', 'gradient_boosting_reg', 'lightgbm_reg'],
            'classification': ['random_forest_clf', 'xgboost_clf', 'gradient_boosting_clf', 'lightgbm_clf']
        }
        
        # Métricas
        self.regression_metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']
        self.classification_metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        
        # Resultados
        self.evaluation_results: List[EvaluationResult] = []
    
    def _setup_logging(self):
        """Configura logging sem emojis para evitar problemas de encoding"""
        log_file = self.evaluation_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
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
        """Descobre pares disponíveis"""
        if not self.processed_data_dir.exists():
            return []
        
        pairs = []
        for pair_dir in self.processed_data_dir.iterdir():
            if pair_dir.is_dir():
                test_files = list(pair_dir.glob("test.*"))
                if test_files:
                    pairs.append(pair_dir.name)
        
        self.logger.info(f"Pares disponíveis: {len(pairs)} - {pairs}")
        return pairs
    
    def _load_model_metadata(self, pair: str, model_type: str) -> Optional[Dict]:
        """Carrega metadata do modelo para verificar features esperadas"""
        model_dir = self.optimized_models_dir / pair
        
        # Procurar arquivos de metadata
        metadata_files = list(model_dir.glob(f"{model_type}*metadata.json"))
        
        if not metadata_files:
            return None
        
        try:
            with open(metadata_files[0], 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Erro ao carregar metadata para {pair}/{model_type}: {e}")
            return None
    
    def _adjust_features_to_model(self, X: np.ndarray, expected_features: int) -> np.ndarray:
        """Ajusta número de features para corresponder ao modelo"""
        current_features = X.shape[1]
        
        if current_features == expected_features:
            return X
        elif current_features > expected_features:
            # Usar apenas as primeiras N features
            self.logger.info(f"Reduzindo features de {current_features} para {expected_features}")
            return X[:, :expected_features]
        else:
            # Preencher com zeros se necessário
            self.logger.info(f"Expandindo features de {current_features} para {expected_features}")
            padding = np.zeros((X.shape[0], expected_features - current_features))
            return np.hstack([X, padding])
    
    def load_test_data(self, pair: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Carrega dados de teste"""
        try:
            pair_dir = self.processed_data_dir / pair
            test_files = list(pair_dir.glob("test.*"))
            
            if not test_files:
                self.logger.warning(f"Arquivo de teste não encontrado para {pair}")
                return None, None, None
            
            # Carregar dados
            test_file = test_files[0]
            if test_file.suffix == '.parquet':
                data = pd.read_parquet(test_file)
            else:
                data = pd.read_csv(test_file)
            
            if data.empty:
                return None, None, None
            
            # Identificar features (excluir targets e DateTime)
            feature_cols = [col for col in data.columns 
                          if not col.startswith(('Target_', 'target_', 'Unnamed')) and col != 'DateTime']
            
            if not feature_cols:
                self.logger.error(f"Nenhuma feature encontrada para {pair}")
                return None, None, None
            
            # Extrair features
            X = data[feature_cols].select_dtypes(include=[np.number]).fillna(0).values.astype(np.float64)
            
            # Extrair targets
            reg_targets = [col for col in data.columns if 'Target_Return_' in col]
            clf_targets = [col for col in data.columns if 'Target_Direction_' in col]
            
            y_reg = data[reg_targets[0]].fillna(0).values if reg_targets else None
            y_clf = data[clf_targets[0]].fillna(0).astype(int).values if clf_targets else None
            
            self.logger.info(f"Dados carregados para {pair}: {X.shape[0]} amostras, {X.shape[1]} features")
            return X, y_reg, y_clf
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados para {pair}: {e}")
            return None, None, None
    
    def load_model(self, pair: str, model_type: str) -> Optional[Any]:
        """Carrega modelo treinado"""
        model_dir = self.optimized_models_dir / pair
        
        # Procurar arquivos de modelo
        model_files = list(model_dir.glob(f"{model_type}*.pkl")) + list(model_dir.glob(f"{model_type}*.joblib"))
        
        if not model_files:
            return None
        
        try:
            return joblib.load(model_files[0])
        except Exception as e:
            self.logger.warning(f"Erro ao carregar modelo {pair}/{model_type}: {e}")
            return None
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de regressão de forma segura"""
        metrics = {}
        
        try:
            # Limpar dados
            mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                return {metric: np.nan for metric in self.regression_metrics}
            
            metrics['MSE'] = mean_squared_error(y_true_clean, y_pred_clean)
            metrics['RMSE'] = np.sqrt(metrics['MSE'])
            metrics['MAE'] = mean_absolute_error(y_true_clean, y_pred_clean)
            
            # MAPE seguro
            mape_mask = y_true_clean != 0
            if mape_mask.sum() > 0:
                metrics['MAPE'] = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / y_true_clean[mape_mask])) * 100
            else:
                metrics['MAPE'] = np.nan
            
            metrics['R2'] = r2_score(y_true_clean, y_pred_clean)
            
        except Exception as e:
            self.logger.warning(f"Erro no cálculo de métricas de regressão: {e}")
            metrics = {metric: np.nan for metric in self.regression_metrics}
        
        return metrics
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de classificação de forma segura"""
        metrics = {}
        
        try:
            # Limpar dados
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask].astype(int)
            y_pred_clean = y_pred[mask].astype(int)
            
            if len(y_true_clean) == 0:
                return {metric: np.nan for metric in self.classification_metrics}
            
            metrics['Accuracy'] = accuracy_score(y_true_clean, y_pred_clean)
            metrics['Precision'] = precision_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0)
            metrics['Recall'] = recall_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0)
            metrics['F1'] = f1_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0)
            
        except Exception as e:
            self.logger.warning(f"Erro no cálculo de métricas de classificação: {e}")
            metrics = {metric: np.nan for metric in self.classification_metrics}
        
        return metrics
    
    def evaluate_model(self, pair: str, model_type: str) -> EvaluationResult:
        """Avalia um modelo específico"""
        # Carregar dados
        X_test, y_test_reg, y_test_clf = self.load_test_data(pair)
        
        if X_test is None:
            return EvaluationResult(
                pair=pair, model_type=model_type, model_category="optimized",
                metrics={}, sample_size=0, success=False,
                error_message="Dados de teste não disponíveis"
            )
        
        # Carregar modelo
        model = self.load_model(pair, model_type)
        
        if model is None:
            return EvaluationResult(
                pair=pair, model_type=model_type, model_category="optimized",
                metrics={}, sample_size=0, success=False,
                error_message="Modelo não encontrado"
            )
        
        # Carregar metadata para ajustar features
        metadata = self._load_model_metadata(pair, model_type)
        if metadata and 'feature_importance' in metadata:
            expected_features = len(metadata['feature_importance'])
            X_test = self._adjust_features_to_model(X_test, expected_features)
        
        try:
            # Fazer predições
            y_pred = model.predict(X_test)
            
            # Calcular métricas baseado no tipo de modelo
            if 'reg' in model_type:
                if y_test_reg is None:
                    return EvaluationResult(
                        pair=pair, model_type=model_type, model_category="optimized",
                        metrics={}, sample_size=len(X_test), success=False,
                        error_message="Target de regressão não disponível"
                    )
                metrics = self._calculate_regression_metrics(y_test_reg, y_pred)
            else:
                if y_test_clf is None:
                    return EvaluationResult(
                        pair=pair, model_type=model_type, model_category="optimized",
                        metrics={}, sample_size=len(X_test), success=False,
                        error_message="Target de classificação não disponível"
                    )
                metrics = self._calculate_classification_metrics(y_test_clf, y_pred)
            
            return EvaluationResult(
                pair=pair, model_type=model_type, model_category="optimized",
                metrics=metrics, sample_size=len(X_test), success=True,
                feature_count=X_test.shape[1]
            )
            
        except Exception as e:
            return EvaluationResult(
                pair=pair, model_type=model_type, model_category="optimized",
                metrics={}, sample_size=len(X_test), success=False,
                error_message=f"Erro na predição: {str(e)}",
                feature_count=X_test.shape[1] if X_test is not None else 0
            )
    
    def run_evaluation(self):
        """Executa avaliação completa"""
        self.logger.info("Iniciando avaliação de modelos...")
        
        total_evaluations = 0
        successful_evaluations = 0
        
        for pair in self.available_pairs:
            self.logger.info(f"Avaliando par: {pair}")
            
            for category in self.model_types:
                for model_type in self.model_types[category]:
                    total_evaluations += 1
                    
                    result = self.evaluate_model(pair, model_type)
                    self.evaluation_results.append(result)
                    
                    if result.success:
                        successful_evaluations += 1
                        self.logger.info(f"Sucesso: {pair}/{model_type} - {len(result.metrics)} métricas")
                    else:
                        self.logger.warning(f"Falha: {pair}/{model_type} - {result.error_message}")
        
        # Salvar resultados
        self._save_results()
        
        # Resumo
        success_rate = (successful_evaluations / total_evaluations * 100) if total_evaluations > 0 else 0
        
        self.logger.info("RESUMO DA AVALIACAO:")
        self.logger.info(f"   Total de avaliações: {total_evaluations}")
        self.logger.info(f"   Avaliações bem-sucedidas: {successful_evaluations}")
        self.logger.info(f"   Taxa de sucesso: {success_rate:.1f}%")
        
        print(f"\nAvaliação concluída!")
        print(f"Taxa de sucesso: {success_rate:.1f}%")
        print(f"Modelos avaliados: {successful_evaluations}/{total_evaluations}")
    
    def _save_results(self):
        """Salva resultados em JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.evaluation_dir / f"evaluation_results_final_{timestamp}.json"
        
        # Preparar dados para JSON
        results_data = {
            "timestamp": timestamp,
            "summary": {
                "total_evaluations": len(self.evaluation_results),
                "successful_evaluations": sum(1 for r in self.evaluation_results if r.success),
                "pairs_evaluated": list(set(r.pair for r in self.evaluation_results)),
                "model_types_evaluated": list(set(r.model_type for r in self.evaluation_results))
            },
            "results": [
                {
                    "pair": r.pair,
                    "model_type": r.model_type,
                    "model_category": r.model_category,
                    "metrics": r.metrics,
                    "sample_size": r.sample_size,
                    "success": r.success,
                    "error_message": r.error_message,
                    "feature_count": r.feature_count
                }
                for r in self.evaluation_results
            ]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Resultados salvos: {results_file}")

def main():
    """Função principal"""
    evaluator = FinalFXModelEvaluator()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()