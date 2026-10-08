import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import logging
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from dataclasses import dataclass
from enum import Enum
import pickle
import warnings
from collections import defaultdict, deque

class SelectionStrategy(Enum):
    """Estratégias de seleção de modelo"""
    BEST_RECENT = "best_recent"  # Melhor performance recente
    WEIGHTED_ENSEMBLE = "weighted_ensemble"  # Ensemble ponderado
    ADAPTIVE_THRESHOLD = "adaptive_threshold"  # Threshold adaptativo
    MOMENTUM_BASED = "momentum_based"  # Baseado em momentum
    RISK_ADJUSTED = "risk_adjusted"  # Ajustado por risco

@dataclass
class ModelPerformance:
    """Estrutura para armazenar performance de modelo"""
    model_name: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    confidence: float = 1.0
    sample_size: int = 0

@dataclass
class ModelWeight:
    """Estrutura para pesos de modelo no ensemble"""
    model_name: str
    weight: float
    confidence: float
    last_updated: datetime

class DynamicModelSelector:
    """
    Sistema de seleção dinâmica de modelos para trading
    Baseado em "Advances in Financial Machine Learning" e técnicas de ensemble adaptativo
    
    Funcionalidades:
    1. Avaliação contínua de múltiplos modelos
    2. Seleção adaptativa baseada em performance
    3. Ensemble dinâmico com pesos adaptativos
    4. Detecção de degradação de modelo
    5. Re-balanceamento automático
    """
    
    def __init__(self,
                 models: Optional[Union[Dict[str, BaseEstimator], List[str]]] = None,
                 model_names: Optional[List[str]] = None,
                 selection_strategy: SelectionStrategy = SelectionStrategy.WEIGHTED_ENSEMBLE,
                 performance_window: int = 100,
                 min_samples: int = 30,
                 rebalance_frequency: int = 50,
                 decay_factor: float = 0.95,
                 confidence_threshold: float = 0.6,
                 evaluation_window: Optional[int] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            models: Dicionário de modelos {nome: modelo} (opcional)
            model_names: Lista de nomes de modelos para criar automaticamente (opcional)
            selection_strategy: Estratégia de seleção
            performance_window: Janela para cálculo de performance
            min_samples: Mínimo de amostras para avaliação
            rebalance_frequency: Frequência de re-balanceamento
            decay_factor: Fator de decaimento para performance histórica
            confidence_threshold: Threshold mínimo de confiança
            logger: Logger para debug
        """
        # Criar modelos se apenas nomes foram fornecidos
        if models is None and model_names is not None:
            self.models = self._create_models_from_names(model_names)
        elif models is not None:
            # Se models é uma lista de strings, converter para dicionário de modelos
            if isinstance(models, list):
                self.models = self._create_models_from_names(models)
            else:
                self.models = models
        else:
            # Modelos padrão
            self.models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42)
            }
        self.selection_strategy = selection_strategy
        self.performance_window = performance_window
        self.min_samples = min_samples
        self.rebalance_frequency = rebalance_frequency
        self.decay_factor = decay_factor
        self.confidence_threshold = confidence_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Histórico de performance
        self.performance_history = defaultdict(list)
        self.prediction_history = defaultdict(deque)
        self.target_history = deque(maxlen=self.performance_window * 2)
        
        # Estado atual
        self.current_weights = {}
        self.active_models = set(self.models.keys())
        self.selected_model = None
        self.ensemble_weights = {}
        
        # Contadores
        self.prediction_count = 0
        self.last_rebalance = 0
        
        # Métricas de trading
        self.trading_metrics = defaultdict(lambda: defaultdict(list))
        
        # Inicializar pesos uniformes
        self._initialize_weights()
        
    def _create_models_from_names(self, model_names: List[str]) -> Dict[str, BaseEstimator]:
        """Cria modelos a partir de uma lista de nomes"""
        available_models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        }
        
        models = {}
        for name in model_names:
            if name in available_models:
                models[name] = available_models[name]
            else:
                self.logger.warning(f"Modelo '{name}' não reconhecido. Modelos disponíveis: {list(available_models.keys())}")
                
        return models
        
    def _initialize_weights(self):
        """Inicializa pesos uniformes para todos os modelos"""
        n_models = len(self.models)
        uniform_weight = 1.0 / n_models if n_models > 0 else 0.0
        
        for model_name in self.models.keys():
            self.current_weights[model_name] = ModelWeight(
                model_name=model_name,
                weight=uniform_weight,
                confidence=0.5,
                last_updated=datetime.now()
            )
            
        self.ensemble_weights = {name: uniform_weight for name in self.models.keys()}
        
    def fit_models(self, X: pd.DataFrame, y: pd.Series):
        """
        Treina todos os modelos
        
        Args:
            X: Features
            y: Targets
        """
        self.logger.info(f"Treinando {len(self.models)} modelos")
        
        for name, model in self.models.items():
            try:
                self.logger.info(f"Treinando modelo: {name}")
                model.fit(X, y)
                
                # Avaliar performance inicial
                y_pred = model.predict(X)
                y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                performance = self._calculate_performance(y, y_pred, y_pred_proba, name)
                self.performance_history[name].append(performance)
                
                self.logger.info(f"Modelo {name} treinado - Accuracy: {performance.accuracy:.3f}")
                
            except Exception as e:
                self.logger.error(f"Erro ao treinar modelo {name}: {e}")
                self.active_models.discard(name)
                
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Faz predição usando estratégia selecionada
        
        Args:
            X: Features para predição
            
        Returns:
            Tupla (predições, confiança por modelo)
        """
        if not self.active_models:
            raise ValueError("Nenhum modelo ativo disponível")
            
        # Obter predições de todos os modelos ativos
        model_predictions = {}
        model_probabilities = {}
        model_confidences = {}
        
        for name in self.active_models:
            try:
                model = self.models[name]
                pred = model.predict(X)
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[:, 1]
                    confidence = np.max(model.predict_proba(X), axis=1).mean()
                else:
                    proba = pred.astype(float)
                    confidence = 0.7  # Confiança padrão para modelos sem probabilidade
                    
                model_predictions[name] = pred
                model_probabilities[name] = proba
                model_confidences[name] = confidence
                
            except Exception as e:
                self.logger.error(f"Erro na predição do modelo {name}: {e}")
                continue
                
        if not model_predictions:
            raise ValueError("Nenhuma predição válida obtida")
            
        # Aplicar estratégia de seleção
        final_predictions, selection_info = self._apply_selection_strategy(
            model_predictions, model_probabilities, model_confidences, X
        )
        
        self.prediction_count += len(X)
        
        # Re-balancear se necessário
        if self.prediction_count - self.last_rebalance >= self.rebalance_frequency:
            self._rebalance_models()
            
        return final_predictions, selection_info
    
    def update_performance(self, y_true: pd.Series, 
                          trading_returns: Optional[pd.Series] = None,
                          model_predictions: Optional[Dict[str, np.ndarray]] = None):
        """
        Atualiza performance dos modelos com resultados reais
        
        Args:
            y_true: Targets reais
            trading_returns: Retornos de trading (opcional)
            model_predictions: Predições dos modelos (opcional)
        """
        self.target_history.extend(y_true.values)
        
        # Se não temos predições, usar as mais recentes armazenadas
        if model_predictions is None:
            model_predictions = {}
            for name in self.active_models:
                if len(self.prediction_history[name]) >= len(y_true):
                    recent_preds = list(self.prediction_history[name])[-len(y_true):]
                    model_predictions[name] = np.array(recent_preds)
                    
        # Atualizar performance de cada modelo
        for name in self.active_models:
            if name not in model_predictions:
                continue
                
            y_pred = model_predictions[name]
            if len(y_pred) != len(y_true):
                continue
                
            # Calcular métricas de ML
            try:
                y_pred_proba = y_pred if y_pred.dtype == float else y_pred.astype(float)
                performance = self._calculate_performance(y_true, y_pred, y_pred_proba, name)
                
                # Adicionar métricas de trading se disponível
                if trading_returns is not None:
                    performance.sharpe_ratio = self._calculate_sharpe_ratio(trading_returns)
                    performance.max_drawdown = self._calculate_max_drawdown(trading_returns)
                    performance.win_rate = (trading_returns > 0).mean()
                    
                    positive_returns = trading_returns[trading_returns > 0].sum()
                    negative_returns = abs(trading_returns[trading_returns < 0].sum())
                    performance.profit_factor = positive_returns / negative_returns if negative_returns > 0 else np.inf
                    
                self.performance_history[name].append(performance)
                
                # Manter apenas janela de performance
                if len(self.performance_history[name]) > self.performance_window:
                    self.performance_history[name] = self.performance_history[name][-self.performance_window:]
                    
                self.logger.debug(f"Performance atualizada para {name}: Acc={performance.accuracy:.3f}")
                
            except Exception as e:
                self.logger.error(f"Erro ao atualizar performance do modelo {name}: {e}")
                
    def _apply_selection_strategy(self, 
                                model_predictions: Dict[str, np.ndarray],
                                model_probabilities: Dict[str, np.ndarray],
                                model_confidences: Dict[str, float],
                                X: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Aplica estratégia de seleção de modelo
        """
        if self.selection_strategy == SelectionStrategy.BEST_RECENT:
            return self._select_best_recent(model_predictions, model_confidences)
            
        elif self.selection_strategy == SelectionStrategy.WEIGHTED_ENSEMBLE:
            return self._weighted_ensemble(model_predictions, model_probabilities)
            
        elif self.selection_strategy == SelectionStrategy.ADAPTIVE_THRESHOLD:
            return self._adaptive_threshold(model_predictions, model_confidences)
            
        elif self.selection_strategy == SelectionStrategy.MOMENTUM_BASED:
            return self._momentum_based_selection(model_predictions, model_confidences)
            
        elif self.selection_strategy == SelectionStrategy.RISK_ADJUSTED:
            return self._risk_adjusted_selection(model_predictions, model_confidences)
            
        else:
            # Fallback para ensemble simples
            return self._weighted_ensemble(model_predictions, model_probabilities)
    
    def _select_best_recent(self, 
                          model_predictions: Dict[str, np.ndarray],
                          model_confidences: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Seleciona modelo com melhor performance recente
        """
        best_model = None
        best_score = -np.inf
        
        for name in model_predictions.keys():
            if name not in self.performance_history or not self.performance_history[name]:
                continue
                
            # Calcular score ponderado recente
            recent_performances = self.performance_history[name][-10:]  # Últimas 10 avaliações
            weights = np.array([self.decay_factor ** i for i in range(len(recent_performances))][::-1])
            
            scores = np.array([p.f1_score for p in recent_performances])
            weighted_score = np.average(scores, weights=weights)
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_model = name
                
        if best_model is None:
            # Fallback para primeiro modelo disponível
            best_model = list(model_predictions.keys())[0]
            
        self.selected_model = best_model
        
        selection_info = {name: 1.0 if name == best_model else 0.0 
                         for name in model_predictions.keys()}
        
        return model_predictions[best_model], selection_info
    
    def _weighted_ensemble(self, 
                         model_predictions: Dict[str, np.ndarray],
                         model_probabilities: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Ensemble ponderado baseado em performance
        """
        # Atualizar pesos baseado em performance
        self._update_ensemble_weights()
        
        # Combinar predições
        ensemble_proba = np.zeros(len(list(model_probabilities.values())[0]))
        total_weight = 0.0
        
        for name, proba in model_probabilities.items():
            weight = self.ensemble_weights.get(name, 0.0)
            ensemble_proba += weight * proba
            total_weight += weight
            
        if total_weight > 0:
            ensemble_proba /= total_weight
            
        # Converter para predições binárias
        ensemble_predictions = (ensemble_proba > 0.5).astype(int)
        
        return ensemble_predictions, self.ensemble_weights.copy()
    
    def _adaptive_threshold(self, 
                          model_predictions: Dict[str, np.ndarray],
                          model_confidences: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Seleção baseada em threshold adaptativo de confiança
        """
        # Calcular threshold dinâmico
        all_confidences = list(model_confidences.values())
        if all_confidences:
            dynamic_threshold = np.mean(all_confidences) + 0.1 * np.std(all_confidences)
        else:
            dynamic_threshold = self.confidence_threshold
            
        # Selecionar modelos acima do threshold
        qualified_models = {name: conf for name, conf in model_confidences.items() 
                          if conf >= dynamic_threshold}
        
        if not qualified_models:
            # Se nenhum modelo qualifica, usar o melhor
            best_model = max(model_confidences.keys(), key=lambda x: model_confidences[x])
            qualified_models = {best_model: model_confidences[best_model]}
            
        # Ensemble dos modelos qualificados
        total_confidence = sum(qualified_models.values())
        weights = {name: conf / total_confidence for name, conf in qualified_models.items()}
        
        # Combinar predições
        ensemble_pred = np.zeros(len(list(model_predictions.values())[0]))
        for name, weight in weights.items():
            ensemble_pred += weight * model_predictions[name]
            
        ensemble_pred = (ensemble_pred > 0.5).astype(int)
        
        # Preencher pesos para todos os modelos
        all_weights = {name: weights.get(name, 0.0) for name in model_predictions.keys()}
        
        return ensemble_pred, all_weights
    
    def _momentum_based_selection(self, 
                                model_predictions: Dict[str, np.ndarray],
                                model_confidences: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Seleção baseada em momentum de performance
        """
        momentum_scores = {}
        
        for name in model_predictions.keys():
            if name not in self.performance_history or len(self.performance_history[name]) < 5:
                momentum_scores[name] = 0.0
                continue
                
            # Calcular momentum (tendência de melhoria)
            recent_scores = [p.f1_score for p in self.performance_history[name][-5:]]
            if len(recent_scores) >= 2:
                momentum = np.mean(np.diff(recent_scores))
                current_score = recent_scores[-1]
                momentum_scores[name] = current_score + 0.5 * momentum  # Score + momentum
            else:
                momentum_scores[name] = recent_scores[-1] if recent_scores else 0.0
                
        if not momentum_scores:
            return self._select_best_recent(model_predictions, model_confidences)
            
        # Selecionar modelo com melhor momentum
        best_model = max(momentum_scores.keys(), key=lambda x: momentum_scores[x])
        
        selection_info = {name: 1.0 if name == best_model else 0.0 
                         for name in model_predictions.keys()}
        
        return model_predictions[best_model], selection_info
    
    def _risk_adjusted_selection(self, 
                               model_predictions: Dict[str, np.ndarray],
                               model_confidences: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Seleção ajustada por risco (Sharpe ratio, drawdown)
        """
        risk_adjusted_scores = {}
        
        for name in model_predictions.keys():
            if name not in self.performance_history or not self.performance_history[name]:
                risk_adjusted_scores[name] = 0.0
                continue
                
            recent_perf = self.performance_history[name][-10:]  # Últimas 10 performances
            
            # Calcular score ajustado por risco
            avg_return = np.mean([p.f1_score for p in recent_perf])
            
            # Usar Sharpe ratio se disponível, senão usar volatilidade de performance
            if any(p.sharpe_ratio is not None for p in recent_perf):
                sharpe_ratios = [p.sharpe_ratio for p in recent_perf if p.sharpe_ratio is not None]
                avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0.0
                risk_adjusted_scores[name] = avg_sharpe
            else:
                # Usar consistência de performance como proxy de risco
                performance_std = np.std([p.f1_score for p in recent_perf])
                risk_adjusted_scores[name] = avg_return / (performance_std + 1e-6)
                
        if not risk_adjusted_scores:
            return self._select_best_recent(model_predictions, model_confidences)
            
        # Selecionar modelo com melhor score ajustado por risco
        best_model = max(risk_adjusted_scores.keys(), key=lambda x: risk_adjusted_scores[x])
        
        selection_info = {name: 1.0 if name == best_model else 0.0 
                         for name in model_predictions.keys()}
        
        return model_predictions[best_model], selection_info
    
    def _update_ensemble_weights(self):
        """
        Atualiza pesos do ensemble baseado em performance recente
        """
        total_score = 0.0
        model_scores = {}
        
        for name in self.active_models:
            if name not in self.performance_history or not self.performance_history[name]:
                model_scores[name] = 0.1  # Score mínimo
                continue
                
            # Calcular score ponderado por tempo
            recent_performances = self.performance_history[name][-20:]  # Últimas 20 avaliações
            weights = np.array([self.decay_factor ** i for i in range(len(recent_performances))][::-1])
            
            # Combinar múltiplas métricas
            f1_scores = np.array([p.f1_score for p in recent_performances])
            accuracy_scores = np.array([p.accuracy for p in recent_performances])
            
            # Score combinado
            combined_scores = 0.6 * f1_scores + 0.4 * accuracy_scores
            weighted_score = np.average(combined_scores, weights=weights)
            
            model_scores[name] = max(weighted_score, 0.01)  # Evitar pesos zero
            total_score += model_scores[name]
            
        # Normalizar pesos
        if total_score > 0:
            self.ensemble_weights = {name: score / total_score 
                                   for name, score in model_scores.items()}
        else:
            # Fallback para pesos uniformes
            n_models = len(self.active_models)
            self.ensemble_weights = {name: 1.0 / n_models for name in self.active_models}
            
    def _rebalance_models(self):
        """
        Re-balanceia modelos removendo os de baixa performance
        """
        self.logger.info("Iniciando re-balanceamento de modelos")
        
        # Calcular performance média recente
        model_avg_performance = {}
        
        for name in list(self.active_models):
            if name not in self.performance_history or len(self.performance_history[name]) < self.min_samples:
                continue
                
            recent_performances = self.performance_history[name][-self.min_samples:]
            avg_f1 = np.mean([p.f1_score for p in recent_performances])
            model_avg_performance[name] = avg_f1
            
        if not model_avg_performance:
            self.logger.warning("Nenhum modelo com performance suficiente para re-balanceamento")
            return
            
        # Remover modelos com performance muito baixa
        performance_threshold = np.percentile(list(model_avg_performance.values()), 25)  # Bottom 25%
        
        models_to_remove = []
        for name, performance in model_avg_performance.items():
            if performance < performance_threshold and performance < 0.5:  # Abaixo do threshold E abaixo de 50%
                models_to_remove.append(name)
                
        # Manter pelo menos 2 modelos ativos
        if len(self.active_models) - len(models_to_remove) >= 2:
            for name in models_to_remove:
                self.active_models.discard(name)
                self.logger.info(f"Modelo {name} removido por baixa performance: {model_avg_performance[name]:.3f}")
                
        # Atualizar pesos
        self._update_ensemble_weights()
        
        self.last_rebalance = self.prediction_count
        
        self.logger.info(f"Re-balanceamento concluído. Modelos ativos: {len(self.active_models)}")
    
    def _calculate_performance(self, y_true: pd.Series, y_pred: np.ndarray, 
                             y_pred_proba: np.ndarray, model_name: str) -> ModelPerformance:
        """
        Calcula métricas de performance
        """
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            # AUC apenas se temos probabilidades válidas
            try:
                auc = roc_auc_score(y_true, y_pred_proba)
            except:
                auc = 0.5
                
            # Confiança baseada na consistência das predições
            confidence = 1.0 - np.std(y_pred_proba) if len(np.unique(y_pred_proba)) > 1 else 0.5
            
            return ModelPerformance(
                model_name=model_name,
                timestamp=datetime.now(),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_score=auc,
                confidence=confidence,
                sample_size=len(y_true)
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular performance para {model_name}: {e}")
            return ModelPerformance(
                model_name=model_name,
                timestamp=datetime.now(),
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                auc_score=0.5,
                confidence=0.0,
                sample_size=len(y_true)
            )
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calcula Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
            
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)  # Anualizado
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calcula maximum drawdown
        """
        if len(returns) == 0:
            return 0.0
            
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min())
    
    def get_model_rankings(self) -> pd.DataFrame:
        """
        Retorna ranking atual dos modelos
        
        Returns:
            DataFrame com ranking e métricas
        """
        rankings = []
        
        for name in self.active_models:
            if name not in self.performance_history or not self.performance_history[name]:
                continue
                
            recent_perf = self.performance_history[name][-10:]  # Últimas 10 performances
            
            avg_accuracy = np.mean([p.accuracy for p in recent_perf])
            avg_f1 = np.mean([p.f1_score for p in recent_perf])
            avg_auc = np.mean([p.auc_score for p in recent_perf])
            
            # Consistência (inverso do desvio padrão)
            consistency = 1.0 / (np.std([p.f1_score for p in recent_perf]) + 1e-6)
            
            current_weight = self.ensemble_weights.get(name, 0.0)
            
            rankings.append({
                'model': name,
                'avg_accuracy': avg_accuracy,
                'avg_f1_score': avg_f1,
                'avg_auc': avg_auc,
                'consistency': consistency,
                'current_weight': current_weight,
                'samples': len(recent_perf),
                'last_updated': recent_perf[-1].timestamp if recent_perf else None
            })
            
        df = pd.DataFrame(rankings)
        if not df.empty:
            df = df.sort_values('avg_f1_score', ascending=False)
            
        return df
    
    def export_model_report(self, filepath: str):
        """
        Exporta relatório detalhado dos modelos
        
        Args:
            filepath: Caminho para salvar o relatório
        """
        report_data = []
        
        for name in self.models.keys():
            if name not in self.performance_history:
                continue
                
            for perf in self.performance_history[name]:
                report_data.append({
                    'model_name': perf.model_name,
                    'timestamp': perf.timestamp,
                    'accuracy': perf.accuracy,
                    'precision': perf.precision,
                    'recall': perf.recall,
                    'f1_score': perf.f1_score,
                    'auc_score': perf.auc_score,
                    'sharpe_ratio': perf.sharpe_ratio,
                    'max_drawdown': perf.max_drawdown,
                    'win_rate': perf.win_rate,
                    'profit_factor': perf.profit_factor,
                    'confidence': perf.confidence,
                    'sample_size': perf.sample_size
                })
                
        df = pd.DataFrame(report_data)
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Relatório de modelos exportado para {filepath}")
    
    def select_best_model(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, strategy: str = 'best_recent') -> str:
        """
        Seleciona o melhor modelo baseado na performance atual
        
        Args:
            X: Features para avaliação (opcional)
            y: Targets para avaliação (opcional)
            strategy: Estratégia de seleção ('best_recent', 'momentum', 'risk_adjusted')
        
        Returns:
            Nome do melhor modelo
        """
        # Se dados foram fornecidos, treinar e avaliar modelos
        if X is not None and y is not None:
            self.fit_models(X, y)
        if not self.performance_history:
            return list(self.models.keys())[0] if self.models else None
            
        if strategy == 'momentum':
            return self._select_best_momentum()
        elif strategy == 'risk_adjusted':
            return self._select_best_risk_adjusted()
        else:  # 'best_recent' or default
            return self._select_best_recent_model()
    
    def _select_best_recent_model(self) -> str:
        """
        Seleciona modelo com melhor performance recente
        """
        model_scores = {}
        
        for name in self.models.keys():
            if name not in self.performance_history or not self.performance_history[name]:
                model_scores[name] = 0.0
                continue
                
            # Usar performance recente
            recent_perf = self.performance_history[name][-5:]  # Últimas 5 performances
            avg_f1 = np.mean([p.f1_score for p in recent_perf])
            model_scores[name] = avg_f1
            
        # Retornar modelo com melhor score
        if model_scores:
            return max(model_scores.keys(), key=lambda x: model_scores[x])
        else:
            return list(self.models.keys())[0] if self.models else None
    
    def _select_best_momentum(self) -> str:
        """
        Seleciona modelo baseado em momentum de performance
        """
        momentum_scores = {}
        
        for name in self.models.keys():
            if name not in self.performance_history or len(self.performance_history[name]) < 3:
                momentum_scores[name] = 0.0
                continue
                
            # Calcular momentum (tendência de melhoria)
            recent_scores = [p.f1_score for p in self.performance_history[name][-5:]]
            if len(recent_scores) >= 2:
                momentum = np.mean(np.diff(recent_scores))
                current_score = recent_scores[-1]
                momentum_scores[name] = current_score + 0.5 * momentum
            else:
                momentum_scores[name] = recent_scores[-1] if recent_scores else 0.0
                
        if momentum_scores:
            return max(momentum_scores.keys(), key=lambda x: momentum_scores[x])
        else:
            return list(self.models.keys())[0] if self.models else None
    
    def _select_best_risk_adjusted(self) -> str:
        """
        Seleciona modelo com melhor score ajustado por risco
        """
        risk_adjusted_scores = {}
        
        for name in self.models.keys():
            if name not in self.performance_history or not self.performance_history[name]:
                risk_adjusted_scores[name] = 0.0
                continue
                
            recent_perf = self.performance_history[name][-10:]  # Últimas 10 performances
            
            # Calcular score ajustado por risco
            avg_return = np.mean([p.f1_score for p in recent_perf])
            performance_std = np.std([p.f1_score for p in recent_perf])
            risk_adjusted_scores[name] = avg_return / (performance_std + 1e-6)
                
        if risk_adjusted_scores:
            return max(risk_adjusted_scores.keys(), key=lambda x: risk_adjusted_scores[x])
        else:
            return list(self.models.keys())[0] if self.models else None

# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Criar modelos
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42),
        'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42),
        'lightgbm': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    }
    
    # Criar seletor dinâmico
    selector = DynamicModelSelector(
        models=models,
        selection_strategy=SelectionStrategy.WEIGHTED_ENSEMBLE,
        logger=logger
    )
    
    # Simular dados
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.choice([0, 1], n_samples))
    
    # Treinar modelos
    selector.fit_models(X[:800], y[:800])
    
    # Fazer predições
    X_test = X[800:]
    y_test = y[800:]
    
    predictions, selection_info = selector.predict(X_test)
    
    print(f"\nPredições feitas usando estratégia: {selector.selection_strategy.value}")
    print(f"Informações de seleção: {selection_info}")
    
    # Simular retornos de trading
    trading_returns = pd.Series(np.random.normal(0.001, 0.02, len(y_test)))
    
    # Atualizar performance
    model_preds = {name: predictions for name in models.keys()}  # Simplificado
    selector.update_performance(y_test, trading_returns, model_preds)
    
    # Ver rankings
    rankings = selector.get_model_rankings()
    print(f"\nRanking de modelos:")
    print(rankings)
    
    # Selecionar melhor modelo com diferentes estratégias
    best_model_recent = selector.select_best_model('best_recent')
    best_model_momentum = selector.select_best_model('momentum')
    best_model_risk = selector.select_best_model('risk_adjusted')
    
    print(f"\nMelhor modelo (recent): {best_model_recent}")
    print(f"Melhor modelo (momentum): {best_model_momentum}")
    print(f"Melhor modelo (risk-adjusted): {best_model_risk}")
    
    # Exportar relatório
    selector.export_model_report('model_performance_report.csv')
    print(f"\nRelatório exportado para model_performance_report.csv")