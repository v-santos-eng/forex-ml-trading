import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import joblib
from pathlib import Path
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

@dataclass
class WalkForwardConfig:
    """Configuração para Walk-Forward Analysis"""
    
    # Parâmetros de janela
    train_window_size: int = 252  # Tamanho da janela de treino (dias)
    test_window_size: int = 21    # Tamanho da janela de teste (dias)
    step_size: int = 21           # Passo para avançar as janelas
    
    # Parâmetros de validação
    min_train_samples: int = 100  # Mínimo de amostras para treino
    purge_window: int = 1         # Janela de purge entre treino e teste
    embargo_window: int = 0       # Janela de embargo após teste
    
    # Parâmetros de otimização
    enable_optimization: bool = True
    optimization_frequency: int = 4  # Otimizar a cada N períodos
    max_optimization_trials: int = 50
    
    # Parâmetros de performance
    enable_parallel: bool = True
    n_jobs: int = -1
    
    # Parâmetros de análise
    confidence_level: float = 0.95
    enable_regime_analysis: bool = True
    enable_stability_analysis: bool = True
    
    # Parâmetros de salvamento
    save_results: bool = True
    results_path: str = "walk_forward_results"

class WalkForwardAnalyzer:
    """
    Sistema avançado de Walk-Forward Analysis para validação temporal
    Baseado em "Advances in Financial Machine Learning" e melhores práticas
    
    Funcionalidades:
    1. Walk-Forward Analysis com purging e embargo
    2. Otimização de hiperparâmetros temporal
    3. Análise de estabilidade de performance
    4. Detecção de regime e adaptação
    5. Análise estatística robusta
    6. Processamento paralelo
    7. Visualizações avançadas
    """
    
    def __init__(self, 
                 config: Optional[WalkForwardConfig] = None,
                 train_period: Optional[int] = None,
                 test_period: Optional[int] = None,
                 train_size: Optional[int] = None,
                 test_size: Optional[int] = None,
                 step_size: Optional[int] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            config: Configuração do walk-forward
            train_period: Período de treino (alias para train_size)
            test_period: Período de teste (alias para test_size)
            train_size: Tamanho da janela de treino
            test_size: Tamanho da janela de teste
            step_size: Tamanho do passo
            logger: Logger para debug
        """
        # Criar configuração baseada nos parâmetros fornecidos
        if config is None:
            config = WalkForwardConfig()
            
        # Sobrescrever configurações com parâmetros diretos se fornecidos
        if train_size is not None or train_period is not None:
            config.train_window_size = train_size or train_period
        if test_size is not None or test_period is not None:
            config.test_window_size = test_size or test_period
        if step_size is not None:
            config.step_size = step_size
            
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Resultados
        self.results = {
            'performance_metrics': [],
            'predictions': [],
            'feature_importance': [],
            'optimization_history': [],
            'regime_analysis': [],
            'stability_metrics': []
        }
        
        # Cache para otimização
        self._optimization_cache = {}
        
        # Criar diretório de resultados
        if self.config.save_results:
            Path(self.config.results_path).mkdir(exist_ok=True)
            
    def run_walk_forward_analysis(self, 
                                 X: pd.DataFrame, 
                                 y: pd.Series,
                                 model_class: type,
                                 param_grid: Optional[Dict] = None,
                                 custom_scorer: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Executa Walk-Forward Analysis completo
        
        Args:
            X: Features
            y: Target
            model_class: Classe do modelo a ser testado
            param_grid: Grid de parâmetros para otimização
            custom_scorer: Função de score customizada
            
        Returns:
            Dicionário com resultados completos
        """
        self.logger.info("Iniciando Walk-Forward Analysis")
        
        # Validar dados
        self._validate_data(X, y)
        
        # Gerar janelas temporais
        windows = self._generate_time_windows(X.index)
        
        self.logger.info(f"Geradas {len(windows)} janelas para análise")
        
        # Executar análise
        if self.config.enable_parallel and len(windows) > 1:
            results = self._run_parallel_analysis(X, y, windows, model_class, param_grid, custom_scorer)
        else:
            results = self._run_sequential_analysis(X, y, windows, model_class, param_grid, custom_scorer)
            
        # Processar resultados
        self._process_results(results)
        
        # Análises adicionais
        if self.config.enable_regime_analysis:
            self._analyze_regimes(X, y)
            
        if self.config.enable_stability_analysis:
            self._analyze_stability()
            
        # Salvar resultados
        if self.config.save_results:
            self._save_results()
            
        # Gerar relatório
        summary = self._generate_summary()
        
        self.logger.info("Walk-Forward Analysis concluído")
        
        return {
            'summary': summary,
            'detailed_results': self.results,
            'config': self.config,
            'performance_metrics': self.results.get('performance_metrics', [])
        }
    
    def run_analysis(self, X: pd.DataFrame, y: pd.Series,
                    model_class: Optional[type] = None,
                    param_grid: Optional[Dict] = None,
                    custom_scorer: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Alias para run_walk_forward_analysis para compatibilidade
        
        Args:
            X: Features
            y: Target
            model_class: Classe do modelo (opcional, usa RandomForest se não fornecido)
            param_grid: Grid de parâmetros para otimização
            custom_scorer: Função de score customizada
            
        Returns:
            Dicionário com resultados completos
        """
        # Usar RandomForest como modelo padrão se não fornecido
        if model_class is None:
            from sklearn.ensemble import RandomForestClassifier
            model_class = RandomForestClassifier
            
        return self.run_walk_forward_analysis(X, y, model_class, param_grid, custom_scorer)
    
    def _validate_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Valida dados de entrada
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X deve ter DatetimeIndex")
            
        if not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError("y deve ter DatetimeIndex")
            
        if len(X) != len(y):
            raise ValueError("X e y devem ter mesmo tamanho")
            
        if len(X) < self.config.min_train_samples + self.config.test_window_size:
            raise ValueError("Dados insuficientes para análise")
            
        # Verificar alinhamento de índices
        if not X.index.equals(y.index):
            self.logger.warning("Índices de X e y não estão alinhados")
            
    def _generate_time_windows(self, index: pd.DatetimeIndex) -> List[Dict[str, Any]]:
        """
        Gera janelas temporais para walk-forward
        
        Returns:
            Lista de dicionários com informações das janelas
        """
        windows = []
        
        start_idx = 0
        
        while True:
            # Definir janela de treino
            train_end_idx = start_idx + self.config.train_window_size
            
            if train_end_idx >= len(index):
                break
                
            # Aplicar purge
            test_start_idx = train_end_idx + self.config.purge_window
            test_end_idx = test_start_idx + self.config.test_window_size
            
            if test_end_idx > len(index):
                break
                
            # Criar janela
            window = {
                'window_id': len(windows),
                'train_start': index[start_idx],
                'train_end': index[train_end_idx - 1],
                'test_start': index[test_start_idx],
                'test_end': index[test_end_idx - 1],
                'train_indices': list(range(start_idx, train_end_idx)),
                'test_indices': list(range(test_start_idx, test_end_idx)),
                'optimize': (len(windows) % self.config.optimization_frequency == 0) if self.config.enable_optimization else False
            }
            
            windows.append(window)
            
            # Avançar janela
            start_idx += self.config.step_size
            
        return windows
    
    def _run_parallel_analysis(self, X: pd.DataFrame, y: pd.Series, 
                             windows: List[Dict], model_class: type,
                             param_grid: Optional[Dict], 
                             custom_scorer: Optional[Callable]) -> List[Dict]:
        """
        Executa análise em paralelo
        """
        self.logger.info("Executando análise em paralelo")
        
        with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = []
            
            for window in windows:
                future = executor.submit(
                    self._analyze_single_window,
                    X, y, window, model_class, param_grid, custom_scorer
                )
                futures.append(future)
                
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Erro na análise de janela: {e}")
                    
        # Ordenar por window_id
        results.sort(key=lambda x: x['window_id'])
        
        return results
    
    def _run_sequential_analysis(self, X: pd.DataFrame, y: pd.Series,
                               windows: List[Dict], model_class: type,
                               param_grid: Optional[Dict],
                               custom_scorer: Optional[Callable]) -> List[Dict]:
        """
        Executa análise sequencial
        """
        self.logger.info("Executando análise sequencial")
        
        results = []
        
        for i, window in enumerate(windows):
            self.logger.info(f"Processando janela {i+1}/{len(windows)}")
            
            try:
                result = self._analyze_single_window(
                    X, y, window, model_class, param_grid, custom_scorer
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Erro na janela {i}: {e}")
                continue
                
        return results
    
    def _analyze_single_window(self, X: pd.DataFrame, y: pd.Series,
                             window: Dict, model_class: type,
                             param_grid: Optional[Dict],
                             custom_scorer: Optional[Callable]) -> Dict[str, Any]:
        """
        Analisa uma única janela temporal
        
        Args:
            X: Features
            y: Target
            window: Informações da janela
            model_class: Classe do modelo
            param_grid: Grid de parâmetros
            custom_scorer: Scorer customizado
            
        Returns:
            Resultados da análise da janela
        """
        # Extrair dados da janela
        X_train = X.iloc[window['train_indices']]
        y_train = y.iloc[window['train_indices']]
        X_test = X.iloc[window['test_indices']]
        y_test = y.iloc[window['test_indices']]
        
        # Limpar dados
        X_train_clean, y_train_clean = self._clean_training_data(X_train, y_train)
        X_test_clean, y_test_clean = self._clean_test_data(X_test, y_test)
        
        # Otimizar hiperparâmetros se necessário
        if window['optimize'] and param_grid:
            best_params = self._optimize_hyperparameters(
                X_train_clean, y_train_clean, model_class, param_grid, custom_scorer
            )
        else:
            best_params = param_grid.get('default', {}) if param_grid else {}
            
        # Treinar modelo
        model = model_class(**best_params)
        model.fit(X_train_clean, y_train_clean)
        
        # Fazer predições
        y_pred = model.predict(X_test_clean)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_clean)
            if y_pred_proba.shape[1] > 1:
                y_pred_proba = y_pred_proba[:, 1]  # Probabilidade da classe positiva
            else:
                y_pred_proba = y_pred_proba.flatten()
        else:
            y_pred_proba = None
            
        # Calcular métricas
        metrics = self._calculate_metrics(y_test_clean, y_pred, y_pred_proba, custom_scorer)
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, X_train_clean.columns)
        
        # Análise de regime
        regime_info = self._analyze_window_regime(X_train_clean, y_train_clean)
        
        return {
            'window_id': window['window_id'],
            'train_period': (window['train_start'], window['train_end']),
            'test_period': (window['test_start'], window['test_end']),
            'train_samples': len(X_train_clean),
            'test_samples': len(X_test_clean),
            'best_params': best_params,
            'metrics': metrics,
            'predictions': {
                'y_true': y_test_clean.values,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'test_dates': X_test_clean.index.tolist()
            },
            'feature_importance': feature_importance,
            'regime_info': regime_info,
            'model_state': self._serialize_model_state(model)
        }
    
    def _clean_training_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Limpa dados de treino
        """
        # Remover NaNs
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Verificar se há dados suficientes
        if len(X_clean) < self.config.min_train_samples:
            raise ValueError(f"Dados de treino insuficientes: {len(X_clean)} < {self.config.min_train_samples}")
            
        return X_clean, y_clean
    
    def _clean_test_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Limpa dados de teste
        """
        # Remover NaNs
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        
        return X_clean, y_clean
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                                model_class: type, param_grid: Dict,
                                custom_scorer: Optional[Callable]) -> Dict[str, Any]:
        """
        Otimiza hiperparâmetros usando validação temporal
        """
        # Cache key
        cache_key = (str(X.index[0]), str(X.index[-1]), str(param_grid))
        
        if cache_key in self._optimization_cache:
            return self._optimization_cache[cache_key]
            
        best_score = -np.inf
        best_params = {}
        
        # Grid search com validação temporal
        param_combinations = list(ParameterGrid(param_grid))
        
        # Limitar número de tentativas
        if len(param_combinations) > self.config.max_optimization_trials:
            np.random.shuffle(param_combinations)
            param_combinations = param_combinations[:self.config.max_optimization_trials]
            
        for params in param_combinations:
            try:
                # Validação cruzada temporal
                scores = self._temporal_cross_validation(X, y, model_class, params, custom_scorer)
                avg_score = np.mean(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params
                    
            except Exception as e:
                self.logger.warning(f"Erro na otimização com params {params}: {e}")
                continue
                
        # Cache resultado
        self._optimization_cache[cache_key] = best_params
        
        return best_params
    
    def _temporal_cross_validation(self, X: pd.DataFrame, y: pd.Series,
                                 model_class: type, params: Dict,
                                 custom_scorer: Optional[Callable],
                                 n_splits: int = 3) -> List[float]:
        """
        Validação cruzada temporal para otimização
        """
        scores = []
        
        # Dividir dados em splits temporais
        split_size = len(X) // (n_splits + 1)
        
        for i in range(n_splits):
            # Definir índices
            train_end = (i + 1) * split_size
            val_start = train_end
            val_end = min(val_start + split_size // 2, len(X))
            
            if val_end <= val_start:
                continue
                
            # Extrair dados
            X_train_cv = X.iloc[:train_end]
            y_train_cv = y.iloc[:train_end]
            X_val_cv = X.iloc[val_start:val_end]
            y_val_cv = y.iloc[val_start:val_end]
            
            # Limpar dados
            X_train_cv, y_train_cv = self._clean_training_data(X_train_cv, y_train_cv)
            X_val_cv, y_val_cv = self._clean_test_data(X_val_cv, y_val_cv)
            
            if len(X_train_cv) < 20 or len(X_val_cv) < 5:
                continue
                
            # Treinar e avaliar
            model = model_class(**params)
            model.fit(X_train_cv, y_train_cv)
            
            y_pred = model.predict(X_val_cv)
            
            if custom_scorer:
                score = custom_scorer(y_val_cv, y_pred)
            else:
                score = accuracy_score(y_val_cv, y_pred)
                
            scores.append(score)
            
        return scores
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                         y_pred_proba: Optional[np.ndarray],
                         custom_scorer: Optional[Callable]) -> Dict[str, float]:
        """
        Calcula métricas de performance
        """
        metrics = {}
        
        try:
            # Métricas básicas
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Métricas probabilísticas
            if y_pred_proba is not None:
                from sklearn.metrics import roc_auc_score, log_loss
                
                try:
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                except:
                    metrics['auc'] = 0.5
                    
                try:
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                except:
                    metrics['log_loss'] = np.inf
                    
            # Métrica customizada
            if custom_scorer:
                try:
                    metrics['custom_score'] = custom_scorer(y_true, y_pred)
                except Exception as e:
                    self.logger.warning(f"Erro no scorer customizado: {e}")
                    
            # Métricas de trading
            metrics.update(self._calculate_trading_metrics(y_true, y_pred))
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de métricas: {e}")
            
        return metrics
    
    def _calculate_trading_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas específicas de trading
        """
        metrics = {}
        
        try:
            # Hit rate
            metrics['hit_rate'] = np.mean(y_true == y_pred)
            
            # Precision por classe
            if len(np.unique(y_true)) == 2:
                # Classificação binária
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                
                metrics['true_positives'] = tp
                metrics['false_positives'] = fp
                metrics['true_negatives'] = tn
                metrics['false_negatives'] = fn
                
                # Precision e Recall por direção
                if tp + fp > 0:
                    metrics['long_precision'] = tp / (tp + fp)
                else:
                    metrics['long_precision'] = 0
                    
                if tn + fn > 0:
                    metrics['short_precision'] = tn / (tn + fn)
                else:
                    metrics['short_precision'] = 0
                    
            # Sharpe ratio das predições (aproximação)
            pred_returns = np.where(y_pred == 1, 1, -1) * np.where(y_true == 1, 1, -1)
            if np.std(pred_returns) > 0:
                metrics['prediction_sharpe'] = np.mean(pred_returns) / np.std(pred_returns)
            else:
                metrics['prediction_sharpe'] = 0
                
        except Exception as e:
            self.logger.warning(f"Erro no cálculo de métricas de trading: {e}")
            
        return metrics
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """
        Extrai importância das features do modelo
        """
        importance = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                for name, imp in zip(feature_names, model.feature_importances_):
                    importance[name] = float(imp)
                    
            elif hasattr(model, 'coef_'):
                # Linear models
                coef = model.coef_
                if coef.ndim > 1:
                    coef = coef[0]  # Para classificação binária
                    
                for name, coef_val in zip(feature_names, coef):
                    importance[name] = float(abs(coef_val))
                    
            else:
                # Modelo sem importância explícita
                importance = {name: 0.0 for name in feature_names}
                
        except Exception as e:
            self.logger.warning(f"Erro ao extrair feature importance: {e}")
            importance = {name: 0.0 for name in feature_names}
            
        return importance
    
    def _analyze_window_regime(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Analisa regime de mercado da janela
        """
        regime_info = {}
        
        try:
            # Volatilidade
            if 'returns' in X.columns:
                returns = X['returns'].dropna()
                regime_info['volatility'] = returns.std()
                regime_info['mean_return'] = returns.mean()
                regime_info['skewness'] = returns.skew()
                regime_info['kurtosis'] = returns.kurtosis()
                
            # Distribuição do target
            regime_info['target_distribution'] = y.value_counts(normalize=True).to_dict()
            
            # Trend strength
            if len(X) > 20:
                # Usar primeira coluna como proxy de preço
                price_proxy = X.iloc[:, 0]
                ma_short = price_proxy.rolling(5).mean()
                ma_long = price_proxy.rolling(20).mean()
                trend_strength = (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
                regime_info['trend_strength'] = trend_strength
                
        except Exception as e:
            self.logger.warning(f"Erro na análise de regime: {e}")
            
        return regime_info
    
    def _serialize_model_state(self, model: Any) -> Optional[bytes]:
        """
        Serializa estado do modelo para análise posterior
        """
        try:
            return pickle.dumps(model)
        except Exception as e:
            self.logger.warning(f"Erro ao serializar modelo: {e}")
            return None
    
    def _process_results(self, results: List[Dict]) -> None:
        """
        Processa e organiza resultados
        """
        # Organizar métricas
        for result in results:
            self.results['performance_metrics'].append({
                'window_id': result['window_id'],
                'train_period': result['train_period'],
                'test_period': result['test_period'],
                **result['metrics']
            })
            
            # Predições
            self.results['predictions'].append(result['predictions'])
            
            # Feature importance
            self.results['feature_importance'].append({
                'window_id': result['window_id'],
                'importance': result['feature_importance']
            })
            
            # Regime info
            self.results['regime_analysis'].append({
                'window_id': result['window_id'],
                **result['regime_info']
            })
    
    def _analyze_regimes(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Análise de regimes de mercado
        """
        self.logger.info("Analisando regimes de mercado")
        
        try:
            # Agrupar resultados por regime
            regime_performance = defaultdict(list)
            
            for result in self.results['regime_analysis']:
                # Classificar regime baseado em volatilidade
                vol = result.get('volatility', 0)
                if vol < 0.01:
                    regime = 'low_vol'
                elif vol < 0.02:
                    regime = 'medium_vol'
                else:
                    regime = 'high_vol'
                    
                # Encontrar performance correspondente
                window_id = result['window_id']
                perf = next((p for p in self.results['performance_metrics'] 
                           if p['window_id'] == window_id), None)
                
                if perf:
                    regime_performance[regime].append(perf['accuracy'])
                    
            # Calcular estatísticas por regime
            regime_stats = {}
            for regime, accuracies in regime_performance.items():
                if accuracies:
                    regime_stats[regime] = {
                        'mean_accuracy': np.mean(accuracies),
                        'std_accuracy': np.std(accuracies),
                        'count': len(accuracies)
                    }
                    
            self.results['regime_performance'] = regime_stats
            
        except Exception as e:
            self.logger.error(f"Erro na análise de regimes: {e}")
    
    def _analyze_stability(self) -> None:
        """
        Análise de estabilidade da performance
        """
        self.logger.info("Analisando estabilidade da performance")
        
        try:
            metrics_df = pd.DataFrame(self.results['performance_metrics'])
            
            if len(metrics_df) == 0:
                return
                
            stability_metrics = {}
            
            # Estabilidade temporal
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                if metric in metrics_df.columns:
                    values = metrics_df[metric].dropna()
                    
                    if len(values) > 1:
                        stability_metrics[f'{metric}_mean'] = values.mean()
                        stability_metrics[f'{metric}_std'] = values.std()
                        stability_metrics[f'{metric}_cv'] = values.std() / (values.mean() + 1e-8)
                        
                        # Teste de estacionariedade
                        try:
                            from statsmodels.tsa.stattools import adfuller
                            adf_stat, adf_pvalue = adfuller(values)[:2]
                            stability_metrics[f'{metric}_adf_pvalue'] = adf_pvalue
                        except:
                            pass
                            
            # Tendência temporal
            if len(metrics_df) > 2:
                time_index = range(len(metrics_df))
                
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    if metric in metrics_df.columns:
                        values = metrics_df[metric].dropna()
                        
                        if len(values) > 2:
                            # Regressão linear para detectar tendência
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                time_index[:len(values)], values
                            )
                            
                            stability_metrics[f'{metric}_trend_slope'] = slope
                            stability_metrics[f'{metric}_trend_pvalue'] = p_value
                            stability_metrics[f'{metric}_trend_r2'] = r_value**2
                            
            self.results['stability_metrics'] = stability_metrics
            
        except Exception as e:
            self.logger.error(f"Erro na análise de estabilidade: {e}")
    
    def _save_results(self) -> None:
        """
        Salva resultados em arquivos
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Salvar métricas
            metrics_df = pd.DataFrame(self.results['performance_metrics'])
            metrics_df.to_csv(f"{self.config.results_path}/metrics_{timestamp}.csv", index=False)
            
            # Salvar resultados completos
            with open(f"{self.config.results_path}/results_{timestamp}.pkl", 'wb') as f:
                pickle.dump(self.results, f)
                
            # Salvar configuração
            with open(f"{self.config.results_path}/config_{timestamp}.pkl", 'wb') as f:
                pickle.dump(self.config, f)
                
            self.logger.info(f"Resultados salvos em {self.config.results_path}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar resultados: {e}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Gera resumo dos resultados
        """
        summary = {}
        
        try:
            metrics_df = pd.DataFrame(self.results['performance_metrics'])
            
            if len(metrics_df) > 0:
                # Estatísticas gerais
                summary['total_windows'] = len(metrics_df)
                summary['date_range'] = {
                    'start': min(metrics_df['train_period'].apply(lambda x: x[0])),
                    'end': max(metrics_df['test_period'].apply(lambda x: x[1]))
                }
                
                # Performance média
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                    if metric in metrics_df.columns:
                        values = metrics_df[metric].dropna()
                        if len(values) > 0:
                            summary[f'avg_{metric}'] = values.mean()
                            summary[f'std_{metric}'] = values.std()
                            summary[f'min_{metric}'] = values.min()
                            summary[f'max_{metric}'] = values.max()
                            
                # Estabilidade
                if 'stability_metrics' in self.results:
                    summary['stability'] = self.results['stability_metrics']
                    
                # Performance por regime
                if 'regime_performance' in self.results:
                    summary['regime_performance'] = self.results['regime_performance']
                    
                # Intervalos de confiança
                conf_level = self.config.confidence_level
                alpha = 1 - conf_level
                
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    if metric in metrics_df.columns:
                        values = metrics_df[metric].dropna()
                        if len(values) > 1:
                            mean_val = values.mean()
                            std_val = values.std()
                            n = len(values)
                            
                            # Intervalo de confiança
                            margin_error = stats.t.ppf(1 - alpha/2, n-1) * std_val / np.sqrt(n)
                            
                            summary[f'{metric}_ci_lower'] = mean_val - margin_error
                            summary[f'{metric}_ci_upper'] = mean_val + margin_error
                            
        except Exception as e:
            self.logger.error(f"Erro ao gerar resumo: {e}")
            
        return summary
    
    def plot_results(self, save_plots: bool = True) -> None:
        """
        Gera visualizações dos resultados
        """
        try:
            metrics_df = pd.DataFrame(self.results['performance_metrics'])
            
            if len(metrics_df) == 0:
                self.logger.warning("Nenhum resultado para plotar")
                return
                
            # Configurar estilo
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Walk-Forward Analysis Results', fontsize=16)
            
            # 1. Performance ao longo do tempo
            ax1 = axes[0, 0]
            test_dates = [pd.to_datetime(period[0]) for period in metrics_df['test_period']]
            
            ax1.plot(test_dates, metrics_df['accuracy'], 'b-', label='Accuracy', alpha=0.7)
            if 'f1' in metrics_df.columns:
                ax1.plot(test_dates, metrics_df['f1'], 'r-', label='F1 Score', alpha=0.7)
                
            ax1.set_title('Performance Over Time')
            ax1.set_xlabel('Test Period')
            ax1.set_ylabel('Score')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Distribuição de performance
            ax2 = axes[0, 1]
            ax2.hist(metrics_df['accuracy'], bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax2.axvline(metrics_df['accuracy'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {metrics_df["accuracy"].mean():.3f}')
            ax2.set_title('Accuracy Distribution')
            ax2.set_xlabel('Accuracy')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Correlação entre métricas
            ax3 = axes[1, 0]
            metric_cols = ['accuracy', 'precision', 'recall', 'f1']
            available_metrics = [col for col in metric_cols if col in metrics_df.columns]
            
            if len(available_metrics) > 1:
                corr_matrix = metrics_df[available_metrics].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
                ax3.set_title('Metrics Correlation')
            else:
                ax3.text(0.5, 0.5, 'Insufficient metrics for correlation', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Metrics Correlation')
                
            # 4. Rolling performance
            ax4 = axes[1, 1]
            window_size = min(5, len(metrics_df) // 2)
            if window_size > 0:
                rolling_acc = metrics_df['accuracy'].rolling(window_size).mean()
                ax4.plot(range(len(rolling_acc)), rolling_acc, 'g-', linewidth=2)
                ax4.fill_between(range(len(rolling_acc)), 
                               rolling_acc - metrics_df['accuracy'].rolling(window_size).std(),
                               rolling_acc + metrics_df['accuracy'].rolling(window_size).std(),
                               alpha=0.3, color='green')
                ax4.set_title(f'Rolling Accuracy (window={window_size})')
                ax4.set_xlabel('Window')
                ax4.set_ylabel('Rolling Accuracy')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Insufficient data for rolling analysis', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Rolling Performance')
                
            plt.tight_layout()
            
            if save_plots:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plt.savefig(f"{self.config.results_path}/walk_forward_analysis_{timestamp}.png", 
                           dpi=300, bbox_inches='tight')
                
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar plots: {e}")

# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Criar dados de exemplo
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Features simuladas
    n_features = 20
    X = pd.DataFrame(
        np.random.randn(1000, n_features),
        index=dates,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Target simulado com alguma estrutura temporal
    y = pd.Series(index=dates, dtype=int)
    for i in range(len(y)):
        # Adicionar alguma dependência temporal
        if i == 0:
            y.iloc[i] = np.random.choice([0, 1])
        else:
            # 70% de chance de manter direção anterior
            if np.random.random() < 0.7:
                y.iloc[i] = y.iloc[i-1]
            else:
                y.iloc[i] = 1 - y.iloc[i-1]
                
    # Configuração
    config = WalkForwardConfig(
        train_window_size=200,
        test_window_size=20,
        step_size=20,
        enable_optimization=True,
        enable_parallel=False  # Para exemplo
    )
    
    # Modelo de exemplo
    from sklearn.ensemble import RandomForestClassifier
    
    # Grid de parâmetros
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5]
    }
    
    # Executar análise
    analyzer = WalkForwardAnalyzer(config=config, logger=logger)
    
    results = analyzer.run_walk_forward_analysis(
        X=X,
        y=y,
        model_class=RandomForestClassifier,
        param_grid=param_grid
    )
    
    # Mostrar resultados
    print("\n=== RESUMO DOS RESULTADOS ===")
    for key, value in results['summary'].items():
        print(f"{key}: {value}")
        
    # Gerar plots
    analyzer.plot_results(save_plots=False)