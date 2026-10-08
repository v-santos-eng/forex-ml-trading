import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Tuple, Callable, Any
import warnings
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, LogisticRegression,
    RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    validation_curve, learning_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass
import joblib
from pathlib import Path


@dataclass
class RegularizationResults:
    """Classe para armazenar resultados de regularização."""
    best_params: Dict[str, Any]
    best_score: float
    cv_scores: np.ndarray
    validation_scores: Dict[str, np.ndarray]
    model: BaseEstimator
    feature_importance: Optional[np.ndarray] = None
    selected_features: Optional[List[str]] = None


class AdvancedRegularization:
    """
    Classe para implementar técnicas avançadas de regularização
    especificamente adaptadas para dados financeiros.
    """
    
    def __init__(self, 
                 problem_type: str = 'regression',
                 scaling_method: str = 'standard',
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Inicializa o sistema de regularização.
        
        Args:
            problem_type: 'regression' ou 'classification'
            scaling_method: 'standard', 'robust', ou 'none'
            cv_folds: Número de folds para cross-validation
            random_state: Seed para reprodutibilidade
        """
        self.problem_type = problem_type
        self.scaling_method = scaling_method
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Configurar scaler
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = None
        
        # Armazenar resultados
        self.results_history: Dict[str, RegularizationResults] = {}
        
    def ridge_regularization(self, 
                           X: pd.DataFrame, 
                           y: pd.Series,
                           alpha_range: Optional[np.ndarray] = None,
                           use_cv: bool = True) -> RegularizationResults:
        """
        Implementa regularização Ridge (L2).
        
        Args:
            X: Features
            y: Target
            alpha_range: Range de valores alpha para testar
            use_cv: Se usar cross-validation para seleção de alpha
        
        Returns:
            RegularizationResults com os melhores parâmetros e modelo
        """
        if alpha_range is None:
            alpha_range = np.logspace(-4, 2, 50)
        
        # Preparar dados
        X_processed = self._preprocess_data(X, fit_scaler=True)
        
        if use_cv:
            if self.problem_type == 'regression':
                model = RidgeCV(
                    alphas=alpha_range,
                    cv=self.cv_folds,
                    scoring='neg_mean_squared_error'
                )
            else:
                model = LogisticRegressionCV(
                    Cs=1/alpha_range,  # C = 1/alpha
                    cv=self.cv_folds,
                    penalty='l2',
                    solver='liblinear',
                    random_state=self.random_state
                )
        else:
            if self.problem_type == 'regression':
                model = Ridge(alpha=1.0)
            else:
                model = LogisticRegression(
                    C=1.0,
                    penalty='l2',
                    solver='liblinear',
                    random_state=self.random_state
                )
        
        # Treinar modelo
        model.fit(X_processed, y)
        
        # Avaliar performance
        cv_scores = cross_val_score(
            model, X_processed, y, 
            cv=self.cv_folds,
            scoring=self._get_scoring_metric()
        )
        
        # Calcular curva de validação
        if use_cv:
            best_alpha = model.alpha_ if hasattr(model, 'alpha_') else 1/model.C_[0]
            validation_scores = self._compute_validation_curve(
                X_processed, y, 'alpha', alpha_range, Ridge if self.problem_type == 'regression' else LogisticRegression
            )
        else:
            best_alpha = 1.0
            validation_scores = {}
        
        # Extrair importância das features
        feature_importance = np.abs(model.coef_).flatten() if hasattr(model, 'coef_') else None
        
        results = RegularizationResults(
            best_params={'alpha': best_alpha},
            best_score=cv_scores.mean(),
            cv_scores=cv_scores,
            validation_scores=validation_scores,
            model=model,
            feature_importance=feature_importance,
            selected_features=list(X.columns)
        )
        
        self.results_history['ridge'] = results
        return results
    
    def lasso_regularization(self, 
                           X: pd.DataFrame, 
                           y: pd.Series,
                           alpha_range: Optional[np.ndarray] = None,
                           use_cv: bool = True,
                           feature_selection: bool = True) -> RegularizationResults:
        """
        Implementa regularização Lasso (L1) com seleção automática de features.
        
        Args:
            X: Features
            y: Target
            alpha_range: Range de valores alpha para testar
            use_cv: Se usar cross-validation para seleção de alpha
            feature_selection: Se aplicar seleção de features baseada em coeficientes
        
        Returns:
            RegularizationResults com features selecionadas
        """
        if alpha_range is None:
            alpha_range = np.logspace(-4, 1, 50)
        
        # Preparar dados
        X_processed = self._preprocess_data(X, fit_scaler=True)
        
        if use_cv:
            if self.problem_type == 'regression':
                model = LassoCV(
                    alphas=alpha_range,
                    cv=self.cv_folds,
                    random_state=self.random_state,
                    max_iter=2000
                )
            else:
                model = LogisticRegressionCV(
                    Cs=1/alpha_range,
                    cv=self.cv_folds,
                    penalty='l1',
                    solver='liblinear',
                    random_state=self.random_state
                )
        else:
            if self.problem_type == 'regression':
                model = Lasso(alpha=0.1, random_state=self.random_state)
            else:
                model = LogisticRegression(
                    C=1.0,
                    penalty='l1',
                    solver='liblinear',
                    random_state=self.random_state
                )
        
        # Treinar modelo
        model.fit(X_processed, y)
        
        # Seleção de features
        selected_features = list(X.columns)
        if feature_selection and hasattr(model, 'coef_'):
            coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
            selected_mask = np.abs(coef) > 1e-6
            selected_features = [X.columns[i] for i in range(len(X.columns)) if selected_mask[i]]
            
            print(f"Lasso selecionou {len(selected_features)} de {len(X.columns)} features")
        
        # Avaliar performance
        cv_scores = cross_val_score(
            model, X_processed, y, 
            cv=self.cv_folds,
            scoring=self._get_scoring_metric()
        )
        
        # Calcular curva de validação
        if use_cv:
            best_alpha = model.alpha_ if hasattr(model, 'alpha_') else 1/model.C_[0]
            validation_scores = self._compute_validation_curve(
                X_processed, y, 'alpha', alpha_range, Lasso if self.problem_type == 'regression' else LogisticRegression
            )
        else:
            best_alpha = 0.1
            validation_scores = {}
        
        # Extrair importância das features
        feature_importance = np.abs(model.coef_).flatten() if hasattr(model, 'coef_') else None
        
        results = RegularizationResults(
            best_params={'alpha': best_alpha},
            best_score=cv_scores.mean(),
            cv_scores=cv_scores,
            validation_scores=validation_scores,
            model=model,
            feature_importance=feature_importance,
            selected_features=selected_features
        )
        
        self.results_history['lasso'] = results
        return results
    
    def elastic_net_regularization(self, 
                                 X: pd.DataFrame, 
                                 y: pd.Series,
                                 alpha_range: Optional[np.ndarray] = None,
                                 l1_ratio_range: Optional[np.ndarray] = None,
                                 use_cv: bool = True) -> RegularizationResults:
        """
        Implementa regularização Elastic Net (combinação L1 + L2).
        
        Args:
            X: Features
            y: Target
            alpha_range: Range de valores alpha
            l1_ratio_range: Range de valores l1_ratio (0=Ridge, 1=Lasso)
            use_cv: Se usar cross-validation
        
        Returns:
            RegularizationResults com parâmetros otimizados
        """
        if alpha_range is None:
            alpha_range = np.logspace(-4, 1, 20)
        if l1_ratio_range is None:
            l1_ratio_range = np.linspace(0.1, 0.9, 9)
        
        # Preparar dados
        X_processed = self._preprocess_data(X, fit_scaler=True)
        
        if use_cv:
            if self.problem_type == 'regression':
                model = ElasticNetCV(
                    alphas=alpha_range,
                    l1_ratio=l1_ratio_range,
                    cv=self.cv_folds,
                    random_state=self.random_state,
                    max_iter=2000
                )
            else:
                # Para classificação, usar GridSearchCV
                param_grid = {
                    'C': 1/alpha_range,
                    'l1_ratio': l1_ratio_range
                }
                base_model = LogisticRegression(
                    penalty='elasticnet',
                    solver='saga',
                    random_state=self.random_state,
                    max_iter=2000
                )
                model = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=self.cv_folds,
                    scoring=self._get_scoring_metric(),
                    n_jobs=-1
                )
        else:
            if self.problem_type == 'regression':
                model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)
            else:
                model = LogisticRegression(
                    penalty='elasticnet',
                    C=1.0,
                    l1_ratio=0.5,
                    solver='saga',
                    random_state=self.random_state
                )
        
        # Treinar modelo
        model.fit(X_processed, y)
        
        # Extrair melhores parâmetros
        if use_cv:
            if self.problem_type == 'regression':
                best_params = {
                    'alpha': model.alpha_,
                    'l1_ratio': model.l1_ratio_
                }
            else:
                best_params = model.best_params_
        else:
            best_params = {'alpha': 0.1, 'l1_ratio': 0.5}
        
        # Avaliar performance
        final_model = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
        cv_scores = cross_val_score(
            final_model, X_processed, y, 
            cv=self.cv_folds,
            scoring=self._get_scoring_metric()
        )
        
        # Seleção de features (similar ao Lasso)
        selected_features = list(X.columns)
        if hasattr(final_model, 'coef_'):
            coef = final_model.coef_.flatten() if final_model.coef_.ndim > 1 else final_model.coef_
            selected_mask = np.abs(coef) > 1e-6
            selected_features = [X.columns[i] for i in range(len(X.columns)) if selected_mask[i]]
        
        # Extrair importância das features
        feature_importance = np.abs(final_model.coef_).flatten() if hasattr(final_model, 'coef_') else None
        
        results = RegularizationResults(
            best_params=best_params,
            best_score=cv_scores.mean(),
            cv_scores=cv_scores,
            validation_scores={},
            model=final_model,
            feature_importance=feature_importance,
            selected_features=selected_features
        )
        
        self.results_history['elastic_net'] = results
        return results
    
    def adaptive_regularization(self, 
                              X: pd.DataFrame, 
                              y: pd.Series,
                              methods: List[str] = None) -> Dict[str, RegularizationResults]:
        """
        Aplica múltiplos métodos de regularização e compara resultados.
        
        Args:
            X: Features
            y: Target
            methods: Lista de métodos ['ridge', 'lasso', 'elastic_net']
        
        Returns:
            Dicionário com resultados de cada método
        """
        if methods is None:
            methods = ['ridge', 'lasso', 'elastic_net']
        
        results = {}
        
        for method in methods:
            print(f"Aplicando regularização {method.upper()}...")
            
            if method == 'ridge':
                results[method] = self.ridge_regularization(X, y)
            elif method == 'lasso':
                results[method] = self.lasso_regularization(X, y)
            elif method == 'elastic_net':
                results[method] = self.elastic_net_regularization(X, y)
            else:
                warnings.warn(f"Método {method} não reconhecido")
        
        return results
    
    def regularization_path_analysis(self, 
                                   X: pd.DataFrame, 
                                   y: pd.Series,
                                   method: str = 'lasso',
                                   n_alphas: int = 100) -> Dict[str, np.ndarray]:
        """
        Analisa o caminho de regularização (como coeficientes mudam com alpha).
        
        Args:
            X: Features
            y: Target
            method: 'lasso', 'ridge', ou 'elastic_net'
            n_alphas: Número de valores alpha para testar
        
        Returns:
            Dicionário com alphas e coeficientes
        """
        from sklearn.linear_model import lasso_path, ridge_path, enet_path
        
        X_processed = self._preprocess_data(X, fit_scaler=True)
        
        if method == 'lasso':
            alphas, coefs, _ = lasso_path(X_processed, y, n_alphas=n_alphas)
        elif method == 'ridge':
            alphas = np.logspace(-4, 2, n_alphas)
            coefs = ridge_path(X_processed, y, alphas=alphas)[1]
        elif method == 'elastic_net':
            alphas, coefs, _ = enet_path(X_processed, y, n_alphas=n_alphas, l1_ratio=0.5)
        else:
            raise ValueError(f"Método {method} não suportado")
        
        return {
            'alphas': alphas,
            'coefs': coefs,
            'feature_names': list(X.columns)
        }
    
    def plot_regularization_path(self, 
                               X: pd.DataFrame, 
                               y: pd.Series,
                               method: str = 'lasso',
                               top_features: int = 10) -> plt.Figure:
        """
        Plota o caminho de regularização.
        
        Args:
            X: Features
            y: Target
            method: Método de regularização
            top_features: Número de features principais para mostrar
        
        Returns:
            Figura matplotlib
        """
        path_data = self.regularization_path_analysis(X, y, method)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Selecionar top features por importância média
        mean_importance = np.mean(np.abs(path_data['coefs']), axis=1)
        top_indices = np.argsort(mean_importance)[-top_features:]
        
        # Plotar coeficientes
        for i in top_indices:
            ax.plot(path_data['alphas'], path_data['coefs'][i], 
                   label=path_data['feature_names'][i], linewidth=2)
        
        ax.set_xscale('log')
        ax.set_xlabel('Alpha (Regularization Strength)')
        ax.set_ylabel('Coefficient Value')
        ax.set_title(f'{method.title()} Regularization Path')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_validation_curves(self, results: RegularizationResults) -> plt.Figure:
        """
        Plota curvas de validação.
        
        Args:
            results: Resultados da regularização
        
        Returns:
            Figura matplotlib
        """
        if not results.validation_scores:
            raise ValueError("Nenhuma curva de validação disponível")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        param_name = list(results.validation_scores.keys())[0]
        param_range = results.validation_scores[param_name]['param_range']
        train_scores = results.validation_scores[param_name]['train_scores']
        val_scores = results.validation_scores[param_name]['val_scores']
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.semilogx(param_range, train_mean, 'o-', color='blue', label='Training Score')
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        ax.semilogx(param_range, val_mean, 'o-', color='red', label='Validation Score')
        ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        # Marcar melhor parâmetro
        best_alpha = results.best_params.get('alpha', results.best_params.get('C', 1.0))
        ax.axvline(x=best_alpha, color='green', linestyle='--', label=f'Best {param_name}: {best_alpha:.4f}')
        
        ax.set_xlabel(param_name.title())
        ax.set_ylabel('Score')
        ax.set_title('Validation Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def compare_methods(self, results_dict: Dict[str, RegularizationResults]) -> pd.DataFrame:
        """
        Compara diferentes métodos de regularização.
        
        Args:
            results_dict: Dicionário com resultados de diferentes métodos
        
        Returns:
            DataFrame com comparação
        """
        comparison_data = []
        
        for method, results in results_dict.items():
            n_features = len(results.selected_features) if results.selected_features else 0
            
            comparison_data.append({
                'Method': method.title(),
                'CV Score': f"{results.best_score:.4f}",
                'CV Std': f"{results.cv_scores.std():.4f}",
                'Selected Features': n_features,
                'Best Alpha': results.best_params.get('alpha', 'N/A'),
                'L1 Ratio': results.best_params.get('l1_ratio', 'N/A')
            })
        
        return pd.DataFrame(comparison_data)
    
    def feature_importance_analysis(self, 
                                  results: RegularizationResults,
                                  top_n: int = 20) -> pd.DataFrame:
        """
        Analisa importância das features.
        
        Args:
            results: Resultados da regularização
            top_n: Número de features principais
        
        Returns:
            DataFrame com importância das features
        """
        if results.feature_importance is None:
            raise ValueError("Importância das features não disponível")
        
        # Garantir que feature_names e feature_importance tenham o mesmo tamanho
        importance_values = results.feature_importance
        if results.selected_features and len(results.selected_features) == len(importance_values):
            feature_names = results.selected_features
        else:
            feature_names = [f"feature_{i}" for i in range(len(importance_values))]
        
        # Verificar se os tamanhos coincidem
        if len(feature_names) != len(importance_values):
            min_len = min(len(feature_names), len(importance_values))
            feature_names = feature_names[:min_len]
            importance_values = importance_values[:min_len]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        return importance_df.head(top_n)
    
    def save_results(self, filepath: str) -> None:
        """
        Salva resultados em arquivo.
        
        Args:
            filepath: Caminho do arquivo
        """
        joblib.dump(self.results_history, filepath)
        print(f"Resultados salvos em {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """
        Carrega resultados de arquivo.
        
        Args:
            filepath: Caminho do arquivo
        """
        self.results_history = joblib.load(filepath)
        print(f"Resultados carregados de {filepath}")
    
    def _preprocess_data(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """Preprocessa dados aplicando scaling se necessário."""
        if self.scaler is not None:
            if fit_scaler:
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)
        return X.values
    
    def _get_scoring_metric(self) -> str:
        """Retorna métrica de scoring apropriada."""
        if self.problem_type == 'regression':
            return 'neg_mean_squared_error'
        else:
            return 'roc_auc'
    
    def _compute_validation_curve(self, 
                                X: np.ndarray, 
                                y: pd.Series,
                                param_name: str,
                                param_range: np.ndarray,
                                estimator_class) -> Dict[str, Any]:
        """Computa curva de validação."""
        if self.problem_type == 'regression':
            estimator = estimator_class(random_state=self.random_state)
        else:
            estimator = estimator_class(random_state=self.random_state, solver='liblinear')
        
        train_scores, val_scores = validation_curve(
            estimator, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=self.cv_folds,
            scoring=self._get_scoring_metric(),
            n_jobs=-1
        )
        
        return {
            param_name: {
                'param_range': param_range,
                'train_scores': train_scores,
                'val_scores': val_scores
            }
        }


class FinancialRegularization(AdvancedRegularization):
    """
    Extensão especializada para dados financeiros com técnicas específicas
    para lidar com características únicas dos mercados financeiros.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.financial_metrics = {}
    
    def time_series_regularization(self, 
                                 X: pd.DataFrame, 
                                 y: pd.Series,
                                 time_index: pd.DatetimeIndex,
                                 lookback_window: int = 252) -> RegularizationResults:
        """
        Regularização adaptada para séries temporais financeiras.
        
        Args:
            X: Features
            y: Target
            time_index: Índice temporal
            lookback_window: Janela de lookback para estabilidade
        
        Returns:
            Resultados com considerações temporais
        """
        # Implementar rolling window regularization
        results_list = []
        
        for i in range(lookback_window, len(X)):
            X_window = X.iloc[i-lookback_window:i]
            y_window = y.iloc[i-lookback_window:i]
            
            # Aplicar regularização na janela
            window_results = self.elastic_net_regularization(X_window, y_window, use_cv=False)
            results_list.append(window_results)
        
        # Agregar resultados
        avg_score = np.mean([r.best_score for r in results_list])
        
        # Usar último modelo como representativo
        final_results = results_list[-1]
        final_results.best_score = avg_score
        
        return final_results
    
    def regime_aware_regularization(self, 
                                  X: pd.DataFrame, 
                                  y: pd.Series,
                                  regime_indicator: pd.Series) -> Dict[str, RegularizationResults]:
        """
        Regularização consciente de regimes de mercado.
        
        Args:
            X: Features
            y: Target
            regime_indicator: Indicador de regime (ex: volatilidade alta/baixa)
        
        Returns:
            Resultados por regime
        """
        regime_results = {}
        
        for regime in regime_indicator.unique():
            regime_mask = regime_indicator == regime
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            
            if len(X_regime) > 50:  # Mínimo de observações
                regime_results[f'regime_{regime}'] = self.adaptive_regularization(
                    X_regime, y_regime
                )
        
        return regime_results
    
    def calculate_financial_metrics(self, 
                                  y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calcula métricas específicas para finanças.
        
        Args:
            y_true: Valores reais
            y_pred: Predições
            returns: Retornos para cálculo de Sharpe
        
        Returns:
            Dicionário com métricas financeiras
        """
        metrics = {}
        
        if self.problem_type == 'regression':
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Directional accuracy
            direction_true = np.sign(y_true)
            direction_pred = np.sign(y_pred)
            metrics['directional_accuracy'] = np.mean(direction_true == direction_pred)
            
        else:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # Sharpe ratio se retornos fornecidos
        if returns is not None:
            metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        return metrics


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados de exemplo
    np.random.seed(42)
    n_samples, n_features = 1000, 50
    
    # Simular dados financeiros com algumas features relevantes
    X = np.random.randn(n_samples, n_features)
    
    # Criar target com apenas algumas features sendo relevantes
    true_coef = np.zeros(n_features)
    true_coef[:10] = np.random.randn(10) * 2  # Apenas primeiras 10 features são relevantes
    
    y = X @ true_coef + np.random.randn(n_samples) * 0.5
    
    # Converter para DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    # Inicializar regularização
    reg = AdvancedRegularization(problem_type='regression')
    
    # Testar diferentes métodos
    print("Testando regularização Ridge...")
    ridge_results = reg.ridge_regularization(X_df, y_series)
    
    print("\nTestando regularização Lasso...")
    lasso_results = reg.lasso_regularization(X_df, y_series)
    
    print("\nTestando regularização Elastic Net...")
    elastic_results = reg.elastic_net_regularization(X_df, y_series)
    
    # Comparar métodos
    all_results = {
        'ridge': ridge_results,
        'lasso': lasso_results,
        'elastic_net': elastic_results
    }
    
    comparison = reg.compare_methods(all_results)
    print("\nComparação de Métodos:")
    print(comparison)
    
    # Análise de importância das features (Lasso)
    print("\nTop 10 Features (Lasso):")
    feature_importance = reg.feature_importance_analysis(lasso_results, top_n=10)
    print(feature_importance)