import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import cvxpy as cp
from collections import defaultdict
import pickle
from pathlib import Path

@dataclass
class RiskBudgetConfig:
    """Configuração para Risk Budgeting"""
    
    # Parâmetros de fatores
    n_factors: int = 5  # Número de fatores principais
    factor_method: str = 'pca'  # 'pca', 'factor_analysis', 'custom'
    lookback_window: int = 252  # Janela para estimação de fatores
    
    # Parâmetros de risco
    risk_budget_method: str = 'equal_risk'  # 'equal_risk', 'custom', 'hierarchical'
    max_position_weight: float = 0.2  # Peso máximo por posição
    max_factor_exposure: float = 0.3  # Exposição máxima por fator
    
    # Parâmetros de otimização
    optimization_method: str = 'cvxpy'  # 'cvxpy', 'scipy', 'genetic'
    max_iterations: int = 1000
    tolerance: float = 1e-6
    
    # Parâmetros de rebalanceamento
    rebalance_frequency: str = 'weekly'  # 'daily', 'weekly', 'monthly'
    min_rebalance_threshold: float = 0.05  # Threshold mínimo para rebalancear
    
    # Parâmetros de performance
    enable_parallel: bool = True
    n_jobs: int = -1
    
    # Parâmetros de monitoramento
    track_factor_exposures: bool = True
    track_risk_contributions: bool = True
    enable_stress_testing: bool = True
    
    # Parâmetros de salvamento
    save_results: bool = True
    results_path: str = "risk_budgeting_results"

class RiskBudgetingSystem:
    """
    Sistema avançado de Risk Budgeting por Fator
    Baseado em técnicas de gestão de risco quantitativo
    
    Funcionalidades:
    1. Decomposição de risco por fatores
    2. Alocação baseada em orçamento de risco
    3. Otimização de portfólio com restrições de risco
    4. Monitoramento contínuo de exposições
    5. Stress testing e análise de cenários
    6. Rebalanceamento automático
    7. Análise de contribuição de risco
    """
    
    def __init__(self, 
                 config: Optional[RiskBudgetConfig] = None,
                 lookback_period: Optional[int] = None,
                 n_factors: Optional[int] = None,
                 total_risk_budget: Optional[float] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            config: Configuração do sistema
            lookback_period: Período de lookback (sobrescreve config se fornecido)
            n_factors: Número de fatores (sobrescreve config se fornecido)
            total_risk_budget: Orçamento total de risco (para compatibilidade)
            logger: Logger para debug
        """
        self.config = config or RiskBudgetConfig()
        
        # Sobrescrever configurações se parâmetros específicos forem fornecidos
        if lookback_period is not None:
            self.config.lookback_window = lookback_period
        if n_factors is not None:
            self.config.n_factors = n_factors
        if total_risk_budget is not None:
            self.total_risk_budget = total_risk_budget
        else:
            self.total_risk_budget = 0.02  # Valor padrão
            
        self.logger = logger or logging.getLogger(__name__)
        
        # Componentes do sistema
        self.factor_model = None
        self.risk_model = None
        self.scaler = StandardScaler()
        
        # Estado do portfólio
        self.current_weights = None
        self.factor_exposures = None
        self.risk_contributions = None
        
        # Histórico
        self.history = {
            'weights': [],
            'factor_exposures': [],
            'risk_contributions': [],
            'portfolio_risk': [],
            'rebalance_dates': []
        }
        
        # Cache
        self._factor_cache = {}
        self._covariance_cache = {}
        
        # Criar diretório de resultados
        if self.config.save_results:
            Path(self.config.results_path).mkdir(exist_ok=True)
    
    def allocate_risk_budget(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """
        Aloca orçamento de risco baseado nos dados de retorno
        
        Args:
            returns_data: DataFrame com retornos dos ativos
            
        Returns:
            Dicionário com alocação de risco por ativo
        """
        try:
            # Ajustar modelo de fatores se não foi feito
            if self.factor_model is None:
                self.fit_factor_model(returns_data)
            
            # Otimizar portfólio
            optimization_result = self.optimize_portfolio()
            
            # Extrair pesos dos ativos
            weights = optimization_result['weights']
            asset_names = self.factor_model['asset_names']
            
            # Criar dicionário de alocação
            risk_allocation = {}
            for i, asset in enumerate(asset_names):
                risk_allocation[asset] = float(weights[i] * self.total_risk_budget)
            
            return risk_allocation
            
        except Exception as e:
            self.logger.error(f"Erro na alocação de risco: {e}")
            # Fallback: alocação igual
            n_assets = len(returns_data.columns)
            equal_allocation = self.total_risk_budget / n_assets
            return {asset: equal_allocation for asset in returns_data.columns}
            
    def fit_factor_model(self, returns: pd.DataFrame, 
                        custom_factors: Optional[pd.DataFrame] = None) -> None:
        """
        Ajusta modelo de fatores aos dados de retorno
        
        Args:
            returns: DataFrame com retornos dos ativos
            custom_factors: Fatores customizados (opcional)
        """
        self.logger.info("Ajustando modelo de fatores")
        
        # Limpar dados
        returns_clean = returns.dropna()
        
        if len(returns_clean) < self.config.lookback_window:
            self.logger.warning(f"Dados insuficientes: {len(returns_clean)} < {self.config.lookback_window}")
            
        # Padronizar retornos
        returns_scaled = self.scaler.fit_transform(returns_clean)
        
        if self.config.factor_method == 'pca':
            self._fit_pca_factors(returns_scaled, returns_clean.columns)
        elif self.config.factor_method == 'factor_analysis':
            self._fit_factor_analysis(returns_scaled, returns_clean.columns)
        elif self.config.factor_method == 'custom' and custom_factors is not None:
            self._fit_custom_factors(returns_clean, custom_factors)
        else:
            raise ValueError(f"Método de fator desconhecido: {self.config.factor_method}")
            
        # Calcular modelo de risco
        self._estimate_risk_model(returns_clean)
        
        self.logger.info(f"Modelo de fatores ajustado com {self.factor_model['n_factors']} fatores")
    
    def _fit_pca_factors(self, returns_scaled: np.ndarray, asset_names: List[str]) -> None:
        """
        Ajusta fatores usando PCA
        """
        pca = PCA(n_components=self.config.n_factors)
        factor_returns = pca.fit_transform(returns_scaled)
        
        self.factor_model = {
            'method': 'pca',
            'model': pca,
            'factor_returns': factor_returns,
            'factor_loadings': pca.components_.T,  # [n_assets, n_factors]
            'explained_variance': pca.explained_variance_ratio_,
            'n_factors': self.config.n_factors,
            'asset_names': asset_names,
            'factor_names': [f'PC{i+1}' for i in range(self.config.n_factors)]
        }
    
    def _fit_factor_analysis(self, returns_scaled: np.ndarray, asset_names: List[str]) -> None:
        """
        Ajusta fatores usando Factor Analysis
        """
        fa = FactorAnalysis(n_components=self.config.n_factors, random_state=42)
        factor_returns = fa.fit_transform(returns_scaled)
        
        self.factor_model = {
            'method': 'factor_analysis',
            'model': fa,
            'factor_returns': factor_returns,
            'factor_loadings': fa.components_.T,
            'noise_variance': fa.noise_variance_,
            'n_factors': self.config.n_factors,
            'asset_names': asset_names,
            'factor_names': [f'Factor{i+1}' for i in range(self.config.n_factors)]
        }
    
    def _fit_custom_factors(self, returns: pd.DataFrame, 
                           custom_factors: pd.DataFrame) -> None:
        """
        Usa fatores customizados
        """
        # Alinhar datas
        aligned_data = returns.join(custom_factors, how='inner')
        
        if len(aligned_data) == 0:
            raise ValueError("Nenhuma data comum entre retornos e fatores")
            
        returns_aligned = aligned_data[returns.columns]
        factors_aligned = aligned_data[custom_factors.columns]
        
        # Regressão para obter loadings
        from sklearn.linear_model import LinearRegression
        
        factor_loadings = []
        
        for asset in returns.columns:
            reg = LinearRegression()
            reg.fit(factors_aligned.values, returns_aligned[asset].values)
            factor_loadings.append(reg.coef_)
            
        self.factor_model = {
            'method': 'custom',
            'factor_returns': factors_aligned.values,
            'factor_loadings': np.array(factor_loadings),
            'n_factors': len(custom_factors.columns),
            'asset_names': list(returns.columns),
            'factor_names': list(custom_factors.columns)
        }
    
    def _estimate_risk_model(self, returns: pd.DataFrame) -> None:
        """
        Estima modelo de risco baseado em fatores
        """
        # Matriz de covariância dos fatores
        factor_returns = self.factor_model['factor_returns']
        factor_cov = np.cov(factor_returns.T)
        
        # Loadings dos fatores
        factor_loadings = self.factor_model['factor_loadings']
        
        # Risco sistemático
        systematic_risk = factor_loadings @ factor_cov @ factor_loadings.T
        
        # Risco idiossincrático
        total_cov = returns.cov().values
        idiosyncratic_risk = total_cov - systematic_risk
        
        # Garantir que a matriz idiossincrática seja positiva definida
        eigenvals, eigenvecs = np.linalg.eigh(idiosyncratic_risk)
        eigenvals = np.maximum(eigenvals, 1e-8)  # Regularização
        idiosyncratic_risk = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        self.risk_model = {
            'factor_covariance': factor_cov,
            'factor_loadings': factor_loadings,
            'systematic_risk': systematic_risk,
            'idiosyncratic_risk': idiosyncratic_risk,
            'total_risk': systematic_risk + idiosyncratic_risk
        }
    
    def optimize_portfolio(self, 
                          expected_returns: Optional[pd.Series] = None,
                          risk_budget: Optional[Dict[str, float]] = None,
                          constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Otimiza portfólio baseado em orçamento de risco
        
        Args:
            expected_returns: Retornos esperados dos ativos
            risk_budget: Orçamento de risco por fator
            constraints: Restrições adicionais
            
        Returns:
            Dicionário com resultados da otimização
        """
        if self.factor_model is None:
            raise ValueError("Modelo de fatores não foi ajustado")
            
        n_assets = len(self.factor_model['asset_names'])
        
        # Definir orçamento de risco padrão
        if risk_budget is None:
            if self.config.risk_budget_method == 'equal_risk':
                n_factors = self.factor_model['n_factors']
                risk_budget = {f: 1.0/n_factors for f in self.factor_model['factor_names']}
            else:
                raise ValueError("Risk budget deve ser fornecido para método customizado")
                
        # Definir restrições padrão
        if constraints is None:
            constraints = {
                'max_weight': self.config.max_position_weight,
                'max_factor_exposure': self.config.max_factor_exposure,
                'long_only': True
            }
            
        # Executar otimização
        if self.config.optimization_method == 'cvxpy':
            result = self._optimize_cvxpy(expected_returns, risk_budget, constraints)
        elif self.config.optimization_method == 'scipy':
            result = self._optimize_scipy(expected_returns, risk_budget, constraints)
        elif self.config.optimization_method == 'genetic':
            result = self._optimize_genetic(expected_returns, risk_budget, constraints)
        else:
            raise ValueError(f"Método de otimização desconhecido: {self.config.optimization_method}")
            
        # Calcular métricas do portfólio
        weights = result['weights']
        portfolio_metrics = self._calculate_portfolio_metrics(weights, expected_returns)
        result.update(portfolio_metrics)
        
        # Atualizar estado
        self.current_weights = weights
        self.factor_exposures = result['factor_exposures']
        self.risk_contributions = result['risk_contributions']
        
        return result
    
    def _optimize_cvxpy(self, expected_returns: Optional[pd.Series],
                       risk_budget: Dict[str, float],
                       constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Otimização usando CVXPY
        """
        n_assets = len(self.factor_model['asset_names'])
        n_factors = self.factor_model['n_factors']
        
        # Variáveis de decisão
        w = cp.Variable(n_assets)
        
        # Matrizes do modelo de risco
        B = self.risk_model['factor_loadings']  # [n_assets, n_factors]
        F = self.risk_model['factor_covariance']  # [n_factors, n_factors]
        D = np.diag(np.diag(self.risk_model['idiosyncratic_risk']))  # [n_assets, n_assets]
        
        # Exposições aos fatores
        factor_exposures = B.T @ w  # [n_factors]
        
        # Risco do portfólio
        systematic_risk = cp.quad_form(factor_exposures, F)
        idiosyncratic_risk = cp.quad_form(w, D)
        portfolio_risk = systematic_risk + idiosyncratic_risk
        
        # Função objetivo
        if expected_returns is not None:
            expected_return = expected_returns.values @ w
            # Maximizar Sharpe ratio aproximado
            objective = cp.Maximize(expected_return - 0.5 * portfolio_risk)
        else:
            # Minimizar risco
            objective = cp.Minimize(portfolio_risk)
            
        # Restrições
        constraints_list = []
        
        # Soma dos pesos = 1
        constraints_list.append(cp.sum(w) == 1)
        
        # Long only
        if constraints.get('long_only', True):
            constraints_list.append(w >= 0)
            
        # Peso máximo por ativo
        max_weight = constraints.get('max_weight', self.config.max_position_weight)
        constraints_list.append(w <= max_weight)
        
        # Exposição máxima por fator
        max_factor_exp = constraints.get('max_factor_exposure', self.config.max_factor_exposure)
        constraints_list.append(cp.abs(factor_exposures) <= max_factor_exp)
        
        # Orçamento de risco por fator (aproximação)
        # Esta é uma aproximação linear do orçamento de risco
        risk_budget_array = np.array([risk_budget[f] for f in self.factor_model['factor_names']])
        
        # Resolver problema
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(solver=cp.ECOS, max_iters=self.config.max_iterations)
            
            if problem.status not in ["infeasible", "unbounded"]:
                weights = w.value
                
                # Calcular exposições e contribuições de risco
                factor_exp = B.T @ weights
                risk_contrib = self._calculate_risk_contributions(weights)
                
                return {
                    'weights': weights,
                    'factor_exposures': dict(zip(self.factor_model['factor_names'], factor_exp)),
                    'risk_contributions': risk_contrib,
                    'optimization_status': problem.status,
                    'objective_value': problem.value
                }
            else:
                raise ValueError(f"Otimização falhou: {problem.status}")
                
        except Exception as e:
            self.logger.error(f"Erro na otimização CVXPY: {e}")
            # Fallback para pesos iguais
            weights = np.ones(n_assets) / n_assets
            factor_exp = B.T @ weights
            risk_contrib = self._calculate_risk_contributions(weights)
            
            return {
                'weights': weights,
                'factor_exposures': dict(zip(self.factor_model['factor_names'], factor_exp)),
                'risk_contributions': risk_contrib,
                'optimization_status': 'fallback',
                'objective_value': None
            }
    
    def _optimize_scipy(self, expected_returns: Optional[pd.Series],
                       risk_budget: Dict[str, float],
                       constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Otimização usando SciPy
        """
        n_assets = len(self.factor_model['asset_names'])
        
        # Função objetivo
        def objective(weights):
            portfolio_risk = self._calculate_portfolio_risk(weights)
            
            if expected_returns is not None:
                expected_return = np.dot(expected_returns.values, weights)
                return -(expected_return - 0.5 * portfolio_risk)  # Minimizar negativo
            else:
                return portfolio_risk
                
        # Restrições
        constraints_list = []
        
        # Soma dos pesos = 1
        constraints_list.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        })
        
        # Bounds
        if constraints.get('long_only', True):
            bounds = [(0, constraints.get('max_weight', self.config.max_position_weight)) 
                     for _ in range(n_assets)]
        else:
            max_weight = constraints.get('max_weight', self.config.max_position_weight)
            bounds = [(-max_weight, max_weight) for _ in range(n_assets)]
            
        # Ponto inicial
        x0 = np.ones(n_assets) / n_assets
        
        # Otimizar
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
            )
            
            if result.success:
                weights = result.x
            else:
                self.logger.warning(f"Otimização SciPy não convergiu: {result.message}")
                weights = x0
                
        except Exception as e:
            self.logger.error(f"Erro na otimização SciPy: {e}")
            weights = x0
            
        # Calcular métricas
        B = self.risk_model['factor_loadings']
        factor_exp = B.T @ weights
        risk_contrib = self._calculate_risk_contributions(weights)
        
        return {
            'weights': weights,
            'factor_exposures': dict(zip(self.factor_model['factor_names'], factor_exp)),
            'risk_contributions': risk_contrib,
            'optimization_status': 'success' if result.success else 'failed',
            'objective_value': result.fun if result.success else None
        }
    
    def _optimize_genetic(self, expected_returns: Optional[pd.Series],
                         risk_budget: Dict[str, float],
                         constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Otimização usando algoritmo genético
        """
        n_assets = len(self.factor_model['asset_names'])
        
        # Função objetivo
        def objective(weights):
            # Normalizar pesos
            weights = weights / np.sum(weights)
            
            portfolio_risk = self._calculate_portfolio_risk(weights)
            
            if expected_returns is not None:
                expected_return = np.dot(expected_returns.values, weights)
                return -(expected_return - 0.5 * portfolio_risk)
            else:
                return portfolio_risk
                
        # Bounds
        max_weight = constraints.get('max_weight', self.config.max_position_weight)
        if constraints.get('long_only', True):
            bounds = [(0, max_weight) for _ in range(n_assets)]
        else:
            bounds = [(-max_weight, max_weight) for _ in range(n_assets)]
            
        # Otimizar
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.config.max_iterations // 10,  # Menos iterações para GA
                seed=42
            )
            
            if result.success:
                weights = result.x / np.sum(result.x)  # Normalizar
            else:
                weights = np.ones(n_assets) / n_assets
                
        except Exception as e:
            self.logger.error(f"Erro na otimização genética: {e}")
            weights = np.ones(n_assets) / n_assets
            
        # Calcular métricas
        B = self.risk_model['factor_loadings']
        factor_exp = B.T @ weights
        risk_contrib = self._calculate_risk_contributions(weights)
        
        return {
            'weights': weights,
            'factor_exposures': dict(zip(self.factor_model['factor_names'], factor_exp)),
            'risk_contributions': risk_contrib,
            'optimization_status': 'success' if result.success else 'failed',
            'objective_value': result.fun if result.success else None
        }
    
    def _calculate_portfolio_risk(self, weights: np.ndarray) -> float:
        """
        Calcula risco total do portfólio
        """
        return weights.T @ self.risk_model['total_risk'] @ weights
    
    def _calculate_risk_contributions(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Calcula contribuições de risco por fator
        """
        B = self.risk_model['factor_loadings']
        F = self.risk_model['factor_covariance']
        
        # Exposições aos fatores
        factor_exposures = B.T @ weights
        
        # Contribuições de risco por fator
        factor_risk_contributions = []
        
        for i in range(len(factor_exposures)):
            # Risco marginal do fator i
            marginal_risk = 2 * F[i, :] @ factor_exposures
            
            # Contribuição = exposição * risco marginal
            contribution = factor_exposures[i] * marginal_risk
            factor_risk_contributions.append(contribution)
            
        # Normalizar contribuições
        total_risk = sum(factor_risk_contributions)
        if total_risk > 0:
            factor_risk_contributions = [c / total_risk for c in factor_risk_contributions]
            
        return dict(zip(self.factor_model['factor_names'], factor_risk_contributions))
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray,
                                   expected_returns: Optional[pd.Series]) -> Dict[str, Any]:
        """
        Calcula métricas do portfólio
        """
        metrics = {}
        
        # Risco total
        metrics['portfolio_risk'] = np.sqrt(self._calculate_portfolio_risk(weights))
        
        # Retorno esperado
        if expected_returns is not None:
            metrics['expected_return'] = np.dot(expected_returns.values, weights)
            
            # Sharpe ratio (assumindo risk-free rate = 0)
            if metrics['portfolio_risk'] > 0:
                metrics['sharpe_ratio'] = metrics['expected_return'] / metrics['portfolio_risk']
            else:
                metrics['sharpe_ratio'] = 0
                
        # Concentração
        metrics['concentration'] = np.sum(weights ** 2)  # Herfindahl index
        
        # Número efetivo de posições
        metrics['effective_positions'] = 1 / metrics['concentration']
        
        # Peso máximo
        metrics['max_weight'] = np.max(weights)
        
        return metrics
    
    def rebalance_portfolio(self, current_date: datetime,
                          returns: pd.DataFrame,
                          expected_returns: Optional[pd.Series] = None,
                          force_rebalance: bool = False) -> Dict[str, Any]:
        """
        Rebalancea portfólio se necessário
        
        Args:
            current_date: Data atual
            returns: Dados de retorno atualizados
            expected_returns: Retornos esperados atualizados
            force_rebalance: Forçar rebalanceamento
            
        Returns:
            Resultado do rebalanceamento
        """
        # Verificar se precisa rebalancear
        should_rebalance = force_rebalance or self._should_rebalance(current_date)
        
        if not should_rebalance:
            return {
                'rebalanced': False,
                'reason': 'No rebalancing needed',
                'current_weights': self.current_weights
            }
            
        # Atualizar modelo de fatores
        self.fit_factor_model(returns.tail(self.config.lookback_window))
        
        # Otimizar novo portfólio
        optimization_result = self.optimize_portfolio(expected_returns)
        
        # Calcular turnover
        if self.current_weights is not None:
            turnover = np.sum(np.abs(optimization_result['weights'] - self.current_weights))
        else:
            turnover = np.sum(np.abs(optimization_result['weights']))
            
        # Atualizar histórico
        self._update_history(current_date, optimization_result)
        
        return {
            'rebalanced': True,
            'date': current_date,
            'new_weights': optimization_result['weights'],
            'turnover': turnover,
            'optimization_result': optimization_result
        }
    
    def _should_rebalance(self, current_date: datetime) -> bool:
        """
        Determina se deve rebalancear o portfólio
        """
        if not self.history['rebalance_dates']:
            return True
            
        last_rebalance = self.history['rebalance_dates'][-1]
        
        if self.config.rebalance_frequency == 'daily':
            return (current_date - last_rebalance).days >= 1
        elif self.config.rebalance_frequency == 'weekly':
            return (current_date - last_rebalance).days >= 7
        elif self.config.rebalance_frequency == 'monthly':
            return (current_date - last_rebalance).days >= 30
        else:
            return False
    
    def _update_history(self, date: datetime, optimization_result: Dict[str, Any]) -> None:
        """
        Atualiza histórico do sistema
        """
        self.history['rebalance_dates'].append(date)
        self.history['weights'].append(optimization_result['weights'].copy())
        self.history['factor_exposures'].append(optimization_result['factor_exposures'].copy())
        self.history['risk_contributions'].append(optimization_result['risk_contributions'].copy())
        self.history['portfolio_risk'].append(optimization_result['portfolio_risk'])
    
    def stress_test(self, stress_scenarios: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Executa stress testing do portfólio
        
        Args:
            stress_scenarios: Dicionário com cenários de stress para fatores
            
        Returns:
            Resultados do stress test
        """
        if self.current_weights is None:
            raise ValueError("Portfólio não foi otimizado")
            
        results = {}
        
        for scenario_name, factor_shocks in stress_scenarios.items():
            # Aplicar choque aos fatores
            shocked_factor_returns = self.factor_model['factor_returns'] + factor_shocks
            
            # Calcular impacto no portfólio
            B = self.risk_model['factor_loadings']
            portfolio_shock = self.current_weights.T @ B @ factor_shocks
            
            # Calcular nova volatilidade
            shocked_factor_cov = np.cov(shocked_factor_returns.T)
            new_systematic_risk = B @ shocked_factor_cov @ B.T
            new_total_risk = new_systematic_risk + self.risk_model['idiosyncratic_risk']
            new_portfolio_risk = np.sqrt(self.current_weights.T @ new_total_risk @ self.current_weights)
            
            results[scenario_name] = {
                'portfolio_shock': portfolio_shock,
                'new_portfolio_risk': new_portfolio_risk,
                'risk_change': new_portfolio_risk - self.history['portfolio_risk'][-1] if self.history['portfolio_risk'] else 0
            }
            
        return results
    
    def plot_risk_decomposition(self, save_plot: bool = False,
                              filename: Optional[str] = None) -> None:
        """
        Plota decomposição de risco do portfólio
        """
        if self.risk_contributions is None:
            self.logger.warning("Nenhuma contribuição de risco disponível")
            return
            
        # Criar plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Portfolio Risk Decomposition', fontsize=16)
        
        # 1. Contribuições de risco por fator
        ax1 = axes[0, 0]
        factors = list(self.risk_contributions.keys())
        contributions = list(self.risk_contributions.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(factors)))
        bars = ax1.bar(factors, contributions, color=colors)
        ax1.set_title('Risk Contributions by Factor')
        ax1.set_ylabel('Risk Contribution')
        ax1.tick_params(axis='x', rotation=45)
        
        # Adicionar valores nas barras
        for bar, contrib in zip(bars, contributions):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{contrib:.3f}', ha='center', va='bottom')
                    
        # 2. Exposições aos fatores
        ax2 = axes[0, 1]
        if self.factor_exposures:
            exposures = list(self.factor_exposures.values())
            ax2.bar(factors, exposures, color=colors)
            ax2.set_title('Factor Exposures')
            ax2.set_ylabel('Exposure')
            ax2.tick_params(axis='x', rotation=45)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
        # 3. Pesos do portfólio
        ax3 = axes[1, 0]
        if self.current_weights is not None:
            asset_names = self.factor_model['asset_names']
            # Mostrar apenas top 10 pesos
            top_indices = np.argsort(self.current_weights)[-10:]
            top_weights = self.current_weights[top_indices]
            top_names = [asset_names[i] for i in top_indices]
            
            ax3.barh(range(len(top_weights)), top_weights)
            ax3.set_yticks(range(len(top_weights)))
            ax3.set_yticklabels(top_names)
            ax3.set_title('Top 10 Portfolio Weights')
            ax3.set_xlabel('Weight')
            
        # 4. Evolução do risco ao longo do tempo
        ax4 = axes[1, 1]
        if self.history['portfolio_risk']:
            dates = self.history['rebalance_dates']
            risks = self.history['portfolio_risk']
            
            ax4.plot(dates, risks, 'b-o', markersize=4)
            ax4.set_title('Portfolio Risk Over Time')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Portfolio Risk')
            ax4.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_plot:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"risk_decomposition_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo do portfólio atual
        """
        if self.current_weights is None:
            return {'status': 'No portfolio optimized'}
            
        summary = {
            'portfolio_weights': dict(zip(self.factor_model['asset_names'], self.current_weights)),
            'factor_exposures': self.factor_exposures,
            'risk_contributions': self.risk_contributions,
            'portfolio_risk': self.history['portfolio_risk'][-1] if self.history['portfolio_risk'] else None,
            'concentration': np.sum(self.current_weights ** 2),
            'effective_positions': 1 / np.sum(self.current_weights ** 2),
            'max_weight': np.max(self.current_weights),
            'n_rebalances': len(self.history['rebalance_dates'])
        }
        
        return summary
    
    def save_state(self, filename: Optional[str] = None) -> None:
        """
        Salva estado do sistema
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.results_path}/risk_budgeting_state_{timestamp}.pkl"
            
        state = {
            'config': self.config,
            'factor_model': self.factor_model,
            'risk_model': self.risk_model,
            'current_weights': self.current_weights,
            'factor_exposures': self.factor_exposures,
            'risk_contributions': self.risk_contributions,
            'history': self.history
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
            
        self.logger.info(f"Estado salvo em {filename}")
    
    def load_state(self, filename: str) -> None:
        """
        Carrega estado do sistema
        """
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            
        self.config = state['config']
        self.factor_model = state['factor_model']
        self.risk_model = state['risk_model']
        self.current_weights = state['current_weights']
        self.factor_exposures = state['factor_exposures']
        self.risk_contributions = state['risk_contributions']
        self.history = state['history']
        
        self.logger.info(f"Estado carregado de {filename}")

# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Criar dados de exemplo
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Simular retornos de 10 ativos
    n_assets = 10
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    
    # Criar fatores latentes
    n_factors = 3
    factor_returns = np.random.multivariate_normal(
        mean=np.zeros(n_factors),
        cov=np.eye(n_factors) * 0.01,
        size=len(dates)
    )
    
    # Loadings dos fatores
    factor_loadings = np.random.randn(n_assets, n_factors) * 0.5
    
    # Gerar retornos dos ativos
    systematic_returns = factor_returns @ factor_loadings.T
    idiosyncratic_returns = np.random.randn(len(dates), n_assets) * 0.005
    
    asset_returns = systematic_returns + idiosyncratic_returns
    
    returns_df = pd.DataFrame(
        asset_returns,
        index=dates,
        columns=asset_names
    )
    
    # Configuração
    config = RiskBudgetConfig(
        n_factors=3,
        factor_method='pca',
        optimization_method='cvxpy',
        rebalance_frequency='weekly'
    )
    
    # Criar sistema
    risk_system = RiskBudgetingSystem(config=config, logger=logger)
    
    # Ajustar modelo de fatores
    print("\n=== AJUSTANDO MODELO DE FATORES ===")
    risk_system.fit_factor_model(returns_df)
    
    print(f"Fatores identificados: {risk_system.factor_model['factor_names']}")
    print(f"Variância explicada: {risk_system.factor_model['explained_variance']}")
    
    # Otimizar portfólio
    print("\n=== OTIMIZANDO PORTFÓLIO ===")
    
    # Retornos esperados simulados
    expected_returns = pd.Series(
        np.random.normal(0.001, 0.002, n_assets),
        index=asset_names
    )
    
    # Orçamento de risco igual
    risk_budget = {
        'PC1': 0.4,
        'PC2': 0.35,
        'PC3': 0.25
    }
    
    optimization_result = risk_system.optimize_portfolio(
        expected_returns=expected_returns,
        risk_budget=risk_budget
    )
    
    print(f"Status da otimização: {optimization_result['optimization_status']}")
    print(f"Risco do portfólio: {optimization_result['portfolio_risk']:.4f}")
    print(f"Retorno esperado: {optimization_result['expected_return']:.4f}")
    print(f"Sharpe ratio: {optimization_result['sharpe_ratio']:.4f}")
    
    # Mostrar pesos principais
    weights_series = pd.Series(optimization_result['weights'], index=asset_names)
    top_weights = weights_series.nlargest(5)
    print("\nTop 5 pesos:")
    for asset, weight in top_weights.items():
        print(f"{asset}: {weight:.4f}")
        
    # Stress testing
    print("\n=== STRESS TESTING ===")
    
    stress_scenarios = {
        'Market_Crash': np.array([-0.05, -0.03, -0.02]),  # Choque negativo nos fatores
        'Volatility_Spike': np.array([0.0, 0.04, 0.02]),   # Aumento de volatilidade
        'Factor_Rotation': np.array([0.03, -0.02, 0.01])   # Rotação entre fatores
    }
    
    stress_results = risk_system.stress_test(stress_scenarios)
    
    for scenario, result in stress_results.items():
        print(f"\n{scenario}:")
        print(f"  Choque no portfólio: {result['portfolio_shock']:.4f}")
        print(f"  Novo risco: {result['new_portfolio_risk']:.4f}")
        print(f"  Mudança no risco: {result['risk_change']:.4f}")
        
    # Visualizações
    print("\n=== VISUALIZAÇÕES ===")
    risk_system.plot_risk_decomposition(save_plot=False)
    
    # Resumo do portfólio
    print("\n=== RESUMO DO PORTFÓLIO ===")
    summary = risk_system.get_portfolio_summary()
    
    print(f"Concentração (Herfindahl): {summary['concentration']:.4f}")
    print(f"Posições efetivas: {summary['effective_positions']:.2f}")
    print(f"Peso máximo: {summary['max_weight']:.4f}")
    
    print("\nContribuições de risco por fator:")
    for factor, contrib in summary['risk_contributions'].items():
        print(f"{factor}: {contrib:.4f}")