import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Tuple, Callable
import warnings
from scipy.optimize import minimize, Bounds
from scipy import linalg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
from datetime import datetime, timedelta


class PortfolioOptimizer:
    """
    Classe para otimização de portfólio usando diferentes metodologias.
    
    Implementa Mean-Variance Optimization, Black-Litterman, Risk Parity,
    e outras técnicas avançadas de alocação de ativos.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 confidence_level: float = 0.95):
        """
        Inicializa o otimizador de portfólio.
        
        Args:
            risk_free_rate: Taxa livre de risco anual
            confidence_level: Nível de confiança para VaR
        """
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.returns_data = None
        self.expected_returns = None
        self.covariance_matrix = None
        self.assets = None
    
    def load_data(self, 
                  returns: pd.DataFrame,
                  expected_returns: Optional[pd.Series] = None) -> None:
        """
        Carrega dados de retornos e retornos esperados.
        
        Args:
            returns: DataFrame com retornos históricos dos ativos
            expected_returns: Série com retornos esperados (opcional)
        """
        self.returns_data = returns.dropna()
        self.assets = list(returns.columns)
        
        if expected_returns is not None:
            self.expected_returns = expected_returns
        else:
            # Usar média histórica como proxy
            self.expected_returns = returns.mean() * 252  # Anualizar
        
        # Calcular matriz de covariância
        self.covariance_matrix = returns.cov() * 252  # Anualizar
    
    def estimate_covariance_matrix(self, 
                                 method: str = 'sample',
                                 shrinkage_target: Optional[str] = None) -> np.ndarray:
        """
        Estima matriz de covariância usando diferentes métodos.
        
        Args:
            method: Método de estimação ('sample', 'ledoit_wolf', 'oas', 'shrunk')
            shrinkage_target: Alvo de shrinkage ('diagonal', 'identity', 'constant_correlation')
            
        Returns:
            Matriz de covariância estimada
        """
        if self.returns_data is None:
            raise ValueError("Dados não carregados. Use load_data() primeiro.")
        
        returns_matrix = self.returns_data.values
        
        if method == 'sample':
            cov_matrix = np.cov(returns_matrix.T) * 252
        
        elif method == 'ledoit_wolf':
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_matrix).covariance_ * 252
        
        elif method == 'oas':
            oas = OAS()
            cov_matrix = oas.fit(returns_matrix).covariance_ * 252
        
        elif method == 'shrunk':
            if shrinkage_target == 'diagonal':
                # Shrinkage para matriz diagonal
                sample_cov = np.cov(returns_matrix.T)
                target = np.diag(np.diag(sample_cov))
            elif shrinkage_target == 'identity':
                # Shrinkage para matriz identidade
                target = np.eye(returns_matrix.shape[1]) * np.trace(np.cov(returns_matrix.T)) / returns_matrix.shape[1]
            else:
                # Shrinkage padrão
                target = None
            
            shrunk = ShrunkCovariance(shrinkage=0.1)
            cov_matrix = shrunk.fit(returns_matrix).covariance_ * 252
        
        else:
            raise ValueError(f"Método {method} não suportado")
        
        self.covariance_matrix = cov_matrix
        return cov_matrix
    
    def mean_variance_optimization(self, 
                                 target_return: Optional[float] = None,
                                 target_risk: Optional[float] = None,
                                 weight_bounds: Tuple[float, float] = (0.0, 1.0),
                                 long_only: bool = True) -> Dict[str, Union[np.ndarray, float]]:
        """
        Otimização Mean-Variance de Markowitz.
        
        Args:
            target_return: Retorno alvo (se None, maximiza Sharpe ratio)
            target_risk: Risco alvo (volatilidade)
            weight_bounds: Limites para pesos dos ativos
            long_only: Se permite apenas posições longas
            
        Returns:
            Dicionário com pesos ótimos e métricas
        """
        if self.expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Dados não carregados ou matriz de covariância não calculada")
        
        n_assets = len(self.assets)
        mu = self.expected_returns.values
        Sigma = self.covariance_matrix
        
        # Função objetivo e restrições
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(Sigma, weights))
        
        def portfolio_return(weights):
            return np.dot(weights, mu)
        
        # Restrições
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Soma = 1
        
        if target_return is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: portfolio_return(x) - target_return
            })
        
        if target_risk is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sqrt(portfolio_variance(x)) - target_risk
            })
        
        # Bounds para pesos
        if long_only:
            bounds = Bounds(lb=np.full(n_assets, weight_bounds[0]), 
                          ub=np.full(n_assets, weight_bounds[1]))
        else:
            bounds = Bounds(lb=np.full(n_assets, -1.0), 
                          ub=np.full(n_assets, 1.0))
        
        # Ponto inicial
        x0 = np.full(n_assets, 1.0 / n_assets)
        
        # Otimização
        if target_return is None and target_risk is None:
            # Maximizar Sharpe ratio
            def negative_sharpe(weights):
                ret = portfolio_return(weights)
                vol = np.sqrt(portfolio_variance(weights))
                return -(ret - self.risk_free_rate) / vol
            
            result = minimize(negative_sharpe, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
        else:
            # Minimizar variância
            result = minimize(portfolio_variance, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
        
        if not result.success:
            warnings.warn(f"Otimização falhou: {result.message}")
        
        optimal_weights = result.x
        
        # Calcular métricas do portfólio ótimo
        portfolio_ret = portfolio_return(optimal_weights)
        portfolio_vol = np.sqrt(portfolio_variance(optimal_weights))
        sharpe_ratio = (portfolio_ret - self.risk_free_rate) / portfolio_vol
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_ret,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'success': result.success
        }
    
    def efficient_frontier(self, 
                         n_points: int = 100,
                         weight_bounds: Tuple[float, float] = (0.0, 1.0)) -> pd.DataFrame:
        """
        Calcula a fronteira eficiente.
        
        Args:
            n_points: Número de pontos na fronteira
            weight_bounds: Limites para pesos
            
        Returns:
            DataFrame com pontos da fronteira eficiente
        """
        # Range de retornos para calcular fronteira
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier_data = []
        
        for target_ret in target_returns:
            try:
                result = self.mean_variance_optimization(
                    target_return=target_ret,
                    weight_bounds=weight_bounds
                )
                
                if result['success']:
                    frontier_data.append({
                        'return': result['expected_return'],
                        'volatility': result['volatility'],
                        'sharpe_ratio': result['sharpe_ratio']
                    })
            except:
                continue
        
        return pd.DataFrame(frontier_data)
    
    def black_litterman_optimization(self, 
                                   views: Dict[str, float],
                                   view_confidences: Dict[str, float],
                                   market_caps: Optional[pd.Series] = None,
                                   tau: float = 0.025,
                                   risk_aversion: float = 3.0) -> Dict[str, Union[np.ndarray, float]]:
        """
        Otimização Black-Litterman.
        
        Args:
            views: Dicionário com views dos ativos {asset: expected_return}
            view_confidences: Confiança nas views {asset: confidence}
            market_caps: Capitalização de mercado dos ativos
            tau: Parâmetro de incerteza
            risk_aversion: Coeficiente de aversão ao risco
            
        Returns:
            Dicionário com pesos ótimos Black-Litterman
        """
        if self.covariance_matrix is None:
            raise ValueError("Matriz de covariância não calculada")
        
        n_assets = len(self.assets)
        Sigma = self.covariance_matrix
        
        # Prior: retornos implícitos do mercado
        if market_caps is not None:
            # Usar market caps como proxy para pesos de mercado
            w_market = market_caps / market_caps.sum()
            pi = risk_aversion * np.dot(Sigma, w_market.values)
        else:
            # Usar pesos iguais como proxy
            w_market = np.full(n_assets, 1.0 / n_assets)
            pi = risk_aversion * np.dot(Sigma, w_market)
        
        # Matriz P: identifica quais ativos têm views
        view_assets = [asset for asset in views.keys() if asset in self.assets]
        n_views = len(view_assets)
        
        if n_views == 0:
            warnings.warn("Nenhuma view válida fornecida")
            return self.mean_variance_optimization()
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        
        for i, asset in enumerate(view_assets):
            asset_idx = self.assets.index(asset)
            P[i, asset_idx] = 1.0
            Q[i] = views[asset]
        
        # Matriz Omega: incerteza das views
        Omega = np.diag([1.0 / view_confidences.get(asset, 1.0) for asset in view_assets])
        
        # Cálculos Black-Litterman
        tau_Sigma = tau * Sigma
        
        # Novo vetor de retornos esperados
        M1 = linalg.inv(tau_Sigma)
        M2 = np.dot(P.T, np.dot(linalg.inv(Omega), P))
        M3 = np.dot(linalg.inv(tau_Sigma), pi) + np.dot(P.T, np.dot(linalg.inv(Omega), Q))
        
        mu_bl = np.dot(linalg.inv(M1 + M2), M3)
        
        # Nova matriz de covariância
        Sigma_bl = linalg.inv(M1 + M2)
        
        # Otimização com novos parâmetros
        def portfolio_variance_bl(weights):
            return np.dot(weights.T, np.dot(Sigma_bl, weights))
        
        def portfolio_return_bl(weights):
            return np.dot(weights, mu_bl)
        
        def negative_sharpe_bl(weights):
            ret = portfolio_return_bl(weights)
            vol = np.sqrt(portfolio_variance_bl(weights))
            return -(ret - self.risk_free_rate) / vol
        
        # Restrições
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = Bounds(lb=np.zeros(n_assets), ub=np.ones(n_assets))
        x0 = np.full(n_assets, 1.0 / n_assets)
        
        result = minimize(negative_sharpe_bl, x0, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return_bl(optimal_weights),
            'volatility': np.sqrt(portfolio_variance_bl(optimal_weights)),
            'sharpe_ratio': (portfolio_return_bl(optimal_weights) - self.risk_free_rate) / 
                          np.sqrt(portfolio_variance_bl(optimal_weights)),
            'bl_returns': mu_bl,
            'bl_covariance': Sigma_bl,
            'success': result.success
        }
    
    def risk_parity_optimization(self, 
                               method: str = 'equal_risk_contribution') -> Dict[str, Union[np.ndarray, float]]:
        """
        Otimização Risk Parity.
        
        Args:
            method: Método de risk parity ('equal_risk_contribution', 'inverse_volatility')
            
        Returns:
            Dicionário com pesos ótimos de risk parity
        """
        if self.covariance_matrix is None:
            raise ValueError("Matriz de covariância não calculada")
        
        n_assets = len(self.assets)
        Sigma = self.covariance_matrix
        
        if method == 'inverse_volatility':
            # Método simples: pesos inversamente proporcionais à volatilidade
            volatilities = np.sqrt(np.diag(Sigma))
            weights = 1.0 / volatilities
            weights = weights / weights.sum()
            
        elif method == 'equal_risk_contribution':
            # Método ERC: contribuições de risco iguais
            def risk_budget_objective(weights):
                """
                Função objetivo para Equal Risk Contribution.
                Minimiza a diferença entre contribuições de risco.
                """
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
                marginal_contrib = np.dot(Sigma, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                
                # Minimizar soma dos quadrados das diferenças
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib) ** 2)
            
            # Restrições
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = Bounds(lb=np.full(n_assets, 0.001), ub=np.full(n_assets, 1.0))
            x0 = np.full(n_assets, 1.0 / n_assets)
            
            result = minimize(risk_budget_objective, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            if not result.success:
                warnings.warn(f"Otimização Risk Parity falhou: {result.message}")
                # Fallback para inverse volatility
                volatilities = np.sqrt(np.diag(Sigma))
                weights = 1.0 / volatilities
                weights = weights / weights.sum()
            else:
                weights = result.x
        
        else:
            raise ValueError(f"Método {method} não suportado")
        
        # Calcular métricas
        portfolio_ret = np.dot(weights, self.expected_returns.values)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
        sharpe_ratio = (portfolio_ret - self.risk_free_rate) / portfolio_vol
        
        return {
            'weights': weights,
            'expected_return': portfolio_ret,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio
        }
    
    def minimum_variance_optimization(self) -> Dict[str, Union[np.ndarray, float]]:
        """
        Otimização de Mínima Variância.
        
        Returns:
            Dicionário com pesos ótimos de mínima variância
        """
        if self.covariance_matrix is None:
            raise ValueError("Matriz de covariância não calculada")
        
        n_assets = len(self.assets)
        Sigma = self.covariance_matrix
        
        # Solução analítica para mínima variância
        ones = np.ones((n_assets, 1))
        Sigma_inv = linalg.inv(Sigma)
        
        weights = np.dot(Sigma_inv, ones) / np.dot(ones.T, np.dot(Sigma_inv, ones))
        weights = weights.flatten()
        
        # Calcular métricas
        portfolio_ret = np.dot(weights, self.expected_returns.values)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
        sharpe_ratio = (portfolio_ret - self.risk_free_rate) / portfolio_vol
        
        return {
            'weights': weights,
            'expected_return': portfolio_ret,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio
        }
    
    def maximum_diversification_optimization(self) -> Dict[str, Union[np.ndarray, float]]:
        """
        Otimização de Máxima Diversificação.
        
        Returns:
            Dicionário com pesos ótimos de máxima diversificação
        """
        if self.covariance_matrix is None:
            raise ValueError("Matriz de covariância não calculada")
        
        n_assets = len(self.assets)
        Sigma = self.covariance_matrix
        volatilities = np.sqrt(np.diag(Sigma))
        
        def diversification_ratio(weights):
            """
            Calcula o ratio de diversificação.
            DR = (w^T * sigma) / sqrt(w^T * Sigma * w)
            """
            weighted_vol = np.dot(weights, volatilities)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
            return -weighted_vol / portfolio_vol  # Negativo para maximização
        
        # Restrições
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = Bounds(lb=np.zeros(n_assets), ub=np.ones(n_assets))
        x0 = np.full(n_assets, 1.0 / n_assets)
        
        result = minimize(diversification_ratio, x0, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        
        if not result.success:
            warnings.warn(f"Otimização de Máxima Diversificação falhou: {result.message}")
        
        weights = result.x
        
        # Calcular métricas
        portfolio_ret = np.dot(weights, self.expected_returns.values)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
        sharpe_ratio = (portfolio_ret - self.risk_free_rate) / portfolio_vol
        
        return {
            'weights': weights,
            'expected_return': portfolio_ret,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'diversification_ratio': -diversification_ratio(weights)
        }
    
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de performance do portfólio.
        
        Args:
            weights: Pesos do portfólio
            
        Returns:
            Dicionário com métricas
        """
        if self.returns_data is None:
            raise ValueError("Dados de retornos não carregados")
        
        # Retornos do portfólio
        portfolio_returns = (self.returns_data * weights).sum(axis=1)
        
        # Métricas básicas
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_vol
        
        # Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR e CVaR
        var_95 = portfolio_returns.quantile(1 - self.confidence_level)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Calmar Ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        # Sortino Ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation != 0 else np.inf
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio
        }
    
    def plot_efficient_frontier(self, 
                              frontier_df: pd.DataFrame,
                              optimal_portfolios: Dict[str, Dict] = None) -> plt.Figure:
        """
        Plota a fronteira eficiente.
        
        Args:
            frontier_df: DataFrame com pontos da fronteira
            optimal_portfolios: Dicionário com portfólios ótimos para destacar
            
        Returns:
            Figura matplotlib
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plotar fronteira eficiente
        ax.plot(frontier_df['volatility'], frontier_df['return'], 
               'b-', linewidth=2, label='Fronteira Eficiente')
        
        # Plotar ativos individuais
        individual_vols = np.sqrt(np.diag(self.covariance_matrix))
        individual_rets = self.expected_returns.values
        
        ax.scatter(individual_vols, individual_rets, 
                  c='red', s=50, alpha=0.7, label='Ativos Individuais')
        
        # Adicionar labels dos ativos
        for i, asset in enumerate(self.assets):
            ax.annotate(asset, (individual_vols[i], individual_rets[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plotar portfólios ótimos se fornecidos
        if optimal_portfolios:
            colors = ['green', 'orange', 'purple', 'brown']
            for i, (name, portfolio) in enumerate(optimal_portfolios.items()):
                ax.scatter(portfolio['volatility'], portfolio['expected_return'],
                          c=colors[i % len(colors)], s=100, marker='*',
                          label=f'{name}', zorder=5)
        
        ax.set_xlabel('Volatilidade (Risco)')
        ax.set_ylabel('Retorno Esperado')
        ax.set_title('Fronteira Eficiente de Markowitz')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def compare_optimization_methods(self, 
                                   views: Optional[Dict[str, float]] = None,
                                   view_confidences: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Compara diferentes métodos de otimização.
        
        Args:
            views: Views para Black-Litterman
            view_confidences: Confiança nas views
            
        Returns:
            DataFrame comparativo
        """
        results = {}
        
        # Mean-Variance (Max Sharpe)
        try:
            mv_result = self.mean_variance_optimization()
            results['Mean-Variance'] = mv_result
        except Exception as e:
            warnings.warn(f"Erro em Mean-Variance: {e}")
        
        # Minimum Variance
        try:
            min_var_result = self.minimum_variance_optimization()
            results['Min Variance'] = min_var_result
        except Exception as e:
            warnings.warn(f"Erro em Min Variance: {e}")
        
        # Risk Parity
        try:
            rp_result = self.risk_parity_optimization()
            results['Risk Parity'] = rp_result
        except Exception as e:
            warnings.warn(f"Erro em Risk Parity: {e}")
        
        # Maximum Diversification
        try:
            max_div_result = self.maximum_diversification_optimization()
            results['Max Diversification'] = max_div_result
        except Exception as e:
            warnings.warn(f"Erro em Max Diversification: {e}")
        
        # Black-Litterman (se views fornecidas)
        if views and view_confidences:
            try:
                bl_result = self.black_litterman_optimization(views, view_confidences)
                results['Black-Litterman'] = bl_result
            except Exception as e:
                warnings.warn(f"Erro em Black-Litterman: {e}")
        
        # Criar DataFrame comparativo
        comparison_data = []
        for method, result in results.items():
            comparison_data.append({
                'Method': method,
                'Expected Return': result['expected_return'],
                'Volatility': result['volatility'],
                'Sharpe Ratio': result['sharpe_ratio']
            })
        
        return pd.DataFrame(comparison_data)


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados de exemplo
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    
    # Simular retornos de 5 ativos
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    n_assets = len(assets)
    
    # Matriz de correlação
    corr_matrix = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Volatilidades individuais
    volatilities = np.random.uniform(0.15, 0.35, n_assets)
    
    # Gerar retornos correlacionados
    returns_data = []
    for _ in range(252):
        random_returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=corr_matrix
        )
        scaled_returns = random_returns * volatilities / 252**0.5
        returns_data.append(scaled_returns)
    
    returns_df = pd.DataFrame(returns_data, columns=assets, index=dates)
    
    # Criar otimizador
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    optimizer.load_data(returns_df)
    
    # Comparar métodos
    comparison = optimizer.compare_optimization_methods()
    print("Comparação de Métodos de Otimização:")
    print(comparison)
    
    # Calcular fronteira eficiente
    frontier = optimizer.efficient_frontier(n_points=50)
    print(f"\nFronteira eficiente calculada com {len(frontier)} pontos")