#!/usr/bin/env python3
"""
Sistema de Gestão de Risco Avançado para Trading Algorítmico
Baseado nos conhecimentos extraídos do livro "Machine Learning for Algorithmic Trading"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RiskMetrics:
    """Métricas de risco calculadas"""
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    cvar_95: float  # Conditional VaR 95%
    cvar_99: float  # Conditional VaR 99%
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    kelly_criterion: float
    correlation_risk: float

@dataclass
class PositionSizing:
    """Resultado do cálculo de position sizing"""
    recommended_size: float
    max_risk_size: float
    kelly_size: float
    volatility_adjusted_size: float
    correlation_adjusted_size: float
    final_size: float
    risk_pct: float

class AdvancedRiskManager:
    """
    Sistema avançado de gestão de risco
    """
    
    def __init__(self,
                 max_portfolio_risk: float = 0.02,  # 2% do portfolio por dia
                 max_position_risk: float = 0.01,   # 1% por posição
                 max_correlation: float = 0.7,      # Correlação máxima entre posições
                 lookback_period: int = 252,        # Período para cálculos (1 ano)
                 confidence_levels: List[float] = [0.95, 0.99]):
        
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_correlation = max_correlation
        self.lookback_period = lookback_period
        self.confidence_levels = confidence_levels
        
        # Estado do portfolio
        self.positions = {}
        self.historical_returns = {}
        self.correlation_matrix = None
        
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calcula Value at Risk usando método histórico
        """
        if len(returns) == 0:
            return 0.0
        
        # Remover NaN
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return 0.0
        
        # VaR histórico
        var = np.percentile(clean_returns, (1 - confidence_level) * 100)
        
        return abs(var)  # Retornar como valor positivo
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calcula Conditional Value at Risk (Expected Shortfall)
        """
        if len(returns) == 0:
            return 0.0
        
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return 0.0
        
        # VaR threshold
        var_threshold = np.percentile(clean_returns, (1 - confidence_level) * 100)
        
        # CVaR = média dos retornos piores que o VaR
        tail_returns = clean_returns[clean_returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return abs(var_threshold)
        
        cvar = tail_returns.mean()
        
        return abs(cvar)
    
    def calculate_maximum_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calcula drawdown máximo
        """
        if len(equity_curve) == 0:
            return 0.0
        
        # Calcular running maximum
        running_max = equity_curve.expanding().max()
        
        # Calcular drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        return abs(drawdown.min())
    
    def calculate_kelly_criterion(self, returns: pd.Series) -> float:
        """
        Calcula Kelly Criterion para otimização de position sizing
        """
        if len(returns) == 0:
            return 0.0
        
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0 or clean_returns.var() == 0:
            return 0.0
        
        # Kelly = (média dos retornos) / (variância dos retornos)
        kelly = clean_returns.mean() / clean_returns.var()
        
        # Limitar Kelly para evitar over-leverage
        return max(0, min(kelly, 0.25))  # Máximo 25%
    
    def calculate_correlation_matrix(self, returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calcula matriz de correlação entre ativos
        """
        if not returns_dict:
            return pd.DataFrame()
        
        # Criar DataFrame com todos os retornos
        returns_df = pd.DataFrame(returns_dict)
        
        # Calcular correlação
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def assess_correlation_risk(self, asset: str, returns_dict: Dict[str, pd.Series]) -> float:
        """
        Avalia risco de correlação para um ativo específico
        """
        if asset not in returns_dict or len(returns_dict) <= 1:
            return 0.0
        
        # Calcular matriz de correlação
        corr_matrix = self.calculate_correlation_matrix(returns_dict)
        
        if asset not in corr_matrix.columns:
            return 0.0
        
        # Correlações do ativo com outros (excluindo autocorrelação)
        asset_correlations = corr_matrix[asset].drop(asset)
        
        if len(asset_correlations) == 0:
            return 0.0
        
        # Risco de correlação = média das correlações absolutas
        correlation_risk = asset_correlations.abs().mean()
        
        return correlation_risk
    
    def calculate_volatility_adjusted_size(self, 
                                         returns: pd.Series, 
                                         target_volatility: float = 0.15) -> float:
        """
        Calcula position size ajustado por volatilidade
        """
        if len(returns) == 0:
            return 0.0
        
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return 0.0
        
        # Volatilidade anualizada
        current_volatility = clean_returns.std() * np.sqrt(252)
        
        if current_volatility == 0:
            return 0.0
        
        # Ajuste de volatilidade
        vol_adjustment = target_volatility / current_volatility
        
        return min(vol_adjustment, 2.0)  # Máximo 2x leverage
    
    def calculate_comprehensive_risk_metrics(self, 
                                           returns: pd.Series, 
                                           equity_curve: pd.Series = None) -> RiskMetrics:
        """
        Calcula métricas de risco abrangentes
        """
        print("Calculando métricas de risco abrangentes...")
        
        if len(returns) == 0:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        clean_returns = returns.dropna()
        
        # VaR e CVaR
        var_95 = self.calculate_var(clean_returns, 0.95)
        var_99 = self.calculate_var(clean_returns, 0.99)
        cvar_95 = self.calculate_cvar(clean_returns, 0.95)
        cvar_99 = self.calculate_cvar(clean_returns, 0.99)
        
        # Drawdown máximo
        if equity_curve is not None and len(equity_curve) > 0:
            max_dd = self.calculate_maximum_drawdown(equity_curve)
        else:
            # Estimar drawdown a partir dos retornos
            cumulative_returns = (1 + clean_returns).cumprod()
            max_dd = self.calculate_maximum_drawdown(cumulative_returns)
        
        # Volatilidade anualizada
        volatility = clean_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio (assumindo risk-free rate = 2%)
        risk_free_rate = 0.02
        excess_returns = clean_returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
        
        # Sortino Ratio
        negative_returns = clean_returns[clean_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else volatility
        sortino_ratio = excess_returns / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar Ratio
        annual_return = clean_returns.mean() * 252
        calmar_ratio = annual_return / max_dd if max_dd > 0 else 0
        
        # Kelly Criterion
        kelly = self.calculate_kelly_criterion(clean_returns)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_dd,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            kelly_criterion=kelly,
            correlation_risk=0.0  # Será calculado separadamente
        )
    
    def calculate_optimal_position_size(self, 
                                      asset: str,
                                      returns: pd.Series,
                                      current_price: float,
                                      stop_loss_price: float,
                                      portfolio_value: float,
                                      other_returns: Dict[str, pd.Series] = None) -> PositionSizing:
        """
        Calcula position size ótimo usando múltiplos métodos
        """
        print(f"Calculando position size ótimo para {asset}...")
        
        if len(returns) == 0 or stop_loss_price == current_price:
            return PositionSizing(0, 0, 0, 0, 0, 0, 0)
        
        # 1. Risk-based sizing (baseado no stop loss)
        risk_per_unit = abs(current_price - stop_loss_price) / current_price
        max_risk_amount = portfolio_value * self.max_position_risk
        max_risk_size = max_risk_amount / (risk_per_unit * current_price) if risk_per_unit > 0 else 0
        
        # 2. Kelly Criterion sizing
        kelly_fraction = self.calculate_kelly_criterion(returns)
        kelly_size = (portfolio_value * kelly_fraction) / current_price
        
        # 3. Volatility-adjusted sizing
        vol_adjustment = self.calculate_volatility_adjusted_size(returns)
        base_size = portfolio_value * 0.1 / current_price  # 10% base allocation
        volatility_adjusted_size = base_size * vol_adjustment
        
        # 4. Correlation-adjusted sizing
        correlation_risk = 0.0
        correlation_adjusted_size = volatility_adjusted_size
        
        if other_returns:
            all_returns = {asset: returns, **other_returns}
            correlation_risk = self.assess_correlation_risk(asset, all_returns)
            
            # Reduzir size se correlação for alta
            correlation_adjustment = max(0.5, 1 - correlation_risk)
            correlation_adjusted_size = volatility_adjusted_size * correlation_adjustment
        
        # 5. Size final (mínimo entre todos os métodos)
        recommended_size = min(
            max_risk_size,
            kelly_size,
            volatility_adjusted_size,
            correlation_adjusted_size
        )
        
        # Garantir que não exceda limites
        max_portfolio_allocation = portfolio_value * 0.2 / current_price  # Máximo 20% do portfolio
        final_size = min(recommended_size, max_portfolio_allocation)
        
        # Calcular risco percentual
        risk_pct = (final_size * risk_per_unit) / portfolio_value if portfolio_value > 0 else 0
        
        return PositionSizing(
            recommended_size=recommended_size,
            max_risk_size=max_risk_size,
            kelly_size=kelly_size,
            volatility_adjusted_size=volatility_adjusted_size,
            correlation_adjusted_size=correlation_adjusted_size,
            final_size=final_size,
            risk_pct=risk_pct
        )
    
    def assess_portfolio_risk(self, 
                            positions: Dict[str, Dict],
                            returns_dict: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Avalia risco total do portfolio
        """
        print("Avaliando risco do portfolio...")
        
        if not positions or not returns_dict:
            return {'total_var': 0, 'concentration_risk': 0, 'correlation_risk': 0}
        
        # Calcular matriz de correlação
        corr_matrix = self.calculate_correlation_matrix(returns_dict)
        
        # Calcular pesos das posições
        total_value = sum(pos['value'] for pos in positions.values())
        weights = {asset: pos['value'] / total_value for asset, pos in positions.items()}
        
        # VaR do portfolio (considerando correlações)
        portfolio_var = 0
        if len(corr_matrix) > 0:
            # Calcular VaR individual de cada posição
            individual_vars = {}
            for asset, returns in returns_dict.items():
                if asset in weights:
                    var_95 = self.calculate_var(returns, 0.95)
                    individual_vars[asset] = var_95 * weights[asset]
            
            # VaR do portfolio considerando correlações
            portfolio_var = 0
            for asset1 in individual_vars:
                for asset2 in individual_vars:
                    if asset1 in corr_matrix.index and asset2 in corr_matrix.columns:
                        correlation = corr_matrix.loc[asset1, asset2]
                        portfolio_var += individual_vars[asset1] * individual_vars[asset2] * correlation
            
            portfolio_var = np.sqrt(max(0, portfolio_var))
        
        # Risco de concentração (Herfindahl Index)
        concentration_risk = sum(w**2 for w in weights.values())
        
        # Risco de correlação média
        avg_correlation = 0
        if len(corr_matrix) > 1:
            # Correlação média (excluindo diagonal)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            avg_correlation = corr_matrix.values[mask].mean()
        
        return {
            'total_var': portfolio_var,
            'concentration_risk': concentration_risk,
            'correlation_risk': avg_correlation,
            'diversification_ratio': 1 - avg_correlation
        }
    
    def generate_risk_report(self, 
                           returns_dict: Dict[str, pd.Series],
                           positions: Dict[str, Dict] = None) -> str:
        """
        Gera relatório completo de risco
        """
        report = "\n=== RELATÓRIO DE RISCO AVANÇADO ===\n"
        
        # Métricas individuais por ativo
        report += "\n1. MÉTRICAS POR ATIVO:\n"
        report += "-" * 50 + "\n"
        
        for asset, returns in returns_dict.items():
            metrics = self.calculate_comprehensive_risk_metrics(returns)
            
            report += f"\n{asset}:\n"
            report += f"  VaR 95%: {metrics.var_95:.4f} ({metrics.var_95*100:.2f}%)\n"
            report += f"  VaR 99%: {metrics.var_99:.4f} ({metrics.var_99*100:.2f}%)\n"
            report += f"  CVaR 95%: {metrics.cvar_95:.4f} ({metrics.cvar_95*100:.2f}%)\n"
            report += f"  Volatilidade: {metrics.volatility:.4f} ({metrics.volatility*100:.2f}%)\n"
            report += f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}\n"
            report += f"  Sortino Ratio: {metrics.sortino_ratio:.2f}\n"
            report += f"  Kelly Criterion: {metrics.kelly_criterion:.4f} ({metrics.kelly_criterion*100:.2f}%)\n"
        
        # Matriz de correlação
        if len(returns_dict) > 1:
            report += "\n2. MATRIZ DE CORRELAÇÃO:\n"
            report += "-" * 50 + "\n"
            
            corr_matrix = self.calculate_correlation_matrix(returns_dict)
            report += corr_matrix.round(3).to_string()
        
        # Risco do portfolio
        if positions:
            report += "\n\n3. RISCO DO PORTFOLIO:\n"
            report += "-" * 50 + "\n"
            
            portfolio_risk = self.assess_portfolio_risk(positions, returns_dict)
            
            report += f"VaR Total do Portfolio: {portfolio_risk['total_var']:.4f}\n"
            report += f"Risco de Concentração: {portfolio_risk['concentration_risk']:.4f}\n"
            report += f"Correlação Média: {portfolio_risk['correlation_risk']:.4f}\n"
            report += f"Ratio de Diversificação: {portfolio_risk['diversification_ratio']:.4f}\n"
        
        # Recomendações
        report += "\n\n4. RECOMENDAÇÕES:\n"
        report += "-" * 50 + "\n"
        
        if len(returns_dict) > 1:
            corr_matrix = self.calculate_correlation_matrix(returns_dict)
            high_corr_pairs = []
            
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > self.max_correlation:
                        asset1 = corr_matrix.index[i]
                        asset2 = corr_matrix.columns[j]
                        high_corr_pairs.append((asset1, asset2, corr_value))
            
            if high_corr_pairs:
                report += "\nALERTA - Correlações Altas Detectadas:\n"
                for asset1, asset2, corr in high_corr_pairs:
                    report += f"  {asset1} x {asset2}: {corr:.3f}\n"
                report += "  Recomendação: Reduzir exposição ou diversificar\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report

def demonstrate_risk_management():
    """
    Demonstração do sistema de gestão de risco
    """
    print("=== DEMONSTRAÇÃO - GESTÃO DE RISCO AVANÇADA ===")
    
    # Criar dados sintéticos para múltiplos ativos
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Simular retornos correlacionados
    n_assets = 4
    correlation_matrix = np.array([
        [1.0, 0.3, 0.1, -0.2],
        [0.3, 1.0, 0.5, 0.1],
        [0.1, 0.5, 1.0, 0.2],
        [-0.2, 0.1, 0.2, 1.0]
    ])
    
    # Gerar retornos correlacionados
    mean_returns = [0.0005, 0.0003, 0.0007, 0.0002]  # Retornos médios diários
    volatilities = [0.015, 0.012, 0.020, 0.010]      # Volatilidades diárias
    
    # Decomposição de Cholesky para gerar correlações
    L = np.linalg.cholesky(correlation_matrix)
    
    returns_data = {}
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    
    for i, asset in enumerate(assets):
        # Gerar retornos independentes
        independent_returns = np.random.normal(mean_returns[i], volatilities[i], len(dates))
        
        # Aplicar correlação
        if i == 0:
            correlated_returns = independent_returns
        else:
            # Combinar com retornos anteriores para criar correlação
            base_returns = np.random.normal(0, volatilities[i], len(dates))
            correlated_returns = mean_returns[i] + L[i, :i+1] @ np.vstack([returns_data[assets[j]] - mean_returns[j] for j in range(i)] + [base_returns]).T
            correlated_returns = correlated_returns.flatten()
        
        returns_data[asset] = pd.Series(correlated_returns, index=dates)
    
    # Criar risk manager
    risk_manager = AdvancedRiskManager(
        max_portfolio_risk=0.02,
        max_position_risk=0.01,
        max_correlation=0.6
    )
    
    # Calcular métricas de risco
    print("\nCalculando métricas de risco...")
    
    # Simular posições do portfolio
    positions = {
        'EURUSD': {'value': 50000, 'size': 45000},
        'GBPUSD': {'value': 30000, 'size': 24000},
        'USDJPY': {'value': 20000, 'size': 18000}
    }
    
    # Gerar relatório de risco
    risk_report = risk_manager.generate_risk_report(returns_data, positions)
    print(risk_report)
    
    # Demonstrar position sizing
    print("\n=== DEMONSTRAÇÃO - POSITION SIZING ===")
    
    current_price = 1.1000
    stop_loss_price = 1.0950
    portfolio_value = 100000
    
    position_sizing = risk_manager.calculate_optimal_position_size(
        asset='EURUSD',
        returns=returns_data['EURUSD'],
        current_price=current_price,
        stop_loss_price=stop_loss_price,
        portfolio_value=portfolio_value,
        other_returns={k: v for k, v in returns_data.items() if k != 'EURUSD'}
    )
    
    print(f"\nPosition Sizing para EURUSD:")
    print(f"Preço Atual: ${current_price:.4f}")
    print(f"Stop Loss: ${stop_loss_price:.4f}")
    print(f"Portfolio Value: ${portfolio_value:,.2f}")
    print(f"\nResultados:")
    print(f"Size Recomendado: {position_sizing.recommended_size:,.0f} unidades")
    print(f"Size Máximo (Risco): {position_sizing.max_risk_size:,.0f} unidades")
    print(f"Size Kelly: {position_sizing.kelly_size:,.0f} unidades")
    print(f"Size Ajustado (Vol): {position_sizing.volatility_adjusted_size:,.0f} unidades")
    print(f"Size Final: {position_sizing.final_size:,.0f} unidades")
    print(f"Risco da Posição: {position_sizing.risk_pct:.2%}")
    
    return risk_manager, returns_data, positions

if __name__ == "__main__":
    demonstrate_risk_management()