import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Union
import logging
from scipy.optimize import minimize_scalar
from dataclasses import dataclass

@dataclass
class TradingStats:
    """Estrutura para armazenar estatísticas de trading"""
    win_rate: float
    avg_win: float
    avg_loss: float
    total_trades: int
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float

class KellyPositionSizing:
    """
    Implementação do Kelly Criterion para position sizing em trading
    Baseado nos princípios de "Advances in Financial Machine Learning"
    
    O Kelly Criterion otimiza o tamanho da posição para maximizar
    o crescimento logarítmico do capital a longo prazo.
    """
    
    def __init__(self, max_leverage: float = 1.0, 
                 max_kelly_fraction: float = 0.25,
                 min_kelly_fraction: float = 0.01,
                 lookback_period: int = 252,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            max_leverage: Alavancagem máxima permitida
            max_kelly_fraction: Fração máxima do Kelly (cap de segurança)
            min_kelly_fraction: Fração mínima do Kelly
            lookback_period: Período para calcular estatísticas
            logger: Logger para debug
        """
        self.max_leverage = max_leverage
        self.max_kelly_fraction = max_kelly_fraction
        self.min_kelly_fraction = min_kelly_fraction
        self.lookback_period = lookback_period
        self.logger = logger or logging.getLogger(__name__)
        
    def calculate_kelly_fraction(self, win_prob: Optional[float] = None, 
                               avg_win: Optional[float] = None, 
                               avg_loss: Optional[float] = None, 
                               method: str = 'basic',
                               win_rate: Optional[float] = None,
                               confidence: float = 1.0) -> float:
        """
        Calcula a fração Kelly
        
        Args:
            win_prob: Probabilidade de ganho (compatibilidade)
            avg_win: Ganho médio
            avg_loss: Perda média
            method: Método de cálculo ('basic', 'modified', 'optimal')
            win_rate: Taxa de vitória (compatibilidade)
            confidence: Fator de confiança
            
        Returns:
            Fração Kelly calculada
        """
        # Compatibilidade com diferentes nomes de parâmetros
        if win_prob is not None:
            win_rate = win_prob
        
        if win_rate is None or avg_win is None or avg_loss is None:
            raise ValueError("Parâmetros win_rate, avg_win e avg_loss são obrigatórios")
        
        if method == 'basic':
            return self._calculate_basic_kelly(win_rate, avg_win, avg_loss, confidence)
        elif method == 'modified':
            return self._calculate_modified_kelly(win_rate, avg_win, avg_loss, confidence)
        elif method == 'optimal':
            return self._calculate_optimal_kelly(win_rate, avg_win, avg_loss, confidence)
        else:
            raise ValueError(f"Método desconhecido: {method}")
    
    def _calculate_basic_kelly(self, win_rate: float, avg_win: float, 
                              avg_loss: float, confidence: float = 1.0) -> float:
        """
        Calcula a fração Kelly básica
        
        Formula: f* = (bp - q) / b
        onde:
        - b = avg_win / avg_loss (payoff ratio)
        - p = win_rate (probabilidade de ganho)
        - q = 1 - p (probabilidade de perda)
        
        Args:
            win_rate: Taxa de acerto (0-1)
            avg_win: Ganho médio por trade vencedor
            avg_loss: Perda média por trade perdedor (valor positivo)
            confidence: Fator de confiança na predição (0-1)
            
        Returns:
            Fração Kelly calculada
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
            
        # Payoff ratio
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        # Kelly fraction
        kelly_f = (b * p - q) / b
        
        # Ajustar por confiança
        kelly_f *= confidence
        
        # Aplicar limites de segurança
        kelly_f = max(self.min_kelly_fraction, min(kelly_f, self.max_kelly_fraction))
        
        self.logger.debug(f"Kelly calculation: win_rate={win_rate:.3f}, b={b:.3f}, kelly_f={kelly_f:.3f}")
        
        return kelly_f
    
    def calculate_modified_kelly(self, returns: pd.Series, 
                               confidence: float = 1.0) -> float:
        """
        Calcula Kelly modificado usando distribuição empírica de retornos
        
        Args:
            returns: Série de retornos históricos
            confidence: Fator de confiança
            
        Returns:
            Fração Kelly modificada
        """
        if len(returns) < 10:
            return self.min_kelly_fraction
            
        # Separar ganhos e perdas
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return self.min_kelly_fraction
            
        # Calcular estatísticas
        win_rate = len(wins) / len(returns)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        return self.calculate_kelly_fraction(win_rate, avg_win, avg_loss, confidence)
    
    def calculate_optimal_kelly(self, returns: pd.Series, 
                              method: str = 'empirical') -> float:
        """
        Calcula Kelly ótimo usando diferentes métodos
        
        Args:
            returns: Série de retornos
            method: 'empirical', 'parametric', ou 'robust'
            
        Returns:
            Fração Kelly ótima
        """
        if method == 'empirical':
            return self._kelly_empirical(returns)
        elif method == 'parametric':
            return self._kelly_parametric(returns)
        elif method == 'robust':
            return self._kelly_robust(returns)
        else:
            raise ValueError(f"Método não reconhecido: {method}")
    
    def _kelly_empirical(self, returns: pd.Series) -> float:
        """
        Kelly empírico baseado na distribuição histórica
        """
        def objective(f):
            if f <= 0:
                return -np.inf
            # Crescimento logarítmico esperado
            log_returns = np.log(1 + f * returns)
            return -log_returns.mean()  # Negativo para maximização
        
        # Otimizar fração Kelly
        result = minimize_scalar(objective, bounds=(0, self.max_kelly_fraction), 
                               method='bounded')
        
        return max(self.min_kelly_fraction, result.x)
    
    def _kelly_parametric(self, returns: pd.Series) -> float:
        """
        Kelly paramétrico assumindo distribuição normal
        """
        mu = returns.mean()
        sigma = returns.std()
        
        if sigma <= 0:
            return self.min_kelly_fraction
            
        # Kelly para distribuição normal: f* = μ/σ²
        kelly_f = mu / (sigma ** 2)
        
        return max(self.min_kelly_fraction, min(kelly_f, self.max_kelly_fraction))
    
    def _kelly_robust(self, returns: pd.Series) -> float:
        """
        Kelly robusto usando estatísticas robustas
        """
        # Usar mediana e MAD (Median Absolute Deviation)
        median_return = returns.median()
        mad = np.median(np.abs(returns - median_return))
        
        if mad <= 0:
            return self.min_kelly_fraction
            
        # Aproximação robusta
        kelly_f = median_return / (1.4826 * mad) ** 2  # 1.4826 é fator de conversão MAD->std
        
        return max(self.min_kelly_fraction, min(kelly_f, self.max_kelly_fraction))
    
    def calculate_position_size(self, kelly_fraction: float,
                              capital: float,
                              price: float,
                              max_position_pct: float = 1.0,
                              account_balance: Optional[float] = None, 
                              entry_price: Optional[float] = None,
                              stop_loss_price: Optional[float] = None,
                              symbol_info: Optional[Dict] = None) -> Dict:
        """
        Calcula tamanho da posição baseado no Kelly Criterion
        
        Args:
            kelly_fraction: Fração Kelly calculada
            capital: Capital disponível
            price: Preço atual do ativo
            max_position_pct: Percentual máximo da posição
            account_balance: Saldo da conta (compatibilidade)
            entry_price: Preço de entrada (compatibilidade)
            stop_loss_price: Preço do stop loss (compatibilidade)
            symbol_info: Informações do símbolo (tick_size, tick_value, etc.)
            
        Returns:
            Dicionário com informações da posição
        """
        # Usar parâmetros principais ou fallback para compatibilidade
        account_balance = account_balance or capital
        entry_price = entry_price or price
        
        # Calcular tamanho básico da posição baseado no Kelly
        kelly_capital = capital * kelly_fraction
        basic_position_size = kelly_capital / price
        
        # Aplicar limite de posição máxima
        max_position_size = (capital * max_position_pct) / price
        position_size = min(basic_position_size, max_position_size)
        
        # Se temos stop loss, calcular risco específico
        if stop_loss_price is not None:
            risk_per_unit = abs(entry_price - stop_loss_price)
            if risk_per_unit > 0:
                risk_capital = account_balance * kelly_fraction
                risk_based_size = risk_capital / risk_per_unit
                position_size = min(position_size, risk_based_size)
        
        # Aplicar alavancagem máxima
        max_leveraged_value = account_balance * self.max_leverage
        max_leveraged_size = max_leveraged_value / entry_price
        position_size = min(position_size, max_leveraged_size)
        
        # Calcular valores finais
        position_value = position_size * entry_price
        actual_risk = position_size * (abs(entry_price - stop_loss_price) if stop_loss_price else entry_price * 0.02)  # 2% default risk
        
        return {
            'position_size': position_size,
            'position_value': position_value,
            'risk_amount': actual_risk,
            'risk_percentage': actual_risk / account_balance * 100 if account_balance > 0 else 0,
            'kelly_fraction': kelly_fraction,
            'leverage_used': position_value / account_balance if account_balance > 0 else 0
        }
    
    def analyze_trading_performance(self, trades_df: pd.DataFrame) -> TradingStats:
        """
        Analisa performance histórica de trades
        
        Args:
            trades_df: DataFrame com colunas ['return', 'pnl', 'timestamp']
            
        Returns:
            TradingStats com estatísticas calculadas
        """
        if len(trades_df) == 0:
            return TradingStats(0, 0, 0, 0, 0, 0, 0)
            
        # Separar ganhos e perdas
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] < 0]
        
        # Calcular estatísticas básicas
        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0
        
        # Profit factor
        total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
        profit_factor = total_wins / total_losses
        
        # Maximum drawdown
        cumulative_pnl = trades_df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / running_max.abs()
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Sharpe ratio
        returns = trades_df['return'] if 'return' in trades_df.columns else trades_df['pnl']
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        return TradingStats(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=len(trades_df),
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio
        )
    
    def dynamic_kelly_adjustment(self, recent_performance: TradingStats,
                               base_kelly: float,
                               adjustment_factor: float = 0.5) -> float:
        """
        Ajusta Kelly dinamicamente baseado na performance recente
        
        Args:
            recent_performance: Estatísticas de performance recente
            base_kelly: Fração Kelly base
            adjustment_factor: Fator de ajuste (0-1)
            
        Returns:
            Kelly ajustado
        """
        # Fatores de ajuste
        performance_multiplier = 1.0
        
        # Ajustar por win rate
        if recent_performance.win_rate > 0.6:
            performance_multiplier *= 1.1
        elif recent_performance.win_rate < 0.4:
            performance_multiplier *= 0.8
            
        # Ajustar por profit factor
        if recent_performance.profit_factor > 1.5:
            performance_multiplier *= 1.1
        elif recent_performance.profit_factor < 1.2:
            performance_multiplier *= 0.8
            
        # Ajustar por drawdown
        if recent_performance.max_drawdown > 0.15:
            performance_multiplier *= 0.7
        elif recent_performance.max_drawdown < 0.05:
            performance_multiplier *= 1.1
            
        # Aplicar ajuste gradual
        adjusted_kelly = base_kelly * (1 + (performance_multiplier - 1) * adjustment_factor)
        
        # Aplicar limites
        adjusted_kelly = max(self.min_kelly_fraction, 
                           min(adjusted_kelly, self.max_kelly_fraction))
        
        self.logger.info(f"Kelly adjustment: base={base_kelly:.3f}, adjusted={adjusted_kelly:.3f}, multiplier={performance_multiplier:.3f}")
        
        return adjusted_kelly
    
    def backtest_kelly_strategy(self, price_data: pd.DataFrame, 
                              signals: pd.Series,
                              initial_capital: float = 10000) -> pd.DataFrame:
        """
        Backtesta estratégia usando Kelly Criterion
        
        Args:
            price_data: DataFrame com preços OHLC
            signals: Série com sinais de entrada (1=long, -1=short, 0=neutro)
            initial_capital: Capital inicial
            
        Returns:
            DataFrame com resultados do backtest
        """
        results = []
        capital = initial_capital
        position = 0
        entry_price = 0
        
        for i, (timestamp, signal) in enumerate(signals.items()):
            if timestamp not in price_data.index:
                continue
                
            current_price = price_data.loc[timestamp, 'close']
            
            # Fechar posição existente
            if position != 0:
                pnl = position * (current_price - entry_price)
                capital += pnl
                
                results.append({
                    'timestamp': timestamp,
                    'action': 'close',
                    'price': current_price,
                    'position': position,
                    'pnl': pnl,
                    'capital': capital,
                    'return': pnl / (abs(position) * entry_price)
                })
                
                position = 0
                
            # Abrir nova posição
            if signal != 0 and len(results) >= 10:  # Aguardar histórico mínimo
                # Calcular Kelly baseado em trades recentes
                recent_trades = pd.DataFrame(results[-20:])  # Últimos 20 trades
                if len(recent_trades) > 0:
                    kelly_f = self.calculate_modified_kelly(recent_trades['return'])
                    
                    # Calcular tamanho da posição
                    # Assumir stop loss de 2%
                    stop_loss_price = current_price * (0.98 if signal > 0 else 1.02)
                    
                    pos_info = self.calculate_position_size(
                        capital, kelly_f, current_price, stop_loss_price
                    )
                    
                    position = pos_info['position_size'] * signal
                    entry_price = current_price
                    
                    results.append({
                        'timestamp': timestamp,
                        'action': 'open',
                        'price': current_price,
                        'position': position,
                        'pnl': 0,
                        'capital': capital,
                        'kelly_fraction': kelly_f,
                        'risk_percentage': pos_info['risk_percentage']
                    })
                    
        return pd.DataFrame(results)

# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Criar instância
    kelly = KellyPositionSizing(logger=logger)
    
    # Dados de exemplo
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 1000))  # Retornos diários
    
    # Calcular Kelly
    kelly_fraction = kelly.calculate_modified_kelly(returns)
    print(f"Kelly Fraction: {kelly_fraction:.4f}")
    
    # Calcular tamanho da posição
    position_info = kelly.calculate_position_size(
        account_balance=10000,
        kelly_fraction=kelly_fraction,
        entry_price=1.1000,
        stop_loss_price=1.0950,
        symbol_info={'tick_size': 0.00001, 'tick_value': 1.0}
    )
    
    print("\nPosition Sizing:")
    for key, value in position_info.items():
        print(f"{key}: {value}")
    
    # Simular trades para análise
    trades_data = {
        'pnl': np.random.normal(10, 50, 100),
        'return': np.random.normal(0.001, 0.02, 100),
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='D')
    }
    trades_df = pd.DataFrame(trades_data)
    
    # Analisar performance
    stats = kelly.analyze_trading_performance(trades_df)
    print(f"\nTrading Stats:")
    print(f"Win Rate: {stats.win_rate:.3f}")
    print(f"Profit Factor: {stats.profit_factor:.3f}")
    print(f"Sharpe Ratio: {stats.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {stats.max_drawdown:.3f}")