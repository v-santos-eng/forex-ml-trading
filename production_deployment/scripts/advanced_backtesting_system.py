#!/usr/bin/env python3
"""
Sistema de Backtesting Avançado para Trading Algorítmico
Baseado nos conhecimentos extraídos do livro "Machine Learning for Algorithmic Trading"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Trade:
    """Representa uma operação individual"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size: float
    direction: str  # 'long' ou 'short'
    pnl: float
    pnl_pct: float
    duration: timedelta
    max_adverse_excursion: float  # MAE
    max_favorable_excursion: float  # MFE

@dataclass
class BacktestResults:
    """Resultados completos do backtesting"""
    trades: List[Trade]
    equity_curve: pd.Series
    returns: pd.Series
    metrics: Dict[str, float]
    drawdown_series: pd.Series
    monthly_returns: pd.DataFrame
    
class AdvancedBacktester:
    """
    Sistema de backtesting avançado com métricas robustas
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission: float = 0.0001,  # 1 pip spread
                 slippage: float = 0.0001,    # 1 pip slippage
                 max_position_size: float = 0.02,  # 2% do capital por trade
                 risk_free_rate: float = 0.02):   # 2% ao ano
        
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate
        
        # Estado do backtesting
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
        
    def calculate_position_size(self, 
                              price: float, 
                              stop_loss: float, 
                              risk_pct: float = 0.01) -> float:
        """
        Calcula tamanho da posição baseado em gestão de risco
        """
        # Risk per trade (1% do capital)
        risk_amount = self.current_capital * risk_pct
        
        # Distance to stop loss
        stop_distance = abs(price - stop_loss)
        
        if stop_distance == 0:
            return 0
        
        # Position size baseado no risco
        position_size = risk_amount / stop_distance
        
        # Limitar ao máximo permitido
        max_size = self.current_capital * self.max_position_size / price
        
        return min(position_size, max_size)
    
    def enter_position(self, 
                      timestamp: datetime,
                      price: float,
                      direction: str,
                      stop_loss: float,
                      take_profit: Optional[float] = None,
                      position_id: str = "default") -> bool:
        """
        Entra em uma posição
        """
        if position_id in self.positions:
            return False  # Já existe posição
        
        # Calcular tamanho da posição
        position_size = self.calculate_position_size(price, stop_loss)
        
        if position_size <= 0:
            return False
        
        # Aplicar slippage
        execution_price = price * (1 + self.slippage if direction == 'long' else 1 - self.slippage)
        
        # Calcular custo da transação
        transaction_cost = position_size * execution_price * self.commission
        
        # Verificar se há capital suficiente
        required_capital = position_size * execution_price + transaction_cost
        if required_capital > self.current_capital:
            return False
        
        # Criar posição
        self.positions[position_id] = {
            'entry_time': timestamp,
            'entry_price': execution_price,
            'position_size': position_size,
            'direction': direction,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'max_adverse': 0,
            'max_favorable': 0
        }
        
        # Atualizar capital
        self.current_capital -= transaction_cost
        
        return True
    
    def update_position_extremes(self, timestamp: datetime, price: float):
        """
        Atualiza MAE e MFE das posições abertas
        """
        for pos_id, position in self.positions.items():
            entry_price = position['entry_price']
            direction = position['direction']
            
            if direction == 'long':
                # Para posição long
                unrealized_pnl = (price - entry_price) * position['position_size']
                adverse = min(0, unrealized_pnl)  # Negativo = adverso
                favorable = max(0, unrealized_pnl)  # Positivo = favorável
            else:
                # Para posição short
                unrealized_pnl = (entry_price - price) * position['position_size']
                adverse = min(0, unrealized_pnl)
                favorable = max(0, unrealized_pnl)
            
            # Atualizar extremos
            position['max_adverse'] = min(position['max_adverse'], adverse)
            position['max_favorable'] = max(position['max_favorable'], favorable)
    
    def exit_position(self, 
                     timestamp: datetime,
                     price: float,
                     position_id: str = "default",
                     reason: str = "manual") -> Optional[Trade]:
        """
        Sai de uma posição
        """
        if position_id not in self.positions:
            return None
        
        position = self.positions[position_id]
        
        # Aplicar slippage
        direction = position['direction']
        execution_price = price * (1 - self.slippage if direction == 'long' else 1 + self.slippage)
        
        # Calcular P&L
        if direction == 'long':
            pnl = (execution_price - position['entry_price']) * position['position_size']
        else:
            pnl = (position['entry_price'] - execution_price) * position['position_size']
        
        # Subtrair comissão de saída
        transaction_cost = position['position_size'] * execution_price * self.commission
        pnl -= transaction_cost
        
        # Calcular P&L percentual
        pnl_pct = pnl / (position['position_size'] * position['entry_price'])
        
        # Criar trade
        trade = Trade(
            entry_time=position['entry_time'],
            exit_time=timestamp,
            entry_price=position['entry_price'],
            exit_price=execution_price,
            position_size=position['position_size'],
            direction=direction,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration=timestamp - position['entry_time'],
            max_adverse_excursion=position['max_adverse'],
            max_favorable_excursion=position['max_favorable']
        )
        
        # Atualizar capital
        self.current_capital += pnl
        
        # Adicionar trade à lista
        self.trades.append(trade)
        
        # Remover posição
        del self.positions[position_id]
        
        return trade
    
    def check_stop_loss_take_profit(self, timestamp: datetime, high: float, low: float):
        """
        Verifica stop loss e take profit
        """
        positions_to_close = []
        
        for pos_id, position in self.positions.items():
            direction = position['direction']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            if direction == 'long':
                # Stop loss hit
                if stop_loss and low <= stop_loss:
                    positions_to_close.append((pos_id, stop_loss, 'stop_loss'))
                # Take profit hit
                elif take_profit and high >= take_profit:
                    positions_to_close.append((pos_id, take_profit, 'take_profit'))
            
            else:  # short
                # Stop loss hit
                if stop_loss and high >= stop_loss:
                    positions_to_close.append((pos_id, stop_loss, 'stop_loss'))
                # Take profit hit
                elif take_profit and low <= take_profit:
                    positions_to_close.append((pos_id, take_profit, 'take_profit'))
        
        # Fechar posições
        for pos_id, exit_price, reason in positions_to_close:
            self.exit_position(timestamp, exit_price, pos_id, reason)
    
    def update_equity_curve(self, timestamp: datetime, current_price: float):
        """
        Atualiza curva de equity
        """
        # Calcular P&L não realizado
        unrealized_pnl = 0
        for position in self.positions.values():
            if position['direction'] == 'long':
                unrealized_pnl += (current_price - position['entry_price']) * position['position_size']
            else:
                unrealized_pnl += (position['entry_price'] - current_price) * position['position_size']
        
        # Equity total
        total_equity = self.current_capital + unrealized_pnl
        
        self.equity_curve.append(total_equity)
        self.timestamps.append(timestamp)
    
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    signals: pd.DataFrame,
                    stop_loss_pct: float = 0.02,
                    take_profit_pct: float = 0.04) -> BacktestResults:
        """
        Executa o backtesting completo
        """
        print("=== INICIANDO BACKTESTING AVANÇADO ===")
        
        # Reset do estado
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
        
        # Alinhar dados e sinais
        common_idx = data.index.intersection(signals.index)
        data_aligned = data.loc[common_idx]
        signals_aligned = signals.loc[common_idx]
        
        print(f"Processando {len(common_idx)} períodos...")
        
        for i, (timestamp, row) in enumerate(data_aligned.iterrows()):
            if i % 1000 == 0:
                print(f"Processando período {i+1}/{len(data_aligned)}")
            
            # Preços OHLC
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # Verificar stop loss e take profit primeiro
            self.check_stop_loss_take_profit(timestamp, high_price, low_price)
            
            # Atualizar extremos das posições
            self.update_position_extremes(timestamp, close_price)
            
            # Verificar sinais
            if timestamp in signals_aligned.index:
                signal = signals_aligned.loc[timestamp]
                
                # Sinal de compra
                if signal.get('signal', 0) == 1 and 'default' not in self.positions:
                    stop_loss = close_price * (1 - stop_loss_pct)
                    take_profit = close_price * (1 + take_profit_pct)
                    self.enter_position(timestamp, close_price, 'long', stop_loss, take_profit)
                
                # Sinal de venda
                elif signal.get('signal', 0) == -1 and 'default' not in self.positions:
                    stop_loss = close_price * (1 + stop_loss_pct)
                    take_profit = close_price * (1 - take_profit_pct)
                    self.enter_position(timestamp, close_price, 'short', stop_loss, take_profit)
                
                # Sinal de saída
                elif signal.get('signal', 0) == 0 and 'default' in self.positions:
                    self.exit_position(timestamp, close_price)
            
            # Atualizar curva de equity
            self.update_equity_curve(timestamp, close_price)
        
        # Fechar posições abertas
        final_timestamp = data_aligned.index[-1]
        final_price = data_aligned.iloc[-1]['close']
        
        for pos_id in list(self.positions.keys()):
            self.exit_position(final_timestamp, final_price, pos_id, 'end_of_data')
        
        print(f"Backtesting concluído. {len(self.trades)} trades executados.")
        
        # Calcular métricas
        return self._calculate_results()
    
    def _calculate_results(self) -> BacktestResults:
        """
        Calcula métricas de performance
        """
        if not self.trades:
            print("Nenhum trade executado")
            return None
        
        # Converter para séries pandas
        equity_series = pd.Series(self.equity_curve, index=self.timestamps)
        returns = equity_series.pct_change().dropna()
        
        # Calcular drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        
        # Métricas básicas
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        
        # Métricas de risco
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (anualizado)
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() * 252 - self.risk_free_rate) / (returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = (returns.mean() * 252 - self.risk_free_rate) / (negative_returns.std() * np.sqrt(252))
        else:
            sortino_ratio = 0
        
        # Calmar ratio
        calmar_ratio = (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        
        # Retornos mensais
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_df = pd.DataFrame({
            'Month': monthly_returns.index.strftime('%Y-%m'),
            'Return': monthly_returns.values
        })
        
        metrics = {
            'total_return': total_return,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'final_capital': self.current_capital
        }
        
        return BacktestResults(
            trades=self.trades,
            equity_curve=equity_series,
            returns=returns,
            metrics=metrics,
            drawdown_series=drawdown,
            monthly_returns=monthly_df
        )
    
    def plot_results(self, results: BacktestResults, save_path: str = None):
        """
        Plota resultados do backtesting
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Resultados do Backtesting Avançado', fontsize=16)
        
        # 1. Curva de Equity
        axes[0, 0].plot(results.equity_curve.index, results.equity_curve.values)
        axes[0, 0].set_title('Curva de Equity')
        axes[0, 0].set_ylabel('Capital ($)')
        axes[0, 0].grid(True)
        
        # 2. Drawdown
        axes[0, 1].fill_between(results.drawdown_series.index, 
                               results.drawdown_series.values, 0, 
                               color='red', alpha=0.3)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)
        
        # 3. Distribuição de retornos
        axes[1, 0].hist(results.returns.values, bins=50, alpha=0.7)
        axes[1, 0].set_title('Distribuição de Retornos')
        axes[1, 0].set_xlabel('Retorno')
        axes[1, 0].set_ylabel('Frequência')
        axes[1, 0].grid(True)
        
        # 4. P&L por trade
        trade_pnls = [t.pnl for t in results.trades]
        axes[1, 1].plot(range(len(trade_pnls)), np.cumsum(trade_pnls))
        axes[1, 1].set_title('P&L Cumulativo por Trade')
        axes[1, 1].set_xlabel('Número do Trade')
        axes[1, 1].set_ylabel('P&L Cumulativo ($)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_summary(self, results: BacktestResults):
        """
        Imprime resumo dos resultados
        """
        print("\n=== RESUMO DO BACKTESTING ===")
        print(f"Capital Inicial: ${self.initial_capital:,.2f}")
        print(f"Capital Final: ${results.metrics['final_capital']:,.2f}")
        print(f"Retorno Total: {results.metrics['total_return']:.2%}")
        print(f"\nEstatísticas de Trading:")
        print(f"Total de Trades: {results.metrics['total_trades']}")
        print(f"Taxa de Acerto: {results.metrics['win_rate']:.2%}")
        print(f"Fator de Lucro: {results.metrics['profit_factor']:.2f}")
        print(f"Ganho Médio: ${results.metrics['avg_win']:,.2f}")
        print(f"Perda Média: ${results.metrics['avg_loss']:,.2f}")
        print(f"\nMétricas de Risco:")
        print(f"Drawdown Máximo: {results.metrics['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {results.metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {results.metrics['sortino_ratio']:.2f}")
        print(f"Calmar Ratio: {results.metrics['calmar_ratio']:.2f}")

def demonstrate_backtesting():
    """
    Demonstração do sistema de backtesting
    """
    print("=== DEMONSTRAÇÃO - SISTEMA DE BACKTESTING AVANÇADO ===")
    
    # Criar dados sintéticos
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    
    # Simular dados OHLC
    base_price = 1.1000
    returns = np.random.normal(0, 0.001, 1000)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.0005, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.0005, 1000))),
        'close': prices
    }, index=dates)
    
    # Ajustar high/low
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Criar sinais sintéticos (estratégia simples de momentum)
    signals = pd.DataFrame(index=dates)
    sma_short = data['close'].rolling(10).mean()
    sma_long = data['close'].rolling(50).mean()
    
    signals['signal'] = 0
    signals.loc[sma_short > sma_long, 'signal'] = 1  # Long
    signals.loc[sma_short < sma_long, 'signal'] = -1  # Short
    
    # Criar backtester
    backtester = AdvancedBacktester(
        initial_capital=100000,
        commission=0.0001,
        slippage=0.0001
    )
    
    # Executar backtesting
    results = backtester.run_backtest(data, signals)
    
    if results:
        # Imprimir resumo
        backtester.print_summary(results)
        
        # Plotar resultados
        backtester.plot_results(results)
    
    return results

if __name__ == "__main__":
    demonstrate_backtesting()