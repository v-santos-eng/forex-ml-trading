import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Tuple, Callable, Any
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict, deque


class OrderType(Enum):
    """Tipos de ordem."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Lado da ordem."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Status da ordem."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Classe para representar uma ordem."""
    id: str
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Trade:
    """Classe para representar uma execução."""
    id: str
    timestamp: pd.Timestamp
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float


@dataclass
class Position:
    """Classe para representar uma posição."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    market_value: float = 0.0


class Portfolio:
    """Classe para gerenciar portfólio."""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.total_commission = 0.0
        self.total_slippage = 0.0
    
    def update_position(self, trade: Trade, current_price: float) -> None:
        """Atualiza posição baseada em uma execução."""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        position = self.positions[symbol]
        
        # Calcular nova quantidade e preço médio
        if trade.side == OrderSide.BUY:
            new_quantity = position.quantity + trade.quantity
            if new_quantity != 0:
                position.avg_price = (
                    (position.quantity * position.avg_price + 
                     trade.quantity * trade.price) / new_quantity
                )
            position.quantity = new_quantity
        else:  # SELL
            # Calcular PnL realizado
            if position.quantity > 0:
                realized_pnl = trade.quantity * (trade.price - position.avg_price)
                position.realized_pnl += realized_pnl
            
            position.quantity -= trade.quantity
            
            # Se posição zerada, resetar preço médio
            if abs(position.quantity) < 1e-8:
                position.quantity = 0.0
                position.avg_price = 0.0
        
        # Atualizar valor de mercado e PnL não realizado
        if position.quantity != 0:
            position.market_value = position.quantity * current_price
            position.unrealized_pnl = position.quantity * (current_price - position.avg_price)
        else:
            position.market_value = 0.0
            position.unrealized_pnl = 0.0
        
        # Atualizar cash
        if trade.side == OrderSide.BUY:
            self.cash -= trade.quantity * trade.price + trade.commission
        else:
            self.cash += trade.quantity * trade.price - trade.commission
        
        self.total_commission += trade.commission
        self.total_slippage += trade.slippage
        self.trades.append(trade)
    
    def get_total_value(self) -> float:
        """Calcula valor total do portfólio."""
        return self.cash + sum(pos.market_value for pos in self.positions.values())
    
    def get_total_pnl(self) -> float:
        """Calcula PnL total."""
        realized = sum(pos.realized_pnl for pos in self.positions.values())
        unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        return realized + unrealized
    
    def update_equity_curve(self, timestamp: pd.Timestamp) -> None:
        """Atualiza curva de equity."""
        total_value = self.get_total_value()
        self.equity_curve.append((timestamp, total_value))


class DataHandler(ABC):
    """Classe abstrata para manipulação de dados."""
    
    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        """Retorna a última barra de dados."""
        pass
    
    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int) -> Optional[pd.DataFrame]:
        """Retorna as últimas N barras."""
        pass
    
    @abstractmethod
    def update_bars(self) -> bool:
        """Atualiza dados para próximo timestamp."""
        pass
    
    @abstractmethod
    def get_current_timestamp(self) -> pd.Timestamp:
        """Retorna timestamp atual."""
        pass


class HistoricalDataHandler(DataHandler):
    """Handler para dados históricos."""
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Inicializa com dados históricos.
        
        Args:
            data: Dicionário {symbol: DataFrame} com dados OHLCV
        """
        self.data = data
        self.symbols = list(data.keys())
        
        # Encontrar range de datas comum
        all_dates = [df.index for df in data.values()]
        self.date_range = sorted(set.intersection(*[set(dates) for dates in all_dates]))
        
        self.current_index = 0
        self.latest_data = {symbol: None for symbol in self.symbols}
    
    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        """Retorna a última barra disponível."""
        return self.latest_data.get(symbol)
    
    def get_latest_bars(self, symbol: str, n: int) -> Optional[pd.DataFrame]:
        """Retorna as últimas N barras."""
        if self.current_index < n:
            return None
        
        end_idx = self.current_index
        start_idx = max(0, end_idx - n)
        
        dates_slice = self.date_range[start_idx:end_idx]
        return self.data[symbol].loc[dates_slice]
    
    def update_bars(self) -> bool:
        """Atualiza para próximo timestamp."""
        if self.current_index >= len(self.date_range):
            return False
        
        current_date = self.date_range[self.current_index]
        
        # Atualizar dados disponíveis
        for symbol in self.symbols:
            if current_date in self.data[symbol].index:
                self.latest_data[symbol] = self.data[symbol].loc[current_date]
        
        self.current_index += 1
        return True
    
    def get_current_timestamp(self) -> pd.Timestamp:
        """Retorna timestamp atual."""
        if self.current_index > 0:
            return self.date_range[self.current_index - 1]
        return self.date_range[0]


class ExecutionHandler(ABC):
    """Classe abstrata para execução de ordens."""
    
    @abstractmethod
    def execute_order(self, order: Order, current_price: float) -> Optional[Trade]:
        """Executa uma ordem."""
        pass


class SimulatedExecutionHandler(ExecutionHandler):
    """Handler de execução simulada."""
    
    def __init__(self, 
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005):
        """
        Inicializa handler de execução.
        
        Args:
            commission_rate: Taxa de comissão (% do valor negociado)
            slippage_rate: Taxa de slippage (% do preço)
        """
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.trade_id_counter = 0
    
    def execute_order(self, order: Order, current_price: float) -> Optional[Trade]:
        """Executa ordem com modelo de custos."""
        if order.status != OrderStatus.PENDING:
            return None
        
        # Determinar preço de execução
        if order.order_type == OrderType.MARKET:
            execution_price = current_price
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and current_price <= order.price:
                execution_price = order.price
            elif order.side == OrderSide.SELL and current_price >= order.price:
                execution_price = order.price
            else:
                return None  # Ordem não executada
        else:
            # Implementar outros tipos de ordem conforme necessário
            execution_price = current_price
        
        # Aplicar slippage
        slippage_amount = execution_price * self.slippage_rate
        if order.side == OrderSide.BUY:
            execution_price += slippage_amount
        else:
            execution_price -= slippage_amount
        
        # Calcular comissão
        commission = order.quantity * execution_price * self.commission_rate
        
        # Criar trade
        self.trade_id_counter += 1
        trade = Trade(
            id=f"trade_{self.trade_id_counter}",
            timestamp=order.timestamp,
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            commission=commission,
            slippage=slippage_amount * order.quantity
        )
        
        # Atualizar status da ordem
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.commission = commission
        order.slippage = slippage_amount * order.quantity
        
        return trade


class Strategy(ABC):
    """Classe abstrata para estratégias."""
    
    def __init__(self, name: str):
        self.name = name
        self.data_handler = None
        self.portfolio = None
        self.order_id_counter = 0
    
    def set_handlers(self, data_handler: DataHandler, portfolio: Portfolio) -> None:
        """Define handlers de dados e portfólio."""
        self.data_handler = data_handler
        self.portfolio = portfolio
    
    @abstractmethod
    def calculate_signals(self) -> List[Order]:
        """Calcula sinais e retorna lista de ordens."""
        pass
    
    def create_order(self, 
                    symbol: str,
                    side: OrderSide,
                    quantity: float,
                    order_type: OrderType = OrderType.MARKET,
                    price: Optional[float] = None) -> Order:
        """Cria uma nova ordem."""
        self.order_id_counter += 1
        return Order(
            id=f"{self.name}_order_{self.order_id_counter}",
            timestamp=self.data_handler.get_current_timestamp(),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price
        )


class BuyAndHoldStrategy(Strategy):
    """Estratégia Buy and Hold simples."""
    
    def __init__(self, symbols: List[str], allocation: Dict[str, float]):
        super().__init__("BuyAndHold")
        self.symbols = symbols
        self.allocation = allocation
        self.invested = False
    
    def calculate_signals(self) -> List[Order]:
        """Investe uma vez no início."""
        if self.invested:
            return []
        
        orders = []
        total_value = self.portfolio.get_total_value()
        
        for symbol in self.symbols:
            if symbol in self.allocation:
                allocation_pct = self.allocation[symbol]
                investment_amount = total_value * allocation_pct
                
                current_bar = self.data_handler.get_latest_bar(symbol)
                if current_bar is not None:
                    current_price = current_bar['close']
                    quantity = investment_amount / current_price
                    
                    order = self.create_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=quantity
                    )
                    orders.append(order)
        
        self.invested = True
        return orders


class MovingAverageCrossoverStrategy(Strategy):
    """Estratégia de cruzamento de médias móveis."""
    
    def __init__(self, 
                 symbol: str,
                 short_window: int = 10,
                 long_window: int = 30,
                 position_size: float = 0.1):
        super().__init__("MA_Crossover")
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        self.last_signal = None
    
    def calculate_signals(self) -> List[Order]:
        """Calcula sinais baseados em cruzamento de MAs."""
        bars = self.data_handler.get_latest_bars(self.symbol, self.long_window + 1)
        
        if bars is None or len(bars) < self.long_window:
            return []
        
        # Calcular médias móveis
        short_ma = bars['close'].rolling(self.short_window).mean().iloc[-1]
        long_ma = bars['close'].rolling(self.long_window).mean().iloc[-1]
        
        # Médias móveis anteriores
        short_ma_prev = bars['close'].rolling(self.short_window).mean().iloc[-2]
        long_ma_prev = bars['close'].rolling(self.long_window).mean().iloc[-2]
        
        current_price = bars['close'].iloc[-1]
        current_position = self.portfolio.positions.get(self.symbol, Position(self.symbol)).quantity
        
        orders = []
        
        # Sinal de compra: MA curta cruza acima da MA longa
        if (short_ma > long_ma and short_ma_prev <= long_ma_prev and 
            current_position <= 0):
            
            # Fechar posição vendida se existir
            if current_position < 0:
                close_order = self.create_order(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    quantity=abs(current_position)
                )
                orders.append(close_order)
            
            # Abrir posição comprada
            portfolio_value = self.portfolio.get_total_value()
            quantity = (portfolio_value * self.position_size) / current_price
            
            buy_order = self.create_order(
                symbol=self.symbol,
                side=OrderSide.BUY,
                quantity=quantity
            )
            orders.append(buy_order)
            self.last_signal = 'BUY'
        
        # Sinal de venda: MA curta cruza abaixo da MA longa
        elif (short_ma < long_ma and short_ma_prev >= long_ma_prev and 
              current_position >= 0):
            
            # Fechar posição comprada se existir
            if current_position > 0:
                close_order = self.create_order(
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    quantity=current_position
                )
                orders.append(close_order)
            
            self.last_signal = 'SELL'
        
        return orders


class BacktestingEngine:
    """Motor principal de backtesting."""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005):
        """
        Inicializa o motor de backtesting.
        
        Args:
            initial_capital: Capital inicial
            commission_rate: Taxa de comissão
            slippage_rate: Taxa de slippage
        """
        self.initial_capital = initial_capital
        self.portfolio = Portfolio(initial_capital)
        self.execution_handler = SimulatedExecutionHandler(commission_rate, slippage_rate)
        self.data_handler = None
        self.strategy = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Métricas de performance
        self.performance_metrics = {}
    
    def set_data_handler(self, data_handler: DataHandler) -> None:
        """Define handler de dados."""
        self.data_handler = data_handler
    
    def set_strategy(self, strategy: Strategy) -> None:
        """Define estratégia."""
        self.strategy = strategy
        self.strategy.set_handlers(self.data_handler, self.portfolio)
    
    def run_backtest(self) -> None:
        """Executa o backtest."""
        if not self.data_handler or not self.strategy:
            raise ValueError("Data handler e strategy devem ser definidos")
        
        self.logger.info(f"Iniciando backtest com capital inicial: ${self.initial_capital:,.2f}")
        
        # Loop principal do backtest
        while self.data_handler.update_bars():
            current_timestamp = self.data_handler.get_current_timestamp()
            
            # Atualizar valores de mercado das posições
            self._update_portfolio_values()
            
            # Calcular sinais da estratégia
            orders = self.strategy.calculate_signals()
            
            # Executar ordens
            for order in orders:
                current_bar = self.data_handler.get_latest_bar(order.symbol)
                if current_bar is not None:
                    current_price = current_bar['close']
                    trade = self.execution_handler.execute_order(order, current_price)
                    
                    if trade:
                        self.portfolio.update_position(trade, current_price)
                        self.logger.debug(f"Executado: {trade.side.value} {trade.quantity} {trade.symbol} @ {trade.price:.4f}")
            
            # Atualizar curva de equity
            self.portfolio.update_equity_curve(current_timestamp)
        
        # Calcular métricas finais
        self._calculate_performance_metrics()
        
        self.logger.info("Backtest concluído")
    
    def _update_portfolio_values(self) -> None:
        """Atualiza valores de mercado das posições."""
        for symbol, position in self.portfolio.positions.items():
            if position.quantity != 0:
                current_bar = self.data_handler.get_latest_bar(symbol)
                if current_bar is not None:
                    current_price = current_bar['close']
                    position.market_value = position.quantity * current_price
                    position.unrealized_pnl = position.quantity * (current_price - position.avg_price)
    
    def _calculate_performance_metrics(self) -> None:
        """Calcula métricas de performance."""
        if not self.portfolio.equity_curve:
            return
        
        # Converter curva de equity para DataFrame
        equity_df = pd.DataFrame(self.portfolio.equity_curve, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Calcular retornos
        equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
        
        # Métricas básicas
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(equity_df)) - 1
        annual_volatility = equity_df['returns'].std() * np.sqrt(252)
        
        # Sharpe Ratio
        risk_free_rate = 0.02  # Assumir 2% ao ano
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        # Calmar Ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        # Win Rate
        winning_trades = [t for t in self.portfolio.trades if self._calculate_trade_pnl(t) > 0]
        win_rate = len(winning_trades) / len(self.portfolio.trades) if self.portfolio.trades else 0
        
        # Profit Factor
        gross_profit = sum(self._calculate_trade_pnl(t) for t in self.portfolio.trades if self._calculate_trade_pnl(t) > 0)
        gross_loss = abs(sum(self._calculate_trade_pnl(t) for t in self.portfolio.trades if self._calculate_trade_pnl(t) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        self.performance_metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.portfolio.trades),
            'total_commission': self.portfolio.total_commission,
            'total_slippage': self.portfolio.total_slippage,
            'final_equity': equity_df['equity'].iloc[-1]
        }
    
    def _calculate_trade_pnl(self, trade: Trade) -> float:
        """Calcula PnL de um trade (simplificado)."""
        # Esta é uma implementação simplificada
        # Em um sistema real, seria necessário rastrear pares de trades
        return 0.0
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Retorna resumo de performance."""
        if not self.performance_metrics:
            self._calculate_performance_metrics()
        
        summary_data = []
        for metric, value in self.performance_metrics.items():
            if isinstance(value, float):
                if metric in ['total_return', 'annual_return', 'annual_volatility', 'max_drawdown', 'win_rate']:
                    formatted_value = f"{value:.2%}"
                elif metric in ['sharpe_ratio', 'calmar_ratio', 'profit_factor']:
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{value:,.2f}"
            else:
                formatted_value = str(value)
            
            summary_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Value': formatted_value
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_equity_curve(self) -> plt.Figure:
        """Plota curva de equity."""
        if not self.portfolio.equity_curve:
            raise ValueError("Nenhum dado de equity disponível")
        
        equity_df = pd.DataFrame(self.portfolio.equity_curve, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Curva de equity
        ax1.plot(equity_df.index, equity_df['equity'], linewidth=2, label='Portfolio Value')
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_title('Portfolio Equity Curve')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        
        ax2.fill_between(equity_df.index, equity_df['drawdown'], 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados de exemplo
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    
    # Simular dados de preços
    symbols = ['AAPL', 'GOOGL']
    data = {}
    
    for symbol in symbols:
        prices = [100.0]
        volumes = []
        
        for _ in range(251):
            change = np.random.randn() * 0.02
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
            volumes.append(np.random.randint(1000, 10000))
        
        # Criar OHLC
        ohlc_data = []
        for i in range(1, len(prices)):
            open_price = prices[i-1]
            close_price = prices[i]
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.005)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.005)
            
            ohlc_data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volumes[i-1]
            })
        
        data[symbol] = pd.DataFrame(ohlc_data, index=dates)
    
    # Criar e executar backtest
    engine = BacktestingEngine(initial_capital=100000)
    
    # Configurar data handler
    data_handler = HistoricalDataHandler(data)
    engine.set_data_handler(data_handler)
    
    # Configurar estratégia
    strategy = MovingAverageCrossoverStrategy('AAPL', short_window=10, long_window=30)
    engine.set_strategy(strategy)
    
    # Executar backtest
    engine.run_backtest()
    
    # Mostrar resultados
    print("Resumo de Performance:")
    print(engine.get_performance_summary())