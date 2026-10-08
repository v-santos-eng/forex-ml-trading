import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from mt5_connector import MT5Connector
from mt5_order_manager import MT5OrderManager, OrderType
from mt5_data_provider import MT5DataProvider

class RiskLevel(Enum):
    """Níveis de risco"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class RiskParameters:
    """Parâmetros de gestão de risco"""
    max_risk_per_trade: float = 0.02  # 2% por trade
    max_daily_loss: float = 0.05  # 5% perda diária máxima
    max_drawdown: float = 0.10  # 10% drawdown máximo
    max_positions: int = 5  # Máximo de posições simultâneas
    max_lot_size: float = 1.0  # Tamanho máximo do lote
    stop_loss_pips: float = 50.0  # SL padrão em pips
    take_profit_pips: float = 100.0  # TP padrão em pips
    trailing_stop_pips: float = 30.0  # Trailing stop em pips
    correlation_limit: float = 0.7  # Limite de correlação entre posições

@dataclass
class RiskAlert:
    """Alerta de risco"""
    level: RiskLevel
    message: str
    timestamp: datetime
    symbol: Optional[str] = None
    position_id: Optional[int] = None
    action_required: bool = False

class MT5RiskManager:
    """
    Gerenciador de risco para operações MT5
    """
    
    def __init__(self, connector: MT5Connector, order_manager: MT5OrderManager, 
                 data_provider: MT5DataProvider, risk_params: Optional[RiskParameters] = None):
        self.connector = connector
        self.order_manager = order_manager
        self.data_provider = data_provider
        self.risk_params = risk_params or RiskParameters()
        self.logger = self._setup_logger()
        
        # Estado do gerenciador
        self.monitoring = False
        self.monitor_thread = None
        self.alerts = []
        self.daily_stats = {}
        self.position_correlations = {}
        
        # Callbacks para alertas
        self.alert_callbacks = []
        
        # Histórico de performance
        self.performance_history = []
        
    def _setup_logger(self) -> logging.Logger:
        """Configura o sistema de logging"""
        logger = logging.getLogger('MT5RiskManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def add_alert_callback(self, callback: Callable[[RiskAlert], None]) -> None:
        """Adiciona callback para alertas de risco"""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert(self, alert: RiskAlert) -> None:
        """Dispara um alerta de risco"""
        self.alerts.append(alert)
        self.logger.warning(f"ALERTA {alert.level.value}: {alert.message}")
        
        # Notificar callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Erro no callback de alerta: {e}")
    
    def calculate_position_size(self, symbol: str, risk_amount: float, 
                              stop_loss_pips: float) -> float:
        """Calcula tamanho da posição baseado no risco"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0.01
                
            # Valor por pip
            pip_value = symbol_info.trade_tick_value * symbol_info.trade_tick_size
            
            if stop_loss_pips <= 0 or pip_value <= 0:
                return symbol_info.volume_min
                
            # Calcular lote baseado no risco
            lot_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Ajustar para o step de volume
            lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
            
            # Aplicar limites
            lot_size = max(symbol_info.volume_min, 
                          min(lot_size, symbol_info.volume_max, self.risk_params.max_lot_size))
            
            return lot_size
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular tamanho da posição: {e}")
            return 0.01
    
    def validate_trade(self, symbol: str, order_type: OrderType, volume: float,
                      stop_loss: Optional[float] = None, 
                      take_profit: Optional[float] = None) -> Tuple[bool, str]:
        """Valida se um trade está dentro dos parâmetros de risco"""
        try:
            # Verificar número máximo de posições
            positions = self.order_manager.get_positions()
            if len(positions) >= self.risk_params.max_positions:
                return False, f"Máximo de {self.risk_params.max_positions} posições atingido"
            
            # Verificar tamanho do lote
            if volume > self.risk_params.max_lot_size:
                return False, f"Volume {volume} excede máximo permitido {self.risk_params.max_lot_size}"
            
            # Verificar risco por trade
            account_info = mt5.account_info()
            if account_info is None:
                return False, "Não foi possível obter informações da conta"
                
            if stop_loss is not None:
                current_price = self.data_provider.get_current_price(symbol)
                if current_price is None:
                    return False, "Não foi possível obter preço atual"
                    
                # Calcular risco do trade
                if order_type == OrderType.BUY:
                    risk_pips = abs(current_price['ask'] - stop_loss)
                else:
                    risk_pips = abs(stop_loss - current_price['bid'])
                    
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is not None:
                    pip_value = symbol_info.trade_tick_value * symbol_info.trade_tick_size
                    trade_risk = risk_pips * pip_value * volume
                    risk_percentage = trade_risk / account_info.balance
                    
                    if risk_percentage > self.risk_params.max_risk_per_trade:
                        return False, f"Risco do trade {risk_percentage:.2%} excede máximo {self.risk_params.max_risk_per_trade:.2%}"
            
            # Verificar correlação com posições existentes
            correlation_risk = self._check_correlation_risk(symbol, order_type)
            if correlation_risk:
                return False, f"Risco de correlação detectado: {correlation_risk}"
            
            # Verificar perda diária
            daily_loss = self._calculate_daily_loss()
            if daily_loss > self.risk_params.max_daily_loss:
                return False, f"Perda diária {daily_loss:.2%} excede máximo {self.risk_params.max_daily_loss:.2%}"
            
            return True, "Trade aprovado"
            
        except Exception as e:
            self.logger.error(f"Erro na validação do trade: {e}")
            return False, f"Erro na validação: {e}"
    
    def _check_correlation_risk(self, symbol: str, order_type: OrderType) -> Optional[str]:
        """Verifica risco de correlação com posições existentes"""
        try:
            positions = self.order_manager.get_positions()
            if not positions:
                return None
                
            # Símbolos correlacionados (simplificado)
            correlations = {
                'EURUSD': ['GBPUSD', 'AUDUSD', 'NZDUSD'],
                'GBPUSD': ['EURUSD', 'AUDUSD', 'NZDUSD'],
                'USDJPY': ['USDCHF', 'USDCAD'],
                'USDCHF': ['USDJPY', 'USDCAD'],
                'AUDUSD': ['NZDUSD', 'EURUSD', 'GBPUSD'],
                'NZDUSD': ['AUDUSD', 'EURUSD', 'GBPUSD']
            }
            
            correlated_symbols = correlations.get(symbol, [])
            
            for pos in positions:
                if pos['symbol'] in correlated_symbols:
                    # Verificar se é a mesma direção
                    pos_direction = pos['type']
                    trade_direction = order_type.name
                    
                    if pos_direction == trade_direction:
                        return f"Posição correlacionada {pos['symbol']} na mesma direção"
                        
            return None
            
        except Exception as e:
            self.logger.error(f"Erro ao verificar correlação: {e}")
            return None
    
    def _calculate_daily_loss(self) -> float:
        """Calcula perda diária atual"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return 0.0
                
            # Obter histórico de deals do dia
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            deals = mt5.history_deals_get(today, datetime.now())
            
            if deals is None:
                return 0.0
                
            daily_profit = sum(deal.profit for deal in deals)
            
            if daily_profit >= 0:
                return 0.0
                
            return abs(daily_profit) / account_info.balance
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular perda diária: {e}")
            return 0.0
    
    def calculate_drawdown(self) -> float:
        """Calcula drawdown atual"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return 0.0
                
            # Obter histórico de 30 dias
            start_date = datetime.now() - timedelta(days=30)
            deals = mt5.history_deals_get(start_date, datetime.now())
            
            if deals is None or len(deals) == 0:
                return 0.0
                
            # Calcular equity histórico
            equity_history = []
            running_profit = 0.0
            
            for deal in deals:
                running_profit += deal.profit
                equity_history.append(account_info.balance + running_profit)
            
            if not equity_history:
                return 0.0
                
            # Calcular drawdown máximo
            peak = equity_history[0]
            max_drawdown = 0.0
            
            for equity in equity_history:
                if equity > peak:
                    peak = equity
                else:
                    drawdown = (peak - equity) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                    
            return max_drawdown
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular drawdown: {e}")
            return 0.0
    
    def apply_trailing_stop(self, symbol: str) -> bool:
        """Aplica trailing stop a posições"""
        try:
            positions = self.order_manager.get_positions(symbol)
            if not positions:
                return True
                
            current_price = self.data_provider.get_current_price(symbol)
            if current_price is None:
                return False
                
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
                
            trailing_distance = self.risk_params.trailing_stop_pips * symbol_info.point
            
            for pos in positions:
                position_type = pos['type']
                current_sl = pos['sl']
                entry_price = pos['price_open']
                
                if position_type == 'BUY':
                    # Para posição de compra
                    new_sl = current_price['bid'] - trailing_distance
                    
                    # Só mover SL para cima
                    if current_sl == 0 or new_sl > current_sl:
                        # Verificar se há lucro suficiente
                        if current_price['bid'] > entry_price + trailing_distance:
                            result = self.order_manager.modify_position(symbol, stop_loss=new_sl)
                            if result:
                                self.logger.info(f"Trailing stop aplicado: {symbol} BUY SL={new_sl}")
                                
                elif position_type == 'SELL':
                    # Para posição de venda
                    new_sl = current_price['ask'] + trailing_distance
                    
                    # Só mover SL para baixo
                    if current_sl == 0 or new_sl < current_sl:
                        # Verificar se há lucro suficiente
                        if current_price['ask'] < entry_price - trailing_distance:
                            result = self.order_manager.modify_position(symbol, stop_loss=new_sl)
                            if result:
                                self.logger.info(f"Trailing stop aplicado: {symbol} SELL SL={new_sl}")
                                
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao aplicar trailing stop: {e}")
            return False
    
    def monitor_positions(self) -> None:
        """Monitora posições em tempo real"""
        try:
            positions = self.order_manager.get_positions()
            
            for pos in positions:
                symbol = pos['symbol']
                
                # Verificar drawdown da posição
                if pos['profit'] < 0:
                    loss_percentage = abs(pos['profit']) / pos['volume']
                    # Implementar lógica de alerta baseada na perda
                    
                # Aplicar trailing stop
                self.apply_trailing_stop(symbol)
                
                # Verificar tempo de posição aberta
                position_age = datetime.now() - pos['time']
                if position_age > timedelta(hours=24):
                    alert = RiskAlert(
                        level=RiskLevel.MEDIUM,
                        message=f"Posição {symbol} aberta há mais de 24h",
                        timestamp=datetime.now(),
                        symbol=symbol,
                        position_id=pos['ticket']
                    )
                    self._trigger_alert(alert)
            
            # Verificar drawdown geral
            drawdown = self.calculate_drawdown()
            if drawdown > self.risk_params.max_drawdown:
                alert = RiskAlert(
                    level=RiskLevel.CRITICAL,
                    message=f"Drawdown {drawdown:.2%} excede máximo {self.risk_params.max_drawdown:.2%}",
                    timestamp=datetime.now(),
                    action_required=True
                )
                self._trigger_alert(alert)
                
            # Verificar perda diária
            daily_loss = self._calculate_daily_loss()
            if daily_loss > self.risk_params.max_daily_loss:
                alert = RiskAlert(
                    level=RiskLevel.CRITICAL,
                    message=f"Perda diária {daily_loss:.2%} excede máximo {self.risk_params.max_daily_loss:.2%}",
                    timestamp=datetime.now(),
                    action_required=True
                )
                self._trigger_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Erro no monitoramento de posições: {e}")
    
    def start_monitoring(self, interval: float = 30.0) -> None:
        """Inicia monitoramento contínuo"""
        if self.monitoring:
            return
            
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                try:
                    self.monitor_positions()
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"Erro no loop de monitoramento: {e}")
                    time.sleep(5)
                    
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Monitoramento de risco iniciado")
    
    def stop_monitoring(self) -> None:
        """Para monitoramento contínuo"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Monitoramento de risco parado")
    
    def get_risk_summary(self) -> Dict:
        """Obtém resumo de risco atual"""
        try:
            account_info = mt5.account_info()
            positions = self.order_manager.get_positions()
            
            total_exposure = sum(pos['volume'] for pos in positions)
            total_profit = sum(pos['profit'] for pos in positions)
            
            return {
                'account_balance': account_info.balance if account_info else 0,
                'account_equity': account_info.equity if account_info else 0,
                'open_positions': len(positions),
                'total_exposure': total_exposure,
                'total_profit': total_profit,
                'daily_loss': self._calculate_daily_loss(),
                'drawdown': self.calculate_drawdown(),
                'recent_alerts': len([a for a in self.alerts if (datetime.now() - a.timestamp).seconds < 3600]),
                'risk_level': self._assess_overall_risk()
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao obter resumo de risco: {e}")
            return {}
    
    def _assess_overall_risk(self) -> RiskLevel:
        """Avalia nível de risco geral"""
        try:
            daily_loss = self._calculate_daily_loss()
            drawdown = self.calculate_drawdown()
            positions = self.order_manager.get_positions()
            
            # Critérios de risco
            if (daily_loss > self.risk_params.max_daily_loss * 0.8 or 
                drawdown > self.risk_params.max_drawdown * 0.8 or
                len(positions) >= self.risk_params.max_positions):
                return RiskLevel.CRITICAL
                
            elif (daily_loss > self.risk_params.max_daily_loss * 0.5 or
                  drawdown > self.risk_params.max_drawdown * 0.5 or
                  len(positions) >= self.risk_params.max_positions * 0.8):
                return RiskLevel.HIGH
                
            elif (daily_loss > self.risk_params.max_daily_loss * 0.25 or
                  drawdown > self.risk_params.max_drawdown * 0.25):
                return RiskLevel.MEDIUM
                
            return RiskLevel.LOW
            
        except Exception as e:
            self.logger.error(f"Erro ao avaliar risco geral: {e}")
            return RiskLevel.MEDIUM

if __name__ == "__main__":
    # Teste do gerenciador de risco
    from mt5_connector import MT5Connector
    from mt5_order_manager import MT5OrderManager
    from mt5_data_provider import MT5DataProvider
    
    connector = MT5Connector()
    
    if connector.connect():
        print("✅ Conectado ao MT5")
        
        order_manager = MT5OrderManager(connector)
        data_provider = MT5DataProvider(connector)
        risk_manager = MT5RiskManager(connector, order_manager, data_provider)
        
        # Testar resumo de risco
        risk_summary = risk_manager.get_risk_summary()
        print(f"📊 Resumo de Risco:")
        for key, value in risk_summary.items():
            print(f"   {key}: {value}")
        
        # Testar validação de trade
        is_valid, message = risk_manager.validate_trade('EURUSD', OrderType.BUY, 0.1)
        print(f"✅ Validação de trade: {is_valid} - {message}")
        
        connector.disconnect()
    else:
        print("❌ Falha na conexão MT5")