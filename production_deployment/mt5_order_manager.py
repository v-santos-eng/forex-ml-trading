import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import logging
import time
from enum import Enum
from mt5_connector import MT5Connector

class OrderType(Enum):
    """Tipos de ordem"""
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL
    BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT
    BUY_STOP = mt5.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt5.ORDER_TYPE_SELL_STOP

class OrderFilling(Enum):
    """Tipos de preenchimento de ordem"""
    FOK = mt5.ORDER_FILLING_FOK  # Fill or Kill
    IOC = mt5.ORDER_FILLING_IOC  # Immediate or Cancel
    RETURN = mt5.ORDER_FILLING_RETURN  # Return

class TradeAction(Enum):
    """Ações de trade"""
    DEAL = mt5.TRADE_ACTION_DEAL
    PENDING = mt5.TRADE_ACTION_PENDING
    SLTP = mt5.TRADE_ACTION_SLTP
    MODIFY = mt5.TRADE_ACTION_MODIFY
    REMOVE = mt5.TRADE_ACTION_REMOVE

class MT5OrderManager:
    """
    Gerenciador de ordens para MetaTrader 5
    """
    
    def __init__(self, connector: MT5Connector):
        self.connector = connector
        self.logger = self._setup_logger()
        self.magic_number = 234000  # Número mágico para identificar ordens do sistema
        
    def _setup_logger(self) -> logging.Logger:
        """Configura o sistema de logging"""
        logger = logging.getLogger('MT5OrderManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _validate_symbol(self, symbol: str) -> bool:
        """Valida se o símbolo está disponível para trading"""
        if not self.connector.connected:
            self.logger.error("MT5 não conectado")
            return False
            
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Símbolo {symbol} não encontrado")
                return False
                
            if not symbol_info.visible:
                # Tentar selecionar o símbolo
                if not mt5.symbol_select(symbol, True):
                    self.logger.error(f"Não foi possível selecionar o símbolo {symbol}")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao validar símbolo {symbol}: {e}")
            return False
    
    def _calculate_lot_size(self, symbol: str, risk_amount: float, stop_loss_pips: float) -> float:
        """Calcula o tamanho do lote baseado no risco"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0.01  # Lote mínimo padrão
                
            account_info = mt5.account_info()
            if account_info is None:
                return 0.01
                
            # Valor por pip
            pip_value = symbol_info.trade_tick_value * symbol_info.trade_tick_size
            
            # Calcular lote baseado no risco
            if stop_loss_pips > 0 and pip_value > 0:
                lot_size = risk_amount / (stop_loss_pips * pip_value)
                
                # Ajustar para o step de volume
                lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
                
                # Limitar aos valores mínimo e máximo
                lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
                
                return lot_size
                
            return symbol_info.volume_min
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular tamanho do lote: {e}")
            return 0.01
    
    def place_market_order(self, symbol: str, order_type: OrderType, volume: float,
                          stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                          comment: str = "Python MT5", magic: Optional[int] = None) -> Optional[Dict]:
        """Coloca uma ordem a mercado"""
        if not self._validate_symbol(symbol):
            return None
            
        try:
            # Obter preço atual
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.error(f"Não foi possível obter tick para {symbol}")
                return None
                
            # Definir preço baseado no tipo de ordem
            if order_type == OrderType.BUY:
                price = tick.ask
            elif order_type == OrderType.SELL:
                price = tick.bid
            else:
                self.logger.error(f"Tipo de ordem {order_type} não suportado para ordem a mercado")
                return None
                
            # Preparar requisição
            request = {
                "action": TradeAction.DEAL.value,
                "symbol": symbol,
                "volume": volume,
                "type": order_type.value,
                "price": price,
                "magic": magic or self.magic_number,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": OrderFilling.IOC.value,
            }
            
            # Adicionar SL/TP se fornecidos
            if stop_loss is not None:
                request["sl"] = stop_loss
            if take_profit is not None:
                request["tp"] = take_profit
                
            # Enviar ordem
            result = mt5.order_send(request)
            
            if result is None:
                self.logger.error("Falha ao enviar ordem - resultado None")
                return None
                
            # Verificar se a ordem foi executada com sucesso
            # retcode 0 também indica sucesso (SUCCESS)
            if result.retcode not in [0, mt5.TRADE_RETCODE_DONE]:
                self.logger.error(f"Falha ao executar ordem: {result.retcode} - {result.comment}")
                return None
                
            self.logger.info(f"Ordem executada: {symbol} {order_type.name} {volume} lotes a {price}")
            
            return {
                "order_id": result.order,
                "deal_id": result.deal,
                "symbol": symbol,
                "volume": result.volume,
                "price": result.price,
                "type": order_type.name,
                "retcode": result.retcode,
                "comment": result.comment,
                "request_id": result.request_id
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao colocar ordem a mercado: {e}")
            return None
    
    def place_pending_order(self, symbol: str, order_type: OrderType, volume: float, price: float,
                           stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                           expiration: Optional[datetime] = None, comment: str = "Python MT5",
                           magic: Optional[int] = None) -> Optional[Dict]:
        """Coloca uma ordem pendente"""
        if not self._validate_symbol(symbol):
            return None
            
        if order_type in [OrderType.BUY, OrderType.SELL]:
            self.logger.error("Use place_market_order para ordens a mercado")
            return None
            
        try:
            request = {
                "action": TradeAction.PENDING.value,
                "symbol": symbol,
                "volume": volume,
                "type": order_type.value,
                "price": price,
                "magic": magic or self.magic_number,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": OrderFilling.RETURN.value,
            }
            
            # Adicionar SL/TP se fornecidos
            if stop_loss is not None:
                request["sl"] = stop_loss
            if take_profit is not None:
                request["tp"] = take_profit
                
            # Adicionar expiração se fornecida
            if expiration is not None:
                request["type_time"] = mt5.ORDER_TIME_SPECIFIED
                request["expiration"] = int(expiration.timestamp())
                
            result = mt5.order_send(request)
            
            if result is None:
                self.logger.error("Falha ao enviar ordem pendente - resultado None")
                return None
                
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Falha ao executar ordem pendente: {result.retcode} - {result.comment}")
                return None
                
            self.logger.info(f"Ordem pendente colocada: {symbol} {order_type.name} {volume} lotes a {price}")
            
            return {
                "order_id": result.order,
                "symbol": symbol,
                "volume": volume,
                "price": price,
                "type": order_type.name,
                "retcode": result.retcode,
                "comment": result.comment
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao colocar ordem pendente: {e}")
            return None
    
    def close_position(self, symbol: str, volume: Optional[float] = None, 
                      comment: str = "Python Close") -> Optional[Dict]:
        """Fecha uma posição"""
        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions is None or len(positions) == 0:
                self.logger.warning(f"Nenhuma posição encontrada para {symbol}")
                return None
                
            position = positions[0]  # Pegar primeira posição
            
            # Determinar tipo de ordem para fechar
            if position.type == mt5.POSITION_TYPE_BUY:
                order_type = OrderType.SELL
            else:
                order_type = OrderType.BUY
                
            # Volume a fechar
            close_volume = volume or position.volume
            
            # Obter preço atual
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.error(f"Não foi possível obter tick para {symbol}")
                return None
                
            price = tick.bid if order_type == OrderType.SELL else tick.ask
            
            request = {
                "action": TradeAction.DEAL.value,
                "symbol": symbol,
                "volume": close_volume,
                "type": order_type.value,
                "position": position.ticket,
                "price": price,
                "magic": self.magic_number,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": OrderFilling.IOC.value,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                self.logger.error("Falha ao fechar posição - resultado None")
                return None
                
            # Verificar se o fechamento foi executado com sucesso
            # retcode 0 também indica sucesso (SUCCESS)
            if result.retcode not in [0, mt5.TRADE_RETCODE_DONE]:
                self.logger.error(f"Falha ao fechar posição: {result.retcode} - {result.comment}")
                return None
                
            self.logger.info(f"Posição fechada: {symbol} {close_volume} lotes")
            
            return {
                "deal_id": result.deal,
                "symbol": symbol,
                "volume": result.volume,
                "price": result.price,
                "profit": position.profit,
                "retcode": result.retcode,
                "comment": result.comment
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao fechar posição: {e}")
            return None
    
    def modify_position(self, symbol: str, stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> Optional[Dict]:
        """Modifica SL/TP de uma posição"""
        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions is None or len(positions) == 0:
                self.logger.warning(f"Nenhuma posição encontrada para {symbol}")
                return None
                
            position = positions[0]
            
            request = {
                "action": TradeAction.SLTP.value,
                "symbol": symbol,
                "position": position.ticket,
                "magic": self.magic_number,
            }
            
            if stop_loss is not None:
                request["sl"] = stop_loss
            if take_profit is not None:
                request["tp"] = take_profit
                
            result = mt5.order_send(request)
            
            if result is None:
                self.logger.error("Falha ao modificar posição - resultado None")
                return None
                
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Falha ao modificar posição: {result.retcode} - {result.comment}")
                return None
                
            self.logger.info(f"Posição modificada: {symbol} SL={stop_loss} TP={take_profit}")
            
            return {
                "symbol": symbol,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "retcode": result.retcode,
                "comment": result.comment
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao modificar posição: {e}")
            return None
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancela uma ordem pendente"""
        try:
            request = {
                "action": TradeAction.REMOVE.value,
                "order": order_id,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                self.logger.error("Falha ao cancelar ordem - resultado None")
                return False
                
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Falha ao cancelar ordem: {result.retcode} - {result.comment}")
                return False
                
            self.logger.info(f"Ordem {order_id} cancelada com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao cancelar ordem: {e}")
            return False
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Obtém posições abertas"""
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
                
            if positions is None:
                return []
                
            result = []
            for pos in positions:
                # Verificar se o atributo commission existe (pode não existir em contas demo)
                commission = getattr(pos, 'commission', 0.0)
                
                result.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "commission": commission,
                    "magic": pos.magic,
                    "comment": pos.comment,
                    "time": datetime.fromtimestamp(pos.time),
                    "sl": pos.sl,
                    "tp": pos.tp
                })
                
            return result
            
        except Exception as e:
            self.logger.error(f"Erro ao obter posições: {e}")
            return []
    
    def get_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Obtém ordens pendentes"""
        try:
            if symbol:
                orders = mt5.orders_get(symbol=symbol)
            else:
                orders = mt5.orders_get()
                
            if orders is None:
                return []
                
            result = []
            for order in orders:
                result.append({
                    "ticket": order.ticket,
                    "symbol": order.symbol,
                    "type": self._get_order_type_name(order.type),
                    "volume": order.volume_initial,
                    "volume_current": order.volume_current,
                    "price_open": order.price_open,
                    "price_current": order.price_current,
                    "magic": order.magic,
                    "comment": order.comment,
                    "time_setup": datetime.fromtimestamp(order.time_setup),
                    "sl": order.sl,
                    "tp": order.tp
                })
                
            return result
            
        except Exception as e:
            self.logger.error(f"Erro ao obter ordens: {e}")
            return []
    
    def _get_order_type_name(self, order_type: int) -> str:
        """Converte tipo de ordem numérico para nome"""
        type_map = {
            mt5.ORDER_TYPE_BUY: "BUY",
            mt5.ORDER_TYPE_SELL: "SELL",
            mt5.ORDER_TYPE_BUY_LIMIT: "BUY_LIMIT",
            mt5.ORDER_TYPE_SELL_LIMIT: "SELL_LIMIT",
            mt5.ORDER_TYPE_BUY_STOP: "BUY_STOP",
            mt5.ORDER_TYPE_SELL_STOP: "SELL_STOP"
        }
        return type_map.get(order_type, "UNKNOWN")
    
    def get_account_summary(self) -> Optional[Dict]:
        """Obtém resumo da conta"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return None
                
            positions = self.get_positions()
            total_profit = sum(pos['profit'] for pos in positions)
            
            return {
                "balance": account_info.balance,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "free_margin": account_info.margin_free,
                "margin_level": account_info.margin_level,
                "profit": account_info.profit,
                "currency": account_info.currency,
                "open_positions": len(positions),
                "total_position_profit": total_profit
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao obter resumo da conta: {e}")
            return None

if __name__ == "__main__":
    # Teste do gerenciador de ordens
    from mt5_connector import MT5Connector
    
    connector = MT5Connector()
    
    if connector.connect():
        print("✅ Conectado ao MT5")
        
        order_manager = MT5OrderManager(connector)
        
        # Testar informações da conta
        account_summary = order_manager.get_account_summary()
        if account_summary:
            print(f"💰 Saldo: {account_summary['balance']} {account_summary['currency']}")
            print(f"📊 Equity: {account_summary['equity']}")
            print(f"📈 Posições abertas: {account_summary['open_positions']}")
        
        # Testar posições
        positions = order_manager.get_positions()
        print(f"📋 Posições: {len(positions)}")
        
        # Testar ordens pendentes
        orders = order_manager.get_orders()
        print(f"📝 Ordens pendentes: {len(orders)}")
        
        connector.disconnect()
    else:
        print("❌ Falha na conexão MT5")