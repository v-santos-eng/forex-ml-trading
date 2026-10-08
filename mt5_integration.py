#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Integração com MetaTrader 5
Implementa conexão nativa com MT5 sem WebSocket
Autor: Sistema Forex ML
Data: 2024
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass, field
from enum import Enum

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mt5_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Tipos de ordem suportados"""
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL
    BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT
    BUY_STOP = mt5.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt5.ORDER_TYPE_SELL_STOP

class TradeAction(Enum):
    """Ações de trade"""
    DEAL = mt5.TRADE_ACTION_DEAL
    PENDING = mt5.TRADE_ACTION_PENDING
    SLTP = mt5.TRADE_ACTION_SLTP
    MODIFY = mt5.TRADE_ACTION_MODIFY
    REMOVE = mt5.TRADE_ACTION_REMOVE

@dataclass
class MT5Config:
    """Configuração para conexão MT5"""
    login: int
    password: str
    server: str
    path: str = ""
    timeout: int = 60000
    portable: bool = False

@dataclass
class TradeRequest:
    """Estrutura para requisição de trade"""
    symbol: str
    volume: float
    action: TradeAction
    type: OrderType
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    deviation: int = 20
    magic: int = 234000
    comment: str = "Python MT5"
    type_time: int = mt5.ORDER_TIME_GTC
    type_filling: int = mt5.ORDER_FILLING_IOC

class MT5Integration:
    """
    Classe principal para integração com MetaTrader 5
    Implementa todas as funcionalidades necessárias para trading automatizado
    com suporte à reconexão automática e persistência de sessão
    """
    
    def __init__(self, config: MT5Config):
        self.config = config
        self._last_config = config  # Armazenar para reconexão
        self.connected = False
        self.account_info = None
        self.symbols_info = {}
        self.last_error = None
        
        # Controle de sessão
        self._session_start_time = None
        self._last_activity_time = None
        self._connection_lock = threading.Lock()
        
        # Auto reconnect manager (será inicializado externamente)
        self.auto_reconnect_manager = None
        
    def connect(self, config: Optional[MT5Config] = None) -> bool:
        """
        Estabelece conexão com MetaTrader 5 com verificação otimizada de sessão existente
        
        Args:
            config: Configuração MT5 (opcional, usa a configuração atual se não fornecida)
        
        Returns:
            bool: True se conectado com sucesso
        """
        with self._connection_lock:
            try:
                # Usar configuração fornecida ou atual
                if config:
                    self.config = config
                    self._last_config = config
                
                # Verificação rápida de sessão existente (evita login desnecessário)
                if self._check_existing_session():
                    # Verificar se é a mesma conta
                    current_account = mt5.account_info()
                    if current_account and current_account.login == self.config.login:
                        logger.info("Reutilizando sessão MT5 existente")
                        return True
                    else:
                        logger.info(f"Conta diferente detectada ({current_account.login if current_account else 'N/A'}), nova conexão necessária")
                
                logger.info("Iniciando nova conexão MT5...")
                
                # Verificar se MT5 já está inicializado mas com conta diferente
                terminal_info = mt5.terminal_info()
                if terminal_info:
                    current_account = mt5.account_info()
                    if (current_account and 
                        current_account.login != self.config.login):
                        logger.info(f"Desconectando conta {current_account.login} para conectar conta {self.config.login}")
                        mt5.shutdown()
                        time.sleep(0.5)  # Pequena pausa para garantir shutdown completo
                
                # Inicializar MT5 com parâmetros otimizados
                init_params = {
                    'login': self.config.login,
                    'password': self.config.password,
                    'server': self.config.server,
                    'timeout': self.config.timeout,
                    'portable': self.config.portable
                }
                
                # Adicionar path apenas se especificado
                if self.config.path:
                    init_params['path'] = self.config.path
                
                if not mt5.initialize(**init_params):
                    self.last_error = f"Falha na inicialização: {mt5.last_error()}"
                    logger.error(self.last_error)
                    return False
                
                # Verificar conexão e obter informações da conta
                account_info = mt5.account_info()
                if account_info is None:
                    self.last_error = f"Falha ao obter informações da conta: {mt5.last_error()}"
                    logger.error(self.last_error)
                    return False
                
                # Verificar se login foi bem-sucedido com a conta correta
                if account_info.login != self.config.login:
                    self.last_error = f"Login incorreto: esperado {self.config.login}, obtido {account_info.login}"
                    logger.error(self.last_error)
                    return False
                
                # Atualizar estado da conexão
                self.account_info = account_info._asdict()
                self.connected = True
                self._session_start_time = datetime.now()
                self._last_activity_time = datetime.now()
                
                logger.info(f"Nova conexão MT5 estabelecida - Conta: {self.account_info['login']}")
                logger.info(f"Servidor: {self.account_info['server']}")
                logger.info(f"Saldo: {self.account_info['balance']:.2f} {self.account_info['currency']}")
                
                return True
                
            except Exception as e:
                self.last_error = f"Erro na conexão: {str(e)}"
                logger.error(self.last_error)
                return False
    
    def _check_existing_session(self) -> bool:
        """
        Verifica se existe uma sessão MT5 ativa e válida de forma eficiente
        (Não valida conta específica - isso é feito no método connect)
        
        Returns:
            bool: True se sessão MT5 válida existe
        """
        try:
            # Verificação rápida: se já temos uma conexão ativa recente
            if (self.connected and self._last_activity_time and 
                (datetime.now() - self._last_activity_time).seconds < 30):
                # Verificação leve para confirmar que ainda está ativo
                try:
                    terminal_info = mt5.terminal_info()
                    if terminal_info and terminal_info.connected:
                        logger.debug("Sessão MT5 ativa confirmada (verificação rápida)")
                        return True
                except:
                    pass
            
            # Verificação completa: MT5 está inicializado e conectado
            terminal_info = mt5.terminal_info()
            if not terminal_info:
                logger.debug("MT5 não está inicializado")
                return False
            
            # Verificar se terminal está conectado ao servidor
            if not terminal_info.connected:
                logger.debug("Terminal MT5 não está conectado ao servidor")
                return False
            
            # Verificar informações da conta (qualquer conta válida)
            account_info = mt5.account_info()
            if account_info is None:
                logger.debug("Não foi possível obter informações da conta")
                return False
            
            # Atualizar última atividade
            self._last_activity_time = datetime.now()
            
            logger.debug(f"Sessão MT5 existente detectada - Conta: {account_info.login}, Servidor: {account_info.server}")
            return True
            
        except Exception as e:
            logger.debug(f"Erro ao verificar sessão existente: {e}")
            return False
    
    def is_connection_healthy(self) -> bool:
        """
        Verifica se a conexão MT5 está saudável
        
        Returns:
            bool: True se conexão está saudável
        """
        try:
            if not self.connected:
                return False
            
            # Verificar terminal info
            terminal_info = mt5.terminal_info()
            if not terminal_info:
                return False
            
            # Verificar se terminal está conectado
            if not terminal_info.connected:
                return False
            
            # Verificar account info
            account_info = mt5.account_info()
            if not account_info:
                return False
            
            # Verificar se é a conta correta
            if self.config and account_info.login != self.config.login:
                return False
            
            # Atualizar última atividade
            self._last_activity_time = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Erro ao verificar saúde da conexão: {e}")
            return False
    
    def is_mt5_ready(self) -> bool:
        """
        Verificação rápida se MT5 está pronto para uso (sem login)
        
        Returns:
            bool: True se MT5 está inicializado e conectado
        """
        try:
            terminal_info = mt5.terminal_info()
            if not terminal_info:
                return False
            
            return terminal_info.connected
            
        except Exception:
            return False
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        Retorna informações da sessão atual
        
        Returns:
            Dict com informações da sessão
        """
        return {
            'connected': self.connected,
            'session_start': self._session_start_time,
            'last_activity': self._last_activity_time,
            'account_login': self.account_info.get('login') if self.account_info else None,
            'server': self.account_info.get('server') if self.account_info else None
        }
    
    def disconnect(self) -> None:
        """
        Desconecta do MetaTrader 5
        """
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Desconectado do MT5")
    
    def get_account_info(self) -> Optional[Dict]:
        """
        Obtém informações da conta
        
        Returns:
            Dict: Informações da conta ou None se erro
        """
        if not self.connected:
            return None
            
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return None
            
            self.account_info = account_info._asdict()
            return self.account_info
            
        except Exception as e:
            logger.error(f"Erro ao obter informações da conta: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Obtém informações do símbolo
        
        Args:
            symbol: Símbolo (ex: EURUSD)
            
        Returns:
            Dict: Informações do símbolo ou None se erro
        """
        if not self.connected:
            return None
            
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Símbolo {symbol} não encontrado")
                return None
            
            info_dict = symbol_info._asdict()
            self.symbols_info[symbol] = info_dict
            return info_dict
            
        except Exception as e:
            logger.error(f"Erro ao obter informações do símbolo {symbol}: {e}")
            return None
    
    def get_tick(self, symbol: str) -> Optional[Dict]:
        """
        Obtém último tick do símbolo
        
        Args:
            symbol: Símbolo
            
        Returns:
            Dict: Dados do tick ou None se erro
        """
        if not self.connected:
            return None
            
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            return tick._asdict()
            
        except Exception as e:
            logger.error(f"Erro ao obter tick de {symbol}: {e}")
            return None
    
    def get_symbol_tick(self, symbol: str) -> Optional[Dict]:
        """
        Alias para get_tick - obtém último tick do símbolo
        
        Args:
            symbol: Símbolo
            
        Returns:
            Dict: Dados do tick ou None se erro
        """
        return self.get_tick(symbol)
    
    def get_market_book(self, symbol: str) -> Optional[Dict]:
        """
        Obtém profundidade de mercado (book de ofertas)
        
        Args:
            symbol: Símbolo
            
        Returns:
            Dict: Book de ofertas ou None se erro
        """
        if not self.connected:
            return None
            
        try:
            # MT5 não fornece book completo via API Python
            # Retornamos bid/ask como aproximação
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
                
            return {
                'symbol': symbol,
                'bids': [{'price': tick.bid, 'volume': tick.volume if hasattr(tick, 'volume') else 1}],
                'asks': [{'price': tick.ask, 'volume': tick.volume if hasattr(tick, 'volume') else 1}],
                'time': datetime.fromtimestamp(tick.time)
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter market book para {symbol}: {e}")
            return None
    
    def get_rates(self, symbol: str, timeframe: int, count: int = 100) -> Optional[pd.DataFrame]:
        """
        Obtém dados históricos de preços
        
        Args:
            symbol: Símbolo
            timeframe: Timeframe (mt5.TIMEFRAME_M1, etc.)
            count: Número de barras
            
        Returns:
            DataFrame: Dados históricos ou None se erro
        """
        if not self.connected:
            return None
            
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None:
                logger.warning(f"Não foi possível obter dados de {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            logger.error(f"Erro ao obter dados históricos de {symbol}: {e}")
            return None
    
    def send_order(self, request: TradeRequest) -> Optional[Dict]:
        """
        Envia ordem para o mercado
        
        Args:
            request: Requisição de trade
            
        Returns:
            Dict: Resultado da ordem ou None se erro
        """
        if not self.connected:
            logger.error("Não conectado ao MT5")
            return None
        
        try:
            # Verificar se símbolo está disponível
            symbol_info = self.get_symbol_info(request.symbol)
            if symbol_info is None:
                logger.error(f"Símbolo {request.symbol} não disponível")
                return None
            
            # Preparar requisição
            trade_request = {
                "action": request.action.value,
                "symbol": request.symbol,
                "volume": request.volume,
                "type": request.type.value,
                "price": request.price,
                "sl": request.sl,
                "tp": request.tp,
                "deviation": request.deviation,
                "magic": request.magic,
                "comment": request.comment,
                "type_time": request.type_time,
                "type_filling": request.type_filling,
            }
            
            # Remover campos None, mas manter SL e TP mesmo se 0.0
            # SL e TP com valor 0.0 são válidos (significa sem stop loss/take profit)
            trade_request = {k: v for k, v in trade_request.items() if v is not None}
            
            # Log para debug de SL/TP
            logger.info(f"Enviando ordem: SL={trade_request.get('sl', 'N/A')}, TP={trade_request.get('tp', 'N/A')}")
            
            # Enviar ordem
            result = mt5.order_send(trade_request)
            
            if result is None:
                self.last_error = f"Falha ao enviar ordem: {mt5.last_error()}"
                logger.error(self.last_error)
                return None
            
            result_dict = result._asdict()
            
            if result_dict['retcode'] != mt5.TRADE_RETCODE_DONE:
                logger.warning(f"Ordem não executada: {result_dict['comment']}")
            else:
                logger.info(f"Ordem executada: {request.symbol} {request.volume} lotes")
            
            return result_dict
            
        except Exception as e:
            self.last_error = f"Erro ao enviar ordem: {str(e)}"
            logger.error(self.last_error)
            return None
    
    def get_positions(self, symbol: str = None) -> List[Dict]:
        """
        Obtém posições abertas
        
        Args:
            symbol: Símbolo específico (opcional)
            
        Returns:
            List[Dict]: Lista de posições
        """
        if not self.connected:
            return []
        
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if positions is None:
                return []
            
            return [pos._asdict() for pos in positions]
            
        except Exception as e:
            logger.error(f"Erro ao obter posições: {e}")
            return []
    
    def get_orders(self, symbol: str = None) -> List[Dict]:
        """
        Obtém ordens pendentes
        
        Args:
            symbol: Símbolo específico (opcional)
            
        Returns:
            List[Dict]: Lista de ordens pendentes
        """
        if not self.connected:
            return []
        
        try:
            if symbol:
                orders = mt5.orders_get(symbol=symbol)
            else:
                orders = mt5.orders_get()
            
            if orders is None:
                return []
            
            return [order._asdict() for order in orders]
            
        except Exception as e:
            logger.error(f"Erro ao obter ordens: {e}")
            return []
    
    def close_position(self, ticket: int) -> Optional[Dict]:
        """
        Fecha posição específica
        
        Args:
            ticket: Ticket da posição
            
        Returns:
            Dict: Resultado da operação ou None se erro
        """
        if not self.connected:
            return None
        
        try:
            # Obter informações da posição
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.error(f"Posição {ticket} não encontrada")
                return None
            
            position = position[0]
            
            # Determinar tipo de ordem para fechamento
            if position.type == mt5.POSITION_TYPE_BUY:
                order_type = OrderType.SELL
            else:
                order_type = OrderType.BUY
            
            # Criar requisição de fechamento
            close_request = TradeRequest(
                symbol=position.symbol,
                volume=position.volume,
                action=TradeAction.DEAL,
                type=order_type,
                magic=position.magic,
                comment=f"Close #{ticket}"
            )
            
            return self.send_order(close_request)
            
        except Exception as e:
            logger.error(f"Erro ao fechar posição {ticket}: {e}")
            return None
    
    def modify_position(self, ticket: int, sl: float = 0.0, tp: float = 0.0) -> Optional[Dict]:
        """
        Modifica stop loss e take profit de uma posição
        
        Args:
            ticket: Ticket da posição
            sl: Novo stop loss
            tp: Novo take profit
            
        Returns:
            Dict: Resultado da operação ou None se erro
        """
        if not self.connected:
            return None
        
        try:
            # Obter informações da posição
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.error(f"Posição {ticket} não encontrada")
                return None
            
            position = position[0]
            
            # Preparar requisição de modificação
            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": ticket,
                "sl": sl,
                "tp": tp,
                "magic": position.magic,
                "comment": f"Modify #{ticket}"
            }
            
            result = mt5.order_send(modify_request)
            
            if result is None:
                self.last_error = f"Falha ao modificar posição: {mt5.last_error()}"
                logger.error(self.last_error)
                return None
            
            result_dict = result._asdict()
            
            if result_dict['retcode'] == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Posição {ticket} modificada com sucesso")
            else:
                logger.warning(f"Falha ao modificar posição: {result_dict['comment']}")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Erro ao modificar posição {ticket}: {e}")
            return None
    
    def get_history_deals(self, date_from: datetime, date_to: datetime) -> List[Dict]:
        """
        Obtém histórico de negociações
        
        Args:
            date_from: Data inicial
            date_to: Data final
            
        Returns:
            List[Dict]: Lista de negociações
        """
        if not self.connected:
            return []
        
        try:
            deals = mt5.history_deals_get(date_from, date_to)
            if deals is None:
                return []
            
            return [deal._asdict() for deal in deals]
            
        except Exception as e:
            logger.error(f"Erro ao obter histórico de negociações: {e}")
            return []
    
    def calculate_margin(self, symbol: str, volume: float, order_type: OrderType) -> Optional[float]:
        """
        Calcula margem necessária para uma operação
        
        Args:
            symbol: Símbolo
            volume: Volume
            order_type: Tipo de ordem
            
        Returns:
            float: Margem necessária ou None se erro
        """
        if not self.connected:
            return None
        
        try:
            margin = mt5.order_calc_margin(order_type.value, symbol, volume, 0.0)
            return margin
            
        except Exception as e:
            logger.error(f"Erro ao calcular margem: {e}")
            return None
    
    def calculate_profit(self, symbol: str, volume: float, order_type: OrderType, 
                        price_open: float, price_close: float) -> Optional[float]:
        """
        Calcula lucro/prejuízo de uma operação
        
        Args:
            symbol: Símbolo
            volume: Volume
            order_type: Tipo de ordem
            price_open: Preço de abertura
            price_close: Preço de fechamento
            
        Returns:
            float: Lucro/prejuízo ou None se erro
        """
        if not self.connected:
            return None
        
        try:
            profit = mt5.order_calc_profit(order_type.value, symbol, volume, price_open, price_close)
            return profit
            
        except Exception as e:
            logger.error(f"Erro ao calcular lucro: {e}")
            return None
    
    def is_market_open(self, symbol: str) -> bool:
        """
        Verifica se o mercado está aberto para o símbolo
        
        Args:
            symbol: Símbolo
            
        Returns:
            bool: True se mercado aberto
        """
        if not self.connected:
            return False
        
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return False
            
            # Verificar se símbolo está ativo para trading
            return symbol_info.get('trade_mode', 0) in [mt5.SYMBOL_TRADE_MODE_FULL, mt5.SYMBOL_TRADE_MODE_LONGONLY, mt5.SYMBOL_TRADE_MODE_SHORTONLY]
            
        except Exception as e:
            logger.error(f"Erro ao verificar status do mercado: {e}")
            return False
    
    def get_server_time(self) -> Optional[datetime]:
        """
        Obtém horário do servidor
        
        Returns:
            datetime: Horário do servidor ou None se erro
        """
        if not self.connected:
            return None
        
        try:
            # Obter tick de qualquer símbolo para pegar o tempo do servidor
            symbols = mt5.symbols_get()
            if symbols and len(symbols) > 0:
                tick = mt5.symbol_info_tick(symbols[0].name)
                if tick:
                    return datetime.fromtimestamp(tick.time)
            
            return datetime.now()
            
        except Exception as e:
            logger.error(f"Erro ao obter horário do servidor: {e}")
            return datetime.now()
    
    def __enter__(self):
        """Context manager entry"""
        if not self.connect():
            raise ConnectionError(f"Falha na conexão com MT5: {self.last_error}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

# Exemplo de uso
if __name__ == "__main__":
    # Configuração de exemplo (substitua pelos seus dados)
    config = MT5Config(
        login=123456,  # Seu login
        password="sua_senha",  # Sua senha
        server="servidor_da_corretora"  # Servidor da corretora
    )
    
    # Teste de conexão
    mt5_client = MT5Integration(config)
    
    if mt5_client.connect():
        print("Conectado com sucesso!")
        
        # Obter informações da conta
        account = mt5_client.get_account_info()
        if account:
            print(f"Saldo: {account['balance']:.2f}")
            print(f"Equity: {account['equity']:.2f}")
        
        # Obter cotação do EURUSD
        tick = mt5_client.get_tick("EURUSD")
        if tick:
            print(f"EURUSD - Bid: {tick['bid']:.5f}, Ask: {tick['ask']:.5f}")
        
        mt5_client.disconnect()
    else:
        print(f"Falha na conexão: {mt5_client.last_error}")