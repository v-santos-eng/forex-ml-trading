import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import time

class MT5Connector:
    """
    Classe para conexão e operações com MetaTrader 5
    """
    
    def __init__(self, login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Configura o sistema de logging"""
        logger = logging.getLogger('MT5Connector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def connect(self) -> bool:
        """
        Estabelece conexão com MT5
        """
        try:
            # Inicializar MT5
            if not mt5.initialize():
                self.logger.error(f"Falha ao inicializar MT5: {mt5.last_error()}")
                return False
            
            # Se credenciais foram fornecidas, fazer login
            if self.login and self.password and self.server:
                if not mt5.login(self.login, self.password, self.server):
                    self.logger.error(f"Falha no login MT5: {mt5.last_error()}")
                    return False
                    
            # Verificar conexão
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Não foi possível obter informações da conta")
                return False
                
            self.connected = True
            self.logger.info(f"Conectado ao MT5 - Conta: {account_info.login}, Servidor: {account_info.server}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na conexão MT5: {e}")
            return False
    
    def disconnect(self) -> None:
        """Desconecta do MT5"""
        try:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Desconectado do MT5")
        except Exception as e:
            self.logger.error(f"Erro ao desconectar: {e}")
    
    def get_account_info(self) -> Optional[Dict]:
        """Obtém informações da conta"""
        if not self.connected:
            self.logger.error("Não conectado ao MT5")
            return None
            
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return None
                
            return {
                'login': account_info.login,
                'server': account_info.server,
                'name': account_info.name,
                'company': account_info.company,
                'currency': account_info.currency,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'profit': account_info.profit
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao obter informações da conta: {e}")
            return None
    
    def get_symbols(self) -> List[str]:
        """Obtém lista de símbolos disponíveis"""
        if not self.connected:
            self.logger.error("Não conectado ao MT5")
            return []
            
        try:
            symbols = mt5.symbols_get()
            if symbols is None:
                return []
                
            return [symbol.name for symbol in symbols if symbol.visible]
            
        except Exception as e:
            self.logger.error(f"Erro ao obter símbolos: {e}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Obtém informações de um símbolo específico"""
        if not self.connected:
            self.logger.error("Não conectado ao MT5")
            return None
            
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
                
            return {
                'name': symbol_info.name,
                'description': symbol_info.description,
                'currency_base': symbol_info.currency_base,
                'currency_profit': symbol_info.currency_profit,
                'currency_margin': symbol_info.currency_margin,
                'digits': symbol_info.digits,
                'point': symbol_info.point,
                'spread': symbol_info.spread,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'trade_mode': symbol_info.trade_mode,
                'bid': symbol_info.bid,
                'ask': symbol_info.ask,
                'last': symbol_info.last,
                'time': symbol_info.time
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao obter informações do símbolo {symbol}: {e}")
            return None
    
    def get_tick_data(self, symbol: str, count: int = 1000) -> Optional[pd.DataFrame]:
        """Obtém dados de tick para um símbolo"""
        if not self.connected:
            self.logger.error("Não conectado ao MT5")
            return None
            
        try:
            ticks = mt5.copy_ticks_from(symbol, datetime.now(), count, mt5.COPY_TICKS_ALL)
            if ticks is None or len(ticks) == 0:
                return None
                
            df = pd.DataFrame(ticks)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao obter ticks para {symbol}: {e}")
            return None
    
    def get_rates(self, symbol: str, timeframe: int, count: int = 1000) -> Optional[pd.DataFrame]:
        """Obtém dados de barras/candlesticks"""
        if not self.connected:
            self.logger.error("Não conectado ao MT5")
            return None
            
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao obter rates para {symbol}: {e}")
            return None
    
    def is_market_open(self, symbol: str) -> bool:
        """Verifica se o mercado está aberto para um símbolo"""
        if not self.connected:
            return False
            
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
                
            # Verificar se o símbolo está selecionado no Market Watch
            if not symbol_info.select:
                if not mt5.symbol_select(symbol, True):
                    return False
                    
            # Verificar se há cotações recentes
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False
                
            # Verificar se a cotação é recente (últimos 5 minutos)
            current_time = datetime.now().timestamp()
            tick_time = tick.time
            
            return (current_time - tick_time) < 300  # 5 minutos
            
        except Exception as e:
            self.logger.error(f"Erro ao verificar se mercado está aberto para {symbol}: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry"""
        if self.connect():
            return self
        else:
            raise ConnectionError("Falha ao conectar com MT5")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

# Constantes de timeframe do MT5
class MT5Timeframes:
    M1 = mt5.TIMEFRAME_M1
    M5 = mt5.TIMEFRAME_M5
    M15 = mt5.TIMEFRAME_M15
    M30 = mt5.TIMEFRAME_M30
    H1 = mt5.TIMEFRAME_H1
    H4 = mt5.TIMEFRAME_H4
    D1 = mt5.TIMEFRAME_D1
    W1 = mt5.TIMEFRAME_W1
    MN1 = mt5.TIMEFRAME_MN1

if __name__ == "__main__":
    # Teste básico da conexão
    connector = MT5Connector()
    
    if connector.connect():
        print("✅ Conexão MT5 estabelecida com sucesso!")
        
        # Testar informações da conta
        account_info = connector.get_account_info()
        if account_info:
            print(f"📊 Conta: {account_info['login']} - Saldo: {account_info['balance']} {account_info['currency']}")
        
        # Testar símbolos
        symbols = connector.get_symbols()[:10]  # Primeiros 10 símbolos
        print(f"📈 Símbolos disponíveis (primeiros 10): {symbols}")
        
        # Testar dados de um símbolo
        if 'EURUSD' in symbols:
            rates = connector.get_rates('EURUSD', MT5Timeframes.M1, 10)
            if rates is not None:
                print(f"📊 Últimas 10 barras EURUSD M1:")
                print(rates[['time', 'Open', 'High', 'Low', 'Close']].tail())
        
        connector.disconnect()
    else:
        print("❌ Falha na conexão MT5")
        print("💡 Certifique-se de que o MetaTrader 5 está instalado e em execução")