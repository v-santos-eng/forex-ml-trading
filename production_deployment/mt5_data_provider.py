import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List, Optional, Callable, Any
import logging
from mt5_connector import MT5Connector, MT5Timeframes

class MT5DataProvider:
    """
    Provedor de dados em tempo real do MetaTrader 5
    """
    
    def __init__(self, connector: MT5Connector):
        self.connector = connector
        self.logger = self._setup_logger()
        self.subscriptions = {}  # {symbol: {callbacks: [], last_update: datetime}}
        self.price_callbacks = {}  # Compatibilidade com StreamingAdapter
        self.running = False
        self.update_thread = None
        self.update_interval = 1.0  # segundos
        
    def _setup_logger(self) -> logging.Logger:
        """Configura o sistema de logging"""
        logger = logging.getLogger('MT5DataProvider')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Obtém preço atual de um símbolo"""
        if not self.connector.connected:
            self.logger.error("MT5 não conectado")
            return None
            
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.warning(f"Não foi possível obter tick para {symbol}")
                return None
                
            return {
                'symbol': symbol,
                'bid': float(tick.bid),
                'ask': float(tick.ask),
                'last': float(tick.last),
                'spread': float(tick.ask - tick.bid),
                'time': datetime.fromtimestamp(tick.time),
                'volume': int(tick.volume)
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao obter preço atual para {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: int, count: int = 1000, 
                          start_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Obtém dados históricos"""
        if not self.connector.connected:
            self.logger.error("MT5 não conectado")
            return None
            
        try:
            if start_date:
                rates = mt5.copy_rates_from(symbol, timeframe, start_date, count)
            else:
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
                
            if rates is None or len(rates) == 0:
                self.logger.warning(f"Nenhum dado histórico encontrado para {symbol}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Manter colunas em minúsculas para compatibilidade com MT5SignalIntegrator
            # Não renomear - manter formato original do MT5
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao obter dados históricos para {symbol}: {e}")
            return None
    
    def subscribe_to_symbol(self, symbol: str, callback: Callable[[Dict], None]) -> bool:
        """Inscreve-se para receber atualizações de preço de um símbolo"""
        try:
            if symbol not in self.subscriptions:
                self.subscriptions[symbol] = {
                    'callbacks': [],
                    'last_update': None,
                    'last_price': None
                }
            
            self.subscriptions[symbol]['callbacks'].append(callback)
            self.logger.info(f"Inscrito para receber atualizações de {symbol}")
            
            # Iniciar thread de atualização se não estiver rodando
            if not self.running:
                self.start_real_time_updates()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao se inscrever para {symbol}: {e}")
            return False
    
    def unsubscribe_from_symbol(self, symbol: str, callback: Optional[Callable] = None) -> bool:
        """Remove inscrição de um símbolo"""
        try:
            if symbol not in self.subscriptions:
                return True
                
            if callback:
                # Remove callback específico
                if callback in self.subscriptions[symbol]['callbacks']:
                    self.subscriptions[symbol]['callbacks'].remove(callback)
            else:
                # Remove todos os callbacks
                self.subscriptions[symbol]['callbacks'].clear()
                
            # Remove símbolo se não há mais callbacks
            if not self.subscriptions[symbol]['callbacks']:
                del self.subscriptions[symbol]
                
            self.logger.info(f"Desinscrito de {symbol}")
            
            # Para atualizações se não há mais inscrições
            if not self.subscriptions and self.running:
                self.stop_real_time_updates()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao se desinscrever de {symbol}: {e}")
            return False
    
    def start_real_time_updates(self) -> None:
        """Inicia atualizações em tempo real"""
        if self.running:
            return
            
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        self.logger.info("Atualizações em tempo real iniciadas")
    
    def stop_real_time_updates(self) -> None:
        """Para atualizações em tempo real"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        self.logger.info("Atualizações em tempo real paradas")
    
    def _update_loop(self) -> None:
        """Loop principal de atualizações"""
        while self.running:
            try:
                for symbol in list(self.subscriptions.keys()):
                    self._update_symbol(symbol)
                    
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de atualizações: {e}")
                time.sleep(1)
    
    def _update_symbol(self, symbol: str) -> None:
        """Atualiza dados de um símbolo específico"""
        try:
            current_price = self.get_current_price(symbol)
            if current_price is None:
                return
                
            subscription = self.subscriptions[symbol]
            
            # Verificar se houve mudança significativa no preço
            if subscription['last_price']:
                price_change = abs(current_price['last'] - subscription['last_price']['last'])
                min_change = 0.00001  # Mudança mínima para notificar
                
                if price_change < min_change:
                    return
            
            # Atualizar dados da inscrição
            subscription['last_update'] = datetime.now()
            subscription['last_price'] = current_price
            
            # Notificar todos os callbacks
            for callback in subscription['callbacks']:
                try:
                    callback(current_price)
                except Exception as e:
                    self.logger.error(f"Erro no callback para {symbol}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Erro ao atualizar {symbol}: {e}")
    
    def get_market_depth(self, symbol: str) -> Optional[Dict]:
        """Obtém profundidade de mercado (book de ofertas)"""
        if not self.connector.connected:
            return None
            
        try:
            # MT5 não fornece book completo via API Python
            # Retornamos bid/ask como aproximação
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
                
            return {
                'symbol': symbol,
                'bids': [{'price': tick.bid, 'volume': tick.volume}],
                'asks': [{'price': tick.ask, 'volume': tick.volume}],
                'time': datetime.fromtimestamp(tick.time)
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao obter market depth para {symbol}: {e}")
            return None
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """Obtém preços de múltiplos símbolos"""
        prices = {}
        
        for symbol in symbols:
            price = self.get_current_price(symbol)
            if price:
                prices[symbol] = price
                
        return prices
    
    def calculate_spread_percentage(self, symbol: str) -> Optional[float]:
        """Calcula spread em percentual"""
        price = self.get_current_price(symbol)
        if not price:
            return None
            
        try:
            spread_pct = (price['spread'] / price['bid']) * 100
            return round(spread_pct, 4)
        except:
            return None
    
    def is_symbol_trading(self, symbol: str) -> bool:
        """Verifica se um símbolo está sendo negociado"""
        return self.connector.is_market_open(symbol)
    
    def get_trading_sessions(self, symbol: str) -> Optional[Dict]:
        """Obtém informações sobre sessões de trading"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
                
            return {
                'symbol': symbol,
                'trade_mode': symbol_info.trade_mode,
                'trade_execution': symbol_info.trade_execution,
                'trade_stops_level': symbol_info.trade_stops_level,
                'trade_freeze_level': symbol_info.trade_freeze_level,
                'session_deals': symbol_info.session_deals,
                'session_buy_orders': symbol_info.session_buy_orders,
                'session_sell_orders': symbol_info.session_sell_orders
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao obter sessões de trading para {symbol}: {e}")
            return None
    
    def __del__(self):
        """Destructor - para atualizações em tempo real"""
        if self.running:
            self.stop_real_time_updates()

class RealTimeDataManager:
    """
    Gerenciador de dados em tempo real com cache e otimizações
    """
    
    def __init__(self, data_provider: MT5DataProvider):
        self.data_provider = data_provider
        self.cache = {}  # Cache de dados recentes
        self.cache_ttl = 5  # TTL do cache em segundos
        self.logger = logging.getLogger('RealTimeDataManager')
    
    def get_cached_price(self, symbol: str) -> Optional[Dict]:
        """Obtém preço do cache se disponível e válido"""
        if symbol in self.cache:
            cached_data = self.cache[symbol]
            age = (datetime.now() - cached_data['timestamp']).total_seconds()
            
            if age < self.cache_ttl:
                return cached_data['data']
        
        # Cache expirado ou não existe, buscar novo
        price = self.data_provider.get_current_price(symbol)
        if price:
            self.cache[symbol] = {
                'data': price,
                'timestamp': datetime.now()
            }
        
        return price
    
    def clear_cache(self) -> None:
        """Limpa o cache"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Obtém estatísticas do cache"""
        return {
            'cached_symbols': len(self.cache),
            'symbols': list(self.cache.keys()),
            'cache_ttl': self.cache_ttl
        }

if __name__ == "__main__":
    # Teste do provedor de dados
    connector = MT5Connector()
    
    if connector.connect():
        print("✅ Conectado ao MT5")
        
        data_provider = MT5DataProvider(connector)
        
        # Testar preço atual
        price = data_provider.get_current_price('EURUSD')
        if price:
            print(f"💰 EURUSD: Bid={price['bid']}, Ask={price['ask']}, Spread={price['spread']:.5f}")
        
        # Testar dados históricos
        historical = data_provider.get_historical_data('EURUSD', MT5Timeframes.M1, 10)
        if historical is not None:
            print(f"📊 Dados históricos EURUSD (últimas 10 barras):")
            print(historical[['Open', 'High', 'Low', 'Close']].tail())
        
        # Testar múltiplos preços
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        prices = data_provider.get_multiple_prices(symbols)
        print(f"📈 Preços múltiplos: {len(prices)} símbolos obtidos")
        
        connector.disconnect()
    else:
        print("❌ Falha na conexão MT5")