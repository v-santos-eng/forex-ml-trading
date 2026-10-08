#!/usr/bin/env python3
"""
Modo Demonstração MT5 - Sistema de Trading
Simula operações MT5 para validação do sistema sem conexão real
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MT5DemoSimulator:
    """Simulador MT5 para modo demonstração"""
    
    def __init__(self):
        self.connected = False
        self.account_info = {
            'login': 'DEMO123456',
            'balance': 10000.0,
            'equity': 10000.0,
            'margin': 0.0,
            'free_margin': 10000.0,
            'server': 'Demo-Server',
            'currency': 'USD',
            'trade_mode': 1  # Demo
        }
        self.positions = []
        self.orders = []
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        self.prices = self._generate_demo_prices()
        
    def _generate_demo_prices(self):
        """Gera preços demo para os símbolos"""
        base_prices = {
            'EURUSD': 1.0850,
            'GBPUSD': 1.2650,
            'USDJPY': 149.50,
            'AUDUSD': 0.6750,
            'USDCAD': 1.3450
        }
        
        prices = {}
        for symbol, base_price in base_prices.items():
            # Simular movimento de preços
            variation = np.random.normal(0, 0.001)
            prices[symbol] = {
                'bid': base_price + variation - 0.0002,
                'ask': base_price + variation + 0.0002,
                'last': base_price + variation,
                'time': datetime.now()
            }
        
        return prices
    
    def initialize(self):
        """Simula inicialização MT5"""
        logger.info("🔄 Inicializando modo demonstração MT5...")
        self.connected = True
        logger.info("✅ Modo demonstração ativado com sucesso!")
        return True
    
    def shutdown(self):
        """Simula desconexão MT5"""
        self.connected = False
        logger.info("🔌 Modo demonstração desconectado")
    
    def account_info_get(self):
        """Retorna informações da conta demo"""
        if not self.connected:
            return None
        return type('AccountInfo', (), self.account_info)()
    
    def symbols_get(self):
        """Retorna símbolos disponíveis"""
        if not self.connected:
            return None
        
        symbols = []
        for symbol in self.symbols:
            symbol_obj = type('Symbol', (), {
                'name': symbol,
                'description': f'{symbol} Demo',
                'currency_base': symbol[:3],
                'currency_profit': symbol[3:],
                'digits': 5 if 'JPY' not in symbol else 3,
                'point': 0.00001 if 'JPY' not in symbol else 0.001
            })()
            symbols.append(symbol_obj)
        
        return symbols
    
    def symbol_info_tick(self, symbol):
        """Retorna tick atual do símbolo"""
        if not self.connected or symbol not in self.prices:
            return None
        
        price_data = self.prices[symbol]
        tick = type('Tick', (), {
            'time': int(price_data['time'].timestamp()),
            'bid': price_data['bid'],
            'ask': price_data['ask'],
            'last': price_data['last'],
            'volume': 1
        })()
        
        return tick
    
    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        """Simula dados históricos"""
        if not self.connected:
            return None
        
        # Gerar dados históricos simulados
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=count)
        
        dates = pd.date_range(start=start_time, end=end_time, periods=count)
        base_price = self.prices.get(symbol, {}).get('last', 1.0)
        
        # Simular OHLCV
        data = []
        current_price = base_price
        
        for i, date in enumerate(dates):
            # Simular movimento de preços
            change = np.random.normal(0, 0.001)
            current_price += change
            
            high = current_price + abs(np.random.normal(0, 0.0005))
            low = current_price - abs(np.random.normal(0, 0.0005))
            open_price = current_price - np.random.normal(0, 0.0002)
            close_price = current_price
            volume = np.random.randint(100, 1000)
            
            data.append({
                'time': int(date.timestamp()),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'tick_volume': volume,
                'spread': 2,
                'real_volume': volume
            })
        
        return np.array([(d['time'], d['open'], d['high'], d['low'], d['close'], 
                         d['tick_volume'], d['spread'], d['real_volume']) for d in data],
                       dtype=[('time', 'i8'), ('open', 'f8'), ('high', 'f8'), 
                             ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8'),
                             ('spread', 'i4'), ('real_volume', 'i8')])
    
    def order_send(self, request):
        """Simula envio de ordem"""
        if not self.connected:
            return type('OrderResult', (), {
                'retcode': 10004,  # No connection
                'comment': 'No connection'
            })()
        
        # Simular execução da ordem
        order_id = len(self.orders) + 1
        
        result = type('OrderResult', (), {
            'retcode': 10009,  # Trade successful
            'deal': order_id,
            'order': order_id,
            'volume': request.get('volume', 0.1),
            'price': request.get('price', 0.0),
            'bid': self.prices.get(request.get('symbol', 'EURUSD'), {}).get('bid', 0.0),
            'ask': self.prices.get(request.get('symbol', 'EURUSD'), {}).get('ask', 0.0),
            'comment': 'Demo order executed',
            'request_id': order_id
        })()
        
        # Adicionar à lista de ordens
        self.orders.append({
            'ticket': order_id,
            'symbol': request.get('symbol'),
            'type': request.get('action'),
            'volume': request.get('volume'),
            'price': request.get('price'),
            'time': datetime.now()
        })
        
        logger.info(f"📈 Ordem demo executada: {request.get('symbol')} - {request.get('volume')} lotes")
        
        return result
    
    def positions_get(self, symbol=None):
        """Retorna posições abertas"""
        if not self.connected:
            return None
        
        # Simular algumas posições
        demo_positions = [
            {
                'ticket': 1001,
                'symbol': 'EURUSD',
                'type': 0,  # Buy
                'volume': 0.1,
                'price_open': 1.0845,
                'price_current': 1.0850,
                'profit': 5.0,
                'time': int((datetime.now() - timedelta(hours=2)).timestamp())
            },
            {
                'ticket': 1002,
                'symbol': 'GBPUSD',
                'type': 1,  # Sell
                'volume': 0.05,
                'price_open': 1.2655,
                'price_current': 1.2650,
                'profit': 2.5,
                'time': int((datetime.now() - timedelta(hours=1)).timestamp())
            }
        ]
        
        positions = []
        for pos_data in demo_positions:
            if symbol is None or pos_data['symbol'] == symbol:
                position = type('Position', (), pos_data)()
                positions.append(position)
        
        return positions
    
    def orders_get(self, symbol=None):
        """Retorna ordens pendentes"""
        if not self.connected:
            return None
        return []  # Sem ordens pendentes no demo

def run_demo_analysis():
    """Executa análise em modo demonstração"""
    print("🚀 INICIANDO SISTEMA MT5 EM MODO DEMONSTRAÇÃO")
    print("=" * 60)
    
    # Inicializar simulador
    mt5_sim = MT5DemoSimulator()
    
    if not mt5_sim.initialize():
        print("❌ Falha ao inicializar modo demonstração")
        return
    
    try:
        # Mostrar informações da conta
        account = mt5_sim.account_info_get()
        print(f"\n📊 INFORMAÇÕES DA CONTA DEMO")
        print(f"Login: {account.login}")
        print(f"Saldo: ${account.balance:,.2f}")
        print(f"Equity: ${account.equity:,.2f}")
        print(f"Servidor: {account.server}")
        print(f"Tipo: {'Demo' if account.trade_mode == 1 else 'Real'}")
        
        # Mostrar símbolos disponíveis
        symbols = mt5_sim.symbols_get()
        print(f"\n📈 SÍMBOLOS DISPONÍVEIS ({len(symbols)})")
        for symbol in symbols[:5]:
            tick = mt5_sim.symbol_info_tick(symbol.name)
            if tick:
                print(f"{symbol.name}: Bid={tick.bid:.5f}, Ask={tick.ask:.5f}")
        
        # Mostrar posições
        positions = mt5_sim.positions_get()
        print(f"\n💼 POSIÇÕES ABERTAS ({len(positions)})")
        for pos in positions:
            pos_type = "BUY" if pos.type == 0 else "SELL"
            print(f"{pos.symbol} {pos_type} {pos.volume} @ {pos.price_open:.5f} | Lucro: ${pos.profit:.2f}")
        
        # Simular análise técnica
        print(f"\n🔍 ANÁLISE TÉCNICA DEMO")
        
        # Importar forex_analyzer se disponível
        try:
            import sys
            sys.path.append('.')
            from forex_analyzer import ForexAnalyzer
            
            analyzer = ForexAnalyzer()
            
            for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
                print(f"\n--- Análise {symbol} ---")
                
                # Obter dados históricos simulados
                rates = mt5_sim.copy_rates_from_pos(symbol, 1, 0, 100)
                if rates is not None and len(rates) > 0:
                    # Converter para DataFrame
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    
                    # Calcular indicadores usando o método correto
                    try:
                        # Renomear colunas para o formato esperado pelo ForexAnalyzer
                        df_analysis = df.copy()
                        df_analysis = df_analysis.rename(columns={
                            'open': 'Open',
                            'high': 'High', 
                            'low': 'Low',
                            'close': 'Close',
                            'tick_volume': 'Volume'
                        })
                        
                        # Usar o método calculate_indicators que existe na classe
                        indicators = analyzer.calculate_indicators(df_analysis)
                        
                        current_rsi = indicators.get('rsi', 50.0)
                        current_macd = indicators.get('macd_line', 0.0)
                        current_signal = indicators.get('macd_signal', 0.0)
                        
                        print(f"RSI: {current_rsi:.2f}")
                        print(f"MACD: {current_macd:.5f}")
                        print(f"Signal: {current_signal:.5f}")
                        
                        # Gerar recomendação simples
                        if current_rsi < 30:
                            recommendation = "BUY (RSI Oversold)"
                        elif current_rsi > 70:
                            recommendation = "SELL (RSI Overbought)"
                        elif current_macd > current_signal:
                            recommendation = "BUY (MACD Bullish)"
                        else:
                            recommendation = "SELL (MACD Bearish)"
                        
                        print(f"Recomendação: {recommendation}")
                        
                    except Exception as e:
                        print(f"Erro na análise: {e}")
                        print("Recomendação: HOLD (Análise indisponível)")
                
        except ImportError:
            print("Forex Analyzer não disponível - usando análise simplificada")
            for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
                tick = mt5_sim.symbol_info_tick(symbol)
                if tick:
                    # Análise simplificada baseada em spread
                    spread = tick.ask - tick.bid
                    if spread < 0.0002:
                        recommendation = "BUY (Low Spread)"
                    else:
                        recommendation = "HOLD (High Spread)"
                    
                    print(f"\n{symbol}: {recommendation}")
                    print(f"Bid: {tick.bid:.5f}, Ask: {tick.ask:.5f}, Spread: {spread:.5f}")
        
        print(f"\n✅ MODO DEMONSTRAÇÃO EXECUTADO COM SUCESSO!")
        print("O sistema está funcionando corretamente.")
        print("\nPara usar com MT5 real:")
        print("1. Abra o MetaTrader 5")
        print("2. Faça login em uma conta")
        print("3. Execute: python mt5_main.py --mode gui")
        
    except Exception as e:
        logger.error(f"Erro durante demonstração: {e}")
        print(f"❌ Erro: {e}")
    
    finally:
        mt5_sim.shutdown()

if __name__ == "__main__":
    run_demo_analysis()