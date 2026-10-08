# Guia de Integração MetaTrader 5 - Sistema Ollama Forex

## 1. CONFIGURAÇÃO DO AMBIENTE MT5

### 1.1 Requisitos de Sistema
- **MetaTrader 5 Terminal**: Versão mais recente
- **Python 3.8+**: Para integração com sistema ML
- **MetaTrader5 Python Package**: `pip install MetaTrader5`
- **Conta de Corretora**: Com permissões para trading automatizado
- **VPS/Servidor**: Windows Server 2019+ ou Ubuntu 20.04+

### 1.2 Instalação e Configuração
```bash
# Instalar dependências Python
pip install MetaTrader5 pandas numpy

# Configurar MT5 no servidor
# 1. Instalar MT5 Terminal
# 2. Fazer login na conta
# 3. Habilitar "Permitir trading automatizado"
# 4. Configurar "Permitir importação de DLL"
```

## 2. INTEGRAÇÃO PYTHON-MT5

### 2.1 Classe Base de Conexão
```python
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class MT5Connector:
    def __init__(self, login, password, server):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        
    def connect(self):
        """Estabelece conexão com MT5"""
        if not mt5.initialize():
            logging.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False
            
        if not mt5.login(self.login, self.password, self.server):
            logging.error(f"MT5 login failed: {mt5.last_error()}")
            return False
            
        self.connected = True
        logging.info("MT5 connection established")
        return True
        
    def disconnect(self):
        """Encerra conexão com MT5"""
        mt5.shutdown()
        self.connected = False
        
    def get_account_info(self):
        """Retorna informações da conta"""
        if not self.connected:
            return None
        return mt5.account_info()._asdict()
        
    def get_symbol_info(self, symbol):
        """Retorna informações do símbolo"""
        return mt5.symbol_info(symbol)._asdict()
```

### 2.2 Sistema de Dados em Tempo Real
```python
class MT5DataFeed:
    def __init__(self, connector):
        self.connector = connector
        self.subscribed_symbols = set()
        
    def subscribe_symbol(self, symbol):
        """Inscreve símbolo para dados em tempo real"""
        if mt5.symbol_select(symbol, True):
            self.subscribed_symbols.add(symbol)
            return True
        return False
        
    def get_tick_data(self, symbol, count=1):
        """Obtém dados de tick em tempo real"""
        ticks = mt5.copy_ticks_from(symbol, datetime.now(), count, mt5.COPY_TICKS_ALL)
        if ticks is not None:
            return pd.DataFrame(ticks)
        return None
        
    def get_rates_data(self, symbol, timeframe, count=100):
        """Obtém dados de barras/candlesticks"""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is not None:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        return None
        
    def get_current_price(self, symbol):
        """Obtém preço atual (bid/ask)"""
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None:
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'time': datetime.fromtimestamp(tick.time)
            }
        return None
```

### 2.3 Sistema de Execução de Ordens
```python
class MT5OrderManager:
    def __init__(self, connector):
        self.connector = connector
        
    def send_order(self, symbol, order_type, volume, price=None, sl=None, tp=None, comment=""):
        """Envia ordem para o mercado"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return None
            
        # Configurar request de ordem
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "deviation": 20,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Adicionar preço se necessário
        if price is not None:
            request["price"] = price
            
        # Adicionar stop loss e take profit
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
            
        # Enviar ordem
        result = mt5.order_send(request)
        return result
        
    def buy_market(self, symbol, volume, sl=None, tp=None):
        """Ordem de compra a mercado"""
        return self.send_order(symbol, mt5.ORDER_TYPE_BUY, volume, sl=sl, tp=tp)
        
    def sell_market(self, symbol, volume, sl=None, tp=None):
        """Ordem de venda a mercado"""
        return self.send_order(symbol, mt5.ORDER_TYPE_SELL, volume, sl=sl, tp=tp)
        
    def get_positions(self):
        """Retorna posições abertas"""
        positions = mt5.positions_get()
        if positions is not None:
            return pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
        return pd.DataFrame()
        
    def close_position(self, ticket):
        """Fecha posição específica"""
        position = mt5.positions_get(ticket=ticket)
        if position is None:
            return None
            
        position = position[0]
        
        # Configurar request de fechamento
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
        else:
            order_type = mt5.ORDER_TYPE_BUY
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        return mt5.order_send(request)
```

## 3. INTEGRAÇÃO COM SISTEMA ML

### 3.1 Classe de Integração ML-MT5
```python
from ml_predictor_local import MLForexPredictorLocal

class MT5MLTrader:
    def __init__(self, mt5_connector, ml_predictor):
        self.mt5 = mt5_connector
        self.ml = ml_predictor
        self.order_manager = MT5OrderManager(mt5_connector)
        self.data_feed = MT5DataFeed(mt5_connector)
        self.risk_manager = RiskManager()
        
    def analyze_and_trade(self, symbol):
        """Análise ML e execução de trade"""
        try:
            # Obter dados em tempo real
            rates_data = self.data_feed.get_rates_data(symbol, mt5.TIMEFRAME_M15, 100)
            if rates_data is None:
                return None
                
            # Fazer predição ML
            direction, confidence = self.ml.predict_direction(symbol)
            
            if direction is None or confidence < 0.6:
                return None
                
            # Calcular tamanho da posição
            account_info = self.mt5.get_account_info()
            position_size = self.risk_manager.calculate_position_size(
                account_info['balance'], symbol, confidence
            )
            
            # Obter preço atual
            current_price = self.data_feed.get_current_price(symbol)
            if current_price is None:
                return None
                
            # Calcular stop loss e take profit
            sl, tp = self.risk_manager.calculate_sl_tp(
                current_price, direction, symbol
            )
            
            # Executar ordem
            if direction == 1:  # Compra
                result = self.order_manager.buy_market(symbol, position_size, sl, tp)
            else:  # Venda
                result = self.order_manager.sell_market(symbol, position_size, sl, tp)
                
            return result
            
        except Exception as e:
            logging.error(f"Erro em analyze_and_trade: {e}")
            return None
```

### 3.2 Sistema de Gerenciamento de Risco
```python
class RiskManager:
    def __init__(self, max_risk_per_trade=0.02, max_drawdown=0.15):
        self.max_risk_per_trade = max_risk_per_trade  # 2% por trade
        self.max_drawdown = max_drawdown  # 15% drawdown máximo
        
    def calculate_position_size(self, balance, symbol, confidence):
        """Calcula tamanho da posição baseado no risco"""
        # Risco ajustado pela confiança
        adjusted_risk = self.max_risk_per_trade * confidence
        
        # Valor em risco
        risk_amount = balance * adjusted_risk
        
        # Obter informações do símbolo
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0.01  # Volume mínimo
            
        # Calcular tamanho da posição
        # Assumindo stop loss de 50 pips
        pip_value = symbol_info.trade_tick_value
        stop_loss_pips = 50
        
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Ajustar para volume mínimo/máximo
        min_volume = symbol_info.volume_min
        max_volume = symbol_info.volume_max
        
        position_size = max(min_volume, min(position_size, max_volume))
        
        return round(position_size, 2)
        
    def calculate_sl_tp(self, current_price, direction, symbol):
        """Calcula stop loss e take profit"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return None, None
            
        point = symbol_info.point
        
        if direction == 1:  # Compra
            sl = current_price['bid'] - (50 * point)  # 50 pips SL
            tp = current_price['bid'] + (100 * point)  # 100 pips TP
        else:  # Venda
            sl = current_price['ask'] + (50 * point)  # 50 pips SL
            tp = current_price['ask'] - (100 * point)  # 100 pips TP
            
        return sl, tp
        
    def check_drawdown(self, initial_balance, current_balance):
        """Verifica se drawdown excede limite"""
        drawdown = (initial_balance - current_balance) / initial_balance
        return drawdown > self.max_drawdown
```

## 4. SISTEMA DE MONITORAMENTO

### 4.1 Monitor de Performance
```python
class MT5PerformanceMonitor:
    def __init__(self, mt5_connector):
        self.mt5 = mt5_connector
        self.initial_balance = None
        
    def start_monitoring(self):
        """Inicia monitoramento"""
        account_info = self.mt5.get_account_info()
        self.initial_balance = account_info['balance']
        
    def get_performance_metrics(self):
        """Calcula métricas de performance"""
        account_info = self.mt5.get_account_info()
        current_balance = account_info['balance']
        
        # Calcular métricas
        total_return = (current_balance - self.initial_balance) / self.initial_balance
        drawdown = max(0, (self.initial_balance - current_balance) / self.initial_balance)
        
        # Obter histórico de trades
        trades = self.get_trade_history()
        
        if len(trades) > 0:
            win_rate = len(trades[trades['profit'] > 0]) / len(trades)
            avg_profit = trades['profit'].mean()
            max_loss = trades['profit'].min()
        else:
            win_rate = 0
            avg_profit = 0
            max_loss = 0
            
        return {
            'total_return': total_return,
            'current_balance': current_balance,
            'drawdown': drawdown,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'max_loss': max_loss,
            'total_trades': len(trades)
        }
        
    def get_trade_history(self, days=30):
        """Obtém histórico de trades"""
        from_date = datetime.now() - timedelta(days=days)
        deals = mt5.history_deals_get(from_date, datetime.now())
        
        if deals is not None:
            return pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
        return pd.DataFrame()
```

## 5. SCRIPT PRINCIPAL DE EXECUÇÃO

### 5.1 Main Trading Bot
```python
import time
import schedule
from datetime import datetime

class MT5TradingBot:
    def __init__(self, login, password, server):
        # Inicializar componentes
        self.connector = MT5Connector(login, password, server)
        self.ml_predictor = MLForexPredictorLocal()
        self.trader = MT5MLTrader(self.connector, self.ml_predictor)
        self.monitor = MT5PerformanceMonitor(self.connector)
        
        # Símbolos para trading
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        
        # Estado do bot
        self.running = False
        
    def start(self):
        """Inicia o bot de trading"""
        if not self.connector.connect():
            logging.error("Falha ao conectar com MT5")
            return False
            
        # Inscrever símbolos
        for symbol in self.symbols:
            self.trader.data_feed.subscribe_symbol(symbol)
            
        # Iniciar monitoramento
        self.monitor.start_monitoring()
        
        self.running = True
        logging.info("Trading bot iniciado")
        
        # Agendar execuções
        schedule.every(15).minutes.do(self.trading_cycle)
        schedule.every(1).hours.do(self.log_performance)
        
        return True
        
    def trading_cycle(self):
        """Ciclo principal de trading"""
        if not self.running:
            return
            
        logging.info("Iniciando ciclo de trading")
        
        for symbol in self.symbols:
            try:
                result = self.trader.analyze_and_trade(symbol)
                if result is not None:
                    logging.info(f"Trade executado para {symbol}: {result}")
            except Exception as e:
                logging.error(f"Erro no trading de {symbol}: {e}")
                
    def log_performance(self):
        """Log de performance"""
        metrics = self.monitor.get_performance_metrics()
        logging.info(f"Performance: {metrics}")
        
        # Parar se drawdown muito alto
        if metrics['drawdown'] > 0.15:
            logging.warning("Drawdown alto - parando trading")
            self.stop()
            
    def stop(self):
        """Para o bot"""
        self.running = False
        self.connector.disconnect()
        logging.info("Trading bot parado")
        
    def run(self):
        """Loop principal"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Verificar a cada minuto

# Execução
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('mt5_trading.log'),
            logging.StreamHandler()
        ]
    )
    
    # Credenciais MT5 (usar variáveis de ambiente)
    LOGIN = 123456789
    PASSWORD = "sua_senha"
    SERVER = "MetaQuotes-Demo"
    
    # Iniciar bot
    bot = MT5TradingBot(LOGIN, PASSWORD, SERVER)
    
    if bot.start():
        try:
            bot.run()
        except KeyboardInterrupt:
            logging.info("Interrompido pelo usuário")
        finally:
            bot.stop()
    else:
        logging.error("Falha ao iniciar bot")
```

## 6. CONFIGURAÇÃO DE PRODUÇÃO

### 6.1 Arquivo de Configuração
```python
# config_mt5.py
class MT5Config:
    # Credenciais MT5
    MT5_LOGIN = 123456789
    MT5_PASSWORD = "sua_senha"
    MT5_SERVER = "MetaQuotes-Demo"
    
    # Símbolos para trading
    TRADING_SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    
    # Parâmetros de risco
    MAX_RISK_PER_TRADE = 0.02  # 2%
    MAX_DRAWDOWN = 0.15  # 15%
    
    # Parâmetros ML
    MIN_CONFIDENCE = 0.6
    
    # Timeframes
    ANALYSIS_TIMEFRAME = 15  # minutos
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "mt5_trading.log"
```

### 6.2 Dockerfile para Deploy
```dockerfile
FROM python:3.9-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    wget \
    wine \
    xvfb

# Instalar MT5 via Wine
RUN wget https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe
RUN wine mt5setup.exe /S

# Copiar código
COPY . /app
WORKDIR /app

# Instalar dependências Python
RUN pip install -r requirements.txt

# Comando de execução
CMD ["python", "mt5_trading_bot.py"]
```

## 7. TESTES E VALIDAÇÃO

### 7.1 Script de Teste
```python
# test_mt5_integration.py
import unittest
from mt5_integration import MT5Connector, MT5DataFeed, MT5OrderManager

class TestMT5Integration(unittest.TestCase):
    def setUp(self):
        self.connector = MT5Connector(123456, "password", "demo-server")
        
    def test_connection(self):
        """Testa conexão com MT5"""
        result = self.connector.connect()
        self.assertTrue(result)
        
    def test_data_feed(self):
        """Testa feed de dados"""
        data_feed = MT5DataFeed(self.connector)
        data = data_feed.get_current_price('EURUSD')
        self.assertIsNotNone(data)
        
    def test_order_management(self):
        """Testa sistema de ordens"""
        order_manager = MT5OrderManager(self.connector)
        positions = order_manager.get_positions()
        self.assertIsInstance(positions, pd.DataFrame)
        
if __name__ == '__main__':
    unittest.main()
```

## 8. MONITORAMENTO E ALERTAS

### 8.1 Sistema de Alertas
```python
import smtplib
from email.mime.text import MIMEText

class AlertSystem:
    def __init__(self, smtp_server, email, password):
        self.smtp_server = smtp_server
        self.email = email
        self.password = password
        
    def send_alert(self, subject, message, to_email):
        """Envia alerta por email"""
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = self.email
        msg['To'] = to_email
        
        with smtplib.SMTP(self.smtp_server, 587) as server:
            server.starttls()
            server.login(self.email, self.password)
            server.send_message(msg)
            
    def check_and_alert(self, metrics):
        """Verifica métricas e envia alertas"""
        if metrics['drawdown'] > 0.10:
            self.send_alert(
                "ALERTA: Drawdown Alto",
                f"Drawdown atual: {metrics['drawdown']:.2%}",
                "admin@empresa.com"
            )
            
        if metrics['win_rate'] < 0.30:
            self.send_alert(
                "ALERTA: Win Rate Baixo",
                f"Win rate atual: {metrics['win_rate']:.2%}",
                "admin@empresa.com"
            )
```

## PRÓXIMOS PASSOS

1. **Configurar Conta Demo**: Abrir conta demo no MT5
2. **Testar Conexão**: Validar conectividade Python-MT5
3. **Implementar Gradualmente**: Começar com dados, depois ordens
4. **Backtesting**: Testar estratégias em dados históricos
5. **Paper Trading**: Simular trades sem capital real
6. **Deploy Gradual**: Começar com capital mínimo

**AVISO CRÍTICO**: Este sistema envolve riscos financeiros significativos. Sempre teste extensivamente em contas demo antes de usar capital real. Nunca invista mais do que pode perder.