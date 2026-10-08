import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import threading
import time
import json
from dataclasses import dataclass, asdict
from enum import Enum
from mt5_connector import MT5Connector
from mt5_order_manager import MT5OrderManager, OrderType, OrderFilling
from mt5_data_provider import MT5DataProvider
from mt5_risk_manager import MT5RiskManager, RiskLevel
from forex_analyzer import ForexAnalyzer

class SignalStrength(Enum):
    """Força do sinal de trading"""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"

class SignalAction(Enum):
    """Ação do sinal"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"

@dataclass
class TradingSignal:
    """Sinal de trading"""
    symbol: str
    action: SignalAction
    strength: SignalStrength
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    volume: Optional[float] = None
    timestamp: datetime = None
    indicators: Dict[str, Any] = None
    reason: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.indicators is None:
            self.indicators = {}

@dataclass
class SignalConfig:
    """Configuração de sinais"""
    min_confidence: float = 0.6  # Confiança mínima para executar
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_threshold: float = 0.0001
    bb_threshold: float = 0.8  # Posição nas Bollinger Bands
    volume_multiplier: float = 1.0  # Multiplicador de volume
    max_spread_pips: float = 3.0  # Spread máximo em pips
    signal_timeout: int = 300  # Timeout do sinal em segundos
    enable_auto_trading: bool = False
    symbols_enabled: List[str] = None
    
    def __post_init__(self):
        if self.symbols_enabled is None:
            self.symbols_enabled = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD']

class MT5SignalIntegrator:
    """
    Integrador de sinais entre forex_analyzer e MT5
    """
    
    def __init__(self, connector: MT5Connector, order_manager: MT5OrderManager,
                 data_provider: MT5DataProvider, risk_manager: MT5RiskManager,
                 config: Optional[SignalConfig] = None):
        self.connector = connector
        self.order_manager = order_manager
        self.data_provider = data_provider
        self.risk_manager = risk_manager
        self.config = config or SignalConfig()
        self.logger = self._setup_logger()
        
        # Inicializar analisador Forex
        self.forex_analyzer = ForexAnalyzer()
        
        # Estado do integrador
        self.running = False
        self.analysis_thread = None
        self.signal_history = []
        self.active_signals = {}
        self.performance_stats = {
            'total_signals': 0,
            'executed_trades': 0,
            'successful_trades': 0,
            'total_profit': 0.0,
            'win_rate': 0.0
        }
        
        # Callbacks
        self.signal_callbacks = []
        self.trade_callbacks = []
        
    def _setup_logger(self) -> logging.Logger:
        """Configura o sistema de logging"""
        logger = logging.getLogger('MT5SignalIntegrator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def add_signal_callback(self, callback) -> None:
        """Adiciona callback para novos sinais"""
        self.signal_callbacks.append(callback)
    
    def add_trade_callback(self, callback) -> None:
        """Adiciona callback para trades executados"""
        self.trade_callbacks.append(callback)
    
    def _notify_signal(self, signal: TradingSignal) -> None:
        """Notifica callbacks sobre novo sinal"""
        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                self.logger.error(f"Erro no callback de sinal: {e}")
    
    def _notify_trade(self, signal: TradingSignal, result: Dict) -> None:
        """Notifica callbacks sobre trade executado"""
        for callback in self.trade_callbacks:
            try:
                callback(signal, result)
            except Exception as e:
                self.logger.error(f"Erro no callback de trade: {e}")
    
    def analyze_market(self) -> List[TradingSignal]:
        """Analisa mercado e gera sinais"""
        signals = []
        
        try:
            for symbol in self.config.symbols_enabled:
                # Verificar se símbolo está disponível
                if not mt5.symbol_select(symbol, True):
                    continue
                    
                # Obter dados históricos
                rates = self.data_provider.get_historical_data(symbol, mt5.TIMEFRAME_H1, 100)
                if rates is None or len(rates) < 50:
                    continue
                    
                # Analisar com forex_analyzer
                analysis = self._analyze_symbol_with_indicators(symbol, rates)
                if analysis is None:
                    continue
                    
                # Gerar sinal baseado na análise
                signal = self._generate_signal_from_analysis(symbol, analysis)
                if signal is not None:
                    signals.append(signal)
                    
        except Exception as e:
            self.logger.error(f"Erro na análise de mercado: {e}")
            
        return signals
    
    def _analyze_symbol_with_indicators(self, symbol: str, rates: pd.DataFrame) -> Optional[Dict]:
        """Analisa símbolo usando indicadores técnicos"""
        try:
            # Converter para formato esperado pelo forex_analyzer
            df = rates.reset_index()  # Reset index para ter coluna 'time'
            
            # Padronizar nomes das colunas para minúsculas
            column_mapping = {
                'time': 'time',
                'Open': 'open', 'open': 'open',
                'High': 'high', 'high': 'high', 
                'Low': 'low', 'low': 'low',
                'Close': 'close', 'close': 'close',
                'Volume': 'tick_volume', 'tick_volume': 'tick_volume',
                'spread': 'spread',
                'real_volume': 'real_volume'
            }
            
            # Renomear colunas usando o mapeamento
            df = df.rename(columns=column_mapping)
            
            # Verificar se temos todas as colunas necessárias
            required_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Colunas ausentes para {symbol}: {missing_cols}")
                return None
            
            # Calcular indicadores
            indicators = {}
            
            # RSI
            rsi = self._calculate_rsi(df['close'])
            indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50.0
            
            # MACD
            macd_line, macd_signal, macd_hist = self._calculate_macd(df['close'])
            indicators['macd'] = {
                'line': macd_line.iloc[-1] if not macd_line.empty else 0.0,
                'signal': macd_signal.iloc[-1] if not macd_signal.empty else 0.0,
                'histogram': macd_hist.iloc[-1] if not macd_hist.empty else 0.0
            }
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
            current_price = df['close'].iloc[-1]
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            indicators['bollinger'] = {
                'position': bb_position,
                'upper': bb_upper.iloc[-1],
                'middle': bb_middle.iloc[-1],
                'lower': bb_lower.iloc[-1]
            }
            
            # Moving Averages
            sma_20 = df['close'].rolling(20).mean()
            sma_50 = df['close'].rolling(50).mean()
            indicators['sma'] = {
                'sma_20': sma_20.iloc[-1] if not sma_20.empty else current_price,
                'sma_50': sma_50.iloc[-1] if not sma_50.empty else current_price
            }
            
            # Preço atual
            indicators['current_price'] = current_price
            
            return {
                'symbol': symbol,
                'indicators': indicators,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise de {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def _generate_signal_from_analysis(self, symbol: str, analysis: Dict) -> Optional[TradingSignal]:
        """Gera sinal de trading baseado na análise"""
        try:
            indicators = analysis['indicators']
            
            # Calcular confiança e ação baseada nos indicadores
            signals_count = 0
            buy_signals = 0
            sell_signals = 0
            
            # Sinal RSI
            rsi = indicators['rsi']
            if rsi < self.config.rsi_oversold:
                buy_signals += 1
                signals_count += 1
            elif rsi > self.config.rsi_overbought:
                sell_signals += 1
                signals_count += 1
            
            # Sinal MACD
            macd = indicators['macd']
            if macd['line'] > macd['signal'] and macd['histogram'] > 0:
                buy_signals += 1
                signals_count += 1
            elif macd['line'] < macd['signal'] and macd['histogram'] < 0:
                sell_signals += 1
                signals_count += 1
            
            # Sinal Bollinger Bands
            bb = indicators['bollinger']
            if bb['position'] < (1 - self.config.bb_threshold):
                buy_signals += 1
                signals_count += 1
            elif bb['position'] > self.config.bb_threshold:
                sell_signals += 1
                signals_count += 1
            
            # Sinal Moving Averages
            sma = indicators['sma']
            current_price = indicators['current_price']
            if current_price > sma['sma_20'] > sma['sma_50']:
                buy_signals += 1
                signals_count += 1
            elif current_price < sma['sma_20'] < sma['sma_50']:
                sell_signals += 1
                signals_count += 1
            
            # Determinar ação e confiança
            if signals_count == 0:
                return None
                
            if buy_signals > sell_signals:
                action = SignalAction.BUY
                confidence = buy_signals / signals_count
            elif sell_signals > buy_signals:
                action = SignalAction.SELL
                confidence = sell_signals / signals_count
            else:
                action = SignalAction.HOLD
                confidence = 0.5
            
            # Verificar confiança mínima
            if confidence < self.config.min_confidence:
                return None
            
            # Determinar força do sinal
            if confidence >= 0.9:
                strength = SignalStrength.VERY_STRONG
            elif confidence >= 0.75:
                strength = SignalStrength.STRONG
            elif confidence >= 0.6:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Calcular preços de entrada, SL e TP
            entry_price = current_price
            
            # Obter informações do símbolo para calcular pips
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
                
            pip_size = symbol_info.point * 10 if symbol_info.digits == 5 or symbol_info.digits == 3 else symbol_info.point
            
            if action == SignalAction.BUY:
                stop_loss = entry_price - (50 * pip_size)  # 50 pips SL
                take_profit = entry_price + (100 * pip_size)  # 100 pips TP
            elif action == SignalAction.SELL:
                stop_loss = entry_price + (50 * pip_size)  # 50 pips SL
                take_profit = entry_price - (100 * pip_size)  # 100 pips TP
            else:
                stop_loss = None
                take_profit = None
            
            # Calcular volume baseado no risco
            volume = self.risk_manager.calculate_position_size(symbol, 
                                                             self.risk_manager.risk_params.max_risk_per_trade * 
                                                             mt5.account_info().balance if mt5.account_info() else 1000,
                                                             50.0)  # 50 pips de risco
            
            # Criar razão do sinal
            reason_parts = []
            if rsi < self.config.rsi_oversold:
                reason_parts.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > self.config.rsi_overbought:
                reason_parts.append(f"RSI overbought ({rsi:.1f})")
            
            if macd['histogram'] > 0:
                reason_parts.append("MACD bullish")
            elif macd['histogram'] < 0:
                reason_parts.append("MACD bearish")
            
            reason = "; ".join(reason_parts) if reason_parts else "Multiple indicators alignment"
            
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                strength=strength,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=volume,
                indicators=indicators,
                reason=reason
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar sinal para {symbol}: {e}")
            return None
    
    def execute_signal(self, signal: TradingSignal) -> Optional[Dict]:
        """Executa um sinal de trading"""
        try:
            # Verificar se auto-trading está habilitado
            if not self.config.enable_auto_trading:
                self.logger.info(f"Auto-trading desabilitado. Sinal {signal.symbol} {signal.action.value} não executado.")
                return None
            
            # Verificar spread
            current_price = self.data_provider.get_current_price(signal.symbol)
            if current_price is None:
                return None
                
            spread = current_price['ask'] - current_price['bid']
            symbol_info = mt5.symbol_info(signal.symbol)
            if symbol_info is None:
                return None
                
            pip_size = symbol_info.point * 10 if symbol_info.digits == 5 or symbol_info.digits == 3 else symbol_info.point
            spread_pips = spread / pip_size
            
            if spread_pips > self.config.max_spread_pips:
                self.logger.warning(f"Spread muito alto para {signal.symbol}: {spread_pips:.1f} pips")
                return None
            
            # Validar com gerenciador de risco
            order_type = OrderType.BUY if signal.action == SignalAction.BUY else OrderType.SELL
            is_valid, validation_message = self.risk_manager.validate_trade(
                signal.symbol, order_type, signal.volume, signal.stop_loss, signal.take_profit
            )
            
            if not is_valid:
                self.logger.warning(f"Trade rejeitado pelo gerenciador de risco: {validation_message}")
                return None
            
            # Executar ordem
            if signal.action == SignalAction.BUY:
                result = self.order_manager.place_market_order(
                    symbol=signal.symbol,
                    order_type=OrderType.BUY,
                    volume=signal.volume,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    comment=f"Signal: {signal.reason[:20]}"
                )
            elif signal.action == SignalAction.SELL:
                result = self.order_manager.place_market_order(
                    symbol=signal.symbol,
                    order_type=OrderType.SELL,
                    volume=signal.volume,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    comment=f"Signal: {signal.reason[:20]}"
                )
            else:
                return None
            
            # Verificar se o trade foi executado com sucesso
            # retcode 0 também indica sucesso (SUCCESS)
            if result and result.get('retcode') in [0, mt5.TRADE_RETCODE_DONE]:
                self.logger.info(f"✅ Trade executado: {signal.symbol} {signal.action.value} {signal.volume} lotes")
                
                # Atualizar estatísticas
                self.performance_stats['executed_trades'] += 1
                
                # Notificar callbacks
                self._notify_trade(signal, result)
                
                return result
            else:
                error_msg = result.get('comment', 'Erro desconhecido') if result else 'Falha na execução'
                self.logger.error(f"❌ Falha ao executar trade: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"Erro ao executar sinal: {e}")
            return None
    
    def process_signals(self, signals: List[TradingSignal]) -> List[Dict]:
        """Processa lista de sinais"""
        results = []
        
        for signal in signals:
            # Adicionar ao histórico
            self.signal_history.append(signal)
            self.performance_stats['total_signals'] += 1
            
            # Notificar callbacks
            self._notify_signal(signal)
            
            # Executar se configurado
            if signal.action in [SignalAction.BUY, SignalAction.SELL]:
                result = self.execute_signal(signal)
                if result:
                    results.append(result)
                    
        return results
    
    def start_analysis_loop(self, interval: float = 300.0) -> None:
        """Inicia loop de análise contínua"""
        if self.running:
            return
            
        self.running = True
        
        def analysis_loop():
            while self.running:
                try:
                    # Analisar mercado
                    signals = self.analyze_market()
                    
                    if signals:
                        self.logger.info(f"📊 {len(signals)} sinais gerados")
                        
                        # Processar sinais
                        results = self.process_signals(signals)
                        
                        if results:
                            self.logger.info(f"💼 {len(results)} trades executados")
                    
                    # Aguardar próximo ciclo
                    time.sleep(interval)
                    
                except Exception as e:
                    self.logger.error(f"Erro no loop de análise: {e}")
                    time.sleep(30)  # Aguardar menos tempo em caso de erro
                    
        self.analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
        self.analysis_thread.start()
        self.logger.info(f"🚀 Loop de análise iniciado (intervalo: {interval}s)")
    
    def stop_analysis_loop(self) -> None:
        """Para loop de análise"""
        self.running = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=10)
        self.logger.info("⏹️ Loop de análise parado")
    
    def get_performance_summary(self) -> Dict:
        """Obtém resumo de performance"""
        try:
            # Calcular win rate
            if self.performance_stats['executed_trades'] > 0:
                self.performance_stats['win_rate'] = (self.performance_stats['successful_trades'] / 
                                                    self.performance_stats['executed_trades']) * 100
            
            # Adicionar estatísticas recentes
            recent_signals = [s for s in self.signal_history if (datetime.now() - s.timestamp).seconds < 3600]
            
            return {
                **self.performance_stats,
                'recent_signals_1h': len(recent_signals),
                'total_signals_history': len(self.signal_history),
                'active_signals': len(self.active_signals),
                'analysis_running': self.running
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao obter resumo de performance: {e}")
            return self.performance_stats

if __name__ == "__main__":
    # Teste do integrador de sinais
    from mt5_connector import MT5Connector
    from mt5_order_manager import MT5OrderManager
    from mt5_data_provider import MT5DataProvider
    from mt5_risk_manager import MT5RiskManager
    
    connector = MT5Connector()
    
    if connector.connect():
        print("✅ Conectado ao MT5")
        
        order_manager = MT5OrderManager(connector)
        data_provider = MT5DataProvider(connector)
        risk_manager = MT5RiskManager(connector, order_manager, data_provider)
        
        # Configurar integrador
        config = SignalConfig(enable_auto_trading=False)  # Desabilitado para teste
        integrator = MT5SignalIntegrator(connector, order_manager, data_provider, risk_manager, config)
        
        # Testar análise de mercado
        print("📊 Analisando mercado...")
        signals = integrator.analyze_market()
        
        print(f"🎯 {len(signals)} sinais gerados:")
        for signal in signals:
            print(f"   {signal.symbol}: {signal.action.value} ({signal.strength.value}) - Confiança: {signal.confidence:.2%}")
            print(f"      Razão: {signal.reason}")
        
        # Testar resumo de performance
        performance = integrator.get_performance_summary()
        print(f"📈 Performance: {performance}")
        
        connector.disconnect()
    else:
        print("❌ Falha na conexão MT5")