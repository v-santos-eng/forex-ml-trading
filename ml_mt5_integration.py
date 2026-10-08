#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integração ML com MetaTrader 5
Sistema de predições em tempo real e sinais de trading automatizados
Autor: Sistema Forex ML
Data: 2024
"""

import sys
import os
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import json
from collections import deque

# Adicionar diretório atual ao path para imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_polling_system import DataPollingSystem, DataType, DataUpdate
    from risk_manager import RiskManager, TradeRisk, RiskParameters
    # ML imports
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
except ImportError as e:
    logging.error(f"Erro ao importar módulos: {e}")
    DataPollingSystem = None
    RiskManager = None
    RandomForestClassifier = None
    GradientBoostingClassifier = None

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Tipos de sinais de trading"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_BUY = "close_buy"
    CLOSE_SELL = "close_sell"

class SignalStrength(Enum):
    """Força do sinal"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class TradingSignal:
    """Estrutura para sinais de trading"""
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    timestamp: datetime
    ml_prediction: Optional[Dict] = None
    technical_indicators: Optional[Dict] = None
    risk_analysis: Optional[TradeRisk] = None
    reasoning: str = ""

@dataclass
class MLPrediction:
    """Resultado de predição ML"""
    symbol: str
    direction: int  # 1 para alta, -1 para baixa, 0 para neutro
    confidence: float
    probability: float
    models_consensus: Dict[str, float]
    features_importance: Dict[str, float]
    timestamp: datetime
    next_candle_prediction: bool = True

class TechnicalAnalyzer:
    """
    Analisador técnico para complementar predições ML
    """
    
    def __init__(self):
        self.indicators_cache = {}
        self.cache_duration = 300  # 5 minutos
    
    def calculate_indicators(self, symbol: str, rates: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula indicadores técnicos
        
        Args:
            symbol: Símbolo
            rates: DataFrame com dados OHLCV
            
        Returns:
            Dict com indicadores calculados
        """
        try:
            if len(rates) < 50:
                return {}
            
            indicators = {}
            
            # Preços
            high = rates['high'].values
            low = rates['low'].values
            close = rates['close'].values
            volume = rates.get('tick_volume', rates.get('volume', np.ones(len(close)))).values
            
            # Médias móveis
            indicators['sma_20'] = np.mean(close[-20:])
            indicators['sma_50'] = np.mean(close[-50:]) if len(close) >= 50 else np.mean(close)
            indicators['ema_12'] = self._calculate_ema(close, 12)
            indicators['ema_26'] = self._calculate_ema(close, 26)
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(close, 14)
            
            # MACD
            macd_line = indicators['ema_12'] - indicators['ema_26']
            indicators['macd'] = macd_line
            indicators['macd_signal'] = self._calculate_ema([macd_line], 9)
            indicators['macd_histogram'] = macd_line - indicators['macd_signal']
            
            # Bollinger Bands
            sma_20 = indicators['sma_20']
            std_20 = np.std(close[-20:])
            indicators['bb_upper'] = sma_20 + (2 * std_20)
            indicators['bb_lower'] = sma_20 - (2 * std_20)
            indicators['bb_position'] = (close[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            
            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(high, low, close, 14)
            indicators['stoch_k'] = stoch_k
            indicators['stoch_d'] = stoch_d
            
            # ATR
            indicators['atr'] = self._calculate_atr(high, low, close, 14)
            
            # Volume indicators
            indicators['volume_sma'] = np.mean(volume[-20:])
            indicators['volume_ratio'] = volume[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1.0
            
            # Support/Resistance levels
            support, resistance = self._find_support_resistance(high, low, close)
            indicators['support'] = support
            indicators['resistance'] = resistance
            
            # Trend indicators
            indicators['trend_strength'] = self._calculate_trend_strength(close)
            indicators['price_momentum'] = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0
            
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores para {symbol}: {e}")
            return {}
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calcula EMA"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calcula RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, 
                            close: np.ndarray, period: int = 14) -> Tuple[float, float]:
        """Calcula Stochastic %K e %D"""
        if len(close) < period:
            return 50.0, 50.0
        
        lowest_low = np.min(low[-period:])
        highest_high = np.max(high[-period:])
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = ((close[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # %D é média móvel simples de %K (simplificado)
        d_percent = k_percent  # Em implementação completa, seria SMA de %K
        
        return k_percent, d_percent
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, period: int = 14) -> float:
        """Calcula Average True Range"""
        if len(close) < 2:
            return 0.0
        
        tr_list = []
        for i in range(1, len(close)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        
        if len(tr_list) < period:
            return np.mean(tr_list) if tr_list else 0.0
        
        return np.mean(tr_list[-period:])
    
    def _find_support_resistance(self, high: np.ndarray, low: np.ndarray, 
                               close: np.ndarray, lookback: int = 20) -> Tuple[float, float]:
        """Encontra níveis de suporte e resistência"""
        if len(close) < lookback:
            return close[-1] * 0.99, close[-1] * 1.01
        
        recent_high = high[-lookback:]
        recent_low = low[-lookback:]
        
        # Resistência: máxima recente
        resistance = np.max(recent_high)
        
        # Suporte: mínima recente
        support = np.min(recent_low)
        
        return support, resistance
    
    def _calculate_trend_strength(self, prices: np.ndarray, period: int = 20) -> float:
        """Calcula força da tendência (-1 a 1)"""
        if len(prices) < period:
            return 0.0
        
        recent_prices = prices[-period:]
        
        # Regressão linear simples
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]
        
        # Normalizar slope
        price_range = np.max(recent_prices) - np.min(recent_prices)
        if price_range > 0:
            normalized_slope = slope / (price_range / period)
        else:
            normalized_slope = 0.0
        
        # Limitar entre -1 e 1
        return max(-1.0, min(1.0, normalized_slope * 10))

class MLTradingEngine:
    """
    Engine principal de trading com ML
    Integra predições ML, análise técnica e gerenciamento de risco
    """
    
    def __init__(self, mt5_client, polling_system: DataPollingSystem = None, 
                 risk_manager: RiskManager = None):
        self.mt5_client = mt5_client
        self.polling_system = polling_system
        self.risk_manager = risk_manager
        
        # Inicializar componentes
        self.ml_predictor = self._create_simple_ml_predictor()
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Estado do sistema
        self.active_symbols = []
        self.active_signals = {}
        self.prediction_cache = {}
        self.running = False
        
        # Configurações
        self.min_confidence = 0.6  # Confiança mínima para sinais
        self.prediction_interval = 60  # Intervalo entre predições (segundos)
        self.max_positions_per_symbol = 1
        
        # Threading
        self.prediction_thread = None
        self.signal_thread = None
        
        # Estatísticas
        self.stats = {
            'predictions_made': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'successful_predictions': 0,
            'failed_predictions': 0
        }
        
        # Configurar handlers de eventos se polling system disponível
        if self.polling_system:
            self._setup_event_handlers()
    
    def _create_simple_ml_predictor(self):
        """Cria um preditor ML simples usando scikit-learn"""
        class SimpleMLPredictor:
            def __init__(self):
                self.models = {}
                self.scalers = {}
                self.is_trained = {}
                
            def predict(self, symbol: str, data: pd.DataFrame) -> Dict:
                """Faz predição para um símbolo"""
                try:
                    if symbol not in self.models or not self.is_trained.get(symbol, False):
                        # Treinar modelo se necessário
                        self._train_model(symbol, data)
                    
                    if len(data) < 50:  # Dados insuficientes
                        return {
                            'direction': 0,
                            'confidence': 0.5,
                            'probability': [0.5, 0.5]
                        }
                    
                    # Preparar features
                    features = self._prepare_features(data)
                    if features is None or len(features) == 0:
                        return {
                            'direction': 0,
                            'confidence': 0.5,
                            'probability': [0.5, 0.5]
                        }
                    
                    # Fazer predição
                    model = self.models[symbol]
                    scaler = self.scalers[symbol]
                    
                    features_scaled = scaler.transform([features])
                    prediction = model.predict(features_scaled)[0]
                    probabilities = model.predict_proba(features_scaled)[0]
                    
                    direction = 1 if prediction == 1 else -1
                    confidence = max(probabilities)
                    
                    return {
                        'direction': direction,
                        'confidence': confidence,
                        'probability': probabilities.tolist()
                    }
                    
                except Exception as e:
                    logger.error(f"Erro na predição ML: {e}")
                    return {
                        'direction': 0,
                        'confidence': 0.5,
                        'probability': [0.5, 0.5]
                    }
            
            def _prepare_features(self, data: pd.DataFrame) -> Optional[List[float]]:
                """Prepara features para o modelo"""
                try:
                    if len(data) < 20:
                        return None
                    
                    close = data['close'].values
                    high = data['high'].values
                    low = data['low'].values
                    
                    features = []
                    
                    # Retornos
                    returns = np.diff(close) / close[:-1]
                    if len(returns) >= 5:
                        features.extend([
                            np.mean(returns[-5:]),  # Retorno médio 5 períodos
                            np.std(returns[-5:]),   # Volatilidade 5 períodos
                            np.mean(returns[-10:]) if len(returns) >= 10 else 0,  # Retorno médio 10 períodos
                        ])
                    else:
                        features.extend([0, 0, 0])
                    
                    # Médias móveis
                    if len(close) >= 10:
                        sma_5 = np.mean(close[-5:])
                        sma_10 = np.mean(close[-10:])
                        features.append((close[-1] - sma_5) / sma_5)  # Distância da SMA5
                        features.append((sma_5 - sma_10) / sma_10)    # Relação SMA5/SMA10
                    else:
                        features.extend([0, 0])
                    
                    # RSI simplificado
                    if len(returns) >= 14:
                        gains = np.where(returns > 0, returns, 0)
                        losses = np.where(returns < 0, -returns, 0)
                        avg_gain = np.mean(gains[-14:])
                        avg_loss = np.mean(losses[-14:])
                        rs = avg_gain / (avg_loss + 1e-10)
                        rsi = 100 - (100 / (1 + rs))
                        features.append((rsi - 50) / 50)  # RSI normalizado
                    else:
                        features.append(0)
                    
                    # Volatilidade
                    if len(high) >= 10:
                        volatility = np.std((high[-10:] - low[-10:]) / close[-10:])
                        features.append(volatility)
                    else:
                        features.append(0)
                    
                    return features
                    
                except Exception as e:
                    logger.error(f"Erro ao preparar features: {e}")
                    return None
            
            def _train_model(self, symbol: str, data: pd.DataFrame):
                """Treina modelo para um símbolo"""
                try:
                    if len(data) < 100:  # Dados insuficientes para treino
                        return
                    
                    # Preparar dados de treino
                    X, y = self._prepare_training_data(data)
                    if X is None or len(X) < 50:
                        return
                    
                    # Dividir dados
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Escalar features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Treinar modelo
                    model = RandomForestClassifier(
                        n_estimators=50,
                        max_depth=10,
                        random_state=42
                    )
                    model.fit(X_train_scaled, y_train)
                    
                    # Avaliar
                    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
                    logger.info(f"Modelo {symbol} treinado com acurácia: {accuracy:.2%}")
                    
                    # Salvar modelo
                    self.models[symbol] = model
                    self.scalers[symbol] = scaler
                    self.is_trained[symbol] = True
                    
                except Exception as e:
                    logger.error(f"Erro ao treinar modelo: {e}")
            
            def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                """Prepara dados para treinamento"""
                try:
                    close = data['close'].values
                    high = data['high'].values
                    low = data['low'].values
                    
                    X = []
                    y = []
                    
                    # Criar features e labels
                    for i in range(20, len(close) - 1):  # Deixar espaço para features e label futuro
                        # Features baseadas nos últimos 20 períodos
                        window_data = pd.DataFrame({
                            'close': close[i-20:i],
                            'high': high[i-20:i],
                            'low': low[i-20:i]
                        })
                        
                        features = self._prepare_features(window_data)
                        if features is None:
                            continue
                        
                        # Label: direção do próximo movimento
                        future_return = (close[i+1] - close[i]) / close[i]
                        label = 1 if future_return > 0 else 0
                        
                        X.append(features)
                        y.append(label)
                    
                    return np.array(X) if X else None, np.array(y) if y else None
                    
                except Exception as e:
                    logger.error(f"Erro ao preparar dados de treino: {e}")
                    return None, None
        
        return SimpleMLPredictor()
    
    def start(self, symbols: List[str]):
        """
        Inicia o engine de trading
        
        Args:
            symbols: Lista de símbolos para monitorar
        """
        if self.running:
            logger.warning("Engine já está rodando")
            return
        
        # ML Predictor sempre disponível com implementação própria
        
        self.active_symbols = symbols.copy()
        self.running = True
        
        # Iniciar threads
        self._start_prediction_thread()
        self._start_signal_processing_thread()
        
        logger.info(f"ML Trading Engine iniciado para símbolos: {symbols}")
    
    def stop(self):
        """Para o engine de trading"""
        if not self.running:
            return
        
        self.running = False
        
        # Aguardar threads finalizarem
        if self.prediction_thread and self.prediction_thread.is_alive():
            self.prediction_thread.join(timeout=5.0)
        
        if self.signal_thread and self.signal_thread.is_alive():
            self.signal_thread.join(timeout=5.0)
        
        logger.info("ML Trading Engine parado")
    
    def get_prediction(self, symbol: str, force_refresh: bool = False) -> Optional[MLPrediction]:
        """
        Obtém predição ML para um símbolo
        
        Args:
            symbol: Símbolo
            force_refresh: Forçar nova predição
            
        Returns:
            MLPrediction ou None
        """
        try:
            # Verificar cache
            cache_key = f"prediction_{symbol}"
            if not force_refresh and cache_key in self.prediction_cache:
                cached_pred, timestamp = self.prediction_cache[cache_key]
                if (datetime.now() - timestamp).seconds < self.prediction_interval:
                    return cached_pred
            
            if not self.ml_predictor:
                return None
            
            # Fazer predição
            direction, confidence = self.ml_predictor.predict_direction(symbol)
            
            if direction is None or confidence == 0:
                return None
            
            # Obter detalhes dos modelos (se disponível)
            models_consensus = {}
            features_importance = {}
            
            try:
                # Tentar obter consenso dos modelos
                if hasattr(self.ml_predictor, 'get_models_consensus'):
                    models_consensus = self.ml_predictor.get_models_consensus(symbol)
                
                # Tentar obter importância das features
                if hasattr(self.ml_predictor, 'get_feature_importance'):
                    features_importance = self.ml_predictor.get_feature_importance(symbol)
            except:
                pass
            
            # Criar predição
            prediction = MLPrediction(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                probability=confidence,  # Simplificado
                models_consensus=models_consensus,
                features_importance=features_importance,
                timestamp=datetime.now()
            )
            
            # Atualizar cache
            self.prediction_cache[cache_key] = (prediction, datetime.now())
            self.stats['predictions_made'] += 1
            
            return prediction
            
        except Exception as e:
            logger.error(f"Erro ao obter predição para {symbol}: {e}")
            return None
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Gera sinal de trading baseado em ML e análise técnica
        
        Args:
            symbol: Símbolo
            
        Returns:
            TradingSignal ou None
        """
        try:
            # Obter predição ML
            ml_prediction = self.get_prediction(symbol)
            if not ml_prediction or ml_prediction.confidence < self.min_confidence:
                return None
            
            # Obter dados técnicos
            rates = self.mt5_client.get_rates(symbol, 16385, 100)  # M1, 100 barras
            if rates is None or len(rates) < 50:
                return None
            
            # Converter para DataFrame
            rates_df = pd.DataFrame(rates)
            
            # Calcular indicadores técnicos
            technical_indicators = self.technical_analyzer.calculate_indicators(symbol, rates_df)
            
            # Obter tick atual
            current_tick = self.mt5_client.get_symbol_tick(symbol)
            if not current_tick:
                return None
            
            current_price = current_tick.get('bid', rates_df['close'].iloc[-1])
            
            # Determinar tipo de sinal
            signal_type = self._determine_signal_type(ml_prediction, technical_indicators)
            
            if signal_type == SignalType.HOLD:
                return None
            
            # Calcular força do sinal
            strength = self._calculate_signal_strength(ml_prediction, technical_indicators)
            
            # Calcular stops ótimos
            direction = 1 if signal_type == SignalType.BUY else -1
            stop_loss, take_profit = self._calculate_optimal_stops(
                symbol, current_price, direction, technical_indicators
            )
            
            # Calcular position size
            position_size = self._calculate_position_size(
                symbol, current_price, stop_loss, ml_prediction.confidence
            )
            
            # Análise de risco
            risk_analysis = None
            if self.risk_manager:
                risk_analysis = self.risk_manager.analyze_trade_risk(
                    symbol, current_price, stop_loss, take_profit, position_size
                )
                
                if not risk_analysis.approved:
                    logger.info(f"Sinal rejeitado pelo risk manager: {risk_analysis.rejection_reason}")
                    return None
            
            # Criar sinal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=ml_prediction.confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                timestamp=datetime.now(),
                ml_prediction=ml_prediction.__dict__,
                technical_indicators=technical_indicators,
                risk_analysis=risk_analysis,
                reasoning=self._generate_reasoning(ml_prediction, technical_indicators, signal_type)
            )
            
            self.stats['signals_generated'] += 1
            
            return signal
            
        except Exception as e:
            logger.error(f"Erro ao gerar sinal para {symbol}: {e}")
            return None
    
    def execute_signal(self, signal: TradingSignal, auto_execute: bool = False) -> bool:
        """
        Executa sinal de trading
        
        Args:
            signal: Sinal para executar
            auto_execute: Executar automaticamente
            
        Returns:
            bool: True se executado com sucesso
        """
        try:
            if not auto_execute:
                logger.info(f"Sinal gerado para {signal.symbol}: {signal.signal_type.value} - Confiança: {signal.confidence:.2%}")
                return False
            
            # Verificar se já existe posição para este símbolo
            existing_positions = self.mt5_client.get_positions(symbol=signal.symbol)
            if len(existing_positions) >= self.max_positions_per_symbol:
                logger.info(f"Máximo de posições atingido para {signal.symbol}")
                return False
            
            # Preparar ordem
            order_type = 0 if signal.signal_type == SignalType.BUY else 1  # 0=BUY, 1=SELL
            
            order_request = {
                'action': 1,  # TRADE_ACTION_DEAL
                'symbol': signal.symbol,
                'volume': signal.position_size,
                'type': order_type,
                'price': signal.entry_price,
                'sl': signal.stop_loss,
                'tp': signal.take_profit,
                'comment': f"ML Signal - Conf: {signal.confidence:.2%}",
                'type_time': 0,  # ORDER_TIME_GTC
                'type_filling': 0  # ORDER_FILLING_FOK
            }
            
            # Executar ordem
            result = self.mt5_client.send_order(order_request)
            
            if result and result.get('retcode') == 10009:  # TRADE_RETCODE_DONE
                logger.info(f"Ordem executada com sucesso: {signal.symbol} {signal.signal_type.value}")
                self.stats['trades_executed'] += 1
                
                # Armazenar sinal ativo
                self.active_signals[signal.symbol] = signal
                
                return True
            else:
                error_msg = result.get('comment', 'Erro desconhecido') if result else 'Falha na comunicação'
                logger.error(f"Falha ao executar ordem: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao executar sinal: {e}")
            return False
    
    def _determine_signal_type(self, ml_prediction: MLPrediction, 
                             technical_indicators: Dict) -> SignalType:
        """Determina tipo de sinal baseado em ML e indicadores"""
        # Sinal base do ML
        if ml_prediction.direction == 1:
            base_signal = SignalType.BUY
        elif ml_prediction.direction == -1:
            base_signal = SignalType.SELL
        else:
            return SignalType.HOLD
        
        # Verificações técnicas
        confirmations = 0
        total_checks = 0
        
        # RSI
        if 'rsi' in technical_indicators:
            rsi = technical_indicators['rsi']
            total_checks += 1
            if base_signal == SignalType.BUY and rsi < 70:  # Não sobrecomprado
                confirmations += 1
            elif base_signal == SignalType.SELL and rsi > 30:  # Não sobrevendido
                confirmations += 1
        
        # MACD
        if 'macd_histogram' in technical_indicators:
            macd_hist = technical_indicators['macd_histogram']
            total_checks += 1
            if base_signal == SignalType.BUY and macd_hist > 0:
                confirmations += 1
            elif base_signal == SignalType.SELL and macd_hist < 0:
                confirmations += 1
        
        # Tendência
        if 'trend_strength' in technical_indicators:
            trend = technical_indicators['trend_strength']
            total_checks += 1
            if base_signal == SignalType.BUY and trend > 0.1:
                confirmations += 1
            elif base_signal == SignalType.SELL and trend < -0.1:
                confirmations += 1
        
        # Bollinger Bands
        if 'bb_position' in technical_indicators:
            bb_pos = technical_indicators['bb_position']
            total_checks += 1
            if base_signal == SignalType.BUY and bb_pos < 0.8:  # Não muito próximo da banda superior
                confirmations += 1
            elif base_signal == SignalType.SELL and bb_pos > 0.2:  # Não muito próximo da banda inferior
                confirmations += 1
        
        # Decisão final
        if total_checks > 0:
            confirmation_ratio = confirmations / total_checks
            if confirmation_ratio >= 0.6:  # 60% de confirmação
                return base_signal
        
        return SignalType.HOLD
    
    def _calculate_signal_strength(self, ml_prediction: MLPrediction, 
                                 technical_indicators: Dict) -> SignalStrength:
        """Calcula força do sinal"""
        strength_score = 0
        
        # Confiança ML (peso 40%)
        strength_score += ml_prediction.confidence * 0.4
        
        # Indicadores técnicos (peso 60%)
        technical_score = 0
        technical_count = 0
        
        # RSI extremo
        if 'rsi' in technical_indicators:
            rsi = technical_indicators['rsi']
            if rsi < 20 or rsi > 80:
                technical_score += 0.3
            elif rsi < 30 or rsi > 70:
                technical_score += 0.2
            else:
                technical_score += 0.1
            technical_count += 1
        
        # Força da tendência
        if 'trend_strength' in technical_indicators:
            trend_strength = abs(technical_indicators['trend_strength'])
            technical_score += trend_strength * 0.3
            technical_count += 1
        
        # Volume
        if 'volume_ratio' in technical_indicators:
            vol_ratio = technical_indicators['volume_ratio']
            if vol_ratio > 1.5:
                technical_score += 0.2
            elif vol_ratio > 1.2:
                technical_score += 0.1
            technical_count += 1
        
        if technical_count > 0:
            strength_score += (technical_score / technical_count) * 0.6
        
        # Determinar categoria
        if strength_score >= 0.8:
            return SignalStrength.VERY_STRONG
        elif strength_score >= 0.65:
            return SignalStrength.STRONG
        elif strength_score >= 0.5:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _calculate_optimal_stops(self, symbol: str, entry_price: float, 
                               direction: int, technical_indicators: Dict) -> Tuple[float, float]:
        """Calcula stop loss e take profit ótimos"""
        try:
            # Usar risk manager se disponível
            if self.risk_manager:
                return self.risk_manager.calculate_optimal_stops(symbol, entry_price, direction)
            
            # Fallback usando ATR
            atr = technical_indicators.get('atr', 0)
            if atr == 0:
                # Usar porcentagem fixa
                stop_distance = entry_price * 0.01  # 1%
                profit_distance = entry_price * 0.02  # 2%
            else:
                stop_distance = atr * 2
                profit_distance = atr * 3
            
            if direction == 1:  # Compra
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + profit_distance
            else:  # Venda
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - profit_distance
            
            # Arredondar
            symbol_info = self.mt5_client.get_symbol_info(symbol)
            digits = symbol_info.get('digits', 5) if symbol_info else 5
            
            stop_loss = round(stop_loss, digits)
            take_profit = round(take_profit, digits)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Erro ao calcular stops: {e}")
            # Valores padrão
            if direction == 1:
                return entry_price * 0.99, entry_price * 1.02
            else:
                return entry_price * 1.01, entry_price * 0.98
    
    def _calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss: float, confidence: float) -> float:
        """Calcula tamanho da posição"""
        try:
            # Usar risk manager se disponível
            if self.risk_manager:
                account_info = self.mt5_client.get_account_info()
                balance = account_info.get('balance', 10000) if account_info else 10000
                return self.risk_manager.calculate_position_size(symbol, entry_price, stop_loss, balance)
            
            # Fallback simples
            base_size = 0.01
            confidence_multiplier = confidence  # Usar confiança como multiplicador
            
            return round(base_size * confidence_multiplier, 2)
            
        except Exception as e:
            logger.error(f"Erro ao calcular position size: {e}")
            return 0.01
    
    def _generate_reasoning(self, ml_prediction: MLPrediction, 
                          technical_indicators: Dict, signal_type: SignalType) -> str:
        """Gera explicação do sinal"""
        reasoning_parts = []
        
        # ML
        reasoning_parts.append(f"ML prediz {ml_prediction.direction} com {ml_prediction.confidence:.1%} confiança")
        
        # Técnicos
        if 'rsi' in technical_indicators:
            rsi = technical_indicators['rsi']
            if rsi < 30:
                reasoning_parts.append("RSI indica sobrevendido")
            elif rsi > 70:
                reasoning_parts.append("RSI indica sobrecomprado")
        
        if 'trend_strength' in technical_indicators:
            trend = technical_indicators['trend_strength']
            if abs(trend) > 0.3:
                direction = "alta" if trend > 0 else "baixa"
                reasoning_parts.append(f"Tendência forte de {direction}")
        
        if 'macd_histogram' in technical_indicators:
            macd = technical_indicators['macd_histogram']
            if macd > 0:
                reasoning_parts.append("MACD positivo")
            else:
                reasoning_parts.append("MACD negativo")
        
        return "; ".join(reasoning_parts)
    
    def _setup_event_handlers(self):
        """Configura handlers para eventos do polling system"""
        if not self.polling_system:
            return
        
        # Handler para ticks
        def on_tick_update(update: DataUpdate):
            if update.symbol in self.active_symbols:
                # Verificar se precisa gerar novo sinal
                self._check_signal_conditions(update.symbol)
        
        self.polling_system.add_event_handler(DataType.TICKS, on_tick_update)
    
    def _check_signal_conditions(self, symbol: str):
        """Verifica condições para gerar novo sinal"""
        try:
            # Evitar spam de sinais
            last_signal_key = f"last_signal_{symbol}"
            if hasattr(self, last_signal_key):
                last_time = getattr(self, last_signal_key)
                if (datetime.now() - last_time).seconds < 300:  # 5 minutos
                    return
            
            # Gerar sinal
            signal = self.generate_signal(symbol)
            if signal:
                setattr(self, last_signal_key, datetime.now())
                logger.info(f"Novo sinal: {signal.symbol} {signal.signal_type.value} - {signal.strength.value}")
                
                # Aqui você pode adicionar lógica para notificar o dashboard
                # ou executar automaticamente se configurado
                
        except Exception as e:
            logger.error(f"Erro ao verificar condições de sinal: {e}")
    
    def _start_prediction_thread(self):
        """Inicia thread para predições periódicas"""
        def prediction_loop():
            while self.running:
                try:
                    for symbol in self.active_symbols:
                        if not self.running:
                            break
                        
                        # Fazer predição
                        prediction = self.get_prediction(symbol, force_refresh=True)
                        if prediction:
                            logger.debug(f"Predição {symbol}: {prediction.direction} ({prediction.confidence:.2%})")
                    
                    # Aguardar próximo ciclo
                    time.sleep(self.prediction_interval)
                    
                except Exception as e:
                    logger.error(f"Erro no loop de predições: {e}")
                    time.sleep(10)
        
        self.prediction_thread = threading.Thread(target=prediction_loop, daemon=True)
        self.prediction_thread.start()
    
    def _start_signal_processing_thread(self):
        """Inicia thread para processamento de sinais"""
        def signal_loop():
            while self.running:
                try:
                    for symbol in self.active_symbols:
                        if not self.running:
                            break
                        
                        # Verificar condições de sinal
                        self._check_signal_conditions(symbol)
                    
                    # Aguardar próximo ciclo
                    time.sleep(30)  # Verificar a cada 30 segundos
                    
                except Exception as e:
                    logger.error(f"Erro no loop de sinais: {e}")
                    time.sleep(10)
        
        self.signal_thread = threading.Thread(target=signal_loop, daemon=True)
        self.signal_thread.start()
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do engine"""
        return {
            'running': self.running,
            'active_symbols': self.active_symbols.copy(),
            'active_signals_count': len(self.active_signals),
            'stats': self.stats.copy(),
            'cache_size': len(self.prediction_cache)
        }
    
    def get_active_signals(self) -> Dict[str, TradingSignal]:
        """Retorna sinais ativos"""
        return self.active_signals.copy()
    
    def clear_signal(self, symbol: str):
        """Remove sinal ativo"""
        if symbol in self.active_signals:
            del self.active_signals[symbol]

# Exemplo de uso
if __name__ == "__main__":
    print("ML MT5 Integration - Sistema de Trading com Machine Learning")
    print("Funcionalidades implementadas:")
    print("- Predições ML em tempo real")
    print("- Análise técnica complementar")
    print("- Geração de sinais inteligentes")
    print("- Integração com gerenciamento de risco")
    print("- Execução automática de ordens")
    print("- Sistema de eventos em tempo real")
    print("- Cálculo otimizado de stops")
    print("- Position sizing inteligente")