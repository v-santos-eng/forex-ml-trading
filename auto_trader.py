#!/usr/bin/env python3
"""
Sistema de Trading Automatizado
Executa trades baseado em sinais ML do Ollama
"""

import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import MetaTrader5 as mt5
from dataclasses import dataclass
import threading
import requests
import numpy as np
from ml_predictor_local import MLForexPredictorLocal as MLPredictor
from detailed_decision_logger import DetailedDecisionLogger, DecisionData, DecisionReasoning, BotAction
from advanced_risk_system import AdvancedRiskSystem, ATRParameters, VolatilityParameters, TrailingStopParameters

@dataclass
class TradeSignal:
    """Sinal de trading"""
    symbol: str
    direction: str  # BUY/SELL
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float
    timestamp: datetime
    model_source: str

@dataclass
class TradeExecution:
    """Execução de trade"""
    ticket: int
    symbol: str
    type: int
    volume: float
    price: float
    sl: float
    tp: float
    comment: str
    timestamp: datetime

class AutoTrader:
    """Sistema de trading automatizado"""
    
    def __init__(self, config_file: str = "auto_trader_config.json"):
        self.config = self._load_config(config_file)
        self.running = False
        self.ml_predictor = MLPredictor(config=self.config)
        
        # Configurações de risco
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)  # 2%
        self.max_daily_trades = self.config.get('max_daily_trades', 10)
        self.min_confidence = self.config.get('min_confidence', 0.65)  # 65%
        self.symbols_to_trade = self.config.get('symbols', ['EURUSD', 'GBPUSD', 'USDJPY'])
        
        # Configurações obrigatórias de SL/TP (fallback em pips)
        self.default_sl_pips = self.config.get('default_sl_pips', 50)  # Stop Loss padrão: 50 pips
        self.default_tp_pips = self.config.get('default_tp_pips', 100)  # Take Profit padrão: 100 pips
        self.min_sl_pips = self.config.get('min_sl_pips', 20)  # SL mínimo: 20 pips
        self.min_tp_pips = self.config.get('min_tp_pips', 30)  # TP mínimo: 30 pips
        self.enforce_sl_tp = self.config.get('enforce_sl_tp', True)  # Forçar SL/TP obrigatórios
        
        # Inicializar Sistema Avançado de Risco
        advanced_risk_config = self.config.get('advanced_risk', {})
        self.use_advanced_risk = advanced_risk_config.get('enabled', False)
        
        if self.use_advanced_risk:
            # Configurar parâmetros ATR
            atr_config = advanced_risk_config.get('atr_params', {})
            atr_params = ATRParameters(
                period=atr_config.get('period', 14),
                multiplier_sl=atr_config.get('multiplier_sl', 2.0),
                multiplier_tp=atr_config.get('multiplier_tp', 3.0),
                volatility_adjustment=atr_config.get('volatility_adjustment', True),
                session_adjustment=atr_config.get('session_adjustment', True)
            )
            
            # Configurar parâmetros de volatilidade
            vol_config = advanced_risk_config.get('volatility_params', {})
            vol_params = VolatilityParameters(
                lookback_period=vol_config.get('lookback_period', 20)
            )
            
            # Configurar trailing stop
            trailing_config = advanced_risk_config.get('trailing_stop_params', {})
            trailing_params = TrailingStopParameters(
                activation_distance=trailing_config.get('activation_distance', 1.5),
                trail_distance=trailing_config.get('trail_distance', 1.0)
            )
            
            # Inicializar sistema avançado
            self.advanced_risk = AdvancedRiskSystem(
                mt5_client=self,  # Passar self como cliente MT5
                atr_params=atr_params,
                volatility_params=vol_params,
                trailing_params=trailing_params
            )
            
            print(f"Sistema Avançado de Risco Ativado:")
            print(f"   ATR Período: {atr_params.period} | SL Mult: {atr_params.multiplier_sl} | TP Mult: {atr_params.multiplier_tp}")
            print(f"   Ajuste Volatilidade: {atr_params.volatility_adjustment} | Ajuste Sessão: {atr_params.session_adjustment}")
        else:
            self.advanced_risk = None
            print(f"Configurações de Proteção (Modo Fixo):")
            print(f"   SL Padrão: {self.default_sl_pips} pips | TP Padrão: {self.default_tp_pips} pips")
            print(f"   SL Mínimo: {self.min_sl_pips} pips | TP Mínimo: {self.min_tp_pips} pips")
        
        # Mostrar status do enforce_sl_tp
        if self.enforce_sl_tp:
            print(f"   ✅ TODAS as ordens DEVEM ter SL e TP obrigatórios (enforce_sl_tp=True)")
        else:
            print(f"   ⚠️ AVISO: enforce_sl_tp está DESATIVADO - ordens podem ser executadas sem SL/TP")
        
        # Controle de trades
        self.daily_trades_count = 0
        self.last_trade_date = None
        self.active_positions = {}
        
        # Database para logging
        self.db_path = 'auto_trader.db'
        self._init_database()
        
        # Sistema de logging detalhado
        self.detailed_logger = DetailedDecisionLogger()
        print(f"Sistema de logging detalhado inicializado")
        
        # Thread principal
        self.trading_thread = None
        
    def _validate_sl_tp(self, signal: TradeSignal) -> bool:
        """Validar se SL e TP estão configurados corretamente"""
        if signal.stop_loss == 0.0 or signal.take_profit == 0.0:
            print(f"❌ ERRO CRÍTICO: SL ou TP zerados para {signal.symbol}")
            print(f"   SL: {signal.stop_loss}, TP: {signal.take_profit}")
            return False
        
        # Validar direções de SL e TP
        if signal.direction == "BUY":
            if signal.stop_loss >= signal.entry_price:
                print(f"❌ ERRO: SL inválido para BUY - SL deve ser menor que entry")
                return False
            if signal.take_profit <= signal.entry_price:
                print(f"❌ ERRO: TP inválido para BUY - TP deve ser maior que entry")
                return False
        else:  # SELL
            if signal.stop_loss <= signal.entry_price:
                print(f"❌ ERRO: SL inválido para SELL - SL deve ser maior que entry")
                return False
            if signal.take_profit >= signal.entry_price:
                print(f"❌ ERRO: TP inválido para SELL - TP deve ser menor que entry")
                return False
        
        return True

    def _load_config(self, config_file: str) -> Dict:
        """Carrega configuração do arquivo"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Configuração padrão
            default_config = {
                "max_risk_per_trade": 0.02,
                "max_daily_trades": 10,
                "min_confidence": 0.65,
                "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
                "ollama_url": "http://localhost:11434",
                "prediction_interval": 300,  # 5 minutos
                "risk_reward_ratio": 2.0,
                "max_spread_pips": 3.0,
                "trading_hours": {
                    "start": "08:00",
                    "end": "22:00"
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
            
    def _init_database(self):
        """Inicializa banco de dados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de sinais
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                confidence REAL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                volume REAL,
                model_source TEXT,
                executed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Tabela de execuções
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                ticket INTEGER,
                symbol TEXT,
                type INTEGER,
                volume REAL,
                open_price REAL,
                sl REAL,
                tp REAL,
                comment TEXT,
                timestamp TEXT,
                status TEXT,
                close_price REAL,
                close_time TEXT,
                profit REAL,
                FOREIGN KEY (signal_id) REFERENCES signals (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def start_trading(self):
        """Inicia sistema de trading automatizado"""
        if self.running:
            print("Sistema já está em execução")
            return
            
        # Verificar conexão MT5
        if not mt5.initialize():
            print("❌ Erro ao conectar com MT5")
            return
            
        account_info = mt5.account_info()
        if not account_info:
            print("❌ Erro ao obter informações da conta")
            return
            
        print(f"Iniciando Auto Trader")
        print(f"Conta: {account_info.login} | Saldo: ${account_info.balance:,.2f}")
        print(f"Símbolos: {', '.join(self.symbols_to_trade)}")
        print(f"Confiança mínima: {self.min_confidence:.1%}")
        print(f"Risco por trade: {self.max_risk_per_trade:.1%}")
        
        self.running = True
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        
    def stop_trading(self):
        """Para o sistema de trading"""
        self.running = False
        if self.trading_thread:
            self.trading_thread.join(timeout=10)
        print("⏹️ Auto Trader parado")
        
    def _trading_loop(self):
        """Loop principal de trading"""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Reset contador diário
                if self.last_trade_date != current_time.date():
                    self.daily_trades_count = 0
                    self.last_trade_date = current_time.date()
                    
                # Verificar horário de trading
                if not self._is_trading_hours():
                    print(f"⏰ Fora do horário de trading: {current_time.strftime('%H:%M')}")
                    time.sleep(300)  # 5 minutos
                    continue
                    
                # Verificar limite diário
                if self.daily_trades_count >= self.max_daily_trades:
                    print(f"📊 Limite diário de trades atingido: {self.daily_trades_count}")
                    time.sleep(3600)  # 1 hora
                    continue
                    
                # Gerar e processar sinais
                self._process_trading_signals()
                
                # Gerenciar posições abertas
                self._manage_open_positions()
                
                # Aguardar próximo ciclo
                time.sleep(self.config.get('prediction_interval', 300))
                
            except Exception as e:
                print(f"❌ Erro no loop de trading: {e}")
                time.sleep(60)
                
    def _is_trading_hours(self) -> bool:
        """Verifica se está no horário de trading"""
        now = datetime.now().time()
        start_time = datetime.strptime(self.config['trading_hours']['start'], '%H:%M').time()
        end_time = datetime.strptime(self.config['trading_hours']['end'], '%H:%M').time()
        
        return start_time <= now <= end_time
        
    def _process_trading_signals(self):
        """Processa sinais de trading"""
        for symbol in self.symbols_to_trade:
            try:
                # Gerar predição ML
                signal = self._generate_ml_signal(symbol)
                
                if signal and signal.confidence >= self.min_confidence:
                    # Verificar se já existe posição para este símbolo
                    if symbol in self.active_positions:
                        continue
                        
                    # Verificar condições de mercado
                    if self._validate_market_conditions(symbol, signal):
                        # Executar trade
                        execution = self._execute_trade(signal)
                        
                        if execution:
                            self._log_signal_and_execution(signal, execution)
                            self.daily_trades_count += 1
                            print(f"✅ Trade executado: {signal.symbol} {signal.direction} | Ticket: {execution.ticket}")
                        
            except Exception as e:
                print(f"❌ Erro ao processar {symbol}: {e}")
                
    def _generate_ml_signal(self, symbol: str) -> Optional[TradeSignal]:
        """Gera sinal usando ML"""
        try:
            # Obter dados de mercado atuais
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return None
                
            # Preparar dados para logging detalhado
            market_data = {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'volume': tick.volume,
                'time': tick.time
            }
            
            # Obter predição do modelo local
            X, y = self.ml_predictor.prepare_data(symbol, period='7d')
            
            if X is None or len(X) < 100:
                # Log decisão de não trading por dados insuficientes
                decision_data = DecisionData(
                    symbol=symbol,
                    action="NO_ACTION",
                    timestamp=datetime.now(),
                    market_data=market_data,
                    technical_analysis={},
                    ml_predictions={},
                    reasoning=DecisionReasoning(
                        factors=[f"Apenas {len(X) if X is not None else 0} amostras disponíveis, mínimo 100 necessário"],
                        confidence_score=0.0,
                        risk_assessment="Baixo risco - sem ação"
                    ),
                    bot_action=BotAction(
                        action_type="NO_ACTION",
                        parameters={"reason": "Dados insuficientes"},
                        result="Aguardar mais dados históricos"
                    )
                )
                self.detailed_logger.log_decision(decision_data)
                return None
                
            # Treinar modelos
            models = self.ml_predictor.train_models(X, y)
            
            if not models:
                # Log decisão de não trading por falha no treinamento
                decision_data = DecisionData(
                    symbol=symbol,
                    action="NO_ACTION",
                    timestamp=datetime.now(),
                    market_data=market_data,
                    technical_analysis={},
                    ml_predictions={},
                    reasoning=DecisionReasoning(
                        factors=["Nenhum modelo foi treinado com sucesso"],
                        confidence_score=0.0,
                        risk_assessment="Alto risco - sem ação"
                    ),
                    bot_action=BotAction(
                        action_type="NO_ACTION",
                        parameters={"reason": "Falha no treinamento ML"},
                        result="Aguardar condições melhores para treinamento"
                    )
                )
                self.detailed_logger.log_decision(decision_data)
                return None
                
            # Fazer predição
            latest_features = X[-1:]
            predictions = []
            confidences = []
            model_details = {}
            
            for name, model_result in models.items():
                try:
                    # Extrair o modelo do resultado do treinamento
                    if isinstance(model_result, dict) and 'model' in model_result:
                        model = model_result['model']
                        
                        # Escalar as features usando o scaler do predictor
                        latest_scaled = self.ml_predictor.scaler.transform(latest_features)
                        
                        pred = model.predict(latest_scaled)[0]
                        prob = model.predict_proba(latest_scaled)[0]
                        confidence = max(prob)
                        
                        predictions.append(pred)
                        confidences.append(confidence)
                        model_details[name] = {
                            'prediction': float(pred),
                            'confidence': float(confidence),
                            'probabilities': [float(p) for p in prob]
                        }
                    else:
                        model_details[name] = {'error': 'Estrutura de modelo inválida'}
                except Exception as e:
                    model_details[name] = {'error': str(e)}
                    continue
                    
            if not predictions:
                # Log decisão de não trading por falha nas predições
                decision_data = DecisionData(
                    symbol=symbol,
                    action="NO_ACTION",
                    timestamp=datetime.now(),
                    market_data=market_data,
                    technical_analysis={},
                    ml_predictions=model_details,
                    reasoning=DecisionReasoning(
                        factors=["Nenhum modelo conseguiu fazer predição válida"],
                        confidence_score=0.0,
                        risk_assessment="Alto risco - sem ação"
                    ),
                    bot_action=BotAction(
                        action_type="NO_ACTION",
                        parameters={"reason": "Falha nas predições"},
                        result="Aguardar estabilização dos modelos"
                    )
                )
                self.detailed_logger.log_decision(decision_data)
                return None
                
            # Consenso das predições
            avg_pred = np.mean(predictions)
            avg_confidence = np.mean(confidences)
            
            # Obter preço atual
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return None
                
            current_price = (tick.bid + tick.ask) / 2
            
            # Determinar direção
            direction = "BUY" if avg_pred > 0.5 else "SELL"
            
            # Calcular stop loss e take profit
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return None
            
            # Determinar preço de entrada
            if direction == "BUY":
                entry_price = tick.ask
                direction_int = 1
            else:
                entry_price = tick.bid
                direction_int = -1
            
            # Calcular SL/TP usando sistema avançado ou fallback
            if self.use_advanced_risk and self.advanced_risk:
                try:
                    # Usar sistema avançado baseado em ATR
                    stop_loss, take_profit, metadata = self.advanced_risk.calculate_optimal_sl_tp(
                        symbol=symbol,
                        entry_price=entry_price,
                        direction=direction_int,
                        confidence=avg_confidence
                    )
                    
                    # Log dos cálculos avançados
                    print(f"📊 Cálculo SL/TP Avançado para {symbol} {direction}:")
                    print(f"   Entry: {entry_price:.5f} | ATR: {metadata['atr_value']:.5f}")
                    print(f"   SL: {stop_loss:.5f} (ATR x {metadata['sl_multiplier']:.2f}) | TP: {take_profit:.5f} (ATR x {metadata['tp_multiplier']:.2f})")
                    print(f"   Risk/Reward: {metadata['risk_reward_ratio']:.2f} | Método: {metadata['atr_method']}")
                    
                    # Registrar decisão avançada de risco no log detalhado
                    if hasattr(self, 'detailed_logger'):
                        atr_data = {
                            'method': metadata['atr_method'],
                            'value': metadata['atr_value'],
                            'volatility_regime': metadata.get('volatility_regime', 'normal'),
                            'gap_protection': metadata.get('gap_protection_active', False),
                            'technical_levels': metadata.get('technical_levels_considered', False),
                            'trailing_stop': metadata.get('trailing_stop_enabled', False),
                            'metadata': metadata
                        }
                        
                        sl_tp_data = {
                            'sl_multiplier': metadata['sl_multiplier'],
                            'tp_multiplier': metadata['tp_multiplier'],
                            'risk_reward_ratio': metadata['risk_reward_ratio'],
                            'sl_price': stop_loss,
                            'tp_price': take_profit,
                            'entry_price': entry_price
                        }
                        
                        market_data = {
                            'symbol': symbol,
                            'bid': tick.bid,
                            'ask': tick.ask,
                            'spread': tick.ask - tick.bid,
                            'direction': direction
                        }
                        
                        self.detailed_logger.log_advanced_risk_decision(
                            symbol=symbol,
                            action="ADVANCED_SL_TP_CALCULATION",
                            confidence=avg_confidence,
                            atr_data=atr_data,
                            sl_tp_data=sl_tp_data,
                            market_data=market_data
                        )
                    
                except Exception as e:
                    print(f"⚠️ Erro no sistema avançado, usando fallback: {e}")
                    # Fallback para método fixo
                    pip_value = symbol_info.point * 10 if symbol_info.digits == 5 else symbol_info.point
                    sl_pips = max(self.default_sl_pips, self.min_sl_pips)
                    tp_pips = max(self.default_tp_pips, self.min_tp_pips)
                    
                    if direction == "BUY":
                        stop_loss = entry_price - (sl_pips * pip_value)
                        take_profit = entry_price + (tp_pips * pip_value)
                    else:
                        stop_loss = entry_price + (sl_pips * pip_value)
                        take_profit = entry_price - (tp_pips * pip_value)
                    
                    print(f"📊 Cálculo SL/TP Fallback para {symbol} {direction}:")
                    print(f"   Entry: {entry_price:.5f} | SL: {sl_pips} pips ({stop_loss:.5f}) | TP: {tp_pips} pips ({take_profit:.5f})")
            else:
                # Método fixo tradicional
                pip_value = symbol_info.point * 10 if symbol_info.digits == 5 else symbol_info.point
                sl_pips = max(self.default_sl_pips, self.min_sl_pips)
                tp_pips = max(self.default_tp_pips, self.min_tp_pips)
                
                if direction == "BUY":
                    stop_loss = entry_price - (sl_pips * pip_value)
                    take_profit = entry_price + (tp_pips * pip_value)
                else:
                    stop_loss = entry_price + (sl_pips * pip_value)
                    take_profit = entry_price - (tp_pips * pip_value)
                
                print(f"📊 Cálculo SL/TP Fixo para {symbol} {direction}:")
                print(f"   Entry: {entry_price:.5f} | SL: {sl_pips} pips ({stop_loss:.5f}) | TP: {tp_pips} pips ({take_profit:.5f})")
                
            # VALIDAÇÃO CRÍTICA: Verificar se SL e TP foram calculados corretamente
            if not stop_loss or not take_profit or stop_loss <= 0 or take_profit <= 0:
                print(f"❌ ERRO CRÍTICO: SL ou TP inválidos calculados para {symbol}")
                print(f"   Entry: {entry_price}, SL: {stop_loss}, TP: {take_profit}")
                return None
            
            # Validar direções de SL e TP
            if direction == "BUY":
                if stop_loss >= entry_price or take_profit <= entry_price:
                    print(f"❌ ERRO CRÍTICO: SL/TP mal calculados para BUY em {symbol}")
                    print(f"   Entry: {entry_price}, SL: {stop_loss}, TP: {take_profit}")
                    return None
            else:  # SELL
                if stop_loss <= entry_price or take_profit >= entry_price:
                    print(f"❌ ERRO CRÍTICO: SL/TP mal calculados para SELL em {symbol}")
                    print(f"   Entry: {entry_price}, SL: {stop_loss}, TP: {take_profit}")
                    return None
            
            # NOVO: Validar distância mínima de stops do broker (stops_level)
            symbol_info = mt5.symbol_info(signal.symbol)
            if not symbol_info:
                print(f"❌ Erro ao obter informações do símbolo {signal.symbol} para validar stops_level")
                return None
            point = getattr(symbol_info, 'point', None) or 0.0
            stops_level = getattr(symbol_info, 'trade_stops_level', None)
            if stops_level is None:
                stops_level = getattr(symbol_info, 'stops_level', 0)
            try:
                stops_level = float(stops_level)
            except Exception:
                stops_level = 0.0

            if point > 0 and stops_level > 0:
                sl_dist_pts = abs(signal.entry_price - signal.stop_loss) / point
                tp_dist_pts = abs(signal.take_profit - signal.entry_price) / point
                if sl_dist_pts < stops_level or tp_dist_pts < stops_level:
                    print(f"❌ INVALID_STOPS: Distância de SL/TP abaixo do mínimo exigido pelo broker para {signal.symbol}")
                    print(f"   stops_level: {stops_level:.0f} pts | SL: {sl_dist_pts:.1f} pts | TP: {tp_dist_pts:.1f} pts")

                    # Log de NO_ACTION para evitar retcode 10016
                    tick = mt5.symbol_info_tick(signal.symbol)
                    market_data = {
                        'symbol': signal.symbol,
                        'bid': tick.bid if tick else 0,
                        'ask': tick.ask if tick else 0,
                        'spread': (tick.ask - tick.bid) if tick else 0,
                        'volume': tick.volume if tick else 0,
                        'time': tick.time if tick else 0
                    }
                    order_type = mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL
                    decision_data = DecisionData(
                        symbol=signal.symbol,
                        action="NO_ACTION",
                        timestamp=datetime.now(),
                        market_data=market_data,
                        technical_analysis={
                            'direction': signal.direction,
                            'entry_price': signal.entry_price,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                            'volume': signal.volume,
                            'order_type': order_type,
                            'stops_level_min_points': stops_level,
                            'sl_distance_points': sl_dist_pts,
                            'tp_distance_points': tp_dist_pts
                        },
                        ml_predictions={
                            'confidence': signal.confidence,
                            'model_source': signal.model_source
                        },
                        reasoning=DecisionReasoning(
                            factors=[
                                "SL/TP abaixo da distância mínima exigida pelo broker (stops_level)",
                                f"stops_level: {stops_level:.0f} pts",
                                f"SL distance: {sl_dist_pts:.1f} pts",
                                f"TP distance: {tp_dist_pts:.1f} pts"
                            ],
                            confidence_score=float(signal.confidence),
                            risk_assessment="Risco de execução - ordem evitada para prevenir INVALID_STOPS (10016)"
                        ),
                        bot_action=BotAction(
                            action_type="NO_ACTION",
                            parameters={
                                'reason': 'INVALID_STOPS_PREVENTED',
                                'stops_level_points': stops_level,
                                'sl_distance_points': sl_dist_pts,
                                'tp_distance_points': tp_dist_pts
                            },
                            result="Ajuste SL/TP ou reavalie parâmetros de risco (ATR/pips mínimos)"
                        )
                    )
                    self.detailed_logger.log_decision(decision_data)
                    return None

            # Preparar request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": signal.volume,
                "type": mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": signal.entry_price,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "deviation": 20,
                "magic": 12345,
                "comment": f"AutoTrader_{signal.model_source}_{signal.confidence:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Log da ordem antes de executar
            print(f"🔍 Executando trade: {signal.symbol} {signal.direction}")
            print(f"   Entry: {signal.entry_price:.5f} | SL: {signal.stop_loss:.5f} | TP: {signal.take_profit:.5f}")
            print(f"   Volume: {signal.volume} | Confiança: {signal.confidence:.2f}%")
            
            # Executar ordem
            result = mt5.order_send(request)
            
            # Preparar dados de mercado para logging
            tick = mt5.symbol_info_tick(signal.symbol)
            market_data = {
                'symbol': signal.symbol,
                'bid': tick.bid if tick else 0,
                'ask': tick.ask if tick else 0,
                'spread': (tick.ask - tick.bid) if tick else 0,
                'volume': tick.volume if tick else 0,
                'time': tick.time if tick else 0
            }
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"❌ Erro ao executar trade: {result.comment}")
                
                # Log execução falhada
                decision_data = DecisionData(
                    symbol=signal.symbol,
                    action="EXECUTE_TRADE",
                    timestamp=datetime.now(),
                    market_data=market_data,
                    technical_analysis={
                        'direction': signal.direction,
                        'entry_price': signal.entry_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'volume': signal.volume,
                        'order_type': request["type"]
                    },
                    ml_predictions={
                        'confidence': signal.confidence,
                        'model_source': signal.model_source
                    },
                    reasoning=DecisionReasoning(
                        factors=[
                            f"Falha na execução da ordem: {result.comment}",
                            f"Código de retorno: {result.retcode}",
                            f"Comentário MT5: {result.comment}",
                            f"Sinal válido gerado com confiança {signal.confidence:.3f}"
                        ],
                        confidence_score=float(signal.confidence),
                        risk_assessment="Risco de execução - ordem rejeitada pelo broker"
                    ),
                    bot_action=BotAction(
                        action_type="EXECUTION_FAILED",
                        parameters={"retcode": result.retcode, "error": f"MT5 rejeitou ordem: {result.comment}"},
                        result="Aguardar próxima oportunidade"
                    )
                )
                self.detailed_logger.log_decision(decision_data)
                return None
                
            # Registrar posição ativa
            self.active_positions[signal.symbol] = {
                'ticket': result.order,
                'direction': signal.direction,
                'volume': signal.volume,
                'open_time': datetime.now()
            }
            
            # Log execução bem-sucedida
            decision_data = DecisionData(
                symbol=signal.symbol,
                action="EXECUTE_TRADE",
                timestamp=datetime.now(),
                market_data=market_data,
                technical_analysis={
                    'direction': signal.direction,
                    'entry_price': signal.entry_price,
                    'executed_price': result.price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'volume': signal.volume,
                    'slippage': abs(result.price - signal.entry_price),
                    'order_type': request["type"],
                    'ticket': result.order
                },
                ml_predictions={
                    'confidence': signal.confidence,
                    'model_source': signal.model_source
                },
                reasoning=DecisionReasoning(
                    factors=[
                        f"Ordem executada com sucesso - Ticket #{result.order}",
                        f"Preço executado: {result.price:.5f}",
                        f"Slippage: {abs(result.price - signal.entry_price):.5f}",
                        f"SL/TP configurados corretamente",
                        f"Volume: {signal.volume}"
                    ],
                    confidence_score=float(signal.confidence),
                    risk_assessment="Risco controlado - posição aberta com SL/TP"
                ),
                bot_action=BotAction(
                    action_type="TRADE_EXECUTED",
                    parameters={"ticket": result.order, "confidence": signal.confidence, "executed_price": result.price},
                    result=f"Lucro potencial: {abs(signal.take_profit - result.price):.5f}, Risco máximo: {abs(result.price - signal.stop_loss):.5f}"
                )
            )
            self.detailed_logger.log_decision(decision_data)
            
            return TradeExecution(
                ticket=result.order,
                symbol=signal.symbol,
                type=request["type"],
                volume=signal.volume,
                price=result.price,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                comment=request["comment"],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"❌ Erro ao executar trade: {e}")
            return None
            
    def _execute_trade(self, signal: TradeSignal) -> Optional[TradeExecution]:
        """Executa uma ordem no MT5 com validações críticas (SL/TP e stops_level)."""
        try:
            # 1) Validar SL/TP básicos
            if not self._validate_sl_tp(signal):
                tick = mt5.symbol_info_tick(signal.symbol)
                market_data = {
                    'symbol': signal.symbol,
                    'bid': tick.bid if tick else 0,
                    'ask': tick.ask if tick else 0,
                    'spread': (tick.ask - tick.bid) if tick else 0,
                    'volume': tick.volume if tick else 0,
                    'time': tick.time if tick else 0
                }
                decision_data = DecisionData(
                    symbol=signal.symbol,
                    action="NO_ACTION",
                    timestamp=datetime.now(),
                    market_data=market_data,
                    technical_analysis={
                        'direction': signal.direction,
                        'entry_price': signal.entry_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'volume': signal.volume
                    },
                    ml_predictions={
                        'confidence': signal.confidence,
                        'model_source': signal.model_source
                    },
                    reasoning=DecisionReasoning(
                        factors=[
                            "SL/TP inválidos conforme validação básica (_validate_sl_tp)"
                        ],
                        confidence_score=float(signal.confidence),
                        risk_assessment="Risco de execução - ordem evitada por SL/TP inválidos"
                    ),
                    bot_action=BotAction(
                        action_type="NO_ACTION",
                        parameters={'reason': 'INVALID_SLTP'},
                        result="Corrigir SL/TP antes de executar"
                    )
                )
                self.detailed_logger.log_decision(decision_data)
                return None

            # 2) Validar distância mínima de stops exigida pelo broker (stops_level)
            symbol_info = mt5.symbol_info(signal.symbol)
            if not symbol_info:
                print(f"❌ Erro ao obter informações do símbolo {signal.symbol} para validar stops_level")
                return None
            point = getattr(symbol_info, 'point', None) or 0.0
            stops_level = getattr(symbol_info, 'trade_stops_level', None)
            if stops_level is None:
                stops_level = getattr(symbol_info, 'stops_level', 0)
            try:
                stops_level = float(stops_level)
            except Exception:
                stops_level = 0.0

            if point > 0 and stops_level > 0:
                sl_dist_pts = abs(signal.entry_price - signal.stop_loss) / point
                tp_dist_pts = abs(signal.take_profit - signal.entry_price) / point
                if sl_dist_pts < stops_level or tp_dist_pts < stops_level:
                    print(f"❌ INVALID_STOPS: Distância de SL/TP abaixo do mínimo exigido pelo broker para {signal.symbol}")
                    print(f"   stops_level: {stops_level:.0f} pts | SL: {sl_dist_pts:.1f} pts | TP: {tp_dist_pts:.1f} pts")

                    # Log de NO_ACTION para evitar retcode 10016
                    tick = mt5.symbol_info_tick(signal.symbol)
                    market_data = {
                        'symbol': signal.symbol,
                        'bid': tick.bid if tick else 0,
                        'ask': tick.ask if tick else 0,
                        'spread': (tick.ask - tick.bid) if tick else 0,
                        'volume': tick.volume if tick else 0,
                        'time': tick.time if tick else 0
                    }
                    order_type = mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL
                    decision_data = DecisionData(
                        symbol=signal.symbol,
                        action="NO_ACTION",
                        timestamp=datetime.now(),
                        market_data=market_data,
                        technical_analysis={
                            'direction': signal.direction,
                            'entry_price': signal.entry_price,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                            'volume': signal.volume,
                            'order_type': order_type,
                            'stops_level_min_points': stops_level,
                            'sl_distance_points': sl_dist_pts,
                            'tp_distance_points': tp_dist_pts
                        },
                        ml_predictions={
                            'confidence': signal.confidence,
                            'model_source': signal.model_source
                        },
                        reasoning=DecisionReasoning(
                            factors=[
                                "SL/TP abaixo da distância mínima exigida pelo broker (stops_level)",
                                f"stops_level: {stops_level:.0f} pts",
                                f"SL distance: {sl_dist_pts:.1f} pts",
                                f"TP distance: {tp_dist_pts:.1f} pts"
                            ],
                            confidence_score=float(signal.confidence),
                            risk_assessment="Risco de execução - ordem evitada para prevenir INVALID_STOPS (10016)"
                        ),
                        bot_action=BotAction(
                            action_type="NO_ACTION",
                            parameters={
                                'reason': 'INVALID_STOPS_PREVENTED',
                                'stops_level_points': stops_level,
                                'sl_distance_points': sl_dist_pts,
                                'tp_distance_points': tp_dist_pts
                            },
                            result="Ajuste SL/TP ou reavalie parâmetros de risco (ATR/pips mínimos)"
                        )
                    )
                    self.detailed_logger.log_decision(decision_data)
                    return None

            # 3) Preparar request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": signal.volume,
                "type": mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": signal.entry_price,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "deviation": 20,
                "magic": 12345,
                "comment": f"AutoTrader_{signal.model_source}_{signal.confidence:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # 4) Log e execução
            print(f"🔍 Executando trade: {signal.symbol} {signal.direction}")
            print(f"   Entry: {signal.entry_price:.5f} | SL: {signal.stop_loss:.5f} | TP: {signal.take_profit:.5f}")
            print(f"   Volume: {signal.volume} | Confiança: {signal.confidence:.2f}%")

            result = mt5.order_send(request)

            # Dados de mercado para logging
            tick = mt5.symbol_info_tick(signal.symbol)
            market_data = {
                'symbol': signal.symbol,
                'bid': tick.bid if tick else 0,
                'ask': tick.ask if tick else 0,
                'spread': (tick.ask - tick.bid) if tick else 0,
                'volume': tick.volume if tick else 0,
                'time': tick.time if tick else 0
            }

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"❌ Erro ao executar trade: {result.comment}")
                decision_data = DecisionData(
                    symbol=signal.symbol,
                    action="EXECUTE_TRADE",
                    timestamp=datetime.now(),
                    market_data=market_data,
                    technical_analysis={
                        'direction': signal.direction,
                        'entry_price': signal.entry_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'volume': signal.volume,
                        'order_type': request["type"]
                    },
                    ml_predictions={
                        'confidence': signal.confidence,
                        'model_source': signal.model_source
                    },
                    reasoning=DecisionReasoning(
                        factors=[
                            f"Falha na execução da ordem: {result.comment}",
                            f"Código de retorno: {result.retcode}",
                            f"Comentário MT5: {result.comment}",
                            f"Sinal válido gerado com confiança {signal.confidence:.3f}"
                        ],
                        confidence_score=float(signal.confidence),
                        risk_assessment="Risco de execução - ordem rejeitada pelo broker"
                    ),
                    bot_action=BotAction(
                        action_type="EXECUTION_FAILED",
                        parameters={"retcode": result.retcode, "error": f"MT5 rejeitou ordem: {result.comment}"},
                        result="Aguardar próxima oportunidade"
                    )
                )
                self.detailed_logger.log_decision(decision_data)
                return None

            # 5) Registrar posição ativa e logar sucesso
            self.active_positions[signal.symbol] = {
                'ticket': result.order,
                'direction': signal.direction,
                'volume': signal.volume,
                'open_time': datetime.now()
            }

            decision_data = DecisionData(
                symbol=signal.symbol,
                action="EXECUTE_TRADE",
                timestamp=datetime.now(),
                market_data=market_data,
                technical_analysis={
                    'direction': signal.direction,
                    'entry_price': signal.entry_price,
                    'executed_price': result.price if hasattr(result, 'price') else signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'volume': signal.volume,
                    'slippage': abs((getattr(result, 'price', signal.entry_price)) - signal.entry_price),
                    'order_type': request["type"],
                    'ticket': result.order
                },
                ml_predictions={
                    'confidence': signal.confidence,
                    'model_source': signal.model_source
                },
                reasoning=DecisionReasoning(
                    factors=[
                        f"Ordem executada com sucesso - Ticket #{result.order}",
                        f"Preço executado: {(getattr(result, 'price', signal.entry_price)):.5f}",
                        f"Slippage: {abs((getattr(result, 'price', signal.entry_price)) - signal.entry_price):.5f}",
                        f"SL/TP configurados corretamente",
                        f"Volume: {signal.volume}"
                    ],
                    confidence_score=float(signal.confidence),
                    risk_assessment="Risco controlado - posição aberto com SL/TP"
                ),
                bot_action=BotAction(
                    action_type="TRADE_EXECUTED",
                    parameters={"ticket": result.order, "confidence": signal.confidence, "executed_price": getattr(result, 'price', signal.entry_price)},
                    result=f"Lucro potencial: {abs(signal.take_profit - (getattr(result, 'price', signal.entry_price))):.5f}, Risco máximo: {abs((getattr(result, 'price', signal.entry_price)) - signal.stop_loss):.5f}"
                )
            )
            self.detailed_logger.log_decision(decision_data)

            return TradeExecution(
                ticket=result.order,
                symbol=signal.symbol,
                type=request["type"],
                volume=signal.volume,
                price=getattr(result, 'price', signal.entry_price),
                sl=signal.stop_loss,
                tp=signal.take_profit,
                comment=request["comment"],
                timestamp=datetime.now()
            )

        except Exception as e:
            print(f"❌ Erro ao executar trade: {e}")
            return None
            
    def _manage_open_positions(self):
        """Gerencia posições abertas"""
        try:
            positions = mt5.positions_get()
            if not positions:
                self.active_positions.clear()
                return
                
            # Atualizar posições ativas
            current_symbols = {pos.symbol for pos in positions}
            
            # Remover posições fechadas
            closed_symbols = set(self.active_positions.keys()) - current_symbols
            for symbol in closed_symbols:
                del self.active_positions[symbol]
                print(f"📊 Posição fechada: {symbol}")
                
        except Exception as e:
            print(f"❌ Erro ao gerenciar posições: {e}")
            
    def _log_signal_and_execution(self, signal: TradeSignal, execution: TradeExecution):
        """Registra sinal e execução no banco"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Inserir sinal
            cursor.execute('''
                INSERT INTO signals (timestamp, symbol, direction, confidence, entry_price, 
                                   stop_loss, take_profit, volume, model_source, executed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.timestamp.isoformat(),
                signal.symbol,
                signal.direction,
                signal.confidence,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit,
                signal.volume,
                signal.model_source,
                True
            ))
            
            signal_id = cursor.lastrowid
            
            # Inserir execução
            cursor.execute('''
                INSERT INTO executions (signal_id, ticket, symbol, type, volume, open_price,
                                      sl, tp, comment, timestamp, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_id,
                execution.ticket,
                execution.symbol,
                execution.type,
                execution.volume,
                execution.price,
                execution.sl,
                execution.tp,
                execution.comment,
                execution.timestamp.isoformat(),
                'OPEN'
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Erro ao registrar no banco: {e}")
            
    def get_performance_stats(self) -> Dict:
        """Retorna estatísticas de performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Estatísticas gerais
            cursor.execute('SELECT COUNT(*) FROM signals WHERE executed = 1')
            total_signals = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM executions')
            total_executions = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM executions WHERE profit > 0')
            result = cursor.fetchone()
            winning_trades = result[0] if result else 0
            
            cursor.execute('SELECT SUM(profit) FROM executions WHERE profit IS NOT NULL')
            result = cursor.fetchone()
            total_profit = result[0] if result and result[0] is not None else 0.0
            
            conn.close()
            
            win_rate = (winning_trades / total_executions * 100) if total_executions > 0 else 0
            
            return {
                'total_signals': total_signals,
                'total_executions': total_executions,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'daily_trades': self.daily_trades_count,
                'active_positions': len(self.active_positions)
            }
            
        except Exception as e:
            print(f"❌ Erro ao obter estatísticas: {e}")
            return {}

def main():
    """Função principal"""
    trader = AutoTrader()
    
    try:
        trader.start_trading()
        
        print("\nAuto Trader em execução...")
        print("Pressione Ctrl+C para parar")
        
        # Loop de status
        while trader.running:
            time.sleep(60)  # Mostrar status a cada minuto
            stats = trader.get_performance_stats()
            
            print(f"\n📊 Status: Sinais: {stats.get('total_signals', 0)} | "
                  f"Trades: {stats.get('total_executions', 0)} | "
                  f"Win Rate: {stats.get('win_rate', 0):.1f}% | "
                  f"Posições: {stats.get('active_positions', 0)}")
            
    except KeyboardInterrupt:
        print("\n🛑 Parando Auto Trader...")
        trader.stop_trading()
        
if __name__ == "__main__":
    main()