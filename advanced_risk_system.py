#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Avançado de Gestão de Risco para Trading
Integra ATR, volatilidade, Kelly Criterion e funcionalidades avançadas
Autor: Sistema Forex ML
Data: 2024
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import warnings
from scipy import stats
from scipy.optimize import minimize_scalar

# Importar classes existentes
from risk_manager import RiskManager, RiskParameters, RiskLevel, PositionSizeMethod
from kelly_position_sizing import KellyPositionSizing, TradingStats

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ATRMethod(Enum):
    """Métodos de cálculo do ATR"""
    SIMPLE = "simple"  # ATR simples
    EXPONENTIAL = "exponential"  # ATR com média móvel exponencial
    ADAPTIVE = "adaptive"  # ATR adaptativo baseado na volatilidade
    WILDER = "wilder"  # ATR original de Wilder

class VolatilityRegime(Enum):
    """Regimes de volatilidade do mercado"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"

class TrailingStopType(Enum):
    """Tipos de trailing stop"""
    ATR_BASED = "atr_based"
    PERCENTAGE = "percentage"
    TECHNICAL_LEVELS = "technical_levels"
    VOLATILITY_ADJUSTED = "volatility_adjusted"

@dataclass
class ATRParameters:
    """Parâmetros para cálculo do ATR"""
    period: int = 14
    method: ATRMethod = ATRMethod.WILDER
    multiplier_sl: float = 2.0  # Multiplicador para Stop Loss
    multiplier_tp: float = 3.0  # Multiplicador para Take Profit
    min_multiplier: float = 1.0
    max_multiplier: float = 5.0
    volatility_adjustment: bool = True
    session_adjustment: bool = True  # Ajuste por sessão de trading

@dataclass
class VolatilityParameters:
    """Parâmetros para análise de volatilidade"""
    lookback_period: int = 20
    regime_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.5,      # Percentil 25
        'normal': 1.0,   # Mediana
        'high': 1.5,     # Percentil 75
        'extreme': 2.0   # Percentil 90
    })
    adjustment_factors: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.8,      # Reduzir SL/TP em baixa volatilidade
        'normal': 1.0,   # Manter padrão
        'high': 1.2,     # Aumentar SL/TP em alta volatilidade
        'extreme': 1.5   # Aumentar significativamente
    })

@dataclass
class TrailingStopParameters:
    """Parâmetros para trailing stop"""
    type: TrailingStopType = TrailingStopType.ATR_BASED
    activation_distance: float = 1.5  # Distância para ativar trailing
    trail_distance: float = 1.0       # Distância do trailing
    step_size: float = 0.5            # Tamanho do passo
    max_trail_distance: float = 3.0   # Distância máxima
    use_breakeven: bool = True        # Mover para breakeven
    breakeven_trigger: float = 1.0    # Trigger para breakeven

@dataclass
class GapProtectionParameters:
    """Parâmetros para proteção contra gaps"""
    enabled: bool = True
    max_gap_threshold: float = 0.005  # 0.5% gap máximo
    pre_news_buffer: int = 30         # Minutos antes de notícias
    post_news_buffer: int = 15        # Minutos após notícias
    weekend_protection: bool = True   # Proteção fim de semana
    correlation_threshold: float = 0.8 # Correlação para reduzir exposição
    max_exposure_reduction: float = 0.5 # Redução máxima da exposição

@dataclass
class TechnicalLevelsParameters:
    """Parâmetros para análise de níveis técnicos"""
    support_resistance_period: int = 50
    fibonacci_levels: List[float] = field(default_factory=lambda: [0.236, 0.382, 0.5, 0.618, 0.786])
    psychological_levels: bool = True  # 00, 50 levels
    pivot_points: bool = True
    volume_profile: bool = True
    level_strength_threshold: int = 3  # Mínimo de toques para validar nível
    proximity_threshold: float = 0.001 # Proximidade para considerar nível

@dataclass
class ScaledTakeProfitParameters:
    """Parâmetros para Take Profit escalonado"""
    enabled: bool = True
    levels: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])  # Múltiplos do ATR
    position_percentages: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])  # % da posição
    trail_after_first: bool = True     # Trailing após primeiro TP
    dynamic_adjustment: bool = True    # Ajuste dinâmico baseado na volatilidade

class AdvancedRiskSystem:
    """
    Sistema Avançado de Gestão de Risco
    Integra todas as funcionalidades de risco avançadas
    """
    
    def __init__(self, 
                 mt5_client,
                 risk_params: RiskParameters = None,
                 atr_params: ATRParameters = None,
                 volatility_params: VolatilityParameters = None,
                 trailing_params: TrailingStopParameters = None,
                 gap_protection_params: GapProtectionParameters = None,
                 technical_params: TechnicalLevelsParameters = None,
                 scaled_tp_params: ScaledTakeProfitParameters = None,
                 logger: Optional[logging.Logger] = None):
        
        self.mt5_client = mt5_client
        self.logger = logger or logging.getLogger(__name__)
        
        # Inicializar parâmetros
        self.risk_params = risk_params or RiskParameters()
        self.atr_params = atr_params or ATRParameters()
        self.volatility_params = volatility_params or VolatilityParameters()
        self.trailing_params = trailing_params or TrailingStopParameters()
        self.gap_protection_params = gap_protection_params or GapProtectionParameters()
        self.technical_params = technical_params or TechnicalLevelsParameters()
        self.scaled_tp_params = scaled_tp_params or ScaledTakeProfitParameters()
        
        # Inicializar componentes
        self.risk_manager = RiskManager(mt5_client, self.risk_params)
        self.kelly_sizer = KellyPositionSizing(
            max_kelly_fraction=0.25,
            min_kelly_fraction=0.01,
            logger=self.logger
        )
        
        # Cache para dados históricos
        self.price_cache = {}
        self.volatility_cache = {}
        self.atr_cache = {}
        self.levels_cache = {}
        
        # Estado do trailing stop
        self.trailing_stops = {}
        
        self.logger.info("Sistema Avançado de Gestão de Risco inicializado")
    
    def calculate_advanced_atr(self, symbol: str, 
                              timeframe: int = 16385,  # M1
                              custom_params: Optional[ATRParameters] = None) -> Dict[str, float]:
        """
        Calcula ATR avançado com diferentes métodos
        
        Args:
            symbol: Símbolo do ativo
            timeframe: Timeframe para análise
            custom_params: Parâmetros customizados
            
        Returns:
            Dict com diferentes valores de ATR
        """
        params = custom_params or self.atr_params
        
        try:
            # Obter dados históricos
            rates = self.mt5_client.get_rates(symbol, timeframe, params.period + 20)
            
            if rates is None or len(rates) < params.period:
                self.logger.warning(f"Dados insuficientes para ATR de {symbol}")
                return self._get_fallback_atr(symbol)
            
            # Converter para DataFrame
            df = pd.DataFrame(rates)
            
            # Calcular True Range
            df['prev_close'] = df['close'].shift(1)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['prev_close'])
            df['tr3'] = abs(df['low'] - df['prev_close'])
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            # Calcular ATR por diferentes métodos
            atr_values = {}
            
            if params.method == ATRMethod.SIMPLE:
                atr_values['simple'] = df['tr'].rolling(params.period).mean().iloc[-1]
            
            elif params.method == ATRMethod.EXPONENTIAL:
                atr_values['exponential'] = df['tr'].ewm(span=params.period).mean().iloc[-1]
            
            elif params.method == ATRMethod.WILDER:
                # Método original de Wilder (RMA)
                alpha = 1.0 / params.period
                atr_series = df['tr'].copy()
                atr_wilder = [atr_series.iloc[0]]
                
                for i in range(1, len(atr_series)):
                    atr_val = alpha * atr_series.iloc[i] + (1 - alpha) * atr_wilder[-1]
                    atr_wilder.append(atr_val)
                
                atr_values['wilder'] = atr_wilder[-1]
            
            elif params.method == ATRMethod.ADAPTIVE:
                # ATR adaptativo baseado na volatilidade recente
                volatility = df['close'].pct_change().rolling(params.period).std().iloc[-1]
                adaptive_period = max(5, min(30, int(params.period * (1 + volatility * 10))))
                atr_values['adaptive'] = df['tr'].rolling(adaptive_period).mean().iloc[-1]
            
            # ATR principal
            main_atr = atr_values.get(params.method.value, atr_values.get('wilder', 0))
            
            # Ajustes por volatilidade
            if params.volatility_adjustment:
                volatility_regime = self._determine_volatility_regime(symbol, df)
                volatility_factor = self.volatility_params.adjustment_factors.get(
                    volatility_regime.value, 1.0
                )
                main_atr *= volatility_factor
            
            # Ajustes por sessão de trading
            if params.session_adjustment:
                session_factor = self._get_session_adjustment_factor()
                main_atr *= session_factor
            
            # Adicionar ATR principal ao resultado
            atr_values['main'] = main_atr
            atr_values['volatility_adjusted'] = main_atr
            
            # Cache do resultado
            self.atr_cache[symbol] = {
                'values': atr_values,
                'timestamp': datetime.now(),
                'params': params
            }
            
            return atr_values
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular ATR avançado para {symbol}: {e}")
            return self._get_fallback_atr(symbol)
    
    def _get_fallback_atr(self, symbol: str) -> Dict[str, float]:
        """Retorna valores de ATR de fallback"""
        symbol_info = self.mt5_client.get_symbol_info(symbol)
        pip_size = symbol_info.get('point', 0.0001) if symbol_info else 0.0001
        fallback_atr = 50 * pip_size  # 50 pips como padrão
        
        return {
            'main': fallback_atr,
            'simple': fallback_atr,
            'wilder': fallback_atr,
            'exponential': fallback_atr,
            'adaptive': fallback_atr,
            'volatility_adjusted': fallback_atr
        }
    
    def _determine_volatility_regime(self, symbol: str, df: pd.DataFrame) -> VolatilityRegime:
        """Determina o regime de volatilidade atual"""
        try:
            # Calcular volatilidade realizada
            returns = df['close'].pct_change().dropna()
            current_vol = returns.rolling(self.volatility_params.lookback_period).std().iloc[-1]
            
            # Calcular percentis históricos
            historical_vol = returns.rolling(252).std()  # Volatilidade anual
            vol_percentiles = historical_vol.quantile([0.25, 0.5, 0.75, 0.9])
            
            # Determinar regime
            if current_vol <= vol_percentiles[0.25]:
                return VolatilityRegime.LOW
            elif current_vol <= vol_percentiles[0.5]:
                return VolatilityRegime.NORMAL
            elif current_vol <= vol_percentiles[0.75]:
                return VolatilityRegime.HIGH
            else:
                return VolatilityRegime.EXTREME
                
        except Exception as e:
            self.logger.error(f"Erro ao determinar regime de volatilidade: {e}")
            return VolatilityRegime.NORMAL
    
    def _get_session_adjustment_factor(self) -> float:
        """Calcula fator de ajuste baseado na sessão de trading"""
        try:
            current_hour = datetime.now().hour
            
            # Sessões de trading (UTC)
            # Sydney: 21:00-06:00, Tokyo: 23:00-08:00
            # London: 07:00-16:00, New York: 12:00-21:00
            
            if 7 <= current_hour <= 16:  # Sessão de Londres
                return 1.2  # Maior volatilidade
            elif 12 <= current_hour <= 21:  # Sessão de Nova York
                return 1.3  # Maior volatilidade
            elif 23 <= current_hour or current_hour <= 8:  # Sessão de Tokyo
                return 1.0  # Volatilidade normal
            else:  # Sessão de Sydney ou entre sessões
                return 0.8  # Menor volatilidade
                
        except Exception:
            return 1.0
    
    def calculate_optimal_sl_tp(self, symbol: str, entry_price: float, 
                               direction: int, confidence: float = 1.0,
                               custom_atr_params: Optional[ATRParameters] = None) -> Tuple[float, float, Dict]:
        """
        Calcula Stop Loss e Take Profit ótimos baseados em ATR e volatilidade
        
        Args:
            symbol: Símbolo do ativo
            entry_price: Preço de entrada
            direction: 1 para compra, -1 para venda
            confidence: Nível de confiança na operação (0-1)
            custom_atr_params: Parâmetros customizados de ATR
            
        Returns:
            Tuple[float, float, Dict]: (stop_loss, take_profit, metadata)
        """
        try:
            # Calcular ATR avançado
            atr_values = self.calculate_advanced_atr(symbol, custom_params=custom_atr_params)
            main_atr = atr_values['main']
            
            # Obter informações do símbolo
            symbol_info = self.mt5_client.get_symbol_info(symbol)
            digits = symbol_info.get('digits', 5) if symbol_info else 5
            
            # Ajustar multiplicadores baseado na confiança
            confidence_factor = 0.5 + (confidence * 0.5)  # 0.5 a 1.0
            
            sl_multiplier = self.atr_params.multiplier_sl * confidence_factor
            tp_multiplier = self.atr_params.multiplier_tp * confidence_factor
            
            # Limitar multiplicadores
            sl_multiplier = max(self.atr_params.min_multiplier, 
                              min(sl_multiplier, self.atr_params.max_multiplier))
            tp_multiplier = max(self.atr_params.min_multiplier, 
                              min(tp_multiplier, self.atr_params.max_multiplier))
            
            # Calcular distâncias
            sl_distance = main_atr * sl_multiplier
            tp_distance = main_atr * tp_multiplier
            
            # Calcular preços
            if direction == 1:  # Compra
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + tp_distance
            else:  # Venda
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - tp_distance
            
            # Arredondar para o número correto de casas decimais
            stop_loss = round(stop_loss, digits)
            take_profit = round(take_profit, digits)
            
            # Metadados
            metadata = {
                'atr_value': main_atr,
                'sl_multiplier': sl_multiplier,
                'tp_multiplier': tp_multiplier,
                'sl_distance': sl_distance,
                'tp_distance': tp_distance,
                'confidence_factor': confidence_factor,
                'risk_reward_ratio': tp_distance / sl_distance,
                'atr_method': self.atr_params.method.value,
                'volatility_adjusted': self.atr_params.volatility_adjustment
            }
            
            self.logger.info(f"SL/TP calculado para {symbol}: SL={stop_loss}, TP={take_profit}, RR={metadata['risk_reward_ratio']:.2f}")
            
            return stop_loss, take_profit, metadata
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular SL/TP ótimo para {symbol}: {e}")
            # Fallback para método básico
            return self.risk_manager.calculate_optimal_stops(symbol, entry_price, direction)
    
    def calculate_kelly_optimized_position(self, symbol: str, entry_price: float,
                                         stop_loss: float, take_profit: float,
                                         direction: int, confidence: float = 1.0,
                                         account_balance: float = None) -> Dict[str, float]:
        """
        Calcula tamanho da posição otimizado usando Kelly Criterion integrado
        
        Args:
            symbol: Símbolo do ativo
            entry_price: Preço de entrada
            stop_loss: Preço do stop loss
            take_profit: Preço do take profit
            direction: 1 para compra, -1 para venda
            confidence: Nível de confiança (0-1)
            account_balance: Saldo da conta
            
        Returns:
            Dict com informações da posição otimizada
        """
        try:
            # Obter saldo da conta se não fornecido
            if account_balance is None:
                account_info = self.mt5_client.get_account_info()
                account_balance = account_info.get('balance', 10000) if account_info else 10000
            
            # Calcular risco e recompensa
            if direction == 1:  # Compra
                risk_per_unit = entry_price - stop_loss
                reward_per_unit = take_profit - entry_price
            else:  # Venda
                risk_per_unit = stop_loss - entry_price
                reward_per_unit = entry_price - take_profit
            
            risk_reward_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0
            
            # Obter estatísticas históricas para Kelly
            trading_stats = self._get_trading_statistics(symbol)
            
            # Ajustar win rate baseado na confiança
            adjusted_win_rate = min(0.9, trading_stats.win_rate * (0.5 + confidence * 0.5))
            
            # Calcular Kelly Fraction
            kelly_fraction = self.kelly_sizer.calculate_kelly_fraction(
                win_rate=adjusted_win_rate,
                avg_win=trading_stats.avg_win,
                avg_loss=trading_stats.avg_loss,
                method='modified'  # Usar Kelly modificado para maior segurança
            )
            
            # Aplicar limitações de risco
            max_risk_amount = account_balance * self.risk_params.max_risk_per_trade
            kelly_risk_amount = account_balance * kelly_fraction
            
            # Usar o menor entre Kelly e limite de risco
            final_risk_amount = min(kelly_risk_amount, max_risk_amount)
            
            # Calcular tamanho da posição
            if risk_per_unit > 0:
                position_size = final_risk_amount / risk_per_unit
            else:
                position_size = 0
            
            # Obter informações do símbolo para ajustar lote
            symbol_info = self.mt5_client.get_symbol_info(symbol)
            if symbol_info:
                min_lot = symbol_info.get('volume_min', 0.01)
                max_lot = symbol_info.get('volume_max', 100.0)
                lot_step = symbol_info.get('volume_step', 0.01)
                
                # Ajustar para múltiplos do lot_step
                position_size = round(position_size / lot_step) * lot_step
                position_size = max(min_lot, min(position_size, max_lot))
            
            # Calcular métricas finais
            final_risk = position_size * risk_per_unit
            final_reward = position_size * reward_per_unit
            risk_percentage = (final_risk / account_balance) * 100
            
            result = {
                'position_size': position_size,
                'risk_amount': final_risk,
                'reward_amount': final_reward,
                'risk_percentage': risk_percentage,
                'kelly_fraction': kelly_fraction,
                'adjusted_win_rate': adjusted_win_rate,
                'risk_reward_ratio': risk_reward_ratio,
                'confidence_factor': confidence,
                'account_balance': account_balance,
                'max_risk_allowed': max_risk_amount
            }
            
            self.logger.info(f"Posição Kelly otimizada para {symbol}: {position_size:.2f} lotes, Risco: {risk_percentage:.2f}%")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular posição Kelly para {symbol}: {e}")
            # Fallback para cálculo básico
            return self._calculate_basic_position_size(symbol, entry_price, stop_loss, account_balance)
    
    def _get_trading_statistics(self, symbol: str) -> TradingStats:
        """
        Obtém estatísticas históricas de trading para o símbolo
        """
        try:
            # Tentar obter do histórico de trades do MT5
            # Por enquanto, usar valores padrão baseados em análise histórica
            return TradingStats(
                total_trades=100,
                winning_trades=55,
                losing_trades=45,
                win_rate=0.55,
                avg_win=150.0,
                avg_loss=100.0,
                profit_factor=1.65,
                max_drawdown=0.15,
                sharpe_ratio=1.2
            )
        except Exception as e:
            self.logger.warning(f"Erro ao obter estatísticas para {symbol}: {e}")
            # Valores conservadores padrão
            return TradingStats(
                total_trades=50,
                winning_trades=25,
                losing_trades=25,
                win_rate=0.50,
                avg_win=100.0,
                avg_loss=100.0,
                profit_factor=1.0,
                max_drawdown=0.20,
                sharpe_ratio=0.8
            )
    
    def _calculate_basic_position_size(self, symbol: str, entry_price: float,
                                     stop_loss: float, account_balance: float) -> Dict[str, float]:
        """
        Cálculo básico de posição como fallback
        """
        risk_per_unit = abs(entry_price - stop_loss)
        max_risk = account_balance * self.risk_params.max_risk_per_trade
        position_size = max_risk / risk_per_unit if risk_per_unit > 0 else 0.01
        
        return {
            'position_size': round(position_size, 2),
            'risk_amount': max_risk,
            'risk_percentage': self.risk_params.max_risk_per_trade * 100,
            'kelly_fraction': 0.02,  # Conservador
            'method': 'basic_fallback'
        }
    
    def initialize_trailing_stop(self, ticket: int, symbol: str, entry_price: float,
                                direction: int, initial_sl: float) -> Dict[str, any]:
        """
        Inicializa trailing stop para uma posição
        
        Args:
            ticket: Ticket da posição
            symbol: Símbolo do ativo
            entry_price: Preço de entrada
            direction: 1 para compra, -1 para venda
            initial_sl: Stop loss inicial
            
        Returns:
            Dict com configuração do trailing stop
        """
        try:
            # Calcular ATR para trailing dinâmico
            atr_values = self.calculate_advanced_atr(symbol)
            main_atr = atr_values['main']
            
            # Configurar trailing stop
            trailing_config = {
                'ticket': ticket,
                'symbol': symbol,
                'entry_price': entry_price,
                'direction': direction,
                'current_sl': initial_sl,
                'best_price': entry_price,
                'trail_distance': main_atr * self.trailing_params.trail_distance,
                'activation_distance': main_atr * self.trailing_params.activation_distance,
                'step_size': main_atr * self.trailing_params.step_size,
                'activated': False,
                'breakeven_moved': False,
                'last_update': datetime.now(),
                'type': self.trailing_params.type.value,
                'atr_value': main_atr
            }
            
            # Armazenar configuração
            self.trailing_stops[ticket] = trailing_config
            
            self.logger.info(f"Trailing stop inicializado para ticket {ticket}: {trailing_config}")
            
            return trailing_config
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar trailing stop para ticket {ticket}: {e}")
            return {}
    
    def update_trailing_stop(self, ticket: int, current_price: float) -> Optional[float]:
        """
        Atualiza trailing stop baseado no preço atual
        
        Args:
            ticket: Ticket da posição
            current_price: Preço atual
            
        Returns:
            Novo stop loss se houver mudança, None caso contrário
        """
        try:
            if ticket not in self.trailing_stops:
                self.logger.warning(f"Trailing stop não encontrado para ticket {ticket}")
                return None
            
            config = self.trailing_stops[ticket]
            direction = config['direction']
            entry_price = config['entry_price']
            current_sl = config['current_sl']
            
            # Atualizar melhor preço
            if direction == 1:  # Compra
                if current_price > config['best_price']:
                    config['best_price'] = current_price
            else:  # Venda
                if current_price < config['best_price']:
                    config['best_price'] = current_price
            
            # Verificar se deve ativar trailing
            if not config['activated']:
                profit_distance = abs(config['best_price'] - entry_price)
                if profit_distance >= config['activation_distance']:
                    config['activated'] = True
                    self.logger.info(f"Trailing stop ativado para ticket {ticket}")
            
            # Mover para breakeven se configurado
            if (self.trailing_params.use_breakeven and 
                not config['breakeven_moved'] and 
                config['activated']):
                
                breakeven_trigger = config['atr_value'] * self.trailing_params.breakeven_trigger
                profit_distance = abs(config['best_price'] - entry_price)
                
                if profit_distance >= breakeven_trigger:
                    new_sl = entry_price
                    config['breakeven_moved'] = True
                    config['current_sl'] = new_sl
                    config['last_update'] = datetime.now()
                    
                    self.logger.info(f"Stop loss movido para breakeven: ticket {ticket}, novo SL: {new_sl}")
                    return new_sl
            
            # Atualizar trailing stop se ativado
            if config['activated']:
                new_sl = None
                
                if direction == 1:  # Compra
                    potential_sl = config['best_price'] - config['trail_distance']
                    if potential_sl > current_sl:
                        new_sl = potential_sl
                else:  # Venda
                    potential_sl = config['best_price'] + config['trail_distance']
                    if potential_sl < current_sl:
                        new_sl = potential_sl
                
                if new_sl is not None:
                    # Verificar step size
                    sl_change = abs(new_sl - current_sl)
                    if sl_change >= config['step_size']:
                        config['current_sl'] = new_sl
                        config['last_update'] = datetime.now()
                        
                        self.logger.info(f"Trailing stop atualizado: ticket {ticket}, novo SL: {new_sl}")
                        return new_sl
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar trailing stop para ticket {ticket}: {e}")
            return None
    
    def check_gap_protection(self, symbol: str, current_price: float,
                           last_price: float, position_size: float) -> Dict[str, any]:
        """
        Verifica proteção contra gaps e sugere ajustes
        
        Args:
            symbol: Símbolo do ativo
            current_price: Preço atual
            last_price: Último preço conhecido
            position_size: Tamanho da posição atual
            
        Returns:
            Dict com recomendações de proteção
        """
        try:
            if not self.gap_protection_params.enabled:
                return {'protection_needed': False}
            
            # Calcular gap percentual
            gap_percentage = abs(current_price - last_price) / last_price
            
            protection_info = {
                'gap_percentage': gap_percentage,
                'gap_threshold': self.gap_protection_params.max_gap_threshold,
                'protection_needed': False,
                'recommended_actions': []
            }
            
            # Verificar se gap excede threshold
            if gap_percentage > self.gap_protection_params.max_gap_threshold:
                protection_info['protection_needed'] = True
                protection_info['recommended_actions'].append('reduce_position_size')
                
                # Calcular redução recomendada
                reduction_factor = min(
                    self.gap_protection_params.max_exposure_reduction,
                    gap_percentage / self.gap_protection_params.max_gap_threshold
                )
                
                new_position_size = position_size * (1 - reduction_factor)
                protection_info['recommended_position_size'] = new_position_size
                protection_info['reduction_factor'] = reduction_factor
            
            # Verificar proximidade de notícias
            if self._is_near_news_event():
                protection_info['news_risk'] = True
                protection_info['recommended_actions'].append('avoid_new_positions')
            
            # Verificar fim de semana
            if self.gap_protection_params.weekend_protection and self._is_weekend_risk():
                protection_info['weekend_risk'] = True
                protection_info['recommended_actions'].append('close_positions_before_weekend')
            
            return protection_info
            
        except Exception as e:
            self.logger.error(f"Erro na verificação de proteção contra gaps: {e}")
            return {'protection_needed': False, 'error': str(e)}
    
    def _is_near_news_event(self) -> bool:
        """
        Verifica se há eventos de notícias próximos
        """
        try:
            # Implementação simplificada - verificar horários típicos de notícias
            current_time = datetime.now()
            hour = current_time.hour
            minute = current_time.minute
            
            # Horários típicos de notícias importantes (UTC)
            news_times = [
                (8, 30),   # EUR news
                (12, 30),  # USD news
                (13, 30),  # USD news
                (14, 0),   # USD news
            ]
            
            for news_hour, news_minute in news_times:
                time_diff = abs((hour * 60 + minute) - (news_hour * 60 + news_minute))
                if time_diff <= self.gap_protection_params.pre_news_buffer:
                    return True
            
            return False
            
        except Exception:
             return False
    
    def analyze_technical_levels(self, symbol: str, current_price: float,
                               timeframe: int = 16385) -> Dict[str, any]:
        """
        Analisa níveis técnicos significativos (suporte/resistência)
        
        Args:
            symbol: Símbolo do ativo
            current_price: Preço atual
            timeframe: Timeframe para análise
            
        Returns:
            Dict com níveis técnicos identificados
        """
        try:
            # Obter dados históricos
            period = self.technical_params.support_resistance_period
            rates = self.mt5_client.get_rates(symbol, timeframe, period * 2)
            
            if rates is None or len(rates) < period:
                self.logger.warning(f"Dados insuficientes para análise técnica de {symbol}")
                return {'levels': [], 'error': 'insufficient_data'}
            
            df = pd.DataFrame(rates)
            
            # Identificar suportes e resistências
            support_levels = self._find_support_levels(df)
            resistance_levels = self._find_resistance_levels(df)
            
            # Calcular níveis de Fibonacci
            fibonacci_levels = self._calculate_fibonacci_levels(df)
            
            # Identificar níveis psicológicos
            psychological_levels = self._find_psychological_levels(current_price)
            
            # Calcular pivot points
            pivot_levels = self._calculate_pivot_points(df)
            
            # Consolidar todos os níveis
            all_levels = {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'fibonacci_levels': fibonacci_levels,
                'psychological_levels': psychological_levels,
                'pivot_levels': pivot_levels,
                'current_price': current_price,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Encontrar níveis mais próximos
            nearest_support = self._find_nearest_level(current_price, support_levels, 'below')
            nearest_resistance = self._find_nearest_level(current_price, resistance_levels, 'above')
            
            all_levels['nearest_support'] = nearest_support
            all_levels['nearest_resistance'] = nearest_resistance
            
            # Cache dos resultados
            self.levels_cache[symbol] = {
                'levels': all_levels,
                'timestamp': datetime.now(),
                'timeframe': timeframe
            }
            
            self.logger.info(f"Análise técnica concluída para {symbol}: {len(support_levels)} suportes, {len(resistance_levels)} resistências")
            
            return all_levels
            
        except Exception as e:
            self.logger.error(f"Erro na análise de níveis técnicos para {symbol}: {e}")
            return {'levels': [], 'error': str(e)}
    
    def _find_support_levels(self, df: pd.DataFrame) -> List[Dict[str, float]]:
        """Identifica níveis de suporte"""
        try:
            supports = []
            lows = df['low'].values
            
            # Encontrar mínimos locais
            for i in range(2, len(lows) - 2):
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    
                    # Verificar força do nível (quantas vezes foi testado)
                    level_price = lows[i]
                    strength = self._calculate_level_strength(df, level_price, 'support')
                    
                    if strength >= self.technical_params.level_strength_threshold:
                        supports.append({
                            'price': level_price,
                            'strength': strength,
                            'index': i,
                            'timestamp': df.iloc[i]['time']
                        })
            
            # Ordenar por força
            supports.sort(key=lambda x: x['strength'], reverse=True)
            return supports[:10]  # Top 10 níveis
            
        except Exception as e:
            self.logger.error(f"Erro ao encontrar suportes: {e}")
            return []
    
    def _find_resistance_levels(self, df: pd.DataFrame) -> List[Dict[str, float]]:
        """Identifica níveis de resistência"""
        try:
            resistances = []
            highs = df['high'].values
            
            # Encontrar máximos locais
            for i in range(2, len(highs) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    
                    # Verificar força do nível
                    level_price = highs[i]
                    strength = self._calculate_level_strength(df, level_price, 'resistance')
                    
                    if strength >= self.technical_params.level_strength_threshold:
                        resistances.append({
                            'price': level_price,
                            'strength': strength,
                            'index': i,
                            'timestamp': df.iloc[i]['time']
                        })
            
            # Ordenar por força
            resistances.sort(key=lambda x: x['strength'], reverse=True)
            return resistances[:10]  # Top 10 níveis
            
        except Exception as e:
            self.logger.error(f"Erro ao encontrar resistências: {e}")
            return []
    
    def _calculate_level_strength(self, df: pd.DataFrame, level_price: float, level_type: str) -> int:
        """Calcula a força de um nível técnico"""
        try:
            strength = 0
            threshold = level_price * self.technical_params.proximity_threshold
            
            for _, row in df.iterrows():
                if level_type == 'support':
                    if abs(row['low'] - level_price) <= threshold:
                        strength += 1
                else:  # resistance
                    if abs(row['high'] - level_price) <= threshold:
                        strength += 1
            
            return strength
            
        except Exception:
            return 0
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> List[Dict[str, float]]:
        """Calcula níveis de Fibonacci"""
        try:
            # Encontrar swing high e swing low
            high_price = df['high'].max()
            low_price = df['low'].min()
            
            price_range = high_price - low_price
            
            fib_levels = []
            for level in self.technical_params.fibonacci_levels:
                fib_price = high_price - (price_range * level)
                fib_levels.append({
                    'price': fib_price,
                    'level': level,
                    'type': 'fibonacci_retracement'
                })
            
            return fib_levels
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular Fibonacci: {e}")
            return []
    
    def _find_psychological_levels(self, current_price: float) -> List[Dict[str, float]]:
        """Identifica níveis psicológicos (00, 50)"""
        try:
            if not self.technical_params.psychological_levels:
                return []
            
            levels = []
            
            # Determinar a magnitude do preço
            if current_price >= 1.0:
                # Para preços >= 1.0, usar níveis .00 e .50
                base = int(current_price)
                levels.extend([
                    {'price': base, 'type': 'psychological_00'},
                    {'price': base + 0.5, 'type': 'psychological_50'},
                    {'price': base + 1.0, 'type': 'psychological_00'}
                ])
            else:
                # Para preços < 1.0, usar níveis baseados em pips
                pip_value = 0.0001 if '.' in str(current_price) else 0.01
                base_pips = int(current_price / pip_value)
                
                for offset in [-100, -50, 0, 50, 100]:
                    level_price = (base_pips + offset) * pip_value
                    if level_price > 0:
                        levels.append({
                            'price': level_price,
                            'type': f'psychological_{abs(offset)}'
                        })
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular níveis psicológicos: {e}")
            return []
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> List[Dict[str, float]]:
        """Calcula pivot points"""
        try:
            if not self.technical_params.pivot_points:
                return []
            
            # Usar dados do último período completo
            last_high = df['high'].iloc[-1]
            last_low = df['low'].iloc[-1]
            last_close = df['close'].iloc[-1]
            
            # Pivot Point clássico
            pivot = (last_high + last_low + last_close) / 3
            
            # Suportes e resistências
            r1 = 2 * pivot - last_low
            s1 = 2 * pivot - last_high
            r2 = pivot + (last_high - last_low)
            s2 = pivot - (last_high - last_low)
            r3 = last_high + 2 * (pivot - last_low)
            s3 = last_low - 2 * (last_high - pivot)
            
            pivot_levels = [
                {'price': pivot, 'type': 'pivot_point'},
                {'price': r1, 'type': 'resistance_1'},
                {'price': r2, 'type': 'resistance_2'},
                {'price': r3, 'type': 'resistance_3'},
                {'price': s1, 'type': 'support_1'},
                {'price': s2, 'type': 'support_2'},
                {'price': s3, 'type': 'support_3'}
            ]
            
            return pivot_levels
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular pivot points: {e}")
            return []
    
    def _find_nearest_level(self, current_price: float, levels: List[Dict], direction: str) -> Optional[Dict]:
        """Encontra o nível mais próximo em uma direção"""
        try:
            if not levels:
                return None
            
            if direction == 'above':
                # Encontrar o menor preço acima do atual
                above_levels = [l for l in levels if l['price'] > current_price]
                if above_levels:
                    return min(above_levels, key=lambda x: x['price'])
            else:  # below
                # Encontrar o maior preço abaixo do atual
                below_levels = [l for l in levels if l['price'] < current_price]
                if below_levels:
                    return max(below_levels, key=lambda x: x['price'])
            
            return None
            
        except Exception:
            return None
    
    def calculate_scaled_take_profit(self, symbol: str, entry_price: float, 
                                   direction: int, base_atr: float) -> List[Dict[str, float]]:
        """
        Calcula Take Profit escalonado com múltiplos níveis
        
        Args:
            symbol: Símbolo do ativo
            entry_price: Preço de entrada
            direction: 1 para compra, -1 para venda
            base_atr: ATR base para cálculos
            
        Returns:
            Lista de níveis de Take Profit
        """
        try:
            if not self.scaled_tp_params.enabled:
                return []
            
            tp_levels = []
            levels = self.scaled_tp_params.levels
            percentages = self.scaled_tp_params.position_percentages
            
            # Garantir que as listas tenham o mesmo tamanho
            min_length = min(len(levels), len(percentages))
            levels = levels[:min_length]
            percentages = percentages[:min_length]
            
            # Normalizar percentagens para somar 100%
            total_percentage = sum(percentages)
            if total_percentage != 1.0:
                percentages = [p / total_percentage for p in percentages]
            
            # Obter informações do símbolo
            symbol_info = self.mt5_client.get_symbol_info(symbol)
            digits = symbol_info.get('digits', 5) if symbol_info else 5
            
            # Ajuste dinâmico baseado na volatilidade
            volatility_adjustment = 1.0
            if self.scaled_tp_params.dynamic_adjustment:
                # Obter dados para análise de volatilidade
                rates = self.mt5_client.get_rates(symbol, 16385, 20)  # M1, 20 períodos
                if rates is not None and len(rates) >= 10:
                    df = pd.DataFrame(rates)
                    current_volatility = df['close'].pct_change().std()
                    avg_volatility = 0.001  # Volatilidade média esperada
                    volatility_adjustment = max(0.5, min(2.0, current_volatility / avg_volatility))
            
            # Calcular níveis de TP
            for i, (level_multiplier, percentage) in enumerate(zip(levels, percentages)):
                # Ajustar multiplicador pela volatilidade
                adjusted_multiplier = level_multiplier * volatility_adjustment
                
                # Calcular distância do TP
                tp_distance = base_atr * adjusted_multiplier
                
                # Calcular preço do TP
                if direction == 1:  # Compra
                    tp_price = entry_price + tp_distance
                else:  # Venda
                    tp_price = entry_price - tp_distance
                
                # Arredondar para o número correto de casas decimais
                tp_price = round(tp_price, digits)
                
                tp_levels.append({
                    'level': i + 1,
                    'price': tp_price,
                    'percentage': percentage,
                    'multiplier': adjusted_multiplier,
                    'distance': tp_distance,
                    'volatility_adjustment': volatility_adjustment
                })
            
            self.logger.info(f"TP escalonado calculado para {symbol}: {len(tp_levels)} níveis")
            
            return tp_levels
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular TP escalonado para {symbol}: {e}")
            return []
    
    def get_comprehensive_risk_analysis(self, symbol: str, entry_price: float,
                                      direction: int, confidence: float = 1.0) -> Dict[str, any]:
        """
        Análise completa de risco integrando todas as funcionalidades
        
        Args:
            symbol: Símbolo do ativo
            entry_price: Preço de entrada
            direction: 1 para compra, -1 para venda
            confidence: Nível de confiança (0-1)
            
        Returns:
            Dict com análise completa de risco
        """
        try:
            analysis = {
                'symbol': symbol,
                'entry_price': entry_price,
                'direction': direction,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            # 1. Calcular SL/TP baseado em ATR
            sl, tp, sl_tp_metadata = self.calculate_optimal_sl_tp(symbol, entry_price, direction, confidence)
            analysis['stop_loss'] = sl
            analysis['take_profit'] = tp
            analysis['sl_tp_metadata'] = sl_tp_metadata
            
            # 2. Calcular posição otimizada com Kelly
            position_info = self.calculate_kelly_optimized_position(
                symbol, entry_price, sl, tp, direction, confidence
            )
            analysis['position_sizing'] = position_info
            
            # 3. Análise de níveis técnicos
            technical_levels = self.analyze_technical_levels(symbol, entry_price)
            analysis['technical_levels'] = technical_levels
            
            # 4. TP escalonado
            scaled_tp = self.calculate_scaled_take_profit(
                symbol, entry_price, direction, sl_tp_metadata.get('atr_value', 0.001)
            )
            analysis['scaled_take_profit'] = scaled_tp
            
            # 5. Verificação de proteção contra gaps
            gap_protection = self.check_gap_protection(symbol, entry_price, entry_price, 
                                                     position_info.get('position_size', 0.01))
            analysis['gap_protection'] = gap_protection
            
            # 6. Recomendações finais
            recommendations = self._generate_risk_recommendations(analysis)
            analysis['recommendations'] = recommendations
            
            # 7. Score de risco geral
            risk_score = self._calculate_overall_risk_score(analysis)
            analysis['risk_score'] = risk_score
            
            self.logger.info(f"Análise completa de risco concluída para {symbol}: Score {risk_score}/100")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Erro na análise completa de risco para {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _generate_risk_recommendations(self, analysis: Dict) -> List[str]:
        """Gera recomendações baseadas na análise de risco"""
        recommendations = []
        
        try:
            # Verificar relação risco/recompensa
            rr_ratio = analysis.get('sl_tp_metadata', {}).get('risk_reward_ratio', 0)
            if rr_ratio < 1.5:
                recommendations.append(f"Relação risco/recompensa baixa ({rr_ratio:.2f}). Considere ajustar TP.")
            
            # Verificar tamanho da posição
            risk_pct = analysis.get('position_sizing', {}).get('risk_percentage', 0)
            if risk_pct > 3.0:
                recommendations.append(f"Risco por operação alto ({risk_pct:.1f}%). Considere reduzir posição.")
            
            # Verificar proteção contra gaps
            gap_info = analysis.get('gap_protection', {})
            if gap_info.get('protection_needed', False):
                recommendations.append("Proteção contra gaps recomendada devido à alta volatilidade.")
            
            # Verificar níveis técnicos
            tech_levels = analysis.get('technical_levels', {})
            nearest_resistance = tech_levels.get('nearest_resistance')
            nearest_support = tech_levels.get('nearest_support')
            
            entry_price = analysis.get('entry_price', 0)
            direction = analysis.get('direction', 1)
            
            if direction == 1 and nearest_resistance:  # Compra
                distance_to_resistance = abs(nearest_resistance['price'] - entry_price)
                if distance_to_resistance < entry_price * 0.005:  # Menos de 0.5%
                    recommendations.append("Entrada próxima à resistência. Considere aguardar rompimento.")
            
            elif direction == -1 and nearest_support:  # Venda
                distance_to_support = abs(entry_price - nearest_support['price'])
                if distance_to_support < entry_price * 0.005:  # Menos de 0.5%
                    recommendations.append("Entrada próxima ao suporte. Considere aguardar rompimento.")
            
            # Verificar confiança
            confidence = analysis.get('confidence', 1.0)
            if confidence < 0.6:
                recommendations.append(f"Confiança baixa ({confidence:.1%}). Considere reduzir exposição.")
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar recomendações: {e}")
            recommendations.append("Erro ao gerar recomendações específicas.")
        
        return recommendations
    
    def _calculate_overall_risk_score(self, analysis: Dict) -> int:
        """Calcula score geral de risco (0-100, onde 100 é menor risco)"""
        try:
            score = 50  # Score base
            
            # Fator 1: Relação risco/recompensa (peso: 25)
            rr_ratio = analysis.get('sl_tp_metadata', {}).get('risk_reward_ratio', 1.0)
            if rr_ratio >= 2.0:
                score += 25
            elif rr_ratio >= 1.5:
                score += 15
            elif rr_ratio >= 1.0:
                score += 5
            else:
                score -= 10
            
            # Fator 2: Percentual de risco (peso: 20)
            risk_pct = analysis.get('position_sizing', {}).get('risk_percentage', 2.0)
            if risk_pct <= 1.0:
                score += 20
            elif risk_pct <= 2.0:
                score += 10
            elif risk_pct <= 3.0:
                score += 0
            else:
                score -= 15
            
            # Fator 3: Confiança (peso: 15)
            confidence = analysis.get('confidence', 1.0)
            score += int(confidence * 15)
            
            # Fator 4: Proteção contra gaps (peso: 10)
            gap_info = analysis.get('gap_protection', {})
            if not gap_info.get('protection_needed', False):
                score += 10
            else:
                score -= 5
            
            # Fator 5: Proximidade a níveis técnicos (peso: 10)
            # Implementação simplificada
            score += 5  # Assumir neutro por enquanto
            
            # Limitar score entre 0 e 100
            score = max(0, min(100, score))
            
            return score
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular score de risco: {e}")
            return 50  # Score neutro em caso de erro
    
    def _is_weekend_risk(self) -> bool:
        """
        Verifica se há risco de fim de semana
        """
        try:
            current_time = datetime.now()
            # Sexta após 20:00 UTC ou fim de semana
            if current_time.weekday() == 4 and current_time.hour >= 20:  # Sexta
                return True
            elif current_time.weekday() in [5, 6]:  # Sábado ou Domingo
                return True
            return False
        except Exception:
            return False