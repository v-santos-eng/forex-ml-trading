#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Gerenciamento de Risco para MT5
Implementa controles automáticos de risco e position sizing
Autor: Sistema Forex ML
Data: 2024
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import json

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Níveis de risco"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class PositionSizeMethod(Enum):
    """Métodos de cálculo de position sizing"""
    FIXED = "fixed"
    PERCENT_BALANCE = "percent_balance"
    PERCENT_EQUITY = "percent_equity"
    KELLY_CRITERION = "kelly_criterion"
    ATR_BASED = "atr_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"

@dataclass
class RiskParameters:
    """Parâmetros de risco"""
    max_risk_per_trade: float = 0.02  # 2% por trade
    max_daily_risk: float = 0.05  # 5% por dia
    max_weekly_risk: float = 0.10  # 10% por semana
    max_monthly_risk: float = 0.20  # 20% por mês
    max_drawdown: float = 0.15  # 15% drawdown máximo
    max_positions: int = 5  # Máximo de posições simultâneas
    max_correlation: float = 0.7  # Correlação máxima entre posições
    min_risk_reward_ratio: float = 1.5  # Ratio mínimo risco/recompensa
    position_size_method: PositionSizeMethod = PositionSizeMethod.PERCENT_BALANCE
    base_position_size: float = 0.01  # Tamanho base da posição
    risk_level: RiskLevel = RiskLevel.MODERATE

@dataclass
class TradeRisk:
    """Análise de risco de um trade"""
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    risk_percentage: float
    approved: bool
    rejection_reason: Optional[str] = None

@dataclass
class PortfolioRisk:
    """Análise de risco do portfólio"""
    total_risk: float
    daily_risk: float
    weekly_risk: float
    monthly_risk: float
    current_drawdown: float
    open_positions: int
    correlation_risk: float
    margin_usage: float
    risk_status: str

class RiskManager:
    """
    Gerenciador de risco principal
    Implementa todos os controles de risco necessários
    """
    
    def __init__(self, mt5_client, risk_params: RiskParameters = None):
        self.mt5_client = mt5_client
        self.risk_params = risk_params or RiskParameters()
        self.trade_history = []
        self.daily_pnl = {}
        self.weekly_pnl = {}
        self.monthly_pnl = {}
        self.correlation_matrix = {}
        self.volatility_data = {}
        
        # Configurações por nível de risco
        self.risk_configs = {
            RiskLevel.CONSERVATIVE: {
                'max_risk_per_trade': 0.01,
                'max_daily_risk': 0.03,
                'max_positions': 3,
                'min_risk_reward_ratio': 2.0
            },
            RiskLevel.MODERATE: {
                'max_risk_per_trade': 0.02,
                'max_daily_risk': 0.05,
                'max_positions': 5,
                'min_risk_reward_ratio': 1.5
            },
            RiskLevel.AGGRESSIVE: {
                'max_risk_per_trade': 0.03,
                'max_daily_risk': 0.08,
                'max_positions': 8,
                'min_risk_reward_ratio': 1.2
            }
        }
        
        self.update_risk_params_by_level()
    
    def update_risk_params_by_level(self):
        """Atualiza parâmetros baseado no nível de risco"""
        config = self.risk_configs.get(self.risk_params.risk_level, {})
        
        for param, value in config.items():
            if hasattr(self.risk_params, param):
                setattr(self.risk_params, param, value)
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss: float, account_balance: float) -> float:
        """
        Calcula tamanho da posição baseado no método configurado
        
        Args:
            symbol: Símbolo do ativo
            entry_price: Preço de entrada
            stop_loss: Preço de stop loss
            account_balance: Saldo da conta
            
        Returns:
            float: Tamanho da posição em lotes
        """
        try:
            # Obter informações do símbolo
            symbol_info = self.mt5_client.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Informações do símbolo {symbol} não disponíveis")
                return self.risk_params.base_position_size
            
            # Calcular risco por pip
            pip_value = symbol_info.get('trade_tick_value', 1.0)
            pip_size = symbol_info.get('point', 0.0001)
            
            # Distância do stop loss em pips
            stop_distance_pips = abs(entry_price - stop_loss) / pip_size
            
            if stop_distance_pips == 0:
                return self.risk_params.base_position_size
            
            # Valor máximo de risco
            max_risk_amount = account_balance * self.risk_params.max_risk_per_trade
            
            # Calcular tamanho baseado no método
            if self.risk_params.position_size_method == PositionSizeMethod.FIXED:
                return self.risk_params.base_position_size
            
            elif self.risk_params.position_size_method == PositionSizeMethod.PERCENT_BALANCE:
                # Baseado na porcentagem do saldo
                risk_per_pip = max_risk_amount / stop_distance_pips
                position_size = risk_per_pip / pip_value
                
            elif self.risk_params.position_size_method == PositionSizeMethod.PERCENT_EQUITY:
                # Baseado na porcentagem do equity
                account_info = self.mt5_client.get_account_info()
                equity = account_info.get('equity', account_balance) if account_info else account_balance
                max_risk_amount = equity * self.risk_params.max_risk_per_trade
                risk_per_pip = max_risk_amount / stop_distance_pips
                position_size = risk_per_pip / pip_value
                
            elif self.risk_params.position_size_method == PositionSizeMethod.KELLY_CRITERION:
                # Critério de Kelly (simplificado)
                win_rate = self.calculate_win_rate(symbol)
                avg_win = self.calculate_average_win(symbol)
                avg_loss = self.calculate_average_loss(symbol)
                
                if avg_loss > 0:
                    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Limitar a 25%
                    position_size = (account_balance * kelly_fraction) / (stop_distance_pips * pip_value)
                else:
                    position_size = self.risk_params.base_position_size
                    
            elif self.risk_params.position_size_method == PositionSizeMethod.ATR_BASED:
                # Baseado no ATR (Average True Range)
                atr = self.calculate_atr(symbol)
                if atr > 0:
                    volatility_multiplier = stop_distance_pips / (atr / pip_size)
                    adjusted_risk = max_risk_amount * (1 / max(volatility_multiplier, 0.5))
                    position_size = adjusted_risk / (stop_distance_pips * pip_value)
                else:
                    position_size = self.risk_params.base_position_size
                    
            elif self.risk_params.position_size_method == PositionSizeMethod.VOLATILITY_ADJUSTED:
                # Ajustado pela volatilidade
                volatility = self.get_symbol_volatility(symbol)
                volatility_factor = 1 / max(volatility, 0.1)  # Inverso da volatilidade
                adjusted_risk = max_risk_amount * volatility_factor
                position_size = adjusted_risk / (stop_distance_pips * pip_value)
                
            else:
                position_size = self.risk_params.base_position_size
            
            # Aplicar limites mínimos e máximos
            min_lot = symbol_info.get('volume_min', 0.01)
            max_lot = symbol_info.get('volume_max', 100.0)
            lot_step = symbol_info.get('volume_step', 0.01)
            
            # Arredondar para o step correto
            position_size = round(position_size / lot_step) * lot_step
            
            # Aplicar limites
            position_size = max(min_lot, min(position_size, max_lot))
            
            return position_size
            
        except Exception as e:
            logger.error(f"Erro ao calcular position size: {e}")
            return self.risk_params.base_position_size
    
    def calculate_optimal_stops(self, symbol: str, entry_price: float, 
                               direction: int, atr_multiplier: float = 2.0) -> Tuple[float, float]:
        """
        Calcula stop loss e take profit ótimos
        
        Args:
            symbol: Símbolo
            entry_price: Preço de entrada
            direction: 1 para compra, -1 para venda
            atr_multiplier: Multiplicador do ATR
            
        Returns:
            Tuple[float, float]: (stop_loss, take_profit)
        """
        try:
            # Obter ATR
            atr = self.calculate_atr(symbol)
            
            # Obter informações do símbolo
            symbol_info = self.mt5_client.get_symbol_info(symbol)
            pip_size = symbol_info.get('point', 0.0001) if symbol_info else 0.0001
            
            if atr == 0:
                # Fallback para valores fixos
                atr = 50 * pip_size  # 50 pips como padrão
            
            # Calcular stop loss baseado no ATR
            stop_distance = atr * atr_multiplier
            
            if direction == 1:  # Compra
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + (stop_distance * self.risk_params.min_risk_reward_ratio)
            else:  # Venda
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - (stop_distance * self.risk_params.min_risk_reward_ratio)
            
            # Arredondar para o número correto de casas decimais
            digits = symbol_info.get('digits', 5) if symbol_info else 5
            stop_loss = round(stop_loss, digits)
            take_profit = round(take_profit, digits)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Erro ao calcular stops: {e}")
            # Retornar valores padrão
            if direction == 1:
                return entry_price * 0.99, entry_price * 1.02
            else:
                return entry_price * 1.01, entry_price * 0.98
    
    def analyze_trade_risk(self, symbol: str, entry_price: float, 
                          stop_loss: float, take_profit: float, 
                          position_size: float) -> TradeRisk:
        """
        Analisa o risco de um trade específico
        
        Args:
            symbol: Símbolo
            entry_price: Preço de entrada
            stop_loss: Stop loss
            take_profit: Take profit
            position_size: Tamanho da posição
            
        Returns:
            TradeRisk: Análise completa do risco
        """
        try:
            # Obter informações da conta
            account_info = self.mt5_client.get_account_info()
            balance = account_info.get('balance', 10000) if account_info else 10000
            
            # Calcular valores de risco e recompensa
            risk_amount = abs(entry_price - stop_loss) * position_size
            reward_amount = abs(take_profit - entry_price) * position_size
            
            # Ajustar pelo valor do pip se necessário
            symbol_info = self.mt5_client.get_symbol_info(symbol)
            if symbol_info:
                contract_size = symbol_info.get('trade_contract_size', 100000)
                risk_amount *= contract_size
                reward_amount *= contract_size
            
            # Calcular ratios
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            risk_percentage = risk_amount / balance if balance > 0 else 0
            
            # Verificar aprovação
            approved = True
            rejection_reason = None
            
            # Verificações de risco
            if risk_percentage > self.risk_params.max_risk_per_trade:
                approved = False
                rejection_reason = f"Risco por trade ({risk_percentage:.2%}) excede máximo ({self.risk_params.max_risk_per_trade:.2%})"
            
            elif risk_reward_ratio < self.risk_params.min_risk_reward_ratio:
                approved = False
                rejection_reason = f"Ratio risco/recompensa ({risk_reward_ratio:.2f}) abaixo do mínimo ({self.risk_params.min_risk_reward_ratio:.2f})"
            
            elif not self.check_daily_risk_limit(risk_amount):
                approved = False
                rejection_reason = "Limite de risco diário excedido"
            
            elif not self.check_position_limits():
                approved = False
                rejection_reason = "Limite de posições simultâneas excedido"
            
            elif not self.check_correlation_risk(symbol):
                approved = False
                rejection_reason = "Risco de correlação muito alto"
            
            return TradeRisk(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_amount=risk_amount,
                reward_amount=reward_amount,
                risk_reward_ratio=risk_reward_ratio,
                risk_percentage=risk_percentage,
                approved=approved,
                rejection_reason=rejection_reason
            )
            
        except Exception as e:
            logger.error(f"Erro na análise de risco: {e}")
            return TradeRisk(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_amount=0,
                reward_amount=0,
                risk_reward_ratio=0,
                risk_percentage=0,
                approved=False,
                rejection_reason=f"Erro na análise: {str(e)}"
            )
    
    def analyze_portfolio_risk(self) -> PortfolioRisk:
        """
        Analisa o risco total do portfólio
        
        Returns:
            PortfolioRisk: Análise completa do portfólio
        """
        try:
            # Obter informações da conta
            account_info = self.mt5_client.get_account_info()
            if not account_info:
                return self.create_empty_portfolio_risk()
            
            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', 0)
            margin = account_info.get('margin', 0)
            
            # Obter posições abertas
            positions = self.mt5_client.get_positions()
            
            # Calcular risco total das posições abertas
            total_risk = 0
            for pos in positions:
                if pos.get('sl', 0) > 0:
                    risk = abs(pos.get('price_open', 0) - pos.get('sl', 0)) * pos.get('volume', 0)
                    symbol_info = self.mt5_client.get_symbol_info(pos.get('symbol', ''))
                    if symbol_info:
                        contract_size = symbol_info.get('trade_contract_size', 100000)
                        risk *= contract_size
                    total_risk += risk
            
            # Calcular drawdown atual
            current_drawdown = max(0, (balance - equity) / balance) if balance > 0 else 0
            
            # Calcular riscos por período
            daily_risk = self.calculate_daily_risk()
            weekly_risk = self.calculate_weekly_risk()
            monthly_risk = self.calculate_monthly_risk()
            
            # Calcular risco de correlação
            correlation_risk = self.calculate_correlation_risk()
            
            # Uso de margem
            margin_usage = margin / equity if equity > 0 else 0
            
            # Determinar status do risco
            risk_status = self.determine_risk_status(
                total_risk / balance if balance > 0 else 0,
                current_drawdown,
                daily_risk,
                len(positions),
                correlation_risk,
                margin_usage
            )
            
            return PortfolioRisk(
                total_risk=total_risk,
                daily_risk=daily_risk,
                weekly_risk=weekly_risk,
                monthly_risk=monthly_risk,
                current_drawdown=current_drawdown,
                open_positions=len(positions),
                correlation_risk=correlation_risk,
                margin_usage=margin_usage,
                risk_status=risk_status
            )
            
        except Exception as e:
            logger.error(f"Erro na análise do portfólio: {e}")
            return self.create_empty_portfolio_risk()
    
    def check_daily_risk_limit(self, additional_risk: float) -> bool:
        """
        Verifica se o risco adicional excede o limite diário
        
        Args:
            additional_risk: Risco adicional do novo trade
            
        Returns:
            bool: True se dentro do limite
        """
        try:
            current_daily_risk = self.calculate_daily_risk()
            account_info = self.mt5_client.get_account_info()
            balance = account_info.get('balance', 10000) if account_info else 10000
            
            total_daily_risk = (current_daily_risk + additional_risk) / balance
            
            return total_daily_risk <= self.risk_params.max_daily_risk
            
        except Exception as e:
            logger.error(f"Erro ao verificar limite diário: {e}")
            return False
    
    def check_position_limits(self) -> bool:
        """
        Verifica se o número de posições está dentro do limite
        
        Returns:
            bool: True se dentro do limite
        """
        try:
            positions = self.mt5_client.get_positions()
            return len(positions) < self.risk_params.max_positions
            
        except Exception as e:
            logger.error(f"Erro ao verificar limite de posições: {e}")
            return False
    
    def check_correlation_risk(self, new_symbol: str) -> bool:
        """
        Verifica se adicionar nova posição excede limite de correlação
        
        Args:
            new_symbol: Símbolo da nova posição
            
        Returns:
            bool: True se correlação aceitável
        """
        try:
            positions = self.mt5_client.get_positions()
            
            for pos in positions:
                existing_symbol = pos.get('symbol', '')
                correlation = self.get_correlation(new_symbol, existing_symbol)
                
                if abs(correlation) > self.risk_params.max_correlation:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao verificar correlação: {e}")
            return True  # Permitir em caso de erro
    
    def calculate_atr(self, symbol: str, period: int = 14) -> float:
        """
        Calcula Average True Range
        
        Args:
            symbol: Símbolo
            period: Período para cálculo
            
        Returns:
            float: Valor do ATR
        """
        try:
            # Obter dados históricos
            rates = self.mt5_client.get_rates(symbol, 16385, period + 10)  # M1
            
            if rates is None or len(rates) < period:
                return 0.0
            
            # Calcular True Range
            high = rates['high'].values
            low = rates['low'].values
            close = rates['close'].values
            
            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))
                )
            )[1:]  # Remove primeiro elemento (NaN)
            
            # Calcular ATR como média móvel do TR
            atr = np.mean(tr[-period:]) if len(tr) >= period else 0.0
            
            return atr
            
        except Exception as e:
            logger.error(f"Erro ao calcular ATR para {symbol}: {e}")
            return 0.0
    
    def get_symbol_volatility(self, symbol: str, period: int = 20) -> float:
        """
        Calcula volatilidade do símbolo
        
        Args:
            symbol: Símbolo
            period: Período para cálculo
            
        Returns:
            float: Volatilidade (desvio padrão dos retornos)
        """
        try:
            # Verificar cache
            if symbol in self.volatility_data:
                cache_time, volatility = self.volatility_data[symbol]
                if (datetime.now() - cache_time).seconds < 3600:  # Cache por 1 hora
                    return volatility
            
            # Obter dados históricos
            rates = self.mt5_client.get_rates(symbol, 16385, period + 1)  # M1
            
            if rates is None or len(rates) < period:
                return 0.1  # Volatilidade padrão
            
            # Calcular retornos
            closes = rates['close'].values
            returns = np.diff(np.log(closes))
            
            # Calcular volatilidade
            volatility = np.std(returns) if len(returns) > 0 else 0.1
            
            # Atualizar cache
            self.volatility_data[symbol] = (datetime.now(), volatility)
            
            return volatility
            
        except Exception as e:
            logger.error(f"Erro ao calcular volatilidade para {symbol}: {e}")
            return 0.1
    
    def get_correlation(self, symbol1: str, symbol2: str, period: int = 50) -> float:
        """
        Calcula correlação entre dois símbolos
        
        Args:
            symbol1: Primeiro símbolo
            symbol2: Segundo símbolo
            period: Período para cálculo
            
        Returns:
            float: Correlação (-1 a 1)
        """
        try:
            # Verificar cache
            cache_key = f"{symbol1}_{symbol2}"
            if cache_key in self.correlation_matrix:
                cache_time, correlation = self.correlation_matrix[cache_key]
                if (datetime.now() - cache_time).seconds < 3600:  # Cache por 1 hora
                    return correlation
            
            # Obter dados históricos
            rates1 = self.mt5_client.get_rates(symbol1, 16385, period)
            rates2 = self.mt5_client.get_rates(symbol2, 16385, period)
            
            if rates1 is None or rates2 is None or len(rates1) < period or len(rates2) < period:
                return 0.0
            
            # Calcular retornos
            returns1 = np.diff(np.log(rates1['close'].values))
            returns2 = np.diff(np.log(rates2['close'].values))
            
            # Garantir mesmo tamanho
            min_len = min(len(returns1), len(returns2))
            returns1 = returns1[-min_len:]
            returns2 = returns2[-min_len:]
            
            # Calcular correlação
            correlation = np.corrcoef(returns1, returns2)[0, 1] if min_len > 1 else 0.0
            
            # Tratar NaN
            if np.isnan(correlation):
                correlation = 0.0
            
            # Atualizar cache
            self.correlation_matrix[cache_key] = (datetime.now(), correlation)
            
            return correlation
            
        except Exception as e:
            logger.error(f"Erro ao calcular correlação entre {symbol1} e {symbol2}: {e}")
            return 0.0
    
    def calculate_win_rate(self, symbol: str = None) -> float:
        """
        Calcula taxa de acerto
        
        Args:
            symbol: Símbolo específico (opcional)
            
        Returns:
            float: Taxa de acerto (0-1)
        """
        try:
            # Obter histórico de negociações
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Últimos 30 dias
            
            deals = self.mt5_client.get_history_deals(start_date, end_date)
            
            if not deals:
                return 0.5  # Taxa neutra se não há histórico
            
            # Filtrar por símbolo se especificado
            if symbol:
                deals = [deal for deal in deals if deal.get('symbol') == symbol]
            
            # Contar trades vencedores
            winning_trades = sum(1 for deal in deals if deal.get('profit', 0) > 0)
            total_trades = len(deals)
            
            return winning_trades / total_trades if total_trades > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Erro ao calcular win rate: {e}")
            return 0.5
    
    def calculate_average_win(self, symbol: str = None) -> float:
        """
        Calcula ganho médio
        
        Args:
            symbol: Símbolo específico (opcional)
            
        Returns:
            float: Ganho médio
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            deals = self.mt5_client.get_history_deals(start_date, end_date)
            
            if not deals:
                return 100.0  # Valor padrão
            
            if symbol:
                deals = [deal for deal in deals if deal.get('symbol') == symbol]
            
            winning_deals = [deal.get('profit', 0) for deal in deals if deal.get('profit', 0) > 0]
            
            return np.mean(winning_deals) if winning_deals else 100.0
            
        except Exception as e:
            logger.error(f"Erro ao calcular ganho médio: {e}")
            return 100.0
    
    def calculate_average_loss(self, symbol: str = None) -> float:
        """
        Calcula perda média
        
        Args:
            symbol: Símbolo específico (opcional)
            
        Returns:
            float: Perda média (valor absoluto)
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            deals = self.mt5_client.get_history_deals(start_date, end_date)
            
            if not deals:
                return 100.0  # Valor padrão
            
            if symbol:
                deals = [deal for deal in deals if deal.get('symbol') == symbol]
            
            losing_deals = [abs(deal.get('profit', 0)) for deal in deals if deal.get('profit', 0) < 0]
            
            return np.mean(losing_deals) if losing_deals else 100.0
            
        except Exception as e:
            logger.error(f"Erro ao calcular perda média: {e}")
            return 100.0
    
    def calculate_daily_risk(self) -> float:
        """
        Calcula risco acumulado do dia
        
        Returns:
            float: Risco diário em valor absoluto
        """
        try:
            today = datetime.now().date()
            
            # Obter negociações do dia
            start_date = datetime.combine(today, datetime.min.time())
            end_date = datetime.now()
            
            deals = self.mt5_client.get_history_deals(start_date, end_date)
            
            # Somar perdas do dia
            daily_loss = sum(abs(deal.get('profit', 0)) for deal in deals if deal.get('profit', 0) < 0)
            
            return daily_loss
            
        except Exception as e:
            logger.error(f"Erro ao calcular risco diário: {e}")
            return 0.0
    
    def calculate_weekly_risk(self) -> float:
        """
        Calcula risco acumulado da semana
        
        Returns:
            float: Risco semanal em valor absoluto
        """
        try:
            today = datetime.now()
            week_start = today - timedelta(days=today.weekday())
            
            deals = self.mt5_client.get_history_deals(week_start, today)
            
            weekly_loss = sum(abs(deal.get('profit', 0)) for deal in deals if deal.get('profit', 0) < 0)
            
            return weekly_loss
            
        except Exception as e:
            logger.error(f"Erro ao calcular risco semanal: {e}")
            return 0.0
    
    def calculate_monthly_risk(self) -> float:
        """
        Calcula risco acumulado do mês
        
        Returns:
            float: Risco mensal em valor absoluto
        """
        try:
            today = datetime.now()
            month_start = today.replace(day=1)
            
            deals = self.mt5_client.get_history_deals(month_start, today)
            
            monthly_loss = sum(abs(deal.get('profit', 0)) for deal in deals if deal.get('profit', 0) < 0)
            
            return monthly_loss
            
        except Exception as e:
            logger.error(f"Erro ao calcular risco mensal: {e}")
            return 0.0
    
    def calculate_correlation_risk(self) -> float:
        """
        Calcula risco de correlação do portfólio
        
        Returns:
            float: Risco de correlação (0-1)
        """
        try:
            positions = self.mt5_client.get_positions()
            
            if len(positions) < 2:
                return 0.0
            
            symbols = [pos.get('symbol', '') for pos in positions]
            correlations = []
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    correlation = abs(self.get_correlation(symbols[i], symbols[j]))
                    correlations.append(correlation)
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.error(f"Erro ao calcular risco de correlação: {e}")
            return 0.0
    
    def determine_risk_status(self, total_risk_pct: float, drawdown: float, 
                            daily_risk: float, positions: int, 
                            correlation: float, margin_usage: float) -> str:
        """
        Determina status geral do risco
        
        Args:
            total_risk_pct: Risco total como porcentagem
            drawdown: Drawdown atual
            daily_risk: Risco diário
            positions: Número de posições
            correlation: Risco de correlação
            margin_usage: Uso de margem
            
        Returns:
            str: Status do risco (LOW, MEDIUM, HIGH, CRITICAL)
        """
        risk_score = 0
        
        # Avaliar cada componente
        if total_risk_pct > self.risk_params.max_risk_per_trade * 3:
            risk_score += 2
        elif total_risk_pct > self.risk_params.max_risk_per_trade * 2:
            risk_score += 1
        
        if drawdown > self.risk_params.max_drawdown * 0.8:
            risk_score += 2
        elif drawdown > self.risk_params.max_drawdown * 0.5:
            risk_score += 1
        
        if positions >= self.risk_params.max_positions:
            risk_score += 1
        
        if correlation > self.risk_params.max_correlation:
            risk_score += 1
        
        if margin_usage > 0.8:
            risk_score += 2
        elif margin_usage > 0.5:
            risk_score += 1
        
        # Determinar status
        if risk_score >= 6:
            return "CRITICAL"
        elif risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def create_empty_portfolio_risk(self) -> PortfolioRisk:
        """Cria análise de risco vazia"""
        return PortfolioRisk(
            total_risk=0.0,
            daily_risk=0.0,
            weekly_risk=0.0,
            monthly_risk=0.0,
            current_drawdown=0.0,
            open_positions=0,
            correlation_risk=0.0,
            margin_usage=0.0,
            risk_status="UNKNOWN"
        )
    
    def update_risk_parameters(self, new_params: RiskParameters):
        """Atualiza parâmetros de risco"""
        self.risk_params = new_params
        self.update_risk_params_by_level()
        logger.info("Parâmetros de risco atualizados")
    
    def get_risk_summary(self) -> Dict:
        """Retorna resumo completo do risco"""
        try:
            portfolio_risk = self.analyze_portfolio_risk()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'risk_level': self.risk_params.risk_level.value,
                'portfolio_risk': portfolio_risk.__dict__,
                'risk_parameters': self.risk_params.__dict__,
                'limits': {
                    'max_risk_per_trade': f"{self.risk_params.max_risk_per_trade:.2%}",
                    'max_daily_risk': f"{self.risk_params.max_daily_risk:.2%}",
                    'max_positions': self.risk_params.max_positions,
                    'max_drawdown': f"{self.risk_params.max_drawdown:.2%}"
                }
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar resumo de risco: {e}")
            return {'error': str(e)}

# Exemplo de uso
if __name__ == "__main__":
    # Este seria usado com um cliente MT5 real
    print("Risk Manager - Sistema de Gerenciamento de Risco para MT5")
    print("Funcionalidades implementadas:")
    print("- Position sizing automático")
    print("- Cálculo de stop loss e take profit ótimos")
    print("- Análise de risco por trade")
    print("- Monitoramento de risco do portfólio")
    print("- Controles de correlação")
    print("- Limites de drawdown")
    print("- Diferentes níveis de risco (Conservative, Moderate, Aggressive)")