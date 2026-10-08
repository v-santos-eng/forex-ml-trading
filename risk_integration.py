#!/usr/bin/env python3
"""
Integração do Sistema Avançado de Gestão de Risco

Este módulo integra o advanced_risk_system.py com o sistema de trading existente,
fornecendo uma interface unificada para:
- Substituir cálculos básicos de SL/TP por versões avançadas baseadas em ATR
- Integrar Kelly Criterion otimizado no dimensionamento de posições
- Adicionar trailing stops dinâmicos
- Implementar proteção contra gaps
- Utilizar análise de níveis técnicos
- Aplicar Take Profit escalonado quando apropriado

Autor: Sistema de Trading ML
Data: 2024
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import json

from advanced_risk_system import (
    AdvancedRiskSystem,
    ATRParameters,
    VolatilityParameters,
    TrailingStopParameters,
    GapProtectionParameters,
    TechnicalLevelsParameters,
    ScaledTakeProfitParameters
)
from risk_manager import RiskManager, RiskParameters
from config import ForexConfig as Config

class RiskIntegrationManager:
    """
    Gerenciador de integração entre o sistema avançado de risco
    e o sistema de trading existente.
    """
    
    def __init__(self, mt5_client, logger: Optional[logging.Logger] = None):
        """
        Inicializa o gerenciador de integração.
        
        Args:
            mt5_client: Cliente MT5 para acesso aos dados de mercado
            logger: Logger para registro de atividades
        """
        self.mt5_client = mt5_client
        self.logger = logger or logging.getLogger(__name__)
        
        # Inicializar sistemas
        self.advanced_risk = AdvancedRiskSystem(mt5_client=mt5_client, logger=logger)
        self.legacy_risk = RiskManager(mt5_client)
        
        # Configurações de integração
        self.config = {
            'use_advanced_sl_tp': True,
            'use_kelly_optimization': True,
            'enable_trailing_stops': True,
            'enable_gap_protection': True,
            'use_technical_levels': True,
            'enable_scaled_tp': False,  # Desabilitado por padrão
            'fallback_to_legacy': True,
            'min_confidence_threshold': 0.6,
            'max_risk_per_trade': 2.0,
            'emergency_risk_limit': 5.0
        }
        
        # Cache para análises recentes
        self._analysis_cache = {}
        self._cache_timeout = 300  # 5 minutos
        
        self.logger.info("RiskIntegrationManager inicializado com sucesso")
    
    def calculate_enhanced_stops(self, symbol: str, entry_price: float, 
                               direction: int, confidence: float = 0.8,
                               use_advanced: bool = None) -> Tuple[float, float, Dict[str, Any]]:
        """
        Calcula Stop Loss e Take Profit usando o sistema avançado ou legacy.
        
        Args:
            symbol: Símbolo do ativo
            entry_price: Preço de entrada
            direction: Direção da operação (1 para compra, -1 para venda)
            confidence: Nível de confiança na operação (0.0 a 1.0)
            use_advanced: Forçar uso do sistema avançado (None = automático)
        
        Returns:
            Tuple[float, float, Dict]: (stop_loss, take_profit, metadata)
        """
        try:
            # Determinar qual sistema usar
            should_use_advanced = (
                use_advanced if use_advanced is not None 
                else (self.config['use_advanced_sl_tp'] and confidence >= self.config['min_confidence_threshold'])
            )
            
            if should_use_advanced:
                self.logger.info(f"Calculando SL/TP avançado para {symbol}")
                
                # Usar sistema avançado
                sl, tp, metadata = self.advanced_risk.calculate_optimal_sl_tp(
                    symbol, entry_price, direction, confidence
                )
                
                metadata['system_used'] = 'advanced'
                metadata['confidence'] = confidence
                
                # Validar resultados
                if self._validate_stops(entry_price, sl, tp, direction):
                    self.logger.info(
                        f"SL/TP avançado calculado: SL={sl:.5f}, TP={tp:.5f}, "
                        f"RR={metadata.get('risk_reward_ratio', 'N/A'):.2f}"
                    )
                    return sl, tp, metadata
                else:
                    self.logger.warning("SL/TP avançado inválido, usando sistema legacy")
                    should_use_advanced = False
            
            if not should_use_advanced:
                self.logger.info(f"Calculando SL/TP legacy para {symbol}")
                
                # Usar sistema legacy
                risk_params = RiskParameters()
                sl, tp = self.legacy_risk.calculate_optimal_stops(
                    symbol, entry_price, direction, risk_params
                )
                
                metadata = {
                    'system_used': 'legacy',
                    'confidence': confidence,
                    'atr_value': 0.0,
                    'risk_reward_ratio': abs((tp - entry_price) / (entry_price - sl)) if direction == 1 else abs((entry_price - tp) / (sl - entry_price)),
                    'volatility_adjustment': 1.0
                }
                
                self.logger.info(f"SL/TP legacy calculado: SL={sl:.5f}, TP={tp:.5f}")
                return sl, tp, metadata
                
        except Exception as e:
            self.logger.error(f"Erro no cálculo de SL/TP para {symbol}: {e}")
            
            # Fallback para valores padrão
            if self.config['fallback_to_legacy']:
                return self._get_default_stops(symbol, entry_price, direction)
            else:
                raise
    
    def calculate_optimized_position_size(self, symbol: str, entry_price: float,
                                        stop_loss: float, take_profit: float,
                                        direction: int, confidence: float = 0.8) -> Dict[str, Any]:
        """
        Calcula o tamanho da posição otimizado usando Kelly Criterion.
        
        Args:
            symbol: Símbolo do ativo
            entry_price: Preço de entrada
            stop_loss: Nível de stop loss
            take_profit: Nível de take profit
            direction: Direção da operação
            confidence: Nível de confiança
        
        Returns:
            Dict com informações de dimensionamento
        """
        try:
            if self.config['use_kelly_optimization'] and confidence >= self.config['min_confidence_threshold']:
                self.logger.info(f"Calculando posição otimizada com Kelly para {symbol}")
                
                position_info = self.advanced_risk.calculate_kelly_optimized_position(
                    symbol, entry_price, stop_loss, take_profit, direction, confidence
                )
                
                # Aplicar limites de segurança
                max_risk = min(self.config['max_risk_per_trade'], self.config['emergency_risk_limit'])
                if position_info['risk_percentage'] > max_risk:
                    self.logger.warning(
                        f"Risco calculado ({position_info['risk_percentage']:.2f}%) "
                        f"excede limite ({max_risk:.2f}%), ajustando..."
                    )
                    
                    # Ajustar tamanho da posição
                    adjustment_factor = max_risk / position_info['risk_percentage']
                    position_info['position_size'] *= adjustment_factor
                    position_info['risk_percentage'] = max_risk
                    position_info['adjusted'] = True
                
                position_info['method'] = 'kelly_optimized'
                return position_info
            
            else:
                self.logger.info(f"Calculando posição padrão para {symbol}")
                
                # Usar cálculo padrão
                account_balance = getattr(Config, 'ACCOUNT_BALANCE', 10000.0)
                risk_amount = account_balance * (self.config['max_risk_per_trade'] / 100)
                pip_value = self._calculate_pip_value(symbol)
                stop_distance_pips = abs(entry_price - stop_loss) / pip_value
                
                if stop_distance_pips > 0:
                    position_size = risk_amount / (stop_distance_pips * pip_value)
                else:
                    position_size = 0.01  # Tamanho mínimo
                
                return {
                    'position_size': round(position_size, 2),
                    'risk_percentage': self.config['max_risk_per_trade'],
                    'kelly_fraction': 0.0,
                    'method': 'standard',
                    'adjusted': False
                }
                
        except Exception as e:
            self.logger.error(f"Erro no cálculo de posição para {symbol}: {e}")
            
            # Retornar posição mínima segura
            return {
                'position_size': 0.01,
                'risk_percentage': 1.0,
                'kelly_fraction': 0.0,
                'method': 'fallback',
                'adjusted': True,
                'error': str(e)
            }
    
    def setup_trailing_stop(self, symbol: str, entry_price: float, 
                          initial_sl: float, direction: int) -> Optional[Dict[str, Any]]:
        """
        Configura trailing stop dinâmico para uma posição.
        
        Args:
            symbol: Símbolo do ativo
            entry_price: Preço de entrada
            initial_sl: Stop loss inicial
            direction: Direção da operação
        
        Returns:
            Dict com configurações do trailing stop ou None se desabilitado
        """
        try:
            if not self.config['enable_trailing_stops']:
                return None
            
            self.logger.info(f"Configurando trailing stop para {symbol}")
            
            trailing_config = self.advanced_risk.initialize_trailing_stop(
                symbol, entry_price, initial_sl, direction
            )
            
            trailing_config['symbol'] = symbol
            trailing_config['direction'] = direction
            trailing_config['created_at'] = datetime.now().isoformat()
            
            self.logger.info(
                f"Trailing stop configurado: distância={trailing_config['trailing_distance']:.5f}, "
                f"ativação={trailing_config['activation_level']:.5f}"
            )
            
            return trailing_config
            
        except Exception as e:
            self.logger.error(f"Erro ao configurar trailing stop para {symbol}: {e}")
            return None
    
    def update_trailing_stop(self, symbol: str, current_price: float,
                           trailing_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Atualiza trailing stop baseado no preço atual.
        
        Args:
            symbol: Símbolo do ativo
            current_price: Preço atual
            trailing_config: Configuração atual do trailing stop
        
        Returns:
            Dict com atualização do trailing stop ou None se não houver mudança
        """
        try:
            if not self.config['enable_trailing_stops'] or not trailing_config:
                return None
            
            direction = trailing_config.get('direction', 1)
            
            update_info = self.advanced_risk.update_trailing_stop(
                symbol, current_price, trailing_config, direction
            )
            
            if update_info.get('updated', False):
                self.logger.info(
                    f"Trailing stop atualizado para {symbol}: "
                    f"novo SL={update_info['new_sl']:.5f}"
                )
                
                # Atualizar configuração
                trailing_config.update(update_info)
                trailing_config['last_update'] = datetime.now().isoformat()
                
                return update_info
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar trailing stop para {symbol}: {e}")
            return None
    
    def check_gap_risk(self, symbol: str, entry_price: float, 
                      current_price: float, position_size: float) -> Dict[str, Any]:
        """
        Verifica risco de gap e sugere ações de proteção.
        
        Args:
            symbol: Símbolo do ativo
            entry_price: Preço de entrada
            current_price: Preço atual
            position_size: Tamanho da posição
        
        Returns:
            Dict com análise de risco de gap
        """
        try:
            if not self.config['enable_gap_protection']:
                return {'protection_needed': False, 'risk_level': 'low'}
            
            gap_analysis = self.advanced_risk.check_gap_protection(
                symbol, entry_price, current_price, position_size
            )
            
            if gap_analysis.get('protection_needed', False):
                self.logger.warning(
                    f"Risco de gap detectado para {symbol}: "
                    f"nível={gap_analysis['risk_level']}"
                )
                
                # Log das recomendações
                recommendations = gap_analysis.get('recommendations', [])
                for rec in recommendations:
                    self.logger.info(f"Recomendação: {rec}")
            
            return gap_analysis
            
        except Exception as e:
            self.logger.error(f"Erro na verificação de gap para {symbol}: {e}")
            return {
                'protection_needed': False,
                'risk_level': 'unknown',
                'error': str(e)
            }
    
    def get_comprehensive_analysis(self, symbol: str, entry_price: float,
                                 direction: int, confidence: float = 0.8) -> Dict[str, Any]:
        """
        Obtém análise completa de risco para uma operação.
        
        Args:
            symbol: Símbolo do ativo
            entry_price: Preço de entrada
            direction: Direção da operação
            confidence: Nível de confiança
        
        Returns:
            Dict com análise completa
        """
        try:
            # Verificar cache
            cache_key = f"{symbol}_{entry_price}_{direction}_{confidence}"
            if cache_key in self._analysis_cache:
                cached_analysis, timestamp = self._analysis_cache[cache_key]
                if (datetime.now() - timestamp).seconds < self._cache_timeout:
                    self.logger.debug(f"Usando análise em cache para {symbol}")
                    return cached_analysis
            
            self.logger.info(f"Executando análise completa para {symbol}")
            
            # Obter análise completa do sistema avançado
            analysis = self.advanced_risk.get_comprehensive_risk_analysis(
                symbol, entry_price, direction, confidence
            )
            
            # Adicionar informações de integração
            analysis['integration_config'] = {
                'advanced_enabled': self.config['use_advanced_sl_tp'],
                'kelly_enabled': self.config['use_kelly_optimization'],
                'trailing_enabled': self.config['enable_trailing_stops'],
                'gap_protection_enabled': self.config['enable_gap_protection'],
                'technical_levels_enabled': self.config['use_technical_levels']
            }
            
            analysis['timestamp'] = datetime.now().isoformat()
            
            # Armazenar no cache
            self._analysis_cache[cache_key] = (analysis, datetime.now())
            
            # Limpar cache antigo
            self._cleanup_cache()
            
            self.logger.info(
                f"Análise completa concluída para {symbol}: "
                f"score={analysis.get('risk_score', 'N/A')}"
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Erro na análise completa para {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'risk_score': 100  # Máximo risco em caso de erro
            }
    
    def _validate_stops(self, entry_price: float, stop_loss: float, 
                       take_profit: float, direction: int) -> bool:
        """
        Valida se os níveis de SL/TP estão corretos.
        
        Args:
            entry_price: Preço de entrada
            stop_loss: Nível de stop loss
            take_profit: Nível de take profit
            direction: Direção da operação
        
        Returns:
            bool: True se válidos, False caso contrário
        """
        try:
            if direction == 1:  # Compra
                return stop_loss < entry_price < take_profit
            else:  # Venda
                return take_profit < entry_price < stop_loss
        except:
            return False
    
    def _get_default_stops(self, symbol: str, entry_price: float, 
                          direction: int) -> Tuple[float, float, Dict[str, Any]]:
        """
        Retorna stops padrão em caso de falha.
        
        Args:
            symbol: Símbolo do ativo
            entry_price: Preço de entrada
            direction: Direção da operação
        
        Returns:
            Tuple com SL, TP e metadata padrão
        """
        pip_value = self._calculate_pip_value(symbol)
        
        if direction == 1:  # Compra
            sl = entry_price - (50 * pip_value)  # 50 pips
            tp = entry_price + (100 * pip_value)  # 100 pips
        else:  # Venda
            sl = entry_price + (50 * pip_value)
            tp = entry_price - (100 * pip_value)
        
        metadata = {
            'system_used': 'fallback',
            'atr_value': 0.0,
            'risk_reward_ratio': 2.0,
            'volatility_adjustment': 1.0
        }
        
        return sl, tp, metadata
    
    def _calculate_pip_value(self, symbol: str) -> float:
        """
        Calcula o valor do pip para um símbolo.
        
        Args:
            symbol: Símbolo do ativo
        
        Returns:
            float: Valor do pip
        """
        try:
            symbol_info = self.mt5_client.get_symbol_info(symbol)
            if symbol_info:
                return symbol_info.get('point', 0.0001)
            return 0.0001  # Valor padrão
        except:
            return 0.0001
    
    def _cleanup_cache(self):
        """
        Remove entradas antigas do cache.
        """
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, (_, timestamp) in self._analysis_cache.items():
            if (current_time - timestamp).seconds > self._cache_timeout:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._analysis_cache[key]
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Atualiza configurações do sistema.
        
        Args:
            new_config: Novas configurações
        """
        self.config.update(new_config)
        self.logger.info(f"Configurações atualizadas: {new_config}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Retorna status atual do sistema de integração.
        
        Returns:
            Dict com informações de status
        """
        return {
            'advanced_risk_enabled': self.config['use_advanced_sl_tp'],
            'kelly_optimization_enabled': self.config['use_kelly_optimization'],
            'trailing_stops_enabled': self.config['enable_trailing_stops'],
            'gap_protection_enabled': self.config['enable_gap_protection'],
            'technical_levels_enabled': self.config['use_technical_levels'],
            'cache_entries': len(self._analysis_cache),
            'max_risk_per_trade': self.config['max_risk_per_trade'],
            'min_confidence_threshold': self.config['min_confidence_threshold'],
            'timestamp': datetime.now().isoformat()
        }

# Função de conveniência para criar instância global
_risk_integration_instance = None

def get_risk_integration_manager(mt5_client, logger=None) -> RiskIntegrationManager:
    """
    Retorna instância singleton do RiskIntegrationManager.
    
    Args:
        mt5_client: Cliente MT5
        logger: Logger opcional
    
    Returns:
        RiskIntegrationManager: Instância do gerenciador
    """
    global _risk_integration_instance
    
    if _risk_integration_instance is None:
        _risk_integration_instance = RiskIntegrationManager(mt5_client, logger)
    
    return _risk_integration_instance