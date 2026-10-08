#!/usr/bin/env python3
"""
Exemplo Prático de Uso do Sistema Avançado de Gestão de Risco

Este arquivo demonstra como integrar e usar todas as funcionalidades
do sistema avançado de risco em um ambiente de trading real.

Autor: Sistema de Trading ML
Data: 2024
"""

import sys
import os
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_risk_system.log'),
        logging.StreamHandler()
    ]
)

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from advanced_risk_system import (
        AdvancedRiskSystem,
        ATRParameters, VolatilityParameters, TrailingStopParameters,
        GapProtectionParameters, TechnicalLevelsParameters, ScaledTakeProfitParameters,
        ATRMethod, TrailingStopType
    )
    from risk_integration import RiskIntegrationManager
    from config import ForexConfig as Config
    from risk_manager import RiskParameters
except ImportError as e:
    print(f"Erro de importação: {e}")
    print("Certifique-se de que todos os módulos estão disponíveis")
    sys.exit(1)

class AdvancedRiskExample:
    """
    Classe de exemplo para demonstrar o uso do sistema avançado de risco
    """
    
    def __init__(self, mt5_client=None):
        """
        Inicializar exemplo com cliente MT5 (pode ser mock para testes)
        """
        self.mt5_client = mt5_client
        self.logger = logging.getLogger(__name__)
        
        # Configurar parâmetros personalizados
        self.setup_parameters()
        
        # Inicializar sistemas
        self.setup_systems()
    
    def setup_parameters(self):
        """
        Configurar parâmetros personalizados para diferentes cenários
        """
        self.logger.info("Configurando parâmetros do sistema avançado de risco")
        
        # Parâmetros ATR - Configuração conservadora
        self.atr_params_conservative = ATRParameters(
            period=21,  # Período maior para suavizar
            method=ATRMethod.WILDER,
            multiplier_sl=2.5,  # SL mais amplo
            multiplier_tp=2.0,  # TP mais conservador
            volatility_adjustment=True,
            session_adjustment=True
        )
        
        # Parâmetros ATR - Configuração agressiva
        self.atr_params_aggressive = ATRParameters(
            period=10,  # Período menor para reação rápida
            method=ATRMethod.EXPONENTIAL,
            multiplier_sl=1.5,  # SL mais apertado
            multiplier_tp=4.0,  # TP mais ambicioso
            volatility_adjustment=True,
            session_adjustment=True
        )
        
        # Parâmetros de volatilidade
        self.volatility_params = VolatilityParameters(
            lookback_period=30,
            regime_thresholds={
                'low': 0.4,
                'normal': 1.0,
                'high': 1.6,
                'extreme': 2.2
            },
            adjustment_factors={
                'low': 0.7,
                'normal': 1.0,
                'high': 1.3,
                'extreme': 1.6
            }
        )
        
        # Parâmetros de trailing stop
        self.trailing_params = TrailingStopParameters(
            type=TrailingStopType.ATR_BASED,
            activation_distance=2.0,  # Ativar após 2x ATR de lucro
            trail_distance=1.5,       # Trailing a 1.5x ATR
            step_size=0.3,            # Passos de 0.3x ATR
            use_breakeven=True,
            breakeven_trigger=1.0     # Breakeven após 1x ATR
        )
        
        # Parâmetros de proteção contra gaps
        self.gap_params = GapProtectionParameters(
            enabled=True,
            max_gap_threshold=0.003,  # 0.3% gap máximo
            pre_news_buffer=45,       # 45 min antes de notícias
            post_news_buffer=20,      # 20 min após notícias
            weekend_protection=True,
            correlation_threshold=0.75,
            max_exposure_reduction=0.6
        )
        
        # Parâmetros de níveis técnicos
        self.tech_params = TechnicalLevelsParameters(
            support_resistance_period=100,
            fibonacci_levels=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
            psychological_levels=True,
            pivot_points=True,
            volume_profile=True,
            level_strength_threshold=4,
            proximity_threshold=0.0008
        )
        
        # Parâmetros de TP escalonado
        self.tp_params = ScaledTakeProfitParameters(
            enabled=True,
            levels=[1.0, 2.5, 4.0],           # Níveis em múltiplos ATR
            position_percentages=[0.4, 0.35, 0.25],  # 40%, 35%, 25%
            trail_after_first=True,
            dynamic_adjustment=True
        )
    
    def setup_systems(self):
        """
        Inicializar os sistemas de risco
        """
        self.logger.info("Inicializando sistemas de risco")
        
        # Sistema conservador
        self.conservative_system = AdvancedRiskSystem(
            mt5_client=self.mt5_client,
            atr_params=self.atr_params_conservative,
            volatility_params=self.volatility_params,
            trailing_params=self.trailing_params,
            gap_protection_params=self.gap_params,
            technical_params=self.tech_params,
            scaled_tp_params=self.tp_params
        )
        
        # Sistema agressivo
        self.aggressive_system = AdvancedRiskSystem(
            mt5_client=self.mt5_client,
            atr_params=self.atr_params_aggressive,
            volatility_params=self.volatility_params,
            trailing_params=self.trailing_params,
            gap_protection_params=self.gap_params,
            technical_params=self.tech_params,
            scaled_tp_params=self.tp_params
        )
        
        # Gerenciador de integração
        self.integration_manager = RiskIntegrationManager(
            mt5_client=self.mt5_client,
            logger=self.logger
        )
        
        # Configurar funcionalidades avançadas
        self.integration_manager.enable_advanced_sl_tp = True
        self.integration_manager.enable_kelly_optimization = True
        self.integration_manager.enable_trailing_stops = True
        self.integration_manager.enable_gap_protection = True
        self.integration_manager.enable_technical_analysis = True
        self.integration_manager.enable_scaled_tp = True
    
    def example_basic_sl_tp_calculation(self):
        """
        Exemplo básico de cálculo de SL/TP
        """
        print("\n" + "="*60)
        print("EXEMPLO 1: Cálculo Básico de SL/TP")
        print("="*60)
        
        symbol = "EURUSD"
        entry_price = 1.1000
        direction = 1  # Compra
        confidence = 0.75
        
        self.logger.info(f"Calculando SL/TP para {symbol} - Entry: {entry_price}")
        
        # Usar sistema conservador
        try:
            result = self.conservative_system.calculate_optimal_sl_tp(
                symbol=symbol,
                entry_price=entry_price,
                direction=direction,
                confidence=confidence
            )
            
            print(f"Símbolo: {symbol}")
            print(f"Preço de entrada: {entry_price:.5f}")
            print(f"Direção: {'COMPRA' if direction == 1 else 'VENDA'}")
            print(f"Confiança: {confidence:.1%}")
            print(f"\nResultados (Sistema Conservador):")
            print(f"Stop Loss: {result['sl']:.5f}")
            print(f"Take Profit: {result['tp']:.5f}")
            print(f"ATR usado: {result.get('atr_value', 'N/A')}")
            print(f"Risk/Reward: {result.get('risk_reward_ratio', 'N/A')}")
            
            if 'metadata' in result:
                metadata = result['metadata']
                print(f"\nMetadados:")
                print(f"Regime de volatilidade: {metadata.get('volatility_regime', 'N/A')}")
                print(f"Ajuste de volatilidade: {metadata.get('volatility_factor', 'N/A')}")
                print(f"Ajuste de sessão: {metadata.get('session_factor', 'N/A')}")
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo SL/TP: {e}")
            print(f"Erro: {e}")
    
    def example_kelly_optimization(self):
        """
        Exemplo de otimização com Kelly Criterion
        """
        print("\n" + "="*60)
        print("EXEMPLO 2: Otimização com Kelly Criterion")
        print("="*60)
        
        symbol = "GBPUSD"
        entry_price = 1.2500
        direction = -1  # Venda
        confidence = 0.85
        account_balance = 10000.0
        
        self.logger.info(f"Otimizando posição com Kelly para {symbol}")
        
        try:
            result = self.aggressive_system.calculate_kelly_optimized_position(
                symbol=symbol,
                entry_price=entry_price,
                direction=direction,
                confidence=confidence,
                account_balance=account_balance
            )
            
            print(f"Símbolo: {symbol}")
            print(f"Preço de entrada: {entry_price:.5f}")
            print(f"Direção: {'COMPRA' if direction == 1 else 'VENDA'}")
            print(f"Saldo da conta: ${account_balance:,.2f}")
            print(f"Confiança: {confidence:.1%}")
            print(f"\nResultados da Otimização Kelly:")
            print(f"Tamanho da posição: {result['position_size']:.4f} lotes")
            print(f"Valor em risco: ${result['risk_amount']:.2f}")
            print(f"Percentual de risco: {result['risk_percentage']:.2%}")
            print(f"Fração Kelly: {result.get('kelly_fraction', 'N/A')}")
            
        except Exception as e:
            self.logger.error(f"Erro na otimização Kelly: {e}")
            print(f"Erro: {e}")
    
    def example_trailing_stop_simulation(self):
        """
        Exemplo de simulação de trailing stop
        """
        print("\n" + "="*60)
        print("EXEMPLO 3: Simulação de Trailing Stop")
        print("="*60)
        
        symbol = "USDJPY"
        entry_price = 150.00
        direction = 1  # Compra
        initial_sl = 149.20
        
        # Simular movimento de preços
        price_sequence = [150.00, 150.25, 150.60, 150.45, 150.80, 151.10, 150.95, 151.30]
        
        self.logger.info(f"Simulando trailing stop para {symbol}")
        
        try:
            # Inicializar trailing stop
            trailing_info = self.conservative_system.initialize_trailing_stop(
                symbol=symbol,
                entry_price=entry_price,
                direction=direction,
                initial_sl=initial_sl
            )
            
            print(f"Símbolo: {symbol}")
            print(f"Preço de entrada: {entry_price:.2f}")
            print(f"Stop Loss inicial: {initial_sl:.2f}")
            print(f"\nSimulação de movimento de preços:")
            
            for i, price in enumerate(price_sequence):
                # Atualizar trailing stop
                trailing_info = self.conservative_system.update_trailing_stop(
                    trailing_info, price
                )
                
                status = "ATIVO" if trailing_info.get('active', False) else "INATIVO"
                current_sl = trailing_info.get('current_sl', initial_sl)
                profit = (price - entry_price) * direction
                
                print(f"  Tick {i+1}: Preço {price:.2f} | SL {current_sl:.2f} | "
                      f"Trailing {status} | Lucro {profit:.2f}")
                
                # Verificar se foi atingido breakeven
                if trailing_info.get('breakeven_activated', False):
                    print(f"    → Breakeven ativado!")
            
            print(f"\nResultado final:")
            print(f"SL final: {trailing_info.get('current_sl', initial_sl):.2f}")
            print(f"Trailing ativo: {trailing_info.get('active', False)}")
            print(f"Breakeven ativado: {trailing_info.get('breakeven_activated', False)}")
            
        except Exception as e:
            self.logger.error(f"Erro na simulação trailing stop: {e}")
            print(f"Erro: {e}")
    
    def example_gap_protection_analysis(self):
        """
        Exemplo de análise de proteção contra gaps
        """
        print("\n" + "="*60)
        print("EXEMPLO 4: Análise de Proteção contra Gaps")
        print("="*60)
        
        scenarios = [
            {"symbol": "EURUSD", "entry": 1.1000, "current": 1.1005, "desc": "Gap pequeno"},
            {"symbol": "GBPUSD", "entry": 1.2500, "current": 1.2580, "desc": "Gap médio"},
            {"symbol": "USDJPY", "entry": 150.00, "current": 151.50, "desc": "Gap grande"},
            {"symbol": "AUDUSD", "entry": 0.6500, "current": 0.6350, "desc": "Gap negativo"}
        ]
        
        for scenario in scenarios:
            symbol = scenario["symbol"]
            entry_price = scenario["entry"]
            current_price = scenario["current"]
            description = scenario["desc"]
            
            self.logger.info(f"Analisando gap para {symbol}: {description}")
            
            try:
                result = self.conservative_system.check_gap_protection(
                    symbol=symbol,
                    entry_price=entry_price,
                    current_price=current_price,
                    position_size=0.1
                )
                
                gap_size = abs(current_price - entry_price)
                gap_percentage = (gap_size / entry_price) * 100
                
                print(f"\n{description.upper()} - {symbol}:")
                print(f"  Preço entrada: {entry_price:.4f}")
                print(f"  Preço atual: {current_price:.4f}")
                print(f"  Gap: {gap_size:.4f} ({gap_percentage:.2f}%)")
                print(f"  Proteção necessária: {result.get('protection_needed', False)}")
                print(f"  Nível de risco: {result.get('risk_level', 'N/A')}")
                
                if 'recommendations' in result and result['recommendations']:
                    print(f"  Recomendações:")
                    for rec in result['recommendations']:
                        print(f"    - {rec}")
                
            except Exception as e:
                self.logger.error(f"Erro na análise de gap para {symbol}: {e}")
                print(f"  Erro: {e}")
    
    def example_comprehensive_analysis(self):
        """
        Exemplo de análise completa integrada
        """
        print("\n" + "="*60)
        print("EXEMPLO 5: Análise Completa Integrada")
        print("="*60)
        
        symbol = "EURUSD"
        entry_price = 1.1000
        direction = 1
        confidence = 0.8
        position_size = 0.1
        
        self.logger.info(f"Executando análise completa para {symbol}")
        
        try:
            # Usar o gerenciador de integração
            result = self.integration_manager.get_comprehensive_risk_analysis(
                symbol=symbol,
                entry_price=entry_price,
                direction=direction,
                confidence=confidence,
                position_size=position_size
            )
            
            print(f"ANÁLISE COMPLETA - {symbol}")
            print(f"Preço de entrada: {entry_price:.5f}")
            print(f"Direção: {'COMPRA' if direction == 1 else 'VENDA'}")
            print(f"Confiança: {confidence:.1%}")
            print(f"Tamanho da posição: {position_size} lotes")
            
            print(f"\n📊 RESULTADOS:")
            print(f"Stop Loss: {result.get('sl', 'N/A')}")
            print(f"Take Profit: {result.get('tp', 'N/A')}")
            print(f"Posição otimizada: {result.get('optimized_position_size', 'N/A')} lotes")
            print(f"Score de risco: {result.get('risk_score', 'N/A')}/10")
            
            if 'technical_levels' in result:
                levels = result['technical_levels']
                print(f"\n🎯 NÍVEIS TÉCNICOS:")
                print(f"Suporte mais próximo: {levels.get('nearest_support', 'N/A')}")
                print(f"Resistência mais próxima: {levels.get('nearest_resistance', 'N/A')}")
            
            if 'scaled_tp' in result:
                tp_levels = result['scaled_tp']
                print(f"\n📈 TAKE PROFIT ESCALONADO:")
                for i, level in enumerate(tp_levels.get('levels', []), 1):
                    percentage = tp_levels.get('percentages', [])[i-1] if i-1 < len(tp_levels.get('percentages', [])) else 'N/A'
                    print(f"  TP{i}: {level} ({percentage}% da posição)")
            
            if 'recommendations' in result:
                print(f"\n💡 RECOMENDAÇÕES:")
                for rec in result['recommendations']:
                    print(f"  - {rec}")
            
        except Exception as e:
            self.logger.error(f"Erro na análise completa: {e}")
            print(f"Erro: {e}")
    
    def run_all_examples(self):
        """
        Executar todos os exemplos
        """
        print("🚀 INICIANDO EXEMPLOS DO SISTEMA AVANÇADO DE RISCO")
        print("="*80)
        
        examples = [
            self.example_basic_sl_tp_calculation,
            self.example_kelly_optimization,
            self.example_trailing_stop_simulation,
            self.example_gap_protection_analysis,
            self.example_comprehensive_analysis
        ]
        
        for i, example in enumerate(examples, 1):
            try:
                self.logger.info(f"Executando exemplo {i}: {example.__name__}")
                example()
            except Exception as e:
                self.logger.error(f"Erro no exemplo {i}: {e}")
                print(f"\n❌ Erro no exemplo {i}: {e}")
        
        print("\n" + "="*80)
        print("✅ TODOS OS EXEMPLOS CONCLUÍDOS")
        print("📋 Verifique o arquivo 'advanced_risk_system.log' para logs detalhados")
        print("="*80)

def main():
    """
    Função principal para executar os exemplos
    """
    print("Sistema Avançado de Gestão de Risco - Exemplos Práticos")
    print("Versão: 1.0")
    print("Data: 2024")
    
    # Criar mock do cliente MT5 para demonstração
    class MockMT5Client:
        def get_rates(self, symbol, timeframe, count):
            # Retornar dados simulados
            import pandas as pd
            import numpy as np
            
            # Gerar dados OHLC simulados
            np.random.seed(42)  # Para resultados consistentes
            base_price = 1.1000 if 'EUR' in symbol else 1.2500 if 'GBP' in symbol else 150.0
            
            data = []
            for i in range(count):
                high = base_price + np.random.uniform(0.0005, 0.0020)
                low = base_price - np.random.uniform(0.0005, 0.0020)
                close = base_price + np.random.uniform(-0.0010, 0.0010)
                data.append({'high': high, 'low': low, 'close': close})
                base_price = close  # Próximo candle baseado no close anterior
            
            return pd.DataFrame(data)
    
    # Inicializar exemplo com mock
    mock_client = MockMT5Client()
    example = AdvancedRiskExample(mt5_client=mock_client)
    
    # Executar todos os exemplos
    example.run_all_examples()

if __name__ == '__main__':
    main()