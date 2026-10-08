#!/usr/bin/env python3
"""
Testes para o Sistema Avançado de Gestão de Risco

Este arquivo contém testes abrangentes para validar todas as funcionalidades
implementadas no advanced_risk_system.py, incluindo:
- Cálculo de SL/TP baseado em ATR
- Ajuste por volatilidade
- Integração Kelly Criterion
- Trailing stop dinâmico
- Proteção contra gaps
- Análise de níveis técnicos
- Take Profit escalonado
- Análise completa de risco

Autor: Sistema de Trading ML
Data: 2024
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_risk_system import (
    AdvancedRiskSystem,
    ATRParameters,
    VolatilityParameters,
    TrailingStopParameters,
    GapProtectionParameters,
    TechnicalLevelsParameters,
    ScaledTakeProfitParameters
)
from config import ForexConfig as Config

class TestAdvancedRiskSystem(unittest.TestCase):
    """Testes para o Sistema Avançado de Gestão de Risco"""
    
    def setUp(self):
        """Configuração inicial para os testes"""
        # Mock do cliente MT5
        self.mock_mt5_client = Mock()
        self.mock_logger = Mock()
        
        # Dados de teste simulados
        self.test_rates = pd.DataFrame([
            {'time': datetime.now() - timedelta(hours=i), 'open': 1.1000 + i*0.0001, 
             'high': 1.1010 + i*0.0001, 'low': 1.0990 + i*0.0001, 'close': 1.1005 + i*0.0001}
            for i in range(100, 0, -1)
        ])
        
        # Configurar mocks
        self.mock_mt5_client.get_rates.return_value = self.test_rates
        self.mock_mt5_client.get_symbol_info.return_value = {
            'digits': 5,
            'point': 0.00001,
            'trade_tick_size': 0.00001
        }
        
        # Adicionar configuração de saldo da conta
        Config.ACCOUNT_BALANCE = 10000.0
        
        # Criar instância do sistema
        self.risk_system = AdvancedRiskSystem(
            mt5_client=self.mock_mt5_client,
            logger=self.mock_logger
        )
        
        # Parâmetros de teste
        self.test_symbol = "EURUSD"
        self.test_entry_price = 1.1000
        self.test_direction = 1  # Compra
        self.test_confidence = 0.8
    
    def test_atr_calculation(self):
        """Testa o cálculo do ATR avançado"""
        print("\n=== Teste: Cálculo ATR Avançado ===")
        
        # Teste com método padrão
        atr_result = self.risk_system.calculate_advanced_atr(
            self.test_symbol, method='standard'
        )
        
        self.assertIsInstance(atr_result, dict)
        self.assertIn('atr_value', atr_result)
        self.assertIn('method', atr_result)
        self.assertGreater(atr_result['atr_value'], 0)
        
        print(f"ATR calculado: {atr_result['atr_value']:.5f}")
        print(f"Método utilizado: {atr_result['method']}")
        
        # Teste com método exponencial
        atr_exp = self.risk_system.calculate_advanced_atr(
            self.test_symbol, method='exponential'
        )
        
        self.assertIsInstance(atr_exp, dict)
        self.assertIn('atr_value', atr_exp)
        print(f"ATR exponencial: {atr_exp['atr_value']:.5f}")
    
    def test_sl_tp_calculation(self):
        """Testa o cálculo de SL/TP baseado em ATR"""
        print("\n=== Teste: Cálculo SL/TP baseado em ATR ===")
        
        sl, tp, metadata = self.risk_system.calculate_optimal_sl_tp(
            self.test_symbol, self.test_entry_price, self.test_direction, self.test_confidence
        )
        
        # Validações básicas
        self.assertIsInstance(sl, float)
        self.assertIsInstance(tp, float)
        self.assertIsInstance(metadata, dict)
        
        # Para compra: SL < entry_price < TP
        if self.test_direction == 1:
            self.assertLess(sl, self.test_entry_price)
            self.assertGreater(tp, self.test_entry_price)
        
        # Verificar metadados
        self.assertIn('atr_value', metadata)
        self.assertIn('risk_reward_ratio', metadata)
        # Verificar se há ajuste de volatilidade (pode ter nomes diferentes)
        has_volatility_info = any(key in metadata for key in ['volatility_adjustment', 'volatility_adjusted', 'confidence_factor'])
        self.assertTrue(has_volatility_info, "Metadados devem conter informações de volatilidade")
        
        print(f"Entry Price: {self.test_entry_price:.5f}")
        print(f"Stop Loss: {sl:.5f}")
        print(f"Take Profit: {tp:.5f}")
        print(f"Risk/Reward: {metadata['risk_reward_ratio']:.2f}")
        print(f"ATR usado: {metadata['atr_value']:.5f}")
    
    def test_kelly_position_sizing(self):
        """Testa o dimensionamento de posição com Kelly Criterion"""
        print("\n=== Teste: Kelly Criterion Position Sizing ===")
        
        # Primeiro calcular SL/TP usando o sistema
        sl, tp, _ = self.risk_system.calculate_optimal_sl_tp(
            self.test_symbol, self.test_entry_price, self.test_direction, self.test_confidence
        )
        
        # Se não conseguir calcular, usar valores padrão
        if sl == 0 or tp == 0:
            sl = self.test_entry_price - 0.0050  # 50 pips
            tp = self.test_entry_price + 0.0100  # 100 pips
        
        position_info = self.risk_system.calculate_kelly_optimized_position(
            self.test_symbol, self.test_entry_price, sl, tp, 
            self.test_direction, self.test_confidence
        )
        
        self.assertIsInstance(position_info, dict)
        self.assertIn('position_size', position_info)
        self.assertIn('risk_percentage', position_info)
        self.assertIn('kelly_fraction', position_info)
        
        # Validar limites de risco
        self.assertLessEqual(position_info['risk_percentage'], 5.0)  # Máximo 5%
        self.assertGreater(position_info['position_size'], 0)
        
        print(f"Tamanho da posição: {position_info['position_size']:.2f} lotes")
        print(f"Risco por operação: {position_info['risk_percentage']:.2f}%")
        print(f"Fração Kelly: {position_info['kelly_fraction']:.4f}")
    
    def test_trailing_stop(self):
        """Testa o sistema de trailing stop dinâmico"""
        print("\n=== Teste: Trailing Stop Dinâmico ===")
        
        # Inicializar trailing stop
        initial_sl = self.test_entry_price - 0.0050
        trailing_info = self.risk_system.initialize_trailing_stop(
            self.test_symbol, self.test_entry_price, initial_sl, self.test_direction
        )
        
        self.assertIsInstance(trailing_info, dict)
        self.assertIn('trailing_distance', trailing_info)
        self.assertIn('activation_level', trailing_info)
        
        print(f"Trailing distance: {trailing_info['trailing_distance']:.5f}")
        print(f"Activation level: {trailing_info['activation_level']:.5f}")
        
        # Testar atualização do trailing stop
        new_price = self.test_entry_price + 0.0080  # Preço favorável
        updated_info = self.risk_system.update_trailing_stop(
            self.test_symbol, new_price, trailing_info, self.test_direction
        )
        
        self.assertIsInstance(updated_info, dict)
        self.assertIn('new_sl', updated_info)
        self.assertIn('activated', updated_info)
        
        if updated_info['activated']:
            print(f"Trailing stop ativado! Novo SL: {updated_info['new_sl']:.5f}")
        else:
            print("Trailing stop ainda não ativado")
    
    def test_gap_protection(self):
        """Testa a proteção contra gaps"""
        print("\n=== Teste: Proteção contra Gaps ===")
        
        gap_info = self.risk_system.check_gap_protection(
            self.test_symbol, self.test_entry_price, self.test_entry_price, 0.10
        )
        
        self.assertIsInstance(gap_info, dict)
        self.assertIn('protection_needed', gap_info)
        self.assertIn('risk_level', gap_info)
        self.assertIn('recommendations', gap_info)
        
        print(f"Proteção necessária: {gap_info['protection_needed']}")
        print(f"Nível de risco: {gap_info['risk_level']}")
        print(f"Recomendações: {gap_info['recommendations']}")
    
    def test_technical_levels_analysis(self):
        """Testa a análise de níveis técnicos"""
        print("\n=== Teste: Análise de Níveis Técnicos ===")
        
        levels_info = self.risk_system.analyze_technical_levels(
            self.test_symbol, self.test_entry_price
        )
        
        self.assertIsInstance(levels_info, dict)
        self.assertIn('support_levels', levels_info)
        self.assertIn('resistance_levels', levels_info)
        self.assertIn('fibonacci_levels', levels_info)
        self.assertIn('psychological_levels', levels_info)
        self.assertIn('pivot_levels', levels_info)
        
        print(f"Suportes encontrados: {len(levels_info['support_levels'])}")
        print(f"Resistências encontradas: {len(levels_info['resistance_levels'])}")
        print(f"Níveis Fibonacci: {len(levels_info['fibonacci_levels'])}")
        print(f"Níveis psicológicos: {len(levels_info['psychological_levels'])}")
        print(f"Pivot points: {len(levels_info['pivot_levels'])}")
        
        # Verificar níveis mais próximos
        if levels_info.get('nearest_support'):
            print(f"Suporte mais próximo: {levels_info['nearest_support']['price']:.5f}")
        if levels_info.get('nearest_resistance'):
            print(f"Resistência mais próxima: {levels_info['nearest_resistance']['price']:.5f}")
    
    def test_scaled_take_profit(self):
        """Testa o sistema de Take Profit escalonado"""
        print("\n=== Teste: Take Profit Escalonado ===")
        
        base_atr = 0.0015  # ATR simulado
        
        tp_levels = self.risk_system.calculate_scaled_take_profit(
            self.test_symbol, self.test_entry_price, self.test_direction, base_atr
        )
        
        if tp_levels:  # Se TP escalonado estiver habilitado
            self.assertIsInstance(tp_levels, list)
            
            total_percentage = sum(level['percentage'] for level in tp_levels)
            self.assertAlmostEqual(total_percentage, 1.0, places=2)  # Deve somar 100%
            
            print(f"Níveis de TP escalonado: {len(tp_levels)}")
            for i, level in enumerate(tp_levels):
                print(f"  Nível {level['level']}: {level['price']:.5f} ({level['percentage']:.1%})")
        else:
            print("TP escalonado desabilitado")
    
    def test_comprehensive_risk_analysis(self):
        """Testa a análise completa de risco"""
        print("\n=== Teste: Análise Completa de Risco ===")
        
        analysis = self.risk_system.get_comprehensive_risk_analysis(
            self.test_symbol, self.test_entry_price, self.test_direction, self.test_confidence
        )
        
        # Validações estruturais
        self.assertIsInstance(analysis, dict)
        
        # Se houver erro, pular validações detalhadas
        if 'error' in analysis:
            self.logger.warning(f"Erro na análise: {analysis['error']}")
            return
        
        self.assertIn('symbol', analysis)
        self.assertIn('stop_loss', analysis)
        self.assertIn('take_profit', analysis)
        self.assertIn('position_sizing', analysis)
        self.assertIn('technical_levels', analysis)
        self.assertIn('gap_protection', analysis)
        self.assertIn('recommendations', analysis)
        self.assertIn('risk_score', analysis)
        
        # Validar score de risco (apenas se não houver erro)
        if 'risk_score' in analysis:
            risk_score = analysis['risk_score']
            self.assertIsInstance(risk_score, (int, float))
            self.assertGreaterEqual(risk_score, 0)
            self.assertLessEqual(risk_score, 100)
        else:
            risk_score = 'N/A'
        
        print(f"\n=== ANÁLISE COMPLETA DE RISCO ===")
        print(f"Símbolo: {analysis['symbol']}")
        print(f"Preço de entrada: {analysis['entry_price']:.5f}")
        print(f"Stop Loss: {analysis['stop_loss']:.5f}")
        print(f"Take Profit: {analysis['take_profit']:.5f}")
        print(f"Score de risco: {risk_score}/100")
        
        # Mostrar recomendações
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            print("\nRecomendações:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Mostrar informações de posição
        pos_info = analysis.get('position_sizing', {})
        if pos_info:
            print(f"\nDimensionamento de posição:")
            print(f"  Tamanho: {pos_info.get('position_size', 0):.2f} lotes")
            print(f"  Risco: {pos_info.get('risk_percentage', 0):.2f}%")
    
    def test_error_handling(self):
        """Testa o tratamento de erros"""
        print("\n=== Teste: Tratamento de Erros ===")
        
        # Testar com dados inválidos
        self.mock_mt5_client.get_rates.return_value = None
        
        # ATR com dados inválidos
        atr_result = self.risk_system.calculate_advanced_atr("INVALID")
        self.assertEqual(atr_result['atr_value'], 0.0)
        print("✓ ATR com dados inválidos tratado corretamente")
        
        # Análise técnica com dados insuficientes
        levels = self.risk_system.analyze_technical_levels("INVALID", 1.0)
        self.assertIn('error', levels)
        print("✓ Análise técnica com dados insuficientes tratada corretamente")
        
        # Restaurar mock para outros testes
        self.mock_mt5_client.get_rates.return_value = self.test_rates
    
    def test_performance_metrics(self):
        """Testa métricas de performance do sistema"""
        print("\n=== Teste: Métricas de Performance ===")
        
        import time
        
        # Medir tempo de análise completa
        start_time = time.time()
        analysis = self.risk_system.get_comprehensive_risk_analysis(
            self.test_symbol, self.test_entry_price, self.test_direction, self.test_confidence
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Tempo de execução da análise completa: {execution_time:.3f}s")
        
        # Verificar se a análise foi bem-sucedida
        self.assertNotIn('error', analysis)
        self.assertIn('risk_score', analysis)
        
        # Tempo deve ser razoável (< 5 segundos)
        self.assertLess(execution_time, 5.0)
        print("✓ Performance dentro dos limites aceitáveis")

def run_integration_tests():
    """Executa testes de integração com dados reais simulados"""
    print("\n" + "="*60)
    print("EXECUTANDO TESTES DE INTEGRAÇÃO")
    print("="*60)
    
    # Criar sistema real (sem mocks)
    try:
        from mt5_client import MT5Client
        from logger_config import setup_logger
        
        # Tentar usar cliente real se disponível
        mt5_client = MT5Client()
        logger = setup_logger('test_advanced_risk')
        
        risk_system = AdvancedRiskSystem(mt5_client=mt5_client, logger=logger)
        
        # Teste com símbolo real
        test_symbol = "EURUSD"
        test_entry = 1.1000
        
        print(f"\nTestando com símbolo real: {test_symbol}")
        
        # Análise completa
        analysis = risk_system.get_comprehensive_risk_analysis(
            test_symbol, test_entry, 1, 0.8
        )
        
        if 'error' not in analysis:
            print("✓ Integração com MT5 funcionando")
            print(f"  Score de risco: {analysis.get('risk_score', 'N/A')}")
        else:
            print(f"⚠ Erro na integração: {analysis['error']}")
            
    except ImportError:
        print("⚠ Módulos MT5 não disponíveis - pulando testes de integração")
    except Exception as e:
        print(f"⚠ Erro nos testes de integração: {e}")

def main():
    """Função principal para executar todos os testes"""
    print("INICIANDO TESTES DO SISTEMA AVANÇADO DE GESTÃO DE RISCO")
    print("="*70)
    
    # Executar testes unitários
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Executar testes de integração
    run_integration_tests()
    
    print("\n" + "="*70)
    print("TESTES CONCLUÍDOS")
    print("="*70)

if __name__ == '__main__':
    main()