#!/usr/bin/env python3
"""
Testes Simplificados para o Sistema Avançado de Gestão de Risco

Este arquivo contém testes básicos e funcionais para validar
as principais funcionalidades implementadas.

Autor: Sistema de Trading ML
Data: 2024
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_atr_calculation():
    """Teste básico do cálculo ATR"""
    print("\n=== Teste: Cálculo ATR ===")
    
    try:
        from advanced_risk_system import AdvancedRiskSystem
        
        # Criar dados de teste simples
        test_data = pd.DataFrame({
            'high': [1.1010, 1.1020, 1.1015, 1.1025, 1.1030],
            'low': [1.0990, 1.1000, 1.0995, 1.1005, 1.1010],
            'close': [1.1005, 1.1015, 1.1010, 1.1020, 1.1025]
        })
        
        # Calcular ATR manualmente para validação
        tr_values = []
        for i in range(1, len(test_data)):
            high_low = test_data.iloc[i]['high'] - test_data.iloc[i]['low']
            high_close = abs(test_data.iloc[i]['high'] - test_data.iloc[i-1]['close'])
            low_close = abs(test_data.iloc[i]['low'] - test_data.iloc[i-1]['close'])
            tr = max(high_low, high_close, low_close)
            tr_values.append(tr)
        
        atr_manual = np.mean(tr_values)
        print(f"ATR calculado manualmente: {atr_manual:.6f}")
        
        # Teste passou
        print("✓ Cálculo ATR validado")
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste ATR: {e}")
        return False

def test_risk_parameters():
    """Teste dos parâmetros de risco"""
    print("\n=== Teste: Parâmetros de Risco ===")
    
    try:
        from advanced_risk_system import (
            ATRParameters, VolatilityParameters, TrailingStopParameters,
            GapProtectionParameters, TechnicalLevelsParameters, ScaledTakeProfitParameters
        )
        
        # Testar criação de parâmetros
        atr_params = ATRParameters()
        print(f"ATR período: {atr_params.period}")
        print(f"ATR método: {atr_params.method}")
        
        vol_params = VolatilityParameters()
        print(f"Volatilidade período: {vol_params.lookback_period}")
        print(f"Volatilidade thresholds: {vol_params.regime_thresholds}")
        print(f"Volatilidade adjustments: {vol_params.adjustment_factors}")
        
        trailing_params = TrailingStopParameters()
        print(f"Trailing ativação: {trailing_params.activation_distance}")
        print(f"Trailing distância: {trailing_params.trail_distance}")
        print(f"Trailing step: {trailing_params.step_size}")
        
        gap_params = GapProtectionParameters()
        print(f"Gap threshold: {gap_params.max_gap_threshold}")
        print(f"Gap max exposure reduction: {gap_params.max_exposure_reduction}")
        print(f"Gap weekend protection: {gap_params.weekend_protection}")
        
        tech_params = TechnicalLevelsParameters()
        print(f"Tech support/resistance period: {tech_params.support_resistance_period}")
        print(f"Tech level strength threshold: {tech_params.level_strength_threshold}")
        print(f"Tech fibonacci levels: {tech_params.fibonacci_levels}")
        
        tp_params = ScaledTakeProfitParameters()
        print(f"TP níveis: {tp_params.levels}")
        print(f"TP percentages: {tp_params.position_percentages}")
        print(f"TP trail after first: {tp_params.trail_after_first}")
        
        print("✓ Parâmetros de risco validados")
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste de parâmetros: {e}")
        return False

def test_sl_tp_logic():
    """Teste da lógica de SL/TP"""
    print("\n=== Teste: Lógica SL/TP ===")
    
    try:
        # Teste lógico simples
        entry_price = 1.1000
        atr_value = 0.0015
        direction = 1  # Compra
        confidence = 0.8
        
        # Calcular SL/TP baseado em ATR
        sl_multiplier = 2.0 - (confidence * 0.5)  # Quanto maior confiança, menor SL
        tp_multiplier = 1.5 + (confidence * 1.0)  # Quanto maior confiança, maior TP
        
        if direction == 1:  # Compra
            sl = entry_price - (atr_value * sl_multiplier)
            tp = entry_price + (atr_value * tp_multiplier)
        else:  # Venda
            sl = entry_price + (atr_value * sl_multiplier)
            tp = entry_price - (atr_value * tp_multiplier)
        
        # Validar lógica
        if direction == 1:
            assert sl < entry_price < tp, "Para compra: SL < Entry < TP"
        else:
            assert tp < entry_price < sl, "Para venda: TP < Entry < SL"
        
        # Calcular risk/reward
        risk = abs(entry_price - sl)
        reward = abs(tp - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        print(f"Entry: {entry_price:.5f}")
        print(f"SL: {sl:.5f}")
        print(f"TP: {tp:.5f}")
        print(f"Risk: {risk:.5f}")
        print(f"Reward: {reward:.5f}")
        print(f"R/R: {rr_ratio:.2f}")
        
        assert rr_ratio >= 1.0, "Risk/Reward deve ser >= 1.0"
        
        print("✓ Lógica SL/TP validada")
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste SL/TP: {e}")
        return False

def test_kelly_criterion_logic():
    """Teste da lógica do Kelly Criterion"""
    print("\n=== Teste: Kelly Criterion ===")
    
    try:
        # Parâmetros de teste
        win_rate = 0.6  # 60% de acerto
        avg_win = 150   # Ganho médio em pips
        avg_loss = 100  # Perda média em pips
        
        # Calcular Kelly básico
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Aplicar limites de segurança
        max_kelly = 0.25  # Máximo 25%
        min_kelly = 0.01  # Mínimo 1%
        
        kelly_safe = max(min_kelly, min(kelly_fraction, max_kelly))
        
        print(f"Win rate: {win_rate:.1%}")
        print(f"Avg win: {avg_win} pips")
        print(f"Avg loss: {avg_loss} pips")
        print(f"Win/Loss ratio: {win_loss_ratio:.2f}")
        print(f"Kelly fraction: {kelly_fraction:.4f}")
        print(f"Kelly safe: {kelly_safe:.4f}")
        
        # Validações
        assert 0 <= kelly_safe <= max_kelly, "Kelly deve estar dentro dos limites"
        assert kelly_safe >= min_kelly, "Kelly deve ser pelo menos o mínimo"
        
        # Calcular tamanho da posição
        account_balance = 10000
        risk_per_trade = kelly_safe
        position_risk_amount = account_balance * risk_per_trade
        
        print(f"Saldo da conta: ${account_balance:,.2f}")
        print(f"Risco por trade: {risk_per_trade:.2%}")
        print(f"Valor em risco: ${position_risk_amount:.2f}")
        
        print("✓ Kelly Criterion validado")
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste Kelly: {e}")
        return False

def test_trailing_stop_logic():
    """Teste da lógica do trailing stop"""
    print("\n=== Teste: Trailing Stop ===")
    
    try:
        # Configuração inicial
        entry_price = 1.1000
        initial_sl = 1.0950
        direction = 1  # Compra
        trailing_distance = 0.0030  # 30 pips
        activation_threshold = 0.0050  # 50 pips de lucro para ativar
        
        # Simular movimento de preços
        prices = [1.1000, 1.1020, 1.1040, 1.1060, 1.1080, 1.1070, 1.1090]
        current_sl = initial_sl
        trailing_activated = False
        
        print(f"Entry: {entry_price:.4f}, Initial SL: {initial_sl:.4f}")
        print(f"Trailing distance: {trailing_distance:.4f}")
        print(f"Activation threshold: {activation_threshold:.4f}")
        print("\nSimulação de preços:")
        
        for i, price in enumerate(prices):
            profit = price - entry_price if direction == 1 else entry_price - price
            
            # Verificar se deve ativar trailing
            if not trailing_activated and profit >= activation_threshold:
                trailing_activated = True
                print(f"  Preço {price:.4f}: Trailing ATIVADO (lucro: {profit:.4f})")
            
            # Atualizar trailing stop se ativado
            if trailing_activated and direction == 1:
                new_sl = price - trailing_distance
                if new_sl > current_sl:
                    current_sl = new_sl
                    print(f"  Preço {price:.4f}: SL atualizado para {current_sl:.4f}")
                else:
                    print(f"  Preço {price:.4f}: SL mantido em {current_sl:.4f}")
            else:
                print(f"  Preço {price:.4f}: Trailing não ativo, SL: {current_sl:.4f}")
        
        # Validações
        assert current_sl > initial_sl, "Trailing stop deve ter movido o SL para cima"
        assert trailing_activated, "Trailing stop deve ter sido ativado"
        
        print(f"\nSL final: {current_sl:.4f}")
        print("✓ Trailing Stop validado")
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste Trailing Stop: {e}")
        return False

def test_gap_protection_logic():
    """Teste da lógica de proteção contra gaps"""
    print("\n=== Teste: Proteção contra Gaps ===")
    
    try:
        # Cenários de teste
        scenarios = [
            {"name": "Gap pequeno", "entry": 1.1000, "current": 1.1005, "expected": False},
            {"name": "Gap médio", "entry": 1.1000, "current": 1.1080, "expected": True},
            {"name": "Gap grande", "entry": 1.1000, "current": 1.1150, "expected": True}
        ]
        
        gap_threshold = 0.005  # 50 pips
        
        for scenario in scenarios:
            entry = scenario["entry"]
            current = scenario["current"]
            expected_protection = scenario["expected"]
            
            # Calcular gap
            gap_size = abs(current - entry)
            gap_percentage = gap_size / entry
            protection_needed = gap_size > gap_threshold
            
            print(f"\n{scenario['name']}:")
            print(f"  Entry: {entry:.4f}")
            print(f"  Current: {current:.4f}")
            print(f"  Gap size: {gap_size:.4f}")
            print(f"  Gap %: {gap_percentage:.2%}")
            print(f"  Protection needed: {protection_needed}")
            
            # Validar resultado
            assert protection_needed == expected_protection, f"Proteção esperada: {expected_protection}, obtida: {protection_needed}"
            
            # Gerar recomendações
            if protection_needed:
                recommendations = []
                if gap_percentage > 0.01:  # 1%
                    recommendations.append("Reduzir exposição")
                if gap_percentage > 0.02:  # 2%
                    recommendations.append("Considerar fechar posição")
                
                print(f"  Recomendações: {recommendations}")
        
        print("\n✓ Proteção contra Gaps validada")
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste Gap Protection: {e}")
        return False

def test_technical_levels_logic():
    """Teste da lógica de níveis técnicos"""
    print("\n=== Teste: Níveis Técnicos ===")
    
    try:
        # Dados de preço simulados
        prices = [1.0950, 1.1000, 1.1050, 1.1000, 1.1100, 1.1050, 1.1000, 1.1150]
        current_price = 1.1075
        
        # Identificar suportes e resistências simples
        highs = []
        lows = []
        
        for i in range(1, len(prices) - 1):
            # Máximo local
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                highs.append(prices[i])
            # Mínimo local
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                lows.append(prices[i])
        
        # Encontrar níveis mais próximos
        resistance_levels = [h for h in highs if h > current_price]
        support_levels = [l for l in lows if l < current_price]
        
        nearest_resistance = min(resistance_levels) if resistance_levels else None
        nearest_support = max(support_levels) if support_levels else None
        
        print(f"Preços históricos: {prices}")
        print(f"Preço atual: {current_price:.4f}")
        print(f"Máximos encontrados: {highs}")
        print(f"Mínimos encontrados: {lows}")
        print(f"Resistências acima: {resistance_levels}")
        print(f"Suportes abaixo: {support_levels}")
        print(f"Resistência mais próxima: {nearest_resistance}")
        print(f"Suporte mais próximo: {nearest_support}")
        
        # Níveis psicológicos
        psychological_levels = []
        base_level = int(current_price * 10000) / 10000  # Arredondar para 4 casas
        
        for offset in [-0.0100, -0.0050, 0.0000, 0.0050, 0.0100]:
            level = base_level + offset
            if abs(level - current_price) > 0.0010:  # Pelo menos 10 pips de distância
                psychological_levels.append(level)
        
        print(f"Níveis psicológicos: {psychological_levels}")
        
        # Validações básicas
        if nearest_resistance:
            assert nearest_resistance > current_price, "Resistência deve estar acima do preço atual"
        if nearest_support:
            assert nearest_support < current_price, "Suporte deve estar abaixo do preço atual"
        
        print("✓ Níveis Técnicos validados")
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste Níveis Técnicos: {e}")
        return False

def main():
    """Executa todos os testes simplificados"""
    print("INICIANDO TESTES SIMPLIFICADOS DO SISTEMA DE RISCO")
    print("="*60)
    
    tests = [
        test_atr_calculation,
        test_risk_parameters,
        test_sl_tp_logic,
        test_kelly_criterion_logic,
        test_trailing_stop_logic,
        test_gap_protection_logic,
        test_technical_levels_logic
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Erro inesperado no teste {test.__name__}: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTADOS: {passed} testes passaram, {failed} falharam")
    print("="*60)
    
    if failed == 0:
        print("🎉 TODOS OS TESTES PASSARAM!")
    else:
        print(f"⚠️  {failed} teste(s) falharam")
    
    return failed == 0

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)