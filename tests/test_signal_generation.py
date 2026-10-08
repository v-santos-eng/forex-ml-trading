#!/usr/bin/env python3
"""
Teste específico para geração de sinais e validação SL/TP
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_trader import AutoTrader, TradeSignal
import MetaTrader5 as mt5
from datetime import datetime
import numpy as np

def test_signal_generation():
    """Testa a geração de sinais"""
    print("=" * 60)
    print("TESTE: Geração de Sinais")
    print("=" * 60)
    
    try:
        # Inicializar MT5
        if not mt5.initialize():
            print("❌ Erro ao inicializar MT5")
            return False
            
        # Criar AutoTrader
        trader = AutoTrader()
        
        # Testar geração de sinal para EURUSD
        symbol = "EURUSD"
        print(f"\n🔍 Testando geração de sinal para {symbol}...")
        
        signal = trader._generate_ml_signal(symbol)
        
        if signal is None:
            print(f"❌ Nenhum sinal gerado para {symbol}")
            
            # Verificar dados disponíveis
            X, y = trader.ml_predictor.prepare_data(symbol, period='7d')
            print(f"📊 Dados disponíveis: {len(X) if X is not None else 0} amostras")
            
            if X is None or len(X) < 100:
                print("❌ Dados insuficientes para análise")
                return False
                
        else:
            print(f"✅ Sinal gerado com sucesso!")
            print(f"   Símbolo: {signal.symbol}")
            print(f"   Direção: {signal.direction}")
            print(f"   Confiança: {signal.confidence:.3f}")
            print(f"   Entry: {signal.entry_price:.5f}")
            print(f"   SL: {signal.stop_loss:.5f} {'✅' if signal.stop_loss and signal.stop_loss > 0 else '❌'}")
            print(f"   TP: {signal.take_profit:.5f} {'✅' if signal.take_profit and signal.take_profit > 0 else '❌'}")
            print(f"   Volume: {signal.volume}")
            
            # Validar SL/TP
            if signal.direction == "BUY":
                sl_valid = signal.stop_loss < signal.entry_price
                tp_valid = signal.take_profit > signal.entry_price
            else:
                sl_valid = signal.stop_loss > signal.entry_price
                tp_valid = signal.take_profit < signal.entry_price
                
            print(f"   SL Válido: {'✅' if sl_valid else '❌'}")
            print(f"   TP Válido: {'✅' if tp_valid else '❌'}")
            
            return signal
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        mt5.shutdown()

def test_manual_signal_creation():
    """Testa criação manual de sinal"""
    print("\n" + "=" * 60)
    print("TESTE: Criação Manual de Sinal")
    print("=" * 60)
    
    try:
        # Inicializar MT5
        if not mt5.initialize():
            print("❌ Erro ao inicializar MT5")
            return False
            
        # Obter tick atual
        tick = mt5.symbol_info_tick("EURUSD")
        if not tick:
            print("❌ Erro ao obter tick")
            return False
            
        # Criar sinal manual
        entry_price = tick.ask
        sl_pips = 50
        tp_pips = 100
        
        symbol_info = mt5.symbol_info("EURUSD")
        pip_value = symbol_info.point * 10 if symbol_info.digits == 5 else symbol_info.point
        
        stop_loss = entry_price - (sl_pips * pip_value)
        take_profit = entry_price + (tp_pips * pip_value)
        
        signal = TradeSignal(
            symbol="EURUSD",
            direction="BUY",
            confidence=0.75,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            volume=0.01,
            timestamp=datetime.now(),
            model_source="MANUAL_TEST"
        )
        
        print(f"✅ Sinal manual criado:")
        print(f"   Entry: {signal.entry_price:.5f}")
        print(f"   SL: {signal.stop_loss:.5f} ({sl_pips} pips)")
        print(f"   TP: {signal.take_profit:.5f} ({tp_pips} pips)")
        
        return signal
        
    except Exception as e:
        print(f"❌ Erro no teste manual: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        mt5.shutdown()

def test_execute_trade_validation(signal):
    """Testa a validação do execute_trade"""
    print("\n" + "=" * 60)
    print("TESTE: Validação Execute Trade")
    print("=" * 60)
    
    try:
        # Inicializar MT5
        if not mt5.initialize():
            print("❌ Erro ao inicializar MT5")
            return False
            
        # Criar AutoTrader
        trader = AutoTrader()
        
        print(f"🔍 Testando validação para sinal:")
        print(f"   SL: {signal.stop_loss}")
        print(f"   TP: {signal.take_profit}")
        
        # Simular validação do execute_trade
        if not signal.stop_loss or not signal.take_profit:
            print(f"❌ ERRO CRÍTICO: SL ou TP não definidos")
            return False
            
        if signal.stop_loss <= 0 or signal.take_profit <= 0:
            print(f"❌ ERRO CRÍTICO: SL ou TP inválidos")
            return False
            
        # Validar direção
        if signal.direction == "BUY":
            if signal.stop_loss >= signal.entry_price or signal.take_profit <= signal.entry_price:
                print(f"❌ ERRO CRÍTICO: SL/TP incorretos para BUY")
                return False
        else:  # SELL
            if signal.stop_loss <= signal.entry_price or signal.take_profit >= signal.entry_price:
                print(f"❌ ERRO CRÍTICO: SL/TP incorretos para SELL")
                return False
                
        print(f"✅ Validação passou - sinal seria executado")
        return True
        
    except Exception as e:
        print(f"❌ Erro na validação: {e}")
        return False
    finally:
        mt5.shutdown()

def main():
    print("🔍 DIAGNÓSTICO COMPLETO: Geração e Validação de Sinais")
    print("=" * 80)
    
    # Teste 1: Geração de sinal real
    signal = test_signal_generation()
    
    # Teste 2: Criação manual de sinal
    manual_signal = test_manual_signal_creation()
    
    # Teste 3: Validação com sinal real (se disponível)
    if signal:
        test_execute_trade_validation(signal)
    
    # Teste 4: Validação com sinal manual
    if manual_signal:
        test_execute_trade_validation(manual_signal)
    
    print("\n" + "=" * 80)
    print("RESUMO DOS TESTES:")
    print(f"   Sinal Real Gerado: {'✅' if signal else '❌'}")
    print(f"   Sinal Manual Criado: {'✅' if manual_signal else '❌'}")
    print("=" * 80)

if __name__ == "__main__":
    main()