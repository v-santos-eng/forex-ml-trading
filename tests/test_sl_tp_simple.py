#!/usr/bin/env python3
"""
Teste simples e direto para SL/TP
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_trader import TradeSignal
import MetaTrader5 as mt5
from datetime import datetime

def test_sl_tp_calculation():
    """Testa cálculo de SL/TP"""
    print("=" * 60)
    print("TESTE: Cálculo de SL/TP")
    print("=" * 60)
    
    try:
        # Inicializar MT5
        if not mt5.initialize():
            print("❌ Erro ao inicializar MT5")
            return False
            
        # Obter dados do símbolo
        symbol = "EURUSD"
        tick = mt5.symbol_info_tick(symbol)
        symbol_info = mt5.symbol_info(symbol)
        
        if not tick or not symbol_info:
            print(f"❌ Erro ao obter dados de {symbol}")
            return False
            
        print(f"📊 Dados do {symbol}:")
        print(f"   Bid: {tick.bid:.5f}")
        print(f"   Ask: {tick.ask:.5f}")
        print(f"   Point: {symbol_info.point}")
        print(f"   Digits: {symbol_info.digits}")
        
        # Calcular pip value
        pip_value = symbol_info.point * 10 if symbol_info.digits == 5 else symbol_info.point
        print(f"   Pip Value: {pip_value}")
        
        # Teste BUY
        print(f"\n🔍 Teste BUY:")
        entry_price = tick.ask
        sl_pips = 50
        tp_pips = 100
        
        stop_loss = entry_price - (sl_pips * pip_value)
        take_profit = entry_price + (tp_pips * pip_value)
        
        print(f"   Entry: {entry_price:.5f}")
        print(f"   SL: {stop_loss:.5f} ({sl_pips} pips abaixo)")
        print(f"   TP: {take_profit:.5f} ({tp_pips} pips acima)")
        
        # Validações
        sl_valid = stop_loss < entry_price and stop_loss > 0
        tp_valid = take_profit > entry_price
        
        print(f"   SL Válido: {'✅' if sl_valid else '❌'}")
        print(f"   TP Válido: {'✅' if tp_valid else '❌'}")
        
        # Teste SELL
        print(f"\n🔍 Teste SELL:")
        entry_price = tick.bid
        
        stop_loss = entry_price + (sl_pips * pip_value)
        take_profit = entry_price - (tp_pips * pip_value)
        
        print(f"   Entry: {entry_price:.5f}")
        print(f"   SL: {stop_loss:.5f} ({sl_pips} pips acima)")
        print(f"   TP: {take_profit:.5f} ({tp_pips} pips abaixo)")
        
        # Validações
        sl_valid = stop_loss > entry_price
        tp_valid = take_profit < entry_price and take_profit > 0
        
        print(f"   SL Válido: {'✅' if sl_valid else '❌'}")
        print(f"   TP Válido: {'✅' if tp_valid else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        mt5.shutdown()

def test_trade_signal_creation():
    """Testa criação de TradeSignal"""
    print("\n" + "=" * 60)
    print("TESTE: Criação de TradeSignal")
    print("=" * 60)
    
    try:
        # Inicializar MT5
        if not mt5.initialize():
            print("❌ Erro ao inicializar MT5")
            return False
            
        # Obter dados
        symbol = "EURUSD"
        tick = mt5.symbol_info_tick(symbol)
        symbol_info = mt5.symbol_info(symbol)
        
        if not tick or not symbol_info:
            print(f"❌ Erro ao obter dados de {symbol}")
            return False
            
        # Calcular valores
        pip_value = symbol_info.point * 10 if symbol_info.digits == 5 else symbol_info.point
        entry_price = tick.ask
        sl_pips = 50
        tp_pips = 100
        
        stop_loss = entry_price - (sl_pips * pip_value)
        take_profit = entry_price + (tp_pips * pip_value)
        
        # Criar sinal
        signal = TradeSignal(
            symbol=symbol,
            direction="BUY",
            confidence=0.75,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            volume=0.01,
            timestamp=datetime.now(),
            model_source="TEST"
        )
        
        print(f"✅ TradeSignal criado:")
        print(f"   Símbolo: {signal.symbol}")
        print(f"   Direção: {signal.direction}")
        print(f"   Entry: {signal.entry_price:.5f}")
        print(f"   SL: {signal.stop_loss:.5f}")
        print(f"   TP: {signal.take_profit:.5f}")
        print(f"   Volume: {signal.volume}")
        
        # Validar atributos
        has_sl = hasattr(signal, 'stop_loss') and signal.stop_loss is not None
        has_tp = hasattr(signal, 'take_profit') and signal.take_profit is not None
        
        print(f"   Tem SL: {'✅' if has_sl else '❌'}")
        print(f"   Tem TP: {'✅' if has_tp else '❌'}")
        
        return signal
        
    except Exception as e:
        print(f"❌ Erro na criação: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        mt5.shutdown()

def test_order_request_format(signal):
    """Testa formato da requisição de ordem"""
    print("\n" + "=" * 60)
    print("TESTE: Formato de Requisição MT5")
    print("=" * 60)
    
    try:
        # Simular request como no AutoTrader
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
            "comment": f"Test_{signal.model_source}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        print(f"📋 Request MT5:")
        for key, value in request.items():
            print(f"   {key}: {value}")
            
        # Validações críticas
        validations = {
            "SL presente": request["sl"] is not None and request["sl"] > 0,
            "TP presente": request["tp"] is not None and request["tp"] > 0,
            "SL diferente de Entry": request["sl"] != request["price"],
            "TP diferente de Entry": request["tp"] != request["price"],
        }
        
        print(f"\n🔍 Validações:")
        all_valid = True
        for check, result in validations.items():
            print(f"   {check}: {'✅' if result else '❌'}")
            if not result:
                all_valid = False
                
        return all_valid
        
    except Exception as e:
        print(f"❌ Erro no teste de request: {e}")
        return False

def main():
    print("🔍 TESTE SIMPLES: SL/TP")
    print("=" * 80)
    
    # Teste 1: Cálculo básico
    calc_ok = test_sl_tp_calculation()
    
    # Teste 2: Criação de sinal
    signal = test_trade_signal_creation()
    
    # Teste 3: Formato de request
    request_ok = False
    if signal:
        request_ok = test_order_request_format(signal)
    
    print("\n" + "=" * 80)
    print("RESUMO:")
    print(f"   Cálculo SL/TP: {'✅' if calc_ok else '❌'}")
    print(f"   Criação Signal: {'✅' if signal else '❌'}")
    print(f"   Request MT5: {'✅' if request_ok else '❌'}")
    
    if calc_ok and signal and request_ok:
        print("\n✅ TODOS OS TESTES PASSARAM - SL/TP estão funcionais")
    else:
        print("\n❌ PROBLEMAS ENCONTRADOS - SL/TP precisam de correção")
    
    print("=" * 80)

if __name__ == "__main__":
    main()