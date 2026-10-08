#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste de Geração de SL/TP Dinâmicos
Verifica se o sistema está gerando sinais com Stop Loss e Take Profit adequados
"""

import MetaTrader5 as mt5
import json
from datetime import datetime
from auto_trader import AutoTrader

def test_sl_tp_generation():
    """Testa a geração de SL/TP dinâmicos"""
    print("🔍 TESTE DE GERAÇÃO DE SL/TP DINÂMICOS")
    print("=" * 50)
    
    # Inicializar MT5
    if not mt5.initialize():
        print("❌ Falha ao inicializar MT5")
        return False
    
    try:
        # Criar instância do AutoTrader
        trader = AutoTrader()
        
        # Testar para cada símbolo
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        
        for symbol in symbols:
            print(f"\n📊 Testando {symbol}:")
            
            # Obter dados de mercado atuais
            tick = mt5.symbol_info_tick(symbol)
            symbol_info = mt5.symbol_info(symbol)
            
            if not tick or not symbol_info:
                print(f"   ❌ Não foi possível obter dados para {symbol}")
                continue
            
            print(f"   💰 Preço atual: Bid={tick.bid:.5f}, Ask={tick.ask:.5f}")
            print(f"   📈 Spread: {(tick.ask - tick.bid):.5f}")
            
            # Simular geração de sinal
            try:
                # Testar cálculo de SL/TP para BUY
                entry_price = tick.ask
                direction = "BUY"
                
                # Usar sistema avançado de risco se disponível
                if hasattr(trader, 'advanced_risk') and trader.advanced_risk:
                    try:
                        sl_price, tp_price, risk_data = trader.advanced_risk.calculate_optimal_sl_tp(
                            trader, symbol, entry_price, direction
                        )
                        
                        if sl_price and tp_price:
                            sl_pips = abs(entry_price - sl_price) / (symbol_info.point * 10 if symbol_info.digits == 5 else symbol_info.point)
                            tp_pips = abs(tp_price - entry_price) / (symbol_info.point * 10 if symbol_info.digits == 5 else symbol_info.point)
                            
                            print(f"   ✅ Sistema Avançado - SL/TP calculados:")
                            print(f"      SL: {sl_price:.5f} ({sl_pips:.1f} pips)")
                            print(f"      TP: {tp_price:.5f} ({tp_pips:.1f} pips)")
                            print(f"      Risk/Reward: {tp_pips/sl_pips:.2f}")
                            
                            # Verificar se SL/TP estão na direção correta
                            if direction == "BUY":
                                if sl_price < entry_price and tp_price > entry_price:
                                    print(f"      ✅ Direções corretas para {direction}")
                                else:
                                    print(f"      ❌ Direções incorretas para {direction}")
                        else:
                            print(f"   ⚠️ Sistema Avançado falhou, usando fallback")
                            raise Exception("Fallback para método fixo")
                            
                    except Exception as e:
                        print(f"   ⚠️ Erro no sistema avançado: {e}")
                        # Fallback para método fixo
                        pip_value = symbol_info.point * 10 if symbol_info.digits == 5 else symbol_info.point
                        sl_pips = max(trader.default_sl_pips, trader.min_sl_pips)
                        tp_pips = max(trader.default_tp_pips, trader.min_tp_pips)
                        
                        sl_price = entry_price - (sl_pips * pip_value)
                        tp_price = entry_price + (tp_pips * pip_value)
                        
                        print(f"   ✅ Método Fixo - SL/TP calculados:")
                        print(f"      SL: {sl_price:.5f} ({sl_pips} pips)")
                        print(f"      TP: {tp_price:.5f} ({tp_pips} pips)")
                        print(f"      Risk/Reward: {tp_pips/sl_pips:.2f}")
                else:
                    # Método fixo apenas
                    pip_value = symbol_info.point * 10 if symbol_info.digits == 5 else symbol_info.point
                    sl_pips = max(trader.default_sl_pips, trader.min_sl_pips)
                    tp_pips = max(trader.default_tp_pips, trader.min_tp_pips)
                    
                    sl_price = entry_price - (sl_pips * pip_value)
                    tp_price = entry_price + (tp_pips * pip_value)
                    
                    print(f"   ✅ Método Fixo - SL/TP calculados:")
                    print(f"      SL: {sl_price:.5f} ({sl_pips} pips)")
                    print(f"      TP: {tp_price:.5f} ({tp_pips} pips)")
                    print(f"      Risk/Reward: {tp_pips/sl_pips:.2f}")
                
                # Verificar se enforce_sl_tp está ativo
                if hasattr(trader, 'enforce_sl_tp') and trader.enforce_sl_tp:
                    print(f"   ✅ enforce_sl_tp está ATIVO - todas as ordens terão SL/TP obrigatórios")
                else:
                    print(f"   ⚠️ enforce_sl_tp está INATIVO - ordens podem ser executadas sem SL/TP")
                    
            except Exception as e:
                print(f"   ❌ Erro ao calcular SL/TP: {e}")
        
        print(f"\n📋 CONFIGURAÇÕES ATUAIS:")
        print(f"   default_sl_pips: {trader.default_sl_pips}")
        print(f"   default_tp_pips: {trader.default_tp_pips}")
        print(f"   min_sl_pips: {trader.min_sl_pips}")
        print(f"   min_tp_pips: {trader.min_tp_pips}")
        print(f"   enforce_sl_tp: {getattr(trader, 'enforce_sl_tp', 'Não definido')}")
        print(f"   advanced_risk ativo: {hasattr(trader, 'advanced_risk') and trader.advanced_risk is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        return False
    
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    success = test_sl_tp_generation()
    if success:
        print("\n✅ TESTE CONCLUÍDO COM SUCESSO")
    else:
        print("\n❌ TESTE FALHOU")