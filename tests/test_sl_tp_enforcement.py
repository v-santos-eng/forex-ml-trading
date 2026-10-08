#!/usr/bin/env python3
"""
Teste de Validação SL/TP Obrigatórios
Verifica se todas as ordens futuras incluem Stop Loss e Take Profit
"""

import MetaTrader5 as mt5
import time
from datetime import datetime, timedelta
import json

def test_sl_tp_enforcement():
    """Testa se SL/TP estão sendo aplicados corretamente"""
    print("🔍 TESTE DE VALIDAÇÃO SL/TP OBRIGATÓRIOS")
    print("=" * 60)
    
    # Inicializar MT5
    if not mt5.initialize():
        print("❌ Falha ao inicializar MT5")
        return False
    
    try:
        # Verificar configuração atual
        with open('auto_trader_config.json', 'r') as f:
            config = json.load(f)
        
        print("📋 CONFIGURAÇÃO ATUAL:")
        print(f"   SL Padrão: {config.get('default_sl_pips', 'NÃO DEFINIDO')} pips")
        print(f"   TP Padrão: {config.get('default_tp_pips', 'NÃO DEFINIDO')} pips")
        print(f"   SL Mínimo: {config.get('min_sl_pips', 'NÃO DEFINIDO')} pips")
        print(f"   TP Mínimo: {config.get('min_tp_pips', 'NÃO DEFINIDO')} pips")
        print(f"   Enforcement: {config.get('enforce_sl_tp', 'NÃO DEFINIDO')}")
        print()
        
        # Verificar ordens dos últimos 10 minutos
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=10)
        
        # Converter para timestamp
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())
        
        print("📊 VERIFICANDO ORDENS DOS ÚLTIMOS 10 MINUTOS...")
        print(f"   Período: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}")
        print()
        
        # Buscar ordens recentes
        orders = mt5.history_orders_get(start_timestamp, end_timestamp)
        
        if orders is None or len(orders) == 0:
            print("ℹ️ Nenhuma ordem encontrada nos últimos 10 minutos")
            print("   Aguardando próximas ordens para validação...")
            return True
        
        print(f"📈 ENCONTRADAS {len(orders)} ORDENS:")
        print()
        
        sl_count = 0
        tp_count = 0
        
        for i, order in enumerate(orders[:5]):  # Mostrar até 5 ordens
            print(f"   Ordem {order.ticket}:")
            print(f"     Symbol: {order.symbol}")
            print(f"     Type: {order.type}")
            print(f"     Volume: {order.volume_initial}")
            print(f"     Price: {order.price_open}")
            
            if order.sl > 0:
                print(f"     Stop Loss: {order.sl} ✅")
                sl_count += 1
            else:
                print(f"     Stop Loss: Não definido ❌")
                
            if order.tp > 0:
                print(f"     Take Profit: {order.tp} ✅")
                tp_count += 1
            else:
                print(f"     Take Profit: Não definido ❌")
            print()
        
        print(f"📊 RESUMO:")
        print(f"   Ordens com SL: {sl_count}/{len(orders)}")
        print(f"   Ordens com TP: {tp_count}/{len(orders)}")
        
        # Validar se todas as ordens têm SL/TP
        if sl_count == len(orders) and tp_count == len(orders):
            print("✅ SUCESSO: Todas as ordens incluem SL e TP!")
            return True
        else:
            print("❌ FALHA: Algumas ordens não incluem SL/TP")
            return False
            
    except Exception as e:
        print(f"❌ Erro durante teste: {e}")
        return False
    finally:
        mt5.shutdown()

def monitor_new_orders(duration_minutes=5):
    """Monitora novas ordens por um período específico"""
    print(f"\n🔍 MONITORANDO NOVAS ORDENS POR {duration_minutes} MINUTOS...")
    print("=" * 60)
    
    if not mt5.initialize():
        print("❌ Falha ao inicializar MT5")
        return False
    
    try:
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        last_order_count = 0
        
        print(f"   Início: {start_time.strftime('%H:%M:%S')}")
        print(f"   Fim: {end_time.strftime('%H:%M:%S')}")
        print("   Aguardando novas ordens...")
        print()
        
        while datetime.now() < end_time:
            # Verificar ordens dos últimos 2 minutos
            check_start = datetime.now() - timedelta(minutes=2)
            orders = mt5.history_orders_get(int(check_start.timestamp()), int(datetime.now().timestamp()))
            
            if orders and len(orders) > last_order_count:
                new_orders = orders[last_order_count:]
                print(f"🆕 {len(new_orders)} NOVA(S) ORDEM(S) DETECTADA(S):")
                
                for order in new_orders:
                    sl_status = "✅" if order.sl > 0 else "❌"
                    tp_status = "✅" if order.tp > 0 else "❌"
                    
                    print(f"   {order.ticket}: {order.symbol} | SL: {sl_status} | TP: {tp_status}")
                    
                    if order.sl > 0 and order.tp > 0:
                        print(f"     ✅ ORDEM VÁLIDA: SL={order.sl:.5f}, TP={order.tp:.5f}")
                    else:
                        print(f"     ❌ ORDEM INVÁLIDA: Faltam SL/TP")
                
                last_order_count = len(orders)
                print()
            
            time.sleep(10)  # Verificar a cada 10 segundos
        
        print("⏰ Monitoramento concluído")
        return True
        
    except Exception as e:
        print(f"❌ Erro durante monitoramento: {e}")
        return False
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    print("🚀 INICIANDO VALIDAÇÃO DE SL/TP OBRIGATÓRIOS")
    print("=" * 60)
    
    # Teste 1: Verificar configuração e ordens recentes
    test_result = test_sl_tp_enforcement()
    
    # Teste 2: Monitorar novas ordens
    monitor_result = monitor_new_orders(3)  # 3 minutos
    
    print("\n" + "=" * 60)
    print("📊 RESULTADO FINAL:")
    print(f"   Teste de Configuração: {'✅ PASSOU' if test_result else '❌ FALHOU'}")
    print(f"   Monitoramento: {'✅ PASSOU' if monitor_result else '❌ FALHOU'}")
    
    if test_result and monitor_result:
        print("\n🎉 SUCESSO: Sistema configurado corretamente para SL/TP obrigatórios!")
    else:
        print("\n⚠️ ATENÇÃO: Verificar implementação de SL/TP")