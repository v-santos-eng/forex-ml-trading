#!/usr/bin/env python3
"""
Teste para verificar se o problema é o formato dos símbolos do Yahoo Finance
"""

import yfinance as yf
from datetime import datetime, timedelta

def test_yahoo_symbols():
    """Testa diferentes formatos de símbolos no Yahoo Finance"""
    print("=" * 60)
    print("TESTE DE SÍMBOLOS DO YAHOO FINANCE")
    print("=" * 60)
    
    # Símbolos para testar
    symbols_to_test = [
        ('EURUSD', 'Formato MT5'),
        ('EURUSD=X', 'Formato Yahoo Finance'),
        ('EUR/USD', 'Formato tradicional'),
        ('GBPUSD', 'Formato MT5'),
        ('GBPUSD=X', 'Formato Yahoo Finance'),
        ('USDJPY', 'Formato MT5'),
        ('USDJPY=X', 'Formato Yahoo Finance')
    ]
    
    for symbol, description in symbols_to_test:
        print(f"\n📊 Testando {symbol} ({description})...")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='7d')
            
            if data is None or data.empty:
                print(f"   ❌ Nenhum dado retornado")
            else:
                print(f"   ✅ Dados obtidos: {len(data)} registros")
                print(f"   📈 Último preço: {data['Close'].iloc[-1]:.5f}")
                print(f"   📅 Período: {data.index[0].date()} a {data.index[-1].date()}")
                
        except Exception as e:
            print(f"   ❌ Erro: {e}")
    
    print("\n" + "=" * 60)
    print("CONCLUSÃO:")
    print("- Símbolos forex no Yahoo Finance precisam do sufixo '=X'")
    print("- Formato correto: EURUSD=X, GBPUSD=X, USDJPY=X")
    print("- O AutoTrader precisa ser corrigido para usar este formato")
    print("=" * 60)

if __name__ == "__main__":
    test_yahoo_symbols()