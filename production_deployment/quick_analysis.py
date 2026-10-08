#!/usr/bin/env python3
"""
Análise Rápida de Forex - Sistema Simplificado
Versão robusta para análise imediata de pares de moedas
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def get_forex_data(pair: str) -> pd.DataFrame:
    """Obter dados forex do Yahoo Finance"""
    symbols = {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X', 
        'USDJPY': 'USDJPY=X'
    }
    
    symbol = symbols.get(pair, f'{pair}=X')
    
    try:
        print(f"📈 Obtendo dados para {pair}...")
        data = yf.download(symbol, period='1mo', interval='1h', progress=False)
        
        if data.empty:
            print(f"❌ Sem dados para {pair}")
            return None
            
        # Limpar dados
        data = data.dropna()
        print(f"✅ {len(data)} registros obtidos")
        return data
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calcular RSI"""
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    except:
        return 50.0

def calculate_macd(prices: pd.Series) -> tuple:
    """Calcular MACD"""
    try:
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd.iloc[-1], signal.iloc[-1]
    except:
        return 0.0, 0.0

def calculate_bollinger_position(prices: pd.Series, period: int = 20) -> str:
    """Calcular posição nas Bollinger Bands"""
    try:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        current_price = prices.iloc[-1]
        upper_band = upper.iloc[-1]
        lower_band = lower.iloc[-1]
        
        if current_price > upper_band:
            return 'ABOVE_UPPER'
        elif current_price < lower_band:
            return 'BELOW_LOWER'
        else:
            return 'MIDDLE'
    except:
        return 'MIDDLE'

def analyze_pair(pair: str) -> dict:
    """Análise completa de um par"""
    print(f"\n🔍 ANALISANDO {pair}")
    print("-" * 30)
    
    # Obter dados
    data = get_forex_data(pair)
    if data is None:
        return {'error': 'Sem dados disponíveis'}
    
    try:
        # Extrair preços de fechamento
        close_prices = data['Close']
        current_price = close_prices.iloc[-1]
        
        # Calcular indicadores
        rsi = calculate_rsi(close_prices)
        macd, macd_signal = calculate_macd(close_prices)
        bb_position = calculate_bollinger_position(close_prices)
        
        # Gerar sinais
        buy_signals = 0
        sell_signals = 0
        
        # Análise RSI
        if rsi < 30:
            buy_signals += 1
            rsi_signal = 'OVERSOLD'
        elif rsi > 70:
            sell_signals += 1
            rsi_signal = 'OVERBOUGHT'
        else:
            rsi_signal = 'NEUTRAL'
        
        # Análise MACD
        if float(macd) > float(macd_signal):
            buy_signals += 1
            macd_trend = 'BULLISH'
        else:
            sell_signals += 1
            macd_trend = 'BEARISH'
        
        # Análise Bollinger Bands
        if bb_position == 'BELOW_LOWER':
            buy_signals += 1
        elif bb_position == 'ABOVE_UPPER':
            sell_signals += 1
        
        # Decisão final
        if buy_signals > sell_signals:
            recommendation = 'BUY'
            confidence = buy_signals / 3
        elif sell_signals > buy_signals:
            recommendation = 'SELL'
            confidence = sell_signals / 3
        else:
            recommendation = 'HOLD'
            confidence = 0.5
        
        # Resultado
        result = {
            'pair': pair,
            'price': current_price,
            'rsi': rsi,
            'rsi_signal': rsi_signal,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_trend': macd_trend,
            'bb_position': bb_position,
            'recommendation': recommendation,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
        
        # Exibir resultados
        print(f"💰 Preço: {current_price:.5f}")
        print(f"📊 RSI: {rsi:.1f} ({rsi_signal})")
        print(f"📈 MACD: {macd:.5f} ({macd_trend})")
        print(f"🎯 Bollinger: {bb_position}")
        print(f"🚀 Recomendação: {recommendation}")
        print(f"🔥 Confiança: {confidence:.1%}")
        
        return result
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        return {'error': str(e)}

def main():
    """Função principal"""
    print("\n" + "="*60)
    print("🎯 ANÁLISE RÁPIDA DE FOREX")
    print("="*60)
    
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
    results = {}
    
    # Analisar cada par
    for pair in pairs:
        results[pair] = analyze_pair(pair)
    
    # Resumo
    print("\n" + "="*60)
    print("📈 RESUMO EXECUTIVO")
    print("="*60)
    
    opportunities = []
    
    for pair, result in results.items():
        if 'error' not in result:
            print(f"\n💎 {pair}:")
            print(f"   Preço: {result['price']:.5f}")
            print(f"   RSI: {result['rsi']:.1f} ({result['rsi_signal']})")
            print(f"   Tendência: {result['macd_trend']}")
            print(f"   Ação: {result['recommendation']} ({result['confidence']:.1%})")
            
            if result['recommendation'] != 'HOLD':
                opportunities.append(result)
        else:
            print(f"\n❌ {pair}: {result['error']}")
    
    # Melhores oportunidades
    if opportunities:
        print("\n" + "="*60)
        print("🏆 MELHORES OPORTUNIDADES")
        print("="*60)
        
        # Ordenar por confiança
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        for i, opp in enumerate(opportunities, 1):
            print(f"\n{i}. {opp['pair']} - {opp['recommendation']}")
            print(f"   Confiança: {opp['confidence']:.1%}")
            print(f"   RSI: {opp['rsi']:.1f} ({opp['rsi_signal']})")
            print(f"   Tendência MACD: {opp['macd_trend']}")
    else:
        print("\n📊 Mercado em consolidação - aguardar melhores oportunidades")
    
    print("\n" + "="*60)
    print("✅ ANÁLISE CONCLUÍDA")
    print("="*60)
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print("\n🎉 Análise executada com sucesso!")
    except Exception as e:
        print(f"\n❌ Erro: {e}")