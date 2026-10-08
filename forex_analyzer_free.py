#!/usr/bin/env python3
"""
Forex Analyzer - Versão Gratuita (Sem API)
Análise de forex usando bibliotecas gratuitas de Python

Este módulo fornece análise técnica e fundamental básica
sem depender de APIs pagas como Anthropic.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib não instalado. Usando indicadores básicos.")

class ForexAnalyzerFree:
    """
    Analisador de Forex gratuito usando bibliotecas open-source
    """
    
    def __init__(self):
        self.major_pairs = [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X',
            'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X'
        ]
        
    def get_forex_data(self, pair, period='1mo', interval='1d'):
        """
        Obtém dados históricos de forex do Yahoo Finance
        
        Args:
            pair (str): Par de moedas (ex: 'EURUSD=X')
            period (str): Período ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Intervalo ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            pd.DataFrame: Dados históricos
        """
        try:
            ticker = yf.Ticker(pair)
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            print(f"Erro ao obter dados para {pair}: {e}")
            return None
    
    def calculate_sma(self, data, window):
        """Calcula Média Móvel Simples"""
        return data['Close'].rolling(window=window).mean()
    
    def calculate_ema(self, data, window):
        """Calcula Média Móvel Exponencial"""
        return data['Close'].ewm(span=window).mean()
    
    def calculate_rsi(self, data, window=14):
        """Calcula RSI (Relative Strength Index)"""
        if TALIB_AVAILABLE:
            rsi_values = talib.RSI(data['Close'].values, timeperiod=window)
            return pd.Series(rsi_values, index=data.index)
        else:
            # Implementação manual do RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calcula MACD"""
        if TALIB_AVAILABLE:
            macd, signal_line, histogram = talib.MACD(data['Close'].values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return pd.DataFrame({
                'MACD': macd,
                'Signal': signal_line,
                'Histogram': histogram
            }, index=data.index)
        else:
            # Implementação manual do MACD
            ema_fast = data['Close'].ewm(span=fast).mean()
            ema_slow = data['Close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            return pd.DataFrame({
                'MACD': macd,
                'Signal': signal_line,
                'Histogram': histogram
            })
    
    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """Calcula Bandas de Bollinger"""
        sma = self.calculate_sma(data, window)
        std = data['Close'].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return pd.DataFrame({
            'Upper': upper_band,
            'Middle': sma,
            'Lower': lower_band
        })
    
    def analyze_pair(self, pair, period='3mo'):
        """
        Realiza análise técnica completa de um par de moedas
        
        Args:
            pair (str): Par de moedas
            period (str): Período de análise
        
        Returns:
            dict: Resultados da análise
        """
        print(f"\nAnalisando {pair}...")
        
        # Obter dados
        data = self.get_forex_data(pair, period=period)
        if data is None or data.empty:
            return None
        
        # Calcular indicadores
        current_price = data['Close'].iloc[-1]
        sma_20 = self.calculate_sma(data, 20).iloc[-1]
        sma_50 = self.calculate_sma(data, 50).iloc[-1]
        ema_12 = self.calculate_ema(data, 12).iloc[-1]
        rsi = self.calculate_rsi(data).iloc[-1]
        macd_data = self.calculate_macd(data)
        bollinger = self.calculate_bollinger_bands(data)
        
        # Análise de tendência
        trend = "LATERAL"
        if current_price > sma_20 > sma_50:
            trend = "ALTA"
        elif current_price < sma_20 < sma_50:
            trend = "BAIXA"
        
        # Sinal RSI
        rsi_signal = "NEUTRO"
        if rsi > 70:
            rsi_signal = "SOBRECOMPRADO"
        elif rsi < 30:
            rsi_signal = "SOBREVENDIDO"
        
        # Sinal MACD
        macd_signal = "NEUTRO"
        if not macd_data.empty:
            macd_current = macd_data['MACD'].iloc[-1]
            signal_current = macd_data['Signal'].iloc[-1]
            if macd_current > signal_current:
                macd_signal = "ALTA"
            else:
                macd_signal = "BAIXA"
        
        # Posição nas Bandas de Bollinger
        bb_position = "MEIO"
        if not bollinger.empty:
            upper = bollinger['Upper'].iloc[-1]
            lower = bollinger['Lower'].iloc[-1]
            if current_price > upper:
                bb_position = "ACIMA BANDA SUPERIOR"
            elif current_price < lower:
                bb_position = "ABAIXO BANDA INFERIOR"
        
        # Recomendação geral
        signals = []
        if trend == "ALTA":
            signals.append("COMPRA")
        elif trend == "BAIXA":
            signals.append("VENDA")
        
        if rsi_signal == "SOBREVENDIDO":
            signals.append("COMPRA")
        elif rsi_signal == "SOBRECOMPRADO":
            signals.append("VENDA")
        
        if macd_signal == "ALTA":
            signals.append("COMPRA")
        elif macd_signal == "BAIXA":
            signals.append("VENDA")
        
        # Determinar recomendação final
        buy_signals = signals.count("COMPRA")
        sell_signals = signals.count("VENDA")
        
        if buy_signals > sell_signals:
            recommendation = "COMPRA"
        elif sell_signals > buy_signals:
            recommendation = "VENDA"
        else:
            recommendation = "NEUTRO"
        
        # Calcular suporte e resistência básicos
        recent_data = data.tail(20)
        support = recent_data['Low'].min()
        resistance = recent_data['High'].max()
        
        return {
            'pair': pair,
            'current_price': round(current_price, 5),
            'trend': trend,
            'rsi': round(rsi, 2) if not pd.isna(rsi) else None,
            'rsi_signal': rsi_signal,
            'macd_signal': macd_signal,
            'bollinger_position': bb_position,
            'support': round(support, 5),
            'resistance': round(resistance, 5),
            'sma_20': round(sma_20, 5),
            'sma_50': round(sma_50, 5),
            'recommendation': recommendation,
            'confidence': round((max(buy_signals, sell_signals) / len(signals)) * 100, 1) if signals else 0
        }
    
    def market_overview(self):
        """
        Fornece uma visão geral do mercado forex
        """
        print("\n" + "="*60)
        print("VISÃO GERAL DO MERCADO FOREX")
        print("="*60)
        
        results = []
        for pair in self.major_pairs:
            analysis = self.analyze_pair(pair, period='1mo')
            if analysis:
                results.append(analysis)
        
        # Resumo por recomendação
        buy_count = sum(1 for r in results if r['recommendation'] == 'COMPRA')
        sell_count = sum(1 for r in results if r['recommendation'] == 'VENDA')
        neutral_count = sum(1 for r in results if r['recommendation'] == 'NEUTRO')
        
        print(f"\nResumo das Recomendações:")
        print(f"COMPRA: {buy_count} pares")
        print(f"VENDA: {sell_count} pares")
        print(f"NEUTRO: {neutral_count} pares")
        
        # Sentimento geral do mercado
        if buy_count > sell_count:
            market_sentiment = "OTIMISTA"
        elif sell_count > buy_count:
            market_sentiment = "PESSIMISTA"
        else:
            market_sentiment = "NEUTRO"
        
        print(f"\nSentimento Geral do Mercado: {market_sentiment}")
        
        return results
    
    def detailed_analysis(self, pair):
        """
        Análise detalhada de um par específico
        """
        print(f"\n" + "="*60)
        print(f"ANÁLISE DETALHADA - {pair}")
        print("="*60)
        
        analysis = self.analyze_pair(pair, period='6mo')
        if not analysis:
            print(f"Não foi possível analisar {pair}")
            return
        
        print(f"\nPreço Atual: {analysis['current_price']}")
        print(f"Tendência: {analysis['trend']}")
        print(f"\nIndicadores Técnicos:")
        print(f"  RSI (14): {analysis['rsi']} - {analysis['rsi_signal']}")
        print(f"  MACD: {analysis['macd_signal']}")
        print(f"  SMA 20: {analysis['sma_20']}")
        print(f"  SMA 50: {analysis['sma_50']}")
        print(f"  Bollinger: {analysis['bollinger_position']}")
        
        print(f"\nNíveis de Suporte e Resistência:")
        print(f"  Suporte: {analysis['support']}")
        print(f"  Resistência: {analysis['resistance']}")
        
        print(f"\nRecomendação: {analysis['recommendation']}")
        print(f"Confiança: {analysis['confidence']}%")
        
        # Sugestões de trading
        print(f"\nSugestões de Trading:")
        if analysis['recommendation'] == 'COMPRA':
            stop_loss = analysis['support'] * 0.999  # 0.1% abaixo do suporte
            take_profit = analysis['resistance'] * 1.001  # 0.1% acima da resistência
            print(f"  Entrada: {analysis['current_price']} (COMPRA)")
            print(f"  Stop Loss: {round(stop_loss, 5)}")
            print(f"  Take Profit: {round(take_profit, 5)}")
        elif analysis['recommendation'] == 'VENDA':
            stop_loss = analysis['resistance'] * 1.001  # 0.1% acima da resistência
            take_profit = analysis['support'] * 0.999  # 0.1% abaixo do suporte
            print(f"  Entrada: {analysis['current_price']} (VENDA)")
            print(f"  Stop Loss: {round(stop_loss, 5)}")
            print(f"  Take Profit: {round(take_profit, 5)}")
        else:
            print(f"  Aguardar sinal mais claro")
        
        return analysis

def main():
    """
    Função principal - demonstração do analisador gratuito
    """
    print("Forex Analyzer - Versão Gratuita")
    print("Análise técnica sem APIs pagas")
    print("="*50)
    
    analyzer = ForexAnalyzerFree()
    
    # Visão geral do mercado
    analyzer.market_overview()
    
    # Análise detalhada do EUR/USD
    analyzer.detailed_analysis('EURUSD=X')
    
    # Análise detalhada do GBP/USD
    analyzer.detailed_analysis('GBPUSD=X')
    
    print("\n" + "="*60)
    print("ANÁLISE CONCLUÍDA")
    print("="*60)
    print("\nRecursos utilizados:")
    print("- Yahoo Finance (dados gratuitos)")
    print("- Pandas/Numpy (análise de dados)")
    if TALIB_AVAILABLE:
        print("- TA-Lib (indicadores técnicos)")
    else:
        print("- Indicadores técnicos implementados manualmente")
    
    print("\nVantagens desta abordagem:")
    print("- Totalmente gratuita")
    print("- Sem limites de API")
    print("- Código open-source")
    print("- Personalizável")
    
    print("\nLimitações:")
    print("- Sem análise fundamental automatizada")
    print("- Sem processamento de linguagem natural")
    print("- Indicadores técnicos básicos")
    print("- Dados limitados do Yahoo Finance")
    
    print("\nDISCLAIMER:")
    print("Este é um sistema de apoio à decisão.")
    print("Não constitui aconselhamento financeiro.")
    print("Sempre faça sua própria pesquisa antes de investir.")

if __name__ == "__main__":
    main()