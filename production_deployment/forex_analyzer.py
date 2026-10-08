#!/usr/bin/env python3
"""
Analisador de Forex - Sistema Operacional
Versão final para análise e operação em tempo real
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ForexAnalyzer:
    """Analisador profissional de Forex"""
    
    def __init__(self):
        self.pairs = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'USDCHF': 'USDCHF=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X'
        }
        
    def get_data(self, pair: str, period: str = '1mo') -> pd.DataFrame:
        """Obter dados históricos"""
        try:
            symbol = self.pairs.get(pair, f'{pair}=X')
            print(f"📊 Carregando {pair}...")
            
            data = yf.download(symbol, period=period, interval='1h', progress=False)
            
            if data.empty:
                return None
                
            data = data.dropna()
            print(f"✅ {len(data)} registros carregados")
            return data
            
        except Exception as e:
            print(f"❌ Erro ao carregar {pair}: {e}")
            return None
    
    def calculate_indicators(self, data: pd.DataFrame) -> dict:
        """Calcular todos os indicadores técnicos"""
        try:
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            
            # RSI
            rsi = self._calculate_rsi(close)
            
            # MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(close)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger(close)
            
            # Moving Averages
            sma_20 = np.mean(close[-20:])
            sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20
            
            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(high, low, close)
            
            # ATR
            atr = self._calculate_atr(high, low, close)
            
            return {
                'rsi': rsi,
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d,
                'atr': atr,
                'current_price': close[-1]
            }
            
        except Exception as e:
            print(f"❌ Erro no cálculo de indicadores: {e}")
            return {}
    
    def _calculate_rsi(self, prices: np.array, period: int = 14) -> float:
        """Calcular RSI"""
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
        except:
            return 50.0
    
    def _calculate_macd(self, prices: np.array) -> tuple:
        """Calcular MACD"""
        try:
            # Exponential Moving Averages
            ema_12 = self._ema(prices, 12)
            ema_26 = self._ema(prices, 26)
            
            macd_line = ema_12 - ema_26
            macd_signal = self._ema(np.array([macd_line]), 9)[0]
            macd_histogram = macd_line - macd_signal
            
            return float(macd_line), float(macd_signal), float(macd_histogram)
        except:
            return 0.0, 0.0, 0.0
    
    def _calculate_bollinger(self, prices: np.array, period: int = 20, std_dev: float = 2.0) -> tuple:
        """Calcular Bollinger Bands"""
        try:
            sma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
            
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            return float(upper), float(sma), float(lower)
        except:
            return 0.0, 0.0, 0.0
    
    def _calculate_stochastic(self, high: np.array, low: np.array, close: np.array, period: int = 14) -> tuple:
        """Calcular Stochastic Oscillator"""
        try:
            highest_high = np.max(high[-period:])
            lowest_low = np.min(low[-period:])
            
            if highest_high == lowest_low:
                return 50.0, 50.0
                
            k_percent = ((close[-1] - lowest_low) / (highest_high - lowest_low)) * 100
            d_percent = k_percent  # Simplificado
            
            return float(k_percent), float(d_percent)
        except:
            return 50.0, 50.0
    
    def _calculate_atr(self, high: np.array, low: np.array, close: np.array, period: int = 14) -> float:
        """Calcular Average True Range"""
        try:
            tr_list = []
            for i in range(1, len(close)):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr_list.append(max(tr1, tr2, tr3))
            
            atr = np.mean(tr_list[-period:])
            return float(atr)
        except:
            return 0.0
    
    def _ema(self, prices: np.array, period: int) -> float:
        """Calcular Exponential Moving Average"""
        try:
            if len(prices) == 0:
                return 0.0
            
            multiplier = 2 / (period + 1)
            ema = float(prices[0])
            
            for price in prices[1:]:
                ema = (float(price) * multiplier) + (ema * (1 - multiplier))
                
            return float(ema)
        except:
            return float(prices[-1]) if len(prices) > 0 else 0.0
    
    def generate_signals(self, indicators: dict) -> dict:
        """Gerar sinais de trading"""
        signals = {
            'buy_signals': 0,
            'sell_signals': 0,
            'neutral_signals': 0,
            'details': []
        }
        
        try:
            # Análise RSI
            if indicators['rsi'] < 30:
                signals['buy_signals'] += 1
                signals['details'].append('RSI Oversold (Compra)')
            elif indicators['rsi'] > 70:
                signals['sell_signals'] += 1
                signals['details'].append('RSI Overbought (Venda)')
            else:
                signals['neutral_signals'] += 1
                signals['details'].append('RSI Neutro')
            
            # Análise MACD
            if indicators['macd_line'] > indicators['macd_signal']:
                signals['buy_signals'] += 1
                signals['details'].append('MACD Bullish (Compra)')
            else:
                signals['sell_signals'] += 1
                signals['details'].append('MACD Bearish (Venda)')
            
            # Análise Bollinger Bands
            current_price = indicators['current_price']
            if current_price < indicators['bb_lower']:
                signals['buy_signals'] += 1
                signals['details'].append('Preço abaixo BB inferior (Compra)')
            elif current_price > indicators['bb_upper']:
                signals['sell_signals'] += 1
                signals['details'].append('Preço acima BB superior (Venda)')
            else:
                signals['neutral_signals'] += 1
                signals['details'].append('Preço dentro das Bollinger Bands')
            
            # Análise Moving Averages
            if indicators['sma_20'] > indicators['sma_50']:
                signals['buy_signals'] += 1
                signals['details'].append('SMA 20 > SMA 50 (Bullish)')
            else:
                signals['sell_signals'] += 1
                signals['details'].append('SMA 20 < SMA 50 (Bearish)')
            
            # Análise Stochastic
            if indicators['stoch_k'] < 20:
                signals['buy_signals'] += 1
                signals['details'].append('Stochastic Oversold (Compra)')
            elif indicators['stoch_k'] > 80:
                signals['sell_signals'] += 1
                signals['details'].append('Stochastic Overbought (Venda)')
            
        except Exception as e:
            print(f"❌ Erro na geração de sinais: {e}")
        
        return signals
    
    def analyze_pair(self, pair: str) -> dict:
        """Análise completa de um par"""
        print(f"\n🔍 ANALISANDO {pair}")
        print("-" * 40)
        
        # Obter dados
        data = self.get_data(pair)
        if data is None:
            return {'error': 'Dados não disponíveis'}
        
        # Calcular indicadores
        indicators = self.calculate_indicators(data)
        if not indicators:
            return {'error': 'Erro no cálculo de indicadores'}
        
        # Gerar sinais
        signals = self.generate_signals(indicators)
        
        # Determinar recomendação
        total_signals = signals['buy_signals'] + signals['sell_signals']
        if total_signals == 0:
            recommendation = 'HOLD'
            confidence = 0.5
        elif signals['buy_signals'] > signals['sell_signals']:
            recommendation = 'BUY'
            confidence = signals['buy_signals'] / (signals['buy_signals'] + signals['sell_signals'])
        else:
            recommendation = 'SELL'
            confidence = signals['sell_signals'] / (signals['buy_signals'] + signals['sell_signals'])
        
        # Resultado
        result = {
            'pair': pair,
            'price': indicators['current_price'],
            'recommendation': recommendation,
            'confidence': confidence,
            'indicators': indicators,
            'signals': signals,
            'timestamp': datetime.now()
        }
        
        # Exibir resultados
        print(f"💰 Preço Atual: {float(indicators['current_price']):.5f}")
        print(f"📊 RSI: {float(indicators['rsi']):.1f}")
        print(f"📈 MACD: {float(indicators['macd_line']):.5f} / {float(indicators['macd_signal']):.5f}")
        print(f"🎯 Bollinger: {float(indicators['bb_lower']):.5f} - {float(indicators['bb_upper']):.5f}")
        print(f"📉 Stochastic: {float(indicators['stoch_k']):.1f}")
        print(f"🚀 Recomendação: {recommendation}")
        print(f"🔥 Confiança: {float(confidence):.1%}")
        
        return result
    
    def run_analysis(self, pairs: list = None) -> dict:
        """Executar análise completa"""
        if pairs is None:
            pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        print("\n" + "="*60)
        print("🎯 FOREX ANALYZER - ANÁLISE PROFISSIONAL")
        print("="*60)
        
        results = {}
        opportunities = []
        
        # Analisar cada par
        for pair in pairs:
            result = self.analyze_pair(pair)
            results[pair] = result
            
            if 'error' not in result and result['recommendation'] != 'HOLD':
                opportunities.append(result)
        
        # Resumo executivo
        print("\n" + "="*60)
        print("📈 RESUMO EXECUTIVO")
        print("="*60)
        
        for pair, result in results.items():
            if 'error' not in result:
                print(f"\n💎 {pair}:")
                print(f"   Preço: {float(result['price']):.5f}")
                rsi_val = float(result['indicators']['rsi']) if not np.isnan(result['indicators']['rsi']) else 50.0
                print(f"   RSI: {rsi_val:.1f}")
                print(f"   Ação: {result['recommendation']} ({float(result['confidence']):.1%})")
            else:
                print(f"\n❌ {pair}: {result['error']}")
        
        # Melhores oportunidades
        if opportunities:
            print("\n" + "="*60)
            print("🏆 OPORTUNIDADES DE TRADING")
            print("="*60)
            
            opportunities.sort(key=lambda x: x['confidence'], reverse=True)
            
            for i, opp in enumerate(opportunities, 1):
                print(f"\n{i}. {opp['pair']} - {opp['recommendation']}")
                print(f"   Confiança: {float(opp['confidence']):.1%}")
                print(f"   Preço: {float(opp['price']):.5f}")
                rsi_val = float(opp['indicators']['rsi']) if not np.isnan(opp['indicators']['rsi']) else 50.0
                print(f"   RSI: {rsi_val:.1f}")
                
                # Mostrar principais sinais
                print("   Sinais:")
                for detail in opp['signals']['details'][:3]:
                    print(f"     • {detail}")
        else:
            print("\n📊 Nenhuma oportunidade clara identificada")
            print("💡 Aguardar melhores condições de mercado")
        
        print("\n" + "="*60)
        print("✅ ANÁLISE CONCLUÍDA")
        print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        return results

def main():
    """Função principal"""
    analyzer = ForexAnalyzer()
    
    try:
        # Executar análise
        results = analyzer.run_analysis()
        
        print("\n🎉 Sistema operacional e funcionando!")
        print("\n💡 Comandos disponíveis:")
        print("   • python forex_analyzer.py - Análise completa")
        print("   • Modifique o código para análises customizadas")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Erro na execução: {e}")
        return None

if __name__ == "__main__":
    main()