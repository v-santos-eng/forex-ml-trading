#!/usr/bin/env python3
"""
Script de Inicialização do Sistema de Trading
Versão simplificada para operação direta
"""

import sys
import os
from pathlib import Path

# Adicionar diretórios ao path
current_dir = Path(__file__).parent
scripts_dir = current_dir / 'scripts'
src_dir = current_dir / 'src'

sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(src_dir))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class SimpleTradingSystem:
    """Sistema de Trading Simplificado para Operação Direta"""
    
    def __init__(self):
        self.models_dir = current_dir / 'models'
        self.available_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
        print("\n" + "="*50)
        print("🚀 SISTEMA DE TRADING FOREX ML ATIVO")
        print("="*50)
        print(f"📁 Modelos disponíveis: {self.available_pairs}")
        print(f"📊 Diretório de modelos: {self.models_dir}")
        
    def get_forex_data(self, pair: str, period: str = '1mo') -> pd.DataFrame:
        """Obter dados forex em tempo real"""
        try:
            # Converter par para formato Yahoo Finance
            if pair == 'EURUSD':
                symbol = 'EURUSD=X'
            elif pair == 'GBPUSD':
                symbol = 'GBPUSD=X'
            elif pair == 'USDJPY':
                symbol = 'USDJPY=X'
            else:
                symbol = f'{pair}=X'
                
            print(f"📈 Obtendo dados para {pair}...")
            data = yf.download(symbol, period=period, interval='1h')
            
            if data.empty:
                print(f"❌ Não foi possível obter dados para {pair}")
                return None
                
            print(f"✅ Dados obtidos: {len(data)} registros")
            return data
            
        except Exception as e:
            print(f"❌ Erro ao obter dados: {e}")
            return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcular indicadores técnicos básicos"""
        df = data.copy()
        
        # Garantir que temos apenas uma coluna Close
        if isinstance(df['Close'], pd.DataFrame):
            df['Close'] = df['Close'].iloc[:, 0]
        
        try:
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Médias móveis
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # Bollinger Bands - versão mais robusta
            bb_period = 20
            bb_std_dev = 2
            
            # Calcular componentes separadamente
            close_series = df['Close']
            bb_middle = close_series.rolling(window=bb_period).mean()
            bb_std = close_series.rolling(window=bb_period).std()
            
            # Atribuir valores
            df.loc[:, 'BB_Middle'] = bb_middle
            df.loc[:, 'BB_Upper'] = bb_middle + (bb_std * bb_std_dev)
            df.loc[:, 'BB_Lower'] = bb_middle - (bb_std * bb_std_dev)
            
        except Exception as e:
            print(f"⚠️ Erro no cálculo de indicadores: {e}")
            # Valores padrão em caso de erro
            df['RSI'] = 50.0
            df['SMA_20'] = df['Close']
            df['SMA_50'] = df['Close']
            df['MACD'] = 0.0
            df['MACD_Signal'] = 0.0
            df['BB_Middle'] = df['Close']
            df['BB_Upper'] = df['Close'] * 1.02
            df['BB_Lower'] = df['Close'] * 0.98
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """Gerar sinais de trading baseados em indicadores"""
        latest = data.iloc[-1]
        
        signals = {
            'timestamp': datetime.now(),
            'pair': 'N/A',
            'price': latest['Close'],
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'macd_signal': latest['MACD_Signal'],
            'bb_position': 'MIDDLE'
        }
        
        # Determinar posição nas Bollinger Bands
        if latest['Close'] > latest['BB_Upper']:
            signals['bb_position'] = 'ABOVE_UPPER'
        elif latest['Close'] < latest['BB_Lower']:
            signals['bb_position'] = 'BELOW_LOWER'
            
        # Sinal de compra/venda
        buy_signals = 0
        sell_signals = 0
        
        # RSI
        if latest['RSI'] < 30:
            buy_signals += 1
        elif latest['RSI'] > 70:
            sell_signals += 1
            
        # MACD
        if latest['MACD'] > latest['MACD_Signal']:
            buy_signals += 1
        else:
            sell_signals += 1
            
        # Bollinger Bands
        if signals['bb_position'] == 'BELOW_LOWER':
            buy_signals += 1
        elif signals['bb_position'] == 'ABOVE_UPPER':
            sell_signals += 1
            
        # Decisão final
        if buy_signals > sell_signals:
            signals['recommendation'] = 'BUY'
            signals['confidence'] = buy_signals / 3
        elif sell_signals > buy_signals:
            signals['recommendation'] = 'SELL'
            signals['confidence'] = sell_signals / 3
        else:
            signals['recommendation'] = 'HOLD'
            signals['confidence'] = 0.5
            
        return signals
    
    def analyze_pair(self, pair: str) -> Dict:
        """Análise completa de um par de moedas"""
        print(f"\n🔍 ANALISANDO {pair}")
        print("-" * 30)
        
        # Obter dados
        data = self.get_forex_data(pair)
        if data is None:
            return {'error': 'Não foi possível obter dados'}
            
        # Calcular indicadores
        data_with_indicators = self.calculate_technical_indicators(data)
        
        # Gerar sinais
        signals = self.generate_signals(data_with_indicators)
        signals['pair'] = pair
        
        # Exibir resultados
        print(f"💰 Preço atual: {signals['price']:.5f}")
        print(f"📊 RSI: {signals['rsi']:.2f}")
        print(f"📈 MACD: {signals['macd']:.5f}")
        print(f"🎯 Recomendação: {signals['recommendation']}")
        print(f"🔥 Confiança: {signals['confidence']:.1%}")
        
        return signals
    
    def run_analysis(self, pair: str = None):
        """Executar análise para um ou todos os pares"""
        if pair:
            if pair in self.available_pairs:
                return self.analyze_pair(pair)
            else:
                print(f"❌ Par {pair} não disponível. Pares disponíveis: {self.available_pairs}")
                return None
        else:
            # Analisar todos os pares
            results = {}
            for p in self.available_pairs:
                results[p] = self.analyze_pair(p)
            return results
    
    def show_menu(self):
        """Exibir menu interativo"""
        while True:
            print("\n" + "="*50)
            print("📋 MENU DE OPERAÇÕES")
            print("="*50)
            print("1. Analisar EURUSD")
            print("2. Analisar GBPUSD")
            print("3. Analisar USDJPY")
            print("4. Analisar todos os pares")
            print("5. Sair")
            
            choice = input("\n👉 Escolha uma opção (1-5): ").strip()
            
            if choice == '1':
                self.run_analysis('EURUSD')
            elif choice == '2':
                self.run_analysis('GBPUSD')
            elif choice == '3':
                self.run_analysis('USDJPY')
            elif choice == '4':
                self.run_analysis()
            elif choice == '5':
                print("\n👋 Encerrando sistema...")
                break
            else:
                print("❌ Opção inválida!")

def main():
    """Função principal"""
    try:
        system = SimpleTradingSystem()
        
        print("\n🎯 Sistema inicializado com sucesso!")
        print("\n💡 Comandos disponíveis:")
        print("   - system.run_analysis('EURUSD')")
        print("   - system.run_analysis('GBPUSD')")
        print("   - system.run_analysis('USDJPY')")
        print("   - system.run_analysis()  # Todos os pares")
        print("   - system.show_menu()     # Menu interativo")
        
        # Disponibilizar sistema globalmente
        globals()['system'] = system
        
        return system
        
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        return None

if __name__ == "__main__":
    system = main()
    if system:
        print("\n🚀 Iniciando menu interativo...")
        system.show_menu()