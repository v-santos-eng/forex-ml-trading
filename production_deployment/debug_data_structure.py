#!/usr/bin/env python3
"""
Script de debug para verificar a estrutura dos dados MT5
"""

import sys
import os
import pandas as pd
import MetaTrader5 as mt5

# Adicionar diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt5_connector import MT5Connector
from mt5_data_provider import MT5DataProvider

def debug_data_structure():
    """Debug da estrutura de dados"""
    print("=== DEBUG: Estrutura de Dados MT5 ===")
    
    # Conectar ao MT5
    connector = MT5Connector()
    if not connector.connect():
        print("❌ Falha na conexão MT5")
        return
    
    data_provider = MT5DataProvider(connector)
    
    # Testar com EURUSD
    symbol = "EURUSD"
    print(f"\n📊 Testando símbolo: {symbol}")
    
    # Obter dados históricos
    rates = data_provider.get_historical_data(symbol, mt5.TIMEFRAME_M15, 100)
    
    if rates is not None:
        print(f"✓ Dados obtidos: {len(rates)} registros")
        print(f"✓ Colunas: {list(rates.columns)}")
        print(f"✓ Shape: {rates.shape}")
        print(f"✓ Index type: {type(rates.index)}")
        
        # Mostrar primeiras linhas
        print("\n📋 Primeiras 3 linhas:")
        print(rates.head(3))
        
        # Testar conversão para formato esperado
        print("\n🔄 Testando conversão de formato...")
        try:
            # Resetar index para ter coluna time
            df_reset = rates.reset_index()
            print(f"✓ Após reset_index: {list(df_reset.columns)}")
            print(f"✓ Shape após reset: {df_reset.shape}")
            
            # Tentar aplicar as colunas esperadas
            expected_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            print(f"✓ Colunas esperadas: {expected_columns}")
            print(f"✓ Número de colunas esperadas: {len(expected_columns)}")
            print(f"✓ Número de colunas atuais: {len(df_reset.columns)}")
            
            # Verificar se há colunas extras
            if len(df_reset.columns) != len(expected_columns):
                print(f"⚠️  PROBLEMA: Número de colunas não coincide!")
                print(f"   Colunas atuais: {list(df_reset.columns)}")
                print(f"   Diferença: {len(df_reset.columns) - len(expected_columns)} colunas")
                
                # Mostrar tipos de dados
                print(f"\n📊 Tipos de dados:")
                for col in df_reset.columns:
                    print(f"   {col}: {df_reset[col].dtype}")
            
        except Exception as e:
            print(f"❌ Erro na conversão: {e}")
    
    else:
        print("❌ Falha ao obter dados históricos")
    
    # Testar dados brutos do MT5
    print(f"\n🔍 Testando dados brutos do MT5...")
    try:
        raw_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 10)
        if raw_rates is not None:
            print(f"✓ Dados brutos obtidos: {len(raw_rates)} registros")
            
            # Converter para DataFrame sem modificações
            raw_df = pd.DataFrame(raw_rates)
            print(f"✓ Colunas brutas: {list(raw_df.columns)}")
            print(f"✓ Shape bruto: {raw_df.shape}")
            
            # Mostrar primeira linha
            print(f"\n📋 Primeira linha bruta:")
            print(raw_df.iloc[0])
            
        else:
            print("❌ Falha ao obter dados brutos")
            
    except Exception as e:
        print(f"❌ Erro ao obter dados brutos: {e}")
    
    connector.disconnect()
    print("\n✅ Debug concluído")

if __name__ == "__main__":
    debug_data_structure()