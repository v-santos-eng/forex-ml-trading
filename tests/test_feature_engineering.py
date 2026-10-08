#!/usr/bin/env python3
"""
Script de teste para identificar problemas no feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import traceback

# Carregar dados de teste
def load_test_data():
    data_path = Path("extracted_fx_data/EURUSD")
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        print("Nenhum arquivo encontrado")
        return None
    
    # Carregar primeiro arquivo
    data = pd.read_csv(
        csv_files[0], 
        sep=';', 
        header=None,
        names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    
    # Converter DateTime
    data['DateTime'] = pd.to_datetime(data['DateTime'], format='%Y%m%d %H%M%S')
    data.set_index('DateTime', inplace=True)
    
    # Garantir tipos numéricos
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Remover linhas com valores nulos
    data = data.dropna()
    
    # Pegar apenas uma amostra pequena para teste
    data = data.head(1000)
    
    print(f"Dados carregados: {len(data)} registros")
    print(f"Colunas: {list(data.columns)}")
    print(f"Primeiras linhas:")
    print(data.head())
    
    return data

# Testar feature engineering
def test_feature_engineering():
    try:
        from advanced_feature_engineering import AdvancedFeatureEngineer
        
        data = load_test_data()
        if data is None:
            return
        
        print("\n=== TESTANDO FEATURE ENGINEERING ===")
        
        # Criar feature engineer
        fe = AdvancedFeatureEngineer()
        
        print("\n1. Testando indicadores técnicos...")
        try:
            tech_features = fe.create_technical_indicators(data)
            print(f"✅ Indicadores técnicos criados: {tech_features.shape}")
        except Exception as e:
            print(f"❌ Erro nos indicadores técnicos: {e}")
            traceback.print_exc()
        
        print("\n2. Testando price action features...")
        try:
            price_features = fe.create_price_action_features(data)
            print(f"✅ Price action features criadas: {price_features.shape}")
        except Exception as e:
            print(f"❌ Erro nas price action features: {e}")
            traceback.print_exc()
        
        print("\n3. Testando microestrutura features...")
        try:
            micro_features = fe.create_market_microstructure_features(data)
            print(f"✅ Microestrutura features criadas: {micro_features.shape}")
        except Exception as e:
            print(f"❌ Erro nas microestrutura features: {e}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"Erro geral: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_engineering()