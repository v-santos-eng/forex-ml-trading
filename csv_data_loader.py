#!/usr/bin/env python3
"""
CSV Data Loader para ML Forex
Carrega dados históricos de 1 minuto de arquivos CSV locais
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import glob

class CSVForexDataLoader:
    """
    Carregador de dados de forex a partir de arquivos CSV
    Suporta dados de 1 minuto com otimização para performance
    """
    
    def __init__(self, data_directory: str):
        """
        Inicializa o carregador de dados CSV
        
        Args:
            data_directory: Diretório contendo os arquivos CSV
        """
        self.data_directory = data_directory
        self.available_pairs = self._scan_available_pairs()
        
    def _scan_available_pairs(self) -> List[str]:
        """
        Escaneia o diretório para encontrar pares disponíveis
        """
        if not os.path.exists(self.data_directory):
            print(f"Diretório não encontrado: {self.data_directory}")
            return []
            
        csv_files = glob.glob(os.path.join(self.data_directory, "*.csv"))
        pairs = []
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            # Extrair nome do par do arquivo (assumindo formato EURUSD_1min.csv ou similar)
            pair_name = filename.replace('.csv', '').replace('_1min', '').replace('_1m', '')
            pairs.append(pair_name)
            
        print(f"Pares encontrados: {pairs}")
        return pairs
    
    def load_forex_data(self, pair: str, period: str = '2y', sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Carrega dados de forex de arquivo CSV
        
        Args:
            pair: Par de moedas (ex: 'EURUSD', 'GBPUSD')
            period: Período dos dados (usado para calcular quantos dados carregar)
            sample_size: Número específico de amostras para carregar (para otimização)
            
        Returns:
            DataFrame com dados OHLCV ou None se não encontrado
        """
        # Procurar arquivo correspondente
        possible_filenames = [
            f"{pair}.csv",
            f"{pair}_1min.csv",
            f"{pair}_1m.csv",
            f"{pair.upper()}.csv",
            f"{pair.upper()}_1min.csv",
            f"{pair.upper()}_1m.csv",
            f"{pair.lower()}.csv",
            f"{pair.lower()}_1min.csv",
            f"{pair.lower()}_1m.csv"
        ]
        
        file_path = None
        for filename in possible_filenames:
            potential_path = os.path.join(self.data_directory, filename)
            if os.path.exists(potential_path):
                file_path = potential_path
                break
                
        if file_path is None:
            print(f"Arquivo não encontrado para {pair}. Tentativas: {possible_filenames}")
            return None
            
        try:
            print(f"Carregando dados de: {file_path}")
            
            # Carregar CSV com otimização
            df = pd.read_csv(file_path)
            
            # Detectar formato das colunas automaticamente
            df = self._standardize_columns(df)
            
            # Converter timestamp para datetime se necessário
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df.set_index('Datetime', inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            # Ordenar por data
            df.sort_index(inplace=True)
            
            # Otimização: selecionar quantidade adequada de dados
            df = self._optimize_data_size(df, period, sample_size)
            
            print(f"Dados carregados: {len(df)} registros de {df.index[0]} a {df.index[-1]}")
            return df
            
        except Exception as e:
            print(f"Erro ao carregar {file_path}: {e}")
            return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Padroniza nomes das colunas para formato esperado
        """
        # Mapeamento de possíveis nomes de colunas
        column_mapping = {
            # Timestamp columns
            'timestamp': 'Datetime',
            'time': 'Datetime',
            'date': 'Datetime',
            'datetime': 'Datetime',
            
            # OHLC columns
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'o': 'Open',
            'h': 'High',
            'l': 'Low', 
            'c': 'Close',
            
            # Volume columns
            'volume': 'Volume',
            'vol': 'Volume',
            'v': 'Volume'
        }
        
        # Renomear colunas
        df_renamed = df.rename(columns={col: column_mapping.get(col.lower(), col) for col in df.columns})
        
        # Garantir que temos pelo menos OHLC
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df_renamed.columns]
        
        if missing_columns:
            print(f"Colunas obrigatórias ausentes: {missing_columns}")
            print(f"Colunas disponíveis: {list(df_renamed.columns)}")
            
        # Adicionar Volume se não existir
        if 'Volume' not in df_renamed.columns:
            df_renamed['Volume'] = 1000  # Volume padrão
            
        return df_renamed
    
    def _optimize_data_size(self, df: pd.DataFrame, period: str, sample_size: Optional[int]) -> pd.DataFrame:
        """
        Otimiza o tamanho dos dados baseado no período e performance
        """
        if sample_size:
            # Usar tamanho específico
            if len(df) > sample_size:
                return df.tail(sample_size)
            return df
            
        # Calcular tamanho baseado no período
        period_mapping = {
            '1mo': 30 * 24 * 60,      # 1 mês em minutos
            '3mo': 90 * 24 * 60,      # 3 meses
            '6mo': 180 * 24 * 60,     # 6 meses
            '1y': 365 * 24 * 60,      # 1 ano
            '2y': 2 * 365 * 24 * 60,  # 2 anos
            '5y': 5 * 365 * 24 * 60   # 5 anos
        }
        
        # Tamanho otimizado para performance vs robustez
        optimal_sizes = {
            '1mo': 50000,   # ~35 dias de dados de 1min
            '3mo': 100000,  # ~70 dias
            '6mo': 200000,  # ~140 dias
            '1y': 300000,   # ~210 dias
            '2y': 500000,   # ~350 dias
            '5y': 700000    # ~490 dias
        }
        
        target_size = optimal_sizes.get(period, 300000)  # Default: ~210 dias
        
        if len(df) > target_size:
            print(f"Otimizando dados: {len(df)} -> {target_size} registros para melhor performance")
            return df.tail(target_size)
            
        return df
    
    def get_data_info(self) -> Dict:
        """
        Retorna informações sobre os dados disponíveis
        """
        info = {
            'directory': self.data_directory,
            'available_pairs': self.available_pairs,
            'total_pairs': len(self.available_pairs)
        }
        
        # Informações detalhadas de cada par
        for pair in self.available_pairs[:5]:  # Limitar a 5 para performance
            try:
                sample_data = self.load_forex_data(pair, sample_size=1000)
                if sample_data is not None:
                    info[f'{pair}_sample_size'] = len(sample_data)
                    info[f'{pair}_date_range'] = f"{sample_data.index[0]} to {sample_data.index[-1]}"
            except:
                continue
                
        return info

# Função de conveniência para integração com sistema existente
def create_csv_data_loader(data_directory: str) -> CSVForexDataLoader:
    """
    Cria um carregador de dados CSV
    """
    return CSVForexDataLoader(data_directory)

# Função para substituir get_forex_data no sistema existente
def get_forex_data_from_csv(pair: str, period: str = '2y', data_directory: str = None) -> Optional[pd.DataFrame]:
    """
    Função drop-in replacement para get_forex_data usando dados CSV
    """
    if data_directory is None:
        data_directory = os.getenv("FX_DATA_DIR", "data/FX-1-Minute-Data")
        
    loader = CSVForexDataLoader(data_directory)
    return loader.load_forex_data(pair, period)

if __name__ == "__main__":
    # Teste do carregador
    data_dir = os.getenv("FX_DATA_DIR", "data/FX-1-Minute-Data")
    loader = CSVForexDataLoader(data_dir)
    
    print("Informações dos dados:")
    info = loader.get_data_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Teste de carregamento
    if loader.available_pairs:
        test_pair = loader.available_pairs[0]
        print(f"\nTestando carregamento de {test_pair}...")
        data = loader.load_forex_data(test_pair, period='1y')
        if data is not None:
            print(f"Sucesso! Dados carregados: {len(data)} registros")
            print(data.head())
        else:
            print("Falha no carregamento")