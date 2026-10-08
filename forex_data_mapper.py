#!/usr/bin/env python3
"""
Mapeador de Dados Forex - Correção de Pares de Moedas
Mapeia símbolos do Yahoo Finance para dados locais disponíveis
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

class ForexDataMapper:
    """
    Classe para mapear e carregar dados forex locais
    """
    
    def __init__(self, data_dir: str = "extracted_fx_data"):
        self.data_dir = Path(data_dir)
        
        # Mapeamento de símbolos Yahoo Finance para dados locais
        self.symbol_mapping = {
            'EURUSD=X': 'EURUSD',
            'GBPUSD=X': 'GBPUSD', 
            'USDJPY=X': 'USDJPY',
            'USDCHF=X': 'USDCHF',
            'AUDUSD=X': 'AUDUSD',
            'USDCAD=X': 'USDCAD',
            'NZDUSD=X': 'NZDUSD',
            'EURGBP=X': 'EURGBP',
            'EURJPY=X': 'EURJPY',
            'GBPJPY=X': 'GBPJPY',
            'AUDJPY=X': 'AUDJPY',
            'AUDCAD=X': 'AUDCAD',
            'AUDCHF=X': 'AUDCHF',
            'AUDNZD=X': 'AUDNZD',
            'CADCHF=X': 'CADCHF',
            'CADJPY=X': 'CADJPY',
            'CHFJPY=X': 'CHFJPY',
            'EURAUD=X': 'EURAUD',
            'EURCAD=X': 'EURCAD',
            'EURCHF=X': 'EURCHF',
            'EURNZD=X': 'EURNZD',
            'GBPAUD=X': 'GBPAUD',
            'GBPCAD=X': 'GBPCAD',
            'GBPCHF=X': 'GBPCHF',
            'GBPNZD=X': 'GBPNZD',
            'NZDCAD=X': 'NZDCAD',
            'NZDCHF=X': 'NZDCHF',
            'NZDJPY=X': 'NZDJPY',
            'XAUUSD=X': 'XAUUSD'
        }
        
        # Verificar pares disponíveis
        self.available_pairs = self._scan_available_pairs()
        print(f"Pares disponíveis localmente: {len(self.available_pairs)}")
        
    def _scan_available_pairs(self) -> list:
        """
        Escaneia o diretório de dados para encontrar pares disponíveis
        """
        available = []
        if self.data_dir.exists():
            for item in self.data_dir.iterdir():
                if item.is_dir():
                    # Verificar se há arquivos CSV no diretório
                    csv_files = list(item.glob("*.csv"))
                    if csv_files:
                        available.append(item.name)
        return available
    
    def get_local_symbol(self, yahoo_symbol: str) -> Optional[str]:
        """
        Converte símbolo do Yahoo Finance para símbolo local
        """
        local_symbol = self.symbol_mapping.get(yahoo_symbol, yahoo_symbol)
        
        # Remover =X se presente
        if local_symbol.endswith('=X'):
            local_symbol = local_symbol[:-2]
            
        return local_symbol if local_symbol in self.available_pairs else None
    
    def load_forex_data(self, yahoo_symbol: str, period: str = '1y') -> Optional[pd.DataFrame]:
        """
        Carrega dados forex locais baseado no símbolo do Yahoo Finance
        
        Args:
            yahoo_symbol: Símbolo do Yahoo Finance (ex: EURUSD=X)
            period: Período de dados (ex: 1y, 6mo, 3mo)
            
        Returns:
            DataFrame com dados OHLCV ou None se não encontrado
        """
        local_symbol = self.get_local_symbol(yahoo_symbol)
        
        if not local_symbol:
            print(f"Par {yahoo_symbol} não disponível localmente")
            return None
            
        pair_dir = self.data_dir / local_symbol
        if not pair_dir.exists():
            print(f"Diretório não encontrado: {pair_dir}")
            return None
            
        # Encontrar arquivos CSV mais recentes
        csv_files = sorted(pair_dir.glob("*.csv"), reverse=True)
        
        if not csv_files:
            print(f"Nenhum arquivo CSV encontrado em {pair_dir}")
            return None
            
        # Carregar dados dos arquivos mais recentes
        dfs = []
        files_to_load = min(3, len(csv_files))  # Carregar até 3 arquivos mais recentes
        
        for csv_file in csv_files[:files_to_load]:
            parsed_df = self._parse_csv_file(str(csv_file))
            if not parsed_df.empty:
                dfs.append(parsed_df)
        
        if not dfs:
            print(f"Nenhum dado válido encontrado para {local_symbol}")
            return None
            
        # Combinar todos os DataFrames
        combined_df = pd.concat(dfs, ignore_index=False)
        combined_df = combined_df.sort_index()
        
        # Remover duplicatas
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        
        # Filtrar por período se especificado
        if period:
            combined_df = self._filter_by_period(combined_df, period)
            
        print(f"Dados carregados para {local_symbol}: {len(combined_df)} registros")
        return combined_df
    
    def _filter_by_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """
        Filtra DataFrame por período especificado
        """
        if df.empty:
            return df
            
        end_date = df.index.max()
        
        # Mapear períodos para dias
        period_days = {
            '1d': 1,
            '5d': 5,
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825,
            'max': None
        }
        
        days = period_days.get(period, 365)
        
        if days is None:
            return df
            
        start_date = end_date - pd.Timedelta(days=days)
        return df[df.index >= start_date]
    
    def get_available_pairs(self) -> Dict[str, str]:
        """
        Retorna mapeamento de símbolos Yahoo Finance para símbolos locais disponíveis
        """
        available_mapping = {}
        for yahoo_symbol, local_symbol in self.symbol_mapping.items():
            if local_symbol in self.available_pairs:
                available_mapping[yahoo_symbol] = local_symbol
        return available_mapping
    
    def list_available_pairs(self) -> None:
        """
        Lista todos os pares disponíveis
        """
        print("\nPares de moedas disponíveis:")
        print("=" * 40)
        
        available_mapping = self.get_available_pairs()
        
        for yahoo_symbol, local_symbol in available_mapping.items():
            pair_dir = self.data_dir / local_symbol
            csv_files = list(pair_dir.glob("*.csv"))
            print(f"{yahoo_symbol:12} -> {local_symbol:8} ({len(csv_files)} arquivos)")
            
        if not available_mapping:
            print("Nenhum par disponível")
            
        print(f"\nTotal: {len(available_mapping)} pares disponíveis")

    def _parse_csv_file(self, file_path: str) -> pd.DataFrame:
        """
        Parse arquivo CSV com dados de forex
        
        Args:
            file_path: Caminho para o arquivo CSV
            
        Returns:
            DataFrame com dados formatados
        """
        try:
            # Ler CSV com separador ponto e vírgula
            df = pd.read_csv(file_path, sep=';', header=None, dtype=str)
            
            # Definir nomes das colunas - formato: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
            df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'][:len(df.columns)]
            
            # Converter DateTime - formato YYYYMMDD HHMMSS
            df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S', errors='coerce')
            
            # Remover linhas com DateTime inválido
            df = df.dropna(subset=['DateTime'])
            df.set_index('DateTime', inplace=True)
            
            # Converter colunas numéricas
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remover linhas com NaN nos preços
            df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Erro ao carregar {file_path}: {e}")
            return pd.DataFrame()


# Função de conveniência
def create_forex_mapper(data_dir: str = "extracted_fx_data") -> ForexDataMapper:
    """
    Cria uma instância do mapeador de dados forex
    """
    return ForexDataMapper(data_dir)

if __name__ == "__main__":
    # Teste do mapeador
    mapper = ForexDataMapper()
    mapper.list_available_pairs()
    
    # Testar carregamento de dados
    test_pairs = ['EURUSD=X', 'GBPUSD=X']
    
    for pair in test_pairs:
        print(f"\nTestando {pair}...")
        data = mapper.load_forex_data(pair, '1mo')
        if data is not None:
            print(f"Dados carregados: {len(data)} registros")
            print(f"Período: {data.index.min()} a {data.index.max()}")
            print(f"Colunas: {list(data.columns)}")
        else:
            print(f"Falha ao carregar dados para {pair}")