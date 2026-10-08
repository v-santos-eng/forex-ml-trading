#!/usr/bin/env python3
"""
Pipeline de Pré-processamento FX - Normalização e Feature Engineering
Prepara dados FX para treinamento de modelos de Machine Learning
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import ta  # Technical Analysis library
import argparse

warnings.filterwarnings('ignore')

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fx_preprocessing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FXPreprocessor:

    def __init__(self, data_directory: str, output_directory: str = "processed_fx_data", timeframe: str = "M1"):
        self.data_directory = Path(data_directory)
        # Configuração de timeframe
        self.timeframe = (timeframe or "M1").upper()
        tf_map = {"M1": "1T", "M5": "5T", "M15": "15T", "M30": "30T"}
        if self.timeframe not in tf_map:
            logger.warning(f"Timeframe inválido '{self.timeframe}', usando 'M1'")
            self.timeframe = "M1"
        self._tf_pandas_rule = tf_map[self.timeframe]

        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        # Configuração de pares prioritários
        self.top_5_pairs = ['usdjpy', 'gbpusd', 'eurusd', 'usdchf', 'audjpy']
        self.all_15_pairs = [
            'usdjpy', 'gbpusd', 'eurusd', 'usdchf', 'audjpy',
            'gbpjpy', 'xauusd', 'eurjpy', 'usdsek', 'usdnok',
            'euraud', 'gbpnzd', 'gbpchf', 'usddkk', 'eurnzd'
        ]
        
        # Configuração de features
        self.feature_config = {
            'technical_indicators': True,
            'price_features': True,
            'time_features': True,
            'volatility_features': True,
            'momentum_features': True,
            'lookback_periods': [5, 10, 20, 50],
            'target_periods': [1, 5, 10]  # Previsão para 1, 5 e 10 períodos
        }
        
        self.scalers = {}
        self.preprocessing_stats = {}
    
    def load_pair_data(self, pair_name: str, sample_size: int = 100000) -> pd.DataFrame:
        """Carrega dados de um par de moedas, ajustando o tamanho da amostra para o timeframe."""
        logger.info(f"📊 Carregando dados para {pair_name.upper()} (Timeframe: {self.timeframe})...")

        # Ajuste dinâmico do sample_size para compensar o resampling
        timeframe_multipliers = {"M1": 1, "M5": 5, "M15": 15, "M30": 30}
        adjusted_sample_size = sample_size * timeframe_multipliers.get(self.timeframe, 1)
        logger.info(f"Amostra base: {sample_size}, Amostra ajustada para {self.timeframe}: {adjusted_sample_size}")

        pair_dir = self.data_directory / pair_name.upper()
        if not pair_dir.exists():
            logger.error(f"Diretório não encontrado: {pair_dir}")
            return pd.DataFrame()

        # Carrega arquivos CSV
        csv_files = sorted(list(pair_dir.glob('*.csv')))
        if not csv_files:
            logger.error(f"Nenhum arquivo CSV encontrado para {pair_name}")
            return pd.DataFrame()

        # Combina dados de múltiplos arquivos
        dataframes = []
        total_loaded = 0

        for csv_file in reversed(csv_files): # Começa pelos mais recentes
            try:
                df = pd.read_csv(csv_file, sep=';', header=None)
                if len(df.columns) >= 6:
                    df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'] + [f'Col_{i}' for i in range(6, len(df.columns))]
                    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S', errors='coerce')
                    for col in ['Open', 'High', 'Low', 'Close']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.dropna(subset=['DateTime', 'Open', 'High', 'Low', 'Close'])

                    if len(df) > 0:
                        dataframes.append(df)
                        total_loaded += len(df)
                        if total_loaded >= adjusted_sample_size:
                            logger.info(f"Atingido o limite de {adjusted_sample_size} amostras.")
                            break
            except Exception as e:
                logger.warning(f"Erro ao carregar {csv_file}: {e}")
                continue

        if not dataframes:
            logger.error(f"Nenhum dado válido carregado para {pair_name}")
            return pd.DataFrame()

        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df = combined_df.sort_values('DateTime').reset_index(drop=True)

        if len(combined_df) > adjusted_sample_size:
            combined_df = combined_df.tail(adjusted_sample_size)

        if self.timeframe != 'M1':
            logger.info(f"Reamostrando dados para {self.timeframe}...")
            combined_df = combined_df.set_index('DateTime')
            resampled_df = combined_df.resample(self._tf_pandas_rule).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            logger.info(f"Amostras antes do resample: {len(combined_df)}, Depois: {len(resampled_df)}")
            return resampled_df.reset_index()
        else:
            return combined_df
     
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features baseadas em preços"""
        logger.info("🔧 Criando features de preço...")
        
        # Features básicas de preço
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['Price_Range'] = df['High'] - df['Low']
        df['Body_Size'] = abs(df['Close'] - df['Open'])
        df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        
        # Retornos
        df['Return_1'] = df['Close'].pct_change(1)
        df['Return_5'] = df['Close'].pct_change(5)
        df['Return_10'] = df['Close'].pct_change(10)
        
        # Log returns
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Preços relativos
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        return df
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria indicadores técnicos"""
        logger.info("📈 Criando indicadores técnicos...")
        
        try:
            # Moving Averages
            for period in [5, 10, 20, 50]:
                df[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
                df[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
            
            # RSI
            df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
            df['RSI_30'] = ta.momentum.rsi(df['Close'], window=30)
            
            # MACD
            df['MACD'] = ta.trend.macd(df['Close'])
            df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
            df['MACD_Histogram'] = ta.trend.macd_diff(df['Close'])
            
            # Bollinger Bands
            df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'])
            df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'])
            df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'])
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Stochastic
            df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
            df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
            
            # ATR (Average True Range)
            df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
            
            # Williams %R
            df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
            
            # CCI (Commodity Channel Index)
            df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
            
        except Exception as e:
            logger.warning(f"Erro ao criar indicadores técnicos: {e}")
        
        return df
     
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features avançadas de volatilidade com 4 metodologias"""
        logger.info("📈 Criando features avançadas de volatilidade...")
        
        # 1. ATR (Average True Range)
        atr_14 = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ATR_14'] = atr_14.average_true_range()
        
        # 2. Bandas de Bollinger
        bb_20 = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bb_20.bollinger_hband()
        df['BB_lower'] = bb_20.bollinger_lband()
        df['BB_bandwidth'] = (df['BB_upper'] - df['BB_lower']) / bb_20.bollinger_mavg()
        
        # 3. Volatilidade EWMA
        for window in [5, 10, 20]:
            returns = np.log(df['Close']/df['Close'].shift(1))
            df[f'Vol_EWMA_{window}'] = returns.ewm(span=window).std() * np.sqrt(252)
        
        # 4. Z-Score de Volatilidade
        mean_vol = df['Close'].rolling(50).mean()
        std_vol = df['Close'].rolling(50).std()
        df['Vol_ZScore'] = (df['Close'] - mean_vol) / std_vol
        
        return df.replace([np.inf, -np.inf], np.nan).dropna()
     
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de momentum"""
        logger.info("🚀 Criando features de momentum...")
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
        
        # Price acceleration
        df['Price_Acceleration'] = df['Return_1'] - df['Return_1'].shift(1)
        
        # Relative strength vs moving average
        for period in [10, 20, 50]:
            sma = df['Close'].rolling(window=period).mean()
            df[f'RS_SMA_{period}'] = df['Close'] / sma - 1
        
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de interação entre indicadores e preços."""
        logger.info("🤝 Criando features de interação...")

        # Interação entre Preço e Médias Móveis
        for period in [5, 10, 20, 50]:
            if f'SMA_{period}' in df.columns:
                df[f'Price_vs_SMA_{period}'] = df['Close'] / df[f'SMA_{period}'] - 1
            if f'EMA_{period}' in df.columns:
                df[f'Price_vs_EMA_{period}'] = df['Close'] / df[f'EMA_{period}'] - 1

        # Cruzamento de Médias Móveis
        if 'SMA_10' in df.columns and 'SMA_50' in df.columns:
            df['SMA_10_vs_SMA_50'] = df['SMA_10'] - df['SMA_50']
        if 'EMA_10' in df.columns and 'EMA_50' in df.columns:
            df['EMA_10_vs_EMA_50'] = df['EMA_10'] - df['EMA_50']

        # Interação RSI e MACD
        if 'RSI_14' in df.columns and 'MACD' in df.columns:
            df['RSI_x_MACD'] = df['RSI_14'] * df['MACD_Signal']

        # Posição do preço dentro do range do dia
        if 'Price_Range' in df.columns and df['Price_Range'].min() > 0:
            df['Price_Position_in_Range'] = (df['Close'] - df['Low']) / df['Price_Range']

        return df
     
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features temporais"""
        logger.info("⏰ Criando features temporais...")
        
        # Extrair componentes de tempo
        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.dayofweek
        df['Month'] = df['DateTime'].dt.month
        df['Quarter'] = df['DateTime'].dt.quarter
        df['Year'] = df['DateTime'].dt.year
        
        # Features cíclicas
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Sessões de trading
        df['Asian_Session'] = ((df['Hour'] >= 0) & (df['Hour'] < 8)).astype(int)
        df['European_Session'] = ((df['Hour'] >= 8) & (df['Hour'] < 16)).astype(int)
        df['American_Session'] = ((df['Hour'] >= 16) & (df['Hour'] < 24)).astype(int)
        
        return df
     
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria variáveis target para previsão"""
        logger.info("🎯 Criando variáveis target...")
        
        # Define um threshold para a zona neutra (ex: 0.05%)
        neutral_threshold = 0.0005

        for period in self.feature_config['target_periods']:
            # Retorno futuro
            df[f'Target_Return_{period}'] = df['Close'].shift(-period) / df['Close'] - 1
            
            # Direção do movimento (classificação com zona neutra)
            # 1 = Alta, 0 = Queda, 2 = Neutro
            conditions = [
                df[f'Target_Return_{period}'] > neutral_threshold,
                df[f'Target_Return_{period}'] < -neutral_threshold
            ]
            choices = [1, 0] # 1 para Alta, 0 para Queda
            df[f'Target_Direction_{period}'] = np.select(conditions, choices, default=2) # 2 para Neutro

            # Magnitude do movimento
            df[f'Target_Magnitude_{period}'] = abs(df[f'Target_Return_{period}'])
            
            # Preço futuro
            df[f'Target_Price_{period}'] = df['Close'].shift(-period)
        
        return df
     
    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa e valida os dados"""
        logger.info("🧹 Limpando e validando dados...")
        
        initial_rows = len(df)
        
        # Remove valores infinitos
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Aplicar Winsorization em vez de remover outliers (preserva dados)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        winsorized_features = 0
        
        # Filtra features de volatilidade para Winsorization de 5%
        volatility_features = [
            'ATR_14',
            'BB_Width',
            'Volatility_EWMA',
            'Vol_ZScore',
            'BB_%B',
            'True_Range'
        ]
        for col in numeric_columns:
            if col in volatility_features:
                # Calcular percentis para Winsorization (5% e 95%)
                lower_percentile = df[col].quantile(0.05)
                upper_percentile = df[col].quantile(0.95)
                
                # Aplicar Winsorization simétrico de 5%
                
                # Contar valores que serão winsorized
                values_below = len(df[df[col] < lower_percentile])
                values_above = len(df[df[col] > upper_percentile])
                total_winsorized = values_below + values_above
                
                if total_winsorized > 0:
                    # Aplicar Winsorization: limitar valores extremos
                    df[col] = df[col].clip(lower=lower_percentile, upper=upper_percentile)
                    winsorized_features += 1
        
        logger.info(f"🔧 Winsorization aplicada em {winsorized_features} features (preservando todas as linhas)")

        # Remove linhas com muitos valores NaN (> 50% das colunas)
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold)
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        
        if initial_rows > 0:
            logger.info(f"🧹 Removidas {removed_rows:,} linhas ({removed_rows/initial_rows*100:.2f}%)")
        else:
            logger.info(f"🧹 Removidas {removed_rows:,} linhas")
        logger.info(f"✅ Dados limpos: {final_rows:,} linhas restantes")
        
        return df
     
    def normalize_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, pair_name: str, scaler_type: str = 'robust') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """Normaliza features numéricas da forma correta para evitar data leakage."""
        logger.info(f"📏 Normalizando features com {scaler_type} scaler (evitando data leakage)...")
        
        exclude_cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'] + \
                       [col for col in train_df.columns if col.startswith('Target_')]
        
        numeric_cols = [col for col in train_df.select_dtypes(include=[np.number]).columns 
                        if col not in exclude_cols]
        
        if not numeric_cols:
            logger.warning("Nenhuma coluna numérica encontrada para normalização")
            return train_df, val_df, test_df, {}

        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:  # robust
            scaler = RobustScaler()
        
        # Clona os dataframes para evitar SettingWithCopyWarning
        train_df_norm = train_df.copy()
        val_df_norm = val_df.copy()
        test_df_norm = test_df.copy()

        # Fit no treino, transform em todos
        train_df_norm[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        val_df_norm[numeric_cols] = scaler.transform(val_df[numeric_cols])
        test_df_norm[numeric_cols] = scaler.transform(test_df[numeric_cols])
        
        scaler_info = {
            'scaler': scaler,
            'columns': numeric_cols,
            'type': scaler_type
        }
        
        logger.info(f"✅ Normalizadas {len(numeric_cols)} features para {pair_name}")
        return train_df_norm, val_df_norm, test_df_norm, scaler_info
     
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Divide dados em treino, validação e teste"""
        logger.info("✂️ Dividindo dados em treino/validação/teste...")
        
        # Remove linhas com targets NaN
        target_cols = [col for col in df.columns if col.startswith('Target_')]
        df_clean = df.dropna(subset=target_cols)
        
        # Divisão temporal (mais realista para séries temporais)
        n_total = len(df_clean)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val
        
        train_df = df_clean.iloc[:n_train].copy()
        val_df = df_clean.iloc[n_train:n_train+n_val].copy()
        test_df = df_clean.iloc[n_train+n_val:].copy()
        
        logger.info(f"📊 Divisão dos dados:")
        logger.info(f"   Treino: {len(train_df):,} amostras ({len(train_df)/n_total*100:.1f}%)")
        logger.info(f"   Validação: {len(val_df):,} amostras ({len(val_df)/n_total*100:.1f}%)")
        logger.info(f"   Teste: {len(test_df):,} amostras ({len(test_df)/n_total*100:.1f}%)")
        
        return train_df, val_df, test_df
     
    def process_pair(self, pair_name: str, sample_size: int = 100000) -> Dict:
        """Processa completamente um par de moedas"""
        logger.info(f"🔄 Processando par {pair_name.upper()}...")
        
        start_time = datetime.now()
        
        # 1. Carregar dados
        df = self.load_pair_data(pair_name, sample_size)
        if df.empty:
            return {'success': False, 'error': 'Falha ao carregar dados'}
        
        # 2. Criar features
        df = self.create_price_features(df)
        df = self.create_technical_indicators(df)
        df = self.create_volatility_features(df)
        df = self.create_momentum_features(df)
        df = self.create_interaction_features(df)
        df = self.create_time_features(df)
        df = self.create_target_variables(df)
        
        # 3. Limpar dados
        df = self.clean_and_validate_data(df)
        
        # 4. Dividir dados ANTES da normalização para evitar data leakage
        train_df, val_df, test_df = self.split_data(df)

        # Checar se os dataframes resultantes da divisão não estão vazios
        if train_df.empty or val_df.empty or test_df.empty:
            error_msg = "Divisão resultou em dataframes vazios. Abortando."
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        # 5. Normalizar features (agora da forma correta)
        train_df, val_df, test_df, scaler_info = self.normalize_features(
            train_df, val_df, test_df, pair_name
        )
        self.scalers[pair_name] = scaler_info

        # 6. Salvar dados processados
        pair_output_dir = self.output_directory / pair_name.upper()
        pair_output_dir.mkdir(exist_ok=True)
        
        train_df.to_parquet(pair_output_dir / 'train.parquet', index=False)
        val_df.to_parquet(pair_output_dir / 'validation.parquet', index=False)
        test_df.to_parquet(pair_output_dir / 'test.parquet', index=False)
        
        # 7. Salvar metadados
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Coleta de colunas a partir do dataframe de treino (que é o mais completo)
        feature_cols = [col for col in train_df.columns if not col.startswith('Target_') and col not in ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        target_cols = [col for col in train_df.columns if col.startswith('Target_')]

        metadata = {
            'pair': pair_name.upper(),
            'timeframe': self.timeframe,
            'processing_time_seconds': processing_time,
            'original_samples': len(df),
            'train_samples': len(train_df),
            'validation_samples': len(val_df),
            'test_samples': len(test_df),
            'total_features': len(feature_cols),
            'target_columns': target_cols,
            'feature_columns': feature_cols,
            'scaler_info': self.scalers.get(pair_name, {}),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(pair_output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ {pair_name.upper()} processado em {processing_time:.2f}s")
        logger.info(f"📁 Dados salvos em: {pair_output_dir}")
        
        return {
            'success': True,
            'pair': pair_name.upper(),
            'processing_time': processing_time,
            'samples': {
                'train': len(train_df),
                'validation': len(val_df),
                'test': len(test_df)
            },
            'features': len(df.columns),
            'output_directory': str(pair_output_dir)
        }
     
    def process_top_5_pairs(self, sample_size: int = 100000) -> Dict:
        """Processa os 5 principais pares"""
        logger.info("🎯 Processando TOP 5 pares prioritários...")
        
        results = {}
        for pair in self.top_5_pairs:
            results[pair] = self.process_pair(pair, sample_size)
        
        return results
     
    def process_all_pairs(self, sample_size: int = 100000) -> Dict:
        """Processa todos os 15 pares"""
        logger.info("📊 Processando todos os 15 pares...")
        
        results = {}
        for pair in self.all_15_pairs:
            results[pair] = self.process_pair(pair, sample_size)
        
        return results
     
    def generate_preprocessing_report(self, results: Dict) -> str:
        """Gera relatório de pré-processamento"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"preprocessing_report_{timestamp}.json"
        
        # Calcula estatísticas
        successful_pairs = [p for p, r in results.items() if r.get('success', False)]
        total_samples = sum([r.get('samples', {}).get('train', 0) + 
                           r.get('samples', {}).get('validation', 0) + 
                           r.get('samples', {}).get('test', 0) 
                           for r in results.values() if r.get('success', False)])
        
        avg_processing_time = np.mean([r.get('processing_time', 0) 
                                    for r in results.values() if r.get('success', False)])
        
        report = {
            'timestamp': timestamp,
            'preprocessing_summary': {
                'total_pairs_processed': len(results),
                'successful_pairs': len(successful_pairs),
                'failed_pairs': len(results) - len(successful_pairs),
                'total_samples_processed': total_samples,
                'average_processing_time_seconds': float(avg_processing_time),
                'output_directory': str(self.output_directory),
                'feature_config': self.feature_config
            },
            'pair_results': results,
            'next_steps': {
                'ready_for_training': successful_pairs,
                'recommended_action': 'Iniciar treinamento de modelos ML',
                'model_types': ['Random Forest', 'XGBoost', 'LSTM', 'Transformer']
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📋 Relatório de pré-processamento salvo: {report_file}")
        return report_file
     
    def print_preprocessing_summary(self, results: Dict):
        """Imprime resumo do pré-processamento"""
        print("\\n" + "="*60)
        print("🔄 RESUMO DO PRÉ-PROCESSAMENTO")
        print("="*60)
        
        successful_pairs = [p for p, r in results.items() if r.get('success', False)]
        failed_pairs = [p for p, r in results.items() if not r.get('success', False)]
        
        print(f"\\n✅ Pares Processados: {len(successful_pairs)}/{len(results)}")
        
        if successful_pairs:
            print("\\n🏆 PARES PRONTOS PARA TREINAMENTO:")
            for i, pair in enumerate(successful_pairs, 1):
                result = results[pair]
                samples = result.get('samples', {})
                total = samples.get('train', 0) + samples.get('validation', 0) + samples.get('test', 0)
                features = result.get('features', 0)
                time_taken = result.get('processing_time', 0)
                print(f"{i}. {pair.upper()}: {total:,} amostras, {features} features ({time_taken:.2f}s)")
        
        if failed_pairs:
            print(f"\\n❌ Pares com Falha: {', '.join(failed_pairs)}")
        
        if len(successful_pairs) >= 5:
            print("\\n🎉 PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
            print("✅ Dados prontos para treinamento de modelos ML")
        else:
            print("\\n⚠️ ATENÇÃO: Poucos pares processados com sucesso")
            print("❌ Revisar configuração antes de prosseguir")


def main():
    parser = argparse.ArgumentParser(description="FX Preprocessor - criação de datasets e features para ML com suporte a multi-timeframe")
    parser.add_argument('--data_dir', default='extracted_fx_data', help='Diretório de dados brutos (por par)')
    parser.add_argument('--output_dir', default=None, help='Diretório de saída dos dados processados (default depende do timeframe)')
    parser.add_argument('--timeframe', default='M1', choices=['M1','M5','M15','M30'], help='Timeframe alvo para processamento (resampling aplicado se != M1)')
    parser.add_argument('--sample_size', type=int, default=50000, help='Número máximo de linhas a carregar por par')
    args = parser.parse_args()

    # Configuração
    data_dir = args.data_dir
    # Define diretório de saída padrão baseado no timeframe
    output_dir = args.output_dir if args.output_dir else ('processed_fx_data' if args.timeframe == 'M1' else f'processed_fx_data_{args.timeframe}')
    
    if not Path(data_dir).exists():
        print(f"❌ Diretório não encontrado: {data_dir}")
        print("Execute primeiro o pipeline de extração e validação")
        return
    
    # Inicializar preprocessor
    preprocessor = FXPreprocessor(data_dir, output_dir, timeframe=args.timeframe)
    
    print("🚀 Iniciando Pré-processamento de Dados FX")
    print("="*50)
    print(f"Timeframe alvo: {preprocessor.timeframe} | Saída: {output_dir}")
    
    # Processar TOP 5 pares primeiro
    print("\\n🎯 Fase 1: Processando TOP 5 pares...")
    top5_results = preprocessor.process_top_5_pairs(sample_size=args.sample_size)  # Amostra menor para teste
    
    # Verificar se TOP 5 foi bem-sucedido
    successful_top5 = [p for p, r in top5_results.items() if r.get('success', False)]
    
    if len(successful_top5) >= 3:
        print("\\n📊 Fase 2: Processando pares adicionais...")
        # Processar pares restantes
        remaining_pairs = [p for p in preprocessor.all_15_pairs if p not in preprocessor.top_5_pairs]
        additional_results = {}
        for pair in remaining_pairs[:5]:  # Processa mais 5 pares
            additional_results[pair] = preprocessor.process_pair(pair, sample_size=args.sample_size)
        
        # Combinar resultados
        all_results = {**top5_results, **additional_results}
    else:
        print("\\n⚠️ Problemas com TOP 5 pares. Continuando apenas com os bem-sucedidos...")
        all_results = top5_results
    
    # Gerar relatório
    report_file = preprocessor.generate_preprocessing_report(all_results)
    
    # Imprimir resumo
    preprocessor.print_preprocessing_summary(all_results)
    
    print(f"\\n📋 Relatório detalhado: {report_file}")
    print(f"📁 Dados processados em: {output_dir}")
    print("\\n🎯 Próximo passo: Treinamento de modelos ML")
 
if __name__ == "__main__":
    main()