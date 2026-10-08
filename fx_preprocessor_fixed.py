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
        """Carrega dados de um par de moedas"""
        logger.info(f"📊 Carregando dados para {pair_name.upper()}...")

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

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, sep=';', header=None)
                if len(df.columns) >= 6:
                    # Formato: DateTime;Open;High;Low;Close;Volume
                    df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'] + [f'Col_{i}' for i in range(6, len(df.columns))]

                    # Converte DateTime
                    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S', errors='coerce')

                    # Converte colunas OHLC para numérico
                    for col in ['Open', 'High', 'Low', 'Close']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Remove linhas com valores inválidos
                    df = df.dropna(subset=['DateTime', 'Open', 'High', 'Low', 'Close'])

                    if len(df) > 0:
                        dataframes.append(df)
                        total_loaded += len(df)

                        # Limita o tamanho da amostra
                        if total_loaded >= sample_size:
                            break

            except Exception as e:
                logger.warning(f"Erro ao carregar {csv_file}: {e}")
                continue

        if not dataframes:
            logger.error(f"Nenhum dado válido carregado para {pair_name}")
            return pd.DataFrame()

        # Combina todos os dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Limita ao tamanho da amostra
        if len(combined_df) > sample_size:
            combined_df = combined_df.tail(sample_size)

        # DateTime já foi criado no loop acima
        combined_df = combined_df.dropna(subset=['DateTime'])
        combined_df = combined_df.sort_values('DateTime').reset_index(drop=True)

        # Resampling para timeframe desejado (se não for M1)
        if self.timeframe != 'M1':
            df_idx = combined_df.set_index('DateTime')
            agg = {}
            if 'Open' in df_idx.columns:
                agg['Open'] = 'first'
            if 'High' in df_idx.columns:
                agg['High'] = 'max'
            if 'Low' in df_idx.columns:
                agg['Low'] = 'min'
            if 'Close' in df_idx.columns:
                agg['Close'] = 'last'
            if 'Volume' in df_idx.columns:
                agg['Volume'] = 'sum'
            if not agg:
                logger.error("Colunas OHLCV não encontradas para resampling")
                return pd.DataFrame()
            resampled = df_idx.resample(self._tf_pandas_rule).agg(agg)
            required_cols = [c for c in ['Open','High','Low','Close'] if c in resampled.columns]
            resampled = resampled.dropna(subset=required_cols).reset_index()
            combined_df = resampled
            logger.info(f"Resampling aplicado para {self.timeframe}: {len(combined_df):,} registros")

        logger.info(f"✅ Carregados {len(combined_df):,} registros para {pair_name.upper()}")
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
        """Cria features de volatilidade"""
        logger.info("📊 Criando features de volatilidade...")
        
        # Volatilidade realizada
        for period in [5, 10, 20]:
            df[f'Volatility_{period}'] = df['Log_Return'].rolling(window=period).std() * np.sqrt(period)
        
        # Volatilidade de Parkinson
        df['Parkinson_Vol'] = np.sqrt((1/(4*np.log(2))) * np.log(df['High']/df['Low'])**2)
        
        # Volatilidade de Garman-Klass
        df['GK_Vol'] = np.sqrt(
            0.5 * np.log(df['High']/df['Low'])**2 - 
            (2*np.log(2)-1) * np.log(df['Close']/df['Open'])**2
        )
        
        # Range verdadeiro
        df['True_Range'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        
        return df

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
        
        for period in self.feature_config['target_periods']:
            # Retorno futuro
            df[f'Target_Return_{period}'] = df['Close'].shift(-period) / df['Close'] - 1
            
            # Direção do movimento (classificação)
            df[f'Target_Direction_{period}'] = (df[f'Target_Return_{period}'] > 0).astype(int)
            
            # Magnitude do movimento
            df[f'Target_Magnitude_{period}'] = abs(df[f'Target_Return_{period}'])
            
            # Preço futuro
            df[f'Target_Price_{period}'] = df['Close'].shift(-period)
        
        return df

    def validate_indicator_ranges(self, df: pd.DataFrame) -> Dict[str, float]:
        """Valida ranges de indicadores em dados NÃO normalizados e retorna métricas."""
        def ratio_outside(s: pd.Series, lo: float, hi: float) -> float:
            s = s.dropna()
            if len(s) == 0:
                return float('nan')
            return float(((s < lo) | (s > hi)).mean())
        
        checks = {}
        if 'RSI_14' in df.columns:
            checks['RSI_14_out_of_[0,100]'] = ratio_outside(df['RSI_14'], 0.0, 100.0)
        if 'Stoch_K' in df.columns:
            checks['Stoch_K_out_of_[0,100]'] = ratio_outside(df['Stoch_K'], 0.0, 100.0)
        if 'BB_Position' in df.columns:
            checks['BB_Position_out_of_[0,1]'] = ratio_outside(df['BB_Position'], 0.0, 1.0)
        if 'ATR_14' in df.columns:
            s = df['ATR_14'].dropna()
            checks['ATR_14_negative_ratio'] = float((s < 0).mean()) if len(s) else float('nan')
        return checks

    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa e valida os dados"""
        logger.info("🧹 Limpando e validando dados...")
        
        initial_rows = len(df)
        
        # Remove valores infinitos
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remove outliers extremos (> 5 desvios padrão)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['DateTime', 'Date', 'Time']:
                mean = df[col].mean()
                std = df[col].std()
                df = df[abs(df[col] - mean) <= 5 * std]
        
        # Remove linhas com muitos valores NaN (> 50% das colunas)
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold)
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        
        logger.info(f"🧹 Removidas {removed_rows:,} linhas ({removed_rows/initial_rows*100:.2f}%)")
        logger.info(f"✅ Dados limpos: {final_rows:,} linhas restantes")
        
        return df

    def normalize_features(self, df: pd.DataFrame, pair_name: str, scaler_type: str = 'robust') -> pd.DataFrame:
        """Normaliza features numéricas"""
        logger.info(f"📏 Normalizando features com {scaler_type} scaler...")
        
        # Seleciona colunas numéricas para normalização
        exclude_cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'] + \
                      [col for col in df.columns if col.startswith('Target_')]
        
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        
        if not numeric_cols:
            logger.warning("Nenhuma coluna numérica encontrada para normalização")
            return df
        
        # Escolhe o scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:  # robust
            scaler = RobustScaler()
        
        # Aplica normalização
        df_normalized = df.copy()
        df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # Salva o scaler para uso futuro
        self.scalers[pair_name] = {
            'scaler': scaler,
            'columns': numeric_cols,
            'type': scaler_type
        }
        
        logger.info(f"✅ Normalizadas {len(numeric_cols)} features")
        return df_normalized

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
        df = self.create_time_features(df)
        df = self.create_target_variables(df)
        
        # 2.1. Checks de indicadores no dado cru (pré-normalização)
        indicator_checks_raw = self.validate_indicator_ranges(df)
        
        # 3. Limpar dados
        df = self.clean_and_validate_data(df)
        
        # 4. Normalizar features
        df = self.normalize_features(df, pair_name)
        
        # 5. Dividir dados
        train_df, val_df, test_df = self.split_data(df)
        
        # 6. Salvar dados processados
        pair_output_dir = self.output_directory / pair_name.upper()
        pair_output_dir.mkdir(exist_ok=True)
        
        train_df.to_parquet(pair_output_dir / 'train.parquet', index=False)
        val_df.to_parquet(pair_output_dir / 'validation.parquet', index=False)
        test_df.to_parquet(pair_output_dir / 'test.parquet', index=False)
        
        # 7. Salvar metadados
        processing_time = (datetime.now() - start_time).total_seconds()
        
        metadata = {
            'pair': pair_name.upper(),
            'timeframe': self.timeframe,
            'processing_time_seconds': processing_time,
            'original_samples': len(df),
            'train_samples': len(train_df),
            'validation_samples': len(val_df),
            'test_samples': len(test_df),
            'total_features': len(df.columns),
            'target_columns': [col for col in df.columns if col.startswith('Target_')],
            'feature_columns': [col for col in df.columns if not col.startswith('Target_') and col not in ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']],
            'scaler_info': self.scalers.get(pair_name, {}),
            'indicator_checks_raw': indicator_checks_raw,
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
        print("\n" + "="*60)
        print("🔄 RESUMO DO PRÉ-PROCESSAMENTO")
        print("="*60)
        
        successful_pairs = [p for p, r in results.items() if r.get('success', False)]
        failed_pairs = [p for p, r in results.items() if not r.get('success', False)]
        
        print(f"\n✅ Pares Processados: {len(successful_pairs)}/{len(results)}")
        
        if successful_pairs:
            print("\n🏆 PARES PRONTOS PARA TREINAMENTO:")
            for i, pair in enumerate(successful_pairs, 1):
                result = results[pair]
                samples = result.get('samples', {})
                total = samples.get('train', 0) + samples.get('validation', 0) + samples.get('test', 0)
                features = result.get('features', 0)
                time_taken = result.get('processing_time', 0)
                print(f"{i}. {pair.upper()}: {total:,} amostras, {features} features ({time_taken:.2f}s)")
        
        if failed_pairs:
            print(f"\n❌ Pares com Falha: {', '.join(failed_pairs)}")
        
        if len(successful_pairs) >= 5:
            print("\n🎉 PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
            print("✅ Dados prontos para treinamento de modelos ML")
        else:
            print("\n⚠️ ATENÇÃO: Poucos pares processados com sucesso")
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
    print("\n🎯 Fase 1: Processando TOP 5 pares...")
    top5_results = preprocessor.process_top_5_pairs(sample_size=args.sample_size)
    
    # Verificar se TOP 5 foi bem-sucedido
    successful_top5 = [p for p, r in top5_results.items() if r.get('success', False)]
    
    if len(successful_top5) >= 3:
        print("\n📊 Fase 2: Processando pares adicionais...")
        # Processar pares restantes
        remaining_pairs = [p for p in preprocessor.all_15_pairs if p not in preprocessor.top_5_pairs]
        additional_results = {}
        for pair in remaining_pairs[:5]:  # Processa mais 5 pares
            additional_results[pair] = preprocessor.process_pair(pair, sample_size=args.sample_size)
        
        # Combinar resultados
        all_results = {**top5_results, **additional_results}
    else:
        print("\n⚠️ Problemas com TOP 5 pares. Continuando apenas com os bem-sucedidos...")
        all_results = top5_results
    
    # Gerar relatório
    report_file = preprocessor.generate_preprocessing_report(all_results)
    
    # Imprimir resumo
    preprocessor.print_preprocessing_summary(all_results)
    
    print(f"\n📋 Relatório detalhado: {report_file}")
    print(f"📁 Dados processados em: {output_dir}")
    print("\n🎯 Próximo passo: Treinamento de modelos ML")

if __name__ == "__main__":
    main()