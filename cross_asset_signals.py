import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from datetime import datetime, timedelta
import yfinance as yf
import requests
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from dataclasses import dataclass, field
import time
import pickle
from pathlib import Path
from collections import defaultdict, deque
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from arch import arch_model

# Suprimir warnings
warnings.filterwarnings('ignore')

@dataclass
class CrossAssetConfig:
    """Configuração para análise cross-asset"""
    
    # Ativos para análise
    forex_pairs: List[str] = field(default_factory=lambda: [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X', 'USDCAD=X'
    ])
    
    equity_indices: List[str] = field(default_factory=lambda: [
        '^GSPC', '^DJI', '^IXIC', '^FTSE', '^GDAXI', '^N225', '^HSI'
    ])
    
    commodities: List[str] = field(default_factory=lambda: [
        'GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F', 'ZC=F', 'ZS=F'
    ])
    
    bonds: List[str] = field(default_factory=lambda: [
        '^TNX', '^FVX', '^TYX', '^IRX', 'GBPUSD=X'  # Yields
    ])
    
    crypto: List[str] = field(default_factory=lambda: [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD'
    ])
    
    volatility_indices: List[str] = field(default_factory=lambda: [
        '^VIX', '^VXN', '^RVX', '^VVIX'
    ])
    
    # Parâmetros de análise
    lookback_period: int = 252  # 1 ano de dados
    correlation_window: int = 60  # Janela para correlações
    cointegration_window: int = 120  # Janela para cointegração
    regime_detection_window: int = 30  # Janela para detecção de regime
    
    # Parâmetros de features
    pca_components: int = 5
    factor_components: int = 3
    clustering_n_clusters: int = 4
    
    # Parâmetros de sinais
    signal_threshold: float = 0.7  # Threshold para sinais
    momentum_window: int = 20
    mean_reversion_window: int = 10
    
    # Parâmetros de performance
    enable_parallel: bool = True
    max_workers: int = 6
    cache_duration_hours: int = 4
    
    # Parâmetros de dados
    data_interval: str = '1d'  # Intervalo dos dados
    min_data_points: int = 100  # Mínimo de pontos para análise
    
    # Base de dados
    database_path: str = "cross_asset_data.db"
    
    # APIs externas
    alpha_vantage_key: Optional[str] = None
    fred_api_key: Optional[str] = None
    
    # Configurações de risco
    max_correlation_exposure: float = 0.8
    diversification_threshold: float = 0.3
    
Base = declarative_base()

class AssetData(Base):
    """Modelo de dados para preços de ativos"""
    __tablename__ = 'asset_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20))
    date = Column(DateTime)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Float)
    returns = Column(Float)
    log_returns = Column(Float)
    volatility = Column(Float)
    created_at = Column(DateTime, default=datetime.now)

class CrossAssetSignal(Base):
    """Modelo para sinais cross-asset"""
    __tablename__ = 'cross_asset_signals'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime)
    signal_type = Column(String(50))
    source_assets = Column(String(200))
    target_asset = Column(String(20))
    signal_strength = Column(Float)
    signal_direction = Column(Integer)  # 1 para compra, -1 para venda
    confidence = Column(Float)
    signal_metadata = Column(Text)  # JSON com informações adicionais
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)

class CrossAssetAnalyzer:
    """
    Sistema avançado de análise cross-asset para trading
    
    Funcionalidades:
    1. Coleta de dados de múltiplos ativos
    2. Análise de correlações dinâmicas
    3. Detecção de cointegração
    4. Análise de componentes principais (PCA)
    5. Factor analysis
    6. Detecção de regimes de mercado
    7. Sinais de momentum cross-asset
    8. Sinais de mean reversion
    9. Análise de volatilidade cruzada
    10. Risk-on/Risk-off signals
    """
    
    def __init__(self, 
                 config: Optional[CrossAssetConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            config: Configuração do sistema
            logger: Logger para debug
        """
        self.config = config or CrossAssetConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Inicializar scalers
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # Cache de dados
        self.data_cache = {}
        self.correlation_cache = {}
        
        # Modelos
        self.pca_model = None
        self.factor_model = None
        self.clustering_model = None
        
        # Inicializar banco de dados
        self._init_database()
        
        # Carregar modelos se existirem
        self._load_models()
    
    def generate_signals(self, assets_data: Dict[str, pd.Series]) -> List[Dict]:
        """
        Gera sinais cross-asset baseado nos dados dos ativos
        
        Args:
            assets_data: Dicionário com dados de preços dos ativos
            
        Returns:
            Lista de sinais gerados
        """
        try:
            # Converter Series para DataFrame se necessário
            market_data = {}
            for asset, data in assets_data.items():
                if isinstance(data, pd.Series):
                    df = pd.DataFrame({
                        'close': data,
                        'open': data,
                        'high': data * 1.001,
                        'low': data * 0.999,
                        'volume': [1000] * len(data)
                    })
                    market_data[asset] = df
                else:
                    market_data[asset] = data
            
            # Gerar sinais usando o método existente
            signals_df = self.generate_cross_asset_signals(market_data)
            
            # Converter DataFrame para lista de dicionários
            if isinstance(signals_df, pd.DataFrame) and not signals_df.empty:
                return signals_df.to_dict('records')
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Erro ao gerar sinais: {e}")
            return []
        
    def _init_database(self) -> None:
        """
        Inicializa banco de dados SQLite
        """
        try:
            self.engine = create_engine(f'sqlite:///{self.config.database_path}')
            Base.metadata.create_all(self.engine)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            self.logger.info("Banco de dados cross-asset inicializado")
        except Exception as e:
            self.logger.error(f"Erro ao inicializar banco: {e}")
            self.session = None
            
    def collect_market_data(self, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          force_update: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Coleta dados de mercado para todos os ativos
        
        Args:
            start_date: Data de início
            end_date: Data de fim
            force_update: Forçar atualização dos dados
            
        Returns:
            Dicionário com DataFrames por categoria de ativo
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=self.config.lookback_period + 50)
            
        if end_date is None:
            end_date = datetime.now()
            
        # Verificar cache
        cache_key = f"{start_date.date()}_{end_date.date()}"
        if not force_update and cache_key in self.data_cache:
            cache_time = self.data_cache[cache_key].get('timestamp', datetime.min)
            if datetime.now() - cache_time < timedelta(hours=self.config.cache_duration_hours):
                return self.data_cache[cache_key]['data']
                
        self.logger.info("Coletando dados de mercado cross-asset")
        
        # Coletar dados por categoria
        all_symbols = {
            'forex': self.config.forex_pairs,
            'equities': self.config.equity_indices,
            'commodities': self.config.commodities,
            'bonds': self.config.bonds,
            'crypto': self.config.crypto,
            'volatility': self.config.volatility_indices
        }
        
        market_data = {}
        
        if self.config.enable_parallel:
            # Coleta paralela
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {}
                
                for category, symbols in all_symbols.items():
                    future = executor.submit(
                        self._collect_category_data, 
                        symbols, start_date, end_date, category
                    )
                    futures[category] = future
                    
                # Coletar resultados
                for category, future in futures.items():
                    try:
                        data = future.result()
                        market_data[category] = data
                    except Exception as e:
                        self.logger.error(f"Erro ao coletar {category}: {e}")
                        market_data[category] = pd.DataFrame()
        else:
            # Coleta sequencial
            for category, symbols in all_symbols.items():
                try:
                    data = self._collect_category_data(symbols, start_date, end_date, category)
                    market_data[category] = data
                except Exception as e:
                    self.logger.error(f"Erro ao coletar {category}: {e}")
                    market_data[category] = pd.DataFrame()
                    
        # Salvar no cache
        self.data_cache[cache_key] = {
            'data': market_data,
            'timestamp': datetime.now()
        }
        
        # Salvar no banco
        self._save_market_data_to_db(market_data)
        
        return market_data
    
    def _collect_category_data(self, symbols: List[str], 
                             start_date: datetime, 
                             end_date: datetime,
                             category: str) -> pd.DataFrame:
        """
        Coleta dados para uma categoria de ativos
        """
        all_data = []
        
        for symbol in symbols:
            try:
                # Usar yfinance para coleta
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=self.config.data_interval
                )
                
                if len(data) < self.config.min_data_points:
                    self.logger.warning(f"Dados insuficientes para {symbol}: {len(data)} pontos")
                    continue
                    
                # Processar dados
                processed_data = self._process_asset_data(data, symbol, category)
                all_data.append(processed_data)
                
                # Delay entre requests
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Erro ao coletar {symbol}: {e}")
                continue
                
        if all_data:
            # Combinar dados
            combined_data = pd.concat(all_data, axis=1)
            return combined_data.dropna()
        else:
            return pd.DataFrame()
    
    def _process_asset_data(self, data: pd.DataFrame, symbol: str, category: str) -> pd.DataFrame:
        """
        Processa dados de um ativo
        """
        processed = pd.DataFrame(index=data.index)
        
        # Preços
        processed[f'{symbol}_close'] = data['Close']
        processed[f'{symbol}_high'] = data['High']
        processed[f'{symbol}_low'] = data['Low']
        processed[f'{symbol}_volume'] = data['Volume']
        
        # Returns
        processed[f'{symbol}_returns'] = data['Close'].pct_change()
        processed[f'{symbol}_log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Volatilidade
        processed[f'{symbol}_volatility'] = (
            processed[f'{symbol}_returns'].rolling(window=20).std() * np.sqrt(252)
        )
        
        # True Range e ATR
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift(1))
        low_close = np.abs(data['Low'] - data['Close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        processed[f'{symbol}_atr'] = true_range.rolling(window=14).mean()
        
        # Momentum indicators
        processed[f'{symbol}_rsi'] = self._calculate_rsi(data['Close'])
        processed[f'{symbol}_momentum'] = data['Close'] / data['Close'].shift(self.config.momentum_window) - 1
        
        # Moving averages
        processed[f'{symbol}_sma_20'] = data['Close'].rolling(window=20).mean()
        processed[f'{symbol}_sma_50'] = data['Close'].rolling(window=50).mean()
        processed[f'{symbol}_ema_12'] = data['Close'].ewm(span=12).mean()
        processed[f'{symbol}_ema_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        macd = processed[f'{symbol}_ema_12'] - processed[f'{symbol}_ema_26']
        processed[f'{symbol}_macd'] = macd
        processed[f'{symbol}_macd_signal'] = macd.ewm(span=9).mean()
        
        # Bollinger Bands
        sma_20 = processed[f'{symbol}_sma_20']
        std_20 = data['Close'].rolling(window=20).std()
        processed[f'{symbol}_bb_upper'] = sma_20 + (2 * std_20)
        processed[f'{symbol}_bb_lower'] = sma_20 - (2 * std_20)
        processed[f'{symbol}_bb_position'] = (data['Close'] - processed[f'{symbol}_bb_lower']) / (processed[f'{symbol}_bb_upper'] - processed[f'{symbol}_bb_lower'])
        
        return processed
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calcula RSI
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def analyze_correlations(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analisa correlações entre ativos
        
        Args:
            market_data: Dados de mercado por categoria
            
        Returns:
            Análise de correlações
        """
        self.logger.info("Analisando correlações cross-asset")
        
        # Combinar todos os returns
        all_returns = []
        asset_names = []
        
        for category, data in market_data.items():
            if len(data) == 0:
                continue
                
            # Extrair colunas de returns
            return_cols = [col for col in data.columns if '_returns' in col and '_log_returns' not in col]
            
            for col in return_cols:
                asset_name = col.replace('_returns', '')
                all_returns.append(data[col])
                asset_names.append(asset_name)
                
        if len(all_returns) == 0:
            return {'status': 'no_data'}
            
        # Criar DataFrame de returns
        returns_df = pd.concat(all_returns, axis=1)
        returns_df.columns = asset_names
        returns_df = returns_df.dropna()
        
        if len(returns_df) < self.config.min_data_points:
            return {'status': 'insufficient_data'}
            
        # Correlação estática
        static_corr = returns_df.corr()
        
        # Correlação dinâmica (rolling)
        window = self.config.correlation_window
        dynamic_corr = {}
        
        for i, asset1 in enumerate(asset_names):
            for j, asset2 in enumerate(asset_names[i+1:], i+1):
                if asset1 in returns_df.columns and asset2 in returns_df.columns:
                    rolling_corr = returns_df[asset1].rolling(window=window).corr(returns_df[asset2])
                    dynamic_corr[f'{asset1}_{asset2}'] = rolling_corr
                    
        # Análise de clusters de correlação
        correlation_clusters = self._analyze_correlation_clusters(static_corr)
        
        # Detecção de mudanças de regime na correlação
        regime_changes = self._detect_correlation_regime_changes(dynamic_corr)
        
        return {
            'static_correlation': static_corr,
            'dynamic_correlation': dynamic_corr,
            'correlation_clusters': correlation_clusters,
            'regime_changes': regime_changes,
            'returns_data': returns_df
        }
    
    def _analyze_correlation_clusters(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa clusters de correlação
        """
        try:
            # Usar correlação como distância
            distance_matrix = 1 - np.abs(correlation_matrix)
            
            # Clustering
            kmeans = KMeans(n_clusters=self.config.clustering_n_clusters, random_state=42)
            clusters = kmeans.fit_predict(distance_matrix)
            
            # Silhouette score
            silhouette = silhouette_score(distance_matrix, clusters)
            
            # Organizar por clusters
            cluster_dict = {}
            for i, asset in enumerate(correlation_matrix.index):
                cluster_id = clusters[i]
                if cluster_id not in cluster_dict:
                    cluster_dict[cluster_id] = []
                cluster_dict[cluster_id].append(asset)
                
            return {
                'clusters': cluster_dict,
                'silhouette_score': silhouette,
                'cluster_labels': clusters
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise de clusters: {e}")
            return {'status': 'error'}
    
    def _detect_correlation_regime_changes(self, dynamic_correlations: Dict[str, pd.Series]) -> Dict[str, List]:
        """
        Detecta mudanças de regime nas correlações
        """
        regime_changes = {}
        
        for pair, corr_series in dynamic_correlations.items():
            try:
                # Detectar mudanças significativas
                changes = []
                
                # Calcular rolling mean e std
                rolling_mean = corr_series.rolling(window=30).mean()
                rolling_std = corr_series.rolling(window=30).std()
                
                # Detectar outliers (mudanças de regime)
                z_scores = np.abs((corr_series - rolling_mean) / rolling_std)
                
                # Pontos onde z-score > 2 (mudança significativa)
                change_points = corr_series[z_scores > 2].index.tolist()
                
                regime_changes[pair] = change_points
                
            except Exception as e:
                self.logger.error(f"Erro ao detectar regime para {pair}: {e}")
                regime_changes[pair] = []
                
        return regime_changes
    
    def analyze_cointegration(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analisa cointegração entre ativos
        
        Args:
            market_data: Dados de mercado
            
        Returns:
            Análise de cointegração
        """
        self.logger.info("Analisando cointegração cross-asset")
        
        # Combinar preços
        all_prices = []
        asset_names = []
        
        for category, data in market_data.items():
            if len(data) == 0:
                continue
                
            # Extrair colunas de preços
            price_cols = [col for col in data.columns if '_close' in col]
            
            for col in price_cols:
                asset_name = col.replace('_close', '')
                all_prices.append(data[col])
                asset_names.append(asset_name)
                
        if len(all_prices) < 2:
            return {'status': 'insufficient_assets'}
            
        # Criar DataFrame de preços
        prices_df = pd.concat(all_prices, axis=1)
        prices_df.columns = asset_names
        prices_df = prices_df.dropna()
        
        # Análise de cointegração por pares
        cointegration_results = {}
        
        for i, asset1 in enumerate(asset_names):
            for j, asset2 in enumerate(asset_names[i+1:], i+1):
                if asset1 in prices_df.columns and asset2 in prices_df.columns:
                    try:
                        # Teste de cointegração de Engle-Granger
                        score, pvalue, _ = coint(prices_df[asset1], prices_df[asset2])
                        
                        cointegration_results[f'{asset1}_{asset2}'] = {
                            'score': score,
                            'pvalue': pvalue,
                            'is_cointegrated': pvalue < 0.05
                        }
                        
                    except Exception as e:
                        self.logger.error(f"Erro na cointegração {asset1}-{asset2}: {e}")
                        continue
                        
        # Teste de Johansen para cointegração múltipla
        johansen_results = None
        try:
            if len(prices_df.columns) >= 2:
                # Selecionar até 10 ativos para Johansen (limitação computacional)
                selected_assets = prices_df.iloc[:, :min(10, len(prices_df.columns))]
                johansen_test = coint_johansen(selected_assets.values, det_order=0, k_ar_diff=1)
                
                johansen_results = {
                    'trace_stat': johansen_test.lr1.tolist(),
                    'max_eigen_stat': johansen_test.lr2.tolist(),
                    'critical_values_trace': johansen_test.cvt.tolist(),
                    'critical_values_max_eigen': johansen_test.cvm.tolist(),
                    'assets': selected_assets.columns.tolist()
                }
                
        except Exception as e:
            self.logger.error(f"Erro no teste de Johansen: {e}")
            
        return {
            'pairwise_cointegration': cointegration_results,
            'johansen_test': johansen_results,
            'prices_data': prices_df
        }
    
    def perform_pca_analysis(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Realiza análise de componentes principais
        
        Args:
            market_data: Dados de mercado
            
        Returns:
            Resultados da análise PCA
        """
        self.logger.info("Realizando análise PCA cross-asset")
        
        # Preparar dados para PCA
        features_data = self._prepare_features_for_pca(market_data)
        
        if len(features_data) == 0:
            return {'status': 'no_data'}
            
        # Normalizar dados
        scaled_data = self.scaler.fit_transform(features_data)
        
        # PCA
        self.pca_model = PCA(n_components=self.config.pca_components)
        pca_components = self.pca_model.fit_transform(scaled_data)
        
        # Criar DataFrame com componentes
        pca_df = pd.DataFrame(
            pca_components,
            index=features_data.index,
            columns=[f'PC{i+1}' for i in range(self.config.pca_components)]
        )
        
        # Análise dos componentes
        component_analysis = self._analyze_pca_components(features_data.columns)
        
        return {
            'pca_components': pca_df,
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.pca_model.explained_variance_ratio_),
            'component_analysis': component_analysis,
            'feature_names': features_data.columns.tolist()
        }
    
    def _prepare_features_for_pca(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepara features para análise PCA
        """
        all_features = []
        
        for category, data in market_data.items():
            if len(data) == 0:
                continue
                
            # Selecionar features relevantes
            feature_cols = []
            
            for col in data.columns:
                if any(suffix in col for suffix in ['_returns', '_volatility', '_rsi', '_momentum', '_bb_position']):
                    feature_cols.append(col)
                    
            if feature_cols:
                category_features = data[feature_cols]
                all_features.append(category_features)
                
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            return combined_features.dropna()
        else:
            return pd.DataFrame()
    
    def _analyze_pca_components(self, feature_names: List[str]) -> Dict[str, Any]:
        """
        Analisa os componentes PCA
        """
        if self.pca_model is None:
            return {}
            
        component_interpretation = {}
        
        for i in range(self.config.pca_components):
            component = self.pca_model.components_[i]
            
            # Top features positivas e negativas
            top_positive = np.argsort(component)[-5:][::-1]
            top_negative = np.argsort(component)[:5]
            
            component_interpretation[f'PC{i+1}'] = {
                'explained_variance': self.pca_model.explained_variance_ratio_[i],
                'top_positive_features': [(feature_names[idx], component[idx]) for idx in top_positive],
                'top_negative_features': [(feature_names[idx], component[idx]) for idx in top_negative]
            }
            
        return component_interpretation
    
    def generate_cross_asset_signals(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Gera sinais cross-asset
        
        Args:
            market_data: Dados de mercado
            
        Returns:
            DataFrame com sinais gerados
        """
        self.logger.info("Gerando sinais cross-asset")
        
        # Análises necessárias
        correlation_analysis = self.analyze_correlations(market_data)
        cointegration_analysis = self.analyze_cointegration(market_data)
        pca_analysis = self.perform_pca_analysis(market_data)
        
        all_signals = []
        
        # 1. Sinais de momentum cross-asset
        momentum_signals = self._generate_momentum_signals(market_data, correlation_analysis)
        all_signals.extend(momentum_signals)
        
        # 2. Sinais de mean reversion
        reversion_signals = self._generate_mean_reversion_signals(market_data, cointegration_analysis)
        all_signals.extend(reversion_signals)
        
        # 3. Sinais de regime change
        regime_signals = self._generate_regime_change_signals(market_data, correlation_analysis)
        all_signals.extend(regime_signals)
        
        # 4. Sinais de PCA
        pca_signals = self._generate_pca_signals(market_data, pca_analysis)
        all_signals.extend(pca_signals)
        
        # 5. Sinais de volatilidade
        volatility_signals = self._generate_volatility_signals(market_data)
        all_signals.extend(volatility_signals)
        
        # 6. Risk-on/Risk-off signals
        risk_signals = self._generate_risk_on_off_signals(market_data)
        all_signals.extend(risk_signals)
        
        # Converter para DataFrame
        if all_signals:
            signals_df = pd.DataFrame(all_signals)
            
            # Salvar sinais no banco
            self._save_signals_to_db(all_signals)
            
            return signals_df
        else:
            return pd.DataFrame()
    
    def _generate_momentum_signals(self, market_data: Dict[str, pd.DataFrame], 
                                 correlation_analysis: Dict[str, Any]) -> List[Dict]:
        """
        Gera sinais de momentum cross-asset
        """
        signals = []
        
        if 'returns_data' not in correlation_analysis:
            return signals
            
        returns_data = correlation_analysis['returns_data']
        
        # Calcular momentum para cada ativo
        momentum_data = {}
        for asset in returns_data.columns:
            momentum = returns_data[asset].rolling(window=self.config.momentum_window).mean()
            momentum_data[asset] = momentum
            
        # Gerar sinais baseados em momentum correlacionado
        for date in returns_data.index[-30:]:  # Últimos 30 dias
            try:
                date_momentum = {asset: momentum_data[asset].loc[date] 
                               for asset in momentum_data.keys() 
                               if not pd.isna(momentum_data[asset].loc[date])}
                
                if len(date_momentum) < 2:
                    continue
                    
                # Identificar ativos com momentum forte
                sorted_momentum = sorted(date_momentum.items(), key=lambda x: x[1], reverse=True)
                
                # Top e bottom performers
                top_performers = sorted_momentum[:3]
                bottom_performers = sorted_momentum[-3:]
                
                # Gerar sinais
                for asset, momentum in top_performers:
                    if momentum > self.config.signal_threshold * 0.01:  # 0.7% threshold
                        signals.append({
                            'date': date,
                            'signal_type': 'cross_asset_momentum',
                            'target_asset': asset,
                            'signal_direction': 1,
                            'signal_strength': min(abs(momentum) * 100, 1.0),
                            'confidence': 0.7,
                            'source_assets': ','.join([a for a, _ in top_performers]),
                            'signal_metadata': json.dumps({
                                'momentum_value': momentum,
                                'rank': 1
                            })
                        })
                        
                for asset, momentum in bottom_performers:
                    if momentum < -self.config.signal_threshold * 0.01:
                        signals.append({
                            'date': date,
                            'signal_type': 'cross_asset_momentum',
                            'target_asset': asset,
                            'signal_direction': -1,
                            'signal_strength': min(abs(momentum) * 100, 1.0),
                            'confidence': 0.7,
                            'source_assets': ','.join([a for a, _ in bottom_performers]),
                            'signal_metadata': json.dumps({
                                'momentum_value': momentum,
                                'rank': -1
                            })
                        })
                        
            except Exception as e:
                self.logger.error(f"Erro ao gerar sinal de momentum para {date}: {e}")
                continue
                
        return signals
    
    def _generate_mean_reversion_signals(self, market_data: Dict[str, pd.DataFrame],
                                       cointegration_analysis: Dict[str, Any]) -> List[Dict]:
        """
        Gera sinais de mean reversion baseados em cointegração
        """
        signals = []
        
        if 'pairwise_cointegration' not in cointegration_analysis:
            return signals
            
        cointegrated_pairs = {
            pair: result for pair, result in cointegration_analysis['pairwise_cointegration'].items()
            if result.get('is_cointegrated', False)
        }
        
        if 'prices_data' not in cointegration_analysis:
            return signals
            
        prices_data = cointegration_analysis['prices_data']
        
        # Gerar sinais para pares cointegrados
        for pair, coint_result in cointegrated_pairs.items():
            try:
                asset1, asset2 = pair.split('_')
                
                if asset1 not in prices_data.columns or asset2 not in prices_data.columns:
                    continue
                    
                # Calcular spread
                spread = prices_data[asset1] - prices_data[asset2]
                
                # Z-score do spread
                spread_mean = spread.rolling(window=self.config.mean_reversion_window * 2).mean()
                spread_std = spread.rolling(window=self.config.mean_reversion_window * 2).std()
                z_score = (spread - spread_mean) / spread_std
                
                # Gerar sinais baseados no z-score
                for date in z_score.index[-30:]:
                    try:
                        current_z = z_score.loc[date]
                        
                        if pd.isna(current_z):
                            continue
                            
                        # Sinal de mean reversion
                        if current_z > 2:  # Spread muito alto, vender asset1, comprar asset2
                            signals.append({
                                'date': date,
                                'signal_type': 'mean_reversion',
                                'target_asset': asset1,
                                'signal_direction': -1,
                                'signal_strength': min(abs(current_z) / 4, 1.0),
                                'confidence': 0.8,
                                'source_assets': f'{asset1},{asset2}',
                                'signal_metadata': json.dumps({
                                    'z_score': current_z,
                                    'cointegration_pvalue': coint_result['pvalue'],
                                    'pair_asset': asset2
                                })
                            })
                            
                        elif current_z < -2:  # Spread muito baixo, comprar asset1, vender asset2
                            signals.append({
                                'date': date,
                                'signal_type': 'mean_reversion',
                                'target_asset': asset1,
                                'signal_direction': 1,
                                'signal_strength': min(abs(current_z) / 4, 1.0),
                                'confidence': 0.8,
                                'source_assets': f'{asset1},{asset2}',
                                'metadata': json.dumps({
                                    'z_score': current_z,
                                    'cointegration_pvalue': coint_result['pvalue'],
                                    'pair_asset': asset2
                                })
                            })
                            
                    except Exception as e:
                        continue
                        
            except Exception as e:
                self.logger.error(f"Erro ao gerar sinal de mean reversion para {pair}: {e}")
                continue
                
        return signals
    
    def _generate_regime_change_signals(self, market_data: Dict[str, pd.DataFrame],
                                      correlation_analysis: Dict[str, Any]) -> List[Dict]:
        """
        Gera sinais de mudança de regime
        """
        signals = []
        
        if 'regime_changes' not in correlation_analysis:
            return signals
            
        regime_changes = correlation_analysis['regime_changes']
        
        # Detectar mudanças recentes (últimos 5 dias)
        recent_date = datetime.now() - timedelta(days=5)
        
        for pair, change_dates in regime_changes.items():
            recent_changes = [date for date in change_dates if pd.to_datetime(date) >= recent_date]
            
            if recent_changes:
                asset1, asset2 = pair.split('_')
                
                for change_date in recent_changes:
                    signals.append({
                        'date': pd.to_datetime(change_date),
                        'signal_type': 'regime_change',
                        'target_asset': asset1,
                        'signal_direction': 0,  # Neutro, apenas alerta
                        'signal_strength': 0.8,
                        'confidence': 0.6,
                        'source_assets': f'{asset1},{asset2}',
                        'metadata': json.dumps({
                            'change_type': 'correlation_regime',
                            'affected_pair': pair
                        })
                    })
                    
        return signals
    
    def _generate_pca_signals(self, market_data: Dict[str, pd.DataFrame],
                            pca_analysis: Dict[str, Any]) -> List[Dict]:
        """
        Gera sinais baseados em análise PCA
        """
        signals = []
        
        if 'pca_components' not in pca_analysis:
            return signals
            
        pca_components = pca_analysis['pca_components']
        
        # Analisar extremos nos componentes principais
        for date in pca_components.index[-30:]:
            try:
                for component in pca_components.columns:
                    value = pca_components.loc[date, component]
                    
                    if pd.isna(value):
                        continue
                        
                    # Calcular z-score do componente
                    component_mean = pca_components[component].rolling(window=60).mean().loc[date]
                    component_std = pca_components[component].rolling(window=60).std().loc[date]
                    
                    if pd.isna(component_mean) or pd.isna(component_std) or component_std == 0:
                        continue
                        
                    z_score = (value - component_mean) / component_std
                    
                    # Gerar sinal se z-score extremo
                    if abs(z_score) > 2:
                        signals.append({
                            'date': date,
                            'signal_type': 'pca_extreme',
                            'target_asset': 'MARKET',  # Sinal de mercado geral
                            'signal_direction': 1 if z_score > 0 else -1,
                            'signal_strength': min(abs(z_score) / 4, 1.0),
                            'confidence': 0.7,
                            'source_assets': 'PCA_ANALYSIS',
                            'signal_metadata': json.dumps({
                                'component': component,
                                'z_score': z_score,
                                'component_value': value
                            })
                        })
                        
            except Exception as e:
                continue
                
        return signals
    
    def _generate_volatility_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Gera sinais baseados em volatilidade cross-asset
        """
        signals = []
        
        # Coletar dados de volatilidade
        volatility_data = []
        asset_names = []
        
        for category, data in market_data.items():
            if len(data) == 0:
                continue
                
            vol_cols = [col for col in data.columns if '_volatility' in col]
            
            for col in vol_cols:
                asset_name = col.replace('_volatility', '')
                volatility_data.append(data[col])
                asset_names.append(asset_name)
                
        if len(volatility_data) == 0:
            return signals
            
        # Combinar dados de volatilidade
        vol_df = pd.concat(volatility_data, axis=1)
        vol_df.columns = asset_names
        vol_df = vol_df.dropna()
        
        # Calcular volatilidade média do mercado
        market_vol = vol_df.mean(axis=1)
        
        # Detectar spikes de volatilidade
        vol_threshold = market_vol.rolling(window=60).quantile(0.9)
        
        for date in market_vol.index[-30:]:
            try:
                current_vol = market_vol.loc[date]
                threshold = vol_threshold.loc[date]
                
                if pd.isna(current_vol) or pd.isna(threshold):
                    continue
                    
                # Spike de volatilidade
                if current_vol > threshold:
                    signals.append({
                        'date': date,
                        'signal_type': 'volatility_spike',
                        'target_asset': 'MARKET',
                        'signal_direction': 0,  # Neutro, alerta de risco
                        'signal_strength': min((current_vol / threshold - 1) * 2, 1.0),
                        'confidence': 0.8,
                        'source_assets': 'VOLATILITY_ANALYSIS',
                        'metadata': json.dumps({
                            'market_volatility': current_vol,
                            'threshold': threshold,
                            'spike_magnitude': current_vol / threshold
                        })
                    })
                    
            except Exception as e:
                continue
                
        return signals
    
    def _generate_risk_on_off_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Gera sinais de risk-on/risk-off
        """
        signals = []
        
        # Indicadores de risk-on/risk-off
        risk_indicators = {}
        
        # VIX (se disponível)
        if 'volatility' in market_data and len(market_data['volatility']) > 0:
            vix_cols = [col for col in market_data['volatility'].columns if 'VIX' in col]
            if vix_cols:
                vix_data = market_data['volatility'][vix_cols[0]]
                risk_indicators['vix'] = vix_data
                
        # Equity indices performance
        if 'equities' in market_data and len(market_data['equities']) > 0:
            equity_returns = []
            for col in market_data['equities'].columns:
                if '_returns' in col:
                    equity_returns.append(market_data['equities'][col])
                    
            if equity_returns:
                avg_equity_return = pd.concat(equity_returns, axis=1).mean(axis=1)
                risk_indicators['equity_performance'] = avg_equity_return
                
        # Safe haven performance (bonds, gold, CHF, JPY)
        safe_haven_returns = []
        
        for category, data in market_data.items():
            if len(data) == 0:
                continue
                
            for col in data.columns:
                if '_returns' in col and any(safe in col.upper() for safe in ['CHF', 'JPY', 'GC=F', 'TNX']):
                    safe_haven_returns.append(data[col])
                    
        if safe_haven_returns:
            avg_safe_haven = pd.concat(safe_haven_returns, axis=1).mean(axis=1)
            risk_indicators['safe_haven_performance'] = avg_safe_haven
            
        # Gerar sinais risk-on/risk-off
        if len(risk_indicators) >= 2:
            for date in list(risk_indicators.values())[0].index[-30:]:
                try:
                    risk_score = 0
                    valid_indicators = 0
                    
                    # VIX: alto = risk-off
                    if 'vix' in risk_indicators:
                        vix_value = risk_indicators['vix'].loc[date]
                        if not pd.isna(vix_value):
                            vix_percentile = stats.percentileofscore(
                                risk_indicators['vix'].dropna().values, vix_value
                            )
                            risk_score += (100 - vix_percentile) / 100  # Inverter: VIX alto = risk-off
                            valid_indicators += 1
                            
                    # Equity performance: positivo = risk-on
                    if 'equity_performance' in risk_indicators:
                        equity_return = risk_indicators['equity_performance'].loc[date]
                        if not pd.isna(equity_return):
                            if equity_return > 0:
                                risk_score += 1
                            else:
                                risk_score -= 1
                            valid_indicators += 1
                            
                    # Safe haven: performance negativa = risk-on
                    if 'safe_haven_performance' in risk_indicators:
                        safe_haven_return = risk_indicators['safe_haven_performance'].loc[date]
                        if not pd.isna(safe_haven_return):
                            if safe_haven_return < 0:
                                risk_score += 1
                            else:
                                risk_score -= 1
                            valid_indicators += 1
                            
                    if valid_indicators > 0:
                        normalized_risk_score = risk_score / valid_indicators
                        
                        # Gerar sinal se score extremo
                        if abs(normalized_risk_score) > 0.5:
                            signals.append({
                                'date': date,
                                'signal_type': 'risk_on_off',
                                'target_asset': 'MARKET',
                                'signal_direction': 1 if normalized_risk_score > 0 else -1,
                                'signal_strength': min(abs(normalized_risk_score), 1.0),
                                'confidence': 0.7,
                                'source_assets': 'RISK_SENTIMENT_ANALYSIS',
                                'signal_metadata': json.dumps({
                                    'risk_score': normalized_risk_score,
                                    'regime': 'risk_on' if normalized_risk_score > 0 else 'risk_off',
                                    'valid_indicators': valid_indicators
                                })
                            })
                            
                except Exception as e:
                    continue
                    
        return signals
    
    def _save_signals_to_db(self, signals: List[Dict]) -> None:
        """
        Salva sinais no banco de dados
        """
        if not self.session:
            return
            
        try:
            for signal_data in signals:
                signal = CrossAssetSignal(
                    date=signal_data['date'],
                    signal_type=signal_data['signal_type'],
                    source_assets=signal_data['source_assets'],
                    target_asset=signal_data['target_asset'],
                    signal_strength=signal_data['signal_strength'],
                    signal_direction=signal_data['signal_direction'],
                    confidence=signal_data['confidence'],
                    signal_metadata=signal_data.get('signal_metadata', '')
                )
                
                self.session.add(signal)
                
            self.session.commit()
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar sinais: {e}")
            self.session.rollback()
    
    def _save_market_data_to_db(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """
        Salva dados de mercado no banco
        """
        if not self.session:
            return
            
        try:
            for category, data in market_data.items():
                if len(data) == 0:
                    continue
                    
                # Extrair dados por ativo
                for col in data.columns:
                    if '_close' in col:
                        symbol = col.replace('_close', '')
                        
                        # Colunas relacionadas
                        related_cols = {
                            'close': f'{symbol}_close',
                            'high': f'{symbol}_high',
                            'low': f'{symbol}_low',
                            'volume': f'{symbol}_volume',
                            'returns': f'{symbol}_returns',
                            'log_returns': f'{symbol}_log_returns',
                            'volatility': f'{symbol}_volatility'
                        }
                        
                        # Salvar dados por data
                        for date, row in data.iterrows():
                            if pd.isna(row.get(related_cols['close'])):
                                continue
                                
                            asset_data = AssetData(
                                symbol=symbol,
                                date=date,
                                close_price=row.get(related_cols['close']),
                                high_price=row.get(related_cols.get('high')),
                                low_price=row.get(related_cols.get('low')),
                                volume=row.get(related_cols.get('volume')),
                                returns=row.get(related_cols.get('returns')),
                                log_returns=row.get(related_cols.get('log_returns')),
                                volatility=row.get(related_cols.get('volatility'))
                            )
                            
                            self.session.add(asset_data)
                            
            self.session.commit()
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar dados de mercado: {e}")
            self.session.rollback()
    
    def _save_models(self) -> None:
        """
        Salva modelos treinados
        """
        try:
            models_data = {
                'pca_model': self.pca_model,
                'factor_model': self.factor_model,
                'clustering_model': self.clustering_model,
                'scaler': self.scaler,
                'trained_at': datetime.now()
            }
            
            with open('cross_asset_models.pkl', 'wb') as f:
                pickle.dump(models_data, f)
                
            self.logger.info("Modelos cross-asset salvos")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelos: {e}")
    
    def _load_models(self) -> None:
        """
        Carrega modelos salvos
        """
        try:
            if Path('cross_asset_models.pkl').exists():
                with open('cross_asset_models.pkl', 'rb') as f:
                    models_data = pickle.load(f)
                    
                self.pca_model = models_data.get('pca_model')
                self.factor_model = models_data.get('factor_model')
                self.clustering_model = models_data.get('clustering_model')
                self.scaler = models_data.get('scaler', StandardScaler())
                
                self.logger.info("Modelos cross-asset carregados")
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelos: {e}")
    
    def plot_cross_asset_analysis(self, market_data: Dict[str, pd.DataFrame],
                                correlation_analysis: Dict[str, Any],
                                save_plot: bool = False) -> None:
        """
        Plota análise cross-asset
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Cross-Asset Analysis', fontsize=16)
        
        # 1. Heatmap de correlação
        if 'static_correlation' in correlation_analysis:
            ax1 = axes[0, 0]
            corr_matrix = correlation_analysis['static_correlation']
            sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0, ax=ax1)
            ax1.set_title('Asset Correlation Matrix')
            
        # 2. Evolução das correlações
        if 'dynamic_correlation' in correlation_analysis:
            ax2 = axes[0, 1]
            dynamic_corr = correlation_analysis['dynamic_correlation']
            
            # Plotar algumas correlações principais
            for i, (pair, corr_series) in enumerate(list(dynamic_corr.items())[:5]):
                ax2.plot(corr_series.index, corr_series.values, label=pair, alpha=0.7)
                
            ax2.set_title('Dynamic Correlations')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
        # 3. Returns distribution por categoria
        ax3 = axes[0, 2]
        
        for category, data in market_data.items():
            if len(data) == 0:
                continue
                
            return_cols = [col for col in data.columns if '_returns' in col and '_log_returns' not in col]
            
            if return_cols:
                category_returns = data[return_cols].mean(axis=1)
                ax3.hist(category_returns.dropna(), bins=30, alpha=0.6, label=category)
                
        ax3.set_title('Returns Distribution by Category')
        ax3.legend()
        ax3.set_xlabel('Returns')
        ax3.set_ylabel('Frequency')
        
        # 4. Volatilidade por categoria
        ax4 = axes[1, 0]
        
        for category, data in market_data.items():
            if len(data) == 0:
                continue
                
            vol_cols = [col for col in data.columns if '_volatility' in col]
            
            if vol_cols:
                category_vol = data[vol_cols].mean(axis=1)
                ax4.plot(category_vol.index, category_vol.values, label=category, alpha=0.8)
                
        ax4.set_title('Volatility by Category')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylabel('Volatility')
        
        # 5. Clusters de correlação
        if 'correlation_clusters' in correlation_analysis:
            ax5 = axes[1, 1]
            clusters = correlation_analysis['correlation_clusters']['clusters']
            
            cluster_text = ""
            for cluster_id, assets in clusters.items():
                cluster_text += f"Cluster {cluster_id}:\n"
                cluster_text += "\n".join(assets[:5])  # Primeiros 5 ativos
                cluster_text += "\n\n"
                
            ax5.text(0.1, 0.9, cluster_text, transform=ax5.transAxes, 
                    verticalalignment='top', fontsize=8)
            ax5.set_title('Correlation Clusters')
            ax5.axis('off')
            
        # 6. PCA explained variance
        ax6 = axes[1, 2]
        
        if self.pca_model is not None:
            explained_var = self.pca_model.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            ax6.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, label='Individual')
            ax6.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-', label='Cumulative')
            ax6.set_title('PCA Explained Variance')
            ax6.set_xlabel('Principal Component')
            ax6.set_ylabel('Explained Variance Ratio')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('cross_asset_analysis.png', dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def get_signal_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Retorna resumo dos sinais gerados
        
        Args:
            days_back: Número de dias para análise
            
        Returns:
            Resumo dos sinais
        """
        if not self.session:
            return {'status': 'no_database'}
            
        try:
            # Data de corte
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Query sinais recentes
            recent_signals = self.session.query(CrossAssetSignal).filter(
                CrossAssetSignal.date >= cutoff_date,
                CrossAssetSignal.is_active == True
            ).all()
            
            if not recent_signals:
                return {'status': 'no_signals'}
                
            # Análise por tipo de sinal
            signal_types = defaultdict(int)
            signal_directions = defaultdict(int)
            avg_confidence = defaultdict(list)
            avg_strength = defaultdict(list)
            
            for signal in recent_signals:
                signal_types[signal.signal_type] += 1
                
                if signal.signal_direction > 0:
                    signal_directions['bullish'] += 1
                elif signal.signal_direction < 0:
                    signal_directions['bearish'] += 1
                else:
                    signal_directions['neutral'] += 1
                    
                avg_confidence[signal.signal_type].append(signal.confidence)
                avg_strength[signal.signal_type].append(signal.signal_strength)
                
            # Calcular médias
            confidence_by_type = {
                signal_type: np.mean(confidences)
                for signal_type, confidences in avg_confidence.items()
            }
            
            strength_by_type = {
                signal_type: np.mean(strengths)
                for signal_type, strengths in avg_strength.items()
            }
            
            return {
                'total_signals': len(recent_signals),
                'signal_types': dict(signal_types),
                'signal_directions': dict(signal_directions),
                'avg_confidence_by_type': confidence_by_type,
                'avg_strength_by_type': strength_by_type,
                'period_days': days_back
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar resumo de sinais: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_active_signals(self, signal_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retorna sinais ativos
        
        Args:
            signal_types: Tipos de sinais para filtrar
            
        Returns:
            DataFrame com sinais ativos
        """
        if not self.session:
            return pd.DataFrame()
            
        try:
            query = self.session.query(CrossAssetSignal).filter(
                CrossAssetSignal.is_active == True
            )
            
            if signal_types:
                query = query.filter(CrossAssetSignal.signal_type.in_(signal_types))
                
            signals = query.order_by(CrossAssetSignal.date.desc()).limit(100).all()
            
            if not signals:
                return pd.DataFrame()
                
            # Converter para DataFrame
            signal_data = []
            for signal in signals:
                signal_data.append({
                    'date': signal.date,
                    'signal_type': signal.signal_type,
                    'target_asset': signal.target_asset,
                    'signal_direction': signal.signal_direction,
                    'signal_strength': signal.signal_strength,
                    'confidence': signal.confidence,
                    'source_assets': signal.source_assets,
                    'signal_metadata': signal.signal_metadata
                })
                
            return pd.DataFrame(signal_data)
            
        except Exception as e:
            self.logger.error(f"Erro ao buscar sinais ativos: {e}")
            return pd.DataFrame()
    
    def cleanup_old_signals(self, days_to_keep: int = 90) -> None:
        """
        Remove sinais antigos do banco
        
        Args:
            days_to_keep: Dias de sinais para manter
        """
        if not self.session:
            return
            
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            deleted_count = self.session.query(CrossAssetSignal).filter(
                CrossAssetSignal.date < cutoff_date
            ).delete()
            
            self.session.commit()
            
            self.logger.info(f"Removidos {deleted_count} sinais antigos")
            
        except Exception as e:
            self.logger.error(f"Erro ao limpar sinais antigos: {e}")
            self.session.rollback()
    
    def run_full_analysis(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Executa análise completa cross-asset
        
        Args:
            save_results: Se deve salvar resultados
            
        Returns:
            Resultados da análise completa
        """
        self.logger.info("Iniciando análise cross-asset completa")
        
        try:
            # 1. Coletar dados de mercado
            market_data = self.collect_market_data()
            
            if not any(len(data) > 0 for data in market_data.values()):
                return {'status': 'no_market_data'}
                
            # 2. Análise de correlações
            correlation_analysis = self.analyze_correlations(market_data)
            
            # 3. Análise de cointegração
            cointegration_analysis = self.analyze_cointegration(market_data)
            
            # 4. Análise PCA
            pca_analysis = self.perform_pca_analysis(market_data)
            
            # 5. Gerar sinais
            signals_df = self.generate_cross_asset_signals(market_data)
            
            # 6. Salvar modelos
            if save_results:
                self._save_models()
                
            # 7. Resumo dos resultados
            results = {
                'status': 'success',
                'market_data_summary': {
                    category: {'assets': len([col for col in data.columns if '_close' in col]),
                              'data_points': len(data)}
                    for category, data in market_data.items()
                },
                'correlation_analysis': {
                    'assets_analyzed': len(correlation_analysis.get('static_correlation', pd.DataFrame())),
                    'avg_correlation': correlation_analysis.get('static_correlation', pd.DataFrame()).mean().mean() if 'static_correlation' in correlation_analysis else None
                },
                'cointegration_analysis': {
                    'pairs_tested': len(cointegration_analysis.get('pairwise_cointegration', {})),
                    'cointegrated_pairs': sum(1 for result in cointegration_analysis.get('pairwise_cointegration', {}).values() if result.get('is_cointegrated', False))
                },
                'pca_analysis': {
                    'components': len(pca_analysis.get('pca_components', pd.DataFrame()).columns),
                    'total_variance_explained': sum(pca_analysis.get('explained_variance_ratio', []))
                },
                'signals_generated': len(signals_df),
                'analysis_timestamp': datetime.now()
            }
            
            # 8. Plot análise (opcional)
            if correlation_analysis.get('status') != 'no_data':
                self.plot_cross_asset_analysis(market_data, correlation_analysis, save_plot=save_results)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Erro na análise completa: {e}")
            return {'status': 'error', 'message': str(e)}

# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Criar configuração
    config = CrossAssetConfig(
        lookback_period=180,  # 6 meses
        correlation_window=30,
        enable_parallel=True,
        max_workers=4
    )
    
    # Inicializar analisador
    analyzer = CrossAssetAnalyzer(config=config)
    
    try:
        # Executar análise completa
        results = analyzer.run_full_analysis(save_results=True)
        
        print("\n=== RESULTADOS DA ANÁLISE CROSS-ASSET ===")
        print(f"Status: {results['status']}")
        
        if results['status'] == 'success':
            print(f"\nDados de Mercado:")
            for category, summary in results['market_data_summary'].items():
                print(f"  {category}: {summary['assets']} ativos, {summary['data_points']} pontos")
                
            print(f"\nAnálise de Correlação:")
            corr_summary = results['correlation_analysis']
            print(f"  Ativos analisados: {corr_summary['assets_analyzed']}")
            if corr_summary['avg_correlation']:
                print(f"  Correlação média: {corr_summary['avg_correlation']:.3f}")
                
            print(f"\nAnálise de Cointegração:")
            coint_summary = results['cointegration_analysis']
            print(f"  Pares testados: {coint_summary['pairs_tested']}")
            print(f"  Pares cointegrados: {coint_summary['cointegrated_pairs']}")
            
            print(f"\nAnálise PCA:")
            pca_summary = results['pca_analysis']
            print(f"  Componentes: {pca_summary['components']}")
            if pca_summary['total_variance_explained']:
                print(f"  Variância explicada: {pca_summary['total_variance_explained']:.1%}")
                
            print(f"\nSinais Gerados: {results['signals_generated']}")
            
            # Mostrar resumo de sinais
            signal_summary = analyzer.get_signal_summary(days_back=7)
            if signal_summary.get('status') != 'no_signals':
                print(f"\nResumo de Sinais (últimos 7 dias):")
                print(f"  Total: {signal_summary['total_signals']}")
                print(f"  Por tipo: {signal_summary['signal_types']}")
                print(f"  Por direção: {signal_summary['signal_directions']}")
                
            # Mostrar sinais ativos
            active_signals = analyzer.get_active_signals()
            if len(active_signals) > 0:
                print(f"\nSinais Ativos Recentes:")
                for _, signal in active_signals.head(5).iterrows():
                    direction = "BUY" if signal['signal_direction'] > 0 else "SELL" if signal['signal_direction'] < 0 else "NEUTRAL"
                    print(f"  {signal['date'].strftime('%Y-%m-%d')} | {signal['signal_type']} | {signal['target_asset']} | {direction} | Força: {signal['signal_strength']:.2f} | Confiança: {signal['confidence']:.2f}")
        else:
            print(f"Erro: {results.get('message', 'Erro desconhecido')}")
            
    except Exception as e:
        print(f"Erro na execução: {e}")
        
    finally:
        # Cleanup
        analyzer.cleanup_old_signals(days_to_keep=60)
        
    print("\nAnálise cross-asset concluída!")

# Alias para compatibilidade com testes
CrossAssetSignalGenerator = CrossAssetAnalyzer