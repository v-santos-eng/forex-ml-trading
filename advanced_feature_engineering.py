#!/usr/bin/env python3
"""
Feature Engineering Avançado para Trading Algorítmico
Baseado nos conhecimentos extraídos do livro "Machine Learning for Algorithmic Trading"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import talib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Classe para feature engineering avançado baseado em conhecimentos
    extraídos do livro de ML para Algorithmic Trading
    """
    
    def __init__(self, lookback_periods: List[int] = [5, 10, 20, 50]):
        self.lookback_periods = lookback_periods
        self.scalers = {}
        self.feature_names = []
        
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria indicadores técnicos avançados mencionados no livro
        """
        print("Criando indicadores técnicos avançados...")
        
        # Preços básicos (garantir que usamos as colunas corretas)
        high = df['High'].values if 'High' in df.columns else df['high'].values
        low = df['Low'].values if 'Low' in df.columns else df['low'].values
        close = df['Close'].values if 'Close' in df.columns else df['close'].values
        volume = df['Volume'].values if 'Volume' in df.columns else (df['volume'].values if 'volume' in df.columns else np.ones(len(df)))
        
        features = df.copy()
        
        # 1. Indicadores de Momentum
        for period in self.lookback_periods:
            # RSI com múltiplos períodos
            features[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)
            
            # MACD
            if period >= 12:
                macd, macd_signal, macd_hist = talib.MACD(close, 
                                                         fastperiod=12, 
                                                         slowperiod=period, 
                                                         signalperiod=9)
                features[f'macd_{period}'] = macd
                features[f'macd_signal_{period}'] = macd_signal
                features[f'macd_histogram_{period}'] = macd_hist
            
            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(high, low, close, 
                                     fastk_period=period, 
                                     slowk_period=3, 
                                     slowd_period=3)
            features[f'stoch_k_{period}'] = slowk
            features[f'stoch_d_{period}'] = slowd
            
            # Williams %R
            features[f'williams_r_{period}'] = talib.WILLR(high, low, close, timeperiod=period)
        
        # 2. Indicadores de Volatilidade
        for period in self.lookback_periods:
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=period)
            features[f'bb_upper_{period}'] = bb_upper
            features[f'bb_middle_{period}'] = bb_middle
            features[f'bb_lower_{period}'] = bb_lower
            features[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
            features[f'bb_position_{period}'] = (close - bb_lower) / (bb_upper - bb_lower)
            
            # Average True Range
            features[f'atr_{period}'] = talib.ATR(high, low, close, timeperiod=period)
            
            # Volatilidade realizada
            returns = np.log(close / np.roll(close, 1))
            features[f'realized_vol_{period}'] = pd.Series(returns).rolling(period).std() * np.sqrt(252)
            
        # 3. Volume features (se disponível)
        if 'Volume' in df.columns or 'volume' in df.columns:
            volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
            volume_data = df[volume_col].values
            for period in self.lookback_periods:
                features[f'volume_sma_{period}'] = talib.SMA(volume_data.astype(float), timeperiod=period)
                features[f'volume_ratio_{period}'] = volume_data / talib.SMA(volume_data.astype(float), timeperiod=period)
                
                # VWAP
                typical_price = (high + low + close) / 3
                features[f'vwap_{period}'] = (pd.Series(typical_price) * df[volume_col]).rolling(period).sum() / df[volume_col].rolling(period).sum()
        
        # 3. Indicadores de Tendência
        for period in self.lookback_periods:
            # Moving Averages
            features[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
            features[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
            features[f'wma_{period}'] = talib.WMA(close, timeperiod=period)
            
            # ADX (Average Directional Index)
            features[f'adx_{period}'] = talib.ADX(high, low, close, timeperiod=period)
            features[f'plus_di_{period}'] = talib.PLUS_DI(high, low, close, timeperiod=period)
            features[f'minus_di_{period}'] = talib.MINUS_DI(high, low, close, timeperiod=period)
            
            # Parabolic SAR
            features[f'sar_{period}'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        
        return features
    
    def create_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features baseadas em price action
        """
        print("Criando features de price action...")
        
        features = df.copy()
        
        # 1. Candlestick Patterns (usando TA-Lib)
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        open_col = 'Open' if 'Open' in df.columns else 'open'
        close_col = 'Close' if 'Close' in df.columns else 'close'
        
        high = df[high_col].values
        low = df[low_col].values
        open_price = df[open_col].values
        close = df[close_col].values
        
        # Padrões de candlestick
        features['doji'] = talib.CDLDOJI(open_price, high, low, close)
        features['hammer'] = talib.CDLHAMMER(open_price, high, low, close)
        features['hanging_man'] = talib.CDLHANGINGMAN(open_price, high, low, close)
        features['engulfing'] = talib.CDLENGULFING(open_price, high, low, close)
        features['morning_star'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
        features['evening_star'] = talib.CDLEVENINGSTAR(open_price, high, low, close)
        
        # 2. Price Action Metrics
        for period in self.lookback_periods:
            # Body size
            features[f'body_size_{period}'] = abs(close - open_price) / (high - low)
            
            # Upper/Lower shadows
            features[f'upper_shadow_{period}'] = (high - np.maximum(open_price, close)) / (high - low)
            features[f'lower_shadow_{period}'] = (np.minimum(open_price, close) - low) / (high - low)
            
            # Price position within range
            features[f'price_position_{period}'] = (close - low) / (high - low)
            
            # Gap analysis
            prev_close = np.roll(close, 1)
            features[f'gap_{period}'] = (open_price - prev_close) / prev_close
        
        return features
    
    def create_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de microestrutura de mercado
        """
        print("Criando features de microestrutura...")
        
        features = df.copy()
        close = df['Close'].values if 'Close' in df.columns else df['close'].values
        
        # Primeiro criar return_1 que será usado em outros cálculos
        features['return_1'] = np.log(close / np.roll(close, 1))
        
        # 1. Returns em múltiplos horizontes
        for period in self.lookback_periods:
            if period != 1:  # return_1 já foi criado
                features[f'return_{period}'] = np.log(close / np.roll(close, period))
            
            # Cumulative returns
            features[f'cum_return_{period}'] = pd.Series(features['return_1']).rolling(period).sum()
            
            # Return autocorrelation
            returns_1 = features['return_1']
            features[f'return_autocorr_{period}'] = pd.Series(returns_1).rolling(period).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
            )
        
        # 2. Volatility clustering
        returns_1 = features['return_1']
        for period in self.lookback_periods:
            # GARCH-like volatility
            features[f'vol_clustering_{period}'] = pd.Series(returns_1**2).rolling(period).mean()
            
            # Volatility of volatility
            vol = pd.Series(returns_1).rolling(period).std()
            features[f'vol_of_vol_{period}'] = vol.rolling(period).std()
        
        # 3. Regime detection features
        for period in self.lookback_periods:
            # Trend strength
            sma = pd.Series(close).rolling(period).mean()
            features[f'trend_strength_{period}'] = (close - sma) / sma
            
            # Market regime (trending vs ranging)
            high_col = 'High' if 'High' in df.columns else 'high'
            low_col = 'Low' if 'Low' in df.columns else 'low'
            high_period = pd.Series(df[high_col]).rolling(period).max()
            low_period = pd.Series(df[low_col]).rolling(period).min()
            features[f'market_regime_{period}'] = (high_period - low_period) / sma
        
        return features
    
    def create_cross_asset_features(self, df: pd.DataFrame, other_pairs: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """
        Cria features baseadas em correlações entre ativos
        """
        print("Criando features cross-asset...")
        
        features = df.copy()
        
        if other_pairs:
            close = df['close']
            
            for pair_name, pair_df in other_pairs.items():
                if len(pair_df) == len(df):
                    other_close = pair_df['close']
                    
                    for period in self.lookback_periods:
                        # Correlação rolling
                        corr = pd.Series(close).rolling(period).corr(pd.Series(other_close))
                        features[f'corr_{pair_name}_{period}'] = corr
                        
                        # Beta rolling
                        returns_self = np.log(close / close.shift(1))
                        returns_other = np.log(other_close / other_close.shift(1))
                        
                        cov = pd.Series(returns_self).rolling(period).cov(pd.Series(returns_other))
                        var_other = pd.Series(returns_other).rolling(period).var()
                        features[f'beta_{pair_name}_{period}'] = cov / var_other
        
        return features
    
    def create_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features baseadas em tempo
        """
        print("Criando features temporais...")
        
        features = df.copy()
        
        # Assumindo que o índice é datetime
        if hasattr(df.index, 'hour'):
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['month'] = df.index.month
            
            # Sessões de trading
            features['asian_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
            features['european_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
            features['american_session'] = ((df.index.hour >= 16) & (df.index.hour < 24)).astype(int)
            
            # Overlaps de sessões
            features['euro_american_overlap'] = ((df.index.hour >= 13) & (df.index.hour < 16)).astype(int)
            features['asian_european_overlap'] = ((df.index.hour >= 7) & (df.index.hour < 9)).astype(int)
        
        return features
    
    def select_best_features(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        Seleciona as melhores features usando métodos estatísticos
        """
        print(f"Selecionando as {k} melhores features...")
        
        # Remove features com muitos NaN
        X_clean = X.dropna(axis=1, thresh=len(X) * 0.7)
        
        # Remove features constantes
        X_clean = X_clean.loc[:, X_clean.var() > 1e-6]
        
        # Alinha X e y
        common_idx = X_clean.index.intersection(y.index)
        X_aligned = X_clean.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        # Remove NaN restantes
        mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
        X_final = X_aligned[mask]
        y_final = y_aligned[mask]
        
        if len(X_final) == 0:
            print("Aviso: Nenhuma observação válida após limpeza")
            return X, []
        
        # Seleção de features
        selector = SelectKBest(score_func=f_regression, k=min(k, X_final.shape[1]))
        X_selected = selector.fit_transform(X_final, y_final)
        
        # Nomes das features selecionadas
        selected_features = X_final.columns[selector.get_support()].tolist()
        
        print(f"Features selecionadas: {len(selected_features)}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X_final.index), selected_features
    
    def engineer_features(self, df: pd.DataFrame, target_col: str = 'target', 
                         other_pairs: Dict[str, pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Pipeline completo de feature engineering
        """
        print("=== INICIANDO FEATURE ENGINEERING AVANÇADO ===")
        
        # 1. Indicadores técnicos
        features = self.create_technical_indicators(df)
        
        # 2. Price action
        features = self.create_price_action_features(features)
        
        # 3. Microestrutura
        features = self.create_market_microstructure_features(features)
        
        # 4. Cross-asset (se disponível)
        if other_pairs:
            features = self.create_cross_asset_features(features, other_pairs)
        
        # 5. Features temporais
        features = self.create_time_based_features(features)
        
        # 6. Criar target se não existir
        if target_col not in features.columns:
            # Target simples: retorno futuro - usar colunas corretas
            close_col = 'Close' if 'Close' in features.columns else 'close'
            features[target_col] = np.log(features[close_col].shift(-1) / features[close_col])
        
        # 7. Separar features e target
        target = features[target_col]
        # Excluir colunas originais (tanto maiúsculas quanto minúsculas)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume', target_col]
        feature_cols = [col for col in features.columns if col not in exclude_cols]
        X = features[feature_cols]
        
        print(f"Features criadas: {len(feature_cols)}")
        print(f"Observações: {len(X)}")
        
        # 8. Seleção de features
        X_selected, selected_features = self.select_best_features(X, target)
        
        self.feature_names = selected_features
        
        print("=== FEATURE ENGINEERING CONCLUÍDO ===")
        
        return X_selected, target

def demonstrate_advanced_features():
    """
    Demonstração do feature engineering avançado
    """
    print("=== DEMONSTRAÇÃO - FEATURE ENGINEERING AVANÇADO ===")
    
    # Criar dados sintéticos para demonstração
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    
    # Simular dados OHLC
    base_price = 1.1000
    returns = np.random.normal(0, 0.001, 1000)
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.0005, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.0005, 1000))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Ajustar high/low
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
    
    # Criar feature engineer
    fe = AdvancedFeatureEngineer(lookback_periods=[5, 10, 20])
    
    # Aplicar feature engineering
    X, y = fe.engineer_features(df)
    
    print(f"\nResultados:")
    print(f"Features finais: {X.shape[1]}")
    print(f"Observações válidas: {X.shape[0]}")
    print(f"\nPrimeiras 10 features:")
    for i, feature in enumerate(fe.feature_names[:10]):
        print(f"{i+1}. {feature}")
    
    return X, y, fe

if __name__ == "__main__":
    # Instalar TA-Lib se necessário
    try:
        import talib
    except ImportError:
        print("TA-Lib não encontrado. Instale com: pip install TA-Lib")
        print("Ou use: conda install -c conda-forge ta-lib")
        exit(1)
    
    demonstrate_advanced_features()