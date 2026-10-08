import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Tuple
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import talib
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt


class FinancialFeatureEngineer:
    """
    Classe para engenharia de features avançada em dados financeiros.
    
    Implementa técnicas baseadas em López de Prado e outras metodologias
    quantitativas para criar features preditivas robustas.
    """
    
    def __init__(self, 
                 lookback_periods: List[int] = None,
                 volatility_windows: List[int] = None):
        """
        Inicializa o engenheiro de features.
        
        Args:
            lookback_periods: Períodos de lookback para features temporais
            volatility_windows: Janelas para cálculo de volatilidade
        """
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 100]
        self.volatility_windows = volatility_windows or [10, 20, 50]
        self.scalers = {}
        self.feature_names = []
    
    def create_price_features(self, 
                             prices: pd.DataFrame,
                             volume: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Cria features baseadas em preços.
        
        Args:
            prices: DataFrame com OHLCV
            volume: Série de volume (opcional se já em prices)
            
        Returns:
            DataFrame com features de preço
        """
        features = pd.DataFrame(index=prices.index)
        
        # Assumir colunas padrão se não especificadas
        if 'close' in prices.columns:
            close = prices['close']
        elif 'Close' in prices.columns:
            close = prices['Close']
        else:
            close = prices.iloc[:, 0]  # Primeira coluna como close
        
        # Features básicas de retorno
        features['returns'] = close.pct_change()
        features['log_returns'] = np.log(close / close.shift(1))
        
        # Features de retorno em múltiplos períodos
        for period in self.lookback_periods:
            features[f'returns_{period}d'] = close.pct_change(period)
            features[f'log_returns_{period}d'] = np.log(close / close.shift(period))
        
        # Features de momentum
        for period in self.lookback_periods:
            features[f'momentum_{period}d'] = close / close.shift(period) - 1
            features[f'price_position_{period}d'] = (close - close.rolling(period).min()) / (
                close.rolling(period).max() - close.rolling(period).min())
        
        # Features de volatilidade
        for window in self.volatility_windows:
            features[f'volatility_{window}d'] = features['returns'].rolling(window).std()
            features[f'realized_vol_{window}d'] = np.sqrt(
                (features['log_returns'] ** 2).rolling(window).sum())
        
        # Features de médias móveis
        for period in self.lookback_periods:
            sma = close.rolling(period).mean()
            features[f'sma_{period}d'] = sma
            features[f'price_to_sma_{period}d'] = close / sma - 1
            
            # EMA
            ema = close.ewm(span=period).mean()
            features[f'ema_{period}d'] = ema
            features[f'price_to_ema_{period}d'] = close / ema - 1
        
        # Features de volume se disponível
        if volume is not None or 'volume' in prices.columns:
            vol = volume if volume is not None else prices['volume']
            
            # Volume features
            features['volume'] = vol
            features['volume_sma_20'] = vol.rolling(20).mean()
            features['volume_ratio'] = vol / features['volume_sma_20']
            
            # Price-Volume features
            features['vwap_20'] = (close * vol).rolling(20).sum() / vol.rolling(20).sum()
            features['price_to_vwap'] = close / features['vwap_20'] - 1
        
        return features
    
    def create_technical_indicators(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Cria indicadores técnicos usando TA-Lib.
        
        Args:
            prices: DataFrame com OHLCV
            
        Returns:
            DataFrame com indicadores técnicos
        """
        features = pd.DataFrame(index=prices.index)
        
        # Extrair séries necessárias
        high = prices['high'] if 'high' in prices.columns else prices.iloc[:, 1]
        low = prices['low'] if 'low' in prices.columns else prices.iloc[:, 2]
        close = prices['close'] if 'close' in prices.columns else prices.iloc[:, 3]
        volume = prices['volume'] if 'volume' in prices.columns else None
        
        try:
            # Momentum Indicators
            features['rsi_14'] = talib.RSI(close, timeperiod=14)
            features['rsi_30'] = talib.RSI(close, timeperiod=30)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            features['stoch_k'] = stoch_k
            features['stoch_d'] = stoch_d
            
            # Williams %R
            features['williams_r'] = talib.WILLR(high, low, close)
            
            # Average True Range
            features['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            
            # Commodity Channel Index
            features['cci_14'] = talib.CCI(high, low, close, timeperiod=14)
            
            # Money Flow Index (se volume disponível)
            if volume is not None:
                features['mfi_14'] = talib.MFI(high, low, close, volume, timeperiod=14)
                
                # On Balance Volume
                features['obv'] = talib.OBV(close, volume)
                features['obv_sma_20'] = features['obv'].rolling(20).mean()
            
        except Exception as e:
            warnings.warn(f"Erro ao calcular indicadores técnicos: {e}")
        
        return features
    
    def create_statistical_features(self, 
                                  prices: pd.DataFrame,
                                  returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Cria features estatísticas avançadas.
        
        Args:
            prices: DataFrame com preços
            returns: Série de retornos (opcional)
            
        Returns:
            DataFrame com features estatísticas
        """
        features = pd.DataFrame(index=prices.index)
        
        close = prices['close'] if 'close' in prices.columns else prices.iloc[:, 0]
        
        if returns is None:
            returns = close.pct_change()
        
        # Features de distribuição de retornos
        for window in self.volatility_windows:
            rolling_returns = returns.rolling(window)
            
            # Momentos estatísticos
            features[f'skewness_{window}d'] = rolling_returns.skew()
            features[f'kurtosis_{window}d'] = rolling_returns.kurt()
            
            # Percentis
            features[f'returns_q25_{window}d'] = rolling_returns.quantile(0.25)
            features[f'returns_q75_{window}d'] = rolling_returns.quantile(0.75)
            features[f'returns_iqr_{window}d'] = (features[f'returns_q75_{window}d'] - 
                                                 features[f'returns_q25_{window}d'])
            
            # Drawdown
            rolling_max = close.rolling(window).max()
            features[f'drawdown_{window}d'] = (close - rolling_max) / rolling_max
            
            # Sharpe ratio rolling
            mean_ret = rolling_returns.mean()
            std_ret = rolling_returns.std()
            features[f'sharpe_{window}d'] = mean_ret / std_ret * np.sqrt(252)
        
        # Features de autocorrelação
        for lag in [1, 5, 10]:
            features[f'autocorr_lag{lag}'] = returns.rolling(50).apply(
                lambda x: x.autocorr(lag) if len(x) > lag else np.nan)
        
        # Hurst Exponent (simplified)
        def hurst_exponent(ts, max_lag=20):
            """Calcula expoente de Hurst simplificado."""
            if len(ts) < max_lag * 2:
                return np.nan
            
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            
            try:
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            except:
                return np.nan
        
        features['hurst_50d'] = close.rolling(50).apply(hurst_exponent)
        
        return features
    
    def create_microstructure_features(self, 
                                     prices: pd.DataFrame,
                                     trades: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Cria features de microestrutura de mercado.
        
        Args:
            prices: DataFrame com OHLCV
            trades: DataFrame com dados de trades (opcional)
            
        Returns:
            DataFrame com features de microestrutura
        """
        features = pd.DataFrame(index=prices.index)
        
        high = prices['high'] if 'high' in prices.columns else prices.iloc[:, 1]
        low = prices['low'] if 'low' in prices.columns else prices.iloc[:, 2]
        close = prices['close'] if 'close' in prices.columns else prices.iloc[:, 3]
        volume = prices['volume'] if 'volume' in prices.columns else None
        
        # Spread estimado (High-Low)
        features['hl_spread'] = (high - low) / close
        features['hl_spread_ma'] = features['hl_spread'].rolling(20).mean()
        
        # Volatility clustering
        returns = close.pct_change()
        features['vol_clustering'] = (returns.abs().rolling(10).mean() / 
                                    returns.abs().rolling(50).mean())
        
        # Intraday patterns
        if prices.index.freq is not None:
            # Hour of day effect (se dados intraday)
            if hasattr(prices.index, 'hour'):
                features['hour'] = prices.index.hour
                features['is_market_open'] = ((features['hour'] >= 9) & 
                                            (features['hour'] <= 16)).astype(int)
        
        # Jump detection
        returns_std = returns.rolling(20).std()
        features['jump_indicator'] = (returns.abs() > 3 * returns_std).astype(int)
        
        # Volume-Price Trend (VPT)
        if volume is not None:
            features['vpt'] = (volume * returns).cumsum()
            features['vpt_ma'] = features['vpt'].rolling(20).mean()
        
        return features
    
    def create_regime_features(self, 
                             prices: pd.DataFrame,
                             market_data: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
        """
        Cria features de regime de mercado.
        
        Args:
            prices: DataFrame com preços
            market_data: Dicionário com dados de mercado adicionais
            
        Returns:
            DataFrame com features de regime
        """
        features = pd.DataFrame(index=prices.index)
        
        close = prices['close'] if 'close' in prices.columns else prices.iloc[:, 0]
        returns = close.pct_change()
        
        # Regime de volatilidade
        vol_20 = returns.rolling(20).std()
        vol_percentile = vol_20.rolling(252).rank(pct=True)
        
        features['vol_regime_low'] = (vol_percentile < 0.33).astype(int)
        features['vol_regime_medium'] = ((vol_percentile >= 0.33) & 
                                       (vol_percentile < 0.67)).astype(int)
        features['vol_regime_high'] = (vol_percentile >= 0.67).astype(int)
        
        # Regime de tendência
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        
        features['trend_regime_bull'] = (sma_50 > sma_200).astype(int)
        features['trend_regime_bear'] = (sma_50 <= sma_200).astype(int)
        
        # Regime de momentum
        momentum_20 = close / close.shift(20) - 1
        momentum_percentile = momentum_20.rolling(252).rank(pct=True)
        
        features['momentum_regime_low'] = (momentum_percentile < 0.33).astype(int)
        features['momentum_regime_high'] = (momentum_percentile > 0.67).astype(int)
        
        # Features de correlação com mercado (se dados disponíveis)
        if market_data:
            for name, series in market_data.items():
                if len(series) > 0:
                    # Correlação rolling
                    market_returns = series.pct_change()
                    aligned_returns = returns.reindex(market_returns.index)
                    
                    corr_30 = aligned_returns.rolling(30).corr(market_returns)
                    features[f'corr_{name}_30d'] = corr_30
                    
                    # Beta rolling
                    def rolling_beta(x, y, window=30):
                        """Calcula beta rolling."""
                        if len(x) < window or len(y) < window:
                            return np.nan
                        return np.cov(x, y)[0, 1] / np.var(y)
                    
                    features[f'beta_{name}_30d'] = aligned_returns.rolling(30).apply(
                        lambda x: rolling_beta(x.values, 
                                             market_returns.reindex(x.index).values))
        
        return features
    
    def create_alpha_factors(self, 
                           prices: pd.DataFrame,
                           fundamentals: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Cria alpha factors baseados em literatura quantitativa.
        
        Args:
            prices: DataFrame com preços
            fundamentals: DataFrame com dados fundamentais (opcional)
            
        Returns:
            DataFrame com alpha factors
        """
        features = pd.DataFrame(index=prices.index)
        
        close = prices['close'] if 'close' in prices.columns else prices.iloc[:, 0]
        volume = prices['volume'] if 'volume' in prices.columns else None
        returns = close.pct_change()
        
        # Alpha Factor 1: Mean Reversion
        for period in [5, 10, 20]:
            features[f'mean_reversion_{period}d'] = (
                close.rolling(period).mean() - close) / close.rolling(period).std()
        
        # Alpha Factor 2: Momentum Cross-Sectional
        for period in [20, 60, 120]:
            momentum = close / close.shift(period) - 1
            features[f'momentum_rank_{period}d'] = momentum.rolling(252).rank(pct=True)
        
        # Alpha Factor 3: Volatility Factor
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        features['vol_factor'] = vol_20 / vol_60 - 1
        
        # Alpha Factor 4: Price-Volume Divergence
        if volume is not None:
            price_change = close.pct_change(20)
            volume_change = volume.pct_change(20)
            features['pv_divergence'] = price_change - volume_change
        
        # Alpha Factor 5: Residual Momentum
        # Momentum ajustado por beta de mercado
        market_proxy = close  # Usar próprio ativo como proxy se não houver mercado
        market_returns = market_proxy.pct_change()
        
        def calculate_residual_momentum(window=60):
            """Calcula momentum residual."""
            residuals = []
            
            for i in range(window, len(returns)):
                y = returns.iloc[i-window:i].values
                x = market_returns.iloc[i-window:i].values
                
                # Remover NaNs
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < 10:
                    residuals.append(np.nan)
                    continue
                
                x_clean, y_clean = x[mask], y[mask]
                
                try:
                    # Regressão linear
                    beta = np.cov(x_clean, y_clean)[0, 1] / np.var(x_clean)
                    alpha = np.mean(y_clean) - beta * np.mean(x_clean)
                    
                    # Residual atual
                    residual = returns.iloc[i] - (alpha + beta * market_returns.iloc[i])
                    residuals.append(residual)
                except:
                    residuals.append(np.nan)
            
            return pd.Series(residuals, index=returns.index[window:])
        
        residual_momentum = calculate_residual_momentum()
        features['residual_momentum'] = residual_momentum
        
        # Alpha Factor 6: Earnings Quality (se fundamentais disponíveis)
        if fundamentals is not None:
            # Implementar factors baseados em fundamentais
            pass
        
        return features
    
    def engineer_features(self, 
                         prices: pd.DataFrame,
                         volume: Optional[pd.Series] = None,
                         market_data: Optional[Dict[str, pd.Series]] = None,
                         fundamentals: Optional[pd.DataFrame] = None,
                         include_technical: bool = True,
                         include_statistical: bool = True,
                         include_microstructure: bool = True,
                         include_regime: bool = True,
                         include_alpha: bool = True) -> pd.DataFrame:
        """
        Método principal para criar todas as features.
        
        Args:
            prices: DataFrame com OHLCV
            volume: Série de volume
            market_data: Dados de mercado adicionais
            fundamentals: Dados fundamentais
            include_*: Flags para incluir diferentes tipos de features
            
        Returns:
            DataFrame com todas as features
        """
        all_features = []
        
        # Features básicas de preço
        price_features = self.create_price_features(prices, volume)
        all_features.append(price_features)
        
        # Indicadores técnicos
        if include_technical:
            try:
                tech_features = self.create_technical_indicators(prices)
                all_features.append(tech_features)
            except Exception as e:
                warnings.warn(f"Erro ao criar indicadores técnicos: {e}")
        
        # Features estatísticas
        if include_statistical:
            stat_features = self.create_statistical_features(prices)
            all_features.append(stat_features)
        
        # Features de microestrutura
        if include_microstructure:
            micro_features = self.create_microstructure_features(prices)
            all_features.append(micro_features)
        
        # Features de regime
        if include_regime:
            regime_features = self.create_regime_features(prices, market_data)
            all_features.append(regime_features)
        
        # Alpha factors
        if include_alpha:
            alpha_features = self.create_alpha_factors(prices, fundamentals)
            all_features.append(alpha_features)
        
        # Combinar todas as features
        combined_features = pd.concat(all_features, axis=1)
        
        # Armazenar nomes das features
        self.feature_names = list(combined_features.columns)
        
        return combined_features
    
    def select_features(self, 
                       X: pd.DataFrame,
                       y: pd.Series,
                       method: str = 'mutual_info',
                       k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        Seleciona as melhores features.
        
        Args:
            X: Features
            y: Target
            method: Método de seleção ('mutual_info', 'f_classif', 'correlation')
            k: Número de features a selecionar
            
        Returns:
            Tupla com (features_selecionadas, nomes_features)
        """
        # Remover features com muitos NaNs
        nan_threshold = 0.5
        valid_features = X.columns[X.isnull().mean() < nan_threshold]
        X_clean = X[valid_features].fillna(method='ffill').fillna(0)
        
        # Alinhar com target
        common_idx = X_clean.index.intersection(y.index)
        X_aligned = X_clean.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(X_aligned.columns)))
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(k, len(X_aligned.columns)))
        elif method == 'correlation':
            # Seleção por correlação absoluta
            correlations = X_aligned.corrwith(y_aligned).abs().sort_values(ascending=False)
            selected_features = correlations.head(k).index.tolist()
            return X_aligned[selected_features], selected_features
        else:
            raise ValueError(f"Método {method} não suportado")
        
        # Aplicar seleção
        X_selected = selector.fit_transform(X_aligned, y_aligned)
        selected_features = X_aligned.columns[selector.get_support()].tolist()
        
        return pd.DataFrame(X_selected, index=X_aligned.index, columns=selected_features), selected_features
    
    def scale_features(self, 
                     X: pd.DataFrame,
                     method: str = 'robust',
                     fit: bool = True) -> pd.DataFrame:
        """
        Escala as features.
        
        Args:
            X: Features para escalar
            method: Método de escalonamento ('standard', 'robust', 'minmax')
            fit: Se deve ajustar o scaler
            
        Returns:
            Features escalonadas
        """
        if method not in self.scalers or fit:
            if method == 'standard':
                self.scalers[method] = StandardScaler()
            elif method == 'robust':
                self.scalers[method] = RobustScaler()
            elif method == 'minmax':
                self.scalers[method] = MinMaxScaler()
            else:
                raise ValueError(f"Método {method} não suportado")
        
        scaler = self.scalers[method]
        
        if fit:
            X_scaled = scaler.fit_transform(X.fillna(0))
        else:
            X_scaled = scaler.transform(X.fillna(0))
        
        return pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    def get_feature_importance_analysis(self, 
                                      X: pd.DataFrame,
                                      y: pd.Series) -> pd.DataFrame:
        """
        Analisa importância das features.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            DataFrame com análise de importância
        """
        # Preparar dados
        common_idx = X.index.intersection(y.index)
        X_clean = X.loc[common_idx].fillna(method='ffill').fillna(0)
        y_clean = y.loc[common_idx]
        
        analysis = pd.DataFrame(index=X_clean.columns)
        
        # Correlação com target
        analysis['correlation'] = X_clean.corrwith(y_clean).abs()
        
        # Mutual information
        try:
            mi_scores = mutual_info_classif(X_clean, y_clean)
            analysis['mutual_info'] = mi_scores
        except:
            analysis['mutual_info'] = np.nan
        
        # F-statistic
        try:
            f_scores, _ = f_classif(X_clean, y_clean)
            analysis['f_statistic'] = f_scores
        except:
            analysis['f_statistic'] = np.nan
        
        # Estatísticas descritivas
        analysis['mean'] = X_clean.mean()
        analysis['std'] = X_clean.std()
        analysis['missing_pct'] = X.isnull().mean()
        
        # Ranking combinado
        analysis['combined_rank'] = (
            analysis['correlation'].rank(ascending=False) * 0.4 +
            analysis['mutual_info'].rank(ascending=False) * 0.4 +
            analysis['f_statistic'].rank(ascending=False) * 0.2
        )
        
        return analysis.sort_values('combined_rank')


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados de exemplo
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Simular dados OHLCV
    base_price = 100
    prices_data = []
    
    for i in range(1000):
        change = np.random.randn() * 0.02
        base_price *= (1 + change)
        
        high = base_price * (1 + abs(np.random.randn()) * 0.01)
        low = base_price * (1 - abs(np.random.randn()) * 0.01)
        volume = np.random.randint(1000, 10000)
        
        prices_data.append({
            'open': base_price,
            'high': high,
            'low': low,
            'close': base_price,
            'volume': volume
        })
    
    prices_df = pd.DataFrame(prices_data, index=dates)
    
    # Criar engenheiro de features
    feature_engineer = FinancialFeatureEngineer()
    
    # Gerar features
    features = feature_engineer.engineer_features(prices_df)
    
    print(f"Criadas {len(features.columns)} features")
    print(f"Período: {features.index.min()} a {features.index.max()}")
    print(f"Features com mais de 50% de dados válidos: {(features.isnull().mean() < 0.5).sum()}")