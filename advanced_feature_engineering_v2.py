import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.special import gamma
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import talib as ta
import warnings
from dataclasses import dataclass
from numba import jit, njit
import bottleneck as bn
from collections import deque

@dataclass
class FeatureConfig:
    """Configuração para feature engineering"""
    enable_fractional_diff: bool = True
    enable_microstructure: bool = True
    enable_regime_features: bool = True
    enable_cross_asset: bool = True
    enable_alternative_data: bool = False
    
    # Parâmetros de diferenciação fracionária
    frac_diff_threshold: float = 0.01
    max_lag: int = 100
    
    # Parâmetros de microestrutura
    tick_rule_window: int = 20
    vpin_window: int = 50
    
    # Parâmetros de regime
    regime_lookback: int = 252
    volatility_window: int = 20

class AdvancedFeatureEngineer:
    """
    Sistema avançado de feature engineering para trading quantitativo
    Baseado em "Advances in Financial Machine Learning" e técnicas de ponta
    
    Funcionalidades:
    1. Diferenciação Fracionária para estacionariedade
    2. Features de Microestrutura (VPIN, Tick Rule, etc.)
    3. Detecção de Regimes de Mercado
    4. Features Cross-Asset
    5. Transformações não-lineares
    6. Seleção automática de features
    """
    
    def __init__(self, 
                 config: Optional[FeatureConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            config: Configuração de features
            logger: Logger para debug
        """
        self.config = config or FeatureConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache para otimização
        self._frac_diff_cache = {}
        self._feature_importance_cache = {}
        
        # Scalers
        self.scaler = RobustScaler()
        self.is_fitted = False
        
        # Histórico para features temporais
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Ajusta o feature engineer e transforma os dados
        
        Args:
            data: DataFrame com dados OHLCV
            target: Série target para seleção supervisionada
            
        Returns:
            DataFrame com features engineered
        """
        self.logger.info("Iniciando feature engineering avançado")
        
        # Validar dados de entrada
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing_cols}")
            
        features_df = pd.DataFrame(index=data.index)
        
        # 1. Features básicas de preço
        price_features = self._create_price_features(data)
        features_df = pd.concat([features_df, price_features], axis=1)
        
        # 2. Diferenciação fracionária
        if self.config.enable_fractional_diff:
            frac_diff_features = self._create_fractional_diff_features(data)
            features_df = pd.concat([features_df, frac_diff_features], axis=1)
            
        # 3. Features de microestrutura
        if self.config.enable_microstructure:
            microstructure_features = self._create_microstructure_features(data)
            features_df = pd.concat([features_df, microstructure_features], axis=1)
            
        # 4. Features de regime de mercado
        if self.config.enable_regime_features:
            regime_features = self._create_regime_features(data)
            features_df = pd.concat([features_df, regime_features], axis=1)
            
        # 5. Indicadores técnicos avançados
        technical_features = self._create_advanced_technical_features(data)
        features_df = pd.concat([features_df, technical_features], axis=1)
        
        # 6. Features estatísticas
        statistical_features = self._create_statistical_features(data)
        features_df = pd.concat([features_df, statistical_features], axis=1)
        
        # 7. Features de volatilidade
        volatility_features = self._create_volatility_features(data)
        features_df = pd.concat([features_df, volatility_features], axis=1)
        
        # 8. Features temporais
        temporal_features = self._create_temporal_features(data)
        features_df = pd.concat([features_df, temporal_features], axis=1)
        
        # Limpar dados
        features_df = self._clean_features(features_df)
        
        # Seleção de features
        if target is not None:
            features_df = self._select_features(features_df, target)
            
        # Normalização
        features_df = self._normalize_features(features_df)
        
        self.is_fitted = True
        
        self.logger.info(f"Feature engineering concluído: {features_df.shape[1]} features criadas")
        
        return features_df
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma novos dados usando configuração já ajustada
        
        Args:
            data: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com features transformadas
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer deve ser ajustado antes da transformação")
            
        # Aplicar mesmas transformações do fit_transform
        # (implementação simplificada - na prática, usar mesmos parâmetros)
        return self.fit_transform(data)
    
    def _create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features básicas de preço
        """
        features = pd.DataFrame(index=data.index)
        
        # Returns
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Multi-period returns
        for period in [2, 3, 5, 10, 20]:
            features[f'returns_{period}d'] = data['close'].pct_change(period)
            features[f'log_returns_{period}d'] = np.log(data['close'] / data['close'].shift(period))
            
        # Price ratios
        features['hl_ratio'] = (data['high'] - data['low']) / data['close']
        features['oc_ratio'] = (data['open'] - data['close']) / data['close']
        features['hc_ratio'] = (data['high'] - data['close']) / data['close']
        features['lc_ratio'] = (data['low'] - data['close']) / data['close']
        
        # True Range
        features['true_range'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        
        # Gaps
        features['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        features['gap_filled'] = (features['gap'] > 0) & (data['low'] <= data['close'].shift(1))
        
        return features
    
    def _create_fractional_diff_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features usando diferenciação fracionária
        Baseado em López de Prado "Advances in Financial Machine Learning"
        """
        features = pd.DataFrame(index=data.index)
        
        # Aplicar diferenciação fracionária no preço de fechamento
        close_prices = data['close'].dropna()
        
        # Encontrar ordem ótima de diferenciação
        optimal_d = self._find_optimal_frac_diff_order(close_prices)
        
        # Aplicar diferenciação fracionária
        frac_diff_close = self._fractional_differentiation(close_prices, optimal_d)
        features['frac_diff_close'] = frac_diff_close
        
        # Aplicar em outras séries
        for col in ['high', 'low', 'volume']:
            if col in data.columns:
                series = data[col].dropna()
                if len(series) > 50:
                    d_opt = self._find_optimal_frac_diff_order(series)
                    frac_diff = self._fractional_differentiation(series, d_opt)
                    features[f'frac_diff_{col}'] = frac_diff
                    
        # Features derivadas da diferenciação fracionária
        if 'frac_diff_close' in features.columns:
            frac_close = features['frac_diff_close']
            
            # Volatilidade da série diferenciada
            features['frac_diff_vol'] = frac_close.rolling(20).std()
            
            # Autocorrelação da série diferenciada
            features['frac_diff_autocorr'] = frac_close.rolling(20).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
            )
            
            # Skewness e Kurtosis
            features['frac_diff_skew'] = frac_close.rolling(20).skew()
            features['frac_diff_kurt'] = frac_close.rolling(20).kurt()
            
        return features
    
    @njit
    def _get_weights_ffd(d: float, size: int) -> np.ndarray:
        """
        Calcula pesos para diferenciação fracionária (Fixed-Width Window)
        """
        w = np.array([1.0])
        for k in range(1, size):
            w_k = -w[-1] * (d - k + 1) / k
            w = np.append(w, w_k)
        return w
    
    def _fractional_differentiation(self, series: pd.Series, d: float, 
                                  threshold: float = 0.01) -> pd.Series:
        """
        Aplica diferenciação fracionária com janela fixa
        
        Args:
            series: Série temporal
            d: Ordem de diferenciação
            threshold: Threshold para truncar pesos
            
        Returns:
            Série diferenciada
        """
        # Cache para evitar recálculos
        cache_key = (d, len(series), threshold)
        if cache_key in self._frac_diff_cache:
            weights = self._frac_diff_cache[cache_key]
        else:
            # Calcular pesos
            weights = [1.0]
            for k in range(1, len(series)):
                weight = -weights[-1] * (d - k + 1) / k
                if abs(weight) < threshold:
                    break
                weights.append(weight)
                
            weights = np.array(weights)
            self._frac_diff_cache[cache_key] = weights
            
        # Aplicar diferenciação
        width = len(weights)
        result = pd.Series(index=series.index, dtype=float)
        
        for i in range(width, len(series)):
            result.iloc[i] = np.dot(weights, series.iloc[i-width+1:i+1][::-1])
            
        return result
    
    def _find_optimal_frac_diff_order(self, series: pd.Series, 
                                    max_d: float = 1.0, step: float = 0.1) -> float:
        """
        Encontra ordem ótima de diferenciação fracionária usando ADF test
        
        Args:
            series: Série temporal
            max_d: Ordem máxima a testar
            step: Passo para busca
            
        Returns:
            Ordem ótima de diferenciação
        """
        from statsmodels.tsa.stattools import adfuller
        
        best_d = 0.0
        best_pvalue = 1.0
        
        d_values = np.arange(0.0, max_d + step, step)
        
        for d in d_values:
            try:
                if d == 0:
                    test_series = series
                else:
                    test_series = self._fractional_differentiation(series, d)
                    test_series = test_series.dropna()
                    
                if len(test_series) < 50:
                    continue
                    
                # Teste ADF
                adf_result = adfuller(test_series, maxlag=1, regression='c', autolag=None)
                pvalue = adf_result[1]
                
                # Queremos estacionariedade (p-value baixo) com mínima diferenciação
                if pvalue < best_pvalue and pvalue < 0.05:
                    best_pvalue = pvalue
                    best_d = d
                    
            except Exception as e:
                continue
                
        return best_d
    
    def _create_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de microestrutura de mercado
        """
        features = pd.DataFrame(index=data.index)
        
        # VPIN (Volume-Synchronized Probability of Informed Trading)
        if 'volume' in data.columns:
            features['vpin'] = self._calculate_vpin(data)
            
        # Tick Rule (inferir direção do trade)
        features['tick_rule'] = self._calculate_tick_rule(data['close'])
        
        # Order Flow Imbalance
        features['order_flow_imbalance'] = self._calculate_order_flow_imbalance(data)
        
        # Roll Measure (bid-ask spread estimator)
        features['roll_measure'] = self._calculate_roll_measure(data['close'])
        
        # Effective Spread
        features['effective_spread'] = self._calculate_effective_spread(data)
        
        # Price Impact
        features['price_impact'] = self._calculate_price_impact(data)
        
        # Amihud Illiquidity
        features['amihud_illiquidity'] = abs(data['close'].pct_change()) / (data['volume'] + 1e-8)
        
        # Kyle's Lambda
        features['kyle_lambda'] = self._calculate_kyle_lambda(data)
        
        return features
    
    def _calculate_vpin(self, data: pd.DataFrame, window: int = 50) -> pd.Series:
        """
        Calcula VPIN (Volume-Synchronized Probability of Informed Trading)
        """
        returns = data['close'].pct_change()
        volume = data['volume']
        
        # Classificar volume como buy/sell baseado no tick rule
        tick_rule = self._calculate_tick_rule(data['close'])
        
        buy_volume = volume * (tick_rule > 0)
        sell_volume = volume * (tick_rule < 0)
        
        # VPIN = |V_buy - V_sell| / (V_buy + V_sell)
        total_volume = buy_volume + sell_volume
        vpin = abs(buy_volume - sell_volume) / (total_volume + 1e-8)
        
        return vpin.rolling(window).mean()
    
    def _calculate_tick_rule(self, prices: pd.Series) -> pd.Series:
        """
        Calcula tick rule para inferir direção do trade
        """
        price_changes = prices.diff()
        
        # Tick rule: +1 se preço subiu, -1 se desceu, manter anterior se igual
        tick_rule = pd.Series(index=prices.index, dtype=float)
        tick_rule.iloc[0] = 0
        
        for i in range(1, len(prices)):
            if price_changes.iloc[i] > 0:
                tick_rule.iloc[i] = 1
            elif price_changes.iloc[i] < 0:
                tick_rule.iloc[i] = -1
            else:
                tick_rule.iloc[i] = tick_rule.iloc[i-1]  # Manter anterior
                
        return tick_rule
    
    def _calculate_order_flow_imbalance(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcula order flow imbalance
        """
        returns = data['close'].pct_change()
        volume = data['volume']
        
        # Aproximação: volume positivo quando retorno positivo
        buy_volume = volume * (returns > 0)
        sell_volume = volume * (returns < 0)
        
        ofi = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-8)
        
        return ofi.rolling(20).mean()
    
    def _calculate_roll_measure(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """
        Calcula Roll measure para estimar bid-ask spread
        """
        returns = prices.pct_change()
        
        # Roll measure = 2 * sqrt(-Cov(r_t, r_{t-1}))
        def roll_calc(x):
            if len(x) < 2:
                return 0
            covariance = np.cov(x[:-1], x[1:])[0, 1]
            return 2 * np.sqrt(max(-covariance, 0))
            
        return returns.rolling(window).apply(roll_calc)
    
    def _calculate_effective_spread(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcula effective spread
        """
        mid_price = (data['high'] + data['low']) / 2
        trade_price = data['close']
        
        # Effective spread = 2 * |trade_price - mid_price| / mid_price
        effective_spread = 2 * abs(trade_price - mid_price) / (mid_price + 1e-8)
        
        return effective_spread.rolling(20).mean()
    
    def _calculate_price_impact(self, data: pd.DataFrame, window: int = 10) -> pd.Series:
        """
        Calcula price impact
        """
        returns = data['close'].pct_change()
        volume = data['volume']
        
        # Price impact = |return| / volume
        price_impact = abs(returns) / (volume + 1e-8)
        
        return price_impact.rolling(window).mean()
    
    def _calculate_kyle_lambda(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calcula Kyle's Lambda (medida de adverse selection)
        """
        returns = data['close'].pct_change()
        volume = data['volume']
        
        def kyle_calc(ret_vol_data):
            if len(ret_vol_data) < 5:
                return 0
            rets = ret_vol_data[:, 0]
            vols = ret_vol_data[:, 1]
            
            # Regressão: |return| = lambda * volume + epsilon
            try:
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                reg.fit(vols.reshape(-1, 1), np.abs(rets))
                return reg.coef_[0]
            except:
                return 0
                
        ret_vol = np.column_stack([returns, volume])
        kyle_lambda = pd.Series(index=data.index, dtype=float)
        
        for i in range(window, len(data)):
            kyle_lambda.iloc[i] = kyle_calc(ret_vol[i-window:i])
            
        return kyle_lambda
    
    def _create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de detecção de regime de mercado
        """
        features = pd.DataFrame(index=data.index)
        
        returns = data['close'].pct_change()
        
        # Volatility regimes
        vol_short = returns.rolling(self.config.volatility_window).std()
        vol_long = returns.rolling(self.config.volatility_window * 4).std()
        features['vol_regime'] = vol_short / (vol_long + 1e-8)
        
        # Trend regimes
        ma_short = data['close'].rolling(20).mean()
        ma_long = data['close'].rolling(50).mean()
        features['trend_regime'] = (ma_short - ma_long) / (ma_long + 1e-8)
        
        # Market stress indicator
        features['market_stress'] = self._calculate_market_stress(data)
        
        # Regime change detection
        features['regime_change'] = self._detect_regime_changes(returns)
        
        # Correlation regime (se temos múltiplos assets)
        if hasattr(self, 'correlation_data'):
            features['correlation_regime'] = self._calculate_correlation_regime()
            
        return features
    
    def _calculate_market_stress(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calcula indicador de stress do mercado
        """
        returns = data['close'].pct_change()
        volume = data['volume']
        
        # Combinar volatilidade, skewness e volume
        vol = returns.rolling(window).std()
        skew = returns.rolling(window).skew()
        vol_surge = volume.rolling(window).mean() / volume.rolling(window*4).mean()
        
        # Stress = volatilidade * |skewness| * volume_surge
        stress = vol * abs(skew) * vol_surge
        
        return stress
    
    def _detect_regime_changes(self, returns: pd.Series, window: int = 50) -> pd.Series:
        """
        Detecta mudanças de regime usando CUSUM
        """
        # CUSUM para detectar mudanças na média
        mean_returns = returns.rolling(window).mean()
        std_returns = returns.rolling(window).std()
        
        # Estatística CUSUM
        cusum_pos = pd.Series(0.0, index=returns.index)
        cusum_neg = pd.Series(0.0, index=returns.index)
        
        threshold = 2.0  # Threshold para detecção
        
        for i in range(window, len(returns)):
            if not pd.isna(mean_returns.iloc[i]) and not pd.isna(std_returns.iloc[i]):
                normalized_return = (returns.iloc[i] - mean_returns.iloc[i]) / (std_returns.iloc[i] + 1e-8)
                
                cusum_pos.iloc[i] = max(0, cusum_pos.iloc[i-1] + normalized_return - 0.5)
                cusum_neg.iloc[i] = max(0, cusum_neg.iloc[i-1] - normalized_return - 0.5)
                
        # Sinal de mudança de regime
        regime_change = (cusum_pos > threshold) | (cusum_neg > threshold)
        
        return regime_change.astype(int)
    
    def _create_advanced_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cria indicadores técnicos avançados
        """
        features = pd.DataFrame(index=data.index)
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        
        # Indicadores TA-Lib avançados
        try:
            # Momentum indicators
            features['adx'] = ta.ADX(high, low, close, timeperiod=14)
            features['aroon_up'], features['aroon_down'] = ta.AROON(high, low, timeperiod=14)
            features['cci'] = ta.CCI(high, low, close, timeperiod=14)
            features['cmo'] = ta.CMO(close, timeperiod=14)
            features['mfi'] = ta.MFI(high, low, close, volume, timeperiod=14)
            
            # Volatility indicators
            features['atr'] = ta.ATR(high, low, close, timeperiod=14)
            features['natr'] = ta.NATR(high, low, close, timeperiod=14)
            features['trange'] = ta.TRANGE(high, low, close)
            
            # Volume indicators
            features['ad'] = ta.AD(high, low, close, volume)
            features['adosc'] = ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
            features['obv'] = ta.OBV(close, volume)
            
            # Cycle indicators
            features['ht_dcperiod'] = ta.HT_DCPERIOD(close)
            features['ht_dcphase'] = ta.HT_DCPHASE(close)
            features['ht_trendmode'] = ta.HT_TRENDMODE(close)
            
        except Exception as e:
            self.logger.warning(f"Erro ao calcular indicadores TA-Lib: {e}")
            
        # Indicadores customizados
        features['price_position'] = self._calculate_price_position(data)
        features['volatility_ratio'] = self._calculate_volatility_ratio(data)
        features['momentum_divergence'] = self._calculate_momentum_divergence(data)
        
        return features
    
    def _calculate_price_position(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calcula posição do preço dentro da faixa de trading
        """
        high_max = data['high'].rolling(window).max()
        low_min = data['low'].rolling(window).min()
        
        price_position = (data['close'] - low_min) / (high_max - low_min + 1e-8)
        
        return price_position
    
    def _calculate_volatility_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcula ratio de volatilidade (curto vs longo prazo)
        """
        returns = data['close'].pct_change()
        
        vol_short = returns.rolling(10).std()
        vol_long = returns.rolling(30).std()
        
        return vol_short / (vol_long + 1e-8)
    
    def _calculate_momentum_divergence(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcula divergência entre preço e momentum
        """
        close = data['close']
        
        # Price momentum
        price_mom = close.rolling(10).apply(lambda x: (x[-1] - x[0]) / x[0])
        
        # RSI momentum
        rsi = ta.RSI(close.values, timeperiod=14)
        rsi_series = pd.Series(rsi, index=close.index)
        rsi_mom = rsi_series.rolling(10).apply(lambda x: x[-1] - x[0])
        
        # Divergence = sign(price_momentum) != sign(rsi_momentum)
        divergence = (np.sign(price_mom) != np.sign(rsi_mom)).astype(int)
        
        return divergence
    
    def _create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features estatísticas avançadas
        """
        features = pd.DataFrame(index=data.index)
        
        returns = data['close'].pct_change()
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            features[f'skew_{window}'] = returns.rolling(window).skew()
            features[f'kurt_{window}'] = returns.rolling(window).kurt()
            features[f'var_{window}'] = returns.rolling(window).var()
            
        # Quantile-based features
        for window in [20, 50]:
            features[f'q25_{window}'] = returns.rolling(window).quantile(0.25)
            features[f'q75_{window}'] = returns.rolling(window).quantile(0.75)
            features[f'iqr_{window}'] = (features[f'q75_{window}'] - features[f'q25_{window}'])
            
        # Autocorrelation features
        for lag in [1, 2, 5, 10]:
            features[f'autocorr_{lag}'] = returns.rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
            )
            
        # Hurst exponent
        features['hurst'] = self._calculate_hurst_exponent(returns)
        
        return features
    
    def _calculate_hurst_exponent(self, returns: pd.Series, window: int = 100) -> pd.Series:
        """
        Calcula expoente de Hurst para detectar persistência/anti-persistência
        """
        def hurst_calc(x):
            if len(x) < 10:
                return 0.5
                
            try:
                # R/S analysis
                n = len(x)
                mean_x = np.mean(x)
                
                # Deviations from mean
                deviations = x - mean_x
                
                # Cumulative deviations
                cumulative_deviations = np.cumsum(deviations)
                
                # Range
                R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                
                # Standard deviation
                S = np.std(x)
                
                if S == 0:
                    return 0.5
                    
                # R/S ratio
                rs = R / S
                
                # Hurst exponent approximation
                if rs > 0:
                    return np.log(rs) / np.log(n)
                else:
                    return 0.5
                    
            except:
                return 0.5
                
        return returns.rolling(window).apply(hurst_calc)
    
    def _create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de volatilidade avançadas
        """
        features = pd.DataFrame(index=data.index)
        
        returns = data['close'].pct_change()
        
        # GARCH-like volatility
        features['garch_vol'] = self._calculate_garch_volatility(returns)
        
        # Realized volatility
        features['realized_vol'] = returns.rolling(20).std() * np.sqrt(252)
        
        # Volatility of volatility
        vol = returns.rolling(20).std()
        features['vol_of_vol'] = vol.rolling(20).std()
        
        # Parkinson volatility (usando high-low)
        features['parkinson_vol'] = self._calculate_parkinson_volatility(data)
        
        # Garman-Klass volatility
        features['gk_vol'] = self._calculate_garman_klass_volatility(data)
        
        return features
    
    def _calculate_garch_volatility(self, returns: pd.Series, window: int = 50) -> pd.Series:
        """
        Calcula volatilidade GARCH simplificada
        """
        # GARCH(1,1) simplificado: sigma^2_t = alpha * r^2_{t-1} + beta * sigma^2_{t-1}
        alpha, beta = 0.1, 0.85
        
        garch_vol = pd.Series(index=returns.index, dtype=float)
        
        # Inicializar com volatilidade histórica
        initial_vol = returns.rolling(window).std().iloc[window-1] if len(returns) >= window else returns.std()
        garch_vol.iloc[0] = initial_vol
        
        for i in range(1, len(returns)):
            if pd.isna(returns.iloc[i-1]) or pd.isna(garch_vol.iloc[i-1]):
                garch_vol.iloc[i] = garch_vol.iloc[i-1]
            else:
                garch_vol.iloc[i] = np.sqrt(
                    alpha * returns.iloc[i-1]**2 + beta * garch_vol.iloc[i-1]**2
                )
                
        return garch_vol
    
    def _calculate_parkinson_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calcula volatilidade de Parkinson usando high-low
        """
        hl_ratio = np.log(data['high'] / data['low'])
        parkinson_vol = hl_ratio.rolling(window).apply(
            lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2)))
        )
        
        return parkinson_vol * np.sqrt(252)  # Anualizar
    
    def _calculate_garman_klass_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calcula volatilidade de Garman-Klass
        """
        log_hl = np.log(data['high'] / data['low'])
        log_co = np.log(data['close'] / data['open'])
        
        gk_vol = (0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2).rolling(window).mean()
        
        return np.sqrt(gk_vol * 252)  # Anualizar
    
    def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features temporais
        """
        features = pd.DataFrame(index=data.index)
        
        # Time-based features
        if isinstance(data.index, pd.DatetimeIndex):
            features['hour'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            
            # Cyclical encoding
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            
        # Session-based features (assumindo Forex)
        features['asian_session'] = self._is_asian_session(data.index)
        features['european_session'] = self._is_european_session(data.index)
        features['us_session'] = self._is_us_session(data.index)
        
        return features
    
    def _is_asian_session(self, index: pd.DatetimeIndex) -> pd.Series:
        """Detecta sessão asiática (00:00-09:00 UTC)"""
        return ((index.hour >= 0) & (index.hour < 9)).astype(int)
    
    def _is_european_session(self, index: pd.DatetimeIndex) -> pd.Series:
        """Detecta sessão europeia (08:00-17:00 UTC)"""
        return ((index.hour >= 8) & (index.hour < 17)).astype(int)
    
    def _is_us_session(self, index: pd.DatetimeIndex) -> pd.Series:
        """Detecta sessão americana (13:00-22:00 UTC)"""
        return ((index.hour >= 13) & (index.hour < 22)).astype(int)
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa e processa features
        """
        # Remover colunas com muitos NaNs
        nan_threshold = 0.5
        features = features.loc[:, features.isnull().mean() < nan_threshold]
        
        # Remover outliers extremos
        for col in features.select_dtypes(include=[np.number]).columns:
            q99 = features[col].quantile(0.99)
            q01 = features[col].quantile(0.01)
            features[col] = features[col].clip(lower=q01, upper=q99)
            
        # Preencher NaNs
        features = features.fillna(method='ffill').fillna(0)
        
        # Remover features constantes
        constant_features = features.columns[features.nunique() <= 1]
        features = features.drop(columns=constant_features)
        
        return features
    
    def _select_features(self, features: pd.DataFrame, target: pd.Series, 
                        max_features: int = 100) -> pd.DataFrame:
        """
        Seleciona features mais relevantes
        """
        # Alinhar índices
        common_index = features.index.intersection(target.index)
        features_aligned = features.loc[common_index]
        target_aligned = target.loc[common_index]
        
        # Remover NaNs
        mask = ~(features_aligned.isnull().any(axis=1) | target_aligned.isnull())
        features_clean = features_aligned[mask]
        target_clean = target_aligned[mask]
        
        if len(features_clean) < 50:
            self.logger.warning("Poucos dados para seleção de features")
            return features
            
        # Seleção univariada
        try:
            selector = SelectKBest(score_func=f_classif, k=min(max_features, features_clean.shape[1]))
            features_selected = selector.fit_transform(features_clean, target_clean)
            
            selected_columns = features_clean.columns[selector.get_support()]
            
            self.logger.info(f"Selecionadas {len(selected_columns)} features de {features.shape[1]}")
            
            return features[selected_columns]
            
        except Exception as e:
            self.logger.error(f"Erro na seleção de features: {e}")
            return features
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza features usando RobustScaler
        """
        try:
            # Ajustar scaler apenas uma vez
            if not hasattr(self.scaler, 'scale_') or self.scaler.scale_ is None:
                numeric_features = features.select_dtypes(include=[np.number])
                if len(numeric_features) > 0:
                    self.scaler.fit(numeric_features.fillna(0))
                    
            # Transformar features numéricas
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                features_scaled = features.copy()
                features_scaled[numeric_cols] = self.scaler.transform(features[numeric_cols].fillna(0))
                return features_scaled
            else:
                return features
                
        except Exception as e:
            self.logger.error(f"Erro na normalização: {e}")
            return features

# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Criar dados de exemplo
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    
    # Simular dados OHLCV
    price = 100
    prices = [price]
    
    for _ in range(999):
        price *= (1 + np.random.normal(0, 0.01))
        prices.append(price)
        
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(10, 1, 1000)
    }, index=dates)
    
    # Criar target
    target = (data['close'].shift(-1) > data['close']).astype(int)[:-1]
    
    # Configuração
    config = FeatureConfig(
        enable_fractional_diff=True,
        enable_microstructure=True,
        enable_regime_features=True
    )
    
    # Feature engineering
    engineer = AdvancedFeatureEngineer(config=config, logger=logger)
    features = engineer.fit_transform(data[:-1], target)
    
    print(f"\nFeatures criadas: {features.shape}")
    print(f"Colunas: {list(features.columns)}")
    print(f"\nEstatísticas das features:")
    print(features.describe())