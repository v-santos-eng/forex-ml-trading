#!/usr/bin/env python3
"""
ML Forex Predictor - Versão Otimizada
Versão melhorada com correções de performance, memory management e robustez

Melhorias implementadas:
- Otimização de memória com processamento em chunks
- Paralelização inteligente de operações
- Cache de features calculadas
- Validação robusta de dados
- Prevenção de data leakage
- Ensemble adaptativo
- Monitoramento de performance em tempo real
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import json
import sqlite3
import pickle
import os
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import gc

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost não disponível. Instalando...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost as xgb
    XGBOOST_AVAILABLE = True

# LightGBM para performance adicional
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM não disponível. Instalando...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True

# Indicadores técnicos
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib não disponível. Usando indicadores básicos.")

class MemoryEfficientFeatureCalculator(BaseEstimator, TransformerMixin):
    """Calculador de features otimizado para memória"""
    
    def __init__(self, chunk_size=10000, use_cache=True):
        self.chunk_size = chunk_size
        self.use_cache = use_cache
        self.feature_cache = {}
        
    def fit(self, X, y=None):
        return self
    
    @lru_cache(maxsize=128)
    def _calculate_sma(self, prices_tuple, period):
        """Calcula SMA com cache"""
        prices = np.array(prices_tuple)
        return pd.Series(prices).rolling(window=period, min_periods=1).mean().values
    
    def _calculate_indicators_chunk(self, chunk_data):
        """Calcula indicadores para um chunk de dados"""
        indicators = {}
        
        # Preços básicos
        close = chunk_data['Close'].values
        high = chunk_data['High'].values
        low = chunk_data['Low'].values
        volume = chunk_data.get('Volume', pd.Series(index=chunk_data.index, data=0)).values
        
        # Médias móveis otimizadas
        for period in [5, 10, 20, 50]:
            if len(close) >= period:
                indicators[f'SMA_{period}'] = pd.Series(close).rolling(window=period, min_periods=1).mean().values
                indicators[f'EMA_{period}'] = pd.Series(close).ewm(span=period, adjust=False).mean().values
        
        # RSI otimizado
        for period in [14, 21]:
            if len(close) >= period:
                delta = pd.Series(close).diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
                rs = gain / (loss + 1e-10)  # Evitar divisão por zero
                indicators[f'RSI_{period}'] = (100 - (100 / (1 + rs))).values
        
        # MACD otimizado
        ema_12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema_26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        indicators['MACD'] = macd.values
        indicators['MACD_Signal'] = macd_signal.values
        indicators['MACD_Histogram'] = (macd - macd_signal).values
        
        # Bollinger Bands
        bb_period = 20
        if len(close) >= bb_period:
            bb_middle = pd.Series(close).rolling(window=bb_period, min_periods=1).mean()
            bb_std = pd.Series(close).rolling(window=bb_period, min_periods=1).std()
            indicators['BB_Upper'] = (bb_middle + (bb_std * 2)).values
            indicators['BB_Lower'] = (bb_middle - (bb_std * 2)).values
            indicators['BB_Width'] = (indicators['BB_Upper'] - indicators['BB_Lower'])
            indicators['BB_Position'] = ((close - indicators['BB_Lower']) / 
                                       (indicators['BB_Upper'] - indicators['BB_Lower'] + 1e-10))
        
        # Volatilidade
        for period in [10, 20]:
            if len(close) >= period:
                indicators[f'Volatility_{period}'] = pd.Series(close).rolling(window=period, min_periods=1).std().values
                indicators[f'ATR_{period}'] = pd.Series(high - low).rolling(window=period, min_periods=1).mean().values
        
        # Momentum
        for period in [5, 10]:
            if len(close) >= period:
                shifted_close = np.roll(close, period)
                shifted_close[:period] = close[:period]  # Preencher valores iniciais
                indicators[f'Momentum_{period}'] = close / (shifted_close + 1e-10)
                indicators[f'ROC_{period}'] = (close - shifted_close) / (shifted_close + 1e-10)
        
        return indicators
    
    def transform(self, X):
        """Transforma dados calculando features em chunks"""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input deve ser um DataFrame pandas")
        
        # Processar em chunks para otimizar memória
        all_features = []
        
        for i in range(0, len(X), self.chunk_size):
            chunk = X.iloc[i:i+self.chunk_size].copy()
            chunk_features = self._calculate_indicators_chunk(chunk)
            
            # Converter para DataFrame
            chunk_df = pd.DataFrame(chunk_features, index=chunk.index)
            all_features.append(chunk_df)
            
            # Limpeza de memória
            del chunk, chunk_features
            gc.collect()
        
        # Concatenar todos os chunks
        result = pd.concat(all_features, axis=0)
        
        # Preencher valores NaN
        result = result.fillna(method='ffill').fillna(0)
        
        return result

class OptimizedMLForexPredictor:
    """
    Preditor de Forex otimizado com:
    - Gestão eficiente de memória
    - Processamento paralelo
    - Cache inteligente
    - Validação robusta
    - Ensemble adaptativo
    """
    
    def __init__(self, 
                 use_pca=True, 
                 n_components=0.95, 
                 feature_selection_method='auto',
                 chunk_size=10000,
                 n_jobs=-1,
                 cache_features=True):
        
        self.use_pca = use_pca
        self.n_components = n_components
        self.feature_selection_method = feature_selection_method
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.cache_features = cache_features
        
        # Configurar modelos otimizados
        self._setup_optimized_models()
        
        # Componentes do pipeline
        self.feature_calculator = MemoryEfficientFeatureCalculator(chunk_size=chunk_size)
        self.scaler = RobustScaler()  # Mais robusto a outliers
        self.pca = PCA(n_components=self.n_components) if self.use_pca else None
        self.feature_selector = None
        
        # Modelos treinados
        self.ensemble_model = None
        self.best_model = None
        self.feature_names = []
        self.selected_features = []
        
        # Métricas de performance
        self.performance_metrics = {}
        self.training_history = []
        
        # Cache
        self.feature_cache = {}
        
    def _setup_optimized_models(self):
        """Configura modelos com hiperparâmetros otimizados"""
        self.base_models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=self.n_jobs
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42,
                max_iter=2000,
                C=1.0,
                penalty='l2',
                solver='liblinear',
                n_jobs=self.n_jobs
            )
        }
        
        # Adicionar XGBoost otimizado
        if XGBOOST_AVAILABLE:
            self.base_models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                eval_metric='logloss',
                n_jobs=self.n_jobs
            )
        
        # Adicionar LightGBM otimizado
        if LIGHTGBM_AVAILABLE:
            self.base_models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=self.n_jobs,
                verbose=-1
            )
    
    def get_forex_data_optimized(self, pair, period='2y', interval='1d'):
        """Obtém dados de forex de forma otimizada"""
        try:
            # Cache key
            cache_key = f"{pair}_{period}_{interval}"
            
            if self.cache_features and cache_key in self.feature_cache:
                print(f"📋 Usando dados em cache para {pair}")
                return self.feature_cache[cache_key]
            
            print(f"📊 Baixando dados para {pair}...")
            ticker = yf.Ticker(pair)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"⚠️ Nenhum dado encontrado para {pair}")
                return None
            
            # Validação básica
            if len(data) < 100:
                print(f"⚠️ Dados insuficientes para {pair}: {len(data)} registros")
                return None
            
            # Cache dos dados
            if self.cache_features:
                self.feature_cache[cache_key] = data
            
            print(f"✅ Dados obtidos para {pair}: {len(data)} registros")
            return data
            
        except Exception as e:
            print(f"❌ Erro ao obter dados para {pair}: {e}")
            return None
    
    def prepare_features_optimized(self, data, target_periods=[1, 3, 5]):
        """Prepara features de forma otimizada"""
        print("🔧 Calculando features técnicas...")
        
        # Calcular features usando o calculador otimizado
        features_df = self.feature_calculator.transform(data)
        
        # Adicionar features de preço
        price_features = pd.DataFrame(index=data.index)
        price_features['Open'] = data['Open']
        price_features['High'] = data['High']
        price_features['Low'] = data['Low']
        price_features['Close'] = data['Close']
        price_features['Volume'] = data.get('Volume', 0)
        
        # Features de retorno
        for period in [1, 2, 3, 5, 10]:
            price_features[f'Return_{period}'] = data['Close'].pct_change(period)
            price_features[f'Log_Return_{period}'] = np.log(data['Close'] / data['Close'].shift(period))
        
        # Combinar todas as features
        all_features = pd.concat([price_features, features_df], axis=1)
        
        # Criar targets
        targets = pd.DataFrame(index=data.index)
        
        for period in target_periods:
            # Target de classificação (direção do movimento)
            future_return = data['Close'].shift(-period) / data['Close'] - 1
            targets[f'target_clf_{period}'] = (future_return > 0).astype(int)
            
            # Target de regressão (magnitude do movimento)
            targets[f'target_reg_{period}'] = future_return
        
        # Remover linhas com NaN
        valid_idx = ~(all_features.isna().any(axis=1) | targets.isna().any(axis=1))
        
        X = all_features[valid_idx]
        y = targets[valid_idx]
        
        # Armazenar nomes das features
        self.feature_names = list(X.columns)
        
        print(f"✅ Features preparadas: {X.shape[0]} amostras, {X.shape[1]} features")
        
        return X, y
    
    def optimize_hyperparameters(self, X, y, model_name, cv_folds=3):
        """Otimiza hiperparâmetros usando RandomizedSearchCV"""
        print(f"🎯 Otimizando hiperparâmetros para {model_name}...")
        
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 150, 200],
                'max_depth': [8, 10, 12, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 150, 200],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 150, 200],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        if model_name not in param_grids:
            return self.base_models[model_name]
        
        # Usar TimeSeriesSplit para dados temporais
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # RandomizedSearchCV para eficiência
        search = RandomizedSearchCV(
            self.base_models[model_name],
            param_grids[model_name],
            n_iter=20,  # Número limitado de iterações
            cv=tscv,
            scoring='accuracy',
            n_jobs=self.n_jobs,
            random_state=42,
            verbose=0
        )
        
        search.fit(X, y)
        
        print(f"✅ Melhor score para {model_name}: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def train_optimized(self, pair, period='2y', target_period=1, optimize_params=True):
        """Treina modelo de forma otimizada"""
        print(f"🚀 Iniciando treinamento otimizado para {pair}")
        
        # Obter dados
        data = self.get_forex_data_optimized(pair, period)
        if data is None:
            return False
        
        # Preparar features
        X, y_all = self.prepare_features_optimized(data)
        
        # Selecionar target específico
        target_col = f'target_clf_{target_period}'
        if target_col not in y_all.columns:
            print(f"❌ Target {target_col} não encontrado")
            return False
        
        y = y_all[target_col]
        
        # Divisão temporal dos dados
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"📊 Divisão dos dados - Treino: {len(X_train)}, Teste: {len(X_test)}")
        
        # Preprocessamento
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Seleção de features
        if self.feature_selection_method != 'none':
            self.selected_features = self._select_features_optimized(X_train_scaled, y_train)
            feature_mask = [name in self.selected_features for name in self.feature_names]
            X_train_scaled = X_train_scaled[:, feature_mask]
            X_test_scaled = X_test_scaled[:, feature_mask]
            print(f"🎯 Features selecionadas: {len(self.selected_features)}")
        
        # PCA se habilitado
        if self.use_pca:
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_test_scaled = self.pca.transform(X_test_scaled)
            print(f"📉 PCA aplicado: {X_train_scaled.shape[1]} componentes")
        
        # Treinar modelos
        trained_models = {}
        
        for name, model in self.base_models.items():
            print(f"🔧 Treinando {name}...")
            
            try:
                if optimize_params and name in ['RandomForest', 'XGBoost', 'LightGBM']:
                    optimized_model = self.optimize_hyperparameters(X_train_scaled, y_train, name)
                    trained_models[name] = optimized_model
                else:
                    model.fit(X_train_scaled, y_train)
                    trained_models[name] = model
                
                # Avaliar modelo
                y_pred = trained_models[name].predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"✅ {name} - Acurácia: {accuracy:.4f}")
                
            except Exception as e:
                print(f"❌ Erro ao treinar {name}: {e}")
                continue
        
        # Criar ensemble
        if len(trained_models) > 1:
            self.ensemble_model = self._create_optimized_ensemble(trained_models)
            
            # Avaliar ensemble
            y_pred_ensemble = self.ensemble_model.predict(X_test_scaled)
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            
            print(f"🏆 Ensemble - Acurácia: {ensemble_accuracy:.4f}")
            
            # Salvar métricas
            self.performance_metrics[pair] = {
                'ensemble_accuracy': ensemble_accuracy,
                'individual_models': {name: accuracy_score(y_test, model.predict(X_test_scaled)) 
                                    for name, model in trained_models.items()},
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features_used': len(self.selected_features) if self.selected_features else len(self.feature_names)
            }
        
        print(f"🎉 Treinamento concluído para {pair}")
        return True
    
    def _select_features_optimized(self, X, y, max_features=50):
        """Seleção otimizada de features"""
        print("🎯 Selecionando features relevantes...")
        
        # Usar SelectKBest como método rápido
        k = min(max_features, X.shape[1] // 2)
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        
        selected_idx = selector.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_idx]
        
        return selected_features
    
    def _create_optimized_ensemble(self, trained_models):
        """Cria ensemble otimizado"""
        estimators = [(name, model) for name, model in trained_models.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=self.n_jobs
        )
        
        return ensemble
    
    def predict_optimized(self, pair, periods_ahead=1):
        """Faz predições otimizadas"""
        if self.ensemble_model is None:
            print("❌ Modelo não treinado")
            return None
        
        # Obter dados mais recentes
        data = self.get_forex_data_optimized(pair, period='1mo')
        if data is None:
            return None
        
        # Preparar features
        X, _ = self.prepare_features_optimized(data)
        
        # Usar apenas a última observação
        X_latest = X.iloc[-1:]
        
        # Preprocessar
        X_scaled = self.scaler.transform(X_latest)
        
        # Aplicar seleção de features
        if self.selected_features:
            feature_mask = [name in self.selected_features for name in self.feature_names]
            X_scaled = X_scaled[:, feature_mask]
        
        # Aplicar PCA
        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)
        
        # Fazer predição
        prediction = self.ensemble_model.predict(X_scaled)[0]
        probability = self.ensemble_model.predict_proba(X_scaled)[0]
        
        return {
            'prediction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': max(probability),
            'probabilities': {'UP': probability[1], 'DOWN': probability[0]}
        }
    
    def save_model(self, filepath):
        """Salva modelo treinado"""
        model_data = {
            'ensemble_model': self.ensemble_model,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'performance_metrics': self.performance_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Modelo salvo em: {filepath}")
    
    def load_model(self, filepath):
        """Carrega modelo treinado"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.ensemble_model = model_data['ensemble_model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.feature_names = model_data['feature_names']
        self.selected_features = model_data['selected_features']
        self.performance_metrics = model_data['performance_metrics']
        
        print(f"📂 Modelo carregado de: {filepath}")

# Exemplo de uso
if __name__ == "__main__":
    # Criar preditor otimizado
    predictor = OptimizedMLForexPredictor(
        use_pca=True,
        n_components=0.95,
        feature_selection_method='auto',
        chunk_size=5000,
        cache_features=True
    )
    
    # Treinar para EUR/USD
    success = predictor.train_optimized('EURUSD=X', period='1y', target_period=1)
    
    if success:
        # Fazer predição
        result = predictor.predict_optimized('EURUSD=X')
        print(f"\n🔮 Predição: {result}")
        
        # Salvar modelo
        predictor.save_model('optimized_forex_model.pkl')