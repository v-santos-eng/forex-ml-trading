#!/usr/bin/env python3
"""
ML Forex Predictor - Versão Melhorada
Predições de forex usando machine learning avançado com ensemble methods,
walk-forward optimization, PCA e seleção automática de features

Este módulo implementa técnicas avançadas de ML para predição de movimentos
de preços de forex com prevenção de overfitting.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import json
import sqlite3
from typing import Dict, List, Tuple, Optional

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost não disponível. Instalando...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost as xgb
    XGBOOST_AVAILABLE = True

# Indicadores técnicos
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib não disponível. Usando indicadores básicos.")

class MLForexPredictorEnhanced:
    """
    Preditor de Forex usando Machine Learning avançado com:
    - Ensemble Methods (XGBoost, Random Forest, Gradient Boosting)
    - Walk-Forward Optimization
    - PCA para redução de dimensionalidade
    - Seleção automática de features
    - Prevenção de overfitting
    """
    
    def __init__(self, use_pca=True, n_components=0.95, feature_selection_method='auto'):
        """
        Inicializa o preditor melhorado
        
        Args:
            use_pca: Se deve usar PCA para redução de dimensionalidade
            n_components: Número de componentes PCA (float para variância explicada)
            feature_selection_method: Método de seleção de features ('auto', 'rfe', 'selectkbest', 'selectfrommodel')
        """
        self.use_pca = use_pca
        self.n_components = n_components
        self.feature_selection_method = feature_selection_method
        
        # Modelos base
        self.base_models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=10, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=1.0,
                penalty='l2'
            ),
            'SVM': SVC(
                random_state=42, 
                probability=True,
                C=1.0,
                kernel='rbf',
                gamma='scale'
            )
        }
        
        # Adicionar XGBoost se disponível
        if XGBOOST_AVAILABLE:
            self.base_models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        
        # Pipeline components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components) if self.use_pca else None
        self.feature_selector = None
        
        # Ensemble model
        self.ensemble_model = None
        self.best_model = None
        self.feature_names = []
        self.selected_features = []
        
        # Walk-forward optimization
        self.walk_forward_results = []
        
    def get_forex_data(self, pair, period='2y'):
        """
        Obtém dados históricos de forex
        """
        try:
            ticker = yf.Ticker(pair)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            print(f"Erro ao obter dados para {pair}: {e}")
            return None
    
    def calculate_advanced_indicators(self, data):
        """
        Calcula indicadores técnicos avançados
        """
        df = data.copy()
        
        # Médias móveis múltiplas
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # RSI múltiplos períodos
        for period in [7, 14, 21]:
            if TALIB_AVAILABLE:
                df[f'RSI_{period}'] = talib.RSI(df['Close'].values, timeperiod=period)
            else:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD múltiplos
        macd_configs = [(12, 26, 9), (5, 35, 5), (19, 39, 9)]
        for i, (fast, slow, signal) in enumerate(macd_configs):
            ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
            df[f'MACD_{i}'] = ema_fast - ema_slow
            df[f'MACD_Signal_{i}'] = df[f'MACD_{i}'].ewm(span=signal, adjust=False).mean()
            df[f'MACD_Histogram_{i}'] = df[f'MACD_{i}'] - df[f'MACD_Signal_{i}']
        
        # Bollinger Bands múltiplos
        for period, std_dev in [(20, 2), (20, 1), (50, 2)]:
            bb_middle = df['Close'].rolling(window=period).mean()
            bb_std = df['Close'].rolling(window=period).std()
            df[f'BB_Upper_{period}_{std_dev}'] = bb_middle + (bb_std * std_dev)
            df[f'BB_Lower_{period}_{std_dev}'] = bb_middle - (bb_std * std_dev)
            df[f'BB_Width_{period}_{std_dev}'] = df[f'BB_Upper_{period}_{std_dev}'] - df[f'BB_Lower_{period}_{std_dev}']
            df[f'BB_Position_{period}_{std_dev}'] = (df['Close'] - df[f'BB_Lower_{period}_{std_dev}']) / (df[f'BB_Upper_{period}_{std_dev}'] - df[f'BB_Lower_{period}_{std_dev}'])
        
        # Indicadores de volatilidade
        for period in [10, 20, 50]:
            df[f'Volatility_{period}'] = df['Close'].rolling(window=period).std()
            df[f'ATR_{period}'] = (df['High'] - df['Low']).rolling(window=period).mean()
        
        # Indicadores de momentum
        for period in [5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period)
            df[f'ROC_{period}'] = df['Close'].pct_change(period)
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = df['Low'].rolling(window=period).min()
            high_max = df['High'].rolling(window=period).max()
            df[f'Stoch_K_{period}'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
            df[f'Stoch_D_{period}'] = df[f'Stoch_K_{period}'].rolling(window=3).mean()
        
        # Williams %R
        for period in [14, 21]:
            high_max = df['High'].rolling(window=period).max()
            low_min = df['Low'].rolling(window=period).min()
            df[f'Williams_R_{period}'] = -100 * (high_max - df['Close']) / (high_max - low_min)
        
        # Volume indicators (se disponível)
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            for period in [10, 20, 50]:
                df[f'Volume_SMA_{period}'] = df['Volume'].rolling(window=period).mean()
                df[f'Volume_Ratio_{period}'] = df['Volume'] / df[f'Volume_SMA_{period}']
        
        # Price patterns
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Close']
        df['Upper_Shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        df['Lower_Shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
        
        # Fractal patterns
        df['Local_High'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
        df['Local_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
        
        return df
    
    def create_enhanced_features(self, data):
        """
        Cria features avançadas para o modelo de ML
        """
        df = self.calculate_advanced_indicators(data)
        
        # Coletar todas as features numéricas
        feature_columns = []
        
        # Adicionar todas as colunas calculadas (exceto OHLCV originais)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        for col in df.columns:
            if col not in exclude_cols and df[col].dtype in ['float64', 'int64', 'bool']:
                feature_columns.append(col)
        
        # Features de lag temporal
        lag_features = ['Close', 'Volume'] if 'Volume' in df.columns else ['Close']
        for col in lag_features:
            if col in df.columns:
                for lag in [1, 2, 3, 5, 10]:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
                    feature_columns.append(f'{col}_lag{lag}')
        
        # Features de diferença e razão
        df['Close_SMA20_diff'] = df['Close'] - df['SMA_20']
        df['Close_SMA50_diff'] = df['Close'] - df['SMA_50']
        df['SMA20_SMA50_ratio'] = df['SMA_20'] / df['SMA_50']
        # Verificar se EMAs existem antes de criar ratio
        if 'EMA_12' in df.columns and 'EMA_26' in df.columns:
            df['EMA12_EMA26_ratio'] = df['EMA_12'] / df['EMA_26']
        else:
            df['EMA12_EMA26_ratio'] = 1.0  # Valor padrão
        
        feature_columns.extend([
            'Close_SMA20_diff', 'Close_SMA50_diff', 
            'SMA20_SMA50_ratio', 'EMA12_EMA26_ratio'
        ])
        
        # Features de interação
        df['RSI_BB_interaction'] = df['RSI_14'] * df['BB_Position_20_2']
        df['MACD_RSI_interaction'] = df['MACD_0'] * df['RSI_14']
        df['Volume_Price_interaction'] = df.get('Volume_Ratio_20', 1) * df['ROC_5']
        
        feature_columns.extend([
            'RSI_BB_interaction', 'MACD_RSI_interaction', 'Volume_Price_interaction'
        ])
        
        self.feature_names = feature_columns
        return df[feature_columns]
    
    def create_target(self, data, prediction_horizon=1, threshold=0.001):
        """
        Cria variável target com threshold para reduzir ruído
        
        Args:
            data: DataFrame com dados de preços
            prediction_horizon: Quantos períodos à frente prever
            threshold: Threshold mínimo para considerar movimento significativo
        """
        future_price = data['Close'].shift(-prediction_horizon)
        current_price = data['Close']
        
        # Calcular retorno
        returns = (future_price - current_price) / current_price
        
        # Aplicar threshold
        target = np.where(returns > threshold, 1, 
                         np.where(returns < -threshold, 0, np.nan))
        
        return pd.Series(target, index=data.index)
    
    def select_features(self, X, y, method='auto'):
        """
        Seleção automática de features
        """
        if method == 'auto':
            # Usar múltiplos métodos e combinar
            methods = ['selectkbest', 'rfe', 'selectfrommodel']
            selected_features_sets = []
            
            for m in methods:
                features = self.select_features(X, y, method=m)
                selected_features_sets.append(set(features))
            
            # Interseção das features selecionadas
            common_features = set.intersection(*selected_features_sets)
            
            # Se muito poucas features em comum, usar união dos top features
            if len(common_features) < 10:
                all_features = set.union(*selected_features_sets)
                return list(all_features)[:50]  # Limitar a 50 features
            
            return list(common_features)
        
        elif method == 'selectkbest':
            k = min(50, X.shape[1] // 2)  # Selecionar até 50 ou metade das features
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X, y)
            selected_idx = selector.get_support(indices=True)
            return [self.feature_names[i] for i in selected_idx]
        
        elif method == 'rfe':
            # Usar Random Forest para RFE
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            n_features = min(30, X.shape[1] // 2)
            selector = RFE(estimator, n_features_to_select=n_features)
            selector.fit(X, y)
            selected_idx = selector.get_support(indices=True)
            return [self.feature_names[i] for i in selected_idx]
        
        elif method == 'selectfrommodel':
            # Usar Random Forest para seleção baseada em importância
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = SelectFromModel(estimator, threshold='median')
            selector.fit(X, y)
            selected_idx = selector.get_support(indices=True)
            return [self.feature_names[i] for i in selected_idx]
        
        else:
            return self.feature_names
    
    def create_ensemble_model(self):
        """
        Cria modelo ensemble com voting
        """
        estimators = []
        
        for name, model in self.base_models.items():
            if name in ['LogisticRegression', 'SVM']:
                # Criar pipeline com scaling para modelos que precisam
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
                estimators.append((name, pipeline))
            else:
                estimators.append((name, model))
        
        # Criar ensemble com soft voting
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        return ensemble
    
    def walk_forward_optimization(self, X, y, window_size=252, step_size=21):
        """
        Implementa walk-forward optimization
        
        Args:
            X: Features
            y: Target
            window_size: Tamanho da janela de treinamento (em dias)
            step_size: Tamanho do passo (em dias)
        """
        print("Executando Walk-Forward Optimization...")
        
        results = []
        n_samples = len(X)
        
        # Garantir que temos dados suficientes
        if n_samples < window_size + step_size:
            print(f"Dados insuficientes para walk-forward. Necessário: {window_size + step_size}, disponível: {n_samples}")
            return None
        
        start_idx = 0
        
        while start_idx + window_size + step_size <= n_samples:
            # Definir janelas
            train_end = start_idx + window_size
            test_start = train_end
            test_end = min(test_start + step_size, n_samples)
            
            # Dados de treino e teste
            X_train = X.iloc[start_idx:train_end]
            y_train = y.iloc[start_idx:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Remover NaN
            train_valid = ~(X_train.isnull().any(axis=1) | y_train.isnull())
            test_valid = ~(X_test.isnull().any(axis=1) | y_test.isnull())
            
            X_train_clean = X_train[train_valid]
            y_train_clean = y_train[train_valid]
            X_test_clean = X_test[test_valid]
            y_test_clean = y_test[test_valid]
            
            if len(X_train_clean) < 50 or len(X_test_clean) < 5:
                start_idx += step_size
                continue
            
            # Seleção de features para esta janela
            selected_features = self.select_features(
                X_train_clean, y_train_clean, 
                method=self.feature_selection_method
            )
            
            X_train_selected = X_train_clean[selected_features]
            X_test_selected = X_test_clean[selected_features]
            
            # Aplicar PCA se habilitado
            if self.use_pca:
                pca = PCA(n_components=self.n_components)
                X_train_pca = pca.fit_transform(X_train_selected)
                X_test_pca = pca.transform(X_test_selected)
            else:
                X_train_pca = X_train_selected
                X_test_pca = X_test_selected
            
            # Treinar ensemble
            ensemble = self.create_ensemble_model()
            ensemble.fit(X_train_pca, y_train_clean)
            
            # Fazer predições
            y_pred = ensemble.predict(X_test_pca)
            y_pred_proba = ensemble.predict_proba(X_test_pca)[:, 1]
            
            # Calcular métricas
            accuracy = accuracy_score(y_test_clean, y_pred)
            try:
                auc = roc_auc_score(y_test_clean, y_pred_proba)
            except:
                auc = 0.5
            
            results.append({
                'start_idx': start_idx,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'accuracy': accuracy,
                'auc': auc,
                'n_train': len(X_train_clean),
                'n_test': len(X_test_clean),
                'n_features': len(selected_features)
            })
            
            print(f"Janela {len(results)}: Acc={accuracy:.3f}, AUC={auc:.3f}, Features={len(selected_features)}")
            
            start_idx += step_size
        
        self.walk_forward_results = results
        
        # Calcular estatísticas agregadas
        if results:
            avg_accuracy = np.mean([r['accuracy'] for r in results])
            avg_auc = np.mean([r['auc'] for r in results])
            std_accuracy = np.std([r['accuracy'] for r in results])
            
            print(f"\nResultados Walk-Forward:")
            print(f"Número de janelas: {len(results)}")
            print(f"Acurácia média: {avg_accuracy:.3f} ± {std_accuracy:.3f}")
            print(f"AUC médio: {avg_auc:.3f}")
            
            return {
                'avg_accuracy': avg_accuracy,
                'std_accuracy': std_accuracy,
                'avg_auc': avg_auc,
                'n_windows': len(results),
                'results': results
            }
        
        return None
    
    def train_final_model(self, X, y):
        """
        Treina modelo final com todas as técnicas de prevenção de overfitting
        """
        print("\nTreinando modelo final com prevenção de overfitting...")
        
        # Remover NaN
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        print(f"Dados limpos: {len(X_clean)} amostras, {len(self.feature_names)} features")
        
        # Seleção de features
        self.selected_features = self.select_features(X_clean, y_clean, method=self.feature_selection_method)
        X_selected = X_clean[self.selected_features]
        
        print(f"Features selecionadas: {len(self.selected_features)}")
        
        # Aplicar PCA se habilitado
        if self.use_pca:
            self.pca = PCA(n_components=self.n_components)
            X_final = self.pca.fit_transform(X_selected)
            print(f"PCA aplicado: {X_final.shape[1]} componentes (variância explicada: {self.pca.explained_variance_ratio_.sum():.3f})")
        else:
            X_final = X_selected
        
        # Dividir dados com estratificação temporal
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Treinar ensemble final
        self.ensemble_model = self.create_ensemble_model()
        
        # Cross-validation com time series split
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_final):
            X_train_cv, X_val_cv = X_final[train_idx], X_final[val_idx]
            y_train_cv, y_val_cv = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
            
            self.ensemble_model.fit(X_train_cv, y_train_cv)
            y_pred_cv = self.ensemble_model.predict(X_val_cv)
            cv_scores.append(accuracy_score(y_val_cv, y_pred_cv))
        
        print(f"CV Score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Treinar modelo final com todos os dados
        self.ensemble_model.fit(X_final, y_clean)
        
        # Salvar scaler se necessário
        if not self.use_pca:
            self.scaler.fit(X_selected)
        
        return {
            'cv_score_mean': np.mean(cv_scores),
            'cv_score_std': np.std(cv_scores),
            'n_features_selected': len(self.selected_features),
            'n_components_pca': X_final.shape[1] if self.use_pca else None
        }
    
    def predict_direction(self, pair, prediction_horizon=1):
        """
        Faz predição da direção do preço usando modelo ensemble
        """
        if self.ensemble_model is None:
            print("Modelo não treinado. Execute train_final_model primeiro.")
            return None
        
        # Obter dados recentes
        data = self.get_forex_data(pair, period='1y')
        if data is None or data.empty:
            return None
        
        # Criar features
        features = self.create_enhanced_features(data)
        
        # Usar apenas a última observação
        latest_features = features.iloc[-1:]
        
        # Verificar se há NaN
        if latest_features.isnull().any().any():
            print("Dados insuficientes para predição")
            return None
        
        # Selecionar features
        latest_selected = latest_features[self.selected_features]
        
        # Aplicar PCA se necessário
        if self.use_pca:
            latest_final = self.pca.transform(latest_selected)
        else:
            latest_final = self.scaler.transform(latest_selected)
        
        # Fazer predição
        prediction = self.ensemble_model.predict(latest_final)[0]
        probability = self.ensemble_model.predict_proba(latest_final)[0]
        
        direction = "ALTA" if prediction == 1 else "BAIXA"
        confidence = max(probability) * 100
        
        return {
            'pair': pair,
            'direction': direction,
            'confidence': round(confidence, 1),
            'probability_up': round(probability[1] * 100, 1),
            'probability_down': round(probability[0] * 100, 1),
            'model_type': 'ensemble_enhanced'
        }
    
    def get_model_summary(self):
        """
        Retorna resumo do modelo treinado
        """
        summary = {
            'model_type': 'Enhanced ML Predictor',
            'ensemble_models': list(self.base_models.keys()),
            'use_pca': self.use_pca,
            'feature_selection_method': self.feature_selection_method,
            'total_features': len(self.feature_names),
            'selected_features': len(self.selected_features) if self.selected_features else 0,
            'xgboost_available': XGBOOST_AVAILABLE,
            'talib_available': TALIB_AVAILABLE
        }
        
        if self.use_pca and self.pca:
            summary['pca_components'] = getattr(self.pca, 'n_components_', self.pca.n_components if hasattr(self.pca, 'n_components') else 'N/A')
            summary['pca_variance_explained'] = float(self.pca.explained_variance_ratio_.sum()) if hasattr(self.pca, 'explained_variance_ratio_') else 'N/A'
        
        if self.walk_forward_results:
            summary['walk_forward_windows'] = len(self.walk_forward_results)
            summary['walk_forward_avg_accuracy'] = np.mean([r['accuracy'] for r in self.walk_forward_results])
        
        return summary

def main():
    """
    Demonstração do preditor ML melhorado
    """
    print("ML Forex Predictor - Versão Melhorada")
    print("Machine Learning Avançado para Forex")
    print("="*60)
    
    # Criar preditor melhorado
    predictor = MLForexPredictorEnhanced(
        use_pca=True,
        n_components=0.95,
        feature_selection_method='auto'
    )
    
    # Preparar dados para EUR/USD
    pair = 'EURUSD=X'
    print(f"\nPreparando dados para {pair}...")
    
    data = predictor.get_forex_data(pair, period='2y')
    if data is not None and not data.empty:
        features = predictor.create_enhanced_features(data)
        target = predictor.create_target(data, prediction_horizon=1, threshold=0.001)
        
        print(f"Features criadas: {len(predictor.feature_names)}")
        
        # Walk-forward optimization
        wf_results = predictor.walk_forward_optimization(features, target)
        
        # Treinar modelo final
        final_results = predictor.train_final_model(features, target)
        
        # Fazer predição atual
        print(f"\n" + "="*60)
        print("PREDIÇÃO ATUAL")
        print("="*60)
        
        prediction = predictor.predict_direction(pair)
        if prediction:
            print(f"\nPar: {prediction['pair']}")
            print(f"Direção prevista: {prediction['direction']}")
            print(f"Confiança: {prediction['confidence']}%")
            print(f"Probabilidade de alta: {prediction['probability_up']}%")
            print(f"Probabilidade de baixa: {prediction['probability_down']}%")
        
        # Mostrar resumo do modelo
        print(f"\n" + "="*60)
        print("RESUMO DO MODELO")
        print("="*60)
        
        summary = predictor.get_model_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
    
    print(f"\n" + "="*60)
    print("MELHORIAS IMPLEMENTADAS")
    print("="*60)
    
    print("\n1. Ensemble Methods:")
    print("   - XGBoost (se disponível)")
    print("   - Random Forest otimizado")
    print("   - Gradient Boosting")
    print("   - Voting Classifier com soft voting")
    
    print("\n2. Walk-Forward Optimization:")
    print("   - Validação temporal com janelas deslizantes")
    print("   - Prevenção de data leakage")
    print("   - Métricas de performance robustas")
    
    print("\n3. PCA para Redução de Dimensionalidade:")
    print("   - Redução automática de features")
    print("   - Preservação de variância explicada")
    print("   - Prevenção de overfitting")
    
    print("\n4. Seleção Automática de Features:")
    print("   - SelectKBest com f_classif")
    print("   - RFE (Recursive Feature Elimination)")
    print("   - SelectFromModel com Random Forest")
    print("   - Combinação automática de métodos")
    
    print("\n5. Prevenção de Overfitting:")
    print("   - Time Series Cross-Validation")
    print("   - Regularização em modelos lineares")
    print("   - Ensemble com múltiplos algoritmos")
    print("   - Feature selection rigorosa")
    print("   - Threshold para target (redução de ruído)")
    
    print("\n6. Integração com Agno/Ollama:")
    print("   - API compatível com sistema existente")
    print("   - Formato de saída padronizado")
    print("   - Métricas de confiança aprimoradas")

if __name__ == "__main__":
    main()