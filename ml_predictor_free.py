#!/usr/bin/env python3
"""
ML Forex Predictor - Versão Gratuita
Predições de forex usando machine learning com bibliotecas gratuitas

Este módulo implementa modelos de ML para predição de movimentos
de preços de forex usando apenas bibliotecas open-source.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Indicadores técnicos
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib não disponível. Usando indicadores básicos.")

class MLForexPredictorFree:
    """
    Preditor de Forex usando Machine Learning gratuito
    """
    
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = []
        
    def get_forex_data(self, pair, period='1y'):
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
    
    def calculate_technical_indicators(self, data):
        """
        Calcula indicadores técnicos para features
        """
        df = data.copy()
        
        # Médias móveis
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        if TALIB_AVAILABLE:
            df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
        else:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volatilidade
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Volume (se disponível)
        if 'Volume' in df.columns:
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5'] = df['Close'].pct_change(5)
        df['High_Low_Ratio'] = df['High'] / df['Low']
        
        # Momentum
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10)
        
        return df
    
    def create_features(self, data):
        """
        Cria features para o modelo de ML
        """
        df = self.calculate_technical_indicators(data)
        
        # Lista de features
        feature_columns = [
            'SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Width', 'BB_Position', 'Volatility',
            'Price_Change', 'Price_Change_5', 'High_Low_Ratio',
            'Momentum_5', 'Momentum_10'
        ]
        
        # Adicionar volume se disponível
        if 'Volume_Ratio' in df.columns:
            feature_columns.extend(['Volume_Ratio'])
        
        # Features de lag (valores anteriores)
        for col in ['Close', 'RSI', 'MACD']:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_lag2'] = df[col].shift(2)
                feature_columns.extend([f'{col}_lag1', f'{col}_lag2'])
        
        # Features de diferença
        df['Close_SMA20_diff'] = df['Close'] - df['SMA_20']
        df['SMA5_SMA20_diff'] = df['SMA_5'] - df['SMA_20']
        feature_columns.extend(['Close_SMA20_diff', 'SMA5_SMA20_diff'])
        
        self.feature_names = feature_columns
        return df[feature_columns]
    
    def create_target(self, data, prediction_horizon=1):
        """
        Cria variável target (1 = alta, 0 = baixa)
        
        Args:
            data: DataFrame com dados de preços
            prediction_horizon: Quantos períodos à frente prever
        """
        future_price = data['Close'].shift(-prediction_horizon)
        current_price = data['Close']
        
        # 1 se o preço futuro for maior que o atual, 0 caso contrário
        target = (future_price > current_price).astype(int)
        
        return target
    
    def prepare_data(self, pair, period='2y', prediction_horizon=1):
        """
        Prepara dados para treinamento
        """
        print(f"Preparando dados para {pair}...")
        
        # Obter dados
        data = self.get_forex_data(pair, period=period)
        if data is None or data.empty:
            return None, None
        
        # Criar features
        features = self.create_features(data)
        
        # Criar target
        target = self.create_target(data, prediction_horizon)
        
        # Remover NaN
        valid_idx = ~(features.isnull().any(axis=1) | target.isnull())
        features_clean = features[valid_idx]
        target_clean = target[valid_idx]
        
        print(f"Dados preparados: {len(features_clean)} amostras, {len(self.feature_names)} features")
        
        return features_clean, target_clean
    
    def train_models(self, X, y):
        """
        Treina múltiplos modelos e seleciona o melhor
        """
        print("\nTreinando modelos...")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Treinando {name}...")
            
            # Treinar modelo
            if name in ['LogisticRegression', 'SVM']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Avaliar modelo
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            if name in ['LogisticRegression', 'SVM']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  Acurácia: {accuracy:.3f}")
            print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Selecionar melhor modelo
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nMelhor modelo: {best_model_name}")
        print(f"Acurácia: {results[best_model_name]['accuracy']:.3f}")
        
        return results
    
    def predict_direction(self, pair, prediction_horizon=1):
        """
        Faz predição da direção do preço
        """
        if self.best_model is None:
            print("Modelo não treinado. Execute train_models primeiro.")
            return None
        
        # Obter dados recentes
        data = self.get_forex_data(pair, period='6mo')
        if data is None or data.empty:
            return None
        
        # Criar features
        features = self.create_features(data)
        
        # Usar apenas a última observação
        latest_features = features.iloc[-1:]
        
        # Verificar se há NaN
        if latest_features.isnull().any().any():
            print("Dados insuficientes para predição")
            return None
        
        # Fazer predição
        if isinstance(self.best_model, (LogisticRegression, SVC)):
            latest_scaled = self.scaler.transform(latest_features)
            prediction = self.best_model.predict(latest_scaled)[0]
            probability = self.best_model.predict_proba(latest_scaled)[0]
        else:
            prediction = self.best_model.predict(latest_features)[0]
            probability = self.best_model.predict_proba(latest_features)[0]
        
        direction = "ALTA" if prediction == 1 else "BAIXA"
        confidence = max(probability) * 100
        
        return {
            'pair': pair,
            'direction': direction,
            'confidence': round(confidence, 1),
            'probability_up': round(probability[1] * 100, 1),
            'probability_down': round(probability[0] * 100, 1)
        }
    
    def feature_importance(self):
        """
        Mostra importância das features (para modelos tree-based)
        """
        if self.best_model is None:
            print("Modelo não treinado")
            return
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\nImportância das Features (Top 10):")
            print(feature_imp.head(10).to_string(index=False))
            
            return feature_imp
        else:
            print("Modelo não suporta feature importance")
    
    def backtest(self, pair, period='1y'):
        """
        Realiza backtest simples da estratégia
        """
        print(f"\nRealizando backtest para {pair}...")
        
        # Obter dados
        data = self.get_forex_data(pair, period=period)
        if data is None or data.empty:
            return None
        
        # Preparar dados
        features = self.create_features(data)
        target = self.create_target(data, 1)
        
        # Remover NaN
        valid_idx = ~(features.isnull().any(axis=1) | target.isnull())
        features_clean = features[valid_idx]
        target_clean = target[valid_idx]
        data_clean = data[valid_idx]
        
        if len(features_clean) < 100:
            print("Dados insuficientes para backtest")
            return None
        
        # Usar 70% para treino, 30% para teste
        split_idx = int(len(features_clean) * 0.7)
        
        X_train = features_clean.iloc[:split_idx]
        y_train = target_clean.iloc[:split_idx]
        X_test = features_clean.iloc[split_idx:]
        y_test = target_clean.iloc[split_idx:]
        test_data = data_clean.iloc[split_idx:]
        
        # Treinar modelo
        if isinstance(self.best_model, (LogisticRegression, SVC)):
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            self.best_model.fit(X_train_scaled, y_train)
            predictions = self.best_model.predict(X_test_scaled)
        else:
            self.best_model.fit(X_train, y_train)
            predictions = self.best_model.predict(X_test)
        
        # Calcular retornos
        test_returns = test_data['Close'].pct_change().dropna()
        
        # Estratégia: comprar quando predição = 1, vender quando = 0
        strategy_returns = []
        for i, pred in enumerate(predictions[:-1]):
            if pred == 1:  # Predição de alta
                strategy_returns.append(test_returns.iloc[i+1])
            else:  # Predição de baixa
                strategy_returns.append(-test_returns.iloc[i+1])
        
        # Métricas de performance
        strategy_returns = np.array(strategy_returns)
        total_return = (1 + strategy_returns).prod() - 1
        accuracy = accuracy_score(y_test[:-1], predictions[:-1])
        
        print(f"\nResultados do Backtest:")
        print(f"Período: {len(strategy_returns)} trades")
        print(f"Acurácia: {accuracy:.3f}")
        print(f"Retorno total: {total_return:.3f} ({total_return*100:.1f}%)")
        print(f"Retorno médio por trade: {strategy_returns.mean():.4f}")
        print(f"Volatilidade: {strategy_returns.std():.4f}")
        
        if strategy_returns.std() > 0:
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            print(f"Sharpe Ratio (anualizado): {sharpe_ratio:.3f}")
        
        return {
            'total_return': total_return,
            'accuracy': accuracy,
            'num_trades': len(strategy_returns),
            'avg_return': strategy_returns.mean(),
            'volatility': strategy_returns.std()
        }

def main():
    """
    Demonstração do preditor ML gratuito
    """
    print("ML Forex Predictor - Versão Gratuita")
    print("Machine Learning para Forex sem APIs pagas")
    print("="*60)
    
    predictor = MLForexPredictorFree()
    
    # Preparar dados para EUR/USD
    pair = 'EURUSD=X'
    X, y = predictor.prepare_data(pair, period='2y')
    
    if X is not None and y is not None:
        # Treinar modelos
        results = predictor.train_models(X, y)
        
        # Mostrar importância das features
        predictor.feature_importance()
        
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
        
        # Realizar backtest
        backtest_results = predictor.backtest(pair)
        
        # Testar outros pares
        print(f"\n" + "="*60)
        print("PREDIÇÕES PARA OUTROS PARES")
        print("="*60)
        
        other_pairs = ['GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
        for test_pair in other_pairs:
            pred = predictor.predict_direction(test_pair)
            if pred:
                print(f"\n{pred['pair']}: {pred['direction']} (Confiança: {pred['confidence']}%)")
    
    print(f"\n" + "="*60)
    print("ANÁLISE CONCLUÍDA")
    print("="*60)
    
    print("\nRecursos utilizados:")
    print("- Scikit-learn (modelos de ML)")
    print("- Yahoo Finance (dados gratuitos)")
    print("- Pandas/Numpy (processamento de dados)")
    if TALIB_AVAILABLE:
        print("- TA-Lib (indicadores técnicos)")
    
    print("\nModelos implementados:")
    print("- Random Forest")
    print("- Gradient Boosting")
    print("- Logistic Regression")
    print("- Support Vector Machine")
    
    print("\nFeatures utilizadas:")
    print("- Indicadores técnicos (RSI, MACD, Bollinger Bands)")
    print("- Médias móveis (SMA, EMA)")
    print("- Momentum e volatilidade")
    print("- Features de lag temporal")
    
    print("\nVantagens:")
    print("- Totalmente gratuito")
    print("- Modelos de ML robustos")
    print("- Backtest automatizado")
    print("- Feature importance")
    
    print("\nLimitações:")
    print("- Apenas dados técnicos")
    print("- Sem análise fundamental")
    print("- Dados limitados do Yahoo Finance")
    print("- Sem processamento de texto/notícias")
    
    print("\nDISCLAIMER:")
    print("Este sistema é apenas para fins educacionais.")
    print("Não constitui aconselhamento financeiro.")
    print("Trading de forex envolve riscos significativos.")
    print("Sempre faça sua própria pesquisa e use stop-loss.")

if __name__ == "__main__":
    main()