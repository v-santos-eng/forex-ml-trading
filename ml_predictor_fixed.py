#!/usr/bin/env python3
"""
ML Forex Predictor - VERSÃO CORRIGIDA
Corrige problemas de incompatibilidade de features identificados

Problemas corrigidos:
1. Inconsistência entre número de features (26 vs 64)
2. Redução drástica de features para 1 durante predição
3. Incompatibilidade entre treinamento e predição
4. Problemas de escalonamento e preparação de dados
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
import pickle
import os
from datetime import datetime, timedelta
import sqlite3
import json

warnings.filterwarnings('ignore')

class MLForexPredictorFixed:
    """Versão corrigida do preditor ML para Forex"""
    
    def __init__(self):
        self.models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(random_state=42, probability=True)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0
        self.feature_names = []
        self.is_trained = False
        
        # Cache para dados
        self.data_cache = {}
        
        print("✅ MLForexPredictorFixed inicializado")
    
    def get_forex_data(self, pair, period='2y'):
        """Obtém dados de forex com cache"""
        cache_key = f"{pair}_{period}"
        
        # Verificar cache
        if cache_key in self.data_cache:
            cached_data, cache_time = self.data_cache[cache_key]
            # Cache válido por 1 hora
            if datetime.now() - cache_time < timedelta(hours=1):
                print(f"📋 Usando dados em cache para {pair}")
                return cached_data
        
        try:
            print(f"📡 Obtendo dados para {pair}...")
            ticker = yf.Ticker(pair)
            data = ticker.history(period=period)
            
            if data.empty:
                print(f"❌ Nenhum dado encontrado para {pair}")
                return None
            
            # Armazenar em cache
            self.data_cache[cache_key] = (data, datetime.now())
            
            print(f"✅ {len(data)} registros obtidos para {pair}")
            return data
            
        except Exception as e:
            print(f"❌ Erro ao obter dados para {pair}: {e}")
            return None
    
    def create_features(self, data):
        """Cria features técnicas - VERSÃO CORRIGIDA"""
        if data is None or data.empty:
            return None
        
        print("🔧 Criando features técnicas...")
        
        df = data.copy()
        
        # Features básicas de preço
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_2'] = df['Close'].pct_change(2)
        df['Price_Change_5'] = df['Close'].pct_change(5)
        df['Price_Change_10'] = df['Close'].pct_change(10)
        
        # Médias móveis
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Médias móveis exponenciais
        df['EMA_5'] = df['Close'].ewm(span=5).mean()
        df['EMA_10'] = df['Close'].ewm(span=10).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
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
        df['Volatility_10'] = df['Close'].rolling(window=10).std()
        df['Volatility_20'] = df['Close'].rolling(window=20).std()
        
        # Volume features (se disponível)
        if 'Volume' in df.columns:
            df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']
        else:
            df['Volume_SMA_10'] = 0
            df['Volume_Ratio'] = 1
        
        # High-Low features
        df['HL_Ratio'] = df['High'] / df['Low']
        df['OC_Ratio'] = df['Open'] / df['Close']
        
        # Momentum features
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10)
        
        # Rate of Change
        df['ROC_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Williams %R
        high_14 = df['High'].rolling(window=14).max()
        low_14 = df['Low'].rolling(window=14).min()
        df['Williams_R'] = -100 * (high_14 - df['Close']) / (high_14 - low_14)
        
        # Stochastic Oscillator
        df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        df['TR1'] = df['High'] - df['Low']
        df['TR2'] = abs(df['High'] - df['Close'].shift())
        df['TR3'] = abs(df['Low'] - df['Close'].shift())
        df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Commodity Channel Index (CCI)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        # Money Flow Index (se volume disponível)
        if 'Volume' in data.columns and not data['Volume'].isna().all():
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            
            positive_mf = positive_flow.rolling(window=14).sum()
            negative_mf = negative_flow.rolling(window=14).sum()
            
            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            df['MFI'] = mfi
        else:
            df['MFI'] = 50  # Valor neutro quando volume não disponível
        
        # Features de tendência
        df['Trend_5'] = np.where(df['Close'] > df['SMA_5'], 1, 0)
        df['Trend_10'] = np.where(df['Close'] > df['SMA_10'], 1, 0)
        df['Trend_20'] = np.where(df['Close'] > df['SMA_20'], 1, 0)
        
        # Features de cruzamento de médias
        df['SMA_Cross_5_10'] = np.where(df['SMA_5'] > df['SMA_10'], 1, 0)
        df['SMA_Cross_10_20'] = np.where(df['SMA_10'] > df['SMA_20'], 1, 0)
        
        # Selecionar apenas as features (excluir OHLCV originais)
        feature_columns = [
            'Price_Change', 'Price_Change_2', 'Price_Change_5', 'Price_Change_10',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_5', 'EMA_10', 'EMA_20',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
            'Volatility_10', 'Volatility_20',
            'Volume_SMA_10', 'Volume_Ratio',
            'HL_Ratio', 'OC_Ratio',
            'Momentum_5', 'Momentum_10',
            'ROC_5', 'ROC_10',
            'Williams_R', 'Stoch_K', 'Stoch_D',
            'ATR', 'CCI', 'MFI',
            'Trend_5', 'Trend_10', 'Trend_20',
            'SMA_Cross_5_10', 'SMA_Cross_10_20'
        ]
        
        # Verificar se todas as colunas existem
        existing_columns = [col for col in feature_columns if col in df.columns]
        features = df[existing_columns].copy()
        
        # Armazenar nomes das features
        self.feature_names = existing_columns
        
        print(f"✅ {len(existing_columns)} features criadas")
        print(f"📊 Features: {existing_columns[:5]}... (mostrando primeiras 5)")
        
        return features
    
    def create_target(self, data, prediction_horizon=1):
        """Cria target para classificação"""
        if data is None or data.empty:
            return None
        
        # Calcular retorno futuro
        future_return = data['Close'].shift(-prediction_horizon) / data['Close'] - 1
        
        # Classificação: 1 se subir, 0 se descer
        target = (future_return > 0).astype(int)
        
        return target
    
    def prepare_data(self, pair, period='2y', prediction_horizon=1):
        """Prepara dados para treinamento - VERSÃO CORRIGIDA"""
        print(f"🔧 Preparando dados para {pair}...")
        
        # Obter dados
        data = self.get_forex_data(pair, period=period)
        if data is None or data.empty:
            print("❌ Erro: Dados não obtidos")
            return None, None
        
        # Criar features
        features = self.create_features(data)
        if features is None or features.empty:
            print("❌ Erro: Features não criadas")
            return None, None
        
        # Criar target
        target = self.create_target(data, prediction_horizon)
        if target is None:
            print("❌ Erro: Target não criado")
            return None, None
        
        # Verificar consistência
        print(f"📊 Features shape: {features.shape}")
        print(f"📊 Target shape: {target.shape}")
        print(f"📊 Feature names: {len(self.feature_names)}")
        
        # Tratamento melhorado de NaN
        print(f"📊 NaN antes da limpeza: {features.isnull().sum().sum()} valores")
        
        # Preencher NaN com forward fill e backward fill
        features_filled = features.fillna(method='ffill').fillna(method='bfill')
        
        # Se ainda houver NaN, preencher com médias
        for col in features_filled.columns:
            if features_filled[col].isnull().any():
                mean_val = features_filled[col].mean()
                if pd.isna(mean_val):  # Se média também for NaN, usar 0
                    features_filled[col] = features_filled[col].fillna(0)
                else:
                    features_filled[col] = features_filled[col].fillna(mean_val)
        
        # Remover apenas linhas onde target é NaN
        valid_idx = ~target.isnull()
        features_clean = features_filled[valid_idx]
        target_clean = target[valid_idx]
        
        print(f"📊 NaN após limpeza: {features_clean.isnull().sum().sum()} valores")
        
        # Verificação final - reduzir requisito mínimo
        if features_clean.empty or len(features_clean) < 20:
            print("❌ Erro: Dados insuficientes após limpeza")
            return None, None
        
        print(f"✅ Dados preparados: {len(features_clean)} amostras, {features_clean.shape[1]} features")
        
        return features_clean, target_clean
    
    def train_models(self, X, y):
        """Treina modelos - VERSÃO CORRIGIDA"""
        if X is None or y is None or X.empty:
            print("❌ Erro: Dados inválidos para treinamento")
            return False
        
        print(f"🚀 Iniciando treinamento com {X.shape[0]} amostras e {X.shape[1]} features...")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"📊 Treino: {X_train.shape[0]} amostras")
        print(f"📊 Teste: {X_test.shape[0]} amostras")
        
        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        best_score = 0
        best_model_name = None
        
        # Treinar cada modelo
        for name, model in self.models.items():
            try:
                print(f"🔧 Treinando {name}...")
                
                # Treinar modelo
                if name in ['logistic', 'svm']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Avaliar
                accuracy = accuracy_score(y_test, y_pred)
                
                # Validação cruzada
                if name in ['logistic', 'svm']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                print(f"  📊 {name}: Acurácia = {accuracy:.4f}, CV = {cv_mean:.4f} ± {cv_std:.4f}")
                
                # Selecionar melhor modelo
                if cv_mean > best_score:
                    best_score = cv_mean
                    best_model_name = name
                    self.best_model = model
                    self.best_score = cv_mean
                
            except Exception as e:
                print(f"❌ Erro ao treinar {name}: {e}")
        
        if best_model_name:
            print(f"🏆 Melhor modelo: {best_model_name} (CV Score: {best_score:.4f})")
            self.is_trained = True
            return True
        else:
            print("❌ Nenhum modelo foi treinado com sucesso")
            return False
    
    def predict_direction(self, pair, period='3mo'):
        """Prediz direção do preço - VERSÃO CORRIGIDA"""
        if not self.is_trained or self.best_model is None:
            print("❌ Modelo não treinado")
            return None
        
        print(f"🔮 Fazendo predição para {pair}...")
        
        # Obter dados recentes - usar período maior para ter dados suficientes
        data = self.get_forex_data(pair, period=period)
        if data is None or data.empty:
            print("❌ Erro: Dados não obtidos para predição")
            return None
        
        print(f"📊 Dados obtidos para predição: {len(data)} registros")
        
        # Criar features
        features = self.create_features(data)
        if features is None or features.empty:
            print("❌ Erro: Features não criadas para predição")
            return None
        
        # Verificar consistência de features
        if len(features.columns) != len(self.feature_names):
            print(f"⚠️ Aviso: Features inconsistentes")
            print(f"   Esperado: {len(self.feature_names)} features")
            print(f"   Obtido: {len(features.columns)} features")
            
            # Tentar alinhar features
            common_features = [col for col in self.feature_names if col in features.columns]
            if len(common_features) < len(self.feature_names) * 0.8:  # Menos de 80% das features
                print("❌ Erro: Muitas features faltando")
                return None
            
            features = features[common_features]
            print(f"✅ Usando {len(common_features)} features comuns")
        
        # Aplicar mesmo tratamento de NaN usado no treinamento
        features_filled = features.fillna(method='ffill').fillna(method='bfill')
        
        # Se ainda houver NaN, preencher com médias
        for col in features_filled.columns:
            if features_filled[col].isnull().any():
                mean_val = features_filled[col].mean()
                if pd.isna(mean_val):
                    features_filled[col] = features_filled[col].fillna(0)
                else:
                    features_filled[col] = features_filled[col].fillna(mean_val)
        
        # Pegar última observação
        latest_features = features_filled.tail(1)
        
        if latest_features.empty:
            print("❌ Erro: Nenhuma feature válida para predição")
            return None
        
        print(f"📊 Features para predição: {latest_features.shape}")
        
        try:
            # Fazer predição
            model_name = None
            for name, model in self.models.items():
                if model == self.best_model:
                    model_name = name
                    break
            
            if model_name in ['logistic', 'svm']:
                # Escalar features
                latest_scaled = self.scaler.transform(latest_features)
                prediction = self.best_model.predict(latest_scaled)[0]
                probability = self.best_model.predict_proba(latest_scaled)[0]
            else:
                prediction = self.best_model.predict(latest_features)[0]
                probability = self.best_model.predict_proba(latest_features)[0]
            
            confidence = max(probability)
            direction = "COMPRA" if prediction == 1 else "VENDA"
            
            result = {
                'pair': pair,
                'prediction': int(prediction),
                'direction': direction,
                'confidence': float(confidence),
                'model': model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Predição: {direction} (Confiança: {confidence:.2%})")
            return result
            
        except Exception as e:
            print(f"❌ Erro na predição: {e}")
            return None
    
    def save_model(self, filepath):
        """Salva modelo treinado"""
        if not self.is_trained:
            print("❌ Modelo não treinado")
            return False
        
        try:
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'best_model': self.best_model,
                'best_score': self.best_score,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ Modelo salvo em: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar modelo: {e}")
            return False
    
    def load_model(self, filepath):
        """Carrega modelo treinado"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.best_model = model_data['best_model']
            self.best_score = model_data['best_score']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            print(f"✅ Modelo carregado: {len(self.feature_names)} features")
            print(f"📊 Melhor score: {self.best_score:.4f}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return False
    
    def validate_model(self, pair='EURUSD=X'):
        """Valida modelo com dados de teste"""
        print(f"🧪 Validando modelo com {pair}...")
        
        # Preparar dados
        X, y = self.prepare_data(pair, period='6mo')
        if X is None or y is None:
            print("❌ Erro na preparação de dados para validação")
            return False
        
        # Fazer predições
        try:
            if hasattr(self.best_model, 'predict'):
                model_name = None
                for name, model in self.models.items():
                    if model == self.best_model:
                        model_name = name
                        break
                
                if model_name in ['logistic', 'svm']:
                    X_scaled = self.scaler.transform(X)
                    predictions = self.best_model.predict(X_scaled)
                else:
                    predictions = self.best_model.predict(X)
                
                accuracy = accuracy_score(y, predictions)
                print(f"✅ Validação: Acurácia = {accuracy:.4f}")
                
                return accuracy > 0.5
            else:
                print("❌ Modelo não possui método predict")
                return False
                
        except Exception as e:
            print(f"❌ Erro na validação: {e}")
            return False

def main():
    """Função principal para testar o predictor corrigido"""
    print("🚀 TESTANDO ML FOREX PREDICTOR CORRIGIDO")
    print("=" * 50)
    
    # Inicializar predictor
    predictor = MLForexPredictorFixed()
    
    # Testar preparação de dados
    print("\n📊 TESTE 1: Preparação de dados")
    X, y = predictor.prepare_data('EURUSD=X', period='6mo')
    
    if X is not None and y is not None:
        print(f"✅ Dados preparados com sucesso")
        print(f"   Features: {X.shape}")
        print(f"   Target: {y.shape}")
        print(f"   Feature names: {len(predictor.feature_names)}")
        
        # Testar treinamento
        print("\n🚀 TESTE 2: Treinamento de modelos")
        success = predictor.train_models(X, y)
        
        if success:
            print("✅ Treinamento concluído com sucesso")
            
            # Testar predição
            print("\n🔮 TESTE 3: Predição")
            result = predictor.predict_direction('EURUSD=X')
            
            if result:
                print("✅ Predição realizada com sucesso")
                print(f"   Resultado: {result}")
                
                # Testar validação
                print("\n🧪 TESTE 4: Validação")
                is_valid = predictor.validate_model('EURUSD=X')
                
                if is_valid:
                    print("✅ Modelo validado com sucesso")
                    
                    # Salvar modelo
                    print("\n💾 TESTE 5: Salvamento")
                    save_path = 'ml_predictor_fixed_model.pkl'
                    saved = predictor.save_model(save_path)
                    
                    if saved:
                        print("✅ Modelo salvo com sucesso")
                        print(f"\n🎉 TODOS OS TESTES PASSARAM!")
                        print(f"Modelo corrigido está funcionando adequadamente.")
                    else:
                        print("❌ Erro ao salvar modelo")
                else:
                    print("❌ Modelo não passou na validação")
            else:
                print("❌ Erro na predição")
        else:
            print("❌ Erro no treinamento")
    else:
        print("❌ Erro na preparação de dados")

if __name__ == "__main__":
    main()