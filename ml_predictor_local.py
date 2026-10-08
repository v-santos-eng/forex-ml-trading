#!/usr/bin/env python3
"""
ML Forex Predictor - Versão Local
Predições de forex usando dados locais e machine learning

Esta versão usa dados históricos locais em vez do Yahoo Finance
"""

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

# Importar o mapeador de dados
from forex_data_mapper import ForexDataMapper

# Indicadores técnicos
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib não disponível. Usando indicadores básicos.")

class MLForexPredictorLocal:
    """
    Preditor de Forex usando dados locais e Machine Learning
    """
    
    def __init__(self, data_dir: str = "extracted_fx_data", config: dict = None):
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = []
        self.data_mapper = ForexDataMapper(data_dir)
        
        # Configurações de treinamento
        self.config = config or {}
        self.max_training_samples = self.config.get('ml_config', {}).get('max_training_samples', None)
        self.fast_training_mode = self.config.get('ml_config', {}).get('fast_training_mode', False)
        
    def get_forex_data(self, pair, period='1y'):
        """
        Obtém dados históricos de forex usando dados locais
        """
        try:
            # Garantir que o símbolo tenha o formato correto para Yahoo Finance
            yahoo_symbol = pair
            if not pair.endswith('=X') and len(pair) == 6 and pair.isalpha():
                yahoo_symbol = f"{pair}=X"
            
            print(f"Tentando carregar dados para {yahoo_symbol} (original: {pair})")
            data = self.data_mapper.load_forex_data(yahoo_symbol, period)
            
            if data is None or data.empty:
                print(f"Nenhum dado encontrado para {yahoo_symbol}")
                return pd.DataFrame()
            
            print(f"Dados carregados com sucesso: {len(data)} registros para {yahoo_symbol}")
            return data
        except Exception as e:
            print(f"Erro ao obter dados para {pair}: {e}")
            return pd.DataFrame()
    
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
            # Evitar divisão por zero quando volume é sempre zero
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA'].replace(0, 1)
            # Se todos os volumes são zero, remover a feature
            if df['Volume'].sum() == 0:
                df = df.drop(['Volume_SMA', 'Volume_Ratio'], axis=1)
        
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
        Prepara dados para treinamento usando dados locais
        """
        print(f"Preparando dados para {pair}...")
        
        # Obter dados locais
        data = self.get_forex_data(pair, period=period)
        if data is None or data.empty:
            return None, None
        
        # Criar features
        features = self.create_features(data)
        
        # Criar target
        target = self.create_target(data, prediction_horizon)
        
        # Combinar features e target para remover NaN de forma consistente
        combined = pd.concat([features, target.rename('target')], axis=1)
        
        # Remover todas as linhas com NaN
        combined_clean = combined.dropna()
        
        if len(combined_clean) == 0:
            print(f"Nenhum dado válido após remoção de NaN para {pair}")
            return None, None
        
        # Separar features e target
        features_clean = combined_clean[self.feature_names]
        target_clean = combined_clean['target']
        
        # Aplicar limitação de amostras se configurado
        if self.fast_training_mode and self.max_training_samples:
            if len(features_clean) > self.max_training_samples:
                print(f"Modo treinamento rápido: usando {self.max_training_samples} de {len(features_clean)} amostras")
                features_clean = features_clean.iloc[:self.max_training_samples]
                target_clean = target_clean.iloc[:self.max_training_samples]
        
        print(f"Dados preparados: {len(features_clean)} amostras, {len(self.feature_names)} features")
        
        return features_clean, target_clean
    
    def train_models(self, X, y, optimize_hyperparams=False):
        """
        Treina múltiplos modelos e seleciona o melhor
        """
        print("\nTreinando modelos...")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        best_score = 0
        
        for name, model in self.models.items():
            print(f"Treinando {name}...")
            
            try:
                # Treinar modelo
                model.fit(X_train_scaled, y_train)
                
                # Avaliar
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                
                results[name] = {
                    'model': model,
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'accuracy': test_score
                }
                
                # Selecionar melhor modelo
                if test_score > best_score:
                    best_score = test_score
                    self.best_model = model
                
                print(f"  Acurácia: {test_score:.3f} (CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
                
            except Exception as e:
                print(f"  Erro no treinamento: {e}")
                continue
        
        return results
    
    def predict_direction(self, pair, period='7d'):
        """
        Prediz direção do preço para um par de moedas
        Compatível com a interface esperada pelo AutoTrader
        
        Args:
            pair: Par de moedas (ex: 'EURUSD')
            period: Período dos dados para predição
            
        Returns:
            tuple: (direction, confidence, prob_up, prob_down) ou None se erro
        """
        try:
            # Preparar dados para predição
            X, y = self.prepare_data(pair)
            
            if X is None or y is None or len(X) == 0:
                print(f"Não foi possível preparar dados para predição de {pair}")
                return None
            
            # Treinar modelo se necessário
            if self.best_model is None:
                print(f"Treinando modelo para {pair}...")
                # Usar amostra menor para predição rápida
                sample_size = min(5000, len(X))
                X_sample = X.iloc[:sample_size]
                y_sample = y.iloc[:sample_size]
                self.train_models(X_sample, y_sample)
                
                if self.best_model is None:
                    print(f"Falha no treinamento do modelo para {pair}")
                    return None
            
            # Usar dados mais recentes para predição
            X_recent = X.iloc[-1:]
            
            # Escalar amostra
            X_scaled = self.scaler.transform(X_recent)
            
            # Predição
            prediction = self.best_model.predict(X_scaled)[0]
            
            # Probabilidades
            if hasattr(self.best_model, 'predict_proba'):
                proba = self.best_model.predict_proba(X_scaled)[0]
                prob_down = proba[0]  # Probabilidade de classe 0 (baixa)
                prob_up = proba[1]    # Probabilidade de classe 1 (alta)
                confidence = max(proba)
            else:
                # Para modelos sem predict_proba (ex: SVM)
                prob_up = 0.7 if prediction == 1 else 0.3
                prob_down = 1.0 - prob_up
                confidence = 0.6
            
            direction = "BUY" if prediction == 1 else "SELL"
            
            print(f"Predição para {pair}: {direction} (confiança: {confidence:.3f})")
            
            return direction, confidence, prob_up, prob_down
            
        except Exception as e:
            print(f"Erro na predição para {pair}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_direction_raw(self, X_sample):
        """
        Prediz direção do preço para uma amostra de features (método original)
        """
        if self.best_model is None:
            return None, 0
        
        # Escalar amostra
        X_scaled = self.scaler.transform(X_sample)
        
        # Predição
        prediction = self.best_model.predict(X_scaled)[0]
        
        # Probabilidade (confiança)
        if hasattr(self.best_model, 'predict_proba'):
            proba = self.best_model.predict_proba(X_scaled)[0]
            confidence = max(proba) * 100
        else:
            confidence = 60  # Valor padrão para SVM sem probabilidade
        
        direction = "COMPRA" if prediction == 1 else "VENDA"
        
        return direction, confidence
    
    def backtest(self, X, y, test_size=0.3):
        """
        Realiza backtest da estratégia
        """
        print("\nRealizando backtest...")
        
        # Dividir dados para backtest
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Treinar com dados históricos
        self.train_models(X_train, y_train)
        
        if self.best_model is None:
            return None
        
        # Predições no período de teste
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.best_model.predict(X_test_scaled)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, predictions)
        
        # Simular trading
        returns = []
        for i in range(len(predictions)):
            if predictions[i] == 1:  # Predição de alta
                returns.append(1 if y_test.iloc[i] == 1 else -1)
            else:  # Predição de baixa
                returns.append(1 if y_test.iloc[i] == 0 else -1)
        
        total_trades = len(returns)
        winning_trades = sum(1 for r in returns if r > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_return': sum(returns)
        }
        
        print(f"Backtest concluído:")
        print(f"  Acurácia: {accuracy:.3f}")
        print(f"  Taxa de acerto: {win_rate:.3f}")
        print(f"  Total de trades: {total_trades}")
        
        return results
    
    def feature_importance(self):
        """
        Mostra importância das features
        """
        if self.best_model is None or not hasattr(self.best_model, 'feature_importances_'):
            print("Importância de features não disponível para este modelo")
            return
        
        importances = self.best_model.feature_importances_
        feature_importance = list(zip(self.feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\nImportância das Features:")
        print("-" * 40)
        for feature, importance in feature_importance[:10]:
            print(f"{feature:20}: {importance:.4f}")
    
    def get_available_pairs(self):
        """
        Retorna pares disponíveis
        """
        return self.data_mapper.get_available_pairs()

def main():
    """
    Demonstração do preditor ML local
    """
    print("ML Forex Predictor - Versão Local")
    print("Machine Learning para Forex com dados locais")
    print("=" * 60)
    
    predictor = MLForexPredictorLocal()
    
    # Listar pares disponíveis
    predictor.data_mapper.list_available_pairs()
    
    # Testar com pares disponíveis
    available_pairs = predictor.get_available_pairs()
    test_pairs = list(available_pairs.keys())[:2]  # Testar primeiros 2 pares
    
    for pair in test_pairs:
        print(f"\n" + "=" * 60)
        print(f"TESTANDO {pair}")
        print("=" * 60)
        
        # Preparar dados
        X, y = predictor.prepare_data(pair, period='6mo')
        
        if X is not None and y is not None and len(X) > 50:
            # Treinar modelos
            results = predictor.train_models(X, y)
            
            # Mostrar importância das features
            predictor.feature_importance()
            
            # Fazer predição atual
            result = predictor.predict_direction(pair)
            if result:
                direction, confidence, prob_up, prob_down = result
                print(f"\nPredição atual:")
                print(f"  Direção: {direction}")
                print(f"  Confiança: {confidence:.3f}")
                print(f"  Prob. Alta: {prob_up:.3f}")
                print(f"  Prob. Baixa: {prob_down:.3f}")
            else:
                print(f"\nFalha na predição para {pair}")
            
            # Realizar backtest
            backtest_results = predictor.backtest(X, y)
            
        else:
            print(f"Dados insuficientes para {pair}")

if __name__ == "__main__":
    main()