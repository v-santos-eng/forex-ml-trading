import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import warnings


class MetaLabeling:
    """
    Implementação do Meta-Labeling para melhorar performance de modelos de trading.
    
    Baseado no conceito de López de Prado, o Meta-Labeling usa um modelo secundário
    para determinar o tamanho da aposta (bet sizing) em vez do lado (side) da operação.
    O modelo primário determina o lado, e o modelo secundário decide se deve apostar ou não.
    
    Isso ajuda a:
    1. Filtrar falsos positivos do modelo primário
    2. Melhorar o F1-score
    3. Reduzir drawdowns
    4. Aumentar o Sharpe ratio
    """
    
    def __init__(self, 
                 primary_model: Optional[BaseEstimator] = None,
                 meta_model: Optional[BaseEstimator] = None,
                 threshold: float = 0.5):
        """
        Inicializa o Meta-Labeling.
        
        Args:
            primary_model: Modelo primário que determina o lado da operação
            meta_model: Modelo secundário que determina o tamanho da aposta
            threshold: Threshold para decisões do modelo secundário
        """
        self.primary_model = primary_model
        self.meta_model = meta_model or RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.threshold = threshold
        self.is_fitted = False
        
    def prepare_meta_labels(self, 
                           primary_predictions: pd.Series,
                           true_labels: pd.Series,
                           returns: pd.Series) -> pd.DataFrame:
        """
        Prepara os labels para o modelo de meta-labeling.
        
        Args:
            primary_predictions: Predições do modelo primário (-1, 0, 1)
            true_labels: Labels verdadeiros (-1, 0, 1)
            returns: Retornos realizados
            
        Returns:
            DataFrame com meta-labels e informações auxiliares
        """
        # Alinhar índices
        common_idx = primary_predictions.index.intersection(
            true_labels.index
        ).intersection(returns.index)
        
        primary_pred = primary_predictions.loc[common_idx]
        true_lab = true_labels.loc[common_idx]
        ret = returns.loc[common_idx]
        
        # Criar DataFrame de trabalho
        df = pd.DataFrame({
            'primary_pred': primary_pred,
            'true_label': true_lab,
            'returns': ret
        })
        
        # Filtrar apenas predições não-neutras do modelo primário
        df = df[df['primary_pred'] != 0].copy()
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Calcular retornos direcionais baseados na predição primária
        df['directional_returns'] = df['primary_pred'] * df['returns']
        
        # Meta-label: 1 se a predição primária foi lucrativa, 0 caso contrário
        df['meta_label'] = (df['directional_returns'] > 0).astype(int)
        
        # Informações adicionais para análise
        df['correct_direction'] = (np.sign(df['primary_pred']) == np.sign(df['true_label'])).astype(int)
        df['abs_returns'] = np.abs(df['returns'])
        
        return df
    
    def create_meta_features(self, 
                           features: pd.DataFrame,
                           primary_predictions: pd.Series,
                           volatility: Optional[pd.Series] = None,
                           volume: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Cria features específicas para o modelo de meta-labeling.
        
        Args:
            features: Features originais
            primary_predictions: Predições do modelo primário
            volatility: Série de volatilidade
            volume: Série de volume
            
        Returns:
            DataFrame com features para meta-labeling
        """
        # Alinhar índices
        common_idx = features.index.intersection(primary_predictions.index)
        
        meta_features = features.loc[common_idx].copy()
        primary_pred = primary_predictions.loc[common_idx]
        
        # Features baseadas na predição primária
        meta_features['primary_pred'] = primary_pred
        meta_features['primary_confidence'] = np.abs(primary_pred)
        
        # Features de volatilidade se disponível
        if volatility is not None:
            vol_aligned = volatility.reindex(common_idx).fillna(method='ffill')
            meta_features['volatility'] = vol_aligned
            meta_features['vol_regime'] = pd.qcut(vol_aligned.rank(), 
                                                 q=3, labels=[0, 1, 2]).astype(float)
        
        # Features de volume se disponível
        if volume is not None:
            vol_aligned = volume.reindex(common_idx).fillna(method='ffill')
            meta_features['volume'] = vol_aligned
            meta_features['volume_ma'] = vol_aligned.rolling(20).mean()
            meta_features['volume_ratio'] = vol_aligned / meta_features['volume_ma']
        
        # Features de momentum da predição primária
        meta_features['pred_momentum'] = primary_pred.rolling(5).mean()
        meta_features['pred_volatility'] = primary_pred.rolling(10).std()
        
        # Features de regime de mercado
        if len(features.columns) > 0:
            # Usar primeira feature como proxy para preço
            price_proxy = features.iloc[:, 0]
            meta_features['price_trend'] = price_proxy.rolling(20).mean()
            meta_features['price_above_trend'] = (price_proxy > meta_features['price_trend']).astype(int)
        
        return meta_features.dropna()
    
    def fit(self, 
            X: pd.DataFrame,
            primary_predictions: pd.Series,
            true_labels: pd.Series,
            returns: pd.Series,
            **fit_params) -> 'MetaLabeling':
        """
        Treina o modelo de meta-labeling.
        
        Args:
            X: Features para o modelo secundário
            primary_predictions: Predições do modelo primário
            true_labels: Labels verdadeiros
            returns: Retornos realizados
            **fit_params: Parâmetros adicionais para o modelo
            
        Returns:
            Self (fitted)
        """
        # Preparar meta-labels
        meta_data = self.prepare_meta_labels(primary_predictions, true_labels, returns)
        
        if len(meta_data) == 0:
            warnings.warn("Nenhum dado válido para meta-labeling")
            return self
        
        # Alinhar features com meta-labels
        common_idx = X.index.intersection(meta_data.index)
        X_aligned = X.loc[common_idx]
        y_meta = meta_data.loc[common_idx, 'meta_label']
        
        # Treinar modelo secundário
        self.meta_model.fit(X_aligned, y_meta, **fit_params)
        
        # Armazenar informações de treinamento
        self.training_info = {
            'n_samples': len(X_aligned),
            'positive_rate': y_meta.mean(),
            'feature_names': list(X_aligned.columns)
        }
        
        self.is_fitted = True
        return self
    
    def predict(self, 
                X: pd.DataFrame,
                primary_predictions: pd.Series) -> pd.Series:
        """
        Faz predições usando meta-labeling.
        
        Args:
            X: Features para predição
            primary_predictions: Predições do modelo primário
            
        Returns:
            Predições finais ajustadas pelo meta-labeling
        """
        if not self.is_fitted:
            raise ValueError("Modelo deve ser treinado antes de fazer predições")
        
        # Alinhar índices
        common_idx = X.index.intersection(primary_predictions.index)
        X_aligned = X.loc[common_idx]
        primary_pred = primary_predictions.loc[common_idx]
        
        # Predições do modelo secundário
        meta_proba = self.meta_model.predict_proba(X_aligned)[:, 1]
        meta_pred = (meta_proba >= self.threshold).astype(int)
        
        # Combinar predições: usar predição primária apenas se meta-modelo aprova
        final_predictions = primary_pred.copy()
        final_predictions[meta_pred == 0] = 0  # Filtrar predições não aprovadas
        
        return final_predictions
    
    def predict_proba(self, 
                     X: pd.DataFrame,
                     primary_predictions: pd.Series) -> pd.DataFrame:
        """
        Retorna probabilidades das predições.
        
        Args:
            X: Features para predição
            primary_predictions: Predições do modelo primário
            
        Returns:
            DataFrame com probabilidades
        """
        if not self.is_fitted:
            raise ValueError("Modelo deve ser treinado antes de fazer predições")
        
        common_idx = X.index.intersection(primary_predictions.index)
        X_aligned = X.loc[common_idx]
        
        meta_proba = self.meta_model.predict_proba(X_aligned)[:, 1]
        
        return pd.DataFrame({
            'meta_proba': meta_proba,
            'primary_pred': primary_predictions.loc[common_idx]
        }, index=common_idx)
    
    def evaluate_performance(self, 
                           X: pd.DataFrame,
                           primary_predictions: pd.Series,
                           true_labels: pd.Series,
                           returns: pd.Series) -> dict:
        """
        Avalia a performance do meta-labeling.
        
        Args:
            X: Features de teste
            primary_predictions: Predições do modelo primário
            true_labels: Labels verdadeiros
            returns: Retornos realizados
            
        Returns:
            Dicionário com métricas de performance
        """
        # Predições com e sem meta-labeling
        meta_predictions = self.predict(X, primary_predictions)
        
        # Preparar dados para avaliação
        common_idx = (X.index.intersection(primary_predictions.index)
                     .intersection(true_labels.index)
                     .intersection(returns.index))
        
        primary_pred = primary_predictions.loc[common_idx]
        meta_pred = meta_predictions.loc[common_idx]
        true_lab = true_labels.loc[common_idx]
        ret = returns.loc[common_idx]
        
        # Métricas de classificação
        def safe_score(func, y_true, y_pred, **kwargs):
            try:
                return func(y_true, y_pred, **kwargs)
            except:
                return np.nan
        
        # Converter para classificação binária (positivo vs não-positivo)
        y_true_binary = (true_lab > 0).astype(int)
        primary_binary = (primary_pred > 0).astype(int)
        meta_binary = (meta_pred > 0).astype(int)
        
        results = {
            # Métricas do modelo primário
            'primary_f1': safe_score(f1_score, y_true_binary, primary_binary),
            'primary_precision': safe_score(precision_score, y_true_binary, primary_binary),
            'primary_recall': safe_score(recall_score, y_true_binary, primary_binary),
            
            # Métricas com meta-labeling
            'meta_f1': safe_score(f1_score, y_true_binary, meta_binary),
            'meta_precision': safe_score(precision_score, y_true_binary, meta_binary),
            'meta_recall': safe_score(recall_score, y_true_binary, meta_binary),
            
            # Métricas de retorno
            'primary_total_return': (primary_pred * ret).sum(),
            'meta_total_return': (meta_pred * ret).sum(),
            
            # Estatísticas de trading
            'primary_n_trades': (primary_pred != 0).sum(),
            'meta_n_trades': (meta_pred != 0).sum(),
            'trade_reduction': 1 - (meta_pred != 0).sum() / max((primary_pred != 0).sum(), 1)
        }
        
        # Sharpe ratios
        primary_returns = primary_pred * ret
        meta_returns = meta_pred * ret
        
        results['primary_sharpe'] = (primary_returns.mean() / max(primary_returns.std(), 1e-8) 
                                   * np.sqrt(252) if len(primary_returns) > 1 else np.nan)
        results['meta_sharpe'] = (meta_returns.mean() / max(meta_returns.std(), 1e-8) 
                                * np.sqrt(252) if len(meta_returns) > 1 else np.nan)
        
        return results
    
    def get_feature_importance(self) -> pd.Series:
        """
        Retorna a importância das features do modelo secundário.
        
        Returns:
            Série com importância das features
        """
        if not self.is_fitted:
            raise ValueError("Modelo deve ser treinado primeiro")
        
        if hasattr(self.meta_model, 'feature_importances_'):
            return pd.Series(
                self.meta_model.feature_importances_,
                index=self.training_info['feature_names']
            ).sort_values(ascending=False)
        else:
            return pd.Series()
    
    def cross_validate(self, 
                      X: pd.DataFrame,
                      primary_predictions: pd.Series,
                      true_labels: pd.Series,
                      returns: pd.Series,
                      cv: int = 5) -> dict:
        """
        Realiza validação cruzada do modelo de meta-labeling.
        
        Args:
            X: Features
            primary_predictions: Predições do modelo primário
            true_labels: Labels verdadeiros
            returns: Retornos realizados
            cv: Número de folds
            
        Returns:
            Dicionário com resultados da validação cruzada
        """
        # Preparar dados
        meta_data = self.prepare_meta_labels(primary_predictions, true_labels, returns)
        
        if len(meta_data) == 0:
            return {'error': 'Nenhum dado válido para validação cruzada'}
        
        common_idx = X.index.intersection(meta_data.index)
        X_aligned = X.loc[common_idx]
        y_meta = meta_data.loc[common_idx, 'meta_label']
        
        # Validação cruzada
        cv_scores = cross_val_score(self.meta_model, X_aligned, y_meta, 
                                   cv=cv, scoring='f1')
        
        return {
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'cv_scores': cv_scores
        }


class MetaLabelingPipeline:
    """
    Pipeline completo para implementar meta-labeling em estratégias de trading.
    """
    
    def __init__(self, primary_model: BaseEstimator, meta_model: Optional[BaseEstimator] = None):
        """
        Inicializa o pipeline de meta-labeling.
        
        Args:
            primary_model: Modelo primário para determinar direção
            meta_model: Modelo secundário para bet sizing
        """
        self.primary_model = primary_model
        self.meta_labeling = MetaLabeling(primary_model, meta_model)
        self.is_fitted = False
    
    def fit(self, 
            X: pd.DataFrame,
            y: pd.Series,
            returns: pd.Series,
            **fit_params) -> 'MetaLabelingPipeline':
        """
        Treina o pipeline completo.
        
        Args:
            X: Features
            y: Labels verdadeiros
            returns: Retornos realizados
            **fit_params: Parâmetros de treinamento
            
        Returns:
            Self (fitted)
        """
        # Treinar modelo primário
        self.primary_model.fit(X, y, **fit_params)
        
        # Obter predições do modelo primário
        primary_pred = pd.Series(self.primary_model.predict(X), index=X.index)
        
        # Treinar meta-labeling
        self.meta_labeling.fit(X, primary_pred, y, returns)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Faz predições usando o pipeline completo.
        
        Args:
            X: Features para predição
            
        Returns:
            Predições finais
        """
        if not self.is_fitted:
            raise ValueError("Pipeline deve ser treinado primeiro")
        
        # Predições do modelo primário
        primary_pred = pd.Series(self.primary_model.predict(X), index=X.index)
        
        # Aplicar meta-labeling
        final_pred = self.meta_labeling.predict(X, primary_pred)
        
        return final_pred