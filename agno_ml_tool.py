from typing import Optional, Dict, Any, List
from agno.tools import Toolkit
from ml_predictor_free import MLForexPredictorFree
import pandas as pd
import numpy as np

class MLForexTool(Toolkit):
    """
    Ferramenta Agno para integrar o ML Forex Predictor
    """
    
    def __init__(self):
        super().__init__(name="ml_forex_tool")
        self.predictor = MLForexPredictorFree()
        
    def predict_forex_direction(self, pair: str = "EURUSD=X", period: str = "1y") -> Dict[str, Any]:
        """
        Prediz a direção de um par de moedas usando Machine Learning
        
        Args:
            pair: Par de moedas (ex: EURUSD=X, GBPUSD=X)
            period: Período de dados (ex: 1y, 6mo, 3mo)
            
        Returns:
            Dict com predição, confiança e detalhes dos modelos
        """
        try:
            # Preparar dados
            X, y = self.predictor.prepare_data(pair, period)
            
            if len(X) < 50:
                return {
                    "error": f"Dados insuficientes para {pair}. Apenas {len(X)} amostras disponíveis.",
                    "pair": pair,
                    "samples": len(X)
                }
            
            # Treinar modelos
            results = self.predictor.train_models(X, y, optimize_hyperparams=False)
            
            if not results:
                return {
                    "error": "Falha no treinamento dos modelos",
                    "pair": pair
                }
            
            # Fazer predição
            direction, confidence = self.predictor.predict_direction(X.iloc[-1:].values)
            
            # Preparar resultado
            result = {
                "pair": pair,
                "prediction": direction,
                "confidence": f"{confidence:.1f}%",
                "best_model": type(self.predictor.best_model).__name__,
                "models_trained": len(results),
                "data_samples": len(X),
                "features_count": len(X.columns),
                "model_scores": {}
            }
            
            # Adicionar scores dos modelos
            for model_name, model_result in results.items():
                result["model_scores"][model_name] = {
                    "accuracy": f"{model_result['accuracy']:.3f}",
                    "cv_score": f"{model_result['cv_mean']:.3f} (+/- {model_result['cv_std']:.3f})"
                }
            
            return result
            
        except Exception as e:
            return {
                "error": f"Erro na predição: {str(e)}",
                "pair": pair
            }
    
    def backtest_strategy(self, pair: str = "EURUSD=X", period: str = "1y") -> Dict[str, Any]:
        """
        Realiza backtest da estratégia de ML
        
        Args:
            pair: Par de moedas
            period: Período de dados
            
        Returns:
            Dict com métricas de performance do backtest
        """
        try:
            # Preparar dados
            X, y = self.predictor.prepare_data(pair, period)
            
            if len(X) < 100:
                return {
                    "error": f"Dados insuficientes para backtest de {pair}. Mínimo 100 amostras.",
                    "pair": pair,
                    "samples": len(X)
                }
            
            # Realizar backtest
            backtest_results = self.predictor.backtest(X, y)
            
            return {
                "pair": pair,
                "accuracy": f"{backtest_results['accuracy']:.3f}",
                "total_return": f"{backtest_results['total_return']:.3f}",
                "sharpe_ratio": f"{backtest_results['sharpe_ratio']:.3f}",
                "best_model": backtest_results['best_model'],
                "test_samples": backtest_results['test_samples']
            }
            
        except Exception as e:
            return {
                "error": f"Erro no backtest: {str(e)}",
                "pair": pair
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre os modelos disponíveis
        
        Returns:
            Dict com informações dos modelos
        """
        return {
            "available_models": [
                "RandomForest",
                "GradientBoosting", 
                "LogisticRegression",
                "SVM",
                "RidgeClassifier",
                "KNeighborsClassifier",
                "XGBoost (se disponível)",
                "Ensemble (VotingClassifier)"
            ],
            "features_count": 26,
            "feature_types": [
                "Indicadores Técnicos (RSI, MACD, Bollinger Bands)",
                "Médias Móveis (SMA, EMA)",
                "Volatilidade (ATR)",
                "Momentum",
                "Features de Preço (Open, High, Low, Close, Volume)"
            ],
            "optimization": "GridSearchCV para hiperparâmetros",
            "validation": "5-fold Cross Validation",
            "ensemble_method": "Voting Classifier (soft/hard voting)"
        }
    
    def compare_pairs(self, pairs: List[str] = None, period: str = "6mo") -> Dict[str, Any]:
        """
        Compara predições para múltiplos pares de moedas
        
        Args:
            pairs: Lista de pares de moedas
            period: Período de dados
            
        Returns:
            Dict com comparação entre pares
        """
        if pairs is None:
            pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
        
        results = {}
        
        for pair in pairs:
            try:
                prediction = self.predict_forex_direction(pair, period)
                if "error" not in prediction:
                    results[pair] = {
                        "prediction": prediction["prediction"],
                        "confidence": prediction["confidence"],
                        "best_model": prediction["best_model"],
                        "data_samples": prediction["data_samples"]
                    }
                else:
                    results[pair] = {"error": prediction["error"]}
            except Exception as e:
                results[pair] = {"error": str(e)}
        
        return {
            "comparison_results": results,
            "period": period,
            "pairs_analyzed": len(pairs),
            "successful_predictions": len([r for r in results.values() if "error" not in r])
        }