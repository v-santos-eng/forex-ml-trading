#!/usr/bin/env python3
"""
Agno ML Tool - Versão Local
Ferramenta de Machine Learning para Forex usando dados locais

Esta versão integra o MLForexPredictorLocal para usar dados históricos locais
"""

import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar o preditor local
from ml_predictor_local import MLForexPredictorLocal

class MLForexToolLocal:
    """
    Ferramenta de ML para Forex usando dados locais
    Integra o MLForexPredictorLocal para análises e predições
    """
    
    def __init__(self, data_dir: str = "extracted_fx_data"):
        """
        Inicializa a ferramenta ML local
        
        Args:
            data_dir: Diretório com dados históricos de forex
        """
        self.predictor = MLForexPredictorLocal(data_dir)
        self.last_predictions = {}
        
    def predict_forex_direction(self, pair: str, period: str = '6mo') -> dict:
        """
        Prediz a direção do par de moedas usando dados locais
        
        Args:
            pair: Par de moedas (ex: 'EURUSD')
            period: Período de dados para análise
            
        Returns:
            dict: Resultado da predição com direção, confiança e métricas
        """
        try:
            print(f"\nPredizindo direção para {pair}...")
            
            # Preparar dados
            X, y = self.predictor.prepare_data(pair, period=period)
            
            if X is None or y is None:
                return {
                    'error': f'Não foi possível obter dados para {pair}',
                    'pair': pair,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Verificar se há dados suficientes
            if len(X) < 100:
                return {
                    'error': f'Dados insuficientes para {pair}. Disponível: {len(X)} amostras (mínimo: 100)',
                    'pair': pair,
                    'samples_available': len(X),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Treinar modelos
            model_results = self.predictor.train_models(X, y)
            
            if not model_results or self.predictor.best_model is None:
                return {
                    'error': f'Falha no treinamento dos modelos para {pair}',
                    'pair': pair,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Fazer predição para a última amostra
            direction, confidence = self.predictor.predict_direction(X.iloc[-1:].values)
            
            # Obter melhor modelo
            best_model_name = None
            best_accuracy = 0
            for name, result in model_results.items():
                if result['accuracy'] > best_accuracy:
                    best_accuracy = result['accuracy']
                    best_model_name = name
            
            # Preparar resultado
            result = {
                'pair': pair,
                'direction': direction,
                'confidence': round(confidence, 2),
                'model': best_model_name,
                'model_accuracy': round(best_accuracy, 3),
                'samples_used': len(X),
                'features_count': len(self.predictor.feature_names),
                'data_period': period,
                'timestamp': datetime.now().isoformat(),
                'model_results': {
                    name: {
                        'accuracy': round(result['accuracy'], 3),
                        'cv_mean': round(result['cv_mean'], 3),
                        'cv_std': round(result['cv_std'], 3)
                    }
                    for name, result in model_results.items()
                }
            }
            
            # Armazenar última predição
            self.last_predictions[pair] = result
            
            print(f"✓ Predição concluída: {direction} ({confidence:.1f}% confiança)")
            return result
            
        except Exception as e:
            error_result = {
                'error': f'Erro na predição para {pair}: {str(e)}',
                'pair': pair,
                'timestamp': datetime.now().isoformat()
            }
            print(f"✗ Erro na predição: {e}")
            return error_result
    
    def backtest_strategy(self, pair: str, period: str = '1y', test_size: float = 0.3) -> dict:
        """
        Realiza backtest da estratégia de trading
        
        Args:
            pair: Par de moedas
            period: Período de dados
            test_size: Proporção dos dados para teste
            
        Returns:
            dict: Resultados do backtest
        """
        try:
            print(f"\nRealizando backtest para {pair}...")
            
            # Preparar dados
            X, y = self.predictor.prepare_data(pair, period=period)
            
            if X is None or y is None:
                return {
                    'error': f'Não foi possível obter dados para backtest de {pair}',
                    'pair': pair,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Verificar dados suficientes
            if len(X) < 200:
                return {
                    'error': f'Dados insuficientes para backtest de {pair}. Disponível: {len(X)} (mínimo: 200)',
                    'pair': pair,
                    'samples_available': len(X),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Realizar backtest
            backtest_results = self.predictor.backtest(X, y, test_size=test_size)
            
            if backtest_results is None:
                return {
                    'error': f'Falha no backtest para {pair}',
                    'pair': pair,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Preparar resultado
            result = {
                'pair': pair,
                'accuracy': round(backtest_results['accuracy'], 3),
                'win_rate': round(backtest_results['win_rate'], 3),
                'total_trades': backtest_results['total_trades'],
                'winning_trades': backtest_results['winning_trades'],
                'total_return': backtest_results['total_return'],
                'test_size': test_size,
                'samples_used': len(X),
                'data_period': period,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✓ Backtest concluído: {result['accuracy']:.3f} acurácia, {result['win_rate']:.3f} taxa de acerto")
            return result
            
        except Exception as e:
            error_result = {
                'error': f'Erro no backtest para {pair}: {str(e)}',
                'pair': pair,
                'timestamp': datetime.now().isoformat()
            }
            print(f"✗ Erro no backtest: {e}")
            return error_result
    
    def analyze_multiple_pairs(self, pairs: list, period: str = '6mo') -> dict:
        """
        Analisa múltiplos pares de moedas
        
        Args:
            pairs: Lista de pares para analisar
            period: Período de dados
            
        Returns:
            dict: Resultados para todos os pares
        """
        print(f"\nAnalisando {len(pairs)} pares de moedas...")
        
        results = {}
        
        for pair in pairs:
            print(f"\n{'-' * 50}")
            print(f"Analisando {pair}")
            print("-" * 50)
            
            # Predição
            prediction = self.predict_forex_direction(pair, period)
            
            # Backtest (apenas se predição foi bem-sucedida)
            backtest = None
            if 'error' not in prediction:
                backtest = self.backtest_strategy(pair, period)
            
            results[pair] = {
                'prediction': prediction,
                'backtest': backtest,
                'timestamp': datetime.now().isoformat()
            }
        
        return results
    
    def get_feature_importance(self, pair: str, period: str = '6mo') -> dict:
        """
        Obtém importância das features para um par
        
        Args:
            pair: Par de moedas
            period: Período de dados
            
        Returns:
            dict: Importância das features
        """
        try:
            # Preparar dados e treinar modelo
            X, y = self.predictor.prepare_data(pair, period=period)
            
            if X is None or y is None or len(X) < 100:
                return {
                    'error': f'Dados insuficientes para análise de features de {pair}',
                    'pair': pair
                }
            
            # Treinar modelos
            self.predictor.train_models(X, y)
            
            if (self.predictor.best_model is None or 
                not hasattr(self.predictor.best_model, 'feature_importances_')):
                return {
                    'error': f'Modelo não suporta análise de importância de features',
                    'pair': pair
                }
            
            # Obter importâncias
            importances = self.predictor.best_model.feature_importances_
            feature_importance = list(zip(self.predictor.feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'pair': pair,
                'feature_importance': [
                    {'feature': feature, 'importance': round(importance, 4)}
                    for feature, importance in feature_importance
                ],
                'top_features': [
                    {'feature': feature, 'importance': round(importance, 4)}
                    for feature, importance in feature_importance[:10]
                ],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'error': f'Erro na análise de features para {pair}: {str(e)}',
                'pair': pair
            }
    
    def get_available_pairs(self) -> dict:
        """
        Retorna pares disponíveis nos dados locais
        
        Returns:
            dict: Pares disponíveis com informações
        """
        return self.predictor.get_available_pairs()
    
    def save_analysis_report(self, results: dict, filename: str = None) -> str:
        """
        Salva relatório de análise em arquivo JSON
        
        Args:
            results: Resultados da análise
            filename: Nome do arquivo (opcional)
            
        Returns:
            str: Caminho do arquivo salvo
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ml_forex_analysis_local_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'ml_forex_local',
            'data_source': 'local_files',
            'results': results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Relatório salvo em: {filename}")
        return filename

def main():
    """
    Demonstração da ferramenta ML local
    """
    print("ML Forex Tool - Versão Local")
    print("Ferramenta de Machine Learning para Forex com dados locais")
    print("=" * 70)
    
    # Inicializar ferramenta
    ml_tool = MLForexToolLocal()
    
    # Obter pares disponíveis
    available_pairs = ml_tool.get_available_pairs()
    print(f"\nPares disponíveis: {len(available_pairs)}")
    
    # Testar com alguns pares
    test_pairs = list(available_pairs.keys())[:3]
    
    if test_pairs:
        print(f"\nTestando com: {', '.join(test_pairs)}")
        
        # Analisar múltiplos pares
        results = ml_tool.analyze_multiple_pairs(test_pairs)
        
        # Salvar relatório
        report_file = ml_tool.save_analysis_report(results)
        
        print(f"\n✅ Análise concluída!")
        print(f"📊 Relatório: {report_file}")
    else:
        print("\n❌ Nenhum par disponível para teste")

if __name__ == "__main__":
    main()