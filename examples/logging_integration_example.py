#!/usr/bin/env python3
"""
Exemplo de Integração do Sistema de Logging Avançado
Demonstra como usar logging, debugging e monitoramento de performance juntos.
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Importar nossos sistemas
from advanced_logging_system import setup_logging, monitor_performance, debug_breakpoint
from interactive_debugger import InteractiveDebugger
from performance_monitor import PerformanceMonitor, monitor_performance

class ForexMLExample:
    """Exemplo de aplicação Forex ML com logging completo"""
    
    def __init__(self):
        # Inicializar sistemas
        self.logging_system = setup_logging()
        self.logger = self.logging_system.get_logger('forex_ml.example')
        self.performance_monitor = PerformanceMonitor()
        self.debugger = InteractiveDebugger()
        
        # Configurar monitoramento
        self.performance_monitor.start_monitoring()
        
        # Adicionar callback de alerta
        self.performance_monitor.add_alert_callback(self._handle_performance_alert)
        
        # Dados simulados
        self.currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDJPY']
        self.models = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']
        
        self.logger.info("Sistema Forex ML inicializado com logging avançado")
    
    def _handle_performance_alert(self, alert_type: str, message: str, metrics: Any):
        """Callback para alertas de performance"""
        self.logger.critical(f"ALERTA DE PERFORMANCE [{alert_type}]: {message}")
        
        # Em caso de alerta crítico, iniciar debugging
        if alert_type in ['memory_high', 'model_memory_leak']:
            self.logger.warning("Iniciando sessão de debugging devido a alerta de memória")
            # debug_here()  # Descomentado em ambiente de produção
    
    def load_forex_data(self, pair: str, days: int = 30) -> pd.DataFrame:
        """Carrega dados forex simulados"""
        self.logger.info(f"Carregando dados para {pair} ({days} dias)")
        
        try:
            # Simular carregamento de dados
            dates = pd.date_range(end=datetime.now(), periods=days*24*60, freq='1min')
            
            # Gerar dados OHLC simulados
            base_price = random.uniform(1.0, 2.0)
            prices = []
            current_price = base_price
            
            for _ in range(len(dates)):
                # Simulação de movimento de preço
                change = random.gauss(0, 0.001)
                current_price *= (1 + change)
                    
                # OHLC para o minuto
                high = current_price * (1 + abs(random.gauss(0, 0.0005)))
                low = current_price * (1 - abs(random.gauss(0, 0.0005)))
                close = current_price
                
                prices.append({
                    'timestamp': dates[len(prices)],
                    'open': current_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': random.randint(100, 10000)
                })
            
            data = pd.DataFrame(prices)
            
            # Adicionar indicadores técnicos
            data = self._add_technical_indicators(data)
            
            self.logger.info(f"Dados carregados: {len(data)} registros para {pair}")
            
            # Log estruturado
            self.logger.info("Dados carregados com sucesso", extra={
                'pair': pair,
                'records': len(data),
                'date_range': f"{data['timestamp'].min()} - {data['timestamp'].max()}",
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2
            })
            
            return data
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados para {pair}: {e}", exc_info=True)
            raise
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores técnicos aos dados"""
        # SMA
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # EMA
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        # Targets (simulados)
        data['target_1'] = (data['close'].shift(-1) > data['close']).astype(int)
        data['target_5'] = (data['close'].shift(-5) > data['close']).astype(int)
        
        return data
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepara features para ML"""
        self.logger.info("Iniciando preparação de features")
        
        try:
            # Remover NaN
            initial_rows = len(data)
            data = data.dropna()
            removed_rows = initial_rows - len(data)
            
            if removed_rows > 0:
                self.logger.warning(f"Removidas {removed_rows} linhas com NaN")
            
            # Selecionar features
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'macd', 'macd_signal', 'rsi',
                'bb_middle', 'bb_upper', 'bb_lower'
            ]
            
            # Verificar se todas as colunas existem
            missing_cols = [col for col in feature_columns if col not in data.columns]
            if missing_cols:
                self.logger.error(f"Colunas ausentes: {missing_cols}")
                raise ValueError(f"Colunas ausentes: {missing_cols}")
            
            features = data[feature_columns].copy()
            
            # Normalização simples
            for col in features.columns:
                if features[col].std() > 0:
                    features[col] = (features[col] - features[col].mean()) / features[col].std()
            
            self.logger.info(f"Features preparadas: {features.shape}")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Erro na preparação de features: {e}", exc_info=True)
            raise
    
    def train_model(self, features: pd.DataFrame, targets: pd.Series, model_type: str) -> Dict[str, Any]:
        """Treina modelo ML"""
        self.logger.info(f"Iniciando treinamento do modelo {model_type}")
        
        try:
            # Simular treinamento
            training_time = random.uniform(1, 5)
            time.sleep(training_time)
            
            # Simular métricas
            accuracy = random.uniform(0.45, 0.85)
            precision = random.uniform(0.4, 0.8)
            recall = random.uniform(0.4, 0.8)
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            # Simular parâmetros do modelo
            model_params = {
                'n_estimators': random.randint(50, 200),
                'max_depth': random.randint(3, 10),
                'learning_rate': random.uniform(0.01, 0.3)
            }
            
            model_info = {
                'model_type': model_type,
                'training_time': training_time,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'parameters': model_params,
                'feature_count': len(features.columns),
                'training_samples': len(features)
            }
            
            # Log estruturado do resultado
            self.logger.info("Modelo treinado com sucesso", extra={
                'model_type': model_type,
                'accuracy': accuracy,
                'f1_score': f1_score,
                'training_time': training_time,
                'feature_count': len(features.columns)
            })
            
            # Alerta se performance baixa
            if accuracy < 0.6:
                self.logger.warning(f"Accuracy baixa para {model_type}: {accuracy:.3f}")
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Erro no treinamento do modelo {model_type}: {e}", exc_info=True)
            raise
    
    def evaluate_models(self, pair: str) -> Dict[str, Any]:
        """Avalia múltiplos modelos para um par de moedas"""
        self.logger.info(f"Iniciando avaliação de modelos para {pair}")
        
        results = {
            'pair': pair,
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'best_model': None,
            'best_accuracy': 0.0
        }
        
        try:
            # Carregar dados
            data = self.load_forex_data(pair)
            features = self.prepare_features(data)
            
            # Preparar targets
            targets = data['target_1'].dropna()
            
            # Alinhar features e targets
            min_len = min(len(features), len(targets))
            features = features.iloc[:min_len]
            targets = targets.iloc[:min_len]
            
            # Treinar cada modelo
            for model_type in self.models:
                self.logger.info(f"Treinando modelo {model_type} para {pair}")
                
                try:
                    model_result = self.train_model(features, targets, model_type)
                    results['models'][model_type] = model_result
                    
                    # Verificar se é o melhor modelo
                    if model_result['accuracy'] > results['best_accuracy']:
                        results['best_accuracy'] = model_result['accuracy']
                        results['best_model'] = model_type
                    
                except Exception as e:
                    self.logger.error(f"Falha no modelo {model_type} para {pair}: {e}")
                    results['models'][model_type] = {'error': str(e)}
            
            # Log do resultado final
            self.logger.info(f"Avaliação concluída para {pair}", extra={
                'pair': pair,
                'models_trained': len([m for m in results['models'].values() if 'error' not in m]),
                'best_model': results['best_model'],
                'best_accuracy': results['best_accuracy']
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erro na avaliação de modelos para {pair}: {e}", exc_info=True)
            raise
    
    def run_full_evaluation(self):
        """Executa avaliação completa de todos os pares"""
        self.logger.info("Iniciando avaliação completa do sistema")
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'pairs': {},
            'summary': {
                'total_pairs': len(self.currency_pairs),
                'successful_pairs': 0,
                'failed_pairs': 0,
                'best_overall_model': None,
                'best_overall_accuracy': 0.0
            }
        }
        
        try:
            for pair in self.currency_pairs:
                self.logger.info(f"Processando par {pair}")
                
                try:
                    pair_results = self.evaluate_models(pair)
                    all_results['pairs'][pair] = pair_results
                    all_results['summary']['successful_pairs'] += 1
                    
                    # Verificar melhor modelo geral
                    if pair_results['best_accuracy'] > all_results['summary']['best_overall_accuracy']:
                        all_results['summary']['best_overall_accuracy'] = pair_results['best_accuracy']
                        all_results['summary']['best_overall_model'] = f"{pair}_{pair_results['best_model']}"
                    
                except Exception as e:
                    self.logger.error(f"Falha no processamento do par {pair}: {e}")
                    all_results['pairs'][pair] = {'error': str(e)}
                    all_results['summary']['failed_pairs'] += 1
            
            # Obter resumo de performance com tratamento robusto de erro
            try:
                if hasattr(self.performance_monitor, 'get_performance_summary'):
                    performance_summary = self.performance_monitor.get_performance_summary(hours=1)
                    all_results['performance'] = performance_summary
                else:
                    self.logger.warning("Método get_performance_summary não encontrado no monitor de performance")
                    all_results['performance'] = {'error': 'Método não disponível'}
            except AttributeError as e:
                self.logger.warning(f"Erro de atributo ao obter resumo de performance: {e}")
                all_results['performance'] = {
                    'error': 'Performance summary unavailable',
                    'message': str(e)
                }
            except Exception as e:
                self.logger.error(f"Erro inesperado ao obter performance: {e}")
                all_results['performance'] = {
                    'error': 'Unexpected error in performance monitoring',
                    'message': str(e)
                }
            
            # Log final
            self.logger.info("Avaliação completa finalizada", extra={
                'successful_pairs': all_results['summary']['successful_pairs'],
                'failed_pairs': all_results['summary']['failed_pairs'],
                'best_model': all_results['summary']['best_overall_model'],
                'best_accuracy': all_results['summary']['best_overall_accuracy']
            })
            
            return all_results
            
        except Exception as e:
            self.logger.critical(f"Erro crítico na avaliação completa: {e}", exc_info=True)
            raise
        
        finally:
            # Cleanup
            self.performance_monitor.stop_monitoring()
    
    def demonstrate_debugging(self):
        """Demonstra funcionalidades de debugging"""
        self.logger.info("Demonstrando funcionalidades de debugging")
        
        # Adicionar algumas variáveis para observação
        test_data = [1, 2, 3, 4, 5]
        test_dict = {'a': 1, 'b': 2, 'c': 3}
        test_string = "Exemplo de debugging"
        
        # Simular condição de debugging
        if len(test_data) > 3:
            self.logger.debug("Condição de debugging atingida")
            # debug_here()  # Descomentado para ativar debugging interativo
        
        return test_data, test_dict, test_string

def main():
    """Função principal de exemplo"""
    print("=== Sistema de Logging Avançado Forex ML ===")
    print("Iniciando demonstração completa...\n")
    
    try:
        # Criar instância da aplicação
        app = ForexMLExample()
        
        # Demonstrar debugging (opcional)
        print("1. Demonstrando debugging...")
        app.demonstrate_debugging()
        
        # Executar avaliação completa
        print("\n2. Executando avaliação completa...")
        results = app.run_full_evaluation()
        
        # Mostrar resumo
        print("\n=== RESUMO DOS RESULTADOS ===")
        print(f"Pares processados: {results['summary']['successful_pairs']}/{results['summary']['total_pairs']}")
        print(f"Melhor modelo: {results['summary']['best_overall_model']}")
        print(f"Melhor accuracy: {results['summary']['best_overall_accuracy']:.3f}")
        
        # Exportar métricas
        print("\n3. Exportando métricas de performance...")
        app.performance_monitor.export_metrics('performance_metrics_export.json', hours=1)
        
        print("\nDemonstração concluída com sucesso!")
        
    except Exception as e:
        print(f"Erro na demonstração: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()