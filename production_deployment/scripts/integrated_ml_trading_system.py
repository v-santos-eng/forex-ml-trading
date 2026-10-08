#!/usr/bin/env python3
"""
Sistema Integrado de Trading com ML
Implementação prática baseada nos conhecimentos extraídos do livro
"Machine Learning for Algorithmic Trading Predictive"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar nossos módulos avançados
from .advanced_feature_engineering import AdvancedFeatureEngineer
from .advanced_backtesting_system import AdvancedBacktester, BacktestResults
from .advanced_risk_management import AdvancedRiskManager, RiskMetrics, PositionSizing

# Importar módulos existentes
from .ml_predictor_optimized import OptimizedMLForexPredictor
try:
    from csv_data_loader import CSVDataLoader as DataLoader
except ImportError:
    # Criar um DataLoader simples se não existir
    class DataLoader:
        def __init__(self, data_path):
            self.data_path = Path(data_path)
        
        def load_pair_data(self, pair):
            # Procurar arquivo CSV para o par na estrutura extracted_fx_data
            pair_folder = self.data_path / pair.upper()
            if pair_folder.exists():
                csv_files = list(pair_folder.glob("*.csv"))
            else:
                # Fallback: procurar em toda a estrutura
                csv_files = list(self.data_path.glob(f"**/*{pair}*.csv"))
            
            if not csv_files:
                return None
            
            # Carregar primeiro arquivo encontrado
            try:
                import pandas as pd
                
                # Formato específico dos dados forex: DateTime;Open;High;Low;Close;Volume
                data = pd.read_csv(
                    csv_files[0], 
                    sep=';', 
                    header=None,
                    names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
                )
                
                # Converter DateTime
                data['DateTime'] = pd.to_datetime(data['DateTime'], format='%Y%m%d %H%M%S')
                data.set_index('DateTime', inplace=True)
                
                # Garantir tipos numéricos
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Remover linhas com valores nulos
                data = data.dropna()
                
                # Adicionar target simples (próximo retorno)
                data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
                
                # Remover última linha (sem target)
                data = data[:-1]
                
                print(f"Dados carregados: {len(data)} registros, colunas: {list(data.columns)}")
                return data
            except Exception as e:
                print(f"Erro ao carregar {csv_files[0]}: {e}")
                return None

class IntegratedMLTradingSystem:
    """
    Sistema integrado que combina:
    - Feature Engineering Avançado
    - Modelos de ML Otimizados
    - Backtesting Robusto
    - Gestão de Risco Avançada
    """
    
    def __init__(self, 
                 config: Dict = None,
                 data_path: str = "extracted_fx_data",
                 models_path: str = "models",
                 results_path: str = "results"):
        
        # Configuração padrão
        self.config = config or self._get_default_config()
        
        # Caminhos
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.results_path = Path(results_path)
        
        # Criar diretórios se não existirem
        self.models_path.mkdir(exist_ok=True)
        self.results_path.mkdir(exist_ok=True)
        
        # Componentes do sistema
        self.data_loader = DataLoader(str(self.data_path))
        self.feature_engineer = AdvancedFeatureEngineer(
            lookback_periods=self.config['feature_engineering']['lookback_periods']
        )
        self.ml_predictor = OptimizedMLForexPredictor()
        self.backtester = AdvancedBacktester(
            initial_capital=self.config['backtesting']['initial_capital'],
            commission=self.config['backtesting']['commission'],
            slippage=self.config['backtesting']['slippage']
        )
        self.risk_manager = AdvancedRiskManager(
            max_portfolio_risk=self.config['risk_management']['max_portfolio_risk'],
            max_position_risk=self.config['risk_management']['max_position_risk']
        )
        
        # Estado do sistema
        self.trained_models = {}
        self.feature_names = {}
        self.performance_metrics = {}
        
        # Configurar logging
        self._setup_logging()
        
    def _get_default_config(self) -> Dict:
        """Configuração padrão do sistema"""
        return {
            'feature_engineering': {
                'lookback_periods': [5, 10, 20, 50],
                'target_horizon': 1,  # Prever 1 período à frente
                'feature_selection_k': 50
            },
            'ml_models': {
                'models_to_train': ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm'],
                'cv_folds': 5,
                'test_size': 0.2,
                'random_state': 42
            },
            'backtesting': {
                'initial_capital': 100000,
                'commission': 0.0001,  # 1 pip
                'slippage': 0.0001,    # 1 pip
                'stop_loss_pct': 0.02,  # 2%
                'take_profit_pct': 0.04  # 4%
            },
            'risk_management': {
                'max_portfolio_risk': 0.02,  # 2% por dia
                'max_position_risk': 0.01,   # 1% por posição
                'max_correlation': 0.7,
                'lookback_period': 252
            },
            'trading': {
                'signal_threshold': 0.6,  # Confiança mínima para sinal
                'min_prediction_samples': 100,
                'rebalance_frequency': 'daily'
            }
        }
    
    def _setup_logging(self):
        """Configura sistema de logging"""
        log_file = self.results_path / f"trading_system_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('IntegratedMLTradingSystem')
        self.logger.info("Sistema de Trading ML Integrado inicializado")
    
    def load_and_prepare_data(self, currency_pairs: List[str]) -> Dict[str, pd.DataFrame]:
        """Carrega e prepara dados para múltiplos pares de moedas"""
        self.logger.info(f"Carregando dados para {len(currency_pairs)} pares de moedas")
        
        prepared_data = {}
        
        for pair in currency_pairs:
            try:
                # Carregar dados brutos
                raw_data = self.data_loader.load_pair_data(pair)
                
                if raw_data is None or len(raw_data) < 1000:
                    self.logger.warning(f"Dados insuficientes para {pair}")
                    continue
                
                self.logger.info(f"Carregados {len(raw_data)} registros para {pair}")
                
                # Aplicar feature engineering
                features, target = self.feature_engineer.engineer_features(
                    raw_data, 
                    target_col='target'
                )
                
                # Combinar features e target
                prepared_data[pair] = {
                    'raw_data': raw_data,
                    'features': features,
                    'target': target,
                    'feature_names': self.feature_engineer.feature_names
                }
                
                self.logger.info(f"Feature engineering concluído para {pair}: {features.shape[1]} features")
                
            except Exception as e:
                self.logger.error(f"Erro ao processar {pair}: {str(e)}")
                continue
        
        return prepared_data
    
    def train_models(self, prepared_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """Treina modelos ML para cada par de moedas"""
        self.logger.info("Iniciando treinamento de modelos ML")
        
        trained_models = {}
        
        for pair, data in prepared_data.items():
            self.logger.info(f"Treinando modelos para {pair}")
            
            try:
                features = data['features']
                target = data['target']
                
                # Alinhar features e target
                common_idx = features.index.intersection(target.index)
                X = features.loc[common_idx]
                y = target.loc[common_idx]
                
                # Remover NaN
                mask = ~(X.isna().any(axis=1) | y.isna())
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(X_clean) < 100:
                    self.logger.warning(f"Dados insuficientes após limpeza para {pair}")
                    continue
                
                # Treinar modelos usando adaptador
                results = self._train_models_adapter(X_clean, y_clean)
                
                if results:
                    trained_models[pair] = {
                        'models': results['models'],
                        'best_model': results['best_model'],
                        'performance': results['performance'],
                        'feature_names': data['feature_names']
                    }
                    
                    # Salvar modelos
                    model_file = self.models_path / f"{pair}_models.joblib"
                    joblib.dump(results, model_file)
                    
                    self.logger.info(f"Modelos treinados e salvos para {pair}")
                    self.logger.info(f"Melhor modelo: {results['best_model']} (Accuracy: {results['performance'][results['best_model']]['accuracy']:.4f})")
                
            except Exception as e:
                self.logger.error(f"Erro no treinamento para {pair}: {str(e)}")
                continue
        
        self.trained_models = trained_models
        return trained_models
    
    def _train_models_adapter(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Adaptador para treinar modelos usando OptimizedMLForexPredictor"""
        try:
            # Preparar dados no formato esperado
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.metrics import accuracy_score, classification_report
            
            # Divisão dos dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['ml_models']['test_size'], 
                random_state=self.config['ml_models']['random_state'],
                stratify=y if len(y.unique()) > 1 else None
            )
            
            # Modelos disponíveis
            models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100, random_state=42
                )
            }
            
            # Adicionar XGBoost se disponível
            try:
                import xgboost as xgb
                models['xgboost'] = xgb.XGBClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
            except ImportError:
                pass
            
            # Adicionar LightGBM se disponível
            try:
                import lightgbm as lgb
                models['lightgbm'] = lgb.LGBMClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
            except ImportError:
                pass
            
            # Treinar e avaliar modelos
            trained_models = {}
            performance = {}
            best_model = None
            best_score = 0
            
            for name, model in models.items():
                if name in self.config['ml_models']['models_to_train']:
                    try:
                        # Treinar modelo
                        model.fit(X_train, y_train)
                        
                        # Fazer predições
                        y_pred = model.predict(X_test)
                        
                        # Calcular métricas
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Armazenar resultados
                        trained_models[name] = model
                        performance[name] = {
                            'accuracy': accuracy,
                            'model': model
                        }
                        
                        # Verificar se é o melhor modelo
                        if accuracy > best_score:
                            best_score = accuracy
                            best_model = name
                        
                        self.logger.info(f"Modelo {name} treinado - Accuracy: {accuracy:.4f}")
                        
                    except Exception as e:
                        self.logger.error(f"Erro ao treinar {name}: {str(e)}")
                        continue
            
            if trained_models:
                return {
                    'models': trained_models,
                    'performance': performance,
                    'best_model': best_model
                }
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Erro no adaptador de treinamento: {str(e)}")
            return None
    
    def generate_signals(self, 
                        prepared_data: Dict[str, Dict], 
                        trained_models: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """Gera sinais de trading usando modelos treinados"""
        self.logger.info("Gerando sinais de trading")
        
        signals = {}
        
        for pair in prepared_data.keys():
            if pair not in trained_models:
                self.logger.warning(f"Modelo não disponível para {pair}")
                continue
            
            try:
                # Dados e modelo
                features = prepared_data[pair]['features']
                model_info = trained_models[pair]
                best_model_name = model_info['best_model']
                best_model = model_info['models'][best_model_name]
                
                # Fazer predições
                predictions = best_model.predict(features)
                prediction_proba = None
                
                # Tentar obter probabilidades se disponível
                if hasattr(best_model, 'predict_proba'):
                    try:
                        proba = best_model.predict_proba(features)
                        if proba.shape[1] == 2:  # Classificação binária
                            prediction_proba = proba[:, 1]  # Probabilidade da classe positiva
                    except:
                        pass
                
                # Converter predições em sinais
                signals_df = pd.DataFrame(index=features.index)
                
                # Sinal baseado na predição
                if prediction_proba is not None:
                    # Usar probabilidades para sinais mais refinados
                    threshold = self.config['trading']['signal_threshold']
                    signals_df['signal'] = 0
                    signals_df.loc[prediction_proba > threshold, 'signal'] = 1  # Long
                    signals_df.loc[prediction_proba < (1 - threshold), 'signal'] = -1  # Short
                    signals_df['confidence'] = np.abs(prediction_proba - 0.5) * 2
                else:
                    # Usar predições diretas
                    signals_df['signal'] = np.where(predictions > 0, 1, -1)
                    signals_df['confidence'] = np.abs(predictions)
                
                signals_df['prediction'] = predictions
                if prediction_proba is not None:
                    signals_df['probability'] = prediction_proba
                
                signals[pair] = signals_df
                
                self.logger.info(f"Sinais gerados para {pair}: {len(signals_df)} períodos")
                
            except Exception as e:
                self.logger.error(f"Erro na geração de sinais para {pair}: {str(e)}")
                continue
        
        return signals
    
    def run_backtesting(self, 
                       prepared_data: Dict[str, Dict], 
                       signals: Dict[str, pd.DataFrame]) -> Dict[str, BacktestResults]:
        """Executa backtesting para cada par de moedas"""
        self.logger.info("Iniciando backtesting")
        
        backtest_results = {}
        
        for pair in signals.keys():
            if pair not in prepared_data:
                continue
            
            try:
                # Dados OHLC
                ohlc_data = prepared_data[pair]['raw_data']
                pair_signals = signals[pair]
                
                # Executar backtesting
                results = self.backtester.run_backtest(
                    ohlc_data,
                    pair_signals,
                    stop_loss_pct=self.config['backtesting']['stop_loss_pct'],
                    take_profit_pct=self.config['backtesting']['take_profit_pct']
                )
                
                if results:
                    backtest_results[pair] = results
                    
                    # Log resultados
                    metrics = results.metrics
                    self.logger.info(f"Backtesting {pair} concluído:")
                    self.logger.info(f"  Retorno Total: {metrics['total_return']:.2%}")
                    self.logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    self.logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
                    self.logger.info(f"  Total Trades: {metrics['total_trades']}")
                    self.logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
                    
                    # Salvar resultados
                    results_file = self.results_path / f"{pair}_backtest_results.joblib"
                    joblib.dump(results, results_file)
                
            except Exception as e:
                self.logger.error(f"Erro no backtesting para {pair}: {str(e)}")
                continue
        
        return backtest_results
    
    def assess_portfolio_risk(self, 
                            backtest_results: Dict[str, BacktestResults]) -> Dict[str, any]:
        """Avalia risco do portfolio completo"""
        self.logger.info("Avaliando risco do portfolio")
        
        # Extrair retornos de cada par
        returns_dict = {}
        for pair, results in backtest_results.items():
            returns_dict[pair] = results.returns
        
        # Simular posições (igual peso para demonstração)
        total_pairs = len(backtest_results)
        positions = {}
        
        for pair in backtest_results.keys():
            positions[pair] = {
                'value': 100000 / total_pairs,  # Dividir capital igualmente
                'size': 100000 / total_pairs
            }
        
        # Gerar relatório de risco
        risk_report = self.risk_manager.generate_risk_report(returns_dict, positions)
        
        # Salvar relatório
        report_file = self.results_path / f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(risk_report)
        
        self.logger.info(f"Relatório de risco salvo em: {report_file}")
        
        return {
            'risk_report': risk_report,
            'returns_dict': returns_dict,
            'positions': positions
        }
    
    def run_complete_analysis(self, currency_pairs: List[str]) -> Dict[str, any]:
        """Executa análise completa do sistema"""
        self.logger.info("=== INICIANDO ANÁLISE COMPLETA DO SISTEMA ===")
        
        results = {}
        
        try:
            # 1. Carregar e preparar dados
            self.logger.info("Etapa 1: Carregamento e preparação de dados")
            prepared_data = self.load_and_prepare_data(currency_pairs)
            results['prepared_data'] = prepared_data
            
            if not prepared_data:
                self.logger.error("Nenhum dado válido carregado")
                return results
            
            # 2. Treinar modelos
            self.logger.info("Etapa 2: Treinamento de modelos ML")
            trained_models = self.train_models(prepared_data)
            results['trained_models'] = trained_models
            
            if not trained_models:
                self.logger.error("Nenhum modelo treinado com sucesso")
                return results
            
            # 3. Gerar sinais
            self.logger.info("Etapa 3: Geração de sinais de trading")
            signals = self.generate_signals(prepared_data, trained_models)
            results['signals'] = signals
            
            # 4. Backtesting
            self.logger.info("Etapa 4: Backtesting")
            backtest_results = self.run_backtesting(prepared_data, signals)
            results['backtest_results'] = backtest_results
            
            # 5. Análise de risco
            self.logger.info("Etapa 5: Análise de risco do portfolio")
            risk_analysis = self.assess_portfolio_risk(backtest_results)
            results['risk_analysis'] = risk_analysis
            
            # 6. Resumo final
            self.logger.info("Etapa 6: Geração de resumo final")
            summary = self._generate_final_summary(results)
            results['summary'] = summary
            
            self.logger.info("=== ANÁLISE COMPLETA CONCLUÍDA ===")
            
        except Exception as e:
            self.logger.error(f"Erro na análise completa: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _generate_final_summary(self, results: Dict[str, any]) -> str:
        """Gera resumo final da análise"""
        summary = "\n" + "=" * 80 + "\n"
        summary += "RESUMO FINAL - SISTEMA INTEGRADO DE TRADING ML\n"
        summary += "=" * 80 + "\n"
        
        # Dados processados
        if 'prepared_data' in results:
            summary += f"\n1. DADOS PROCESSADOS:\n"
            summary += f"   Pares de moedas: {len(results['prepared_data'])}\n"
            for pair, data in results['prepared_data'].items():
                summary += f"   {pair}: {data['features'].shape[0]} observações, {data['features'].shape[1]} features\n"
        
        # Modelos treinados
        if 'trained_models' in results:
            summary += f"\n2. MODELOS TREINADOS:\n"
            for pair, model_info in results['trained_models'].items():
                best_model = model_info['best_model']
                accuracy = model_info['performance'][best_model]['accuracy']
                summary += f"   {pair}: {best_model} (Accuracy: {accuracy:.4f})\n"
        
        # Resultados de backtesting
        if 'backtest_results' in results:
            summary += f"\n3. PERFORMANCE DE BACKTESTING:\n"
            total_return = 0
            total_trades = 0
            
            for pair, bt_results in results['backtest_results'].items():
                metrics = bt_results.metrics
                summary += f"   {pair}:\n"
                summary += f"     Retorno: {metrics['total_return']:.2%}\n"
                summary += f"     Sharpe: {metrics['sharpe_ratio']:.2f}\n"
                summary += f"     Drawdown: {metrics['max_drawdown']:.2%}\n"
                summary += f"     Trades: {metrics['total_trades']}\n"
                summary += f"     Win Rate: {metrics['win_rate']:.2%}\n"
                
                total_return += metrics['total_return']
                total_trades += metrics['total_trades']
            
            avg_return = total_return / len(results['backtest_results']) if results['backtest_results'] else 0
            summary += f"\n   MÉDIAS DO PORTFOLIO:\n"
            summary += f"     Retorno Médio: {avg_return:.2%}\n"
            summary += f"     Total de Trades: {total_trades}\n"
        
        # Análise de risco
        if 'risk_analysis' in results:
            summary += f"\n4. ANÁLISE DE RISCO:\n"
            summary += f"   Relatório detalhado salvo em arquivo\n"
        
        summary += "\n" + "=" * 80 + "\n"
        
        # Salvar resumo
        summary_file = self.results_path / f"final_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        return summary

def main():
    """Função principal para demonstração"""
    print("=== SISTEMA INTEGRADO DE TRADING ML ===")
    print("Baseado nos conhecimentos do livro 'Machine Learning for Algorithmic Trading'")
    
    # Pares de moedas para análise
    currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    
    # Criar sistema integrado
    trading_system = IntegratedMLTradingSystem()
    
    # Executar análise completa
    results = trading_system.run_complete_analysis(currency_pairs)
    
    # Imprimir resumo
    if 'summary' in results:
        print(results['summary'])
    
    if 'error' in results:
        print(f"Erro durante execução: {results['error']}")
    
    return trading_system, results

if __name__ == "__main__":
    main()