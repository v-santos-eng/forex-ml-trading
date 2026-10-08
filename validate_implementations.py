#!/usr/bin/env python3
"""
Script de Validação das Implementações Avançadas

Este script testa todas as funcionalidades implementadas para garantir
que estão funcionando corretamente antes da integração ao sistema principal.

Autor: Sistema de Trading ML Avançado
Data: 2024
"""

import os
import sys
import warnings
import traceback
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImplementationValidator:
    """
    Classe para validar todas as implementações avançadas
    """
    
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': [],
            'summary': {}
        }
        
        # Gerar dados sintéticos para testes
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Gera dados sintéticos para testes
        
        Args:
            n_samples: Número de amostras
            
        Returns:
            DataFrame com dados de teste
        """
        np.random.seed(42)
        
        # Gerar timestamps
        start_date = datetime.now() - timedelta(days=n_samples)
        timestamps = pd.date_range(start=start_date, periods=n_samples, freq='1H')
        
        # Gerar preços com random walk
        returns = np.random.normal(0.0001, 0.02, n_samples)
        prices = 1.1000 + np.cumsum(returns)
        
        # Gerar OHLCV
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices + np.random.normal(0, 0.0001, n_samples),
            'high': prices + np.abs(np.random.normal(0, 0.0005, n_samples)),
            'low': prices - np.abs(np.random.normal(0, 0.0005, n_samples)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_samples)
        })
        
        data.set_index('timestamp', inplace=True)
        
        return data
    
    def test_triple_barrier_labeling(self) -> bool:
        """
        Testa o Triple-Barrier Labeling
        """
        try:
            from triple_barrier_labeling import (
                get_events, apply_triple_barrier, get_sample_weights,
                analyze_barriers_performance
            )
            
            logger.info("Testando Triple-Barrier Labeling...")
            
            # Teste 1: Identificar eventos
            events = get_events(
                close=self.test_data['close'],
                timestamps=self.test_data.index,
                volatility_lookback=20
            )
            
            assert len(events) > 0, "Nenhum evento identificado"
            assert 't1' in events.columns, "Coluna t1 não encontrada"
            
            # Teste 2: Aplicar triple-barrier
            labels_df = apply_triple_barrier(
                close=self.test_data['close'],
                events=events.head(100),  # Testar com subset
                profit_taking_factor=2.0,
                stop_loss_factor=1.0,
                max_holding_period=pd.Timedelta(hours=4)
            )
            
            assert len(labels_df) > 0, "Nenhum label gerado"
            assert 'label' in labels_df.columns, "Coluna label não encontrada"
            assert 'return' in labels_df.columns, "Coluna return não encontrada"
            
            # Teste 3: Calcular pesos das amostras
            sample_weights = get_sample_weights(
                labels_df['t_touch'].dropna(),
                num_threads=2
            )
            
            assert len(sample_weights) > 0, "Nenhum peso calculado"
            
            # Teste 4: Análise de performance
            performance = analyze_barriers_performance(labels_df)
            
            assert 'barrier_counts' in performance, "Análise de performance falhou"
            
            self.results['passed'].append('Triple-Barrier Labeling')
            return True
            
        except Exception as e:
            self.results['failed'].append(f'Triple-Barrier Labeling: {str(e)}')
            logger.error(f"Erro no Triple-Barrier Labeling: {e}")
            return False
    
    def test_purged_cross_validation(self) -> bool:
        """
        Testa o Purged Cross-Validation
        """
        try:
            from purged_cross_validation import (
                PurgedKFold, TimeSeriesPurgedSplit, CrossValidationAnalyzer
            )
            
            logger.info("Testando Purged Cross-Validation...")
            
            # Preparar dados
            X = self.test_data[['open', 'high', 'low', 'volume']].iloc[1:]
            y = (self.test_data['close'].pct_change() > 0).astype(int).iloc[1:]  # Alinhar com y
            
            # Teste 1: PurgedKFold
            cv = PurgedKFold(
                n_splits=3,
                purge_length=pd.Timedelta(hours=1),
                embargo_length=pd.Timedelta(minutes=30)
            )
            
            splits = list(cv.split(X, y, self.test_data.index[1:]))
            assert len(splits) == 3, "Número incorreto de splits"
            
            # Teste 2: TimeSeriesPurgedSplit
            ts_cv = TimeSeriesPurgedSplit(
                n_splits=3,
                test_size=0.2,
                gap=2  # Número de observações, não Timedelta
            )
            
            ts_splits = list(ts_cv.split(X, y, self.test_data.index[1:]))
            assert len(ts_splits) > 0, "Nenhum split temporal gerado"
            
            # Teste 3: CrossValidationAnalyzer
            from sklearn.model_selection import KFold
            from sklearn.ensemble import RandomForestClassifier
            
            cv_analyzer = CrossValidationAnalyzer()
            
            # Criar métodos de CV para comparar
            cv_methods = {
                'standard_kfold': KFold(n_splits=3),
                'purged_kfold': PurgedKFold(n_splits=3)
            }
            
            # Modelo simples para teste
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            
            comparison = cv_analyzer.compare_cv_methods(
                X=X[:500],  # Subset para teste rápido
                y=y[:500],
                model=model,
                cv_methods=cv_methods
            )
            
            assert len(comparison) > 0, "Comparação de CV falhou"
            
            self.results['passed'].append('Purged Cross-Validation')
            return True
            
        except Exception as e:
            self.results['failed'].append(f'Purged Cross-Validation: {str(e)}')
            logger.error(f"Erro no Purged Cross-Validation: {e}")
            return False
    
    def test_kelly_position_sizing(self) -> bool:
        """
        Testa o Kelly Position Sizing
        """
        try:
            from kelly_position_sizing import KellyPositionSizing
            
            logger.info("Testando Kelly Position Sizing...")
            
            kelly_sizer = KellyPositionSizing()
            
            # Teste 1: Calcular fração Kelly básica
            kelly_fraction = kelly_sizer.calculate_kelly_fraction(
                win_prob=0.55,
                avg_win=0.02,
                avg_loss=-0.01,
                method='basic'
            )
            
            assert 0 <= kelly_fraction <= 1, f"Fração Kelly inválida: {kelly_fraction}"
            
            # Teste 2: Calcular tamanho da posição
            position_info = kelly_sizer.calculate_position_size(
                kelly_fraction=kelly_fraction,
                capital=100000,
                price=1.1000,
                max_position_pct=0.1
            )
            
            assert isinstance(position_info, dict), "Deve retornar um dicionário"
            assert 'position_size' in position_info, "Deve conter position_size"
            
            # Teste 3: Análise de performance histórica
            returns = self.test_data['close'].pct_change().dropna()
            # Criar DataFrame de trades simulados
            trades_df = pd.DataFrame({
                'pnl': np.random.normal(10, 50, 50),
                'return': returns.head(50),
                'timestamp': self.test_data.index[:50]
            })
            
            performance = kelly_sizer.analyze_trading_performance(trades_df)
            
            assert hasattr(performance, 'win_rate'), "Análise de performance falhou"
            
            self.results['passed'].append('Kelly Position Sizing')
            return True
            
        except Exception as e:
            self.results['failed'].append(f'Kelly Position Sizing: {str(e)}')
            logger.error(f"Erro no Kelly Position Sizing: {e}")
            return False
    
    def test_model_drift_detection(self) -> bool:
        """
        Testa a Detecção de Model Drift
        """
        try:
            from model_drift_detection import ModelDriftDetector
            
            logger.info("Testando Model Drift Detection...")
            
            # Preparar dados
            features = self.test_data[['open', 'high', 'low', 'volume']]
            predictions = pd.Series(np.random.rand(len(features)), index=features.index)
            targets = (self.test_data['close'].pct_change() > 0).astype(int)
            performance_metrics = {'accuracy': 0.75, 'precision': 0.73, 'f1': 0.74}
            
            drift_detector = ModelDriftDetector(
                reference_window=200,
                detection_window=50,
                drift_threshold=0.05
            )
            
            # Teste 1: Configurar baseline com todos os parâmetros obrigatórios
            drift_detector.update_baseline(
                features=features[:200],
                predictions=predictions[:200],
                targets=targets[:200],
                performance_metrics=performance_metrics
            )
            
            # Teste 2: Detectar drift
            drift_alerts = drift_detector.detect_drift(
                features=features[200:250],
                predictions=predictions[200:250],
                targets=targets[200:250],
                performance_metrics=performance_metrics
            )
            
            # Verificar se o método retorna uma lista (mesmo que vazia)
            assert isinstance(drift_alerts, list), "Método detect_drift deve retornar uma lista"
            
            # Verificar se os tipos de drift estão sendo detectados corretamente
            # (não necessariamente precisa haver drift, mas o método deve funcionar)
            drift_types = [alert.drift_type.value for alert in drift_alerts] if drift_alerts else []
            
            # Teste adicional: forçar drift com dados muito diferentes
            drift_features = features[200:250].copy()
            drift_features['open'] = drift_features['open'] * 10  # Mudança drástica
            
            forced_drift_alerts = drift_detector.detect_drift(
                features=drift_features,
                predictions=predictions[200:250],
                targets=targets[200:250],
                performance_metrics={'accuracy': 0.3, 'precision': 0.2, 'f1': 0.25}  # Performance muito pior
            )
            
            # Deve detectar pelo menos algum tipo de drift com dados tão diferentes
            assert len(forced_drift_alerts) > 0, "Detecção de drift falhou com dados drasticamente diferentes"
            
            self.results['passed'].append('Model Drift Detection')
            return True
            
        except Exception as e:
            self.results['failed'].append(f'Model Drift Detection: {str(e)}')
            logger.error(f"Erro no Model Drift Detection: {e}")
            return False
    
    def test_dynamic_model_selection(self) -> bool:
        """
        Testa a Seleção Dinâmica de Modelos
        """
        try:
            from dynamic_model_selection import DynamicModelSelector, SelectionStrategy
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            
            logger.info("Testando Dynamic Model Selection...")
            
            # Preparar modelos
            models = {
                'rf': RandomForestClassifier(n_estimators=10, random_state=42),
                'lr': LogisticRegression(random_state=42, max_iter=100)
            }
            
            # Preparar dados
            X = self.test_data[['open', 'high', 'low', 'volume']].iloc[1:]
            y = (self.test_data['close'].pct_change() > 0).astype(int).iloc[1:]
            
            model_selector = DynamicModelSelector(
                models=models,
                evaluation_window=50,
                min_samples=100
            )
            
            # Teste 1: Treinar modelos iniciais
            model_selector.fit_models(X[:200], y[:200])
            
            # Teste 2: Selecionar melhor modelo
            best_model = model_selector.select_best_model(strategy='best_recent')
            assert best_model is not None, "Nenhum modelo selecionado"
            
            # Teste 3: Fazer predições
            predictions, selection_info = model_selector.predict(X.iloc[200:210])
            assert len(predictions) == 10, "Número incorreto de predições"
            
            # Teste 4: Atualizar performance
            model_selector.update_performance(
                y_true=y.iloc[200:210],
                model_predictions={'rf': predictions, 'lr': predictions}
            )
            
            # Teste 5: Ensemble dinâmico (usando predict com estratégia weighted)
            model_selector.selection_strategy = SelectionStrategy.WEIGHTED_ENSEMBLE
            ensemble_pred, _ = model_selector.predict(X.iloc[210:220])
            assert len(ensemble_pred) == 10, "Ensemble prediction falhou"
            
            self.results['passed'].append('Dynamic Model Selection')
            return True
            
        except Exception as e:
            self.results['failed'].append(f'Dynamic Model Selection: {str(e)}')
            logger.error(f"Erro no Dynamic Model Selection: {e}")
            return False
    
    def test_advanced_feature_engineering(self) -> bool:
        """
        Testa o Advanced Feature Engineering
        """
        try:
            from advanced_feature_engineering_v2 import AdvancedFeatureEngineer
            
            logger.info("Testando Advanced Feature Engineering...")
            
            feature_engineer = AdvancedFeatureEngineer()
            
            # Teste 1: Criar todas as features usando fit_transform
            all_features = feature_engineer.fit_transform(self.test_data)
            
            assert len(all_features) > 0, "Nenhuma feature gerada"
            assert len(all_features.columns) > 10, "Poucas features geradas"
            
            # Teste 2: Features de microestrutura
            micro_features = feature_engineer._create_microstructure_features(self.test_data)
            assert 'vpin' in micro_features.columns, "VPIN não calculado"
            
            # Teste 3: Detecção de regimes
            regimes = feature_engineer._create_regime_features(self.test_data)
            assert 'vol_regime' in regimes.columns, "Regime de volatilidade não detectado"
            
            # Teste 4: Features estatísticas
            stat_features = feature_engineer._create_statistical_features(self.test_data)
            assert len(stat_features.columns) > 5, "Poucas features estatísticas"
            
            self.results['passed'].append('Advanced Feature Engineering')
            return True
            
        except Exception as e:
            self.results['failed'].append(f'Advanced Feature Engineering: {str(e)}')
            logger.error(f"Erro no Advanced Feature Engineering: {e}")
            return False
    
    def test_fractional_differentiation(self) -> bool:
        """
        Testa a Diferenciação Fracionária
        """
        try:
            from fractional_differentiation import FractionalDifferentiator, FractionalDiffConfig
            
            logger.info("Testando Fractional Differentiation...")
            
            # Teste 1: Aplicar diferenciação fracionária
            config = FractionalDiffConfig(optimize_d=False, verbose=False, parallel_processing=False)
            frac_diff = FractionalDifferentiator(config=config)
            diff_series = frac_diff.frac_diff(self.test_data['close'], d=0.5)
            
            assert len(diff_series) > 0, "Série diferenciada vazia"
            assert not diff_series.isnull().all(), "Série diferenciada contém apenas NaN"
            
            # Teste 2: Otimizar parâmetro d
            config_opt = FractionalDiffConfig(optimize_d=True, verbose=False, parallel_processing=False)
            optimizer = FractionalDifferentiator(config=config_opt)
            opt_result = optimizer.find_optimal_d(self.test_data['close'])
            optimal_d = opt_result['optimal_d']
            
            assert 0.0 <= optimal_d <= 1.0, f"Parâmetro d otimizado inválido: {optimal_d}"
            
            # Teste 3: Processar múltiplas séries
            multi_data = self.test_data[['open', 'high', 'low', 'close']]
            processed_data = optimizer.process_multiple_series(multi_data)
            
            assert processed_data.shape[1] == 4, "Número incorreto de séries processadas"
            
            self.results['passed'].append('Fractional Differentiation')
            return True
            
        except Exception as e:
            self.results['failed'].append(f'Fractional Differentiation: {str(e)}')
            logger.error(f"Erro no Fractional Differentiation: {e}")
            return False
    
    def test_walk_forward_analysis(self) -> bool:
        """
        Testa a Walk-Forward Analysis
        """
        try:
            from walk_forward_analysis import WalkForwardAnalyzer
            from sklearn.ensemble import RandomForestClassifier
            
            logger.info("Testando Walk-Forward Analysis...")
            
            # Preparar dados
            features = self.test_data[['open', 'high', 'low', 'volume']]
            labels = (self.test_data['close'].pct_change() > 0).astype(int)
            
            from walk_forward_analysis import WalkForwardConfig
            
            wf_config = WalkForwardConfig(
                train_window_size=30,
                test_window_size=7,
                purge_window=1,
                embargo_window=1,
                enable_parallel=False,
                min_train_samples=20  # Ajustar para permitir janelas menores
            )
            
            wfa = WalkForwardAnalyzer(config=wf_config)
            
            # Teste com subset menor para velocidade
            subset_size = 300
            results = wfa.run_walk_forward_analysis(
                X=features.head(subset_size),
                y=labels.head(subset_size),
                model_class=RandomForestClassifier,
                param_grid={'n_estimators': [10, 20]}
            )
            
            assert 'summary' in results, "Resumo dos resultados não encontrado"
            assert 'detailed_results' in results, "Resultados detalhados não encontrados"
            assert 'performance_metrics' in results['detailed_results'], "Métricas de performance não encontradas"
            
            self.results['passed'].append('Walk-Forward Analysis')
            return True
            
        except Exception as e:
            self.results['failed'].append(f'Walk-Forward Analysis: {str(e)}')
            logger.error(f"Erro no Walk-Forward Analysis: {e}")
            return False
    
    def test_risk_budgeting(self) -> bool:
        """
        Testa o Risk Budgeting System
        """
        try:
            from risk_budgeting import RiskBudgetingSystem
            
            logger.info("Testando Risk Budgeting System...")
            
            # Gerar dados de retornos sintéticos para múltiplos ativos
            np.random.seed(42)
            n_assets = 5
            n_periods = 200
            
            returns_data = pd.DataFrame(
                np.random.multivariate_normal(
                    mean=[0.0001] * n_assets,
                    cov=np.eye(n_assets) * 0.0004,
                    size=n_periods
                ),
                columns=[f'Asset_{i}' for i in range(n_assets)],
                index=pd.date_range(start='2023-01-01', periods=n_periods, freq='D')
            )
            
            from risk_budgeting import RiskBudgetConfig
            
            config = RiskBudgetConfig(
                lookback_window=100,
                rebalance_frequency='monthly',
                n_factors=3,
                optimization_method='scipy'  # Usar scipy ao invés de cvxpy
            )
            
            risk_system = RiskBudgetingSystem(config=config)
            
            # Teste 1: Ajustar modelo de fatores
            risk_system.fit_factor_model(returns_data)
            
            assert risk_system.factor_model is not None, "Modelo de fatores não foi ajustado"
            
            # Teste 2: Otimização de portfólio
            risk_budgets = {
                'PC1': 0.4,
                'PC2': 0.3,
                'PC3': 0.3
            }
            
            portfolio_result = risk_system.optimize_portfolio(
                expected_returns=None,
                risk_budget=risk_budgets,
                constraints={'max_weight': 0.3}
            )
            
            assert 'weights' in portfolio_result, "Otimização de portfólio falhou"
            optimal_weights = portfolio_result['weights']
            assert len(optimal_weights) == n_assets, "Pesos ótimos incorretos"
            assert abs(np.sum(optimal_weights) - 1.0) < 0.01, "Pesos não somam 1"
            
            self.results['passed'].append('Risk Budgeting System')
            return True
            
        except Exception as e:
            self.results['failed'].append(f'Risk Budgeting System: {str(e)}')
            logger.error(f"Erro no Risk Budgeting System: {e}")
            return False
    
    def test_sentiment_analysis(self) -> bool:
        """
        Testa o Sentiment Analysis (modo offline)
        """
        try:
            from sentiment_analysis import SentimentAnalyzer, SentimentConfig
            
            logger.info("Testando Sentiment Analysis (modo offline)...")
            
            config = SentimentConfig(
                news_api_key=None,  # Teste sem API
                cache_duration_hours=1
            )
            
            sentiment_analyzer = SentimentAnalyzer(config=config)
            
            # Teste 1: Análise de texto simples
            test_texts = [
                "The market is performing very well today with strong gains",
                "Economic uncertainty continues to weigh on investor sentiment",
                "Neutral market conditions with mixed signals"
            ]
            
            for text in test_texts:
                sentiment = sentiment_analyzer._analyze_sentiment(text)
                assert 'sentiment_textblob' in sentiment, "Análise TextBlob falhou"
                assert 'sentiment_vader' in sentiment, "Análise VADER falhou"
            
            # Teste 2: Processamento de texto
            processed = sentiment_analyzer._preprocess_text(test_texts[0])
            assert len(processed) > 0, "Preprocessamento de texto falhou"
            
            self.results['passed'].append('Sentiment Analysis (offline)')
            return True
            
        except Exception as e:
            self.results['failed'].append(f'Sentiment Analysis: {str(e)}')
            logger.error(f"Erro no Sentiment Analysis: {e}")
            return False
    
    def test_cross_asset_signals(self) -> bool:
        """
        Testa o Cross-Asset Signals (modo offline)
        """
        try:
            from cross_asset_signals import CrossAssetAnalyzer, CrossAssetConfig
            
            logger.info("Testando Cross-Asset Signals (modo offline)...")
            
            config = CrossAssetConfig(
                lookback_period=100,
                correlation_window=20,
                enable_parallel=False  # Desabilitar para teste
            )
            
            analyzer = CrossAssetAnalyzer(config=config)
            
            # Teste 1: Gerar dados sintéticos de múltiplos ativos
            market_data = {}
            for category in ['forex', 'stocks', 'commodities']:
                n_assets = 3
                asset_data = pd.DataFrame()
                
                for i in range(n_assets):
                    prices = 100 + np.cumsum(np.random.normal(0, 1, 200))
                    asset_data[f'{category}_asset_{i}_close'] = prices
                    # Adicionar colunas de returns necessárias
                    asset_data[f'{category}_asset_{i}_returns'] = asset_data[f'{category}_asset_{i}_close'].pct_change()
                    
                asset_data.index = pd.date_range(start='2023-01-01', periods=200, freq='D')
                market_data[category] = asset_data
            
            # Teste 2: Análise de correlações
            corr_analysis = analyzer.analyze_correlations(market_data)
            assert 'static_correlation' in corr_analysis, "Análise de correlação falhou"
            
            # Teste 3: Análise PCA
            pca_analysis = analyzer.perform_pca_analysis(market_data)
            assert 'pca_components' in pca_analysis, "Análise PCA falhou"
            
            self.results['passed'].append('Cross-Asset Signals (offline)')
            return True
            
        except Exception as e:
            self.results['failed'].append(f'Cross-Asset Signals: {str(e)}')
            logger.error(f"Erro no Cross-Asset Signals: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Executa todos os testes de validação
        
        Returns:
            Resultados completos dos testes
        """
        logger.info("=== INICIANDO VALIDAÇÃO DAS IMPLEMENTAÇÕES ===")
        
        # Lista de testes a executar
        tests = [
            ('Triple-Barrier Labeling', self.test_triple_barrier_labeling),
            ('Purged Cross-Validation', self.test_purged_cross_validation),
            ('Kelly Position Sizing', self.test_kelly_position_sizing),
            ('Model Drift Detection', self.test_model_drift_detection),
            ('Dynamic Model Selection', self.test_dynamic_model_selection),
            ('Advanced Feature Engineering', self.test_advanced_feature_engineering),
            ('Fractional Differentiation', self.test_fractional_differentiation),
            ('Walk-Forward Analysis', self.test_walk_forward_analysis),
            ('Risk Budgeting System', self.test_risk_budgeting),
            ('Sentiment Analysis', self.test_sentiment_analysis),
            ('Cross-Asset Signals', self.test_cross_asset_signals)
        ]
        
        # Executar testes
        total_tests = len(tests)
        passed_tests = 0
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n--- Executando: {test_name} ---")
                if test_func():
                    passed_tests += 1
                    logger.info(f"✓ {test_name}: PASSOU")
                else:
                    logger.error(f"✗ {test_name}: FALHOU")
            except Exception as e:
                logger.error(f"✗ {test_name}: ERRO - {str(e)}")
                self.results['failed'].append(f'{test_name}: {str(e)}')
        
        # Compilar resultados finais
        self.results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests) * 100,
            'timestamp': datetime.now()
        }
        
        return self.results
    
    def print_summary(self) -> None:
        """
        Imprime resumo dos resultados
        """
        summary = self.results['summary']
        
        print("\n" + "="*60)
        print("RESUMO DA VALIDAÇÃO DAS IMPLEMENTAÇÕES")
        print("="*60)
        
        print(f"\nTestes Executados: {summary['total_tests']}")
        print(f"Testes Aprovados: {summary['passed_tests']}")
        print(f"Testes Falharam: {summary['failed_tests']}")
        print(f"Taxa de Sucesso: {summary['success_rate']:.1f}%")
        
        if self.results['passed']:
            print("\n✓ TESTES APROVADOS:")
            for test in self.results['passed']:
                print(f"  - {test}")
        
        if self.results['failed']:
            print("\n✗ TESTES FALHARAM:")
            for test in self.results['failed']:
                print(f"  - {test}")
        
        if self.results['warnings']:
            print("\n⚠ AVISOS:")
            for warning in self.results['warnings']:
                print(f"  - {warning}")
        
        print(f"\nValidação concluída em: {summary['timestamp']}")
        print("="*60)
        
        # Recomendações baseadas nos resultados
        if summary['success_rate'] >= 90:
            print("\n🎉 EXCELENTE! Sistema pronto para integração.")
        elif summary['success_rate'] >= 70:
            print("\n✅ BOM! Revisar testes falharam antes da integração.")
        elif summary['success_rate'] >= 50:
            print("\n⚠️  ATENÇÃO! Corrigir falhas críticas antes de prosseguir.")
        else:
            print("\n❌ CRÍTICO! Revisar implementações antes da integração.")

def main():
    """
    Função principal para executar a validação
    """
    try:
        # Criar validador
        validator = ImplementationValidator()
        
        # Executar todos os testes
        results = validator.run_all_tests()
        
        # Imprimir resumo
        validator.print_summary()
        
        # Salvar resultados em arquivo
        results_file = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("RESULTADOS DA VALIDAÇÃO DAS IMPLEMENTAÇÕES\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Data/Hora: {results['summary']['timestamp']}\n")
            f.write(f"Total de Testes: {results['summary']['total_tests']}\n")
            f.write(f"Testes Aprovados: {results['summary']['passed_tests']}\n")
            f.write(f"Testes Falharam: {results['summary']['failed_tests']}\n")
            f.write(f"Taxa de Sucesso: {results['summary']['success_rate']:.1f}%\n\n")
            
            if results['passed']:
                f.write("TESTES APROVADOS:\n")
                for test in results['passed']:
                    f.write(f"  ✓ {test}\n")
                f.write("\n")
            
            if results['failed']:
                f.write("TESTES FALHARAM:\n")
                for test in results['failed']:
                    f.write(f"  ✗ {test}\n")
                f.write("\n")
        
        print(f"\nResultados salvos em: {results_file}")
        
        # Retornar código de saída baseado no sucesso
        if results['summary']['success_rate'] >= 70:
            return 0  # Sucesso
        else:
            return 1  # Falha
            
    except Exception as e:
        logger.error(f"Erro na execução da validação: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())