#!/usr/bin/env python3
"""
Teste de Integração do Sistema de Logging Detalhado

Este script testa a integração completa do sistema de logging
com o pipeline ML, validando todas as funcionalidades implementadas.

Autor: Sistema de Trading ML
Data: 2024
"""

import unittest
import tempfile
import shutil
import os
import json
from datetime import datetime
from typing import Dict, Any

# Importa componentes para teste
from comprehensive_analysis_logger import (
    ComprehensiveAnalysisLogger, AnalysisType, DataSource, AnalysisData,
    initialize_comprehensive_logger, get_comprehensive_logger
)
from detailed_decision_logger import (
    DetailedDecisionLogger, DecisionType, LogLevel, MarketConditions, AccountState,
    get_detailed_logger
)
from trading_decision_integrator import (
    TradingDecisionIntegrator, initialize_trading_integrator
)

class TestLoggingIntegration(unittest.TestCase):
    """
    Testes para validar a integração do sistema de logging
    """
    
    def setUp(self):
        """Configuração inicial para cada teste"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_symbol = 'EURUSD'
        
        # Inicializa componentes de logging
        initialize_comprehensive_logger(log_dir=self.temp_dir)
        initialize_trading_integrator()
        
        self.comprehensive_logger = get_comprehensive_logger()
        self.detailed_logger = get_detailed_logger()
        self.trading_integrator = TradingDecisionIntegrator()
    
    def tearDown(self):
        """Limpeza após cada teste"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_comprehensive_logger_initialization(self):
        """Testa inicialização do logger abrangente"""
        self.assertIsNotNone(self.comprehensive_logger)
        self.assertIsInstance(self.comprehensive_logger, ComprehensiveAnalysisLogger)
    
    def test_detailed_logger_initialization(self):
        """Testa inicialização do logger detalhado"""
        self.assertIsNotNone(self.detailed_logger)
        self.assertIsInstance(self.detailed_logger, DetailedDecisionLogger)
    
    def test_trading_integrator_initialization(self):
        """Testa inicialização do integrador de trading"""
        self.assertIsNotNone(self.trading_integrator)
        self.assertIsInstance(self.trading_integrator, TradingDecisionIntegrator)
    
    def test_technical_analysis_logging(self):
        """Testa logging de análise técnica"""
        # Dados de teste
        analysis_data = AnalysisData(
            analysis_type=AnalysisType.TECHNICAL,
            symbol=self.test_symbol,
            data_source=DataSource.ML_MODEL,
            raw_data={
                'rsi': 65.5,
                'macd': 0.0012,
                'bollinger_position': 0.75
            },
            confidence_score=0.85,
            processing_time_ms=150.0
        )
        
        # Registra análise
        result = self.comprehensive_logger.log_technical_analysis(analysis_data)
        self.assertTrue(result)
        
        # Verifica se foi registrada
        summary = self.comprehensive_logger.get_session_summary()
        self.assertGreater(summary['total_analyses'], 0)
        self.assertGreater(summary['technical_analyses'], 0)
    
    def test_risk_analysis_logging(self):
        """Testa logging de análise de risco"""
        # Dados de teste
        analysis_data = AnalysisData(
            analysis_type=AnalysisType.RISK,
            symbol=self.test_symbol,
            data_source=DataSource.ML_MODEL,
            raw_data={
                'var_1d': 0.025,
                'expected_shortfall': 0.045,
                'sharpe_ratio': 1.2
            },
            confidence_score=0.90,
            processing_time_ms=200.0
        )
        
        # Registra análise
        result = self.comprehensive_logger.log_risk_analysis(analysis_data)
        self.assertTrue(result)
        
        # Verifica se foi registrada
        summary = self.comprehensive_logger.get_session_summary()
        self.assertGreater(summary['risk_analyses'], 0)
    
    def test_market_analysis_logging(self):
        """Testa logging de análise de mercado"""
        # Condições de mercado de teste
        market_conditions = MarketConditions(
            volatility='medium',
            trend='bullish',
            volume='high',
            spread=0.0002,
            liquidity='normal'
        )
        
        # Estado da conta de teste
        account_state = AccountState(
            balance=10000.0,
            equity=10500.0,
            margin=500.0,
            free_margin=10000.0,
            margin_level=2100.0,
            open_positions=1
        )
        
        # Dados de análise
        analysis_data = {
            'prediction': 'buy',
            'probability': 0.75,
            'target_price': 1.1250
        }
        
        # Registra análise
        result = self.detailed_logger.log_market_analysis(
            symbol=self.test_symbol,
            market_conditions=market_conditions,
            account_state=account_state,
            analysis_data=analysis_data,
            confidence_score=0.75
        )
        self.assertTrue(result)
        
        # Verifica se foi registrada
        summary = self.detailed_logger.get_session_summary()
        self.assertGreater(summary['total_decisions'], 0)
    
    def test_comprehensive_analysis_logging(self):
        """Testa logging de análise abrangente"""
        # Dados de teste
        analysis_data = AnalysisData(
            analysis_type=AnalysisType.COMPREHENSIVE,
            symbol=self.test_symbol,
            data_source=DataSource.ML_MODEL,
            raw_data={
                'technical_score': 0.75,
                'fundamental_score': 0.65,
                'risk_score': 0.80,
                'final_recommendation': 'buy',
                'confidence': 0.73
            },
            confidence_score=0.73,
            processing_time_ms=500.0
        )
        
        # Registra análise
        result = self.comprehensive_logger.log_comprehensive_analysis(analysis_data)
        self.assertTrue(result)
        
        # Verifica se foi registrada
        summary = self.comprehensive_logger.get_session_summary()
        self.assertGreater(summary['comprehensive_analyses'], 0)
    
    def test_session_summary_generation(self):
        """Testa geração de resumo da sessão"""
        # Registra algumas análises
        self._register_sample_analyses()
        
        # Obtém resumo do logger abrangente
        comp_summary = self.comprehensive_logger.get_session_summary()
        self.assertIsInstance(comp_summary, dict)
        self.assertIn('total_analyses', comp_summary)
        self.assertIn('session_start_time', comp_summary)
        
        # Obtém resumo do logger detalhado
        detail_summary = self.detailed_logger.get_session_summary()
        self.assertIsInstance(detail_summary, dict)
        self.assertIn('total_decisions', detail_summary)
        self.assertIn('session_start_time', detail_summary)
    
    def test_report_export(self):
        """Testa exportação de relatórios"""
        # Registra algumas análises
        self._register_sample_analyses()
        
        # Exporta relatório do logger abrangente
        comp_report_path = self.comprehensive_logger.export_session_report(self.temp_dir)
        self.assertTrue(os.path.exists(comp_report_path))
        
        # Verifica conteúdo do relatório
        with open(comp_report_path, 'r') as f:
            report_data = json.load(f)
        
        self.assertIn('session_summary', report_data)
        self.assertIn('analyses', report_data)
        
        # Exporta relatório do logger detalhado
        detail_report_path = self.detailed_logger.export_session_report(self.temp_dir)
        self.assertTrue(os.path.exists(detail_report_path))
    
    def test_trading_integrator_functionality(self):
        """Testa funcionalidades do integrador de trading"""
        # Inicia monitoramento
        self.trading_integrator.start_monitoring()
        
        # Verifica se está ativo
        summary = self.trading_integrator.get_session_summary()
        self.assertTrue(summary['session_active'])
        
        # Para monitoramento
        self.trading_integrator.stop_monitoring()
        
        # Verifica se parou
        summary = self.trading_integrator.get_session_summary()
        self.assertFalse(summary['session_active'])
    
    def test_file_persistence(self):
        """Testa persistência de dados em arquivos"""
        # Registra análises
        self._register_sample_analyses()
        
        # Verifica se arquivos foram criados
        log_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.jsonl')]
        self.assertGreater(len(log_files), 0)
        
        # Verifica conteúdo dos arquivos
        for log_file in log_files:
            file_path = os.path.join(self.temp_dir, log_file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                self.assertGreater(len(lines), 0)
                
                # Verifica se cada linha é JSON válido
                for line in lines:
                    try:
                        json.loads(line.strip())
                    except json.JSONDecodeError:
                        self.fail(f"Linha inválida no arquivo {log_file}: {line}")
    
    def test_error_handling(self):
        """Testa tratamento de erros"""
        # Testa com dados inválidos
        try:
            invalid_analysis = AnalysisData(
                analysis_type=AnalysisType.TECHNICAL,
                symbol="",  # Símbolo vazio
                data_source=DataSource.ML_MODEL,
                raw_data=None,  # Dados nulos
                confidence_score=-1.0,  # Confiança inválida
                processing_time_ms=-100.0  # Tempo negativo
            )
            
            # Deve lidar com dados inválidos graciosamente
            result = self.comprehensive_logger.log_technical_analysis(invalid_analysis)
            # O resultado pode ser False, mas não deve gerar exceção
            
        except Exception as e:
            self.fail(f"Tratamento de erro falhou: {e}")
    
    def _register_sample_analyses(self):
        """Registra análises de exemplo para testes"""
        # Análise técnica
        technical_data = AnalysisData(
            analysis_type=AnalysisType.TECHNICAL,
            symbol=self.test_symbol,
            data_source=DataSource.ML_MODEL,
            raw_data={'rsi': 70, 'macd': 0.001},
            confidence_score=0.80,
            processing_time_ms=100.0
        )
        self.comprehensive_logger.log_technical_analysis(technical_data)
        
        # Análise de risco
        risk_data = AnalysisData(
            analysis_type=AnalysisType.RISK,
            symbol=self.test_symbol,
            data_source=DataSource.ML_MODEL,
            raw_data={'var': 0.02, 'sharpe': 1.5},
            confidence_score=0.85,
            processing_time_ms=200.0
        )
        self.comprehensive_logger.log_risk_analysis(risk_data)
        
        # Análise de mercado
        market_conditions = MarketConditions(
            volatility='low',
            trend='neutral',
            volume='normal',
            spread=0.0001,
            liquidity='high'
        )
        
        account_state = AccountState(
            balance=5000.0,
            equity=5200.0,
            margin=200.0,
            free_margin=5000.0,
            margin_level=2600.0,
            open_positions=0
        )
        
        self.detailed_logger.log_market_analysis(
            symbol=self.test_symbol,
            market_conditions=market_conditions,
            account_state=account_state,
            analysis_data={'test': 'data'},
            confidence_score=0.75
        )

class TestMLPipelineIntegration(unittest.TestCase):
    """
    Testes específicos para integração com pipeline ML
    """
    
    def setUp(self):
        """Configuração inicial"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Limpeza"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_pipeline_logging_methods_exist(self):
        """Testa se os métodos de logging existem no pipeline"""
        from ml_production_pipeline import MLProductionPipeline
        
        # Verifica se os métodos foram adicionados
        pipeline_methods = dir(MLProductionPipeline)
        
        self.assertIn('log_ml_analysis', pipeline_methods)
        self.assertIn('log_ml_prediction', pipeline_methods)
        self.assertIn('get_logging_summary', pipeline_methods)
        self.assertIn('export_logging_reports', pipeline_methods)
    
    def test_pipeline_imports(self):
        """Testa se as importações do pipeline estão corretas"""
        try:
            from ml_production_pipeline import MLProductionPipeline
            # Se chegou até aqui, as importações estão funcionando
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Erro de importação no pipeline: {e}")

def run_integration_tests():
    """
    Executa todos os testes de integração
    """
    print("🧪 Executando Testes de Integração do Sistema de Logging")
    print("=" * 60)
    
    # Cria suite de testes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Adiciona testes
    suite.addTests(loader.loadTestsFromTestCase(TestLoggingIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestMLPipelineIntegration))
    
    # Executa testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Relatório final
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ Todos os testes passaram com sucesso!")
        print(f"   - Testes executados: {result.testsRun}")
        print(f"   - Falhas: {len(result.failures)}")
        print(f"   - Erros: {len(result.errors)}")
    else:
        print("❌ Alguns testes falharam:")
        print(f"   - Testes executados: {result.testsRun}")
        print(f"   - Falhas: {len(result.failures)}")
        print(f"   - Erros: {len(result.errors)}")
        
        if result.failures:
            print("\n🔍 Falhas:")
            for test, traceback in result.failures:
                print(f"   - {test}: {traceback}")
        
        if result.errors:
            print("\n🚨 Erros:")
            for test, traceback in result.errors:
                print(f"   - {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)