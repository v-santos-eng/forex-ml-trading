#!/usr/bin/env python3
"""
Script de Validação de Produção
Testa se o deployment foi realizado corretamente
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionValidator:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.production_dir = self.base_dir / 'ml_forex_production'
        
        # Adicionar diretório de produção ao path
        sys.path.insert(0, str(self.production_dir))
        
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'overall_status': 'PENDING'
        }
    
    def test_file_imports(self):
        """Testa se todos os módulos podem ser importados"""
        logger.info("Testando importações dos módulos...")
        
        modules_to_test = [
            'walk_forward_analysis',
            'model_drift_detection',
            'dynamic_model_selection',
            'fractional_differentiation',
            'risk_budgeting',
            'sentiment_analysis',
            'cross_asset_signals',
            'advanced_feature_engineering',
            'triple_barrier_labeling',
            'purged_cross_validation',
            'kelly_position_sizing'
        ]
        
        import_results = {}
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
                import_results[module_name] = {'status': 'SUCCESS', 'error': None}
                logger.info(f"✅ {module_name} importado com sucesso")
            except Exception as e:
                import_results[module_name] = {'status': 'FAILED', 'error': str(e)}
                logger.error(f"❌ Erro ao importar {module_name}: {e}")
        
        self.validation_results['tests']['imports'] = import_results
        return all(result['status'] == 'SUCCESS' for result in import_results.values())
    
    def test_class_instantiation(self):
        """Testa se as classes principais podem ser instanciadas"""
        logger.info("Testando instanciação das classes principais...")
        
        class_tests = {}
        
        # Teste WalkForwardAnalyzer
        try:
            from walk_forward_analysis import WalkForwardAnalyzer, WalkForwardConfig
            config = WalkForwardConfig()
            analyzer = WalkForwardAnalyzer(config=config)
            class_tests['WalkForwardAnalyzer'] = {'status': 'SUCCESS', 'error': None}
            logger.info("✅ WalkForwardAnalyzer instanciado com sucesso")
        except Exception as e:
            class_tests['WalkForwardAnalyzer'] = {'status': 'FAILED', 'error': str(e)}
            logger.error(f"❌ Erro ao instanciar WalkForwardAnalyzer: {e}")
        
        # Teste ModelDriftDetector
        try:
            from model_drift_detection import ModelDriftDetector
            detector = ModelDriftDetector()
            class_tests['ModelDriftDetector'] = {'status': 'SUCCESS', 'error': None}
            logger.info("✅ ModelDriftDetector instanciado com sucesso")
        except Exception as e:
            class_tests['ModelDriftDetector'] = {'status': 'FAILED', 'error': str(e)}
            logger.error(f"❌ Erro ao instanciar ModelDriftDetector: {e}")
        
        # Teste DynamicModelSelector
        try:
            from dynamic_model_selection import DynamicModelSelector
            from sklearn.ensemble import RandomForestClassifier
            models = [RandomForestClassifier()]
            selector = DynamicModelSelector(models=models)
            class_tests['DynamicModelSelector'] = {'status': 'SUCCESS', 'error': None}
            logger.info("✅ DynamicModelSelector instanciado com sucesso")
        except Exception as e:
            class_tests['DynamicModelSelector'] = {'status': 'FAILED', 'error': str(e)}
            logger.error(f"❌ Erro ao instanciar DynamicModelSelector: {e}")
        
        # Teste RiskBudgetingSystem
        try:
            from risk_budgeting import RiskBudgetingSystem, RiskBudgetConfig
            config = RiskBudgetConfig()
            risk_budgeter = RiskBudgetingSystem(config=config)
            class_tests['RiskBudgetingSystem'] = {'status': 'SUCCESS', 'error': None}
            logger.info("✅ RiskBudgetingSystem instanciado com sucesso")
        except Exception as e:
            class_tests['RiskBudgetingSystem'] = {'status': 'FAILED', 'error': str(e)}
            logger.error(f"❌ Erro ao instanciar RiskBudgetingSystem: {e}")
        
        self.validation_results['tests']['class_instantiation'] = class_tests
        return all(result['status'] == 'SUCCESS' for result in class_tests.values())
    
    def test_production_config(self):
        """Testa se a configuração de produção está válida"""
        logger.info("Testando configuração de produção...")
        
        config_file = self.production_dir / 'production_config.json'
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Verificar campos obrigatórios
            required_fields = ['environment', 'deployment_date', 'qa_tests_passed', 'version']
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                raise ValueError(f"Campos obrigatórios ausentes: {missing_fields}")
            
            if config['environment'] != 'production':
                raise ValueError(f"Environment incorreto: {config['environment']}")
            
            if not config['qa_tests_passed']:
                raise ValueError("QA tests não passaram")
            
            self.validation_results['tests']['production_config'] = {
                'status': 'SUCCESS',
                'config': config,
                'error': None
            }
            logger.info("✅ Configuração de produção válida")
            return True
            
        except Exception as e:
            self.validation_results['tests']['production_config'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"❌ Erro na configuração de produção: {e}")
            return False
    
    def test_basic_functionality(self):
        """Testa funcionalidades básicas dos módulos"""
        logger.info("Testando funcionalidades básicas...")
        
        functionality_tests = {}
        
        # Teste básico do Kelly Position Sizing
        try:
            from kelly_position_sizing import KellyPositionSizing
            import numpy as np
            
            kelly = KellyPositionSizing()
            returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
            kelly_fraction = kelly.calculate_kelly_fraction(returns)
            position_result = kelly.calculate_position_size(
                kelly_fraction=kelly_fraction,
                capital=10000.0,
                price=1.2000
            )
            
            if isinstance(position_result, dict) and 'position_size' in position_result:
                functionality_tests['kelly_position_sizing'] = {'status': 'SUCCESS', 'error': None}
                logger.info("✅ Kelly Position Sizing funcionando")
            else:
                raise ValueError(f"Position result inválido: {position_result}")
                
        except Exception as e:
            functionality_tests['kelly_position_sizing'] = {'status': 'FAILED', 'error': str(e)}
            logger.error(f"❌ Erro no Kelly Position Sizing: {e}")
        
        # Teste básico do Sentiment Analysis
        try:
            from sentiment_analysis import SentimentAnalyzer, SentimentConfig
            
            config = SentimentConfig()
            sentiment = SentimentAnalyzer(config=config)
            test_text = "The market is looking bullish today"
            result = sentiment.analyze_text(test_text)
            
            if isinstance(result, dict) and 'sentiment_score' in result:
                functionality_tests['sentiment_analysis'] = {'status': 'SUCCESS', 'error': None}
                logger.info("✅ Sentiment Analysis funcionando")
            else:
                raise ValueError(f"Resultado inválido: {result}")
                
        except Exception as e:
            functionality_tests['sentiment_analysis'] = {'status': 'FAILED', 'error': str(e)}
            logger.error(f"❌ Erro no Sentiment Analysis: {e}")
        
        self.validation_results['tests']['basic_functionality'] = functionality_tests
        return all(result['status'] == 'SUCCESS' for result in functionality_tests.values())
    
    def generate_validation_report(self):
        """Gera relatório de validação"""
        # Calcular status geral
        all_tests_passed = True
        total_tests = 0
        passed_tests = 0
        
        for test_category, tests in self.validation_results['tests'].items():
            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    total_tests += 1
                    if isinstance(result, dict) and result.get('status') == 'SUCCESS':
                        passed_tests += 1
                    else:
                        all_tests_passed = False
        
        self.validation_results['overall_status'] = 'SUCCESS' if all_tests_passed else 'FAILED'
        self.validation_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%"
        }
        
        # Salvar relatório
        report_file = self.base_dir / f'production_validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Relatório de validação salvo: {report_file}")
        return report_file
    
    def run_validation(self):
        """Executa validação completa"""
        logger.info("=" * 50)
        logger.info("INICIANDO VALIDAÇÃO DE PRODUÇÃO")
        logger.info("=" * 50)
        
        try:
            # Executar testes
            imports_ok = self.test_file_imports()
            classes_ok = self.test_class_instantiation()
            config_ok = self.test_production_config()
            functionality_ok = self.test_basic_functionality()
            
            # Gerar relatório
            report_file = self.generate_validation_report()
            
            # Resultado final
            all_ok = imports_ok and classes_ok and config_ok and functionality_ok
            
            logger.info("=" * 50)
            if all_ok:
                logger.info("✅ VALIDAÇÃO DE PRODUÇÃO CONCLUÍDA COM SUCESSO")
                logger.info("Sistema pronto para uso em produção")
            else:
                logger.error("❌ VALIDAÇÃO DE PRODUÇÃO FALHOU")
                logger.error("Verifique os erros acima antes de usar em produção")
            
            logger.info(f"Relatório detalhado: {report_file}")
            logger.info("=" * 50)
            
            return all_ok
            
        except Exception as e:
            logger.error(f"Erro durante validação: {e}")
            logger.error(traceback.format_exc())
            return False

if __name__ == "__main__":
    validator = ProductionValidator()
    success = validator.run_validation()
    
    if success:
        print("\n🎉 PRODUÇÃO VALIDADA COM SUCESSO!")
        print("Sistema aprovado para uso em produção.")
    else:
        print("\n⚠️ VALIDAÇÃO DE PRODUÇÃO FALHOU!")
        print("Corrija os problemas antes de usar em produção.")
        exit(1)