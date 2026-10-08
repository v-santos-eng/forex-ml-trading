#!/usr/bin/env python3
"""
Validação Simplificada de Produção
Testa apenas os aspectos críticos para deployment
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleProductionValidator:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.production_dir = self.base_dir / 'ml_forex_production'
        
        # Adicionar diretório de produção ao path
        sys.path.insert(0, str(self.production_dir))
        
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'PENDING',
            'critical_tests': {},
            'summary': {}
        }
    
    def test_critical_imports(self):
        """Testa importações críticas para o sistema"""
        logger.info("Testando importações críticas...")
        
        critical_modules = [
            'walk_forward_analysis',
            'model_drift_detection', 
            'dynamic_model_selection',
            'kelly_position_sizing',
            'risk_budgeting'
        ]
        
        import_results = {}
        all_imports_ok = True
        
        for module_name in critical_modules:
            try:
                __import__(module_name)
                import_results[module_name] = 'SUCCESS'
                logger.info(f"✅ {module_name} - OK")
            except Exception as e:
                import_results[module_name] = f'FAILED: {str(e)}'
                logger.error(f"❌ {module_name} - ERRO: {e}")
                all_imports_ok = False
        
        self.validation_results['critical_tests']['imports'] = import_results
        return all_imports_ok
    
    def test_core_classes(self):
        """Testa instanciação das classes principais"""
        logger.info("Testando classes principais...")
        
        class_results = {}
        all_classes_ok = True
        
        # Teste WalkForwardAnalyzer
        try:
            from walk_forward_analysis import WalkForwardAnalyzer, WalkForwardConfig
            config = WalkForwardConfig()
            analyzer = WalkForwardAnalyzer(config=config)
            class_results['WalkForwardAnalyzer'] = 'SUCCESS'
            logger.info("✅ WalkForwardAnalyzer - OK")
        except Exception as e:
            class_results['WalkForwardAnalyzer'] = f'FAILED: {str(e)}'
            logger.error(f"❌ WalkForwardAnalyzer - ERRO: {e}")
            all_classes_ok = False
        
        # Teste ModelDriftDetector
        try:
            from model_drift_detection import ModelDriftDetector
            detector = ModelDriftDetector()
            class_results['ModelDriftDetector'] = 'SUCCESS'
            logger.info("✅ ModelDriftDetector - OK")
        except Exception as e:
            class_results['ModelDriftDetector'] = f'FAILED: {str(e)}'
            logger.error(f"❌ ModelDriftDetector - ERRO: {e}")
            all_classes_ok = False
        
        # Teste KellyPositionSizing
        try:
            from kelly_position_sizing import KellyPositionSizing
            kelly = KellyPositionSizing()
            class_results['KellyPositionSizing'] = 'SUCCESS'
            logger.info("✅ KellyPositionSizing - OK")
        except Exception as e:
            class_results['KellyPositionSizing'] = f'FAILED: {str(e)}'
            logger.error(f"❌ KellyPositionSizing - ERRO: {e}")
            all_classes_ok = False
        
        self.validation_results['critical_tests']['classes'] = class_results
        return all_classes_ok
    
    def test_production_config(self):
        """Testa configuração de produção"""
        logger.info("Testando configuração de produção...")
        
        config_file = self.production_dir / 'production_config.json'
        
        try:
            if not config_file.exists():
                raise FileNotFoundError("Arquivo production_config.json não encontrado")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Verificar campos essenciais
            required_fields = ['environment', 'deployment_date', 'qa_tests_passed']
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                raise ValueError(f"Campos obrigatórios ausentes: {missing_fields}")
            
            if config['environment'] != 'production':
                raise ValueError(f"Environment incorreto: {config['environment']}")
            
            if not config['qa_tests_passed']:
                raise ValueError("QA tests não passaram")
            
            self.validation_results['critical_tests']['config'] = 'SUCCESS'
            logger.info("✅ Configuração de produção - OK")
            return True
            
        except Exception as e:
            self.validation_results['critical_tests']['config'] = f'FAILED: {str(e)}'
            logger.error(f"❌ Configuração de produção - ERRO: {e}")
            return False
    
    def test_file_structure(self):
        """Testa estrutura de arquivos essenciais"""
        logger.info("Testando estrutura de arquivos...")
        
        essential_files = [
            'walk_forward_analysis.py',
            'model_drift_detection.py',
            'kelly_position_sizing.py',
            'production_config.json'
        ]
        
        file_results = {}
        all_files_ok = True
        
        for filename in essential_files:
            file_path = self.production_dir / filename
            if file_path.exists():
                file_results[filename] = 'SUCCESS'
                logger.info(f"✅ {filename} - OK")
            else:
                file_results[filename] = 'FAILED: Arquivo não encontrado'
                logger.error(f"❌ {filename} - AUSENTE")
                all_files_ok = False
        
        self.validation_results['critical_tests']['files'] = file_results
        return all_files_ok
    
    def generate_summary_report(self):
        """Gera relatório resumido"""
        # Contar sucessos e falhas
        total_tests = 0
        passed_tests = 0
        
        for test_category, results in self.validation_results['critical_tests'].items():
            if isinstance(results, dict):
                for test_name, result in results.items():
                    total_tests += 1
                    if result == 'SUCCESS':
                        passed_tests += 1
        
        # Determinar status geral
        all_passed = (passed_tests == total_tests)
        self.validation_results['status'] = 'SUCCESS' if all_passed else 'FAILED'
        
        # Criar resumo
        self.validation_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%",
            'deployment_ready': all_passed
        }
        
        # Salvar relatório
        report_file = self.base_dir / f'simple_validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Relatório salvo: {report_file}")
        return report_file, all_passed
    
    def run_validation(self):
        """Executa validação simplificada"""
        logger.info("=" * 60)
        logger.info("VALIDAÇÃO SIMPLIFICADA DE PRODUÇÃO")
        logger.info("=" * 60)
        
        try:
            # Executar testes críticos
            imports_ok = self.test_critical_imports()
            classes_ok = self.test_core_classes()
            config_ok = self.test_production_config()
            files_ok = self.test_file_structure()
            
            # Gerar relatório
            report_file, all_ok = self.generate_summary_report()
            
            # Resultado final
            logger.info("=" * 60)
            if all_ok:
                logger.info("🎉 VALIDAÇÃO CONCLUÍDA COM SUCESSO")
                logger.info("✅ Sistema aprovado para produção")
                logger.info("✅ Deployment pode prosseguir")
            else:
                logger.error("❌ VALIDAÇÃO FALHOU")
                logger.error("⚠️ Corrija os problemas antes do deployment")
            
            logger.info(f"📄 Relatório: {report_file}")
            logger.info("=" * 60)
            
            return all_ok
            
        except Exception as e:
            logger.error(f"Erro crítico durante validação: {e}")
            return False

if __name__ == "__main__":
    validator = SimpleProductionValidator()
    success = validator.run_validation()
    
    if success:
        print("\n🚀 PRODUÇÃO VALIDADA - DEPLOYMENT APROVADO!")
        exit(0)
    else:
        print("\n🛑 VALIDAÇÃO FALHOU - DEPLOYMENT BLOQUEADO!")
        exit(1)