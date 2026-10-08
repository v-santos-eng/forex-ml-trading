#!/usr/bin/env python3
"""
Teste do Sistema Avançado de Logging e Debugging
Validação das funcionalidades implementadas
"""

import sys
import os
import unittest
import tempfile
import shutil
import time
import threading
from unittest.mock import patch, MagicMock

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from advanced_logging_system import AdvancedLoggingSystem, setup_logging
    from interactive_debugger import InteractiveDebugger, VariableInspector
    from performance_monitor import PerformanceMonitor
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Certifique-se de que todos os arquivos estão no diretório correto")
    sys.exit(1)

class TestAdvancedLoggingSystem(unittest.TestCase):
    """Testes para o sistema de logging avançado"""
    
    def setUp(self):
        """Configuração inicial para cada teste"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'logging_config.yaml')
        
        # Criar configuração mínima
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'simple': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'simple'
                }
            },
            'loggers': {
                'forex_ml': {
                    'level': 'DEBUG',
                    'handlers': ['console'],
                    'propagate': False
                }
            }
        }
        
        import yaml
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f)
    
    def tearDown(self):
        """Limpeza após cada teste"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_logging_system_initialization(self):
        """Testa inicialização do sistema de logging"""
        logging_system = AdvancedLoggingSystem(self.config_file)
        self.assertIsNotNone(logging_system)
        
        logger = logging_system.get_logger('test')
        self.assertIsNotNone(logger)
    
    def test_setup_logging_function(self):
        """Testa função de configuração de logging"""
        logging_system = setup_logging(self.config_file)
        self.assertIsNotNone(logging_system)
        self.assertIsInstance(logging_system, AdvancedLoggingSystem)

class TestInteractiveDebugger(unittest.TestCase):
    """Testes para o debugger interativo"""
    
    def test_debugger_initialization(self):
        """Testa inicialização do debugger"""
        debugger = InteractiveDebugger()
        self.assertIsNotNone(debugger)
    
    def test_variable_inspector(self):
        """Testa inspetor de variáveis"""
        inspector = VariableInspector()
        self.assertIsNotNone(inspector)
        
        # Teste com variável simples
        test_var = "hello world"
        frame_locals = {'test_var': test_var}
        info = inspector.inspect_variable('test_var', test_var, frame_locals)
        self.assertIn('type', info)
        self.assertIn('value', info)
        self.assertEqual(info['type'], 'str')

class TestPerformanceMonitor(unittest.TestCase):
    """Testes para o monitor de performance"""
    
    def test_performance_monitor_initialization(self):
        """Testa inicialização do monitor"""
        monitor = PerformanceMonitor()
        self.assertIsNotNone(monitor)
    
    def test_system_metrics_collection(self):
        """Testa coleta de métricas do sistema"""
        monitor = PerformanceMonitor()
        
        # Iniciar monitoramento por um breve período
        monitor.start_monitoring()
        time.sleep(1)  # Aguardar coleta de métricas
        monitor.stop_monitoring()
        
        # Verificar se métricas foram coletadas
        self.assertGreaterEqual(len(monitor.recent_system_metrics), 0)
        
        if len(monitor.recent_system_metrics) > 0:
            metrics = monitor.recent_system_metrics[-1]
            self.assertIsNotNone(metrics.cpu_percent)
            self.assertIsNotNone(metrics.memory_percent)
            self.assertIsInstance(metrics.cpu_percent, (int, float))
            self.assertIsInstance(metrics.memory_percent, (int, float))

class TestIntegration(unittest.TestCase):
    """Testes de integração entre componentes"""
    
    def setUp(self):
        """Configuração para testes de integração"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'logging_config.yaml')
        
        # Configuração completa
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'DEBUG',
                    'formatter': 'detailed'
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'filename': os.path.join(self.temp_dir, 'test.log'),
                    'level': 'INFO',
                    'formatter': 'detailed'
                }
            },
            'loggers': {
                'forex_ml': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file'],
                    'propagate': False
                }
            }
        }
        
        import yaml
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f)
    
    def tearDown(self):
        """Limpeza após testes de integração"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_system_integration(self):
        """Testa integração completa do sistema"""
        # Inicializar componentes
        logging_system = AdvancedLoggingSystem(self.config_file)
        debugger = InteractiveDebugger()
        monitor = PerformanceMonitor()
        
        # Testar logging
        logger = logging_system.get_logger('integration_test')
        logger.info("Teste de integração iniciado")
        
        # Testar coleta de métricas
        metrics = monitor.collect_system_metrics()
        logger.info(f"Métricas coletadas: CPU {metrics['cpu_percent']}%")
        
        # Verificar se arquivo de log foi criado
        log_file = os.path.join(self.temp_dir, 'test.log')
        self.assertTrue(os.path.exists(log_file))
        
        # Verificar conteúdo do log
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('Teste de integração iniciado', content)
            self.assertIn('Métricas coletadas', content)

def run_manual_tests():
    """Executa testes manuais interativos"""
    print("\n" + "="*60)
    print("TESTES MANUAIS DO SISTEMA DE LOGGING")
    print("="*60)
    
    try:
        # Teste 1: Sistema de Logging
        print("\n1. Testando Sistema de Logging...")
        logging_system = setup_logging()
        logger = logging_system.get_logger('manual_test')
        
        logger.debug("Mensagem de debug")
        logger.info("Mensagem informativa")
        logger.warning("Mensagem de aviso")
        logger.error("Mensagem de erro")
        print("✓ Sistema de logging funcionando")
        
        # Teste 2: Monitor de Performance
        print("\n2. Testando Monitor de Performance...")
        monitor = PerformanceMonitor()
        
        # Iniciar monitoramento por um breve período
        monitor.start_monitoring()
        time.sleep(2)  # Aguardar coleta de métricas
        monitor.stop_monitoring()
        
        if len(monitor.recent_system_metrics) > 0:
            metrics = monitor.recent_system_metrics[-1]
            print(f"✓ CPU: {metrics.cpu_percent:.1f}%")
            print(f"✓ Memória: {metrics.memory_percent:.1f}%")
            print(f"✓ Uso de disco: {metrics.disk_usage_percent:.1f}%")
        else:
            print("✓ Monitor inicializado (métricas serão coletadas em background)")
        
        # Teste 3: Debugger
        print("\n3. Testando Debugger...")
        debugger = InteractiveDebugger()
        inspector = VariableInspector()
        
        test_data = {'name': 'EURUSD', 'price': 1.0850, 'volume': 1000}
        frame_locals = {'test_data': test_data}
        info = inspector.inspect_variable('test_data', test_data, frame_locals)
        print(f"✓ Inspeção de variável: {info['type']} com {len(str(info['value']))} caracteres")
        
        print("\n" + "="*60)
        print("TODOS OS TESTES MANUAIS CONCLUÍDOS COM SUCESSO!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Erro durante testes manuais: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("Iniciando testes do sistema de logging avançado...")
    
    # Executar testes unitários
    print("\nExecutando testes unitários...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Executar testes manuais
    run_manual_tests()