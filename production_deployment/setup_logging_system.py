#!/usr/bin/env python3
"""
Script de Configuração Automática do Sistema de Logging Detalhado

Este script automatiza a instalação e configuração inicial do sistema
de logging detalhado, incluindo:
- Verificação de dependências
- Criação de estrutura de diretórios
- Configuração inicial dos componentes
- Testes de funcionalidade
- Geração de arquivos de exemplo

Autor: Sistema de Trading ML
Versão: 1.0.0
Data: 2024
"""

import os
import sys
import json
import yaml
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import subprocess
import shutil


class LoggingSystemSetup:
    """Configurador automático do sistema de logging."""
    
    def __init__(self):
        """Inicializar configurador."""
        self.base_dir = Path.cwd()
        self.setup_log = []
        self.errors = []
        
        # Estrutura de diretórios necessária
        self.required_dirs = [
            'logs',
            'logs/comprehensive',
            'logs/detailed',
            'logs/trading',
            'reports',
            'reports/daily',
            'reports/weekly',
            'reports/monthly',
            'config',
            'backups',
            'temp'
        ]
        
        # Arquivos de configuração necessários
        self.config_files = {
            'logging_config.yaml': self._get_default_logging_config(),
            'system_config.json': self._get_default_system_config()
        }
        
        # Dependências Python necessárias
        self.required_packages = [
            'pyyaml>=6.0',
            'pandas>=1.5.0',
            'numpy>=1.21.0',
            'asyncio',
            'pathlib',
            'typing',
            'dataclasses',
            'enum',
            'json',
            'datetime'
        ]
    
    def log_step(self, message: str, success: bool = True) -> None:
        """Registrar passo da configuração."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        status = "✓" if success else "✗"
        log_entry = f"[{timestamp}] {status} {message}"
        
        self.setup_log.append(log_entry)
        print(log_entry)
        
        if not success:
            self.errors.append(message)
    
    def check_python_version(self) -> bool:
        """Verificar versão do Python."""
        try:
            version = sys.version_info
            if version.major >= 3 and version.minor >= 8:
                self.log_step(f"Python {version.major}.{version.minor}.{version.micro} - Compatível")
                return True
            else:
                self.log_step(f"Python {version.major}.{version.minor}.{version.micro} - Versão muito antiga (requer 3.8+)", False)
                return False
        except Exception as e:
            self.log_step(f"Erro ao verificar versão do Python: {e}", False)
            return False
    
    def check_dependencies(self) -> bool:
        """Verificar e instalar dependências."""
        self.log_step("Verificando dependências Python...")
        
        missing_packages = []
        
        for package in self.required_packages:
            try:
                # Extrair nome do pacote (sem versão)
                package_name = package.split('>=')[0].split('==')[0]
                
                if package_name in ['asyncio', 'pathlib', 'typing', 'dataclasses', 'enum', 'json', 'datetime']:
                    # Módulos built-in do Python
                    __import__(package_name)
                    self.log_step(f"  {package_name} - Disponível (built-in)")
                else:
                    # Pacotes externos
                    __import__(package_name)
                    self.log_step(f"  {package_name} - Instalado")
            
            except ImportError:
                missing_packages.append(package)
                self.log_step(f"  {package} - Não encontrado", False)
        
        if missing_packages:
            self.log_step(f"Encontrados {len(missing_packages)} pacotes faltando")
            
            # Tentar instalar pacotes faltando
            for package in missing_packages:
                try:
                    self.log_step(f"Instalando {package}...")
                    result = subprocess.run(
                        [sys.executable, '-m', 'pip', 'install', package],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        self.log_step(f"  {package} - Instalado com sucesso")
                    else:
                        self.log_step(f"  {package} - Falha na instalação: {result.stderr}", False)
                        return False
                
                except subprocess.TimeoutExpired:
                    self.log_step(f"  {package} - Timeout na instalação", False)
                    return False
                except Exception as e:
                    self.log_step(f"  {package} - Erro na instalação: {e}", False)
                    return False
        
        self.log_step("Todas as dependências estão disponíveis")
        return True
    
    def create_directory_structure(self) -> bool:
        """Criar estrutura de diretórios."""
        self.log_step("Criando estrutura de diretórios...")
        
        try:
            for dir_path in self.required_dirs:
                full_path = self.base_dir / dir_path
                full_path.mkdir(parents=True, exist_ok=True)
                self.log_step(f"  {dir_path}/ - Criado")
            
            # Criar arquivo .gitkeep nos diretórios vazios
            for dir_path in self.required_dirs:
                gitkeep_path = self.base_dir / dir_path / '.gitkeep'
                if not gitkeep_path.exists():
                    gitkeep_path.touch()
            
            self.log_step("Estrutura de diretórios criada com sucesso")
            return True
        
        except Exception as e:
            self.log_step(f"Erro ao criar diretórios: {e}", False)
            return False
    
    def create_config_files(self) -> bool:
        """Criar arquivos de configuração."""
        self.log_step("Criando arquivos de configuração...")
        
        try:
            config_dir = self.base_dir / 'config'
            
            for filename, content in self.config_files.items():
                file_path = config_dir / filename
                
                if file_path.exists():
                    # Fazer backup do arquivo existente
                    backup_path = file_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}{file_path.suffix}')
                    shutil.copy2(file_path, backup_path)
                    self.log_step(f"  {filename} - Backup criado: {backup_path.name}")
                
                # Escrever novo arquivo
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        yaml.dump(content, f, default_flow_style=False, allow_unicode=True, indent=2)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(content, f, indent=2, ensure_ascii=False)
                
                self.log_step(f"  {filename} - Criado")
            
            self.log_step("Arquivos de configuração criados com sucesso")
            return True
        
        except Exception as e:
            self.log_step(f"Erro ao criar arquivos de configuração: {e}", False)
            return False
    
    def verify_logging_components(self) -> bool:
        """Verificar se os componentes de logging estão disponíveis."""
        self.log_step("Verificando componentes de logging...")
        
        required_files = [
            'comprehensive_analysis_logger.py',
            'detailed_decision_logger.py',
            'trading_decision_integrator.py',
            'logging_utilities.py'
        ]
        
        missing_files = []
        
        for filename in required_files:
            file_path = self.base_dir / filename
            if file_path.exists():
                self.log_step(f"  {filename} - Encontrado")
                
                # Tentar importar para verificar sintaxe
                try:
                    if filename == 'comprehensive_analysis_logger.py':
                        from comprehensive_analysis_logger import ComprehensiveAnalysisLogger
                    elif filename == 'detailed_decision_logger.py':
                        from detailed_decision_logger import DetailedDecisionLogger
                    elif filename == 'trading_decision_integrator.py':
                        from trading_decision_integrator import TradingDecisionIntegrator
                    elif filename == 'logging_utilities.py':
                        from logging_utilities import LoggingUtilities
                    
                    self.log_step(f"    {filename} - Importação OK")
                
                except Exception as e:
                    self.log_step(f"    {filename} - Erro na importação: {e}", False)
                    return False
            else:
                missing_files.append(filename)
                self.log_step(f"  {filename} - Não encontrado", False)
        
        if missing_files:
            self.log_step(f"Arquivos faltando: {', '.join(missing_files)}", False)
            return False
        
        self.log_step("Todos os componentes de logging estão disponíveis")
        return True
    
    def run_functionality_tests(self) -> bool:
        """Executar testes básicos de funcionalidade."""
        self.log_step("Executando testes de funcionalidade...")
        
        try:
            # Teste 1: Inicialização dos loggers
            from comprehensive_analysis_logger import initialize_comprehensive_logger, get_comprehensive_logger
            from detailed_decision_logger import initialize_detailed_logger, get_detailed_logger
            from trading_decision_integrator import initialize_trading_integrator
            
            initialize_comprehensive_logger()
            initialize_detailed_logger()
            initialize_trading_integrator()
            
            comp_logger = get_comprehensive_logger()
            detail_logger = get_detailed_logger()
            
            self.log_step("  Inicialização dos loggers - OK")
            
            # Teste 2: Criação de logs de teste
            from comprehensive_analysis_logger import AnalysisData, AnalysisType, DataSource
            
            test_analysis = AnalysisData(
                analysis_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                symbol='EURUSD',
                timeframe='H1',
                analysis_type=AnalysisType.TECHNICAL,
                data_sources=[DataSource.ML_MODEL],
                technical_indicators=[],
                price_levels=[],
                chart_patterns=[],
                economic_events=[],
                news_items=[],
                risk_metrics=[],
                market_conditions=[],
                correlations={},
                sentiment_data={'test': 'data'},
                key_observations=[],
                risk_factors=[],
                opportunities=[],
                recommendations=[],
                analyst='system',
                confidence_score=0.85,
                processing_time_ms=100,
                data_quality_score=0.8
            )
            
            comp_logger._save_analysis(test_analysis, comp_logger.technical_log)
            self.log_step("  Log de análise técnica - OK")
            
            # Teste 3: Verificação de arquivos de log
            log_files = list((self.base_dir / 'logs').rglob('*.jsonl'))
            if log_files:
                self.log_step(f"  Arquivos de log criados: {len(log_files)}")
            else:
                self.log_step("  Nenhum arquivo de log encontrado", False)
                return False
            
            self.log_step("Testes de funcionalidade concluídos com sucesso")
            return True
        
        except Exception as e:
            self.log_step(f"Erro nos testes de funcionalidade: {e}", False)
            return False
    
    def create_example_scripts(self) -> bool:
        """Criar scripts de exemplo."""
        self.log_step("Criando scripts de exemplo...")
        
        try:
            examples_dir = self.base_dir / 'examples'
            examples_dir.mkdir(exist_ok=True)
            
            # Script de exemplo básico
            basic_example = '''
#!/usr/bin/env python3
"""
Exemplo Básico do Sistema de Logging
"""

from comprehensive_analysis_logger import (
    initialize_comprehensive_logger, get_comprehensive_logger,
    AnalysisData, AnalysisType, DataSource
)
from detailed_decision_logger import (
    initialize_detailed_logger, get_detailed_logger,
    MarketConditions, AccountState
)

def main():
    # Inicializar loggers
    initialize_comprehensive_logger()
    initialize_detailed_logger()
    
    comp_logger = get_comprehensive_logger()
    detail_logger = get_detailed_logger()
    
    # Exemplo de análise técnica
    analysis = AnalysisData(
                analysis_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                symbol='EURUSD',
                timeframe='H1',
                analysis_type=AnalysisType.TECHNICAL,
                data_sources=[DataSource.ML_MODEL],
                technical_indicators=[],
                price_levels=[],
                chart_patterns=[],
                economic_events=[],
                news_items=[],
                risk_metrics=[],
                market_conditions=[],
                correlations={},
                sentiment_data={
                    'rsi': 65.5,
                    'macd': 0.0012,
                    'trend': 'bullish'
                },
                key_observations=[],
                risk_factors=[],
                opportunities=[],
                recommendations=[],
                analyst='system',
                confidence_score=0.87,
                processing_time_ms=150,
                data_quality_score=0.9
            )
    
    comp_logger._save_analysis(test_analysis, comp_logger.technical_log)
    print("Análise técnica registrada com sucesso!")
    
    # Exemplo de decisão de não operar
    market_conditions = MarketConditions(
        volatility='high',
        trend='neutral',
        volume='low',
        spread=0.0003,
        liquidity='normal'
    )
    
    account_state = AccountState(
        balance=10000.0,
        equity=10050.0,
        margin=200.0,
        free_margin=9850.0,
        margin_level=5025.0,
        open_positions=1
    )
    
    detail_logger.log_no_trade_decision(
        symbol='EURUSD',
        market_conditions=market_conditions,
        account_state=account_state,
        reasoning='Volatilidade muito alta para entrada segura'
    )
    
    print("Decisão de não operar registrada com sucesso!")
    
    # Gerar relatório
    report_path = comp_logger.export_session_report('./reports')
    print(f"Relatório gerado: {report_path}")

if __name__ == '__main__':
    main()
'''
            
            with open(examples_dir / 'basic_logging_example.py', 'w', encoding='utf-8') as f:
                f.write(basic_example)
            
            self.log_step("  basic_logging_example.py - Criado")
            
            # README para exemplos
            readme_content = '''
# Exemplos do Sistema de Logging

Este diretório contém exemplos de uso do sistema de logging detalhado.

## Arquivos Disponíveis

- `basic_logging_example.py`: Exemplo básico de uso dos loggers
- `production_logging_demo.py`: Demonstração completa em ambiente de produção

## Como Executar

```bash
# Exemplo básico
python examples/basic_logging_example.py

# Demonstração completa
python production_logging_demo.py
```

## Estrutura de Logs

Os logs são salvos em:
- `logs/comprehensive/`: Análises abrangentes
- `logs/detailed/`: Decisões detalhadas
- `logs/trading/`: Integrações de trading

## Relatórios

Os relatórios são gerados em:
- `reports/`: Relatórios de sessão
- `reports/daily/`: Relatórios diários
- `reports/weekly/`: Relatórios semanais
'''
            
            with open(examples_dir / 'README.md', 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            self.log_step("  README.md - Criado")
            
            self.log_step("Scripts de exemplo criados com sucesso")
            return True
        
        except Exception as e:
            self.log_step(f"Erro ao criar scripts de exemplo: {e}", False)
            return False
    
    def generate_setup_report(self) -> str:
        """Gerar relatório da configuração."""
        report = {
            'setup_timestamp': datetime.now().isoformat(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'base_directory': str(self.base_dir),
            'setup_log': self.setup_log,
            'errors': self.errors,
            'success': len(self.errors) == 0,
            'directories_created': self.required_dirs,
            'config_files_created': list(self.config_files.keys()),
            'next_steps': [
                "Execute 'python examples/basic_logging_example.py' para testar o sistema",
                "Execute 'python production_logging_demo.py' para demonstração completa",
                "Revise os arquivos de configuração em 'config/' conforme necessário",
                "Configure monitoramento automático se desejado"
            ]
        }
        
        report_path = self.base_dir / 'setup_report.json'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return str(report_path)
    
    def _get_default_logging_config(self) -> Dict[str, Any]:
        """Obter configuração padrão de logging."""
        return {
            'general': {
                'log_directory': './logs',
                'report_directory': './reports',
                'log_level': 'INFO',
                'timestamp_format': '%Y-%m-%d %H:%M:%S.%f',
                'timezone': 'UTC',
                'async_logging': True,
                'max_cache_size': 1000
            },
            'comprehensive_logger': {
                'enabled': True,
                'log_file_prefix': 'comprehensive_analysis',
                'rotate_logs': True,
                'max_file_size_mb': 50,
                'backup_count': 10,
                'analysis_types': {
                    'technical': {'enabled': True, 'min_confidence': 0.5},
                    'fundamental': {'enabled': True, 'min_confidence': 0.4},
                    'risk': {'enabled': True, 'min_confidence': 0.6},
                    'comprehensive': {'enabled': True, 'min_confidence': 0.7}
                }
            },
            'detailed_logger': {
                'enabled': True,
                'log_file_prefix': 'detailed_decisions',
                'rotate_logs': True,
                'max_file_size_mb': 30,
                'backup_count': 15,
                'decision_types': {
                    'trade_operation': {'enabled': True},
                    'no_trade': {'enabled': True},
                    'balance_update': {'enabled': True}
                }
            },
            'trading_integrator': {
                'enabled': True,
                'log_file_prefix': 'trading_integration',
                'rotate_logs': True,
                'max_file_size_mb': 40,
                'backup_count': 12
            },
            'performance': {
                'buffer_size': 100,
                'flush_interval_seconds': 30,
                'compression_enabled': True,
                'indexing_enabled': True,
                'max_threads': 4
            },
            'maintenance': {
                'auto_cleanup': True,
                'log_retention_days': 90,
                'report_retention_days': 180,
                'cleanup_interval_hours': 24,
                'min_free_disk_gb': 5
            }
        }
    
    def _get_default_system_config(self) -> Dict[str, Any]:
        """Obter configuração padrão do sistema."""
        return {
            'system': {
                'name': 'ML Trading Logging System',
                'version': '1.0.0',
                'environment': 'development',
                'debug_mode': True
            },
            'paths': {
                'base_directory': str(self.base_dir),
                'logs_directory': str(self.base_dir / 'logs'),
                'reports_directory': str(self.base_dir / 'reports'),
                'config_directory': str(self.base_dir / 'config'),
                'backups_directory': str(self.base_dir / 'backups'),
                'temp_directory': str(self.base_dir / 'temp')
            },
            'security': {
                'sanitize_logs': True,
                'encrypt_sensitive_data': False,
                'access_control_enabled': False,
                'audit_trail_enabled': True
            },
            'monitoring': {
                'health_check_interval_minutes': 15,
                'alert_thresholds': {
                    'error_rate_percent': 5,
                    'disk_usage_percent': 85,
                    'memory_usage_percent': 80
                },
                'notification_channels': []
            },
            'integration': {
                'web_dashboard_enabled': False,
                'api_endpoints_enabled': False,
                'external_exports_enabled': True,
                'real_time_streaming': False
            }
        }
    
    def run_setup(self) -> bool:
        """Executar configuração completa."""
        print("="*60)
        print("CONFIGURAÇÃO DO SISTEMA DE LOGGING DETALHADO")
        print("Sistema de Trading ML - Versão 1.0.0")
        print("="*60)
        
        success = True
        
        # Verificar versão do Python
        if not self.check_python_version():
            success = False
        
        # Verificar dependências
        if success and not self.check_dependencies():
            success = False
        
        # Criar estrutura de diretórios
        if success and not self.create_directory_structure():
            success = False
        
        # Criar arquivos de configuração
        if success and not self.create_config_files():
            success = False
        
        # Verificar componentes de logging
        if success and not self.verify_logging_components():
            success = False
        
        # Executar testes de funcionalidade
        if success and not self.run_functionality_tests():
            success = False
        
        # Criar scripts de exemplo
        if success and not self.create_example_scripts():
            success = False
        
        # Gerar relatório
        report_path = self.generate_setup_report()
        
        print("\n" + "="*60)
        if success:
            self.log_step("CONFIGURAÇÃO CONCLUÍDA COM SUCESSO!")
            print(f"\nRelatório da configuração: {report_path}")
            print("\nPróximos passos:")
            print("1. Execute 'python examples/basic_logging_example.py' para testar")
            print("2. Execute 'python production_logging_demo.py' para demonstração completa")
            print("3. Revise as configurações em 'config/' conforme necessário")
        else:
            self.log_step("CONFIGURAÇÃO FALHOU!", False)
            print(f"\nErros encontrados: {len(self.errors)}")
            for error in self.errors:
                print(f"  - {error}")
            print(f"\nRelatório detalhado: {report_path}")
        
        print("="*60)
        
        return success


def main():
    """Função principal."""
    setup = LoggingSystemSetup()
    return setup.run_setup()


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[SETUP] Configuração interrompida pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Erro fatal durante a configuração: {e}")
        sys.exit(1)