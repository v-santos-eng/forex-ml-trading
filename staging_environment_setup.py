#!/usr/bin/env python3
"""
Configuração do Ambiente de Staging para ML Forex System
Cria um ambiente isolado para testes de produção das 11 funcionalidades avançadas
"""

import os
import shutil
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys

class StagingEnvironmentSetup:
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.staging_path = self.base_path / "staging_environment"
        self.production_path = self.base_path / "ml_forex_production"
        self.backup_path = self.base_path / "production_backups"
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('staging_setup.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_staging_environment(self):
        """Cria ambiente de staging isolado"""
        try:
            self.logger.info("Iniciando criação do ambiente de staging...")
            
            # Criar diretórios
            self._create_directories()
            
            # Copiar arquivos essenciais
            self._copy_core_files()
            
            # Configurar banco de dados de staging
            self._setup_staging_databases()
            
            # Criar configurações de staging
            self._create_staging_configs()
            
            # Instalar dependências
            self._install_dependencies()
            
            # Configurar variáveis de ambiente
            self._setup_environment_variables()
            
            self.logger.info("Ambiente de staging criado com sucesso!")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao criar ambiente de staging: {e}")
            return False
    
    def _create_directories(self):
        """Cria estrutura de diretórios do staging"""
        directories = [
            self.staging_path,
            self.staging_path / "logs",
            self.staging_path / "models",
            self.staging_path / "data",
            self.staging_path / "config",
            self.staging_path / "tests",
            self.staging_path / "monitoring",
            self.staging_path / "backups"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Diretório criado: {directory}")
    
    def _copy_core_files(self):
        """Copia arquivos essenciais para staging"""
        core_files = [
            # Funcionalidades ML implementadas
            "triple_barrier_labeling.py",
            "purged_cross_validation.py",
            "kelly_position_sizing.py",
            "model_drift_detection.py",
            "dynamic_model_selection.py",
            "advanced_feature_engineering.py",
            "fractional_differentiation.py",
            "walk_forward_analysis.py",
            "risk_budgeting.py",
            "sentiment_analysis.py",
            "cross_asset_signals.py",
            
            # Arquivos de configuração
            "config.py",
            "config_manager.py",
            "security_manager.py",
            "logging_system.py",
            
            # Sistema principal
            "ml_forex_predictor.py",
            "auto_trader.py",
            "trading_monitor.py",
            
            # Testes
            "validate_implementations.py",
            "test_advanced_ml.py"
        ]
        
        for file_name in core_files:
            source = self.base_path / file_name
            if source.exists():
                destination = self.staging_path / file_name
                shutil.copy2(source, destination)
                self.logger.info(f"Arquivo copiado: {file_name}")
            else:
                self.logger.warning(f"Arquivo não encontrado: {file_name}")
    
    def _setup_staging_databases(self):
        """Configura bancos de dados para staging"""
        databases = [
            "auto_trader_staging.db",
            "forex_memory_staging.db",
            "performance_metrics_staging.db",
            "trading_monitor_staging.db",
            "cross_asset_data_staging.db",
            "sentiment_data_staging.db"
        ]
        
        for db_name in databases:
            db_path = self.staging_path / "data" / db_name
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE IF NOT EXISTS staging_test (id INTEGER PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            conn.commit()
            conn.close()
            self.logger.info(f"Banco de dados criado: {db_name}")
    
    def _create_staging_configs(self):
        """Cria configurações específicas para staging"""
        staging_config = {
            "environment": "staging",
            "debug_mode": True,
            "log_level": "DEBUG",
            "database_prefix": "staging_",
            "mt5_demo_account": True,
            "risk_limits": {
                "max_position_size": 0.01,  # Tamanho mínimo para testes
                "max_daily_trades": 5,
                "max_drawdown": 0.02
            },
            "ml_features": {
                "triple_barrier_labeling": True,
                "purged_cross_validation": True,
                "kelly_position_sizing": True,
                "model_drift_detection": True,
                "dynamic_model_selection": True,
                "advanced_feature_engineering": True,
                "fractional_differentiation": True,
                "walk_forward_analysis": True,
                "risk_budgeting": True,
                "sentiment_analysis": True,
                "cross_asset_signals": True
            },
            "monitoring": {
                "performance_tracking": True,
                "error_alerting": True,
                "resource_monitoring": True
            }
        }
        
        config_path = self.staging_path / "config" / "staging_config.json"
        with open(config_path, 'w') as f:
            json.dump(staging_config, f, indent=2)
        
        self.logger.info("Configuração de staging criada")
    
    def _install_dependencies(self):
        """Instala dependências no ambiente de staging"""
        try:
            # Criar requirements específico para staging
            staging_requirements = [
                "numpy>=1.21.0",
                "pandas>=1.3.0",
                "scikit-learn>=1.0.0",
                "xgboost>=1.5.0",
                "lightgbm>=3.3.0",
                "optuna>=3.0.0",
                "MetaTrader5>=5.0.45",
                "psutil>=5.8.0",
                "pytest>=7.0.0",
                "pytest-cov>=4.0.0",
                "memory-profiler>=0.60.0",
                "line-profiler>=4.0.0"
            ]
            
            requirements_path = self.staging_path / "requirements_staging.txt"
            with open(requirements_path, 'w') as f:
                f.write('\n'.join(staging_requirements))
            
            self.logger.info("Requirements de staging criado")
            
        except Exception as e:
            self.logger.error(f"Erro ao configurar dependências: {e}")
    
    def _setup_environment_variables(self):
        """Configura variáveis de ambiente para staging"""
        env_vars = {
            "ML_FOREX_ENV": "staging",
            "ML_FOREX_DEBUG": "true",
            "ML_FOREX_LOG_LEVEL": "DEBUG",
            "ML_FOREX_DB_PREFIX": "staging_",
            "ML_FOREX_RISK_MODE": "conservative"
        }
        
        env_file = self.staging_path / ".env_staging"
        with open(env_file, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        self.logger.info("Variáveis de ambiente configuradas")
    
    def run_staging_tests(self):
        """Executa testes no ambiente de staging"""
        try:
            self.logger.info("Iniciando testes no ambiente de staging...")
            
            # Mudar para diretório de staging
            original_cwd = os.getcwd()
            os.chdir(self.staging_path)
            
            # Executar testes de validação
            result = subprocess.run(
                [sys.executable, "validate_implementations.py"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info("Todos os testes passaram no ambiente de staging")
                return True
            else:
                self.logger.error(f"Falhas nos testes: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro ao executar testes: {e}")
            return False
        finally:
            os.chdir(original_cwd)
    
    def create_deployment_package(self):
        """Cria pacote para deployment em produção"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            package_name = f"ml_forex_deployment_{timestamp}"
            package_path = self.staging_path / "packages" / package_name
            
            package_path.mkdir(parents=True, exist_ok=True)
            
            # Copiar arquivos validados
            validated_files = [
                "triple_barrier_labeling.py",
                "purged_cross_validation.py",
                "kelly_position_sizing.py",
                "model_drift_detection.py",
                "dynamic_model_selection.py",
                "advanced_feature_engineering.py",
                "fractional_differentiation.py",
                "walk_forward_analysis.py",
                "risk_budgeting.py",
                "sentiment_analysis.py",
                "cross_asset_signals.py"
            ]
            
            for file_name in validated_files:
                source = self.staging_path / file_name
                if source.exists():
                    shutil.copy2(source, package_path / file_name)
            
            # Criar manifesto do pacote
            manifest = {
                "package_name": package_name,
                "created_at": timestamp,
                "files": validated_files,
                "version": "1.0.0",
                "tested": True,
                "environment": "staging"
            }
            
            with open(package_path / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            self.logger.info(f"Pacote de deployment criado: {package_name}")
            return package_path
            
        except Exception as e:
            self.logger.error(f"Erro ao criar pacote: {e}")
            return None

def main():
    """Função principal para configurar ambiente de staging"""
    setup = StagingEnvironmentSetup()
    
    print("=== Configuração do Ambiente de Staging ML Forex ===")
    print("1. Criando ambiente de staging...")
    
    if setup.create_staging_environment():
        print("✓ Ambiente de staging criado com sucesso")
        
        print("\n2. Executando testes de validação...")
        if setup.run_staging_tests():
            print("✓ Todos os testes passaram")
            
            print("\n3. Criando pacote de deployment...")
            package_path = setup.create_deployment_package()
            if package_path:
                print(f"✓ Pacote criado: {package_path}")
                print("\n🎉 Ambiente de staging configurado e validado!")
                print("\nPróximos passos:")
                print("- Executar testes de performance")
                print("- Configurar monitoramento")
                print("- Preparar deployment em produção")
            else:
                print("✗ Erro ao criar pacote de deployment")
        else:
            print("✗ Falhas nos testes - revisar antes de prosseguir")
    else:
        print("✗ Erro ao criar ambiente de staging")

if __name__ == "__main__":
    main()