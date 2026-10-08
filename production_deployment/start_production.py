#!/usr/bin/env python3
"""
Script de Inicialização do Sistema ML de Produção para Forex

Este script facilita o deployment e gerenciamento do pipeline completo de ML em produção.
Inclui verificações de saúde, inicialização de componentes e monitoramento.

Uso:
    python start_production.py [opções]
    
Exemplos:
    python start_production.py --config production_config.yaml
    python start_production.py --environment production --daemon
    python start_production.py --check-health
    python start_production.py --stop
"""

import argparse
import sys
import os
import json
import yaml
import time
import signal
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import psutil

# Adiciona o diretório atual ao path para imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_production_pipeline import MLProductionPipeline, PipelineConfig

class ProductionManager:
    """Gerenciador do sistema de produção ML"""
    
    def __init__(self):
        self.pipeline: Optional[MLProductionPipeline] = None
        self.config: Optional[PipelineConfig] = None
        self.pid_file = "ml_production.pid"
        self.log_file = "logs/production_manager.log"
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Configura logging para o gerenciador"""
        logger = logging.getLogger("ProductionManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Cria diretório de logs
            Path("logs").mkdir(exist_ok=True)
            
            # File handler
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def load_config(self, config_path: str, environment: str = None) -> PipelineConfig:
        """Carrega configuração do arquivo YAML"""
        try:
            self.logger.info(f"Carregando configuração de {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Aplica configurações específicas do ambiente
            if environment and environment in config_data.get('environments', {}):
                env_config = config_data['environments'][environment]
                config_data = self._merge_configs(config_data, env_config)
                self.logger.info(f"Configurações do ambiente '{environment}' aplicadas")
            
            # Converte para PipelineConfig
            pipeline_config = config_data.get('pipeline', {})
            retraining_config = config_data.get('retraining', {})
            quality_config = config_data.get('quality_monitoring', {})
            data_config = config_data.get('data', {})
            
            self.config = PipelineConfig(
                enable_scheduler=pipeline_config.get('components', {}).get('scheduler', True),
                enable_performance_monitor=pipeline_config.get('components', {}).get('performance_monitor', True),
                enable_hyperparameter_optimizer=pipeline_config.get('components', {}).get('hyperparameter_optimizer', True),
                enable_quality_monitor=pipeline_config.get('components', {}).get('quality_monitor', True),
                enable_memory_system=pipeline_config.get('components', {}).get('memory_system', True),
                enable_dashboard=pipeline_config.get('components', {}).get('dashboard', True),
                
                auto_retrain_enabled=retraining_config.get('auto_enabled', True),
                retrain_interval_hours=retraining_config.get('interval_hours', 24),
                performance_threshold=retraining_config.get('triggers', {}).get('performance_threshold', 0.6),
                quality_threshold=retraining_config.get('triggers', {}).get('quality_threshold', 0.65),
                
                optimize_on_retrain=config_data.get('hyperparameter_optimization', {}).get('optimize_on_retrain', True),
                optimization_method=config_data.get('hyperparameter_optimization', {}).get('default_method', 'optuna_tpe'),
                optimization_trials=config_data.get('hyperparameter_optimization', {}).get('methods', {}).get('optuna_tpe', {}).get('n_trials', 50),
                
                monitoring_interval_seconds=pipeline_config.get('monitoring', {}).get('interval_seconds', 30),
                alert_cooldown_minutes=pipeline_config.get('monitoring', {}).get('alert_cooldown_minutes', 60),
                dashboard_port=quality_config.get('dashboard', {}).get('port', 8050),
                
                memory_retention_days=config_data.get('memory_system', {}).get('learning', {}).get('retention_days', 90),
                learning_session_interval_hours=config_data.get('memory_system', {}).get('learning', {}).get('session_interval_hours', 12),
                
                currency_pairs=data_config.get('currency_pairs', {}).get('major', ['EURUSD', 'GBPUSD']),
                models_to_monitor=data_config.get('models', {}).get('primary', ['xgboost', 'random_forest'])
            )
            
            self.logger.info("Configuração carregada com sucesso")
            return self.config
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar configuração: {e}")
            raise
    
    def _merge_configs(self, base_config: Dict, env_config: Dict) -> Dict:
        """Mescla configurações base com configurações específicas do ambiente"""
        def merge_dict(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result
        
        return merge_dict(base_config, env_config)
    
    def check_dependencies(self) -> bool:
        """Verifica dependências do sistema"""
        try:
            self.logger.info("Verificando dependências do sistema...")
            
            # Verifica Python
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
                self.logger.error("Python 3.8+ é necessário")
                return False
            
            # Verifica pacotes essenciais
            required_packages = [
                'numpy', 'pandas', 'sklearn', 'xgboost', 
                'optuna', 'dash', 'plotly', 'sqlite3'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                self.logger.error(f"Pacotes faltando: {', '.join(missing_packages)}")
                return False
            
            # Verifica diretórios necessários
            required_dirs = ['logs', 'data', 'models', 'reports', 'backups']
            for dir_name in required_dirs:
                Path(dir_name).mkdir(exist_ok=True)
            
            # Verifica espaço em disco (mínimo 1GB)
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 1.0:
                self.logger.warning(f"Pouco espaço em disco: {free_gb:.2f}GB disponível")
            
            # Verifica memória (mínimo 2GB)
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            if available_gb < 2.0:
                self.logger.warning(f"Pouca memória disponível: {available_gb:.2f}GB")
            
            self.logger.info("Verificação de dependências concluída com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na verificação de dependências: {e}")
            return False
    
    def is_running(self) -> bool:
        """Verifica se o pipeline já está rodando"""
        try:
            if not Path(self.pid_file).exists():
                return False
            
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Verifica se o processo ainda existe
            return psutil.pid_exists(pid)
            
        except Exception:
            return False
    
    def start_pipeline(self, daemon: bool = False) -> bool:
        """Inicia o pipeline de produção"""
        try:
            if self.is_running():
                self.logger.warning("Pipeline já está rodando")
                return True
            
            if not self.config:
                self.logger.error("Configuração não carregada")
                return False
            
            self.logger.info("Iniciando pipeline de produção...")
            
            # Cria pipeline
            self.pipeline = MLProductionPipeline(self.config)
            
            # Configura handlers de sinal
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Inicia pipeline
            if not self.pipeline.start_pipeline():
                self.logger.error("Falha ao iniciar pipeline")
                return False
            
            # Salva PID
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            
            self.logger.info("Pipeline iniciado com sucesso")
            
            if self.config.enable_dashboard:
                self.logger.info(f"Dashboard disponível em: http://localhost:{self.config.dashboard_port}")
            
            if daemon:
                self._run_daemon()
            else:
                self._run_interactive()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar pipeline: {e}")
            return False
    
    def stop_pipeline(self) -> bool:
        """Para o pipeline de produção"""
        try:
            if not self.is_running():
                self.logger.info("Pipeline não está rodando")
                return True
            
            self.logger.info("Parando pipeline de produção...")
            
            if self.pipeline:
                self.pipeline.stop_pipeline()
            
            # Remove arquivo PID
            if Path(self.pid_file).exists():
                Path(self.pid_file).unlink()
            
            self.logger.info("Pipeline parado com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao parar pipeline: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Obtém status do sistema"""
        try:
            status = {
                "running": self.is_running(),
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "platform": sys.platform,
                    "cpu_count": psutil.cpu_count(),
                    "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "disk_free_gb": round(psutil.disk_usage('.').free / (1024**3), 2)
                }
            }
            
            if self.pipeline:
                pipeline_status = self.pipeline.get_pipeline_status()
                status["pipeline"] = pipeline_status
            
            return status
            
        except Exception as e:
            return {"error": str(e)}
    
    def check_health(self) -> bool:
        """Verifica saúde completa do sistema"""
        try:
            self.logger.info("Executando verificação de saúde...")
            
            health_checks = {
                "dependencies": self.check_dependencies(),
                "running": self.is_running(),
                "disk_space": psutil.disk_usage('.').free > 1024**3,  # > 1GB
                "memory": psutil.virtual_memory().available > 2 * 1024**3,  # > 2GB
                "cpu_usage": psutil.cpu_percent(interval=1) < 90  # < 90%
            }
            
            all_healthy = all(health_checks.values())
            
            self.logger.info(f"Verificação de saúde: {'[SAUDAVEL]' if all_healthy else '[PROBLEMAS DETECTADOS]'}")
            
            for check, result in health_checks.items():
                status = "[OK]" if result else "[FAIL]"
                self.logger.info(f"  {status} {check}")
            
            if self.pipeline:
                pipeline_status = self.pipeline.get_pipeline_status()
                component_health = pipeline_status.get('component_health', {})
                
                self.logger.info("Status dos componentes:")
                for component, health in component_health.items():
                    status = "✓" if health['status'] == 'active' else "✗"
                    self.logger.info(f"  {status} {component}: {health['status']}")
            
            return all_healthy
            
        except Exception as e:
            self.logger.error(f"Erro na verificação de saúde: {e}")
            return False
    
    def _signal_handler(self, signum, frame):
        """Handler para sinais do sistema"""
        self.logger.info(f"Sinal recebido: {signum}")
        self.stop_pipeline()
        sys.exit(0)
    
    def _run_daemon(self):
        """Executa em modo daemon"""
        self.logger.info("Executando em modo daemon")
        
        try:
            while True:
                # Verifica saúde periodicamente
                if not self.check_health():
                    self.logger.warning("Problemas de saúde detectados")
                
                # Aguarda próxima verificação
                time.sleep(300)  # 5 minutos
                
        except KeyboardInterrupt:
            self.logger.info("Interrupção recebida")
        except Exception as e:
            self.logger.error(f"Erro no modo daemon: {e}")
        finally:
            self.stop_pipeline()
    
    def _run_interactive(self):
        """Executa em modo interativo"""
        self.logger.info("Executando em modo interativo")
        self.logger.info("Pressione Ctrl+C para parar")
        
        try:
            while True:
                # Mostra status periodicamente
                status = self.get_status()
                if "pipeline" in status:
                    pipeline_status = status["pipeline"]
                    uptime = pipeline_status.get('uptime_seconds', 0)
                    metrics = pipeline_status.get('pipeline_metrics', {})
                    
                    self.logger.info(
                        f"Status: {pipeline_status['pipeline_status']} | "
                        f"Uptime: {uptime:.0f}s | "
                        f"Predições: {metrics.get('total_predictions', 0)} | "
                        f"Retreinamentos: {metrics.get('successful_retrains', 0)}"
                    )
                
                time.sleep(60)  # Atualiza a cada minuto
                
        except KeyboardInterrupt:
            self.logger.info("Interrupção recebida")
        except Exception as e:
            self.logger.error(f"Erro no modo interativo: {e}")
        finally:
            self.stop_pipeline()

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Sistema ML de Produção para Forex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python start_production.py --config production_config.yaml
  python start_production.py --environment production --daemon
  python start_production.py --check-health
  python start_production.py --stop
  python start_production.py --status
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='production_config.yaml',
        help='Arquivo de configuração YAML (padrão: production_config.yaml)'
    )
    
    parser.add_argument(
        '--environment', '-e',
        choices=['development', 'staging', 'production'],
        help='Ambiente de execução'
    )
    
    parser.add_argument(
        '--daemon', '-d',
        action='store_true',
        help='Executa em modo daemon (background)'
    )
    
    parser.add_argument(
        '--check-health',
        action='store_true',
        help='Verifica saúde do sistema e sai'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Mostra status do sistema e sai'
    )
    
    parser.add_argument(
        '--stop',
        action='store_true',
        help='Para o pipeline se estiver rodando'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Logging verboso'
    )
    
    args = parser.parse_args()
    
    # Configura logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Cria gerenciador
    manager = ProductionManager()
    
    try:
        # Executa ação solicitada
        if args.stop:
            success = manager.stop_pipeline()
            sys.exit(0 if success else 1)
        
        elif args.check_health:
            healthy = manager.check_health()
            sys.exit(0 if healthy else 1)
        
        elif args.status:
            status = manager.get_status()
            print(json.dumps(status, indent=2, default=str))
            sys.exit(0)
        
        else:
            # Carrega configuração
            if not Path(args.config).exists():
                print(f"Erro: Arquivo de configuração '{args.config}' não encontrado")
                sys.exit(1)
            
            manager.load_config(args.config, args.environment)
            
            # Verifica dependências
            if not manager.check_dependencies():
                print("Erro: Dependências não atendidas")
                sys.exit(1)
            
            # Inicia pipeline
            success = manager.start_pipeline(daemon=args.daemon)
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
        manager.stop_pipeline()
        sys.exit(0)
    
    except Exception as e:
        print(f"Erro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()