#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Recuperação Automática para Componentes MT5
Implementa recuperação inteligente para dashboard MT5 e componentes críticos
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

# Importar sistema de logging de decisões
try:
    from decision_logging_system import DecisionLogger, DecisionType, CriticalityLevel
    DECISION_LOGGING_AVAILABLE = True
except ImportError:
    DECISION_LOGGING_AVAILABLE = False
    print("Sistema de logging de decisões não disponível")

# Importar autenticação MT5
try:
    from mt5_secure_auth import MT5Authenticator
    MT5_AUTH_AVAILABLE = True
except ImportError:
    MT5_AUTH_AVAILABLE = False
    print("Autenticação MT5 não disponível")

class RecoveryStrategy(Enum):
    """Estratégias de recuperação"""
    IMMEDIATE_RESTART = "immediate_restart"
    DELAYED_RESTART = "delayed_restart"
    GRACEFUL_RESTART = "graceful_restart"
    FAILOVER = "failover"
    ROLLBACK = "rollback"
    MAINTENANCE_MODE = "maintenance_mode"
    CUSTOM_RECOVERY = "custom_recovery"

class ComponentType(Enum):
    """Tipos de componentes"""
    MT5_DASHBOARD = "mt5_dashboard"
    ML_PIPELINE = "ml_pipeline"
    DATA_COLLECTOR = "data_collector"
    API_SERVER = "api_server"
    NOTIFICATION_SERVICE = "notification_service"
    MONITORING_SERVICE = "monitoring_service"
    DATABASE_SERVICE = "database_service"

class HealthStatus(Enum):
    """Status de saúde dos componentes"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"

@dataclass
class RecoveryMetrics:
    """Métricas de recuperação"""
    total_recoveries: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    average_recovery_time: float = 0.0
    last_recovery_time: Optional[datetime] = None
    recovery_success_rate: float = 0.0

@dataclass
class ComponentHealth:
    """Saúde do componente"""
    status: HealthStatus
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time: float = 0.0
    error_rate: float = 0.0
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0

@dataclass
class RecoveryConfig:
    """Configuração de recuperação para um componente"""
    name: str
    component_type: ComponentType
    command: str
    working_dir: str
    strategy: RecoveryStrategy = RecoveryStrategy.GRACEFUL_RESTART
    max_recovery_attempts: int = 3
    recovery_delay: int = 30
    health_check_interval: int = 60
    health_thresholds: Dict[str, float] = None
    dependencies: List[str] = None
    pre_recovery_actions: List[Callable] = None
    post_recovery_actions: List[Callable] = None
    custom_recovery_function: Optional[Callable] = None
    critical: bool = True
    auto_recovery_enabled: bool = True
    
    def __post_init__(self):
        if self.health_thresholds is None:
            self.health_thresholds = {
                'cpu_threshold': 80.0,
                'memory_threshold': 85.0,
                'response_time_threshold': 5.0,
                'error_rate_threshold': 0.1
            }
        if self.dependencies is None:
            self.dependencies = []
        if self.pre_recovery_actions is None:
            self.pre_recovery_actions = []
        if self.post_recovery_actions is None:
            self.post_recovery_actions = []

class MT5RecoverySystem:
    """Sistema de recuperação automática para componentes MT5"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.components: Dict[str, RecoveryConfig] = {}
        self.health_status: Dict[str, ComponentHealth] = {}
        self.recovery_metrics: Dict[str, RecoveryMetrics] = {}
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.recovery_threads: Dict[str, threading.Thread] = {}
        self.monitoring_active = False
        self.shutdown_event = threading.Event()
        
        # Setup logging
        self.setup_logging()
        
        # Setup decision logger se disponível
        if DECISION_LOGGING_AVAILABLE:
            self.decision_logger = DecisionLogger("MT5RecoverySystem")
        else:
            self.decision_logger = None
        
        # Setup MT5 authenticator se disponível
        if MT5_AUTH_AVAILABLE:
            self.mt5_auth = MT5Authenticator()
        else:
            self.mt5_auth = None
        
        # Carregar configuração
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        else:
            self.setup_default_components()
        
        # Criar diretórios necessários
        self.setup_directories()
    
    def setup_logging(self):
        """Configura sistema de logging"""
        self.logger = logging.getLogger('MT5RecoverySystem')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        Path("logs").mkdir(exist_ok=True)
        file_handler = logging.FileHandler('logs/mt5_recovery.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def load_config(self, config_file: str):
        """Carrega configuração de arquivo JSON"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.logger.info(f"Carregando configuração de: {config_file}")
            
            # Carregar componentes
            components_config = config_data.get('components', {})
            for name, comp_config in components_config.items():
                try:
                    # Converter string para enum
                    component_type = ComponentType(comp_config['component_type'])
                    strategy = RecoveryStrategy(comp_config.get('strategy', 'GRACEFUL_RESTART'))
                    
                    # Criar configuração do componente
                    recovery_config = RecoveryConfig(
                        name=name,
                        component_type=component_type,
                        command=comp_config['command'],
                        working_dir=comp_config.get('working_dir', '.'),
                        strategy=strategy,
                        max_recovery_attempts=comp_config.get('max_recovery_attempts', 3),
                        recovery_delay=comp_config.get('recovery_delay', 30),
                        health_check_interval=comp_config.get('health_check_interval', 60),
                        health_thresholds=comp_config.get('health_thresholds', {}),
                        dependencies=comp_config.get('dependencies', []),
                        critical=comp_config.get('critical', True),
                        auto_recovery_enabled=comp_config.get('auto_recovery_enabled', True)
                    )
                    
                    # Adicionar componente
                    self.components[name] = recovery_config
                    self.health_status[name] = ComponentHealth(status=HealthStatus.UNKNOWN)
                    self.recovery_metrics[name] = RecoveryMetrics()
                    
                    self.logger.info(f"Componente carregado: {name} ({component_type.value})")
                    
                except Exception as e:
                    self.logger.error(f"Erro ao carregar componente {name}: {e}")
            
            self.logger.info(f"Configuração carregada: {len(self.components)} componentes")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar configuração de {config_file}: {e}")
            self.logger.info("Usando configuração padrão")
            self.setup_default_components()
    
    def setup_directories(self):
        """Cria diretórios necessários"""
        directories = [
            "logs", "recovery", "backups", "health_reports", "recovery_reports"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def setup_default_components(self):
        """Configura componentes padrão"""
        # Dashboard MT5
        self.add_component(
            name="mt5_dashboard",
            component_type=ComponentType.MT5_DASHBOARD,
            command="python mt5_dashboard_simple_secure.py",
            working_dir=".",
            strategy=RecoveryStrategy.GRACEFUL_RESTART,
            max_recovery_attempts=5,
            recovery_delay=15,
            health_check_interval=30,
            critical=True
        )
        
        # Pipeline ML
        self.add_component(
            name="ml_pipeline",
            component_type=ComponentType.ML_PIPELINE,
            command="python ../ml_production_pipeline.py",
            working_dir=".",
            strategy=RecoveryStrategy.DELAYED_RESTART,
            max_recovery_attempts=3,
            recovery_delay=60,
            health_check_interval=120,
            critical=True
        )
        
        # Monitor de produção
        self.add_component(
            name="production_monitor",
            component_type=ComponentType.MONITORING_SERVICE,
            command="python production_monitor.py --daemon --interval 30",
            working_dir=".",
            strategy=RecoveryStrategy.IMMEDIATE_RESTART,
            max_recovery_attempts=5,
            recovery_delay=10,
            health_check_interval=60,
            critical=True
        )
    
    def add_component(self, name: str, component_type: ComponentType, 
                     command: str, working_dir: str, **kwargs):
        """Adiciona componente ao sistema de recuperação"""
        config = RecoveryConfig(
            name=name,
            component_type=component_type,
            command=command,
            working_dir=working_dir,
            **kwargs
        )
        
        self.components[name] = config
        self.health_status[name] = ComponentHealth(status=HealthStatus.UNKNOWN)
        self.recovery_metrics[name] = RecoveryMetrics()
        
        self.logger.info(f"Componente adicionado: {name} ({component_type.value})")
        
        # Log da decisão
        if self.decision_logger:
            self.decision_logger.log_decision(
                decision_type=DecisionType.SYSTEM_CONFIGURATION,
                description=f"Componente {name} adicionado ao sistema de recuperação",
                criteria={"component_type": component_type.value, "critical": config.critical},
                context={"command": command, "working_dir": working_dir},
                criticality=CriticalityLevel.MEDIUM
            )
    
    def register_existing_process(self, component_name: str, pid: int):
        """Registra um processo existente no sistema de recuperação"""
        try:
            import psutil
            
            if component_name not in self.components:
                self.logger.warning(f"Componente {component_name} não encontrado na configuração")
                return False
            
            # Verificar se processo existe e está ativo
            if not psutil.pid_exists(pid):
                self.logger.error(f"PID {pid} não existe")
                return False
            
            process = psutil.Process(pid)
            if not process.is_running():
                self.logger.error(f"Processo {pid} não está em execução")
                return False
            
            # Atualizar status do componente
            if component_name in self.health_status:
                self.health_status[component_name].status = HealthStatus.HEALTHY
                self.health_status[component_name].last_check = datetime.now()
                self.health_status[component_name].process_info = {
                    'pid': pid,
                    'name': process.name(),
                    'status': process.status(),
                    'cpu_percent': process.cpu_percent(),
                    'memory_percent': process.memory_percent()
                }
            
            # Salvar PID em arquivo
            pid_file = f"{component_name}.pid"
            with open(pid_file, 'w') as f:
                f.write(str(pid))
            
            self.logger.info(f"Processo existente registrado: {component_name} (PID: {pid})")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao registrar processo existente {component_name}: {e}")
            return False
    
    def check_component_health(self, name: str) -> ComponentHealth:
        """Verifica saúde de um componente"""
        if name not in self.components:
            return ComponentHealth(status=HealthStatus.UNKNOWN)
        
        config = self.components[name]
        health = self.health_status[name]
        
        try:
            # Verificar se processo está rodando
            if name in self.active_processes:
                process = self.active_processes[name]
                
                if process.poll() is None:  # Processo ainda rodando
                    # Obter métricas do processo
                    try:
                        ps_process = psutil.Process(process.pid)
                        health.cpu_usage = ps_process.cpu_percent()
                        health.memory_usage = ps_process.memory_percent()
                        
                        # Verificar thresholds
                        thresholds = config.health_thresholds
                        
                        if (health.cpu_usage > thresholds['cpu_threshold'] or 
                            health.memory_usage > thresholds['memory_threshold']):
                            health.status = HealthStatus.DEGRADED
                        else:
                            health.status = HealthStatus.HEALTHY
                            health.consecutive_failures = 0
                        
                    except psutil.NoSuchProcess:
                        health.status = HealthStatus.FAILED
                        health.consecutive_failures += 1
                else:
                    health.status = HealthStatus.FAILED
                    health.consecutive_failures += 1
            else:
                health.status = HealthStatus.FAILED
                health.consecutive_failures += 1
            
            health.last_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Erro ao verificar saúde do componente {name}: {e}")
            health.status = HealthStatus.UNKNOWN
            health.consecutive_failures += 1
        
        return health
    
    def start_component(self, name: str) -> bool:
        """Inicia um componente"""
        if name not in self.components:
            self.logger.error(f"Componente {name} não encontrado")
            return False
        
        config = self.components[name]
        
        try:
            self.logger.info(f"Iniciando componente: {name}")
            
            # Executar ações pré-recuperação
            for action in config.pre_recovery_actions:
                try:
                    action(name, config)
                except Exception as e:
                    self.logger.warning(f"Erro em ação pré-recuperação para {name}: {e}")
            
            # Preparar ambiente
            env = os.environ.copy()
            
            # Configurar logs
            log_file = f"logs/{name}.log"
            
            with open(log_file, 'a') as log:
                process = subprocess.Popen(
                    config.command,
                    shell=True,
                    cwd=config.working_dir,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT
                )
            
            self.active_processes[name] = process
            
            # Aguardar um pouco para verificar se iniciou corretamente
            time.sleep(2)
            
            if process.poll() is None:
                self.logger.info(f"Componente {name} iniciado com PID {process.pid}")
                
                # Executar ações pós-recuperação
                for action in config.post_recovery_actions:
                    try:
                        action(name, config)
                    except Exception as e:
                        self.logger.warning(f"Erro em ação pós-recuperação para {name}: {e}")
                
                # Log da decisão
                if self.decision_logger:
                    self.decision_logger.log_decision(
                        decision_type=DecisionType.SYSTEM_RECOVERY,
                        description=f"Componente {name} iniciado com sucesso",
                        criteria={"pid": process.pid, "strategy": config.strategy.value},
                        context={"command": config.command},
                        criticality=CriticalityLevel.HIGH if config.critical else CriticalityLevel.MEDIUM
                    )
                
                return True
            else:
                self.logger.error(f"Componente {name} falhou ao iniciar")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro ao iniciar componente {name}: {e}")
            return False
    
    def stop_component(self, name: str, graceful: bool = True) -> bool:
        """Para um componente"""
        if name not in self.active_processes:
            self.logger.warning(f"Componente {name} não está rodando")
            return True
        
        try:
            process = self.active_processes[name]
            
            if graceful:
                self.logger.info(f"Parando componente graciosamente: {name}")
                process.terminate()
                
                # Aguardar término gracioso
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Forçando término do componente {name}")
                    process.kill()
            else:
                self.logger.info(f"Forçando parada do componente: {name}")
                process.kill()
            
            del self.active_processes[name]
            self.logger.info(f"Componente {name} parado")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao parar componente {name}: {e}")
            return False
    
    def recover_component(self, name: str) -> bool:
        """Executa recuperação de um componente"""
        if name not in self.components:
            self.logger.error(f"Componente {name} não encontrado")
            return False
        
        config = self.components[name]
        metrics = self.recovery_metrics[name]
        
        # Verificar se já excedeu tentativas máximas
        if metrics.total_recoveries >= config.max_recovery_attempts:
            self.logger.error(f"Máximo de tentativas de recuperação excedido para {name}")
            return False
        
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Iniciando recuperação do componente: {name}")
            
            # Log da decisão de recuperação
            if self.decision_logger:
                self.decision_logger.log_decision(
                    decision_type=DecisionType.SYSTEM_RECOVERY,
                    description=f"Iniciando recuperação do componente {name}",
                    criteria={
                        "strategy": config.strategy.value,
                        "attempt": metrics.total_recoveries + 1,
                        "max_attempts": config.max_recovery_attempts
                    },
                    context={"health_status": self.health_status[name].status.value},
                    criticality=CriticalityLevel.HIGH if config.critical else CriticalityLevel.MEDIUM
                )
            
            # Parar componente se estiver rodando
            if name in self.active_processes:
                self.stop_component(name, graceful=(config.strategy == RecoveryStrategy.GRACEFUL_RESTART))
            
            # Aguardar delay de recuperação
            if config.recovery_delay > 0:
                self.logger.info(f"Aguardando {config.recovery_delay}s antes da recuperação")
                time.sleep(config.recovery_delay)
            
            # Executar recuperação baseada na estratégia
            success = False
            
            if config.strategy == RecoveryStrategy.CUSTOM_RECOVERY and config.custom_recovery_function:
                success = config.custom_recovery_function(name, config)
            else:
                success = self.start_component(name)
            
            # Atualizar métricas
            metrics.total_recoveries += 1
            recovery_time = (datetime.now() - start_time).total_seconds()
            
            if success:
                metrics.successful_recoveries += 1
                metrics.last_recovery_time = datetime.now()
                
                # Calcular tempo médio de recuperação
                if metrics.successful_recoveries > 0:
                    metrics.average_recovery_time = (
                        (metrics.average_recovery_time * (metrics.successful_recoveries - 1) + recovery_time) /
                        metrics.successful_recoveries
                    )
                
                self.logger.info(f"Recuperação bem-sucedida para {name} em {recovery_time:.2f}s")
            else:
                metrics.failed_recoveries += 1
                self.logger.error(f"Falha na recuperação de {name}")
            
            # Calcular taxa de sucesso
            metrics.recovery_success_rate = (
                metrics.successful_recoveries / metrics.total_recoveries * 100
                if metrics.total_recoveries > 0 else 0
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Erro durante recuperação de {name}: {e}")
            metrics.failed_recoveries += 1
            return False
    
    def start_monitoring(self):
        """Inicia monitoramento contínuo"""
        if self.monitoring_active:
            self.logger.warning("Monitoramento já está ativo")
            return
        
        self.monitoring_active = True
        self.logger.info("Iniciando monitoramento de componentes")
        
        def monitor_loop():
            while self.monitoring_active and not self.shutdown_event.is_set():
                try:
                    for name in self.components:
                        if not self.monitoring_active:
                            break
                        
                        config = self.components[name]
                        
                        if not config.auto_recovery_enabled:
                            continue
                        
                        # Verificar saúde
                        health = self.check_component_health(name)
                        
                        # Decidir se precisa de recuperação
                        needs_recovery = (
                            health.status in [HealthStatus.FAILED, HealthStatus.CRITICAL] or
                            health.consecutive_failures >= 3
                        )
                        
                        if needs_recovery:
                            self.logger.warning(f"Componente {name} precisa de recuperação: {health.status.value}")
                            
                            # Executar recuperação em thread separada
                            if name not in self.recovery_threads or not self.recovery_threads[name].is_alive():
                                recovery_thread = threading.Thread(
                                    target=self.recover_component,
                                    args=(name,),
                                    name=f"Recovery-{name}"
                                )
                                recovery_thread.start()
                                self.recovery_threads[name] = recovery_thread
                    
                    # Aguardar próximo ciclo
                    self.shutdown_event.wait(30)  # Verificar a cada 30 segundos
                    
                except Exception as e:
                    self.logger.error(f"Erro no loop de monitoramento: {e}")
                    time.sleep(10)
        
        # Iniciar thread de monitoramento
        monitor_thread = threading.Thread(target=monitor_loop, name="MonitoringLoop")
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Para monitoramento"""
        self.monitoring_active = False
        self.shutdown_event.set()
        self.logger.info("Monitoramento parado")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtém status geral do sistema"""
        healthy_components = [name for name, health in self.health_status.items() 
                            if health.status == HealthStatus.HEALTHY]
        failed_components = [name for name, health in self.health_status.items() 
                           if health.status == HealthStatus.FAILED]
        unknown_components = [name for name, health in self.health_status.items() 
                            if health.status == HealthStatus.UNKNOWN]
        
        total_recoveries = sum(metrics.total_recoveries for metrics in self.recovery_metrics.values())
        successful_recoveries = sum(metrics.successful_recoveries for metrics in self.recovery_metrics.values())
        
        return {
            'total_components': len(self.components),
            'active_components': len(healthy_components),
            'healthy_components': healthy_components,
            'failed_components': failed_components,
            'unknown_components': unknown_components,
            'total_recoveries': total_recoveries,
            'successful_recoveries': successful_recoveries,
            'system_healthy': len(failed_components) == 0,
            'summary': f'Componentes: {len(self.components)} | Saudáveis: {len(healthy_components)} | Falhados: {len(failed_components)}'
        }
    
    def generate_health_report(self) -> str:
        """Gera relatório de saúde do sistema"""
        status = self.get_system_status()
        
        report = f"""
=== RELATÓRIO DE SAÚDE DO SISTEMA MT5 ===
Timestamp: {datetime.now().isoformat()}
Monitoramento Ativo: {self.monitoring_active}

RESUMO:
- Total de Componentes: {status['total_components']}
- Componentes Saudáveis: {len(status['healthy_components'])}
- Componentes Falhados: {len(status['failed_components'])}
- Componentes Desconhecidos: {len(status['unknown_components'])}
- Total de Recuperações: {status['total_recoveries']}
- Recuperações Bem-sucedidas: {status['successful_recoveries']}
- Sistema Saudável: {'Sim' if status['system_healthy'] else 'Não'}

DETALHES DOS COMPONENTES:
"""
        
        for name, config in self.components.items():
            health = self.health_status[name]
            metrics = self.recovery_metrics[name]
            
            report += f"""
{name.upper()} ({config.component_type.value}):
  Status: {health.status.value} {'(CRÍTICO)' if config.critical else ''}
  CPU: {health.cpu_usage:.1f}% | Memória: {health.memory_usage:.1f}%
  Falhas Consecutivas: {health.consecutive_failures}
  Recuperações: {metrics.successful_recoveries}/{metrics.total_recoveries} ({metrics.recovery_success_rate:.1f}%)
  Tempo Médio de Recuperação: {metrics.average_recovery_time:.2f}s
  Rodando: {'Sim' if name in self.active_processes and self.active_processes[name].poll() is None else 'Não'}
"""
        
        return report
    
    def shutdown(self):
        """Desliga sistema de recuperação"""
        self.logger.info("Iniciando shutdown do sistema de recuperação")
        
        # Parar monitoramento
        self.stop_monitoring()
        
        # Parar todos os componentes
        for name in list(self.active_processes.keys()):
            self.stop_component(name)
        
        # Aguardar threads de recuperação
        for thread in self.recovery_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.logger.info("Sistema de recuperação desligado")

def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema de Recuperação MT5")
    parser.add_argument("--config", help="Arquivo de configuração")
    parser.add_argument("--start", help="Iniciar componente específico")
    parser.add_argument("--stop", help="Parar componente específico")
    parser.add_argument("--recover", help="Recuperar componente específico")
    parser.add_argument("--status", action="store_true", help="Mostrar status do sistema")
    parser.add_argument("--report", action="store_true", help="Gerar relatório de saúde")
    parser.add_argument("--monitor", action="store_true", help="Iniciar monitoramento contínuo")
    
    args = parser.parse_args()
    
    # Criar sistema de recuperação
    recovery_system = MT5RecoverySystem(args.config)
    
    try:
        if args.start:
            success = recovery_system.start_component(args.start)
            print(f"Componente {args.start}: {'Iniciado' if success else 'Falha ao iniciar'}")
        
        elif args.stop:
            success = recovery_system.stop_component(args.stop)
            print(f"Componente {args.stop}: {'Parado' if success else 'Falha ao parar'}")
        
        elif args.recover:
            success = recovery_system.recover_component(args.recover)
            print(f"Recuperação {args.recover}: {'Sucesso' if success else 'Falha'}")
        
        elif args.status:
            status = recovery_system.get_system_status()
            print(json.dumps(status, indent=2, default=str))
        
        elif args.report:
            report = recovery_system.generate_health_report()
            print(report)
            
            # Salvar relatório
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"health_reports/health_report_{timestamp}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"\nRelatório salvo em: {report_file}")
        
        elif args.monitor:
            print("Iniciando monitoramento contínuo...")
            recovery_system.start_monitoring()
            
            try:
                while True:
                    time.sleep(60)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Sistema ativo - Componentes monitorados: {len(recovery_system.components)}")
            except KeyboardInterrupt:
                print("\nParando monitoramento...")
        
        else:
            print("Sistema de Recuperação MT5 inicializado")
            print("Use --help para ver opções disponíveis")
    
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    
    finally:
        recovery_system.shutdown()

if __name__ == "__main__":
    main()