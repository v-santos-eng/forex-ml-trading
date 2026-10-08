#!/usr/bin/env python3
"""
Sistema de Recuperação Automática para Componentes Críticos
Implementa mecanismos de restart, failover e recuperação de falhas
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import psutil
from dataclasses import dataclass, asdict
from enum import Enum

class ComponentStatus(Enum):
    """Status dos componentes"""
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"

class RecoveryAction(Enum):
    """Ações de recuperação"""
    RESTART = "restart"
    FAILOVER = "failover"
    ROLLBACK = "rollback"
    ALERT_ONLY = "alert_only"
    CUSTOM = "custom"

@dataclass
class ComponentConfig:
    """Configuração de um componente"""
    name: str
    command: str
    working_dir: str
    critical: bool = True
    max_restarts: int = 3
    restart_delay: int = 30  # segundos
    health_check_interval: int = 60  # segundos
    health_check_timeout: int = 30  # segundos
    health_check_command: Optional[str] = None
    health_check_url: Optional[str] = None
    recovery_action: RecoveryAction = RecoveryAction.RESTART
    dependencies: List[str] = None
    environment: Dict[str, str] = None
    log_file: Optional[str] = None
    pid_file: Optional[str] = None
    backup_command: Optional[str] = None
    custom_recovery_function: Optional[Callable] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.environment is None:
            self.environment = {}

@dataclass
class ComponentState:
    """Estado atual de um componente"""
    status: ComponentStatus
    pid: Optional[int] = None
    last_start_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    restart_count: int = 0
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    process: Optional[subprocess.Popen] = None
    health_status: bool = True
    recovery_in_progress: bool = False

class AutoRecoverySystem:
    """Sistema de recuperação automática"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.components: Dict[str, ComponentConfig] = {}
        self.states: Dict[str, ComponentState] = {}
        self.recovery_threads: Dict[str, threading.Thread] = {}
        self.monitoring_active = False
        self.shutdown_event = threading.Event()
        
        # Setup logging
        self.setup_logging()
        
        # Carregar configuração
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        
        # Criar diretórios necessários
        Path("logs").mkdir(exist_ok=True)
        Path("recovery").mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Configura sistema de logging"""
        self.logger = logging.getLogger('AutoRecoverySystem')
        self.logger.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('logs/auto_recovery.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Recovery logger
        self.recovery_logger = logging.getLogger('RecoveryLogger')
        self.recovery_logger.setLevel(logging.INFO)
        recovery_handler = logging.FileHandler('logs/recovery_actions.log')
        recovery_handler.setFormatter(formatter)
        self.recovery_logger.addHandler(recovery_handler)
    
    def load_config(self, config_file: str):
        """Carrega configuração de arquivo JSON"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            for comp_name, comp_config in config_data.get('components', {}).items():
                self.add_component(
                    name=comp_name,
                    command=comp_config['command'],
                    working_dir=comp_config.get('working_dir', '.'),
                    critical=comp_config.get('critical', True),
                    max_restarts=comp_config.get('max_restarts', 3),
                    restart_delay=comp_config.get('restart_delay', 30),
                    health_check_interval=comp_config.get('health_check_interval', 60),
                    health_check_command=comp_config.get('health_check_command'),
                    health_check_url=comp_config.get('health_check_url'),
                    recovery_action=RecoveryAction(comp_config.get('recovery_action', 'restart')),
                    dependencies=comp_config.get('dependencies', []),
                    environment=comp_config.get('environment', {}),
                    log_file=comp_config.get('log_file'),
                    backup_command=comp_config.get('backup_command')
                )
            
            self.logger.info(f"Configuração carregada: {len(self.components)} componentes")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar configuração: {e}")
    
    def add_component(self, name: str, command: str, working_dir: str = ".", **kwargs):
        """Adiciona componente ao sistema"""
        config = ComponentConfig(
            name=name,
            command=command,
            working_dir=working_dir,
            **kwargs
        )
        
        self.components[name] = config
        self.states[name] = ComponentState(status=ComponentStatus.STOPPED)
        
        self.logger.info(f"Componente adicionado: {name}")
    
    def start_component(self, name: str) -> bool:
        """Inicia um componente"""
        if name not in self.components:
            self.logger.error(f"Componente {name} não encontrado")
            return False
        
        config = self.components[name]
        state = self.states[name]
        
        # Verificar dependências
        for dep in config.dependencies:
            if dep in self.states and self.states[dep].status != ComponentStatus.RUNNING:
                self.logger.warning(f"Dependência {dep} não está rodando para {name}")
                return False
        
        try:
            # Preparar ambiente
            env = os.environ.copy()
            env.update(config.environment)
            
            # Iniciar processo
            self.logger.info(f"Iniciando componente: {name}")
            
            # Configurar redirecionamento de logs
            stdout = None
            stderr = None
            
            if config.log_file:
                log_path = Path(config.log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                stdout = open(log_path, 'a')
                stderr = subprocess.STDOUT
            
            process = subprocess.Popen(
                config.command,
                shell=True,
                cwd=config.working_dir,
                env=env,
                stdout=stdout,
                stderr=stderr
            )
            
            # Atualizar estado
            state.process = process
            state.pid = process.pid
            state.status = ComponentStatus.RUNNING
            state.last_start_time = datetime.now()
            state.restart_count += 1
            
            # Salvar PID se configurado
            if config.pid_file:
                with open(config.pid_file, 'w') as f:
                    f.write(str(process.pid))
            
            self.logger.info(f"Componente {name} iniciado com PID {process.pid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar componente {name}: {e}")
            state.status = ComponentStatus.FAILED
            state.failure_count += 1
            state.last_failure_time = datetime.now()
            return False
    
    def stop_component(self, name: str, force: bool = False) -> bool:
        """Para um componente"""
        if name not in self.components:
            self.logger.error(f"Componente {name} não encontrado")
            return False
        
        state = self.states[name]
        
        if state.status != ComponentStatus.RUNNING or not state.process:
            self.logger.warning(f"Componente {name} não está rodando")
            return True
        
        try:
            self.logger.info(f"Parando componente: {name}")
            
            if force:
                state.process.kill()
            else:
                state.process.terminate()
                
                # Aguardar término gracioso
                try:
                    state.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Forçando término do componente {name}")
                    state.process.kill()
            
            state.status = ComponentStatus.STOPPED
            state.process = None
            state.pid = None
            
            # Remover PID file
            config = self.components[name]
            if config.pid_file and os.path.exists(config.pid_file):
                os.remove(config.pid_file)
            
            self.logger.info(f"Componente {name} parado")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao parar componente {name}: {e}")
            return False
    
    def check_component_health(self, name: str) -> bool:
        """Verifica saúde de um componente"""
        if name not in self.components:
            return False
        
        config = self.components[name]
        state = self.states[name]
        
        # Verificar se processo ainda existe
        if state.process and state.process.poll() is not None:
            self.logger.warning(f"Processo do componente {name} terminou")
            state.status = ComponentStatus.FAILED
            state.failure_count += 1
            state.last_failure_time = datetime.now()
            return False
        
        # Health check customizado
        if config.health_check_command:
            try:
                result = subprocess.run(
                    config.health_check_command,
                    shell=True,
                    cwd=config.working_dir,
                    timeout=config.health_check_timeout,
                    capture_output=True
                )
                
                if result.returncode != 0:
                    self.logger.warning(f"Health check falhou para {name}: {result.stderr.decode()}")
                    return False
                    
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Health check timeout para {name}")
                return False
            except Exception as e:
                self.logger.error(f"Erro no health check de {name}: {e}")
                return False
        
        # Health check via URL (se configurado)
        if config.health_check_url:
            try:
                import requests
                response = requests.get(
                    config.health_check_url,
                    timeout=config.health_check_timeout
                )
                
                if response.status_code != 200:
                    self.logger.warning(f"Health check URL falhou para {name}: {response.status_code}")
                    return False
                    
            except Exception as e:
                self.logger.warning(f"Erro no health check URL de {name}: {e}")
                return False
        
        state.last_health_check = datetime.now()
        state.health_status = True
        return True
    
    def recover_component(self, name: str):
        """Executa recuperação de um componente"""
        if name not in self.components:
            return
        
        config = self.components[name]
        state = self.states[name]
        
        if state.recovery_in_progress:
            return
        
        state.recovery_in_progress = True
        state.status = ComponentStatus.RECOVERING
        
        try:
            self.recovery_logger.info(f"Iniciando recuperação do componente: {name}")
            
            # Verificar se ainda pode tentar restart
            if state.restart_count >= config.max_restarts:
                self.logger.error(f"Máximo de restarts atingido para {name}")
                
                if config.recovery_action == RecoveryAction.FAILOVER:
                    self.execute_failover(name)
                elif config.recovery_action == RecoveryAction.ALERT_ONLY:
                    self.send_critical_alert(name, "Máximo de restarts atingido")
                
                return
            
            # Executar backup se configurado
            if config.backup_command:
                self.logger.info(f"Executando backup para {name}")
                try:
                    subprocess.run(
                        config.backup_command,
                        shell=True,
                        cwd=config.working_dir,
                        timeout=300
                    )
                except Exception as e:
                    self.logger.error(f"Erro no backup de {name}: {e}")
            
            # Parar componente se ainda estiver rodando
            if state.process:
                self.stop_component(name, force=True)
            
            # Aguardar delay de restart
            self.logger.info(f"Aguardando {config.restart_delay}s antes de reiniciar {name}")
            time.sleep(config.restart_delay)
            
            # Executar ação de recuperação
            if config.recovery_action == RecoveryAction.RESTART:
                success = self.start_component(name)
                
                if success:
                    self.recovery_logger.info(f"Componente {name} recuperado com sucesso")
                else:
                    self.recovery_logger.error(f"Falha na recuperação de {name}")
                    
            elif config.recovery_action == RecoveryAction.CUSTOM and config.custom_recovery_function:
                try:
                    config.custom_recovery_function(name, config, state)
                    self.recovery_logger.info(f"Recuperação customizada executada para {name}")
                except Exception as e:
                    self.recovery_logger.error(f"Erro na recuperação customizada de {name}: {e}")
            
        except Exception as e:
            self.logger.error(f"Erro durante recuperação de {name}: {e}")
        finally:
            state.recovery_in_progress = False
    
    def execute_failover(self, name: str):
        """Executa failover para um componente"""
        self.logger.info(f"Executando failover para {name}")
        # Implementar lógica de failover específica
        # Por exemplo, iniciar instância backup, redirecionar tráfego, etc.
        pass
    
    def send_critical_alert(self, name: str, message: str):
        """Envia alerta crítico"""
        alert_msg = f"🔥 ALERTA CRÍTICO - Componente: {name} - {message}"
        self.logger.critical(alert_msg)
        
        # Aqui você pode integrar com sistemas de notificação
        # Email, Slack, SMS, etc.
    
    def monitor_components(self):
        """Thread de monitoramento contínuo"""
        self.logger.info("Iniciando monitoramento de componentes")
        
        while not self.shutdown_event.is_set():
            try:
                for name in list(self.components.keys()):
                    config = self.components[name]
                    state = self.states[name]
                    
                    # Pular se em manutenção
                    if state.status == ComponentStatus.MAINTENANCE:
                        continue
                    
                    # Verificar se deve fazer health check
                    now = datetime.now()
                    if (state.last_health_check is None or 
                        (now - state.last_health_check).total_seconds() >= config.health_check_interval):
                        
                        if state.status == ComponentStatus.RUNNING:
                            if not self.check_component_health(name):
                                # Componente falhou, iniciar recuperação
                                if config.critical:
                                    self.logger.error(f"Componente crítico {name} falhou")
                                    
                                    # Iniciar recuperação em thread separada
                                    if name not in self.recovery_threads or not self.recovery_threads[name].is_alive():
                                        recovery_thread = threading.Thread(
                                            target=self.recover_component,
                                            args=(name,)
                                        )
                                        recovery_thread.start()
                                        self.recovery_threads[name] = recovery_thread
                
                # Aguardar próximo ciclo
                self.shutdown_event.wait(10)  # Check a cada 10 segundos
                
            except Exception as e:
                self.logger.error(f"Erro no monitoramento: {e}")
                time.sleep(10)
    
    def start_monitoring(self):
        """Inicia monitoramento automático"""
        if self.monitoring_active:
            self.logger.warning("Monitoramento já está ativo")
            return
        
        self.monitoring_active = True
        self.shutdown_event.clear()
        
        # Iniciar thread de monitoramento
        monitor_thread = threading.Thread(target=self.monitor_components)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        self.logger.info("Monitoramento automático iniciado")
    
    def stop_monitoring(self):
        """Para monitoramento automático"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.shutdown_event.set()
        
        self.logger.info("Monitoramento automático parado")
    
    def get_status_report(self) -> Dict:
        """Gera relatório de status"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self.monitoring_active,
            'components': {}
        }
        
        for name, config in self.components.items():
            state = self.states[name]
            
            report['components'][name] = {
                'status': state.status.value,
                'critical': config.critical,
                'pid': state.pid,
                'restart_count': state.restart_count,
                'failure_count': state.failure_count,
                'last_start_time': state.last_start_time.isoformat() if state.last_start_time else None,
                'last_health_check': state.last_health_check.isoformat() if state.last_health_check else None,
                'health_status': state.health_status,
                'recovery_in_progress': state.recovery_in_progress
            }
        
        return report
    
    def shutdown(self):
        """Desliga sistema graciosamente"""
        self.logger.info("Iniciando shutdown do sistema de recuperação")
        
        # Parar monitoramento
        self.stop_monitoring()
        
        # Parar todos os componentes
        for name in list(self.components.keys()):
            if self.states[name].status == ComponentStatus.RUNNING:
                self.stop_component(name)
        
        # Aguardar threads de recuperação
        for thread in self.recovery_threads.values():
            if thread.is_alive():
                thread.join(timeout=30)
        
        self.logger.info("Sistema de recuperação desligado")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema de Recuperação Automática')
    parser.add_argument('--config', type=str, help='Arquivo de configuração')
    parser.add_argument('--start', type=str, help='Iniciar componente')
    parser.add_argument('--stop', type=str, help='Parar componente')
    parser.add_argument('--status', action='store_true', help='Mostrar status')
    parser.add_argument('--monitor', action='store_true', help='Iniciar monitoramento')
    
    args = parser.parse_args()
    
    # Criar sistema
    recovery_system = AutoRecoverySystem(args.config)
    
    try:
        if args.start:
            recovery_system.start_component(args.start)
        elif args.stop:
            recovery_system.stop_component(args.stop)
        elif args.status:
            report = recovery_system.get_status_report()
            print(json.dumps(report, indent=2))
        elif args.monitor:
            recovery_system.start_monitoring()
            
            # Manter rodando até Ctrl+C
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            print("Use --help para ver opções disponíveis")
    
    finally:
        recovery_system.shutdown()

if __name__ == '__main__':
    main()