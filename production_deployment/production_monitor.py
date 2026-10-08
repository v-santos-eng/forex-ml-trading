#!/usr/bin/env python3
"""
Sistema de Monitoramento Contínuo de Recursos em Produção
Integra com o sistema de recuperação automática para garantir estabilidade
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import psutil
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import statistics
import sqlite3

@dataclass
class SystemMetrics:
    """Métricas do sistema"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_processes: int
    system_load_1min: float
    system_load_5min: float
    system_load_15min: float

@dataclass
class ProcessMetrics:
    """Métricas de processo específico"""
    timestamp: datetime
    pid: int
    name: str
    cpu_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    memory_percent: float
    num_threads: int
    num_handles: int
    num_fds: int
    io_read_bytes: int
    io_write_bytes: int
    connections: int
    status: str
    create_time: datetime

@dataclass
class AlertThresholds:
    """Limites para alertas"""
    cpu_warning: float = 70.0
    cpu_critical: float = 90.0
    memory_warning: float = 80.0
    memory_critical: float = 95.0
    disk_warning: float = 85.0
    disk_critical: float = 95.0
    process_memory_warning_mb: float = 1024.0
    process_memory_critical_mb: float = 2048.0
    process_cpu_warning: float = 50.0
    process_cpu_critical: float = 80.0
    handle_growth_rate_warning: float = 10.0  # handles por minuto
    handle_growth_rate_critical: float = 20.0
    memory_growth_rate_warning_mb: float = 50.0  # MB por minuto
    memory_growth_rate_critical_mb: float = 100.0

class ProductionMonitor:
    """Monitor de produção com integração ao sistema de recuperação"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.monitored_processes: Dict[str, int] = {}  # name -> pid
        self.process_metrics_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.system_metrics_history: deque = deque(maxlen=1000)
        self.alert_thresholds = AlertThresholds()
        self.monitoring_active = False
        self.shutdown_event = threading.Event()
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.alert_cooldown_duration = timedelta(minutes=5)
        
        # Setup logging primeiro
        self.setup_logging()
        
        # Database para persistência
        self.db_path = "production_monitoring.db"
        self.init_database()
        
        # Carregar configuração
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        
        # Criar diretórios necessários
        Path("logs").mkdir(exist_ok=True)
        Path("alerts").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Configura sistema de logging"""
        self.logger = logging.getLogger('ProductionMonitor')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('logs/production_monitor.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Alert logger
        self.alert_logger = logging.getLogger('AlertLogger')
        self.alert_logger.setLevel(logging.WARNING)
        alert_handler = logging.FileHandler('logs/production_alerts.log')
        alert_handler.setFormatter(formatter)
        self.alert_logger.addHandler(alert_handler)
    
    def init_database(self):
        """Inicializa banco de dados SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabela de métricas do sistema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_available_gb REAL,
                    disk_usage_percent REAL,
                    disk_free_gb REAL,
                    network_bytes_sent INTEGER,
                    network_bytes_recv INTEGER,
                    active_processes INTEGER,
                    system_load_1min REAL,
                    system_load_5min REAL,
                    system_load_15min REAL
                )
            ''')
            
            # Tabela de métricas de processos
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS process_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pid INTEGER,
                    name TEXT,
                    cpu_percent REAL,
                    memory_rss_mb REAL,
                    memory_vms_mb REAL,
                    memory_percent REAL,
                    num_threads INTEGER,
                    num_handles INTEGER,
                    num_fds INTEGER,
                    io_read_bytes INTEGER,
                    io_write_bytes INTEGER,
                    connections INTEGER,
                    status TEXT,
                    create_time TEXT
                )
            ''')
            
            # Tabela de alertas
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT,
                    severity TEXT,
                    component TEXT,
                    message TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_timestamp TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database inicializado com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar database: {e}")
    
    def load_config(self, config_file: str):
        """Carrega configuração"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Carregar limites de alerta
            if 'alert_thresholds' in config:
                thresholds = config['alert_thresholds']
                for key, value in thresholds.items():
                    if hasattr(self.alert_thresholds, key):
                        setattr(self.alert_thresholds, key, value)
            
            # Carregar processos para monitorar
            if 'monitored_processes' in config:
                for name, pid in config['monitored_processes'].items():
                    self.add_process(name, pid)
            
            self.logger.info(f"Configuração carregada de {config_file}")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar configuração: {e}")
    
    def add_process(self, name: str, pid: int):
        """Adiciona processo para monitoramento"""
        try:
            # Verificar se processo existe
            process = psutil.Process(pid)
            self.monitored_processes[name] = pid
            self.logger.info(f"Processo adicionado para monitoramento: {name} (PID: {pid})")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"Processo com PID {pid} não encontrado")
            return False
    
    def remove_process(self, name: str):
        """Remove processo do monitoramento"""
        if name in self.monitored_processes:
            pid = self.monitored_processes.pop(name)
            if pid in self.process_metrics_history:
                del self.process_metrics_history[pid]
            self.logger.info(f"Processo removido do monitoramento: {name}")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Coleta métricas do sistema"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memória
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disco
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            disk_free_gb = disk.free / (1024**3)
            
            # Rede
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Processos ativos
            active_processes = len(psutil.pids())
            
            # Load average (apenas em sistemas Unix)
            try:
                load_avg = os.getloadavg()
                system_load_1min = load_avg[0]
                system_load_5min = load_avg[1]
                system_load_15min = load_avg[2]
            except (OSError, AttributeError):
                # Windows não tem getloadavg
                system_load_1min = cpu_percent / 100.0
                system_load_5min = cpu_percent / 100.0
                system_load_15min = cpu_percent / 100.0
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                active_processes=active_processes,
                system_load_1min=system_load_1min,
                system_load_5min=system_load_5min,
                system_load_15min=system_load_15min
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao coletar métricas do sistema: {e}")
            return None
    
    def collect_process_metrics(self, pid: int) -> Optional[ProcessMetrics]:
        """Coleta métricas de um processo específico"""
        try:
            process = psutil.Process(pid)
            
            # Informações básicas
            name = process.name()
            status = process.status()
            create_time = datetime.fromtimestamp(process.create_time())
            
            # CPU
            cpu_percent = process.cpu_percent()
            
            # Memória
            memory_info = process.memory_info()
            memory_rss_mb = memory_info.rss / (1024**2)
            memory_vms_mb = memory_info.vms / (1024**2)
            memory_percent = process.memory_percent()
            
            # Threads
            num_threads = process.num_threads()
            
            # Handles/File descriptors
            try:
                if hasattr(process, 'num_handles'):
                    num_handles = process.num_handles()
                else:
                    num_handles = 0
            except (psutil.AccessDenied, AttributeError):
                num_handles = 0
            
            try:
                num_fds = process.num_fds()
            except (psutil.AccessDenied, AttributeError, OSError):
                num_fds = 0
            
            # I/O
            try:
                io_counters = process.io_counters()
                io_read_bytes = io_counters.read_bytes
                io_write_bytes = io_counters.write_bytes
            except (psutil.AccessDenied, AttributeError):
                io_read_bytes = 0
                io_write_bytes = 0
            
            # Conexões de rede
            try:
                connections = len(process.net_connections())
            except (psutil.AccessDenied, AttributeError):
                connections = 0
            
            return ProcessMetrics(
                timestamp=datetime.now(),
                pid=pid,
                name=name,
                cpu_percent=cpu_percent,
                memory_rss_mb=memory_rss_mb,
                memory_vms_mb=memory_vms_mb,
                memory_percent=memory_percent,
                num_threads=num_threads,
                num_handles=num_handles,
                num_fds=num_fds,
                io_read_bytes=io_read_bytes,
                io_write_bytes=io_write_bytes,
                connections=connections,
                status=status,
                create_time=create_time
            )
            
        except psutil.NoSuchProcess:
            self.logger.warning(f"Processo {pid} não existe mais")
            return None
        except Exception as e:
            self.logger.error(f"Erro ao coletar métricas do processo {pid}: {e}")
            return None
    
    def analyze_trends(self, pid: int) -> Dict[str, float]:
        """Analisa tendências de crescimento de recursos"""
        if pid not in self.process_metrics_history:
            return {}
        
        history = list(self.process_metrics_history[pid])
        if len(history) < 5:  # Precisa de pelo menos 5 pontos
            return {}
        
        try:
            # Calcular taxas de crescimento
            time_span_minutes = (history[-1].timestamp - history[0].timestamp).total_seconds() / 60
            
            if time_span_minutes <= 0:
                return {}
            
            # Taxa de crescimento de handles
            handle_growth = (history[-1].num_handles - history[0].num_handles) / time_span_minutes
            
            # Taxa de crescimento de memória
            memory_growth = (history[-1].memory_rss_mb - history[0].memory_rss_mb) / time_span_minutes
            
            # Média de CPU
            cpu_values = [m.cpu_percent for m in history]
            avg_cpu = statistics.mean(cpu_values)
            
            # Desvio padrão de CPU (instabilidade)
            cpu_stability = statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            
            return {
                'handle_growth_rate': handle_growth,
                'memory_growth_rate': memory_growth,
                'avg_cpu': avg_cpu,
                'cpu_stability': cpu_stability,
                'time_span_minutes': time_span_minutes
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao analisar tendências para PID {pid}: {e}")
            return {}
    
    def check_alerts(self, system_metrics: SystemMetrics, process_metrics: Dict[int, ProcessMetrics]):
        """Verifica condições de alerta"""
        alerts = []
        
        # Alertas do sistema
        if system_metrics.cpu_percent >= self.alert_thresholds.cpu_critical:
            alerts.append(('CRITICAL', 'SYSTEM', f'CPU crítico: {system_metrics.cpu_percent:.1f}%'))
        elif system_metrics.cpu_percent >= self.alert_thresholds.cpu_warning:
            alerts.append(('WARNING', 'SYSTEM', f'CPU alto: {system_metrics.cpu_percent:.1f}%'))
        
        if system_metrics.memory_percent >= self.alert_thresholds.memory_critical:
            alerts.append(('CRITICAL', 'SYSTEM', f'Memória crítica: {system_metrics.memory_percent:.1f}%'))
        elif system_metrics.memory_percent >= self.alert_thresholds.memory_warning:
            alerts.append(('WARNING', 'SYSTEM', f'Memória alta: {system_metrics.memory_percent:.1f}%'))
        
        if system_metrics.disk_usage_percent >= self.alert_thresholds.disk_critical:
            alerts.append(('CRITICAL', 'SYSTEM', f'Disco crítico: {system_metrics.disk_usage_percent:.1f}%'))
        elif system_metrics.disk_usage_percent >= self.alert_thresholds.disk_warning:
            alerts.append(('WARNING', 'SYSTEM', f'Disco alto: {system_metrics.disk_usage_percent:.1f}%'))
        
        # Alertas de processos
        for pid, metrics in process_metrics.items():
            process_name = metrics.name
            
            # CPU do processo
            if metrics.cpu_percent >= self.alert_thresholds.process_cpu_critical:
                alerts.append(('CRITICAL', f'PROCESS_{process_name}', 
                             f'CPU crítico do processo {process_name}: {metrics.cpu_percent:.1f}%'))
            elif metrics.cpu_percent >= self.alert_thresholds.process_cpu_warning:
                alerts.append(('WARNING', f'PROCESS_{process_name}', 
                             f'CPU alto do processo {process_name}: {metrics.cpu_percent:.1f}%'))
            
            # Memória do processo
            if metrics.memory_rss_mb >= self.alert_thresholds.process_memory_critical_mb:
                alerts.append(('CRITICAL', f'PROCESS_{process_name}', 
                             f'Memória crítica do processo {process_name}: {metrics.memory_rss_mb:.1f}MB'))
            elif metrics.memory_rss_mb >= self.alert_thresholds.process_memory_warning_mb:
                alerts.append(('WARNING', f'PROCESS_{process_name}', 
                             f'Memória alta do processo {process_name}: {metrics.memory_rss_mb:.1f}MB'))
            
            # Análise de tendências
            trends = self.analyze_trends(pid)
            if trends:
                # Handle growth
                if trends['handle_growth_rate'] >= self.alert_thresholds.handle_growth_rate_critical:
                    alerts.append(('CRITICAL', f'PROCESS_{process_name}', 
                                 f'Vazamento crítico de handles em {process_name}: {trends["handle_growth_rate"]:.1f}/min'))
                elif trends['handle_growth_rate'] >= self.alert_thresholds.handle_growth_rate_warning:
                    alerts.append(('WARNING', f'PROCESS_{process_name}', 
                                 f'Possível vazamento de handles em {process_name}: {trends["handle_growth_rate"]:.1f}/min'))
                
                # Memory growth
                if trends['memory_growth_rate'] >= self.alert_thresholds.memory_growth_rate_critical_mb:
                    alerts.append(('CRITICAL', f'PROCESS_{process_name}', 
                                 f'Vazamento crítico de memória em {process_name}: {trends["memory_growth_rate"]:.1f}MB/min'))
                elif trends['memory_growth_rate'] >= self.alert_thresholds.memory_growth_rate_warning_mb:
                    alerts.append(('WARNING', f'PROCESS_{process_name}', 
                                 f'Possível vazamento de memória em {process_name}: {trends["memory_growth_rate"]:.1f}MB/min'))
        
        # Processar alertas
        for severity, component, message in alerts:
            self.send_alert(severity, component, message)
    
    def send_alert(self, severity: str, component: str, message: str):
        """Envia alerta"""
        alert_key = f"{component}_{severity}"
        now = datetime.now()
        
        # Verificar cooldown
        if alert_key in self.alert_cooldowns:
            if now - self.alert_cooldowns[alert_key] < self.alert_cooldown_duration:
                return  # Ainda em cooldown
        
        self.alert_cooldowns[alert_key] = now
        
        # Log do alerta
        if severity == 'CRITICAL':
            self.alert_logger.critical(f"[{component}] {message}")
        else:
            self.alert_logger.warning(f"[{component}] {message}")
        
        # Salvar no database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO alerts (timestamp, alert_type, severity, component, message) VALUES (?, ?, ?, ?, ?)",
                (now.isoformat(), 'RESOURCE', severity, component, message)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Erro ao salvar alerta no database: {e}")
        
        # Integração com sistema de recuperação (se crítico)
        if severity == 'CRITICAL' and component.startswith('PROCESS_'):
            self.trigger_recovery_action(component, message)
    
    def trigger_recovery_action(self, component: str, message: str):
        """Dispara ação de recuperação automática"""
        try:
            # Extrair nome do processo
            process_name = component.replace('PROCESS_', '')
            
            # Verificar se existe sistema de recuperação
            recovery_config = "recovery_config.json"
            if os.path.exists(recovery_config):
                self.logger.info(f"Disparando recuperação automática para {process_name}")
                
                # Executar sistema de recuperação
                subprocess.Popen([
                    sys.executable, "auto_recovery_system.py",
                    "--config", recovery_config,
                    "--recover", process_name
                ])
            else:
                self.logger.warning("Sistema de recuperação não configurado")
                
        except Exception as e:
            self.logger.error(f"Erro ao disparar recuperação: {e}")
    
    def save_metrics_to_db(self, system_metrics: SystemMetrics, process_metrics: Dict[int, ProcessMetrics]):
        """Salva métricas no database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Salvar métricas do sistema
            cursor.execute(
                """INSERT INTO system_metrics 
                   (timestamp, cpu_percent, memory_percent, memory_available_gb, 
                    disk_usage_percent, disk_free_gb, network_bytes_sent, network_bytes_recv,
                    active_processes, system_load_1min, system_load_5min, system_load_15min)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    system_metrics.timestamp.isoformat(),
                    system_metrics.cpu_percent,
                    system_metrics.memory_percent,
                    system_metrics.memory_available_gb,
                    system_metrics.disk_usage_percent,
                    system_metrics.disk_free_gb,
                    system_metrics.network_bytes_sent,
                    system_metrics.network_bytes_recv,
                    system_metrics.active_processes,
                    system_metrics.system_load_1min,
                    system_metrics.system_load_5min,
                    system_metrics.system_load_15min
                )
            )
            
            # Salvar métricas dos processos
            for pid, metrics in process_metrics.items():
                cursor.execute(
                    """INSERT INTO process_metrics 
                       (timestamp, pid, name, cpu_percent, memory_rss_mb, memory_vms_mb,
                        memory_percent, num_threads, num_handles, num_fds, io_read_bytes,
                        io_write_bytes, connections, status, create_time)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        metrics.timestamp.isoformat(),
                        metrics.pid,
                        metrics.name,
                        metrics.cpu_percent,
                        metrics.memory_rss_mb,
                        metrics.memory_vms_mb,
                        metrics.memory_percent,
                        metrics.num_threads,
                        metrics.num_handles,
                        metrics.num_fds,
                        metrics.io_read_bytes,
                        metrics.io_write_bytes,
                        metrics.connections,
                        metrics.status,
                        metrics.create_time.isoformat()
                    )
                )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar métricas no database: {e}")
    
    def monitoring_loop(self, interval: int = 30):
        """Loop principal de monitoramento"""
        self.logger.info(f"Iniciando monitoramento contínuo (intervalo: {interval}s)")
        
        while not self.shutdown_event.is_set():
            try:
                # Coletar métricas do sistema
                system_metrics = self.collect_system_metrics()
                if system_metrics:
                    self.system_metrics_history.append(system_metrics)
                
                # Coletar métricas dos processos
                process_metrics = {}
                for name, pid in list(self.monitored_processes.items()):
                    metrics = self.collect_process_metrics(pid)
                    if metrics:
                        process_metrics[pid] = metrics
                        self.process_metrics_history[pid].append(metrics)
                    else:
                        # Processo não existe mais
                        self.logger.warning(f"Removendo processo inexistente: {name} (PID: {pid})")
                        self.remove_process(name)
                
                # Verificar alertas
                if system_metrics:
                    self.check_alerts(system_metrics, process_metrics)
                
                # Salvar no database
                if system_metrics:
                    self.save_metrics_to_db(system_metrics, process_metrics)
                
                # Log de status
                if system_metrics:
                    self.logger.info(
                        f"Sistema: CPU {system_metrics.cpu_percent:.1f}%, "
                        f"Mem {system_metrics.memory_percent:.1f}%, "
                        f"Disco {system_metrics.disk_usage_percent:.1f}% | "
                        f"Processos monitorados: {len(process_metrics)}"
                    )
                
                # Aguardar próximo ciclo
                self.shutdown_event.wait(interval)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de monitoramento: {e}")
                time.sleep(interval)
    
    def start_monitoring(self, interval: int = 30):
        """Inicia monitoramento em thread separada"""
        if self.monitoring_active:
            self.logger.warning("Monitoramento já está ativo")
            return
        
        self.monitoring_active = True
        self.shutdown_event.clear()
        
        # Iniciar thread de monitoramento
        monitor_thread = threading.Thread(
            target=self.monitoring_loop,
            args=(interval,)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        self.logger.info("Monitoramento iniciado")
    
    def stop_monitoring(self):
        """Para monitoramento"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.shutdown_event.set()
        
        self.logger.info("Monitoramento parado")
    
    def generate_report(self, hours: int = 24) -> Dict:
        """Gera relatório de monitoramento"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Período do relatório
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            report = {
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'hours': hours
                },
                'system_summary': {},
                'process_summary': {},
                'alerts_summary': {},
                'recommendations': []
            }
            
            # Resumo do sistema
            cursor.execute(
                """SELECT AVG(cpu_percent), MAX(cpu_percent), AVG(memory_percent), MAX(memory_percent),
                          AVG(disk_usage_percent), MAX(disk_usage_percent)
                   FROM system_metrics WHERE timestamp >= ?""",
                (start_time.isoformat(),)
            )
            
            system_stats = cursor.fetchone()
            if system_stats and system_stats[0] is not None:
                report['system_summary'] = {
                    'avg_cpu': round(system_stats[0], 2),
                    'max_cpu': round(system_stats[1], 2),
                    'avg_memory': round(system_stats[2], 2),
                    'max_memory': round(system_stats[3], 2),
                    'avg_disk': round(system_stats[4], 2),
                    'max_disk': round(system_stats[5], 2)
                }
            
            # Resumo de alertas
            cursor.execute(
                "SELECT severity, COUNT(*) FROM alerts WHERE timestamp >= ? GROUP BY severity",
                (start_time.isoformat(),)
            )
            
            alert_counts = dict(cursor.fetchall())
            report['alerts_summary'] = alert_counts
            
            # Recomendações baseadas nos dados
            if report['system_summary']:
                if report['system_summary']['max_cpu'] > 90:
                    report['recommendations'].append("CPU atingiu níveis críticos - considere otimização ou upgrade")
                if report['system_summary']['max_memory'] > 95:
                    report['recommendations'].append("Memória atingiu níveis críticos - investigar vazamentos")
                if report['system_summary']['max_disk'] > 90:
                    report['recommendations'].append("Espaço em disco baixo - limpeza necessária")
            
            if alert_counts.get('CRITICAL', 0) > 0:
                report['recommendations'].append(f"Foram registrados {alert_counts['CRITICAL']} alertas críticos - revisão urgente necessária")
            
            conn.close()
            return report
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar relatório: {e}")
            return {}

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor de Produção')
    parser.add_argument('--config', type=str, help='Arquivo de configuração')
    parser.add_argument('--interval', type=int, default=30, help='Intervalo de monitoramento (segundos)')
    parser.add_argument('--add-process', nargs=2, metavar=('NAME', 'PID'), help='Adicionar processo')
    parser.add_argument('--report', type=int, metavar='HOURS', help='Gerar relatório das últimas N horas')
    parser.add_argument('--daemon', action='store_true', help='Executar como daemon')
    
    args = parser.parse_args()
    
    # Criar monitor
    monitor = ProductionMonitor(args.config)
    
    try:
        if args.add_process:
            name, pid = args.add_process
            monitor.add_process(name, int(pid))
        
        if args.report:
            report = monitor.generate_report(args.report)
            print(json.dumps(report, indent=2))
            return
        
        if args.daemon or not args.add_process:
            # Iniciar monitoramento
            monitor.start_monitoring(args.interval)
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    
    finally:
        monitor.stop_monitoring()

if __name__ == '__main__':
    main()