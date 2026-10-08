#!/usr/bin/env python3
"""
Sistema de Monitoramento Contínuo para Produção
Monitora processos críticos 24/7 com alertas automáticos
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import psutil
from dataclasses import dataclass, asdict

@dataclass
class MonitorConfig:
    """Configuração do monitoramento"""
    check_interval: int = 30  # segundos
    alert_cooldown: int = 300  # 5 minutos
    max_cpu_percent: float = 80.0
    max_memory_mb: float = 1024.0
    max_handles: int = 1000
    max_threads: int = 50
    max_open_files: int = 100
    handle_growth_rate: float = 10.0  # handles por minuto
    memory_growth_rate: float = 50.0  # MB por minuto
    log_file: str = "continuous_monitor.log"
    alert_log: str = "alerts.log"
    metrics_file: str = "metrics.json"
    enable_console: bool = True
    enable_file_logging: bool = True

class ContinuousMonitor:
    """Monitor contínuo de processos"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.processes: Dict[int, Dict] = {}
        self.last_alerts: Dict[str, datetime] = {}
        self.metrics_history: List[Dict] = []
        
        # Setup logging
        self.setup_logging()
        
        # Criar diretórios necessários
        Path(self.config.log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.alert_log).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.metrics_file).parent.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Configura sistema de logging"""
        self.logger = logging.getLogger('ContinuousMonitor')
        self.logger.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if self.config.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.enable_file_logging:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Alert logger
        self.alert_logger = logging.getLogger('AlertLogger')
        self.alert_logger.setLevel(logging.WARNING)
        alert_handler = logging.FileHandler(self.config.alert_log)
        alert_handler.setFormatter(formatter)
        self.alert_logger.addHandler(alert_handler)
    
    def add_process(self, pid: int, name: str, critical: bool = False):
        """Adiciona processo ao monitoramento"""
        try:
            process = psutil.Process(pid)
            self.processes[pid] = {
                'name': name,
                'critical': critical,
                'process': process,
                'start_time': datetime.now(),
                'last_metrics': None,
                'alert_count': 0,
                'history': []
            }
            self.logger.info(f"Processo adicionado: {name} (PID: {pid}, Crítico: {critical})")
        except psutil.NoSuchProcess:
            self.logger.error(f"Processo {pid} não encontrado")
    
    def remove_process(self, pid: int):
        """Remove processo do monitoramento"""
        if pid in self.processes:
            name = self.processes[pid]['name']
            del self.processes[pid]
            self.logger.info(f"Processo removido: {name} (PID: {pid})")
    
    def collect_metrics(self, pid: int) -> Optional[Dict]:
        """Coleta métricas de um processo"""
        try:
            process_info = self.processes[pid]
            process = process_info['process']
            
            # Verificar se processo ainda existe
            if not process.is_running():
                return None
            
            # Coletar métricas
            memory_info = process.memory_info()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'pid': pid,
                'name': process_info['name'],
                'cpu_percent': process.cpu_percent(),
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
                'threads': process.num_threads(),
                'status': process.status(),
                'create_time': process.create_time()
            }
            
            # Métricas específicas do Windows
            try:
                metrics['handles'] = process.num_handles()
            except (AttributeError, psutil.AccessDenied):
                metrics['handles'] = 0
            
            # Arquivos abertos
            try:
                metrics['open_files'] = len(process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                metrics['open_files'] = 0
            
            # Conexões de rede
            try:
                metrics['connections'] = len(process.net_connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
                metrics['connections'] = 0
            
            return metrics
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.warning(f"Erro ao coletar métricas do PID {pid}: {e}")
            return None
    
    def analyze_metrics(self, pid: int, current_metrics: Dict) -> List[str]:
        """Analisa métricas e detecta anomalias"""
        alerts = []
        process_info = self.processes[pid]
        last_metrics = process_info['last_metrics']
        
        # Verificar limites absolutos
        if current_metrics['cpu_percent'] > self.config.max_cpu_percent:
            alerts.append(f"CPU alta: {current_metrics['cpu_percent']:.1f}%")
        
        if current_metrics['memory_rss_mb'] > self.config.max_memory_mb:
            alerts.append(f"Memória alta: {current_metrics['memory_rss_mb']:.1f}MB")
        
        if current_metrics['handles'] > self.config.max_handles:
            alerts.append(f"Handles excessivos: {current_metrics['handles']}")
        
        if current_metrics['threads'] > self.config.max_threads:
            alerts.append(f"Threads excessivas: {current_metrics['threads']}")
        
        if current_metrics['open_files'] > self.config.max_open_files:
            alerts.append(f"Arquivos abertos excessivos: {current_metrics['open_files']}")
        
        # Verificar taxas de crescimento
        if last_metrics:
            time_diff = (datetime.fromisoformat(current_metrics['timestamp']) - 
                        datetime.fromisoformat(last_metrics['timestamp'])).total_seconds() / 60
            
            if time_diff > 0:
                # Taxa de crescimento de handles
                handle_growth = (current_metrics['handles'] - last_metrics['handles']) / time_diff
                if handle_growth > self.config.handle_growth_rate:
                    alerts.append(f"Crescimento rápido de handles: {handle_growth:.1f}/min")
                
                # Taxa de crescimento de memória
                memory_growth = (current_metrics['memory_rss_mb'] - last_metrics['memory_rss_mb']) / time_diff
                if memory_growth > self.config.memory_growth_rate:
                    alerts.append(f"Crescimento rápido de memória: {memory_growth:.1f}MB/min")
        
        return alerts
    
    def send_alert(self, pid: int, alerts: List[str]):
        """Envia alertas"""
        process_info = self.processes[pid]
        alert_key = f"{pid}_{','.join(alerts)}"
        
        # Verificar cooldown
        now = datetime.now()
        if alert_key in self.last_alerts:
            if (now - self.last_alerts[alert_key]).total_seconds() < self.config.alert_cooldown:
                return
        
        self.last_alerts[alert_key] = now
        process_info['alert_count'] += 1
        
        # Log do alerta
        alert_msg = f"🚨 ALERTA - {process_info['name']} (PID: {pid}): {', '.join(alerts)}"
        
        if process_info['critical']:
            alert_msg = f"🔥 CRÍTICO - {alert_msg}"
        
        self.logger.warning(alert_msg)
        self.alert_logger.warning(alert_msg)
        
        # Console alert
        if self.config.enable_console:
            print(f"\n{alert_msg}\n")
    
    def save_metrics(self):
        """Salva métricas em arquivo JSON"""
        try:
            # Manter apenas últimas 1000 entradas
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            with open(self.config.metrics_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'config': asdict(self.config),
                    'metrics': self.metrics_history[-100:]  # Últimas 100 entradas
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Erro ao salvar métricas: {e}")
    
    def monitor_cycle(self):
        """Executa um ciclo de monitoramento"""
        cycle_metrics = {
            'timestamp': datetime.now().isoformat(),
            'processes': {}
        }
        
        for pid in list(self.processes.keys()):
            metrics = self.collect_metrics(pid)
            
            if metrics is None:
                # Processo morreu
                process_name = self.processes[pid]['name']
                self.logger.warning(f"Processo {process_name} (PID: {pid}) não está mais em execução")
                self.remove_process(pid)
                continue
            
            # Analisar métricas
            alerts = self.analyze_metrics(pid, metrics)
            
            if alerts:
                self.send_alert(pid, alerts)
            
            # Atualizar histórico
            self.processes[pid]['last_metrics'] = metrics
            self.processes[pid]['history'].append(metrics)
            
            # Manter apenas últimas 100 entradas no histórico
            if len(self.processes[pid]['history']) > 100:
                self.processes[pid]['history'] = self.processes[pid]['history'][-100:]
            
            cycle_metrics['processes'][str(pid)] = {
                'metrics': metrics,
                'alerts': alerts,
                'alert_count': self.processes[pid]['alert_count']
            }
        
        # Adicionar ao histórico global
        self.metrics_history.append(cycle_metrics)
        
        # Salvar métricas periodicamente (a cada 10 ciclos)
        if len(self.metrics_history) % 10 == 0:
            self.save_metrics()
    
    def run(self, duration_minutes: Optional[int] = None):
        """Executa monitoramento contínuo"""
        self.logger.info("Iniciando monitoramento contínuo")
        
        if duration_minutes:
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            self.logger.info(f"Monitoramento por {duration_minutes} minutos")
        else:
            end_time = None
            self.logger.info("Monitoramento contínuo (pressione Ctrl+C para parar)")
        
        try:
            while True:
                if end_time and datetime.now() >= end_time:
                    break
                
                self.monitor_cycle()
                time.sleep(self.config.check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Monitoramento interrompido pelo usuário")
        except Exception as e:
            self.logger.error(f"Erro durante monitoramento: {e}")
        finally:
            self.save_metrics()
            self.logger.info("Monitoramento finalizado")
    
    def status_report(self) -> Dict:
        """Gera relatório de status"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'processes': {},
            'summary': {
                'total_processes': len(self.processes),
                'critical_processes': sum(1 for p in self.processes.values() if p['critical']),
                'total_alerts': sum(p['alert_count'] for p in self.processes.values()),
                'uptime_hours': 0
            }
        }
        
        for pid, process_info in self.processes.items():
            uptime = (datetime.now() - process_info['start_time']).total_seconds() / 3600
            
            report['processes'][str(pid)] = {
                'name': process_info['name'],
                'critical': process_info['critical'],
                'uptime_hours': uptime,
                'alert_count': process_info['alert_count'],
                'latest_metrics': process_info['last_metrics']
            }
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Monitor Contínuo de Processos')
    parser.add_argument('--config', type=str, help='Arquivo de configuração JSON')
    parser.add_argument('--pid', type=int, action='append', help='PID para monitorar')
    parser.add_argument('--name', type=str, action='append', help='Nome do processo')
    parser.add_argument('--critical', action='store_true', help='Marcar como crítico')
    parser.add_argument('--duration', type=int, help='Duração em minutos (padrão: contínuo)')
    parser.add_argument('--interval', type=int, default=30, help='Intervalo de verificação em segundos')
    parser.add_argument('--status', action='store_true', help='Mostrar status atual')
    
    args = parser.parse_args()
    
    # Carregar configuração
    config = MonitorConfig()
    config.check_interval = args.interval
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Criar monitor
    monitor = ContinuousMonitor(config)
    
    # Adicionar processos
    if args.pid:
        pids = args.pid
        names = args.name if args.name else [f"Process-{pid}" for pid in pids]
        
        for i, pid in enumerate(pids):
            name = names[i] if i < len(names) else f"Process-{pid}"
            monitor.add_process(pid, name, args.critical)
    
    # Mostrar status ou executar monitoramento
    if args.status:
        report = monitor.status_report()
        print(json.dumps(report, indent=2))
    else:
        if not monitor.processes:
            print("Nenhum processo especificado para monitoramento")
            print("Use --pid <PID> --name <NOME> para adicionar processos")
            return
        
        monitor.run(args.duration)

if __name__ == '__main__':
    main()