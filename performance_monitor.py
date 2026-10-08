#!/usr/bin/env python3
"""
Sistema de Monitoramento de Performance em Tempo Real
Monitora CPU, memória, I/O, latência de rede e performance de modelos ML.
"""

import os
import sys
import time
import threading
import psutil
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import numpy as np
from contextlib import contextmanager
import functools
import traceback
from advanced_logging_system import setup_logging

@dataclass
class SystemMetrics:
    """Métricas do sistema"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_read_mb: float
    disk_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    load_average: Optional[List[float]] = None
    
@dataclass
class ProcessMetrics:
    """Métricas de processo específico"""
    timestamp: datetime
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    num_threads: int
    num_fds: int
    io_read_mb: float
    io_write_mb: float
    status: str
    
@dataclass
class ModelPerformanceMetrics:
    """Métricas de performance de modelo ML"""
    timestamp: datetime
    model_name: str
    operation: str  # 'training', 'prediction', 'evaluation'
    duration_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    cpu_usage_percent: float
    input_size: int
    output_size: int
    success: bool
    error_message: Optional[str] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertThreshold:
    """Limites para alertas"""
    metric_name: str
    threshold_value: float
    comparison: str  # 'greater', 'less', 'equal'
    duration_seconds: int  # Duração que deve ser mantida para disparar alerta
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

class PerformanceDatabase:
    """Banco de dados para armazenar métricas de performance"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self._init_database()
    
    def _init_database(self):
        """Inicializa banco de dados"""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.execute('PRAGMA journal_mode=WAL')
        
        # Tabela de métricas do sistema
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_percent REAL,
                memory_percent REAL,
                memory_used_gb REAL,
                memory_available_gb REAL,
                disk_usage_percent REAL,
                disk_read_mb REAL,
                disk_write_mb REAL,
                network_sent_mb REAL,
                network_recv_mb REAL,
                process_count INTEGER,
                load_average TEXT
            )
        ''')
        
        # Tabela de métricas de processo
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS process_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pid INTEGER,
                name TEXT,
                cpu_percent REAL,
                memory_percent REAL,
                memory_rss_mb REAL,
                memory_vms_mb REAL,
                num_threads INTEGER,
                num_fds INTEGER,
                io_read_mb REAL,
                io_write_mb REAL,
                status TEXT
            )
        ''')
        
        # Tabela de métricas de modelos ML
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT,
                operation TEXT,
                duration_ms REAL,
                memory_before_mb REAL,
                memory_after_mb REAL,
                memory_peak_mb REAL,
                cpu_usage_percent REAL,
                input_size INTEGER,
                output_size INTEGER,
                success BOOLEAN,
                error_message TEXT,
                additional_metrics TEXT
            )
        ''')
        
        # Índices para performance
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)')
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_process_timestamp ON process_metrics(timestamp)')
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_model_timestamp ON model_metrics(timestamp)')
        
        self.connection.commit()
    
    def insert_system_metrics(self, metrics: SystemMetrics):
        """Insere métricas do sistema"""
        self.connection.execute('''
            INSERT INTO system_metrics (
                timestamp, cpu_percent, memory_percent, memory_used_gb,
                memory_available_gb, disk_usage_percent, disk_read_mb,
                disk_write_mb, network_sent_mb, network_recv_mb,
                process_count, load_average
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp.isoformat(),
            metrics.cpu_percent,
            metrics.memory_percent,
            metrics.memory_used_gb,
            metrics.memory_available_gb,
            metrics.disk_usage_percent,
            metrics.disk_read_mb,
            metrics.disk_write_mb,
            metrics.network_sent_mb,
            metrics.network_recv_mb,
            metrics.process_count,
            json.dumps(metrics.load_average) if metrics.load_average else None
        ))
        self.connection.commit()
    
    def insert_process_metrics(self, metrics: ProcessMetrics):
        """Insere métricas de processo"""
        self.connection.execute('''
            INSERT INTO process_metrics (
                timestamp, pid, name, cpu_percent, memory_percent,
                memory_rss_mb, memory_vms_mb, num_threads, num_fds,
                io_read_mb, io_write_mb, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp.isoformat(),
            metrics.pid,
            metrics.name,
            metrics.cpu_percent,
            metrics.memory_percent,
            metrics.memory_rss_mb,
            metrics.memory_vms_mb,
            metrics.num_threads,
            metrics.num_fds,
            metrics.io_read_mb,
            metrics.io_write_mb,
            metrics.status
        ))
        self.connection.commit()
    
    def insert_model_metrics(self, metrics: ModelPerformanceMetrics):
        """Insere métricas de modelo ML"""
        self.connection.execute('''
            INSERT INTO model_metrics (
                timestamp, model_name, operation, duration_ms,
                memory_before_mb, memory_after_mb, memory_peak_mb,
                cpu_usage_percent, input_size, output_size,
                success, error_message, additional_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp.isoformat(),
            metrics.model_name,
            metrics.operation,
            metrics.duration_ms,
            metrics.memory_before_mb,
            metrics.memory_after_mb,
            metrics.memory_peak_mb,
            metrics.cpu_usage_percent,
            metrics.input_size,
            metrics.output_size,
            metrics.success,
            metrics.error_message,
            json.dumps(metrics.additional_metrics)
        ))
        self.connection.commit()
    
    def get_recent_metrics(self, table: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Obtém métricas recentes"""
        since = datetime.now() - timedelta(hours=hours)
        cursor = self.connection.execute(
            f'SELECT * FROM {table} WHERE timestamp > ? ORDER BY timestamp DESC',
            (since.isoformat(),)
        )
        
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Remove dados antigos"""
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        cutoff_str = cutoff.isoformat()
        
        for table in ['system_metrics', 'process_metrics', 'model_metrics']:
            self.connection.execute(f'DELETE FROM {table} WHERE timestamp < ?', (cutoff_str,))
        
        self.connection.commit()
        self.connection.execute('VACUUM')

class PerformanceMonitor:
    """Monitor de performance em tempo real"""
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        self.logging_system = setup_logging()
        self.logger = self.logging_system.get_logger('forex_ml.performance')
        
        self.db = PerformanceDatabase(db_path)
        self.is_monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 5  # segundos
        
        # Métricas em memória para análise rápida
        self.recent_system_metrics = deque(maxlen=1000)
        self.recent_process_metrics = deque(maxlen=1000)
        self.recent_model_metrics = deque(maxlen=1000)
        
        # Alertas
        self.alert_thresholds = {}
        self.alert_callbacks = []
        
        # Baseline para comparações
        self.baseline_metrics = {}
        
        # Processo atual
        self.current_process = psutil.Process()
        
        # Contadores de I/O anteriores para calcular deltas
        self.prev_disk_io = None
        self.prev_network_io = None
        
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self):
        """Configura limites padrão para alertas"""
        self.alert_thresholds = {
            'cpu_high': AlertThreshold('cpu_percent', 80.0, 'greater', 60),
            'memory_high': AlertThreshold('memory_percent', 85.0, 'greater', 30),
            'disk_high': AlertThreshold('disk_usage_percent', 90.0, 'greater', 300),
            'model_slow': AlertThreshold('model_duration_ms', 5000.0, 'greater', 0),
            'model_memory_leak': AlertThreshold('memory_growth_mb', 100.0, 'greater', 0)
        }
    
    def start_monitoring(self):
        """Inicia monitoramento em background"""
        if self.is_monitoring:
            self.logger.warning("Monitoramento já está ativo")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Monitoramento de performance iniciado")
    
    def stop_monitoring(self):
        """Para monitoramento"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        self.logger.info("Monitoramento de performance parado")
    
    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        while self.is_monitoring:
            try:
                # Coletar métricas do sistema
                system_metrics = self._collect_system_metrics()
                if system_metrics:
                    self.recent_system_metrics.append(system_metrics)
                    self.db.insert_system_metrics(system_metrics)
                    self._check_system_alerts(system_metrics)
                
                # Coletar métricas do processo atual
                process_metrics = self._collect_process_metrics()
                if process_metrics:
                    self.recent_process_metrics.append(process_metrics)
                    self.db.insert_process_metrics(process_metrics)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de monitoramento: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Coleta métricas do sistema"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memória
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # Disco
            disk_usage = psutil.disk_usage('/')
            disk_usage_percent = disk_usage.percent
            
            # I/O de disco
            disk_io = psutil.disk_io_counters()
            if self.prev_disk_io:
                disk_read_mb = (disk_io.read_bytes - self.prev_disk_io.read_bytes) / (1024**2)
                disk_write_mb = (disk_io.write_bytes - self.prev_disk_io.write_bytes) / (1024**2)
            else:
                disk_read_mb = disk_write_mb = 0.0
            self.prev_disk_io = disk_io
            
            # Rede
            network_io = psutil.net_io_counters()
            if self.prev_network_io:
                network_sent_mb = (network_io.bytes_sent - self.prev_network_io.bytes_sent) / (1024**2)
                network_recv_mb = (network_io.bytes_recv - self.prev_network_io.bytes_recv) / (1024**2)
            else:
                network_sent_mb = network_recv_mb = 0.0
            self.prev_network_io = network_io
            
            # Processos
            process_count = len(psutil.pids())
            
            # Load average (apenas em sistemas Unix)
            load_average = None
            if hasattr(os, 'getloadavg'):
                try:
                    load_average = list(os.getloadavg())
                except:
                    pass
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                disk_read_mb=disk_read_mb,
                disk_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                process_count=process_count,
                load_average=load_average
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao coletar métricas do sistema: {e}")
            return None
    
    def _collect_process_metrics(self) -> Optional[ProcessMetrics]:
        """Coleta métricas do processo atual"""
        try:
            process = self.current_process
            
            # Informações básicas
            pid = process.pid
            name = process.name()
            status = process.status()
            
            # CPU e memória
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            memory_rss_mb = memory_info.rss / (1024**2)
            memory_vms_mb = memory_info.vms / (1024**2)
            
            # Threads e file descriptors
            num_threads = process.num_threads()
            try:
                num_fds = process.num_fds() if hasattr(process, 'num_fds') else 0
            except:
                num_fds = 0
            
            # I/O
            try:
                io_counters = process.io_counters()
                io_read_mb = io_counters.read_bytes / (1024**2)
                io_write_mb = io_counters.write_bytes / (1024**2)
            except:
                io_read_mb = io_write_mb = 0.0
            
            return ProcessMetrics(
                timestamp=datetime.now(),
                pid=pid,
                name=name,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_rss_mb=memory_rss_mb,
                memory_vms_mb=memory_vms_mb,
                num_threads=num_threads,
                num_fds=num_fds,
                io_read_mb=io_read_mb,
                io_write_mb=io_write_mb,
                status=status
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao coletar métricas do processo: {e}")
            return None
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Verifica alertas do sistema"""
        current_time = datetime.now()
        
        # Verificar CPU
        if 'cpu_high' in self.alert_thresholds:
            threshold = self.alert_thresholds['cpu_high']
            if threshold.enabled and metrics.cpu_percent > threshold.threshold_value:
                self._trigger_alert('cpu_high', f"CPU alta: {metrics.cpu_percent:.1f}%", metrics)
        
        # Verificar memória
        if 'memory_high' in self.alert_thresholds:
            threshold = self.alert_thresholds['memory_high']
            if threshold.enabled and metrics.memory_percent > threshold.threshold_value:
                self._trigger_alert('memory_high', f"Memória alta: {metrics.memory_percent:.1f}%", metrics)
        
        # Verificar disco
        if 'disk_high' in self.alert_thresholds:
            threshold = self.alert_thresholds['disk_high']
            if threshold.enabled and metrics.disk_usage_percent > threshold.threshold_value:
                self._trigger_alert('disk_high', f"Disco cheio: {metrics.disk_usage_percent:.1f}%", metrics)
    
    def _trigger_alert(self, alert_type: str, message: str, metrics: Any):
        """Dispara alerta"""
        threshold = self.alert_thresholds.get(alert_type)
        if not threshold:
            return
        
        current_time = datetime.now()
        
        # Verificar se já foi disparado recentemente
        if threshold.last_triggered:
            time_since_last = (current_time - threshold.last_triggered).total_seconds()
            if time_since_last < threshold.duration_seconds:
                return
        
        threshold.last_triggered = current_time
        threshold.trigger_count += 1
        
        # Log do alerta
        self.logger.warning(f"ALERTA [{alert_type}]: {message}")
        
        # Executar callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, message, metrics)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback de alerta: {e}")
    
    def add_alert_callback(self, callback: Callable[[str, str, Any], None]):
        """Adiciona callback para alertas"""
        self.alert_callbacks.append(callback)
    
    def monitor_model_performance(self, model_name: str, operation: str = 'prediction'):
        """Decorator para monitorar performance de modelos ML"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Métricas antes da execução
                start_time = time.time()
                memory_before = psutil.virtual_memory().used / (1024**2)
                cpu_before = psutil.cpu_percent()
                
                success = True
                error_message = None
                result = None
                
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    success = False
                    error_message = str(e)
                    raise
                finally:
                    # Métricas após a execução
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000
                    memory_after = psutil.virtual_memory().used / (1024**2)
                    cpu_after = psutil.cpu_percent()
                    
                    # Estimar tamanhos de entrada e saída
                    input_size = len(str(args)) + len(str(kwargs))
                    output_size = len(str(result)) if result is not None else 0
                    
                    # Criar métricas
                    metrics = ModelPerformanceMetrics(
                        timestamp=datetime.now(),
                        model_name=model_name,
                        operation=operation,
                        duration_ms=duration_ms,
                        memory_before_mb=memory_before,
                        memory_after_mb=memory_after,
                        memory_peak_mb=max(memory_before, memory_after),
                        cpu_usage_percent=(cpu_before + cpu_after) / 2,
                        input_size=input_size,
                        output_size=output_size,
                        success=success,
                        error_message=error_message
                    )
                    
                    # Armazenar métricas
                    self.recent_model_metrics.append(metrics)
                    self.db.insert_model_metrics(metrics)
                    
                    # Verificar alertas de modelo
                    self._check_model_alerts(metrics)
                    
                    # Log de performance
                    if success:
                        self.logger.info(f"Modelo {model_name} ({operation}): {duration_ms:.1f}ms, "
                                       f"Memória: {memory_after-memory_before:+.1f}MB")
                    else:
                        self.logger.error(f"Modelo {model_name} ({operation}) falhou: {error_message}")
                
                return result
            return wrapper
        return decorator
    
    def _check_model_alerts(self, metrics: ModelPerformanceMetrics):
        """Verifica alertas de modelo"""
        # Alerta de modelo lento
        if 'model_slow' in self.alert_thresholds:
            threshold = self.alert_thresholds['model_slow']
            if threshold.enabled and metrics.duration_ms > threshold.threshold_value:
                message = f"Modelo {metrics.model_name} lento: {metrics.duration_ms:.1f}ms"
                self._trigger_alert('model_slow', message, metrics)
        
        # Alerta de vazamento de memória
        memory_growth = metrics.memory_after_mb - metrics.memory_before_mb
        if 'model_memory_leak' in self.alert_thresholds:
            threshold = self.alert_thresholds['model_memory_leak']
            if threshold.enabled and memory_growth > threshold.threshold_value:
                message = f"Possível vazamento de memória no modelo {metrics.model_name}: +{memory_growth:.1f}MB"
                self._trigger_alert('model_memory_leak', message, metrics)
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Obtém resumo de performance"""
        summary = {
            'period_hours': hours,
            'timestamp': datetime.now().isoformat(),
            'system': {},
            'process': {},
            'models': {}
        }
        
        # Métricas do sistema
        system_data = self.db.get_recent_metrics('system_metrics', hours)
        if system_data:
            cpu_values = [m['cpu_percent'] for m in system_data if m['cpu_percent'] is not None]
            memory_values = [m['memory_percent'] for m in system_data if m['memory_percent'] is not None]
            
            summary['system'] = {
                'cpu_avg': statistics.mean(cpu_values) if cpu_values else 0,
                'cpu_max': max(cpu_values) if cpu_values else 0,
                'memory_avg': statistics.mean(memory_values) if memory_values else 0,
                'memory_max': max(memory_values) if memory_values else 0,
                'samples': len(system_data)
            }
        
        # Métricas de processo
        process_data = self.db.get_recent_metrics('process_metrics', hours)
        if process_data:
            cpu_values = [m['cpu_percent'] for m in process_data if m['cpu_percent'] is not None]
            memory_values = [m['memory_rss_mb'] for m in process_data if m['memory_rss_mb'] is not None]
            
            summary['process'] = {
                'cpu_avg': statistics.mean(cpu_values) if cpu_values else 0,
                'memory_avg_mb': statistics.mean(memory_values) if memory_values else 0,
                'memory_max_mb': max(memory_values) if memory_values else 0,
                'samples': len(process_data)
            }
        
        # Métricas de modelos
        model_data = self.db.get_recent_metrics('model_metrics', hours)
        if model_data:
            models_summary = defaultdict(lambda: {'count': 0, 'durations': [], 'successes': 0})
            
            for m in model_data:
                model_name = m['model_name']
                models_summary[model_name]['count'] += 1
                if m['duration_ms'] is not None:
                    models_summary[model_name]['durations'].append(m['duration_ms'])
                if m['success']:
                    models_summary[model_name]['successes'] += 1
            
            for model_name, data in models_summary.items():
                durations = data['durations']
                summary['models'][model_name] = {
                    'total_calls': data['count'],
                    'success_rate': data['successes'] / data['count'] if data['count'] > 0 else 0,
                    'avg_duration_ms': statistics.mean(durations) if durations else 0,
                    'max_duration_ms': max(durations) if durations else 0
                }
        
        return summary
    
    def export_metrics(self, output_file: str, hours: int = 24, format: str = 'json'):
        """Exporta métricas para arquivo"""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'period_hours': hours,
            'system_metrics': self.db.get_recent_metrics('system_metrics', hours),
            'process_metrics': self.db.get_recent_metrics('process_metrics', hours),
            'model_metrics': self.db.get_recent_metrics('model_metrics', hours)
        }
        
        if format.lower() == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Formato não suportado: {format}")
        
        self.logger.info(f"Métricas exportadas para {output_file}")

# Instância global do monitor
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Obtém instância global do monitor de performance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def monitor_performance(model_name: str, operation: str = 'prediction'):
    """Decorator para monitorar performance de função"""
    return get_performance_monitor().monitor_model_performance(model_name, operation)

if __name__ == "__main__":
    # Exemplo de uso
    import random
    
    monitor = get_performance_monitor()
    monitor.start_monitoring()
    
    @monitor_performance('test_model', 'training')
    def simulate_model_training():
        """Simula treinamento de modelo"""
        time.sleep(random.uniform(0.1, 2.0))
        # Simular uso de memória
        data = [random.random() for _ in range(100000)]
        return sum(data)
    
    print("Iniciando simulação de monitoramento...")
    
    # Executar algumas operações
    for i in range(5):
        result = simulate_model_training()
        print(f"Operação {i+1} concluída: {result:.2f}")
        time.sleep(1)
    
    # Obter resumo
    summary = monitor.get_performance_summary(hours=1)
    print("\nResumo de Performance:")
    print(json.dumps(summary, indent=2))
    
    monitor.stop_monitoring()