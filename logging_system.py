#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Logs e Monitoramento para MT5 Trading
Registra operações, erros, performance e métricas para auditoria e debugging
Autor: Sistema Forex ML
Data: 2024
"""

import os
import sys
import logging
import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import sqlite3
from collections import deque, defaultdict
import psutil
import traceback
from contextlib import contextmanager

class LogLevel(Enum):
    """Níveis de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class EventType(Enum):
    """Tipos de eventos"""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONNECTION = "connection"
    DISCONNECTION = "disconnection"
    ORDER_SENT = "order_sent"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_MODIFIED = "position_modified"
    ML_PREDICTION = "ml_prediction"
    SIGNAL_GENERATED = "signal_generated"
    RISK_VIOLATION = "risk_violation"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"
    CONFIG_CHANGED = "config_changed"
    DATA_UPDATE = "data_update"

@dataclass
class LogEntry:
    """Entrada de log"""
    timestamp: datetime
    level: LogLevel
    event_type: EventType
    message: str
    component: str
    details: Dict[str, Any] = None
    exception: str = None
    execution_time_ms: float = None
    memory_usage_mb: float = None
    thread_id: str = None
    session_id: str = None

@dataclass
class PerformanceMetric:
    """Métrica de performance"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    component: str
    additional_data: Dict[str, Any] = None

@dataclass
class TradingEvent:
    """Evento de trading"""
    timestamp: datetime
    event_type: EventType
    symbol: str
    action: str
    details: Dict[str, Any]
    result: str = None
    error_message: str = None
    execution_time_ms: float = None

class DatabaseLogger:
    """
    Logger que armazena dados em SQLite
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self._init_database()
    
    def _init_database(self):
        """Inicializa banco de dados"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.execute('PRAGMA journal_mode=WAL')
            
            # Criar tabelas
            self._create_tables()
            
        except Exception as e:
            print(f"Erro ao inicializar banco de dados: {e}")
    
    def _create_tables(self):
        """Cria tabelas do banco"""
        # Tabela de logs
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                component TEXT NOT NULL,
                details TEXT,
                exception TEXT,
                execution_time_ms REAL,
                memory_usage_mb REAL,
                thread_id TEXT,
                session_id TEXT
            )
        ''')
        
        # Tabela de métricas
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                component TEXT NOT NULL,
                additional_data TEXT
            )
        ''')
        
        # Tabela de eventos de trading
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS trading_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                result TEXT,
                error_message TEXT,
                execution_time_ms REAL
            )
        ''')
        
        # Criar índices
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)')
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level)')
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_logs_component ON logs(component)')
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)')
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_trading_timestamp ON trading_events(timestamp)')
        
        self.connection.commit()
    
    def log_entry(self, entry: LogEntry):
        """Registra entrada de log"""
        try:
            self.connection.execute('''
                INSERT INTO logs (
                    timestamp, level, event_type, message, component,
                    details, exception, execution_time_ms, memory_usage_mb,
                    thread_id, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.timestamp.isoformat(),
                entry.level.value,
                entry.event_type.value,
                entry.message,
                entry.component,
                json.dumps(entry.details) if entry.details else None,
                entry.exception,
                entry.execution_time_ms,
                entry.memory_usage_mb,
                entry.thread_id,
                entry.session_id
            ))
            self.connection.commit()
            
        except Exception as e:
            print(f"Erro ao registrar log: {e}")
    
    def log_metric(self, metric: PerformanceMetric):
        """Registra métrica de performance"""
        try:
            self.connection.execute('''
                INSERT INTO performance_metrics (
                    timestamp, metric_name, value, unit, component, additional_data
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                metric.timestamp.isoformat(),
                metric.metric_name,
                metric.value,
                metric.unit,
                metric.component,
                json.dumps(metric.additional_data) if metric.additional_data else None
            ))
            self.connection.commit()
            
        except Exception as e:
            print(f"Erro ao registrar métrica: {e}")
    
    def log_trading_event(self, event: TradingEvent):
        """Registra evento de trading"""
        try:
            self.connection.execute('''
                INSERT INTO trading_events (
                    timestamp, event_type, symbol, action, details,
                    result, error_message, execution_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.timestamp.isoformat(),
                event.event_type.value,
                event.symbol,
                event.action,
                json.dumps(event.details),
                event.result,
                event.error_message,
                event.execution_time_ms
            ))
            self.connection.commit()
            
        except Exception as e:
            print(f"Erro ao registrar evento de trading: {e}")
    
    def get_logs(self, start_time: datetime = None, end_time: datetime = None,
                level: LogLevel = None, component: str = None, limit: int = 1000) -> List[Dict]:
        """Recupera logs com filtros"""
        try:
            query = "SELECT * FROM logs WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            if level:
                query += " AND level = ?"
                params.append(level.value)
            
            if component:
                query += " AND component = ?"
                params.append(component)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = self.connection.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
        except Exception as e:
            print(f"Erro ao recuperar logs: {e}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Remove dados antigos"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            # Limpar logs antigos
            self.connection.execute("DELETE FROM logs WHERE timestamp < ?", (cutoff_date,))
            
            # Limpar métricas antigas
            self.connection.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff_date,))
            
            # Limpar eventos antigos
            self.connection.execute("DELETE FROM trading_events WHERE timestamp < ?", (cutoff_date,))
            
            self.connection.commit()
            
            # Vacuum para recuperar espaço
            self.connection.execute("VACUUM")
            
        except Exception as e:
            print(f"Erro ao limpar dados antigos: {e}")
    
    def close(self):
        """Fecha conexão com banco"""
        if self.connection:
            self.connection.close()

class PerformanceMonitor:
    """
    Monitor de performance do sistema
    """
    
    def __init__(self):
        self.metrics_history = defaultdict(deque)
        self.max_history_size = 1000
        self.monitoring_active = False
        self.monitor_thread = None
        self.callbacks = []
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Inicia monitoramento"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Para monitoramento"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def add_callback(self, callback: Callable[[PerformanceMetric], None]):
        """Adiciona callback para métricas"""
        self.callbacks.append(callback)
    
    def _monitoring_loop(self, interval: float):
        """Loop principal de monitoramento"""
        while self.monitoring_active:
            try:
                # Coletar métricas
                metrics = self._collect_metrics()
                
                # Armazenar no histórico
                for metric in metrics:
                    history = self.metrics_history[metric.metric_name]
                    history.append(metric)
                    
                    # Limitar tamanho do histórico
                    if len(history) > self.max_history_size:
                        history.popleft()
                    
                    # Chamar callbacks
                    for callback in self.callbacks:
                        try:
                            callback(metric)
                        except Exception as e:
                            print(f"Erro em callback de métrica: {e}")
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Erro no loop de monitoramento: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> List[PerformanceMetric]:
        """Coleta métricas do sistema"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent()
            metrics.append(PerformanceMetric(
                timestamp=timestamp,
                metric_name="cpu_usage",
                value=cpu_percent,
                unit="percent",
                component="system"
            ))
            
            # Memória
            memory = psutil.virtual_memory()
            metrics.append(PerformanceMetric(
                timestamp=timestamp,
                metric_name="memory_usage",
                value=memory.percent,
                unit="percent",
                component="system"
            ))
            
            metrics.append(PerformanceMetric(
                timestamp=timestamp,
                metric_name="memory_available",
                value=memory.available / (1024**3),  # GB
                unit="GB",
                component="system"
            ))
            
            # Processo atual
            process = psutil.Process()
            
            # CPU do processo
            process_cpu = process.cpu_percent()
            metrics.append(PerformanceMetric(
                timestamp=timestamp,
                metric_name="process_cpu_usage",
                value=process_cpu,
                unit="percent",
                component="process"
            ))
            
            # Memória do processo
            process_memory = process.memory_info()
            metrics.append(PerformanceMetric(
                timestamp=timestamp,
                metric_name="process_memory_rss",
                value=process_memory.rss / (1024**2),  # MB
                unit="MB",
                component="process"
            ))
            
            # Threads
            thread_count = threading.active_count()
            metrics.append(PerformanceMetric(
                timestamp=timestamp,
                metric_name="thread_count",
                value=thread_count,
                unit="count",
                component="process"
            ))
            
            # Disco (se necessário)
            disk_usage = psutil.disk_usage('/')
            metrics.append(PerformanceMetric(
                timestamp=timestamp,
                metric_name="disk_usage",
                value=disk_usage.percent,
                unit="percent",
                component="system"
            ))
            
        except Exception as e:
            print(f"Erro ao coletar métricas: {e}")
        
        return metrics
    
    def get_metric_history(self, metric_name: str, limit: int = None) -> List[PerformanceMetric]:
        """Retorna histórico de uma métrica"""
        history = list(self.metrics_history.get(metric_name, []))
        if limit:
            history = history[-limit:]
        return history
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Retorna métricas atuais"""
        current = {}
        for metric_name, history in self.metrics_history.items():
            if history:
                current[metric_name] = history[-1].value
        return current

class LoggingSystem:
    """
    Sistema principal de logging e monitoramento
    """
    
    def __init__(self, log_dir: str = None, session_id: str = None):
        # Configuração
        if log_dir is None:
            log_dir = os.path.join(os.path.expanduser("~"), ".mt5_trading", "logs")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = session_id or f"session_{int(time.time())}"
        
        # Componentes
        self.db_logger = DatabaseLogger(str(self.log_dir / "trading.db"))
        self.performance_monitor = PerformanceMonitor()
        
        # Configurar logging padrão do Python
        self._setup_python_logging()
        
        # Buffer para logs em memória
        self.memory_buffer = deque(maxlen=1000)
        
        # Callbacks
        self.log_callbacks = []
        
        # Configurar monitor de performance
        self.performance_monitor.add_callback(self._on_performance_metric)
        
        # Estatísticas
        self.stats = {
            'logs_written': 0,
            'errors_logged': 0,
            'warnings_logged': 0,
            'trading_events_logged': 0,
            'metrics_collected': 0
        }
    
    def _setup_python_logging(self):
        """Configura logging padrão do Python"""
        # Arquivo de log rotativo
        log_file = self.log_dir / f"mt5_trading_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Configurar formatador
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Handler para arquivo
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Configurar logger raiz
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Handler customizado para capturar logs
        custom_handler = LoggingHandler(self)
        custom_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(custom_handler)
    
    def start(self):
        """Inicia sistema de logging"""
        self.log(LogLevel.INFO, EventType.SYSTEM_START, "Sistema de logging iniciado", "logging_system")
        self.performance_monitor.start_monitoring()
    
    def stop(self):
        """Para sistema de logging"""
        self.performance_monitor.stop_monitoring()
        self.log(LogLevel.INFO, EventType.SYSTEM_STOP, "Sistema de logging parado", "logging_system")
        self.db_logger.close()
    
    def log(self, level: LogLevel, event_type: EventType, message: str, 
           component: str, details: Dict[str, Any] = None, 
           exception: Exception = None, execution_time_ms: float = None):
        """Registra entrada de log"""
        try:
            # Criar entrada
            entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                event_type=event_type,
                message=message,
                component=component,
                details=details,
                exception=str(exception) if exception else None,
                execution_time_ms=execution_time_ms,
                memory_usage_mb=self._get_memory_usage(),
                thread_id=str(threading.current_thread().ident),
                session_id=self.session_id
            )
            
            # Armazenar no banco
            self.db_logger.log_entry(entry)
            
            # Armazenar no buffer de memória
            self.memory_buffer.append(entry)
            
            # Atualizar estatísticas
            self.stats['logs_written'] += 1
            if level == LogLevel.ERROR:
                self.stats['errors_logged'] += 1
            elif level == LogLevel.WARNING:
                self.stats['warnings_logged'] += 1
            
            # Chamar callbacks
            for callback in self.log_callbacks:
                try:
                    callback(entry)
                except Exception as e:
                    print(f"Erro em callback de log: {e}")
            
        except Exception as e:
            print(f"Erro ao registrar log: {e}")
    
    def log_trading_event(self, event_type: EventType, symbol: str, action: str,
                         details: Dict[str, Any], result: str = None,
                         error_message: str = None, execution_time_ms: float = None):
        """Registra evento de trading"""
        try:
            event = TradingEvent(
                timestamp=datetime.now(),
                event_type=event_type,
                symbol=symbol,
                action=action,
                details=details,
                result=result,
                error_message=error_message,
                execution_time_ms=execution_time_ms
            )
            
            self.db_logger.log_trading_event(event)
            self.stats['trading_events_logged'] += 1
            
            # Log também como entrada normal
            level = LogLevel.ERROR if error_message else LogLevel.INFO
            message = f"Trading: {action} {symbol}"
            if result:
                message += f" - {result}"
            
            self.log(level, event_type, message, "trading", details)
            
        except Exception as e:
            print(f"Erro ao registrar evento de trading: {e}")
    
    def _on_performance_metric(self, metric: PerformanceMetric):
        """Callback para métricas de performance"""
        try:
            self.db_logger.log_metric(metric)
            self.stats['metrics_collected'] += 1
            
            # Log crítico se CPU ou memória muito altos
            if metric.metric_name == "cpu_usage" and metric.value > 90:
                self.log(LogLevel.WARNING, EventType.PERFORMANCE_METRIC,
                        f"CPU usage alto: {metric.value:.1f}%", "performance")
            
            elif metric.metric_name == "memory_usage" and metric.value > 90:
                self.log(LogLevel.WARNING, EventType.PERFORMANCE_METRIC,
                        f"Uso de memória alto: {metric.value:.1f}%", "performance")
            
        except Exception as e:
            print(f"Erro ao processar métrica: {e}")
    
    def _get_memory_usage(self) -> float:
        """Retorna uso de memória atual em MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024**2)
        except:
            return 0.0
    
    def add_log_callback(self, callback: Callable[[LogEntry], None]):
        """Adiciona callback para logs"""
        self.log_callbacks.append(callback)
    
    def get_recent_logs(self, limit: int = 100) -> List[LogEntry]:
        """Retorna logs recentes do buffer de memória"""
        return list(self.memory_buffer)[-limit:]
    
    def get_logs_from_db(self, **kwargs) -> List[Dict]:
        """Retorna logs do banco de dados"""
        return self.db_logger.get_logs(**kwargs)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Retorna métricas atuais de performance"""
        return self.performance_monitor.get_current_metrics()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do sistema"""
        stats = self.stats.copy()
        stats['session_id'] = self.session_id
        stats['uptime_seconds'] = (datetime.now() - datetime.fromisoformat(self.session_id.split('_')[1])).total_seconds() if '_' in self.session_id else 0
        stats['performance_metrics'] = self.get_performance_metrics()
        return stats
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Remove logs antigos"""
        try:
            self.db_logger.cleanup_old_data(days_to_keep)
            self.log(LogLevel.INFO, EventType.SYSTEM_START,
                    f"Limpeza de logs concluída - mantidos {days_to_keep} dias", "logging_system")
        except Exception as e:
            self.log(LogLevel.ERROR, EventType.ERROR_OCCURRED,
                    f"Erro na limpeza de logs: {e}", "logging_system", exception=e)

class LoggingHandler(logging.Handler):
    """
    Handler customizado para capturar logs do Python
    """
    
    def __init__(self, logging_system: LoggingSystem):
        super().__init__()
        self.logging_system = logging_system
    
    def emit(self, record):
        try:
            # Mapear níveis
            level_map = {
                logging.DEBUG: LogLevel.DEBUG,
                logging.INFO: LogLevel.INFO,
                logging.WARNING: LogLevel.WARNING,
                logging.ERROR: LogLevel.ERROR,
                logging.CRITICAL: LogLevel.CRITICAL
            }
            
            level = level_map.get(record.levelno, LogLevel.INFO)
            
            # Determinar tipo de evento
            event_type = EventType.ERROR_OCCURRED if level in [LogLevel.ERROR, LogLevel.CRITICAL] else EventType.SYSTEM_START
            
            # Extrair detalhes
            details = {
                'logger_name': record.name,
                'filename': record.filename,
                'lineno': record.lineno,
                'funcName': record.funcName
            }
            
            # Registrar no sistema
            self.logging_system.log(
                level=level,
                event_type=event_type,
                message=record.getMessage(),
                component=record.name,
                details=details,
                exception=record.exc_info[1] if record.exc_info else None
            )
            
        except Exception:
            self.handleError(record)

@contextmanager
def log_execution_time(logging_system: LoggingSystem, operation_name: str, component: str):
    """Context manager para medir tempo de execução"""
    start_time = time.time()
    try:
        yield
    finally:
        execution_time = (time.time() - start_time) * 1000  # ms
        logging_system.log(
            LogLevel.DEBUG,
            EventType.PERFORMANCE_METRIC,
            f"Operação '{operation_name}' executada em {execution_time:.2f}ms",
            component,
            execution_time_ms=execution_time
        )

# Singleton para acesso global
_logging_system_instance = None

def get_logging_system() -> LoggingSystem:
    """Retorna instância singleton do LoggingSystem"""
    global _logging_system_instance
    if _logging_system_instance is None:
        _logging_system_instance = LoggingSystem()
        _logging_system_instance.start()
    return _logging_system_instance

# Exemplo de uso
if __name__ == "__main__":
    print("Logging System - Sistema de Logs e Monitoramento MT5")
    
    # Criar sistema
    logging_sys = LoggingSystem()
    logging_sys.start()
    
    # Exemplos de uso
    logging_sys.log(LogLevel.INFO, EventType.SYSTEM_START, "Sistema iniciado", "main")
    
    # Evento de trading
    logging_sys.log_trading_event(
        EventType.ORDER_SENT,
        "EURUSD",
        "BUY",
        {"volume": 0.01, "price": 1.1234},
        "SUCCESS",
        execution_time_ms=15.5
    )
    
    # Usar context manager
    with log_execution_time(logging_sys, "test_operation", "main"):
        time.sleep(0.1)  # Simular operação
    
    # Aguardar um pouco para coletar métricas
    time.sleep(2)
    
    # Mostrar estatísticas
    stats = logging_sys.get_stats()
    print(f"Estatísticas: {stats}")
    
    # Mostrar métricas
    metrics = logging_sys.get_performance_metrics()
    print(f"Métricas: {metrics}")
    
    # Parar sistema
    logging_sys.stop()
    
    print("\nFuncionalidades implementadas:")
    print("- Logging estruturado com SQLite")
    print("- Monitoramento de performance em tempo real")
    print("- Registro de eventos de trading")
    print("- Métricas de sistema (CPU, memória, threads)")
    print("- Context manager para tempo de execução")
    print("- Callbacks customizáveis")
    print("- Limpeza automática de logs antigos")
    print("- Buffer de memória para logs recentes")
    print("- Integração com logging padrão do Python")
    print("- Acesso singleton global")