#!/usr/bin/env python3
"""
Sistema de Monitoramento em Produção
Monitoramento contínuo das 11 funcionalidades ML em ambiente de produção
"""

import os
import sys
import time
import json
import logging
import psutil
import sqlite3
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
# Email functionality disabled for compatibility
# import smtplib
# from email.mime.text import MimeText
# from email.mime.multipart import MimeMultipart
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MonitoringAlert:
    """Estrutura para alertas de monitoramento"""
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    component: str
    metric: str
    current_value: float
    threshold: float
    message: str
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class PerformanceMetrics:
    """Métricas de performance de um componente"""
    component_name: str
    timestamp: datetime
    execution_time: float
    memory_usage: float
    cpu_usage: float
    success_rate: float
    error_count: int
    throughput: float
    custom_metrics: Dict[str, float]

class ProductionMonitoringSystem:
    def __init__(self, config_path: str = None):
        self.config_path = Path(config_path) if config_path else Path.cwd() / "monitoring_config.json"
        self.monitoring_db = Path.cwd() / "monitoring" / "production_monitoring.db"
        self.monitoring_db.parent.mkdir(exist_ok=True)
        
        # Configurar logging
        log_path = Path.cwd() / "monitoring" / "logs"
        log_path.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / 'production_monitoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Carregar configuração
        self.config = self._load_config()
        
        # Inicializar banco de dados
        self._init_database()
        
        # Buffers para métricas em tempo real
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.alerts_buffer = deque(maxlen=500)
        
        # Status do sistema
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Componentes ML a monitorar
        self.ml_components = [
            "triple_barrier_labeling",
            "purged_cross_validation",
            "kelly_position_sizing",
            "model_drift_detection",
            "dynamic_model_selection",
            "advanced_feature_engineering",
            "fractional_differentiation",
            "walk_forward_analysis",
            "risk_budgeting",
            "sentiment_analysis",
            "cross_asset_signals"
        ]
        
        # Contadores de performance
        self.performance_counters = defaultdict(lambda: {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_execution_time': 0.0,
            'last_execution': None
        })
    
    def _load_config(self) -> Dict:
        """Carrega configuração de monitoramento"""
        default_config = {
            "monitoring_interval": 30,  # segundos
            "alert_thresholds": {
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "execution_time": 30.0,
                "error_rate": 0.05,
                "success_rate": 0.95
            },
            "retention_days": 30,
            "email_alerts": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "recipients": []
            },
            "dashboard": {
                "enabled": True,
                "port": 8080,
                "refresh_interval": 5
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge com configuração padrão
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                self.logger.warning(f"Erro ao carregar configuração: {e}. Usando configuração padrão.")
        
        # Salvar configuração padrão
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _init_database(self):
        """Inicializa banco de dados de monitoramento"""
        conn = sqlite3.connect(self.monitoring_db)
        cursor = conn.cursor()
        
        # Tabela de métricas de performance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                execution_time REAL,
                memory_usage REAL,
                cpu_usage REAL,
                success_rate REAL,
                error_count INTEGER,
                throughput REAL,
                custom_metrics TEXT
            )
        """)
        
        # Tabela de alertas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS monitoring_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                severity TEXT NOT NULL,
                component TEXT NOT NULL,
                metric TEXT NOT NULL,
                current_value REAL,
                threshold REAL,
                message TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_time DATETIME
            )
        """)
        
        # Tabela de eventos do sistema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                event_type TEXT NOT NULL,
                component TEXT,
                description TEXT,
                metadata TEXT
            )
        """)
        
        # Índices para performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_component ON performance_metrics(component_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON monitoring_alerts(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp)")
        
        conn.commit()
        conn.close()
        
        self.logger.info("Banco de dados de monitoramento inicializado")
    
    def start_monitoring(self):
        """Inicia o monitoramento em produção"""
        if self.monitoring_active:
            self.logger.warning("Monitoramento já está ativo")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Sistema de monitoramento iniciado")
        self._log_system_event("monitoring_started", "system", "Sistema de monitoramento iniciado")
    
    def stop_monitoring(self):
        """Para o monitoramento"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Sistema de monitoramento parado")
        self._log_system_event("monitoring_stopped", "system", "Sistema de monitoramento parado")
    
    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        while self.monitoring_active:
            try:
                # Coletar métricas do sistema
                self._collect_system_metrics()
                
                # Coletar métricas dos componentes ML
                self._collect_ml_metrics()
                
                # Verificar alertas
                self._check_alerts()
                
                # Limpeza de dados antigos
                self._cleanup_old_data()
                
                # Aguardar próximo ciclo
                time.sleep(self.config["monitoring_interval"])
                
            except Exception as e:
                self.logger.error(f"Erro no loop de monitoramento: {e}")
                time.sleep(5)  # Aguardar antes de tentar novamente
    
    def _collect_system_metrics(self):
        """Coleta métricas do sistema"""
        try:
            # Métricas de CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Métricas de memória
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Métricas de disco
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Métricas de rede (se disponível)
            try:
                network = psutil.net_io_counters()
                network_sent = network.bytes_sent
                network_recv = network.bytes_recv
            except:
                network_sent = network_recv = 0
            
            # Armazenar métricas
            system_metrics = PerformanceMetrics(
                component_name="system",
                timestamp=datetime.now(),
                execution_time=0.0,
                memory_usage=memory_percent,
                cpu_usage=cpu_percent,
                success_rate=1.0,
                error_count=0,
                throughput=0.0,
                custom_metrics={
                    "disk_usage": disk_percent,
                    "network_sent": network_sent,
                    "network_recv": network_recv,
                    "memory_available_gb": memory.available / (1024**3)
                }
            )
            
            self._store_metrics(system_metrics)
            
        except Exception as e:
            self.logger.error(f"Erro ao coletar métricas do sistema: {e}")
    
    def _collect_ml_metrics(self):
        """Coleta métricas dos componentes ML"""
        for component in self.ml_components:
            try:
                counter = self.performance_counters[component]
                
                # Calcular métricas
                success_rate = (
                    counter['successful_calls'] / max(counter['total_calls'], 1)
                )
                
                avg_execution_time = (
                    counter['total_execution_time'] / max(counter['successful_calls'], 1)
                )
                
                error_rate = (
                    counter['failed_calls'] / max(counter['total_calls'], 1)
                )
                
                # Criar métricas
                ml_metrics = PerformanceMetrics(
                    component_name=component,
                    timestamp=datetime.now(),
                    execution_time=avg_execution_time,
                    memory_usage=0.0,  # Será atualizado durante execução
                    cpu_usage=0.0,     # Será atualizado durante execução
                    success_rate=success_rate,
                    error_count=counter['failed_calls'],
                    throughput=counter['total_calls'] / max(self.config["monitoring_interval"], 1),
                    custom_metrics={
                        "total_calls": counter['total_calls'],
                        "error_rate": error_rate,
                        "last_execution": counter['last_execution'].isoformat() if counter['last_execution'] else None
                    }
                )
                
                self._store_metrics(ml_metrics)
                
            except Exception as e:
                self.logger.error(f"Erro ao coletar métricas de {component}: {e}")
    
    def _store_metrics(self, metrics: PerformanceMetrics):
        """Armazena métricas no banco de dados"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO performance_metrics (
                    component_name, timestamp, execution_time, memory_usage,
                    cpu_usage, success_rate, error_count, throughput, custom_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.component_name,
                metrics.timestamp,
                metrics.execution_time,
                metrics.memory_usage,
                metrics.cpu_usage,
                metrics.success_rate,
                metrics.error_count,
                metrics.throughput,
                json.dumps(metrics.custom_metrics)
            ))
            
            conn.commit()
            conn.close()
            
            # Adicionar ao buffer para dashboard
            self.metrics_buffer[metrics.component_name].append(metrics)
            
        except Exception as e:
            self.logger.error(f"Erro ao armazenar métricas: {e}")
    
    def _check_alerts(self):
        """Verifica condições de alerta"""
        try:
            # Verificar métricas do sistema
            latest_system_metrics = self._get_latest_metrics("system")
            if latest_system_metrics:
                self._check_system_alerts(latest_system_metrics)
            
            # Verificar métricas dos componentes ML
            for component in self.ml_components:
                latest_metrics = self._get_latest_metrics(component)
                if latest_metrics:
                    self._check_component_alerts(component, latest_metrics)
            
        except Exception as e:
            self.logger.error(f"Erro ao verificar alertas: {e}")
    
    def _check_system_alerts(self, metrics: PerformanceMetrics):
        """Verifica alertas do sistema"""
        thresholds = self.config["alert_thresholds"]
        
        # Alerta de CPU
        if metrics.cpu_usage > thresholds["cpu_usage"]:
            self._create_alert(
                "high", "system", "cpu_usage",
                metrics.cpu_usage, thresholds["cpu_usage"],
                f"Uso de CPU alto: {metrics.cpu_usage:.1f}%"
            )
        
        # Alerta de memória
        if metrics.memory_usage > thresholds["memory_usage"]:
            self._create_alert(
                "high", "system", "memory_usage",
                metrics.memory_usage, thresholds["memory_usage"],
                f"Uso de memória alto: {metrics.memory_usage:.1f}%"
            )
        
        # Alerta de disco (se disponível)
        if "disk_usage" in metrics.custom_metrics:
            disk_usage = metrics.custom_metrics["disk_usage"]
            if disk_usage > 90.0:
                self._create_alert(
                    "critical", "system", "disk_usage",
                    disk_usage, 90.0,
                    f"Uso de disco crítico: {disk_usage:.1f}%"
                )
    
    def _check_component_alerts(self, component: str, metrics: PerformanceMetrics):
        """Verifica alertas de componentes ML"""
        thresholds = self.config["alert_thresholds"]
        
        # Alerta de tempo de execução
        if metrics.execution_time > thresholds["execution_time"]:
            self._create_alert(
                "medium", component, "execution_time",
                metrics.execution_time, thresholds["execution_time"],
                f"Tempo de execução alto em {component}: {metrics.execution_time:.2f}s"
            )
        
        # Alerta de taxa de sucesso
        if metrics.success_rate < thresholds["success_rate"]:
            self._create_alert(
                "high", component, "success_rate",
                metrics.success_rate, thresholds["success_rate"],
                f"Taxa de sucesso baixa em {component}: {metrics.success_rate:.2%}"
            )
        
        # Alerta de taxa de erro
        if "error_rate" in metrics.custom_metrics:
            error_rate = metrics.custom_metrics["error_rate"]
            if error_rate > thresholds["error_rate"]:
                self._create_alert(
                    "high", component, "error_rate",
                    error_rate, thresholds["error_rate"],
                    f"Taxa de erro alta em {component}: {error_rate:.2%}"
                )
    
    def _create_alert(self, severity: str, component: str, metric: str,
                     current_value: float, threshold: float, message: str):
        """Cria um novo alerta"""
        alert = MonitoringAlert(
            timestamp=datetime.now(),
            severity=severity,
            component=component,
            metric=metric,
            current_value=current_value,
            threshold=threshold,
            message=message
        )
        
        # Verificar se alerta similar já existe (evitar spam)
        if not self._is_duplicate_alert(alert):
            self._store_alert(alert)
            self.alerts_buffer.append(alert)
            
            # Log do alerta
            self.logger.warning(f"ALERTA [{severity.upper()}]: {message}")
            
            # Enviar email se configurado
            if self.config["email_alerts"]["enabled"]:
                self._send_email_alert(alert)
    
    def _is_duplicate_alert(self, new_alert: MonitoringAlert) -> bool:
        """Verifica se é um alerta duplicado"""
        # Verificar alertas recentes (últimos 10 minutos)
        recent_time = datetime.now() - timedelta(minutes=10)
        
        for alert in reversed(self.alerts_buffer):
            if alert.timestamp < recent_time:
                break
            
            if (alert.component == new_alert.component and
                alert.metric == new_alert.metric and
                alert.severity == new_alert.severity and
                not alert.resolved):
                return True
        
        return False
    
    def _store_alert(self, alert: MonitoringAlert):
        """Armazena alerta no banco de dados"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO monitoring_alerts (
                    timestamp, severity, component, metric, current_value,
                    threshold, message, resolved, resolution_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.timestamp,
                alert.severity,
                alert.component,
                alert.metric,
                alert.current_value,
                alert.threshold,
                alert.message,
                alert.resolved,
                alert.resolution_time
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Erro ao armazenar alerta: {e}")
    
    def _send_email_alert(self, alert: MonitoringAlert):
        """Envia alerta por email"""
        try:
            email_config = self.config["email_alerts"]
            
            if not email_config["recipients"]:
                return
            
            # Criar mensagem
            msg = MimeMultipart()
            msg['From'] = email_config["username"]
            msg['To'] = ", ".join(email_config["recipients"])
            msg['Subject'] = f"[ML Forex] Alerta {alert.severity.upper()}: {alert.component}"
            
            body = f"""
            Alerta de Monitoramento - ML Forex System
            
            Severidade: {alert.severity.upper()}
            Componente: {alert.component}
            Métrica: {alert.metric}
            Valor Atual: {alert.current_value}
            Limite: {alert.threshold}
            
            Mensagem: {alert.message}
            
            Timestamp: {alert.timestamp.strftime('%d/%m/%Y %H:%M:%S')}
            
            ---
            Sistema de Monitoramento ML Forex
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Enviar email
            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            server.starttls()
            server.login(email_config["username"], email_config["password"])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email de alerta enviado para {len(email_config['recipients'])} destinatários")
            
        except Exception as e:
            self.logger.error(f"Erro ao enviar email de alerta: {e}")
    
    def _get_latest_metrics(self, component: str) -> Optional[PerformanceMetrics]:
        """Obtém as métricas mais recentes de um componente"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM performance_metrics 
                WHERE component_name = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (component,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return PerformanceMetrics(
                    component_name=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    execution_time=row[3],
                    memory_usage=row[4],
                    cpu_usage=row[5],
                    success_rate=row[6],
                    error_count=row[7],
                    throughput=row[8],
                    custom_metrics=json.loads(row[9]) if row[9] else {}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro ao obter métricas de {component}: {e}")
            return None
    
    def _cleanup_old_data(self):
        """Remove dados antigos do banco"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config["retention_days"])
            
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()
            
            # Limpar métricas antigas
            cursor.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff_date,))
            
            # Limpar alertas resolvidos antigos
            cursor.execute("""
                DELETE FROM monitoring_alerts 
                WHERE timestamp < ? AND resolved = TRUE
            """, (cutoff_date,))
            
            # Limpar eventos antigos
            cursor.execute("DELETE FROM system_events WHERE timestamp < ?", (cutoff_date,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Erro na limpeza de dados antigos: {e}")
    
    def _log_system_event(self, event_type: str, component: str, description: str, metadata: Dict = None):
        """Registra evento do sistema"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO system_events (timestamp, event_type, component, description, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                event_type,
                component,
                description,
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Erro ao registrar evento: {e}")
    
    # Métodos para instrumentação dos componentes ML
    
    def instrument_ml_component(self, component_name: str):
        """Decorator para instrumentar componentes ML"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                try:
                    # Executar função
                    result = func(*args, **kwargs)
                    
                    # Registrar sucesso
                    execution_time = time.time() - start_time
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    
                    self.performance_counters[component_name]['total_calls'] += 1
                    self.performance_counters[component_name]['successful_calls'] += 1
                    self.performance_counters[component_name]['total_execution_time'] += execution_time
                    self.performance_counters[component_name]['last_execution'] = datetime.now()
                    
                    # Log de performance
                    self.logger.debug(
                        f"{component_name} executado com sucesso em {execution_time:.3f}s, "
                        f"memória: {end_memory - start_memory:.1f}MB"
                    )
                    
                    return result
                    
                except Exception as e:
                    # Registrar falha
                    execution_time = time.time() - start_time
                    
                    self.performance_counters[component_name]['total_calls'] += 1
                    self.performance_counters[component_name]['failed_calls'] += 1
                    self.performance_counters[component_name]['last_execution'] = datetime.now()
                    
                    # Log de erro
                    self.logger.error(f"Erro em {component_name}: {e}")
                    
                    # Criar alerta de erro
                    self._create_alert(
                        "high", component_name, "execution_error",
                        1.0, 0.0,
                        f"Erro de execução em {component_name}: {str(e)[:100]}"
                    )
                    
                    raise
            
            return wrapper
        return decorator
    
    def get_dashboard_data(self) -> Dict:
        """Obtém dados para dashboard de monitoramento"""
        try:
            # Métricas recentes
            recent_metrics = {}
            for component in ["system"] + self.ml_components:
                metrics = list(self.metrics_buffer[component])[-10:]  # Últimas 10 métricas
                recent_metrics[component] = [asdict(m) for m in metrics]
            
            # Alertas ativos
            active_alerts = [asdict(alert) for alert in self.alerts_buffer if not alert.resolved]
            
            # Estatísticas gerais
            stats = {
                "total_components": len(self.ml_components),
                "active_alerts": len(active_alerts),
                "monitoring_uptime": self._get_monitoring_uptime(),
                "system_health": self._calculate_system_health()
            }
            
            return {
                "metrics": recent_metrics,
                "alerts": active_alerts,
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao obter dados do dashboard: {e}")
            return {"error": str(e)}
    
    def _get_monitoring_uptime(self) -> float:
        """Calcula tempo de atividade do monitoramento"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp FROM system_events 
                WHERE event_type = 'monitoring_started' 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                start_time = datetime.fromisoformat(row[0])
                uptime = (datetime.now() - start_time).total_seconds() / 3600  # horas
                return uptime
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular uptime: {e}")
            return 0.0
    
    def _calculate_system_health(self) -> str:
        """Calcula saúde geral do sistema"""
        try:
            # Contar alertas por severidade
            critical_alerts = sum(1 for alert in self.alerts_buffer 
                                if alert.severity == "critical" and not alert.resolved)
            high_alerts = sum(1 for alert in self.alerts_buffer 
                            if alert.severity == "high" and not alert.resolved)
            
            # Determinar saúde
            if critical_alerts > 0:
                return "critical"
            elif high_alerts > 2:
                return "warning"
            elif high_alerts > 0:
                return "degraded"
            else:
                return "healthy"
                
        except Exception as e:
            self.logger.error(f"Erro ao calcular saúde do sistema: {e}")
            return "unknown"
    
    def generate_monitoring_report(self, hours: int = 24) -> Dict:
        """Gera relatório de monitoramento"""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()
            
            # Métricas do período
            cursor.execute("""
                SELECT component_name, 
                       AVG(execution_time) as avg_exec_time,
                       AVG(memory_usage) as avg_memory,
                       AVG(cpu_usage) as avg_cpu,
                       AVG(success_rate) as avg_success_rate,
                       SUM(error_count) as total_errors,
                       COUNT(*) as total_measurements
                FROM performance_metrics 
                WHERE timestamp >= ? 
                GROUP BY component_name
            """, (start_time,))
            
            metrics_summary = {}
            for row in cursor.fetchall():
                metrics_summary[row[0]] = {
                    "avg_execution_time": row[1],
                    "avg_memory_usage": row[2],
                    "avg_cpu_usage": row[3],
                    "avg_success_rate": row[4],
                    "total_errors": row[5],
                    "total_measurements": row[6]
                }
            
            # Alertas do período
            cursor.execute("""
                SELECT severity, COUNT(*) as count
                FROM monitoring_alerts 
                WHERE timestamp >= ? 
                GROUP BY severity
            """, (start_time,))
            
            alerts_summary = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "period_hours": hours,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "metrics_summary": metrics_summary,
                "alerts_summary": alerts_summary,
                "system_health": self._calculate_system_health()
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar relatório: {e}")
            return {"error": str(e)}

def main():
    """Função principal para executar o sistema de monitoramento"""
    monitor = ProductionMonitoringSystem()
    
    print("=== Sistema de Monitoramento em Produção - ML Forex ===")
    print("\nIniciando monitoramento...")
    
    try:
        # Iniciar monitoramento
        monitor.start_monitoring()
        
        print("\nMonitoramento ativo. Pressione Ctrl+C para parar.")
        print("\nComandos disponíveis:")
        print("  status  - Mostrar status atual")
        print("  alerts  - Mostrar alertas ativos")
        print("  report  - Gerar relatório")
        print("  quit    - Parar monitoramento")
        
        # Loop de comandos
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == "status":
                    dashboard_data = monitor.get_dashboard_data()
                    print(f"\nStatus do Sistema: {dashboard_data['stats']['system_health'].upper()}")
                    print(f"Alertas Ativos: {dashboard_data['stats']['active_alerts']}")
                    print(f"Uptime: {dashboard_data['stats']['monitoring_uptime']:.1f} horas")
                
                elif command == "alerts":
                    dashboard_data = monitor.get_dashboard_data()
                    alerts = dashboard_data['alerts']
                    if alerts:
                        print(f"\n{len(alerts)} alertas ativos:")
                        for alert in alerts[-5:]:  # Últimos 5
                            print(f"  [{alert['severity'].upper()}] {alert['component']}: {alert['message']}")
                    else:
                        print("\nNenhum alerta ativo")
                
                elif command == "report":
                    report = monitor.generate_monitoring_report(24)
                    print("\nRelatório das últimas 24 horas:")
                    print(f"Saúde do Sistema: {report['system_health'].upper()}")
                    if 'alerts_summary' in report:
                        print(f"Alertas: {report['alerts_summary']}")
                
                elif command == "quit":
                    break
                
                else:
                    print("Comando não reconhecido")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        # Parar monitoramento
        print("\nParando monitoramento...")
        monitor.stop_monitoring()
        print("Monitoramento parado.")
        
    except KeyboardInterrupt:
        print("\nParando monitoramento...")
        monitor.stop_monitoring()
        print("Monitoramento parado.")
    
    except Exception as e:
        print(f"Erro: {e}")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()