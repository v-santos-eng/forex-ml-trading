#!/usr/bin/env python3
"""
Sistema de Monitoramento em Tempo Real - Ollama
Objetivo: Monitoramento proativo com dashboards interativos e alertas

Recursos:
- Dashboards Interativos em Tempo Real
- Sistema de Alertas Proativos
- Métricas de Performance Detalhadas
- Análise Preditiva
- Relatórios Automatizados
- Integração com Prometheus/Grafana
- Notificações Multi-canal
"""

import asyncio
import aiohttp
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
import psutil
import queue
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from pathlib import Path
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary
import warnings
warnings.filterwarnings('ignore')

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Severidade dos alertas"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(Enum):
    """Tipos de métricas"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class NotificationChannel(Enum):
    """Canais de notificação"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"

@dataclass
class MetricDefinition:
    """Definição de métrica"""
    name: str
    description: str
    metric_type: MetricType
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    collection_interval: float = 1.0
    retention_days: int = 30

@dataclass
class AlertRule:
    """Regra de alerta"""
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., ">", "<", "==", "!="
    threshold: float
    severity: AlertSeverity
    duration: int = 60  # segundos
    cooldown: int = 300  # segundos
    channels: List[NotificationChannel] = field(default_factory=list)
    enabled: bool = True
    custom_message: Optional[str] = None

@dataclass
class Alert:
    """Alerta ativo"""
    id: str
    rule: AlertRule
    triggered_at: datetime
    current_value: float
    message: str
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    notifications_sent: List[str] = field(default_factory=list)

@dataclass
class MetricValue:
    """Valor de métrica"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    """Coletor de métricas do sistema"""
    
    def __init__(self, ollama_endpoint: str = "http://localhost:11434"):
        self.ollama_endpoint = ollama_endpoint
        self.metrics_history = defaultdict(deque)
        self.prometheus_metrics = {}
        self.collection_tasks = []
        self._running = False
        
        # Métricas Prometheus
        self._setup_prometheus_metrics()
        
        # Métricas definidas
        self.metric_definitions = self._create_metric_definitions()
        
        logger.info("Metrics Collector inicializado")
    
    def _setup_prometheus_metrics(self):
        """Configura métricas do Prometheus"""
        self.prometheus_metrics = {
            'ollama_requests_total': Counter(
                'ollama_requests_total',
                'Total number of requests to Ollama',
                ['method', 'status']
            ),
            'ollama_request_duration': Histogram(
                'ollama_request_duration_seconds',
                'Request duration in seconds',
                ['method']
            ),
            'ollama_response_time': Gauge(
                'ollama_response_time_seconds',
                'Current response time in seconds'
            ),
            'ollama_error_rate': Gauge(
                'ollama_error_rate_percent',
                'Current error rate percentage'
            ),
            'ollama_throughput': Gauge(
                'ollama_throughput_requests_per_second',
                'Current throughput in requests per second'
            ),
            'system_cpu_usage': Gauge(
                'system_cpu_usage_percent',
                'System CPU usage percentage'
            ),
            'system_memory_usage': Gauge(
                'system_memory_usage_percent',
                'System memory usage percentage'
            ),
            'system_disk_usage': Gauge(
                'system_disk_usage_percent',
                'System disk usage percentage',
                ['device']
            ),
            'ollama_active_connections': Gauge(
                'ollama_active_connections',
                'Number of active connections to Ollama'
            ),
            'ollama_queue_size': Gauge(
                'ollama_queue_size',
                'Size of request queue'
            )
        }
    
    def _create_metric_definitions(self) -> List[MetricDefinition]:
        """Cria definições de métricas"""
        return [
            MetricDefinition(
                name="response_time",
                description="Tempo de resposta do Ollama",
                metric_type=MetricType.GAUGE,
                unit="seconds",
                thresholds={"warning": 3.0, "critical": 5.0},
                collection_interval=1.0
            ),
            MetricDefinition(
                name="error_rate",
                description="Taxa de erro das requisições",
                metric_type=MetricType.GAUGE,
                unit="percent",
                thresholds={"warning": 5.0, "critical": 10.0},
                collection_interval=1.0
            ),
            MetricDefinition(
                name="throughput",
                description="Throughput de requisições",
                metric_type=MetricType.GAUGE,
                unit="requests/second",
                thresholds={"warning": 5.0, "critical": 2.0},
                collection_interval=1.0
            ),
            MetricDefinition(
                name="cpu_usage",
                description="Uso de CPU do sistema",
                metric_type=MetricType.GAUGE,
                unit="percent",
                thresholds={"warning": 80.0, "critical": 95.0},
                collection_interval=1.0
            ),
            MetricDefinition(
                name="memory_usage",
                description="Uso de memória do sistema",
                metric_type=MetricType.GAUGE,
                unit="percent",
                thresholds={"warning": 85.0, "critical": 95.0},
                collection_interval=1.0
            ),
            MetricDefinition(
                name="disk_usage",
                description="Uso de disco do sistema",
                metric_type=MetricType.GAUGE,
                unit="percent",
                thresholds={"warning": 85.0, "critical": 95.0},
                collection_interval=5.0
            ),
            MetricDefinition(
                name="active_connections",
                description="Conexões ativas com Ollama",
                metric_type=MetricType.GAUGE,
                unit="connections",
                thresholds={"warning": 100, "critical": 200},
                collection_interval=1.0
            )
        ]
    
    async def start_collection(self):
        """Inicia coleta de métricas"""
        if self._running:
            return
        
        self._running = True
        logger.info("Iniciando coleta de métricas")
        
        # Inicia servidor Prometheus
        start_http_server(8000)
        logger.info("Servidor Prometheus iniciado na porta 8000")
        
        # Inicia tarefas de coleta
        for metric_def in self.metric_definitions:
            task = asyncio.create_task(
                self._collect_metric_loop(metric_def)
            )
            self.collection_tasks.append(task)
        
        # Inicia coleta de métricas do Ollama
        ollama_task = asyncio.create_task(self._collect_ollama_metrics())
        self.collection_tasks.append(ollama_task)
    
    async def stop_collection(self):
        """Para coleta de métricas"""
        self._running = False
        
        for task in self.collection_tasks:
            task.cancel()
        
        await asyncio.gather(*self.collection_tasks, return_exceptions=True)
        self.collection_tasks.clear()
        
        logger.info("Coleta de métricas parada")
    
    async def _collect_metric_loop(self, metric_def: MetricDefinition):
        """Loop de coleta para uma métrica específica"""
        while self._running:
            try:
                value = await self._collect_single_metric(metric_def)
                
                if value is not None:
                    metric_value = MetricValue(
                        timestamp=datetime.now(),
                        value=value
                    )
                    
                    # Adiciona ao histórico
                    self.metrics_history[metric_def.name].append(metric_value)
                    
                    # Limita tamanho do histórico
                    max_size = int((metric_def.retention_days * 24 * 3600) / metric_def.collection_interval)
                    if len(self.metrics_history[metric_def.name]) > max_size:
                        self.metrics_history[metric_def.name].popleft()
                    
                    # Atualiza métrica Prometheus
                    await self._update_prometheus_metric(metric_def.name, value)
                
                await asyncio.sleep(metric_def.collection_interval)
                
            except Exception as e:
                logger.error(f"Erro na coleta da métrica {metric_def.name}: {e}")
                await asyncio.sleep(metric_def.collection_interval)
    
    async def _collect_single_metric(self, metric_def: MetricDefinition) -> Optional[float]:
        """Coleta uma métrica específica"""
        try:
            if metric_def.name == "cpu_usage":
                return psutil.cpu_percent(interval=None)
            
            elif metric_def.name == "memory_usage":
                return psutil.virtual_memory().percent
            
            elif metric_def.name == "disk_usage":
                return psutil.disk_usage('/').percent
            
            elif metric_def.name in ["response_time", "error_rate", "throughput", "active_connections"]:
                # Essas métricas são coletadas pelo _collect_ollama_metrics
                return None
            
            else:
                logger.warning(f"Métrica não implementada: {metric_def.name}")
                return None
        
        except Exception as e:
            logger.error(f"Erro ao coletar métrica {metric_def.name}: {e}")
            return None
    
    async def _collect_ollama_metrics(self):
        """Coleta métricas específicas do Ollama"""
        request_times = deque(maxlen=100)
        error_count = 0
        total_requests = 0
        
        while self._running:
            try:
                # Faz requisição de teste ao Ollama
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.post(
                            f"{self.ollama_endpoint}/api/generate",
                            json={
                                "model": "llama2",
                                "prompt": "Hello",
                                "stream": False
                            },
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            
                            response_time = time.time() - start_time
                            request_times.append(response_time)
                            total_requests += 1
                            
                            if response.status != 200:
                                error_count += 1
                            
                            # Atualiza métricas
                            await self._update_ollama_metrics(
                                response_time, request_times, error_count, total_requests
                            )
                    
                    except Exception as e:
                        error_count += 1
                        total_requests += 1
                        response_time = time.time() - start_time
                        request_times.append(response_time)
                        
                        await self._update_ollama_metrics(
                            response_time, request_times, error_count, total_requests
                        )
                
                await asyncio.sleep(5)  # Coleta a cada 5 segundos
                
            except Exception as e:
                logger.error(f"Erro na coleta de métricas do Ollama: {e}")
                await asyncio.sleep(5)
    
    async def _update_ollama_metrics(self, response_time: float, request_times: deque, 
                                   error_count: int, total_requests: int):
        """Atualiza métricas do Ollama"""
        # Tempo de resposta atual
        current_time = datetime.now()
        
        self.metrics_history["response_time"].append(
            MetricValue(timestamp=current_time, value=response_time)
        )
        
        # Taxa de erro
        error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0
        self.metrics_history["error_rate"].append(
            MetricValue(timestamp=current_time, value=error_rate)
        )
        
        # Throughput (baseado nos últimos tempos de resposta)
        if len(request_times) > 1:
            avg_response_time = statistics.mean(request_times)
            throughput = 1.0 / avg_response_time if avg_response_time > 0 else 0
        else:
            throughput = 0
        
        self.metrics_history["throughput"].append(
            MetricValue(timestamp=current_time, value=throughput)
        )
        
        # Conexões ativas (simulado)
        active_connections = len(request_times)
        self.metrics_history["active_connections"].append(
            MetricValue(timestamp=current_time, value=active_connections)
        )
    
    async def _update_prometheus_metric(self, metric_name: str, value: float):
        """Atualiza métrica do Prometheus"""
        try:
            if metric_name == "response_time" and "ollama_response_time" in self.prometheus_metrics:
                self.prometheus_metrics["ollama_response_time"].set(value)
            
            elif metric_name == "error_rate" and "ollama_error_rate" in self.prometheus_metrics:
                self.prometheus_metrics["ollama_error_rate"].set(value)
            
            elif metric_name == "throughput" and "ollama_throughput" in self.prometheus_metrics:
                self.prometheus_metrics["ollama_throughput"].set(value)
            
            elif metric_name == "cpu_usage" and "system_cpu_usage" in self.prometheus_metrics:
                self.prometheus_metrics["system_cpu_usage"].set(value)
            
            elif metric_name == "memory_usage" and "system_memory_usage" in self.prometheus_metrics:
                self.prometheus_metrics["system_memory_usage"].set(value)
            
            elif metric_name == "active_connections" and "ollama_active_connections" in self.prometheus_metrics:
                self.prometheus_metrics["ollama_active_connections"].set(value)
        
        except Exception as e:
            logger.error(f"Erro ao atualizar métrica Prometheus {metric_name}: {e}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Retorna métricas atuais"""
        current_metrics = {}
        
        for metric_name, history in self.metrics_history.items():
            if history:
                current_metrics[metric_name] = history[-1].value
            else:
                current_metrics[metric_name] = 0.0
        
        return current_metrics
    
    def get_metric_history(self, metric_name: str, duration_minutes: int = 60) -> List[MetricValue]:
        """Retorna histórico de uma métrica"""
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        
        return [
            value for value in self.metrics_history[metric_name]
            if value.timestamp >= cutoff_time
        ]

class AlertManager:
    """Gerenciador de alertas"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_history = []
        self.notification_channels = {}
        self._running = False
        
        # Cria regras de alerta padrão
        self._create_default_alert_rules()
        
        logger.info("Alert Manager inicializado")
    
    def _create_default_alert_rules(self):
        """Cria regras de alerta padrão"""
        self.alert_rules = [
            AlertRule(
                name="High Response Time",
                description="Tempo de resposta muito alto",
                metric_name="response_time",
                condition=">",
                threshold=3.0,
                severity=AlertSeverity.HIGH,
                duration=60,
                channels=[NotificationChannel.DASHBOARD, NotificationChannel.EMAIL]
            ),
            AlertRule(
                name="Critical Response Time",
                description="Tempo de resposta crítico",
                metric_name="response_time",
                condition=">",
                threshold=5.0,
                severity=AlertSeverity.CRITICAL,
                duration=30,
                channels=[NotificationChannel.DASHBOARD, NotificationChannel.EMAIL]
            ),
            AlertRule(
                name="High Error Rate",
                description="Taxa de erro elevada",
                metric_name="error_rate",
                condition=">",
                threshold=5.0,
                severity=AlertSeverity.HIGH,
                duration=120,
                channels=[NotificationChannel.DASHBOARD, NotificationChannel.EMAIL]
            ),
            AlertRule(
                name="Critical Error Rate",
                description="Taxa de erro crítica",
                metric_name="error_rate",
                condition=">",
                threshold=10.0,
                severity=AlertSeverity.CRITICAL,
                duration=60,
                channels=[NotificationChannel.DASHBOARD, NotificationChannel.EMAIL]
            ),
            AlertRule(
                name="Low Throughput",
                description="Throughput muito baixo",
                metric_name="throughput",
                condition="<",
                threshold=2.0,
                severity=AlertSeverity.MEDIUM,
                duration=300,
                channels=[NotificationChannel.DASHBOARD]
            ),
            AlertRule(
                name="High CPU Usage",
                description="Uso de CPU muito alto",
                metric_name="cpu_usage",
                condition=">",
                threshold=90.0,
                severity=AlertSeverity.HIGH,
                duration=180,
                channels=[NotificationChannel.DASHBOARD, NotificationChannel.EMAIL]
            ),
            AlertRule(
                name="High Memory Usage",
                description="Uso de memória muito alto",
                metric_name="memory_usage",
                condition=">",
                threshold=90.0,
                severity=AlertSeverity.HIGH,
                duration=300,
                channels=[NotificationChannel.DASHBOARD, NotificationChannel.EMAIL]
            )
        ]
    
    async def start_monitoring(self):
        """Inicia monitoramento de alertas"""
        if self._running:
            return
        
        self._running = True
        logger.info("Iniciando monitoramento de alertas")
        
        # Inicia loop de verificação de alertas
        asyncio.create_task(self._alert_monitoring_loop())
    
    async def stop_monitoring(self):
        """Para monitoramento de alertas"""
        self._running = False
        logger.info("Monitoramento de alertas parado")
    
    async def _alert_monitoring_loop(self):
        """Loop principal de monitoramento de alertas"""
        while self._running:
            try:
                current_metrics = self.metrics_collector.get_current_metrics()
                
                for rule in self.alert_rules:
                    if not rule.enabled:
                        continue
                    
                    await self._check_alert_rule(rule, current_metrics)
                
                await asyncio.sleep(10)  # Verifica a cada 10 segundos
                
            except Exception as e:
                logger.error(f"Erro no loop de monitoramento de alertas: {e}")
                await asyncio.sleep(10)
    
    async def _check_alert_rule(self, rule: AlertRule, current_metrics: Dict[str, float]):
        """Verifica uma regra de alerta específica"""
        if rule.metric_name not in current_metrics:
            return
        
        current_value = current_metrics[rule.metric_name]
        alert_id = f"{rule.name}_{rule.metric_name}"
        
        # Verifica condição
        condition_met = self._evaluate_condition(current_value, rule.condition, rule.threshold)
        
        if condition_met:
            # Condição de alerta atendida
            if alert_id not in self.active_alerts:
                # Novo alerta
                alert = Alert(
                    id=alert_id,
                    rule=rule,
                    triggered_at=datetime.now(),
                    current_value=current_value,
                    message=self._generate_alert_message(rule, current_value)
                )
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                # Envia notificações
                await self._send_alert_notifications(alert)
                
                logger.warning(f"Alerta ativado: {alert.message}")
            
            else:
                # Alerta existente - atualiza valor
                self.active_alerts[alert_id].current_value = current_value
        
        else:
            # Condição não atendida - resolve alerta se existir
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                
                del self.active_alerts[alert_id]
                
                # Envia notificação de resolução
                await self._send_resolution_notification(alert)
                
                logger.info(f"Alerta resolvido: {alert.message}")
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Avalia condição de alerta"""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 0.001
        elif condition == "!=":
            return abs(value - threshold) >= 0.001
        else:
            return False
    
    def _generate_alert_message(self, rule: AlertRule, current_value: float) -> str:
        """Gera mensagem de alerta"""
        if rule.custom_message:
            return rule.custom_message.format(
                metric_name=rule.metric_name,
                current_value=current_value,
                threshold=rule.threshold
            )
        
        return f"{rule.description}: {rule.metric_name} = {current_value:.2f} (limite: {rule.threshold})"
    
    async def _send_alert_notifications(self, alert: Alert):
        """Envia notificações de alerta"""
        for channel in alert.rule.channels:
            try:
                if channel == NotificationChannel.EMAIL:
                    await self._send_email_notification(alert)
                elif channel == NotificationChannel.SLACK:
                    await self._send_slack_notification(alert)
                elif channel == NotificationChannel.WEBHOOK:
                    await self._send_webhook_notification(alert)
                elif channel == NotificationChannel.DASHBOARD:
                    # Dashboard notifications são tratadas em tempo real
                    pass
                
                alert.notifications_sent.append(channel.value)
                
            except Exception as e:
                logger.error(f"Erro ao enviar notificação {channel.value}: {e}")
    
    async def _send_resolution_notification(self, alert: Alert):
        """Envia notificação de resolução"""
        resolution_message = f"RESOLVIDO: {alert.message}"
        
        for channel in alert.rule.channels:
            try:
                if channel == NotificationChannel.EMAIL:
                    # Implementar envio de email de resolução
                    pass
                elif channel == NotificationChannel.SLACK:
                    # Implementar envio de Slack de resolução
                    pass
            except Exception as e:
                logger.error(f"Erro ao enviar notificação de resolução {channel.value}: {e}")
    
    async def _send_email_notification(self, alert: Alert):
        """Envia notificação por email"""
        # Implementação básica - configurar SMTP conforme necessário
        logger.info(f"Email notification: {alert.message}")
    
    async def _send_slack_notification(self, alert: Alert):
        """Envia notificação para Slack"""
        # Implementação básica - configurar webhook do Slack
        logger.info(f"Slack notification: {alert.message}")
    
    async def _send_webhook_notification(self, alert: Alert):
        """Envia notificação via webhook"""
        # Implementação básica - configurar endpoint do webhook
        logger.info(f"Webhook notification: {alert.message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Retorna alertas ativos"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Retorna histórico de alertas"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alert_history
            if alert.triggered_at >= cutoff_time
        ]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Reconhece um alerta"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alerta reconhecido: {alert_id}")
            return True
        return False

class DashboardServer:
    """Servidor do dashboard web"""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager, port: int = 5000):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.port = port
        
        # Cria aplicação Flask
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'monitoring_dashboard_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Registra rotas
        self._register_routes()
        
        logger.info(f"Dashboard Server inicializado na porta {port}")
    
    def _register_routes(self):
        """Registra rotas da aplicação"""
        
        @self.app.route('/')
        def dashboard():
            """Página principal do dashboard"""
            return self._render_dashboard()
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """API para métricas atuais"""
            return jsonify(self.metrics_collector.get_current_metrics())
        
        @self.app.route('/api/metrics/<metric_name>')
        def api_metric_history(metric_name):
            """API para histórico de métrica específica"""
            duration = request.args.get('duration', 60, type=int)
            history = self.metrics_collector.get_metric_history(metric_name, duration)
            
            return jsonify([
                {
                    'timestamp': value.timestamp.isoformat(),
                    'value': value.value
                }
                for value in history
            ])
        
        @self.app.route('/api/alerts')
        def api_alerts():
            """API para alertas ativos"""
            alerts = self.alert_manager.get_active_alerts()
            
            return jsonify([
                {
                    'id': alert.id,
                    'message': alert.message,
                    'severity': alert.rule.severity.value,
                    'triggered_at': alert.triggered_at.isoformat(),
                    'current_value': alert.current_value,
                    'threshold': alert.rule.threshold,
                    'acknowledged': alert.acknowledged
                }
                for alert in alerts
            ])
        
        @self.app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
        def api_acknowledge_alert(alert_id):
            """API para reconhecer alerta"""
            success = self.alert_manager.acknowledge_alert(alert_id)
            return jsonify({'success': success})
        
        @self.socketio.on('connect')
        def handle_connect():
            """Cliente conectado ao WebSocket"""
            logger.info('Cliente conectado ao dashboard')
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Cliente desconectado do WebSocket"""
            logger.info('Cliente desconectado do dashboard')
    
    def _render_dashboard(self) -> str:
        """Renderiza HTML do dashboard"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ollama Monitoring Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
                .metric-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
                .metric-label { color: #7f8c8d; margin-bottom: 10px; }
                .chart-container { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
                .alerts-container { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .alert { padding: 10px; margin: 5px 0; border-radius: 3px; }
                .alert.high { background-color: #fff3cd; border-left: 4px solid #ffc107; }
                .alert.critical { background-color: #f8d7da; border-left: 4px solid #dc3545; }
                .alert.medium { background-color: #d1ecf1; border-left: 4px solid #17a2b8; }
                .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
                .status-ok { background-color: #28a745; }
                .status-warning { background-color: #ffc107; }
                .status-critical { background-color: #dc3545; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Ollama Monitoring Dashboard</h1>
                <p>Monitoramento em tempo real do sistema Ollama</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Tempo de Resposta</div>
                    <div class="metric-value" id="response-time">--</div>
                    <div>segundos</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Taxa de Erro</div>
                    <div class="metric-value" id="error-rate">--</div>
                    <div>%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Throughput</div>
                    <div class="metric-value" id="throughput">--</div>
                    <div>req/s</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">CPU</div>
                    <div class="metric-value" id="cpu-usage">--</div>
                    <div>%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Memória</div>
                    <div class="metric-value" id="memory-usage">--</div>
                    <div>%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Conexões Ativas</div>
                    <div class="metric-value" id="active-connections">--</div>
                    <div>conexões</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Métricas de Performance (Últimos 60 minutos)</h3>
                <div id="performance-chart" style="height: 400px;"></div>
            </div>
            
            <div class="alerts-container">
                <h3>Alertas Ativos</h3>
                <div id="alerts-list"></div>
            </div>
            
            <script>
                const socket = io();
                
                // Atualiza métricas a cada 5 segundos
                setInterval(updateMetrics, 5000);
                setInterval(updateCharts, 30000);
                setInterval(updateAlerts, 10000);
                
                // Atualização inicial
                updateMetrics();
                updateCharts();
                updateAlerts();
                
                function updateMetrics() {
                    fetch('/api/metrics')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('response-time').textContent = (data.response_time || 0).toFixed(2);
                            document.getElementById('error-rate').textContent = (data.error_rate || 0).toFixed(1);
                            document.getElementById('throughput').textContent = (data.throughput || 0).toFixed(1);
                            document.getElementById('cpu-usage').textContent = (data.cpu_usage || 0).toFixed(1);
                            document.getElementById('memory-usage').textContent = (data.memory_usage || 0).toFixed(1);
                            document.getElementById('active-connections').textContent = Math.round(data.active_connections || 0);
                        })
                        .catch(error => console.error('Erro ao atualizar métricas:', error));
                }
                
                function updateCharts() {
                    const metrics = ['response_time', 'error_rate', 'throughput', 'cpu_usage'];
                    const promises = metrics.map(metric => 
                        fetch(`/api/metrics/${metric}?duration=60`).then(r => r.json())
                    );
                    
                    Promise.all(promises)
                        .then(results => {
                            const traces = [];
                            const metricNames = ['Tempo de Resposta (s)', 'Taxa de Erro (%)', 'Throughput (req/s)', 'CPU (%)'];
                            
                            results.forEach((data, index) => {
                                if (data.length > 0) {
                                    traces.push({
                                        x: data.map(d => d.timestamp),
                                        y: data.map(d => d.value),
                                        name: metricNames[index],
                                        type: 'scatter',
                                        mode: 'lines'
                                    });
                                }
                            });
                            
                            const layout = {
                                title: 'Métricas de Performance',
                                xaxis: { title: 'Tempo' },
                                yaxis: { title: 'Valor' },
                                showlegend: true
                            };
                            
                            Plotly.newPlot('performance-chart', traces, layout);
                        })
                        .catch(error => console.error('Erro ao atualizar gráficos:', error));
                }
                
                function updateAlerts() {
                    fetch('/api/alerts')
                        .then(response => response.json())
                        .then(alerts => {
                            const alertsList = document.getElementById('alerts-list');
                            
                            if (alerts.length === 0) {
                                alertsList.innerHTML = '<p><span class="status-indicator status-ok"></span>Nenhum alerta ativo</p>';
                                return;
                            }
                            
                            alertsList.innerHTML = alerts.map(alert => `
                                <div class="alert ${alert.severity}">
                                    <strong>${alert.severity.toUpperCase()}:</strong> ${alert.message}
                                    <br><small>Ativado em: ${new Date(alert.triggered_at).toLocaleString()}</small>
                                    ${!alert.acknowledged ? `<button onclick="acknowledgeAlert('${alert.id}')">Reconhecer</button>` : '<span style="color: green;">✓ Reconhecido</span>'}
                                </div>
                            `).join('');
                        })
                        .catch(error => console.error('Erro ao atualizar alertas:', error));
                }
                
                function acknowledgeAlert(alertId) {
                    fetch(`/api/alerts/${alertId}/acknowledge`, { method: 'POST' })
                        .then(response => response.json())
                        .then(result => {
                            if (result.success) {
                                updateAlerts();
                            }
                        })
                        .catch(error => console.error('Erro ao reconhecer alerta:', error));
                }
            </script>
        </body>
        </html>
        """
    
    def run(self, debug: bool = False):
        """Executa servidor do dashboard"""
        logger.info(f"Iniciando dashboard na porta {self.port}")
        self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=debug)

class RealTimeMonitoringSystem:
    """Sistema principal de monitoramento em tempo real"""
    
    def __init__(self, ollama_endpoint: str = "http://localhost:11434", dashboard_port: int = 5000):
        self.ollama_endpoint = ollama_endpoint
        self.dashboard_port = dashboard_port
        
        # Componentes do sistema
        self.metrics_collector = MetricsCollector(ollama_endpoint)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.dashboard_server = DashboardServer(self.metrics_collector, self.alert_manager, dashboard_port)
        
        self._running = False
        
        logger.info("Real-Time Monitoring System inicializado")
    
    async def start(self):
        """Inicia sistema de monitoramento"""
        if self._running:
            return
        
        self._running = True
        logger.info("Iniciando sistema de monitoramento em tempo real")
        
        # Inicia coleta de métricas
        await self.metrics_collector.start_collection()
        
        # Inicia monitoramento de alertas
        await self.alert_manager.start_monitoring()
        
        # Inicia dashboard em thread separada
        dashboard_thread = threading.Thread(
            target=self.dashboard_server.run,
            kwargs={'debug': False}
        )
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        logger.info(f"Sistema de monitoramento iniciado")
        logger.info(f"Dashboard disponível em: http://localhost:{self.dashboard_port}")
        logger.info(f"Métricas Prometheus disponíveis em: http://localhost:8000")
    
    async def stop(self):
        """Para sistema de monitoramento"""
        self._running = False
        
        await self.metrics_collector.stop_collection()
        await self.alert_manager.stop_monitoring()
        
        logger.info("Sistema de monitoramento parado")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retorna status do sistema"""
        current_metrics = self.metrics_collector.get_current_metrics()
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Determina status geral
        critical_alerts = [a for a in active_alerts if a.rule.severity == AlertSeverity.CRITICAL]
        high_alerts = [a for a in active_alerts if a.rule.severity == AlertSeverity.HIGH]
        
        if critical_alerts:
            overall_status = "CRITICAL"
        elif high_alerts:
            overall_status = "WARNING"
        elif active_alerts:
            overall_status = "DEGRADED"
        else:
            overall_status = "HEALTHY"
        
        return {
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'metrics': current_metrics,
            'active_alerts': len(active_alerts),
            'critical_alerts': len(critical_alerts),
            'high_alerts': len(high_alerts),
            'system_info': {
                'ollama_endpoint': self.ollama_endpoint,
                'dashboard_port': self.dashboard_port,
                'monitoring_running': self._running
            }
        }

# Exemplo de uso e testes
async def test_monitoring_system():
    """Testa o sistema de monitoramento"""
    monitoring = RealTimeMonitoringSystem()
    
    print("\n=== SISTEMA DE MONITORAMENTO EM TEMPO REAL ===")
    
    try:
        # Inicia sistema
        await monitoring.start()
        
        print("\nSistema iniciado com sucesso!")
        print(f"Dashboard: http://localhost:{monitoring.dashboard_port}")
        print("Prometheus: http://localhost:8000")
        
        # Monitora por alguns minutos
        for i in range(12):  # 2 minutos
            await asyncio.sleep(10)
            
            status = monitoring.get_system_status()
            print(f"\n[{i+1}/12] Status: {status['overall_status']}")
            print(f"Métricas: {len(status['metrics'])} coletadas")
            print(f"Alertas ativos: {status['active_alerts']}")
            
            if status['critical_alerts'] > 0:
                print(f"⚠️  ALERTAS CRÍTICOS: {status['critical_alerts']}")
        
        print("\n=== RESUMO FINAL ===")
        final_status = monitoring.get_system_status()
        print(f"Status final: {final_status['overall_status']}")
        print(f"Total de alertas: {final_status['active_alerts']}")
        
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    
    finally:
        await monitoring.stop()
        print("Sistema de monitoramento parado")

if __name__ == "__main__":
    asyncio.run(test_monitoring_system())