#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor de Saúde da Conexão MT5
Implementa monitoramento contínuo da qualidade da conexão
Autor: Sistema Forex ML
Data: 2024
"""

import time
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

logger = logging.getLogger(__name__)

class ConnectionStatus(Enum):
    """Status da conexão"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DISCONNECTED = "disconnected"
    UNKNOWN = "unknown"

class AlertLevel(Enum):
    """Nível de alerta"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class HealthMetrics:
    """Métricas de saúde da conexão"""
    timestamp: datetime
    is_connected: bool
    ping_time: Optional[float] = None  # ms
    account_info_time: Optional[float] = None  # ms
    symbol_count: Optional[int] = None
    last_tick_time: Optional[datetime] = None
    server_time: Optional[datetime] = None
    trade_allowed: Optional[bool] = None
    
@dataclass
class HealthAlert:
    """Alerta de saúde"""
    timestamp: datetime
    level: AlertLevel
    message: str
    metrics: Optional[HealthMetrics] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class HealthStatistics:
    """Estatísticas de saúde"""
    avg_ping: Optional[float] = None
    max_ping: Optional[float] = None
    min_ping: Optional[float] = None
    uptime_percentage: float = 0.0
    total_disconnections: int = 0
    avg_reconnect_time: Optional[float] = None
    last_check: Optional[datetime] = None

class ConnectionHealthMonitor:
    """
    Monitor de saúde da conexão MT5
    Realiza verificações periódicas e gera alertas
    """
    
    def __init__(self, check_interval: float = 30.0, 
                 history_size: int = 1000,
                 ping_threshold_warning: float = 1000.0,  # ms
                 ping_threshold_critical: float = 3000.0,  # ms
                 disconnect_threshold: int = 3):  # falhas consecutivas
        
        self.check_interval = check_interval
        self.history_size = history_size
        self.ping_threshold_warning = ping_threshold_warning
        self.ping_threshold_critical = ping_threshold_critical
        self.disconnect_threshold = disconnect_threshold
        
        # Estado do monitor
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Histórico e métricas
        self._metrics_history: deque = deque(maxlen=history_size)
        self._alerts: List[HealthAlert] = []
        self._current_status = ConnectionStatus.UNKNOWN
        self._consecutive_failures = 0
        self._last_successful_check: Optional[datetime] = None
        
        # Callbacks
        self._status_callbacks: List[Callable] = []
        self._alert_callbacks: List[Callable] = []
        
        # Estatísticas
        self._start_time: Optional[datetime] = None
        self._total_checks = 0
        self._successful_checks = 0
        self._disconnection_times: List[datetime] = []
        self._reconnection_times: List[float] = []  # segundos
    
    def start_monitoring(self) -> bool:
        """
        Inicia o monitoramento de saúde
        
        Returns:
            bool: True se iniciado com sucesso
        """
        with self._lock:
            if self._running:
                logger.warning("Monitor de saúde já está em execução")
                return False
            
            if mt5 is None:
                logger.error("MetaTrader5 não está disponível")
                return False
            
            self._running = True
            self._start_time = datetime.now()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="HealthMonitor",
                daemon=True
            )
            self._monitor_thread.start()
            
            logger.info(f"Monitor de saúde iniciado (intervalo: {self.check_interval}s)")
            return True
    
    def stop_monitoring(self) -> bool:
        """
        Para o monitoramento de saúde
        
        Returns:
            bool: True se parado com sucesso
        """
        with self._lock:
            if not self._running:
                return False
            
            self._running = False
            
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
            
            logger.info("Monitor de saúde parado")
            return True
    
    def _monitor_loop(self):
        """
        Loop principal do monitoramento
        """
        logger.info("Iniciando loop de monitoramento de saúde")
        
        while self._running:
            try:
                # Realizar verificação de saúde
                metrics = self._perform_health_check()
                
                with self._lock:
                    # Adicionar ao histórico
                    self._metrics_history.append(metrics)
                    self._total_checks += 1
                    
                    # Analisar métricas e gerar alertas
                    self._analyze_metrics(metrics)
                    
                    # Atualizar status
                    old_status = self._current_status
                    self._current_status = self._determine_status(metrics)
                    
                    # Notificar mudança de status
                    if old_status != self._current_status:
                        self._notify_status_change(old_status, self._current_status, metrics)
                
                # Aguardar próxima verificação
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Erro no loop de monitoramento: {e}")
                time.sleep(min(self.check_interval, 10.0))
    
    def _perform_health_check(self) -> HealthMetrics:
        """
        Realiza uma verificação completa de saúde
        
        Returns:
            HealthMetrics: Métricas coletadas
        """
        timestamp = datetime.now()
        metrics = HealthMetrics(timestamp=timestamp, is_connected=False)
        
        try:
            # Verificar se MT5 está inicializado
            if not mt5.initialize():
                return metrics
            
            # Teste básico de conectividade
            start_time = time.time()
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                return metrics
            
            metrics.is_connected = True
            
            # Medir tempo de resposta para informações da conta
            account_start = time.time()
            account_info = mt5.account_info()
            if account_info:
                metrics.account_info_time = (time.time() - account_start) * 1000
                metrics.trade_allowed = account_info.trade_allowed
            
            # Contar símbolos disponíveis
            symbols = mt5.symbols_get()
            if symbols:
                metrics.symbol_count = len(symbols)
            
            # Obter tempo do servidor
            try:
                # Tentar obter tick mais recente de um símbolo comum
                tick = mt5.symbol_info_tick("EURUSD")
                if tick:
                    metrics.last_tick_time = datetime.fromtimestamp(tick.time)
                    metrics.ping_time = (time.time() - start_time) * 1000
            except:
                pass
            
            # Tempo do servidor (aproximado)
            server_time_info = mt5.terminal_info()
            if server_time_info and hasattr(server_time_info, 'time'):
                metrics.server_time = datetime.fromtimestamp(server_time_info.time)
            
            self._successful_checks += 1
            self._last_successful_check = timestamp
            self._consecutive_failures = 0
            
        except Exception as e:
            logger.error(f"Erro na verificação de saúde: {e}")
            metrics.is_connected = False
            self._consecutive_failures += 1
        
        return metrics
    
    def _analyze_metrics(self, metrics: HealthMetrics):
        """
        Analisa métricas e gera alertas se necessário
        
        Args:
            metrics: Métricas para analisar
        """
        alerts = []
        
        # Verificar desconexão
        if not metrics.is_connected:
            if self._consecutive_failures >= self.disconnect_threshold:
                alerts.append(HealthAlert(
                    timestamp=metrics.timestamp,
                    level=AlertLevel.CRITICAL,
                    message=f"Conexão perdida há {self._consecutive_failures} verificações consecutivas",
                    metrics=metrics
                ))
                
                # Registrar tempo de desconexão
                if self._consecutive_failures == self.disconnect_threshold:
                    self._disconnection_times.append(metrics.timestamp)
        
        # Verificar latência alta
        if metrics.ping_time is not None:
            if metrics.ping_time > self.ping_threshold_critical:
                alerts.append(HealthAlert(
                    timestamp=metrics.timestamp,
                    level=AlertLevel.ERROR,
                    message=f"Latência crítica: {metrics.ping_time:.1f}ms",
                    metrics=metrics
                ))
            elif metrics.ping_time > self.ping_threshold_warning:
                alerts.append(HealthAlert(
                    timestamp=metrics.timestamp,
                    level=AlertLevel.WARNING,
                    message=f"Latência alta: {metrics.ping_time:.1f}ms",
                    metrics=metrics
                ))
        
        # Verificar se trading está desabilitado
        if metrics.trade_allowed is False:
            alerts.append(HealthAlert(
                timestamp=metrics.timestamp,
                level=AlertLevel.WARNING,
                message="Trading não está permitido na conta",
                metrics=metrics
            ))
        
        # Verificar poucos símbolos disponíveis
        if metrics.symbol_count is not None and metrics.symbol_count < 10:
            alerts.append(HealthAlert(
                timestamp=metrics.timestamp,
                level=AlertLevel.WARNING,
                message=f"Poucos símbolos disponíveis: {metrics.symbol_count}",
                metrics=metrics
            ))
        
        # Adicionar alertas à lista
        for alert in alerts:
            self._alerts.append(alert)
            self._notify_alert(alert)
        
        # Limitar histórico de alertas
        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-500:]
    
    def _determine_status(self, metrics: HealthMetrics) -> ConnectionStatus:
        """
        Determina o status geral da conexão
        
        Args:
            metrics: Métricas atuais
        
        Returns:
            ConnectionStatus: Status determinado
        """
        if not metrics.is_connected:
            return ConnectionStatus.DISCONNECTED
        
        # Verificar alertas críticos recentes
        recent_critical = any(
            alert.level == AlertLevel.CRITICAL and 
            not alert.resolved and
            (metrics.timestamp - alert.timestamp).total_seconds() < 300
            for alert in self._alerts[-10:]
        )
        
        if recent_critical:
            return ConnectionStatus.CRITICAL
        
        # Verificar latência
        if metrics.ping_time is not None:
            if metrics.ping_time > self.ping_threshold_critical:
                return ConnectionStatus.CRITICAL
            elif metrics.ping_time > self.ping_threshold_warning:
                return ConnectionStatus.WARNING
        
        # Verificar trading desabilitado
        if metrics.trade_allowed is False:
            return ConnectionStatus.WARNING
        
        return ConnectionStatus.HEALTHY
    
    def _notify_status_change(self, old_status: ConnectionStatus, 
                            new_status: ConnectionStatus, 
                            metrics: HealthMetrics):
        """
        Notifica mudança de status
        
        Args:
            old_status: Status anterior
            new_status: Novo status
            metrics: Métricas atuais
        """
        logger.info(f"Status da conexão mudou: {old_status.value} -> {new_status.value}")
        
        for callback in self._status_callbacks:
            try:
                callback(old_status, new_status, metrics)
            except Exception as e:
                logger.error(f"Erro no callback de status: {e}")
    
    def _notify_alert(self, alert: HealthAlert):
        """
        Notifica novo alerta
        
        Args:
            alert: Alerta gerado
        """
        logger.log(
            logging.ERROR if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL] else logging.WARNING,
            f"Alerta de saúde [{alert.level.value}]: {alert.message}"
        )
        
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Erro no callback de alerta: {e}")
    
    def get_current_status(self) -> ConnectionStatus:
        """
        Retorna o status atual da conexão
        
        Returns:
            ConnectionStatus: Status atual
        """
        with self._lock:
            return self._current_status
    
    def get_latest_metrics(self) -> Optional[HealthMetrics]:
        """
        Retorna as métricas mais recentes
        
        Returns:
            Optional[HealthMetrics]: Métricas mais recentes ou None
        """
        with self._lock:
            return self._metrics_history[-1] if self._metrics_history else None
    
    def get_statistics(self) -> HealthStatistics:
        """
        Calcula e retorna estatísticas de saúde
        
        Returns:
            HealthStatistics: Estatísticas calculadas
        """
        with self._lock:
            stats = HealthStatistics()
            
            if not self._metrics_history:
                return stats
            
            # Calcular estatísticas de ping
            ping_times = [m.ping_time for m in self._metrics_history if m.ping_time is not None]
            if ping_times:
                stats.avg_ping = statistics.mean(ping_times)
                stats.max_ping = max(ping_times)
                stats.min_ping = min(ping_times)
            
            # Calcular uptime
            if self._total_checks > 0:
                stats.uptime_percentage = (self._successful_checks / self._total_checks) * 100
            
            # Contar desconexões
            stats.total_disconnections = len(self._disconnection_times)
            
            # Tempo médio de reconexão
            if self._reconnection_times:
                stats.avg_reconnect_time = statistics.mean(self._reconnection_times)
            
            # Última verificação
            stats.last_check = self._metrics_history[-1].timestamp
            
            return stats
    
    def add_status_callback(self, callback: Callable):
        """
        Adiciona callback para mudanças de status
        
        Args:
            callback: Função a ser chamada (old_status, new_status, metrics)
        """
        with self._lock:
            if callback not in self._status_callbacks:
                self._status_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """
        Adiciona callback para alertas
        
        Args:
            callback: Função a ser chamada (alert)
        """
        with self._lock:
            if callback not in self._alert_callbacks:
                self._alert_callbacks.append(callback)
    
    def get_recent_alerts(self, hours: int = 24) -> List[HealthAlert]:
        """
        Retorna alertas recentes
        
        Args:
            hours: Número de horas para buscar
        
        Returns:
            List[HealthAlert]: Lista de alertas recentes
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [alert for alert in self._alerts if alert.timestamp >= cutoff]
    
    def is_running(self) -> bool:
        """
        Verifica se o monitor está em execução
        
        Returns:
            bool: True se em execução
        """
        with self._lock:
            return self._running