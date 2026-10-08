#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Reconnect Manager - Gerenciamento de reconexão automática MT5
Autor: Sistema MT5 Trading
Data: 2024
"""

import time
import threading
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class ConnectionState(Enum):
    """Estados de conexão"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class ReconnectConfig:
    """Configuração de reconexão"""
    max_retry_attempts: int = 10
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    backoff_multiplier: float = 2.0
    connection_timeout: float = 30.0
    health_check_interval: float = 10.0
    session_timeout_minutes: int = 60
    enable_notifications: bool = True


@dataclass
class ConnectionStats:
    """Estatísticas de conexão"""
    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    reconnection_attempts: int = 0
    last_connection_time: Optional[datetime] = None
    last_disconnection_time: Optional[datetime] = None
    average_connection_time: float = 0.0
    uptime_percentage: float = 0.0


class AutoReconnectManager:
    """Gerenciador de reconexão automática para MT5"""
    
    def __init__(self, 
                 mt5_integration,
                 config: Optional[ReconnectConfig] = None,
                 notification_callback: Optional[Callable] = None):
        
        self.mt5_integration = mt5_integration
        self.config = config or ReconnectConfig()
        self.notification_callback = notification_callback
        self.logger = logging.getLogger(__name__)
        
        # Estado da conexão
        self.connection_state = ConnectionState.DISCONNECTED
        self.stats = ConnectionStats()
        
        # Controle de threads
        self._reconnect_thread: Optional[threading.Thread] = None
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._connection_lock = threading.Lock()
        
        # Controle de retry
        self._current_retry_delay = self.config.initial_retry_delay
        self._retry_count = 0
        
        # Sessão
        self._session_start_time: Optional[datetime] = None
        self._last_successful_ping: Optional[datetime] = None
        
        # Callbacks
        self._on_connected_callbacks = []
        self._on_disconnected_callbacks = []
        self._on_reconnecting_callbacks = []
    
    def start_monitoring(self):
        """Inicia monitoramento de conexão"""
        try:
            if self._health_check_thread and self._health_check_thread.is_alive():
                return
            
            self._stop_event.clear()
            
            # Iniciar thread de monitoramento
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True,
                name="MT5HealthCheck"
            )
            self._health_check_thread.start()
            
            self.logger.info("Monitoramento de conexão iniciado")
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar monitoramento: {e}")
    
    def stop_monitoring(self):
        """Para monitoramento de conexão"""
        try:
            self._stop_event.set()
            
            # Aguardar threads terminarem
            if self._health_check_thread and self._health_check_thread.is_alive():
                self._health_check_thread.join(timeout=5.0)
            
            if self._reconnect_thread and self._reconnect_thread.is_alive():
                self._reconnect_thread.join(timeout=5.0)
            
            self.logger.info("Monitoramento de conexão parado")
            
        except Exception as e:
            self.logger.error(f"Erro ao parar monitoramento: {e}")
    
    def connect_with_retry(self, credentials: Dict[str, Any]) -> bool:
        """Conecta com retry automático"""
        with self._connection_lock:
            try:
                self.connection_state = ConnectionState.CONNECTING
                self._notify_state_change("Conectando ao MT5...")
                
                start_time = time.time()
                
                # Tentar conexão
                success = self._attempt_connection(credentials)
                
                connection_time = time.time() - start_time
                self._update_connection_stats(success, connection_time)
                
                if success:
                    self.connection_state = ConnectionState.CONNECTED
                    self._session_start_time = datetime.now()
                    self._last_successful_ping = datetime.now()
                    self._reset_retry_state()
                    
                    self._notify_state_change("Conectado ao MT5 com sucesso")
                    self._trigger_callbacks(self._on_connected_callbacks)
                    
                    return True
                else:
                    self.connection_state = ConnectionState.FAILED
                    self._notify_state_change("Falha na conexão MT5")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Erro na conexão: {e}")
                self.connection_state = ConnectionState.FAILED
                return False
    
    def _attempt_connection(self, credentials: Dict[str, Any]) -> bool:
        """Tenta estabelecer conexão"""
        try:
            # Configurar MT5
            from mt5_integration import MT5Config
            
            mt5_config = MT5Config(
                login=credentials.get('login'),
                password=credentials.get('password'),
                server=credentials.get('server'),
                path=credentials.get('path', '')
            )
            
            # Tentar conectar
            success = self.mt5_integration.connect(mt5_config)
            
            if success:
                # Verificar se conexão está realmente ativa
                account_info = self.mt5_integration.get_account_info()
                return account_info is not None
            
            return False
            
        except Exception as e:
            self.logger.error(f"Erro na tentativa de conexão: {e}")
            return False
    
    def _health_check_loop(self):
        """Loop de verificação de saúde da conexão"""
        while not self._stop_event.is_set():
            try:
                if self.connection_state == ConnectionState.CONNECTED:
                    if not self._check_connection_health():
                        self._handle_connection_lost()
                
                # Aguardar próxima verificação
                self._stop_event.wait(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Erro no health check: {e}")
                time.sleep(self.config.health_check_interval)
    
    def _check_connection_health(self) -> bool:
        """Verifica saúde da conexão"""
        try:
            # Verificar se MT5 está conectado
            if not self.mt5_integration.is_connected():
                return False
            
            # Verificar timeout de sessão
            if self._session_start_time:
                session_duration = datetime.now() - self._session_start_time
                if session_duration > timedelta(minutes=self.config.session_timeout_minutes):
                    self.logger.warning("Timeout de sessão atingido")
                    return False
            
            # Tentar obter informações da conta (ping)
            account_info = self.mt5_integration.get_account_info()
            if account_info:
                self._last_successful_ping = datetime.now()
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Erro na verificação de saúde: {e}")
            return False
    
    def _handle_connection_lost(self):
        """Trata perda de conexão"""
        try:
            self.logger.warning("Conexão MT5 perdida - iniciando reconexão")
            
            self.connection_state = ConnectionState.DISCONNECTED
            self.stats.last_disconnection_time = datetime.now()
            
            self._notify_state_change("Conexão perdida - tentando reconectar...")
            self._trigger_callbacks(self._on_disconnected_callbacks)
            
            # Iniciar processo de reconexão
            self._start_reconnection()
            
        except Exception as e:
            self.logger.error(f"Erro ao tratar perda de conexão: {e}")
    
    def _start_reconnection(self):
        """Inicia processo de reconexão"""
        try:
            if self._reconnect_thread and self._reconnect_thread.is_alive():
                return
            
            self._reconnect_thread = threading.Thread(
                target=self._reconnection_loop,
                daemon=True,
                name="MT5Reconnect"
            )
            self._reconnect_thread.start()
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar reconexão: {e}")
    
    def _reconnection_loop(self):
        """Loop de reconexão com backoff exponencial"""
        self.connection_state = ConnectionState.RECONNECTING
        self._trigger_callbacks(self._on_reconnecting_callbacks)
        
        while (not self._stop_event.is_set() and 
               self._retry_count < self.config.max_retry_attempts and
               self.connection_state != ConnectionState.CONNECTED):
            
            try:
                self._retry_count += 1
                self.stats.reconnection_attempts += 1
                
                self.logger.info(f"Tentativa de reconexão {self._retry_count}/{self.config.max_retry_attempts}")
                self._notify_state_change(f"Reconectando... (tentativa {self._retry_count})")
                
                # Aguardar delay antes da tentativa
                if not self._stop_event.wait(self._current_retry_delay):
                    break
                
                # Tentar reconectar (usar últimas credenciais conhecidas)
                if hasattr(self.mt5_integration, '_last_config') and self.mt5_integration._last_config:
                    credentials = {
                        'login': self.mt5_integration._last_config.login,
                        'password': self.mt5_integration._last_config.password,
                        'server': self.mt5_integration._last_config.server,
                        'path': self.mt5_integration._last_config.path
                    }
                    
                    if self._attempt_connection(credentials):
                        self.connection_state = ConnectionState.CONNECTED
                        self._session_start_time = datetime.now()
                        self._last_successful_ping = datetime.now()
                        self._reset_retry_state()
                        
                        self.logger.info("Reconexão bem-sucedida")
                        self._notify_state_change("Reconectado com sucesso")
                        self._trigger_callbacks(self._on_connected_callbacks)
                        return
                
                # Aumentar delay para próxima tentativa (backoff exponencial)
                self._current_retry_delay = min(
                    self._current_retry_delay * self.config.backoff_multiplier,
                    self.config.max_retry_delay
                )
                
            except Exception as e:
                self.logger.error(f"Erro na tentativa de reconexão: {e}")
        
        # Se chegou aqui, esgotou tentativas
        if self._retry_count >= self.config.max_retry_attempts:
            self.connection_state = ConnectionState.FAILED
            self.logger.error("Máximo de tentativas de reconexão atingido")
            self._notify_state_change("Falha na reconexão - máximo de tentativas atingido")
    
    def _reset_retry_state(self):
        """Reseta estado de retry"""
        self._retry_count = 0
        self._current_retry_delay = self.config.initial_retry_delay
    
    def _update_connection_stats(self, success: bool, connection_time: float):
        """Atualiza estatísticas de conexão"""
        self.stats.total_connections += 1
        
        if success:
            self.stats.successful_connections += 1
            self.stats.last_connection_time = datetime.now()
            
            # Atualizar tempo médio de conexão
            if self.stats.average_connection_time == 0:
                self.stats.average_connection_time = connection_time
            else:
                self.stats.average_connection_time = (
                    (self.stats.average_connection_time + connection_time) / 2
                )
        else:
            self.stats.failed_connections += 1
        
        # Calcular uptime
        if self.stats.total_connections > 0:
            self.stats.uptime_percentage = (
                self.stats.successful_connections / self.stats.total_connections * 100
            )
    
    def _notify_state_change(self, message: str):
        """Notifica mudança de estado"""
        if self.config.enable_notifications and self.notification_callback:
            try:
                self.notification_callback({
                    'type': 'connection_state',
                    'state': self.connection_state.value,
                    'message': message,
                    'timestamp': datetime.now()
                })
            except Exception as e:
                self.logger.error(f"Erro na notificação: {e}")
    
    def _trigger_callbacks(self, callbacks: list):
        """Dispara callbacks"""
        for callback in callbacks:
            try:
                callback(self.connection_state, self.stats)
            except Exception as e:
                self.logger.error(f"Erro no callback: {e}")
    
    def add_connection_callback(self, callback: Callable):
        """Adiciona callback para conexão estabelecida"""
        self._on_connected_callbacks.append(callback)
    
    def add_disconnection_callback(self, callback: Callable):
        """Adiciona callback para desconexão"""
        self._on_disconnected_callbacks.append(callback)
    
    def add_reconnecting_callback(self, callback: Callable):
        """Adiciona callback para reconexão em andamento"""
        self._on_reconnecting_callbacks.append(callback)
    
    def force_reconnect(self) -> bool:
        """Força reconexão imediata"""
        try:
            self.logger.info("Reconexão forçada solicitada")
            
            # Parar monitoramento atual
            self._stop_event.set()
            
            # Desconectar
            if self.mt5_integration.is_connected():
                self.mt5_integration.disconnect()
            
            # Resetar estado
            self._reset_retry_state()
            self.connection_state = ConnectionState.DISCONNECTED
            
            # Reiniciar monitoramento
            self._stop_event.clear()
            self.start_monitoring()
            
            # Iniciar reconexão
            self._start_reconnection()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na reconexão forçada: {e}")
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Retorna status detalhado da conexão"""
        return {
            'state': self.connection_state.value,
            'is_connected': self.connection_state == ConnectionState.CONNECTED,
            'retry_count': self._retry_count,
            'max_retries': self.config.max_retry_attempts,
            'current_delay': self._current_retry_delay,
            'session_start': self._session_start_time.isoformat() if self._session_start_time else None,
            'last_ping': self._last_successful_ping.isoformat() if self._last_successful_ping else None,
            'stats': {
                'total_connections': self.stats.total_connections,
                'successful_connections': self.stats.successful_connections,
                'failed_connections': self.stats.failed_connections,
                'reconnection_attempts': self.stats.reconnection_attempts,
                'uptime_percentage': round(self.stats.uptime_percentage, 2),
                'average_connection_time': round(self.stats.average_connection_time, 2)
            }
        }
    
    def update_config(self, new_config: ReconnectConfig):
        """Atualiza configuração de reconexão"""
        self.config = new_config
        self.logger.info("Configuração de reconexão atualizada")