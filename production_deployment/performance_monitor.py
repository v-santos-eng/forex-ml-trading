import logging
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from pathlib import Path
import warnings
from collections import deque, defaultdict
import statistics

class AlertLevel(Enum):
    """Níveis de alerta"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(Enum):
    """Tipos de métricas monitoradas"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    PROFIT_FACTOR = "profit_factor"
    SHARPE_RATIO = "sharpe_ratio"
    DRAWDOWN = "drawdown"
    WIN_RATE = "win_rate"
    LOSS_STREAK = "loss_streak"
    VOLATILITY = "volatility"
    PREDICTION_CONFIDENCE = "prediction_confidence"
    EXECUTION_TIME = "execution_time"

@dataclass
class PerformanceAlert:
    """Alerta de performance"""
    alert_id: str
    pair: str
    metric_type: MetricType
    alert_level: AlertLevel
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class MetricThreshold:
    """Configuração de threshold para uma métrica"""
    metric_type: MetricType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    emergency_threshold: Optional[float] = None
    lookback_periods: int = 10  # Períodos para análise de tendência
    trend_threshold: float = 0.1  # Threshold para detecção de tendência

class PerformanceDegradationMonitor:
    """Monitor de degradação de performance de modelos ML"""
    
    def __init__(self, db_path: str = "performance_monitor.db"):
        self.logger = self._setup_logger()
        self.db_path = db_path
        self.alerts: List[PerformanceAlert] = []
        self.metric_history: Dict[str, Dict[MetricType, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
        
        # Configurações padrão de thresholds
        self.thresholds = self._initialize_default_thresholds()
        
        # Estado do monitor
        self.is_running = False
        self.monitor_thread = None
        self._stop_event = threading.Event()
        
        # Configurações
        self.config = {
            "monitoring_interval_seconds": 300,  # 5 minutos
            "alert_cooldown_minutes": 30,
            "max_alerts_per_pair": 10,
            "enable_trend_analysis": True,
            "enable_anomaly_detection": True,
            "notification_webhook": None,
            "email_notifications": False
        }
        
        self._initialize_database()
        self._load_historical_data()
    
    def _setup_logger(self) -> logging.Logger:
        """Configura o sistema de logging"""
        logger = logging.getLogger("PerformanceMonitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            handler = logging.FileHandler("logs/performance_monitor.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger
    
    def _initialize_default_thresholds(self) -> Dict[MetricType, MetricThreshold]:
        """Inicializa thresholds padrão para todas as métricas"""
        return {
            MetricType.ACCURACY: MetricThreshold(
                metric_type=MetricType.ACCURACY,
                min_value=0.52,
                warning_threshold=0.55,
                critical_threshold=0.50,
                emergency_threshold=0.45
            ),
            MetricType.PRECISION: MetricThreshold(
                metric_type=MetricType.PRECISION,
                min_value=0.50,
                warning_threshold=0.53,
                critical_threshold=0.48,
                emergency_threshold=0.43
            ),
            MetricType.RECALL: MetricThreshold(
                metric_type=MetricType.RECALL,
                min_value=0.50,
                warning_threshold=0.53,
                critical_threshold=0.48,
                emergency_threshold=0.43
            ),
            MetricType.F1_SCORE: MetricThreshold(
                metric_type=MetricType.F1_SCORE,
                min_value=0.50,
                warning_threshold=0.53,
                critical_threshold=0.48,
                emergency_threshold=0.43
            ),
            MetricType.PROFIT_FACTOR: MetricThreshold(
                metric_type=MetricType.PROFIT_FACTOR,
                min_value=1.0,
                warning_threshold=1.2,
                critical_threshold=0.9,
                emergency_threshold=0.7
            ),
            MetricType.SHARPE_RATIO: MetricThreshold(
                metric_type=MetricType.SHARPE_RATIO,
                min_value=0.3,
                warning_threshold=0.5,
                critical_threshold=0.2,
                emergency_threshold=0.0
            ),
            MetricType.DRAWDOWN: MetricThreshold(
                metric_type=MetricType.DRAWDOWN,
                max_value=0.20,
                warning_threshold=0.15,
                critical_threshold=0.25,
                emergency_threshold=0.35
            ),
            MetricType.WIN_RATE: MetricThreshold(
                metric_type=MetricType.WIN_RATE,
                min_value=0.45,
                warning_threshold=0.50,
                critical_threshold=0.40,
                emergency_threshold=0.35
            ),
            MetricType.LOSS_STREAK: MetricThreshold(
                metric_type=MetricType.LOSS_STREAK,
                max_value=8,
                warning_threshold=5,
                critical_threshold=10,
                emergency_threshold=15
            ),
            MetricType.VOLATILITY: MetricThreshold(
                metric_type=MetricType.VOLATILITY,
                max_value=0.05,
                warning_threshold=0.03,
                critical_threshold=0.07,
                emergency_threshold=0.10
            ),
            MetricType.PREDICTION_CONFIDENCE: MetricThreshold(
                metric_type=MetricType.PREDICTION_CONFIDENCE,
                min_value=0.60,
                warning_threshold=0.65,
                critical_threshold=0.55,
                emergency_threshold=0.50
            ),
            MetricType.EXECUTION_TIME: MetricThreshold(
                metric_type=MetricType.EXECUTION_TIME,
                max_value=10.0,  # segundos
                warning_threshold=5.0,
                critical_threshold=15.0,
                emergency_threshold=30.0
            )
        }
    
    def _initialize_database(self) -> None:
        """Inicializa o banco de dados SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Tabela de métricas históricas
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pair TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        value REAL NOT NULL,
                        timestamp DATETIME NOT NULL,
                        model_version TEXT,
                        additional_data TEXT
                    )
                """)
                
                # Tabela de alertas
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT UNIQUE NOT NULL,
                        pair TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        alert_level TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        threshold_value REAL NOT NULL,
                        message TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        acknowledged BOOLEAN DEFAULT FALSE,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolution_time DATETIME
                    )
                """)
                
                # Índices para performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_pair_type_time ON performance_metrics(pair, metric_type, timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_pair_level ON performance_alerts(pair, alert_level)")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar banco de dados: {e}")
    
    def _load_historical_data(self) -> None:
        """Carrega dados históricos do banco para memória"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Carrega métricas dos últimos 30 dias
                cutoff_date = datetime.now() - timedelta(days=30)
                
                query = """
                    SELECT pair, metric_type, value, timestamp 
                    FROM performance_metrics 
                    WHERE timestamp > ? 
                    ORDER BY timestamp ASC
                """
                
                cursor = conn.cursor()
                cursor.execute(query, (cutoff_date,))
                
                for row in cursor.fetchall():
                    pair, metric_type_str, value, timestamp_str = row
                    metric_type = MetricType(metric_type_str)
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
                    self.metric_history[pair][metric_type].append({
                        'value': value,
                        'timestamp': timestamp
                    })
                
                self.logger.info(f"Dados históricos carregados para {len(self.metric_history)} pares")
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados históricos: {e}")
    
    def start_monitoring(self) -> None:
        """Inicia o monitoramento contínuo"""
        if self.is_running:
            self.logger.warning("Monitor já está em execução")
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Monitor de performance iniciado")
    
    def stop_monitoring(self) -> None:
        """Para o monitoramento"""
        self.is_running = False
        self._stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        
        self.logger.info("Monitor de performance parado")
    
    def _monitoring_loop(self) -> None:
        """Loop principal de monitoramento"""
        interval = self.config.get("monitoring_interval_seconds", 300)
        
        while not self._stop_event.is_set():
            try:
                self._collect_and_analyze_metrics()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Erro no loop de monitoramento: {e}")
                time.sleep(interval)
    
    def _collect_and_analyze_metrics(self) -> None:
        """Coleta e analisa métricas atuais"""
        # Lista de pares para monitorar (seria obtida da configuração)
        pairs_to_monitor = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        
        for pair in pairs_to_monitor:
            try:
                current_metrics = self._get_current_metrics(pair)
                
                for metric_type, value in current_metrics.items():
                    # Armazena métrica
                    self.record_metric(pair, metric_type, value)
                    
                    # Analisa para alertas
                    self._analyze_metric_for_alerts(pair, metric_type, value)
                    
            except Exception as e:
                self.logger.error(f"Erro ao coletar métricas para {pair}: {e}")
    
    def _get_current_metrics(self, pair: str) -> Dict[MetricType, float]:
        """Obtém métricas atuais do modelo (placeholder)"""
        # Esta função seria integrada com o sistema real de métricas
        # Por enquanto, simula métricas realistas
        
        import random
        
        # Simula degradação gradual ao longo do tempo
        base_accuracy = 0.58
        time_factor = (datetime.now().hour % 24) / 24.0
        noise = random.uniform(-0.05, 0.05)
        
        return {
            MetricType.ACCURACY: max(0.4, min(0.8, base_accuracy - time_factor * 0.1 + noise)),
            MetricType.PRECISION: random.uniform(0.45, 0.65),
            MetricType.RECALL: random.uniform(0.45, 0.65),
            MetricType.F1_SCORE: random.uniform(0.45, 0.65),
            MetricType.PROFIT_FACTOR: random.uniform(0.8, 1.8),
            MetricType.SHARPE_RATIO: random.uniform(0.1, 1.2),
            MetricType.DRAWDOWN: random.uniform(0.05, 0.30),
            MetricType.WIN_RATE: random.uniform(0.40, 0.65),
            MetricType.LOSS_STREAK: random.randint(0, 12),
            MetricType.VOLATILITY: random.uniform(0.01, 0.08),
            MetricType.PREDICTION_CONFIDENCE: random.uniform(0.50, 0.85),
            MetricType.EXECUTION_TIME: random.uniform(1.0, 20.0)
        }
    
    def record_metric(self, pair: str, metric_type: MetricType, value: float, 
                     model_version: str = None, additional_data: Dict = None) -> None:
        """Registra uma métrica no sistema"""
        timestamp = datetime.now()
        
        # Adiciona à memória
        self.metric_history[pair][metric_type].append({
            'value': value,
            'timestamp': timestamp
        })
        
        # Salva no banco de dados
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO performance_metrics 
                    (pair, metric_type, value, timestamp, model_version, additional_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    pair, 
                    metric_type.value, 
                    value, 
                    timestamp.isoformat(),
                    model_version,
                    json.dumps(additional_data) if additional_data else None
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Erro ao salvar métrica no banco: {e}")
    
    def _analyze_metric_for_alerts(self, pair: str, metric_type: MetricType, current_value: float) -> None:
        """Analisa uma métrica para gerar alertas"""
        if metric_type not in self.thresholds:
            return
        
        threshold = self.thresholds[metric_type]
        alert_level = self._determine_alert_level(current_value, threshold)
        
        if alert_level:
            # Verifica cooldown de alertas
            if self._is_alert_in_cooldown(pair, metric_type, alert_level):
                return
            
            # Cria alerta
            alert = self._create_alert(pair, metric_type, alert_level, current_value, threshold)
            self._process_alert(alert)
        
        # Análise de tendência se habilitada
        if self.config.get("enable_trend_analysis", True):
            self._analyze_trend(pair, metric_type)
        
        # Detecção de anomalias se habilitada
        if self.config.get("enable_anomaly_detection", True):
            self._detect_anomaly(pair, metric_type, current_value)
    
    def _determine_alert_level(self, value: float, threshold: MetricThreshold) -> Optional[AlertLevel]:
        """Determina o nível de alerta baseado no valor e threshold"""
        # Para métricas que devem ser minimizadas (drawdown, loss_streak, etc.)
        if threshold.max_value is not None:
            if threshold.emergency_threshold and value >= threshold.emergency_threshold:
                return AlertLevel.EMERGENCY
            elif threshold.critical_threshold and value >= threshold.critical_threshold:
                return AlertLevel.CRITICAL
            elif threshold.warning_threshold and value >= threshold.warning_threshold:
                return AlertLevel.WARNING
        
        # Para métricas que devem ser maximizadas (accuracy, precision, etc.)
        if threshold.min_value is not None:
            if threshold.emergency_threshold and value <= threshold.emergency_threshold:
                return AlertLevel.EMERGENCY
            elif threshold.critical_threshold and value <= threshold.critical_threshold:
                return AlertLevel.CRITICAL
            elif threshold.warning_threshold and value <= threshold.warning_threshold:
                return AlertLevel.WARNING
        
        return None
    
    def _is_alert_in_cooldown(self, pair: str, metric_type: MetricType, alert_level: AlertLevel) -> bool:
        """Verifica se um alerta está em período de cooldown"""
        cooldown_minutes = self.config.get("alert_cooldown_minutes", 30)
        cutoff_time = datetime.now() - timedelta(minutes=cooldown_minutes)
        
        recent_alerts = [
            alert for alert in self.alerts
            if (alert.pair == pair and 
                alert.metric_type == metric_type and 
                alert.alert_level == alert_level and 
                alert.timestamp > cutoff_time)
        ]
        
        return len(recent_alerts) > 0
    
    def _create_alert(self, pair: str, metric_type: MetricType, alert_level: AlertLevel, 
                     current_value: float, threshold: MetricThreshold) -> PerformanceAlert:
        """Cria um novo alerta"""
        alert_id = f"{pair}_{metric_type.value}_{alert_level.value}_{int(datetime.now().timestamp())}"
        
        # Determina valor do threshold violado
        if alert_level == AlertLevel.EMERGENCY:
            threshold_value = threshold.emergency_threshold
        elif alert_level == AlertLevel.CRITICAL:
            threshold_value = threshold.critical_threshold
        elif alert_level == AlertLevel.WARNING:
            threshold_value = threshold.warning_threshold
        else:
            threshold_value = threshold.min_value or threshold.max_value
        
        message = self._generate_alert_message(pair, metric_type, alert_level, current_value, threshold_value)
        
        return PerformanceAlert(
            alert_id=alert_id,
            pair=pair,
            metric_type=metric_type,
            alert_level=alert_level,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            timestamp=datetime.now()
        )
    
    def _generate_alert_message(self, pair: str, metric_type: MetricType, alert_level: AlertLevel, 
                               current_value: float, threshold_value: float) -> str:
        """Gera mensagem descritiva para o alerta"""
        metric_name = metric_type.value.replace('_', ' ').title()
        
        if alert_level == AlertLevel.EMERGENCY:
            severity = "EMERGÊNCIA"
        elif alert_level == AlertLevel.CRITICAL:
            severity = "CRÍTICO"
        elif alert_level == AlertLevel.WARNING:
            severity = "AVISO"
        else:
            severity = "INFO"
        
        return f"[{severity}] {pair}: {metric_name} = {current_value:.4f} (threshold: {threshold_value:.4f})"
    
    def _process_alert(self, alert: PerformanceAlert) -> None:
        """Processa um novo alerta"""
        # Adiciona à lista de alertas
        self.alerts.append(alert)
        
        # Salva no banco de dados
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO performance_alerts 
                    (alert_id, pair, metric_type, alert_level, current_value, 
                     threshold_value, message, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    alert.pair,
                    alert.metric_type.value,
                    alert.alert_level.value,
                    alert.current_value,
                    alert.threshold_value,
                    alert.message,
                    alert.timestamp.isoformat()
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Erro ao salvar alerta no banco: {e}")
        
        # Log do alerta
        if alert.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            self.logger.error(alert.message)
        elif alert.alert_level == AlertLevel.WARNING:
            self.logger.warning(alert.message)
        else:
            self.logger.info(alert.message)
        
        # Envia notificações se configuradas
        self._send_notifications(alert)
        
        # Limita número de alertas por par
        self._cleanup_old_alerts()
    
    def _analyze_trend(self, pair: str, metric_type: MetricType) -> None:
        """Analisa tendência de uma métrica"""
        history = self.metric_history[pair][metric_type]
        
        if len(history) < 5:  # Precisa de pelo menos 5 pontos
            return
        
        # Pega os últimos N valores
        lookback = self.thresholds[metric_type].lookback_periods
        recent_values = [item['value'] for item in list(history)[-lookback:]]
        
        if len(recent_values) < 3:
            return
        
        # Calcula tendência usando regressão linear simples
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            trend_threshold = self.thresholds[metric_type].trend_threshold
            
            # Para métricas que devem aumentar (accuracy, precision, etc.)
            if metric_type in [MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, 
                              MetricType.F1_SCORE, MetricType.PROFIT_FACTOR, MetricType.SHARPE_RATIO]:
                if slope < -trend_threshold:
                    self._create_trend_alert(pair, metric_type, "declining", slope)
            
            # Para métricas que devem diminuir (drawdown, loss_streak, etc.)
            elif metric_type in [MetricType.DRAWDOWN, MetricType.LOSS_STREAK, MetricType.VOLATILITY]:
                if slope > trend_threshold:
                    self._create_trend_alert(pair, metric_type, "increasing", slope)
                    
        except Exception as e:
            self.logger.debug(f"Erro na análise de tendência para {pair}/{metric_type}: {e}")
    
    def _create_trend_alert(self, pair: str, metric_type: MetricType, trend_direction: str, slope: float) -> None:
        """Cria alerta de tendência"""
        alert_id = f"{pair}_{metric_type.value}_trend_{int(datetime.now().timestamp())}"
        message = f"[TENDÊNCIA] {pair}: {metric_type.value} está {trend_direction} (slope: {slope:.6f})"
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            pair=pair,
            metric_type=metric_type,
            alert_level=AlertLevel.INFO,
            current_value=slope,
            threshold_value=self.thresholds[metric_type].trend_threshold,
            message=message,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        self.logger.info(message)
    
    def _detect_anomaly(self, pair: str, metric_type: MetricType, current_value: float) -> None:
        """Detecta anomalias usando análise estatística"""
        history = self.metric_history[pair][metric_type]
        
        if len(history) < 20:  # Precisa de histórico suficiente
            return
        
        # Pega os últimos 50 valores (excluindo o atual)
        recent_values = [item['value'] for item in list(history)[-50:-1]]
        
        if len(recent_values) < 10:
            return
        
        try:
            mean = statistics.mean(recent_values)
            stdev = statistics.stdev(recent_values)
            
            # Calcula z-score
            z_score = abs(current_value - mean) / stdev if stdev > 0 else 0
            
            # Considera anomalia se z-score > 2.5 (aproximadamente 1.2% de probabilidade)
            if z_score > 2.5:
                self._create_anomaly_alert(pair, metric_type, current_value, z_score, mean, stdev)
                
        except Exception as e:
            self.logger.debug(f"Erro na detecção de anomalia para {pair}/{metric_type}: {e}")
    
    def _create_anomaly_alert(self, pair: str, metric_type: MetricType, current_value: float, 
                             z_score: float, mean: float, stdev: float) -> None:
        """Cria alerta de anomalia"""
        alert_id = f"{pair}_{metric_type.value}_anomaly_{int(datetime.now().timestamp())}"
        message = f"[ANOMALIA] {pair}: {metric_type.value} = {current_value:.4f} (z-score: {z_score:.2f}, média: {mean:.4f})"
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            pair=pair,
            metric_type=metric_type,
            alert_level=AlertLevel.WARNING,
            current_value=current_value,
            threshold_value=z_score,
            message=message,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        self.logger.warning(message)
    
    def _send_notifications(self, alert: PerformanceAlert) -> None:
        """Envia notificações do alerta"""
        # Implementação de notificações (webhook, email, etc.)
        # Por enquanto, apenas log
        if alert.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            self.logger.critical(f"NOTIFICAÇÃO CRÍTICA: {alert.message}")
    
    def _cleanup_old_alerts(self) -> None:
        """Remove alertas antigos para evitar acúmulo excessivo"""
        max_alerts_per_pair = self.config.get("max_alerts_per_pair", 10)
        cutoff_date = datetime.now() - timedelta(days=7)
        
        # Remove alertas muito antigos
        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_date]
        
        # Limita alertas por par
        pair_alert_counts = defaultdict(int)
        filtered_alerts = []
        
        # Ordena por timestamp (mais recentes primeiro)
        sorted_alerts = sorted(self.alerts, key=lambda x: x.timestamp, reverse=True)
        
        for alert in sorted_alerts:
            if pair_alert_counts[alert.pair] < max_alerts_per_pair:
                filtered_alerts.append(alert)
                pair_alert_counts[alert.pair] += 1
        
        self.alerts = filtered_alerts
    
    def get_alerts(self, pair: str = None, alert_level: AlertLevel = None, 
                   unacknowledged_only: bool = False) -> List[Dict]:
        """Retorna alertas filtrados"""
        filtered_alerts = self.alerts
        
        if pair:
            filtered_alerts = [a for a in filtered_alerts if a.pair == pair]
        
        if alert_level:
            filtered_alerts = [a for a in filtered_alerts if a.alert_level == alert_level]
        
        if unacknowledged_only:
            filtered_alerts = [a for a in filtered_alerts if not a.acknowledged]
        
        return [asdict(alert) for alert in filtered_alerts]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Marca um alerta como reconhecido"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self.logger.info(f"Alerta reconhecido: {alert_id}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Marca um alerta como resolvido"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                self.logger.info(f"Alerta resolvido: {alert_id}")
                return True
        return False
    
    def get_performance_summary(self, pair: str = None, hours: int = 24) -> Dict:
        """Retorna resumo de performance"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        summary = {
            "period_hours": hours,
            "pairs": {},
            "total_alerts": 0,
            "critical_alerts": 0,
            "warning_alerts": 0
        }
        
        pairs_to_analyze = [pair] if pair else list(self.metric_history.keys())
        
        for p in pairs_to_analyze:
            pair_summary = {
                "metrics": {},
                "alerts": 0,
                "last_update": None
            }
            
            for metric_type, history in self.metric_history[p].items():
                recent_values = [
                    item for item in history 
                    if item['timestamp'] > cutoff_time
                ]
                
                if recent_values:
                    values = [item['value'] for item in recent_values]
                    pair_summary["metrics"][metric_type.value] = {
                        "current": values[-1],
                        "average": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }
                    
                    if pair_summary["last_update"] is None:
                        pair_summary["last_update"] = recent_values[-1]['timestamp'].isoformat()
                    else:
                        last_timestamp = datetime.fromisoformat(pair_summary["last_update"]) if isinstance(pair_summary["last_update"], str) else pair_summary["last_update"]
                        if recent_values[-1]['timestamp'] > last_timestamp:
                            pair_summary["last_update"] = recent_values[-1]['timestamp'].isoformat()
            
            # Conta alertas do período
            pair_alerts = [
                alert for alert in self.alerts
                if alert.pair == p and alert.timestamp > cutoff_time
            ]
            
            pair_summary["alerts"] = len(pair_alerts)
            summary["pairs"][p] = pair_summary
        
        # Estatísticas gerais de alertas
        recent_alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
        summary["total_alerts"] = len(recent_alerts)
        summary["critical_alerts"] = len([a for a in recent_alerts if a.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]])
        summary["warning_alerts"] = len([a for a in recent_alerts if a.alert_level == AlertLevel.WARNING])
        
        return summary
    
    def update_thresholds(self, metric_type: MetricType, **kwargs) -> None:
        """Atualiza thresholds de uma métrica"""
        if metric_type in self.thresholds:
            threshold = self.thresholds[metric_type]
            for key, value in kwargs.items():
                if hasattr(threshold, key):
                    setattr(threshold, key, value)
                    self.logger.info(f"Threshold atualizado: {metric_type.value}.{key} = {value}")
    
    def export_metrics(self, pair: str = None, start_date: datetime = None, 
                      end_date: datetime = None) -> pd.DataFrame:
        """Exporta métricas para DataFrame"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM performance_metrics WHERE 1=1"
                params = []
                
                if pair:
                    query += " AND pair = ?"
                    params.append(pair)
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date.isoformat())
                
                query += " ORDER BY timestamp ASC"
                
                return pd.read_sql_query(query, conn, params=params)
                
        except Exception as e:
            self.logger.error(f"Erro ao exportar métricas: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Exemplo de uso
    monitor = PerformanceDegradationMonitor()
    
    try:
        monitor.start_monitoring()
        
        # Simula algumas métricas
        for i in range(10):
            monitor.record_metric("EURUSD", MetricType.ACCURACY, 0.55 - i * 0.01)
            monitor.record_metric("EURUSD", MetricType.DRAWDOWN, 0.10 + i * 0.02)
            time.sleep(2)
        
        # Mostra alertas
        alerts = monitor.get_alerts()
        print(f"Total de alertas: {len(alerts)}")
        
        for alert in alerts[-5:]:  # Últimos 5 alertas
            print(f"- {alert['message']}")
        
        # Mostra resumo
        summary = monitor.get_performance_summary("EURUSD", hours=1)
        print(f"\nResumo de performance: {summary}")
        
    except KeyboardInterrupt:
        print("Parando monitor...")
    finally:
        monitor.stop_monitoring()