import logging
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from pathlib import Path
import warnings
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Bibliotecas para visualização e dashboard
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly não disponível. Dashboards limitados.")

try:
    import dash
    from dash import dcc, html, Input, Output, State
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    # Mock objects para quando Dash não está disponível
    class MockHTML:
        class Div:
            def __init__(self, *args, **kwargs):
                pass
        class H1:
            def __init__(self, *args, **kwargs):
                pass
        class H2:
            def __init__(self, *args, **kwargs):
                pass
        class P:
            def __init__(self, *args, **kwargs):
                pass
        class Table:
            def __init__(self, *args, **kwargs):
                pass
        class Thead:
            def __init__(self, *args, **kwargs):
                pass
        class Tbody:
            def __init__(self, *args, **kwargs):
                pass
        class Tr:
            def __init__(self, *args, **kwargs):
                pass
        class Th:
            def __init__(self, *args, **kwargs):
                pass
        class Td:
            def __init__(self, *args, **kwargs):
                pass
    
    class MockDCC:
        class Graph:
            def __init__(self, *args, **kwargs):
                pass
        class Interval:
            def __init__(self, *args, **kwargs):
                pass
    
    html = MockHTML()
    dcc = MockDCC()
    print("Dash não disponível. Interface web não disponível.")

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class PredictionQuality(Enum):
    """Níveis de qualidade das predições"""
    EXCELLENT = "excellent"  # > 80%
    GOOD = "good"           # 70-80%
    AVERAGE = "average"     # 60-70%
    POOR = "poor"           # 50-60%
    CRITICAL = "critical"   # < 50%

class MetricType(Enum):
    """Tipos de métricas monitoradas"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    PREDICTION_CONFIDENCE = "prediction_confidence"
    PREDICTION_LATENCY = "prediction_latency"
    MODEL_DRIFT = "model_drift"
    DATA_DRIFT = "data_drift"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    EXECUTION_TIME = "execution_time"

class AlertSeverity(Enum):
    """Severidade dos alertas"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class PredictionRecord:
    """Registro de uma predição"""
    prediction_id: str
    pair: str
    model_name: str
    timestamp: datetime
    predicted_value: float
    predicted_class: int
    confidence: float
    actual_value: Optional[float] = None
    actual_class: Optional[int] = None
    execution_time: float = 0.0
    features_hash: Optional[str] = None
    market_conditions: Optional[Dict] = None

@dataclass
class QualityMetric:
    """Métrica de qualidade"""
    metric_id: str
    metric_type: MetricType
    pair: str
    model_name: str
    value: float
    timestamp: datetime
    window_size: int
    threshold_breached: bool = False
    severity: Optional[AlertSeverity] = None

@dataclass
class QualityAlert:
    """Alerta de qualidade"""
    alert_id: str
    metric_type: MetricType
    pair: str
    model_name: str
    severity: AlertSeverity
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False

class PredictionQualityMonitor:
    """Monitor de qualidade das predições em tempo real"""
    
    def __init__(self, db_path: str = "prediction_quality.db", 
                 dashboard_port: int = 8050, enable_dashboard: bool = True):
        self.logger = self._setup_logger()
        self.db_path = db_path
        self.dashboard_port = dashboard_port
        self.enable_dashboard = enable_dashboard and DASH_AVAILABLE
        
        # Armazenamento em memória para acesso rápido
        self.prediction_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_buffer: List[QualityAlert] = []
        
        # Configurações de monitoramento
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.dashboard_thread: Optional[threading.Thread] = None
        
        # Thresholds padrão para métricas
        self.quality_thresholds = self._initialize_thresholds()
        
        # Configurações de janela deslizante
        self.window_sizes = {
            'short': 100,    # Últimas 100 predições
            'medium': 500,   # Últimas 500 predições
            'long': 1000     # Últimas 1000 predições
        }
        
        # Cache para cálculos otimizados
        self.metrics_cache: Dict[str, Tuple[float, datetime]] = {}
        self.cache_ttl = 60  # Cache válido por 60 segundos
        
        self._initialize_database()
        self._load_historical_data()
        
        if self.enable_dashboard:
            self._setup_dashboard()
    
    def _setup_logger(self) -> logging.Logger:
        """Configura o sistema de logging"""
        logger = logging.getLogger("PredictionQualityMonitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            handler = logging.FileHandler("logs/prediction_quality_monitor.log")
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
    
    def _initialize_thresholds(self) -> Dict[MetricType, Dict[str, float]]:
        """Inicializa thresholds padrão para métricas de qualidade"""
        return {
            MetricType.ACCURACY: {
                'excellent': 0.80,
                'good': 0.70,
                'average': 0.60,
                'poor': 0.50,
                'critical': 0.40
            },
            MetricType.PRECISION: {
                'excellent': 0.80,
                'good': 0.70,
                'average': 0.60,
                'poor': 0.50,
                'critical': 0.40
            },
            MetricType.RECALL: {
                'excellent': 0.80,
                'good': 0.70,
                'average': 0.60,
                'poor': 0.50,
                'critical': 0.40
            },
            MetricType.F1_SCORE: {
                'excellent': 0.80,
                'good': 0.70,
                'average': 0.60,
                'poor': 0.50,
                'critical': 0.40
            },
            MetricType.PREDICTION_CONFIDENCE: {
                'excellent': 0.90,
                'good': 0.80,
                'average': 0.70,
                'poor': 0.60,
                'critical': 0.50
            },
            MetricType.PREDICTION_LATENCY: {
                'excellent': 0.1,   # < 100ms
                'good': 0.5,        # < 500ms
                'average': 1.0,     # < 1s
                'poor': 2.0,        # < 2s
                'critical': 5.0     # > 5s é crítico
            },
            MetricType.WIN_RATE: {
                'excellent': 0.70,
                'good': 0.60,
                'average': 0.55,
                'poor': 0.50,
                'critical': 0.45
            },
            MetricType.PROFIT_FACTOR: {
                'excellent': 2.0,
                'good': 1.5,
                'average': 1.2,
                'poor': 1.0,
                'critical': 0.8
            },
            MetricType.SHARPE_RATIO: {
                'excellent': 2.0,
                'good': 1.5,
                'average': 1.0,
                'poor': 0.5,
                'critical': 0.0
            }
        }
    
    def _initialize_database(self) -> None:
        """Inicializa o banco de dados SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Tabela de predições
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        prediction_id TEXT UNIQUE NOT NULL,
                        pair TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        predicted_value REAL NOT NULL,
                        predicted_class INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        actual_value REAL,
                        actual_class INTEGER,
                        execution_time REAL DEFAULT 0.0,
                        features_hash TEXT,
                        market_conditions TEXT
                    )
                """)
                
                # Tabela de métricas de qualidade
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quality_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_id TEXT UNIQUE NOT NULL,
                        metric_type TEXT NOT NULL,
                        pair TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        value REAL NOT NULL,
                        timestamp DATETIME NOT NULL,
                        window_size INTEGER NOT NULL,
                        threshold_breached BOOLEAN DEFAULT FALSE,
                        severity TEXT
                    )
                """)
                
                # Tabela de alertas
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quality_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT UNIQUE NOT NULL,
                        metric_type TEXT NOT NULL,
                        pair TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        threshold_value REAL NOT NULL,
                        timestamp DATETIME NOT NULL,
                        acknowledged BOOLEAN DEFAULT FALSE,
                        resolved BOOLEAN DEFAULT FALSE
                    )
                """)
                
                # Índices para performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_pair_model ON predictions(pair, model_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_pair_model ON quality_metrics(pair, model_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON quality_metrics(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON quality_alerts(timestamp)")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar banco de dados: {e}")
    
    def _load_historical_data(self) -> None:
        """Carrega dados históricos do banco"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Carrega predições recentes
                df_predictions = pd.read_sql_query("""
                    SELECT * FROM predictions 
                    WHERE timestamp > datetime('now', '-7 days')
                    ORDER BY timestamp DESC
                    LIMIT 10000
                """, conn)
                
                for _, row in df_predictions.iterrows():
                    prediction = PredictionRecord(
                        prediction_id=row['prediction_id'],
                        pair=row['pair'],
                        model_name=row['model_name'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        predicted_value=row['predicted_value'],
                        predicted_class=row['predicted_class'],
                        confidence=row['confidence'],
                        actual_value=row['actual_value'],
                        actual_class=row['actual_class'],
                        execution_time=row['execution_time'],
                        features_hash=row['features_hash'],
                        market_conditions=json.loads(row['market_conditions']) if row['market_conditions'] else None
                    )
                    
                    key = f"{prediction.pair}_{prediction.model_name}"
                    self.prediction_buffer[key].append(prediction)
                
                # Carrega alertas não resolvidos
                df_alerts = pd.read_sql_query("""
                    SELECT * FROM quality_alerts 
                    WHERE resolved = FALSE
                    ORDER BY timestamp DESC
                """, conn)
                
                for _, row in df_alerts.iterrows():
                    alert = QualityAlert(
                        alert_id=row['alert_id'],
                        metric_type=MetricType(row['metric_type']),
                        pair=row['pair'],
                        model_name=row['model_name'],
                        severity=AlertSeverity(row['severity']),
                        message=row['message'],
                        current_value=row['current_value'],
                        threshold_value=row['threshold_value'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        acknowledged=bool(row['acknowledged']),
                        resolved=bool(row['resolved'])
                    )
                    self.alert_buffer.append(alert)
                
                self.logger.info(f"Carregados {len(df_predictions)} predições e {len(df_alerts)} alertas")
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados históricos: {e}")
    
    def record_prediction(self, prediction: PredictionRecord) -> None:
        """Registra uma nova predição"""
        try:
            # Adiciona ao buffer em memória
            key = f"{prediction.pair}_{prediction.model_name}"
            self.prediction_buffer[key].append(prediction)
            
            # Salva no banco de dados
            self._save_prediction_to_db(prediction)
            
            # Atualiza métricas se temos valor real
            if prediction.actual_value is not None and prediction.actual_class is not None:
                self._update_quality_metrics(prediction.pair, prediction.model_name)
            
        except Exception as e:
            self.logger.error(f"Erro ao registrar predição: {e}")
    
    def update_prediction_outcome(self, prediction_id: str, actual_value: float, 
                                actual_class: int) -> None:
        """Atualiza o resultado real de uma predição"""
        try:
            # Atualiza no banco de dados
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE predictions 
                    SET actual_value = ?, actual_class = ?
                    WHERE prediction_id = ?
                """, (actual_value, actual_class, prediction_id))
                conn.commit()
            
            # Atualiza no buffer em memória
            for key, buffer in self.prediction_buffer.items():
                for prediction in buffer:
                    if prediction.prediction_id == prediction_id:
                        prediction.actual_value = actual_value
                        prediction.actual_class = actual_class
                        
                        # Atualiza métricas
                        pair, model_name = key.split('_', 1)
                        self._update_quality_metrics(pair, model_name)
                        break
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar resultado da predição: {e}")
    
    def _save_prediction_to_db(self, prediction: PredictionRecord) -> None:
        """Salva predição no banco de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO predictions 
                    (prediction_id, pair, model_name, timestamp, predicted_value,
                     predicted_class, confidence, actual_value, actual_class,
                     execution_time, features_hash, market_conditions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction.prediction_id,
                    prediction.pair,
                    prediction.model_name,
                    prediction.timestamp.isoformat(),
                    prediction.predicted_value,
                    prediction.predicted_class,
                    prediction.confidence,
                    prediction.actual_value,
                    prediction.actual_class,
                    prediction.execution_time,
                    prediction.features_hash,
                    json.dumps(prediction.market_conditions) if prediction.market_conditions else None
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Erro ao salvar predição no banco: {e}")
    
    def _update_quality_metrics(self, pair: str, model_name: str) -> None:
        """Atualiza métricas de qualidade para um par/modelo"""
        try:
            key = f"{pair}_{model_name}"
            predictions = list(self.prediction_buffer[key])
            
            # Filtra predições com resultados conhecidos
            completed_predictions = [
                p for p in predictions 
                if p.actual_value is not None and p.actual_class is not None
            ]
            
            if len(completed_predictions) < 10:  # Mínimo de predições para calcular métricas
                return
            
            # Calcula métricas para diferentes janelas
            for window_name, window_size in self.window_sizes.items():
                recent_predictions = completed_predictions[-window_size:]
                
                if len(recent_predictions) < 5:
                    continue
                
                metrics = self._calculate_metrics(recent_predictions)
                
                # Salva métricas e verifica thresholds
                for metric_type, value in metrics.items():
                    metric = QualityMetric(
                        metric_id=f"{pair}_{model_name}_{metric_type.value}_{window_name}_{int(time.time())}",
                        metric_type=metric_type,
                        pair=pair,
                        model_name=model_name,
                        value=value,
                        timestamp=datetime.now(),
                        window_size=len(recent_predictions)
                    )
                    
                    # Verifica threshold
                    severity = self._check_threshold(metric_type, value)
                    if severity:
                        metric.threshold_breached = True
                        metric.severity = severity
                        
                        # Cria alerta se necessário
                        self._create_quality_alert(metric)
                    
                    # Adiciona ao buffer e salva
                    metric_key = f"{pair}_{model_name}_{metric_type.value}"
                    self.metric_buffer[metric_key].append(metric)
                    self._save_metric_to_db(metric)
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar métricas de qualidade: {e}")
    
    def _calculate_metrics(self, predictions: List[PredictionRecord]) -> Dict[MetricType, float]:
        """Calcula métricas de qualidade para uma lista de predições"""
        if not predictions or not SKLEARN_AVAILABLE:
            return {}
        
        try:
            # Extrai valores reais e preditos
            y_true = [p.actual_class for p in predictions]
            y_pred = [p.predicted_class for p in predictions]
            y_true_values = [p.actual_value for p in predictions]
            y_pred_values = [p.predicted_value for p in predictions]
            confidences = [p.confidence for p in predictions]
            execution_times = [p.execution_time for p in predictions]
            
            metrics = {}
            
            # Métricas de classificação
            if len(set(y_true)) > 1:  # Verifica se há variação nas classes
                try:
                    metrics[MetricType.ACCURACY] = accuracy_score(y_true, y_pred)
                    metrics[MetricType.PRECISION] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics[MetricType.RECALL] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics[MetricType.F1_SCORE] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                except Exception as e:
                    self.logger.warning(f"Erro ao calcular métricas de classificação: {e}")
            
            # Métricas de confiança
            if confidences:
                metrics[MetricType.PREDICTION_CONFIDENCE] = np.mean(confidences)
            
            # Métricas de latência
            if execution_times:
                metrics[MetricType.PREDICTION_LATENCY] = np.mean(execution_times)
                metrics[MetricType.EXECUTION_TIME] = np.percentile(execution_times, 95)  # P95
            
            # Métricas financeiras (simuladas)
            if y_true_values and y_pred_values:
                # Win rate
                correct_direction = sum(1 for i in range(len(y_true_values)-1) 
                                      if (y_true_values[i+1] - y_true_values[i]) * 
                                         (y_pred_values[i+1] - y_pred_values[i]) > 0)
                metrics[MetricType.WIN_RATE] = correct_direction / max(1, len(y_true_values) - 1)
                
                # Profit factor simulado
                profits = []
                for i in range(len(y_true_values) - 1):
                    actual_change = y_true_values[i+1] - y_true_values[i]
                    predicted_change = y_pred_values[i+1] - y_pred_values[i]
                    if predicted_change != 0:
                        profit = actual_change * np.sign(predicted_change)
                        profits.append(profit)
                
                if profits:
                    positive_profits = [p for p in profits if p > 0]
                    negative_profits = [abs(p) for p in profits if p < 0]
                    
                    if positive_profits and negative_profits:
                        metrics[MetricType.PROFIT_FACTOR] = sum(positive_profits) / sum(negative_profits)
                    elif positive_profits:
                        metrics[MetricType.PROFIT_FACTOR] = 2.0  # Só lucros
                    else:
                        metrics[MetricType.PROFIT_FACTOR] = 0.0  # Só perdas
                    
                    # Sharpe ratio simulado
                    if len(profits) > 1:
                        mean_profit = np.mean(profits)
                        std_profit = np.std(profits)
                        metrics[MetricType.SHARPE_RATIO] = mean_profit / max(std_profit, 0.001)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular métricas: {e}")
            return {}
    
    def _check_threshold(self, metric_type: MetricType, value: float) -> Optional[AlertSeverity]:
        """Verifica se uma métrica violou algum threshold"""
        if metric_type not in self.quality_thresholds:
            return None
        
        thresholds = self.quality_thresholds[metric_type]
        
        # Para métricas onde maior é melhor (accuracy, precision, etc.)
        if metric_type in [MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, 
                          MetricType.F1_SCORE, MetricType.PREDICTION_CONFIDENCE, 
                          MetricType.WIN_RATE, MetricType.PROFIT_FACTOR, MetricType.SHARPE_RATIO]:
            if value < thresholds['critical']:
                return AlertSeverity.EMERGENCY
            elif value < thresholds['poor']:
                return AlertSeverity.CRITICAL
            elif value < thresholds['average']:
                return AlertSeverity.WARNING
        
        # Para métricas onde menor é melhor (latency, execution_time)
        elif metric_type in [MetricType.PREDICTION_LATENCY, MetricType.EXECUTION_TIME]:
            if value > thresholds['critical']:
                return AlertSeverity.EMERGENCY
            elif value > thresholds['poor']:
                return AlertSeverity.CRITICAL
            elif value > thresholds['average']:
                return AlertSeverity.WARNING
        
        return None
    
    def _create_quality_alert(self, metric: QualityMetric) -> None:
        """Cria um alerta de qualidade"""
        try:
            alert_id = f"alert_{metric.pair}_{metric.model_name}_{metric.metric_type.value}_{int(time.time())}"
            
            # Determina threshold violado
            thresholds = self.quality_thresholds.get(metric.metric_type, {})
            threshold_value = 0.0
            
            if metric.severity == AlertSeverity.EMERGENCY:
                threshold_value = thresholds.get('critical', 0.0)
            elif metric.severity == AlertSeverity.CRITICAL:
                threshold_value = thresholds.get('poor', 0.0)
            elif metric.severity == AlertSeverity.WARNING:
                threshold_value = thresholds.get('average', 0.0)
            
            message = f"{metric.metric_type.value.upper()} para {metric.pair}/{metric.model_name} "
            message += f"está em {metric.value:.3f} (threshold: {threshold_value:.3f})"
            
            alert = QualityAlert(
                alert_id=alert_id,
                metric_type=metric.metric_type,
                pair=metric.pair,
                model_name=metric.model_name,
                severity=metric.severity,
                message=message,
                current_value=metric.value,
                threshold_value=threshold_value,
                timestamp=datetime.now()
            )
            
            # Verifica se já existe alerta similar recente
            recent_alerts = [
                a for a in self.alert_buffer
                if (a.metric_type == alert.metric_type and 
                    a.pair == alert.pair and 
                    a.model_name == alert.model_name and
                    not a.resolved and
                    (datetime.now() - a.timestamp).total_seconds() < 3600)  # 1 hora
            ]
            
            if not recent_alerts:
                self.alert_buffer.append(alert)
                self._save_alert_to_db(alert)
                self.logger.warning(f"Alerta criado: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"Erro ao criar alerta: {e}")
    
    def _save_metric_to_db(self, metric: QualityMetric) -> None:
        """Salva métrica no banco de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO quality_metrics 
                    (metric_id, metric_type, pair, model_name, value, timestamp,
                     window_size, threshold_breached, severity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.metric_id,
                    metric.metric_type.value,
                    metric.pair,
                    metric.model_name,
                    metric.value,
                    metric.timestamp.isoformat(),
                    metric.window_size,
                    metric.threshold_breached,
                    metric.severity.value if metric.severity else None
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Erro ao salvar métrica no banco: {e}")
    
    def _save_alert_to_db(self, alert: QualityAlert) -> None:
        """Salva alerta no banco de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO quality_alerts 
                    (alert_id, metric_type, pair, model_name, severity, message,
                     current_value, threshold_value, timestamp, acknowledged, resolved)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    alert.metric_type.value,
                    alert.pair,
                    alert.model_name,
                    alert.severity.value,
                    alert.message,
                    alert.current_value,
                    alert.threshold_value,
                    alert.timestamp.isoformat(),
                    alert.acknowledged,
                    alert.resolved
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Erro ao salvar alerta no banco: {e}")
    
    def start_monitoring(self) -> None:
        """Inicia o monitoramento contínuo"""
        if self.monitoring_active:
            self.logger.warning("Monitoramento já está ativo")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        if self.enable_dashboard:
            self.dashboard_thread = threading.Thread(target=self._start_dashboard, daemon=True)
            self.dashboard_thread.start()
        
        self.logger.info("Monitoramento de qualidade iniciado")
    
    def stop_monitoring(self) -> None:
        """Para o monitoramento contínuo"""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Monitoramento de qualidade parado")
    
    def _monitoring_loop(self) -> None:
        """Loop principal de monitoramento"""
        while self.monitoring_active:
            try:
                # Atualiza métricas para todos os pares/modelos ativos
                for key in list(self.prediction_buffer.keys()):
                    if '_' in key:
                        pair, model_name = key.split('_', 1)
                        self._update_quality_metrics(pair, model_name)
                
                # Limpa cache antigo
                self._cleanup_cache()
                
                # Limpa alertas antigos
                self._cleanup_old_alerts()
                
                # Aguarda próximo ciclo
                time.sleep(30)  # Atualiza a cada 30 segundos
                
            except Exception as e:
                self.logger.error(f"Erro no loop de monitoramento: {e}")
                time.sleep(60)  # Aguarda mais tempo em caso de erro
    
    def _cleanup_cache(self) -> None:
        """Limpa cache antigo"""
        current_time = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.metrics_cache.items()
            if (current_time - timestamp).total_seconds() > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.metrics_cache[key]
    
    def _cleanup_old_alerts(self) -> None:
        """Limpa alertas antigos resolvidos"""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        # Remove alertas antigos do buffer
        self.alert_buffer = [
            alert for alert in self.alert_buffer
            if not alert.resolved or alert.timestamp > cutoff_time
        ]
        
        # Remove alertas antigos do banco
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM quality_alerts 
                    WHERE resolved = TRUE AND timestamp < ?
                """, (cutoff_time.isoformat(),))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Erro ao limpar alertas antigos: {e}")
    
    def get_quality_summary(self, pair: str = None, model_name: str = None, 
                           hours: int = 24) -> Dict[str, Any]:
        """Retorna resumo da qualidade das predições"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filtra predições
            all_predictions = []
            for key, buffer in self.prediction_buffer.items():
                key_pair, key_model = key.split('_', 1)
                
                if pair and key_pair != pair:
                    continue
                if model_name and key_model != model_name:
                    continue
                
                recent_predictions = [
                    p for p in buffer 
                    if p.timestamp > cutoff_time and 
                       p.actual_value is not None and 
                       p.actual_class is not None
                ]
                all_predictions.extend(recent_predictions)
            
            if not all_predictions:
                return {"error": "Nenhuma predição encontrada no período"}
            
            # Calcula métricas gerais
            metrics = self._calculate_metrics(all_predictions)
            
            # Conta alertas ativos
            active_alerts = [
                alert for alert in self.alert_buffer
                if not alert.resolved and alert.timestamp > cutoff_time
            ]
            
            # Agrupa alertas por severidade
            alerts_by_severity = defaultdict(int)
            for alert in active_alerts:
                alerts_by_severity[alert.severity.value] += 1
            
            # Determina qualidade geral
            overall_quality = self._determine_overall_quality(metrics)
            
            return {
                "period_hours": hours,
                "total_predictions": len(all_predictions),
                "overall_quality": overall_quality.value,
                "metrics": {k.value: v for k, v in metrics.items()},
                "active_alerts": len(active_alerts),
                "alerts_by_severity": dict(alerts_by_severity),
                "pairs_monitored": len(set(p.pair for p in all_predictions)),
                "models_monitored": len(set(p.model_name for p in all_predictions)),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar resumo de qualidade: {e}")
            return {"error": str(e)}
    
    def _determine_overall_quality(self, metrics: Dict[MetricType, float]) -> PredictionQuality:
        """Determina qualidade geral baseada nas métricas"""
        if not metrics:
            return PredictionQuality.CRITICAL
        
        # Pesos para diferentes métricas
        weights = {
            MetricType.ACCURACY: 0.3,
            MetricType.F1_SCORE: 0.2,
            MetricType.PRECISION: 0.15,
            MetricType.RECALL: 0.15,
            MetricType.WIN_RATE: 0.1,
            MetricType.PROFIT_FACTOR: 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric_type, value in metrics.items():
            if metric_type in weights:
                # Normaliza valor para 0-1
                if metric_type == MetricType.PROFIT_FACTOR:
                    normalized_value = min(value / 2.0, 1.0)  # Profit factor de 2.0 = 100%
                elif metric_type == MetricType.SHARPE_RATIO:
                    normalized_value = min(max(value, 0) / 2.0, 1.0)  # Sharpe de 2.0 = 100%
                else:
                    normalized_value = min(max(value, 0), 1.0)
                
                weighted_score += normalized_value * weights[metric_type]
                total_weight += weights[metric_type]
        
        if total_weight == 0:
            return PredictionQuality.CRITICAL
        
        final_score = weighted_score / total_weight
        
        if final_score >= 0.8:
            return PredictionQuality.EXCELLENT
        elif final_score >= 0.7:
            return PredictionQuality.GOOD
        elif final_score >= 0.6:
            return PredictionQuality.AVERAGE
        elif final_score >= 0.5:
            return PredictionQuality.POOR
        else:
            return PredictionQuality.CRITICAL
    
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Dict]:
        """Retorna alertas ativos"""
        active_alerts = [alert for alert in self.alert_buffer if not alert.resolved]
        
        if severity:
            active_alerts = [alert for alert in active_alerts if alert.severity == severity]
        
        return [asdict(alert) for alert in active_alerts]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Reconhece um alerta"""
        try:
            # Atualiza no buffer
            for alert in self.alert_buffer:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    break
            
            # Atualiza no banco
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE quality_alerts 
                    SET acknowledged = TRUE 
                    WHERE alert_id = ?
                """, (alert_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"Erro ao reconhecer alerta: {e}")
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve um alerta"""
        try:
            # Atualiza no buffer
            for alert in self.alert_buffer:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    break
            
            # Atualiza no banco
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE quality_alerts 
                    SET resolved = TRUE 
                    WHERE alert_id = ?
                """, (alert_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"Erro ao resolver alerta: {e}")
            return False
    
    def _setup_dashboard(self) -> None:
        """Configura o dashboard web"""
        if not DASH_AVAILABLE:
            self.logger.warning("Dash não disponível. Dashboard desabilitado.")
            return
        
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Layout do dashboard
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Monitor de Qualidade das Predições", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Resumo geral
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Qualidade Geral", className="card-title"),
                            html.H2(id="overall-quality", className="text-center")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Predições (24h)", className="card-title"),
                            html.H2(id="total-predictions", className="text-center")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Alertas Ativos", className="card-title"),
                            html.H2(id="active-alerts", className="text-center")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Modelos Ativos", className="card-title"),
                            html.H2(id="active-models", className="text-center")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Gráficos de métricas
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="metrics-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="alerts-chart")
                ], width=6)
            ], className="mb-4"),
            
            # Tabela de alertas
            dbc.Row([
                dbc.Col([
                    html.H4("Alertas Recentes"),
                    html.Div(id="alerts-table")
                ])
            ]),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Atualiza a cada 30 segundos
                n_intervals=0
            )
        ], fluid=True)
        
        # Callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self) -> None:
        """Configura callbacks do dashboard"""
        @self.app.callback(
            [Output('overall-quality', 'children'),
             Output('total-predictions', 'children'),
             Output('active-alerts', 'children'),
             Output('active-models', 'children'),
             Output('metrics-chart', 'figure'),
             Output('alerts-chart', 'figure'),
             Output('alerts-table', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            try:
                # Obtém resumo
                summary = self.get_quality_summary(hours=24)
                
                if "error" in summary:
                    return "N/A", "0", "0", "0", {}, {}, "Nenhum dado disponível"
                
                # Cards de resumo
                overall_quality = summary.get('overall_quality', 'N/A')
                total_predictions = summary.get('total_predictions', 0)
                active_alerts = summary.get('active_alerts', 0)
                active_models = summary.get('models_monitored', 0)
                
                # Gráfico de métricas
                metrics_fig = self._create_metrics_chart(summary.get('metrics', {}))
                
                # Gráfico de alertas
                alerts_fig = self._create_alerts_chart(summary.get('alerts_by_severity', {}))
                
                # Tabela de alertas
                alerts_table = self._create_alerts_table()
                
                return (overall_quality, str(total_predictions), str(active_alerts), 
                       str(active_models), metrics_fig, alerts_fig, alerts_table)
                
            except Exception as e:
                self.logger.error(f"Erro ao atualizar dashboard: {e}")
                return "Erro", "0", "0", "0", {}, {}, "Erro ao carregar dados"
    
    def _create_metrics_chart(self, metrics: Dict[str, float]) -> go.Figure:
        """Cria gráfico de métricas"""
        if not PLOTLY_AVAILABLE or not metrics:
            return go.Figure()
        
        fig = go.Figure()
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig.add_trace(go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color=['green' if v >= 0.7 else 'orange' if v >= 0.5 else 'red' for v in metric_values]
        ))
        
        fig.update_layout(
            title="Métricas de Qualidade",
            xaxis_title="Métrica",
            yaxis_title="Valor",
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def _create_alerts_chart(self, alerts_by_severity: Dict[str, int]) -> go.Figure:
        """Cria gráfico de alertas por severidade"""
        if not PLOTLY_AVAILABLE or not alerts_by_severity:
            return go.Figure()
        
        fig = go.Figure()
        
        colors = {
            'info': 'blue',
            'warning': 'orange',
            'critical': 'red',
            'emergency': 'darkred'
        }
        
        fig.add_trace(go.Pie(
            labels=list(alerts_by_severity.keys()),
            values=list(alerts_by_severity.values()),
            marker_colors=[colors.get(k, 'gray') for k in alerts_by_severity.keys()]
        ))
        
        fig.update_layout(title="Alertas por Severidade")
        
        return fig
    
    def _create_alerts_table(self) -> html.Div:
        """Cria tabela de alertas"""
        active_alerts = self.get_active_alerts()
        
        if not active_alerts:
            return html.P("Nenhum alerta ativo")
        
        # Limita a 10 alertas mais recentes
        recent_alerts = sorted(active_alerts, key=lambda x: x['timestamp'], reverse=True)[:10]
        
        table_rows = []
        for alert in recent_alerts:
            severity_color = {
                'info': 'info',
                'warning': 'warning',
                'critical': 'danger',
                'emergency': 'dark'
            }.get(alert['severity'], 'secondary')
            
            row = dbc.Row([
                dbc.Col(alert['pair'], width=2),
                dbc.Col(alert['model_name'], width=2),
                dbc.Col(alert['metric_type'], width=2),
                dbc.Col(dbc.Badge(alert['severity'], color=severity_color), width=2),
                dbc.Col(f"{alert['current_value']:.3f}", width=2),
                dbc.Col(alert['timestamp'][:19], width=2)
            ], className="mb-1")
            table_rows.append(row)
        
        header = dbc.Row([
            dbc.Col(html.B("Par"), width=2),
            dbc.Col(html.B("Modelo"), width=2),
            dbc.Col(html.B("Métrica"), width=2),
            dbc.Col(html.B("Severidade"), width=2),
            dbc.Col(html.B("Valor"), width=2),
            dbc.Col(html.B("Timestamp"), width=2)
        ], className="mb-2")
        
        return html.Div([header] + table_rows)
    
    def _start_dashboard(self) -> None:
        """Inicia o servidor do dashboard"""
        try:
            self.app.run(host='0.0.0.0', port=self.dashboard_port, debug=False)
        except Exception as e:
            self.logger.error(f"Erro ao iniciar dashboard: {e}")
    
    def export_metrics_report(self, pair: str = None, model_name: str = None, 
                             days: int = 7) -> str:
        """Exporta relatório de métricas"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Coleta dados do banco
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM quality_metrics 
                    WHERE timestamp > ?
                """
                params = [cutoff_time.isoformat()]
                
                if pair:
                    query += " AND pair = ?"
                    params.append(pair)
                
                if model_name:
                    query += " AND model_name = ?"
                    params.append(model_name)
                
                query += " ORDER BY timestamp DESC"
                
                df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                return "Nenhum dado encontrado para o período especificado"
            
            # Gera relatório
            report_path = f"reports/quality_report_{int(time.time())}.json"
            Path("reports").mkdir(exist_ok=True)
            
            report_data = {
                "generated_at": datetime.now().isoformat(),
                "period_days": days,
                "pair_filter": pair,
                "model_filter": model_name,
                "total_metrics": len(df),
                "metrics_by_type": df['metric_type'].value_counts().to_dict(),
                "metrics_by_pair": df['pair'].value_counts().to_dict(),
                "metrics_by_model": df['model_name'].value_counts().to_dict(),
                "average_values": df.groupby('metric_type')['value'].mean().to_dict(),
                "threshold_breaches": df['threshold_breached'].sum(),
                "detailed_metrics": df.to_dict('records')
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Relatório exportado para {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Erro ao exportar relatório: {e}")
            return f"Erro: {e}"

if __name__ == "__main__":
    # Exemplo de uso
    monitor = PredictionQualityMonitor(enable_dashboard=True)
    
    # Inicia monitoramento
    monitor.start_monitoring()
    
    # Simula algumas predições
    import uuid
    
    for i in range(100):
        prediction = PredictionRecord(
            prediction_id=str(uuid.uuid4()),
            pair="EURUSD",
            model_name="xgboost_v1",
            timestamp=datetime.now() - timedelta(minutes=i),
            predicted_value=1.1000 + np.random.normal(0, 0.001),
            predicted_class=np.random.randint(0, 2),
            confidence=np.random.uniform(0.6, 0.95),
            actual_value=1.1000 + np.random.normal(0, 0.001),
            actual_class=np.random.randint(0, 2),
            execution_time=np.random.uniform(0.05, 0.2)
        )
        
        monitor.record_prediction(prediction)
    
    # Mostra resumo
    summary = monitor.get_quality_summary()
    print(f"Resumo de qualidade: {json.dumps(summary, indent=2)}")
    
    # Mostra alertas
    alerts = monitor.get_active_alerts()
    print(f"Alertas ativos: {len(alerts)}")
    
    # Exporta relatório
    report_path = monitor.export_metrics_report()
    print(f"Relatório exportado: {report_path}")
    
    print(f"Dashboard disponível em: http://localhost:{monitor.dashboard_port}")
    
    try:
        # Mantém o programa rodando
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("Monitoramento parado")