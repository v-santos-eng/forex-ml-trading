import logging
import json
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Importa os módulos desenvolvidos
from ml_scheduler import MLModelScheduler, TriggerType, SchedulerStatus
from performance_monitor import PerformanceDegradationMonitor, AlertLevel
from hyperparameter_optimizer import HyperparameterOptimizer, OptimizationMethod
from prediction_quality_monitor import PredictionQualityMonitor, PredictionRecord, MetricType
from streaming_adapter import StreamingAdapter
from mt5_data_provider import MT5DataProvider
from mt5_connector import MT5Connector
from comprehensive_analysis_logger import ComprehensiveAnalysisLogger, get_comprehensive_logger, initialize_comprehensive_logger
from detailed_decision_logger import DetailedDecisionLogger, get_detailed_logger
from trading_decision_integrator import TradingDecisionIntegrator, initialize_trading_integrator
# from scripts.agno_memory_system import ForexMemorySystem
# Sistema de memória será integrado posteriormente quando as dependências estiverem disponíveis

# Importa configurações
try:
    from config.config import ForexConfig
except ImportError:
    # Fallback para configuração básica
    class ForexConfig:
        CURRENCY_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        DEFAULT_MODEL = 'xgboost'
        OPENAI_API_KEY = None

class PipelineStatus(Enum):
    """Status do pipeline de produção"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class ComponentStatus(Enum):
    """Status dos componentes individuais"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class PipelineConfig:
    """Configuração do pipeline de produção"""
    # Configurações gerais
    enable_scheduler: bool = True
    enable_performance_monitor: bool = True
    enable_hyperparameter_optimizer: bool = True
    enable_quality_monitor: bool = True
    enable_memory_system: bool = True
    enable_dashboard: bool = True
    enable_streaming: bool = True
    
    # Configurações de retreinamento
    auto_retrain_enabled: bool = True
    retrain_interval_hours: int = 24
    performance_threshold: float = 0.6
    quality_threshold: float = 0.65
    
    # Configurações de otimização
    optimize_on_retrain: bool = True
    optimization_method: str = "optuna_tpe"
    optimization_trials: int = 50
    
    # Configurações de monitoramento
    monitoring_interval_seconds: int = 30
    alert_cooldown_minutes: int = 60
    dashboard_port: int = 8050
    
    # Configurações de streaming
    streaming_host: str = "0.0.0.0"
    streaming_port: int = 8765
    streaming_metrics_interval_seconds: int = 60
    min_price_change_threshold: float = 0.0001
    
    # Configurações de memória
    memory_retention_days: int = 90
    learning_session_interval_hours: int = 12
    
    # Configurações de dados
    currency_pairs: List[str] = None
    models_to_monitor: List[str] = None
    
    def __post_init__(self):
        if self.currency_pairs is None:
            self.currency_pairs = ForexConfig.CURRENCY_PAIRS
        if self.models_to_monitor is None:
            self.models_to_monitor = ['xgboost', 'random_forest', 'gradient_boosting']

@dataclass
class ComponentHealth:
    """Saúde de um componente"""
    name: str
    status: ComponentStatus
    last_update: datetime
    error_count: int = 0
    last_error: Optional[str] = None
    uptime_seconds: float = 0.0
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}

class MLProductionPipeline:
    """Pipeline completo de ML em produção para Forex"""
    
    def __init__(self, config: PipelineConfig = None, config_path: str = None):
        # Configuração
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                self.config = PipelineConfig(**config_dict)
        else:
            self.config = config or PipelineConfig()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Status do pipeline
        self.status = PipelineStatus.STOPPED
        self.start_time: Optional[datetime] = None
        self.component_health: Dict[str, ComponentHealth] = {}
        
        # Componentes do pipeline
        self.scheduler: Optional[MLModelScheduler] = None
        self.performance_monitor: Optional[PerformanceDegradationMonitor] = None
        self.hyperparameter_optimizer: Optional[HyperparameterOptimizer] = None
        self.quality_monitor: Optional[PredictionQualityMonitor] = None
        # self.memory_system: Optional[ForexMemorySystem] = None
        self.memory_system = None  # Será integrado posteriormente
        
        # Componentes de logging detalhado
        self.comprehensive_logger: Optional[ComprehensiveAnalysisLogger] = None
        self.detailed_logger: Optional[DetailedDecisionLogger] = None
        self.trading_integrator: Optional[TradingDecisionIntegrator] = None
        
        # Componentes de streaming de dados em tempo real
        self.mt5_connector: Optional[MT5Connector] = None
        self.mt5_data_provider: Optional[MT5DataProvider] = None
        self.streaming_adapter: Optional[StreamingAdapter] = None
        self.price_callbacks: Dict[str, List[Callable]] = {}
        self.last_prices: Dict[str, Dict[str, float]] = {}
        
        # Threading e controle
        self.main_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Métricas e estatísticas
        self.pipeline_metrics = {
            'total_predictions': 0,
            'successful_retrains': 0,
            'failed_retrains': 0,
            'alerts_generated': 0,
            'optimizations_completed': 0,
            'memory_sessions': 0,
            'streaming_messages': 0,
            'streaming_latency_ms': 0.0
        }
        
        # Cache para otimização
        self.prediction_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = 300  # 5 minutos
        
        self._initialize_components()
    
    def _setup_logger(self) -> logging.Logger:
        """Configura o sistema de logging"""
        logger = logging.getLogger("MLProductionPipeline")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Cria diretório de logs
            Path("logs").mkdir(exist_ok=True)
            
            # File handler
            handler = logging.FileHandler("logs/ml_production_pipeline.log")
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
    
    def _initialize_components(self) -> None:
        """Inicializa todos os componentes do pipeline"""
        try:
            self.logger.info("Inicializando componentes do pipeline...")
            
            # Inicializa agendador
            if self.config.enable_scheduler:
                self.scheduler = MLModelScheduler()
                self._register_component_health("scheduler", ComponentStatus.INACTIVE)
                self.logger.info("Agendador ML inicializado")
            
            # Inicializa monitor de performance
            if self.config.enable_performance_monitor:
                self.performance_monitor = PerformanceDegradationMonitor()
                self._register_component_health("performance_monitor", ComponentStatus.INACTIVE)
                self.logger.info("Monitor de performance inicializado")
            
            # Inicializa otimizador de hiperparâmetros
            if self.config.enable_hyperparameter_optimizer:
                self.hyperparameter_optimizer = HyperparameterOptimizer()
                self._register_component_health("hyperparameter_optimizer", ComponentStatus.INACTIVE)
                self.logger.info("Otimizador de hiperparâmetros inicializado")
            
            # Inicializa monitor de qualidade
            if self.config.enable_quality_monitor:
                self.quality_monitor = PredictionQualityMonitor(
                    enable_dashboard=self.config.enable_dashboard,
                    dashboard_port=self.config.dashboard_port
                )
                self._register_component_health("quality_monitor", ComponentStatus.INACTIVE)
                self.logger.info("Monitor de qualidade inicializado")
            
            # Inicializa sistema de memória
            # if self.config.enable_memory_system:
            #     self.memory_system = ForexMemorySystem()
            #     self._register_component_health("memory_system", ComponentStatus.INACTIVE)
            #     self.logger.info("Sistema de memória inicializado")
            self.logger.info("Sistema de memória será integrado posteriormente")
            
            # Inicializa componentes de logging detalhado
            try:
                # Inicializa logger abrangente
                initialize_comprehensive_logger()
                self.comprehensive_logger = get_comprehensive_logger()
                self._register_component_health("comprehensive_logger", ComponentStatus.ACTIVE)
                self.logger.info("Logger abrangente de análises inicializado")
                
                # Inicializa logger de decisões detalhadas
                self.detailed_logger = get_detailed_logger()
                self._register_component_health("detailed_logger", ComponentStatus.ACTIVE)
                self.logger.info("Logger de decisões detalhadas inicializado")
                
                # Inicializa integrador de trading
                initialize_trading_integrator()
                self.trading_integrator = TradingDecisionIntegrator()
                self._register_component_health("trading_integrator", ComponentStatus.ACTIVE)
                self.logger.info("Integrador de decisões de trading inicializado")
                
            except Exception as e:
                self.logger.error(f"Erro ao inicializar componentes de logging: {e}")
                # Não falha o pipeline se o logging falhar
                pass
            
            # Inicializa componentes de streaming de dados em tempo real
            if self.config.enable_streaming:
                # Inicializa dicionários para callbacks e preços
                self.price_callbacks = {}
                self.last_prices = {}
                
                # Inicializa conexão com MT5
                self.mt5_connector = MT5Connector()
                
                # Inicializa provedor de dados MT5
                if self.mt5_connector.connect():
                    self.mt5_data_provider = MT5DataProvider(self.mt5_connector)
                    
                    # Inicializa adaptador de streaming
                    self.streaming_adapter = StreamingAdapter(
                        host=self.config.streaming_host,
                        port=self.config.streaming_port,
                        mt5_data_provider=self.mt5_data_provider
                    )
                    self._register_component_health("streaming_adapter", ComponentStatus.INACTIVE)
                    self.logger.info("Sistema de streaming de dados em tempo real inicializado")
                else:
                    self.logger.error("Falha ao conectar ao MT5. Sistema de streaming não será iniciado.")
            
            self.logger.info("Todos os componentes inicializados com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar componentes: {e}")
            self.status = PipelineStatus.ERROR
            raise
    
    def _register_component_health(self, component_name: str, status: ComponentStatus) -> None:
        """Registra saúde de um componente"""
        self.component_health[component_name] = ComponentHealth(
            name=component_name,
            status=status,
            last_update=datetime.now()
        )
    
    def _update_component_health(self, component_name: str, status: ComponentStatus, 
                               error: str = None, metrics: Dict[str, float] = None) -> None:
        """Atualiza saúde de um componente"""
        if component_name in self.component_health:
            health = self.component_health[component_name]
            health.status = status
            health.last_update = datetime.now()
            
            if error:
                health.error_count += 1
                health.last_error = error
            
            if metrics:
                health.performance_metrics.update(metrics)
            
            # Calcula uptime
            if self.start_time:
                health.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
    
    def start_pipeline(self) -> bool:
        """Inicia o pipeline completo"""
        try:
            if self.status == PipelineStatus.RUNNING:
                self.logger.warning("Pipeline já está rodando")
                return True
            
            self.logger.info("Iniciando pipeline de produção ML...")
            self.status = PipelineStatus.STARTING
            self.start_time = datetime.now()
            
            # Inicia componentes
            self._start_components()
            
            # Inicia thread principal
            self.stop_event.clear()
            self.main_thread = threading.Thread(target=self._main_pipeline_loop, daemon=True)
            self.main_thread.start()
            
            self.status = PipelineStatus.RUNNING
            self.logger.info("Pipeline de produção iniciado com sucesso")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar pipeline: {e}")
            self.status = PipelineStatus.ERROR
            return False
    
    def _start_components(self) -> None:
        """Inicia todos os componentes"""
        try:
            # Inicia agendador
            if self.scheduler:
                self.scheduler.start_scheduler()
                self._update_component_health("scheduler", ComponentStatus.ACTIVE)
                
                # Agenda retreinamentos iniciais
                self._schedule_initial_retrains()
            
            # Inicia monitor de performance
            if self.performance_monitor:
                self.performance_monitor.start_monitoring()
                self._update_component_health("performance_monitor", ComponentStatus.ACTIVE)
            
            # Inicia monitor de qualidade
            if self.quality_monitor:
                self.quality_monitor.start_monitoring()
                self._update_component_health("quality_monitor", ComponentStatus.ACTIVE)
            
            # Sistema de memória e otimizador não precisam de start explícito
            if self.memory_system:
                self._update_component_health("memory_system", ComponentStatus.ACTIVE)
            
            if self.hyperparameter_optimizer:
                self._update_component_health("hyperparameter_optimizer", ComponentStatus.ACTIVE)
                
            # Inicia sistema de streaming de dados em tempo real
            if self.streaming_adapter:
                success = self.streaming_adapter.start()
                if success:
                    self._update_component_health("streaming_adapter", ComponentStatus.ACTIVE)
                    
                    # Registra callbacks para os pares de moedas configurados
                    for pair in self.config.currency_pairs:
                        # Criar wrapper para compatibilizar assinaturas de callback
                        def create_callback_wrapper(symbol):
                            def wrapper(price_data):
                                # Adicionar símbolo aos dados se não estiver presente
                                if 'symbol' not in price_data:
                                    price_data['symbol'] = symbol
                                return self._on_price_update(price_data)
                            return wrapper
                        
                        callback_wrapper = create_callback_wrapper(pair)
                        self.streaming_adapter.subscribe(pair, callback_wrapper)
                        self.logger.info(f"Registrado para receber atualizações em tempo real de {pair}")
                else:
                    self._update_component_health("streaming_adapter", ComponentStatus.ERROR, "Falha ao iniciar adaptador de streaming")
                    self.logger.error("Falha ao iniciar sistema de streaming de dados em tempo real")
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar componentes: {e}")
            raise
    
    def _schedule_initial_retrains(self) -> None:
        """Agenda retreinamentos iniciais para todos os pares/modelos"""
        if not self.scheduler:
            return
        
        try:
            for pair in self.config.currency_pairs:
                for model in self.config.models_to_monitor:
                    # Agenda retreinamento periódico
                    task_id = self.scheduler.schedule_retraining(
                        pair=pair,
                        trigger_type=TriggerType.TIME_INTERVAL,
                        scheduled_time=None,  # Será agendado automaticamente
                        priority=2  # 2=média prioridade
                    )
                    
                    self.logger.info(f"Retreinamento agendado para {pair}/{model}: {task_id}")
            
        except Exception as e:
            self.logger.error(f"Erro ao agendar retreinamentos iniciais: {e}")
    
    def _main_pipeline_loop(self) -> None:
        """Loop principal do pipeline"""
        self.logger.info("Iniciando loop principal do pipeline")
        
        # Contador para métricas de streaming
        streaming_metrics_counter = 0
        
        while not self.stop_event.is_set():
            try:
                # Verifica saúde dos componentes
                self._check_component_health()
                
                # Processa alertas de performance
                self._process_performance_alerts()
                
                # Executa sessões de aprendizado contínuo
                self._run_learning_sessions()
                
                # Otimiza modelos se necessário
                self._check_optimization_needs()
                
                # Limpa cache antigo
                self._cleanup_cache()
                
                # Atualiza métricas do pipeline
                self._update_pipeline_metrics()
                
                # Verifica métricas de streaming periodicamente
                streaming_metrics_counter += 1
                if self.streaming_adapter and streaming_metrics_counter >= (self.config.streaming_metrics_interval_seconds // self.config.monitoring_interval_seconds):
                    self._check_streaming_metrics()
                    streaming_metrics_counter = 0
                
                # Aguarda próximo ciclo
                self.stop_event.wait(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Erro no loop principal: {e}")
                self._update_component_health("pipeline", ComponentStatus.ERROR, str(e))
                time.sleep(60)  # Aguarda mais tempo em caso de erro
    
    def _check_component_health(self) -> None:
        """Verifica saúde de todos os componentes"""
        try:
            current_time = datetime.now()
            
            for component_name, health in self.component_health.items():
                # Verifica se componente está responsivo
                time_since_update = (current_time - health.last_update).total_seconds()
                
                if time_since_update > 300:  # 5 minutos sem atualização
                    self._update_component_health(
                        component_name, 
                        ComponentStatus.ERROR, 
                        "Componente não responsivo"
                    )
                    self.logger.warning(f"Componente {component_name} não responsivo")
                
                # Verifica métricas específicas
                if component_name == "scheduler" and self.scheduler:
                    try:
                        status = self.scheduler.get_status()
                        self.logger.debug(f"Status do scheduler: {status}")
                        self.logger.debug(f"Tipo do scheduler: {type(self.scheduler)}")
                        self.logger.debug(f"Tipo do tasks: {type(getattr(self.scheduler, 'tasks', None))}")
                        
                        # O scheduler já retorna contadores, não listas
                        metrics = {
                            'pending_tasks': status.get('pending_tasks', 0),
                            'completed_tasks': status.get('completed_tasks', 0),
                            'failed_tasks': status.get('failed_tasks', 0),
                            'running_tasks': status.get('running_tasks', 0)
                        }
                        self._update_component_health(component_name, ComponentStatus.ACTIVE, metrics=metrics)
                    except Exception as scheduler_error:
                        self.logger.error(f"Erro específico do scheduler: {scheduler_error}")
                        self.logger.error(f"Scheduler object: {self.scheduler}")
                        if hasattr(self.scheduler, 'tasks'):
                            self.logger.error(f"Tasks type: {type(self.scheduler.tasks)}, value: {self.scheduler.tasks}")
                
                elif component_name == "quality_monitor" and self.quality_monitor:
                    try:
                        summary = self.quality_monitor.get_quality_summary(hours=1)
                        if "error" not in summary:
                            metrics = {
                                'predictions_1h': summary.get('total_predictions', 0),
                                'active_alerts': summary.get('active_alerts', 0),
                                'overall_quality': self._quality_to_score(summary.get('overall_quality', 'critical'))
                            }
                            self._update_component_health(component_name, ComponentStatus.ACTIVE, metrics=metrics)
                    except Exception as quality_error:
                        self.logger.error(f"Erro específico do quality_monitor: {quality_error}")
        
        except Exception as e:
            self.logger.error(f"Erro ao verificar saúde dos componentes: {e}")
            import traceback
            self.logger.error(f"Traceback completo: {traceback.format_exc()}")
    
    def _quality_to_score(self, quality: str) -> float:
        """Converte qualidade textual para score numérico"""
        quality_scores = {
            'excellent': 1.0,
            'good': 0.8,
            'average': 0.6,
            'poor': 0.4,
            'critical': 0.2
        }
        return quality_scores.get(quality.lower(), 0.0)
    
    def _process_performance_alerts(self) -> None:
        """Processa alertas de performance e toma ações"""
        try:
            if not self.performance_monitor:
                return
            
            # Obtém alertas críticos
            critical_alerts = self.performance_monitor.get_alerts(
                alert_level=AlertLevel.CRITICAL,
                unacknowledged_only=True
            )
            
            for alert in critical_alerts:
                self.logger.warning(f"Processando alerta crítico: {alert['message']}")
                
                # Agenda retreinamento urgente se necessário
                metric_type_str = str(alert['metric_type']).lower() if hasattr(alert['metric_type'], 'name') else str(alert['metric_type']).lower()
                if "accuracy" in metric_type_str or "performance" in metric_type_str:
                    if self.scheduler:
                        task_id = self.scheduler.schedule_retraining(
                            pair=alert['pair'],
                            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                            scheduled_time=None,  # Imediato
                            priority=1  # 1=alta prioridade
                        )
                        
                        self.logger.info(f"Retreinamento urgente agendado: {task_id}")
                        self.pipeline_metrics['alerts_generated'] += 1
                
                # Reconhece o alerta
                self.performance_monitor.acknowledge_alert(alert['alert_id'])
        
        except Exception as e:
            self.logger.error(f"Erro ao processar alertas de performance: {e}")
    
    def _run_learning_sessions(self) -> None:
        """Executa sessões de aprendizado contínuo"""
        try:
            if not self.memory_system:
                return
            
            # Verifica se é hora de executar sessão de aprendizado
            last_session_key = "last_learning_session"
            last_session = self.pipeline_metrics.get(last_session_key, 0)
            current_time = time.time()
            
            session_interval = self.config.learning_session_interval_hours * 3600
            
            if current_time - last_session >= session_interval:
                self.logger.info("Iniciando sessão de aprendizado contínuo")
                
                # Executa sessão de aprendizado
            # learning_results = self.memory_system.continuous_learning_session()
            # 
            # # Processa insights
            # insights = learning_results.get('learning_insights', '')
            # if "retraining" in insights.lower() or "retrain" in insights.lower():
            #     self.logger.info("Sistema de memória recomenda retreinamento")
            #     # Pode agendar retreinamentos baseado nos insights
            pass  # Sistema de memória será integrado posteriormente
            
            self.pipeline_metrics[last_session_key] = current_time
            self.pipeline_metrics['memory_sessions'] += 1
            
            self.logger.info("Sessão de aprendizado concluída")
        
        except Exception as e:
            self.logger.error(f"Erro na sessão de aprendizado: {e}")
    
    def _check_optimization_needs(self) -> None:
        """Verifica se algum modelo precisa de otimização"""
        try:
            if not self.hyperparameter_optimizer or not self.config.optimize_on_retrain:
                return
            
            # Verifica modelos com performance baixa
            if self.quality_monitor:
                for pair in self.config.currency_pairs:
                    for model in self.config.models_to_monitor:
                        summary = self.quality_monitor.get_quality_summary(
                            pair=pair, model_name=model, hours=24
                        )
                        
                        if "error" not in summary:
                            overall_quality = summary.get('overall_quality', 'critical')
                            quality_score = self._quality_to_score(overall_quality)
                            
                            if quality_score < self.config.quality_threshold:
                                self.logger.info(f"Otimização necessária para {pair}/{model}")
                                
                                # Executa otimização assíncrona
                                self.executor.submit(
                                    self._optimize_model_async,
                                    pair, model
                                )
        
        except Exception as e:
            self.logger.error(f"Erro ao verificar necessidades de otimização: {e}")
    
    def _optimize_model_async(self, pair: str, model_name: str) -> None:
        """Otimiza modelo de forma assíncrona"""
        try:
            self.logger.info(f"Iniciando otimização para {pair}/{model_name}")
            
            # Simula dados para otimização (em produção, carregaria dados reais)
            X_train = np.random.randn(1000, 10)
            y_train = np.random.randint(0, 2, 1000)
            
            # Executa otimização
            result = self.hyperparameter_optimizer.optimize_model(
                model_type=model_name,
                X_train=X_train,
                y_train=y_train,
                method=OptimizationMethod(self.config.optimization_method),
                n_trials=self.config.optimization_trials
            )
            
            if result['success']:
                self.logger.info(f"Otimização concluída para {pair}/{model_name}: {result['best_score']:.4f}")
                self.pipeline_metrics['optimizations_completed'] += 1
            else:
                self.logger.error(f"Falha na otimização para {pair}/{model_name}: {result['error']}")
        
        except Exception as e:
            self.logger.error(f"Erro na otimização assíncrona: {e}")
    
    def _cleanup_cache(self) -> None:
        """Limpa cache antigo"""
        try:
            current_time = datetime.now()
            expired_keys = [
                key for key, (_, timestamp) in self.prediction_cache.items()
                if (current_time - timestamp).total_seconds() > self.cache_ttl
            ]
            
            for key in expired_keys:
                del self.prediction_cache[key]
        
        except Exception as e:
            self.logger.error(f"Erro ao limpar cache: {e}")
    
    def _update_pipeline_metrics(self) -> None:
        """Atualiza métricas do pipeline"""
        try:
            # Atualiza contadores baseado nos componentes
            if self.quality_monitor:
                summary = self.quality_monitor.get_quality_summary(hours=24)
                if "error" not in summary:
                    self.pipeline_metrics['total_predictions'] = summary.get('total_predictions', 0)
            
            if self.scheduler:
                status = self.scheduler.get_status()
                # O scheduler retorna contadores, não listas
                self.pipeline_metrics['successful_retrains'] = status.get('completed_tasks', 0)
                self.pipeline_metrics['failed_retrains'] = status.get('failed_tasks', 0)
                self.pipeline_metrics['pending_retrains'] = status.get('pending_tasks', 0)
                self.pipeline_metrics['running_retrains'] = status.get('running_tasks', 0)
            
            # Atualiza métricas de streaming se disponível
            if self.streaming_adapter:
                stats = self.streaming_adapter.get_metrics()
                if stats:
                    self.pipeline_metrics['streaming_messages'] = stats.get('total_messages_sent', 0)
                    self.pipeline_metrics['streaming_latency_ms'] = stats.get('average_latency_ms', 0.0)
        
        except Exception as e:
            self.logger.error(f"Erro ao atualizar métricas do pipeline: {e}")
    
    def record_prediction(self, pair: str, model_name: str, predicted_value: float,
                         predicted_class: int, confidence: float, 
                         execution_time: float = 0.0) -> str:
        """Registra uma nova predição no sistema"""
        try:
            import uuid
            
            prediction_id = str(uuid.uuid4())
            
            # Registra no monitor de qualidade
            if self.quality_monitor:
                prediction = PredictionRecord(
                    prediction_id=prediction_id,
                    pair=pair,
                    model_name=model_name,
                    timestamp=datetime.now(),
                    predicted_value=predicted_value,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    execution_time=execution_time
                )
                
                # Enriquece a predição com dados de preço em tempo real se disponíveis
                if pair in self.last_prices:
                    prediction.current_price = self.last_prices[pair].get("price", None)
                    prediction.price_timestamp = self.last_prices[pair].get("timestamp", None)
                    self.logger.debug(f"Predição enriquecida com dados de preço em tempo real: {pair}")
                
                self.quality_monitor.record_prediction(prediction)
            
            # Registra no sistema de memória
            # if self.memory_system:
            #     prediction_text = "ALTA" if predicted_class == 1 else "BAIXA"
            #     self.memory_system.store_prediction(
            #         pair=pair,
            #         prediction=prediction_text,
            #         confidence=confidence,
            #         model_used=model_name
            #     )
            pass  # Sistema de memória será integrado posteriormente
            
            return prediction_id
            
        except Exception as e:
            self.logger.error(f"Erro ao registrar predição: {e}")
            return ""
    
    def update_prediction_outcome(self, prediction_id: str, actual_value: float, 
                                actual_class: int) -> None:
        """Atualiza o resultado real de uma predição"""
        try:
            # Atualiza no monitor de qualidade
            if self.quality_monitor:
                self.quality_monitor.update_prediction_outcome(
                    prediction_id, actual_value, actual_class
                )
            
            # Calcula acurácia e atualiza sistema de memória
            # if self.memory_system:
            #     # Busca predição original para calcular acurácia
            #     # Em implementação real, buscaria do banco de dados
            #     accuracy = 1.0 if actual_class == 1 else 0.0  # Simplificado
            #     actual_text = "ALTA" if actual_class == 1 else "BAIXA"
            #     
            #     # Atualiza no sistema de memória (precisaria do ID interno)
            #     # self.memory_system.update_prediction_outcome(prediction_id, actual_text, accuracy)
            pass  # Sistema de memória será integrado posteriormente
        
        except Exception as e:
            self.logger.error(f"Erro ao atualizar resultado da predição: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Retorna status completo do pipeline"""
        try:
            uptime = 0.0
            if self.start_time:
                uptime = (datetime.now() - self.start_time).total_seconds()
            
            status_dict = {
                "pipeline_status": self.status.value,
                "uptime_seconds": uptime,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "component_health": {
                    name: {
                        "status": health.status.value,
                        "last_update": health.last_update.isoformat(),
                        "error_count": health.error_count,
                        "last_error": health.last_error,
                        "uptime_seconds": health.uptime_seconds,
                        "performance_metrics": health.performance_metrics
                    }
                    for name, health in self.component_health.items()
                },
                "pipeline_metrics": self.pipeline_metrics,
                "configuration": asdict(self.config)
            }
            
            # Adiciona informações de streaming se disponível
            if self.streaming_adapter:
                status_dict["streaming"] = {
                    "active": self.streaming_adapter.running,
                    "subscribed_symbols": list(self.price_callbacks.keys()),
                    "last_update": {symbol: data.get("timestamp", None).isoformat() 
                                   for symbol, data in self.last_prices.items() 
                                   if "timestamp" in data and data.get("timestamp")}
                }
            
            return status_dict
        
        except Exception as e:
            self.logger.error(f"Erro ao obter status do pipeline: {e}")
            return {"error": str(e)}
    
    def stop_pipeline(self) -> None:
        """Para o pipeline completo"""
        try:
            self.logger.info("Parando pipeline de produção...")
            self.status = PipelineStatus.STOPPED
            
            # Para thread principal
            self.stop_event.set()
            if self.main_thread:
                self.main_thread.join(timeout=10)
            
            # Para componentes
            if self.scheduler:
                self.scheduler.stop_scheduler()
                self._update_component_health("scheduler", ComponentStatus.INACTIVE)
            
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
                self._update_component_health("performance_monitor", ComponentStatus.INACTIVE)
            
            if self.quality_monitor:
                self.quality_monitor.stop_monitoring()
                self._update_component_health("quality_monitor", ComponentStatus.INACTIVE)
                
            # Para sistema de streaming
            if self.streaming_adapter:
                self.streaming_adapter.stop()
                self._update_component_health("streaming_adapter", ComponentStatus.INACTIVE)
                
            # Desconecta do MT5
            if self.mt5_connector and self.mt5_connector.connected:
                self.mt5_connector.disconnect()
            
            # Fecha executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Pipeline parado com sucesso")
        
        except Exception as e:
            self.logger.error(f"Erro ao parar pipeline: {e}")
    
    def save_config(self, config_path: str) -> None:
        """Salva configuração atual"""
        try:
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            self.logger.info(f"Configuração salva em {config_path}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar configuração: {e}")
    
    def export_pipeline_report(self) -> str:
        """Exporta relatório completo do pipeline"""
        try:
            report_path = f"reports/pipeline_report_{int(time.time())}.json"
            Path("reports").mkdir(exist_ok=True)
            
            report_data = {
                "generated_at": datetime.now().isoformat(),
                "pipeline_status": self.get_pipeline_status(),
                "component_reports": {}
            }
            
            # Adiciona relatórios dos componentes
            if self.quality_monitor:
                report_data["component_reports"]["quality_monitor"] = \
                    self.quality_monitor.get_quality_summary(hours=24)
            
            # if self.memory_system:
            #     report_data["component_reports"]["memory_system"] = \
            #         self.memory_system.get_memory_stats()
            pass  # Sistema de memória será integrado posteriormente
            
            if self.scheduler:
                report_data["component_reports"]["scheduler"] = \
                    self.scheduler.get_status()
                    
            # Adiciona relatório do sistema de streaming
            if self.streaming_adapter:
                report_data["component_reports"]["streaming"] = {
                    "active_connections": self.streaming_adapter.get_metrics().get("connected_clients", 0),
                    "messages_sent": self.pipeline_metrics.get("streaming_messages", 0),
                    "average_latency_ms": self.pipeline_metrics.get("streaming_latency_ms", 0.0),
                    "subscribed_symbols": list(self.price_callbacks.keys())
                }
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Relatório do pipeline exportado para {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Erro ao exportar relatório: {e}")
            return {"error": str(e)}
    
    def _on_price_update(self, price_data: Dict[str, Any]) -> None:
        """Callback para processar atualizações de preço em tempo real"""
        try:
            # Extrair símbolo dos dados de preço
            symbol = price_data.get("symbol")
            if not symbol:
                self.logger.warning("Dados de preço recebidos sem símbolo")
                return
            
            # Armazena o último preço recebido
            self.last_prices[symbol] = {
                "price": price_data.get("bid", 0.0),
                "ask": price_data.get("ask", 0.0),
                "spread": price_data.get("spread", 0.0),
                "timestamp": datetime.now()
            }
            
            # Log apenas se a mudança de preço for significativa
            if symbol in self.last_prices:
                price_change = abs(price_data.get("bid", 0.0) - self.last_prices[symbol].get("price", 0.0))
                if price_change >= self.config.min_price_change_threshold:
                    self.logger.debug(f"Atualização significativa de preço para {symbol}: {price_data.get('bid', 0.0)}")
            
            # Executa callbacks registrados para este símbolo
            if symbol in self.price_callbacks:
                for callback in self.price_callbacks[symbol]:
                    try:
                        callback(symbol, price_data)
                    except Exception as callback_error:
                        self.logger.error(f"Erro ao executar callback para {symbol}: {callback_error}")
        
        except Exception as e:
            self.logger.error(f"Erro ao processar atualização de preço: {e}")
    
    def _check_streaming_metrics(self) -> None:
        """Verifica métricas do sistema de streaming"""
        try:
            if not self.streaming_adapter:
                return
            
            stats = self.streaming_adapter.get_metrics()
            if not stats:
                return
            
            # Verifica latência
            avg_latency = stats.get('average_latency_ms', 0.0)
            if avg_latency > 1000:  # Latência maior que 1 segundo
                self.logger.warning(f"Alta latência detectada no streaming: {avg_latency:.2f}ms")
                self._update_component_health(
                    "streaming_adapter", 
                    ComponentStatus.ERROR, 
                    f"Alta latência: {avg_latency:.2f}ms"
                )
            
            # Verifica conexões ativas
            active_connections = stats.get('active_connections', 0)
            if active_connections == 0:
                self.logger.warning("Nenhuma conexão ativa no sistema de streaming")
            
            # Atualiza métricas
            self.pipeline_metrics['streaming_messages'] = stats.get('total_messages_sent', 0)
            self.pipeline_metrics['streaming_latency_ms'] = avg_latency
            
            self.logger.debug(f"Métricas de streaming: {active_connections} conexões, {avg_latency:.2f}ms latência")
        
        except Exception as e:
            self.logger.error(f"Erro ao verificar métricas de streaming: {e}")
    
    def log_ml_analysis(self, analysis_type: str, symbol: str, analysis_data: Dict[str, Any]) -> None:
        """Registra análise ML no sistema de logging abrangente"""
        try:
            if self.comprehensive_logger:
                from comprehensive_analysis_logger import AnalysisType, DataSource, AnalysisData
                
                # Mapeia tipo de análise
                analysis_type_enum = AnalysisType.TECHNICAL
                if 'fundamental' in analysis_type.lower():
                    analysis_type_enum = AnalysisType.FUNDAMENTAL
                elif 'risk' in analysis_type.lower():
                    analysis_type_enum = AnalysisType.RISK
                elif 'comprehensive' in analysis_type.lower():
                    analysis_type_enum = AnalysisType.COMPREHENSIVE
                
                # Cria dados de análise
                analysis = AnalysisData(
                    analysis_type=analysis_type_enum,
                    symbol=symbol,
                    data_source=DataSource.ML_MODEL,
                    raw_data=analysis_data,
                    confidence_score=analysis_data.get('confidence', 0.0),
                    processing_time_ms=analysis_data.get('processing_time_ms', 0.0)
                )
                
                self.comprehensive_logger.log_comprehensive_analysis(analysis)
                self.logger.debug(f"Análise ML registrada: {analysis_type} para {symbol}")
        except Exception as e:
            self.logger.error(f"Erro ao registrar análise ML: {e}")
    
    def log_ml_prediction(self, symbol: str, prediction_data: Dict[str, Any]) -> None:
        """Registra predição ML no sistema de logging de decisões"""
        try:
            if self.detailed_logger:
                from detailed_decision_logger import DecisionType, LogLevel, MarketConditions, AccountState
                
                # Cria condições de mercado baseadas na predição
                market_conditions = MarketConditions(
                    volatility=prediction_data.get('volatility', 'medium'),
                    trend=prediction_data.get('trend', 'neutral'),
                    volume=prediction_data.get('volume', 'normal'),
                    spread=prediction_data.get('spread', 0.0),
                    liquidity=prediction_data.get('liquidity', 'normal')
                )
                
                # Estado da conta (simulado para ML)
                account_state = AccountState(
                    balance=0.0,  # Não aplicável para predições ML
                    equity=0.0,
                    margin=0.0,
                    free_margin=0.0,
                    margin_level=0.0,
                    open_positions=0
                )
                
                # Registra como análise de mercado
                self.detailed_logger.log_market_analysis(
                    symbol=symbol,
                    market_conditions=market_conditions,
                    account_state=account_state,
                    analysis_data=prediction_data,
                    confidence_score=prediction_data.get('confidence', 0.0),
                    log_level=LogLevel.INFO
                )
                
                self.logger.debug(f"Predição ML registrada para {symbol}")
        except Exception as e:
            self.logger.error(f"Erro ao registrar predição ML: {e}")
    
    def get_logging_summary(self) -> Dict[str, Any]:
        """Obtém resumo dos sistemas de logging"""
        summary = {
            'comprehensive_logger': None,
            'detailed_logger': None,
            'trading_integrator': None
        }
        
        try:
            if self.comprehensive_logger:
                summary['comprehensive_logger'] = self.comprehensive_logger.get_session_summary()
            
            if self.detailed_logger:
                summary['detailed_logger'] = self.detailed_logger.get_session_summary()
            
            if self.trading_integrator:
                summary['trading_integrator'] = self.trading_integrator.get_session_summary()
                
        except Exception as e:
            self.logger.error(f"Erro ao obter resumo de logging: {e}")
            summary['error'] = str(e)
        
        return summary
    
    def export_logging_reports(self, output_dir: str = None) -> Dict[str, str]:
        """Exporta relatórios dos sistemas de logging"""
        reports = {}
        
        try:
            if output_dir is None:
                output_dir = os.path.join(os.getcwd(), 'logging_reports')
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Exporta relatório do logger abrangente
            if self.comprehensive_logger:
                comp_report = self.comprehensive_logger.export_session_report(output_dir)
                reports['comprehensive_analysis'] = comp_report
            
            # Exporta relatório do logger detalhado
            if self.detailed_logger:
                detail_report = self.detailed_logger.export_session_report(output_dir)
                reports['detailed_decisions'] = detail_report
            
            # Exporta relatório do integrador de trading
            if self.trading_integrator:
                trading_report = self.trading_integrator.export_session_report(output_dir)
                reports['trading_decisions'] = trading_report
            
            self.logger.info(f"Relatórios de logging exportados para {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Erro ao exportar relatórios de logging: {e}")
            reports['error'] = str(e)
        
        return reports

def create_production_pipeline(config_path: str = None) -> MLProductionPipeline:
    """Função de conveniência para criar pipeline de produção"""
    return MLProductionPipeline(config_path=config_path)

if __name__ == "__main__":
    # Exemplo de uso
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\nParando pipeline...")
        pipeline.stop_pipeline()
        sys.exit(0)
    
    # Configura handler para Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Cria e inicia pipeline
    config = PipelineConfig(
        enable_dashboard=True,
        retrain_interval_hours=6,  # Retreina a cada 6 horas
        monitoring_interval_seconds=30,
        currency_pairs=['EURUSD', 'GBPUSD'],
        models_to_monitor=['xgboost', 'random_forest']
    )
    
    pipeline = MLProductionPipeline(config)
    
    if pipeline.start_pipeline():
        print("Pipeline iniciado com sucesso!")
        print(f"Dashboard disponível em: http://localhost:{config.dashboard_port}")
        print("Pressione Ctrl+C para parar")
        
        # Simula algumas predições
        import time
        import numpy as np
        
        for i in range(10):
            prediction_id = pipeline.record_prediction(
                pair="EURUSD",
                model_name="xgboost",
                predicted_value=1.1000 + np.random.normal(0, 0.001),
                predicted_class=np.random.randint(0, 2),
                confidence=np.random.uniform(0.6, 0.95),
                execution_time=np.random.uniform(0.05, 0.2)
            )
            
            print(f"Predição registrada: {prediction_id}")
            time.sleep(5)
        
        # Mantém pipeline rodando
        try:
            while True:
                status = pipeline.get_pipeline_status()
                print(f"Status: {status['pipeline_status']} - Uptime: {status['uptime_seconds']:.0f}s")
                time.sleep(60)
        except KeyboardInterrupt:
            pass
    else:
        print("Falha ao iniciar pipeline")