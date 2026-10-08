import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from pathlib import Path

class SchedulerStatus(Enum):
    """Status do agendador"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class TriggerType(Enum):
    """Tipos de gatilhos para retreinamento"""
    TIME_INTERVAL = "time_interval"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    MANUAL = "manual"
    MARKET_VOLATILITY = "market_volatility"

@dataclass
class RetrainingTask:
    """Tarefa de retreinamento"""
    task_id: str
    pair: str
    trigger_type: TriggerType
    scheduled_time: datetime
    priority: int = 1  # 1=alta, 2=média, 3=baixa
    status: str = "pending"
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class PerformanceThresholds:
    """Limiares de performance para gatilhos"""
    min_accuracy: float = 0.55  # Acurácia mínima
    max_loss_streak: int = 5    # Máximo de perdas consecutivas
    min_sharpe_ratio: float = 0.5  # Sharpe ratio mínimo
    max_drawdown: float = 0.15  # Drawdown máximo (15%)
    min_profit_factor: float = 1.2  # Profit factor mínimo
    volatility_threshold: float = 0.02  # Threshold de volatilidade

class MLModelScheduler:
    """Agendador automático para retreinamento de modelos ML"""
    
    def __init__(self, config_path: str = None):
        self.logger = self._setup_logger()
        self.status = SchedulerStatus.STOPPED
        self.tasks: List[RetrainingTask] = []
        self.performance_thresholds = PerformanceThresholds()
        self.config_path = config_path or "scheduler_config.json"
        self.data_path = Path("scheduler_data")
        self.data_path.mkdir(exist_ok=True)
        
        # Configurações padrão
        self.config = {
            "retrain_interval_hours": 24,
            "max_concurrent_tasks": 2,
            "performance_check_interval_minutes": 30,
            "enabled_pairs": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
            "auto_start": True,
            "backup_models": True,
            "notification_enabled": True
        }
        
        self.load_config()
        self.running_tasks: Dict[str, threading.Thread] = {}
        self.performance_monitor = None
        self.scheduler_thread = None
        self._stop_event = threading.Event()
        
    def _setup_logger(self) -> logging.Logger:
        """Configura o sistema de logging"""
        logger = logging.getLogger("MLScheduler")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("logs/ml_scheduler.log")
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
    
    def load_config(self) -> None:
        """Carrega configurações do arquivo"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                self.logger.info(f"Configuração carregada de {self.config_path}")
            else:
                self.save_config()
                self.logger.info("Configuração padrão criada")
        except Exception as e:
            self.logger.error(f"Erro ao carregar configuração: {e}")
    
    def save_config(self) -> None:
        """Salva configurações no arquivo"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Configuração salva em {self.config_path}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar configuração: {e}")
    
    def start_scheduler(self) -> None:
        """Inicia o agendador"""
        if self.status == SchedulerStatus.RUNNING:
            self.logger.warning("Agendador já está em execução")
            return
        
        self.status = SchedulerStatus.RUNNING
        self._stop_event.clear()
        
        # Thread principal do agendador
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        # Thread de monitoramento de performance
        self.performance_monitor = threading.Thread(target=self._performance_monitor_loop, daemon=True)
        self.performance_monitor.start()
        
        self.logger.info("Agendador iniciado com sucesso")
        
        # Agenda retreinamentos iniciais se necessário
        self._schedule_initial_retraining()
    
    def stop_scheduler(self) -> None:
        """Para o agendador"""
        self.status = SchedulerStatus.STOPPED
        self._stop_event.set()
        
        # Aguarda threads terminarem
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        if self.performance_monitor and self.performance_monitor.is_alive():
            self.performance_monitor.join(timeout=5)
        
        self.logger.info("Agendador parado")
    
    def _scheduler_loop(self) -> None:
        """Loop principal do agendador"""
        while not self._stop_event.is_set():
            try:
                self._process_pending_tasks()
                self._cleanup_completed_tasks()
                time.sleep(60)  # Verifica a cada minuto
            except Exception as e:
                self.logger.error(f"Erro no loop do agendador: {e}")
                time.sleep(60)
    
    def _performance_monitor_loop(self) -> None:
        """Loop de monitoramento de performance"""
        check_interval = self.config.get("performance_check_interval_minutes", 30) * 60
        
        while not self._stop_event.is_set():
            try:
                self._check_performance_triggers()
                time.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Erro no monitoramento de performance: {e}")
                time.sleep(check_interval)
    
    def schedule_retraining(self, pair: str, trigger_type: TriggerType, 
                          scheduled_time: datetime = None, priority: int = 1) -> str:
        """Agenda um retreinamento"""
        if scheduled_time is None:
            scheduled_time = datetime.now()
        
        task_id = f"{pair}_{trigger_type.value}_{int(scheduled_time.timestamp())}"
        
        task = RetrainingTask(
            task_id=task_id,
            pair=pair,
            trigger_type=trigger_type,
            scheduled_time=scheduled_time,
            priority=priority
        )
        
        self.tasks.append(task)
        self.logger.info(f"Retreinamento agendado: {task_id} para {pair} em {scheduled_time}")
        
        return task_id
    
    def _schedule_initial_retraining(self) -> None:
        """Agenda retreinamentos iniciais baseados em intervalo"""
        interval_hours = self.config.get("retrain_interval_hours", 24)
        
        for pair in self.config.get("enabled_pairs", []):
            # Verifica se já existe um agendamento recente
            recent_tasks = [
                t for t in self.tasks 
                if t.pair == pair and t.trigger_type == TriggerType.TIME_INTERVAL
                and t.scheduled_time > datetime.now() - timedelta(hours=interval_hours)
            ]
            
            if not recent_tasks:
                next_run = datetime.now() + timedelta(hours=interval_hours)
                self.schedule_retraining(pair, TriggerType.TIME_INTERVAL, next_run)
    
    def _process_pending_tasks(self) -> None:
        """Processa tarefas pendentes"""
        now = datetime.now()
        pending_tasks = [
            t for t in self.tasks 
            if t.status == "pending" and t.scheduled_time <= now
        ]
        
        # Ordena por prioridade e tempo
        pending_tasks.sort(key=lambda x: (x.priority, x.scheduled_time))
        
        max_concurrent = self.config.get("max_concurrent_tasks", 2)
        active_count = len([t for t in self.running_tasks.values() if t.is_alive()])
        
        for task in pending_tasks[:max_concurrent - active_count]:
            self._execute_retraining_task(task)
    
    def _execute_retraining_task(self, task: RetrainingTask) -> None:
        """Executa uma tarefa de retreinamento"""
        task.status = "running"
        
        def run_task():
            try:
                self.logger.info(f"Iniciando retreinamento: {task.task_id}")
                
                # Aqui seria chamado o sistema de retreinamento real
                success, metrics = self._perform_model_retraining(task.pair)
                
                if success:
                    task.status = "completed"
                    task.performance_metrics = metrics
                    task.completed_at = datetime.now()
                    self.logger.info(f"Retreinamento concluído: {task.task_id}")
                    
                    # Agenda próximo retreinamento se for baseado em intervalo
                    if task.trigger_type == TriggerType.TIME_INTERVAL:
                        interval_hours = self.config.get("retrain_interval_hours", 24)
                        next_run = datetime.now() + timedelta(hours=interval_hours)
                        self.schedule_retraining(task.pair, TriggerType.TIME_INTERVAL, next_run)
                else:
                    task.status = "failed"
                    task.error_message = "Falha no retreinamento"
                    self.logger.error(f"Falha no retreinamento: {task.task_id}")
                    
            except Exception as e:
                task.status = "failed"
                task.error_message = str(e)
                self.logger.error(f"Erro no retreinamento {task.task_id}: {e}")
            finally:
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
        
        thread = threading.Thread(target=run_task, daemon=True)
        self.running_tasks[task.task_id] = thread
        thread.start()
    
    def _perform_model_retraining(self, pair: str) -> tuple[bool, Dict]:
        """Executa o retreinamento do modelo (placeholder)"""
        # Esta função seria integrada com o sistema de treinamento existente
        # Por enquanto, simula o processo
        
        self.logger.info(f"Retreinando modelo para {pair}...")
        time.sleep(5)  # Simula tempo de treinamento
        
        # Métricas simuladas
        metrics = {
            "accuracy": 0.62,
            "precision": 0.58,
            "recall": 0.65,
            "f1_score": 0.61,
            "training_time": 300,
            "model_size_mb": 15.2
        }
        
        return True, metrics
    
    def _check_performance_triggers(self) -> None:
        """Verifica gatilhos de performance"""
        for pair in self.config.get("enabled_pairs", []):
            try:
                current_metrics = self._get_current_performance_metrics(pair)
                
                if self._should_trigger_retraining(current_metrics):
                    # Verifica se já não há um retreinamento recente agendado
                    recent_tasks = [
                        t for t in self.tasks 
                        if t.pair == pair and t.trigger_type == TriggerType.PERFORMANCE_DEGRADATION
                        and t.scheduled_time > datetime.now() - timedelta(hours=6)
                    ]
                    
                    if not recent_tasks:
                        self.schedule_retraining(
                            pair, 
                            TriggerType.PERFORMANCE_DEGRADATION, 
                            datetime.now(),
                            priority=1  # Alta prioridade
                        )
                        self.logger.warning(f"Degradação de performance detectada para {pair}")
                        
            except Exception as e:
                self.logger.error(f"Erro ao verificar performance de {pair}: {e}")
    
    def _get_current_performance_metrics(self, pair: str) -> Dict:
        """Obtém métricas atuais de performance (placeholder)"""
        # Esta função seria integrada com o sistema de monitoramento real
        # Por enquanto, retorna métricas simuladas
        
        import random
        return {
            "accuracy": random.uniform(0.45, 0.70),
            "loss_streak": random.randint(0, 8),
            "sharpe_ratio": random.uniform(0.2, 1.5),
            "drawdown": random.uniform(0.05, 0.25),
            "profit_factor": random.uniform(0.8, 2.0),
            "volatility": random.uniform(0.01, 0.05)
        }
    
    def _should_trigger_retraining(self, metrics: Dict) -> bool:
        """Determina se deve disparar retreinamento baseado nas métricas"""
        thresholds = self.performance_thresholds
        
        triggers = [
            metrics.get("accuracy", 1.0) < thresholds.min_accuracy,
            metrics.get("loss_streak", 0) > thresholds.max_loss_streak,
            metrics.get("sharpe_ratio", 1.0) < thresholds.min_sharpe_ratio,
            metrics.get("drawdown", 0.0) > thresholds.max_drawdown,
            metrics.get("profit_factor", 1.0) < thresholds.min_profit_factor,
            metrics.get("volatility", 0.0) > thresholds.volatility_threshold
        ]
        
        return any(triggers)
    
    def _cleanup_completed_tasks(self) -> None:
        """Remove tarefas antigas completadas"""
        cutoff_date = datetime.now() - timedelta(days=7)
        
        self.tasks = [
            t for t in self.tasks 
            if not (t.status in ["completed", "failed"] and 
                   (t.completed_at or t.created_at) < cutoff_date)
        ]
    
    def get_status(self) -> Dict:
        """Retorna status atual do agendador"""
        return {
            "status": self.status.value,
            "total_tasks": len(self.tasks),
            "pending_tasks": len([t for t in self.tasks if t.status == "pending"]),
            "running_tasks": len([t for t in self.tasks if t.status == "running"]),
            "completed_tasks": len([t for t in self.tasks if t.status == "completed"]),
            "failed_tasks": len([t for t in self.tasks if t.status == "failed"]),
            "active_threads": len([t for t in self.running_tasks.values() if t.is_alive()]),
            "config": self.config
        }
    
    def get_tasks(self, status: str = None, pair: str = None) -> List[Dict]:
        """Retorna lista de tarefas filtradas"""
        filtered_tasks = self.tasks
        
        if status:
            filtered_tasks = [t for t in filtered_tasks if t.status == status]
        
        if pair:
            filtered_tasks = [t for t in filtered_tasks if t.pair == pair]
        
        return [asdict(task) for task in filtered_tasks]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancela uma tarefa"""
        for task in self.tasks:
            if task.task_id == task_id and task.status == "pending":
                task.status = "cancelled"
                self.logger.info(f"Tarefa cancelada: {task_id}")
                return True
        
        return False
    
    def update_thresholds(self, **kwargs) -> None:
        """Atualiza limiares de performance"""
        for key, value in kwargs.items():
            if hasattr(self.performance_thresholds, key):
                setattr(self.performance_thresholds, key, value)
                self.logger.info(f"Threshold atualizado: {key} = {value}")
    
    def force_retraining(self, pair: str) -> str:
        """Força retreinamento imediato"""
        return self.schedule_retraining(
            pair, 
            TriggerType.MANUAL, 
            datetime.now(),
            priority=1
        )

if __name__ == "__main__":
    # Exemplo de uso
    scheduler = MLModelScheduler()
    
    try:
        scheduler.start_scheduler()
        
        # Força um retreinamento para teste
        task_id = scheduler.force_retraining("EURUSD")
        print(f"Retreinamento forçado agendado: {task_id}")
        
        # Monitora por alguns minutos
        for i in range(10):
            status = scheduler.get_status()
            print(f"Status: {status}")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("Parando agendador...")
    finally:
        scheduler.stop_scheduler()