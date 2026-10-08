#!/usr/bin/env python3
"""
Sistema Avançado de Logging e Debugging para Aplicação Forex ML
Implementa logging configurável, rotação de logs, múltiplos destinos e debugging robusto.
"""

import os
import sys
import json
import yaml
import time
import logging
import logging.config
import logging.handlers
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from contextlib import contextmanager
import traceback
import inspect
import threading
from dataclasses import dataclass, field
from collections import defaultdict, deque
import psutil
import gc

@dataclass
class PerformanceMetrics:
    """Métricas de performance para monitoramento"""
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_delta: Optional[float] = None
    cpu_percent: Optional[float] = None
    function_name: str = ""
    module_name: str = ""
    
    def finalize(self):
        """Finaliza as métricas"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if self.memory_before is not None:
            self.memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self.memory_delta = self.memory_after - self.memory_before

@dataclass
class DebugBreakpoint:
    """Definição de breakpoint condicional"""
    id: str
    file_path: str
    line_number: int
    condition: str = ""
    enabled: bool = True
    hit_count: int = 0
    max_hits: Optional[int] = None
    variables_to_inspect: List[str] = field(default_factory=list)
    action: str = "break"  # break, log, execute
    action_code: str = ""

class SensitiveDataFilter(logging.Filter):
    """Filtro para remover dados sensíveis dos logs"""
    
    SENSITIVE_PATTERNS = [
        'password', 'token', 'key', 'secret', 'api_key',
        'authorization', 'credential', 'private'
    ]
    
    def filter(self, record):
        """Filtra dados sensíveis da mensagem de log"""
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            for pattern in self.SENSITIVE_PATTERNS:
                if pattern.lower() in msg.lower():
                    record.msg = msg.replace(pattern, '[REDACTED]')
        return True

class RateLimitFilter(logging.Filter):
    """Filtro para limitar taxa de logs"""
    
    def __init__(self, rate_limit=100, time_window=60):
        super().__init__()
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.log_counts = defaultdict(deque)
    
    def filter(self, record):
        """Aplica limite de taxa por logger"""
        now = time.time()
        logger_name = record.name
        
        # Remove logs antigos da janela de tempo
        while (self.log_counts[logger_name] and 
               now - self.log_counts[logger_name][0] > self.time_window):
            self.log_counts[logger_name].popleft()
        
        # Verifica se excedeu o limite
        if len(self.log_counts[logger_name]) >= self.rate_limit:
            return False
        
        # Adiciona o log atual
        self.log_counts[logger_name].append(now)
        return True

class PerformanceHandler(logging.Handler):
    """Handler especializado para logs de performance"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_history = deque(maxlen=1000)
    
    def emit(self, record):
        """Processa logs de performance"""
        if hasattr(record, 'performance_metrics'):
            self.metrics_history.append(record.performance_metrics)
        
        # Chama o handler pai para processar normalmente
        super().emit(record)

class AdvancedLoggingSystem:
    """Sistema avançado de logging e debugging"""
    
    def __init__(self, config_file: str = "logging_config.yaml"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.loggers = {}
        self.breakpoints = {}
        self.performance_metrics = {}
        self.debug_enabled = self.config.get('system', {}).get('debugging', {}).get('enabled', False)
        
        # Inicializar sistema
        self._setup_logging()
        self._setup_directories()
        self._load_breakpoints()
        
        # Logger principal
        self.logger = self.get_logger('forex_ml')
        
        # Thread para limpeza de logs antigos
        self._start_cleanup_thread()
    
    def _load_config(self) -> Dict[str, Any]:
        """Carrega configuração do arquivo YAML"""
        if not self.config_file.exists():
            return self._get_default_config()
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuração padrão caso o arquivo não exista"""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'detailed'
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console']
            },
            'system': {
                'log_directory': 'logs',
                'debugging': {'enabled': True}
            }
        }
    
    def _setup_logging(self):
        """Configura o sistema de logging"""
        try:
            # Adicionar filtros customizados
            self._register_custom_filters()
            
            # Configurar logging usando dictConfig
            logging.config.dictConfig(self.config)
            
            # Adicionar handlers customizados
            self._setup_custom_handlers()
            
        except Exception as e:
            print(f"Erro na configuração de logging: {e}")
            logging.basicConfig(level=logging.INFO)
    
    def _register_custom_filters(self):
        """Registra filtros customizados"""
        # Registrar classes de filtros
        logging.Filter.SensitiveDataFilter = SensitiveDataFilter
        logging.Filter.RateLimitFilter = RateLimitFilter
    
    def _setup_custom_handlers(self):
        """Configura handlers customizados"""
        # Adicionar filtros aos handlers existentes
        for handler_name, handler_config in self.config.get('handlers', {}).items():
            handler = logging.getLogger().handlers
            for h in handler:
                if hasattr(h, 'baseFilename') and handler_name in str(h.baseFilename):
                    h.addFilter(SensitiveDataFilter())
                    h.addFilter(RateLimitFilter())
    
    def _setup_directories(self):
        """Cria diretórios necessários"""
        log_dir = Path(self.config.get('system', {}).get('log_directory', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        # Criar subdiretórios
        (log_dir / 'archive').mkdir(exist_ok=True)
        (log_dir / 'debug').mkdir(exist_ok=True)
    
    def _load_breakpoints(self):
        """Carrega breakpoints salvos"""
        breakpoint_file = self.config.get('system', {}).get('debugging', {}).get('breakpoint_file', 'debug_breakpoints.json')
        breakpoint_path = Path(breakpoint_file)
        
        if breakpoint_path.exists():
            try:
                with open(breakpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for bp_data in data:
                        bp = DebugBreakpoint(**bp_data)
                        self.breakpoints[bp.id] = bp
            except Exception as e:
                self.logger.warning(f"Erro ao carregar breakpoints: {e}")
    
    def _save_breakpoints(self):
        """Salva breakpoints no arquivo"""
        breakpoint_file = self.config.get('system', {}).get('debugging', {}).get('breakpoint_file', 'debug_breakpoints.json')
        
        try:
            data = []
            for bp in self.breakpoints.values():
                data.append({
                    'id': bp.id,
                    'file_path': bp.file_path,
                    'line_number': bp.line_number,
                    'condition': bp.condition,
                    'enabled': bp.enabled,
                    'hit_count': bp.hit_count,
                    'max_hits': bp.max_hits,
                    'variables_to_inspect': bp.variables_to_inspect,
                    'action': bp.action,
                    'action_code': bp.action_code
                })
            
            with open(breakpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar breakpoints: {e}")
    
    def _start_cleanup_thread(self):
        """Inicia thread para limpeza automática de logs"""
        def cleanup_old_logs():
            while True:
                try:
                    self._cleanup_old_logs()
                    time.sleep(3600)  # Executar a cada hora
                except Exception as e:
                    print(f"Erro na limpeza de logs: {e}")
                    time.sleep(3600)
        
        cleanup_thread = threading.Thread(target=cleanup_old_logs, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_old_logs(self):
        """Remove logs antigos baseado na configuração"""
        log_dir = Path(self.config.get('system', {}).get('log_directory', 'logs'))
        retention_days = self.config.get('system', {}).get('retention_days', 30)
        
        if not log_dir.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        for log_file in log_dir.glob('*.log*'):
            try:
                if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                    # Mover para arquivo ao invés de deletar
                    archive_dir = log_dir / 'archive'
                    archive_dir.mkdir(exist_ok=True)
                    log_file.rename(archive_dir / log_file.name)
            except Exception as e:
                print(f"Erro ao arquivar {log_file}: {e}")
    
    def get_logger(self, name: str) -> logging.Logger:
        """Obtém logger configurado"""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]
    
    def performance_monitor(self, func: Optional[Callable] = None, *, 
                          threshold: float = 1.0, 
                          memory_tracking: bool = True,
                          log_args: bool = False):
        """Decorator para monitoramento de performance"""
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                # Inicializar métricas
                metrics = PerformanceMetrics(
                    start_time=time.time(),
                    function_name=f.__name__,
                    module_name=f.__module__
                )
                
                if memory_tracking:
                    metrics.memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                perf_logger = self.get_logger('forex_ml.performance')
                
                try:
                    # Executar função
                    result = f(*args, **kwargs)
                    
                    # Finalizar métricas
                    metrics.finalize()
                    
                    # Log se exceder threshold
                    if metrics.duration >= threshold:
                        log_msg = f"Function {f.__name__} took {metrics.duration:.4f}s"
                        
                        if memory_tracking and metrics.memory_delta:
                            log_msg += f", Memory delta: {metrics.memory_delta:.2f}MB"
                        
                        if log_args:
                            log_msg += f", Args: {args[:3]}..., Kwargs: {list(kwargs.keys())}"
                        
                        # Criar record customizado com métricas
                        record = perf_logger.makeRecord(
                            perf_logger.name, logging.INFO, f.__code__.co_filename,
                            f.__code__.co_firstlineno, log_msg, (), None, f.__name__
                        )
                        record.performance_metrics = metrics
                        record.duration = metrics.duration
                        
                        perf_logger.handle(record)
                    
                    return result
                    
                except Exception as e:
                    metrics.finalize()
                    perf_logger.error(f"Function {f.__name__} failed after {metrics.duration:.4f}s: {e}")
                    raise
            
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def add_breakpoint(self, file_path: str, line_number: int, 
                      condition: str = "", variables: List[str] = None,
                      action: str = "break", max_hits: Optional[int] = None) -> str:
        """Adiciona breakpoint condicional"""
        bp_id = f"{Path(file_path).name}:{line_number}"
        
        breakpoint = DebugBreakpoint(
            id=bp_id,
            file_path=file_path,
            line_number=line_number,
            condition=condition,
            variables_to_inspect=variables or [],
            action=action,
            max_hits=max_hits
        )
        
        self.breakpoints[bp_id] = breakpoint
        self._save_breakpoints()
        
        debug_logger = self.get_logger('forex_ml.debug')
        debug_logger.info(f"Breakpoint adicionado: {bp_id} com condição '{condition}'")
        
        return bp_id
    
    def remove_breakpoint(self, bp_id: str):
        """Remove breakpoint"""
        if bp_id in self.breakpoints:
            del self.breakpoints[bp_id]
            self._save_breakpoints()
            
            debug_logger = self.get_logger('forex_ml.debug')
            debug_logger.info(f"Breakpoint removido: {bp_id}")
    
    def check_breakpoint(self, file_path: str, line_number: int, local_vars: Dict[str, Any]):
        """Verifica se deve parar em um breakpoint"""
        if not self.debug_enabled:
            return False
        
        bp_id = f"{Path(file_path).name}:{line_number}"
        
        if bp_id not in self.breakpoints:
            return False
        
        breakpoint = self.breakpoints[bp_id]
        
        if not breakpoint.enabled:
            return False
        
        # Verificar limite de hits
        if breakpoint.max_hits and breakpoint.hit_count >= breakpoint.max_hits:
            return False
        
        # Avaliar condição
        if breakpoint.condition:
            try:
                if not eval(breakpoint.condition, {}, local_vars):
                    return False
            except Exception as e:
                debug_logger = self.get_logger('forex_ml.debug')
                debug_logger.warning(f"Erro na condição do breakpoint {bp_id}: {e}")
                return False
        
        # Incrementar contador
        breakpoint.hit_count += 1
        
        # Executar ação
        debug_logger = self.get_logger('forex_ml.debug')
        
        if breakpoint.action == "log":
            self._log_breakpoint_hit(breakpoint, local_vars)
            return False
        elif breakpoint.action == "execute":
            self._execute_breakpoint_code(breakpoint, local_vars)
            return False
        else:  # break
            self._log_breakpoint_hit(breakpoint, local_vars)
            return True
    
    def _log_breakpoint_hit(self, breakpoint: DebugBreakpoint, local_vars: Dict[str, Any]):
        """Log informações do breakpoint"""
        debug_logger = self.get_logger('forex_ml.debug')
        
        msg = f"Breakpoint hit: {breakpoint.id} (hit #{breakpoint.hit_count})"
        
        if breakpoint.variables_to_inspect:
            var_info = {}
            for var_name in breakpoint.variables_to_inspect:
                if var_name in local_vars:
                    var_info[var_name] = str(local_vars[var_name])[:200]  # Limitar tamanho
            
            if var_info:
                msg += f"\nVariables: {json.dumps(var_info, indent=2)}"
        
        debug_logger.info(msg)
    
    def _execute_breakpoint_code(self, breakpoint: DebugBreakpoint, local_vars: Dict[str, Any]):
        """Executa código customizado no breakpoint"""
        if not breakpoint.action_code:
            return
        
        debug_logger = self.get_logger('forex_ml.debug')
        
        try:
            exec(breakpoint.action_code, {}, local_vars)
            debug_logger.info(f"Código executado no breakpoint {breakpoint.id}")
        except Exception as e:
            debug_logger.error(f"Erro ao executar código no breakpoint {breakpoint.id}: {e}")
    
    @contextmanager
    def debug_context(self, context_name: str):
        """Context manager para debugging com informações de contexto"""
        debug_logger = self.get_logger('forex_ml.debug')
        start_time = time.time()
        
        debug_logger.debug(f"Entrando no contexto: {context_name}")
        
        try:
            yield
        except Exception as e:
            duration = time.time() - start_time
            debug_logger.error(f"Erro no contexto {context_name} após {duration:.4f}s: {e}")
            debug_logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise
        else:
            duration = time.time() - start_time
            debug_logger.debug(f"Saindo do contexto: {context_name} (duração: {duration:.4f}s)")
    
    def inspect_variables(self, frame_depth: int = 1) -> Dict[str, Any]:
        """Inspeciona variáveis no frame atual"""
        frame = inspect.currentframe()
        
        # Navegar para o frame desejado
        for _ in range(frame_depth):
            if frame.f_back:
                frame = frame.f_back
        
        local_vars = frame.f_locals.copy()
        
        # Filtrar variáveis internas
        filtered_vars = {}
        for name, value in local_vars.items():
            if not name.startswith('_'):
                try:
                    # Tentar serializar para verificar se é "inspecionável"
                    str_value = str(value)[:200]
                    filtered_vars[name] = {
                        'type': type(value).__name__,
                        'value': str_value,
                        'size': len(str_value)
                    }
                except:
                    filtered_vars[name] = {
                        'type': type(value).__name__,
                        'value': '<não serializável>',
                        'size': 0
                    }
        
        return filtered_vars
    
    def log_exception(self, exc_info=None, context: str = ""):
        """Log detalhado de exceções"""
        if exc_info is None:
            exc_info = sys.exc_info()
        
        error_logger = self.get_logger('forex_ml')
        
        if context:
            error_logger.error(f"Exceção no contexto '{context}':")
        
        # Log da exceção com stack trace
        error_logger.error("Detalhes da exceção:", exc_info=True)
        
        # Informações adicionais do sistema
        try:
            process = psutil.Process()
            system_info = {
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads(),
                'open_files': len(process.open_files())
            }
            
            error_logger.error(f"Estado do sistema: {json.dumps(system_info, indent=2)}")
        except:
            pass
        
        # Variáveis locais se debugging estiver habilitado
        if self.debug_enabled:
            try:
                variables = self.inspect_variables(frame_depth=2)
                if variables:
                    error_logger.debug(f"Variáveis locais: {json.dumps(variables, indent=2)}")
            except:
                pass
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Retorna resumo de performance"""
        summary = {
            'total_breakpoints': len(self.breakpoints),
            'active_breakpoints': sum(1 for bp in self.breakpoints.values() if bp.enabled),
            'total_hits': sum(bp.hit_count for bp in self.breakpoints.values()),
            'system_info': {
                'log_directory': str(Path(self.config.get('system', {}).get('log_directory', 'logs')).absolute()),
                'debug_enabled': self.debug_enabled,
                'active_loggers': len(self.loggers)
            }
        }
        
        return summary
    
    def shutdown(self):
        """Finaliza o sistema de logging"""
        self._save_breakpoints()
        
        # Fechar todos os handlers
        for logger in self.loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        
        self.logger.info("Sistema de logging finalizado")

# Instância global do sistema de logging
_logging_system = None

def get_logging_system() -> AdvancedLoggingSystem:
    """Obtém instância global do sistema de logging"""
    global _logging_system
    if _logging_system is None:
        _logging_system = AdvancedLoggingSystem()
    return _logging_system

def setup_logging(config_file: str = "logging_config.yaml") -> AdvancedLoggingSystem:
    """Configura sistema de logging global"""
    global _logging_system
    _logging_system = AdvancedLoggingSystem(config_file)
    return _logging_system

# Decorators de conveniência
def monitor_performance(threshold: float = 1.0, memory_tracking: bool = True):
    """Decorator de conveniência para monitoramento de performance"""
    return get_logging_system().performance_monitor(threshold=threshold, memory_tracking=memory_tracking)

def debug_breakpoint(condition: str = "", variables: List[str] = None):
    """Decorator para adicionar breakpoint em função"""
    def decorator(func):
        frame = inspect.currentframe().f_back
        file_path = frame.f_code.co_filename
        line_number = frame.f_lineno
        
        get_logging_system().add_breakpoint(
            file_path, line_number, condition, variables
        )
        
        return func
    return decorator

if __name__ == "__main__":
    # Exemplo de uso
    logging_system = setup_logging()
    
    logger = logging_system.get_logger('forex_ml.test')
    
    # Teste de logging básico
    logger.info("Sistema de logging inicializado")
    logger.debug("Mensagem de debug")
    logger.warning("Mensagem de aviso")
    logger.error("Mensagem de erro")
    
    # Teste de performance monitoring
    @monitor_performance(threshold=0.1)
    def test_function():
        time.sleep(0.2)
        return "resultado"
    
    result = test_function()
    
    # Teste de breakpoint
    logging_system.add_breakpoint(__file__, 50, "x > 10", ["x", "y"])
    
    # Resumo
    summary = logging_system.get_performance_summary()
    logger.info(f"Resumo do sistema: {json.dumps(summary, indent=2)}")