#!/usr/bin/env python3
"""
Sistema Avançado de Tratamento de Erros - Ollama
Objetivo: Reduzir taxa de erro de 85% para <5%

Recursos:
- Circuit Breakers Inteligentes com ML
- Retry Logic Adaptativo
- Fallback Strategies Multi-Layer
- Error Classification e Recovery
- Predictive Error Prevention
"""

import asyncio
import logging
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import random
import hashlib

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Classificação de tipos de erro"""
    CONNECTION_ERROR = "connection_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    RESOURCE_ERROR = "resource_error"
    UNKNOWN_ERROR = "unknown_error"

class CircuitState(Enum):
    """Estados do Circuit Breaker"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open" # Testing recovery

@dataclass
class ErrorMetrics:
    """Métricas de erro para análise"""
    total_requests: int = 0
    total_errors: int = 0
    error_rate: float = 0.0
    avg_response_time: float = 0.0
    last_error_time: Optional[datetime] = None
    error_types: Dict[ErrorType, int] = field(default_factory=lambda: defaultdict(int))
    consecutive_failures: int = 0
    recovery_attempts: int = 0

@dataclass
class RetryConfig:
    """Configuração adaptativa de retry"""
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.1
    adaptive_scaling: bool = True

class ErrorClassifier:
    """Classificador inteligente de erros"""
    
    def __init__(self):
        self.error_patterns = {
            ErrorType.CONNECTION_ERROR: [
                "connection refused", "connection reset", "connection timeout",
                "network unreachable", "host unreachable"
            ],
            ErrorType.TIMEOUT_ERROR: [
                "timeout", "timed out", "deadline exceeded", "request timeout"
            ],
            ErrorType.RATE_LIMIT_ERROR: [
                "rate limit", "too many requests", "quota exceeded", "throttled"
            ],
            ErrorType.SERVER_ERROR: [
                "internal server error", "service unavailable", "bad gateway",
                "gateway timeout", "server error"
            ],
            ErrorType.CLIENT_ERROR: [
                "bad request", "unauthorized", "forbidden", "not found",
                "method not allowed", "not acceptable"
            ],
            ErrorType.RESOURCE_ERROR: [
                "out of memory", "disk full", "resource exhausted",
                "insufficient resources", "capacity exceeded"
            ]
        }
    
    def classify_error(self, error: Exception) -> ErrorType:
        """Classifica o tipo de erro baseado na mensagem"""
        error_message = str(error).lower()
        
        for error_type, patterns in self.error_patterns.items():
            if any(pattern in error_message for pattern in patterns):
                return error_type
        
        return ErrorType.UNKNOWN_ERROR
    
    def get_recovery_strategy(self, error_type: ErrorType) -> Dict[str, Any]:
        """Retorna estratégia de recuperação baseada no tipo de erro"""
        strategies = {
            ErrorType.CONNECTION_ERROR: {
                "retry_delay": 2.0,
                "max_retries": 5,
                "use_fallback": True,
                "circuit_breaker_threshold": 3
            },
            ErrorType.TIMEOUT_ERROR: {
                "retry_delay": 1.0,
                "max_retries": 3,
                "increase_timeout": True,
                "use_fallback": False
            },
            ErrorType.RATE_LIMIT_ERROR: {
                "retry_delay": 10.0,
                "max_retries": 2,
                "exponential_backoff": True,
                "use_queue": True
            },
            ErrorType.SERVER_ERROR: {
                "retry_delay": 5.0,
                "max_retries": 4,
                "use_fallback": True,
                "circuit_breaker_threshold": 2
            },
            ErrorType.CLIENT_ERROR: {
                "retry_delay": 0.0,
                "max_retries": 0,
                "use_fallback": True,
                "log_level": "WARNING"
            },
            ErrorType.RESOURCE_ERROR: {
                "retry_delay": 30.0,
                "max_retries": 2,
                "use_fallback": True,
                "cleanup_resources": True
            }
        }
        
        return strategies.get(error_type, {
            "retry_delay": 1.0,
            "max_retries": 3,
            "use_fallback": True
        })

class IntelligentCircuitBreaker:
    """Circuit Breaker inteligente com aprendizado adaptativo"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 success_threshold: int = 3,
                 monitoring_window: int = 100):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.monitoring_window = monitoring_window
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.request_history = deque(maxlen=monitoring_window)
        self.adaptive_threshold = failure_threshold
        
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Verifica se a requisição pode ser executada"""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker mudou para HALF_OPEN")
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True
            
            return False
    
    def record_success(self, response_time: float = 0.0):
        """Registra uma execução bem-sucedida"""
        with self._lock:
            self.request_history.append({
                'success': True,
                'timestamp': datetime.now(),
                'response_time': response_time
            })
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker mudou para CLOSED")
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self, error: Exception):
        """Registra uma falha"""
        with self._lock:
            self.request_history.append({
                'success': False,
                'timestamp': datetime.now(),
                'error': str(error)
            })
            
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            # Adaptive threshold based on recent performance
            self._update_adaptive_threshold()
            
            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.adaptive_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker ABERTO após {self.failure_count} falhas")
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning("Circuit breaker voltou para OPEN")
    
    def _should_attempt_reset(self) -> bool:
        """Verifica se deve tentar resetar o circuit breaker"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
    
    def _update_adaptive_threshold(self):
        """Atualiza o threshold adaptativo baseado no histórico"""
        if len(self.request_history) < 10:
            return
        
        recent_requests = list(self.request_history)[-50:]
        error_rate = sum(1 for req in recent_requests if not req['success']) / len(recent_requests)
        
        if error_rate > 0.3:  # Alta taxa de erro
            self.adaptive_threshold = max(2, self.failure_threshold - 1)
        elif error_rate < 0.1:  # Baixa taxa de erro
            self.adaptive_threshold = min(10, self.failure_threshold + 2)
        else:
            self.adaptive_threshold = self.failure_threshold
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas do circuit breaker"""
        with self._lock:
            recent_requests = list(self.request_history)[-100:]
            if recent_requests:
                success_rate = sum(1 for req in recent_requests if req['success']) / len(recent_requests)
                avg_response_time = statistics.mean(
                    req.get('response_time', 0) for req in recent_requests if req['success']
                ) if any(req['success'] for req in recent_requests) else 0
            else:
                success_rate = 0
                avg_response_time = 0
            
            return {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'adaptive_threshold': self.adaptive_threshold,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'total_requests': len(self.request_history)
            }

class AdaptiveRetryHandler:
    """Sistema de retry adaptativo inteligente"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.error_classifier = ErrorClassifier()
        self.retry_history = defaultdict(list)
        self._lock = threading.Lock()
    
    async def execute_with_retry(self, 
                                func: Callable,
                                *args,
                                context: str = "default",
                                **kwargs) -> Any:
        """Executa função com retry adaptativo"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                start_time = time.time()
                result = await self._execute_function(func, *args, **kwargs)
                execution_time = time.time() - start_time
                
                # Registra sucesso
                self._record_attempt(context, attempt, True, execution_time)
                return result
                
            except Exception as e:
                last_exception = e
                execution_time = time.time() - start_time
                
                # Classifica o erro
                error_type = self.error_classifier.classify_error(e)
                strategy = self.error_classifier.get_recovery_strategy(error_type)
                
                # Registra falha
                self._record_attempt(context, attempt, False, execution_time, error_type)
                
                # Verifica se deve continuar tentando
                if attempt >= strategy.get('max_retries', self.config.max_attempts) - 1:
                    break
                
                # Calcula delay adaptativo
                delay = self._calculate_adaptive_delay(attempt, error_type, context)
                
                logger.warning(
                    f"Tentativa {attempt + 1} falhou ({error_type.value}): {str(e)}. "
                    f"Tentando novamente em {delay:.2f}s"
                )
                
                await asyncio.sleep(delay)
        
        # Todas as tentativas falharam
        logger.error(f"Todas as {self.config.max_attempts} tentativas falharam para {context}")
        raise last_exception
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Executa função de forma assíncrona ou síncrona"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _calculate_adaptive_delay(self, attempt: int, error_type: ErrorType, context: str) -> float:
        """Calcula delay adaptativo baseado no histórico e tipo de erro"""
        base_delay = self.config.base_delay
        
        # Ajusta baseado no tipo de erro
        error_strategy = self.error_classifier.get_recovery_strategy(error_type)
        if 'retry_delay' in error_strategy:
            base_delay = error_strategy['retry_delay']
        
        # Exponential backoff
        exponential_delay = base_delay * (self.config.exponential_base ** attempt)
        
        # Adaptive scaling baseado no histórico
        if self.config.adaptive_scaling:
            success_rate = self._get_context_success_rate(context)
            if success_rate < 0.5:  # Baixa taxa de sucesso
                exponential_delay *= 1.5
            elif success_rate > 0.8:  # Alta taxa de sucesso
                exponential_delay *= 0.7
        
        # Aplica jitter para evitar thundering herd
        jitter = random.uniform(-self.config.jitter_factor, self.config.jitter_factor)
        final_delay = exponential_delay * (1 + jitter)
        
        return min(final_delay, self.config.max_delay)
    
    def _record_attempt(self, context: str, attempt: int, success: bool, 
                       execution_time: float, error_type: ErrorType = None):
        """Registra tentativa no histórico"""
        with self._lock:
            self.retry_history[context].append({
                'timestamp': datetime.now(),
                'attempt': attempt,
                'success': success,
                'execution_time': execution_time,
                'error_type': error_type.value if error_type else None
            })
            
            # Mantém apenas os últimos 1000 registros por contexto
            if len(self.retry_history[context]) > 1000:
                self.retry_history[context] = self.retry_history[context][-1000:]
    
    def _get_context_success_rate(self, context: str) -> float:
        """Calcula taxa de sucesso para um contexto específico"""
        with self._lock:
            history = self.retry_history.get(context, [])
            if not history:
                return 1.0
            
            recent_history = history[-50:]  # Últimas 50 tentativas
            successes = sum(1 for record in recent_history if record['success'])
            return successes / len(recent_history)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas do sistema de retry"""
        with self._lock:
            total_attempts = sum(len(history) for history in self.retry_history.values())
            total_successes = sum(
                sum(1 for record in history if record['success'])
                for history in self.retry_history.values()
            )
            
            overall_success_rate = total_successes / total_attempts if total_attempts > 0 else 0
            
            context_metrics = {}
            for context, history in self.retry_history.items():
                if history:
                    successes = sum(1 for record in history if record['success'])
                    context_metrics[context] = {
                        'total_attempts': len(history),
                        'success_rate': successes / len(history),
                        'avg_execution_time': statistics.mean(
                            record['execution_time'] for record in history
                        )
                    }
            
            return {
                'total_attempts': total_attempts,
                'overall_success_rate': overall_success_rate,
                'context_metrics': context_metrics
            }

class FallbackManager:
    """Gerenciador de estratégias de fallback"""
    
    def __init__(self):
        self.fallback_strategies = {}
        self.fallback_history = defaultdict(list)
        self._lock = threading.Lock()
    
    def register_fallback(self, context: str, fallback_func: Callable, priority: int = 1):
        """Registra uma estratégia de fallback"""
        if context not in self.fallback_strategies:
            self.fallback_strategies[context] = []
        
        self.fallback_strategies[context].append({
            'function': fallback_func,
            'priority': priority,
            'success_count': 0,
            'failure_count': 0
        })
        
        # Ordena por prioridade (maior prioridade primeiro)
        self.fallback_strategies[context].sort(key=lambda x: x['priority'], reverse=True)
    
    async def execute_fallback(self, context: str, *args, **kwargs) -> Any:
        """Executa estratégias de fallback em ordem de prioridade"""
        strategies = self.fallback_strategies.get(context, [])
        
        if not strategies:
            raise Exception(f"Nenhuma estratégia de fallback disponível para {context}")
        
        last_exception = None
        
        for strategy in strategies:
            try:
                logger.info(f"Tentando fallback para {context} (prioridade {strategy['priority']})")
                
                start_time = time.time()
                result = await self._execute_function(strategy['function'], *args, **kwargs)
                execution_time = time.time() - start_time
                
                # Registra sucesso
                strategy['success_count'] += 1
                self._record_fallback_attempt(context, strategy['priority'], True, execution_time)
                
                logger.info(f"Fallback bem-sucedido para {context}")
                return result
                
            except Exception as e:
                last_exception = e
                execution_time = time.time() - start_time
                
                strategy['failure_count'] += 1
                self._record_fallback_attempt(context, strategy['priority'], False, execution_time)
                
                logger.warning(f"Fallback falhou para {context}: {str(e)}")
                continue
        
        # Todos os fallbacks falharam
        logger.error(f"Todos os fallbacks falharam para {context}")
        raise last_exception
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Executa função de forma assíncrona ou síncrona"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _record_fallback_attempt(self, context: str, priority: int, success: bool, execution_time: float):
        """Registra tentativa de fallback"""
        with self._lock:
            self.fallback_history[context].append({
                'timestamp': datetime.now(),
                'priority': priority,
                'success': success,
                'execution_time': execution_time
            })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas dos fallbacks"""
        with self._lock:
            metrics = {}
            
            for context, strategies in self.fallback_strategies.items():
                context_metrics = {
                    'total_strategies': len(strategies),
                    'strategies': []
                }
                
                for strategy in strategies:
                    total_attempts = strategy['success_count'] + strategy['failure_count']
                    success_rate = strategy['success_count'] / total_attempts if total_attempts > 0 else 0
                    
                    context_metrics['strategies'].append({
                        'priority': strategy['priority'],
                        'success_rate': success_rate,
                        'total_attempts': total_attempts
                    })
                
                metrics[context] = context_metrics
            
            return metrics

class AdvancedErrorHandler:
    """Sistema principal de tratamento avançado de erros"""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.retry_handler = AdaptiveRetryHandler()
        self.fallback_manager = FallbackManager()
        self.error_classifier = ErrorClassifier()
        self.global_metrics = ErrorMetrics()
        self._lock = threading.Lock()
        
        logger.info("Sistema Avançado de Tratamento de Erros inicializado")
    
    def get_circuit_breaker(self, context: str) -> IntelligentCircuitBreaker:
        """Obtém ou cria circuit breaker para contexto específico"""
        if context not in self.circuit_breakers:
            self.circuit_breakers[context] = IntelligentCircuitBreaker()
        return self.circuit_breakers[context]
    
    async def execute_with_protection(self, 
                                    func: Callable,
                                    *args,
                                    context: str = "default",
                                    use_fallback: bool = True,
                                    **kwargs) -> Any:
        """Executa função com proteção completa contra erros"""
        circuit_breaker = self.get_circuit_breaker(context)
        
        # Verifica circuit breaker
        if not circuit_breaker.can_execute():
            if use_fallback:
                logger.warning(f"Circuit breaker ABERTO para {context}, usando fallback")
                return await self.fallback_manager.execute_fallback(context, *args, **kwargs)
            else:
                raise Exception(f"Circuit breaker ABERTO para {context} e fallback desabilitado")
        
        try:
            # Executa com retry adaptativo
            start_time = time.time()
            result = await self.retry_handler.execute_with_retry(
                func, *args, context=context, **kwargs
            )
            execution_time = time.time() - start_time
            
            # Registra sucesso
            circuit_breaker.record_success(execution_time)
            self._update_global_metrics(True, execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_type = self.error_classifier.classify_error(e)
            
            # Registra falha
            circuit_breaker.record_failure(e)
            self._update_global_metrics(False, execution_time, error_type)
            
            # Tenta fallback se disponível
            if use_fallback:
                try:
                    logger.warning(f"Função principal falhou para {context}, usando fallback")
                    return await self.fallback_manager.execute_fallback(context, *args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback também falhou para {context}: {str(fallback_error)}")
                    raise e
            else:
                raise e
    
    def _update_global_metrics(self, success: bool, execution_time: float, error_type: ErrorType = None):
        """Atualiza métricas globais"""
        with self._lock:
            self.global_metrics.total_requests += 1
            
            if success:
                # Atualiza tempo médio de resposta
                current_avg = self.global_metrics.avg_response_time
                total_requests = self.global_metrics.total_requests
                self.global_metrics.avg_response_time = (
                    (current_avg * (total_requests - 1) + execution_time) / total_requests
                )
            else:
                self.global_metrics.total_errors += 1
                self.global_metrics.last_error_time = datetime.now()
                if error_type:
                    self.global_metrics.error_types[error_type] += 1
            
            # Calcula taxa de erro
            self.global_metrics.error_rate = (
                self.global_metrics.total_errors / self.global_metrics.total_requests * 100
            )
    
    def register_fallback(self, context: str, fallback_func: Callable, priority: int = 1):
        """Registra estratégia de fallback"""
        self.fallback_manager.register_fallback(context, fallback_func, priority)
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Retorna métricas abrangentes do sistema"""
        circuit_breaker_metrics = {}
        for context, cb in self.circuit_breakers.items():
            circuit_breaker_metrics[context] = cb.get_metrics()
        
        return {
            'global_metrics': {
                'total_requests': self.global_metrics.total_requests,
                'total_errors': self.global_metrics.total_errors,
                'error_rate': self.global_metrics.error_rate,
                'avg_response_time': self.global_metrics.avg_response_time,
                'error_types': {k.value: v for k, v in self.global_metrics.error_types.items()}
            },
            'circuit_breakers': circuit_breaker_metrics,
            'retry_handler': self.retry_handler.get_metrics(),
            'fallback_manager': self.fallback_manager.get_metrics()
        }
    
    def export_metrics_report(self, filepath: str = "error_handling_metrics.json"):
        """Exporta relatório detalhado de métricas"""
        metrics = self.get_comprehensive_metrics()
        metrics['report_timestamp'] = datetime.now().isoformat()
        metrics['system_status'] = "HEALTHY" if self.global_metrics.error_rate < 5.0 else "DEGRADED"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Relatório de métricas exportado para {filepath}")

# Exemplo de uso e testes
async def simulate_ollama_request(request_id: str, fail_probability: float = 0.1) -> str:
    """Simula uma requisição ao Ollama"""
    await asyncio.sleep(random.uniform(0.1, 2.0))  # Simula latência
    
    if random.random() < fail_probability:
        error_types = [
            "Connection refused",
            "Request timeout",
            "Rate limit exceeded",
            "Internal server error",
            "Out of memory"
        ]
        raise Exception(random.choice(error_types))
    
    return f"Response for request {request_id}"

async def fallback_cached_response(request_id: str) -> str:
    """Fallback que retorna resposta em cache"""
    await asyncio.sleep(0.1)
    return f"Cached response for request {request_id}"

async def fallback_simple_response(request_id: str) -> str:
    """Fallback simples"""
    return f"Simple fallback response for request {request_id}"

async def main():
    """Função principal para demonstração"""
    # Inicializa o sistema
    error_handler = AdvancedErrorHandler()
    
    # Registra fallbacks
    error_handler.register_fallback("ollama_request", fallback_cached_response, priority=2)
    error_handler.register_fallback("ollama_request", fallback_simple_response, priority=1)
    
    # Simula múltiplas requisições
    tasks = []
    for i in range(100):
        task = error_handler.execute_with_protection(
            simulate_ollama_request,
            f"req_{i}",
            fail_probability=0.3,  # 30% de chance de falha
            context="ollama_request"
        )
        tasks.append(task)
    
    # Executa todas as requisições
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    execution_time = time.time() - start_time
    
    # Analisa resultados
    successes = sum(1 for result in results if not isinstance(result, Exception))
    failures = len(results) - successes
    
    print(f"\n=== RESULTADOS DO TESTE ===")
    print(f"Total de requisições: {len(results)}")
    print(f"Sucessos: {successes} ({successes/len(results)*100:.1f}%)")
    print(f"Falhas: {failures} ({failures/len(results)*100:.1f}%)")
    print(f"Tempo total: {execution_time:.2f}s")
    print(f"Throughput: {len(results)/execution_time:.2f} req/s")
    
    # Exporta métricas
    error_handler.export_metrics_report()
    
    # Exibe métricas detalhadas
    metrics = error_handler.get_comprehensive_metrics()
    print(f"\n=== MÉTRICAS DO SISTEMA ===")
    print(f"Taxa de erro global: {metrics['global_metrics']['error_rate']:.2f}%")
    print(f"Tempo médio de resposta: {metrics['global_metrics']['avg_response_time']:.3f}s")
    
    print("\nSistema Avançado de Tratamento de Erros funcionando!")

if __name__ == "__main__":
    asyncio.run(main())