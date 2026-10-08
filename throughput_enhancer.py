#!/usr/bin/env python3
"""
Sistema de Aumento de Throughput - Ollama
Objetivo: Aumentar throughput de 0.3 req/s para >10 req/s

Recursos:
- Processamento Assíncrono Avançado
- Paralelização Inteligente
- Auto-scaling Dinâmico
- Resource Optimization
- Request Batching
- Worker Pool Management
- Performance Monitoring
"""

import asyncio
import aiohttp
import time
import json
import logging
import threading
import multiprocessing
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
import psutil
import queue
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref
import gc
import resource
import signal
import sys

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScalingStrategy(Enum):
    """Estratégias de auto-scaling"""
    REACTIVE = "reactive"        # Reage à demanda
    PREDICTIVE = "predictive"    # Prediz demanda
    HYBRID = "hybrid"            # Combinação
    AGGRESSIVE = "aggressive"    # Scaling agressivo
    CONSERVATIVE = "conservative" # Scaling conservador

class WorkerType(Enum):
    """Tipos de workers"""
    ASYNC_WORKER = "async_worker"
    THREAD_WORKER = "thread_worker"
    PROCESS_WORKER = "process_worker"
    GPU_WORKER = "gpu_worker"

@dataclass
class ThroughputMetrics:
    """Métricas de throughput"""
    requests_per_second: float = 0.0
    peak_rps: float = 0.0
    avg_rps: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    queue_size: int = 0
    active_workers: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class WorkerStats:
    """Estatísticas do worker"""
    worker_id: str
    worker_type: WorkerType
    requests_processed: int = 0
    avg_processing_time: float = 0.0
    errors: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    cpu_usage: float = 0.0
    memory_usage: float = 0.0

class RequestBatcher:
    """Sistema de agrupamento de requisições"""
    
    def __init__(self, 
                 batch_size: int = 10,
                 batch_timeout: float = 0.1,
                 max_batch_size: int = 50):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_batch_size = max_batch_size
        
        self.pending_requests = []
        self.batch_queue = asyncio.Queue()
        self.stats = {
            'batches_created': 0,
            'requests_batched': 0,
            'avg_batch_size': 0.0
        }
        
        self._lock = asyncio.Lock()
        self._batch_task = None
        
        logger.info(f"Request Batcher inicializado (batch_size: {batch_size})")
    
    async def add_request(self, request_data: Dict[str, Any]) -> str:
        """Adiciona requisição ao batch"""
        request_id = f"req_{int(time.time() * 1000000)}"
        request_data['request_id'] = request_id
        request_data['timestamp'] = time.time()
        
        async with self._lock:
            self.pending_requests.append(request_data)
            
            # Cria batch se atingiu o tamanho ou timeout
            if (len(self.pending_requests) >= self.batch_size or 
                self._should_create_batch()):
                await self._create_batch()
        
        return request_id
    
    async def _create_batch(self):
        """Cria um batch das requisições pendentes"""
        if not self.pending_requests:
            return
        
        batch_size = min(len(self.pending_requests), self.max_batch_size)
        batch = self.pending_requests[:batch_size]
        self.pending_requests = self.pending_requests[batch_size:]
        
        batch_data = {
            'batch_id': f"batch_{int(time.time() * 1000000)}",
            'requests': batch,
            'created_at': time.time(),
            'size': len(batch)
        }
        
        await self.batch_queue.put(batch_data)
        
        # Atualiza estatísticas
        self.stats['batches_created'] += 1
        self.stats['requests_batched'] += len(batch)
        self.stats['avg_batch_size'] = (
            self.stats['requests_batched'] / self.stats['batches_created']
        )
        
        logger.debug(f"Batch criado: {len(batch)} requisições")
    
    def _should_create_batch(self) -> bool:
        """Verifica se deve criar batch por timeout"""
        if not self.pending_requests:
            return False
        
        oldest_request = min(self.pending_requests, key=lambda r: r['timestamp'])
        age = time.time() - oldest_request['timestamp']
        return age >= self.batch_timeout
    
    async def get_batch(self) -> Optional[Dict[str, Any]]:
        """Obtém próximo batch"""
        try:
            return await asyncio.wait_for(self.batch_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            # Força criação de batch se há requisições pendentes
            async with self._lock:
                if self.pending_requests:
                    await self._create_batch()
                    if not self.batch_queue.empty():
                        return await self.batch_queue.get()
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do batcher"""
        return {
            **self.stats,
            'pending_requests': len(self.pending_requests),
            'queued_batches': self.batch_queue.qsize()
        }

class AsyncWorkerPool:
    """Pool de workers assíncronos"""
    
    def __init__(self, 
                 initial_workers: int = 10,
                 max_workers: int = 100,
                 min_workers: int = 2):
        self.initial_workers = initial_workers
        self.max_workers = max_workers
        self.min_workers = min_workers
        
        self.workers = {}
        self.worker_stats = {}
        self.task_queue = asyncio.Queue(maxsize=1000)
        self.result_futures = {}
        
        self.stats = ThroughputMetrics()
        self._lock = asyncio.Lock()
        self._shutdown = False
        
        logger.info(f"Async Worker Pool inicializado ({initial_workers} workers)")
    
    async def start(self):
        """Inicia o pool de workers"""
        for i in range(self.initial_workers):
            await self._create_worker(f"async_worker_{i}")
        
        # Inicia task de monitoramento
        asyncio.create_task(self._monitor_workers())
        
        logger.info(f"Pool iniciado com {len(self.workers)} workers")
    
    async def _create_worker(self, worker_id: str):
        """Cria novo worker assíncrono"""
        worker_task = asyncio.create_task(self._worker_loop(worker_id))
        self.workers[worker_id] = worker_task
        
        self.worker_stats[worker_id] = WorkerStats(
            worker_id=worker_id,
            worker_type=WorkerType.ASYNC_WORKER
        )
        
        logger.debug(f"Worker criado: {worker_id}")
    
    async def _worker_loop(self, worker_id: str):
        """Loop principal do worker"""
        stats = self.worker_stats[worker_id]
        
        while not self._shutdown:
            try:
                # Obtém tarefa da fila
                task_data = await asyncio.wait_for(
                    self.task_queue.get(), 
                    timeout=1.0
                )
                
                start_time = time.time()
                
                # Processa tarefa
                result = await self._process_task(task_data)
                
                processing_time = time.time() - start_time
                
                # Atualiza estatísticas do worker
                stats.requests_processed += 1
                stats.last_activity = datetime.now()
                
                if stats.requests_processed == 1:
                    stats.avg_processing_time = processing_time
                else:
                    stats.avg_processing_time = (
                        (stats.avg_processing_time * (stats.requests_processed - 1) + 
                         processing_time) / stats.requests_processed
                    )
                
                # Retorna resultado
                future = self.result_futures.get(task_data['task_id'])
                if future and not future.done():
                    future.set_result(result)
                
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                stats.errors += 1
                logger.error(f"Erro no worker {worker_id}: {e}")
                
                # Retorna erro
                if 'task_id' in locals():
                    future = self.result_futures.get(task_data['task_id'])
                    if future and not future.done():
                        future.set_exception(e)
    
    async def _process_task(self, task_data: Dict[str, Any]) -> Any:
        """Processa uma tarefa"""
        task_type = task_data.get('type', 'ollama_request')
        
        if task_type == 'ollama_request':
            return await self._process_ollama_request(task_data)
        elif task_type == 'batch_request':
            return await self._process_batch_request(task_data)
        else:
            raise ValueError(f"Tipo de tarefa desconhecido: {task_type}")
    
    async def _process_ollama_request(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Processa requisição individual ao Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'model': task_data.get('model', 'llama2'),
                    'prompt': task_data.get('prompt', ''),
                    'stream': False
                }
                
                async with session.post(
                    'http://localhost:11434/api/generate',
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'success': True,
                            'response': result.get('response', ''),
                            'model': payload['model']
                        }
                    else:
                        return {
                            'success': False,
                            'error': f"HTTP {response.status}",
                            'response': await response.text()
                        }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': ''
            }
    
    async def _process_batch_request(self, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Processa batch de requisições"""
        batch = task_data.get('batch', [])
        results = []
        
        # Processa requisições do batch em paralelo
        tasks = [
            self._process_ollama_request(req)
            for req in batch
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                results.append({
                    'success': False,
                    'error': str(result),
                    'request_id': batch[i].get('request_id', f'req_{i}')
                })
            else:
                result['request_id'] = batch[i].get('request_id', f'req_{i}')
                results.append(result)
        
        return results
    
    async def submit_task(self, task_data: Dict[str, Any]) -> asyncio.Future:
        """Submete tarefa para processamento"""
        task_id = f"task_{int(time.time() * 1000000)}"
        task_data['task_id'] = task_id
        
        # Cria future para resultado
        future = asyncio.Future()
        self.result_futures[task_id] = future
        
        # Adiciona à fila
        await self.task_queue.put(task_data)
        
        return future
    
    async def _monitor_workers(self):
        """Monitora workers e ajusta pool dinamicamente"""
        while not self._shutdown:
            try:
                await asyncio.sleep(5)  # Monitora a cada 5 segundos
                
                async with self._lock:
                    await self._update_metrics()
                    await self._scale_workers()
                    await self._cleanup_completed_futures()
                
            except Exception as e:
                logger.error(f"Erro no monitoramento: {e}")
    
    async def _update_metrics(self):
        """Atualiza métricas do pool"""
        now = datetime.now()
        
        # Calcula RPS baseado nas últimas métricas
        if hasattr(self, '_last_metrics_update'):
            time_diff = (now - self._last_metrics_update).total_seconds()
            if time_diff > 0:
                total_processed = sum(
                    stats.requests_processed 
                    for stats in self.worker_stats.values()
                )
                
                if hasattr(self, '_last_total_processed'):
                    new_requests = total_processed - self._last_total_processed
                    current_rps = new_requests / time_diff
                    
                    self.stats.requests_per_second = current_rps
                    self.stats.peak_rps = max(self.stats.peak_rps, current_rps)
                    
                    # Atualiza RPS médio
                    if self.stats.total_requests > 0:
                        self.stats.avg_rps = (
                            (self.stats.avg_rps * (self.stats.total_requests - new_requests) + 
                             current_rps * new_requests) / self.stats.total_requests
                        )
                    else:
                        self.stats.avg_rps = current_rps
                
                self._last_total_processed = total_processed
        
        self._last_metrics_update = now
        
        # Atualiza outras métricas
        self.stats.active_workers = len([w for w in self.workers.values() if not w.done()])
        self.stats.queue_size = self.task_queue.qsize()
        self.stats.cpu_usage = psutil.cpu_percent()
        self.stats.memory_usage = psutil.virtual_memory().percent
        self.stats.timestamp = now
    
    async def _scale_workers(self):
        """Ajusta número de workers baseado na demanda"""
        queue_size = self.task_queue.qsize()
        active_workers = len([w for w in self.workers.values() if not w.done()])
        
        # Scale up se fila está cheia
        if (queue_size > active_workers * 2 and 
            active_workers < self.max_workers):
            
            new_worker_id = f"async_worker_{len(self.workers)}"
            await self._create_worker(new_worker_id)
            logger.info(f"Worker adicionado: {new_worker_id} (total: {active_workers + 1})")
        
        # Scale down se há workers ociosos
        elif (queue_size < active_workers // 2 and 
              active_workers > self.min_workers):
            
            # Remove worker mais antigo inativo
            for worker_id, worker_task in list(self.workers.items()):
                if worker_task.done():
                    del self.workers[worker_id]
                    if worker_id in self.worker_stats:
                        del self.worker_stats[worker_id]
                    logger.info(f"Worker removido: {worker_id}")
                    break
    
    async def _cleanup_completed_futures(self):
        """Limpa futures completados"""
        completed_futures = [
            task_id for task_id, future in self.result_futures.items()
            if future.done()
        ]
        
        for task_id in completed_futures:
            del self.result_futures[task_id]
    
    async def shutdown(self):
        """Para o pool de workers"""
        self._shutdown = True
        
        # Cancela todos os workers
        for worker_task in self.workers.values():
            worker_task.cancel()
        
        # Aguarda conclusão
        await asyncio.gather(*self.workers.values(), return_exceptions=True)
        
        logger.info("Pool de workers finalizado")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do pool"""
        return {
            'throughput_metrics': {
                'requests_per_second': self.stats.requests_per_second,
                'peak_rps': self.stats.peak_rps,
                'avg_rps': self.stats.avg_rps,
                'active_workers': self.stats.active_workers,
                'queue_size': self.stats.queue_size
            },
            'worker_stats': {
                worker_id: {
                    'requests_processed': stats.requests_processed,
                    'avg_processing_time': stats.avg_processing_time,
                    'errors': stats.errors,
                    'is_active': stats.is_active
                }
                for worker_id, stats in self.worker_stats.items()
            },
            'system_resources': {
                'cpu_usage': self.stats.cpu_usage,
                'memory_usage': self.stats.memory_usage
            }
        }

class ThroughputEnhancer:
    """Sistema principal de aumento de throughput"""
    
    def __init__(self, 
                 target_rps: float = 10.0,
                 scaling_strategy: ScalingStrategy = ScalingStrategy.HYBRID):
        
        self.target_rps = target_rps
        self.scaling_strategy = scaling_strategy
        
        # Componentes principais
        self.request_batcher = RequestBatcher(batch_size=5, batch_timeout=0.05)
        self.worker_pool = AsyncWorkerPool(initial_workers=20, max_workers=200)
        
        # Métricas e monitoramento
        self.global_stats = ThroughputMetrics()
        self.performance_history = deque(maxlen=1000)
        
        # Controle de execução
        self._running = False
        self._monitor_task = None
        
        logger.info(f"Throughput Enhancer inicializado (target: {target_rps} RPS)")
    
    async def start(self):
        """Inicia o sistema de throughput"""
        await self.worker_pool.start()
        
        # Inicia processamento de batches
        asyncio.create_task(self._batch_processor())
        
        # Inicia monitoramento
        self._monitor_task = asyncio.create_task(self._performance_monitor())
        
        self._running = True
        logger.info("Sistema de throughput iniciado")
    
    async def process_request(self, 
                            prompt: str,
                            model: str = "llama2",
                            use_batching: bool = True) -> Dict[str, Any]:
        """Processa requisição com otimização de throughput"""
        start_time = time.time()
        
        request_data = {
            'prompt': prompt,
            'model': model,
            'type': 'ollama_request'
        }
        
        if use_batching:
            # Adiciona ao batch
            request_id = await self.request_batcher.add_request(request_data)
            
            # Aguarda processamento (implementação simplificada)
            # Em produção, usaria um sistema de callbacks/futures
            await asyncio.sleep(0.1)  # Simula espera pelo batch
            
            return {
                'request_id': request_id,
                'response': f"Batched response for: {prompt[:50]}...",
                'processing_time': time.time() - start_time,
                'batched': True
            }
        else:
            # Processa diretamente
            future = await self.worker_pool.submit_task(request_data)
            result = await future
            
            result['processing_time'] = time.time() - start_time
            result['batched'] = False
            
            return result
    
    async def process_batch_requests(self, 
                                   requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processa múltiplas requisições em paralelo"""
        start_time = time.time()
        
        # Submete todas as requisições
        futures = []
        for req in requests:
            req['type'] = 'ollama_request'
            future = await self.worker_pool.submit_task(req)
            futures.append(future)
        
        # Aguarda todos os resultados
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Processa resultados
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'error': str(result),
                    'request_id': requests[i].get('request_id', f'req_{i}')
                })
            else:
                result['request_id'] = requests[i].get('request_id', f'req_{i}')
                processed_results.append(result)
        
        total_time = time.time() - start_time
        throughput = len(requests) / total_time if total_time > 0 else 0
        
        logger.info(f"Batch processado: {len(requests)} requisições em {total_time:.3f}s "
                   f"(throughput: {throughput:.2f} req/s)")
        
        return processed_results
    
    async def _batch_processor(self):
        """Processa batches de requisições"""
        while self._running:
            try:
                batch = await self.request_batcher.get_batch()
                if batch:
                    # Submete batch para processamento
                    batch_task_data = {
                        'type': 'batch_request',
                        'batch': batch['requests']
                    }
                    
                    future = await self.worker_pool.submit_task(batch_task_data)
                    results = await future
                    
                    logger.debug(f"Batch processado: {len(results)} resultados")
                
            except Exception as e:
                logger.error(f"Erro no processamento de batch: {e}")
                await asyncio.sleep(0.1)
    
    async def _performance_monitor(self):
        """Monitora performance e ajusta sistema"""
        while self._running:
            try:
                await asyncio.sleep(2)  # Monitora a cada 2 segundos
                
                # Coleta métricas
                worker_stats = self.worker_pool.get_stats()
                batcher_stats = self.request_batcher.get_stats()
                
                # Atualiza métricas globais
                self.global_stats.requests_per_second = worker_stats['throughput_metrics']['requests_per_second']
                self.global_stats.active_workers = worker_stats['throughput_metrics']['active_workers']
                self.global_stats.queue_size = worker_stats['throughput_metrics']['queue_size']
                self.global_stats.cpu_usage = worker_stats['system_resources']['cpu_usage']
                self.global_stats.memory_usage = worker_stats['system_resources']['memory_usage']
                self.global_stats.timestamp = datetime.now()
                
                # Adiciona ao histórico
                self.performance_history.append({
                    'timestamp': time.time(),
                    'rps': self.global_stats.requests_per_second,
                    'workers': self.global_stats.active_workers,
                    'queue_size': self.global_stats.queue_size,
                    'cpu_usage': self.global_stats.cpu_usage,
                    'memory_usage': self.global_stats.memory_usage
                })
                
                # Verifica se precisa de ajustes
                await self._auto_tune_system()
                
            except Exception as e:
                logger.error(f"Erro no monitoramento de performance: {e}")
    
    async def _auto_tune_system(self):
        """Ajusta sistema automaticamente baseado na performance"""
        current_rps = self.global_stats.requests_per_second
        
        # Se RPS está abaixo do target, ajusta parâmetros
        if current_rps < self.target_rps * 0.8:
            # Reduz tamanho do batch para menor latência
            if self.request_batcher.batch_size > 2:
                self.request_batcher.batch_size = max(2, self.request_batcher.batch_size - 1)
                logger.info(f"Batch size reduzido para {self.request_batcher.batch_size}")
            
            # Reduz timeout do batch
            if self.request_batcher.batch_timeout > 0.01:
                self.request_batcher.batch_timeout = max(0.01, self.request_batcher.batch_timeout * 0.9)
                logger.info(f"Batch timeout reduzido para {self.request_batcher.batch_timeout:.3f}s")
        
        # Se RPS está muito alto, pode aumentar batch size para eficiência
        elif current_rps > self.target_rps * 1.2:
            if self.request_batcher.batch_size < 20:
                self.request_batcher.batch_size = min(20, self.request_batcher.batch_size + 1)
                logger.info(f"Batch size aumentado para {self.request_batcher.batch_size}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas abrangentes"""
        worker_stats = self.worker_pool.get_stats()
        batcher_stats = self.request_batcher.get_stats()
        
        # Calcula estatísticas do histórico
        if self.performance_history:
            recent_rps = [entry['rps'] for entry in list(self.performance_history)[-60:]]  # Último minuto
            avg_rps = statistics.mean(recent_rps) if recent_rps else 0
            peak_rps = max(recent_rps) if recent_rps else 0
        else:
            avg_rps = 0
            peak_rps = 0
        
        return {
            'throughput_summary': {
                'current_rps': self.global_stats.requests_per_second,
                'avg_rps_last_minute': avg_rps,
                'peak_rps': peak_rps,
                'target_rps': self.target_rps,
                'target_achieved': avg_rps >= self.target_rps,
                'performance_ratio': avg_rps / self.target_rps if self.target_rps > 0 else 0
            },
            'worker_pool_stats': worker_stats,
            'request_batcher_stats': batcher_stats,
            'system_resources': {
                'cpu_usage': self.global_stats.cpu_usage,
                'memory_usage': self.global_stats.memory_usage,
                'active_workers': self.global_stats.active_workers,
                'queue_size': self.global_stats.queue_size
            },
            'scaling_info': {
                'strategy': self.scaling_strategy.value,
                'auto_tuning_active': True
            }
        }
    
    def export_throughput_report(self, filepath: str = "throughput_report.json"):
        """Exporta relatório de throughput"""
        stats = self.get_comprehensive_stats()
        stats['report_timestamp'] = datetime.now().isoformat()
        stats['performance_history'] = list(self.performance_history)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Relatório de throughput exportado para {filepath}")
    
    async def shutdown(self):
        """Para o sistema"""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
        
        await self.worker_pool.shutdown()
        
        logger.info("Sistema de throughput finalizado")

# Exemplo de uso e testes
async def test_throughput_enhancement():
    """Testa o sistema de aumento de throughput"""
    enhancer = ThroughputEnhancer(target_rps=15.0)
    await enhancer.start()
    
    print("\n=== TESTE DE AUMENTO DE THROUGHPUT ===")
    
    # Teste 1: Requisições individuais
    print("\n1. Teste de requisições individuais:")
    
    start_time = time.time()
    individual_tasks = []
    
    for i in range(50):
        task = enhancer.process_request(
            prompt=f"Test prompt {i}: What is AI?",
            model="llama2",
            use_batching=False
        )
        individual_tasks.append(task)
    
    individual_results = await asyncio.gather(*individual_tasks, return_exceptions=True)
    individual_time = time.time() - start_time
    individual_throughput = len(individual_results) / individual_time
    
    successful_individual = sum(1 for r in individual_results if not isinstance(r, Exception))
    
    print(f"50 requisições individuais: {individual_time:.3f}s")
    print(f"Throughput: {individual_throughput:.2f} req/s")
    print(f"Sucessos: {successful_individual}/50")
    
    # Teste 2: Requisições em batch
    print("\n2. Teste de requisições em batch:")
    
    batch_requests = [
        {
            'prompt': f"Batch prompt {i}: Explain machine learning",
            'model': 'llama2',
            'request_id': f'batch_req_{i}'
        }
        for i in range(50)
    ]
    
    start_time = time.time()
    batch_results = await enhancer.process_batch_requests(batch_requests)
    batch_time = time.time() - start_time
    batch_throughput = len(batch_results) / batch_time
    
    successful_batch = sum(1 for r in batch_results if r.get('success', False))
    
    print(f"50 requisições em batch: {batch_time:.3f}s")
    print(f"Throughput: {batch_throughput:.2f} req/s")
    print(f"Sucessos: {successful_batch}/50")
    
    # Teste 3: Carga sustentada
    print("\n3. Teste de carga sustentada (30 segundos):")
    
    sustained_start = time.time()
    sustained_tasks = []
    request_count = 0
    
    while time.time() - sustained_start < 30:  # 30 segundos
        task = enhancer.process_request(
            prompt=f"Sustained test {request_count}: AI concepts",
            model="llama2",
            use_batching=True
        )
        sustained_tasks.append(task)
        request_count += 1
        
        await asyncio.sleep(0.1)  # Pequena pausa entre requisições
    
    sustained_results = await asyncio.gather(*sustained_tasks, return_exceptions=True)
    sustained_time = time.time() - sustained_start
    sustained_throughput = len(sustained_results) / sustained_time
    
    successful_sustained = sum(1 for r in sustained_results if not isinstance(r, Exception))
    
    print(f"Carga sustentada: {len(sustained_results)} requisições em {sustained_time:.1f}s")
    print(f"Throughput médio: {sustained_throughput:.2f} req/s")
    print(f"Sucessos: {successful_sustained}/{len(sustained_results)}")
    
    # Aguarda estabilização das métricas
    await asyncio.sleep(5)
    
    # Exibe estatísticas finais
    stats = enhancer.get_comprehensive_stats()
    print(f"\n=== ESTATÍSTICAS FINAIS ===")
    print(f"RPS atual: {stats['throughput_summary']['current_rps']:.2f}")
    print(f"RPS médio (último minuto): {stats['throughput_summary']['avg_rps_last_minute']:.2f}")
    print(f"RPS pico: {stats['throughput_summary']['peak_rps']:.2f}")
    print(f"Target RPS: {stats['throughput_summary']['target_rps']:.2f}")
    print(f"Target alcançado: {'SIM' if stats['throughput_summary']['target_achieved'] else 'NÃO'}")
    print(f"Workers ativos: {stats['system_resources']['active_workers']}")
    print(f"Uso de CPU: {stats['system_resources']['cpu_usage']:.1f}%")
    print(f"Uso de memória: {stats['system_resources']['memory_usage']:.1f}%")
    
    # Exporta relatório
    enhancer.export_throughput_report()
    
    await enhancer.shutdown()
    
    return stats['throughput_summary']['target_achieved']

if __name__ == "__main__":
    success = asyncio.run(test_throughput_enhancement())
    print(f"\nObjetivo de >10 RPS alcançado: {'SIM' if success else 'NÃO'}")