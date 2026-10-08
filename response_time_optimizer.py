#!/usr/bin/env python3
"""
Sistema de Otimização de Tempo de Resposta - Ollama
Objetivo: Reduzir tempo médio de resposta de 31.7s para <3s

Recursos:
- Cache Inteligente Multi-Layer
- Connection Pooling Avançado
- Load Balancing Dinâmico
- Query Optimization
- Predictive Caching
- Response Compression
"""

import asyncio
import aiohttp
import time
import json
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from enum import Enum
import statistics
import pickle
import gzip
import weakref
from concurrent.futures import ThreadPoolExecutor
import psutil
import random

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Níveis de cache"""
    L1_MEMORY = "l1_memory"      # Cache em memória rápido
    L2_COMPRESSED = "l2_compressed"  # Cache comprimido
    L3_PERSISTENT = "l3_persistent"  # Cache persistente

class LoadBalanceStrategy(Enum):
    """Estratégias de balanceamento de carga"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME = "response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    ADAPTIVE = "adaptive"

@dataclass
class CacheEntry:
    """Entrada do cache"""
    key: str
    value: Any
    timestamp: datetime
    ttl: float
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    compression_ratio: float = 1.0

@dataclass
class ServerNode:
    """Nó do servidor"""
    url: str
    weight: float = 1.0
    active_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_response_time: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.now)
    is_healthy: bool = True
    max_connections: int = 100

class IntelligentCache:
    """Sistema de cache inteligente multi-layer"""
    
    def __init__(self, 
                 l1_max_size: int = 1000,
                 l2_max_size: int = 5000,
                 l3_max_size: int = 10000,
                 default_ttl: float = 300.0):
        
        self.l1_cache = OrderedDict()  # LRU cache rápido
        self.l2_cache = {}  # Cache comprimido
        self.l3_cache = {}  # Cache persistente
        
        self.l1_max_size = l1_max_size
        self.l2_max_size = l2_max_size
        self.l3_max_size = l3_max_size
        self.default_ttl = default_ttl
        
        self.access_patterns = defaultdict(list)
        self.prediction_model = {}
        
        self._lock = threading.RLock()
        self.stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'l3_hits': 0, 'l3_misses': 0,
            'total_requests': 0,
            'cache_size_bytes': 0
        }
        
        # Thread para limpeza automática
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("Cache inteligente multi-layer inicializado")
    
    def _generate_key(self, data: Any) -> str:
        """Gera chave única para os dados"""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _compress_data(self, data: Any) -> bytes:
        """Comprime dados para armazenamento"""
        serialized = pickle.dumps(data)
        compressed = gzip.compress(serialized)
        return compressed
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Descomprime dados"""
        decompressed = gzip.decompress(compressed_data)
        return pickle.loads(decompressed)
    
    def _calculate_size(self, data: Any) -> int:
        """Calcula tamanho dos dados em bytes"""
        try:
            return len(pickle.dumps(data))
        except:
            return len(str(data).encode())
    
    async def get(self, key: str) -> Optional[Any]:
        """Recupera item do cache (multi-layer)"""
        with self._lock:
            self.stats['total_requests'] += 1
            
            # Tenta L1 (mais rápido)
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                if self._is_valid(entry):
                    # Move para o final (LRU)
                    self.l1_cache.move_to_end(key)
                    entry.access_count += 1
                    entry.last_access = datetime.now()
                    self.stats['l1_hits'] += 1
                    self._record_access_pattern(key)
                    return entry.value
                else:
                    del self.l1_cache[key]
            
            self.stats['l1_misses'] += 1
            
            # Tenta L2 (comprimido)
            if key in self.l2_cache:
                entry = self.l2_cache[key]
                if self._is_valid(entry):
                    try:
                        value = self._decompress_data(entry.value)
                        # Promove para L1
                        await self._promote_to_l1(key, value, entry.ttl)
                        entry.access_count += 1
                        entry.last_access = datetime.now()
                        self.stats['l2_hits'] += 1
                        self._record_access_pattern(key)
                        return value
                    except Exception as e:
                        logger.warning(f"Erro ao descomprimir cache L2: {e}")
                        del self.l2_cache[key]
            
            self.stats['l2_misses'] += 1
            
            # Tenta L3 (persistente)
            if key in self.l3_cache:
                entry = self.l3_cache[key]
                if self._is_valid(entry):
                    # Promove para L2
                    await self._promote_to_l2(key, entry.value, entry.ttl)
                    entry.access_count += 1
                    entry.last_access = datetime.now()
                    self.stats['l3_hits'] += 1
                    self._record_access_pattern(key)
                    return entry.value
                else:
                    del self.l3_cache[key]
            
            self.stats['l3_misses'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: float = None) -> bool:
        """Armazena item no cache"""
        if ttl is None:
            ttl = self.default_ttl
        
        with self._lock:
            size_bytes = self._calculate_size(value)
            
            # Sempre armazena em L1 primeiro
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            await self._store_in_l1(key, entry)
            
            # Decide se deve armazenar em níveis inferiores baseado no padrão de acesso
            if self._should_store_in_l2(key, size_bytes):
                await self._store_in_l2(key, value, ttl)
            
            if self._should_store_in_l3(key, size_bytes):
                await self._store_in_l3(key, value, ttl)
            
            self._record_access_pattern(key)
            return True
    
    async def _store_in_l1(self, key: str, entry: CacheEntry):
        """Armazena em L1 (memória rápida)"""
        # Remove itens antigos se necessário
        while len(self.l1_cache) >= self.l1_max_size:
            oldest_key = next(iter(self.l1_cache))
            del self.l1_cache[oldest_key]
        
        self.l1_cache[key] = entry
        self.l1_cache.move_to_end(key)
    
    async def _store_in_l2(self, key: str, value: Any, ttl: float):
        """Armazena em L2 (comprimido)"""
        try:
            compressed_value = self._compress_data(value)
            original_size = self._calculate_size(value)
            compressed_size = len(compressed_value)
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                timestamp=datetime.now(),
                ttl=ttl,
                size_bytes=compressed_size,
                compression_ratio=compression_ratio
            )
            
            # Remove itens antigos se necessário
            while len(self.l2_cache) >= self.l2_max_size:
                oldest_key = min(self.l2_cache.keys(), 
                               key=lambda k: self.l2_cache[k].last_access)
                del self.l2_cache[oldest_key]
            
            self.l2_cache[key] = entry
            
        except Exception as e:
            logger.warning(f"Erro ao comprimir para L2: {e}")
    
    async def _store_in_l3(self, key: str, value: Any, ttl: float):
        """Armazena em L3 (persistente)"""
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=datetime.now(),
            ttl=ttl,
            size_bytes=self._calculate_size(value)
        )
        
        # Remove itens antigos se necessário
        while len(self.l3_cache) >= self.l3_max_size:
            oldest_key = min(self.l3_cache.keys(), 
                           key=lambda k: self.l3_cache[k].last_access)
            del self.l3_cache[oldest_key]
        
        self.l3_cache[key] = entry
    
    async def _promote_to_l1(self, key: str, value: Any, ttl: float):
        """Promove item para L1"""
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=datetime.now(),
            ttl=ttl,
            size_bytes=self._calculate_size(value)
        )
        await self._store_in_l1(key, entry)
    
    async def _promote_to_l2(self, key: str, value: Any, ttl: float):
        """Promove item para L2"""
        await self._store_in_l2(key, value, ttl)
        await self._promote_to_l1(key, value, ttl)
    
    def _is_valid(self, entry: CacheEntry) -> bool:
        """Verifica se entrada do cache ainda é válida"""
        age = (datetime.now() - entry.timestamp).total_seconds()
        return age < entry.ttl
    
    def _should_store_in_l2(self, key: str, size_bytes: int) -> bool:
        """Decide se deve armazenar em L2"""
        # Armazena em L2 se o item é grande ou tem padrão de acesso frequente
        access_frequency = len(self.access_patterns.get(key, []))
        return size_bytes > 1024 or access_frequency > 5
    
    def _should_store_in_l3(self, key: str, size_bytes: int) -> bool:
        """Decide se deve armazenar em L3"""
        # Armazena em L3 se tem padrão de acesso muito frequente
        access_frequency = len(self.access_patterns.get(key, []))
        return access_frequency > 10
    
    def _record_access_pattern(self, key: str):
        """Registra padrão de acesso para predição"""
        now = datetime.now()
        self.access_patterns[key].append(now)
        
        # Mantém apenas os últimos 100 acessos
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]
    
    def _cleanup_worker(self):
        """Worker thread para limpeza automática"""
        while True:
            try:
                time.sleep(60)  # Executa a cada minuto
                self._cleanup_expired_entries()
            except Exception as e:
                logger.error(f"Erro na limpeza do cache: {e}")
    
    def _cleanup_expired_entries(self):
        """Remove entradas expiradas"""
        with self._lock:
            now = datetime.now()
            
            # Limpa L1
            expired_keys = [
                key for key, entry in self.l1_cache.items()
                if (now - entry.timestamp).total_seconds() > entry.ttl
            ]
            for key in expired_keys:
                del self.l1_cache[key]
            
            # Limpa L2
            expired_keys = [
                key for key, entry in self.l2_cache.items()
                if (now - entry.timestamp).total_seconds() > entry.ttl
            ]
            for key in expired_keys:
                del self.l2_cache[key]
            
            # Limpa L3
            expired_keys = [
                key for key, entry in self.l3_cache.items()
                if (now - entry.timestamp).total_seconds() > entry.ttl
            ]
            for key in expired_keys:
                del self.l3_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        with self._lock:
            total_hits = self.stats['l1_hits'] + self.stats['l2_hits'] + self.stats['l3_hits']
            total_requests = self.stats['total_requests']
            hit_rate = total_hits / total_requests if total_requests > 0 else 0
            
            return {
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'l1_size': len(self.l1_cache),
                'l2_size': len(self.l2_cache),
                'l3_size': len(self.l3_cache),
                'l1_hit_rate': self.stats['l1_hits'] / total_requests if total_requests > 0 else 0,
                'l2_hit_rate': self.stats['l2_hits'] / total_requests if total_requests > 0 else 0,
                'l3_hit_rate': self.stats['l3_hits'] / total_requests if total_requests > 0 else 0,
                'memory_usage_mb': sum(
                    entry.size_bytes for entry in list(self.l1_cache.values()) + 
                    list(self.l2_cache.values()) + list(self.l3_cache.values())
                ) / (1024 * 1024)
            }

class ConnectionPool:
    """Pool de conexões avançado"""
    
    def __init__(self, 
                 max_connections: int = 100,
                 max_connections_per_host: int = 30,
                 connection_timeout: float = 10.0,
                 read_timeout: float = 30.0):
        
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.connection_timeout = connection_timeout
        self.read_timeout = read_timeout
        
        # Configuração do connector
        connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections_per_host,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        # Timeout configuration
        timeout = aiohttp.ClientTimeout(
            total=read_timeout,
            connect=connection_timeout
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        self.stats = {
            'active_connections': 0,
            'total_requests': 0,
            'connection_errors': 0,
            'timeout_errors': 0
        }
        
        logger.info(f"Connection pool inicializado (max: {max_connections})")
    
    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Executa requisição usando o pool"""
        self.stats['total_requests'] += 1
        self.stats['active_connections'] += 1
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                return response
        except asyncio.TimeoutError:
            self.stats['timeout_errors'] += 1
            raise
        except aiohttp.ClientError:
            self.stats['connection_errors'] += 1
            raise
        finally:
            self.stats['active_connections'] -= 1
    
    async def close(self):
        """Fecha o pool de conexões"""
        await self.session.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do pool"""
        return self.stats.copy()

class LoadBalancer:
    """Balanceador de carga inteligente"""
    
    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.ADAPTIVE):
        self.strategy = strategy
        self.servers = []
        self.current_index = 0
        self.stats = defaultdict(int)
        self._lock = threading.Lock()
        
        # Health check thread
        self.health_check_thread = threading.Thread(target=self._health_check_worker, daemon=True)
        self.health_check_thread.start()
        
        logger.info(f"Load balancer inicializado (estratégia: {strategy.value})")
    
    def add_server(self, url: str, weight: float = 1.0, max_connections: int = 100):
        """Adiciona servidor ao pool"""
        server = ServerNode(
            url=url,
            weight=weight,
            max_connections=max_connections
        )
        self.servers.append(server)
        logger.info(f"Servidor adicionado: {url} (peso: {weight})")
    
    def get_server(self) -> Optional[ServerNode]:
        """Seleciona servidor baseado na estratégia"""
        with self._lock:
            healthy_servers = [s for s in self.servers if s.is_healthy]
            
            if not healthy_servers:
                logger.warning("Nenhum servidor saudável disponível")
                return None
            
            if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
                return self._round_robin_selection(healthy_servers)
            elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(healthy_servers)
            elif self.strategy == LoadBalanceStrategy.RESPONSE_TIME:
                return self._response_time_selection(healthy_servers)
            elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(healthy_servers)
            elif self.strategy == LoadBalanceStrategy.ADAPTIVE:
                return self._adaptive_selection(healthy_servers)
            else:
                return healthy_servers[0]
    
    def _round_robin_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Seleção round robin"""
        server = servers[self.current_index % len(servers)]
        self.current_index += 1
        return server
    
    def _least_connections_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Seleção por menor número de conexões"""
        return min(servers, key=lambda s: s.active_connections)
    
    def _response_time_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Seleção por menor tempo de resposta"""
        return min(servers, key=lambda s: s.avg_response_time)
    
    def _weighted_round_robin_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Seleção round robin ponderada"""
        total_weight = sum(s.weight for s in servers)
        random_weight = random.uniform(0, total_weight)
        
        current_weight = 0
        for server in servers:
            current_weight += server.weight
            if random_weight <= current_weight:
                return server
        
        return servers[-1]
    
    def _adaptive_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Seleção adaptativa baseada em múltiplas métricas"""
        def score_server(server: ServerNode) -> float:
            # Combina múltiplas métricas
            connection_score = 1.0 - (server.active_connections / server.max_connections)
            response_time_score = 1.0 / (1.0 + server.avg_response_time)
            error_rate = server.total_errors / max(server.total_requests, 1)
            error_score = 1.0 - error_rate
            weight_score = server.weight
            
            return (connection_score * 0.3 + 
                   response_time_score * 0.4 + 
                   error_score * 0.2 + 
                   weight_score * 0.1)
        
        return max(servers, key=score_server)
    
    def record_request(self, server: ServerNode, response_time: float, success: bool):
        """Registra resultado da requisição"""
        with self._lock:
            server.total_requests += 1
            
            if success:
                # Atualiza tempo médio de resposta
                if server.total_requests == 1:
                    server.avg_response_time = response_time
                else:
                    server.avg_response_time = (
                        (server.avg_response_time * (server.total_requests - 1) + response_time) /
                        server.total_requests
                    )
            else:
                server.total_errors += 1
    
    def _health_check_worker(self):
        """Worker thread para health checks"""
        while True:
            try:
                time.sleep(30)  # Health check a cada 30 segundos
                asyncio.run(self._perform_health_checks())
            except Exception as e:
                logger.error(f"Erro no health check: {e}")
    
    async def _perform_health_checks(self):
        """Executa health checks em todos os servidores"""
        for server in self.servers:
            try:
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{server.url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            server.is_healthy = True
                            response_time = time.time() - start_time
                            server.last_health_check = datetime.now()
                        else:
                            server.is_healthy = False
                            
            except Exception:
                server.is_healthy = False
                logger.warning(f"Health check falhou para {server.url}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do load balancer"""
        with self._lock:
            return {
                'total_servers': len(self.servers),
                'healthy_servers': sum(1 for s in self.servers if s.is_healthy),
                'strategy': self.strategy.value,
                'servers': [
                    {
                        'url': s.url,
                        'is_healthy': s.is_healthy,
                        'active_connections': s.active_connections,
                        'total_requests': s.total_requests,
                        'total_errors': s.total_errors,
                        'avg_response_time': s.avg_response_time,
                        'error_rate': s.total_errors / max(s.total_requests, 1)
                    }
                    for s in self.servers
                ]
            }

class ResponseTimeOptimizer:
    """Sistema principal de otimização de tempo de resposta"""
    
    def __init__(self):
        self.cache = IntelligentCache()
        self.connection_pool = ConnectionPool()
        self.load_balancer = LoadBalancer()
        self.executor = ThreadPoolExecutor(max_workers=50)
        
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'avg_response_time': 0.0,
            'requests_under_3s': 0
        }
        
        self._lock = threading.Lock()
        
        # Adiciona servidores Ollama padrão
        self.load_balancer.add_server("http://localhost:11434", weight=1.0)
        
        logger.info("Sistema de Otimização de Tempo de Resposta inicializado")
    
    def add_ollama_server(self, url: str, weight: float = 1.0):
        """Adiciona servidor Ollama ao pool"""
        self.load_balancer.add_server(url, weight)
    
    async def optimized_request(self, 
                              prompt: str,
                              model: str = "llama2",
                              use_cache: bool = True,
                              cache_ttl: float = 300.0) -> Dict[str, Any]:
        """Executa requisição otimizada ao Ollama"""
        start_time = time.time()
        
        with self._lock:
            self.stats['total_requests'] += 1
        
        # Gera chave do cache
        cache_key = self.cache._generate_key({
            'prompt': prompt,
            'model': model
        })
        
        # Tenta cache primeiro
        if use_cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                response_time = time.time() - start_time
                
                with self._lock:
                    self.stats['cache_hits'] += 1
                    self._update_response_time_stats(response_time)
                
                logger.info(f"Cache HIT - Tempo: {response_time:.3f}s")
                return {
                    'response': cached_result,
                    'cached': True,
                    'response_time': response_time,
                    'server_used': 'cache'
                }
        
        # Cache miss - executa requisição
        try:
            result = await self._execute_ollama_request(prompt, model)
            response_time = time.time() - start_time
            
            # Armazena no cache
            if use_cache and result.get('response'):
                await self.cache.set(cache_key, result['response'], cache_ttl)
            
            with self._lock:
                self._update_response_time_stats(response_time)
            
            result.update({
                'cached': False,
                'response_time': response_time
            })
            
            logger.info(f"Requisição executada - Tempo: {response_time:.3f}s")
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Erro na requisição: {e} - Tempo: {response_time:.3f}s")
            raise
    
    async def _execute_ollama_request(self, prompt: str, model: str) -> Dict[str, Any]:
        """Executa requisição ao Ollama com load balancing"""
        server = self.load_balancer.get_server()
        if not server:
            raise Exception("Nenhum servidor Ollama disponível")
        
        server.active_connections += 1
        start_time = time.time()
        
        try:
            # Prepara payload
            payload = {
                'model': model,
                'prompt': prompt,
                'stream': False
            }
            
            # Executa requisição
            response = await self.connection_pool.request(
                'POST',
                f"{server.url}/api/generate",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status == 200:
                result = await response.json()
                response_time = time.time() - start_time
                
                # Registra sucesso
                self.load_balancer.record_request(server, response_time, True)
                
                return {
                    'response': result.get('response', ''),
                    'server_used': server.url,
                    'model': model
                }
            else:
                raise Exception(f"Erro HTTP {response.status}: {await response.text()}")
                
        except Exception as e:
            response_time = time.time() - start_time
            self.load_balancer.record_request(server, response_time, False)
            raise
        finally:
            server.active_connections -= 1
    
    def _update_response_time_stats(self, response_time: float):
        """Atualiza estatísticas de tempo de resposta"""
        # Atualiza tempo médio
        current_avg = self.stats['avg_response_time']
        total_requests = self.stats['total_requests']
        
        self.stats['avg_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
        
        # Conta requisições sob 3 segundos
        if response_time < 3.0:
            self.stats['requests_under_3s'] += 1
    
    async def batch_request(self, 
                          requests: List[Dict[str, Any]],
                          max_concurrent: int = 10) -> List[Dict[str, Any]]:
        """Executa múltiplas requisições em paralelo"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_request(req_data):
            async with semaphore:
                return await self.optimized_request(**req_data)
        
        tasks = [process_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas abrangentes"""
        with self._lock:
            cache_stats = self.cache.get_stats()
            pool_stats = self.connection_pool.get_stats()
            lb_stats = self.load_balancer.get_stats()
            
            success_rate = (
                self.stats['requests_under_3s'] / self.stats['total_requests']
                if self.stats['total_requests'] > 0 else 0
            )
            
            return {
                'response_time_stats': {
                    'avg_response_time': self.stats['avg_response_time'],
                    'requests_under_3s': self.stats['requests_under_3s'],
                    'success_rate_under_3s': success_rate,
                    'total_requests': self.stats['total_requests']
                },
                'cache_stats': cache_stats,
                'connection_pool_stats': pool_stats,
                'load_balancer_stats': lb_stats,
                'system_resources': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'active_threads': threading.active_count()
                }
            }
    
    def export_performance_report(self, filepath: str = "response_time_report.json"):
        """Exporta relatório de performance"""
        stats = self.get_comprehensive_stats()
        stats['report_timestamp'] = datetime.now().isoformat()
        stats['target_achieved'] = stats['response_time_stats']['avg_response_time'] < 3.0
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Relatório de performance exportado para {filepath}")
    
    async def cleanup(self):
        """Limpa recursos"""
        await self.connection_pool.close()
        self.executor.shutdown(wait=True)

# Exemplo de uso e testes
async def test_response_time_optimization():
    """Testa o sistema de otimização"""
    optimizer = ResponseTimeOptimizer()
    
    # Testa requisições individuais
    test_prompts = [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?",
        "How does AI work?",
        "What is machine learning?"  # Repetida para testar cache
    ]
    
    print("\n=== TESTE DE OTIMIZAÇÃO DE TEMPO DE RESPOSTA ===")
    
    start_time = time.time()
    results = []
    
    for i, prompt in enumerate(test_prompts):
        try:
            result = await optimizer.optimized_request(
                prompt=prompt,
                model="llama2",
                use_cache=True
            )
            results.append(result)
            
            print(f"Requisição {i+1}: {result['response_time']:.3f}s "
                  f"({'CACHE' if result['cached'] else 'SERVIDOR'})")
            
        except Exception as e:
            print(f"Requisição {i+1} falhou: {e}")
            results.append({'error': str(e), 'response_time': 0})
    
    total_time = time.time() - start_time
    
    # Testa requisições em lote
    print("\n=== TESTE DE REQUISIÇÕES EM LOTE ===")
    
    batch_requests = [
        {'prompt': f"Question {i}: What is AI?", 'model': 'llama2'}
        for i in range(10)
    ]
    
    batch_start = time.time()
    batch_results = await optimizer.batch_request(batch_requests, max_concurrent=5)
    batch_time = time.time() - batch_start
    
    successful_batch = [r for r in batch_results if not isinstance(r, Exception)]
    
    print(f"Lote de 10 requisições: {batch_time:.3f}s")
    print(f"Sucessos: {len(successful_batch)}/10")
    
    # Exibe estatísticas finais
    stats = optimizer.get_comprehensive_stats()
    print(f"\n=== ESTATÍSTICAS FINAIS ===")
    print(f"Tempo médio de resposta: {stats['response_time_stats']['avg_response_time']:.3f}s")
    print(f"Requisições < 3s: {stats['response_time_stats']['success_rate_under_3s']:.1%}")
    print(f"Taxa de cache hit: {stats['cache_stats']['hit_rate']:.1%}")
    
    # Exporta relatório
    optimizer.export_performance_report()
    
    await optimizer.cleanup()
    
    return stats['response_time_stats']['avg_response_time'] < 3.0

if __name__ == "__main__":
    success = asyncio.run(test_response_time_optimization())
    print(f"\nObjetivo de <3s alcançado: {'SIM' if success else 'NÃO'}")