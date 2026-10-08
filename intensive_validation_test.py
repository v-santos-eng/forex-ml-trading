#!/usr/bin/env python3
"""
Teste Intensivo de Validação das Otimizações - Ollama
Objetivo: Validar completamente as melhorias implementadas
"""

import asyncio
import aiohttp
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import statistics
import psutil
from pathlib import Path
import numpy as np

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class IntensiveTestConfig:
    """Configuração do teste intensivo"""
    name: str
    concurrent_users: int
    duration_seconds: int
    ramp_up_seconds: int = 10
    timeout: int = 45
    endpoint: str = "http://localhost:11434/api/generate"
    model: str = "llama3.2:1b"
    prompts: List[str] = None
    
    def __post_init__(self):
        if self.prompts is None:
            self.prompts = [
                "What is AI?",
                "Explain machine learning.",
                "How does neural network work?",
                "What is deep learning?",
                "Describe natural language processing."
            ]

@dataclass
class IntensiveMetrics:
    """Métricas detalhadas do teste intensivo"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    response_times: List[float] = None
    throughput_history: List[Tuple[float, float]] = None  # (timestamp, throughput)
    error_history: List[Tuple[float, float]] = None  # (timestamp, error_rate)
    cpu_history: List[Tuple[float, float]] = None
    memory_history: List[Tuple[float, float]] = None
    start_time: datetime = None
    end_time: datetime = None
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []
        if self.throughput_history is None:
            self.throughput_history = []
        if self.error_history is None:
            self.error_history = []
        if self.cpu_history is None:
            self.cpu_history = []
        if self.memory_history is None:
            self.memory_history = []
    
    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100
    
    @property
    def timeout_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.timeout_requests / self.total_requests) * 100
    
    @property
    def success_rate(self) -> float:
        return 100 - self.error_rate
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def p95_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return np.percentile(self.response_times, 95)
    
    @property
    def p99_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return np.percentile(self.response_times, 99)
    
    @property
    def throughput(self) -> float:
        if not self.start_time or not self.end_time:
            return 0.0
        duration = (self.end_time - self.start_time).total_seconds()
        if duration <= 0:
            return 0.0
        return self.successful_requests / duration
    
    @property
    def peak_throughput(self) -> float:
        if not self.throughput_history:
            return 0.0
        return max(tp for _, tp in self.throughput_history)

class IntensiveLoadTester:
    """Testador de carga intensivo"""
    
    def __init__(self, config: IntensiveTestConfig):
        self.config = config
        self.session = None
        self._stop_event = asyncio.Event()
        self._request_counter = 0
        self._success_counter = 0
        
    async def run_test(self) -> IntensiveMetrics:
        """Executa o teste intensivo"""
        logger.info(f"Iniciando teste intensivo: {self.config.name}")
        logger.info(f"Configuração: {self.config.concurrent_users} usuários, {self.config.duration_seconds}s")
        
        metrics = IntensiveMetrics()
        metrics.start_time = datetime.now()
        
        # Configuração otimizada da sessão HTTP
        connector = aiohttp.TCPConnector(
            limit=200,
            limit_per_host=50,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=self.config.timeout,
            connect=5,
            sock_read=40
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        try:
            # Inicia monitoramento de sistema
            monitor_task = asyncio.create_task(self._monitor_system(metrics))
            
            # Inicia workers com ramp-up gradual
            await self._run_ramped_test(metrics)
                
        except Exception as e:
            logger.error(f"Erro durante o teste: {e}")
        finally:
            if self.session:
                await self.session.close()
            
        metrics.end_time = datetime.now()
        return metrics
    
    async def _run_ramped_test(self, metrics: IntensiveMetrics):
        """Executa teste com ramp-up gradual"""
        start_time = time.time()
        
        # Fase 1: Ramp-up gradual
        logger.info("Fase 1: Ramp-up gradual")
        for i in range(1, self.config.concurrent_users + 1):
            if i <= self.config.concurrent_users:
                task = asyncio.create_task(self._worker_loop(metrics, f"worker_{i}"))
                
            # Pausa entre adição de workers
            if i < self.config.concurrent_users:
                await asyncio.sleep(self.config.ramp_up_seconds / self.config.concurrent_users)
        
        # Fase 2: Carga sustentada
        logger.info("Fase 2: Carga sustentada")
        steady_duration = self.config.duration_seconds - self.config.ramp_up_seconds
        await asyncio.sleep(steady_duration)
        
        # Fase 3: Finalização
        logger.info("Fase 3: Finalizando teste")
        self._stop_event.set()
    
    async def _worker_loop(self, metrics: IntensiveMetrics, worker_id: str):
        """Loop do worker com variação de prompts"""
        prompt_index = 0
        
        while not self._stop_event.is_set():
            try:
                # Varia os prompts para simular carga real
                prompt = self.config.prompts[prompt_index % len(self.config.prompts)]
                prompt_index += 1
                
                await self._make_request(metrics, prompt)
                
                # Pausa variável entre requisições (0.1 a 0.5s)
                await asyncio.sleep(0.1 + (prompt_index % 5) * 0.1)
                
            except Exception as e:
                logger.error(f"Erro no worker {worker_id}: {e}")
                await asyncio.sleep(1)
    
    async def _make_request(self, metrics: IntensiveMetrics, prompt: str):
        """Faz uma requisição otimizada"""
        start_time = time.time()
        
        try:
            payload = {
                'model': self.config.model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 50  # Limita resposta para acelerar
                }
            }
            
            async with self.session.post(
                self.config.endpoint,
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                
                response_time = time.time() - start_time
                metrics.total_requests += 1
                metrics.response_times.append(response_time)
                
                if response.status == 200:
                    metrics.successful_requests += 1
                    self._success_counter += 1
                else:
                    metrics.failed_requests += 1
                    
                self._request_counter += 1
                    
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            metrics.total_requests += 1
            metrics.timeout_requests += 1
            metrics.failed_requests += 1
            metrics.response_times.append(response_time)
            
        except Exception as e:
            response_time = time.time() - start_time
            metrics.total_requests += 1
            metrics.failed_requests += 1
            metrics.response_times.append(response_time)
    
    async def _monitor_system(self, metrics: IntensiveMetrics):
        """Monitora métricas do sistema"""
        start_time = time.time()
        
        while not self._stop_event.is_set():
            try:
                current_time = time.time() - start_time
                
                # CPU e Memória
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                metrics.cpu_history.append((current_time, cpu_percent))
                metrics.memory_history.append((current_time, memory_percent))
                
                # Throughput instantâneo (últimos 5 segundos)
                if current_time > 5:
                    recent_success = self._success_counter
                    throughput = recent_success / 5 if recent_success > 0 else 0
                    metrics.throughput_history.append((current_time, throughput))
                    self._success_counter = 0  # Reset counter
                
                # Taxa de erro instantânea
                if metrics.total_requests > 0:
                    error_rate = (metrics.failed_requests / metrics.total_requests) * 100
                    metrics.error_history.append((current_time, error_rate))
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Erro no monitoramento: {e}")
                await asyncio.sleep(1)

def generate_intensive_report(config: IntensiveTestConfig, metrics: IntensiveMetrics) -> Dict[str, Any]:
    """Gera relatório detalhado do teste intensivo"""
    return {
        'test_name': config.name,
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'concurrent_users': config.concurrent_users,
            'duration_seconds': config.duration_seconds,
            'ramp_up_seconds': config.ramp_up_seconds,
            'timeout': config.timeout,
            'model': config.model,
            'prompts_count': len(config.prompts)
        },
        'results': {
            'total_requests': metrics.total_requests,
            'successful_requests': metrics.successful_requests,
            'failed_requests': metrics.failed_requests,
            'timeout_requests': metrics.timeout_requests,
            'error_rate': round(metrics.error_rate, 2),
            'timeout_rate': round(metrics.timeout_rate, 2),
            'success_rate': round(metrics.success_rate, 2),
            'avg_response_time': round(metrics.avg_response_time, 2),
            'p95_response_time': round(metrics.p95_response_time, 2),
            'p99_response_time': round(metrics.p99_response_time, 2),
            'min_response_time': round(min(metrics.response_times) if metrics.response_times else 0, 2),
            'max_response_time': round(max(metrics.response_times) if metrics.response_times else 0, 2),
            'avg_throughput': round(metrics.throughput, 2),
            'peak_throughput': round(metrics.peak_throughput, 2),
            'duration': round((metrics.end_time - metrics.start_time).total_seconds(), 2)
        },
        'performance_analysis': {
            'avg_cpu_usage': round(statistics.mean([cpu for _, cpu in metrics.cpu_history]) if metrics.cpu_history else 0, 2),
            'peak_cpu_usage': round(max([cpu for _, cpu in metrics.cpu_history]) if metrics.cpu_history else 0, 2),
            'avg_memory_usage': round(statistics.mean([mem for _, mem in metrics.memory_history]) if metrics.memory_history else 0, 2),
            'peak_memory_usage': round(max([mem for _, mem in metrics.memory_history]) if metrics.memory_history else 0, 2)
        },
        'optimization_validation': {
            'error_rate_target': '< 5%',
            'error_rate_achieved': f"{metrics.error_rate:.1f}%",
            'error_rate_status': 'PASS' if metrics.error_rate < 5 else 'FAIL',
            
            'response_time_target': '< 3s',
            'response_time_achieved': f"{metrics.avg_response_time:.2f}s",
            'response_time_status': 'PASS' if metrics.avg_response_time < 3 else 'FAIL',
            
            'throughput_target': '> 1 req/s',
            'throughput_achieved': f"{metrics.throughput:.2f} req/s",
            'throughput_status': 'PASS' if metrics.throughput > 1 else 'FAIL'
        },
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2)
        }
    }

async def run_intensive_validation():
    """Executa validação intensiva das otimizações"""
    logger.info("=== INICIANDO VALIDAÇÃO INTENSIVA DAS OTIMIZAÇÕES ===")
    
    test_configs = [
        IntensiveTestConfig(
            name="Teste de Estresse Leve",
            concurrent_users=3,
            duration_seconds=90,
            ramp_up_seconds=15
        ),
        IntensiveTestConfig(
            name="Teste de Estresse Moderado",
            concurrent_users=5,
            duration_seconds=120,
            ramp_up_seconds=20
        ),
        IntensiveTestConfig(
            name="Teste de Estresse Intenso",
            concurrent_users=8,
            duration_seconds=150,
            ramp_up_seconds=30
        )
    ]
    
    results = []
    reports_dir = Path("intensive_validation_reports")
    reports_dir.mkdir(exist_ok=True)
    
    for config in test_configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Executando: {config.name}")
        logger.info(f"{'='*50}")
        
        tester = IntensiveLoadTester(config)
        metrics = await tester.run_test()
        report = generate_intensive_report(config, metrics)
        results.append(report)
        
        # Salva relatório individual
        timestamp = int(time.time())
        filename = f"{config.name.lower().replace(' ', '_')}_{timestamp}.json"
        with open(reports_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Exibe resultados
        print(f"\n=== RESULTADOS: {config.name} ===")
        print(f"Requisições totais: {metrics.total_requests}")
        print(f"Taxa de sucesso: {metrics.success_rate:.1f}%")
        print(f"Taxa de erro: {metrics.error_rate:.1f}%")
        print(f"Tempo médio: {metrics.avg_response_time:.2f}s")
        print(f"P95: {metrics.p95_response_time:.2f}s")
        print(f"P99: {metrics.p99_response_time:.2f}s")
        print(f"Throughput médio: {metrics.throughput:.2f} req/s")
        print(f"Throughput pico: {metrics.peak_throughput:.2f} req/s")
        
        # Validação dos objetivos
        validation = report['optimization_validation']
        print(f"\n=== VALIDAÇÃO DOS OBJETIVOS ===")
        print(f"Taxa de erro: {validation['error_rate_status']} ({validation['error_rate_achieved']})")
        print(f"Tempo de resposta: {validation['response_time_status']} ({validation['response_time_achieved']})")
        print(f"Throughput: {validation['throughput_status']} ({validation['throughput_achieved']})")
        
        # Pausa entre testes para estabilização
        if config != test_configs[-1]:
            logger.info("Aguardando estabilização do sistema...")
            await asyncio.sleep(30)
    
    # Relatório consolidado
    consolidated_report = {
        'validation_summary': {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'tests_passed': sum(1 for r in results if all(
                r['optimization_validation'][key].endswith('PASS') 
                for key in ['error_rate_status', 'response_time_status', 'throughput_status']
            )),
            'overall_status': 'PASS' if all(
                all(r['optimization_validation'][key].endswith('PASS') 
                    for key in ['error_rate_status', 'response_time_status', 'throughput_status'])
                for r in results
            ) else 'PARTIAL'
        },
        'individual_results': results
    }
    
    # Salva relatório consolidado
    with open(reports_dir / f"consolidated_validation_{int(time.time())}.json", 'w', encoding='utf-8') as f:
        json.dump(consolidated_report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nRelatórios salvos em: {reports_dir}")
    
    # Resumo final
    print(f"\n{'='*60}")
    print("RESUMO FINAL DA VALIDAÇÃO")
    print(f"{'='*60}")
    
    summary = consolidated_report['validation_summary']
    print(f"Testes executados: {summary['total_tests']}")
    print(f"Testes aprovados: {summary['tests_passed']}")
    print(f"Status geral: {summary['overall_status']}")
    
    if summary['overall_status'] == 'PASS':
        print("\n✅ TODAS AS OTIMIZAÇÕES FORAM VALIDADAS COM SUCESSO!")
        print("\nAs melhorias implementadas atingiram os objetivos:")
        print("- Taxa de erro < 5%")
        print("- Tempo de resposta < 3s")
        print("- Throughput > 1 req/s")
    else:
        print("\n⚠️  VALIDAÇÃO PARCIAL - ALGUMAS OTIMIZAÇÕES PRECISAM DE AJUSTES")
        print("\nVerifique os relatórios detalhados para identificar áreas de melhoria.")

if __name__ == "__main__":
    asyncio.run(run_intensive_validation())