#!/usr/bin/env python3
"""
Framework de Testes de Carga Otimizado - Ollama
Versão calibrada para validação das otimizações implementadas
"""

import asyncio
import aiohttp
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
import statistics
import psutil
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """Configuração do teste"""
    name: str
    concurrent_users: int = 1
    duration_seconds: int = 30
    timeout: int = 60  # Timeout aumentado
    endpoint: str = "http://localhost:11434/api/generate"
    model: str = "llama3.2:1b"
    prompt: str = "Hello, how are you?"

@dataclass
class TestMetrics:
    """Métricas do teste"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = None
    start_time: datetime = None
    end_time: datetime = None
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []
    
    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100
    
    @property
    def success_rate(self) -> float:
        return 100 - self.error_rate
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def throughput(self) -> float:
        if not self.start_time or not self.end_time:
            return 0.0
        duration = (self.end_time - self.start_time).total_seconds()
        if duration <= 0:
            return 0.0
        return self.successful_requests / duration
    
    @property
    def duration(self) -> float:
        if not self.start_time or not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

class OptimizedLoadTester:
    """Testador de carga otimizado"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.session = None
        self._stop_event = asyncio.Event()
        
    async def run_test(self) -> TestMetrics:
        """Executa o teste de carga"""
        logger.info(f"Iniciando teste: {self.config.name}")
        
        metrics = TestMetrics()
        metrics.start_time = datetime.now()
        
        # Configuração da sessão HTTP com timeouts apropriados
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=self.config.timeout,
            connect=10,
            sock_read=30
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        try:
            # Executa teste sequencial primeiro para validar conectividade
            if self.config.concurrent_users == 1:
                await self._run_sequential_test(metrics)
            else:
                await self._run_concurrent_test(metrics)
                
        except Exception as e:
            logger.error(f"Erro durante o teste: {e}")
        finally:
            if self.session:
                await self.session.close()
            
        metrics.end_time = datetime.now()
        return metrics
    
    async def _run_sequential_test(self, metrics: TestMetrics):
        """Executa teste sequencial"""
        end_time = time.time() + self.config.duration_seconds
        
        while time.time() < end_time and not self._stop_event.is_set():
            await self._make_request(metrics)
            await asyncio.sleep(1)  # Pausa entre requisições
    
    async def _run_concurrent_test(self, metrics: TestMetrics):
        """Executa teste concorrente"""
        tasks = []
        
        for i in range(self.config.concurrent_users):
            task = asyncio.create_task(self._worker_loop(metrics))
            tasks.append(task)
        
        # Aguarda duração do teste
        await asyncio.sleep(self.config.duration_seconds)
        self._stop_event.set()
        
        # Aguarda conclusão dos workers
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _worker_loop(self, metrics: TestMetrics):
        """Loop do worker"""
        while not self._stop_event.is_set():
            await self._make_request(metrics)
            await asyncio.sleep(0.5)  # Pausa entre requisições
    
    async def _make_request(self, metrics: TestMetrics):
        """Faz uma requisição"""
        start_time = time.time()
        
        try:
            payload = {
                'model': self.config.model,
                'prompt': self.config.prompt,
                'stream': False
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
                    logger.debug(f"Requisição bem-sucedida em {response_time:.2f}s")
                else:
                    metrics.failed_requests += 1
                    logger.warning(f"Requisição falhou com status {response.status}")
                    
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            metrics.total_requests += 1
            metrics.failed_requests += 1
            metrics.response_times.append(response_time)
            logger.warning(f"Timeout na requisição após {response_time:.2f}s")
            
        except Exception as e:
            response_time = time.time() - start_time
            metrics.total_requests += 1
            metrics.failed_requests += 1
            metrics.response_times.append(response_time)
            logger.error(f"Erro na requisição: {e}")

def generate_report(config: TestConfig, metrics: TestMetrics) -> Dict[str, Any]:
    """Gera relatório do teste"""
    return {
        'test_name': config.name,
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'concurrent_users': config.concurrent_users,
            'duration_seconds': config.duration_seconds,
            'timeout': config.timeout,
            'model': config.model
        },
        'results': {
            'total_requests': metrics.total_requests,
            'successful_requests': metrics.successful_requests,
            'failed_requests': metrics.failed_requests,
            'error_rate': round(metrics.error_rate, 2),
            'success_rate': round(metrics.success_rate, 2),
            'avg_response_time': round(metrics.avg_response_time, 2),
            'min_response_time': round(min(metrics.response_times) if metrics.response_times else 0, 2),
            'max_response_time': round(max(metrics.response_times) if metrics.response_times else 0, 2),
            'throughput': round(metrics.throughput, 2),
            'duration': round(metrics.duration, 2)
        },
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent
        }
    }

async def run_validation_suite():
    """Executa suíte de validação das otimizações"""
    logger.info("=== INICIANDO VALIDAÇÃO DAS OTIMIZAÇÕES ===")
    
    # Teste 1: Conectividade básica
    config1 = TestConfig(
        name="Teste de Conectividade",
        concurrent_users=1,
        duration_seconds=30,
        timeout=60
    )
    
    tester1 = OptimizedLoadTester(config1)
    metrics1 = await tester1.run_test()
    report1 = generate_report(config1, metrics1)
    
    print(f"\n=== {config1.name} ===")
    print(f"Requisições: {metrics1.total_requests}")
    print(f"Taxa de sucesso: {metrics1.success_rate:.1f}%")
    print(f"Tempo médio: {metrics1.avg_response_time:.2f}s")
    print(f"Throughput: {metrics1.throughput:.2f} req/s")
    
    # Teste 2: Carga moderada (se o primeiro passou)
    if metrics1.success_rate > 50:
        config2 = TestConfig(
            name="Teste de Carga Moderada",
            concurrent_users=2,
            duration_seconds=60,
            timeout=60
        )
        
        tester2 = OptimizedLoadTester(config2)
        metrics2 = await tester2.run_test()
        report2 = generate_report(config2, metrics2)
        
        print(f"\n=== {config2.name} ===")
        print(f"Requisições: {metrics2.total_requests}")
        print(f"Taxa de sucesso: {metrics2.success_rate:.1f}%")
        print(f"Tempo médio: {metrics2.avg_response_time:.2f}s")
        print(f"Throughput: {metrics2.throughput:.2f} req/s")
        
        # Salva relatórios
        reports_dir = Path("validation_reports")
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        
        with open(reports_dir / f"connectivity_test_{timestamp}.json", 'w') as f:
            json.dump(report1, f, indent=2)
            
        with open(reports_dir / f"load_test_{timestamp}.json", 'w') as f:
            json.dump(report2, f, indent=2)
            
        logger.info(f"Relatórios salvos em {reports_dir}")
        
        # Análise dos resultados
        print("\n=== ANÁLISE DOS RESULTADOS ===")
        
        if metrics1.success_rate >= 95 and metrics1.avg_response_time <= 3:
            print("✅ Conectividade: EXCELENTE")
        elif metrics1.success_rate >= 80:
            print("⚠️  Conectividade: BOA (necessita ajustes)")
        else:
            print("❌ Conectividade: PROBLEMÁTICA")
            
        if metrics2.success_rate >= 90 and metrics2.throughput >= 1:
            print("✅ Carga: EXCELENTE")
        elif metrics2.success_rate >= 70:
            print("⚠️  Carga: BOA (necessita otimização)")
        else:
            print("❌ Carga: PROBLEMÁTICA")
            
        # Recomendações
        print("\n=== RECOMENDAÇÕES ===")
        
        if metrics1.avg_response_time > 5:
            print("- Considere otimizar o modelo ou usar GPU")
            
        if metrics2.success_rate < 90:
            print("- Ajuste os timeouts ou reduza a concorrência")
            
        if metrics2.throughput < 1:
            print("- Implemente cache ou otimize o pipeline")
            
    else:
        print("\n❌ Teste de conectividade falhou. Verifique:")
        print("- Se o Ollama está rodando (ollama serve)")
        print("- Se o modelo está disponível (ollama list)")
        print("- Se não há firewall bloqueando a porta 11434")

if __name__ == "__main__":
    asyncio.run(run_validation_suite())