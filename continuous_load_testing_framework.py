#!/usr/bin/env python3
"""
Framework de Testes de Carga Contínuos - Ollama
Objetivo: Validar escalabilidade e performance automaticamente

Recursos:
- Stress Testing Automatizado
- Performance Benchmarking
- Validação de Escalabilidade
- Testes de Resiliência
- Análise de Capacidade
- Relatórios Detalhados
- Alertas Proativos
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestType(Enum):
    """Tipos de teste de carga"""
    SMOKE_TEST = "smoke_test"              # Teste básico de funcionamento
    LOAD_TEST = "load_test"                # Teste de carga normal
    STRESS_TEST = "stress_test"            # Teste de stress
    SPIKE_TEST = "spike_test"              # Teste de picos de carga
    VOLUME_TEST = "volume_test"            # Teste de volume
    ENDURANCE_TEST = "endurance_test"      # Teste de resistência
    SCALABILITY_TEST = "scalability_test"  # Teste de escalabilidade
    CAPACITY_TEST = "capacity_test"        # Teste de capacidade máxima

class TestStatus(Enum):
    """Status do teste"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AlertLevel(Enum):
    """Níveis de alerta"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class TestConfiguration:
    """Configuração de teste"""
    test_type: TestType
    name: str
    description: str
    duration_seconds: int
    concurrent_users: int
    requests_per_second: float
    ramp_up_time: int = 30
    ramp_down_time: int = 30
    think_time: float = 1.0
    timeout: int = 30
    target_endpoint: str = "http://localhost:11434/api/generate"
    test_data: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    failure_criteria: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestMetrics:
    """Métricas de teste"""
    test_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    error_rates: List[float] = field(default_factory=list)
    throughput_history: List[float] = field(default_factory=list)
    cpu_usage_history: List[float] = field(default_factory=list)
    memory_usage_history: List[float] = field(default_factory=list)
    network_io_history: List[Dict[str, float]] = field(default_factory=list)
    custom_metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def avg_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0.0
    
    @property
    def p95_response_time(self) -> float:
        if self.response_times:
            return np.percentile(self.response_times, 95)
        return 0.0
    
    @property
    def p99_response_time(self) -> float:
        if self.response_times:
            return np.percentile(self.response_times, 99)
        return 0.0
    
    @property
    def error_rate(self) -> float:
        if self.total_requests > 0:
            return (self.failed_requests / self.total_requests) * 100
        return 0.0
    
    @property
    def throughput(self) -> float:
        if self.duration > 0:
            return self.successful_requests / self.duration
        return 0.0

@dataclass
class TestResult:
    """Resultado do teste"""
    test_id: str
    configuration: TestConfiguration
    metrics: TestMetrics
    status: TestStatus
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)  # Caminhos para gráficos, logs, etc.
    
    @property
    def passed(self) -> bool:
        """Verifica se o teste passou nos critérios de sucesso"""
        if self.status != TestStatus.COMPLETED:
            return False
        
        criteria = self.configuration.success_criteria
        
        # Verifica critérios básicos
        if 'max_error_rate' in criteria:
            if self.metrics.error_rate > criteria['max_error_rate']:
                return False
        
        if 'max_avg_response_time' in criteria:
            if self.metrics.avg_response_time > criteria['max_avg_response_time']:
                return False
        
        if 'min_throughput' in criteria:
            if self.metrics.throughput < criteria['min_throughput']:
                return False
        
        return True

class LoadGenerator:
    """Gerador de carga para testes"""
    
    def __init__(self, config: TestConfiguration):
        self.config = config
        self.session = None
        self.active_requests = 0
        self.request_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()
        self._stop_event = asyncio.Event()
        
        logger.info(f"Load Generator criado para {config.name}")
    
    async def start_test(self) -> TestMetrics:
        """Inicia o teste de carga"""
        test_id = f"{self.config.test_type.value}_{int(time.time())}"
        metrics = TestMetrics(test_id=test_id, start_time=datetime.now())
        
        logger.info(f"Iniciando teste: {self.config.name}")
        
        # Cria sessão HTTP
        connector = aiohttp.TCPConnector(
            limit=self.config.concurrent_users * 2,
            limit_per_host=self.config.concurrent_users * 2
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        
        try:
            # Inicia workers
            workers = []
            for i in range(self.config.concurrent_users):
                worker = asyncio.create_task(
                    self._worker_loop(f"worker_{i}", metrics)
                )
                workers.append(worker)
            
            # Inicia gerador de requisições
            request_generator = asyncio.create_task(
                self._generate_requests(metrics)
            )
            
            # Inicia coletor de métricas
            metrics_collector = asyncio.create_task(
                self._collect_metrics(metrics)
            )
            
            # Aguarda conclusão do teste
            await asyncio.sleep(self.config.duration_seconds)
            
            # Para o teste
            self._stop_event.set()
            
            # Aguarda finalização dos workers
            await asyncio.gather(*workers, return_exceptions=True)
            request_generator.cancel()
            metrics_collector.cancel()
            
            metrics.end_time = datetime.now()
            
            logger.info(f"Teste concluído: {metrics.total_requests} requisições em {metrics.duration:.1f}s")
            
        finally:
            if self.session:
                await self.session.close()
        
        return metrics
    
    async def _generate_requests(self, metrics: TestMetrics):
        """Gera requisições baseado na configuração"""
        start_time = time.time()
        request_interval = 1.0 / self.config.requests_per_second if self.config.requests_per_second > 0 else 1.0
        
        while not self._stop_event.is_set():
            try:
                # Calcula fase do teste (ramp-up, steady, ramp-down)
                elapsed = time.time() - start_time
                
                if elapsed < self.config.ramp_up_time:
                    # Fase de ramp-up
                    ramp_factor = elapsed / self.config.ramp_up_time
                    current_rps = self.config.requests_per_second * ramp_factor
                elif elapsed > (self.config.duration_seconds - self.config.ramp_down_time):
                    # Fase de ramp-down
                    remaining = self.config.duration_seconds - elapsed
                    ramp_factor = remaining / self.config.ramp_down_time
                    current_rps = self.config.requests_per_second * ramp_factor
                else:
                    # Fase steady
                    current_rps = self.config.requests_per_second
                
                if current_rps > 0:
                    interval = 1.0 / current_rps
                    
                    # Gera requisição
                    request_data = self._create_request_data()
                    await self.request_queue.put(request_data)
                    
                    await asyncio.sleep(interval)
                else:
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Erro na geração de requisições: {e}")
                await asyncio.sleep(0.1)
    
    def _create_request_data(self) -> Dict[str, Any]:
        """Cria dados da requisição"""
        base_data = {
            'model': 'llama3.2:1b',
            'prompt': 'What is artificial intelligence?',
            'stream': False
        }
        
        # Aplica dados customizados do teste
        if self.config.test_data:
            base_data.update(self.config.test_data)
        
        # Adiciona variação nos prompts para testes mais realistas
        prompts = [
            "What is artificial intelligence?",
            "Explain machine learning concepts.",
            "How does deep learning work?",
            "What are neural networks?",
            "Describe natural language processing."
        ]
        
        base_data['prompt'] = np.random.choice(prompts)
        
        return base_data
    
    async def _worker_loop(self, worker_id: str, metrics: TestMetrics):
        """Loop do worker para processar requisições"""
        while not self._stop_event.is_set():
            try:
                # Obtém requisição da fila
                request_data = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=1.0
                )
                
                # Processa requisição
                start_time = time.time()
                success = await self._make_request(request_data)
                response_time = time.time() - start_time
                
                # Atualiza métricas
                metrics.total_requests += 1
                metrics.response_times.append(response_time)
                
                if success:
                    metrics.successful_requests += 1
                else:
                    metrics.failed_requests += 1
                
                # Think time
                if self.config.think_time > 0:
                    await asyncio.sleep(self.config.think_time)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Erro no worker {worker_id}: {e}")
                metrics.failed_requests += 1
    
    async def _make_request(self, request_data: Dict[str, Any]) -> bool:
        """Faz requisição HTTP"""
        try:
            async with self.session.post(
                self.config.target_endpoint,
                json=request_data
            ) as response:
                
                if response.status == 200:
                    await response.json()  # Consome resposta
                    return True
                else:
                    logger.debug(f"Requisição falhou: HTTP {response.status}")
                    return False
        
        except Exception as e:
            logger.debug(f"Erro na requisição: {e}")
            return False
    
    async def _collect_metrics(self, metrics: TestMetrics):
        """Coleta métricas do sistema durante o teste"""
        while not self._stop_event.is_set():
            try:
                # Coleta métricas do sistema
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_usage = psutil.virtual_memory().percent
                
                # Coleta métricas de rede
                net_io = psutil.net_io_counters()
                network_metrics = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
                
                # Adiciona às métricas
                metrics.cpu_usage_history.append(cpu_usage)
                metrics.memory_usage_history.append(memory_usage)
                metrics.network_io_history.append(network_metrics)
                
                # Calcula taxa de erro atual
                if metrics.total_requests > 0:
                    current_error_rate = (metrics.failed_requests / metrics.total_requests) * 100
                    metrics.error_rates.append(current_error_rate)
                
                # Calcula throughput atual
                if len(metrics.response_times) >= 10:  # Últimas 10 requisições
                    recent_times = metrics.response_times[-10:]
                    if recent_times:
                        avg_time = statistics.mean(recent_times)
                        current_throughput = 1.0 / avg_time if avg_time > 0 else 0
                        metrics.throughput_history.append(current_throughput)
                
                await asyncio.sleep(1)  # Coleta a cada segundo
                
            except Exception as e:
                logger.error(f"Erro na coleta de métricas: {e}")
                await asyncio.sleep(1)

class TestAnalyzer:
    """Analisador de resultados de teste"""
    
    def __init__(self):
        self.reports_dir = Path("load_test_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        logger.info("Test Analyzer inicializado")
    
    def analyze_test_result(self, result: TestResult) -> TestResult:
        """Analisa resultado do teste e gera recomendações"""
        logger.info(f"Analisando resultado do teste: {result.test_id}")
        
        # Gera alertas baseado nas métricas
        self._generate_alerts(result)
        
        # Gera recomendações
        self._generate_recommendations(result)
        
        # Cria visualizações
        self._create_visualizations(result)
        
        # Gera relatório
        self._generate_report(result)
        
        return result
    
    def _generate_alerts(self, result: TestResult):
        """Gera alertas baseado nas métricas"""
        metrics = result.metrics
        config = result.configuration
        
        # Alerta de taxa de erro alta
        if metrics.error_rate > 10:
            result.alerts.append({
                'level': AlertLevel.CRITICAL.value,
                'message': f"Taxa de erro muito alta: {metrics.error_rate:.1f}%",
                'metric': 'error_rate',
                'value': metrics.error_rate,
                'threshold': 10
            })
        elif metrics.error_rate > 5:
            result.alerts.append({
                'level': AlertLevel.WARNING.value,
                'message': f"Taxa de erro elevada: {metrics.error_rate:.1f}%",
                'metric': 'error_rate',
                'value': metrics.error_rate,
                'threshold': 5
            })
        
        # Alerta de tempo de resposta alto
        if metrics.avg_response_time > 5:
            result.alerts.append({
                'level': AlertLevel.CRITICAL.value,
                'message': f"Tempo de resposta muito alto: {metrics.avg_response_time:.2f}s",
                'metric': 'avg_response_time',
                'value': metrics.avg_response_time,
                'threshold': 5
            })
        elif metrics.avg_response_time > 3:
            result.alerts.append({
                'level': AlertLevel.WARNING.value,
                'message': f"Tempo de resposta elevado: {metrics.avg_response_time:.2f}s",
                'metric': 'avg_response_time',
                'value': metrics.avg_response_time,
                'threshold': 3
            })
        
        # Alerta de throughput baixo
        if metrics.throughput < 5:
            result.alerts.append({
                'level': AlertLevel.WARNING.value,
                'message': f"Throughput baixo: {metrics.throughput:.2f} req/s",
                'metric': 'throughput',
                'value': metrics.throughput,
                'threshold': 5
            })
        
        # Alerta de uso de recursos
        if metrics.cpu_usage_history:
            avg_cpu = statistics.mean(metrics.cpu_usage_history)
            if avg_cpu > 90:
                result.alerts.append({
                    'level': AlertLevel.CRITICAL.value,
                    'message': f"Uso de CPU muito alto: {avg_cpu:.1f}%",
                    'metric': 'cpu_usage',
                    'value': avg_cpu,
                    'threshold': 90
                })
        
        if metrics.memory_usage_history:
            avg_memory = statistics.mean(metrics.memory_usage_history)
            if avg_memory > 85:
                result.alerts.append({
                    'level': AlertLevel.WARNING.value,
                    'message': f"Uso de memória alto: {avg_memory:.1f}%",
                    'metric': 'memory_usage',
                    'value': avg_memory,
                    'threshold': 85
                })
    
    def _generate_recommendations(self, result: TestResult):
        """Gera recomendações baseado na análise"""
        metrics = result.metrics
        
        # Recomendações baseadas na taxa de erro
        if metrics.error_rate > 10:
            result.recommendations.extend([
                "Implementar circuit breakers para evitar cascata de falhas",
                "Revisar configuração de timeout e retry policies",
                "Verificar capacidade do servidor Ollama",
                "Considerar implementação de rate limiting"
            ])
        
        # Recomendações baseadas no tempo de resposta
        if metrics.avg_response_time > 3:
            result.recommendations.extend([
                "Implementar cache para requisições frequentes",
                "Otimizar queries e processamento",
                "Considerar load balancing",
                "Avaliar upgrade de hardware (CPU/GPU)"
            ])
        
        # Recomendações baseadas no throughput
        if metrics.throughput < 10:
            result.recommendations.extend([
                "Implementar processamento assíncrono",
                "Aumentar número de workers",
                "Otimizar connection pooling",
                "Considerar arquitetura distribuída"
            ])
        
        # Recomendações baseadas no uso de recursos
        if metrics.cpu_usage_history and statistics.mean(metrics.cpu_usage_history) > 80:
            result.recommendations.extend([
                "Considerar scaling horizontal",
                "Otimizar algoritmos de processamento",
                "Implementar auto-scaling baseado em CPU"
            ])
    
    def _create_visualizations(self, result: TestResult):
        """Cria visualizações dos resultados"""
        metrics = result.metrics
        test_id = result.test_id
        
        # Configura estilo dos gráficos
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Cria figura com subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Análise de Performance - {result.configuration.name}', fontsize=16)
        
        # 1. Distribuição de tempos de resposta
        if metrics.response_times:
            axes[0, 0].hist(metrics.response_times, bins=50, alpha=0.7, color='skyblue')
            axes[0, 0].axvline(metrics.avg_response_time, color='red', linestyle='--', 
                              label=f'Média: {metrics.avg_response_time:.2f}s')
            axes[0, 0].axvline(metrics.p95_response_time, color='orange', linestyle='--', 
                              label=f'P95: {metrics.p95_response_time:.2f}s')
            axes[0, 0].set_title('Distribuição de Tempos de Resposta')
            axes[0, 0].set_xlabel('Tempo de Resposta (s)')
            axes[0, 0].set_ylabel('Frequência')
            axes[0, 0].legend()
        
        # 2. Throughput ao longo do tempo
        if metrics.throughput_history:
            time_points = list(range(len(metrics.throughput_history)))
            axes[0, 1].plot(time_points, metrics.throughput_history, color='green', linewidth=2)
            axes[0, 1].set_title('Throughput ao Longo do Tempo')
            axes[0, 1].set_xlabel('Tempo (s)')
            axes[0, 1].set_ylabel('Requisições/s')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Taxa de erro ao longo do tempo
        if metrics.error_rates:
            time_points = list(range(len(metrics.error_rates)))
            axes[0, 2].plot(time_points, metrics.error_rates, color='red', linewidth=2)
            axes[0, 2].set_title('Taxa de Erro ao Longo do Tempo')
            axes[0, 2].set_xlabel('Tempo (s)')
            axes[0, 2].set_ylabel('Taxa de Erro (%)')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Uso de CPU
        if metrics.cpu_usage_history:
            time_points = list(range(len(metrics.cpu_usage_history)))
            axes[1, 0].plot(time_points, metrics.cpu_usage_history, color='purple', linewidth=2)
            axes[1, 0].set_title('Uso de CPU')
            axes[1, 0].set_xlabel('Tempo (s)')
            axes[1, 0].set_ylabel('CPU (%)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Uso de Memória
        if metrics.memory_usage_history:
            time_points = list(range(len(metrics.memory_usage_history)))
            axes[1, 1].plot(time_points, metrics.memory_usage_history, color='brown', linewidth=2)
            axes[1, 1].set_title('Uso de Memória')
            axes[1, 1].set_xlabel('Tempo (s)')
            axes[1, 1].set_ylabel('Memória (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Resumo de métricas
        axes[1, 2].axis('off')
        summary_text = f"""
        RESUMO DO TESTE
        
        Duração: {metrics.duration:.1f}s
        Total de Requisições: {metrics.total_requests}
        Requisições Bem-sucedidas: {metrics.successful_requests}
        Taxa de Erro: {metrics.error_rate:.1f}%
        
        Tempo de Resposta:
        • Médio: {metrics.avg_response_time:.2f}s
        • P95: {metrics.p95_response_time:.2f}s
        • P99: {metrics.p99_response_time:.2f}s
        
        Throughput: {metrics.throughput:.2f} req/s
        
        Status: {'PASSOU' if result.passed else 'FALHOU'}
        """
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Salva gráfico
        chart_path = self.reports_dir / f"{test_id}_performance_chart.png"
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        result.artifacts['performance_chart'] = str(chart_path)
        
        logger.info(f"Gráfico de performance salvo: {chart_path}")
    
    def _generate_report(self, result: TestResult):
        """Gera relatório detalhado do teste"""
        report_data = {
            'test_info': {
                'test_id': result.test_id,
                'test_name': result.configuration.name,
                'test_type': result.configuration.test_type.value,
                'description': result.configuration.description,
                'timestamp': result.metrics.start_time.isoformat(),
                'duration': result.metrics.duration,
                'status': result.status.value,
                'passed': result.passed
            },
            'configuration': {
                'concurrent_users': result.configuration.concurrent_users,
                'requests_per_second': result.configuration.requests_per_second,
                'duration_seconds': result.configuration.duration_seconds,
                'ramp_up_time': result.configuration.ramp_up_time,
                'ramp_down_time': result.configuration.ramp_down_time,
                'target_endpoint': result.configuration.target_endpoint
            },
            'metrics': {
                'total_requests': result.metrics.total_requests,
                'successful_requests': result.metrics.successful_requests,
                'failed_requests': result.metrics.failed_requests,
                'error_rate': result.metrics.error_rate,
                'avg_response_time': result.metrics.avg_response_time,
                'p95_response_time': result.metrics.p95_response_time,
                'p99_response_time': result.metrics.p99_response_time,
                'throughput': result.metrics.throughput
            },
            'system_resources': {
                'avg_cpu_usage': statistics.mean(result.metrics.cpu_usage_history) if result.metrics.cpu_usage_history else 0,
                'max_cpu_usage': max(result.metrics.cpu_usage_history) if result.metrics.cpu_usage_history else 0,
                'avg_memory_usage': statistics.mean(result.metrics.memory_usage_history) if result.metrics.memory_usage_history else 0,
                'max_memory_usage': max(result.metrics.memory_usage_history) if result.metrics.memory_usage_history else 0
            },
            'alerts': result.alerts,
            'recommendations': result.recommendations,
            'artifacts': result.artifacts
        }
        
        # Salva relatório JSON
        report_path = self.reports_dir / f"{result.test_id}_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        result.artifacts['json_report'] = str(report_path)
        
        # Gera relatório HTML
        html_report = self._generate_html_report(report_data)
        html_path = self.reports_dir / f"{result.test_id}_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        result.artifacts['html_report'] = str(html_path)
        
        logger.info(f"Relatórios salvos: {report_path}, {html_path}")
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Gera relatório HTML"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório de Teste de Carga - {report_data['test_info']['test_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                .alert {{ padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .alert.warning {{ background-color: #fff3cd; border-color: #ffeaa7; }}
                .alert.critical {{ background-color: #f8d7da; border-color: #f5c6cb; }}
                .passed {{ color: green; font-weight: bold; }}
                .failed {{ color: red; font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Relatório de Teste de Carga</h1>
                <h2>{report_data['test_info']['test_name']}</h2>
                <p><strong>Tipo:</strong> {report_data['test_info']['test_type']}</p>
                <p><strong>Data/Hora:</strong> {report_data['test_info']['timestamp']}</p>
                <p><strong>Status:</strong> <span class="{'passed' if report_data['test_info']['passed'] else 'failed'}">
                    {'PASSOU' if report_data['test_info']['passed'] else 'FALHOU'}
                </span></p>
            </div>
            
            <div class="section">
                <h3>Configuração do Teste</h3>
                <table>
                    <tr><th>Parâmetro</th><th>Valor</th></tr>
                    <tr><td>Usuários Concorrentes</td><td>{report_data['configuration']['concurrent_users']}</td></tr>
                    <tr><td>Requisições por Segundo</td><td>{report_data['configuration']['requests_per_second']}</td></tr>
                    <tr><td>Duração</td><td>{report_data['configuration']['duration_seconds']}s</td></tr>
                    <tr><td>Tempo de Ramp-up</td><td>{report_data['configuration']['ramp_up_time']}s</td></tr>
                    <tr><td>Tempo de Ramp-down</td><td>{report_data['configuration']['ramp_down_time']}s</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h3>Métricas de Performance</h3>
                <div class="metric">
                    <strong>Total de Requisições:</strong><br>
                    {report_data['metrics']['total_requests']}
                </div>
                <div class="metric">
                    <strong>Taxa de Erro:</strong><br>
                    {report_data['metrics']['error_rate']:.1f}%
                </div>
                <div class="metric">
                    <strong>Tempo de Resposta Médio:</strong><br>
                    {report_data['metrics']['avg_response_time']:.2f}s
                </div>
                <div class="metric">
                    <strong>Tempo de Resposta P95:</strong><br>
                    {report_data['metrics']['p95_response_time']:.2f}s
                </div>
                <div class="metric">
                    <strong>Throughput:</strong><br>
                    {report_data['metrics']['throughput']:.2f} req/s
                </div>
            </div>
            
            <div class="section">
                <h3>Uso de Recursos do Sistema</h3>
                <div class="metric">
                    <strong>CPU Médio:</strong><br>
                    {report_data['system_resources']['avg_cpu_usage']:.1f}%
                </div>
                <div class="metric">
                    <strong>CPU Máximo:</strong><br>
                    {report_data['system_resources']['max_cpu_usage']:.1f}%
                </div>
                <div class="metric">
                    <strong>Memória Média:</strong><br>
                    {report_data['system_resources']['avg_memory_usage']:.1f}%
                </div>
                <div class="metric">
                    <strong>Memória Máxima:</strong><br>
                    {report_data['system_resources']['max_memory_usage']:.1f}%
                </div>
            </div>
        """
        
        # Adiciona alertas se existirem
        if report_data['alerts']:
            html_template += """
            <div class="section">
                <h3>Alertas</h3>
            """
            for alert in report_data['alerts']:
                html_template += f"""
                <div class="alert {alert['level']}">
                    <strong>{alert['level'].upper()}:</strong> {alert['message']}
                </div>
                """
            html_template += "</div>"
        
        # Adiciona recomendações se existirem
        if report_data['recommendations']:
            html_template += """
            <div class="section">
                <h3>Recomendações</h3>
                <ul>
            """
            for rec in report_data['recommendations']:
                html_template += f"<li>{rec}</li>"
            html_template += "</ul></div>"
        
        html_template += """
        </body>
        </html>
        """
        
        return html_template

class ContinuousLoadTestingFramework:
    """Framework principal de testes de carga contínuos"""
    
    def __init__(self):
        self.test_analyzer = TestAnalyzer()
        self.test_history = []
        self.scheduled_tests = []
        self._running = False
        
        # Configurações padrão de testes
        self.default_test_configs = self._create_default_test_configs()
        
        logger.info("Continuous Load Testing Framework inicializado")
    
    def _create_default_test_configs(self) -> List[TestConfiguration]:
        """Cria configurações padrão de testes"""
        configs = []
        
        # Smoke Test
        configs.append(TestConfiguration(
            test_type=TestType.SMOKE_TEST,
            name="Smoke Test - Verificação Básica",
            description="Teste básico para verificar se o sistema está funcionando",
            duration_seconds=60,
            concurrent_users=1,
            requests_per_second=1.0,
            success_criteria={
                'max_error_rate': 5.0,
                'max_avg_response_time': 10.0
            }
        ))
        
        # Load Test
        configs.append(TestConfiguration(
            test_type=TestType.LOAD_TEST,
            name="Load Test - Carga Normal",
            description="Teste com carga normal esperada",
            duration_seconds=300,
            concurrent_users=10,
            requests_per_second=5.0,
            success_criteria={
                'max_error_rate': 5.0,
                'max_avg_response_time': 3.0,
                'min_throughput': 4.0
            }
        ))
        
        # Stress Test
        configs.append(TestConfiguration(
            test_type=TestType.STRESS_TEST,
            name="Stress Test - Carga Elevada",
            description="Teste com carga elevada para encontrar limites",
            duration_seconds=600,
            concurrent_users=50,
            requests_per_second=20.0,
            success_criteria={
                'max_error_rate': 10.0,
                'max_avg_response_time': 5.0,
                'min_throughput': 15.0
            }
        ))
        
        # Spike Test
        configs.append(TestConfiguration(
            test_type=TestType.SPIKE_TEST,
            name="Spike Test - Picos de Carga",
            description="Teste com picos súbitos de carga",
            duration_seconds=180,
            concurrent_users=100,
            requests_per_second=50.0,
            ramp_up_time=10,
            ramp_down_time=10,
            success_criteria={
                'max_error_rate': 15.0,
                'max_avg_response_time': 8.0
            }
        ))
        
        # Endurance Test
        configs.append(TestConfiguration(
            test_type=TestType.ENDURANCE_TEST,
            name="Endurance Test - Resistência",
            description="Teste de longa duração para verificar estabilidade",
            duration_seconds=1800,  # 30 minutos
            concurrent_users=20,
            requests_per_second=10.0,
            success_criteria={
                'max_error_rate': 5.0,
                'max_avg_response_time': 4.0,
                'min_throughput': 8.0
            }
        ))
        
        return configs
    
    async def run_test_suite(self, test_configs: Optional[List[TestConfiguration]] = None) -> List[TestResult]:
        """Executa suíte completa de testes"""
        if test_configs is None:
            test_configs = self.default_test_configs
        
        logger.info(f"Iniciando suíte de testes com {len(test_configs)} testes")
        
        results = []
        
        for config in test_configs:
            logger.info(f"Executando teste: {config.name}")
            
            try:
                # Executa teste
                result = await self.run_single_test(config)
                results.append(result)
                
                # Pausa entre testes para estabilização
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Erro no teste {config.name}: {e}")
                
                # Cria resultado de falha
                failed_result = TestResult(
                    test_id=f"{config.test_type.value}_{int(time.time())}",
                    configuration=config,
                    metrics=TestMetrics(
                        test_id=f"{config.test_type.value}_{int(time.time())}",
                        start_time=datetime.now()
                    ),
                    status=TestStatus.FAILED
                )
                failed_result.alerts.append({
                    'level': AlertLevel.CRITICAL.value,
                    'message': f"Teste falhou com erro: {str(e)}",
                    'metric': 'test_execution',
                    'value': 0,
                    'threshold': 1
                })
                
                results.append(failed_result)
        
        # Gera relatório consolidado
        self._generate_suite_report(results)
        
        logger.info(f"Suíte de testes concluída: {len(results)} testes executados")
        
        return results
    
    async def run_single_test(self, config: TestConfiguration) -> TestResult:
        """Executa um teste individual"""
        # Cria gerador de carga
        load_generator = LoadGenerator(config)
        
        # Executa teste
        metrics = await load_generator.start_test()
        
        # Cria resultado
        result = TestResult(
            test_id=metrics.test_id,
            configuration=config,
            metrics=metrics,
            status=TestStatus.COMPLETED
        )
        
        # Analisa resultado
        result = self.test_analyzer.analyze_test_result(result)
        
        # Adiciona ao histórico
        self.test_history.append(result)
        
        return result
    
    def _generate_suite_report(self, results: List[TestResult]):
        """Gera relatório consolidado da suíte"""
        suite_data = {
            'suite_info': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(results),
                'passed_tests': sum(1 for r in results if r.passed),
                'failed_tests': sum(1 for r in results if not r.passed)
            },
            'test_results': []
        }
        
        for result in results:
            suite_data['test_results'].append({
                'test_id': result.test_id,
                'test_name': result.configuration.name,
                'test_type': result.configuration.test_type.value,
                'status': result.status.value,
                'passed': result.passed,
                'duration': result.metrics.duration,
                'error_rate': result.metrics.error_rate,
                'avg_response_time': result.metrics.avg_response_time,
                'throughput': result.metrics.throughput,
                'alerts_count': len(result.alerts),
                'critical_alerts': len([a for a in result.alerts if a['level'] == AlertLevel.CRITICAL.value])
            })
        
        # Salva relatório da suíte
        suite_report_path = self.test_analyzer.reports_dir / f"test_suite_{int(time.time())}.json"
        with open(suite_report_path, 'w', encoding='utf-8') as f:
            json.dump(suite_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Relatório da suíte salvo: {suite_report_path}")
    
    def get_test_history_summary(self) -> Dict[str, Any]:
        """Retorna resumo do histórico de testes"""
        if not self.test_history:
            return {'message': 'Nenhum teste executado ainda'}
        
        total_tests = len(self.test_history)
        passed_tests = sum(1 for r in self.test_history if r.passed)
        
        # Agrupa por tipo de teste
        by_type = defaultdict(list)
        for result in self.test_history:
            by_type[result.configuration.test_type.value].append(result)
        
        type_summary = {}
        for test_type, results in by_type.items():
            type_summary[test_type] = {
                'total': len(results),
                'passed': sum(1 for r in results if r.passed),
                'avg_error_rate': statistics.mean([r.metrics.error_rate for r in results]),
                'avg_response_time': statistics.mean([r.metrics.avg_response_time for r in results]),
                'avg_throughput': statistics.mean([r.metrics.throughput for r in results])
            }
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests / total_tests) * 100,
            'by_type': type_summary,
            'latest_test': {
                'test_id': self.test_history[-1].test_id,
                'name': self.test_history[-1].configuration.name,
                'passed': self.test_history[-1].passed,
                'timestamp': self.test_history[-1].metrics.start_time.isoformat()
            }
        }

# Exemplo de uso e testes
async def test_continuous_load_testing():
    """Testa o framework de testes de carga contínuos"""
    framework = ContinuousLoadTestingFramework()
    
    print("\n=== FRAMEWORK DE TESTES DE CARGA CONTÍNUOS ===")
    
    # Executa apenas smoke test e load test para demonstração
    test_configs = [
        framework.default_test_configs[0],  # Smoke Test
        framework.default_test_configs[1]   # Load Test
    ]
    
    print(f"\nExecutando {len(test_configs)} testes...")
    
    results = await framework.run_test_suite(test_configs)
    
    print(f"\n=== RESULTADOS DOS TESTES ===")
    
    for result in results:
        print(f"\nTeste: {result.configuration.name}")
        print(f"Status: {'PASSOU' if result.passed else 'FALHOU'}")
        print(f"Duração: {result.metrics.duration:.1f}s")
        print(f"Total de requisições: {result.metrics.total_requests}")
        print(f"Taxa de erro: {result.metrics.error_rate:.1f}%")
        print(f"Tempo de resposta médio: {result.metrics.avg_response_time:.2f}s")
        print(f"Throughput: {result.metrics.throughput:.2f} req/s")
        print(f"Alertas: {len(result.alerts)}")
        
        if result.alerts:
            print("Alertas críticos:")
            for alert in result.alerts:
                if alert['level'] == AlertLevel.CRITICAL.value:
                    print(f"  - {alert['message']}")
    
    # Exibe resumo do histórico
    summary = framework.get_test_history_summary()
    print(f"\n=== RESUMO DO HISTÓRICO ===")
    print(f"Total de testes: {summary['total_tests']}")
    print(f"Testes aprovados: {summary['passed_tests']}")
    print(f"Taxa de sucesso: {summary['success_rate']:.1f}%")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(test_continuous_load_testing())
    print(f"\nFramework executado com sucesso: {len(results)} testes realizados")