#!/usr/bin/env python3
"""
Sistema de Monitoramento de Saúde dos Processos
Monitora continuamente a saúde do pipeline ML e alerta sobre anomalias
"""

import psutil
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable
import smtplib
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    # Fallback for Python 3.13+ compatibility
    MimeText = None
    MimeMultipart = None

@dataclass
class HealthMetrics:
    """Métricas de saúde do processo"""
    timestamp: str
    pid: int
    cpu_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    handles: int
    threads: int
    open_files: int
    connections: int
    is_responsive: bool
    uptime_seconds: float

@dataclass
class HealthThresholds:
    """Limites para alertas de saúde"""
    max_cpu_percent: float = 80.0
    max_memory_mb: float = 1000.0
    max_handles: int = 1000
    max_threads: int = 50
    max_open_files: int = 100
    handle_growth_rate_per_hour: int = 100
    memory_growth_rate_mb_per_hour: float = 100.0
    min_response_time_seconds: float = 30.0

class ProcessHealthMonitor:
    """Monitor de saúde de processos"""
    
    def __init__(self, config_file: str = "health_monitor_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.thresholds = HealthThresholds(**self.config.get('thresholds', {}))
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Estado do monitor
        self.monitored_processes: Dict[int, Dict] = {}
        self.health_history: Dict[int, List[HealthMetrics]] = {}
        self.alerts_sent: Dict[str, datetime] = {}
        self.is_running = False
        self.monitor_thread = None
        
        # Callbacks para alertas
        self.alert_callbacks: List[Callable] = []
        
    def _load_config(self) -> Dict:
        """Carrega configuração do arquivo"""
        config_path = Path(self.config_file)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Configuração padrão
            default_config = {
                'thresholds': asdict(HealthThresholds()),
                'monitoring': {
                    'interval_seconds': 30,
                    'history_retention_hours': 24,
                    'alert_cooldown_minutes': 15
                },
                'alerts': {
                    'email_enabled': False,
                    'email_smtp_server': 'smtp.gmail.com',
                    'email_smtp_port': 587,
                    'email_from': '',
                    'email_password': '',
                    'email_to': [],
                    'log_enabled': True,
                    'console_enabled': True
                },
                'processes_to_monitor': []
            }
            
            # Salvar configuração padrão
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Configura logging"""
        logger = logging.getLogger('ProcessHealthMonitor')
        logger.setLevel(logging.INFO)
        
        # Handler para arquivo
        log_file = Path('logs/process_health_monitor.log')
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def add_process(self, pid: int, name: str = None, critical: bool = True):
        """Adiciona processo para monitoramento"""
        try:
            process = psutil.Process(pid)
            process_name = name or process.name()
            
            self.monitored_processes[pid] = {
                'name': process_name,
                'critical': critical,
                'start_time': datetime.now(),
                'last_seen': datetime.now(),
                'process': process
            }
            
            self.health_history[pid] = []
            
            self.logger.info(f"Processo adicionado ao monitoramento: {process_name} (PID: {pid})")
            
        except psutil.NoSuchProcess:
            self.logger.error(f"Processo {pid} não encontrado")
            raise
    
    def remove_process(self, pid: int):
        """Remove processo do monitoramento"""
        if pid in self.monitored_processes:
            process_name = self.monitored_processes[pid]['name']
            del self.monitored_processes[pid]
            del self.health_history[pid]
            self.logger.info(f"Processo removido do monitoramento: {process_name} (PID: {pid})")
    
    def _collect_metrics(self, pid: int) -> Optional[HealthMetrics]:
        """Coleta métricas de um processo"""
        try:
            process_info = self.monitored_processes[pid]
            process = process_info['process']
            
            # Verificar se processo ainda existe
            if not process.is_running():
                return None
            
            # Coletar métricas básicas
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            
            # Handles (Windows específico)
            try:
                handles = process.num_handles()
            except AttributeError:
                handles = 0
            
            # Arquivos abertos
            try:
                open_files = len(process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = 0
            
            # Conexões de rede
            try:
                connections = len(process.net_connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                connections = 0
            
            # Threads
            try:
                threads = process.num_threads()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                threads = 0
            
            # Tempo de atividade
            start_time = process_info['start_time']
            uptime = (datetime.now() - start_time).total_seconds()
            
            # Teste de responsividade (simplificado)
            is_responsive = cpu_percent < 100.0  # Se CPU não está em 100%
            
            metrics = HealthMetrics(
                timestamp=datetime.now().isoformat(),
                pid=pid,
                cpu_percent=cpu_percent,
                memory_rss_mb=memory_info.rss / 1024 / 1024,
                memory_vms_mb=memory_info.vms / 1024 / 1024,
                handles=handles,
                threads=threads,
                open_files=open_files,
                connections=connections,
                is_responsive=is_responsive,
                uptime_seconds=uptime
            )
            
            return metrics
            
        except psutil.NoSuchProcess:
            self.logger.warning(f"Processo {pid} não existe mais")
            return None
        except Exception as e:
            self.logger.error(f"Erro coletando métricas do processo {pid}: {e}")
            return None
    
    def _analyze_health(self, pid: int, metrics: HealthMetrics) -> List[str]:
        """Analisa métricas e retorna lista de alertas"""
        alerts = []
        
        # Verificar limites absolutos
        if metrics.cpu_percent > self.thresholds.max_cpu_percent:
            alerts.append(f"CPU alta: {metrics.cpu_percent:.1f}% (limite: {self.thresholds.max_cpu_percent}%)")
        
        if metrics.memory_rss_mb > self.thresholds.max_memory_mb:
            alerts.append(f"Memória alta: {metrics.memory_rss_mb:.1f}MB (limite: {self.thresholds.max_memory_mb}MB)")
        
        if metrics.handles > self.thresholds.max_handles:
            alerts.append(f"Handles altos: {metrics.handles} (limite: {self.thresholds.max_handles})")
        
        if metrics.threads > self.thresholds.max_threads:
            alerts.append(f"Threads altas: {metrics.threads} (limite: {self.thresholds.max_threads})")
        
        if metrics.open_files > self.thresholds.max_open_files:
            alerts.append(f"Arquivos abertos: {metrics.open_files} (limite: {self.thresholds.max_open_files})")
        
        if not metrics.is_responsive:
            alerts.append("Processo não responsivo")
        
        # Verificar tendências (se há histórico suficiente)
        history = self.health_history[pid]
        if len(history) >= 10:  # Pelo menos 10 medições
            # Análise de crescimento de handles
            recent_metrics = history[-10:]
            oldest_handles = recent_metrics[0].handles
            newest_handles = metrics.handles
            
            time_diff_hours = (datetime.fromisoformat(metrics.timestamp) - 
                             datetime.fromisoformat(recent_metrics[0].timestamp)).total_seconds() / 3600
            
            if time_diff_hours > 0:
                handle_growth_rate = (newest_handles - oldest_handles) / time_diff_hours
                
                if handle_growth_rate > self.thresholds.handle_growth_rate_per_hour:
                    alerts.append(f"Crescimento rápido de handles: {handle_growth_rate:.1f}/hora (limite: {self.thresholds.handle_growth_rate_per_hour}/hora)")
                
                # Análise de crescimento de memória
                oldest_memory = recent_metrics[0].memory_rss_mb
                newest_memory = metrics.memory_rss_mb
                memory_growth_rate = (newest_memory - oldest_memory) / time_diff_hours
                
                if memory_growth_rate > self.thresholds.memory_growth_rate_mb_per_hour:
                    alerts.append(f"Crescimento rápido de memória: {memory_growth_rate:.1f}MB/hora (limite: {self.thresholds.memory_growth_rate_mb_per_hour}MB/hora)")
        
        return alerts
    
    def _send_alert(self, pid: int, process_name: str, alerts: List[str]):
        """Envia alertas"""
        alert_key = f"{pid}_{hash('_'.join(alerts))}"
        
        # Verificar cooldown
        cooldown_minutes = self.config['monitoring']['alert_cooldown_minutes']
        if alert_key in self.alerts_sent:
            time_since_last = datetime.now() - self.alerts_sent[alert_key]
            if time_since_last < timedelta(minutes=cooldown_minutes):
                return  # Ainda em cooldown
        
        self.alerts_sent[alert_key] = datetime.now()
        
        alert_message = f"ALERTA - Processo {process_name} (PID: {pid}):\n"
        for alert in alerts:
            alert_message += f"  • {alert}\n"
        
        # Log
        if self.config['alerts']['log_enabled']:
            self.logger.warning(alert_message.strip())
        
        # Console
        if self.config['alerts']['console_enabled']:
            print(f"🚨 {alert_message}")
        
        # Email (se configurado)
        if self.config['alerts']['email_enabled']:
            self._send_email_alert(process_name, pid, alerts)
        
        # Callbacks customizados
        for callback in self.alert_callbacks:
            try:
                callback(pid, process_name, alerts)
            except Exception as e:
                self.logger.error(f"Erro executando callback de alerta: {e}")
    
    def _send_email_alert(self, process_name: str, pid: int, alerts: List[str]):
        """Envia alerta por email"""
        if not MimeText or not MimeMultipart:
            self.logger.warning("Funcionalidade de email não disponível (classes MIME não importadas)")
            return
            
        try:
            email_config = self.config['alerts']
            
            msg = MimeMultipart()
            msg['From'] = email_config['email_from']
            msg['To'] = ', '.join(email_config['email_to'])
            msg['Subject'] = f"Alerta de Saúde - {process_name} (PID: {pid})"
            
            body = f"Alertas detectados no processo {process_name} (PID: {pid}):\n\n"
            for alert in alerts:
                body += f"• {alert}\n"
            
            body += f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['email_smtp_server'], email_config['email_smtp_port'])
            server.starttls()
            server.login(email_config['email_from'], email_config['email_password'])
            
            text = msg.as_string()
            server.sendmail(email_config['email_from'], email_config['email_to'], text)
            server.quit()
            
            self.logger.info(f"Email de alerta enviado para {email_config['email_to']}")
            
        except Exception as e:
            self.logger.error(f"Erro enviando email de alerta: {e}")
    
    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        interval = self.config['monitoring']['interval_seconds']
        retention_hours = self.config['monitoring']['history_retention_hours']
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Monitorar cada processo
                for pid in list(self.monitored_processes.keys()):
                    metrics = self._collect_metrics(pid)
                    
                    if metrics is None:
                        # Processo não existe mais
                        process_name = self.monitored_processes[pid]['name']
                        is_critical = self.monitored_processes[pid]['critical']
                        
                        if is_critical:
                            self._send_alert(pid, process_name, ["Processo crítico terminou inesperadamente"])
                        
                        self.remove_process(pid)
                        continue
                    
                    # Adicionar ao histórico
                    self.health_history[pid].append(metrics)
                    
                    # Limpar histórico antigo
                    cutoff_time = current_time - timedelta(hours=retention_hours)
                    self.health_history[pid] = [
                        m for m in self.health_history[pid]
                        if datetime.fromisoformat(m.timestamp) > cutoff_time
                    ]
                    
                    # Analisar saúde
                    alerts = self._analyze_health(pid, metrics)
                    
                    if alerts:
                        process_name = self.monitored_processes[pid]['name']
                        self._send_alert(pid, process_name, alerts)
                    
                    # Atualizar última visualização
                    self.monitored_processes[pid]['last_seen'] = current_time
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de monitoramento: {e}")
                time.sleep(interval)
    
    def start_monitoring(self):
        """Inicia monitoramento"""
        if self.is_running:
            self.logger.warning("Monitoramento já está em execução")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Monitoramento de saúde iniciado")
    
    def stop_monitoring(self):
        """Para monitoramento"""
        self.is_running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Monitoramento de saúde parado")
    
    def get_health_report(self, pid: int = None) -> Dict:
        """Gera relatório de saúde"""
        if pid:
            pids = [pid] if pid in self.monitored_processes else []
        else:
            pids = list(self.monitored_processes.keys())
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'processes': {}
        }
        
        for pid in pids:
            process_info = self.monitored_processes[pid]
            history = self.health_history[pid]
            
            if history:
                latest_metrics = history[-1]
                
                report['processes'][pid] = {
                    'name': process_info['name'],
                    'critical': process_info['critical'],
                    'uptime_hours': (datetime.now() - process_info['start_time']).total_seconds() / 3600,
                    'latest_metrics': asdict(latest_metrics),
                    'history_count': len(history),
                    'alerts': self._analyze_health(pid, latest_metrics)
                }
        
        return report
    
    def add_alert_callback(self, callback: Callable):
        """Adiciona callback customizado para alertas"""
        self.alert_callbacks.append(callback)

def main():
    """Função principal para execução standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor de Saúde de Processos')
    parser.add_argument('--config', default='health_monitor_config.json', help='Arquivo de configuração')
    parser.add_argument('--pid', type=int, action='append', help='PID para monitorar (pode ser usado múltiplas vezes)')
    parser.add_argument('--name', action='append', help='Nome do processo (correspondente aos PIDs)')
    parser.add_argument('--critical', action='store_true', help='Marcar processos como críticos')
    parser.add_argument('--duration', type=int, default=0, help='Duração do monitoramento em minutos (0 = infinito)')
    
    args = parser.parse_args()
    
    # Criar monitor
    monitor = ProcessHealthMonitor(args.config)
    
    # Adicionar processos
    if args.pid:
        names = args.name or [None] * len(args.pid)
        for i, pid in enumerate(args.pid):
            name = names[i] if i < len(names) else None
            try:
                monitor.add_process(pid, name, args.critical)
            except psutil.NoSuchProcess:
                print(f"❌ Processo {pid} não encontrado")
                return 1
    
    if not monitor.monitored_processes:
        print("❌ Nenhum processo válido para monitorar")
        return 1
    
    # Iniciar monitoramento
    monitor.start_monitoring()
    
    try:
        if args.duration > 0:
            print(f"🔍 Monitorando por {args.duration} minutos...")
            time.sleep(args.duration * 60)
        else:
            print("🔍 Monitoramento iniciado. Pressione Ctrl+C para parar...")
            while True:
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n⏹️  Parando monitoramento...")
    
    finally:
        monitor.stop_monitoring()
        
        # Relatório final
        report = monitor.get_health_report()
        print("\n📊 RELATÓRIO FINAL:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())