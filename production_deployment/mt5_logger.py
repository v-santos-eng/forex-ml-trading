import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import json
import traceback
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque
import MetaTrader5 as mt5

class LogLevel(Enum):
    """Níveis de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class EventType(Enum):
    """Tipos de eventos do sistema"""
    CONNECTION = "CONNECTION"
    TRADE = "TRADE"
    SIGNAL = "SIGNAL"
    RISK = "RISK"
    DATA = "DATA"
    SYSTEM = "SYSTEM"
    ERROR = "ERROR"

@dataclass
class LogEntry:
    """Entrada de log estruturada"""
    timestamp: datetime
    level: LogLevel
    event_type: EventType
    message: str
    component: str
    details: Dict[str, Any] = None
    exception: Optional[str] = None
    trade_id: Optional[int] = None
    symbol: Optional[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

@dataclass
class ErrorInfo:
    """Informações de erro"""
    error_code: int
    error_message: str
    function_name: str
    timestamp: datetime
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

class MT5Logger:
    """
    Sistema de logging avançado para operações MT5
    """
    
    def __init__(self, log_dir: str = "logs", max_log_files: int = 30, 
                 max_memory_entries: int = 1000):
        self.log_dir = Path(log_dir)
        self.max_log_files = max_log_files
        self.max_memory_entries = max_memory_entries
        
        # Criar diretório de logs
        self.log_dir.mkdir(exist_ok=True)
        
        # Configurar loggers
        self.loggers = {}
        self._setup_loggers()
        
        # Buffer de logs em memória
        self.memory_buffer = deque(maxlen=max_memory_entries)
        self.error_buffer = deque(maxlen=100)
        
        # Estatísticas
        self.stats = {
            'total_logs': 0,
            'errors_count': 0,
            'warnings_count': 0,
            'trades_logged': 0,
            'signals_logged': 0
        }
        
        # Callbacks para eventos críticos
        self.critical_callbacks = []
        self.error_callbacks = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Inicializar
        self._cleanup_old_logs()
        self.log(LogLevel.INFO, EventType.SYSTEM, "MT5Logger", "Sistema de logging inicializado")
    
    def _setup_loggers(self) -> None:
        """Configura os loggers por componente"""
        components = ['MT5Connector', 'MT5OrderManager', 'MT5DataProvider', 
                     'MT5RiskManager', 'MT5SignalIntegrator', 'ForexAnalyzer']
        
        for component in components:
            logger = logging.getLogger(component)
            logger.setLevel(logging.DEBUG)
            
            # Remover handlers existentes
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Handler para arquivo
            log_file = self.log_dir / f"{component.lower()}_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # Handler para console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            # Evitar propagação para root logger
            logger.propagate = False
            
            self.loggers[component] = logger
    
    def _cleanup_old_logs(self) -> None:
        """Remove logs antigos"""
        try:
            log_files = list(self.log_dir.glob('*.log'))
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Manter apenas os arquivos mais recentes
            for old_file in log_files[self.max_log_files:]:
                old_file.unlink()
                
        except Exception as e:
            print(f"Erro ao limpar logs antigos: {e}")
    
    def add_critical_callback(self, callback: Callable[[LogEntry], None]) -> None:
        """Adiciona callback para eventos críticos"""
        self.critical_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[ErrorInfo], None]) -> None:
        """Adiciona callback para erros"""
        self.error_callbacks.append(callback)
    
    def log(self, level: LogLevel, event_type: EventType, component: str, 
            message: str, details: Optional[Dict] = None, exception: Optional[Exception] = None,
            trade_id: Optional[int] = None, symbol: Optional[str] = None) -> None:
        """Log principal do sistema"""
        try:
            with self.lock:
                # Criar entrada de log
                entry = LogEntry(
                    timestamp=datetime.now(),
                    level=level,
                    event_type=event_type,
                    message=message,
                    component=component,
                    details=details or {},
                    exception=str(exception) if exception else None,
                    trade_id=trade_id,
                    symbol=symbol
                )
                
                # Adicionar ao buffer de memória
                self.memory_buffer.append(entry)
                
                # Atualizar estatísticas
                self.stats['total_logs'] += 1
                if level == LogLevel.ERROR:
                    self.stats['errors_count'] += 1
                elif level == LogLevel.WARNING:
                    self.stats['warnings_count'] += 1
                
                if event_type == EventType.TRADE:
                    self.stats['trades_logged'] += 1
                elif event_type == EventType.SIGNAL:
                    self.stats['signals_logged'] += 1
                
                # Log usando logger apropriado
                logger = self.loggers.get(component, logging.getLogger(component))
                
                log_message = message
                if details:
                    log_message += f" | Detalhes: {json.dumps(details, default=str, ensure_ascii=False)}"
                if symbol:
                    log_message += f" | Símbolo: {symbol}"
                if trade_id:
                    log_message += f" | Trade ID: {trade_id}"
                
                # Fazer log baseado no nível
                if level == LogLevel.DEBUG:
                    logger.debug(log_message)
                elif level == LogLevel.INFO:
                    logger.info(log_message)
                elif level == LogLevel.WARNING:
                    logger.warning(log_message)
                elif level == LogLevel.ERROR:
                    logger.error(log_message)
                    if exception:
                        logger.error(f"Exception: {traceback.format_exc()}")
                elif level == LogLevel.CRITICAL:
                    logger.critical(log_message)
                    if exception:
                        logger.critical(f"Exception: {traceback.format_exc()}")
                
                # Notificar callbacks para eventos críticos
                if level == LogLevel.CRITICAL:
                    for callback in self.critical_callbacks:
                        try:
                            callback(entry)
                        except Exception as e:
                            print(f"Erro no callback crítico: {e}")
                
        except Exception as e:
            print(f"Erro no sistema de logging: {e}")
    
    def log_mt5_error(self, function_name: str, context: Optional[Dict] = None) -> ErrorInfo:
        """Log específico para erros do MT5"""
        error_code = mt5.last_error()
        error_message = self._get_mt5_error_message(error_code)
        
        error_info = ErrorInfo(
            error_code=error_code,
            error_message=error_message,
            function_name=function_name,
            timestamp=datetime.now(),
            context=context or {}
        )
        
        # Adicionar ao buffer de erros
        with self.lock:
            self.error_buffer.append(error_info)
        
        # Log do erro
        self.log(
            LogLevel.ERROR,
            EventType.ERROR,
            "MT5",
            f"Erro MT5 em {function_name}: [{error_code}] {error_message}",
            details=context
        )
        
        # Notificar callbacks de erro
        for callback in self.error_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                print(f"Erro no callback de erro: {e}")
        
        return error_info
    
    def _get_mt5_error_message(self, error_code: int) -> str:
        """Obtém mensagem de erro MT5"""
        error_messages = {
            0: "Sucesso",
            1: "Erro genérico",
            2: "Parâmetro inválido",
            3: "Sem memória",
            4: "Função não suportada",
            5: "Versão inválida",
            6: "Autorização falhada",
            7: "Conexão falhada",
            8: "Muitas requisições",
            9: "Operação rejeitada",
            10: "Apenas para investidores",
            11: "Conexão cancelada",
            12: "Símbolo não encontrado",
            13: "Símbolo não selecionado",
            14: "Dados não encontrados",
            15: "Erro de rede",
            16: "Timeout",
            17: "Dados inválidos",
            18: "Conta não encontrada",
            19: "Trade desabilitado",
            20: "Mercado fechado",
            21: "Volume inválido",
            22: "Preço inválido",
            23: "Stop inválido",
            24: "Ordem não encontrada",
            25: "Posição não encontrada"
        }
        
        return error_messages.get(error_code, f"Erro desconhecido: {error_code}")
    
    def log_trade(self, action: str, symbol: str, volume: float, price: float,
                  trade_id: Optional[int] = None, result: Optional[Dict] = None) -> None:
        """Log específico para trades"""
        details = {
            'action': action,
            'volume': volume,
            'price': price
        }
        
        if result:
            details.update(result)
        
        self.log(
            LogLevel.INFO,
            EventType.TRADE,
            "MT5OrderManager",
            f"Trade {action}: {symbol} {volume} lotes @ {price}",
            details=details,
            trade_id=trade_id,
            symbol=symbol
        )
    
    def log_signal(self, symbol: str, action: str, confidence: float, 
                   indicators: Optional[Dict] = None) -> None:
        """Log específico para sinais"""
        details = {
            'action': action,
            'confidence': confidence
        }
        
        if indicators:
            details['indicators'] = indicators
        
        self.log(
            LogLevel.INFO,
            EventType.SIGNAL,
            "MT5SignalIntegrator",
            f"Sinal gerado: {symbol} {action} (confiança: {confidence:.2%})",
            details=details,
            symbol=symbol
        )
    
    def log_risk_alert(self, level: str, message: str, symbol: Optional[str] = None,
                       details: Optional[Dict] = None) -> None:
        """Log específico para alertas de risco"""
        log_level = LogLevel.WARNING if level in ['LOW', 'MEDIUM'] else LogLevel.ERROR
        
        self.log(
            log_level,
            EventType.RISK,
            "MT5RiskManager",
            f"Alerta de risco {level}: {message}",
            details=details,
            symbol=symbol
        )
    
    def get_recent_logs(self, minutes: int = 60, level: Optional[LogLevel] = None,
                       event_type: Optional[EventType] = None) -> List[LogEntry]:
        """Obtém logs recentes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            filtered_logs = []
            for entry in self.memory_buffer:
                if entry.timestamp >= cutoff_time:
                    if level is None or entry.level == level:
                        if event_type is None or entry.event_type == event_type:
                            filtered_logs.append(entry)
            
            return sorted(filtered_logs, key=lambda x: x.timestamp, reverse=True)
    
    def get_error_summary(self, hours: int = 24) -> Dict:
        """Obtém resumo de erros"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_errors = [e for e in self.error_buffer if e.timestamp >= cutoff_time]
            
            error_counts = {}
            for error in recent_errors:
                key = f"{error.error_code}: {error.error_message}"
                error_counts[key] = error_counts.get(key, 0) + 1
            
            return {
                'total_errors': len(recent_errors),
                'unique_errors': len(error_counts),
                'error_breakdown': error_counts,
                'most_recent': recent_errors[-1] if recent_errors else None
            }
    
    def get_stats(self) -> Dict:
        """Obtém estatísticas do logger"""
        with self.lock:
            return {
                **self.stats,
                'memory_buffer_size': len(self.memory_buffer),
                'error_buffer_size': len(self.error_buffer),
                'active_loggers': len(self.loggers)
            }
    
    def export_logs(self, start_date: datetime, end_date: datetime, 
                   output_file: str, format: str = 'json') -> bool:
        """Exporta logs para arquivo"""
        try:
            with self.lock:
                filtered_logs = [
                    entry for entry in self.memory_buffer
                    if start_date <= entry.timestamp <= end_date
                ]
            
            if format.lower() == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(
                        [asdict(entry) for entry in filtered_logs],
                        f,
                        default=str,
                        indent=2,
                        ensure_ascii=False
                    )
            elif format.lower() == 'csv':
                import pandas as pd
                df = pd.DataFrame([asdict(entry) for entry in filtered_logs])
                df.to_csv(output_file, index=False, encoding='utf-8')
            
            self.log(
                LogLevel.INFO,
                EventType.SYSTEM,
                "MT5Logger",
                f"Logs exportados para {output_file}",
                details={'count': len(filtered_logs), 'format': format}
            )
            
            return True
            
        except Exception as e:
            self.log(
                LogLevel.ERROR,
                EventType.SYSTEM,
                "MT5Logger",
                f"Erro ao exportar logs: {e}",
                exception=e
            )
            return False
    
    def cleanup(self) -> None:
        """Limpeza do logger"""
        try:
            # Fechar todos os handlers
            for logger in self.loggers.values():
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
            
            self.log(
                LogLevel.INFO,
                EventType.SYSTEM,
                "MT5Logger",
                "Sistema de logging finalizado"
            )
            
        except Exception as e:
            print(f"Erro na limpeza do logger: {e}")

class MT5ErrorHandler:
    """
    Manipulador de erros específico para MT5
    """
    
    def __init__(self, logger: MT5Logger):
        self.logger = logger
        self.retry_counts = {}
        self.max_retries = 3
        self.retry_delay = 1.0
    
    def handle_error(self, function_name: str, error_code: int, 
                    context: Optional[Dict] = None) -> bool:
        """Manipula erro MT5 e determina se deve tentar novamente"""
        # Log do erro
        error_info = self.logger.log_mt5_error(function_name, context)
        
        # Determinar se deve tentar novamente
        retry_key = f"{function_name}_{error_code}"
        
        # Erros que não devem ser repetidos
        no_retry_errors = [2, 6, 10, 12, 19, 20]  # Parâmetro inválido, auth failed, etc.
        
        if error_code in no_retry_errors:
            return False
        
        # Verificar contador de tentativas
        current_retries = self.retry_counts.get(retry_key, 0)
        if current_retries >= self.max_retries:
            self.retry_counts[retry_key] = 0  # Reset counter
            return False
        
        # Incrementar contador e permitir retry
        self.retry_counts[retry_key] = current_retries + 1
        
        # Aguardar antes de tentar novamente
        import time
        time.sleep(self.retry_delay * current_retries)
        
        return True
    
    def reset_retry_count(self, function_name: str, error_code: int) -> None:
        """Reset contador de tentativas após sucesso"""
        retry_key = f"{function_name}_{error_code}"
        self.retry_counts.pop(retry_key, None)

# Instância global do logger
_global_logger = None

def get_logger() -> MT5Logger:
    """Obtém instância global do logger"""
    global _global_logger
    if _global_logger is None:
        _global_logger = MT5Logger()
    return _global_logger

def setup_global_logger(log_dir: str = "logs", max_log_files: int = 30) -> MT5Logger:
    """Configura logger global"""
    global _global_logger
    _global_logger = MT5Logger(log_dir, max_log_files)
    return _global_logger

if __name__ == "__main__":
    # Teste do sistema de logging
    logger = MT5Logger("test_logs")
    
    # Testar diferentes tipos de log
    logger.log(LogLevel.INFO, EventType.SYSTEM, "Test", "Sistema iniciado")
    logger.log(LogLevel.WARNING, EventType.RISK, "Test", "Alerta de teste", 
              details={'risk_level': 'medium'})
    logger.log(LogLevel.ERROR, EventType.ERROR, "Test", "Erro de teste", 
              exception=Exception("Erro simulado"))
    
    # Testar logs específicos
    logger.log_trade("BUY", "EURUSD", 0.1, 1.1234, trade_id=12345)
    logger.log_signal("GBPUSD", "SELL", 0.85, {'rsi': 75, 'macd': -0.001})
    logger.log_risk_alert("HIGH", "Drawdown elevado", "USDJPY")
    
    # Testar estatísticas
    stats = logger.get_stats()
    print(f"📊 Estatísticas do logger: {stats}")
    
    # Testar logs recentes
    recent_logs = logger.get_recent_logs(minutes=5)
    print(f"📝 {len(recent_logs)} logs recentes")
    
    # Limpeza
    logger.cleanup()
    print("✅ Teste do logger concluído")