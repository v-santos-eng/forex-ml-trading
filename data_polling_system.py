#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Polling para Dados MT5
Atualização periódica de cotações, posições e saldo sem WebSocket
Autor: Sistema Forex ML
Data: 2024
"""

import threading
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import queue
import json
from enum import Enum

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataType(Enum):
    """Tipos de dados para polling"""
    TICKS = "ticks"
    ACCOUNT = "account"
    POSITIONS = "positions"
    ORDERS = "orders"
    SYMBOLS = "symbols"
    RATES = "rates"
    MARKET_DEPTH = "market_depth"

class PollingMode(Enum):
    """Modos de polling"""
    FAST = "fast"          # 100ms - Para ticks críticos
    NORMAL = "normal"      # 500ms - Para dados gerais
    SLOW = "slow"          # 2000ms - Para dados menos críticos
    CUSTOM = "custom"      # Intervalo personalizado

@dataclass
class PollingConfig:
    """Configuração de polling para um tipo de dados"""
    data_type: DataType
    interval_ms: int
    enabled: bool = True
    symbols: List[str] = field(default_factory=list)
    callback: Optional[Callable] = None
    max_history: int = 1000
    priority: int = 1  # 1=alta, 2=média, 3=baixa

@dataclass
class DataUpdate:
    """Estrutura para atualizações de dados"""
    data_type: DataType
    timestamp: datetime
    data: Any
    symbol: Optional[str] = None
    change_detected: bool = False

class DataCache:
    """
    Cache otimizado para dados de mercado
    Armazena apenas mudanças para reduzir uso de memória
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.data = {}
        self.history = {}
        self.last_update = {}
        self.change_detection = {}
        self._lock = threading.RLock()
    
    def update(self, key: str, data: Any, detect_changes: bool = True) -> bool:
        """Atualiza dados no cache e detecta mudanças"""
        with self._lock:
            current_time = datetime.now()
            
            # Detectar mudanças
            has_changed = True
            if detect_changes and key in self.data:
                has_changed = self._detect_change(key, data)
            
            # Atualizar dados
            self.data[key] = data
            self.last_update[key] = current_time
            
            # Manter histórico se houve mudança
            if has_changed:
                if key not in self.history:
                    self.history[key] = deque(maxlen=self.max_size)
                
                self.history[key].append({
                    'timestamp': current_time,
                    'data': data
                })
            
            return has_changed
    
    def get(self, key: str, default=None):
        """Obtém dados do cache"""
        with self._lock:
            return self.data.get(key, default)
    
    def get_history(self, key: str, limit: int = None) -> List:
        """Obtém histórico de um item"""
        with self._lock:
            history = self.history.get(key, [])
            if limit:
                return list(history)[-limit:]
            return list(history)
    
    def get_last_update(self, key: str) -> Optional[datetime]:
        """Obtém timestamp da última atualização"""
        with self._lock:
            return self.last_update.get(key)
    
    def _detect_change(self, key: str, new_data: Any) -> bool:
        """Detecta se houve mudança nos dados"""
        old_data = self.data.get(key)
        
        if old_data is None:
            return True
        
        # Para dados numéricos simples
        if isinstance(new_data, (int, float)):
            return abs(new_data - old_data) > 1e-10
        
        # Para strings
        if isinstance(new_data, str):
            return new_data != old_data
        
        # Para listas/dicts (comparação por hash)
        try:
            return hash(str(new_data)) != hash(str(old_data))
        except:
            return str(new_data) != str(old_data)
    
    def clear_old_data(self, max_age_hours: int = 24):
        """Remove dados antigos do cache"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            keys_to_remove = []
            for key, last_update in self.last_update.items():
                if last_update < cutoff_time:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.data.pop(key, None)
                self.history.pop(key, None)
                self.last_update.pop(key, None)
            
            logger.info(f"Removidos {len(keys_to_remove)} itens antigos do cache")

class PollingWorker(threading.Thread):
    """
    Worker thread para polling de dados específicos
    Otimizado para baixa latência e alta eficiência
    """
    
    def __init__(self, config: PollingConfig, mt5_client, data_cache: DataCache, 
                 event_queue: queue.Queue):
        super().__init__(daemon=True)
        self.config = config
        self.mt5_client = mt5_client
        self.data_cache = data_cache
        self.event_queue = event_queue
        self.running = False
        self.paused = False
        self.error_count = 0
        self.max_errors = 10
        self.last_successful_poll = datetime.now()
        
        # Otimizações de performance
        self.sleep_interval = self.config.interval_ms / 1000.0
        self.adaptive_interval = self.sleep_interval
        self.performance_stats = {
            'polls': 0,
            'changes': 0,
            'errors': 0,
            'avg_duration': 0.0
        }
    
    def run(self):
        """Loop principal do worker"""
        self.running = True
        logger.info(f"Iniciando polling worker para {self.config.data_type.value}")
        
        while self.running:
            try:
                if not self.paused and self.config.enabled:
                    start_time = time.time()
                    
                    # Executar polling baseado no tipo de dados
                    success = self._poll_data()
                    
                    # Atualizar estatísticas
                    duration = time.time() - start_time
                    self._update_stats(duration, success)
                    
                    # Ajustar intervalo adaptativamente
                    self._adjust_interval(duration, success)
                    
                    if success:
                        self.error_count = 0
                        self.last_successful_poll = datetime.now()
                    else:
                        self.error_count += 1
                        
                        # Parar se muitos erros
                        if self.error_count >= self.max_errors:
                            logger.error(f"Muitos erros no polling {self.config.data_type.value}, pausando")
                            self.paused = True
                
                # Sleep adaptativo
                time.sleep(self.adaptive_interval)
                
            except Exception as e:
                logger.error(f"Erro no polling worker {self.config.data_type.value}: {e}")
                self.error_count += 1
                time.sleep(self.sleep_interval * 2)  # Sleep maior em caso de erro
        
        logger.info(f"Polling worker {self.config.data_type.value} finalizado")
    
    def _poll_data(self) -> bool:
        """Executa polling baseado no tipo de dados"""
        try:
            if self.config.data_type == DataType.TICKS:
                return self._poll_ticks()
            elif self.config.data_type == DataType.ACCOUNT:
                return self._poll_account()
            elif self.config.data_type == DataType.POSITIONS:
                return self._poll_positions()
            elif self.config.data_type == DataType.ORDERS:
                return self._poll_orders()
            elif self.config.data_type == DataType.SYMBOLS:
                return self._poll_symbols()
            elif self.config.data_type == DataType.RATES:
                return self._poll_rates()
            elif self.config.data_type == DataType.MARKET_DEPTH:
                return self._poll_market_depth()
            else:
                logger.warning(f"Tipo de dados não suportado: {self.config.data_type}")
                return False
                
        except Exception as e:
            logger.error(f"Erro no polling de {self.config.data_type.value}: {e}")
            return False
    
    def _poll_ticks(self) -> bool:
        """Polling de ticks em tempo real"""
        success_count = 0
        
        for symbol in self.config.symbols:
            try:
                # Obter último tick
                tick = self.mt5_client.get_symbol_tick(symbol)
                
                if tick:
                    cache_key = f"tick_{symbol}"
                    has_changed = self.data_cache.update(cache_key, tick, detect_changes=True)
                    
                    if has_changed:
                        # Enviar evento de mudança
                        update = DataUpdate(
                            data_type=DataType.TICKS,
                            timestamp=datetime.now(),
                            data=tick,
                            symbol=symbol,
                            change_detected=True
                        )
                        
                        try:
                            self.event_queue.put_nowait(update)
                        except queue.Full:
                            pass  # Ignorar se fila estiver cheia
                        
                        # Callback se configurado
                        if self.config.callback:
                            try:
                                self.config.callback(update)
                            except Exception as e:
                                logger.error(f"Erro no callback de tick: {e}")
                    
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Erro ao obter tick de {symbol}: {e}")
        
        return success_count > 0
    
    def _poll_account(self) -> bool:
        """Polling de informações da conta"""
        try:
            account_info = self.mt5_client.get_account_info()
            
            if account_info:
                cache_key = "account_info"
                has_changed = self.data_cache.update(cache_key, account_info, detect_changes=True)
                
                if has_changed:
                    update = DataUpdate(
                        data_type=DataType.ACCOUNT,
                        timestamp=datetime.now(),
                        data=account_info,
                        change_detected=True
                    )
                    
                    try:
                        self.event_queue.put_nowait(update)
                    except queue.Full:
                        pass
                    
                    if self.config.callback:
                        try:
                            self.config.callback(update)
                        except Exception as e:
                            logger.error(f"Erro no callback de conta: {e}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erro ao obter informações da conta: {e}")
            return False
    
    def _poll_positions(self) -> bool:
        """Polling de posições abertas"""
        try:
            positions = self.mt5_client.get_positions()
            
            cache_key = "positions"
            has_changed = self.data_cache.update(cache_key, positions, detect_changes=True)
            
            if has_changed:
                update = DataUpdate(
                    data_type=DataType.POSITIONS,
                    timestamp=datetime.now(),
                    data=positions,
                    change_detected=True
                )
                
                try:
                    self.event_queue.put_nowait(update)
                except queue.Full:
                    pass
                
                if self.config.callback:
                    try:
                        self.config.callback(update)
                    except Exception as e:
                        logger.error(f"Erro no callback de posições: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao obter posições: {e}")
            return False
    
    def _poll_orders(self) -> bool:
        """Polling de ordens pendentes"""
        try:
            orders = self.mt5_client.get_orders()
            
            cache_key = "orders"
            has_changed = self.data_cache.update(cache_key, orders, detect_changes=True)
            
            if has_changed:
                update = DataUpdate(
                    data_type=DataType.ORDERS,
                    timestamp=datetime.now(),
                    data=orders,
                    change_detected=True
                )
                
                try:
                    self.event_queue.put_nowait(update)
                except queue.Full:
                    pass
                
                if self.config.callback:
                    try:
                        self.config.callback(update)
                    except Exception as e:
                        logger.error(f"Erro no callback de ordens: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao obter ordens: {e}")
            return False
    
    def _poll_symbols(self) -> bool:
        """Polling de informações de símbolos"""
        success_count = 0
        
        for symbol in self.config.symbols:
            try:
                symbol_info = self.mt5_client.get_symbol_info(symbol)
                
                if symbol_info:
                    cache_key = f"symbol_{symbol}"
                    has_changed = self.data_cache.update(cache_key, symbol_info, detect_changes=True)
                    
                    if has_changed:
                        update = DataUpdate(
                            data_type=DataType.SYMBOLS,
                            timestamp=datetime.now(),
                            data=symbol_info,
                            symbol=symbol,
                            change_detected=True
                        )
                        
                        try:
                            self.event_queue.put_nowait(update)
                        except queue.Full:
                            pass
                        
                        if self.config.callback:
                            try:
                                self.config.callback(update)
                            except Exception as e:
                                logger.error(f"Erro no callback de símbolo: {e}")
                    
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Erro ao obter informações de {symbol}: {e}")
        
        return success_count > 0
    
    def _poll_rates(self) -> bool:
        """Polling de dados de preços (rates)"""
        success_count = 0
        
        for symbol in self.config.symbols:
            try:
                # Obter últimas 10 barras M1
                rates = self.mt5_client.get_rates(symbol, 16385, 10)  # M1
                
                if rates is not None and len(rates) > 0:
                    cache_key = f"rates_{symbol}"
                    has_changed = self.data_cache.update(cache_key, rates, detect_changes=True)
                    
                    if has_changed:
                        update = DataUpdate(
                            data_type=DataType.RATES,
                            timestamp=datetime.now(),
                            data=rates,
                            symbol=symbol,
                            change_detected=True
                        )
                        
                        try:
                            self.event_queue.put_nowait(update)
                        except queue.Full:
                            pass
                        
                        if self.config.callback:
                            try:
                                self.config.callback(update)
                            except Exception as e:
                                logger.error(f"Erro no callback de rates: {e}")
                    
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Erro ao obter rates de {symbol}: {e}")
        
        return success_count > 0
    
    def _poll_market_depth(self) -> bool:
        """Polling de profundidade de mercado"""
        success_count = 0
        
        for symbol in self.config.symbols:
            try:
                # Obter book de ofertas
                depth = self.mt5_client.get_market_book(symbol)
                
                if depth:
                    cache_key = f"depth_{symbol}"
                    has_changed = self.data_cache.update(cache_key, depth, detect_changes=True)
                    
                    if has_changed:
                        update = DataUpdate(
                            data_type=DataType.MARKET_DEPTH,
                            timestamp=datetime.now(),
                            data=depth,
                            symbol=symbol,
                            change_detected=True
                        )
                        
                        try:
                            self.event_queue.put_nowait(update)
                        except queue.Full:
                            pass
                        
                        if self.config.callback:
                            try:
                                self.config.callback(update)
                            except Exception as e:
                                logger.error(f"Erro no callback de depth: {e}")
                    
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Erro ao obter depth de {symbol}: {e}")
        
        return success_count > 0
    
    def _update_stats(self, duration: float, success: bool):
        """Atualiza estatísticas de performance"""
        self.performance_stats['polls'] += 1
        
        if success:
            # Atualizar média de duração
            current_avg = self.performance_stats['avg_duration']
            polls = self.performance_stats['polls']
            self.performance_stats['avg_duration'] = (current_avg * (polls - 1) + duration) / polls
        else:
            self.performance_stats['errors'] += 1
    
    def _adjust_interval(self, duration: float, success: bool):
        """Ajusta intervalo de polling adaptativamente"""
        if not success:
            # Aumentar intervalo em caso de erro
            self.adaptive_interval = min(self.adaptive_interval * 1.5, self.sleep_interval * 5)
        else:
            # Se polling está muito rápido (duração > 50% do intervalo), aumentar intervalo
            if duration > self.sleep_interval * 0.5:
                self.adaptive_interval = min(self.adaptive_interval * 1.1, self.sleep_interval * 2)
            # Se polling está eficiente, voltar ao intervalo normal
            elif duration < self.sleep_interval * 0.1:
                self.adaptive_interval = max(self.adaptive_interval * 0.95, self.sleep_interval)
    
    def stop(self):
        """Para o worker"""
        self.running = False
    
    def pause(self):
        """Pausa o worker"""
        self.paused = True
    
    def resume(self):
        """Resume o worker"""
        self.paused = False
        self.error_count = 0
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do worker"""
        return {
            'data_type': self.config.data_type.value,
            'running': self.running,
            'paused': self.paused,
            'error_count': self.error_count,
            'last_successful_poll': self.last_successful_poll.isoformat(),
            'adaptive_interval': self.adaptive_interval,
            'performance': self.performance_stats.copy()
        }

class DataPollingSystem:
    """
    Sistema principal de polling de dados
    Gerencia múltiplos workers e otimiza performance
    """
    
    def __init__(self, mt5_client):
        self.mt5_client = mt5_client
        self.data_cache = DataCache()
        self.event_queue = queue.Queue(maxsize=10000)
        self.workers = {}
        self.event_handlers = {}
        self.running = False
        
        # Thread para processar eventos
        self.event_processor = None
        
        # Configurações padrão
        self.default_configs = {
            DataType.TICKS: PollingConfig(
                data_type=DataType.TICKS,
                interval_ms=100,  # 100ms para ticks
                priority=1
            ),
            DataType.ACCOUNT: PollingConfig(
                data_type=DataType.ACCOUNT,
                interval_ms=1000,  # 1s para conta
                priority=2
            ),
            DataType.POSITIONS: PollingConfig(
                data_type=DataType.POSITIONS,
                interval_ms=500,  # 500ms para posições
                priority=1
            ),
            DataType.ORDERS: PollingConfig(
                data_type=DataType.ORDERS,
                interval_ms=500,  # 500ms para ordens
                priority=1
            ),
            DataType.SYMBOLS: PollingConfig(
                data_type=DataType.SYMBOLS,
                interval_ms=2000,  # 2s para símbolos
                priority=3
            ),
            DataType.RATES: PollingConfig(
                data_type=DataType.RATES,
                interval_ms=1000,  # 1s para rates
                priority=2
            ),
            DataType.MARKET_DEPTH: PollingConfig(
                data_type=DataType.MARKET_DEPTH,
                interval_ms=200,  # 200ms para depth
                priority=2
            )
        }
    
    def start(self, symbols: List[str] = None):
        """Inicia o sistema de polling"""
        if self.running:
            logger.warning("Sistema de polling já está rodando")
            return
        
        self.running = True
        
        # Símbolos padrão se não especificados
        if symbols is None:
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
        
        # Iniciar workers com configurações padrão
        self._start_default_workers(symbols)
        
        # Iniciar processador de eventos
        self._start_event_processor()
        
        logger.info(f"Sistema de polling iniciado com {len(self.workers)} workers")
    
    def stop(self):
        """Para o sistema de polling"""
        if not self.running:
            return
        
        self.running = False
        
        # Parar todos os workers
        for worker in self.workers.values():
            worker.stop()
        
        # Aguardar finalização
        for worker in self.workers.values():
            if worker.is_alive():
                worker.join(timeout=2.0)
        
        # Parar processador de eventos
        if self.event_processor and self.event_processor.is_alive():
            self.event_processor.join(timeout=2.0)
        
        self.workers.clear()
        logger.info("Sistema de polling parado")
    
    def add_polling_config(self, config: PollingConfig):
        """Adiciona nova configuração de polling"""
        if config.data_type in self.workers:
            logger.warning(f"Worker para {config.data_type.value} já existe, parando o anterior")
            self.workers[config.data_type].stop()
            if self.workers[config.data_type].is_alive():
                self.workers[config.data_type].join(timeout=2.0)
        
        # Criar e iniciar novo worker
        worker = PollingWorker(config, self.mt5_client, self.data_cache, self.event_queue)
        worker.start()
        self.workers[config.data_type] = worker
        
        logger.info(f"Worker adicionado para {config.data_type.value}")
    
    def remove_polling_config(self, data_type: DataType):
        """Remove configuração de polling"""
        if data_type in self.workers:
            self.workers[data_type].stop()
            if self.workers[data_type].is_alive():
                self.workers[data_type].join(timeout=2.0)
            del self.workers[data_type]
            logger.info(f"Worker removido para {data_type.value}")
    
    def add_event_handler(self, data_type: DataType, handler: Callable):
        """Adiciona handler para eventos de um tipo de dados"""
        if data_type not in self.event_handlers:
            self.event_handlers[data_type] = []
        self.event_handlers[data_type].append(handler)
    
    def remove_event_handler(self, data_type: DataType, handler: Callable):
        """Remove handler de eventos"""
        if data_type in self.event_handlers:
            try:
                self.event_handlers[data_type].remove(handler)
            except ValueError:
                pass
    
    def get_latest_data(self, data_type: DataType, symbol: str = None) -> Any:
        """Obtém dados mais recentes do cache"""
        if symbol:
            cache_key = f"{data_type.value}_{symbol}"
        else:
            cache_key = data_type.value
        
        return self.data_cache.get(cache_key)
    
    def get_data_history(self, data_type: DataType, symbol: str = None, limit: int = None) -> List:
        """Obtém histórico de dados"""
        if symbol:
            cache_key = f"{data_type.value}_{symbol}"
        else:
            cache_key = data_type.value
        
        return self.data_cache.get_history(cache_key, limit)
    
    def get_system_stats(self) -> Dict:
        """Retorna estatísticas do sistema"""
        stats = {
            'running': self.running,
            'workers': {},
            'cache_size': len(self.data_cache.data),
            'event_queue_size': self.event_queue.qsize(),
            'event_handlers': {dt.value: len(handlers) for dt, handlers in self.event_handlers.items()}
        }
        
        for data_type, worker in self.workers.items():
            stats['workers'][data_type.value] = worker.get_stats()
        
        return stats
    
    def _start_default_workers(self, symbols: List[str]):
        """Inicia workers com configurações padrão"""
        # Workers que precisam de símbolos
        symbol_dependent = [DataType.TICKS, DataType.SYMBOLS, DataType.RATES, DataType.MARKET_DEPTH]
        
        for data_type, config in self.default_configs.items():
            if data_type in symbol_dependent:
                config.symbols = symbols.copy()
            
            worker = PollingWorker(config, self.mt5_client, self.data_cache, self.event_queue)
            worker.start()
            self.workers[data_type] = worker
    
    def _start_event_processor(self):
        """Inicia thread para processar eventos"""
        def process_events():
            while self.running:
                try:
                    # Aguardar evento com timeout
                    update = self.event_queue.get(timeout=1.0)
                    
                    # Processar handlers para este tipo de dados
                    handlers = self.event_handlers.get(update.data_type, [])
                    
                    for handler in handlers:
                        try:
                            handler(update)
                        except Exception as e:
                            logger.error(f"Erro no handler de evento: {e}")
                    
                    self.event_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Erro no processador de eventos: {e}")
        
        self.event_processor = threading.Thread(target=process_events, daemon=True)
        self.event_processor.start()
    
    def pause_worker(self, data_type: DataType):
        """Pausa worker específico"""
        if data_type in self.workers:
            self.workers[data_type].pause()
    
    def resume_worker(self, data_type: DataType):
        """Resume worker específico"""
        if data_type in self.workers:
            self.workers[data_type].resume()
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Limpa dados antigos do cache"""
        self.data_cache.clear_old_data(max_age_hours)

# Exemplo de uso
if __name__ == "__main__":
    # Este seria usado com um cliente MT5 real
    print("Data Polling System - Sistema de Polling para MT5")
    print("Funcionalidades implementadas:")
    print("- Polling otimizado de ticks em tempo real")
    print("- Monitoramento de conta, posições e ordens")
    print("- Cache inteligente com detecção de mudanças")
    print("- Sistema de eventos assíncronos")
    print("- Intervalos adaptativos para otimização")
    print("- Múltiplos workers com prioridades")
    print("- Estatísticas de performance")
    print("- Baixa latência sem WebSocket")