import logging
import threading
import time
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import asyncio

# WebSocket removido - usando apenas dados diretos do MT5
from mt5_data_provider import MT5DataProvider
from mt5_connector import MT5Connector

class StreamingAdapter:
    """
    Adaptador para integrar o sistema de streaming de dados com o pipeline de ML existente.
    Permite que componentes do pipeline se inscrevam para receber atualizações de dados em tempo real.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765, 
                 mt5_data_provider: Optional[MT5DataProvider] = None):
        self.host = host
        self.port = port
        self.mt5_data_provider = mt5_data_provider
        # WebSocket removido - usando apenas dados diretos do MT5
        self.server_thread = None
        self.logger = self._setup_logger()
        self.running = False
        self.callbacks: Dict[str, List[Callable[[Dict], None]]] = {}
        self.metrics = {
            "messages_processed": 0,
            "updates_sent": 0,
            "errors": 0,
            "latency_ms": []
        }
        self.max_latency_samples = 1000
    
    def _setup_logger(self) -> logging.Logger:
        """Configura o sistema de logging"""
        logger = logging.getLogger('StreamingAdapter')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def start(self) -> bool:
        """Inicia o adaptador de streaming"""
        if self.running:
            self.logger.warning("Adaptador de streaming já está em execução")
            return True
            
        # Verificar se o provedor de dados MT5 está disponível
        if not self.mt5_data_provider:
            self.logger.error("Provedor de dados MT5 não disponível")
            return False
            
        try:
            # WebSocket removido - usando apenas dados diretos do MT5
            
            # Registrar callback no provedor de dados MT5 para receber atualizações
            self._register_mt5_callbacks()
            
            self.running = True
            self.logger.info(f"Adaptador de streaming iniciado em ws://{self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao iniciar adaptador de streaming: {e}")
            return False
    
    def stop(self) -> None:
        """Para o adaptador de streaming"""
        if not self.running:
            return
            
        try:
            # Cancelar callbacks do MT5
            self._unregister_mt5_callbacks()
            
            # WebSocket removido - usando apenas dados diretos do MT5
                
            self.running = False
            self.logger.info("Adaptador de streaming parado")
        except Exception as e:
            self.logger.error(f"Erro ao parar adaptador de streaming: {e}")
    
    def _register_mt5_callbacks(self) -> None:
        """Registra callbacks no provedor de dados MT5"""
        if not self.mt5_data_provider:
            return
            
        # Para cada símbolo que o MT5 está monitorando, registrar nosso callback
        for symbol in self.mt5_data_provider.price_callbacks.keys():
            self.mt5_data_provider.subscribe_to_symbol(symbol, self._on_price_update)
            self.logger.info(f"Registrado callback para {symbol} no MT5")
    
    def _unregister_mt5_callbacks(self) -> None:
        """Cancela callbacks no provedor de dados MT5"""
        if not self.mt5_data_provider:
            return
            
        # Para cada símbolo que o MT5 está monitorando, cancelar nosso callback
        for symbol in list(self.mt5_data_provider.price_callbacks.keys()):
            self.mt5_data_provider.unsubscribe_from_symbol(symbol, self._on_price_update)
            self.logger.info(f"Cancelado callback para {symbol} no MT5")
    
    def _on_price_update(self, price_data: Dict) -> None:
        """Callback para atualizações de preço do MT5"""
        try:
            start_time = time.time()
            symbol = price_data.get("symbol")
            
            if not symbol:
                return
                
            # Processar dados e chamar callbacks registrados
            self._process_price_update(symbol, price_data)
            
            # Calcular latência
            latency_ms = (time.time() - start_time) * 1000
            self._update_metrics(latency_ms)
            
            self.metrics["messages_processed"] += 1
        except Exception as e:
            self.logger.error(f"Erro ao processar atualização de preço: {e}")
            self.metrics["errors"] += 1
    
    def _process_price_update(self, symbol: str, price_data: Dict) -> None:
        """Processa atualizações de preço e notifica callbacks registrados"""
        # Verificar se há callbacks registrados para este símbolo
        if symbol in self.callbacks and self.callbacks[symbol]:
            for callback in self.callbacks[symbol]:
                try:
                    callback(price_data)
                    self.metrics["updates_sent"] += 1
                except Exception as e:
                    self.logger.error(f"Erro no callback para {symbol}: {e}")
                    self.metrics["errors"] += 1
    
    def _update_metrics(self, latency_ms: float) -> None:
        """Atualiza métricas de desempenho"""
        self.metrics["latency_ms"].append(latency_ms)
        
        # Manter apenas as últimas N amostras de latência
        if len(self.metrics["latency_ms"]) > self.max_latency_samples:
            self.metrics["latency_ms"] = self.metrics["latency_ms"][-self.max_latency_samples:]
    
    def subscribe(self, symbol: str, callback: Callable[[Dict], None]) -> bool:
        """Inscreve um callback para receber atualizações de um símbolo"""
        if not self.running:
            self.logger.error("Adaptador de streaming não está em execução")
            return False
            
        # Inicializar lista de callbacks se necessário
        if symbol not in self.callbacks:
            self.callbacks[symbol] = []
            
        # Adicionar callback se ainda não estiver registrado
        if callback not in self.callbacks[symbol]:
            self.callbacks[symbol].append(callback)
            
            # Se este é o primeiro callback para este símbolo, inscrever no MT5
            if len(self.callbacks[symbol]) == 1 and self.mt5_data_provider:
                self.mt5_data_provider.subscribe_to_symbol(symbol, self._on_price_update)
                
            self.logger.info(f"Callback registrado para {symbol}")
            return True
            
        return False
    
    def unsubscribe(self, symbol: str, callback: Callable[[Dict], None]) -> bool:
        """Cancela inscrição de um callback para um símbolo"""
        if symbol in self.callbacks and callback in self.callbacks[symbol]:
            self.callbacks[symbol].remove(callback)
            
            # Se não há mais callbacks para este símbolo, cancelar inscrição no MT5
            if not self.callbacks[symbol] and self.mt5_data_provider:
                self.mt5_data_provider.unsubscribe_from_symbol(symbol, self._on_price_update)
                
            # Remover símbolo se não há mais callbacks
            if not self.callbacks[symbol]:
                del self.callbacks[symbol]
                
            self.logger.info(f"Callback removido para {symbol}")
            return True
            
        return False
    
    def get_metrics(self) -> Dict:
        """Retorna métricas de desempenho do adaptador"""
        metrics = self.metrics.copy()
        
        # Calcular estatísticas de latência
        latency_samples = metrics["latency_ms"]
        if latency_samples:
            metrics["avg_latency_ms"] = sum(latency_samples) / len(latency_samples)
            metrics["min_latency_ms"] = min(latency_samples)
            metrics["max_latency_ms"] = max(latency_samples)
            
            # Calcular percentis (50%, 95%, 99%)
            sorted_latency = sorted(latency_samples)
            metrics["p50_latency_ms"] = sorted_latency[int(len(sorted_latency) * 0.5)]
            metrics["p95_latency_ms"] = sorted_latency[int(len(sorted_latency) * 0.95)]
            metrics["p99_latency_ms"] = sorted_latency[int(len(sorted_latency) * 0.99)]
        else:
            metrics["avg_latency_ms"] = 0
            metrics["min_latency_ms"] = 0
            metrics["max_latency_ms"] = 0
            metrics["p50_latency_ms"] = 0
            metrics["p95_latency_ms"] = 0
            metrics["p99_latency_ms"] = 0
            
        # Remover lista completa de latências para não sobrecarregar a resposta
        metrics.pop("latency_ms")
        
        # Adicionar informações do provedor de dados se disponível
        if hasattr(self, 'data_provider') and self.data_provider:
            metrics["data_provider_status"] = "active"
            metrics["last_update"] = getattr(self.data_provider, 'last_update', None)
            
        return metrics
    
    def get_status(self) -> Dict:
        """Retorna status do adaptador de streaming"""
        status = {
            "running": self.running,
            "server_address": f"ws://{self.host}:{self.port}" if self.running else None,
            "mt5_connected": False,
            "active_symbols": list(self.callbacks.keys()),
            "metrics": self.get_metrics(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Verificar conexão com MT5
        if self.mt5_data_provider and self.mt5_data_provider.connector:
            status["mt5_connected"] = self.mt5_data_provider.connector.connected
            
        return status

# Exemplo de uso
if __name__ == "__main__":
    # Configurar MT5
    connector = MT5Connector()
    if connector.connect():
        # Inicializar provedor de dados
        data_provider = MT5DataProvider(connector)
        
        # Função de callback para processar atualizações de preço
        def price_callback(price_data):
            print(f"Adaptador recebeu atualização: {price_data}")
        
        # Inicializar e iniciar adaptador de streaming
        adapter = StreamingAdapter(host="0.0.0.0", port=8765, mt5_data_provider=data_provider)
        if adapter.start():
            # Inscrever-se para receber atualizações de um símbolo
            adapter.subscribe("EURUSD", price_callback)
            
            try:
                print("Adaptador de streaming iniciado. Pressione Ctrl+C para parar...")
                
                # Exibir métricas periodicamente
                while True:
                    time.sleep(10)
                    metrics = adapter.get_metrics()
                    print(f"Métricas: {json.dumps(metrics, indent=2)}")
            except KeyboardInterrupt:
                print("Parando adaptador...")
                adapter.unsubscribe("EURUSD", price_callback)
                adapter.stop()
                connector.disconnect()
        else:
            print("Falha ao iniciar adaptador de streaming")
    else:
        print("Falha ao conectar ao MT5")