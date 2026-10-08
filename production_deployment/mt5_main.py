#!/usr/bin/env python3
"""
Sistema Principal de Trading MT5
Integração completa do sistema de análise Forex com MetaTrader 5

Este módulo serve como ponto de entrada principal para o sistema de trading automatizado,
coordinando todos os componentes e fornecendo diferentes modos de operação.
"""

import sys
import os
import argparse
import time
import signal
from datetime import datetime
from typing import Optional, Dict, Any
import threading

# Adicionar diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar componentes do sistema
from mt5_connector import MT5Connector
from mt5_order_manager import MT5OrderManager
from mt5_data_provider import MT5DataProvider
from mt5_risk_manager import MT5RiskManager, RiskLevel
from mt5_signal_integrator import MT5SignalIntegrator, SignalConfig
from mt5_logger import MT5Logger, LogLevel, EventType
from mt5_dashboard import MT5Dashboard
from forex_analyzer import ForexAnalyzer

class MT5TradingSystem:
    """
    Sistema principal de trading MT5
    Coordena todos os componentes e gerencia o ciclo de vida do sistema
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = MT5Logger()
        
        # Componentes principais
        self.connector: Optional[MT5Connector] = None
        self.order_manager: Optional[MT5OrderManager] = None
        self.data_provider: Optional[MT5DataProvider] = None
        self.risk_manager: Optional[MT5RiskManager] = None
        self.signal_integrator: Optional[MT5SignalIntegrator] = None
        self.forex_analyzer: Optional[ForexAnalyzer] = None
        
        # Estado do sistema
        self.running = False
        self.connected = False
        self.auto_trading_enabled = False
        
        # Threads
        self.main_thread: Optional[threading.Thread] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Configurar handlers de sinal
        self._setup_signal_handlers()
        
        self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", "Sistema inicializado")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Retorna configuração padrão do sistema"""
        return {
            'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD'],
            'timeframe': 'M15',
            'analysis_interval': 300,  # 5 minutos
            'risk_level': RiskLevel.MEDIUM,
            'max_risk_per_trade': 0.02,  # 2%
            'max_daily_loss': 0.05,  # 5%
            'max_drawdown': 0.10,  # 10%
            'enable_auto_trading': False,
            'enable_notifications': True,
            'log_level': LogLevel.INFO
        }
    
    def _setup_signal_handlers(self):
        """Configura handlers para sinais do sistema"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handler para sinais de interrupção"""
        self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", 
                       f"Sinal {signum} recebido, iniciando shutdown")
        self.shutdown()
    
    def initialize(self) -> bool:
        """Inicializa todos os componentes do sistema"""
        try:
            self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", "Inicializando componentes")
            
            # 1. Conectar ao MT5
            self.connector = MT5Connector()
            if not self.connector.connect():
                self.logger.log(LogLevel.ERROR, EventType.CONNECTION, "MT5System", 
                               "Falha ao conectar com MT5")
                return False
            
            self.connected = True
            self.logger.log(LogLevel.INFO, EventType.CONNECTION, "MT5System", "Conectado ao MT5")
            
            # 2. Inicializar gerenciador de ordens
            self.order_manager = MT5OrderManager(self.connector)
            self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", "Order Manager inicializado")
            
            # 3. Inicializar provedor de dados
            self.data_provider = MT5DataProvider(self.connector)
            self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", "Data Provider inicializado")
            
            # 4. Inicializar gerenciador de risco
            self.risk_manager = MT5RiskManager(
                self.connector, 
                self.order_manager, 
                self.data_provider
            )
            
            # Configurar parâmetros de risco diretamente
            self.risk_manager.risk_params.max_risk_per_trade = self.config['max_risk_per_trade']
            self.risk_manager.risk_params.max_daily_loss = self.config['max_daily_loss']
            self.risk_manager.risk_params.max_drawdown = self.config['max_drawdown']
            
            self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", "Risk Manager inicializado")
            
            # 5. Inicializar analisador Forex
            self.forex_analyzer = ForexAnalyzer()
            self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", "Forex Analyzer inicializado")
            
            # 6. Inicializar integrador de sinais
            signal_config = SignalConfig()
            signal_config.enable_auto_trading = self.config['enable_auto_trading']
            signal_config.symbols_enabled = self.config['symbols']
            
            self.signal_integrator = MT5SignalIntegrator(
                self.connector,
                self.order_manager,
                self.data_provider,
                self.risk_manager,
                signal_config
            )
            
            self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", "Signal Integrator inicializado")
            
            # 7. Iniciar monitoramento de risco
            self.risk_manager.start_monitoring()
            self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", "Monitoramento de risco iniciado")
            
            self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", 
                           "Todos os componentes inicializados com sucesso")
            return True
            
        except Exception as e:
            self.logger.log(LogLevel.ERROR, EventType.SYSTEM, "MT5System", 
                           f"Erro na inicialização: {e}", exception=e)
            return False
    
    def start_auto_trading(self) -> bool:
        """Inicia o trading automatizado"""
        if not self.connected:
            self.logger.log(LogLevel.ERROR, EventType.SYSTEM, "MT5System", 
                           "Sistema não conectado")
            return False
        
        try:
            self.auto_trading_enabled = True
            self.signal_integrator.config.enable_auto_trading = True
            
            # Iniciar loop de análise
            self.signal_integrator.start_analysis_loop()
            
            self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", 
                           "Trading automatizado iniciado")
            return True
            
        except Exception as e:
            self.logger.log(LogLevel.ERROR, EventType.SYSTEM, "MT5System", 
                           f"Erro ao iniciar auto trading: {e}", exception=e)
            return False
    
    def stop_auto_trading(self):
        """Para o trading automatizado"""
        try:
            self.auto_trading_enabled = False
            
            if self.signal_integrator:
                self.signal_integrator.config.enable_auto_trading = False
                self.signal_integrator.stop_analysis_loop()
            
            self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", 
                           "Trading automatizado parado")
            
        except Exception as e:
            self.logger.log(LogLevel.ERROR, EventType.SYSTEM, "MT5System", 
                           f"Erro ao parar auto trading: {e}", exception=e)
    
    def run_analysis(self, symbols: Optional[list] = None) -> Dict[str, Any]:
        """Executa análise manual dos símbolos"""
        if not self.signal_integrator:
            self.logger.log(LogLevel.ERROR, EventType.SYSTEM, "MT5System", 
                           "Sistema não inicializado")
            return {}
        
        try:
            symbols = symbols or self.config['symbols']
            
            self.logger.log(LogLevel.INFO, EventType.ANALYSIS, "MT5System", 
                           f"Iniciando análise manual para {len(symbols)} símbolos")
            
            # Executar análise
            signals = self.signal_integrator.analyze_market(symbols)
            
            # Preparar resultado
            result = {
                'timestamp': datetime.now(),
                'symbols_analyzed': len(symbols),
                'signals_generated': len(signals),
                'signals': []
            }
            
            for signal in signals:
                result['signals'].append({
                    'symbol': signal.symbol,
                    'action': signal.action.value,
                    'confidence': signal.confidence,
                    'indicators': signal.indicators,
                    'timestamp': signal.timestamp
                })
            
            self.logger.log(LogLevel.INFO, EventType.ANALYSIS, "MT5System", 
                           f"Análise concluída: {len(signals)} sinais gerados")
            
            return result
            
        except Exception as e:
            self.logger.log(LogLevel.ERROR, EventType.ANALYSIS, "MT5System", 
                           f"Erro na análise: {e}", exception=e)
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retorna status completo do sistema"""
        try:
            status = {
                'timestamp': datetime.now(),
                'connected': self.connected,
                'auto_trading': self.auto_trading_enabled,
                'running': self.running
            }
            
            if self.connected and self.order_manager:
                # Informações da conta
                account_summary = self.order_manager.get_account_summary()
                status['account'] = account_summary
                
                # Posições abertas
                positions = self.order_manager.get_positions()
                status['open_positions'] = len(positions)
                
                # Informações de risco
                if self.risk_manager:
                    risk_summary = self.risk_manager.get_risk_summary()
                    status['risk'] = risk_summary
                
                # Performance dos sinais
                if self.signal_integrator:
                    performance = self.signal_integrator.get_performance_summary()
                    status['performance'] = performance
            
            return status
            
        except Exception as e:
            self.logger.log(LogLevel.ERROR, EventType.SYSTEM, "MT5System", 
                           f"Erro ao obter status: {e}", exception=e)
            return {'error': str(e)}
    
    def run_console_mode(self):
        """Executa o sistema em modo console"""
        self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", "Iniciando modo console")
        
        if not self.initialize():
            print("Erro: Falha na inicialização do sistema")
            return
        
        self.running = True
        
        try:
            print("\n" + "="*60)
            print("         SISTEMA DE TRADING MT5 - MODO CONSOLE")
            print("="*60)
            print("\nComandos disponíveis:")
            print("  start    - Iniciar trading automatizado")
            print("  stop     - Parar trading automatizado")
            print("  analyze  - Executar análise manual")
            print("  status   - Mostrar status do sistema")
            print("  positions- Mostrar posições abertas")
            print("  risk     - Mostrar informações de risco")
            print("  help     - Mostrar esta ajuda")
            print("  quit     - Sair do sistema")
            print("\n" + "-"*60)
            
            while self.running:
                try:
                    command = input("\nMT5> ").strip().lower()
                    
                    if command == 'start':
                        if self.start_auto_trading():
                            print("✓ Trading automatizado iniciado")
                        else:
                            print("✗ Erro ao iniciar trading automatizado")
                    
                    elif command == 'stop':
                        self.stop_auto_trading()
                        print("✓ Trading automatizado parado")
                    
                    elif command == 'analyze':
                        print("Executando análise...")
                        result = self.run_analysis()
                        if result:
                            print(f"✓ Análise concluída: {result['signals_generated']} sinais gerados")
                            for signal in result['signals']:
                                print(f"  {signal['symbol']}: {signal['action']} (Confiança: {signal['confidence']:.1%})")
                        else:
                            print("✗ Erro na análise")
                    
                    elif command == 'status':
                        status = self.get_system_status()
                        print("\nStatus do Sistema:")
                        print(f"  Conectado: {'✓' if status['connected'] else '✗'}")
                        print(f"  Auto Trading: {'✓' if status['auto_trading'] else '✗'}")
                        if 'account' in status:
                            acc = status['account']
                            print(f"  Saldo: ${acc.get('balance', 0):,.2f}")
                            print(f"  Equity: ${acc.get('equity', 0):,.2f}")
                        print(f"  Posições Abertas: {status.get('open_positions', 0)}")
                    
                    elif command == 'positions':
                        if self.order_manager:
                            positions = self.order_manager.get_positions()
                            if positions:
                                print(f"\nPosições Abertas ({len(positions)}):")
                                for pos in positions:
                                    print(f"  {pos['symbol']}: {pos['type']} {pos['volume']} @ {pos['price_open']} (Lucro: ${pos['profit']:.2f})")
                            else:
                                print("\nNenhuma posição aberta")
                        else:
                            print("Sistema não inicializado")
                    
                    elif command == 'risk':
                        if self.risk_manager:
                            risk_summary = self.risk_manager.get_risk_summary()
                            print("\nInformações de Risco:")
                            print(f"  Nível de Risco: {risk_summary.get('risk_level', 'N/A')}")
                            print(f"  Drawdown: {risk_summary.get('drawdown', 0):.2%}")
                            print(f"  Perda Diária: {risk_summary.get('daily_loss', 0):.2%}")
                        else:
                            print("Sistema não inicializado")
                    
                    elif command == 'help':
                        print("\nComandos disponíveis:")
                        print("  start, stop, analyze, status, positions, risk, help, quit")
                    
                    elif command in ['quit', 'exit', 'q']:
                        print("Encerrando sistema...")
                        break
                    
                    elif command:
                        print(f"Comando desconhecido: {command}. Digite 'help' para ver os comandos disponíveis.")
                
                except KeyboardInterrupt:
                    print("\nEncerrando sistema...")
                    break
                except EOFError:
                    break
        
        finally:
            self.shutdown()
    
    def run_gui_mode(self):
        """Executa o sistema em modo GUI (Dashboard)"""
        self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", "Iniciando modo GUI")
        
        try:
            dashboard = MT5Dashboard()
            dashboard.run()
        except Exception as e:
            self.logger.log(LogLevel.ERROR, EventType.SYSTEM, "MT5System", 
                           f"Erro no modo GUI: {e}", exception=e)
    
    def shutdown(self):
        """Encerra o sistema de forma segura"""
        self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", "Iniciando shutdown")
        
        try:
            self.running = False
            
            # Parar trading automatizado
            if self.auto_trading_enabled:
                self.stop_auto_trading()
            
            # Parar monitoramento de risco
            if self.risk_manager:
                self.risk_manager.stop_monitoring()
            
            # Desconectar do MT5
            if self.connector and self.connected:
                self.connector.disconnect()
                self.connected = False
            
            # Cleanup do logger
            if self.logger:
                self.logger.cleanup()
            
            self.logger.log(LogLevel.INFO, EventType.SYSTEM, "MT5System", "Shutdown concluído")
            
        except Exception as e:
            print(f"Erro durante shutdown: {e}")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Sistema de Trading MT5')
    parser.add_argument('--mode', choices=['console', 'gui', 'analysis'], 
                       default='gui', help='Modo de execução')
    parser.add_argument('--symbols', nargs='+', 
                       default=['EURUSD', 'GBPUSD', 'USDJPY'],
                       help='Símbolos para análise')
    parser.add_argument('--auto-start', action='store_true',
                       help='Iniciar trading automatizado')
    parser.add_argument('--config', type=str,
                       help='Arquivo de configuração JSON')
    
    args = parser.parse_args()
    
    # Carregar configuração
    config = None
    if args.config and os.path.exists(args.config):
        try:
            import json
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
    
    # Criar sistema
    system = MT5TradingSystem(config)
    
    try:
        if args.mode == 'console':
            system.run_console_mode()
        
        elif args.mode == 'gui':
            system.run_gui_mode()
        
        elif args.mode == 'analysis':
            if system.initialize():
                print("Executando análise...")
                result = system.run_analysis(args.symbols)
                
                if result:
                    print(f"\nAnálise concluída em {result['timestamp']}")
                    print(f"Símbolos analisados: {result['symbols_analyzed']}")
                    print(f"Sinais gerados: {result['signals_generated']}")
                    
                    if result['signals']:
                        print("\nSinais:")
                        for signal in result['signals']:
                            print(f"  {signal['symbol']}: {signal['action']} (Confiança: {signal['confidence']:.1%})")
                    else:
                        print("\nNenhum sinal gerado")
                else:
                    print("Erro na análise")
            else:
                print("Erro na inicialização do sistema")
    
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    
    except Exception as e:
        print(f"Erro: {e}")
    
    finally:
        system.shutdown()

if __name__ == "__main__":
    main()