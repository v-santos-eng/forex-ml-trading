#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launcher Simplificado do Sistema MT5 Trading
Inicia o sistema sem GUI para operação direta
"""

import os
import sys
import time
import threading
from datetime import datetime
import signal

# Importar componentes do sistema
try:
    from mt5_integration import MT5Integration, MT5Config
    from risk_manager import RiskManager, RiskParameters, RiskLevel
    from data_polling_system import DataPollingSystem
    from ml_mt5_integration import MLTradingEngine
    from logging_system import LoggingSystem, LogLevel, EventType, get_logging_system
    from config_manager import ConfigManager
except ImportError as e:
    print(f"Erro ao importar componentes: {e}")
    sys.exit(1)

class SimpleMT5Launcher:
    """Launcher simplificado sem GUI"""
    
    def __init__(self):
        self.mt5_integration = None
        self.risk_manager = None
        self.data_polling = None
        self.ml_engine = None
        self.logging_system = None
        self.running = False
        
        # Configurar handler para CTRL+C
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handler para interrupção do sistema"""
        print("\nInterrupção detectada. Parando sistema...")
        self.shutdown()
    
    def start_system(self):
        """Inicia todos os componentes do sistema"""
        print("Iniciando Sistema MT5 Trading...")
        print("================================")
        
        try:
            # 1. Inicializar logging
            print("[1/5] Inicializando sistema de logging...")
            self.logging_system = get_logging_system()
            self.logging_system.log(LogLevel.INFO, EventType.SYSTEM_START,
                                  "Iniciando sistema MT5 Trading", "simple_launcher")
            
            # 2. Conectar MT5
            print("[2/5] Conectando ao MetaTrader 5...")
            
            # Criar configuração MT5 (usar conta já conectada)
            mt5_config = MT5Config(
                login=int(os.getenv("MT5_LOGIN", "0")),  # Conta já conectada
                password="",
                server="ActivTradesMarkets-Server",  # Servidor correto
                path=""
            )
            
            self.mt5_integration = MT5Integration(mt5_config)
            
            # Tentar conectar (modo demo se não houver credenciais)
            if self.mt5_integration.connect():
                print("✓ MT5 conectado com sucesso")
                account_info = self.mt5_integration.get_account_info()
                if account_info:
                    print(f"  Conta: {account_info.get('login', 'N/A')}")
                    print(f"  Saldo: ${account_info.get('balance', 0):,.2f}")
                    print(f"  Servidor: {account_info.get('server', 'N/A')}")
            else:
                print("✗ Falha na conexão MT5")
                return False
            
            # 3. Inicializar Risk Manager
            print("[3/5] Inicializando gerenciador de risco...")
            risk_params = RiskParameters(
                max_risk_per_trade=0.02,  # 2% por trade
                max_daily_risk=0.05,      # 5% perda diária
                max_positions=5,
                risk_level=RiskLevel.MODERATE
            )
            self.risk_manager = RiskManager(risk_params)
            print("✓ Risk Manager inicializado")
            
            # 4. Inicializar sistema de dados
            print("[4/5] Inicializando sistema de polling de dados...")
            self.data_polling = DataPollingSystem(self.mt5_integration)
            
            # Iniciar polling em thread separada
            polling_thread = threading.Thread(target=self.data_polling.start, daemon=True)
            polling_thread.start()
            print("✓ Sistema de dados iniciado")
            
            # 5. Inicializar ML Engine
            print("[5/5] Inicializando engine de Machine Learning...")
            try:
                self.ml_engine = MLTradingEngine(
                    mt5_integration=self.mt5_integration,
                    risk_manager=self.risk_manager
                )
                
                # Iniciar ML em thread separada
                ml_thread = threading.Thread(target=self.ml_engine.start, daemon=True)
                ml_thread.start()
                print("✓ ML Engine iniciado")
            except Exception as e:
                print(f"⚠ ML Engine não pôde ser iniciado: {e}")
                print("  Sistema continuará sem ML")
            
            self.running = True
            print("\n🚀 Sistema MT5 Trading iniciado com sucesso!")
            print("\nPressione Ctrl+C para parar o sistema")
            print("Status dos componentes:")
            print(f"  MT5: {'✓ Conectado' if self.mt5_integration.connected else '✗ Desconectado'}")
            print(f"  Risk Manager: ✓ Ativo")
            print(f"  Data Polling: ✓ Ativo")
            print(f"  ML Engine: {'✓ Ativo' if self.ml_engine else '⚠ Inativo'}")
            
            # Loop principal
            self._main_loop()
            
        except Exception as e:
            print(f"Erro ao iniciar sistema: {e}")
            import traceback
            traceback.print_exc()
            self.shutdown()
            return False
    
    def _main_loop(self):
        """Loop principal do sistema"""
        try:
            while self.running:
                time.sleep(1)
                
                # Verificar saúde da conexão a cada 30 segundos
                if int(time.time()) % 30 == 0:
                    if self.mt5_integration and not self.mt5_integration.is_connection_healthy():
                        print("⚠ Conexão MT5 não saudável, tentando reconectar...")
                        if self.mt5_integration.connect():
                            print("✓ Reconexão MT5 bem-sucedida")
                        else:
                            print("✗ Falha na reconexão MT5")
                
        except KeyboardInterrupt:
            print("\nSistema interrompido pelo usuário")
        except Exception as e:
            print(f"Erro no loop principal: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Para todos os componentes do sistema"""
        if not self.running:
            return
            
        print("\nParando sistema...")
        self.running = False
        
        try:
            # Parar componentes na ordem inversa
            if self.ml_engine:
                print("Parando ML Engine...")
                try:
                    self.ml_engine.stop()
                except:
                    pass
            
            if self.data_polling:
                print("Parando sistema de dados...")
                try:
                    self.data_polling.stop_polling()
                except:
                    pass
            
            if self.mt5_integration:
                print("Desconectando MT5...")
                try:
                    self.mt5_integration.disconnect()
                except:
                    pass
            
            if self.logging_system:
                print("Parando sistema de logging...")
                try:
                    self.logging_system.log(LogLevel.INFO, EventType.SYSTEM_STOP,
                                          "Sistema MT5 Trading parado", "simple_launcher")
                    self.logging_system.stop()
                except:
                    pass
            
            print("✓ Sistema parado com sucesso")
            
        except Exception as e:
            print(f"Erro ao parar sistema: {e}")
        
        finally:
            sys.exit(0)

def main():
    """Função principal"""
    print("MT5 Trading System - Launcher Simplificado")
    print("==========================================\n")
    
    launcher = SimpleMT5Launcher()
    launcher.start_system()

if __name__ == "__main__":
    main()