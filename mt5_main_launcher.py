#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launcher Principal do Sistema MT5 Trading
Integra todos os componentes: MT5, Dashboard, ML, Risk Management, Logging
Autor: Sistema Forex ML
Data: 2024
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

# Importar componentes do sistema
try:
    from mt5_integration import MT5Integration, MT5Config, TradeRequest, TradeAction, OrderType
    from mt5_dashboard import MT5Dashboard
    from risk_manager import RiskManager, RiskParameters, RiskLevel
    from data_polling_system import DataPollingSystem
    from ml_mt5_integration import MLTradingEngine
    from logging_system import LoggingSystem, LogLevel, EventType, get_logging_system
    from config_manager import ConfigManager
    from security_manager import SecurityManager
    from auto_reconnect_manager import AutoReconnectManager
    from credential_validator import CredentialValidator
    from connection_health_monitor import ConnectionHealthMonitor
    from secure_config_manager import SecureConfigManager
except ImportError as e:
    print(f"Erro ao importar componentes: {e}")
    print("Certifique-se de que todos os arquivos estão no mesmo diretório")
    sys.exit(1)

class MT5SystemStatus:
    """Status do sistema MT5"""
    def __init__(self):
        self.mt5_connected = False
        self.dashboard_running = False
        self.ml_engine_active = False
        self.risk_manager_active = False
        self.data_polling_active = False
        self.logging_active = False
        self.auto_trading_enabled = False
        self.last_update = datetime.now()
        self.errors = []
        self.warnings = []

class MT5SystemLauncher:
    """
    Launcher principal que gerencia todos os componentes do sistema
    """
    
    def __init__(self):
        # Status do sistema
        self.status = MT5SystemStatus()
        
        # Componentes
        self.config_manager = None
        self.logging_system = None
        self.mt5_integration = None
        self.risk_manager = None
        self.data_polling = None
        self.ml_engine = None
        self.dashboard = None
        
        # Novos componentes de segurança e reconexão
        self.security_manager = None
        self.auto_reconnect_manager = None
        self.credential_validator = None
        self.health_monitor = None
        self.secure_config_manager = None
        
        # Threading
        self.shutdown_event = threading.Event()
        self.system_thread = None
        
        # Configurações
        self.config_file = "mt5_system_config.json"
        self.secure_config_file = "mt5_secure_config.enc"
        self.default_config = self._get_default_config()
        
        # Interface de inicialização
        self.init_window = None
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuração padrão do sistema"""
        return {
            "mt5": {
                "login": 0,
                "password": "",
                "server": "",
                "path": "",
                "timeout": 10000,
                "auto_connect": True
            },
            "risk": {
                "max_daily_loss_percent": 2.0,
                "max_position_size_percent": 1.0,
                "max_positions": 5,
                "max_correlation": 0.7,
                "emergency_stop_loss_percent": 5.0,
                "use_dynamic_position_sizing": True,
                "risk_per_trade_percent": 0.5
            },
            "ml": {
                "prediction_interval_seconds": 30,
                "min_confidence_threshold": 0.6,
                "use_ensemble_predictions": True,
                "retrain_interval_hours": 24,
                "max_lookback_periods": 100
            },
            "data_polling": {
                "tick_interval_ms": 100,
                "account_interval_ms": 1000,
                "positions_interval_ms": 500,
                "orders_interval_ms": 500,
                "symbols_interval_ms": 5000
            },
            "dashboard": {
                "auto_start": True,
                "theme": "default",
                "update_interval_ms": 1000,
                "show_charts": True,
                "chart_timeframe": "M1"
            },
            "logging": {
                "log_level": "INFO",
                "max_log_days": 30,
                "enable_performance_monitoring": True,
                "log_trading_events": True
            },
            "system": {
                "auto_start_components": True,
                "enable_auto_trading": False,
                "startup_delay_seconds": 2,
                "shutdown_timeout_seconds": 10
            }
        }
    
    def load_config(self) -> Dict[str, Any]:
        """Carrega configuração do arquivo"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Mesclar com configuração padrão
                merged_config = self.default_config.copy()
                self._deep_update(merged_config, config)
                return merged_config
            else:
                return self.default_config.copy()
                
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
            return self.default_config.copy()
    
    def save_config(self, config: Dict[str, Any]):
        """Salva configuração no arquivo"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar configuração: {e}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Atualiza dicionário recursivamente"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def show_initialization_window(self):
        """Mostra janela de inicialização"""
        self.init_window = tk.Tk()
        self.init_window.title("MT5 Trading System - Inicialização")
        self.init_window.geometry("800x700")
        self.init_window.resizable(True, True)
        self.init_window.minsize(600, 500)
        
        # Centralizar janela
        self.init_window.update_idletasks()
        x = (self.init_window.winfo_screenwidth() // 2) - (800 // 2)
        y = (self.init_window.winfo_screenheight() // 2) - (700 // 2)
        self.init_window.geometry(f"800x700+{x}+{y}")
        
        # Frame principal
        main_frame = ttk.Frame(self.init_window, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Título
        title_label = ttk.Label(main_frame, text="Sistema de Trading MT5", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Configurações MT5
        mt5_frame = ttk.LabelFrame(main_frame, text="Configurações MetaTrader 5", padding="10")
        mt5_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Campos de configuração
        self.login_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.server_var = tk.StringVar()
        self.path_var = tk.StringVar()
        
        ttk.Label(mt5_frame, text="Login:").grid(row=0, column=0, sticky=tk.W, pady=2)
        login_entry = ttk.Entry(mt5_frame, textvariable=self.login_var, width=30)
        login_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))
        
        ttk.Label(mt5_frame, text="Senha:").grid(row=1, column=0, sticky=tk.W, pady=2)
        password_entry = ttk.Entry(mt5_frame, textvariable=self.password_var, show="*", width=30)
        password_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))
        
        ttk.Label(mt5_frame, text="Servidor:").grid(row=2, column=0, sticky=tk.W, pady=2)
        server_entry = ttk.Entry(mt5_frame, textvariable=self.server_var, width=30)
        server_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))
        
        ttk.Label(mt5_frame, text="Caminho MT5:").grid(row=3, column=0, sticky=tk.W, pady=2)
        path_entry = ttk.Entry(mt5_frame, textvariable=self.path_var, width=30)
        path_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))
        
        # Botão para detectar MT5
        detect_btn = ttk.Button(mt5_frame, text="Detectar MT5", command=self._detect_mt5_path)
        detect_btn.grid(row=4, column=1, sticky=tk.E, pady=(10, 0))
        
        # Opções do sistema
        options_frame = ttk.LabelFrame(main_frame, text="Opções do Sistema", padding="10")
        options_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.auto_start_var = tk.BooleanVar(value=True)
        self.enable_ml_var = tk.BooleanVar(value=True)
        self.enable_auto_trading_var = tk.BooleanVar(value=False)
        self.show_dashboard_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(options_frame, text="Iniciar componentes automaticamente", 
                       variable=self.auto_start_var).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        ttk.Checkbutton(options_frame, text="Habilitar Machine Learning", 
                       variable=self.enable_ml_var).grid(row=1, column=0, sticky=tk.W, pady=2)
        
        ttk.Checkbutton(options_frame, text="Habilitar Auto Trading (CUIDADO!)", 
                       variable=self.enable_auto_trading_var).grid(row=2, column=0, sticky=tk.W, pady=2)
        
        ttk.Checkbutton(options_frame, text="Mostrar Dashboard", 
                       variable=self.show_dashboard_var).grid(row=3, column=0, sticky=tk.W, pady=2)
        
        # Status
        self.status_text = tk.Text(main_frame, height=10, width=80, state=tk.DISABLED, wrap=tk.WORD)
        self.status_text.grid(row=3, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar para status
        status_scroll = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        status_scroll.grid(row=3, column=2, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=status_scroll.set)
        
        # Botões
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(button_frame, text="Carregar Config", 
                  command=self._load_config_to_ui).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Salvar Config", 
                  command=self._save_config_from_ui).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Backup Manual", 
                  command=self._create_manual_backup).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Relatório Integridade", 
                  command=self._show_integrity_report).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Iniciar Sistema", 
                  command=self._start_system_from_ui).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Sair", 
                  command=self._exit_application).pack(side=tk.RIGHT)
        
        # Configurar grid weights para redimensionamento adequado
        self.init_window.columnconfigure(0, weight=1)
        self.init_window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)  # Status text area
        mt5_frame.columnconfigure(1, weight=1)
        options_frame.columnconfigure(0, weight=1)
        
        # Carregar configuração inicial
        self._load_config_to_ui()
        
        # Adicionar status inicial
        self._add_status_message("Sistema pronto para inicialização")
        
        self.init_window.mainloop()
    
    def _detect_mt5_path(self):
        """Detecta caminho do MT5 automaticamente"""
        common_paths = [
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
            r"C:\Users\{}\AppData\Roaming\MetaQuotes\Terminal\*\terminal64.exe".format(os.getenv('USERNAME')),
        ]
        
        for path_pattern in common_paths:
            if '*' in path_pattern:
                # Usar glob para padrões
                import glob
                matches = glob.glob(path_pattern)
                if matches:
                    self.path_var.set(matches[0])
                    self._add_status_message(f"MT5 detectado: {matches[0]}")
                    return
            else:
                if os.path.exists(path_pattern):
                    self.path_var.set(path_pattern)
                    self._add_status_message(f"MT5 detectado: {path_pattern}")
                    return
        
        self._add_status_message("MT5 não detectado automaticamente. Configure manualmente.")
    
    def _load_config_to_ui(self):
        """Carrega configuração para a interface"""
        try:
            # Usar SecureConfigManager se disponível, senão usar método tradicional
            if hasattr(self, 'secure_config_manager') and self.secure_config_manager:
                try:
                    config = self.secure_config_manager.load_config(validate=True)
                    self._add_status_message("Configuração carregada com validação")
                except Exception as e:
                    self._add_status_message(f"Erro no SecureConfigManager: {e}")
                    # Fallback para método tradicional
                    config = self.load_config()
            else:
                config = self.load_config()
            
            # Tentar carregar configuração segura se disponível
            if hasattr(self, 'security_manager') and self.security_manager:
                try:
                    secure_config = self.security_manager.load_secure_config(self.secure_config_file)
                    if secure_config and 'mt5' in secure_config:
                        config['mt5'].update(secure_config['mt5'])
                        self._add_status_message("Configuração segura carregada")
                except Exception as e:
                    self._add_status_message(f"Aviso: Não foi possível carregar configuração segura: {e}")
            
            # Carregar configurações MT5
            mt5_config = config.get('mt5', {})
            self.login_var.set(str(mt5_config.get('login', '')))
            self.password_var.set(mt5_config.get('password', ''))
            self.server_var.set(mt5_config.get('server', ''))
            self.path_var.set(mt5_config.get('path', ''))
            
            # Carregar opções do sistema
            system_config = config.get('system', {})
            self.auto_start_var.set(system_config.get('auto_start_components', True))
            self.enable_auto_trading_var.set(system_config.get('enable_auto_trading', False))
            
            dashboard_config = config.get('dashboard', {})
            self.show_dashboard_var.set(dashboard_config.get('auto_start', True))
            
            ml_config = config.get('ml', {})
            self.enable_ml_var.set(ml_config.get('prediction_interval_seconds', 30) > 0)
            
            self._add_status_message("Configuração carregada com sucesso")
            
        except Exception as e:
            self._add_status_message(f"Erro ao carregar configuração: {e}")
    
    def _save_config_from_ui(self):
        """Salva configuração da interface"""
        try:
            config = self.load_config()
            
            # Atualizar configurações MT5
            mt5_config = {
                'login': int(self.login_var.get()) if self.login_var.get().isdigit() else 0,
                'password': self.password_var.get(),
                'server': self.server_var.get(),
                'path': self.path_var.get()
            }
            
            # Validar credenciais se o validador estiver disponível
            if hasattr(self, 'credential_validator') and self.credential_validator:
                try:
                    validation_report = self.credential_validator.validate_credentials(mt5_config)
                    
                    # Mostrar erros críticos
                    if validation_report.errors:
                        error_messages = [issue.message for issue in validation_report.errors]
                        self._add_status_message(f"Erros de validação: {'; '.join(error_messages)}", "error")
                        return  # Não salvar se houver erros críticos
                    
                    # Mostrar avisos
                    if validation_report.warnings:
                        warning_messages = [issue.message for issue in validation_report.warnings]
                        self._add_status_message(f"Avisos: {'; '.join(warning_messages)}", "warning")
                    
                    if validation_report.is_valid:
                        self._add_status_message("Credenciais validadas com sucesso")
                        
                except Exception as e:
                    self._add_status_message(f"Erro na validação de credenciais: {e}", "warning")
                    # Continuar mesmo com erro de validação
            
            # Salvar credenciais de forma segura se o SecurityManager estiver disponível
            if hasattr(self, 'security_manager') and self.security_manager:
                try:
                    # Criar configuração segura apenas com credenciais sensíveis
                    secure_config = {
                        'mt5': {
                            'login': mt5_config['login'],
                            'password': mt5_config['password'],
                            'server': mt5_config['server']
                        }
                    }
                    
                    # Salvar configuração criptografada
                    self.security_manager.save_secure_config(secure_config, self.secure_config_file)
                    
                    # Remover senha da configuração normal (manter apenas path)
                    config['mt5']['login'] = mt5_config['login']
                    config['mt5']['password'] = "***ENCRYPTED***"
                    config['mt5']['server'] = mt5_config['server']
                    config['mt5']['path'] = mt5_config['path']
                    
                    self._add_status_message("Credenciais salvas de forma segura")
                    
                except Exception as e:
                    self._add_status_message(f"Aviso: Erro ao salvar configuração segura: {e}")
                    # Fallback para salvamento normal
                    config['mt5'].update(mt5_config)
            else:
                # Salvamento normal se SecurityManager não estiver disponível
                config['mt5'].update(mt5_config)
            
            # Atualizar opções do sistema
            config['system']['auto_start_components'] = self.auto_start_var.get()
            config['system']['enable_auto_trading'] = self.enable_auto_trading_var.get()
            config['dashboard']['auto_start'] = self.show_dashboard_var.get()
            
            # Salvar configuração usando SecureConfigManager se disponível
            if hasattr(self, 'secure_config_manager') and self.secure_config_manager:
                try:
                    self.secure_config_manager.save_config(config, create_backup=True)
                    self._add_status_message("Configuração salva com backup automático")
                except Exception as e:
                    self._add_status_message(f"Erro no SecureConfigManager: {e}")
                    # Fallback para método tradicional
                    self.save_config(config)
                    self._add_status_message("Configuração salva (método tradicional)")
            else:
                # Salvar configuração principal
                self.save_config(config)
                self._add_status_message("Configuração salva com sucesso")
            
        except Exception as e:
            self._add_status_message(f"Erro ao salvar configuração: {e}")
    
    def _start_system_from_ui(self):
        """Inicia sistema a partir da interface"""
        try:
            # Salvar configuração atual
            self._save_config_from_ui()
            
            # Validar configurações mínimas
            if not self.login_var.get() or not self.password_var.get() or not self.server_var.get():
                messagebox.showerror("Erro", "Preencha pelo menos Login, Senha e Servidor do MT5")
                return
            
            self._add_status_message("Iniciando sistema...")
            
            # Minimizar janela de inicialização em vez de fechar
            self.init_window.withdraw()
            
            # Criar janela de status do sistema
            self._create_system_status_window()
            
            # Iniciar sistema em thread separada
            self.system_thread = threading.Thread(target=self._run_system, daemon=False)
            self.system_thread.start()
            
        except Exception as e:
            self._add_status_message(f"Erro ao iniciar sistema: {e}")
            messagebox.showerror("Erro", f"Erro ao iniciar sistema: {e}")
    
    def _create_system_status_window(self):
        """Cria janela de status do sistema"""
        self.status_window = tk.Toplevel()
        self.status_window.title("MT5 Trading System - Status")
        self.status_window.geometry("600x400")
        self.status_window.resizable(True, True)
        
        # Centralizar janela
        self.status_window.update_idletasks()
        x = (self.status_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.status_window.winfo_screenheight() // 2) - (400 // 2)
        self.status_window.geometry(f"600x400+{x}+{y}")
        
        # Frame principal
        main_frame = ttk.Frame(self.status_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = ttk.Label(main_frame, text="Sistema MT5 Trading - Ativo", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Status do sistema
        self.system_status_text = tk.Text(main_frame, height=15, width=70, state=tk.DISABLED, wrap=tk.WORD)
        self.system_status_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Scrollbar para status
        status_scroll = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.system_status_text.yview)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.system_status_text.configure(yscrollcommand=status_scroll.set)
        
        # Botões
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Parar Sistema", 
                  command=self._stop_system).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Mostrar Configuração", 
                  command=self._show_config_window).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Minimizar", 
                  command=self.status_window.withdraw).pack(side=tk.RIGHT)
        
        # Protocolo de fechamento
        self.status_window.protocol("WM_DELETE_WINDOW", self._on_status_window_close)
        
        # Adicionar mensagem inicial
        self._add_system_status_message("Sistema MT5 Trading iniciado com sucesso")
        
    def _create_manual_backup(self):
        """Cria backup manual da configuração"""
        try:
            if hasattr(self, 'secure_config_manager') and self.secure_config_manager:
                backup_path = self.secure_config_manager.create_backup("manual")
                if backup_path:
                    self._add_status_message(f"Backup manual criado: {backup_path.name}")
                else:
                    self._add_status_message("Erro ao criar backup manual")
            else:
                self._add_status_message("SecureConfigManager não disponível")
        except Exception as e:
            self._add_status_message(f"Erro ao criar backup: {e}")
    
    def _show_integrity_report(self):
        """Mostra relatório de integridade em janela separada"""
        try:
            if hasattr(self, 'secure_config_manager') and self.secure_config_manager:
                report = self.secure_config_manager.get_integrity_report()
                self._display_integrity_report(report)
            else:
                self._add_status_message("SecureConfigManager não disponível")
        except Exception as e:
            self._add_status_message(f"Erro ao gerar relatório: {e}")
    
    def _display_integrity_report(self, report):
        """Exibe relatório de integridade em janela popup"""
        import tkinter.messagebox as msgbox
        
        # Formatar relatório
        report_text = f"""RELATÓRIO DE INTEGRIDADE
        
Timestamp: {report['timestamp']}
        
Arquivo de Configuração:
- Existe: {'Sim' if report['config_file']['exists'] else 'Não'}
- Tamanho: {report['config_file']['size']} bytes
- Checksum: {report['config_file']['checksum'] or 'N/A'}
        
Backups Disponíveis: {len(report['backups'])}
        """
        
        if report['backups']:
            report_text += "\nÚltimos Backups:\n"
            for i, backup in enumerate(report['backups'][:5]):  # Mostrar apenas os 5 mais recentes
                report_text += f"- {backup['filename']} ({backup['size']} bytes) - {backup['status']}\n"
        
        if report['validation']:
            validation = report['validation']
            if 'error' in validation:
                report_text += f"\nValidação: ERRO - {validation['error']}"
            else:
                status = "VÁLIDA" if validation['is_valid'] else "INVÁLIDA"
                report_text += f"\nValidação: {status}"
                if validation['issues']:
                    report_text += f"\nProblemas: {', '.join(validation['issues'])}"
        
        # Criar janela de relatório
        report_window = tk.Toplevel(self.init_window)
        report_window.title("Relatório de Integridade")
        report_window.geometry("600x400")
        
        # Texto do relatório
        text_widget = tk.Text(report_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, report_text)
        text_widget.config(state=tk.DISABLED)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(report_window, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Botão fechar
        ttk.Button(report_window, text="Fechar", 
                  command=report_window.destroy).pack(pady=10)
    
    def _exit_application(self):
        """Sai da aplicação"""
        # Parar backup automático se estiver rodando
        if hasattr(self, 'secure_config_manager') and self.secure_config_manager:
            self.secure_config_manager.stop_auto_backup()
        
        if messagebox.askyesno("Confirmar", "Deseja realmente sair?"):
            if self.init_window:
                self.init_window.quit()  # Usar quit() ao invés de destroy() para sair do mainloop
                self.init_window.destroy()
    
    def _add_status_message(self, message: str):
        """Adiciona mensagem ao status"""
        if hasattr(self, 'status_text'):
            self.status_text.configure(state=tk.NORMAL)
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.status_text.see(tk.END)
            self.status_text.configure(state=tk.DISABLED)
            self.status_text.update()
    
    def _add_system_status_message(self, message: str):
        """Adiciona mensagem ao status do sistema"""
        if hasattr(self, 'system_status_text'):
            self.system_status_text.configure(state=tk.NORMAL)
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.system_status_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.system_status_text.see(tk.END)
            self.system_status_text.configure(state=tk.DISABLED)
            self.system_status_text.update()
    
    def _stop_system(self):
        """Para o sistema"""
        if messagebox.askyesno("Confirmar", "Deseja realmente parar o sistema?"):
            self._add_system_status_message("Parando sistema...")
            self.shutdown()
    
    def _show_config_window(self):
        """Mostra janela de configuração"""
        if hasattr(self, 'init_window'):
            self.init_window.deiconify()
        else:
            messagebox.showinfo("Info", "Janela de configuração não disponível")
    
    def _on_status_window_close(self):
        """Chamado quando janela de status é fechada"""
        if messagebox.askyesno("Confirmar", "Deseja parar o sistema?"):
            self.shutdown()
        else:
            self.status_window.withdraw()
    
    def _run_system(self):
        """Executa o sistema principal"""
        try:
            # Carregar configuração
            config = self.load_config()
            
            print("=== MT5 Trading System - Iniciando ===")
            print(f"Timestamp: {datetime.now()}")
            
            # 1. Inicializar sistema de logging
            print("[1/8] Inicializando sistema de logging...")
            self.logging_system = get_logging_system()
            self.status.logging_active = True
            
            self.logging_system.log(LogLevel.INFO, EventType.SYSTEM_START, 
                                  "Sistema MT5 Trading iniciado", "launcher")
            
            # 2. Inicializar gerenciador de configuração
            print("[2/8] Inicializando gerenciador de configuração...")
            self.config_manager = ConfigManager()
            
            # 2.1. Inicializar gerenciador de segurança
            print("[2.1/7] Inicializando gerenciador de segurança...")
            self.security_manager = SecurityManager()
            
            # 2.2. Inicializar gerenciador de configuração segura
            print("[2.2/7] Inicializando gerenciador de configuração segura...")
            self.secure_config_manager = SecureConfigManager(
                config_file="config.json",
                backup_dir="config_backups",
                max_backups=10,
                backup_interval_hours=6,  # Backup a cada 6 horas
                auto_backup=True
            )
            
            # 2.3. Inicializar validador de credenciais
            print("[2.3/7] Inicializando validador de credenciais...")
            self.credential_validator = CredentialValidator()
            
            # 2.4. Inicializar monitor de saúde da conexão
            print("[2.4/7] Inicializando monitor de saúde...")
            self.health_monitor = ConnectionHealthMonitor(
                check_interval=30.0,  # Verificar a cada 30 segundos
                ping_threshold_warning=1000.0,  # 1 segundo
                ping_threshold_critical=3000.0,  # 3 segundos
                disconnect_threshold=3  # 3 falhas consecutivas
            )
            
            # 3. Inicializar integração MT5
            print("[3/8] Conectando ao MetaTrader 5...")
            mt5_config = MT5Config(
                login=config['mt5']['login'],
                password=config['mt5']['password'],
                server=config['mt5']['server'],
                path=config['mt5']['path'],
                timeout=config['mt5']['timeout']
            )
            
            self.mt5_integration = MT5Integration(mt5_config)
            
            # 3.1. Inicializar gerenciador de reconexão automática
            print("[3.1/8] Configurando reconexão automática...")
            from auto_reconnect_manager import ReconnectConfig
            reconnect_config = ReconnectConfig(
                max_retry_attempts=5,
                initial_retry_delay=1.0,
                max_retry_delay=60.0,
                backoff_multiplier=2.0
            )
            self.auto_reconnect_manager = AutoReconnectManager(
                mt5_integration=self.mt5_integration,
                config=reconnect_config
            )
            
            # Conectar MT5 com suporte à reconexão
            self.mt5_integration.auto_reconnect_manager = self.auto_reconnect_manager
            
            if self.mt5_integration.connect():
                self.status.mt5_connected = True
                print("✓ Conectado ao MT5 com sucesso")
                
                # Iniciar monitoramento de reconexão
                self.auto_reconnect_manager.start_monitoring()
                
                # Iniciar monitoramento de saúde da conexão
                if self.health_monitor:
                    try:
                        self.health_monitor.start_monitoring()
                        print("✓ Monitor de saúde da conexão iniciado")
                    except Exception as e:
                        print(f"⚠ Erro ao iniciar monitor de saúde: {e}")
                
                account_info = self.mt5_integration.get_account_info()
                if account_info:
                    print(f"  Conta: {account_info.get('login')} - {account_info.get('name')}")
                    print(f"  Saldo: {account_info.get('balance'):.2f} {account_info.get('currency')}")
            else:
                raise Exception("Falha ao conectar com MT5")
            
            # 4. Inicializar gerenciador de risco
            print("[4/8] Inicializando gerenciador de risco...")
            risk_params = RiskParameters(
                max_daily_risk=config['risk']['max_daily_loss_percent'],
                base_position_size=config['risk']['max_position_size_percent'],
                max_positions=config['risk']['max_positions'],
                max_correlation=config['risk']['max_correlation'],
                max_drawdown=config['risk']['emergency_stop_loss_percent'],
                risk_level=RiskLevel.MODERATE  # Definir nível de risco padrão
            )
            
            self.risk_manager = RiskManager(self.mt5_integration, risk_params)
            self.status.risk_manager_active = True
            print("✓ Gerenciador de risco ativo")
            
            # 5. Inicializar sistema de polling de dados
            print("[5/8] Inicializando sistema de dados...")
            
            self.data_polling = DataPollingSystem(self.mt5_integration)
            self.data_polling.start()
            self.status.data_polling_active = True
            print("✓ Sistema de dados ativo")
            
            # 6. Inicializar engine ML (se habilitado)
            if self.enable_ml_var.get():
                print("[6/8] Inicializando engine de Machine Learning...")
                try:
                    self.ml_engine = MLTradingEngine(
                        mt5_client=self.mt5_integration,
                        polling_system=self.data_polling,
                        risk_manager=self.risk_manager
                    )
                    
                    self.ml_engine.start()
                    self.status.ml_engine_active = True
                    print("✓ Engine ML ativo")
                except Exception as e:
                    print(f"⚠ Erro ao inicializar ML: {e}")
                    self.status.warnings.append(f"ML não disponível: {e}")
            else:
                print("[6/8] Machine Learning desabilitado")
            
            # 7. Inicializar dashboard (se habilitado)
            if self.show_dashboard_var.get():
                print("[7/8] Inicializando dashboard...")
                
                # Dashboard deve rodar na thread principal
                def start_dashboard():
                    try:
                        self.dashboard = MT5Dashboard(
                            mt5_integration=self.mt5_integration,
                            risk_manager=self.risk_manager,
                            data_polling=self.data_polling,
                            ml_engine=self.ml_engine,
                            logging_system=self.logging_system
                        )
                        
                        self.status.dashboard_running = True
                        print("✓ Dashboard iniciado")
                        
                        # Executar dashboard
                        self.dashboard.run()
                        
                    except Exception as e:
                        print(f"Erro no dashboard: {e}")
                        self.logging_system.log(LogLevel.ERROR, EventType.ERROR_OCCURRED,
                                              f"Erro no dashboard: {e}", "dashboard", exception=e)
                
                # Agendar dashboard para thread principal
                import tkinter as tk
                root = tk.Tk()
                root.after(100, start_dashboard)
                root.withdraw()  # Esconder janela raiz
                
            else:
                print("[7/8] Dashboard desabilitado")
                
                # Se dashboard não for usado, manter sistema rodando
                print("\n=== Sistema MT5 Trading Ativo ===")
                print("Pressione Ctrl+C para parar")
                
                try:
                    while not self.shutdown_event.is_set():
                        time.sleep(1)
                        
                        # Verificar status dos componentes
                        self._check_system_health()
                        
                except KeyboardInterrupt:
                    print("\nParando sistema...")
                    self.shutdown()
            
        except Exception as e:
            error_msg = f"Erro crítico no sistema: {e}"
            print(f"❌ {error_msg}")
            
            if self.logging_system:
                self.logging_system.log(LogLevel.CRITICAL, EventType.ERROR_OCCURRED,
                                      error_msg, "launcher", exception=e)
            
            # Mostrar erro para usuário
            try:
                messagebox.showerror("Erro Crítico", error_msg)
            except:
                pass
            
            self.shutdown()
    
    def _check_system_health(self):
        """Verifica saúde do sistema"""
        try:
            # Verificar conexão MT5
            if self.mt5_integration and not self.mt5_integration.is_connected():
                if self.status.mt5_connected:
                    self.status.mt5_connected = False
                    self.status.errors.append("Conexão MT5 perdida")
                    
                    if self.logging_system:
                        self.logging_system.log(LogLevel.ERROR, EventType.DISCONNECTION,
                                              "Conexão MT5 perdida", "mt5")
            
            # Verificar outros componentes...
            self.status.last_update = datetime.now()
            
        except Exception as e:
            if self.logging_system:
                self.logging_system.log(LogLevel.ERROR, EventType.ERROR_OCCURRED,
                                      f"Erro na verificação de saúde: {e}", "launcher", exception=e)
    
    def shutdown(self):
        """Para todos os componentes do sistema"""
        print("\n=== Parando Sistema MT5 Trading ===")
        
        self.shutdown_event.set()
        
        try:
            # Parar componentes na ordem inversa
            if self.dashboard:
                print("Parando dashboard...")
                try:
                    self.dashboard.stop()
                except:
                    pass
            
            if self.ml_engine:
                print("Parando engine ML...")
                try:
                    self.ml_engine.stop()
                except:
                    pass
            
            if self.data_polling:
                print("Parando sistema de dados...")
                try:
                    self.data_polling.stop()
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
                                          "Sistema MT5 Trading parado", "launcher")
                    self.logging_system.stop()
                except:
                    pass
            
            print("✓ Sistema parado com sucesso")
            
        except Exception as e:
            print(f"Erro ao parar sistema: {e}")
        
        finally:
            # Forçar saída se necessário
            import os
            os._exit(0)
    
    def get_system_status(self) -> MT5SystemStatus:
        """Retorna status atual do sistema"""
        return self.status

def main():
    """Função principal"""
    print("MT5 Trading System Launcher")
    print("===========================\n")
    
    launcher = None
    try:
        # Criar launcher
        launcher = MT5SystemLauncher()
        
        # Mostrar janela de inicialização
        launcher.show_initialization_window()
        
        # O mainloop da janela gerencia o ciclo de vida
        # Quando a janela é fechada, o programa continua aqui
        
    except KeyboardInterrupt:
        print("\nSistema interrompido pelo usuário")
        if launcher:
            launcher.shutdown()
    except Exception as e:
        print(f"Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        if launcher:
            launcher.shutdown()
    
    # Não chamar sys.exit automaticamente
    # Deixar que o sistema termine naturalmente ou seja gerenciado pelas janelas

if __name__ == "__main__":
    main()