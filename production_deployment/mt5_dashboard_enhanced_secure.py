import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

# Importar componentes desenvolvidos
from advanced_widgets import (
    AnimatedProgressBar, CircularProgress, LiveChart, 
    MetricCard, StatusIndicator, NotificationType
)
from notification_system import (
    NotificationManager, NotificationPanel, NotificationLevel, 
    NotificationCategory, Notification, NotificationAction,
    get_notification_manager, notify_info, notify_success, 
    notify_warning, notify_error, notify_trade
)
from export_system import ExportManager
from settings_system import SettingsManager, SettingsDialog, ThemeManager
from user_preferences import UserPreferences, PreferencesDialog
from performance_optimization import get_performance_optimizer, performance_timer

# Importar novos sistemas
from decision_logging_system import (
    get_decision_logger, DecisionType, DecisionLevel,
    log_trade_decision, log_risk_decision, log_system_decision
)
from mt5_secure_auth import (
    get_mt5_authenticator, connect_mt5, disconnect_mt5, 
    is_mt5_connected, get_mt5_account_info, ConnectionStatus
)

# Importar componentes originais
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    print("MetaTrader5 não disponível - modo simulação")

class ModernTheme:
    """Tema moderno para o dashboard"""
    
    # Cores principais
    PRIMARY = '#0078d4'
    SECONDARY = '#6c757d'
    SUCCESS = '#107c10'
    WARNING = '#ff8c00'
    DANGER = '#d13438'
    INFO = '#17a2b8'
    
    # Cores de fundo
    BG_PRIMARY = '#1e1e1e'
    BG_SECONDARY = '#2d2d2d'
    BG_TERTIARY = '#3d3d3d'
    BG_QUATERNARY = '#4d4d4d'
    
    # Cores de texto
    TEXT_PRIMARY = '#ffffff'
    TEXT_SECONDARY = '#cccccc'
    TEXT_MUTED = '#999999'
    TEXT_DISABLED = '#666666'
    
    # Bordas
    BORDER_COLOR = '#484848'
    BORDER_LIGHT = '#5a5a5a'
    
    @classmethod
    def configure_ttk_style(cls):
        """Configura estilos TTK"""
        style = ttk.Style()
        
        # Configurar tema base
        style.theme_use('clam')
        
        # Notebook
        style.configure('Modern.TNotebook', 
                       background=cls.BG_SECONDARY,
                       borderwidth=0)
        style.configure('Modern.TNotebook.Tab',
                       background=cls.BG_TERTIARY,
                       foreground=cls.TEXT_SECONDARY,
                       padding=[20, 10],
                       borderwidth=0)
        style.map('Modern.TNotebook.Tab',
                 background=[('selected', cls.BG_SECONDARY),
                           ('active', cls.BG_QUATERNARY)])
        
        # Treeview
        style.configure('Modern.Treeview',
                       background=cls.BG_SECONDARY,
                       foreground=cls.TEXT_PRIMARY,
                       fieldbackground=cls.BG_SECONDARY,
                       borderwidth=0)
        style.configure('Modern.Treeview.Heading',
                       background=cls.BG_TERTIARY,
                       foreground=cls.TEXT_PRIMARY,
                       borderwidth=1,
                       relief='solid')
        
        # Scrollbar
        style.configure('Modern.Vertical.TScrollbar',
                       background=cls.BG_TERTIARY,
                       troughcolor=cls.BG_SECONDARY,
                       borderwidth=0,
                       arrowcolor=cls.TEXT_MUTED)

class ResponsiveFrame(tk.Frame):
    """Frame responsivo que se adapta ao redimensionamento"""
    
    def __init__(self, parent, min_width=200, min_height=100, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.min_width = min_width
        self.min_height = min_height
        
        # Bind para redimensionamento
        self.bind('<Configure>', self._on_configure)
        
        # Grid weights para responsividade
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
    
    def _on_configure(self, event):
        """Callback para redimensionamento"""
        if event.widget == self:
            width = max(self.min_width, event.width)
            height = max(self.min_height, event.height)
            
            # Ajustar layout baseado no tamanho
            self._adjust_layout(width, height)
    
    def _adjust_layout(self, width, height):
        """Ajusta layout baseado no tamanho"""
        # Implementação específica em subclasses
        pass

class SecureTradingControlPanel(ResponsiveFrame):
    """Painel de controle de trading com autenticação segura e logging detalhado"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=ModernTheme.BG_SECONDARY, **kwargs)
        
        # Inicializar sistemas
        self.decision_logger = get_decision_logger()
        self.mt5_authenticator = get_mt5_authenticator()
        
        # Estado da conexão
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.auto_trading = False
        self.last_update = time.time()
        
        # Thread para monitoramento
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self._create_widgets()
        self._start_monitoring()
        
        # Log inicial
        log_system_decision(
            "Inicialização do painel de controle",
            "Painel de trading com autenticação segura inicializado",
            "Sistema pronto para operação",
            DecisionLevel.INFO
        )
    
    def _create_widgets(self):
        """Cria widgets do painel com logging integrado"""
        # Título com indicador de logging
        title_frame = tk.Frame(self, bg=ModernTheme.BG_SECONDARY, height=40)
        title_frame.pack(fill=tk.X, padx=15, pady=(15, 0))
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="Controle de Trading Seguro",
                font=('Segoe UI', 12, 'bold'),
                bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_PRIMARY).pack(side=tk.LEFT, pady=10)
        
        # Indicador de logging ativo
        self.logging_indicator = StatusIndicator(title_frame, size=12)
        self.logging_indicator.pack(side=tk.RIGHT, padx=10)
        self.logging_indicator.set_status('online')
        
        tk.Label(title_frame, text="Log Ativo",
                font=('Segoe UI', 8),
                bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.SUCCESS).pack(side=tk.RIGHT, padx=5)
        
        # Status de conexão com detalhes
        status_frame = tk.Frame(self, bg=ModernTheme.BG_SECONDARY)
        status_frame.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(status_frame, text="Status MT5:",
                font=('Segoe UI', 9),
                bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_SECONDARY).pack(side=tk.LEFT)
        
        self.status_indicator = StatusIndicator(status_frame, size=16)
        self.status_indicator.pack(side=tk.LEFT, padx=10)
        
        self.status_label = tk.Label(status_frame, text="Desconectado",
                                    font=('Segoe UI', 9, 'bold'),
                                    bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.DANGER)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Informações de conexão detalhadas
        self.connection_details = tk.Label(status_frame, text="",
                                          font=('Segoe UI', 8),
                                          bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_MUTED)
        self.connection_details.pack(side=tk.LEFT, padx=10)
        
        # Botões de controle com logging
        buttons_frame = tk.Frame(self, bg=ModernTheme.BG_SECONDARY)
        buttons_frame.pack(fill=tk.X, padx=15, pady=10)
        
        self.connect_btn = tk.Button(buttons_frame, text="Conectar MT5 Seguro",
                                    font=('Segoe UI', 9),
                                    bg=ModernTheme.SUCCESS, fg=ModernTheme.TEXT_PRIMARY,
                                    borderwidth=0, pady=8, padx=20,
                                    command=self._toggle_connection_with_logging)
        self.connect_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.auto_trade_btn = tk.Button(buttons_frame, text="Auto Trading: OFF",
                                       font=('Segoe UI', 9),
                                       bg=ModernTheme.BG_TERTIARY, fg=ModernTheme.TEXT_SECONDARY,
                                       borderwidth=0, pady=8, padx=20,
                                       command=self._toggle_auto_trading_with_logging)
        self.auto_trade_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botão para visualizar logs
        self.view_logs_btn = tk.Button(buttons_frame, text="Ver Logs",
                                      font=('Segoe UI', 9),
                                      bg=ModernTheme.INFO, fg=ModernTheme.TEXT_PRIMARY,
                                      borderwidth=0, pady=8, padx=20,
                                      command=self._show_decision_logs)
        self.view_logs_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Informações da conta com métricas de decisão
        account_frame = tk.LabelFrame(self, text="Informações da Conta & Decisões",
                                     font=('Segoe UI', 9, 'bold'),
                                     bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_PRIMARY,
                                     borderwidth=1, relief='solid')
        account_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Cards de métricas da conta
        account_metrics = tk.Frame(account_frame, bg=ModernTheme.BG_SECONDARY)
        account_metrics.pack(fill=tk.X, padx=10, pady=5)
        
        self.balance_card = MetricCard(account_metrics, "Saldo", "$0.00")
        self.balance_card.pack(side=tk.LEFT, padx=(0, 10))
        
        self.equity_card = MetricCard(account_metrics, "Patrimônio", "$0.00")
        self.equity_card.pack(side=tk.LEFT, padx=(0, 10))
        
        self.margin_card = MetricCard(account_metrics, "Margem", "$0.00")
        self.margin_card.pack(side=tk.LEFT)
        
        # Cards de métricas de decisão
        decision_metrics = tk.Frame(account_frame, bg=ModernTheme.BG_SECONDARY)
        decision_metrics.pack(fill=tk.X, padx=10, pady=5)
        
        self.decisions_count_card = MetricCard(decision_metrics, "Decisões", "0")
        self.decisions_count_card.pack(side=tk.LEFT, padx=(0, 10))
        
        self.avg_confidence_card = MetricCard(decision_metrics, "Confiança Média", "0%")
        self.avg_confidence_card.pack(side=tk.LEFT, padx=(0, 10))
        
        self.avg_exec_time_card = MetricCard(decision_metrics, "Tempo Médio", "0ms")
        self.avg_exec_time_card.pack(side=tk.LEFT)
    
    def _toggle_connection_with_logging(self):
        """Alterna conexão MT5 com logging detalhado"""
        with self.decision_logger.decision_context(
            DecisionType.SYSTEM_OPERATION,
            "Alternar conexão MT5",
            DecisionLevel.HIGH
        ) as ctx:
            
            current_status = self.mt5_authenticator.get_connection_status()
            
            ctx.add_criteria("current_status", current_status.value, 1.0)
            ctx.add_criteria("user_action", "toggle_connection", 1.0)
            
            if current_status == ConnectionStatus.DISCONNECTED:
                ctx.set_justification("Usuário solicitou conexão ao MT5")
                self._connect_mt5_secure(ctx)
            else:
                ctx.set_justification("Usuário solicitou desconexão do MT5")
                self._disconnect_mt5_secure(ctx)
    
    def _connect_mt5_secure(self, decision_ctx):
        """Conecta ao MT5 usando autenticação segura"""
        try:
            # Tentar conexão segura
            result = self.mt5_authenticator.connect_with_auth()
            
            decision_ctx.add_criteria("connection_success", result.success, 1.0)
            decision_ctx.add_criteria("connection_time_ms", result.connection_time or 0, 0.5)
            
            if result.success:
                self.connection_status = result.status
                
                decision_ctx.set_outcome(f"Conectado com sucesso: {result.message}")
                decision_ctx.set_confidence(0.9)
                
                notify_success("Conexão Segura MT5", result.message)
                
                # Atualizar informações da conta
                self._update_account_info_with_logging()
                
            else:
                decision_ctx.add_criteria("error_message", result.message, 1.0)
                decision_ctx.set_outcome(f"Falha na conexão: {result.message}")
                decision_ctx.set_confidence(0.1)
                
                notify_error("Erro de Conexão", result.message)
            
            self._update_status_display()
            
        except Exception as e:
            error_msg = f"Exceção durante conexão: {str(e)}"
            decision_ctx.add_criteria("exception", str(e), 1.0)
            decision_ctx.set_outcome(error_msg)
            decision_ctx.set_confidence(0.0)
            
            notify_error("Erro Crítico", error_msg)
    
    def _disconnect_mt5_secure(self, decision_ctx):
        """Desconecta do MT5 com logging"""
        try:
            success = self.mt5_authenticator.disconnect()
            
            decision_ctx.add_criteria("disconnect_success", success, 1.0)
            
            if success:
                self.connection_status = ConnectionStatus.DISCONNECTED
                self.auto_trading = False
                
                decision_ctx.set_outcome("Desconectado com sucesso")
                decision_ctx.set_confidence(0.9)
                
                notify_info("Desconexão MT5", "Desconectado com segurança")
            else:
                decision_ctx.set_outcome("Falha na desconexão")
                decision_ctx.set_confidence(0.3)
                
                notify_warning("Desconexão", "Possível falha na desconexão")
            
            self._update_status_display()
            
        except Exception as e:
            error_msg = f"Exceção durante desconexão: {str(e)}"
            decision_ctx.add_criteria("exception", str(e), 1.0)
            decision_ctx.set_outcome(error_msg)
            decision_ctx.set_confidence(0.0)
            
            notify_error("Erro na Desconexão", error_msg)
    
    def _toggle_auto_trading_with_logging(self):
        """Alterna auto trading com logging detalhado"""
        with self.decision_logger.decision_context(
            DecisionType.SYSTEM_OPERATION,
            "Alternar auto trading",
            DecisionLevel.CRITICAL
        ) as ctx:
            
            if not self.mt5_authenticator.is_connected():
                ctx.add_criteria("mt5_connected", False, 1.0)
                ctx.set_justification("Auto trading requer conexão ativa com MT5")
                ctx.set_outcome("Operação negada - MT5 não conectado")
                ctx.set_confidence(0.0)
                
                notify_warning("Auto Trading", "Conecte ao MT5 primeiro")
                return
            
            # Obter informações da conta para análise de risco
            account_info = get_mt5_account_info()
            
            ctx.add_criteria("mt5_connected", True, 1.0)
            ctx.add_criteria("current_auto_trading", self.auto_trading, 1.0)
            
            if account_info:
                ctx.add_criteria("account_balance", account_info['balance'], 0.8)
                ctx.add_criteria("margin_level", account_info['margin_level'], 0.9)
                ctx.add_criteria("trade_allowed", account_info['trade_allowed'], 1.0)
            
            # Alternar estado
            self.auto_trading = not self.auto_trading
            
            if self.auto_trading:
                # Verificações de segurança para ativar auto trading
                risk_checks = self._perform_risk_checks(account_info)
                
                ctx.add_criteria("risk_checks_passed", risk_checks['passed'], 1.0)
                
                if risk_checks['passed']:
                    ctx.set_justification(
                        f"Auto trading ativado após verificações de segurança. "
                        f"Saldo: {account_info['balance']}, Nível de margem: {account_info['margin_level']}%"
                    )
                    ctx.set_outcome("Auto trading ativado com segurança")
                    ctx.set_confidence(0.8)
                    
                    notify_success("Auto Trading", "Ativado com verificações de segurança")
                else:
                    self.auto_trading = False  # Reverter se falhou nas verificações
                    
                    ctx.set_justification(f"Falha nas verificações: {risk_checks['reason']}")
                    ctx.set_outcome("Auto trading negado por segurança")
                    ctx.set_confidence(0.2)
                    
                    notify_error("Auto Trading", f"Negado: {risk_checks['reason']}")
            else:
                ctx.set_justification("Usuário desativou auto trading")
                ctx.set_outcome("Auto trading desativado")
                ctx.set_confidence(0.9)
                
                notify_info("Auto Trading", "Desativado pelo usuário")
            
            self._update_status_display()
    
    def _perform_risk_checks(self, account_info: Dict) -> Dict[str, Any]:
        """Realiza verificações de risco antes de ativar auto trading"""
        if not account_info:
            return {'passed': False, 'reason': 'Informações da conta indisponíveis'}
        
        # Verificar se trading é permitido
        if not account_info.get('trade_allowed', False):
            return {'passed': False, 'reason': 'Trading não permitido na conta'}
        
        # Verificar nível de margem
        margin_level = account_info.get('margin_level', 0)
        if margin_level < 200:  # Menos de 200% é arriscado
            return {'passed': False, 'reason': f'Nível de margem muito baixo: {margin_level}%'}
        
        # Verificar saldo mínimo
        balance = account_info.get('balance', 0)
        if balance < 1000:  # Saldo mínimo de $1000
            return {'passed': False, 'reason': f'Saldo insuficiente: ${balance}'}
        
        return {'passed': True, 'reason': 'Todas as verificações passaram'}
    
    def _update_account_info_with_logging(self):
        """Atualiza informações da conta com logging"""
        try:
            account_info = get_mt5_account_info()
            
            if account_info:
                self.balance_card.update_value(f"${account_info['balance']:,.2f}")
                self.equity_card.update_value(f"${account_info['equity']:,.2f}")
                self.margin_card.update_value(f"${account_info['margin']:,.2f}")
                
                # Log da atualização
                log_system_decision(
                    "Atualização de informações da conta",
                    f"Saldo: ${account_info['balance']:,.2f}, Patrimônio: ${account_info['equity']:,.2f}",
                    "Informações atualizadas com sucesso",
                    DecisionLevel.LOW
                )
            else:
                # Dados simulados com log
                import random
                balance = random.uniform(10000, 50000)
                equity = balance + random.uniform(-1000, 1000)
                margin = random.uniform(100, 2000)
                
                self.balance_card.update_value(f"${balance:,.2f}")
                self.equity_card.update_value(f"${equity:,.2f}")
                self.margin_card.update_value(f"${margin:,.2f}")
                
                log_system_decision(
                    "Modo simulação - dados gerados",
                    "MT5 não disponível, usando dados simulados",
                    f"Saldo simulado: ${balance:,.2f}",
                    DecisionLevel.INFO
                )
            
        except Exception as e:
            notify_error("Erro de Dados", f"Falha ao obter informações: {str(e)}")
            
            log_system_decision(
                "Erro na atualização da conta",
                f"Exceção: {str(e)}",
                "Falha na obtenção de dados da conta",
                DecisionLevel.MEDIUM
            )
    
    def _update_decision_metrics(self):
        """Atualiza métricas de decisão"""
        try:
            stats = self.decision_logger.get_decision_statistics()
            
            if stats:
                self.decisions_count_card.update_value(str(stats.get('total_decisions', 0)))
                
                avg_confidence = stats.get('avg_confidence', 0) * 100
                self.avg_confidence_card.update_value(f"{avg_confidence:.1f}%")
                
                avg_time = stats.get('avg_execution_time', 0)
                self.avg_exec_time_card.update_value(f"{avg_time:.1f}ms")
            
        except Exception as e:
            # Silenciosamente falhar para não spam de logs
            pass
    
    def _update_status_display(self):
        """Atualiza display de status"""
        status = self.mt5_authenticator.get_connection_status()
        
        if status == ConnectionStatus.AUTHENTICATED:
            self.status_indicator.set_status('online')
            self.status_label.config(text="Conectado Seguro", fg=ModernTheme.SUCCESS)
            self.connect_btn.config(text="Desconectar", bg=ModernTheme.DANGER)
            
            # Mostrar detalhes da conexão
            account_info = get_mt5_account_info()
            if account_info:
                details = f"Login: {account_info['login']} | Servidor: {account_info['server']}"
                self.connection_details.config(text=details)
            
        else:
            self.status_indicator.set_status('offline')
            self.status_label.config(text="Desconectado", fg=ModernTheme.DANGER)
            self.connect_btn.config(text="Conectar MT5 Seguro", bg=ModernTheme.SUCCESS)
            self.connection_details.config(text="")
        
        # Atualizar botão auto trading
        if self.auto_trading:
            self.auto_trade_btn.config(text="Auto Trading: ON", 
                                      bg=ModernTheme.SUCCESS, 
                                      fg=ModernTheme.TEXT_PRIMARY)
        else:
            self.auto_trade_btn.config(text="Auto Trading: OFF", 
                                      bg=ModernTheme.BG_TERTIARY, 
                                      fg=ModernTheme.TEXT_SECONDARY)
    
    def _show_decision_logs(self):
        """Mostra janela com logs de decisão"""
        log_window = tk.Toplevel(self)
        log_window.title("Logs de Decisão")
        log_window.geometry("800x600")
        log_window.configure(bg=ModernTheme.BG_PRIMARY)
        
        # Criar área de texto com scroll
        text_frame = tk.Frame(log_window, bg=ModernTheme.BG_PRIMARY)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_area = tk.Text(text_frame, 
                           bg=ModernTheme.BG_SECONDARY,
                           fg=ModernTheme.TEXT_PRIMARY,
                           font=('Consolas', 10),
                           wrap=tk.WORD)
        
        scrollbar = tk.Scrollbar(text_frame, command=text_area.yview)
        text_area.config(yscrollcommand=scrollbar.set)
        
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Carregar logs recentes
        recent_decisions = self.decision_logger.get_recent_decisions(20)
        
        if recent_decisions:
            for decision in recent_decisions:
                log_text = (
                    f"[{decision.timestamp.strftime('%H:%M:%S')}] "
                    f"{decision.decision_type.value.upper()} - {decision.description}\n"
                    f"Justificativa: {decision.justification}\n"
                    f"Resultado: {decision.outcome}\n"
                    f"Confiança: {decision.confidence_score:.2f}\n"
                    f"Tempo: {decision.execution_time_ms:.2f}ms\n"
                    f"{'-'*60}\n\n"
                )
                text_area.insert(tk.END, log_text)
        else:
            text_area.insert(tk.END, "Nenhum log de decisão disponível.")
        
        text_area.config(state=tk.DISABLED)
    
    def _start_monitoring(self):
        """Inicia thread de monitoramento"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _monitoring_loop(self):
        """Loop de monitoramento em background"""
        while self.monitoring_active:
            try:
                # Atualizar métricas de decisão
                self._update_decision_metrics()
                
                # Verificar conexão se necessário
                if self.connection_status == ConnectionStatus.AUTHENTICATED:
                    if not self.mt5_authenticator.is_connected():
                        log_system_decision(
                            "Perda de conexão detectada",
                            "Conexão MT5 foi perdida durante monitoramento",
                            "Status atualizado para desconectado",
                            DecisionLevel.HIGH
                        )
                        self.connection_status = ConnectionStatus.DISCONNECTED
                        self.auto_trading = False
                        
                        # Atualizar UI na thread principal
                        self.after(0, self._update_status_display)
                
                time.sleep(5)  # Atualizar a cada 5 segundos
                
            except Exception as e:
                # Log de erro no monitoramento
                log_system_decision(
                    "Erro no monitoramento",
                    f"Exceção no loop de monitoramento: {str(e)}",
                    "Continuando monitoramento",
                    DecisionLevel.MEDIUM
                )
                time.sleep(10)  # Esperar mais tempo em caso de erro
    
    def destroy(self):
        """Cleanup ao destruir o widget"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1)
        
        log_system_decision(
            "Finalização do painel de controle",
            "Painel de trading sendo finalizado",
            "Cleanup realizado com sucesso",
            DecisionLevel.INFO
        )
        
        super().destroy()

class EnhancedMT5Dashboard(tk.Tk):
    """Dashboard MT5 aprimorado com logging e autenticação segura"""
    
    def __init__(self):
        super().__init__()
        
        # Configurações da janela
        self.title("MT5 Dashboard - Seguro & Monitorado")
        self.geometry("1400x900")
        self.configure(bg=ModernTheme.BG_PRIMARY)
        
        # Configurar tema
        ModernTheme.configure_ttk_style()
        
        # Inicializar sistemas
        self.decision_logger = get_decision_logger()
        
        # Criar interface
        self._create_interface()
        
        # Log de inicialização
        log_system_decision(
            "Inicialização do dashboard",
            "Dashboard MT5 com recursos de segurança e logging iniciado",
            "Sistema pronto para operação",
            DecisionLevel.INFO
        )
        
        # Protocolo de fechamento
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_interface(self):
        """Cria interface principal"""
        # Frame principal
        main_frame = tk.Frame(self, bg=ModernTheme.BG_PRIMARY)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Painel de controle seguro
        self.control_panel = SecureTradingControlPanel(main_frame)
        self.control_panel.pack(fill=tk.X, pady=(0, 10))
        
        # Área para outros componentes (posições, gráficos, etc.)
        content_frame = tk.Frame(main_frame, bg=ModernTheme.BG_SECONDARY, height=400)
        content_frame.pack(fill=tk.BOTH, expand=True)
        content_frame.pack_propagate(False)
        
        # Placeholder para conteúdo adicional
        placeholder_label = tk.Label(content_frame, 
                                   text="Área para gráficos, posições e análises\n(Componentes adicionais podem ser integrados aqui)",
                                   font=('Segoe UI', 12),
                                   bg=ModernTheme.BG_SECONDARY, 
                                   fg=ModernTheme.TEXT_MUTED)
        placeholder_label.pack(expand=True)
    
    def _on_closing(self):
        """Callback para fechamento da janela"""
        with self.decision_logger.decision_context(
            DecisionType.SYSTEM_OPERATION,
            "Fechamento do dashboard",
            DecisionLevel.MEDIUM
        ) as ctx:
            
            ctx.set_justification("Usuário solicitou fechamento do dashboard")
            
            # Desconectar MT5 se conectado
            if hasattr(self, 'control_panel'):
                if self.control_panel.mt5_authenticator.is_connected():
                    self.control_panel.mt5_authenticator.disconnect()
                    ctx.add_criteria("mt5_disconnected", True, 1.0)
                
                # Parar monitoramento
                self.control_panel.monitoring_active = False
            
            ctx.set_outcome("Dashboard fechado com segurança")
            ctx.set_confidence(0.9)
        
        self.destroy()

if __name__ == "__main__":
    # Criar e executar dashboard
    app = EnhancedMT5Dashboard()
    
    try:
        app.mainloop()
    except KeyboardInterrupt:
        print("\nDashboard interrompido pelo usuário")
    except Exception as e:
        print(f"Erro no dashboard: {e}")
        log_system_decision(
            "Erro crítico no dashboard",
            f"Exceção não tratada: {str(e)}",
            "Dashboard encerrado por erro",
            DecisionLevel.CRITICAL
        )
    finally:
        # Cleanup final
        try:
            disconnect_mt5()
        except:
            pass
        print("Dashboard finalizado")