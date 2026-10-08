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
# WebSocket removido - dados obtidos diretamente do MT5
from export_system import ExportManager
from settings_system import SettingsManager, SettingsDialog, ThemeManager
from user_preferences import UserPreferences, PreferencesDialog
from performance_optimization import get_performance_optimizer, performance_timer

# Importar componentes originais (assumindo que existem)
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

class TradingControlPanel(ResponsiveFrame):
    """Painel de controle de trading"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=ModernTheme.BG_SECONDARY, **kwargs)
        
        self.mt5_connected = False
        self.auto_trading = False
        
        self._create_widgets()
        self._update_status()
    
    def _create_widgets(self):
        """Cria widgets do painel"""
        # Título
        title_frame = tk.Frame(self, bg=ModernTheme.BG_SECONDARY, height=40)
        title_frame.pack(fill=tk.X, padx=15, pady=(15, 0))
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="Controle de Trading",
                font=('Segoe UI', 12, 'bold'),
                bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_PRIMARY).pack(side=tk.LEFT, pady=10)
        
        # Status de conexão
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
        
        # Botões de controle
        buttons_frame = tk.Frame(self, bg=ModernTheme.BG_SECONDARY)
        buttons_frame.pack(fill=tk.X, padx=15, pady=10)
        
        self.connect_btn = tk.Button(buttons_frame, text="Conectar MT5",
                                    font=('Segoe UI', 9),
                                    bg=ModernTheme.SUCCESS, fg=ModernTheme.TEXT_PRIMARY,
                                    borderwidth=0, pady=8, padx=20,
                                    command=self._toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.auto_trade_btn = tk.Button(buttons_frame, text="Auto Trading: OFF",
                                       font=('Segoe UI', 9),
                                       bg=ModernTheme.BG_TERTIARY, fg=ModernTheme.TEXT_SECONDARY,
                                       borderwidth=0, pady=8, padx=20,
                                       command=self._toggle_auto_trading)
        self.auto_trade_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # WebSocket removido - dados obtidos diretamente do MT5
        
        # Informações da conta
        account_frame = tk.LabelFrame(self, text="Informações da Conta",
                                     font=('Segoe UI', 9, 'bold'),
                                     bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_PRIMARY,
                                     borderwidth=1, relief='solid')
        account_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Cards de métricas
        metrics_container = tk.Frame(account_frame, bg=ModernTheme.BG_SECONDARY)
        metrics_container.pack(fill=tk.X, padx=10, pady=10)
        
        self.balance_card = MetricCard(metrics_container, "Saldo", "$0.00")
        self.balance_card.pack(side=tk.LEFT, padx=(0, 10))
        
        self.equity_card = MetricCard(metrics_container, "Patrimônio", "$0.00")
        self.equity_card.pack(side=tk.LEFT, padx=(0, 10))
        
        self.margin_card = MetricCard(metrics_container, "Margem", "$0.00")
        self.margin_card.pack(side=tk.LEFT)
    
    def _toggle_connection(self):
        """Alterna conexão MT5"""
        if not self.mt5_connected:
            self._connect_mt5()
        else:
            self._disconnect_mt5()
    
    def _connect_mt5(self):
        """Conecta ao MT5"""
        try:
            if mt5 and mt5.initialize():
                self.mt5_connected = True
                notify_success("Conexão MT5", "Conectado com sucesso ao MetaTrader 5")
            else:
                # Modo simulação
                self.mt5_connected = True
                notify_info("Modo Simulação", "Executando em modo simulação (MT5 não disponível)")
            
            self._update_status()
            self._update_account_info()
            
        except Exception as e:
            notify_error("Erro de Conexão", f"Falha ao conectar: {str(e)}")
    
    def _disconnect_mt5(self):
        """Desconecta do MT5"""
        try:
            if mt5:
                mt5.shutdown()
            
            self.mt5_connected = False
            self.auto_trading = False
            
            notify_info("Desconexão MT5", "Desconectado do MetaTrader 5")
            self._update_status()
            
        except Exception as e:
            notify_error("Erro de Desconexão", f"Falha ao desconectar: {str(e)}")
    
    def _toggle_auto_trading(self):
        """Alterna auto trading"""
        if not self.mt5_connected:
            notify_warning("Auto Trading", "Conecte ao MT5 primeiro")
            return
        
        self.auto_trading = not self.auto_trading
        
        if self.auto_trading:
            notify_success("Auto Trading", "Auto trading ativado")
        else:
            notify_info("Auto Trading", "Auto trading desativado")
        
        self._update_status()
    
    def _update_status(self):
        """Atualiza status visual"""
        if self.mt5_connected:
            self.status_indicator.set_status('online')
            self.status_label.config(text="Conectado", fg=ModernTheme.SUCCESS)
            self.connect_btn.config(text="Desconectar", bg=ModernTheme.DANGER)
        else:
            self.status_indicator.set_status('offline')
            self.status_label.config(text="Desconectado", fg=ModernTheme.DANGER)
            self.connect_btn.config(text="Conectar MT5", bg=ModernTheme.SUCCESS)
        
        # Atualizar botão auto trading
        if self.auto_trading:
            self.auto_trade_btn.config(text="Auto Trading: ON", 
                                      bg=ModernTheme.SUCCESS, 
                                      fg=ModernTheme.TEXT_PRIMARY)
        else:
            self.auto_trade_btn.config(text="Auto Trading: OFF", 
                                      bg=ModernTheme.BG_TERTIARY, 
                                      fg=ModernTheme.TEXT_SECONDARY)
    
    def _update_account_info(self):
        """Atualiza informações da conta"""
        try:
            if mt5 and self.mt5_connected:
                account_info = mt5.account_info()
                if account_info:
                    self.balance_card.update_value(f"${account_info.balance:,.2f}")
                    self.equity_card.update_value(f"${account_info.equity:,.2f}")
                    self.margin_card.update_value(f"${account_info.margin:,.2f}")
                    return
            
            # Dados simulados
            import random
            balance = random.uniform(10000, 50000)
            equity = balance + random.uniform(-1000, 1000)
            margin = random.uniform(100, 2000)
            
            self.balance_card.update_value(f"${balance:,.2f}")
            self.equity_card.update_value(f"${equity:,.2f}")
            self.margin_card.update_value(f"${margin:,.2f}")
            
        except Exception as e:
            notify_error("Erro de Dados", f"Falha ao obter informações da conta: {str(e)}")
    
    # Funções WebSocket removidas - dados obtidos diretamente do MT5

class PositionsPanel(ResponsiveFrame):
    """Painel de posições"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=ModernTheme.BG_SECONDARY, **kwargs)
        
        self._create_widgets()
        self._update_positions()
    
    def _create_widgets(self):
        """Cria widgets do painel"""
        # Título
        title_frame = tk.Frame(self, bg=ModernTheme.BG_SECONDARY, height=40)
        title_frame.pack(fill=tk.X, padx=15, pady=(15, 0))
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="Posições Abertas",
                font=('Segoe UI', 12, 'bold'),
                bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_PRIMARY).pack(side=tk.LEFT, pady=10)
        
        # Botões de controle
        controls_frame = tk.Frame(title_frame, bg=ModernTheme.BG_SECONDARY)
        controls_frame.pack(side=tk.RIGHT, pady=5)
        
        # Indicador WebSocket removido - dados obtidos diretamente do MT5
        
        # Botão atualizar
        refresh_btn = tk.Button(controls_frame, text="🔄",
                               font=('Segoe UI', 10),
                               bg=ModernTheme.BG_TERTIARY, fg=ModernTheme.TEXT_PRIMARY,
                               borderwidth=0, width=3,
                               command=self._update_positions)
        refresh_btn.pack(side=tk.RIGHT, padx=2)
        
        # Botão Export
        export_btn = tk.Button(controls_frame, text="📊",
                              font=('Segoe UI', 10),
                              bg=ModernTheme.BG_TERTIARY, fg=ModernTheme.TEXT_PRIMARY,
                              borderwidth=0, width=3,
                              command=self._export_positions)
        export_btn.pack(side=tk.RIGHT, padx=2)
        
        # Treeview para posições
        tree_frame = tk.Frame(self, bg=ModernTheme.BG_SECONDARY)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Configurar estilo
        ModernTheme.configure_ttk_style()
        
        # Colunas
        columns = ('Symbol', 'Type', 'Volume', 'Price', 'Current', 'Profit', 'Time')
        
        self.positions_tree = ttk.Treeview(tree_frame, columns=columns, show='headings',
                                          style='Modern.Treeview')
        
        # Configurar cabeçalhos
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=100, anchor='center')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", 
                                 command=self.positions_tree.yview,
                                 style='Modern.Vertical.TScrollbar')
        self.positions_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack
        self.positions_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Menu de contexto
        self.context_menu = tk.Menu(self, tearoff=0, bg=ModernTheme.BG_TERTIARY, 
                                   fg=ModernTheme.TEXT_PRIMARY, borderwidth=0)
        self.context_menu.add_command(label="Fechar Posição", command=self._close_position)
        self.context_menu.add_command(label="Modificar SL/TP", command=self._modify_position)
        
        self.positions_tree.bind("<Button-3>", self._show_context_menu)
    
    def _show_context_menu(self, event):
        """Mostra menu de contexto"""
        item = self.positions_tree.selection()[0] if self.positions_tree.selection() else None
        if item:
            self.context_menu.post(event.x_root, event.y_root)
    
    def _close_position(self):
        """Fecha posição selecionada"""
        selection = self.positions_tree.selection()
        if selection:
            notify_info("Fechar Posição", "Funcionalidade em desenvolvimento")
    
    def _modify_position(self):
        """Modifica posição selecionada"""
        selection = self.positions_tree.selection()
        if selection:
            notify_info("Modificar Posição", "Funcionalidade em desenvolvimento")
    
    def _update_positions(self):
        """Atualiza lista de posições"""
        # Limpar dados existentes
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)
        
        try:
            if mt5:
                positions = mt5.positions_get()
                if positions:
                    for pos in positions:
                        self.positions_tree.insert('', 'end', values=(
                            pos.symbol,
                            'Buy' if pos.type == 0 else 'Sell',
                            pos.volume,
                            f"{pos.price_open:.5f}",
                            f"{pos.price_current:.5f}",
                            f"{pos.profit:.2f}",
                            datetime.fromtimestamp(pos.time).strftime('%H:%M:%S')
                        ))
                    return
            
            # Dados simulados
            import random
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
            
            for i in range(random.randint(0, 5)):
                symbol = random.choice(symbols)
                trade_type = random.choice(['Buy', 'Sell'])
                volume = round(random.uniform(0.01, 1.0), 2)
                price_open = random.uniform(1.0, 1.5)
                price_current = price_open + random.uniform(-0.01, 0.01)
                profit = (price_current - price_open) * volume * 100000
                if trade_type == 'Sell':
                    profit = -profit
                
                self.positions_tree.insert('', 'end', values=(
                    symbol, trade_type, volume,
                    f"{price_open:.5f}", f"{price_current:.5f}",
                    f"{profit:.2f}", datetime.now().strftime('%H:%M:%S')
                ))
                
        except Exception as e:
            notify_error("Erro de Posições", f"Falha ao obter posições: {str(e)}")
    
    # Indicador WebSocket removido - dados obtidos diretamente do MT5
    
    def _export_positions(self):
        """Exporta posições do painel"""
        try:
            # Obter referência ao dashboard principal
            dashboard = self.master.master.master
            if hasattr(dashboard, 'export_manager'):
                dashboard.export_manager.quick_export_positions()
            else:
                notify_warning("Exportação", "Sistema de exportação não disponível")
        except Exception as e:
            notify_error("Erro", f"Falha na exportação: {e}")

class AnalysisPanel(ResponsiveFrame):
    """Painel de análise com gráficos"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=ModernTheme.BG_SECONDARY, **kwargs)
        
        self.charts = {}
        self._create_widgets()
        self._start_data_simulation()
    
    def _create_widgets(self):
        """Cria widgets do painel"""
        # Título
        title_frame = tk.Frame(self, bg=ModernTheme.BG_SECONDARY, height=40)
        title_frame.pack(fill=tk.X, padx=15, pady=(15, 0))
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="Análise de Performance",
                font=('Segoe UI', 12, 'bold'),
                bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_PRIMARY).pack(side=tk.LEFT, pady=10)
        
        # Container para gráficos
        charts_container = tk.Frame(self, bg=ModernTheme.BG_SECONDARY)
        charts_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Gráfico de P&L
        pnl_frame = tk.LabelFrame(charts_container, text="Profit & Loss",
                                 font=('Segoe UI', 9, 'bold'),
                                 bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_PRIMARY)
        pnl_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.charts['pnl'] = LiveChart(pnl_frame, width=500, height=150, max_points=50)
        self.charts['pnl'].pack(padx=10, pady=10)
        
        # Gráfico de Drawdown
        dd_frame = tk.LabelFrame(charts_container, text="Drawdown",
                                font=('Segoe UI', 9, 'bold'),
                                bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_PRIMARY)
        dd_frame.pack(fill=tk.BOTH, expand=True)
        
        self.charts['drawdown'] = LiveChart(dd_frame, width=500, height=150, max_points=50)
        self.charts['drawdown'].pack(padx=10, pady=10)
    
    def _start_data_simulation(self):
        """Inicia simulação de dados"""
        def simulate_data():
            import random
            
            while True:
                try:
                    # Simular P&L
                    pnl_value = random.uniform(-100, 200)
                    self.charts['pnl'].add_point(pnl_value)
                    
                    # Simular Drawdown
                    dd_value = random.uniform(0, 50)
                    self.charts['drawdown'].add_point(dd_value)
                    
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Erro na simulação: {e}")
                    break
        
        thread = threading.Thread(target=simulate_data, daemon=True)
        thread.start()

class MT5DashboardIntegrated:
    """Dashboard MT5 integrado com todas as melhorias"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.notification_manager = get_notification_manager()
        
        # Configurar janela principal
        self._setup_window()
        
        # Configurar tema
        ModernTheme.configure_ttk_style()
        
        # Criar interface
        self._create_interface()
        
        # Inicializar sistemas
        try:
            self.settings_manager = SettingsManager()
            self.theme_manager = ThemeManager(self.settings_manager)
            self.user_preferences = UserPreferences(self.settings_manager)
        except Exception as e:
            print(f"Erro ao inicializar sistemas: {e}")
            self.settings_manager = None
            self.theme_manager = None
            self.user_preferences = None
        
        # Inicializar otimizador de performance
        try:
            self.performance_optimizer = get_performance_optimizer()
            self.performance_optimizer.optimize_dashboard(self)
        except Exception as e:
            print(f"Erro ao inicializar otimizador de performance: {e}")
            self.performance_optimizer = None
        
        # WebSocket removido - dados obtidos diretamente do MT5
        
        # Inicializar sistema de exportação
        self.export_manager = ExportManager(self)
        
        # Aplicar preferências do usuário
        self._apply_user_preferences()
        
        # Observar mudanças de configuração
        self.settings_manager.add_observer(self._on_setting_changed)
        
        # Iniciar atualizações
        self._start_updates()
    
    def _setup_window(self):
        """Configura janela principal"""
        self.root.title("MT5 Dashboard - Versão Integrada")
        
        # Aplicar configurações salvas de janela
        if hasattr(self, 'settings_manager') and self.settings_manager.get('remember_window_size', True):
            saved_geometry = self.settings_manager.get('window_geometry', '1600x1000+100+100')
            self.root.geometry(saved_geometry)
        else:
            self.root.geometry("1600x1000")
        
        self.root.minsize(1200, 800)
        
        # Aplicar opacidade se disponível
        if hasattr(self, 'settings_manager'):
            opacity = self.settings_manager.get('window_opacity', 1.0)
            self.root.attributes('-alpha', opacity)
        self.root.configure(bg=ModernTheme.BG_PRIMARY)
        
        # Ícone (se disponível)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
        
        # Protocolo de fechamento
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_interface(self):
        """Cria interface principal"""
        # Frame principal
        main_frame = tk.Frame(self.root, bg=ModernTheme.BG_PRIMARY)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Barra superior
        self._create_top_bar(main_frame)
        
        # Container principal
        content_frame = tk.Frame(main_frame, bg=ModernTheme.BG_PRIMARY)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Layout responsivo com PanedWindow
        main_paned = tk.PanedWindow(content_frame, orient=tk.HORIZONTAL, 
                                   bg=ModernTheme.BG_PRIMARY, sashwidth=5,
                                   sashrelief=tk.RAISED)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Painel esquerdo
        left_frame = tk.Frame(main_paned, bg=ModernTheme.BG_PRIMARY)
        main_paned.add(left_frame, minsize=400)
        
        # Painel direito
        right_frame = tk.Frame(main_paned, bg=ModernTheme.BG_PRIMARY)
        main_paned.add(right_frame, minsize=400)
        
        # Criar painéis
        self._create_left_panels(left_frame)
        self._create_right_panels(right_frame)
    
    def _create_top_bar(self, parent):
        """Cria barra superior"""
        top_bar = tk.Frame(parent, bg=ModernTheme.BG_SECONDARY, height=60)
        top_bar.pack(fill=tk.X)
        top_bar.pack_propagate(False)
        
        # Logo/Título
        title_frame = tk.Frame(top_bar, bg=ModernTheme.BG_SECONDARY)
        title_frame.pack(side=tk.LEFT, padx=20, pady=15)
        
        tk.Label(title_frame, text="MT5 Dashboard",
                font=('Segoe UI', 16, 'bold'),
                bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_PRIMARY).pack(side=tk.LEFT)
        
        # Status global
        status_frame = tk.Frame(top_bar, bg=ModernTheme.BG_SECONDARY)
        status_frame.pack(side=tk.RIGHT, padx=20, pady=15)
        
        # Indicador de notificações
        self.notification_btn = tk.Button(status_frame, text="🔔",
                                         font=('Segoe UI', 14),
                                         bg=ModernTheme.BG_TERTIARY, fg=ModernTheme.TEXT_PRIMARY,
                                         borderwidth=0, width=3,
                                         command=self._toggle_notifications)
        self.notification_btn.pack(side=tk.RIGHT, padx=5)
        
        # Contador de notificações
        self.notification_count = tk.Label(status_frame, text="0",
                                          font=('Segoe UI', 8, 'bold'),
                                          bg=ModernTheme.DANGER, fg=ModernTheme.TEXT_PRIMARY,
                                          padx=4, pady=1)
        self.notification_count.pack(side=tk.RIGHT, padx=(0, 5))
        
        # Criar menu de exportação
        self._create_export_menu(status_frame)
        
        # Relógio
        self.clock_label = tk.Label(status_frame, text="",
                                   font=('Segoe UI', 10),
                                   bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_SECONDARY)
        self.clock_label.pack(side=tk.RIGHT, padx=10)
    
    def _create_left_panels(self, parent):
        """Cria painéis do lado esquerdo"""
        # Notebook para abas
        notebook = ttk.Notebook(parent, style='Modern.TNotebook')
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Aba de controle
        control_frame = tk.Frame(notebook, bg=ModernTheme.BG_PRIMARY)
        notebook.add(control_frame, text="Controle")
        
        self.trading_panel = TradingControlPanel(control_frame)
        self.trading_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Aba de posições
        positions_frame = tk.Frame(notebook, bg=ModernTheme.BG_PRIMARY)
        notebook.add(positions_frame, text="Posições")
        
        self.positions_panel = PositionsPanel(positions_frame)
        self.positions_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _create_right_panels(self, parent):
        """Cria painéis do lado direito"""
        # PanedWindow vertical
        right_paned = tk.PanedWindow(parent, orient=tk.VERTICAL,
                                    bg=ModernTheme.BG_PRIMARY, sashwidth=5)
        right_paned.pack(fill=tk.BOTH, expand=True)
        
        # Painel de análise
        analysis_frame = tk.Frame(right_paned, bg=ModernTheme.BG_PRIMARY)
        right_paned.add(analysis_frame, minsize=300)
        
        self.analysis_panel = AnalysisPanel(analysis_frame)
        self.analysis_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Painel de notificações
        notifications_frame = tk.Frame(right_paned, bg=ModernTheme.BG_PRIMARY)
        right_paned.add(notifications_frame, minsize=200)
        
        self.notification_panel = NotificationPanel(notifications_frame, self.notification_manager)
        self.notification_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _toggle_notifications(self):
        """Alterna visibilidade do painel de notificações"""
        # Implementar toggle do painel
        notify_info("Notificações", "Painel de notificações sempre visível nesta versão")
    
    def _start_updates(self):
        """Inicia atualizações periódicas"""
        def update_clock():
            current_time = datetime.now().strftime('%H:%M:%S')
            self.clock_label.config(text=current_time)
            
            # Atualizar contador de notificações
            unread_count = self.notification_manager.get_unread_count()
            self.notification_count.config(text=str(unread_count))
            
            # Agendar próxima atualização
            self.root.after(1000, update_clock)
        
        def periodic_updates():
            start_time = time.time()
            try:
                # Atualizar dados de trading
                if hasattr(self, 'trading_panel'):
                    if self.trading_panel.mt5_connected:
                        self.trading_panel._update_account_info()
                
                # Atualizar posições
                if hasattr(self, 'positions_panel'):
                    self.positions_panel._update_positions()
                
            except Exception as e:
                print(f"Erro nas atualizações periódicas: {e}")
            finally:
                # Registrar tempo de atualização se otimizador disponível
                if hasattr(self, 'performance_optimizer') and self.performance_optimizer:
                    duration = time.time() - start_time
                    self.performance_optimizer.monitor.record_update_time(duration)
            
            # Agendar próxima atualização
            self.root.after(30000, periodic_updates)  # 30 segundos
        
        # Iniciar atualizações
        update_clock()
        periodic_updates()
        
        # WebSocket removido - dados obtidos diretamente do MT5
        
        # Notificação de boas-vindas
        self.root.after(1000, lambda: notify_success(
            "Dashboard Iniciado", 
            "MT5 Dashboard integrado carregado com sucesso"
        ))
    
    def _on_closing(self):
        """Callback para fechamento da aplicação"""
        try:
            # Salvar geometria da janela
            if hasattr(self, 'settings_manager'):
                geometry = self.root.geometry()
                self.settings_manager.set('window_geometry', geometry)
            
            # Desconectar MT5 se conectado
            if hasattr(self, 'trading_panel') and self.trading_panel.mt5_connected:
                self.trading_panel._disconnect_mt5()
            
            # Salvar configurações
            if hasattr(self, 'settings_manager'):
                self.settings_manager.save_settings()
            
            self.notification_manager._save_settings()
            
            notify_info("Sistema", "Dashboard finalizado")
            
        except Exception as e:
            print(f"Erro ao fechar: {e}")
        finally:
            self.root.destroy()
    
    def run(self):
        """Executa o dashboard"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self._on_closing()
    
    def _apply_theme_changes(self):
        """Aplica mudanças de tema"""
        try:
            ModernTheme.configure_ttk_style()
            self.root.configure(bg=ModernTheme.BG_PRIMARY)
        except Exception as e:
            print(f"Erro ao aplicar tema: {e}")
    
    def _update_refresh_interval(self, interval):
        """Atualiza intervalo de atualização"""
        # Implementar lógica de atualização de intervalo
        pass
    
    def _toggle_theme(self):
        """Alterna entre temas"""
        if hasattr(self, 'settings_manager'):
            current_theme = self.settings_manager.get('theme_mode', 'dark')
            new_theme = 'light' if current_theme == 'dark' else 'dark'
            self.settings_manager.set('theme_mode', new_theme)
        else:
            self.notification_manager.add_notification(
                "warning", "Tema", "Sistema de configurações não disponível"
            )
    
    def _show_preferences(self):
        """Exibir diálogo de preferências do usuário"""
        if hasattr(self, 'user_preferences') and self.user_preferences:
            try:
                dialog = PreferencesDialog(self.root, self.user_preferences)
                dialog.show()
                self.notification_manager.add_notification(
                     "Preferências atualizadas com sucesso",
                     level="info",
                     category="system"
                 )
            except Exception as e:
                self.notification_manager.add_notification(
                     f"Erro ao abrir preferências: {str(e)}",
                     level="error",
                     category="system"
                 )
        else:
            self.notification_manager.add_notification(
                 "Sistema de preferências não disponível",
                 level="warning",
                 category="system"
             )
    
    def _apply_user_preferences(self):
        """Aplicar preferências do usuário ao dashboard"""
        if not hasattr(self, 'user_preferences') or not self.user_preferences:
            return
        
        try:
            # Aplicar configurações de layout
            layout_prefs = self.user_preferences.get_preference('layout')
            if layout_prefs:
                # Aplicar visibilidade dos painéis
                panels_visible = layout_prefs.get('panels_visible', {})
                for panel_name, visible in panels_visible.items():
                    self._toggle_panel_visibility(panel_name, visible)
                
                # Aplicar posições dos painéis
                panel_positions = layout_prefs.get('panel_positions', {})
                for panel_name, position in panel_positions.items():
                    self._set_panel_position(panel_name, position)
            
            # Aplicar configurações de trading
            trading_prefs = self.user_preferences.get_preference('trading')
            if trading_prefs:
                # Configurar símbolos favoritos
                favorite_symbols = trading_prefs.get('favorite_symbols', [])
                self._update_favorite_symbols(favorite_symbols)
                
                # Configurar alertas
                alert_settings = trading_prefs.get('alert_settings', {})
                self._update_alert_settings(alert_settings)
        
        except Exception as e:
            print(f"Erro ao aplicar preferências: {e}")
    
    def _toggle_panel_visibility(self, panel_name, visible):
        """Alternar visibilidade de um painel"""
        # Implementar lógica para mostrar/ocultar painéis
        try:
            if panel_name == 'positions' and hasattr(self, 'positions_panel'):
                if visible:
                    self.positions_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                else:
                    self.positions_panel.pack_forget()
            elif panel_name == 'analysis' and hasattr(self, 'analysis_panel'):
                if visible:
                    self.analysis_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                else:
                    self.analysis_panel.pack_forget()
        except Exception as e:
            print(f"Erro ao alternar visibilidade do painel {panel_name}: {e}")
    
    def _set_panel_position(self, panel_name, position):
        """Definir posição de um painel"""
        # Implementar lógica para reposicionar painéis
        try:
            # Lógica básica de reposicionamento
            print(f"Reposicionando painel {panel_name} para posição {position}")
        except Exception as e:
            print(f"Erro ao reposicionar painel {panel_name}: {e}")
    
    def _update_favorite_symbols(self, symbols):
        """Atualizar símbolos favoritos"""
        try:
            # Implementar lógica para atualizar símbolos favoritos
            if hasattr(self, 'positions_panel'):
                # Atualizar lista de símbolos no painel de posições
                print(f"Símbolos favoritos atualizados: {symbols}")
        except Exception as e:
            print(f"Erro ao atualizar símbolos favoritos: {e}")
    
    def _update_alert_settings(self, settings):
        """Atualizar configurações de alertas"""
        try:
            # Implementar lógica para configurar alertas
            print(f"Configurações de alertas atualizadas: {settings}")
        except Exception as e:
            print(f"Erro ao atualizar configurações de alertas: {e}")
    
    # Funções WebSocket removidas - dados obtidos diretamente do MT5
    
    def _create_export_menu(self, parent):
        """Cria menu de exportação"""
        export_frame = tk.Frame(parent, bg=ModernTheme.BG_SECONDARY)
        export_frame.pack(side=tk.RIGHT, padx=10)
        
        # Botão de menu de exportação
        export_btn = tk.Button(
            export_frame,
            text="📊 Exportar",
            bg=ModernTheme.BG_TERTIARY,
            fg=ModernTheme.TEXT_PRIMARY,
            font=('Segoe UI', 9),
            borderwidth=0,
            padx=10,
            command=self._show_export_menu
        )
        export_btn.pack(side=tk.LEFT, padx=2)
        
        # Botão de exportação rápida
        quick_export_btn = tk.Button(
            export_frame,
            text="⚡ Export Rápido",
            bg=ModernTheme.PRIMARY,
            fg=ModernTheme.TEXT_PRIMARY,
            font=('Segoe UI', 9),
            borderwidth=0,
            padx=10,
            command=self._quick_export
        )
        quick_export_btn.pack(side=tk.LEFT, padx=2)
        
        # Botão de relatório
        report_btn = tk.Button(
            export_frame,
            text="📋 Relatório",
            bg=ModernTheme.SUCCESS,
            fg=ModernTheme.TEXT_PRIMARY,
            font=('Segoe UI', 9),
            borderwidth=0,
            padx=10,
            command=self._generate_report
        )
        report_btn.pack(side=tk.LEFT, padx=2)
        
        # Botão de configurações
        settings_btn = tk.Button(
            export_frame,
            text="⚙️ Config",
            bg=ModernTheme.BG_TERTIARY,
            fg=ModernTheme.TEXT_PRIMARY,
            font=('Segoe UI', 9),
            borderwidth=0,
            padx=10,
            command=self._show_settings
        )
        settings_btn.pack(side=tk.LEFT, padx=2)
        
        # Botão de tema
        theme_btn = tk.Button(
            export_frame,
            text="🎨 Tema",
            bg=ModernTheme.BG_TERTIARY,
            fg=ModernTheme.TEXT_PRIMARY,
            font=('Segoe UI', 9),
            borderwidth=0,
            padx=10,
            command=self._toggle_theme
        )
        theme_btn.pack(side=tk.LEFT, padx=2)
        
        # Botão de preferências
        prefs_btn = tk.Button(
            export_frame,
            text="👤 Preferências",
            bg=ModernTheme.BG_TERTIARY,
            fg=ModernTheme.TEXT_PRIMARY,
            font=('Segoe UI', 9),
            borderwidth=0,
            padx=10,
            command=self._show_preferences
        )
        prefs_btn.pack(side=tk.LEFT, padx=2)
        
        # Botão de performance
        perf_btn = tk.Button(
            export_frame,
            text="📊 Performance",
            bg=ModernTheme.BG_TERTIARY,
            fg=ModernTheme.TEXT_PRIMARY,
            font=('Segoe UI', 9),
            borderwidth=0,
            padx=10,
            command=self._show_performance_report
        )
        perf_btn.pack(side=tk.LEFT, padx=2)
    
    def _show_export_menu(self):
        """Mostra menu de exportação completo"""
        if hasattr(self, 'export_manager'):
            self.export_manager.show_export_dialog()
        else:
            notify_warning("Exportação", "Sistema de exportação não inicializado")
    
    def _quick_export(self):
        """Exportação rápida de posições"""
        if hasattr(self, 'export_manager'):
            self.export_manager.quick_export_positions()
        else:
            notify_warning("Exportação", "Sistema de exportação não inicializado")
    
    def _generate_report(self):
        """Gera relatório de trading"""
        if hasattr(self, 'export_manager'):
            self.export_manager.generate_trading_report()
        else:
            notify_warning("Exportação", "Sistema de exportação não inicializado")
    
    def _show_settings(self):
        """Mostra diálogo de configurações"""
        if hasattr(self, 'settings_manager'):
            SettingsDialog(self.root, self.settings_manager)
        else:
            self.notification_manager.add_notification(
                "warning", "Configurações", "Sistema de configurações não inicializado"
            )
    
    def _on_setting_changed(self, key: str, old_value, new_value):
        """Callback para mudanças de configuração"""
        try:
            if key == 'window_opacity':
                self.root.attributes('-alpha', new_value)
            elif key == 'theme_mode':
                self._apply_theme_changes()
                self.notification_manager.add_notification(
                    "info", "Tema", f"Tema alterado para: {new_value}"
                )
            elif key == 'refresh_interval':
                self._update_refresh_interval(new_value)
                self.notification_manager.add_notification(
                    "info", "Configuração", f"Intervalo de atualização: {new_value}s"
                )
            elif key == 'auto_trading':
                self.notification_manager.add_notification(
                    "info", "Trading", f"Trading automático: {'Ativado' if new_value else 'Desativado'}"
                )
        except Exception as e:
            self.notification_manager.add_notification(
                "error", "Configuração", f"Erro ao aplicar configuração: {e}"
            )
    
    def _show_performance_report(self):
        """Mostra relatório de performance do sistema"""
        if hasattr(self, 'performance_optimizer') and self.performance_optimizer:
            try:
                report = self.performance_optimizer.get_performance_report()
                
                # Criar janela de relatório
                report_window = tk.Toplevel(self.root)
                report_window.title("Relatório de Performance")
                report_window.geometry("600x400")
                report_window.configure(bg=ModernTheme.BG_SECONDARY)
                
                # Texto do relatório
                text_widget = tk.Text(
                    report_window,
                    bg=ModernTheme.BG_TERTIARY,
                    fg=ModernTheme.TEXT_PRIMARY,
                    font=('Consolas', 10),
                    wrap=tk.WORD
                )
                text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # Inserir dados do relatório
                text_widget.insert(tk.END, "=== RELATÓRIO DE PERFORMANCE ===\n\n")
                text_widget.insert(tk.END, f"Tempo médio de atualização: {report.get('average_update_time', 0):.4f}s\n")
                text_widget.insert(tk.END, f"Tendência de memória: {report.get('memory_trend', 'N/A')}\n")
                text_widget.insert(tk.END, f"Tamanho do cache: {report.get('cache_size', 0)} itens\n")
                text_widget.insert(tk.END, f"Otimização ativa: {'Sim' if report.get('optimization_enabled', False) else 'Não'}\n")
                text_widget.insert(tk.END, f"Última atualização: {datetime.now().strftime('%H:%M:%S')}\n\n")
                
                # Informações de memória detalhadas
                memory_info = report.get('memory_info', {})
                if memory_info:
                    text_widget.insert(tk.END, "=== INFORMAÇÕES DE MEMÓRIA ===\n")
                    text_widget.insert(tk.END, f"RSS: {memory_info.get('rss_mb', 0):.2f} MB\n")
                    text_widget.insert(tk.END, f"VMS: {memory_info.get('vms_mb', 0):.2f} MB\n")
                    text_widget.insert(tk.END, f"Percentual: {memory_info.get('percent', 0):.2f}%\n")
                    text_widget.insert(tk.END, f"Objetos GC: {memory_info.get('gc_objects', 0)}\n")
                
                text_widget.config(state=tk.DISABLED)
                
                if hasattr(self, 'notification_manager'):
                    self.notification_manager.info("Performance", "Relatório de performance gerado")
                
            except Exception as e:
                if hasattr(self, 'notification_manager'):
                    self.notification_manager.error("Performance", f"Erro ao gerar relatório: {e}")
        else:
            if hasattr(self, 'notification_manager'):
                self.notification_manager.warning("Performance", "Sistema de otimização de performance não disponível")

# Função principal
def main():
    """Função principal"""
    try:
        dashboard = MT5DashboardIntegrated()
        dashboard.run()
    except Exception as e:
        print(f"Erro ao iniciar dashboard: {e}")
        messagebox.showerror("Erro", f"Falha ao iniciar dashboard: {str(e)}")

if __name__ == "__main__":
    main()