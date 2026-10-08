import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from dataclasses import asdict
import pandas as pd
from mt5_connector import MT5Connector
from mt5_order_manager import MT5OrderManager, OrderType
from mt5_data_provider import MT5DataProvider
from mt5_risk_manager import MT5RiskManager, RiskLevel
from mt5_signal_integrator import MT5SignalIntegrator, SignalConfig, SignalAction
from mt5_logger import MT5Logger, LogLevel, EventType
import MetaTrader5 as mt5

class ResponsiveFrame(ttk.Frame):
    """Frame responsivo que se adapta ao redimensionamento"""
    
    def __init__(self, parent, min_width=200, min_height=100, **kwargs):
        super().__init__(parent, **kwargs)
        self.min_width = min_width
        self.min_height = min_height
        self.bind('<Configure>', self._on_configure)
    
    def _on_configure(self, event):
        """Manipula redimensionamento do frame"""
        if event.widget == self:
            width = max(event.width, self.min_width)
            height = max(event.height, self.min_height)
            self.configure(width=width, height=height)

class ModernTheme:
    """Tema moderno e profissional para o dashboard"""
    
    # Paleta de cores moderna
    COLORS = {
        'bg_primary': '#1e1e1e',      # Fundo principal escuro
        'bg_secondary': '#2d2d2d',    # Fundo secundário
        'bg_tertiary': '#3d3d3d',     # Fundo terciário
        'accent_primary': '#0078d4',   # Azul Microsoft
        'accent_secondary': '#106ebe', # Azul mais escuro
        'success': '#107c10',         # Verde sucesso
        'warning': '#ff8c00',         # Laranja aviso
        'error': '#d13438',           # Vermelho erro
        'text_primary': '#ffffff',    # Texto principal
        'text_secondary': '#cccccc',  # Texto secundário
        'text_muted': '#999999',      # Texto esmaecido
        'border': '#484848',          # Bordas
        'hover': '#404040'            # Hover state
    }
    
    @classmethod
    def configure_styles(cls, style: ttk.Style):
        """Configura estilos modernos para widgets TTK"""
        
        # Configurar tema base
        style.theme_use('clam')
        
        # Frame styles
        style.configure('Modern.TFrame', 
                       background=cls.COLORS['bg_primary'],
                       borderwidth=1,
                       relief='solid',
                       bordercolor=cls.COLORS['border'])
        
        style.configure('Card.TFrame',
                       background=cls.COLORS['bg_secondary'],
                       borderwidth=1,
                       relief='solid',
                       bordercolor=cls.COLORS['border'],
                       padding=15)
        
        # Label styles
        style.configure('Title.TLabel',
                       font=('Segoe UI', 14, 'bold'),
                       background=cls.COLORS['bg_secondary'],
                       foreground=cls.COLORS['text_primary'])
        
        style.configure('Subtitle.TLabel',
                       font=('Segoe UI', 11, 'bold'),
                       background=cls.COLORS['bg_secondary'],
                       foreground=cls.COLORS['text_secondary'])
        
        style.configure('Info.TLabel',
                       font=('Segoe UI', 9),
                       background=cls.COLORS['bg_secondary'],
                       foreground=cls.COLORS['text_secondary'])
        
        style.configure('Status.Success.TLabel',
                       font=('Segoe UI', 9, 'bold'),
                       background=cls.COLORS['bg_secondary'],
                       foreground=cls.COLORS['success'])
        
        style.configure('Status.Warning.TLabel',
                       font=('Segoe UI', 9, 'bold'),
                       background=cls.COLORS['bg_secondary'],
                       foreground=cls.COLORS['warning'])
        
        style.configure('Status.Error.TLabel',
                       font=('Segoe UI', 9, 'bold'),
                       background=cls.COLORS['bg_secondary'],
                       foreground=cls.COLORS['error'])
        
        # Button styles
        style.configure('Modern.TButton',
                       font=('Segoe UI', 9),
                       background=cls.COLORS['accent_primary'],
                       foreground=cls.COLORS['text_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       padding=(12, 8))
        
        style.map('Modern.TButton',
                 background=[('active', cls.COLORS['accent_secondary']),
                           ('pressed', cls.COLORS['accent_secondary'])])
        
        style.configure('Success.TButton',
                       font=('Segoe UI', 9),
                       background=cls.COLORS['success'],
                       foreground=cls.COLORS['text_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       padding=(12, 8))
        
        style.configure('Warning.TButton',
                       font=('Segoe UI', 9),
                       background=cls.COLORS['warning'],
                       foreground=cls.COLORS['text_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       padding=(12, 8))
        
        # Treeview styles
        style.configure('Modern.Treeview',
                       background=cls.COLORS['bg_tertiary'],
                       foreground=cls.COLORS['text_primary'],
                       fieldbackground=cls.COLORS['bg_tertiary'],
                       borderwidth=1,
                       bordercolor=cls.COLORS['border'])
        
        style.configure('Modern.Treeview.Heading',
                       font=('Segoe UI', 9, 'bold'),
                       background=cls.COLORS['bg_secondary'],
                       foreground=cls.COLORS['text_primary'],
                       borderwidth=1,
                       bordercolor=cls.COLORS['border'])
        
        # Notebook styles
        style.configure('Modern.TNotebook',
                       background=cls.COLORS['bg_primary'],
                       borderwidth=0)
        
        style.configure('Modern.TNotebook.Tab',
                       font=('Segoe UI', 9),
                       background=cls.COLORS['bg_secondary'],
                       foreground=cls.COLORS['text_secondary'],
                       padding=(12, 8),
                       borderwidth=1,
                       bordercolor=cls.COLORS['border'])
        
        style.map('Modern.TNotebook.Tab',
                 background=[('selected', cls.COLORS['accent_primary']),
                           ('active', cls.COLORS['hover'])],
                 foreground=[('selected', cls.COLORS['text_primary'])])

class MT5DashboardEnhanced:
    """Dashboard MT5 aprimorado com interface moderna e responsiva"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MT5 Trading Dashboard - Enhanced")
        self.root.geometry("1600x1000")
        self.root.minsize(1200, 800)
        self.root.configure(bg=ModernTheme.COLORS['bg_primary'])
        
        # Configurar tema moderno
        self.style = ttk.Style()
        ModernTheme.configure_styles(self.style)
        
        # Componentes MT5
        self.connector = None
        self.order_manager = None
        self.data_provider = None
        self.risk_manager = None
        self.signal_integrator = None
        self.logger = MT5Logger()
        
        # Estado da aplicação
        self.connected = False
        self.auto_trading_enabled = False
        self.monitoring_active = False
        
        # Variáveis de controle
        self.connection_status = tk.StringVar(value="Desconectado")
        self.account_balance = tk.StringVar(value="$0.00")
        self.account_equity = tk.StringVar(value="$0.00")
        self.open_positions = tk.StringVar(value="0")
        self.daily_profit = tk.StringVar(value="$0.00")
        self.auto_trading_status = tk.StringVar(value="Desabilitado")
        
        # Threads de atualização
        self.update_thread = None
        self.running = False
        
        # Configurar grid weights para responsividade
        self._configure_grid_weights()
        
        # Criar interface
        self._create_interface()
        
        # Configurar callbacks
        self._setup_callbacks()
        
        # Iniciar thread de atualização
        self._start_update_thread()
    
    def _configure_grid_weights(self):
        """Configura pesos do grid para layout responsivo"""
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def _create_interface(self):
        """Cria a interface moderna do dashboard"""
        # Container principal com padding
        main_container = ResponsiveFrame(self.root, style='Modern.TFrame')
        main_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        main_container.grid_rowconfigure(1, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        
        # Header com título e status
        self._create_header(main_container)
        
        # Notebook para organizar seções
        self.notebook = ttk.Notebook(main_container, style='Modern.TNotebook')
        self.notebook.grid(row=1, column=0, sticky="nsew", pady=(20, 0))
        
        # Criar abas
        self._create_overview_tab()
        self._create_trading_tab()
        self._create_positions_tab()
        self._create_signals_tab()
        self._create_risk_tab()
        self._create_logs_tab()
        self._create_settings_tab()
    
    def _create_header(self, parent):
        """Cria header com título e informações principais"""
        header_frame = ResponsiveFrame(parent, style='Card.TFrame')
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Título
        title_label = ttk.Label(header_frame, text="MT5 Trading Dashboard", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, sticky="w", padx=(0, 20))
        
        # Status de conexão
        status_frame = ttk.Frame(header_frame, style='Modern.TFrame')
        status_frame.grid(row=0, column=1, sticky="e")
        
        ttk.Label(status_frame, text="Status:", style='Info.TLabel').pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, textvariable=self.connection_status, 
                                     style='Status.Error.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=(5, 15))
        
        # Botões de conexão
        ttk.Button(status_frame, text="Conectar", command=self._connect_mt5,
                  style='Success.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(status_frame, text="Desconectar", command=self._disconnect_mt5,
                  style='Warning.TButton').pack(side=tk.LEFT, padx=2)
    
    def _create_overview_tab(self):
        """Cria aba de visão geral"""
        overview_frame = ResponsiveFrame(self.notebook, style='Modern.TFrame')
        self.notebook.add(overview_frame, text="Visão Geral")
        
        # Configurar grid
        overview_frame.grid_rowconfigure(0, weight=0)
        overview_frame.grid_rowconfigure(1, weight=1)
        overview_frame.grid_columnconfigure(0, weight=1)
        overview_frame.grid_columnconfigure(1, weight=1)
        
        # Cards de métricas principais
        metrics_frame = ttk.Frame(overview_frame, style='Modern.TFrame')
        metrics_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        metrics_frame.grid_columnconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(1, weight=1)
        metrics_frame.grid_columnconfigure(2, weight=1)
        metrics_frame.grid_columnconfigure(3, weight=1)
        
        # Card Saldo
        self._create_metric_card(metrics_frame, "Saldo da Conta", 
                               self.account_balance, 0, 0)
        
        # Card Equity
        self._create_metric_card(metrics_frame, "Equity", 
                               self.account_equity, 0, 1)
        
        # Card Posições
        self._create_metric_card(metrics_frame, "Posições Abertas", 
                               self.open_positions, 0, 2)
        
        # Card Lucro Diário
        self._create_metric_card(metrics_frame, "Lucro Diário", 
                               self.daily_profit, 0, 3)
        
        # Gráfico de performance (placeholder)
        chart_frame = ResponsiveFrame(overview_frame, style='Card.TFrame')
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        
        ttk.Label(chart_frame, text="Gráfico de Performance", 
                 style='Subtitle.TLabel').pack(pady=10)
        
        # Canvas para gráfico
        self.chart_canvas = tk.Canvas(chart_frame, 
                                     bg=ModernTheme.COLORS['bg_tertiary'],
                                     highlightthickness=0)
        self.chart_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Resumo de atividades
        activity_frame = ResponsiveFrame(overview_frame, style='Card.TFrame')
        activity_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
        
        ttk.Label(activity_frame, text="Atividades Recentes", 
                 style='Subtitle.TLabel').pack(pady=(0, 10))
        
        # Lista de atividades
        self.activity_listbox = tk.Listbox(activity_frame, 
                                          bg=ModernTheme.COLORS['bg_tertiary'],
                                          fg=ModernTheme.COLORS['text_primary'],
                                          selectbackground=ModernTheme.COLORS['accent_primary'],
                                          borderwidth=0,
                                          font=('Segoe UI', 9))
        self.activity_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    
    def _create_metric_card(self, parent, title, variable, row, col):
        """Cria card de métrica"""
        card = ResponsiveFrame(parent, style='Card.TFrame', min_width=150)
        card.grid(row=row, column=col, sticky="ew", padx=5)
        
        ttk.Label(card, text=title, style='Info.TLabel').pack(pady=(0, 5))
        ttk.Label(card, textvariable=variable, 
                 style='Status.Success.TLabel', 
                 font=('Segoe UI', 12, 'bold')).pack()
    
    def _create_trading_tab(self):
        """Cria aba de trading"""
        trading_frame = ResponsiveFrame(self.notebook, style='Modern.TFrame')
        self.notebook.add(trading_frame, text="Trading")
        
        # Configurar grid
        trading_frame.grid_rowconfigure(0, weight=0)
        trading_frame.grid_rowconfigure(1, weight=1)
        trading_frame.grid_columnconfigure(0, weight=1)
        trading_frame.grid_columnconfigure(1, weight=1)
        
        # Controles de auto trading
        control_frame = ResponsiveFrame(trading_frame, style='Card.TFrame')
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        
        ttk.Label(control_frame, text="Controle de Auto Trading", 
                 style='Subtitle.TLabel').pack(pady=(0, 10))
        
        # Status e controles
        status_control_frame = ttk.Frame(control_frame, style='Modern.TFrame')
        status_control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(status_control_frame, text="Status:", 
                 style='Info.TLabel').pack(side=tk.LEFT)
        ttk.Label(status_control_frame, textvariable=self.auto_trading_status, 
                 style='Status.Warning.TLabel').pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Button(status_control_frame, text="Habilitar", 
                  command=self._enable_auto_trading,
                  style='Success.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(status_control_frame, text="Desabilitar", 
                  command=self._disable_auto_trading,
                  style='Warning.TButton').pack(side=tk.LEFT, padx=5)
        
        # Ações rápidas
        actions_frame = ttk.Frame(control_frame, style='Modern.TFrame')
        actions_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(actions_frame, text="Fechar Todas Posições", 
                  command=self._close_all_positions,
                  style='Warning.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Análise Manual", 
                  command=self._manual_analysis,
                  style='Modern.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Gerar Relatório", 
                  command=self._generate_report,
                  style='Modern.TButton').pack(side=tk.LEFT, padx=5)
    
    def _create_positions_tab(self):
        """Cria aba de posições"""
        positions_frame = ResponsiveFrame(self.notebook, style='Modern.TFrame')
        self.notebook.add(positions_frame, text="Posições")
        
        # Configurar grid
        positions_frame.grid_rowconfigure(0, weight=1)
        positions_frame.grid_columnconfigure(0, weight=1)
        
        # Frame para treeview
        tree_frame = ResponsiveFrame(positions_frame, style='Card.TFrame')
        tree_frame.grid(row=0, column=0, sticky="nsew")
        tree_frame.grid_rowconfigure(1, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Label(tree_frame, text="Posições Abertas", 
                 style='Subtitle.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        # Treeview para posições
        columns = ('Symbol', 'Type', 'Volume', 'Open Price', 'Current Price', 'Profit', 'Time')
        self.positions_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', 
                                          style='Modern.Treeview')
        
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=120, anchor='center')
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, 
                                command=self.positions_tree.yview)
        h_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, 
                                command=self.positions_tree.xview)
        
        self.positions_tree.configure(yscrollcommand=v_scroll.set, 
                                     xscrollcommand=h_scroll.set)
        
        self.positions_tree.grid(row=1, column=0, sticky="nsew")
        v_scroll.grid(row=1, column=1, sticky="ns")
        h_scroll.grid(row=2, column=0, sticky="ew")
    
    def _create_signals_tab(self):
        """Cria aba de sinais"""
        signals_frame = ResponsiveFrame(self.notebook, style='Modern.TFrame')
        self.notebook.add(signals_frame, text="Sinais")
        
        # Configurar grid
        signals_frame.grid_rowconfigure(0, weight=1)
        signals_frame.grid_columnconfigure(0, weight=1)
        
        # Frame para treeview
        tree_frame = ResponsiveFrame(signals_frame, style='Card.TFrame')
        tree_frame.grid(row=0, column=0, sticky="nsew")
        tree_frame.grid_rowconfigure(1, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Label(tree_frame, text="Sinais de Trading", 
                 style='Subtitle.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        # Treeview para sinais
        columns = ('Time', 'Symbol', 'Action', 'Confidence', 'Status', 'Result')
        self.signals_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', 
                                        style='Modern.Treeview')
        
        for col in columns:
            self.signals_tree.heading(col, text=col)
            self.signals_tree.column(col, width=100, anchor='center')
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, 
                                command=self.signals_tree.yview)
        h_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, 
                                command=self.signals_tree.xview)
        
        self.signals_tree.configure(yscrollcommand=v_scroll.set, 
                                   xscrollcommand=h_scroll.set)
        
        self.signals_tree.grid(row=1, column=0, sticky="nsew")
        v_scroll.grid(row=1, column=1, sticky="ns")
        h_scroll.grid(row=2, column=0, sticky="ew")
    
    def _create_risk_tab(self):
        """Cria aba de gestão de risco"""
        risk_frame = ResponsiveFrame(self.notebook, style='Modern.TFrame')
        self.notebook.add(risk_frame, text="Risco")
        
        # Configurar grid
        risk_frame.grid_rowconfigure(0, weight=0)
        risk_frame.grid_rowconfigure(1, weight=1)
        risk_frame.grid_columnconfigure(0, weight=1)
        risk_frame.grid_columnconfigure(1, weight=1)
        
        # Métricas de risco
        metrics_frame = ResponsiveFrame(risk_frame, style='Card.TFrame')
        metrics_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        
        ttk.Label(metrics_frame, text="Métricas de Risco", 
                 style='Subtitle.TLabel').pack(pady=(0, 15))
        
        # Variáveis de risco
        self.risk_level = tk.StringVar(value="BAIXO")
        self.drawdown = tk.StringVar(value="0.00%")
        self.daily_loss = tk.StringVar(value="0.00%")
        self.win_rate = tk.StringVar(value="0.00%")
        
        # Grid de métricas
        metrics_grid = ttk.Frame(metrics_frame, style='Modern.TFrame')
        metrics_grid.pack(fill=tk.X, pady=10)
        
        # Configurar grid
        for i in range(4):
            metrics_grid.grid_columnconfigure(i, weight=1)
        
        self._create_risk_metric(metrics_grid, "Nível de Risco", self.risk_level, 0, 0)
        self._create_risk_metric(metrics_grid, "Drawdown", self.drawdown, 0, 1)
        self._create_risk_metric(metrics_grid, "Perda Diária", self.daily_loss, 0, 2)
        self._create_risk_metric(metrics_grid, "Taxa de Acerto", self.win_rate, 0, 3)
        
        # Alertas
        alerts_frame = ResponsiveFrame(risk_frame, style='Card.TFrame')
        alerts_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        
        ttk.Label(alerts_frame, text="Alertas de Risco", 
                 style='Subtitle.TLabel').pack(pady=(0, 10))
        
        self.alerts_listbox = tk.Listbox(alerts_frame, 
                                        bg=ModernTheme.COLORS['bg_tertiary'],
                                        fg=ModernTheme.COLORS['text_primary'],
                                        selectbackground=ModernTheme.COLORS['accent_primary'],
                                        borderwidth=0,
                                        font=('Segoe UI', 9))
        self.alerts_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Configurações de risco
        config_frame = ResponsiveFrame(risk_frame, style='Card.TFrame')
        config_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
        
        ttk.Label(config_frame, text="Configurações de Risco", 
                 style='Subtitle.TLabel').pack(pady=(0, 10))
        
        # Placeholder para configurações
        ttk.Label(config_frame, text="Configurações em desenvolvimento", 
                 style='Info.TLabel').pack(pady=20)
    
    def _create_risk_metric(self, parent, title, variable, row, col):
        """Cria métrica de risco"""
        frame = ttk.Frame(parent, style='Modern.TFrame')
        frame.grid(row=row, column=col, sticky="ew", padx=10)
        
        ttk.Label(frame, text=title, style='Info.TLabel').pack()
        ttk.Label(frame, textvariable=variable, 
                 style='Status.Warning.TLabel',
                 font=('Segoe UI', 11, 'bold')).pack(pady=(5, 0))
    
    def _create_logs_tab(self):
        """Cria aba de logs"""
        logs_frame = ResponsiveFrame(self.notebook, style='Modern.TFrame')
        self.notebook.add(logs_frame, text="Logs")
        
        # Configurar grid
        logs_frame.grid_rowconfigure(1, weight=1)
        logs_frame.grid_columnconfigure(0, weight=1)
        
        # Controles de filtro
        filter_frame = ResponsiveFrame(logs_frame, style='Card.TFrame')
        filter_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        ttk.Label(filter_frame, text="Filtros de Log", 
                 style='Subtitle.TLabel').pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Button(filter_frame, text="Exportar Logs", 
                  command=self._export_logs,
                  style='Modern.TButton').pack(side=tk.RIGHT, padx=5)
        ttk.Button(filter_frame, text="Limpar Logs", 
                  command=self._clear_logs,
                  style='Warning.TButton').pack(side=tk.RIGHT, padx=5)
        
        # Área de logs
        logs_container = ResponsiveFrame(logs_frame, style='Card.TFrame')
        logs_container.grid(row=1, column=0, sticky="nsew")
        logs_container.grid_rowconfigure(0, weight=1)
        logs_container.grid_columnconfigure(0, weight=1)
        
        # Text widget para logs
        self.logs_text = tk.Text(logs_container, 
                                bg=ModernTheme.COLORS['bg_tertiary'],
                                fg=ModernTheme.COLORS['text_primary'],
                                insertbackground=ModernTheme.COLORS['text_primary'],
                                selectbackground=ModernTheme.COLORS['accent_primary'],
                                borderwidth=0,
                                font=('Consolas', 9),
                                wrap=tk.WORD)
        
        # Scrollbar para logs
        logs_scroll = ttk.Scrollbar(logs_container, orient=tk.VERTICAL, 
                                   command=self.logs_text.yview)
        self.logs_text.configure(yscrollcommand=logs_scroll.set)
        
        self.logs_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        logs_scroll.grid(row=0, column=1, sticky="ns", pady=10)
    
    def _create_settings_tab(self):
        """Cria aba de configurações"""
        settings_frame = ResponsiveFrame(self.notebook, style='Modern.TFrame')
        self.notebook.add(settings_frame, text="Configurações")
        
        # Placeholder para configurações
        ttk.Label(settings_frame, text="Configurações do Sistema", 
                 style='Title.TLabel').pack(pady=50)
        ttk.Label(settings_frame, text="Funcionalidade em desenvolvimento", 
                 style='Info.TLabel').pack()
    
    def _setup_callbacks(self):
        """Configura callbacks do sistema"""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Bind para redimensionamento
        self.root.bind('<Configure>', self._on_window_configure)
    
    def _on_window_configure(self, event):
        """Manipula redimensionamento da janela"""
        if event.widget == self.root:
            # Atualizar layout responsivo se necessário
            pass
    
    def _start_update_thread(self):
        """Inicia thread de atualização da interface"""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def _update_loop(self):
        """Loop de atualização da interface"""
        while self.running:
            try:
                self._update_interface()
                time.sleep(2)
            except Exception as e:
                if self.logger:
                    self.logger.log(LogLevel.ERROR, EventType.SYSTEM, "Dashboard", 
                                  f"Erro no loop de atualização: {e}", exception=e)
                time.sleep(5)
    
    def _update_interface(self):
        """Atualiza dados da interface"""
        if not self.connected:
            return
        
        try:
            # Atualizar informações da conta
            account_info = mt5.account_info()
            if account_info:
                self.account_balance.set(f"${account_info.balance:,.2f}")
                self.account_equity.set(f"${account_info.equity:,.2f}")
                
                # Calcular lucro diário
                daily_profit = account_info.equity - account_info.balance
                self.daily_profit.set(f"${daily_profit:,.2f}")
            
            # Atualizar posições
            self._update_positions()
            
            # Atualizar sinais
            self._update_signals()
            
            # Atualizar informações de risco
            self._update_risk_info()
            
            # Atualizar logs
            self._update_logs()
            
            # Atualizar atividades
            self._update_activities()
            
        except Exception as e:
            if self.logger:
                self.logger.log(LogLevel.ERROR, EventType.SYSTEM, "Dashboard", 
                              f"Erro na atualização da interface: {e}", exception=e)
    
    def _update_positions(self):
        """Atualiza lista de posições"""
        if not self.order_manager:
            return
        
        try:
            positions = self.order_manager.get_positions()
            self.open_positions.set(str(len(positions)))
            
            # Limpar treeview
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            
            # Adicionar posições
            for pos in positions:
                current_price = (self.data_provider.get_current_price(pos['symbol']) 
                               if self.data_provider else None)
                current = current_price['bid'] if current_price else pos['price_open']
                
                # Determinar cor baseada no lucro
                profit_color = 'success' if pos['profit'] >= 0 else 'error'
                
                item = self.positions_tree.insert('', 'end', values=(
                    pos['symbol'],
                    pos['type'],
                    f"{pos['volume']:.2f}",
                    f"{pos['price_open']:.5f}",
                    f"{current:.5f}",
                    f"${pos['profit']:.2f}",
                    pos['time'].strftime('%H:%M:%S')
                ))
                
                # Aplicar cor baseada no lucro
                if pos['profit'] >= 0:
                    self.positions_tree.set(item, 'Profit', f"+${pos['profit']:.2f}")
                
        except Exception as e:
            if self.logger:
                self.logger.log(LogLevel.ERROR, EventType.SYSTEM, "Dashboard", 
                              f"Erro ao atualizar posições: {e}", exception=e)
    
    def _update_signals(self):
        """Atualiza lista de sinais"""
        if not self.signal_integrator:
            return
        
        try:
            # Obter sinais recentes
            recent_signals = [s for s in self.signal_integrator.signal_history 
                            if (datetime.now() - s.timestamp).seconds < 3600]
            
            # Limpar treeview
            for item in self.signals_tree.get_children():
                self.signals_tree.delete(item)
            
            # Adicionar sinais
            for signal in recent_signals[-20:]:  # Últimos 20 sinais
                status = ("Executado" if signal.symbol in self.signal_integrator.active_signals 
                         else "Pendente")
                
                self.signals_tree.insert('', 'end', values=(
                    signal.timestamp.strftime('%H:%M:%S'),
                    signal.symbol,
                    signal.action.value,
                    f"{signal.confidence:.1%}",
                    status,
                    "N/A"  # Resultado será implementado
                ))
        
        except Exception as e:
            if self.logger:
                self.logger.log(LogLevel.ERROR, EventType.SYSTEM, "Dashboard", 
                              f"Erro ao atualizar sinais: {e}", exception=e)
    
    def _update_risk_info(self):
        """Atualiza informações de risco"""
        if not self.risk_manager:
            return
        
        try:
            risk_summary = self.risk_manager.get_risk_summary()
            
            # Atualizar nível de risco
            risk_level = risk_summary.get('risk_level', RiskLevel.LOW)
            self.risk_level.set(risk_level.value if hasattr(risk_level, 'value') else str(risk_level))
            
            # Atualizar métricas
            self.drawdown.set(f"{risk_summary.get('drawdown', 0):.2%}")
            self.daily_loss.set(f"{risk_summary.get('daily_loss', 0):.2%}")
            self.win_rate.set(f"{risk_summary.get('win_rate', 0):.2%}")
            
            # Atualizar alertas
            self.alerts_listbox.delete(0, tk.END)
            for alert in self.risk_manager.alerts[-10:]:  # Últimos 10 alertas
                alert_text = f"[{alert.timestamp.strftime('%H:%M')}] {alert.level.value}: {alert.message}"
                self.alerts_listbox.insert(tk.END, alert_text)
                
        except Exception as e:
            if self.logger:
                self.logger.log(LogLevel.ERROR, EventType.SYSTEM, "Dashboard", 
                              f"Erro ao atualizar informações de risco: {e}", exception=e)
    
    def _update_logs(self):
        """Atualiza logs na interface"""
        try:
            if not self.logger:
                return
                
            recent_logs = self.logger.get_recent_logs(minutes=30)
            
            # Manter posição do scroll
            scroll_position = self.logs_text.yview()[1]
            at_bottom = scroll_position >= 0.95
            
            # Limpar e adicionar logs
            self.logs_text.delete(1.0, tk.END)
            
            for log_entry in recent_logs[-50:]:  # Últimos 50 logs
                timestamp = log_entry.timestamp.strftime('%H:%M:%S')
                log_line = f"[{timestamp}] {log_entry.level.value} - {log_entry.component}: {log_entry.message}\n"
                
                # Inserir com cor baseada no nível
                if log_entry.level == LogLevel.ERROR:
                    self.logs_text.insert(tk.END, log_line, 'error')
                elif log_entry.level == LogLevel.WARNING:
                    self.logs_text.insert(tk.END, log_line, 'warning')
                elif log_entry.level == LogLevel.INFO:
                    self.logs_text.insert(tk.END, log_line, 'info')
                else:
                    self.logs_text.insert(tk.END, log_line)
            
            # Configurar tags de cor
            self.logs_text.tag_config('error', foreground=ModernTheme.COLORS['error'])
            self.logs_text.tag_config('warning', foreground=ModernTheme.COLORS['warning'])
            self.logs_text.tag_config('info', foreground=ModernTheme.COLORS['success'])
            
            # Manter scroll na posição ou ir para o final
            if at_bottom:
                self.logs_text.see(tk.END)
            
        except Exception as e:
            print(f"Erro ao atualizar logs: {e}")
    
    def _update_activities(self):
        """Atualiza lista de atividades recentes"""
        try:
            # Placeholder para atividades
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Manter apenas últimas 10 atividades
            if self.activity_listbox.size() > 10:
                self.activity_listbox.delete(0)
            
            # Adicionar atividade de exemplo (será substituído por dados reais)
            if self.connected:
                activity = f"[{current_time}] Sistema monitorando mercado..."
                self.activity_listbox.insert(tk.END, activity)
                self.activity_listbox.see(tk.END)
        
        except Exception as e:
            print(f"Erro ao atualizar atividades: {e}")
    
    # Métodos de ação (mantidos do dashboard original)
    def _connect_mt5(self):
        """Conecta ao MT5"""
        try:
            self.connector = MT5Connector()
            if self.connector.connect():
                self.connected = True
                self.connection_status.set("Conectado")
                self.status_label.configure(style='Status.Success.TLabel')
                
                # Inicializar componentes
                self.order_manager = MT5OrderManager(self.connector)
                self.data_provider = MT5DataProvider(self.connector)
                self.risk_manager = MT5RiskManager(self.connector, self.order_manager, self.data_provider)
                
                config = SignalConfig(enable_auto_trading=self.auto_trading_enabled)
                self.signal_integrator = MT5SignalIntegrator(
                    self.connector, self.order_manager, self.data_provider, self.risk_manager, config
                )
                
                # Iniciar monitoramento
                self.risk_manager.start_monitoring()
                
                self.logger.log(LogLevel.INFO, EventType.CONNECTION, "Dashboard", "Conectado ao MT5")
                
            else:
                messagebox.showerror("Erro", "Falha ao conectar com MT5")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na conexão: {e}")
            if self.logger:
                self.logger.log(LogLevel.ERROR, EventType.CONNECTION, "Dashboard", 
                              f"Erro na conexão: {e}", exception=e)
    
    def _disconnect_mt5(self):
        """Desconecta do MT5"""
        try:
            if self.risk_manager:
                self.risk_manager.stop_monitoring()
            
            if self.signal_integrator:
                self.signal_integrator.stop_analysis_loop()
            
            if self.connector:
                self.connector.disconnect()
            
            self.connected = False
            self.connection_status.set("Desconectado")
            self.status_label.configure(style='Status.Error.TLabel')
            
            if self.logger:
                self.logger.log(LogLevel.INFO, EventType.CONNECTION, "Dashboard", "Desconectado do MT5")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na desconexão: {e}")
    
    def _enable_auto_trading(self):
        """Habilita auto trading"""
        if not self.connected:
            messagebox.showwarning("Aviso", "Conecte ao MT5 primeiro")
            return
        
        self.auto_trading_enabled = True
        self.auto_trading_status.set("Habilitado")
        
        if self.signal_integrator:
            self.signal_integrator.config.enable_auto_trading = True
            self.signal_integrator.start_analysis_loop()
        
        if self.logger:
            self.logger.log(LogLevel.INFO, EventType.SYSTEM, "Dashboard", "Auto trading habilitado")
    
    def _disable_auto_trading(self):
        """Desabilita auto trading"""
        self.auto_trading_enabled = False
        self.auto_trading_status.set("Desabilitado")
        
        if self.signal_integrator:
            self.signal_integrator.config.enable_auto_trading = False
            self.signal_integrator.stop_analysis_loop()
        
        if self.logger:
            self.logger.log(LogLevel.INFO, EventType.SYSTEM, "Dashboard", "Auto trading desabilitado")
    
    def _close_all_positions(self):
        """Fecha todas as posições"""
        if not self.order_manager:
            messagebox.showwarning("Aviso", "Conecte ao MT5 primeiro")
            return
        
        result = messagebox.askyesno("Confirmação", "Deseja fechar todas as posições?")
        if result:
            try:
                positions = self.order_manager.get_positions()
                closed_count = 0
                
                for pos in positions:
                    if self.order_manager.close_position(pos['symbol']):
                        closed_count += 1
                
                messagebox.showinfo("Sucesso", f"{closed_count} posições fechadas")
                if self.logger:
                    self.logger.log(LogLevel.INFO, EventType.TRADE, "Dashboard", 
                                  f"{closed_count} posições fechadas")
                
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao fechar posições: {e}")
    
    def _manual_analysis(self):
        """Executa análise manual"""
        if not self.signal_integrator:
            messagebox.showwarning("Aviso", "Conecte ao MT5 primeiro")
            return
        
        try:
            signals = self.signal_integrator.analyze_market()
            
            if signals:
                message = f"{len(signals)} sinais gerados:\n\n"
                for signal in signals:
                    message += f"{signal.symbol}: {signal.action.value} (Confiança: {signal.confidence:.1%})\n"
                
                messagebox.showinfo("Análise Manual", message)
            else:
                messagebox.showinfo("Análise Manual", "Nenhum sinal gerado")
        
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na análise: {e}")
    
    def _generate_report(self):
        """Gera relatório de performance"""
        if not self.signal_integrator:
            messagebox.showwarning("Aviso", "Conecte ao MT5 primeiro")
            return
        
        try:
            performance = self.signal_integrator.get_performance_summary()
            risk_summary = self.risk_manager.get_risk_summary() if self.risk_manager else {}
            
            report = f"""RELATÓRIO DE PERFORMANCE
{'='*50}

Sinais:
- Total de sinais: {performance.get('total_signals', 0)}
- Trades executados: {performance.get('executed_trades', 0)}
- Taxa de acerto: {performance.get('win_rate', 0):.1f}%
- Lucro total: ${performance.get('total_profit', 0):.2f}

Risco:
- Nível atual: {risk_summary.get('risk_level', 'N/A')}
- Drawdown: {risk_summary.get('drawdown', 0):.2%}
- Perda diária: {risk_summary.get('daily_loss', 0):.2%}

Conta:
- Saldo: {self.account_balance.get()}
- Equity: {self.account_equity.get()}
- Posições abertas: {self.open_positions.get()}

Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            # Mostrar relatório em nova janela
            report_window = tk.Toplevel(self.root)
            report_window.title("Relatório de Performance")
            report_window.geometry("700x500")
            report_window.configure(bg=ModernTheme.COLORS['bg_primary'])
            
            # Frame para o relatório
            report_frame = ResponsiveFrame(report_window, style='Card.TFrame')
            report_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            report_frame.grid_rowconfigure(0, weight=1)
            report_frame.grid_columnconfigure(0, weight=1)
            
            # Text widget
            text_widget = tk.Text(report_frame, 
                                 wrap=tk.WORD, 
                                 font=('Consolas', 10),
                                 bg=ModernTheme.COLORS['bg_tertiary'],
                                 fg=ModernTheme.COLORS['text_primary'],
                                 borderwidth=0)
            text_widget.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
            text_widget.insert(1.0, report)
            text_widget.configure(state=tk.DISABLED)
            
            # Scrollbar
            scrollbar = ttk.Scrollbar(report_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            scrollbar.grid(row=0, column=1, sticky="ns", pady=10)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar relatório: {e}")
    
    def _export_logs(self):
        """Exporta logs para arquivo"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv")]
            )
            
            if filename:
                start_date = datetime.now() - timedelta(days=1)
                end_date = datetime.now()
                
                format_type = 'json' if filename.endswith('.json') else 'csv'
                
                if self.logger and self.logger.export_logs(start_date, end_date, filename, format_type):
                    messagebox.showinfo("Sucesso", f"Logs exportados para {filename}")
                else:
                    messagebox.showerror("Erro", "Falha ao exportar logs")
        
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na exportação: {e}")
    
    def _clear_logs(self):
        """Limpa logs da interface"""
        result = messagebox.askyesno("Confirmação", "Deseja limpar todos os logs da interface?")
        if result:
            self.logs_text.delete(1.0, tk.END)
    
    def _on_closing(self):
        """Manipula fechamento da aplicação"""
        try:
            self.running = False
            
            if self.connected:
                self._disconnect_mt5()
            
            if self.logger:
                self.logger.cleanup()
            
            self.root.destroy()
        
        except Exception as e:
            print(f"Erro ao fechar aplicação: {e}")
            self.root.destroy()
    
    def run(self):
        """Executa o dashboard"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self._on_closing()

if __name__ == "__main__":
    # Executar dashboard aprimorado
    dashboard = MT5DashboardEnhanced()
    dashboard.run()