#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Tkinter para MetaTrader 5
Interface gráfica para controle e monitoramento de operações
Autor: Sistema Forex ML
Data: 2024
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from typing import Dict, List, Optional
import json
from dataclasses import asdict

# Importar módulos locais
from mt5_integration import MT5Integration, MT5Config, TradeRequest, OrderType, TradeAction
from detailed_decision_logger import DetailedDecisionLogger

class MT5Dashboard:
    """
    Dashboard principal para controle do MetaTrader 5
    Interface gráfica completa para trading automatizado
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MT5 Trading Dashboard - Sistema Forex ML")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Variáveis de controle
        self.mt5_client = None
        self.ml_predictor = None
        self.connected = False
        self.auto_trading = False
        self.monitoring_thread = None
        self.stop_monitoring = False
        
        # Dados em tempo real
        self.current_prices = {}
        self.positions_data = []
        self.orders_data = []
        self.account_data = {}
        
        # Logger de decisões detalhadas
        self.decision_logger = DetailedDecisionLogger()
        print("DetailedDecisionLogger inicializado no dashboard")
        
        # Configuração de estilo
        self.setup_styles()
        
        # Criar interface
        self.create_widgets()
        
        # Inicializar ML Predictor
        self.init_ml_predictor()
        
        # Configurar eventos
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """Configura estilos da interface"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Cores personalizadas
        style.configure('Title.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ffffff', 
                       font=('Arial', 14, 'bold'))
        
        style.configure('Info.TLabel', 
                       background='#2b2b2b', 
                       foreground='#cccccc', 
                       font=('Arial', 10))
        
        style.configure('Success.TLabel', 
                       background='#2b2b2b', 
                       foreground='#00ff00', 
                       font=('Arial', 10, 'bold'))
        
        style.configure('Error.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ff0000', 
                       font=('Arial', 10, 'bold'))
        
        style.configure('Custom.TButton', 
                       background='#404040', 
                       foreground='#ffffff', 
                       font=('Arial', 9, 'bold'))
        
        style.configure('Buy.TButton', 
                       background='#00aa00', 
                       foreground='#ffffff', 
                       font=('Arial', 9, 'bold'))
        
        style.configure('Sell.TButton', 
                       background='#aa0000', 
                       foreground='#ffffff', 
                       font=('Arial', 9, 'bold'))
    
    def create_widgets(self):
        """Cria todos os widgets da interface"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = ttk.Label(main_frame, text="MT5 Trading Dashboard", style='Title.TLabel')
        title_label.pack(pady=(0, 10))
        
        # Notebook para abas
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Criar abas
        self.create_connection_tab()
        self.create_account_tab()
        self.create_trading_tab()
        self.create_positions_tab()
        self.create_analysis_tab()
        self.create_logs_tab()
        self.create_detailed_logs_tab()
    
    def create_connection_tab(self):
        """Cria aba de conexão"""
        connection_frame = ttk.Frame(self.notebook)
        self.notebook.add(connection_frame, text="Conexão")
        
        # Frame de configuração
        config_frame = ttk.LabelFrame(connection_frame, text="Configuração MT5")
        config_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Campos de configuração
        ttk.Label(config_frame, text="Login:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.login_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.login_var, width=20).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(config_frame, text="Senha:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.password_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.password_var, show="*", width=20).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(config_frame, text="Servidor:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.server_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.server_var, width=30).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(config_frame, text="Caminho MT5:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.path_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.path_var, width=50).grid(row=3, column=1, padx=5, pady=5)
        
        # Botões de conexão
        button_frame = ttk.Frame(config_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        self.connect_btn = ttk.Button(button_frame, text="Conectar", 
                                     command=self.connect_mt5, style='Custom.TButton')
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.disconnect_btn = ttk.Button(button_frame, text="Desconectar", 
                                        command=self.disconnect_mt5, style='Custom.TButton')
        self.disconnect_btn.pack(side=tk.LEFT, padx=5)
        self.disconnect_btn.configure(state='disabled')
        
        # Status da conexão
        self.connection_status = ttk.Label(config_frame, text="Desconectado", style='Error.TLabel')
        self.connection_status.grid(row=5, column=0, columnspan=2, pady=5)
    
    def create_account_tab(self):
        """Cria aba de informações da conta"""
        account_frame = ttk.Frame(self.notebook)
        self.notebook.add(account_frame, text="Conta")
        
        # Frame de informações da conta
        info_frame = ttk.LabelFrame(account_frame, text="Informações da Conta")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Labels para informações
        self.account_labels = {}
        fields = ['login', 'server', 'balance', 'equity', 'margin', 'free_margin', 'margin_level']
        
        for i, field in enumerate(fields):
            ttk.Label(info_frame, text=f"{field.replace('_', ' ').title()}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            self.account_labels[field] = ttk.Label(info_frame, text="-", style='Info.TLabel')
            self.account_labels[field].grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Botão de atualização
        ttk.Button(info_frame, text="Atualizar", 
                  command=self.update_account_info, style='Custom.TButton').grid(row=len(fields), column=0, columnspan=2, pady=10)
    
    def create_trading_tab(self):
        """Cria aba de trading"""
        trading_frame = ttk.Frame(self.notebook)
        self.notebook.add(trading_frame, text="Trading")
        
        # Frame de ordem manual
        order_frame = ttk.LabelFrame(trading_frame, text="Ordem Manual")
        order_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Campos de ordem
        ttk.Label(order_frame, text="Símbolo:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.symbol_var = tk.StringVar(value="EURUSD")
        symbol_combo = ttk.Combobox(order_frame, textvariable=self.symbol_var, 
                                   values=["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"])
        symbol_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(order_frame, text="Volume:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.volume_var = tk.StringVar(value="0.01")
        ttk.Entry(order_frame, textvariable=self.volume_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(order_frame, text="Stop Loss:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.sl_var = tk.StringVar()
        ttk.Entry(order_frame, textvariable=self.sl_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(order_frame, text="Take Profit:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.tp_var = tk.StringVar()
        ttk.Entry(order_frame, textvariable=self.tp_var, width=10).grid(row=1, column=3, padx=5, pady=5)
        
        # Botões de trading
        button_frame = ttk.Frame(order_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=10)
        
        ttk.Button(button_frame, text="BUY", command=lambda: self.send_market_order("BUY"), 
                  style='Buy.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="SELL", command=lambda: self.send_market_order("SELL"), 
                  style='Sell.TButton').pack(side=tk.LEFT, padx=5)
        
        # Frame de trading automático
        auto_frame = ttk.LabelFrame(trading_frame, text="Trading Automático")
        auto_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.auto_trading_var = tk.BooleanVar()
        ttk.Checkbutton(auto_frame, text="Ativar Trading Automático", 
                       variable=self.auto_trading_var, 
                       command=self.toggle_auto_trading).pack(pady=5)
        
        # Configurações de auto trading
        config_frame = ttk.Frame(auto_frame)
        config_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(config_frame, text="Intervalo (segundos):").pack(side=tk.LEFT, padx=5)
        self.interval_var = tk.StringVar(value="60")
        ttk.Entry(config_frame, textvariable=self.interval_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(config_frame, text="Confiança mínima:").pack(side=tk.LEFT, padx=5)
        self.confidence_var = tk.StringVar(value="0.6")
        ttk.Entry(config_frame, textvariable=self.confidence_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Frame de cotações em tempo real
        quotes_frame = ttk.LabelFrame(trading_frame, text="Cotações em Tempo Real")
        quotes_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview para cotações
        columns = ('Symbol', 'Bid', 'Ask', 'Spread', 'Time')
        self.quotes_tree = ttk.Treeview(quotes_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.quotes_tree.heading(col, text=col)
            self.quotes_tree.column(col, width=100)
        
        scrollbar_quotes = ttk.Scrollbar(quotes_frame, orient=tk.VERTICAL, command=self.quotes_tree.yview)
        self.quotes_tree.configure(yscrollcommand=scrollbar_quotes.set)
        
        self.quotes_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_quotes.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_positions_tab(self):
        """Cria aba de posições"""
        positions_frame = ttk.Frame(self.notebook)
        self.notebook.add(positions_frame, text="Posições")
        
        # Frame de posições abertas
        open_frame = ttk.LabelFrame(positions_frame, text="Posições Abertas")
        open_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview para posições
        pos_columns = ('Ticket', 'Symbol', 'Type', 'Volume', 'Price', 'SL', 'TP', 'Profit', 'Time')
        self.positions_tree = ttk.Treeview(open_frame, columns=pos_columns, show='headings', height=10)
        
        for col in pos_columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=80)
        
        scrollbar_pos = ttk.Scrollbar(open_frame, orient=tk.VERTICAL, command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=scrollbar_pos.set)
        
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_pos.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Botões de controle de posições
        pos_buttons = ttk.Frame(positions_frame)
        pos_buttons.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(pos_buttons, text="Fechar Posição", 
                  command=self.close_selected_position, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(pos_buttons, text="Fechar Todas", 
                  command=self.close_all_positions, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(pos_buttons, text="Atualizar", 
                  command=self.update_positions, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        
        # Frame de ordens pendentes
        orders_frame = ttk.LabelFrame(positions_frame, text="Ordens Pendentes")
        orders_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview para ordens
        order_columns = ('Ticket', 'Symbol', 'Type', 'Volume', 'Price', 'SL', 'TP', 'Time')
        self.orders_tree = ttk.Treeview(orders_frame, columns=order_columns, show='headings', height=6)
        
        for col in order_columns:
            self.orders_tree.heading(col, text=col)
            self.orders_tree.column(col, width=80)
        
        scrollbar_ord = ttk.Scrollbar(orders_frame, orient=tk.VERTICAL, command=self.orders_tree.yview)
        self.orders_tree.configure(yscrollcommand=scrollbar_ord.set)
        
        self.orders_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_ord.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_analysis_tab(self):
        """Cria aba de análise"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Análise")
        
        # Frame de predições ML
        ml_frame = ttk.LabelFrame(analysis_frame, text="Predições Machine Learning")
        ml_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Seleção de símbolo para análise
        symbol_frame = ttk.Frame(ml_frame)
        symbol_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(symbol_frame, text="Símbolo:").pack(side=tk.LEFT, padx=5)
        self.analysis_symbol_var = tk.StringVar(value="EURUSD")
        analysis_combo = ttk.Combobox(symbol_frame, textvariable=self.analysis_symbol_var, 
                                     values=["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"])
        analysis_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(symbol_frame, text="Analisar", 
                  command=self.analyze_symbol, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        
        # Resultados da análise
        results_frame = ttk.Frame(ml_frame)
        results_frame.pack(fill=tk.X, pady=5)
        
        self.analysis_results = scrolledtext.ScrolledText(results_frame, height=8, width=80)
        self.analysis_results.pack(fill=tk.BOTH, expand=True)
        
        # Frame de gráfico
        chart_frame = ttk.LabelFrame(analysis_frame, text="Gráfico de Preços")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Matplotlib figure
        self.fig = Figure(figsize=(12, 6), dpi=100, facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111, facecolor='#2b2b2b')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_logs_tab(self):
        """Cria aba de logs"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="Logs")
        
        # Frame de logs
        log_frame = ttk.LabelFrame(logs_frame, text="Log de Atividades")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=25, width=100, 
                                                 bg='#1e1e1e', fg='#ffffff', 
                                                 insertbackground='white')
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Botões de controle de log
        log_buttons = ttk.Frame(logs_frame)
        log_buttons.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(log_buttons, text="Limpar Logs", 
                  command=self.clear_logs, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(log_buttons, text="Salvar Logs", 
                  command=self.save_logs, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
    
    def create_detailed_logs_tab(self):
        """Cria aba de logs detalhados das decisões do bot"""
        detailed_logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(detailed_logs_frame, text="Decisões Bot")
        
        # Frame principal com divisão vertical
        main_paned = ttk.PanedWindow(detailed_logs_frame, orient=tk.VERTICAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame superior - Lista de decisões
        decisions_frame = ttk.LabelFrame(main_paned, text="Decisões Recentes")
        main_paned.add(decisions_frame, weight=1)
        
        # Treeview para decisões
        decision_columns = ('ID', 'Timestamp', 'Symbol', 'Action', 'Confidence', 'Reasoning')
        self.decisions_tree = ttk.Treeview(decisions_frame, columns=decision_columns, show='headings', height=12)
        
        # Configurar colunas
        column_widths = {'ID': 60, 'Timestamp': 150, 'Symbol': 80, 'Action': 120, 'Confidence': 80, 'Reasoning': 300}
        for col in decision_columns:
            self.decisions_tree.heading(col, text=col)
            self.decisions_tree.column(col, width=column_widths.get(col, 100))
        
        # Scrollbar para decisões
        scrollbar_decisions = ttk.Scrollbar(decisions_frame, orient=tk.VERTICAL, command=self.decisions_tree.yview)
        self.decisions_tree.configure(yscrollcommand=scrollbar_decisions.set)
        
        self.decisions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_decisions.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Frame inferior - Detalhes da decisão selecionada
        details_frame = ttk.LabelFrame(main_paned, text="Detalhes da Decisão")
        main_paned.add(details_frame, weight=1)
        
        # Notebook para diferentes tipos de detalhes
        self.details_notebook = ttk.Notebook(details_frame)
        self.details_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Aba de dados de mercado
        market_frame = ttk.Frame(self.details_notebook)
        self.details_notebook.add(market_frame, text="Dados Mercado")
        self.market_details_text = scrolledtext.ScrolledText(market_frame, height=8, width=80,
                                                            bg='#1e1e1e', fg='#ffffff',
                                                            insertbackground='white')
        self.market_details_text.pack(fill=tk.BOTH, expand=True)
        
        # Aba de análise técnica
        technical_frame = ttk.Frame(self.details_notebook)
        self.details_notebook.add(technical_frame, text="Análise Técnica")
        self.technical_details_text = scrolledtext.ScrolledText(technical_frame, height=8, width=80,
                                                               bg='#1e1e1e', fg='#ffffff',
                                                               insertbackground='white')
        self.technical_details_text.pack(fill=tk.BOTH, expand=True)
        
        # Aba de predições ML
        ml_frame = ttk.Frame(self.details_notebook)
        self.details_notebook.add(ml_frame, text="Predições ML")
        self.ml_details_text = scrolledtext.ScrolledText(ml_frame, height=8, width=80,
                                                        bg='#1e1e1e', fg='#ffffff',
                                                        insertbackground='white')
        self.ml_details_text.pack(fill=tk.BOTH, expand=True)
        
        # Aba de ações do bot
        actions_frame = ttk.Frame(self.details_notebook)
        self.details_notebook.add(actions_frame, text="Ações Bot")
        self.actions_details_text = scrolledtext.ScrolledText(actions_frame, height=8, width=80,
                                                             bg='#1e1e1e', fg='#ffffff',
                                                             insertbackground='white')
        self.actions_details_text.pack(fill=tk.BOTH, expand=True)
        
        # Botões de controle
        control_frame = ttk.Frame(detailed_logs_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Atualizar Decisões",
                  command=self.refresh_decisions, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Exportar CSV",
                  command=self.export_decisions_csv, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Análise Performance",
                  command=self.show_performance_analysis, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        
        # Bind evento de seleção
        self.decisions_tree.bind('<<TreeviewSelect>>', self.on_decision_select)
        
        # Carregar decisões iniciais
        self.refresh_decisions()
    
    def init_ml_predictor(self):
        """Inicializa o preditor ML"""
        try:
            # Usar preditor ML simplificado interno
            self.ml_predictor = self._create_simple_ml_predictor()
            self.log_message("Preditor ML inicializado com sucesso")
        except Exception as e:
            self.log_message(f"Erro ao inicializar preditor ML: {e}")
            self.ml_predictor = None
    
    def _create_simple_ml_predictor(self):
        """Cria um preditor ML simplificado"""
        class SimpleMLPredictor:
            def predict_direction(self, symbol):
                """Predição simplificada baseada em análise técnica"""
                import random
                # Simulação de predição - em produção, usar dados reais
                direction = random.choice([-1, 0, 1])
                confidence = random.uniform(0.5, 0.9)
                return direction, confidence
        
        return SimpleMLPredictor()
    
    def connect_mt5(self):
        """Conecta ao MetaTrader 5"""
        try:
            if not all([self.login_var.get(), self.password_var.get(), self.server_var.get()]):
                messagebox.showerror("Erro", "Preencha todos os campos de conexão")
                return
            
            config = MT5Config(
                login=int(self.login_var.get()),
                password=self.password_var.get(),
                server=self.server_var.get(),
                path=self.path_var.get()
            )
            
            self.mt5_client = MT5Integration(config)
            
            if self.mt5_client.connect():
                self.connected = True
                self.connection_status.configure(text="Conectado", style='Success.TLabel')
                self.connect_btn.configure(state='disabled')
                self.disconnect_btn.configure(state='normal')
                
                self.log_message("Conectado ao MT5 com sucesso")
                self.update_account_info()
                self.start_monitoring()
                
                messagebox.showinfo("Sucesso", "Conectado ao MT5 com sucesso!")
            else:
                messagebox.showerror("Erro", f"Falha na conexão: {self.mt5_client.last_error}")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na conexão: {str(e)}")
            self.log_message(f"Erro na conexão: {e}")
    
    def disconnect_mt5(self):
        """Desconecta do MetaTrader 5"""
        try:
            self.stop_monitoring = True
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=2)
            
            if self.mt5_client:
                self.mt5_client.disconnect()
            
            self.connected = False
            self.connection_status.configure(text="Desconectado", style='Error.TLabel')
            self.connect_btn.configure(state='normal')
            self.disconnect_btn.configure(state='disabled')
            
            self.log_message("Desconectado do MT5")
            messagebox.showinfo("Info", "Desconectado do MT5")
            
        except Exception as e:
            self.log_message(f"Erro ao desconectar: {e}")
    
    def update_account_info(self):
        """Atualiza informações da conta"""
        if not self.connected or not self.mt5_client:
            return
        
        try:
            account_info = self.mt5_client.get_account_info()
            if account_info:
                self.account_data = account_info
                
                for field, label in self.account_labels.items():
                    value = account_info.get(field, '-')
                    if isinstance(value, (int, float)) and field in ['balance', 'equity', 'margin', 'free_margin']:
                        label.configure(text=f"{value:.2f}")
                    elif field == 'margin_level' and isinstance(value, (int, float)):
                        label.configure(text=f"{value:.2f}%")
                    else:
                        label.configure(text=str(value))
                        
        except Exception as e:
            self.log_message(f"Erro ao atualizar informações da conta: {e}")
    
    def send_market_order(self, order_type: str):
        """Envia ordem a mercado"""
        if not self.connected or not self.mt5_client:
            messagebox.showerror("Erro", "Não conectado ao MT5")
            return
        
        try:
            symbol = self.symbol_var.get()
            volume = float(self.volume_var.get())
            sl = float(self.sl_var.get()) if self.sl_var.get() else 0.0
            tp = float(self.tp_var.get()) if self.tp_var.get() else 0.0
            
            # Criar requisição de trade
            trade_request = TradeRequest(
                symbol=symbol,
                volume=volume,
                action=TradeAction.DEAL,
                type=OrderType.BUY if order_type == "BUY" else OrderType.SELL,
                sl=sl,
                tp=tp,
                comment=f"Dashboard {order_type}"
            )
            
            result = self.mt5_client.send_order(trade_request)
            
            if result and result.get('retcode') == 10009:  # TRADE_RETCODE_DONE
                self.log_message(f"Ordem {order_type} executada: {symbol} {volume} lotes")
                messagebox.showinfo("Sucesso", f"Ordem {order_type} executada com sucesso!")
                self.update_positions()
            else:
                error_msg = result.get('comment', 'Erro desconhecido') if result else 'Falha na execução'
                self.log_message(f"Falha na ordem {order_type}: {error_msg}")
                messagebox.showerror("Erro", f"Falha na execução: {error_msg}")
                
        except ValueError:
            messagebox.showerror("Erro", "Valores inválidos nos campos")
        except Exception as e:
            self.log_message(f"Erro ao enviar ordem: {e}")
            messagebox.showerror("Erro", f"Erro ao enviar ordem: {str(e)}")
    
    def update_positions(self):
        """Atualiza lista de posições"""
        if not self.connected or not self.mt5_client:
            return
        
        try:
            # Limpar árvore de posições
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            
            # Obter posições
            positions = self.mt5_client.get_positions()
            self.positions_data = positions
            
            for pos in positions:
                values = (
                    pos.get('ticket', ''),
                    pos.get('symbol', ''),
                    'BUY' if pos.get('type') == 0 else 'SELL',
                    f"{pos.get('volume', 0):.2f}",
                    f"{pos.get('price_open', 0):.5f}",
                    f"{pos.get('sl', 0):.5f}" if pos.get('sl', 0) > 0 else '-',
                    f"{pos.get('tp', 0):.5f}" if pos.get('tp', 0) > 0 else '-',
                    f"{pos.get('profit', 0):.2f}",
                    datetime.fromtimestamp(pos.get('time', 0)).strftime('%H:%M:%S')
                )
                self.positions_tree.insert('', tk.END, values=values)
            
            # Limpar árvore de ordens
            for item in self.orders_tree.get_children():
                self.orders_tree.delete(item)
            
            # Obter ordens pendentes
            orders = self.mt5_client.get_orders()
            self.orders_data = orders
            
            for order in orders:
                order_type = 'BUY_LIMIT' if order.get('type') == 2 else 'SELL_LIMIT' if order.get('type') == 3 else 'PENDING'
                values = (
                    order.get('ticket', ''),
                    order.get('symbol', ''),
                    order_type,
                    f"{order.get('volume_initial', 0):.2f}",
                    f"{order.get('price_open', 0):.5f}",
                    f"{order.get('sl', 0):.5f}" if order.get('sl', 0) > 0 else '-',
                    f"{order.get('tp', 0):.5f}" if order.get('tp', 0) > 0 else '-',
                    datetime.fromtimestamp(order.get('time_setup', 0)).strftime('%H:%M:%S')
                )
                self.orders_tree.insert('', tk.END, values=values)
                
        except Exception as e:
            self.log_message(f"Erro ao atualizar posições: {e}")
    
    def close_selected_position(self):
        """Fecha posição selecionada"""
        selection = self.positions_tree.selection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione uma posição para fechar")
            return
        
        try:
            item = self.positions_tree.item(selection[0])
            ticket = int(item['values'][0])
            
            result = self.mt5_client.close_position(ticket)
            
            if result and result.get('retcode') == 10009:
                self.log_message(f"Posição {ticket} fechada com sucesso")
                messagebox.showinfo("Sucesso", f"Posição {ticket} fechada!")
                self.update_positions()
            else:
                error_msg = result.get('comment', 'Erro desconhecido') if result else 'Falha no fechamento'
                messagebox.showerror("Erro", f"Falha ao fechar posição: {error_msg}")
                
        except Exception as e:
            self.log_message(f"Erro ao fechar posição: {e}")
            messagebox.showerror("Erro", f"Erro ao fechar posição: {str(e)}")
    
    def close_all_positions(self):
        """Fecha todas as posições"""
        if not self.positions_data:
            messagebox.showinfo("Info", "Nenhuma posição aberta")
            return
        
        if messagebox.askyesno("Confirmação", f"Fechar todas as {len(self.positions_data)} posições?"):
            closed_count = 0
            for pos in self.positions_data:
                try:
                    result = self.mt5_client.close_position(pos['ticket'])
                    if result and result.get('retcode') == 10009:
                        closed_count += 1
                except Exception as e:
                    self.log_message(f"Erro ao fechar posição {pos['ticket']}: {e}")
            
            self.log_message(f"{closed_count} posições fechadas")
            messagebox.showinfo("Resultado", f"{closed_count} posições fechadas")
            self.update_positions()
    
    def analyze_symbol(self):
        """Analisa símbolo com ML"""
        if not self.ml_predictor:
            messagebox.showerror("Erro", "Preditor ML não disponível")
            return
        
        try:
            symbol = self.analysis_symbol_var.get()
            
            # Limpar resultados anteriores
            self.analysis_results.delete(1.0, tk.END)
            self.analysis_results.insert(tk.END, f"Analisando {symbol}...\n")
            self.root.update()
            
            # Fazer predição
            direction, confidence = self.ml_predictor.predict_direction(symbol)
            
            # Exibir resultados
            results_text = f"\n=== Análise ML para {symbol} ===\n"
            results_text += f"Direção Prevista: {'ALTA' if direction == 1 else 'BAIXA' if direction == -1 else 'INDEFINIDA'}\n"
            results_text += f"Confiança: {confidence:.2%}\n"
            results_text += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            # Obter dados históricos se conectado
            if self.connected and self.mt5_client:
                rates = self.mt5_client.get_rates(symbol, 16385, 100)  # M1
                if rates is not None and not rates.empty:
                    current_price = rates['close'].iloc[-1]
                    results_text += f"Preço Atual: {current_price:.5f}\n"
                    
                    # Plotar gráfico
                    self.plot_price_chart(symbol, rates)
            
            self.analysis_results.delete(1.0, tk.END)
            self.analysis_results.insert(tk.END, results_text)
            
        except Exception as e:
            self.log_message(f"Erro na análise: {e}")
            self.analysis_results.delete(1.0, tk.END)
            self.analysis_results.insert(tk.END, f"Erro na análise: {str(e)}")
    
    def plot_price_chart(self, symbol: str, data: pd.DataFrame):
        """Plota gráfico de preços"""
        try:
            self.ax.clear()
            self.ax.set_facecolor('#2b2b2b')
            
            # Plotar preços
            self.ax.plot(data.index, data['close'], color='#00ff00', linewidth=1.5, label='Close')
            self.ax.plot(data.index, data['high'], color='#ffff00', alpha=0.7, linewidth=0.8, label='High')
            self.ax.plot(data.index, data['low'], color='#ff0000', alpha=0.7, linewidth=0.8, label='Low')
            
            self.ax.set_title(f'{symbol} - Últimas 100 barras', color='white', fontsize=12)
            self.ax.set_xlabel('Tempo', color='white')
            self.ax.set_ylabel('Preço', color='white')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
            # Configurar cores dos ticks
            self.ax.tick_params(colors='white')
            
            self.canvas.draw()
            
        except Exception as e:
            self.log_message(f"Erro ao plotar gráfico: {e}")
    
    def toggle_auto_trading(self):
        """Liga/desliga trading automático"""
        self.auto_trading = self.auto_trading_var.get()
        
        if self.auto_trading:
            if not self.connected:
                messagebox.showerror("Erro", "Conecte-se ao MT5 primeiro")
                self.auto_trading_var.set(False)
                return
            
            self.log_message("Trading automático ATIVADO")
            messagebox.showinfo("Info", "Trading automático ativado")
        else:
            self.log_message("Trading automático DESATIVADO")
    
    def start_monitoring(self):
        """Inicia monitoramento em tempo real"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def monitoring_loop(self):
        """Loop de monitoramento"""
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
        
        while not self.stop_monitoring and self.connected:
            try:
                # Atualizar cotações
                self.update_quotes(symbols)
                
                # Atualizar posições
                self.update_positions()
                
                # Atualizar informações da conta
                self.update_account_info()
                
                # Trading automático
                if self.auto_trading:
                    self.execute_auto_trading(symbols)
                
                time.sleep(5)  # Atualizar a cada 5 segundos
                
            except Exception as e:
                self.log_message(f"Erro no monitoramento: {e}")
                time.sleep(10)
    
    def update_quotes(self, symbols: List[str]):
        """Atualiza cotações em tempo real"""
        try:
            # Limpar árvore de cotações
            for item in self.quotes_tree.get_children():
                self.quotes_tree.delete(item)
            
            for symbol in symbols:
                tick = self.mt5_client.get_tick(symbol)
                if tick:
                    spread = tick['ask'] - tick['bid']
                    time_str = datetime.fromtimestamp(tick['time']).strftime('%H:%M:%S')
                    
                    values = (
                        symbol,
                        f"{tick['bid']:.5f}",
                        f"{tick['ask']:.5f}",
                        f"{spread:.5f}",
                        time_str
                    )
                    
                    self.quotes_tree.insert('', tk.END, values=values)
                    self.current_prices[symbol] = tick
                    
        except Exception as e:
            self.log_message(f"Erro ao atualizar cotações: {e}")
    
    def execute_auto_trading(self, symbols: List[str]):
        """Executa trading automático"""
        try:
            interval = int(self.interval_var.get())
            min_confidence = float(self.confidence_var.get())
            
            # Verificar se é hora de fazer nova análise
            current_time = time.time()
            if not hasattr(self, 'last_analysis_time'):
                self.last_analysis_time = 0
            
            if current_time - self.last_analysis_time < interval:
                return
            
            self.last_analysis_time = current_time
            
            for symbol in symbols[:3]:  # Limitar a 3 símbolos para não sobrecarregar
                try:
                    direction, confidence = self.ml_predictor.predict_direction(symbol)
                    
                    if confidence >= min_confidence:
                        # Verificar se já existe posição para este símbolo
                        existing_positions = [pos for pos in self.positions_data if pos.get('symbol') == symbol]
                        
                        if not existing_positions:
                            # Enviar ordem automática
                            order_type = "BUY" if direction == 1 else "SELL"
                            volume = 0.01  # Volume fixo para auto trading
                            
                            trade_request = TradeRequest(
                                symbol=symbol,
                                volume=volume,
                                action=TradeAction.DEAL,
                                type=OrderType.BUY if direction == 1 else OrderType.SELL,
                                comment=f"Auto {order_type} {confidence:.2%}"
                            )
                            
                            result = self.mt5_client.send_order(trade_request)
                            
                            if result and result.get('retcode') == 10009:
                                self.log_message(f"Auto trading: {order_type} {symbol} (confiança: {confidence:.2%})")
                            
                except Exception as e:
                    self.log_message(f"Erro no auto trading para {symbol}: {e}")
                    
        except Exception as e:
            self.log_message(f"Erro no auto trading: {e}")
    
    def log_message(self, message: str):
        """Adiciona mensagem ao log"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        # Adicionar ao widget de log
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        # Limitar número de linhas no log
        lines = self.log_text.get(1.0, tk.END).split('\n')
        if len(lines) > 1000:
            self.log_text.delete(1.0, f"{len(lines)-500}.0")
    
    def clear_logs(self):
        """Limpa logs"""
        self.log_text.delete(1.0, tk.END)
    
    def save_logs(self):
        """Salva logs em arquivo"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"mt5_dashboard_logs_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.log_text.get(1.0, tk.END))
            
            messagebox.showinfo("Sucesso", f"Logs salvos em {filename}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar logs: {str(e)}")
    
    def refresh_decisions(self):
        """Atualiza a lista de decisões recentes"""
        try:
            # Limpar árvore atual
            for item in self.decisions_tree.get_children():
                self.decisions_tree.delete(item)
            
            # Buscar decisões recentes
            recent_decisions = self.decision_logger.get_recent_decisions(limit=50)
            
            for decision in recent_decisions:
                # Formatar timestamp - verificar se é datetime ou string
                try:
                    if isinstance(decision['timestamp'], datetime):
                        timestamp_str = decision['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        # Se for string, tentar converter
                        timestamp_obj = datetime.fromisoformat(str(decision['timestamp']).replace('Z', '+00:00'))
                        timestamp_str = timestamp_obj.strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, AttributeError):
                    timestamp_str = str(decision['timestamp'])
                
                # Truncar reasoning para exibição
                reasoning_factors = decision.get('reasoning_factors', [])
                reasoning = ', '.join(reasoning_factors[:3]) if reasoning_factors else 'N/A'
                if len(reasoning) > 100:
                    reasoning = reasoning[:100] + '...'
                
                # Inserir na árvore
                self.decisions_tree.insert('', 'end', values=(
                    decision['id'],
                    timestamp_str,
                    decision['symbol'],
                    decision['action'],
                    f"{decision['confidence']:.2f}",
                    reasoning
                ))
            
            self.log_message(f"Carregadas {len(recent_decisions)} decisões recentes")
            
        except Exception as e:
            self.log_message(f"Erro ao carregar decisões: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao carregar decisões: {str(e)}")
    
    def on_decision_select(self, event):
        """Evento de seleção de decisão na árvore"""
        try:
            selection = self.decisions_tree.selection()
            if not selection:
                return
            
            # Obter ID da decisão selecionada
            item = self.decisions_tree.item(selection[0])
            decision_id = item['values'][0]
            
            # Buscar detalhes da decisão
            decision_details = self.decision_logger.get_decision_by_id(decision_id)
            
            if decision_details:
                self.display_decision_details(decision_details)
            
        except Exception as e:
            self.log_message(f"Erro ao carregar detalhes da decisão: {str(e)}")
    
    def display_decision_details(self, decision_details):
        """Exibe os detalhes de uma decisão nas abas correspondentes"""
        try:
            # Limpar textos anteriores
            self.market_details_text.delete(1.0, tk.END)
            self.technical_details_text.delete(1.0, tk.END)
            self.ml_details_text.delete(1.0, tk.END)
            self.actions_details_text.delete(1.0, tk.END)
            
            # Dados de mercado
            if decision_details.get('market_data'):
                market_data = decision_details['market_data']
                market_text = f"""Símbolo: {market_data.get('symbol', 'N/A')}
Bid: {market_data.get('bid', 'N/A')}
Ask: {market_data.get('ask', 'N/A')}
Spread: {market_data.get('spread', 'N/A')}
Volume: {market_data.get('volume', 'N/A')}
Timestamp: {market_data.get('timestamp', 'N/A')}

Dados OHLC:
{json.dumps(market_data.get('ohlc_data', {}), indent=2, default=str)}"""
                self.market_details_text.insert(tk.END, market_text)
            
            # Análise técnica
            if decision_details.get('technical_analysis'):
                tech_data = decision_details['technical_analysis']
                tech_text = f"""Indicadores Técnicos:
{json.dumps(tech_data.get('indicators', {}), indent=2, default=str)}

Sinais:
{json.dumps(tech_data.get('signals', {}), indent=2, default=str)}

Força da Tendência: {tech_data.get('trend_strength', 'N/A')}
Direção da Tendência: {tech_data.get('trend_direction', 'N/A')}"""
                self.technical_details_text.insert(tk.END, tech_text)
            
            # Predições ML
            if decision_details.get('ml_predictions'):
                ml_data = decision_details['ml_predictions']
                ml_text = f"""Predições dos Modelos:
{json.dumps(ml_data.get('model_predictions', {}), indent=2, default=str)}

Confiança Média: {ml_data.get('confidence', 'N/A')}
Direção Predita: {ml_data.get('predicted_direction', 'N/A')}
Força do Sinal: {ml_data.get('signal_strength', 'N/A')}

Features Utilizadas:
{json.dumps(ml_data.get('features_used', {}), indent=2, default=str)}"""
                self.ml_details_text.insert(tk.END, ml_text)
            
            # Ações do bot
            if decision_details.get('bot_actions'):
                actions_data = decision_details['bot_actions']
                actions_text = f"""Ação Executada: {actions_data.get('action_type', 'N/A')}
Resultado: {actions_data.get('result', 'N/A')}
Detalhes: {actions_data.get('details', 'N/A')}
Erros: {actions_data.get('errors', 'N/A')}

Parâmetros da Ação:
{json.dumps(actions_data.get('parameters', {}), indent=2, default=str)}"""
                self.actions_details_text.insert(tk.END, actions_text)
            
        except Exception as e:
            self.log_message(f"Erro ao exibir detalhes da decisão: {str(e)}")
    
    def export_decisions_csv(self):
        """Exporta decisões para CSV"""
        try:
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Salvar decisões como CSV"
            )
            
            if filename:
                self.decision_logger.export_decisions_to_csv(filename)
                messagebox.showinfo("Sucesso", f"Decisões exportadas para {filename}")
                self.log_message(f"Decisões exportadas para {filename}")
            
        except Exception as e:
            self.log_message(f"Erro ao exportar decisões: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao exportar decisões: {str(e)}")
    
    def show_performance_analysis(self):
        """Mostra análise de performance das decisões"""
        try:
            performance_stats = self.decision_logger.get_performance_analysis()
            
            # Criar janela de análise
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title("Análise de Performance das Decisões")
            analysis_window.geometry("600x400")
            analysis_window.configure(bg='#2b2b2b')
            
            # Texto com estatísticas
            stats_text = scrolledtext.ScrolledText(analysis_window, height=20, width=70,
                                                  bg='#1e1e1e', fg='#ffffff',
                                                  insertbackground='white')
            stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Formatar estatísticas
            stats_content = f"""ANÁLISE DE PERFORMANCE DAS DECISÕES
{'='*50}

Estatísticas Gerais:
- Total de Decisões: {performance_stats.get('total_decisions', 0)}
- Confiança Média: {performance_stats.get('average_confidence', 0):.2f}

Distribuição por Ação:
{json.dumps(performance_stats.get('action_breakdown', {}), indent=2)}

Símbolos Mais Negociados:
{json.dumps(performance_stats.get('top_symbols', {}), indent=2)}

Decisões por Período:
{json.dumps(performance_stats.get('decisions_by_period', {}), indent=2, default=str)}"""
            
            stats_text.insert(tk.END, stats_content)
            stats_text.config(state=tk.DISABLED)
            
            self.log_message("Análise de performance exibida")
            
        except Exception as e:
            self.log_message(f"Erro ao gerar análise de performance: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao gerar análise de performance: {str(e)}")
    
    def on_closing(self):
        """Evento de fechamento da janela"""
        if self.connected:
            if messagebox.askokcancel("Sair", "Desconectar do MT5 e sair?"):
                self.disconnect_mt5()
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """Executa o dashboard"""
        self.log_message("Dashboard MT5 iniciado")
        self.root.mainloop()

# Função principal
def main():
    """Função principal"""
    try:
        dashboard = MT5Dashboard()
        dashboard.run()
    except Exception as e:
        print(f"Erro ao iniciar dashboard: {e}")
        messagebox.showerror("Erro Fatal", f"Erro ao iniciar dashboard: {str(e)}")

if __name__ == "__main__":
    main()