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

class MT5Dashboard:
    """
    Dashboard de controle para sistema de trading MT5
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MT5 Trading Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
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
        self.account_balance = tk.StringVar(value="0.00")
        self.account_equity = tk.StringVar(value="0.00")
        self.open_positions = tk.StringVar(value="0")
        self.daily_profit = tk.StringVar(value="0.00")
        self.auto_trading_status = tk.StringVar(value="Desabilitado")
        
        # Threads de atualização
        self.update_thread = None
        self.running = False
        
        # Criar interface
        self._create_interface()
        
        # Configurar callbacks
        self._setup_callbacks()
        
        # Iniciar thread de atualização
        self._start_update_thread()
    
    def _create_interface(self):
        """Cria a interface do dashboard"""
        # Estilo
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar cores
        style.configure('Title.TLabel', font=('Arial', 12, 'bold'), background='#2b2b2b', foreground='white')
        style.configure('Status.TLabel', font=('Arial', 10), background='#2b2b2b', foreground='#00ff00')
        style.configure('Error.TLabel', font=('Arial', 10), background='#2b2b2b', foreground='#ff0000')
        style.configure('Info.TLabel', font=('Arial', 9), background='#2b2b2b', foreground='#cccccc')
        
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Criar seções
        self._create_connection_section(main_frame)
        self._create_account_section(main_frame)
        self._create_trading_section(main_frame)
        self._create_positions_section(main_frame)
        self._create_signals_section(main_frame)
        self._create_risk_section(main_frame)
        self._create_logs_section(main_frame)
        self._create_control_section(main_frame)
    
    def _create_connection_section(self, parent):
        """Seção de conexão"""
        frame = ttk.LabelFrame(parent, text="Conexão MT5", padding=10)
        frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Status da conexão
        ttk.Label(frame, text="Status:", style='Info.TLabel').grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.connection_status, style='Status.TLabel').grid(row=0, column=1, sticky="w", padx=10)
        
        # Botões de conexão
        ttk.Button(frame, text="Conectar", command=self._connect_mt5).grid(row=0, column=2, padx=5)
        ttk.Button(frame, text="Desconectar", command=self._disconnect_mt5).grid(row=0, column=3, padx=5)
        ttk.Button(frame, text="Reconectar", command=self._reconnect_mt5).grid(row=0, column=4, padx=5)
    
    def _create_account_section(self, parent):
        """Seção de informações da conta"""
        frame = ttk.LabelFrame(parent, text="Informações da Conta", padding=10)
        frame.grid(row=1, column=0, sticky="ew", pady=5, padx=(0, 5))
        
        # Informações da conta
        info_frame = ttk.Frame(frame)
        info_frame.pack(fill=tk.X)
        
        ttk.Label(info_frame, text="Saldo:", style='Info.TLabel').grid(row=0, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.account_balance, style='Status.TLabel').grid(row=0, column=1, sticky="w", padx=10)
        
        ttk.Label(info_frame, text="Equity:", style='Info.TLabel').grid(row=1, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.account_equity, style='Status.TLabel').grid(row=1, column=1, sticky="w", padx=10)
        
        ttk.Label(info_frame, text="Posições:", style='Info.TLabel').grid(row=2, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.open_positions, style='Status.TLabel').grid(row=2, column=1, sticky="w", padx=10)
        
        ttk.Label(info_frame, text="Lucro Diário:", style='Info.TLabel').grid(row=3, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.daily_profit, style='Status.TLabel').grid(row=3, column=1, sticky="w", padx=10)
    
    def _create_trading_section(self, parent):
        """Seção de controle de trading"""
        frame = ttk.LabelFrame(parent, text="Controle de Trading", padding=10)
        frame.grid(row=1, column=1, sticky="ew", pady=5, padx=(5, 0))
        
        # Auto trading
        ttk.Label(frame, text="Auto Trading:", style='Info.TLabel').grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.auto_trading_status, style='Status.TLabel').grid(row=0, column=1, sticky="w", padx=10)
        
        # Botões de controle
        control_frame = ttk.Frame(frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(control_frame, text="Habilitar Auto Trading", 
                  command=self._enable_auto_trading).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Desabilitar Auto Trading", 
                  command=self._disable_auto_trading).pack(side=tk.LEFT, padx=5)
        
        # Botões de ação
        action_frame = ttk.Frame(frame)
        action_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        ttk.Button(action_frame, text="Fechar Todas Posições", 
                  command=self._close_all_positions).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Análise Manual", 
                  command=self._manual_analysis).pack(side=tk.LEFT, padx=5)
    
    def _create_positions_section(self, parent):
        """Seção de posições abertas"""
        frame = ttk.LabelFrame(parent, text="Posições Abertas", padding=10)
        frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Treeview para posições
        columns = ('Symbol', 'Type', 'Volume', 'Price', 'Current', 'Profit', 'Time')
        self.positions_tree = ttk.Treeview(frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=100)
        
        # Scrollbar
        positions_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=positions_scroll.set)
        
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        positions_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_signals_section(self, parent):
        """Seção de sinais"""
        frame = ttk.LabelFrame(parent, text="Sinais de Trading", padding=10)
        frame.grid(row=3, column=0, sticky="ew", pady=5, padx=(0, 5))
        
        # Treeview para sinais
        columns = ('Time', 'Symbol', 'Action', 'Confidence', 'Status')
        self.signals_tree = ttk.Treeview(frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.signals_tree.heading(col, text=col)
            self.signals_tree.column(col, width=80)
        
        # Scrollbar
        signals_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.signals_tree.yview)
        self.signals_tree.configure(yscrollcommand=signals_scroll.set)
        
        self.signals_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        signals_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_risk_section(self, parent):
        """Seção de gestão de risco"""
        frame = ttk.LabelFrame(parent, text="Gestão de Risco", padding=10)
        frame.grid(row=3, column=1, sticky="ew", pady=5, padx=(5, 0))
        
        # Indicadores de risco
        self.risk_level = tk.StringVar(value="LOW")
        self.drawdown = tk.StringVar(value="0.00%")
        self.daily_loss = tk.StringVar(value="0.00%")
        
        ttk.Label(frame, text="Nível de Risco:", style='Info.TLabel').grid(row=0, column=0, sticky="w")
        self.risk_label = ttk.Label(frame, textvariable=self.risk_level, style='Status.TLabel')
        self.risk_label.grid(row=0, column=1, sticky="w", padx=10)
        
        ttk.Label(frame, text="Drawdown:", style='Info.TLabel').grid(row=1, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.drawdown, style='Info.TLabel').grid(row=1, column=1, sticky="w", padx=10)
        
        ttk.Label(frame, text="Perda Diária:", style='Info.TLabel').grid(row=2, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.daily_loss, style='Info.TLabel').grid(row=2, column=1, sticky="w", padx=10)
        
        # Alertas recentes
        ttk.Label(frame, text="Alertas Recentes:", style='Info.TLabel').grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 5))
        
        self.alerts_listbox = tk.Listbox(frame, height=4, bg='#3b3b3b', fg='white')
        self.alerts_listbox.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
    
    def _create_logs_section(self, parent):
        """Seção de logs"""
        frame = ttk.LabelFrame(parent, text="Logs do Sistema", padding=10)
        frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Text widget para logs
        self.logs_text = tk.Text(frame, height=8, bg='#1e1e1e', fg='#cccccc', 
                                font=('Consolas', 9))
        
        # Scrollbar para logs
        logs_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.logs_text.yview)
        self.logs_text.configure(yscrollcommand=logs_scroll.set)
        
        self.logs_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        logs_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_control_section(self, parent):
        """Seção de controles"""
        frame = ttk.Frame(parent)
        frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)
        
        # Botões de controle
        ttk.Button(frame, text="Exportar Logs", command=self._export_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Configurações", command=self._open_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Relatório", command=self._generate_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Sobre", command=self._show_about).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_bar = ttk.Label(frame, text="Sistema iniciado", style='Info.TLabel')
        self.status_bar.pack(side=tk.RIGHT, padx=10)
    
    def _setup_callbacks(self):
        """Configura callbacks do sistema"""
        # Callback para fechar aplicação
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
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
                time.sleep(2)  # Atualizar a cada 2 segundos
            except Exception as e:
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
            
        except Exception as e:
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
                current_price = self.data_provider.get_current_price(pos['symbol']) if self.data_provider else None
                current = current_price['bid'] if current_price else pos['price_open']
                
                self.positions_tree.insert('', 'end', values=(
                    pos['symbol'],
                    pos['type'],
                    f"{pos['volume']:.2f}",
                    f"{pos['price_open']:.5f}",
                    f"{current:.5f}",
                    f"${pos['profit']:.2f}",
                    pos['time'].strftime('%H:%M:%S')
                ))
                
        except Exception as e:
            self.logger.log(LogLevel.ERROR, EventType.SYSTEM, "Dashboard", 
                          f"Erro ao atualizar posições: {e}", exception=e)
    
    def _update_signals(self):
        """Atualiza lista de sinais"""
        if not self.signal_integrator:
            return
            
        try:
            # Obter sinais recentes do histórico
            recent_signals = [s for s in self.signal_integrator.signal_history 
                            if (datetime.now() - s.timestamp).seconds < 3600]
            
            # Limpar treeview
            for item in self.signals_tree.get_children():
                self.signals_tree.delete(item)
            
            # Adicionar sinais
            for signal in recent_signals[-10:]:  # Últimos 10 sinais
                status = "Executado" if signal.symbol in self.signal_integrator.active_signals else "Pendente"
                
                self.signals_tree.insert('', 'end', values=(
                    signal.timestamp.strftime('%H:%M:%S'),
                    signal.symbol,
                    signal.action.value,
                    f"{signal.confidence:.1%}",
                    status
                ))
                
        except Exception as e:
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
            
            # Atualizar cor do label baseado no risco
            if risk_level == RiskLevel.CRITICAL:
                self.risk_label.configure(style='Error.TLabel')
            elif risk_level == RiskLevel.HIGH:
                self.risk_label.configure(foreground='orange')
            else:
                self.risk_label.configure(style='Status.TLabel')
            
            # Atualizar drawdown e perda diária
            self.drawdown.set(f"{risk_summary.get('drawdown', 0):.2%}")
            self.daily_loss.set(f"{risk_summary.get('daily_loss', 0):.2%}")
            
            # Atualizar alertas
            self.alerts_listbox.delete(0, tk.END)
            for alert in self.risk_manager.alerts[-5:]:  # Últimos 5 alertas
                alert_text = f"[{alert.timestamp.strftime('%H:%M')}] {alert.level.value}: {alert.message}"
                self.alerts_listbox.insert(tk.END, alert_text)
                
        except Exception as e:
            self.logger.log(LogLevel.ERROR, EventType.SYSTEM, "Dashboard", 
                          f"Erro ao atualizar informações de risco: {e}", exception=e)
    
    def _update_logs(self):
        """Atualiza logs na interface"""
        try:
            recent_logs = self.logger.get_recent_logs(minutes=30)
            
            # Limpar text widget
            self.logs_text.delete(1.0, tk.END)
            
            # Adicionar logs recentes
            for log_entry in recent_logs[-20:]:  # Últimos 20 logs
                timestamp = log_entry.timestamp.strftime('%H:%M:%S')
                log_line = f"[{timestamp}] {log_entry.level.value} - {log_entry.component}: {log_entry.message}\n"
                
                # Colorir baseado no nível
                if log_entry.level == LogLevel.ERROR:
                    self.logs_text.insert(tk.END, log_line, 'error')
                elif log_entry.level == LogLevel.WARNING:
                    self.logs_text.insert(tk.END, log_line, 'warning')
                else:
                    self.logs_text.insert(tk.END, log_line)
            
            # Configurar tags de cor
            self.logs_text.tag_config('error', foreground='#ff6b6b')
            self.logs_text.tag_config('warning', foreground='#ffa500')
            
            # Scroll para o final
            self.logs_text.see(tk.END)
            
        except Exception as e:
            print(f"Erro ao atualizar logs: {e}")
    
    def _connect_mt5(self):
        """Conecta ao MT5"""
        try:
            self.connector = MT5Connector()
            if self.connector.connect():
                self.connected = True
                self.connection_status.set("Conectado")
                
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
                self.status_bar.configure(text="Conectado ao MT5")
                
            else:
                messagebox.showerror("Erro", "Falha ao conectar com MT5")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na conexão: {e}")
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
            self.status_bar.configure(text="Desconectado do MT5")
            
            self.logger.log(LogLevel.INFO, EventType.CONNECTION, "Dashboard", "Desconectado do MT5")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na desconexão: {e}")
    
    def _reconnect_mt5(self):
        """Reconecta ao MT5"""
        self._disconnect_mt5()
        time.sleep(1)
        self._connect_mt5()
    
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
        
        self.logger.log(LogLevel.INFO, EventType.SYSTEM, "Dashboard", "Auto trading habilitado")
    
    def _disable_auto_trading(self):
        """Desabilita auto trading"""
        self.auto_trading_enabled = False
        self.auto_trading_status.set("Desabilitado")
        
        if self.signal_integrator:
            self.signal_integrator.config.enable_auto_trading = False
            self.signal_integrator.stop_analysis_loop()
        
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
                
                if self.logger.export_logs(start_date, end_date, filename, format_type):
                    messagebox.showinfo("Sucesso", f"Logs exportados para {filename}")
                else:
                    messagebox.showerror("Erro", "Falha ao exportar logs")
                    
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na exportação: {e}")
    
    def _open_settings(self):
        """Abre janela de configurações"""
        messagebox.showinfo("Configurações", "Funcionalidade em desenvolvimento")
    
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
            report_window.geometry("600x400")
            
            text_widget = tk.Text(report_window, wrap=tk.WORD, font=('Consolas', 10))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text_widget.insert(1.0, report)
            text_widget.configure(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar relatório: {e}")
    
    def _show_about(self):
        """Mostra informações sobre o sistema"""
        about_text = """MT5 Trading Dashboard v1.0

Sistema integrado de trading automatizado para MetaTrader 5

Funcionalidades:
• Conexão com MT5
• Análise técnica automatizada
• Gestão de risco
• Execução de trades
• Monitoramento em tempo real
• Sistema de logging avançado

Desenvolvido para trading profissional de Forex"""
        
        messagebox.showinfo("Sobre", about_text)
    
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
    # Executar dashboard
    dashboard = MT5Dashboard()
    dashboard.run()