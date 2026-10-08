import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import threading
import time

class UserPreferences:
    """Gerenciador de preferências específicas do usuário"""
    
    def __init__(self, settings_manager):
        self.settings_manager = settings_manager
        self.preferences_file = "user_preferences.json"
        self.preferences = {}
        self.auto_save_timer = None
        
        # Carregar preferências
        self.load_preferences()
        
        # Iniciar auto-save se habilitado
        if self.settings_manager.get('backup_settings', True):
            self._start_auto_save()
    
    def load_preferences(self):
        """Carrega preferências do usuário"""
        try:
            if os.path.exists(self.preferences_file):
                with open(self.preferences_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.preferences = data.get('preferences', {})
            else:
                self._create_default_preferences()
        except Exception as e:
            print(f"Erro ao carregar preferências: {e}")
            self._create_default_preferences()
    
    def _create_default_preferences(self):
        """Cria preferências padrão"""
        self.preferences = {
            # Layout e Interface
            'dashboard_layout': 'standard',
            'panel_positions': {
                'trading': {'x': 0, 'y': 0, 'width': 400, 'height': 300},
                'positions': {'x': 400, 'y': 0, 'width': 400, 'height': 300},
                'analysis': {'x': 0, 'y': 300, 'width': 800, 'height': 300}
            },
            'visible_panels': {
                'trading': True,
                'positions': True,
                'analysis': True,
                'notifications': True,
                'logs': True
            },
            
            # Favoritos e Atalhos
            'favorite_symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
            'quick_actions': [
                {'name': 'Close All', 'action': 'close_all_positions'},
                {'name': 'Export Positions', 'action': 'export_positions'},
                {'name': 'Risk Check', 'action': 'check_risk'}
            ],
            'keyboard_shortcuts': {
                'Ctrl+E': 'export_data',
                'Ctrl+R': 'refresh_data',
                'Ctrl+S': 'save_settings',
                'F5': 'reconnect_mt5'
            },
            
            # Alertas e Notificações
            'alert_preferences': {
                'sound_enabled': True,
                'popup_enabled': True,
                'email_enabled': False,
                'profit_threshold': 100.0,
                'loss_threshold': -50.0,
                'margin_warning': 80.0
            },
            
            # Dados e Histórico
            'data_preferences': {
                'auto_refresh': True,
                'refresh_interval': 2,
                'keep_history_days': 30,
                'max_log_entries': 1000,
                'chart_timeframe': 'M15'
            },
            
            # Trading
            'trading_preferences': {
                'default_symbol': 'EURUSD',
                'default_lot_size': 0.1,
                'max_spread': 3.0,
                'auto_sl_pips': 20,
                'auto_tp_pips': 40,
                'risk_per_trade': 2.0
            },
            
            # Personalização
            'ui_customization': {
                'show_tooltips': True,
                'animate_updates': True,
                'compact_mode': False,
                'show_grid_lines': True,
                'highlight_changes': True
            },
            
            # Sessão
            'session_data': {
                'last_login': None,
                'session_count': 0,
                'total_runtime': 0,
                'last_backup': None
            }
        }
    
    def get_preference(self, key: str, default=None):
        """Obtém preferência do usuário"""
        keys = key.split('.')
        value = self.preferences
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_preference(self, key: str, value):
        """Define preferência do usuário"""
        keys = key.split('.')
        current = self.preferences
        
        try:
            # Navegar até o penúltimo nível
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Definir valor
            current[keys[-1]] = value
            return True
            
        except Exception as e:
            print(f"Erro ao definir preferência {key}: {e}")
            return False
    
    def save_preferences(self):
        """Salva preferências no arquivo"""
        try:
            # Atualizar dados da sessão
            self.preferences['session_data']['last_backup'] = datetime.now().isoformat()
            
            data = {
                'preferences': self.preferences,
                'saved_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.preferences_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Erro ao salvar preferências: {e}")
            return False
    
    def _start_auto_save(self):
        """Inicia salvamento automático"""
        def auto_save_worker():
            while True:
                time.sleep(300)  # Salvar a cada 5 minutos
                self.save_preferences()
        
        self.auto_save_timer = threading.Thread(target=auto_save_worker, daemon=True)
        self.auto_save_timer.start()
    
    def add_favorite_symbol(self, symbol: str):
        """Adiciona símbolo aos favoritos"""
        favorites = self.get_preference('favorite_symbols', [])
        if symbol not in favorites:
            favorites.append(symbol)
            self.set_preference('favorite_symbols', favorites)
            return True
        return False
    
    def remove_favorite_symbol(self, symbol: str):
        """Remove símbolo dos favoritos"""
        favorites = self.get_preference('favorite_symbols', [])
        if symbol in favorites:
            favorites.remove(symbol)
            self.set_preference('favorite_symbols', favorites)
            return True
        return False
    
    def get_panel_position(self, panel_name: str):
        """Obtém posição salva do painel"""
        return self.get_preference(f'panel_positions.{panel_name}', {})
    
    def save_panel_position(self, panel_name: str, x: int, y: int, width: int, height: int):
        """Salva posição do painel"""
        position = {'x': x, 'y': y, 'width': width, 'height': height}
        self.set_preference(f'panel_positions.{panel_name}', position)
    
    def is_panel_visible(self, panel_name: str):
        """Verifica se painel está visível"""
        return self.get_preference(f'visible_panels.{panel_name}', True)
    
    def set_panel_visibility(self, panel_name: str, visible: bool):
        """Define visibilidade do painel"""
        self.set_preference(f'visible_panels.{panel_name}', visible)
    
    def get_quick_actions(self):
        """Obtém ações rápidas configuradas"""
        return self.get_preference('quick_actions', [])
    
    def add_quick_action(self, name: str, action: str):
        """Adiciona ação rápida"""
        actions = self.get_quick_actions()
        actions.append({'name': name, 'action': action})
        self.set_preference('quick_actions', actions)
    
    def get_keyboard_shortcuts(self):
        """Obtém atalhos de teclado"""
        return self.get_preference('keyboard_shortcuts', {})
    
    def update_session_stats(self, runtime_seconds: int):
        """Atualiza estatísticas da sessão"""
        session_count = self.get_preference('session_data.session_count', 0) + 1
        total_runtime = self.get_preference('session_data.total_runtime', 0) + runtime_seconds
        
        self.set_preference('session_data.session_count', session_count)
        self.set_preference('session_data.total_runtime', total_runtime)
        self.set_preference('session_data.last_login', datetime.now().isoformat())

class PreferencesDialog:
    """Dialog para configurar preferências do usuário"""
    
    def __init__(self, parent, user_preferences: UserPreferences):
        self.parent = parent
        self.user_preferences = user_preferences
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Preferências do Usuário")
        self.dialog.geometry("600x500")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self._create_widgets()
        self._load_current_preferences()
        self._center_dialog()
    
    def _create_widgets(self):
        """Cria widgets do dialog"""
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Notebook para categorias
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Aba Interface
        self._create_interface_tab(notebook)
        
        # Aba Trading
        self._create_trading_tab(notebook)
        
        # Aba Alertas
        self._create_alerts_tab(notebook)
        
        # Aba Favoritos
        self._create_favorites_tab(notebook)
        
        # Frame de botões
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Salvar", 
                  command=self._save_preferences).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancelar", 
                  command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Restaurar Padrões", 
                  command=self._reset_preferences).pack(side=tk.LEFT, padx=5)
    
    def _create_interface_tab(self, notebook):
        """Cria aba de interface"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Interface")
        
        # Layout
        layout_frame = ttk.LabelFrame(frame, text="Layout")
        layout_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.compact_mode_var = tk.BooleanVar()
        ttk.Checkbutton(layout_frame, text="Modo Compacto", 
                       variable=self.compact_mode_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.show_tooltips_var = tk.BooleanVar()
        ttk.Checkbutton(layout_frame, text="Mostrar Dicas", 
                       variable=self.show_tooltips_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.animate_updates_var = tk.BooleanVar()
        ttk.Checkbutton(layout_frame, text="Animar Atualizações", 
                       variable=self.animate_updates_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Painéis Visíveis
        panels_frame = ttk.LabelFrame(frame, text="Painéis Visíveis")
        panels_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.panel_vars = {}
        panels = ['trading', 'positions', 'analysis', 'notifications', 'logs']
        
        for panel in panels:
            var = tk.BooleanVar()
            self.panel_vars[panel] = var
            ttk.Checkbutton(panels_frame, text=panel.title(), 
                           variable=var).pack(anchor=tk.W, padx=5, pady=2)
    
    def _create_trading_tab(self, notebook):
        """Cria aba de trading"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Trading")
        
        # Configurações padrão
        defaults_frame = ttk.LabelFrame(frame, text="Configurações Padrão")
        defaults_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Símbolo padrão
        ttk.Label(defaults_frame, text="Símbolo Padrão:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.default_symbol_var = tk.StringVar()
        symbol_combo = ttk.Combobox(defaults_frame, textvariable=self.default_symbol_var,
                                   values=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'])
        symbol_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Tamanho do lote
        ttk.Label(defaults_frame, text="Tamanho do Lote:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.lot_size_var = tk.DoubleVar()
        ttk.Spinbox(defaults_frame, from_=0.01, to=10.0, increment=0.01, 
                   textvariable=self.lot_size_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Risco por trade
        ttk.Label(defaults_frame, text="Risco por Trade (%):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.risk_var = tk.DoubleVar()
        ttk.Spinbox(defaults_frame, from_=0.1, to=10.0, increment=0.1, 
                   textvariable=self.risk_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
    
    def _create_alerts_tab(self, notebook):
        """Cria aba de alertas"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Alertas")
        
        # Tipos de alerta
        alert_frame = ttk.LabelFrame(frame, text="Tipos de Alerta")
        alert_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.sound_alert_var = tk.BooleanVar()
        ttk.Checkbutton(alert_frame, text="Alertas Sonoros", 
                       variable=self.sound_alert_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.popup_alert_var = tk.BooleanVar()
        ttk.Checkbutton(alert_frame, text="Alertas Popup", 
                       variable=self.popup_alert_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Limites
        limits_frame = ttk.LabelFrame(frame, text="Limites de Alerta")
        limits_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(limits_frame, text="Lucro ($):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.profit_threshold_var = tk.DoubleVar()
        ttk.Entry(limits_frame, textvariable=self.profit_threshold_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(limits_frame, text="Perda ($):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.loss_threshold_var = tk.DoubleVar()
        ttk.Entry(limits_frame, textvariable=self.loss_threshold_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
    
    def _create_favorites_tab(self, notebook):
        """Cria aba de favoritos"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Favoritos")
        
        # Lista de símbolos favoritos
        favorites_frame = ttk.LabelFrame(frame, text="Símbolos Favoritos")
        favorites_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Listbox com scrollbar
        list_frame = ttk.Frame(favorites_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.favorites_listbox = tk.Listbox(list_frame)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.favorites_listbox.yview)
        self.favorites_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.favorites_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Botões
        btn_frame = ttk.Frame(favorites_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Adicionar", 
                  command=self._add_favorite).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Remover", 
                  command=self._remove_favorite).pack(side=tk.LEFT, padx=2)
    
    def _load_current_preferences(self):
        """Carrega preferências atuais nos widgets"""
        # Interface
        self.compact_mode_var.set(self.user_preferences.get_preference('ui_customization.compact_mode', False))
        self.show_tooltips_var.set(self.user_preferences.get_preference('ui_customization.show_tooltips', True))
        self.animate_updates_var.set(self.user_preferences.get_preference('ui_customization.animate_updates', True))
        
        # Painéis
        for panel, var in self.panel_vars.items():
            var.set(self.user_preferences.is_panel_visible(panel))
        
        # Trading
        self.default_symbol_var.set(self.user_preferences.get_preference('trading_preferences.default_symbol', 'EURUSD'))
        self.lot_size_var.set(self.user_preferences.get_preference('trading_preferences.default_lot_size', 0.1))
        self.risk_var.set(self.user_preferences.get_preference('trading_preferences.risk_per_trade', 2.0))
        
        # Alertas
        self.sound_alert_var.set(self.user_preferences.get_preference('alert_preferences.sound_enabled', True))
        self.popup_alert_var.set(self.user_preferences.get_preference('alert_preferences.popup_enabled', True))
        self.profit_threshold_var.set(self.user_preferences.get_preference('alert_preferences.profit_threshold', 100.0))
        self.loss_threshold_var.set(self.user_preferences.get_preference('alert_preferences.loss_threshold', -50.0))
        
        # Favoritos
        favorites = self.user_preferences.get_preference('favorite_symbols', [])
        for symbol in favorites:
            self.favorites_listbox.insert(tk.END, symbol)
    
    def _save_preferences(self):
        """Salva preferências"""
        try:
            # Interface
            self.user_preferences.set_preference('ui_customization.compact_mode', self.compact_mode_var.get())
            self.user_preferences.set_preference('ui_customization.show_tooltips', self.show_tooltips_var.get())
            self.user_preferences.set_preference('ui_customization.animate_updates', self.animate_updates_var.get())
            
            # Painéis
            for panel, var in self.panel_vars.items():
                self.user_preferences.set_panel_visibility(panel, var.get())
            
            # Trading
            self.user_preferences.set_preference('trading_preferences.default_symbol', self.default_symbol_var.get())
            self.user_preferences.set_preference('trading_preferences.default_lot_size', self.lot_size_var.get())
            self.user_preferences.set_preference('trading_preferences.risk_per_trade', self.risk_var.get())
            
            # Alertas
            self.user_preferences.set_preference('alert_preferences.sound_enabled', self.sound_alert_var.get())
            self.user_preferences.set_preference('alert_preferences.popup_enabled', self.popup_alert_var.get())
            self.user_preferences.set_preference('alert_preferences.profit_threshold', self.profit_threshold_var.get())
            self.user_preferences.set_preference('alert_preferences.loss_threshold', self.loss_threshold_var.get())
            
            # Favoritos
            favorites = []
            for i in range(self.favorites_listbox.size()):
                favorites.append(self.favorites_listbox.get(i))
            self.user_preferences.set_preference('favorite_symbols', favorites)
            
            # Salvar no arquivo
            if self.user_preferences.save_preferences():
                messagebox.showinfo("Sucesso", "Preferências salvas com sucesso!")
                self.dialog.destroy()
            else:
                messagebox.showerror("Erro", "Falha ao salvar preferências")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar preferências: {e}")
    
    def _reset_preferences(self):
        """Restaura preferências padrão"""
        if messagebox.askyesno("Confirmar", "Restaurar todas as preferências para os valores padrão?"):
            self.user_preferences._create_default_preferences()
            self._load_current_preferences()
    
    def _add_favorite(self):
        """Adiciona símbolo favorito"""
        symbol = tk.simpledialog.askstring("Adicionar Favorito", "Digite o símbolo:")
        if symbol:
            symbol = symbol.upper()
            if symbol not in [self.favorites_listbox.get(i) for i in range(self.favorites_listbox.size())]:
                self.favorites_listbox.insert(tk.END, symbol)
    
    def _remove_favorite(self):
        """Remove símbolo favorito"""
        selection = self.favorites_listbox.curselection()
        if selection:
            self.favorites_listbox.delete(selection[0])
    
    def _center_dialog(self):
        """Centraliza o dialog"""
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")