import tkinter as tk
from tkinter import ttk, colorchooser, messagebox, filedialog
import json
import os
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from datetime import datetime

class SettingType(Enum):
    """Tipos de configuração disponíveis"""
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    COLOR = "color"
    FONT = "font"
    CHOICE = "choice"
    PATH = "path"

@dataclass
class SettingDefinition:
    """Definição de uma configuração"""
    key: str
    name: str
    description: str
    setting_type: SettingType
    default_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    choices: Optional[list] = None
    category: str = "Geral"
    requires_restart: bool = False
    validation_func: Optional[Callable] = None

class SettingsManager:
    """Gerenciador de configurações"""
    
    def __init__(self, config_file: str = "dashboard_settings.json"):
        self.config_file = config_file
        self.settings = {}
        self.observers = []
        self.setting_definitions = {}
        
        # Definir configurações padrão
        self._define_default_settings()
        
        # Carregar configurações salvas
        self.load_settings()
    
    def _define_default_settings(self):
        """Define configurações padrão do sistema"""
        definitions = [
            # Aparência
            SettingDefinition(
                "theme_mode", "Modo do Tema", "Tema claro ou escuro",
                SettingType.CHOICE, "dark", choices=["light", "dark"], category="Aparência"
            ),
            SettingDefinition(
                "primary_color", "Cor Primária", "Cor principal da interface",
                SettingType.COLOR, "#2E86AB", category="Aparência"
            ),
            SettingDefinition(
                "accent_color", "Cor de Destaque", "Cor de destaque para botões",
                SettingType.COLOR, "#A23B72", category="Aparência"
            ),
            SettingDefinition(
                "font_family", "Família da Fonte", "Fonte principal da interface",
                SettingType.CHOICE, "Segoe UI", 
                choices=["Segoe UI", "Arial", "Helvetica", "Calibri", "Tahoma"], 
                category="Aparência", requires_restart=True
            ),
            SettingDefinition(
                "font_size", "Tamanho da Fonte", "Tamanho base da fonte",
                SettingType.INTEGER, 10, min_value=8, max_value=16, 
                category="Aparência", requires_restart=True
            ),
            SettingDefinition(
                "window_opacity", "Opacidade da Janela", "Transparência da janela principal",
                SettingType.FLOAT, 1.0, min_value=0.7, max_value=1.0, category="Aparência"
            ),
            
            # Interface
            SettingDefinition(
                "auto_refresh", "Atualização Automática", "Ativar atualização automática de dados",
                SettingType.BOOLEAN, True, category="Interface"
            ),
            SettingDefinition(
                "refresh_interval", "Intervalo de Atualização", "Intervalo em segundos para atualização",
                SettingType.INTEGER, 2, min_value=1, max_value=60, category="Interface"
            ),
            SettingDefinition(
                "show_notifications", "Mostrar Notificações", "Exibir notificações do sistema",
                SettingType.BOOLEAN, True, category="Interface"
            ),
            SettingDefinition(
                "notification_duration", "Duração das Notificações", "Tempo em segundos para notificações",
                SettingType.INTEGER, 5, min_value=1, max_value=30, category="Interface"
            ),
            SettingDefinition(
                "confirm_actions", "Confirmar Ações", "Solicitar confirmação para ações críticas",
                SettingType.BOOLEAN, True, category="Interface"
            ),
            SettingDefinition(
                "remember_window_size", "Lembrar Tamanho da Janela", "Salvar posição e tamanho da janela",
                SettingType.BOOLEAN, True, category="Interface"
            ),
            
            # Trading
            SettingDefinition(
                "auto_trading", "Trading Automático", "Ativar trading automático",
                SettingType.BOOLEAN, False, category="Trading"
            ),
            SettingDefinition(
                "max_positions", "Máximo de Posições", "Número máximo de posições simultâneas",
                SettingType.INTEGER, 5, min_value=1, max_value=20, category="Trading"
            ),
            SettingDefinition(
                "default_lot_size", "Tamanho Padrão do Lote", "Tamanho padrão para novas posições",
                SettingType.FLOAT, 0.1, min_value=0.01, max_value=10.0, category="Trading"
            ),
            SettingDefinition(
                "risk_percentage", "Percentual de Risco", "Percentual máximo de risco por trade",
                SettingType.FLOAT, 2.0, min_value=0.1, max_value=10.0, category="Trading"
            ),
            
            # Dados
            SettingDefinition(
                "data_provider_auto_start", "Iniciar Provedor de Dados Automaticamente", 
                "Iniciar coleta de dados MT5 na inicialização",
                SettingType.BOOLEAN, True, category="Dados"
            ),
            SettingDefinition(
                "data_history_days", "Dias de Histórico", "Número de dias de histórico a manter",
                SettingType.INTEGER, 30, min_value=1, max_value=365, category="Dados"
            ),
            SettingDefinition(
                "export_format", "Formato de Exportação Padrão", "Formato padrão para exportações",
                SettingType.CHOICE, "csv", choices=["csv", "xlsx", "json"], category="Dados"
            ),
            SettingDefinition(
                "log_level", "Nível de Log", "Nível de detalhamento dos logs",
                SettingType.CHOICE, "INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                category="Dados"
            ),
            
            # Avançado
            SettingDefinition(
                "debug_mode", "Modo Debug", "Ativar modo de depuração",
                SettingType.BOOLEAN, False, category="Avançado", requires_restart=True
            ),
            SettingDefinition(
                "performance_monitoring", "Monitoramento de Performance", 
                "Ativar monitoramento de performance",
                SettingType.BOOLEAN, False, category="Avançado"
            ),
            SettingDefinition(
                "backup_settings", "Backup Automático", "Fazer backup automático das configurações",
                SettingType.BOOLEAN, True, category="Avançado"
            )
        ]
        
        for definition in definitions:
            self.setting_definitions[definition.key] = definition
            if definition.key not in self.settings:
                self.settings[definition.key] = definition.default_value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtém valor de configuração"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any, notify: bool = True) -> bool:
        """Define valor de configuração"""
        try:
            # Validar se a configuração existe
            if key not in self.setting_definitions:
                raise ValueError(f"Configuração '{key}' não encontrada")
            
            definition = self.setting_definitions[key]
            
            # Validar tipo e valor
            if not self._validate_value(definition, value):
                return False
            
            # Aplicar validação customizada se existir
            if definition.validation_func and not definition.validation_func(value):
                return False
            
            old_value = self.settings.get(key)
            self.settings[key] = value
            
            # Notificar observadores
            if notify and old_value != value:
                self._notify_observers(key, old_value, value)
            
            return True
            
        except Exception as e:
            print(f"Erro ao definir configuração {key}: {e}")
            return False
    
    def _validate_value(self, definition: SettingDefinition, value: Any) -> bool:
        """Valida valor de configuração"""
        try:
            # Validar tipo
            if definition.setting_type == SettingType.BOOLEAN:
                if not isinstance(value, bool):
                    return False
            elif definition.setting_type == SettingType.INTEGER:
                if not isinstance(value, int):
                    return False
                if definition.min_value is not None and value < definition.min_value:
                    return False
                if definition.max_value is not None and value > definition.max_value:
                    return False
            elif definition.setting_type == SettingType.FLOAT:
                if not isinstance(value, (int, float)):
                    return False
                if definition.min_value is not None and value < definition.min_value:
                    return False
                if definition.max_value is not None and value > definition.max_value:
                    return False
            elif definition.setting_type == SettingType.STRING:
                if not isinstance(value, str):
                    return False
            elif definition.setting_type == SettingType.CHOICE:
                if definition.choices and value not in definition.choices:
                    return False
            elif definition.setting_type == SettingType.COLOR:
                if not isinstance(value, str) or not value.startswith('#'):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def add_observer(self, callback: Callable[[str, Any, Any], None]):
        """Adiciona observador para mudanças de configuração"""
        self.observers.append(callback)
    
    def remove_observer(self, callback: Callable[[str, Any, Any], None]):
        """Remove observador"""
        if callback in self.observers:
            self.observers.remove(callback)
    
    def _notify_observers(self, key: str, old_value: Any, new_value: Any):
        """Notifica observadores sobre mudanças"""
        for observer in self.observers:
            try:
                observer(key, old_value, new_value)
            except Exception as e:
                print(f"Erro ao notificar observador: {e}")
    
    def save_settings(self) -> bool:
        """Salva configurações no arquivo"""
        try:
            # Criar backup se habilitado
            if self.get('backup_settings', True) and os.path.exists(self.config_file):
                backup_file = f"{self.config_file}.backup"
                import shutil
                shutil.copy2(self.config_file, backup_file)
            
            # Salvar configurações
            config_data = {
                'settings': self.settings,
                'last_saved': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Erro ao salvar configurações: {e}")
            return False
    
    def load_settings(self) -> bool:
        """Carrega configurações do arquivo"""
        try:
            if not os.path.exists(self.config_file):
                return True  # Usar configurações padrão
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Carregar configurações salvas
            saved_settings = config_data.get('settings', {})
            
            for key, value in saved_settings.items():
                if key in self.setting_definitions:
                    self.settings[key] = value
            
            return True
            
        except Exception as e:
            print(f"Erro ao carregar configurações: {e}")
            return False
    
    def reset_to_defaults(self, category: Optional[str] = None) -> bool:
        """Restaura configurações padrão"""
        try:
            for key, definition in self.setting_definitions.items():
                if category is None or definition.category == category:
                    self.settings[key] = definition.default_value
            
            # Notificar observadores
            for observer in self.observers:
                try:
                    observer('reset', None, category)
                except Exception:
                    pass
            
            return True
            
        except Exception as e:
            print(f"Erro ao restaurar configurações: {e}")
            return False
    
    def export_settings(self, filepath: str) -> bool:
        """Exporta configurações para arquivo"""
        try:
            export_data = {
                'settings': self.settings,
                'exported_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Erro ao exportar configurações: {e}")
            return False
    
    def import_settings(self, filepath: str) -> bool:
        """Importa configurações de arquivo"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            imported_settings = import_data.get('settings', {})
            
            for key, value in imported_settings.items():
                if key in self.setting_definitions:
                    self.set(key, value, notify=False)
            
            # Notificar observadores sobre importação
            for observer in self.observers:
                try:
                    observer('import', None, filepath)
                except Exception:
                    pass
            
            return True
            
        except Exception as e:
            print(f"Erro ao importar configurações: {e}")
            return False
    
    def get_categories(self) -> list:
        """Obtém lista de categorias disponíveis"""
        categories = set()
        for definition in self.setting_definitions.values():
            categories.add(definition.category)
        return sorted(list(categories))
    
    def get_settings_by_category(self, category: str) -> Dict[str, SettingDefinition]:
        """Obtém configurações por categoria"""
        return {
            key: definition for key, definition in self.setting_definitions.items()
            if definition.category == category
        }

class SettingsDialog:
    """Dialog de configurações"""
    
    def __init__(self, parent, settings_manager: SettingsManager):
        self.parent = parent
        self.settings_manager = settings_manager
        self.widgets = {}
        self.changes_made = False
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Configurações")
        self.dialog.geometry("700x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self._create_widgets()
        self._center_dialog()
        
        # Interceptar fechamento
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_widgets(self):
        """Cria widgets do dialog"""
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Notebook para categorias
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Criar abas por categoria
        categories = self.settings_manager.get_categories()
        for category in categories:
            self._create_category_tab(category)
        
        # Frame de botões
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Botões de ação
        ttk.Button(button_frame, text="Aplicar", 
                  command=self._apply_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="OK", 
                  command=self._ok_clicked).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancelar", 
                  command=self._cancel_clicked).pack(side=tk.RIGHT, padx=5)
        
        # Botões de utilitários
        ttk.Button(button_frame, text="Restaurar Padrões", 
                  command=self._reset_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exportar...", 
                  command=self._export_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Importar...", 
                  command=self._import_settings).pack(side=tk.LEFT, padx=5)
    
    def _create_category_tab(self, category: str):
        """Cria aba para categoria"""
        # Frame da aba
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text=category)
        
        # Scrollable frame
        canvas = tk.Canvas(tab_frame)
        scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Criar widgets para configurações da categoria
        settings = self.settings_manager.get_settings_by_category(category)
        row = 0
        
        for key, definition in settings.items():
            self._create_setting_widget(scrollable_frame, key, definition, row)
            row += 1
    
    def _create_setting_widget(self, parent, key: str, definition: SettingDefinition, row: int):
        """Cria widget para configuração específica"""
        # Label
        label_frame = ttk.Frame(parent)
        label_frame.grid(row=row, column=0, sticky=(tk.W, tk.N), padx=5, pady=5)
        
        ttk.Label(label_frame, text=definition.name, 
                 font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W)
        ttk.Label(label_frame, text=definition.description, 
                 font=('Segoe UI', 8), foreground='gray').pack(anchor=tk.W)
        
        # Widget de entrada
        widget_frame = ttk.Frame(parent)
        widget_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        current_value = self.settings_manager.get(key)
        
        if definition.setting_type == SettingType.BOOLEAN:
            var = tk.BooleanVar(value=current_value)
            widget = ttk.Checkbutton(widget_frame, variable=var)
            widget.pack(anchor=tk.W)
            self.widgets[key] = var
            
        elif definition.setting_type == SettingType.INTEGER:
            var = tk.IntVar(value=current_value)
            widget = ttk.Spinbox(widget_frame, from_=definition.min_value or 0, 
                               to=definition.max_value or 100, textvariable=var, width=10)
            widget.pack(anchor=tk.W)
            self.widgets[key] = var
            
        elif definition.setting_type == SettingType.FLOAT:
            var = tk.DoubleVar(value=current_value)
            widget = ttk.Spinbox(widget_frame, from_=definition.min_value or 0.0, 
                               to=definition.max_value or 100.0, increment=0.1, 
                               textvariable=var, width=10)
            widget.pack(anchor=tk.W)
            self.widgets[key] = var
            
        elif definition.setting_type == SettingType.STRING:
            var = tk.StringVar(value=current_value)
            widget = ttk.Entry(widget_frame, textvariable=var, width=30)
            widget.pack(anchor=tk.W)
            self.widgets[key] = var
            
        elif definition.setting_type == SettingType.CHOICE:
            var = tk.StringVar(value=current_value)
            widget = ttk.Combobox(widget_frame, textvariable=var, 
                                values=definition.choices, state="readonly", width=20)
            widget.pack(anchor=tk.W)
            self.widgets[key] = var
            
        elif definition.setting_type == SettingType.COLOR:
            var = tk.StringVar(value=current_value)
            
            color_frame = ttk.Frame(widget_frame)
            color_frame.pack(anchor=tk.W)
            
            entry = ttk.Entry(color_frame, textvariable=var, width=10)
            entry.pack(side=tk.LEFT, padx=(0, 5))
            
            color_btn = tk.Button(color_frame, text="🎨", width=3,
                                command=lambda: self._choose_color(var))
            color_btn.pack(side=tk.LEFT)
            
            self.widgets[key] = var
            
        elif definition.setting_type == SettingType.PATH:
            var = tk.StringVar(value=current_value)
            
            path_frame = ttk.Frame(widget_frame)
            path_frame.pack(anchor=tk.W, fill=tk.X)
            
            entry = ttk.Entry(path_frame, textvariable=var)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
            
            browse_btn = ttk.Button(path_frame, text="...", width=3,
                                  command=lambda: self._browse_path(var))
            browse_btn.pack(side=tk.RIGHT)
            
            self.widgets[key] = var
        
        # Configurar grid weights
        parent.columnconfigure(1, weight=1)
    
    def _choose_color(self, var: tk.StringVar):
        """Abre seletor de cor"""
        color = colorchooser.askcolor(initialcolor=var.get())
        if color[1]:  # Se uma cor foi selecionada
            var.set(color[1])
            self.changes_made = True
    
    def _browse_path(self, var: tk.StringVar):
        """Abre seletor de arquivo/pasta"""
        path = filedialog.askdirectory(initialdir=var.get())
        if path:
            var.set(path)
            self.changes_made = True
    
    def _apply_settings(self):
        """Aplica configurações"""
        restart_required = False
        
        for key, widget_var in self.widgets.items():
            definition = self.settings_manager.setting_definitions[key]
            new_value = widget_var.get()
            
            if self.settings_manager.set(key, new_value):
                if definition.requires_restart:
                    restart_required = True
            else:
                messagebox.showerror("Erro", f"Valor inválido para {definition.name}")
                return
        
        # Salvar configurações
        if self.settings_manager.save_settings():
            self.changes_made = False
            messagebox.showinfo("Sucesso", "Configurações aplicadas com sucesso!")
            
            if restart_required:
                messagebox.showwarning("Reinicialização Necessária", 
                                     "Algumas alterações requerem reinicialização do aplicativo.")
        else:
            messagebox.showerror("Erro", "Falha ao salvar configurações")
    
    def _ok_clicked(self):
        """Botão OK clicado"""
        self._apply_settings()
        if not self.changes_made:
            self.dialog.destroy()
    
    def _cancel_clicked(self):
        """Botão Cancelar clicado"""
        if self.changes_made:
            if messagebox.askyesno("Confirmar", "Descartar alterações?"):
                self.dialog.destroy()
        else:
            self.dialog.destroy()
    
    def _reset_defaults(self):
        """Restaura configurações padrão"""
        if messagebox.askyesno("Confirmar", "Restaurar todas as configurações para os valores padrão?"):
            # Obter categoria atual
            current_tab = self.notebook.tab(self.notebook.select(), "text")
            
            if messagebox.askyesno("Escopo", f"Restaurar apenas a categoria '{current_tab}'?"):
                self.settings_manager.reset_to_defaults(current_tab)
            else:
                self.settings_manager.reset_to_defaults()
            
            # Atualizar widgets
            self._update_widgets()
            self.changes_made = True
    
    def _export_settings(self):
        """Exporta configurações"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Exportar Configurações"
        )
        
        if filepath:
            if self.settings_manager.export_settings(filepath):
                messagebox.showinfo("Sucesso", f"Configurações exportadas para:\n{filepath}")
            else:
                messagebox.showerror("Erro", "Falha ao exportar configurações")
    
    def _import_settings(self):
        """Importa configurações"""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Importar Configurações"
        )
        
        if filepath:
            if messagebox.askyesno("Confirmar", "Importar configurações? Isso substituirá as configurações atuais."):
                if self.settings_manager.import_settings(filepath):
                    self._update_widgets()
                    self.changes_made = True
                    messagebox.showinfo("Sucesso", "Configurações importadas com sucesso!")
                else:
                    messagebox.showerror("Erro", "Falha ao importar configurações")
    
    def _update_widgets(self):
        """Atualiza valores dos widgets"""
        for key, widget_var in self.widgets.items():
            current_value = self.settings_manager.get(key)
            widget_var.set(current_value)
    
    def _center_dialog(self):
        """Centraliza o dialog"""
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
    
    def _on_closing(self):
        """Intercepta fechamento da janela"""
        self._cancel_clicked()

class ThemeManager:
    """Gerenciador de temas"""
    
    def __init__(self, settings_manager: SettingsManager):
        self.settings_manager = settings_manager
        self.current_theme = {}
        
        # Observar mudanças de configuração
        settings_manager.add_observer(self._on_setting_changed)
        
        # Aplicar tema inicial
        self._update_theme()
    
    def _on_setting_changed(self, key: str, old_value: Any, new_value: Any):
        """Callback para mudanças de configuração"""
        theme_keys = ['theme_mode', 'primary_color', 'accent_color', 'font_family', 'font_size']
        
        if key in theme_keys:
            self._update_theme()
    
    def _update_theme(self):
        """Atualiza tema atual"""
        mode = self.settings_manager.get('theme_mode', 'dark')
        
        if mode == 'dark':
            self.current_theme = {
                'bg_primary': '#1E1E1E',
                'bg_secondary': '#2D2D30',
                'bg_tertiary': '#3E3E42',
                'text_primary': '#FFFFFF',
                'text_secondary': '#CCCCCC',
                'primary': self.settings_manager.get('primary_color', '#2E86AB'),
                'accent': self.settings_manager.get('accent_color', '#A23B72'),
                'success': '#28A745',
                'warning': '#FFC107',
                'error': '#DC3545',
                'font_family': self.settings_manager.get('font_family', 'Segoe UI'),
                'font_size': self.settings_manager.get('font_size', 10)
            }
        else:  # light mode
            self.current_theme = {
                'bg_primary': '#FFFFFF',
                'bg_secondary': '#F8F9FA',
                'bg_tertiary': '#E9ECEF',
                'text_primary': '#212529',
                'text_secondary': '#6C757D',
                'primary': self.settings_manager.get('primary_color', '#007BFF'),
                'accent': self.settings_manager.get('accent_color', '#6F42C1'),
                'success': '#28A745',
                'warning': '#FFC107',
                'error': '#DC3545',
                'font_family': self.settings_manager.get('font_family', 'Segoe UI'),
                'font_size': self.settings_manager.get('font_size', 10)
            }
    
    def get_theme(self) -> Dict[str, Any]:
        """Obtém tema atual"""
        return self.current_theme.copy()
    
    def apply_to_widget(self, widget, style_type: str = 'default'):
        """Aplica tema a um widget"""
        try:
            if style_type == 'primary':
                widget.config(
                    bg=self.current_theme['primary'],
                    fg=self.current_theme['text_primary'],
                    font=(self.current_theme['font_family'], self.current_theme['font_size'])
                )
            elif style_type == 'secondary':
                widget.config(
                    bg=self.current_theme['bg_secondary'],
                    fg=self.current_theme['text_primary'],
                    font=(self.current_theme['font_family'], self.current_theme['font_size'])
                )
            else:  # default
                widget.config(
                    bg=self.current_theme['bg_primary'],
                    fg=self.current_theme['text_primary'],
                    font=(self.current_theme['font_family'], self.current_theme['font_size'])
                )
        except tk.TclError:
            pass  # Widget pode não suportar todas as opções