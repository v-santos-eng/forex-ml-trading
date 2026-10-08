import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import uuid
import json
from pathlib import Path

class NotificationLevel(Enum):
    """Níveis de notificação"""
    DEBUG = "debug"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    TRADE = "trade"

class NotificationCategory(Enum):
    """Categorias de notificação"""
    SYSTEM = "system"
    TRADING = "trading"
    CONNECTION = "connection"
    ANALYSIS = "analysis"
    PERFORMANCE = "performance"
    USER = "user"

@dataclass
class NotificationAction:
    """Ação de notificação"""
    text: str
    callback: Callable
    style: str = "primary"  # primary, secondary, danger

@dataclass
class Notification:
    """Estrutura completa de notificação"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    message: str = ""
    level: NotificationLevel = NotificationLevel.INFO
    category: NotificationCategory = NotificationCategory.SYSTEM
    timestamp: datetime = field(default_factory=datetime.now)
    duration: int = 5000  # ms para toast
    persistent: bool = False  # Se deve ficar no centro de notificações
    show_toast: bool = True  # Se deve mostrar toast
    actions: List[NotificationAction] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    read: bool = False
    dismissed: bool = False

class NotificationManager:
    """Gerenciador central de notificações"""
    
    def __init__(self, max_notifications: int = 100):
        self.notifications: List[Notification] = []
        self.max_notifications = max_notifications
        self.observers: List[Callable] = []
        self.toast_queue: List[Notification] = []
        self.settings = {
            'show_toasts': True,
            'show_system': True,
            'show_trading': True,
            'show_connection': True,
            'show_analysis': True,
            'show_performance': True,
            'show_user': True,
            'sound_enabled': True,
            'auto_dismiss_duration': 5000
        }
        
        # Carregar configurações
        self._load_settings()
        
        # Thread para processar toasts
        self._toast_thread = None
        self._toast_running = False
    
    def add_observer(self, callback: Callable):
        """Adiciona observador para mudanças"""
        self.observers.append(callback)
    
    def remove_observer(self, callback: Callable):
        """Remove observador"""
        if callback in self.observers:
            self.observers.remove(callback)
    
    def _notify_observers(self, notification: Notification, action: str):
        """Notifica observadores sobre mudanças"""
        for observer in self.observers:
            try:
                observer(notification, action)
            except Exception as e:
                print(f"Erro ao notificar observador: {e}")
    
    def add_notification(self, notification: Notification) -> str:
        """Adiciona nova notificação"""
        # Verificar se categoria está habilitada
        category_key = f"show_{notification.category.value}"
        if not self.settings.get(category_key, True):
            return notification.id
        
        # Adicionar à lista
        self.notifications.insert(0, notification)
        
        # Manter limite
        if len(self.notifications) > self.max_notifications:
            removed = self.notifications.pop()
            self._notify_observers(removed, 'removed')
        
        # Notificar observadores
        self._notify_observers(notification, 'added')
        
        # Adicionar à fila de toast se necessário
        if notification.show_toast and self.settings.get('show_toasts', True):
            self.toast_queue.append(notification)
            self._start_toast_processor()
        
        return notification.id
    
    def _start_toast_processor(self):
        """Inicia processador de toasts"""
        if not self._toast_running:
            self._toast_running = True
            self._toast_thread = threading.Thread(target=self._process_toasts, daemon=True)
            self._toast_thread.start()
    
    def _process_toasts(self):
        """Processa fila de toasts"""
        while self._toast_running and self.toast_queue:
            notification = self.toast_queue.pop(0)
            self._notify_observers(notification, 'toast')
            
            # Aguardar antes do próximo toast
            time.sleep(0.5)
        
        self._toast_running = False
    
    def get_notifications(self, category: Optional[NotificationCategory] = None,
                        level: Optional[NotificationLevel] = None,
                        unread_only: bool = False) -> List[Notification]:
        """Obtém notificações filtradas"""
        filtered = self.notifications
        
        if category:
            filtered = [n for n in filtered if n.category == category]
        
        if level:
            filtered = [n for n in filtered if n.level == level]
        
        if unread_only:
            filtered = [n for n in filtered if not n.read]
        
        return filtered
    
    def mark_as_read(self, notification_id: str):
        """Marca notificação como lida"""
        for notification in self.notifications:
            if notification.id == notification_id:
                notification.read = True
                self._notify_observers(notification, 'updated')
                break
    
    def mark_all_as_read(self, category: Optional[NotificationCategory] = None):
        """Marca todas como lidas"""
        for notification in self.notifications:
            if not category or notification.category == category:
                notification.read = True
                self._notify_observers(notification, 'updated')
    
    def dismiss_notification(self, notification_id: str):
        """Descarta notificação"""
        for i, notification in enumerate(self.notifications):
            if notification.id == notification_id:
                notification.dismissed = True
                removed = self.notifications.pop(i)
                self._notify_observers(removed, 'dismissed')
                break
    
    def clear_notifications(self, category: Optional[NotificationCategory] = None,
                          older_than: Optional[datetime] = None):
        """Limpa notificações"""
        to_remove = []
        
        for notification in self.notifications:
            should_remove = True
            
            if category and notification.category != category:
                should_remove = False
            
            if older_than and notification.timestamp > older_than:
                should_remove = False
            
            if should_remove:
                to_remove.append(notification)
        
        for notification in to_remove:
            self.notifications.remove(notification)
            self._notify_observers(notification, 'removed')
    
    def get_unread_count(self, category: Optional[NotificationCategory] = None) -> int:
        """Obtém contagem de não lidas"""
        notifications = self.get_notifications(category=category, unread_only=True)
        return len(notifications)
    
    def update_settings(self, settings: Dict[str, Any]):
        """Atualiza configurações"""
        self.settings.update(settings)
        self._save_settings()
    
    def _load_settings(self):
        """Carrega configurações do arquivo"""
        try:
            settings_file = Path("notification_settings.json")
            if settings_file.exists():
                with open(settings_file, 'r', encoding='utf-8') as f:
                    saved_settings = json.load(f)
                    self.settings.update(saved_settings)
        except Exception as e:
            print(f"Erro ao carregar configurações de notificação: {e}")
    
    def _save_settings(self):
        """Salva configurações no arquivo"""
        try:
            settings_file = Path("notification_settings.json")
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar configurações de notificação: {e}")
    
    # Métodos de conveniência para criar notificações
    def info(self, title: str, message: str, **kwargs) -> str:
        """Cria notificação de informação"""
        notification = Notification(
            title=title,
            message=message,
            level=NotificationLevel.INFO,
            **kwargs
        )
        return self.add_notification(notification)
    
    def success(self, title: str, message: str, **kwargs) -> str:
        """Cria notificação de sucesso"""
        notification = Notification(
            title=title,
            message=message,
            level=NotificationLevel.SUCCESS,
            **kwargs
        )
        return self.add_notification(notification)
    
    def warning(self, title: str, message: str, **kwargs) -> str:
        """Cria notificação de aviso"""
        notification = Notification(
            title=title,
            message=message,
            level=NotificationLevel.WARNING,
            **kwargs
        )
        return self.add_notification(notification)
    
    def error(self, title: str, message: str, **kwargs) -> str:
        """Cria notificação de erro"""
        notification = Notification(
            title=title,
            message=message,
            level=NotificationLevel.ERROR,
            persistent=True,
            **kwargs
        )
        return self.add_notification(notification)
    
    def trade_alert(self, title: str, message: str, **kwargs) -> str:
        """Cria alerta de trading"""
        notification = Notification(
            title=title,
            message=message,
            level=NotificationLevel.TRADE,
            category=NotificationCategory.TRADING,
            persistent=True,
            **kwargs
        )
        return self.add_notification(notification)

class NotificationWidget(tk.Frame):
    """Widget de notificação individual"""
    
    def __init__(self, parent, notification: Notification, 
                 on_dismiss: Optional[Callable] = None,
                 on_action: Optional[Callable] = None, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.notification = notification
        self.on_dismiss = on_dismiss
        self.on_action = on_action
        
        # Cores por nível
        self.level_colors = {
            NotificationLevel.DEBUG: '#6c757d',
            NotificationLevel.INFO: '#0078d4',
            NotificationLevel.SUCCESS: '#107c10',
            NotificationLevel.WARNING: '#ff8c00',
            NotificationLevel.ERROR: '#d13438',
            NotificationLevel.CRITICAL: '#8b0000',
            NotificationLevel.TRADE: '#6b46c1'
        }
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Cria widgets da notificação"""
        self.configure(bg='#2d2d2d', relief='solid', borderwidth=1)
        
        # Barra lateral colorida
        accent_color = self.level_colors.get(self.notification.level, '#0078d4')
        side_bar = tk.Frame(self, bg=accent_color, width=4)
        side_bar.pack(side=tk.LEFT, fill=tk.Y)
        
        # Conteúdo principal
        content_frame = tk.Frame(self, bg='#2d2d2d')
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=10)
        
        # Cabeçalho
        header_frame = tk.Frame(content_frame, bg='#2d2d2d')
        header_frame.pack(fill=tk.X)
        
        # Ícone do nível
        level_icons = {
            NotificationLevel.DEBUG: '🔍',
            NotificationLevel.INFO: 'ℹ️',
            NotificationLevel.SUCCESS: '✅',
            NotificationLevel.WARNING: '⚠️',
            NotificationLevel.ERROR: '❌',
            NotificationLevel.CRITICAL: '🚨',
            NotificationLevel.TRADE: '💹'
        }
        
        icon = level_icons.get(self.notification.level, 'ℹ️')
        icon_label = tk.Label(header_frame, text=icon, font=('Segoe UI', 12),
                             bg='#2d2d2d', fg='white')
        icon_label.pack(side=tk.LEFT, padx=(0, 8))
        
        # Título
        title_label = tk.Label(header_frame, text=self.notification.title,
                              font=('Segoe UI', 10, 'bold'),
                              bg='#2d2d2d', fg='white')
        title_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Timestamp
        time_str = self.notification.timestamp.strftime('%H:%M:%S')
        time_label = tk.Label(header_frame, text=time_str,
                             font=('Segoe UI', 8),
                             bg='#2d2d2d', fg='#999999')
        time_label.pack(side=tk.RIGHT)
        
        # Mensagem
        if self.notification.message:
            message_label = tk.Label(content_frame, text=self.notification.message,
                                    font=('Segoe UI', 9),
                                    bg='#2d2d2d', fg='#cccccc',
                                    wraplength=300, justify=tk.LEFT)
            message_label.pack(fill=tk.X, pady=(8, 0), anchor=tk.W)
        
        # Ações
        if self.notification.actions:
            actions_frame = tk.Frame(content_frame, bg='#2d2d2d')
            actions_frame.pack(fill=tk.X, pady=(10, 0))
            
            for action in self.notification.actions:
                self._create_action_button(actions_frame, action)
        
        # Botões de controle
        controls_frame = tk.Frame(self, bg='#2d2d2d')
        controls_frame.pack(side=tk.RIGHT, padx=8)
        
        # Botão marcar como lida
        if not self.notification.read:
            read_btn = tk.Button(controls_frame, text="👁",
                               font=('Segoe UI', 10),
                               bg='#3d3d3d', fg='#cccccc',
                               borderwidth=0, width=3,
                               command=self._mark_as_read)
            read_btn.pack(pady=2)
        
        # Botão dismiss
        dismiss_btn = tk.Button(controls_frame, text="×",
                              font=('Segoe UI', 12, 'bold'),
                              bg='#d13438', fg='white',
                              borderwidth=0, width=3,
                              command=self._dismiss)
        dismiss_btn.pack(pady=2)
    
    def _create_action_button(self, parent, action: NotificationAction):
        """Cria botão de ação"""
        style_colors = {
            'primary': '#0078d4',
            'secondary': '#6c757d',
            'danger': '#d13438'
        }
        
        bg_color = style_colors.get(action.style, '#0078d4')
        
        btn = tk.Button(parent, text=action.text,
                       font=('Segoe UI', 8),
                       bg=bg_color, fg='white',
                       borderwidth=0, padx=12, pady=4,
                       command=lambda: self._handle_action(action))
        btn.pack(side=tk.LEFT, padx=(0, 8))
    
    def _handle_action(self, action: NotificationAction):
        """Manipula ação"""
        try:
            action.callback()
            if self.on_action:
                self.on_action(self.notification, action)
        except Exception as e:
            print(f"Erro ao executar ação: {e}")
    
    def _mark_as_read(self):
        """Marca como lida"""
        self.notification.read = True
        if self.on_action:
            self.on_action(self.notification, 'read')
    
    def _dismiss(self):
        """Descarta notificação"""
        if self.on_dismiss:
            self.on_dismiss(self.notification)

class NotificationPanel(tk.Frame):
    """Painel de notificações para o dashboard"""
    
    def __init__(self, parent, notification_manager: NotificationManager, **kwargs):
        super().__init__(parent, bg='#1e1e1e', **kwargs)
        
        self.notification_manager = notification_manager
        self.notification_widgets = {}
        self.current_filter = None
        
        # Registrar como observador
        self.notification_manager.add_observer(self._on_notification_change)
        
        self._create_widgets()
        self._refresh_notifications()
    
    def _create_widgets(self):
        """Cria widgets do painel"""
        # Cabeçalho
        header_frame = tk.Frame(self, bg='#2d2d2d', height=50)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        header_frame.pack_propagate(False)
        
        # Título
        title_label = tk.Label(header_frame, text="Notificações",
                              font=('Segoe UI', 14, 'bold'),
                              bg='#2d2d2d', fg='white')
        title_label.pack(side=tk.LEFT, padx=15, pady=15)
        
        # Contador
        self.count_label = tk.Label(header_frame, text="0",
                                   font=('Segoe UI', 10),
                                   bg='#0078d4', fg='white',
                                   padx=8, pady=2)
        self.count_label.pack(side=tk.LEFT, padx=10)
        
        # Botões de controle
        controls_frame = tk.Frame(header_frame, bg='#2d2d2d')
        controls_frame.pack(side=tk.RIGHT, padx=15)
        
        # Filtros
        filter_btn = tk.Button(controls_frame, text="🔍",
                              font=('Segoe UI', 10),
                              bg='#3d3d3d', fg='white',
                              borderwidth=0, padx=8,
                              command=self._show_filters)
        filter_btn.pack(side=tk.LEFT, padx=2)
        
        # Marcar todas como lidas
        read_all_btn = tk.Button(controls_frame, text="👁",
                               font=('Segoe UI', 10),
                               bg='#3d3d3d', fg='white',
                               borderwidth=0, padx=8,
                               command=self._mark_all_read)
        read_all_btn.pack(side=tk.LEFT, padx=2)
        
        # Limpar todas
        clear_btn = tk.Button(controls_frame, text="🗑",
                             font=('Segoe UI', 10),
                             bg='#d13438', fg='white',
                             borderwidth=0, padx=8,
                             command=self._clear_all)
        clear_btn.pack(side=tk.LEFT, padx=2)
        
        # Área de notificações com scroll
        self.canvas = tk.Canvas(self, bg='#1e1e1e', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='#1e1e1e')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=10)
        
        # Bind scroll do mouse
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
    
    def _on_mousewheel(self, event):
        """Manipula scroll do mouse"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _on_notification_change(self, notification: Notification, action: str):
        """Callback para mudanças nas notificações"""
        # Executar na thread principal
        self.after_idle(lambda: self._refresh_notifications())
    
    def _refresh_notifications(self):
        """Atualiza lista de notificações"""
        # Limpar widgets existentes
        for widget in self.notification_widgets.values():
            widget.destroy()
        self.notification_widgets.clear()
        
        # Obter notificações filtradas
        notifications = self.notification_manager.get_notifications()
        
        # Aplicar filtro atual
        if self.current_filter:
            notifications = [n for n in notifications if self._matches_filter(n)]
        
        # Atualizar contador
        unread_count = len([n for n in notifications if not n.read])
        self.count_label.config(text=str(unread_count))
        
        # Criar widgets
        for notification in notifications:
            if not notification.dismissed:
                widget = NotificationWidget(
                    self.scrollable_frame,
                    notification,
                    on_dismiss=self._dismiss_notification,
                    on_action=self._handle_notification_action
                )
                widget.pack(fill=tk.X, pady=2, padx=5)
                self.notification_widgets[notification.id] = widget
        
        # Atualizar scroll
        self.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _matches_filter(self, notification: Notification) -> bool:
        """Verifica se notificação corresponde ao filtro"""
        if not self.current_filter:
            return True
        
        # Implementar lógica de filtro
        return True
    
    def _dismiss_notification(self, notification: Notification):
        """Descarta notificação"""
        self.notification_manager.dismiss_notification(notification.id)
    
    def _handle_notification_action(self, notification: Notification, action):
        """Manipula ação da notificação"""
        if action == 'read':
            self.notification_manager.mark_as_read(notification.id)
    
    def _show_filters(self):
        """Mostra opções de filtro"""
        # Implementar janela de filtros
        pass
    
    def _mark_all_read(self):
        """Marca todas como lidas"""
        self.notification_manager.mark_all_as_read()
    
    def _clear_all(self):
        """Limpa todas as notificações"""
        self.notification_manager.clear_notifications()

# Instância global do gerenciador
_notification_manager = None

def get_notification_manager() -> NotificationManager:
    """Obtém instância global do gerenciador"""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager

# Funções de conveniência
def notify_info(title: str, message: str, **kwargs) -> str:
    """Notificação de informação"""
    return get_notification_manager().info(title, message, **kwargs)

def notify_success(title: str, message: str, **kwargs) -> str:
    """Notificação de sucesso"""
    return get_notification_manager().success(title, message, **kwargs)

def notify_warning(title: str, message: str, **kwargs) -> str:
    """Notificação de aviso"""
    return get_notification_manager().warning(title, message, **kwargs)

def notify_error(title: str, message: str, **kwargs) -> str:
    """Notificação de erro"""
    return get_notification_manager().error(title, message, **kwargs)

def notify_trade(title: str, message: str, **kwargs) -> str:
    """Notificação de trading"""
    return get_notification_manager().trade_alert(title, message, **kwargs)

# Exemplo de uso
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sistema de Notificações")
    root.geometry("600x800")
    root.configure(bg='#1e1e1e')
    
    # Criar gerenciador
    manager = get_notification_manager()
    
    # Criar painel
    panel = NotificationPanel(root, manager)
    panel.pack(fill=tk.BOTH, expand=True)
    
    # Adicionar notificações de teste
    def add_test_notifications():
        notify_info("Sistema Iniciado", "Dashboard conectado com sucesso")
        notify_success("Trade Executado", "Compra EURUSD executada com sucesso")
        notify_warning("Margem Baixa", "Margem disponível abaixo de 20%")
        notify_error("Conexão Perdida", "Falha na conexão com o servidor MT5")
        notify_trade("Sinal Detectado", "Oportunidade de compra em GBPUSD")
    
    # Botão para teste
    test_btn = tk.Button(root, text="Adicionar Notificações de Teste",
                        command=add_test_notifications,
                        bg='#0078d4', fg='white', borderwidth=0, pady=10)
    test_btn.pack(pady=10)
    
    root.mainloop()