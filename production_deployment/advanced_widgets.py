import tkinter as tk
from tkinter import ttk
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import threading
import time

class NotificationType(Enum):
    """Tipos de notificação"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    TRADE = "trade"

@dataclass
class Notification:
    """Estrutura de notificação"""
    id: str
    type: NotificationType
    title: str
    message: str
    timestamp: datetime
    duration: int = 5000  # ms
    action_callback: Optional[callable] = None
    action_text: Optional[str] = None

class AnimatedProgressBar(ttk.Frame):
    """Barra de progresso animada personalizada"""
    
    def __init__(self, parent, width=200, height=20, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.width = width
        self.height = height
        self.progress = 0.0
        self.animated = False
        
        # Canvas para desenhar a barra
        self.canvas = tk.Canvas(self, width=width, height=height, 
                               highlightthickness=0, bg='#2d2d2d')
        self.canvas.pack()
        
        # Cores
        self.bg_color = '#3d3d3d'
        self.fill_color = '#0078d4'
        self.border_color = '#484848'
        
        self._draw_bar()
    
    def _draw_bar(self):
        """Desenha a barra de progresso"""
        self.canvas.delete('all')
        
        # Fundo
        self.canvas.create_rectangle(0, 0, self.width, self.height, 
                                   fill=self.bg_color, outline=self.border_color)
        
        # Progresso
        if self.progress > 0:
            fill_width = int(self.width * self.progress)
            self.canvas.create_rectangle(2, 2, fill_width-2, self.height-2, 
                                       fill=self.fill_color, outline='')
        
        # Texto
        text = f"{self.progress:.1%}"
        self.canvas.create_text(self.width//2, self.height//2, 
                              text=text, fill='white', font=('Segoe UI', 8))
    
    def set_progress(self, value: float):
        """Define o progresso (0.0 a 1.0)"""
        self.progress = max(0.0, min(1.0, value))
        self._draw_bar()
    
    def animate_to(self, target: float, duration: int = 1000):
        """Anima até o valor alvo"""
        if self.animated:
            return
        
        self.animated = True
        start_value = self.progress
        start_time = time.time()
        
        def animate():
            elapsed = (time.time() - start_time) * 1000
            if elapsed >= duration:
                self.set_progress(target)
                self.animated = False
                return
            
            # Interpolação suave
            t = elapsed / duration
            t = t * t * (3.0 - 2.0 * t)  # Smoothstep
            current = start_value + (target - start_value) * t
            self.set_progress(current)
            
            self.after(16, animate)  # ~60 FPS
        
        animate()

class CircularProgress(tk.Canvas):
    """Widget de progresso circular"""
    
    def __init__(self, parent, size=100, thickness=8, **kwargs):
        super().__init__(parent, width=size, height=size, 
                        highlightthickness=0, bg='#2d2d2d', **kwargs)
        
        self.size = size
        self.thickness = thickness
        self.progress = 0.0
        self.center = size // 2
        self.radius = (size - thickness) // 2
        
        # Cores
        self.bg_color = '#3d3d3d'
        self.fill_color = '#0078d4'
        self.text_color = '#ffffff'
        
        self._draw_circle()
    
    def _draw_circle(self):
        """Desenha o círculo de progresso"""
        self.delete('all')
        
        # Círculo de fundo
        x1 = self.center - self.radius
        y1 = self.center - self.radius
        x2 = self.center + self.radius
        y2 = self.center + self.radius
        
        self.create_oval(x1, y1, x2, y2, outline=self.bg_color, 
                        width=self.thickness, fill='')
        
        # Arco de progresso
        if self.progress > 0:
            extent = 360 * self.progress
            self.create_arc(x1, y1, x2, y2, start=90, extent=-extent,
                          outline=self.fill_color, width=self.thickness, 
                          style='arc')
        
        # Texto central
        text = f"{self.progress:.1%}"
        self.create_text(self.center, self.center, text=text, 
                        fill=self.text_color, font=('Segoe UI', 10, 'bold'))
    
    def set_progress(self, value: float):
        """Define o progresso (0.0 a 1.0)"""
        self.progress = max(0.0, min(1.0, value))
        self._draw_circle()

class LiveChart(tk.Canvas):
    """Gráfico de linha em tempo real"""
    
    def __init__(self, parent, width=400, height=200, max_points=100, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        highlightthickness=0, bg='#2d2d2d', **kwargs)
        
        self.width = width
        self.height = height
        self.max_points = max_points
        self.data_points = []
        self.labels = []
        
        # Margens
        self.margin_left = 50
        self.margin_right = 20
        self.margin_top = 20
        self.margin_bottom = 30
        
        # Área do gráfico
        self.chart_width = width - self.margin_left - self.margin_right
        self.chart_height = height - self.margin_top - self.margin_bottom
        
        # Cores
        self.line_color = '#0078d4'
        self.grid_color = '#484848'
        self.text_color = '#cccccc'
        self.bg_color = '#2d2d2d'
        
        self._draw_chart()
    
    def add_point(self, value: float, label: str = None):
        """Adiciona um ponto ao gráfico"""
        self.data_points.append(value)
        self.labels.append(label or datetime.now().strftime('%H:%M:%S'))
        
        # Manter apenas os últimos pontos
        if len(self.data_points) > self.max_points:
            self.data_points.pop(0)
            self.labels.pop(0)
        
        self._draw_chart()
    
    def _draw_chart(self):
        """Desenha o gráfico"""
        self.delete('all')
        
        if not self.data_points:
            self._draw_empty_chart()
            return
        
        # Calcular escala
        min_val = min(self.data_points)
        max_val = max(self.data_points)
        
        if min_val == max_val:
            min_val -= 1
            max_val += 1
        
        value_range = max_val - min_val
        
        # Desenhar grid
        self._draw_grid(min_val, max_val)
        
        # Desenhar linha
        if len(self.data_points) > 1:
            points = []
            for i, value in enumerate(self.data_points):
                x = self.margin_left + (i / (len(self.data_points) - 1)) * self.chart_width
                y = self.margin_top + self.chart_height - ((value - min_val) / value_range) * self.chart_height
                points.extend([x, y])
            
            if len(points) >= 4:
                self.create_line(points, fill=self.line_color, width=2, smooth=True)
        
        # Desenhar pontos
        for i, value in enumerate(self.data_points):
            x = self.margin_left + (i / max(1, len(self.data_points) - 1)) * self.chart_width
            y = self.margin_top + self.chart_height - ((value - min_val) / value_range) * self.chart_height
            
            self.create_oval(x-2, y-2, x+2, y+2, fill=self.line_color, outline='')
    
    def _draw_grid(self, min_val: float, max_val: float):
        """Desenha o grid do gráfico"""
        # Linhas horizontais
        for i in range(5):
            y = self.margin_top + (i / 4) * self.chart_height
            self.create_line(self.margin_left, y, 
                           self.margin_left + self.chart_width, y, 
                           fill=self.grid_color, width=1)
            
            # Labels dos valores
            value = max_val - (i / 4) * (max_val - min_val)
            self.create_text(self.margin_left - 5, y, text=f"{value:.2f}", 
                           fill=self.text_color, anchor='e', font=('Segoe UI', 8))
        
        # Linhas verticais
        for i in range(6):
            x = self.margin_left + (i / 5) * self.chart_width
            self.create_line(x, self.margin_top, 
                           x, self.margin_top + self.chart_height, 
                           fill=self.grid_color, width=1)
    
    def _draw_empty_chart(self):
        """Desenha gráfico vazio"""
        self.create_text(self.width // 2, self.height // 2, 
                        text="Aguardando dados...", 
                        fill=self.text_color, font=('Segoe UI', 10))
    
    def clear(self):
        """Limpa todos os dados"""
        self.data_points.clear()
        self.labels.clear()
        self._draw_chart()

class NotificationCenter(tk.Toplevel):
    """Centro de notificações"""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        self.parent = parent
        self.notifications = []
        self.notification_widgets = {}
        
        # Configurar janela
        self.withdraw()  # Inicialmente oculta
        self.overrideredirect(True)
        self.configure(bg='#1e1e1e')
        
        # Posicionar no canto superior direito
        self.geometry("350x500")
        self._position_window()
        
        # Frame principal
        self.main_frame = tk.Frame(self, bg='#1e1e1e')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_frame = tk.Frame(self.main_frame, bg='#2d2d2d', height=40)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="Notificações", 
                font=('Segoe UI', 12, 'bold'), 
                bg='#2d2d2d', fg='#ffffff').pack(side=tk.LEFT, padx=10, pady=10)
        
        # Botão fechar
        close_btn = tk.Button(title_frame, text="×", 
                             font=('Segoe UI', 14, 'bold'),
                             bg='#d13438', fg='#ffffff', 
                             borderwidth=0, width=3,
                             command=self.hide)
        close_btn.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Área de notificações com scroll
        self.canvas = tk.Canvas(self.main_frame, bg='#1e1e1e', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='#1e1e1e')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind para fechar ao clicar fora
        self.bind('<FocusOut>', lambda e: self.hide())
    
    def _position_window(self):
        """Posiciona a janela no canto superior direito"""
        self.update_idletasks()
        
        # Obter dimensões da tela
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        # Posicionar
        x = screen_width - 370  # 350 + 20 de margem
        y = 50
        
        self.geometry(f"350x500+{x}+{y}")
    
    def show(self):
        """Mostra o centro de notificações"""
        self._position_window()
        self.deiconify()
        self.lift()
        self.focus_set()
    
    def hide(self):
        """Oculta o centro de notificações"""
        self.withdraw()
    
    def add_notification(self, notification: Notification):
        """Adiciona uma notificação"""
        self.notifications.insert(0, notification)  # Mais recente no topo
        
        # Manter apenas as últimas 50 notificações
        if len(self.notifications) > 50:
            old_notification = self.notifications.pop()
            if old_notification.id in self.notification_widgets:
                self.notification_widgets[old_notification.id].destroy()
                del self.notification_widgets[old_notification.id]
        
        self._create_notification_widget(notification)
        self._update_scroll_region()
    
    def _create_notification_widget(self, notification: Notification):
        """Cria widget para a notificação"""
        # Cores baseadas no tipo
        colors = {
            NotificationType.INFO: '#0078d4',
            NotificationType.SUCCESS: '#107c10',
            NotificationType.WARNING: '#ff8c00',
            NotificationType.ERROR: '#d13438',
            NotificationType.TRADE: '#6b46c1'
        }
        
        accent_color = colors.get(notification.type, '#0078d4')
        
        # Frame da notificação
        notif_frame = tk.Frame(self.scrollable_frame, bg='#2d2d2d', 
                              relief='solid', borderwidth=1)
        notif_frame.pack(fill=tk.X, pady=2, padx=5)
        
        # Barra lateral colorida
        side_bar = tk.Frame(notif_frame, bg=accent_color, width=4)
        side_bar.pack(side=tk.LEFT, fill=tk.Y)
        
        # Conteúdo
        content_frame = tk.Frame(notif_frame, bg='#2d2d2d')
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=8)
        
        # Cabeçalho
        header_frame = tk.Frame(content_frame, bg='#2d2d2d')
        header_frame.pack(fill=tk.X)
        
        # Título
        title_label = tk.Label(header_frame, text=notification.title, 
                              font=('Segoe UI', 9, 'bold'),
                              bg='#2d2d2d', fg='#ffffff')
        title_label.pack(side=tk.LEFT)
        
        # Timestamp
        time_label = tk.Label(header_frame, 
                             text=notification.timestamp.strftime('%H:%M:%S'),
                             font=('Segoe UI', 8),
                             bg='#2d2d2d', fg='#999999')
        time_label.pack(side=tk.RIGHT)
        
        # Mensagem
        message_label = tk.Label(content_frame, text=notification.message,
                                font=('Segoe UI', 8),
                                bg='#2d2d2d', fg='#cccccc',
                                wraplength=280, justify=tk.LEFT)
        message_label.pack(fill=tk.X, pady=(5, 0))
        
        # Botão de ação (se houver)
        if notification.action_callback and notification.action_text:
            action_btn = tk.Button(content_frame, 
                                  text=notification.action_text,
                                  font=('Segoe UI', 8),
                                  bg=accent_color, fg='#ffffff',
                                  borderwidth=0, pady=2,
                                  command=notification.action_callback)
            action_btn.pack(anchor=tk.W, pady=(5, 0))
        
        # Botão remover
        remove_btn = tk.Button(notif_frame, text="×", 
                              font=('Segoe UI', 10, 'bold'),
                              bg='#3d3d3d', fg='#999999',
                              borderwidth=0, width=2,
                              command=lambda: self._remove_notification(notification.id))
        remove_btn.pack(side=tk.RIGHT, padx=5)
        
        self.notification_widgets[notification.id] = notif_frame
    
    def _remove_notification(self, notification_id: str):
        """Remove uma notificação"""
        # Remover da lista
        self.notifications = [n for n in self.notifications if n.id != notification_id]
        
        # Remover widget
        if notification_id in self.notification_widgets:
            self.notification_widgets[notification_id].destroy()
            del self.notification_widgets[notification_id]
        
        self._update_scroll_region()
    
    def _update_scroll_region(self):
        """Atualiza a região de scroll"""
        self.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

class ToastNotification(tk.Toplevel):
    """Notificação toast (popup temporário)"""
    
    def __init__(self, parent, notification: Notification):
        super().__init__(parent)
        
        self.notification = notification
        
        # Configurar janela
        self.overrideredirect(True)
        self.configure(bg='#2d2d2d')
        self.attributes('-topmost', True)
        
        # Cores baseadas no tipo
        colors = {
            NotificationType.INFO: '#0078d4',
            NotificationType.SUCCESS: '#107c10',
            NotificationType.WARNING: '#ff8c00',
            NotificationType.ERROR: '#d13438',
            NotificationType.TRADE: '#6b46c1'
        }
        
        accent_color = colors.get(notification.type, '#0078d4')
        
        # Frame principal
        main_frame = tk.Frame(self, bg='#2d2d2d', relief='solid', borderwidth=1)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Barra superior colorida
        top_bar = tk.Frame(main_frame, bg=accent_color, height=4)
        top_bar.pack(fill=tk.X)
        
        # Conteúdo
        content_frame = tk.Frame(main_frame, bg='#2d2d2d')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Título
        title_label = tk.Label(content_frame, text=notification.title,
                              font=('Segoe UI', 10, 'bold'),
                              bg='#2d2d2d', fg='#ffffff')
        title_label.pack(anchor=tk.W)
        
        # Mensagem
        message_label = tk.Label(content_frame, text=notification.message,
                                font=('Segoe UI', 9),
                                bg='#2d2d2d', fg='#cccccc',
                                wraplength=250, justify=tk.LEFT)
        message_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Botão de ação (se houver)
        if notification.action_callback and notification.action_text:
            action_btn = tk.Button(content_frame, 
                                  text=notification.action_text,
                                  font=('Segoe UI', 8),
                                  bg=accent_color, fg='#ffffff',
                                  borderwidth=0, pady=3, padx=10,
                                  command=self._handle_action)
            action_btn.pack(anchor=tk.W, pady=(8, 0))
        
        # Posicionar e mostrar
        self._position_toast()
        self._schedule_auto_close()
    
    def _position_toast(self):
        """Posiciona o toast no canto inferior direito"""
        self.update_idletasks()
        
        # Obter dimensões
        width = self.winfo_reqwidth()
        height = self.winfo_reqheight()
        
        # Posicionar
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        x = screen_width - width - 20
        y = screen_height - height - 100
        
        self.geometry(f"{width}x{height}+{x}+{y}")
        
        # Animação de entrada
        self._animate_in()
    
    def _animate_in(self):
        """Animação de entrada"""
        self.attributes('-alpha', 0.0)
        
        def fade_in(alpha=0.0):
            if alpha < 1.0:
                alpha += 0.1
                self.attributes('-alpha', alpha)
                self.after(20, lambda: fade_in(alpha))
        
        fade_in()
    
    def _animate_out(self, callback=None):
        """Animação de saída"""
        def fade_out(alpha=1.0):
            if alpha > 0.0:
                alpha -= 0.1
                self.attributes('-alpha', alpha)
                self.after(20, lambda: fade_out(alpha))
            else:
                if callback:
                    callback()
                self.destroy()
        
        fade_out()
    
    def _schedule_auto_close(self):
        """Agenda fechamento automático"""
        self.after(self.notification.duration, self._auto_close)
    
    def _auto_close(self):
        """Fecha automaticamente"""
        self._animate_out()
    
    def _handle_action(self):
        """Manipula ação do botão"""
        if self.notification.action_callback:
            self.notification.action_callback()
        self._animate_out()

class StatusIndicator(tk.Canvas):
    """Indicador de status com LED"""
    
    def __init__(self, parent, size=16, **kwargs):
        super().__init__(parent, width=size, height=size, 
                        highlightthickness=0, bg='#2d2d2d', **kwargs)
        
        self.size = size
        self.status = 'offline'
        
        # Estados e cores
        self.colors = {
            'online': '#107c10',
            'offline': '#d13438',
            'warning': '#ff8c00',
            'connecting': '#0078d4'
        }
        
        self._draw_indicator()
    
    def _draw_indicator(self):
        """Desenha o indicador"""
        self.delete('all')
        
        color = self.colors.get(self.status, '#999999')
        center = self.size // 2
        radius = (self.size - 4) // 2
        
        # LED principal
        self.create_oval(center - radius, center - radius,
                        center + radius, center + radius,
                        fill=color, outline='', width=0)
        
        # Brilho
        highlight_radius = radius // 2
        self.create_oval(center - highlight_radius, center - highlight_radius,
                        center + highlight_radius, center + highlight_radius,
                        fill='white', outline='', stipple='gray25')
    
    def set_status(self, status: str):
        """Define o status"""
        self.status = status
        self._draw_indicator()
    
    def animate_pulse(self):
        """Anima pulsação para status de conexão"""
        if self.status == 'connecting':
            # Implementar animação de pulsação
            pass

class MetricCard(tk.Frame):
    """Card de métrica com animação"""
    
    def __init__(self, parent, title: str, value: str = "0", 
                 unit: str = "", trend: Optional[float] = None, **kwargs):
        super().__init__(parent, bg='#2d2d2d', relief='solid', 
                        borderwidth=1, **kwargs)
        
        self.title = title
        self.current_value = value
        self.unit = unit
        self.trend = trend
        
        # Padding interno
        self.configure(padx=15, pady=12)
        
        # Título
        self.title_label = tk.Label(self, text=title, 
                                   font=('Segoe UI', 9),
                                   bg='#2d2d2d', fg='#999999')
        self.title_label.pack(anchor=tk.W)
        
        # Frame para valor e trend
        value_frame = tk.Frame(self, bg='#2d2d2d')
        value_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Valor
        self.value_label = tk.Label(value_frame, text=f"{value}{unit}",
                                   font=('Segoe UI', 14, 'bold'),
                                   bg='#2d2d2d', fg='#ffffff')
        self.value_label.pack(side=tk.LEFT)
        
        # Indicador de tendência
        if trend is not None:
            self.trend_label = tk.Label(value_frame, text=self._format_trend(trend),
                                       font=('Segoe UI', 10),
                                       bg='#2d2d2d', 
                                       fg='#107c10' if trend >= 0 else '#d13438')
            self.trend_label.pack(side=tk.RIGHT)
    
    def _format_trend(self, trend: float) -> str:
        """Formata indicador de tendência"""
        symbol = "▲" if trend >= 0 else "▼"
        return f"{symbol} {abs(trend):.1f}%"
    
    def update_value(self, value: str, trend: Optional[float] = None):
        """Atualiza o valor com animação"""
        self.current_value = value
        self.value_label.config(text=f"{value}{self.unit}")
        
        if trend is not None:
            self.trend = trend
            if hasattr(self, 'trend_label'):
                self.trend_label.config(
                    text=self._format_trend(trend),
                    fg='#107c10' if trend >= 0 else '#d13438'
                )
        
        # Animação de destaque
        self._animate_update()
    
    def _animate_update(self):
        """Anima atualização do valor"""
        original_bg = self.cget('bg')
        highlight_bg = '#3d3d3d'
        
        def animate(step=0):
            if step < 10:
                # Fade para highlight
                self.configure(bg=highlight_bg)
                self.after(50, lambda: animate(step + 1))
            elif step < 20:
                # Fade de volta
                self.configure(bg=original_bg)
                self.after(50, lambda: animate(step + 1))
        
        animate()

# Exemplo de uso e testes
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Teste de Widgets Avançados")
    root.geometry("800x600")
    root.configure(bg='#1e1e1e')
    
    # Teste de diferentes widgets
    main_frame = tk.Frame(root, bg='#1e1e1e')
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    # Progress bars
    progress_frame = tk.Frame(main_frame, bg='#1e1e1e')
    progress_frame.pack(fill=tk.X, pady=10)
    
    tk.Label(progress_frame, text="Barras de Progresso:", 
            font=('Segoe UI', 12, 'bold'), bg='#1e1e1e', fg='white').pack(anchor=tk.W)
    
    animated_bar = AnimatedProgressBar(progress_frame)
    animated_bar.pack(pady=5)
    animated_bar.animate_to(0.75)
    
    circular = CircularProgress(progress_frame, size=80)
    circular.pack(pady=5)
    circular.set_progress(0.6)
    
    # Gráfico
    chart_frame = tk.Frame(main_frame, bg='#1e1e1e')
    chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    tk.Label(chart_frame, text="Gráfico em Tempo Real:", 
            font=('Segoe UI', 12, 'bold'), bg='#1e1e1e', fg='white').pack(anchor=tk.W)
    
    chart = LiveChart(chart_frame, width=600, height=200)
    chart.pack(pady=5)
    
    # Adicionar dados de exemplo
    import random
    for i in range(20):
        chart.add_point(random.uniform(10, 100))
    
    # Cards de métrica
    metrics_frame = tk.Frame(main_frame, bg='#1e1e1e')
    metrics_frame.pack(fill=tk.X, pady=10)
    
    tk.Label(metrics_frame, text="Cards de Métrica:", 
            font=('Segoe UI', 12, 'bold'), bg='#1e1e1e', fg='white').pack(anchor=tk.W)
    
    cards_container = tk.Frame(metrics_frame, bg='#1e1e1e')
    cards_container.pack(fill=tk.X, pady=5)
    
    card1 = MetricCard(cards_container, "Saldo", "$1,234.56", trend=2.5)
    card1.pack(side=tk.LEFT, padx=5)
    
    card2 = MetricCard(cards_container, "Lucro", "$89.12", trend=-1.2)
    card2.pack(side=tk.LEFT, padx=5)
    
    # Status indicators
    status_frame = tk.Frame(main_frame, bg='#1e1e1e')
    status_frame.pack(fill=tk.X, pady=10)
    
    tk.Label(status_frame, text="Indicadores de Status:", 
            font=('Segoe UI', 12, 'bold'), bg='#1e1e1e', fg='white').pack(anchor=tk.W)
    
    indicators_container = tk.Frame(status_frame, bg='#1e1e1e')
    indicators_container.pack(fill=tk.X, pady=5)
    
    status1 = StatusIndicator(indicators_container)
    status1.pack(side=tk.LEFT, padx=5)
    status1.set_status('online')
    
    tk.Label(indicators_container, text="Online", bg='#1e1e1e', fg='white').pack(side=tk.LEFT, padx=(0, 20))
    
    status2 = StatusIndicator(indicators_container)
    status2.pack(side=tk.LEFT, padx=5)
    status2.set_status('warning')
    
    tk.Label(indicators_container, text="Warning", bg='#1e1e1e', fg='white').pack(side=tk.LEFT)
    
    root.mainloop()