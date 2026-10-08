import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import os
from trading_decision_integrator import initialize_trading_integrator, get_trading_integrator
from detailed_decision_logger import DecisionType, LogLevel

# Importar MetaTrader5 se disponível
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    print("MetaTrader5 não disponível - modo simulação")

class DecisionLogger:
    """Sistema de logging de decisões simplificado"""
    
    def __init__(self):
        self.decisions = []
        self.log_file = "decision_logs.json"
        self._load_existing_logs()
    
    def _load_existing_logs(self):
        """Carrega logs existentes"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    self.decisions = json.load(f)
        except Exception as e:
            print(f"Erro ao carregar logs: {e}")
            self.decisions = []
    
    def log_decision(self, decision_type: str, description: str, 
                    justification: str, outcome: str, confidence: float = 0.5):
        """Registra uma decisão"""
        decision = {
            'timestamp': datetime.now().isoformat(),
            'type': decision_type,
            'description': description,
            'justification': justification,
            'outcome': outcome,
            'confidence': confidence
        }
        
        self.decisions.append(decision)
        
        # Manter apenas os últimos 100 registros
        if len(self.decisions) > 100:
            self.decisions = self.decisions[-100:]
        
        self._save_logs()
        print(f"[LOG] {decision_type}: {description} - {outcome}")
    
    def _save_logs(self):
        """Salva logs no arquivo"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.decisions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar logs: {e}")
    
    def get_recent_decisions(self, count: int = 10) -> List[Dict]:
        """Retorna decisões recentes"""
        return self.decisions[-count:] if self.decisions else []
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas das decisões"""
        if not self.decisions:
            return {'total': 0, 'avg_confidence': 0}
        
        total = len(self.decisions)
        avg_confidence = sum(d.get('confidence', 0) for d in self.decisions) / total
        
        return {
            'total': total,
            'avg_confidence': avg_confidence
        }

class MT5Authenticator:
    """Autenticador MT5 simplificado"""
    
    def __init__(self):
        self.login = int(os.getenv("MT5_LOGIN", "0"))  # Credenciais fornecidas
        self.password = os.getenv("MT5_PASSWORD", "")
        self.server = "ActivTradesMarkets-Server"  # Servidor correto
        self.connected = False
        self.account_info = None
        
        # Inicializar integrador de trading
        self.trading_integrator = None
        
        self._check_existing_session()
    
    def _check_existing_session(self):
        """Verifica se já existe uma sessão ativa do MT5"""
        if not MT5_AVAILABLE:
            return
        
        try:
            # Verificar se MT5 já está inicializado
            account_info = mt5.account_info()
            if account_info is not None:
                # Verificar se é a mesma conta
                if account_info.login == self.login and account_info.server == self.server:
                    self.connected = True
                    self.account_info = account_info._asdict()
                    print(f"Sessão existente reutilizada - Login: {self.login}, Servidor: {self.server}")
                    return True
                else:
                    # Desconectar da conta diferente
                    mt5.shutdown()
        except Exception as e:
            print(f"Erro ao verificar sessão existente: {e}")
        
        return False
    
    def connect(self) -> Dict[str, Any]:
        """Conecta ao MT5"""
        # Verificar se já existe sessão ativa
        if self.connected:
            return {
                'success': True,
                'message': f'Já conectado ao servidor {self.server} (sessão reutilizada)',
                'simulation': False,
                'reused_session': True
            }
        
        if not MT5_AVAILABLE:
            # Modo simulação
            self.connected = True
            self.account_info = {
                'login': self.login,
                'server': self.server,
                'balance': 10000.0,
                'equity': 10000.0,
                'margin': 0.0,
                'margin_level': 0.0,
                'trade_allowed': True
            }
            
            # Inicializar integrador de trading em modo simulação
            session_id = f"simulation_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.trading_integrator = initialize_trading_integrator(self, session_id)
            if self.trading_integrator:
                self.trading_integrator.start_monitoring()
                print("Sistema de logging detalhado iniciado (modo simulação)")
            
            return {
                'success': True,
                'message': 'Conectado em modo simulação',
                'simulation': True
            }
        
        try:
            # Inicializar MT5
            if not mt5.initialize():
                return {
                    'success': False,
                    'message': f'Falha na inicialização MT5: {mt5.last_error()}'
                }
            
            # Tentar login
            if not mt5.login(self.login, password=self.password, server=self.server):
                error = mt5.last_error()
                mt5.shutdown()
                return {
                    'success': False,
                    'message': f'Falha no login: {error}'
                }
            
            # Obter informações da conta
            account_info = mt5.account_info()
            if account_info is None:
                mt5.shutdown()
                return {
                    'success': False,
                    'message': 'Falha ao obter informações da conta'
                }
            
            self.connected = True
            self.account_info = account_info._asdict()
            
            # Inicializar integrador de trading
            session_id = f"mt5_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.trading_integrator = initialize_trading_integrator(self, session_id)
            if self.trading_integrator:
                self.trading_integrator.start_monitoring()
                print("Sistema de logging detalhado iniciado")
            
            return {
                'success': True,
                'message': f'Conectado ao servidor {self.server}',
                'simulation': False,
                'reused_session': False
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Erro na conexão: {str(e)}'
            }
    
    def disconnect(self) -> bool:
        """Desconecta do MT5"""
        try:
            # Parar monitoramento do integrador
            if self.trading_integrator:
                self.trading_integrator.stop_monitoring()
                print("Sistema de logging detalhado parado")
            
            if MT5_AVAILABLE and self.connected:
                mt5.shutdown()
            
            self.connected = False
            self.account_info = None
            return True
            
        except Exception as e:
            print(f"Erro na desconexão: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Verifica se está conectado"""
        return self.connected
    
    def get_account_info(self) -> Optional[Dict]:
        """Retorna informações da conta"""
        if not self.connected:
            return None
        
        if not MT5_AVAILABLE:
            # Simular variação nos dados
            import random
            if self.account_info:
                self.account_info['balance'] += random.uniform(-100, 100)
                self.account_info['equity'] = self.account_info['balance'] + random.uniform(-50, 50)
        
        return self.account_info
    
    def log_market_analysis(self, symbols: List[str], analysis_results: Dict[str, Any],
                           observations: List[str], recommendations: List[str],
                           confidence: float = 0.8) -> str:
        """Registra análise de mercado usando o sistema de logging detalhado"""
        if not self.trading_integrator:
            print("Integrador de trading não disponível")
            return ""
        
        return self.trading_integrator.log_market_analysis_session(
            symbols=symbols,
            analysis_results=analysis_results,
            observations=observations,
            recommendations=recommendations,
            confidence=confidence
        )
    
    def log_trade_decision(self, symbol: str, action: str, volume: float,
                          analysis_factors: List[str], justification: str,
                          confidence: float = 0.8) -> str:
        """Registra decisão de trading usando o sistema de logging detalhado"""
        if not self.trading_integrator:
            print("Integrador de trading não disponível")
            return ""
        
        return self.trading_integrator.log_manual_trade_decision(
            symbol=symbol,
            action=action,
            volume=volume,
            analysis_factors=analysis_factors,
            justification=justification,
            confidence=confidence
        )
    
    def log_no_trade_analysis(self, symbol: str, reasons: List[str],
                             analysis_performed: List[str], confidence: float = 0.8) -> str:
        """Registra análise que resultou em não operação"""
        if not self.trading_integrator:
            print("Integrador de trading não disponível")
            return ""
        
        return self.trading_integrator.log_no_trade_analysis(
            symbol=symbol,
            reasons=reasons,
            analysis_performed=analysis_performed,
            confidence=confidence
        )
    
    def get_detailed_session_summary(self) -> Dict[str, Any]:
        """Retorna resumo detalhado da sessão de trading"""
        if not self.trading_integrator:
            return {}
        
        return self.trading_integrator.get_session_summary()
    
    def export_detailed_report(self) -> str:
        """Exporta relatório detalhado da sessão"""
        if not self.trading_integrator:
            return ""
        
        return self.trading_integrator.export_session_report()

class ModernTheme:
    """Tema moderno simplificado"""
    
    PRIMARY = '#0078d4'
    SUCCESS = '#107c10'
    WARNING = '#ff8c00'
    DANGER = '#d13438'
    INFO = '#17a2b8'
    
    BG_PRIMARY = '#1e1e1e'
    BG_SECONDARY = '#2d2d2d'
    BG_TERTIARY = '#3d3d3d'
    
    TEXT_PRIMARY = '#ffffff'
    TEXT_SECONDARY = '#cccccc'
    TEXT_MUTED = '#999999'

class MetricCard(tk.Frame):
    """Card de métrica simples"""
    
    def __init__(self, parent, title: str, value: str = "0", **kwargs):
        super().__init__(parent, bg=ModernTheme.BG_TERTIARY, 
                        relief='solid', borderwidth=1, **kwargs)
        
        self.title_label = tk.Label(self, text=title,
                                   font=('Segoe UI', 8),
                                   bg=ModernTheme.BG_TERTIARY,
                                   fg=ModernTheme.TEXT_MUTED)
        self.title_label.pack(pady=(5, 0))
        
        self.value_label = tk.Label(self, text=value,
                                   font=('Segoe UI', 12, 'bold'),
                                   bg=ModernTheme.BG_TERTIARY,
                                   fg=ModernTheme.TEXT_PRIMARY)
        self.value_label.pack(pady=(0, 5))
    
    def update_value(self, value: str):
        """Atualiza o valor do card"""
        self.value_label.config(text=value)

class StatusIndicator(tk.Canvas):
    """Indicador de status simples"""
    
    def __init__(self, parent, size=16, **kwargs):
        super().__init__(parent, width=size, height=size, 
                        bg=ModernTheme.BG_SECONDARY, highlightthickness=0, **kwargs)
        
        self.size = size
        self.status = 'offline'
        self._draw_indicator()
    
    def _draw_indicator(self):
        """Desenha o indicador"""
        self.delete('all')
        
        color = ModernTheme.SUCCESS if self.status == 'online' else ModernTheme.DANGER
        
        self.create_oval(2, 2, self.size-2, self.size-2, 
                        fill=color, outline=color)
    
    def set_status(self, status: str):
        """Define o status"""
        self.status = status
        self._draw_indicator()

class SecureTradingPanel(tk.Frame):
    """Painel de trading seguro simplificado"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=ModernTheme.BG_SECONDARY, **kwargs)
        
        # Inicializar componentes
        self.logger = DecisionLogger()
        self.authenticator = MT5Authenticator()
        
        # Estado
        self.auto_trading = False
        self.monitoring_active = False
        
        self._create_widgets()
        self._start_monitoring()
        
        # Log inicial
        self.logger.log_decision(
            "SYSTEM", "Inicialização do painel",
            "Painel de trading inicializado com recursos de segurança",
            "Sistema pronto para operação", 0.9
        )
    
    def _create_widgets(self):
        """Cria widgets do painel"""
        # Título
        title_frame = tk.Frame(self, bg=ModernTheme.BG_SECONDARY)
        title_frame.pack(fill=tk.X, padx=15, pady=15)
        
        tk.Label(title_frame, text="MT5 Dashboard - Seguro & Monitorado",
                font=('Segoe UI', 14, 'bold'),
                bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_PRIMARY).pack(side=tk.LEFT)
        
        # Indicador de logging
        self.logging_indicator = StatusIndicator(title_frame)
        self.logging_indicator.pack(side=tk.RIGHT, padx=10)
        self.logging_indicator.set_status('online')
        
        tk.Label(title_frame, text="Logging Ativo",
                font=('Segoe UI', 9),
                bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.SUCCESS).pack(side=tk.RIGHT, padx=5)
        
        # Status de conexão
        status_frame = tk.Frame(self, bg=ModernTheme.BG_SECONDARY)
        status_frame.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(status_frame, text="Status MT5:",
                font=('Segoe UI', 10),
                bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_SECONDARY).pack(side=tk.LEFT)
        
        self.status_indicator = StatusIndicator(status_frame, size=20)
        self.status_indicator.pack(side=tk.LEFT, padx=10)
        
        self.status_label = tk.Label(status_frame, text="Desconectado",
                                    font=('Segoe UI', 10, 'bold'),
                                    bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.DANGER)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Credenciais (mascaradas)
        cred_text = f"Login: {self.authenticator.login} | Servidor: {self.authenticator.server}"
        self.cred_label = tk.Label(status_frame, text=cred_text,
                                  font=('Segoe UI', 9),
                                  bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_MUTED)
        self.cred_label.pack(side=tk.LEFT, padx=20)
        
        # Botões de controle
        buttons_frame = tk.Frame(self, bg=ModernTheme.BG_SECONDARY)
        buttons_frame.pack(fill=tk.X, padx=15, pady=15)
        
        self.connect_btn = tk.Button(buttons_frame, text="Conectar MT5",
                                    font=('Segoe UI', 10),
                                    bg=ModernTheme.SUCCESS, fg=ModernTheme.TEXT_PRIMARY,
                                    borderwidth=0, pady=10, padx=25,
                                    command=self._toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        self.auto_trade_btn = tk.Button(buttons_frame, text="Auto Trading: OFF",
                                       font=('Segoe UI', 10),
                                       bg=ModernTheme.BG_TERTIARY, fg=ModernTheme.TEXT_SECONDARY,
                                       borderwidth=0, pady=10, padx=25,
                                       command=self._toggle_auto_trading)
        self.auto_trade_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        self.logs_btn = tk.Button(buttons_frame, text="Ver Logs de Decisão",
                                 font=('Segoe UI', 10),
                                 bg=ModernTheme.INFO, fg=ModernTheme.TEXT_PRIMARY,
                                 borderwidth=0, pady=10, padx=25,
                                 command=self._show_logs)
        self.logs_btn.pack(side=tk.LEFT)
        
        # Informações da conta
        account_frame = tk.LabelFrame(self, text="Informações da Conta",
                                     font=('Segoe UI', 11, 'bold'),
                                     bg=ModernTheme.BG_SECONDARY, fg=ModernTheme.TEXT_PRIMARY)
        account_frame.pack(fill=tk.X, padx=15, pady=15)
        
        metrics_frame = tk.Frame(account_frame, bg=ModernTheme.BG_SECONDARY)
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.balance_card = MetricCard(metrics_frame, "Saldo", "$0.00")
        self.balance_card.pack(side=tk.LEFT, padx=(0, 15), fill=tk.Y)
        
        self.equity_card = MetricCard(metrics_frame, "Patrimônio", "$0.00")
        self.equity_card.pack(side=tk.LEFT, padx=(0, 15), fill=tk.Y)
        
        self.margin_card = MetricCard(metrics_frame, "Margem", "$0.00")
        self.margin_card.pack(side=tk.LEFT, padx=(0, 15), fill=tk.Y)
        
        # Métricas de decisão
        decision_frame = tk.Frame(account_frame, bg=ModernTheme.BG_SECONDARY)
        decision_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.decisions_card = MetricCard(decision_frame, "Total Decisões", "0")
        self.decisions_card.pack(side=tk.LEFT, padx=(0, 15), fill=tk.Y)
        
        self.confidence_card = MetricCard(decision_frame, "Confiança Média", "0%")
        self.confidence_card.pack(side=tk.LEFT, fill=tk.Y)
    
    def _toggle_connection(self):
        """Alterna conexão MT5 com logging"""
        if not self.authenticator.is_connected():
            self.logger.log_decision(
                "CONNECTION", "Tentativa de conexão MT5",
                f"Usuário solicitou conexão com login {self.authenticator.login} no servidor {self.authenticator.server}",
                "Iniciando processo de conexão", 0.8
            )
            
            result = self.authenticator.connect()
            
            if result['success']:
                session_type = "Simulação" if result.get('simulation') else "Real"
                reused = result.get('reused_session', False)
                
                self.logger.log_decision(
                    "CONNECTION", "Conexão MT5 estabelecida",
                    f"Conectado com sucesso. Modo: {session_type}. Sessão reutilizada: {reused}. Servidor: {self.authenticator.server}",
                    result['message'], 0.9
                )
                
                if reused:
                    messagebox.showinfo("Conexão", f"Sessão existente reutilizada!\n{result['message']}")
                else:
                    messagebox.showinfo("Conexão", result['message'])
                
                self._update_account_info()
            else:
                self.logger.log_decision(
                    "CONNECTION", "Falha na conexão MT5",
                    f"Credenciais ou servidor podem estar incorretos. Servidor: {self.authenticator.server}",
                    result['message'], 0.1
                )
                
                messagebox.showerror("Erro de Conexão", result['message'])
        else:
            self.logger.log_decision(
                "CONNECTION", "Desconexão MT5",
                f"Usuário solicitou desconexão do servidor {self.authenticator.server}",
                "Desconectando com segurança", 0.9
            )
            
            self.authenticator.disconnect()
            self.auto_trading = False
            
            messagebox.showinfo("Desconexão", "Desconectado com segurança")
        
        self._update_status_display()
    
    def _toggle_auto_trading(self):
        """Alterna auto trading com verificações de segurança"""
        if not self.authenticator.is_connected():
            self.logger.log_decision(
                "AUTO_TRADING", "Tentativa de ativar auto trading",
                "Auto trading requer conexão ativa com MT5",
                "Operação negada - não conectado", 0.0
            )
            
            messagebox.showwarning("Auto Trading", "Conecte ao MT5 primeiro")
            return
        
        account_info = self.authenticator.get_account_info()
        
        if not self.auto_trading:
            # Verificações de segurança
            if account_info and account_info.get('balance', 0) < 1000:
                self.logger.log_decision(
                    "RISK_CHECK", "Auto trading negado",
                    f"Saldo insuficiente: ${account_info['balance']}",
                    "Operação negada por segurança", 0.2
                )
                
                messagebox.showerror("Risco", "Saldo insuficiente para auto trading")
                return
            
            self.auto_trading = True
            
            self.logger.log_decision(
                "AUTO_TRADING", "Auto trading ativado",
                f"Verificações de segurança aprovadas. Saldo: ${account_info.get('balance', 0) if account_info else 'N/A'}",
                "Auto trading ativo com monitoramento", 0.8
            )
            
            messagebox.showinfo("Auto Trading", "Ativado com verificações de segurança")
        else:
            self.auto_trading = False
            
            self.logger.log_decision(
                "AUTO_TRADING", "Auto trading desativado",
                "Usuário desativou auto trading",
                "Auto trading desativado", 0.9
            )
            
            messagebox.showinfo("Auto Trading", "Desativado")
        
        self._update_status_display()
    
    def _update_account_info(self):
        """Atualiza informações da conta"""
        account_info = self.authenticator.get_account_info()
        
        if account_info:
            self.balance_card.update_value(f"${account_info['balance']:,.2f}")
            self.equity_card.update_value(f"${account_info['equity']:,.2f}")
            self.margin_card.update_value(f"${account_info['margin']:,.2f}")
            
            self.logger.log_decision(
                "DATA_UPDATE", "Informações da conta atualizadas",
                "Dados obtidos com sucesso do MT5",
                f"Saldo: ${account_info['balance']:,.2f}", 0.7
            )
    
    def _update_decision_metrics(self):
        """Atualiza métricas de decisão"""
        stats = self.logger.get_statistics()
        
        self.decisions_card.update_value(str(stats['total']))
        self.confidence_card.update_value(f"{stats['avg_confidence']*100:.1f}%")
    
    def _update_status_display(self):
        """Atualiza display de status"""
        if self.authenticator.is_connected():
            self.status_indicator.set_status('online')
            self.status_label.config(text="Conectado", fg=ModernTheme.SUCCESS)
            self.connect_btn.config(text="Desconectar", bg=ModernTheme.DANGER)
        else:
            self.status_indicator.set_status('offline')
            self.status_label.config(text="Desconectado", fg=ModernTheme.DANGER)
            self.connect_btn.config(text="Conectar MT5", bg=ModernTheme.SUCCESS)
        
        if self.auto_trading:
            self.auto_trade_btn.config(text="Auto Trading: ON", 
                                      bg=ModernTheme.SUCCESS, 
                                      fg=ModernTheme.TEXT_PRIMARY)
        else:
            self.auto_trade_btn.config(text="Auto Trading: OFF", 
                                      bg=ModernTheme.BG_TERTIARY, 
                                      fg=ModernTheme.TEXT_SECONDARY)
    
    def _show_logs(self):
        """Mostra janela de logs"""
        log_window = tk.Toplevel(self)
        log_window.title("Logs de Decisão Detalhados")
        log_window.geometry("900x700")
        log_window.configure(bg=ModernTheme.BG_PRIMARY)
        
        # Frame principal
        main_frame = tk.Frame(log_window, bg=ModernTheme.BG_PRIMARY)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = tk.Label(main_frame, text="Registro de Decisões do Sistema",
                              font=('Segoe UI', 14, 'bold'),
                              bg=ModernTheme.BG_PRIMARY, fg=ModernTheme.TEXT_PRIMARY)
        title_label.pack(pady=(0, 10))
        
        # Área de texto com scroll
        text_frame = tk.Frame(main_frame, bg=ModernTheme.BG_PRIMARY)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_area = tk.Text(text_frame, 
                           bg=ModernTheme.BG_SECONDARY,
                           fg=ModernTheme.TEXT_PRIMARY,
                           font=('Consolas', 10),
                           wrap=tk.WORD,
                           insertbackground=ModernTheme.TEXT_PRIMARY)
        
        scrollbar = tk.Scrollbar(text_frame, command=text_area.yview)
        text_area.config(yscrollcommand=scrollbar.set)
        
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Carregar logs
        recent_decisions = self.logger.get_recent_decisions(30)
        
        if recent_decisions:
            text_area.insert(tk.END, "=== LOGS DE DECISÃO DO SISTEMA ===\n\n")
            
            for i, decision in enumerate(recent_decisions, 1):
                timestamp = decision['timestamp'][:19].replace('T', ' ')
                
                log_text = (
                    f"{i:2d}. [{timestamp}] {decision['type']}\n"
                    f"    Descrição: {decision['description']}\n"
                    f"    Justificativa: {decision['justification']}\n"
                    f"    Resultado: {decision['outcome']}\n"
                    f"    Confiança: {decision['confidence']:.2f}\n"
                    f"    {'-'*70}\n\n"
                )
                text_area.insert(tk.END, log_text)
        else:
            text_area.insert(tk.END, "Nenhum log de decisão disponível ainda.\n\n")
            text_area.insert(tk.END, "Os logs serão exibidos aqui conforme você usar o sistema.")
        
        text_area.config(state=tk.DISABLED)
        
        # Botão para atualizar
        refresh_btn = tk.Button(main_frame, text="Atualizar Logs",
                               font=('Segoe UI', 10),
                               bg=ModernTheme.INFO, fg=ModernTheme.TEXT_PRIMARY,
                               borderwidth=0, pady=8, padx=20,
                               command=lambda: self._refresh_logs(text_area))
        refresh_btn.pack(pady=10)
    
    def _refresh_logs(self, text_area):
        """Atualiza logs na janela"""
        text_area.config(state=tk.NORMAL)
        text_area.delete(1.0, tk.END)
        
        recent_decisions = self.logger.get_recent_decisions(30)
        
        if recent_decisions:
            text_area.insert(tk.END, "=== LOGS DE DECISÃO DO SISTEMA (ATUALIZADOS) ===\n\n")
            
            for i, decision in enumerate(recent_decisions, 1):
                timestamp = decision['timestamp'][:19].replace('T', ' ')
                
                log_text = (
                    f"{i:2d}. [{timestamp}] {decision['type']}\n"
                    f"    Descrição: {decision['description']}\n"
                    f"    Justificativa: {decision['justification']}\n"
                    f"    Resultado: {decision['outcome']}\n"
                    f"    Confiança: {decision['confidence']:.2f}\n"
                    f"    {'-'*70}\n\n"
                )
                text_area.insert(tk.END, log_text)
        
        text_area.config(state=tk.DISABLED)
    
    def _start_monitoring(self):
        """Inicia monitoramento em background"""
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Atualizar métricas
                    self.after(0, self._update_decision_metrics)
                    
                    # Atualizar informações da conta se conectado
                    if self.authenticator.is_connected():
                        self.after(0, self._update_account_info)
                    
                    time.sleep(10)  # Atualizar a cada 10 segundos
                    
                except Exception as e:
                    self.logger.log_decision(
                        "MONITOR_ERROR", "Erro no monitoramento",
                        f"Exceção no loop de monitoramento: {str(e)}",
                        "Continuando monitoramento", 0.3
                    )
                    time.sleep(15)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def destroy(self):
        """Cleanup ao destruir"""
        self.monitoring_active = False
        
        self.logger.log_decision(
            "SYSTEM", "Finalização do painel",
            "Sistema sendo finalizado pelo usuário",
            "Cleanup realizado com sucesso", 0.9
        )
        
        if self.authenticator.is_connected():
            self.authenticator.disconnect()
        
        super().destroy()

class MT5DashboardApp(tk.Tk):
    """Aplicação principal do dashboard"""
    
    def __init__(self):
        super().__init__()
        
        self.title("MT5 Dashboard Seguro - Logging Detalhado")
        self.geometry("1200x800")
        self.configure(bg=ModernTheme.BG_PRIMARY)
        
        # Centralizar janela
        self.center_window()
        
        # Criar interface
        self.create_interface()
        
        # Protocolo de fechamento
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        print("Dashboard MT5 iniciado com sucesso!")
        print("Recursos implementados:")
        print("- Autenticação segura com credenciais fornecidas")
        print("- Sistema de logging detalhado de decisões")
        print("- Monitoramento contínuo de status")
        print("- Interface moderna e responsiva")
    
    def center_window(self):
        """Centraliza a janela na tela"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_interface(self):
        """Cria a interface principal"""
        # Frame principal
        main_frame = tk.Frame(self, bg=ModernTheme.BG_PRIMARY)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Painel de trading
        self.trading_panel = SecureTradingPanel(main_frame)
        self.trading_panel.pack(fill=tk.BOTH, expand=True)
    
    def on_closing(self):
        """Callback para fechamento"""
        if hasattr(self, 'trading_panel'):
            self.trading_panel.destroy()
        
        print("Dashboard finalizado com segurança")
        self.destroy()

# Flask Web Interface
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
import secrets

# Configurar Flask app
flask_app = Flask(__name__)
flask_app.secret_key = secrets.token_hex(16)
flask_app.config['SESSION_TYPE'] = 'filesystem'
Session(flask_app)

# Instâncias globais para Flask
mt5_auth = MT5Authenticator()
decision_logger = DecisionLogger()

@flask_app.route('/')
def index():
    """Página principal"""
    if not session.get('authenticated'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@flask_app.route('/login', methods=['GET', 'POST'])
def login():
    """Página de login"""
    if request.method == 'POST':
        # Autenticação simples (em produção, usar hash seguro)
        if request.form.get('password') == 'mt5secure2024':
            session['authenticated'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Senha incorreta')
    return render_template('login.html')

@flask_app.route('/logout')
def logout():
    """Logout"""
    session.clear()
    return redirect(url_for('login'))

@flask_app.route('/api/connect', methods=['POST'])
def connect_mt5():
    """API para conectar ao MT5"""
    if not session.get('authenticated'):
        return jsonify({'error': 'Não autenticado'}), 401
    
    try:
        result = mt5_auth.connect()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@flask_app.route('/api/disconnect', methods=['POST'])
def disconnect_mt5():
    """API para desconectar do MT5"""
    if not session.get('authenticated'):
        return jsonify({'error': 'Não autenticado'}), 401
    
    try:
        success = mt5_auth.disconnect()
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@flask_app.route('/api/account_info')
def get_account_info():
    """API para obter informações da conta"""
    if not session.get('authenticated'):
        return jsonify({'error': 'Não autenticado'}), 401
    
    try:
        account_info = mt5_auth.get_account_info()
        return jsonify({
            'success': True,
            'account_info': account_info,
            'connected': mt5_auth.is_connected()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@flask_app.route('/api/decisions')
def get_decisions():
    """API para obter decisões recentes"""
    if not session.get('authenticated'):
        return jsonify({'error': 'Não autenticado'}), 401
    
    try:
        decisions = decision_logger.get_recent_decisions(50)
        return jsonify({
            'success': True,
            'decisions': decisions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@flask_app.route('/api/detailed_session_summary')
def get_detailed_session_summary():
    """API para obter resumo detalhado da sessão"""
    if not session.get('authenticated'):
        return jsonify({'error': 'Não autenticado'}), 401
    
    try:
        summary = mt5_auth.get_detailed_session_summary()
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@flask_app.route('/api/log_market_analysis', methods=['POST'])
def log_market_analysis_api():
    """API para registrar análise de mercado"""
    if not session.get('authenticated'):
        return jsonify({'error': 'Não autenticado'}), 401
    
    try:
        data = request.get_json()
        decision_id = mt5_auth.log_market_analysis(
            symbols=data.get('symbols', []),
            analysis_results=data.get('analysis_results', {}),
            observations=data.get('observations', []),
            recommendations=data.get('recommendations', []),
            confidence=data.get('confidence', 0.8)
        )
        
        return jsonify({
            'success': True,
            'decision_id': decision_id
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@flask_app.route('/api/log_trade_decision', methods=['POST'])
def log_trade_decision_api():
    """API para registrar decisão de trading"""
    if not session.get('authenticated'):
        return jsonify({'error': 'Não autenticado'}), 401
    
    try:
        data = request.get_json()
        decision_id = mt5_auth.log_trade_decision(
            symbol=data.get('symbol'),
            action=data.get('action'),
            volume=data.get('volume'),
            analysis_factors=data.get('analysis_factors', []),
            justification=data.get('justification', ''),
            confidence=data.get('confidence', 0.8)
        )
        
        return jsonify({
            'success': True,
            'decision_id': decision_id
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@flask_app.route('/api/log_no_trade', methods=['POST'])
def log_no_trade_api():
    """API para registrar análise sem operação"""
    if not session.get('authenticated'):
        return jsonify({'error': 'Não autenticado'}), 401
    
    try:
        data = request.get_json()
        decision_id = mt5_auth.log_no_trade_analysis(
            symbol=data.get('symbol'),
            reasons=data.get('reasons', []),
            analysis_performed=data.get('analysis_performed', []),
            confidence=data.get('confidence', 0.8)
        )
        
        return jsonify({
            'success': True,
            'decision_id': decision_id
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@flask_app.route('/api/export_detailed_report')
def export_detailed_report_api():
    """API para exportar relatório detalhado"""
    if not session.get('authenticated'):
        return jsonify({'error': 'Não autenticado'}), 401
    
    try:
        report_file = mt5_auth.export_detailed_report()
        return jsonify({
            'success': True,
            'report_file': report_file
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--web':
        # Executar interface web Flask
        print("Iniciando interface web Flask...")
        print("Acesse: http://localhost:5001")
        print("Senha: mt5secure2024")
        flask_app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        # Executar interface Tkinter
        try:
            app = MT5DashboardApp()
            app.mainloop()
        except KeyboardInterrupt:
            print("\nDashboard interrompido pelo usuário")
        except Exception as e:
            print(f"Erro no dashboard: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Aplicação finalizada")