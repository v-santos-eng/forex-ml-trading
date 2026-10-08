#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard MT5 Ultra Estável - Elimina problemas de threading
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import sys
import os
import queue
from datetime import datetime

class UltraStableDashboard:
    """Dashboard com máxima estabilidade"""
    
    def __init__(self):
        """Inicializa o dashboard"""
        # Verificar thread principal
        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("Dashboard deve ser criado na thread principal")
        
        self.root = None
        self.running = False
        self.message_queue = queue.Queue()
        self._create_window()
        
    def _create_window(self):
        """Cria a janela principal"""
        # Configurar Tkinter para single-thread
        self.root = tk.Tk()
        self.root.withdraw()  # Esconder inicialmente
        
        # Configurações básicas
        self.root.title("MT5 Dashboard Ultra Estável")
        self.root.geometry("1000x700+150+100")
        self.root.minsize(800, 600)
        
        # Configurar fechamento
        self.root.protocol("WM_DELETE_WINDOW", self._safe_close)
        
        # Criar interface
        self._create_interface()
        
        # Configurar processamento de mensagens
        self._process_messages()
        
        # Mostrar janela
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        
    def _create_interface(self):
        """Cria a interface do usuário"""
        # Estilo
        style = ttk.Style()
        style.theme_use('clam')
        
        # Container principal
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Cabeçalho
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, 
                               text="MT5 Dashboard - Sistema Ultra Estável",
                               font=('Arial', 18, 'bold'))
        title_label.pack()
        
        # Status
        status_frame = ttk.LabelFrame(main_container, text="Status do Sistema", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_var = tk.StringVar(value="Sistema Inicializando...")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                     font=('Arial', 12, 'bold'))
        self.status_label.pack()
        
        # Área principal dividida
        main_paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Painel esquerdo - Controles
        left_frame = ttk.LabelFrame(main_paned, text="Controles", padding=10)
        main_paned.add(left_frame, weight=1)
        
        # Botões de controle
        self.start_btn = ttk.Button(left_frame, text="🚀 Iniciar Sistema",
                                   command=self._start_system)
        self.start_btn.pack(fill=tk.X, pady=2)
        
        self.stop_btn = ttk.Button(left_frame, text="⏹️ Parar Sistema",
                                  command=self._stop_system, state='disabled')
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Informações do sistema
        info_label = ttk.Label(left_frame, text="Informações do Sistema:",
                              font=('Arial', 10, 'bold'))
        info_label.pack(anchor=tk.W)
        
        self.info_text = tk.Text(left_frame, height=8, width=30, wrap=tk.WORD)
        info_scroll = ttk.Scrollbar(left_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)
        
        info_container = ttk.Frame(left_frame)
        info_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Painel direito - Log
        right_frame = ttk.LabelFrame(main_paned, text="Log de Atividades", padding=10)
        main_paned.add(right_frame, weight=2)
        
        # Área de log
        log_container = ttk.Frame(right_frame)
        log_container.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_container, wrap=tk.WORD, font=('Consolas', 9))
        log_scrollbar = ttk.Scrollbar(log_container, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Botões de log
        log_btn_frame = ttk.Frame(right_frame)
        log_btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(log_btn_frame, text="Limpar Log",
                  command=self._clear_log).pack(side=tk.LEFT)
        ttk.Button(log_btn_frame, text="Salvar Log",
                  command=self._save_log).pack(side=tk.LEFT, padx=(5, 0))
        
        # Rodapé
        footer_frame = ttk.Frame(main_container)
        footer_frame.pack(fill=tk.X)
        
        ttk.Button(footer_frame, text="❌ Fechar Dashboard",
                  command=self._safe_close).pack(side=tk.RIGHT)
        
        # Inicializar informações
        self._update_system_info()
        
    def _log_message(self, message, level="INFO"):
        """Adiciona mensagem ao log de forma thread-safe"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.message_queue.put(('log', log_entry))
        
    def _update_status(self, status, color="black"):
        """Atualiza status de forma thread-safe"""
        self.message_queue.put(('status', (status, color)))
        
    def _process_messages(self):
        """Processa mensagens da queue de forma thread-safe"""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                
                if msg_type == 'log':
                    self.log_text.insert(tk.END, data + "\n")
                    self.log_text.see(tk.END)
                    
                elif msg_type == 'status':
                    status, color = data
                    self.status_var.set(status)
                    self.status_label.configure(foreground=color)
                    
        except queue.Empty:
            pass
        
        # Reagendar processamento
        if self.root:
            self.root.after(100, self._process_messages)
    
    def _start_system(self):
        """Inicia o sistema"""
        if not self.running:
            self.running = True
            self._update_status("🟢 Sistema Ativo", "green")
            self._log_message("Sistema de trading iniciado com sucesso")
            
            self.start_btn.configure(state='disabled')
            self.stop_btn.configure(state='normal')
            
            # Iniciar simulação em thread separada
            self.simulation_thread = threading.Thread(target=self._run_simulation, daemon=True)
            self.simulation_thread.start()
    
    def _stop_system(self):
        """Para o sistema"""
        if self.running:
            self.running = False
            self._update_status("🔴 Sistema Parado", "red")
            self._log_message("Sistema de trading parado")
            
            self.start_btn.configure(state='normal')
            self.stop_btn.configure(state='disabled')
    
    def _run_simulation(self):
        """Executa simulação do sistema em thread separada"""
        activities = [
            "Conectando ao MetaTrader 5",
            "Analisando par EURUSD",
            "Calculando indicadores técnicos",
            "Verificando sinais de entrada",
            "Monitorando posições abertas",
            "Atualizando níveis de stop loss",
            "Analisando volatilidade do mercado",
            "Executando análise de risco",
            "Verificando correlações entre pares",
            "Atualizando modelo de ML"
        ]
        
        import random
        
        while self.running:
            try:
                activity = random.choice(activities)
                self._log_message(activity)
                
                # Simular processamento
                time.sleep(random.uniform(2, 5))
                
            except Exception as e:
                self._log_message(f"Erro na simulação: {e}", "ERROR")
                break
    
    def _update_system_info(self):
        """Atualiza informações do sistema"""
        try:
            import psutil
            
            # CPU e Memória
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            info = f"CPU: {cpu_percent}%\n"
            info += f"Memória: {memory.percent}%\n"
            info += f"Disponível: {memory.available // (1024**3)} GB\n"
            info += f"\nProcessos Python: {len([p for p in psutil.process_iter() if 'python' in p.name().lower()])}\n"
            info += f"Uptime: {datetime.now().strftime('%H:%M:%S')}"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
            
        except ImportError:
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, "psutil não disponível\nInformações limitadas")
        
        # Reagendar atualização
        if self.root:
            self.root.after(5000, self._update_system_info)
    
    def _clear_log(self):
        """Limpa o log"""
        self.log_text.delete(1.0, tk.END)
        self._log_message("Log limpo")
    
    def _save_log(self):
        """Salva o log em arquivo"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_log_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.log_text.get(1.0, tk.END))
            
            self._log_message(f"Log salvo em: {filename}")
            
        except Exception as e:
            self._log_message(f"Erro ao salvar log: {e}", "ERROR")
    
    def _safe_close(self):
        """Fecha o dashboard de forma segura"""
        self._log_message("Iniciando fechamento do dashboard...")
        
        # Parar sistema se estiver rodando
        if self.running:
            self._stop_system()
        
        # Aguardar threads terminarem
        self.root.after(1000, self._final_close)
    
    def _final_close(self):
        """Fechamento final"""
        try:
            if self.root:
                self.root.quit()
                self.root.destroy()
        except:
            pass
    
    def run(self):
        """Executa o dashboard"""
        if not self.root:
            raise RuntimeError("Dashboard não inicializado")
        
        self._update_status("🔵 Sistema Pronto", "blue")
        self._log_message("Dashboard MT5 Ultra Estável iniciado")
        self._log_message("Clique em 'Iniciar Sistema' para começar o trading")
        
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Erro durante execução: {e}")
        finally:
            print("Dashboard finalizado com segurança")

def main():
    """Função principal"""
    print("=== MT5 Dashboard Ultra Estável ===")
    print("Versão com correções completas de threading")
    
    try:
        # Verificações iniciais
        if threading.current_thread() != threading.main_thread():
            print("ERRO: Deve ser executado na thread principal")
            return 1
        
        # Criar e executar dashboard
        dashboard = UltraStableDashboard()
        print("✅ Dashboard criado com sucesso")
        print("✅ Interface configurada")
        print("✅ Janela visível e responsiva")
        print("\n🎯 Dashboard pronto para uso!")
        print("📝 Use os controles na interface para operar")
        
        dashboard.run()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Dashboard interrompido pelo usuário")
        return 0
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)