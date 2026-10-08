#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versão corrigida do dashboard que resolve problemas de threading
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import sys
import os
from datetime import datetime

class FixedMT5Dashboard:
    """Dashboard MT5 com correções de threading"""
    
    def __init__(self):
        """Inicializa o dashboard"""
        self.root = None
        self.running = False
        self._setup_window()
        self._setup_ui()
        
    def _setup_window(self):
        """Configura a janela principal"""
        # Garantir que estamos na thread principal
        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("Dashboard deve ser criado na thread principal")
            
        self.root = tk.Tk()
        self.root.title("MT5 Dashboard - Versão Estável")
        self.root.geometry("1200x800+100+50")
        
        # Configurar protocolo de fechamento
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Configurações de visibilidade
        self.root.lift()
        self.root.focus_force()
        self.root.state('normal')
        
    def _setup_ui(self):
        """Configura a interface do usuário"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="MT5 Dashboard - Sistema de Trading", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Status do sistema
        status_frame = ttk.LabelFrame(main_frame, text="Status do Sistema", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Sistema Inicializando...", 
                                     foreground="orange")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Informações do sistema
        info_frame = ttk.LabelFrame(main_frame, text="Informações", padding="10")
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Log de atividades
        log_frame = ttk.LabelFrame(main_frame, text="Log de Atividades", padding="10")
        log_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Text widget para logs
        self.log_text = tk.Text(log_frame, height=20, width=50)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Botões de controle
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(button_frame, text="Iniciar Sistema", 
                  command=self._start_system).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Parar Sistema", 
                  command=self._stop_system).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Limpar Log", 
                  command=self._clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Fechar", 
                  command=self._on_closing).pack(side=tk.LEFT, padx=(5, 0))
        
    def _log_message(self, message):
        """Adiciona mensagem ao log"""
        if self.root and self.log_text:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            # Usar after para garantir thread safety
            self.root.after(0, lambda: self._append_to_log(log_entry))
    
    def _append_to_log(self, text):
        """Adiciona texto ao log (thread-safe)"""
        try:
            self.log_text.insert(tk.END, text)
            self.log_text.see(tk.END)
        except tk.TclError:
            pass  # Janela foi fechada
    
    def _start_system(self):
        """Inicia o sistema de trading"""
        if not self.running:
            self.running = True
            self.status_label.config(text="Sistema Ativo", foreground="green")
            self._log_message("Sistema de trading iniciado")
            
            # Simular atividade do sistema
            self._simulate_activity()
    
    def _stop_system(self):
        """Para o sistema de trading"""
        if self.running:
            self.running = False
            self.status_label.config(text="Sistema Parado", foreground="red")
            self._log_message("Sistema de trading parado")
    
    def _clear_log(self):
        """Limpa o log"""
        self.log_text.delete(1.0, tk.END)
        self._log_message("Log limpo")
    
    def _simulate_activity(self):
        """Simula atividade do sistema"""
        if self.running and self.root:
            activities = [
                "Analisando mercado EURUSD",
                "Verificando sinais de entrada",
                "Monitorando posições abertas",
                "Calculando níveis de risco",
                "Atualizando indicadores técnicos"
            ]
            
            import random
            activity = random.choice(activities)
            self._log_message(activity)
            
            # Agendar próxima atividade
            self.root.after(3000, self._simulate_activity)
    
    def _on_closing(self):
        """Manipula o fechamento da janela"""
        self._log_message("Fechando dashboard...")
        self.running = False
        
        # Aguardar um pouco para processar mensagens pendentes
        self.root.after(500, self._destroy_window)
    
    def _destroy_window(self):
        """Destrói a janela de forma segura"""
        try:
            if self.root:
                self.root.quit()
                self.root.destroy()
        except tk.TclError:
            pass  # Janela já foi fechada
    
    def run(self):
        """Executa o dashboard"""
        if not self.root:
            raise RuntimeError("Dashboard não foi inicializado")
        
        # Configurar status inicial
        self.status_label.config(text="Sistema Pronto", foreground="blue")
        self._log_message("Dashboard MT5 iniciado com sucesso")
        self._log_message("Clique em 'Iniciar Sistema' para começar")
        
        try:
            # Executar mainloop
            self.root.mainloop()
        except Exception as e:
            print(f"Erro durante execução: {e}")
        finally:
            print("Dashboard finalizado")

def main():
    """Função principal"""
    print("Iniciando Dashboard MT5 - Versão Corrigida")
    
    try:
        # Verificar se estamos na thread principal
        if threading.current_thread() != threading.main_thread():
            print("ERRO: Dashboard deve ser executado na thread principal")
            return
        
        # Criar e executar dashboard
        dashboard = FixedMT5Dashboard()
        print("Dashboard criado com sucesso")
        print("Janela deveria estar visível agora")
        print("Pressione o botão 'Fechar' ou Ctrl+C para sair")
        
        dashboard.run()
        
    except KeyboardInterrupt:
        print("\nDashboard interrompido pelo usuário")
    except Exception as e:
        print(f"Erro ao executar dashboard: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Dashboard finalizado")

if __name__ == "__main__":
    main()