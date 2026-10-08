#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de diagnóstico para identificar problemas de estabilidade do dashboard
"""

import tkinter as tk
import sys
import os
import time
import threading
import traceback
from datetime import datetime

def log_message(message):
    """Log com timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def test_basic_tkinter():
    """Testa funcionalidade básica do Tkinter"""
    try:
        log_message("Testando Tkinter básico...")
        root = tk.Tk()
        root.title("Teste Básico")
        root.geometry("300x200")
        
        # Teste de 5 segundos
        def close_after_delay():
            time.sleep(5)
            root.quit()
            root.destroy()
        
        thread = threading.Thread(target=close_after_delay)
        thread.daemon = True
        thread.start()
        
        log_message("Janela Tkinter criada, testando por 5 segundos...")
        root.mainloop()
        log_message("Teste Tkinter básico: SUCESSO")
        return True
        
    except Exception as e:
        log_message(f"Erro no teste Tkinter básico: {e}")
        traceback.print_exc()
        return False

def test_dashboard_import():
    """Testa importação do dashboard"""
    try:
        log_message("Testando importação do dashboard...")
        from mt5_dashboard_integrated import MT5DashboardIntegrated
        log_message("Importação do dashboard: SUCESSO")
        return True
    except Exception as e:
        log_message(f"Erro na importação do dashboard: {e}")
        traceback.print_exc()
        return False

def test_dashboard_creation():
    """Testa criação do dashboard"""
    try:
        log_message("Testando criação do dashboard...")
        from mt5_dashboard_integrated import MT5DashboardIntegrated
        
        dashboard = MT5DashboardIntegrated()
        log_message("Dashboard criado com sucesso")
        
        # Configurar janela
        dashboard.root.geometry("800x600")
        dashboard.root.title("Teste Dashboard")
        
        # Teste de estabilidade por 10 segundos
        def monitor_dashboard():
            start_time = time.time()
            while time.time() - start_time < 10:
                try:
                    # Verificar se a janela ainda existe
                    dashboard.root.winfo_exists()
                    time.sleep(1)
                except tk.TclError:
                    log_message("ERRO: Janela foi fechada inesperadamente!")
                    return False
            
            log_message("Dashboard permaneceu estável por 10 segundos")
            dashboard.root.quit()
            dashboard.root.destroy()
            return True
        
        # Executar monitoramento em thread separada
        monitor_thread = threading.Thread(target=monitor_dashboard)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        log_message("Iniciando mainloop do dashboard...")
        dashboard.root.mainloop()
        
        monitor_thread.join(timeout=1)
        log_message("Teste de criação do dashboard: SUCESSO")
        return True
        
    except Exception as e:
        log_message(f"Erro na criação do dashboard: {e}")
        traceback.print_exc()
        return False

def test_dashboard_with_exception_handling():
    """Testa dashboard com captura detalhada de exceções"""
    try:
        log_message("Testando dashboard com captura de exceções...")
        from mt5_dashboard_integrated import MT5DashboardIntegrated
        
        # Configurar captura de exceções do Tkinter
        def handle_exception(exc, val, tb):
            log_message(f"EXCEÇÃO CAPTURADA: {exc.__name__}: {val}")
            traceback.print_exception(exc, val, tb)
        
        # Instalar handler de exceções
        sys.excepthook = handle_exception
        
        dashboard = MT5DashboardIntegrated()
        
        # Configurar callback para detectar fechamento
        def on_closing():
            log_message("Dashboard sendo fechado via protocolo WM_DELETE_WINDOW")
            dashboard.root.destroy()
        
        dashboard.root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Configurar timer para fechar automaticamente
        def auto_close():
            time.sleep(15)
            log_message("Fechando dashboard automaticamente após 15 segundos")
            try:
                dashboard.root.quit()
            except:
                pass
        
        timer_thread = threading.Thread(target=auto_close)
        timer_thread.daemon = True
        timer_thread.start()
        
        log_message("Dashboard configurado, iniciando execução...")
        dashboard.run()
        
        log_message("Dashboard executado com sucesso")
        return True
        
    except Exception as e:
        log_message(f"Erro no teste com exception handling: {e}")
        traceback.print_exc()
        return False

def main():
    """Função principal de diagnóstico"""
    log_message("=== INICIANDO DIAGNÓSTICO DE ESTABILIDADE DO DASHBOARD ===")
    
    tests = [
        ("Teste Tkinter Básico", test_basic_tkinter),
        ("Teste Importação Dashboard", test_dashboard_import),
        ("Teste Criação Dashboard", test_dashboard_creation),
        ("Teste Dashboard com Exception Handling", test_dashboard_with_exception_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        log_message(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            log_message(f"{test_name}: {'PASSOU' if result else 'FALHOU'}")
        except Exception as e:
            log_message(f"{test_name}: ERRO - {e}")
            results.append((test_name, False))
        
        # Pausa entre testes
        time.sleep(2)
    
    # Resumo final
    log_message("\n=== RESUMO DOS TESTES ===")
    for test_name, result in results:
        status = "✓ PASSOU" if result else "✗ FALHOU"
        log_message(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    log_message(f"\nResultado: {passed}/{total} testes passaram")
    
    if passed == total:
        log_message("DIAGNÓSTICO: Dashboard deveria funcionar normalmente")
    else:
        log_message("DIAGNÓSTICO: Problemas identificados que podem causar instabilidade")

if __name__ == "__main__":
    main()