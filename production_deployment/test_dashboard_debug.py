#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de diagnóstico para o dashboard MT5
Testa cada componente individualmente para identificar problemas
"""

import sys
import traceback
import os
from pathlib import Path

def test_basic_imports():
    """Testa importações básicas"""
    print("=== Testando Importações Básicas ===")
    try:
        import tkinter as tk
        print("✓ tkinter importado com sucesso")
        
        from tkinter import ttk, messagebox, filedialog
        print("✓ módulos tkinter importados com sucesso")
        
        import threading
        import time
        from datetime import datetime, timedelta
        from typing import Dict, List, Optional, Any
        import json
        print("✓ módulos padrão importados com sucesso")
        
        return True
    except Exception as e:
        print(f"✗ Erro nas importações básicas: {e}")
        traceback.print_exc()
        return False

def test_custom_imports():
    """Testa importações customizadas"""
    print("\n=== Testando Importações Customizadas ===")
    
    modules_to_test = [
        'advanced_widgets',
        'notification_system', 
        'export_system',
        'settings_system',
        'user_preferences',
        'performance_optimization'
    ]
    
    success_count = 0
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module} importado com sucesso")
            success_count += 1
        except Exception as e:
            print(f"✗ Erro ao importar {module}: {e}")
            traceback.print_exc()
    
    return success_count == len(modules_to_test)

def test_mt5_import():
    """Testa importação do MetaTrader5"""
    print("\n=== Testando MetaTrader5 ===")
    try:
        import MetaTrader5 as mt5
        print("✓ MetaTrader5 importado com sucesso")
        return True
    except ImportError:
        print("⚠ MetaTrader5 não disponível - modo simulação")
        return True  # Não é erro crítico
    except Exception as e:
        print(f"✗ Erro ao importar MetaTrader5: {e}")
        return False

def test_tkinter_window():
    """Testa criação de janela Tkinter básica"""
    print("\n=== Testando Janela Tkinter ===")
    try:
        import tkinter as tk
        
        root = tk.Tk()
        root.title("Teste Dashboard Debug")
        root.geometry("400x300")
        
        label = tk.Label(root, text="Teste de Janela", font=('Arial', 16))
        label.pack(expand=True)
        
        # Mostrar janela por 2 segundos
        root.after(2000, root.destroy)
        root.mainloop()
        
        print("✓ Janela Tkinter criada e exibida com sucesso")
        return True
        
    except Exception as e:
        print(f"✗ Erro ao criar janela Tkinter: {e}")
        traceback.print_exc()
        return False

def test_dashboard_class():
    """Testa importação e criação da classe do dashboard"""
    print("\n=== Testando Classe Dashboard ===")
    try:
        from mt5_dashboard_integrated import MT5DashboardIntegrated
        print("✓ Classe MT5DashboardIntegrated importada com sucesso")
        
        # Tentar criar instância sem executar
        dashboard = MT5DashboardIntegrated()
        print("✓ Instância do dashboard criada com sucesso")
        
        # Verificar se a janela foi criada
        if hasattr(dashboard, 'root') and dashboard.root:
            print("✓ Janela root criada com sucesso")
            
            # Tentar mostrar a janela por 3 segundos
            dashboard.root.after(3000, dashboard.root.destroy)
            dashboard.root.mainloop()
            
            print("✓ Dashboard executado com sucesso")
            return True
        else:
            print("✗ Janela root não foi criada")
            return False
            
    except Exception as e:
        print(f"✗ Erro ao testar classe dashboard: {e}")
        traceback.print_exc()
        return False

def main():
    """Função principal de diagnóstico"""
    print("DIAGNÓSTICO DO DASHBOARD MT5")
    print("=" * 50)
    
    # Verificar diretório atual
    print(f"Diretório atual: {os.getcwd()}")
    print(f"Python: {sys.version}")
    print(f"Plataforma: {sys.platform}")
    
    tests = [
        ("Importações Básicas", test_basic_imports),
        ("Importações Customizadas", test_custom_imports),
        ("MetaTrader5", test_mt5_import),
        ("Janela Tkinter", test_tkinter_window),
        ("Classe Dashboard", test_dashboard_class)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Erro crítico no teste {test_name}: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Resumo dos resultados
    print("\n" + "=" * 50)
    print("RESUMO DOS TESTES")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASSOU" if result else "✗ FALHOU"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTestes aprovados: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 Todos os testes passaram! O dashboard deveria funcionar.")
    else:
        print(f"\n⚠ {len(results) - passed} teste(s) falharam. Verifique os erros acima.")

if __name__ == "__main__":
    main()