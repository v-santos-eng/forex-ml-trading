#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para lançar o dashboard com configurações de visibilidade forçada
"""

import tkinter as tk
from mt5_dashboard_integrated import MT5DashboardIntegrated
import sys
import os

def force_window_visible(root):
    """Força a janela a aparecer na frente e ser visível"""
    try:
        # Trazer para frente
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(lambda: root.attributes('-topmost', False))
        
        # Focar na janela
        root.focus_force()
        
        # Garantir que não está minimizada
        root.state('normal')
        
        # Atualizar a janela
        root.update()
        
        print("Janela configurada para ser visível")
        
    except Exception as e:
        print(f"Erro ao configurar visibilidade: {e}")

def main():
    """Função principal"""
    print("Iniciando Dashboard MT5 com visibilidade forçada...")
    
    try:
        # Criar dashboard
        dashboard = MT5DashboardIntegrated()
        
        # Configurar visibilidade
        force_window_visible(dashboard.root)
        
        # Configurações adicionais de janela
        dashboard.root.geometry("1400x900+100+50")  # Posição específica
        dashboard.root.title("MT5 Dashboard - ATIVO")
        
        print("Dashboard criado e configurado")
        print("Janela deveria estar visível agora")
        print("Pressione Ctrl+C para fechar")
        
        # Executar
        dashboard.run()
        
    except KeyboardInterrupt:
        print("\nDashboard fechado pelo usuário")
    except Exception as e:
        print(f"Erro ao executar dashboard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()