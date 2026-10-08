#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Teste de Performance e Responsividade do Dashboard MT5

Este script testa:
- Tempo de inicialização
- Responsividade da interface
- Uso de memória
- Performance de widgets
- Funcionalidade de redimensionamento

Autor: Sistema de Trading ML
Data: 2024
"""

import time
import psutil
import threading
import tkinter as tk
from tkinter import messagebox
import sys
import os

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from mt5_dashboard_integrated import MT5DashboardIntegrated
except ImportError as e:
    print(f"Erro ao importar dashboard: {e}")
    sys.exit(1)

class DashboardPerformanceTester:
    """Classe para testar performance do dashboard"""
    
    def __init__(self):
        self.results = {}
        self.dashboard = None
        self.root = None
        
    def measure_startup_time(self):
        """Medir tempo de inicialização do dashboard"""
        print("Testando tempo de inicialização...")
        
        start_time = time.time()
        
        try:
            # A classe MT5DashboardIntegrated cria sua própria janela
            self.dashboard = MT5DashboardIntegrated()
            self.root = self.dashboard.root
            
            # Aguardar a interface ser totalmente carregada
            self.root.update()
            
            end_time = time.time()
            startup_time = end_time - start_time
            
            self.results['startup_time'] = startup_time
            print(f"✓ Tempo de inicialização: {startup_time:.2f}s")
            
            return startup_time < 5.0  # Deve inicializar em menos de 5 segundos
            
        except Exception as e:
            print(f"✗ Erro na inicialização: {e}")
            self.results['startup_error'] = str(e)
            return False
    
    def measure_memory_usage(self):
        """Medir uso de memória"""
        print("Testando uso de memória...")
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.results['memory_usage_mb'] = memory_mb
            print(f"✓ Uso de memória: {memory_mb:.1f} MB")
            
            return memory_mb < 200  # Deve usar menos de 200MB
            
        except Exception as e:
            print(f"✗ Erro ao medir memória: {e}")
            return False
    
    def test_widget_responsiveness(self):
        """Testar responsividade dos widgets"""
        print("Testando responsividade dos widgets...")
        
        if not self.dashboard or not self.root:
            print("✗ Dashboard não inicializado")
            return False
        
        try:
            # Testar atualização de widgets múltiplas vezes
            start_time = time.time()
            
            for i in range(10):
                self.root.update()
                time.sleep(0.01)
            
            end_time = time.time()
            update_time = end_time - start_time
            
            self.results['widget_update_time'] = update_time
            print(f"✓ Tempo de atualização (10x): {update_time:.3f}s")
            
            return update_time < 1.0  # Deve atualizar rapidamente
            
        except Exception as e:
            print(f"✗ Erro na responsividade: {e}")
            return False
    
    def test_window_resizing(self):
        """Testar redimensionamento da janela"""
        print("Testando redimensionamento...")
        
        if not self.root:
            print("✗ Janela não inicializada")
            return False
        
        try:
            original_size = (self.root.winfo_width(), self.root.winfo_height())
            
            # Testar diferentes tamanhos
            test_sizes = [(800, 600), (1200, 800), (1600, 1000)]
            
            for width, height in test_sizes:
                self.root.geometry(f"{width}x{height}")
                self.root.update()
                time.sleep(0.1)
            
            # Restaurar tamanho original
            self.root.geometry(f"{original_size[0]}x{original_size[1]}")
            self.root.update()
            
            print("✓ Redimensionamento funcionando")
            return True
            
        except Exception as e:
            print(f"✗ Erro no redimensionamento: {e}")
            return False
    
    def test_notification_system(self):
        """Testar sistema de notificações"""
        print("Testando sistema de notificações...")
        
        if not self.dashboard:
            print("✗ Dashboard não inicializado")
            return False
        
        try:
            # Testar diferentes tipos de notificação
            if hasattr(self.dashboard, 'notification_manager'):
                self.dashboard.notification_manager.info(
                    "Teste",
                    "Teste de notificação"
                )
                
                self.root.update()
                print("✓ Sistema de notificações funcionando")
                return True
            else:
                print("✗ Sistema de notificações não encontrado")
                return False
                
        except Exception as e:
            print(f"✗ Erro no sistema de notificações: {e}")
            return False
    
    def test_data_provider_integration(self):
        """Testar integração do provedor de dados"""
        print("Testando integração do provedor de dados...")
        
        if not self.dashboard:
            print("✗ Dashboard não inicializado")
            return False
        
        try:
            if hasattr(self.dashboard, 'data_provider'):
                # Verificar se o provedor de dados existe
                print("✓ Provedor de dados disponível")
                return True
            else:
                print("⚠ Provedor de dados não encontrado")
                return False
                
        except Exception as e:
            print(f"✗ Erro no provedor de dados: {e}")
            return False
    
    def test_export_system(self):
        """Testar sistema de exportação"""
        print("Testando sistema de exportação...")
        
        if not self.dashboard:
            print("✗ Dashboard não inicializado")
            return False
        
        try:
            if hasattr(self.dashboard, 'export_manager'):
                print("✓ Sistema de exportação disponível")
                return True
            else:
                print("✗ Sistema de exportação não encontrado")
                return False
                
        except Exception as e:
            print(f"✗ Erro no sistema de exportação: {e}")
            return False
    
    def run_all_tests(self):
        """Executar todos os testes"""
        print("=" * 50)
        print("INICIANDO TESTES DE PERFORMANCE DO DASHBOARD")
        print("=" * 50)
        
        tests = [
            ("Inicialização", self.measure_startup_time),
            ("Uso de Memória", self.measure_memory_usage),
            ("Responsividade", self.test_widget_responsiveness),
            ("Redimensionamento", self.test_window_resizing),
            ("Notificações", self.test_notification_system),
            ("Data Provider", self.test_data_provider_integration),
            ("Exportação", self.test_export_system)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n[{test_name}]")
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                print(f"✗ Erro inesperado em {test_name}: {e}")
        
        print("\n" + "=" * 50)
        print(f"RESULTADOS: {passed}/{total} testes passaram")
        print("=" * 50)
        
        # Exibir métricas detalhadas
        if self.results:
            print("\nMÉTRICAS DETALHADAS:")
            for key, value in self.results.items():
                print(f"  {key}: {value}")
        
        # Cleanup
        if self.root:
            self.root.destroy()
        
        return passed == total
    
    def run_stress_test(self, duration=30):
        """Executar teste de stress por um período determinado"""
        print(f"\nIniciando teste de stress por {duration} segundos...")
        
        if not self.dashboard or not self.root:
            print("✗ Dashboard não inicializado para teste de stress")
            return False
        
        start_time = time.time()
        update_count = 0
        
        try:
            while time.time() - start_time < duration:
                self.root.update()
                update_count += 1
                
                # Simular atividade
                if update_count % 100 == 0:
                    if hasattr(self.dashboard, 'notification_manager'):
                        self.dashboard.notification_manager.info(
                            "Stress Test",
                            f"Teste stress #{update_count//100}"
                        )
                
                time.sleep(0.01)
            
            end_time = time.time()
            actual_duration = end_time - start_time
            updates_per_second = update_count / actual_duration
            
            print(f"✓ Teste de stress concluído:")
            print(f"  - Duração: {actual_duration:.1f}s")
            print(f"  - Atualizações: {update_count}")
            print(f"  - FPS médio: {updates_per_second:.1f}")
            
            return updates_per_second > 50  # Deve manter pelo menos 50 FPS
            
        except Exception as e:
            print(f"✗ Erro no teste de stress: {e}")
            return False

def main():
    """Função principal"""
    tester = DashboardPerformanceTester()
    
    try:
        # Executar testes básicos
        success = tester.run_all_tests()
        
        if success:
            print("\n🎉 Todos os testes básicos passaram!")
            
            # Perguntar se deve executar teste de stress
            response = input("\nExecutar teste de stress? (s/n): ")
            if response.lower() in ['s', 'sim', 'y', 'yes']:
                # Reinicializar para teste de stress
                tester = DashboardPerformanceTester()
                tester.measure_startup_time()
                stress_success = tester.run_stress_test(30)
                
                if stress_success:
                    print("\n🚀 Teste de stress passou!")
                else:
                    print("\n⚠ Teste de stress falhou")
        else:
            print("\n❌ Alguns testes falharam. Verifique os problemas acima.")
    
    except KeyboardInterrupt:
        print("\n\n⏹ Testes interrompidos pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
    
    print("\nTestes finalizados.")

if __name__ == "__main__":
    main()