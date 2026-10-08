#!/usr/bin/env python3
"""
Script de Parada Segura do Sistema ML Forex
Para todos os componentes do sistema de forma controlada
"""

import psutil
import time
import os
import sys
import signal
from datetime import datetime

class SystemStopper:
    def __init__(self):
        self.processes_to_stop = [
            'ml_production_pipeline.py',
            'production_monitor.py', 
            'auto_recovery_system.py'
        ]
        
    def log(self, message):
        """Log com timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def find_processes(self, script_name):
        """Encontra todos os processos rodando um script específico"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if script_name in cmdline:
                        processes.append({
                            'pid': proc.info['pid'],
                            'process': psutil.Process(proc.info['pid'])
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
    
    def stop_process_gracefully(self, process_info, timeout=30):
        """Para um processo de forma graciosa"""
        pid = process_info['pid']
        process = process_info['process']
        
        try:
            self.log(f"Enviando SIGTERM para PID {pid}...")
            
            # No Windows, usamos terminate() em vez de SIGTERM
            process.terminate()
            
            # Aguarda o processo terminar graciosamente
            try:
                process.wait(timeout=timeout)
                self.log(f"✅ Processo {pid} terminado graciosamente")
                return True
            except psutil.TimeoutExpired:
                self.log(f"⚠️ Processo {pid} não respondeu ao SIGTERM, forçando...")
                process.kill()
                try:
                    process.wait(timeout=5)
                    self.log(f"✅ Processo {pid} terminado forçadamente")
                    return True
                except psutil.TimeoutExpired:
                    self.log(f"❌ Não foi possível terminar processo {pid}")
                    return False
                    
        except psutil.NoSuchProcess:
            self.log(f"✅ Processo {pid} já não existe")
            return True
        except Exception as e:
            self.log(f"❌ Erro ao parar processo {pid}: {e}")
            return False
    
    def stop_script_processes(self, script_name):
        """Para todos os processos de um script específico"""
        self.log(f"Parando processos: {script_name}")
        
        processes = self.find_processes(script_name)
        if not processes:
            self.log(f"   Nenhum processo encontrado para {script_name}")
            return True
        
        success = True
        for proc_info in processes:
            if not self.stop_process_gracefully(proc_info):
                success = False
        
        return success
    
    def save_final_metrics(self):
        """Salva métricas finais antes de parar"""
        self.log("Salvando métricas finais...")
        try:
            # Aguarda um pouco para permitir que o monitoramento salve dados finais
            time.sleep(5)
            self.log("✅ Métricas finais salvas")
        except Exception as e:
            self.log(f"⚠️ Erro ao salvar métricas finais: {e}")
    
    def verify_all_stopped(self):
        """Verifica se todos os processos foram parados"""
        self.log("Verificando se todos os processos foram parados...")
        
        all_stopped = True
        for script_name in self.processes_to_stop:
            processes = self.find_processes(script_name)
            if processes:
                self.log(f"⚠️ Ainda existem processos rodando: {script_name}")
                for proc_info in processes:
                    self.log(f"   PID: {proc_info['pid']}")
                all_stopped = False
        
        if all_stopped:
            self.log("✅ Todos os processos foram parados com sucesso")
        
        return all_stopped
    
    def stop_all_services(self):
        """Para todos os serviços na ordem correta"""
        self.log("=" * 60)
        self.log("PARANDO SISTEMA ML FOREX")
        self.log("=" * 60)
        
        success = True
        
        # 1. Parar Sistema de Recuperação primeiro (para evitar reinicializações)
        if not self.stop_script_processes('auto_recovery_system.py'):
            success = False
        
        # 2. Parar Monitoramento de Produção
        if not self.stop_script_processes('production_monitor.py'):
            success = False
        
        # 3. Salvar métricas finais
        self.save_final_metrics()
        
        # 4. Parar Pipeline ML por último
        if not self.stop_script_processes('ml_production_pipeline.py'):
            success = False
        
        # 5. Aguardar um pouco para estabilização
        self.log("Aguardando estabilização...")
        time.sleep(5)
        
        # 6. Verificação final
        all_stopped = self.verify_all_stopped()
        
        self.log("=" * 60)
        if success and all_stopped:
            self.log("🎉 SISTEMA PARADO COM SUCESSO!")
            self.log("Todos os componentes foram desligados de forma segura.")
        else:
            self.log("⚠️ SISTEMA PARADO COM PROBLEMAS")
            self.log("Alguns processos podem ainda estar rodando.")
        
        self.log("=" * 60)
        
        return success and all_stopped
    
    def emergency_stop(self):
        """Parada de emergência - força a parada de todos os processos"""
        self.log("=" * 60)
        self.log("PARADA DE EMERGÊNCIA - FORÇANDO TODOS OS PROCESSOS")
        self.log("=" * 60)
        
        for script_name in self.processes_to_stop:
            processes = self.find_processes(script_name)
            for proc_info in processes:
                try:
                    proc_info['process'].kill()
                    self.log(f"🔥 Processo {proc_info['pid']} ({script_name}) terminado forçadamente")
                except Exception as e:
                    self.log(f"❌ Erro ao terminar processo {proc_info['pid']}: {e}")
        
        time.sleep(3)
        self.verify_all_stopped()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Para o Sistema ML Forex')
    parser.add_argument('--emergency', action='store_true', 
                       help='Parada de emergência (força todos os processos)')
    
    args = parser.parse_args()
    
    stopper = SystemStopper()
    
    if args.emergency:
        stopper.emergency_stop()
        sys.exit(0)
    else:
        success = stopper.stop_all_services()
        sys.exit(0 if success else 1)