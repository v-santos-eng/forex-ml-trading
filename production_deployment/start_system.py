#!/usr/bin/env python3
"""
Script de Inicialização do Sistema ML Forex
Inicia todos os componentes do sistema na ordem correta
"""

import subprocess
import time
import os
import sys
import psutil
from datetime import datetime

class SystemStarter:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.ml_dir = os.path.dirname(self.base_dir)
        
    def log(self, message):
        """Log com timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def check_process_running(self, script_name):
        """Verifica se um processo Python específico está rodando"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if script_name in cmdline:
                        return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def start_ml_pipeline(self):
        """Inicia o pipeline ML principal"""
        self.log("Verificando Pipeline ML...")
        pid = self.check_process_running('ml_production_pipeline.py')
        
        if pid:
            self.log(f"Pipeline ML já está rodando (PID: {pid})")
            return pid
        
        self.log("Iniciando Pipeline ML...")
        try:
            # Muda para o diretório do ML e inicia o pipeline
            process = subprocess.Popen(
                [sys.executable, 'ml_production_pipeline.py'],
                cwd=self.ml_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Aguarda um pouco para verificar se iniciou corretamente
            time.sleep(3)
            
            if process.poll() is None:  # Processo ainda rodando
                pid = self.check_process_running('ml_production_pipeline.py')
                if pid:
                    self.log(f"✅ Pipeline ML iniciado com sucesso (PID: {pid})")
                    return pid
            
            self.log("❌ Falha ao iniciar Pipeline ML")
            return None
            
        except Exception as e:
            self.log(f"❌ Erro ao iniciar Pipeline ML: {e}")
            return None
    
    def start_production_monitor(self, ml_pid):
        """Inicia o monitoramento de produção"""
        self.log("Verificando Monitoramento de Produção...")
        pid = self.check_process_running('production_monitor.py')
        
        if pid:
            self.log(f"Monitoramento de Produção já está rodando (PID: {pid})")
            return pid
        
        if not ml_pid:
            self.log("❌ Não é possível iniciar monitoramento sem o Pipeline ML")
            return None
        
        self.log(f"Iniciando Monitoramento de Produção para PID {ml_pid}...")
        try:
            process = subprocess.Popen(
                [sys.executable, 'production_monitor.py', 
                 '--add-process', 'ML_Pipeline', str(ml_pid),
                 '--interval', '20', '--daemon'],
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            time.sleep(3)
            
            pid = self.check_process_running('production_monitor.py')
            if pid:
                self.log(f"✅ Monitoramento de Produção iniciado (PID: {pid})")
                return pid
            
            self.log("❌ Falha ao iniciar Monitoramento de Produção")
            return None
            
        except Exception as e:
            self.log(f"❌ Erro ao iniciar Monitoramento de Produção: {e}")
            return None
    
    def start_recovery_system(self):
        """Inicia o sistema de recuperação automática"""
        self.log("Verificando Sistema de Recuperação...")
        pid = self.check_process_running('auto_recovery_system.py')
        
        if pid:
            self.log(f"Sistema de Recuperação já está rodando (PID: {pid})")
            return pid
        
        self.log("Iniciando Sistema de Recuperação Automática...")
        try:
            process = subprocess.Popen(
                [sys.executable, 'auto_recovery_system.py',
                 '--config', 'recovery_config.json', '--monitor'],
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            time.sleep(3)
            
            pid = self.check_process_running('auto_recovery_system.py')
            if pid:
                self.log(f"✅ Sistema de Recuperação iniciado (PID: {pid})")
                return pid
            
            self.log("❌ Falha ao iniciar Sistema de Recuperação")
            return None
            
        except Exception as e:
            self.log(f"❌ Erro ao iniciar Sistema de Recuperação: {e}")
            return None
    
    def verify_system_health(self):
        """Verifica a saúde geral do sistema"""
        self.log("Verificando saúde do sistema...")
        
        # Executa o script de status
        try:
            result = subprocess.run(
                [sys.executable, 'system_status.py'],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.log("✅ Sistema totalmente operacional")
                return True
            else:
                self.log("⚠️ Sistema com problemas detectados")
                return False
                
        except Exception as e:
            self.log(f"❌ Erro ao verificar saúde do sistema: {e}")
            return False
    
    def start_all_services(self):
        """Inicia todos os serviços na ordem correta"""
        self.log("=" * 60)
        self.log("INICIANDO SISTEMA ML FOREX")
        self.log("=" * 60)
        
        # 1. Iniciar Pipeline ML
        ml_pid = self.start_ml_pipeline()
        if not ml_pid:
            self.log("❌ FALHA CRÍTICA: Não foi possível iniciar o Pipeline ML")
            return False
        
        # 2. Iniciar Monitoramento de Produção
        monitor_pid = self.start_production_monitor(ml_pid)
        if not monitor_pid:
            self.log("⚠️ AVISO: Monitoramento de Produção não iniciado")
        
        # 3. Iniciar Sistema de Recuperação
        recovery_pid = self.start_recovery_system()
        if not recovery_pid:
            self.log("⚠️ AVISO: Sistema de Recuperação não iniciado")
        
        # 4. Aguardar estabilização
        self.log("Aguardando estabilização do sistema...")
        time.sleep(10)
        
        # 5. Verificar saúde final
        system_healthy = self.verify_system_health()
        
        self.log("=" * 60)
        if system_healthy:
            self.log("🎉 SISTEMA INICIADO COM SUCESSO!")
            self.log("Todos os componentes estão operacionais.")
        else:
            self.log("⚠️ SISTEMA INICIADO COM PROBLEMAS")
            self.log("Verifique os logs para mais detalhes.")
        
        self.log("=" * 60)
        
        return system_healthy

if __name__ == "__main__":
    starter = SystemStarter()
    success = starter.start_all_services()
    sys.exit(0 if success else 1)