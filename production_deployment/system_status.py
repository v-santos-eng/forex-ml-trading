#!/usr/bin/env python3
"""
Script de Verificação de Status do Sistema ML Forex
Verifica o status de todos os componentes críticos do sistema
"""

import psutil
import sqlite3
import json
import os
from datetime import datetime
import subprocess
import sys

class SystemStatusChecker:
    def __init__(self):
        self.db_path = "production_monitoring.db"
        self.config_path = "recovery_config.json"
        
    def check_process_by_pid(self, pid):
        """Verifica se um processo está rodando pelo PID"""
        try:
            process = psutil.Process(pid)
            return {
                'running': True,
                'name': process.name(),
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'status': process.status(),
                'create_time': datetime.fromtimestamp(process.create_time()).isoformat()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {'running': False}
    
    def check_database_health(self):
        """Verifica a saúde do banco de dados de monitoramento"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Verificar tabelas
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Contar registros recentes (última hora)
            cursor.execute("""
                SELECT COUNT(*) FROM system_metrics 
                WHERE timestamp > datetime('now', '-1 hour')
            """)
            recent_system_metrics = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM process_metrics 
                WHERE timestamp > datetime('now', '-1 hour')
            """)
            recent_process_metrics = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM alerts")
            total_alerts = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'healthy': True,
                'tables': tables,
                'recent_system_metrics': recent_system_metrics,
                'recent_process_metrics': recent_process_metrics,
                'total_alerts': total_alerts
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def check_system_resources(self):
        """Verifica recursos do sistema"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('C:\\')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / 1024**3,
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / 1024**3
        }
    
    def find_ml_pipeline_process(self):
        """Encontra o processo do pipeline ML"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'ml_production_pipeline.py' in cmdline:
                        return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def generate_status_report(self):
        """Gera relatório completo de status"""
        print("=" * 60)
        print(f"RELATÓRIO DE STATUS DO SISTEMA ML FOREX")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 60)
        
        # 1. Verificar Pipeline ML Principal
        print("\n1. PIPELINE ML PRINCIPAL:")
        ml_pid = self.find_ml_pipeline_process()
        if ml_pid:
            ml_status = self.check_process_by_pid(ml_pid)
            print(f"   ✅ RODANDO - PID: {ml_pid}")
            print(f"   📊 CPU: {ml_status['cpu_percent']:.1f}%")
            print(f"   💾 Memória: {ml_status['memory_mb']:.1f} MB")
            print(f"   ⏰ Iniciado: {ml_status['create_time']}")
        else:
            print("   ❌ NÃO ENCONTRADO")
        
        # 2. Verificar Monitoramento de Produção
        print("\n2. MONITORAMENTO DE PRODUÇÃO:")
        prod_monitor_running = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'production_monitor.py' in cmdline:
                        prod_monitor_running = True
                        print(f"   ✅ RODANDO - PID: {proc.info['pid']}")
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if not prod_monitor_running:
            print("   ❌ NÃO RODANDO")
        
        # 3. Verificar Sistema de Recuperação
        print("\n3. SISTEMA DE RECUPERAÇÃO AUTOMÁTICA:")
        recovery_running = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'auto_recovery_system.py' in cmdline:
                        recovery_running = True
                        print(f"   ✅ RODANDO - PID: {proc.info['pid']}")
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if not recovery_running:
            print("   ❌ NÃO RODANDO")
        
        # 4. Verificar Banco de Dados
        print("\n4. BANCO DE DADOS DE MONITORAMENTO:")
        db_status = self.check_database_health()
        if db_status['healthy']:
            print("   ✅ SAUDÁVEL")
            print(f"   📈 Métricas sistema (1h): {db_status['recent_system_metrics']}")
            print(f"   📊 Métricas processo (1h): {db_status['recent_process_metrics']}")
            print(f"   🚨 Total de alertas: {db_status['total_alerts']}")
        else:
            print(f"   ❌ ERRO: {db_status['error']}")
        
        # 5. Recursos do Sistema
        print("\n5. RECURSOS DO SISTEMA:")
        resources = self.check_system_resources()
        print(f"   🖥️  CPU: {resources['cpu_percent']:.1f}%")
        print(f"   💾 Memória: {resources['memory_percent']:.1f}% (Livre: {resources['memory_available_gb']:.1f} GB)")
        print(f"   💿 Disco: {resources['disk_percent']:.1f}% (Livre: {resources['disk_free_gb']:.1f} GB)")
        
        # 6. Status Geral
        print("\n6. STATUS GERAL:")
        all_systems_ok = (
            ml_pid is not None and
            prod_monitor_running and
            recovery_running and
            db_status['healthy'] and
            resources['cpu_percent'] < 80 and
            resources['memory_percent'] < 85 and
            resources['disk_percent'] < 90
        )
        
        if all_systems_ok:
            print("   ✅ SISTEMA TOTALMENTE OPERACIONAL")
        else:
            print("   ⚠️  SISTEMA COM PROBLEMAS - VERIFICAR COMPONENTES ACIMA")
        
        print("\n" + "=" * 60)
        return all_systems_ok

if __name__ == "__main__":
    checker = SystemStatusChecker()
    status_ok = checker.generate_status_report()
    sys.exit(0 if status_ok else 1)