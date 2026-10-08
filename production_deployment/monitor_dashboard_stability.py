#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor de estabilidade do dashboard
Verifica se o dashboard permanece ativo e detecta fechamentos inesperados
"""

import psutil
import time
import sys
from datetime import datetime, timedelta

def find_dashboard_processes():
    """Encontra processos do dashboard"""
    dashboard_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'dashboard' in cmdline.lower():
                    dashboard_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline,
                        'create_time': proc.info['create_time'],
                        'process': proc
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return dashboard_processes

def format_uptime(create_time):
    """Formata tempo de execução"""
    uptime = datetime.now() - datetime.fromtimestamp(create_time)
    
    hours = int(uptime.total_seconds() // 3600)
    minutes = int((uptime.total_seconds() % 3600) // 60)
    seconds = int(uptime.total_seconds() % 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def monitor_dashboard(check_interval=10, max_duration=300):
    """Monitora estabilidade do dashboard"""
    print("=== Monitor de Estabilidade do Dashboard ===")
    print(f"Intervalo de verificação: {check_interval}s")
    print(f"Duração máxima: {max_duration}s ({max_duration//60} minutos)")
    print("\nIniciando monitoramento...\n")
    
    start_time = time.time()
    last_processes = {}
    check_count = 0
    
    while time.time() - start_time < max_duration:
        check_count += 1
        current_time = datetime.now().strftime("%H:%M:%S")
        
        print(f"[{current_time}] Verificação #{check_count}")
        
        # Encontrar processos do dashboard
        current_processes = find_dashboard_processes()
        
        if not current_processes:
            print("  ❌ ALERTA: Nenhum processo do dashboard encontrado!")
        else:
            print(f"  ✅ {len(current_processes)} processo(s) do dashboard ativo(s):")
            
            for proc_info in current_processes:
                pid = proc_info['pid']
                uptime = format_uptime(proc_info['create_time'])
                
                try:
                    # Obter informações de recursos
                    proc = proc_info['process']
                    cpu_percent = proc.cpu_percent()
                    memory_mb = proc.memory_info().rss / (1024 * 1024)
                    
                    print(f"    PID: {pid} | Uptime: {uptime} | CPU: {cpu_percent:.1f}% | RAM: {memory_mb:.1f}MB")
                    
                    # Verificar se é um processo novo (reinicialização)
                    if pid in last_processes:
                        if proc_info['create_time'] != last_processes[pid]['create_time']:
                            print(f"    ⚠️  REINICIALIZAÇÃO DETECTADA para PID {pid}")
                    else:
                        if last_processes:  # Não é a primeira verificação
                            print(f"    🆕 NOVO PROCESSO detectado: PID {pid}")
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    print(f"    ❌ Erro ao acessar processo {pid}: {e}")
        
        # Verificar processos que desapareceram
        current_pids = {proc['pid'] for proc in current_processes}
        disappeared_pids = set(last_processes.keys()) - current_pids
        
        if disappeared_pids:
            for pid in disappeared_pids:
                uptime = format_uptime(last_processes[pid]['create_time'])
                print(f"  💀 PROCESSO ENCERRADO: PID {pid} (uptime: {uptime})")
        
        # Atualizar lista de processos
        last_processes = {proc['pid']: proc for proc in current_processes}
        
        print()  # Linha em branco
        
        # Aguardar próxima verificação
        time.sleep(check_interval)
    
    # Resumo final
    total_time = time.time() - start_time
    print(f"\n=== RESUMO DO MONITORAMENTO ===")
    print(f"Duração total: {total_time:.1f}s ({total_time/60:.1f} minutos)")
    print(f"Verificações realizadas: {check_count}")
    
    final_processes = find_dashboard_processes()
    if final_processes:
        print(f"✅ Dashboard ainda ativo: {len(final_processes)} processo(s)")
        for proc in final_processes:
            uptime = format_uptime(proc['create_time'])
            print(f"  PID {proc['pid']}: uptime {uptime}")
    else:
        print("❌ Dashboard não está mais ativo")

def main():
    """Função principal"""
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            print("Uso: python monitor_dashboard_stability.py [duração_em_segundos]")
            return 1
    else:
        duration = 300  # 5 minutos por padrão
    
    try:
        monitor_dashboard(check_interval=10, max_duration=duration)
        return 0
    except KeyboardInterrupt:
        print("\n⚠️ Monitoramento interrompido pelo usuário")
        return 0
    except Exception as e:
        print(f"❌ Erro durante monitoramento: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)