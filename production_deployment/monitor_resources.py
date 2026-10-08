#!/usr/bin/env python3
"""
Script de monitoramento de recursos do sistema para análise de estabilidade
"""

import psutil
import time
import datetime
import json
import os

def monitor_process(pid, duration_minutes=10, interval_seconds=30):
    """
    Monitora um processo específico por um período determinado
    """
    try:
        process = psutil.Process(pid)
        print(f"Monitorando processo: {process.name()} (PID: {pid})")
        print(f"Duração: {duration_minutes} minutos, Intervalo: {interval_seconds} segundos")
        print("-" * 80)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        measurements = []
        
        while time.time() < end_time:
            try:
                # Coleta métricas do processo
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()
                num_threads = process.num_threads()
                num_fds = process.num_handles() if hasattr(process, 'num_handles') else 0
                
                # Métricas do sistema
                system_memory = psutil.virtual_memory()
                system_cpu = psutil.cpu_percent()
                
                measurement = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'process': {
                        'pid': pid,
                        'name': process.name(),
                        'memory_rss_mb': round(memory_info.rss / 1024 / 1024, 2),
                        'memory_vms_mb': round(memory_info.vms / 1024 / 1024, 2),
                        'cpu_percent': cpu_percent,
                        'num_threads': num_threads,
                        'num_handles': num_fds,
                        'status': process.status()
                    },
                    'system': {
                        'memory_percent': system_memory.percent,
                        'memory_available_mb': round(system_memory.available / 1024 / 1024, 2),
                        'cpu_percent': system_cpu
                    }
                }
                
                measurements.append(measurement)
                
                # Exibe informações em tempo real
                print(f"{measurement['timestamp'][:19]} | "
                      f"RSS: {measurement['process']['memory_rss_mb']:>8.2f}MB | "
                      f"VMS: {measurement['process']['memory_vms_mb']:>8.2f}MB | "
                      f"CPU: {measurement['process']['cpu_percent']:>6.2f}% | "
                      f"Threads: {measurement['process']['num_threads']:>3d} | "
                      f"Handles: {measurement['process']['num_handles']:>4d}")
                
                # Verifica por possíveis problemas
                if measurement['process']['memory_rss_mb'] > 500:
                    print(f"⚠️  AVISO: Uso de memória RSS alto: {measurement['process']['memory_rss_mb']:.2f}MB")
                
                if measurement['process']['num_threads'] > 50:
                    print(f"⚠️  AVISO: Muitas threads: {measurement['process']['num_threads']}")
                
                if measurement['process']['num_handles'] > 1000:
                    print(f"⚠️  AVISO: Muitos handles: {measurement['process']['num_handles']}")
                
                time.sleep(interval_seconds)
                
            except psutil.NoSuchProcess:
                print(f"❌ Processo {pid} não encontrado")
                break
            except Exception as e:
                print(f"❌ Erro ao coletar métricas: {e}")
                time.sleep(interval_seconds)
        
        # Salva resultados
        output_file = f"resource_monitoring_{pid}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(measurements, f, indent=2)
        
        print(f"\n📊 Monitoramento concluído. Dados salvos em: {output_file}")
        
        # Análise básica
        if measurements:
            analyze_measurements(measurements)
        
    except psutil.NoSuchProcess:
        print(f"❌ Processo {pid} não encontrado")
    except Exception as e:
        print(f"❌ Erro: {e}")

def analyze_measurements(measurements):
    """
    Análise básica dos dados coletados
    """
    print("\n📈 ANÁLISE DE RECURSOS:")
    print("-" * 40)
    
    memory_values = [m['process']['memory_rss_mb'] for m in measurements]
    cpu_values = [m['process']['cpu_percent'] for m in measurements]
    thread_values = [m['process']['num_threads'] for m in measurements]
    handle_values = [m['process']['num_handles'] for m in measurements]
    
    print(f"Memória RSS (MB):")
    print(f"  Inicial: {memory_values[0]:.2f} | Final: {memory_values[-1]:.2f} | "
          f"Máximo: {max(memory_values):.2f} | Média: {sum(memory_values)/len(memory_values):.2f}")
    
    print(f"CPU (%):")
    print(f"  Máximo: {max(cpu_values):.2f} | Média: {sum(cpu_values)/len(cpu_values):.2f}")
    
    print(f"Threads:")
    print(f"  Inicial: {thread_values[0]} | Final: {thread_values[-1]} | Máximo: {max(thread_values)}")
    
    print(f"Handles:")
    print(f"  Inicial: {handle_values[0]} | Final: {handle_values[-1]} | Máximo: {max(handle_values)}")
    
    # Detecta vazamentos potenciais
    memory_growth = memory_values[-1] - memory_values[0]
    thread_growth = thread_values[-1] - thread_values[0]
    handle_growth = handle_values[-1] - handle_values[0]
    
    print(f"\n🔍 ANÁLISE DE VAZAMENTOS:")
    if memory_growth > 50:
        print(f"⚠️  Possível vazamento de memória: +{memory_growth:.2f}MB")
    else:
        print(f"✅ Memória estável: {memory_growth:+.2f}MB")
    
    if thread_growth > 5:
        print(f"⚠️  Possível vazamento de threads: +{thread_growth}")
    else:
        print(f"✅ Threads estáveis: {thread_growth:+d}")
    
    if handle_growth > 50:
        print(f"⚠️  Possível vazamento de handles: +{handle_growth}")
    else:
        print(f"✅ Handles estáveis: {handle_growth:+d}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python monitor_resources.py <PID> [duração_minutos] [intervalo_segundos]")
        print("\nProcessos Python ativos:")
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                if 'python' in proc.info['name'].lower():
                    memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                    print(f"  PID {proc.info['pid']}: {proc.info['name']} ({memory_mb:.1f}MB)")
            except:
                pass
        sys.exit(1)
    
    pid = int(sys.argv[1])
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    interval = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    monitor_process(pid, duration, interval)