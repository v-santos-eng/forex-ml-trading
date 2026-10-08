#!/usr/bin/env python3
"""
Script para investigar a correlação entre handles e memória
"""

import psutil
import time
import json
from datetime import datetime
import sys

def analyze_memory_handle_correlation(pid, duration_minutes=10, interval_seconds=30):
    """
    Analisa a correlação entre handles e memória em tempo real
    """
    print(f"🔍 ANÁLISE DE CORRELAÇÃO MEMÓRIA-HANDLES")
    print(f"📊 PID: {pid} | Duração: {duration_minutes}min | Intervalo: {interval_seconds}s")
    print("=" * 80)
    
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"❌ Processo {pid} não encontrado")
        return
    
    data_points = []
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    iteration = 0
    
    while time.time() < end_time:
        try:
            # Coletar dados detalhados
            memory_info = process.memory_info()
            memory_full = process.memory_full_info()
            
            # Handles (Windows específico)
            try:
                num_handles = process.num_handles()
            except AttributeError:
                num_handles = len(process.open_files()) + len(process.connections())
            
            # Informações adicionais
            open_files = process.open_files()
            try:
                connections = process.net_connections()
            except (psutil.AccessDenied, AttributeError):
                connections = []
            threads = process.threads()
            
            timestamp = datetime.now()
            
            data_point = {
                'timestamp': timestamp.isoformat(),
                'iteration': iteration,
                'handles': num_handles,
                'memory': {
                    'rss': memory_info.rss,
                    'vms': memory_info.vms,
                    'rss_mb': memory_info.rss / 1024 / 1024,
                    'vms_mb': memory_info.vms / 1024 / 1024
                },
                'memory_full': {
                    'uss': getattr(memory_full, 'uss', 0),
                    'pss': getattr(memory_full, 'pss', 0),
                    'swap': getattr(memory_full, 'swap', 0)
                },
                'resources': {
                    'open_files': len(open_files),
                    'connections': len(connections),
                    'threads': len(threads)
                }
            }
            
            data_points.append(data_point)
            
            # Log em tempo real
            print(f"{timestamp.strftime('%H:%M:%S')} | "
                  f"Handles: {num_handles:4d} | "
                  f"RSS: {memory_info.rss/1024/1024:6.1f}MB | "
                  f"VMS: {memory_info.vms/1024/1024:6.1f}MB | "
                  f"Files: {len(open_files):2d} | "
                  f"Conn: {len(connections):2d}")
            
            iteration += 1
            time.sleep(interval_seconds)
            
        except psutil.NoSuchProcess:
            print(f"❌ Processo {pid} terminou durante a análise")
            break
        except Exception as e:
            print(f"⚠️  Erro coletando dados: {e}")
            time.sleep(interval_seconds)
            continue
    
    # Análise dos dados coletados
    if len(data_points) < 2:
        print("❌ Dados insuficientes para análise")
        return
    
    print("\n" + "=" * 80)
    print("📊 ANÁLISE DE CORRELAÇÃO")
    print("=" * 80)
    
    # Extrair séries temporais
    handles_series = [dp['handles'] for dp in data_points]
    rss_series = [dp['memory']['rss_mb'] for dp in data_points]
    vms_series = [dp['memory']['vms_mb'] for dp in data_points]
    
    # Calcular variações
    handle_changes = [handles_series[i] - handles_series[i-1] for i in range(1, len(handles_series))]
    rss_changes = [rss_series[i] - rss_series[i-1] for i in range(1, len(rss_series))]
    vms_changes = [vms_series[i] - vms_series[i-1] for i in range(1, len(vms_series))]
    
    # Estatísticas básicas
    print(f"📈 Variações Observadas:")
    print(f"   • Handles: {min(handle_changes):+.0f} a {max(handle_changes):+.0f} (total: {handles_series[-1] - handles_series[0]:+.0f})")
    print(f"   • RSS: {min(rss_changes):+.2f}MB a {max(rss_changes):+.2f}MB (total: {rss_series[-1] - rss_series[0]:+.2f}MB)")
    print(f"   • VMS: {min(vms_changes):+.2f}MB a {max(vms_changes):+.2f}MB (total: {vms_series[-1] - vms_series[0]:+.2f}MB)")
    print()
    
    # Análise de correlação simples
    def simple_correlation(x, y):
        if len(x) != len(y) or len(x) < 2:
            return 0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    corr_handles_rss = simple_correlation(handles_series, rss_series)
    corr_handles_vms = simple_correlation(handles_series, vms_series)
    corr_changes_handles_rss = simple_correlation(handle_changes, rss_changes)
    
    print(f"🔗 Correlações:")
    print(f"   • Handles vs RSS: {corr_handles_rss:.3f}")
    print(f"   • Handles vs VMS: {corr_handles_vms:.3f}")
    print(f"   • Δ Handles vs Δ RSS: {corr_changes_handles_rss:.3f}")
    print()
    
    # Detectar padrões suspeitos
    suspicious_patterns = []
    
    if abs(corr_handles_rss) > 0.9:
        suspicious_patterns.append("Correlação muito forte entre handles e RSS")
    
    if abs(corr_changes_handles_rss) > 0.7:
        suspicious_patterns.append("Variações de handles e memória altamente correlacionadas")
    
    # Detectar crescimento simultâneo
    simultaneous_growth = 0
    for i in range(len(handle_changes)):
        if handle_changes[i] > 0 and rss_changes[i] > 0:
            simultaneous_growth += 1
    
    if simultaneous_growth > len(handle_changes) * 0.7:
        suspicious_patterns.append(f"Crescimento simultâneo em {simultaneous_growth}/{len(handle_changes)} observações")
    
    # Salvar dados detalhados
    output_file = f"memory_handle_correlation_{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    analysis_data = {
        'analysis': {
            'pid': pid,
            'duration_minutes': duration_minutes,
            'interval_seconds': interval_seconds,
            'total_observations': len(data_points),
            'correlations': {
                'handles_rss': corr_handles_rss,
                'handles_vms': corr_handles_vms,
                'changes_handles_rss': corr_changes_handles_rss
            },
            'variations': {
                'handles': {
                    'min': min(handle_changes) if handle_changes else 0,
                    'max': max(handle_changes) if handle_changes else 0,
                    'total': handles_series[-1] - handles_series[0]
                },
                'rss_mb': {
                    'min': min(rss_changes) if rss_changes else 0,
                    'max': max(rss_changes) if rss_changes else 0,
                    'total': rss_series[-1] - rss_series[0]
                }
            },
            'suspicious_patterns': suspicious_patterns,
            'simultaneous_growth_ratio': simultaneous_growth / len(handle_changes) if handle_changes else 0
        },
        'data_points': data_points
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Dados salvos em: {output_file}")
    print()
    
    # Recomendações
    print("🔧 RECOMENDAÇÕES:")
    
    if suspicious_patterns:
        print("   ⚠️  PADRÕES SUSPEITOS DETECTADOS:")
        for pattern in suspicious_patterns:
            print(f"      • {pattern}")
        print("   🔍 Investigação adicional recomendada")
    else:
        print("   ✅ Nenhum padrão suspeito detectado")
    
    if abs(corr_handles_rss) > 0.9:
        print("   🔍 Investigar se handles e memória estão vazando juntos")
        print("   📋 Verificar se objetos não estão sendo liberados corretamente")
    
    return analysis_data

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python memory_handle_correlation_analyzer.py <PID> [duração_minutos] [intervalo_segundos]")
        sys.exit(1)
    
    pid = int(sys.argv[1])
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    interval = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    try:
        analyze_memory_handle_correlation(pid, duration, interval)
    except KeyboardInterrupt:
        print("\n⏹️  Análise interrompida pelo usuário")
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        sys.exit(1)