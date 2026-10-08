#!/usr/bin/env python3
"""
Analisador de Vazamento de Handles
Investiga quais tipos de handles estão sendo criados e não liberados
"""

import psutil
import time
import json
from datetime import datetime
from collections import defaultdict
import sys

def get_process_handles_info(pid):
    """Obtém informações detalhadas sobre handles do processo"""
    try:
        process = psutil.Process(pid)
        
        # Informações básicas
        info = {
            'pid': pid,
            'name': process.name(),
            'num_handles': process.num_handles(),
            'num_threads': process.num_threads(),
            'memory_info': process.memory_info()._asdict(),
            'open_files': [],
            'connections': [],
            'threads': []
        }
        
        # Arquivos abertos
        try:
            open_files = process.open_files()
            info['open_files'] = [{'path': f.path, 'fd': f.fd} for f in open_files]
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            info['open_files'] = []
        
        # Conexões de rede
        try:
            connections = process.connections()
            info['connections'] = [{
                'fd': conn.fd,
                'family': str(conn.family),
                'type': str(conn.type),
                'laddr': conn.laddr,
                'raddr': conn.raddr,
                'status': conn.status
            } for conn in connections]
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            info['connections'] = []
        
        # Threads
        try:
            threads = process.threads()
            info['threads'] = [{
                'id': thread.id,
                'user_time': thread.user_time,
                'system_time': thread.system_time
            } for thread in threads]
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            info['threads'] = []
        
        return info
        
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        return {'error': str(e)}

def analyze_handle_growth(data_points):
    """Analisa o crescimento de handles ao longo do tempo"""
    if len(data_points) < 2:
        return {}
    
    analysis = {
        'total_growth': data_points[-1]['num_handles'] - data_points[0]['num_handles'],
        'file_growth': len(data_points[-1]['open_files']) - len(data_points[0]['open_files']),
        'connection_growth': len(data_points[-1]['connections']) - len(data_points[0]['connections']),
        'thread_growth': len(data_points[-1]['threads']) - len(data_points[0]['threads']),
        'memory_growth_mb': (data_points[-1]['memory_info']['rss'] - data_points[0]['memory_info']['rss']) / 1024 / 1024
    }
    
    # Análise de arquivos que permanecem abertos
    initial_files = {f['path'] for f in data_points[0]['open_files']}
    final_files = {f['path'] for f in data_points[-1]['open_files']}
    
    analysis['new_files'] = list(final_files - initial_files)
    analysis['persistent_files'] = list(initial_files & final_files)
    
    # Análise de conexões
    initial_connections = len(data_points[0]['connections'])
    final_connections = len(data_points[-1]['connections'])
    
    analysis['connection_types'] = defaultdict(int)
    for conn in data_points[-1]['connections']:
        analysis['connection_types'][f"{conn['family']}_{conn['type']}"] += 1
    
    return analysis

def monitor_handles(pid, duration_minutes=10, interval_seconds=30):
    """Monitora handles por um período específico"""
    print(f"🔍 Analisando vazamento de handles para PID {pid}")
    print(f"⏱️  Duração: {duration_minutes} minutos, Intervalo: {interval_seconds} segundos")
    print("-" * 80)
    
    data_points = []
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    while time.time() < end_time:
        timestamp = datetime.now().isoformat()
        handle_info = get_process_handles_info(pid)
        
        if 'error' in handle_info:
            print(f"❌ Erro ao obter informações: {handle_info['error']}")
            break
        
        data_points.append({
            'timestamp': timestamp,
            **handle_info
        })
        
        print(f"{timestamp[:19]} | Handles: {handle_info['num_handles']:4d} | "
              f"Files: {len(handle_info['open_files']):3d} | "
              f"Connections: {len(handle_info['connections']):3d} | "
              f"Threads: {len(handle_info['threads']):2d} | "
              f"Memory: {handle_info['memory_info']['rss']/1024/1024:.1f}MB")
        
        time.sleep(interval_seconds)
    
    # Análise final
    if len(data_points) >= 2:
        analysis = analyze_handle_growth(data_points)
        
        print("\n" + "=" * 80)
        print("📊 ANÁLISE DE VAZAMENTO DE HANDLES")
        print("=" * 80)
        
        print(f"🔢 Crescimento total de handles: {analysis['total_growth']}")
        print(f"📁 Novos arquivos abertos: {analysis['file_growth']}")
        print(f"🌐 Novas conexões: {analysis['connection_growth']}")
        print(f"🧵 Novas threads: {analysis['thread_growth']}")
        print(f"💾 Crescimento de memória: {analysis['memory_growth_mb']:.2f}MB")
        
        if analysis['new_files']:
            print(f"\n📂 Novos arquivos detectados ({len(analysis['new_files'])}):")
            for file_path in analysis['new_files'][:10]:  # Mostrar apenas os primeiros 10
                print(f"  - {file_path}")
            if len(analysis['new_files']) > 10:
                print(f"  ... e mais {len(analysis['new_files']) - 10} arquivos")
        
        if analysis['connection_types']:
            print(f"\n🌐 Tipos de conexão ativas:")
            for conn_type, count in analysis['connection_types'].items():
                print(f"  - {conn_type}: {count}")
        
        # Salvar dados detalhados
        filename = f"handle_analysis_{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis': analysis,
                'data_points': data_points
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Dados detalhados salvos em: {filename}")
        
        # Recomendações
        print("\n🔧 RECOMENDAÇÕES:")
        if analysis['total_growth'] > 50:
            print("⚠️  Vazamento significativo de handles detectado!")
            if analysis['file_growth'] > 10:
                print("  - Verificar se arquivos estão sendo fechados adequadamente")
            if analysis['connection_growth'] > 5:
                print("  - Verificar se conexões de rede estão sendo fechadas")
            if analysis['thread_growth'] > 2:
                print("  - Verificar se threads estão sendo finalizadas corretamente")
        else:
            print("✅ Crescimento de handles dentro do esperado")
    
    return data_points

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python handle_leak_analyzer.py <PID> [duração_minutos] [intervalo_segundos]")
        sys.exit(1)
    
    pid = int(sys.argv[1])
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    interval = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    try:
        monitor_handles(pid, duration, interval)
    except KeyboardInterrupt:
        print("\n⏹️  Monitoramento interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro durante monitoramento: {e}")