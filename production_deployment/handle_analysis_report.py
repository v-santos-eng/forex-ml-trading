#!/usr/bin/env python3
"""
Script para análise detalhada dos dados de handles coletados
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def analyze_handle_data(json_file):
    """
    Analisa os dados de handles coletados e gera relatório detalhado
    """
    print("📊 ANÁLISE DETALHADA DE HANDLES")
    print("=" * 80)
    
    # Carregar dados
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extrair informações básicas
    analysis = data['analysis']
    data_points = data['data_points']
    
    print(f"📈 Resumo da Análise:")
    print(f"   • Total de handles: {analysis['total_growth']} (crescimento líquido)")
    print(f"   • Arquivos novos: {analysis['file_growth']}")
    print(f"   • Conexões novas: {analysis['connection_growth']}")
    print(f"   • Threads novas: {analysis['thread_growth']}")
    print(f"   • Memória: {analysis['memory_growth_mb']:.2f}MB")
    print()
    
    # Análise temporal dos handles
    timestamps = []
    handles = []
    memory_rss = []
    
    for point in data_points:
        timestamps.append(datetime.fromisoformat(point['timestamp']))
        handles.append(point['num_handles'])
        memory_rss.append(point['memory_info']['rss'] / 1024 / 1024)  # MB
    
    # Converter para DataFrame para análise
    df = pd.DataFrame({
        'timestamp': timestamps,
        'handles': handles,
        'memory_mb': memory_rss
    })
    
    print(f"📊 Estatísticas dos Handles:")
    print(f"   • Mínimo: {df['handles'].min()}")
    print(f"   • Máximo: {df['handles'].max()}")
    print(f"   • Média: {df['handles'].mean():.1f}")
    print(f"   • Desvio padrão: {df['handles'].std():.1f}")
    print()
    
    # Detectar eventos de limpeza
    handle_diff = df['handles'].diff()
    cleanup_events = handle_diff[handle_diff < -50]  # Quedas significativas
    
    if len(cleanup_events) > 0:
        print(f"🧹 Eventos de Limpeza Detectados: {len(cleanup_events)}")
        for idx, drop in cleanup_events.items():
            timestamp = df.loc[idx, 'timestamp']
            print(f"   • {timestamp.strftime('%H:%M:%S')}: {drop:.0f} handles liberados")
        print()
    
    # Análise de crescimento por período
    growth_rates = []
    for i in range(1, len(handles)):
        if i > 0:
            time_diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 60  # minutos
            handle_diff = handles[i] - handles[i-1]
            if time_diff > 0:
                growth_rate = handle_diff / time_diff  # handles por minuto
                growth_rates.append(growth_rate)
    
    if growth_rates:
        avg_growth_rate = np.mean(growth_rates)
        print(f"📈 Taxa de Crescimento:")
        print(f"   • Média: {avg_growth_rate:.2f} handles/minuto")
        print(f"   • Máxima: {max(growth_rates):.2f} handles/minuto")
        print(f"   • Mínima: {min(growth_rates):.2f} handles/minuto")
        print()
    
    # Análise de correlação memória vs handles
    correlation = df['handles'].corr(df['memory_mb'])
    print(f"🔗 Correlação Handles vs Memória: {correlation:.3f}")
    if correlation > 0.7:
        print("   ⚠️  Correlação forte - possível vazamento conjunto")
    elif correlation > 0.3:
        print("   ⚡ Correlação moderada - crescimento relacionado")
    else:
        print("   ✅ Correlação baixa - crescimentos independentes")
    print()
    
    # Arquivos persistentes
    persistent_files = analysis.get('persistent_files', [])
    print(f"📁 Arquivos Persistentemente Abertos: {len(persistent_files)}")
    for file_path in persistent_files:
        print(f"   • {file_path}")
    print()
    
    # Recomendações
    print("🔧 RECOMENDAÇÕES:")
    
    if analysis['total_growth'] < 0:
        print("   ✅ Sistema possui mecanismo de limpeza automática eficiente")
    elif analysis['total_growth'] > 100:
        print("   ⚠️  Crescimento significativo de handles - investigar")
    else:
        print("   ✅ Crescimento de handles dentro do esperado")
    
    if len(cleanup_events) > 0:
        print("   ✅ Eventos de limpeza automática detectados")
    
    if correlation > 0.7:
        print("   ⚠️  Investigar vazamento conjunto de memória e handles")
    
    if len(persistent_files) > 20:
        print("   ⚠️  Muitos arquivos persistentemente abertos")
    
    # Salvar gráfico se matplotlib disponível
    try:
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Handles ao longo do tempo
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, handles, 'b-', linewidth=2, label='Handles')
        plt.title('Evolução dos Handles ao Longo do Tempo')
        plt.ylabel('Número de Handles')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Subplot 2: Memória ao longo do tempo
        plt.subplot(2, 1, 2)
        plt.plot(timestamps, memory_rss, 'r-', linewidth=2, label='Memória RSS')
        plt.title('Evolução da Memória ao Longo do Tempo')
        plt.xlabel('Tempo')
        plt.ylabel('Memória (MB)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Salvar gráfico
        graph_file = json_file.replace('.json', '_graph.png')
        plt.savefig(graph_file, dpi=300, bbox_inches='tight')
        print(f"\n📊 Gráfico salvo em: {graph_file}")
        
    except ImportError:
        print("\n📊 Matplotlib não disponível - gráfico não gerado")
    
    return df, analysis

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Uso: python handle_analysis_report.py <arquivo_json>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        df, analysis = analyze_handle_data(json_file)
        print("\n✅ Análise concluída com sucesso!")
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        sys.exit(1)