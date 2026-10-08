#!/usr/bin/env python3
"""
Análise Específica dos Dados FX-1-Minute-Data
Este script analisa os dados disponíveis no diretório FX-1-Minute-Data
e fornece recomendações otimizadas para treinamento de ML.
"""

import os
import zipfile
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import numpy as np

class FXDataAnalyzer:
    def __init__(self, data_directory):
        self.data_directory = Path(data_directory)
        self.analysis_results = {}
        
    def scan_available_pairs(self):
        """Escaneia todos os pares disponíveis no diretório"""
        pairs = []
        if self.data_directory.exists():
            for item in self.data_directory.iterdir():
                if item.is_dir():
                    # Verifica se há arquivos ZIP no diretório
                    zip_files = list(item.glob('*.zip'))
                    if zip_files:
                        pairs.append(item.name.upper())
        return sorted(pairs)
    
    def analyze_pair_data(self, pair):
        """Analisa os dados disponíveis para um par específico"""
        pair_dir = self.data_directory / pair.lower()
        if not pair_dir.exists():
            return None
            
        zip_files = list(pair_dir.glob('*.zip'))
        years = []
        total_estimated_records = 0
        
        for zip_file in zip_files:
            # Extrai o ano do nome do arquivo
            year = zip_file.stem.split('_')[-1]
            years.append(int(year))
            
            # Estima registros baseado no tamanho do arquivo
            file_size_mb = zip_file.stat().st_size / (1024 * 1024)
            # Aproximadamente 50-100 registros por MB para dados de 1 minuto compactados
            estimated_records = int(file_size_mb * 75 * 1000)  # Estimativa conservadora
            total_estimated_records += estimated_records
        
        return {
            'pair': pair,
            'years_available': sorted(years),
            'year_range': f"{min(years)}-{max(years)}" if years else "N/A",
            'total_files': len(zip_files),
            'estimated_total_records': total_estimated_records,
            'estimated_size_mb': sum(f.stat().st_size for f in zip_files) / (1024 * 1024)
        }
    
    def get_sample_data(self, pair, year=None):
        """Obtém uma amostra dos dados para análise"""
        pair_dir = self.data_directory / pair.lower()
        if not pair_dir.exists():
            return None
            
        # Pega o arquivo mais recente se year não especificado
        zip_files = list(pair_dir.glob('*.zip'))
        if not zip_files:
            return None
            
        if year:
            target_file = next((f for f in zip_files if str(year) in f.name), None)
        else:
            target_file = max(zip_files, key=lambda x: x.stat().st_mtime)
            
        if not target_file:
            return None
            
        try:
            with zipfile.ZipFile(target_file, 'r') as zip_ref:
                # Lista arquivos no ZIP
                file_list = zip_ref.namelist()
                if not file_list:
                    return None
                    
                # Lê o primeiro arquivo CSV
                csv_file = file_list[0]
                with zip_ref.open(csv_file) as csv_data:
                    # Lê apenas as primeiras 1000 linhas para análise
                    df = pd.read_csv(csv_data, nrows=1000, header=None)
                    
                    # Assume formato padrão: Date, Time, Open, High, Low, Close, Volume
                    if len(df.columns) >= 6:
                        df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close'] + [f'Col_{i}' for i in range(6, len(df.columns))]
                        return df
                        
        except Exception as e:
            print(f"Erro ao ler {target_file}: {e}")
            return None
    
    def calculate_data_recommendations(self, available_pairs):
        """Calcula recomendações baseadas nos dados disponíveis"""
        total_pairs = len(available_pairs)
        
        # Análise dos dados disponíveis
        pair_analysis = {}
        total_estimated_records = 0
        
        for pair in available_pairs[:10]:  # Analisa os primeiros 10 pares
            analysis = self.analyze_pair_data(pair)
            if analysis:
                pair_analysis[pair] = analysis
                total_estimated_records += analysis['estimated_total_records']
        
        avg_records_per_pair = total_estimated_records / len(pair_analysis) if pair_analysis else 0
        
        # Recomendações baseadas na quantidade de dados
        recommendations = {
            'desenvolvimento_rapido': {
                'samples_per_pair': min(50000, int(avg_records_per_pair * 0.1)),
                'max_pairs': 3,
                'estimated_time_minutes': 5,
                'memory_gb': 1,
                'description': 'Ideal para testes rápidos e desenvolvimento'
            },
            'producao_balanceada': {
                'samples_per_pair': min(200000, int(avg_records_per_pair * 0.3)),
                'max_pairs': 8,
                'estimated_time_minutes': 25,
                'memory_gb': 4,
                'description': 'Equilibrio entre performance e robustez'
            },
            'pesquisa_robusta': {
                'samples_per_pair': min(500000, int(avg_records_per_pair * 0.6)),
                'max_pairs': 15,
                'estimated_time_minutes': 60,
                'memory_gb': 8,
                'description': 'Máxima robustez para pesquisa avançada'
            }
        }
        
        return {
            'total_pairs_available': total_pairs,
            'analyzed_pairs': len(pair_analysis),
            'avg_records_per_pair': int(avg_records_per_pair),
            'pair_details': pair_analysis,
            'recommendations': recommendations
        }
    
    def generate_report(self):
        """Gera relatório completo da análise"""
        print("🔍 Analisando Dados FX-1-Minute-Data")
        print("=" * 50)
        
        # Escaneia pares disponíveis
        available_pairs = self.scan_available_pairs()
        print(f"📊 Pares encontrados: {len(available_pairs)}")
        
        if not available_pairs:
            print("❌ Nenhum par de dados encontrado!")
            return
        
        print(f"💱 Pares disponíveis: {', '.join(available_pairs[:10])}{'...' if len(available_pairs) > 10 else ''}")
        
        # Calcula recomendações
        analysis = self.calculate_data_recommendations(available_pairs)
        
        print(f"\n📈 Estatísticas dos Dados:")
        print(f"  Total de pares: {analysis['total_pairs_available']}")
        print(f"  Pares analisados: {analysis['analyzed_pairs']}")
        print(f"  Média de registros por par: {analysis['avg_records_per_pair']:,}")
        
        print(f"\n🎯 Recomendações de Otimização:")
        for scenario, config in analysis['recommendations'].items():
            print(f"\n--- {scenario.replace('_', ' ').title()} ---")
            print(f"📊 Amostras por par: {config['samples_per_pair']:,}")
            print(f"🔢 Máximo de pares: {config['max_pairs']}")
            print(f"⏱️ Tempo estimado: {config['estimated_time_minutes']} minutos")
            print(f"💾 Memória estimada: {config['memory_gb']}GB")
            print(f"📝 {config['description']}")
        
        # Análise detalhada de alguns pares principais
        major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
        available_major = [p for p in major_pairs if p in available_pairs]
        
        if available_major:
            print(f"\n💎 Análise Detalhada dos Pares Principais:")
            for pair in available_major[:3]:
                if pair in analysis['pair_details']:
                    details = analysis['pair_details'][pair]
                    print(f"\n{pair}:")
                    print(f"  📅 Anos: {details['year_range']}")
                    print(f"  📁 Arquivos: {details['total_files']}")
                    print(f"  📊 Registros estimados: {details['estimated_total_records']:,}")
                    print(f"  💾 Tamanho: {details['estimated_size_mb']:.1f}MB")
        
        # Salva relatório
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"fx_data_analysis_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Relatório salvo: {report_file}")
        
        return analysis

def main():
    # Diretório dos dados FX
    data_dir = os.getenv("FX_DATA_DIR", "data/FX-1-Minute-Data")
    
    analyzer = FXDataAnalyzer(data_dir)
    analysis = analyzer.generate_report()
    
    return analysis

if __name__ == "__main__":
    main()