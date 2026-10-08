#!/usr/bin/env python3
"""
Relatório de Análise dos Resultados da Avaliação de Modelos Forex
Analisa os resultados da avaliação final e gera insights sobre performance.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

class EvaluationAnalyzer:
    """Analisador de resultados de avaliação"""
    
    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.results_data = self._load_results()
        self.df = self._create_dataframe()
    
    def _load_results(self) -> dict:
        """Carrega resultados do arquivo JSON"""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Cria DataFrame dos resultados"""
        results = []
        for result in self.results_data['results']:
            row = {
                'pair': result['pair'],
                'model_type': result['model_type'],
                'model_category': result['model_category'],
                'success': result['success'],
                'sample_size': result['sample_size'],
                'error_message': result['error_message'],
                'feature_count': result['feature_count']
            }
            
            # Adicionar métricas
            if result['success'] and result['metrics']:
                for metric, value in result['metrics'].items():
                    row[f'metric_{metric}'] = value
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def analyze_success_rates(self):
        """Analisa taxas de sucesso por diferentes dimensões"""
        print("=== ANÁLISE DE TAXAS DE SUCESSO ===")
        print()
        
        # Taxa geral
        total = len(self.df)
        successful = self.df['success'].sum()
        print(f"Taxa de Sucesso Geral: {successful}/{total} ({successful/total*100:.1f}%)")
        print()
        
        # Por tipo de modelo
        print("Taxa de Sucesso por Tipo de Modelo:")
        model_success = self.df.groupby('model_type')['success'].agg(['count', 'sum', 'mean'])
        model_success['success_rate'] = model_success['mean'] * 100
        model_success = model_success.sort_values('success_rate', ascending=False)
        
        for model_type, row in model_success.iterrows():
            print(f"  {model_type}: {int(row['sum'])}/{int(row['count'])} ({row['success_rate']:.1f}%)")
        print()
        
        # Por par de moedas
        print("Taxa de Sucesso por Par de Moedas:")
        pair_success = self.df.groupby('pair')['success'].agg(['count', 'sum', 'mean'])
        pair_success['success_rate'] = pair_success['mean'] * 100
        pair_success = pair_success.sort_values('success_rate', ascending=False)
        
        for pair, row in pair_success.iterrows():
            print(f"  {pair}: {int(row['sum'])}/{int(row['count'])} ({row['success_rate']:.1f}%)")
        print()
    
    def analyze_failures(self):
        """Analisa padrões de falhas"""
        print("=== ANÁLISE DE FALHAS ===")
        print()
        
        failed_df = self.df[~self.df['success']]
        
        if len(failed_df) == 0:
            print("Nenhuma falha encontrada!")
            return
        
        # Contagem de tipos de erro
        error_counts = Counter(failed_df['error_message'].fillna('Erro desconhecido'))
        
        print("Tipos de Erro (frequência):")
        for error, count in error_counts.most_common():
            print(f"  {error}: {count} ocorrências")
        print()
        
        # Modelos mais problemáticos
        print("Modelos com Mais Falhas:")
        failed_by_model = failed_df['model_type'].value_counts()
        for model_type, count in failed_by_model.items():
            total_for_model = len(self.df[self.df['model_type'] == model_type])
            print(f"  {model_type}: {count}/{total_for_model} falhas")
        print()
    
    def analyze_performance_metrics(self):
        """Analisa métricas de performance dos modelos bem-sucedidos"""
        print("=== ANÁLISE DE MÉTRICAS DE PERFORMANCE ===")
        print()
        
        successful_df = self.df[self.df['success']]
        
        if len(successful_df) == 0:
            print("Nenhum modelo bem-sucedido para analisar.")
            return
        
        # Identificar colunas de métricas
        metric_cols = [col for col in successful_df.columns if col.startswith('metric_')]
        
        if not metric_cols:
            print("Nenhuma métrica encontrada.")
            return
        
        # Estatísticas por tipo de modelo
        print("Métricas por Tipo de Modelo (média ± desvio):")
        
        for model_type in successful_df['model_type'].unique():
            model_data = successful_df[successful_df['model_type'] == model_type]
            print(f"\n{model_type} ({len(model_data)} modelos):")
            
            for metric_col in metric_cols:
                if metric_col in model_data.columns:
                    values = model_data[metric_col].dropna()
                    if len(values) > 0:
                        metric_name = metric_col.replace('metric_', '')
                        mean_val = values.mean()
                        std_val = values.std()
                        print(f"  {metric_name}: {mean_val:.6f} ± {std_val:.6f}")
        print()
    
    def analyze_feature_usage(self):
        """Analisa uso de features"""
        print("=== ANÁLISE DE FEATURES ===")
        print()
        
        successful_df = self.df[self.df['success']]
        
        if len(successful_df) == 0:
            print("Nenhum modelo bem-sucedido para analisar features.")
            return
        
        feature_counts = successful_df['feature_count'].value_counts().sort_index()
        
        print("Distribuição de Número de Features:")
        for count, freq in feature_counts.items():
            print(f"  {count} features: {freq} modelos")
        print()
        
        # Tamanho médio das amostras
        print("Estatísticas de Amostras:")
        sample_stats = successful_df['sample_size'].describe()
        print(f"  Média: {sample_stats['mean']:.0f} amostras")
        print(f"  Mediana: {sample_stats['50%']:.0f} amostras")
        print(f"  Min-Max: {sample_stats['min']:.0f} - {sample_stats['max']:.0f} amostras")
        print()
    
    def generate_recommendations(self):
        """Gera recomendações baseadas na análise"""
        print("=== RECOMENDAÇÕES ===")
        print()
        
        # Análise de modelos mais confiáveis
        model_success_rates = self.df.groupby('model_type')['success'].mean().sort_values(ascending=False)
        
        print("1. Modelos Mais Confiáveis:")
        for model_type, success_rate in model_success_rates.head(3).items():
            print(f"   - {model_type}: {success_rate*100:.1f}% de sucesso")
        print()
        
        # Análise de pares mais problemáticos
        pair_success_rates = self.df.groupby('pair')['success'].mean().sort_values()
        
        print("2. Pares que Precisam de Atenção:")
        for pair, success_rate in pair_success_rates.head(3).items():
            if success_rate < 0.8:  # Menos de 80% de sucesso
                print(f"   - {pair}: {success_rate*100:.1f}% de sucesso")
        print()
        
        # Análise de erros mais comuns
        failed_df = self.df[~self.df['success']]
        if len(failed_df) > 0:
            most_common_error = failed_df['error_message'].value_counts().index[0]
            error_count = failed_df['error_message'].value_counts().iloc[0]
            
            print("3. Problema Mais Crítico:")
            print(f"   - '{most_common_error}': {error_count} ocorrências")
            print("   - Recomendação: Investigar e corrigir este erro prioritariamente")
        print()
        
        print("4. Próximos Passos Sugeridos:")
        print("   - Implementar modelos LightGBM ausentes")
        print("   - Verificar integridade dos dados de classificação")
        print("   - Otimizar pipeline de preprocessamento")
        print("   - Implementar validação cruzada temporal")
        print()
    
    def generate_full_report(self):
        """Gera relatório completo"""
        print(f"RELATÓRIO DE AVALIAÇÃO DE MODELOS FOREX")
        print(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Arquivo analisado: {self.results_file.name}")
        print("=" * 60)
        print()
        
        self.analyze_success_rates()
        self.analyze_failures()
        self.analyze_performance_metrics()
        self.analyze_feature_usage()
        self.generate_recommendations()
        
        print("=" * 60)
        print("Relatório concluído.")

def main():
    """Função principal"""
    # Encontrar o arquivo de resultados mais recente
    results_dir = Path("model_evaluation_final")
    
    if not results_dir.exists():
        print("Diretório de resultados não encontrado!")
        return
    
    result_files = list(results_dir.glob("evaluation_results_final_*.json"))
    
    if not result_files:
        print("Nenhum arquivo de resultados encontrado!")
        return
    
    # Usar o arquivo mais recente
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Analisando: {latest_file.name}")
    print()
    
    analyzer = EvaluationAnalyzer(latest_file)
    analyzer.generate_full_report()

if __name__ == "__main__":
    main()