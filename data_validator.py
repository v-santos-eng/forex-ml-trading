#!/usr/bin/env python3
"""
Validador de Dados FX - Verificação de Integridade e Consistência
Verifica a qualidade dos dados extraídos para treinamento ML
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_validation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FXDataValidator:
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        self.validation_results = {}
        self.top_5_pairs = ['gbpjpy', 'xauusd', 'eurjpy', 'usdsek', 'usdnok']
        self.all_15_pairs = [
            'gbpjpy', 'xauusd', 'eurjpy', 'usdsek', 'usdnok',
            'euraud', 'usdjpy', 'usdchf', 'gbpnzd', 'gbpchf',
            'usddkk', 'eurnzd', 'audjpy', 'eurusd', 'gbpusd'
        ]
        
    def validate_file_structure(self, pair_name: str) -> Dict:
        """Valida a estrutura de arquivos de um par"""
        logger.info(f"🔍 Validando estrutura de arquivos para {pair_name.upper()}...")
        
        pair_dir = self.data_directory / pair_name.upper()
        if not pair_dir.exists():
            return {
                'exists': False,
                'error': f'Diretório não encontrado: {pair_dir}'
            }
        
        csv_files = list(pair_dir.glob('*.csv'))
        txt_files = list(pair_dir.glob('*.txt'))
        
        return {
            'exists': True,
            'csv_files': len(csv_files),
            'txt_files': len(txt_files),
            'total_files': len(csv_files) + len(txt_files),
            'csv_file_list': [f.name for f in csv_files],
            'txt_file_list': [f.name for f in txt_files]
        }
    
    def validate_data_format(self, file_path: Path) -> Dict:
        """Valida o formato dos dados em um arquivo"""
        try:
            # Lê uma amostra do arquivo
            df_sample = pd.read_csv(file_path, sep=';', header=None, nrows=1000)
            
            # Verifica número de colunas
            expected_cols = 6  # Date, Time, Open, High, Low, Close
            actual_cols = len(df_sample.columns)
            
            # Verifica se há dados
            if len(df_sample) == 0:
                return {
                    'valid': False,
                    'error': 'Arquivo vazio',
                    'records': 0
                }
            
            # Tenta converter colunas OHLC para float
            ohlc_valid = True
            try:
                if actual_cols >= 6:
                    pd.to_numeric(df_sample.iloc[:, 2])  # Open
                    pd.to_numeric(df_sample.iloc[:, 3])  # High
                    pd.to_numeric(df_sample.iloc[:, 4])  # Low
                    pd.to_numeric(df_sample.iloc[:, 5])  # Close
            except:
                ohlc_valid = False
            
            # Calcula estatísticas básicas
            total_records = len(df_sample)
            null_count = df_sample.isnull().sum().sum()
            completeness = 1 - (null_count / (total_records * actual_cols))
            
            return {
                'valid': True,
                'columns': actual_cols,
                'expected_columns': expected_cols,
                'format_correct': bool(actual_cols >= expected_cols),
                'ohlc_valid': bool(ohlc_valid),
                'records_sample': total_records,
                'null_values': int(null_count),
                'completeness': float(completeness),
                'data_quality_score': float(completeness * (1 if ohlc_valid else 0.5))
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'records': 0
            }
    
    def validate_data_consistency(self, file_path: Path) -> Dict:
        """Valida a consistência dos dados OHLC"""
        try:
            df = pd.read_csv(file_path, sep=';', header=None, nrows=5000)
            
            if len(df.columns) < 6:
                return {'valid': False, 'error': 'Colunas insuficientes'}
            
            # Converte OHLC para numérico
            open_col = pd.to_numeric(df.iloc[:, 2], errors='coerce')
            high_col = pd.to_numeric(df.iloc[:, 3], errors='coerce')
            low_col = pd.to_numeric(df.iloc[:, 4], errors='coerce')
            close_col = pd.to_numeric(df.iloc[:, 5], errors='coerce')
            
            # Verifica consistência OHLC
            high_valid = (high_col >= open_col) & (high_col >= close_col) & (high_col >= low_col)
            low_valid = (low_col <= open_col) & (low_col <= close_col) & (low_col <= high_col)
            
            consistency_score = (high_valid.sum() + low_valid.sum()) / (2 * len(df))
            
            # Detecta outliers (variações > 10%)
            price_changes = abs(close_col.pct_change())
            outliers = (price_changes > 0.1).sum()
            outlier_ratio = outliers / len(df)
            
            return {
                'valid': True,
                'consistency_score': float(consistency_score),
                'outliers': int(outliers),
                'outlier_ratio': float(outlier_ratio),
                'high_low_consistency': float(high_valid.mean()),
                'data_integrity': float((1 - outlier_ratio) * consistency_score)
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def estimate_total_records(self, pair_name: str) -> int:
        """Estima o total de registros para um par"""
        pair_dir = self.data_directory / pair_name.upper()
        total_records = 0
        
        for csv_file in pair_dir.glob('*.csv'):
            try:
                # Estima baseado no tamanho do arquivo
                file_size = csv_file.stat().st_size
                estimated_records = file_size // 50  # ~50 bytes por linha
                total_records += estimated_records
            except:
                continue
                
        return total_records
    
    def validate_pair(self, pair_name: str) -> Dict:
        """Valida completamente um par de moedas"""
        logger.info(f"📊 Validando par {pair_name.upper()}...")
        
        # 1. Validar estrutura de arquivos
        structure_result = self.validate_file_structure(pair_name)
        
        if not structure_result['exists']:
            return {
                'pair': pair_name.upper(),
                'valid': False,
                'error': structure_result['error']
            }
        
        # 2. Validar arquivos CSV
        pair_dir = self.data_directory / pair_name.upper()
        csv_files = list(pair_dir.glob('*.csv'))
        
        if not csv_files:
            return {
                'pair': pair_name.upper(),
                'valid': False,
                'error': 'Nenhum arquivo CSV encontrado'
            }
        
        # 3. Validar formato e consistência
        format_results = []
        consistency_results = []
        
        for csv_file in csv_files[:10]:  # Valida primeiros 10 arquivos
            format_result = self.validate_data_format(csv_file)
            if format_result['valid']:
                format_results.append(format_result)
                
                consistency_result = self.validate_data_consistency(csv_file)
                if consistency_result['valid']:
                    consistency_results.append(consistency_result)
        
        if not format_results:
            return {
                'pair': pair_name.upper(),
                'valid': False,
                'error': 'Nenhum arquivo com formato válido'
            }
        
        # 4. Calcular métricas agregadas
        avg_completeness = np.mean([r['completeness'] for r in format_results])
        avg_data_quality = np.mean([r['data_quality_score'] for r in format_results])
        
        if consistency_results:
            avg_consistency = np.mean([r['consistency_score'] for r in consistency_results])
            avg_integrity = np.mean([r['data_integrity'] for r in consistency_results])
        else:
            avg_consistency = 0.0
            avg_integrity = 0.0
        
        # 5. Estimar total de registros
        total_records = self.estimate_total_records(pair_name)
        
        # 6. Calcular score final (ajustado para dados reais)
        # Prioriza completude e qualidade dos dados sobre consistência perfeita
        final_score = (avg_completeness * 0.4 + 
                      avg_data_quality * 0.4 + 
                      avg_consistency * 0.1 + 
                      avg_integrity * 0.1)
        
        return {
            'pair': pair_name.upper(),
            'valid': True,
            'files_analyzed': len(format_results),
            'total_files': len(csv_files),
            'estimated_records': total_records,
            'completeness': float(avg_completeness),
            'data_quality': float(avg_data_quality),
            'consistency': float(avg_consistency),
            'integrity': float(avg_integrity),
            'final_score': float(final_score),
            'quality_grade': self.get_quality_grade(final_score),
            'ready_for_training': bool(final_score >= 0.6)  # Critério mais realista
        }
    
    def get_quality_grade(self, score: float) -> str:
        """Converte score em nota de qualidade"""
        if score >= 0.9:
            return 'A+ (Excelente)'
        elif score >= 0.8:
            return 'A (Muito Bom)'
        elif score >= 0.7:
            return 'B (Bom)'
        elif score >= 0.6:
            return 'C (Regular)'
        else:
            return 'D (Insuficiente)'
    
    def validate_top_5_pairs(self) -> Dict:
        """Valida os 5 principais pares"""
        logger.info("🎯 Validando TOP 5 pares para treinamento prioritário...")
        
        results = {}
        for pair in self.top_5_pairs:
            results[pair] = self.validate_pair(pair)
        
        return results
    
    def validate_all_pairs(self) -> Dict:
        """Valida todos os 15 pares selecionados"""
        logger.info("📊 Validando todos os 15 pares selecionados...")
        
        results = {}
        for pair in self.all_15_pairs:
            results[pair] = self.validate_pair(pair)
        
        return results
    
    def generate_validation_report(self, results: Dict) -> str:
        """Gera relatório de validação"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"data_validation_report_{timestamp}.json"
        
        # Calcula estatísticas gerais
        valid_pairs = [p for p, r in results.items() if r.get('valid', False)]
        ready_pairs = [p for p, r in results.items() if r.get('ready_for_training', False)]
        
        avg_score = np.mean([r.get('final_score', 0) for r in results.values() if r.get('valid', False)])
        total_records = sum([r.get('estimated_records', 0) for r in results.values()])
        
        report = {
            'timestamp': timestamp,
            'validation_summary': {
                'total_pairs_analyzed': len(results),
                'valid_pairs': len(valid_pairs),
                'ready_for_training': len(ready_pairs),
                'average_quality_score': float(avg_score),
                'total_estimated_records': total_records,
                'validation_status': 'PASSED' if len(ready_pairs) >= 5 else 'NEEDS_ATTENTION'
            },
            'pair_results': results,
            'recommendations': self.generate_recommendations(results)
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Relatório de validação salvo: {report_file}")
        return report_file
    
    def generate_recommendations(self, results: Dict) -> Dict:
        """Gera recomendações baseadas na validação"""
        ready_pairs = [p for p, r in results.items() if r.get('ready_for_training', False)]
        problem_pairs = [p for p, r in results.items() if not r.get('ready_for_training', False)]
        
        recommendations = {
            'immediate_training': {
                'pairs': ready_pairs[:5],
                'reason': 'Pares com qualidade suficiente para treinamento imediato',
                'action': 'Iniciar pré-processamento e treinamento'
            },
            'secondary_training': {
                'pairs': ready_pairs[5:],
                'reason': 'Pares adicionais para expansão do treinamento',
                'action': 'Incluir após validação do modelo inicial'
            },
            'needs_attention': {
                'pairs': problem_pairs,
                'reason': 'Pares com problemas de qualidade',
                'action': 'Revisar dados ou excluir do treinamento'
            }
        }
        
        return recommendations
    
    def print_validation_summary(self, results: Dict):
        """Imprime resumo da validação"""
        print("\n" + "="*60)
        print("📊 RESUMO DA VALIDAÇÃO DE DADOS")
        print("="*60)
        
        valid_pairs = [p for p, r in results.items() if r.get('valid', False)]
        ready_pairs = [p for p, r in results.items() if r.get('ready_for_training', False)]
        
        print(f"\n✅ Pares Válidos: {len(valid_pairs)}/{len(results)}")
        print(f"🎯 Prontos para Treinamento: {len(ready_pairs)}")
        
        print("\n🏆 TOP 5 PARES RECOMENDADOS:")
        sorted_pairs = sorted(
            [(p, r) for p, r in results.items() if r.get('valid', False)],
            key=lambda x: x[1].get('final_score', 0),
            reverse=True
        )
        
        for i, (pair, result) in enumerate(sorted_pairs[:5], 1):
            score = result.get('final_score', 0)
            grade = result.get('quality_grade', 'N/A')
            records = result.get('estimated_records', 0)
            print(f"{i}. {pair}: {score:.3f} ({grade}) - {records:,} registros")
        
        if len(ready_pairs) >= 5:
            print("\n🎉 VALIDAÇÃO CONCLUÍDA COM SUCESSO!")
            print("✅ Dados prontos para pré-processamento e treinamento")
        elif len(ready_pairs) >= 3:
            print("\n⚠️ VALIDAÇÃO PARCIAL: Pares suficientes para início")
            print("✅ Prosseguir com pré-processamento dos melhores pares")
        else:
            print("\n⚠️ ATENÇÃO: Poucos pares prontos para treinamento")
            print("❌ Revisar qualidade dos dados antes de prosseguir")

def main():
    # Configuração
    data_dir = "extracted_fx_data"
    
    if not Path(data_dir).exists():
        print(f"❌ Diretório não encontrado: {data_dir}")
        print("Execute primeiro o pipeline de extração de dados")
        return
    
    # Inicializar validador
    validator = FXDataValidator(data_dir)
    
    print("🚀 Iniciando Validação de Dados FX")
    print("="*50)
    
    # Validar TOP 5 pares primeiro
    print("\n🎯 Fase 1: Validando TOP 5 pares...")
    top5_results = validator.validate_top_5_pairs()
    
    # Validar todos os pares
    print("\n📊 Fase 2: Validando todos os 15 pares...")
    all_results = validator.validate_all_pairs()
    
    # Gerar relatório
    report_file = validator.generate_validation_report(all_results)
    
    # Imprimir resumo
    validator.print_validation_summary(all_results)
    
    print(f"\n📋 Relatório detalhado: {report_file}")
    print("\n🎯 Próximo passo: Pré-processamento dos dados validados")

if __name__ == "__main__":
    main()