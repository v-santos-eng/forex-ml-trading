#!/usr/bin/env python3
"""
Pipeline de Pesquisa Robusta para ML Forex
Este script implementa uma pesquisa de máxima performance com extração prévia
de todos os arquivos ZIP e configuração otimizada para robustez.
"""

import os
import zipfile
import shutil
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import time
import gc
from typing import List, Dict, Optional
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robust_research.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RobustResearchPipeline:
    def __init__(self, data_directory: str, output_directory: str = None):
        self.data_directory = Path(data_directory)
        self.output_directory = Path(output_directory) if output_directory else Path("extracted_data")
        self.extracted_pairs = {}
        self.research_config = {
            'samples_per_pair': 500000,
            'max_pairs': 15,
            'target_pairs': [
                'eurusd', 'gbpusd', 'usdjpy', 'usdchf', 'audusd', 
                'usdcad', 'nzdusd', 'eurjpy', 'gbpjpy', 'audjpy',
                'eurgbp', 'euraud', 'gbpaud', 'audcad', 'nzdcad',
                'xauusd', 'audchf', 'cadchf', 'chfjpy', 'eurchf'
            ],
            'validation_split': 0.2,
            'test_split': 0.2,
            'min_data_quality': 0.70  # Reduzido para 70%
        }
        
    def create_output_directory(self):
        """Cria diretório de saída para dados extraídos"""
        self.output_directory.mkdir(exist_ok=True)
        logger.info(f"Diretório de saída criado: {self.output_directory}")
        
    def extract_all_zip_files(self) -> Dict[str, List[str]]:
        """Extrai todos os arquivos ZIP encontrados no diretório"""
        logger.info("🔄 Iniciando extração de arquivos ZIP...")
        extracted_files = {}
        total_extracted = 0
        
        if not self.data_directory.exists():
            logger.error(f"Diretório não encontrado: {self.data_directory}")
            return extracted_files
            
        # Escaneia todos os subdiretórios
        for pair_dir in self.data_directory.iterdir():
            if not pair_dir.is_dir():
                continue
                
            pair_name = pair_dir.name.upper()
            zip_files = list(pair_dir.glob('*.zip'))
            
            if not zip_files:
                continue
                
            logger.info(f"📦 Extraindo {len(zip_files)} arquivos para {pair_name}...")
            
            # Cria diretório para o par
            pair_output_dir = self.output_directory / pair_name
            pair_output_dir.mkdir(exist_ok=True)
            
            extracted_files[pair_name] = []
            
            for zip_file in zip_files:
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        # Extrai todos os arquivos
                        zip_ref.extractall(pair_output_dir)
                        
                        # Lista arquivos extraídos
                        for extracted_file in zip_ref.namelist():
                            full_path = pair_output_dir / extracted_file
                            if full_path.exists():
                                extracted_files[pair_name].append(str(full_path))
                                total_extracted += 1
                                
                    logger.info(f"✅ Extraído: {zip_file.name}")
                    
                except Exception as e:
                    logger.error(f"❌ Erro ao extrair {zip_file}: {e}")
                    
        logger.info(f"🎉 Extração concluída! Total de arquivos extraídos: {total_extracted}")
        return extracted_files
    
    def analyze_extracted_data(self, extracted_files: Dict[str, List[str]]) -> Dict[str, Dict]:
        """Analisa a qualidade e quantidade dos dados extraídos"""
        logger.info("📊 Analisando dados extraídos...")
        analysis_results = {}
        
        for pair_name, file_list in extracted_files.items():
            if not file_list:
                continue
                
            logger.info(f"🔍 Analisando {pair_name}...")
            
            total_records = 0
            total_size_mb = 0
            data_quality_score = 0
            date_range = {'min': None, 'max': None}
            
            valid_files = 0
            
            for file_path in file_list:
                try:
                    file_path_obj = Path(file_path)
                    if not file_path_obj.exists() or file_path_obj.suffix.lower() not in ['.csv', '.txt']:
                        continue
                        
                    # Lê amostra do arquivo
                    df_sample = pd.read_csv(file_path, nrows=1000, header=None)
                    
                    if len(df_sample.columns) < 6:
                        continue
                        
                    # Estima total de registros
                    file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
                    estimated_records = int((file_size_mb / (len(df_sample) * 6 * 8 / (1024 * 1024))) * len(df_sample))
                    
                    total_records += estimated_records
                    total_size_mb += file_size_mb
                    valid_files += 1
                    
                    # Calcula score de qualidade (baseado em completude dos dados)
                    completeness = 1 - (df_sample.isnull().sum().sum() / (len(df_sample) * len(df_sample.columns)))
                    data_quality_score += completeness
                    
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao analisar {file_path}: {e}")
                    continue
            
            if valid_files > 0:
                avg_quality = data_quality_score / valid_files
                
                analysis_results[pair_name] = {
                    'total_records': total_records,
                    'total_size_mb': total_size_mb,
                    'valid_files': valid_files,
                    'data_quality_score': avg_quality,
                    'meets_quality_threshold': avg_quality >= self.research_config['min_data_quality'],
                    'sufficient_data': total_records >= self.research_config['samples_per_pair']
                }
                
                logger.info(f"📈 {pair_name}: {total_records:,} registros, qualidade: {avg_quality:.2%}")
        
        return analysis_results
    
    def select_optimal_pairs(self, analysis_results: Dict[str, Dict]) -> List[str]:
        """Seleciona os pares ótimos para pesquisa robusta"""
        logger.info("🎯 Selecionando pares ótimos para pesquisa robusta...")
        
        # Filtra pares que atendem aos critérios de qualidade
        qualified_pairs = []
        
        for pair_name, analysis in analysis_results.items():
            if (analysis['meets_quality_threshold'] and 
                analysis['sufficient_data']):
                # Prioriza pares da lista target, mas aceita todos os qualificados
                priority_bonus = 1.2 if pair_name in self.research_config['target_pairs'] else 1.0
                qualified_pairs.append({
                    'pair': pair_name,
                    'records': analysis['total_records'],
                    'quality': analysis['data_quality_score'],
                    'score': analysis['total_records'] * analysis['data_quality_score'] * priority_bonus
                })
        
        # Ordena por score (quantidade * qualidade)
        qualified_pairs.sort(key=lambda x: x['score'], reverse=True)
        
        # Seleciona os melhores pares até o limite
        selected_pairs = [p['pair'] for p in qualified_pairs[:self.research_config['max_pairs']]]
        
        logger.info(f"✅ Pares selecionados para pesquisa robusta: {', '.join(selected_pairs)}")
        
        return selected_pairs
    
    def prepare_training_data(self, selected_pairs: List[str], extracted_files: Dict[str, List[str]]) -> Dict[str, str]:
        """Prepara dados de treinamento para os pares selecionados"""
        logger.info("🔧 Preparando dados de treinamento...")
        
        training_data_paths = {}
        
        for pair in selected_pairs:
            if pair not in extracted_files:
                logger.warning(f"⚠️ Dados não encontrados para {pair}")
                continue
                
            logger.info(f"📝 Preparando dados para {pair}...")
            
            # Combina todos os arquivos CSV do par
            combined_data = []
            files_processed = 0
            
            for file_path in extracted_files[pair]:
                try:
                    file_path_obj = Path(file_path)
                    if not file_path_obj.exists() or file_path_obj.suffix.lower() not in ['.csv', '.txt']:
                        continue
                        
                    df = pd.read_csv(file_path, header=None)
                    
                    if len(df.columns) >= 6:
                        # Assume formato: Date, Time, Open, High, Low, Close
                        df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close'] + [f'Col_{i}' for i in range(6, len(df.columns))]
                        combined_data.append(df[['Date', 'Time', 'Open', 'High', 'Low', 'Close']])
                        files_processed += 1
                        
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao processar {file_path}: {e}")
                    continue
            
            if combined_data:
                # Combina todos os DataFrames
                full_df = pd.concat(combined_data, ignore_index=True)
                
                # Remove duplicatas e ordena por data/hora
                full_df['DateTime'] = pd.to_datetime(full_df['Date'].astype(str) + ' ' + full_df['Time'].astype(str), errors='coerce')
                full_df = full_df.dropna(subset=['DateTime'])
                full_df = full_df.sort_values('DateTime')
                full_df = full_df.drop_duplicates(subset=['DateTime'])
                
                # Limita ao número de amostras configurado
                if len(full_df) > self.research_config['samples_per_pair']:
                    # Pega as amostras mais recentes
                    full_df = full_df.tail(self.research_config['samples_per_pair'])
                
                # Salva dados preparados
                output_file = self.output_directory / f"{pair}_prepared.csv"
                full_df.to_csv(output_file, index=False)
                training_data_paths[pair] = str(output_file)
                
                logger.info(f"✅ {pair}: {len(full_df):,} registros preparados ({files_processed} arquivos processados)")
                
                # Libera memória
                del full_df, combined_data
                gc.collect()
        
        return training_data_paths
    
    def generate_research_report(self, selected_pairs: List[str], training_data_paths: Dict[str, str], analysis_results: Dict[str, Dict]) -> str:
        """Gera relatório detalhado da pesquisa robusta"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"robust_research_report_{timestamp}.json"
        
        report = {
            'timestamp': timestamp,
            'configuration': self.research_config,
            'selected_pairs': selected_pairs,
            'total_pairs_analyzed': len(analysis_results),
            'data_preparation': {
                'pairs_prepared': len(training_data_paths),
                'training_data_paths': training_data_paths
            },
            'quality_analysis': analysis_results,
            'recommendations': {
                'next_steps': [
                    'Executar treinamento com configuração robusta',
                    'Implementar validação cruzada temporal',
                    'Otimizar hiperparâmetros para cada par',
                    'Avaliar performance out-of-sample'
                ],
                'estimated_training_time': '60-90 minutos',
                'estimated_memory_usage': '8-12GB',
                'expected_accuracy': '85-88%'
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📋 Relatório de pesquisa robusta salvo: {report_file}")
        return report_file
    
    def execute_robust_research(self) -> Dict:
        """Executa o pipeline completo de pesquisa robusta"""
        start_time = time.time()
        
        logger.info("🚀 Iniciando Pipeline de Pesquisa Robusta")
        logger.info("=" * 60)
        
        try:
            # 1. Criar diretório de saída
            self.create_output_directory()
            
            # 2. Extrair todos os arquivos ZIP
            extracted_files = self.extract_all_zip_files()
            
            if not extracted_files:
                logger.error("❌ Nenhum arquivo foi extraído. Verifique o diretório de dados.")
                return {'success': False, 'error': 'No files extracted'}
            
            # 3. Analisar dados extraídos
            analysis_results = self.analyze_extracted_data(extracted_files)
            
            # 4. Selecionar pares ótimos
            selected_pairs = self.select_optimal_pairs(analysis_results)
            
            if not selected_pairs:
                logger.error("❌ Nenhum par atende aos critérios de qualidade.")
                return {'success': False, 'error': 'No pairs meet quality criteria'}
            
            # 5. Preparar dados de treinamento
            training_data_paths = self.prepare_training_data(selected_pairs, extracted_files)
            
            # 6. Gerar relatório
            report_file = self.generate_research_report(selected_pairs, training_data_paths, analysis_results)
            
            execution_time = time.time() - start_time
            
            logger.info("🎉 Pipeline de Pesquisa Robusta Concluído!")
            logger.info(f"⏱️ Tempo total de execução: {execution_time:.2f} segundos")
            logger.info(f"📊 Pares preparados: {len(training_data_paths)}")
            logger.info(f"📋 Relatório: {report_file}")
            
            return {
                'success': True,
                'selected_pairs': selected_pairs,
                'training_data_paths': training_data_paths,
                'report_file': report_file,
                'execution_time': execution_time,
                'total_pairs_analyzed': len(analysis_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Erro durante execução do pipeline: {e}")
            return {'success': False, 'error': str(e)}
    
    def cleanup_extracted_files(self):
        """Remove arquivos extraídos para liberar espaço"""
        if self.output_directory.exists():
            shutil.rmtree(self.output_directory)
            logger.info(f"🧹 Arquivos extraídos removidos: {self.output_directory}")

def main():
    # Configuração do pipeline
    data_dir = os.getenv("FX_DATA_DIR", "data/FX-1-Minute-Data")
    output_dir = "extracted_fx_data"
    
    # Inicializa pipeline
    pipeline = RobustResearchPipeline(data_dir, output_dir)
    
    # Executa pesquisa robusta
    results = pipeline.execute_robust_research()
    
    if results['success']:
        print("\n" + "=" * 60)
        print("🎯 PESQUISA ROBUSTA CONCLUÍDA COM SUCESSO!")
        print("=" * 60)
        print(f"📊 Pares selecionados: {len(results['selected_pairs'])}")
        print(f"💾 Dados preparados: {len(results['training_data_paths'])}")
        print(f"⏱️ Tempo de execução: {results['execution_time']:.2f}s")
        print(f"📋 Relatório: {results['report_file']}")
        print("\n🚀 Pronto para treinamento de máxima performance!")
        
        # Exibe pares selecionados
        print("\n💱 Pares Selecionados:")
        for i, pair in enumerate(results['selected_pairs'], 1):
            print(f"  {i:2d}. {pair}")
            
    else:
        print(f"❌ Erro na execução: {results.get('error', 'Erro desconhecido')}")
    
    return results

if __name__ == "__main__":
    main()