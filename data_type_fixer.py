#!/usr/bin/env python3
"""
Script para diagnosticar e corrigir problemas de tipos de dados nos arquivos processados
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
import json
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataTypeFixer:
    def __init__(self, processed_data_dir="processed_fx_data"):
        self.processed_data_dir = Path(processed_data_dir)
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "pairs_analyzed": [],
            "issues_found": [],
            "fixes_applied": [],
            "summary": {}
        }
    
    def diagnose_data_types(self, pair_name):
        """Diagnostica problemas de tipos de dados para um par específico"""
        pair_dir = self.processed_data_dir / pair_name
        
        if not pair_dir.exists():
            logger.warning(f"Diretório não encontrado: {pair_dir}")
            return None
        
        issues = []
        
        for file_type in ['train', 'test', 'validation']:
            file_path = pair_dir / f"{file_type}.parquet"
            
            if not file_path.exists():
                issues.append(f"Arquivo {file_type}.parquet não encontrado")
                continue
            
            try:
                # Tentar carregar o arquivo
                df = pd.read_parquet(file_path)
                
                # Verificar tipos de dados
                type_issues = self._check_data_types(df, file_type)
                issues.extend(type_issues)
                
                # Verificar valores NaN/Inf
                nan_issues = self._check_nan_inf_values(df, file_type)
                issues.extend(nan_issues)
                
                # Verificar colunas mistas
                mixed_issues = self._check_mixed_types(df, file_type)
                issues.extend(mixed_issues)
                
                logger.info(f"✅ {pair_name}/{file_type}.parquet - {len(df)} linhas, {len(df.columns)} colunas")
                
            except Exception as e:
                error_msg = f"Erro ao carregar {file_type}.parquet: {str(e)}"
                issues.append(error_msg)
                logger.error(f"❌ {pair_name}/{file_type}.parquet - {error_msg}")
        
        return issues
    
    def _check_data_types(self, df, file_type):
        """Verifica tipos de dados problemáticos"""
        issues = []
        
        for col in df.columns:
            dtype = df[col].dtype
            
            # Verificar tipos object que deveriam ser numéricos
            if dtype == 'object':
                # Tentar converter para numérico
                try:
                    pd.to_numeric(df[col], errors='raise')
                    issues.append(f"{file_type}: Coluna '{col}' é object mas deveria ser numérica")
                except:
                    # Se não conseguir converter, pode ser string válida
                    pass
            
            # Verificar tipos mistos
            if dtype == 'object':
                unique_types = set(type(x).__name__ for x in df[col].dropna().iloc[:100])
                if len(unique_types) > 1:
                    issues.append(f"{file_type}: Coluna '{col}' tem tipos mistos: {unique_types}")
        
        return issues
    
    def _check_nan_inf_values(self, df, file_type):
        """Verifica valores NaN e Inf problemáticos"""
        issues = []
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Verificar NaN
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    nan_pct = (nan_count / len(df)) * 100
                    if nan_pct > 50:
                        issues.append(f"{file_type}: Coluna '{col}' tem {nan_pct:.1f}% valores NaN")
                
                # Verificar Inf
                if np.isinf(df[col]).any():
                    inf_count = np.isinf(df[col]).sum()
                    issues.append(f"{file_type}: Coluna '{col}' tem {inf_count} valores infinitos")
        
        return issues
    
    def _check_mixed_types(self, df, file_type):
        """Verifica colunas com tipos mistos"""
        issues = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().iloc[:100]
                types_found = set()
                
                for val in sample_values:
                    if isinstance(val, str):
                        # Tentar identificar se é número em string
                        try:
                            float(val)
                            types_found.add('numeric_string')
                        except:
                            types_found.add('string')
                    else:
                        types_found.add(type(val).__name__)
                
                if len(types_found) > 1:
                    issues.append(f"{file_type}: Coluna '{col}' tem tipos mistos: {types_found}")
        
        return issues
    
    def fix_data_types(self, pair_name):
        """Corrige problemas de tipos de dados para um par específico"""
        pair_dir = self.processed_data_dir / pair_name
        
        if not pair_dir.exists():
            logger.warning(f"Diretório não encontrado: {pair_dir}")
            return False
        
        fixes_applied = []
        
        for file_type in ['train', 'test', 'validation']:
            file_path = pair_dir / f"{file_type}.parquet"
            
            if not file_path.exists():
                continue
            
            try:
                # Carregar dados
                df = pd.read_parquet(file_path)
                original_shape = df.shape
                
                # Aplicar correções
                df_fixed = self._apply_fixes(df)
                
                # Salvar arquivo corrigido
                backup_path = pair_dir / f"{file_type}_backup.parquet"
                df.to_parquet(backup_path)  # Backup
                df_fixed.to_parquet(file_path)  # Arquivo corrigido
                
                fixes_applied.append(f"{file_type}: {original_shape} -> {df_fixed.shape}")
                logger.info(f"✅ Corrigido {pair_name}/{file_type}.parquet")
                
            except Exception as e:
                error_msg = f"Erro ao corrigir {file_type}.parquet: {str(e)}"
                logger.error(f"❌ {error_msg}")
                fixes_applied.append(error_msg)
        
        return fixes_applied
    
    def _apply_fixes(self, df):
        """Aplica correções aos dados"""
        df_fixed = df.copy()
        
        for col in df_fixed.columns:
            # Converter colunas object para numérico quando possível
            if df_fixed[col].dtype == 'object':
                try:
                    # Tentar conversão numérica
                    df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
                except:
                    pass
            
            # Tratar valores infinitos
            if df_fixed[col].dtype in ['float64', 'float32']:
                # Substituir inf por NaN
                df_fixed[col] = df_fixed[col].replace([np.inf, -np.inf], np.nan)
                
                # Preencher NaN com mediana (para colunas numéricas)
                if df_fixed[col].isna().any():
                    median_val = df_fixed[col].median()
                    if not np.isnan(median_val):
                        df_fixed[col] = df_fixed[col].fillna(median_val)
        
        # Remover linhas com muitos NaN
        nan_threshold = len(df_fixed.columns) * 0.5  # 50% das colunas
        df_fixed = df_fixed.dropna(thresh=nan_threshold)
        
        return df_fixed
    
    def run_diagnosis(self):
        """Executa diagnóstico completo"""
        logger.info("🔍 Iniciando diagnóstico de tipos de dados...")
        
        if not self.processed_data_dir.exists():
            logger.error(f"Diretório não encontrado: {self.processed_data_dir}")
            return
        
        # Listar pares disponíveis
        pairs = [d.name for d in self.processed_data_dir.iterdir() if d.is_dir()]
        logger.info(f"📊 Encontrados {len(pairs)} pares: {pairs}")
        
        total_issues = 0
        
        for pair in pairs:
            logger.info(f"\n🎯 Analisando {pair}...")
            issues = self.diagnose_data_types(pair)
            
            if issues:
                total_issues += len(issues)
                self.report["issues_found"].extend([f"{pair}: {issue}" for issue in issues])
                logger.warning(f"⚠️  {pair}: {len(issues)} problemas encontrados")
                for issue in issues:
                    logger.warning(f"   - {issue}")
            else:
                logger.info(f"✅ {pair}: Nenhum problema encontrado")
            
            self.report["pairs_analyzed"].append(pair)
        
        # Resumo
        self.report["summary"] = {
            "total_pairs": len(pairs),
            "pairs_with_issues": len([p for p in pairs if any(p in issue for issue in self.report["issues_found"])]),
            "total_issues": total_issues
        }
        
        logger.info(f"\n📋 RESUMO DO DIAGNÓSTICO:")
        logger.info(f"   Pares analisados: {self.report['summary']['total_pairs']}")
        logger.info(f"   Pares com problemas: {self.report['summary']['pairs_with_issues']}")
        logger.info(f"   Total de problemas: {self.report['summary']['total_issues']}")
        
        # Salvar relatório
        report_file = f"data_type_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Relatório salvo: {report_file}")
        
        return total_issues == 0
    
    def run_fixes(self):
        """Executa correções para todos os pares"""
        logger.info("🔧 Iniciando correção de tipos de dados...")
        
        pairs = [d.name for d in self.processed_data_dir.iterdir() if d.is_dir()]
        
        for pair in pairs:
            logger.info(f"\n🎯 Corrigindo {pair}...")
            fixes = self.fix_data_types(pair)
            
            if fixes:
                self.report["fixes_applied"].extend([f"{pair}: {fix}" for fix in fixes])
        
        logger.info("✅ Correções concluídas!")

def main():
    fixer = DataTypeFixer()
    
    # Executar diagnóstico
    diagnosis_ok = fixer.run_diagnosis()
    
    if not diagnosis_ok:
        print("\n🔧 Problemas encontrados. Executando correções...")
        fixer.run_fixes()
        
        # Executar diagnóstico novamente
        print("\n🔍 Executando diagnóstico pós-correção...")
        fixer.run_diagnosis()
    else:
        print("\n✅ Nenhum problema encontrado nos dados!")

if __name__ == "__main__":
    main()