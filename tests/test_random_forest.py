#!/usr/bin/env python3
"""
Script de teste para verificar se o RandomForest funciona com dados pequenos
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_random_forest_small():
    """Testa RandomForest com dados sintéticos pequenos"""
    logger.info("🧪 Testando RandomForest com dados sintéticos pequenos...")
    
    # Criar dados sintéticos pequenos
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    # Configurar RandomizedSearchCV
    param_distributions = {
        'n_estimators': randint(10, 50),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 5),
        'min_samples_leaf': randint(1, 3),
        'max_features': ['sqrt', 'log2']
    }
    
    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=5,
        cv=3,
        scoring='r2',
        n_jobs=1,
        random_state=42,
        verbose=1
    )
    
    logger.info("🔧 Iniciando otimização com dados sintéticos...")
    search.fit(X, y)
    logger.info(f"✅ Otimização concluída! Melhor score: {search.best_score_:.4f}")
    logger.info(f"📊 Melhores parâmetros: {search.best_params_}")
    
    return True

def test_random_forest_real_data_sample():
    """Testa RandomForest com uma pequena amostra dos dados reais"""
    logger.info("🧪 Testando RandomForest com amostra dos dados reais...")
    
    try:
        # Carregar uma pequena amostra dos dados reais
        data_path = Path('processed_fx_data/USDJPY/train.parquet')
        if not data_path.exists():
            logger.warning(f"❌ Arquivo não encontrado: {data_path}")
            return False
        
        df = pd.read_parquet(data_path)
        logger.info(f"📊 Dados carregados: {len(df)} linhas")
        
        # Pegar apenas uma pequena amostra
        sample_size = min(1000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        logger.info(f"📊 Usando amostra de {len(df_sample)} linhas")
        
        # Preparar features (usando Target_Return_1 como target)
        if 'Target_Return_1' not in df_sample.columns:
            logger.warning("❌ Coluna 'Target_Return_1' não encontrada")
            return False
        
        # Selecionar features numéricas (excluindo todas as colunas target)
        target_cols = [col for col in df_sample.columns if col.startswith('Target_')]
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remover todas as colunas target das features
        for target_col in target_cols:
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
        
        if len(numeric_cols) == 0:
            logger.warning("❌ Nenhuma feature numérica encontrada")
            return False
        
        X = df_sample[numeric_cols].fillna(0)
        y = df_sample['Target_Return_1'].fillna(0)
        
        logger.info(f"📊 Features: {X.shape[1]}, Amostras: {X.shape[0]}")
        
        # Configurar RandomizedSearchCV com parâmetros muito simples
        param_distributions = {
            'n_estimators': randint(10, 30),
            'max_depth': randint(3, 8),
            'min_samples_split': randint(2, 5),
            'max_features': ['sqrt']
        }
        
        search = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_distributions=param_distributions,
            n_iter=3,  # Muito poucas iterações
            cv=2,      # Apenas 2 folds
            scoring='r2',
            n_jobs=1,
            random_state=42,
            verbose=1
        )
        
        logger.info("🔧 Iniciando otimização com dados reais...")
        search.fit(X, y)
        logger.info(f"✅ Otimização concluída! Melhor score: {search.best_score_:.4f}")
        logger.info(f"📊 Melhores parâmetros: {search.best_params_}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no teste com dados reais: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Iniciando testes do RandomForest...")
    
    # Teste 1: Dados sintéticos
    try:
        success1 = test_random_forest_small()
        logger.info(f"✅ Teste 1 (dados sintéticos): {'SUCESSO' if success1 else 'FALHOU'}")
    except Exception as e:
        logger.error(f"❌ Teste 1 falhou com erro: {e}")
        success1 = False
    
    # Teste 2: Dados reais (amostra)
    try:
        success2 = test_random_forest_real_data_sample()
        logger.info(f"✅ Teste 2 (dados reais): {'SUCESSO' if success2 else 'FALHOU'}")
    except Exception as e:
        logger.error(f"❌ Teste 2 falhou com erro: {e}")
        success2 = False
    
    # Resumo
    logger.info("\n" + "="*50)
    logger.info("📋 RESUMO DOS TESTES")
    logger.info("="*50)
    logger.info(f"🧪 Teste com dados sintéticos: {'✅ PASSOU' if success1 else '❌ FALHOU'}")
    logger.info(f"🧪 Teste com dados reais: {'✅ PASSOU' if success2 else '❌ FALHOU'}")
    
    if success1 and success2:
        logger.info("🎉 Todos os testes passaram! O problema pode ser com o dataset grande.")
    elif success1 and not success2:
        logger.info("⚠️ Problema específico com os dados reais.")
    elif not success1:
        logger.info("❌ Problema fundamental com RandomForest/sklearn.")
    
    logger.info("="*50)