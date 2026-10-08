#!/usr/bin/env python3
"""
Script: run_batch_hybrid_optimization.py
Descrição:
    Executa otimização de hiperparâmetros em lote utilizando o novo método
    HYBRID_RS_BO para todos os pares disponíveis em processed_fx_data.
    Salva relatório JSON detalhado com resultados individuais e agregados.
Requisitos:
    - Pandas (para leitura de Parquet)
    - production_deployment.hyperparameter_optimizer disponível no PYTHONPATH
Execução:
    python run_batch_hybrid_optimization.py
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Importa classes do otimizador
from production_deployment.hyperparameter_optimizer import (
    HyperparameterOptimizer,
    OptimizationConfig,
    ModelType,
    OptimizationMethod,
)

# Configura logger simples
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("batch_hybrid_optimizer")

# Diretório de dados processados
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_fx_data"
RESULTS_DIR = BASE_DIR / "batch_optimization_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Verifica disponibilidade de dados
if not DATA_DIR.exists():
    logger.error("Diretório processed_fx_data não encontrado: %s", DATA_DIR)
    raise SystemExit(1)

pairs = [p.name for p in DATA_DIR.iterdir() if p.is_dir()]
if not pairs:
    logger.error("Nenhum par encontrado em processed_fx_data.")
    raise SystemExit(1)

logger.info("Pares detectados para otimização: %s", ", ".join(pairs))

# Instancia otimizador
optimizer = HyperparameterOptimizer()

# Resultado agregado
batch_results = {}

TARGET_COLUMN = "Target_Direction_1"  # Ajuste caso utilize outra métrica
N_TRIALS = 50  # Total de tentativas (será dividido internamente entre RS e BO)
TIMEOUT_SECONDS = 1800  # 30 minutos por par (ajuste conforme necessário)

for idx, pair in enumerate(pairs, 1):
    logger.info("[%d/%d] Iniciando otimização para %s", idx, len(pairs), pair)
    pair_dir = DATA_DIR / pair
    train_file = pair_dir / "train.parquet"

    if not train_file.exists():
        logger.warning("Arquivo de treino não encontrado para %s, pulando.", pair)
        batch_results[pair] = {"status": "missing_train_data"}
        continue

    # Carrega dados
    try:
        df = pd.read_parquet(train_file)
    except Exception as exc:
        logger.exception("Falha ao carregar Parquet de %s: %s", pair, exc)
        batch_results[pair] = {"status": "load_error", "error": str(exc)}
        continue

    if TARGET_COLUMN not in df.columns:
        logger.warning("Coluna alvo %s ausente em %s, pulando.", TARGET_COLUMN, pair)
        batch_results[pair] = {"status": "missing_target"}
        continue

    # Separa features e target
    y = df[TARGET_COLUMN].astype(int).values  # target binário/direcional
    # Filtra somente colunas numéricas para evitar erros (ex.: Timestamps)
    numeric_df = df.drop(columns=[TARGET_COLUMN]).select_dtypes(include=["number"])
    if numeric_df.empty:
        logger.warning("Nenhuma coluna numérica após filtragem para %s, pulando.", pair)
        batch_results[pair] = {"status": "no_numeric_columns"}
        continue
    X = numeric_df.values.astype(np.float32)

    if len(y) < 100:
        logger.warning("Dados insuficientes(%d linhas) para %s, pulando.", len(y), pair)
        batch_results[pair] = {"status": "insufficient_data", "samples": int(len(y))}
        continue

    # Configuração da otimização
    config = OptimizationConfig(
        model_type=ModelType.RANDOM_FOREST,
        method=OptimizationMethod.HYBRID_RS_BO,
        n_trials=N_TRIALS,
        timeout_seconds=TIMEOUT_SECONDS,
    )

    try:
        result = optimizer.optimize_model(
            pair,
            X,
            y,
            ModelType.RANDOM_FOREST,
            config,
        )

        batch_results[pair] = {
            "status": "success",
            "best_score": result.best_score,
            "best_params": result.best_params,
            "n_trials": result.n_trials,
            "optimization_time": result.optimization_time,
            "timestamp": result.timestamp.isoformat(),
        }
        logger.info(
            "✔ %s otimizado | score=%.4f | trials=%d | tempo=%.1fs",
            pair,
            result.best_score,
            result.n_trials,
            result.optimization_time,
        )

    except Exception as exc:
        logger.exception("Falha na otimização de %s: %s", pair, exc)
        batch_results[pair] = {"status": "opt_error", "error": str(exc)}

# Salva relatório
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = RESULTS_DIR / f"batch_hybrid_optimization_report_{timestamp}.json"
with open(report_path, "w", encoding="utf-8") as fp:
    json.dump(batch_results, fp, indent=2, ensure_ascii=False)

logger.info("Relatório salvo em %s", report_path)

# Resumo final
success = sum(1 for r in batch_results.values() if r.get("status") == "success")
logger.info("Processo concluído: %d/%d pares otimizados com sucesso.", success, len(pairs))