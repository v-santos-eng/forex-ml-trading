#!/usr/bin/env python3
"""
Teste simples para verificar descoberta de pares
"""

import sys
from pathlib import Path
import logging

# Configurar logging simples
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def discover_pairs():
    """Testa descoberta de pares"""
    base_dir = Path(".")
    processed_data_dir = base_dir / "processed_fx_data"
    
    logger.info(f"Verificando diretório: {processed_data_dir}")
    logger.info(f"Diretório existe: {processed_data_dir.exists()}")
    logger.info(f"Caminho absoluto: {processed_data_dir.absolute()}")
    
    pairs = []
    
    if processed_data_dir.exists():
        logger.info("Listando conteúdo do diretório...")
        for pair_dir in processed_data_dir.iterdir():
            logger.info(f"Verificando: {pair_dir.name} (é diretório: {pair_dir.is_dir()})")
            if pair_dir.is_dir():
                test_file = pair_dir / "test.parquet"
                logger.info(f"Arquivo test.parquet existe em {pair_dir.name}: {test_file.exists()}")
                if test_file.exists():
                    pairs.append(pair_dir.name)
                    logger.info(f"Par adicionado: {pair_dir.name}")
    else:
        logger.error(f"Diretório não encontrado: {processed_data_dir}")
    
    logger.info(f"Pares descobertos: {pairs}")
    return sorted(pairs)

if __name__ == "__main__":
    logger.info("Iniciando teste de descoberta de pares...")
    pairs = discover_pairs()
    logger.info(f"Resultado final: {len(pairs)} pares encontrados: {pairs}")