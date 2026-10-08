#!/usr/bin/env python3
"""
Forex ML Trading System - Production Deployment
Script principal para inicialização do sistema de trading com ML

Autor: Sistema ML Forex
Versão: 1.0.0
Data: 2025-08-18
"""

import sys
import os
from pathlib import Path

# Adicionar diretórios ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'scripts'))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'config'))

def main():
    """
    Função principal para inicializar o sistema de trading
    """
    print("=" * 60)
    print("FOREX ML TRADING SYSTEM - PRODUCTION")
    print("=" * 60)
    
    try:
        # Importar sistema integrado
        from integrated_ml_trading_system import IntegratedMLTradingSystem
        
        print("✓ Módulos carregados com sucesso")
        
        # Inicializar sistema
        trading_system = IntegratedMLTradingSystem()
        
        print("✓ Sistema de trading inicializado")
        print("\nSistema pronto para operação!")
        print("\nPara executar análises:")
        print("- trading_system.run_analysis('EURUSD')")
        print("- trading_system.run_backtest('GBPUSD')")
        
        return trading_system
        
    except ImportError as e:
        print(f"❌ Erro ao importar módulos: {e}")
        print("Verifique se todas as dependências estão instaladas:")
        print("pip install -r requirements.txt")
        return None
        
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        return None

if __name__ == "__main__":
    system = main()