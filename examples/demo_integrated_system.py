#!/usr/bin/env python3
"""
Demonstração do Sistema Integrado de Trading ML
Script para testar e validar todas as funcionalidades implementadas
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Adicionar diretório atual ao path
sys.path.append(str(Path.cwd()))

try:
    from integrated_ml_trading_system import IntegratedMLTradingSystem
except ImportError as e:
    print(f"Erro ao importar sistema integrado: {e}")
    print("Certifique-se de que todos os módulos estão disponíveis")
    sys.exit(1)

def check_data_availability():
    """Verificar se há dados disponíveis"""
    data_path = Path("extracted_fx_data")
    if not data_path.exists():
        print("❌ Pasta extracted_fx_data não encontrada.")
        return False
    
    # Contar arquivos CSV em todas as subpastas
    csv_files = list(data_path.glob("**/*.csv"))
    
    print(f"✅ Encontrados {len(csv_files)} arquivos CSV")
    
    if len(csv_files) == 0:
        print("❌ Dados não disponíveis. Execute o data_loader primeiro.")
        return False
    
    return True

def check_dependencies():
    """Verifica dependências necessárias"""
    print("\n=== VERIFICAÇÃO DE DEPENDÊNCIAS ===")
    
    required_modules = [
        'pandas', 'numpy', 'sklearn', 'joblib', 
        'xgboost', 'lightgbm', 'matplotlib', 'seaborn'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - FALTANDO")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️  Módulos faltando: {', '.join(missing_modules)}")
        print("Execute: pip install", ' '.join(missing_modules))
        return False
    
    return True

def run_quick_test():
    """Executa teste rápido do sistema"""
    print("\n=== TESTE RÁPIDO DO SISTEMA ===")
    
    try:
        # Configuração para teste rápido
        config = {
            'feature_engineering': {
                'lookback_periods': [5, 10],  # Reduzido para teste
                'target_horizon': 1,
                'feature_selection_k': 20  # Reduzido
            },
            'ml_models': {
                'models_to_train': ['random_forest'],  # Apenas um modelo
                'cv_folds': 3,  # Reduzido
                'test_size': 0.3,
                'random_state': 42
            },
            'backtesting': {
                'initial_capital': 10000,  # Capital menor
                'commission': 0.0001,
                'slippage': 0.0001,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04
            },
            'risk_management': {
                'max_portfolio_risk': 0.02,
                'max_position_risk': 0.01,
                'max_correlation': 0.7,
                'lookback_period': 100  # Reduzido
            },
            'trading': {
                'signal_threshold': 0.55,  # Menos restritivo
                'min_prediction_samples': 50,  # Reduzido
                'rebalance_frequency': 'daily'
            }
        }
        
        # Criar sistema com configuração de teste
        system = IntegratedMLTradingSystem(data_path="extracted_fx_data", config=config)
        
        # Testar com apenas um par de moedas
        test_pairs = ['EURUSD']
        
        print(f"Testando com: {test_pairs}")
        
        # Executar análise
        results = system.run_complete_analysis(test_pairs)
        
        # Verificar resultados
        if 'error' in results:
            print(f"❌ Erro durante teste: {results['error']}")
            return False
        
        if 'summary' in results:
            print("✅ Teste concluído com sucesso!")
            print("\n=== RESUMO DO TESTE ===")
            print(results['summary'])
            return True
        else:
            print("⚠️  Teste concluído mas sem resumo")
            return False
            
    except Exception as e:
        print(f"❌ Erro durante teste: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_full_demo():
    """Executa demonstração completa"""
    print("\n=== DEMONSTRAÇÃO COMPLETA ===")
    
    try:
        # Criar sistema com configuração padrão
        system = IntegratedMLTradingSystem()
        
        # Pares principais para análise
        currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        print(f"Analisando pares: {currency_pairs}")
        print("⚠️  Esta análise pode demorar alguns minutos...")
        
        # Executar análise completa
        results = system.run_complete_analysis(currency_pairs)
        
        # Verificar resultados
        if 'error' in results:
            print(f"❌ Erro durante análise: {results['error']}")
            return False
        
        if 'summary' in results:
            print("✅ Análise completa concluída!")
            print(results['summary'])
            
            # Estatísticas adicionais
            print("\n=== ESTATÍSTICAS DETALHADAS ===")
            
            if 'backtest_results' in results:
                for pair, bt_results in results['backtest_results'].items():
                    print(f"\n{pair}:")
                    metrics = bt_results.metrics
                    print(f"  Retorno Anualizado: {metrics.get('annualized_return', 0):.2%}")
                    print(f"  Volatilidade: {metrics.get('volatility', 0):.2%}")
                    print(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
                    print(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
            
            return True
        else:
            print("⚠️  Análise concluída mas sem resumo")
            return False
            
    except Exception as e:
        print(f"❌ Erro durante demonstração: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_performance_report(system, results):
    """Gera relatório de performance detalhado"""
    print("\n=== RELATÓRIO DE PERFORMANCE ===")
    
    if 'backtest_results' not in results:
        print("❌ Resultados de backtesting não disponíveis")
        return
    
    # Criar DataFrame com métricas
    metrics_data = []
    
    for pair, bt_results in results['backtest_results'].items():
        metrics = bt_results.metrics
        metrics_data.append({
            'Pair': pair,
            'Total_Return': metrics.get('total_return', 0),
            'Sharpe_Ratio': metrics.get('sharpe_ratio', 0),
            'Max_Drawdown': metrics.get('max_drawdown', 0),
            'Win_Rate': metrics.get('win_rate', 0),
            'Total_Trades': metrics.get('total_trades', 0),
            'Avg_Trade_Return': metrics.get('avg_trade_return', 0)
        })
    
    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        print("\nMétricas por Par de Moedas:")
        print(df_metrics.to_string(index=False, float_format='%.4f'))
        
        # Salvar relatório
        report_file = Path("results") / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        report_file.parent.mkdir(exist_ok=True)
        df_metrics.to_csv(report_file, index=False)
        print(f"\n📊 Relatório salvo em: {report_file}")

def main():
    """Função principal da demonstração"""
    print("🚀 DEMONSTRAÇÃO DO SISTEMA INTEGRADO DE TRADING ML")
    print("Baseado nos conhecimentos extraídos do livro:")
    print("'Machine Learning for Algorithmic Trading Predictive'")
    print("=" * 60)
    
    # 1. Verificar dependências
    if not check_dependencies():
        print("\n❌ Dependências faltando. Instale os módulos necessários.")
        return
    
    # 2. Verificar dados
    if not check_data_availability():
        print("\n❌ Dados não disponíveis. Execute o data_loader primeiro.")
        return
    
    # 3. Perguntar tipo de teste
    print("\n=== OPÇÕES DE TESTE ===")
    print("1. Teste Rápido (1 par, configuração simplificada)")
    print("2. Demonstração Completa (múltiplos pares, análise completa)")
    
    try:
        choice = input("\nEscolha uma opção (1 ou 2): ").strip()
        
        if choice == '1':
            success = run_quick_test()
        elif choice == '2':
            success = run_full_demo()
        else:
            print("Opção inválida. Executando teste rápido...")
            success = run_quick_test()
        
        if success:
            print("\n🎉 Demonstração concluída com sucesso!")
            print("\n📁 Verifique os seguintes diretórios para resultados:")
            print("  - results/: Relatórios e logs")
            print("  - models/: Modelos treinados")
        else:
            print("\n⚠️  Demonstração concluída com problemas.")
            print("Verifique os logs para mais detalhes.")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstração interrompida pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {str(e)}")

if __name__ == "__main__":
    main()