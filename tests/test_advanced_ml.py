#!/usr/bin/env python3
"""
Script de teste simplificado para validar as importações do módulo advanced_ml.
Este script verifica se todos os módulos podem ser importados corretamente.
"""

import sys
import os
import warnings
import traceback
from pathlib import Path
import pandas as pd

# Adicionar o diretório src ao path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Suprimir warnings para output mais limpo
warnings.filterwarnings('ignore')

def test_imports():
    """Testa se todos os módulos podem ser importados."""
    print("=== TESTE DE IMPORTAÇÕES ===")
    
    modules_to_test = [
        ('TripleBarrierLabeling', 'advanced_ml.triple_barrier'),
        ('MetaLabeling', 'advanced_ml.meta_labeling'),
        ('FractionalDifferentiation', 'advanced_ml.fractional_differentiation'),
        ('PurgedKFold', 'advanced_ml.financial_cv'),
        ('CombinatorialPurgedCV', 'advanced_ml.financial_cv'),
        ('FinancialCrossValidator', 'advanced_ml.financial_cv'),
        ('FinancialFeatureEngineer', 'advanced_ml.feature_engineering'),
        ('PortfolioOptimizer', 'advanced_ml.portfolio_optimization'),
        ('BacktestingEngine', 'advanced_ml.backtesting_engine'),
        ('AdvancedRegularization', 'advanced_ml.regularization'),
        ('FinancialRegularization', 'advanced_ml.regularization')
    ]
    
    passed = 0
    failed = 0
    
    for class_name, module_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✓ {class_name} importado com sucesso")
            passed += 1
        except Exception as e:
            print(f"✗ Erro ao importar {class_name}: {e}")
            failed += 1
    
    return passed, failed

def test_main_import():
    """Testa a importação principal do módulo."""
    print("\n=== TESTE IMPORTAÇÃO PRINCIPAL ===")
    
    try:
        from advanced_ml import (
            TripleBarrierLabeling,
            MetaLabeling,
            FractionalDifferentiation,
            PurgedKFold,
            CombinatorialPurgedCV,
            FinancialCrossValidator,
            FinancialFeatureEngineer,
            PortfolioOptimizer,
            BacktestingEngine,
            AdvancedRegularization,
            FinancialRegularization
        )
        print("✓ Importação principal bem-sucedida")
        return 1, 0
    except Exception as e:
        print(f"✗ Erro na importação principal: {e}")
        traceback.print_exc()
        return 0, 1

def test_basic_instantiation():
    """Testa a instanciação básica das classes."""
    print("\n=== TESTE INSTANCIAÇÃO BÁSICA ===")
    
    classes_to_test = [
        ('TripleBarrierLabeling', lambda: TripleBarrierLabeling(pd.Series([100, 101, 102], index=pd.date_range('2020-01-01', periods=3)))),
        ('MetaLabeling', lambda: MetaLabeling()),
        ('FractionalDifferentiation', lambda: FractionalDifferentiation()),
        ('PurgedKFold', lambda: PurgedKFold(n_splits=3)),
        ('CombinatorialPurgedCV', lambda: CombinatorialPurgedCV(n_splits=4, n_test_splits=2)),
        ('FinancialCrossValidator', lambda: FinancialCrossValidator()),
        ('FinancialFeatureEngineer', lambda: FinancialFeatureEngineer()),
        ('PortfolioOptimizer', lambda: PortfolioOptimizer()),
        ('BacktestingEngine', lambda: BacktestingEngine(initial_capital=10000)),
        ('AdvancedRegularization', lambda: AdvancedRegularization()),
        ('FinancialRegularization', lambda: FinancialRegularization())
    ]
    
    passed = 0
    failed = 0
    
    # Importar as classes necessárias
    try:
        from advanced_ml import (
            TripleBarrierLabeling,
            MetaLabeling,
            FractionalDifferentiation,
            PurgedKFold,
            CombinatorialPurgedCV,
            FinancialCrossValidator,
            FinancialFeatureEngineer,
            PortfolioOptimizer,
            BacktestingEngine,
            AdvancedRegularization,
            FinancialRegularization
        )
    except Exception as e:
        print(f"✗ Erro ao importar classes para teste de instanciação: {e}")
        return 0, len(classes_to_test)
    
    for class_name, instantiate_func in classes_to_test:
        try:
            instance = instantiate_func()
            print(f"✓ {class_name} instanciado com sucesso")
            passed += 1
        except Exception as e:
            print(f"✗ Erro ao instanciar {class_name}: {e}")
            failed += 1
    
    return passed, failed

def test_dependencies():
    """Testa se as dependências principais estão disponíveis."""
    print("\n=== TESTE DEPENDÊNCIAS ===")
    
    dependencies = [
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'matplotlib',
        'seaborn',
        'cvxpy'
    ]
    
    passed = 0
    failed = 0
    
    for dep in dependencies:
        try:
            if dep == 'scikit-learn':
                import sklearn
                print(f"✓ {dep} disponível (versão {sklearn.__version__})")
            else:
                module = __import__(dep)
                version = getattr(module, '__version__', 'desconhecida')
                print(f"✓ {dep} disponível (versão {version})")
            passed += 1
        except ImportError:
            print(f"✗ {dep} não disponível")
            failed += 1
        except Exception as e:
            print(f"⚠ {dep} disponível mas com aviso: {e}")
            passed += 1
    
    return passed, failed

def main():
    """Executa todos os testes."""
    print("INICIANDO TESTES SIMPLIFICADOS DO MÓDULO ADVANCED_ML")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    # Teste de dependências
    passed, failed = test_dependencies()
    total_passed += passed
    total_failed += failed
    
    # Teste de importações individuais
    passed, failed = test_imports()
    total_passed += passed
    total_failed += failed
    
    # Teste de importação principal
    passed, failed = test_main_import()
    total_passed += passed
    total_failed += failed
    
    # Teste de instanciação básica
    passed, failed = test_basic_instantiation()
    total_passed += passed
    total_failed += failed
    
    print("\n" + "=" * 60)
    print(f"RESUMO FINAL DOS TESTES:")
    print(f"✓ Testes aprovados: {total_passed}")
    print(f"✗ Testes falharam: {total_failed}")
    print(f"Total: {total_passed + total_failed}")
    
    if total_failed == 0:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("O módulo advanced_ml está pronto para uso.")
        print("\nPróximos passos:")
        print("1. Execute 'python examples/advanced_ml_example.py' para ver exemplos")
        print("2. Consulte 'src/advanced_ml/README.md' para documentação")
        print("3. Veja 'ADVANCED_ML_DOCUMENTATION.md' para teoria detalhada")
        return True
    else:
        print(f"\n⚠️  {total_failed} teste(s) falharam.")
        print("Verifique as dependências e instalações necessárias.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)