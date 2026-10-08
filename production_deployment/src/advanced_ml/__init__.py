"""Advanced Machine Learning for Financial Trading.

Este módulo implementa técnicas avançadas de machine learning
especificamente adaptadas para trading algorítmico e análise financeira,
baseado nas metodologias do livro "Advances in Financial Machine Learning"
de Marcos López de Prado e "Machine Learning for Algorithmic Trading"
de Stefan Jansen.

Módulos disponíveis:
- triple_barrier: Triple Barrier Method para rotulagem de dados
- meta_labeling: Meta-Labeling para melhoria de performance
- fractional_differentiation: Diferenciação Fracionária (FFD)
- financial_cv: Cross-Validation financeiro com Purged K-Fold
- feature_engineering: Feature engineering avançado para finanças
- portfolio_optimization: Otimização de portfólio multi-método
- backtesting_engine: Motor de backtesting event-driven
- regularization: Regularização avançada (L1, L2, Elastic Net)
"""

from .triple_barrier import TripleBarrierLabeling
from .meta_labeling import MetaLabeling, MetaLabelingPipeline
from .fractional_differentiation import FractionalDifferentiation, OptimalDifferentiation
from .financial_cv import PurgedKFold, CombinatorialPurgedCV, FinancialCrossValidator
from .feature_engineering import FinancialFeatureEngineer
from .portfolio_optimization import PortfolioOptimizer
from .backtesting_engine import (
    BacktestingEngine, 
    HistoricalDataHandler, 
    SimulatedExecutionHandler,
    BuyAndHoldStrategy,
    MovingAverageCrossoverStrategy,
    Portfolio,
    Order,
    Trade,
    Position
)
from .regularization import AdvancedRegularization, FinancialRegularization

__version__ = "1.0.0"
__author__ = "Advanced ML Financial Team"

__all__ = [
    # Triple Barrier
    'TripleBarrierLabeling',
    
    # Meta-Labeling
    'MetaLabeling',
    'MetaLabelingPipeline',
    
    # Fractional Differentiation
    'FractionalDifferentiation',
    'OptimalDifferentiation',
    
    # Financial Cross-Validation
    'PurgedKFold',
    'CombinatorialPurgedCV',
    'FinancialCrossValidator',
    
    # Feature Engineering
    'FinancialFeatureEngineer',
    
    # Portfolio Optimization
    'PortfolioOptimizer',
    
    # Backtesting
    'BacktestingEngine',
    'HistoricalDataHandler',
    'SimulatedExecutionHandler',
    'BuyAndHoldStrategy',
    'MovingAverageCrossoverStrategy',
    'Portfolio',
    'Order',
    'Trade',
    'Position',
    
    # Regularization
    'AdvancedRegularization',
    'FinancialRegularization'
]