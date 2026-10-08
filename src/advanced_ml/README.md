# Advanced ML for Trading

Um módulo Python completo implementando técnicas avançadas de Machine Learning para trading financeiro, baseado no livro "Advances in Financial Machine Learning" de Marcos López de Prado.

## Instalação

### Dependências Principais

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
pip install ta-lib  # Para indicadores técnicos
pip install statsmodels  # Para testes estatísticos
pip install cvxpy  # Para otimização de portfólio
```

### Instalação do TA-Lib

**Windows:**
```bash
# Baixar wheel apropriado de https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
```

**Linux/Mac:**
```bash
# Instalar dependências do sistema
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

## Uso Rápido

```python
# Importar módulos
from advanced_ml import (
    TripleBarrierLabeling,
    MetaLabeling,
    FractionalDifferentiation,
    PurgedKFold,
    FinancialFeatureEngineer,
    PortfolioOptimizer,
    BacktestingEngine,
    AdvancedRegularization
)

# Exemplo básico
import pandas as pd
import numpy as np

# Dados de exemplo
prices = pd.Series(np.random.randn(1000).cumsum(), 
                  index=pd.date_range('2020-01-01', periods=1000))

# 1. Triple Barrier Labeling
tb = TripleBarrierLabeling()
events = prices.index[::50]  # Eventos a cada 50 períodos
labels = tb.get_bins(prices, events, ptSl=[1, 1])

# 2. Diferenciação Fracionária
ffd = FractionalDifferentiation()
stationary_series = ffd.frac_diff_ffd(prices, d=0.4)

# 3. Cross-Validation Financeiro
cv = PurgedKFold(n_splits=5, pct_embargo=0.01)
```

## Módulos Disponíveis

### 1. Triple Barrier Method (`triple_barrier.py`)

Rotulagem de dados financeiros usando três barreiras:

```python
from advanced_ml import TripleBarrierLabeling

tb = TripleBarrierLabeling()
labels = tb.get_bins(
    prices=prices,
    events=events,
    ptSl=[1, 1],  # Stop loss e take profit
    t1=end_time,  # Horizonte temporal
    trgt=volatility  # Target de volatilidade
)
```

**Principais métodos:**
- `get_events()`: Detecta eventos baseados em filtros
- `get_bins()`: Aplica triple barrier e gera labels
- `_get_daily_vol()`: Calcula volatilidade diária

### 2. Meta-Labeling (`meta_labeling.py`)

Filtragem de predições usando modelo secundário:

```python
from advanced_ml import MetaLabeling

ml = MetaLabeling()
ml.fit(features, meta_labels)
filtered_predictions = ml.predict(features, primary_predictions)
```

**Principais métodos:**
- `prepare_meta_labels()`: Prepara labels para meta-modelo
- `create_meta_features()`: Cria features específicas
- `fit()`: Treina modelo secundário
- `predict()`: Faz predições filtradas

### 3. Diferenciação Fracionária (`fractional_differentiation.py`)

Preserva memória de longo prazo mantendo estacionariedade:

```python
from advanced_ml import FractionalDifferentiation

ffd = FractionalDifferentiation()
optimal_d = ffd.find_optimal_d(prices)
stationary_series = ffd.frac_diff_ffd(prices, d=optimal_d)
```

**Principais métodos:**
- `get_weights()`: Calcula pesos de diferenciação
- `frac_diff_ffd()`: Aplica FFD com janela fixa
- `find_optimal_d()`: Encontra grau ótimo

### 4. Cross-Validation Financeiro (`financial_cv.py`)

Validação cruzada adaptada para dados financeiros:

```python
from advanced_ml import PurgedKFold, CombinatorialPurgedCV

# Purged K-Fold
cv = PurgedKFold(n_splits=5, pct_embargo=0.01)

# Combinatorial Purged CV
cpcv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2)
```

**Características:**
- Previne data leakage
- Aplica embargo temporal
- Respeita dependências temporais

### 5. Feature Engineering (`feature_engineering.py`)

Geração avançada de features financeiras:

```python
from advanced_ml import FinancialFeatureEngineer

fe = FinancialFeatureEngineer()
features = fe.create_all_features(ohlcv_data)
selected = fe.select_features(features, target, k=20)
```

**Tipos de features:**
- Features de preço e retorno
- Indicadores técnicos (TA-Lib)
- Features estatísticas
- Alpha factors
- Features de microestrutura

### 6. Otimização de Portfólio (`portfolio_optimization.py`)

Métodos avançados de otimização:

```python
from advanced_ml import PortfolioOptimizer

opt = PortfolioOptimizer()
opt.load_data(returns_df)

# Mean-Variance
weights_mv, metrics_mv = opt.optimize_mean_variance()

# Risk Parity
weights_rp, metrics_rp = opt.optimize_risk_parity()

# Black-Litterman
weights_bl, metrics_bl = opt.optimize_black_litterman(views, confidences)
```

### 7. Motor de Backtesting (`backtesting_engine.py`)

Backtesting event-driven robusto:

```python
from advanced_ml import BacktestingEngine, MovingAverageCrossoverStrategy

engine = BacktestingEngine(initial_capital=10000)
strategy = MovingAverageCrossoverStrategy('EURUSD', 10, 30)

engine.set_strategy(strategy)
engine.run_backtest()
performance = engine.get_performance_summary()
```

**Componentes:**
- Event-driven architecture
- Controle de data leakage
- Métricas de performance completas
- Suporte a múltiplas estratégias

### 8. Regularização Avançada (`regularization.py`)

Técnicas de regularização para dados financeiros:

```python
from advanced_ml import AdvancedRegularization

reg = AdvancedRegularization(problem_type='classification')
results = reg.adaptive_regularization(X, y)
comparison = reg.compare_methods(results)
```

**Métodos disponíveis:**
- Ridge, Lasso, Elastic Net
- Regularização temporal
- Regularização por regime
- Seleção automática de hiperparâmetros

## Exemplo Completo

Veja `examples/advanced_ml_example.py` para um exemplo completo demonstrando todas as técnicas.

## Estrutura de Arquivos

```
advanced_ml/
├── __init__.py                    # Importações principais
├── triple_barrier.py              # Triple Barrier Method
├── meta_labeling.py               # Meta-Labeling
├── fractional_differentiation.py  # FFD
├── financial_cv.py                # Cross-Validation
├── feature_engineering.py         # Feature Engineering
├── portfolio_optimization.py      # Otimização de Portfólio
├── backtesting_engine.py          # Backtesting
├── regularization.py              # Regularização
└── README.md                      # Este arquivo
```

## Testes

Para executar os testes:

```python
# Teste básico
python examples/advanced_ml_example.py

# Testes individuais
from advanced_ml import TripleBarrierLabeling
tb = TripleBarrierLabeling()
print("Triple Barrier Method carregado com sucesso!")
```

## Considerações de Performance

### Otimizações Implementadas

1. **Processamento Paralelo**: Métodos computacionalmente intensivos suportam multiprocessing
2. **Vectorização NumPy**: Operações otimizadas para arrays
3. **Caching**: Resultados intermediários são cached quando apropriado
4. **Memory Management**: Uso eficiente de memória para datasets grandes

### Recomendações de Uso

- Para datasets > 100k observações, use `n_jobs=-1` nos métodos que suportam paralelização
- Configure `chunk_size` apropriadamente para sua RAM disponível
- Use `thres` parameter em FFD para controlar trade-off precisão/velocidade

## Troubleshooting

### Problemas Comuns

**1. Erro de importação TA-Lib:**
```
ImportError: No module named 'talib'
```
**Solução:** Instalar TA-Lib conforme instruções acima.

**2. Erro de convergência em otimização:**
```
OptimizationError: Solver failed to converge
```
**Solução:** Verificar dados de entrada e ajustar parâmetros do solver.

**3. Memory Error em datasets grandes:**
```
MemoryError: Unable to allocate array
```
**Solução:** Reduzir `chunk_size` ou processar dados em batches.

### Logs e Debug

Para ativar logging detalhado:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Agora todos os módulos mostrarão informações detalhadas
```

## Contribuição

Para contribuir com o projeto:

1. Fork o repositório
2. Crie uma branch para sua feature
3. Implemente testes para novas funcionalidades
4. Siga PEP 8 para style guide
5. Submeta um pull request

## Licença

Este projeto é distribuído sob licença MIT. Veja LICENSE para mais detalhes.

## Referências

- López de Prado, M. (2018). "Advances in Financial Machine Learning"
- Jansen, S. (2020). "Machine Learning for Algorithmic Trading"
- Chan, E. (2013). "Algorithmic Trading: Winning Strategies and Their Rationale"

## Suporte

Para dúvidas e suporte:
- Consulte a documentação técnica completa
- Verifique os exemplos em `examples/`
- Abra uma issue no repositório para bugs ou sugestões