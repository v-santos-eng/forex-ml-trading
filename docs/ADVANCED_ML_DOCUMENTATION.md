# Documentação Técnica: Advanced Machine Learning for Trading

## Visão Geral

Este documento apresenta uma implementação completa de técnicas avançadas de Machine Learning para trading financeiro, baseadas no livro "Advances in Financial Machine Learning" de Marcos López de Prado e outras fontes acadêmicas de referência.

## Estrutura do Projeto

```
src/advanced_ml/
├── __init__.py                    # Módulo principal com importações
├── triple_barrier.py              # Triple Barrier Method
├── meta_labeling.py               # Meta-Labeling
├── fractional_differentiation.py  # Diferenciação Fracionária (FFD)
├── financial_cv.py                # Cross-Validation Financeiro
├── feature_engineering.py         # Feature Engineering Avançado
├── portfolio_optimization.py      # Otimização de Portfólio
├── backtesting_engine.py          # Motor de Backtesting
└── regularization.py              # Regularização Avançada
```

## 1. Triple Barrier Method

### Teoria

O Triple Barrier Method é uma técnica de rotulagem que define três barreiras para cada evento:
- **Upper Barrier**: Take profit (ganho alvo)
- **Lower Barrier**: Stop loss (perda máxima)
- **Vertical Barrier**: Horizonte temporal máximo

O label é determinado pela primeira barreira tocada:
- `1`: Upper barrier (compra lucrativa)
- `-1`: Lower barrier (compra não lucrativa)
- `0`: Vertical barrier (resultado inconclusivo)

### Implementação

```python
from advanced_ml import TripleBarrierLabeling

# Inicializar
tb = TripleBarrierLabeling()

# Detectar eventos (exemplo: breakouts)
events = prices[prices > sma_20.shift(1)].index

# Aplicar Triple Barrier
labels = tb.get_bins(
    prices=prices,
    events=events,
    ptSl=[1, 1],  # Stop loss e take profit iguais
    t1=prices.index[-1],  # Horizonte máximo
    trgt=tb._get_daily_vol(prices, span=20)  # Volatilidade como target
)
```

### Vantagens
- Evita look-ahead bias
- Considera tanto magnitude quanto direção
- Permite análise de holding periods variáveis
- Reduz ruído em labels

## 2. Meta-Labeling

### Teoria

Meta-labeling é uma técnica que usa um modelo secundário para filtrar as predições de um modelo primário, melhorando a precisão e o F1-score. O modelo primário decide a direção (compra/venda) e o modelo secundário decide se deve executar a operação.

### Implementação

```python
from advanced_ml import MetaLabeling

# Inicializar
ml = MetaLabeling()

# Preparar meta-labels
meta_labels = ml.prepare_meta_labels(labels, predictions_primary)

# Treinar modelo secundário
ml.fit(features, meta_labels)

# Fazer predições filtradas
filtered_predictions = ml.predict(features, predictions_primary)
```

### Benefícios
- Melhora F1-score significativamente
- Reduz falsos positivos
- Aumenta Sharpe ratio
- Permite especialização de modelos

## 3. Diferenciação Fracionária (FFD)

### Teoria

A diferenciação fracionária generaliza a diferenciação inteira, permitindo preservar memória de longo prazo enquanto mantém estacionariedade. Para um grau `d`:

```
X_t^(d) = Σ(k=0 to ∞) w_k * X_{t-k}
```

Onde os pesos seguem:
```
w_k = (-1)^k * Γ(d+1) / (Γ(k+1) * Γ(d-k+1))
```

### Implementação

```python
from advanced_ml import FractionalDifferentiation

# Inicializar
ffd = FractionalDifferentiation()

# Encontrar grau ótimo
optimal_d = ffd.find_optimal_d(prices, max_d=1.0)

# Aplicar FFD
ffd_series = ffd.frac_diff_ffd(prices, d=optimal_d, thres=0.01)
```

### Aplicações
- Preservar informação preditiva
- Manter estacionariedade
- Melhorar performance de modelos ML
- Análise de séries com memória longa

## 4. Cross-Validation Financeiro

### Problema com CV Tradicional

O K-Fold tradicional falha em dados financeiros devido a:
- **Data leakage**: Informações futuras influenciam predições passadas
- **Overlapping labels**: Labels podem se sobrepor temporalmente
- **Temporal dependencies**: Violação da independência dos dados

### Solução: Purged K-Fold CV

```python
from advanced_ml import PurgedKFold

# Inicializar com embargo
cv = PurgedKFold(n_splits=5, pct_embargo=0.01)

# Usar com sklearn
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=cv)
```

### Características
- **Purging**: Remove observações que podem causar leakage
- **Embargo**: Adiciona gap temporal entre treino e teste
- **Temporal awareness**: Respeita ordem cronológica

## 5. Feature Engineering Avançado

### Categorias de Features

#### 5.1 Features de Preço
- Returns (1d, 5d, 20d)
- Log returns
- Volatilidade rolling
- Price momentum

#### 5.2 Indicadores Técnicos (TA-Lib)
- RSI, MACD, Bollinger Bands
- Stochastic, Williams %R
- ADX, CCI, MFI

#### 5.3 Features Estatísticas
- Skewness e Kurtosis rolling
- Sharpe ratio rolling
- Hurst Exponent
- Autocorrelação

#### 5.4 Alpha Factors
- Mean reversion
- Momentum cross-sectional
- Residual momentum

### Implementação

```python
from advanced_ml import FinancialFeatureEngineer

# Inicializar
fe = FinancialFeatureEngineer()

# Gerar todas as features
features = fe.create_all_features(ohlcv_data)

# Seleção de features
selected = fe.select_features(features, target, method='mutual_info', k=20)
```

## 6. Otimização de Portfólio

### Métodos Implementados

#### 6.1 Mean-Variance (Markowitz)
- Maximização do Sharpe ratio
- Fronteira eficiente
- Restrições de peso

#### 6.2 Black-Litterman
- Incorpora views de mercado
- Bayesian approach
- Reduz instabilidade

#### 6.3 Risk Parity
- Contribuição de risco igual
- Volatilidade inversa
- Diversificação máxima

### Implementação

```python
from advanced_ml import PortfolioOptimizer

# Inicializar
opt = PortfolioOptimizer()
opt.load_data(returns_df)

# Otimização Mean-Variance
weights, metrics = opt.optimize_mean_variance(target_return=0.1)

# Risk Parity
weights_rp, metrics_rp = opt.optimize_risk_parity()
```

## 7. Motor de Backtesting

### Arquitetura Event-Driven

```
DataHandler → Strategy → Portfolio → ExecutionHandler
     ↓           ↓          ↓            ↓
   Market     Signals   Position    Order
    Data               Management  Execution
```

### Componentes Principais

#### 7.1 Handlers
- **DataHandler**: Fornece dados históricos
- **ExecutionHandler**: Simula execução de ordens
- **Strategy**: Implementa lógica de trading

#### 7.2 Estruturas de Dados
- **Order**: Representa ordens de compra/venda
- **Trade**: Registra trades executados
- **Position**: Gerencia posições abertas
- **Portfolio**: Controla capital e performance

### Implementação

```python
from advanced_ml import BacktestingEngine, MovingAverageCrossoverStrategy

# Configurar engine
engine = BacktestingEngine(initial_capital=10000)

# Definir estratégia
strategy = MovingAverageCrossoverStrategy(
    symbol='EURUSD',
    short_window=10,
    long_window=30
)

# Executar backtest
engine.set_strategy(strategy)
engine.run_backtest()

# Analisar resultados
performance = engine.get_performance_summary()
```

### Métricas Calculadas
- Total Return, Annual Return
- Volatilidade, Sharpe Ratio
- Maximum Drawdown, Calmar Ratio
- Win Rate, Profit Factor

## 8. Regularização Avançada

### Técnicas Implementadas

#### 8.1 Regularização Clássica
- **Ridge (L2)**: Reduz magnitude dos coeficientes
- **Lasso (L1)**: Seleção automática de features
- **Elastic Net**: Combina L1 e L2

#### 8.2 Regularização Financeira
- **Time Series Regularization**: Adaptada para séries temporais
- **Regime-Aware Regularization**: Considera regimes de mercado

### Implementação

```python
from advanced_ml import AdvancedRegularization

# Inicializar
reg = AdvancedRegularization(problem_type='regression')

# Regularização adaptativa
results = reg.adaptive_regularization(X, y)

# Comparar métodos
comparison = reg.compare_methods(results)

# Análise de importância
importance = reg.feature_importance_analysis(results['lasso'])
```

## Exemplo Completo de Uso

```python
# Importar módulos
from advanced_ml import (
    TripleBarrierLabeling,
    FractionalDifferentiation,
    FinancialFeatureEngineer,
    PurgedKFold,
    AdvancedRegularization
)

# 1. Preparar dados
prices = data['close']
ffd = FractionalDifferentiation()
stationary_prices = ffd.frac_diff_ffd(prices, d=0.4)

# 2. Gerar labels
tb = TripleBarrierLabeling()
events = detect_events(prices)  # Sua lógica de eventos
labels = tb.get_bins(prices, events, ptSl=[1, 1])

# 3. Feature engineering
fe = FinancialFeatureEngineer()
features = fe.create_all_features(data)
selected_features = fe.select_features(features, labels, k=20)

# 4. Cross-validation
cv = PurgedKFold(n_splits=5, pct_embargo=0.01)

# 5. Modelagem com regularização
reg = AdvancedRegularization()
results = reg.adaptive_regularization(selected_features, labels)

# 6. Avaliação
from sklearn.model_selection import cross_val_score
scores = cross_val_score(results['best_model'], selected_features, labels, cv=cv)
print(f"CV Score: {scores.mean():.4f} ± {scores.std():.4f}")
```

## Considerações Importantes

### 1. Data Leakage
- Sempre usar Purged K-Fold CV
- Cuidado com features que olham para o futuro
- Validar pipeline completo

### 2. Overfitting
- Usar regularização apropriada
- Validação out-of-sample rigorosa
- Monitorar performance em produção

### 3. Regime Changes
- Considerar mudanças estruturais
- Retreinar modelos periodicamente
- Usar técnicas robustas

### 4. Transaction Costs
- Incluir custos no backtesting
- Considerar slippage e spread
- Otimizar frequência de trading

## Referências

1. López de Prado, M. (2018). "Advances in Financial Machine Learning". Wiley.
2. Jansen, S. (2020). "Machine Learning for Algorithmic Trading". Packt.
3. Chan, E. (2013). "Algorithmic Trading: Winning Strategies and Their Rationale". Wiley.
4. Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio". Journal of Portfolio Management.

## Conclusão

Esta implementação fornece um framework completo para aplicação de técnicas avançadas de ML em trading financeiro. Cada módulo foi desenvolvido seguindo as melhores práticas acadêmicas e industriais, com foco em:

- **Rigor científico**: Implementações baseadas em literatura peer-reviewed
- **Robustez**: Tratamento adequado de data leakage e overfitting
- **Flexibilidade**: Módulos independentes e reutilizáveis
- **Performance**: Código otimizado para dados financeiros

O framework está pronto para uso em pesquisa quantitativa e desenvolvimento de estratégias de trading sistemático.