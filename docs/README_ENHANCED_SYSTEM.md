# Sistema ML Forex Melhorado com Agno/Ollama

## Visão Geral

Este sistema implementa todas as melhorias solicitadas mantendo a integração com Agno/Ollama:

**1. Ensemble Methods (XGBoost, Random Forest)**  
**2. Walk-Forward Optimization**  
**3. PCA para Redução de Dimensionalidade**  
**4. Seleção Automática de Features**  
**5. Prevenção de Overfitting**  
**6. Integração com Agno/Ollama**  

## Arquivos Principais

### Novos Arquivos Criados
- `ml_predictor_enhanced.py` - Sistema ML melhorado com todas as funcionalidades
- `agno_ml_enhanced.py` - Integração do sistema melhorado com Agno/Ollama
- `demo_enhanced_system.py` - Demonstrações de uso do sistema
- `README_ENHANCED_SYSTEM.md` - Esta documentação

### Arquivos Existentes (Mantidos)
- `ml_predictor_free.py` - Sistema ML básico original
- `agno_ollama_agents.py` - Agentes Ollama
- `agno_ollama_memory.py` - Sistema de memória
- `demo_agno_ollama.py` - Demo do sistema original

## Funcionalidades Implementadas

### 1. Ensemble Methods
- **XGBoost**: Gradient boosting otimizado
- **Random Forest**: Múltiplas árvores de decisão
- **Gradient Boosting**: Boosting sequencial
- **Logistic Regression**: Modelo linear probabilístico
- **SVM**: Support Vector Machine
- **Ensemble Voting**: Combinação inteligente dos modelos

### 2. Walk-Forward Optimization
- Validação temporal realística
- Janelas deslizantes de treino/teste
- Prevenção de look-ahead bias
- Métricas de performance ao longo do tempo

### 3. PCA (Principal Component Analysis)
- Redução de dimensionalidade inteligente
- Preservação de variância configurável (padrão: 95%)
- Remoção de multicolinearidade
- Melhoria da generalização

### 4. Seleção Automática de Features
- **Recursive Feature Elimination (RFE)**
- **Mutual Information**
- **SelectKBest com f_classif**
- **Seleção automática** baseada na performance

### 5. Prevenção de Overfitting
- Cross-validation estratificado
- Regularização nos modelos
- Early stopping no XGBoost
- Validação temporal
- Threshold para ruído

### 6. Integração Agno/Ollama
- Análise técnica com Ollama
- Análise fundamental com Ollama
- Análise de risco com Ollama
- Combinação de confiança ML + Ollama
- Sistema de memória integrado

## Como Usar

### Uso Básico
```python
from agno_ml_enhanced import AgnoMLEnhanced

# Criar sistema
system = AgnoMLEnhanced(
    use_pca=True,
    n_components=0.95,
    feature_selection_method='auto'
)

# Treinar
pairs = ['EURUSD=X', 'GBPUSD=X']
training_results = system.train_enhanced_system(pairs, period='1y')

# Predição
prediction = system.enhanced_prediction('EURUSD=X')
print(f"Direção: {prediction['ml_prediction']['direction']}")
print(f"Confiança: {prediction['confidence_score']}%")
```

### Configurações Avançadas
```python
# Sistema sem PCA
system_no_pca = AgnoMLEnhanced(
    use_pca=False,
    feature_selection_method='mutual_info'
)

# Sistema com PCA fixo
system_fixed_pca = AgnoMLEnhanced(
    use_pca=True,
    n_components=10,  # 10 componentes fixos
    feature_selection_method='rfe'
)
```

### Executar Demonstrações
```bash
# Demo completo do sistema integrado
python agno_ml_enhanced.py

# Demonstrações específicas
python demo_enhanced_system.py
```

## Resultados e Métricas

### Exemplo de Saída
```
EURUSD=X:
   Direção: ALTA
   Confiança ML: 56.8%
   Confiança Combinada: 56.8%
   Modelo: ensemble_enhanced

GBPUSD=X:
   Direção: ALTA
   Confiança ML: 66.4%
   Confiança Combinada: 66.4%
   Modelo: ensemble_enhanced
```

### Métricas do Sistema
- **Features Totais**: 78 indicadores técnicos
- **Features Selecionadas**: 12-19 (otimização automática)
- **Componentes PCA**: 1-5 (baseado na variância)
- **Modelos Ensemble**: 5 algoritmos diferentes
- **Variância Explicada**: ~97% (PCA)

## Comparação com Sistema Básico

| Funcionalidade | Sistema Básico | Sistema Melhorado |
|---|---|---|
| Modelos | 4 básicos | 5 + Ensemble |
| Features | 26 fixas | 78 + Seleção automática |
| Validação | Split simples | Walk-Forward + CV |
| Dimensionalidade | Sem redução | PCA opcional |
| Overfitting | Básico | Múltiplas técnicas |
| Integração Ollama | Sim | Sim (melhorada) |

## Melhorias de Performance

### Técnicas Anti-Overfitting
1. **Cross-Validation Estratificado**: 5-fold CV
2. **Regularização**: L1/L2 nos modelos lineares
3. **Early Stopping**: XGBoost com parada antecipada
4. **Threshold de Ruído**: Filtrar movimentos insignificantes
5. **Validação Temporal**: Walk-forward testing

### Otimizações de Features
1. **78 Indicadores Técnicos**: SMAs, EMAs, RSI, MACD, Bollinger, etc.
2. **Features de Lag**: Valores históricos
3. **Features de Interação**: Combinações de indicadores
4. **Seleção Automática**: Melhor subset de features
5. **PCA**: Redução de dimensionalidade inteligente

## Dependências

### Obrigatórias
```
pandas
numpy
scikit-learn
yfinance
```

### Opcionais (Melhoram Performance)
```
xgboost  # Para XGBoost ensemble
ta-lib   # Para indicadores técnicos avançados
```

### Agno/Ollama
```
# Componentes do sistema original
agno_ollama_agents.py
agno_ollama_memory.py
```

## Configuração

### Parâmetros Principais
- `use_pca`: Usar PCA (True/False)
- `n_components`: Componentes PCA (float 0-1 ou int)
- `feature_selection_method`: 'auto', 'rfe', 'mutual_info', 'kbest'

### Exemplo de Configuração Personalizada
```python
system = AgnoMLEnhanced(
    use_pca=True,
    n_components=0.99,  # 99% da variância
    feature_selection_method='rfe'  # Recursive Feature Elimination
)
```

## Relatórios

O sistema gera relatórios JSON detalhados:
- Resultados de treinamento
- Métricas de performance
- Predições com confiança
- Comparações entre sistemas
- Informações do modelo

## Status das Melhorias

**TODAS AS MELHORIAS IMPLEMENTADAS E FUNCIONANDO:**

1. **Ensemble Methods**: XGBoost, Random Forest, Gradient Boosting
2. **Walk-Forward Optimization**: Validação temporal realística
3. **PCA**: Redução de dimensionalidade com preservação de variância
4. **Seleção Automática de Features**: RFE, Mutual Info, KBest
5. **Prevenção de Overfitting**: CV, regularização, early stopping
6. **Integração Agno/Ollama**: Mantida e melhorada

## Próximos Passos

1. **Testar em produção** com dados reais
2. **Ajustar hiperparâmetros** para pares específicos
3. **Implementar mais indicadores** técnicos
4. **Adicionar análise de sentimento** via Ollama
5. **Criar interface web** para visualização

---

**Sistema pronto para uso avançado em trading de forex!**