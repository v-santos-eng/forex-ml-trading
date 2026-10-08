# Forex Analyzer - Versão Gratuita

Esta é uma versão completamente gratuita do sistema de análise de forex que não requer APIs pagas. Utiliza apenas bibliotecas open-source e dados gratuitos do Yahoo Finance.

## Características da Versão Gratuita

### O que está incluído:
- Análise técnica completa com 15+ indicadores
- Machine Learning com múltiplos algoritmos
- Dados históricos gratuitos via Yahoo Finance
- Predições de direção do mercado
- Sinais de trading automatizados
- Backtesting básico
- Análise de importância de features

### O que NÃO está incluído:
- Análise fundamental via AI (requer API paga)
- Análise de sentimento de notícias
- Dados em tempo real de alta frequência
- Análise de texto avançada

## Instalação

### Pré-requisitos
- Python 3.8 ou superior
- Conexão com internet (para baixar dados)

### Instalação das Dependências

```bash
# Instalar dependências básicas
pip install -r requirements_free.txt
```

### Instalação do TA-Lib (Opcional, mas Recomendado)

O TA-Lib fornece indicadores técnicos mais precisos:

**Windows:**
```bash
pip install TA-Lib
```

**Linux/Ubuntu:**
```bash
sudo apt-get install build-essential
pip install TA-Lib
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Alternativa (se TA-Lib não funcionar):**
```bash
pip install ta
```

## Como Usar

### 1. Demonstração Rápida

```bash
python demo_free_version.py
```

Este comando executa uma análise completa do EUR/USD incluindo:
- Análise técnica com indicadores
- Predição com Machine Learning
- Sinais de trading
- Recomendações

### 2. Análise Técnica Personalizada

```python
from forex_analyzer_free import ForexAnalyzer

# Criar analisador
analyzer = ForexAnalyzer()

# Obter dados
data = analyzer.get_data("EURUSD=X", "6mo")

# Calcular indicadores
indicators = analyzer.calculate_indicators(data)

# Gerar sinais
signals = analyzer.generate_signals(data, indicators)

print("Sinais de Trading:", signals)
```

### 3. Predição com Machine Learning

```python
from ml_predictor_free import MLForexPredictor

# Criar preditor
predictor = MLForexPredictor()

# Preparar dados
data = predictor.prepare_data("GBPUSD=X", "1y")

# Treinar e prever
results = predictor.train_and_predict(data)

print(f"Predição: {results['prediction']}")
print(f"Confiança: {results['confidence']:.2%}")
```

## Pares de Moedas Disponíveis

Todos os principais pares estão disponíveis via Yahoo Finance:

| Símbolo | Par de Moedas |
|---------|---------------|
| EURUSD=X | EUR/USD |
| GBPUSD=X | GBP/USD |
| USDJPY=X | USD/JPY |
| USDCHF=X | USD/CHF |
| AUDUSD=X | AUD/USD |
| USDCAD=X | USD/CAD |
| NZDUSD=X | NZD/USD |
| EURGBP=X | EUR/GBP |
| EURJPY=X | EUR/JPY |
| GBPJPY=X | GBP/JPY |

## Indicadores Técnicos Incluídos

### Indicadores de Tendência
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)

### Indicadores de Momentum
- RSI (Relative Strength Index)
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)

### Indicadores de Volatilidade
- Bollinger Bands
- ATR (Average True Range)

### Indicadores de Volume
- OBV (On-Balance Volume)
- Volume SMA

## Modelos de Machine Learning

A versão gratuita inclui vários algoritmos:

1. **Random Forest** - Robusto e interpretável
2. **Gradient Boosting** - Alta performance
3. **Logistic Regression** - Simples e rápido
4. **Support Vector Machine** - Bom para dados não-lineares

### Seleção Automática de Modelo
O sistema testa todos os modelos e seleciona automaticamente o melhor baseado na acurácia de validação cruzada.

## Estratégias de Trading Implementadas

### 1. Estratégia de Cruzamento de Médias
- Sinal de compra: EMA rápida cruza acima da EMA lenta
- Sinal de venda: EMA rápida cruza abaixo da EMA lenta

### 2. Estratégia RSI
- Compra: RSI < 30 (sobrevenda)
- Venda: RSI > 70 (sobrecompra)

### 3. Estratégia MACD
- Compra: MACD cruza acima da linha de sinal
- Venda: MACD cruza abaixo da linha de sinal

### 4. Estratégia Bollinger Bands
- Compra: Preço toca a banda inferior
- Venda: Preço toca a banda superior

## Limitações da Versão Gratuita

### Dados
- Dados com delay de ~15 minutos (Yahoo Finance)
- Histórico limitado (máximo ~10 anos)
- Sem dados de tick ou alta frequência

### Análise
- Sem análise fundamental automatizada
- Sem análise de sentimento de notícias
- Sem integração com calendário econômico

### Trading
- Sem execução automática de ordens
- Sem integração com brokers
- Sem gestão de risco avançada

## Melhorias Futuras

Para expandir o sistema gratuitamente:

1. **Mais Indicadores**
   - Ichimoku Cloud
   - Fibonacci Retracements
   - Pivot Points

2. **Análise Multi-Timeframe**
   - Combinar sinais de diferentes períodos
   - Análise de confluência

3. **Backtesting Avançado**
   - Métricas de performance detalhadas
   - Análise de drawdown
   - Otimização de parâmetros

4. **Visualizações**
   - Gráficos interativos com Plotly
   - Dashboard web simples
   - Relatórios em PDF

## Solução de Problemas

### Erro: "No module named 'talib'"
```bash
# Instalar TA-Lib ou usar alternativa
pip install ta
```

### Erro: "Failed to download data"
- Verifique conexão com internet
- Tente símbolo diferente
- Reduza período de dados

### Erro: "Insufficient data"
- Use período maior (ex: "1y" em vez de "1mo")
- Verifique se o símbolo está correto

### Performance Lenta
- Reduza período de dados
- Use menos modelos ML
- Instale TA-Lib para indicadores mais rápidos

## Contribuição

Este é um projeto open-source. Contribuições são bem-vindas:

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Aviso Legal

Este software é apenas para fins educacionais e de pesquisa. Não constitui aconselhamento financeiro. Trading de forex envolve riscos significativos e você pode perder todo seu capital investido.

Sempre:
- Teste estratégias em conta demo primeiro
- Use gestão de risco adequada
- Nunca invista mais do que pode perder
- Consulte um consultor financeiro qualificado

## Licença

MIT License - Veja arquivo LICENSE para detalhes.

## Suporte

Para dúvidas e suporte:
- Abra uma issue no GitHub
- Consulte a documentação
- Verifique exemplos incluídos

---

**Versão Gratuita - Sem Custos, Máximo Valor!**