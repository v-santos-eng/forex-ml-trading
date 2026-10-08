# Agno ML Forex - Edição Ollama

Sistema de trading Forex com IA usando **modelos locais gratuitos** via Ollama, eliminando a dependência de APIs pagas.

## Visão Geral

Este sistema implementa as três fases do Agno ML Forex usando o Ollama para executar modelos de IA localmente:

1. **Fase 1**: Ferramentas ML Encapsuladas (funciona independente do Ollama)
2. **Fase 2**: Sistema Multi-Agente usando Ollama
3. **Fase 3**: Memória e Raciocínio usando Ollama

## Vantagens da Edição Ollama

- **Gratuito**: Sem custos de API
- **Privado**: Dados não saem do seu computador
- **Offline**: Funciona sem internet (após setup inicial)
- **Sem Limites**: Sem quotas ou restrições de uso
- **Rápido**: Processamento local otimizado

## Arquivos Principais

### Sistema Ollama
- `agno_ollama_agents.py` - Agentes de trading usando Ollama
- `agno_ollama_memory.py` - Sistema de memória usando Ollama
- `demo_agno_ollama.py` - Demo completo com Ollama
- `OLLAMA_SETUP.md` - Guia de instalação do Ollama

### Sistema Original (APIs Pagas)
- `agno_forex_agents.py` - Agentes usando OpenAI
- `agno_memory_system.py` - Memória usando OpenAI
- `demo_agno_complete.py` - Demo com APIs pagas

### Base ML (Comum)
- `agno_ml_tool.py` - Ferramenta ML para Agno
- `ml_predictor_free.py` - Preditor ML base
- `config.py` - Configurações do sistema

## Instalação Rápida

### 1. Instalar Dependências Python

```bash
pip install ollama agno yfinance pandas numpy scikit-learn
```

### 2. Instalar Ollama

**Windows**:
1. Baixe de: https://ollama.com/download
2. Execute o instalador
3. Abra terminal e execute: `ollama serve`

### 3. Baixar Modelo

```bash
# Modelo recomendado
ollama pull llama3.2

# Alternativa mais leve
ollama pull llama3.2:1b
```

### 4. Executar Demo

```bash
python demo_agno_ollama.py
```

## Uso do Sistema

### Demo Completo

```python
# Executar todas as fases
python demo_agno_ollama.py
```

### Usar Apenas Agentes

```python
from agno_ollama_agents import OllamaForexAgentTeam

# Criar equipe de agentes
team = OllamaForexAgentTeam(model_id="llama3.2")

# Análise completa
analysis = team.analyze_pair("EURUSD=X")
print(analysis)

# Análise rápida
quick = team.quick_analysis("GBPUSD=X")
print(quick)
```

### Usar Sistema de Memória

```python
from agno_ollama_memory import OllamaForexMemorySystem

# Criar sistema de memória
memory = OllamaForexMemorySystem(model_id="llama3.2")

# Armazenar decisão
session_id = memory.store_trading_decision("EURUSD=X", decision_data)

# Analisar histórico
history = memory.analyze_trading_history("EURUSD=X")
print(history)
```

### Usar Apenas ML (Sem Ollama)

```python
from agno_ml_tool import MLForexTool

# Criar ferramenta ML
ml_tool = MLForexTool()

# Predição
result = ml_tool.predict_forex_direction("EURUSD=X")
print(result)

# Backtest
backtest = ml_tool.backtest_strategy("EURUSD=X")
print(backtest)
```

## Configuração Avançada

### Alterar Modelo

```python
# Usar modelo diferente
team = OllamaForexAgentTeam(model_id="mistral")
memory = OllamaForexMemorySystem(model_id="mistral")
```

### Servidor Remoto

```python
# Ollama em outro servidor
team = OllamaForexAgentTeam(
    model_id="llama3.2",
    host="http://192.168.1.100:11434"
)
```

## Modelos Recomendados

| Modelo | Tamanho | Qualidade | Velocidade | Uso |
|--------|---------|-----------|------------|-----|
| llama3.2 | ~4.7GB | Excelente | Boa | Produção |
| llama3.2:1b | ~1.3GB | Boa | Muito rápida | Desenvolvimento |
| mistral | ~4.1GB | Excelente | Rápida | Alternativa |
| codellama | ~3.8GB | Boa | Boa | Análise técnica |

## Estrutura dos Agentes

### Agentes de Trading

1. **Technical Analyst**
   - Análise técnica usando ML
   - Predições com níveis de confiança
   - Comparação de modelos

2. **Fundamental Analyst**
   - Análise de notícias econômicas
   - Indicadores macroeconômicos
   - Eventos geopolíticos

3. **Risk Manager**
   - Gestão de risco
   - Cálculo de posições
   - Stop-loss e take-profit

4. **Trading Coordinator**
   - Integração de análises
   - Decisões finais
   - Recomendações BUY/SELL/HOLD

### Agentes de Memória

1. **Memory Analyst**
   - Análise de histórico
   - Identificação de padrões
   - Insights de performance

2. **Learning Agent**
   - Aprendizado contínuo
   - Otimização de estratégias
   - Melhoria incremental

## Arquivos Gerados

- `ollama_ml_report.json` - Relatório da Fase 1 (ML)
- `ollama_complete_report.json` - Relatório completo
- `forex_memory.db` - Banco de dados de memória

## Solução de Problemas

### Ollama não conecta

```bash
# Verificar se está rodando
ollama serve

# Em outro terminal
curl http://localhost:11434/api/tags
```

### Modelo não encontrado

```bash
# Listar modelos
ollama list

# Baixar modelo
ollama pull llama3.2
```

### Erro de memória

```bash
# Usar modelo menor
ollama pull llama3.2:1b

# Ou fechar outros programas
```

### Dados ML insuficientes

O sistema ML pode mostrar "0 amostras" devido a:
- Problemas de conexão com Yahoo Finance
- Símbolos de moedas incorretos
- Períodos muito curtos

**Solução**: Aguardar ou tentar novamente mais tarde.

## Comparação: Ollama vs APIs Pagas

| Aspecto | Ollama | OpenAI/Anthropic |
|---------|--------|------------------|
| **Custo** | Gratuito | $0.01-0.06/1K tokens |
| **Privacidade** | Total | Dados enviados |
| **Velocidade** | Depende do HW | Geralmente rápida |
| **Qualidade** | Boa-Excelente | Excelente |
| **Disponibilidade** | Sempre | Limitado por quota |
| **Setup** | Mais complexo | Simples (só API key) |
| **Offline** | Sim | Não |

## Performance Esperada

### Fase 1 (ML)
- Sempre funciona (independente do Ollama)
- Predições de direção de moedas
- Backtest de estratégias
- Comparação de modelos

### Fase 2 (Agentes)
- Funciona com Ollama rodando
- Análises técnica e fundamental
- Gestão de risco
- Decisões de trading

### Fase 3 (Memória)
- Funciona com Ollama rodando
- Armazenamento de decisões
- Análise de histórico
- Aprendizado contínuo

## Próximos Passos

1. **Instalar Ollama** seguindo `OLLAMA_SETUP.md`
2. **Executar demo**: `python demo_agno_ollama.py`
3. **Personalizar** modelos e configurações
4. **Integrar** com sistemas de trading reais

## Suporte

- **Documentação Ollama**: https://ollama.com/docs
- **Modelos disponíveis**: https://ollama.com/library
- **GitHub Ollama**: https://github.com/ollama/ollama

---

**Nota**: Esta edição Ollama fornece uma alternativa completamente gratuita e privada ao sistema original que dependia de APIs pagas, mantendo toda a funcionalidade de IA para trading de Forex.