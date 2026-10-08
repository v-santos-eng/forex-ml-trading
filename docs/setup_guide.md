# Guia de Configuração - Forex ML com Agno

## Pré-requisitos

1. **Python 3.8+** instalado
2. **Conta Anthropic** para obter API key

## Configuração Passo a Passo

### 1. Obter API Key do Anthropic

1. Acesse [console.anthropic.com](https://console.anthropic.com/)
2. Faça login ou crie uma conta
3. Vá para "API Keys"
4. Clique em "Create Key"
5. Copie sua API key (formato: `sk-ant-api03-xxxx`)

### 2. Configurar Variável de Ambiente

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-api03-sua-chave-aqui"
```

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-api03-sua-chave-aqui
```

**Linux/Mac:**
```bash
export ANTHROPIC_API_KEY=sk-ant-api03-sua-chave-aqui
```

### 3. Instalar Dependências

**Opção 1 - Instalação Básica:**
```bash
pip install agno anthropic yfinance pandas numpy
```

**Opção 2 - Instalação Completa:**
```bash
pip install -r requirements.txt
```

### 4. Testar Instalação

```bash
python simple_example.py
```

Se tudo estiver configurado corretamente, você verá:
```
Iniciando Forex Agent com Agno...
Agent criado com sucesso!
```

## Estrutura do Projeto

```
forex-ml-trading/
├── simple_example.py       # Exemplo básico - COMECE AQUI
├── forex_agent.py          # Agente individual especializado
├── forex_analysis.py       # Team de agentes especializados
├── ml_forex_predictor.py   # Preditor ML avançado
├── config.py              # Configurações centralizadas
├── requirements.txt       # Dependências do projeto
├── README.md              # Documentação principal
└── setup_guide.md         # Este guia
```

## Primeiros Passos

### 1. Teste Básico
```bash
python simple_example.py
```

### 2. Agente Individual
```python
from forex_agent import ForexMLAgent

agent = ForexMLAgent()
agent.analyze_currency_pair("EUR/USD")
```

### 3. Team de Agentes
```python
from forex_analysis import ForexAnalysisTeam

team = ForexAnalysisTeam()
team.comprehensive_analysis("EUR/USD")
```

### 4. ML Predictor
```python
from ml_forex_predictor import MLForexPredictor

predictor = MLForexPredictor()
predictor.generate_trading_signals("EUR/USD")
```

## Solução de Problemas

### Erro: "ANTHROPIC_API_KEY não encontrada"
**Solução:** Configure a variável de ambiente conforme seção 2.

### Erro: "ModuleNotFoundError: No module named 'agno'"
**Solução:** Instale as dependências: `pip install agno`

### Erro: "API key inválida"
**Solução:** Verifique se a API key está correta e tem créditos.

### Erro de conexão
**Solução:** Verifique sua conexão com a internet.

## Dicas de Uso

1. **Comece simples:** Use `simple_example.py` primeiro
2. **Explore gradualmente:** Teste cada arquivo individualmente
3. **Personalize:** Modifique as instruções dos agentes em `config.py`
4. **Monitore custos:** APIs têm custos - monitore seu uso
5. **Backup:** Faça backup de suas configurações personalizadas

## Avisos Importantes

- **Não é aconselhamento financeiro**
- **Forex envolve riscos significativos**
- **Sempre faça sua própria pesquisa**
- **Use stop-loss em trades reais**
- **Teste em paper trading primeiro**

## Suporte

- **Documentação Agno:** [docs.agno.com](https://docs.agno.com)
- **GitHub Agno:** [github.com/agno-agi/agno](https://github.com/agno-agi/agno)
- **Anthropic Docs:** [docs.anthropic.com](https://docs.anthropic.com)

## Próximos Passos

1. Configure a API key
2. Teste o exemplo básico
3. Explore os diferentes agentes
4. Personalize para suas necessidades
5. Implemente estratégias de ML
6. Faça backtesting
7. Teste em paper trading

---

**Powered by [Agno](https://github.com/agno-agi/agno) - Framework para Sistemas Multi-Agentes**