# Guia de Deployment - Sistema de Trading ML Forex

## Visão Geral

Este documento descreve o processo completo de deployment do sistema de trading automatizado com Machine Learning para o mercado Forex.

## Estrutura do Projeto

```
forex-ml-trading/
├── auto_trader.py              # Bot principal de trading
├── mt5_dashboard.py            # Interface gráfica do dashboard
├── trading_monitor.py          # Monitor em tempo real
├── detailed_decision_logger.py # Sistema de logging detalhado
├── deploy_to_production.py     # Script de deployment
├── production_config.json      # Configurações de produção
├── models/                     # Modelos ML treinados
├── data/                       # Dados históricos e logs
└── logs/                       # Arquivos de log
```

## Pré-requisitos

### Software Necessário
- Python 3.8+
- MetaTrader 5 instalado e configurado
- Conta de trading ativa (demo ou real)
- Bibliotecas Python (ver requirements.txt)

### Configuração do Ambiente
1. **Terminal MetaTrader 5**
   - Instalar e configurar MT5
   - Habilitar trading algorítmico
   - Configurar conta de trading

2. **Dependências Python**
   ```bash
   pip install -r requirements.txt
   ```

## Processo de Deployment

### 1. Preparação

#### Validação do Ambiente de Staging
```bash
# Executar testes no ambiente de staging
python -m pytest tests/
python test_detailed_logging.py
```

#### Backup dos Dados
```bash
# Fazer backup dos modelos e configurações atuais
cp -r models/ backup_models_$(date +%Y%m%d)/
cp production_config.json backup_config_$(date +%Y%m%d).json
```

### 2. Deployment Automatizado

#### Execução do Script de Deployment
```bash
python deploy_to_production.py
```

O script automaticamente:
- Copia arquivos do staging para produção
- Atualiza configurações
- Valida integridade dos arquivos
- Cria backups de segurança

### 3. Configuração de Produção

#### Arquivo production_config.json
```json
{
  "environment": "production",
  "mt5_settings": {
    "server": "MetaQuotes-Demo",
    "login": "SEU_LOGIN",
    "password": "SUA_SENHA",
    "timeout": 60000
  },
  "trading_settings": {
    "max_risk_per_trade": 0.02,
    "max_daily_trades": 10,
    "min_confidence_threshold": 0.65,
    "symbols": ["EURUSD", "GBPUSD", "USDJPY"]
  },
  "logging": {
    "level": "INFO",
    "detailed_logging": true,
    "log_retention_days": 30
  }
}
```

### 4. Inicialização dos Componentes

#### Ordem de Inicialização
1. **Auto Trader (Bot Principal)**
   ```bash
   python auto_trader.py --production
   ```

2. **Dashboard MT5 (Interface Gráfica)**
   ```bash
   python mt5_dashboard.py --production
   ```

3. **Monitor de Trading (Tempo Real)**
   ```bash
   python trading_monitor.py --production
   ```

## Validação do Deployment

### 1. Testes Funcionais

#### Sistema de Logging Detalhado
```bash
python test_detailed_logging.py
```

**Resultados Esperados:**
- ✅ Inicialização do logger
- ✅ Registro de decisões de sinal
- ✅ Registro de execuções de trade
- ✅ Registro de não-ações
- ✅ Recuperação de decisões
- ✅ Análise de performance
- ✅ Exportação para CSV
- ✅ Geração de resumos

#### Conectividade MT5
```bash
# Verificar logs do auto_trader
tail -f logs/auto_trader.log
```

**Status Esperado:**
- Conexão MT5 estabelecida
- Símbolos carregados
- Modelos ML inicializados
- Sistema de monitoramento ativo

### 2. Monitoramento em Tempo Real

#### Métricas Principais
- **Status da Conta**: Saldo, equity, margem
- **Estatísticas de Trading**: Sinais gerados, trades executados, win rate
- **Posições Abertas**: Quantidade e status
- **Últimos Sinais**: Histórico recente de decisões

#### Dashboard Web
- Interface gráfica funcionando
- Abas de logs detalhados operacionais
- Exportação de dados funcionando
- Análise de performance disponível

## Estrutura de Logs

### Tipos de Log

1. **Logs Operacionais**
   - `logs/auto_trader.log`: Atividade do bot principal
   - `logs/mt5_integration.log`: Interações com MT5
   - `logs/ml_predictions.log`: Predições dos modelos

2. **Logs Detalhados de Decisão**
   - Banco SQLite: `data/detailed_decisions.db`
   - Exportação CSV: Disponível via dashboard
   - Análise de performance: Métricas agregadas

### Formato dos Logs Detalhados

```json
{
  "decision_id": "uuid",
  "timestamp": "2025-08-24T11:51:03.084210",
  "symbol": "EURUSD",
  "action": "GENERATE_SIGNAL",
  "confidence": 0.72,
  "market_data": {
    "bid": 1.0850,
    "ask": 1.0852,
    "spread": 2,
    "volume": 1500
  },
  "technical_analysis": {
    "rsi": 65.5,
    "macd": 0.0012,
    "bollinger_position": "middle"
  },
  "ml_predictions": {
    "model_1": 0.75,
    "model_2": 0.68,
    "ensemble": 0.72
  },
  "reasoning_factors": [
    "RSI em 65.5 indica momentum bullish moderado",
    "MACD positivo confirma tendência de alta",
    "Modelos ML convergem para sinal de compra"
  ]
}
```

## Monitoramento e Manutenção

### Verificações Diárias
1. **Status dos Processos**
   ```bash
   ps aux | grep python | grep -E "(auto_trader|mt5_dashboard|trading_monitor)"
   ```

2. **Logs de Erro**
   ```bash
   grep -i error logs/*.log
   ```

3. **Performance do Sistema**
   - CPU e memória utilizados
   - Espaço em disco disponível
   - Conectividade de rede

### Verificações Semanais
1. **Análise de Performance**
   - Win rate dos trades
   - Drawdown máximo
   - Sharpe ratio
   - Profit factor

2. **Manutenção dos Modelos**
   - Retreinamento com dados recentes
   - Validação de performance
   - Atualização se necessário

### Verificações Mensais
1. **Limpeza de Logs**
   ```bash
   find logs/ -name "*.log" -mtime +30 -delete
   ```

2. **Backup Completo**
   ```bash
   tar -czf backup_$(date +%Y%m%d).tar.gz models/ data/ logs/ *.json *.py
   ```

## Troubleshooting

### Problemas Comuns

#### 1. Falha na Conexão MT5
**Sintomas:**
- Logs mostram "Market closed" repetidamente
- Dashboard não mostra cotações

**Soluções:**
- Verificar se MT5 está aberto e logado
- Confirmar configurações de servidor
- Verificar conectividade de internet

#### 2. Modelos ML Não Carregam
**Sintomas:**
- Erro ao inicializar auto_trader
- Predições sempre retornam valores padrão

**Soluções:**
- Verificar integridade dos arquivos em `models/`
- Retreinar modelos se necessário
- Verificar compatibilidade de versões

#### 3. Sistema de Logging Não Funciona
**Sintomas:**
- Banco de dados não é criado
- Dashboard não mostra logs detalhados

**Soluções:**
- Verificar permissões de escrita em `data/`
- Executar `test_detailed_logging.py`
- Verificar logs de erro do sistema

## Rollback

### Processo de Rollback
1. **Parar Todos os Processos**
   ```bash
   pkill -f "python.*auto_trader"
   pkill -f "python.*mt5_dashboard"
   pkill -f "python.*trading_monitor"
   ```

2. **Restaurar Backup**
   ```bash
   cp backup_models_YYYYMMDD/* models/
   cp backup_config_YYYYMMDD.json production_config.json
   ```

3. **Reinicializar Sistema**
   ```bash
   python auto_trader.py --production
   python mt5_dashboard.py --production
   python trading_monitor.py --production
   ```

## Contatos e Suporte

### Equipe Técnica
- **Desenvolvedor Principal**: [Nome]
- **Administrador de Sistema**: [Nome]
- **Analista Quantitativo**: [Nome]

### Documentação Adicional
- `README.md`: Visão geral do projeto
- `API_DOCUMENTATION.md`: Documentação da API
- `MODEL_DOCUMENTATION.md`: Documentação dos modelos ML

---

**Última Atualização**: 24/08/2025
**Versão**: 1.0
**Status**: Produção Ativa