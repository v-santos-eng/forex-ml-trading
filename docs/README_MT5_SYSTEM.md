# Sistema de Trading MT5 com Machine Learning

## Visão Geral

Sistema completo de trading automatizado para MetaTrader 5 (MT5) integrado com Machine Learning, desenvolvido sem uso de WebSocket ou StreamLib. O sistema oferece:

- **Integração Nativa MT5**: Conexão direta via API Python do MetaTrader 5
- **Dashboard Tkinter**: Interface gráfica robusta para controle e monitoramento
- **Machine Learning**: Predições em tempo real e sinais automatizados
- **Gerenciamento de Risco**: Controle avançado de posições e exposição
- **Sistema de Logs**: Monitoramento completo e auditoria
- **Polling de Dados**: Atualização otimizada sem WebSocket

## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    MT5 Main Launcher                       │
│                 (Controle Central)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌────▼────┐ ┌─────▼─────┐
│ MT5 Integration│ │Dashboard│ │ML Engine  │
│   (API Nativa) │ │(Tkinter)│ │(Predições)│
└───────┬──────┘ └────┬────┘ └─────┬─────┘
        │             │             │
┌───────▼──────┐ ┌────▼────┐ ┌─────▼─────┐
│Risk Manager  │ │Data     │ │Logging    │
│(Controle)    │ │Polling  │ │System     │
└──────────────┘ └─────────┘ └───────────┘
```

## Requisitos do Sistema

### Software Necessário

1. **Python 3.8+**
2. **MetaTrader 5** (instalado e configurado)
3. **Conta de trading** (demo ou real)

### Dependências Python

```bash
pip install MetaTrader5
pip install tkinter
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install psutil
pip install sqlite3
```

### Instalação Completa

```bash
# Clone ou baixe os arquivos do sistema
# Instale as dependências
pip install -r requirements.txt

# Verifique se o MT5 está instalado
# Certifique-se de ter uma conta de trading configurada
```

## Estrutura de Arquivos

```
mt5_trading_system/
├── mt5_main_launcher.py          # Launcher principal
├── mt5_integration.py            # Integração MT5
├── mt5_dashboard.py              # Dashboard Tkinter
├── risk_manager.py               # Gerenciamento de risco
├── data_polling_system.py        # Sistema de dados
├── ml_mt5_integration.py         # Engine ML
├── logging_system.py             # Sistema de logs
├── config_manager.py             # Gerenciador de config
├── README_MT5_SYSTEM.md          # Esta documentação
├── requirements.txt              # Dependências
└── logs/                         # Diretório de logs
    ├── trading.db               # Banco de dados
    └── mt5_trading_YYYYMMDD.log # Logs diários
```

## Configuração Inicial

### 1. Configuração do MetaTrader 5

1. **Instale o MT5** da MetaQuotes
2. **Configure uma conta** (demo recomendado para testes)
3. **Habilite trading automatizado**:
   - Tools → Options → Expert Advisors
   - Marque "Allow automated trading"
   - Marque "Allow DLL imports"

### 2. Configuração da Conta

Anote as seguintes informações da sua conta MT5:
- **Login**: Número da conta
- **Senha**: Senha da conta
- **Servidor**: Nome do servidor (ex: "MetaQuotes-Demo")
- **Caminho**: Localização do terminal64.exe

### 3. Primeira Execução

```bash
python mt5_main_launcher.py
```

O sistema abrirá uma janela de configuração onde você deve:

1. **Inserir credenciais MT5**
2. **Configurar opções do sistema**
3. **Salvar configuração**
4. **Iniciar sistema**

## Uso do Sistema

### Interface de Inicialização

A janela inicial permite:

- **Configurar credenciais MT5**
- **Detectar instalação MT5 automaticamente**
- **Habilitar/desabilitar componentes**:
  - Machine Learning
  - Auto Trading (CUIDADO!)
  - Dashboard
- **Salvar/carregar configurações**

### Dashboard Principal

Após a inicialização, o dashboard oferece:

#### Aba "Conta"
- Informações da conta em tempo real
- Saldo, equity, margem
- Status da conexão

#### Aba "Trading"
- Envio manual de ordens
- Seleção de símbolos
- Configuração de volume, SL, TP
- Botões BUY/SELL

#### Aba "Posições"
- Lista de posições abertas
- Lucro/prejuízo em tempo real
- Botões para fechar posições

#### Aba "Ordens"
- Ordens pendentes
- Modificação e cancelamento

#### Aba "ML Analysis"
- Predições do Machine Learning
- Sinais de compra/venda
- Confiança das predições

#### Aba "Risk"
- Status do gerenciamento de risco
- Métricas de exposição
- Alertas de risco

#### Aba "Logs"
- Logs do sistema em tempo real
- Filtros por nível e componente
- Histórico de operações

### Auto Trading

⚠️ **ATENÇÃO**: O auto trading executa operações reais!

**Para habilitar**:
1. Configure adequadamente o gerenciamento de risco
2. Teste extensivamente em conta demo
3. Marque "Habilitar Auto Trading" na inicialização
4. O sistema executará trades baseados em sinais ML

**Controles de segurança**:
- Stop-loss automático
- Limites de posição
- Controle de drawdown
- Parada de emergência

## Configurações Avançadas

### Arquivo de Configuração

O sistema gera `mt5_system_config.json` com todas as configurações:

```json
{
  "mt5": {
    "login": 12345678,
    "password": "sua_senha",
    "server": "MetaQuotes-Demo",
    "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
    "timeout": 10000,
    "auto_connect": true
  },
  "risk": {
    "max_daily_loss_percent": 2.0,
    "max_position_size_percent": 1.0,
    "max_positions": 5,
    "max_correlation": 0.7,
    "emergency_stop_loss_percent": 5.0,
    "use_dynamic_position_sizing": true,
    "risk_per_trade_percent": 0.5
  },
  "ml": {
    "prediction_interval_seconds": 30,
    "min_confidence_threshold": 0.6,
    "use_ensemble_predictions": true,
    "retrain_interval_hours": 24,
    "max_lookback_periods": 100
  }
}
```

### Personalização de Risco

**Parâmetros principais**:
- `max_daily_loss_percent`: Perda máxima diária (% do saldo)
- `max_position_size_percent`: Tamanho máximo de posição
- `max_positions`: Número máximo de posições simultâneas
- `risk_per_trade_percent`: Risco por trade

### Configuração ML

**Parâmetros de predição**:
- `prediction_interval_seconds`: Intervalo entre predições
- `min_confidence_threshold`: Confiança mínima para sinais
- `use_ensemble_predictions`: Usar múltiplos modelos

## Monitoramento e Logs

### Sistema de Logs

O sistema registra:
- **Conexões/desconexões MT5**
- **Todas as operações de trading**
- **Predições ML e sinais**
- **Violações de risco**
- **Erros e exceções**
- **Métricas de performance**

### Localização dos Logs

- **Banco SQLite**: `~/.mt5_trading/logs/trading.db`
- **Logs texto**: `~/.mt5_trading/logs/mt5_trading_YYYYMMDD.log`

### Consulta de Logs

```python
from logging_system import get_logging_system

logging_sys = get_logging_system()

# Logs recentes
recent_logs = logging_sys.get_recent_logs(limit=100)

# Logs do banco com filtros
logs = logging_sys.get_logs_from_db(
    start_time=datetime.now() - timedelta(hours=1),
    level=LogLevel.ERROR
)
```

## Troubleshooting

### Problemas Comuns

#### 1. Erro de Conexão MT5

**Sintomas**: "Falha ao conectar com MT5"

**Soluções**:
- Verifique se o MT5 está rodando
- Confirme credenciais (login, senha, servidor)
- Teste conexão manual no MT5
- Verifique firewall/antivírus

#### 2. Erro "MetaTrader5 module not found"

**Solução**:
```bash
pip install MetaTrader5
```

#### 3. Dashboard não abre

**Soluções**:
- Verifique se tkinter está instalado
- Execute como administrador
- Verifique logs de erro

#### 4. ML não funciona

**Soluções**:
- Verifique dependências: pandas, numpy, scikit-learn
- Aguarde coleta de dados históricos
- Verifique logs ML na aba correspondente

#### 5. Auto Trading não executa

**Verificações**:
- Auto trading habilitado no MT5
- Configurações de risco adequadas
- Sinais ML com confiança suficiente
- Margem disponível na conta

### Logs de Debug

Para debug detalhado, modifique em `config_manager.py`:

```python
"logging": {
    "log_level": "DEBUG",  # Mude de INFO para DEBUG
    # ...
}
```

### Verificação de Saúde do Sistema

O sistema monitora automaticamente:
- **Conexão MT5**: Reconecta automaticamente
- **Uso de CPU/Memória**: Alertas se muito alto
- **Threads ativas**: Detecta travamentos
- **Latência de dados**: Monitora performance

## Performance e Otimização

### Configurações de Performance

**Para trading de alta frequência**:
```json
"data_polling": {
    "tick_interval_ms": 50,     # Mais rápido
    "account_interval_ms": 500,
    "positions_interval_ms": 250
}
```

**Para uso normal**:
```json
"data_polling": {
    "tick_interval_ms": 100,
    "account_interval_ms": 1000,
    "positions_interval_ms": 500
}
```

### Monitoramento de Recursos

O sistema monitora:
- **CPU**: Alerta se > 90%
- **Memória**: Alerta se > 90%
- **Threads**: Conta threads ativas
- **Latência**: Tempo de resposta MT5

## Segurança

### Boas Práticas

1. **Sempre teste em conta demo primeiro**
2. **Configure limites de risco conservadores**
3. **Monitore o sistema constantemente**
4. **Mantenha backups das configurações**
5. **Use senhas fortes**
6. **Não compartilhe credenciais**

### Controles de Risco Implementados

- **Stop-loss obrigatório** em todas as posições
- **Limite de drawdown diário**
- **Controle de correlação** entre posições
- **Parada de emergência** automática
- **Validação de margem** antes de trades

## Extensibilidade

### Adicionando Novos Indicadores

Em `ml_mt5_integration.py`, classe `TechnicalAnalyzer`:

```python
def custom_indicator(self, data: pd.DataFrame) -> pd.Series:
    """Seu indicador customizado"""
    # Implementar lógica
    return result
```

### Novos Modelos ML

Em `ml_mt5_integration.py`, método `_train_models`:

```python
# Adicionar novo modelo
from sklearn.ensemble import ExtraTreesClassifier

models['ExtraTrees'] = ExtraTreesClassifier(
    n_estimators=100,
    random_state=42
)
```

### Callbacks Customizados

```python
def my_callback(log_entry):
    """Callback personalizado para logs"""
    if log_entry.level == LogLevel.ERROR:
        # Enviar notificação, email, etc.
        pass

logging_sys.add_log_callback(my_callback)
```

## Suporte e Contribuição

### Reportar Problemas

1. **Colete logs relevantes**
2. **Descreva o problema detalhadamente**
3. **Inclua configuração (sem senhas)**
4. **Especifique versão do Python e MT5**

### Estrutura para Issues

```
**Problema**: Descrição breve

**Ambiente**:
- Python: 3.x.x
- MT5: Build xxxx
- OS: Windows xx

**Passos para reproduzir**:
1. ...
2. ...

**Comportamento esperado**: ...

**Comportamento atual**: ...

**Logs relevantes**: ...
```

## Changelog

### v1.0.0 (2024)
- Implementação inicial
- Integração MT5 nativa
- Dashboard Tkinter completo
- Sistema ML integrado
- Gerenciamento de risco avançado
- Sistema de logs robusto
- Polling de dados otimizado
- Launcher centralizado

## Licença

Este sistema é fornecido "como está" para fins educacionais e de desenvolvimento. 

**AVISO IMPORTANTE**: Trading envolve risco de perda financeira. Use sempre contas demo para testes e nunca invista mais do que pode perder. Os desenvolvedores não se responsabilizam por perdas financeiras.

---

**Sistema MT5 Trading com ML - Desenvolvido para máxima robustez e facilidade de uso**