# Sistema ML Forex - Produção

## Visão Geral

Este diretório contém todos os componentes necessários para operar o sistema ML Forex em produção, incluindo monitoramento, recuperação automática e scripts de gerenciamento.

## Características Principais

### 🎯 Interface Moderna
- Design profissional com tema escuro
- Layout responsivo e redimensionável
- Widgets customizados e otimizados
- Navegação intuitiva por abas

### 📊 Monitoramento em Tempo Real
- Dados de conta atualizados automaticamente
- Posições e ordens em tempo real
- Gráficos interativos e análises
- Histórico de operações detalhado

### 🔔 Sistema de Notificações
- Alertas visuais e sonoros
- Categorização por tipo e prioridade
- Histórico completo de notificações
- Configurações personalizáveis

### 🚀 Otimização de Performance
- Monitoramento automático de recursos
- Cache inteligente de dados
- Otimização dinâmica de memória
- Relatórios detalhados de performance

### 🌐 Coleta de Dados MT5
- Comunicação direta com MT5
- Conexão otimizada
- Coleta eficiente de dados
- Sincronização em tempo real

### 📄 Sistema de Exportação
- Relatórios em PDF profissionais
- Exportação de dados em CSV/Excel
- Logs detalhados do sistema
- Configurações flexíveis

## Requisitos do Sistema

### Software Necessário
- **Python**: 3.8 ou superior
- **MetaTrader 5**: Build 3815 ou superior
- **Sistema Operacional**: Windows 10/11 (recomendado)

### Dependências Python
```bash
pip install -r requirements.txt
```

**Principais bibliotecas:**
- `MetaTrader5`: Integração com MT5
- `tkinter`: Interface gráfica
- `pandas`: Manipulação de dados
- `matplotlib`: Gráficos e visualizações
- `reportlab`: Geração de PDFs
- `pyyaml`: Configurações
- `psutil`: Monitoramento de sistema

## Instalação e Configuração

### 1. Preparação do Ambiente

```bash
# Clone o repositório
git clone <repository-url>
cd forex-ml-trading/production_deployment

# Instale as dependências
pip install -r requirements.txt

# Copie o arquivo de configuração
cp config_example.yaml config.yaml
```

### 2. Configuração do MT5

1. **Instale o MetaTrader 5**
2. **Configure sua conta de trading**
3. **Habilite o trading algorítmico**:
   - Ferramentas → Opções → Expert Advisors
   - Marque "Permitir trading algorítmico"
   - Marque "Permitir importação de DLL"

### 3. Configuração do Sistema

Edite o arquivo `config.yaml`:

```yaml
# Configurações básicas
trading:
  auto_start: true
  default_lot_size: 0.01
  max_positions: 10

performance:
  update_interval: 30
  cache_size: 1000
  memory_limit_mb: 500

data_provider:
  update_interval: 30
  cache_enabled: true
```

### 4. Primeira Execução

```bash
# Execute o dashboard principal
python mt5_dashboard_integrated.py
```

## Estrutura do Projeto

```
production_deployment/
├── 📁 Core Components
│   ├── mt5_dashboard_integrated.py     # Aplicação principal
│   ├── mt5_connector.py                # Conexão MT5
│   ├── mt5_data_provider.py            # Provedor de dados
│   └── streaming_adapter.py            # Adaptador de dados
│
├── 📁 UI Components
│   ├── advanced_widgets.py             # Widgets customizados
│   ├── notification_system.py          # Sistema de notificações
│   └── settings_system.py              # Gerenciamento de configurações
│
├── 📁 Business Logic
│   ├── performance_optimization.py     # Otimização de performance
│   ├── export_system.py                # Sistema de exportação
│   ├── ml_model_scheduler.py           # Agendador de modelos ML
│   └── user_preferences.py             # Preferências do usuário
│
├── 📁 Configuration
│   ├── config.yaml                     # Configurações principais
│   ├── config_example.yaml             # Exemplo de configuração
│   └── requirements.txt                # Dependências Python
│
├── 📁 Documentation
│   ├── README.md                       # Este arquivo
│   ├── DASHBOARD_USER_GUIDE.md         # Guia do usuário
│   ├── TECHNICAL_DOCUMENTATION.md      # Documentação técnica
│   └── API_REFERENCE.md                # Referência da API
│
├── 📁 Tests
│   ├── test_dashboard_performance.py   # Testes de performance
│   ├── test_mt5_integration.py         # Testes de integração MT5
│   └── test_data_provider.py           # Testes do provedor de dados
│
└── 📁 Assets
    ├── logs/                           # Arquivos de log
    ├── exports/                        # Arquivos exportados
    ├── backups/                        # Backups automáticos
    └── sounds/                         # Sons de notificação
```

## Guia de Uso Rápido

### Iniciando o Sistema

1. **Execute o dashboard principal**:
   ```bash
   python mt5_dashboard_integrated.py
   ```

2. **Inicie o dashboard**:
   ```bash
   python mt5_dashboard_integrated.py
   ```

3. **Conecte ao MT5**:
   - O sistema tentará conectar automaticamente
   - Verifique o status na barra inferior

### Funcionalidades Principais

#### 📊 Painel de Trading
- **Informações da Conta**: Saldo, equity, margem
- **Controles de Trading**: Abrir/fechar posições
- **Configurações de Risco**: Stop loss, take profit

#### 📈 Análise de Posições
- **Posições Abertas**: Lista detalhada com P&L
- **Histórico**: Operações passadas
- **Estatísticas**: Performance geral

#### 🔔 Notificações
- **Alertas em Tempo Real**: Operações, erros, avisos
- **Histórico**: Todas as notificações registradas
- **Configurações**: Personalizar tipos e sons

#### ⚡ Performance
- **Monitoramento**: CPU, memória, rede
- **Otimização**: Automática e manual
- **Relatórios**: Métricas detalhadas

### Atalhos de Teclado

- `Ctrl+N`: Nova operação
- `Ctrl+R`: Atualizar dados
- `Ctrl+E`: Exportar relatório
- `Ctrl+S`: Salvar configurações
- `F5`: Reconectar MT5
- `F11`: Tela cheia
- `Esc`: Fechar janelas modais

## Configurações Avançadas

### Otimização de Performance

```yaml
performance:
  # Cache de dados
  cache_size: 1000
  
  # Intervalo de atualização (segundos)
  update_interval: 30
  
  # Limite de memória (MB)
  memory_limit_mb: 500
  
  # Otimização automática
  auto_optimization:
    enabled: true
    memory_threshold: 80
    cpu_threshold: 70
```

### Personalização da Interface

```yaml
ui:
  theme: "dark"  # dark/light
  window_size:
    width: 1400
    height: 900
  charts:
    default_timeframe: "M15"
    max_candles: 1000
```

### Sistema de Notificações

```yaml
notifications:
  enabled: true
  types:
    trade_opened: true
    trade_closed: true
    connection_lost: true
  sound:
    enabled: true
    volume: 0.5
```

## Solução de Problemas

### Problemas Comuns

#### ❌ Erro de Conexão MT5
**Sintomas**: "MT5 não conectado" na barra de status

**Soluções**:
1. Verifique se o MT5 está aberto
2. Confirme as credenciais de login
3. Habilite o trading algorítmico
4. Reinicie o MT5 e o dashboard

#### ❌ Dados MT5 Não Atualizando
**Sintomas**: Dados não atualizando em tempo real

**Soluções**:
1. Verifique se o MT5 está aberto e conectado
2. Confirme a conexão com o servidor MT5
3. Verifique se o trading algorítmico está habilitado
4. Reinicie o dashboard

#### ❌ Performance Lenta
**Sintomas**: Interface travando ou lenta

**Soluções**:
1. Reduza o intervalo de atualização
2. Diminua o tamanho do cache
3. Feche outras aplicações pesadas
4. Ative a otimização automática

#### ❌ Erro de Exportação
**Sintomas**: Falha ao gerar relatórios

**Soluções**:
1. Verifique permissões da pasta de exportação
2. Confirme espaço em disco disponível
3. Instale dependências de PDF (reportlab)
4. Verifique logs para detalhes do erro

### Logs e Diagnóstico

#### Localização dos Logs
```
logs/
├── dashboard.log          # Log principal
├── mt5_connector.log      # Conexão MT5
├── data_provider.log      # Provedor de dados
├── performance.log        # Performance
└── error.log              # Erros críticos
```

#### Níveis de Log
- **DEBUG**: Informações detalhadas para desenvolvimento
- **INFO**: Operações normais do sistema
- **WARNING**: Situações que merecem atenção
- **ERROR**: Erros que afetam funcionalidades
- **CRITICAL**: Erros que impedem o funcionamento

## Desenvolvimento e Contribuição

### Ambiente de Desenvolvimento

```bash
# Clone o repositório
git clone <repository-url>
cd forex-ml-trading/production_deployment

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instale dependências de desenvolvimento
pip install -r requirements-dev.txt

# Execute os testes
python -m pytest tests/
```

### Estrutura de Testes

```bash
# Testes de performance
python test_dashboard_performance.py

# Testes de integração
python test_mt5_integration.py

# Testes de Data Provider
python test_data_provider.py
```

### Padrões de Código

- **PEP 8**: Estilo de código Python
- **Type Hints**: Tipagem estática
- **Docstrings**: Documentação inline
- **Error Handling**: Tratamento robusto de exceções

## API e Integrações

### API de Dados MT5

```python
# Acessar dados diretamente do MT5
from mt5_data_provider import MT5DataProvider
from mt5_connector import MT5Connector

# Conectar ao MT5
connector = MT5Connector()
if connector.connect():
    data_provider = MT5DataProvider(connector)
    
    # Obter informações da conta
    account_info = data_provider.get_account_info()
    
    # Obter posições
    positions = data_provider.get_positions()
```

### REST API (Opcional)

```python
# Habilitar no config.yaml
api:
  rest_enabled: true
  rest_port: 8080

# Endpoints disponíveis
GET /api/account        # Informações da conta
GET /api/positions      # Posições abertas
GET /api/history        # Histórico de operações
POST /api/trade         # Executar operação
```

## Segurança e Backup

### Configurações de Segurança

```yaml
security:
  encrypt_credentials: true
  session_timeout: 60
  integrity_check: true
  auto_backup: true
```

### Sistema de Backup

```yaml
backup:
  enabled: true
  directory: "backups"
  interval: 6  # horas
  max_backups: 10
  compress: true
```

## Suporte e Comunidade

### Documentação Adicional
- **[Guia do Usuário](DASHBOARD_USER_GUIDE.md)**: Manual completo de uso
- **[Documentação Técnica](TECHNICAL_DOCUMENTATION.md)**: Arquitetura e implementação
- **[Referência da API](API_REFERENCE.md)**: Documentação da API

### Contato e Suporte
- **Issues**: Reporte bugs e solicite funcionalidades
- **Discussions**: Tire dúvidas e compartilhe experiências
- **Wiki**: Documentação colaborativa

### Roadmap

#### Versão 2.1 (Próxima)
- [ ] Suporte a múltiplas contas MT5
- [ ] Análise técnica avançada
- [ ] Backtesting integrado
- [ ] Mobile app companion

#### Versão 2.2 (Futuro)
- [ ] Machine Learning para análise
- [ ] Copy trading
- [ ] Social trading features
- [ ] Cloud synchronization

---

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

## Changelog

### v2.0.0 (Atual)
- ✅ Interface moderna com tema escuro
- ✅ Sistema de notificações completo
- ✅ Otimização automática de performance
- ✅ Coleta direta de dados MT5
- ✅ Sistema de exportação avançado
- ✅ Configurações personalizáveis
- ✅ Documentação completa

### v1.0.0 (Inicial)
- ✅ Conexão básica com MT5
- ✅ Interface Tkinter simples
- ✅ Monitoramento de posições
- ✅ Logs básicos

---

**Desenvolvido com ❤️ para a comunidade de trading**

*Dashboard MT5 Integrado - Transformando dados em decisões inteligentes*