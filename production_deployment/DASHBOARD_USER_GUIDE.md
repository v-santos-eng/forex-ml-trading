# Guia do Usuário - Dashboard MT5 Integrado

## Visão Geral

O Dashboard MT5 Integrado é uma interface gráfica moderna e responsiva desenvolvida em Tkinter para trading automatizado no MetaTrader 5. Este sistema oferece monitoramento em tempo real, análise de performance, gestão de risco e coleta direta de dados do MT5.

## Funcionalidades Principais

### 1. Interface Moderna
- **Tema Escuro Profissional**: Interface com paleta de cores otimizada para uso prolongado
- **Layout Responsivo**: Redimensionamento automático de componentes
- **Widgets Avançados**: Treeview melhorado, Canvas para gráficos, Notebook com abas

### 2. Monitoramento em Tempo Real
- **Dados de Conta**: Saldo, equity, margem livre em tempo real
- **Posições Ativas**: Lista atualizada de todas as posições abertas
- **Histórico de Ordens**: Registro completo de transações
- **Atualização Automática**: Dados atualizados a cada 30 segundos

### 3. Sistema de Notificações
- **Alertas Visuais**: Notificações coloridas por tipo (info, sucesso, aviso, erro)
- **Histórico de Notificações**: Painel dedicado para visualizar todas as notificações
- **Notificações de Trading**: Alertas automáticos para operações e eventos

### 4. Coleta de Dados MT5
- **Dados em Tempo Real**: Coleta direta de cotações e dados de mercado do MT5
- **Conexão Direta**: Acesso direto aos dados via MT5DataProvider
- **Status de Conexão**: Indicador visual do status da conexão MT5

### 5. Sistema de Exportação
- **Relatórios PDF**: Exportação de dados de performance e posições
- **Dados CSV**: Exportação de histórico de ordens e posições
- **Logs do Sistema**: Exportação de logs para análise

### 6. Configurações Personalizáveis
- **Preferências do Usuário**: Salvamento automático de configurações
- **Temas**: Alternância entre temas claro e escuro
- **Parâmetros de Trading**: Configuração de stop loss, take profit, etc.

### 7. Otimização de Performance
- **Monitoramento de Memória**: Controle automático do uso de RAM
- **Cache Inteligente**: Sistema de cache para dados frequentemente acessados
- **Relatórios de Performance**: Métricas detalhadas de desempenho

## Como Usar

### Inicialização

1. **Executar o Dashboard**:
   ```bash
   python mt5_dashboard_integrated.py
   ```

2. **Conectar ao MT5**:
   - O sistema tentará conectar automaticamente ao MetaTrader 5
   - Verifique se o MT5 está aberto e o trading automático está habilitado

### Navegação

#### Painel Principal
- **Aba Account**: Informações da conta e saldo
- **Aba Positions**: Posições abertas e histórico
- **Aba Trading**: Controles de trading e configurações
- **Aba Analytics**: Gráficos e análises de performance

#### Barra de Ferramentas
- **Botão Performance**: Abre relatório detalhado de performance
- **Botão Configurações**: Acesso às configurações do sistema
- **Botão Exportar**: Menu de opções de exportação
- **Indicador de Status**: Status da conexão MT5

### Funcionalidades Avançadas

#### Sistema de Notificações
1. **Visualizar Notificações**: Clique no painel de notificações
2. **Filtrar por Tipo**: Use os filtros por categoria (info, erro, etc.)
3. **Limpar Histórico**: Botão para limpar notificações antigas

#### Exportação de Dados
1. **Relatório PDF**:
   - Clique em "Exportar" → "Relatório PDF"
   - Escolha o local para salvar
   - O relatório inclui posições, performance e configurações

2. **Dados CSV**:
   - Clique em "Exportar" → "Dados CSV"
   - Selecione o tipo de dados (posições, ordens, etc.)
   - Escolha o período de dados

#### Configurações
1. **Preferências Gerais**:
   - Tema da interface
   - Frequência de atualização
   - Configurações de notificação

2. **Parâmetros de Trading**:
   - Stop loss padrão
   - Take profit padrão
   - Tamanho de lote padrão
   - Configurações de risco

#### Monitoramento de Performance
1. **Relatório de Performance**:
   - Clique no botão "Performance"
   - Visualize métricas de tempo de resposta
   - Monitore uso de memória
   - Verifique status das otimizações

## Solução de Problemas

### Problemas Comuns

#### Conexão MT5
- **Erro**: "MT5 não conectado"
- **Solução**: 
  1. Verifique se o MT5 está aberto
  2. Habilite o trading automático
  3. Verifique as configurações de firewall

#### Performance Lenta
- **Sintomas**: Interface lenta, alta utilização de memória
- **Solução**:
  1. Clique no botão "Performance"
  2. Verifique o relatório de otimização
  3. Reinicie o dashboard se necessário

#### Dados MT5 Não Atualizando
- **Sintomas**: Dados não atualizando em tempo real
- **Solução**:
  1. Verifique o status da conexão MT5 na barra de ferramentas
  2. Verifique se o MetaTrader 5 está aberto e conectado
  3. Reinicie o dashboard se necessário

### Logs e Diagnóstico

#### Localização dos Logs
- **Logs do Sistema**: `logs/debug.log`
- **Logs de Performance**: `logs/performance.log`
- **Logs de Erro**: `logs/error.log`

#### Informações de Diagnóstico
1. **Relatório de Performance**: Métricas detalhadas do sistema
2. **Status de Conexões**: Verificação de MT5
3. **Uso de Memória**: Monitoramento em tempo real

## Requisitos do Sistema

### Software
- Python 3.8+
- MetaTrader 5
- Bibliotecas Python (ver requirements.txt)

### Hardware Recomendado
- RAM: 4GB mínimo, 8GB recomendado
- CPU: Dual-core 2.0GHz mínimo
- Resolução: 1366x768 mínimo, 1920x1080 recomendado

## Atualizações e Manutenção

### Backup de Configurações
- As configurações são salvas automaticamente em `user_preferences.json`
- Faça backup regular deste arquivo

### Limpeza de Logs
- Os logs são rotacionados automaticamente
- Logs antigos são arquivados na pasta `logs/archive/`

### Monitoramento de Performance
- Execute o relatório de performance regularmente
- Monitore o uso de memória
- Verifique a integridade das conexões

## Suporte Técnico

Para suporte técnico:
1. Consulte os logs de erro
2. Execute o diagnóstico de performance
3. Verifique a documentação técnica
4. Reporte problemas com logs detalhados

---

**Versão**: 2.0  
**Última Atualização**: Janeiro 2025  
**Compatibilidade**: MT5 Build 3815+