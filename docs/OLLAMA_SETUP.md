# Guia de Configuração do Ollama para Agno ML Forex

Este guia explica como configurar o Ollama para usar modelos de IA locais gratuitos em vez das APIs pagas.

## O que é o Ollama?

O Ollama é uma ferramenta que permite executar modelos de linguagem grandes (LLMs) localmente em seu computador, sem necessidade de APIs pagas ou conexão com a internet.

### Vantagens:
- **Gratuito**: Sem custos de API
- **Privado**: Dados não saem do seu computador
- **Offline**: Funciona sem internet
- **Rápido**: Processamento local

## Instalação do Ollama

### Windows

1. **Baixar o Ollama**:
   - Acesse: https://ollama.com/download
   - Baixe o instalador para Windows
   - Execute o arquivo `.exe` baixado

2. **Verificar Instalação**:
   ```powershell
   ollama --version
   ```

### Alternativa: Instalação Manual

Se preferir instalar manualmente:

```powershell
# Baixar e instalar via PowerShell
Invoke-WebRequest -Uri "https://ollama.com/install.ps1" -OutFile "install.ps1"
.\install.ps1
```

## Configuração Inicial

### 1. Iniciar o Servidor Ollama

```powershell
# Iniciar o servidor Ollama
ollama serve
```

**Importante**: Mantenha este terminal aberto enquanto usar o sistema.

### 2. Baixar Modelos Recomendados

Em um novo terminal:

```powershell
# Modelo principal (recomendado)
ollama pull llama3.2

# Modelos alternativos (opcionais)
ollama pull llama3.2:1b    # Versão mais leve
ollama pull mistral        # Alternativa rápida
ollama pull codellama      # Especializado em código
```

### 3. Verificar Modelos Instalados

```powershell
ollama list
```

### 4. Testar Funcionamento

```powershell
# Teste básico
ollama run llama3.2 "Hello, how are you?"
```

## Configuração para o Agno ML Forex

### 1. Verificar Requisitos Python

Certifique-se de que as dependências estão instaladas:

```powershell
pip install requests agno
```

### 2. Testar Integração

```powershell
# Executar o demo com Ollama
python demo_agno_ollama.py
```

### 3. Verificar Status

```powershell
# Testar apenas os agentes Ollama
python agno_ollama_agents.py

# Testar apenas o sistema de memória
python agno_ollama_memory.py
```

## Modelos Recomendados

### Para Trading Forex:

1. **llama3.2** (Padrão)
   - Tamanho: ~4.7GB
   - Qualidade: Excelente
   - Velocidade: Boa
   ```powershell
   ollama pull llama3.2
   ```

2. **llama3.2:1b** (Leve)
   - Tamanho: ~1.3GB
   - Qualidade: Boa
   - Velocidade: Muito rápida
   ```powershell
   ollama pull llama3.2:1b
   ```

3. **mistral** (Alternativa)
   - Tamanho: ~4.1GB
   - Qualidade: Excelente
   - Velocidade: Rápida
   ```powershell
   ollama pull mistral
   ```

## Configuração Avançada

### Alterar Modelo Padrão

Para usar um modelo diferente, edite os arquivos:

```python
# Em agno_ollama_agents.py
team = OllamaForexAgentTeam(model_id="mistral")  # Em vez de "llama3.2"

# Em agno_ollama_memory.py
memory = OllamaForexMemorySystem(model_id="mistral")
```

### Configurar Host Personalizado

Se o Ollama estiver em outro servidor:

```python
team = OllamaForexAgentTeam(
    model_id="llama3.2",
    host="http://192.168.1.100:11434"  # IP do servidor
)
```

### Otimização de Performance

1. **Aumentar Memória GPU** (se disponível):
   ```bash
   # Definir variável de ambiente
   set OLLAMA_GPU_MEMORY=8GB
   ```

2. **Configurar Threads CPU**:
   ```bash
   set OLLAMA_NUM_THREADS=8
   ```

## Solução de Problemas

### Problema: "Connection refused"

**Solução**:
```powershell
# Verificar se o servidor está rodando
ollama serve

# Em outro terminal, testar
curl http://localhost:11434/api/tags
```

### Problema: "Model not found"

**Solução**:
```powershell
# Listar modelos instalados
ollama list

# Baixar o modelo necessário
ollama pull llama3.2
```

### Problema: Muito lento

**Soluções**:
1. Use um modelo menor: `llama3.2:1b`
2. Feche outros programas pesados
3. Considere usar GPU se disponível

### Problema: Erro de memória

**Soluções**:
1. Use modelo menor
2. Feche outros programas
3. Reinicie o Ollama:
   ```powershell
   # Parar o servidor (Ctrl+C)
   # Depois reiniciar
   ollama serve
   ```

## Comandos Úteis

### Gerenciamento de Modelos

```powershell
# Listar modelos
ollama list

# Remover modelo
ollama rm llama3.2

# Atualizar modelo
ollama pull llama3.2

# Informações do modelo
ollama show llama3.2
```

### Monitoramento

```powershell
# Ver logs do servidor
ollama logs

# Status do sistema
ollama ps
```

## Integração com o Sistema

### Arquivos Principais

1. **agno_ollama_agents.py**: Agentes de trading usando Ollama
2. **agno_ollama_memory.py**: Sistema de memória usando Ollama
3. **demo_agno_ollama.py**: Demo completo com Ollama

### Fluxo de Uso

1. Iniciar Ollama: `ollama serve`
2. Executar demo: `python demo_agno_ollama.py`
3. Verificar resultados nos arquivos JSON gerados

## Comparação: Ollama vs APIs Pagas

| Aspecto | Ollama (Local) | APIs Pagas |
|---------|----------------|------------|
| Custo | Gratuito | Pago por uso |
| Privacidade | Total | Dados enviados |
| Velocidade | Depende do HW | Geralmente rápida |
| Qualidade | Boa a Excelente | Excelente |
| Disponibilidade | Sempre | Depende de quota |
| Setup | Mais complexo | Simples |

## Próximos Passos

1. **Instalar Ollama** seguindo este guia
2. **Baixar modelo**: `ollama pull llama3.2`
3. **Testar sistema**: `python demo_agno_ollama.py`
4. **Personalizar** conforme necessário

## Suporte

- **Documentação Ollama**: https://ollama.com/docs
- **Modelos disponíveis**: https://ollama.com/library
- **GitHub Ollama**: https://github.com/ollama/ollama

---

**Nota**: Este sistema Ollama substitui completamente a necessidade de APIs pagas, fornecendo uma solução de IA local, gratuita e privada para trading de Forex.