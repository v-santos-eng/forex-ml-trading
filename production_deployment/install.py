#!/usr/bin/env python3
"""
Script de Instalação Automatizada - Forex ML Trading System
Automatiza a instalação e configuração do sistema de produção

Autor: Sistema ML Forex
Versão: 1.0.0
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Verifica se a versão do Python é compatível"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ é necessário")
        print(f"Versão atual: {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor} detectado")
    return True

def install_requirements():
    """Instala as dependências do requirements.txt"""
    print("\n📦 Instalando dependências...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependências instaladas com sucesso")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro na instalação: {e}")
        return False

def check_talib_installation():
    """Verifica se TA-Lib está instalado corretamente"""
    print("\n🔍 Verificando TA-Lib...")
    try:
        import talib
        print("✓ TA-Lib instalado e funcionando")
        return True
    except ImportError:
        print("❌ TA-Lib não encontrado")
        print("\n📋 Instruções para instalar TA-Lib:")
        
        if platform.system() == "Windows":
            print("Windows:")
            print("1. Baixe o wheel de: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
            print("2. pip install TA_Lib-0.4.25-cp39-cp39-win_amd64.whl")
        else:
            print("Linux/Mac:")
            print("sudo apt-get install build-essential")
            print("wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz")
            print("tar -xzf ta-lib-0.4.0-src.tar.gz")
            print("cd ta-lib/ && ./configure --prefix=/usr && make && sudo make install")
            print("pip install TA-Lib")
        
        return False

def create_env_file():
    """Cria arquivo .env com configurações padrão"""
    env_path = Path(".env")
    if env_path.exists():
        print("✓ Arquivo .env já existe")
        return True
    
    print("\n📝 Criando arquivo .env...")
    env_content = """# Configurações do Forex ML Trading System

# API Keys (opcional)
# ANTHROPIC_API_KEY=your_key_here

# Configurações de dados
DATA_SOURCE=yfinance
DATA_INTERVAL=1h

# Configurações de ML
MODEL_RETRAIN_INTERVAL=24h
FEATURE_SELECTION=auto

# Configurações de risco
MAX_POSITION_SIZE=0.02
STOP_LOSS=0.01
TAKE_PROFIT=0.02

# Configurações de logging
LOG_LEVEL=INFO
LOG_TO_FILE=true
"""
    
    try:
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("✓ Arquivo .env criado")
        return True
    except Exception as e:
        print(f"❌ Erro ao criar .env: {e}")
        return False

def test_imports():
    """Testa importações principais"""
    print("\n🧪 Testando importações...")
    
    test_modules = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', None),
        ('xgboost', 'xgb'),
        ('lightgbm', 'lgb'),
        ('yfinance', 'yf'),
    ]
    
    failed_imports = []
    
    for module, alias in test_modules:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"✓ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Módulos com falha: {', '.join(failed_imports)}")
        return False
    
    print("✓ Todas as importações funcionando")
    return True

def verify_models():
    """Verifica se os modelos estão disponíveis"""
    print("\n🤖 Verificando modelos...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Diretório de modelos não encontrado")
        return False
    
    model_files = list(models_dir.glob("*.joblib"))
    if not model_files:
        print("❌ Nenhum modelo .joblib encontrado")
        return False
    
    print(f"✓ {len(model_files)} modelos encontrados")
    for model in model_files:
        print(f"  - {model.name}")
    
    return True

def main():
    """Função principal de instalação"""
    print("=" * 60)
    print("FOREX ML TRADING SYSTEM - INSTALAÇÃO")
    print("=" * 60)
    
    # Verificações
    checks = [
        ("Versão do Python", check_python_version),
        ("Dependências", install_requirements),
        ("TA-Lib", check_talib_installation),
        ("Arquivo .env", create_env_file),
        ("Importações", test_imports),
        ("Modelos", verify_models),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{'='*20} {name} {'='*20}")
        result = check_func()
        results.append((name, result))
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO DA INSTALAÇÃO")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASSOU" if result else "❌ FALHOU"
        print(f"{name:.<30} {status}")
        if not result:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 INSTALAÇÃO CONCLUÍDA COM SUCESSO!")
        print("\nPróximos passos:")
        print("1. python main.py")
        print("2. Configurar APIs no arquivo .env (se necessário)")
        print("3. Executar testes: python -c 'from main import main; main()'")
    else:
        print("❌ INSTALAÇÃO INCOMPLETA")
        print("\nCorreja os problemas acima e execute novamente.")
    
    print("="*60)
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)