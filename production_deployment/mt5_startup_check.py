#!/usr/bin/env python3
"""
Script de Verificação e Diagnóstico MT5
Verifica a conectividade e configuração do MetaTrader 5
"""

import MetaTrader5 as mt5
import sys
import os
from pathlib import Path

def check_mt5_installation():
    """Verifica se o MT5 está instalado"""
    print("🔍 Verificando instalação do MetaTrader 5...")
    
    # Caminhos comuns do MT5
    common_paths = [
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
        r"C:\Program Files\ActivTrades Markets MetaTrader 5\terminal64.exe",
        r"C:\Program Files (x86)\ActivTrades Markets MetaTrader 5\terminal64.exe"
    ]
    
    found_paths = []
    for path in common_paths:
        if os.path.exists(path):
            found_paths.append(path)
            print(f"✅ Encontrado: {path}")
    
    if not found_paths:
        print("❌ MetaTrader 5 não encontrado nos caminhos padrão")
        return False
    
    return found_paths[0]

def test_mt5_connection():
    """Testa conexão com MT5"""
    print("\n🔗 Testando conexão com MetaTrader 5...")
    
    # Tentar inicializar MT5
    if not mt5.initialize():
        error_code = mt5.last_error()
        print(f"❌ Falha ao inicializar MT5. Código de erro: {error_code}")
        
        # Diagnóstico detalhado
        if error_code == (1, 'Terminal: no connection'):
            print("💡 Diagnóstico: Terminal não está conectado")
            print("   - Verifique se o MT5 está aberto")
            print("   - Verifique se está logado em uma conta")
            print("   - Verifique a conexão com a internet")
        elif error_code == (1, 'Terminal: not found'):
            print("💡 Diagnóstico: Terminal não encontrado")
            print("   - Verifique se o MT5 está instalado")
            print("   - Verifique se o caminho está correto")
        else:
            print(f"💡 Diagnóstico: Erro desconhecido - {error_code}")
        
        return False
    
    print("✅ Conexão com MT5 estabelecida com sucesso!")
    
    # Obter informações da conta
    account_info = mt5.account_info()
    if account_info:
        print(f"📊 Conta: {account_info.login}")
        print(f"💰 Saldo: {account_info.balance}")
        print(f"🏢 Servidor: {account_info.server}")
        print(f"💱 Moeda: {account_info.currency}")
        print(f"📈 Tipo: {'Demo' if account_info.trade_mode == 1 else 'Real'}")
    
    # Testar obtenção de símbolos
    symbols = mt5.symbols_get()
    if symbols:
        print(f"📈 Símbolos disponíveis: {len(symbols)}")
        print(f"   Primeiros 5: {[s.name for s in symbols[:5]]}")
    
    mt5.shutdown()
    return True

def check_python_packages():
    """Verifica pacotes Python necessários"""
    print("\n📦 Verificando pacotes Python...")
    
    required_packages = [
        'MetaTrader5',
        'pandas',
        'numpy',
        'talib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - FALTANDO")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Pacotes faltando: {', '.join(missing_packages)}")
        print("Execute: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Função principal de diagnóstico"""
    print("🚀 DIAGNÓSTICO DO SISTEMA MT5")
    print("=" * 50)
    
    # Verificar Python
    print(f"🐍 Python: {sys.version}")
    
    # Verificar pacotes
    packages_ok = check_python_packages()
    
    # Verificar instalação MT5
    mt5_path = check_mt5_installation()
    
    # Testar conexão
    if mt5_path and packages_ok:
        connection_ok = test_mt5_connection()
        
        if connection_ok:
            print("\n✅ SISTEMA PRONTO PARA OPERAÇÃO!")
            print("Você pode iniciar o sistema com:")
            print("python mt5_main.py --mode gui")
        else:
            print("\n❌ PROBLEMAS DE CONEXÃO DETECTADOS")
            print("Soluções recomendadas:")
            print("1. Abra o MetaTrader 5 manualmente")
            print("2. Faça login em uma conta (demo recomendado)")
            print("3. Verifique a conexão com a internet")
            print("4. Execute este script novamente")
    else:
        print("\n❌ PROBLEMAS DE CONFIGURAÇÃO DETECTADOS")
        print("Resolva os problemas acima antes de continuar")

if __name__ == "__main__":
    main()