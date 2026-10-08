#!/usr/bin/env python3
"""
Guia de Configuração MT5 - Sistema de Trading
Guia interativo para configurar conexão com MetaTrader 5
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import winreg
import psutil

def print_header(title):
    """Imprime cabeçalho formatado"""
    print("\n" + "=" * 60)
    print(f"🔧 {title}")
    print("=" * 60)

def print_step(step_num, description):
    """Imprime passo formatado"""
    print(f"\n📋 PASSO {step_num}: {description}")
    print("-" * 40)

def check_mt5_installation():
    """Verifica instalação do MT5"""
    print_header("VERIFICAÇÃO DA INSTALAÇÃO MT5")
    
    # Procurar MT5 no registro do Windows
    mt5_paths = []
    
    try:
        # Verificar registro HKEY_LOCAL_MACHINE
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall") as key:
            i = 0
            while True:
                try:
                    subkey_name = winreg.EnumKey(key, i)
                    with winreg.OpenKey(key, subkey_name) as subkey:
                        try:
                            display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                            if "MetaTrader 5" in display_name:
                                try:
                                    install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                                    mt5_paths.append((display_name, install_location))
                                except FileNotFoundError:
                                    pass
                        except FileNotFoundError:
                            pass
                    i += 1
                except OSError:
                    break
    except Exception as e:
        print(f"⚠️ Erro ao verificar registro: {e}")
    
    # Procurar em locais comuns
    common_paths = [
        r"C:\Program Files\MetaTrader 5",
        r"C:\Program Files (x86)\MetaTrader 5",
        r"C:\Users\{}\AppData\Roaming\MetaQuotes\Terminal".format(os.getenv('USERNAME')),
    ]
    
    # Adicionar caminhos encontrados no registro
    for _, path in mt5_paths:
        if path not in common_paths:
            common_paths.append(path)
    
    found_installations = []
    
    for path in common_paths:
        if os.path.exists(path):
            # Procurar terminal64.exe
            terminal_exe = os.path.join(path, "terminal64.exe")
            if os.path.exists(terminal_exe):
                found_installations.append(path)
    
    if found_installations:
        print("✅ Instalações MT5 encontradas:")
        for i, path in enumerate(found_installations, 1):
            print(f"   {i}. {path}")
        return found_installations
    else:
        print("❌ Nenhuma instalação MT5 encontrada!")
        print("\n📥 Para instalar o MT5:")
        print("1. Visite: https://www.metatrader5.com/pt/download")
        print("2. Baixe e instale o MetaTrader 5")
        print("3. Execute este script novamente")
        return []

def check_mt5_running():
    """Verifica se MT5 está executando"""
    print_header("VERIFICAÇÃO DE PROCESSOS MT5")
    
    mt5_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            if proc.info['name'] and 'terminal64.exe' in proc.info['name'].lower():
                mt5_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if mt5_processes:
        print("✅ MT5 está executando:")
        for proc in mt5_processes:
            print(f"   PID: {proc.info['pid']} - {proc.info['exe']}")
        return True
    else:
        print("❌ MT5 não está executando")
        return False

def launch_mt5(installation_path):
    """Lança o MT5"""
    print_step(1, "INICIANDO METATRADER 5")
    
    terminal_exe = os.path.join(installation_path, "terminal64.exe")
    
    if not os.path.exists(terminal_exe):
        print(f"❌ Executável não encontrado: {terminal_exe}")
        return False
    
    try:
        print(f"🚀 Iniciando MT5: {terminal_exe}")
        subprocess.Popen([terminal_exe], shell=True)
        print("✅ MT5 iniciado com sucesso!")
        print("⏳ Aguarde o MT5 carregar completamente...")
        time.sleep(5)
        return True
    except Exception as e:
        print(f"❌ Erro ao iniciar MT5: {e}")
        return False

def guide_account_setup():
    """Guia para configuração de conta"""
    print_step(2, "CONFIGURAÇÃO DE CONTA DEMO")
    
    print("📋 Siga estas instruções no MetaTrader 5:")
    print("\n1. 🏢 ESCOLHER CORRETORA:")
    print("   - Clique em 'File' > 'Open an Account'")
    print("   - Escolha uma corretora confiável (ex: XM, FXCM, ActivTrades)")
    print("   - Selecione 'Demo Account' para testes seguros")
    
    print("\n2. 📝 PREENCHER DADOS:")
    print("   - Nome: Seu nome completo")
    print("   - Email: Email válido")
    print("   - Telefone: Seu telefone")
    print("   - Tipo de conta: Standard ou ECN")
    print("   - Depósito inicial: $10,000 (recomendado para demo)")
    print("   - Alavancagem: 1:100 ou 1:500 (para iniciantes)")
    
    print("\n3. ✅ FINALIZAR CADASTRO:")
    print("   - Clique em 'Next' e aguarde aprovação")
    print("   - Anote o LOGIN e SENHA fornecidos")
    print("   - Teste a conexão fazendo login")
    
    input("\n⏸️ Pressione ENTER após configurar a conta demo...")

def test_connection():
    """Testa conexão com MT5"""
    print_step(3, "TESTE DE CONEXÃO")
    
    try:
        import MetaTrader5 as mt5
        
        print("🔄 Testando conexão com MT5...")
        
        if not mt5.initialize():
            error_code = mt5.last_error()
            print(f"❌ Falha na conexão: {error_code}")
            
            if error_code[0] == -6:
                print("\n🔐 ERRO DE AUTORIZAÇÃO:")
                print("1. Verifique se você fez login no MT5")
                print("2. Confirme que a conta está ativa")
                print("3. Verifique a conexão com internet")
                print("4. Tente reiniciar o MT5")
            
            return False
        
        # Obter informações da conta
        account_info = mt5.account_info()
        if account_info is None:
            print("❌ Não foi possível obter informações da conta")
            mt5.shutdown()
            return False
        
        print("✅ CONEXÃO ESTABELECIDA COM SUCESSO!")
        print(f"\n📊 INFORMAÇÕES DA CONTA:")
        print(f"   Login: {account_info.login}")
        print(f"   Servidor: {account_info.server}")
        print(f"   Saldo: ${account_info.balance:,.2f}")
        print(f"   Equity: ${account_info.equity:,.2f}")
        print(f"   Moeda: {account_info.currency}")
        print(f"   Tipo: {'Demo' if account_info.trade_mode == 1 else 'Real'}")
        
        # Testar símbolos
        symbols = mt5.symbols_get()
        if symbols:
            print(f"   Símbolos disponíveis: {len(symbols)}")
            print(f"   Principais: {', '.join([s.name for s in symbols[:5]])}")
        
        mt5.shutdown()
        return True
        
    except ImportError:
        print("❌ Biblioteca MetaTrader5 não instalada!")
        print("Execute: pip install MetaTrader5")
        return False
    except Exception as e:
        print(f"❌ Erro durante teste: {e}")
        return False

def create_startup_script():
    """Cria script de inicialização"""
    print_step(4, "CRIAÇÃO DE SCRIPT DE INICIALIZAÇÃO")
    
    startup_script = """@echo off
echo 🚀 Iniciando Sistema MT5 Trading...
echo.

REM Verificar se MT5 está executando
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo ✅ MT5 já está executando
) else (
    echo ⚠️ MT5 não está executando
    echo Por favor, inicie o MetaTrader 5 manualmente
    pause
    exit /b 1
)

echo.
echo 🔄 Iniciando sistema de trading...
python mt5_main.py --mode gui

pause
"""
    
    script_path = "start_mt5_system.bat"
    
    try:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(startup_script)
        
        print(f"✅ Script criado: {script_path}")
        print("💡 Use este script para iniciar o sistema facilmente!")
        
    except Exception as e:
        print(f"❌ Erro ao criar script: {e}")

def main():
    """Função principal do guia"""
    print("🎯 GUIA DE CONFIGURAÇÃO MT5")
    print("Este guia ajudará você a configurar o MetaTrader 5 para uso com o sistema de trading.")
    
    # Verificar instalação
    installations = check_mt5_installation()
    if not installations:
        return
    
    # Escolher instalação se houver múltiplas
    if len(installations) > 1:
        print("\n🔍 Múltiplas instalações encontradas:")
        for i, path in enumerate(installations, 1):
            print(f"   {i}. {path}")
        
        while True:
            try:
                choice = int(input("\nEscolha uma instalação (número): ")) - 1
                if 0 <= choice < len(installations):
                    selected_installation = installations[choice]
                    break
                else:
                    print("❌ Escolha inválida!")
            except ValueError:
                print("❌ Digite um número válido!")
    else:
        selected_installation = installations[0]
    
    print(f"\n✅ Usando instalação: {selected_installation}")
    
    # Verificar se MT5 está executando
    if not check_mt5_running():
        response = input("\n❓ Deseja iniciar o MT5 automaticamente? (s/n): ").lower()
        if response in ['s', 'sim', 'y', 'yes']:
            if not launch_mt5(selected_installation):
                print("❌ Falha ao iniciar MT5. Inicie manualmente e execute este script novamente.")
                return
        else:
            print("📋 Inicie o MetaTrader 5 manualmente e execute este script novamente.")
            return
    
    # Guiar configuração de conta
    guide_account_setup()
    
    # Testar conexão
    if test_connection():
        print("\n🎉 CONFIGURAÇÃO CONCLUÍDA COM SUCESSO!")
        
        # Criar script de inicialização
        create_startup_script()
        
        print("\n🚀 PRÓXIMOS PASSOS:")
        print("1. Execute: python mt5_main.py --mode gui")
        print("2. Ou use o script: start_mt5_system.bat")
        print("3. Configure os parâmetros de risco no arquivo config.json")
        print("4. Comece com valores pequenos para testes")
        
        print("\n⚠️ LEMBRETE DE SEGURANÇA:")
        print("- Sempre use conta DEMO para testes iniciais")
        print("- Nunca arrisque mais do que pode perder")
        print("- Monitore as operações constantemente")
        print("- Mantenha o sistema atualizado")
        
    else:
        print("\n❌ CONFIGURAÇÃO INCOMPLETA")
        print("Verifique os erros acima e tente novamente.")
        print("\n📞 SUPORTE:")
        print("- Verifique se o MT5 está logado em uma conta")
        print("- Confirme a conexão com internet")
        print("- Reinicie o MT5 se necessário")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Configuração cancelada pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        print("Por favor, reporte este erro para suporte.")