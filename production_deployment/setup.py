#!/usr/bin/env python3
"""
Script de Setup e Instalação do Sistema MT5 Trading

Este script automatiza a instalação e configuração inicial do sistema de trading MT5,
incluindo a instalação de dependências, criação de diretórios e configuração inicial.
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any

class MT5SystemSetup:
    """
    Classe para setup e instalação do sistema MT5
    """
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.requirements = [
            'MetaTrader5>=5.0.45',
            'pandas>=1.5.0',
            'numpy>=1.21.0',
            'yfinance>=0.2.0',
            'ta>=0.10.0',
            'python-dateutil>=2.8.0',
            'pytz>=2022.1'
        ]
        
        self.directories = [
            'logs',
            'data',
            'backups',
            'performance',
            'config',
            'reports'
        ]
    
    def check_python_version(self) -> bool:
        """Verifica se a versão do Python é compatível"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Erro: Python 3.8 ou superior é necessário")
            print(f"   Versão atual: {version.major}.{version.minor}.{version.micro}")
            return False
        
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    
    def check_mt5_installation(self) -> bool:
        """Verifica se o MetaTrader 5 está instalado"""
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                info = mt5.terminal_info()
                if info:
                    print(f"✅ MetaTrader 5 encontrado - Versão: {info.build}")
                    mt5.shutdown()
                    return True
                else:
                    print("⚠️  MetaTrader 5 encontrado mas não está configurado")
                    mt5.shutdown()
                    return True
            else:
                print("❌ MetaTrader 5 não pôde ser inicializado")
                return False
        except ImportError:
            print("❌ Biblioteca MetaTrader5 não encontrada")
            return False
        except Exception as e:
            print(f"❌ Erro ao verificar MT5: {e}")
            return False
    
    def install_requirements(self) -> bool:
        """Instala as dependências necessárias"""
        print("\n📦 Instalando dependências...")
        
        try:
            # Criar arquivo requirements.txt temporário
            req_file = self.base_dir / 'requirements_temp.txt'
            with open(req_file, 'w') as f:
                for req in self.requirements:
                    f.write(f"{req}\n")
            
            # Instalar dependências
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', str(req_file)
            ], capture_output=True, text=True)
            
            # Remover arquivo temporário
            req_file.unlink()
            
            if result.returncode == 0:
                print("✅ Dependências instaladas com sucesso")
                return True
            else:
                print(f"❌ Erro na instalação: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao instalar dependências: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Cria os diretórios necessários"""
        print("\n📁 Criando estrutura de diretórios...")
        
        try:
            for directory in self.directories:
                dir_path = self.base_dir / directory
                dir_path.mkdir(exist_ok=True)
                print(f"   ✅ {directory}/")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao criar diretórios: {e}")
            return False
    
    def create_config_file(self) -> bool:
        """Cria arquivo de configuração personalizado"""
        print("\n⚙️  Configurando sistema...")
        
        try:
            config_source = self.base_dir / 'config_example.json'
            config_target = self.base_dir / 'config' / 'config.json'
            
            if not config_source.exists():
                print("❌ Arquivo de configuração exemplo não encontrado")
                return False
            
            # Copiar configuração exemplo
            shutil.copy2(config_source, config_target)
            
            # Personalizar configuração
            with open(config_target, 'r') as f:
                config = json.load(f)
            
            # Solicitar configurações básicas do usuário
            print("\n📝 Configuração inicial:")
            
            # Símbolos de trading
            symbols_input = input("   Símbolos para trading (separados por vírgula) [EURUSD,GBPUSD,USDJPY]: ")
            if symbols_input.strip():
                symbols = [s.strip().upper() for s in symbols_input.split(',')]
                config['trading']['symbols'] = symbols
            
            # Nível de risco
            risk_levels = ['LOW', 'MEDIUM', 'HIGH']
            print("   Níveis de risco disponíveis: LOW, MEDIUM, HIGH")
            risk_input = input("   Nível de risco [MEDIUM]: ").upper()
            if risk_input in risk_levels:
                config['risk_management']['risk_level'] = risk_input
            
            # Risco por trade
            risk_per_trade = input("   Risco máximo por trade (%) [2]: ")
            if risk_per_trade.strip():
                try:
                    risk_value = float(risk_per_trade) / 100
                    config['risk_management']['max_risk_per_trade'] = risk_value
                except ValueError:
                    pass
            
            # Auto trading
            auto_trading = input("   Habilitar auto trading? (s/n) [n]: ").lower()
            config['trading']['enable_auto_trading'] = auto_trading == 's'
            
            # Salvar configuração
            with open(config_target, 'w') as f:
                json.dump(config, f, indent=2)
            
            print("✅ Configuração criada em config/config.json")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao criar configuração: {e}")
            return False
    
    def create_startup_scripts(self) -> bool:
        """Cria scripts de inicialização"""
        print("\n🚀 Criando scripts de inicialização...")
        
        try:
            # Script para Windows
            windows_script = self.base_dir / 'start_mt5_system.bat'
            with open(windows_script, 'w') as f:
                f.write(f"""@echo off
echo Iniciando Sistema MT5 Trading...
cd /d "{self.base_dir}"
python mt5_main.py --mode gui
pause
""")
            
            # Script para console
            console_script = self.base_dir / 'start_console.bat'
            with open(console_script, 'w') as f:
                f.write(f"""@echo off
echo Iniciando Sistema MT5 Trading - Modo Console...
cd /d "{self.base_dir}"
python mt5_main.py --mode console
pause
""")
            
            # Script para análise
            analysis_script = self.base_dir / 'run_analysis.bat'
            with open(analysis_script, 'w') as f:
                f.write(f"""@echo off
echo Executando Análise MT5...
cd /d "{self.base_dir}"
python mt5_main.py --mode analysis
pause
""")
            
            print("   ✅ start_mt5_system.bat")
            print("   ✅ start_console.bat")
            print("   ✅ run_analysis.bat")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao criar scripts: {e}")
            return False
    
    def create_requirements_file(self) -> bool:
        """Cria arquivo requirements.txt"""
        try:
            req_file = self.base_dir / 'requirements.txt'
            with open(req_file, 'w') as f:
                for req in self.requirements:
                    f.write(f"{req}\n")
            
            print("✅ requirements.txt criado")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao criar requirements.txt: {e}")
            return False
    
    def test_system(self) -> bool:
        """Testa se o sistema está funcionando"""
        print("\n🧪 Testando sistema...")
        
        try:
            # Testar importações
            print("   Testando importações...")
            
            modules_to_test = [
                'mt5_connector',
                'mt5_order_manager',
                'mt5_data_provider',
                'mt5_risk_manager',
                'mt5_signal_integrator',
                'mt5_logger',
                'forex_analyzer'
            ]
            
            for module in modules_to_test:
                try:
                    __import__(module)
                    print(f"     ✅ {module}")
                except ImportError as e:
                    print(f"     ❌ {module}: {e}")
                    return False
            
            # Testar conexão MT5 (opcional)
            print("   Testando conexão MT5...")
            try:
                import MetaTrader5 as mt5
                if mt5.initialize():
                    print("     ✅ Conexão MT5 OK")
                    mt5.shutdown()
                else:
                    print("     ⚠️  MT5 não conectado (normal se não estiver configurado)")
            except Exception as e:
                print(f"     ⚠️  Erro MT5: {e}")
            
            print("✅ Testes concluídos")
            return True
            
        except Exception as e:
            print(f"❌ Erro nos testes: {e}")
            return False
    
    def show_next_steps(self):
        """Mostra próximos passos para o usuário"""
        print("\n" + "="*60)
        print("🎉 INSTALAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*60)
        print("\n📋 PRÓXIMOS PASSOS:")
        print("\n1. 📊 Configure o MetaTrader 5:")
        print("   - Abra o MT5 e faça login em sua conta")
        print("   - Habilite 'Permitir DLL' nas configurações")
        print("   - Habilite 'Permitir WebRequest' para URLs necessárias")
        print("\n2. ⚙️  Personalize a configuração:")
        print("   - Edite config/config.json conforme necessário")
        print("   - Configure símbolos, timeframes e parâmetros de risco")
        print("\n3. 🚀 Execute o sistema:")
        print("   - Interface Gráfica: execute start_mt5_system.bat")
        print("   - Modo Console: execute start_console.bat")
        print("   - Análise Manual: execute run_analysis.bat")
        print("\n4. 📚 Documentação:")
        print("   - Leia o README.md para informações detalhadas")
        print("   - Consulte os logs em logs/ para troubleshooting")
        print("\n5. ⚠️  IMPORTANTE:")
        print("   - SEMPRE teste em conta demo primeiro")
        print("   - Configure stop loss e take profit adequados")
        print("   - Monitore o sistema regularmente")
        print("   - Nunca arrisque mais do que pode perder")
        print("\n" + "="*60)
        print("📧 Suporte: Consulte a documentação ou logs para troubleshooting")
        print("="*60)
    
    def run_setup(self) -> bool:
        """Executa o setup completo"""
        print("🔧 SETUP DO SISTEMA MT5 TRADING")
        print("="*50)
        
        steps = [
            ("Verificando Python", self.check_python_version),
            ("Verificando MT5", self.check_mt5_installation),
            ("Instalando dependências", self.install_requirements),
            ("Criando diretórios", self.create_directories),
            ("Criando configuração", self.create_config_file),
            ("Criando scripts", self.create_startup_scripts),
            ("Criando requirements.txt", self.create_requirements_file),
            ("Testando sistema", self.test_system)
        ]
        
        for step_name, step_func in steps:
            print(f"\n🔄 {step_name}...")
            if not step_func():
                print(f"\n❌ Setup falhou na etapa: {step_name}")
                return False
        
        self.show_next_steps()
        return True

def main():
    """Função principal"""
    try:
        setup = MT5SystemSetup()
        
        print("Bem-vindo ao Setup do Sistema MT5 Trading!")
        print("Este script irá configurar automaticamente o sistema.\n")
        
        response = input("Deseja continuar com a instalação? (s/n): ")
        if response.lower() != 's':
            print("Setup cancelado.")
            return
        
        success = setup.run_setup()
        
        if success:
            print("\n✅ Setup concluído com sucesso!")
        else:
            print("\n❌ Setup falhou. Verifique os erros acima.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrompido pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")

if __name__ == "__main__":
    main()