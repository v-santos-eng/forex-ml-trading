#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste de Verificação de Sessão Existente MT5
Demonstra a funcionalidade de reutilização de sessão implementada
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar MetaTrader5 se disponível
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    print("MetaTrader5 não disponível - executando em modo simulação")

from typing import Dict, Any

class SessionVerificationTest:
    """Classe para testar verificação de sessão MT5"""
    
    def __init__(self):
        self.login = int(os.getenv("MT5_LOGIN", "0"))
        self.password = os.getenv("MT5_PASSWORD", "")
        self.server = "ActivTradesMarkets-Server"
        self.connected = False
        self.account_info = None
    
    def check_existing_session(self) -> Dict[str, Any]:
        """Verifica se já existe uma sessão ativa do MT5"""
        print("\n=== VERIFICAÇÃO DE SESSÃO EXISTENTE ===")
        
        if not MT5_AVAILABLE:
            print("❌ MT5 não disponível - modo simulação")
            return {
                'session_found': False,
                'reason': 'MT5 não disponível',
                'simulation': True
            }
        
        try:
            print("🔍 Verificando se MT5 já está inicializado...")
            
            # Verificar se MT5 já está inicializado
            account_info = mt5.account_info()
            
            if account_info is not None:
                print(f"✅ Sessão MT5 ativa encontrada!")
                print(f"   Login atual: {account_info.login}")
                print(f"   Servidor atual: {account_info.server}")
                print(f"   Login esperado: {self.login}")
                print(f"   Servidor esperado: {self.server}")
                
                # Verificar se é a mesma conta
                if account_info.login == self.login and account_info.server == self.server:
                    print("🎉 Sessão compatível encontrada - pode ser reutilizada!")
                    self.connected = True
                    self.account_info = account_info._asdict()
                    
                    return {
                        'session_found': True,
                        'compatible': True,
                        'login': account_info.login,
                        'server': account_info.server,
                        'balance': account_info.balance,
                        'equity': account_info.equity,
                        'reason': 'Sessão compatível reutilizada'
                    }
                else:
                    print("⚠️  Sessão incompatível encontrada")
                    print("   Será necessário desconectar e reconectar")
                    
                    # Desconectar da conta diferente
                    mt5.shutdown()
                    print("🔄 Sessão anterior desconectada")
                    
                    return {
                        'session_found': True,
                        'compatible': False,
                        'reason': 'Sessão incompatível - desconectada',
                        'previous_login': account_info.login,
                        'previous_server': account_info.server
                    }
            else:
                print("ℹ️  Nenhuma sessão MT5 ativa encontrada")
                return {
                    'session_found': False,
                    'reason': 'Nenhuma sessão ativa'
                }
                
        except Exception as e:
            print(f"❌ Erro ao verificar sessão existente: {e}")
            return {
                'session_found': False,
                'reason': f'Erro: {str(e)}',
                'error': True
            }
    
    def attempt_connection(self) -> Dict[str, Any]:
        """Tenta conectar ao MT5 com verificação de sessão"""
        print("\n=== TENTATIVA DE CONEXÃO ===")
        
        # Primeiro, verificar sessão existente
        session_check = self.check_existing_session()
        
        if session_check.get('session_found') and session_check.get('compatible'):
            print("✅ Usando sessão existente - conexão não necessária")
            return {
                'success': True,
                'message': f'Sessão reutilizada - Login: {session_check["login"]}, Servidor: {session_check["server"]}',
                'reused_session': True,
                'session_info': session_check
            }
        
        if not MT5_AVAILABLE:
            print("🔄 Iniciando modo simulação...")
            self.connected = True
            self.account_info = {
                'login': self.login,
                'server': self.server,
                'balance': 10000.0,
                'equity': 10000.0,
                'margin': 0.0
            }
            return {
                'success': True,
                'message': 'Conectado em modo simulação',
                'simulation': True,
                'reused_session': False
            }
        
        try:
            print(f"🔄 Inicializando MT5...")
            if not mt5.initialize():
                error = mt5.last_error()
                print(f"❌ Falha na inicialização: {error}")
                return {
                    'success': False,
                    'message': f'Falha na inicialização MT5: {error}'
                }
            
            print(f"🔐 Tentando login...")
            print(f"   Login: {self.login}")
            print(f"   Servidor: {self.server}")
            
            if not mt5.login(self.login, password=self.password, server=self.server):
                error = mt5.last_error()
                print(f"❌ Falha no login: {error}")
                mt5.shutdown()
                return {
                    'success': False,
                    'message': f'Falha no login: {error}'
                }
            
            # Obter informações da conta
            account_info = mt5.account_info()
            if account_info is None:
                print("❌ Falha ao obter informações da conta")
                mt5.shutdown()
                return {
                    'success': False,
                    'message': 'Falha ao obter informações da conta'
                }
            
            self.connected = True
            self.account_info = account_info._asdict()
            
            print(f"✅ Conexão estabelecida com sucesso!")
            print(f"   Login: {account_info.login}")
            print(f"   Servidor: {account_info.server}")
            print(f"   Saldo: ${account_info.balance:,.2f}")
            
            return {
                'success': True,
                'message': f'Conectado ao servidor {self.server}',
                'simulation': False,
                'reused_session': False,
                'account_info': self.account_info
            }
            
        except Exception as e:
            print(f"❌ Erro na conexão: {e}")
            return {
                'success': False,
                'message': f'Erro na conexão: {str(e)}'
            }
    
    def disconnect(self):
        """Desconecta do MT5"""
        print("\n=== DESCONEXÃO ===")
        
        try:
            if MT5_AVAILABLE and self.connected:
                mt5.shutdown()
                print("✅ Desconectado do MT5")
            
            self.connected = False
            self.account_info = None
            
        except Exception as e:
            print(f"❌ Erro na desconexão: {e}")
    
    def run_test(self):
        """Executa o teste completo"""
        print("\n" + "="*60)
        print("TESTE DE VERIFICAÇÃO DE SESSÃO MT5")
        print("="*60)
        print(f"Configuração:")
        print(f"  Login: {self.login}")
        print(f"  Servidor: {self.server}")
        print(f"  MT5 Disponível: {MT5_AVAILABLE}")
        
        # Teste 1: Verificação inicial
        print("\n📋 TESTE 1: Verificação de sessão inicial")
        session_check = self.check_existing_session()
        print(f"Resultado: {session_check}")
        
        # Teste 2: Tentativa de conexão
        print("\n📋 TESTE 2: Tentativa de conexão")
        connection_result = self.attempt_connection()
        print(f"Resultado: {connection_result}")
        
        # Teste 3: Segunda verificação (deve encontrar sessão)
        if connection_result.get('success'):
            print("\n📋 TESTE 3: Verificação após conexão")
            session_check2 = self.check_existing_session()
            print(f"Resultado: {session_check2}")
            
            # Teste 4: Segunda tentativa de conexão (deve reutilizar)
            print("\n📋 TESTE 4: Segunda tentativa de conexão")
            connection_result2 = self.attempt_connection()
            print(f"Resultado: {connection_result2}")
        
        # Cleanup
        self.disconnect()
        
        print("\n" + "="*60)
        print("TESTE CONCLUÍDO")
        print("="*60)

def main():
    """Função principal"""
    test = SessionVerificationTest()
    test.run_test()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTeste interrompido pelo usuário")
    except Exception as e:
        print(f"\nErro no teste: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nTeste finalizado")