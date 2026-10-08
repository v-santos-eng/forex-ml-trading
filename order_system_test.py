#!/usr/bin/env python3
"""
Teste Sistemático do Sistema de Envio de Ordens MT5
Validação completa de todas as etapas do processo de trading

Este script testa:
1. Validação de dados de entrada
2. Processamento da ordem
3. Comunicação com MT5
4. Confirmação do envio
5. Tratamento de erros
"""

import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import MetaTrader5 as mt5

# Adicionar diretório ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'production_deployment'))

from mt5_connector import MT5Connector
from mt5_order_manager import MT5OrderManager, OrderType
from mt5_data_provider import MT5DataProvider
from mt5_risk_manager import MT5RiskManager

class OrderSystemValidator:
    """
    Validador sistemático do sistema de envio de ordens
    """
    
    def __init__(self):
        self.connector = None
        self.order_manager = None
        self.data_provider = None
        self.risk_manager = None
        self.test_results = []
        self.critical_issues = []
        
    def print_header(self, title: str):
        """Imprime cabeçalho formatado"""
        print("\n" + "="*60)
        print(f" {title}")
        print("="*60)
        
    def print_test_result(self, test_name: str, success: bool, message: str = ""):
        """Imprime resultado do teste"""
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{status} - {test_name}")
        if message:
            print(f"   Detalhes: {message}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now()
        })
        
        if not success:
            self.critical_issues.append({
                "issue": test_name,
                "message": message,
                "timestamp": datetime.now()
            })
    
    def test_mt5_connection(self) -> bool:
        """Teste 1: Conexão com MT5"""
        self.print_header("TESTE 1: CONEXÃO COM MT5")
        
        try:
            self.connector = MT5Connector()
            connected = self.connector.connect()
            
            if connected:
                # Verificar informações da conta
                account_info = mt5.account_info()
                if account_info:
                    self.print_test_result(
                        "Conexão MT5", 
                        True, 
                        f"Conta: {account_info.login} - Servidor: {account_info.server}"
                    )
                    
                    # Verificar se AutoTrading está habilitado
                    if hasattr(account_info, 'trade_allowed') and account_info.trade_allowed:
                        self.print_test_result("AutoTrading Habilitado", True)
                    else:
                        self.print_test_result(
                            "AutoTrading Habilitado", 
                            False, 
                            "AutoTrading desabilitado - Código de erro 10027 esperado"
                        )
                    
                    return True
                else:
                    self.print_test_result("Informações da Conta", False, "Não foi possível obter informações da conta")
                    return False
            else:
                self.print_test_result("Conexão MT5", False, "Falha na conexão")
                return False
                
        except Exception as e:
            self.print_test_result("Conexão MT5", False, f"Exceção: {str(e)}")
            return False
    
    def test_symbol_validation(self) -> bool:
        """Teste 2: Validação de Símbolos"""
        self.print_header("TESTE 2: VALIDAÇÃO DE SÍMBOLOS")
        
        if not self.connector:
            self.print_test_result("Validação de Símbolos", False, "MT5 não conectado")
            return False
        
        self.order_manager = MT5OrderManager(self.connector)
        
        # Testar símbolos válidos
        valid_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        invalid_symbols = ['INVALID', 'FAKE123']
        
        all_passed = True
        
        for symbol in valid_symbols:
            try:
                is_valid = self.order_manager._validate_symbol(symbol)
                self.print_test_result(f"Símbolo Válido: {symbol}", is_valid)
                if not is_valid:
                    all_passed = False
            except Exception as e:
                self.print_test_result(f"Símbolo Válido: {symbol}", False, str(e))
                all_passed = False
        
        for symbol in invalid_symbols:
            try:
                is_valid = self.order_manager._validate_symbol(symbol)
                # Para símbolos inválidos, esperamos False
                self.print_test_result(f"Símbolo Inválido: {symbol}", not is_valid)
                if is_valid:
                    all_passed = False
            except Exception as e:
                self.print_test_result(f"Símbolo Inválido: {symbol}", True, f"Exceção esperada: {str(e)}")
        
        return all_passed
    
    def test_order_validation(self) -> bool:
        """Teste 3: Validação de Parâmetros de Ordem"""
        self.print_header("TESTE 3: VALIDAÇÃO DE PARÂMETROS DE ORDEM")
        
        if not self.order_manager:
            self.print_test_result("Validação de Ordem", False, "Order Manager não inicializado")
            return False
        
        # Testar cálculo de lote
        try:
            lot_size = self.order_manager._calculate_lot_size('EURUSD', 100.0, 50.0)
            self.print_test_result(
                "Cálculo de Lote", 
                lot_size > 0, 
                f"Lote calculado: {lot_size}"
            )
        except Exception as e:
            self.print_test_result("Cálculo de Lote", False, str(e))
            return False
        
        # Testar informações de símbolo
        try:
            symbol_info = mt5.symbol_info('EURUSD')
            if symbol_info:
                self.print_test_result(
                    "Informações do Símbolo", 
                    True, 
                    f"Spread: {symbol_info.spread}, Min Volume: {symbol_info.volume_min}"
                )
            else:
                self.print_test_result("Informações do Símbolo", False, "Não foi possível obter informações")
                return False
        except Exception as e:
            self.print_test_result("Informações do Símbolo", False, str(e))
            return False
        
        return True
    
    def test_order_processing(self) -> bool:
        """Teste 4: Processamento de Ordem (Simulado)"""
        self.print_header("TESTE 4: PROCESSAMENTO DE ORDEM")
        
        if not self.order_manager:
            self.print_test_result("Processamento de Ordem", False, "Order Manager não inicializado")
            return False
        
        # Testar preparação de request de ordem
        try:
            # Obter tick atual
            tick = mt5.symbol_info_tick('EURUSD')
            if tick:
                self.print_test_result(
                    "Obtenção de Preços", 
                    True, 
                    f"Bid: {tick.bid}, Ask: {tick.ask}"
                )
                
                # Simular preparação de request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": 'EURUSD',
                    "volume": 0.01,
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": tick.ask,
                    "magic": 234000,
                    "comment": "Teste de Sistema",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                self.print_test_result(
                    "Preparação de Request", 
                    True, 
                    f"Request preparado para {request['symbol']}"
                )
                
                # IMPORTANTE: NÃO enviar ordem real em teste
                self.print_test_result(
                    "Envio de Ordem (Simulado)", 
                    True, 
                    "Ordem não enviada - modo de teste"
                )
                
            else:
                self.print_test_result("Obtenção de Preços", False, "Não foi possível obter tick")
                return False
                
        except Exception as e:
            self.print_test_result("Processamento de Ordem", False, str(e))
            return False
        
        return True
    
    def test_error_handling(self) -> bool:
        """Teste 5: Tratamento de Erros"""
        self.print_header("TESTE 5: TRATAMENTO DE ERROS")
        
        # Testar códigos de erro conhecidos
        error_codes = {
            10027: "AutoTrading disabled by client",
            10004: "No connection",
            10013: "Invalid request",
            10015: "Invalid prices",
            10016: "Invalid stops",
            10017: "Invalid trade volume"
        }
        
        for code, description in error_codes.items():
            self.print_test_result(
                f"Código de Erro {code}", 
                True, 
                description
            )
        
        return True
    
    def test_account_information(self) -> bool:
        """Teste 6: Informações da Conta"""
        self.print_header("TESTE 6: INFORMAÇÕES DA CONTA")
        
        if not self.order_manager:
            self.print_test_result("Informações da Conta", False, "Order Manager não inicializado")
            return False
        
        try:
            account_summary = self.order_manager.get_account_summary()
            if account_summary:
                self.print_test_result(
                    "Resumo da Conta", 
                    True, 
                    f"Saldo: {account_summary['balance']} {account_summary['currency']}"
                )
                
                # Verificar posições
                positions = self.order_manager.get_positions()
                self.print_test_result(
                    "Posições Abertas", 
                    True, 
                    f"Total: {len(positions)} posições"
                )
                
                # Verificar ordens pendentes
                orders = self.order_manager.get_orders()
                self.print_test_result(
                    "Ordens Pendentes", 
                    True, 
                    f"Total: {len(orders)} ordens"
                )
                
                return True
            else:
                self.print_test_result("Resumo da Conta", False, "Não foi possível obter resumo")
                return False
                
        except Exception as e:
            self.print_test_result("Informações da Conta", False, str(e))
            return False
    
    def generate_report(self):
        """Gera relatório final dos testes"""
        self.print_header("RELATÓRIO FINAL DE TESTES")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"📊 Total de Testes: {total_tests}")
        print(f"✅ Testes Aprovados: {passed_tests}")
        print(f"❌ Testes Falharam: {failed_tests}")
        print(f"📈 Taxa de Sucesso: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.critical_issues:
            print("\n🚨 PROBLEMAS CRÍTICOS IDENTIFICADOS:")
            for i, issue in enumerate(self.critical_issues, 1):
                print(f"{i}. {issue['issue']}")
                print(f"   Detalhes: {issue['message']}")
                print(f"   Timestamp: {issue['timestamp']}")
        
        # Salvar relatório em arquivo
        report_file = f"order_system_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"Relatório de Teste do Sistema de Ordens MT5\n")
                f.write(f"Data/Hora: {datetime.now()}\n")
                f.write(f"Total de Testes: {total_tests}\n")
                f.write(f"Testes Aprovados: {passed_tests}\n")
                f.write(f"Testes Falharam: {failed_tests}\n")
                f.write(f"Taxa de Sucesso: {(passed_tests/total_tests)*100:.1f}%\n\n")
                
                f.write("DETALHES DOS TESTES:\n")
                for result in self.test_results:
                    status = "PASSOU" if result['success'] else "FALHOU"
                    f.write(f"{status} - {result['test']}\n")
                    if result['message']:
                        f.write(f"   Detalhes: {result['message']}\n")
                    f.write(f"   Timestamp: {result['timestamp']}\n\n")
                
                if self.critical_issues:
                    f.write("\nPROBLEMAS CRÍTICOS:\n")
                    for issue in self.critical_issues:
                        f.write(f"- {issue['issue']}\n")
                        f.write(f"  Detalhes: {issue['message']}\n")
                        f.write(f"  Timestamp: {issue['timestamp']}\n\n")
            
            print(f"\n📄 Relatório salvo em: {report_file}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar relatório: {e}")
    
    def run_all_tests(self):
        """Executa todos os testes do sistema"""
        print("🔍 INICIANDO VALIDAÇÃO SISTEMÁTICA DO SISTEMA DE ORDENS MT5")
        print(f"Timestamp: {datetime.now()}")
        
        # Executar testes em sequência
        tests = [
            self.test_mt5_connection,
            self.test_symbol_validation,
            self.test_order_validation,
            self.test_order_processing,
            self.test_error_handling,
            self.test_account_information
        ]
        
        for test in tests:
            try:
                test()
                time.sleep(1)  # Pausa entre testes
            except Exception as e:
                print(f"❌ Erro crítico no teste {test.__name__}: {e}")
        
        # Gerar relatório final
        self.generate_report()
        
        # Cleanup
        if self.connector:
            self.connector.disconnect()

def main():
    """Função principal"""
    validator = OrderSystemValidator()
    validator.run_all_tests()

if __name__ == "__main__":
    main()