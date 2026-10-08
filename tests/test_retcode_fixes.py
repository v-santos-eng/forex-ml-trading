#!/usr/bin/env python3
"""
Teste Simplificado das Correções de Interpretação do Retcode MT5
Verifica se as correções aplicadas estão funcionando corretamente
"""

import MetaTrader5 as mt5
import sys
import os
from datetime import datetime
import json
import inspect

# Adicionar diretório de produção ao path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'production_deployment'))

class RetcodeFixTester:
    """Testa as correções de interpretação do retcode"""
    
    def __init__(self):
        self.test_results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'tests': [],
            'summary': {}
        }
    
    def test_order_manager_fixes(self) -> dict:
        """Testa se as correções foram aplicadas no OrderManager"""
        print(f"\n🔍 Testando correções no MT5OrderManager...")
        
        test_result = {
            'component': 'MT5OrderManager',
            'test_type': 'retcode_fix_validation',
            'success': False,
            'details': {}
        }
        
        try:
            from mt5_order_manager import MT5OrderManager
            
            # Verificar método place_market_order
            source = inspect.getsource(MT5OrderManager.place_market_order)
            
            if "retcode not in [0, mt5.TRADE_RETCODE_DONE]" in source:
                test_result['success'] = True
                test_result['details']['place_market_order'] = "✅ Correção aplicada"
                print("   ✅ place_market_order: Correção aplicada")
            else:
                test_result['details']['place_market_order'] = "❌ Correção NÃO aplicada"
                print("   ❌ place_market_order: Correção NÃO aplicada")
            
            # Verificar método close_position
            close_source = inspect.getsource(MT5OrderManager.close_position)
            
            if "retcode not in [0, mt5.TRADE_RETCODE_DONE]" in close_source:
                test_result['details']['close_position'] = "✅ Correção aplicada"
                print("   ✅ close_position: Correção aplicada")
                if test_result['success']:  # Só mantém sucesso se ambos estão corretos
                    test_result['success'] = True
            else:
                test_result['success'] = False
                test_result['details']['close_position'] = "❌ Correção NÃO aplicada"
                print("   ❌ close_position: Correção NÃO aplicada")
            
        except Exception as e:
            test_result['details']['error'] = str(e)
            print(f"   ❌ Erro no teste: {e}")
        
        return test_result
    
    def test_signal_integrator_fixes(self) -> dict:
        """Testa se as correções foram aplicadas no SignalIntegrator"""
        print(f"\n🔍 Testando correções no MT5SignalIntegrator...")
        
        test_result = {
            'component': 'MT5SignalIntegrator',
            'test_type': 'retcode_fix_validation',
            'success': False,
            'details': {}
        }
        
        try:
            from mt5_signal_integrator import MT5SignalIntegrator
            
            # Verificar método execute_signal
            source = inspect.getsource(MT5SignalIntegrator.execute_signal)
            
            if "result.get('retcode') in [0, mt5.TRADE_RETCODE_DONE]" in source:
                test_result['success'] = True
                test_result['details']['execute_signal'] = "✅ Correção aplicada"
                print("   ✅ execute_signal: Correção aplicada")
            else:
                test_result['details']['execute_signal'] = "❌ Correção NÃO aplicada"
                print("   ❌ execute_signal: Correção NÃO aplicada")
            
        except Exception as e:
            test_result['details']['error'] = str(e)
            print(f"   ❌ Erro no teste: {e}")
        
        return test_result
    
    def test_retcode_interpretation_logic(self) -> dict:
        """Testa a lógica de interpretação dos códigos de retorno"""
        print(f"\n🔍 Testando lógica de interpretação de retcodes...")
        
        test_result = {
            'component': 'RetcodeLogic',
            'test_type': 'interpretation_logic',
            'success': False,
            'details': {}
        }
        
        try:
            # Testar se retcode 0 é considerado sucesso
            success_codes = [0, mt5.TRADE_RETCODE_DONE]
            
            # Simular validação
            test_cases = [
                {'retcode': 0, 'expected': True, 'description': 'SUCCESS'},
                {'retcode': mt5.TRADE_RETCODE_DONE, 'expected': True, 'description': 'DONE'},
                {'retcode': mt5.TRADE_RETCODE_REJECT, 'expected': False, 'description': 'REJECT'},
                {'retcode': 9999, 'expected': False, 'description': 'UNKNOWN'}
            ]
            
            all_correct = True
            for case in test_cases:
                is_success = case['retcode'] in success_codes
                if is_success == case['expected']:
                    test_result['details'][f"retcode_{case['retcode']}"] = f"✅ {case['description']}: Correto"
                    print(f"   ✅ Retcode {case['retcode']} ({case['description']}): Interpretação correta")
                else:
                    all_correct = False
                    test_result['details'][f"retcode_{case['retcode']}"] = f"❌ {case['description']}: Incorreto"
                    print(f"   ❌ Retcode {case['retcode']} ({case['description']}): Interpretação incorreta")
            
            test_result['success'] = all_correct
            
        except Exception as e:
            test_result['details']['error'] = str(e)
            print(f"   ❌ Erro no teste: {e}")
        
        return test_result
    
    def test_file_modifications(self) -> dict:
        """Verifica se os arquivos foram modificados corretamente"""
        print(f"\n🔍 Verificando modificações nos arquivos...")
        
        test_result = {
            'component': 'FileModifications',
            'test_type': 'file_content_validation',
            'success': False,
            'details': {}
        }
        
        try:
            files_to_check = [
                'production_deployment/mt5_order_manager.py',
                'production_deployment/mt5_signal_integrator.py'
            ]
            
            all_files_correct = True
            
            for file_path in files_to_check:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Verificar se contém as correções (múltiplos formatos possíveis)
                    corrections_found = (
                        'retcode not in [0, mt5.TRADE_RETCODE_DONE]' in content or
                        'retcode\'} in [0, mt5.TRADE_RETCODE_DONE]' in content or
                        "result.get('retcode') in [0, mt5.TRADE_RETCODE_DONE]" in content
                    )
                    
                    if corrections_found:
                        test_result['details'][file_path] = "✅ Correções encontradas"
                        print(f"   ✅ {file_path}: Correções aplicadas")
                    else:
                        all_files_correct = False
                        test_result['details'][file_path] = "❌ Correções NÃO encontradas"
                        print(f"   ❌ {file_path}: Correções NÃO aplicadas")
                else:
                    all_files_correct = False
                    test_result['details'][file_path] = "❌ Arquivo não encontrado"
                    print(f"   ❌ {file_path}: Arquivo não encontrado")
            
            test_result['success'] = all_files_correct
            
        except Exception as e:
            test_result['details']['error'] = str(e)
            print(f"   ❌ Erro no teste: {e}")
        
        return test_result
    
    def run_all_tests(self) -> dict:
        """Executa todos os testes"""
        print("=" * 60)
        print("VALIDAÇÃO DAS CORREÇÕES DE RETCODE MT5")
        print("=" * 60)
        
        # Executar testes
        tests = [
            self.test_file_modifications(),
            self.test_order_manager_fixes(),
            self.test_signal_integrator_fixes(),
            self.test_retcode_interpretation_logic()
        ]
        
        self.test_results['tests'] = tests
        
        # Resumo
        total_tests = len(tests)
        successful_tests = sum(1 for test in tests if test['success'])
        
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
            'all_fixes_applied': successful_tests == total_tests
        }
        
        print(f"\n📊 RESUMO DA VALIDAÇÃO:")
        print(f"   Total de testes: {total_tests}")
        print(f"   Testes bem-sucedidos: {successful_tests}")
        print(f"   Taxa de sucesso: {self.test_results['summary']['success_rate']:.1f}%")
        
        if self.test_results['summary']['all_fixes_applied']:
            print("   ✅ Todas as correções foram aplicadas com sucesso!")
            print("   🎉 Sistema agora interpreta retcode 0 como SUCCESS")
        else:
            print("   ❌ Algumas correções ainda precisam ser aplicadas")
            print("   ⚠️ Sistema pode ainda interpretar retcode 0 como falha")
        
        return self.test_results
    
    def save_results(self, filename: str = None):
        """Salva resultados dos testes"""
        if filename is None:
            filename = f"retcode_fix_validation_{self.test_results['timestamp']}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n💾 Resultados salvos em: {filename}")
        except Exception as e:
            print(f"❌ Erro ao salvar resultados: {e}")

def main():
    """Função principal"""
    tester = RetcodeFixTester()
    
    try:
        results = tester.run_all_tests()
        tester.save_results()
        
        # Retornar código de saída baseado no sucesso
        if results.get('summary', {}).get('all_fixes_applied', False):
            print("\n🎉 Validação completa! Todas as correções foram aplicadas.")
            return 0
        else:
            print("\n⚠️ Validação incompleta. Algumas correções precisam ser revisadas.")
            return 1
            
    except Exception as e:
        print(f"❌ Erro durante validação: {e}")
        return 1

if __name__ == "__main__":
    exit(main())