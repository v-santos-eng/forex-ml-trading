#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Teste para Sistema de Recuperação MT5
Testa funcionalidades de recuperação automática e monitoramento
"""

import os
import sys
import time
import json
import threading
from datetime import datetime
from pathlib import Path

# Adicionar diretório atual ao path
sys.path.append('.')

try:
    from mt5_recovery_system import MT5RecoverySystem, ComponentType, RecoveryStrategy, HealthStatus
    RECOVERY_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Erro ao importar sistema de recuperação: {e}")
    RECOVERY_SYSTEM_AVAILABLE = False

class MT5RecoveryTester:
    """Classe para testar sistema de recuperação MT5"""
    
    def __init__(self):
        self.recovery_system = None
        self.test_results = []
        
    def setup_test_environment(self):
        """Configura ambiente de teste"""
        print("=== CONFIGURANDO AMBIENTE DE TESTE ===")
        
        # Criar diretórios necessários
        directories = ["logs", "recovery", "backups", "health_reports", "recovery_reports"]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            print(f"Diretório criado/verificado: {directory}")
        
        # Inicializar sistema de recuperação
        if RECOVERY_SYSTEM_AVAILABLE:
            config_file = "mt5_recovery_config.json"
            if os.path.exists(config_file):
                print(f"Carregando configuração de: {config_file}")
                self.recovery_system = MT5RecoverySystem(config_file)
            else:
                print("Usando configuração padrão")
                self.recovery_system = MT5RecoverySystem()
            
            print(f"Sistema de recuperação inicializado com {len(self.recovery_system.components)} componentes")
        else:
            print("Sistema de recuperação não disponível - executando em modo simulação")
        
        print("Ambiente de teste configurado com sucesso\n")
    
    def test_component_health_check(self):
        """Testa verificação de saúde dos componentes"""
        print("=== TESTE: VERIFICAÇÃO DE SAÚDE DOS COMPONENTES ===")
        
        if not self.recovery_system:
            print("Sistema de recuperação não disponível")
            self.test_results.append({"test": "health_check", "status": "skipped", "reason": "system_unavailable"})
            return
        
        try:
            for component_name in self.recovery_system.components:
                print(f"Verificando saúde do componente: {component_name}")
                
                health = self.recovery_system.check_component_health(component_name)
                
                print(f"  Status: {health.status.value}")
                print(f"  CPU: {health.cpu_usage:.1f}%")
                print(f"  Memória: {health.memory_usage:.1f}%")
                print(f"  Falhas consecutivas: {health.consecutive_failures}")
                print(f"  Última verificação: {health.last_check}")
                print()
            
            self.test_results.append({"test": "health_check", "status": "success", "timestamp": datetime.now()})
            print("Teste de verificação de saúde concluído com sucesso\n")
            
        except Exception as e:
            print(f"Erro no teste de verificação de saúde: {e}")
            self.test_results.append({"test": "health_check", "status": "failed", "error": str(e)})
    
    def test_component_start_stop(self):
        """Testa inicialização e parada de componentes"""
        print("=== TESTE: INICIALIZAÇÃO E PARADA DE COMPONENTES ===")
        
        if not self.recovery_system:
            print("Sistema de recuperação não disponível")
            self.test_results.append({"test": "start_stop", "status": "skipped", "reason": "system_unavailable"})
            return
        
        try:
            # Testar com componente de teste
            test_component = "production_monitor"
            
            if test_component in self.recovery_system.components:
                print(f"Testando inicialização do componente: {test_component}")
                
                # Tentar iniciar
                start_success = self.recovery_system.start_component(test_component)
                print(f"Inicialização: {'Sucesso' if start_success else 'Falha'}")
                
                if start_success:
                    # Aguardar um pouco
                    print("Aguardando 5 segundos...")
                    time.sleep(5)
                    
                    # Verificar se está rodando
                    health = self.recovery_system.check_component_health(test_component)
                    print(f"Status após inicialização: {health.status.value}")
                    
                    # Parar componente
                    print(f"Parando componente: {test_component}")
                    stop_success = self.recovery_system.stop_component(test_component)
                    print(f"Parada: {'Sucesso' if stop_success else 'Falha'}")
                
                self.test_results.append({
                    "test": "start_stop", 
                    "status": "success" if start_success else "partial", 
                    "component": test_component,
                    "start_success": start_success,
                    "timestamp": datetime.now()
                })
            else:
                print(f"Componente {test_component} não encontrado")
                self.test_results.append({"test": "start_stop", "status": "failed", "reason": "component_not_found"})
            
            print("Teste de inicialização/parada concluído\n")
            
        except Exception as e:
            print(f"Erro no teste de inicialização/parada: {e}")
            self.test_results.append({"test": "start_stop", "status": "failed", "error": str(e)})
    
    def test_recovery_mechanism(self):
        """Testa mecanismo de recuperação"""
        print("=== TESTE: MECANISMO DE RECUPERAÇÃO ===")
        
        if not self.recovery_system:
            print("Sistema de recuperação não disponível")
            self.test_results.append({"test": "recovery", "status": "skipped", "reason": "system_unavailable"})
            return
        
        try:
            # Testar recuperação de componente
            test_component = "production_monitor"
            
            if test_component in self.recovery_system.components:
                print(f"Testando recuperação do componente: {test_component}")
                
                # Simular falha parando o componente
                if test_component in self.recovery_system.active_processes:
                    print("Parando componente para simular falha...")
                    self.recovery_system.stop_component(test_component)
                
                # Executar recuperação
                print("Executando recuperação...")
                recovery_success = self.recovery_system.recover_component(test_component)
                print(f"Recuperação: {'Sucesso' if recovery_success else 'Falha'}")
                
                if recovery_success:
                    # Verificar métricas de recuperação
                    metrics = self.recovery_system.recovery_metrics[test_component]
                    print(f"Métricas de recuperação:")
                    print(f"  Total de recuperações: {metrics.total_recoveries}")
                    print(f"  Recuperações bem-sucedidas: {metrics.successful_recoveries}")
                    print(f"  Taxa de sucesso: {metrics.recovery_success_rate:.1f}%")
                    print(f"  Tempo médio: {metrics.average_recovery_time:.2f}s")
                
                self.test_results.append({
                    "test": "recovery", 
                    "status": "success" if recovery_success else "failed", 
                    "component": test_component,
                    "recovery_success": recovery_success,
                    "timestamp": datetime.now()
                })
            else:
                print(f"Componente {test_component} não encontrado")
                self.test_results.append({"test": "recovery", "status": "failed", "reason": "component_not_found"})
            
            print("Teste de recuperação concluído\n")
            
        except Exception as e:
            print(f"Erro no teste de recuperação: {e}")
            self.test_results.append({"test": "recovery", "status": "failed", "error": str(e)})
    
    def test_monitoring_system(self):
        """Testa sistema de monitoramento"""
        print("=== TESTE: SISTEMA DE MONITORAMENTO ===")
        
        if not self.recovery_system:
            print("Sistema de recuperação não disponível")
            self.test_results.append({"test": "monitoring", "status": "skipped", "reason": "system_unavailable"})
            return
        
        try:
            print("Iniciando monitoramento...")
            self.recovery_system.start_monitoring()
            
            print("Monitoramento ativo. Aguardando 30 segundos para observar comportamento...")
            
            # Monitorar por 30 segundos
            for i in range(6):
                time.sleep(5)
                status = self.recovery_system.get_system_status()
                print(f"[{i*5+5}s] Componentes monitorados: {status['summary']['total_components']} | "
                      f"Saudáveis: {status['summary']['healthy_components']} | "
                      f"Falhados: {status['summary']['failed_components']}")
            
            print("Parando monitoramento...")
            self.recovery_system.stop_monitoring()
            
            self.test_results.append({"test": "monitoring", "status": "success", "timestamp": datetime.now()})
            print("Teste de monitoramento concluído\n")
            
        except Exception as e:
            print(f"Erro no teste de monitoramento: {e}")
            self.test_results.append({"test": "monitoring", "status": "failed", "error": str(e)})
    
    def test_system_status_report(self):
        """Testa geração de relatórios de status"""
        print("=== TESTE: RELATÓRIOS DE STATUS ===")
        
        if not self.recovery_system:
            print("Sistema de recuperação não disponível")
            self.test_results.append({"test": "status_report", "status": "skipped", "reason": "system_unavailable"})
            return
        
        try:
            # Gerar status do sistema
            print("Gerando status do sistema...")
            status = self.recovery_system.get_system_status()
            
            print("Status do Sistema:")
            print(f"  Timestamp: {status['timestamp']}")
            print(f"  Monitoramento ativo: {status['monitoring_active']}")
            print(f"  Total de componentes: {status['summary']['total_components']}")
            print(f"  Componentes saudáveis: {status['summary']['healthy_components']}")
            print(f"  Componentes falhados: {status['summary']['failed_components']}")
            print(f"  Total de recuperações: {status['summary']['total_recoveries']}")
            print()
            
            # Gerar relatório de saúde
            print("Gerando relatório de saúde...")
            health_report = self.recovery_system.generate_health_report()
            
            # Salvar relatório
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"health_reports/test_health_report_{timestamp}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(health_report)
            
            print(f"Relatório de saúde salvo em: {report_file}")
            print("Primeiras linhas do relatório:")
            print(health_report[:500] + "..." if len(health_report) > 500 else health_report)
            
            self.test_results.append({
                "test": "status_report", 
                "status": "success", 
                "report_file": report_file,
                "timestamp": datetime.now()
            })
            print("Teste de relatórios concluído\n")
            
        except Exception as e:
            print(f"Erro no teste de relatórios: {e}")
            self.test_results.append({"test": "status_report", "status": "failed", "error": str(e)})
    
    def test_configuration_loading(self):
        """Testa carregamento de configuração"""
        print("=== TESTE: CARREGAMENTO DE CONFIGURAÇÃO ===")
        
        try:
            config_file = "mt5_recovery_config.json"
            
            if os.path.exists(config_file):
                print(f"Testando carregamento de configuração de: {config_file}")
                
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                print("Configuração carregada com sucesso")
                print(f"Componentes configurados: {len(config_data.get('components', {}))}")
                print(f"Estratégias de recuperação: {len(config_data.get('recovery_strategies', {}))}")
                
                # Listar componentes
                print("Componentes configurados:")
                for comp_name, comp_config in config_data.get('components', {}).items():
                    print(f"  - {comp_name}: {comp_config.get('component_type', 'N/A')} "
                          f"({'Crítico' if comp_config.get('critical', False) else 'Normal'})")
                
                self.test_results.append({"test": "config_loading", "status": "success", "timestamp": datetime.now()})
            else:
                print(f"Arquivo de configuração não encontrado: {config_file}")
                self.test_results.append({"test": "config_loading", "status": "failed", "reason": "file_not_found"})
            
            print("Teste de carregamento de configuração concluído\n")
            
        except Exception as e:
            print(f"Erro no teste de configuração: {e}")
            self.test_results.append({"test": "config_loading", "status": "failed", "error": str(e)})
    
    def run_all_tests(self):
        """Executa todos os testes"""
        print("=== INICIANDO TESTES DO SISTEMA DE RECUPERAÇÃO MT5 ===")
        print(f"Timestamp: {datetime.now()}")
        print()
        
        # Setup
        self.setup_test_environment()
        
        # Executar testes
        test_methods = [
            self.test_configuration_loading,
            self.test_component_health_check,
            self.test_system_status_report,
            self.test_component_start_stop,
            self.test_recovery_mechanism,
            self.test_monitoring_system
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"Erro crítico no teste {test_method.__name__}: {e}")
                self.test_results.append({
                    "test": test_method.__name__, 
                    "status": "critical_error", 
                    "error": str(e)
                })
        
        # Gerar relatório final
        self.generate_test_report()
        
        # Cleanup
        if self.recovery_system:
            print("Executando limpeza...")
            self.recovery_system.shutdown()
    
    def generate_test_report(self):
        """Gera relatório final dos testes"""
        print("=== RELATÓRIO FINAL DOS TESTES ===")
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r['status'] == 'success'])
        failed_tests = len([r for r in self.test_results if r['status'] in ['failed', 'critical_error']])
        skipped_tests = len([r for r in self.test_results if r['status'] == 'skipped'])
        
        print(f"Total de testes: {total_tests}")
        print(f"Testes bem-sucedidos: {successful_tests}")
        print(f"Testes falhados: {failed_tests}")
        print(f"Testes ignorados: {skipped_tests}")
        print(f"Taxa de sucesso: {(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
        print()
        
        print("Detalhes dos testes:")
        for result in self.test_results:
            status_icon = {
                'success': '✓',
                'failed': '✗',
                'skipped': '⊝',
                'partial': '⚠',
                'critical_error': '💥'
            }.get(result['status'], '?')
            
            print(f"  {status_icon} {result['test']}: {result['status']}")
            if 'error' in result:
                print(f"    Erro: {result['error']}")
            if 'reason' in result:
                print(f"    Motivo: {result['reason']}")
        
        # Salvar relatório
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"recovery_reports/test_report_{timestamp}.json"
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "skipped_tests": skipped_tests,
                "success_rate": (successful_tests/total_tests*100) if total_tests > 0 else 0
            },
            "test_results": self.test_results
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\nRelatório detalhado salvo em: {report_file}")
        print("=== TESTES CONCLUÍDOS ===")

def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Testes do Sistema de Recuperação MT5")
    parser.add_argument("--test", choices=[
        "all", "config", "health", "start_stop", "recovery", "monitoring", "reports"
    ], default="all", help="Teste específico para executar")
    
    args = parser.parse_args()
    
    tester = MT5RecoveryTester()
    
    if args.test == "all":
        tester.run_all_tests()
    elif args.test == "config":
        tester.setup_test_environment()
        tester.test_configuration_loading()
    elif args.test == "health":
        tester.setup_test_environment()
        tester.test_component_health_check()
    elif args.test == "start_stop":
        tester.setup_test_environment()
        tester.test_component_start_stop()
    elif args.test == "recovery":
        tester.setup_test_environment()
        tester.test_recovery_mechanism()
    elif args.test == "monitoring":
        tester.setup_test_environment()
        tester.test_monitoring_system()
    elif args.test == "reports":
        tester.setup_test_environment()
        tester.test_system_status_report()
    
    if tester.recovery_system:
        tester.recovery_system.shutdown()

if __name__ == "__main__":
    main()