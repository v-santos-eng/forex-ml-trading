#!/usr/bin/env python3
"""
Script de Inicialização do Sistema de Recuperação MT5
Autor: Assistente IA
Data: 2025-08-19

Este script inicializa o sistema de recuperação automática para componentes MT5.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Adicionar diretório pai ao path
sys.path.append(str(Path(__file__).parent.parent))

from production_deployment.mt5_recovery_system import MT5RecoverySystem

def setup_logging():
    """Configura logging para o script de inicialização"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/mt5_recovery_startup.log')
        ]
    )
    return logging.getLogger('MT5RecoveryStartup')

def check_existing_processes():
    """Verifica processos existentes que podem ser integrados"""
    logger = logging.getLogger('MT5RecoveryStartup')
    
    # Verificar arquivos PID existentes
    pid_files = {
        'ml_pipeline': 'ml_pipeline.pid',
        'production_monitor': 'production_monitor.pid',
        'mt5_dashboard': 'mt5_dashboard.pid'
    }
    
    active_processes = {}
    
    for component, pid_file in pid_files.items():
        if os.path.exists(pid_file):
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # Verificar se processo está ativo (Windows)
                import psutil
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    if process.is_running():
                        active_processes[component] = {
                            'pid': pid,
                            'name': process.name(),
                            'status': 'running'
                        }
                        logger.info(f"Processo ativo encontrado: {component} (PID: {pid})")
                    else:
                        logger.warning(f"Arquivo PID encontrado mas processo não está ativo: {component}")
                        os.remove(pid_file)
                else:
                    logger.warning(f"PID {pid} não existe, removendo arquivo: {pid_file}")
                    os.remove(pid_file)
                    
            except Exception as e:
                logger.error(f"Erro ao verificar processo {component}: {e}")
    
    return active_processes

def start_recovery_system(config_file=None, monitor=True, daemon=False):
    """Inicia o sistema de recuperação MT5"""
    logger = setup_logging()
    
    try:
        logger.info("=== INICIANDO SISTEMA DE RECUPERAÇÃO MT5 ===")
        
        # Verificar processos existentes
        active_processes = check_existing_processes()
        if active_processes:
            logger.info(f"Processos ativos encontrados: {list(active_processes.keys())}")
        
        # Inicializar sistema de recuperação
        config_path = config_file or 'mt5_recovery_config.json'
        recovery_system = MT5RecoverySystem(config_file=config_path)
        
        # Adicionar processos ativos ao sistema de monitoramento
        for component, process_info in active_processes.items():
            try:
                recovery_system.register_existing_process(
                    component, 
                    process_info['pid']
                )
                logger.info(f"Processo {component} registrado no sistema de recuperação")
            except Exception as e:
                logger.warning(f"Não foi possível registrar processo {component}: {e}")
        
        # Iniciar componentes não ativos
        components_to_start = ['mt5_dashboard', 'ml_pipeline', 'production_monitor']
        
        for component in components_to_start:
            if component not in active_processes:
                try:
                    logger.info(f"Iniciando componente: {component}")
                    success = recovery_system.start_component(component)
                    if success:
                        logger.info(f"Componente {component} iniciado com sucesso")
                    else:
                        logger.error(f"Falha ao iniciar componente {component}")
                except Exception as e:
                    logger.error(f"Erro ao iniciar componente {component}: {e}")
            else:
                logger.info(f"Componente {component} já está ativo")
        
        # Iniciar monitoramento se solicitado
        if monitor:
            logger.info("Iniciando monitoramento automático...")
            
            if daemon:
                logger.info("Executando em modo daemon")
                recovery_system.start_monitoring()
                
                # Manter processo ativo
                try:
                    while True:
                        time.sleep(60)
                        status = recovery_system.get_system_status()
                        logger.info(f"Status do sistema: {status['summary']}")
                        
                except KeyboardInterrupt:
                    logger.info("Interrupção recebida, parando sistema...")
                    recovery_system.shutdown()
            else:
                # Monitoramento por tempo limitado
                logger.info("Executando monitoramento por 5 minutos...")
                recovery_system.start_monitoring()
                
                for i in range(30):  # 5 minutos
                    time.sleep(10)
                    status = recovery_system.get_system_status()
                    logger.info(f"[{(i+1)*10}s] {status['summary']}")
                
                recovery_system.stop_monitoring()
                logger.info("Monitoramento concluído")
        
        # Gerar relatório final
        status = recovery_system.get_system_status()
        logger.info("=== STATUS FINAL DO SISTEMA ===")
        logger.info(f"Componentes ativos: {status['active_components']}")
        logger.info(f"Componentes com falha: {status['failed_components']}")
        logger.info(f"Sistema saudável: {status['system_healthy']}")
        
        # Salvar relatório
        report_file = recovery_system.generate_health_report()
        logger.info(f"Relatório de saúde salvo em: {report_file}")
        
        return recovery_system
        
    except Exception as e:
        logger.error(f"Erro crítico no sistema de recuperação: {e}")
        raise

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Sistema de Recuperação Automática MT5',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python start_mt5_recovery.py                    # Iniciar com configuração padrão
  python start_mt5_recovery.py --daemon           # Executar em modo daemon
  python start_mt5_recovery.py --no-monitor       # Apenas iniciar componentes
  python start_mt5_recovery.py --config custom.json  # Usar configuração customizada
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='Arquivo de configuração JSON (padrão: mt5_recovery_config.json)'
    )
    
    parser.add_argument(
        '--daemon', 
        action='store_true', 
        help='Executar em modo daemon (monitoramento contínuo)'
    )
    
    parser.add_argument(
        '--no-monitor', 
        action='store_true', 
        help='Apenas iniciar componentes, sem monitoramento'
    )
    
    parser.add_argument(
        '--test', 
        action='store_true', 
        help='Executar em modo de teste (monitoramento por 5 minutos)'
    )
    
    args = parser.parse_args()
    
    try:
        # Criar diretórios necessários
        Path('logs').mkdir(exist_ok=True)
        Path('recovery_reports').mkdir(exist_ok=True)
        Path('health_reports').mkdir(exist_ok=True)
        
        # Iniciar sistema
        monitor = not args.no_monitor
        daemon = args.daemon and not args.test
        
        recovery_system = start_recovery_system(
            config_file=args.config,
            monitor=monitor,
            daemon=daemon
        )
        
        if not daemon and not args.no_monitor:
            recovery_system.shutdown()
            
        print("\n=== SISTEMA DE RECUPERAÇÃO MT5 INICIADO COM SUCESSO ===")
        
    except KeyboardInterrupt:
        print("\nInterrupção recebida, encerrando...")
    except Exception as e:
        print(f"\nErro: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()