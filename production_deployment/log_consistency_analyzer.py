#!/usr/bin/env python3
"""
Analisador de Consistência de Logs
Verifica a consistência entre logs de diferentes componentes do sistema MT5
"""

import os
import re
from datetime import datetime
from collections import defaultdict
import pandas as pd

class LogConsistencyAnalyzer:
    """
    Analisa consistência entre logs de diferentes componentes
    """
    
    def __init__(self, log_directory):
        self.log_directory = log_directory
        self.signal_integrator_logs = []
        self.order_manager_logs = []
        self.risk_manager_logs = []
        self.data_provider_logs = []
        
    def parse_log_line(self, line):
        """
        Extrai informações de uma linha de log
        """
        # Padrão: 2025-08-19 08:45:07 - ComponentName - LEVEL - Message
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - ([^-]+) - ([^-]+) - (.+)'
        match = re.match(pattern, line.strip())
        
        if match:
            timestamp_str, component, level, message = match.groups()
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            return {
                'timestamp': timestamp,
                'component': component.strip(),
                'level': level.strip(),
                'message': message.strip()
            }
        return None
    
    def load_logs(self):
        """
        Carrega todos os logs do diretório
        """
        print("📂 Carregando logs...")
        
        log_files = {
            'signal_integrator': 'mt5signalintegrator_20250819.log',
            'order_manager': 'mt5ordermanager_20250819.log',
            'risk_manager': 'mt5riskmanager_20250819.log',
            'data_provider': 'mt5dataprovider_20250819.log'
        }
        
        logs_data = {}
        
        for component, filename in log_files.items():
            filepath = os.path.join(self.log_directory, filename)
            logs_data[component] = []
            
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            parsed = self.parse_log_line(line)
                            if parsed:
                                logs_data[component].append(parsed)
                    print(f"✅ {component}: {len(logs_data[component])} entradas carregadas")
                except Exception as e:
                    print(f"❌ Erro ao carregar {component}: {e}")
            else:
                print(f"⚠️  Arquivo não encontrado: {filename}")
        
        return logs_data
    
    def extract_trade_info(self, message):
        """
        Extrai informações de trade de uma mensagem
        """
        # Padrões para diferentes tipos de mensagens
        patterns = {
            'order_executed': r'Ordem executada: (\w+) (BUY|SELL) ([\d.]+) lotes a ([\d.]+)',
            'trade_executed': r'✅ Trade executado: (\w+) (BUY|SELL) ([\d.]+) lotes',
            'trade_rejected': r'Trade rejeitado pelo gerenciador de risco: (.+)',
            'signals_generated': r'📊 (\d+) sinais gerados',
            'trades_executed': r'💼 (\d+) trades executados'
        }
        
        for pattern_name, pattern in patterns.items():
            match = re.search(pattern, message)
            if match:
                if pattern_name == 'order_executed':
                    return {
                        'type': 'order_executed',
                        'symbol': match.group(1),
                        'direction': match.group(2),
                        'volume': float(match.group(3)),
                        'price': float(match.group(4))
                    }
                elif pattern_name == 'trade_executed':
                    return {
                        'type': 'trade_executed',
                        'symbol': match.group(1),
                        'direction': match.group(2),
                        'volume': float(match.group(3))
                    }
                elif pattern_name == 'trade_rejected':
                    return {
                        'type': 'trade_rejected',
                        'reason': match.group(1)
                    }
                elif pattern_name == 'signals_generated':
                    return {
                        'type': 'signals_generated',
                        'count': int(match.group(1))
                    }
                elif pattern_name == 'trades_executed':
                    return {
                        'type': 'trades_executed',
                        'count': int(match.group(1))
                    }
        
        return None
    
    def analyze_trade_consistency(self, logs_data):
        """
        Analisa consistência entre trades reportados pelo SignalIntegrator e OrderManager
        """
        print("\n🔍 Analisando consistência de trades...")
        
        signal_trades = []
        order_trades = []
        
        # Extrair trades do SignalIntegrator
        for log_entry in logs_data['signal_integrator']:
            trade_info = self.extract_trade_info(log_entry['message'])
            if trade_info and trade_info['type'] == 'trade_executed':
                signal_trades.append({
                    'timestamp': log_entry['timestamp'],
                    'symbol': trade_info['symbol'],
                    'direction': trade_info['direction'],
                    'volume': trade_info['volume']
                })
        
        # Extrair trades do OrderManager
        for log_entry in logs_data['order_manager']:
            trade_info = self.extract_trade_info(log_entry['message'])
            if trade_info and trade_info['type'] == 'order_executed':
                order_trades.append({
                    'timestamp': log_entry['timestamp'],
                    'symbol': trade_info['symbol'],
                    'direction': trade_info['direction'],
                    'volume': trade_info['volume'],
                    'price': trade_info['price']
                })
        
        print(f"📊 SignalIntegrator reportou: {len(signal_trades)} trades")
        print(f"📊 OrderManager executou: {len(order_trades)} ordens")
        
        # Verificar consistência
        inconsistencies = []
        
        for signal_trade in signal_trades:
            # Procurar ordem correspondente (dentro de 5 segundos)
            matching_order = None
            for order_trade in order_trades:
                time_diff = abs((signal_trade['timestamp'] - order_trade['timestamp']).total_seconds())
                if (time_diff <= 5 and 
                    signal_trade['symbol'] == order_trade['symbol'] and
                    signal_trade['direction'] == order_trade['direction'] and
                    signal_trade['volume'] == order_trade['volume']):
                    matching_order = order_trade
                    break
            
            if not matching_order:
                inconsistencies.append({
                    'type': 'missing_order',
                    'signal_trade': signal_trade
                })
        
        # Verificar ordens sem trade correspondente
        for order_trade in order_trades:
            matching_signal = None
            for signal_trade in signal_trades:
                time_diff = abs((signal_trade['timestamp'] - order_trade['timestamp']).total_seconds())
                if (time_diff <= 5 and 
                    signal_trade['symbol'] == order_trade['symbol'] and
                    signal_trade['direction'] == order_trade['direction'] and
                    signal_trade['volume'] == order_trade['volume']):
                    matching_signal = signal_trade
                    break
            
            if not matching_signal:
                inconsistencies.append({
                    'type': 'orphan_order',
                    'order_trade': order_trade
                })
        
        if inconsistencies:
            print(f"❌ {len(inconsistencies)} inconsistências detectadas:")
            for inc in inconsistencies:
                if inc['type'] == 'missing_order':
                    trade = inc['signal_trade']
                    print(f"  - Trade sem ordem: {trade['symbol']} {trade['direction']} {trade['volume']} em {trade['timestamp']}")
                elif inc['type'] == 'orphan_order':
                    order = inc['order_trade']
                    print(f"  - Ordem sem trade: {order['symbol']} {order['direction']} {order['volume']} em {order['timestamp']}")
        else:
            print("✅ Todos os trades estão consistentes entre SignalIntegrator e OrderManager")
        
        return len(inconsistencies) == 0
    
    def analyze_signal_execution_ratio(self, logs_data):
        """
        Analisa a proporção entre sinais gerados e trades executados
        """
        print("\n📈 Analisando proporção sinais/execuções...")
        
        signal_counts = []
        execution_counts = []
        rejection_counts = []
        
        for log_entry in logs_data['signal_integrator']:
            trade_info = self.extract_trade_info(log_entry['message'])
            if trade_info:
                if trade_info['type'] == 'signals_generated':
                    signal_counts.append({
                        'timestamp': log_entry['timestamp'],
                        'count': trade_info['count']
                    })
                elif trade_info['type'] == 'trades_executed':
                    execution_counts.append({
                        'timestamp': log_entry['timestamp'],
                        'count': trade_info['count']
                    })
                elif trade_info['type'] == 'trade_rejected':
                    rejection_counts.append({
                        'timestamp': log_entry['timestamp'],
                        'reason': trade_info['reason']
                    })
        
        total_signals = sum(s['count'] for s in signal_counts)
        total_executions = sum(e['count'] for e in execution_counts)
        total_rejections = len(rejection_counts)
        
        print(f"📊 Total de sinais gerados: {total_signals}")
        print(f"📊 Total de trades executados: {total_executions}")
        print(f"📊 Total de trades rejeitados: {total_rejections}")
        
        if total_signals > 0:
            execution_rate = (total_executions / total_signals) * 100
            rejection_rate = (total_rejections / total_signals) * 100
            print(f"📈 Taxa de execução: {execution_rate:.1f}%")
            print(f"📈 Taxa de rejeição: {rejection_rate:.1f}%")
            
            # Verificar se as contas batem
            expected_total = total_executions + total_rejections
            if abs(expected_total - total_signals) <= 1:  # Tolerância de 1
                print("✅ Contabilidade de sinais consistente")
                return True
            else:
                print(f"❌ Inconsistência: {total_signals} sinais vs {expected_total} processados")
                return False
        else:
            print("⚠️  Nenhum sinal encontrado")
            return False
    
    def analyze_timestamp_consistency(self, logs_data):
        """
        Analisa consistência de timestamps entre componentes
        """
        print("\n⏰ Analisando consistência de timestamps...")
        
        all_timestamps = []
        
        for component, logs in logs_data.items():
            for log_entry in logs:
                all_timestamps.append({
                    'component': component,
                    'timestamp': log_entry['timestamp'],
                    'message': log_entry['message']
                })
        
        # Ordenar por timestamp
        all_timestamps.sort(key=lambda x: x['timestamp'])
        
        # Verificar gaps grandes entre timestamps
        gaps = []
        for i in range(1, len(all_timestamps)):
            time_diff = (all_timestamps[i]['timestamp'] - all_timestamps[i-1]['timestamp']).total_seconds()
            if time_diff > 600:  # Gap maior que 10 minutos
                gaps.append({
                    'start': all_timestamps[i-1]['timestamp'],
                    'end': all_timestamps[i]['timestamp'],
                    'duration': time_diff
                })
        
        if gaps:
            print(f"⚠️  {len(gaps)} gaps grandes detectados:")
            for gap in gaps:
                print(f"  - Gap de {gap['duration']:.0f}s entre {gap['start']} e {gap['end']}")
        else:
            print("✅ Timestamps consistentes (sem gaps grandes)")
        
        return len(gaps) == 0
    
    def generate_summary_report(self, logs_data):
        """
        Gera relatório resumo da análise
        """
        print("\n" + "=" * 60)
        print("📋 RELATÓRIO DE CONSISTÊNCIA DOS LOGS")
        print("=" * 60)
        
        # Contagem de entradas por componente
        for component, logs in logs_data.items():
            if logs:
                start_time = min(log['timestamp'] for log in logs)
                end_time = max(log['timestamp'] for log in logs)
                duration = (end_time - start_time).total_seconds() / 60
                print(f"{component.replace('_', ' ').title()}: {len(logs)} entradas ({duration:.1f} min)")
            else:
                print(f"{component.replace('_', ' ').title()}: Sem dados")
        
        # Executar análises
        trade_consistency = self.analyze_trade_consistency(logs_data)
        signal_ratio_consistency = self.analyze_signal_execution_ratio(logs_data)
        timestamp_consistency = self.analyze_timestamp_consistency(logs_data)
        
        # Resultado final
        print("\n" + "=" * 60)
        print("🎯 RESULTADO FINAL")
        print("=" * 60)
        
        results = {
            'Consistência de Trades': trade_consistency,
            'Proporção Sinais/Execuções': signal_ratio_consistency,
            'Consistência de Timestamps': timestamp_consistency
        }
        
        passed = sum(results.values())
        total = len(results)
        
        for test, result in results.items():
            status = "✅ PASSOU" if result else "❌ FALHOU"
            print(f"{test}: {status}")
        
        print(f"\n🎯 Resultado: {passed}/{total} testes aprovados")
        
        if passed == total:
            print("🎉 TODOS OS LOGS ESTÃO CONSISTENTES!")
        elif passed >= total - 1:
            print("✅ LOGS MAJORITARIAMENTE CONSISTENTES")
        else:
            print("⚠️  PROBLEMAS DE CONSISTÊNCIA DETECTADOS!")
        
        return passed >= total - 1
    
    def run_analysis(self):
        """
        Executa análise completa de consistência
        """
        print("🔍 Iniciando Análise de Consistência dos Logs")
        print("=" * 60)
        
        # Carregar logs
        logs_data = self.load_logs()
        
        if not any(logs_data.values()):
            print("❌ Nenhum log encontrado para análise")
            return False
        
        # Gerar relatório
        return self.generate_summary_report(logs_data)

def main():
    """
    Função principal
    """
    try:
        log_directory = "logs"
        analyzer = LogConsistencyAnalyzer(log_directory)
        success = analyzer.run_analysis()
        
        if success:
            print("\n✅ Análise de consistência concluída com sucesso!")
            return 0
        else:
            print("\n❌ Problemas de consistência detectados!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Erro durante a análise: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())