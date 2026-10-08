#!/usr/bin/env python3
"""
Demonstração Completa do Sistema de Logging em Produção

Este script demonstra o uso completo do sistema de logging detalhado
em um cenário real de produção, incluindo:
- Inicialização de todos os componentes
- Simulação de análises ML e decisões de trading
- Monitoramento contínuo
- Geração de relatórios
- Manutenção automática

Autor: Sistema de Trading ML
Versão: 1.0.0
Data: 2024
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
from pathlib import Path

# Importar componentes do sistema de logging
try:
    from comprehensive_analysis_logger import (
        ComprehensiveAnalysisLogger, AnalysisType, DataSource, AnalysisData,
        initialize_comprehensive_logger, get_comprehensive_logger
    )
    from detailed_decision_logger import (
        DetailedDecisionLogger, DecisionType, MarketConditions, AccountState,
        initialize_detailed_logger, get_detailed_logger
    )
    from trading_decision_integrator import (
        TradingDecisionIntegrator, initialize_trading_integrator
    )
    from logging_utilities import LoggingUtilities
except ImportError as e:
    print(f"Erro ao importar componentes de logging: {e}")
    print("Certifique-se de que todos os arquivos de logging estão no diretório correto.")
    exit(1)


class ProductionLoggingDemo:
    """Demonstração do sistema de logging em produção."""
    
    def __init__(self):
        """Inicializar demonstração."""
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        self.running = False
        self.session_start_time = datetime.now()
        
        # Componentes de logging
        self.comprehensive_logger = None
        self.detailed_logger = None
        self.trading_integrator = None
        self.utilities = None
        
        # Estatísticas da sessão
        self.session_stats = {
            'analyses_performed': 0,
            'decisions_made': 0,
            'trades_executed': 0,
            'no_trade_decisions': 0,
            'errors_encountered': 0
        }
    
    def initialize_logging_system(self) -> bool:
        """Inicializar todos os componentes do sistema de logging."""
        try:
            print("[INIT] Inicializando sistema de logging detalhado...")
            
            # Criar diretórios necessários
            Path('./logs').mkdir(exist_ok=True)
            Path('./reports').mkdir(exist_ok=True)
            
            # Inicializar loggers
            initialize_comprehensive_logger()
            initialize_detailed_logger()
            initialize_trading_integrator()
            
            # Obter instâncias
            self.comprehensive_logger = get_comprehensive_logger()
            self.detailed_logger = get_detailed_logger()
            self.trading_integrator = TradingDecisionIntegrator()
            self.utilities = LoggingUtilities()
            
            print("[INIT] Sistema de logging inicializado com sucesso!")
            return True
            
        except Exception as e:
            print(f"[ERROR] Falha ao inicializar sistema de logging: {e}")
            return False
    
    def simulate_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """Simular análise técnica para um símbolo."""
        # Simular indicadores técnicos
        indicators = {
            'rsi': random.uniform(20, 80),
            'macd': random.uniform(-0.005, 0.005),
            'bollinger_position': random.uniform(0, 1),
            'sma_20': random.uniform(1.0800, 1.1200) if 'EUR' in symbol else random.uniform(0.8000, 1.2000),
            'sma_50': random.uniform(1.0750, 1.1150) if 'EUR' in symbol else random.uniform(0.7950, 1.1950),
            'support_level': random.uniform(1.0700, 1.0900) if 'EUR' in symbol else random.uniform(0.7800, 1.1800),
            'resistance_level': random.uniform(1.1000, 1.1300) if 'EUR' in symbol else random.uniform(1.0000, 1.3000),
            'trend_strength': random.uniform(0.3, 0.9),
            'volatility': random.uniform(0.1, 0.8)
        }
        
        # Determinar tendência baseada nos indicadores
        if indicators['rsi'] > 70:
            trend = 'overbought'
        elif indicators['rsi'] < 30:
            trend = 'oversold'
        elif indicators['macd'] > 0:
            trend = 'bullish'
        else:
            trend = 'bearish'
        
        indicators['trend'] = trend
        
        return indicators
    
    def simulate_fundamental_analysis(self, symbol: str) -> Dict[str, Any]:
        """Simular análise fundamental para um símbolo."""
        # Simular eventos econômicos
        economic_events = [
            {'event': 'GDP Release', 'impact': random.choice(['low', 'medium', 'high'])},
            {'event': 'Interest Rate Decision', 'impact': random.choice(['medium', 'high'])},
            {'event': 'Employment Data', 'impact': random.choice(['low', 'medium', 'high'])},
            {'event': 'Inflation Report', 'impact': random.choice(['medium', 'high'])}
        ]
        
        # Simular notícias de mercado
        news_sentiment = random.choice(['positive', 'negative', 'neutral'])
        news_impact = random.uniform(0.1, 0.9)
        
        return {
            'economic_events': random.sample(economic_events, random.randint(1, 3)),
            'news_sentiment': news_sentiment,
            'news_impact_score': news_impact,
            'market_sentiment': random.choice(['risk_on', 'risk_off', 'neutral']),
            'correlation_strength': random.uniform(0.2, 0.8),
            'fundamental_score': random.uniform(-1, 1)
        }
    
    def simulate_risk_analysis(self, symbol: str) -> Dict[str, Any]:
        """Simular análise de risco para um símbolo."""
        return {
            'var_1d': random.uniform(0.005, 0.025),  # Value at Risk 1 dia
            'var_1w': random.uniform(0.015, 0.075),  # Value at Risk 1 semana
            'max_drawdown': random.uniform(0.02, 0.15),
            'sharpe_ratio': random.uniform(-0.5, 2.5),
            'correlation_matrix': {
                other_symbol: random.uniform(-0.8, 0.8) 
                for other_symbol in self.symbols if other_symbol != symbol
            },
            'volatility_regime': random.choice(['low', 'normal', 'high']),
            'liquidity_score': random.uniform(0.3, 1.0),
            'stress_test_results': {
                'scenario_1': random.uniform(-0.1, 0.1),
                'scenario_2': random.uniform(-0.15, 0.15),
                'scenario_3': random.uniform(-0.2, 0.2)
            }
        }
    
    def perform_comprehensive_analysis(self, symbol: str) -> None:
        """Realizar análise abrangente e registrar no sistema."""
        try:
            # Análise técnica
            technical_data = self.simulate_technical_analysis(symbol)
            technical_analysis = AnalysisData(
                analysis_type=AnalysisType.TECHNICAL,
                symbol=symbol,
                data_source=DataSource.ML_MODEL,
                raw_data=technical_data,
                confidence_score=random.uniform(0.6, 0.95),
                processing_time_ms=random.uniform(50, 200)
            )
            
            self.comprehensive_logger.log_technical_analysis(technical_analysis)
            
            # Análise fundamental
            fundamental_data = self.simulate_fundamental_analysis(symbol)
            fundamental_analysis = AnalysisData(
                analysis_type=AnalysisType.FUNDAMENTAL,
                symbol=symbol,
                data_source=DataSource.EXTERNAL_API,
                raw_data=fundamental_data,
                confidence_score=random.uniform(0.5, 0.85),
                processing_time_ms=random.uniform(100, 300)
            )
            
            self.comprehensive_logger.log_fundamental_analysis(fundamental_analysis)
            
            # Análise de risco
            risk_data = self.simulate_risk_analysis(symbol)
            risk_analysis = AnalysisData(
                analysis_type=AnalysisType.RISK,
                symbol=symbol,
                data_source=DataSource.ML_MODEL,
                raw_data=risk_data,
                confidence_score=random.uniform(0.7, 0.9),
                processing_time_ms=random.uniform(75, 250)
            )
            
            self.comprehensive_logger.log_risk_analysis(risk_analysis)
            
            # Análise abrangente (combinando todas)
            comprehensive_data = {
                'technical_score': technical_analysis.confidence_score,
                'fundamental_score': fundamental_analysis.confidence_score,
                'risk_score': risk_analysis.confidence_score,
                'combined_signal': self._calculate_combined_signal(
                    technical_data, fundamental_data, risk_data
                ),
                'recommendation': self._generate_recommendation(
                    technical_data, fundamental_data, risk_data
                )
            }
            
            comprehensive_analysis = AnalysisData(
                analysis_type=AnalysisType.COMPREHENSIVE,
                symbol=symbol,
                data_source=DataSource.ML_MODEL,
                raw_data=comprehensive_data,
                confidence_score=(
                    technical_analysis.confidence_score + 
                    fundamental_analysis.confidence_score + 
                    risk_analysis.confidence_score
                ) / 3,
                processing_time_ms=random.uniform(200, 500)
            )
            
            self.comprehensive_logger.log_comprehensive_analysis(comprehensive_analysis)
            
            self.session_stats['analyses_performed'] += 4
            
            print(f"[ANALYSIS] Análise abrangente concluída para {symbol} - Recomendação: {comprehensive_data['recommendation']}")
            
        except Exception as e:
            print(f"[ERROR] Erro na análise abrangente para {symbol}: {e}")
            self.session_stats['errors_encountered'] += 1
    
    def _calculate_combined_signal(self, technical: Dict, fundamental: Dict, risk: Dict) -> str:
        """Calcular sinal combinado baseado nas análises."""
        # Lógica simplificada para combinar sinais
        technical_score = 1 if technical['trend'] in ['bullish'] else -1 if technical['trend'] in ['bearish'] else 0
        fundamental_score = 1 if fundamental['fundamental_score'] > 0.2 else -1 if fundamental['fundamental_score'] < -0.2 else 0
        risk_score = 1 if risk['sharpe_ratio'] > 1.0 else -1 if risk['sharpe_ratio'] < 0 else 0
        
        combined = technical_score + fundamental_score + risk_score
        
        if combined >= 2:
            return 'strong_buy'
        elif combined == 1:
            return 'buy'
        elif combined == -1:
            return 'sell'
        elif combined <= -2:
            return 'strong_sell'
        else:
            return 'hold'
    
    def _generate_recommendation(self, technical: Dict, fundamental: Dict, risk: Dict) -> str:
        """Gerar recomendação baseada nas análises."""
        signal = self._calculate_combined_signal(technical, fundamental, risk)
        
        recommendations = {
            'strong_buy': 'Compra forte recomendada - Todos os indicadores apontam para alta',
            'buy': 'Compra recomendada - Maioria dos indicadores positivos',
            'hold': 'Manter posição - Sinais mistos ou neutros',
            'sell': 'Venda recomendada - Maioria dos indicadores negativos',
            'strong_sell': 'Venda forte recomendada - Todos os indicadores apontam para baixa'
        }
        
        return recommendations.get(signal, 'Análise inconclusiva')
    
    def simulate_trading_decision(self, symbol: str, analysis_result: str) -> None:
        """Simular decisão de trading baseada na análise."""
        try:
            # Simular condições de mercado
            market_conditions = MarketConditions(
                volatility=random.choice(['low', 'medium', 'high']),
                trend=random.choice(['bullish', 'bearish', 'neutral']),
                volume=random.choice(['low', 'normal', 'high']),
                spread=random.uniform(0.0001, 0.0005),
                liquidity=random.choice(['low', 'normal', 'high'])
            )
            
            # Simular estado da conta
            account_state = AccountState(
                balance=random.uniform(9500, 10500),
                equity=random.uniform(9800, 10800),
                margin=random.uniform(100, 800),
                free_margin=random.uniform(9000, 10000),
                margin_level=random.uniform(1200, 3000),
                open_positions=random.randint(0, 5)
            )
            
            # Decidir se vai operar ou não
            should_trade = random.choice([True, False, False])  # 33% chance de operar
            
            if should_trade and 'buy' in analysis_result.lower():
                # Simular operação de compra
                trade_details = {
                    'operation_type': 'buy',
                    'volume': round(random.uniform(0.01, 0.1), 2),
                    'entry_price': random.uniform(1.0850, 1.0950) if 'EUR' in symbol else random.uniform(0.8000, 1.2000),
                    'stop_loss': None,
                    'take_profit': None,
                    'order_type': 'market'
                }
                
                # Calcular SL e TP
                entry_price = trade_details['entry_price']
                trade_details['stop_loss'] = round(entry_price * 0.995, 5)  # 0.5% SL
                trade_details['take_profit'] = round(entry_price * 1.01, 5)  # 1% TP
                
                reasoning = f"Operação de compra executada baseada em análise {analysis_result}. Condições de mercado favoráveis."
                
                self.detailed_logger.log_trade_operation(
                    symbol=symbol,
                    operation_type='buy',
                    trade_details=trade_details,
                    market_conditions=market_conditions,
                    reasoning=reasoning
                )
                
                self.session_stats['trades_executed'] += 1
                print(f"[TRADE] Compra executada para {symbol} - Volume: {trade_details['volume']} - Preço: {trade_details['entry_price']}")
                
            elif should_trade and 'sell' in analysis_result.lower():
                # Simular operação de venda
                trade_details = {
                    'operation_type': 'sell',
                    'volume': round(random.uniform(0.01, 0.1), 2),
                    'entry_price': random.uniform(1.0850, 1.0950) if 'EUR' in symbol else random.uniform(0.8000, 1.2000),
                    'stop_loss': None,
                    'take_profit': None,
                    'order_type': 'market'
                }
                
                # Calcular SL e TP
                entry_price = trade_details['entry_price']
                trade_details['stop_loss'] = round(entry_price * 1.005, 5)  # 0.5% SL
                trade_details['take_profit'] = round(entry_price * 0.99, 5)  # 1% TP
                
                reasoning = f"Operação de venda executada baseada em análise {analysis_result}. Condições de mercado favoráveis."
                
                self.detailed_logger.log_trade_operation(
                    symbol=symbol,
                    operation_type='sell',
                    trade_details=trade_details,
                    market_conditions=market_conditions,
                    reasoning=reasoning
                )
                
                self.session_stats['trades_executed'] += 1
                print(f"[TRADE] Venda executada para {symbol} - Volume: {trade_details['volume']} - Preço: {trade_details['entry_price']}")
                
            else:
                # Decisão de não operar
                reasons = [
                    "Condições de mercado desfavoráveis",
                    "Spread muito alto",
                    "Baixa liquidez detectada",
                    "Sinais conflitantes nas análises",
                    "Risco/retorno inadequado",
                    "Limite de exposição atingido",
                    "Volatilidade excessiva"
                ]
                
                reasoning = random.choice(reasons)
                
                self.detailed_logger.log_no_trade_decision(
                    symbol=symbol,
                    market_conditions=market_conditions,
                    account_state=account_state,
                    reasoning=reasoning
                )
                
                self.session_stats['no_trade_decisions'] += 1
                print(f"[NO_TRADE] Decisão de não operar {symbol} - Motivo: {reasoning}")
            
            self.session_stats['decisions_made'] += 1
            
        except Exception as e:
            print(f"[ERROR] Erro na decisão de trading para {symbol}: {e}")
            self.session_stats['errors_encountered'] += 1
    
    def simulate_balance_update(self) -> None:
        """Simular atualização de saldo da conta."""
        try:
            # Simular mudança no saldo
            old_balance = random.uniform(9800, 10200)
            change = random.uniform(-50, 100)  # Pode ser positivo ou negativo
            new_balance = old_balance + change
            
            # Determinar motivo da mudança
            if change > 0:
                reasons = [
                    "Lucro realizado em operação fechada",
                    "Dividendos recebidos",
                    "Ajuste positivo de swap",
                    "Depósito realizado"
                ]
            else:
                reasons = [
                    "Prejuízo realizado em operação fechada",
                    "Taxa de corretagem cobrada",
                    "Ajuste negativo de swap",
                    "Saque realizado"
                ]
            
            reason = random.choice(reasons)
            
            # Simular estado da conta
            account_state = AccountState(
                balance=new_balance,
                equity=new_balance + random.uniform(-20, 50),
                margin=random.uniform(50, 300),
                free_margin=new_balance - random.uniform(50, 300),
                margin_level=random.uniform(1500, 2500),
                open_positions=random.randint(0, 3)
            )
            
            self.detailed_logger.log_balance_update(
                old_balance=old_balance,
                new_balance=new_balance,
                reason=reason,
                account_state=account_state
            )
            
            print(f"[BALANCE] Saldo atualizado: {old_balance:.2f} → {new_balance:.2f} ({change:+.2f}) - {reason}")
            
        except Exception as e:
            print(f"[ERROR] Erro na atualização de saldo: {e}")
            self.session_stats['errors_encountered'] += 1
    
    async def run_trading_session(self, duration_minutes: int = 30) -> None:
        """Executar sessão de trading simulada."""
        print(f"[SESSION] Iniciando sessão de trading de {duration_minutes} minutos...")
        
        self.running = True
        session_end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        cycle_count = 0
        
        try:
            while self.running and datetime.now() < session_end_time:
                cycle_count += 1
                print(f"\n[CYCLE {cycle_count}] Iniciando ciclo de análise...")
                
                # Selecionar símbolo aleatório
                symbol = random.choice(self.symbols)
                
                # Realizar análise abrangente
                self.perform_comprehensive_analysis(symbol)
                
                # Aguardar um pouco para simular processamento
                await asyncio.sleep(random.uniform(1, 3))
                
                # Obter resultado da análise (simulado)
                analysis_results = ['strong_buy', 'buy', 'hold', 'sell', 'strong_sell']
                result = random.choice(analysis_results)
                
                # Tomar decisão de trading
                self.simulate_trading_decision(symbol, result)
                
                # Ocasionalmente simular atualização de saldo
                if random.random() < 0.3:  # 30% chance
                    self.simulate_balance_update()
                
                # Aguardar antes do próximo ciclo
                await asyncio.sleep(random.uniform(5, 15))
                
                # A cada 10 ciclos, mostrar estatísticas
                if cycle_count % 10 == 0:
                    self.print_session_statistics()
        
        except KeyboardInterrupt:
            print("\n[SESSION] Sessão interrompida pelo usuário")
        except Exception as e:
            print(f"\n[ERROR] Erro durante a sessão: {e}")
        finally:
            self.running = False
            print(f"\n[SESSION] Sessão finalizada após {cycle_count} ciclos")
    
    def print_session_statistics(self) -> None:
        """Imprimir estatísticas da sessão atual."""
        duration = datetime.now() - self.session_start_time
        
        print(f"\n{'='*50}")
        print(f"ESTATÍSTICAS DA SESSÃO")
        print(f"{'='*50}")
        print(f"Duração: {duration}")
        print(f"Análises realizadas: {self.session_stats['analyses_performed']}")
        print(f"Decisões tomadas: {self.session_stats['decisions_made']}")
        print(f"Operações executadas: {self.session_stats['trades_executed']}")
        print(f"Decisões de não operar: {self.session_stats['no_trade_decisions']}")
        print(f"Erros encontrados: {self.session_stats['errors_encountered']}")
        
        if self.session_stats['decisions_made'] > 0:
            trade_rate = (self.session_stats['trades_executed'] / self.session_stats['decisions_made']) * 100
            print(f"Taxa de execução: {trade_rate:.1f}%")
        
        print(f"{'='*50}\n")
    
    def generate_session_reports(self) -> Dict[str, str]:
        """Gerar relatórios da sessão."""
        print("[REPORTS] Gerando relatórios da sessão...")
        
        reports = {}
        
        try:
            # Relatório do logger abrangente
            if self.comprehensive_logger:
                comp_report_path = self.comprehensive_logger.export_session_report('./reports')
                reports['comprehensive'] = comp_report_path
                print(f"[REPORTS] Relatório abrangente: {comp_report_path}")
            
            # Relatório do logger detalhado
            if self.detailed_logger:
                detail_report_path = self.detailed_logger.export_session_report('./reports')
                reports['detailed'] = detail_report_path
                print(f"[REPORTS] Relatório detalhado: {detail_report_path}")
            
            # Relatório de manutenção
            if self.utilities:
                maintenance_report = self.utilities.generate_maintenance_report()
                maintenance_path = f"./reports/maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(maintenance_path, 'w', encoding='utf-8') as f:
                    json.dump(maintenance_report, f, indent=2, ensure_ascii=False, default=str)
                
                reports['maintenance'] = maintenance_path
                print(f"[REPORTS] Relatório de manutenção: {maintenance_path}")
        
        except Exception as e:
            print(f"[ERROR] Erro ao gerar relatórios: {e}")
        
        return reports
    
    def perform_maintenance(self) -> None:
        """Realizar manutenção do sistema de logging."""
        print("[MAINTENANCE] Iniciando manutenção do sistema...")
        
        try:
            if self.utilities:
                # Verificar saúde do sistema
                health = self.utilities.get_system_health()
                print(f"[MAINTENANCE] Status do sistema: {health['overall_status']}")
                
                if health['alerts']:
                    print("[MAINTENANCE] Alertas detectados:")
                    for alert in health['alerts']:
                        print(f"  - [{alert['severity']}] {alert['message']}")
                
                # Verificar integridade
                integrity = self.utilities.verify_log_integrity()
                print(f"[MAINTENANCE] Integridade dos logs: {integrity['summary']['integrity_percentage']:.1f}%")
                
                # Limpeza automática se necessário
                if health['disk_usage']['free_gb'] < 5:
                    print("[MAINTENANCE] Espaço em disco baixo, executando limpeza...")
                    cleanup_result = self.utilities.cleanup_old_logs(30)
                    print(f"[MAINTENANCE] Limpeza concluída: {cleanup_result['total_space_freed_mb']:.2f}MB liberados")
        
        except Exception as e:
            print(f"[ERROR] Erro durante manutenção: {e}")
    
    def stop_session(self) -> None:
        """Parar a sessão de trading."""
        print("[SESSION] Parando sessão...")
        self.running = False


async def main():
    """Função principal da demonstração."""
    print("="*60)
    print("DEMONSTRAÇÃO DO SISTEMA DE LOGGING DETALHADO")
    print("Sistema de Trading ML - Versão 1.0.0")
    print("="*60)
    
    # Criar instância da demonstração
    demo = ProductionLoggingDemo()
    
    # Inicializar sistema de logging
    if not demo.initialize_logging_system():
        print("[ERROR] Falha na inicialização. Encerrando demonstração.")
        return
    
    try:
        # Executar sessão de trading
        await demo.run_trading_session(duration_minutes=15)  # 15 minutos de demonstração
        
        # Imprimir estatísticas finais
        demo.print_session_statistics()
        
        # Gerar relatórios
        reports = demo.generate_session_reports()
        
        # Realizar manutenção
        demo.perform_maintenance()
        
        print("\n[DEMO] Demonstração concluída com sucesso!")
        print("\nArquivos gerados:")
        print("- Logs detalhados em: ./logs/")
        print("- Relatórios em: ./reports/")
        
        if reports:
            print("\nRelatórios específicos:")
            for report_type, path in reports.items():
                print(f"  - {report_type.capitalize()}: {path}")
    
    except Exception as e:
        print(f"\n[ERROR] Erro durante a demonstração: {e}")
    
    finally:
        # Cleanup
        demo.stop_session()
        print("\n[DEMO] Recursos liberados.")


if __name__ == '__main__':
    # Executar demonstração
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[DEMO] Demonstração interrompida pelo usuário.")
    except Exception as e:
        print(f"\n[ERROR] Erro fatal: {e}")