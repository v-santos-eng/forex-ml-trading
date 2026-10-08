#!/usr/bin/env python3
"""
Exemplo de Integração do Sistema de Logging Detalhado com Pipeline ML

Este arquivo demonstra como usar o sistema de logging detalhado integrado
com o pipeline ML de produção para registrar análises, predições e decisões.

Autor: Sistema de Trading ML
Data: 2024
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np

# Importa componentes do pipeline ML
from ml_production_pipeline import MLProductionPipeline, PipelineConfig
from comprehensive_analysis_logger import AnalysisType, DataSource
from detailed_decision_logger import DecisionType, LogLevel

class MLLoggingIntegrationDemo:
    """
    Demonstra a integração do sistema de logging com o pipeline ML
    """
    
    def __init__(self):
        self.pipeline = None
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        self.demo_running = False
    
    def initialize_pipeline(self) -> bool:
        """
        Inicializa o pipeline ML com configurações de demonstração
        """
        try:
            # Configuração para demonstração
            config = PipelineConfig(
                enable_dashboard=True,
                enable_scheduler=True,
                enable_performance_monitor=True,
                enable_quality_monitor=True,
                retrain_interval_hours=24,
                monitoring_interval_seconds=60,
                currency_pairs=self.symbols,
                models_to_monitor=['xgboost', 'random_forest', 'lstm']
            )
            
            # Cria pipeline
            self.pipeline = MLProductionPipeline(config)
            
            # Inicia pipeline
            if self.pipeline.start_pipeline():
                print("✅ Pipeline ML inicializado com sistema de logging integrado")
                return True
            else:
                print("❌ Falha ao inicializar pipeline ML")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao inicializar pipeline: {e}")
            return False
    
    def simulate_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Simula análise técnica para demonstração
        """
        return {
            'symbol': symbol,
            'indicators': {
                'rsi': random.uniform(20, 80),
                'macd': random.uniform(-0.01, 0.01),
                'bollinger_position': random.uniform(0, 1),
                'moving_average_trend': random.choice(['bullish', 'bearish', 'neutral'])
            },
            'support_resistance': {
                'support': random.uniform(1.0500, 1.0800),
                'resistance': random.uniform(1.0900, 1.1200)
            },
            'confidence': random.uniform(0.6, 0.95),
            'processing_time_ms': random.uniform(50, 200),
            'timestamp': datetime.now().isoformat()
        }
    
    def simulate_ml_prediction(self, symbol: str) -> Dict[str, Any]:
        """
        Simula predição ML para demonstração
        """
        direction = random.choice(['buy', 'sell', 'hold'])
        
        return {
            'symbol': symbol,
            'prediction': {
                'direction': direction,
                'probability': random.uniform(0.55, 0.85),
                'price_target': random.uniform(1.0500, 1.1200),
                'time_horizon': random.choice(['1h', '4h', '1d'])
            },
            'model_ensemble': {
                'xgboost': random.uniform(0.5, 0.9),
                'random_forest': random.uniform(0.5, 0.9),
                'lstm': random.uniform(0.5, 0.9)
            },
            'market_conditions': {
                'volatility': random.choice(['low', 'medium', 'high']),
                'trend': random.choice(['bullish', 'bearish', 'neutral']),
                'volume': random.choice(['low', 'normal', 'high']),
                'spread': random.uniform(0.0001, 0.0005),
                'liquidity': random.choice(['low', 'normal', 'high'])
            },
            'confidence': random.uniform(0.6, 0.95),
            'processing_time_ms': random.uniform(100, 500),
            'timestamp': datetime.now().isoformat()
        }
    
    def simulate_risk_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Simula análise de risco para demonstração
        """
        return {
            'symbol': symbol,
            'risk_metrics': {
                'var_1d': random.uniform(0.01, 0.05),
                'expected_shortfall': random.uniform(0.02, 0.08),
                'sharpe_ratio': random.uniform(0.5, 2.0),
                'max_drawdown': random.uniform(0.05, 0.20)
            },
            'correlation_analysis': {
                'correlation_with_major_pairs': random.uniform(-0.8, 0.8),
                'correlation_with_indices': random.uniform(-0.6, 0.6)
            },
            'market_stress_scenarios': {
                'scenario_1_impact': random.uniform(-0.10, 0.10),
                'scenario_2_impact': random.uniform(-0.15, 0.15)
            },
            'confidence': random.uniform(0.7, 0.95),
            'processing_time_ms': random.uniform(200, 800),
            'timestamp': datetime.now().isoformat()
        }
    
    async def run_analysis_cycle(self, symbol: str) -> None:
        """
        Executa um ciclo completo de análise para um símbolo
        """
        try:
            print(f"\n🔍 Iniciando análise para {symbol}...")
            
            # 1. Análise Técnica
            technical_data = self.simulate_technical_analysis(symbol)
            self.pipeline.log_ml_analysis('technical', symbol, technical_data)
            print(f"  ✅ Análise técnica registrada para {symbol}")
            
            await asyncio.sleep(0.5)  # Simula tempo de processamento
            
            # 2. Predição ML
            prediction_data = self.simulate_ml_prediction(symbol)
            self.pipeline.log_ml_prediction(symbol, prediction_data)
            print(f"  ✅ Predição ML registrada para {symbol}")
            
            await asyncio.sleep(0.5)
            
            # 3. Análise de Risco
            risk_data = self.simulate_risk_analysis(symbol)
            self.pipeline.log_ml_analysis('risk', symbol, risk_data)
            print(f"  ✅ Análise de risco registrada para {symbol}")
            
            await asyncio.sleep(0.5)
            
            # 4. Análise Abrangente (combinando todos os dados)
            comprehensive_data = {
                'technical_analysis': technical_data,
                'ml_prediction': prediction_data,
                'risk_analysis': risk_data,
                'final_recommendation': {
                    'action': random.choice(['buy', 'sell', 'hold']),
                    'confidence': random.uniform(0.6, 0.9),
                    'reasoning': 'Baseado na combinação de análise técnica, predição ML e avaliação de risco'
                },
                'processing_time_ms': sum([
                    technical_data['processing_time_ms'],
                    prediction_data['processing_time_ms'],
                    risk_data['processing_time_ms']
                ])
            }
            
            self.pipeline.log_ml_analysis('comprehensive', symbol, comprehensive_data)
            print(f"  ✅ Análise abrangente registrada para {symbol}")
            
        except Exception as e:
            print(f"  ❌ Erro na análise de {symbol}: {e}")
    
    async def run_demo_session(self, duration_minutes: int = 5) -> None:
        """
        Executa uma sessão de demonstração do sistema de logging
        """
        print(f"\n🚀 Iniciando sessão de demonstração ({duration_minutes} minutos)...")
        
        self.demo_running = True
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        cycle_count = 0
        
        try:
            while self.demo_running and time.time() < end_time:
                cycle_count += 1
                print(f"\n📊 Ciclo de Análise #{cycle_count}")
                
                # Executa análise para todos os símbolos em paralelo
                tasks = [self.run_analysis_cycle(symbol) for symbol in self.symbols]
                await asyncio.gather(*tasks)
                
                # Aguarda antes do próximo ciclo
                await asyncio.sleep(30)  # 30 segundos entre ciclos
                
        except KeyboardInterrupt:
            print("\n⏹️  Demonstração interrompida pelo usuário")
        
        self.demo_running = False
        
        print(f"\n✅ Sessão de demonstração concluída")
        print(f"   - Duração: {(time.time() - start_time)/60:.1f} minutos")
        print(f"   - Ciclos executados: {cycle_count}")
        print(f"   - Análises registradas: {cycle_count * len(self.symbols) * 4}")
    
    def generate_session_report(self) -> None:
        """
        Gera relatório da sessão de logging
        """
        try:
            print("\n📋 Gerando relatório da sessão...")
            
            # Obtém resumo dos sistemas de logging
            logging_summary = self.pipeline.get_logging_summary()
            
            print("\n📊 Resumo dos Sistemas de Logging:")
            
            if logging_summary.get('comprehensive_logger'):
                comp_summary = logging_summary['comprehensive_logger']
                print(f"\n  🔍 Logger Abrangente:")
                print(f"    - Total de análises: {comp_summary.get('total_analyses', 0)}")
                print(f"    - Análises técnicas: {comp_summary.get('technical_analyses', 0)}")
                print(f"    - Análises de risco: {comp_summary.get('risk_analyses', 0)}")
                print(f"    - Análises abrangentes: {comp_summary.get('comprehensive_analyses', 0)}")
            
            if logging_summary.get('detailed_logger'):
                detail_summary = logging_summary['detailed_logger']
                print(f"\n  📝 Logger Detalhado:")
                print(f"    - Total de decisões: {detail_summary.get('total_decisions', 0)}")
                print(f"    - Análises de mercado: {detail_summary.get('market_analyses', 0)}")
            
            if logging_summary.get('trading_integrator'):
                trading_summary = logging_summary['trading_integrator']
                print(f"\n  🔄 Integrador de Trading:")
                print(f"    - Sessão ativa: {trading_summary.get('session_active', False)}")
                print(f"    - Tempo de sessão: {trading_summary.get('session_duration', 0)} segundos")
            
            # Exporta relatórios detalhados
            print("\n💾 Exportando relatórios detalhados...")
            reports = self.pipeline.export_logging_reports()
            
            if reports:
                print("\n📁 Relatórios exportados:")
                for report_type, report_path in reports.items():
                    if report_type != 'error':
                        print(f"    - {report_type}: {report_path}")
            
        except Exception as e:
            print(f"❌ Erro ao gerar relatório: {e}")
    
    def cleanup(self) -> None:
        """
        Limpa recursos e para o pipeline
        """
        try:
            if self.pipeline:
                print("\n🛑 Parando pipeline ML...")
                self.pipeline.stop_pipeline()
                print("✅ Pipeline parado com sucesso")
        except Exception as e:
            print(f"❌ Erro ao parar pipeline: {e}")

async def main():
    """
    Função principal da demonstração
    """
    demo = MLLoggingIntegrationDemo()
    
    try:
        # Inicializa pipeline
        if not demo.initialize_pipeline():
            return
        
        # Aguarda inicialização completa
        await asyncio.sleep(2)
        
        # Executa sessão de demonstração
        await demo.run_demo_session(duration_minutes=3)
        
        # Gera relatório
        demo.generate_session_report()
        
    except KeyboardInterrupt:
        print("\n⏹️  Demonstração interrompida")
    except Exception as e:
        print(f"❌ Erro na demonstração: {e}")
    finally:
        # Cleanup
        demo.cleanup()

if __name__ == "__main__":
    print("🎯 Demonstração de Integração do Sistema de Logging ML")
    print("=" * 60)
    print("\nEsta demonstração mostra como o sistema de logging detalhado")
    print("se integra com o pipeline ML para registrar:")
    print("  • Análises técnicas")
    print("  • Predições ML")
    print("  • Análises de risco")
    print("  • Decisões abrangentes")
    print("\nPressione Ctrl+C para parar a qualquer momento.")
    print("=" * 60)
    
    # Executa demonstração
    asyncio.run(main())