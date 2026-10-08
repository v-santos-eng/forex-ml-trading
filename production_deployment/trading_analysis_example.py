#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemplo Prático de Uso do Sistema de Logging Detalhado
Demonstra como registrar análises de mercado, decisões de trading e situações sem operação
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt5_dashboard_simple_secure import MT5Authenticator
from detailed_decision_logger import get_detailed_logger, initialize_detailed_logger
from trading_decision_integrator import initialize_trading_integrator
from datetime import datetime
import time
import random

class TradingAnalysisDemo:
    """Demonstração do sistema de análise e logging de trading"""
    
    def __init__(self):
        # Inicializar componentes
        self.mt5_auth = MT5Authenticator()
        self.session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Conectar (modo simulação se MT5 não disponível)
        print("Inicializando sistema de trading...")
        self.mt5_auth.connect()
        
        # Obter integrador de trading
        self.trading_integrator = self.mt5_auth.trading_integrator
        
        if not self.trading_integrator:
            print("ERRO: Integrador de trading não disponível")
            return
        
        print(f"Sistema inicializado - Sessão: {self.session_id}")
        print(f"Modo: {'Simulação' if self.mt5_auth.simulation_mode else 'Real'}")
    
    def simulate_market_analysis_session(self):
        """Simula uma sessão completa de análise de mercado"""
        print("\n=== INICIANDO ANÁLISE DE MERCADO ===")
        
        # Lista de símbolos para análise
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        
        # Simular análise técnica
        analysis_results = {
            'technical_indicators': {
                'RSI': {'EURUSD': 65.2, 'GBPUSD': 45.8, 'USDJPY': 72.1, 'AUDUSD': 38.9},
                'MACD': {'EURUSD': 'bullish', 'GBPUSD': 'bearish', 'USDJPY': 'bullish', 'AUDUSD': 'bearish'},
                'Moving_Averages': {'EURUSD': 'above_200MA', 'GBPUSD': 'below_200MA', 'USDJPY': 'above_200MA', 'AUDUSD': 'below_200MA'}
            },
            'fundamental_analysis': {
                'economic_calendar': ['USD NFP release', 'EUR CPI data', 'GBP BoE meeting'],
                'market_sentiment': 'risk-on',
                'volatility_index': 18.5
            },
            'risk_assessment': {
                'overall_risk': 'medium',
                'correlation_analysis': 'EUR and GBP showing high correlation',
                'liquidity_conditions': 'normal'
            }
        }
        
        observations = [
            "Mercado em tendência de alta moderada",
            "Volatilidade dentro dos parâmetros normais",
            "Correlação EUR/GBP elevada - risco de exposição dupla",
            "Calendário econômico com eventos de alto impacto",
            "Sentimento de mercado favorável ao risco",
            "Liquidez adequada para operações de médio volume"
        ]
        
        recommendations = [
            "Considerar posições compradas em EURUSD e USDJPY",
            "Evitar exposição simultânea em EUR e GBP",
            "Monitorar volatilidade antes dos eventos econômicos",
            "Aplicar gestão de risco rigorosa",
            "Considerar redução de posições antes do NFP"
        ]
        
        # Registrar análise de mercado
        decision_id = self.trading_integrator.log_market_analysis_session(
            symbols=symbols,
            analysis_results=analysis_results,
            observations=observations,
            recommendations=recommendations,
            confidence=0.75
        )
        
        print(f"Análise de mercado registrada - ID: {decision_id}")
        return analysis_results
    
    def simulate_trade_decision(self, symbol: str, should_trade: bool = True):
        """Simula uma decisão de trading"""
        print(f"\n=== ANALISANDO {symbol} ===")
        
        if should_trade:
            # Simular decisão de compra
            analysis_factors = [
                "RSI em zona de sobrevenda (32.5)",
                "MACD cruzou para cima da linha de sinal",
                "Preço rompeu resistência em 1.0850",
                "Volume acima da média",
                "Suporte forte em 1.0820",
                "Confluência com Fibonacci 61.8%"
            ]
            
            justification = (
                f"Análise técnica de {symbol} indica oportunidade de compra. "
                f"RSI saindo de zona de sobrevenda, MACD com divergência bullish, "
                f"e rompimento de resistência com volume confirmando movimento. "
                f"Stop loss em 1.0815 (risco de 35 pips) e take profit em 1.0920 "
                f"(potencial de 70 pips), resultando em R/R de 1:2."
            )
            
            decision_id = self.trading_integrator.log_manual_trade_decision(
                symbol=symbol,
                action="BUY",
                volume=0.1,
                analysis_factors=analysis_factors,
                justification=justification,
                confidence=0.8
            )
            
            print(f"Decisão de COMPRA registrada para {symbol} - ID: {decision_id}")
            
        else:
            # Simular decisão de não operar
            reasons = [
                "Volatilidade excessiva (ATR > 150 pips)",
                "Ausência de confluência técnica",
                "Proximidade de evento de alto impacto (NFP em 2 horas)",
                "Correlação negativa com posição existente",
                "Spread muito elevado (4.2 pips vs média de 1.8)",
                "Falta de volume confirmando movimento"
            ]
            
            analysis_performed = [
                "Análise de múltiplos timeframes (M15, H1, H4, D1)",
                "Verificação de suportes e resistências",
                "Análise de indicadores técnicos (RSI, MACD, Bollinger)",
                "Avaliação do calendário econômico",
                "Análise de correlação com outras posições",
                "Verificação das condições de mercado (spread, liquidez)",
                "Análise de gestão de risco (exposição total, drawdown)"
            ]
            
            decision_id = self.trading_integrator.log_no_trade_analysis(
                symbol=symbol,
                reasons=reasons,
                analysis_performed=analysis_performed,
                confidence=0.9
            )
            
            print(f"Decisão de NÃO OPERAR registrada para {symbol} - ID: {decision_id}")
    
    def simulate_trading_session(self):
        """Simula uma sessão completa de trading"""
        print("\n" + "="*60)
        print("SIMULAÇÃO DE SESSÃO DE TRADING COMPLETA")
        print("="*60)
        
        # 1. Análise geral de mercado
        market_analysis = self.simulate_market_analysis_session()
        
        # 2. Análise individual de símbolos
        symbols_to_analyze = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF']
        
        for i, symbol in enumerate(symbols_to_analyze):
            # Simular diferentes cenários
            should_trade = random.choice([True, False])  # 50% chance de operar
            
            self.simulate_trade_decision(symbol, should_trade)
            
            # Pequena pausa entre análises
            time.sleep(1)
        
        # 3. Resumo da sessão
        print("\n=== RESUMO DA SESSÃO ===")
        summary = self.trading_integrator.get_session_summary()
        
        print(f"Total de decisões: {summary.get('total_decisions', 0)}")
        print(f"Operações registradas: {summary.get('trade_operations', 0)}")
        print(f"Análises sem operação: {summary.get('no_trade_decisions', 0)}")
        print(f"Análises de mercado: {summary.get('market_analyses', 0)}")
        print(f"Atualizações de saldo: {summary.get('balance_updates', 0)}")
        
        # 4. Exportar relatório
        print("\n=== EXPORTANDO RELATÓRIO ===")
        report_file = self.trading_integrator.export_session_report()
        if report_file:
            print(f"Relatório exportado: {report_file}")
        else:
            print("Erro ao exportar relatório")
    
    def demonstrate_api_usage(self):
        """Demonstra uso das APIs do sistema"""
        print("\n=== DEMONSTRAÇÃO DE APIS ===")
        
        # Exemplo de registro via API
        print("\n1. Registrando análise de mercado via API:")
        decision_id = self.mt5_auth.log_market_analysis(
            symbols=['XAUUSD'],
            analysis_results={
                'gold_analysis': {
                    'trend': 'bullish',
                    'support': 1985.50,
                    'resistance': 2010.00,
                    'rsi': 58.2
                }
            },
            observations=["Ouro em tendência de alta", "Suporte forte em 1985"],
            recommendations=["Considerar compras em pullbacks"],
            confidence=0.7
        )
        print(f"   Análise registrada - ID: {decision_id}")
        
        print("\n2. Registrando decisão de trading via API:")
        decision_id = self.mt5_auth.log_trade_decision(
            symbol='XAUUSD',
            action='BUY',
            volume=0.05,
            analysis_factors=["Rompimento de resistência", "Volume confirmando"],
            justification="Ouro rompeu resistência com volume, entrada em pullback",
            confidence=0.75
        )
        print(f"   Decisão registrada - ID: {decision_id}")
        
        print("\n3. Registrando análise sem operação via API:")
        decision_id = self.mt5_auth.log_no_trade_analysis(
            symbol='GBPJPY',
            reasons=["Volatilidade excessiva", "Falta de confluência"],
            analysis_performed=["Análise técnica", "Verificação de risco"],
            confidence=0.85
        )
        print(f"   Análise registrada - ID: {decision_id}")
        
        print("\n4. Obtendo resumo da sessão:")
        summary = self.mt5_auth.get_detailed_session_summary()
        for key, value in summary.items():
            print(f"   {key}: {value}")
    
    def cleanup(self):
        """Limpa recursos e finaliza sessão"""
        print("\n=== FINALIZANDO SESSÃO ===")
        
        if self.trading_integrator:
            # Obter estatísticas finais
            final_stats = self.trading_integrator.get_session_summary()
            print(f"Estatísticas finais: {final_stats}")
            
            # Parar monitoramento
            self.trading_integrator.stop_monitoring()
        
        # Desconectar
        self.mt5_auth.disconnect()
        print("Sistema finalizado com sucesso")

def main():
    """Função principal de demonstração"""
    print("SISTEMA DE LOGGING DETALHADO - DEMONSTRAÇÃO")
    print("=" * 50)
    
    try:
        # Criar instância da demonstração
        demo = TradingAnalysisDemo()
        
        if not demo.trading_integrator:
            print("Erro na inicialização. Encerrando.")
            return
        
        # Executar demonstração completa
        demo.simulate_trading_session()
        
        # Demonstrar uso das APIs
        demo.demonstrate_api_usage()
        
        # Aguardar um pouco para mostrar monitoramento
        print("\nAguardando 10 segundos para demonstrar monitoramento...")
        time.sleep(10)
        
    except KeyboardInterrupt:
        print("\nDemonstração interrompida pelo usuário")
    except Exception as e:
        print(f"Erro durante demonstração: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Sempre limpar recursos
        if 'demo' in locals():
            demo.cleanup()

if __name__ == "__main__":
    main()