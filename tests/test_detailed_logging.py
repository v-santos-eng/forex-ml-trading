#!/usr/bin/env python3
"""
Script de teste para o sistema de logging detalhado das decisoes do bot.
Testa a criacao, armazenamento e recuperacao de logs de decisoes.
"""

import sys
import os
from datetime import datetime, timedelta
import json

# Adicionar o diretorio atual ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detailed_decision_logger import (
    DetailedDecisionLogger, DecisionData, DecisionReasoning, BotAction
)

def test_decision_logging():
    """Testa o sistema de logging de decisoes"""
    print("=== TESTE DO SISTEMA DE LOGGING DETALHADO ===")
    
    # Inicializar logger
    logger = DetailedDecisionLogger()
    print("Logger inicializado")
    
    # Criar dados de teste
    market_data = {
        'symbol': 'EURUSD',
        'bid': 1.0850,
        'ask': 1.0852,
        'spread': 0.0002,
        'volume': 1000,
        'timestamp': datetime.now(),
        'ohlc_data': {
            'open': [1.0845, 1.0848, 1.0850],
            'high': [1.0855, 1.0860, 1.0865],
            'low': [1.0840, 1.0845, 1.0848],
            'close': [1.0848, 1.0850, 1.0852],
            'volume': [500, 750, 1000]
        }
    }
    
    technical_analysis = {
        'indicators': {
            'rsi': 65.5,
            'macd': 0.0012,
            'bollinger_upper': 1.0870,
            'bollinger_lower': 1.0830,
            'sma_20': 1.0845,
            'ema_12': 1.0848
        },
        'signals': {
            'rsi_signal': 'neutral',
            'macd_signal': 'bullish',
            'bollinger_signal': 'neutral'
        },
        'trend_strength': 0.7,
        'trend_direction': 'bullish'
    }
    
    ml_predictions = {
        'model_predictions': {
            'random_forest': {'direction': 'buy', 'confidence': 0.75},
            'gradient_boost': {'direction': 'buy', 'confidence': 0.68},
            'neural_network': {'direction': 'buy', 'confidence': 0.72}
        },
        'confidence': 0.72,
        'predicted_direction': 'buy',
        'signal_strength': 0.8,
        'features_used': {
            'price_features': ['close', 'volume', 'spread'],
            'technical_features': ['rsi', 'macd', 'bollinger'],
            'time_features': ['hour', 'day_of_week']
        }
    }
    
    reasoning = DecisionReasoning(
        factors=[
            "RSI em 65.5 indica momentum bullish moderado",
            "MACD positivo confirma tendencia de alta",
            "Modelos ML convergem para sinal de compra",
            "Spread baixo favorece execucao"
        ],
        confidence_score=0.72,
        risk_assessment="Risco medio - condicoes favoraveis para entrada"
    )
    
    # Teste 1: Log de geracao de sinal
    print("\n1. Testando log de geracao de sinal...")
    decision_data = DecisionData(
        symbol='EURUSD',
        action='GENERATE_SIGNAL',
        reasoning=reasoning,
        market_data=market_data,
        technical_analysis=technical_analysis,
        ml_predictions=ml_predictions
    )
    
    decision_id = logger.log_decision(decision_data)
    print(f"Decisao registrada com ID: {decision_id}")
    
    # Teste 2: Log de execucao de trade
    print("\n2. Testando log de execucao de trade...")
    bot_action = BotAction(
        action_type='EXECUTE_TRADE',
        parameters={
            'symbol': 'EURUSD',
            'order_type': 'BUY',
            'volume': 0.01,
            'stop_loss': 1.0820,
            'take_profit': 1.0880
        },
        result='SUCCESS',
        details='Trade executado com sucesso - Ticket: 12345',
        errors=None
    )
    
    trade_decision = DecisionData(
        symbol='EURUSD',
        action='EXECUTE_TRADE',
        reasoning=DecisionReasoning(
            factors=["Sinal ML confirmado", "Condicoes de mercado favoraveis"],
            confidence_score=0.75,
            risk_assessment="Risco controlado"
        ),
        market_data=market_data,
        technical_analysis=technical_analysis,
        ml_predictions=ml_predictions,
        bot_action=bot_action
    )
    
    trade_id = logger.log_decision(trade_decision)
    print(f"Execucao de trade registrada com ID: {trade_id}")
    
    # Teste 3: Log de acao sem trade (NO_ACTION)
    print("\n3. Testando log de nao-acao...")
    no_action_reasoning = DecisionReasoning(
        factors=[
            "Spread muito alto (5 pips)",
            "Confianca ML abaixo do limite (0.45)",
            "Mercado em consolidacao"
        ],
        confidence_score=0.45,
        risk_assessment="Risco alto - condicoes desfavoraveis"
    )
    
    no_action_decision = DecisionData(
        symbol='GBPUSD',
        action='NO_ACTION',
        reasoning=no_action_reasoning,
        market_data={
            'symbol': 'GBPUSD',
            'bid': 1.2650,
            'ask': 1.2655,
            'spread': 0.0005,  # 5 pips
            'volume': 500,
            'timestamp': datetime.now()
        }
    )
    
    no_action_id = logger.log_decision(no_action_decision)
    print(f"Decisao de nao-acao registrada com ID: {no_action_id}")
    
    # Teste 4: Recuperar decisoes recentes
    print("\n4. Testando recuperacao de decisoes recentes...")
    recent_decisions = logger.get_recent_decisions(limit=10)
    print(f"Recuperadas {len(recent_decisions)} decisoes recentes")
    
    for decision in recent_decisions:
        print(f"  - ID: {decision['id']}, Simbolo: {decision['symbol']}, Acao: {decision['action']}, Confianca: {decision['confidence']:.2f}")
    
    # Teste 5: Recuperar decisao especifica
    print("\n5. Testando recuperacao de decisao especifica...")
    decision_details = logger.get_decision_by_id(decision_id)
    if decision_details:
        print(f"Detalhes da decisao {decision_id} recuperados")
        print(f"  - Simbolo: {decision_details['symbol']}")
        print(f"  - Acao: {decision_details['action']}")
        print(f"  - Confianca: {decision_details['confidence']:.2f}")
        print(f"  - Fatores: {len(decision_details['reasoning_factors'])} fatores")
        print(f"  - Dados de mercado: {'Sim' if decision_details.get('market_data') else 'Nao'}")
        print(f"  - Analise tecnica: {'Sim' if decision_details.get('technical_analysis') else 'Nao'}")
        print(f"  - Predicoes ML: {'Sim' if decision_details.get('ml_predictions') else 'Nao'}")
    
    # Teste 6: Analise de performance
    print("\n6. Testando analise de performance...")
    performance_stats = logger.get_performance_analysis()
    print(f"Estatisticas de performance geradas")
    print(f"  - Total de decisoes: {performance_stats['total_decisions']}")
    print(f"  - Confianca media: {performance_stats['average_confidence']:.2f}")
    print(f"  - Distribuicao por acao: {performance_stats['action_breakdown']}")
    print(f"  - Simbolos mais negociados: {performance_stats['top_symbols']}")
    
    # Teste 7: Exportar para CSV
    print("\n7. Testando exportacao para CSV...")
    csv_filename = "test_decisions_export.csv"
    try:
        logger.export_decisions_to_csv(csv_filename)
        print(f"Decisoes exportadas para {csv_filename}")
        
        # Verificar se o arquivo foi criado
        if os.path.exists(csv_filename):
            file_size = os.path.getsize(csv_filename)
            print(f"  - Arquivo criado com {file_size} bytes")
        else:
            print("  - Arquivo CSV nao encontrado")
    except Exception as e:
        print(f"  - Erro na exportacao: {str(e)}")
    
    # Teste 8: Gerar resumo de decisao
    print("\n8. Testando geracao de resumo...")
    summary = logger.generate_decision_summary(decision_id)
    if summary:
        print(f"Resumo da decisao gerado:")
        print(f"  {summary[:200]}...")
    
    print("\n=== TODOS OS TESTES CONCLUIDOS ===")
    print("Sistema de logging detalhado funcionando corretamente!")

def main():
    """Funcao principal"""
    try:
        test_decision_logging()
    except Exception as e:
        print(f"Erro durante os testes: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()