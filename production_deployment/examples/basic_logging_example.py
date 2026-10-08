
#!/usr/bin/env python3
"""
Exemplo Básico do Sistema de Logging
"""

from comprehensive_analysis_logger import (
    initialize_comprehensive_logger, get_comprehensive_logger,
    AnalysisData, AnalysisType, DataSource
)
from detailed_decision_logger import (
    initialize_detailed_logger, get_detailed_logger,
    MarketConditions, AccountState
)

def main():
    # Inicializar loggers
    initialize_comprehensive_logger()
    initialize_detailed_logger()
    
    comp_logger = get_comprehensive_logger()
    detail_logger = get_detailed_logger()
    
    # Exemplo de análise técnica
    analysis = AnalysisData(
                analysis_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                symbol='EURUSD',
                timeframe='H1',
                analysis_type=AnalysisType.TECHNICAL,
                data_sources=[DataSource.ML_MODEL],
                technical_indicators=[],
                price_levels=[],
                chart_patterns=[],
                economic_events=[],
                news_items=[],
                risk_metrics=[],
                market_conditions=[],
                correlations={},
                sentiment_data={
                    'rsi': 65.5,
                    'macd': 0.0012,
                    'trend': 'bullish'
                },
                key_observations=[],
                risk_factors=[],
                opportunities=[],
                recommendations=[],
                analyst='system',
                confidence_score=0.87,
                processing_time_ms=150,
                data_quality_score=0.9
            )
    
    comp_logger._save_analysis(test_analysis, comp_logger.technical_log)
    print("Análise técnica registrada com sucesso!")
    
    # Exemplo de decisão de não operar
    market_conditions = MarketConditions(
        volatility='high',
        trend='neutral',
        volume='low',
        spread=0.0003,
        liquidity='normal'
    )
    
    account_state = AccountState(
        balance=10000.0,
        equity=10050.0,
        margin=200.0,
        free_margin=9850.0,
        margin_level=5025.0,
        open_positions=1
    )
    
    detail_logger.log_no_trade_decision(
        symbol='EURUSD',
        market_conditions=market_conditions,
        account_state=account_state,
        reasoning='Volatilidade muito alta para entrada segura'
    )
    
    print("Decisão de não operar registrada com sucesso!")
    
    # Gerar relatório
    report_path = comp_logger.export_session_report('./reports')
    print(f"Relatório gerado: {report_path}")

if __name__ == '__main__':
    main()
