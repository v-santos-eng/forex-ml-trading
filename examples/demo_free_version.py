#!/usr/bin/env python3
"""
Demo da Versão Gratuita do Forex Analyzer
Este exemplo mostra como usar o sistema sem APIs pagas.
"""

import sys
import os
from datetime import datetime, timedelta

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from forex_analyzer_free import ForexAnalyzerFree
    from ml_predictor_free import MLForexPredictorFree
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Certifique-se de que os arquivos forex_analyzer_free.py e ml_predictor_free.py estão no mesmo diretório.")
    sys.exit(1)

def main():
    print("=" * 60)
    print("FOREX ANALYZER - VERSÃO GRATUITA")
    print("Sistema de Análise de Forex sem APIs Pagas")
    print("=" * 60)
    
    # Configurações
    symbol = "EURUSD=X"  # Par de moedas para Yahoo Finance
    period = "1y"  # 1 ano de dados históricos
    
    print(f"\nAnalisando: {symbol}")
    print(f"Período: {period}")
    print("-" * 40)
    
    try:
        # 1. Análise Técnica Básica
        print("\n1. ANÁLISE TÉCNICA")
        print("-" * 20)
        
        analyzer = ForexAnalyzerFree()
        
        # Realizar análise completa
        print("Realizando análise técnica...")
        analysis_result = analyzer.analyze_pair(symbol, period)
        
        if analysis_result is not None:
            print("\nResultados da Análise Técnica:")
            for key, value in analysis_result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.5f}")
                else:
                    print(f"  {key}: {value}")
            
            # Obter dados para ML
            data = analyzer.get_forex_data(symbol, period)
            if data is not None and not data.empty:
                print(f"\nDados obtidos: {len(data)} registros")
                print(f"Período: {data.index[0].date()} até {data.index[-1].date()}")
        
        # 2. Predição com Machine Learning
        print("\n\n2. PREDIÇÃO COM MACHINE LEARNING")
        print("-" * 35)
        
        predictor = MLForexPredictorFree()
        
        print("Preparando dados para ML...")
        ml_features, ml_target = predictor.prepare_data(symbol, period)
        
        if ml_features is not None and ml_target is not None and len(ml_features) > 0:
            print(f"Dados preparados: {len(ml_features)} registros")
            
            # Treinar modelo
            print("Treinando modelos de ML...")
            results = predictor.train_models(ml_features, ml_target)
            
            if results:
                print("\nResultados da Predição:")
                print(f"  Melhor Modelo: {results['best_model']}")
                print(f"  Acurácia: {results['accuracy']:.2%}")
                print(f"  Predição Próxima: {results['prediction']}")
                print(f"  Confiança: {results['confidence']:.2%}")
                
                # Importância das features
                if 'feature_importance' in results:
                    print("\n  Top 5 Indicadores Mais Importantes:")
                    for i, (feature, importance) in enumerate(results['feature_importance'][:5], 1):
                        print(f"    {i}. {feature}: {importance:.3f}")
        
        # 3. Resumo e Recomendações
        print("\n\n3. RESUMO E RECOMENDAÇÕES")
        print("-" * 30)
        
        if 'data' in locals() and data is not None:
            last_price = data['Close'].iloc[-1]
            price_change = ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100
            
            print(f"Preço Atual: {last_price:.5f}")
            print(f"Variação (1 dia): {price_change:+.2f}%")
            
            # Recomendação baseada na análise técnica
            if 'analysis_result' in locals() and analysis_result:
                trend = analysis_result.get('Tendência', 'NEUTRO')
                rsi_signal = analysis_result.get('RSI_Signal', 'NEUTRO')
                macd_signal = analysis_result.get('MACD_Signal', 'NEUTRO')
                
                print(f"\nAnálise Técnica:")
                print(f"Tendência: {trend}")
                print(f"RSI: {rsi_signal}")
                print(f"MACD: {macd_signal}")
                
                # Recomendação geral
                bullish_signals = sum(1 for signal in [trend, macd_signal] if signal in ['ALTA', 'COMPRA'])
                bearish_signals = sum(1 for signal in [trend, macd_signal] if signal in ['BAIXA', 'VENDA'])
                
                if bullish_signals > bearish_signals:
                    recommendation = "COMPRA"
                elif bearish_signals > bullish_signals:
                    recommendation = "VENDA"
                else:
                    recommendation = "NEUTRO"
                
                print(f"\nRecomendação Geral: {recommendation}")
            
            # Combinar com ML se disponível
            if 'results' in locals() and results:
                ml_signal = "COMPRA" if results['prediction'] == 1 else "VENDA"
                print(f"Predição ML: {ml_signal} (Confiança: {results['confidence']:.1%})")
        
        print("\n" + "=" * 60)
        print("ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("\nPróximos Passos:")
        print("1. Analise os sinais técnicos e predições ML")
        print("2. Considere o contexto fundamental do mercado")
        print("3. Implemente gestão de risco adequada")
        print("4. Teste estratégias em conta demo primeiro")
        print("\nAviso: Esta análise é apenas educacional.")
        print("Não constitui aconselhamento financeiro.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nErro durante a análise: {e}")
        print("\nPossíveis soluções:")
        print("1. Verifique sua conexão com a internet")
        print("2. Instale as dependências: pip install -r requirements_free.txt")
        print("3. Tente com um símbolo diferente (ex: GBPUSD=X, USDJPY=X)")
        return False
    
    return True

def show_available_symbols():
    """Mostra símbolos disponíveis para análise"""
    print("\nSímbolos Disponíveis (Yahoo Finance):")
    symbols = {
        "EURUSD=X": "Euro / Dólar Americano",
        "GBPUSD=X": "Libra / Dólar Americano", 
        "USDJPY=X": "Dólar Americano / Iene Japonês",
        "USDCHF=X": "Dólar Americano / Franco Suíço",
        "AUDUSD=X": "Dólar Australiano / Dólar Americano",
        "USDCAD=X": "Dólar Americano / Dólar Canadense",
        "NZDUSD=X": "Dólar Neozelandês / Dólar Americano"
    }
    
    for symbol, description in symbols.items():
        print(f"  {symbol}: {description}")

if __name__ == "__main__":
    print("Iniciando demonstração da versão gratuita...")
    
    # Mostrar símbolos disponíveis
    show_available_symbols()
    
    # Executar análise principal
    success = main()
    
    if success:
        print("\nDemonstração concluída com sucesso!")
    else:
        print("\nDemonstração finalizada com erros.")