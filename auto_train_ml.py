#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treinamento Automatizado de Modelos ML para Forex
Utiliza dados do Yahoo Finance com configurações otimizadas
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, timedelta
from ml_predictor_enhanced import MLForexPredictorEnhanced

def convert_numpy_types(obj):
    """Converte tipos numpy para tipos Python nativos para serialização JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def main():
    print("🤖 Treinamento Automatizado de Modelos ML para Forex")
    print("=" * 60)
    
    # Configuração otimizada
    selected_pairs = [
        'EURUSD=X',  # Euro/USD
        'GBPUSD=X',  # Libra/USD
        'USDJPY=X',  # USD/Yen
        'AUDUSD=X',  # Dólar Australiano/USD
        'USDCAD=X'   # USD/Dólar Canadense
    ]
    
    # Período de dados otimizado para robustez
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 ano de dados
    
    print(f"📅 Período de treinamento: {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")
    print(f"💱 Pares selecionados: {', '.join(selected_pairs)}")
    print(f"⏱️  Timeframe: 1 hora (otimizado para ML)")
    print("\n🚀 Iniciando treinamento...\n")
    
    results = {}
    
    for i, pair in enumerate(selected_pairs, 1):
        print(f"\n[{i}/{len(selected_pairs)}] 📈 Treinando modelo para {pair}")
        print("-" * 50)
        
        try:
            # Inicializar o preditor
            predictor = MLForexPredictorEnhanced(
                use_pca=True,
                n_components=0.95,
                feature_selection_method='auto'
            )
            
            # Obter dados
            data = predictor.get_forex_data(pair, period='1y')
            if data is None or data.empty:
                print(f"❌ Dados insuficientes para {pair}")
                results[pair] = {
                    'status': 'failed',
                    'error': 'Dados insuficientes',
                    'timestamp': datetime.now().isoformat()
                }
                continue
            
            # Criar features e target
            features = predictor.create_enhanced_features(data)
            target = predictor.create_target(data, prediction_horizon=1, threshold=0.001)
            
            print(f"   📊 Features criadas: {len(predictor.feature_names)}")
            
            # Walk-forward optimization
            print(f"   🔄 Executando Walk-Forward Optimization...")
            wf_results = predictor.walk_forward_optimization(features, target)
            
            # Treinar modelo final
            print(f"   🎯 Treinando modelo final...")
            model_info = predictor.train_final_model(features, target)
            
            # Fazer predição
            prediction = predictor.predict_direction(pair)
            
            # Obter resumo do modelo
            model_summary = predictor.get_model_summary()
            
            if model_info and prediction:
                print(f"✅ Modelo treinado com sucesso para {pair}")
                print(f"   📊 CV Score: {model_info.get('cv_score_mean', 'N/A'):.4f} ± {model_info.get('cv_score_std', 0):.4f}")
                print(f"   🎯 Direção: {prediction.get('direction', 'N/A')}")
                print(f"   📈 Confiança: {prediction.get('confidence', 'N/A'):.1f}%")
                print(f"   🔢 Features: {model_info.get('n_features_selected', 'N/A')}")
                
                results[pair] = {
                    'status': 'success',
                    'model_info': model_info,
                    'prediction': prediction,
                    'model_summary': model_summary,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                print(f"❌ Falha no treinamento para {pair}")
                results[pair] = {
                    'status': 'failed',
                    'error': 'Treinamento ou predição falharam',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"❌ Erro no treinamento para {pair}: {str(e)}")
            results[pair] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Salvar relatório
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'auto_ml_training_report_{timestamp}.json'
    
    # Converter tipos numpy para serialização JSON
    results_serializable = convert_numpy_types(results)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("📋 RESUMO DO TREINAMENTO")
    print("=" * 60)
    
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"✅ Modelos treinados com sucesso: {successful}/{len(selected_pairs)}")
    print(f"❌ Falhas no treinamento: {failed}/{len(selected_pairs)}")
    print(f"📄 Relatório salvo em: {report_file}")
    
    if successful > 0:
        print("\n🎉 Treinamento concluído! Os modelos estão prontos para uso.")
    else:
        print("\n⚠️  Nenhum modelo foi treinado com sucesso. Verifique os logs para mais detalhes.")

if __name__ == '__main__':
    main()