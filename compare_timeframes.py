#!/usr/bin/env python3
"""
Script para consolidar e comparar métricas de treinamento entre timeframes M5, M15, M30
"""
import json
import os
from datetime import datetime

def main():
    # Buscar todos os relatórios de treinamento
    reports = []
    for file in sorted(os.listdir('.')):
        if file.startswith('model_training_report_') and file.endswith('.json'):
            reports.append(file)

    print(f"📊 ANÁLISE COMPARATIVA DE TIMEFRAMES")
    print(f"=" * 60)
    print(f"Relatórios encontrados: {len(reports)}")

    results = {}

    for rep_file in reports:
        with open(rep_file, 'r', encoding='utf-8') as f:
            rep = json.load(f)
        
        timestamp = rep.get('timestamp', '')
        summ = rep.get('training_summary', {})
        models_dir = summ.get('models_directory', '')
        
        # Inferir timeframe pelo diretório
        timeframe = 'M1'
        if 'M5' in models_dir:
            timeframe = 'M5'
        elif 'M15' in models_dir:
            timeframe = 'M15'
        elif 'M30' in models_dir:
            timeframe = 'M30'
        
        pairs_data = rep.get('pair_results', {})
        
        print(f"\n🎯 TIMEFRAME {timeframe} (Rel: {rep_file})")
        print(f"   Timestamp: {timestamp}")
        print(f"   Pares treinados: {summ.get('total_pairs_trained', 0)}")
        print(f"   Modelos salvos: {summ.get('total_models_trained', 0)}")
        
        results[timeframe] = {}
        
        for pair, data in pairs_data.items():
            s = data.get('summary', {})
            results[timeframe][pair] = {
                'train_samples': s.get('training_samples', 0),
                'val_samples': s.get('validation_samples', 0),
                'features': s.get('features_count', 0),
                'best_r2': s.get('best_regression_r2', 0),
                'best_f1': s.get('best_classification_f1', 0),
                'best_reg_model': data.get('best_regression_model', ''),
                'best_clf_model': data.get('best_classification_model', '')
            }
            
            print(f"   - {pair}: R²={s.get('best_regression_r2', 0):.3f}, F1={s.get('best_classification_f1', 0):.3f}")
            print(f"     Samples: train={s.get('training_samples', 0)}, val={s.get('validation_samples', 0)}")

    print(f"\n📈 COMPARAÇÃO POR PAR:")
    print(f"=" * 80)

    for pair in ['EURUSD', 'USDJPY']:
        print(f"\n🔹 {pair}:")
        print(f"   TF   │ Train  │ Val   │ Feat │   R²    │   F1    │ Best Reg Model")
        print(f"   ─────┼────────┼───────┼──────┼─────────┼─────────┼──────────────────")
        
        for tf in ['M5', 'M15', 'M30']:
            if tf in results and pair.lower() in results[tf]:
                data = results[tf][pair.lower()]
                reg_model = data['best_reg_model'].replace('_target_1', '')
                print(f"   {tf:4} │ {data['train_samples']:6} │ {data['val_samples']:5} │ {data['features']:4} │ {data['best_r2']:7.3f} │ {data['best_f1']:7.3f} │ {reg_model}")
            else:
                print(f"   {tf:4} │      - │     - │    - │       - │       - │ (não encontrado)")

    print(f"\n💡 OBSERVAÇÕES:")
    print(f"   - R² negativo indica modelos abaixo do baseline")
    print(f"   - F1 próximo de 0.5 sugere pouco poder preditivo")
    print(f"   - Timeframes maiores tendem a ter menos amostras")

if __name__ == "__main__":
    main()