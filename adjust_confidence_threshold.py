#!/usr/bin/env python3
"""
Script para ajustar o threshold de confiança baseado na análise estatística dos sinais ML
"""

import json
import shutil
from datetime import datetime
from ml_predictor_local import MLForexPredictorLocal as MLPredictor

def load_config():
    """Carrega configuração do AutoTrader"""
    try:
        with open('auto_trader_config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Erro ao carregar configuração: {e}")
        return {}

def backup_config():
    """Cria backup da configuração atual"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f'auto_trader_config_backup_{timestamp}.json'
    shutil.copy('auto_trader_config.json', backup_file)
    print(f"📋 Backup criado: {backup_file}")
    return backup_file

def analyze_confidence_distribution(symbols, num_samples=5):
    """Analisa a distribuição de confiança dos sinais ML"""
    print("🔍 ANÁLISE DE DISTRIBUIÇÃO DE CONFIANÇA")
    print("=" * 50)
    
    ml_predictor = MLPredictor()
    confidence_data = []
    
    for symbol in symbols:
        print(f"\n📊 Analisando {symbol}...")
        symbol_confidences = []
        
        for i in range(num_samples):
            try:
                prediction_result = ml_predictor.predict_direction(symbol + "=X")
                if prediction_result:
                    direction, confidence, prob_up, prob_down = prediction_result
                    symbol_confidences.append(confidence)
                    confidence_data.append(confidence)
                    print(f"   Amostra {i+1}: {direction} - {confidence:.3f}")
            except Exception as e:
                print(f"   ❌ Erro na amostra {i+1}: {e}")
        
        if symbol_confidences:
            avg_conf = sum(symbol_confidences) / len(symbol_confidences)
            min_conf = min(symbol_confidences)
            max_conf = max(symbol_confidences)
            print(f"   📈 {symbol}: Média={avg_conf:.3f}, Min={min_conf:.3f}, Max={max_conf:.3f}")
    
    return confidence_data

def calculate_optimal_threshold(confidence_data, target_pass_rate=0.6):
    """Calcula threshold ótimo baseado na taxa de aprovação desejada"""
    if not confidence_data:
        return 0.65
    
    confidence_data.sort()
    n = len(confidence_data)
    
    # Calcular threshold para diferentes taxas de aprovação
    thresholds = {
        0.8: confidence_data[int(n * 0.2)],  # 80% dos sinais passam
        0.7: confidence_data[int(n * 0.3)],  # 70% dos sinais passam
        0.6: confidence_data[int(n * 0.4)],  # 60% dos sinais passam
        0.5: confidence_data[int(n * 0.5)],  # 50% dos sinais passam
    }
    
    print(f"\n📊 THRESHOLDS SUGERIDOS:")
    print(f"   Para 80% aprovação: {thresholds[0.8]:.3f}")
    print(f"   Para 70% aprovação: {thresholds[0.7]:.3f}")
    print(f"   Para 60% aprovação: {thresholds[0.6]:.3f}")
    print(f"   Para 50% aprovação: {thresholds[0.5]:.3f}")
    
    # Estatísticas gerais
    avg_conf = sum(confidence_data) / len(confidence_data)
    min_conf = min(confidence_data)
    max_conf = max(confidence_data)
    
    print(f"\n📈 ESTATÍSTICAS GERAIS:")
    print(f"   Confiança média: {avg_conf:.3f}")
    print(f"   Confiança mínima: {min_conf:.3f}")
    print(f"   Confiança máxima: {max_conf:.3f}")
    print(f"   Total de amostras: {len(confidence_data)}")
    
    return thresholds.get(target_pass_rate, avg_conf - 0.05)

def update_config_threshold(new_threshold):
    """Atualiza o threshold na configuração"""
    config = load_config()
    old_threshold = config.get('min_confidence', 0.65)
    
    # Criar backup antes de modificar
    backup_file = backup_config()
    
    # Atualizar threshold
    config['min_confidence'] = round(new_threshold, 3)
    
    # Salvar configuração atualizada
    with open('auto_trader_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ CONFIGURAÇÃO ATUALIZADA:")
    print(f"   Threshold anterior: {old_threshold}")
    print(f"   Novo threshold: {config['min_confidence']}")
    print(f"   Backup salvo em: {backup_file}")
    
    return config['min_confidence']

def main():
    """Função principal"""
    print("🎯 AJUSTE DE THRESHOLD DE CONFIANÇA")
    print("=" * 60)
    
    # Carregar configuração
    config = load_config()
    symbols = config.get('symbols', ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'])
    current_threshold = config.get('min_confidence', 0.65)
    
    print(f"📋 Configuração atual:")
    print(f"   Threshold atual: {current_threshold}")
    print(f"   Símbolos: {symbols}")
    
    # Analisar distribuição de confiança
    confidence_data = analyze_confidence_distribution(symbols, num_samples=3)
    
    if not confidence_data:
        print("❌ Não foi possível coletar dados de confiança")
        return
    
    # Calcular threshold ótimo
    optimal_threshold = calculate_optimal_threshold(confidence_data, target_pass_rate=0.6)
    
    print(f"\n💡 RECOMENDAÇÃO:")
    print(f"   Threshold ótimo sugerido: {optimal_threshold:.3f}")
    print(f"   Isso deve permitir ~60% dos sinais passarem")
    
    # Perguntar se deve aplicar a mudança
    response = input(f"\n❓ Deseja atualizar o threshold de {current_threshold} para {optimal_threshold:.3f}? (s/n): ")
    
    if response.lower() in ['s', 'sim', 'y', 'yes']:
        new_threshold = update_config_threshold(optimal_threshold)
        print(f"\n🎉 Threshold atualizado com sucesso para {new_threshold}!")
        print(f"\n📝 PRÓXIMOS PASSOS:")
        print(f"   1. Reinicie o AutoTrader para aplicar as mudanças")
        print(f"   2. Monitore a taxa de execução de sinais")
        print(f"   3. Ajuste novamente se necessário")
    else:
        print(f"\n⏸️  Threshold mantido em {current_threshold}")
        print(f"   Para aplicar manualmente, edite 'min_confidence' em auto_trader_config.json")

if __name__ == "__main__":
    main()