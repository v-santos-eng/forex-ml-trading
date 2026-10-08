#!/usr/bin/env python3
"""
Script automatizado para ajustar o threshold de confiança baseado na análise estatística
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

def quick_confidence_analysis(symbols, num_samples=2):
    """Análise rápida de confiança dos sinais ML"""
    print("🔍 ANÁLISE RÁPIDA DE CONFIANÇA")
    print("=" * 40)
    
    ml_predictor = MLPredictor()
    confidence_data = []
    
    for symbol in symbols[:3]:  # Apenas 3 símbolos para ser rápido
        print(f"📊 {symbol}...", end=" ")
        symbol_confidences = []
        
        for i in range(num_samples):
            try:
                prediction_result = ml_predictor.predict_direction(symbol + "=X")
                if prediction_result:
                    direction, confidence, prob_up, prob_down = prediction_result
                    symbol_confidences.append(confidence)
                    confidence_data.append(confidence)
            except Exception as e:
                print(f"Erro: {e}")
                break
        
        if symbol_confidences:
            avg_conf = sum(symbol_confidences) / len(symbol_confidences)
            print(f"Média: {avg_conf:.3f}")
    
    return confidence_data

def calculate_optimal_threshold(confidence_data):
    """Calcula threshold ótimo baseado nos dados"""
    if not confidence_data:
        return 0.55  # Fallback conservador
    
    avg_conf = sum(confidence_data) / len(confidence_data)
    min_conf = min(confidence_data)
    max_conf = max(confidence_data)
    
    print(f"\n📈 ESTATÍSTICAS:")
    print(f"   Confiança média: {avg_conf:.3f}")
    print(f"   Faixa: {min_conf:.3f} - {max_conf:.3f}")
    print(f"   Amostras: {len(confidence_data)}")
    
    # Se a variação é muito baixa (todos os valores similares), usar média - margem
    if max_conf - min_conf < 0.05:
        optimal = max(avg_conf - 0.02, 0.50)  # Margem pequena, mínimo 0.50
        print(f"   Baixa variação detectada, usando média - 0.02")
    else:
        # Usar percentil 40 (60% dos sinais passam)
        confidence_data.sort()
        optimal = confidence_data[int(len(confidence_data) * 0.4)]
        print(f"   Usando percentil 40 para 60% aprovação")
    
    return round(optimal, 3)

def update_config_threshold(new_threshold):
    """Atualiza o threshold na configuração"""
    config = load_config()
    old_threshold = config.get('min_confidence', 0.65)
    
    # Criar backup
    backup_file = backup_config()
    
    # Atualizar threshold
    config['min_confidence'] = new_threshold
    
    # Salvar configuração
    with open('auto_trader_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ THRESHOLD ATUALIZADO:")
    print(f"   {old_threshold} → {new_threshold}")
    print(f"   Backup: {backup_file}")
    
    return new_threshold

def main():
    """Função principal"""
    print("🎯 AJUSTE AUTOMÁTICO DE THRESHOLD")
    print("=" * 50)
    
    # Carregar configuração
    config = load_config()
    symbols = config.get('symbols', ['EURUSD', 'GBPUSD', 'USDJPY'])
    current_threshold = config.get('min_confidence', 0.65)
    
    print(f"📋 Threshold atual: {current_threshold}")
    print(f"📋 Símbolos: {symbols[:3]}")
    
    # Análise rápida
    confidence_data = quick_confidence_analysis(symbols)
    
    if not confidence_data:
        print("❌ Não foi possível coletar dados de confiança")
        return
    
    # Calcular threshold ótimo
    optimal_threshold = calculate_optimal_threshold(confidence_data)
    
    print(f"\n💡 RECOMENDAÇÃO: {optimal_threshold}")
    
    # Aplicar automaticamente se a diferença for significativa
    if abs(current_threshold - optimal_threshold) > 0.05:
        print(f"\n🔄 Aplicando ajuste automático...")
        new_threshold = update_config_threshold(optimal_threshold)
        
        print(f"\n🎉 AJUSTE CONCLUÍDO!")
        print(f"\n📝 PRÓXIMOS PASSOS:")
        print(f"   1. Reiniciar AutoTrader para aplicar mudanças")
        print(f"   2. Monitorar taxa de execução de sinais")
        print(f"   3. Executar novamente se necessário")
        
        return new_threshold
    else:
        print(f"\n⏸️  Threshold atual ({current_threshold}) já está adequado")
        print(f"   Diferença pequena: {abs(current_threshold - optimal_threshold):.3f}")
        return current_threshold

if __name__ == "__main__":
    result = main()
    print(f"\n🏁 Processo concluído. Threshold final: {result}")