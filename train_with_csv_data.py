#!/usr/bin/env python3
"""
Treinamento de Modelos ML com Dados CSV
Script para treinar modelos de ML usando dados históricos de 1 minuto do diretório local
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Optional

# Importar módulos do projeto
from csv_data_loader import CSVForexDataLoader
from ml_predictor_enhanced import MLForexPredictorEnhanced
from config import ForexConfig

class CSVMLTrainer:
    """
    Treinador de modelos ML usando dados CSV locais
    """
    
    def __init__(self, data_directory: str):
        """
        Inicializa o treinador
        
        Args:
            data_directory: Diretório contendo arquivos CSV
        """
        self.data_directory = data_directory
        self.csv_loader = CSVForexDataLoader(data_directory)
        self.ml_predictor = MLForexPredictorEnhanced(
            use_pca=True,
            n_components=0.95,
            feature_selection_method='rfe'
        )
        self.training_results = {}
        
    def get_available_pairs(self) -> List[str]:
        """
        Retorna pares disponíveis nos dados CSV
        """
        return self.csv_loader.available_pairs
    
    def optimize_data_selection(self, total_pairs: int) -> Dict[str, int]:
        """
        Otimiza seleção de dados baseado na quantidade de pares e recursos disponíveis
        
        Returns:
            Configuração otimizada com tamanhos de dados
        """
        # Configurações baseadas em performance vs robustez
        if total_pairs <= 2:
            # Poucos pares: usar mais dados por par
            config = {
                'samples_per_pair': 500000,  # ~350 dias de dados 1min
                'period': '2y',
                'description': 'Configuração robusta - máximos dados por par'
            }
        elif total_pairs <= 5:
            # Pares moderados: balancear dados e performance
            config = {
                'samples_per_pair': 300000,  # ~210 dias
                'period': '1y', 
                'description': 'Configuração balanceada - dados moderados'
            }
        elif total_pairs <= 10:
            # Muitos pares: otimizar para performance
            config = {
                'samples_per_pair': 200000,  # ~140 dias
                'period': '6mo',
                'description': 'Configuração otimizada - foco em performance'
            }
        else:
            # Muitos pares: configuração rápida
            config = {
                'samples_per_pair': 100000,  # ~70 dias
                'period': '3mo',
                'description': 'Configuração rápida - dados essenciais'
            }
            
        print(f"Configuração selecionada: {config['description']}")
        print(f"Amostras por par: {config['samples_per_pair']:,}")
        print(f"Período equivalente: {config['period']}")
        
        return config
    
    def train_single_pair(self, pair: str, config: Dict) -> Optional[Dict]:
        """
        Treina modelo para um único par
        
        Args:
            pair: Par de moedas
            config: Configuração de treinamento
            
        Returns:
            Resultados do treinamento ou None se falhou
        """
        print(f"\n{'='*60}")
        print(f"Treinando modelo para {pair}")
        print(f"{'='*60}")
        
        try:
            # Carregar dados
            print("1. Carregando dados...")
            data = self.csv_loader.load_forex_data(
                pair, 
                period=config['period'],
                sample_size=config['samples_per_pair']
            )
            
            if data is None or data.empty:
                print(f"❌ Dados insuficientes para {pair}")
                return None
                
            print(f"✅ Dados carregados: {len(data):,} registros")
            print(f"   Período: {data.index[0]} até {data.index[-1]}")
            
            # Modificar temporariamente o método get_forex_data
            original_method = self.ml_predictor.get_forex_data
            self.ml_predictor.get_forex_data = lambda p, period: data
            
            # Criar features
            print("2. Criando features...")
            features = self.ml_predictor.create_enhanced_features(data)
            target = self.ml_predictor.create_target(data, prediction_horizon=1, threshold=0.001)
            
            print(f"✅ Features criadas: {len(self.ml_predictor.feature_names)}")
            
            # Verificar dados válidos
            valid_idx = ~(features.isnull().any(axis=1) | target.isnull())
            features_clean = features[valid_idx]
            target_clean = target[valid_idx]
            
            if len(features_clean) < 100:
                print(f"❌ Dados insuficientes após limpeza: {len(features_clean)} amostras")
                return None
                
            print(f"✅ Dados limpos: {len(features_clean):,} amostras válidas")
            
            # Walk-forward optimization
            print("3. Executando Walk-Forward Optimization...")
            wf_results = self.ml_predictor.walk_forward_optimization(features_clean, target_clean)
            
            # Treinar modelo final
            print("4. Treinando modelo final...")
            final_results = self.ml_predictor.train_final_model(features_clean, target_clean)
            
            # Restaurar método original
            self.ml_predictor.get_forex_data = original_method
            
            # Compilar resultados
            results = {
                'pair': pair,
                'data_samples': len(data),
                'valid_samples': len(features_clean),
                'features_count': len(self.ml_predictor.feature_names),
                'selected_features_count': len(self.ml_predictor.selected_features),
                'walk_forward_results': wf_results,
                'final_model_results': final_results,
                'data_period': f"{data.index[0]} to {data.index[-1]}",
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Treinamento concluído para {pair}")
            if final_results:
                print(f"   Acurácia CV: {final_results.get('cv_score_mean', 0):.3f}")
                print(f"   Features selecionadas: {len(self.ml_predictor.selected_features)}")
            
            return results
            
        except Exception as e:
            print(f"❌ Erro no treinamento de {pair}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_multiple_pairs(self, pairs: Optional[List[str]] = None, max_pairs: int = 10) -> Dict:
        """
        Treina modelos para múltiplos pares
        
        Args:
            pairs: Lista específica de pares (None para usar todos disponíveis)
            max_pairs: Máximo de pares para treinar
            
        Returns:
            Resultados consolidados do treinamento
        """
        print("🚀 Iniciando Treinamento de Modelos ML com Dados CSV")
        print("="*70)
        
        # Determinar pares para treinar
        if pairs is None:
            available_pairs = self.get_available_pairs()
            pairs = available_pairs[:max_pairs]
        
        if not pairs:
            print("❌ Nenhum par disponível para treinamento")
            return {'error': 'Nenhum par disponível'}
            
        print(f"📊 Pares selecionados para treinamento: {pairs}")
        print(f"📁 Diretório de dados: {self.data_directory}")
        
        # Otimizar configuração
        config = self.optimize_data_selection(len(pairs))
        
        # Treinar cada par
        training_results = {}
        successful_trainings = 0
        
        for i, pair in enumerate(pairs, 1):
            print(f"\n🔄 Progresso: {i}/{len(pairs)} pares")
            
            result = self.train_single_pair(pair, config)
            if result:
                training_results[pair] = result
                successful_trainings += 1
            else:
                training_results[pair] = {'error': 'Treinamento falhou'}
        
        # Compilar relatório final
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': config,
            'summary': {
                'total_pairs_attempted': len(pairs),
                'successful_trainings': successful_trainings,
                'failed_trainings': len(pairs) - successful_trainings,
                'success_rate': successful_trainings / len(pairs) if pairs else 0
            },
            'training_results': training_results,
            'data_directory': self.data_directory,
            'ml_configuration': {
                'use_pca': self.ml_predictor.use_pca,
                'n_components': self.ml_predictor.n_components,
                'feature_selection_method': self.ml_predictor.feature_selection_method,
                'models': list(self.ml_predictor.base_models.keys())
            }
        }
        
        # Salvar relatório
        report_filename = f"csv_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
            
        print(f"\n🎉 Treinamento Concluído!")
        print(f"📈 Taxa de sucesso: {final_report['summary']['success_rate']:.1%}")
        print(f"✅ Modelos treinados: {successful_trainings}/{len(pairs)}")
        print(f"📄 Relatório salvo: {report_filename}")
        
        return final_report
    
    def quick_test(self) -> Dict:
        """
        Teste rápido com um par para verificar funcionamento
        """
        available_pairs = self.get_available_pairs()
        if not available_pairs:
            return {'error': 'Nenhum par disponível'}
            
        test_pair = available_pairs[0]
        print(f"🧪 Teste rápido com {test_pair}")
        
        config = {
            'samples_per_pair': 50000,  # ~35 dias para teste rápido
            'period': '1mo',
            'description': 'Configuração de teste'
        }
        
        return self.train_single_pair(test_pair, config)

def main():
    """
    Função principal
    """
    # Diretório dos dados
    data_directory = os.getenv("FX_DATA_DIR", "data/FX-1-Minute-Data")
    
    # Verificar se diretório existe
    if not os.path.exists(data_directory):
        print(f"❌ Diretório não encontrado: {data_directory}")
        print("Por favor, verifique o caminho dos dados.")
        return
    
    # Criar treinador
    trainer = CSVMLTrainer(data_directory)
    
    # Verificar dados disponíveis
    available_pairs = trainer.get_available_pairs()
    if not available_pairs:
        print("❌ Nenhum arquivo CSV encontrado no diretório")
        return
        
    print(f"📊 Pares disponíveis: {available_pairs}")
    
    # Opções de treinamento
    print("\n🎯 Opções de treinamento:")
    print("1. Teste rápido (1 par, dados limitados)")
    print("2. Treinamento otimizado (até 5 pares)")
    print("3. Treinamento completo (todos os pares)")
    print("4. Pares específicos")
    
    choice = input("\nEscolha uma opção (1-4): ").strip()
    
    if choice == '1':
        # Teste rápido
        result = trainer.quick_test()
        print("\n📋 Resultado do teste:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif choice == '2':
        # Treinamento otimizado
        result = trainer.train_multiple_pairs(max_pairs=5)
        
    elif choice == '3':
        # Treinamento completo
        result = trainer.train_multiple_pairs()
        
    elif choice == '4':
        # Pares específicos
        print(f"\nPares disponíveis: {available_pairs}")
        selected = input("Digite os pares separados por vírgula: ").strip()
        pairs = [p.strip().upper() for p in selected.split(',') if p.strip()]
        
        if pairs:
            result = trainer.train_multiple_pairs(pairs=pairs)
        else:
            print("❌ Nenhum par válido selecionado")
    
    else:
        print("❌ Opção inválida")

if __name__ == "__main__":
    main()