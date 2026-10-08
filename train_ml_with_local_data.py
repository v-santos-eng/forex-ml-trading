#!/usr/bin/env python3
"""
Treinamento ML com Dados Locais
Script flexível para treinar modelos ML com diferentes fontes de dados
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import yfinance as yf

# Importar módulos do projeto
from ml_predictor_enhanced import MLForexPredictorEnhanced
from config import ForexConfig

class FlexibleMLTrainer:
    """
    Treinador flexível de modelos ML que pode usar diferentes fontes de dados
    """
    
    def __init__(self, data_source: str = "yahoo", data_directory: str = None):
        """
        Inicializa o treinador
        
        Args:
            data_source: Fonte dos dados ('yahoo', 'csv', 'simulated')
            data_directory: Diretório para dados CSV (se aplicável)
        """
        self.data_source = data_source
        self.data_directory = data_directory
        self.ml_predictor = MLForexPredictorEnhanced(
            use_pca=True,
            n_components=0.95,
            feature_selection_method='rfe'
        )
        self.training_results = {}
        
    def get_forex_data_flexible(self, pair: str, period: str = '2y', samples: int = None) -> Optional[pd.DataFrame]:
        """
        Obtém dados de forex de diferentes fontes
        
        Args:
            pair: Par de moedas
            period: Período dos dados
            samples: Número de amostras para limitar (otimização)
            
        Returns:
            DataFrame com dados OHLCV
        """
        if self.data_source == "yahoo":
            return self._get_yahoo_data(pair, period, samples)
        elif self.data_source == "csv":
            return self._get_csv_data(pair, period, samples)
        elif self.data_source == "simulated":
            return self._get_simulated_data(pair, period, samples)
        else:
            raise ValueError(f"Fonte de dados não suportada: {self.data_source}")
    
    def _get_yahoo_data(self, pair: str, period: str, samples: int = None) -> Optional[pd.DataFrame]:
        """
        Obtém dados do Yahoo Finance com otimização
        """
        try:
            # Converter formato do par se necessário
            if '/' in pair:
                pair = pair.replace('/', '') + '=X'
            elif not pair.endswith('=X'):
                pair = pair + '=X'
                
            print(f"Obtendo dados do Yahoo Finance para {pair}...")
            ticker = yf.Ticker(pair)
            data = ticker.history(period=period)
            
            if data.empty:
                print(f"Nenhum dado encontrado para {pair}")
                return None
                
            # Otimizar tamanho se especificado
            if samples and len(data) > samples:
                data = data.tail(samples)
                print(f"Dados otimizados: {len(data)} amostras (últimas {samples})")
                
            print(f"Dados obtidos: {len(data)} registros de {data.index[0].date()} a {data.index[-1].date()}")
            return data
            
        except Exception as e:
            print(f"Erro ao obter dados do Yahoo para {pair}: {e}")
            return None
    
    def _get_csv_data(self, pair: str, period: str, samples: int = None) -> Optional[pd.DataFrame]:
        """
        Obtém dados de arquivos CSV
        """
        if not self.data_directory or not os.path.exists(self.data_directory):
            print(f"Diretório CSV não encontrado: {self.data_directory}")
            return None
            
        # Procurar arquivo CSV
        possible_files = [
            f"{pair}.csv",
            f"{pair.upper()}.csv",
            f"{pair.lower()}.csv",
            f"{pair}_1min.csv",
            f"{pair.upper()}_1min.csv"
        ]
        
        file_path = None
        for filename in possible_files:
            potential_path = os.path.join(self.data_directory, filename)
            if os.path.exists(potential_path):
                file_path = potential_path
                break
                
        if not file_path:
            print(f"Arquivo CSV não encontrado para {pair} em {self.data_directory}")
            return None
            
        try:
            print(f"Carregando dados CSV: {file_path}")
            df = pd.read_csv(file_path)
            
            # Padronizar colunas
            df = self._standardize_csv_columns(df)
            
            # Converter timestamp
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df.set_index('Datetime', inplace=True)
            
            # Otimizar tamanho
            if samples and len(df) > samples:
                df = df.tail(samples)
                
            print(f"Dados CSV carregados: {len(df)} registros")
            return df
            
        except Exception as e:
            print(f"Erro ao carregar CSV {file_path}: {e}")
            return None
    
    def _get_simulated_data(self, pair: str, period: str, samples: int = None) -> pd.DataFrame:
        """
        Gera dados simulados para teste
        """
        print(f"Gerando dados simulados para {pair}...")
        
        # Determinar número de amostras
        period_samples = {
            '1mo': 30 * 24 * 60,
            '3mo': 90 * 24 * 60,
            '6mo': 180 * 24 * 60,
            '1y': 365 * 24 * 60,
            '2y': 2 * 365 * 24 * 60
        }
        
        n_samples = samples or min(period_samples.get(period, 100000), 200000)
        
        # Gerar dados realistas
        np.random.seed(42)  # Para reprodutibilidade
        
        # Preço inicial baseado no par
        initial_prices = {
            'EURUSD': 1.1000,
            'GBPUSD': 1.3000,
            'USDJPY': 110.00,
            'USDCHF': 0.9200,
            'AUDUSD': 0.7500,
            'USDCAD': 1.2500
        }
        
        base_price = initial_prices.get(pair.replace('=X', '').replace('/', ''), 1.0000)
        
        # Gerar série temporal com random walk + tendência
        returns = np.random.normal(0, 0.0001, n_samples)  # Volatilidade realista
        trend = np.linspace(0, 0.02, n_samples)  # Tendência sutil
        
        prices = [base_price]
        for i in range(1, n_samples):
            new_price = prices[-1] * (1 + returns[i] + trend[i]/n_samples)
            prices.append(new_price)
        
        prices = np.array(prices)
        
        # Criar OHLC
        high_noise = np.random.uniform(0.0001, 0.0005, n_samples)
        low_noise = np.random.uniform(-0.0005, -0.0001, n_samples)
        
        data = pd.DataFrame({
            'Open': prices,
            'High': prices + high_noise,
            'Low': prices + low_noise,
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, n_samples)
        })
        
        # Criar índice temporal
        start_date = datetime.now() - timedelta(minutes=n_samples)
        date_range = pd.date_range(start=start_date, periods=n_samples, freq='1min')
        data.index = date_range
        
        print(f"Dados simulados gerados: {len(data)} registros")
        return data
    
    def _standardize_csv_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Padroniza nomes das colunas CSV
        """
        column_mapping = {
            'timestamp': 'Datetime',
            'time': 'Datetime',
            'date': 'Datetime',
            'datetime': 'Datetime',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        df_renamed = df.rename(columns={col: column_mapping.get(col.lower(), col) for col in df.columns})
        
        # Adicionar Volume se não existir
        if 'Volume' not in df_renamed.columns:
            df_renamed['Volume'] = 1000
            
        return df_renamed
    
    def optimize_training_config(self, pairs: List[str]) -> Dict:
        """
        Otimiza configuração de treinamento baseada no número de pares
        """
        n_pairs = len(pairs)
        
        if n_pairs <= 2:
            config = {
                'samples_per_pair': 300000,
                'period': '2y',
                'description': 'Configuração robusta - máximos dados'
            }
        elif n_pairs <= 5:
            config = {
                'samples_per_pair': 200000,
                'period': '1y',
                'description': 'Configuração balanceada'
            }
        else:
            config = {
                'samples_per_pair': 100000,
                'period': '6mo',
                'description': 'Configuração otimizada para performance'
            }
            
        print(f"\n📊 Configuração selecionada: {config['description']}")
        print(f"   Amostras por par: {config['samples_per_pair']:,}")
        print(f"   Período: {config['period']}")
        
        return config
    
    def train_single_pair(self, pair: str, config: Dict) -> Optional[Dict]:
        """
        Treina modelo para um único par
        """
        print(f"\n{'='*60}")
        print(f"🎯 Treinando modelo para {pair}")
        print(f"{'='*60}")
        
        try:
            # Obter dados
            print("1️⃣ Obtendo dados...")
            data = self.get_forex_data_flexible(
                pair, 
                period=config['period'],
                samples=config['samples_per_pair']
            )
            
            if data is None or data.empty:
                print(f"❌ Dados insuficientes para {pair}")
                return None
                
            print(f"✅ Dados obtidos: {len(data):,} registros")
            
            # Substituir método temporariamente
            original_method = self.ml_predictor.get_forex_data
            self.ml_predictor.get_forex_data = lambda p, period: data
            
            # Criar features
            print("2️⃣ Criando features...")
            features = self.ml_predictor.create_enhanced_features(data)
            target = self.ml_predictor.create_target(data, prediction_horizon=1, threshold=0.001)
            
            print(f"✅ Features criadas: {len(self.ml_predictor.feature_names)}")
            
            # Limpar dados
            valid_idx = ~(features.isnull().any(axis=1) | target.isnull())
            features_clean = features[valid_idx]
            target_clean = target[valid_idx]
            
            if len(features_clean) < 100:
                print(f"❌ Dados insuficientes após limpeza: {len(features_clean)}")
                return None
                
            print(f"✅ Dados limpos: {len(features_clean):,} amostras válidas")
            
            # Walk-forward optimization
            print("3️⃣ Executando Walk-Forward Optimization...")
            wf_results = self.ml_predictor.walk_forward_optimization(features_clean, target_clean)
            
            # Treinar modelo final
            print("4️⃣ Treinando modelo final...")
            final_results = self.ml_predictor.train_final_model(features_clean, target_clean)
            
            # Restaurar método
            self.ml_predictor.get_forex_data = original_method
            
            # Compilar resultados
            results = {
                'pair': pair,
                'data_source': self.data_source,
                'data_samples': len(data),
                'valid_samples': len(features_clean),
                'features_count': len(self.ml_predictor.feature_names),
                'selected_features_count': len(self.ml_predictor.selected_features),
                'walk_forward_results': wf_results,
                'final_model_results': final_results,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Treinamento concluído para {pair}")
            if final_results:
                cv_score = final_results.get('cv_score_mean', 0)
                print(f"   📈 Acurácia CV: {cv_score:.3f}")
                print(f"   🎯 Features selecionadas: {len(self.ml_predictor.selected_features)}")
            
            return results
            
        except Exception as e:
            print(f"❌ Erro no treinamento de {pair}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_multiple_pairs(self, pairs: List[str]) -> Dict:
        """
        Treina modelos para múltiplos pares
        """
        print("🚀 Iniciando Treinamento de Modelos ML")
        print("="*70)
        print(f"📊 Fonte de dados: {self.data_source}")
        print(f"🎯 Pares selecionados: {pairs}")
        
        # Otimizar configuração
        config = self.optimize_training_config(pairs)
        
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
            'data_source': self.data_source,
            'configuration': config,
            'summary': {
                'total_pairs_attempted': len(pairs),
                'successful_trainings': successful_trainings,
                'failed_trainings': len(pairs) - successful_trainings,
                'success_rate': successful_trainings / len(pairs) if pairs else 0
            },
            'training_results': training_results,
            'ml_configuration': {
                'use_pca': self.ml_predictor.use_pca,
                'n_components': self.ml_predictor.n_components,
                'feature_selection_method': self.ml_predictor.feature_selection_method,
                'models': list(self.ml_predictor.base_models.keys())
            }
        }
        
        # Salvar relatório
        report_filename = f"ml_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
            
        print(f"\n🎉 Treinamento Concluído!")
        print(f"📈 Taxa de sucesso: {final_report['summary']['success_rate']:.1%}")
        print(f"✅ Modelos treinados: {successful_trainings}/{len(pairs)}")
        print(f"📄 Relatório salvo: {report_filename}")
        
        return final_report

def main():
    """
    Função principal interativa
    """
    print("🤖 Treinamento de Modelos ML para Forex")
    print("="*50)
    
    # Escolher fonte de dados
    print("\n📊 Fontes de dados disponíveis:")
    print("1. Yahoo Finance (recomendado)")
    print("2. Arquivos CSV locais")
    print("3. Dados simulados (para teste)")
    
    source_choice = input("\nEscolha a fonte de dados (1-3): ").strip()
    
    if source_choice == '1':
        data_source = "yahoo"
        data_directory = None
    elif source_choice == '2':
        data_source = "csv"
        data_directory = input("Digite o caminho do diretório CSV: ").strip()
        if not data_directory:
            data_directory = os.getenv("FX_DATA_DIR", "data/FX-1-Minute-Data")
    elif source_choice == '3':
        data_source = "simulated"
        data_directory = None
    else:
        print("❌ Opção inválida")
        return
    
    # Criar treinador
    trainer = FlexibleMLTrainer(data_source, data_directory)
    
    # Escolher pares
    print("\n🎯 Seleção de pares:")
    print("1. Pares principais (EUR/USD, GBP/USD)")
    print("2. Pares estendidos (5 pares principais)")
    print("3. Pares personalizados")
    
    pair_choice = input("\nEscolha os pares (1-3): ").strip()
    
    if pair_choice == '1':
        pairs = ['EURUSD', 'GBPUSD']
    elif pair_choice == '2':
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
    elif pair_choice == '3':
        custom_pairs = input("Digite os pares separados por vírgula (ex: EURUSD,GBPUSD): ").strip()
        pairs = [p.strip().upper() for p in custom_pairs.split(',') if p.strip()]
    else:
        print("❌ Opção inválida")
        return
    
    if not pairs:
        print("❌ Nenhum par selecionado")
        return
    
    # Executar treinamento
    print(f"\n🚀 Iniciando treinamento com {len(pairs)} pares...")
    result = trainer.train_multiple_pairs(pairs)
    
    # Mostrar resumo
    if result:
        print("\n📋 Resumo dos Resultados:")
        for pair, pair_result in result['training_results'].items():
            if 'error' not in pair_result:
                cv_score = pair_result.get('final_model_results', {}).get('cv_score_mean', 0)
                print(f"  {pair}: ✅ Acurácia {cv_score:.3f}")
            else:
                print(f"  {pair}: ❌ Falhou")

if __name__ == "__main__":
    main()