#!/usr/bin/env python3
"""
Data Optimizer para ML Forex
Analisa e otimiza a quantidade de dados para treinamento de modelos ML
Funciona com dados CSV locais e Yahoo Finance
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Importar módulos do projeto
try:
    from csv_data_loader import CSVForexDataLoader
    from ml_predictor_enhanced import MLForexPredictorEnhanced
    from config import ForexConfig
except ImportError as e:
    print(f"Aviso: Alguns módulos não puderam ser importados: {e}")

@dataclass
class DataConfiguration:
    """Configuração de dados otimizada"""
    samples_per_pair: int
    period: str
    description: str
    expected_performance: str
    memory_usage_mb: float
    training_time_estimate: str
    robustness_score: float
    performance_score: float

class DataOptimizer:
    """
    Otimizador de dados para treinamento ML
    Analisa diferentes configurações e recomenda a quantidade ideal
    """
    
    def __init__(self, data_source: str = 'yahoo', data_directory: str = None):
        """
        Inicializa o otimizador
        
        Args:
            data_source: 'yahoo', 'csv' ou 'both'
            data_directory: Diretório para dados CSV (se aplicável)
        """
        self.data_source = data_source
        self.data_directory = data_directory
        self.csv_loader = None
        self.ml_predictor = None
        
        # Configurações predefinidas
        self.configurations = self._create_configurations()
        
        # Inicializar carregadores se disponíveis
        if data_source in ['csv', 'both'] and data_directory:
            try:
                self.csv_loader = CSVForexDataLoader(data_directory)
            except:
                print("Aviso: CSV Loader não disponível")
                
        try:
            self.ml_predictor = MLForexPredictorEnhanced()
        except:
            print("Aviso: ML Predictor não disponível")
    
    def _create_configurations(self) -> Dict[str, DataConfiguration]:
        """
        Cria configurações predefinidas baseadas em diferentes cenários
        """
        configs = {
            'minimal': DataConfiguration(
                samples_per_pair=50000,   # ~35 dias de dados 1min
                period='1mo',
                description='Configuração mínima - teste rápido',
                expected_performance='Baixa',
                memory_usage_mb=100,
                training_time_estimate='1-2 minutos',
                robustness_score=2.0,
                performance_score=9.0
            ),
            'light': DataConfiguration(
                samples_per_pair=100000,  # ~70 dias
                period='3mo', 
                description='Configuração leve - desenvolvimento',
                expected_performance='Moderada',
                memory_usage_mb=200,
                training_time_estimate='3-5 minutos',
                robustness_score=4.0,
                performance_score=8.0
            ),
            'balanced': DataConfiguration(
                samples_per_pair=200000,  # ~140 dias
                period='6mo',
                description='Configuração balanceada - produção padrão',
                expected_performance='Boa',
                memory_usage_mb=400,
                training_time_estimate='5-10 minutos',
                robustness_score=6.0,
                performance_score=6.0
            ),
            'robust': DataConfiguration(
                samples_per_pair=300000,  # ~210 dias
                period='1y',
                description='Configuração robusta - alta qualidade',
                expected_performance='Alta',
                memory_usage_mb=600,
                training_time_estimate='10-15 minutos',
                robustness_score=8.0,
                performance_score=4.0
            ),
            'maximum': DataConfiguration(
                samples_per_pair=500000,  # ~350 dias
                period='2y',
                description='Configuração máxima - pesquisa/backtesting',
                expected_performance='Máxima',
                memory_usage_mb=1000,
                training_time_estimate='15-30 minutos',
                robustness_score=10.0,
                performance_score=2.0
            )
        }
        return configs
    
    def analyze_available_data(self) -> Dict:
        """
        Analisa dados disponíveis em diferentes fontes
        """
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'data_sources': {},
            'recommendations': {}
        }
        
        # Analisar dados CSV se disponível
        if self.csv_loader:
            csv_analysis = self._analyze_csv_data()
            analysis['data_sources']['csv'] = csv_analysis
            
        # Analisar dados Yahoo Finance
        if self.data_source in ['yahoo', 'both']:
            yahoo_analysis = self._analyze_yahoo_data()
            analysis['data_sources']['yahoo'] = yahoo_analysis
            
        return analysis
    
    def _analyze_csv_data(self) -> Dict:
        """
        Analisa dados CSV disponíveis
        """
        if not self.csv_loader:
            return {'error': 'CSV Loader não disponível'}
            
        try:
            pairs = self.csv_loader.available_pairs
            analysis = {
                'available_pairs': pairs,
                'total_pairs': len(pairs),
                'data_directory': self.data_directory,
                'pair_details': {}
            }
            
            # Analisar alguns pares em detalhes
            for pair in pairs[:5]:  # Limitar para performance
                try:
                    sample_data = self.csv_loader.load_forex_data(pair, sample_size=1000)
                    if sample_data is not None:
                        analysis['pair_details'][pair] = {
                            'sample_size': len(sample_data),
                            'date_range': f"{sample_data.index[0]} to {sample_data.index[-1]}",
                            'data_quality': self._assess_data_quality(sample_data)
                        }
                except Exception as e:
                    analysis['pair_details'][pair] = {'error': str(e)}
                    
            return analysis
            
        except Exception as e:
            return {'error': f'Erro na análise CSV: {e}'}
    
    def _analyze_yahoo_data(self) -> Dict:
        """
        Analisa disponibilidade de dados do Yahoo Finance
        """
        try:
            # Pares comuns para teste
            test_pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X']
            
            analysis = {
                'available_pairs': test_pairs,
                'total_pairs': len(test_pairs),
                'data_source': 'Yahoo Finance',
                'pair_details': {}
            }
            
            if self.ml_predictor:
                for pair in test_pairs[:3]:  # Testar alguns pares
                    try:
                        sample_data = self.ml_predictor.get_forex_data(pair, period='1mo')
                        if sample_data is not None:
                            analysis['pair_details'][pair] = {
                                'sample_size': len(sample_data),
                                'date_range': f"{sample_data.index[0]} to {sample_data.index[-1]}",
                                'data_quality': self._assess_data_quality(sample_data)
                            }
                    except Exception as e:
                        analysis['pair_details'][pair] = {'error': str(e)}
                        
            return analysis
            
        except Exception as e:
            return {'error': f'Erro na análise Yahoo: {e}'}
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict:
        """
        Avalia qualidade dos dados
        """
        try:
            quality = {
                'completeness': (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
                'consistency': self._check_data_consistency(data),
                'coverage': len(data),
                'timespan_days': (data.index[-1] - data.index[0]).days if len(data) > 1 else 0
            }
            
            # Score geral de qualidade
            quality['overall_score'] = (
                quality['completeness'] * 0.4 +
                quality['consistency'] * 0.3 +
                min(quality['coverage'] / 10000, 1) * 100 * 0.3
            )
            
            return quality
            
        except Exception as e:
            return {'error': f'Erro na avaliação: {e}'}
    
    def _check_data_consistency(self, data: pd.DataFrame) -> float:
        """
        Verifica consistência dos dados OHLC
        """
        try:
            if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                return 50.0  # Score médio se não temos OHLC completo
                
            # Verificar se High >= max(Open, Close) e Low <= min(Open, Close)
            high_valid = (data['High'] >= data[['Open', 'Close']].max(axis=1)).mean() * 100
            low_valid = (data['Low'] <= data[['Open', 'Close']].min(axis=1)).mean() * 100
            
            return (high_valid + low_valid) / 2
            
        except:
            return 50.0
    
    def recommend_configuration(self, 
                              num_pairs: int, 
                              available_memory_gb: float = 4.0,
                              max_training_time_minutes: int = 15,
                              priority: str = 'balanced') -> Dict:
        """
        Recomenda configuração ideal baseada em restrições
        
        Args:
            num_pairs: Número de pares para treinar
            available_memory_gb: Memória disponível em GB
            max_training_time_minutes: Tempo máximo de treinamento
            priority: 'performance', 'balanced', 'robustness'
            
        Returns:
            Configuração recomendada com justificativa
        """
        recommendations = []
        
        for name, config in self.configurations.items():
            # Calcular recursos necessários
            total_memory_mb = config.memory_usage_mb * num_pairs
            estimated_time = self._estimate_training_time(config, num_pairs)
            
            # Verificar se atende restrições
            memory_ok = total_memory_mb <= (available_memory_gb * 1024 * 0.8)  # 80% da memória
            time_ok = estimated_time <= max_training_time_minutes
            
            if memory_ok and time_ok:
                # Calcular score baseado na prioridade
                if priority == 'performance':
                    score = config.performance_score
                elif priority == 'robustness':
                    score = config.robustness_score
                else:  # balanced
                    score = (config.performance_score + config.robustness_score) / 2
                    
                recommendations.append({
                    'name': name,
                    'config': config,
                    'score': score,
                    'total_memory_mb': total_memory_mb,
                    'estimated_time_minutes': estimated_time,
                    'fits_constraints': True
                })
            else:
                recommendations.append({
                    'name': name,
                    'config': config,
                    'score': 0,
                    'total_memory_mb': total_memory_mb,
                    'estimated_time_minutes': estimated_time,
                    'fits_constraints': False,
                    'issues': {
                        'memory': not memory_ok,
                        'time': not time_ok
                    }
                })
        
        # Ordenar por score (maior primeiro)
        valid_recommendations = [r for r in recommendations if r['fits_constraints']]
        valid_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'input_parameters': {
                'num_pairs': num_pairs,
                'available_memory_gb': available_memory_gb,
                'max_training_time_minutes': max_training_time_minutes,
                'priority': priority
            },
            'recommended_config': valid_recommendations[0] if valid_recommendations else None,
            'all_options': recommendations,
            'analysis': self._generate_recommendation_analysis(recommendations, num_pairs)
        }
        
        return result
    
    def _estimate_training_time(self, config: DataConfiguration, num_pairs: int) -> float:
        """
        Estima tempo de treinamento baseado na configuração
        """
        # Tempo base por par (em minutos)
        base_times = {
            50000: 1.5,   # minimal
            100000: 3.0,  # light
            200000: 6.0,  # balanced
            300000: 10.0, # robust
            500000: 20.0  # maximum
        }
        
        base_time = base_times.get(config.samples_per_pair, 10.0)
        
        # Fator de escala para múltiplos pares (não linear)
        if num_pairs <= 1:
            scale_factor = 1.0
        elif num_pairs <= 3:
            scale_factor = 0.9
        elif num_pairs <= 5:
            scale_factor = 0.8
        else:
            scale_factor = 0.7
            
        return base_time * num_pairs * scale_factor
    
    def _generate_recommendation_analysis(self, recommendations: List[Dict], num_pairs: int) -> Dict:
        """
        Gera análise das recomendações
        """
        valid_configs = [r for r in recommendations if r['fits_constraints']]
        
        analysis = {
            'total_configurations': len(recommendations),
            'valid_configurations': len(valid_configs),
            'constraints_summary': {
                'memory_issues': len([r for r in recommendations if not r['fits_constraints'] and r.get('issues', {}).get('memory', False)]),
                'time_issues': len([r for r in recommendations if not r['fits_constraints'] and r.get('issues', {}).get('time', False)])
            }
        }
        
        if valid_configs:
            best_config = valid_configs[0]
            analysis['recommendation_summary'] = {
                'best_option': best_config['name'],
                'expected_memory_usage_gb': best_config['total_memory_mb'] / 1024,
                'expected_training_time_minutes': best_config['estimated_time_minutes'],
                'samples_per_pair': best_config['config'].samples_per_pair,
                'total_samples': best_config['config'].samples_per_pair * num_pairs,
                'robustness_level': best_config['config'].robustness_score / 10,
                'performance_level': best_config['config'].performance_score / 10
            }
        else:
            analysis['recommendation_summary'] = {
                'warning': 'Nenhuma configuração atende às restrições especificadas',
                'suggestion': 'Considere aumentar a memória disponível ou tempo de treinamento'
            }
            
        return analysis
    
    def generate_optimization_report(self, 
                                   num_pairs: int = 5,
                                   scenarios: List[str] = None) -> Dict:
        """
        Gera relatório completo de otimização
        """
        if scenarios is None:
            scenarios = ['performance', 'balanced', 'robustness']
            
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_analysis': self.analyze_available_data(),
            'optimization_scenarios': {},
            'configuration_comparison': self._compare_configurations(),
            'recommendations_summary': {}
        }
        
        # Gerar recomendações para diferentes cenários
        for scenario in scenarios:
            recommendation = self.recommend_configuration(
                num_pairs=num_pairs,
                priority=scenario
            )
            report['optimization_scenarios'][scenario] = recommendation
            
        # Resumo das recomendações
        report['recommendations_summary'] = self._summarize_recommendations(
            report['optimization_scenarios']
        )
        
        return report
    
    def _compare_configurations(self) -> Dict:
        """
        Compara todas as configurações disponíveis
        """
        comparison = {
            'configurations': {},
            'trade_offs': {
                'performance_vs_robustness': [],
                'memory_vs_quality': [],
                'time_vs_accuracy': []
            }
        }
        
        for name, config in self.configurations.items():
            comparison['configurations'][name] = {
                'samples': config.samples_per_pair,
                'period': config.period,
                'memory_mb': config.memory_usage_mb,
                'robustness_score': config.robustness_score,
                'performance_score': config.performance_score,
                'description': config.description
            }
            
            # Trade-offs
            comparison['trade_offs']['performance_vs_robustness'].append({
                'name': name,
                'performance': config.performance_score,
                'robustness': config.robustness_score
            })
            
        return comparison
    
    def _summarize_recommendations(self, scenarios: Dict) -> Dict:
        """
        Sumariza recomendações de diferentes cenários
        """
        summary = {
            'best_for_performance': None,
            'best_for_robustness': None,
            'best_balanced': None,
            'resource_requirements': {}
        }
        
        for scenario_name, scenario_data in scenarios.items():
            if scenario_data.get('recommended_config'):
                config_name = scenario_data['recommended_config']['name']
                
                if scenario_name == 'performance':
                    summary['best_for_performance'] = config_name
                elif scenario_name == 'robustness':
                    summary['best_for_robustness'] = config_name
                elif scenario_name == 'balanced':
                    summary['best_balanced'] = config_name
                    
                # Recursos necessários
                summary['resource_requirements'][scenario_name] = {
                    'memory_gb': scenario_data['recommended_config']['total_memory_mb'] / 1024,
                    'time_minutes': scenario_data['recommended_config']['estimated_time_minutes']
                }
                
        return summary
    
    def save_report(self, report: Dict, filename: str = None) -> str:
        """
        Salva relatório em arquivo JSON
        """
        if filename is None:
            filename = f"data_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
        return filename

def main():
    """
    Função principal para demonstração
    """
    print("🔍 Data Optimizer para ML Forex")
    print("="*50)
    
    # Configurações de exemplo
    scenarios = [
        {'pairs': 1, 'memory': 2.0, 'time': 5, 'priority': 'performance'},
        {'pairs': 3, 'memory': 4.0, 'time': 10, 'priority': 'balanced'},
        {'pairs': 5, 'memory': 8.0, 'time': 20, 'priority': 'robustness'}
    ]
    
    # Criar otimizador
    optimizer = DataOptimizer(data_source='yahoo')
    
    print("\n📊 Analisando cenários de otimização...\n")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"Cenário {i}: {scenario['pairs']} pares, {scenario['memory']}GB RAM, {scenario['time']}min")
        
        recommendation = optimizer.recommend_configuration(
            num_pairs=scenario['pairs'],
            available_memory_gb=scenario['memory'],
            max_training_time_minutes=scenario['time'],
            priority=scenario['priority']
        )
        
        if recommendation['recommended_config']:
            config = recommendation['recommended_config']
            print(f"  ✅ Recomendação: {config['name']}")
            print(f"     Amostras por par: {config['config'].samples_per_pair:,}")
            print(f"     Memória estimada: {config['total_memory_mb']:.0f}MB")
            print(f"     Tempo estimado: {config['estimated_time_minutes']:.1f}min")
        else:
            print(f"  ❌ Nenhuma configuração atende às restrições")
        print()
    
    # Gerar relatório completo
    print("📋 Gerando relatório completo de otimização...")
    report = optimizer.generate_optimization_report(num_pairs=5)
    
    # Salvar relatório
    filename = optimizer.save_report(report)
    print(f"✅ Relatório salvo: {filename}")
    
    # Mostrar resumo
    if 'recommendations_summary' in report:
        summary = report['recommendations_summary']
        print("\n🎯 Resumo das Recomendações:")
        print(f"  Melhor para Performance: {summary.get('best_for_performance', 'N/A')}")
        print(f"  Melhor Balanceado: {summary.get('best_balanced', 'N/A')}")
        print(f"  Melhor para Robustez: {summary.get('best_for_robustness', 'N/A')}")

if __name__ == "__main__":
    main()