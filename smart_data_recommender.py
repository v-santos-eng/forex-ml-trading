#!/usr/bin/env python3
"""
Smart Data Recommender para ML Forex
Sistema inteligente de recomendação de quantidade ideal de dados
Baseado em recursos disponíveis, objetivos e análise de performance
"""

import pandas as pd
import numpy as np
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SystemResources:
    """Recursos do sistema disponíveis"""
    total_memory_gb: float
    available_memory_gb: float
    cpu_cores: int
    cpu_usage_percent: float
    disk_space_gb: float

@dataclass
class TrainingObjective:
    """Objetivos do treinamento"""
    purpose: str  # 'research', 'production', 'testing', 'backtesting'
    accuracy_priority: float  # 0-10
    speed_priority: float     # 0-10
    robustness_priority: float # 0-10
    max_training_time_hours: float
    target_pairs: List[str]

@dataclass
class DataRecommendation:
    """Recomendação de dados"""
    samples_per_pair: int
    total_samples: int
    period_description: str
    estimated_memory_gb: float
    estimated_time_hours: float
    confidence_score: float
    robustness_level: str
    performance_level: str
    justification: List[str]
    warnings: List[str]
    alternatives: List[Dict]

class SmartDataRecommender:
    """
    Recomendador inteligente de quantidade de dados
    Usa análise de recursos, objetivos e heurísticas de ML
    """
    
    def __init__(self):
        self.system_resources = self._analyze_system_resources()
        self.data_efficiency_curves = self._load_efficiency_curves()
        self.memory_overhead_factor = 1.5  # Fator de segurança para memória
        
    def _analyze_system_resources(self) -> SystemResources:
        """
        Analisa recursos disponíveis do sistema
        """
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            return SystemResources(
                total_memory_gb=memory.total / (1024**3),
                available_memory_gb=memory.available / (1024**3),
                cpu_cores=psutil.cpu_count(),
                cpu_usage_percent=psutil.cpu_percent(interval=1),
                disk_space_gb=disk.free / (1024**3)
            )
        except Exception as e:
            print(f"Aviso: Não foi possível analisar recursos do sistema: {e}")
            # Valores padrão conservadores
            return SystemResources(
                total_memory_gb=4.0,
                available_memory_gb=2.0,
                cpu_cores=4,
                cpu_usage_percent=50.0,
                disk_space_gb=10.0
            )
    
    def _load_efficiency_curves(self) -> Dict:
        """
        Carrega curvas de eficiência baseadas em experiência empírica
        """
        return {
            # Relação samples -> accuracy improvement (diminishing returns)
            'accuracy_curve': {
                10000: 0.45,   # Baseline muito baixo
                25000: 0.55,   # Melhoria significativa
                50000: 0.65,   # Boa performance
                100000: 0.72,  # Performance sólida
                200000: 0.78,  # Alta performance
                300000: 0.82,  # Performance muito alta
                500000: 0.85,  # Performance máxima prática
                1000000: 0.87  # Retornos diminuindo
            },
            # Relação samples -> training time (aproximadamente linear)
            'time_curve': {
                10000: 0.5,    # 30 segundos
                25000: 1.0,    # 1 minuto
                50000: 2.0,    # 2 minutos
                100000: 4.0,   # 4 minutos
                200000: 8.0,   # 8 minutos
                300000: 12.0,  # 12 minutos
                500000: 20.0,  # 20 minutos
                1000000: 40.0  # 40 minutos
            },
            # Relação samples -> memory usage (MB)
            'memory_curve': {
                10000: 50,
                25000: 100,
                50000: 180,
                100000: 320,
                200000: 600,
                300000: 850,
                500000: 1300,
                1000000: 2500
            }
        }
    
    def _interpolate_curve(self, curve: Dict, target_samples: int) -> float:
        """
        Interpola valores nas curvas de eficiência
        """
        samples = sorted(curve.keys())
        
        if target_samples <= samples[0]:
            return curve[samples[0]]
        if target_samples >= samples[-1]:
            return curve[samples[-1]]
            
        # Interpolação linear
        for i in range(len(samples) - 1):
            if samples[i] <= target_samples <= samples[i + 1]:
                x1, y1 = samples[i], curve[samples[i]]
                x2, y2 = samples[i + 1], curve[samples[i + 1]]
                
                # Interpolação linear
                ratio = (target_samples - x1) / (x2 - x1)
                return y1 + ratio * (y2 - y1)
                
        return curve[samples[-1]]
    
    def recommend_data_amount(self, 
                            objective: TrainingObjective,
                            available_data_samples: Optional[int] = None) -> DataRecommendation:
        """
        Recomenda quantidade ideal de dados baseado em objetivos e recursos
        
        Args:
            objective: Objetivos do treinamento
            available_data_samples: Amostras disponíveis (None = ilimitado)
            
        Returns:
            Recomendação detalhada
        """
        num_pairs = len(objective.target_pairs)
        
        # Calcular score de prioridade ponderado
        priority_weights = {
            'accuracy': objective.accuracy_priority / 10,
            'speed': objective.speed_priority / 10,
            'robustness': objective.robustness_priority / 10
        }
        
        # Normalizar pesos
        total_weight = sum(priority_weights.values())
        if total_weight > 0:
            priority_weights = {k: v/total_weight for k, v in priority_weights.items()}
        
        # Gerar candidatos de configuração
        candidates = self._generate_candidates(objective, num_pairs)
        
        # Avaliar cada candidato
        scored_candidates = []
        for candidate in candidates:
            score = self._score_candidate(candidate, objective, priority_weights)
            scored_candidates.append((score, candidate))
            
        # Ordenar por score (maior primeiro)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Selecionar melhor candidato que atende restrições
        best_candidate = None
        for score, candidate in scored_candidates:
            if self._meets_constraints(candidate, objective, num_pairs):
                best_candidate = candidate
                break
                
        if best_candidate is None:
            # Se nenhum candidato atende, usar o com melhor score
            best_candidate = scored_candidates[0][1] if scored_candidates else self._get_fallback_candidate()
            
        # Aplicar limitações de dados disponíveis
        if available_data_samples:
            best_candidate = min(best_candidate, available_data_samples)
            
        # Gerar recomendação detalhada
        return self._create_detailed_recommendation(
            best_candidate, objective, num_pairs, priority_weights
        )
    
    def _generate_candidates(self, objective: TrainingObjective, num_pairs: int) -> List[int]:
        """
        Gera candidatos de quantidade de dados baseado no propósito
        """
        purpose_configs = {
            'testing': [10000, 25000, 50000],
            'research': [100000, 200000, 300000, 500000],
            'production': [200000, 300000, 500000],
            'backtesting': [300000, 500000, 1000000]
        }
        
        base_candidates = purpose_configs.get(objective.purpose, [50000, 100000, 200000, 300000])
        
        # Ajustar baseado no número de pares
        if num_pairs > 5:
            # Muitos pares: reduzir amostras por par
            base_candidates = [int(c * 0.7) for c in base_candidates]
        elif num_pairs <= 2:
            # Poucos pares: pode usar mais amostras
            base_candidates = [int(c * 1.3) for c in base_candidates]
            
        # Adicionar candidatos baseados em recursos
        memory_limited_candidates = self._get_memory_limited_candidates(num_pairs)
        time_limited_candidates = self._get_time_limited_candidates(objective.max_training_time_hours, num_pairs)
        
        # Combinar e remover duplicatas
        all_candidates = list(set(base_candidates + memory_limited_candidates + time_limited_candidates))
        
        return sorted([c for c in all_candidates if c >= 5000])  # Mínimo de 5k amostras
    
    def _get_memory_limited_candidates(self, num_pairs: int) -> List[int]:
        """
        Gera candidatos baseados em limitações de memória
        """
        available_memory_mb = self.system_resources.available_memory_gb * 1024 * 0.7  # 70% da memória
        memory_per_pair = available_memory_mb / num_pairs / self.memory_overhead_factor
        
        candidates = []
        for samples in [25000, 50000, 100000, 200000, 300000, 500000]:
            required_memory = self._interpolate_curve(self.data_efficiency_curves['memory_curve'], samples)
            if required_memory <= memory_per_pair:
                candidates.append(samples)
                
        return candidates
    
    def _get_time_limited_candidates(self, max_hours: float, num_pairs: int) -> List[int]:
        """
        Gera candidatos baseados em limitações de tempo
        """
        max_minutes_per_pair = (max_hours * 60) / num_pairs
        
        candidates = []
        for samples in [25000, 50000, 100000, 200000, 300000, 500000]:
            required_minutes = self._interpolate_curve(self.data_efficiency_curves['time_curve'], samples)
            if required_minutes <= max_minutes_per_pair:
                candidates.append(samples)
                
        return candidates
    
    def _score_candidate(self, samples: int, objective: TrainingObjective, weights: Dict) -> float:
        """
        Calcula score para um candidato de quantidade de dados
        """
        # Scores individuais (0-1)
        accuracy_score = self._interpolate_curve(self.data_efficiency_curves['accuracy_curve'], samples)
        
        # Speed score (inverso do tempo)
        time_minutes = self._interpolate_curve(self.data_efficiency_curves['time_curve'], samples)
        speed_score = max(0, 1 - (time_minutes / 60))  # Normalizar para 1 hora = 0
        
        # Robustness score (similar ao accuracy, mas com curva diferente)
        robustness_score = min(1.0, accuracy_score * 1.1)  # Ligeiramente melhor que accuracy
        
        # Score ponderado
        total_score = (
            weights['accuracy'] * accuracy_score +
            weights['speed'] * speed_score +
            weights['robustness'] * robustness_score
        )
        
        return total_score
    
    def _meets_constraints(self, samples: int, objective: TrainingObjective, num_pairs: int) -> bool:
        """
        Verifica se candidato atende às restrições
        """
        # Verificar memória
        required_memory_mb = self._interpolate_curve(self.data_efficiency_curves['memory_curve'], samples)
        total_memory_mb = required_memory_mb * num_pairs * self.memory_overhead_factor
        available_memory_mb = self.system_resources.available_memory_gb * 1024
        
        if total_memory_mb > available_memory_mb:
            return False
            
        # Verificar tempo
        required_time_minutes = self._interpolate_curve(self.data_efficiency_curves['time_curve'], samples)
        total_time_hours = (required_time_minutes * num_pairs) / 60
        
        if total_time_hours > objective.max_training_time_hours:
            return False
            
        return True
    
    def _get_fallback_candidate(self) -> int:
        """
        Retorna candidato de fallback conservador
        """
        return 50000  # 50k amostras como padrão seguro
    
    def _create_detailed_recommendation(self, 
                                      samples: int, 
                                      objective: TrainingObjective, 
                                      num_pairs: int,
                                      weights: Dict) -> DataRecommendation:
        """
        Cria recomendação detalhada
        """
        # Calcular métricas
        estimated_accuracy = self._interpolate_curve(self.data_efficiency_curves['accuracy_curve'], samples)
        estimated_time_minutes = self._interpolate_curve(self.data_efficiency_curves['time_curve'], samples)
        estimated_memory_mb = self._interpolate_curve(self.data_efficiency_curves['memory_curve'], samples)
        
        total_samples = samples * num_pairs
        total_time_hours = (estimated_time_minutes * num_pairs) / 60
        total_memory_gb = (estimated_memory_mb * num_pairs * self.memory_overhead_factor) / 1024
        
        # Determinar níveis
        robustness_level = self._get_level_description(estimated_accuracy, 'robustness')
        performance_level = self._get_level_description(1 - (estimated_time_minutes / 60), 'performance')
        
        # Calcular confiança
        confidence_score = self._calculate_confidence(samples, objective, num_pairs)
        
        # Gerar justificativas
        justification = self._generate_justification(samples, objective, weights, estimated_accuracy)
        
        # Gerar avisos
        warnings = self._generate_warnings(samples, objective, num_pairs, total_memory_gb, total_time_hours)
        
        # Gerar alternativas
        alternatives = self._generate_alternatives(samples, objective, num_pairs)
        
        # Descrição do período
        period_description = self._get_period_description(samples)
        
        return DataRecommendation(
            samples_per_pair=samples,
            total_samples=total_samples,
            period_description=period_description,
            estimated_memory_gb=total_memory_gb,
            estimated_time_hours=total_time_hours,
            confidence_score=confidence_score,
            robustness_level=robustness_level,
            performance_level=performance_level,
            justification=justification,
            warnings=warnings,
            alternatives=alternatives
        )
    
    def _get_level_description(self, score: float, metric_type: str) -> str:
        """
        Converte score numérico em descrição qualitativa
        """
        if score >= 0.85:
            return "Excelente"
        elif score >= 0.75:
            return "Alto"
        elif score >= 0.65:
            return "Bom"
        elif score >= 0.55:
            return "Moderado"
        else:
            return "Baixo"
    
    def _calculate_confidence(self, samples: int, objective: TrainingObjective, num_pairs: int) -> float:
        """
        Calcula confiança na recomendação
        """
        confidence_factors = []
        
        # Fator 1: Adequação aos recursos
        memory_usage_ratio = self._get_memory_usage_ratio(samples, num_pairs)
        if memory_usage_ratio < 0.8:
            confidence_factors.append(0.9)
        elif memory_usage_ratio < 0.95:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
            
        # Fator 2: Adequação ao tempo
        time_minutes = self._interpolate_curve(self.data_efficiency_curves['time_curve'], samples)
        total_time_hours = (time_minutes * num_pairs) / 60
        time_ratio = total_time_hours / objective.max_training_time_hours
        
        if time_ratio < 0.7:
            confidence_factors.append(0.9)
        elif time_ratio < 0.9:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
            
        # Fator 3: Adequação ao propósito
        purpose_adequacy = self._assess_purpose_adequacy(samples, objective.purpose)
        confidence_factors.append(purpose_adequacy)
        
        return np.mean(confidence_factors)
    
    def _get_memory_usage_ratio(self, samples: int, num_pairs: int) -> float:
        """
        Calcula ratio de uso de memória
        """
        required_memory_mb = self._interpolate_curve(self.data_efficiency_curves['memory_curve'], samples)
        total_memory_mb = required_memory_mb * num_pairs * self.memory_overhead_factor
        available_memory_mb = self.system_resources.available_memory_gb * 1024
        
        return total_memory_mb / available_memory_mb
    
    def _assess_purpose_adequacy(self, samples: int, purpose: str) -> float:
        """
        Avalia adequação da quantidade de dados ao propósito
        """
        adequacy_ranges = {
            'testing': (10000, 100000),
            'research': (100000, 500000),
            'production': (200000, 500000),
            'backtesting': (300000, 1000000)
        }
        
        min_samples, max_samples = adequacy_ranges.get(purpose, (50000, 300000))
        
        if min_samples <= samples <= max_samples:
            return 0.9
        elif samples < min_samples:
            return 0.6
        else:  # samples > max_samples
            return 0.7
    
    def _generate_justification(self, samples: int, objective: TrainingObjective, weights: Dict, accuracy: float) -> List[str]:
        """
        Gera justificativas para a recomendação
        """
        justification = []
        
        # Justificativa baseada no propósito
        purpose_justifications = {
            'testing': f"Para testes e desenvolvimento, {samples:,} amostras oferecem um bom equilíbrio entre velocidade e qualidade.",
            'research': f"Para pesquisa, {samples:,} amostras proporcionam robustez estatística adequada para análises confiáveis.",
            'production': f"Para produção, {samples:,} amostras garantem performance estável e confiável em ambiente real.",
            'backtesting': f"Para backtesting, {samples:,} amostras oferecem histórico suficiente para validação robusta."
        }
        
        justification.append(purpose_justifications.get(
            objective.purpose, 
            f"Para o propósito especificado, {samples:,} amostras oferecem um bom equilíbrio."
        ))
        
        # Justificativa baseada em prioridades
        if weights['accuracy'] > 0.4:
            justification.append(f"Prioridade em acurácia: esperada performance de {accuracy:.1%} com esta quantidade de dados.")
            
        if weights['speed'] > 0.4:
            time_minutes = self._interpolate_curve(self.data_efficiency_curves['time_curve'], samples)
            justification.append(f"Prioridade em velocidade: treinamento estimado em {time_minutes:.1f} minutos por par.")
            
        if weights['robustness'] > 0.4:
            justification.append(f"Prioridade em robustez: {samples:,} amostras proporcionam estabilidade estatística adequada.")
            
        return justification
    
    def _generate_warnings(self, samples: int, objective: TrainingObjective, num_pairs: int, memory_gb: float, time_hours: float) -> List[str]:
        """
        Gera avisos sobre a recomendação
        """
        warnings = []
        
        # Avisos de memória
        if memory_gb > self.system_resources.available_memory_gb * 0.9:
            warnings.append(f"Alto uso de memória: {memory_gb:.1f}GB de {self.system_resources.available_memory_gb:.1f}GB disponíveis.")
            
        # Avisos de tempo
        if time_hours > objective.max_training_time_hours * 0.9:
            warnings.append(f"Tempo próximo ao limite: {time_hours:.1f}h de {objective.max_training_time_hours:.1f}h máximo.")
            
        # Avisos de quantidade
        if samples < 25000:
            warnings.append("Quantidade de dados pode ser insuficiente para modelos robustos.")
        elif samples > 500000:
            warnings.append("Quantidade de dados muito alta pode ter retornos diminuídos.")
            
        # Avisos de propósito
        if objective.purpose == 'production' and samples < 100000:
            warnings.append("Para produção, considere usar mais dados para maior robustez.")
            
        return warnings
    
    def _generate_alternatives(self, samples: int, objective: TrainingObjective, num_pairs: int) -> List[Dict]:
        """
        Gera alternativas à recomendação principal
        """
        alternatives = []
        
        # Alternativa mais rápida
        fast_samples = int(samples * 0.6)
        if fast_samples >= 10000:
            fast_time = self._interpolate_curve(self.data_efficiency_curves['time_curve'], fast_samples)
            alternatives.append({
                'type': 'Mais Rápida',
                'samples_per_pair': fast_samples,
                'description': f"Reduz tempo de treinamento para ~{fast_time:.1f}min por par",
                'trade_off': 'Menor robustez'
            })
            
        # Alternativa mais robusta
        robust_samples = int(samples * 1.5)
        robust_memory = self._interpolate_curve(self.data_efficiency_curves['memory_curve'], robust_samples)
        total_robust_memory = (robust_memory * num_pairs * self.memory_overhead_factor) / 1024
        
        if total_robust_memory <= self.system_resources.available_memory_gb:
            alternatives.append({
                'type': 'Mais Robusta',
                'samples_per_pair': robust_samples,
                'description': f"Aumenta robustez com {robust_samples:,} amostras por par",
                'trade_off': 'Maior tempo de treinamento'
            })
            
        return alternatives
    
    def _get_period_description(self, samples: int) -> str:
        """
        Converte amostras em descrição de período (assumindo dados de 1 minuto)
        """
        days = samples / (24 * 60)  # 1440 minutos por dia
        
        if days < 30:
            return f"~{days:.0f} dias de dados de 1 minuto"
        elif days < 365:
            months = days / 30
            return f"~{months:.1f} meses de dados de 1 minuto"
        else:
            years = days / 365
            return f"~{years:.1f} anos de dados de 1 minuto"
    
    def analyze_data_efficiency(self, sample_ranges: List[int] = None) -> Dict:
        """
        Analisa eficiência de diferentes quantidades de dados
        """
        if sample_ranges is None:
            sample_ranges = [25000, 50000, 100000, 200000, 300000, 500000]
            
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'system_resources': asdict(self.system_resources),
            'efficiency_analysis': []
        }
        
        for samples in sample_ranges:
            accuracy = self._interpolate_curve(self.data_efficiency_curves['accuracy_curve'], samples)
            time_minutes = self._interpolate_curve(self.data_efficiency_curves['time_curve'], samples)
            memory_mb = self._interpolate_curve(self.data_efficiency_curves['memory_curve'], samples)
            
            # Calcular eficiência (accuracy per minute)
            efficiency = accuracy / time_minutes if time_minutes > 0 else 0
            
            analysis['efficiency_analysis'].append({
                'samples': samples,
                'period': self._get_period_description(samples),
                'expected_accuracy': accuracy,
                'training_time_minutes': time_minutes,
                'memory_usage_mb': memory_mb,
                'efficiency_score': efficiency,
                'accuracy_per_gb': accuracy / (memory_mb / 1024) if memory_mb > 0 else 0
            })
            
        return analysis
    
    def save_recommendation(self, recommendation: DataRecommendation, filename: str = None) -> str:
        """
        Salva recomendação em arquivo JSON
        """
        if filename is None:
            filename = f"data_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        recommendation_dict = asdict(recommendation)
        recommendation_dict['timestamp'] = datetime.now().isoformat()
        recommendation_dict['system_resources'] = asdict(self.system_resources)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(recommendation_dict, f, indent=2, ensure_ascii=False)
            
        return filename

def create_training_objective(purpose: str = 'production',
                            accuracy_priority: float = 7.0,
                            speed_priority: float = 5.0,
                            robustness_priority: float = 8.0,
                            max_training_time_hours: float = 2.0,
                            target_pairs: List[str] = None) -> TrainingObjective:
    """
    Função de conveniência para criar objetivos de treinamento
    """
    if target_pairs is None:
        target_pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
        
    return TrainingObjective(
        purpose=purpose,
        accuracy_priority=accuracy_priority,
        speed_priority=speed_priority,
        robustness_priority=robustness_priority,
        max_training_time_hours=max_training_time_hours,
        target_pairs=target_pairs
    )

def main():
    """
    Demonstração do Smart Data Recommender
    """
    print("🧠 Smart Data Recommender para ML Forex")
    print("="*50)
    
    # Criar recomendador
    recommender = SmartDataRecommender()
    
    # Mostrar recursos do sistema
    resources = recommender.system_resources
    print(f"\n💻 Recursos do Sistema:")
    print(f"  Memória Total: {resources.total_memory_gb:.1f}GB")
    print(f"  Memória Disponível: {resources.available_memory_gb:.1f}GB")
    print(f"  CPU Cores: {resources.cpu_cores}")
    print(f"  Uso CPU: {resources.cpu_usage_percent:.1f}%")
    
    # Cenários de exemplo
    scenarios = [
        {
            'name': 'Desenvolvimento Rápido',
            'objective': create_training_objective(
                purpose='testing',
                accuracy_priority=5.0,
                speed_priority=9.0,
                robustness_priority=3.0,
                max_training_time_hours=0.5,
                target_pairs=['EURUSD=X']
            )
        },
        {
            'name': 'Produção Balanceada',
            'objective': create_training_objective(
                purpose='production',
                accuracy_priority=7.0,
                speed_priority=6.0,
                robustness_priority=8.0,
                max_training_time_hours=2.0,
                target_pairs=['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
            )
        },
        {
            'name': 'Pesquisa Robusta',
            'objective': create_training_objective(
                purpose='research',
                accuracy_priority=9.0,
                speed_priority=3.0,
                robustness_priority=10.0,
                max_training_time_hours=5.0,
                target_pairs=['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X']
            )
        }
    ]
    
    print("\n🎯 Analisando Cenários:")
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        recommendation = recommender.recommend_data_amount(scenario['objective'])
        
        print(f"📊 Recomendação: {recommendation.samples_per_pair:,} amostras por par")
        print(f"📅 Período: {recommendation.period_description}")
        print(f"💾 Memória: {recommendation.estimated_memory_gb:.1f}GB")
        print(f"⏱️ Tempo: {recommendation.estimated_time_hours:.1f}h")
        print(f"🎯 Confiança: {recommendation.confidence_score:.1%}")
        print(f"🔒 Robustez: {recommendation.robustness_level}")
        print(f"⚡ Performance: {recommendation.performance_level}")
        
        if recommendation.warnings:
            print(f"⚠️ Avisos: {'; '.join(recommendation.warnings)}")
            
        # Salvar recomendação
        filename = recommender.save_recommendation(
            recommendation, 
            f"recommendation_{scenario['name'].lower().replace(' ', '_')}.json"
        )
        print(f"💾 Salvo: {filename}")
    
    # Análise de eficiência
    print("\n📈 Análise de Eficiência de Dados:")
    efficiency_analysis = recommender.analyze_data_efficiency()
    
    for analysis in efficiency_analysis['efficiency_analysis']:
        print(f"  {analysis['samples']:,} amostras: {analysis['expected_accuracy']:.1%} acurácia, "
              f"{analysis['training_time_minutes']:.1f}min, eficiência {analysis['efficiency_score']:.4f}")

if __name__ == "__main__":
    main()