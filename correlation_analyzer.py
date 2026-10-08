import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor


class CorrelationType(Enum):
    """Tipos de correlação"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    ROLLING = "rolling"
    DYNAMIC = "dynamic"


class CorrelationStrength(Enum):
    """Classificação da força da correlação"""
    VERY_STRONG_POSITIVE = "very_strong_positive"  # > 0.8
    STRONG_POSITIVE = "strong_positive"            # 0.6 - 0.8
    MODERATE_POSITIVE = "moderate_positive"        # 0.3 - 0.6
    WEAK_POSITIVE = "weak_positive"                # 0.1 - 0.3
    NEGLIGIBLE = "negligible"                      # -0.1 - 0.1
    WEAK_NEGATIVE = "weak_negative"                # -0.3 - -0.1
    MODERATE_NEGATIVE = "moderate_negative"        # -0.6 - -0.3
    STRONG_NEGATIVE = "strong_negative"            # -0.8 - -0.6
    VERY_STRONG_NEGATIVE = "very_strong_negative"  # < -0.8


@dataclass
class CorrelationPair:
    """Representa um par de correlação entre dois ativos"""
    asset1: str
    asset2: str
    correlation: float
    strength: CorrelationStrength
    p_value: Optional[float] = None
    sample_size: int = 0
    last_updated: datetime = None
    stability_score: float = 0.0  # Quão estável é a correlação ao longo do tempo


@dataclass
class CorrelationMatrix:
    """Matriz de correlação completa"""
    matrix: pd.DataFrame
    timestamp: datetime
    assets: List[str]
    correlation_type: CorrelationType
    significant_pairs: List[CorrelationPair]
    summary_stats: Dict


@dataclass
class CorrelationParameters:
    """Parâmetros para análise de correlação"""
    # Configurações temporais
    lookback_periods: int = 50  # Períodos para cálculo
    rolling_window: int = 20    # Janela para correlação rolling
    update_frequency: int = 5   # Frequência de atualização (em períodos)
    
    # Thresholds de significância
    min_correlation_threshold: float = 0.3  # Correlação mínima para considerar significativa
    p_value_threshold: float = 0.05         # P-value para significância estatística
    min_sample_size: int = 20               # Tamanho mínimo da amostra
    
    # Configurações de estabilidade
    stability_window: int = 10              # Janela para calcular estabilidade
    stability_threshold: float = 0.7        # Threshold para correlação estável
    
    # Configurações de performance
    max_assets: int = 20                    # Máximo de ativos para analisar simultaneamente
    cache_duration: int = 300               # Duração do cache em segundos
    
    # Configurações de alertas
    correlation_change_threshold: float = 0.2  # Mudança mínima para alerta
    breakdown_threshold: float = 0.5           # Threshold para breakdown de correlação


class CorrelationAnalyzer:
    """Analisador de correlações entre ativos em tempo real"""
    
    def __init__(self, params: CorrelationParameters, logger: Optional[logging.Logger] = None):
        self.params = params
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache e histórico
        self._correlation_cache: Dict[str, CorrelationMatrix] = {}
        self._price_data_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=params.lookback_periods * 2))
        self._correlation_history: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=params.stability_window))
        
        # Controle de tempo
        self._last_update: Dict[str, datetime] = {}
        self._update_counter = 0
        
        # Thread pool para cálculos paralelos
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("CorrelationAnalyzer inicializado")
    
    def add_price_data(self, asset: str, price: float, timestamp: datetime = None) -> None:
        """Adiciona dados de preço para um ativo"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self._price_data_cache[asset].append({
            'price': price,
            'timestamp': timestamp,
            'returns': None  # Será calculado posteriormente
        })
        
        # Calcular retornos se temos dados suficientes
        if len(self._price_data_cache[asset]) >= 2:
            current_data = self._price_data_cache[asset][-1]
            previous_data = self._price_data_cache[asset][-2]
            
            if previous_data['price'] > 0:
                returns = (current_data['price'] - previous_data['price']) / previous_data['price']
                self._price_data_cache[asset][-1]['returns'] = returns
    
    def calculate_correlation_matrix(self, assets: List[str], 
                                   correlation_type: CorrelationType = CorrelationType.PEARSON) -> CorrelationMatrix:
        """Calcula matriz de correlação para os ativos especificados"""
        try:
            # Verificar se temos dados suficientes
            available_assets = []
            for asset in assets:
                if (asset in self._price_data_cache and 
                    len(self._price_data_cache[asset]) >= self.params.min_sample_size):
                    available_assets.append(asset)
            
            if len(available_assets) < 2:
                self.logger.warning(f"Dados insuficientes para correlação. Assets disponíveis: {len(available_assets)}")
                return self._get_empty_correlation_matrix(assets, correlation_type)
            
            # Preparar dados para correlação
            returns_data = self._prepare_returns_data(available_assets)
            
            if returns_data.empty:
                return self._get_empty_correlation_matrix(assets, correlation_type)
            
            # Calcular matriz de correlação
            if correlation_type == CorrelationType.PEARSON:
                corr_matrix = returns_data.corr(method='pearson')
            elif correlation_type == CorrelationType.SPEARMAN:
                corr_matrix = returns_data.corr(method='spearman')
            elif correlation_type == CorrelationType.KENDALL:
                corr_matrix = returns_data.corr(method='kendall')
            else:
                corr_matrix = returns_data.corr(method='pearson')
            
            # Identificar pares significativos
            significant_pairs = self._identify_significant_pairs(corr_matrix, returns_data)
            
            # Calcular estatísticas resumo
            summary_stats = self._calculate_summary_statistics(corr_matrix, significant_pairs)
            
            correlation_matrix = CorrelationMatrix(
                matrix=corr_matrix,
                timestamp=datetime.now(),
                assets=available_assets,
                correlation_type=correlation_type,
                significant_pairs=significant_pairs,
                summary_stats=summary_stats
            )
            
            # Atualizar cache
            cache_key = f"{'-'.join(sorted(available_assets))}_{correlation_type.value}"
            self._correlation_cache[cache_key] = correlation_matrix
            self._last_update[cache_key] = datetime.now()
            
            self.logger.info(
                f"Matriz de correlação calculada para {len(available_assets)} ativos. "
                f"Pares significativos: {len(significant_pairs)}"
            )
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular matriz de correlação: {e}")
            return self._get_empty_correlation_matrix(assets, correlation_type)
    
    def _prepare_returns_data(self, assets: List[str]) -> pd.DataFrame:
        """Prepara dados de retornos para cálculo de correlação"""
        returns_dict = {}
        min_length = float('inf')
        
        # Coletar retornos de todos os ativos
        for asset in assets:
            if asset in self._price_data_cache:
                returns = []
                for data_point in self._price_data_cache[asset]:
                    if data_point['returns'] is not None:
                        returns.append(data_point['returns'])
                
                if len(returns) >= self.params.min_sample_size:
                    returns_dict[asset] = returns[-self.params.lookback_periods:]
                    min_length = min(min_length, len(returns_dict[asset]))
        
        if not returns_dict or min_length == float('inf'):
            return pd.DataFrame()
        
        # Alinhar séries temporais (usar os últimos min_length pontos)
        aligned_returns = {}
        for asset, returns in returns_dict.items():
            aligned_returns[asset] = returns[-min_length:]
        
        return pd.DataFrame(aligned_returns)
    
    def _identify_significant_pairs(self, corr_matrix: pd.DataFrame, 
                                  returns_data: pd.DataFrame) -> List[CorrelationPair]:
        """Identifica pares de correlação significativos"""
        significant_pairs = []
        assets = corr_matrix.columns.tolist()
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i >= j:  # Evitar duplicatas e auto-correlação
                    continue
                
                correlation = corr_matrix.loc[asset1, asset2]
                
                # Verificar se é significativo
                if abs(correlation) >= self.params.min_correlation_threshold:
                    # Calcular p-value (aproximação)
                    n = len(returns_data)
                    if n > 2:
                        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
                        # Aproximação do p-value usando distribuição t
                        from scipy import stats
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                    else:
                        p_value = 1.0
                    
                    # Classificar força da correlação
                    strength = self._classify_correlation_strength(correlation)
                    
                    # Calcular estabilidade
                    stability_score = self._calculate_stability_score(asset1, asset2, correlation)
                    
                    pair = CorrelationPair(
                        asset1=asset1,
                        asset2=asset2,
                        correlation=correlation,
                        strength=strength,
                        p_value=p_value,
                        sample_size=n,
                        last_updated=datetime.now(),
                        stability_score=stability_score
                    )
                    
                    significant_pairs.append(pair)
                    
                    # Atualizar histórico de correlação
                    pair_key = tuple(sorted([asset1, asset2]))
                    self._correlation_history[pair_key].append({
                        'correlation': correlation,
                        'timestamp': datetime.now()
                    })
        
        # Ordenar por força da correlação
        significant_pairs.sort(key=lambda x: abs(x.correlation), reverse=True)
        
        return significant_pairs
    
    def _classify_correlation_strength(self, correlation: float) -> CorrelationStrength:
        """Classifica a força da correlação"""
        abs_corr = abs(correlation)
        
        if abs_corr > 0.8:
            return CorrelationStrength.VERY_STRONG_POSITIVE if correlation > 0 else CorrelationStrength.VERY_STRONG_NEGATIVE
        elif abs_corr > 0.6:
            return CorrelationStrength.STRONG_POSITIVE if correlation > 0 else CorrelationStrength.STRONG_NEGATIVE
        elif abs_corr > 0.3:
            return CorrelationStrength.MODERATE_POSITIVE if correlation > 0 else CorrelationStrength.MODERATE_NEGATIVE
        elif abs_corr > 0.1:
            return CorrelationStrength.WEAK_POSITIVE if correlation > 0 else CorrelationStrength.WEAK_NEGATIVE
        else:
            return CorrelationStrength.NEGLIGIBLE
    
    def _calculate_stability_score(self, asset1: str, asset2: str, current_correlation: float) -> float:
        """Calcula score de estabilidade da correlação"""
        pair_key = tuple(sorted([asset1, asset2]))
        
        if pair_key not in self._correlation_history:
            return 0.0
        
        history = list(self._correlation_history[pair_key])
        if len(history) < 3:
            return 0.0
        
        # Calcular variabilidade das correlações históricas
        correlations = [h['correlation'] for h in history]
        correlations.append(current_correlation)
        
        # Estabilidade = 1 - (desvio padrão / média absoluta)
        std_dev = np.std(correlations)
        mean_abs = np.mean(np.abs(correlations))
        
        if mean_abs == 0:
            return 0.0
        
        stability = max(0, 1 - (std_dev / mean_abs))
        return min(stability, 1.0)
    
    def _calculate_summary_statistics(self, corr_matrix: pd.DataFrame, 
                                    significant_pairs: List[CorrelationPair]) -> Dict:
        """Calcula estatísticas resumo da matriz de correlação"""
        # Extrair valores da matriz (excluindo diagonal)
        values = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                values.append(corr_matrix.iloc[i, j])
        
        if not values:
            return {}
        
        stats = {
            'total_pairs': len(values),
            'significant_pairs': len(significant_pairs),
            'mean_correlation': np.mean(values),
            'median_correlation': np.median(values),
            'std_correlation': np.std(values),
            'max_correlation': np.max(values),
            'min_correlation': np.min(values),
            'positive_correlations': len([v for v in values if v > 0]),
            'negative_correlations': len([v for v in values if v < 0]),
            'strong_correlations': len([v for v in values if abs(v) > 0.6]),
            'weak_correlations': len([v for v in values if abs(v) < 0.3])
        }
        
        # Estatísticas por força
        strength_counts = defaultdict(int)
        for pair in significant_pairs:
            strength_counts[pair.strength.value] += 1
        
        stats['strength_distribution'] = dict(strength_counts)
        
        return stats
    
    def get_rolling_correlation(self, asset1: str, asset2: str, 
                              window: int = None) -> List[Dict]:
        """Calcula correlação rolling entre dois ativos"""
        if window is None:
            window = self.params.rolling_window
        
        try:
            # Verificar se temos dados suficientes
            if (asset1 not in self._price_data_cache or 
                asset2 not in self._price_data_cache):
                return []
            
            # Preparar dados
            returns_data = self._prepare_returns_data([asset1, asset2])
            
            if returns_data.empty or len(returns_data) < window:
                return []
            
            # Calcular correlação rolling
            rolling_corr = returns_data[asset1].rolling(window=window).corr(returns_data[asset2])
            
            # Converter para lista de dicionários
            results = []
            for i, corr in enumerate(rolling_corr):
                if not np.isnan(corr):
                    results.append({
                        'index': i,
                        'correlation': corr,
                        'strength': self._classify_correlation_strength(corr).value,
                        'window_size': window
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular correlação rolling: {e}")
            return []
    
    def detect_correlation_breakdowns(self, threshold: float = None) -> List[Dict]:
        """Detecta quebras significativas nas correlações"""
        if threshold is None:
            threshold = self.params.breakdown_threshold
        
        breakdowns = []
        
        try:
            for pair_key, history in self._correlation_history.items():
                if len(history) < 3:
                    continue
                
                history_list = list(history)
                recent_corr = history_list[-1]['correlation']
                older_corr = np.mean([h['correlation'] for h in history_list[:-1]])
                
                change = abs(recent_corr - older_corr)
                
                if change >= threshold:
                    breakdowns.append({
                        'asset1': pair_key[0],
                        'asset2': pair_key[1],
                        'previous_correlation': older_corr,
                        'current_correlation': recent_corr,
                        'change': change,
                        'change_percentage': (change / abs(older_corr)) * 100 if older_corr != 0 else 0,
                        'timestamp': history_list[-1]['timestamp'],
                        'severity': 'high' if change >= threshold * 1.5 else 'medium'
                    })
            
            # Ordenar por magnitude da mudança
            breakdowns.sort(key=lambda x: x['change'], reverse=True)
            
            return breakdowns
            
        except Exception as e:
            self.logger.error(f"Erro ao detectar quebras de correlação: {e}")
            return []
    
    def get_correlation_clusters(self, correlation_matrix: CorrelationMatrix, 
                               min_cluster_size: int = 3) -> List[Dict]:
        """Identifica clusters de ativos altamente correlacionados"""
        try:
            if correlation_matrix.matrix.empty:
                return []
            
            # Usar clustering hierárquico baseado na correlação
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            
            # Converter correlação em distância (1 - |correlação|)
            distance_matrix = 1 - np.abs(correlation_matrix.matrix.values)
            
            # Preencher diagonal com zeros
            np.fill_diagonal(distance_matrix, 0)
            
            # Clustering hierárquico
            condensed_distances = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Formar clusters
            max_clusters = min(len(correlation_matrix.assets) // min_cluster_size, 5)
            if max_clusters < 2:
                return []
            
            cluster_labels = fcluster(linkage_matrix, max_clusters, criterion='maxclust')
            
            # Organizar clusters
            clusters = defaultdict(list)
            for i, asset in enumerate(correlation_matrix.assets):
                clusters[cluster_labels[i]].append(asset)
            
            # Filtrar clusters pequenos e calcular estatísticas
            result_clusters = []
            for cluster_id, assets in clusters.items():
                if len(assets) >= min_cluster_size:
                    # Calcular correlação média interna do cluster
                    internal_correlations = []
                    for i, asset1 in enumerate(assets):
                        for j, asset2 in enumerate(assets):
                            if i < j:
                                corr = correlation_matrix.matrix.loc[asset1, asset2]
                                internal_correlations.append(abs(corr))
                    
                    avg_internal_correlation = np.mean(internal_correlations) if internal_correlations else 0
                    
                    result_clusters.append({
                        'cluster_id': cluster_id,
                        'assets': assets,
                        'size': len(assets),
                        'avg_internal_correlation': avg_internal_correlation,
                        'strength': 'strong' if avg_internal_correlation > 0.7 else 'moderate' if avg_internal_correlation > 0.4 else 'weak'
                    })
            
            # Ordenar por força da correlação interna
            result_clusters.sort(key=lambda x: x['avg_internal_correlation'], reverse=True)
            
            return result_clusters
            
        except Exception as e:
            self.logger.error(f"Erro ao identificar clusters de correlação: {e}")
            return []
    
    def generate_trading_insights(self, correlation_matrix: CorrelationMatrix, 
                                target_asset: str) -> Dict:
        """Gera insights de trading baseados nas correlações"""
        try:
            if target_asset not in correlation_matrix.assets:
                return {'error': f'Asset {target_asset} não encontrado na matriz'}
            
            insights = {
                'target_asset': target_asset,
                'timestamp': datetime.now(),
                'correlations': {},
                'hedging_opportunities': [],
                'diversification_suggestions': [],
                'risk_warnings': [],
                'trading_pairs': []
            }
            
            # Analisar correlações do ativo alvo
            target_correlations = correlation_matrix.matrix[target_asset].drop(target_asset)
            
            for asset, correlation in target_correlations.items():
                strength = self._classify_correlation_strength(correlation)
                
                insights['correlations'][asset] = {
                    'correlation': correlation,
                    'strength': strength.value,
                    'abs_correlation': abs(correlation)
                }
                
                # Oportunidades de hedge (correlações negativas fortes)
                if correlation < -0.6:
                    insights['hedging_opportunities'].append({
                        'hedge_asset': asset,
                        'correlation': correlation,
                        'effectiveness': abs(correlation)
                    })
                
                # Sugestões de diversificação (correlações baixas)
                elif abs(correlation) < 0.3:
                    insights['diversification_suggestions'].append({
                        'diversification_asset': asset,
                        'correlation': correlation,
                        'diversification_benefit': 1 - abs(correlation)
                    })
                
                # Avisos de risco (correlações muito altas)
                elif correlation > 0.8:
                    insights['risk_warnings'].append({
                        'risky_asset': asset,
                        'correlation': correlation,
                        'risk_level': 'high',
                        'warning': f'Alta correlação com {asset} pode aumentar risco concentrado'
                    })
                
                # Pares de trading (correlações moderadas a fortes)
                elif 0.4 <= abs(correlation) <= 0.8:
                    insights['trading_pairs'].append({
                        'pair_asset': asset,
                        'correlation': correlation,
                        'strategy': 'pairs_trading' if correlation > 0 else 'contrarian',
                        'strength': strength.value
                    })
            
            # Ordenar listas por relevância
            insights['hedging_opportunities'].sort(key=lambda x: x['effectiveness'], reverse=True)
            insights['diversification_suggestions'].sort(key=lambda x: x['diversification_benefit'], reverse=True)
            insights['trading_pairs'].sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar insights de trading: {e}")
            return {'error': str(e)}
    
    def _get_empty_correlation_matrix(self, assets: List[str], 
                                    correlation_type: CorrelationType) -> CorrelationMatrix:
        """Retorna uma matriz de correlação vazia"""
        return CorrelationMatrix(
            matrix=pd.DataFrame(),
            timestamp=datetime.now(),
            assets=assets,
            correlation_type=correlation_type,
            significant_pairs=[],
            summary_stats={}
        )
    
    def export_correlation_data(self, correlation_matrix: CorrelationMatrix, 
                              format: str = 'json') -> str:
        """Exporta dados de correlação em formato especificado"""
        try:
            export_data = {
                'timestamp': correlation_matrix.timestamp.isoformat(),
                'correlation_type': correlation_matrix.correlation_type.value,
                'assets': correlation_matrix.assets,
                'matrix': correlation_matrix.matrix.to_dict() if not correlation_matrix.matrix.empty else {},
                'significant_pairs': [
                    {
                        'asset1': pair.asset1,
                        'asset2': pair.asset2,
                        'correlation': pair.correlation,
                        'strength': pair.strength.value,
                        'p_value': pair.p_value,
                        'sample_size': pair.sample_size,
                        'stability_score': pair.stability_score
                    }
                    for pair in correlation_matrix.significant_pairs
                ],
                'summary_stats': correlation_matrix.summary_stats
            }
            
            if format.lower() == 'json':
                return json.dumps(export_data, indent=2, default=str)
            else:
                return str(export_data)
                
        except Exception as e:
            self.logger.error(f"Erro ao exportar dados de correlação: {e}")
            return '{}'
    
    def cleanup_old_data(self, max_age_hours: int = 24) -> None:
        """Remove dados antigos do cache"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Limpar cache de correlação
            keys_to_remove = []
            for key, timestamp in self._last_update.items():
                if timestamp < cutoff_time:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                if key in self._correlation_cache:
                    del self._correlation_cache[key]
                del self._last_update[key]
            
            # Limpar histórico de correlações antigas
            for pair_key in list(self._correlation_history.keys()):
                history = self._correlation_history[pair_key]
                # Manter apenas dados recentes
                recent_data = deque(maxlen=self.params.stability_window)
                for data_point in history:
                    if data_point['timestamp'] >= cutoff_time:
                        recent_data.append(data_point)
                
                if recent_data:
                    self._correlation_history[pair_key] = recent_data
                else:
                    del self._correlation_history[pair_key]
            
            self.logger.info(f"Limpeza de dados concluída. Removidos {len(keys_to_remove)} itens do cache")
            
        except Exception as e:
            self.logger.error(f"Erro na limpeza de dados: {e}")


# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Parâmetros de exemplo
    params = CorrelationParameters(
        lookback_periods=50,
        min_correlation_threshold=0.3,
        rolling_window=20
    )
    
    # Inicializar analisador
    analyzer = CorrelationAnalyzer(params, logger)
    
    # Simular dados de preços
    import random
    
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    base_prices = {'EURUSD': 1.1000, 'GBPUSD': 1.3000, 'USDJPY': 110.00, 'AUDUSD': 0.7500}
    
    # Adicionar dados históricos simulados
    for i in range(100):
        timestamp = datetime.now() - timedelta(hours=100-i)
        
        for asset in assets:
            # Simular movimento correlacionado
            if asset == 'EURUSD':
                change = random.gauss(0, 0.001)
            elif asset == 'GBPUSD':
                # Correlação positiva com EUR
                eur_change = (base_prices['EURUSD'] - 1.1000) / 1.1000
                change = eur_change * 0.7 + random.gauss(0, 0.0008)
            elif asset == 'USDJPY':
                # Correlação negativa com EUR
                eur_change = (base_prices['EURUSD'] - 1.1000) / 1.1000
                change = -eur_change * 0.5 + random.gauss(0, 0.0012)
            else:  # AUDUSD
                # Correlação moderada com EUR
                eur_change = (base_prices['EURUSD'] - 1.1000) / 1.1000
                change = eur_change * 0.4 + random.gauss(0, 0.001)
            
            new_price = base_prices[asset] * (1 + change)
            base_prices[asset] = new_price
            
            analyzer.add_price_data(asset, new_price, timestamp)
    
    # Calcular matriz de correlação
    correlation_matrix = analyzer.calculate_correlation_matrix(assets)
    
    # Gerar insights para EURUSD
    insights = analyzer.generate_trading_insights(correlation_matrix, 'EURUSD')
    
    # Detectar quebras de correlação
    breakdowns = analyzer.detect_correlation_breakdowns()
    
    # Identificar clusters
    clusters = analyzer.get_correlation_clusters(correlation_matrix)
    
    print("\n=== Análise de Correlação ===")
    print(f"Assets analisados: {correlation_matrix.assets}")
    print(f"Pares significativos: {len(correlation_matrix.significant_pairs)}")
    
    print("\n=== Matriz de Correlação ===")
    if not correlation_matrix.matrix.empty:
        print(correlation_matrix.matrix.round(3))
    
    print("\n=== Insights para EURUSD ===")
    if 'hedging_opportunities' in insights:
        print(f"Oportunidades de hedge: {len(insights['hedging_opportunities'])}")
        for hedge in insights['hedging_opportunities'][:2]:
            print(f"  - {hedge['hedge_asset']}: {hedge['correlation']:.3f}")
    
    print(f"\nSugestões de diversificação: {len(insights.get('diversification_suggestions', []))}")
    print(f"Pares de trading: {len(insights.get('trading_pairs', []))}")
    
    if breakdowns:
        print(f"\n=== Quebras de Correlação Detectadas ===")
        for breakdown in breakdowns[:3]:
            print(f"{breakdown['asset1']}-{breakdown['asset2']}: {breakdown['change']:.3f}")
    
    if clusters:
        print(f"\n=== Clusters Identificados ===")
        for cluster in clusters:
            print(f"Cluster {cluster['cluster_id']}: {cluster['assets']} (Correlação: {cluster['avg_internal_correlation']:.3f})")