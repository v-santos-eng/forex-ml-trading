import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta


class VolumeProfileType(Enum):
    """Tipos de perfil de volume"""
    FIXED_RANGE = "fixed_range"
    VISIBLE_RANGE = "visible_range"
    SESSION = "session"
    MARKET_PROFILE = "market_profile"


class VolumeNodeType(Enum):
    """Tipos de nós de volume"""
    POC = "point_of_control"  # Ponto de Controle
    VAH = "value_area_high"   # Área de Valor Alta
    VAL = "value_area_low"    # Área de Valor Baixa
    HVN = "high_volume_node"  # Nó de Alto Volume
    LVN = "low_volume_node"   # Nó de Baixo Volume


@dataclass
class VolumeNode:
    """Representa um nó de volume em um nível de preço"""
    price_level: float
    volume: float
    percentage: float
    node_type: VolumeNodeType
    significance: float  # 0-1, onde 1 é mais significativo


@dataclass
class ValueArea:
    """Representa a área de valor (70% do volume)"""
    high: float
    low: float
    poc: float  # Point of Control
    volume_percentage: float
    total_volume: float


@dataclass
class VolumeProfileParameters:
    """Parâmetros para análise de volume profile"""
    # Configurações básicas
    price_bins: int = 50  # Número de bins de preço
    value_area_percentage: float = 0.70  # Percentual da área de valor
    min_volume_threshold: float = 0.01  # Volume mínimo para considerar um nível
    
    # Configurações de nós
    hvn_threshold: float = 0.05  # Threshold para High Volume Nodes
    lvn_threshold: float = 0.01  # Threshold para Low Volume Nodes
    
    # Configurações temporais
    lookback_periods: int = 100  # Períodos para análise
    session_start_hour: int = 9   # Hora de início da sessão
    session_end_hour: int = 17    # Hora de fim da sessão
    
    # Configurações de significância
    poc_weight: float = 3.0       # Peso do POC
    value_area_weight: float = 2.0 # Peso da área de valor
    hvn_weight: float = 1.5       # Peso dos HVNs


class VolumeProfileAnalyzer:
    """Analisador de perfil de volume para identificação de áreas de interesse"""
    
    def __init__(self, params: VolumeProfileParameters, logger: Optional[logging.Logger] = None):
        self.params = params
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache para perfis calculados
        self._profile_cache: Dict[str, Dict] = {}
        self._last_update: Dict[str, datetime] = {}
        
        self.logger.info("VolumeProfileAnalyzer inicializado")
    
    def calculate_volume_profile(
        self, 
        data: pd.DataFrame, 
        profile_type: VolumeProfileType = VolumeProfileType.FIXED_RANGE
    ) -> Dict:
        """
        Calcula o perfil de volume para os dados fornecidos
        
        Args:
            data: DataFrame com colunas ['high', 'low', 'close', 'volume', 'timestamp']
            profile_type: Tipo de perfil a ser calculado
            
        Returns:
            Dict com informações do perfil de volume
        """
        try:
            if data.empty or len(data) < 10:
                self.logger.warning("Dados insuficientes para análise de volume profile")
                return self._get_empty_profile()
            
            # Determinar range de preços
            price_range = self._get_price_range(data, profile_type)
            
            # Criar bins de preço
            price_bins = np.linspace(
                price_range['min'], 
                price_range['max'], 
                self.params.price_bins + 1
            )
            
            # Calcular volume por bin
            volume_profile = self._calculate_volume_distribution(
                data, price_bins
            )
            
            # Identificar nós importantes
            volume_nodes = self._identify_volume_nodes(volume_profile)
            
            # Calcular área de valor
            value_area = self._calculate_value_area(volume_profile)
            
            # Calcular métricas de significância
            significance_metrics = self._calculate_significance_metrics(
                volume_profile, volume_nodes, value_area
            )
            
            profile_result = {
                'timestamp': datetime.now(),
                'profile_type': profile_type,
                'price_range': price_range,
                'volume_profile': volume_profile,
                'volume_nodes': volume_nodes,
                'value_area': value_area,
                'significance_metrics': significance_metrics,
                'total_volume': data['volume'].sum(),
                'price_levels': len(price_bins) - 1
            }
            
            self.logger.info(
                f"Volume profile calculado: {len(volume_nodes)} nós identificados, "
                f"POC em {value_area.poc:.5f}"
            )
            
            return profile_result
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular volume profile: {e}")
            return self._get_empty_profile()
    
    def _get_price_range(self, data: pd.DataFrame, profile_type: VolumeProfileType) -> Dict[str, float]:
        """Determina o range de preços baseado no tipo de perfil"""
        if profile_type == VolumeProfileType.SESSION:
            # Filtrar dados da sessão atual
            session_data = self._filter_session_data(data)
            if not session_data.empty:
                data = session_data
        
        return {
            'min': data['low'].min(),
            'max': data['high'].max(),
            'range': data['high'].max() - data['low'].min()
        }
    
    def _filter_session_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filtra dados para a sessão atual"""
        try:
            if 'timestamp' not in data.columns:
                return data
            
            # Converter timestamp se necessário
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Filtrar por horário da sessão
            session_mask = (
                (data['timestamp'].dt.hour >= self.params.session_start_hour) &
                (data['timestamp'].dt.hour <= self.params.session_end_hour)
            )
            
            return data[session_mask]
            
        except Exception as e:
            self.logger.warning(f"Erro ao filtrar dados da sessão: {e}")
            return data
    
    def _calculate_volume_distribution(self, data: pd.DataFrame, price_bins: np.ndarray) -> pd.DataFrame:
        """Calcula a distribuição de volume por bins de preço"""
        volume_distribution = []
        
        for i in range(len(price_bins) - 1):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]
            bin_mid = (bin_low + bin_high) / 2
            
            # Calcular volume no bin (considerando que o preço pode estar em qualquer lugar do range OHLC)
            volume_in_bin = 0
            
            for _, row in data.iterrows():
                # Se o range OHLC intersecta com o bin
                if row['low'] <= bin_high and row['high'] >= bin_low:
                    # Calcular proporção do volume que está no bin
                    overlap_low = max(row['low'], bin_low)
                    overlap_high = min(row['high'], bin_high)
                    overlap_range = overlap_high - overlap_low
                    
                    if overlap_range > 0:
                        candle_range = row['high'] - row['low']
                        if candle_range > 0:
                            proportion = overlap_range / candle_range
                            volume_in_bin += row['volume'] * proportion
                        else:
                            # Se high == low, todo volume vai para o bin
                            if bin_low <= row['close'] <= bin_high:
                                volume_in_bin += row['volume']
            
            volume_distribution.append({
                'price_level': bin_mid,
                'price_low': bin_low,
                'price_high': bin_high,
                'volume': volume_in_bin
            })
        
        df_volume = pd.DataFrame(volume_distribution)
        total_volume = df_volume['volume'].sum()
        
        if total_volume > 0:
            df_volume['percentage'] = df_volume['volume'] / total_volume
        else:
            df_volume['percentage'] = 0
        
        return df_volume.sort_values('volume', ascending=False)
    
    def _identify_volume_nodes(self, volume_profile: pd.DataFrame) -> List[VolumeNode]:
        """Identifica nós de volume importantes"""
        nodes = []
        
        if volume_profile.empty:
            return nodes
        
        # Point of Control (POC) - maior volume
        poc_row = volume_profile.iloc[0]
        nodes.append(VolumeNode(
            price_level=poc_row['price_level'],
            volume=poc_row['volume'],
            percentage=poc_row['percentage'],
            node_type=VolumeNodeType.POC,
            significance=1.0
        ))
        
        # High Volume Nodes (HVN)
        hvn_threshold = self.params.hvn_threshold
        hvn_candidates = volume_profile[volume_profile['percentage'] >= hvn_threshold]
        
        for _, row in hvn_candidates.iterrows():
            if row['price_level'] != poc_row['price_level']:  # Evitar duplicar POC
                significance = min(row['percentage'] / poc_row['percentage'], 1.0)
                nodes.append(VolumeNode(
                    price_level=row['price_level'],
                    volume=row['volume'],
                    percentage=row['percentage'],
                    node_type=VolumeNodeType.HVN,
                    significance=significance * 0.8  # HVN tem significância menor que POC
                ))
        
        # Low Volume Nodes (LVN) - gaps no volume
        lvn_threshold = self.params.lvn_threshold
        lvn_candidates = volume_profile[volume_profile['percentage'] <= lvn_threshold]
        
        for _, row in lvn_candidates.iterrows():
            if row['percentage'] > 0:  # Evitar zeros
                significance = 1.0 - (row['percentage'] / lvn_threshold)
                nodes.append(VolumeNode(
                    price_level=row['price_level'],
                    volume=row['volume'],
                    percentage=row['percentage'],
                    node_type=VolumeNodeType.LVN,
                    significance=significance * 0.6  # LVN tem significância menor
                ))
        
        return sorted(nodes, key=lambda x: x.significance, reverse=True)
    
    def _calculate_value_area(self, volume_profile: pd.DataFrame) -> ValueArea:
        """Calcula a área de valor (70% do volume)"""
        if volume_profile.empty:
            return ValueArea(0, 0, 0, 0, 0)
        
        total_volume = volume_profile['volume'].sum()
        target_volume = total_volume * self.params.value_area_percentage
        
        # Começar do POC e expandir para cima e para baixo
        poc_idx = 0  # Já está ordenado por volume
        poc_price = volume_profile.iloc[poc_idx]['price_level']
        
        # Ordenar por preço para facilitar expansão
        sorted_by_price = volume_profile.sort_values('price_level')
        poc_price_idx = sorted_by_price[sorted_by_price['price_level'] == poc_price].index[0]
        
        # Expandir área de valor
        accumulated_volume = volume_profile.iloc[poc_idx]['volume']
        upper_idx = poc_price_idx
        lower_idx = poc_price_idx
        
        while accumulated_volume < target_volume and (upper_idx < len(sorted_by_price) - 1 or lower_idx > 0):
            # Decidir se expandir para cima ou para baixo
            upper_volume = 0
            lower_volume = 0
            
            if upper_idx < len(sorted_by_price) - 1:
                upper_volume = sorted_by_price.iloc[upper_idx + 1]['volume']
            
            if lower_idx > 0:
                lower_volume = sorted_by_price.iloc[lower_idx - 1]['volume']
            
            # Expandir para o lado com maior volume
            if upper_volume >= lower_volume and upper_idx < len(sorted_by_price) - 1:
                upper_idx += 1
                accumulated_volume += upper_volume
            elif lower_idx > 0:
                lower_idx -= 1
                accumulated_volume += lower_volume
            else:
                break
        
        value_area_high = sorted_by_price.iloc[upper_idx]['price_level']
        value_area_low = sorted_by_price.iloc[lower_idx]['price_level']
        
        return ValueArea(
            high=value_area_high,
            low=value_area_low,
            poc=poc_price,
            volume_percentage=accumulated_volume / total_volume if total_volume > 0 else 0,
            total_volume=accumulated_volume
        )
    
    def _calculate_significance_metrics(self, volume_profile: pd.DataFrame, 
                                      volume_nodes: List[VolumeNode], 
                                      value_area: ValueArea) -> Dict:
        """Calcula métricas de significância dos níveis"""
        if volume_profile.empty:
            return {}
        
        total_volume = volume_profile['volume'].sum()
        
        # Métricas básicas
        metrics = {
            'total_volume': total_volume,
            'poc_volume_percentage': volume_profile.iloc[0]['percentage'] if not volume_profile.empty else 0,
            'value_area_coverage': value_area.volume_percentage,
            'volume_concentration': self._calculate_volume_concentration(volume_profile),
            'price_range_efficiency': self._calculate_price_range_efficiency(volume_profile, value_area)
        }
        
        # Métricas dos nós
        node_metrics = {
            'total_nodes': len(volume_nodes),
            'hvn_count': len([n for n in volume_nodes if n.node_type == VolumeNodeType.HVN]),
            'lvn_count': len([n for n in volume_nodes if n.node_type == VolumeNodeType.LVN]),
            'avg_node_significance': np.mean([n.significance for n in volume_nodes]) if volume_nodes else 0
        }
        
        metrics.update(node_metrics)
        return metrics
    
    def _calculate_volume_concentration(self, volume_profile: pd.DataFrame) -> float:
        """Calcula o índice de concentração de volume (Herfindahl-Hirschman Index)"""
        if volume_profile.empty:
            return 0
        
        # HHI = soma dos quadrados das percentagens
        hhi = (volume_profile['percentage'] ** 2).sum()
        return hhi
    
    def _calculate_price_range_efficiency(self, volume_profile: pd.DataFrame, value_area: ValueArea) -> float:
        """Calcula a eficiência do range de preços (quão concentrado está o volume)"""
        if volume_profile.empty or value_area.high == value_area.low:
            return 0
        
        total_price_range = volume_profile['price_level'].max() - volume_profile['price_level'].min()
        value_area_range = value_area.high - value_area.low
        
        if total_price_range == 0:
            return 1.0
        
        # Eficiência = (volume na área de valor) / (range da área de valor / range total)
        range_ratio = value_area_range / total_price_range
        efficiency = value_area.volume_percentage / range_ratio if range_ratio > 0 else 0
        
        return min(efficiency, 1.0)  # Limitar a 1.0
    
    def get_support_resistance_levels(self, profile_data: Dict, 
                                    min_significance: float = 0.3) -> Dict[str, List[float]]:
        """Extrai níveis de suporte e resistência baseados no volume profile"""
        try:
            volume_nodes = profile_data.get('volume_nodes', [])
            value_area = profile_data.get('value_area')
            
            support_levels = []
            resistance_levels = []
            key_levels = []
            
            # Adicionar POC como nível chave
            if value_area and value_area.poc > 0:
                key_levels.append(value_area.poc)
            
            # Adicionar limites da área de valor
            if value_area:
                if value_area.high > 0:
                    resistance_levels.append(value_area.high)
                if value_area.low > 0:
                    support_levels.append(value_area.low)
            
            # Processar nós de volume
            for node in volume_nodes:
                if node.significance >= min_significance:
                    if node.node_type in [VolumeNodeType.POC, VolumeNodeType.HVN]:
                        # HVNs podem atuar como suporte ou resistência
                        key_levels.append(node.price_level)
                    elif node.node_type == VolumeNodeType.LVN:
                        # LVNs são áreas de baixo interesse, podem ser breakout points
                        pass
            
            # Remover duplicatas e ordenar
            support_levels = sorted(list(set(support_levels)))
            resistance_levels = sorted(list(set(resistance_levels)))
            key_levels = sorted(list(set(key_levels)))
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'key_levels': key_levels,
                'poc': value_area.poc if value_area else 0,
                'value_area_high': value_area.high if value_area else 0,
                'value_area_low': value_area.low if value_area else 0
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao extrair níveis de suporte/resistência: {e}")
            return {
                'support_levels': [],
                'resistance_levels': [],
                'key_levels': [],
                'poc': 0,
                'value_area_high': 0,
                'value_area_low': 0
            }
    
    def analyze_price_action_at_levels(self, current_price: float, 
                                     profile_data: Dict) -> Dict:
        """Analisa a ação do preço em relação aos níveis de volume"""
        try:
            levels = self.get_support_resistance_levels(profile_data)
            value_area = profile_data.get('value_area')
            
            analysis = {
                'current_price': current_price,
                'price_position': 'unknown',
                'nearest_support': 0,
                'nearest_resistance': 0,
                'distance_to_poc': 0,
                'in_value_area': False,
                'volume_context': 'neutral'
            }
            
            if not value_area:
                return analysis
            
            # Determinar posição do preço
            if current_price > value_area.high:
                analysis['price_position'] = 'above_value_area'
                analysis['volume_context'] = 'low_volume_area'
            elif current_price < value_area.low:
                analysis['price_position'] = 'below_value_area'
                analysis['volume_context'] = 'low_volume_area'
            else:
                analysis['price_position'] = 'in_value_area'
                analysis['in_value_area'] = True
                analysis['volume_context'] = 'high_volume_area'
            
            # Encontrar suporte e resistência mais próximos
            all_levels = levels['key_levels'] + levels['support_levels'] + levels['resistance_levels']
            all_levels = sorted(list(set(all_levels)))
            
            supports_below = [level for level in all_levels if level < current_price]
            resistances_above = [level for level in all_levels if level > current_price]
            
            if supports_below:
                analysis['nearest_support'] = max(supports_below)
            if resistances_above:
                analysis['nearest_resistance'] = min(resistances_above)
            
            # Distância ao POC
            analysis['distance_to_poc'] = abs(current_price - value_area.poc)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Erro ao analisar ação do preço: {e}")
            return {'error': str(e)}
    
    def _get_empty_profile(self) -> Dict:
        """Retorna um perfil vazio em caso de erro"""
        return {
            'timestamp': datetime.now(),
            'profile_type': VolumeProfileType.FIXED_RANGE,
            'price_range': {'min': 0, 'max': 0, 'range': 0},
            'volume_profile': pd.DataFrame(),
            'volume_nodes': [],
            'value_area': ValueArea(0, 0, 0, 0, 0),
            'significance_metrics': {},
            'total_volume': 0,
            'price_levels': 0
        }
    
    def get_trading_signals(self, current_price: float, profile_data: Dict, 
                          price_trend: str = 'neutral') -> Dict:
        """Gera sinais de trading baseados no volume profile"""
        try:
            analysis = self.analyze_price_action_at_levels(current_price, profile_data)
            levels = self.get_support_resistance_levels(profile_data)
            value_area = profile_data.get('value_area')
            
            signals = {
                'signal_strength': 0,  # -1 a 1
                'signal_type': 'neutral',
                'confidence': 0,  # 0 a 1
                'key_levels': levels,
                'analysis': analysis,
                'recommendations': []
            }
            
            if not value_area or value_area.poc == 0:
                return signals
            
            # Lógica de sinais baseada na posição do preço
            if analysis['price_position'] == 'above_value_area':
                if price_trend == 'bullish':
                    signals['signal_type'] = 'continuation_bullish'
                    signals['signal_strength'] = 0.6
                    signals['recommendations'].append("Preço em área de baixo volume - possível continuação de alta")
                else:
                    signals['signal_type'] = 'reversal_bearish'
                    signals['signal_strength'] = -0.4
                    signals['recommendations'].append("Preço acima da área de valor - possível reversão")
            
            elif analysis['price_position'] == 'below_value_area':
                if price_trend == 'bearish':
                    signals['signal_type'] = 'continuation_bearish'
                    signals['signal_strength'] = -0.6
                    signals['recommendations'].append("Preço em área de baixo volume - possível continuação de baixa")
                else:
                    signals['signal_type'] = 'reversal_bullish'
                    signals['signal_strength'] = 0.4
                    signals['recommendations'].append("Preço abaixo da área de valor - possível reversão")
            
            elif analysis['in_value_area']:
                # Dentro da área de valor - analisar proximidade ao POC
                distance_to_poc_pct = abs(current_price - value_area.poc) / value_area.poc if value_area.poc > 0 else 0
                
                if distance_to_poc_pct < 0.001:  # Muito próximo ao POC
                    signals['signal_type'] = 'consolidation'
                    signals['signal_strength'] = 0
                    signals['recommendations'].append("Preço próximo ao POC - área de equilíbrio")
                else:
                    signals['signal_type'] = 'range_trading'
                    signals['signal_strength'] = 0.2 if current_price > value_area.poc else -0.2
                    signals['recommendations'].append("Preço na área de valor - trading de range")
            
            # Calcular confiança baseada nas métricas de significância
            significance_metrics = profile_data.get('significance_metrics', {})
            volume_concentration = significance_metrics.get('volume_concentration', 0)
            
            # Maior concentração de volume = maior confiança
            signals['confidence'] = min(volume_concentration * 2, 1.0)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar sinais de trading: {e}")
            return {
                'signal_strength': 0,
                'signal_type': 'error',
                'confidence': 0,
                'error': str(e)
            }


# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Parâmetros de exemplo
    params = VolumeProfileParameters(
        price_bins=30,
        value_area_percentage=0.70,
        hvn_threshold=0.05,
        lvn_threshold=0.01
    )
    
    # Inicializar analisador
    analyzer = VolumeProfileAnalyzer(params, logger)
    
    # Dados de exemplo
    sample_data = pd.DataFrame({
        'high': [1.1050, 1.1055, 1.1060, 1.1058, 1.1062],
        'low': [1.1045, 1.1048, 1.1052, 1.1054, 1.1058],
        'close': [1.1048, 1.1053, 1.1057, 1.1056, 1.1060],
        'volume': [1000, 1500, 2000, 1200, 800],
        'timestamp': pd.date_range('2024-01-01 09:00', periods=5, freq='1H')
    })
    
    # Calcular volume profile
    profile = analyzer.calculate_volume_profile(sample_data)
    
    # Obter níveis de suporte/resistência
    levels = analyzer.get_support_resistance_levels(profile)
    
    # Analisar ação do preço
    current_price = 1.1059
    analysis = analyzer.analyze_price_action_at_levels(current_price, profile)
    
    # Gerar sinais
    signals = analyzer.get_trading_signals(current_price, profile, 'bullish')
    
    print(f"Volume Profile Analysis:")
    print(f"POC: {profile['value_area'].poc:.5f}")
    print(f"Value Area: {profile['value_area'].low:.5f} - {profile['value_area'].high:.5f}")
    print(f"Current Price Analysis: {analysis['price_position']}")
    print(f"Trading Signal: {signals['signal_type']} (Strength: {signals['signal_strength']:.2f})")