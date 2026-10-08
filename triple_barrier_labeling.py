import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
import logging
from datetime import datetime, timedelta

class TripleBarrierLabeling:
    """
    Implementação do Triple-Barrier Method conforme López de Prado
    "Advances in Financial Machine Learning"
    
    Este método cria labels mais robustos considerando:
    1. Upper barrier (take profit)
    2. Lower barrier (stop loss) 
    3. Time barrier (horizonte temporal)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def get_events(self, close: pd.Series, t_events: pd.DatetimeIndex, 
                   pt_sl: list = [1, 1], target: Optional[pd.Series] = None,
                   min_ret: float = 0.0, num_threads: int = 1, 
                   t1: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Identifica eventos para aplicação do triple-barrier method
        
        Args:
            close: Série de preços de fechamento
            t_events: Timestamps dos eventos (sinais de entrada)
            pt_sl: [profit_taking_factor, stop_loss_factor]
            target: Volatilidade alvo para cada evento
            min_ret: Retorno mínimo para considerar o evento
            num_threads: Número de threads para processamento
            t1: Timestamps de expiração para cada evento
            
        Returns:
            DataFrame com eventos formatados
        """
        # Calcular target (volatilidade) se não fornecido
        if target is None:
            target = self._get_daily_vol(close, span=100)
            
        # Criar DataFrame de eventos
        events = pd.DataFrame(index=t_events)
        events['t1'] = t1
        events['trgt'] = target.reindex(t_events)
        events['side'] = 1  # Assumir side long por padrão
        
        # Filtrar eventos com target mínimo
        events = events[events['trgt'] > min_ret]
        
        self.logger.info(f"Criados {len(events)} eventos para triple-barrier labeling")
        return events
    
    def apply_triple_barrier(self, close: pd.Series, events: pd.DataFrame,
                           pt_sl: list = [1, 1], molecule: Optional[list] = None) -> pd.DataFrame:
        """
        Aplica o triple-barrier method aos eventos
        
        Args:
            close: Série de preços de fechamento
            events: DataFrame com eventos (output de get_events)
            pt_sl: [profit_taking_factor, stop_loss_factor]
            molecule: Subset de eventos para processar
            
        Returns:
            DataFrame com labels e informações dos barriers
        """
        if molecule is None:
            molecule = events.index
            
        # Inicializar resultado
        out = pd.DataFrame(index=molecule)
        
        for loc in molecule:
            try:
                # Obter dados do evento
                t0 = loc
                t1 = events.at[loc, 't1'] if pd.notna(events.at[loc, 't1']) else close.index[-1]
                target = events.at[loc, 'trgt']
                side = events.at[loc, 'side']
                
                # Calcular barriers
                pt = pt_sl[0] * target  # Profit taking
                sl = -pt_sl[1] * target  # Stop loss
                
                # Obter preços no período
                prices = close[t0:t1]
                if len(prices) == 0:
                    continue
                    
                # Calcular retornos
                returns = (prices / close[t0] - 1) * side
                
                # Encontrar primeiro toque nos barriers
                pt_touch = returns[returns >= pt].index
                sl_touch = returns[returns <= sl].index
                
                # Determinar qual barrier foi tocado primeiro
                if len(pt_touch) > 0 and len(sl_touch) > 0:
                    # Ambos tocados - usar o primeiro
                    if pt_touch[0] <= sl_touch[0]:
                        t_touch = pt_touch[0]
                        label = 1  # Profit taking
                    else:
                        t_touch = sl_touch[0]
                        label = -1  # Stop loss
                elif len(pt_touch) > 0:
                    t_touch = pt_touch[0]
                    label = 1  # Profit taking
                elif len(sl_touch) > 0:
                    t_touch = sl_touch[0]
                    label = -1  # Stop loss
                else:
                    # Nenhum barrier tocado - usar time barrier
                    t_touch = t1
                    final_return = returns.iloc[-1]
                    label = 1 if final_return > 0 else -1
                
                # Armazenar resultado
                out.at[loc, 'label'] = label
                out.at[loc, 't_touch'] = t_touch
                out.at[loc, 'return'] = returns.loc[t_touch]
                out.at[loc, 'barrier_type'] = self._get_barrier_type(label, pt_touch, sl_touch, t_touch, t1)
                
            except Exception as e:
                self.logger.error(f"Erro processando evento {loc}: {e}")
                continue
                
        self.logger.info(f"Triple-barrier aplicado a {len(out)} eventos")
        return out.dropna()
    
    def meta_labeling(self, events: pd.DataFrame, close: pd.Series,
                     primary_forecast: pd.Series, pt_sl: list = [1, 1]) -> pd.DataFrame:
        """
        Aplica meta-labeling para filtrar sinais do modelo primário
        
        Args:
            events: DataFrame com eventos
            close: Série de preços
            primary_forecast: Predições do modelo primário
            pt_sl: Fatores de profit taking e stop loss
            
        Returns:
            DataFrame com meta-labels (0=não operar, 1=operar)
        """
        # Aplicar side do modelo primário
        events_meta = events.copy()
        events_meta['side'] = primary_forecast.reindex(events.index)
        
        # Aplicar triple-barrier
        labels = self.apply_triple_barrier(close, events_meta, pt_sl)
        
        # Meta-label: 1 se o sinal foi lucrativo, 0 caso contrário
        meta_labels = pd.DataFrame(index=labels.index)
        meta_labels['meta_label'] = (labels['return'] > 0).astype(int)
        meta_labels['confidence'] = abs(labels['return'])  # Usar retorno absoluto como confiança
        
        self.logger.info(f"Meta-labeling aplicado: {meta_labels['meta_label'].sum()} sinais positivos de {len(meta_labels)}")
        return meta_labels
    
    def _get_daily_vol(self, close: pd.Series, span: int = 100) -> pd.Series:
        """
        Calcula volatilidade diária usando EWMA
        
        Args:
            close: Série de preços de fechamento
            span: Janela para EWMA
            
        Returns:
            Série de volatilidade diária
        """
        # Calcular retornos logarítmicos
        returns = np.log(close / close.shift(1)).dropna()
        
        # Volatilidade EWMA
        vol = returns.ewm(span=span).std()
        
        return vol
    
    def _get_barrier_type(self, label: int, pt_touch: pd.DatetimeIndex, 
                         sl_touch: pd.DatetimeIndex, t_touch: pd.Timestamp, 
                         t1: pd.Timestamp) -> str:
        """
        Determina qual tipo de barrier foi tocado
        """
        if len(pt_touch) > 0 and t_touch == pt_touch[0]:
            return 'profit_taking'
        elif len(sl_touch) > 0 and t_touch == sl_touch[0]:
            return 'stop_loss'
        elif t_touch == t1:
            return 'time_barrier'
        else:
            return 'unknown'
    
    def get_sample_weights(self, labels: pd.DataFrame, close: pd.Series) -> pd.Series:
        """
        Calcula pesos das amostras baseado na sobreposição temporal
        Implementa o método de López de Prado para lidar com sobreposição
        
        Args:
            labels: DataFrame com labels do triple-barrier
            close: Série de preços
            
        Returns:
            Série com pesos das amostras
        """
        # Criar matriz de sobreposição
        overlap_matrix = self._get_overlap_matrix(labels)
        
        # Calcular pesos baseado na sobreposição
        weights = pd.Series(index=labels.index, dtype=float)
        
        for i, idx in enumerate(labels.index):
            # Número de observações sobrepostas
            num_overlaps = overlap_matrix.iloc[i].sum()
            
            # Peso inversamente proporcional à sobreposição
            weights[idx] = 1.0 / max(num_overlaps, 1)
            
        # Normalizar pesos
        weights = weights / weights.sum() * len(weights)
        
        self.logger.info(f"Pesos calculados para {len(weights)} amostras")
        return weights
    
    def _get_overlap_matrix(self, labels: pd.DataFrame) -> pd.DataFrame:
        """
        Cria matriz de sobreposição temporal entre eventos
        """
        n = len(labels)
        overlap_matrix = pd.DataFrame(0, index=range(n), columns=range(n))
        
        for i, (idx_i, row_i) in enumerate(labels.iterrows()):
            t0_i = idx_i
            t1_i = row_i.get('t_touch', idx_i)
            
            for j, (idx_j, row_j) in enumerate(labels.iterrows()):
                if i == j:
                    overlap_matrix.iloc[i, j] = 1
                    continue
                    
                t0_j = idx_j
                t1_j = row_j.get('t_touch', idx_j)
                
                # Verificar sobreposição
                if (t0_i <= t1_j) and (t1_i >= t0_j):
                    overlap_matrix.iloc[i, j] = 1
                    
        return overlap_matrix
    
    def analyze_barriers_performance(self, labels: pd.DataFrame) -> Dict:
        """
        Analisa a performance dos diferentes tipos de barriers
        
        Args:
            labels: DataFrame com labels do triple-barrier
            
        Returns:
            Dicionário com estatísticas de performance
        """
        stats = {}
        
        # Estatísticas por tipo de barrier
        barrier_types = labels['barrier_type'].value_counts()
        stats['barrier_distribution'] = barrier_types.to_dict()
        
        # Performance por tipo
        for barrier_type in labels['barrier_type'].unique():
            mask = labels['barrier_type'] == barrier_type
            subset = labels[mask]
            
            stats[f'{barrier_type}_count'] = len(subset)
            stats[f'{barrier_type}_avg_return'] = subset['return'].mean()
            stats[f'{barrier_type}_win_rate'] = (subset['return'] > 0).mean()
            stats[f'{barrier_type}_std_return'] = subset['return'].std()
        
        # Estatísticas gerais
        stats['total_events'] = len(labels)
        stats['overall_win_rate'] = (labels['return'] > 0).mean()
        stats['avg_return'] = labels['return'].mean()
        stats['return_std'] = labels['return'].std()
        stats['sharpe_ratio'] = stats['avg_return'] / stats['return_std'] if stats['return_std'] > 0 else 0
        
        self.logger.info(f"Análise de performance concluída: {stats['total_events']} eventos")
        return stats

# Funções auxiliares para compatibilidade com o script de validação
def get_events(close: pd.Series, timestamps: pd.DatetimeIndex, 
               volatility_lookback: int = 100, **kwargs) -> pd.DataFrame:
    """
    Função wrapper para compatibilidade com o script de validação
    """
    tbl = TripleBarrierLabeling()
    
    # Calcular volatilidade
    target = tbl._get_daily_vol(close, span=volatility_lookback)
    
    # Criar t1 padrão (4 horas após cada evento)
    t1 = pd.Series(index=timestamps, data=timestamps + pd.Timedelta(hours=4))
    
    # Criar eventos usando os parâmetros corretos
    events = tbl.get_events(
        close=close,
        t_events=timestamps,
        target=target,
        t1=t1,
        **kwargs
    )
    
    return events

def apply_triple_barrier(close: pd.Series, events: pd.DataFrame,
                        profit_taking_factor: float = 2.0,
                        stop_loss_factor: float = 1.0,
                        max_holding_period: pd.Timedelta = None,
                        **kwargs) -> pd.DataFrame:
    """
    Função wrapper para compatibilidade com o script de validação
    """
    tbl = TripleBarrierLabeling()
    
    # Ajustar t1 se max_holding_period fornecido
    if max_holding_period is not None:
        events = events.copy()
        events['t1'] = events.index + max_holding_period
    
    # Aplicar triple-barrier
    labels = tbl.apply_triple_barrier(
        close=close,
        events=events,
        pt_sl=[profit_taking_factor, stop_loss_factor],
        **kwargs
    )
    
    return labels

def get_sample_weights(timestamps: pd.Series, num_threads: int = 1) -> pd.Series:
    """
    Função wrapper para compatibilidade com o script de validação
    """
    tbl = TripleBarrierLabeling()
    
    # Criar DataFrame de labels dummy
    labels_df = pd.DataFrame({
        't1': timestamps,
        'label': np.ones(len(timestamps))
    }, index=timestamps.index)
    
    # Criar dados dummy para calcular pesos
    dummy_close = pd.Series(np.ones(len(timestamps)), index=timestamps.index)
    
    weights = tbl.get_sample_weights(labels_df, dummy_close)
    
    return weights

def analyze_barriers_performance(labels: pd.DataFrame) -> Dict:
    """
    Função wrapper para compatibilidade com o script de validação
    """
    tbl = TripleBarrierLabeling()
    
    # Adicionar coluna barrier_counts para compatibilidade
    performance = tbl.analyze_barriers_performance(labels)
    performance['barrier_counts'] = performance.get('barrier_distribution', {})
    
    return performance

# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Criar instância
    tbl = TripleBarrierLabeling(logger)
    
    # Exemplo com dados sintéticos
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    np.random.seed(42)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(1000) * 0.01)), index=dates)
    
    # Eventos de exemplo (a cada 24 horas)
    event_times = dates[::24]
    
    # Criar eventos
    events = tbl.get_events(prices, event_times, pt_sl=[2, 1])
    
    # Aplicar triple-barrier
    labels = tbl.apply_triple_barrier(prices, events, pt_sl=[2, 1])
    
    # Analisar performance
    performance = tbl.analyze_barriers_performance(labels)
    
    print("Performance Analysis:")
    for key, value in performance.items():
        print(f"{key}: {value}")
    
    # Calcular pesos das amostras
    weights = tbl.get_sample_weights(labels, prices)
    print(f"\nPesos calculados: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")