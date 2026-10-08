import numpy as np
import pandas as pd
from typing import Optional, Union
import multiprocessing as mp
from functools import partial


class TripleBarrierLabeling:
    """
    Implementação do Triple Barrier Method para rotulagem de dados financeiros.
    
    Baseado no livro "Advances in Financial Machine Learning" de Marcos López de Prado.
    O método define três barreiras para cada evento:
    1. Upper barrier (profit-taking)
    2. Lower barrier (stop-loss)
    3. Time barrier (holding period limit)
    
    O primeiro toque em qualquer barreira determina o label do evento.
    """
    
    def __init__(self, close_prices: pd.Series):
        """
        Inicializa o Triple Barrier Labeling.
        
        Args:
            close_prices: Série temporal de preços de fechamento
        """
        self.close_prices = close_prices.copy()
        self.close_prices.index = pd.to_datetime(self.close_prices.index)
        
    def get_events(self, 
                   t_events: pd.DatetimeIndex,
                   pt_sl: Optional[Union[float, pd.Series]] = None,
                   t1: Optional[pd.Series] = None,
                   trgt: Optional[pd.Series] = None,
                   min_ret: float = 0.0,
                   num_threads: int = 1) -> pd.DataFrame:
        """
        Identifica eventos baseados no Triple Barrier Method.
        
        Args:
            t_events: Timestamps dos eventos a serem analisados
            pt_sl: Profit-taking/stop-loss múltiplos (simétricos se float)
            t1: Timestamps das barreiras de tempo para cada evento
            trgt: Target volatility para cada evento
            min_ret: Retorno mínimo para considerar um evento válido
            num_threads: Número de threads para processamento paralelo
            
        Returns:
            DataFrame com colunas: t1 (time barrier), trgt (target), side (direção)
        """
        # Validação de entrada
        t_events = t_events.intersection(self.close_prices.index)
        
        if len(t_events) == 0:
            return pd.DataFrame(columns=['t1', 'trgt', 'side'])
            
        # Configuração padrão para pt_sl
        if pt_sl is None:
            pt_sl = pd.Series(index=t_events, data=1.0)
        elif isinstance(pt_sl, (int, float)):
            pt_sl = pd.Series(index=t_events, data=pt_sl)
            
        # Configuração padrão para t1 (barreira de tempo)
        if t1 is None:
            t1 = pd.Series(index=t_events, data=pd.NaT)
            
        # Configuração padrão para target volatility
        if trgt is None:
            trgt = self._get_daily_vol(span=100).reindex(t_events)
            trgt = trgt.fillna(method='ffill').fillna(0.01)
            
        # Criar DataFrame de eventos
        events = pd.DataFrame({
            't1': t1,
            'trgt': trgt,
            'side': pd.Series(index=t_events, data=1.0)  # Assumindo long por padrão
        })
        
        # Filtrar eventos com target válido
        events = events.dropna(subset=['trgt'])
        events = events[events['trgt'] > min_ret]
        
        return events
    
    def get_bins(self, events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """
        Calcula os labels baseados no primeiro toque nas barreiras.
        
        Args:
            events: DataFrame retornado por get_events
            close: Série de preços de fechamento
            
        Returns:
            DataFrame com labels (-1, 0, 1) e timestamps dos toques
        """
        # Aplicar Triple Barrier para cada evento
        out = events[['t1', 'trgt', 'side']].copy(deep=True)
        
        # Calcular preços das barreiras
        px = self._get_barrier_prices(events, close)
        
        # Encontrar primeiro toque
        out['t1'] = px.dropna(how='all').index  # Atualizar t1 com tempo real do toque
        out['bin'] = px.apply(self._get_bin, axis=1)
        
        return out.dropna()
    
    def _get_daily_vol(self, span: int = 100) -> pd.Series:
        """
        Calcula volatilidade diária usando EWMA dos log returns.
        
        Args:
            span: Janela para EWMA
            
        Returns:
            Série de volatilidade diária
        """
        df0 = self.close_prices.index.searchsorted(self.close_prices.index - pd.Timedelta(days=1))
        df0 = df0[df0 > 0]
        
        # Log returns
        df0 = pd.Series(df0, index=self.close_prices.index[df0])
        df0 = np.log(self.close_prices.loc[df0.index] / self.close_prices.loc[df0.values].values)
        
        # EWMA volatility
        df0 = df0.ewm(span=span).std()
        
        return df0
    
    def _get_barrier_prices(self, events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """
        Calcula os preços das barreiras superior e inferior.
        
        Args:
            events: DataFrame de eventos
            close: Série de preços
            
        Returns:
            DataFrame com preços das barreiras
        """
        barriers = pd.DataFrame(index=events.index)
        
        for loc, t1 in events['t1'].fillna(close.index[-1]).items():
            df0 = close[loc:t1]  # Preços no período do evento
            
            if len(df0) == 0:
                continue
                
            # Preço inicial
            p0 = close[loc]
            
            # Target e side
            trgt = events.at[loc, 'trgt']
            side = events.at[loc, 'side']
            
            # Barreiras
            upper = p0 * (1 + side * trgt)
            lower = p0 * (1 - side * trgt)
            
            # Verificar toques
            df1 = df0.copy()
            
            # Toque na barreira superior (profit-taking)
            if side > 0:
                df1[df1 >= upper] = 1
            else:
                df1[df1 <= upper] = 1
                
            # Toque na barreira inferior (stop-loss)
            if side > 0:
                df1[df1 <= lower] = -1
            else:
                df1[df1 >= lower] = -1
                
            # Manter apenas valores de toque
            df1 = df1[(df1 == 1) | (df1 == -1)]
            
            if len(df1) > 0:
                barriers.loc[loc, 'barrier'] = df1.iloc[0]
                barriers.loc[loc, 't1'] = df1.index[0]
            else:
                # Sem toque - usar barreira de tempo
                barriers.loc[loc, 'barrier'] = 0
                barriers.loc[loc, 't1'] = t1
                
        return barriers
    
    def _get_bin(self, row) -> int:
        """
        Determina o bin baseado no primeiro toque.
        
        Args:
            row: Linha do DataFrame com informações da barreira
            
        Returns:
            Label: -1 (sell), 0 (hold), 1 (buy)
        """
        if pd.isna(row.get('barrier')):
            return 0
        return int(row['barrier'])
    
    def apply_pt_sl_on_t1(self, 
                          close: pd.Series,
                          events: pd.DataFrame,
                          pt_sl: Union[float, pd.Series],
                          molecule: list) -> pd.DataFrame:
        """
        Aplica profit-taking e stop-loss em um subconjunto de eventos.
        Função auxiliar para processamento paralelo.
        
        Args:
            close: Série de preços
            events: DataFrame de eventos
            pt_sl: Múltiplos de profit-taking/stop-loss
            molecule: Lista de índices para processar
            
        Returns:
            DataFrame com resultados do subconjunto
        """
        events_ = events.loc[molecule]
        out = events_[['t1']].copy(deep=True)
        
        if isinstance(pt_sl, (int, float)):
            pt_sl_ = pd.Series(index=events_.index, data=pt_sl)
        else:
            pt_sl_ = pt_sl.loc[events_.index]
            
        for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
            df0 = close[loc:t1]
            
            if len(df0) == 0:
                continue
                
            df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']
            
            # Profit-taking
            df0_pt = df0[df0 > pt_sl_.at[loc]]
            
            # Stop-loss
            df0_sl = df0[df0 < -pt_sl_.at[loc]]
            
            if len(df0_pt) > 0:
                out.loc[loc, 't1'] = df0_pt.index[0]  # Primeiro toque profit-taking
            elif len(df0_sl) > 0:
                out.loc[loc, 't1'] = df0_sl.index[0]  # Primeiro toque stop-loss
                
        return out
    
    def mp_pandas_obj(self, 
                      func: callable,
                      pd_obj: Union[pd.DataFrame, pd.Series],
                      num_threads: int = 1,
                      mp_batches: int = 1,
                      lin_mols: bool = True,
                      **kargs) -> Union[pd.DataFrame, pd.Series]:
        """
        Processamento paralelo para objetos pandas.
        
        Args:
            func: Função a ser aplicada
            pd_obj: Objeto pandas para processar
            num_threads: Número de threads
            mp_batches: Número de batches
            lin_mols: Se True, particiona linearmente
            **kargs: Argumentos adicionais para func
            
        Returns:
            Resultado concatenado do processamento paralelo
        """
        if lin_mols:
            parts = np.linspace(0, len(pd_obj.index), min(mp_batches, len(pd_obj.index)) + 1)
            parts = np.ceil(parts).astype(int)
        else:
            parts = np.array_split(np.arange(len(pd_obj.index)), mp_batches)
            
        jobs = []
        for i in range(1, len(parts)):
            job = {pd_obj.index[parts[i-1]:parts[i]]}
            job = list(job)[0]
            jobs.append(job)
            
        if num_threads == 1:
            out = pd.concat([func(job, **kargs) for job in jobs], sort=False)
        else:
            pool = mp.Pool(processes=num_threads)
            outputs = pool.map(partial(func, **kargs), jobs)
            out = pd.concat(outputs, sort=False)
            pool.close()
            pool.join()
            
        return out


def get_t1(close: pd.Series, t_events: pd.DatetimeIndex, num_days: int) -> pd.Series:
    """
    Calcula timestamps t1 baseados em número de dias úteis.
    
    Args:
        close: Série de preços
        t_events: Timestamps dos eventos
        num_days: Número de dias úteis para barreira de tempo
        
    Returns:
        Série com timestamps t1
    """
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])
    
    return t1


def get_bins_new(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    Versão otimizada de get_bins com melhor performance.
    
    Args:
        events: DataFrame de eventos
        close: Série de preços
        
    Returns:
        DataFrame com labels
    """
    # Implementação vetorizada para melhor performance
    events_ = events.dropna(subset=['t1'])
    out = events_[['t1', 'trgt', 'side']].copy(deep=True)
    
    px = {}
    for loc, t1 in events_['t1'].items():
        df0 = close.loc[loc:t1]
        
        if len(df0) <= 1:
            continue
            
        # Calcular retornos
        ret = df0.iloc[-1] / df0.iloc[0] - 1
        
        # Determinar label baseado no retorno e target
        if abs(ret) < events_.at[loc, 'trgt']:
            px[loc] = 0  # Sem movimento significativo
        else:
            px[loc] = np.sign(ret) * events_.at[loc, 'side']
            
    out['bin'] = pd.Series(px)
    
    return out.dropna(subset=['bin'])