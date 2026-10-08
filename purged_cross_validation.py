import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from typing import Generator, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta

class PurgedKFold(BaseCrossValidator):
    """
    Implementação de Purged Cross-Validation conforme López de Prado
    "Advances in Financial Machine Learning"
    
    Este método previne data leakage em séries temporais:
    1. Purging: Remove observações sobrepostas entre train/test
    2. Embargo: Adiciona gap temporal após o conjunto de teste
    3. Respeita ordem temporal dos dados
    """
    
    def __init__(self, n_splits: int = 5, t1: Optional[pd.Series] = None, 
                 pct_embargo: float = 0.01, logger: Optional[logging.Logger] = None,
                 purge_length: Optional[pd.Timedelta] = None,
                 embargo_length: Optional[pd.Timedelta] = None):
        """
        Args:
            n_splits: Número de splits para cross-validation
            t1: Série com timestamps de fim para cada observação
            pct_embargo: Percentual de embargo após conjunto de teste
            logger: Logger para debug
            purge_length: Duração do purging (compatibilidade)
            embargo_length: Duração do embargo (compatibilidade)
        """
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.purge_length = purge_length
        self.embargo_length = embargo_length
        self.logger = logger or logging.getLogger(__name__)
        
    def split(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Optional[Union[pd.Series, np.ndarray]] = None, 
              timestamps: Optional[pd.DatetimeIndex] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Gera índices para train/test splits com purging e embargo
        
        Args:
            X: Features (DataFrame ou array)
            y: Target (opcional)
            timestamps: Timestamps para os dados (terceiro parâmetro para compatibilidade)
            
        Yields:
            Tuplas (train_indices, test_indices)
        """
        if isinstance(X, pd.DataFrame):
            indices = X.index
        else:
            indices = np.arange(len(X))
            
        # Converter para array se necessário
        if isinstance(indices, pd.DatetimeIndex):
            timestamps = indices
        else:
            timestamps = pd.to_datetime(indices) if not isinstance(indices[0], (int, np.integer)) else None
            
        # Calcular tamanho de cada fold
        n_samples = len(indices)
        test_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Definir conjunto de teste
            test_start = i * test_size
            test_end = (i + 1) * test_size if i < self.n_splits - 1 else n_samples
            
            test_indices = np.arange(test_start, test_end)
            
            # Aplicar purging e embargo
            train_indices = self._get_train_times(
                indices, test_indices, self.t1, self.pct_embargo
            )
            
            self.logger.debug(f"Fold {i+1}: Train={len(train_indices)}, Test={len(test_indices)}")
            
            yield train_indices, test_indices
    
    def _get_train_times(self, times: Union[pd.Index, np.ndarray], 
                        test_indices: np.ndarray, 
                        t1: Optional[pd.Series] = None, 
                        pct_embargo: float = 0.01) -> np.ndarray:
        """
        Determina índices de treino aplicando purging e embargo
        
        Args:
            times: Índices temporais completos
            test_indices: Índices do conjunto de teste
            t1: Timestamps de fim para cada observação
            pct_embargo: Percentual de embargo
            
        Returns:
            Array com índices de treino válidos
        """
        if len(test_indices) == 0:
            return np.array([], dtype=int)
            
        # Obter timestamps de teste
        test_times = times[test_indices]
        
        if isinstance(test_times, pd.DatetimeIndex):
            test_start = test_times[0]
            test_end = test_times[-1]
        else:
            test_start = test_indices[0]
            test_end = test_indices[-1]
            
        # Calcular embargo
        if isinstance(times, pd.DatetimeIndex) and pct_embargo > 0:
            total_time = times[-1] - times[0]
            embargo_time = total_time * pct_embargo
            embargo_end = test_end + embargo_time
        else:
            embargo_samples = int(len(times) * pct_embargo)
            embargo_end = test_end + embargo_samples
            
        # Inicializar máscara de treino
        train_mask = np.ones(len(times), dtype=bool)
        
        # Aplicar purging se t1 fornecido
        if t1 is not None:
            train_mask = self._apply_purging(times, test_indices, t1, train_mask)
            
        # Aplicar embargo - remover observações após o teste
        if isinstance(times, pd.DatetimeIndex):
            train_mask &= (times < test_start) | (times > embargo_end)
        else:
            train_mask &= (np.arange(len(times)) < test_start) | (np.arange(len(times)) > embargo_end)
            
        return np.where(train_mask)[0]
    
    def _apply_purging(self, times: Union[pd.Index, np.ndarray], 
                      test_indices: np.ndarray, 
                      t1: pd.Series, 
                      train_mask: np.ndarray) -> np.ndarray:
        """
        Aplica purging removendo observações sobrepostas
        
        Args:
            times: Índices temporais
            test_indices: Índices de teste
            t1: Timestamps de fim
            train_mask: Máscara atual de treino
            
        Returns:
            Máscara atualizada após purging
        """
        # Obter período de teste
        test_times = times[test_indices]
        
        if isinstance(test_times, pd.DatetimeIndex):
            test_start = test_times[0]
            test_end = test_times[-1]
        else:
            test_start = test_indices[0]
            test_end = test_indices[-1]
            
        # Para cada observação de treino, verificar sobreposição
        for i, time in enumerate(times):
            if not train_mask[i]:
                continue
                
            # Obter fim da observação
            if time in t1.index:
                obs_end = t1[time]
            else:
                obs_end = time  # Assumir que termina no mesmo timestamp
                
            # Verificar sobreposição com período de teste
            if isinstance(times, pd.DatetimeIndex):
                if obs_end > test_start:
                    train_mask[i] = False
            else:
                # Para índices numéricos, usar posição
                obs_end_pos = np.where(times == obs_end)[0]
                if len(obs_end_pos) > 0 and obs_end_pos[0] > test_indices[0]:
                    train_mask[i] = False
                    
        return train_mask
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Retorna número de splits"""
        return self.n_splits

class TimeSeriesPurgedSplit:
    """
    Implementação alternativa focada em séries temporais financeiras
    Mais simples que PurgedKFold mas específica para dados financeiros
    """
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2, 
                 gap: int = 0, logger: Optional[logging.Logger] = None):
        """
        Args:
            n_splits: Número de splits
            test_size: Proporção do conjunto de teste
            gap: Número de observações de gap entre train/test
            logger: Logger para debug
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.logger = logger or logging.getLogger(__name__)
        
    def split(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Optional[Union[pd.Series, np.ndarray]] = None,
              timestamps: Optional[pd.DatetimeIndex] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Gera splits com gap temporal entre train/test
        
        Args:
            X: Features
            y: Target (opcional, para compatibilidade)
            timestamps: Timestamps (opcional, para compatibilidade)
            
        Yields:
            Tuplas (train_indices, test_indices)
        """
        n_samples = len(X)
        test_samples = int(n_samples * self.test_size)
        
        # Calcular posições dos splits
        total_samples_per_split = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Definir fim do conjunto de treino
            train_end = (i + 1) * total_samples_per_split - test_samples - self.gap
            
            if train_end <= 0:
                continue
                
            # Definir conjunto de teste
            test_start = train_end + self.gap
            test_end = test_start + test_samples
            
            if test_end > n_samples:
                test_end = n_samples
                test_start = test_end - test_samples
                
            # Criar índices
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            self.logger.debug(f"Split {i+1}: Train=[0:{train_end}], Test=[{test_start}:{test_end}], Gap={self.gap}")
            
            yield train_indices, test_indices

class CrossValidationAnalyzer:
    """
    Classe para analisar e comparar diferentes métodos de cross-validation
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def compare_cv_methods(self, X: pd.DataFrame, y: pd.Series, 
                          model, cv_methods: dict) -> pd.DataFrame:
        """
        Compara diferentes métodos de cross-validation
        
        Args:
            X: Features
            y: Target
            model: Modelo para treinar
            cv_methods: Dicionário {nome: cv_object}
            
        Returns:
            DataFrame com resultados comparativos
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.base import clone
        
        results = []
        
        for cv_name, cv_method in cv_methods.items():
            self.logger.info(f"Testando método: {cv_name}")
            
            scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
            
            try:
                for train_idx, test_idx in cv_method.split(X, y):
                    # Treinar modelo
                    model_clone = clone(model)
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    model_clone.fit(X_train, y_train)
                    y_pred = model_clone.predict(X_test)
                    
                    # Calcular métricas
                    scores['accuracy'].append(accuracy_score(y_test, y_pred))
                    scores['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                    scores['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                    scores['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                    
                # Calcular estatísticas
                for metric in scores:
                    results.append({
                        'cv_method': cv_name,
                        'metric': metric,
                        'mean': np.mean(scores[metric]),
                        'std': np.std(scores[metric]),
                        'min': np.min(scores[metric]),
                        'max': np.max(scores[metric])
                    })
                    
            except Exception as e:
                self.logger.error(f"Erro em {cv_name}: {e}")
                continue
                
        return pd.DataFrame(results)
    
    def analyze_temporal_leakage(self, X: pd.DataFrame, splits: list) -> dict:
        """
        Analisa vazamento temporal nos splits
        
        Args:
            X: DataFrame com índice temporal
            splits: Lista de tuplas (train_idx, test_idx)
            
        Returns:
            Dicionário com análise de vazamento
        """
        leakage_analysis = {
            'total_splits': len(splits),
            'leakage_detected': 0,
            'avg_gap_size': 0,
            'min_gap_size': float('inf'),
            'max_gap_size': 0
        }
        
        gaps = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
                
            # Verificar se há sobreposição temporal
            train_times = X.index[train_idx]
            test_times = X.index[test_idx]
            
            train_max = train_times.max()
            test_min = test_times.min()
            
            # Calcular gap
            if isinstance(X.index, pd.DatetimeIndex):
                gap = (test_min - train_max).total_seconds() / 3600  # Gap em horas
            else:
                gap = test_min - train_max
                
            gaps.append(gap)
            
            # Detectar vazamento (gap negativo)
            if gap < 0:
                leakage_analysis['leakage_detected'] += 1
                self.logger.warning(f"Vazamento detectado no split {i+1}: gap={gap}")
                
        if gaps:
            leakage_analysis['avg_gap_size'] = np.mean(gaps)
            leakage_analysis['min_gap_size'] = np.min(gaps)
            leakage_analysis['max_gap_size'] = np.max(gaps)
            
        return leakage_analysis

# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Criar dados de exemplo
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    np.random.seed(42)
    
    X = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000)
    }, index=dates)
    
    y = pd.Series(np.random.choice([0, 1], 1000), index=dates)
    
    # Criar t1 (fim das observações)
    t1 = pd.Series(dates + pd.Timedelta(hours=2), index=dates)
    
    # Testar PurgedKFold
    purged_cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.02, logger=logger)
    
    print("Testando PurgedKFold:")
    for i, (train_idx, test_idx) in enumerate(purged_cv.split(X, y)):
        print(f"Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
        
    # Testar TimeSeriesPurgedSplit
    ts_cv = TimeSeriesPurgedSplit(n_splits=5, test_size=0.2, gap=10, logger=logger)
    
    print("\nTestando TimeSeriesPurgedSplit:")
    for i, (train_idx, test_idx) in enumerate(ts_cv.split(X)):
        print(f"Split {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
        
    # Analisar vazamento temporal
    analyzer = CrossValidationAnalyzer(logger)
    splits = list(purged_cv.split(X, y))
    leakage_analysis = analyzer.analyze_temporal_leakage(X, splits)
    
    print("\nAnálise de Vazamento Temporal:")
    for key, value in leakage_analysis.items():
        print(f"{key}: {value}")