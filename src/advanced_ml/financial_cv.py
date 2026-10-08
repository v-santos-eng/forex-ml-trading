import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union, Iterator
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
from itertools import combinations
import matplotlib.pyplot as plt
from datetime import timedelta


class PurgedKFold(BaseCrossValidator):
    """
    Implementação do Purged K-Fold Cross-Validation para dados financeiros.
    
    Baseado no trabalho de López de Prado, este método:
    1. Remove observações que podem causar vazamento de informações
    2. Implementa embargo para evitar contaminação temporal
    3. Considera sobreposição de labels temporais
    
    O método é essencial para validação adequada em dados financeiros onde
    observações adjacentes podem estar correlacionadas.
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 t1: Optional[pd.Series] = None,
                 pct_embargo: float = 0.01):
        """
        Inicializa o Purged K-Fold CV.
        
        Args:
            n_splits: Número de folds
            t1: Série com timestamps de fim de cada observação
            pct_embargo: Percentual de embargo após cada fold de teste
        """
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
    
    def split(self, 
              X: pd.DataFrame, 
              y: Optional[pd.Series] = None, 
              groups: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Gera índices de treino e teste para cada fold.
        
        Args:
            X: Features
            y: Labels (opcional)
            groups: Grupos (não usado)
            
        Yields:
            Tuplas com (train_indices, test_indices)
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X deve ter DatetimeIndex para Purged K-Fold")
        
        # Usar t1 fornecido ou assumir que cada observação termina no mesmo timestamp
        if self.t1 is not None:
            t1 = self.t1.reindex(X.index).fillna(X.index)
        else:
            t1 = pd.Series(X.index, index=X.index)
        
        # Dividir dados em folds temporais
        test_ranges = self._get_test_ranges(X.index)
        
        for test_start, test_end in test_ranges:
            # Índices de teste
            test_indices = X.index[(X.index >= test_start) & (X.index <= test_end)]
            
            # Purgar observações que se sobrepõem com período de teste
            train_indices = self._get_train_times(X.index, t1, test_indices)
            
            # Aplicar embargo
            if self.pct_embargo > 0:
                train_indices = self._apply_embargo(train_indices, test_indices, X.index)
            
            # Converter para posições numéricas
            train_pos = X.index.get_indexer(train_indices)
            test_pos = X.index.get_indexer(test_indices)
            
            # Filtrar índices inválidos
            train_pos = train_pos[train_pos >= 0]
            test_pos = test_pos[test_pos >= 0]
            
            if len(train_pos) > 0 and len(test_pos) > 0:
                yield train_pos, test_pos
    
    def _get_test_ranges(self, index: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Calcula os ranges temporais para cada fold de teste.
        
        Args:
            index: Índice temporal dos dados
            
        Returns:
            Lista de tuplas (início, fim) para cada fold
        """
        start_time = index.min()
        end_time = index.max()
        total_duration = end_time - start_time
        fold_duration = total_duration / self.n_splits
        
        ranges = []
        for i in range(self.n_splits):
            fold_start = start_time + i * fold_duration
            fold_end = start_time + (i + 1) * fold_duration
            
            # Ajustar para o último fold
            if i == self.n_splits - 1:
                fold_end = end_time
            
            ranges.append((fold_start, fold_end))
        
        return ranges
    
    def _get_train_times(self, 
                        times: pd.DatetimeIndex,
                        t1: pd.Series,
                        test_times: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Determina tempos de treino removendo sobreposições com teste.
        
        Args:
            times: Todos os timestamps disponíveis
            t1: Timestamps de fim de cada observação
            test_times: Timestamps do conjunto de teste
            
        Returns:
            Timestamps válidos para treino
        """
        if len(test_times) == 0:
            return times
        
        test_start = test_times.min()
        test_end = test_times.max()
        
        # Remover observações que se sobrepõem com período de teste
        train_mask = pd.Series(True, index=times)
        
        for time in times:
            # Verificar se a observação se sobrepõe com período de teste
            obs_end = t1.get(time, time)
            
            # Remover se:
            # 1. A observação começa durante o período de teste
            # 2. A observação termina durante o período de teste
            # 3. A observação engloba o período de teste
            if (test_start <= time <= test_end or 
                test_start <= obs_end <= test_end or
                (time <= test_start and obs_end >= test_end)):
                train_mask[time] = False
        
        return times[train_mask]
    
    def _apply_embargo(self, 
                      train_times: pd.DatetimeIndex,
                      test_times: pd.DatetimeIndex,
                      all_times: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Aplica embargo removendo observações próximas ao período de teste.
        
        Args:
            train_times: Timestamps de treino
            test_times: Timestamps de teste
            all_times: Todos os timestamps
            
        Returns:
            Timestamps de treino após embargo
        """
        if len(test_times) == 0:
            return train_times
        
        # Calcular duração do embargo
        total_duration = all_times.max() - all_times.min()
        embargo_duration = total_duration * self.pct_embargo
        
        test_end = test_times.max()
        embargo_end = test_end + embargo_duration
        
        # Remover observações dentro do período de embargo
        embargo_mask = train_times <= embargo_end
        
        return train_times[~embargo_mask]
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Retorna o número de folds."""
        return self.n_splits


class CombinatorialPurgedCV(BaseCrossValidator):
    """
    Implementação do Combinatorial Purged Cross-Validation.
    
    Este método gera todas as combinações possíveis de folds de teste,
    proporcionando uma validação mais robusta mas computacionalmente intensiva.
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 n_test_splits: int = 2,
                 t1: Optional[pd.Series] = None,
                 pct_embargo: float = 0.01):
        """
        Inicializa o Combinatorial Purged CV.
        
        Args:
            n_splits: Número total de folds
            n_test_splits: Número de folds usados para teste em cada iteração
            t1: Série com timestamps de fim
            pct_embargo: Percentual de embargo
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
        
        # Calcular número de combinações
        from math import comb
        self.n_combinations = comb(n_splits, n_test_splits)
    
    def split(self, 
              X: pd.DataFrame, 
              y: Optional[pd.Series] = None, 
              groups: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Gera todas as combinações de folds de teste.
        
        Args:
            X: Features
            y: Labels (opcional)
            groups: Grupos (não usado)
            
        Yields:
            Tuplas com (train_indices, test_indices)
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X deve ter DatetimeIndex")
        
        # Criar folds base
        base_cv = PurgedKFold(n_splits=self.n_splits, t1=self.t1, pct_embargo=0)
        base_splits = list(base_cv.split(X, y))
        
        # Gerar todas as combinações de folds de teste
        for test_fold_indices in combinations(range(self.n_splits), self.n_test_splits):
            # Combinar folds de teste selecionados
            test_indices = np.concatenate([base_splits[i][1] for i in test_fold_indices])
            
            # Usar folds restantes para treino
            train_fold_indices = [i for i in range(self.n_splits) if i not in test_fold_indices]
            train_indices = np.concatenate([base_splits[i][0] for i in train_fold_indices])
            
            # Aplicar embargo se necessário
            if self.pct_embargo > 0:
                test_times = X.index[test_indices]
                train_times = X.index[train_indices]
                
                # Filtrar treino com embargo
                purged_cv = PurgedKFold(n_splits=1, t1=self.t1, pct_embargo=self.pct_embargo)
                train_times_filtered = purged_cv._apply_embargo(train_times, test_times, X.index)
                train_indices = X.index.get_indexer(train_times_filtered)
                train_indices = train_indices[train_indices >= 0]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Retorna o número de combinações."""
        return self.n_combinations


class FinancialCrossValidator:
    """
    Classe principal para validação cruzada em dados financeiros.
    
    Integra diferentes métodos de CV e fornece análise de performance.
    """
    
    def __init__(self, 
                 method: str = 'purged_kfold',
                 n_splits: int = 5,
                 pct_embargo: float = 0.01,
                 **kwargs):
        """
        Inicializa o validador financeiro.
        
        Args:
            method: Método de CV ('purged_kfold' ou 'combinatorial')
            n_splits: Número de folds
            pct_embargo: Percentual de embargo
            **kwargs: Argumentos adicionais para o método específico
        """
        self.method = method
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo
        self.kwargs = kwargs
        
        self.cv_results = None
        self.feature_importance = None
    
    def _get_cv_splitter(self, t1: Optional[pd.Series] = None) -> BaseCrossValidator:
        """
        Retorna o splitter de CV apropriado.
        
        Args:
            t1: Série com timestamps de fim
            
        Returns:
            Objeto de CV
        """
        if self.method == 'purged_kfold':
            return PurgedKFold(
                n_splits=self.n_splits,
                t1=t1,
                pct_embargo=self.pct_embargo
            )
        elif self.method == 'combinatorial':
            return CombinatorialPurgedCV(
                n_splits=self.n_splits,
                t1=t1,
                pct_embargo=self.pct_embargo,
                **self.kwargs
            )
        else:
            raise ValueError(f"Método {self.method} não suportado")
    
    def cross_validate(self, 
                      estimator: BaseEstimator,
                      X: pd.DataFrame,
                      y: pd.Series,
                      t1: Optional[pd.Series] = None,
                      scoring: List[str] = None,
                      return_train_score: bool = False) -> dict:
        """
        Executa validação cruzada financeira.
        
        Args:
            estimator: Modelo a ser validado
            X: Features
            y: Labels
            t1: Timestamps de fim das observações
            scoring: Métricas a calcular
            return_train_score: Se deve retornar scores de treino
            
        Returns:
            Dicionário com resultados
        """
        if scoring is None:
            scoring = ['accuracy', 'f1', 'precision', 'recall']
        
        # Obter splitter
        cv_splitter = self._get_cv_splitter(t1)
        
        # Inicializar resultados
        results = {
            'test_scores': {metric: [] for metric in scoring},
            'train_scores': {metric: [] for metric in scoring} if return_train_score else None,
            'fit_times': [],
            'score_times': [],
            'fold_sizes': {'train': [], 'test': []}
        }
        
        # Executar CV
        fold_idx = 0
        for train_idx, test_idx in cv_splitter.split(X, y):
            fold_idx += 1
            print(f"Processando fold {fold_idx}/{cv_splitter.get_n_splits()}...")
            
            # Dividir dados
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Treinar modelo
            import time
            start_time = time.time()
            estimator.fit(X_train, y_train)
            fit_time = time.time() - start_time
            
            # Fazer predições
            start_time = time.time()
            y_pred_test = estimator.predict(X_test)
            if return_train_score:
                y_pred_train = estimator.predict(X_train)
            score_time = time.time() - start_time
            
            # Calcular métricas
            for metric in scoring:
                test_score = self._calculate_metric(metric, y_test, y_pred_test)
                results['test_scores'][metric].append(test_score)
                
                if return_train_score:
                    train_score = self._calculate_metric(metric, y_train, y_pred_train)
                    results['train_scores'][metric].append(train_score)
            
            # Armazenar tempos e tamanhos
            results['fit_times'].append(fit_time)
            results['score_times'].append(score_time)
            results['fold_sizes']['train'].append(len(X_train))
            results['fold_sizes']['test'].append(len(X_test))
        
        # Converter para arrays numpy
        for metric in scoring:
            results['test_scores'][metric] = np.array(results['test_scores'][metric])
            if return_train_score:
                results['train_scores'][metric] = np.array(results['train_scores'][metric])
        
        results['fit_times'] = np.array(results['fit_times'])
        results['score_times'] = np.array(results['score_times'])
        
        self.cv_results = results
        return results
    
    def _calculate_metric(self, metric: str, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Calcula métrica específica.
        
        Args:
            metric: Nome da métrica
            y_true: Labels verdadeiros
            y_pred: Predições
            
        Returns:
            Valor da métrica
        """
        try:
            if metric == 'accuracy':
                return accuracy_score(y_true, y_pred)
            elif metric == 'f1':
                return f1_score(y_true, y_pred, average='weighted')
            elif metric == 'precision':
                return precision_score(y_true, y_pred, average='weighted')
            elif metric == 'recall':
                return recall_score(y_true, y_pred, average='weighted')
            else:
                warnings.warn(f"Métrica {metric} não reconhecida")
                return np.nan
        except Exception as e:
            warnings.warn(f"Erro ao calcular {metric}: {e}")
            return np.nan
    
    def plot_cv_results(self, metric: str = 'f1') -> None:
        """
        Plota resultados da validação cruzada.
        
        Args:
            metric: Métrica para plotar
        """
        if self.cv_results is None:
            raise ValueError("Execute cross_validate primeiro")
        
        if metric not in self.cv_results['test_scores']:
            raise ValueError(f"Métrica {metric} não encontrada nos resultados")
        
        test_scores = self.cv_results['test_scores'][metric]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Scores por fold
        folds = range(1, len(test_scores) + 1)
        ax1.plot(folds, test_scores, 'bo-', linewidth=2, markersize=8)
        ax1.axhline(y=test_scores.mean(), color='r', linestyle='--', 
                   label=f'Média: {test_scores.mean():.4f}')
        ax1.fill_between(folds, 
                        test_scores.mean() - test_scores.std(),
                        test_scores.mean() + test_scores.std(),
                        alpha=0.2, color='red')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel(f'{metric.upper()} Score')
        ax1.set_title(f'Scores de {metric.upper()} por Fold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribuição dos scores
        ax2.hist(test_scores, bins=min(10, len(test_scores)), alpha=0.7, color='blue')
        ax2.axvline(x=test_scores.mean(), color='r', linestyle='--', 
                   label=f'Média: {test_scores.mean():.4f}')
        ax2.axvline(x=test_scores.mean() - test_scores.std(), color='orange', 
                   linestyle=':', label=f'±1 Std: {test_scores.std():.4f}')
        ax2.axvline(x=test_scores.mean() + test_scores.std(), color='orange', linestyle=':')
        ax2.set_xlabel(f'{metric.upper()} Score')
        ax2.set_ylabel('Frequência')
        ax2.set_title(f'Distribuição dos Scores de {metric.upper()}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_cv_summary(self) -> pd.DataFrame:
        """
        Retorna resumo dos resultados de CV.
        
        Returns:
            DataFrame com estatísticas resumidas
        """
        if self.cv_results is None:
            raise ValueError("Execute cross_validate primeiro")
        
        summary_data = []
        
        for metric, scores in self.cv_results['test_scores'].items():
            summary_data.append({
                'Métrica': metric.upper(),
                'Média': scores.mean(),
                'Std': scores.std(),
                'Min': scores.min(),
                'Max': scores.max(),
                'CV': scores.std() / scores.mean() if scores.mean() != 0 else np.inf
            })
        
        return pd.DataFrame(summary_data)
    
    def compare_with_standard_cv(self, 
                                estimator: BaseEstimator,
                                X: pd.DataFrame,
                                y: pd.Series,
                                standard_cv_splits: int = 5) -> dict:
        """
        Compara resultados com CV padrão (não-purged).
        
        Args:
            estimator: Modelo a ser testado
            X: Features
            y: Labels
            standard_cv_splits: Número de folds para CV padrão
            
        Returns:
            Dicionário com comparação
        """
        from sklearn.model_selection import KFold, cross_val_score
        
        # CV financeiro (purged)
        financial_results = self.cross_validate(estimator, X, y)
        
        # CV padrão
        standard_cv = KFold(n_splits=standard_cv_splits, shuffle=False)
        standard_scores = cross_val_score(estimator, X, y, cv=standard_cv, scoring='f1_weighted')
        
        comparison = {
            'financial_cv': {
                'mean_f1': financial_results['test_scores']['f1'].mean(),
                'std_f1': financial_results['test_scores']['f1'].std(),
                'scores': financial_results['test_scores']['f1']
            },
            'standard_cv': {
                'mean_f1': standard_scores.mean(),
                'std_f1': standard_scores.std(),
                'scores': standard_scores
            }
        }
        
        # Calcular diferença
        comparison['difference'] = {
            'mean_f1_diff': comparison['financial_cv']['mean_f1'] - comparison['standard_cv']['mean_f1'],
            'std_f1_diff': comparison['financial_cv']['std_f1'] - comparison['standard_cv']['std_f1']
        }
        
        return comparison


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados de exemplo
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Features sintéticas
    X = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000)
    }, index=dates)
    
    # Labels sintéticos com alguma estrutura temporal
    y = pd.Series(np.random.choice([0, 1], 1000, p=[0.6, 0.4]), index=dates)
    
    # Timestamps de fim (exemplo: cada observação dura 3 dias)
    t1 = pd.Series(dates + timedelta(days=3), index=dates)
    
    # Criar validador
    cv = FinancialCrossValidator(method='purged_kfold', n_splits=5, pct_embargo=0.02)
    
    # Modelo de exemplo
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Executar validação cruzada
    results = cv.cross_validate(model, X, y, t1)
    
    # Mostrar resultados
    print(cv.get_cv_summary())
    cv.plot_cv_results('f1')