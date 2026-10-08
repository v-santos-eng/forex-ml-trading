import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from scipy.special import gamma
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from numba import jit, prange
import seaborn as sns

@dataclass
class FractionalDiffConfig:
    """Configuração para Fractional Differentiation"""
    
    # Parâmetros de diferenciação
    d_range: Tuple[float, float] = (0.0, 1.0)  # Range para busca do d ótimo
    threshold: float = 1e-5  # Threshold para truncar pesos
    max_window: int = 100    # Janela máxima para cálculo
    
    # Parâmetros de otimização
    optimize_d: bool = True  # Otimizar d automaticamente
    target_adf_pvalue: float = 0.05  # P-value alvo para ADF test
    min_memory_preservation: float = 0.95  # Mínima preservação de memória
    
    # Parâmetros de performance
    use_numba: bool = True   # Usar aceleração Numba
    parallel_processing: bool = True
    n_jobs: int = -1
    
    # Parâmetros de validação
    validate_stationarity: bool = True
    validate_memory: bool = True
    
    # Logging
    verbose: bool = True

class FractionalDifferentiator:
    """
    Implementação de Fractional Differentiation baseada em "Advances in Financial Machine Learning"
    
    A diferenciação fracionária permite:
    1. Tornar séries estacionárias preservando memória
    2. Controlar o trade-off entre estacionariedade e preservação de informação
    3. Melhorar a performance de modelos ML em séries temporais financeiras
    
    Funcionalidades:
    - Cálculo eficiente de pesos fracionários
    - Otimização automática do parâmetro d
    - Validação de estacionariedade e preservação de memória
    - Processamento paralelo para múltiplas séries
    - Análise de sensibilidade
    - Visualizações avançadas
    """
    
    def __init__(self, 
                 config: Optional[FractionalDiffConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            config: Configuração da diferenciação fracionária
            logger: Logger para debug
        """
        self.config = config or FractionalDiffConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache para pesos
        self._weights_cache = {}
        
        # Resultados da otimização
        self.optimization_results = {}
        
    def get_weights(self, d: float, size: int) -> np.ndarray:
        """
        Calcula pesos para diferenciação fracionária
        
        Args:
            d: Parâmetro de diferenciação fracionária
            size: Número de pesos a calcular
            
        Returns:
            Array com os pesos
        """
        # Verificar cache
        cache_key = (d, size)
        if cache_key in self._weights_cache:
            return self._weights_cache[cache_key]
            
        if self.config.use_numba:
            weights = self._get_weights_numba(d, size, self.config.threshold)
        else:
            weights = self._get_weights_python(d, size)
            
        # Cache resultado
        self._weights_cache[cache_key] = weights
        
        return weights
    
    @staticmethod
    @jit(nopython=True)
    def _get_weights_numba(d: float, size: int, threshold: float) -> np.ndarray:
        """
        Versão otimizada com Numba para cálculo de pesos
        """
        weights = np.zeros(size)
        weights[0] = 1.0
        
        for k in range(1, size):
            weight = -weights[k-1] * (d - k + 1) / k
            
            if abs(weight) < threshold:
                break
                
            weights[k] = weight
            
        return weights
    
    def _get_weights_python(self, d: float, size: int) -> np.ndarray:
        """
        Versão Python pura para cálculo de pesos
        """
        weights = np.zeros(size)
        weights[0] = 1.0
        
        for k in range(1, size):
            weight = -weights[k-1] * (d - k + 1) / k
            
            if abs(weight) < self.config.threshold:
                break
                
            weights[k] = weight
            
        return weights
    
    def frac_diff(self, series: pd.Series, d: float, 
                  window: Optional[int] = None) -> pd.Series:
        """
        Aplica diferenciação fracionária a uma série
        
        Args:
            series: Série temporal
            d: Parâmetro de diferenciação
            window: Janela para cálculo (None = usar série completa)
            
        Returns:
            Série diferenciada fracionalmente
        """
        if window is None:
            window = min(len(series), self.config.max_window)
            
        # Obter pesos
        weights = self.get_weights(d, window)
        
        # Aplicar convolução
        if self.config.use_numba:
            result = self._apply_frac_diff_numba(
                series.values, weights, len(series)
            )
        else:
            result = self._apply_frac_diff_python(series.values, weights)
            
        # Criar série resultado
        frac_diff_series = pd.Series(
            result, 
            index=series.index,
            name=f"{series.name}_fracdiff_{d:.3f}" if series.name else f"fracdiff_{d:.3f}"
        )
        
        return frac_diff_series.dropna()
    
    def fractional_diff(self, series: pd.Series, d: float, 
                       window: Optional[int] = None) -> pd.Series:
        """
        Método público para diferenciação fracionária (alias para frac_diff)
        
        Args:
            series: Série temporal
            d: Parâmetro de diferenciação
            window: Janela para cálculo (None = usar série completa)
            
        Returns:
            Série diferenciada fracionalmente
        """
        return self.frac_diff(series, d, window)
    
    @staticmethod
    @jit(nopython=True)
    def _apply_frac_diff_numba(values: np.ndarray, weights: np.ndarray, 
                              n: int) -> np.ndarray:
        """
        Aplica diferenciação fracionária com Numba
        """
        result = np.full(n, np.nan)
        w_len = len(weights)
        
        for i in range(w_len - 1, n):
            result[i] = 0.0
            for j in range(w_len):
                if i - j >= 0:
                    result[i] += weights[j] * values[i - j]
                    
        return result
    
    def _apply_frac_diff_python(self, values: np.ndarray, 
                               weights: np.ndarray) -> np.ndarray:
        """
        Aplica diferenciação fracionária em Python puro
        """
        result = np.full(len(values), np.nan)
        w_len = len(weights)
        
        for i in range(w_len - 1, len(values)):
            result[i] = np.dot(weights, values[i - w_len + 1:i + 1][::-1])
            
        return result
    
    def find_optimal_d(self, series: pd.Series, 
                      method: str = 'adf_memory') -> Dict[str, Any]:
        """
        Encontra o valor ótimo de d para uma série
        
        Args:
            series: Série temporal
            method: Método de otimização ('adf', 'memory', 'adf_memory')
            
        Returns:
            Dicionário com resultados da otimização
        """
        if self.config.verbose:
            self.logger.info(f"Otimizando d para série {series.name} usando método {method}")
            
        if method == 'adf':
            return self._optimize_d_adf(series)
        elif method == 'memory':
            return self._optimize_d_memory(series)
        elif method == 'adf_memory':
            return self._optimize_d_adf_memory(series)
        else:
            raise ValueError(f"Método desconhecido: {method}")
    
    def _optimize_d_adf(self, series: pd.Series) -> Dict[str, Any]:
        """
        Otimiza d baseado no teste ADF
        """
        def objective(d):
            try:
                frac_diff_series = self.frac_diff(series, d)
                if len(frac_diff_series) < 10:
                    return 1.0  # Penalizar d que resulta em séries muito curtas
                    
                adf_stat, adf_pvalue = adfuller(frac_diff_series.dropna())[:2]
                return adf_pvalue
            except:
                return 1.0
                
        # Otimização
        result = minimize_scalar(
            objective,
            bounds=self.config.d_range,
            method='bounded'
        )
        
        optimal_d = result.x
        optimal_pvalue = result.fun
        
        # Calcular métricas adicionais
        frac_diff_series = self.frac_diff(series, optimal_d)
        memory_preservation = self._calculate_memory_preservation(series, frac_diff_series)
        
        return {
            'optimal_d': optimal_d,
            'adf_pvalue': optimal_pvalue,
            'memory_preservation': memory_preservation,
            'is_stationary': optimal_pvalue < self.config.target_adf_pvalue,
            'optimization_success': result.success,
            'method': 'adf'
        }
    
    def _optimize_d_memory(self, series: pd.Series) -> Dict[str, Any]:
        """
        Otimiza d baseado na preservação de memória
        """
        def objective(d):
            try:
                frac_diff_series = self.frac_diff(series, d)
                if len(frac_diff_series) < 10:
                    return -1.0  # Penalizar
                    
                memory = self._calculate_memory_preservation(series, frac_diff_series)
                return -memory  # Maximizar memória (minimizar negativo)
            except:
                return -1.0
                
        # Otimização
        result = minimize_scalar(
            objective,
            bounds=self.config.d_range,
            method='bounded'
        )
        
        optimal_d = result.x
        optimal_memory = -result.fun
        
        # Calcular ADF
        frac_diff_series = self.frac_diff(series, optimal_d)
        try:
            adf_stat, adf_pvalue = adfuller(frac_diff_series.dropna())[:2]
        except:
            adf_pvalue = 1.0
            
        return {
            'optimal_d': optimal_d,
            'adf_pvalue': adf_pvalue,
            'memory_preservation': optimal_memory,
            'is_stationary': adf_pvalue < self.config.target_adf_pvalue,
            'optimization_success': result.success,
            'method': 'memory'
        }
    
    def _optimize_d_adf_memory(self, series: pd.Series) -> Dict[str, Any]:
        """
        Otimiza d balanceando ADF e preservação de memória
        """
        def objective(d):
            try:
                frac_diff_series = self.frac_diff(series, d)
                if len(frac_diff_series) < 10:
                    return 10.0  # Penalizar
                    
                # ADF test
                adf_stat, adf_pvalue = adfuller(frac_diff_series.dropna())[:2]
                
                # Memory preservation
                memory = self._calculate_memory_preservation(series, frac_diff_series)
                
                # Função objetivo combinada
                # Penalizar p-value alto (não estacionário) e baixa memória
                adf_penalty = max(0, adf_pvalue - self.config.target_adf_pvalue)
                memory_penalty = max(0, self.config.min_memory_preservation - memory)
                
                return adf_penalty + memory_penalty
                
            except:
                return 10.0
                
        # Otimização
        result = minimize_scalar(
            objective,
            bounds=self.config.d_range,
            method='bounded'
        )
        
        optimal_d = result.x
        
        # Calcular métricas finais
        frac_diff_series = self.frac_diff(series, optimal_d)
        try:
            adf_stat, adf_pvalue = adfuller(frac_diff_series.dropna())[:2]
        except:
            adf_pvalue = 1.0
            
        memory_preservation = self._calculate_memory_preservation(series, frac_diff_series)
        
        return {
            'optimal_d': optimal_d,
            'adf_pvalue': adf_pvalue,
            'memory_preservation': memory_preservation,
            'is_stationary': adf_pvalue < self.config.target_adf_pvalue,
            'meets_memory_threshold': memory_preservation >= self.config.min_memory_preservation,
            'optimization_success': result.success,
            'objective_value': result.fun,
            'method': 'adf_memory'
        }
    
    def _calculate_memory_preservation(self, original: pd.Series, 
                                     transformed: pd.Series) -> float:
        """
        Calcula preservação de memória usando correlação
        
        Args:
            original: Série original
            transformed: Série transformada
            
        Returns:
            Correlação entre as séries (proxy para preservação de memória)
        """
        try:
            # Alinhar séries
            aligned_original = original.loc[transformed.index]
            
            # Calcular correlação
            correlation = aligned_original.corr(transformed)
            
            # Retornar correlação absoluta
            return abs(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Erro no cálculo de preservação de memória: {e}")
            return 0.0
    
    def process_multiple_series(self, data: pd.DataFrame, 
                              d_values: Optional[Dict[str, float]] = None,
                              optimize_d: bool = None) -> pd.DataFrame:
        """
        Processa múltiplas séries com diferenciação fracionária
        
        Args:
            data: DataFrame com séries temporais
            d_values: Dicionário com valores de d por coluna (opcional)
            optimize_d: Se deve otimizar d (sobrescreve config)
            
        Returns:
            DataFrame com séries diferenciadas
        """
        if optimize_d is None:
            optimize_d = self.config.optimize_d
            
        results = {}
        
        if self.config.parallel_processing and len(data.columns) > 1:
            # Processamento paralelo
            with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
                futures = {}
                
                for col in data.columns:
                    d_val = d_values.get(col, None) if d_values else None
                    future = executor.submit(
                        self._process_single_series, 
                        data[col], col, d_val, optimize_d
                    )
                    futures[col] = future
                    
                for col, future in futures.items():
                    try:
                        result = future.result()
                        results[col] = result
                    except Exception as e:
                        self.logger.error(f"Erro processando coluna {col}: {e}")
        else:
            # Processamento sequencial
            for col in data.columns:
                try:
                    d_val = d_values.get(col, None) if d_values else None
                    result = self._process_single_series(
                        data[col], col, d_val, optimize_d
                    )
                    results[col] = result
                except Exception as e:
                    self.logger.error(f"Erro processando coluna {col}: {e}")
                    
        # Combinar resultados
        if results:
            frac_diff_data = pd.DataFrame(results)
            return frac_diff_data
        else:
            return pd.DataFrame()
    
    def _process_single_series(self, series: pd.Series, col_name: str,
                             d_value: Optional[float], 
                             optimize_d: bool) -> pd.Series:
        """
        Processa uma única série
        """
        if optimize_d and d_value is None:
            # Otimizar d
            opt_result = self.find_optimal_d(series)
            d_value = opt_result['optimal_d']
            
            # Armazenar resultado da otimização
            self.optimization_results[col_name] = opt_result
            
            if self.config.verbose:
                self.logger.info(
                    f"Coluna {col_name}: d={d_value:.4f}, "
                    f"ADF p-value={opt_result['adf_pvalue']:.4f}, "
                    f"Memory={opt_result['memory_preservation']:.4f}"
                )
        elif d_value is None:
            d_value = 0.5  # Valor padrão
            
        # Aplicar diferenciação fracionária
        frac_diff_series = self.frac_diff(series, d_value)
        
        return frac_diff_series
    
    def analyze_sensitivity(self, series: pd.Series, 
                          d_range: Optional[Tuple[float, float]] = None,
                          n_points: int = 20) -> Dict[str, Any]:
        """
        Análise de sensibilidade do parâmetro d
        
        Args:
            series: Série temporal
            d_range: Range de valores d para testar
            n_points: Número de pontos a testar
            
        Returns:
            Dicionário com resultados da análise
        """
        if d_range is None:
            d_range = self.config.d_range
            
        d_values = np.linspace(d_range[0], d_range[1], n_points)
        
        results = {
            'd_values': d_values,
            'adf_pvalues': [],
            'memory_preservation': [],
            'series_length': []
        }
        
        for d in d_values:
            try:
                # Aplicar diferenciação
                frac_diff_series = self.frac_diff(series, d)
                
                # ADF test
                try:
                    adf_stat, adf_pvalue = adfuller(frac_diff_series.dropna())[:2]
                except:
                    adf_pvalue = 1.0
                    
                # Preservação de memória
                memory = self._calculate_memory_preservation(series, frac_diff_series)
                
                results['adf_pvalues'].append(adf_pvalue)
                results['memory_preservation'].append(memory)
                results['series_length'].append(len(frac_diff_series.dropna()))
                
            except Exception as e:
                self.logger.warning(f"Erro na análise de sensibilidade para d={d}: {e}")
                results['adf_pvalues'].append(1.0)
                results['memory_preservation'].append(0.0)
                results['series_length'].append(0)
                
        return results
    
    def plot_sensitivity_analysis(self, series: pd.Series, 
                                save_plot: bool = False,
                                filename: Optional[str] = None) -> None:
        """
        Plota análise de sensibilidade
        """
        # Executar análise
        sensitivity = self.analyze_sensitivity(series)
        
        # Criar plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Fractional Differentiation Sensitivity Analysis - {series.name}', 
                    fontsize=16)
        
        # 1. ADF p-values
        ax1 = axes[0, 0]
        ax1.plot(sensitivity['d_values'], sensitivity['adf_pvalues'], 'b-o', markersize=4)
        ax1.axhline(y=0.05, color='r', linestyle='--', label='5% significance')
        ax1.set_xlabel('d parameter')
        ax1.set_ylabel('ADF p-value')
        ax1.set_title('Stationarity (ADF Test)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Memory preservation
        ax2 = axes[0, 1]
        ax2.plot(sensitivity['d_values'], sensitivity['memory_preservation'], 'g-o', markersize=4)
        ax2.axhline(y=self.config.min_memory_preservation, color='r', linestyle='--', 
                   label=f'{self.config.min_memory_preservation:.1%} threshold')
        ax2.set_xlabel('d parameter')
        ax2.set_ylabel('Memory Preservation')
        ax2.set_title('Memory Preservation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Series length
        ax3 = axes[1, 0]
        ax3.plot(sensitivity['d_values'], sensitivity['series_length'], 'm-o', markersize=4)
        ax3.set_xlabel('d parameter')
        ax3.set_ylabel('Series Length')
        ax3.set_title('Resulting Series Length')
        ax3.grid(True, alpha=0.3)
        
        # 4. Combined score
        ax4 = axes[1, 1]
        
        # Calcular score combinado
        adf_scores = [1 - p if p < 0.05 else 0 for p in sensitivity['adf_pvalues']]
        memory_scores = sensitivity['memory_preservation']
        combined_scores = [(a + m) / 2 for a, m in zip(adf_scores, memory_scores)]
        
        ax4.plot(sensitivity['d_values'], combined_scores, 'r-o', markersize=4)
        
        # Marcar ótimo
        best_idx = np.argmax(combined_scores)
        best_d = sensitivity['d_values'][best_idx]
        best_score = combined_scores[best_idx]
        
        ax4.axvline(x=best_d, color='orange', linestyle='--', 
                   label=f'Optimal d = {best_d:.3f}')
        ax4.scatter([best_d], [best_score], color='orange', s=100, zorder=5)
        
        ax4.set_xlabel('d parameter')
        ax4.set_ylabel('Combined Score')
        ax4.set_title('Combined Score (Stationarity + Memory)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            if filename is None:
                filename = f"fractional_diff_sensitivity_{series.name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def plot_comparison(self, original: pd.Series, d_value: float,
                      save_plot: bool = False,
                      filename: Optional[str] = None) -> None:
        """
        Plota comparação entre série original e diferenciada
        """
        # Aplicar diferenciação
        frac_diff_series = self.frac_diff(original, d_value)
        
        # Criar plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Fractional Differentiation Comparison (d={d_value:.3f})', 
                    fontsize=16)
        
        # 1. Séries temporais
        ax1 = axes[0, 0]
        ax1.plot(original.index, original.values, 'b-', label='Original', alpha=0.7)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.set_title('Original Series')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Série diferenciada
        ax2 = axes[0, 1]
        ax2.plot(frac_diff_series.index, frac_diff_series.values, 'r-', 
                label='Fractionally Differentiated', alpha=0.7)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value')
        ax2.set_title('Fractionally Differentiated Series')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribuições
        ax3 = axes[1, 0]
        ax3.hist(original.dropna().values, bins=50, alpha=0.7, label='Original', 
                color='blue', density=True)
        ax3.hist(frac_diff_series.dropna().values, bins=50, alpha=0.7, 
                label='Frac Diff', color='red', density=True)
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Autocorrelação
        ax4 = axes[1, 1]
        
        # Calcular autocorrelações
        lags = range(1, min(21, len(original) // 4))
        
        orig_autocorr = [original.autocorr(lag=lag) for lag in lags]
        diff_autocorr = [frac_diff_series.autocorr(lag=lag) for lag in lags]
        
        ax4.plot(lags, orig_autocorr, 'b-o', label='Original', markersize=4)
        ax4.plot(lags, diff_autocorr, 'r-o', label='Frac Diff', markersize=4)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Lag')
        ax4.set_ylabel('Autocorrelation')
        ax4.set_title('Autocorrelation Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            if filename is None:
                filename = f"fractional_diff_comparison_{original.name}_d{d_value:.3f}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """
        Retorna resumo das otimizações realizadas
        """
        if not self.optimization_results:
            return pd.DataFrame()
            
        summary_data = []
        
        for col_name, result in self.optimization_results.items():
            summary_data.append({
                'series': col_name,
                'optimal_d': result['optimal_d'],
                'adf_pvalue': result['adf_pvalue'],
                'memory_preservation': result['memory_preservation'],
                'is_stationary': result['is_stationary'],
                'method': result['method']
            })
            
        return pd.DataFrame(summary_data)

# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Criar dados de exemplo
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Série com tendência e ruído
    trend = np.linspace(100, 200, len(dates))
    noise = np.random.normal(0, 5, len(dates))
    ar_component = np.zeros(len(dates))
    
    # Adicionar componente AR(1)
    for i in range(1, len(dates)):
        ar_component[i] = 0.7 * ar_component[i-1] + np.random.normal(0, 2)
        
    price_series = pd.Series(
        trend + noise + ar_component,
        index=dates,
        name='price'
    )
    
    # Configuração
    config = FractionalDiffConfig(
        optimize_d=True,
        verbose=True,
        use_numba=True
    )
    
    # Criar diferenciador
    differentiator = FractionalDifferentiator(config=config, logger=logger)
    
    # Encontrar d ótimo
    print("\n=== OTIMIZAÇÃO DO PARÂMETRO D ===")
    opt_result = differentiator.find_optimal_d(price_series, method='adf_memory')
    
    for key, value in opt_result.items():
        print(f"{key}: {value}")
        
    # Aplicar diferenciação com d ótimo
    optimal_d = opt_result['optimal_d']
    frac_diff_series = differentiator.frac_diff(price_series, optimal_d)
    
    print(f"\nSérie original: {len(price_series)} pontos")
    print(f"Série diferenciada: {len(frac_diff_series)} pontos")
    print(f"Preservação de memória: {opt_result['memory_preservation']:.4f}")
    
    # Análise de sensibilidade
    print("\n=== ANÁLISE DE SENSIBILIDADE ===")
    differentiator.plot_sensitivity_analysis(price_series, save_plot=False)
    
    # Comparação visual
    print("\n=== COMPARAÇÃO VISUAL ===")
    differentiator.plot_comparison(price_series, optimal_d, save_plot=False)
    
    # Processar múltiplas séries
    print("\n=== PROCESSAMENTO MÚLTIPLAS SÉRIES ===")
    
    # Criar DataFrame com múltiplas séries
    data = pd.DataFrame({
        'price1': price_series,
        'price2': price_series * 1.2 + np.random.normal(0, 3, len(price_series)),
        'price3': price_series * 0.8 + np.random.normal(0, 2, len(price_series))
    })
    
    # Processar todas as séries
    frac_diff_data = differentiator.process_multiple_series(data)
    
    print(f"\nDados originais: {data.shape}")
    print(f"Dados diferenciados: {frac_diff_data.shape}")
    
    # Resumo das otimizações
    summary = differentiator.get_optimization_summary()
    print("\n=== RESUMO DAS OTIMIZAÇÕES ===")
    print(summary)