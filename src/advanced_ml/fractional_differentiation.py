import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import warnings


class FractionalDifferentiation:
    """
    Implementação da Diferenciação Fracionária baseada em López de Prado.
    
    A diferenciação fracionária generaliza processos ARIMA permitindo que o grau
    de diferenciação (d) assuma valores fracionários. Isso permite:
    
    1. Preservar memória de longo prazo quando d > 0 e d < 1
    2. Manter estacionariedade
    3. Melhorar poder preditivo
    4. Evitar over-differencing
    
    Implementa tanto a versão expanding window quanto fixed-width window (FFD).
    """
    
    def __init__(self, d: float = 0.5, threshold: float = 1e-5):
        """
        Inicializa a diferenciação fracionária.
        
        Args:
            d: Grau de diferenciação fracionária (0 < d < 1)
            threshold: Threshold para truncar pesos pequenos
        """
        if not 0 < d < 1:
            warnings.warn("d deve estar entre 0 e 1 para preservar memória")
        
        self.d = d
        self.threshold = threshold
        self._weights = None
        self._max_lag = None
    
    def get_weights(self, d: float, size: int) -> np.ndarray:
        """
        Calcula os pesos da diferenciação fracionária usando expansão binomial.
        
        A fórmula é baseada na expansão de (1-B)^d onde B é o operador lag:
        ω_k = (-1)^k * Γ(d+1) / (Γ(k+1) * Γ(d-k+1))
        
        Args:
            d: Grau de diferenciação
            size: Número de pesos a calcular
            
        Returns:
            Array com os pesos
        """
        weights = np.zeros(size)
        weights[0] = 1.0
        
        for k in range(1, size):
            # Fórmula recursiva: ω_k = ω_{k-1} * (d - k + 1) / k
            weights[k] = weights[k-1] * (d - k + 1) / k
        
        return weights
    
    def get_weights_ffd(self, d: float, threshold: float = 1e-5) -> np.ndarray:
        """
        Calcula pesos para Fixed-Width Window Fractional Differentiation (FFD).
        
        Trunca os pesos quando ficam menores que o threshold para evitar
        janelas muito grandes.
        
        Args:
            d: Grau de diferenciação
            threshold: Threshold para truncar pesos
            
        Returns:
            Array com pesos truncados
        """
        weights = [1.0]
        k = 1
        
        while True:
            weight = weights[-1] * (d - k + 1) / k
            
            if abs(weight) < threshold:
                break
                
            weights.append(weight)
            k += 1
        
        return np.array(weights)
    
    def plot_weights(self, d_values: list = None, max_lags: int = 50) -> None:
        """
        Plota os pesos da diferenciação fracionária para diferentes valores de d.
        
        Args:
            d_values: Lista de valores de d para plotar
            max_lags: Número máximo de lags para mostrar
        """
        if d_values is None:
            d_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        plt.figure(figsize=(12, 8))
        
        for d in d_values:
            weights = self.get_weights(d, max_lags)
            plt.plot(range(len(weights)), weights, 
                    label=f'd = {d}', marker='o', markersize=3)
        
        plt.xlabel('Lag')
        plt.ylabel('Peso')
        plt.title('Pesos da Diferenciação Fracionária')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.show()
    
    def frac_diff_expanding(self, 
                           series: pd.Series, 
                           d: float, 
                           threshold: float = 0.01) -> pd.Series:
        """
        Diferenciação fracionária com janela expansiva.
        
        Esta versão usa todos os dados históricos disponíveis, mas pode
        introduzir viés negativo devido ao crescimento da janela.
        
        Args:
            series: Série temporal para diferenciar
            d: Grau de diferenciação
            threshold: Threshold para pesos mínimos
            
        Returns:
            Série diferenciada
        """
        # Calcular pesos máximos necessários
        weights = self.get_weights_ffd(d, threshold)
        max_lag = len(weights)
        
        # Inicializar série de saída
        output = pd.Series(index=series.index, dtype=float)
        
        for i in range(max_lag, len(series)):
            # Usar todos os pesos disponíveis até o ponto atual
            current_weights = weights[:min(i+1, len(weights))]
            
            # Aplicar pesos aos valores históricos
            values = series.iloc[i-len(current_weights)+1:i+1].values[::-1]
            output.iloc[i] = np.dot(current_weights, values)
        
        return output.dropna()
    
    def frac_diff_ffd(self, 
                     series: pd.Series, 
                     d: float, 
                     threshold: float = 1e-5) -> pd.Series:
        """
        Fixed-Width Window Fractional Differentiation (FFD).
        
        Esta é a versão preferida pois evita o viés negativo da janela expansiva
        mantendo uma janela de tamanho fixo.
        
        Args:
            series: Série temporal para diferenciar
            d: Grau de diferenciação
            threshold: Threshold para truncar pesos
            
        Returns:
            Série diferenciada
        """
        # Calcular pesos
        weights = self.get_weights_ffd(d, threshold)
        width = len(weights)
        
        # Aplicar convolução
        output = pd.Series(index=series.index, dtype=float)
        
        for i in range(width-1, len(series)):
            # Extrair janela de dados
            window = series.iloc[i-width+1:i+1].values[::-1]
            
            # Aplicar pesos
            output.iloc[i] = np.dot(weights, window)
        
        return output.dropna()
    
    def fit_transform(self, 
                     series: pd.Series, 
                     method: str = 'ffd') -> pd.Series:
        """
        Aplica diferenciação fracionária à série.
        
        Args:
            series: Série temporal
            method: 'ffd' para fixed-width ou 'expanding' para janela expansiva
            
        Returns:
            Série diferenciada
        """
        if method == 'ffd':
            result = self.frac_diff_ffd(series, self.d, self.threshold)
        elif method == 'expanding':
            result = self.frac_diff_expanding(series, self.d, self.threshold)
        else:
            raise ValueError("method deve ser 'ffd' ou 'expanding'")
        
        # Armazenar informações para uso posterior
        self._weights = self.get_weights_ffd(self.d, self.threshold)
        self._max_lag = len(self._weights)
        
        return result
    
    def transform(self, series: pd.Series, method: str = 'ffd') -> pd.Series:
        """
        Aplica transformação usando parâmetros já ajustados.
        
        Args:
            series: Série temporal
            method: Método de diferenciação
            
        Returns:
            Série diferenciada
        """
        return self.fit_transform(series, method)


class OptimalDifferentiation:
    """
    Classe para encontrar o grau ótimo de diferenciação fracionária.
    
    Usa testes de estacionariedade (ADF) para encontrar o menor d que
    torna a série estacionária, preservando máxima memória.
    """
    
    def __init__(self, 
                 d_range: Tuple[float, float] = (0.0, 1.0),
                 step: float = 0.01,
                 adf_threshold: float = 0.05):
        """
        Inicializa a busca por d ótimo.
        
        Args:
            d_range: Range de valores de d para testar
            step: Passo para busca
            adf_threshold: P-value threshold para teste ADF
        """
        self.d_range = d_range
        self.step = step
        self.adf_threshold = adf_threshold
        self.optimal_d = None
        self.test_results = None
    
    def test_stationarity(self, series: pd.Series) -> Tuple[float, bool]:
        """
        Testa estacionariedade usando teste ADF.
        
        Args:
            series: Série temporal
            
        Returns:
            Tuple com (p-value, is_stationary)
        """
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            p_value = result[1]
            is_stationary = p_value < self.adf_threshold
            return p_value, is_stationary
        except:
            return 1.0, False
    
    def find_optimal_d(self, 
                      series: pd.Series, 
                      method: str = 'ffd',
                      verbose: bool = True) -> float:
        """
        Encontra o grau ótimo de diferenciação.
        
        Args:
            series: Série temporal original
            method: Método de diferenciação
            verbose: Se deve imprimir progresso
            
        Returns:
            Valor ótimo de d
        """
        d_values = np.arange(self.d_range[0], self.d_range[1] + self.step, self.step)
        results = []
        
        frac_diff = FractionalDifferentiation()
        
        for d in d_values:
            frac_diff.d = d
            
            try:
                # Aplicar diferenciação fracionária
                diff_series = frac_diff.fit_transform(series, method)
                
                if len(diff_series) < 10:  # Série muito curta
                    continue
                
                # Testar estacionariedade
                p_value, is_stationary = self.test_stationarity(diff_series)
                
                # Calcular correlação com série original (preservação de memória)
                correlation = series.corr(diff_series) if len(diff_series) > 1 else 0
                
                results.append({
                    'd': d,
                    'p_value': p_value,
                    'is_stationary': is_stationary,
                    'correlation': correlation,
                    'series_length': len(diff_series)
                })
                
                if verbose and len(results) % 10 == 0:
                    print(f"Testado d = {d:.3f}, p-value = {p_value:.4f}, "
                          f"estacionário = {is_stationary}")
                
            except Exception as e:
                if verbose:
                    print(f"Erro para d = {d}: {e}")
                continue
        
        if not results:
            raise ValueError("Nenhum resultado válido encontrado")
        
        # Converter para DataFrame
        self.test_results = pd.DataFrame(results)
        
        # Encontrar menor d que resulta em estacionariedade
        stationary_results = self.test_results[self.test_results['is_stationary']]
        
        if len(stationary_results) == 0:
            # Se nenhum d resulta em estacionariedade, usar o com menor p-value
            self.optimal_d = self.test_results.loc[self.test_results['p_value'].idxmin(), 'd']
            if verbose:
                print(f"Nenhum d resultou em estacionariedade. Usando d = {self.optimal_d:.3f} "
                      f"(menor p-value = {self.test_results['p_value'].min():.4f})")
        else:
            # Usar menor d que resulta em estacionariedade
            self.optimal_d = stationary_results['d'].min()
            if verbose:
                print(f"D ótimo encontrado: {self.optimal_d:.3f}")
        
        return self.optimal_d
    
    def plot_results(self) -> None:
        """
        Plota os resultados da busca por d ótimo.
        """
        if self.test_results is None:
            raise ValueError("Execute find_optimal_d primeiro")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: P-values do teste ADF
        ax1.plot(self.test_results['d'], self.test_results['p_value'], 
                'b-', linewidth=2, label='P-value ADF')
        ax1.axhline(y=self.adf_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({self.adf_threshold})')
        ax1.axvline(x=self.optimal_d, color='g', linestyle='--', 
                   label=f'D ótimo ({self.optimal_d:.3f})')
        ax1.set_xlabel('Grau de Diferenciação (d)')
        ax1.set_ylabel('P-value (ADF Test)')
        ax1.set_title('Teste de Estacionariedade vs. Grau de Diferenciação')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Correlação com série original
        ax2.plot(self.test_results['d'], self.test_results['correlation'], 
                'g-', linewidth=2, label='Correlação com original')
        ax2.axvline(x=self.optimal_d, color='g', linestyle='--', 
                   label=f'D ótimo ({self.optimal_d:.3f})')
        ax2.set_xlabel('Grau de Diferenciação (d)')
        ax2.set_ylabel('Correlação')
        ax2.set_title('Preservação de Memória vs. Grau de Diferenciação')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def compare_differentiation_methods(series: pd.Series, 
                                  d: float = 0.5,
                                  plot: bool = True) -> dict:
    """
    Compara diferentes métodos de diferenciação.
    
    Args:
        series: Série temporal original
        d: Grau de diferenciação
        plot: Se deve plotar comparação
        
    Returns:
        Dicionário com resultados da comparação
    """
    frac_diff = FractionalDifferentiation(d)
    
    # Aplicar diferentes métodos
    original = series.copy()
    first_diff = series.diff().dropna()
    ffd = frac_diff.frac_diff_ffd(series, d)
    expanding = frac_diff.frac_diff_expanding(series, d)
    
    # Testes de estacionariedade
    opt_diff = OptimalDifferentiation()
    
    results = {
        'original': {
            'series': original,
            'adf_pvalue': opt_diff.test_stationarity(original)[0],
            'is_stationary': opt_diff.test_stationarity(original)[1]
        },
        'first_difference': {
            'series': first_diff,
            'adf_pvalue': opt_diff.test_stationarity(first_diff)[0],
            'is_stationary': opt_diff.test_stationarity(first_diff)[1]
        },
        'ffd': {
            'series': ffd,
            'adf_pvalue': opt_diff.test_stationarity(ffd)[0],
            'is_stationary': opt_diff.test_stationarity(ffd)[1]
        },
        'expanding': {
            'series': expanding,
            'adf_pvalue': opt_diff.test_stationarity(expanding)[0],
            'is_stationary': opt_diff.test_stationarity(expanding)[1]
        }
    }
    
    # Calcular correlações com série original
    for method in ['first_difference', 'ffd', 'expanding']:
        method_series = results[method]['series']
        if len(method_series) > 1:
            results[method]['correlation_with_original'] = original.corr(method_series)
        else:
            results[method]['correlation_with_original'] = np.nan
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot séries
        axes[0, 0].plot(original.index, original.values, label='Original')
        axes[0, 0].set_title('Série Original')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(first_diff.index, first_diff.values, label='Primeira Diferença', color='orange')
        axes[0, 1].set_title('Primeira Diferença')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(ffd.index, ffd.values, label=f'FFD (d={d})', color='green')
        axes[1, 0].set_title('Fixed-Width Fractional Differentiation')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(expanding.index, expanding.values, label=f'Expanding (d={d})', color='red')
        axes[1, 1].set_title('Expanding Window Fractional Differentiation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Tabela de resultados
        print("\nComparação de Métodos de Diferenciação:")
        print("=" * 60)
        for method, data in results.items():
            print(f"\n{method.upper()}:")
            print(f"  ADF p-value: {data['adf_pvalue']:.6f}")
            print(f"  Estacionário: {data['is_stationary']}")
            if 'correlation_with_original' in data:
                print(f"  Correlação com original: {data['correlation_with_original']:.4f}")
    
    return results


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados de exemplo
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Série com tendência e ruído
    trend = np.cumsum(np.random.randn(1000) * 0.01)
    noise = np.random.randn(1000) * 0.1
    series = pd.Series(trend + noise, index=dates)
    
    # Encontrar d ótimo
    opt_diff = OptimalDifferentiation()
    optimal_d = opt_diff.find_optimal_d(series)
    
    # Comparar métodos
    comparison = compare_differentiation_methods(series, optimal_d)