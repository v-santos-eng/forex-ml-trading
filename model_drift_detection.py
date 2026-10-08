import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator
import warnings
from dataclasses import dataclass
from enum import Enum

class DriftType(Enum):
    """Tipos de drift detectados"""
    NO_DRIFT = "no_drift"
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    CONCEPT_DRIFT = "concept_drift"

@dataclass
class DriftAlert:
    """Estrutura para alertas de drift"""
    timestamp: datetime
    drift_type: DriftType
    severity: str  # 'low', 'medium', 'high', 'critical'
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    confidence: float
    message: str

class ModelDriftDetector:
    """
    Sistema de detecção de drift para modelos de ML em trading
    Baseado em "Advances in Financial Machine Learning" e melhores práticas
    
    Detecta diferentes tipos de drift:
    1. Feature Drift: Mudanças na distribuição das features
    2. Prediction Drift: Mudanças na distribuição das predições
    3. Performance Drift: Degradação das métricas de performance
    4. Concept Drift: Mudanças na relação feature-target
    """
    
    def __init__(self, 
                 baseline_window: int = 1000,
                 detection_window: int = 100,
                 min_samples: int = 50,
                 significance_level: float = 0.05,
                 reference_window: Optional[int] = None,
                 drift_threshold: float = 0.05,
                 performance_threshold: float = 0.1,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            baseline_window: Tamanho da janela baseline para comparação
            detection_window: Tamanho da janela para detecção
            min_samples: Número mínimo de amostras para detecção
            significance_level: Nível de significância para testes estatísticos
            performance_threshold: Threshold para degradação de performance
            logger: Logger para debug
        """
        self.baseline_window = baseline_window
        self.detection_window = detection_window
        self.min_samples = min_samples
        self.significance_level = significance_level
        self.performance_threshold = performance_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Armazenamento de dados históricos
        self.feature_history = pd.DataFrame()
        self.prediction_history = pd.DataFrame()
        self.performance_history = pd.DataFrame()
        self.target_history = pd.Series(dtype=float)
        
        # Baselines
        self.feature_baseline = {}
        self.prediction_baseline = {}
        self.performance_baseline = {}
        
        # Alertas
        self.alerts = []
        
    def update_baseline(self, features: pd.DataFrame, 
                       predictions: pd.Series,
                       targets: pd.Series,
                       performance_metrics: Dict[str, float],
                       labels: Optional[pd.Series] = None):
        """
        Atualiza baseline com novos dados
        
        Args:
            features: DataFrame com features
            predictions: Série com predições
            targets: Série com targets reais
            performance_metrics: Dicionário com métricas de performance
        """
        # Atualizar histórico
        self.feature_history = pd.concat([self.feature_history, features]).tail(self.baseline_window)
        
        pred_df = pd.DataFrame({'predictions': predictions}, index=predictions.index)
        self.prediction_history = pd.concat([self.prediction_history, pred_df]).tail(self.baseline_window)
        
        self.target_history = pd.concat([self.target_history, targets]).tail(self.baseline_window)
        
        perf_df = pd.DataFrame([performance_metrics], 
                              index=[datetime.now()])
        self.performance_history = pd.concat([self.performance_history, perf_df]).tail(self.baseline_window)
        
        # Recalcular baselines
        self._calculate_baselines()
        
        self.logger.info(f"Baseline atualizado: {len(self.feature_history)} amostras")
    
    def detect_drift(self, features: pd.DataFrame,
                    predictions: pd.Series,
                    targets: Optional[pd.Series] = None,
                    performance_metrics: Optional[Dict[str, float]] = None) -> List[DriftAlert]:
        """
        Detecta drift nos dados atuais
        
        Args:
            features: Features atuais
            predictions: Predições atuais
            targets: Targets reais (se disponível)
            performance_metrics: Métricas de performance atuais
            
        Returns:
            Lista de alertas de drift detectados
        """
        alerts = []
        
        # Verificar se temos baseline suficiente
        if len(self.feature_history) < self.min_samples:
            self.logger.warning("Baseline insuficiente para detecção de drift")
            return alerts
            
        # Detectar feature drift
        feature_alerts = self._detect_feature_drift(features)
        alerts.extend(feature_alerts)
        
        # Detectar prediction drift
        prediction_alerts = self._detect_prediction_drift(predictions)
        alerts.extend(prediction_alerts)
        
        # Detectar performance drift
        if performance_metrics:
            performance_alerts = self._detect_performance_drift(performance_metrics)
            alerts.extend(performance_alerts)
            
        # Detectar concept drift
        if targets is not None:
            concept_alerts = self._detect_concept_drift(features, targets)
            alerts.extend(concept_alerts)
            
        # Armazenar alertas
        self.alerts.extend(alerts)
        
        if alerts:
            self.logger.warning(f"Detectados {len(alerts)} alertas de drift")
            
        return alerts

    def detect_feature_drift(self, reference_data: pd.DataFrame, new_data: pd.DataFrame = None) -> Dict:
        """
        Método público para detectar drift nas features
        
        Args:
            reference_data: DataFrame com as features de referência (baseline)
            new_data: DataFrame com as features atuais (opcional, se None usa reference_data)
            
        Returns:
            Dicionário com resultados de drift por feature
        """
        if new_data is None:
            # Compatibilidade com chamada de um parâmetro
            alerts = self._detect_feature_drift(reference_data)
            return {alert.metric_name: {'drift_detected': True} for alert in alerts}
        
        # Atualizar baseline com dados de referência
        # Criar dados dummy para compatibilidade
        dummy_predictions = pd.Series([0] * len(reference_data), index=reference_data.index)
        dummy_targets = pd.Series([0] * len(reference_data), index=reference_data.index)
        dummy_metrics = {'accuracy': 0.5}
        
        self.update_baseline(reference_data, dummy_predictions, dummy_targets, dummy_metrics)
        
        # Detectar drift nos novos dados
        alerts = self._detect_feature_drift(new_data)
        
        # Converter para formato esperado pelo teste
        results = {}
        for column in new_data.columns:
            column_alerts = [alert for alert in alerts if alert.metric_name == f"feature_{column}"]
            results[column] = {
                'drift_detected': len(column_alerts) > 0,
                'severity': column_alerts[0].severity if column_alerts else 'none'
            }
        
        return results

    def _detect_feature_drift(self, current_features: pd.DataFrame) -> List[DriftAlert]:
        """
        Detecta drift nas features usando testes estatísticos
        """
        alerts = []
        
        for column in current_features.columns:
            if column not in self.feature_history.columns:
                continue
                
            baseline_data = self.feature_history[column].dropna()
            current_data = current_features[column].dropna()
            
            if len(baseline_data) < self.min_samples or len(current_data) < 10:
                continue
                
            # Teste Kolmogorov-Smirnov
            ks_stat, ks_p_value = stats.ks_2samp(baseline_data, current_data)
            
            # Teste Mann-Whitney U (não-paramétrico)
            mw_stat, mw_p_value = stats.mannwhitneyu(baseline_data, current_data, 
                                                    alternative='two-sided')
            
            # Verificar significância
            if ks_p_value < self.significance_level or mw_p_value < self.significance_level:
                severity = self._calculate_severity(min(ks_p_value, mw_p_value))
                
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    drift_type=DriftType.FEATURE_DRIFT,
                    severity=severity,
                    metric_name=f"feature_{column}",
                    current_value=current_data.mean(),
                    baseline_value=baseline_data.mean(),
                    threshold=self.significance_level,
                    confidence=1 - min(ks_p_value, mw_p_value),
                    message=f"Feature drift detectado em {column}: KS p-value={ks_p_value:.4f}, MW p-value={mw_p_value:.4f}"
                )
                
                alerts.append(alert)
                
        return alerts
    
    def _detect_prediction_drift(self, current_predictions: pd.Series) -> List[DriftAlert]:
        """
        Detecta drift nas predições do modelo
        """
        alerts = []
        
        if len(self.prediction_history) < self.min_samples:
            return alerts
            
        baseline_preds = self.prediction_history['predictions'].dropna()
        current_preds = current_predictions.dropna()
        
        if len(current_preds) < 10:
            return alerts
            
        # Teste de distribuição
        ks_stat, ks_p_value = stats.ks_2samp(baseline_preds, current_preds)
        
        # Comparar estatísticas descritivas
        baseline_mean = baseline_preds.mean()
        current_mean = current_preds.mean()
        mean_change = abs(current_mean - baseline_mean) / abs(baseline_mean) if baseline_mean != 0 else 0
        
        baseline_std = baseline_preds.std()
        current_std = current_preds.std()
        std_change = abs(current_std - baseline_std) / baseline_std if baseline_std != 0 else 0
        
        # Alertas baseados em mudanças significativas
        if ks_p_value < self.significance_level:
            alert = DriftAlert(
                timestamp=datetime.now(),
                drift_type=DriftType.PREDICTION_DRIFT,
                severity=self._calculate_severity(ks_p_value),
                metric_name="prediction_distribution",
                current_value=current_mean,
                baseline_value=baseline_mean,
                threshold=self.significance_level,
                confidence=1 - ks_p_value,
                message=f"Prediction drift detectado: KS p-value={ks_p_value:.4f}"
            )
            alerts.append(alert)
            
        if mean_change > 0.2:  # 20% de mudança na média
            alert = DriftAlert(
                timestamp=datetime.now(),
                drift_type=DriftType.PREDICTION_DRIFT,
                severity="medium" if mean_change < 0.5 else "high",
                metric_name="prediction_mean",
                current_value=current_mean,
                baseline_value=baseline_mean,
                threshold=0.2,
                confidence=mean_change,
                message=f"Mudança significativa na média das predições: {mean_change:.2%}"
            )
            alerts.append(alert)
            
        return alerts
    
    def _detect_performance_drift(self, current_metrics: Dict[str, float]) -> List[DriftAlert]:
        """
        Detecta degradação nas métricas de performance
        """
        alerts = []
        
        if len(self.performance_history) < self.min_samples:
            return alerts
            
        for metric_name, current_value in current_metrics.items():
            if metric_name not in self.performance_baseline:
                continue
                
            baseline_value = self.performance_baseline[metric_name]
            
            # Calcular degradação relativa
            if baseline_value != 0:
                degradation = (baseline_value - current_value) / abs(baseline_value)
            else:
                degradation = abs(current_value)
                
            # Verificar se degradação excede threshold
            if degradation > self.performance_threshold:
                severity = "medium" if degradation < 0.2 else "high" if degradation < 0.4 else "critical"
                
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    drift_type=DriftType.PERFORMANCE_DRIFT,
                    severity=severity,
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    threshold=self.performance_threshold,
                    confidence=min(degradation, 1.0),
                    message=f"Degradação de performance em {metric_name}: {degradation:.2%}"
                )
                
                alerts.append(alert)
                
        return alerts
    
    def _detect_concept_drift(self, features: pd.DataFrame, 
                            targets: pd.Series) -> List[DriftAlert]:
        """
        Detecta concept drift (mudança na relação feature-target)
        """
        alerts = []
        
        if len(self.feature_history) < self.min_samples or len(self.target_history) < self.min_samples:
            return alerts
            
        # Calcular correlações atuais vs baseline
        for column in features.columns:
            if column not in self.feature_history.columns:
                continue
                
            # Correlação baseline
            baseline_features = self.feature_history[column].dropna()
            baseline_targets = self.target_history.reindex(baseline_features.index).dropna()
            
            if len(baseline_targets) < self.min_samples:
                continue
                
            baseline_corr = baseline_features.corr(baseline_targets)
            
            # Correlação atual
            current_feature = features[column].dropna()
            current_targets_aligned = targets.reindex(current_feature.index).dropna()
            
            if len(current_targets_aligned) < 10:
                continue
                
            current_corr = current_feature.corr(current_targets_aligned)
            
            # Verificar mudança significativa na correlação
            if not np.isnan(baseline_corr) and not np.isnan(current_corr):
                corr_change = abs(current_corr - baseline_corr)
                
                if corr_change > 0.3:  # Mudança > 30% na correlação
                    alert = DriftAlert(
                        timestamp=datetime.now(),
                        drift_type=DriftType.CONCEPT_DRIFT,
                        severity="medium" if corr_change < 0.5 else "high",
                        metric_name=f"correlation_{column}",
                        current_value=current_corr,
                        baseline_value=baseline_corr,
                        threshold=0.3,
                        confidence=corr_change,
                        message=f"Concept drift detectado em {column}: correlação mudou de {baseline_corr:.3f} para {current_corr:.3f}"
                    )
                    
                    alerts.append(alert)
                    
        return alerts
    
    def _calculate_baselines(self):
        """
        Calcula baselines para comparação
        """
        # Feature baselines (estatísticas descritivas)
        self.feature_baseline = {
            col: {
                'mean': self.feature_history[col].mean(),
                'std': self.feature_history[col].std(),
                'median': self.feature_history[col].median(),
                'q25': self.feature_history[col].quantile(0.25),
                'q75': self.feature_history[col].quantile(0.75)
            }
            for col in self.feature_history.columns
        }
        
        # Prediction baselines
        if 'predictions' in self.prediction_history.columns:
            preds = self.prediction_history['predictions']
            self.prediction_baseline = {
                'mean': preds.mean(),
                'std': preds.std(),
                'median': preds.median()
            }
            
        # Performance baselines
        if len(self.performance_history) > 0:
            self.performance_baseline = self.performance_history.mean().to_dict()
            
    def _calculate_severity(self, p_value: float) -> str:
        """
        Calcula severidade baseada no p-value
        """
        if p_value < 0.001:
            return "critical"
        elif p_value < 0.01:
            return "high"
        elif p_value < 0.05:
            return "medium"
        else:
            return "low"
    
    def get_drift_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Retorna resumo dos drifts detectados nas últimas horas
        
        Args:
            hours: Número de horas para análise
            
        Returns:
            Dicionário com resumo
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
        
        summary = {
            'total_alerts': len(recent_alerts),
            'by_type': {},
            'by_severity': {},
            'critical_alerts': [],
            'recommendations': []
        }
        
        # Contar por tipo
        for alert in recent_alerts:
            drift_type = alert.drift_type.value
            summary['by_type'][drift_type] = summary['by_type'].get(drift_type, 0) + 1
            
            severity = alert.severity
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            if severity == 'critical':
                summary['critical_alerts'].append(alert)
                
        # Gerar recomendações
        if summary['total_alerts'] > 0:
            summary['recommendations'].append("Monitorar modelo de perto")
            
        if summary['by_severity'].get('critical', 0) > 0:
            summary['recommendations'].append("URGENTE: Re-treinar modelo imediatamente")
        elif summary['by_severity'].get('high', 0) > 2:
            summary['recommendations'].append("Considerar re-treinamento do modelo")
        elif summary['by_type'].get('performance_drift', 0) > 0:
            summary['recommendations'].append("Revisar estratégia de trading")
            
        return summary
    
    def should_retrain_model(self) -> Tuple[bool, str]:
        """
        Determina se o modelo deve ser re-treinado
        
        Returns:
            Tupla (should_retrain, reason)
        """
        recent_summary = self.get_drift_summary(hours=24)
        
        # Critérios para re-treinamento
        critical_alerts = recent_summary['by_severity'].get('critical', 0)
        high_alerts = recent_summary['by_severity'].get('high', 0)
        performance_drift = recent_summary['by_type'].get('performance_drift', 0)
        
        if critical_alerts > 0:
            return True, f"Detectados {critical_alerts} alertas críticos"
        elif high_alerts >= 3:
            return True, f"Detectados {high_alerts} alertas de alta severidade"
        elif performance_drift >= 2:
            return True, f"Detectada degradação de performance em {performance_drift} métricas"
        else:
            return False, "Modelo estável"
    
    def export_drift_report(self, filepath: str):
        """
        Exporta relatório detalhado de drift
        
        Args:
            filepath: Caminho para salvar o relatório
        """
        report_data = []
        
        for alert in self.alerts:
            report_data.append({
                'timestamp': alert.timestamp,
                'drift_type': alert.drift_type.value,
                'severity': alert.severity,
                'metric_name': alert.metric_name,
                'current_value': alert.current_value,
                'baseline_value': alert.baseline_value,
                'threshold': alert.threshold,
                'confidence': alert.confidence,
                'message': alert.message
            })
            
        df = pd.DataFrame(report_data)
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Relatório de drift exportado para {filepath}")

# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Criar detector
    detector = ModelDriftDetector(logger=logger)
    
    # Simular dados baseline
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    
    baseline_features = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000),
        'feature3': np.random.exponential(1, 1000)
    }, index=dates)
    
    baseline_predictions = pd.Series(np.random.choice([0, 1], 1000), index=dates)
    baseline_targets = pd.Series(np.random.choice([0, 1], 1000), index=dates)
    baseline_metrics = {'accuracy': 0.75, 'precision': 0.73, 'f1': 0.74}
    
    # Atualizar baseline
    detector.update_baseline(baseline_features, baseline_predictions, 
                           baseline_targets, baseline_metrics)
    
    # Simular dados com drift
    current_features = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, 100),  # Drift na média e variância
        'feature2': np.random.normal(5, 2, 100),      # Sem drift
        'feature3': np.random.exponential(1.5, 100)   # Drift no parâmetro
    })
    
    current_predictions = pd.Series(np.random.choice([0, 1], 100, p=[0.3, 0.7]))  # Drift na distribuição
    current_targets = pd.Series(np.random.choice([0, 1], 100))
    current_metrics = {'accuracy': 0.65, 'precision': 0.60, 'f1': 0.62}  # Performance degradada
    
    # Detectar drift
    alerts = detector.detect_drift(current_features, current_predictions, 
                                 current_targets, current_metrics)
    
    print(f"\nDetectados {len(alerts)} alertas de drift:")
    for alert in alerts:
        print(f"- {alert.drift_type.value}: {alert.message} (Severidade: {alert.severity})")
    
    # Verificar se deve re-treinar
    should_retrain, reason = detector.should_retrain_model()
    print(f"\nRe-treinar modelo: {should_retrain} - {reason}")
    
    # Resumo de drift
    summary = detector.get_drift_summary()
    print(f"\nResumo de drift: {summary}")