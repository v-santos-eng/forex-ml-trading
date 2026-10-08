#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Logging Abrangente para Análises de Trading
Captura todas as informações relevantes analisadas durante o processo
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

class AnalysisType(Enum):
    """Tipos de análise realizadas"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    RISK_MANAGEMENT = "risk_management"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    ECONOMIC_CALENDAR = "economic_calendar"
    NEWS_IMPACT = "news_impact"
    SEASONAL = "seasonal"
    INTERMARKET = "intermarket"
    FLOW_ANALYSIS = "flow_analysis"

class DataSource(Enum):
    """Fontes de dados utilizadas"""
    MT5 = "mt5"
    ECONOMIC_CALENDAR = "economic_calendar"
    NEWS_FEED = "news_feed"
    SENTIMENT_INDICATOR = "sentiment_indicator"
    COT_REPORT = "cot_report"
    CENTRAL_BANK = "central_bank"
    TECHNICAL_INDICATOR = "technical_indicator"
    PRICE_ACTION = "price_action"
    VOLUME_PROFILE = "volume_profile"
    ML_MODEL = "ml_model"
    MANUAL_ANALYSIS = "manual_analysis"

@dataclass
class TechnicalIndicator:
    """Dados de indicador técnico"""
    name: str
    value: Union[float, str]
    timeframe: str
    signal: str  # bullish, bearish, neutral
    confidence: float
    parameters: Dict[str, Any]
    timestamp: datetime

@dataclass
class PriceLevel:
    """Nível de preço importante"""
    level: float
    type: str  # support, resistance, pivot, fibonacci
    strength: float  # 0-1
    touches: int
    last_test: datetime
    description: str

@dataclass
class EconomicEvent:
    """Evento econômico"""
    title: str
    country: str
    currency: str
    impact: str  # low, medium, high
    forecast: Optional[str]
    previous: Optional[str]
    actual: Optional[str]
    time: datetime
    relevance_score: float

@dataclass
class NewsItem:
    """Item de notícia"""
    headline: str
    source: str
    sentiment: str  # positive, negative, neutral
    relevance: float
    impact_currencies: List[str]
    timestamp: datetime
    summary: str

@dataclass
class RiskMetric:
    """Métrica de risco"""
    name: str
    value: float
    threshold: float
    status: str  # normal, warning, critical
    description: str
    recommendation: str

@dataclass
class MarketCondition:
    """Condição de mercado"""
    aspect: str  # volatility, liquidity, trend, momentum
    value: Union[float, str]
    classification: str  # low/medium/high, bullish/bearish/neutral
    impact_on_trading: str
    confidence: float

@dataclass
class AnalysisData:
    """Dados completos de uma análise"""
    analysis_id: str
    timestamp: datetime
    symbol: str
    timeframe: str
    analysis_type: AnalysisType
    data_sources: List[DataSource]
    
    # Dados técnicos
    technical_indicators: List[TechnicalIndicator]
    price_levels: List[PriceLevel]
    chart_patterns: List[Dict[str, Any]]
    
    # Dados fundamentais
    economic_events: List[EconomicEvent]
    news_items: List[NewsItem]
    
    # Análise de risco
    risk_metrics: List[RiskMetric]
    
    # Condições de mercado
    market_conditions: List[MarketCondition]
    
    # Correlações
    correlations: Dict[str, float]
    
    # Análise de sentimento
    sentiment_data: Dict[str, Any]
    
    # Conclusões
    key_observations: List[str]
    risk_factors: List[str]
    opportunities: List[str]
    recommendations: List[str]
    
    # Metadados
    analyst: str
    confidence_score: float
    processing_time_ms: float
    data_quality_score: float

class ComprehensiveAnalysisLogger:
    """Logger abrangente para análises de trading"""
    
    def __init__(self, base_dir: str = "analysis_logs"):
        self.base_dir = base_dir
        self.session_id = f"analysis_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Criar diretórios
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "technical"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "fundamental"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "risk"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "market_conditions"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "comprehensive"), exist_ok=True)
        
        # Arquivos de log
        self.technical_log = os.path.join(base_dir, "technical", f"technical_{datetime.now().strftime('%Y%m%d')}.jsonl")
        self.fundamental_log = os.path.join(base_dir, "fundamental", f"fundamental_{datetime.now().strftime('%Y%m%d')}.jsonl")
        self.risk_log = os.path.join(base_dir, "risk", f"risk_{datetime.now().strftime('%Y%m%d')}.jsonl")
        self.market_log = os.path.join(base_dir, "market_conditions", f"market_{datetime.now().strftime('%Y%m%d')}.jsonl")
        self.comprehensive_log = os.path.join(base_dir, "comprehensive", f"comprehensive_{datetime.now().strftime('%Y%m%d')}.jsonl")
        
        # Cache em memória
        self.analysis_cache: List[AnalysisData] = []
        self.max_cache_size = 100
        
        print(f"ComprehensiveAnalysisLogger inicializado - Sessão: {self.session_id}")
    
    def log_technical_analysis(self, 
                             symbol: str,
                             timeframe: str,
                             indicators: List[Dict[str, Any]],
                             price_levels: List[Dict[str, Any]],
                             chart_patterns: List[Dict[str, Any]] = None,
                             analyst: str = "system",
                             confidence: float = 0.8) -> str:
        """Registra análise técnica completa"""
        
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        # Converter indicadores
        tech_indicators = []
        for ind in indicators:
            tech_indicators.append(TechnicalIndicator(
                name=ind.get('name', ''),
                value=ind.get('value', 0),
                timeframe=ind.get('timeframe', timeframe),
                signal=ind.get('signal', 'neutral'),
                confidence=ind.get('confidence', 0.5),
                parameters=ind.get('parameters', {}),
                timestamp=timestamp
            ))
        
        # Converter níveis de preço
        price_lvls = []
        for level in price_levels:
            price_lvls.append(PriceLevel(
                level=level.get('level', 0),
                type=level.get('type', 'unknown'),
                strength=level.get('strength', 0.5),
                touches=level.get('touches', 0),
                last_test=level.get('last_test', timestamp),
                description=level.get('description', '')
            ))
        
        # Criar dados de análise
        analysis_data = AnalysisData(
            analysis_id=analysis_id,
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            analysis_type=AnalysisType.TECHNICAL,
            data_sources=[DataSource.TECHNICAL_INDICATOR, DataSource.PRICE_ACTION],
            technical_indicators=tech_indicators,
            price_levels=price_lvls,
            chart_patterns=chart_patterns or [],
            economic_events=[],
            news_items=[],
            risk_metrics=[],
            market_conditions=[],
            correlations={},
            sentiment_data={},
            key_observations=[],
            risk_factors=[],
            opportunities=[],
            recommendations=[],
            analyst=analyst,
            confidence_score=confidence,
            processing_time_ms=0,
            data_quality_score=0.8
        )
        
        # Salvar
        self._save_analysis(analysis_data, self.technical_log)
        self._add_to_cache(analysis_data)
        
        return analysis_id
    
    def log_fundamental_analysis(self,
                               symbol: str,
                               economic_events: List[Dict[str, Any]],
                               news_items: List[Dict[str, Any]] = None,
                               analyst: str = "system",
                               confidence: float = 0.7) -> str:
        """Registra análise fundamental completa"""
        
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        # Converter eventos econômicos
        econ_events = []
        for event in economic_events:
            econ_events.append(EconomicEvent(
                title=event.get('title', ''),
                country=event.get('country', ''),
                currency=event.get('currency', ''),
                impact=event.get('impact', 'medium'),
                forecast=event.get('forecast'),
                previous=event.get('previous'),
                actual=event.get('actual'),
                time=event.get('time', timestamp),
                relevance_score=event.get('relevance_score', 0.5)
            ))
        
        # Converter notícias
        news_list = []
        if news_items:
            for news in news_items:
                news_list.append(NewsItem(
                    headline=news.get('headline', ''),
                    source=news.get('source', ''),
                    sentiment=news.get('sentiment', 'neutral'),
                    relevance=news.get('relevance', 0.5),
                    impact_currencies=news.get('impact_currencies', []),
                    timestamp=news.get('timestamp', timestamp),
                    summary=news.get('summary', '')
                ))
        
        # Criar dados de análise
        analysis_data = AnalysisData(
            analysis_id=analysis_id,
            timestamp=timestamp,
            symbol=symbol,
            timeframe="D1",
            analysis_type=AnalysisType.FUNDAMENTAL,
            data_sources=[DataSource.ECONOMIC_CALENDAR, DataSource.NEWS_FEED],
            technical_indicators=[],
            price_levels=[],
            chart_patterns=[],
            economic_events=econ_events,
            news_items=news_list,
            risk_metrics=[],
            market_conditions=[],
            correlations={},
            sentiment_data={},
            key_observations=[],
            risk_factors=[],
            opportunities=[],
            recommendations=[],
            analyst=analyst,
            confidence_score=confidence,
            processing_time_ms=0,
            data_quality_score=0.7
        )
        
        # Salvar
        self._save_analysis(analysis_data, self.fundamental_log)
        self._add_to_cache(analysis_data)
        
        return analysis_id
    
    def log_risk_analysis(self,
                         symbol: str,
                         risk_metrics: List[Dict[str, Any]],
                         market_conditions: List[Dict[str, Any]] = None,
                         correlations: Dict[str, float] = None,
                         analyst: str = "system",
                         confidence: float = 0.9) -> str:
        """Registra análise de risco completa"""
        
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        # Converter métricas de risco
        risk_list = []
        for metric in risk_metrics:
            risk_list.append(RiskMetric(
                name=metric.get('name', ''),
                value=metric.get('value', 0),
                threshold=metric.get('threshold', 0),
                status=metric.get('status', 'normal'),
                description=metric.get('description', ''),
                recommendation=metric.get('recommendation', '')
            ))
        
        # Converter condições de mercado
        market_conds = []
        if market_conditions:
            for condition in market_conditions:
                market_conds.append(MarketCondition(
                    aspect=condition.get('aspect', ''),
                    value=condition.get('value', 0),
                    classification=condition.get('classification', ''),
                    impact_on_trading=condition.get('impact_on_trading', ''),
                    confidence=condition.get('confidence', 0.5)
                ))
        
        # Criar dados de análise
        analysis_data = AnalysisData(
            analysis_id=analysis_id,
            timestamp=timestamp,
            symbol=symbol,
            timeframe="ALL",
            analysis_type=AnalysisType.RISK_MANAGEMENT,
            data_sources=[DataSource.MT5, DataSource.MANUAL_ANALYSIS],
            technical_indicators=[],
            price_levels=[],
            chart_patterns=[],
            economic_events=[],
            news_items=[],
            risk_metrics=risk_list,
            market_conditions=market_conds,
            correlations=correlations or {},
            sentiment_data={},
            key_observations=[],
            risk_factors=[],
            opportunities=[],
            recommendations=[],
            analyst=analyst,
            confidence_score=confidence,
            processing_time_ms=0,
            data_quality_score=0.9
        )
        
        # Salvar
        self._save_analysis(analysis_data, self.risk_log)
        self._add_to_cache(analysis_data)
        
        return analysis_id
    
    def log_comprehensive_analysis(self,
                                 symbol: str,
                                 timeframe: str,
                                 technical_data: Dict[str, Any] = None,
                                 fundamental_data: Dict[str, Any] = None,
                                 risk_data: Dict[str, Any] = None,
                                 sentiment_data: Dict[str, Any] = None,
                                 key_observations: List[str] = None,
                                 risk_factors: List[str] = None,
                                 opportunities: List[str] = None,
                                 recommendations: List[str] = None,
                                 analyst: str = "system",
                                 confidence: float = 0.8) -> str:
        """Registra análise abrangente completa"""
        
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        # Processar dados técnicos
        tech_indicators = []
        price_levels = []
        chart_patterns = []
        
        if technical_data:
            for ind in technical_data.get('indicators', []):
                tech_indicators.append(TechnicalIndicator(
                    name=ind.get('name', ''),
                    value=ind.get('value', 0),
                    timeframe=ind.get('timeframe', timeframe),
                    signal=ind.get('signal', 'neutral'),
                    confidence=ind.get('confidence', 0.5),
                    parameters=ind.get('parameters', {}),
                    timestamp=timestamp
                ))
            
            for level in technical_data.get('price_levels', []):
                price_levels.append(PriceLevel(
                    level=level.get('level', 0),
                    type=level.get('type', 'unknown'),
                    strength=level.get('strength', 0.5),
                    touches=level.get('touches', 0),
                    last_test=level.get('last_test', timestamp),
                    description=level.get('description', '')
                ))
            
            chart_patterns = technical_data.get('chart_patterns', [])
        
        # Processar dados fundamentais
        economic_events = []
        news_items = []
        
        if fundamental_data:
            for event in fundamental_data.get('economic_events', []):
                economic_events.append(EconomicEvent(
                    title=event.get('title', ''),
                    country=event.get('country', ''),
                    currency=event.get('currency', ''),
                    impact=event.get('impact', 'medium'),
                    forecast=event.get('forecast'),
                    previous=event.get('previous'),
                    actual=event.get('actual'),
                    time=event.get('time', timestamp),
                    relevance_score=event.get('relevance_score', 0.5)
                ))
            
            for news in fundamental_data.get('news_items', []):
                news_items.append(NewsItem(
                    headline=news.get('headline', ''),
                    source=news.get('source', ''),
                    sentiment=news.get('sentiment', 'neutral'),
                    relevance=news.get('relevance', 0.5),
                    impact_currencies=news.get('impact_currencies', []),
                    timestamp=news.get('timestamp', timestamp),
                    summary=news.get('summary', '')
                ))
        
        # Processar dados de risco
        risk_metrics = []
        market_conditions = []
        correlations = {}
        
        if risk_data:
            for metric in risk_data.get('risk_metrics', []):
                risk_metrics.append(RiskMetric(
                    name=metric.get('name', ''),
                    value=metric.get('value', 0),
                    threshold=metric.get('threshold', 0),
                    status=metric.get('status', 'normal'),
                    description=metric.get('description', ''),
                    recommendation=metric.get('recommendation', '')
                ))
            
            for condition in risk_data.get('market_conditions', []):
                market_conditions.append(MarketCondition(
                    aspect=condition.get('aspect', ''),
                    value=condition.get('value', 0),
                    classification=condition.get('classification', ''),
                    impact_on_trading=condition.get('impact_on_trading', ''),
                    confidence=condition.get('confidence', 0.5)
                ))
            
            correlations = risk_data.get('correlations', {})
        
        # Criar dados de análise completa
        analysis_data = AnalysisData(
            analysis_id=analysis_id,
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            analysis_type=AnalysisType.TECHNICAL,  # Tipo principal
            data_sources=[DataSource.MT5, DataSource.TECHNICAL_INDICATOR, 
                         DataSource.ECONOMIC_CALENDAR, DataSource.NEWS_FEED,
                         DataSource.MANUAL_ANALYSIS],
            technical_indicators=tech_indicators,
            price_levels=price_levels,
            chart_patterns=chart_patterns,
            economic_events=economic_events,
            news_items=news_items,
            risk_metrics=risk_metrics,
            market_conditions=market_conditions,
            correlations=correlations,
            sentiment_data=sentiment_data or {},
            key_observations=key_observations or [],
            risk_factors=risk_factors or [],
            opportunities=opportunities or [],
            recommendations=recommendations or [],
            analyst=analyst,
            confidence_score=confidence,
            processing_time_ms=0,
            data_quality_score=0.8
        )
        
        # Salvar
        self._save_analysis(analysis_data, self.comprehensive_log)
        self._add_to_cache(analysis_data)
        
        return analysis_id
    
    def _save_analysis(self, analysis_data: AnalysisData, log_file: str):
        """Salva análise em arquivo JSONL"""
        try:
            # Converter para dict
            data_dict = asdict(analysis_data)
            
            # Converter datetime para string
            data_dict['timestamp'] = analysis_data.timestamp.isoformat()
            
            # Converter datetimes aninhados
            for indicator in data_dict['technical_indicators']:
                indicator['timestamp'] = indicator['timestamp'].isoformat() if isinstance(indicator['timestamp'], datetime) else indicator['timestamp']
            
            for level in data_dict['price_levels']:
                level['last_test'] = level['last_test'].isoformat() if isinstance(level['last_test'], datetime) else level['last_test']
            
            for event in data_dict['economic_events']:
                event['time'] = event['time'].isoformat() if isinstance(event['time'], datetime) else event['time']
            
            for news in data_dict['news_items']:
                news['timestamp'] = news['timestamp'].isoformat() if isinstance(news['timestamp'], datetime) else news['timestamp']
            
            # Converter enums para string
            data_dict['analysis_type'] = analysis_data.analysis_type.value
            data_dict['data_sources'] = [ds.value for ds in analysis_data.data_sources]
            
            # Salvar em arquivo
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data_dict, ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"Erro ao salvar análise: {e}")
    
    def _add_to_cache(self, analysis_data: AnalysisData):
        """Adiciona análise ao cache em memória"""
        self.analysis_cache.append(analysis_data)
        
        # Manter tamanho do cache
        if len(self.analysis_cache) > self.max_cache_size:
            self.analysis_cache.pop(0)
    
    def get_recent_analyses(self, limit: int = 10, analysis_type: AnalysisType = None) -> List[AnalysisData]:
        """Obtém análises recentes do cache"""
        analyses = self.analysis_cache
        
        if analysis_type:
            analyses = [a for a in analyses if a.analysis_type == analysis_type]
        
        return analyses[-limit:]
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Obtém resumo das análises da sessão"""
        total_analyses = len(self.analysis_cache)
        
        by_type = {}
        by_symbol = {}
        avg_confidence = 0
        
        for analysis in self.analysis_cache:
            # Por tipo
            type_key = analysis.analysis_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1
            
            # Por símbolo
            by_symbol[analysis.symbol] = by_symbol.get(analysis.symbol, 0) + 1
            
            # Confiança média
            avg_confidence += analysis.confidence_score
        
        if total_analyses > 0:
            avg_confidence /= total_analyses
        
        return {
            'session_id': self.session_id,
            'total_analyses': total_analyses,
            'analyses_by_type': by_type,
            'analyses_by_symbol': by_symbol,
            'average_confidence': round(avg_confidence, 3),
            'cache_size': len(self.analysis_cache),
            'log_files': {
                'technical': self.technical_log,
                'fundamental': self.fundamental_log,
                'risk': self.risk_log,
                'market': self.market_log,
                'comprehensive': self.comprehensive_log
            }
        }
    
    def export_session_report(self, output_file: str = None) -> str:
        """Exporta relatório da sessão"""
        if not output_file:
            output_file = os.path.join(self.base_dir, f"session_report_{self.session_id}.json")
        
        try:
            report = {
                'session_info': {
                    'session_id': self.session_id,
                    'start_time': datetime.now().isoformat(),
                    'total_analyses': len(self.analysis_cache)
                },
                'summary': self.get_analysis_summary(),
                'recent_analyses': []
            }
            
            # Adicionar análises recentes (últimas 20)
            for analysis in self.analysis_cache[-20:]:
                analysis_dict = {
                    'analysis_id': analysis.analysis_id,
                    'timestamp': analysis.timestamp.isoformat(),
                    'symbol': analysis.symbol,
                    'type': analysis.analysis_type.value,
                    'confidence': analysis.confidence_score,
                    'key_observations': analysis.key_observations,
                    'recommendations': analysis.recommendations
                }
                report['recent_analyses'].append(analysis_dict)
            
            # Salvar relatório
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            return output_file
            
        except Exception as e:
            print(f"Erro ao exportar relatório: {e}")
            return None

# Instância global
_comprehensive_logger = None

def get_comprehensive_logger() -> ComprehensiveAnalysisLogger:
    """Obtém instância global do logger abrangente"""
    global _comprehensive_logger
    if _comprehensive_logger is None:
        _comprehensive_logger = ComprehensiveAnalysisLogger()
    return _comprehensive_logger

def initialize_comprehensive_logger(base_dir: str = "analysis_logs") -> ComprehensiveAnalysisLogger:
    """Inicializa logger abrangente"""
    global _comprehensive_logger
    _comprehensive_logger = ComprehensiveAnalysisLogger(base_dir)
    return _comprehensive_logger