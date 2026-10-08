import logging
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from contextlib import contextmanager

class DecisionType(Enum):
    """Tipos de decisão no sistema de trading"""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    POSITION_SIZING = "position_sizing"
    RISK_MANAGEMENT = "risk_management"
    MARKET_ANALYSIS = "market_analysis"
    SYSTEM_OPERATION = "system_operation"
    ERROR_HANDLING = "error_handling"
    CONFIGURATION = "configuration"

class DecisionLevel(Enum):
    """Níveis de criticidade das decisões"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class DecisionCriteria:
    """Critérios utilizados na tomada de decisão"""
    name: str
    value: Any
    weight: float
    threshold: Optional[float] = None
    operator: Optional[str] = None  # >, <, >=, <=, ==, !=
    description: Optional[str] = None

@dataclass
class DecisionContext:
    """Contexto da decisão"""
    market_conditions: Dict[str, Any]
    account_state: Dict[str, Any]
    system_state: Dict[str, Any]
    external_factors: Dict[str, Any]
    timestamp: datetime

@dataclass
class DecisionRecord:
    """Registro completo de uma decisão"""
    decision_id: str
    timestamp: datetime
    decision_type: DecisionType
    level: DecisionLevel
    description: str
    criteria: List[DecisionCriteria]
    context: DecisionContext
    justification: str
    outcome: str
    confidence_score: float
    execution_time_ms: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    related_decisions: List[str] = None
    metadata: Dict[str, Any] = None

class DecisionLogger:
    """Sistema de logging detalhado para decisões de trading"""
    
    def __init__(self, log_dir: str = "logs", max_file_size_mb: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.current_session = self._generate_session_id()
        self.decision_counter = 0
        self._lock = threading.Lock()
        
        # Configurar loggers
        self._setup_loggers()
        
        # Cache de decisões em memória para análise rápida
        self.decision_cache: List[DecisionRecord] = []
        self.max_cache_size = 1000
    
    def _generate_session_id(self) -> str:
        """Gera ID único para a sessão"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 10000}"
    
    def _setup_loggers(self):
        """Configura os loggers especializados"""
        # Logger para decisões estruturadas (JSON)
        self.structured_logger = logging.getLogger('decision_structured')
        self.structured_logger.setLevel(logging.INFO)
        
        structured_handler = logging.FileHandler(
            self.log_dir / f"decisions_structured_{datetime.now().strftime('%Y%m%d')}.jsonl",
            encoding='utf-8'
        )
        structured_handler.setFormatter(logging.Formatter('%(message)s'))
        self.structured_logger.addHandler(structured_handler)
        
        # Logger para decisões legíveis (texto)
        self.readable_logger = logging.getLogger('decision_readable')
        self.readable_logger.setLevel(logging.INFO)
        
        readable_handler = logging.FileHandler(
            self.log_dir / f"decisions_readable_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        readable_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        readable_handler.setFormatter(readable_formatter)
        self.readable_logger.addHandler(readable_handler)
        
        # Logger para análise de performance
        self.performance_logger = logging.getLogger('decision_performance')
        self.performance_logger.setLevel(logging.INFO)
        
        performance_handler = logging.FileHandler(
            self.log_dir / f"decisions_performance_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        performance_handler.setFormatter(readable_formatter)
        self.performance_logger.addHandler(performance_handler)
    
    def _generate_decision_id(self) -> str:
        """Gera ID único para a decisão"""
        with self._lock:
            self.decision_counter += 1
            return f"{self.current_session}_D{self.decision_counter:06d}"
    
    @contextmanager
    def decision_context(self, decision_type: DecisionType, description: str, 
                        level: DecisionLevel = DecisionLevel.MEDIUM):
        """Context manager para logging automático de decisões"""
        start_time = time.time()
        decision_id = self._generate_decision_id()
        
        context_data = {
            'decision_id': decision_id,
            'start_time': start_time,
            'decision_type': decision_type,
            'description': description,
            'level': level,
            'criteria': [],
            'context': None,
            'justification': '',
            'outcome': '',
            'confidence_score': 0.0
        }
        
        try:
            yield DecisionContextManager(context_data, self)
        finally:
            execution_time = (time.time() - start_time) * 1000
            self._finalize_decision(context_data, execution_time)
    
    def _finalize_decision(self, context_data: Dict, execution_time: float):
        """Finaliza e registra a decisão"""
        decision_record = DecisionRecord(
            decision_id=context_data['decision_id'],
            timestamp=datetime.now(timezone.utc),
            decision_type=context_data['decision_type'],
            level=context_data['level'],
            description=context_data['description'],
            criteria=context_data['criteria'],
            context=context_data['context'] or self._get_default_context(),
            justification=context_data['justification'],
            outcome=context_data['outcome'],
            confidence_score=context_data['confidence_score'],
            execution_time_ms=execution_time,
            session_id=self.current_session,
            related_decisions=[],
            metadata={}
        )
        
        self._log_decision(decision_record)
    
    def _get_default_context(self) -> DecisionContext:
        """Obtém contexto padrão quando não fornecido"""
        return DecisionContext(
            market_conditions={"status": "unknown"},
            account_state={"status": "unknown"},
            system_state={"status": "active"},
            external_factors={},
            timestamp=datetime.now(timezone.utc)
        )
    
    def _log_decision(self, decision: DecisionRecord):
        """Registra a decisão nos logs"""
        with self._lock:
            # Log estruturado (JSON)
            decision_dict = asdict(decision)
            decision_dict['timestamp'] = decision.timestamp.isoformat()
            decision_dict['context']['timestamp'] = decision.context.timestamp.isoformat()
            decision_dict['decision_type'] = decision.decision_type.value
            decision_dict['level'] = decision.level.value
            
            self.structured_logger.info(json.dumps(decision_dict, ensure_ascii=False))
            
            # Log legível
            readable_msg = self._format_readable_decision(decision)
            self.readable_logger.info(readable_msg)
            
            # Log de performance
            performance_msg = (
                f"Decision {decision.decision_id} | Type: {decision.decision_type.value} | "
                f"Level: {decision.level.value} | Execution: {decision.execution_time_ms:.2f}ms | "
                f"Confidence: {decision.confidence_score:.2f}"
            )
            self.performance_logger.info(performance_msg)
            
            # Adicionar ao cache
            self.decision_cache.append(decision)
            if len(self.decision_cache) > self.max_cache_size:
                self.decision_cache.pop(0)
    
    def _format_readable_decision(self, decision: DecisionRecord) -> str:
        """Formata decisão para log legível"""
        lines = [
            f"DECISÃO {decision.decision_id} [{decision.decision_type.value.upper()}]",
            f"Descrição: {decision.description}",
            f"Nível: {decision.level.value.upper()}",
            f"Tempo de Execução: {decision.execution_time_ms:.2f}ms",
            f"Confiança: {decision.confidence_score:.2f}",
            "",
            "CRITÉRIOS:"
        ]
        
        for i, criteria in enumerate(decision.criteria, 1):
            criteria_line = f"  {i}. {criteria.name}: {criteria.value}"
            if criteria.threshold is not None:
                criteria_line += f" (threshold: {criteria.threshold}, weight: {criteria.weight})"
            if criteria.description:
                criteria_line += f" - {criteria.description}"
            lines.append(criteria_line)
        
        lines.extend([
            "",
            f"JUSTIFICATIVA: {decision.justification}",
            f"RESULTADO: {decision.outcome}",
            "-" * 80
        ])
        
        return "\n".join(lines)
    
    def get_recent_decisions(self, count: int = 10, 
                           decision_type: Optional[DecisionType] = None) -> List[DecisionRecord]:
        """Obtém decisões recentes do cache"""
        decisions = self.decision_cache
        
        if decision_type:
            decisions = [d for d in decisions if d.decision_type == decision_type]
        
        return decisions[-count:] if decisions else []
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas das decisões"""
        if not self.decision_cache:
            return {}
        
        stats = {
            'total_decisions': len(self.decision_cache),
            'by_type': {},
            'by_level': {},
            'avg_execution_time': 0,
            'avg_confidence': 0,
            'session_id': self.current_session
        }
        
        total_time = 0
        total_confidence = 0
        
        for decision in self.decision_cache:
            # Por tipo
            type_key = decision.decision_type.value
            stats['by_type'][type_key] = stats['by_type'].get(type_key, 0) + 1
            
            # Por nível
            level_key = decision.level.value
            stats['by_level'][level_key] = stats['by_level'].get(level_key, 0) + 1
            
            # Métricas
            total_time += decision.execution_time_ms
            total_confidence += decision.confidence_score
        
        stats['avg_execution_time'] = total_time / len(self.decision_cache)
        stats['avg_confidence'] = total_confidence / len(self.decision_cache)
        
        return stats

class DecisionContextManager:
    """Gerenciador de contexto para decisões"""
    
    def __init__(self, context_data: Dict, logger: DecisionLogger):
        self.context_data = context_data
        self.logger = logger
    
    def add_criteria(self, name: str, value: Any, weight: float = 1.0, 
                    threshold: Optional[float] = None, operator: Optional[str] = None,
                    description: Optional[str] = None):
        """Adiciona critério à decisão"""
        criteria = DecisionCriteria(
            name=name,
            value=value,
            weight=weight,
            threshold=threshold,
            operator=operator,
            description=description
        )
        self.context_data['criteria'].append(criteria)
    
    def set_context(self, market_conditions: Dict = None, account_state: Dict = None,
                   system_state: Dict = None, external_factors: Dict = None):
        """Define contexto da decisão"""
        self.context_data['context'] = DecisionContext(
            market_conditions=market_conditions or {},
            account_state=account_state or {},
            system_state=system_state or {},
            external_factors=external_factors or {},
            timestamp=datetime.now(timezone.utc)
        )
    
    def set_justification(self, justification: str):
        """Define justificativa da decisão"""
        self.context_data['justification'] = justification
    
    def set_outcome(self, outcome: str):
        """Define resultado da decisão"""
        self.context_data['outcome'] = outcome
    
    def set_confidence(self, confidence: float):
        """Define nível de confiança (0.0 a 1.0)"""
        self.context_data['confidence_score'] = max(0.0, min(1.0, confidence))

# Instância global do logger
_decision_logger = None

def get_decision_logger() -> DecisionLogger:
    """Obtém instância global do decision logger"""
    global _decision_logger
    if _decision_logger is None:
        _decision_logger = DecisionLogger()
    return _decision_logger

# Funções de conveniência
def log_trade_decision(description: str, criteria: List[tuple], justification: str, 
                      outcome: str, confidence: float = 0.5, 
                      level: DecisionLevel = DecisionLevel.HIGH):
    """Log rápido para decisões de trade"""
    logger = get_decision_logger()
    
    with logger.decision_context(DecisionType.TRADE_ENTRY, description, level) as ctx:
        for name, value, weight in criteria:
            ctx.add_criteria(name, value, weight)
        ctx.set_justification(justification)
        ctx.set_outcome(outcome)
        ctx.set_confidence(confidence)

def log_risk_decision(description: str, risk_metrics: Dict, justification: str,
                     outcome: str, confidence: float = 0.7):
    """Log rápido para decisões de risco"""
    logger = get_decision_logger()
    
    with logger.decision_context(DecisionType.RISK_MANAGEMENT, description, 
                               DecisionLevel.CRITICAL) as ctx:
        for metric, value in risk_metrics.items():
            ctx.add_criteria(metric, value, 1.0)
        ctx.set_justification(justification)
        ctx.set_outcome(outcome)
        ctx.set_confidence(confidence)

def log_system_decision(description: str, justification: str, outcome: str,
                       level: DecisionLevel = DecisionLevel.MEDIUM):
    """Log rápido para decisões do sistema"""
    logger = get_decision_logger()
    
    with logger.decision_context(DecisionType.SYSTEM_OPERATION, description, level) as ctx:
        ctx.set_justification(justification)
        ctx.set_outcome(outcome)
        ctx.set_confidence(0.8)

if __name__ == "__main__":
    # Exemplo de uso
    logger = get_decision_logger()
    
    # Exemplo 1: Decisão de entrada em trade
    with logger.decision_context(DecisionType.TRADE_ENTRY, 
                               "Análise de entrada EURUSD", 
                               DecisionLevel.HIGH) as ctx:
        
        ctx.add_criteria("RSI", 25.5, weight=0.3, threshold=30, operator="<", 
                        description="RSI indica sobrevenda")
        ctx.add_criteria("MACD_Signal", "bullish_crossover", weight=0.4, 
                        description="MACD cruzou para cima")
        ctx.add_criteria("Support_Level", 1.0850, weight=0.3, 
                        description="Preço próximo ao suporte")
        
        ctx.set_context(
            market_conditions={"trend": "sideways", "volatility": "medium"},
            account_state={"balance": 10000, "equity": 9850, "margin_level": 250},
            system_state={"ml_confidence": 0.75, "signal_strength": "strong"}
        )
        
        ctx.set_justification(
            "Confluência de sinais técnicos: RSI em sobrevenda (25.5), "
            "MACD com cruzamento bullish e preço testando suporte em 1.0850. "
            "ML model indica 75% de confiança para movimento de alta."
        )
        
        ctx.set_outcome("Ordem de compra executada: 0.1 lote EURUSD @ 1.0852")
        ctx.set_confidence(0.75)
    
    print("Sistema de logging de decisões inicializado com sucesso!")
    print(f"Logs salvos em: {logger.log_dir}")
    print(f"Sessão atual: {logger.current_session}")