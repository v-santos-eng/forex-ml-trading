import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
import threading
from enum import Enum

class DecisionType(Enum):
    """Tipos de decisão do sistema"""
    TRADE_EXECUTION = "trade_execution"
    TRADE_ANALYSIS = "trade_analysis"
    NO_TRADE = "no_trade"
    BALANCE_UPDATE = "balance_update"
    MARKET_ANALYSIS = "market_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    SYSTEM_OPERATION = "system_operation"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE_MONITORING = "performance_monitoring"

class LogLevel(Enum):
    """Níveis de log"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class MarketConditions:
    """Condições de mercado no momento da decisão"""
    symbol: str
    bid: float
    ask: float
    spread: float
    volatility: float
    trend_direction: str
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    moving_average_20: Optional[float] = None
    moving_average_50: Optional[float] = None
    volume: Optional[int] = None
    news_sentiment: Optional[str] = None

@dataclass
class AccountState:
    """Estado da conta no momento da decisão"""
    balance: float
    equity: float
    margin: float
    margin_level: float
    free_margin: float
    profit: float
    open_positions: int
    pending_orders: int
    daily_profit: float
    weekly_profit: float
    monthly_profit: float
    max_drawdown: float
    win_rate: float
    risk_percentage: float

@dataclass
class TradeDetails:
    """Detalhes de uma operação de trading"""
    symbol: str
    action: str  # BUY, SELL, CLOSE
    volume: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    order_type: str = "MARKET"
    ticket: Optional[int] = None
    profit: Optional[float] = None
    commission: Optional[float] = None
    swap: Optional[float] = None
    duration_minutes: Optional[int] = None

@dataclass
class DetailedDecision:
    """Estrutura completa de uma decisão detalhada"""
    decision_id: str
    timestamp: str
    decision_type: DecisionType
    level: LogLevel
    description: str
    
    # Contexto da decisão
    market_conditions: Optional[MarketConditions]
    account_state: Optional[AccountState]
    trade_details: Optional[TradeDetails]
    
    # Análise e justificativa
    analysis_factors: List[str]
    decision_criteria: List[str]
    justification: str
    confidence_score: float
    
    # Resultado
    outcome: str
    execution_time_ms: float
    
    # Metadados
    session_id: str
    user_id: Optional[str] = None
    related_decisions: List[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.related_decisions is None:
            self.related_decisions = []
        if self.tags is None:
            self.tags = []

class DetailedDecisionLogger:
    """Sistema de logging detalhado para decisões de trading"""
    
    def __init__(self, log_directory: str = "logs", session_id: str = None):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.decision_counter = 0
        self.lock = threading.Lock()
        
        # Arquivos de log
        today = datetime.now().strftime('%Y%m%d')
        self.structured_log_file = self.log_directory / f"detailed_decisions_{today}.jsonl"
        self.readable_log_file = self.log_directory / f"detailed_decisions_readable_{today}.log"
        self.balance_log_file = self.log_directory / f"balance_updates_{today}.log"
        self.operations_log_file = self.log_directory / f"trading_operations_{today}.log"
        self.analysis_log_file = self.log_directory / f"market_analysis_{today}.log"
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Cache de decisões em memória
        self.recent_decisions: List[DetailedDecision] = []
        self.max_memory_decisions = 1000
        
        self.logger.info(f"Sistema de logging detalhado inicializado - Sessão: {self.session_id}")
    
    def _setup_logger(self) -> logging.Logger:
        """Configura o sistema de logging"""
        logger = logging.getLogger(f"DetailedDecisionLogger_{self.session_id}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Handler para arquivo geral
            handler = logging.FileHandler(self.readable_log_file, encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Handler para console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _generate_decision_id(self) -> str:
        """Gera ID único para a decisão"""
        with self.lock:
            self.decision_counter += 1
            return f"{self.session_id}_D{self.decision_counter:06d}"
    
    def log_balance_update(self, 
                          old_balance: float, 
                          new_balance: float, 
                          old_equity: float, 
                          new_equity: float,
                          reason: str,
                          account_state: AccountState,
                          additional_info: Dict[str, Any] = None) -> str:
        """Registra atualização de saldo detalhadamente"""
        
        decision_id = self._generate_decision_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        balance_change = new_balance - old_balance
        equity_change = new_equity - old_equity
        
        analysis_factors = [
            f"Saldo anterior: ${old_balance:,.2f}",
            f"Saldo atual: ${new_balance:,.2f}",
            f"Variação do saldo: ${balance_change:+,.2f}",
            f"Equity anterior: ${old_equity:,.2f}",
            f"Equity atual: ${new_equity:,.2f}",
            f"Variação da equity: ${equity_change:+,.2f}",
            f"Margem livre: ${account_state.free_margin:,.2f}",
            f"Nível de margem: {account_state.margin_level:.2f}%",
            f"Posições abertas: {account_state.open_positions}",
            f"Drawdown máximo: {account_state.max_drawdown:.2f}%"
        ]
        
        if additional_info:
            for key, value in additional_info.items():
                analysis_factors.append(f"{key}: {value}")
        
        decision = DetailedDecision(
            decision_id=decision_id,
            timestamp=timestamp,
            decision_type=DecisionType.BALANCE_UPDATE,
            level=LogLevel.HIGH if abs(balance_change) > 100 else LogLevel.MEDIUM,
            description=f"Atualização de saldo: {reason}",
            market_conditions=None,
            account_state=account_state,
            trade_details=None,
            analysis_factors=analysis_factors,
            decision_criteria=[
                "Monitoramento contínuo de saldo",
                "Verificação de integridade da conta",
                "Controle de risco"
            ],
            justification=f"Saldo atualizado devido a: {reason}. Variação: ${balance_change:+,.2f}",
            confidence_score=1.0,
            outcome=f"Saldo: ${new_balance:,.2f} | Equity: ${new_equity:,.2f}",
            execution_time_ms=0.0,
            session_id=self.session_id,
            tags=["balance", "monitoring", "account_state"]
        )
        
        self._save_decision(decision)
        
        # Log específico para balanço
        balance_msg = (
            f"[SALDO] {timestamp} | "
            f"Anterior: ${old_balance:,.2f} → Atual: ${new_balance:,.2f} | "
            f"Variação: ${balance_change:+,.2f} | "
            f"Motivo: {reason} | "
            f"Equity: ${old_equity:,.2f} → ${new_equity:,.2f} | "
            f"Margem Livre: ${account_state.free_margin:,.2f}"
        )
        
        with open(self.balance_log_file, 'a', encoding='utf-8') as f:
            f.write(balance_msg + "\n")
        
        return decision_id
    
    def log_trade_operation(self,
                           trade_details: TradeDetails,
                           market_conditions: MarketConditions,
                           account_state: AccountState,
                           decision_factors: List[str],
                           justification: str,
                           confidence: float,
                           outcome: str) -> str:
        """Registra operação de trading detalhadamente"""
        
        decision_id = self._generate_decision_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        analysis_factors = [
            f"Símbolo: {trade_details.symbol}",
            f"Ação: {trade_details.action}",
            f"Volume: {trade_details.volume}",
            f"Preço: {trade_details.price}",
            f"Spread atual: {market_conditions.spread}",
            f"Volatilidade: {market_conditions.volatility}",
            f"Tendência: {market_conditions.trend_direction}",
            f"RSI: {market_conditions.rsi}",
            f"MACD: {market_conditions.macd}",
            f"Saldo disponível: ${account_state.balance:,.2f}",
            f"Margem livre: ${account_state.free_margin:,.2f}",
            f"Risco da operação: {account_state.risk_percentage:.2f}%"
        ]
        
        if trade_details.stop_loss:
            analysis_factors.append(f"Stop Loss: {trade_details.stop_loss}")
        if trade_details.take_profit:
            analysis_factors.append(f"Take Profit: {trade_details.take_profit}")
        
        decision = DetailedDecision(
            decision_id=decision_id,
            timestamp=timestamp,
            decision_type=DecisionType.TRADE_EXECUTION,
            level=LogLevel.HIGH,
            description=f"Operação {trade_details.action} {trade_details.symbol}",
            market_conditions=market_conditions,
            account_state=account_state,
            trade_details=trade_details,
            analysis_factors=analysis_factors,
            decision_criteria=decision_factors,
            justification=justification,
            confidence_score=confidence,
            outcome=outcome,
            execution_time_ms=0.0,
            session_id=self.session_id,
            tags=["trade", "execution", trade_details.symbol.lower(), trade_details.action.lower()]
        )
        
        self._save_decision(decision)
        
        # Log específico para operações
        operation_msg = (
            f"[OPERAÇÃO] {timestamp} | "
            f"{trade_details.action} {trade_details.volume} {trade_details.symbol} @ {trade_details.price} | "
            f"Spread: {market_conditions.spread} | "
            f"Confiança: {confidence:.2f} | "
            f"Justificativa: {justification} | "
            f"Resultado: {outcome}"
        )
        
        with open(self.operations_log_file, 'a', encoding='utf-8') as f:
            f.write(operation_msg + "\n")
        
        return decision_id
    
    def log_no_trade_decision(self,
                             symbol: str,
                             market_conditions: MarketConditions,
                             account_state: AccountState,
                             reasons: List[str],
                             analysis_performed: List[str],
                             confidence: float) -> str:
        """Registra decisão de não operar e os motivos"""
        
        decision_id = self._generate_decision_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        analysis_factors = [
            f"Símbolo analisado: {symbol}",
            f"Preço atual (bid/ask): {market_conditions.bid}/{market_conditions.ask}",
            f"Spread: {market_conditions.spread}",
            f"Volatilidade: {market_conditions.volatility}",
            f"Tendência: {market_conditions.trend_direction}",
            f"RSI: {market_conditions.rsi}",
            f"MACD: {market_conditions.macd}",
            f"Saldo disponível: ${account_state.balance:,.2f}",
            f"Posições abertas: {account_state.open_positions}",
            f"Nível de margem: {account_state.margin_level:.2f}%"
        ]
        
        analysis_factors.extend(analysis_performed)
        
        decision = DetailedDecision(
            decision_id=decision_id,
            timestamp=timestamp,
            decision_type=DecisionType.NO_TRADE,
            level=LogLevel.MEDIUM,
            description=f"Decisão de não operar {symbol}",
            market_conditions=market_conditions,
            account_state=account_state,
            trade_details=None,
            analysis_factors=analysis_factors,
            decision_criteria=reasons,
            justification=f"Não operação devido a: {'; '.join(reasons)}",
            confidence_score=confidence,
            outcome="Nenhuma operação executada - Aguardando melhores condições",
            execution_time_ms=0.0,
            session_id=self.session_id,
            tags=["no_trade", "analysis", symbol.lower(), "waiting"]
        )
        
        self._save_decision(decision)
        
        return decision_id
    
    def log_market_analysis(self,
                           symbols: List[str],
                           analysis_results: Dict[str, Any],
                           observations: List[str],
                           recommendations: List[str],
                           confidence: float) -> str:
        """Registra análise de mercado detalhada"""
        
        decision_id = self._generate_decision_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        analysis_factors = [
            f"Símbolos analisados: {', '.join(symbols)}",
            f"Período de análise: {analysis_results.get('period', 'N/A')}",
            f"Indicadores utilizados: {', '.join(analysis_results.get('indicators', []))}",
        ]
        
        analysis_factors.extend(observations)
        
        for symbol, data in analysis_results.get('symbol_data', {}).items():
            analysis_factors.append(f"{symbol}: {data}")
        
        decision = DetailedDecision(
            decision_id=decision_id,
            timestamp=timestamp,
            decision_type=DecisionType.MARKET_ANALYSIS,
            level=LogLevel.MEDIUM,
            description=f"Análise de mercado - {len(symbols)} símbolos",
            market_conditions=None,
            account_state=None,
            trade_details=None,
            analysis_factors=analysis_factors,
            decision_criteria=recommendations,
            justification=f"Análise completa de {len(symbols)} símbolos com {len(observations)} observações",
            confidence_score=confidence,
            outcome=f"Análise concluída - {len(recommendations)} recomendações geradas",
            execution_time_ms=0.0,
            session_id=self.session_id,
            tags=["market_analysis", "indicators"] + [s.lower() for s in symbols]
        )
        
        self._save_decision(decision)
        
        # Log específico para análises
        analysis_msg = (
            f"[ANÁLISE] {timestamp} | "
            f"Símbolos: {', '.join(symbols)} | "
            f"Observações: {len(observations)} | "
            f"Recomendações: {len(recommendations)} | "
            f"Confiança: {confidence:.2f}"
        )
        
        with open(self.analysis_log_file, 'a', encoding='utf-8') as f:
            f.write(analysis_msg + "\n")
            for obs in observations:
                f.write(f"  → {obs}\n")
            for rec in recommendations:
                f.write(f"  ★ {rec}\n")
            f.write("\n")
        
        return decision_id
    
    def _save_decision(self, decision: DetailedDecision) -> None:
        """Salva decisão nos arquivos de log"""
        with self.lock:
            # Adiciona à memória
            self.recent_decisions.append(decision)
            if len(self.recent_decisions) > self.max_memory_decisions:
                self.recent_decisions = self.recent_decisions[-self.max_memory_decisions:]
            
            # Salva no arquivo estruturado (JSONL)
            with open(self.structured_log_file, 'a', encoding='utf-8') as f:
                json.dump(asdict(decision), f, ensure_ascii=False, default=str)
                f.write('\n')
            
            # Log legível
            readable_msg = (
                f"\n{'='*80}\n"
                f"DECISÃO: {decision.description}\n"
                f"ID: {decision.decision_id}\n"
                f"Timestamp: {decision.timestamp}\n"
                f"Tipo: {decision.decision_type.value}\n"
                f"Nível: {decision.level.value}\n"
                f"Confiança: {decision.confidence_score:.2f}\n"
                f"\nANÁLISE REALIZADA:\n"
            )
            
            for factor in decision.analysis_factors:
                readable_msg += f"  • {factor}\n"
            
            readable_msg += f"\nCRITÉRIOS DE DECISÃO:\n"
            for criteria in decision.decision_criteria:
                readable_msg += f"  ✓ {criteria}\n"
            
            readable_msg += (
                f"\nJUSTIFICATIVA:\n{decision.justification}\n"
                f"\nRESULTADO:\n{decision.outcome}\n"
                f"{'='*80}\n"
            )
            
            self.logger.info(readable_msg)
    
    def get_recent_decisions(self, count: int = 50) -> List[Dict[str, Any]]:
        """Retorna decisões recentes"""
        with self.lock:
            recent = self.recent_decisions[-count:] if self.recent_decisions else []
            return [asdict(decision) for decision in recent]
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas da sessão"""
        with self.lock:
            if not self.recent_decisions:
                return {'total_decisions': 0}
            
            decisions_by_type = {}
            total_confidence = 0
            
            for decision in self.recent_decisions:
                decision_type = decision.decision_type.value
                decisions_by_type[decision_type] = decisions_by_type.get(decision_type, 0) + 1
                total_confidence += decision.confidence_score
            
            return {
                'total_decisions': len(self.recent_decisions),
                'decisions_by_type': decisions_by_type,
                'average_confidence': total_confidence / len(self.recent_decisions),
                'session_id': self.session_id,
                'session_start': self.recent_decisions[0].timestamp if self.recent_decisions else None
            }

# Instância global para uso em todo o sistema
detailed_logger = None

def get_detailed_logger() -> DetailedDecisionLogger:
    """Retorna instância global do logger detalhado"""
    global detailed_logger
    if detailed_logger is None:
        detailed_logger = DetailedDecisionLogger()
    return detailed_logger

def initialize_detailed_logger(log_directory: str = "logs", session_id: str = None) -> DetailedDecisionLogger:
    """Inicializa o logger detalhado"""
    global detailed_logger
    detailed_logger = DetailedDecisionLogger(log_directory, session_id)
    return detailed_logger