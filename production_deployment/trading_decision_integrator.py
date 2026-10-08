import MetaTrader5 as mt5
from detailed_decision_logger import (
    DetailedDecisionLogger, MarketConditions, AccountState, TradeDetails,
    DecisionType, LogLevel, get_detailed_logger, initialize_detailed_logger
)
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime
import threading
import json
from pathlib import Path

class TradingDecisionIntegrator:
    """Integrador que conecta o sistema de trading com o logging detalhado"""
    
    def __init__(self, mt5_authenticator=None, session_id: str = None):
        self.mt5_auth = mt5_authenticator
        self.detailed_logger = initialize_detailed_logger(session_id=session_id)
        
        # Cache de estados anteriores para detectar mudanças
        self.last_balance = None
        self.last_equity = None
        self.last_account_state = None
        self.last_positions = {}
        
        # Configurações de monitoramento
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 5  # segundos
        
        # Histórico de análises
        self.analysis_history = []
        self.max_analysis_history = 100
        
        self.detailed_logger.logger.info("Integrador de decisões de trading inicializado")
    
    def start_monitoring(self) -> bool:
        """Inicia monitoramento contínuo"""
        if self.monitoring_active:
            return True
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.detailed_logger.logger.info("Monitoramento contínuo iniciado")
        return True
    
    def stop_monitoring(self) -> None:
        """Para monitoramento contínuo"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        self.detailed_logger.logger.info("Monitoramento contínuo parado")
    
    def _monitoring_loop(self) -> None:
        """Loop principal de monitoramento"""
        while self.monitoring_active:
            try:
                self._check_account_changes()
                self._check_position_changes()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.detailed_logger.logger.error(f"Erro no loop de monitoramento: {e}")
                time.sleep(30)  # Aguarda mais tempo em caso de erro
    
    def _get_current_account_state(self) -> Optional[AccountState]:
        """Obtém estado atual da conta"""
        try:
            if not self.mt5_auth or not self.mt5_auth.connected:
                return None
            
            if hasattr(mt5, 'account_info'):
                account_info = mt5.account_info()
                if account_info is None:
                    return None
                
                # Calcula métricas adicionais
                positions = mt5.positions_get()
                open_positions = len(positions) if positions else 0
                
                orders = mt5.orders_get()
                pending_orders = len(orders) if orders else 0
                
                # Calcula profit atual
                current_profit = sum(pos.profit for pos in positions) if positions else 0.0
                
                return AccountState(
                    balance=account_info.balance,
                    equity=account_info.equity,
                    margin=account_info.margin,
                    margin_level=account_info.margin_level if account_info.margin > 0 else 0.0,
                    free_margin=account_info.margin_free,
                    profit=current_profit,
                    open_positions=open_positions,
                    pending_orders=pending_orders,
                    daily_profit=0.0,  # Será calculado posteriormente
                    weekly_profit=0.0,  # Será calculado posteriormente
                    monthly_profit=0.0,  # Será calculado posteriormente
                    max_drawdown=0.0,  # Será calculado posteriormente
                    win_rate=0.0,  # Será calculado posteriormente
                    risk_percentage=0.0  # Será calculado posteriormente
                )
            else:
                # Modo simulação
                return AccountState(
                    balance=self.mt5_auth.account_info.get('balance', 10000.0),
                    equity=self.mt5_auth.account_info.get('equity', 10000.0),
                    margin=self.mt5_auth.account_info.get('margin', 0.0),
                    margin_level=0.0,
                    free_margin=self.mt5_auth.account_info.get('balance', 10000.0),
                    profit=0.0,
                    open_positions=0,
                    pending_orders=0,
                    daily_profit=0.0,
                    weekly_profit=0.0,
                    monthly_profit=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    risk_percentage=0.0
                )
        except Exception as e:
            self.detailed_logger.logger.error(f"Erro ao obter estado da conta: {e}")
            return None
    
    def _get_market_conditions(self, symbol: str) -> Optional[MarketConditions]:
        """Obtém condições atuais do mercado para um símbolo"""
        try:
            if not self.mt5_auth or not self.mt5_auth.connected:
                return None
            
            if hasattr(mt5, 'symbol_info_tick'):
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    return None
                
                return MarketConditions(
                    symbol=symbol,
                    bid=tick.bid,
                    ask=tick.ask,
                    spread=tick.ask - tick.bid,
                    volatility=0.0,  # Será calculado posteriormente
                    trend_direction="unknown",  # Será calculado posteriormente
                    volume=tick.volume if hasattr(tick, 'volume') else None
                )
            else:
                # Modo simulação
                return MarketConditions(
                    symbol=symbol,
                    bid=1.1000,
                    ask=1.1002,
                    spread=0.0002,
                    volatility=0.5,
                    trend_direction="sideways"
                )
        except Exception as e:
            self.detailed_logger.logger.error(f"Erro ao obter condições de mercado para {symbol}: {e}")
            return None
    
    def _check_account_changes(self) -> None:
        """Verifica mudanças na conta e registra"""
        current_state = self._get_current_account_state()
        if current_state is None:
            return
        
        # Verifica mudanças no saldo
        if self.last_balance is not None and self.last_equity is not None:
            balance_changed = abs(current_state.balance - self.last_balance) > 0.01
            equity_changed = abs(current_state.equity - self.last_equity) > 0.01
            
            if balance_changed or equity_changed:
                # Determina o motivo da mudança
                reason = self._determine_balance_change_reason(
                    self.last_balance, current_state.balance,
                    self.last_equity, current_state.equity,
                    current_state
                )
                
                self.detailed_logger.log_balance_update(
                    old_balance=self.last_balance,
                    new_balance=current_state.balance,
                    old_equity=self.last_equity,
                    new_equity=current_state.equity,
                    reason=reason,
                    account_state=current_state,
                    additional_info={
                        "positions_count": current_state.open_positions,
                        "pending_orders": current_state.pending_orders,
                        "margin_level": f"{current_state.margin_level:.2f}%"
                    }
                )
        
        self.last_balance = current_state.balance
        self.last_equity = current_state.equity
        self.last_account_state = current_state
    
    def _determine_balance_change_reason(self, old_balance: float, new_balance: float,
                                       old_equity: float, new_equity: float,
                                       account_state: AccountState) -> str:
        """Determina o motivo da mudança no saldo"""
        balance_diff = new_balance - old_balance
        equity_diff = new_equity - old_equity
        
        if abs(balance_diff) < 0.01 and abs(equity_diff) > 0.01:
            return "Flutuação de equity por posições abertas"
        elif balance_diff > 0:
            return "Lucro realizado ou depósito"
        elif balance_diff < 0:
            return "Prejuízo realizado ou saque"
        elif account_state.open_positions > 0:
            return "Atualização de equity com posições abertas"
        else:
            return "Atualização de conta"
    
    def _check_position_changes(self) -> None:
        """Verifica mudanças nas posições"""
        try:
            if not self.mt5_auth or not self.mt5_auth.connected:
                return
            
            if hasattr(mt5, 'positions_get'):
                current_positions = mt5.positions_get()
                if current_positions is None:
                    current_positions = []
                
                # Converte para dicionário para comparação
                current_pos_dict = {pos.ticket: pos for pos in current_positions}
                
                # Verifica novas posições
                for ticket, position in current_pos_dict.items():
                    if ticket not in self.last_positions:
                        self._log_position_opened(position)
                
                # Verifica posições fechadas
                for ticket in self.last_positions:
                    if ticket not in current_pos_dict:
                        self._log_position_closed(ticket, self.last_positions[ticket])
                
                self.last_positions = current_pos_dict
        except Exception as e:
            self.detailed_logger.logger.error(f"Erro ao verificar mudanças de posição: {e}")
    
    def _log_position_opened(self, position) -> None:
        """Registra abertura de posição"""
        try:
            market_conditions = self._get_market_conditions(position.symbol)
            account_state = self._get_current_account_state()
            
            if market_conditions is None or account_state is None:
                return
            
            trade_details = TradeDetails(
                symbol=position.symbol,
                action="BUY" if position.type == 0 else "SELL",
                volume=position.volume,
                price=position.price_open,
                stop_loss=position.sl if position.sl > 0 else None,
                take_profit=position.tp if position.tp > 0 else None,
                ticket=position.ticket,
                profit=position.profit
            )
            
            decision_factors = [
                "Análise técnica favorável",
                "Condições de mercado adequadas",
                "Gestão de risco aplicada",
                "Sinal de entrada confirmado"
            ]
            
            justification = (
                f"Posição {trade_details.action} aberta em {position.symbol} "
                f"com volume {position.volume} ao preço {position.price_open}. "
                f"Stop Loss: {position.sl if position.sl > 0 else 'Não definido'}, "
                f"Take Profit: {position.tp if position.tp > 0 else 'Não definido'}"
            )
            
            outcome = f"Posição #{position.ticket} aberta com sucesso"
            
            self.detailed_logger.log_trade_operation(
                trade_details=trade_details,
                market_conditions=market_conditions,
                account_state=account_state,
                decision_factors=decision_factors,
                justification=justification,
                confidence=0.8,
                outcome=outcome
            )
        except Exception as e:
            self.detailed_logger.logger.error(f"Erro ao registrar abertura de posição: {e}")
    
    def _log_position_closed(self, ticket: int, position) -> None:
        """Registra fechamento de posição"""
        try:
            market_conditions = self._get_market_conditions(position.symbol)
            account_state = self._get_current_account_state()
            
            if market_conditions is None or account_state is None:
                return
            
            trade_details = TradeDetails(
                symbol=position.symbol,
                action="CLOSE",
                volume=position.volume,
                price=position.price_current,
                ticket=ticket,
                profit=position.profit
            )
            
            decision_factors = [
                "Stop Loss atingido" if position.profit < 0 else "Take Profit atingido",
                "Gestão de risco",
                "Fechamento manual" if abs(position.profit) < 1 else "Fechamento automático"
            ]
            
            result_type = "LUCRO" if position.profit > 0 else "PREJUÍZO" if position.profit < 0 else "EMPATE"
            
            justification = (
                f"Posição #{ticket} em {position.symbol} fechada com {result_type.lower()} "
                f"de ${position.profit:.2f}. Volume: {position.volume}, "
                f"Preço de abertura: {position.price_open}, "
                f"Preço de fechamento: {position.price_current}"
            )
            
            outcome = f"Posição #{ticket} fechada - {result_type}: ${position.profit:.2f}"
            
            self.detailed_logger.log_trade_operation(
                trade_details=trade_details,
                market_conditions=market_conditions,
                account_state=account_state,
                decision_factors=decision_factors,
                justification=justification,
                confidence=1.0,
                outcome=outcome
            )
        except Exception as e:
            self.detailed_logger.logger.error(f"Erro ao registrar fechamento de posição: {e}")
    
    def log_manual_trade_decision(self, symbol: str, action: str, volume: float,
                                 analysis_factors: List[str], justification: str,
                                 confidence: float) -> str:
        """Registra decisão manual de trading"""
        market_conditions = self._get_market_conditions(symbol)
        account_state = self._get_current_account_state()
        
        if market_conditions is None or account_state is None:
            return ""
        
        trade_details = TradeDetails(
            symbol=symbol,
            action=action,
            volume=volume,
            price=market_conditions.ask if action == "BUY" else market_conditions.bid
        )
        
        outcome = f"Decisão registrada - {action} {volume} {symbol}"
        
        return self.detailed_logger.log_trade_operation(
            trade_details=trade_details,
            market_conditions=market_conditions,
            account_state=account_state,
            decision_factors=analysis_factors,
            justification=justification,
            confidence=confidence,
            outcome=outcome
        )
    
    def log_no_trade_analysis(self, symbol: str, reasons: List[str],
                             analysis_performed: List[str], confidence: float) -> str:
        """Registra análise que resultou em não operação"""
        market_conditions = self._get_market_conditions(symbol)
        account_state = self._get_current_account_state()
        
        if market_conditions is None or account_state is None:
            return ""
        
        return self.detailed_logger.log_no_trade_decision(
            symbol=symbol,
            market_conditions=market_conditions,
            account_state=account_state,
            reasons=reasons,
            analysis_performed=analysis_performed,
            confidence=confidence
        )
    
    def log_market_analysis_session(self, symbols: List[str],
                                   analysis_results: Dict[str, Any],
                                   observations: List[str],
                                   recommendations: List[str],
                                   confidence: float) -> str:
        """Registra sessão completa de análise de mercado"""
        decision_id = self.detailed_logger.log_market_analysis(
            symbols=symbols,
            analysis_results=analysis_results,
            observations=observations,
            recommendations=recommendations,
            confidence=confidence
        )
        
        # Adiciona ao histórico
        self.analysis_history.append({
            'decision_id': decision_id,
            'timestamp': datetime.now().isoformat(),
            'symbols': symbols,
            'observations_count': len(observations),
            'recommendations_count': len(recommendations),
            'confidence': confidence
        })
        
        # Mantém histórico limitado
        if len(self.analysis_history) > self.max_analysis_history:
            self.analysis_history = self.analysis_history[-self.max_analysis_history:]
        
        return decision_id
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Retorna resumo da sessão atual"""
        stats = self.detailed_logger.get_session_statistics()
        
        return {
            **stats,
            'monitoring_active': self.monitoring_active,
            'analysis_sessions': len(self.analysis_history),
            'last_balance': self.last_balance,
            'last_equity': self.last_equity,
            'open_positions': self.last_account_state.open_positions if self.last_account_state else 0
        }
    
    def export_session_report(self) -> str:
        """Exporta relatório completo da sessão"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = Path(f"logs/session_report_{timestamp}.json")
            
            report_data = {
                'session_summary': self.get_session_summary(),
                'recent_decisions': self.detailed_logger.get_recent_decisions(100),
                'analysis_history': self.analysis_history,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.detailed_logger.logger.info(f"Relatório da sessão exportado: {report_file}")
            return str(report_file)
        except Exception as e:
            self.detailed_logger.logger.error(f"Erro ao exportar relatório: {e}")
            return ""

# Instância global
trading_integrator = None

def get_trading_integrator() -> Optional[TradingDecisionIntegrator]:
    """Retorna instância global do integrador"""
    return trading_integrator

def initialize_trading_integrator(mt5_authenticator=None, session_id: str = None) -> TradingDecisionIntegrator:
    """Inicializa o integrador de trading"""
    global trading_integrator
    trading_integrator = TradingDecisionIntegrator(mt5_authenticator, session_id)
    return trading_integrator