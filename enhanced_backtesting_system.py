#!/usr/bin/env python3
"""
Sistema de Backtesting Robusto com Correções Automáticas
Corrige problemas de:
1. Total de Trades = 0
2. Inconsistências nas colunas OHLC
3. Sinais de trading inválidos
4. Problemas de alinhamento de dados

Autor: Sistema ML Trading
Data: 2025-01-18
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Métricas e visualização
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Logging
import logging

class EnhancedBacktestingSystem:
    """
    Sistema de backtesting robusto com correções automáticas
    """
    
    def __init__(self, initial_capital: float = 10000.0, 
                 commission: float = 0.0001, 
                 slippage: float = 0.0001,
                 max_position_size: float = 0.02,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.04):
        
        # Parâmetros de trading
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Configurações de correção (deve ser inicializado antes de reset_state)
        self.correction_stats = {
            'ohlc_corrections': 0,
            'signal_corrections': 0,
            'data_alignment_fixes': 0,
            'invalid_trades_filtered': 0,
            'total_signals_processed': 0,
            'valid_signals_generated': 0
        }
        
        # Estado do backtesting
        self.reset_state()
        
        # Setup logging
        self._setup_logging()
        
        # Mapeamentos de colunas OHLC flexíveis
        self.ohlc_mappings = {
            'open': ['open', 'Open', 'OPEN', 'o', 'O'],
            'high': ['high', 'High', 'HIGH', 'h', 'H'],
            'low': ['low', 'Low', 'LOW', 'l', 'L'],
            'close': ['close', 'Close', 'CLOSE', 'c', 'C'],
            'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'Vol', 'VOL', 'v', 'V']
        }
    
    def _setup_logging(self):
        """Configura sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _calculate_duration(self, end_time, start_time):
        """Calcula duração entre timestamps de forma segura"""
        if not start_time:
            return 0
        
        try:
            # Converter para pandas Timestamp se necessário
            if isinstance(end_time, (int, np.int64, np.int32)):
                end_time = pd.Timestamp(end_time)
            if isinstance(start_time, (int, np.int64, np.int32)):
                start_time = pd.Timestamp(start_time)
            
            # Calcular diferença
            duration = end_time - start_time
            
            # Converter para horas
            if hasattr(duration, 'total_seconds'):
                return duration.total_seconds() / 3600
            else:
                # Se for um Timedelta do pandas
                return duration.total_seconds() / 3600
        except Exception as e:
            self.logger.warning(f"⚠️ Erro ao calcular duração: {e}")
            return 0
    
    def reset_state(self):
        """Reseta estado do backtesting"""
        self.capital = self.initial_capital
        self.position = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        
        # Histórico de trades
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        
        # Métricas de performance
        self.max_equity = self.initial_capital
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_capital
        
        # Estatísticas de correções
        for key in self.correction_stats:
            self.correction_stats[key] = 0
    
    def _find_ohlc_columns(self, data: pd.DataFrame) -> Dict[str, str]:
        """Encontra colunas OHLC com mapeamento flexível"""
        found_columns = {}
        
        for ohlc_type, possible_names in self.ohlc_mappings.items():
            for name in possible_names:
                if name in data.columns:
                    found_columns[ohlc_type] = name
                    break
            
            if ohlc_type not in found_columns:
                self.logger.warning(f"⚠️ Coluna {ohlc_type} não encontrada")
        
        return found_columns
    
    def _validate_and_correct_ohlc_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Valida e corrige dados OHLC"""
        data_corrected = data.copy()
        
        # Encontrar colunas OHLC
        ohlc_cols = self._find_ohlc_columns(data_corrected)
        
        if len(ohlc_cols) < 4:
            self.logger.error(f"❌ Colunas OHLC insuficientes encontradas: {list(ohlc_cols.keys())}")
            return data_corrected
        
        # Padronizar nomes das colunas
        column_mapping = {}
        for ohlc_type, col_name in ohlc_cols.items():
            if col_name != ohlc_type:
                column_mapping[col_name] = ohlc_type
        
        if column_mapping:
            data_corrected = data_corrected.rename(columns=column_mapping)
            self.correction_stats['ohlc_corrections'] += len(column_mapping)
            self.logger.info(f"🔧 Colunas OHLC padronizadas: {column_mapping}")
        
        # Validar relações OHLC
        required_cols = ['open', 'high', 'low', 'close']
        if all(col in data_corrected.columns for col in required_cols):
            
            # Corrigir violações de OHLC
            initial_violations = 0
            
            # High deve ser >= max(open, close)
            violation_mask = data_corrected['high'] < np.maximum(data_corrected['open'], data_corrected['close'])
            if violation_mask.any():
                initial_violations += violation_mask.sum()
                data_corrected.loc[violation_mask, 'high'] = np.maximum(
                    data_corrected.loc[violation_mask, 'open'],
                    data_corrected.loc[violation_mask, 'close']
                )
            
            # Low deve ser <= min(open, close)
            violation_mask = data_corrected['low'] > np.minimum(data_corrected['open'], data_corrected['close'])
            if violation_mask.any():
                initial_violations += violation_mask.sum()
                data_corrected.loc[violation_mask, 'low'] = np.minimum(
                    data_corrected.loc[violation_mask, 'open'],
                    data_corrected.loc[violation_mask, 'close']
                )
            
            if initial_violations > 0:
                self.correction_stats['ohlc_corrections'] += initial_violations
                self.logger.info(f"🔧 Corrigidas {initial_violations} violações OHLC")
        
        # Remover valores inválidos
        numeric_cols = data_corrected.select_dtypes(include=[np.number]).columns
        invalid_mask = data_corrected[numeric_cols].isin([np.inf, -np.inf]).any(axis=1)
        
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            data_corrected = data_corrected[~invalid_mask]
            self.correction_stats['data_alignment_fixes'] += invalid_count
            self.logger.info(f"🔧 Removidas {invalid_count} linhas com valores inválidos")
        
        return data_corrected
    
    def _validate_and_correct_signals(self, signals: pd.Series) -> pd.Series:
        """Valida e corrige sinais de trading"""
        signals_corrected = signals.copy()
        
        self.correction_stats['total_signals_processed'] = len(signals_corrected)
        
        # Remover valores NaN
        nan_mask = signals_corrected.isna()
        if nan_mask.any():
            signals_corrected = signals_corrected.fillna(0)
            self.correction_stats['signal_corrections'] += nan_mask.sum()
        
        # Garantir que sinais estão no range válido (-1, 0, 1)
        # Converter sinais contínuos para discretos
        original_signals = signals_corrected.copy()
        
        # Aplicar thresholds para sinais contínuos
        buy_threshold = 0.6
        sell_threshold = -0.6
        
        signals_corrected = np.where(
            signals_corrected > buy_threshold, 1,
            np.where(signals_corrected < sell_threshold, -1, 0)
        )
        
        # Contar correções aplicadas
        corrections = (original_signals != signals_corrected).sum()
        if corrections > 0:
            self.correction_stats['signal_corrections'] += corrections
            self.logger.info(f"🔧 Aplicados thresholds a {corrections} sinais")
        
        # Filtrar sinais consecutivos iguais (manter apenas mudanças)
        signals_series = pd.Series(signals_corrected, index=signals.index)
        signals_diff = signals_series.diff().fillna(signals_series.iloc[0])
        
        # Manter apenas sinais que representam mudanças
        valid_signals = signals_series.copy()
        valid_signals[signals_diff == 0] = 0
        
        self.correction_stats['valid_signals_generated'] = (valid_signals != 0).sum()
        
        self.logger.info(f"📊 Sinais processados: {len(signals)} -> {self.correction_stats['valid_signals_generated']} válidos")
        
        return valid_signals
    
    def _calculate_position_size(self, price: float, signal_strength: float = 1.0) -> float:
        """Calcula tamanho da posição baseado no capital disponível"""
        if self.capital <= 0:
            return 0.0
        
        # Tamanho base da posição
        base_size = (self.capital * self.max_position_size) / price
        
        # Ajustar pela força do sinal
        adjusted_size = base_size * abs(signal_strength)
        
        return adjusted_size
    
    def _execute_trade(self, timestamp: pd.Timestamp, price: float, signal: float, 
                      high: float, low: float) -> bool:
        """Executa trade com validação robusta"""
        
        # Validar preços
        if not all(np.isfinite([price, high, low])):
            self.correction_stats['invalid_trades_filtered'] += 1
            return False
        
        # Validar relações de preços
        if not (low <= price <= high):
            self.correction_stats['invalid_trades_filtered'] += 1
            return False
        
        trade_executed = False
        
        # Fechar posição existente se sinal oposto
        if self.position != 0 and np.sign(self.position) != np.sign(signal):
            self._close_position(timestamp, price)
            trade_executed = True
        
        # Abrir nova posição
        if signal != 0 and self.position == 0:
            position_size = self._calculate_position_size(price, abs(signal))
            
            if position_size > 0:
                # Aplicar slippage
                execution_price = price * (1 + self.slippage * np.sign(signal))
                
                # Calcular custos
                trade_value = position_size * execution_price
                commission_cost = trade_value * self.commission
                
                if self.capital >= commission_cost:
                    # Executar trade
                    self.position = position_size * np.sign(signal)
                    self.position_value = trade_value
                    self.entry_price = execution_price
                    self.entry_time = timestamp
                    self.capital -= commission_cost
                    
                    # Definir stop loss e take profit
                    if signal > 0:  # Long
                        self.stop_loss_price = execution_price * (1 - self.stop_loss_pct)
                        self.take_profit_price = execution_price * (1 + self.take_profit_pct)
                    else:  # Short
                        self.stop_loss_price = execution_price * (1 + self.stop_loss_pct)
                        self.take_profit_price = execution_price * (1 - self.take_profit_pct)
                    
                    trade_executed = True
                    
                    self.logger.debug(f"📈 Trade aberto: {signal:+.0f} @ {execution_price:.5f}")
        
        return trade_executed
    
    def _check_exit_conditions(self, timestamp: pd.Timestamp, high: float, low: float, close: float) -> bool:
        """Verifica condições de saída (stop loss / take profit)"""
        if self.position == 0:
            return False
        
        exit_price = None
        exit_reason = None
        
        if self.position > 0:  # Long position
            # Check stop loss
            if low <= self.stop_loss_price:
                exit_price = self.stop_loss_price
                exit_reason = "Stop Loss"
            # Check take profit
            elif high >= self.take_profit_price:
                exit_price = self.take_profit_price
                exit_reason = "Take Profit"
        
        else:  # Short position
            # Check stop loss
            if high >= self.stop_loss_price:
                exit_price = self.stop_loss_price
                exit_reason = "Stop Loss"
            # Check take profit
            elif low <= self.take_profit_price:
                exit_price = self.take_profit_price
                exit_reason = "Take Profit"
        
        if exit_price is not None:
            self._close_position(timestamp, exit_price, exit_reason)
            return True
        
        return False
    
    def _close_position(self, timestamp: pd.Timestamp, price: float, reason: str = "Signal"):
        """Fecha posição atual"""
        if self.position == 0:
            return
        
        # Aplicar slippage
        execution_price = price * (1 - self.slippage * np.sign(self.position))
        
        # Calcular P&L
        if self.position > 0:  # Long
            pnl = self.position * (execution_price - self.entry_price)
        else:  # Short
            pnl = abs(self.position) * (self.entry_price - execution_price)
        
        # Aplicar comissão
        trade_value = abs(self.position) * execution_price
        commission_cost = trade_value * self.commission
        net_pnl = pnl - commission_cost
        
        # Atualizar capital
        self.capital += net_pnl
        
        # Registrar trade
        trade_record = {
            'entry_time': self.entry_time,
            'exit_time': timestamp,
            'entry_price': self.entry_price,
            'exit_price': execution_price,
            'position_size': self.position,
            'pnl': net_pnl,
            'commission': commission_cost,
            'exit_reason': reason,
            'duration': self._calculate_duration(timestamp, self.entry_time)
        }
        
        self.trades.append(trade_record)
        
        # Reset position
        self.position = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        
        self.logger.debug(f"📉 Trade fechado: {reason} @ {execution_price:.5f}, P&L: {net_pnl:.2f}")
    
    def _update_equity_curve(self, timestamp: pd.Timestamp, current_price: float):
        """Atualiza curva de equity"""
        # Calcular valor atual da posição
        if self.position != 0:
            if self.position > 0:  # Long
                unrealized_pnl = self.position * (current_price - self.entry_price)
            else:  # Short
                unrealized_pnl = abs(self.position) * (self.entry_price - current_price)
            
            current_equity = self.capital + unrealized_pnl
        else:
            current_equity = self.capital
        
        # Atualizar máximo
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calcular drawdown
        drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # Registrar
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': current_equity,
            'capital': self.capital,
            'unrealized_pnl': unrealized_pnl if self.position != 0 else 0,
            'drawdown': drawdown
        })
    
    def run_backtest(self, data: pd.DataFrame, signals: pd.Series, 
                    pair_name: str = "Unknown") -> Dict[str, Any]:
        """Executa backtesting robusto com correções automáticas"""
        
        self.logger.info(f"🚀 Iniciando backtesting para {pair_name}...")
        
        # Reset state
        self.reset_state()
        
        # Validar e corrigir dados
        data_corrected = self._validate_and_correct_ohlc_data(data)
        signals_corrected = self._validate_and_correct_signals(signals)
        
        # Alinhar dados e sinais
        common_index = data_corrected.index.intersection(signals_corrected.index)
        
        if len(common_index) == 0:
            self.logger.error("❌ Nenhum índice comum entre dados e sinais")
            return self._generate_empty_results(pair_name)
        
        data_aligned = data_corrected.loc[common_index]
        signals_aligned = signals_corrected.loc[common_index]
        
        self.correction_stats['data_alignment_fixes'] += len(data_corrected) - len(data_aligned)
        
        self.logger.info(f"📊 Dados alinhados: {len(data_aligned)} períodos")
        
        # Encontrar colunas OHLC
        ohlc_cols = self._find_ohlc_columns(data_aligned)
        
        if len(ohlc_cols) < 4:
            self.logger.error(f"❌ Colunas OHLC insuficientes: {list(ohlc_cols.keys())}")
            return self._generate_empty_results(pair_name)
        
        # Executar backtesting
        total_signals = 0
        executed_trades = 0
        
        for timestamp in data_aligned.index:
            try:
                # Obter dados do período
                row = data_aligned.loc[timestamp]
                signal = signals_aligned.loc[timestamp]
                
                open_price = row[ohlc_cols['open']]
                high_price = row[ohlc_cols['high']]
                low_price = row[ohlc_cols['low']]
                close_price = row[ohlc_cols['close']]
                
                # Verificar condições de saída primeiro
                self._check_exit_conditions(timestamp, high_price, low_price, close_price)
                
                # Processar sinal de entrada
                if signal != 0:
                    total_signals += 1
                    if self._execute_trade(timestamp, open_price, signal, high_price, low_price):
                        executed_trades += 1
                
                # Atualizar equity curve
                self._update_equity_curve(timestamp, close_price)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Erro no período {timestamp}: {e}")
                continue
        
        # Fechar posição final se existir
        if self.position != 0 and len(data_aligned) > 0:
            final_price = data_aligned.iloc[-1][ohlc_cols['close']]
            self._close_position(data_aligned.index[-1], final_price, "End of Data")
        
        # Gerar resultados
        results = self._calculate_performance_metrics(pair_name, total_signals, executed_trades)
        
        self.logger.info(f"✅ Backtesting concluído: {len(self.trades)} trades executados")
        
        return results
    
    def _generate_empty_results(self, pair_name: str) -> Dict[str, Any]:
        """Gera resultados vazios em caso de erro"""
        return {
            'pair': pair_name,
            'total_trades': 0,
            'profitable_trades': 0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_trade_duration': 0.0,
            'correction_stats': self.correction_stats.copy(),
            'trades': [],
            'equity_curve': [],
            'error': 'Dados insuficientes ou inválidos'
        }
    
    def _calculate_performance_metrics(self, pair_name: str, total_signals: int, 
                                     executed_trades: int) -> Dict[str, Any]:
        """Calcula métricas de performance"""
        
        if not self.trades:
            self.logger.warning(f"⚠️ Nenhum trade executado para {pair_name}")
            return self._generate_empty_results(pair_name)
        
        # Métricas básicas
        total_trades = len(self.trades)
        profitable_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
        total_pnl = sum(trade['pnl'] for trade in self.trades)
        
        # Retorno total
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        # Win rate
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Duração média dos trades
        avg_duration = np.mean([trade['duration'] for trade in self.trades]) if self.trades else 0
        
        # Sharpe ratio (simplificado)
        if self.equity_curve:
            returns = []
            for i in range(1, len(self.equity_curve)):
                ret = (self.equity_curve[i]['equity'] - self.equity_curve[i-1]['equity']) / self.equity_curve[i-1]['equity']
                returns.append(ret)
            
            if returns and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Anualizado
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Profit factor
        winning_trades_pnl = sum(trade['pnl'] for trade in self.trades if trade['pnl'] > 0)
        losing_trades_pnl = abs(sum(trade['pnl'] for trade in self.trades if trade['pnl'] < 0))
        profit_factor = winning_trades_pnl / losing_trades_pnl if losing_trades_pnl > 0 else float('inf')
        
        results = {
            'pair': pair_name,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'avg_trade_duration': avg_duration,
            'final_capital': self.capital,
            'signal_efficiency': (executed_trades / total_signals * 100) if total_signals > 0 else 0,
            'correction_stats': self.correction_stats.copy(),
            'trades': self.trades.copy(),
            'equity_curve': self.equity_curve.copy()
        }
        
        return results
    
    def generate_backtest_report(self, results: Dict[str, Any], 
                               save_path: Optional[str] = None) -> str:
        """Gera relatório detalhado do backtesting"""
        
        report = f"""
# Relatório de Backtesting - {results['pair']}

## 📊 Resumo de Performance

- **Total de Trades:** {results['total_trades']}
- **Trades Lucrativos:** {results['profitable_trades']}
- **Taxa de Acerto:** {results['win_rate']:.1f}%
- **Retorno Total:** {results['total_return']:.2f}%
- **P&L Total:** ${results.get('total_pnl', 0):.2f}
- **Máximo Drawdown:** {results['max_drawdown']:.2f}%
- **Sharpe Ratio:** {results['sharpe_ratio']:.3f}
- **Profit Factor:** {results.get('profit_factor', 0):.2f}
- **Duração Média dos Trades:** {results['avg_trade_duration']:.1f} horas
- **Eficiência de Sinais:** {results.get('signal_efficiency', 0):.1f}%

## 🔧 Estatísticas de Correções

- **Correções OHLC:** {results['correction_stats']['ohlc_corrections']}
- **Correções de Sinais:** {results['correction_stats']['signal_corrections']}
- **Correções de Alinhamento:** {results['correction_stats']['data_alignment_fixes']}
- **Trades Inválidos Filtrados:** {results['correction_stats']['invalid_trades_filtered']}
- **Sinais Processados:** {results['correction_stats']['total_signals_processed']}
- **Sinais Válidos Gerados:** {results['correction_stats']['valid_signals_generated']}

## 📈 Análise de Trades

"""
        
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            
            # Estatísticas de P&L
            report += f"""
### Distribuição de P&L
- **Melhor Trade:** ${trades_df['pnl'].max():.2f}
- **Pior Trade:** ${trades_df['pnl'].min():.2f}
- **P&L Médio:** ${trades_df['pnl'].mean():.2f}
- **Desvio Padrão P&L:** ${trades_df['pnl'].std():.2f}

### Análise Temporal
- **Duração Mínima:** {trades_df['duration'].min():.1f} horas
- **Duração Máxima:** {trades_df['duration'].max():.1f} horas
- **Duração Mediana:** {trades_df['duration'].median():.1f} horas

"""
        
        # Recomendações
        report += "## 🎯 Recomendações\n\n"
        
        if results['total_trades'] == 0:
            report += "- ❌ **Nenhum Trade Executado:** Verificar sinais de trading e dados de entrada.\n"
        elif results['win_rate'] < 40:
            report += "- ⚠️ **Taxa de Acerto Baixa:** Revisar estratégia de trading.\n"
        elif results['max_drawdown'] > 20:
            report += "- ⚠️ **Drawdown Alto:** Implementar melhor gestão de risco.\n"
        else:
            report += "- ✅ **Performance Satisfatória:** Estratégia mostra potencial.\n"
        
        if results['correction_stats']['ohlc_corrections'] > 0:
            report += "- 🔧 **Dados Corrigidos:** Verificar qualidade dos dados de entrada.\n"
        
        if results.get('signal_efficiency', 0) < 50:
            report += "- ⚠️ **Baixa Eficiência de Sinais:** Muitos sinais não resultaram em trades.\n"
        
        report += "\n---\n*Relatório gerado pelo Sistema de Backtesting Robusto*\n"
        
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info(f"📋 Relatório salvo em: {save_path}")
            except Exception as e:
                self.logger.error(f"❌ Erro ao salvar relatório: {e}")
        
        return report

def main():
    """Função principal para teste"""
    print("🚀 Sistema de Backtesting Robusto")
    
    # Exemplo de uso
    backtester = EnhancedBacktestingSystem(
        initial_capital=10000,
        commission=0.0001,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )
    
    print("✅ Sistema inicializado com correções automáticas")
    print("📊 Pronto para executar backtesting robusto")

if __name__ == "__main__":
    main()