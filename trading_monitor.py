#!/usr/bin/env python3
"""
Sistema de Monitoramento de Trading em Tempo Real
Monitora sinais, posições e performance do sistema de trading
"""

import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import MetaTrader5 as mt5
from dataclasses import dataclass
import threading
import os

@dataclass
class TradingSignal:
    """Estrutura para sinais de trading"""
    timestamp: datetime
    symbol: str
    direction: str  # BUY/SELL
    confidence: float
    price: float
    model: str
    features: Dict

@dataclass
class Position:
    """Estrutura para posições"""
    ticket: int
    symbol: str
    type: str
    volume: float
    open_price: float
    current_price: float
    profit: float
    open_time: datetime

class TradingMonitor:
    """Monitor de trading em tempo real"""
    
    def __init__(self):
        self.running = False
        self.signals_history = []
        self.positions_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'current_balance': 0.0
        }
        
        # Configurar conexão com banco de dados
        self.db_path = 'trading_monitor.db'
        self._init_database()
        
        # Thread de monitoramento
        self.monitor_thread = None
        
    def _init_database(self):
        """Inicializa banco de dados para histórico"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de sinais
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                confidence REAL,
                price REAL,
                model TEXT,
                features TEXT
            )
        ''')
        
        # Tabela de posições
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket INTEGER,
                symbol TEXT,
                type TEXT,
                volume REAL,
                open_price REAL,
                close_price REAL,
                profit REAL,
                open_time TEXT,
                close_time TEXT,
                duration_minutes INTEGER
            )
        ''')
        
        # Tabela de performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                balance REAL,
                equity REAL,
                margin REAL,
                free_margin REAL,
                profit REAL,
                drawdown REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def start_monitoring(self):
        """Inicia monitoramento em tempo real"""
        if self.running:
            print("Monitor já está em execução")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Monitor de trading iniciado")
        
    def stop_monitoring(self):
        """Para o monitoramento"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("⏹️ Monitor de trading parado")
        
    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        while self.running:
            try:
                # Verificar conexão MT5
                if not mt5.initialize():
                    print("⚠️ Erro na conexão MT5")
                    time.sleep(10)
                    continue
                    
                # Coletar dados atuais
                self._collect_account_data()
                self._collect_positions_data()
                self._update_performance_stats()
                
                # Exibir status a cada 30 segundos
                self._display_status()
                
                time.sleep(30)  # Atualizar a cada 30 segundos
                
            except Exception as e:
                print(f"❌ Erro no monitoramento: {e}")
                time.sleep(10)
                
    def _collect_account_data(self):
        """Coleta dados da conta"""
        account_info = mt5.account_info()
        if account_info:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance (timestamp, balance, equity, margin, free_margin, profit, drawdown)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                account_info.balance,
                account_info.equity,
                account_info.margin,
                account_info.margin_free,
                account_info.profit,
                0.0  # Calcular drawdown depois
            ))
            
            conn.commit()
            conn.close()
            
            self.performance_stats['current_balance'] = account_info.balance
            
    def _collect_positions_data(self):
        """Coleta dados das posições abertas"""
        positions = mt5.positions_get()
        if positions:
            current_positions = []
            for pos in positions:
                position = Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    type='BUY' if pos.type == 0 else 'SELL',
                    volume=pos.volume,
                    open_price=pos.price_open,
                    current_price=pos.price_current,
                    profit=pos.profit,
                    open_time=datetime.fromtimestamp(pos.time)
                )
                current_positions.append(position)
                
            self.positions_history = current_positions
            
    def _update_performance_stats(self):
        """Atualiza estatísticas de performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Contar sinais totais
        cursor.execute('SELECT COUNT(*) FROM signals')
        self.performance_stats['total_signals'] = cursor.fetchone()[0]
        
        # Contar trades totais
        cursor.execute('SELECT COUNT(*) FROM positions WHERE close_time IS NOT NULL')
        self.performance_stats['total_trades'] = cursor.fetchone()[0]
        
        # Calcular trades vencedores
        cursor.execute('SELECT COUNT(*) FROM positions WHERE close_time IS NOT NULL AND profit > 0')
        self.performance_stats['winning_trades'] = cursor.fetchone()[0]
        
        # Calcular lucro total
        cursor.execute('SELECT SUM(profit) FROM positions WHERE close_time IS NOT NULL')
        result = cursor.fetchone()[0]
        self.performance_stats['total_profit'] = result if result else 0.0
        
        conn.close()
        
    def _display_status(self):
        """Exibe status atual do sistema"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("="*80)
        print("🚀 MONITOR DE TRADING EM TEMPO REAL")
        print("="*80)
        print(f"⏰ Última atualização: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        # Status da conta
        print("💰 STATUS DA CONTA:")
        print(f"   Saldo: ${self.performance_stats['current_balance']:,.2f}")
        print(f"   Lucro Total: ${self.performance_stats['total_profit']:,.2f}")
        print()
        
        # Estatísticas de trading
        print("📊 ESTATÍSTICAS DE TRADING:")
        print(f"   Sinais Gerados: {self.performance_stats['total_signals']}")
        print(f"   Trades Executados: {self.performance_stats['total_trades']}")
        
        if self.performance_stats['total_trades'] > 0:
            win_rate = (self.performance_stats['winning_trades'] / self.performance_stats['total_trades']) * 100
            print(f"   Taxa de Acerto: {win_rate:.1f}%")
        else:
            print(f"   Taxa de Acerto: N/A")
        print()
        
        # Posições abertas
        print("📈 POSIÇÕES ABERTAS:")
        if self.positions_history:
            for pos in self.positions_history:
                duration = datetime.now() - pos.open_time
                duration_str = str(duration).split('.')[0]  # Remove microsegundos
                
                profit_color = "🟢" if pos.profit >= 0 else "🔴"
                print(f"   {profit_color} {pos.symbol} {pos.type} {pos.volume} @ {pos.open_price:.5f}")
                print(f"      Lucro: ${pos.profit:.2f} | Duração: {duration_str}")
        else:
            print("   Nenhuma posição aberta")
        print()
        
        # Últimos sinais
        print("🎯 ÚLTIMOS SINAIS (últimas 5):")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, symbol, direction, confidence, price, model
            FROM signals 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''')
        
        recent_signals = cursor.fetchall()
        if recent_signals:
            for signal in recent_signals:
                timestamp, symbol, direction, confidence, price, model = signal
                time_str = datetime.fromisoformat(timestamp).strftime('%H:%M:%S')
                direction_icon = "📈" if direction == "BUY" else "📉"
                print(f"   {direction_icon} {time_str} | {symbol} {direction} | Conf: {confidence:.1%} | {model}")
        else:
            print("   Nenhum sinal registrado")
        
        conn.close()
        print()
        print("Pressione Ctrl+C para parar o monitor")
        
    def log_signal(self, signal: TradingSignal):
        """Registra um novo sinal de trading"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals (timestamp, symbol, direction, confidence, price, model, features)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.timestamp.isoformat(),
            signal.symbol,
            signal.direction,
            signal.confidence,
            signal.price,
            signal.model,
            json.dumps(signal.features)
        ))
        
        conn.commit()
        conn.close()
        
        self.signals_history.append(signal)
        print(f"📡 Novo sinal: {signal.symbol} {signal.direction} (Conf: {signal.confidence:.1%})")
        
    def get_daily_summary(self) -> Dict:
        """Retorna resumo do dia"""
        today = datetime.now().date()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sinais do dia
        cursor.execute('''
            SELECT COUNT(*) FROM signals 
            WHERE date(timestamp) = ?
        ''', (today.isoformat(),))
        daily_signals = cursor.fetchone()[0]
        
        # Trades do dia
        cursor.execute('''
            SELECT COUNT(*), SUM(profit) FROM positions 
            WHERE date(close_time) = ? AND close_time IS NOT NULL
        ''', (today.isoformat(),))
        result = cursor.fetchone()
        daily_trades = result[0] if result[0] else 0
        daily_profit = result[1] if result[1] else 0.0
        
        conn.close()
        
        return {
            'date': today.isoformat(),
            'signals': daily_signals,
            'trades': daily_trades,
            'profit': daily_profit
        }

def main():
    """Função principal"""
    monitor = TradingMonitor()
    
    try:
        monitor.start_monitoring()
        
        # Manter o programa rodando
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Parando monitor...")
        monitor.stop_monitoring()
        
if __name__ == "__main__":
    main()