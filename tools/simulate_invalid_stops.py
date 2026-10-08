import os
import sys
from datetime import datetime

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import auto_trader as at

# Monkeypatch MetaTrader5 module used inside auto_trader
class DummySymbolInfo:
    def __init__(self, point=0.00001, digits=5, stops_level=100):
        self.point = point
        self.digits = digits
        # Some brokers expose trade_stops_level, others stops_level
        self.trade_stops_level = stops_level
        self.stops_level = stops_level

class DummyTick:
    def __init__(self, bid, ask):
        self.bid = bid
        self.ask = ask
        self.volume = 100
        self.time = int(datetime.now().timestamp())

class DummyResult:
    def __init__(self, retcode=0, order=0, comment="OK"):
        self.retcode = retcode
        self.order = order
        self.comment = comment

class DummyMT5:
    # Constants used in code
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 0
    TRADE_RETCODE_DONE = 10009

    def __init__(self, symbol_info, tick, retcode_after=10009):
        self._symbol_info = symbol_info
        self._tick = tick
        self._retcode_after = retcode_after

    def symbol_info(self, symbol):
        return self._symbol_info

    def symbol_info_tick(self, symbol):
        return self._tick

    def order_send(self, request):
        # This should not be called in INVALID_STOPS_PREVENTED path,
        # but keep here for completeness.
        return DummyResult(retcode=self._retcode_after, order=123456, comment="SIMULATION")

# Prepare dummy environment with tight SL/TP below stops_level
symbol = "EURUSD"
entry = 1.23456
# point=1e-5, stops_level=100 points => SL must be >= 0.00100 away; we set only 0.00006
sl = entry - 0.00006  # 6 points
tp = entry + 0.00006  # 6 points

mock_symbol_info = DummySymbolInfo(point=1e-5, digits=5, stops_level=100)
mock_tick = DummyTick(bid=entry-0.00002, ask=entry+0.00002)

# Inject dummy mt5 into auto_trader module
at.mt5 = DummyMT5(symbol_info=mock_symbol_info, tick=mock_tick)

# Instantiate AutoTrader (will not connect to real MT5)
trader = at.AutoTrader()
# Ensure advanced risk won't override behavior in this test
trader.use_advanced_risk = False

# Build a TradeSignal triggering INVALID_STOPS_PREVENTED
signal = at.TradeSignal(
    symbol=symbol,
    direction="BUY",
    confidence=0.77,
    entry_price=entry,
    stop_loss=sl,
    take_profit=tp,
    volume=0.01,
    timestamp=datetime.now(),
    model_source="unit_test"
)

# Call _execute_trade which should log NO_ACTION with INVALID_STOPS_PREVENTED and return None
result = trader._execute_trade(signal)
print("_execute_trade result:", result)
print("Simulation complete. Check detailed_decisions.db for NO_ACTION INVALID_STOPS_PREVENTED entry.")