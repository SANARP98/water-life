#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Scalp — OpenAlgo Live Trading Bot (IST)
------------------------------------------------
Converts your backtest strategy `random_scalp.py` into a tradable OpenAlgo bot.

Behavior (long-only, same as backtest):
- On every Nth bar close, queue a LONG entry for the next bar open.
- After entry, place a fixed rupee Target (LIMIT) and Stop (SL/SL-M).
- Intraday square-off at configured time (default 15:15 IST).
- Prints quotes/LTP immediately when fetched.

Notes:
- Uses APScheduler (IST) and OpenAlgo API v1.
- Uses start_date/end_date for any history fetch (disabled by default here).
- No DB writes; no indicators required.
- Lot-size resolved via API search, with latest index fallbacks.

Order Constants: Exchange/Product/PriceType follow OpenAlgo.
"""
from __future__ import annotations
import os, sys, time, json, logging, signal
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
from datetime import datetime, time as dtime, timedelta

import pytz
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

# ----------------------------
# Bootstrap
# ----------------------------
load_dotenv()
IST = pytz.timezone("Asia/Kolkata")
STRATEGY_NAME = "random_scalp_live"

# ==================== Strategy Metadata ====================
STRATEGY_METADATA = {
    "name": "Random Scalp (Live)",
    "description": "Long-only random scalp bot that queues a buy on every Nth bar and manages fixed rupee TP/SL exits.",
    "version": "1.0",
    "features": [
        "Fixed-interval long entries",
        "Static rupee profit target and stop loss",
        "Intraday square-off with APScheduler",
        "OpenAlgo order execution with quote validation"
    ],
    "has_trade_direction": False,
    "author": "Random Scalp Port"
}

CONFIG_FIELD_ORDER = [
    "api_key",
    "api_host",
    "symbol",
    "exchange",
    "product",
    "lots",
    "interval",
    "trade_every_n_bars",
    "profit_target_rupees",
    "stop_loss_rupees",
    "brokerage_per_trade",
    "slippage_rupees",
    "ignore_entry_delta",
    "enable_eod_square_off",
    "square_off_time",
    "test_mode",
    "log_to_file",
    "persist_state",
    "use_history",
    "history_start_date",
    "history_end_date",
]

CONFIG_FIELD_DEFS = {
    "trade_every_n_bars": {"group": "Strategy Behaviour", "label": "Trade Every N Bars"},
    "profit_target_rupees": {"group": "Strategy Behaviour", "label": "Profit Target (₹)", "number_format": "float", "step": 0.5},
    "stop_loss_rupees": {"group": "Strategy Behaviour", "label": "Stop Loss (₹)", "number_format": "float", "step": 0.5},
    "brokerage_per_trade": {"group": "Costs", "label": "Brokerage per Trade (₹)", "number_format": "float", "step": 0.5},
    "slippage_rupees": {"group": "Costs", "label": "Slippage (₹)", "number_format": "float", "step": 0.5},
    "ignore_entry_delta": {"group": "Strategy Behaviour", "label": "Ignore Entry Timing Delta"},
    "test_mode": {"group": "Execution", "label": "Test Mode"},
    "log_to_file": {"group": "Execution", "label": "Log to File"},
    "persist_state": {"group": "Execution", "label": "Persist State"},
    "use_history": {"group": "Backtest & History", "label": "Use Historical Data"},
    "history_start_date": {"group": "Backtest & History", "label": "History Start Date", "placeholder": "YYYY-MM-DD"},
    "history_end_date": {"group": "Backtest & History", "label": "History End Date", "placeholder": "YYYY-MM-DD"},
}

try:
    import openalgo
except Exception:
    print("[FATAL] openalgo library is required. pip install openalgo")
    raise

# ----------------------------
# Latest (May 2025) index lot sizes (fallbacks)
# ----------------------------
INDEX_LOT_SIZES = {
    # NSE Index (NSE_INDEX)
    "NIFTY": 75,
    "NIFTYNXT50": 25,
    "FINNIFTY": 65,
    "BANKNIFTY": 35,
    "MIDCPNIFTY": 140,
    # BSE Index (BSE_INDEX)
    "SENSEX": 20,
    "BANKEX": 30,
    "SENSEX50": 60,
}

# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    # API
    api_key: str = os.getenv("OPENALGO_API_KEY", "")
    api_host: str = os.getenv("OPENALGO_API_HOST", "http://127.0.0.1:5000")
    ws_url: Optional[str] = os.getenv("OPENALGO_WS_URL", "ws://127.0.0.1:8765")

    # Instrument
    symbol: str = os.getenv("SYMBOL", "NIFTY24OCT2524500CE")  # Use an actual tradable symbol
    exchange: str = os.getenv("EXCHANGE", "NFO")               # e.g., NSE, NFO, BSE, MCX
    product: str = os.getenv("PRODUCT", "MIS")                  # MIS / CNC / NRML
    lots: int = int(os.getenv("LOTS", 1))                       # multiplier over lotsize

    # Engine / session
    interval: str = os.getenv("INTERVAL", "5m")                 # 1m/3m/5m/15m/60m/D
    session_windows: Tuple[Tuple[int, int, int, int], ...] = (
        (9, 20, 15, 15),
    )

    # Random scalp params (map 1:1 from backtest)
    trade_every_n_bars: int = int(os.getenv("TRADE_EVERY_N_BARS", 1))
    profit_target_rupees: float = float(os.getenv("PROFIT_TARGET_RUPEES", 2.0))
    stop_loss_rupees: float = float(os.getenv("STOP_LOSS_RUPEES", 1.0))
    brokerage_per_trade: float = float(os.getenv("BROKERAGE_PER_TRADE_RUPEES", 0.0))
    slippage_rupees: float = float(os.getenv("SLIPPAGE_RUPEES", 0.0))

    # Risk & timing
    ignore_entry_delta: bool = os.getenv("IGNORE_ENTRY_DELTA", "true").lower() == "true"
    enable_eod_square_off: bool = os.getenv("ENABLE_EOD_SQUARE_OFF", "true").lower() == "true"
    square_off_time: Tuple[int, int] = (15, 15)

    # Operational
    test_mode: bool = os.getenv("TEST_MODE", "false").lower() == "true"  # use openalgo.simulator
    log_to_file: bool = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    persist_state: bool = os.getenv("PERSIST_STATE", "true").lower() == "true"

    # History warmup (optional; off by default here)
    use_history: bool = os.getenv("USE_HISTORY", "false").lower() == "true"
    history_start_date: Optional[str] = os.getenv("HISTORY_START_DATE")     # YYYY-MM-DD
    history_end_date: Optional[str] = os.getenv("HISTORY_END_DATE")         # YYYY-MM-DD

# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger(STRATEGY_NAME)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _console = logging.StreamHandler(sys.stdout)
    _console.setLevel(logging.INFO)
    _console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_console)

def _ensure_file_logging(cfg: Config) -> None:
    """Attach file logging once per process when enabled."""
    if not cfg.log_to_file:
        return
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return
    _file = logging.FileHandler(f"{STRATEGY_NAME}.log")
    _file.setLevel(logging.INFO)
    _file.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(_file)

log = logger.info

# ----------------------------
# Utilities
# ----------------------------

def now_ist() -> datetime:
    return datetime.now(IST)

def in_session(t: dtime, windows: Tuple[Tuple[int, int, int, int], ...]) -> bool:
    for (h1, m1, h2, m2) in windows:
        if (t >= dtime(h1, m1)) and (t <= dtime(h2, m2)):
            return True
    return False

def is_square_off_time(cfg: Config, t: dtime) -> bool:
    sh, sm = cfg.square_off_time
    return (t.hour, t.minute) >= (sh, sm)

# Symbol helpers
def resolve_quantity(client, cfg: Config, symbol: Optional[str] = None) -> int:
    sym = (symbol or cfg.symbol).upper()
    try:
        result = client.search(query=sym, exchange=cfg.exchange)
        if result.get('status') == 'success':
            for inst in result.get('data', []):
                if inst.get('symbol', '').upper() == sym:
                    lot = int(inst.get('lotsize') or 1)
                    log(f"[{STRATEGY_NAME}] Resolved lot size for {sym} via API: {lot}")
                    return cfg.lots * lot
    except Exception as e:
        log(f"[{STRATEGY_NAME}] [WARN] resolve_quantity API failed: {e}")

    for idx in INDEX_LOT_SIZES.keys():
        if sym.startswith(idx):
            lot = INDEX_LOT_SIZES[idx]
            log(f"[{STRATEGY_NAME}] Using fallback lot size for {sym} (detected {idx}): {lot}")
            return cfg.lots * lot

    log(f"[{STRATEGY_NAME}] [WARN] Could not determine lot size for {sym}, using default 1")
    return cfg.lots * 1

def validate_symbol(client, symbol: str, exchange: str) -> bool:
    try:
        log(f"[{STRATEGY_NAME}] [VALIDATION] Validating {symbol} on {exchange}…")
        # Try quotes first
        try:
            q = client.quotes(symbol=symbol, exchange=exchange)
            print(q)
            if q.get('status') == 'success':
                log(f"[{STRATEGY_NAME}] [VALIDATION] ✅ quotes OK")
                return True
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [VALIDATION] quotes failed: {e}")
        # Fallback: search
        try:
            res = client.search(query=symbol, exchange=exchange)
            if res.get('status') == 'success':
                syms = [s.get('symbol', '').upper() for s in res.get('data', [])]
                return symbol.upper() in syms
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [VALIDATION] search failed: {e}")
        log(f"[{STRATEGY_NAME}] [VALIDATION] Assuming valid (fail-open)")
        return True
    except Exception as e:
        log(f"[{STRATEGY_NAME}] [VALIDATION] Exception: {e}")
        return True

# Optional history (disabled by default)
def get_history(client, cfg: Config, symbol: Optional[str] = None) -> pd.DataFrame:
    sym = symbol or cfg.symbol
    start_date = cfg.history_start_date or (now_ist().date() - timedelta(days=2)).strftime('%Y-%m-%d')
    end_date = cfg.history_end_date or (now_ist().date() - timedelta(days=1)).strftime('%Y-%m-%d')
    log(f"[{STRATEGY_NAME}] [HISTORY] Fetching {sym}@{cfg.exchange} {cfg.interval} {start_date}→{end_date}")
    df = client.history(symbol=sym, exchange=cfg.exchange, interval=cfg.interval,
                        start_date=start_date, end_date=end_date)
    if isinstance(df, pd.DataFrame):
        if 'timestamp' not in df.columns:
            df = df.reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('Asia/Kolkata')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata')
        log(f"[{STRATEGY_NAME}] [HISTORY] Bars: {len(df)}")
        return df.sort_values('timestamp').reset_index(drop=True)
    else:
        log(f"[{STRATEGY_NAME}] [HISTORY] Non-DataFrame response: {df}")
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])    

# ----------------------------
# Trading Engine
# ----------------------------
class ScalpWithTrendBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = openalgo.simulator(api_key=cfg.api_key) if cfg.test_mode else openalgo.api(api_key=cfg.api_key, host=cfg.api_host, ws_url=cfg.ws_url)
        self.scheduler = BackgroundScheduler(timezone=IST)
        _ensure_file_logging(cfg)

        # State
        self.qty: int = resolve_quantity(self.client, cfg, symbol=cfg.symbol)
        self.in_position: bool = False
        self.side: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.tp_level: Optional[float] = None
        self.sl_level: Optional[float] = None
        self.entry_order_id: Optional[str] = None
        self.tp_order_id: Optional[str] = None
        self.sl_order_id: Optional[str] = None

        self.pending_signal: bool = False
        self.next_entry_time: Optional[datetime] = None
        self.bar_counter: int = 0

        self.realized_pnl_today: float = 0.0

    # ---- Time helpers
    def _parse_interval_minutes(self, interval: str) -> int:
        s = interval.strip().lower()
        if s.endswith('m'):
            return int(s.replace('m',''))
        if s.endswith('h'):
            return int(s.replace('h','')) * 60
        if s in ('d','1d','day','daily','d1','interval="D"'):
            return 60  # treat as hourly boundary, but we never schedule daily here
        return 5

    # ---- Bar close: set signal every N bars
    def on_bar_close(self):
        now = now_ist()
        if not in_session(now.time(), self.cfg.session_windows):
            log(f"[{STRATEGY_NAME}] [BAR_CLOSE] Outside session @ {now.time()}")
            return
        self.bar_counter += 1
        log(f"[{STRATEGY_NAME}] [BAR_CLOSE] bar={self.bar_counter}")

        # Print current quote/LTP
        try:
            q = self.client.quotes(symbol=self.cfg.symbol, exchange=self.cfg.exchange)
            print(q)
            ltp = float(q.get('data',{}).get('ltp') or 0)
            if ltp:
                log(f"[{STRATEGY_NAME}] [QUOTE] {self.cfg.exchange}:{self.cfg.symbol} LTP ₹{ltp:.2f}")
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [WARN] quotes failed: {e}")

        if self.bar_counter % max(self.cfg.trade_every_n_bars, 1) == 0:
            self.pending_signal = True
            iv = self._parse_interval_minutes(self.cfg.interval)
            self.next_entry_time = (now + timedelta(minutes=iv)).replace(second=1, microsecond=0)
            log(f"[{STRATEGY_NAME}] ⚡ [SIGNAL] LONG queued for next bar open @ {self.next_entry_time}")
        else:
            log(f"[{STRATEGY_NAME}] [NO_SIGNAL] waiting…")

    # ---- Next bar open: place entry if a signal is pending
    def on_bar_open(self):
        now = now_ist()
        if not in_session(now.time(), self.cfg.session_windows):
            return
        if not self.pending_signal or self.in_position:
            return
        if self.next_entry_time:
            delta = (now - self.next_entry_time).total_seconds()
            if not self.cfg.ignore_entry_delta and not (0 <= delta <= 10):
                return
        self.place_entry()

    # ---- Orders
    def place_entry(self):
        try:
            action = "BUY"  # long-only to mirror backtest
            log(f"[{STRATEGY_NAME}] 🚀 [ENTRY] {action} {self.cfg.symbol} x {self.qty}")
            resp = self.client.placeorder(
                strategy=STRATEGY_NAME,
                symbol=self.cfg.symbol,
                exchange=self.cfg.exchange,
                product=self.cfg.product,
                action=action,
                price_type="MARKET",
                quantity=self.qty,
            )
            if resp.get('status') != 'success':
                log(f"[{STRATEGY_NAME}] ❌ placeorder failed: {resp}")
                self.pending_signal = False
                return
            self.entry_order_id = resp.get('orderid')
            log(f"[{STRATEGY_NAME}] [ORDER] entry oid={self.entry_order_id}")

            # Small poll to fetch average_price
            time.sleep(0.5)
            st = self.client.orderstatus(order_id=self.entry_order_id, strategy=STRATEGY_NAME)
            data = st.get('data', {})
            self.entry_price = float(data.get('average_price') or 0)
            if not self.entry_price:
                log(f"[{STRATEGY_NAME}] [WARN] average_price not ready; aborting exit legs for now")
                return

            # Compute exits (fixed rupees from backtest)
            self.tp_level = round(self.entry_price + self.cfg.profit_target_rupees, 2)
            self.sl_level = round(self.entry_price - self.cfg.stop_loss_rupees, 2)
            self.in_position = True
            self.side = 'LONG'
            self.pending_signal = False

            log(f"[{STRATEGY_NAME}] ✅ ENTRY avg ₹{self.entry_price:.2f} | TP ₹{self.tp_level:.2f} | SL ₹{self.sl_level:.2f}")
            self.place_exit_legs()
            self._persist()
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [ERROR] place_entry: {e}")

    def place_exit_legs(self):
        if not self.in_position or self.entry_price is None:
            return
        try:
            # TP (SELL LIMIT)
            tp = self.client.placeorder(
                strategy=STRATEGY_NAME,
                symbol=self.cfg.symbol,
                exchange=self.cfg.exchange,
                product=self.cfg.product,
                action="SELL",
                price_type="LIMIT",
                price=self.tp_level,
                quantity=self.qty,
            )
            # SL (SELL SL-M by default)
            sl = self.client.placeorder(
                strategy=STRATEGY_NAME,
                symbol=self.cfg.symbol,
                exchange=self.cfg.exchange,
                product=self.cfg.product,
                action="SELL",
                price_type="SL-M",
                trigger_price=self.sl_level,
                quantity=self.qty,
            )
            self.tp_order_id = tp.get('orderid')
            self.sl_order_id = sl.get('orderid')
            log(f"[{STRATEGY_NAME}] [EXITS] TP oid={self.tp_order_id} @{self.tp_level:.2f} | SL oid={self.sl_order_id} @{self.sl_level:.2f}")
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [ERROR] place_exit_legs: {e}")

    def cancel_order_silent(self, oid: Optional[str]):
        if not oid:
            return
        try:
            self.client.cancelorder(order_id=oid)
            log(f"[{STRATEGY_NAME}] [CANCEL] order_id={oid}")
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [WARN] cancel failed: {e}")

    def _realize_exit(self, exit_price: float, reason: str):
        if not self.in_position or self.entry_price is None:
            return
        points = exit_price - self.entry_price  # long-only
        # Backtest style costs (per round-trip): 2*brokerage + slippage
        costs = 2 * self.cfg.brokerage_per_trade + self.cfg.slippage_rupees
        gross = points * self.qty
        net = gross - costs
        self.realized_pnl_today += net
        emoji = "💰" if net > 0 else "💸"
        log(f"[{STRATEGY_NAME}] {emoji} [EXIT] {reason} | Entry ₹{self.entry_price:.2f} → Exit ₹{exit_price:.2f} | Gross ₹{gross:+.2f} | Costs ₹{costs:.2f} | Net ₹{net:+.2f} | Day ₹{self.realized_pnl_today:+.2f}")
        self._flat_state()
        self._persist()

    def square_off(self):
        now = now_ist()
        if not self.in_position:
            if is_square_off_time(self.cfg, now.time()):
                # Cancel any open exit legs
                self.cancel_order_silent(self.tp_order_id)
                self.cancel_order_silent(self.sl_order_id)
            return
        try:
            action = 'SELL'  # long-only
            resp = self.client.placeorder(
                strategy=STRATEGY_NAME,
                symbol=self.cfg.symbol,
                exchange=self.cfg.exchange,
                product=self.cfg.product,
                action=action,
                price_type="MARKET",
                quantity=self.qty,
            )
            if resp.get('status') == 'success':
                # Fetch last traded price for P&L approximation
                try:
                    q = self.client.quotes(symbol=self.cfg.symbol, exchange=self.cfg.exchange)
                    print(q)
                    ltp = float(q.get('data',{}).get('ltp') or 0)
                except Exception:
                    ltp = self.entry_price or 0
                self._realize_exit(ltp, reason="EOD Square-Off")
            else:
                log(f"[{STRATEGY_NAME}] [EOD] close failed: {resp}")
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [EOD] Exception: {e}")

    # ---- State
    def _persist(self):
        if not self.cfg.persist_state:
            return
        try:
            with open(f"{STRATEGY_NAME}_state.json","w") as f:
                json.dump({
                    "in_position": self.in_position,
                    "side": self.side,
                    "entry_price": self.entry_price,
                    "tp_level": self.tp_level,
                    "sl_level": self.sl_level,
                    "qty": self.qty,
                    "tp_order_id": self.tp_order_id,
                    "sl_order_id": self.sl_order_id,
                    "realized_pnl_today": self.realized_pnl_today,
                }, f)
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [WARN] persist failed: {e}")

    def _flat_state(self):
        self.in_position = False
        self.side = None
        self.entry_price = None
        self.tp_level = None
        self.sl_level = None
        self.entry_order_id = None
        self.tp_order_id = None
        self.sl_order_id = None

    # ---- Lifecycle
    def start(self):
        print("🔁 OpenAlgo Python Bot is running.")
        log(f"\n[{STRATEGY_NAME}] Starting with config:\n{json.dumps(asdict(self.cfg), indent=2)}")

        if not self.cfg.api_key:
            log(f"[{STRATEGY_NAME}] [FATAL] Please set OPENALGO_API_KEY")
            sys.exit(1)
        if not validate_symbol(self.client, self.cfg.symbol, self.cfg.exchange):
            log(f"[{STRATEGY_NAME}] [FATAL] Symbol validation failed")
            sys.exit(1)

        # Schedule bar close/open ticks
        iv = self._parse_interval_minutes(self.cfg.interval)
        # Close ~ at :59s of each interval bucket
        self.scheduler.add_job(self.on_bar_close,
                               CronTrigger(minute=f"*/{iv}", second="55", timezone=IST))
        # Open ~ shortly after new bar starts
        self.scheduler.add_job(self.on_bar_open,
                               CronTrigger(minute=f"*/{iv}", second="1", timezone=IST))

        # EOD square-off
        if self.cfg.enable_eod_square_off:
            h, m = self.cfg.square_off_time
            self.scheduler.add_job(self.square_off, CronTrigger(hour=h, minute=m, second="0", timezone=IST))

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._graceful_exit)
        signal.signal(signal.SIGTERM, self._graceful_exit)

        self.scheduler.start()
        log(f"[{STRATEGY_NAME}] Scheduler started @ interval={iv}m")

        # Idle loop to keep process alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self._graceful_exit()

    def _graceful_exit(self, *args):
        log(f"\n[{STRATEGY_NAME}] Shutting down…")
        try:
            if self.in_position:
                self.client.placeorder(
                    strategy=STRATEGY_NAME,
                    symbol=self.cfg.symbol,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action='SELL',
                    price_type='MARKET',
                    quantity=self.qty,
                )
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [WARN] close-on-exit failed: {e}")
        try:
            self.cancel_order_silent(self.tp_order_id)
            self.cancel_order_silent(self.sl_order_id)
        except Exception:
            pass
        try:
            self.scheduler.shutdown(wait=False)
        except Exception:
            pass
        log(f"[{STRATEGY_NAME}] Bye.")
        sys.exit(0)

# ----------------------------
# Entrypoint
# ----------------------------

def main():
    cfg = Config()
    bot = ScalpWithTrendBot(cfg)
    bot.start()

if __name__ == "__main__":
    main()
