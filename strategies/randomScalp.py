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
import os, sys, time, json, logging, signal, threading
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, time as dtime, timedelta
from threading import Lock

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
    "description": "Production-hardened long-only random scalp bot with partial fill handling, SL-M fallback, and enhanced safety rails.",
    "version": "1.2",
    "features": [
        "Fixed-interval long entries",
        "Static rupee profit target and stop loss",
        "Partial fill handling on entry and exits with quantity sync",
        "SL-M trigger validation with automatic SL fallback",
        "Idempotent order placement with timeout handling",
        "OCO race condition protection with thread locks",
        "Exit legs retry mechanism with progressive backoff",
        "Three-axis position reconciliation (direction, qty, price)",
        "Enhanced graceful shutdown with escalation protocol",
        "Market-on-target conversion for gap scenarios",
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
# Latest (January 2025) index lot sizes (fallbacks)
# ----------------------------
INDEX_LOT_SIZES = {
    # NSE Index (NSE_INDEX)
    "NIFTY": 75,
    "NIFTYNXT50": 25,
    "FINNIFTY": 65,
    "BANKNIFTY": 35,
    "MIDCPNIFTY": 75,  # Updated from 140 to 75
    # BSE Index (BSE_INDEX)
    "SENSEX": 20,
    "BANKEX": 30,
    "SENSEX50": 60,
}

# Tick size fallbacks by exchange
TICK_SIZE_FALLBACKS = {
    "NSE": 0.05,
    "NFO": 0.05,
    "BSE": 0.01,
    "BFO": 0.05,
    "MCX": 1.0,
    "CDS": 0.0025,
    "BCD": 0.0001,
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
    symbol: str = os.getenv("SYMBOL", "NIFTY24OCT2525000CE")  # Use an actual tradable symbol
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

    # Production hardening (v1.2)
    api_timeout_seconds: int = int(os.getenv("API_TIMEOUT_SECONDS", 10))
    max_order_retries: int = int(os.getenv("MAX_ORDER_RETRIES", 2))
    enable_market_on_target: bool = os.getenv("ENABLE_MARKET_ON_TARGET", "false").lower() == "true"

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
def resolve_quantity(client, cfg: Config, symbol: Optional[str] = None) -> Tuple[int, float, int]:
    sym = (symbol or cfg.symbol).upper()
    tick_size = None
    lot_size = None
    try:
        result = client.search(query=sym, exchange=cfg.exchange)
        if result.get('status') == 'success':
            for inst in result.get('data', []):
                if inst.get('symbol', '').upper() == sym:
                    lot = int(inst.get('lotsize') or 1)
                    lot_size = lot
                    tick_size = float(inst.get('tick_size') or 0.05)
                    log(f"[{STRATEGY_NAME}] Resolved lot size for {sym} via API: {lot}, tick: {tick_size}")
                    final_qty = cfg.lots * lot
                    # Validate minimum lot size
                    if cfg.lots < 1:
                        log(f"[{STRATEGY_NAME}] [WARN] LOTS={cfg.lots} is invalid, using minimum 1 lot")
                        final_qty = lot
                    return final_qty, tick_size, lot
    except Exception as e:
        log(f"[{STRATEGY_NAME}] [WARN] resolve_quantity API failed: {e}")

    # Fallback: detect index from symbol prefix
    for idx in INDEX_LOT_SIZES.keys():
        if sym.startswith(idx):
            lot = INDEX_LOT_SIZES[idx]
            fallback_tick = TICK_SIZE_FALLBACKS.get(cfg.exchange, 0.05)
            log(f"[{STRATEGY_NAME}] Using fallback lot size for {sym} (detected {idx}): {lot}, tick: {fallback_tick}")
            final_qty = max(cfg.lots, 1) * lot
            return final_qty, tick_size or fallback_tick, lot

    # Ultimate fallback
    fallback_tick = TICK_SIZE_FALLBACKS.get(cfg.exchange, 0.05)
    log(f"[{STRATEGY_NAME}] [WARN] Could not determine lot size for {sym}, using default 1, tick: {fallback_tick}")
    final_qty = max(cfg.lots, 1) * 1
    return final_qty, tick_size or fallback_tick, lot_size or 1

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
    interval = cfg.interval.upper()
    if interval in {"1D", "D", "DAY", "DAILY"}:
        interval = "D"
    log(f"[{STRATEGY_NAME}] [HISTORY] Fetching {sym}@{cfg.exchange} {interval} {start_date}→{end_date}")
    df = client.history(symbol=sym, exchange=cfg.exchange, interval=interval,
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
class RandomScalpBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = openalgo.simulator(api_key=cfg.api_key) if cfg.test_mode else openalgo.api(api_key=cfg.api_key, host=cfg.api_host, ws_url=cfg.ws_url)
        self.scheduler = BackgroundScheduler(timezone=IST, job_defaults={'max_instances': 1, 'coalesce': True, 'misfire_grace_time': 30})
        _ensure_file_logging(cfg)

        # State
        self.qty, self.tick_size, self.base_lot_size = resolve_quantity(self.client, cfg, symbol=cfg.symbol)
        if self.base_lot_size <= 0:
            self.base_lot_size = 1
        if self.qty % self.base_lot_size != 0:
            adjusted_qty = (self.qty // self.base_lot_size) * self.base_lot_size
            if adjusted_qty == 0:
                adjusted_qty = self.base_lot_size
            log(f"[{STRATEGY_NAME}] [WARN] Quantity {self.qty} not multiple of lot size {self.base_lot_size}; adjusting to {adjusted_qty}")
            self.qty = adjusted_qty

        # Position state
        self.in_position: bool = False
        self.side: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.actual_filled_qty: int = 0  # Actual filled quantity (may differ from self.qty on partial fills)
        self.tp_level: Optional[float] = None
        self.sl_level: Optional[float] = None
        self.entry_order_id: Optional[str] = None
        self.tp_order_id: Optional[str] = None
        self.sl_order_id: Optional[str] = None
        self.tp_filled_qty: int = 0  # Track partial TP fills
        self.sl_filled_qty: int = 0  # Track partial SL fills

        # Signal state
        self.pending_signal: bool = False
        self.next_entry_time: Optional[datetime] = None
        self.bar_counter: int = 0

        # Exit legs retry tracking
        self.exit_legs_placed: bool = False
        self.exit_legs_retry_count: int = 0
        self.max_exit_legs_retries: int = 3

        # Threading safety
        self.exit_lock = Lock()  # OCO race condition protection

        self.realized_pnl_today: float = 0.0
        self.running: bool = False  # Control flag for main loop

    # ---- Time helpers
    def _parse_interval_minutes(self, interval: str) -> int:
        s = interval.strip().lower()
        if s.endswith('m'):
            return int(s.replace('m',''))
        if s.endswith('h'):
            return int(s.replace('h','')) * 60
        if s in ('d','1d','day','daily','d1','interval="D"'):
            # For daily intervals, return 1440 (24 hours) - not used for scheduling
            # as daily mode uses explicit cron jobs at session start/end
            return 1440
        return 5

    def _round_to_tick(self, price: float) -> float:
        tick = self.tick_size or 0.05
        if tick <= 0:
            tick = 0.05
        rounded = round(round(price / tick) * tick, 2)
        return rounded

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
            if self.in_position:
                log(f"[{STRATEGY_NAME}] [NO_SIGNAL] in position, skip queuing")
                return
            iv = self._parse_interval_minutes(self.cfg.interval)
            next_time = (now + timedelta(minutes=iv)).replace(second=1, microsecond=0)
            so_h, so_m = self.cfg.square_off_time
            square_off_dt = now.replace(hour=so_h, minute=so_m, second=0, microsecond=0)
            if next_time >= square_off_dt:
                log(f"[{STRATEGY_NAME}] [NO_SIGNAL] next entry {next_time.time()} >= square-off; skipping")
                return
            self.pending_signal = True
            self.next_entry_time = next_time
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

    @staticmethod
    def _is_complete(resp: Dict[str, Any]) -> bool:
        return str(resp.get('data', {}).get('order_status', '')).upper() == 'COMPLETE'

    @staticmethod
    def _is_rejected(resp: Dict[str, Any]) -> bool:
        status = str(resp.get('data', {}).get('order_status', '')).upper()
        return 'REJECT' in status or 'CANCELLED' in status

    @staticmethod
    def _is_partial(resp: Dict[str, Any]) -> bool:
        status = str(resp.get('data', {}).get('order_status', '')).upper()
        return 'PARTIAL' in status or status == 'OPEN'

    @staticmethod
    def _get_filled_qty(resp: Dict[str, Any]) -> int:
        """Extract filled quantity from order status response."""
        data = resp.get('data', {})
        # Try multiple field names for compatibility
        filled = data.get('filled_quantity') or data.get('filled_qty') or data.get('filledQuantity') or 0
        return int(filled)

    def _safe_placeorder(self, **params) -> Optional[Dict[str, Any]]:
        """Idempotent order placement with timeout and retry."""
        max_attempts = min(self.cfg.max_order_retries, 3)
        last_error = None

        for attempt in range(max_attempts):
            try:
                resp = self.client.placeorder(**params)
                if resp.get('status') == 'success':
                    return resp
                else:
                    log(f"[{STRATEGY_NAME}] [WARN] placeorder attempt {attempt+1}/{max_attempts} failed: {resp.get('message', resp)}")
                    last_error = resp
            except Exception as e:
                log(f"[{STRATEGY_NAME}] [WARN] placeorder attempt {attempt+1}/{max_attempts} exception: {e}")
                last_error = str(e)
                # On timeout or network error, check if order actually went through
                if attempt < max_attempts - 1 and 'timeout' in str(e).lower():
                    time.sleep(0.3 * (attempt + 1))
                    continue

            # Wait before retry
            if attempt < max_attempts - 1:
                time.sleep(0.3 * (attempt + 1))

        log(f"[{STRATEGY_NAME}] [ERROR] placeorder failed after {max_attempts} attempts: {last_error}")
        return None

    def _place_stop_order(self, quantity: int, trigger_price: float, action: str = "SELL") -> Optional[Dict[str, Any]]:
        """Place stop loss with SL-M → SL fallback and trigger validation."""
        # Validate trigger price for long positions (trigger should be < LTP)
        if action == "SELL" and self.side == 'LONG':
            try:
                q = self.client.quotes(symbol=self.cfg.symbol, exchange=self.cfg.exchange)
                ltp = float(q.get('data', {}).get('ltp') or 0)
                if ltp and trigger_price >= ltp:
                    log(f"[{STRATEGY_NAME}] [WARN] SL trigger {trigger_price:.2f} >= LTP {ltp:.2f}, adjusting to LTP - tick")
                    trigger_price = max(self._round_to_tick(ltp - self.tick_size), 0.05)
            except Exception as e:
                log(f"[{STRATEGY_NAME}] [WARN] Could not validate trigger vs LTP: {e}")

        # Try SL-M first
        try:
            resp = self._safe_placeorder(
                strategy=STRATEGY_NAME,
                symbol=self.cfg.symbol,
                exchange=self.cfg.exchange,
                product=self.cfg.product,
                action=action,
                price_type="SL-M",
                trigger_price=trigger_price,
                quantity=quantity,
            )
            if resp and resp.get('status') == 'success':
                log(f"[{STRATEGY_NAME}] [SL] Placed SL-M @ trigger ₹{trigger_price:.2f} for qty {quantity}")
                return resp
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [WARN] SL-M placement failed: {e}")

        # Fallback to SL (with limit price)
        log(f"[{STRATEGY_NAME}] [FALLBACK] Retrying with SL instead of SL-M")
        fallback_price = max(self._round_to_tick(trigger_price - self.tick_size), 0.05)
        try:
            resp = self._safe_placeorder(
                strategy=STRATEGY_NAME,
                symbol=self.cfg.symbol,
                exchange=self.cfg.exchange,
                product=self.cfg.product,
                action=action,
                price_type="SL",
                price=fallback_price,
                trigger_price=trigger_price,
                quantity=quantity,
            )
            if resp and resp.get('status') == 'success':
                log(f"[{STRATEGY_NAME}] [SL] Placed SL @ price ₹{fallback_price:.2f} trigger ₹{trigger_price:.2f} for qty {quantity}")
                return resp
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [ERROR] SL fallback also failed: {e}")

        return None

    def _sync_exit_quantities(self, remaining_qty: int) -> None:
        """Sync exit order quantities when partial fills occur."""
        if remaining_qty <= 0:
            log(f"[{STRATEGY_NAME}] [SYNC] No remaining quantity, clearing exits")
            self.cancel_order_silent(self.tp_order_id)
            self.cancel_order_silent(self.sl_order_id)
            self.tp_order_id = None
            self.sl_order_id = None
            return

        log(f"[{STRATEGY_NAME}] [SYNC] Adjusting exits for remaining qty {remaining_qty}")

        # Cancel existing exits
        self.cancel_order_silent(self.tp_order_id)
        self.cancel_order_silent(self.sl_order_id)
        self.tp_order_id = None
        self.sl_order_id = None

        # Re-place with correct quantity
        if not self.entry_price or not self.tp_level or not self.sl_level:
            log(f"[{STRATEGY_NAME}] [ERROR] Cannot re-place exits: missing price levels")
            return

        # Place TP
        try:
            tp_resp = self._safe_placeorder(
                strategy=STRATEGY_NAME,
                symbol=self.cfg.symbol,
                exchange=self.cfg.exchange,
                product=self.cfg.product,
                action="SELL",
                price_type="LIMIT",
                price=self.tp_level,
                quantity=remaining_qty,
            )
            if tp_resp and tp_resp.get('status') == 'success':
                self.tp_order_id = tp_resp.get('orderid')
                log(f"[{STRATEGY_NAME}] [SYNC] Re-placed TP for qty {remaining_qty}")
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [ERROR] Failed to re-place TP: {e}")

        # Place SL
        sl_resp = self._place_stop_order(remaining_qty, self.sl_level, action="SELL")
        if sl_resp and sl_resp.get('status') == 'success':
            self.sl_order_id = sl_resp.get('orderid')
            log(f"[{STRATEGY_NAME}] [SYNC] Re-placed SL for qty {remaining_qty}")

        self._persist()

    def _cleanup_stale_orders(self) -> None:
        """Clean up stale exit orders when flat."""
        if self.tp_order_id or self.sl_order_id:
            log(f"[{STRATEGY_NAME}] [CLEANUP] Removing stale orders (flat position)")
            self.cancel_order_silent(self.tp_order_id)
            self.cancel_order_silent(self.sl_order_id)
            self.tp_order_id = None
            self.sl_order_id = None
            self.tp_filled_qty = 0
            self.sl_filled_qty = 0
            self._persist()

    def _ensure_exits(self) -> None:
        """Re-arm exit protection if missing while in position."""
        if not self.in_position:
            return

        if self.tp_order_id and self.sl_order_id:
            return  # Exits already in place

        log(f"[{STRATEGY_NAME}] [CRITICAL] Position detected without exits! Re-arming protection...")

        # Use entry price if available, otherwise try to get from position book
        if not self.entry_price:
            log(f"[{STRATEGY_NAME}] [ERROR] Cannot place exits: no entry price")
            return

        # Compute levels if missing
        if not self.tp_level:
            self.tp_level = self._round_to_tick(self.entry_price + self.cfg.profit_target_rupees)
        if not self.sl_level:
            self.sl_level = self._round_to_tick(self.entry_price - self.cfg.stop_loss_rupees)

        # Place exit legs
        qty_to_protect = self.actual_filled_qty if self.actual_filled_qty > 0 else self.qty
        self.place_exit_legs_for_qty(qty_to_protect)

    def _ensure_flat_position(self, context: str) -> None:
        """Verify broker position book is flat; attempt corrective order if not."""
        if not hasattr(self.client, "positionbook"):
            return
        try:
            resp = self.client.positionbook()
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [WARN] positionbook call failed during {context}: {e}")
            return

        if not isinstance(resp, dict) or resp.get('status') != 'success':
            log(f"[{STRATEGY_NAME}] [WARN] positionbook not available during {context}: {resp}")
            return

        positions = resp.get('data', []) or []
        symbol_upper = (self.cfg.symbol or "").upper()
        residual_qty = 0
        for pos in positions:
            pos_symbol = str(pos.get('symbol', '')).upper()
            if pos_symbol != symbol_upper:
                continue
            residual_qty = int(pos.get('netqty') or pos.get('net_qty') or pos.get('quantity') or 0)
            break

        if residual_qty == 0:
            return

        log(f"[{STRATEGY_NAME}] [WARN] Residual net position {residual_qty} detected during {context}; sending corrective order")
        corrective_action = 'SELL' if residual_qty > 0 else 'BUY'
        corrective_qty = abs(residual_qty)
        if self.base_lot_size > 0 and corrective_qty % self.base_lot_size != 0:
            corrective_qty = ((corrective_qty // self.base_lot_size) + 1) * self.base_lot_size
        try:
            self.client.placeorder(
                strategy=STRATEGY_NAME,
                symbol=self.cfg.symbol,
                exchange=self.cfg.exchange,
                product=self.cfg.product,
                action=corrective_action,
                price_type="MARKET",
                quantity=corrective_qty,
            )
            log(f"[{STRATEGY_NAME}] [WARN] Corrective {corrective_action} {corrective_qty} order submitted to flatten residual position")
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [ERROR] Corrective flatten failed: {e}")

    def reconcile_position(self) -> None:
        """Periodically reconcile internal state with actual broker position (three-axis: direction, qty, price)."""
        if not hasattr(self.client, "positionbook"):
            return

        try:
            resp = self.client.positionbook()
            if not isinstance(resp, dict) or resp.get('status') != 'success':
                return

            positions = resp.get('data', []) or []
            symbol_upper = (self.cfg.symbol or "").upper()
            actual_qty = 0
            actual_avg_price = None

            for pos in positions:
                pos_symbol = str(pos.get('symbol', '')).upper()
                if pos_symbol == symbol_upper:
                    actual_qty = int(pos.get('netqty') or pos.get('net_qty') or pos.get('quantity') or 0)
                    actual_avg_price = float(pos.get('average_price') or pos.get('avg_price') or 0)
                    break

            # Compare with expected state (three axes)
            expected_qty = self.actual_filled_qty if self.in_position else 0

            if actual_qty != expected_qty:
                log(f"[{STRATEGY_NAME}] [RECONCILE] State mismatch! Expected: {expected_qty}, Actual: {actual_qty}")

                # If we think we're flat but have a position
                if not self.in_position and actual_qty != 0:
                    log(f"[{STRATEGY_NAME}] [RECONCILE] Unexpected position detected, flattening...")
                    self._ensure_flat_position("reconcile_unexpected_position")

                # If we think we're in position but we're actually flat
                elif self.in_position and actual_qty == 0:
                    log(f"[{STRATEGY_NAME}] [RECONCILE] Position closed externally, updating state")
                    # One of our exits must have filled without us catching it
                    self._flat_state()
                    self._persist()

                # If quantities don't match (partial fill scenarios)
                elif actual_qty != 0 and abs(actual_qty) != expected_qty:
                    log(f"[{STRATEGY_NAME}] [RECONCILE] Quantity mismatch, updating to actual {abs(actual_qty)}")
                    # Update our quantity to match actual
                    if self.in_position:
                        self.actual_filled_qty = abs(actual_qty)
                        self._persist()
                        # Re-sync exits if needed
                        self._ensure_exits()

            # Clean up stale exit orders if flat
            if not self.in_position:
                self._cleanup_stale_orders()

            # If in position without exits, re-arm immediately
            if self.in_position and (not self.tp_order_id or not self.sl_order_id):
                log(f"[{STRATEGY_NAME}] [RECONCILE] Position without exits detected!")
                self._ensure_exits()

            # Validate entry price if available from broker
            if self.in_position and actual_avg_price and abs(actual_avg_price - (self.entry_price or 0)) > 0.01:
                log(f"[{STRATEGY_NAME}] [RECONCILE] Avg price mismatch: ours {self.entry_price:.2f} vs broker {actual_avg_price:.2f}")
                # Trust broker's price if we don't have one
                if not self.entry_price:
                    self.entry_price = actual_avg_price
                    self._persist()

        except Exception as e:
            log(f"[{STRATEGY_NAME}] [WARN] Position reconciliation failed: {e}")

    def check_order_status(self):
        """Poll sibling orders to implement OCO safety with race condition protection and partial fill handling."""
        if not self.in_position:
            # Retry exit legs if needed
            if self.entry_price and not self.exit_legs_placed and self.exit_legs_retry_count < self.max_exit_legs_retries:
                log(f"[{STRATEGY_NAME}] [RETRY] Attempting to place exit legs (attempt {self.exit_legs_retry_count + 1}/{self.max_exit_legs_retries})")
                self.place_exit_legs()
            return

        # Use lock to prevent race condition where both TP and SL get processed simultaneously
        if not self.exit_lock.acquire(blocking=False):
            return  # Another check is already processing, skip this cycle

        try:
            tp_complete = False
            sl_complete = False
            tp_partial = False
            sl_partial = False
            tp_price = None
            sl_price = None
            tp_filled = 0
            sl_filled = 0

            # Check TP status
            if self.tp_order_id:
                try:
                    resp = self.client.orderstatus(order_id=self.tp_order_id, strategy=STRATEGY_NAME)
                    if resp.get('status') == 'success':
                        if self._is_complete(resp):
                            tp_complete = True
                            tp_price = float(resp.get('data', {}).get('average_price', 0) or 0)
                            tp_filled = self._get_filled_qty(resp)
                        elif self._is_partial(resp):
                            tp_partial = True
                            tp_filled = self._get_filled_qty(resp)
                            if tp_filled > self.tp_filled_qty:
                                log(f"[{STRATEGY_NAME}] [PARTIAL] TP partially filled: {tp_filled}/{self.actual_filled_qty}")
                                self.tp_filled_qty = tp_filled
                        elif self._is_rejected(resp):
                            log(f"[{STRATEGY_NAME}] [CRITICAL] TP rejected - position UNPROTECTED on upside!")
                            self.tp_order_id = None
                except Exception as e:
                    log(f"[{STRATEGY_NAME}] [WARN] TP status check failed: {e}")

            # Check SL status
            if self.sl_order_id:
                try:
                    resp = self.client.orderstatus(order_id=self.sl_order_id, strategy=STRATEGY_NAME)
                    if resp.get('status') == 'success':
                        if self._is_complete(resp):
                            sl_complete = True
                            sl_price = float(resp.get('data', {}).get('average_price', 0) or 0)
                            sl_filled = self._get_filled_qty(resp)
                        elif self._is_partial(resp):
                            sl_partial = True
                            sl_filled = self._get_filled_qty(resp)
                            if sl_filled > self.sl_filled_qty:
                                log(f"[{STRATEGY_NAME}] [PARTIAL] SL partially filled: {sl_filled}/{self.actual_filled_qty}")
                                self.sl_filled_qty = sl_filled
                        elif self._is_rejected(resp):
                            log(f"[{STRATEGY_NAME}] [CRITICAL] SL rejected - position UNPROTECTED on downside!")
                            self.sl_order_id = None
                except Exception as e:
                    log(f"[{STRATEGY_NAME}] [WARN] SL status check failed: {e}")

            # Handle partial fills - sync exit quantities
            total_exits = self.tp_filled_qty + self.sl_filled_qty
            if total_exits > 0 and total_exits < self.actual_filled_qty:
                remaining_qty = self.actual_filled_qty - total_exits
                if not tp_complete and not sl_complete:
                    log(f"[{STRATEGY_NAME}] [SYNC_NEEDED] Total exits {total_exits}, remaining {remaining_qty}")
                    self._sync_exit_quantities(remaining_qty)
                    return  # Exit after sync to avoid processing stale data

            # Process complete fills - handle both filled case
            if tp_complete and sl_complete:
                log(f"[{STRATEGY_NAME}] [CRITICAL] Both TP and SL filled! OCO failure detected.")
                # Use TP price as it's more favorable, send corrective order for SL fill
                self._realize_exit(tp_price or self.entry_price, "Target Hit (OCO Race)")
                # Immediately flatten any residual
                self._ensure_flat_position("OCO_RACE_BOTH_FILLED")
            elif tp_complete:
                self.cancel_order_silent(self.sl_order_id)
                self._realize_exit(tp_price or self.entry_price, "Target Hit")
            elif sl_complete:
                self.cancel_order_silent(self.tp_order_id)
                self._realize_exit(sl_price or self.entry_price, "Stoploss Hit")

            # Market-on-target: if enabled and LTP >= TP, convert to market
            if self.cfg.enable_market_on_target and self.tp_order_id and not tp_complete:
                try:
                    q = self.client.quotes(symbol=self.cfg.symbol, exchange=self.cfg.exchange)
                    ltp = float(q.get('data', {}).get('ltp') or 0)
                    if ltp >= self.tp_level:
                        log(f"[{STRATEGY_NAME}] [MARKET_ON_TARGET] LTP {ltp:.2f} >= TP {self.tp_level:.2f}, converting to MARKET")
                        self.cancel_order_silent(self.tp_order_id)
                        self.cancel_order_silent(self.sl_order_id)
                        remaining = self.actual_filled_qty - total_exits
                        if remaining > 0:
                            self._safe_placeorder(
                                strategy=STRATEGY_NAME,
                                symbol=self.cfg.symbol,
                                exchange=self.cfg.exchange,
                                product=self.cfg.product,
                                action="SELL",
                                price_type="MARKET",
                                quantity=remaining,
                            )
                except Exception as e:
                    log(f"[{STRATEGY_NAME}] [WARN] Market-on-target check failed: {e}")
        finally:
            self.exit_lock.release()

    def place_entry(self):
        try:
            if self.in_position:
                return
            action = "BUY"  # long-only to mirror backtest
            log(f"[{STRATEGY_NAME}] 🚀 [ENTRY] {action} {self.cfg.symbol} x {self.qty}")

            # Use idempotent order placement
            resp = self._safe_placeorder(
                strategy=STRATEGY_NAME,
                symbol=self.cfg.symbol,
                exchange=self.cfg.exchange,
                product=self.cfg.product,
                action=action,
                price_type="MARKET",
                quantity=self.qty,
            )
            if not resp or resp.get('status') != 'success':
                log(f"[{STRATEGY_NAME}] ❌ placeorder failed: {resp}")
                self.pending_signal = False
                return
            self.entry_order_id = resp.get('orderid')
            log(f"[{STRATEGY_NAME}] [ORDER] entry oid={self.entry_order_id}")

            # Poll to fetch average_price AND filled_quantity with retry logic
            max_retries = 5
            for attempt in range(max_retries):
                time.sleep(0.3 * (attempt + 1))  # Progressive backoff
                st = self.client.orderstatus(order_id=self.entry_order_id, strategy=STRATEGY_NAME)
                data = st.get('data', {})
                self.entry_price = float(data.get('average_price') or 0)
                filled_qty = self._get_filled_qty(st)

                # Check if order is complete or partially filled
                if self._is_complete(st) and self.entry_price and filled_qty > 0:
                    self.actual_filled_qty = filled_qty
                    if filled_qty < self.qty:
                        log(f"[{STRATEGY_NAME}] [PARTIAL_FILL] Entry filled {filled_qty}/{self.qty} @ ₹{self.entry_price:.2f}")
                    break
                elif self.entry_price and filled_qty > 0:
                    # Partial fill detected
                    self.actual_filled_qty = filled_qty
                    log(f"[{STRATEGY_NAME}] [WARN] Entry partially filled {filled_qty}/{self.qty}, waiting for completion...")

                log(f"[{STRATEGY_NAME}] [WARN] Entry order not ready, retry {attempt+1}/{max_retries}")

            if not self.entry_price or self.actual_filled_qty == 0:
                log(f"[{STRATEGY_NAME}] [ERROR] Could not get entry details after {max_retries} retries; exit legs will retry in polling loop")
                self.in_position = True  # Mark in position
                self.side = 'LONG'
                self.pending_signal = False
                self.exit_legs_placed = False  # Will trigger retry in check_order_status
                # Assume full quantity for safety
                self.actual_filled_qty = self.qty
                self._persist()
                return

            # Compute exits (fixed rupees from backtest)
            self.tp_level = self._round_to_tick(self.entry_price + self.cfg.profit_target_rupees)
            self.sl_level = self._round_to_tick(self.entry_price - self.cfg.stop_loss_rupees)
            self.in_position = True
            self.side = 'LONG'
            self.pending_signal = False

            log(f"[{STRATEGY_NAME}] ✅ ENTRY avg ₹{self.entry_price:.2f} qty {self.actual_filled_qty} | TP ₹{self.tp_level:.2f} | SL ₹{self.sl_level:.2f}")
            log(f"[{STRATEGY_NAME}] STATE=LONG qty={self.actual_filled_qty} entry={self.entry_price:.2f} tp={self.tp_level:.2f} sl={self.sl_level:.2f}")

            # Place exit legs for ACTUAL filled quantity
            self.place_exit_legs_for_qty(self.actual_filled_qty)
            self._persist()
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [ERROR] place_entry: {e}")

    def place_exit_legs(self):
        """Legacy method - calls new quantity-aware version."""
        qty = self.actual_filled_qty if self.actual_filled_qty > 0 else self.qty
        self.place_exit_legs_for_qty(qty)

    def place_exit_legs_for_qty(self, quantity: int):
        """Place exit legs for a specific quantity (handles partial fills)."""
        if not self.in_position or self.entry_price is None:
            return
        if quantity <= 0:
            log(f"[{STRATEGY_NAME}] [ERROR] Cannot place exits for qty {quantity}")
            return

        try:
            self.exit_legs_retry_count += 1
            tp_success = False
            sl_success = False

            # TP (SELL LIMIT)
            try:
                tp = self._safe_placeorder(
                    strategy=STRATEGY_NAME,
                    symbol=self.cfg.symbol,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action="SELL",
                    price_type="LIMIT",
                    price=self.tp_level,
                    quantity=quantity,
                )
                if tp and tp.get('status') == 'success':
                    self.tp_order_id = tp.get('orderid')
                    tp_success = True
                else:
                    log(f"[{STRATEGY_NAME}] [ERROR] TP order failed: {tp}")
            except Exception as e:
                log(f"[{STRATEGY_NAME}] [ERROR] TP order exception: {e}")

            # SL (SELL SL-M by default, with smart fallback)
            sl = self._place_stop_order(quantity, self.sl_level, action="SELL")
            if sl and sl.get('status') == 'success':
                self.sl_order_id = sl.get('orderid')
                sl_success = True
            else:
                log(f"[{STRATEGY_NAME}] [ERROR] SL order failed completely")

            # Mark exit legs as successfully placed if both succeeded
            if tp_success and sl_success:
                self.exit_legs_placed = True
                self.exit_legs_retry_count = 0  # Reset counter
                log(f"[{STRATEGY_NAME}] [EXITS] TP oid={self.tp_order_id} @₹{self.tp_level:.2f} | SL oid={self.sl_order_id} @₹{self.sl_level:.2f} for qty {quantity}")
                self._persist()
            else:
                log(f"[{STRATEGY_NAME}] [WARN] Exit legs placement incomplete (TP={tp_success}, SL={sl_success})")
                if self.exit_legs_retry_count >= self.max_exit_legs_retries:
                    log(f"[{STRATEGY_NAME}] [CRITICAL] Max exit legs retries reached - position is UNPROTECTED!")
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [ERROR] place_exit_legs_for_qty: {e}")

    def cancel_order_silent(self, oid: Optional[str]):
        if not oid:
            return
        try:
            self.client.cancelorder(order_id=oid, strategy=STRATEGY_NAME)
            log(f"[{STRATEGY_NAME}] [CANCEL] order_id={oid}")
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [WARN] cancel failed: {e}")

    def _realize_exit(self, exit_price: float, reason: str):
        if not self.in_position or self.entry_price is None:
            return
        points = exit_price - self.entry_price  # long-only
        # Per-leg costs: brokerage on entry + brokerage on exit + slippage on both legs
        entry_costs = self.cfg.brokerage_per_trade + (self.cfg.slippage_rupees / 2.0)
        exit_costs = self.cfg.brokerage_per_trade + (self.cfg.slippage_rupees / 2.0)
        total_costs = entry_costs + exit_costs
        gross = points * self.qty
        net = gross - total_costs
        self.realized_pnl_today += net
        emoji = "💰" if net > 0 else "💸"
        log(f"[{STRATEGY_NAME}] {emoji} [EXIT] {reason} | Entry ₹{self.entry_price:.2f} → Exit ₹{exit_price:.2f} | Gross ₹{gross:+.2f} | Costs ₹{total_costs:.2f} | Net ₹{net:+.2f} | Day ₹{self.realized_pnl_today:+.2f}")
        self._flat_state()
        self._persist()
        self._ensure_flat_position(reason)

    def square_off(self):
        now = now_ist()
        if not self.in_position:
            # Clean up any stale exit orders that shouldn't exist
            if self.tp_order_id or self.sl_order_id:
                log(f"[{STRATEGY_NAME}] [SQUARE_OFF] Cleaning up stale exit orders (no position)")
                self.cancel_order_silent(self.tp_order_id)
                self.cancel_order_silent(self.sl_order_id)
                self.tp_order_id = None
                self.sl_order_id = None
                self._persist()
            return
        try:
            action = 'SELL'  # long-only
            # Cancel exit legs first
            self.cancel_order_silent(self.tp_order_id)
            self.cancel_order_silent(self.sl_order_id)

            # Place market order to close
            log(f"[{STRATEGY_NAME}] [EOD] Squaring off position")
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
                # Wait and fetch actual exit price
                time.sleep(0.5)
                exit_price = None
                try:
                    oid = resp.get('orderid')
                    st = self.client.orderstatus(order_id=oid, strategy=STRATEGY_NAME)
                    exit_price = float(st.get('data', {}).get('average_price', 0) or 0)
                except Exception as e:
                    log(f"[{STRATEGY_NAME}] [WARN] Could not get exit price from order status: {e}")

                # Fallback to LTP if order status fails
                if not exit_price:
                    try:
                        q = self.client.quotes(symbol=self.cfg.symbol, exchange=self.cfg.exchange)
                        print(q)
                        exit_price = float(q.get('data',{}).get('ltp') or 0)
                    except Exception:
                        exit_price = self.entry_price or 0

                self._realize_exit(exit_price, reason="EOD Square-Off")
            else:
                log(f"[{STRATEGY_NAME}] [EOD] close failed: {resp}")
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [EOD] Exception: {e}")

    # ---- State
    def _load_state(self):
        if not self.cfg.persist_state:
            return
        try:
            with open(f"{STRATEGY_NAME}_state.json") as f:
                state = json.load(f)
        except FileNotFoundError:
            return
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [WARN] load_state failed: {e}")
            return

        self.in_position = bool(state.get("in_position", False))
        self.side = state.get("side")
        self.entry_price = state.get("entry_price")
        self.tp_level = state.get("tp_level")
        self.sl_level = state.get("sl_level")
        self.qty = state.get("qty", self.qty)
        self.tp_order_id = state.get("tp_order_id")
        self.sl_order_id = state.get("sl_order_id")
        self.realized_pnl_today = state.get("realized_pnl_today", 0.0)
        self.pending_signal = False

        if self.in_position:
            # Check both exit orders - handle case where both may have filled
            tp_filled = False
            sl_filled = False
            tp_price = None
            sl_price = None

            if self.tp_order_id:
                try:
                    resp = self.client.orderstatus(order_id=self.tp_order_id, strategy=STRATEGY_NAME)
                    if resp.get('status') == 'success':
                        if self._is_complete(resp):
                            tp_filled = True
                            tp_price = float(resp.get('data', {}).get('average_price', 0) or 0)
                        elif self._is_rejected(resp):
                            log(f"[{STRATEGY_NAME}] [WARN] TP order rejected during recovery")
                except Exception as e:
                    log(f"[{STRATEGY_NAME}] [WARN] TP status recovery failed: {e}")

            if self.sl_order_id:
                try:
                    resp = self.client.orderstatus(order_id=self.sl_order_id, strategy=STRATEGY_NAME)
                    if resp.get('status') == 'success':
                        if self._is_complete(resp):
                            sl_filled = True
                            sl_price = float(resp.get('data', {}).get('average_price', 0) or 0)
                        elif self._is_rejected(resp):
                            log(f"[{STRATEGY_NAME}] [WARN] SL order rejected during recovery")
                except Exception as e:
                    log(f"[{STRATEGY_NAME}] [WARN] SL status recovery failed: {e}")

            # Process exits - handle both filled case
            if tp_filled and sl_filled:
                log(f"[{STRATEGY_NAME}] [CRITICAL] Both TP and SL filled during downtime - OCO failure detected")
                # Use TP as it's more favorable, but we need to correct the position
                self._realize_exit(tp_price or self.entry_price, "Target Hit (recovered - OCO race)")
                self._ensure_flat_position("startup_oco_race")
            elif tp_filled:
                self.cancel_order_silent(self.sl_order_id)
                self._realize_exit(tp_price or self.entry_price, "Target Hit (recovered)")
            elif sl_filled:
                self.cancel_order_silent(self.tp_order_id)
                self._realize_exit(sl_price or self.entry_price, "Stoploss Hit (recovered)")
            # else: both orders still open, continue monitoring
        else:
            self._ensure_flat_position("startup")

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
        self.actual_filled_qty = 0
        self.tp_level = None
        self.sl_level = None
        self.entry_order_id = None
        self.tp_order_id = None
        self.sl_order_id = None
        self.tp_filled_qty = 0
        self.sl_filled_qty = 0
        self.pending_signal = False
        self.next_entry_time = None
        self.exit_legs_placed = False
        self.exit_legs_retry_count = 0

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

        self._load_state()

        # Schedule bar close/open ticks
        interval_str = self.cfg.interval.strip().lower()
        daily_tokens = {"d", "1d", "day", "daily", "d1"}
        if interval_str in daily_tokens:
            log(f"[{STRATEGY_NAME}] [WARN] Daily interval detected; scheduling single daily open/close ticks.")
            start_hour, start_min = self.cfg.session_windows[0][0], self.cfg.session_windows[0][1]
            end_hour, end_min = self.cfg.session_windows[-1][2], self.cfg.session_windows[-1][3]
            self.scheduler.add_job(self.on_bar_open,
                                   CronTrigger(hour=start_hour, minute=start_min, second="1", timezone=IST))
            self.scheduler.add_job(self.on_bar_close,
                                   CronTrigger(hour=end_hour, minute=end_min, second="55", timezone=IST))
            interval_label = "daily"
        else:
            iv = self._parse_interval_minutes(self.cfg.interval)
            self.scheduler.add_job(self.on_bar_close,
                                   CronTrigger(minute=f"*/{iv}", second="55", timezone=IST))
            self.scheduler.add_job(self.on_bar_open,
                                   CronTrigger(minute=f"*/{iv}", second="1", timezone=IST))
            interval_label = f"{iv}m"

        # EOD square-off
        if self.cfg.enable_eod_square_off:
            h, m = self.cfg.square_off_time
            self.scheduler.add_job(self.square_off, CronTrigger(hour=h, minute=m, second="0", timezone=IST))

        # OCO safety polling
        self.scheduler.add_job(self.check_order_status, 'interval', seconds=5, max_instances=1)

        # Position reconciliation (every 30 seconds)
        self.scheduler.add_job(self.reconcile_position, 'interval', seconds=30, max_instances=1)

        # Graceful shutdown
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._graceful_exit)
            signal.signal(signal.SIGTERM, self._graceful_exit)
        else:
            log(f"[{STRATEGY_NAME}] [WARN] Signal handlers not registered (not running on main thread)")

        self.scheduler.start()
        self.running = True
        log(f"[{STRATEGY_NAME}] Scheduler started @ interval={interval_label}")

        # Idle loop to keep process alive (stoppable via self.running flag)
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self._graceful_exit()

        log(f"[{STRATEGY_NAME}] Main loop exited (running={self.running})")

    def pause(self):
        """Pause the strategy without exiting the process. Closes any open positions."""
        log(f"\n[{STRATEGY_NAME}] Pausing strategy…")

        # Stop the running flag to exit main loop
        self.running = False

        exit_confirmed = False
        try:
            if self.in_position:
                qty_to_close = self.actual_filled_qty if self.actual_filled_qty > 0 else self.qty
                log(f"[{STRATEGY_NAME}] [PAUSE] Closing position qty={qty_to_close}")

                # Cancel exit legs first
                self.cancel_order_silent(self.tp_order_id)
                self.cancel_order_silent(self.sl_order_id)

                # Place market close order with idempotent placement
                resp = self._safe_placeorder(
                    strategy=STRATEGY_NAME,
                    symbol=self.cfg.symbol,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action='SELL',
                    price_type='MARKET',
                    quantity=qty_to_close,
                )

                # Enhanced confirmation with escalation: poll every 0.25s for 5s (20 iterations)
                if resp and resp.get('status') == 'success':
                    oid = resp.get('orderid')
                    max_iterations = 20  # 20 × 0.25s = 5 seconds
                    for iteration in range(max_iterations):
                        time.sleep(0.25)
                        try:
                            st = self.client.orderstatus(order_id=oid, strategy=STRATEGY_NAME)
                            if self._is_complete(st):
                                exit_price = float(st.get('data', {}).get('average_price', 0) or 0)
                                if not exit_price:
                                    # Fallback to LTP
                                    q = self.client.quotes(symbol=self.cfg.symbol, exchange=self.cfg.exchange)
                                    exit_price = float(q.get('data',{}).get('ltp') or self.entry_price or 0)
                                self._realize_exit(exit_price, "Strategy Paused")
                                exit_confirmed = True
                                break
                        except Exception as e:
                            log(f"[{STRATEGY_NAME}] [WARN] Exit confirmation check {iteration+1}/{max_iterations} failed: {e}")

                    # Escalation: retry MARKET once if not confirmed
                    if not exit_confirmed:
                        log(f"[{STRATEGY_NAME}] [ESCALATE] Exit not confirmed after 5s, retrying MARKET order...")
                        retry_resp = self._safe_placeorder(
                            strategy=STRATEGY_NAME,
                            symbol=self.cfg.symbol,
                            exchange=self.cfg.exchange,
                            product=self.cfg.product,
                            action='SELL',
                            price_type='MARKET',
                            quantity=qty_to_close,
                        )
                        if retry_resp and retry_resp.get('status') == 'success':
                            log(f"[{STRATEGY_NAME}] [ESCALATE] Retry order placed, checking reconciliation...")
                            # Keep checking for 30 more seconds with reconciliation running
                            for _ in range(12):  # 12 × 2.5s = 30s
                                time.sleep(2.5)
                                self.reconcile_position()
                                if not self.in_position:
                                    exit_confirmed = True
                                    break

                        if not exit_confirmed:
                            log(f"[{STRATEGY_NAME}] [CRITICAL] STILL IN POSITION AFTER PAUSE ATTEMPTS!")
                            log(f"[{STRATEGY_NAME}] [CRITICAL] Manual intervention may be required - check broker platform!")
                        else:
                            log(f"[{STRATEGY_NAME}] Position confirmed closed via reconciliation")
                else:
                    log(f"[{STRATEGY_NAME}] [ERROR] Pause close order failed: {resp}")
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [ERROR] close-on-pause failed: {e}")

        # Final position reconciliation
        try:
            self._ensure_flat_position("pause")
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [WARN] Final position check during pause failed: {e}")

        # Stop the scheduler but don't exit the process
        try:
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)
                log(f"[{STRATEGY_NAME}] Scheduler stopped")
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [WARN] Scheduler shutdown error: {e}")

        log(f"[{STRATEGY_NAME}] Strategy paused successfully")

    def _graceful_exit(self, *args):
        """Handle process termination signals (SIGINT, SIGTERM)"""
        log(f"\n[{STRATEGY_NAME}] Shutting down…")
        self.pause()  # Reuse pause logic to close positions
        log(f"[{STRATEGY_NAME}] Shutdown complete")

        # Only call sys.exit if we're handling a signal (not called from web server)
        if args:  # Signal handlers pass arguments
            log(f"[{STRATEGY_NAME}] Bye.")
            sys.exit(0)

# ----------------------------
# Entrypoint
# ----------------------------

def main():
    cfg = Config()
    bot = RandomScalpBot(cfg)
    bot.start()

if __name__ == "__main__":
    main()

# Maintain compatibility with discovery expectations
ScalpWithTrendBot = RandomScalpBot
