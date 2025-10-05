#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalp-with-Trend — Multi‑Bar Hold (Intraday Square‑Off)
Single-file OpenAlgo live trading bot (IST, 5‑minute bars) — Corrected Version

Features:
- strategy="scalp_with_trend" in all orders
- Symbol validation via OpenAlgo API
- Dynamic lot size resolution
- Logs prefixed with strategy name
- Safer OCO handling with order status polling
- Corrected parameter names and API methods

Docs:
- https://openalgo.in/discord
- https://docs.openalgo.in
"""

from __future__ import annotations
import os, sys, time, json, signal
from dataclasses import dataclass, asdict
from datetime import datetime, date, time as dtime, timedelta
from typing import Optional, Tuple, Dict
from dotenv import load_dotenv
import pytz, pandas as pd, numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Load env vars
load_dotenv()

# OpenAlgo client
try:
    import openalgo
except Exception:
    print("[FATAL] openalgo library is required. Please install and configure your API keys.")
    raise

IST = pytz.timezone("Asia/Kolkata")
STRATEGY_NAME = "scalp_with_trend"

# ----------------------------
# Strategy Metadata (for auto-discovery)
# ----------------------------
STRATEGY_METADATA = {
    "name": "Scalping v2 Claude (Corrected)",
    "description": "Claude-corrected version with enhanced error handling and fixes",
    "version": "2.1",
    "features": ["EMA Trend Following", "ATR Filter", "OCO Orders", "Trade Direction Control", "Dynamic Lot Sizes", "Enhanced Error Handling"],
    "has_trade_direction": True,
    "author": "OpenAlgo Community + Claude"
}

# ----------------------------
# Strategy Configuration
# ----------------------------
@dataclass
class Config:
    api_key: str = os.getenv("OPENALGO_API_KEY", "")
    api_host: str = os.getenv("OPENALGO_API_HOST", "http://127.0.0.1:5000")
    ws_url: Optional[str] = os.getenv("OPENALGO_WS_URL", "ws://127.0.0.1:8765")
    symbol: str = os.getenv("SYMBOL", "NIFTY")
    exchange: str = os.getenv("EXCHANGE", "NFO")
    product: str = os.getenv("PRODUCT", "MIS")
    lots: int = int(os.getenv("LOTS", 1))
    interval: str = os.getenv("INTERVAL", "5m")
    session_windows: Tuple[Tuple[int, int, int, int], ...] = (
        (9, 20, 11, 0),
        (11, 15, 15, 5),
    )
    ema_fast: int = int(os.getenv("EMA_FAST", 5))
    ema_slow: int = int(os.getenv("EMA_SLOW", 20))
    atr_window: int = int(os.getenv("ATR_WINDOW", 14))
    atr_min_points: float = float(os.getenv("ATR_MIN_POINTS", 2.0))
    target_points: float = float(os.getenv("TARGET_POINTS", 10.0))
    stoploss_points: float = float(os.getenv("STOPLOSS_POINTS", 2.0))
    confirm_trend_at_entry: bool = os.getenv("CONFIRM_TREND_AT_ENTRY", "true").lower() == "true"
    trade_direction: str = os.getenv("TRADE_DIRECTION", "both").lower()  # "long", "short", or "both"
    daily_loss_cap: float = float(os.getenv("DAILY_LOSS_CAP", -1000.0))
    enable_eod_square_off: bool = os.getenv("ENABLE_EOD_SQUARE_OFF", "true").lower() == "true"
    square_off_time: Tuple[int, int] = (15, 25)
    warmup_days: int = int(os.getenv("WARMUP_DAYS", 10))
    brokerage_per_trade: float = float(os.getenv("BROKERAGE_PER_TRADE", 20.0))
    slippage_points: float = float(os.getenv("SLIPPAGE_POINTS", 0.10))
    history_start_date: Optional[str] = os.getenv("HISTORY_START_DATE")
    history_end_date: Optional[str] = os.getenv("HISTORY_END_DATE")
    sl_order_type: str = os.getenv("SL_ORDER_TYPE", "SL-M")  # or "SL" depending on broker

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

def validate_symbol(client, symbol: str, exchange: str) -> bool:
    """Validate symbol using search API"""
    try:
        result = client.search(query=symbol, exchange=exchange)
        if result.get('status') == 'success':
            symbols = [s['symbol'].upper() for s in result.get('data', [])]
            return symbol.upper() in symbols
        return False
    except Exception as e:
        print(f"[{STRATEGY_NAME}] [WARN] Symbol validation failed: {e}")
        return True  # Proceed with caution

def resolve_quantity(client, cfg: Config) -> int:
    """Resolve quantity using search API"""
    try:
        result = client.search(query=cfg.symbol, exchange=cfg.exchange)
        if result.get('status') == 'success':
            for inst in result.get('data', []):
                if inst['symbol'].upper() == cfg.symbol.upper():
                    lot = int(inst.get("lotsize") or 1)
                    return cfg.lots * lot
    except Exception as e:
        print(f"[{STRATEGY_NAME}] [ERROR] resolve_quantity: {e}")
    raise ValueError(f"[{STRATEGY_NAME}] Lot size for {cfg.symbol} not found")

def compute_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = openalgo.ta.ema(out["close"].values, cfg.ema_fast)
    out["ema_slow"] = openalgo.ta.ema(out["close"].values, cfg.ema_slow)
    out["atr"] = openalgo.ta.atr(out[["high", "low", "close"]].values, cfg.atr_window)
    return out

def get_history(client, cfg: Config) -> pd.DataFrame:
    today = now_ist().date()
    start_date = cfg.history_start_date or (today - timedelta(days=cfg.warmup_days)).strftime("%Y-%m-%d")
    end_date = cfg.history_end_date or today.strftime("%Y-%m-%d")
    df = client.history(symbol=cfg.symbol, exchange=cfg.exchange, interval=cfg.interval,
                        start_date=start_date, end_date=end_date)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("Asia/Kolkata") \
        if not pd.api.types.is_datetime64tz_dtype(df["timestamp"]) else df["timestamp"].dt.tz_convert("Asia/Kolkata")
    return df[["timestamp","open","high","low","close","volume"]].sort_values("timestamp").reset_index(drop=True)

# ----------------------------
# Trading Engine
# ----------------------------
class ScalpWithTrendBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = openalgo.api(api_key=cfg.api_key, host=cfg.api_host, ws_url=cfg.ws_url)
        self.scheduler = BackgroundScheduler(timezone=IST)
        self.qty = resolve_quantity(self.client, cfg)
        self.pending_signal = None
        self.next_entry_time = None
        self.in_position = False
        self.side = None
        self.entry_price = None
        self.tp_level = None
        self.sl_level = None
        self.entry_order_id = None
        self.tp_order_id = None
        self.sl_order_id = None
        self.realized_pnl_today = 0.0
        self.today = now_ist().date()

    # WS Handlers
    def on_ltp(self, data: Dict):
        ts = now_ist().strftime('%Y-%m-%d %H:%M:%S')
        ltp = data.get('ltp')
        print(f"[{STRATEGY_NAME}] [LTP] {ts} {self.cfg.symbol}@{self.cfg.exchange} LTP: {ltp}")

    # Lifecycle
    def start(self):
        print(f"\n🔁 {STRATEGY_NAME} Bot is running.\nConfig: {json.dumps(asdict(self.cfg), indent=2)}\nResolved quantity: {self.qty}\n")
        
        # Setup WebSocket (corrected method calls)
        try:
            instruments = [{"exchange": self.cfg.exchange, "symbol": self.cfg.symbol}]
            self.client.connect()
            self.client.subscribe_ltp(instruments, on_data_received=self.on_ltp)
            print(f"[{STRATEGY_NAME}] WebSocket connected and subscribed")
        except Exception as e:
            print(f"[{STRATEGY_NAME}] [WARN] WebSocket setup failed: {e}. Continuing without WebSocket.")
        
        # Schedule jobs
        self.scheduler.add_job(self.on_bar_close_tick, CronTrigger(minute="*/5", second=2, timezone=IST), id="bar_close")
        self.scheduler.add_job(self.on_bar_open_tick, CronTrigger(minute="*/5", second=5, timezone=IST), id="bar_open")
        self.scheduler.add_job(self.check_order_status, 'interval', seconds=2, id="order_status_check")
        
        if self.cfg.enable_eod_square_off:
            soh, som = self.cfg.square_off_time
            self.scheduler.add_job(self.square_off_job, CronTrigger(hour=soh, minute=som, second=30, timezone=IST), id="square_off")
        
        self.scheduler.start()
        signal.signal(signal.SIGINT, self._graceful_exit)
        signal.signal(signal.SIGTERM, self._graceful_exit)
        
        # Main loop
        try:
            while True:
                time.sleep(1)
                if now_ist().date() != self.today:
                    self.today = now_ist().date()
                    self.realized_pnl_today = 0.0
                    print(f"[{STRATEGY_NAME}] [DAY ROLLOVER] New day: {self.today}")
        finally:
            self._graceful_exit()

    # Jobs
    def on_bar_close_tick(self):
        if not in_session(now_ist().time(), self.cfg.session_windows):
            return
        try:
            df = compute_indicators(get_history(self.client, self.cfg), self.cfg)
            i = len(df) - 1
            if i < 1:
                return
            prev, cur = df.iloc[i-1], df.iloc[i]
            if float(cur['atr']) < self.cfg.atr_min_points:
                return
            
            trend_up = cur['ema_fast'] > cur['ema_slow']
            trend_down = cur['ema_fast'] < cur['ema_slow']
            long_sig = (cur['high'] > prev['high']) and trend_up
            short_sig = (cur['low'] < prev['low']) and trend_down

            # Filter by trade direction setting
            if self.cfg.trade_direction == "long":
                short_sig = False
            elif self.cfg.trade_direction == "short":
                long_sig = False

            if self.in_position:
                return
            if long_sig or short_sig:
                self.pending_signal = 'LONG' if long_sig else 'SHORT'
                self.next_entry_time = (cur['timestamp'] + pd.Timedelta(minutes=5)).to_pydatetime()
                print(f"[{STRATEGY_NAME}] [SIGNAL] {self.pending_signal} next entry at {self.next_entry_time}")
        except Exception as e:
            print(f"[{STRATEGY_NAME}] [ERROR] on_bar_close_tick: {e}")

    def on_bar_open_tick(self):
        if not in_session(now_ist().time(), self.cfg.session_windows):
            return
        if self.pending_signal and self.next_entry_time:
            delta = abs((now_ist() - self.next_entry_time).total_seconds())
            if delta <= 10 and not self.in_position:
                if self.realized_pnl_today <= self.cfg.daily_loss_cap:
                    print(f"[{STRATEGY_NAME}] [RISK] Daily loss cap breached.")
                    self._clear_pending_signal()
                    return
                self.place_entry(self.pending_signal)
                self._clear_pending_signal()

    def check_order_status(self):
        """Poll order status to detect fills (replaces WebSocket order updates)"""
        if not self.in_position:
            return
        
        try:
            # Check TP order
            if self.tp_order_id:
                resp = self.client.orderstatus(order_id=self.tp_order_id, strategy=STRATEGY_NAME)
                if resp.get('status') == 'success':
                    order_data = resp.get('data', {})
                    if order_data.get('order_status') in ('complete', 'COMPLETE'):
                        price = float(order_data.get('average_price', 0))
                        if price > 0:
                            self._realize_exit(price, "Target Hit")
                            self.cancel_order_silent(self.sl_order_id)
                            return
            
            # Check SL order
            if self.sl_order_id:
                resp = self.client.orderstatus(order_id=self.sl_order_id, strategy=STRATEGY_NAME)
                if resp.get('status') == 'success':
                    order_data = resp.get('data', {})
                    if order_data.get('order_status') in ('complete', 'COMPLETE'):
                        price = float(order_data.get('average_price', 0))
                        if price > 0:
                            self._realize_exit(price, "Stoploss Hit")
                            self.cancel_order_silent(self.tp_order_id)
        except Exception as e:
            print(f"[{STRATEGY_NAME}] [ERROR] check_order_status: {e}")

    def square_off_job(self):
        if not self.in_position:
            return
        if is_square_off_time(self.cfg, now_ist().time()):
            print(f"[{STRATEGY_NAME}] [EOD] Square-off.")
            self.cancel_order_silent(self.tp_order_id)
            self.cancel_order_silent(self.sl_order_id)
            
            try:
                action = "SELL" if self.side == 'LONG' else 'BUY'
                resp = self.client.placeorder(
                    strategy=STRATEGY_NAME,
                    symbol=self.cfg.symbol,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action=action,  # CORRECTED: was transaction_type
                    price_type="MARKET",  # CORRECTED: was order_type
                    quantity=self.qty
                )
                
                # Get filled price from order status
                if resp.get('status') == 'success':
                    time.sleep(0.5)  # Brief delay for order execution
                    status_resp = self.client.orderstatus(order_id=resp.get('orderid'), strategy=STRATEGY_NAME)
                    px = float(status_resp.get('data', {}).get('average_price', 0))
                    self._realize_exit(px, "Square-off EOD")
            except Exception as e:
                print(f"[{STRATEGY_NAME}] [ERROR] square_off_job: {e}")

    # Orders
    def place_entry(self, side: str):
        try:
            action = "BUY" if side == 'LONG' else 'SELL'
            resp = self.client.placeorder(
                strategy=STRATEGY_NAME,
                symbol=self.cfg.symbol,
                exchange=self.cfg.exchange,
                product=self.cfg.product,
                action=action,  # CORRECTED: was transaction_type
                price_type="MARKET",  # CORRECTED: was order_type
                quantity=self.qty
            )
            
            if resp.get('status') != 'success':
                print(f"[{STRATEGY_NAME}] [ERROR] Order failed: {resp}")
                self._flat_state()
                return
            
            self.entry_order_id = resp.get('orderid')  # CORRECTED: was order_id
            
            # Get filled price from order status
            time.sleep(0.5)  # Brief delay for order execution
            status_resp = self.client.orderstatus(order_id=self.entry_order_id, strategy=STRATEGY_NAME)
            self.entry_price = float(status_resp.get('data', {}).get('average_price', 0))
            
            if self.entry_price == 0:
                print(f"[{STRATEGY_NAME}] [WARN] Could not get entry price, waiting for fill...")
                return
            
            self.in_position = True
            self.side = side
            self.tp_level = self.entry_price + (self.cfg.target_points if side == 'LONG' else -self.cfg.target_points)
            self.sl_level = self.entry_price - (self.cfg.stoploss_points if side == 'LONG' else -self.cfg.stoploss_points)
            print(f"[{STRATEGY_NAME}] [ENTRY] {side} Filled ~{self.entry_price:.2f} TP={self.tp_level:.2f} SL={self.sl_level:.2f}")
            self.place_exit_legs()
        except Exception as e:
            print(f"[{STRATEGY_NAME}] [ERROR] place_entry: {e}")
            self._flat_state()

    def place_exit_legs(self):
        if not self.in_position or self.entry_price is None:
            return
        try:
            if self.side == 'LONG':
                tp_resp = self.client.placeorder(
                    strategy=STRATEGY_NAME,
                    symbol=self.cfg.symbol,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action="SELL",  # CORRECTED: was transaction_type
                    price_type="LIMIT",  # CORRECTED: was order_type
                    price=self.tp_level,
                    quantity=self.qty
                )
                sl_resp = self.client.placeorder(
                    strategy=STRATEGY_NAME,
                    symbol=self.cfg.symbol,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action="SELL",  # CORRECTED: was transaction_type
                    price_type=self.cfg.sl_order_type,  # CORRECTED: was order_type
                    trigger_price=self.sl_level,
                    quantity=self.qty
                )
            else:
                tp_resp = self.client.placeorder(
                    strategy=STRATEGY_NAME,
                    symbol=self.cfg.symbol,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action="BUY",  # CORRECTED: was transaction_type
                    price_type="LIMIT",  # CORRECTED: was order_type
                    price=self.tp_level,
                    quantity=self.qty
                )
                sl_resp = self.client.placeorder(
                    strategy=STRATEGY_NAME,
                    symbol=self.cfg.symbol,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action="BUY",  # CORRECTED: was transaction_type
                    price_type=self.cfg.sl_order_type,  # CORRECTED: was order_type
                    trigger_price=self.sl_level,
                    quantity=self.qty
                )
            
            self.tp_order_id = tp_resp.get('orderid')  # CORRECTED: was order_id
            self.sl_order_id = sl_resp.get('orderid')  # CORRECTED: was order_id
            print(f"[{STRATEGY_NAME}] [EXITS] TP oid={self.tp_order_id} SL oid={self.sl_order_id}")
        except Exception as e:
            print(f"[{STRATEGY_NAME}] [ERROR] place_exit_legs: {e}")

    def cancel_order_silent(self, order_id: Optional[str]):
        if not order_id:
            return
        try:
            self.client.cancelorder(order_id=order_id)
            print(f"[{STRATEGY_NAME}] [CANCEL] order_id={order_id}")
        except Exception as e:
            print(f"[{STRATEGY_NAME}] [WARN] cancel_order_silent: {e}")

    def _realize_exit(self, price: float, reason: str):
        if not self.in_position or self.entry_price is None:
            return
        points = (price - self.entry_price) if self.side == 'LONG' else (self.entry_price - price)
        gross = points * self.qty
        costs = 2 * self.cfg.brokerage_per_trade + 2 * self.cfg.slippage_points * self.qty
        net = gross - costs
        self.realized_pnl_today += net
        print(f"[{STRATEGY_NAME}] [EXIT] {reason} @ {price:.2f} | pts={points:.2f} gross={gross:.2f} net={net:.2f} | dailyPnL={self.realized_pnl_today:.2f}")
        self._flat_state()

    def _flat_state(self):
        self.in_position = False
        self.side = None
        self.entry_price = None
        self.tp_level = None
        self.sl_level = None
        self.entry_order_id = None
        self.tp_order_id = None
        self.sl_order_id = None

    def _clear_pending_signal(self):
        self.pending_signal = None
        self.next_entry_time = None

    def _graceful_exit(self, *args):
        print(f"\n[{STRATEGY_NAME}] [SHUTDOWN] Closing...")
        try:
            if self.in_position:
                action = 'SELL' if self.side == 'LONG' else 'BUY'
                self.client.placeorder(
                    strategy=STRATEGY_NAME,
                    symbol=self.cfg.symbol,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action=action,  # CORRECTED: was transaction_type
                    price_type="MARKET",  # CORRECTED: was order_type
                    quantity=self.qty
                )
        except Exception as e:
            print(f"[{STRATEGY_NAME}] [WARN] Shutdown exit failed: {e}")
        
        try:
            self.cancel_order_silent(self.tp_order_id)
            self.cancel_order_silent(self.sl_order_id)
        except Exception:
            pass
        
        try:
            self.scheduler.shutdown(wait=False)
        except Exception:
            pass
        
        try:
            self.client.disconnect()  # CORRECTED: was ws_disconnect
        except Exception:
            pass
        
        print(f"[{STRATEGY_NAME}] [SHUTDOWN] Done.")
        sys.exit(0)

# ----------------------------
# Main Entrypoint
# ----------------------------
def main():
    cfg = Config()
    if not cfg.api_key:
        print(f"[{STRATEGY_NAME}] [FATAL] Please set OPENALGO_API_KEY")
        sys.exit(1)
    if cfg.trade_direction not in ["long", "short", "both"]:
        print(f"[{STRATEGY_NAME}] [FATAL] TRADE_DIRECTION must be 'long', 'short', or 'both'")
        sys.exit(1)
    
    client = openalgo.api(api_key=cfg.api_key, host=cfg.api_host, ws_url=cfg.ws_url)
    if not validate_symbol(client, cfg.symbol, cfg.exchange):
        print(f"[{STRATEGY_NAME}] [FATAL] Symbol {cfg.symbol} not valid on {cfg.exchange}")
        sys.exit(1)
    
    bot = ScalpWithTrendBot(cfg)
    bot.start()

if __name__ == "__main__":
    main()