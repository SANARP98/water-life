#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalp-with-Trend — Multi‑Bar Hold (Intraday Square‑Off)
Single-file OpenAlgo live trading bot (IST, 5‑minute bars)

Requirements:
- OpenAlgo Python SDK (client + ta)
- APScheduler
- pandas, numpy, pytz

Notes:
- Uses openalgo.ta for EMA/ATR
- Schedules entries at NEXT bar open; holds until TP/SL or EOD square‑off
- Streams LTP/Quotes/Depth and prints them immediately
- No DB writes/logs by default (prints only)

Community & Docs:
- https://openalgo.in/discord
- https://docs.openalgo.in
"""

from __future__ import annotations
import os
import sys
import time
import json
import math
import signal
from dataclasses import dataclass, asdict
from datetime import datetime, date, time as dtime, timedelta
from typing import Optional, List, Tuple, Dict
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env

import pytz
import pandas as pd
import numpy as np

# Scheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# === OpenAlgo SDK imports ===
try:
    import openalgo
except Exception as e:
    print("[FATAL] openalgo library is required. Please install and configure your API keys.")
    raise

IST = pytz.timezone("Asia/Kolkata")

# ----------------------------
# Strategy Metadata (for auto-discovery)
# ----------------------------
STRATEGY_METADATA = {
    "name": "Scalping v1 (Original)",
    "description": "Original scalping strategy with hardcoded lot sizes and trade direction control",
    "version": "1.0",
    "features": ["EMA Trend Following", "ATR Filter", "OCO Orders", "Trade Direction Control"],
    "has_trade_direction": True,
    "author": "OpenAlgo Community"
}

# Index lot sizes (May 2025)
INDEX_LOT_SIZES = {
    # NSE Index
    "NIFTY": 75,
    "NIFTYNXT50": 25,
    "FINNIFTY": 65,
    "BANKNIFTY": 35,
    "MIDCPNIFTY": 140,
    # BSE Index
    "SENSEX": 20,
    "BANKEX": 30,
    "SENSEX50": 60,
}

# ----------------------------
# Strategy Configuration
# ----------------------------
@dataclass
class Config:
    # Connection
    api_key: str = os.environ.get("OPENALGO_API_KEY", "")
    api_host: str = os.environ.get("OPENALGO_API_HOST", "https://api.openalgo.in")
    ws_url: Optional[str] = os.environ.get("OPENALGO_WS_URL")

    # Instrument
    symbol: str = os.environ.get("SYMBOL", "NIFTY")   # Logical name; we resolve to proper OpenAlgo symbol
    exchange: str = os.environ.get("EXCHANGE", "NSE_INDEX")
    product: str = os.environ.get("PRODUCT", "MIS")    # MIS intraday

    # Quantity controls
    lots: int = int(os.environ.get("LOTS", 2))         # user lots

    # Bar/Session
    interval: str = os.environ.get("INTERVAL", "5m")   # 5‑minute
    session_windows: Tuple[Tuple[int, int, int, int], ...] = (
        (9, 20, 11, 0),   # 09:20–11:00
        (11, 15, 15, 5),  # 11:15–15:05
    )

    # Indicators
    ema_fast: int = int(os.environ.get("EMA_FAST", 5))
    ema_slow: int = int(os.environ.get("EMA_SLOW", 20))
    atr_window: int = int(os.environ.get("ATR_WINDOW", 14))
    atr_min_points: float = float(os.environ.get("ATR_MIN_POINTS", 2.0))

    # Risk/Reward
    target_points: float = float(os.environ.get("TARGET_POINTS", 10.0))
    stoploss_points: float = float(os.environ.get("STOPLOSS_POINTS", 2.0))
    confirm_trend_at_entry: bool = os.environ.get("CONFIRM_TREND_AT_ENTRY", "true").lower() == "true"
    trade_direction: str = os.environ.get("TRADE_DIRECTION", "both").lower()  # "long", "short", or "both"

    # Daily risk cap (₹)
    daily_loss_cap: float = float(os.environ.get("DAILY_LOSS_CAP", -1000.0))

    # EOD Square-off
    enable_eod_square_off: bool = os.environ.get("ENABLE_EOD_SQUARE_OFF", "true").lower() == "true"
    square_off_time: Tuple[int, int] = (15, 25)  # 15:25 IST

    # History warmup window (days)
    warmup_days: int = int(os.environ.get("WARMUP_DAYS", 10))

    # Costs (for reporting only; NOT used for routing)
    brokerage_per_trade: float = float(os.environ.get("BROKERAGE_PER_TRADE", 20.0))  # per leg
    slippage_points: float = float(os.environ.get("SLIPPAGE_POINTS", 0.10))           # per leg

    # Start/End for history (explicit dates)
    history_start_date: Optional[str] = os.environ.get("HISTORY_START_DATE")  # YYYY-MM-DD
    history_end_date: Optional[str] = os.environ.get("HISTORY_END_DATE")      # YYYY-MM-DD


# ----------------------------
# Utilities
# ----------------------------

def now_ist() -> datetime:
    return datetime.now(IST)


def ist_datetime(y: int, m: int, d: int, hh: int, mm: int, ss: int = 0) -> datetime:
    return IST.localize(datetime(y, m, d, hh, mm, ss))


def in_session(t: dtime, windows: Tuple[Tuple[int, int, int, int], ...]) -> bool:
    for (h1, m1, h2, m2) in windows:
        if (t >= dtime(h1, m1)) and (t <= dtime(h2, m2)):
            return True
    return False


def is_square_off_time(cfg: Config, t: dtime) -> bool:
    sh, sm = cfg.square_off_time
    return (t.hour, t.minute) >= (sh, sm)


def resolve_quantity(cfg: Config) -> int:
    base = INDEX_LOT_SIZES.get(cfg.symbol.upper())
    if base is None:
        raise ValueError(f"Lot size for symbol {cfg.symbol} not in predefined table. Please edit INDEX_LOT_SIZES.")
    qty = cfg.lots * base
    return int(qty)


def compute_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Assumes df columns: ['timestamp','open','high','low','close','volume']
    out = df.copy()
    out["ema_fast"] = openalgo.ta.ema(out["close"].values, cfg.ema_fast)
    out["ema_slow"] = openalgo.ta.ema(out["close"].values, cfg.ema_slow)
    out["atr"] = openalgo.ta.atr(out[["high", "low", "close"]].values, cfg.atr_window)
    return out


def last_complete_bar(df: pd.DataFrame) -> int:
    # When we fetch history at bar-close, the last row is the just-closed bar
    return len(df) - 1


def get_history(client, cfg: Config) -> pd.DataFrame:
    """Fetch recent history (uses explicit start_date/end_date controls as required)."""
    today = now_ist().date()
    if cfg.history_start_date:
        start_date = cfg.history_start_date
    else:
        start_date = (today - timedelta(days=cfg.warmup_days)).strftime("%Y-%m-%d")

    end_date = cfg.history_end_date or today.strftime("%Y-%m-%d")

    # OpenAlgo history expected to return pandas-like data (per docs output is DataFrame)
    df = client.history(
        symbol=cfg.symbol,
        exchange=cfg.exchange,
        interval=cfg.interval,
        start_date=start_date,
        end_date=end_date,
    )

    # Standardize columns/ensure timestamp tz-aware IST
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("Asia/Kolkata") \
            if pd.api.types.is_datetime64tz_dtype(df["timestamp"]) else pd.to_datetime(df["timestamp"]).dt.tz_localize("Asia/Kolkata")
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        raise ValueError("history() must return a DataFrame with 'timestamp' column")

    # Ensure required columns in correct order
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"history() missing required columns: {missing}")

    return df[cols]


# ----------------------------
# Trading Engine
# ----------------------------
class ScalpWithTrendBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = openalgo.api(api_key=cfg.api_key, host=cfg.api_host, ws_url=cfg.ws_url)
        self.scheduler = BackgroundScheduler(timezone=IST)

        # Derived
        self.qty = resolve_quantity(cfg)

        # State
        self.pending_signal: Optional[str] = None  # 'LONG'/'SHORT'
        self.pending_signal_time: Optional[datetime] = None  # bar close time that produced signal
        self.next_entry_time: Optional[datetime] = None      # next bar open (when to enter)

        self.in_position: bool = False
        self.side: Optional[str] = None  # 'LONG'/'SHORT'
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[datetime] = None
        self.tp_level: Optional[float] = None
        self.sl_level: Optional[float] = None

        self.entry_order_id: Optional[str] = None
        self.tp_order_id: Optional[str] = None
        self.sl_order_id: Optional[str] = None

        # P&L tracking
        self.realized_pnl_today: float = 0.0
        self.today: date = now_ist().date()

    # ---------------- WebSocket handlers ----------------
    def on_ltp(self, data: Dict):
        # Print quotes immediately (as per requirement)
        ts = now_ist().strftime('%Y-%m-%d %H:%M:%S')
        ltp = data.get('ltp')
        print(f"[LTP] {ts} {self.cfg.symbol}@{self.cfg.exchange} LTP: {ltp}")

    def on_quotes(self, data: Dict):
        ts = now_ist().strftime('%Y-%m-%d %H:%M:%S')
        best_bid = data.get('best_bid'); best_ask = data.get('best_ask')
        print(f"[QUOTE] {ts} {self.cfg.symbol}@{self.cfg.exchange} bid: {best_bid} ask: {best_ask}")

    def on_depth(self, data: Dict):
        ts = now_ist().strftime('%Y-%m-%d %H:%M:%S')
        depth = data.get('depth')
        # Print a compact snapshot (top 3 levels if available)
        if isinstance(depth, dict):
            bids = depth.get('bids', [])[:3]; asks = depth.get('asks', [])[:3]
            print(f"[DEPTH] {ts} bids:{bids} asks:{asks}")
        else:
            print(f"[DEPTH] {ts} {depth}")

    def on_order_update(self, data: Dict):
        # Handle order events to manage OCO (cancel the other leg on fill) and compute realized P&L
        oid = data.get('order_id'); status = data.get('status'); filled_avg = data.get('average_price')
        side = data.get('transaction_type')  # BUY/SELL

        if oid == self.entry_order_id and status in ("COMPLETE", "FILLED"):
            print(f"[ORDER] Entry filled @ {filled_avg}")
            # Immediately place TP/SL legs
            self.place_exit_legs()

        # TP leg filled
        if self.tp_order_id and oid == self.tp_order_id and status in ("COMPLETE", "FILLED"):
            print(f"[ORDER] TP filled @ {filled_avg}")
            self._realize_exit(price=float(filled_avg), reason="Target Hit")
            self.cancel_order_silent(self.sl_order_id)

        # SL leg filled
        if self.sl_order_id and oid == self.sl_order_id and status in ("COMPLETE", "FILLED"):
            print(f"[ORDER] SL filled @ {filled_avg}")
            self._realize_exit(price=float(filled_avg), reason="Stoploss Hit")
            self.cancel_order_silent(self.tp_order_id)

    # ---------------- Lifecycle ----------------
    def start(self):
        print("\n🔁 OpenAlgo Python Bot is running.\n")
        print("Config:", json.dumps(asdict(self.cfg), indent=2))
        print(f"Resolved quantity: {self.qty}")

        # Connect WS for LTP/quotes/depth and order updates
        self.client.ws_connect(on_ltp=self.on_ltp, on_quotes=self.on_quotes, on_depth=self.on_depth, on_order=self.on_order_update)
        self.client.ws_subscribe(symbol=self.cfg.symbol, exchange=self.cfg.exchange, channels=["ltp","quotes","depth","orders"])  # prints will occur in handlers

        # Schedule bar-close (compute signals) every 5 minutes inside sessions
        self.scheduler.add_job(self.on_bar_close_tick, CronTrigger(minute="*/5", second=2, timezone=IST), id="bar_close")
        # Schedule bar-open (place entries) every 5 minutes
        self.scheduler.add_job(self.on_bar_open_tick, CronTrigger(minute="*/5", second=5, timezone=IST), id="bar_open")

        # Square-off job
        if self.cfg.enable_eod_square_off:
            soh, som = self.cfg.square_off_time
            self.scheduler.add_job(self.square_off_job, CronTrigger(hour=soh, minute=som, second=30, timezone=IST), id="square_off")

        self.scheduler.start()

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._graceful_exit)
        signal.signal(signal.SIGTERM, self._graceful_exit)

        # Keep alive
        try:
            while True:
                time.sleep(1)
                # Reset daily P&L at day change
                if now_ist().date() != self.today:
                    self.today = now_ist().date()
                    self.realized_pnl_today = 0.0
                    print(f"[DAY ROLLOVER] Reset daily P&L. New day: {self.today}")
        finally:
            self.scheduler.shutdown(wait=False)
            self.client.ws_disconnect()

    # ---------------- Jobs ----------------
    def on_bar_close_tick(self):
        now = now_ist()
        if not in_session(now.time(), self.cfg.session_windows):
            return
        try:
            df = get_history(self.client, self.cfg)
            df = compute_indicators(df, self.cfg)
            i = last_complete_bar(df)
            if i < 1:
                return
            prev, cur = df.iloc[i-1], df.iloc[i]

            # ATR filter
            if float(cur["atr"]) < self.cfg.atr_min_points:
                return

            # Determine trend
            trend_up = cur["ema_fast"] > cur["ema_slow"]
            trend_down = cur["ema_fast"] < cur["ema_slow"]

            long_sig = (cur["high"] > prev["high"]) and trend_up
            short_sig = (cur["low"] < prev["low"]) and trend_down

            # Filter by trade direction setting
            if self.cfg.trade_direction == "long":
                short_sig = False
            elif self.cfg.trade_direction == "short":
                long_sig = False

            if self.in_position:
                return

            if long_sig or short_sig:
                # Optional conservative check (same bar, already using cur)
                if self.cfg.confirm_trend_at_entry:
                    if long_sig and not trend_up:
                        return
                    if short_sig and not trend_down:
                        return

                # Schedule entry at next bar open (cur.timestamp + 5m)
                self.pending_signal = 'LONG' if long_sig else 'SHORT'
                self.pending_signal_time = cur["timestamp"]
                self.next_entry_time = (cur["timestamp"] + pd.Timedelta(minutes=5)).to_pydatetime()
                print(f"[SIGNAL] {self.pending_signal} at bar close {self.pending_signal_time}. Next entry at {self.next_entry_time} IST")
        except Exception as e:
            print(f"[ERROR] on_bar_close_tick: {e}")

    def on_bar_open_tick(self):
        now = now_ist()
        if not in_session(now.time(), self.cfg.session_windows):
            return
        if self.pending_signal and self.next_entry_time:
            # Allow small tolerance window (+/- 10s)
            delta = abs((now - self.next_entry_time).total_seconds())
            if delta <= 10 and (not self.in_position):
                # Daily loss cap check
                if self.realized_pnl_today <= self.cfg.daily_loss_cap:
                    print("[RISK] Daily loss cap breached. Skipping new entries today.")
                    self._clear_pending_signal()
                    return
                self.place_entry(self.pending_signal)
                self._clear_pending_signal()

    def square_off_job(self):
        now = now_ist()
        if not self.in_position:
            return
        if is_square_off_time(self.cfg, now.time()):
            print(f"[EOD] Square-off time reached ({self.cfg.square_off_time[0]:02d}:{self.cfg.square_off_time[1]:02d}). Exiting position.")
            # Cancel exit legs and market out
            self.cancel_order_silent(self.tp_order_id)
            self.cancel_order_silent(self.sl_order_id)
            try:
                action = "SELL" if self.side == 'LONG' else "BUY"
                resp = self.client.placeorder(symbol=self.cfg.symbol, exchange=self.cfg.exchange, product=self.cfg.product,
                                              transaction_type=action, order_type="MARKET", quantity=self.qty)
                px = float(resp.get('average_price') or resp.get('ltp') or 0.0)
                self._realize_exit(price=px, reason="Square-off EOD")
            except Exception as e:
                print(f"[ERROR] EOD square-off failed: {e}")

    # ---------------- Order Helpers ----------------
    def place_entry(self, side: str):
        assert side in ("LONG","SHORT")
        try:
            action = "BUY" if side == 'LONG' else "SELL"
            print(f"[ENTRY] {side} sending MARKET for {self.qty} {self.cfg.symbol}@{self.cfg.exchange}")
            resp = self.client.placeorder(symbol=self.cfg.symbol, exchange=self.cfg.exchange, product=self.cfg.product,
                                          transaction_type=action, order_type="MARKET", quantity=self.qty)
            self.entry_order_id = resp.get('order_id')
            avg = resp.get('average_price')
            ltp = resp.get('ltp')
            self.entry_price = float(avg or ltp)
            self.entry_time = now_ist()
            self.in_position = True
            self.side = side
            # Prepare TP/SL levels
            if side == 'LONG':
                self.tp_level = self.entry_price + self.cfg.target_points
                self.sl_level = self.entry_price - self.cfg.stoploss_points
            else:
                self.tp_level = self.entry_price - self.cfg.target_points
                self.sl_level = self.entry_price + self.cfg.stoploss_points
            print(f"[ENTRY] Filled ~{self.entry_price:.2f}. TP={self.tp_level:.2f} SL={self.sl_level:.2f}")
            # Place exit legs immediately if not relying on async order update
            self.place_exit_legs()
        except Exception as e:
            print(f"[ERROR] place_entry: {e}")
            self._flat_state()

    def place_exit_legs(self):
        if not self.in_position or self.entry_price is None:
            return
        try:
            if self.side == 'LONG':
                # TP: SELL LIMIT @ tp_level, SL: SELL SL-M @ sl_level
                tp_resp = self.client.placeorder(symbol=self.cfg.symbol, exchange=self.cfg.exchange, product=self.cfg.product,
                                                 transaction_type="SELL", order_type="LIMIT", price=self.tp_level, quantity=self.qty)
                sl_resp = self.client.placeorder(symbol=self.cfg.symbol, exchange=self.cfg.exchange, product=self.cfg.product,
                                                 transaction_type="SELL", order_type="SL-M", trigger_price=self.sl_level, quantity=self.qty)
            else:
                # SHORT: TP BUY LIMIT @ tp_level, SL BUY SL-M @ sl_level
                tp_resp = self.client.placeorder(symbol=self.cfg.symbol, exchange=self.cfg.exchange, product=self.cfg.product,
                                                 transaction_type="BUY", order_type="LIMIT", price=self.tp_level, quantity=self.qty)
                sl_resp = self.client.placeorder(symbol=self.cfg.symbol, exchange=self.cfg.exchange, product=self.cfg.product,
                                                 transaction_type="BUY", order_type="SL-M", trigger_price=self.sl_level, quantity=self.qty)
            self.tp_order_id = tp_resp.get('order_id')
            self.sl_order_id = sl_resp.get('order_id')
            print(f"[EXITS] Placed TP oid={self.tp_order_id} @ {self.tp_level:.2f} and SL oid={self.sl_order_id} @ {self.sl_level:.2f}")
        except Exception as e:
            print(f"[ERROR] place_exit_legs: {e}")

    def cancel_order_silent(self, order_id: Optional[str]):
        if not order_id:
            return
        try:
            self.client.cancelorder(order_id=order_id)
            print(f"[CANCEL] order_id={order_id} cancelled")
        except Exception as e:
            print(f"[WARN] cancel_order_silent failed for {order_id}: {e}")

    # ---------------- P&L Helpers ----------------
    def _realize_exit(self, price: float, reason: str):
        if not self.in_position or self.entry_price is None:
            return
        points = (price - self.entry_price) if self.side == 'LONG' else (self.entry_price - price)
        gross = points * self.qty  # qty already accounts for lot multiplier (₹ per point)
        costs = 2 * self.cfg.brokerage_per_trade + 2 * self.cfg.slippage_points * self.qty
        net = gross - costs
        self.realized_pnl_today += net
        print(f"[EXIT] {reason} @ {price:.2f} | points={points:.2f} gross={gross:.2f} costs={costs:.2f} net={net:.2f} | dailyPnL={self.realized_pnl_today:.2f}")
        self._flat_state()

    def _flat_state(self):
        self.in_position = False
        self.side = None
        self.entry_price = None
        self.entry_time = None
        self.tp_level = None
        self.sl_level = None
        self.entry_order_id = None
        self.tp_order_id = None
        self.sl_order_id = None

    def _clear_pending_signal(self):
        self.pending_signal = None
        self.pending_signal_time = None
        self.next_entry_time = None

    # ---------------- Shutdown ----------------
    def _graceful_exit(self, *args):
        print("\n[SHUTDOWN] Closing connections and scheduler...")
        try:
            if self.in_position:
                print("[SHUTDOWN] Position open. Attempting market exit...")
                action = "SELL" if self.side == 'LONG' else "BUY"
                self.client.placeorder(symbol=self.cfg.symbol, exchange=self.cfg.exchange, product=self.cfg.product,
                                       transaction_type=action, order_type="MARKET", quantity=self.qty)
        except Exception as e:
            print(f"[WARN] Shutdown exit failed: {e}")
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
            self.client.ws_disconnect()
        except Exception:
            pass
        print("[SHUTDOWN] Done. Bye.")
        sys.exit(0)


# ----------------------------
# Main Entrypoint
# ----------------------------

def main():
    cfg = Config()
    if not cfg.api_key:
        print("[FATAL] Please set OPENALGO_API_KEY in environment.")
        sys.exit(1)

    if cfg.trade_direction not in ["long", "short", "both"]:
        print(f"[FATAL] TRADE_DIRECTION must be 'long', 'short', or 'both'")
        sys.exit(1)

    # Validate symbol format as per OpenAlgo conventions (basic check)
    sym = cfg.symbol.upper()
    if sym not in INDEX_LOT_SIZES:
        print(f"[WARN] {sym} not in predefined index lot table. Ensure symbol follows OpenAlgo Symbol Format.")

    bot = ScalpWithTrendBot(cfg)
    bot.start()


if __name__ == "__main__":
    main()
