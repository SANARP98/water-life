#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalp-with-Trend — Multi-Bar Hold (Intraday Square-Off)
Single-file OpenAlgo live trading bot (IST, 5-minute bars) — Final Version

Features:
- strategy="scalp_with_trend" in all orders
- Symbol validation via OpenAlgo API
- Dynamic lot size resolution
- Logs prefixed with strategy name
- Safer OCO handling with order status polling
- Corrected parameter names and API methods
- Log to file (optional)
- Position/PNL persistence (JSON file)
- Live Equity Option ATM Selector (CE for LONG, PE for SHORT)
- Test mode enabled via environment

Docs:
- https://openalgo.in/discord
- https://docs.openalgo.in
"""

from __future__ import annotations
import os, sys, time, json, signal, logging
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
# Strategy Configuration
# ----------------------------
@dataclass
class Config:
    api_key: str = os.getenv("OPENALGO_API_KEY", "")
    api_host: str = os.getenv("OPENALGO_API_HOST", "http://127.0.0.1:5000")
    ws_url: Optional[str] = os.getenv("OPENALGO_WS_URL", "ws://127.0.0.1:8765")

    # Instrument selection
    symbol: str = os.getenv("SYMBOL", "NIFTY")         # Can be index (NIFTY/BANKNIFTY/FINNIFTY) or an exact tradable symbol
    exchange: str = os.getenv("EXCHANGE", "NFO")       # Futures/Options exchange
    product: str = os.getenv("PRODUCT", "MIS")         # MIS (intraday)
    lots: int = int(os.getenv("LOTS", 1))              # Lot multiplier

    # Engine
    interval: str = os.getenv("INTERVAL", "5m")
    log_to_file: bool = os.getenv("LOG_TO_FILE", "false").lower() == "true"
    test_mode: bool = os.getenv("TEST_MODE", "false").lower() == "true"
    persist_state: bool = os.getenv("PERSIST_STATE", "true").lower() == "true"
    option_auto: bool = os.getenv("OPTION_AUTO", "true").lower() == "true"  # Auto-pick ATM CE/PE each entry if symbol is an index

    # Sessions
    session_windows: Tuple[Tuple[int, int, int, int], ...] = (
        (9, 20, 11, 0),
        (11, 15, 15, 5),
    )

    # Indicators
    ema_fast: int = int(os.getenv("EMA_FAST", 5))
    ema_slow: int = int(os.getenv("EMA_SLOW", 20))
    atr_window: int = int(os.getenv("ATR_WINDOW", 14))
    atr_min_points: float = float(os.getenv("ATR_MIN_POINTS", 2.0))

    # Risk
    target_points: float = float(os.getenv("TARGET_POINTS", 10.0))
    stoploss_points: float = float(os.getenv("STOPLOSS_POINTS", 2.0))
    confirm_trend_at_entry: bool = os.getenv("CONFIRM_TREND_AT_ENTRY", "true").lower() == "true"
    trade_direction: str = os.getenv("TRADE_DIRECTION", "both").lower()  # "long", "short", or "both"
    daily_loss_cap: float = float(os.getenv("DAILY_LOSS_CAP", -1000.0))

    # EOD square-off
    enable_eod_square_off: bool = os.getenv("ENABLE_EOD_SQUARE_OFF", "true").lower() == "true"
    square_off_time: Tuple[int, int] = (15, 25)

    # History fetch
    warmup_days: int = int(os.getenv("WARMUP_DAYS", 10))
    history_start_date: Optional[str] = os.getenv("HISTORY_START_DATE")
    history_end_date: Optional[str] = os.getenv("HISTORY_END_DATE")

    # Costs (reporting only)
    brokerage_per_trade: float = float(os.getenv("BROKERAGE_PER_TRADE", 20.0))
    slippage_points: float = float(os.getenv("SLIPPAGE_POINTS", 0.10))

    # Stop-loss order type
    sl_order_type: str = os.getenv("SL_ORDER_TYPE", "SL-M")  # "SL" or "SL-M"

# ----------------------------
# Logging + Persistence
# ----------------------------
if os.getenv("LOG_TO_FILE", "false").lower() == "true":
    logging.basicConfig(filename='scalp_with_trend.log', level=logging.INFO, format='%(asctime)s %(message)s')
    log = logging.info
else:
    log = print

def save_state(bot):
    if not bot.cfg.persist_state:
        return
    try:
        with open("state.json", "w") as f:
            json.dump({
                "in_position": bot.in_position,
                "side": bot.side,
                "entry_price": bot.entry_price,
                "tp_level": bot.tp_level,
                "sl_level": bot.sl_level,
                "realized_pnl_today": bot.realized_pnl_today,
                "symbol_in_use": bot.symbol_in_use,
            }, f)
    except Exception as e:
        log(f"[{STRATEGY_NAME}] [WARN] Failed to persist state: {e}")

def load_state(bot):
    if not bot.cfg.persist_state or not os.path.exists("state.json"):
        return
    try:
        with open("state.json", "r") as f:
            state = json.load(f)
            bot.in_position = state.get("in_position", False)
            bot.side = state.get("side")
            bot.entry_price = state.get("entry_price")
            bot.tp_level = state.get("tp_level")
            bot.sl_level = state.get("sl_level")
            bot.realized_pnl_today = state.get("realized_pnl_today", 0.0)
            bot.symbol_in_use = state.get("symbol_in_use", bot.cfg.symbol)
    except Exception as e:
        log(f"[{STRATEGY_NAME}] [WARN] Failed to load state: {e}")

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
    """Validate tradability using OpenAlgo search API."""
    try:
        result = client.search(query=symbol, exchange=exchange)
        if result.get('status') == 'success':
            symbols = [s.get('symbol', '').upper() for s in result.get('data', [])]
            return symbol.upper() in symbols
        return False
    except Exception as e:
        log(f"[{STRATEGY_NAME}] [WARN] Symbol validation failed: {e}")
        return True  # fail-open so user can still run

def resolve_quantity(client, cfg: Config, symbol: Optional[str] = None) -> int:
    """Resolve quantity via OpenAlgo search -> lotsize."""
    sym = symbol or cfg.symbol
    try:
        result = client.search(query=sym, exchange=cfg.exchange)
        if result.get('status') == 'success':
            for inst in result.get('data', []):
                if inst.get('symbol', '').upper() == sym.upper():
                    lot = int(inst.get("lotsize") or 1)
                    return cfg.lots * lot
    except Exception as e:
        log(f"[{STRATEGY_NAME}] [ERROR] resolve_quantity: {e}")
    raise ValueError(f"[{STRATEGY_NAME}] Lot size for {sym} not found")

def compute_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = openalgo.ta.ema(out["close"].values, cfg.ema_fast)
    out["ema_slow"] = openalgo.ta.ema(out["close"].values, cfg.ema_slow)
    out["atr"] = openalgo.ta.atr(out[["high", "low", "close"]].values, cfg.atr_window)
    return out

def get_history(client, cfg: Config, symbol: Optional[str] = None) -> pd.DataFrame:
    today = now_ist().date()
    start_date = cfg.history_start_date or (today - timedelta(days=cfg.warmup_days)).strftime("%Y-%m-%d")
    end_date = cfg.history_end_date or today.strftime("%Y-%m-%d")
    sym = symbol or cfg.symbol
    df = client.history(symbol=sym, exchange=cfg.exchange, interval=cfg.interval,
                        start_date=start_date, end_date=end_date)
    if "timestamp" not in df.columns:
        raise ValueError("history() must return a DataFrame with 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if not pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    return df[["timestamp","open","high","low","close","volume"]].sort_values("timestamp").reset_index(drop=True)

def get_spot_ltp(client, underlying_symbol, exchange) -> float:
    try:
        resp = client.ltp(symbol=underlying_symbol, exchange=exchange)
        return float(resp.get('ltp', 0))
    except Exception:
        return 0.0

def get_atm_option_symbol(client, underlying_symbol: str, exchange: str, option_type: str) -> Optional[str]:
    """
    Pick ATM CE/PE from option chain for the given underlying.
    option_type: 'CE' or 'PE'
    """
    try:
        chain = client.optionchain(symbol=underlying_symbol, exchange=exchange)
        if chain.get('status') != 'success':
            return None
        data = chain.get('data', [])
        # unique strikes
        strikes = sorted({opt.get('strike') for opt in data if opt.get('strike') is not None})
        spot = get_spot_ltp(client, underlying_symbol, exchange) or 0
        if not strikes or spot == 0:
            return None
        atm_strike = min(strikes, key=lambda x: abs(float(x) - spot))
        # choose the exact option symbol
        for opt in data:
            if opt.get('strike') == atm_strike and opt.get('option_type') == option_type:
                return opt.get('symbol')
        return None
    except Exception as e:
        log(f"[{STRATEGY_NAME}] [ERROR] get_atm_option_symbol: {e}")
        return None

def is_index_name(sym: str) -> bool:
    return sym.upper() in ("NIFTY", "BANKNIFTY", "FINNIFTY", "NIFTYNXT50", "MIDCPNIFTY")

# ----------------------------
# Trading Engine
# ----------------------------
class ScalpWithTrendBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = openalgo.simulator(api_key=cfg.api_key) if cfg.test_mode else openalgo.api(api_key=cfg.api_key, host=cfg.api_host, ws_url=cfg.ws_url)
        self.scheduler = BackgroundScheduler(timezone=IST)

        # Active symbol might change per-trade if OPTION_AUTO is used
        self.symbol_in_use: str = cfg.symbol

        self.qty = resolve_quantity(self.client, cfg, symbol=self.symbol_in_use)

        # State
        self.pending_signal: Optional[str] = None
        self.next_entry_time: Optional[datetime] = None
        self.in_position: bool = False
        self.side: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.tp_level: Optional[float] = None
        self.sl_level: Optional[float] = None
        self.entry_order_id: Optional[str] = None
        self.tp_order_id: Optional[str] = None
        self.sl_order_id: Optional[str] = None
        self.realized_pnl_today: float = 0.0
        self.today: date = now_ist().date()

    def start(self):
        # required by your standard
        print("🔁 OpenAlgo Python Bot is running.")
        log(f"\n[{STRATEGY_NAME}] Bot is starting.\nConfig: {json.dumps(asdict(self.cfg), indent=2)}")
        load_state(self)
        log(f"[{STRATEGY_NAME}] Resolved quantity for {self.symbol_in_use}: {self.qty}")

        # Jobs
        self.scheduler.add_job(self.on_bar_close_tick, CronTrigger(minute="*/5", second=2, timezone=IST), id="bar_close")
        self.scheduler.add_job(self.on_bar_open_tick, CronTrigger(minute="*/5", second=5, timezone=IST), id="bar_open")
        self.scheduler.add_job(self.check_order_status, 'interval', seconds=2, id="order_status_check")
        if self.cfg.enable_eod_square_off:
            soh, som = self.cfg.square_off_time
            self.scheduler.add_job(self.square_off_job, CronTrigger(hour=soh, minute=som, second=30, timezone=IST), id="square_off")

        self.scheduler.start()
        signal.signal(signal.SIGINT, self._graceful_exit)
        signal.signal(signal.SIGTERM, self._graceful_exit)

        try:
            while True:
                time.sleep(1)
                if now_ist().date() != self.today:
                    self.today = now_ist().date()
                    self.realized_pnl_today = 0.0
                    log(f"[{STRATEGY_NAME}] [DAY ROLLOVER] New day: {self.today}")
                    save_state(self)
        finally:
            self._graceful_exit()

    # ----- Jobs -----
    def on_bar_close_tick(self):
        if not in_session(now_ist().time(), self.cfg.session_windows):
            return
        try:
            # Always compute indicators on the active/symbol_in_use (index if not in position; if option, still OK)
            df = compute_indicators(get_history(self.client, self.cfg, symbol=self.symbol_in_use), self.cfg)
            i = len(df) - 1
            if i < 1:
                return
            prev, cur = df.iloc[i-1], df.iloc[i]

            # ATR filter
            if float(cur['atr']) < self.cfg.atr_min_points:
                return

            trend_up = cur['ema_fast'] > cur['ema_slow']
            trend_down = cur['ema_fast'] < cur['ema_slow']
            long_sig = (cur['high'] > prev['high']) and trend_up
            short_sig = (cur['low'] < prev['low']) and trend_down

            # direction filter
            if self.cfg.trade_direction == "long":
                short_sig = False
            elif self.cfg.trade_direction == "short":
                long_sig = False

            if self.in_position:
                return

            if long_sig or short_sig:
                self.pending_signal = 'LONG' if long_sig else 'SHORT'
                self.next_entry_time = (cur['timestamp'] + pd.Timedelta(minutes=5)).to_pydatetime()
                log(f"[{STRATEGY_NAME}] [SIGNAL] {self.pending_signal} next entry at {self.next_entry_time}")
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [ERROR] on_bar_close_tick: {e}")

    def on_bar_open_tick(self):
        if not in_session(now_ist().time(), self.cfg.session_windows):
            return
        if self.pending_signal and self.next_entry_time:
            delta = abs((now_ist() - self.next_entry_time).total_seconds())
            if delta <= 10 and not self.in_position:
                # Daily loss cap check (stop trading if <= cap)
                if self.realized_pnl_today <= self.cfg.daily_loss_cap:
                    log(f"[{STRATEGY_NAME}] [RISK] Daily loss cap breached. Skipping entries.")
                    self._clear_pending_signal()
                    return
                self.place_entry(self.pending_signal)
                self._clear_pending_signal()

    def check_order_status(self):
        """Poll TP/SL order status to detect fills (safer than relying on WS for all brokers)."""
        if not self.in_position:
            return
        try:
            # Check TP
            if self.tp_order_id:
                resp = self.client.orderstatus(order_id=self.tp_order_id, strategy=STRATEGY_NAME)
                if resp.get('status') == 'success':
                    od = resp.get('data', {})
                    if str(od.get('order_status', '')).upper() == 'COMPLETE':
                        price = float(od.get('average_price', 0) or 0)
                        if price > 0:
                            self._realize_exit(price, "Target Hit")
                            self.cancel_order_silent(self.sl_order_id)
                            save_state(self)
                            return
            # Check SL
            if self.sl_order_id:
                resp = self.client.orderstatus(order_id=self.sl_order_id, strategy=STRATEGY_NAME)
                if resp.get('status') == 'success':
                    od = resp.get('data', {})
                    if str(od.get('order_status', '')).upper() == 'COMPLETE':
                        price = float(od.get('average_price', 0) or 0)
                        if price > 0:
                            self._realize_exit(price, "Stoploss Hit")
                            self.cancel_order_silent(self.tp_order_id)
                            save_state(self)
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [ERROR] check_order_status: {e}")

    def square_off_job(self):
        if not self.in_position:
            return
        if is_square_off_time(self.cfg, now_ist().time()):
            log(f"[{STRATEGY_NAME}] [EOD] Square-off.")
            self.cancel_order_silent(self.tp_order_id)
            self.cancel_order_silent(self.sl_order_id)
            try:
                action = "SELL" if self.side == 'LONG' else 'BUY'
                resp = self.client.placeorder(
                    strategy=STRATEGY_NAME,
                    symbol=self.symbol_in_use,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action=action,
                    price_type="MARKET",
                    quantity=self.qty
                )
                if resp.get('status') == 'success':
                    time.sleep(0.5)
                    status_resp = self.client.orderstatus(order_id=resp.get('orderid'), strategy=STRATEGY_NAME)
                    px = float(status_resp.get('data', {}).get('average_price', 0) or 0)
                    self._realize_exit(px, "Square-off EOD")
                    save_state(self)
            except Exception as e:
                log(f"[{STRATEGY_NAME}] [ERROR] square_off_job: {e}")

    # ----- Orders -----
    def _maybe_select_atm_option(self, side: str):
        """If option_auto is ON and current symbol is an index, switch to ATM CE/PE per side."""
        if not self.cfg.option_auto:
            return
        if not is_index_name(self.cfg.symbol):
            return
        opt_type = 'CE' if side == 'LONG' else 'PE'
        atm_sym = get_atm_option_symbol(self.client, self.cfg.symbol, self.cfg.exchange, opt_type)
        if atm_sym:
            self.symbol_in_use = atm_sym
            # Update qty for this option instrument (lot size might differ)
            self.qty = resolve_quantity(self.client, self.cfg, symbol=self.symbol_in_use)
            log(f"[{STRATEGY_NAME}] [ATM] Using {opt_type} {self.symbol_in_use} | qty={self.qty}")
        else:
            # fallback to original symbol (likely index FUT if directly tradable)
            self.symbol_in_use = self.cfg.symbol
            self.qty = resolve_quantity(self.client, self.cfg, symbol=self.symbol_in_use)
            log(f"[{STRATEGY_NAME}] [ATM] Fallback to {self.symbol_in_use} | qty={self.qty}")

    def place_entry(self, side: str):
        try:
            # Option auto-selection (per-entry, so CE for LONG, PE for SHORT)
            self._maybe_select_atm_option(side)

            action = "BUY" if side == 'LONG' else 'SELL'
            resp = self.client.placeorder(
                strategy=STRATEGY_NAME,
                symbol=self.symbol_in_use,
                exchange=self.cfg.exchange,
                product=self.cfg.product,
                action=action,
                price_type="MARKET",
                quantity=self.qty
            )
            if resp.get('status') != 'success':
                log(f"[{STRATEGY_NAME}] [ERROR] Order failed: {resp}")
                self._flat_state()
                return

            self.entry_order_id = resp.get('orderid')
            time.sleep(0.5)
            status_resp = self.client.orderstatus(order_id=self.entry_order_id, strategy=STRATEGY_NAME)
            self.entry_price = float(status_resp.get('data', {}).get('average_price', 0) or 0)
            if self.entry_price == 0:
                log(f"[{STRATEGY_NAME}] [WARN] Could not get entry price yet...")
                return

            self.in_position = True
            self.side = side
            if side == 'LONG':
                self.tp_level = self.entry_price + self.cfg.target_points
                self.sl_level = self.entry_price - self.cfg.stoploss_points
            else:
                self.tp_level = self.entry_price - self.cfg.target_points
                self.sl_level = self.entry_price + self.cfg.stoploss_points

            log(f"[{STRATEGY_NAME}] [ENTRY] {side} {self.symbol_in_use} ~{self.entry_price:.2f} TP={self.tp_level:.2f} SL={self.sl_level:.2f}")
            self.place_exit_legs()
            save_state(self)
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [ERROR] place_entry: {e}")
            self._flat_state()

    def place_exit_legs(self):
        if not self.in_position or self.entry_price is None:
            return
        try:
            if self.side == 'LONG':
                # TP: SELL LIMIT
                tp_resp = self.client.placeorder(
                    strategy=STRATEGY_NAME,
                    symbol=self.symbol_in_use,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action="SELL",
                    price_type="LIMIT",
                    price=self.tp_level,
                    quantity=self.qty
                )
                # SL: SELL SL/SL-M
                sl_resp = self.client.placeorder(
                    strategy=STRATEGY_NAME,
                    symbol=self.symbol_in_use,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action="SELL",
                    price_type=self.cfg.sl_order_type,  # "SL" or "SL-M"
                    trigger_price=self.sl_level,
                    quantity=self.qty
                )
            else:
                # SHORT: TP BUY LIMIT
                tp_resp = self.client.placeorder(
                    strategy=STRATEGY_NAME,
                    symbol=self.symbol_in_use,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action="BUY",
                    price_type="LIMIT",
                    price=self.tp_level,
                    quantity=self.qty
                )
                # SL: BUY SL/SL-M
                sl_resp = self.client.placeorder(
                    strategy=STRATEGY_NAME,
                    symbol=self.symbol_in_use,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action="BUY",
                    price_type=self.cfg.sl_order_type,
                    trigger_price=self.sl_level,
                    quantity=self.qty
                )

            self.tp_order_id = tp_resp.get('orderid')
            self.sl_order_id = sl_resp.get('orderid')
            log(f"[{STRATEGY_NAME}] [EXITS] TP oid={self.tp_order_id} @ {self.tp_level:.2f} | SL oid={self.sl_order_id} @ {self.sl_level:.2f}")
            save_state(self)
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [ERROR] place_exit_legs: {e}")

    def cancel_order_silent(self, order_id: Optional[str]):
        if not order_id:
            return
        try:
            self.client.cancelorder(order_id=order_id)
            log(f"[{STRATEGY_NAME}] [CANCEL] order_id={order_id}")
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [WARN] cancel_order_silent: {e}")

    # ----- P&L -----
    def _realize_exit(self, price: float, reason: str):
        if not self.in_position or self.entry_price is None:
            return
        points = (price - self.entry_price) if self.side == 'LONG' else (self.entry_price - price)
        gross = points * self.qty
        costs = 2 * self.cfg.brokerage_per_trade + 2 * self.cfg.slippage_points * self.qty
        net = gross - costs
        self.realized_pnl_today += net
        log(f"[{STRATEGY_NAME}] [EXIT] {reason} @ {price:.2f} | pts={points:.2f} gross={gross:.2f} net={net:.2f} | dailyPnL={self.realized_pnl_today:.2f}")
        self._flat_state()
        save_state(self)

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
        log(f"\n[{STRATEGY_NAME}] [SHUTDOWN] Closing...")
        try:
            if self.in_position:
                action = 'SELL' if self.side == 'LONG' else 'BUY'
                self.client.placeorder(
                    strategy=STRATEGY_NAME,
                    symbol=self.symbol_in_use,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action=action,
                    price_type="MARKET",
                    quantity=self.qty
                )
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [WARN] Shutdown exit failed: {e}")

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
            # If you had WS: self.client.disconnect()
            pass
        except Exception:
            pass

        log(f"[{STRATEGY_NAME}] [SHUTDOWN] Done.")
        sys.exit(0)

# ----------------------------
# Main Entrypoint
# ----------------------------
def main():
    cfg = Config()
    if not cfg.api_key:
        log(f"[{STRATEGY_NAME}] [FATAL] Please set OPENALGO_API_KEY")
        sys.exit(1)
    if cfg.trade_direction not in ["long", "short", "both"]:
        log(f"[{STRATEGY_NAME}] [FATAL] TRADE_DIRECTION must be 'long', 'short', or 'both'")
        sys.exit(1)

    # Create a temporary client for validation/selector
    client = openalgo.simulator(api_key=cfg.api_key) if cfg.test_mode else openalgo.api(api_key=cfg.api_key, host=cfg.api_host, ws_url=cfg.ws_url)

    # Validate symbol (initial)
    if not validate_symbol(client, cfg.symbol, cfg.exchange):
        log(f"[{STRATEGY_NAME}] [FATAL] Symbol {cfg.symbol} not valid on {cfg.exchange}")
        sys.exit(1)

    # Run bot
    bot = ScalpWithTrendBot(cfg)
    bot.start()

if __name__ == "__main__":
    main()
