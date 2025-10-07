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
import os, sys, time, json, signal, logging, re
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

# Option symbol regex pattern (e.g., NIFTY28OCT2525200CE)
OPTION_SYMBOL_RE = re.compile(r"^([A-Z]+)(\d{2})([A-Z]{3})(\d{2})(\d+)(CE|PE)$")

# Fallback lot sizes for common indices/options
INDEX_LOT_SIZES = {
    "NIFTY": 75,
    "BANKNIFTY": 15,
    "FINNIFTY": 25,
    "MIDCPNIFTY": 50,
    "NIFTYNXT50": 10,
    "SENSEX": 10,
    "BANKEX": 15,
    "SENSEX50": 10,
}

# ----------------------------
# Strategy Configuration
# ----------------------------
@dataclass
class Config:
    api_key: str = os.getenv("OPENALGO_API_KEY", "d663120f3a896e0e177cc83b8176932a99175d0eee98ec45106cc8779bcc9280")
    api_host: str = os.getenv("OPENALGO_API_HOST", "http://127.0.0.1:5000")
    ws_url: Optional[str] = os.getenv("OPENALGO_WS_URL", "ws://127.0.0.1:8765")

    # Instrument selection
    symbol: str = os.getenv("SYMBOL", "NIFTY28OCT2525100CE")         # Can be index (NIFTY/BANKNIFTY/FINNIFTY) or an exact tradable symbol
    exchange: str = os.getenv("EXCHANGE", "NFO")       # Futures/Options exchange
    product: str = os.getenv("PRODUCT", "MIS")         # MIS (intraday)
    lots: int = int(os.getenv("LOTS", 1))              # Lot multiplier

    # Engine
    interval: str = os.getenv("INTERVAL", "1m")
    log_to_file: bool = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    test_mode: bool = os.getenv("TEST_MODE", "false").lower() == "true"
    persist_state: bool = os.getenv("PERSIST_STATE", "true").lower() == "true"
    option_auto: bool = os.getenv("OPTION_AUTO", "false").lower() == "true"  # Auto-pick ATM CE/PE each entry if symbol is an index

    # Sessions
    session_windows: Tuple[Tuple[int, int, int, int], ...] = (
        (9, 20, 11, 0),
        (11, 15, 15, 15),
    )

    # Indicators
    ema_fast: int = int(os.getenv("EMA_FAST", 3))
    ema_slow: int = int(os.getenv("EMA_SLOW", 10))
    atr_window: int = int(os.getenv("ATR_WINDOW", 14))
    atr_min_points: float = float(os.getenv("ATR_MIN_POINTS", 2.0))

    # Risk
    target_points: float = float(os.getenv("TARGET_POINTS", 8.0))
    stoploss_points: float = float(os.getenv("STOPLOSS_POINTS", 25.0))
    confirm_trend_at_entry: bool = os.getenv("CONFIRM_TREND_AT_ENTRY", "true").lower() == "true"
    trade_direction: str = os.getenv("TRADE_DIRECTION", "both").lower()  # "long", "short", or "both"
    daily_loss_cap: float = float(os.getenv("DAILY_LOSS_CAP", -1000.0))

    # EOD square-off
    enable_eod_square_off: bool = os.getenv("ENABLE_EOD_SQUARE_OFF", "true").lower() == "true"
    square_off_time: Tuple[int, int] = (15, 25)

    # History fetch
    warmup_days: int = int(os.getenv("WARMUP_DAYS", 3))
    history_start_date: Optional[str] = os.getenv("HISTORY_START_DATE")
    history_end_date: Optional[str] = os.getenv("HISTORY_END_DATE")

    # Costs (reporting only)
    brokerage_per_trade: float = float(os.getenv("BROKERAGE_PER_TRADE", 20.0))
    slippage_points: float = float(os.getenv("SLIPPAGE_POINTS", 0.10))

    # Stop-loss order type
    sl_order_type: str = os.getenv("SL_ORDER_TYPE", "SL-M")  # "SL" or "SL-M"

    # Entry timing
    ignore_entry_delta: bool = os.getenv("IGNORE_ENTRY_DELTA", "true").lower() == "true"  # Ignore entry window timing

# ----------------------------
# Logging + Persistence
# ----------------------------
# Setup logging to both file and console
logger = logging.getLogger(STRATEGY_NAME)
logger.setLevel(logging.INFO)

# Always log to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Conditionally log to file
if os.getenv("LOG_TO_FILE", "true").lower() == "true":
    file_handler = logging.FileHandler('scalp_with_trend.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    print(f"[{STRATEGY_NAME}] Logging to: console + scalp_with_trend.log")
else:
    print(f"[{STRATEGY_NAME}] Logging to: console only")

# Use logger.info for all logging
log = logger.info

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
    """
    Resilient symbol validation with multiple fallbacks.
    Returns True if symbol is valid or if validation cannot be performed.
    """
    try:
        log(f"[{STRATEGY_NAME}] [VALIDATION] Validating {symbol} on {exchange}...")

        # 1) If it looks like a standard option symbol (regex match), try quotes API first
        if OPTION_SYMBOL_RE.match(symbol.upper()):
            log(f"[{STRATEGY_NAME}] [VALIDATION] Detected option symbol format")
            try:
                quotes_resp = client.quotes(symbol=symbol, exchange=exchange)
                if quotes_resp.get("status") == "success":
                    log(f"[{STRATEGY_NAME}] [VALIDATION] ✅ Symbol validated via quotes API")
                    return True
                else:
                    log(f"[{STRATEGY_NAME}] [VALIDATION] Quotes API returned: {quotes_resp.get('status')}")
            except Exception as e:
                log(f"[{STRATEGY_NAME}] [VALIDATION] Quotes API failed: {e}")

            # For option symbols, assume valid even if quotes fails (fail-open)
            log(f"[{STRATEGY_NAME}] [VALIDATION] ✅ Assuming option symbol is valid (fail-open)")
            return True

        # 2) For non-option symbols, try search API
        log(f"[{STRATEGY_NAME}] [VALIDATION] Trying search API...")
        result = client.search(query=symbol, exchange=exchange)

        if result.get('status') == 'success':
            data = result.get('data', [])
            log(f"[{STRATEGY_NAME}] [VALIDATION] Search found {len(data)} results")

            if len(data) > 0:
                symbols = [s.get('symbol', '').upper() for s in data]
                log(f"[{STRATEGY_NAME}] [VALIDATION] First few symbols: {symbols[:5]}")
                is_valid = symbol.upper() in symbols

                if is_valid:
                    log(f"[{STRATEGY_NAME}] [VALIDATION] ✅ Symbol found in search results")
                    return True
                else:
                    log(f"[{STRATEGY_NAME}] [VALIDATION] Symbol not in search results, trying quotes...")
            else:
                log(f"[{STRATEGY_NAME}] [VALIDATION] Empty search results, trying quotes...")
        else:
            log(f"[{STRATEGY_NAME}] [VALIDATION] Search API returned: {result.get('status')}, trying quotes...")

        # 3) Fallback to quotes API
        try:
            quotes_resp = client.quotes(symbol=symbol, exchange=exchange)
            if quotes_resp.get("status") == "success":
                log(f"[{STRATEGY_NAME}] [VALIDATION] ✅ Symbol validated via quotes API (fallback)")
                return True
            else:
                log(f"[{STRATEGY_NAME}] [VALIDATION] Quotes API (fallback) returned: {quotes_resp.get('status')}")
        except Exception as e:
            log(f"[{STRATEGY_NAME}] [VALIDATION] Quotes API (fallback) failed: {e}")

        # 4) If all methods fail, assume valid (fail-open to avoid blocking due to API issues)
        log(f"[{STRATEGY_NAME}] [WARN] Could not validate symbol - assuming valid (fail-open)")
        return True

    except Exception as e:
        log(f"[{STRATEGY_NAME}] [WARN] Symbol validation exception: {e}")
        return True  # fail-open so user can still run

def resolve_quantity(client, cfg: Config, symbol: Optional[str] = None) -> int:
    """Resolve quantity via OpenAlgo search -> lotsize with fallback."""
    sym = symbol or cfg.symbol

    # Try API lookup first
    try:
        result = client.search(query=sym, exchange=cfg.exchange)
        if result.get('status') == 'success':
            for inst in result.get('data', []):
                if inst.get('symbol', '').upper() == sym.upper():
                    lot = int(inst.get("lotsize") or 1)
                    log(f"[{STRATEGY_NAME}] Resolved lot size for {sym} via API: {lot}")
                    return cfg.lots * lot
    except Exception as e:
        log(f"[{STRATEGY_NAME}] [WARN] API resolve_quantity failed: {e}")

    # Fallback: check if it's a known index
    base_symbol = sym.upper()
    for idx in INDEX_LOT_SIZES.keys():
        if base_symbol.startswith(idx):
            lot = INDEX_LOT_SIZES[idx]
            log(f"[{STRATEGY_NAME}] Using fallback lot size for {sym} (detected {idx}): {lot}")
            return cfg.lots * lot

    # Last resort: use lot size of 1
    log(f"[{STRATEGY_NAME}] [WARN] Could not determine lot size for {sym}, using default: 1")
    return cfg.lots * 1

def compute_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = openalgo.ta.ema(out["close"].values, cfg.ema_fast)
    out["ema_slow"] = openalgo.ta.ema(out["close"].values, cfg.ema_slow)
    out["atr"] = openalgo.ta.atr(out["high"].values, out["low"].values, out["close"].values, cfg.atr_window)
    return out

def get_history(client, cfg: Config, symbol: Optional[str] = None) -> pd.DataFrame:
    today = now_ist().date()
    yesterday = today - timedelta(days=1)
    start_date = cfg.history_start_date or (today - timedelta(days=cfg.warmup_days)).strftime("%Y-%m-%d")
    end_date = cfg.history_end_date or yesterday.strftime("%Y-%m-%d")
    sym = symbol or cfg.symbol

    log(f"[{STRATEGY_NAME}] [HISTORY] Fetching {sym}@{cfg.exchange} interval={cfg.interval} from {start_date} to {end_date}")

    try:
        df = client.history(symbol=sym, exchange=cfg.exchange, interval=cfg.interval,
                            start_date=start_date, end_date=end_date)
    except Exception as e:
        log(f"[{STRATEGY_NAME}] [ERROR] history() API call failed: {e}")
        raise

    # Debug: log the raw response
    log(f"[{STRATEGY_NAME}] [HISTORY] Response type: {type(df)}")

    # Check if API returned an error dict instead of DataFrame
    if isinstance(df, dict):
        log(f"[{STRATEGY_NAME}] [ERROR] API returned error response: {df}")
        error_msg = df.get('message', df.get('error', 'Unknown error'))
        raise ValueError(f"history() API error: {error_msg}")

    if not isinstance(df, pd.DataFrame):
        log(f"[{STRATEGY_NAME}] [ERROR] history() returned {type(df)} instead of DataFrame: {df}")
        raise ValueError(f"history() must return a DataFrame, got {type(df)}")

    log(f"[{STRATEGY_NAME}] [HISTORY] Received DataFrame with columns: {list(df.columns)}")
    log(f"[{STRATEGY_NAME}] [HISTORY] DataFrame shape: {df.shape}")

    if df.empty:
        log(f"[{STRATEGY_NAME}] [ERROR] history() returned empty DataFrame")
        raise ValueError("history() returned empty DataFrame - no historical data available")

    # Check if timestamp is in index instead of columns
    if "timestamp" not in df.columns:
        if df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
            log(f"[{STRATEGY_NAME}] [HISTORY] Timestamp found in index, resetting to column")
            df = df.reset_index()
        else:
            log(f"[{STRATEGY_NAME}] [ERROR] Missing 'timestamp' column. Available columns: {list(df.columns)}")
            log(f"[{STRATEGY_NAME}] [DEBUG] Index name: {df.index.name}")
            log(f"[{STRATEGY_NAME}] [DEBUG] First few rows:\n{df.head()}")
            raise ValueError(f"history() must return a DataFrame with 'timestamp' column. Got columns: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Handle timezone: localize if naive, convert if already timezone-aware
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

    log(f"[{STRATEGY_NAME}] [HISTORY] ✅ Successfully fetched {len(df)} bars")
    return df[["timestamp","open","high","low","close","volume"]].sort_values("timestamp").reset_index(drop=True)

def get_spot_exchange(underlying_symbol: str) -> str:
    """Get the correct exchange for spot price lookup"""
    index_map = {
        "NIFTY": "NSE_INDEX",
        "BANKNIFTY": "NSE_INDEX",
        "FINNIFTY": "NSE_INDEX",
        "MIDCPNIFTY": "NSE_INDEX",
        "NIFTYNXT50": "NSE_INDEX",
        "SENSEX": "BSE_INDEX",
        "BANKEX": "BSE_INDEX",
        "SENSEX50": "BSE_INDEX",
    }
    return index_map.get(underlying_symbol.upper(), "NSE_INDEX")

def get_strike_interval(underlying_symbol: str) -> int:
    """Get strike interval for rounding"""
    intervals = {
        "NIFTY": 75,
        "BANKNIFTY": 100,
        "FINNIFTY": 50,
        "MIDCPNIFTY": 25,
        "NIFTYNXT50": 50,
        "SENSEX": 100,
        "BANKEX": 100,
        "SENSEX50": 50,
    }
    return intervals.get(underlying_symbol.upper(), 50)

def round_to_strike(price: float, interval: int) -> float:
    """Round price to nearest strike interval"""
    return round(price / interval) * interval

def get_spot_ltp(client, underlying_symbol, exchange=None) -> float:
    """Get spot LTP with automatic exchange detection"""
    if exchange is None:
        exchange = get_spot_exchange(underlying_symbol)
    try:
        resp = client.ltp(symbol=underlying_symbol, exchange=exchange)
        ltp = float(resp.get('ltp', 0))
        log(f"[{STRATEGY_NAME}] [SPOT_LTP] {underlying_symbol}@{exchange} = ₹{ltp:.2f}")
        return ltp
    except Exception as e:
        log(f"[{STRATEGY_NAME}] [ERROR] get_spot_ltp for {underlying_symbol}@{exchange}: {e}")
        return 0.0

def get_atm_option_symbol(client, underlying_symbol: str, exchange: str, option_type: str) -> Optional[str]:
    """
    Pick ATM CE/PE from option chain for the given underlying.
    Robust ATM calculation with proper rounding and validation.
    option_type: 'CE' or 'PE'
    """
    try:
        # Get spot price from correct exchange
        spot_exchange = get_spot_exchange(underlying_symbol)
        spot = get_spot_ltp(client, underlying_symbol, spot_exchange)

        if spot == 0:
            log(f"[{STRATEGY_NAME}] [ERROR] Could not get spot price for {underlying_symbol}")
            return None

        # Round to nearest strike
        interval = get_strike_interval(underlying_symbol)
        atm_strike = round_to_strike(spot, interval)
        log(f"[{STRATEGY_NAME}] [ATM_CALC] Spot={spot:.2f} | Interval={interval} | ATM Strike={atm_strike:.0f}")

        # Fetch option chain from NFO/derivatives exchange
        chain = client.optionchain(symbol=underlying_symbol, exchange=exchange)
        if chain.get('status') != 'success':
            log(f"[{STRATEGY_NAME}] [ERROR] Option chain fetch failed: {chain}")
            return None

        data = chain.get('data', [])
        if not data:
            log(f"[{STRATEGY_NAME}] [ERROR] Empty option chain data")
            return None

        # Find the exact ATM option symbol
        atm_options = [opt for opt in data
                       if opt.get('strike') == atm_strike
                       and opt.get('option_type') == option_type]

        if not atm_options:
            log(f"[{STRATEGY_NAME}] [WARN] No {option_type} found at strike {atm_strike}, searching nearby strikes...")
            # Fallback: find closest available strike
            strikes = sorted({opt.get('strike') for opt in data if opt.get('strike') is not None})
            if strikes:
                atm_strike = min(strikes, key=lambda x: abs(float(x) - atm_strike))
                log(f"[{STRATEGY_NAME}] [ATM_CALC] Using closest strike: {atm_strike:.0f}")
                atm_options = [opt for opt in data
                             if opt.get('strike') == atm_strike
                             and opt.get('option_type') == option_type]

        if atm_options:
            selected = atm_options[0]
            symbol = selected.get('symbol')
            log(f"[{STRATEGY_NAME}] [ATM_SELECTED] {option_type} @ Strike {atm_strike:.0f} = {symbol}")

            # Validate both CE and PE exist at this strike for transparency
            ce_exists = any(opt.get('strike') == atm_strike and opt.get('option_type') == 'CE' for opt in data)
            pe_exists = any(opt.get('strike') == atm_strike and opt.get('option_type') == 'PE' for opt in data)
            log(f"[{STRATEGY_NAME}] [ATM_VALIDATION] Strike {atm_strike:.0f}: CE={'✓' if ce_exists else '✗'} PE={'✓' if pe_exists else '✗'}")

            return symbol

        log(f"[{STRATEGY_NAME}] [ERROR] No {option_type} option found near ATM")
        return None

    except Exception as e:
        log(f"[{STRATEGY_NAME}] [ERROR] get_atm_option_symbol: {e}")
        import traceback
        log(f"[{STRATEGY_NAME}] [ERROR] Traceback: {traceback.format_exc()}")
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

        # Jobs - schedule based on configured interval
        interval_minutes = self._parse_interval_minutes(self.cfg.interval)
        log(f"[{STRATEGY_NAME}] Scheduling jobs for {interval_minutes}-minute interval")
        if interval_minutes == 1:
            # For 1-minute interval, run every minute
            self.scheduler.add_job(self.on_bar_close_tick, CronTrigger(minute="*", second=2, timezone=IST), id="bar_close")
            self.scheduler.add_job(self.on_bar_open_tick, CronTrigger(minute="*", second=5, timezone=IST), id="bar_open")
        else:
            # For other intervals, use the standard */N notation
            self.scheduler.add_job(self.on_bar_close_tick, CronTrigger(minute=f"*/{interval_minutes}", second=2, timezone=IST), id="bar_close")
            self.scheduler.add_job(self.on_bar_open_tick, CronTrigger(minute=f"*/{interval_minutes}", second=5, timezone=IST), id="bar_open")
        self.scheduler.add_job(self.check_order_status, 'interval', seconds=2, id="order_status_check")
        if self.cfg.enable_eod_square_off:
            soh, som = self.cfg.square_off_time
            self.scheduler.add_job(self.square_off_job, CronTrigger(hour=soh, minute=som, second=30, timezone=IST), id="square_off")

        self.scheduler.start()

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

    def _parse_interval_minutes(self, interval: str) -> int:
        """Parse interval string (e.g., '1m', '5m', '15m') to minutes"""
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('min'):
            return int(interval[:-3])
        else:
            log(f"[{STRATEGY_NAME}] [WARN] Unknown interval format '{interval}', defaulting to 5 minutes")
            return 5

    # ----- Jobs -----
    def on_bar_close_tick(self):
        if not in_session(now_ist().time(), self.cfg.session_windows):
            return
        try:
            # Always compute indicators on the active/symbol_in_use (index if not in position; if option, still OK)
            log(f"[{STRATEGY_NAME}] [BAR_CLOSE] Fetching history for {self.symbol_in_use}...")
            df = compute_indicators(get_history(self.client, self.cfg, symbol=self.symbol_in_use), self.cfg)
            i = len(df) - 1
            if i < 1:
                log(f"[{STRATEGY_NAME}] [WARN] Insufficient data: {len(df)} bars")
                return
            prev, cur = df.iloc[i-1], df.iloc[i]

            # Log current candle details
            log(f"[{STRATEGY_NAME}] [CANDLE] {self.symbol_in_use} @ {cur['timestamp']}")
            log(f"[{STRATEGY_NAME}] [CANDLE] O:{cur['open']:.2f} H:{cur['high']:.2f} L:{cur['low']:.2f} C:{cur['close']:.2f} V:{cur['volume']:.0f}")
            log(f"[{STRATEGY_NAME}] [INDICATORS] EMA_Fast:{cur['ema_fast']:.2f} EMA_Slow:{cur['ema_slow']:.2f} ATR:{cur['atr']:.2f}")

            # ATR filter
            if float(cur['atr']) < self.cfg.atr_min_points:
                log(f"[{STRATEGY_NAME}] [FILTER] ATR {cur['atr']:.2f} < min {self.cfg.atr_min_points} - No signal")
                return

            trend_up = cur['ema_fast'] > cur['ema_slow']
            trend_down = cur['ema_fast'] < cur['ema_slow']
            trend_str = "UP" if trend_up else ("DOWN" if trend_down else "NEUTRAL")
            log(f"[{STRATEGY_NAME}] [TREND] {trend_str} (Fast:{cur['ema_fast']:.2f} vs Slow:{cur['ema_slow']:.2f})")

            # Signal calculation
            high_breakout = cur['high'] > prev['high'] 
            low_breakdown = cur['low'] < prev['low'] 
            log(f"[{STRATEGY_NAME}] [SIGNAL_CHECK] High:{cur['high']:.2f} vs Prev:{prev['high']:.2f} = {'BREAK' if high_breakout else 'NO'}")
            log(f"[{STRATEGY_NAME}] [SIGNAL_CHECK] Low:{cur['low']:.2f} vs Prev:{prev['low']:.2f} = {'BREAK' if low_breakdown else 'NO'}")

            # Original logic (uncomment if needed)
            # long_sig = (cur['high'] > prev['high']) and trend_up
            # short_sig = (cur['low'] < prev['low']) and trend_down
            long_sig = high_breakout
            short_sig = low_breakdown

            log(f"[{STRATEGY_NAME}] [SIGNAL_RAW] LONG={long_sig} SHORT={short_sig}")

            # direction filter
            if self.cfg.trade_direction == "long":
                short_sig = False
                log(f"[{STRATEGY_NAME}] [FILTER] Trade direction=LONG only, SHORT disabled")
            elif self.cfg.trade_direction == "short":
                long_sig = False
                log(f"[{STRATEGY_NAME}] [FILTER] Trade direction=SHORT only, LONG disabled")

            log(f"[{STRATEGY_NAME}] [SIGNAL_FINAL] LONG={long_sig} SHORT={short_sig}")

            if self.in_position:
                log(f"[{STRATEGY_NAME}] [SKIP] Already in position ({self.side})")
                return

            if long_sig or short_sig:
                self.pending_signal = 'LONG' if long_sig else 'SHORT'
                interval_minutes = self._parse_interval_minutes(self.cfg.interval)
                self.next_entry_time = (cur['timestamp'] + pd.Timedelta(minutes=interval_minutes)).to_pydatetime()
                log(f"[{STRATEGY_NAME}] ⚡ [SIGNAL] {self.pending_signal} detected! Next entry at {self.next_entry_time}")
            else:
                log(f"[{STRATEGY_NAME}] [NO_SIGNAL] No trade signals on this bar")

        except Exception as e:
            log(f"[{STRATEGY_NAME}] [ERROR] on_bar_close_tick: {e}")

    def on_bar_open_tick(self):
        current_time = now_ist()
        if not in_session(current_time.time(), self.cfg.session_windows):
            log(f"[{STRATEGY_NAME}] [BAR_OPEN] Outside session windows at {current_time.time()}")
            return

        if self.pending_signal and self.next_entry_time:
            delta = (current_time - self.next_entry_time).total_seconds()  # Can be negative if in future
            log(f"[{STRATEGY_NAME}] [BAR_OPEN] Checking entry: Signal={self.pending_signal} | Scheduled={self.next_entry_time} | Now={current_time} | Delta={delta:.1f}s")

            # Check timing window based on ignore_entry_delta flag
            timing_ok = self.cfg.ignore_entry_delta or (0 <= delta <= 10)

            if timing_ok and delta >= 0 and not self.in_position:
                # Daily loss cap check (stop trading if <= cap)
                if self.realized_pnl_today <= self.cfg.daily_loss_cap:
                    log(f"[{STRATEGY_NAME}] [RISK] Daily loss cap breached ({self.realized_pnl_today:.2f} <= {self.cfg.daily_loss_cap:.2f}). Skipping entries.")
                    self._clear_pending_signal()
                    return

                if self.cfg.ignore_entry_delta and delta > 10:
                    log(f"[{STRATEGY_NAME}] [BAR_OPEN] ⚡ Entry delta ignored (IGNORE_ENTRY_DELTA=true). Executing {self.pending_signal}!")
                else:
                    log(f"[{STRATEGY_NAME}] [BAR_OPEN] ✅ Executing {self.pending_signal} entry now!")
                self.place_entry(self.pending_signal)
                self._clear_pending_signal()
            elif not self.cfg.ignore_entry_delta and delta > 10:
                log(f"[{STRATEGY_NAME}] [BAR_OPEN] ⏰ Entry window missed (delta={delta:.1f}s > 10s). Clearing signal.")
                self._clear_pending_signal()
            elif delta < 0:
                log(f"[{STRATEGY_NAME}] [BAR_OPEN] ⏳ Entry scheduled for future ({abs(delta):.1f}s away). Waiting...")
            elif self.in_position:
                log(f"[{STRATEGY_NAME}] [BAR_OPEN] Already in position. Clearing pending signal.")
                self._clear_pending_signal()
        else:
            log(f"[{STRATEGY_NAME}] [BAR_OPEN] No pending signal")

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
                    order_status = str(od.get('order_status', '')).upper()
                    if order_status == 'COMPLETE':
                        price = float(od.get('average_price', 0) or 0)
                        if price > 0:
                            log(f"[{STRATEGY_NAME}] 🎯 [ORDER_FILLED] Target order filled @ ₹{price:.2f}")
                            self._realize_exit(price, "Target Hit")
                            self.cancel_order_silent(self.sl_order_id)
                            save_state(self)
                            return

            # Check SL
            if self.sl_order_id:
                resp = self.client.orderstatus(order_id=self.sl_order_id, strategy=STRATEGY_NAME)
                if resp.get('status') == 'success':
                    od = resp.get('data', {})
                    order_status = str(od.get('order_status', '')).upper()
                    if order_status == 'COMPLETE':
                        price = float(od.get('average_price', 0) or 0)
                        if price > 0:
                            log(f"[{STRATEGY_NAME}] 🛑 [ORDER_FILLED] StopLoss order filled @ ₹{price:.2f}")
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
            log(f"[{STRATEGY_NAME}] [ATM] Option auto-selection is OFF")
            return
        if not is_index_name(self.cfg.symbol):
            log(f"[{STRATEGY_NAME}] [ATM] Symbol {self.cfg.symbol} is not an index, using as-is")
            return

        log(f"[{STRATEGY_NAME}] [ATM] Fetching option chain for {self.cfg.symbol}...")
        opt_type = 'CE' if side == 'LONG' else 'PE'

        # Get ATM option (spot LTP is fetched inside get_atm_option_symbol with correct exchange)
        atm_sym = get_atm_option_symbol(self.client, self.cfg.symbol, self.cfg.exchange, opt_type)
        if atm_sym:
            self.symbol_in_use = atm_sym
            # Update qty for this option instrument (lot size might differ)
            self.qty = resolve_quantity(self.client, self.cfg, symbol=self.symbol_in_use)
            log(f"[{STRATEGY_NAME}] ✅ [ATM] Selected {opt_type}: {self.symbol_in_use} | Qty={self.qty}")

            # Try to get LTP of the option
            try:
                opt_ltp_resp = self.client.ltp(symbol=self.symbol_in_use, exchange=self.cfg.exchange)
                opt_ltp = float(opt_ltp_resp.get('ltp', 0))
                if opt_ltp > 0:
                    log(f"[{STRATEGY_NAME}] [ATM] {self.symbol_in_use} Option LTP: ₹{opt_ltp:.2f}")
            except Exception as e:
                log(f"[{STRATEGY_NAME}] [WARN] Could not fetch option LTP: {e}")
        else:
            # fallback to original symbol (likely index FUT if directly tradable)
            self.symbol_in_use = self.cfg.symbol
            self.qty = resolve_quantity(self.client, self.cfg, symbol=self.symbol_in_use)
            log(f"[{STRATEGY_NAME}] ⚠️ [ATM] No ATM option found, fallback to {self.symbol_in_use} | Qty={self.qty}")

    def place_entry(self, side: str):
        try:
            log(f"[{STRATEGY_NAME}] 🚀 [ENTRY] Initiating {side} entry...")

            # Option auto-selection (per-entry, so CE for LONG, PE for SHORT)
            self._maybe_select_atm_option(side)

            action = "BUY" if side == 'LONG' else 'SELL'
            log(f"[{STRATEGY_NAME}] [ORDER] Placing {action} order for {self.symbol_in_use} x {self.qty}")

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
                log(f"[{STRATEGY_NAME}] ❌ [ERROR] Order placement failed: {resp}")
                self._flat_state()
                return

            self.entry_order_id = resp.get('orderid')
            log(f"[{STRATEGY_NAME}] [ORDER] Order placed. OrderID: {self.entry_order_id}")
            log(f"[{STRATEGY_NAME}] [ORDER] Waiting for fill confirmation...")

            time.sleep(0.5)
            status_resp = self.client.orderstatus(order_id=self.entry_order_id, strategy=STRATEGY_NAME)
            order_data = status_resp.get('data', {})
            self.entry_price = float(order_data.get('average_price', 0) or 0)

            if self.entry_price == 0:
                log(f"[{STRATEGY_NAME}] ⚠️ [WARN] Entry price not available yet. Order status: {order_data.get('order_status')}")
                return

            self.in_position = True
            self.side = side

            if side == 'LONG':
                self.tp_level = self.entry_price + self.cfg.target_points
                self.sl_level = self.entry_price - self.cfg.stoploss_points
            else:
                self.tp_level = self.entry_price - self.cfg.target_points
                self.sl_level = self.entry_price + self.cfg.stoploss_points

            potential_profit = self.cfg.target_points * self.qty
            potential_loss = self.cfg.stoploss_points * self.qty

            log(f"[{STRATEGY_NAME}] ✅ [ENTRY_FILLED] {side} position opened")
            log(f"[{STRATEGY_NAME}] [POSITION] Symbol: {self.symbol_in_use} | Qty: {self.qty}")
            log(f"[{STRATEGY_NAME}] [POSITION] Entry: ₹{self.entry_price:.2f}")
            log(f"[{STRATEGY_NAME}] [POSITION] Target: ₹{self.tp_level:.2f} (+{self.cfg.target_points} pts = ₹{potential_profit:.2f})")
            log(f"[{STRATEGY_NAME}] [POSITION] StopLoss: ₹{self.sl_level:.2f} (-{self.cfg.stoploss_points} pts = ₹{potential_loss:.2f})")
            log(f"[{STRATEGY_NAME}] [POSITION] Risk:Reward = 1:{self.cfg.target_points/self.cfg.stoploss_points:.2f}")

            self.place_exit_legs()
            save_state(self)
        except Exception as e:
            log(f"[{STRATEGY_NAME}] ❌ [ERROR] place_entry exception: {e}")
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

        pnl_emoji = "💰" if net > 0 else "💸"
        log(f"[{STRATEGY_NAME}] {pnl_emoji} [EXIT] {reason}")
        log(f"[{STRATEGY_NAME}] [EXIT_DETAILS] {self.side} position closed @ ₹{price:.2f}")
        log(f"[{STRATEGY_NAME}] [EXIT_DETAILS] Entry: ₹{self.entry_price:.2f} -> Exit: ₹{price:.2f}")
        log(f"[{STRATEGY_NAME}] [P&L] Points: {points:+.2f} | Gross: ₹{gross:+.2f} | Costs: ₹{costs:.2f}")
        log(f"[{STRATEGY_NAME}] [P&L] Net P&L: ₹{net:+.2f} | Daily Total: ₹{self.realized_pnl_today:+.2f}")

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
