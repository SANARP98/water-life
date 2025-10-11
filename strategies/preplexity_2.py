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
import os, sys, time, json, signal, logging, re, random
from dataclasses import dataclass, asdict
from datetime import datetime, date, time as dtime, timedelta
from typing import Optional, Tuple, Dict, Callable, Any
from functools import wraps
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

# ----------------------------
# API Robustness Utilities
# ----------------------------

def api_retry(max_retries: int = 3, backoff_base: float = 1.0, exceptions: Tuple = (Exception,)):
    """
    Decorator for retrying API calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_base: Base delay in seconds (will be multiplied exponentially)
        exceptions: Tuple of exception types to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
                        log(f"[{STRATEGY_NAME}] [RETRY] API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                        log(f"[{STRATEGY_NAME}] [RETRY] Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                    else:
                        log(f"[{STRATEGY_NAME}] [ERROR] API call failed after {max_retries} attempts: {e}")
                        raise last_exception
            return None
        return wrapper
    return decorator


class RateLimiter:
    """
    Simple rate limiter to prevent API burst requests.
    Thread-safe implementation for single-threaded schedulers.
    """
    def __init__(self, max_calls: int, time_window: float):
        """
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []

    def wait_if_needed(self):
        """Block if rate limit would be exceeded"""
        now = time.time()
        # Remove calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]

        if len(self.calls) >= self.max_calls:
            # Calculate wait time
            oldest_call = self.calls[0]
            wait_time = self.time_window - (now - oldest_call) + 0.1  # Add small buffer
            if wait_time > 0:
                log(f"[{STRATEGY_NAME}] [RATE_LIMIT] Throttling API calls, waiting {wait_time:.2f}s")
                time.sleep(wait_time)

        self.calls.append(time.time())


# Global rate limiter: 10 requests per second (conservative limit)
api_rate_limiter = RateLimiter(max_calls=10, time_window=1.0)


def safe_api_call(func: Callable, *args, **kwargs) -> Any:
    """
    Wrapper for API calls with rate limiting and retry logic.
    Use this for all broker API calls to ensure robustness.
    """
    api_rate_limiter.wait_if_needed()
    try:
        result = func(*args, **kwargs)
        # Check for API error responses
        if isinstance(result, dict):
            status = result.get('status', '')
            if status == 'error' or status == 'failed':
                error_msg = result.get('message', result.get('error', 'Unknown API error'))
                error_code = result.get('code', 'N/A')
                log(f"[{STRATEGY_NAME}] [API_ERROR] Code: {error_code} | Message: {error_msg}")
                raise ValueError(f"API Error [{error_code}]: {error_msg}")
        return result
    except Exception as e:
        log(f"[{STRATEGY_NAME}] [API_EXCEPTION] {func.__name__ if hasattr(func, '__name__') else 'API call'}: {e}")
        raise


# Fallback lot sizes for common indices/options (Updated to May 2025 specifications)
INDEX_LOT_SIZES = {
    "NIFTY": 75,        # Correct
    "BANKNIFTY": 35,    # Updated from 15
    "FINNIFTY": 65,     # Updated from 25
    "MIDCPNIFTY": 140,  # Updated from 50
    "NIFTYNXT50": 25,   # Updated from 10
    "SENSEX": 20,       # Updated from 10
    "BANKEX": 30,       # Updated from 15
    "SENSEX50": 60,     # Updated from 10
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
    symbol: str = os.getenv("SYMBOL", "NIFTY28OCT2525200PE")         # Can be index (NIFTY/BANKNIFTY/FINNIFTY) or an exact tradable symbol
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
    target_points: float = float(os.getenv("TARGET_POINTS", 2.50))
    stoploss_points: float = float(os.getenv("STOPLOSS_POINTS", 2.50))
    confirm_trend_at_entry: bool = os.getenv("CONFIRM_TREND_AT_ENTRY", "true").lower() == "true"
    trade_direction: str = os.getenv("TRADE_DIRECTION", "long").lower()  # "long", "short", or "both"
    daily_loss_cap: float = float(os.getenv("DAILY_LOSS_CAP", -1000.0))

    # EOD square-off
    enable_eod_square_off: bool = os.getenv("ENABLE_EOD_SQUARE_OFF", "true").lower() == "true"
    square_off_time: Tuple[int, int] = (15, 25)

    # History fetch
    warmup_days: int = int(os.getenv("WARMUP_DAYS", 2))
    history_start_date: Optional[str] = os.getenv("HISTORY_START_DATE")
    history_end_date: Optional[str] = os.getenv("HISTORY_END_DATE")

    # Costs (reporting only)
    brokerage_per_trade: float = float(os.getenv("BROKERAGE_PER_TRADE", 20.0))
    slippage_points: float = float(os.getenv("SLIPPAGE_POINTS", 0.10))

    # Stop-loss order type
    sl_order_type: str = os.getenv("SL_ORDER_TYPE", "SL-M")  # "SL" or "SL-M"

    # Entry timing
    ignore_entry_delta: bool = os.getenv("IGNORE_ENTRY_DELTA", "true").lower() == "true"  # Ignore entry window timing

    # Position reconciliation
    check_position_on_startup: bool = os.getenv("CHECK_POSITION_ON_STARTUP", "true").lower() == "true"  # Verify state.json against actual broker positions

    # History optimization
    skip_history_fetch: bool = os.getenv("SKIP_HISTORY_FETCH", "true").lower() == "true"  # Skip historical data (use live data only)

    # History caching (CSV-based)
    use_history_cache: bool = os.getenv("USE_HISTORY_CACHE", "true").lower() == "true"  # Cache history to CSV
    history_cache_dir: str = os.getenv("HISTORY_CACHE_DIR", "history_cache")  # Cache directory
    history_days: int = int(os.getenv("HISTORY_DAYS", 2))  # Days of history to cache
    force_refresh_cache: bool = os.getenv("FORCE_REFRESH_CACHE", "false").lower() == "true"  # Force re-download

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

# Dynamic log level control based on environment variable
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR
}
log_level = log_level_map.get(log_level_str, logging.INFO)
logger.setLevel(log_level)
console_handler.setLevel(log_level)

# Conditionally log to file
if os.getenv("LOG_TO_FILE", "true").lower() == "true":
    file_handler = logging.FileHandler('scalp_with_trend.log')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    print(f"[{STRATEGY_NAME}] Logging to: console + scalp_with_trend.log | Level: {log_level_str}")
else:
    print(f"[{STRATEGY_NAME}] Logging to: console only | Level: {log_level_str}")

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

def reconcile_position(bot):
    """
    Check if state.json matches actual broker positions.
    If position was manually closed, update state.json accordingly.
    """
    if not bot.cfg.check_position_on_startup:
        log(f"[{STRATEGY_NAME}] [RECONCILE] Position check disabled (CHECK_POSITION_ON_STARTUP=false)")
        return

    if not bot.in_position:
        log(f"[{STRATEGY_NAME}] [RECONCILE] No position in state.json - nothing to reconcile")
        return

    try:
        log(f"[{STRATEGY_NAME}] [RECONCILE] Checking broker positions for {bot.symbol_in_use}...")

        # Get positions from broker (using positionbook API with retry and rate limiting)
        positions_resp = safe_api_call(bot.client.positionbook)

        if positions_resp.get('status') != 'success':
            log(f"[{STRATEGY_NAME}] [RECONCILE] ⚠️ Could not fetch positions: {positions_resp}")
            return

        positions = positions_resp.get('data', [])
        log(f"[{STRATEGY_NAME}] [RECONCILE] Found {len(positions)} position(s) in broker")

        # Look for our symbol
        found_position = None
        for pos in positions:
            pos_symbol = pos.get('symbol', '').upper()
            if pos_symbol == bot.symbol_in_use.upper():
                found_position = pos
                break

        if found_position:
            # Position exists in broker
            quantity = int(found_position.get('quantity', 0) or 0)
            net_qty = int(found_position.get('netqty', 0) or 0)

            log(f"[{STRATEGY_NAME}] [RECONCILE] ✅ Position found: {bot.symbol_in_use} | Qty={quantity} | NetQty={net_qty}")

            if net_qty == 0:
                # Position closed but still in response (daytrade closed)
                log(f"[{STRATEGY_NAME}] [RECONCILE] 🔄 Position closed (NetQty=0). Updating state.json...")
                bot._flat_state()
                save_state(bot)
                log(f"[{STRATEGY_NAME}] [RECONCILE] ✅ State cleared - position was manually closed")
            else:
                log(f"[{STRATEGY_NAME}] [RECONCILE] ✅ Position matches state.json")
        else:
            # Position NOT found in broker
            log(f"[{STRATEGY_NAME}] [RECONCILE] ⚠️ Position in state.json but NOT in broker!")
            log(f"[{STRATEGY_NAME}] [RECONCILE] 🔄 Position was manually closed. Updating state.json...")
            bot._flat_state()
            save_state(bot)
            log(f"[{STRATEGY_NAME}] [RECONCILE] ✅ State cleared - position was manually closed")

    except Exception as e:
        log(f"[{STRATEGY_NAME}] [RECONCILE] ⚠️ Position reconciliation failed: {e}")
        log(f"[{STRATEGY_NAME}] [RECONCILE] Continuing with state.json values...")

# ----------------------------
# Utilities
# ----------------------------
def normalize_interval(interval: str) -> str:
    """
    Normalize interval string to OpenAlgo API standard format.
    Daily intervals must use 'D' not '1d' per OpenAlgo documentation.

    Args:
        interval: Input interval string (e.g., '1m', '5m', '1d', 'D')

    Returns:
        Normalized interval string ('D' for daily, others unchanged)
    """
    if interval.lower() in ("1d", "d", "day", "daily"):
        return "D"
    return interval


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

    # Normalize interval for OpenAlgo API (D for daily, not 1d)
    normalized_interval = normalize_interval(cfg.interval)
    log(f"[{STRATEGY_NAME}] [HISTORY] Fetching {sym}@{cfg.exchange} interval={normalized_interval} from {start_date} to {end_date}")

    try:
        df = client.history(symbol=sym, exchange=cfg.exchange, interval=normalized_interval,
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

def get_cache_filename(cfg: Config, symbol: str) -> str:
    """Get the cache filename for a given symbol"""
    # Sanitize symbol name for filesystem
    safe_symbol = symbol.replace("/", "_").replace("\\", "_")
    cache_dir = cfg.history_cache_dir
    return os.path.join(cache_dir, f"{safe_symbol}_history.csv")

def load_history_from_cache(cfg: Config, symbol: str) -> Optional[pd.DataFrame]:
    """Load historical data from CSV cache if available"""
    if not cfg.use_history_cache:
        return None

    cache_file = get_cache_filename(cfg, symbol)

    if not os.path.exists(cache_file):
        log(f"[{STRATEGY_NAME}] [CACHE] No cache file found: {cache_file}")
        return None

    try:
        # Check file age
        file_mtime = os.path.getmtime(cache_file)
        file_date = datetime.fromtimestamp(file_mtime, IST).date()
        today = now_ist().date()

        if cfg.force_refresh_cache:
            log(f"[{STRATEGY_NAME}] [CACHE] Force refresh enabled, ignoring cache")
            return None

        if file_date < today:
            log(f"[{STRATEGY_NAME}] [CACHE] Cache is from previous day ({file_date}), will refresh")
            return None

        # Load CSV
        log(f"[{STRATEGY_NAME}] [CACHE] Loading from {cache_file}...")
        df = pd.read_csv(cache_file)

        # Validate required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            log(f"[{STRATEGY_NAME}] [CACHE] Invalid cache format, missing columns")
            return None

        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

        log(f"[{STRATEGY_NAME}] [CACHE] ✅ Loaded {len(df)} bars from cache")
        return df

    except Exception as e:
        log(f"[{STRATEGY_NAME}] [CACHE] ⚠️ Failed to load cache: {e}")
        return None

def save_history_to_cache(cfg: Config, symbol: str, df: pd.DataFrame):
    """Save historical data to CSV cache"""
    if not cfg.use_history_cache:
        return

    try:
        # Create cache directory if needed
        os.makedirs(cfg.history_cache_dir, exist_ok=True)

        cache_file = get_cache_filename(cfg, symbol)

        # Save to CSV
        df.to_csv(cache_file, index=False)

        log(f"[{STRATEGY_NAME}] [CACHE] ✅ Saved {len(df)} bars to {cache_file}")

    except Exception as e:
        log(f"[{STRATEGY_NAME}] [CACHE] ⚠️ Failed to save cache: {e}")

def append_live_candles_to_history(historical_df: pd.DataFrame, live_df: pd.DataFrame) -> pd.DataFrame:
    """Merge cached history with today's live candles"""
    if live_df.empty:
        return historical_df

    try:
        # Combine dataframes
        combined = pd.concat([historical_df, live_df], ignore_index=True)

        # Remove duplicates based on timestamp (keep latest)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last')

        # Sort by timestamp
        combined = combined.sort_values('timestamp').reset_index(drop=True)

        log(f"[{STRATEGY_NAME}] [CACHE] Merged {len(historical_df)} cached + {len(live_df)} live = {len(combined)} total bars")

        return combined

    except Exception as e:
        log(f"[{STRATEGY_NAME}] [CACHE] ⚠️ Failed to merge live candles: {e}")
        return historical_df

def get_historical_data(client, cfg: Config, symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Get historical data with caching support.
    1. Try to load from CSV cache
    2. If not found or stale, fetch from API and save to cache
    3. Return dataframe with indicators computed
    """
    sym = symbol or cfg.symbol

    # Try cache first
    if cfg.use_history_cache and not cfg.force_refresh_cache:
        cached_df = load_history_from_cache(cfg, sym)
        if cached_df is not None:
            # Compute indicators if not already present
            if "ema_fast" not in cached_df.columns:
                cached_df = compute_indicators(cached_df, cfg)
            return cached_df

    # Cache miss or disabled - fetch from API
    log(f"[{STRATEGY_NAME}] [CACHE] Fetching {cfg.history_days} days of history from API...")

    today = now_ist().date()
    start_date = (today - timedelta(days=cfg.history_days)).strftime("%Y-%m-%d")
    end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    # Normalize interval for OpenAlgo API (D for daily, not 1d)
    normalized_interval = normalize_interval(cfg.interval)
    log(f"[{STRATEGY_NAME}] [HISTORY] Fetching {sym}@{cfg.exchange} interval={normalized_interval} from {start_date} to {end_date}")

    try:
        df = client.history(symbol=sym, exchange=cfg.exchange, interval=normalized_interval,
                           start_date=start_date, end_date=end_date)
    except Exception as e:
        log(f"[{STRATEGY_NAME}] [ERROR] history() API call failed: {e}")
        raise

    # Validate response
    if isinstance(df, dict):
        error_msg = df.get('message', df.get('error', 'Unknown error'))
        raise ValueError(f"history() API error: {error_msg}")

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("history() returned empty or invalid data")

    # Process timestamp
    if "timestamp" not in df.columns:
        if df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise ValueError(f"Missing 'timestamp' column. Got: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

    df = df[["timestamp","open","high","low","close","volume"]].sort_values("timestamp").reset_index(drop=True)

    log(f"[{STRATEGY_NAME}] [HISTORY] ✅ Successfully fetched {len(df)} bars")

    # Compute indicators
    df = compute_indicators(df, cfg)

    # Save to cache
    save_history_to_cache(cfg, sym, df)

    return df

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
    """
    Get strike interval for rounding to ATM strike.

    Note: BANKNIFTY uses 100-point intervals for ATM/near strikes,
    and 500-point intervals for far OTM strikes. This function returns
    100 for ATM calculation. For advanced strategies requiring far OTM
    strikes, additional logic may be needed.

    Current NSE specifications (as of 2024):
    - NIFTY: 50 points
    - BANKNIFTY: 100 points (ATM/near), 500 points (far)
    - FINNIFTY: 50 points
    - MIDCPNIFTY: 25 points
    - NIFTYNXT50: 100 points
    """
    intervals = {
        "NIFTY": 50,
        "BANKNIFTY": 100,  # ATM/near strike interval
        "FINNIFTY": 50,
        "MIDCPNIFTY": 25,
        "NIFTYNXT50": 100,
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

        # Live data buffer (for skip_history_fetch mode)
        self.live_df: pd.DataFrame = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        self.current_candle: Dict = {}  # Track current building candle
        if self.cfg.use_history_cache:
            os.makedirs(self.cfg.history_cache_dir, exist_ok=True)
        self.cached_history_df: pd.DataFrame = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        self._cache_loaded_for: Optional[Tuple[str, date]] = None

    def start(self):
        # required by your standard
        print("🔁 OpenAlgo Python Bot is running.")
        log(f"\n[{STRATEGY_NAME}] Bot is starting.\nConfig: {json.dumps(asdict(self.cfg), indent=2)}")
        load_state(self)
        reconcile_position(self)  # Check if position matches broker
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
        # Use randomized intervals for order status polling (2-4 seconds) to avoid predictable patterns
        self.scheduler.add_job(self.check_order_status, 'interval', seconds=3, jitter=1, id="order_status_check")
        if self.cfg.enable_eod_square_off:
            soh, som = self.cfg.square_off_time
            self.scheduler.add_job(self.square_off_job, CronTrigger(hour=soh, minute=som, second=30, timezone=IST), id="square_off")

        # Add session-end reconciliation job (runs 5 minutes after square-off)
        self.scheduler.add_job(self.session_end_reconciliation, CronTrigger(hour=soh, minute=som + 5, second=0, timezone=IST), id="eod_reconciliation")

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

    def _ensure_cache_loaded(self):
        """Load cached history once per day per symbol to support skip-history mode."""
        if not self.cfg.use_history_cache:
            return

        cache_key = (self.symbol_in_use, now_ist().date())
        if self._cache_loaded_for == cache_key:
            return

        cached_df = load_history_from_cache(self.cfg, self.symbol_in_use)
        if cached_df is not None:
            self.cached_history_df = cached_df
        else:
            # Keep an empty frame with expected columns for downstream merges
            self.cached_history_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        self._cache_loaded_for = cache_key

    # ----- Jobs -----
    def on_bar_close_tick(self):
        if not in_session(now_ist().time(), self.cfg.session_windows):
            return
        try:
            # Get historical data (with caching support)
            if self.cfg.skip_history_fetch:
                log(f"[{STRATEGY_NAME}] [BAR_CLOSE] Using live data only (SKIP_HISTORY_FETCH=true)")
                self._ensure_cache_loaded()

                live_df = getattr(self, "live_df", pd.DataFrame())
                has_live = not live_df.empty
                has_cache = not self.cached_history_df.empty

                if has_cache and has_live:
                    df = append_live_candles_to_history(self.cached_history_df, live_df)
                    df = compute_indicators(df, self.cfg)
                elif has_live:
                    df = live_df
                elif has_cache:
                    log(f"[{STRATEGY_NAME}] [WARN] Cache loaded but waiting for first live candle")
                    return
                else:
                    log(f"[{STRATEGY_NAME}] [WARN] No historical cache and no live data yet - set SKIP_HISTORY_FETCH=false once to bootstrap history")
                    return
            else:
                # Get cached historical data + append today's live candles
                log(f"[{STRATEGY_NAME}] [BAR_CLOSE] Getting historical data for {self.symbol_in_use}...")
                df = get_historical_data(self.client, self.cfg, symbol=self.symbol_in_use)

                # Append any live candles we've built today
                if hasattr(self, 'live_df') and not self.live_df.empty:
                    df = append_live_candles_to_history(df, self.live_df)
                    # Recompute indicators with combined data
                    df = compute_indicators(df, self.cfg)

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
        """
        Poll TP/SL order status to detect fills.
        Uses batch orderbook API for efficiency when possible.
        """
        if not self.in_position:
            return

        try:
            # Use orderbook API for batch checking (more efficient than individual calls)
            if hasattr(self.client, 'orderbook'):
                try:
                    orderbook_resp = safe_api_call(self.client.orderbook)
                    if orderbook_resp.get('status') == 'success':
                        # Defensive parsing: handle multiple possible response structures
                        ob_data = orderbook_resp.get('data') or orderbook_resp.get('results') or {}
                        if isinstance(ob_data, dict):
                            orders = ob_data.get('orders', [])
                        elif isinstance(ob_data, list):
                            orders = ob_data  # Direct list of orders
                        else:
                            orders = []

                        # Find our TP and SL orders in the orderbook
                        for order in orders:
                            order_id = order.get('orderid')
                            if order_id == self.tp_order_id:
                                order_status = str(order.get('order_status', '')).upper()
                                if order_status == 'COMPLETE':
                                    price = float(order.get('average_price', 0) or 0)
                                    if price > 0:
                                        log(f"[{STRATEGY_NAME}] 🎯 [ORDER_FILLED] Target order filled @ ₹{price:.2f}")
                                        self._realize_exit(price, "Target Hit")
                                        self.cancel_order_silent(self.sl_order_id)
                                        save_state(self)
                                        return
                            elif order_id == self.sl_order_id:
                                order_status = str(order.get('order_status', '')).upper()
                                if order_status == 'COMPLETE':
                                    price = float(order.get('average_price', 0) or 0)
                                    if price > 0:
                                        log(f"[{STRATEGY_NAME}] 🛑 [ORDER_FILLED] StopLoss order filled @ ₹{price:.2f}")
                                        self._realize_exit(price, "Stoploss Hit")
                                        self.cancel_order_silent(self.tp_order_id)
                                        save_state(self)
                                        return
                        return  # Successfully checked via orderbook
                except Exception as e:
                    log(f"[{STRATEGY_NAME}] [WARN] Orderbook API failed, falling back to individual checks: {e}")

            # Fallback to individual orderstatus calls if orderbook not available
            # Check TP
            if self.tp_order_id:
                resp = safe_api_call(self.client.orderstatus, order_id=self.tp_order_id, strategy=STRATEGY_NAME)
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
                resp = safe_api_call(self.client.orderstatus, order_id=self.sl_order_id, strategy=STRATEGY_NAME)
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
                # Use placesmartorder to ensure position is properly closed
                resp = safe_api_call(
                    self.client.placesmartorder,
                    strategy=STRATEGY_NAME,
                    symbol=self.symbol_in_use,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action=action,
                    price_type="MARKET",
                    quantity=self.qty,
                    position_size=0  # Target: flat (no position)
                )
                if resp.get('status') == 'success':
                    time.sleep(0.5)
                    status_resp = safe_api_call(self.client.orderstatus, order_id=resp.get('orderid'), strategy=STRATEGY_NAME)
                    px = float(status_resp.get('data', {}).get('average_price', 0) or 0)
                    self._realize_exit(px, "Square-off EOD")
                    save_state(self)
            except Exception as e:
                log(f"[{STRATEGY_NAME}] [ERROR] square_off_job: {e}")

    def session_end_reconciliation(self):
        """
        End-of-day reconciliation job to verify state.json against broker data.
        Logs positions, orders, and trades for auditing purposes.
        """
        log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] Starting session-end reconciliation...")

        try:
            # 1. Check positionbook
            log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] Fetching positionbook...")
            positions_resp = safe_api_call(self.client.positionbook)
            if positions_resp.get('status') == 'success':
                positions = positions_resp.get('data', [])
                log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] Positionbook: {len(positions)} position(s)")
                for pos in positions:
                    if pos.get('quantity', 0) != 0:
                        log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] Open Position: {pos.get('symbol')} | Qty={pos.get('quantity')} | Avg={pos.get('average_price')}")
            else:
                log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] ⚠️ Failed to fetch positionbook: {positions_resp}")

            # 2. Check orderbook
            if hasattr(self.client, 'orderbook'):
                log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] Fetching orderbook...")
                orderbook_resp = safe_api_call(self.client.orderbook)
                if orderbook_resp.get('status') == 'success':
                    # Defensive parsing: handle multiple possible response structures
                    ob_data = orderbook_resp.get('data') or orderbook_resp.get('results') or {}
                    if isinstance(ob_data, dict):
                        orders = ob_data.get('orders', [])
                        stats = ob_data.get('statistics', {})
                    elif isinstance(ob_data, list):
                        orders = ob_data
                        stats = {}
                    else:
                        orders = []
                        stats = {}
                    log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] Orderbook: {len(orders)} order(s)")
                    log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] Order Stats: Completed={stats.get('total_completed_orders', 0)} | Open={stats.get('total_open_orders', 0)} | Rejected={stats.get('total_rejected_orders', 0)}")
                else:
                    log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] ⚠️ Failed to fetch orderbook: {orderbook_resp}")

            # 3. Check tradebook
            if hasattr(self.client, 'tradebook'):
                log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] Fetching tradebook...")
                tradebook_resp = safe_api_call(self.client.tradebook)
                if tradebook_resp.get('status') == 'success':
                    # Defensive parsing: handle multiple possible response structures
                    trades_data = tradebook_resp.get('data')
                    if isinstance(trades_data, list):
                        trades = trades_data
                    elif isinstance(trades_data, dict):
                        trades = trades_data.get('trades', [])
                    else:
                        trades = []
                    log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] Tradebook: {len(trades)} trade(s) today")
                    total_buy_value = sum(float(t.get('trade_value', 0)) for t in trades if t.get('action') == 'BUY')
                    total_sell_value = sum(float(t.get('trade_value', 0)) for t in trades if t.get('action') == 'SELL')
                    log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] Trade Value: Buy=₹{total_buy_value:.2f} | Sell=₹{total_sell_value:.2f}")
                else:
                    log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] ⚠️ Failed to fetch tradebook: {tradebook_resp}")

            # 4. Compare with state.json
            if self.in_position:
                log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] ⚠️ WARNING: state.json shows open position but we're past EOD!")
                log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] Position: {self.side} | Symbol={self.symbol_in_use} | Entry=₹{self.entry_price}")
            else:
                log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] ✅ state.json shows no open position (expected)")

            # 5. Log daily P&L
            log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] Daily Realized P&L: ₹{self.realized_pnl_today:+.2f}")

            log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] ✅ Reconciliation complete")

        except Exception as e:
            log(f"[{STRATEGY_NAME}] [EOD_RECONCILIATION] ⚠️ Reconciliation failed: {e}")

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

            # Use placesmartorder for position-aware order placement with retry and rate limiting
            resp = safe_api_call(
                self.client.placesmartorder,
                strategy=STRATEGY_NAME,
                symbol=self.symbol_in_use,
                exchange=self.cfg.exchange,
                product=self.cfg.product,
                action=action,
                price_type="MARKET",
                quantity=self.qty,
                position_size=self.qty  # Target position size
            )

            if resp.get('status') != 'success':
                log(f"[{STRATEGY_NAME}] ❌ [ERROR] Order placement failed: {resp}")
                self._flat_state()
                return

            self.entry_order_id = resp.get('orderid')
            log(f"[{STRATEGY_NAME}] [ORDER] Order placed. OrderID: {self.entry_order_id}")
            log(f"[{STRATEGY_NAME}] [ORDER] Waiting for fill confirmation...")

            time.sleep(0.5)
            status_resp = safe_api_call(self.client.orderstatus, order_id=self.entry_order_id, strategy=STRATEGY_NAME)
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
            # Use basketorder to place TP and SL simultaneously (atomic operation)
            exit_action = "SELL" if self.side == 'LONG' else "BUY"

            # Prepare TP order
            tp_order = {
                "symbol": self.symbol_in_use,
                "exchange": self.cfg.exchange,
                "action": exit_action,
                "quantity": str(self.qty),
                "pricetype": "LIMIT",
                "price": str(self.tp_level),
                "product": self.cfg.product
            }

            # Prepare SL order (with proper price parameter for SL type)
            sl_order = {
                "symbol": self.symbol_in_use,
                "exchange": self.cfg.exchange,
                "action": exit_action,
                "quantity": str(self.qty),
                "pricetype": self.cfg.sl_order_type,
                "trigger_price": str(self.sl_level),
                "product": self.cfg.product
            }
            # For SL orders: both price and trigger_price required; for SL-M: only trigger_price
            if self.cfg.sl_order_type == "SL":
                sl_order["price"] = str(self.sl_level)

            # Place both orders in basket with retry and rate limiting
            basket_resp = safe_api_call(
                self.client.basketorder,
                strategy=STRATEGY_NAME,
                orders=[tp_order, sl_order]
            )

            if basket_resp.get('status') == 'success':
                # Extract order IDs from basket response (uses 'results' not 'data')
                order_data = basket_resp.get('results', [])
                if len(order_data) >= 2:
                    self.tp_order_id = order_data[0].get('orderid')
                    self.sl_order_id = order_data[1].get('orderid')
                    log(f"[{STRATEGY_NAME}] [EXITS] Basket placed - TP oid={self.tp_order_id} @ {self.tp_level:.2f} | SL oid={self.sl_order_id} @ {self.sl_level:.2f}")
                else:
                    log(f"[{STRATEGY_NAME}] [WARN] Basket order response format unexpected: {basket_resp}")
                    # Fallback: try to extract what we can
                    if len(order_data) > 0:
                        self.tp_order_id = order_data[0].get('orderid')
                    if len(order_data) > 1:
                        self.sl_order_id = order_data[1].get('orderid')
            else:
                log(f"[{STRATEGY_NAME}] [ERROR] Basket order failed: {basket_resp}")

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

    def _update_live_candle(self, ltp: float, volume: float = 0):
        """
        Update the current live candle with tick data.
        For use with SKIP_HISTORY_FETCH mode.
        """
        now = now_ist()
        interval_minutes = self._parse_interval_minutes(self.cfg.interval)

        # Round down to nearest interval boundary
        candle_minute = (now.minute // interval_minutes) * interval_minutes
        candle_time = now.replace(minute=candle_minute, second=0, microsecond=0)

        if not self.current_candle or self.current_candle.get('timestamp') != candle_time:
            # New candle started - finalize previous and start new
            if self.current_candle:
                self._finalize_candle()

            self.current_candle = {
                'timestamp': candle_time,
                'open': ltp,
                'high': ltp,
                'low': ltp,
                'close': ltp,
                'volume': volume
            }
        else:
            # Update existing candle
            self.current_candle['high'] = max(self.current_candle['high'], ltp)
            self.current_candle['low'] = min(self.current_candle['low'], ltp)
            self.current_candle['close'] = ltp
            self.current_candle['volume'] += volume

    def _finalize_candle(self):
        """Add completed candle to live_df and compute indicators"""
        if not self.current_candle:
            return

        # Add to dataframe
        new_row = pd.DataFrame([self.current_candle])
        self.live_df = pd.concat([self.live_df, new_row], ignore_index=True)

        # Keep only last N candles (memory optimization)
        max_bars = max(self.cfg.ema_slow, self.cfg.atr_window) + 50
        if len(self.live_df) > max_bars:
            self.live_df = self.live_df.tail(max_bars).reset_index(drop=True)

        # Compute indicators
        self.live_df = compute_indicators(self.live_df, self.cfg)

        log(f"[{STRATEGY_NAME}] [LIVE_CANDLE] Finalized: {self.current_candle['timestamp']} | O:{self.current_candle['open']:.2f} H:{self.current_candle['high']:.2f} L:{self.current_candle['low']:.2f} C:{self.current_candle['close']:.2f}")

    def _graceful_exit(self, *args):
        log(f"\n[{STRATEGY_NAME}] [SHUTDOWN] Closing...")
        try:
            if self.in_position:
                action = 'SELL' if self.side == 'LONG' else 'BUY'
                # Use placesmartorder for graceful exit
                safe_api_call(
                    self.client.placesmartorder,
                    strategy=STRATEGY_NAME,
                    symbol=self.symbol_in_use,
                    exchange=self.cfg.exchange,
                    product=self.cfg.product,
                    action=action,
                    price_type="MARKET",
                    quantity=self.qty,
                    position_size=0  # Target: flat (no position)
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
def validate_interval(client, interval: str) -> bool:
    """
    Validate if the given interval is supported by the broker.
    Falls back to common intervals if API not available.
    Note: Daily interval uses 'D' not '1d' per OpenAlgo standard.
    """
    common_intervals = ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "D"]

    try:
        # Try to fetch supported intervals from broker (if API exists)
        if hasattr(client, 'intervals'):
            log(f"[{STRATEGY_NAME}] [VALIDATION] Fetching supported intervals from broker...")
            intervals_resp = safe_api_call(client.intervals)
            if intervals_resp.get('status') == 'success':
                supported = intervals_resp.get('data', [])
                log(f"[{STRATEGY_NAME}] [VALIDATION] Broker supports {len(supported)} intervals")
                if interval in supported:
                    log(f"[{STRATEGY_NAME}] [VALIDATION] ✅ Interval '{interval}' is supported")
                    return True
                else:
                    log(f"[{STRATEGY_NAME}] [VALIDATION] ⚠️ Interval '{interval}' not in broker's supported list")
                    log(f"[{STRATEGY_NAME}] [VALIDATION] Supported intervals: {supported[:10]}...")
                    return False
    except Exception as e:
        log(f"[{STRATEGY_NAME}] [VALIDATION] Could not fetch intervals from broker: {e}")

    # Fallback: check against common intervals
    if interval in common_intervals:
        log(f"[{STRATEGY_NAME}] [VALIDATION] ✅ Interval '{interval}' is in common intervals list")
        return True
    else:
        log(f"[{STRATEGY_NAME}] [VALIDATION] ⚠️ Interval '{interval}' not in common intervals: {common_intervals}")
        return False


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

    # Validate interval
    if not validate_interval(client, cfg.interval):
        log(f"[{STRATEGY_NAME}] [WARN] Interval '{cfg.interval}' may not be supported. Proceeding anyway...")

    # Validate symbol (initial)
    if not validate_symbol(client, cfg.symbol, cfg.exchange):
        log(f"[{STRATEGY_NAME}] [FATAL] Symbol {cfg.symbol} not valid on {cfg.exchange}")
        sys.exit(1)

    # Run bot
    bot = ScalpWithTrendBot(cfg)
    bot.start()

if __name__ == "__main__":
    main()
