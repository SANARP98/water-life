#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Template - Copy this file to create your own strategy

This template shows the required structure for auto-discovery by the web server.

REQUIRED:
1. STRATEGY_METADATA dictionary
2. Config dataclass
3. ScalpWithTrendBot class (or your bot class name)
4. main() function for standalone execution
"""

from __future__ import annotations
import os
import sys
import time
import signal
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple
from dotenv import load_dotenv
import pytz

load_dotenv()

# OpenAlgo SDK
try:
    import openalgo
except Exception:
    print("[FATAL] openalgo library required")
    raise

IST = pytz.timezone("Asia/Kolkata")

# ==================== REQUIRED: Strategy Metadata ====================
# This tells the auto-discovery system about your strategy
STRATEGY_METADATA = {
    "name": "My Custom Strategy",  # Display name in UI
    "description": "Description of what this strategy does",
    "version": "1.0",
    "features": [
        "Feature 1",
        "Feature 2",
        "Feature 3"
    ],
    "has_trade_direction": True,  # Does it support trade direction control?
    "author": "Your Name"
}

# Optional: Define lot sizes if using hardcoded values (otherwise use API)
INDEX_LOT_SIZES = {
    "NIFTY": 75,
    "BANKNIFTY": 35,
    # Add more as needed
}

# ==================== REQUIRED: Configuration ====================
@dataclass
class Config:
    # Connection
    api_key: str = os.getenv("OPENALGO_API_KEY", "")
    api_host: str = os.getenv("OPENALGO_API_HOST", "https://api.openalgo.in")
    ws_url: Optional[str] = os.getenv("OPENALGO_WS_URL")

    # Instrument
    symbol: str = os.getenv("SYMBOL", "NIFTY")
    exchange: str = os.getenv("EXCHANGE", "NSE_INDEX")
    product: str = os.getenv("PRODUCT", "MIS")
    lots: int = int(os.getenv("LOTS", 1))

    # Strategy parameters
    interval: str = os.getenv("INTERVAL", "5m")

    # Risk parameters
    target_points: float = float(os.getenv("TARGET_POINTS", 10.0))
    stoploss_points: float = float(os.getenv("STOPLOSS_POINTS", 5.0))
    trade_direction: str = os.getenv("TRADE_DIRECTION", "both").lower()  # long, short, or both

    # Other parameters
    daily_loss_cap: float = float(os.getenv("DAILY_LOSS_CAP", -1000.0))
    enable_eod_square_off: bool = os.getenv("ENABLE_EOD_SQUARE_OFF", "true").lower() == "true"
    square_off_time: Tuple[int, int] = (15, 25)

# ==================== REQUIRED: Bot Class ====================
class ScalpWithTrendBot:
    """
    Main trading bot class - MUST be named 'ScalpWithTrendBot' for auto-discovery
    (or update the discovery function to look for a different class name)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = openalgo.api(api_key=cfg.api_key, host=cfg.api_host, ws_url=cfg.ws_url)

        # State variables
        self.in_position = False
        self.side = None
        self.entry_price = None
        self.tp_level = None
        self.sl_level = None
        self.pending_signal = None
        self.realized_pnl_today = 0.0

        # Resolve quantity
        self.qty = self._resolve_quantity()

    def _resolve_quantity(self) -> int:
        """Calculate position size"""
        # Option 1: Use hardcoded lot sizes
        if self.cfg.symbol.upper() in INDEX_LOT_SIZES:
            return self.cfg.lots * INDEX_LOT_SIZES[self.cfg.symbol.upper()]

        # Option 2: Use OpenAlgo API (if available)
        # instruments = self.client.symbols(exchange=self.cfg.exchange)
        # for inst in instruments:
        #     if inst['symbol'].upper() == self.cfg.symbol.upper():
        #         return self.cfg.lots * int(inst.get("lot_size", 1))

        # Default
        return self.cfg.lots * 50

    def start(self):
        """Main entry point - implement your strategy logic here"""
        print(f"[{STRATEGY_METADATA['name']}] Bot starting...")
        print(f"Symbol: {self.cfg.symbol}, Qty: {self.qty}")

        # TODO: Implement your strategy logic
        # - Connect to websocket
        # - Subscribe to data feeds
        # - Implement signal generation
        # - Place orders
        # - Handle order updates
        # - Risk management

        # Example: Keep alive loop
        try:
            while True:
                time.sleep(1)
                # Your logic here

        except KeyboardInterrupt:
            self._graceful_exit()

    def _graceful_exit(self, *args):
        """Cleanup on shutdown"""
        print(f"[{STRATEGY_METADATA['name']}] Shutting down...")
        # Close positions, cancel orders, disconnect websocket, etc.
        sys.exit(0)

# ==================== REQUIRED: Main Function ====================
def main():
    cfg = Config()

    # Validation
    if not cfg.api_key:
        print(f"[{STRATEGY_METADATA['name']}] [FATAL] Please set OPENALGO_API_KEY")
        sys.exit(1)

    if cfg.trade_direction not in ["long", "short", "both"]:
        print(f"[{STRATEGY_METADATA['name']}] [FATAL] TRADE_DIRECTION must be 'long', 'short', or 'both'")
        sys.exit(1)

    # Start bot
    bot = ScalpWithTrendBot(cfg)
    bot.start()

if __name__ == "__main__":
    main()
