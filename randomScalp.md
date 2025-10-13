# randomScalp — OpenAlgo Live Trading Bot (IST)

> Version: **1.0**
> Strategy: **Random Scalp (Long-only)**
> Runtime: **Python 3.x**, **Asia/Kolkata (IST)**, **APScheduler**
> Broker API: **OpenAlgo** (HTTP + optional WS)

---

## What this bot does

`randomScalp.py` converts your backtest logic into a **tradable** OpenAlgo strategy. It:

* **Queues a BUY** signal on **every Nth bar close** (configurable).
* **Executes the entry** at the **next bar open** (market order).
* Immediately places **OCO-style exits**:

  * **Target** (SELL **LIMIT**) at `entry + ₹profit_target_rupees`
  * **Stop** (SELL **SL-M**) at `entry − ₹stop_loss_rupees`
  * **Sibling cancellation**: when one exit fills, the other is cancelled.
* **Squares off intraday** at a configurable IST time (default **15:15**), cancelling exits **before** the forced close.
* Prints the banner `🔁 OpenAlgo Python Bot is running.` on start and **prints quotes/LTP immediately** when fetched.
* Uses **start_date / end_date** for any historical backfill (no timedeltas) and returns a **pandas.DataFrame**.
* Respects **IST** session windows via APScheduler; supports **intraday** and special handling for **daily** intervals.

This implementation is structured for **production** with safety rails: entry idempotency, OCO polling, square‑off gating, misfire/overlap handling, and tick‑size rounding.

---

## High-level flow

1. **Bar Close (signal stage)**

   * If **in position**, do nothing.
   * Compute next bar’s open time; **skip** if that time is **at/after square‑off**.
   * Set `pending_signal=True`, store `next_entry_time`.

2. **Next Bar Open (execution stage)**

   * If there’s a pending signal and we’re flat, **place a BUY MARKET** order.
   * After fill, compute TP/SL levels, **rounded to tick size**, and place exit legs:

     * TP: **LIMIT**
     * SL: **SL‑M** (with broker‑safe fallback logic recommended if SL‑M not supported)

3. **OCO Polling (every 5s)**

   * Check `order_status` for TP/SL. If **COMPLETE**, cancel sibling and **realize P&L**.
   * Handles **REJECT**/**PARTIAL** gracefully; logs and keeps sibling as appropriate.

4. **Square‑Off / Shutdown**

   * **Cancel exits first**, then **send MARKET** to flatten.
   * On restart, optional warm‑start reconciliation is supported in the codebase pattern.

---

## Configuration

Environment-driven with safe defaults. Example `.env`:

```env
# --- OpenAlgo ---
OPENALGO_API_KEY=your_api_key_here
OPENALGO_API_HOST=http://127.0.0.1:5000
OPENALGO_WS_URL=ws://127.0.0.1:8765

# --- Instrument ---
EXCHANGE=NFO
SYMBOL=NIFTY24OCT2524500CE     # Use an actual, live tradable symbol
PRODUCT=MIS                     # MIS | CNC | NRML
LOTS=1                          # multiplier over exchange lot size

# --- Engine ---
INTERVAL=5m                     # 1m/3m/5m/15m/60m/D

# --- Strategy params ---
TRADE_EVERY_N_BARS=1
PROFIT_TARGET_RUPEES=2.0
STOP_LOSS_RUPEES=1.0
BROKERAGE_PER_TRADE_RUPEES=0.0
SLIPPAGE_RUPEES=0.0

IGNORE_ENTRY_DELTA=true
ENABLE_EOD_SQUARE_OFF=true

# --- Ops (toggle as needed) ---
TEST_MODE=false                 # true uses openalgo.simulator
LOG_TO_FILE=false               # set true only if you want file logs
PERSIST_STATE=false             # set true only if you want JSON state file

# --- Optional history backfill ---
USE_HISTORY=false
HISTORY_START_DATE=2025-01-01
HISTORY_END_DATE=2025-01-31
```

### Key fields

* **EXCHANGE**: one of `NSE, NFO, BSE, BFO, CDS, BCD, MCX, NCDEX`.
* **PRODUCT**: `MIS` (intraday), `CNC` (equity delivery), `NRML` (F&O carry).
* **INTERVAL**: intraday (`Xm`) or daily (`D`). For daily, the bot schedules a single open/close.
* **LOTS**: multiplier; final quantity = `LOTS × lotsize`. Lotsize is resolved via `search()`; if unavailable, the bot falls back to **January‑2025 index lots** (e.g., `NIFTY=75`, `BANKNIFTY=35`, `FINNIFTY=65`, `MIDCPNIFTY=75`, `NIFTYNXT50=25`; BSE indexes as configured). Minimum value is 1.

> **Note:** For daily history, the code normalizes `interval="D"` to match OpenAlgo’s daily convention.

---

## Order constants used

* **Action**: `BUY`, `SELL`
* **Price Type**: `MARKET`, `LIMIT`, `SL‑M`
* **Exchange**: `NSE`, `NFO`, `BSE`, `BFO`, `CDS`, `BCD`, `MCX`, `NCDEX`
* **Product**: `MIS`, `CNC`, `NRML`

These match OpenAlgo’s common constants and are compatible with supported brokers.

---

## Scheduling & session

* Scheduler: **APScheduler** with `timezone=IST`, job defaults:
  `max_instances=1`, `coalesce=True`, `misfire_grace_time=30`.
* Session window (default): **09:20 → 15:15** IST. Signals and entries only occur within windows.
* Daily interval: one **open** tick near session start, one **close** tick near session end.

---

## Risk & safety rails (v1.2 - Production Hardened)

* **No re‑entry while in position** (signals are suppressed while long).
* **Square‑off gate**: if the *next* entry would occur at/after square‑off time, the signal is skipped.
* **OCO enforcement with race protection**: TP/SL polled every 5s with thread lock; handles both fills simultaneously.
* **Partial fill handling**: Entry/exit orders track actual `filled_quantity`; exits sized to actual fills only.
* **Idempotent order placement**: `_safe_placeorder()` with timeout handling prevents duplicate orders on network issues.
* **SL-M trigger validation**: Checks trigger_price < LTP before placement; auto-fallback to SL if SL-M unsupported.
* **Partial exit synchronization**: Monitors TP/SL for partial fills; adjusts sibling quantities dynamically.
* **Exit legs retry**: If exit placement fails, retries up to 3 times with tracking.
* **Entry price retry**: Polls average_price with progressive backoff (max 5 attempts).
* **Forced exits** (EOD/shutdown): **cancel exits first**, then send market close with escalation protocol (0.25s polling → retry → 30s keep-alive).
* **Three-axis reconciliation**: Checks direction, quantity, AND avg_price every 30s; re-arms exits if missing.
* **Child order cleanup**: Removes stale TP/SL when flat; handles rejected orders to prevent retry loops.
* **Market-on-target** (optional): Converts TP LIMIT to MARKET when LTP >= target (handles gap scenarios).
* **Tick‑size rounding**: TP/SL levels rounded to discovered **tick size** (with exchange-specific fallbacks) to minimize rejects.
* **State recovery**: Handles both TP and SL filled during downtime, with corrective orders.
* **Enhanced logging**: One-line state summaries + critical warnings for unprotected positions.

---

## Historical data

* Uses `client.history(symbol, exchange, interval, start_date, end_date)`.
* Always returns a **pandas.DataFrame**; timestamps are coerced to **Asia/Kolkata** and sorted.
* For daily: the interval is normalized to **"D"**.

---

## Running the bot

```bash
python randomScalp.py
```

Console banner:

```
🔁 OpenAlgo Python Bot is running.
```

You’ll also see immediate LTP prints whenever quotes are fetched, plus concise trade/state logs. If you enable file logging, a `random_scalp_live.log` will be created.

### Simulator mode

Set `TEST_MODE=true` to use `openalgo.simulator(...)` without touching the live broker.

---

## Symbol format (quick primer)

* Options typically look like: `NIFTY28MAR2420800CE`
  `[INDEX][DD][MON][YY][Strike][CE|PE]`
* The bot validates tradability via `quotes()` first, then `search()` as a fallback. Lotsize and tick size are picked up when available.

> Refer to your **OpenAlgo Symbol Format** doc for full details and special cases (weekly/monthly expiries, underscores, etc.).

---

## Troubleshooting

* **Orders getting rejected (SL‑M not supported)**: Switch to `SL` with `price = trigger ± tick` as per direction (code supports this variant if enabled).
* **No fills**: Verify symbol validity: `EXCHANGE`, `SYMBOL`, trading hours, and lot multiples.
* **Unexpected re‑entries**: Confirm `TRADE_EVERY_N_BARS`, session windows, and that `in_position` isn’t being reset by external code.
* **Daily mode triggers too often**: Ensure `INTERVAL=D` (or synonyms) and check the daily scheduling branch.

---

## Extending the bot

* **Add indicators**: Prefer **OpenAlgo indicators** (`openalgo` TA modules). You can also opt into `ta-lib` or `pandas_ta` if you explicitly choose to—this bot currently **does not** require indicators.
* **Depth printing**: If you fetch market depth, print it immediately to stay within your house rule (quotes are already printed).
* **Database**: If/when you need persistence, use **SQLAlchemy** (per your policy). The current bot writes no DB by default.
* **Scheduling**: To run timed entries (e.g., first 30 minutes only), adjust `session_windows`.

---

## Quick API call map (OpenAlgo)

* `api = openalgo.api(api_key, host, ws_url)` or `openalgo.simulator(...)`
* `api.quotes(symbol, exchange) → { status, data:{ ltp, ... } }`
* `api.search(query, exchange) → { status, data:[ { symbol, lotsize, tick_size, ... } ] }`
* `api.placeorder(strategy, symbol, exchange, product, action, price_type, quantity, [price], [trigger_price]) → { status, orderid }`
* `api.orderstatus(order_id, strategy) → { status, data:{ order_status, average_price, ... } }`
* `api.cancelorder(order_id[, strategy]) → { status, ... }`
* `api.history(symbol, exchange, interval, start_date, end_date) → pandas.DataFrame`

---

## Changelog

**1.2** (Current - Production Hardened)

* **Real-World Trading Edge Cases:**
  * **Partial Fill Handling:** Entry and exit orders track actual filled_quantity vs requested; exits sized to actual fills only
  * **Idempotent Order Placement:** `_safe_placeorder()` with timeout handling, retry logic, and duplicate order prevention
  * **SL-M Trigger Validation:** Validates trigger_price < LTP before SL-M placement; auto-fallback to SL if unsupported
  * **Partial Exit Sync:** Monitors TP/SL for partial fills; dynamically adjusts sibling quantities via `_sync_exit_quantities()`
  * **Three-Axis Reconciliation:** Compares direction, quantity, AND avg_price from broker; re-arms exits if missing
  * **Child Order Cleanup:** Removes stale TP/SL when flat; handles rejected orders to prevent retry loops
  * **Enhanced Graceful Exit:** 0.25s polling (20 iterations = 5s) → retry MARKET → keep alive 30s with reconciliation
  * **Market-on-Target:** Optional conversion of TP LIMIT to MARKET when LTP >= target (handles gap scenarios)
* **New Config Parameters:**
  * `API_TIMEOUT_SECONDS` (default: 10) - timeout for API calls
  * `MAX_ORDER_RETRIES` (default: 2) - max retry attempts for order placement
  * `ENABLE_MARKET_ON_TARGET` (default: false) - convert TP to MARKET on gap
* **Enhanced Logging:**
  * One-line state summaries: `STATE=LONG qty=75 entry=100.50 tp=102.50 sl=99.50`
  * Critical warnings for unprotected positions (rejected TP/SL)
  * Detailed partial fill logging
* **New Helper Methods:**
  * `_safe_placeorder()` - idempotent with timeout
  * `_place_stop_order()` - SL-M with SL fallback
  * `_sync_exit_quantities()` - partial fill adjustment
  * `_cleanup_stale_orders()` - reconciliation cleanup
  * `_ensure_exits()` - re-arm protection if missing
  * `place_exit_legs_for_qty()` - quantity-aware exit placement

**1.1**

* **Critical fixes:**
  * Fixed OCO race condition using thread lock to prevent both TP and SL from executing simultaneously
  * Added retry logic for exit legs placement with progressive backoff
  * Fixed entry price polling with retry mechanism to prevent unprotected positions
  * Fixed state recovery to handle both exits filled during downtime
  * Added exit confirmation in graceful shutdown with timeout
* **Improvements:**
  * Fixed lot size validation with minimum checks (prevents LOTS < 1)
  * Improved tick size fallback mechanism by exchange type
  * Fixed daily interval parsing (now returns 1440 minutes instead of 60)
  * Fixed slippage modeling to per-leg basis (more accurate cost calculation)
  * Enhanced square-off logic with stale order cleanup
  * Added position reconciliation every 30 seconds to detect state mismatches
  * Updated MIDCPNIFTY lot size from 140 to 75 (correct as of Jan 2025)

**1.0**

* Initial production release of `randomScalp.py`: intraday/daily scheduling, OCO exits with polling and sibling cancellation, square‑off gating, tick‑size rounding, history normalization, and IST-safe scheduling.

---

## Support

* Docs: [https://docs.openalgo.in](https://docs.openalgo.in)
* Community: [https://openalgo.in/discord](https://openalgo.in/discord)
