# RandomScalp.py - Fixes Applied (v1.1)

## Summary
All 10 identified critical and medium issues have been fixed using OpenAlgo best practices.

---

## Critical Issues Fixed

### 1. OCO Race Condition (Lines 476-535)
**Problem:** Both TP and SL orders could fill simultaneously before either is cancelled, creating unintended short positions.

**Fix:**
- Added `threading.Lock()` for OCO safety (`self.exit_lock`)
- Check both TP and SL status atomically before processing
- Detect and handle both-filled scenario with corrective orders
- Added non-blocking lock acquisition to prevent job blocking

**Code Location:** `check_order_status()` method

---

### 2. Entry Price Failure Recovery (Lines 537-590)
**Problem:** If `average_price` unavailable after 0.5s sleep, bot aborts exit legs but keeps position UNPROTECTED.

**Fix:**
- Implemented progressive backoff retry (5 attempts, 0.3s → 1.5s)
- Added `exit_legs_placed` flag to track successful placement
- Retry logic in `check_order_status()` polling loop
- Added `exit_legs_retry_count` with max 3 retries
- Critical warning if max retries exceeded

**Code Location:** `place_entry()` and `check_order_status()` methods

---

### 3. Lot Size Validation (Lines 216-251)
**Problem:** When qty < base_lot_size, rounds up which could execute larger positions than intended.

**Fix:**
- Added explicit check for `cfg.lots < 1` with warning
- Use `max(cfg.lots, 1)` in fallback paths
- Added validation in `resolve_quantity()` with clear logging
- Updated MIDCPNIFTY lot size from 140 → 75 (correct Jan 2025 value)

**Code Location:** `resolve_quantity()` function

---

## Medium Issues Fixed

### 4. Daily Interval Parsing (Lines 351-361)
**Problem:** Daily interval returned 60 minutes (confusing), used incorrectly in next_time calculation.

**Fix:**
- Changed to return 1440 minutes (24 hours) for clarity
- Added comment that daily mode uses explicit cron jobs, not this value
- Prevents confusion in bar timing calculations

**Code Location:** `_parse_interval_minutes()` method

---

### 5. Square-off State Inconsistency (Lines 694-747)
**Problem:** Attempts to cancel TP/SL when not in position, suggesting state corruption scenarios.

**Fix:**
- Added stale order cleanup with explicit None assignment
- Enhanced square-off to get actual exit price from orderstatus
- Fallback chain: orderstatus → quotes → entry_price
- Added logging for state cleanup

**Code Location:** `square_off()` method

---

### 6. Tick Size Fallback (Lines 118-127, 216-251)
**Problem:** Hardcoded 0.05 fallback incorrect for many instruments.

**Fix:**
- Added `TICK_SIZE_FALLBACKS` dict by exchange:
  - NSE/NFO: 0.05
  - BSE: 0.01
  - MCX: 1.0
  - CDS: 0.0025
  - BCD: 0.0001
- Used exchange-specific fallback in `resolve_quantity()`

**Code Location:** Constants and `resolve_quantity()` function

---

### 7. State Recovery Both Exits Filled (Lines 773-818)
**Problem:** Only first checked order's exit realized; other filled order's P&L lost.

**Fix:**
- Check both TP and SL status before processing
- Detect both-filled scenario with critical warning
- Use more favorable price (TP), send corrective order
- Call `_ensure_flat_position()` to reconcile broker position

**Code Location:** `_load_state()` method

---

### 8. Graceful Exit Confirmation (Lines 909-972)
**Problem:** Market close order sent without confirmation; position could remain if rejected.

**Fix:**
- Added 5-second timeout loop checking orderstatus
- Progressive checks every 0.5s for fill confirmation
- Fallback to LTP if orderstatus fails
- Final `_ensure_flat_position()` reconciliation
- Proper state cleanup if timeout occurs

**Code Location:** `_graceful_exit()` method

---

### 9. Slippage Per-Leg (Lines 677-692)
**Problem:** Applied fixed slippage per round-trip; should be per leg.

**Fix:**
- Split costs into entry and exit legs:
  - Entry: `brokerage + slippage/2`
  - Exit: `brokerage + slippage/2`
- More accurate cost modeling
- Maintains backward compatibility with config

**Code Location:** `_realize_exit()` method

---

### 10. Position Reconciliation (Lines 476-523, 945)
**Problem:** No verification that internal state matches broker position.

**Fix:**
- Added `reconcile_position()` method
- Runs every 30 seconds via scheduler
- Detects three scenarios:
  1. Unexpected position when flat → flatten
  2. Flat at broker but think we're in position → update state
  3. Quantity mismatch → sync to actual
- Comprehensive logging for mismatches

**Code Location:** `reconcile_position()` method and scheduler setup

---

## Additional Improvements

### Enhanced Exit Legs Placement (Lines 592-666)
- Individual try/except for TP and SL orders
- Track success of each leg separately
- Only mark `exit_legs_placed = True` if BOTH succeed
- Detailed error logging for debugging

### Updated _flat_state() (Lines 839-851)
- Reset `exit_legs_placed` and `exit_legs_retry_count`
- Ensures clean state for next position

### Documentation Updates
- Updated randomScalp.md with v1.1 changelog
- Updated lot sizes to January 2025 values
- Enhanced risk & safety rails section
- Added all fix details to changelog

---

## Testing Recommendations

1. **OCO Race Test:** Manually fill both TP and SL simultaneously to verify lock works
2. **Entry Price Retry:** Delay orderstatus response to test retry mechanism
3. **Lot Size:** Test with LOTS=0, LOTS=0.5 to verify validation
4. **State Recovery:** Stop bot mid-position and restart to test recovery
5. **Graceful Exit:** SIGINT/SIGTERM during position to test confirmation
6. **Reconciliation:** Manually close position at broker to test detection

---

## Version Info
- **Previous:** 1.0
- **Current:** 1.1
- **Lines Changed:** ~300
- **New Dependencies:** `threading.Lock` (stdlib)
- **Breaking Changes:** None (backward compatible)
