# Strategy Evolution: `preplexity` vs `preplexity_2` vs `preplexity_3`

This document outlines the key differences, bug fixes, and feature enhancements across the three versions of the "Scalp-with-Trend" trading bot. The files represent a clear progression from a functional strategy to a production-ready, enterprise-grade system.

---

##  เวอร์ชั่น `preplexity.py`: The Optimized Base

This file is a solid, optimized version of the initial strategy. It moves beyond basic API calls to use more advanced, position-aware methods. It serves as a strong foundation but lacks the resilience needed for handling real-world network and API issues.

### Key Features

-   **Smart Order Placement**: Uses `client.placesmartorder()` for entries, which is aware of the current position and helps prevent duplicate trades.
-   **Atomic Exits**: Implements `client.basketorder()` to place Take Profit (TP) and Stop Loss (SL) orders simultaneously. This is a major improvement for ensuring the position is always protected.
-   **Correct API Usage**: It correctly uses `client.positionbook()` instead of the deprecated `client.positions()`.
-   **Position Reconciliation**: Includes a `reconcile_position()` function that runs on startup to sync the bot's state with the broker's actual positions.
-   **Advanced Features**: It already has sophisticated features like automatic ATM option selection, history caching, and state persistence via `state.json`.

### ⚠️ Critical Issues

-   **Incorrect Lot Sizes**: Uses outdated lot sizes (e.g., `BANKNIFTY: 15`), leading to incorrect position sizing and risk management.
-   **Buggy Basket Order Parsing**: It incorrectly tries to read `basket_resp.get('data')`, which causes the TP/SL order IDs to be lost, leaving the position unprotected.
-   **No API Resilience**: Lacks any retry logic or rate limiting, making it vulnerable to temporary network or API server issues.

**Conclusion:** `preplexity.py` is a good, functional strategy but is **not safe for live trading** due to critical bugs in order handling and risk management.

---

##  เวอร์ชั่น `preplexity_2.py`: The Enterprise-Grade Leap

This version is a major upgrade, focusing heavily on making the bot resilient to API failures and more efficient. It introduces several "enterprise-grade" features for robustness, transforming the functional bot into a production-ready system.

### Key Differences from `preplexity.py`

1.  **API Retry Logic**:
    -   Introduces an `@api_retry` decorator that automatically retries failed API calls up to 3 times with **exponential backoff** (e.g., waiting 1s, then 2s, then 4s). This makes the bot resilient to temporary network glitches or API server issues.

2.  **API Rate Limiting**:
    -   Adds a global `RateLimiter` to prevent the bot from sending too many requests too quickly (e.g., more than 10 per second), which could get it temporarily blocked by the broker's API.

3.  **Safe API Call Wrapper**:
    -   All critical API calls are wrapped in a `safe_api_call()` function. This function combines the rate limiting and retry logic, and also intelligently parses API error responses to provide clearer log messages.

4.  **Randomized Polling**:
    -   The job that checks order status now runs on a randomized interval (e.g., every 2-4 seconds instead of exactly every 3 seconds). This avoids creating predictable traffic patterns that some broker systems might flag.

5.  **Batch Order Checking**:
    -   It now uses `client.orderbook()` to check the status of both the TP and SL orders in a single API call, **reducing API traffic by 50%** during position monitoring. It also has a fallback to check them individually if the batch call fails.

6.  **Session-End Reconciliation**:
    -   A new scheduled job runs 5 minutes *after* the market closes to fetch the `positionbook`, `orderbook`, and `tradebook`. It logs a complete summary for end-of-day auditing and verifies that the bot's final state matches the broker's.

7.  **Startup Validation & Dynamic Logging**:
    -   It adds functions to validate the configured `INTERVAL` against the broker's supported intervals and allows the logging verbosity (`DEBUG`, `INFO`, etc.) to be controlled via an environment variable.

### ⚠️ Critical Issues

-   **Inherited Bugs**: This version still contains the critical bugs from `preplexity.py`: incorrect lot sizes and faulty basket order parsing.

**Conclusion:** `preplexity_2.py` is a robust and resilient system, but it is **still unsafe for live trading** because it carries over the core risk management and order handling bugs.

---

##  เวอร์ชั่น `preplexity_3.py`: The Polished & Corrected Finalist

This version builds upon `preplexity_2.py` by fixing the critical data and logic errors, making it fully compliant with the latest OpenAlgo API specifications and market standards. It is the only version recommended for live trading.

### Key Differences from `preplexity_2.py`

1.  🔴 **CRITICAL FIX: Correct Lot Sizes (May 2025)**:
    -   The `INDEX_LOT_SIZES` dictionary was updated to reflect the correct, most recent lot sizes for indices like `BANKNIFTY` (from 15 to **35**), `FINNIFTY` (from 25 to **65**), and `MIDCPNIFTY` (from 50 to **140**).
    -   **This is the most critical change**, as the previous versions were trading with incorrect quantities, leading to improper risk management.

2.  🔴 **CRITICAL FIX: Basket Order Response Parsing**:
    -   A bug was fixed where the code was looking for the `'data'` key in the `basketorder` response, but the OpenAlgo API returns the order IDs in a `'results'` key.
    -   **This was a silent but critical failure**, as it meant the TP and SL order IDs were being lost, leaving the position unprotected. `preplexity_3.py` corrects this.

3.  🟡 **IMPORTANT FIX: Daily Interval Normalization**:
    -   The OpenAlgo API expects `"D"` for a daily interval, not `"1d"`. This version adds a `normalize_interval()` function to ensure all history requests for daily data are correctly formatted, fixing errors in daily backtesting or trading.

4.  **Defensive API Response Parsing**:
    -   The code that reads data from `orderbook()` and `tradebook()` was made more defensive. It can now handle multiple possible response structures (e.g., data in a list vs. a dictionary), preventing errors if different brokers have slightly different API implementations.

**Conclusion:** `preplexity_3.py` contains all the enterprise-grade features of the previous version **plus critical bug fixes** that make it safe and reliable for live trading.

---

## Evolution Summary

| Version | Focus | Key Characteristic | Live Trading Safe? |
| :--- | :--- | :--- | :--- |
| **`preplexity.py`** | **Functionality** | A working, optimized strategy using smart/basket orders. | ❌ **No** (Critical Bugs) |
| **`preplexity_2.py`** | **Robustness** | Adds enterprise-grade resilience: retries, rate limiting, and better monitoring. | ❌ **No** (Critical Bugs) |
| **`preplexity_3.py`** | **Correctness** | Fixes critical data bugs (lot sizes, API responses) and refines compliance. | ✅ **Yes** |

---

## Feature Comparison Matrix

| Feature | `preplexity.py` | `preplexity_2.py` | `preplexity_3.py` |
|:---|:---:|:---:|:---:|
| **Core Logic** | | | |
| Smart Order Entry (`placesmartorder`) | ✅ | ✅ | ✅ |
| Atomic Exits (`basketorder`) | ✅ | ✅ | ✅ |
| Position Reconciliation (Startup) | ✅ | ✅ | ✅ |
| ATM Option Selector | ✅ | ✅ | ✅ |
| History Caching | ✅ | ✅ | ✅ |
| State Persistence (`state.json`) | ✅ | ✅ | ✅ |
| | | | |
| **Robustness Features** | | | |
| API Retry Logic (Exponential Backoff) | ❌ | ✅ | ✅ |
| API Rate Limiting | ❌ | ✅ | ✅ |
| Safe API Call Wrapper | ❌ | ✅ | ✅ |
| Randomized Polling Interval | ❌ | ✅ | ✅ |
| Batch Order Checking (`orderbook`) | ❌ | ✅ | ✅ |
| Session-End Reconciliation | ❌ | ✅ | ✅ |
| Startup Interval Validation | ❌ | ✅ | ✅ |
| Dynamic Log Level Control | ❌ | ✅ | ✅ |
| | | | |
| **Critical Bug Fixes** | | | |
| **Correct Lot Sizes (May 2025)** | ❌ | ❌ | ✅ |
| **Correct Basket Order Parsing** | ❌ | ❌ | ✅ |
| **Correct Daily Interval ("D")** | ❌ | ❌ | ✅ |
| **Defensive API Response Parsing** | ❌ | ❌ | ✅ |

---

## Recommendation

For any live trading, **`preplexity_3.py` is the only recommended version**. It contains crucial bug fixes related to risk management and order handling that are absent in the previous files, in addition to all the enterprise-grade robustness features. Using any other version poses a significant risk of financial loss due to unprotected positions and incorrect trade sizing.

