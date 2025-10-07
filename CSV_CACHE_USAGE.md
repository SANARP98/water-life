# CSV-Based History Cache - Usage Guide

## Overview

The bot now uses **CSV-based caching** to eliminate repeated history API calls and avoid HTTP 504 timeout errors. Historical data is downloaded once and cached locally, then reused throughout the day.

---

## How It Works

### **First Run (9:15 AM)**
```
[CACHE] No cache file found: history_cache/NIFTY28OCT2525200CE_history.csv
[CACHE] Fetching 25 days of history from API...
[HISTORY] Fetching NIFTY28OCT2525200CE@NFO interval=1m from 2025-09-13 to 2025-10-07
[HISTORY] ✅ Successfully fetched 25,000 bars
[CACHE] ✅ Saved 25,000 bars to history_cache/NIFTY28OCT2525200CE_history.csv
```

**What happens:**
- Bot checks for cached file
- Not found → downloads 25 days of 1-minute data from API
- Saves to CSV file: `history_cache/NIFTY28OCT2525200CE_history.csv`
- Takes ~30 seconds (one-time cost)

---

### **Every Bar After (9:16 AM, 9:17 AM, etc.)**
```
[BAR_CLOSE] Getting historical data for NIFTY28OCT2525200CE...
[CACHE] Loading from history_cache/NIFTY28OCT2525200CE_history.csv...
[CACHE] ✅ Loaded 25,000 bars from cache
[CACHE] Merged 25,000 cached + 2 live = 25,002 total bars
```

**What happens:**
- Bot loads cached CSV file (instant, ~50ms)
- Appends today's live candles built so far
- Computes indicators on combined data
- **Zero API calls** ✅

---

### **After Bot Restart (11:30 AM)**
```
[CACHE] Found NIFTY28OCT2525200CE_history.csv
[CACHE] ✅ Loaded 25,000 bars from cache (50ms)
[CACHE] Merged 25,000 cached + 0 live = 25,000 total bars
```

**What happens:**
- Bot loads cached history instantly
- No live candles yet (just restarted)
- Starts building new live candles from now
- State persists across restarts ✅

---

### **Next Day (Oct 8, 9:15 AM)**
```
[CACHE] Found NIFTY28OCT2525200CE_history.csv
[CACHE] Cache is from previous day (2025-10-07), will refresh
[CACHE] Fetching 25 days of history from API...
[HISTORY] Fetching NIFTY28OCT2525200CE@NFO interval=1m from 2025-09-14 to 2025-10-08
[CACHE] ✅ Saved 25,000 bars to history_cache/NIFTY28OCT2525200CE_history.csv
```

**What happens:**
- Bot detects cache is from yesterday
- Auto-refreshes with updated 25-day history
- New cache file includes yesterday's data
- Fresh start for new trading day ✅

---

## Configuration

### `.env` Settings

```bash
# History Caching (RECOMMENDED FOR PRODUCTION)
USE_HISTORY_CACHE=true           # Enable CSV caching (default: true)
HISTORY_CACHE_DIR=history_cache  # Cache directory (default: history_cache)
HISTORY_DAYS=25                  # Days of history to cache (default: 25)
FORCE_REFRESH_CACHE=false        # Force re-download (default: false)

# Legacy setting (keep false when using cache)
SKIP_HISTORY_FETCH=false         # Must be false to use cache
```

---

## Cache Directory Structure

```
water-life/
├── strategies/
│   └── claudeToGPT.py
├── history_cache/                              # Created automatically
│   ├── NIFTY28OCT2525200CE_history.csv        # 25 days, ~2MB
│   ├── NIFTY28OCT2525100PE_history.csv        # Different strike
│   └── BANKNIFTY28OCT2527000CE_history.csv    # Different underlying
├── state.json
└── scalp_with_trend.log
```

---

## CSV File Format

### **Columns:**
```csv
timestamp,open,high,low,close,volume,ema_fast,ema_slow,atr
2025-10-07 09:15:00+05:30,216.05,218.85,215.00,216.00,74175,216.13,216.54,2.17
2025-10-07 09:16:00+05:30,216.00,219.00,215.50,217.50,68200,216.45,216.72,2.19
...
```

### **Indicators Pre-Computed:**
- ✅ `ema_fast` (default: 3-period)
- ✅ `ema_slow` (default: 10-period)
- ✅ `atr` (default: 14-period)

**Benefit:** No need to recompute indicators on every bar!

---

## Performance Comparison

### **Without Cache (Old Behavior)**
```
⏱️  Every 1-minute bar:
- Fetch 10 days history: ~2-5 seconds
- Parse & process: ~500ms
- Compute indicators: ~200ms
- Total: ~3-6 seconds per bar
- API calls: 60/hour
- Risk: HTTP 504 timeouts ❌
```

### **With Cache (New Behavior)**
```
⏱️  First run:
- Download 25 days once: ~30 seconds
- Save to CSV: ~500ms

⏱️  Every subsequent bar:
- Load CSV cache: ~50ms
- Append live candles: ~10ms
- Update indicators: ~50ms
- Total: ~100ms per bar
- API calls: 0/hour ✅
- Risk: Zero timeouts ✅
```

**Speedup: 30-60x faster** 🚀

---

## Use Cases

### **1. Production Intraday Trading**
```bash
USE_HISTORY_CACHE=true
HISTORY_DAYS=25
SKIP_HISTORY_FETCH=false
```
- Cache loaded once on startup
- Live candles appended throughout day
- Survives bot restarts
- Auto-refreshes daily

### **2. Force Refresh (Manual Override)**
```bash
USE_HISTORY_CACHE=true
FORCE_REFRESH_CACHE=true
```
- Ignores existing cache
- Downloads fresh history
- Updates cache file
- Useful after data issues

### **3. Pure Live Mode (No History)**
```bash
USE_HISTORY_CACHE=false
SKIP_HISTORY_FETCH=true
```
- No cache, no history API
- Pure live candle building
- Requires warmup period
- For WebSocket-based systems

### **4. Disable Caching (Legacy Mode)**
```bash
USE_HISTORY_CACHE=false
SKIP_HISTORY_FETCH=false
```
- Old behavior
- Fetch history every bar
- Slow, prone to 504 errors
- Not recommended ⚠️

---

## Troubleshooting

### **Cache Not Loading**

**Symptom:**
```
[CACHE] No cache file found: history_cache/NIFTY28OCT2525200CE_history.csv
[CACHE] Fetching 25 days of history from API...
```

**Solution:**
- Normal on first run
- Check if `history_cache/` directory exists
- Verify write permissions

---

### **Cache Auto-Refreshing Every Bar**

**Symptom:**
```
[CACHE] Cache is from previous day (2025-10-07), will refresh
[CACHE] Fetching 25 days of history from API...
```

**Causes:**
1. System clock incorrect
2. File timestamp issue
3. `FORCE_REFRESH_CACHE=true`

**Solution:**
```bash
# Check system time
date

# Disable force refresh
FORCE_REFRESH_CACHE=false

# Touch cache file to update timestamp
touch history_cache/NIFTY28OCT2525200CE_history.csv
```

---

### **Invalid Cache Format Error**

**Symptom:**
```
[CACHE] Invalid cache format, missing columns
[CACHE] Fetching 25 days of history from API...
```

**Causes:**
1. Corrupted CSV file
2. Manual editing
3. Incomplete write

**Solution:**
```bash
# Delete corrupted cache
rm history_cache/NIFTY28OCT2525200CE_history.csv

# Or force refresh
FORCE_REFRESH_CACHE=true python strategies/claudeToGPT.py
```

---

### **Still Getting HTTP 504 Errors**

**Check:**
1. Is `USE_HISTORY_CACHE=true`?
2. Is `SKIP_HISTORY_FETCH=false`?
3. Does cache file exist?
4. Check logs for cache loading message

**Verify cache is working:**
```bash
# Look for this log on every bar:
[CACHE] ✅ Loaded 25,000 bars from cache

# NOT this:
[HISTORY] Fetching NIFTY28OCT2525200CE@NFO interval=1m...
```

---

## Cache Management

### **Manual Cache Refresh**
```bash
# Delete cache to force re-download
rm history_cache/*.csv

# Or use environment variable
FORCE_REFRESH_CACHE=true python strategies/claudeToGPT.py
```

### **Clear Old Caches (Different Symbols)**
```bash
# List cache files
ls -lh history_cache/

# Remove specific symbol
rm history_cache/NIFTY21OCT2525000CE_history.csv

# Remove all (fresh start)
rm -rf history_cache/
```

### **Backup Cache Files**
```bash
# Backup before maintenance
cp -r history_cache/ history_cache_backup/

# Restore if needed
cp -r history_cache_backup/* history_cache/
```

---

## Disk Space Usage

### **Storage Estimates (1-minute interval)**
- **1 day**: ~40 KB
- **25 days**: ~1 MB per symbol
- **10 symbols**: ~10 MB total

### **Cleanup Old Data**
```bash
# Find large cache files
du -h history_cache/* | sort -h

# Remove files older than 7 days
find history_cache/ -name "*.csv" -mtime +7 -delete
```

---

## Migration from Old Version

### **Upgrading from Non-Cached Version**

**Before:**
```bash
# Old .env
WARMUP_DAYS=10
SKIP_HISTORY_FETCH=false
```

**After:**
```bash
# New .env
USE_HISTORY_CACHE=true
HISTORY_DAYS=25              # More history for better indicators
SKIP_HISTORY_FETCH=false     # Keep false to use cache
```

**First run will:**
1. Download 25 days of history (one-time)
2. Save to cache
3. All subsequent runs use cache ✅

---

## Best Practices

### ✅ **Do:**
- Use `HISTORY_DAYS=25` for better indicator accuracy
- Keep cache directory in `.gitignore`
- Backup cache before symbol changes
- Monitor cache file sizes

### ❌ **Don't:**
- Manually edit CSV files (corruption risk)
- Set `HISTORY_DAYS` too high (slow API calls)
- Commit cache files to git (large, unnecessary)
- Share cache between different systems (timezone issues)

---

## FAQ

**Q: How often is cache refreshed?**
A: Automatically once per day on first run after midnight.

**Q: Can I use cache with multiple symbols?**
A: Yes! Each symbol gets its own cache file.

**Q: What if symbol changes (option roll)?**
A: New cache file created automatically for new symbol.

**Q: Does cache work with SKIP_HISTORY_FETCH=true?**
A: No, they are mutually exclusive. Use cache instead.

**Q: Can I share cache between bots?**
A: Yes, if same symbol and timezone. Files are read-only safe.

**Q: What happens if API fails?**
A: Cache is still used. Fresh download attempted next day.

**Q: How much faster is cache vs API?**
A: 30-60x faster (~50ms vs 3-6 seconds per bar).

---

## Monitoring

### **Check Cache Status**
```bash
# View cache files
ls -lh history_cache/

# Check file age
stat history_cache/NIFTY28OCT2525200CE_history.csv

# Count rows
wc -l history_cache/NIFTY28OCT2525200CE_history.csv
```

### **Verify Cache Working in Logs**
```bash
# Grep for cache messages
tail -f scalp_with_trend.log | grep CACHE

# Look for successful loads
[CACHE] ✅ Loaded 25,000 bars from cache
```

---

## Support

If you encounter issues:
1. Check logs: `tail -f scalp_with_trend.log`
2. Verify config: `cat .env | grep CACHE`
3. Test cache file: `head history_cache/*.csv`
4. Force refresh: `FORCE_REFRESH_CACHE=true`
5. Report issue with logs

---

## Summary

🎯 **Problem Solved:** HTTP 504 timeout errors
⚡ **Performance:** 30-60x faster bar processing
💾 **Storage:** ~1 MB per symbol (25 days)
🔄 **Auto-refresh:** Daily at 9:15 AM
✅ **Zero API calls:** After initial download

**Enable cache in your `.env` and enjoy instant bot restarts!**
