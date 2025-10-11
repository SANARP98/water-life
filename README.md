# Scalping Strategy Web Manager

A professional web-based interface for managing and monitoring your OpenAlgo scalping trading strategy with real-time updates, paper trading mode, and comprehensive logging.

## Features

🎯 **Intuitive Web Interface** - Clean, modern UI accessible from any browser
📊 **Real-time Monitoring** - Live position tracking, P&L updates, and trade signals
📝 **Activity Logs** - Real-time streaming of all bot activities to the UI
🔄 **Paper Trading Mode** - Test your strategy without risking real capital
⚙️ **Full Configuration** - All strategy parameters adjustable through the UI
🚀 **Start/Stop Control** - Easy bot lifecycle management with one click
🔀 **Multi-Strategy Support** - Choose from multiple strategy implementations
📍 **Trade Direction Control** - Select long-only, short-only, or both (v2 strategies)

## Quick Start

### Option 1: Docker (Recommended) 🐳

**Fastest way to get started!**

```bash
# 1. Create .env file with your credentials
cat > .env << EOF
OPENALGO_API_KEY=your_api_key_here
OPENALGO_API_HOST=https://api.openalgo.in
SYMBOL=NIFTY
LOTS=2
EOF

# 2. Start with Docker Compose
docker-compose up -d

# 3. Access the UI
# Open browser: http://localhost:7777

# 4. View logs
docker-compose logs -f

# 5. Stop
docker-compose down
```

**Benefits:**
- ✅ No Python installation needed
- ✅ Consistent environment
- ✅ Easy deployment
- ✅ Automatic restarts
- ✅ Resource limits

### Option 2: Local Installation

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Configure Environment

Create a `.env` file with your OpenAlgo credentials:

```bash
OPENALGO_API_KEY=your_api_key_here
OPENALGO_API_HOST=https://api.openalgo.in
OPENALGO_WS_URL=wss://api.openalgo.in/ws  # Optional

# Optional: Set defaults for direct strategy execution
SYMBOL=NIFTY
EXCHANGE=NSE_INDEX
LOTS=2
TRADE_DIRECTION=both  # long, short, or both
```

#### 3. Start the Web Server

```bash
python3 web_server.py
```

Then open your browser: `http://localhost:7777`

#### 4. OR Run Strategies Directly

All strategies can be run standalone without the web server:

```bash
# Run original scalping strategy
python3 strategies/scalping.py

# Run enhanced v2 strategy
python3 strategies/scalping2.py

# Run Claude-corrected version
python3 strategies/scalping2_claude.py
```

Each strategy reads configuration from environment variables or `.env` file.

## Usage Guide

### Configuration Panel

1. **Strategy Selection**
   - **Scalping v1 (Original)**: Hardcoded lot sizes, proven logic, trade direction control
   - **Scalping v2 (Enhanced)**: API-based lot sizes, trade direction control
   - **Scalping v2 Claude (Corrected)**: Enhanced error handling and fixes
   - Each strategy shows description and available features
   - **All strategies support trade direction control** (long-only, short-only, or both)

2. **Trading Mode Toggle**
   - 📝 **Paper Trading** (Green): Simulates trades without real orders
   - 🔴 **Live Trading** (Red): Places real orders on your broker

3. **API Settings**
   - Enter your OpenAlgo API key and host URL
   - These can also be set via environment variables

4. **Symbol & Exchange**
   - Choose from: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, etc.
   - Select exchange: NSE_INDEX or BSE_INDEX
   - Product type: MIS (Intraday), CNC, or NRML

5. **Trade Direction** (All strategies)
   - **Both**: Take both long and short trades (default)
   - **Long Only**: Only take long positions
   - **Short Only**: Only take short positions
   - Available in all strategy versions

6. **Indicator Settings**
   - **EMA Fast**: Fast EMA period (default: 5)
   - **EMA Slow**: Slow EMA period (default: 20)
   - **ATR Window**: ATR calculation period (default: 14)
   - **ATR Min Points**: Minimum ATR to trade (filters low volatility)

7. **Risk Management**
   - **Target Points**: Take profit level in points
   - **Stoploss Points**: Stop loss level in points
   - **Daily Loss Cap**: Maximum daily loss before stopping new trades
   - **Confirm Trend at Entry**: Additional trend confirmation filter
   - **EOD Square-off**: Auto-exit positions at end of day
   - **Square-off Time**: Time to square-off positions (e.g., 15:25)

8. **Save Configuration**
   - Click "💾 Save Configuration" to apply settings
   - Settings are validated before saving

### Monitoring Panel

1. **Current Position Card**
   - Status: FLAT or IN POSITION
   - Side: LONG or SHORT
   - Entry Price, Target, Stoploss levels
   - Daily P&L (color-coded: green for profit, red for loss)

2. **Bot Controls**
   - ▶️ **Start Bot**: Begins trading with current configuration
   - ⏹️ **Stop Bot**: Gracefully stops the bot and closes connections

3. **Stats**
   - Pending Signal: Shows upcoming trade signals
   - Last Update: Timestamp of last status update

### Activity Logs

- Real-time streaming of all bot activities
- Color-coded log levels (Info, Warning, Error)
- Auto-scrolling to latest entries
- Clear logs button to reset display
- Terminal-style monospace font for easy reading

## Strategy Logic

The bot implements a **Scalp-with-Trend** strategy:

1. **Signal Generation** (Every 5 minutes at bar close)
   - Calculates EMA Fast and EMA Slow
   - Checks ATR filter (minimum volatility requirement)
   - **LONG Signal**: Current high > Previous high AND EMA Fast > EMA Slow
   - **SHORT Signal**: Current low < Previous low AND EMA Fast < EMA Slow

2. **Entry Execution** (Next bar open after signal)
   - Places market order at bar open following signal
   - Immediately sets TP (limit order) and SL (stop-market order)
   - OCO (One-Cancels-Other) logic: TP hit cancels SL and vice versa

3. **Risk Controls**
   - Daily loss cap prevents further trading after threshold
   - EOD square-off closes all positions before market close
   - Trend confirmation filter for additional safety

4. **Position Management**
   - Holds position until TP, SL, or EOD
   - WebSocket streams for real-time price updates
   - Automatic order status tracking

## File Structure

```
water-life/
├── strategies/           # Trading strategies package (auto-discovery enabled!)
│   ├── __init__.py      # Auto-discovery logic
│   ├── TEMPLATE.py      # Template for new strategies
│   ├── scalping.py      # Original scalping strategy
│   ├── scalping2.py     # Enhanced scalping strategy
│   └── scalping2_claude.py  # Claude-enhanced version
│   └── [your_strategy.py]   # Add your own - appears automatically!
├── web_server.py         # Flask web server and API
├── templates/
│   └── index.html        # Web UI
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Docker orchestration
├── .dockerignore         # Docker build exclusions
├── .env                  # API credentials (create this)
└── README.md            # This file
```

## Available Strategies

All strategies are **automatically discovered** from the `strategies/` folder!

### Built-in Strategies

#### Scalping v1 (Original)
- **File**: `strategies/scalping.py`
- **Lot Sizes**: Hardcoded INDEX_LOT_SIZES dictionary
- **Features**: EMA trend following, ATR filter, OCO orders, Trade Direction Control
- **Best for**: Stable, proven implementation with hardcoded lot sizes

#### Scalping v2 (Enhanced)
- **File**: `strategies/scalping2.py`
- **Lot Sizes**: Dynamic from OpenAlgo API
- **Features**: All v1 features + Trade Direction Control
- **Best for**: Flexibility and one-directional trading
- **New**: `TRADE_DIRECTION` env variable (long/short/both)

#### Scalping v2 Claude (Corrected)
- **File**: `strategies/scalping2_claude.py`
- **Lot Sizes**: Dynamic from OpenAlgo API
- **Features**: All v2 features + Enhanced error handling
- **Best for**: Production use with robust error handling

#### 🚀 Preplexity (OpenAlgo Optimized)
- **File**: `strategies/preplexity.py`
- **Status**: ✅ Production Ready
- **Features**: Smart orders, basket orders, proper API compliance
- **Best for**: Optimized OpenAlgo API usage

#### 🌟 Preplexity v2 (Enterprise-Grade) **⭐ RECOMMENDED**
- **File**: `strategies/preplexity_2.py`
- **Status**: ✅ Production Ready | ✅ OpenAlgo API Compliant
- **Lot Sizes**: May 2025 NSE specifications with API fallback
- **Advanced Features**:
  - ✅ Exponential backoff retry logic (3 attempts)
  - ✅ Rate limiting (10 req/sec)
  - ✅ Enhanced error handling with API error codes
  - ✅ Randomized polling intervals (2-4s)
  - ✅ Batch order status checking (50% fewer API calls)
  - ✅ Session-end reconciliation
  - ✅ Interval validation
  - ✅ Dynamic log levels (DEBUG/INFO/WARNING/ERROR)
  - ✅ Defensive API response parsing
  - ✅ Fixed basket order response parsing (critical fix!)
  - ✅ Updated lot sizes to May 2025 specs
  - ✅ Daily interval normalization (1d → D)
- **Best for**: Production trading with enterprise-grade robustness
- **Documentation**: See `strategies/preplexity_2_DOCUMENTATION.md` for complete guide

**📚 For detailed comparison and migration guide, see:**
- `strategies/UPDATES_SUMMARY.md` - Quick update summary
- `strategies/preplexity_2_DOCUMENTATION.md` - Complete reference manual

### Adding Your Own Strategy

**Zero-config strategy addition!** Just drop a `.py` file in `strategies/` folder and it appears in the UI automatically.

#### Quick Start:
```bash
# 1. Copy the template
cp strategies/TEMPLATE.py strategies/my_strategy.py

# 2. Edit the metadata and implement your logic
# (See template for required structure)

# 3. Restart web server
python3 web_server.py
# Your strategy now appears in the dropdown!
```

#### Required Structure:
Every strategy file must have:

1. **STRATEGY_METADATA dictionary:**
```python
STRATEGY_METADATA = {
    "name": "My Strategy Name",
    "description": "What it does",
    "version": "1.0",
    "features": ["Feature 1", "Feature 2"],
    "has_trade_direction": True,
    "author": "Your Name"
}
```

2. **Config dataclass:**
```python
@dataclass
class Config:
    api_key: str = os.getenv("OPENALGO_API_KEY", "")
    # ... other config parameters
```

3. **ScalpWithTrendBot class:**
```python
class ScalpWithTrendBot:
    def __init__(self, cfg: Config):
        # Initialize

    def start(self):
        # Your strategy logic
```

4. **main() function:**
```python
def main():
    cfg = Config()
    bot = ScalpWithTrendBot(cfg)
    bot.start()

if __name__ == "__main__":
    main()
```

#### Auto-Discovery Features:
✅ **Zero configuration** - No need to edit web_server.py
✅ **Hot-swappable** - Add/remove strategies without code changes
✅ **Self-documenting** - Metadata in the strategy file itself
✅ **Error tolerant** - Invalid strategies are skipped with warnings
✅ **Direct execution** - All strategies work standalone too

## API Endpoints

### GET `/api/strategies`
Returns list of available strategies with metadata

### GET `/api/config`
Returns current configuration and available symbols

### POST `/api/config`
Updates configuration with provided settings (includes `selected_strategy`)

### POST `/api/start`
Starts the selected trading bot

### POST `/api/stop`
Stops the trading bot

### GET `/api/status`
Returns current bot status and position info

### GET `/api/logs`
Returns recent log entries

## WebSocket Events

- `connect`: Client connected to server
- `disconnect`: Client disconnected
- `log`: Real-time log entry pushed to UI
- `status_update`: Position and P&L updates every 2 seconds
- `request_status`: Client can request immediate status update

## Safety Notes

⚠️ **Important Safety Information**

1. **Always Test with Paper Trading First**
   - Toggle to Paper Trading mode before testing
   - Verify strategy behavior without risking capital

2. **API Key Security**
   - Never commit your `.env` file to version control
   - Keep your API keys secure and private

3. **Risk Management**
   - Set appropriate lot sizes for your capital
   - Configure daily loss caps to limit downside
   - Use stop losses on every trade

4. **Market Hours**
   - Strategy is designed for intraday trading
   - EOD square-off prevents overnight positions
   - Sessions configurable in `scalping.py`

5. **Monitoring**
   - Keep the web UI open to monitor activity
   - Check logs regularly for errors or warnings
   - Review P&L and position status frequently

## Running Strategies Directly

Each strategy can be executed standalone without the web interface:

```bash
# Method 1: Using environment variables
export OPENALGO_API_KEY=your_key
export SYMBOL=NIFTY
export LOTS=2
export TRADE_DIRECTION=both  # v2 strategies only
python3 strategies/scalping2.py

# Method 2: Using .env file (recommended)
# Create .env with your settings, then:
python3 strategies/scalping2.py
```

### Environment Variables for Direct Execution

All strategies support these variables:
```bash
OPENALGO_API_KEY=your_api_key
OPENALGO_API_HOST=https://api.openalgo.in
SYMBOL=NIFTY
EXCHANGE=NSE_INDEX
PRODUCT=MIS
LOTS=2
EMA_FAST=5
EMA_SLOW=20
ATR_WINDOW=14
ATR_MIN_POINTS=2.0
TARGET_POINTS=10.0
STOPLOSS_POINTS=2.0
DAILY_LOSS_CAP=-1000.0
```

Additional for all strategies:
```bash
TRADE_DIRECTION=both  # Options: long, short, both (default: both)
```

**Trade Direction Examples:**
```bash
# Only long trades
TRADE_DIRECTION=long python3 strategies/scalping.py

# Only short trades
TRADE_DIRECTION=short python3 strategies/scalping2.py

# Both directions (default)
TRADE_DIRECTION=both python3 strategies/scalping2_claude.py
```

## Docker Deployment

### Building and Running

```bash
# Build the image
docker-compose build

# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f trading-bot

# Stop the container
docker-compose down

# Restart
docker-compose restart
```

### Docker Commands Cheat Sheet

```bash
# Build without cache
docker-compose build --no-cache

# Start with specific strategy
docker-compose run -e SYMBOL=BANKNIFTY trading-bot

# Access container shell
docker-compose exec trading-bot bash

# View resource usage
docker stats scalping-strategy-bot

# Remove everything (including volumes)
docker-compose down -v
```

### Environment Variables in Docker

All environment variables can be set in:
1. **`.env` file** (recommended)
2. **`docker-compose.yml`** environment section
3. **Command line**: `docker-compose run -e VAR=value`

Example `.env`:
```bash
OPENALGO_API_KEY=your_key
SYMBOL=NIFTY
LOTS=2
TRADE_DIRECTION=both
TARGET_POINTS=15.0
STOPLOSS_POINTS=5.0
```

### Production Deployment

**Using Docker on a Server:**

```bash
# 1. Copy project to server
scp -r water-life/ user@server:/path/to/app/

# 2. SSH to server
ssh user@server

# 3. Navigate to directory
cd /path/to/app/water-life/

# 4. Create .env with production credentials
nano .env

# 5. Start in detached mode
docker-compose up -d

# 6. Enable auto-restart on server reboot
docker update --restart=always scalping-strategy-bot
```

**Resource Limits:**
The docker-compose.yml includes sensible defaults:
- **CPU Limit**: 1 core max, 0.5 core reserved
- **Memory Limit**: 1GB max, 512MB reserved
- **Logs**: Max 10MB per file, 3 files rotation

### Docker Volumes

Mounted volumes for persistence:
- `./strategies:/app/strategies:ro` - Strategy files (read-only)
- `./logs:/app/logs` - Log files (persistent)
- `./.env:/app/.env:ro` - Environment config (read-only)

### Adding Strategies with Docker

```bash
# 1. Add new strategy file to strategies folder
cp strategies/TEMPLATE.py strategies/new_strategy.py

# 2. Edit the strategy
nano strategies/new_strategy.py

# 3. Restart container to pick up new strategy
docker-compose restart

# Strategy appears automatically in UI!
```

### Health Checks

The container includes automatic health monitoring:
- Checks every 30 seconds
- Verifies web server is responding
- Auto-restart if unhealthy (with docker-compose restart policy)

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' scalping-strategy-bot
```

## Troubleshooting

### Docker Issues

**Container won't start:**
```bash
# Check logs
docker-compose logs trading-bot

# Verify .env file exists
cat .env

# Check port 7777 not in use
lsof -i :7777
```

**Permission errors:**
```bash
# Fix permissions on logs directory
chmod 755 logs/

# Rebuild without cache
docker-compose build --no-cache
```

**Image too large:**
```bash
# Clean up unused images
docker system prune -a

# View image size
docker images | grep scalping
```

### Web server won't start
- Check if port 7777 is already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`

### Bot won't start
- Ensure API key is configured in `.env` or UI
- Check OpenAlgo API host is reachable
- Review logs for specific error messages

### Strategy not appearing in dropdown
- **Check strategy has required components**: `STRATEGY_METADATA`, `Config`, `ScalpWithTrendBot`
- **Check console logs** when starting web server - shows discovered strategies
- **Verify filename**: Must be `.py` file, not named `__init__.py` or `TEMPLATE.py`
- **Check for errors**: Invalid strategies are skipped with warning messages
- **Restart web server** after adding new strategy file

### No trades being executed
- Verify market is open and in configured session windows
- Check if ATR filter is too restrictive
- Ensure symbol has sufficient volatility
- Review signal generation logic in logs
- **Check TRADE_DIRECTION setting**: If set to "long" but market is trending down, no trades will be taken (and vice versa)
- Consider using "both" for TRADE_DIRECTION to allow trades in either direction

### Paper trading not working
- Verify toggle is in Paper Trading position (green)
- Check logs confirm "[PAPER TRADING MODE]" message
- Paper trades use simulated prices

## Support & Documentation

- **OpenAlgo Docs**: https://docs.openalgo.in
- **OpenAlgo Discord**: https://openalgo.in/discord
- **Strategy Files**: Review files in `strategies/` folder for detailed logic
- **Web Server**: Review `web_server.py` for API details

## License

This is a trading bot for educational and personal use. Use at your own risk. Trading involves substantial risk of loss.

---

**Happy Trading! 🚀**
