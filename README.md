# Scalping Strategy Web Manager

A professional web-based interface for managing and monitoring your OpenAlgo scalping trading strategy with real-time updates, paper trading mode, and comprehensive logging.

## Features

🎯 **Intuitive Web Interface** - Clean, modern UI accessible from any browser
📊 **Real-time Monitoring** - Live position tracking, P&L updates, and trade signals
📝 **Activity Logs** - Real-time streaming of all bot activities to the UI
🔄 **Paper Trading Mode** - Test your strategy without risking real capital
⚙️ **Full Configuration** - All strategy parameters adjustable through the UI
🚀 **Start/Stop Control** - Easy bot lifecycle management with one click

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with your OpenAlgo credentials:

```bash
OPENALGO_API_KEY=your_api_key_here
OPENALGO_API_HOST=https://api.openalgo.in
OPENALGO_WS_URL=wss://api.openalgo.in/ws  # Optional
```

### 3. Start the Web Server

```bash
python web_server.py
```

### 4. Access the UI

Open your browser and navigate to:
```
http://localhost:7777
```

## Usage Guide

### Configuration Panel

1. **Trading Mode Toggle**
   - 📝 **Paper Trading** (Green): Simulates trades without real orders
   - 🔴 **Live Trading** (Red): Places real orders on your broker

2. **API Settings**
   - Enter your OpenAlgo API key and host URL
   - These can also be set via environment variables

3. **Symbol & Exchange**
   - Choose from: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, etc.
   - Select exchange: NSE_INDEX or BSE_INDEX
   - Product type: MIS (Intraday), CNC, or NRML

4. **Indicator Settings**
   - **EMA Fast**: Fast EMA period (default: 5)
   - **EMA Slow**: Slow EMA period (default: 20)
   - **ATR Window**: ATR calculation period (default: 14)
   - **ATR Min Points**: Minimum ATR to trade (filters low volatility)

5. **Risk Management**
   - **Target Points**: Take profit level in points
   - **Stoploss Points**: Stop loss level in points
   - **Daily Loss Cap**: Maximum daily loss before stopping new trades
   - **Confirm Trend at Entry**: Additional trend confirmation filter
   - **EOD Square-off**: Auto-exit positions at end of day
   - **Square-off Time**: Time to square-off positions (e.g., 15:25)

6. **Save Configuration**
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
├── scalping.py           # Core trading bot logic
├── web_server.py         # Flask web server and API
├── templates/
│   └── index.html        # Web UI
├── requirements.txt      # Python dependencies
├── .env                  # API credentials (create this)
└── README.md            # This file
```

## API Endpoints

### GET `/api/config`
Returns current configuration and available symbols

### POST `/api/config`
Updates configuration with provided settings

### POST `/api/start`
Starts the trading bot

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

## Troubleshooting

### Web server won't start
- Check if port 7777 is already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`

### Bot won't start
- Ensure API key is configured in `.env` or UI
- Check OpenAlgo API host is reachable
- Review logs for specific error messages

### No trades being executed
- Verify market is open and in configured session windows
- Check if ATR filter is too restrictive
- Ensure symbol has sufficient volatility
- Review signal generation logic in logs

### Paper trading not working
- Verify toggle is in Paper Trading position (green)
- Check logs confirm "[PAPER TRADING MODE]" message
- Paper trades use simulated prices

## Support & Documentation

- **OpenAlgo Docs**: https://docs.openalgo.in
- **OpenAlgo Discord**: https://openalgo.in/discord
- **Strategy File**: Review `scalping.py` for detailed logic
- **Web Server**: Review `web_server.py` for API details

## License

This is a trading bot for educational and personal use. Use at your own risk. Trading involves substantial risk of loss.

---

**Happy Trading! 🚀**
