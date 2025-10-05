#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalping Strategy Web Interface
Web server on port 7777 to manage and monitor the trading strategy
"""

import os
import sys
import json
import logging
from datetime import datetime
from threading import Thread, Lock
from queue import Queue
from typing import Optional, Dict, List
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pytz

# Import auto-discovery
from strategies import DISCOVERED_STRATEGIES

# Build strategy registry from discovered strategies
STRATEGY_REGISTRY = DISCOVERED_STRATEGIES

print(f"[INFO] Loaded {len(STRATEGY_REGISTRY)} strategies: {', '.join(STRATEGY_REGISTRY.keys())}")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'scalping-strategy-secret-key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

IST = pytz.timezone("Asia/Kolkata")

# Global state
class BotManager:
    def __init__(self):
        self.bot = None
        self.bot_thread: Optional[Thread] = None
        self.config = None
        self.selected_strategy: str = "scalping"  # Default strategy
        self.is_running = False
        self.is_paper_trading = True
        self.lock = Lock()
        self.log_queue = Queue()
        self.stats = {
            "status": "stopped",
            "strategy": "scalping",
            "in_position": False,
            "side": None,
            "entry_price": None,
            "tp_level": None,
            "sl_level": None,
            "realized_pnl_today": 0.0,
            "pending_signal": None,
            "last_update": None
        }

    def update_stats(self):
        """Update stats from bot"""
        if self.bot:
            with self.lock:
                self.stats.update({
                    "status": "running" if self.is_running else "stopped",
                    "strategy": self.selected_strategy,
                    "in_position": self.bot.in_position,
                    "side": self.bot.side,
                    "entry_price": self.bot.entry_price,
                    "tp_level": self.bot.tp_level,
                    "sl_level": self.bot.sl_level,
                    "realized_pnl_today": self.bot.realized_pnl_today,
                    "pending_signal": self.bot.pending_signal,
                    "last_update": datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
                })

bot_manager = BotManager()

# Custom logging handler to capture logs
class WebSocketLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = {
            'timestamp': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'message': self.format(record)
        }
        bot_manager.log_queue.put(log_entry)
        socketio.emit('log', log_entry, namespace='/')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
ws_handler = WebSocketLogHandler()
ws_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ws_handler)

# Monkey patch print to capture console output
original_print = print
def custom_print(*args, **kwargs):
    message = ' '.join(map(str, args))
    log_entry = {
        'timestamp': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
        'level': 'INFO',
        'message': message
    }
    bot_manager.log_queue.put(log_entry)
    socketio.emit('log', log_entry, namespace='/')
    original_print(*args, **kwargs)

# Override print in all strategy modules
for strategy_id, strategy_info in STRATEGY_REGISTRY.items():
    strategy_info['module'].print = custom_print

# ==================== API ROUTES ====================

@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')

@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Get list of available strategies"""
    strategies = []
    for key, info in STRATEGY_REGISTRY.items():
        strategies.append({
            'id': key,
            'name': info['name'],
            'description': info['description'],
            'features': info['features'],
            'has_trade_direction': info['has_trade_direction']
        })
    return jsonify({
        'strategies': strategies,
        'current': bot_manager.selected_strategy
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    if bot_manager.config:
        config_dict = {
            'api_key': bot_manager.config.api_key[:10] + '...' if bot_manager.config.api_key else '',
            'api_host': bot_manager.config.api_host,
            'symbol': bot_manager.config.symbol,
            'exchange': bot_manager.config.exchange,
            'product': bot_manager.config.product,
            'lots': bot_manager.config.lots,
            'interval': bot_manager.config.interval,
            'ema_fast': bot_manager.config.ema_fast,
            'ema_slow': bot_manager.config.ema_slow,
            'atr_window': bot_manager.config.atr_window,
            'atr_min_points': bot_manager.config.atr_min_points,
            'target_points': bot_manager.config.target_points,
            'stoploss_points': bot_manager.config.stoploss_points,
            'confirm_trend_at_entry': bot_manager.config.confirm_trend_at_entry,
            'daily_loss_cap': bot_manager.config.daily_loss_cap,
            'enable_eod_square_off': bot_manager.config.enable_eod_square_off,
            'square_off_time': f"{bot_manager.config.square_off_time[0]:02d}:{bot_manager.config.square_off_time[1]:02d}",
            'warmup_days': bot_manager.config.warmup_days,
        }
    else:
        # Default config
        config_dict = {
            'api_key': os.environ.get("OPENALGO_API_KEY", ""),
            'api_host': os.environ.get("OPENALGO_API_HOST", "https://api.openalgo.in"),
            'symbol': 'NIFTY',
            'exchange': 'NSE_INDEX',
            'product': 'MIS',
            'lots': 2,
            'interval': '5m',
            'ema_fast': 5,
            'ema_slow': 20,
            'atr_window': 14,
            'atr_min_points': 2.0,
            'target_points': 10.0,
            'stoploss_points': 2.0,
            'confirm_trend_at_entry': True,
            'daily_loss_cap': -1000.0,
            'enable_eod_square_off': True,
            'square_off_time': '15:25',
            'warmup_days': 10,
        }

    # Get lot sizes from current strategy
    strategy_info = STRATEGY_REGISTRY.get(bot_manager.selected_strategy, STRATEGY_REGISTRY['scalping'])
    available_symbols = list(strategy_info['lot_sizes'].keys()) if strategy_info['lot_sizes'] else ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50', 'SENSEX', 'BANKEX', 'SENSEX50']

    return jsonify({
        'config': config_dict,
        'is_paper_trading': bot_manager.is_paper_trading,
        'selected_strategy': bot_manager.selected_strategy,
        'has_trade_direction': strategy_info.get('has_trade_direction', False),
        'available_symbols': available_symbols
    })

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration"""
    data = request.json

    try:
        # Update selected strategy
        selected_strategy = data.get('selected_strategy', 'scalping')
        if selected_strategy not in STRATEGY_REGISTRY:
            return jsonify({'success': False, 'error': f'Invalid strategy: {selected_strategy}'}), 400

        bot_manager.selected_strategy = selected_strategy
        strategy_info = STRATEGY_REGISTRY[selected_strategy]
        ConfigClass = strategy_info['config_class']

        # Parse square_off_time
        square_off_parts = data.get('square_off_time', '15:25').split(':')
        square_off_time = (int(square_off_parts[0]), int(square_off_parts[1]))

        # Build config dict
        config_params = {
            'api_key': data.get('api_key', os.environ.get("OPENALGO_API_KEY", "")),
            'api_host': data.get('api_host', "https://api.openalgo.in"),
            'ws_url': os.environ.get("OPENALGO_WS_URL"),
            'symbol': data.get('symbol', 'NIFTY'),
            'exchange': data.get('exchange', 'NSE_INDEX'),
            'product': data.get('product', 'MIS'),
            'lots': int(data.get('lots', 2)),
            'interval': data.get('interval', '5m'),
            'ema_fast': int(data.get('ema_fast', 5)),
            'ema_slow': int(data.get('ema_slow', 20)),
            'atr_window': int(data.get('atr_window', 14)),
            'atr_min_points': float(data.get('atr_min_points', 2.0)),
            'target_points': float(data.get('target_points', 10.0)),
            'stoploss_points': float(data.get('stoploss_points', 2.0)),
            'confirm_trend_at_entry': bool(data.get('confirm_trend_at_entry', True)),
            'daily_loss_cap': float(data.get('daily_loss_cap', -1000.0)),
            'enable_eod_square_off': bool(data.get('enable_eod_square_off', True)),
            'square_off_time': square_off_time,
            'warmup_days': int(data.get('warmup_days', 10)),
        }

        # Add trade_direction for strategies that support it
        if strategy_info.get('has_trade_direction'):
            config_params['trade_direction'] = data.get('trade_direction', 'both')

        config = ConfigClass(**config_params)

        bot_manager.config = config
        bot_manager.is_paper_trading = data.get('is_paper_trading', True)

        return jsonify({'success': True, 'message': 'Configuration updated'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    with bot_manager.lock:
        if bot_manager.is_running:
            return jsonify({'success': False, 'error': 'Bot is already running'}), 400

        if not bot_manager.config:
            return jsonify({'success': False, 'error': 'Configuration not set'}), 400

        try:
            # Get selected strategy info
            strategy_info = STRATEGY_REGISTRY[bot_manager.selected_strategy]
            BotClass = strategy_info['bot_class']

            # Create bot instance
            if bot_manager.is_paper_trading:
                # Wrap bot for paper trading
                bot_manager.bot = PaperTradingBot(bot_manager.config, BotClass)
                custom_print(f"[PAPER TRADING MODE] {strategy_info['name']} - Simulating trades without real orders")
            else:
                bot_manager.bot = BotClass(bot_manager.config)
                custom_print(f"[LIVE TRADING MODE] {strategy_info['name']} - Placing real orders")

            # Start bot in separate thread
            bot_manager.bot_thread = Thread(target=bot_manager.bot.start, daemon=True)
            bot_manager.bot_thread.start()
            bot_manager.is_running = True

            return jsonify({'success': True, 'message': 'Bot started successfully'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    with bot_manager.lock:
        if not bot_manager.is_running:
            return jsonify({'success': False, 'error': 'Bot is not running'}), 400

        try:
            if bot_manager.bot:
                bot_manager.bot._graceful_exit()
            bot_manager.is_running = False
            bot_manager.bot = None

            return jsonify({'success': True, 'message': 'Bot stopped successfully'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get bot status and stats"""
    bot_manager.update_stats()
    return jsonify(bot_manager.stats)

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get recent logs"""
    logs = []
    while not bot_manager.log_queue.empty():
        logs.append(bot_manager.log_queue.get())
    return jsonify({'logs': logs})

# ==================== PAPER TRADING BOT ====================

class PaperTradingBot:
    """Paper trading wrapper that simulates orders for any strategy"""

    def __init__(self, cfg, BotClass):
        # Initialize the actual bot
        self.bot = BotClass(cfg)
        self.cfg = cfg
        self.paper_orders = {}
        self.paper_order_counter = 1000

        # Proxy attributes to the wrapped bot
        for attr in ['in_position', 'side', 'entry_price', 'tp_level', 'sl_level',
                     'realized_pnl_today', 'pending_signal', 'qty']:
            setattr(self, attr, getattr(self.bot, attr, None))

    def __getattr__(self, name):
        """Proxy any other attribute access to wrapped bot"""
        return getattr(self.bot, name)

    def start(self):
        """Start the wrapped bot"""
        # Override place_entry and place_exit_legs before starting
        original_place_entry = self.bot.place_entry
        original_place_exit_legs = self.bot.place_exit_legs

        def paper_place_entry(side: str):
            assert side in ("LONG","SHORT")
            custom_print(f"[PAPER] {side} ENTRY simulated for {self.bot.qty} {self.bot.cfg.symbol}@{self.bot.cfg.exchange}")

            self.bot.entry_order_id = f"PAPER_{self.paper_order_counter}"
            self.paper_order_counter += 1
            self.bot.entry_price = 100.0  # Placeholder
            self.bot.in_position = True
            self.bot.side = side

            if side == 'LONG':
                self.bot.tp_level = self.bot.entry_price + self.bot.cfg.target_points
                self.bot.sl_level = self.bot.entry_price - self.bot.cfg.stoploss_points
            else:
                self.bot.tp_level = self.bot.entry_price - self.bot.cfg.target_points
                self.bot.sl_level = self.bot.entry_price + self.bot.cfg.stoploss_points

            custom_print(f"[PAPER] Filled ~{self.bot.entry_price:.2f}. TP={self.bot.tp_level:.2f} SL={self.bot.sl_level:.2f}")
            self.bot.place_exit_legs()

        def paper_place_exit_legs():
            if not self.bot.in_position or self.bot.entry_price is None:
                return
            self.bot.tp_order_id = f"PAPER_{self.paper_order_counter}"
            self.paper_order_counter += 1
            self.bot.sl_order_id = f"PAPER_{self.paper_order_counter}"
            self.paper_order_counter += 1
            custom_print(f"[PAPER] Exit orders simulated: TP @ {self.bot.tp_level:.2f}, SL @ {self.bot.sl_level:.2f}")

        # Monkey patch the methods
        self.bot.place_entry = paper_place_entry
        self.bot.place_exit_legs = paper_place_exit_legs

        # Start the wrapped bot
        self.bot.start()

# ==================== WebSocket Events ====================

@socketio.on('connect')
def handle_connect():
    emit('connected', {'message': 'Connected to strategy server'})
    custom_print("[WS] Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    custom_print("[WS] Client disconnected")

@socketio.on('request_status')
def handle_status_request():
    bot_manager.update_stats()
    emit('status_update', bot_manager.stats)

# Background task to push updates
def background_status_pusher():
    """Push status updates to connected clients"""
    while True:
        socketio.sleep(2)
        bot_manager.update_stats()
        socketio.emit('status_update', bot_manager.stats, namespace='/')

# ==================== MAIN ====================

def main():
    custom_print("=" * 60)
    custom_print("Scalping Strategy Web Interface")
    custom_print("=" * 60)
    custom_print(f"Server starting on http://localhost:7777")
    custom_print("=" * 60)

    # Start background task
    socketio.start_background_task(background_status_pusher)

    # Run server
    socketio.run(app, host='0.0.0.0', port=7777, debug=False, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    main()
