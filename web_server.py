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
from typing import Optional, Dict, List
from collections import deque
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pytz

# Optional rate limiting
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False
    print("[INFO] flask-limiter not installed. Rate limiting disabled. Install with: pip install flask-limiter")

# Import auto-discovery
import strategies as strategies_pkg
from strategies import DISCOVERED_STRATEGIES

# Build strategy registry from discovered strategies
STRATEGY_REGISTRY = DISCOVERED_STRATEGIES

print(f"[INFO] Loaded {len(STRATEGY_REGISTRY)} strategies: {', '.join(STRATEGY_REGISTRY.keys())}")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'scalping-strategy-secret-key'
CORS(app)

_default_async_mode = os.getenv("SOCKETIO_ASYNC_MODE", "eventlet")
try:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode=_default_async_mode)
except ValueError:
    # Fallback if requested async mode is unavailable (e.g., missing eventlet)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Setup rate limiting if available
if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://"
    )
    print("[INFO] Rate limiting enabled")
else:
    limiter = None

IST = pytz.timezone("Asia/Kolkata")
DEFAULT_SYMBOLS = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50', 'SENSEX', 'BANKEX', 'SENSEX50']

_background_status_task = None


def ensure_background_tasks():
    """Start long-lived background jobs once per worker."""
    global _background_status_task
    if _background_status_task is None:
        _background_status_task = socketio.start_background_task(background_status_pusher)


def parse_config_value(field_schema, raw_value):
    field_type = field_schema.get('type')
    if raw_value is None or raw_value == '':
        return None

    if field_type == 'boolean':
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str):
            return raw_value.lower() in ('true', '1', 'yes', 'on')
        return bool(raw_value)

    if field_type == 'number':
        fmt = field_schema.get('number_format', 'float')
        try:
            if fmt == 'int':
                return int(raw_value)
            if fmt == 'float':
                return float(raw_value)
        except (TypeError, ValueError):
            return None
        return raw_value

    if field_type == 'time':
        if isinstance(raw_value, (list, tuple)) and len(raw_value) >= 2:
            return (int(raw_value[0]), int(raw_value[1]))
        if isinstance(raw_value, str):
            parts = raw_value.split(':')
            if len(parts) == 2:
                try:
                    return (int(parts[0]), int(parts[1]))
                except ValueError:
                    return None
        return None

    # Default to returning raw value (string or other types)
    return raw_value


def get_strategy_schema(strategy_id: str):
    strategy_info = STRATEGY_REGISTRY.get(strategy_id)
    if not strategy_info:
        return []
    return strategy_info.get('config_schema', [])


def serialize_config_for_response(strategy_id: str, config_instance):
    strategy_info = STRATEGY_REGISTRY.get(strategy_id)
    if not strategy_info:
        return {}
    schema = strategy_info.get('config_schema', [])
    if not config_instance:
        return strategy_info.get('default_config', {})
    return strategies_pkg.serialize_config(config_instance, schema)

# Global state
class BotManager:
    def __init__(self):
        self.bot = None
        self.bot_thread: Optional[Thread] = None
        self.config = None
        self.selected_strategy: str = "scalping"  # Default strategy
        self.is_running = False
        self.is_paper_trading = True
        self.lock = Lock()  # Main state lock
        self.stats_lock = Lock()  # Separate lock for stats updates
        self.log_queue = deque(maxlen=1000)  # Bounded queue - max 1000 logs
        self.health_check_thread: Optional[Thread] = None
        self.health_check_running = False
        self.last_stats = None  # Cache for stats comparison
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
        """Update stats from bot with thread safety"""
        with self.stats_lock:
            try:
                if self.bot and self.is_running:
                    self.stats.update({
                        "status": "running",
                        "strategy": self.selected_strategy,
                        "in_position": getattr(self.bot, 'in_position', False),
                        "side": getattr(self.bot, 'side', None),
                        "entry_price": getattr(self.bot, 'entry_price', None),
                        "tp_level": getattr(self.bot, 'tp_level', None),
                        "sl_level": getattr(self.bot, 'sl_level', None),
                        "realized_pnl_today": getattr(self.bot, 'realized_pnl_today', 0.0),
                        "pending_signal": getattr(self.bot, 'pending_signal', None),
                        "last_update": datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
                    })
                else:
                    self.stats["status"] = "stopped"
                    self.stats["last_update"] = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                logging.error(f"[BotManager] Error updating stats: {e}")

    def start_health_check(self):
        """Start health check thread to monitor bot"""
        if not self.health_check_running:
            self.health_check_running = True
            self.health_check_thread = Thread(target=self._health_check_loop, daemon=True)
            self.health_check_thread.start()
            logging.info("[BotManager] Health check started")

    def stop_health_check(self):
        """Stop health check thread"""
        self.health_check_running = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        logging.info("[BotManager] Health check stopped")

    def _health_check_loop(self):
        """Monitor bot health and update status if thread dies"""
        import time
        while self.health_check_running:
            try:
                time.sleep(5)  # Check every 5 seconds
                if self.is_running:
                    if not self.bot_thread or not self.bot_thread.is_alive():
                        logging.error("[HEALTH] Bot thread died unexpectedly!")
                        with self.lock:
                            self.is_running = False
                            self.bot = None
                        socketio.emit('log', {
                            'timestamp': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
                            'level': 'ERROR',
                            'message': '❌ Bot thread crashed! Please check logs and restart.'
                        }, namespace='/')
            except Exception as e:
                logging.error(f"[HEALTH] Health check error: {e}")

bot_manager = BotManager()

# Custom logging handler to capture logs with bounded queue
class WebSocketLogHandler(logging.Handler):
    def emit(self, record):
        try:
            log_entry = {
                'timestamp': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
                'level': record.levelname,
                'message': self.format(record)
            }
            # deque with maxlen automatically removes oldest items
            bot_manager.log_queue.append(log_entry)
            socketio.emit('log', log_entry, namespace='/')
        except Exception as e:
            # Fail silently to avoid logging loops
            print(f"[WARN] Log emit failed: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
ws_handler = WebSocketLogHandler()
ws_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ws_handler)

# Monkey patch print to capture console output with bounded queue
original_print = print
def custom_print(*args, **kwargs):
    try:
        message = ' '.join(map(str, args))
        log_entry = {
            'timestamp': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
            'level': 'INFO',
            'message': message
        }
        bot_manager.log_queue.append(log_entry)
        socketio.emit('log', log_entry, namespace='/')
        original_print(*args, **kwargs)
    except Exception as e:
        # Fail silently to avoid print loops
        original_print(f"[WARN] custom_print failed: {e}")

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
            'has_trade_direction': info['has_trade_direction'],
            'config_schema': info.get('config_schema', []),
            'default_config': info.get('default_config', {}),
            'lot_sizes': info.get('lot_sizes', {})
        })
    return jsonify({
        'strategies': strategies,
        'current': bot_manager.selected_strategy
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    if bot_manager.config:
        config_dict = serialize_config_for_response(bot_manager.selected_strategy, bot_manager.config)
    else:
        strategy_info = STRATEGY_REGISTRY.get(bot_manager.selected_strategy)
        if strategy_info:
            config_dict = strategy_info.get('default_config', {})
        else:
            config_dict = {}

    # Get lot sizes from current strategy
    strategy_info = STRATEGY_REGISTRY.get(bot_manager.selected_strategy, STRATEGY_REGISTRY['scalping'])
    available_symbols = list(strategy_info['lot_sizes'].keys()) if strategy_info['lot_sizes'] else DEFAULT_SYMBOLS

    return jsonify({
        'config': config_dict,
        'is_paper_trading': bot_manager.is_paper_trading,
        'selected_strategy': bot_manager.selected_strategy,
        'has_trade_direction': strategy_info.get('has_trade_direction', False),
        'available_symbols': available_symbols,
        'schema': strategy_info.get('config_schema', [])
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
        schema = strategy_info.get('config_schema', [])

        config_params = {}
        for field_schema in schema:
            name = field_schema['name']
            raw_value = data.get(name, field_schema.get('default'))
            parsed_value = parse_config_value(field_schema, raw_value)
            if parsed_value is not None:
                config_params[name] = parsed_value

        config = ConfigClass(**config_params)

        bot_manager.config = config
        bot_manager.is_paper_trading = data.get('is_paper_trading', True)

        return jsonify({'success': True, 'message': 'Configuration updated'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    # Apply rate limiting if available
    if RATE_LIMITING_AVAILABLE and limiter:
        try:
            limiter.check()
        except Exception:
            return jsonify({'success': False, 'error': 'Rate limit exceeded. Please wait before starting again.'}), 429

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

            # Start health check monitoring
            bot_manager.start_health_check()

            return jsonify({'success': True, 'message': 'Bot started successfully'})
        except Exception as e:
            bot_manager.is_running = False
            bot_manager.bot = None
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    with bot_manager.lock:
        if not bot_manager.is_running:
            return jsonify({'success': False, 'error': 'Bot is not running'}), 400

        try:
            # Stop health check first
            bot_manager.stop_health_check()

            # Pause the bot instead of calling _graceful_exit() which would terminate the process
            if bot_manager.bot:
                # Check if the bot has a pause method (for newer strategies)
                if hasattr(bot_manager.bot, 'pause'):
                    bot_manager.bot.pause()
                elif hasattr(bot_manager.bot, 'bot') and hasattr(bot_manager.bot.bot, 'pause'):
                    # For PaperTradingBot wrapper
                    bot_manager.bot.bot.pause()
                else:
                    # Fallback for older strategies without pause method
                    custom_print("[WARN] Strategy doesn't support pause, stopping scheduler only")
                    if hasattr(bot_manager.bot, 'scheduler'):
                        bot_manager.bot.scheduler.shutdown(wait=False)
                    elif hasattr(bot_manager.bot, 'bot') and hasattr(bot_manager.bot.bot, 'scheduler'):
                        bot_manager.bot.bot.scheduler.shutdown(wait=False)

            bot_manager.is_running = False
            bot_manager.bot = None

            return jsonify({'success': True, 'message': 'Bot stopped successfully'})
        except Exception as e:
            bot_manager.is_running = False
            bot_manager.bot = None
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get bot status and stats"""
    bot_manager.update_stats()
    return jsonify(bot_manager.stats)

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get recent logs from bounded deque"""
    # Convert deque to list for JSON serialization
    logs = list(bot_manager.log_queue)
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
    ensure_background_tasks()
    emit('connected', {'message': 'Connected to strategy server'})
    custom_print("[WS] Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    custom_print("[WS] Client disconnected")

@socketio.on('request_status')
def handle_status_request():
    bot_manager.update_stats()
    emit('status_update', bot_manager.stats)

# Background task to push updates (optimized with change detection)
def background_status_pusher():
    """Push status updates to connected clients only when changed"""
    last_status = None
    while True:
        socketio.sleep(2)
        try:
            bot_manager.update_stats()

            # Only emit if status changed (reduce unnecessary updates)
            current_status = json.dumps(bot_manager.stats, sort_keys=True)
            if current_status != last_status:
                socketio.emit('status_update', bot_manager.stats, namespace='/')
                last_status = current_status
        except Exception as e:
            logging.error(f"[STATUS_PUSHER] Error: {e}")

# ==================== MAIN ====================

def main():
    custom_print("=" * 60)
    custom_print("Scalping Strategy Web Interface")
    custom_print("=" * 60)
    custom_print(f"Server starting on http://localhost:7777")
    custom_print("=" * 60)

    # Start background task
    ensure_background_tasks()

    # Run server
    socketio.run(app, host='0.0.0.0', port=7777, debug=False)

if __name__ == '__main__':
    main()
