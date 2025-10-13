"""
Trading Strategies Package
Contains various trading strategy implementations with auto-discovery
"""

import importlib
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_type_hints, get_args, get_origin

# Default UI configuration for strategies. Individual strategies can override these.
DEFAULT_FIELD_ORDER: List[str] = [
    "api_key",
    "api_host",
    "symbol",
    "exchange",
    "product",
    "lots",
    "interval",
    "ema_fast",
    "ema_slow",
    "atr_window",
    "atr_min_points",
    "target_points",
    "stoploss_points",
    "confirm_trend_at_entry",
    "trade_direction",
    "daily_loss_cap",
    "enable_eod_square_off",
    "square_off_time",
    "warmup_days",
]

DEFAULT_EXCLUDE_FIELDS = {
    "session_windows",
    "history_cache_dir",
    "history_days",
    "force_refresh_cache",
    "use_history_cache",
}

DEFAULT_FIELD_DEFS: Dict[str, Dict[str, Any]] = {
    "api_key": {
        "label": "API Key",
        "group": "Connection",
        "placeholder": "Your OpenAlgo API Key",
    },
    "api_host": {
        "label": "API Host",
        "group": "Connection",
        "placeholder": "https://api.openalgo.in",
    },
    "symbol": {
        "label": "Symbol",
        "group": "Instrument",
        "placeholder": "NIFTY / BANKNIFTY / Option Symbol",
    },
    "exchange": {
        "label": "Exchange",
        "group": "Instrument",
        "type": "select",
        "options": ["NSE_INDEX", "BSE_INDEX", "NFO", "BSE", "MCX"],
    },
    "product": {
        "label": "Product",
        "group": "Instrument",
        "type": "select",
        "options": ["MIS", "CNC", "NRML"],
    },
    "lots": {
        "label": "Lots",
        "group": "Instrument",
        "number_format": "int",
        "step": 1,
        "min": 1,
    },
    "interval": {
        "label": "Interval",
        "group": "Engine",
        "placeholder": "e.g. 1m / 5m / 15m",
    },
    "ema_fast": {
        "label": "EMA Fast",
        "group": "Indicators",
        "number_format": "int",
        "step": 1,
        "min": 1,
    },
    "ema_slow": {
        "label": "EMA Slow",
        "group": "Indicators",
        "number_format": "int",
        "step": 1,
        "min": 1,
    },
    "atr_window": {
        "label": "ATR Window",
        "group": "Indicators",
        "number_format": "int",
        "step": 1,
        "min": 1,
    },
    "atr_min_points": {
        "label": "ATR Min Points",
        "group": "Indicators",
        "number_format": "float",
        "step": 0.1,
    },
    "target_points": {
        "label": "Target Points",
        "group": "Risk Management",
        "number_format": "float",
        "step": 0.5,
    },
    "stoploss_points": {
        "label": "Stoploss Points",
        "group": "Risk Management",
        "number_format": "float",
        "step": 0.5,
    },
    "confirm_trend_at_entry": {
        "label": "Confirm Trend at Entry",
        "group": "Risk Management",
    },
    "trade_direction": {
        "label": "Trade Direction",
        "group": "Risk Management",
        "type": "select",
        "options": ["both", "long", "short"],
    },
    "daily_loss_cap": {
        "label": "Daily Loss Cap (₹)",
        "group": "Risk Management",
        "number_format": "float",
        "step": 100,
    },
    "enable_eod_square_off": {
        "label": "Enable EOD Square-off",
        "group": "Risk Management",
    },
    "square_off_time": {
        "label": "Square-off Time",
        "group": "Risk Management",
        "type": "time",
        "placeholder": "15:25",
    },
    "warmup_days": {
        "label": "Warmup Days",
        "group": "Backtest & History",
        "number_format": "int",
        "step": 1,
        "min": 0,
    },
    "trade_every_n_bars": {
        "label": "Trade Every N Bars",
        "group": "Strategy Behaviour",
        "number_format": "int",
        "step": 1,
        "min": 1,
    },
    "profit_target_rupees": {
        "label": "Profit Target (₹)",
        "group": "Strategy Behaviour",
        "number_format": "float",
        "step": 0.5,
    },
    "stop_loss_rupees": {
        "label": "Stop Loss (₹)",
        "group": "Strategy Behaviour",
        "number_format": "float",
        "step": 0.5,
    },
    "brokerage_per_trade": {
        "label": "Brokerage per Trade (₹)",
        "group": "Costs",
        "number_format": "float",
        "step": 0.5,
    },
    "slippage_rupees": {
        "label": "Slippage (₹)",
        "group": "Costs",
        "number_format": "float",
        "step": 0.5,
    },
    "test_mode": {
        "label": "Test Mode",
        "group": "Execution",
    },
    "log_to_file": {
        "label": "Log to File",
        "group": "Execution",
    },
    "persist_state": {
        "label": "Persist State",
        "group": "Execution",
    },
    "ignore_entry_delta": {
        "label": "Ignore Entry Timing Delta",
        "group": "Strategy Behaviour",
    },
    "use_history": {
        "label": "Use Historical Data",
        "group": "Backtest & History",
    },
}


def _field_default(dc_field) -> Any:
    if dc_field.default is not MISSING:
        return dc_field.default
    if dc_field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return dc_field.default_factory()  # type: ignore[attr-defined]
    return None


def _infer_ui_type(py_type: Optional[Any], default: Any) -> Dict[str, Any]:
    """
    Infer the UI field type, number format, and step size from typing hints/defaults.
    Returns dict with keys: type, number_format, step.
    """
    result = {"type": None, "number_format": None, "step": None}

    if py_type is None and default is not None:
        py_type = type(default)

    origin = get_origin(py_type) if py_type else None
    args = get_args(py_type) if py_type else ()

    # Handle Optional/Union
    if origin is Union:
        non_none = [arg for arg in args if arg is not type(None)]  # noqa: E721
        if non_none:
            return _infer_ui_type(non_none[0], default)

    if py_type in (bool,):
        result["type"] = "boolean"
        return result

    if py_type in (int,):
        result["type"] = "number"
        result["number_format"] = "int"
        result["step"] = 1
        return result

    if py_type in (float,):
        result["type"] = "number"
        result["number_format"] = "float"
        result["step"] = 0.1
        return result

    if py_type in (str,):
        result["type"] = "text"
        return result

    if origin in (tuple, list) and len(args) == 2 and all(arg in (int, type(None)) for arg in args):
        result["type"] = "time"
        return result

    # Fall back to default value inspection
    if isinstance(default, bool):
        result["type"] = "boolean"
    elif isinstance(default, int):
        result["type"] = "number"
        result["number_format"] = "int"
        result["step"] = 1
    elif isinstance(default, float):
        result["type"] = "number"
        result["number_format"] = "float"
        result["step"] = 0.1
    elif isinstance(default, tuple) and len(default) == 2 and all(isinstance(v, int) for v in default):
        result["type"] = "time"
    else:
        result["type"] = "text"

    return result


def _format_value_for_ui(field_schema: Dict[str, Any], value: Any) -> Any:
    if value is None:
        return ""

    field_type = field_schema.get("type")
    if field_type == "boolean":
        return bool(value)
    if field_type == "number":
        number_format = field_schema.get("number_format")
        if number_format == "int":
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    if field_type == "time":
        if isinstance(value, str):
            return value
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return f"{int(value[0]):02d}:{int(value[1]):02d}"
        return ""
    return value


def build_config_schema(module: Any, config_class: Any) -> List[Dict[str, Any]]:
    if not is_dataclass(config_class):
        return []

    try:
        type_hints = get_type_hints(config_class, module.__dict__)
    except Exception:
        type_hints = {}

    module_field_defs: Dict[str, Dict[str, Any]] = getattr(module, "CONFIG_FIELD_DEFS", {})
    module_field_order: Optional[List[str]] = getattr(module, "CONFIG_FIELD_ORDER", None)
    module_exclude: List[str] = getattr(module, "CONFIG_FIELD_EXCLUDE", [])

    exclude_fields = set(DEFAULT_EXCLUDE_FIELDS).union(module_exclude)
    dataclass_fields = {f.name: f for f in fields(config_class)}

    # Build final order
    ordered_names: List[str] = []
    base_order = module_field_order or DEFAULT_FIELD_ORDER
    for name in base_order:
        if name in dataclass_fields and name not in exclude_fields:
            ordered_names.append(name)

    for name in dataclass_fields.keys():
        if name not in ordered_names and name not in exclude_fields:
            ordered_names.append(name)

    schema: List[Dict[str, Any]] = []
    for name in ordered_names:
        dc_field = dataclass_fields[name]
        field_info = {**DEFAULT_FIELD_DEFS.get(name, {}), **module_field_defs.get(name, {})}
        if field_info.get("skip"):
            continue

        default_value = _field_default(dc_field)
        ui_inference = _infer_ui_type(type_hints.get(name), default_value)

        field_schema: Dict[str, Any] = {
            "name": name,
            "label": field_info.get("label", name.replace("_", " ").title()),
            "group": field_info.get("group", "General"),
            "type": field_info.get("type", ui_inference["type"]),
        }

        if ui_inference["number_format"] and "number_format" not in field_info:
            field_schema["number_format"] = ui_inference["number_format"]
        if ui_inference["step"] and "step" not in field_info:
            field_schema["step"] = ui_inference["step"]

        # Override with explicit definitions if provided
        if "number_format" in field_info:
            field_schema["number_format"] = field_info["number_format"]
        if "step" in field_info:
            field_schema["step"] = field_info["step"]
        if "min" in field_info:
            field_schema["min"] = field_info["min"]
        if "max" in field_info:
            field_schema["max"] = field_info["max"]
        if "placeholder" in field_info:
            field_schema["placeholder"] = field_info["placeholder"]
        if "options" in field_info:
            field_schema["options"] = field_info["options"]
        if "help_text" in field_info:
            field_schema["help_text"] = field_info["help_text"]

        field_schema["default"] = _format_value_for_ui(field_schema, default_value)
        schema.append(field_schema)

    return schema


def serialize_config(config_instance: Any, schema: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for field_schema in schema:
        name = field_schema["name"]
        value = getattr(config_instance, name, None)
        payload[name] = _format_value_for_ui(field_schema, value)
    return payload

def discover_strategies() -> Dict[str, Dict[str, Any]]:
    """
    Automatically discover all strategy modules in this package.

    Returns:
        Dict mapping strategy_id to strategy info including:
        - module: The imported module
        - name: Display name
        - description: Strategy description
        - bot_class: The trading bot class
        - config_class: The configuration class
        - features: List of features
        - has_trade_direction: Boolean
        - lot_sizes: Lot size dictionary (if available)
    """
    strategies = {}
    strategies_dir = Path(__file__).parent

    # Get all .py files except __init__.py and TEMPLATE.py
    strategy_files = [
        f for f in strategies_dir.glob("*.py")
        if f.name not in ("__init__.py", "TEMPLATE.py") and not f.name.startswith("_")
    ]

    for strategy_file in strategy_files:
        strategy_id = strategy_file.stem  # filename without .py
        module_name = f"strategies.{strategy_id}"

        try:
            # Dynamically import the module
            module = importlib.import_module(module_name)

            # Check for required components
            if not hasattr(module, "ScalpWithTrendBot"):
                print(f"[WARN] Strategy '{strategy_id}' missing ScalpWithTrendBot class, skipping")
                continue

            if not hasattr(module, "Config"):
                print(f"[WARN] Strategy '{strategy_id}' missing Config class, skipping")
                continue

            # Get metadata (optional but recommended)
            metadata = getattr(module, "STRATEGY_METADATA", {})

            # Get lot sizes if available
            lot_sizes = getattr(module, "INDEX_LOT_SIZES", {})

            config_schema = build_config_schema(module, module.Config)
            default_config_instance = module.Config()
            default_config = serialize_config(default_config_instance, config_schema)

            # Build strategy info
            strategy_info = {
                "module": module,
                "name": metadata.get("name", strategy_id.replace("_", " ").title()),
                "description": metadata.get("description", f"Strategy: {strategy_id}"),
                "version": metadata.get("version", "1.0"),
                "bot_class": module.ScalpWithTrendBot,
                "config_class": module.Config,
                "features": metadata.get("features", ["EMA Trend Following", "ATR Filter", "OCO Orders"]),
                "has_trade_direction": metadata.get("has_trade_direction", True),
                "lot_sizes": lot_sizes,
                "author": metadata.get("author", "Unknown"),
                "config_schema": config_schema,
                "default_config": default_config,
            }

            strategies[strategy_id] = strategy_info
            print(f"[INFO] Discovered strategy: {strategy_info['name']} (v{strategy_info['version']})")

        except Exception as e:
            print(f"[ERROR] Failed to load strategy '{strategy_id}': {e}")
            continue

    if not strategies:
        print("[WARN] No strategies discovered! Check strategies folder.")

    return strategies

# Auto-discover on import
DISCOVERED_STRATEGIES = discover_strategies()

# Export for convenience
__all__ = ['discover_strategies', 'DISCOVERED_STRATEGIES']
