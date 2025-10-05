"""
Trading Strategies Package
Contains various trading strategy implementations with auto-discovery
"""

import importlib
from pathlib import Path
from typing import Dict, Any

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
                "author": metadata.get("author", "Unknown")
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
