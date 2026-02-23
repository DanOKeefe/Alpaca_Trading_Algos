from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.strategies.base import Strategy

_REGISTRY: dict[str, Strategy] = {}


def register(key: str, strategy: Strategy) -> None:
    """Register a strategy instance under a lookup key."""
    _REGISTRY[key] = strategy


def get_strategy(key: str) -> Strategy:
    """Look up a strategy by key. Raises KeyError if not found."""
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Unknown strategy '{key}'. Available: {available}"
        )
    return _REGISTRY[key]


def list_strategies() -> list[str]:
    """Return all registered strategy keys."""
    return sorted(_REGISTRY.keys())


def _register_defaults() -> None:
    """Register all built-in strategies."""
    from src.strategies.black_litterman import BlackLittermanStrategy
    from src.strategies.gmv import GMVStrategy
    from src.strategies.max_sharpe import MaxSharpeStrategy
    from src.strategies.momentum import MomentumStrategy
    from src.strategies.risk_parity import RiskParityStrategy

    register("gmv", GMVStrategy())
    register("max_sharpe", MaxSharpeStrategy())
    register("risk_parity", RiskParityStrategy())
    register("momentum", MomentumStrategy())
    register("black_litterman", BlackLittermanStrategy())


_register_defaults()
