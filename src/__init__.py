"""
Core package for tax-aware factor portfolio builder.
"""

from src.models import default_target_portfolio_spec  # noqa: F401
from src.tax import estimate_tax_drag_per_fund, build_tax_drag_table, estimate_after_tax_expected_return  # noqa: F401
from src.monte_carlo import simulate_factor_premia, simulate_portfolio_paths, simulate_after_tax_wealth  # noqa: F401

__all__ = [
    "data_loaders",
    "factors",
    "optimization",
    "tax",
    "monte_carlo",
    "models",
    "default_target_portfolio_spec",
    "estimate_tax_drag_per_fund",
    "build_tax_drag_table",
    "estimate_after_tax_expected_return",
    "simulate_factor_premia",
    "simulate_portfolio_paths",
    "simulate_after_tax_wealth",
]
