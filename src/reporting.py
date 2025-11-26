import math
from typing import Dict, Iterable, Optional

import numpy as np

from src.models import Fund, TargetPortfolioSpec


def _format_breakdown_row(items: Dict[str, float], top_n: int = 3) -> str:
    if not items:
        return "n/a"
    sorted_items = sorted(items.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return ", ".join(f"{k}: {v:.2%}" for k, v in sorted_items)


def _factor_lines(factor_list: Iterable[str], portfolio: Dict[str, float], targets: Dict[str, float]) -> str:
    lines = []
    for f in factor_list:
        tgt = targets.get(f, 0.0)
        port = portfolio.get(f, 0.0)
        diff = port - tgt
        lines.append(f"  {f}: target {tgt:.3f}, portfolio {port:.3f}, diff {diff:+.3f}")
    return "\n".join(lines)


def _weight_stats(weights: np.ndarray) -> str:
    nonzero = weights[np.abs(weights) > 0]
    if nonzero.size == 0:
        return "no non-zero weights"
    return (
        f"count {len(weights)}, non-zero {len(nonzero)}, "
        f"min {nonzero.min():.4f}, max {nonzero.max():.4f}, "
        f"mean {nonzero.mean():.4f}"
    )


def format_run_summary(
    *,
    config: Dict[str, object],
    target_spec: TargetPortfolioSpec,
    weights: np.ndarray,
    funds: Iterable[Fund],
    portfolio_factors: Dict[str, float],
    asset_breakdown: Dict[str, Dict[str, float]],
    tax_stats: Optional[Dict[str, float]] = None,
) -> str:
    """
    Build a human-readable text summary of inputs and outputs for a notebook run.
    """
    config_lines = [
        f"Data mode: {config.get('data_mode')}",
        f"Start date: {config.get('start_date')}",
        f"Frequency: {config.get('frequency')}",
        f"Factor file: {config.get('factor_data_path')}",
        f"Weight floor: {config.get('weight_floor', 0.0)}",
    ]

    targets_lines = _factor_lines(target_spec.factor_list, portfolio_factors, target_spec.factor_targets)
    breakdown_lines = [
        f"Asset class: {_format_breakdown_row(asset_breakdown.get('asset_class', {}))}",
        f"Manager: {_format_breakdown_row(asset_breakdown.get('manager', {}))}",
        f"Region: {_format_breakdown_row(asset_breakdown.get('region', {}))}",
    ]

    tax_line = ""
    if tax_stats:
        opt = tax_stats.get("optimized_bps")
        naive = tax_stats.get("naive_bps")
        if opt is not None and naive is not None and not (math.isnan(opt) or math.isnan(naive)):
            tax_line = f"\nTax drag (bps): optimized {opt:.2f}, naive {naive:.2f}, delta {(naive - opt):.2f}"

    summary = "\n".join(
        [
            "=== Inputs ===",
            *config_lines,
            "",
            "=== Targets vs Portfolio ===",
            targets_lines,
            "",
            "=== Weights ===",
            _weight_stats(np.array(weights)),
            "",
            "=== Breakdown ===",
            *breakdown_lines,
            tax_line.strip(),
        ]
    ).strip()
    return summary
