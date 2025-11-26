import numpy as np

from src.models import Fund, TargetPortfolioSpec
from src.optimization import _apply_weight_floor, build_pre_tax_optimization, solve_pre_tax_optimization


def _sample_funds():
    return [
        Fund(
            ticker="US1",
            name="US Fund",
            manager="Vanguard",
            asset_class="US_Equity",
            region="US",
            vehicle_type="ETF",
            expense_ratio=0.001,
            factor_loadings={"MKT": 1.0, "SMB": 0.2},
        ),
        Fund(
            ticker="INTL1",
            name="Intl Fund",
            manager="Avantis",
            asset_class="Intl_Equity",
            region="Global",
            vehicle_type="ETF",
            expense_ratio=0.0012,
            factor_loadings={"MKT": 0.9, "SMB": 0.1},
        ),
        Fund(
            ticker="BOND1",
            name="Bond Fund",
            manager="Vanguard",
            asset_class="Bond",
            region="US",
            vehicle_type="ETF",
            expense_ratio=0.0005,
            factor_loadings={"MKT": 0.1, "SMB": 0.0},
        ),
    ]


def _target_spec():
    return TargetPortfolioSpec(
        equity_weight=0.7,
        bond_weight=0.25,
        alts_weight=0.05,
        us_equity_share_of_equity=0.6,
        intl_equity_share_of_equity=0.4,
        reit_share_of_equity=None,
        factor_list=["MKT", "SMB"],
        factor_targets={"MKT": 0.95, "SMB": 0.15},
        min_weight_per_fund=0.0,
        max_weight_per_fund=0.8,
    )


def test_build_pre_tax_optimization_shapes():
    funds = _sample_funds()
    target = _target_spec()
    problem, w = build_pre_tax_optimization(funds, target)
    assert w.shape == (len(funds),)
    assert len(problem.constraints) >= 4


def test_solve_pre_tax_optimization_respects_weights():
    funds = _sample_funds()
    target = _target_spec()
    weights = solve_pre_tax_optimization(funds, target)
    assert abs(weights.sum() - 1.0) < 1e-6
    # Asset class checks
    us_weight = weights[0]
    intl_weight = weights[1]
    bond_weight = weights[2]
    assert us_weight >= target.equity_weight * target.us_equity_share_of_equity - 1e-6
    assert intl_weight >= target.equity_weight * target.intl_equity_share_of_equity - 1e-6
    assert bond_weight >= target.bond_weight - 1e-6
    # Factor proximity: should be finite and reasonable
    assert np.isfinite(weights).all()


def test_apply_weight_floor_zeroes_tiny_positions():
    weights = np.array([0.5, 0.0001, 0.2, 0.2999])
    floored = _apply_weight_floor(weights, floor=0.01)
    assert floored[1] == 0.0
    assert abs(floored.sum() - 1.0) < 1e-6
    # Remaining weights keep proportional relationships
    original_nonzero = weights[[0, 2, 3]]
    floored_nonzero = floored[[0, 2, 3]]
    ratio_original = original_nonzero / original_nonzero.sum()
    assert np.allclose(floored_nonzero, ratio_original, atol=1e-6)
