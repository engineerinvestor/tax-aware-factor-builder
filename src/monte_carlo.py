import numpy as np
import pandas as pd

from src.models import Account, Fund


def simulate_factor_premia(T: int, n_sims: int, factor_list: list[str], mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Draw factor premia paths using multivariate normal.
    Returns array shape (n_sims, T, n_factors).
    """
    n_factors = len(factor_list)
    draws = np.random.multivariate_normal(mean=mu, cov=cov, size=(n_sims, T))
    return draws.reshape(n_sims, T, n_factors)


def simulate_portfolio_paths(factor_premia_sims: np.ndarray, portfolio_betas: dict[str, float], rf: float) -> np.ndarray:
    """
    Compute portfolio returns given factor simulations and portfolio betas.
    Returns array shape (n_sims, T).
    """
    factor_list = list(portfolio_betas.keys())
    beta_vec = np.array([portfolio_betas[f] for f in factor_list])
    # factor_premia_sims: n_sims x T x n_factors
    proj = factor_premia_sims @ beta_vec
    return proj + rf


def simulate_after_tax_wealth(
    initial_total_value: float,
    allocation_matrix: np.ndarray,
    tax_drag_table: pd.DataFrame,
    funds: list[Fund],
    accounts: list[Account],
    portfolio_returns: np.ndarray,
) -> np.ndarray:
    """
    Track taxable, tax-deferred, tax-free balances over time using portfolio_returns (n_sims x T).
    Applies annual tax drag for taxable holdings per fund/account.
    Returns array of terminal after-tax wealth (n_sims,).
    """
    n_sims, T = portfolio_returns.shape
    total_value = allocation_matrix.sum()
    if total_value <= 0:
        raise ValueError("Allocation matrix must have positive total")

    # Build account-type masks
    acct_types = [a.type for a in accounts]
    taxable_idx = [i for i, a in enumerate(acct_types) if a.lower() == "taxable"]
    taxdeferred_idx = [i for i, a in enumerate(acct_types) if a.lower() in {"taxdeferred", "tax_deferred"}]
    taxfree_idx = [i for i, a in enumerate(acct_types) if a.lower() in {"taxfree", "tax_free"}]

    # Starting balances scaled to initial_total_value
    scale = initial_total_value / total_value
    alloc_scaled = allocation_matrix * scale

    taxable_bal = alloc_scaled[:, taxable_idx].sum(axis=1) if taxable_idx else np.zeros(alloc_scaled.shape[0])
    taxdeferred_bal = alloc_scaled[:, taxdeferred_idx].sum(axis=1) if taxdeferred_idx else np.zeros(alloc_scaled.shape[0])
    taxfree_bal = alloc_scaled[:, taxfree_idx].sum(axis=1) if taxfree_idx else np.zeros(alloc_scaled.shape[0])

    # Precompute weighted tax drag for taxable sleeve
    taxable_drag = 0.0
    if taxable_idx:
        for i, fund in enumerate(funds):
            for j in taxable_idx:
                drag_key = (fund.ticker, accounts[j].type)
                drag = tax_drag_table.loc[drag_key, "tax_drag"] if drag_key in tax_drag_table.index else 0.0
                weight = alloc_scaled[i, j] / taxable_bal.sum() if taxable_bal.sum() else 0.0
                taxable_drag += drag * weight

    terminal_wealth = np.zeros(n_sims)
    for s in range(n_sims):
        taxable = taxable_bal.sum()
        taxdeferred = taxdeferred_bal.sum()
        taxfree = taxfree_bal.sum()
        for t in range(T):
            r = portfolio_returns[s, t]
            taxable = taxable * (1 + r - taxable_drag)
            taxdeferred = taxdeferred * (1 + r)
            taxfree = taxfree * (1 + r)
        # Apply ordinary tax on tax-deferred at withdrawal (simplified)
        after_tax_taxdeferred = taxdeferred * (1 - 0.25)
        terminal_wealth[s] = taxable + after_tax_taxdeferred + taxfree
    return terminal_wealth
