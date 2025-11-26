# Tax-Aware Factor Portfolio Builder

## Abstract
This paper describes an open-source Python toolchain that builds tax-aware, factor-tilted portfolios across U.S. and global asset classes. The system estimates fund-level factor exposures, targets user-specified factor and asset allocation profiles, enforces tax-location heuristics, and quantifies tax drag. We outline the motivation, data pipeline, regression methodology, optimization problem, tax model, and Monte Carlo stress testing. Implementation details and example use cases are provided to illustrate how practitioners can adapt the framework to their own policy constraints.

## Motivation
1. **Factor-consistent allocations.** Investors want exposure to rewarded factors (value, size, profitability, investment, momentum) while respecting tracking-error, liquidity, and diversification requirements.
2. **Tax efficiency.** Asset location materially affects after-tax wealth; naive allocations leave basis in tax-inefficient accounts and increase tax drag.
3. **Global coverage.** U.S.-only factor data leads to biased regression estimates for global funds. The tool ingests Fama-French factor sets for U.S., Developed, Developed ex-U.S., and Emerging markets.
4. **Operational realism.** Outputs should avoid dust weights, enforce per-fund/manager/region bounds, and remain explainable to stakeholders.

## Data Pipeline
- **Prices/returns:** `yfinance` (default) or CSV fallback; converted to monthly (`ME`) or daily returns.
- **Factors:** Ken French library CSVs (5 factors + MOM) for U.S., Developed, Developed ex-U.S., Emerging. Missing values (`-99.99`) are sanitized; momentum can be merged from WML/Mom files.
- **Fund metadata:** Manager, asset class, region, vehicle type, expense ratio, tax characteristics.

## Factor Regression
For a fund with returns $ r_t $ and factors $ F_t $, excess return is
$r^{e}_t = r_t - RF_t.$
We estimate
$r^{e}_t = \alpha + \beta^\top F_t + \varepsilon_t$
via OLS with intercept. Diagnostics include \( R^2 \), \( t \)-stats, and sample size. Region-aware factor sets are chosen based on fund region metadata (U.S., Developed, Developed ex-U.S., Emerging), falling back gracefully if a region is missing.

## Optimization Problem
Let $ w \in \mathbb{R}^n $ be fund weights, $ B \in \mathbb{R}^{k \times n} $ factor loadings, and $ f^\star $ target factors. We solve
$\min_w \|Bw - f^\star\|_2^2$
subject to:
- Budget: $ \sum_i w_i = 1 $
- Box: $ w_{\min} \le w_i \le w_{\max} $
- Asset-class mins: U.S. equity, Intl equity, bonds, alts, REITs
- Manager/region min/max (optional)
- **Weight floor regularization:** post-solution, weights with \( |w_i| < \tau \) (default 1%) are set to zero and the remainder renormalized. This removes operationally insignificant positions.

Preferred solvers: ECOS, OSQP, SCS, CLARABEL via `cvxpy`.

## Tax Model and Location
- Each fund carries expected dividend yield, qualified dividend ratio, capital gains distribution yield, and income character.
- Investor profile: ordinary and LTCG/qualified rates, state tax, muni usage, horizon.
- **Tax drag table:** Estimates per-fund annual drag in each account type (Taxable, Tax-Deferred, Tax-Free).
- **Tax location optimization:** Assigns fund weights to accounts to minimize aggregate drag while honoring account availability and priorities.

## Monte Carlo (Optional)
Simulates factor premia via multivariate normal draws:
$F_{t} \sim \mathcal{N}(\mu, \Sigma),\quad R_{p,t} = \alpha_p + \beta_p^\top F_t.$
Aggregates wealth paths (pre- or after-tax) to assess dispersion and tail outcomes.

## Reporting
- **Text summary:** Human-readable run summary (inputs, factor target vs. achieved, weight stats, asset/manager/region breakdown, tax drag delta).
- **Tables/plots:** Weight tables, factor bar charts, tax drag comparison.

## Implementation Notes
- Code is pure Python with `pandas`, `numpy`, `cvxpy`, `yfinance`, and `pytest` test suite.
- Region-aware factor ingestion lives in `src/data_loaders.py`; regression in `src/factors.py`; optimization in `src/optimization.py`; tax logic in `src/tax.py` and `src/tax_location.py`; reporting in `src/reporting.py`.
- Weight floor defaults to 0 but can be set (e.g., 1%) via `TargetPortfolioSpec.weight_floor`.

## Use Cases
1. **Advisor portfolio construction:** Build factor-tilted, tax-aware ETF portfolios with manager and region caps; export weight tables for implementation.
2. **Model-policy testing:** Stress-test how factor targets and weight floors interact with tax constraints; compare tax drag between naive and optimized locations.
3. **Academic prototyping:** Experiment with factor timing, alternative factor sets, or bespoke constraints using the open `cvxpy` formulation.
4. **Global fund screening:** Regress non-U.S. funds on appropriate regional factor sets to avoid U.S.-centric bias in estimated betas.

## Limitations and Extensions
- Factor coverage depends on available regional datasets; country-specific factors are not modeled.
- Transaction costs, liquidity, and turnover are not yet penalized.
- Tax rules are U.S.-centric; multi-jurisdiction support would require additional modeling.
- Extensions: robust optimization for factor uncertainty, transaction cost penalty, ESG constraints, or Black-Litterman overlays.

## Conclusion
The Tax-Aware Factor Portfolio Builder integrates factor regression, constrained optimization, and tax-location logic into a reproducible Python workflow. By combining global factor data, operational weight controls, and tax drag minimization, it delivers implementable portfolios that align with both investment policy and after-tax objectives.
