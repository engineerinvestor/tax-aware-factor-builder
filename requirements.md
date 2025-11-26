````markdown
# Tax-Aware Factor Portfolio Builder  
### Requirements & Design Document

## 1. Overview

This project is a **Jupyter notebook–driven Python tool** that helps investors build and analyze **factor-targeted, tax-aware portfolios** using funds from:

- **Vanguard**
- **Avantis**
- **Dimensional (DFA)**
- **AQR**

The notebook should work **out of the box** by downloading data via `yfinance`, with a **CSV-based fallback** for offline/custom usage.

Core capabilities:

1. Estimate **factor loadings** of funds (e.g., Fama–French + Momentum).
2. Optimize fund weights to hit **asset allocation** and **factor targets**.
3. Optimize **tax location** (which funds go in which accounts) to reduce tax drag.
4. Report **portfolio composition**, **factor exposures**, and **estimated tax impact**.

---

## 2. Goals

### 2.1 Primary goals

- Allow a user to:
  - Specify **tax profile**, **accounts**, and **target factor/asset allocation**.
  - Select from a **universe of Vanguard/Avantis/DFA/AQR funds**.
  - Automatically:
    - Pull or load **return data**.
    - Run **factor regressions**.
    - Optimize **pre-tax portfolio weights**.
    - Optimize **account-level tax location**.
    - Produce readable **tables & plots** summarizing the results.

### 2.2 Secondary goals

- Config-driven: easily re-use for multiple investors.
- Modular: functions can be reused outside the notebook.
- Extensible: future Monte Carlo, more factors, more managers.

---

## 3. Assumptions & Scope

- Investor is a **U.S. taxpayer** with:
  - At least one **Taxable** account.
  - One or more **Tax-Deferred** accounts (e.g., Trad 401(k), Trad IRA).
  - Optional **Tax-Free** accounts (Roth, HSA-as-IRA).

- Focus is on **public funds**:
  - Vanguard index funds & ETFs.
  - Avantis ETFs.
  - DFA mutual funds & ETFs.
  - AQR mutual funds (and/or ETFs when applicable).

- Factor model:
  - Minimum: **MKT**, **SMB**, **HML**.
  - Preferred: **MKT, SMB, HML, RMW, CMA, MOM**.

- Data sources:
  - Default: **`yfinance`** for historical prices/returns.
  - Fallback: **user CSV uploads** for fund returns.
  - Factor data from pre-downloaded **CSV** (e.g., Ken French library).

---

## 4. Data Sources & Modes

### 4.1 Global config

At the top of the notebook:

```python
DATA_MODE = "yfinance"   # options: "yfinance", "csv"
DATA_START_DATE = "2010-01-01"
DATA_FREQUENCY = "M"     # "D" for daily, "M" for monthly (preferred)
FACTOR_DATA_PATH = "./data/factors/fama_french_5_factors.csv"
CSV_FUND_DATA_DIR = "./data/funds"
````

### 4.2 yfinance mode (default)

* For each fund ticker (e.g., "AVUV", "VTI", "DFSV"):

  * Use `yfinance` to download **Adjusted Close** prices from `DATA_START_DATE` to today.
  * Convert prices to returns:

    * Daily or monthly, depending on `DATA_FREQUENCY`.
  * Store as `pd.Series` of periodic returns (indexed by date).

### 4.3 CSV mode (fallback)

* For each fund, load a CSV from `CSV_FUND_DATA_DIR`, named:
  `TICKER.csv` (e.g., `AVUV.csv`).

* Required columns:

  * `Date` (YYYY-MM-DD)
  * Either:

    * `AdjClose` (prices) → notebook computes returns, or
    * `Return` (periodic returns) → notebook uses directly.

### 4.4 Factor data

* A pre-downloaded factor CSV at `FACTOR_DATA_PATH` with at least:

  * `MKT_RF`, `SMB`, `HML`, `RMW`, `CMA`, `MOM`, `RF`
* Notebook loads and:

  * Parses `Date`.
  * Converts to same periodicity as fund returns (`DATA_FREQUENCY`).
  * Computes excess returns: `FundReturn - RF`.

---

## 5. Core Data Structures

Use either lightweight classes (`@dataclass`) or dicts.

### 5.1 Fund

```python
class Fund:
    ticker: str
    name: str
    manager: str              # "Vanguard", "Avantis", "DFA", "AQR"
    asset_class: str          # "US_Equity", "Intl_Equity", "Bond", "Alt", "REIT", etc.
    region: str               # "US", "Developed ex-US", "EM", "Global", etc.
    vehicle_type: str         # "ETF", "MutualFund"
    expense_ratio: float

    is_tax_efficient: bool    # ETF with in-kind redemptions, low turnover, etc.
    dividend_yield: float     # trailing 12m
    qualified_dividend_ratio: float   # 0–1
    cap_gain_distribution_yield: float
    income_character: str     # "mostly_dividends", "mostly_interest", "complex"

    allowed_accounts: list[str]   # e.g., ["Taxable", "TaxDeferred", "TaxFree"]

    historical_returns: pd.Series | None   # periodic returns
    factor_loadings: dict[str, float]      # "MKT", "SMB", "HML", "RMW", "CMA", "MOM"
    alpha: float | None
    regression_stats: dict | None          # R², t-stats, etc.
```

### 5.2 Account

```python
class Account:
    name: str             # "Taxable_Brokerage", "401k", "Roth_IRA"
    type: str             # "Taxable", "TaxDeferred", "TaxFree"
    current_value: float
    available_funds: list[str]  # tickers or "ANY"
    priority_for_tax_inefficient_assets: int  # e.g., 1=highest priority
```

### 5.3 InvestorProfile

```python
class InvestorProfile:
    federal_ordinary_rate: float         # e.g., 0.37
    federal_ltcg_qualdiv_rate: float     # e.g., 0.20 or 0.238
    state_income_rate: float             # e.g., 0.05
    uses_municipal_bonds: bool
    time_horizon_years: int
    # optional:
    risk_tolerance: str | float
```

### 5.4 TargetPortfolioSpec

```python
class TargetPortfolioSpec:
    # Asset class level
    equity_weight: float
    bond_weight: float
    alts_weight: float

    us_equity_share_of_equity: float
    intl_equity_share_of_equity: float
    reit_share_of_equity: float | None

    # Factor model
    factor_list: list[str]            # e.g. ["MKT", "SMB", "HML", "RMW", "MOM"]
    factor_targets: dict[str, float]  # e.g. {"MKT": 1.0, "SMB": 0.4, "HML": 0.4, ...}

    # Constraints
    min_weight_per_fund: float
    max_weight_per_fund: float
    min_weight_per_manager: dict[str, float] | None
    max_weight_per_manager: dict[str, float] | None
    min_weight_per_region: dict[str, float] | None
    max_weight_per_region: dict[str, float] | None

    # Sparsity preference (optional)
    max_number_of_funds: int | None
```

---

## 6. Notebook Workflow / Sections

### 6.1 Section A – Setup & Configuration

* Import libraries:

  * `numpy`, `pandas`, `matplotlib`
  * `scipy.optimize` and/or `cvxpy`
  * `yfinance`
  * `dataclasses` (if using dataclasses)

* Load:

  * Fund universe definition (CSV/JSON → list of `Fund` objects).
  * Factor data from `FACTOR_DATA_PATH`.

* Define or load:

  * `InvestorProfile`
  * `Account` list
  * `TargetPortfolioSpec`

* Cell to optionally load a **config JSON/YAML** that sets all of the above.

---

### 6.2 Section B – Fund Return Data (yfinance / CSV)

#### Functions

```python
def fetch_price_history_yf(
    ticker: str,
    start: str,
    end: str | None = None,
    frequency: str = "M"
) -> pd.Series:
    """
    Download adjusted close via yfinance, compute periodic returns, and return a Series.
    """


def fetch_all_fund_returns_yf(
    funds: list[Fund],
    start: str,
    frequency: str = "M"
) -> dict[str, pd.Series]:
    """
    For each fund, download returns via yfinance and return a dict {ticker: return_series}.
    """


def load_price_history_csv(
    filepath: str,
    frequency: str = "M"
) -> pd.Series:
    """
    Load CSV for a fund and return periodic returns.
    """


def load_all_fund_returns_csv(
    funds: list[Fund],
    base_dir: str,
    frequency: str = "M"
) -> dict[str, pd.Series]:
    """
    For each fund, load returns from {base_dir}/{ticker}.csv and return {ticker: return_series}.
    """
```

#### Logic

```python
if DATA_MODE == "yfinance":
    fund_return_series = fetch_all_fund_returns_yf(funds, start=DATA_START_DATE, frequency=DATA_FREQUENCY)
elif DATA_MODE == "csv":
    fund_return_series = load_all_fund_returns_csv(funds, base_dir=CSV_FUND_DATA_DIR, frequency=DATA_FREQUENCY)
else:
    raise ValueError("Unsupported DATA_MODE")

for fund in funds:
    if fund.ticker in fund_return_series:
        fund.historical_returns = fund_return_series[fund.ticker]
    else:
        # Warn or mark as unavailable
        ...
```

---

### 6.3 Section C – Factor Data & Factor Estimation

#### Factor loading estimation

```python
def load_factor_data_default(
    filepath: str,
    frequency: str = "M"
) -> pd.DataFrame:
    """
    Load factor data (MKT, SMB, HML, RMW, CMA, MOM, RF) from CSV.
    Return DataFrame indexed by Date with the chosen frequency.
    """


def estimate_factor_loadings(
    fund_returns: pd.Series,
    factor_data: pd.DataFrame,
    factor_list: list[str]
) -> dict:
    """
    Regress excess fund returns on selected factor returns.
    Returns:
      {
        "alpha": float,
        "betas": {factor_name: beta_value},
        "r2": float,
        "tstats": {factor_name: t_stat, "alpha": t_stat},
        "n_obs": int
      }
    """


def apply_factor_estimation_to_all_funds(
    funds: list[Fund],
    factor_data: pd.DataFrame,
    factor_list: list[str]
) -> None:
    """
    Estimate and store factor loadings + regression stats in each Fund.
    """
```

* For each fund:

  * Align dates with `factor_data`.
  * Use excess returns: `fund_return - RF`.
  * Store results in `fund.factor_loadings`, `fund.alpha`, `fund.regression_stats`.

* Outputs:

  * Table of funds with factor loadings.
  * Example scatter/line plots comparing actual vs model-predicted returns.

---

### 6.4 Section D – Pre-Tax Portfolio Optimization

Goal: find **fund weights** that:

* Hit **asset class targets**.
* Approximate **factor targets**.
* Obey constraints.

#### Key functions

```python
def compute_portfolio_factor_loadings(
    weights: np.ndarray,
    funds: list[Fund],
    factor_list: list[str]
) -> dict[str, float]:
    """
    Weighted sum of fund factor loadings.
    """


def compute_portfolio_asset_class_breakdown(
    weights: np.ndarray,
    funds: list[Fund]
) -> dict[str, float]:
    """
    Compute share of total weight per asset_class, region, manager, etc.
    """
```

#### Optimization model

* Decision vector: `w` (length = number of funds).

Constraints:

* `sum(w) = 1` (for full portfolio or per sleeve).
* `min_weight_per_fund <= w_i <= max_weight_per_fund`.
* Asset class constraints:

  * e.g. `sum(w_i for US_Equity) = equity_weight * us_equity_share_of_equity`.
  * `sum(w_i for Intl_Equity) = equity_weight * intl_equity_share_of_equity`.
  * `sum(w_i for Bond) = bond_weight`.
  * etc.
* Manager/region constraints if defined.

Objective:

* Minimize **factor deviation**:

  * `∑_f (FactorPortfolio[f] - FactorTarget[f])²`
* Optional secondary penalties:

  * Deviation from **benchmark weights** (tracking error proxy).
  * Number of funds used (sparsity).

Implementation:

```python
def build_pre_tax_optimization(
    funds: list[Fund],
    target_spec: TargetPortfolioSpec
):
    """
    Build an optimization problem (cvxpy or scipy) to find w.
    """


def solve_pre_tax_optimization(
    funds: list[Fund],
    target_spec: TargetPortfolioSpec
) -> np.ndarray:
    """
    Solve the optimization and return optimal weights w.
    """
```

Outputs:

* Table: `Fund | Manager | AssetClass | Weight | Factor Loadings`.
* Bar chart: **Target vs Actual** factor exposures.

---

### 6.5 Section E – Tax Modeling

Goal: estimate **annual tax drag** per fund in different account types.

#### Tax drag per fund per account type

For each **fund** and **account type** ("Taxable", "TaxDeferred", "TaxFree"):

```python
def estimate_tax_drag_per_fund(
    fund: Fund,
    investor: InvestorProfile,
    account_type: str
) -> float:
    """
    Return expected annual tax drag (as a decimal, e.g. 0.005 = 50 bps).
    """
```

Logic (simplified):

* For **Taxable**:

  * `div_tax = dividend_yield * (qualified_dividend_ratio * ltcg_rate                              + (1 - qualified_dividend_ratio) * ordinary_rate)`
  * `cg_tax = cap_gain_distribution_yield * ltcg_rate`
  * `other_tax` for interest-like or non-qualified distributions if needed.
  * `tax_drag = div_tax + cg_tax + other_tax`

* For **TaxDeferred**:

  * Treat annual **tax drag = 0** for ongoing distributions (growth is tax-deferred).
  * Optionally add a simple haircut later for “eventual ordinary taxation on withdrawal”.

* For **TaxFree** (Roth/HSA):

  * `tax_drag = 0`

Build a table:

```python
def build_tax_drag_table(
    funds: list[Fund],
    investor: InvestorProfile,
    accounts: list[Account]
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by [fund, account_type] with tax_drag values.
    """
```

#### After-tax expected return (optional)

```python
def estimate_after_tax_expected_return(
    fund: Fund,
    expected_pre_tax_return: float,
    tax_drag: float
) -> float:
    """
    Approximate after-tax expected return for a given account type.
    """
```

Expected pre-tax return can be:

* A simple factor-premia model.
* User-provided.
* Or left as a relative comparison only.

---

### 6.6 Section F – Tax-Location Optimization

Goal: assign fund holdings to accounts to minimize **portfolio-wide tax drag**, given:

* Target fund weights from pre-tax optimization.
* Account sizes and types.

#### Decision variables

* `x[fund, account]`: **dollar amount** of each fund in each account.

Constraints:

1. For each account `a`:

   * `∑_f x[f, a] = account_value[a]`
2. For each fund `f`:

   * `∑_a x[f, a] = w_target[f] * total_portfolio_value`
3. Non-negativity: `x[f, a] >= 0`.
4. Availability: if fund not available in account → `x[f, a] = 0`.

Objective:

* Minimize total tax drag:

  [
  \text{TaxDragTotal} = \sum_{f,a} \left( \frac{x[f,a]}{\text{TotalValue}} \times \text{tax_drag}(f,a) \right)
  ]

Implementation:

```python
def build_tax_location_problem(
    w_target: np.ndarray,
    funds: list[Fund],
    accounts: list[Account],
    tax_drag_table: pd.DataFrame
):
    """
    Build LP/QP to allocate fund holdings across accounts.
    """


def solve_tax_location_problem(
    w_target: np.ndarray,
    funds: list[Fund],
    accounts: list[Account],
    tax_drag_table: pd.DataFrame
) -> np.ndarray:
    """
    Solve for allocation matrix x[f,a] and return as a 2D array or DataFrame.
    """
```

Outputs:

* Table per account:

  * `Account | Fund | DollarAmount | % of Account | % of Total`.
* Summary of aggregate tax drag:

  * **Optimized** vs a naive (pro-rata) allocation.

---

### 6.7 Section G – Reporting & Visualization

Produce user-friendly outputs:

1. **Portfolio Composition**

   * Overall:

     * Bar chart by **asset class**.
     * Bar chart by **manager** (Vanguard/Avantis/DFA/AQR).
   * By account:

     * Table or stacked bar chart of fund allocations.

2. **Factor Exposures**

   * Table of:

     * Target factor loadings.
     * Actual portfolio factor loadings.
   * Bar chart: **Target vs Actual** for each factor.

3. **Tax View**

   * Table:

     * `Fund | AccountType | WeightedTaxDrag | After-Tax Expected Return (optional)`.
   * Summary metrics:

     * `Total expected annual tax drag (bps)`.
     * Difference vs naive tax location.

4. **Optional Backtest**

   * If enough history, backtest:

     * Optimized portfolio.
     * Simple benchmark (e.g. VTI+VXUS+AGG).
   * Show:

     * CAGR, volatility, Sharpe, max drawdown.

---

## 7. Function Summary

For convenience, the key function families:

### 7.1 Data I/O

* `load_fund_universe(filepath) -> list[Fund]`
* `load_factor_data_default(filepath, frequency) -> pd.DataFrame`
* `fetch_all_fund_returns_yf(funds, start, frequency) -> dict[str, pd.Series]`
* `load_all_fund_returns_csv(funds, base_dir, frequency) -> dict[str, pd.Series]`

### 7.2 Factor Analytics

* `estimate_factor_loadings(fund_returns, factor_data, factor_list) -> dict`
* `apply_factor_estimation_to_all_funds(funds, factor_data, factor_list) -> None`

### 7.3 Portfolio Math

* `compute_portfolio_factor_loadings(weights, funds, factor_list) -> dict[str, float]`
* `compute_portfolio_asset_class_breakdown(weights, funds) -> dict[str, float]`

### 7.4 Pre-Tax Optimization

* `build_pre_tax_optimization(funds, target_spec)`
* `solve_pre_tax_optimization(funds, target_spec) -> np.ndarray`

### 7.5 Tax Modeling

* `estimate_tax_drag_per_fund(fund, investor, account_type) -> float`
* `build_tax_drag_table(funds, investor, accounts) -> pd.DataFrame`
* `estimate_after_tax_expected_return(fund, expected_pre_tax_return, tax_drag) -> float` (optional)

### 7.6 Tax-Location Optimization

* `build_tax_location_problem(w_target, funds, accounts, tax_drag_table)`
* `solve_tax_location_problem(w_target, funds, accounts, tax_drag_table) -> allocation_matrix`

### 7.7 Reporting

* `summarize_portfolio(weights, allocation_matrix, funds, accounts) -> dict[pd.DataFrame]`
* `plot_factor_exposures(target_factors, portfolio_factors)`
* `plot_asset_allocation_by_class(weights, funds)`
* `plot_account_allocations(allocation_matrix, funds, accounts)`
* Visualizations:
  * Factor targets vs portfolio: grouped bars and optional radar chart for multi-factor shape.
  * Asset allocation breakdown: stacked bars by asset_class/manager/region with labels.
  * Tax drag heatmap: fund vs account-type showing drag (bps) to highlight placement.
  * Tax-location allocation: stacked bars per account, optimized vs naive side-by-side.
  * Factor regression diagnostics: actual vs predicted returns scatter/line and residuals for outliers.
  * Monte Carlo: KDE/violin or histogram for terminal after-tax wealth vs benchmark, quantile table, and optional fan chart of median path with bands.

---

## 8. UX & Usability Requirements

* **Single parameter cell** near top:

  * `DATA_MODE`, `DATA_START_DATE`, `DATA_FREQUENCY`.
  * Paths for CSVs.
  * Basic investor & target parameters (or config file path).

* **Config-driven option**:

  * Ability to load:

    * Investor profile.
    * Accounts.
    * Target portfolio spec.
  * From a JSON/YAML file.

* **Error handling**:

  * If ticker can’t be pulled from yfinance → clear warning.
  * If CSV missing or malformed → clear error.
  * Validate:

    * Account balances sum to total portfolio.
    * Target weights sum properly.
    * Funds referenced in config exist in universe.

* **Clear section headings** in notebook:

  1. Setup & Config
  2. Data Loading (Funds & Factors)
  3. Factor Estimation
  4. Pre-Tax Optimization
  5. Tax Modeling
  6. Tax-Location Optimization
  7. Reports & Plots

---

## 9. Future Extensions (V2+)

* `ipywidgets` sliders for:

  * Equity %.
  * Small/value tilt strength.
  * AQR/alt allocation %.
* Monte Carlo simulations:

  * Stochastic factor premia → wealth distribution after-tax.
* Extended managers & universes (Schwab, iShares, etc.).
* Automated PDF/HTML report generation for clients.

---

## 10. Disclaimer

This tool is for **educational and research purposes only**.
It does **not** constitute tax, legal, or investment advice. Users should validate outputs against their own circumstances and consult appropriate professionals.

```

::contentReference[oaicite:0]{index=0}
```
