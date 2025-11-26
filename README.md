# Tax-Aware Factor Portfolio Builder

Educational use only; not tax, legal, or investment advice.

## Quickstart
- Create venv: `python3 -m venv .venv`
- Activate:
  - macOS/Linux: `source .venv/bin/activate`
  - Windows (PowerShell): `.venv\\Scripts\\Activate.ps1`
- Upgrade pip (inside venv): `pip install --upgrade pip`
- Install deps: `pip install -r requirements.txt`
- Run tests: `pytest`

## Notebook
- Launch Jupyter (inside venv): `jupyter notebook`
- Open `notebooks/tax_aware_factor_portfolio.ipynb`.
- Configure top cell: `DATA_MODE`, `DATA_START_DATE`, `DATA_FREQUENCY`, `CSV_FUND_DATA_DIR`, `FACTOR_DATA_PATH`.
- Switch data mode:
  - `DATA_MODE = "yfinance"` for live downloads.
  - `DATA_MODE = "csv"` to use local files in `data/funds/`.

## Data Files
- Fund CSVs live in `data/funds/` named `{TICKER}.csv` with `Date` plus `AdjClose` or `Return`.
- Factor CSV in `data/factors/` with columns `MKT_RF, SMB, HML, RMW, CMA, MOM, RF`. The included US/Developed/Developed ex-US/Emerging factor files and momentum series are sourced from the Kenneth R. French Data Library (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html); please review and comply with the library's terms of use when redistributing.
- Global factor sets are included in `data/factors/`:
  - US (`fama_french_5_factors.csv`), Developed (`Developed_5_Factors.csv`), Developed ex-US (`Developed_ex_US_5_Factors.csv` + momentum), and Emerging (`Emerging_5_Factors.csv` + momentum).
  - Load all of them at once via `load_factor_data_by_region("./data/factors", frequency="M")` and pass the mapping into `apply_factor_estimation_to_all_funds` to regress funds against the right regional factors.

## Running Tests
- From repo root (venv active): `pytest`

## Local Dev Environment
Create and use a virtual env, install deps, and run tests:
```bash
cd /path/to/factor-weighted-portfilio-analyzer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest
```
If you run into a broken pip bundle on macOS (missing `pip._vendor.*` errors), reinstall pip inside the venv with `python -m pip install --upgrade pip` to refresh its vendored modules.

## Notes
- Network may be required for `yfinance` mode.
- Sample CSVs are included for offline testing.***
