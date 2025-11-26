import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.models import Fund

logger = logging.getLogger(__name__)


def _read_factor_file(filepath: str) -> pd.DataFrame:
    """
    Robustly read Ken French style factor CSVs that may have leading text rows.
    """
    try:
        df = pd.read_csv(filepath)
        if "Date" in df.columns or any("Mkt" in col or "WML" in col or "Mom" in col for col in df.columns):
            return df
    except (UnicodeDecodeError, pd.errors.ParserError):
        df = None

    with open(filepath, "r", encoding="latin1") as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if any(token in line for token in ("Mkt-RF", "Mkt_RF", "Mom", "WML")) and "," in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Unable to locate factor header row in {filepath}")
    return pd.read_csv(filepath, skiprows=header_idx)


def _returns_from_prices(prices: pd.Series, frequency: str) -> pd.Series:
    freq = "ME" if frequency == "M" else frequency
    prices = prices.dropna().sort_index()
    if prices.empty:
        return prices
    if freq == "D":
        return prices.pct_change().dropna()
    if freq == "ME":
        monthly = prices.resample("ME").last()
        return monthly.pct_change().dropna()
    raise ValueError(f"Unsupported frequency: {frequency}")


def _resample_returns(returns: pd.Series, frequency: str) -> pd.Series:
    freq = "ME" if frequency == "M" else frequency
    returns = returns.dropna().sort_index()
    if returns.empty:
        return returns
    if freq == "D":
        return returns
    if freq == "ME":
        compounded = (1 + returns).resample("ME").prod() - 1
        return compounded.dropna()
    raise ValueError(f"Unsupported frequency: {frequency}")


def fetch_price_history_yf(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    frequency: str = "M",
) -> pd.Series:
    """
    Download adjusted close via yfinance, compute periodic returns, and return a Series.
    """
    data = None
    returns = pd.Series(dtype=float)
    # Try multiple strategies similar to the working example
    for auto_adjust in (False, True):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=auto_adjust)
        except Exception as exc:  # pragma: no cover - network issues
            logger.warning("yfinance download failed for %s (auto_adjust=%s): %s", ticker, auto_adjust, exc)
            continue
        price_col = "Adj Close" if "Adj Close" in data else "Close" if "Close" in data else None
        if price_col is None or data.empty:
            logger.warning("No price data for %s (auto_adjust=%s)", ticker, auto_adjust)
            continue
        returns = _returns_from_prices(data[price_col], frequency=frequency)
        if not returns.empty:
            return returns

    # Fallback via Ticker.history
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if not hist.empty and "Close" in hist:
            returns = _returns_from_prices(hist["Close"], frequency=frequency)
    except Exception as exc:  # pragma: no cover - network issues
        logger.warning("Ticker.history failed for %s: %s", ticker, exc)

    if returns.empty:
        logger.warning("No returns computed for %s", ticker)
    return returns


def fetch_all_fund_returns_yf(
    funds: List[Fund],
    start: str,
    frequency: str = "M",
) -> Dict[str, pd.Series]:
    """
    For each fund, download returns via yfinance and return a dict {ticker: return_series}.
    """
    results: Dict[str, pd.Series] = {}
    for fund in funds:
        try:
            series = fetch_price_history_yf(fund.ticker, start=start, frequency=frequency)
            if not series.empty:
                results[fund.ticker] = series
            else:
                logger.warning("No return data for %s", fund.ticker)
        except Exception as exc:  # pragma: no cover - network issues are logged
            logger.warning("Failed to fetch %s: %s", fund.ticker, exc)
    return results


def load_price_history_csv(filepath: str, frequency: str = "M") -> pd.Series:
    """
    Load CSV for a fund and return periodic returns.
    Accepts a Date column plus either AdjClose (prices) or Return (periodic returns).
    """
    df = pd.read_csv(filepath)
    if "Date" not in df.columns:
        raise ValueError("CSV must include a Date column")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    if "Return" in df.columns:
        returns = df["Return"].astype(float)
        return _resample_returns(returns, frequency=frequency)

    if "AdjClose" in df.columns:
        prices = df["AdjClose"].astype(float)
        return _returns_from_prices(prices, frequency=frequency)

    raise ValueError("CSV must include either AdjClose or Return column")


def load_all_fund_returns_csv(
    funds: List[Fund],
    base_dir: str,
    frequency: str = "M",
) -> Dict[str, pd.Series]:
    """
    For each fund, load returns from {base_dir}/{ticker}.csv and return {ticker: return_series}.
    """
    results: Dict[str, pd.Series] = {}
    for fund in funds:
        path = os.path.join(base_dir, f"{fund.ticker}.csv")
        if not os.path.exists(path):
            logger.warning("Missing CSV for %s at %s", fund.ticker, path)
            continue
        try:
            series = load_price_history_csv(path, frequency=frequency)
            if not series.empty:
                results[fund.ticker] = series
            else:
                logger.warning("Loaded CSV for %s but returns are empty", fund.ticker)
        except Exception as exc:
            logger.warning("Failed to load CSV for %s: %s", fund.ticker, exc)
    return results


def load_factor_data_default(filepath: str, frequency: str = "M", momentum_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load factor data (MKT, SMB, HML, RMW, CMA, MOM, RF) from CSV.
    Return DataFrame indexed by Date with the chosen frequency.
    """
    required_cols = {"MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"}
    df = _read_factor_file(filepath)

    # Handle Ken French monthly file format (first column is YYYYMM, no Date header)
    if "Date" not in df.columns and any(col.startswith("Mkt") for col in df.columns):
        if df.columns[0].startswith("Unnamed"):
            df = df.rename(columns={df.columns[0]: "Date"})
        # Drop any footer rows that are non-numeric in Date
        df = df[pd.to_numeric(df["Date"], errors="coerce").notnull()]
        df["Date"] = df["Date"].astype(str).str.strip()
        df = df[df["Date"].str.len() == 6]  # keep YYYYMM monthly rows
        df["Date"] = pd.to_datetime(df["Date"].astype(int).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
        df = df.rename(
            columns={
                "Mkt-RF": "MKT_RF",
                "Mkt_RF": "MKT_RF",
                "SMB": "SMB",
                "HML": "HML",
                "RMW": "RMW",
                "CMA": "CMA",
                "RF": "RF",
                "Mom   ": "MOM",
                "Mom": "MOM",
                "WML": "MOM",
            }
        )
        # Convert percent values to decimals
        factor_cols = [c for c in df.columns if c != "Date"]
        df[factor_cols] = df[factor_cols].replace(-99.99, np.nan).astype(float) / 100.0
        all_nan = [c for c in factor_cols if df[c].notna().sum() == 0]
        for col in all_nan:
            df[col] = 0.0
    elif "Date" not in df.columns:
        raise ValueError("Factor CSV must include a Date column or Ken French format with YYYYMM first column")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # If MOM missing or zeroed, try to append momentum file if present
    if ("MOM" not in df.columns or df["MOM"].abs().max() == 0.0) and not df.empty:
        if momentum_path is None:
            momentum_path = os.path.join(os.path.dirname(filepath), "F-F_Momentum_Factor.csv")
        if os.path.exists(momentum_path):
            try:
                mom_df = _read_factor_file(momentum_path)
            except Exception:
                mom_df = None
            mom_cols = set(c.strip() for c in mom_df.columns) if mom_df is not None else set()
            if mom_df is None or (not mom_cols.intersection({"Mom", "Mom   ", "WML"}) and not any(c.startswith("Mom") for c in mom_cols)):
                with open(momentum_path, "r", encoding="latin1") as f:
                    lines = f.readlines()
                header_idx = None
                for i, line in enumerate(lines):
                    if "Mom" in line and "," in line:
                        header_idx = i
                        break
                if header_idx is not None:
                    mom_df = pd.read_csv(momentum_path, skiprows=header_idx)
                else:
                    mom_df = pd.read_csv(momentum_path, encoding="latin1")
            if "Date" not in mom_df.columns and (any(col.startswith("Mom") for col in mom_df.columns) or "WML" in mom_df.columns):
                if mom_df.columns[0].startswith("Unnamed"):
                    mom_df = mom_df.rename(columns={mom_df.columns[0]: "Date"})
                mom_df = mom_df[pd.to_numeric(mom_df["Date"], errors="coerce").notnull()]
                mom_df["Date"] = mom_df["Date"].astype(str).str.strip()
                mom_df = mom_df[mom_df["Date"].str.len() == 6]
                mom_df["Date"] = pd.to_datetime(mom_df["Date"].astype(int).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
                mom_df = mom_df.rename(columns={"Mom   ": "MOM", "Mom": "MOM", "WML": "MOM"})
                mom_df["MOM"] = mom_df["MOM"].replace(-99.99, np.nan).astype(float) / 100.0
                mom_df = mom_df.set_index("Date").sort_index()
                if frequency == "M":
                    mom_df = mom_df.resample("ME").sum()
            if "MOM" in mom_df.columns:
                df = df.join(mom_df[["MOM"]], how="left")
                df["MOM"] = df["MOM"].fillna(0.0)

    missing = required_cols - set(df.columns)
    for col in missing:
        df[col] = 0.0

    df = df[list(sorted(required_cols, key=lambda x: x))]  # deterministic column order

    if frequency == "M":
        df = df.resample("ME").sum()
    elif frequency != "D":
        raise ValueError(f"Unsupported frequency: {frequency}")

    return df.dropna()


def load_factor_data_by_region(base_dir: str, frequency: str = "M") -> Dict[str, pd.DataFrame]:
    """
    Load region-specific factor datasets (US, Developed, Developed ex-US, Emerging) if present.
    Sources are intended to mirror the Kenneth R. French Data Library.
    Returns a dict keyed by region slug usable with apply_factor_estimation_to_all_funds.
    """
    factor_files = {
        "us": ("fama_french_5_factors.csv", "F-F_Momentum_Factor.csv"),
        "developed": ("Developed_5_Factors.csv", None),
        "developed_ex_us": ("Developed_ex_US_5_Factors.csv", "Developed_ex_US_MOM_Factor.csv"),
        "emerging": ("Emerging_5_Factors.csv", "Emerging_MOM_Factor.csv"),
    }
    loaded: Dict[str, pd.DataFrame] = {}
    for region, (filename, mom_filename) in factor_files.items():
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            logger.warning("Missing factor file for %s at %s", region, path)
            continue
        mom_path = os.path.join(base_dir, mom_filename) if mom_filename else None
        loaded[region] = load_factor_data_default(path, frequency=frequency, momentum_path=mom_path)

    if not loaded:
        raise FileNotFoundError(f"No factor datasets found in {base_dir}")
    return loaded
