from pathlib import Path

import pandas as pd

from src.data_loaders import (
    fetch_price_history_yf,
    load_all_fund_returns_csv,
    load_factor_data_by_region,
    load_factor_data_default,
    load_price_history_csv,
)
from src.models import Fund


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_price_history_csv_monthly_returns():
    path = REPO_ROOT / "data" / "funds" / "VTI.csv"
    series = load_price_history_csv(str(path), frequency="M")
    assert not series.empty
    # First monthly return based on sample data: (225.50 / 220) - 1
    assert round(series.iloc[0], 5) == round((225.50 / 220.0) - 1, 5)


def test_load_all_fund_returns_csv_with_missing(caplog):
    funds = [
        Fund(
            ticker="VTI",
            name="Vanguard Total Stock Market",
            manager="Vanguard",
            asset_class="US_Equity",
            region="US",
            vehicle_type="ETF",
            expense_ratio=0.0003,
        ),
        Fund(
            ticker="MISSING",
            name="Missing Fund",
            manager="Vanguard",
            asset_class="US_Equity",
            region="US",
            vehicle_type="ETF",
            expense_ratio=0.0003,
        ),
    ]
    base_dir = REPO_ROOT / "data" / "funds"
    with caplog.at_level("WARNING"):
        returns = load_all_fund_returns_csv(funds, base_dir=str(base_dir), frequency="M")
    assert "VTI" in returns
    assert "MISSING" not in returns
    assert any("Missing CSV for MISSING" in record.message for record in caplog.records)


def test_load_factor_data_default_monthly():
    path = REPO_ROOT / "data" / "factors" / "fama_french_5_factors.csv"
    df = load_factor_data_default(str(path), frequency="M")
    expected_cols = {"MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"}
    assert expected_cols.issubset(df.columns)
    assert len(df) > 0
    assert df.index.freqstr in {"M", "ME"}


def test_load_factor_data_default_ken_french():
    path = REPO_ROOT / "data" / "factors" / "F-F_Research_Data_5_Factors_2x3 2.csv"
    df = load_factor_data_default(str(path), frequency="M")
    expected_cols = {"MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"}
    assert expected_cols.issubset(df.columns)
    assert len(df) > 0
    # Values should be decimals (percent / 100)
    assert df["MKT_RF"].abs().max() < 1


def test_load_factor_data_with_momentum_file():
    path = REPO_ROOT / "data" / "factors" / "F-F_Research_Data_5_Factors_2x3 2.csv"
    mom_path = REPO_ROOT / "data" / "factors" / "F-F_Momentum_Factor.csv"
    df = load_factor_data_default(str(path), frequency="M", momentum_path=str(mom_path))
    assert "MOM" in df.columns
    assert df["MOM"].abs().max() > 0.01


def test_load_factor_data_default_global_developed_ex_us():
    path = REPO_ROOT / "data" / "factors" / "Developed_ex_US_5_Factors.csv"
    mom_path = REPO_ROOT / "data" / "factors" / "Developed_ex_US_MOM_Factor.csv"
    df = load_factor_data_default(str(path), frequency="M", momentum_path=str(mom_path))
    expected_cols = {"MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"}
    assert expected_cols.issubset(df.columns)
    assert len(df) > 0
    assert df["MOM"].abs().max() > 0.001


def test_load_factor_data_by_region_loads_all_available(caplog):
    base_dir = REPO_ROOT / "data" / "factors"
    with caplog.at_level("WARNING"):
        data = load_factor_data_by_region(str(base_dir), frequency="M")
    # All files are present in the repo; ensure we loaded multiple regional sets.
    assert {"us", "developed", "developed_ex_us", "emerging"}.issubset(set(data.keys()))
    assert all(not df.empty for df in data.values())


def test_fetch_price_history_yf_monkeypatched(monkeypatch):
    def fake_download(ticker, start, end=None, progress=False, auto_adjust=False):
        idx = pd.date_range("2024-01-31", periods=2, freq="ME")
        return pd.DataFrame({"Adj Close": [100.0, 102.0]}, index=idx)

    monkeypatch.setattr("src.data_loaders.yf.download", fake_download)
    series = fetch_price_history_yf("TEST", start="2024-01-01", frequency="M")
    assert not series.empty
    assert round(series.iloc[0], 4) == round((102.0 / 100.0) - 1, 4)
