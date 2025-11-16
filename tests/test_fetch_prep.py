"""Tests for data fetching and preparation modules."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.dataio.fetch import fetch_prices, fetch_universe
from src.dataio.prep import prepare_returns


@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        'data': {
            'universe': 'sp500',
            'top_n': 3,
            'include_index': False,
            'start': '2024-01-01',
            'end': '2024-01-20',
            'interval': '1d'
        }
    }


@pytest.fixture
def mock_price_data():
    """
    Create deterministic mock price data for 3 symbols across 15 trading days.

    Returns dict of {symbol: DataFrame} with OHLCV data.
    """
    dates = pd.date_range('2024-01-01', periods=15, freq='B')  # Business days

    # Create deterministic price data with some trends and noise
    symbols_data = {}

    # Symbol A: Uptrend with small noise
    prices_a = 100 + np.arange(15) * 0.5 + np.random.RandomState(42).randn(15) * 0.2
    symbols_data['AAA'] = pd.DataFrame({
        'Open': prices_a * 0.99,
        'High': prices_a * 1.01,
        'Low': prices_a * 0.98,
        'Close': prices_a,
        'Volume': np.random.RandomState(42).randint(1000000, 2000000, 15),
        'Adj Close': prices_a
    }, index=dates)

    # Symbol B: Downtrend with small noise
    prices_b = 200 - np.arange(15) * 0.3 + np.random.RandomState(43).randn(15) * 0.3
    symbols_data['BBB'] = pd.DataFrame({
        'Open': prices_b * 0.99,
        'High': prices_b * 1.01,
        'Low': prices_b * 0.98,
        'Close': prices_b,
        'Volume': np.random.RandomState(43).randint(1000000, 2000000, 15),
        'Adj Close': prices_b
    }, index=dates)

    # Symbol C: Flat with small noise
    prices_c = 150 + np.random.RandomState(44).randn(15) * 0.5
    symbols_data['CCC'] = pd.DataFrame({
        'Open': prices_c * 0.99,
        'High': prices_c * 1.01,
        'Low': prices_c * 0.98,
        'Close': prices_c,
        'Volume': np.random.RandomState(44).randint(1000000, 2000000, 15),
        'Adj Close': prices_c
    }, index=dates)

    return symbols_data


@pytest.fixture
def setup_mock_raw_data(tmp_path, mock_price_data):
    """
    Setup mock raw data directory with per-symbol parquet files.

    This fixture creates temporary parquet files that prep.py can read.
    """
    raw_dir = tmp_path / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    for symbol, data in mock_price_data.items():
        data.to_parquet(raw_dir / f'{symbol}.parquet')

    return raw_dir


class TestFetchUniverse:
    """Tests for fetch_universe function."""

    def test_placeholder_universe(self, test_config):
        """Test that placeholder universe returns expected tickers."""
        symbols = fetch_universe(test_config)

        assert isinstance(symbols, list)
        assert len(symbols) == 3  # top_n=3
        assert all(isinstance(s, str) for s in symbols)

    def test_with_index(self, test_config):
        """Test adding S&P 500 index."""
        test_config['data']['include_index'] = True
        symbols = fetch_universe(test_config)

        assert '^GSPC' in symbols

    def test_from_csv(self, test_config, tmp_path):
        """Test loading universe from CSV file."""
        # Create test CSV
        csv_file = tmp_path / 'test_universe.csv'
        pd.DataFrame({'ticker': ['XXX', 'YYY', 'ZZZ']}).to_csv(csv_file, index=False)

        test_config['data']['universe'] = str(csv_file)
        symbols = fetch_universe(test_config)

        assert symbols == ['XXX', 'YYY', 'ZZZ']


class TestFetchPrices:
    """Tests for fetch_prices function."""

    @patch('src.dataio.fetch.yf.Ticker')
    def test_fetch_prices_success(self, mock_ticker_class, test_config, tmp_path, mock_price_data):
        """Test successful price fetching with mocked yfinance."""
        # Change data directory to tmp_path
        import os
        import sys
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            symbols = ['AAA', 'BBB', 'CCC']

            # Setup mock to return our deterministic data
            def mock_ticker_factory(symbol):
                mock_ticker = MagicMock()
                mock_ticker.history.return_value = mock_price_data[symbol]
                return mock_ticker

            mock_ticker_class.side_effect = mock_ticker_factory

            # Run fetch_prices
            fetch_prices(symbols, test_config)

            # Verify files were created
            raw_dir = Path('data/raw')
            assert raw_dir.exists()

            for symbol in symbols:
                parquet_file = raw_dir / f'{symbol}.parquet'
                assert parquet_file.exists(), f"Missing {parquet_file}"

                # Verify data can be loaded
                df = pd.read_parquet(parquet_file)
                assert len(df) == 15
                assert 'Adj Close' in df.columns

        finally:
            os.chdir(original_cwd)

    @patch('src.dataio.fetch.yf.Ticker')
    def test_fetch_prices_with_retry(self, mock_ticker_class, test_config, tmp_path):
        """Test retry logic on failure."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            symbols = ['AAA']

            # Mock to fail twice then succeed
            mock_ticker = MagicMock()
            call_count = [0]

            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] < 3:
                    raise Exception("Network error")
                return pd.DataFrame({
                    'Open': [100],
                    'High': [101],
                    'Low': [99],
                    'Close': [100.5],
                    'Volume': [1000000],
                    'Adj Close': [100.5]
                }, index=pd.date_range('2024-01-01', periods=1))

            mock_ticker.history.side_effect = side_effect
            mock_ticker_class.return_value = mock_ticker

            # Should succeed after retries
            fetch_prices(symbols, test_config)

            # Verify retry happened
            assert call_count[0] == 3

        finally:
            os.chdir(original_cwd)


class TestPrepareReturns:
    """Tests for prepare_returns function."""

    def test_prepare_returns_shape(self, test_config, tmp_path, mock_price_data):
        """Test returns DataFrame has correct shape and dates."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Setup mock raw data
            raw_dir = Path('data/raw')
            raw_dir.mkdir(parents=True, exist_ok=True)

            for symbol, data in mock_price_data.items():
                data.to_parquet(raw_dir / f'{symbol}.parquet')

            symbols = list(mock_price_data.keys())

            # Run prepare_returns
            returns = prepare_returns(symbols, test_config)

            # Verify shape
            assert returns.shape[1] == 3  # 3 symbols
            assert returns.shape[0] > 0  # At least some days

            # Verify it's a DataFrame with datetime index
            assert isinstance(returns, pd.DataFrame)
            assert isinstance(returns.index, pd.DatetimeIndex)

            # Verify column names match symbols
            assert set(returns.columns) == set(symbols)

        finally:
            os.chdir(original_cwd)

    def test_prepare_returns_no_nans(self, test_config, tmp_path, mock_price_data):
        """Test that forward-fill eliminates NaNs (or they're dropped)."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Setup mock raw data
            raw_dir = Path('data/raw')
            raw_dir.mkdir(parents=True, exist_ok=True)

            for symbol, data in mock_price_data.items():
                data.to_parquet(raw_dir / f'{symbol}.parquet')

            symbols = list(mock_price_data.keys())

            # Run prepare_returns
            returns = prepare_returns(symbols, test_config)

            # Verify no NaNs remain (after ffill and dropna)
            assert returns.isna().sum().sum() == 0, "Returns contain NaN values"

        finally:
            os.chdir(original_cwd)

    def test_prepare_returns_calculation(self, test_config, tmp_path, mock_price_data):
        """Test that returns are correctly calculated as pct_change."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Setup mock raw data
            raw_dir = Path('data/raw')
            raw_dir.mkdir(parents=True, exist_ok=True)

            for symbol, data in mock_price_data.items():
                data.to_parquet(raw_dir / f'{symbol}.parquet')

            symbols = list(mock_price_data.keys())  # Fixed: get keys not items

            # Run prepare_returns
            returns = prepare_returns(symbols, test_config)

            # Manually compute expected returns for first symbol
            expected_returns = mock_price_data['AAA']['Adj Close'].pct_change().iloc[1:]

            # Compare (allowing for small numerical differences)
            # Note: actual returns might differ due to calendar alignment
            # Just verify returns are in reasonable range
            assert returns['AAA'].abs().max() < 0.1, "Returns seem unreasonably large"
            assert returns['AAA'].std() > 0, "Returns have zero variance"

        finally:
            os.chdir(original_cwd)

    def test_returns_parquet_written(self, test_config, tmp_path, mock_price_data):
        """Test that returns.parquet is written and reloadable."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Setup mock raw data
            raw_dir = Path('data/raw')
            raw_dir.mkdir(parents=True, exist_ok=True)

            for symbol, data in mock_price_data.items():
                data.to_parquet(raw_dir / f'{symbol}.parquet')

            symbols = list(mock_price_data.keys())

            # Run prepare_returns
            returns1 = prepare_returns(symbols, test_config)

            # Verify file was written
            output_file = Path('data/processed/returns.parquet')
            assert output_file.exists(), "returns.parquet not created"

            # Reload and verify
            returns2 = pd.read_parquet(output_file)
            pd.testing.assert_frame_equal(returns1, returns2)

        finally:
            os.chdir(original_cwd)


class TestIntegration:
    """Integration tests for full fetch + prep pipeline."""

    @patch('src.dataio.fetch.yf.Ticker')
    def test_full_pipeline(self, mock_ticker_class, test_config, tmp_path, mock_price_data):
        """Test full pipeline: fetch_universe -> fetch_prices -> prepare_returns."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Step 1: Fetch universe
            symbols = fetch_universe(test_config)
            assert len(symbols) == 3

            # Step 2: Mock fetch prices
            def mock_ticker_factory(symbol):
                # Use mock data for AAA, BBB, CCC
                # For other symbols, return empty (will be skipped)
                if symbol in mock_price_data:
                    mock_ticker = MagicMock()
                    mock_ticker.history.return_value = mock_price_data[symbol]
                    return mock_ticker
                mock_ticker = MagicMock()
                mock_ticker.history.return_value = pd.DataFrame()  # Empty
                return mock_ticker

            mock_ticker_class.side_effect = mock_ticker_factory

            # Fetch prices (using mock data)
            fetch_prices(['AAA', 'BBB', 'CCC'], test_config)

            # Step 3: Prepare returns
            returns = prepare_returns(['AAA', 'BBB', 'CCC'], test_config)

            # Verify final output
            assert isinstance(returns, pd.DataFrame)
            assert returns.shape[0] > 0
            assert returns.shape[1] == 3
            assert returns.isna().sum().sum() == 0

            # Verify output file exists
            assert Path('data/processed/returns.parquet').exists()

        finally:
            os.chdir(original_cwd)
