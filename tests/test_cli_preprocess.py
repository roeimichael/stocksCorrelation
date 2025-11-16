"""Tests for CLI preprocessing script."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.preprocess import check_existing_data, run_preprocessing


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return {
        'data': {
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'universe': 'sp500',
            'top_n': 10,
            'include_index': False,
            'start': '2024-01-01',
            'end': '2024-01-31',
            'interval': '1d'
        },
        'windows': {
            'length': 10,
            'normalization': 'zscore',
            'min_history_days': 20
        },
        'light_mode': {
            'enabled': False
        }
    }


@pytest.fixture
def mock_prices():
    """Create mock prices DataFrame."""
    dates = pd.date_range('2024-01-01', periods=30, freq='B')
    return pd.DataFrame({
        'STOCK_A': np.random.randn(30) * 10 + 100,
        'STOCK_B': np.random.randn(30) * 10 + 200
    }, index=dates)


@pytest.fixture
def mock_returns():
    """Create mock returns DataFrame."""
    dates = pd.date_range('2024-01-01', periods=30, freq='B')
    return pd.DataFrame({
        'STOCK_A': np.random.randn(30) * 0.02,
        'STOCK_B': np.random.randn(30) * 0.02
    }, index=dates)


class TestCheckExistingData:
    """Tests for check_existing_data function."""

    def test_check_existing_data_no_files(self, tmp_path, monkeypatch):
        """Test when no files exist."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create data/processed directory
        (tmp_path / 'data' / 'processed').mkdir(parents=True)

        status = check_existing_data()

        assert status['prices_exists'] is False
        assert status['prices_last_date'] is None
        assert status['returns_exists'] is False
        assert status['returns_last_date'] is None

    def test_check_existing_data_with_files(self, tmp_path, monkeypatch, mock_prices, mock_returns):
        """Test when files exist."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create data/processed directory
        processed_dir = tmp_path / 'data' / 'processed'
        processed_dir.mkdir(parents=True)

        # Save mock data
        mock_prices.to_parquet(processed_dir / 'prices_clean.parquet')
        mock_returns.to_parquet(processed_dir / 'returns.parquet')

        status = check_existing_data()

        assert status['prices_exists'] is True
        assert status['prices_last_date'] == mock_prices.index.max()
        assert status['returns_exists'] is True
        assert status['returns_last_date'] == mock_returns.index.max()


class TestRunPreprocessing:
    """Tests for run_preprocessing function."""

    @patch('scripts.preprocess.fetch_universe')
    @patch('scripts.preprocess.fetch_prices')
    @patch('scripts.preprocess.prepare_returns')
    @patch('scripts.preprocess.build_windows')
    def test_run_preprocessing_creates_files(
        self,
        mock_build_windows,
        mock_prepare_returns,
        mock_fetch_prices,
        mock_fetch_universe,
        tmp_path,
        monkeypatch,
        mock_config,
        mock_prices,
        mock_returns
    ):
        """Test that preprocessing creates expected files."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create data/processed directory
        processed_dir = tmp_path / 'data' / 'processed'
        processed_dir.mkdir(parents=True)

        # Mock returns
        mock_fetch_universe.return_value = ['STOCK_A', 'STOCK_B']
        # fetch_prices doesn't return anything, it writes to files
        mock_fetch_prices.return_value = None
        mock_prepare_returns.return_value = mock_returns

        # Mock windows DataFrame
        mock_windows_df = pd.DataFrame({
            'symbol': ['STOCK_A'] * 10 + ['STOCK_B'] * 10,
            'start_date': [pd.Timestamp('2024-01-01')] * 20,
            'end_date': [pd.Timestamp('2024-01-10')] * 20,
            'features': [[0.1] * 10] * 20,
            'label': [1, 0] * 10
        })
        mock_build_windows.return_value = mock_windows_df

        # Run preprocessing
        run_preprocessing(mock_config, force_full=True)

        # Verify functions were called
        mock_fetch_universe.assert_called_once()
        mock_fetch_prices.assert_called_once()
        mock_prepare_returns.assert_called_once()
        mock_build_windows.assert_called_once()

    @patch('scripts.preprocess.fetch_universe')
    @patch('scripts.preprocess.fetch_prices')
    @patch('scripts.preprocess.prepare_returns')
    @patch('scripts.preprocess.build_windows')
    def test_run_preprocessing_light_mode(
        self,
        mock_build_windows,
        mock_prepare_returns,
        mock_fetch_prices,
        mock_fetch_universe,
        tmp_path,
        monkeypatch,
        mock_config,
        mock_prices,
        mock_returns
    ):
        """Test preprocessing with light mode enabled."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create data/processed directory
        (tmp_path / 'data' / 'processed').mkdir(parents=True)

        # Enable light mode
        mock_config['light_mode'] = {'enabled': True, 'top_n_stocks': 1}

        # Mock returns
        all_tickers = ['STOCK_A', 'STOCK_B', 'STOCK_C']
        mock_fetch_universe.return_value = all_tickers
        mock_fetch_prices.return_value = None
        mock_prepare_returns.return_value = mock_returns

        # Mock windows
        mock_windows_df = pd.DataFrame({
            'symbol': ['STOCK_A'] * 10,
            'start_date': [pd.Timestamp('2024-01-01')] * 10,
            'end_date': [pd.Timestamp('2024-01-10')] * 10,
            'features': [[0.1] * 10] * 10,
            'label': [1, 0] * 5
        })
        mock_build_windows.return_value = mock_windows_df

        # Run preprocessing
        run_preprocessing(mock_config, force_full=True)

        # Verify only top 1 stock was used
        call_args = mock_fetch_prices.call_args
        tickers_used = call_args[0][0]
        assert len(tickers_used) == 1
        assert tickers_used[0] == 'STOCK_A'

    @patch('scripts.preprocess.fetch_universe')
    @patch('scripts.preprocess.fetch_prices')
    @patch('scripts.preprocess.prepare_returns')
    @patch('scripts.preprocess.build_windows')
    def test_run_preprocessing_incremental_update(
        self,
        mock_build_windows,
        mock_prepare_returns,
        mock_fetch_prices,
        mock_fetch_universe,
        tmp_path,
        monkeypatch,
        mock_config,
        mock_prices,
        mock_returns
    ):
        """Test incremental update when existing data exists."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create data/processed directory
        processed_dir = tmp_path / 'data' / 'processed'
        processed_dir.mkdir(parents=True)

        # Save existing data
        existing_prices = mock_prices.iloc[:20]  # Only first 20 days
        existing_prices.to_parquet(processed_dir / 'prices_clean.parquet')

        # Mock returns
        mock_fetch_universe.return_value = ['STOCK_A', 'STOCK_B']
        mock_fetch_prices.return_value = None
        mock_prepare_returns.return_value = mock_returns

        # Mock windows
        mock_windows_df = pd.DataFrame({
            'symbol': ['STOCK_A'] * 10,
            'start_date': [pd.Timestamp('2024-01-01')] * 10,
            'end_date': [pd.Timestamp('2024-01-10')] * 10,
            'features': [[0.1] * 10] * 10,
            'label': [1, 0] * 5
        })
        mock_build_windows.return_value = mock_windows_df

        # Run preprocessing (not force_full)
        run_preprocessing(mock_config, force_full=False)

        # Verify incremental fetch was called
        mock_fetch_prices.assert_called_once()

        # Verify prepare_returns and build_windows were called
        mock_prepare_returns.assert_called_once()
        mock_build_windows.assert_called_once()


class TestIntegration:
    """Integration tests for preprocessing."""

    @patch('scripts.preprocess.fetch_universe')
    @patch('scripts.preprocess.fetch_prices')
    @patch('scripts.preprocess.prepare_returns')
    @patch('scripts.preprocess.build_windows')
    def test_full_pipeline_smoke_test(
        self,
        mock_build_windows,
        mock_prepare_returns,
        mock_fetch_prices,
        mock_fetch_universe,
        tmp_path,
        monkeypatch,
        mock_config,
        mock_prices,
        mock_returns
    ):
        """Smoke test: full pipeline runs without errors."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create data/processed directory
        processed_dir = tmp_path / 'data' / 'processed'
        processed_dir.mkdir(parents=True)

        # Mock universe and prices
        mock_fetch_universe.return_value = ['STOCK_A', 'STOCK_B']
        mock_fetch_prices.return_value = None
        mock_prepare_returns.return_value = mock_returns

        # Mock windows
        mock_windows_df = pd.DataFrame({
            'symbol': ['STOCK_A'] * 10,
            'start_date': [pd.Timestamp('2024-01-01')] * 10,
            'end_date': [pd.Timestamp('2024-01-10')] * 10,
            'features': [[0.1] * 10] * 10,
            'label': [1, 0] * 5
        })
        mock_build_windows.return_value = mock_windows_df

        # Run preprocessing - should not raise
        run_preprocessing(mock_config, force_full=True)

        # Verify all functions were called
        mock_fetch_universe.assert_called_once()
        mock_fetch_prices.assert_called_once()
        mock_prepare_returns.assert_called_once()
        mock_build_windows.assert_called_once()
