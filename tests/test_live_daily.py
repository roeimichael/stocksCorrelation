"""Tests for live daily trading pipeline."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.live_daily import load_positions_state, run_live_daily, save_positions_state
from src.dataio.live_append import append_new_bars


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
        'similarity': {
            'metric': 'pearson',
            'top_k': 25,
            'min_sim': 0.0
        },
        'vote': {
            'scheme': 'similarity_weighted',
            'threshold': 0.70,
            'abstain_if_below_k': 5
        },
        'backtest': {
            'max_positions': 5,
            'costs_bps': 5.0,
            'slippage_bps': 2.0
        }
    }


@pytest.fixture
def sample_price_data():
    """Create sample price data for ingestion."""
    dates = pd.date_range('2024-01-25', periods=5, freq='B')
    df = pd.DataFrame({
        'Open': [100, 101, 102, 103, 104],
        'High': [105, 106, 107, 108, 109],
        'Low': [99, 100, 101, 102, 103],
        'Close': [102, 103, 104, 105, 106],
        'Volume': [1000000, 1100000, 1200000, 1300000, 1400000],
        'Adj Close': [102, 103, 104, 105, 106]
    }, index=dates)
    df.index.name = 'Date'
    return df


@pytest.fixture
def mock_returns():
    """Create mock returns DataFrame."""
    dates = pd.date_range('2024-01-01', periods=30, freq='B')
    return pd.DataFrame({
        'AAPL': np.random.randn(30) * 0.02,
        'MSFT': np.random.randn(30) * 0.02
    }, index=dates)


@pytest.fixture
def mock_windows():
    """Create mock windows DataFrame."""
    dates = pd.date_range('2024-01-01', periods=20, freq='B')
    windows = []

    for i in range(10):
        windows.append({
            'symbol': 'AAPL',
            'start_date': dates[i],
            'end_date': dates[i + 9],
            'features': np.random.randn(10),
            'label': 1 if i % 2 == 0 else 0
        })

    for i in range(10):
        windows.append({
            'symbol': 'MSFT',
            'start_date': dates[i],
            'end_date': dates[i + 9],
            'features': np.random.randn(10),
            'label': 1 if i % 2 == 0 else 0
        })

    return pd.DataFrame(windows)


class TestAppendNewBars:
    """Tests for append_new_bars function."""

    def test_append_new_bars_csv(self, tmp_path, sample_price_data):
        """Test appending new bars from CSV file."""
        # Setup directories
        live_dir = tmp_path / 'live_ingest'
        raw_dir = tmp_path / 'raw'
        live_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)

        # Write sample CSV
        csv_file = live_dir / 'AAPL.csv'
        sample_price_data.to_csv(csv_file)

        # Run append
        updated = append_new_bars(str(live_dir), str(raw_dir))

        # Verify
        assert 'AAPL' in updated
        assert (raw_dir / 'AAPL.parquet').exists()

        # Verify file was archived
        assert (live_dir / 'processed' / 'AAPL.csv').exists()
        assert not csv_file.exists()

    def test_append_new_bars_parquet(self, tmp_path, sample_price_data):
        """Test appending new bars from Parquet file."""
        # Setup directories
        live_dir = tmp_path / 'live_ingest'
        raw_dir = tmp_path / 'raw'
        live_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)

        # Write sample Parquet
        parquet_file = live_dir / 'MSFT.parquet'
        sample_price_data.to_parquet(parquet_file)

        # Run append
        updated = append_new_bars(str(live_dir), str(raw_dir))

        # Verify
        assert 'MSFT' in updated
        assert (raw_dir / 'MSFT.parquet').exists()

    def test_append_new_bars_merge(self, tmp_path, sample_price_data):
        """Test merging new bars with existing data."""
        # Setup directories
        live_dir = tmp_path / 'live_ingest'
        raw_dir = tmp_path / 'raw'
        live_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)

        # Create existing data
        existing_dates = pd.date_range('2024-01-01', periods=20, freq='B')
        existing_data = pd.DataFrame({
            'Open': np.random.randn(20) * 10 + 100,
            'High': np.random.randn(20) * 10 + 105,
            'Low': np.random.randn(20) * 10 + 95,
            'Close': np.random.randn(20) * 10 + 100,
            'Volume': [1000000] * 20,
            'Adj Close': np.random.randn(20) * 10 + 100
        }, index=existing_dates)

        existing_file = raw_dir / 'AAPL.parquet'
        existing_data.to_parquet(existing_file)

        # Write new data
        csv_file = live_dir / 'AAPL.csv'
        sample_price_data.to_csv(csv_file)

        # Run append
        updated = append_new_bars(str(live_dir), str(raw_dir))

        # Verify merge
        assert 'AAPL' in updated

        merged_data = pd.read_parquet(raw_dir / 'AAPL.parquet')
        # Should have at least as many rows as before (deduplication may occur)
        assert len(merged_data) >= len(existing_data)

    def test_append_new_bars_no_files(self, tmp_path):
        """Test when no files to ingest."""
        # Setup empty directories
        live_dir = tmp_path / 'live_ingest'
        raw_dir = tmp_path / 'raw'
        live_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)

        # Run append
        updated = append_new_bars(str(live_dir), str(raw_dir))

        # Verify
        assert len(updated) == 0

    def test_append_new_bars_missing_columns(self, tmp_path):
        """Test handling of files with missing required columns."""
        # Setup directories
        live_dir = tmp_path / 'live_ingest'
        raw_dir = tmp_path / 'raw'
        live_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)

        # Write invalid CSV (missing Volume column)
        dates = pd.date_range('2024-01-25', periods=5, freq='B')
        invalid_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'Close': [102, 103, 104, 105, 106]
        }, index=dates)

        csv_file = live_dir / 'INVALID.csv'
        invalid_data.to_csv(csv_file)

        # Run append
        updated = append_new_bars(str(live_dir), str(raw_dir))

        # Verify - should skip invalid file
        assert 'INVALID' not in updated


class TestPositionsState:
    """Tests for positions state management."""

    def test_load_positions_state_new_file(self, tmp_path):
        """Test loading positions state when file doesn't exist."""
        state_file = tmp_path / 'positions_state.json'

        state = load_positions_state(state_file)

        assert 'open_positions' in state
        assert len(state['open_positions']) == 0

    def test_save_and_load_positions_state(self, tmp_path):
        """Test saving and loading positions state."""
        state_file = tmp_path / 'positions_state.json'

        # Create sample state
        state = {
            'open_positions': [
                {
                    'symbol': 'AAPL',
                    'entry_date': '2024-01-15',
                    'side': 'UP',
                    'p_up': 0.75,
                    'confidence': 0.25,
                    'analogs': [
                        {'symbol': 'MSFT', 'end_date': '2024-01-10', 'sim': 0.85, 'label': 1}
                    ]
                }
            ]
        }

        # Save
        save_positions_state(state, state_file)

        # Verify file exists
        assert state_file.exists()

        # Load
        loaded_state = load_positions_state(state_file)

        # Verify
        assert len(loaded_state['open_positions']) == 1
        assert loaded_state['open_positions'][0]['symbol'] == 'AAPL'
        assert loaded_state['open_positions'][0]['side'] == 'UP'


class TestLiveDailyPipeline:
    """Tests for run_live_daily function."""

    @patch('scripts.live_daily.append_new_bars')
    @patch('src.dataio.fetch.fetch_universe')
    @patch('scripts.live_daily.prepare_returns')
    @patch('scripts.live_daily.build_windows')
    def test_run_live_daily_creates_signals_file(
        self,
        mock_build_windows,
        mock_prepare_returns,
        mock_fetch_universe,
        mock_append_new_bars,
        tmp_path,
        monkeypatch,
        mock_config,
        mock_returns,
        mock_windows
    ):
        """Test that live daily creates signals file."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Mock returns
        mock_append_new_bars.return_value = ['AAPL']
        mock_fetch_universe.return_value = ['AAPL', 'MSFT']
        mock_prepare_returns.return_value = mock_returns
        mock_build_windows.return_value = mock_windows

        # Run live daily
        run_live_daily(mock_config, signal_date='2024-01-30')

        # Verify signals file created
        signals_file = tmp_path / 'results' / 'live' / 'signals_2024-01-30.csv'
        assert signals_file.exists()

        # Verify signals CSV has content
        signals_df = pd.read_csv(signals_file)
        assert len(signals_df) > 0
        assert 'symbol' in signals_df.columns
        assert 'signal' in signals_df.columns
        assert 'confidence' in signals_df.columns

    @patch('scripts.live_daily.append_new_bars')
    @patch('src.dataio.fetch.fetch_universe')
    @patch('scripts.live_daily.prepare_returns')
    @patch('scripts.live_daily.build_windows')
    def test_run_live_daily_creates_positions_state(
        self,
        mock_build_windows,
        mock_prepare_returns,
        mock_fetch_universe,
        mock_append_new_bars,
        tmp_path,
        monkeypatch,
        mock_config,
        mock_returns,
        mock_windows
    ):
        """Test that live daily creates/updates positions_state.json."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Mock returns
        mock_append_new_bars.return_value = ['AAPL']
        mock_fetch_universe.return_value = ['AAPL', 'MSFT']
        mock_prepare_returns.return_value = mock_returns
        mock_build_windows.return_value = mock_windows

        # Run live daily
        run_live_daily(mock_config, signal_date='2024-01-30')

        # Verify positions state file created
        state_file = tmp_path / 'results' / 'live' / 'positions_state.json'
        assert state_file.exists()

        # Load and verify structure
        with open(state_file) as f:
            state = json.load(f)

        assert 'open_positions' in state
        assert isinstance(state['open_positions'], list)

        # If positions exist, verify structure
        if len(state['open_positions']) > 0:
            pos = state['open_positions'][0]
            assert 'symbol' in pos
            assert 'entry_date' in pos
            assert 'side' in pos
            assert 'p_up' in pos
            assert 'confidence' in pos
            assert 'analogs' in pos

    @patch('scripts.live_daily.append_new_bars')
    @patch('src.dataio.fetch.fetch_universe')
    @patch('scripts.live_daily.prepare_returns')
    @patch('scripts.live_daily.build_windows')
    def test_run_live_daily_no_new_bars(
        self,
        mock_build_windows,
        mock_prepare_returns,
        mock_fetch_universe,
        mock_append_new_bars,
        tmp_path,
        monkeypatch,
        mock_config,
        mock_returns,
        mock_windows
    ):
        """Test live daily when no new bars to ingest."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Mock returns - no new bars
        mock_append_new_bars.return_value = []
        mock_fetch_universe.return_value = ['AAPL', 'MSFT']
        mock_prepare_returns.return_value = mock_returns
        mock_build_windows.return_value = mock_windows

        # Run live daily - should not raise
        run_live_daily(mock_config, signal_date='2024-01-30')

        # Verify signals file still created
        signals_file = tmp_path / 'results' / 'live' / 'signals_2024-01-30.csv'
        assert signals_file.exists()

    @patch('scripts.live_daily.append_new_bars')
    @patch('src.dataio.fetch.fetch_universe')
    @patch('scripts.live_daily.prepare_returns')
    @patch('scripts.live_daily.build_windows')
    def test_run_live_daily_uses_last_date(
        self,
        mock_build_windows,
        mock_prepare_returns,
        mock_fetch_universe,
        mock_append_new_bars,
        tmp_path,
        monkeypatch,
        mock_config,
        mock_returns,
        mock_windows
    ):
        """Test live daily uses last available date when date not specified."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Mock returns
        mock_append_new_bars.return_value = []
        mock_fetch_universe.return_value = ['AAPL', 'MSFT']
        mock_prepare_returns.return_value = mock_returns
        mock_build_windows.return_value = mock_windows

        # Run live daily without specifying date
        run_live_daily(mock_config, signal_date=None)

        # Get expected date (last date in mock_returns)
        expected_date = mock_returns.index[-1].strftime('%Y-%m-%d')

        # Verify signals file created with last date
        signals_file = tmp_path / 'results' / 'live' / f'signals_{expected_date}.csv'
        assert signals_file.exists()


class TestIntegration:
    """Integration tests for live daily pipeline."""

    def test_full_pipeline_with_ingestion(self, tmp_path, monkeypatch, mock_config, sample_price_data):
        """Integration test: ingest data, generate signals, update state."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Setup directories
        live_dir = tmp_path / 'data' / 'live_ingest'
        raw_dir = tmp_path / 'data' / 'raw'
        live_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)

        # Create existing raw data with proper datetime index
        dates = pd.date_range('2024-01-01', periods=25, freq='B')
        for symbol in ['AAPL', 'MSFT']:
            # Ensure positive prices
            base_price = 100
            existing_data = pd.DataFrame({
                'Open': base_price + np.random.randn(25),
                'High': base_price + 5 + np.random.randn(25),
                'Low': base_price - 5 + np.random.randn(25),
                'Close': base_price + np.random.randn(25),
                'Volume': [1000000] * 25,
                'Adj Close': base_price + np.random.randn(25)
            }, index=dates)
            # Ensure index name is 'Date' for consistency
            existing_data.index.name = 'Date'
            existing_data.to_parquet(raw_dir / f'{symbol}.parquet')

        # Add new data to ingest
        new_file = live_dir / 'AAPL.csv'
        sample_price_data.to_csv(new_file)

        # Mock fetch_universe to return our test symbols
        with patch('src.dataio.fetch.fetch_universe', return_value=['AAPL', 'MSFT']):
            # Run live daily
            run_live_daily(mock_config, signal_date='2024-01-30')

        # Verify all outputs created
        assert (tmp_path / 'results' / 'live' / 'signals_2024-01-30.csv').exists()
        assert (tmp_path / 'results' / 'live' / 'positions_state.json').exists()

        # Verify new data was ingested
        assert not new_file.exists()  # Should be archived
        assert (live_dir / 'processed' / 'AAPL.csv').exists()

        # Verify merged data (new bars should have been added to raw file)
        merged_data = pd.read_parquet(raw_dir / 'AAPL.parquet')
        # The merged file should contain data (may overlap with existing dates)
        assert len(merged_data) >= 25  # At least the original bars
