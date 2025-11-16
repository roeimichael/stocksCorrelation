"""Tests for position monitoring and drift detection."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.monitor import monitor_positions
from src.trading.monitor import (
    classify_alert,
    correlation_decay,
    directional_concordance,
    pattern_deviation_z,
    similarity_retention,
)


@pytest.fixture
def base_config():
    """Create base configuration with monitoring thresholds."""
    return {
        'windows': {
            'length': 10,
            'normalization': 'zscore'
        },
        'similarity': {
            'metric': 'pearson'
        },
        'monitor': {
            'corr_window_days': 20,
            'lookback_days': 3,
            'deviation_window_days': 30,
            'thresholds': {
                # Red thresholds (most severe)
                'sr_floor_red': 0.3,
                'dc_floor_red': 0.4,
                'cd_drop_alert_red': -0.3,
                'pds_z_alert_red': 3.0,
                # Yellow thresholds (warning)
                'sr_floor_yellow': 0.5,
                'dc_floor_yellow': 0.5,
                'cd_drop_alert_yellow': -0.2,
                'pds_z_alert_yellow': 2.0
            }
        }
    }


@pytest.fixture
def sample_position():
    """Create sample position dict."""
    return {
        'symbol': 'AAPL',
        'entry_date': '2024-01-15',
        'side': 'UP',
        'p_up': 0.75,
        'confidence': 0.25,
        'analogs': [
            {'symbol': 'MSFT', 'end_date': '2024-01-10', 'sim': 0.8, 'label': 1},
            {'symbol': 'GOOGL', 'end_date': '2024-01-12', 'sim': 0.7, 'label': 1},
            {'symbol': 'AMZN', 'end_date': '2024-01-11', 'sim': 0.6, 'label': 1}
        ]
    }


@pytest.fixture
def sample_returns():
    """Create sample returns DataFrame."""
    dates = pd.date_range('2024-01-01', periods=40, freq='B')

    # Create correlated returns for AAPL and analogs
    base_returns = np.random.randn(40) * 0.02

    return pd.DataFrame({
        'AAPL': base_returns + np.random.randn(40) * 0.005,
        'MSFT': base_returns + np.random.randn(40) * 0.005,
        'GOOGL': base_returns + np.random.randn(40) * 0.005,
        'AMZN': base_returns + np.random.randn(40) * 0.005
    }, index=dates)


@pytest.fixture
def sample_windows():
    """Create sample windows DataFrame."""
    windows = []

    dates = pd.date_range('2024-01-01', periods=30, freq='B')

    for symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
        for i in range(10):
            end_date = dates[i + 9]
            windows.append({
                'symbol': symbol,
                'start_date': dates[i],
                'end_date': end_date,
                'features': np.random.randn(10),
                'label': 1 if i % 2 == 0 else 0
            })

    return pd.DataFrame(windows)


class TestClassifyAlert:
    """Tests for classify_alert function."""

    def test_classify_alert_green(self, base_config):
        """Test GREEN classification when all metrics are healthy."""
        # Good metrics: high similarity, high concordance, stable correlation, low deviation
        sr = 0.8
        dc = 0.9
        dcd = 0.1  # Positive correlation change
        pdz = 0.5  # Low deviation

        alert = classify_alert(sr, dc, dcd, pdz, base_config)

        assert alert == "GREEN"

    def test_classify_alert_yellow_sr(self, base_config):
        """Test YELLOW classification due to low similarity retention."""
        # Similarity retention below yellow threshold but above red
        sr = 0.45  # Below sr_floor_yellow (0.5) but above sr_floor_red (0.3)
        dc = 0.9
        dcd = 0.1
        pdz = 0.5

        alert = classify_alert(sr, dc, dcd, pdz, base_config)

        assert alert == "YELLOW"

    def test_classify_alert_yellow_dc(self, base_config):
        """Test YELLOW classification due to low directional concordance."""
        sr = 0.8
        dc = 0.45  # Below dc_floor_yellow (0.5)
        dcd = 0.1
        pdz = 0.5

        alert = classify_alert(sr, dc, dcd, pdz, base_config)

        assert alert == "YELLOW"

    def test_classify_alert_yellow_correlation_drop(self, base_config):
        """Test YELLOW classification due to correlation drop."""
        sr = 0.8
        dc = 0.9
        dcd = -0.25  # Below cd_drop_alert_yellow (-0.2)
        pdz = 0.5

        alert = classify_alert(sr, dc, dcd, pdz, base_config)

        assert alert == "YELLOW"

    def test_classify_alert_yellow_pattern_deviation(self, base_config):
        """Test YELLOW classification due to high pattern deviation."""
        sr = 0.8
        dc = 0.9
        dcd = 0.1
        pdz = 2.5  # Above pds_z_alert_yellow (2.0)

        alert = classify_alert(sr, dc, dcd, pdz, base_config)

        assert alert == "YELLOW"

    def test_classify_alert_red_sr(self, base_config):
        """Test RED classification due to very low similarity retention."""
        sr = 0.2  # Below sr_floor_red (0.3)
        dc = 0.9
        dcd = 0.1
        pdz = 0.5

        alert = classify_alert(sr, dc, dcd, pdz, base_config)

        assert alert == "RED"

    def test_classify_alert_red_dc(self, base_config):
        """Test RED classification due to very low directional concordance."""
        sr = 0.8
        dc = 0.3  # Below dc_floor_red (0.4)
        dcd = 0.1
        pdz = 0.5

        alert = classify_alert(sr, dc, dcd, pdz, base_config)

        assert alert == "RED"

    def test_classify_alert_red_correlation_drop(self, base_config):
        """Test RED classification due to severe correlation drop."""
        sr = 0.8
        dc = 0.9
        dcd = -0.4  # Below cd_drop_alert_red (-0.3)
        pdz = 0.5

        alert = classify_alert(sr, dc, dcd, pdz, base_config)

        assert alert == "RED"

    def test_classify_alert_red_pattern_deviation(self, base_config):
        """Test RED classification due to severe pattern deviation."""
        sr = 0.8
        dc = 0.9
        dcd = 0.1
        pdz = 3.5  # Above pds_z_alert_red (3.0)

        alert = classify_alert(sr, dc, dcd, pdz, base_config)

        assert alert == "RED"

    def test_classify_alert_red_multiple_issues(self, base_config):
        """Test RED classification with multiple issues."""
        # Multiple metrics failing red thresholds
        sr = 0.2
        dc = 0.3
        dcd = -0.4
        pdz = 4.0

        alert = classify_alert(sr, dc, dcd, pdz, base_config)

        assert alert == "RED"

    def test_classify_alert_threshold_boundaries(self, base_config):
        """Test exact threshold boundaries."""
        # Exactly at yellow threshold (should be YELLOW)
        sr = 0.5
        dc = 0.9
        dcd = 0.1
        pdz = 0.5

        alert = classify_alert(sr, dc, dcd, pdz, base_config)
        assert alert == "GREEN"  # Not below, so GREEN

        # Just below yellow threshold
        sr = 0.499
        alert = classify_alert(sr, dc, dcd, pdz, base_config)
        assert alert == "YELLOW"

        # Exactly at red threshold
        sr = 0.3
        alert = classify_alert(sr, dc, dcd, pdz, base_config)
        assert alert == "YELLOW"  # Not below, so YELLOW

        # Just below red threshold
        sr = 0.299
        alert = classify_alert(sr, dc, dcd, pdz, base_config)
        assert alert == "RED"


class TestSimilarityRetention:
    """Tests for similarity_retention function."""

    def test_similarity_retention_high(self, sample_position, sample_returns, sample_windows, base_config):
        """Test high similarity retention when pattern matches."""
        today = pd.Timestamp('2024-01-30')

        # All analogs have similar features to current window
        sr = similarity_retention(sample_position, today, sample_returns, sample_windows, base_config)

        # Should return a float
        assert isinstance(sr, float)
        assert 0.0 <= sr <= 1.0

    def test_similarity_retention_no_analogs(self, base_config, sample_returns, sample_windows):
        """Test with no analogs."""
        position = {
            'symbol': 'AAPL',
            'entry_date': '2024-01-15',
            'analogs': []
        }

        today = pd.Timestamp('2024-01-30')
        sr = similarity_retention(position, today, sample_returns, sample_windows, base_config)

        assert sr == 0.0

    def test_similarity_retention_insufficient_data(self, sample_position, sample_returns, sample_windows, base_config):
        """Test with insufficient data."""
        today = pd.Timestamp('2024-01-05')  # Too early, not enough history

        sr = similarity_retention(sample_position, today, sample_returns, sample_windows, base_config)

        assert sr == 0.0


class TestDirectionalConcordance:
    """Tests for directional_concordance function."""

    def test_directional_concordance_perfect(self, base_config, sample_returns):
        """Test perfect concordance when direction matches all analogs."""
        # Create position with all UP labels
        position = {
            'symbol': 'AAPL',
            'entry_date': '2024-01-15',
            'analogs': [
                {'symbol': 'MSFT', 'end_date': '2024-01-10', 'sim': 1.0, 'label': 1},
                {'symbol': 'GOOGL', 'end_date': '2024-01-10', 'sim': 1.0, 'label': 1}
            ]
        }

        # Find a date where AAPL has positive return
        positive_dates = sample_returns[sample_returns['AAPL'] > 0].index
        if len(positive_dates) > 0:
            today = positive_dates[0]

            dc = directional_concordance(position, today, sample_returns, base_config)

            # Should be 1.0 (perfect concordance)
            assert dc == 1.0

    def test_directional_concordance_zero(self, base_config, sample_returns):
        """Test zero concordance when direction opposes all analogs."""
        # Create position with all UP labels
        position = {
            'symbol': 'AAPL',
            'entry_date': '2024-01-15',
            'analogs': [
                {'symbol': 'MSFT', 'end_date': '2024-01-10', 'sim': 1.0, 'label': 1},
                {'symbol': 'GOOGL', 'end_date': '2024-01-10', 'sim': 1.0, 'label': 1}
            ]
        }

        # Find a date where AAPL has negative return
        negative_dates = sample_returns[sample_returns['AAPL'] < 0].index
        if len(negative_dates) > 0:
            today = negative_dates[0]

            dc = directional_concordance(position, today, sample_returns, base_config)

            # Should be 0.0 (no concordance)
            assert dc == 0.0

    def test_directional_concordance_no_analogs(self, base_config, sample_returns):
        """Test with no analogs."""
        position = {
            'symbol': 'AAPL',
            'entry_date': '2024-01-15',
            'analogs': []
        }

        today = pd.Timestamp('2024-01-30')
        dc = directional_concordance(position, today, sample_returns, base_config)

        assert dc == 0.0


class TestCorrelationDecay:
    """Tests for correlation_decay function."""

    def test_correlation_decay_returns_tuple(self, sample_position, sample_returns, base_config):
        """Test that correlation_decay returns a tuple."""
        today = pd.Timestamp('2024-02-20')  # Late enough to have history

        corr_today, delta_corr = correlation_decay(sample_position, today, sample_returns, base_config)

        assert isinstance(corr_today, float)
        assert isinstance(delta_corr, float)
        assert -1.0 <= corr_today <= 1.0

    def test_correlation_decay_insufficient_data(self, sample_position, sample_returns, base_config):
        """Test with insufficient data."""
        today = pd.Timestamp('2024-01-05')  # Too early

        corr_today, delta_corr = correlation_decay(sample_position, today, sample_returns, base_config)

        assert corr_today == 0.0
        assert delta_corr == 0.0


class TestPatternDeviationZ:
    """Tests for pattern_deviation_z function."""

    def test_pattern_deviation_z_returns_float(self, sample_position, sample_returns, base_config):
        """Test that pattern_deviation_z returns a float."""
        today = pd.Timestamp('2024-02-20')

        pdz = pattern_deviation_z(sample_position, today, sample_returns, base_config)

        assert isinstance(pdz, float)

    def test_pattern_deviation_z_insufficient_data(self, sample_position, sample_returns, base_config):
        """Test with insufficient data."""
        # Use entry date itself
        today = pd.Timestamp('2024-01-15')

        pdz = pattern_deviation_z(sample_position, today, sample_returns, base_config)

        # Should return 0.0 or small value due to insufficient history
        assert isinstance(pdz, float)


class TestMonitorIntegration:
    """Integration tests for monitor_positions script."""

    @patch('scripts.monitor.load_positions_state')
    def test_monitor_positions_no_positions(self, mock_load_state, tmp_path, monkeypatch, base_config):
        """Test monitoring with no open positions."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Mock empty positions
        mock_load_state.return_value = {'open_positions': []}

        # Should not raise
        monitor_positions(base_config, monitor_date='2024-01-30')

    @patch('scripts.monitor.load_positions_state')
    @patch('scripts.monitor.save_positions_state')
    def test_monitor_positions_creates_alerts(
        self,
        mock_save_state,
        mock_load_state,
        tmp_path,
        monkeypatch,
        base_config,
        sample_position,
        sample_returns,
        sample_windows
    ):
        """Test that monitoring creates alerts CSV."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create data directories and files
        (tmp_path / 'data' / 'processed').mkdir(parents=True)
        sample_returns.to_parquet(tmp_path / 'data' / 'processed' / 'returns.parquet')
        sample_windows.to_parquet(tmp_path / 'data' / 'processed' / 'windows.parquet')

        # Mock positions
        mock_load_state.return_value = {'open_positions': [sample_position]}

        # Run monitoring
        monitor_positions(base_config, monitor_date='2024-02-20')

        # Verify alerts file created
        alerts_file = tmp_path / 'results' / 'live' / 'alerts_2024-02-20.csv'
        assert alerts_file.exists()

        # Verify alerts CSV has content
        alerts_df = pd.read_csv(alerts_file)
        assert len(alerts_df) == 1
        assert 'symbol' in alerts_df.columns
        assert 'alert_level' in alerts_df.columns
        assert 'similarity_retention' in alerts_df.columns

        # Verify state was saved
        mock_save_state.assert_called_once()
