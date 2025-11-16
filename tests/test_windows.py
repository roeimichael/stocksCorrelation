"""Tests for window construction and normalization."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.windows import build_windows, normalize_window


@pytest.fixture
def test_config():
    """Test configuration for window building."""
    return {
        'windows': {
            'length': 5,  # Short window for testing
            'normalization': 'zscore',
            'min_history_days': 10  # Short history for testing
        }
    }


@pytest.fixture
def synthetic_returns():
    """
    Create synthetic returns for 2 symbols with known patterns.

    Returns DataFrame with 20 days of data:
    - Symbol A: Uptrend (positive returns)
    - Symbol B: Downtrend (negative returns)
    """
    dates = pd.date_range('2024-01-01', periods=20, freq='B')

    # Symbol A: Uptrend with positive returns
    returns_a = np.array([0.01, 0.02, 0.01, 0.015, 0.01,
                          0.02, 0.01, 0.015, 0.01, 0.02,
                          0.01, 0.02, 0.015, 0.01, 0.02,
                          0.01, 0.015, 0.02, 0.01, 0.02])

    # Symbol B: Downtrend with negative returns
    returns_b = np.array([-0.01, -0.02, -0.01, -0.015, -0.01,
                          -0.02, -0.01, -0.015, -0.01, -0.02,
                          -0.01, -0.02, -0.015, -0.01, -0.02,
                          -0.01, -0.015, -0.02, -0.01, -0.02])

    returns_df = pd.DataFrame({
        'STOCK_A': returns_a,
        'STOCK_B': returns_b
    }, index=dates)

    return returns_df


class TestNormalizeWindow:
    """Tests for normalize_window function."""

    def test_zscore_normalization(self):
        """Test z-score normalization produces mean≈0, std≈1."""
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_window(vec, method='zscore')

        # Check mean is approximately 0
        assert np.abs(np.mean(normalized)) < 1e-10, f"Mean should be ~0, got {np.mean(normalized)}"

        # Check std is approximately 1 (allow small floating point error)
        assert np.abs(np.std(normalized) - 1.0) < 1e-6, f"Std should be ~1, got {np.std(normalized)}"

        # Check length preserved
        assert len(normalized) == len(vec)

    def test_zscore_with_zero_variance(self):
        """Test z-score handles constant values (zero variance)."""
        vec = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        normalized = normalize_window(vec, method='zscore')

        # Should return zero-centered values (all zeros)
        assert np.allclose(normalized, 0.0), "Constant values should normalize to zero"

    def test_rank_normalization(self):
        """Test rank normalization produces values in [-0.5, 0.5]."""
        vec = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
        normalized = normalize_window(vec, method='rank')

        # Check range is [-0.5, 0.5]
        assert normalized.min() >= -0.5, f"Min should be >= -0.5, got {normalized.min()}"
        assert normalized.max() <= 0.5, f"Max should be <= 0.5, got {normalized.max()}"

        # Check length preserved
        assert len(normalized) == len(vec)

        # Check ranks are correct (1->-0.5, 5->0.5, etc.)
        # Smallest value (1.0) should map to -0.5
        # Largest value (5.0) should map to 0.5
        assert normalized[0] == -0.5  # vec[0]=1.0 is smallest
        assert normalized[1] == 0.5   # vec[1]=5.0 is largest

    def test_rank_normalization_with_ties(self):
        """Test rank normalization handles tied values."""
        vec = np.array([1.0, 2.0, 2.0, 3.0])
        normalized = normalize_window(vec, method='rank')

        # Ties should get average rank
        # Ranks: 1, 2.5, 2.5, 4
        # Normalized: (ranks-1)/(n-1) - 0.5 = (ranks-1)/3 - 0.5
        assert len(normalized) == len(vec)
        assert -0.5 <= normalized.min() <= normalized.max() <= 0.5

    def test_vol_normalization(self):
        """Test volatility normalization scales by std."""
        vec = np.array([0.01, 0.02, 0.015, 0.01, 0.02])
        normalized = normalize_window(vec, method='vol')

        # Normalized std should be approximately 1
        normalized_std = np.std(normalized)
        assert np.abs(normalized_std - 1.0) < 0.01, f"Normalized std should be ~1, got {normalized_std}"

        # Length preserved
        assert len(normalized) == len(vec)

        # Original pattern preserved (just scaled)
        # Check that dividing by std approximately recovers the scaling
        original_std = np.std(vec)
        # Verify normalized std is close to 1 (accounting for epsilon in denominator)
        expected_normalized_std = 1.0
        assert np.abs(normalized_std - expected_normalized_std) < 0.05

    def test_unknown_method_raises_error(self):
        """Test that unknown normalization method raises ValueError."""
        vec = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_window(vec, method='unknown')

    def test_empty_array(self):
        """Test handling of empty array."""
        vec = np.array([])
        normalized = normalize_window(vec, method='zscore')
        assert len(normalized) == 0

    def test_single_value(self):
        """Test handling of single-value array."""
        vec = np.array([5.0])

        # Z-score of single value should be 0 (zero-centered)
        normalized_z = normalize_window(vec, method='zscore')
        assert np.allclose(normalized_z, 0.0)

        # Rank of single value should be 0 (centered)
        normalized_r = normalize_window(vec, method='rank')
        assert np.allclose(normalized_r, 0.0)


class TestBuildWindows:
    """Tests for build_windows function."""

    def test_window_count(self, synthetic_returns, test_config, tmp_path):
        """Test that correct number of windows is created."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            windows_df = build_windows(synthetic_returns, test_config)

            # Expected windows per symbol:
            # Data: 20 days
            # min_history: 10 days
            # window_length: 5 days
            # Start index: max(5, 10) = 10
            # Windows: from index 10 to 20 (inclusive) = 11 windows per symbol
            # Total: 11 * 2 symbols = 22 windows

            expected_count = 11 * 2  # 11 windows per symbol × 2 symbols
            assert len(windows_df) == expected_count, f"Expected {expected_count} windows, got {len(windows_df)}"

        finally:
            os.chdir(original_cwd)

    def test_features_length(self, synthetic_returns, test_config, tmp_path):
        """Test that all features arrays have length X."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            windows_df = build_windows(synthetic_returns, test_config)

            # Check all features have correct length
            window_length = test_config['windows']['length']
            for idx, row in windows_df.iterrows():
                features = row['features']
                assert len(features) == window_length, f"Features length should be {window_length}, got {len(features)}"

        finally:
            os.chdir(original_cwd)

    def test_labels_computed_correctly(self, synthetic_returns, test_config, tmp_path):
        """Test that labels are correctly computed based on next-day return."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            windows_df = build_windows(synthetic_returns, test_config)

            # Symbol A has all positive returns, so all labels should be 1 (except last)
            windows_a = windows_df[windows_df['symbol'] == 'STOCK_A']
            labels_a = windows_a['label'].values

            # All but last window should have label 1 (next day is positive)
            assert (labels_a[:-1] == 1).all(), "STOCK_A should have all labels = 1 (except last)"

            # Last window should have label -1 (no next day)
            assert labels_a[-1] == -1, "Last window should have label = -1"

            # Symbol B has all negative returns, so all labels should be 0 (except last)
            windows_b = windows_df[windows_df['symbol'] == 'STOCK_B']
            labels_b = windows_b['label'].values

            # All but last window should have label 0 (next day is negative)
            assert (labels_b[:-1] == 0).all(), "STOCK_B should have all labels = 0 (except last)"

            # Last window should have label -1 (no next day)
            assert labels_b[-1] == -1, "Last window should have label = -1"

        finally:
            os.chdir(original_cwd)

    def test_zscore_normalization_mean_near_zero(self, synthetic_returns, test_config, tmp_path):
        """Test that z-score normalized features have mean≈0."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            test_config['windows']['normalization'] = 'zscore'
            windows_df = build_windows(synthetic_returns, test_config)

            # Check a few random windows
            for idx in [0, 5, 10]:
                if idx < len(windows_df):
                    features = windows_df.iloc[idx]['features']
                    mean = np.mean(features)
                    assert np.abs(mean) < 1e-10, f"Z-score normalized mean should be ~0, got {mean}"

        finally:
            os.chdir(original_cwd)

    def test_rank_normalization_range(self, synthetic_returns, test_config, tmp_path):
        """Test that rank normalized features are in [-0.5, 0.5]."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            test_config['windows']['normalization'] = 'rank'
            windows_df = build_windows(synthetic_returns, test_config)

            # Check all windows
            for idx, row in windows_df.iterrows():
                features = row['features']
                assert features.min() >= -0.5, f"Rank features min should be >= -0.5, got {features.min()}"
                assert features.max() <= 0.5, f"Rank features max should be <= 0.5, got {features.max()}"

        finally:
            os.chdir(original_cwd)

    def test_vol_normalization(self, synthetic_returns, test_config, tmp_path):
        """Test that vol normalized features preserve relative magnitudes."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            test_config['windows']['normalization'] = 'vol'
            windows_df = build_windows(synthetic_returns, test_config)

            # Vol normalization should scale by std
            # Check that std of normalized features is approximately 1
            for idx in [0, 5, 10]:
                if idx < len(windows_df):
                    features = windows_df.iloc[idx]['features']
                    std = np.std(features)
                    assert 0.5 < std < 2.0, f"Vol normalized std should be ~1, got {std}"

        finally:
            os.chdir(original_cwd)

    def test_last_window_no_label(self, synthetic_returns, test_config, tmp_path):
        """Test that last window per symbol has no label (label=-1)."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            windows_df = build_windows(synthetic_returns, test_config)

            # Get last window for each symbol
            for symbol in ['STOCK_A', 'STOCK_B']:
                symbol_windows = windows_df[windows_df['symbol'] == symbol]
                last_window = symbol_windows.iloc[-1]

                assert last_window['label'] == -1, f"Last window for {symbol} should have label=-1"

        finally:
            os.chdir(original_cwd)

    def test_windows_saved_to_parquet(self, synthetic_returns, test_config, tmp_path):
        """Test that windows are saved to parquet and reloadable."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            windows_df = build_windows(synthetic_returns, test_config)

            # Check file exists
            parquet_file = Path('data/processed/windows.parquet')
            assert parquet_file.exists(), "windows.parquet should exist"

            # Reload and verify
            reloaded_df = pd.read_parquet(parquet_file)

            # Check shapes match
            assert reloaded_df.shape == windows_df.shape

            # Check columns match
            assert list(reloaded_df.columns) == list(windows_df.columns)

            # Check some values match
            assert reloaded_df['symbol'].tolist() == windows_df['symbol'].tolist()

        finally:
            os.chdir(original_cwd)

    def test_insufficient_data_raises_error(self, test_config):
        """Test that insufficient data raises ValueError."""
        # Create returns with fewer days than min_history
        short_returns = pd.DataFrame({
            'A': [0.01, 0.02, 0.01, 0.02, 0.01]
        }, index=pd.date_range('2024-01-01', periods=5))

        with pytest.raises(ValueError, match="Insufficient data"):
            build_windows(short_returns, test_config)

    def test_date_ranges(self, synthetic_returns, test_config, tmp_path):
        """Test that start_date and end_date are correctly set."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            windows_df = build_windows(synthetic_returns, test_config)

            window_length = test_config['windows']['length']

            for idx, row in windows_df.iterrows():
                start_date = row['start_date']
                end_date = row['end_date']

                # Check that dates are valid
                assert isinstance(start_date, pd.Timestamp)
                assert isinstance(end_date, pd.Timestamp)

                # Check that end_date > start_date
                assert end_date > start_date

                # Check that window spans correct number of days
                # Note: This is business days, so we can't check exact day difference
                # Just verify they're in chronological order
                assert start_date <= end_date

        finally:
            os.chdir(original_cwd)


class TestIntegration:
    """Integration tests for window building."""

    def test_full_pipeline_multiple_normalizations(self, synthetic_returns, test_config, tmp_path):
        """Test building windows with all normalization methods."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            for method in ['zscore', 'rank', 'vol']:
                test_config['windows']['normalization'] = method

                windows_df = build_windows(synthetic_returns, test_config)

                # Basic checks
                assert len(windows_df) > 0, f"Should create windows with {method} normalization"
                assert 'features' in windows_df.columns
                assert 'label' in windows_df.columns
                assert 'symbol' in windows_df.columns

                # Check features are properly normalized
                window_length = test_config['windows']['length']
                for idx, row in windows_df.head(3).iterrows():
                    features = row['features']
                    assert len(features) == window_length
                    assert not np.isnan(features).any()

        finally:
            os.chdir(original_cwd)
