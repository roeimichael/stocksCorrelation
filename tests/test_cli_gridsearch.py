"""Tests for CLI grid search script."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gridsearch import run_grid_search


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return {
        'data': {
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'test_start_date': '2024-01-20'
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
        },
        'light_mode': {
            'enabled': True,
            'top_n_stocks': 10,
            'test_days': 5
        }
    }


@pytest.fixture
def deterministic_summary():
    """Create deterministic backtest summary."""
    return {
        'total_return': 1000.0,
        'sharpe': 1.5,
        'max_dd': -0.10,
        'hit_rate': 0.60,
        'n_trades': 10
    }


class TestRunGridSearch:
    """Tests for run_grid_search function."""

    @patch('scripts.gridsearch.run_backtest')
    def test_run_grid_search_returns_dataframe(self, mock_run_backtest, mock_config, deterministic_summary):
        """Test that grid search returns a DataFrame."""
        # Mock run_backtest to return deterministic results
        mock_run_backtest.return_value = deterministic_summary

        # Run grid search (with small grid to avoid timeout)
        results_df = run_grid_search(mock_config)

        # Should return DataFrame
        assert isinstance(results_df, pd.DataFrame)

        # Should have results
        assert len(results_df) > 0

    @patch('scripts.gridsearch.run_backtest')
    def test_run_grid_search_has_required_columns(self, mock_run_backtest, mock_config, deterministic_summary):
        """Test that results DataFrame has required columns."""
        # Mock run_backtest
        mock_run_backtest.return_value = deterministic_summary

        # Run grid search
        results_df = run_grid_search(mock_config)

        # Check required columns
        required_columns = [
            'window_length',
            'metric',
            'top_k',
            'threshold',
            'total_return',
            'sharpe',
            'max_dd',
            'hit_rate',
            'n_trades'
        ]

        for col in required_columns:
            assert col in results_df.columns, f"Missing column: {col}"

    @patch('scripts.gridsearch.run_backtest')
    def test_run_grid_search_correct_number_of_combinations(self, mock_run_backtest, mock_config, deterministic_summary):
        """Test that grid search runs correct number of combinations."""
        # Mock run_backtest
        mock_run_backtest.return_value = deterministic_summary

        # Run grid search
        results_df = run_grid_search(mock_config)

        # Expected number of combinations:
        # windows.length: [5, 10, 15] = 3
        # similarity.metric: ['pearson', 'spearman', 'cosine'] = 3
        # similarity.top_k: [10, 25, 50] = 3
        # vote.threshold: [0.60, 0.65, 0.70, 0.75] = 4
        # Total: 3 * 3 * 3 * 4 = 108

        expected_combinations = 3 * 3 * 3 * 4
        assert len(results_df) == expected_combinations

        # Verify run_backtest was called correct number of times
        assert mock_run_backtest.call_count == expected_combinations

    @patch('scripts.gridsearch.run_backtest')
    def test_run_grid_search_parameter_sweep(self, mock_run_backtest, mock_config, deterministic_summary):
        """Test that grid search sweeps all parameters correctly."""
        # Mock run_backtest
        mock_run_backtest.return_value = deterministic_summary

        # Run grid search
        results_df = run_grid_search(mock_config)

        # Check that all parameter values appear in results
        assert set(results_df['window_length'].unique()) == {5, 10, 15}
        assert set(results_df['metric'].unique()) == {'pearson', 'spearman', 'cosine'}
        assert set(results_df['top_k'].unique()) == {10, 25, 50}
        assert set(results_df['threshold'].unique()) == {0.60, 0.65, 0.70, 0.75}

    @patch('scripts.gridsearch.run_backtest')
    def test_run_grid_search_handles_failures(self, mock_run_backtest, mock_config):
        """Test that grid search handles failed backtests gracefully."""
        # Mock run_backtest to raise exception
        mock_run_backtest.side_effect = Exception("Backtest failed")

        # Run grid search - should not raise
        results_df = run_grid_search(mock_config)

        # Should still return DataFrame with failed results
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) > 0

        # All results should have zero metrics
        assert (results_df['sharpe'] == 0.0).all()
        assert (results_df['hit_rate'] == 0.0).all()
        assert (results_df['n_trades'] == 0).all()

    @patch('scripts.gridsearch.run_backtest')
    def test_run_grid_search_varying_returns(self, mock_run_backtest, mock_config):
        """Test grid search with varying backtest returns."""
        # Mock run_backtest to return different results based on parameters
        def mock_backtest_side_effect(cfg):
            # Better performance for window_length=10
            if cfg['windows']['length'] == 10:
                return {
                    'total_return': 2000.0,
                    'sharpe': 2.0,
                    'max_dd': -0.05,
                    'hit_rate': 0.70,
                    'n_trades': 15
                }
            return {
                'total_return': 500.0,
                'sharpe': 0.5,
                'max_dd': -0.15,
                'hit_rate': 0.50,
                'n_trades': 5
            }

        mock_run_backtest.side_effect = mock_backtest_side_effect

        # Run grid search
        results_df = run_grid_search(mock_config)

        # Check that window_length=10 has better results
        win_10_results = results_df[results_df['window_length'] == 10]
        win_5_results = results_df[results_df['window_length'] == 5]

        assert win_10_results['sharpe'].mean() > win_5_results['sharpe'].mean()
        assert win_10_results['hit_rate'].mean() > win_5_results['hit_rate'].mean()


class TestIntegration:
    """Integration tests for grid search."""

    @patch('scripts.gridsearch.run_backtest')
    def test_grid_search_creates_output_file(self, mock_run_backtest, tmp_path, monkeypatch, mock_config, deterministic_summary):
        """Test that grid search creates output CSV file."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Mock run_backtest
        mock_run_backtest.return_value = deterministic_summary

        # Import main to run full script
        from scripts.gridsearch import main as gridsearch_main

        # Mock argv to pass config
        test_config_path = tmp_path / 'test_config.yaml'
        import yaml
        with open(test_config_path, 'w') as f:
            yaml.dump(mock_config, f)

        with patch('sys.argv', ['gridsearch.py', '--config', str(test_config_path)]):
            gridsearch_main()

        # Check that results directory was created
        results_dir = tmp_path / 'results' / 'experiments'
        assert results_dir.exists()

        # Check that CSV file was created
        csv_files = list(results_dir.glob('gridsearch_*.csv'))
        assert len(csv_files) == 1

        # Verify CSV has correct structure
        results_df = pd.read_csv(csv_files[0])
        required_columns = ['window_length', 'metric', 'top_k', 'threshold', 'sharpe', 'hit_rate']
        for col in required_columns:
            assert col in results_df.columns

    @patch('scripts.gridsearch.run_backtest')
    def test_grid_search_best_result_selection(self, mock_run_backtest, mock_config):
        """Test that best result is selected correctly."""
        # Mock run_backtest with varying results
        call_count = [0]

        def mock_backtest_side_effect(cfg):
            call_count[0] += 1
            # Make first call best
            if call_count[0] == 1:
                return {
                    'total_return': 5000.0,
                    'sharpe': 3.0,
                    'max_dd': -0.02,
                    'hit_rate': 0.80,
                    'n_trades': 20
                }
            return {
                'total_return': 500.0,
                'sharpe': 0.5,
                'max_dd': -0.15,
                'hit_rate': 0.50,
                'n_trades': 5
            }

        mock_run_backtest.side_effect = mock_backtest_side_effect

        # Run grid search
        results_df = run_grid_search(mock_config)

        # Sort by sharpe (as main does)
        results_df = results_df.sort_values(['sharpe', 'hit_rate'], ascending=[False, False])

        # Best result should be first
        best = results_df.iloc[0]
        assert best['sharpe'] == 3.0
        assert best['hit_rate'] == 0.80
        assert best['n_trades'] == 20

    @patch('scripts.gridsearch.run_backtest')
    def test_grid_search_empty_results(self, mock_run_backtest, mock_config):
        """Test grid search when all backtests fail."""
        # Mock run_backtest to always fail
        mock_run_backtest.side_effect = Exception("All backtests failed")

        # Run grid search
        results_df = run_grid_search(mock_config)

        # Should still have results with zeros
        assert len(results_df) > 0
        assert (results_df['sharpe'] == 0.0).all()
        assert (results_df['n_trades'] == 0).all()
