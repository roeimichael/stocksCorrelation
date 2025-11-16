"""Tests for evaluation and metrics modules."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evals.correlation_matrix import corr_matrix, save_heatmap
from src.evals.metrics import equity_from_trades, summary_metrics


@pytest.fixture
def sample_returns():
    """Create sample returns DataFrame for testing."""
    dates = pd.date_range('2024-01-01', periods=20, freq='B')
    np.random.seed(42)

    returns = pd.DataFrame({
        'STOCK_A': np.random.randn(20) * 0.02,
        'STOCK_B': np.random.randn(20) * 0.02,
        'STOCK_C': np.random.randn(20) * 0.02
    }, index=dates)

    return returns


@pytest.fixture
def fake_trades_positive():
    """Create fake trades with known positive PnL."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5, freq='B'),
        'pnl': [100, 50, 150, 80, 120],  # All positive
        'notional': [10000] * 5
    })


@pytest.fixture
def fake_trades_mixed():
    """Create fake trades with mixed PnL."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='B'),
        'pnl': [100, -50, 150, -30, 80, 120, -40, 60, -20, 90],  # 6 wins, 4 losses
        'notional': [10000] * 10
    })


@pytest.fixture
def fake_trades_no_notional():
    """Create fake trades without notional column."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5, freq='B'),
        'pnl': [100, 50, 150, 80, 120]
    })


class TestCorrelationMatrix:
    """Tests for correlation matrix computation."""

    def test_corr_matrix_shape(self, sample_returns):
        """Test correlation matrix has correct shape (N x N)."""
        corr = corr_matrix(sample_returns)

        # Shape should be (n_symbols, n_symbols)
        n_symbols = len(sample_returns.columns)
        assert corr.shape == (n_symbols, n_symbols), \
            f"Expected shape ({n_symbols}, {n_symbols}), got {corr.shape}"

    def test_corr_matrix_symmetric(self, sample_returns):
        """Test correlation matrix is symmetric."""
        corr = corr_matrix(sample_returns)

        # Check symmetry: corr[i,j] == corr[j,i]
        assert np.allclose(corr.values, corr.values.T), "Correlation matrix should be symmetric"

    def test_corr_matrix_diagonal_ones(self, sample_returns):
        """Test correlation matrix has 1.0 on diagonal."""
        corr = corr_matrix(sample_returns)

        # Diagonal should be all 1.0 (correlation with self)
        diagonal = np.diag(corr.values)
        assert np.allclose(diagonal, 1.0), f"Diagonal should be all 1.0, got {diagonal}"

    def test_corr_matrix_values_in_range(self, sample_returns):
        """Test correlation values are in [-1, 1]."""
        corr = corr_matrix(sample_returns)

        # All values should be in [-1, 1]
        assert corr.min().min() >= -1.0, f"Min correlation should be >= -1, got {corr.min().min()}"
        assert corr.max().max() <= 1.0, f"Max correlation should be <= 1, got {corr.max().max()}"

    def test_corr_matrix_with_date_filter(self, sample_returns):
        """Test correlation matrix with start/end date filtering."""
        # Filter to first 10 days
        start = sample_returns.index[0]
        end = sample_returns.index[9]

        corr = corr_matrix(sample_returns, start=start, end=end)

        # Should still be square matrix
        n_symbols = len(sample_returns.columns)
        assert corr.shape == (n_symbols, n_symbols)

    def test_corr_matrix_start_only(self, sample_returns):
        """Test correlation matrix with only start date."""
        start = sample_returns.index[5]  # Start from 6th day

        corr = corr_matrix(sample_returns, start=start)

        # Should still be square matrix
        n_symbols = len(sample_returns.columns)
        assert corr.shape == (n_symbols, n_symbols)

    def test_corr_matrix_end_only(self, sample_returns):
        """Test correlation matrix with only end date."""
        end = sample_returns.index[14]  # End at 15th day

        corr = corr_matrix(sample_returns, end=end)

        # Should still be square matrix
        n_symbols = len(sample_returns.columns)
        assert corr.shape == (n_symbols, n_symbols)

    def test_save_heatmap_creates_file(self, sample_returns, tmp_path):
        """Test save_heatmap creates output file."""
        corr = corr_matrix(sample_returns)
        output_path = tmp_path / "test_corr.png"

        save_heatmap(corr, output_path=str(output_path))

        # Check file was created
        assert output_path.exists(), f"Heatmap file should exist at {output_path}"

    def test_save_heatmap_default_path(self, sample_returns):
        """Test save_heatmap with default path (timestamp)."""
        corr = corr_matrix(sample_returns)

        # Call without output_path (will use default with timestamp)
        save_heatmap(corr)

        # Check that results/plots directory was created
        plots_dir = Path('results/plots')
        assert plots_dir.exists(), "results/plots directory should be created"

        # Check that at least one PNG file exists
        png_files = list(plots_dir.glob('corr_*.png'))
        assert len(png_files) > 0, "Should have created at least one corr_*.png file"


class TestEquityFromTrades:
    """Tests for equity curve computation."""

    def test_equity_from_trades_positive_pnl(self, fake_trades_positive):
        """Test equity curve with all positive PnL."""
        equity_df = equity_from_trades(fake_trades_positive, costs_bps=5.0, slippage_bps=2.0)

        # Check shape
        assert len(equity_df) == len(fake_trades_positive), \
            f"Expected {len(fake_trades_positive)} equity points, got {len(equity_df)}"

        # Check equity is cumulative (should be increasing with all positive PnL)
        equity_values = equity_df['equity'].values
        assert np.all(np.diff(equity_values) > 0), "Equity should be increasing with all positive PnL"

    def test_equity_from_trades_costs_applied(self, fake_trades_positive):
        """Test that costs are correctly applied."""
        # Without costs
        equity_no_costs = equity_from_trades(fake_trades_positive, costs_bps=0.0, slippage_bps=0.0)

        # With costs
        equity_with_costs = equity_from_trades(fake_trades_positive, costs_bps=5.0, slippage_bps=2.0)

        # Final equity with costs should be less than without costs
        final_no_costs = equity_no_costs['equity'].iloc[-1]
        final_with_costs = equity_with_costs['equity'].iloc[-1]

        assert final_with_costs < final_no_costs, \
            f"Equity with costs ({final_with_costs}) should be less than without costs ({final_no_costs})"

    def test_equity_from_trades_cumsum(self, fake_trades_mixed):
        """Test that equity is cumulative sum of net PnL."""
        equity_df = equity_from_trades(fake_trades_mixed, costs_bps=5.0, slippage_bps=2.0)

        # Compute expected: net_pnl - costs
        total_cost_rate = (5.0 + 2.0) / 10000.0
        costs = fake_trades_mixed['notional'] * total_cost_rate
        net_pnl = fake_trades_mixed['pnl'] - costs
        expected_equity = net_pnl.cumsum().values

        # Check equity matches expected cumsum
        actual_equity = equity_df['equity'].values

        assert np.allclose(actual_equity, expected_equity, atol=0.01), \
            "Equity should be cumsum of net PnL"

    def test_equity_from_trades_without_notional(self, fake_trades_no_notional):
        """Test equity computation without notional column (estimates notional)."""
        equity_df = equity_from_trades(fake_trades_no_notional, costs_bps=5.0, slippage_bps=2.0)

        # Should still produce equity curve
        assert len(equity_df) == len(fake_trades_no_notional)
        assert 'equity' in equity_df.columns

    def test_equity_from_trades_empty(self):
        """Test equity computation with empty trades."""
        empty_trades = pd.DataFrame(columns=['date', 'pnl', 'notional'])

        equity_df = equity_from_trades(empty_trades, costs_bps=5.0, slippage_bps=2.0)

        # Should return empty DataFrame with equity column
        assert len(equity_df) == 0
        assert 'equity' in equity_df.columns

    def test_equity_from_trades_missing_date_raises(self):
        """Test that missing 'date' column raises error."""
        trades = pd.DataFrame({'pnl': [100, 50]})

        with pytest.raises(ValueError, match="must have 'date' column"):
            equity_from_trades(trades, costs_bps=5.0, slippage_bps=2.0)

    def test_equity_from_trades_missing_pnl_raises(self):
        """Test that missing 'pnl' column raises error."""
        trades = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=2)})

        with pytest.raises(ValueError, match="must have 'pnl' column"):
            equity_from_trades(trades, costs_bps=5.0, slippage_bps=2.0)


class TestSummaryMetrics:
    """Tests for summary metrics computation."""

    def test_summary_metrics_positive_equity(self, fake_trades_positive):
        """Test summary metrics with positive equity curve."""
        equity_df = equity_from_trades(fake_trades_positive, costs_bps=5.0, slippage_bps=2.0)
        metrics = summary_metrics(equity_df, trades_df=fake_trades_positive)

        # Check all metrics are present
        assert 'total_return' in metrics
        assert 'sharpe' in metrics
        assert 'max_dd' in metrics
        assert 'hit_rate' in metrics

        # With all positive PnL, total_return should be positive
        assert metrics['total_return'] > 0, f"Expected positive total_return, got {metrics['total_return']}"

        # With all positive PnL, hit_rate should be 1.0
        assert metrics['hit_rate'] == 1.0, f"Expected hit_rate=1.0 (all wins), got {metrics['hit_rate']}"

    def test_summary_metrics_hit_rate_calculation(self, fake_trades_mixed):
        """Test hit rate calculation with known wins/losses."""
        equity_df = equity_from_trades(fake_trades_mixed, costs_bps=5.0, slippage_bps=2.0)
        metrics = summary_metrics(equity_df, trades_df=fake_trades_mixed)

        # fake_trades_mixed has 6 wins, 4 losses -> hit_rate = 6/10 = 0.6
        winning_trades = (fake_trades_mixed['pnl'] > 0).sum()
        total_trades = len(fake_trades_mixed)
        expected_hit_rate = winning_trades / total_trades

        assert np.isclose(metrics['hit_rate'], expected_hit_rate), \
            f"Expected hit_rate={expected_hit_rate}, got {metrics['hit_rate']}"

    def test_summary_metrics_max_dd_negative(self, fake_trades_mixed):
        """Test that max drawdown is negative (or zero if no drawdown)."""
        equity_df = equity_from_trades(fake_trades_mixed, costs_bps=5.0, slippage_bps=2.0)
        metrics = summary_metrics(equity_df)

        # Max drawdown should be <= 0
        assert metrics['max_dd'] <= 0, f"Expected max_dd <= 0, got {metrics['max_dd']}"

    def test_summary_metrics_sharpe_ratio(self, fake_trades_positive):
        """Test Sharpe ratio calculation."""
        equity_df = equity_from_trades(fake_trades_positive, costs_bps=5.0, slippage_bps=2.0)
        metrics = summary_metrics(equity_df)

        # Sharpe should be a float
        assert isinstance(metrics['sharpe'], float)

        # With positive equity, Sharpe should be positive
        assert metrics['sharpe'] > 0, f"Expected positive Sharpe, got {metrics['sharpe']}"

    def test_summary_metrics_without_trades(self, fake_trades_mixed):
        """Test summary metrics without trades_df (hit_rate should be None)."""
        equity_df = equity_from_trades(fake_trades_mixed, costs_bps=5.0, slippage_bps=2.0)
        metrics = summary_metrics(equity_df, trades_df=None)

        # hit_rate should be None when trades_df not provided
        assert metrics['hit_rate'] is None

    def test_summary_metrics_empty_equity(self):
        """Test summary metrics with empty equity DataFrame."""
        empty_equity = pd.DataFrame(columns=['equity'])

        metrics = summary_metrics(empty_equity)

        # Should return zeros
        assert metrics['total_return'] == 0.0
        assert metrics['sharpe'] == 0.0
        assert metrics['max_dd'] == 0.0
        assert metrics['hit_rate'] is None

    def test_summary_metrics_missing_equity_column_raises(self):
        """Test that missing 'equity' column raises error."""
        df = pd.DataFrame({'not_equity': [100, 150, 120]})

        with pytest.raises(ValueError, match="must have 'equity' column"):
            summary_metrics(df)


class TestIntegration:
    """Integration tests for evaluation modules."""

    def test_full_pipeline_equity_to_metrics(self, fake_trades_mixed):
        """Test full pipeline from trades to metrics."""
        # Step 1: Compute equity curve
        equity_df = equity_from_trades(fake_trades_mixed, costs_bps=5.0, slippage_bps=2.0)

        # Step 2: Compute summary metrics
        metrics = summary_metrics(equity_df, trades_df=fake_trades_mixed)

        # Verify all metrics are computed
        assert 'total_return' in metrics
        assert 'sharpe' in metrics
        assert 'max_dd' in metrics
        assert 'hit_rate' in metrics

        # Verify hit_rate matches known value (6 wins out of 10)
        expected_hit_rate = 0.6
        assert np.isclose(metrics['hit_rate'], expected_hit_rate)

    def test_correlation_and_metrics_independent(self, sample_returns, fake_trades_positive):
        """Test that correlation and metrics modules work independently."""
        # Correlation analysis
        corr = corr_matrix(sample_returns)
        assert corr.shape[0] == len(sample_returns.columns)

        # Metrics analysis
        equity_df = equity_from_trades(fake_trades_positive, costs_bps=5.0, slippage_bps=2.0)
        metrics = summary_metrics(equity_df)
        assert metrics['total_return'] > 0

    def test_known_pnl_equity_sum(self):
        """Test equity curve sums correctly with known PnL values."""
        # Create trades with known PnL
        trades = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=4, freq='B'),
            'pnl': [100, -50, 150, -30],  # Sum = 170
            'notional': [10000] * 4
        })

        # Zero costs to make calculation simple
        equity_df = equity_from_trades(trades, costs_bps=0.0, slippage_bps=0.0)

        # Final equity should equal sum of PnL
        expected_total = sum(trades['pnl'])
        actual_total = equity_df['equity'].iloc[-1]

        assert np.isclose(actual_total, expected_total), \
            f"Expected final equity={expected_total}, got {actual_total}"

    def test_equity_path_correctness(self):
        """Test that equity path is computed correctly step by step."""
        # Create simple trades
        trades = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3, freq='B'),
            'pnl': [100, 50, -30],
            'notional': [10000] * 3
        })

        # Zero costs for simplicity
        equity_df = equity_from_trades(trades, costs_bps=0.0, slippage_bps=0.0)

        # Expected equity path: cumsum([100, 50, -30]) = [100, 150, 120]
        expected_equity = [100, 150, 120]
        actual_equity = equity_df['equity'].values

        assert np.allclose(actual_equity, expected_equity), \
            f"Expected equity path {expected_equity}, got {actual_equity}"
