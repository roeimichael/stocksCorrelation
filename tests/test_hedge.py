"""Tests for hedging utilities."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trading.hedging import (
    compute_hedge_volatilities,
    create_hedge_info,
    needs_rebalance,
    select_hedge_basket,
    size_hedge,
)


@pytest.fixture
def base_config():
    """Create base configuration with hedging parameters."""
    return {
        'windows': {
            'length': 10,
            'normalization': 'zscore'
        },
        'similarity': {
            'metric': 'pearson'
        },
        'hedge': {
            'enabled': True,
            'basket_size': 5,
            'min_neg_sim': -0.1,
            'target_ratio': 1.0,
            'vol_eps': 1e-6,
            'rebalance_days': 5,
            'vol_window_days': 20
        },
        'monitor': {
            'corr_window_days': 20
        }
    }


@pytest.fixture
def contrived_windows_negative_sim():
    """Create windows with contrived negative similarities."""
    # Create target-like features
    target_features = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.3, -0.1, 0.1, 0.3])

    # Create windows with varying similarity to target
    windows = []
    dates = pd.date_range('2024-01-01', periods=50, freq='B')  # More dates to avoid index issues

    # Positive similarity windows (should NOT be selected)
    for i in range(5):
        windows.append({
            'symbol': f'POS_{i}',
            'start_date': dates[i],
            'end_date': dates[i + 9],
            'features': target_features + np.random.randn(10) * 0.1,  # Similar to target
            'label': 1
        })

    # Negative similarity windows (should be selected)
    for i in range(5):
        windows.append({
            'symbol': f'NEG_{i}',
            'start_date': dates[i + 10],
            'end_date': dates[i + 19],
            'features': -target_features + np.random.randn(10) * 0.1,  # Opposite to target
            'label': 0
        })

    # Zero similarity windows (should NOT be selected)
    for i in range(5):
        windows.append({
            'symbol': f'ZERO_{i}',
            'start_date': dates[i + 25],
            'end_date': dates[i + 34],
            'features': np.random.randn(10) * 0.5,  # Uncorrelated
            'label': 1
        })

    return pd.DataFrame(windows), target_features


@pytest.fixture
def sample_returns():
    """Create sample returns DataFrame."""
    dates = pd.date_range('2024-01-01', periods=40, freq='B')

    returns_dict = {}

    # Create returns with different volatilities
    # High vol stock
    returns_dict['HIGH_VOL'] = np.random.randn(40) * 0.03

    # Low vol stock
    returns_dict['LOW_VOL'] = np.random.randn(40) * 0.01

    # Medium vol stocks for basket
    for i in range(5):
        returns_dict[f'BASKET_{i}'] = np.random.randn(40) * 0.02

    return pd.DataFrame(returns_dict, index=dates)


class TestSelectHedgeBasket:
    """Tests for select_hedge_basket function."""

    def test_select_negative_similarity_only(self, contrived_windows_negative_sim, base_config):
        """Test that only negative similarity securities are selected."""
        windows_df, target_vec = contrived_windows_negative_sim

        cutoff_date = pd.Timestamp('2024-02-15')

        basket = select_hedge_basket(
            target_vec=target_vec,
            bank_df=windows_df,
            cutoff_date=cutoff_date,
            cfg=base_config
        )

        # Should only select NEG_ symbols
        selected_symbols = [s for s, w in basket]

        # All selected should be NEG_ symbols
        assert all(s.startswith('NEG_') for s in selected_symbols), \
            f"Expected only NEG_ symbols, got: {selected_symbols}"

        # Should have at most basket_size (5) symbols
        assert len(basket) <= base_config['hedge']['basket_size']

    def test_hedge_basket_weights_are_positive(self, contrived_windows_negative_sim, base_config):
        """Test that hedge basket weights are positive (absolute values)."""
        windows_df, target_vec = contrived_windows_negative_sim

        cutoff_date = pd.Timestamp('2024-02-15')

        basket = select_hedge_basket(
            target_vec=target_vec,
            bank_df=windows_df,
            cutoff_date=cutoff_date,
            cfg=base_config
        )

        # All weights should be positive (abs of negative sim)
        for symbol, weight in basket:
            assert weight > 0, f"Weight for {symbol} should be positive, got {weight}"

    def test_hedge_basket_sorted_by_abs_sim(self, contrived_windows_negative_sim, base_config):
        """Test that basket is sorted by |negative_sim| descending."""
        windows_df, target_vec = contrived_windows_negative_sim

        cutoff_date = pd.Timestamp('2024-02-15')

        basket = select_hedge_basket(
            target_vec=target_vec,
            bank_df=windows_df,
            cutoff_date=cutoff_date,
            cfg=base_config
        )

        # Weights should be in descending order
        if len(basket) > 1:
            weights = [w for _, w in basket]
            assert weights == sorted(weights, reverse=True), \
                "Basket should be sorted by weight descending"

    def test_hedge_basket_excludes_symbol(self, contrived_windows_negative_sim, base_config):
        """Test that exclude_symbol is respected."""
        windows_df, target_vec = contrived_windows_negative_sim

        # Add windows for symbol to exclude
        exclude_sym = 'EXCLUDE_ME'
        new_window = pd.DataFrame([{
            'symbol': exclude_sym,
            'start_date': pd.Timestamp('2024-01-10'),
            'end_date': pd.Timestamp('2024-01-20'),
            'features': -target_vec,  # Strong negative similarity
            'label': 0
        }])

        windows_df = pd.concat([windows_df, new_window], ignore_index=True)

        cutoff_date = pd.Timestamp('2024-02-15')

        basket = select_hedge_basket(
            target_vec=target_vec,
            bank_df=windows_df,
            cutoff_date=cutoff_date,
            cfg=base_config,
            exclude_symbol=exclude_sym
        )

        selected_symbols = [s for s, w in basket]
        assert exclude_sym not in selected_symbols, \
            f"Excluded symbol {exclude_sym} should not be in basket"

    def test_hedge_basket_empty_when_no_negatives(self, base_config):
        """Test that empty basket is returned when no negative similarities."""
        # Create windows with only positive similarities
        target_vec = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.3, -0.1, 0.1, 0.3])

        windows = []
        dates = pd.date_range('2024-01-01', periods=20, freq='B')

        for i in range(5):
            windows.append({
                'symbol': f'POS_{i}',
                'start_date': dates[i],
                'end_date': dates[i + 9],
                'features': target_vec + np.random.randn(10) * 0.1,  # All positive sim
                'label': 1
            })

        windows_df = pd.DataFrame(windows)
        cutoff_date = pd.Timestamp('2024-02-15')

        basket = select_hedge_basket(
            target_vec=target_vec,
            bank_df=windows_df,
            cutoff_date=cutoff_date,
            cfg=base_config
        )

        assert len(basket) == 0, "Should return empty basket when no negative similarities"


class TestSizeHedge:
    """Tests for size_hedge function."""

    def test_size_hedge_equal_vol(self, base_config):
        """Test hedge sizing when volatilities are equal."""
        long_notional = 10000.0
        vol_long = 0.02
        vol_basket = 0.02

        hedge_notional = size_hedge(long_notional, vol_long, vol_basket, base_config)

        # With equal vols and target_ratio=1.0, should be equal to long_notional
        assert hedge_notional == pytest.approx(long_notional, rel=0.01)

    def test_size_hedge_high_long_vol(self, base_config):
        """Test hedge sizing when long position has higher volatility."""
        long_notional = 10000.0
        vol_long = 0.04  # 2x higher
        vol_basket = 0.02

        hedge_notional = size_hedge(long_notional, vol_long, vol_basket, base_config)

        # Should need 2x the notional to hedge
        expected = 10000.0 * (0.04 / 0.02)
        assert hedge_notional == pytest.approx(expected, rel=0.01)
        assert hedge_notional == pytest.approx(20000.0, rel=0.01)

    def test_size_hedge_low_long_vol(self, base_config):
        """Test hedge sizing when long position has lower volatility."""
        long_notional = 10000.0
        vol_long = 0.01  # 0.5x
        vol_basket = 0.02

        hedge_notional = size_hedge(long_notional, vol_long, vol_basket, base_config)

        # Should need 0.5x the notional
        expected = 10000.0 * (0.01 / 0.02)
        assert hedge_notional == pytest.approx(expected, rel=0.01)
        assert hedge_notional == pytest.approx(5000.0, rel=0.01)

    def test_size_hedge_zero_basket_vol(self, base_config):
        """Test hedge sizing with near-zero basket volatility (uses eps)."""
        long_notional = 10000.0
        vol_long = 0.02
        vol_basket = 0.0  # Zero vol

        hedge_notional = size_hedge(long_notional, vol_long, vol_basket, base_config)

        # Should use eps to prevent division by zero
        eps = base_config['hedge']['vol_eps']
        expected = 10000.0 * (0.02 / eps)

        assert hedge_notional == pytest.approx(expected, rel=0.01)

    def test_size_hedge_custom_target_ratio(self, base_config):
        """Test hedge sizing with custom target ratio."""
        # Half hedge
        base_config['hedge']['target_ratio'] = 0.5

        long_notional = 10000.0
        vol_long = 0.02
        vol_basket = 0.02

        hedge_notional = size_hedge(long_notional, vol_long, vol_basket, base_config)

        # With target_ratio=0.5, should be half
        expected = 0.5 * 10000.0
        assert hedge_notional == pytest.approx(expected, rel=0.01)


class TestNeedsRebalance:
    """Tests for needs_rebalance function."""

    def test_needs_rebalance_time_trigger(self, base_config, sample_returns):
        """Test rebalance triggered by time elapsed."""
        entry_date = pd.Timestamp('2024-01-15')
        current_date = pd.Timestamp('2024-01-25')  # 10 days later

        position = {
            'symbol': 'HIGH_VOL',
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'hedge': {
                'basket': [{'symbol': 'BASKET_0', 'raw_weight': 0.5}],
                'last_rebalance_date': entry_date.strftime('%Y-%m-%d'),
                'entry_correlation': -0.3
            }
        }

        # rebalance_days = 5, so 10 days should trigger
        needs_rebal = needs_rebalance(position, current_date, sample_returns, base_config)

        assert needs_rebal, "Should need rebalance after 10 days (threshold is 5)"

    def test_no_rebalance_within_window(self, base_config, sample_returns):
        """Test no rebalance when within time window."""
        entry_date = pd.Timestamp('2024-01-15')
        current_date = pd.Timestamp('2024-01-18')  # 3 days later

        position = {
            'symbol': 'HIGH_VOL',
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'hedge': {
                'basket': [{'symbol': 'BASKET_0', 'raw_weight': 0.5}],
                'last_rebalance_date': entry_date.strftime('%Y-%m-%d'),
                'entry_correlation': -0.3
            }
        }

        # rebalance_days = 5, so 3 days should NOT trigger
        needs_rebal = needs_rebalance(position, current_date, sample_returns, base_config)

        assert not needs_rebal, "Should NOT need rebalance within 5 days"

    def test_needs_rebalance_no_hedge_info(self, base_config, sample_returns):
        """Test with position that has no hedge info."""
        position = {
            'symbol': 'HIGH_VOL',
            'entry_date': '2024-01-15'
        }

        current_date = pd.Timestamp('2024-01-25')

        needs_rebal = needs_rebalance(position, current_date, sample_returns, base_config)

        assert not needs_rebal, "Should return False when no hedge info"


class TestComputeHedgeVolatilities:
    """Tests for compute_hedge_volatilities function."""

    def test_compute_volatilities(self, base_config, sample_returns):
        """Test volatility computation."""
        symbol = 'HIGH_VOL'
        hedge_symbols = ['BASKET_0', 'BASKET_1', 'BASKET_2']
        current_date = pd.Timestamp('2024-02-20')

        vol_long, vol_basket = compute_hedge_volatilities(
            symbol=symbol,
            hedge_symbols=hedge_symbols,
            returns_df=sample_returns,
            current_date=current_date,
            cfg=base_config
        )

        # Both should be positive
        assert vol_long > 0
        assert vol_basket > 0

        # HIGH_VOL should have higher vol than basket average
        # (Not guaranteed due to randomness, but generally true)
        assert isinstance(vol_long, float)
        assert isinstance(vol_basket, float)

    def test_compute_volatilities_missing_symbol(self, base_config, sample_returns):
        """Test with missing symbol."""
        symbol = 'MISSING'
        hedge_symbols = ['BASKET_0']
        current_date = pd.Timestamp('2024-02-20')

        vol_long, vol_basket = compute_hedge_volatilities(
            symbol=symbol,
            hedge_symbols=hedge_symbols,
            returns_df=sample_returns,
            current_date=current_date,
            cfg=base_config
        )

        # Should return zeros
        assert vol_long == 0.0
        assert vol_basket == 0.0


class TestCreateHedgeInfo:
    """Tests for create_hedge_info function."""

    def test_create_hedge_info_structure(self, contrived_windows_negative_sim, sample_returns, base_config):
        """Test that hedge info has correct structure."""
        windows_df, target_vec = contrived_windows_negative_sim

        # Add NEG_ symbols to returns
        for i in range(5):
            sample_returns[f'NEG_{i}'] = np.random.randn(len(sample_returns)) * 0.02

        # Add a symbol that will be our long position
        sample_returns['LONG_POS'] = np.random.randn(len(sample_returns)) * 0.02

        hedge_info = create_hedge_info(
            target_vec=target_vec,
            symbol='LONG_POS',
            long_notional=10000.0,
            bank_df=windows_df,
            returns_df=sample_returns,
            current_date=pd.Timestamp('2024-02-20'),
            cfg=base_config
        )

        # Should have required keys
        assert 'basket' in hedge_info
        assert 'notional' in hedge_info
        assert 'entry_correlation' in hedge_info
        assert 'last_rebalance_date' in hedge_info
        assert 'vol_long' in hedge_info
        assert 'vol_basket' in hedge_info

        # Basket should be list of dicts
        assert isinstance(hedge_info['basket'], list)
        if len(hedge_info['basket']) > 0:
            assert 'symbol' in hedge_info['basket'][0]
            assert 'raw_weight' in hedge_info['basket'][0]

        # Notional should be positive
        assert hedge_info['notional'] > 0

    def test_create_hedge_info_excludes_long_symbol(self, contrived_windows_negative_sim, sample_returns, base_config):
        """Test that long symbol is excluded from hedge basket."""
        windows_df, target_vec = contrived_windows_negative_sim

        # Add NEG_ symbols to returns
        for i in range(5):
            sample_returns[f'NEG_{i}'] = np.random.randn(len(sample_returns)) * 0.02

        # Add windows for LONG_POS with negative sim
        long_pos_window = pd.DataFrame([{
            'symbol': 'LONG_POS',
            'start_date': pd.Timestamp('2024-01-10'),
            'end_date': pd.Timestamp('2024-01-20'),
            'features': -target_vec,  # Strong negative similarity
            'label': 0
        }])

        windows_df = pd.concat([windows_df, long_pos_window], ignore_index=True)

        sample_returns['LONG_POS'] = np.random.randn(len(sample_returns)) * 0.02

        hedge_info = create_hedge_info(
            target_vec=target_vec,
            symbol='LONG_POS',
            long_notional=10000.0,
            bank_df=windows_df,
            returns_df=sample_returns,
            current_date=pd.Timestamp('2024-02-20'),
            cfg=base_config
        )

        # LONG_POS should not be in basket
        basket_symbols = [item['symbol'] for item in hedge_info.get('basket', [])]
        assert 'LONG_POS' not in basket_symbols, \
            "Long position symbol should not be in hedge basket"
