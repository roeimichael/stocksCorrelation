"""Tests for position closing logic."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.close_positions import append_to_ledger, close_positions, compute_pnl, get_prices, should_close_position


@pytest.fixture
def base_config():
    """Create base configuration with closing policy."""
    return {
        'close_policy': {
            'close_on_red': True,
            'close_on_reverse': True,
            'close_on_abstain': False
        },
        'backtest': {
            'costs_bps': 5.0,
            'slippage_bps': 2.0
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
        'notional': 10000.0,
        'alert_level': 'GREEN'
    }


@pytest.fixture
def sample_returns():
    """Create sample returns DataFrame."""
    dates = pd.date_range('2024-01-01', periods=40, freq='B')

    # Create returns with known cumulative behavior
    returns = np.array([0.01] * 20 + [-0.01] * 20)  # Up 20 days, down 20 days

    return pd.DataFrame({
        'AAPL': returns,
        'MSFT': returns * 0.5,
        'GOOGL': returns * -1  # Inverse
    }, index=dates)


@pytest.fixture
def sample_signals():
    """Create sample signals DataFrame."""
    return pd.DataFrame([
        {'symbol': 'AAPL', 'signal': 'UP', 'p_up': 0.8, 'confidence': 0.3},
        {'symbol': 'MSFT', 'signal': 'DOWN', 'p_up': 0.3, 'confidence': 0.4},
        {'symbol': 'GOOGL', 'signal': 'ABSTAIN', 'p_up': 0.5, 'confidence': 0.0}
    ])


class TestShouldClosePosition:
    """Tests for should_close_position function."""

    def test_close_on_red_alert(self, base_config, sample_signals):
        """Test closing on RED alert."""
        position = {
            'symbol': 'AAPL',
            'side': 'UP',
            'alert_level': 'RED'
        }

        should_close, reason = should_close_position(position, sample_signals, base_config)

        assert should_close is True
        assert reason == 'RED_ALERT'

    def test_no_close_on_yellow_alert(self, base_config, sample_signals):
        """Test no closing on YELLOW alert."""
        position = {
            'symbol': 'AAPL',
            'side': 'UP',
            'alert_level': 'YELLOW'
        }

        should_close, reason = should_close_position(position, sample_signals, base_config)

        assert should_close is False

    def test_close_on_reverse_signal_up_to_down(self, base_config, sample_signals):
        """Test closing when signal reverses from UP to DOWN."""
        position = {
            'symbol': 'MSFT',
            'side': 'UP',  # Position is UP
            'alert_level': 'GREEN'
        }

        # MSFT has DOWN signal in sample_signals
        should_close, reason = should_close_position(position, sample_signals, base_config)

        assert should_close is True
        assert reason == 'REVERSE_SIGNAL'

    def test_close_on_reverse_signal_down_to_up(self, base_config, sample_signals):
        """Test closing when signal reverses from DOWN to UP."""
        position = {
            'symbol': 'AAPL',
            'side': 'DOWN',  # Position is DOWN
            'alert_level': 'GREEN'
        }

        # AAPL has UP signal in sample_signals
        should_close, reason = should_close_position(position, sample_signals, base_config)

        assert should_close is True
        assert reason == 'REVERSE_SIGNAL'

    def test_no_close_on_matching_signal(self, base_config, sample_signals):
        """Test no closing when signal matches position."""
        position = {
            'symbol': 'AAPL',
            'side': 'UP',  # Matches signal
            'alert_level': 'GREEN'
        }

        should_close, reason = should_close_position(position, sample_signals, base_config)

        assert should_close is False

    def test_close_on_abstain_when_enabled(self, base_config, sample_signals):
        """Test closing on ABSTAIN signal when policy enables it."""
        base_config['close_policy']['close_on_abstain'] = True

        position = {
            'symbol': 'GOOGL',
            'side': 'UP',
            'alert_level': 'GREEN'
        }

        should_close, reason = should_close_position(position, sample_signals, base_config)

        assert should_close is True
        assert reason == 'ABSTAIN_SIGNAL'

    def test_no_close_on_abstain_when_disabled(self, base_config, sample_signals):
        """Test no closing on ABSTAIN when policy disables it."""
        base_config['close_policy']['close_on_abstain'] = False

        position = {
            'symbol': 'GOOGL',
            'side': 'UP',
            'alert_level': 'GREEN'
        }

        should_close, reason = should_close_position(position, sample_signals, base_config)

        assert should_close is False

    def test_close_disabled_in_policy(self, base_config, sample_signals):
        """Test no closing when policy disables it."""
        base_config['close_policy']['close_on_red'] = False
        base_config['close_policy']['close_on_reverse'] = False

        position = {
            'symbol': 'AAPL',
            'side': 'DOWN',
            'alert_level': 'RED'  # RED but policy disabled
        }

        should_close, reason = should_close_position(position, sample_signals, base_config)

        assert should_close is False


class TestGetPrices:
    """Tests for get_prices function."""

    def test_get_prices_positive_returns(self, sample_returns):
        """Test price computation with positive returns."""
        entry_date = '2024-01-01'
        exit_date = '2024-01-25'  # After 20 up days

        entry_price, exit_price = get_prices('AAPL', entry_date, exit_date, sample_returns)

        # Entry price is 100
        assert entry_price == 100.0

        # With 1% daily returns for ~20 days, exit price should be > 100
        assert exit_price > entry_price

    def test_get_prices_negative_returns(self, sample_returns):
        """Test price computation with negative returns."""
        entry_date = '2024-01-25'
        exit_date = '2024-02-20'  # During down period

        entry_price, exit_price = get_prices('AAPL', entry_date, exit_date, sample_returns)

        # Entry price is 100
        assert entry_price == 100.0

        # With negative returns, exit price should be < 100
        assert exit_price < entry_price

    def test_get_prices_missing_symbol(self, sample_returns):
        """Test price computation with missing symbol."""
        entry_date = '2024-01-01'
        exit_date = '2024-01-25'

        entry_price, exit_price = get_prices('MISSING', entry_date, exit_date, sample_returns)

        # Should return default prices
        assert entry_price == 100.0
        assert exit_price == 100.0


class TestComputePnl:
    """Tests for compute_pnl function."""

    def test_compute_pnl_long_profit(self, base_config):
        """Test PnL computation for profitable long position."""
        position = {'side': 'UP'}
        entry_price = 100.0
        exit_price = 110.0
        notional = 10000.0

        gross_pnl, net_pnl = compute_pnl(position, entry_price, exit_price, notional, base_config)

        # Gross PnL: 10% gain on $10,000 = $1,000
        expected_gross = 1000.0
        assert gross_pnl == pytest.approx(expected_gross, rel=0.01)

        # Net PnL: Gross - costs
        # Costs: (5 + 2) * 2 = 14 bps = 0.14% = $14
        expected_net = expected_gross - 14.0
        assert net_pnl == pytest.approx(expected_net, rel=0.01)

    def test_compute_pnl_long_loss(self, base_config):
        """Test PnL computation for losing long position."""
        position = {'side': 'UP'}
        entry_price = 100.0
        exit_price = 90.0
        notional = 10000.0

        gross_pnl, net_pnl = compute_pnl(position, entry_price, exit_price, notional, base_config)

        # Gross PnL: -10% on $10,000 = -$1,000
        expected_gross = -1000.0
        assert gross_pnl == pytest.approx(expected_gross, rel=0.01)

        # Net PnL: Gross - costs (costs make it worse)
        expected_net = expected_gross - 14.0
        assert net_pnl == pytest.approx(expected_net, rel=0.01)

    def test_compute_pnl_short_profit(self, base_config):
        """Test PnL computation for profitable short position."""
        position = {'side': 'DOWN'}
        entry_price = 100.0
        exit_price = 90.0  # Price declined
        notional = 10000.0

        gross_pnl, net_pnl = compute_pnl(position, entry_price, exit_price, notional, base_config)

        # Gross PnL: Short profits from decline: +10% on $10,000 = $1,000
        expected_gross = 1000.0
        assert gross_pnl == pytest.approx(expected_gross, rel=0.01)

        # Net PnL: Gross - costs
        expected_net = expected_gross - 14.0
        assert net_pnl == pytest.approx(expected_net, rel=0.01)

    def test_compute_pnl_short_loss(self, base_config):
        """Test PnL computation for losing short position."""
        position = {'side': 'DOWN'}
        entry_price = 100.0
        exit_price = 110.0  # Price increased (bad for short)
        notional = 10000.0

        gross_pnl, net_pnl = compute_pnl(position, entry_price, exit_price, notional, base_config)

        # Gross PnL: Short loses from increase: -10% on $10,000 = -$1,000
        expected_gross = -1000.0
        assert gross_pnl == pytest.approx(expected_gross, rel=0.01)

        # Net PnL: Gross - costs
        expected_net = expected_gross - 14.0
        assert net_pnl == pytest.approx(expected_net, rel=0.01)


class TestAppendToLedger:
    """Tests for append_to_ledger function."""

    def test_append_to_new_ledger(self, tmp_path):
        """Test appending to new (non-existent) ledger."""
        ledger_file = tmp_path / 'portfolio_ledger.csv'

        trade = {
            'symbol': 'AAPL',
            'entry_date': '2024-01-15',
            'exit_date': '2024-01-30',
            'side': 'UP',
            'entry_price': 100.0,
            'exit_price': 110.0,
            'notional': 10000.0,
            'gross_pnl': 1000.0,
            'net_pnl': 986.0,
            'reason': 'RED_ALERT'
        }

        append_to_ledger(trade, ledger_file)

        # Verify file created
        assert ledger_file.exists()

        # Verify contents
        ledger_df = pd.read_csv(ledger_file)
        assert len(ledger_df) == 1
        assert ledger_df.iloc[0]['symbol'] == 'AAPL'
        assert ledger_df.iloc[0]['gross_pnl'] == 1000.0

    def test_append_to_existing_ledger(self, tmp_path):
        """Test appending to existing ledger."""
        ledger_file = tmp_path / 'portfolio_ledger.csv'

        # Create initial ledger
        trade1 = {
            'symbol': 'AAPL',
            'entry_date': '2024-01-15',
            'exit_date': '2024-01-30',
            'side': 'UP',
            'entry_price': 100.0,
            'exit_price': 110.0,
            'notional': 10000.0,
            'gross_pnl': 1000.0,
            'net_pnl': 986.0,
            'reason': 'RED_ALERT'
        }

        append_to_ledger(trade1, ledger_file)

        # Append second trade
        trade2 = {
            'symbol': 'MSFT',
            'entry_date': '2024-01-16',
            'exit_date': '2024-01-30',
            'side': 'DOWN',
            'entry_price': 200.0,
            'exit_price': 190.0,
            'notional': 10000.0,
            'gross_pnl': 500.0,
            'net_pnl': 486.0,
            'reason': 'REVERSE_SIGNAL'
        }

        append_to_ledger(trade2, ledger_file)

        # Verify ledger has 2 trades
        ledger_df = pd.read_csv(ledger_file)
        assert len(ledger_df) == 2
        assert ledger_df.iloc[0]['symbol'] == 'AAPL'
        assert ledger_df.iloc[1]['symbol'] == 'MSFT'


class TestClosePositionsIntegration:
    """Integration tests for close_positions function."""

    @patch('scripts.close_positions.load_positions_state')
    @patch('scripts.close_positions.save_positions_state')
    def test_close_positions_red_alert(
        self,
        mock_save_state,
        mock_load_state,
        tmp_path,
        monkeypatch,
        base_config,
        sample_returns
    ):
        """Test closing positions with RED alert."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create data directory
        (tmp_path / 'data' / 'processed').mkdir(parents=True)
        sample_returns.to_parquet(tmp_path / 'data' / 'processed' / 'returns.parquet')

        # Create position with RED alert
        position = {
            'symbol': 'AAPL',
            'entry_date': '2024-01-15',
            'side': 'UP',
            'notional': 10000.0,
            'alert_level': 'RED',
            'p_up': 0.75,
            'confidence': 0.25
        }

        mock_load_state.return_value = {'open_positions': [position]}

        # Create signals file
        (tmp_path / 'results' / 'live').mkdir(parents=True)
        signals_df = pd.DataFrame([{'symbol': 'AAPL', 'signal': 'UP', 'p_up': 0.8}])
        signals_df.to_csv(tmp_path / 'results' / 'live' / 'signals_2024-01-30.csv', index=False)

        # Run closing
        close_positions(base_config, close_date='2024-01-30')

        # Verify ledger created
        ledger_file = tmp_path / 'results' / 'live' / 'portfolio_ledger.csv'
        assert ledger_file.exists()

        # Verify trade recorded
        ledger_df = pd.read_csv(ledger_file)
        assert len(ledger_df) == 1
        assert ledger_df.iloc[0]['symbol'] == 'AAPL'
        assert ledger_df.iloc[0]['reason'] == 'RED_ALERT'

        # Verify state updated (empty positions)
        mock_save_state.assert_called_once()
        saved_state = mock_save_state.call_args[0][0]
        assert len(saved_state['open_positions']) == 0

    @patch('scripts.close_positions.load_positions_state')
    @patch('scripts.close_positions.save_positions_state')
    def test_close_positions_reverse_signal(
        self,
        mock_save_state,
        mock_load_state,
        tmp_path,
        monkeypatch,
        base_config,
        sample_returns
    ):
        """Test closing positions with reverse signal."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create data directory
        (tmp_path / 'data' / 'processed').mkdir(parents=True)
        sample_returns.to_parquet(tmp_path / 'data' / 'processed' / 'returns.parquet')

        # Create UP position
        position = {
            'symbol': 'AAPL',
            'entry_date': '2024-01-15',
            'side': 'UP',
            'notional': 10000.0,
            'alert_level': 'GREEN',
            'p_up': 0.75,
            'confidence': 0.25
        }

        mock_load_state.return_value = {'open_positions': [position]}

        # Create signals file with DOWN signal (reverse)
        (tmp_path / 'results' / 'live').mkdir(parents=True)
        signals_df = pd.DataFrame([{'symbol': 'AAPL', 'signal': 'DOWN', 'p_up': 0.2}])
        signals_df.to_csv(tmp_path / 'results' / 'live' / 'signals_2024-01-30.csv', index=False)

        # Run closing
        close_positions(base_config, close_date='2024-01-30')

        # Verify trade recorded with reverse signal reason
        ledger_file = tmp_path / 'results' / 'live' / 'portfolio_ledger.csv'
        ledger_df = pd.read_csv(ledger_file)
        assert len(ledger_df) == 1
        assert ledger_df.iloc[0]['reason'] == 'REVERSE_SIGNAL'

    @patch('scripts.close_positions.load_positions_state')
    @patch('scripts.close_positions.save_positions_state')
    def test_close_positions_no_close(
        self,
        mock_save_state,
        mock_load_state,
        tmp_path,
        monkeypatch,
        base_config,
        sample_returns
    ):
        """Test when no positions meet closing criteria."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create data directory
        (tmp_path / 'data' / 'processed').mkdir(parents=True)
        sample_returns.to_parquet(tmp_path / 'data' / 'processed' / 'returns.parquet')

        # Create position with GREEN alert and matching signal
        position = {
            'symbol': 'AAPL',
            'entry_date': '2024-01-15',
            'side': 'UP',
            'notional': 10000.0,
            'alert_level': 'GREEN',
            'p_up': 0.75,
            'confidence': 0.25
        }

        mock_load_state.return_value = {'open_positions': [position]}

        # Create signals file with UP signal (matches position)
        (tmp_path / 'results' / 'live').mkdir(parents=True)
        signals_df = pd.DataFrame([{'symbol': 'AAPL', 'signal': 'UP', 'p_up': 0.8}])
        signals_df.to_csv(tmp_path / 'results' / 'live' / 'signals_2024-01-30.csv', index=False)

        # Run closing
        close_positions(base_config, close_date='2024-01-30')

        # Verify no ledger created (no trades closed)
        ledger_file = tmp_path / 'results' / 'live' / 'portfolio_ledger.csv'
        assert not ledger_file.exists()

        # Verify state saved with same position
        mock_save_state.assert_not_called()  # Not called when no changes
