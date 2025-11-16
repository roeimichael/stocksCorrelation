"""Tests for voting logic."""
import numpy as np
import pandas as pd
import pytest

from src.modeling.vote import vote


@pytest.fixture
def analogs_all_up():
    """Create analogs DataFrame where all labels are UP (1)."""
    return pd.DataFrame({
        'symbol': ['STOCK_A', 'STOCK_B', 'STOCK_C'],
        'end_date': pd.date_range('2024-01-01', periods=3, freq='B'),
        'sim': [0.9, 0.8, 0.7],
        'label': [1, 1, 1]
    })


@pytest.fixture
def analogs_all_down():
    """Create analogs DataFrame where all labels are DOWN (0)."""
    return pd.DataFrame({
        'symbol': ['STOCK_A', 'STOCK_B', 'STOCK_C'],
        'end_date': pd.date_range('2024-01-01', periods=3, freq='B'),
        'sim': [0.9, 0.8, 0.7],
        'label': [0, 0, 0]
    })


@pytest.fixture
def analogs_mixed():
    """Create analogs DataFrame with mixed labels (2 UP, 1 DOWN)."""
    return pd.DataFrame({
        'symbol': ['STOCK_A', 'STOCK_B', 'STOCK_C'],
        'end_date': pd.date_range('2024-01-01', periods=3, freq='B'),
        'sim': [0.9, 0.8, 0.7],
        'label': [1, 1, 0]
    })


@pytest.fixture
def analogs_weighted_scenario():
    """
    Create analogs for testing weighted voting.

    Scenario:
    - 2 UP (labels=1) with low similarity (0.3, 0.2)
    - 1 DOWN (label=0) with high similarity (0.9)

    Majority vote: 2/3 = 0.667 (UP)
    Weighted vote: (0.3*1 + 0.2*1 + 0.9*0) / (0.3+0.2+0.9) = 0.5/1.4 = 0.357 (DOWN)
    """
    return pd.DataFrame({
        'symbol': ['STOCK_A', 'STOCK_B', 'STOCK_C'],
        'end_date': pd.date_range('2024-01-01', periods=3, freq='B'),
        'sim': [0.3, 0.2, 0.9],
        'label': [1, 1, 0]
    })


class TestMajorityVote:
    """Tests for majority voting scheme."""

    def test_majority_all_up(self, analogs_all_up):
        """Test majority vote when all analogs are UP."""
        result = vote(
            analogs_df=analogs_all_up,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=2
        )

        # All UP means p_up = 1.0
        assert result['p_up'] == 1.0, f"Expected p_up=1.0, got {result['p_up']}"
        assert result['signal'] == 'UP', f"Expected signal='UP', got {result['signal']}"
        assert result['confidence'] == 0.5, f"Expected confidence=0.5, got {result['confidence']}"
        assert result['n_analogs'] == 3

    def test_majority_all_down(self, analogs_all_down):
        """Test majority vote when all analogs are DOWN."""
        result = vote(
            analogs_df=analogs_all_down,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=2
        )

        # All DOWN means p_up = 0.0
        assert result['p_up'] == 0.0, f"Expected p_up=0.0, got {result['p_up']}"
        assert result['signal'] == 'DOWN', f"Expected signal='DOWN', got {result['signal']}"
        assert result['confidence'] == 0.5, f"Expected confidence=0.5, got {result['confidence']}"
        assert result['n_analogs'] == 3

    def test_majority_mixed(self, analogs_mixed):
        """Test majority vote with mixed labels (2 UP, 1 DOWN)."""
        result = vote(
            analogs_df=analogs_mixed,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=2
        )

        # 2 UP, 1 DOWN -> p_up = 2/3 = 0.6667
        expected_p_up = 2.0 / 3.0
        assert np.isclose(result['p_up'], expected_p_up), f"Expected p_up={expected_p_up}, got {result['p_up']}"

        # p_up = 0.6667 < 0.70, and (1-p_up) = 0.3333 < 0.70 -> ABSTAIN
        assert result['signal'] == 'ABSTAIN', f"Expected signal='ABSTAIN', got {result['signal']}"

        # Confidence = abs(0.6667 - 0.5) = 0.1667
        expected_confidence = abs(expected_p_up - 0.5)
        assert np.isclose(result['confidence'], expected_confidence), f"Expected confidence={expected_confidence}, got {result['confidence']}"
        assert result['n_analogs'] == 3

    def test_majority_exactly_threshold(self):
        """Test majority vote when p_up exactly equals threshold."""
        # Create scenario where p_up = 0.70 exactly (7 UP, 3 DOWN)
        analogs = pd.DataFrame({
            'symbol': [f'STOCK_{i}' for i in range(10)],
            'end_date': pd.date_range('2024-01-01', periods=10, freq='B'),
            'sim': [0.9] * 10,
            'label': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  # 7 UP, 3 DOWN
        })

        result = vote(
            analogs_df=analogs,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=5
        )

        # p_up = 7/10 = 0.70 exactly
        assert result['p_up'] == 0.70, f"Expected p_up=0.70, got {result['p_up']}"

        # p_up >= 0.70 -> UP
        assert result['signal'] == 'UP', f"Expected signal='UP', got {result['signal']}"

    def test_majority_exactly_threshold_down(self):
        """Test majority vote when (1-p_up) exactly equals threshold."""
        # Create scenario where p_up = 0.30 (1-p_up = 0.70) exactly
        analogs = pd.DataFrame({
            'symbol': [f'STOCK_{i}' for i in range(10)],
            'end_date': pd.date_range('2024-01-01', periods=10, freq='B'),
            'sim': [0.9] * 10,
            'label': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  # 3 UP, 7 DOWN
        })

        result = vote(
            analogs_df=analogs,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=5
        )

        # p_up = 3/10 = 0.30
        assert result['p_up'] == 0.30, f"Expected p_up=0.30, got {result['p_up']}"

        # (1 - p_up) = 0.70 >= 0.70 -> DOWN
        assert result['signal'] == 'DOWN', f"Expected signal='DOWN', got {result['signal']}"


class TestSimilarityWeightedVote:
    """Tests for similarity-weighted voting scheme."""

    def test_weighted_all_up(self, analogs_all_up):
        """Test weighted vote when all analogs are UP."""
        result = vote(
            analogs_df=analogs_all_up,
            scheme='similarity_weighted',
            threshold=0.70,
            abstain_if_below_k=2
        )

        # All UP means p_up = 1.0 regardless of weights
        assert result['p_up'] == 1.0, f"Expected p_up=1.0, got {result['p_up']}"
        assert result['signal'] == 'UP'
        assert result['confidence'] == 0.5
        assert result['n_analogs'] == 3

    def test_weighted_all_down(self, analogs_all_down):
        """Test weighted vote when all analogs are DOWN."""
        result = vote(
            analogs_df=analogs_all_down,
            scheme='similarity_weighted',
            threshold=0.70,
            abstain_if_below_k=2
        )

        # All DOWN means p_up = 0.0 regardless of weights
        assert result['p_up'] == 0.0, f"Expected p_up=0.0, got {result['p_up']}"
        assert result['signal'] == 'DOWN'
        assert result['confidence'] == 0.5
        assert result['n_analogs'] == 3

    def test_weighted_vs_majority(self, analogs_weighted_scenario):
        """Test that weighted vote differs from majority vote in expected scenario."""
        # Majority vote
        result_majority = vote(
            analogs_df=analogs_weighted_scenario,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=2
        )

        # Weighted vote
        result_weighted = vote(
            analogs_df=analogs_weighted_scenario,
            scheme='similarity_weighted',
            threshold=0.70,
            abstain_if_below_k=2
        )

        # Majority: 2 UP, 1 DOWN -> p_up = 2/3 = 0.6667
        expected_majority_p_up = 2.0 / 3.0
        assert np.isclose(result_majority['p_up'], expected_majority_p_up)

        # Weighted: (0.3*1 + 0.2*1 + 0.9*0) / (0.3+0.2+0.9) = 0.5/1.4 = 0.357
        expected_weighted_p_up = (0.3 * 1 + 0.2 * 1 + 0.9 * 0) / (0.3 + 0.2 + 0.9)
        assert np.isclose(result_weighted['p_up'], expected_weighted_p_up, atol=0.001), \
            f"Expected weighted p_up={expected_weighted_p_up}, got {result_weighted['p_up']}"

        # They should be different
        assert not np.isclose(result_majority['p_up'], result_weighted['p_up'])

    def test_weighted_computation(self):
        """Test weighted vote computation with known values."""
        # Create controlled scenario:
        # sim=[0.5, 0.3, 0.2], label=[1, 0, 1]
        # Weighted: (0.5*1 + 0.3*0 + 0.2*1) / (0.5+0.3+0.2) = 0.7 / 1.0 = 0.70
        analogs = pd.DataFrame({
            'symbol': ['A', 'B', 'C'],
            'end_date': pd.date_range('2024-01-01', periods=3, freq='B'),
            'sim': [0.5, 0.3, 0.2],
            'label': [1, 0, 1]
        })

        result = vote(
            analogs_df=analogs,
            scheme='similarity_weighted',
            threshold=0.70,
            abstain_if_below_k=2
        )

        # p_up = 0.70 exactly
        assert np.isclose(result['p_up'], 0.70, atol=0.001), f"Expected p_up=0.70, got {result['p_up']}"

        # p_up >= 0.70 -> UP
        assert result['signal'] == 'UP'

    def test_weighted_zero_similarities_fallback(self):
        """Test weighted vote falls back to majority when all similarities are zero."""
        analogs = pd.DataFrame({
            'symbol': ['A', 'B', 'C'],
            'end_date': pd.date_range('2024-01-01', periods=3, freq='B'),
            'sim': [0.0, 0.0, 0.0],
            'label': [1, 1, 0]
        })

        result = vote(
            analogs_df=analogs,
            scheme='similarity_weighted',
            threshold=0.70,
            abstain_if_below_k=2
        )

        # Should fall back to majority: 2/3 = 0.6667
        expected_p_up = 2.0 / 3.0
        assert np.isclose(result['p_up'], expected_p_up), f"Expected p_up={expected_p_up}, got {result['p_up']}"


class TestAbstain:
    """Tests for abstention logic."""

    def test_abstain_below_minimum_analogs(self, analogs_mixed):
        """Test abstain when n_analogs < abstain_if_below_k."""
        result = vote(
            analogs_df=analogs_mixed,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=10  # Require 10 but only have 3
        )

        assert result['signal'] == 'ABSTAIN', f"Expected signal='ABSTAIN', got {result['signal']}"
        assert result['p_up'] == 0.5, f"Expected p_up=0.5, got {result['p_up']}"
        assert result['confidence'] == 0.0, f"Expected confidence=0.0, got {result['confidence']}"
        assert result['n_analogs'] == 3

    def test_abstain_empty_dataframe(self):
        """Test abstain with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['symbol', 'end_date', 'sim', 'label'])

        result = vote(
            analogs_df=empty_df,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=2
        )

        assert result['signal'] == 'ABSTAIN'
        assert result['p_up'] == 0.5
        assert result['confidence'] == 0.0
        assert result['n_analogs'] == 0

    def test_abstain_exactly_minimum_analogs(self):
        """Test that exactly abstain_if_below_k analogs does NOT abstain."""
        analogs = pd.DataFrame({
            'symbol': ['A', 'B', 'C'],
            'end_date': pd.date_range('2024-01-01', periods=3, freq='B'),
            'sim': [0.9, 0.8, 0.7],
            'label': [1, 1, 1]
        })

        result = vote(
            analogs_df=analogs,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=3  # Exactly 3 analogs
        )

        # Should NOT abstain (have exactly the minimum)
        assert result['signal'] == 'UP', f"Expected signal='UP', got {result['signal']}"
        assert result['p_up'] == 1.0

    def test_abstain_one_below_minimum(self):
        """Test that abstain_if_below_k-1 analogs DOES abstain."""
        analogs = pd.DataFrame({
            'symbol': ['A', 'B'],
            'end_date': pd.date_range('2024-01-01', periods=2, freq='B'),
            'sim': [0.9, 0.8],
            'label': [1, 1]
        })

        result = vote(
            analogs_df=analogs,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=3  # Need 3, have 2
        )

        # Should abstain (one below minimum)
        assert result['signal'] == 'ABSTAIN'
        assert result['p_up'] == 0.5
        assert result['n_analogs'] == 2


class TestThresholdEdges:
    """Tests for threshold edge cases."""

    def test_threshold_edges_up_just_above(self):
        """Test signal when p_up is just above threshold."""
        # p_up = 0.71, threshold = 0.70
        analogs = pd.DataFrame({
            'symbol': [f'STOCK_{i}' for i in range(100)],
            'end_date': pd.date_range('2024-01-01', periods=100, freq='B'),
            'sim': [0.9] * 100,
            'label': [1] * 71 + [0] * 29  # 71 UP, 29 DOWN
        })

        result = vote(
            analogs_df=analogs,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=10
        )

        assert result['p_up'] == 0.71
        assert result['signal'] == 'UP'

    def test_threshold_edges_up_just_below(self):
        """Test signal when p_up is just below threshold."""
        # p_up = 0.69, threshold = 0.70
        analogs = pd.DataFrame({
            'symbol': [f'STOCK_{i}' for i in range(100)],
            'end_date': pd.date_range('2024-01-01', periods=100, freq='B'),
            'sim': [0.9] * 100,
            'label': [1] * 69 + [0] * 31  # 69 UP, 31 DOWN
        })

        result = vote(
            analogs_df=analogs,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=10
        )

        assert result['p_up'] == 0.69
        # p_up = 0.69 < 0.70, and (1-p_up) = 0.31 < 0.70 -> ABSTAIN
        assert result['signal'] == 'ABSTAIN'

    def test_threshold_edges_down_just_above(self):
        """Test signal when (1-p_up) is just above threshold."""
        # p_up = 0.29, (1-p_up) = 0.71, threshold = 0.70
        analogs = pd.DataFrame({
            'symbol': [f'STOCK_{i}' for i in range(100)],
            'end_date': pd.date_range('2024-01-01', periods=100, freq='B'),
            'sim': [0.9] * 100,
            'label': [1] * 29 + [0] * 71  # 29 UP, 71 DOWN
        })

        result = vote(
            analogs_df=analogs,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=10
        )

        assert result['p_up'] == 0.29
        assert result['signal'] == 'DOWN'

    def test_threshold_edges_down_just_below(self):
        """Test signal when (1-p_up) is just below threshold."""
        # p_up = 0.31, (1-p_up) = 0.69, threshold = 0.70
        analogs = pd.DataFrame({
            'symbol': [f'STOCK_{i}' for i in range(100)],
            'end_date': pd.date_range('2024-01-01', periods=100, freq='B'),
            'sim': [0.9] * 100,
            'label': [1] * 31 + [0] * 69  # 31 UP, 69 DOWN
        })

        result = vote(
            analogs_df=analogs,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=10
        )

        assert result['p_up'] == 0.31
        # p_up = 0.31 < 0.70, and (1-p_up) = 0.69 < 0.70 -> ABSTAIN
        assert result['signal'] == 'ABSTAIN'

    def test_threshold_50_50_split(self):
        """Test signal with 50/50 split (p_up = 0.5)."""
        analogs = pd.DataFrame({
            'symbol': ['A', 'B', 'C', 'D'],
            'end_date': pd.date_range('2024-01-01', periods=4, freq='B'),
            'sim': [0.9] * 4,
            'label': [1, 1, 0, 0]  # 50/50 split
        })

        result = vote(
            analogs_df=analogs,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=2
        )

        assert result['p_up'] == 0.5
        assert result['signal'] == 'ABSTAIN'
        assert result['confidence'] == 0.0  # No conviction


class TestConfidence:
    """Tests for confidence calculation."""

    def test_confidence_maximum(self):
        """Test confidence is 0.5 when p_up is 0.0 or 1.0."""
        # p_up = 1.0
        analogs_up = pd.DataFrame({
            'symbol': ['A'],
            'end_date': pd.date_range('2024-01-01', periods=1, freq='B'),
            'sim': [0.9],
            'label': [1]
        })

        result_up = vote(analogs_up, 'majority', 0.70, 1)
        assert result_up['confidence'] == 0.5

        # p_up = 0.0
        analogs_down = pd.DataFrame({
            'symbol': ['A'],
            'end_date': pd.date_range('2024-01-01', periods=1, freq='B'),
            'sim': [0.9],
            'label': [0]
        })

        result_down = vote(analogs_down, 'majority', 0.70, 1)
        assert result_down['confidence'] == 0.5

    def test_confidence_minimum(self):
        """Test confidence is 0.0 when p_up is 0.5."""
        analogs = pd.DataFrame({
            'symbol': ['A', 'B'],
            'end_date': pd.date_range('2024-01-01', periods=2, freq='B'),
            'sim': [0.9, 0.8],
            'label': [1, 0]  # 50/50
        })

        result = vote(analogs, 'majority', 0.70, 1)
        assert result['confidence'] == 0.0

    def test_confidence_calculation(self):
        """Test confidence is correctly calculated as abs(p_up - 0.5)."""
        # p_up = 0.75 -> confidence = 0.25
        analogs = pd.DataFrame({
            'symbol': ['A', 'B', 'C', 'D'],
            'end_date': pd.date_range('2024-01-01', periods=4, freq='B'),
            'sim': [0.9] * 4,
            'label': [1, 1, 1, 0]  # 3/4 = 0.75
        })

        result = vote(analogs, 'majority', 0.70, 1)
        assert result['p_up'] == 0.75
        assert result['confidence'] == 0.25


class TestErrorHandling:
    """Tests for error handling."""

    def test_unknown_scheme_raises_error(self, analogs_mixed):
        """Test that unknown voting scheme raises ValueError."""
        with pytest.raises(ValueError, match="Unknown voting scheme"):
            vote(analogs_mixed, scheme='unknown', threshold=0.70, abstain_if_below_k=2)

    def test_result_dictionary_keys(self, analogs_mixed):
        """Test that result dictionary has all required keys."""
        result = vote(analogs_mixed, 'majority', 0.70, 2)

        # Check all required keys are present
        assert 'p_up' in result
        assert 'signal' in result
        assert 'confidence' in result
        assert 'n_analogs' in result

        # Check types
        assert isinstance(result['p_up'], float)
        assert isinstance(result['signal'], str)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['n_analogs'], int)


class TestIntegration:
    """Integration tests for voting module."""

    def test_full_pipeline_majority(self):
        """Test full pipeline with majority vote."""
        # Create realistic scenario
        analogs = pd.DataFrame({
            'symbol': [f'STOCK_{i}' for i in range(25)],
            'end_date': pd.date_range('2024-01-01', periods=25, freq='B'),
            'sim': np.linspace(0.9, 0.5, 25),  # Descending similarities
            'label': [1] * 18 + [0] * 7  # 18 UP, 7 DOWN -> 72% UP
        })

        result = vote(
            analogs_df=analogs,
            scheme='majority',
            threshold=0.70,
            abstain_if_below_k=10
        )

        # 18/25 = 0.72 > 0.70 -> UP
        assert result['p_up'] == 0.72
        assert result['signal'] == 'UP'
        assert result['n_analogs'] == 25

    def test_full_pipeline_weighted(self):
        """Test full pipeline with weighted vote."""
        # Create scenario where majority and weighted differ
        analogs = pd.DataFrame({
            'symbol': ['A', 'B', 'C', 'D', 'E'],
            'end_date': pd.date_range('2024-01-01', periods=5, freq='B'),
            'sim': [0.9, 0.8, 0.3, 0.2, 0.1],
            'label': [1, 1, 0, 0, 0]  # 2 UP (high sim), 3 DOWN (low sim)
        })

        result = vote(
            analogs_df=analogs,
            scheme='similarity_weighted',
            threshold=0.70,
            abstain_if_below_k=3
        )

        # Weighted: (0.9*1 + 0.8*1 + 0.3*0 + 0.2*0 + 0.1*0) / (0.9+0.8+0.3+0.2+0.1)
        # = 1.7 / 2.3 = 0.739
        expected_p_up = (0.9 + 0.8) / (0.9 + 0.8 + 0.3 + 0.2 + 0.1)
        assert np.isclose(result['p_up'], expected_p_up, atol=0.001)
        assert result['signal'] == 'UP'
