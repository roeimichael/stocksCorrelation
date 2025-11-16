"""Tests for similarity computation and analog ranking."""
import numpy as np
import pandas as pd
import pytest

from src.modeling.similarity import compute_similarity, rank_analogs, sim_cosine, sim_pearson, sim_spearman


@pytest.fixture
def mini_bank():
    """
    Create mini window bank with 3 windows for testing.

    Windows:
    - STOCK_A: end_date=2024-01-01, uptrend [1,2,3,4,5], label=1
    - STOCK_B: end_date=2024-01-02, downtrend [5,4,3,2,1], label=0
    - STOCK_C: end_date=2024-01-03, uptrend [1,2,3,4,5], label=1
    """
    dates = pd.date_range('2024-01-01', periods=3, freq='B')

    return pd.DataFrame({
        'symbol': ['STOCK_A', 'STOCK_B', 'STOCK_C'],
        'start_date': [dates[0] - pd.Timedelta(days=4)] * 3,
        'end_date': [dates[0], dates[1], dates[2]],
        'features': [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # Uptrend
            np.array([5.0, 4.0, 3.0, 2.0, 1.0]),  # Downtrend (opposite)
            np.array([1.0, 2.0, 3.0, 4.0, 5.0])   # Identical to A
        ],
        'label': [1, 0, 1]
    })


class TestSimilarityMetrics:
    """Tests for individual similarity metrics."""

    def test_pearson_identical_vectors(self):
        """Test Pearson correlation with identical vectors returns ~1.0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        sim = sim_pearson(a, b)

        assert 0.99 < sim <= 1.0, f"Identical vectors should have sim ~1.0, got {sim}"

    def test_pearson_opposite_vectors(self):
        """Test Pearson correlation with opposite vectors returns ~-1.0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        sim = sim_pearson(a, b)

        assert -1.0 <= sim < -0.99, f"Opposite vectors should have sim ~-1.0, got {sim}"

    def test_pearson_zero_variance(self):
        """Test Pearson handles zero variance (constant values)."""
        a = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        sim = sim_pearson(a, b)

        assert sim == 0.0, f"Zero variance should return 0.0, got {sim}"

    def test_pearson_length_mismatch_raises_error(self):
        """Test Pearson raises error for vectors of different lengths."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="Vectors must have same length"):
            sim_pearson(a, b)

    def test_spearman_identical_vectors(self):
        """Test Spearman correlation with identical vectors returns ~1.0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        sim = sim_spearman(a, b)

        assert 0.99 < sim <= 1.0, f"Identical vectors should have sim ~1.0, got {sim}"

    def test_spearman_opposite_vectors(self):
        """Test Spearman correlation with opposite vectors returns ~-1.0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        sim = sim_spearman(a, b)

        assert -1.0 <= sim < -0.99, f"Opposite vectors should have sim ~-1.0, got {sim}"

    def test_spearman_constant_values(self):
        """Test Spearman handles constant values (all ranks equal)."""
        a = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        sim = sim_spearman(a, b)

        assert sim == 0.0, f"Constant values should return 0.0, got {sim}"

    def test_cosine_identical_vectors(self):
        """Test cosine similarity with identical vectors returns ~1.0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        sim = sim_cosine(a, b)

        assert 0.99 < sim <= 1.0, f"Identical vectors should have sim ~1.0, got {sim}"

    def test_cosine_opposite_vectors(self):
        """Test cosine similarity with opposite vectors returns ~-1.0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])

        sim = sim_cosine(a, b)

        assert -1.0 <= sim < -0.99, f"Opposite vectors should have sim ~-1.0, got {sim}"

    def test_cosine_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors returns ~0.0."""
        # Create orthogonal vectors
        a = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

        sim = sim_cosine(a, b)

        assert np.abs(sim) < 0.01, f"Orthogonal vectors should have sim ~0.0, got {sim}"

    def test_cosine_zero_vector(self):
        """Test cosine handles zero vector."""
        a = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        sim = sim_cosine(a, b)

        assert sim == 0.0, f"Zero vector should return 0.0, got {sim}"


class TestComputeSimilarity:
    """Tests for compute_similarity unified interface."""

    def test_compute_similarity_pearson(self):
        """Test compute_similarity dispatches to sim_pearson."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        sim = compute_similarity(a, b, metric='pearson')

        assert 0.99 < sim <= 1.0

    def test_compute_similarity_spearman(self):
        """Test compute_similarity dispatches to sim_spearman."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        sim = compute_similarity(a, b, metric='spearman')

        assert 0.99 < sim <= 1.0

    def test_compute_similarity_cosine(self):
        """Test compute_similarity dispatches to sim_cosine."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        sim = compute_similarity(a, b, metric='cosine')

        assert 0.99 < sim <= 1.0

    def test_compute_similarity_unknown_metric_raises_error(self):
        """Test compute_similarity raises error for unknown metric."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="Unknown metric"):
            compute_similarity(a, b, metric='unknown')


class TestRankAnalogs:
    """Tests for rank_analogs function."""

    def test_rank_analogs_ordering(self, mini_bank):
        """Test rank_analogs returns results sorted by similarity descending."""
        # Target vector identical to STOCK_A and STOCK_C (uptrend)
        target_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # No cutoff_date filter, include all windows
        cutoff_date = pd.Timestamp('2024-01-10')

        # Use min_sim=-1.0 to include STOCK_B (negative correlation)
        result = rank_analogs(
            target_vec=target_vec,
            bank_df=mini_bank,
            cutoff_date=cutoff_date,
            metric='pearson',
            top_k=3,
            min_sim=-1.0
        )

        # Should return 3 results
        assert len(result) == 3, f"Expected 3 results, got {len(result)}"

        # First two should be STOCK_A and STOCK_C (identical, sim ~1.0)
        # STOCK_B should be last (opposite, sim ~-1.0)
        assert result.iloc[0]['symbol'] in ['STOCK_A', 'STOCK_C']
        assert result.iloc[1]['symbol'] in ['STOCK_A', 'STOCK_C']
        assert result.iloc[2]['symbol'] == 'STOCK_B'

        # Check similarity values are descending
        sims = result['sim'].values
        assert np.all(sims[:-1] >= sims[1:]), f"Similarities not descending: {sims}"

    def test_rank_analogs_min_sim_filtering(self, mini_bank):
        """Test rank_analogs filters by min_sim threshold."""
        # Target vector identical to STOCK_A and STOCK_C (uptrend)
        target_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        cutoff_date = pd.Timestamp('2024-01-10')

        # Set min_sim = 0.5, should exclude STOCK_B (sim ~-1.0)
        result = rank_analogs(
            target_vec=target_vec,
            bank_df=mini_bank,
            cutoff_date=cutoff_date,
            metric='pearson',
            top_k=10,
            min_sim=0.5
        )

        # Should only return STOCK_A and STOCK_C
        assert len(result) == 2, f"Expected 2 results with min_sim=0.5, got {len(result)}"
        assert 'STOCK_B' not in result['symbol'].values

        # All similarities should be >= 0.5
        assert (result['sim'] >= 0.5).all(), f"Some similarities below min_sim: {result['sim'].values}"

    def test_rank_analogs_cutoff_date_filter(self, mini_bank):
        """Test rank_analogs filters by cutoff_date (no look-ahead bias)."""
        target_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Set cutoff_date to 2024-01-01, should only include STOCK_A
        cutoff_date = pd.Timestamp('2024-01-01')

        result = rank_analogs(
            target_vec=target_vec,
            bank_df=mini_bank,
            cutoff_date=cutoff_date,
            metric='pearson',
            top_k=10,
            min_sim=0.0
        )

        # Should only return STOCK_A (end_date = 2024-01-01)
        assert len(result) == 1, f"Expected 1 result with cutoff_date=2024-01-01, got {len(result)}"
        assert result.iloc[0]['symbol'] == 'STOCK_A'

        # Verify no windows with end_date > cutoff_date
        assert (result['end_date'] <= cutoff_date).all(), "Some windows have end_date > cutoff_date"

    def test_rank_analogs_exclude_symbol(self, mini_bank):
        """Test rank_analogs excludes specified symbol."""
        target_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cutoff_date = pd.Timestamp('2024-01-10')

        # Exclude STOCK_A, use min_sim=-1.0 to include STOCK_B
        result = rank_analogs(
            target_vec=target_vec,
            bank_df=mini_bank,
            cutoff_date=cutoff_date,
            metric='pearson',
            top_k=10,
            min_sim=-1.0,
            exclude_symbol='STOCK_A'
        )

        # Should return STOCK_B and STOCK_C only
        assert len(result) == 2, f"Expected 2 results after excluding STOCK_A, got {len(result)}"
        assert 'STOCK_A' not in result['symbol'].values

    def test_rank_analogs_top_k_limit(self, mini_bank):
        """Test rank_analogs returns at most top_k results."""
        target_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cutoff_date = pd.Timestamp('2024-01-10')

        # Request only top 2
        result = rank_analogs(
            target_vec=target_vec,
            bank_df=mini_bank,
            cutoff_date=cutoff_date,
            metric='pearson',
            top_k=2,
            min_sim=0.0
        )

        # Should return exactly 2 results
        assert len(result) == 2, f"Expected 2 results with top_k=2, got {len(result)}"

        # Should be the two most similar (STOCK_A and STOCK_C)
        assert result.iloc[0]['symbol'] in ['STOCK_A', 'STOCK_C']
        assert result.iloc[1]['symbol'] in ['STOCK_A', 'STOCK_C']

    def test_rank_analogs_returns_correct_columns(self, mini_bank):
        """Test rank_analogs returns DataFrame with correct columns."""
        target_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cutoff_date = pd.Timestamp('2024-01-10')

        result = rank_analogs(
            target_vec=target_vec,
            bank_df=mini_bank,
            cutoff_date=cutoff_date,
            metric='pearson',
            top_k=3,
            min_sim=-1.0
        )

        # Check columns
        expected_columns = ['symbol', 'end_date', 'sim', 'label']
        assert list(result.columns) == expected_columns, f"Expected columns {expected_columns}, got {list(result.columns)}"

        # Check data types
        assert result['symbol'].dtype == object
        assert pd.api.types.is_datetime64_any_dtype(result['end_date'])
        assert result['sim'].dtype == float
        # Check label is integer type (int, int32, int64, etc.)
        assert np.issubdtype(result['label'].dtype, np.integer), f"Expected integer dtype, got {result['label'].dtype}"

    def test_rank_analogs_empty_result_no_candidates(self, mini_bank):
        """Test rank_analogs returns empty DataFrame when no candidates meet criteria."""
        target_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Set cutoff_date before any windows
        cutoff_date = pd.Timestamp('2023-12-31')

        result = rank_analogs(
            target_vec=target_vec,
            bank_df=mini_bank,
            cutoff_date=cutoff_date,
            metric='pearson',
            top_k=10,
            min_sim=0.0
        )

        # Should return empty DataFrame with correct columns
        assert len(result) == 0, f"Expected empty result, got {len(result)} rows"
        assert list(result.columns) == ['symbol', 'end_date', 'sim', 'label']

    def test_rank_analogs_empty_result_min_sim_too_high(self, mini_bank):
        """Test rank_analogs returns empty DataFrame when min_sim is too high."""
        target_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cutoff_date = pd.Timestamp('2024-01-10')

        # Set min_sim impossibly high
        result = rank_analogs(
            target_vec=target_vec,
            bank_df=mini_bank,
            cutoff_date=cutoff_date,
            metric='pearson',
            top_k=10,
            min_sim=1.5  # Impossible (sim is in [-1, 1])
        )

        # Should return empty DataFrame
        assert len(result) == 0, f"Expected empty result with min_sim=1.5, got {len(result)} rows"


class TestIntegration:
    """Integration tests for similarity module."""

    def test_full_pipeline_all_metrics(self, mini_bank):
        """Test rank_analogs works with all three metrics."""
        target_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cutoff_date = pd.Timestamp('2024-01-10')

        for metric in ['pearson', 'spearman', 'cosine']:
            result = rank_analogs(
                target_vec=target_vec,
                bank_df=mini_bank,
                cutoff_date=cutoff_date,
                metric=metric,
                top_k=3,
                min_sim=-1.0  # Include negative similarities
            )

            # Should return 3 results
            assert len(result) == 3, f"Expected 3 results for {metric}, got {len(result)}"

            # Should have correct columns
            assert list(result.columns) == ['symbol', 'end_date', 'sim', 'label']

            # Similarities should be descending
            sims = result['sim'].values
            assert np.all(sims[:-1] >= sims[1:]), f"{metric}: similarities not descending"
