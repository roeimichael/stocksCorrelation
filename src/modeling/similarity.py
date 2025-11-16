"""Similarity computation and analog ranking for pattern matching."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.core.logger import get_logger


logger = get_logger(__name__)


def sim_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient between two vectors.

    Args:
        a: First vector (returns or normalized features)
        b: Second vector (must be same length as a)

    Returns:
        Pearson correlation coefficient in [-1, 1]

    Notes:
        - Measures linear relationship
        - +1 = perfect positive correlation
        - -1 = perfect negative correlation
        - 0 = no linear relationship
        - Returns 0.0 if either vector has zero variance
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError(f"Vectors must have same length: {len(a)} != {len(b)}")

    # Check for zero variance
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0

    try:
        corr, _ = pearsonr(a, b)
        # Handle NaN (can occur with perfect correlation of zero-variance after filtering)
        if np.isnan(corr):
            return 0.0
        return float(corr)
    except Exception as e:
        logger.warning(f"Error computing Pearson correlation: {e}")
        return 0.0


def sim_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Spearman rank correlation coefficient between two vectors.

    Args:
        a: First vector (returns or normalized features)
        b: Second vector (must be same length as a)

    Returns:
        Spearman correlation coefficient in [-1, 1]

    Notes:
        - Measures monotonic relationship (rank-based)
        - More robust to outliers than Pearson
        - +1 = perfect monotonic increasing
        - -1 = perfect monotonic decreasing
        - Returns 0.0 if either vector has all equal values
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError(f"Vectors must have same length: {len(a)} != {len(b)}")

    # Check for constant values (all ranks equal)
    if len(np.unique(a)) == 1 or len(np.unique(b)) == 1:
        return 0.0

    try:
        corr, _ = spearmanr(a, b)
        if np.isnan(corr):
            return 0.0
        return float(corr)
    except Exception as e:
        logger.warning(f"Error computing Spearman correlation: {e}")
        return 0.0


def sim_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector (returns or normalized features)
        b: Second vector (must be same length as a)

    Returns:
        Cosine similarity in [-1, 1]

    Formula:
        cos(θ) = (a · b) / (||a|| * ||b||)

    Notes:
        - Measures angle between vectors (direction similarity)
        - +1 = same direction
        - -1 = opposite direction
        - 0 = orthogonal (no similarity)
        - Returns 0.0 if either vector is zero
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError(f"Vectors must have same length: {len(a)} != {len(b)}")

    # Compute norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Check for zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Compute cosine similarity
    similarity = np.dot(a, b) / (norm_a * norm_b)

    # Clamp to [-1, 1] to handle numerical errors
    similarity = np.clip(similarity, -1.0, 1.0)

    return float(similarity)


def compute_similarity(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    """
    Compute similarity between two vectors using specified metric.

    Args:
        a: First vector
        b: Second vector (must be same length as a)
        metric: Similarity metric - "pearson", "spearman", or "cosine"

    Returns:
        Similarity score in [-1, 1]

    Raises:
        ValueError: If metric is unknown or vectors have different lengths
    """
    if metric == 'pearson':
        return sim_pearson(a, b)
    if metric == 'spearman':
        return sim_spearman(a, b)
    if metric == 'cosine':
        return sim_cosine(a, b)
    raise ValueError(f"Unknown metric: {metric}. Use 'pearson', 'spearman', or 'cosine'.")


def rank_analogs(
    target_vec: np.ndarray,
    bank_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    metric: str,
    top_k: int,
    min_sim: float = 0.0,
    exclude_symbol: str | None = None
) -> pd.DataFrame:
    """
    Find and rank top-K most similar analogs from window bank.

    Args:
        target_vec: Target window features (normalized returns of length X)
        bank_df: DataFrame with columns [symbol, start_date, end_date, features, label]
        cutoff_date: Only consider windows with end_date <= cutoff_date (prevent leakage)
        metric: Similarity metric - "pearson", "spearman", or "cosine"
        top_k: Number of top analogs to return
        min_sim: Minimum similarity threshold (default: 0.0)
        exclude_symbol: Optional symbol to exclude from candidates (e.g., target's own symbol)

    Returns:
        DataFrame with columns [symbol, end_date, sim, label] sorted by sim descending,
        containing at most top_k rows

    Process:
        1. Filter windows: end_date <= cutoff_date (no look-ahead bias)
        2. Optionally exclude_symbol (avoid matching to same stock)
        3. Compute similarity for each candidate
        4. Keep only candidates with sim >= min_sim
        5. Sort by similarity descending
        6. Take top K

    Notes:
        - Returns empty DataFrame if no candidates meet criteria
        - Logs warning if fewer than top_k analogs found
    """
    target_vec = np.asarray(target_vec, dtype=np.float64)

    # Filter by cutoff_date (no look-ahead bias)
    candidates = bank_df[bank_df['end_date'] <= cutoff_date].copy()

    logger.debug(f"Candidates after cutoff_date filter: {len(candidates)}")

    # Optionally exclude symbol
    if exclude_symbol is not None:
        candidates = candidates[candidates['symbol'] != exclude_symbol].copy()
        logger.debug(f"Candidates after excluding {exclude_symbol}: {len(candidates)}")

    if len(candidates) == 0:
        logger.warning(f"No candidates found for ranking (cutoff_date={cutoff_date}, exclude_symbol={exclude_symbol})")
        return pd.DataFrame(columns=['symbol', 'end_date', 'sim', 'label'])

    # Compute similarity for each candidate
    similarities = []
    for _idx, row in candidates.iterrows():
        candidate_vec = row['features']
        try:
            sim = compute_similarity(target_vec, candidate_vec, metric)
            similarities.append(sim)
        except Exception as e:
            logger.warning(f"Error computing similarity for {row['symbol']} at {row['end_date']}: {e}")
            similarities.append(0.0)

    candidates['sim'] = similarities

    # Filter by minimum similarity
    candidates = candidates[candidates['sim'] >= min_sim].copy()

    logger.debug(f"Candidates after min_sim={min_sim} filter: {len(candidates)}")

    if len(candidates) == 0:
        logger.warning(f"No candidates with sim >= {min_sim}")
        return pd.DataFrame(columns=['symbol', 'end_date', 'sim', 'label'])

    # Sort by similarity descending
    candidates = candidates.sort_values('sim', ascending=False)

    # Take top K
    top_analogs = candidates.head(top_k)

    if len(top_analogs) < top_k:
        logger.info(f"Found {len(top_analogs)} analogs (requested {top_k})")

    # Return only required columns
    result = top_analogs[['symbol', 'end_date', 'sim', 'label']].copy()

    return result
