"""Similarity computation and analog retrieval."""
import numpy as np
from typing import List, Tuple
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
from src.modeling.windows import Window
from src.core.logger import get_logger

logger = get_logger()


def compute_similarity(
    target: np.ndarray,
    candidate: np.ndarray,
    metric: str = 'pearson'
) -> float:
    """
    Compute similarity between two feature vectors.

    Args:
        target: Target window features
        candidate: Candidate window features
        metric: Similarity metric ('pearson', 'spearman', 'cosine')

    Returns:
        Similarity score
    """
    if len(target) != len(candidate):
        raise ValueError("Feature vectors must have same length")

    if metric == 'pearson':
        # Pearson correlation
        corr, _ = pearsonr(target, candidate)
        return corr if not np.isnan(corr) else 0.0

    elif metric == 'spearman':
        # Spearman rank correlation
        corr, _ = spearmanr(target, candidate)
        return corr if not np.isnan(corr) else 0.0

    elif metric == 'cosine':
        # Cosine similarity (1 - cosine distance)
        dist = cosine(target, candidate)
        return 1.0 - dist if not np.isnan(dist) else 0.0

    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


def find_top_analogs(
    target_window: Window,
    candidate_windows: List[Window],
    top_k: int,
    metric: str = 'pearson',
    min_similarity: float = -1.0
) -> List[Tuple[Window, float]]:
    """
    Find the top K most similar windows to target.

    Args:
        target_window: Target window to match
        candidate_windows: Pool of candidate windows
        top_k: Number of top analogs to return
        metric: Similarity metric
        min_similarity: Minimum similarity threshold

    Returns:
        List of (window, similarity) tuples, sorted by similarity (descending)
    """
    similarities = []

    for candidate in candidate_windows:
        # Skip if same window (shouldn't happen with proper date filtering)
        if (candidate.symbol == target_window.symbol and
            candidate.end_date == target_window.end_date):
            continue

        sim = compute_similarity(
            target_window.features,
            candidate.features,
            metric=metric
        )

        # Apply minimum similarity filter
        if sim >= min_similarity:
            similarities.append((candidate, sim))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Take top K
    top_analogs = similarities[:top_k]

    return top_analogs


def batch_find_analogs(
    target_windows: List[Window],
    candidate_pool: List[Window],
    top_k: int,
    metric: str = 'pearson',
    min_similarity: float = -1.0
) -> List[List[Tuple[Window, float]]]:
    """
    Find top analogs for multiple target windows.

    Args:
        target_windows: List of target windows
        candidate_pool: Pool of candidate windows
        top_k: Number of analogs per target
        metric: Similarity metric
        min_similarity: Minimum similarity threshold

    Returns:
        List of analog lists (one per target window)
    """
    logger.info(f"Finding top {top_k} analogs for {len(target_windows)} targets")

    all_analogs = []

    for i, target in enumerate(target_windows):
        # Filter candidates to avoid look-ahead bias
        valid_candidates = [
            c for c in candidate_pool
            if c.end_date < target.end_date  # Strictly before target
        ]

        analogs = find_top_analogs(
            target,
            valid_candidates,
            top_k=top_k,
            metric=metric,
            min_similarity=min_similarity
        )

        all_analogs.append(analogs)

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(target_windows)} targets")

    logger.info("Analog search complete")

    return all_analogs


def get_analog_statistics(
    analogs: List[Tuple[Window, float]]
) -> dict:
    """
    Compute statistics about a set of analogs.

    Args:
        analogs: List of (window, similarity) tuples

    Returns:
        Dictionary with statistics
    """
    if not analogs:
        return {
            'count': 0,
            'avg_similarity': 0.0,
            'min_similarity': 0.0,
            'max_similarity': 0.0,
            'up_count': 0,
            'down_count': 0,
            'missing_count': 0
        }

    similarities = [sim for _, sim in analogs]
    labels = [w.label for w, _ in analogs]

    stats = {
        'count': len(analogs),
        'avg_similarity': np.mean(similarities),
        'min_similarity': np.min(similarities),
        'max_similarity': np.max(similarities),
        'up_count': sum(1 for l in labels if l == 1),
        'down_count': sum(1 for l in labels if l == 0),
        'missing_count': sum(1 for l in labels if l == -1)
    }

    return stats


def compute_similarity_matrix(
    windows: List[Window],
    metric: str = 'pearson',
    sample_size: int = 1000
) -> np.ndarray:
    """
    Compute pairwise similarity matrix for a sample of windows.
    Useful for EDA and understanding similarity distributions.

    Args:
        windows: List of windows
        metric: Similarity metric
        sample_size: Maximum number of windows to sample

    Returns:
        Similarity matrix (sample_size x sample_size)
    """
    # Sample if too many windows
    if len(windows) > sample_size:
        indices = np.random.choice(len(windows), sample_size, replace=False)
        sampled_windows = [windows[i] for i in indices]
    else:
        sampled_windows = windows

    n = len(sampled_windows)
    sim_matrix = np.zeros((n, n))

    logger.info(f"Computing similarity matrix for {n} windows")

    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sim = compute_similarity(
                    sampled_windows[i].features,
                    sampled_windows[j].features,
                    metric=metric
                )
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim  # Symmetric

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{n} rows")

    return sim_matrix
