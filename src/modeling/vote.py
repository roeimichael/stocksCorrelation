"""Voting logic for analog-based predictions."""
from typing import Any

import pandas as pd

from src.core.logger import get_logger


logger = get_logger(__name__)


def vote(
    analogs_df: pd.DataFrame,
    scheme: str,
    threshold: float,
    abstain_if_below_k: int
) -> dict[str, Any]:
    """
    Aggregate analog labels to produce a trading signal.

    Args:
        analogs_df: DataFrame with columns [symbol, end_date, sim, label]
                    Typically output from rank_analogs()
        scheme: Voting scheme - "majority" or "similarity_weighted"
        threshold: Probability threshold for UP/DOWN signal (e.g., 0.70 = 70%)
        abstain_if_below_k: Minimum number of analogs required (abstain if fewer)

    Returns:
        Dictionary with:
            - p_up: float, probability of upward movement [0, 1]
            - signal: str, trading signal - "UP", "DOWN", or "ABSTAIN"
            - confidence: float, abs(p_up - 0.5) - distance from neutral
            - n_analogs: int, number of analogs used in vote

    Voting Schemes:
        - "majority": Simple mean of labels (unweighted)
          p_up = mean(labels) where label in {0, 1}

        - "similarity_weighted": Weighted by similarity scores
          p_up = sum(sim * label) / sum(sim)

    Decision Logic:
        1. If n_analogs < abstain_if_below_k -> signal = "ABSTAIN"
        2. Else if p_up >= threshold -> signal = "UP"
        3. Else if (1 - p_up) >= threshold -> signal = "DOWN"
        4. Else -> signal = "ABSTAIN"

    Notes:
        - threshold should typically be in [0.5, 1.0] for meaningful signals
        - confidence = 0.0 means p_up = 0.5 (no conviction)
        - confidence = 0.5 means p_up = 0.0 or 1.0 (maximum conviction)
        - Empty DataFrame returns p_up=0.5, signal="ABSTAIN", confidence=0.0

    Example:
        >>> analogs = rank_analogs(target_vec, bank_df, cutoff_date, 'pearson', top_k=25)
        >>> result = vote(analogs, scheme='similarity_weighted', threshold=0.70, abstain_if_below_k=10)
        >>> print(result)
        {'p_up': 0.75, 'signal': 'UP', 'confidence': 0.25, 'n_analogs': 25}
    """
    n_analogs = len(analogs_df)

    # Check if we have enough analogs
    if n_analogs < abstain_if_below_k:
        logger.debug(f"Insufficient analogs: {n_analogs} < {abstain_if_below_k}, abstaining")
        return {
            'p_up': 0.5,
            'signal': 'ABSTAIN',
            'confidence': 0.0,
            'n_analogs': n_analogs
        }

    # Handle empty DataFrame
    if n_analogs == 0:
        logger.warning("Empty analogs DataFrame, abstaining")
        return {
            'p_up': 0.5,
            'signal': 'ABSTAIN',
            'confidence': 0.0,
            'n_analogs': 0
        }

    # Compute p_up based on scheme
    if scheme == 'majority':
        # Simple mean of labels (unweighted)
        p_up = analogs_df['label'].mean()

    elif scheme == 'similarity_weighted':
        # Weighted by similarity scores
        sims = analogs_df['sim'].values
        labels = analogs_df['label'].values

        # Compute weighted average: sum(sim * label) / sum(sim)
        total_sim = sims.sum()

        if total_sim == 0:
            # If all similarities are zero, fall back to majority
            logger.warning("All similarities are zero, falling back to majority vote")
            p_up = labels.mean()
        else:
            weighted_sum = (sims * labels).sum()
            p_up = weighted_sum / total_sim

    else:
        raise ValueError(f"Unknown voting scheme: {scheme}. Use 'majority' or 'similarity_weighted'.")

    # Ensure p_up is in [0, 1]
    p_up = float(p_up)
    p_up = max(0.0, min(1.0, p_up))

    # Compute confidence (distance from neutral 0.5)
    confidence = abs(p_up - 0.5)

    # Determine signal based on threshold
    if p_up >= threshold:
        signal = 'UP'
    elif (1.0 - p_up) >= threshold:
        signal = 'DOWN'
    else:
        signal = 'ABSTAIN'

    logger.debug(f"Vote result: p_up={p_up:.3f}, signal={signal}, confidence={confidence:.3f}, n_analogs={n_analogs}")

    return {
        'p_up': p_up,
        'signal': signal,
        'confidence': confidence,
        'n_analogs': n_analogs
    }
