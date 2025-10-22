"""Voting logic for analog-based predictions."""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from src.modeling.windows import Window
from src.core.logger import get_logger

logger = get_logger()


@dataclass
class Signal:
    """Trading signal with confidence."""
    direction: str  # 'UP', 'DOWN', or 'ABSTAIN'
    confidence: float  # Probability or vote share
    num_analogs: int  # Number of valid analogs used
    avg_similarity: float  # Average similarity of analogs


def majority_vote(
    analogs: List[Tuple[Window, float]],
    threshold: float = 0.70,
    min_analogs: int = 10
) -> Signal:
    """
    Simple majority vote on next-day direction.

    Args:
        analogs: List of (window, similarity) tuples
        threshold: Minimum vote share to emit signal
        min_analogs: Minimum number of valid analogs required

    Returns:
        Signal object
    """
    # Filter out analogs with missing labels
    valid_analogs = [(w, sim) for w, sim in analogs if w.label != -1]

    num_valid = len(valid_analogs)

    if num_valid < min_analogs:
        return Signal(
            direction='ABSTAIN',
            confidence=0.0,
            num_analogs=num_valid,
            avg_similarity=0.0
        )

    # Count up and down
    up_count = sum(1 for w, _ in valid_analogs if w.label == 1)
    down_count = num_valid - up_count

    # Compute vote shares
    p_up = up_count / num_valid
    p_down = down_count / num_valid

    # Average similarity
    avg_sim = np.mean([sim for _, sim in valid_analogs])

    # Make decision
    if p_up >= threshold:
        direction = 'UP'
        confidence = p_up
    elif p_down >= threshold:
        direction = 'DOWN'
        confidence = p_down
    else:
        direction = 'ABSTAIN'
        confidence = max(p_up, p_down)

    return Signal(
        direction=direction,
        confidence=confidence,
        num_analogs=num_valid,
        avg_similarity=avg_sim
    )


def similarity_weighted_vote(
    analogs: List[Tuple[Window, float]],
    threshold: float = 0.70,
    min_analogs: int = 10
) -> Signal:
    """
    Similarity-weighted vote on next-day direction.
    More similar analogs get higher weight.

    Args:
        analogs: List of (window, similarity) tuples
        threshold: Minimum weighted vote share to emit signal
        min_analogs: Minimum number of valid analogs required

    Returns:
        Signal object
    """
    # Filter out analogs with missing labels
    valid_analogs = [(w, sim) for w, sim in analogs if w.label != -1]

    num_valid = len(valid_analogs)

    if num_valid < min_analogs:
        return Signal(
            direction='ABSTAIN',
            confidence=0.0,
            num_analogs=num_valid,
            avg_similarity=0.0
        )

    # Ensure non-negative weights (shift if needed)
    similarities = np.array([sim for _, sim in valid_analogs])
    min_sim = similarities.min()
    if min_sim < 0:
        weights = similarities - min_sim + 1e-6  # Shift to positive
    else:
        weights = similarities + 1e-6  # Small epsilon to avoid division by zero

    # Weighted vote
    total_weight = weights.sum()
    up_weight = sum(w * (window.label == 1) for (window, _), w in zip(valid_analogs, weights))
    down_weight = total_weight - up_weight

    # Compute weighted shares
    p_up = up_weight / total_weight
    p_down = down_weight / total_weight

    # Average similarity
    avg_sim = similarities.mean()

    # Make decision
    if p_up >= threshold:
        direction = 'UP'
        confidence = p_up
    elif p_down >= threshold:
        direction = 'DOWN'
        confidence = p_down
    else:
        direction = 'ABSTAIN'
        confidence = max(p_up, p_down)

    return Signal(
        direction=direction,
        confidence=confidence,
        num_analogs=num_valid,
        avg_similarity=avg_sim
    )


def generate_signal(
    analogs: List[Tuple[Window, float]],
    config: dict
) -> Signal:
    """
    Generate signal based on configuration.

    Args:
        analogs: List of (window, similarity) tuples
        config: Configuration dictionary

    Returns:
        Signal object
    """
    scheme = config['vote']['scheme']
    threshold = config['vote']['threshold']
    min_analogs = config['vote'].get('abstain_if_below_k', 10)

    if scheme == 'majority':
        return majority_vote(analogs, threshold, min_analogs)
    elif scheme == 'similarity_weighted':
        return similarity_weighted_vote(analogs, threshold, min_analogs)
    else:
        raise ValueError(f"Unknown vote scheme: {scheme}")


def batch_generate_signals(
    all_analogs: List[List[Tuple[Window, float]]],
    config: dict
) -> List[Signal]:
    """
    Generate signals for multiple sets of analogs.

    Args:
        all_analogs: List of analog lists
        config: Configuration dictionary

    Returns:
        List of signals
    """
    logger.info(f"Generating {len(all_analogs)} signals")

    signals = []
    for analogs in all_analogs:
        signal = generate_signal(analogs, config)
        signals.append(signal)

    # Log summary
    directions = [s.direction for s in signals]
    up_count = sum(1 for d in directions if d == 'UP')
    down_count = sum(1 for d in directions if d == 'DOWN')
    abstain_count = sum(1 for d in directions if d == 'ABSTAIN')

    logger.info(f"Signal distribution: UP={up_count}, DOWN={down_count}, ABSTAIN={abstain_count}")

    return signals


def signal_to_position(
    signal: Signal,
    side: str = 'both'
) -> int:
    """
    Convert signal to position (-1 = short, 0 = no position, 1 = long).

    Args:
        signal: Signal object
        side: Trading side ('long', 'short', 'both')

    Returns:
        Position: -1, 0, or 1
    """
    if signal.direction == 'ABSTAIN':
        return 0

    if signal.direction == 'UP':
        if side in ['long', 'both']:
            return 1
        else:
            return 0

    if signal.direction == 'DOWN':
        if side in ['short', 'both']:
            return -1
        else:
            return 0

    return 0


def filter_top_signals(
    signals: List[Signal],
    target_windows: List[Window],
    max_positions: int
) -> List[Tuple[Window, Signal]]:
    """
    Filter to top N signals by confidence.

    Args:
        signals: List of signals
        target_windows: Corresponding target windows
        max_positions: Maximum number of positions

    Returns:
        List of (window, signal) tuples for top positions
    """
    # Pair signals with windows
    paired = list(zip(target_windows, signals))

    # Filter out abstain signals
    active = [(w, s) for w, s in paired if s.direction != 'ABSTAIN']

    # Sort by confidence (descending)
    active.sort(key=lambda x: x[1].confidence, reverse=True)

    # Take top N
    top_signals = active[:max_positions]

    logger.info(f"Filtered to top {len(top_signals)} signals from {len(active)} active")

    return top_signals
