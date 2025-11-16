"""
Hedging utilities for portfolio risk management.

This module provides functions to construct hedge baskets using negatively
correlated securities and manage hedge sizing/rebalancing.
"""
from typing import Any

import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.modeling.similarity import compute_similarity


logger = get_logger(__name__)


def select_hedge_basket(
    target_vec: np.ndarray,
    bank_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    cfg: dict[str, Any],
    exclude_symbol: str = None
) -> list[tuple[str, float]]:
    """
    Find K securities with lowest/negative similarity for hedging.

    Args:
        target_vec: Target window features (normalized returns)
        bank_df: DataFrame with windows [symbol, start_date, end_date, features, label]
        cutoff_date: Only consider windows with end_date <= cutoff_date
        cfg: Configuration dictionary with hedge section
        exclude_symbol: Optional symbol to exclude (typically the long position)

    Returns:
        List of (symbol, raw_weight) tuples representing the hedge basket.
        Weights are based on |negative_similarity| - higher absolute negative sim = higher weight.

    Process:
        1. Filter windows by cutoff_date
        2. Compute similarity for each candidate
        3. Keep only negative similarities (sim < 0)
        4. Aggregate by symbol: take max(|negative_sim|) per symbol
        5. Select top K symbols by |negative_sim|
        6. Return as list of (symbol, |negative_sim|) tuples

    Example:
        If AAPL is our long position, we want securities that move opposite to AAPL's pattern.
        Securities with sim = -0.8 are stronger hedges than sim = -0.2.
    """
    hedge_config = cfg.get('hedge', {})
    top_k = hedge_config.get('basket_size', 10)
    min_neg_sim = hedge_config.get('min_neg_sim', -0.1)  # Minimum negative similarity to consider
    metric = cfg['similarity']['metric']

    target_vec = np.asarray(target_vec, dtype=np.float64)

    # Filter by cutoff_date
    candidates = bank_df[bank_df['end_date'] <= cutoff_date].copy()

    logger.debug(f"Hedge basket: {len(candidates)} candidates after cutoff_date filter")

    # Optionally exclude symbol
    if exclude_symbol is not None:
        candidates = candidates[candidates['symbol'] != exclude_symbol].copy()
        logger.debug(f"Hedge basket: {len(candidates)} candidates after excluding {exclude_symbol}")

    if len(candidates) == 0:
        logger.warning("No candidates for hedge basket")
        return []

    # Compute similarity for each candidate
    similarities = []
    for _idx, row in candidates.iterrows():
        candidate_vec = row['features']
        try:
            sim = compute_similarity(target_vec, candidate_vec, metric)
            similarities.append(sim)
        except Exception as e:
            logger.warning(f"Error computing similarity for {row['symbol']}: {e}")
            similarities.append(0.0)

    candidates['sim'] = similarities

    # Keep only negative similarities
    neg_candidates = candidates[candidates['sim'] < 0].copy()

    logger.debug(f"Hedge basket: {len(neg_candidates)} candidates with negative similarity")

    if len(neg_candidates) == 0:
        logger.warning("No candidates with negative similarity for hedge basket")
        return []

    # Filter by minimum negative similarity threshold
    neg_candidates = neg_candidates[neg_candidates['sim'] <= min_neg_sim].copy()

    logger.debug(f"Hedge basket: {len(neg_candidates)} candidates after min_neg_sim filter")

    if len(neg_candidates) == 0:
        logger.warning(f"No candidates with sim <= {min_neg_sim}")
        return []

    # Compute absolute negative similarity
    neg_candidates['abs_neg_sim'] = neg_candidates['sim'].abs()

    # Aggregate by symbol: take max |negative_sim| per symbol
    symbol_hedges = neg_candidates.groupby('symbol')['abs_neg_sim'].max().reset_index()

    # Sort by |negative_sim| descending
    symbol_hedges = symbol_hedges.sort_values('abs_neg_sim', ascending=False)

    # Take top K
    top_hedges = symbol_hedges.head(top_k)

    logger.info(f"Selected {len(top_hedges)} hedge symbols from {len(symbol_hedges)} candidates")

    # Return as list of (symbol, weight) tuples
    hedge_basket = [(row['symbol'], row['abs_neg_sim']) for _, row in top_hedges.iterrows()]

    return hedge_basket


def size_hedge(
    long_notional: float,
    vol_long: float,
    vol_basket: float,
    cfg: dict[str, Any]
) -> float:
    """
    Compute hedge notional based on volatility ratio.

    Args:
        long_notional: Notional value of long position ($)
        vol_long: Volatility (std dev) of long position returns
        vol_basket: Volatility (std dev) of hedge basket returns
        cfg: Configuration dictionary with hedge section

    Returns:
        Hedge notional in dollars.

    Formula:
        hedge_notional = target_ratio * (vol_long / max(vol_basket, eps)) * long_notional

    Where:
        - target_ratio: Desired hedge ratio (default 1.0 = fully hedged)
        - eps: Small value to prevent division by zero

    Example:
        If long position is $10,000 with vol=0.02, and basket has vol=0.01:
        hedge_notional = 1.0 * (0.02 / 0.01) * 10000 = $20,000
        (Need 2x the long position to match volatility exposure)
    """
    hedge_config = cfg.get('hedge', {})
    target_ratio = hedge_config.get('target_ratio', 1.0)
    eps = hedge_config.get('vol_eps', 1e-6)

    # Prevent division by zero
    vol_basket_safe = max(vol_basket, eps)

    # Compute hedge notional
    hedge_notional = target_ratio * (vol_long / vol_basket_safe) * long_notional

    return float(hedge_notional)


def needs_rebalance(
    position: dict[str, Any],
    current_date: pd.Timestamp,
    returns_df: pd.DataFrame,
    cfg: dict[str, Any]
) -> bool:
    """
    Determine if a hedge position needs rebalancing.

    Args:
        position: Position dict with hedge info
        current_date: Current date
        returns_df: DataFrame with returns
        cfg: Configuration dictionary with hedge section

    Returns:
        True if rebalance is needed, False otherwise.

    Rebalance triggers:
        1. Time-based: Every hedge.rebalance_days days
        2. Signal-based: Correlation sign flips (positive -> negative or vice versa)

    Process:
        - Check days since last rebalance
        - Check current correlation vs. entry correlation
        - Return True if any trigger condition is met
    """
    hedge_config = cfg.get('hedge', {})
    rebalance_days = hedge_config.get('rebalance_days', 5)

    # Check if position has hedge info
    hedge_info = position.get('hedge', {})
    if not hedge_info:
        return False

    # Get last rebalance date (or entry date)
    last_rebalance = pd.Timestamp(hedge_info.get('last_rebalance_date', position.get('entry_date')))

    # Check time-based trigger
    days_since_rebalance = (current_date - last_rebalance).days
    if days_since_rebalance >= rebalance_days:
        logger.info(f"Rebalance needed for {position['symbol']}: {days_since_rebalance} days since last rebalance")
        return True

    # Check correlation sign flip
    entry_corr = hedge_info.get('entry_correlation', 0.0)

    # Compute current correlation
    symbol = position['symbol']
    hedge_symbols = [h['symbol'] for h in hedge_info.get('basket', [])]

    if not hedge_symbols or symbol not in returns_df.columns:
        return False

    # Get returns for symbol and hedge basket
    available_hedges = [s for s in hedge_symbols if s in returns_df.columns]
    if not available_hedges:
        return False

    # Get recent returns (last 20 days)
    corr_window = cfg.get('monitor', {}).get('corr_window_days', 20)

    try:
        symbol_returns = returns_df.loc[:current_date, symbol].tail(corr_window)
        hedge_returns = returns_df.loc[:current_date, available_hedges].tail(corr_window)

        # Compute basket return (equal weighted for simplicity)
        basket_return = hedge_returns.mean(axis=1)

        # Compute correlation
        current_corr = symbol_returns.corr(basket_return)

        # Check for sign flip
        if not pd.isna(current_corr) and not pd.isna(entry_corr) and (
            (entry_corr >= 0 and current_corr < 0) or (entry_corr < 0 and current_corr >= 0)
        ):
            logger.info(f"Rebalance needed for {position['symbol']}: correlation sign flip ({entry_corr:.3f} -> {current_corr:.3f})")
            return True

    except Exception as e:
        logger.warning(f"Error checking rebalance condition: {e}")
        return False

    return False


def compute_hedge_volatilities(
    symbol: str,
    hedge_symbols: list[str],
    returns_df: pd.DataFrame,
    current_date: pd.Timestamp,
    cfg: dict[str, Any]
) -> tuple[float, float]:
    """
    Compute volatilities for position and hedge basket.

    Args:
        symbol: Long position symbol
        hedge_symbols: List of hedge basket symbols
        returns_df: DataFrame with returns
        current_date: Current date
        cfg: Configuration dictionary

    Returns:
        Tuple of (vol_long, vol_basket)

    Process:
        1. Get recent returns (last N days) for symbol and hedge basket
        2. Compute standard deviation of returns
        3. Return as volatilities
    """
    vol_window = cfg.get('hedge', {}).get('vol_window_days', 20)

    # Get returns for symbol
    if symbol not in returns_df.columns:
        logger.warning(f"Symbol {symbol} not found in returns_df")
        return 0.0, 0.0

    symbol_returns = returns_df.loc[:current_date, symbol].tail(vol_window)

    # Get returns for hedge basket
    available_hedges = [s for s in hedge_symbols if s in returns_df.columns]
    if not available_hedges:
        logger.warning("No hedge symbols found in returns_df")
        return symbol_returns.std(), 0.0

    hedge_returns = returns_df.loc[:current_date, available_hedges].tail(vol_window)

    # Compute basket return (equal weighted)
    basket_return = hedge_returns.mean(axis=1)

    # Compute volatilities
    vol_long = symbol_returns.std()
    vol_basket = basket_return.std()

    return float(vol_long), float(vol_basket)


def create_hedge_info(
    target_vec: np.ndarray,
    symbol: str,
    long_notional: float,
    bank_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    current_date: pd.Timestamp,
    cfg: dict[str, Any]
) -> dict[str, Any]:
    """
    Create hedge information for a new position.

    Args:
        target_vec: Target window features
        symbol: Long position symbol
        long_notional: Notional value of long position
        bank_df: Windows bank
        returns_df: Returns DataFrame
        current_date: Current date
        cfg: Configuration dictionary

    Returns:
        Dictionary with hedge information:
        {
            'basket': [{'symbol': str, 'raw_weight': float}, ...],
            'notional': float,
            'entry_correlation': float,
            'last_rebalance_date': str,
            'vol_long': float,
            'vol_basket': float
        }
    """
    # Select hedge basket
    hedge_basket = select_hedge_basket(
        target_vec=target_vec,
        bank_df=bank_df,
        cutoff_date=current_date - pd.Timedelta(days=1),
        cfg=cfg,
        exclude_symbol=symbol
    )

    if not hedge_basket:
        logger.warning(f"No hedge basket found for {symbol}")
        return {}

    # Get hedge symbols
    hedge_symbols = [h[0] for h in hedge_basket]

    # Compute volatilities
    vol_long, vol_basket = compute_hedge_volatilities(
        symbol=symbol,
        hedge_symbols=hedge_symbols,
        returns_df=returns_df,
        current_date=current_date,
        cfg=cfg
    )

    # Size hedge
    hedge_notional = size_hedge(
        long_notional=long_notional,
        vol_long=vol_long,
        vol_basket=vol_basket,
        cfg=cfg
    )

    # Compute entry correlation
    corr_window = cfg.get('monitor', {}).get('corr_window_days', 20)
    try:
        symbol_returns = returns_df.loc[:current_date, symbol].tail(corr_window)
        hedge_returns_df = returns_df.loc[:current_date, hedge_symbols].tail(corr_window)
        basket_return = hedge_returns_df.mean(axis=1)
        entry_corr = symbol_returns.corr(basket_return)
        if pd.isna(entry_corr):
            entry_corr = 0.0
    except Exception:
        entry_corr = 0.0

    # Create hedge info
    hedge_info = {
        'basket': [{'symbol': s, 'raw_weight': w} for s, w in hedge_basket],
        'notional': hedge_notional,
        'entry_correlation': float(entry_corr),
        'last_rebalance_date': current_date.strftime('%Y-%m-%d'),
        'vol_long': float(vol_long),
        'vol_basket': float(vol_basket)
    }

    logger.info(f"Created hedge for {symbol}:")
    logger.info(f"  Basket size: {len(hedge_basket)} securities")
    logger.info(f"  Hedge notional: ${hedge_notional:,.2f}")
    logger.info(f"  Vol ratio: {vol_long/max(vol_basket, 1e-6):.2f}")
    logger.info(f"  Entry correlation: {entry_corr:.3f}")

    return hedge_info
