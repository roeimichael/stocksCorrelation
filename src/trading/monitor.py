"""
Trading position monitoring with drift detection metrics.

This module provides functions to monitor open trading positions and detect
when patterns have drifted from their entry conditions.
"""
from typing import Any

import numpy as np
import pandas as pd

from src.core.constants import MonitoringConstants, TradingConstants
from src.core.logger import get_logger
from src.modeling.similarity import compute_similarity
from src.modeling.windows import normalize_window


logger = get_logger(__name__)


def similarity_retention(
    position: dict[str, Any],
    today: pd.Timestamp,
    returns_df: pd.DataFrame,
    windows_bank: pd.DataFrame,
    cfg: dict[str, Any]
) -> float:
    """
    Compute similarity retention: compare today's target window to each saved analog.

    Args:
        position: Position dict with keys: symbol, entry_date, analogs (list of dicts)
        today: Current date for which to compute metrics
        returns_df: DataFrame with returns (index=dates, columns=symbols)
        windows_bank: DataFrame with all windows (for looking up analog features)
        cfg: Configuration dictionary

    Returns:
        Weighted average similarity between today's window and entry analogs.
        Returns 0.0 if insufficient data or computation fails.

    Process:
        1. Form today's target window for position symbol
        2. For each analog from entry, retrieve its feature vector from windows_bank
        3. Compute similarity between today's window and each analog's window
        4. Return weighted average (using analog sim scores as weights)
    """
    symbol = position['symbol']
    analogs = position.get('analogs', [])

    if not analogs:
        logger.warning(f"No analogs found for {symbol}")
        return 0.0

    window_length = cfg['windows']['length']
    normalization = cfg['windows']['normalization']
    similarity_metric = cfg['similarity']['metric']
    epsilon = cfg['windows'].get('epsilon', TradingConstants.EPSILON)

    # Get returns for symbol up to today
    if symbol not in returns_df.columns:
        logger.warning(f"Symbol {symbol} not found in returns_df")
        return 0.0

    symbol_returns = returns_df.loc[:today, symbol]

    # Need at least window_length returns
    if len(symbol_returns) < window_length:
        logger.warning(f"Insufficient data for {symbol}: {len(symbol_returns)} < {window_length}")
        return 0.0

    # Form today's target window
    recent_returns = symbol_returns.iloc[-window_length:]
    if recent_returns.isna().any():
        logger.warning(f"NaN values in recent returns for {symbol}")
        return 0.0

    try:
        today_vec = normalize_window(recent_returns.values, method=normalization, epsilon=epsilon)
    except (ValueError, ZeroDivisionError) as e:
        logger.warning(f"Failed to normalize today's window for {symbol}: {e}")
        return 0.0

    # Compute similarity to each analog
    similarities = []
    weights = []

    for analog in analogs:
        analog_symbol = analog['symbol']
        analog_end_date = pd.Timestamp(analog['end_date'])
        analog_sim = analog['sim']  # Original similarity score used as weight

        # Find analog's window in bank
        matches = windows_bank[
            (windows_bank['symbol'] == analog_symbol) &
            (windows_bank['end_date'] == analog_end_date)
        ]

        if len(matches) == 0:
            logger.debug(f"Analog window not found: {analog_symbol} ending {analog_end_date}")
            continue

        analog_features = matches.iloc[0]['features']

        # Compute similarity between today's window and analog's window
        try:
            sim = compute_similarity(today_vec, analog_features, similarity_metric)
            similarities.append(sim)
            weights.append(analog_sim)
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Failed to compute similarity to analog {analog_symbol}: {e}")
            continue

    if not similarities:
        logger.warning(f"No valid similarities computed for {symbol}")
        return 0.0

    # Weighted average
    weights_array = np.array(weights)
    similarities_array = np.array(similarities)

    weighted_sim = np.average(similarities_array, weights=weights_array)

    return float(weighted_sim)


def directional_concordance(
    position: dict[str, Any],
    today: pd.Timestamp,
    returns_df: pd.DataFrame,
    cfg: dict[str, Any]
) -> float:
    """
    Compute directional concordance: weighted share where current return sign matches analog's historical direction.

    Args:
        position: Position dict with keys: symbol, entry_date, side, analogs
        today: Current date
        returns_df: DataFrame with returns
        cfg: Configuration dictionary

    Returns:
        Weighted fraction of analogs where sign(current return) matches analog's label.
        Returns 0.0 if insufficient data.

    Process:
        1. Get today's return for position symbol
        2. For each analog, check if sign(today's return) matches analog's label
           - label=1 (UP): expect positive return
           - label=0 (DOWN): expect negative return
        3. Return weighted average of matches (using analog sim as weights)
    """
    symbol = position['symbol']
    analogs = position.get('analogs', [])

    if not analogs:
        logger.warning(f"No analogs found for {symbol}")
        return 0.0

    # Get today's return
    if symbol not in returns_df.columns or today not in returns_df.index:
        logger.warning(f"No return data for {symbol} on {today}")
        return 0.0

    today_return = returns_df.loc[today, symbol]

    if pd.isna(today_return):
        logger.warning(f"NaN return for {symbol} on {today}")
        return 0.0

    # Determine current direction
    current_direction = 1 if today_return > 0 else 0

    # Check concordance with each analog
    matches = []
    weights = []

    for analog in analogs:
        analog_label = analog['label']
        analog_sim = analog['sim']

        # Check if directions match
        # analog_label=1 means analog went up, expect current to go up
        # analog_label=0 means analog went down, expect current to go down
        is_concordant = (current_direction == analog_label)

        matches.append(1.0 if is_concordant else 0.0)
        weights.append(analog_sim)

    if not matches:
        return 0.0

    # Weighted average
    weights_array = np.array(weights)
    matches_array = np.array(matches)

    concordance = np.average(matches_array, weights=weights_array)

    return float(concordance)


def correlation_decay(
    position: dict[str, Any],
    today: pd.Timestamp,
    returns_df: pd.DataFrame,
    cfg: dict[str, Any]
) -> tuple[float, float]:
    """
    Compute correlation decay: rolling correlation between symbol and analog basket.

    Args:
        position: Position dict with keys: symbol, entry_date, analogs
        today: Current date
        returns_df: DataFrame with returns
        cfg: Configuration dictionary with monitor section

    Returns:
        Tuple of (corr_today, delta_corr_3d):
        - corr_today: Current M-day rolling correlation
        - delta_corr_3d: Change in correlation over past 3 days

    Process:
        1. Create weighted analog basket return (weighted by sim scores)
        2. Compute M-day rolling correlation between symbol and basket
        3. Get today's correlation and correlation from 3 days ago
        4. Return (corr_today, corr_today - corr_3d_ago)
    """
    symbol = position['symbol']
    analogs = position.get('analogs', [])

    if not analogs:
        logger.warning(f"No analogs found for {symbol}")
        return 0.0, 0.0

    # Get monitoring window length
    corr_window = cfg.get('monitor', {}).get('corr_window_days', MonitoringConstants.CORR_WINDOW_DAYS)
    lookback_days = cfg.get('monitor', {}).get('lookback_days', MonitoringConstants.CORR_LOOKBACK_DAYS)

    # Get returns for symbol
    if symbol not in returns_df.columns:
        logger.warning(f"Symbol {symbol} not found in returns_df")
        return 0.0, 0.0

    # Need enough history for rolling correlation
    symbol_returns = returns_df.loc[:today, symbol]
    if len(symbol_returns) < corr_window + lookback_days:
        logger.warning(f"Insufficient data for correlation: {len(symbol_returns)}")
        return 0.0, 0.0

    # Create analog basket return (weighted average)
    analog_symbols = [a['symbol'] for a in analogs]
    analog_weights = np.array([a['sim'] for a in analogs])
    analog_weights = analog_weights / analog_weights.sum()  # Normalize to sum to 1

    # Get returns for analog symbols
    available_analogs = [s for s in analog_symbols if s in returns_df.columns]

    if not available_analogs:
        logger.warning("No analog symbols found in returns_df")
        return 0.0, 0.0

    # Filter weights for available analogs
    available_weights = []
    for i, analog_symbol in enumerate(analog_symbols):
        if analog_symbol in available_analogs:
            available_weights.append(analog_weights[i])

    available_weights = np.array(available_weights)
    available_weights = available_weights / available_weights.sum()  # Re-normalize

    # Create basket returns
    analog_returns = returns_df.loc[:today, available_analogs]
    basket_returns = (analog_returns * available_weights).sum(axis=1)

    # Compute rolling correlation
    symbol_series = symbol_returns.iloc[-(corr_window + lookback_days):]
    basket_series = basket_returns.iloc[-(corr_window + lookback_days):]

    # Compute rolling correlation
    rolling_corr = symbol_series.rolling(window=corr_window).corr(basket_series)

    # Get today's correlation
    if pd.isna(rolling_corr.iloc[-1]):
        logger.warning(f"NaN correlation for {symbol}")
        return 0.0, 0.0

    corr_today = rolling_corr.iloc[-1]

    # Get correlation from lookback_days ago
    if len(rolling_corr) < lookback_days + 1:
        corr_past = rolling_corr.iloc[0] if not pd.isna(rolling_corr.iloc[0]) else corr_today
    else:
        corr_past = rolling_corr.iloc[-(lookback_days + 1)]
        if pd.isna(corr_past):
            corr_past = corr_today

    delta_corr = corr_today - corr_past

    return float(corr_today), float(delta_corr)


def pattern_deviation_z(
    position: dict[str, Any],
    today: pd.Timestamp,
    returns_df: pd.DataFrame,
    cfg: dict[str, Any]
) -> float:
    """
    Compute pattern deviation z-score: how much has the pattern drifted from entry.

    Args:
        position: Position dict with keys: symbol, entry_date
        today: Current date
        returns_df: DataFrame with returns
        cfg: Configuration dictionary

    Returns:
        Z-score of ||current window - entry window||_2 relative to symbol's rolling distribution.
        Returns 0.0 if insufficient data.

    Process:
        1. Reconstruct entry window (from entry_date backwards)
        2. Form today's window
        3. Compute L2 distance between windows
        4. Compute rolling mean/std of L2 distances over past N windows
        5. Return z-score: (current_distance - mean) / std
    """
    symbol = position['symbol']
    entry_date = pd.Timestamp(position['entry_date'])

    window_length = cfg['windows']['length']
    normalization = cfg['windows']['normalization']
    epsilon = cfg['windows'].get('epsilon', TradingConstants.EPSILON)

    # Get monitor config
    deviation_window = cfg.get('monitor', {}).get('deviation_window_days', MonitoringConstants.DEVIATION_WINDOW_DAYS)

    # Get returns for symbol
    if symbol not in returns_df.columns:
        logger.warning(f"Symbol {symbol} not found in returns_df")
        return 0.0

    symbol_returns = returns_df[symbol]

    # Need data from entry to today
    if entry_date not in symbol_returns.index or today not in symbol_returns.index:
        logger.warning(f"Entry date or today not in returns for {symbol}")
        return 0.0

    # Reconstruct entry window
    entry_returns = symbol_returns.loc[:entry_date]
    if len(entry_returns) < window_length:
        logger.warning(f"Insufficient data for entry window: {len(entry_returns)}")
        return 0.0

    entry_window = entry_returns.iloc[-window_length:].values

    try:
        entry_vec = normalize_window(entry_window, method=normalization, epsilon=epsilon)
    except (ValueError, ZeroDivisionError) as e:
        logger.warning(f"Failed to normalize entry window: {e}")
        return 0.0

    # Form today's window
    today_returns = symbol_returns.loc[:today]
    if len(today_returns) < window_length:
        return 0.0

    today_window = today_returns.iloc[-window_length:].values

    try:
        today_vec = normalize_window(today_window, method=normalization, epsilon=epsilon)
    except (ValueError, ZeroDivisionError) as e:
        logger.warning(f"Failed to normalize today's window: {e}")
        return 0.0

    # Compute L2 distance
    current_distance = np.linalg.norm(today_vec - entry_vec)

    # Compute rolling distribution of L2 distances
    # Get all possible windows from entry to today
    all_returns = symbol_returns.loc[entry_date:today]

    if len(all_returns) < window_length + deviation_window:
        # Not enough data for distribution, just return raw distance
        return float(current_distance)

    # Compute L2 distances for rolling windows
    distances = []
    for i in range(window_length, len(all_returns)):
        window = all_returns.iloc[i-window_length:i].values
        try:
            window_vec = normalize_window(window, method=normalization, epsilon=epsilon)
            dist = np.linalg.norm(window_vec - entry_vec)
            distances.append(dist)
        except (ValueError, ZeroDivisionError):
            continue

    if len(distances) < 2:
        return float(current_distance)

    # Compute z-score
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    if std_dist == 0:
        return 0.0

    z_score = (current_distance - mean_dist) / std_dist

    return float(z_score)


def classify_alert(
    sr: float,
    dc: float,
    dcd: float,
    pdz: float,
    cfg: dict[str, Any]
) -> str:
    """
    Classify alert level based on drift metrics.

    Args:
        sr: Similarity retention
        dc: Directional concordance
        dcd: Delta correlation (3-day change)
        pdz: Pattern deviation z-score
        cfg: Configuration dictionary with monitor.thresholds section

    Returns:
        "GREEN", "YELLOW", or "RED"

    Logic:
        RED if ANY of:
        - sr < sr_floor_red
        - dc < dc_floor_red
        - dcd < cd_drop_alert_red
        - pdz > pds_z_alert_red

        YELLOW if ANY of:
        - sr < sr_floor_yellow
        - dc < dc_floor_yellow
        - dcd < cd_drop_alert_yellow
        - pdz > pds_z_alert_yellow

        Otherwise GREEN
    """
    thresholds = cfg.get('monitor', {}).get('thresholds', {})

    # Red thresholds (most severe)
    sr_floor_red = thresholds.get('sr_floor_red', MonitoringConstants.SR_FLOOR_RED)
    dc_floor_red = thresholds.get('dc_floor_red', MonitoringConstants.DC_FLOOR_RED)
    cd_drop_alert_red = thresholds.get('cd_drop_alert_red', MonitoringConstants.CD_DROP_ALERT_RED)
    pds_z_alert_red = thresholds.get('pds_z_alert_red', MonitoringConstants.PDS_Z_ALERT_RED)

    # Yellow thresholds (warning)
    sr_floor_yellow = thresholds.get('sr_floor_yellow', MonitoringConstants.SR_FLOOR_YELLOW)
    dc_floor_yellow = thresholds.get('dc_floor_yellow', MonitoringConstants.DC_FLOOR_YELLOW)
    cd_drop_alert_yellow = thresholds.get('cd_drop_alert_yellow', MonitoringConstants.CD_DROP_ALERT_YELLOW)
    pds_z_alert_yellow = thresholds.get('pds_z_alert_yellow', MonitoringConstants.PDS_Z_ALERT_YELLOW)

    # Check for RED conditions
    if (sr < sr_floor_red or
        dc < dc_floor_red or
        dcd < cd_drop_alert_red or
        pdz > pds_z_alert_red):
        return "RED"

    # Check for YELLOW conditions
    if (sr < sr_floor_yellow or
        dc < dc_floor_yellow or
        dcd < cd_drop_alert_yellow or
        pdz > pds_z_alert_yellow):
        return "YELLOW"

    # Otherwise GREEN
    return "GREEN"
