"""Service for correlation analysis."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from src.core.logger import get_logger
from src.core.constants import Paths
from src.core.data_loader import load_returns

logger = get_logger(__name__)


class CorrelationsService:
    """Service for computing and analyzing correlations."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.returns_file = Paths.RETURNS_FILE

    def get_correlation_matrix(self, symbols: Optional[List[str]] = None, lookback_days: int = 60) -> Dict:
        """
        Compute correlation matrix for given symbols.

        Args:
            symbols: List of symbols (if None, use all portfolio symbols)
            lookback_days: Number of days to look back

        Returns:
            Dictionary with correlation matrix and analysis
        """
        try:
            # Load returns data
            returns_df = load_returns()

            if returns_df is None or returns_df.empty:
                logger.warning("No returns data available")
                return self._empty_response()

            # Filter to lookback period
            end_date = returns_df.index[-1]
            start_date = end_date - timedelta(days=lookback_days)
            returns_df = returns_df[returns_df.index >= start_date]

            # Filter to specified symbols
            if symbols:
                available_symbols = [s for s in symbols if s in returns_df.columns]
                if not available_symbols:
                    logger.warning(f"No data available for symbols: {symbols}")
                    return self._empty_response()
                returns_df = returns_df[available_symbols]
            else:
                # Use all available symbols (limit to reasonable number)
                returns_df = returns_df.iloc[:, :100]  # Max 100 symbols

            # Compute correlation matrix
            corr_matrix = returns_df.corr()

            # Convert to list format for JSON
            symbols_list = corr_matrix.columns.tolist()
            matrix_list = corr_matrix.values.tolist()

            # Find top correlated pairs
            top_positive, top_negative = self._find_top_pairs(corr_matrix)

            return {
                "symbols": symbols_list,
                "matrix": matrix_list,
                "lookback_days": lookback_days,
                "top_positive": top_positive,
                "top_negative": top_negative,
            }

        except Exception as e:
            logger.error(f"Error computing correlation matrix: {e}")
            return self._empty_response()

    def get_portfolio_correlation(self, positions_symbols: List[str], lookback_days: int = 60) -> Dict:
        """
        Analyze correlation for current portfolio positions.

        Args:
            positions_symbols: List of symbols in current portfolio
            lookback_days: Number of days to look back

        Returns:
            Dictionary with portfolio correlation analysis
        """
        try:
            if not positions_symbols:
                return self._empty_portfolio_response()

            # Get correlation matrix for portfolio
            corr_data = self.get_correlation_matrix(symbols=positions_symbols, lookback_days=lookback_days)

            if not corr_data or not corr_data.get("symbols"):
                return self._empty_portfolio_response()

            # Calculate diversification score
            corr_matrix = np.array(corr_data["matrix"])
            diversification_score = self._calculate_diversification_score(corr_matrix)

            # Calculate concentration risk (inverse of diversification)
            concentration_risk = 1.0 - diversification_score

            # Find suggested hedges
            suggested_hedges = self._find_hedge_candidates(positions_symbols, lookback_days)

            return {
                "portfolio_symbols": positions_symbols,
                "correlation_matrix": corr_data,
                "diversification_score": diversification_score,
                "concentration_risk": concentration_risk,
                "suggested_hedges": suggested_hedges,
            }

        except Exception as e:
            logger.error(f"Error analyzing portfolio correlation: {e}")
            return self._empty_portfolio_response()

    def _find_top_pairs(self, corr_matrix: pd.DataFrame, n: int = 10) -> tuple:
        """Find top N positively and negatively correlated pairs."""
        # Get upper triangle (avoid duplicates and self-correlation)
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Stack to get pairs
        pairs = upper_tri.stack().reset_index()
        pairs.columns = ["symbol1", "symbol2", "correlation"]

        # Sort and get top/bottom
        top_positive = pairs.nlargest(n, "correlation").to_dict("records")
        top_negative = pairs.nsmallest(n, "correlation").to_dict("records")

        # Add lookback_days to each pair
        for pair in top_positive + top_negative:
            pair["lookback_days"] = 60

        return top_positive, top_negative

    def _calculate_diversification_score(self, corr_matrix: np.ndarray) -> float:
        """
        Calculate diversification score from correlation matrix.
        Higher score = better diversification (lower average correlation).
        """
        # Get upper triangle (excluding diagonal)
        n = len(corr_matrix)
        if n <= 1:
            return 1.0

        upper_tri_indices = np.triu_indices(n, k=1)
        correlations = corr_matrix[upper_tri_indices]

        # Average absolute correlation
        avg_corr = np.abs(correlations).mean()

        # Diversification score (inverse of average correlation)
        # Scale to 0-1 range
        diversification_score = max(0.0, 1.0 - avg_corr)

        return round(diversification_score, 3)

    def _find_hedge_candidates(self, portfolio_symbols: List[str], lookback_days: int = 60, n: int = 5) -> List[Dict]:
        """
        Find potential hedge candidates (negatively correlated stocks).

        Args:
            portfolio_symbols: Current portfolio symbols
            lookback_days: Lookback period
            n: Number of hedge candidates to return

        Returns:
            List of hedge candidate dicts
        """
        try:
            # Load all returns
            returns_df = load_returns()
            if returns_df is None or returns_df.empty:
                return []

            # Filter to lookback period
            end_date = returns_df.index[-1]
            start_date = end_date - timedelta(days=lookback_days)
            returns_df = returns_df[returns_df.index >= start_date]

            # Calculate portfolio returns (equal weight)
            portfolio_returns = returns_df[portfolio_symbols].mean(axis=1)

            # Calculate correlation with all other stocks
            all_symbols = [s for s in returns_df.columns if s not in portfolio_symbols]
            correlations = []

            for symbol in all_symbols[:200]:  # Limit search
                try:
                    corr = portfolio_returns.corr(returns_df[symbol])
                    if pd.notna(corr):
                        correlations.append({"symbol": symbol, "correlation": round(corr, 3)})
                except Exception:
                    continue

            # Sort by correlation (most negative first)
            correlations.sort(key=lambda x: x["correlation"])

            # Return top N negative correlations
            return correlations[:n]

        except Exception as e:
            logger.error(f"Error finding hedge candidates: {e}")
            return []

    def _empty_response(self) -> Dict:
        """Return empty correlation response."""
        return {
            "symbols": [],
            "matrix": [],
            "lookback_days": 0,
            "top_positive": [],
            "top_negative": [],
        }

    def _empty_portfolio_response(self) -> Dict:
        """Return empty portfolio correlation response."""
        return {
            "portfolio_symbols": [],
            "correlation_matrix": self._empty_response(),
            "diversification_score": 0.0,
            "concentration_risk": 0.0,
            "suggested_hedges": [],
        }
