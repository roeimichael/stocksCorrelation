"""Correlation analysis for EDA."""
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.core.logger import get_logger


logger = get_logger(__name__)


def corr_matrix(
    returns_df: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None
) -> pd.DataFrame:
    """
    Compute correlation matrix for stock returns.

    Args:
        returns_df: DataFrame with returns (index=dates, columns=symbols)
        start: Optional start date for filtering (default: use all data)
        end: Optional end date for filtering (default: use all data)

    Returns:
        Correlation matrix as DataFrame (symbols x symbols)

    Notes:
        - Uses Pearson correlation
        - Filters by date range if start/end provided
        - Handles NaN values (pairwise complete observations)

    Example:
        >>> returns_df = pd.read_parquet('data/processed/returns.parquet')
        >>> corr = corr_matrix(returns_df, start=pd.Timestamp('2023-01-01'))
        >>> print(corr.shape)  # (N, N) where N = number of symbols
        >>> corr.to_csv('results/correlation.csv')
    """
    # Filter by date range if provided
    if start is not None or end is not None:
        logger.info(f"Filtering returns: start={start}, end={end}")

        if start is not None and end is not None:
            filtered_df = returns_df.loc[start:end]
        elif start is not None:
            filtered_df = returns_df.loc[start:]
        else:  # end is not None
            filtered_df = returns_df.loc[:end]
    else:
        filtered_df = returns_df

    logger.info(f"Computing correlation matrix for {len(filtered_df.columns)} symbols "
                f"over {len(filtered_df)} days")

    # Compute Pearson correlation
    corr = filtered_df.corr(method='pearson')

    # Log summary statistics (upper triangle excluding diagonal)
    upper_triangle = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )
    correlations = upper_triangle.values.flatten()
    correlations = correlations[~np.isnan(correlations)]

    if len(correlations) > 0:
        logger.info(f"Correlation statistics: "
                    f"mean={correlations.mean():.3f}, "
                    f"median={np.median(correlations):.3f}, "
                    f"std={correlations.std():.3f}, "
                    f"min={correlations.min():.3f}, "
                    f"max={correlations.max():.3f}")
    else:
        logger.warning("No valid correlations computed")

    return corr


def save_heatmap(
    corr: pd.DataFrame,
    output_path: str | None = None,
    title: str = "Stock Returns Correlation Matrix",
    figsize: tuple = (12, 10),
    cmap: str = 'RdBu_r'
) -> None:
    """
    Save correlation matrix as heatmap PNG.

    Args:
        corr: Correlation matrix DataFrame
        output_path: Path to save figure (default: results/plots/corr_<timestamp>.png)
        title: Plot title
        figsize: Figure size in inches
        cmap: Colormap name (default: 'RdBu_r' for red-blue diverging)

    Notes:
        - If output_path not provided, uses results/plots/corr_<timestamp>.png
        - Creates output directory if it doesn't exist
        - Closes figure after saving (no display)
        - For large matrices (>50 symbols), consider sampling first

    Example:
        >>> corr = corr_matrix(returns_df)
        >>> save_heatmap(corr)  # Saves to results/plots/corr_YYYYMMDD_HHMMSS.png
        >>> save_heatmap(corr, 'my_corr.png')  # Custom path
    """
    if output_path is None:
        # Generate default path with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'results/plots/corr_{timestamp}.png'

    logger.info(f"Creating correlation heatmap: {corr.shape[0]} x {corr.shape[1]}")

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Create figure
    plt.figure(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        corr,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0,
        cbar_kws={"shrink": 0.8},
        xticklabels=True,
        yticklabels=True
    )

    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved heatmap to {output_path}")

    # Close figure to free memory
    plt.close()
