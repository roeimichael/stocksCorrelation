"""Correlation analysis for EDA."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from src.core.logger import get_logger

logger = get_logger()


def compute_correlation_matrix(
    returns: pd.DataFrame,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Compute correlation matrix for returns.

    Args:
        returns: DataFrame with returns (dates x tickers)
        method: Correlation method ('pearson' or 'spearman')

    Returns:
        Correlation matrix DataFrame
    """
    logger.info(f"Computing {method} correlation matrix for {len(returns.columns)} tickers")

    corr_matrix = returns.corr(method=method)

    # Log summary statistics
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    correlations = upper_triangle.values.flatten()
    correlations = correlations[~np.isnan(correlations)]

    logger.info(f"Correlation stats: mean={correlations.mean():.3f}, "
                f"median={np.median(correlations):.3f}, "
                f"std={correlations.std():.3f}")

    return corr_matrix


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Stock Returns Correlation Matrix",
    figsize: tuple = (12, 10)
) -> None:
    """
    Plot correlation matrix as heatmap.

    Args:
        corr_matrix: Correlation matrix
        output_path: Optional path to save figure
        title: Plot title
        figsize: Figure size
    """
    logger.info("Plotting correlation heatmap")

    plt.figure(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0,
        cbar_kws={"shrink": 0.8}
    )

    plt.title(title)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved heatmap to {output_path}")

    plt.close()


def plot_correlation_distribution(
    corr_matrix: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Distribution of Pairwise Correlations"
) -> None:
    """
    Plot histogram of pairwise correlations.

    Args:
        corr_matrix: Correlation matrix
        output_path: Optional path to save figure
        title: Plot title
    """
    logger.info("Plotting correlation distribution")

    # Extract upper triangle (exclude diagonal)
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    correlations = upper_triangle.values.flatten()
    correlations = correlations[~np.isnan(correlations)]

    plt.figure(figsize=(10, 6))

    plt.hist(correlations, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(correlations.mean(), color='red', linestyle='--',
                label=f'Mean: {correlations.mean():.3f}')
    plt.axvline(np.median(correlations), color='green', linestyle='--',
                label=f'Median: {np.median(correlations):.3f}')

    plt.xlabel('Correlation')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved distribution plot to {output_path}")

    plt.close()


def find_highly_correlated_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.8
) -> pd.DataFrame:
    """
    Find pairs of stocks with high correlation.

    Args:
        corr_matrix: Correlation matrix
        threshold: Minimum correlation to report

    Returns:
        DataFrame with highly correlated pairs
    """
    logger.info(f"Finding pairs with correlation >= {threshold}")

    pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) >= threshold:
                pairs.append({
                    'ticker1': corr_matrix.columns[i],
                    'ticker2': corr_matrix.columns[j],
                    'correlation': corr
                })

    df = pd.DataFrame(pairs)

    if len(df) > 0:
        df = df.sort_values('correlation', ascending=False)
        logger.info(f"Found {len(df)} highly correlated pairs")
    else:
        logger.info("No highly correlated pairs found")

    return df


def analyze_correlations(
    returns: pd.DataFrame,
    output_dir: str = 'results/experiments'
) -> dict:
    """
    Complete correlation analysis with plots.

    Args:
        returns: DataFrame with returns
        output_dir: Directory to save outputs

    Returns:
        Dictionary with correlation statistics
    """
    logger.info("Running correlation analysis")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(returns, method='pearson')

    # Plot heatmap (sample if too large)
    if len(corr_matrix) > 50:
        logger.info("Sampling 50 tickers for heatmap visualization")
        sample_tickers = corr_matrix.columns[:50]
        corr_sample = corr_matrix.loc[sample_tickers, sample_tickers]
        plot_correlation_heatmap(
            corr_sample,
            output_path=str(output_path / 'correlation_heatmap_sample.png')
        )
    else:
        plot_correlation_heatmap(
            corr_matrix,
            output_path=str(output_path / 'correlation_heatmap.png')
        )

    # Plot distribution
    plot_correlation_distribution(
        corr_matrix,
        output_path=str(output_path / 'correlation_distribution.png')
    )

    # Find highly correlated pairs
    high_corr_pairs = find_highly_correlated_pairs(corr_matrix, threshold=0.8)
    if len(high_corr_pairs) > 0:
        high_corr_pairs.to_csv(output_path / 'high_correlation_pairs.csv', index=False)
        logger.info(f"Saved high correlation pairs to {output_path / 'high_correlation_pairs.csv'}")

    # Summary statistics
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    correlations = upper_triangle.values.flatten()
    correlations = correlations[~np.isnan(correlations)]

    stats = {
        'mean': float(correlations.mean()),
        'median': float(np.median(correlations)),
        'std': float(correlations.std()),
        'min': float(correlations.min()),
        'max': float(correlations.max()),
        'num_pairs': len(correlations),
        'high_corr_pairs': len(high_corr_pairs)
    }

    logger.info(f"Correlation analysis complete: {stats}")

    return stats
