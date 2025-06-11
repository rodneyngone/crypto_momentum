# crypto_momentum_backtest/risk/correlation_monitor.py
"""Correlation monitoring for portfolio risk management."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import logging


class CorrelationMonitor:
    """
    Monitors and analyzes correlation dynamics in the portfolio.
    """
    
    def __init__(
        self,
        lookback_period: int = 60,
        rolling_window: int = 30,
        correlation_threshold: float = 0.8,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize correlation monitor.
        
        Args:
            lookback_period: Historical lookback for correlation
            rolling_window: Rolling correlation window
            correlation_threshold: Threshold for high correlation warning
            logger: Logger instance
        """
        self.lookback_period = lookback_period
        self.rolling_window = rolling_window
        self.correlation_threshold = correlation_threshold
        self.logger = logger or logging.getLogger(__name__)
        
    def calculate_correlation_matrix(
        self,
        returns: pd.DataFrame,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix.
        
        Args:
            returns: Returns DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix
        """
        return returns.corr(method=method)
    
    def calculate_rolling_correlation(
        self,
        returns: pd.DataFrame,
        asset1: str,
        asset2: str
    ) -> pd.Series:
        """
        Calculate rolling correlation between two assets.
        
        Args:
            returns: Returns DataFrame
            asset1: First asset
            asset2: Second asset
            
        Returns:
            Rolling correlation series
        """
        return returns[asset1].rolling(
            window=self.rolling_window
        ).corr(returns[asset2])
    
    def find_high_correlations(
        self,
        returns: pd.DataFrame
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs with high correlation.
        
        Args:
            returns: Returns DataFrame
            
        Returns:
            List of (asset1, asset2, correlation) tuples
        """
        corr_matrix = self.calculate_correlation_matrix(returns)
        high_corr_pairs = []
        
        # Get upper triangle of correlation matrix
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) > self.correlation_threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))
        
        # Sort by absolute correlation
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return high_corr_pairs
    
    def calculate_correlation_stability(
        self,
        returns: pd.DataFrame,
        lookback_windows: List[int] = [30, 60, 90]
    ) -> pd.DataFrame:
        """
        Analyze correlation stability across different time windows.
        
        Args:
            returns: Returns DataFrame
            lookback_windows: List of lookback periods
            
        Returns:
            Stability analysis DataFrame
        """
        stability_results = []
        
        for window in lookback_windows:
            corr_matrix = returns.iloc[-window:].corr()
            
            # Calculate average correlation
            mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
            avg_corr = corr_matrix.where(mask).stack().mean()
            
            stability_results.append({
                'window': window,
                'avg_correlation': avg_corr,
                'max_correlation': corr_matrix.where(mask).stack().max(),
                'min_correlation': corr_matrix.where(mask).stack().min(),
                'std_correlation': corr_matrix.where(mask).stack().std()
            })
        
        return pd.DataFrame(stability_results)
    
    def perform_clustering(
        self,
        returns: pd.DataFrame,
        n_clusters: int = 3
    ) -> Dict[str, int]:
        """
        Perform hierarchical clustering on assets.
        
        Args:
            returns: Returns DataFrame
            n_clusters: Number of clusters
            
        Returns:
            Asset to cluster mapping
        """
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(returns)
        
        # Convert correlation to distance
        distance_matrix = 1 - corr_matrix
        
        # Perform hierarchical clustering
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = hierarchy.linkage(condensed_dist, method='ward')
        
        # Get cluster labels
        cluster_labels = hierarchy.fcluster(
            linkage_matrix,
            n_clusters,
            criterion='maxclust'
        )
        
        # Create mapping
        cluster_mapping = {
            asset: int(cluster)
            for asset, cluster in zip(returns.columns, cluster_labels)
        }
        
        return cluster_mapping
    
    def calculate_diversification_ratio(
        self,
        returns: pd.DataFrame,
        weights: pd.Series
    ) -> float:
        """
        Calculate portfolio diversification ratio.
        
        Args:
            returns: Returns DataFrame
            weights: Portfolio weights
            
        Returns:
            Diversification ratio
        """
        # Calculate weighted average volatility
        individual_vols = returns.std()
        weighted_avg_vol = (weights * individual_vols).sum()
        
        # Calculate portfolio volatility
        cov_matrix = returns.cov()
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        # Diversification ratio
        div_ratio = weighted_avg_vol / portfolio_vol
        
        return div_ratio
    
    def plot_correlation_heatmap(
        self,
        returns: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Plot correlation heatmap.
        
        Args:
            returns: Returns DataFrame
            figsize: Figure size
        """
        corr_matrix = self.calculate_correlation_matrix(returns)
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('Asset Correlation Matrix')
        plt.tight_layout()
        
        return plt.gcf()
    
    def generate_correlation_report(
        self,
        returns: pd.DataFrame,
        positions: pd.Series
    ) -> Dict:
        """
        Generate comprehensive correlation report.
        
        Args:
            returns: Returns DataFrame
            positions: Current positions
            
        Returns:
            Correlation analysis report
        """
        # Filter to active positions
        active_assets = positions[positions != 0].index
        active_returns = returns[active_assets]
        
        report = {
            'timestamp': pd.Timestamp.now(),
            'n_assets': len(active_assets),
            'correlation_matrix': self.calculate_correlation_matrix(active_returns)
        }
        
        # High correlation pairs
        high_corr = self.find_high_correlations(active_returns)
        report['high_correlation_pairs'] = high_corr
        report['n_high_correlation_pairs'] = len(high_corr)
        
        # Correlation statistics
        corr_matrix = report['correlation_matrix']
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        corr_values = corr_matrix.where(mask).stack()
        
        report['correlation_stats'] = {
            'mean': corr_values.mean(),
            'std': corr_values.std(),
            'min': corr_values.min(),
            'max': corr_values.max(),
            'median': corr_values.median()
        }
        
        # Clustering
        if len(active_assets) > 3:
            report['clusters'] = self.perform_clustering(active_returns)
        
        # Diversification metrics
        normalized_positions = positions[active_assets] / positions[active_assets].sum()
        report['diversification_ratio'] = self.calculate_diversification_ratio(
            active_returns,
            normalized_positions
        )
        
        return report
