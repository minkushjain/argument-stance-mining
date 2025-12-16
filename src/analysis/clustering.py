"""
Clustering Analysis for Essay Stance Structure

Implements:
- K-means clustering with optimal k selection
- Hierarchical/Agglomerative clustering
- Visualization with t-SNE/PCA
- Cluster profiling and interpretation
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset_builder import DatasetBuilder
from src.analysis.essay_features import EssayFeatureExtractor, EssayFeatures


class ClusteringAnalysis:
    """
    Clustering analysis for essay stance structure patterns.
    """
    
    # Features to use for clustering (excluding IDs and categorical)
    CLUSTERING_FEATURES = [
        'for_ratio', 'against_ratio', 'stance_balance',
        'total_components', 'major_claim_count', 'claim_count', 'premise_count',
        'evidence_density', 'claim_density',
        'total_relations', 'support_count', 'attack_count',
        'support_ratio', 'attack_ratio', 'relations_per_component',
        'for_in_first_third', 'for_in_middle_third', 'for_in_last_third',
        'against_in_first_third', 'against_in_middle_third', 'against_in_last_third'
    ]
    
    def __init__(self, features_df: pd.DataFrame, output_dir: str = "reports/figures"):
        """
        Initialize clustering analysis.
        
        Args:
            features_df: DataFrame with essay features
            output_dir: Directory to save figures
        """
        self.df = features_df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter to available features
        self.feature_cols = [c for c in self.CLUSTERING_FEATURES if c in self.df.columns]
        
        # Prepare feature matrix
        self.X = self.df[self.feature_cols].values
        
        # Handle NaN/inf values
        self.X = np.nan_to_num(self.X, nan=0, posinf=0, neginf=0)
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Store results
        self.kmeans_model = None
        self.hierarchical_model = None
        self.optimal_k = None
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def find_optimal_k(self, k_range: Tuple[int, int] = (2, 10)) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            k_range: Range of k values to test
            
        Returns:
            Optimal k value
        """
        k_values = range(k_range[0], k_range[1] + 1)
        inertias = []
        silhouette_scores = []
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
            
            if k > 1:
                score = silhouette_score(self.X_scaled, kmeans.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
        
        # Plot elbow and silhouette
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        ax1 = axes[0]
        ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        ax1.set_title('Elbow Method for Optimal k', fontsize=14)
        ax1.set_xticks(list(k_values))
        
        # Silhouette plot
        ax2 = axes[1]
        ax2.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Score for Different k', fontsize=14)
        ax2.set_xticks(list(k_values))
        
        # Mark optimal k (best silhouette)
        optimal_idx = np.argmax(silhouette_scores)
        self.optimal_k = list(k_values)[optimal_idx]
        ax2.axvline(self.optimal_k, color='red', linestyle='--', 
                   label=f'Optimal k={self.optimal_k}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'optimal_k_selection.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Optimal k: {self.optimal_k} (silhouette score: {silhouette_scores[optimal_idx]:.3f})")
        return self.optimal_k
    
    def run_kmeans(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Run K-means clustering.
        
        Args:
            n_clusters: Number of clusters (uses optimal_k if None)
            
        Returns:
            Cluster labels
        """
        if n_clusters is None:
            if self.optimal_k is None:
                self.find_optimal_k()
            n_clusters = self.optimal_k
        
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = self.kmeans_model.fit_predict(self.X_scaled)
        
        self.df['kmeans_cluster'] = labels
        
        print(f"\nK-means Clustering (k={n_clusters})")
        print("Cluster distribution:")
        print(self.df['kmeans_cluster'].value_counts().sort_index())
        
        return labels
    
    def run_hierarchical(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Run hierarchical/agglomerative clustering.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.optimal_k or 4
        
        self.hierarchical_model = AgglomerativeClustering(
            n_clusters=n_clusters, linkage='ward'
        )
        labels = self.hierarchical_model.fit_predict(self.X_scaled)
        
        self.df['hierarchical_cluster'] = labels
        
        print(f"\nHierarchical Clustering (k={n_clusters})")
        print("Cluster distribution:")
        print(self.df['hierarchical_cluster'].value_counts().sort_index())
        
        return labels
    
    def plot_dendrogram(self, max_d: Optional[float] = None):
        """Plot dendrogram for hierarchical clustering."""
        linkage_matrix = linkage(self.X_scaled, method='ward')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        dendrogram(
            linkage_matrix,
            truncate_mode='lastp',
            p=30,  # Show only last 30 merges
            leaf_rotation=90,
            leaf_font_size=10,
            ax=ax
        )
        
        if max_d:
            ax.axhline(y=max_d, color='r', linestyle='--', label=f'Cut at {max_d}')
            ax.legend()
        
        ax.set_xlabel('Sample Index or Cluster Size', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dendrogram.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved dendrogram.png")
    
    def visualize_clusters_2d(self, method: str = 'pca'):
        """
        Visualize clusters in 2D using PCA or t-SNE.
        
        Args:
            method: 'pca' or 'tsne'
        """
        if 'kmeans_cluster' not in self.df.columns:
            self.run_kmeans()
        
        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(self.X_scaled)
            title_suffix = f"(PCA, explained var: {reducer.explained_variance_ratio_.sum():.1%})"
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            X_2d = reducer.fit_transform(self.X_scaled)
            title_suffix = "(t-SNE)"
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Color by K-means cluster
        ax1 = axes[0]
        scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], 
                              c=self.df['kmeans_cluster'], cmap='viridis',
                              alpha=0.7, edgecolors='black', s=60)
        ax1.set_xlabel('Component 1', fontsize=12)
        ax1.set_ylabel('Component 2', fontsize=12)
        ax1.set_title(f'K-means Clusters {title_suffix}', fontsize=14)
        plt.colorbar(scatter1, ax=ax1, label='Cluster')
        
        # Color by for_ratio
        ax2 = axes[1]
        scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1],
                              c=self.df['for_ratio'], cmap='RdYlGn',
                              alpha=0.7, edgecolors='black', s=60)
        ax2.set_xlabel('Component 1', fontsize=12)
        ax2.set_ylabel('Component 2', fontsize=12)
        ax2.set_title(f'Essays by For Ratio {title_suffix}', fontsize=14)
        plt.colorbar(scatter2, ax=ax2, label='For Ratio')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'clusters_{method}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved clusters_{method}.png")
    
    def profile_clusters(self, cluster_col: str = 'kmeans_cluster') -> pd.DataFrame:
        """
        Create profiles for each cluster.
        
        Args:
            cluster_col: Column with cluster labels
            
        Returns:
            DataFrame with cluster profiles
        """
        if cluster_col not in self.df.columns:
            raise ValueError(f"Column {cluster_col} not found. Run clustering first.")
        
        # Key features for profiling
        profile_features = [
            'for_ratio', 'against_ratio', 'stance_balance',
            'evidence_density', 'attack_ratio',
            'total_components', 'claim_count', 'premise_count'
        ]
        
        available_features = [f for f in profile_features if f in self.df.columns]
        
        # Compute mean for each cluster
        profiles = self.df.groupby(cluster_col)[available_features].mean()
        profiles['size'] = self.df.groupby(cluster_col).size()
        profiles['pct'] = (profiles['size'] / len(self.df) * 100).round(1)
        
        print(f"\n=== Cluster Profiles ({cluster_col}) ===")
        print(profiles.round(3))
        
        return profiles
    
    def plot_cluster_profiles(self, cluster_col: str = 'kmeans_cluster'):
        """Create radar chart comparing cluster profiles."""
        profiles = self.profile_clusters(cluster_col)
        
        # Features for radar chart
        radar_features = ['for_ratio', 'evidence_density', 'attack_ratio', 'claim_count']
        available = [f for f in radar_features if f in profiles.columns]
        
        if len(available) < 3:
            print("Not enough features for radar chart")
            return
        
        # Normalize features for radar chart
        normalized = profiles[available].copy()
        for col in available:
            col_min, col_max = normalized[col].min(), normalized[col].max()
            if col_max > col_min:
                normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
            else:
                normalized[col] = 0.5
        
        # Create radar chart
        n_clusters = len(profiles)
        n_features = len(available)
        
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
        
        for idx, (cluster_id, row) in enumerate(normalized.iterrows()):
            values = row.tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}', 
                   color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available, fontsize=11)
        ax.set_title('Cluster Profiles (Normalized)', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_profiles_radar.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved cluster_profiles_radar.png")
    
    def name_clusters(self, cluster_col: str = 'kmeans_cluster') -> Dict[int, str]:
        """
        Assign descriptive names to clusters based on their profiles.
        
        Returns:
            Dictionary mapping cluster ID to name
        """
        profiles = self.profile_clusters(cluster_col)
        
        cluster_names = {}
        
        for cluster_id in profiles.index:
            profile = profiles.loc[cluster_id]
            
            # Determine name based on characteristics
            name_parts = []
            
            # Stance characteristic
            if profile.get('for_ratio', 0.5) > 0.85:
                name_parts.append("one-sided_for")
            elif profile.get('for_ratio', 0.5) < 0.6:
                name_parts.append("dialectical")
            else:
                name_parts.append("balanced")
            
            # Evidence characteristic
            if profile.get('evidence_density', 2.5) > 3.5:
                name_parts.append("evidence_rich")
            elif profile.get('evidence_density', 2.5) < 2.0:
                name_parts.append("evidence_sparse")
            
            # Attack characteristic
            if profile.get('attack_ratio', 0) > 0.1:
                name_parts.append("attack_heavy")
            
            cluster_names[cluster_id] = "_".join(name_parts) if name_parts else f"cluster_{cluster_id}"
        
        print("\n=== Cluster Names ===")
        for cid, name in cluster_names.items():
            print(f"  Cluster {cid}: {name}")
        
        # Add to dataframe
        self.df[f'{cluster_col}_name'] = self.df[cluster_col].map(cluster_names)
        
        return cluster_names
    
    def plot_cluster_comparison(self, cluster_col: str = 'kmeans_cluster'):
        """Create boxplots comparing clusters on key features."""
        features_to_compare = [
            ('for_ratio', 'For Ratio'),
            ('evidence_density', 'Evidence Density'),
            ('attack_ratio', 'Attack Ratio'),
            ('total_components', 'Total Components')
        ]
        
        available = [(f, l) for f, l in features_to_compare if f in self.df.columns]
        n_features = len(available)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (feature, label) in enumerate(available):
            ax = axes[idx]
            
            clusters = sorted(self.df[cluster_col].unique())
            data = [self.df[self.df[cluster_col] == c][feature] for c in clusters]
            
            bp = ax.boxplot(data, tick_labels=[f'C{c}' for c in clusters], patch_artist=True)
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(clusters)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Cluster', fontsize=12)
            ax.set_ylabel(label, fontsize=12)
            ax.set_title(f'{label} by Cluster', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved cluster_comparison.png")
    
    def run_full_analysis(self, n_clusters: Optional[int] = None):
        """Run complete clustering analysis."""
        print("\n" + "="*60)
        print("CLUSTERING ANALYSIS")
        print("="*60)
        
        # Find optimal k
        print("\n1. Finding optimal number of clusters...")
        optimal_k = self.find_optimal_k()
        
        if n_clusters is None:
            n_clusters = optimal_k
        
        # Run K-means
        print(f"\n2. Running K-means clustering (k={n_clusters})...")
        self.run_kmeans(n_clusters)
        
        # Run hierarchical
        print(f"\n3. Running hierarchical clustering (k={n_clusters})...")
        self.run_hierarchical(n_clusters)
        
        # Visualizations
        print("\n4. Creating visualizations...")
        self.plot_dendrogram()
        self.visualize_clusters_2d(method='pca')
        self.visualize_clusters_2d(method='tsne')
        
        # Profiling
        print("\n5. Profiling clusters...")
        self.plot_cluster_profiles()
        self.plot_cluster_comparison()
        
        # Naming
        print("\n6. Naming clusters...")
        cluster_names = self.name_clusters()
        
        # Save results
        self.df.to_csv(self.output_dir.parent / 'essays_with_clusters.csv', index=False)
        print(f"\nSaved essays_with_clusters.csv")
        
        return self.df


def main():
    """Run clustering analysis on essay features."""
    dataset_dir = Path(__file__).parent.parent.parent / "dataset" / "ArgumentAnnotatedEssays-2.0"
    output_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
    
    print("Loading dataset...")
    builder = DatasetBuilder(str(dataset_dir))
    builder.build()
    
    print("\nExtracting features...")
    extractor = EssayFeatureExtractor()
    all_essays = builder.get_all_essays()
    features = extractor.extract_all(all_essays)
    
    # Convert to DataFrame
    df = extractor.to_dataframe(features)
    
    print(f"\nDataset shape: {df.shape}")
    
    # Run clustering analysis
    analysis = ClusteringAnalysis(df, str(output_dir))
    result_df = analysis.run_full_analysis()
    
    print(f"\n✓ Clustering analysis complete!")
    print(f"✓ All figures saved to {output_dir}")


if __name__ == "__main__":
    main()

