"""
Exploratory Data Analysis for Essay Stance Structure

Creates comprehensive visualizations:
- Distribution plots for all features
- Correlation heatmaps
- Stance distribution analysis
- Feature statistics by split (train/val/test)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset_builder import DatasetBuilder
from src.analysis.essay_features import EssayFeatureExtractor, EssayFeatures


class EDA:
    """
    Exploratory Data Analysis for essay stance structure.
    """
    
    def __init__(self, features_df: pd.DataFrame, output_dir: str = "reports/figures"):
        """
        Initialize EDA with features DataFrame.
        
        Args:
            features_df: DataFrame with essay features
            output_dir: Directory to save figures
        """
        self.df = features_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_stance_distribution(self):
        """Plot stance distribution across essays."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. For ratio histogram
        ax1 = axes[0, 0]
        ax1.hist(self.df['for_ratio'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(self.df['for_ratio'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {self.df["for_ratio"].mean():.2f}')
        ax1.set_xlabel('For Ratio', fontsize=12)
        ax1.set_ylabel('Number of Essays', fontsize=12)
        ax1.set_title('Distribution of For Ratio (For Claims / Total Stance Claims)', fontsize=12)
        ax1.legend()
        
        # 2. For vs Against counts scatter
        ax2 = axes[0, 1]
        scatter = ax2.scatter(self.df['for_count'], self.df['against_count'], 
                             c=self.df['for_ratio'], cmap='RdYlGn', alpha=0.6, edgecolors='black')
        ax2.set_xlabel('For Count', fontsize=12)
        ax2.set_ylabel('Against Count', fontsize=12)
        ax2.set_title('For vs Against Claim Counts per Essay', fontsize=12)
        plt.colorbar(scatter, ax=ax2, label='For Ratio')
        # Add diagonal line for balance
        max_val = max(self.df['for_count'].max(), self.df['against_count'].max())
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Balance line')
        ax2.legend()
        
        # 3. Stance category distribution
        ax3 = axes[1, 0]
        category_counts = self.df['stance_category'].value_counts()
        colors = {'mostly_for': '#2ecc71', 'balanced': '#3498db', 'mostly_against': '#e74c3c', 'no_stance': '#95a5a6'}
        bar_colors = [colors.get(cat, '#95a5a6') for cat in category_counts.index]
        bars = ax3.bar(category_counts.index, category_counts.values, color=bar_colors, edgecolor='black')
        ax3.set_xlabel('Stance Category', fontsize=12)
        ax3.set_ylabel('Number of Essays', fontsize=12)
        ax3.set_title('Essay Stance Category Distribution', fontsize=12)
        # Add percentage labels
        for bar, count in zip(bars, category_counts.values):
            pct = count / len(self.df) * 100
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{pct:.1f}%', ha='center', fontsize=10)
        
        # 4. Stance balance distribution
        ax4 = axes[1, 1]
        ax4.hist(self.df['stance_balance'], bins=20, edgecolor='black', alpha=0.7, color='coral')
        ax4.axvline(self.df['stance_balance'].mean(), color='red', linestyle='--',
                    label=f'Mean: {self.df["stance_balance"].mean():.2f}')
        ax4.set_xlabel('Stance Balance (|For - Against| / Total)', fontsize=12)
        ax4.set_ylabel('Number of Essays', fontsize=12)
        ax4.set_title('Distribution of Stance Balance (1 = one-sided, 0 = balanced)', fontsize=12)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stance_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved stance_distribution.png")
    
    def plot_component_structure(self):
        """Plot component and structure analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Component counts distribution
        ax1 = axes[0, 0]
        component_data = self.df[['major_claim_count', 'claim_count', 'premise_count']]
        component_data.plot(kind='box', ax=ax1)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Distribution of Component Counts per Essay', fontsize=12)
        ax1.set_xticklabels(['MajorClaims', 'Claims', 'Premises'], rotation=0)
        
        # 2. Evidence density distribution
        ax2 = axes[0, 1]
        ax2.hist(self.df['evidence_density'], bins=20, edgecolor='black', alpha=0.7, color='teal')
        ax2.axvline(self.df['evidence_density'].mean(), color='red', linestyle='--',
                    label=f'Mean: {self.df["evidence_density"].mean():.2f}')
        ax2.set_xlabel('Evidence Density (Premises / Claims)', fontsize=12)
        ax2.set_ylabel('Number of Essays', fontsize=12)
        ax2.set_title('Distribution of Evidence Density', fontsize=12)
        ax2.legend()
        
        # 3. Relations distribution
        ax3 = axes[1, 0]
        relations_data = self.df[['support_count', 'attack_count']]
        relations_data.plot(kind='box', ax=ax3)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('Distribution of Relation Counts per Essay', fontsize=12)
        ax3.set_xticklabels(['Support Relations', 'Attack Relations'], rotation=0)
        
        # 4. Attack ratio distribution
        ax4 = axes[1, 1]
        ax4.hist(self.df['attack_ratio'], bins=20, edgecolor='black', alpha=0.7, color='indianred')
        ax4.axvline(self.df['attack_ratio'].mean(), color='red', linestyle='--',
                    label=f'Mean: {self.df["attack_ratio"].mean():.2f}')
        ax4.set_xlabel('Attack Ratio (Attacks / Total Relations)', fontsize=12)
        ax4.set_ylabel('Number of Essays', fontsize=12)
        ax4.set_title('Distribution of Attack Ratio', fontsize=12)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'component_structure.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved component_structure.png")
    
    def plot_positional_analysis(self):
        """Plot where stances appear in essays (positional analysis)."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. For claims by position
        ax1 = axes[0, 0]
        for_positions = self.df[['for_in_first_third', 'for_in_middle_third', 'for_in_last_third']]
        for_means = for_positions.mean()
        ax1.bar(['First Third', 'Middle Third', 'Last Third'], for_means.values, 
               color='#2ecc71', edgecolor='black')
        ax1.set_ylabel('Average Count', fontsize=12)
        ax1.set_title('Average "For" Claims by Essay Position', fontsize=12)
        
        # 2. Against claims by position
        ax2 = axes[0, 1]
        against_positions = self.df[['against_in_first_third', 'against_in_middle_third', 'against_in_last_third']]
        against_means = against_positions.mean()
        ax2.bar(['First Third', 'Middle Third', 'Last Third'], against_means.values,
               color='#e74c3c', edgecolor='black')
        ax2.set_ylabel('Average Count', fontsize=12)
        ax2.set_title('Average "Against" Claims by Essay Position', fontsize=12)
        
        # 3. First Against position distribution
        ax3 = axes[1, 0]
        first_against = self.df[self.df['first_against_position'] >= 0]['first_against_position']
        if len(first_against) > 0:
            ax3.hist(first_against, bins=20, edgecolor='black', alpha=0.7, color='#e74c3c')
            ax3.axvline(first_against.mean(), color='black', linestyle='--',
                       label=f'Mean: {first_against.mean():.2f}')
            ax3.set_xlabel('Normalized Position (0=start, 1=end)', fontsize=12)
            ax3.set_ylabel('Number of Essays', fontsize=12)
            ax3.set_title('Position of First "Against" Claim', fontsize=12)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No Against Claims', ha='center', va='center', fontsize=14)
        
        # 4. Stacked bar: For vs Against by position
        ax4 = axes[1, 1]
        positions = ['First Third', 'Middle Third', 'Last Third']
        for_vals = [self.df['for_in_first_third'].sum(), 
                   self.df['for_in_middle_third'].sum(), 
                   self.df['for_in_last_third'].sum()]
        against_vals = [self.df['against_in_first_third'].sum(),
                       self.df['against_in_middle_third'].sum(),
                       self.df['against_in_last_third'].sum()]
        
        x = np.arange(len(positions))
        width = 0.35
        ax4.bar(x - width/2, for_vals, width, label='For', color='#2ecc71', edgecolor='black')
        ax4.bar(x + width/2, against_vals, width, label='Against', color='#e74c3c', edgecolor='black')
        ax4.set_xlabel('Essay Position', fontsize=12)
        ax4.set_ylabel('Total Count', fontsize=12)
        ax4.set_title('Total For vs Against Claims by Essay Position', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels(positions)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'positional_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved positional_analysis.png")
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap of key features."""
        # Select numerical columns for correlation
        numerical_cols = [
            'for_ratio', 'against_ratio', 'stance_balance',
            'total_components', 'major_claim_count', 'claim_count', 'premise_count',
            'evidence_density', 'total_relations', 'support_count', 'attack_count',
            'attack_ratio', 'relations_per_component',
            'for_in_first_third', 'for_in_middle_third', 'for_in_last_third',
            'against_in_first_third', 'against_in_middle_third', 'against_in_last_third'
        ]
        
        # Filter to columns that exist
        available_cols = [c for c in numerical_cols if c in self.df.columns]
        corr_matrix = self.df[available_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=ax, square=True, linewidths=0.5,
                   cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved correlation_heatmap.png")
    
    def plot_stance_by_structure(self):
        """Analyze relationship between stance and structural features."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. For ratio vs Evidence density
        ax1 = axes[0, 0]
        ax1.scatter(self.df['evidence_density'], self.df['for_ratio'], 
                   alpha=0.5, c='steelblue', edgecolors='black', s=50)
        z = np.polyfit(self.df['evidence_density'], self.df['for_ratio'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(self.df['evidence_density'].min(), self.df['evidence_density'].max(), 100)
        ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
        ax1.set_xlabel('Evidence Density', fontsize=12)
        ax1.set_ylabel('For Ratio', fontsize=12)
        ax1.set_title('For Ratio vs Evidence Density', fontsize=12)
        ax1.legend()
        
        # 2. Attack ratio vs For ratio
        ax2 = axes[0, 1]
        ax2.scatter(self.df['attack_ratio'], self.df['for_ratio'],
                   alpha=0.5, c='coral', edgecolors='black', s=50)
        ax2.set_xlabel('Attack Ratio', fontsize=12)
        ax2.set_ylabel('For Ratio', fontsize=12)
        ax2.set_title('For Ratio vs Attack Ratio', fontsize=12)
        
        # 3. Boxplot of evidence density by stance category
        ax3 = axes[1, 0]
        categories = ['mostly_for', 'balanced', 'mostly_against']
        cat_data = [self.df[self.df['stance_category'] == cat]['evidence_density'] for cat in categories]
        cat_data = [d for d in cat_data if len(d) > 0]  # Filter empty
        cat_labels = [cat for cat, d in zip(categories, [self.df[self.df['stance_category'] == c] for c in categories]) if len(d) > 0]
        if cat_data:
            bp = ax3.boxplot(cat_data, labels=cat_labels, patch_artist=True)
            colors = ['#2ecc71', '#3498db', '#e74c3c']
            for patch, color in zip(bp['boxes'], colors[:len(cat_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax3.set_ylabel('Evidence Density', fontsize=12)
        ax3.set_title('Evidence Density by Stance Category', fontsize=12)
        
        # 4. Boxplot of total components by stance category
        ax4 = axes[1, 1]
        cat_data = [self.df[self.df['stance_category'] == cat]['total_components'] for cat in categories]
        cat_data = [d for d in cat_data if len(d) > 0]
        if cat_data:
            bp = ax4.boxplot(cat_data, labels=cat_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(cat_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax4.set_ylabel('Total Components', fontsize=12)
        ax4.set_title('Total Components by Stance Category', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stance_by_structure.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved stance_by_structure.png")
    
    def generate_summary_stats(self) -> pd.DataFrame:
        """Generate summary statistics table."""
        numerical_cols = [
            'for_count', 'against_count', 'for_ratio', 'stance_balance',
            'total_components', 'claim_count', 'premise_count',
            'evidence_density', 'support_count', 'attack_count', 'attack_ratio',
            'essay_length'
        ]
        
        available_cols = [c for c in numerical_cols if c in self.df.columns]
        stats = self.df[available_cols].describe().T
        stats['median'] = self.df[available_cols].median()
        stats = stats[['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max']]
        
        return stats
    
    def run_full_eda(self):
        """Run complete EDA analysis."""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60 + "\n")
        
        # Generate plots
        print("Generating visualizations...")
        self.plot_stance_distribution()
        self.plot_component_structure()
        self.plot_positional_analysis()
        self.plot_correlation_heatmap()
        self.plot_stance_by_structure()
        
        # Generate summary stats
        print("\nGenerating summary statistics...")
        stats = self.generate_summary_stats()
        print("\n=== Summary Statistics ===")
        print(stats.round(3))
        
        # Save stats to CSV
        stats.to_csv(self.output_dir / 'summary_statistics.csv')
        print(f"\nSaved summary_statistics.csv")
        
        # Category distribution
        print("\n=== Stance Category Distribution ===")
        print(self.df['stance_category'].value_counts())
        
        # Key insights
        print("\n=== Key Insights ===")
        print(f"1. Average For Ratio: {self.df['for_ratio'].mean():.2%}")
        print(f"2. Essays with any Against claims: {(self.df['against_count'] > 0).sum()} / {len(self.df)}")
        print(f"3. Average Evidence Density: {self.df['evidence_density'].mean():.2f} premises per claim")
        print(f"4. Essays with Attack relations: {(self.df['attack_count'] > 0).sum()} / {len(self.df)}")
        print(f"5. Most common stance category: {self.df['stance_category'].mode()[0]}")
        
        return stats


def main():
    """Run EDA on the essay features."""
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
    
    # Add split information
    essay_splits = {eid: essay.split for eid, essay in builder.essays.items()}
    df['split'] = df['essay_id'].map(essay_splits)
    
    print(f"\nDataset shape: {df.shape}")
    
    # Run EDA
    eda = EDA(df, str(output_dir))
    eda.run_full_eda()
    
    print(f"\n✓ All figures saved to {output_dir}")


if __name__ == "__main__":
    main()

