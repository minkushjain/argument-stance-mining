"""
Traditional ML Baseline for Essay-Level Stance Prediction

Predicts essay-level characteristics from raw essay text:
- for_ratio (regression)
- stance_category (classification)
- evidence_density (regression)

Uses TF-IDF features with various classical ML models.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset_builder import DatasetBuilder
from src.analysis.essay_features import EssayFeatureExtractor


class EssayPredictorBaseline:
    """
    Traditional ML baseline for predicting essay-level stance features.
    """
    
    def __init__(self, output_dir: str = "models"):
        """
        Initialize the predictor.
        
        Args:
            output_dir: Directory to save models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Vectorizers - improved settings
        self.tfidf = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),  # Include trigrams for better phrase capture
            min_df=3,
            max_df=0.90,
            stop_words='english',
            sublinear_tf=True  # Apply sublinear tf scaling
        )
        
        # Models
        self.regression_models = {}
        self.classification_models = {}
        
        # Label encoder for classification
        self.label_encoder = LabelEncoder()
        
        # Results storage
        self.results = {}
    
    def prepare_data(
        self, 
        train_essays: List, 
        val_essays: List, 
        test_essays: List,
        train_features: pd.DataFrame,
        val_features: pd.DataFrame,
        test_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Prepare data for training.
        
        Returns:
            Dictionary with train/val/test splits for text and targets
        """
        # Get essay texts
        X_train_text = [e.text for e in train_essays]
        X_val_text = [e.text for e in val_essays]
        X_test_text = [e.text for e in test_essays]
        
        # Fit TF-IDF on training data
        X_train_tfidf = self.tfidf.fit_transform(X_train_text)
        X_val_tfidf = self.tfidf.transform(X_val_text)
        X_test_tfidf = self.tfidf.transform(X_test_text)
        
        # Get targets
        # Regression targets
        y_train_for_ratio = train_features['for_ratio'].values
        y_val_for_ratio = val_features['for_ratio'].values
        y_test_for_ratio = test_features['for_ratio'].values
        
        y_train_evidence = train_features['evidence_density'].values
        y_val_evidence = val_features['evidence_density'].values
        y_test_evidence = test_features['evidence_density'].values
        
        # Classification target
        y_train_category = train_features['stance_category'].values
        y_val_category = val_features['stance_category'].values
        y_test_category = test_features['stance_category'].values
        
        # Fit label encoder on all categories
        all_categories = np.concatenate([y_train_category, y_val_category, y_test_category])
        self.label_encoder.fit(all_categories)
        
        y_train_category_enc = self.label_encoder.transform(y_train_category)
        y_val_category_enc = self.label_encoder.transform(y_val_category)
        y_test_category_enc = self.label_encoder.transform(y_test_category)
        
        return {
            'X_train': X_train_tfidf,
            'X_val': X_val_tfidf,
            'X_test': X_test_tfidf,
            'y_train_for_ratio': y_train_for_ratio,
            'y_val_for_ratio': y_val_for_ratio,
            'y_test_for_ratio': y_test_for_ratio,
            'y_train_evidence': y_train_evidence,
            'y_val_evidence': y_val_evidence,
            'y_test_evidence': y_test_evidence,
            'y_train_category': y_train_category_enc,
            'y_val_category': y_val_category_enc,
            'y_test_category': y_test_category_enc,
            'category_names': self.label_encoder.classes_
        }
    
    def train_regression_models(
        self,
        X_train, y_train,
        X_val, y_val,
        target_name: str
    ) -> Dict[str, Any]:
        """
        Train regression models for a target variable.
        
        Returns:
            Dictionary with trained models and results
        """
        models = {
            'Ridge': Ridge(alpha=0.5),  # Lower regularization
            'RandomForest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15,
                min_samples_split=5,
                random_state=42, 
                n_jobs=-1
            ),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        }
        
        results = {}
        
        print(f"\n--- Training Regression Models for {target_name} ---")
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_val_pred = model.predict(X_val)
            
            # Metrics
            mae = mean_absolute_error(y_val, y_val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            r2 = r2_score(y_val, y_val_pred)
            
            results[name] = {
                'model': model,
                'val_mae': mae,
                'val_rmse': rmse,
                'val_r2': r2,
                'val_predictions': y_val_pred
            }
            
            print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Store best model
        best_model_name = min(results, key=lambda x: results[x]['val_mae'])
        self.regression_models[target_name] = results[best_model_name]['model']
        
        print(f"\nBest model for {target_name}: {best_model_name}")
        
        return results
    
    def train_classification_models(
        self,
        X_train, y_train,
        X_val, y_val,
        target_name: str = 'stance_category'
    ) -> Dict[str, Any]:
        """
        Train classification models for stance category.
        
        Returns:
            Dictionary with trained models and results
        """
        models = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000, 
                C=0.5,  # Stronger regularization
                random_state=42, 
                class_weight='balanced'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=20,
                min_samples_split=5,
                random_state=42, 
                class_weight='balanced', 
                n_jobs=-1
            ),
            'SVC': SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42),
        }
        
        results = {}
        
        print(f"\n--- Training Classification Models for {target_name} ---")
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_val_pred = model.predict(X_val)
            
            # Metrics
            accuracy = accuracy_score(y_val, y_val_pred)
            macro_f1 = f1_score(y_val, y_val_pred, average='macro')
            
            results[name] = {
                'model': model,
                'val_accuracy': accuracy,
                'val_macro_f1': macro_f1,
                'val_predictions': y_val_pred
            }
            
            print(f"  Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}")
        
        # Store best model
        best_model_name = max(results, key=lambda x: results[x]['val_macro_f1'])
        self.classification_models[target_name] = results[best_model_name]['model']
        
        print(f"\nBest model for {target_name}: {best_model_name}")
        
        return results
    
    def evaluate_on_test(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all models on test set.
        
        Returns:
            Dictionary with test results
        """
        test_results = {}
        
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        
        # Regression: for_ratio
        if 'for_ratio' in self.regression_models:
            model = self.regression_models['for_ratio']
            y_pred = model.predict(data['X_test'])
            y_true = data['y_test_for_ratio']
            
            test_results['for_ratio'] = {
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred),
                'predictions': y_pred,
                'true_values': y_true
            }
            
            print(f"\nFor Ratio (Regression):")
            print(f"  MAE: {test_results['for_ratio']['mae']:.4f}")
            print(f"  RMSE: {test_results['for_ratio']['rmse']:.4f}")
            print(f"  R²: {test_results['for_ratio']['r2']:.4f}")
        
        # Regression: evidence_density
        if 'evidence_density' in self.regression_models:
            model = self.regression_models['evidence_density']
            y_pred = model.predict(data['X_test'])
            y_true = data['y_test_evidence']
            
            test_results['evidence_density'] = {
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred),
                'predictions': y_pred,
                'true_values': y_true
            }
            
            print(f"\nEvidence Density (Regression):")
            print(f"  MAE: {test_results['evidence_density']['mae']:.4f}")
            print(f"  RMSE: {test_results['evidence_density']['rmse']:.4f}")
            print(f"  R²: {test_results['evidence_density']['r2']:.4f}")
        
        # Classification: stance_category
        if 'stance_category' in self.classification_models:
            model = self.classification_models['stance_category']
            y_pred = model.predict(data['X_test'])
            y_true = data['y_test_category']
            
            # Get unique labels in test set
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            label_names = [data['category_names'][i] for i in unique_labels]
            
            test_results['stance_category'] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'macro_f1': f1_score(y_true, y_pred, average='macro'),
                'classification_report': classification_report(
                    y_true, y_pred, 
                    labels=unique_labels,
                    target_names=label_names,
                    output_dict=True
                ),
                'confusion_matrix': confusion_matrix(y_true, y_pred, labels=unique_labels),
                'predictions': y_pred,
                'true_values': y_true,
                'label_names': label_names
            }
            
            print(f"\nStance Category (Classification):")
            print(f"  Accuracy: {test_results['stance_category']['accuracy']:.4f}")
            print(f"  Macro-F1: {test_results['stance_category']['macro_f1']:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(y_true, y_pred, labels=unique_labels, target_names=label_names))
        
        return test_results
    
    def plot_results(self, test_results: Dict, category_names: List[str], output_dir: str):
        """Create visualization of results."""
        output_dir = Path(output_dir)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. For Ratio: Predicted vs Actual
        if 'for_ratio' in test_results:
            ax1 = axes[0, 0]
            y_true = test_results['for_ratio']['true_values']
            y_pred = test_results['for_ratio']['predictions']
            
            ax1.scatter(y_true, y_pred, alpha=0.5, edgecolors='black')
            ax1.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
            ax1.set_xlabel('True For Ratio', fontsize=12)
            ax1.set_ylabel('Predicted For Ratio', fontsize=12)
            ax1.set_title(f"For Ratio: Predicted vs Actual\n(R²={test_results['for_ratio']['r2']:.3f})", fontsize=12)
            ax1.legend()
        
        # 2. Evidence Density: Predicted vs Actual
        if 'evidence_density' in test_results:
            ax2 = axes[0, 1]
            y_true = test_results['evidence_density']['true_values']
            y_pred = test_results['evidence_density']['predictions']
            
            ax2.scatter(y_true, y_pred, alpha=0.5, edgecolors='black', color='green')
            max_val = max(y_true.max(), y_pred.max())
            ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
            ax2.set_xlabel('True Evidence Density', fontsize=12)
            ax2.set_ylabel('Predicted Evidence Density', fontsize=12)
            ax2.set_title(f"Evidence Density: Predicted vs Actual\n(R²={test_results['evidence_density']['r2']:.3f})", fontsize=12)
            ax2.legend()
        
        # 3. Confusion Matrix
        if 'stance_category' in test_results:
            ax3 = axes[1, 0]
            cm = test_results['stance_category']['confusion_matrix']
            label_names = test_results['stance_category'].get('label_names', category_names)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                       xticklabels=label_names, yticklabels=label_names)
            ax3.set_xlabel('Predicted', fontsize=12)
            ax3.set_ylabel('True', fontsize=12)
            ax3.set_title('Stance Category Confusion Matrix', fontsize=12)
        
        # 4. Per-class F1 scores
        if 'stance_category' in test_results:
            ax4 = axes[1, 1]
            report = test_results['stance_category']['classification_report']
            label_names = test_results['stance_category'].get('label_names', category_names)
            
            classes = [c for c in label_names if c in report]
            f1_scores = [report[c]['f1-score'] for c in classes]
            
            bars = ax4.bar(classes, f1_scores, color=['#2ecc71', '#3498db', '#e74c3c'][:len(classes)],
                          edgecolor='black')
            ax4.set_xlabel('Stance Category', fontsize=12)
            ax4.set_ylabel('F1 Score', fontsize=12)
            ax4.set_title('Per-class F1 Scores', fontsize=12)
            ax4.set_ylim(0, 1)
            
            # Add value labels
            for bar, score in zip(bars, f1_scores):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{score:.3f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'baseline_model_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved baseline_model_results.png")
    
    def save_models(self):
        """Save trained models to disk."""
        for name, model in self.regression_models.items():
            path = self.output_dir / f'regression_{name}.pkl'
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {path}")
        
        for name, model in self.classification_models.items():
            path = self.output_dir / f'classification_{name}.pkl'
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {path}")
        
        # Save vectorizer
        with open(self.output_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf, f)
        print(f"Saved tfidf_vectorizer.pkl")
        
        # Save label encoder
        with open(self.output_dir / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Saved label_encoder.pkl")


def main():
    """Train and evaluate baseline models."""
    dataset_dir = Path(__file__).parent.parent.parent / "dataset" / "ArgumentAnnotatedEssays-2.0"
    output_dir = Path(__file__).parent.parent.parent / "models"
    figures_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
    
    print("Loading dataset...")
    builder = DatasetBuilder(str(dataset_dir))
    builder.build()
    
    # Get essays by split
    train_essays = builder.get_essays_by_split('train')
    val_essays = builder.get_essays_by_split('val')
    test_essays = builder.get_essays_by_split('test')
    
    print(f"\nSplit sizes: Train={len(train_essays)}, Val={len(val_essays)}, Test={len(test_essays)}")
    
    # Extract features
    print("\nExtracting features...")
    extractor = EssayFeatureExtractor()
    
    train_features = extractor.to_dataframe(extractor.extract_all(train_essays))
    val_features = extractor.to_dataframe(extractor.extract_all(val_essays))
    test_features = extractor.to_dataframe(extractor.extract_all(test_essays))
    
    # Initialize predictor
    predictor = EssayPredictorBaseline(str(output_dir))
    
    # Prepare data
    print("\nPreparing data...")
    data = predictor.prepare_data(
        train_essays, val_essays, test_essays,
        train_features, val_features, test_features
    )
    
    print(f"TF-IDF features: {data['X_train'].shape[1]}")
    print(f"Category classes: {list(data['category_names'])}")
    
    # Train models
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    # Train for_ratio regression
    predictor.train_regression_models(
        data['X_train'], data['y_train_for_ratio'],
        data['X_val'], data['y_val_for_ratio'],
        'for_ratio'
    )
    
    # Train evidence_density regression
    predictor.train_regression_models(
        data['X_train'], data['y_train_evidence'],
        data['X_val'], data['y_val_evidence'],
        'evidence_density'
    )
    
    # Train stance_category classification
    predictor.train_classification_models(
        data['X_train'], data['y_train_category'],
        data['X_val'], data['y_val_category'],
        'stance_category'
    )
    
    # Evaluate on test set
    test_results = predictor.evaluate_on_test(data)
    
    # Plot results
    predictor.plot_results(test_results, list(data['category_names']), str(figures_dir))
    
    # Save models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    predictor.save_models()
    
    # Save results summary
    results_summary = {
        'for_ratio': {
            'mae': float(test_results['for_ratio']['mae']),
            'rmse': float(test_results['for_ratio']['rmse']),
            'r2': float(test_results['for_ratio']['r2'])
        },
        'evidence_density': {
            'mae': float(test_results['evidence_density']['mae']),
            'rmse': float(test_results['evidence_density']['rmse']),
            'r2': float(test_results['evidence_density']['r2'])
        },
        'stance_category': {
            'accuracy': float(test_results['stance_category']['accuracy']),
            'macro_f1': float(test_results['stance_category']['macro_f1'])
        }
    }
    
    with open(figures_dir.parent / 'baseline_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved baseline_results.json")
    
    print("\n✓ Baseline model training complete!")


if __name__ == "__main__":
    main()

