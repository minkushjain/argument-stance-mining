"""
Inference Script for Essay Stance Structure Analysis

Run predictions on test essays or new essays using trained models.

Usage:
    python -m src.inference --essay_id essay001
    python -m src.inference --essay_id essay005 --model baseline
    python -m src.inference --test_sample 5
    python -m src.inference --essay_id essay004 --attribution    # Sentence importance
    python -m src.inference --essay_id essay004 --rhetorical     # Rhetorical roles
    python -m src.inference --essay_id essay004 --full           # All analyses
"""

import sys
from pathlib import Path
import argparse
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_builder import DatasetBuilder
from src.analysis.essay_features import EssayFeatureExtractor
from src.analysis.sentence_attribution import SentenceAttributionAnalyzer
from src.models.rhetorical_classifier import RhetoricalRoleClassifier
from src.data.rhetorical_labels import RhetoricalRole


class EssayPredictor:
    """Load trained models and run inference on essays."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.feature_extractor = EssayFeatureExtractor()
        
        # Set device
        self.device = torch.device(
            'mps' if torch.backends.mps.is_available() else 
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Load models
        self._load_baseline_models()
        self._load_transformer_models()
        
        # Initialize sentence attribution analyzer
        self.attribution_analyzer = SentenceAttributionAnalyzer(
            self.transformer_regressor, 
            self.tokenizer, 
            self.device
        )
        
        # Load rhetorical role classifier
        self._load_rhetorical_classifier()
        
        print("✓ All models loaded successfully\n")
    
    def _load_rhetorical_classifier(self):
        """Load the rhetorical role classifier (V2 improved model)."""
        print("Loading rhetorical role classifier...")
        
        # Try V2 model first, fall back to V1
        rhetorical_model_path = self.models_dir / 'rhetorical_classifier_v2_best.pt'
        if not rhetorical_model_path.exists():
            rhetorical_model_path = self.models_dir / 'rhetorical_classifier_best.pt'
        
        if not rhetorical_model_path.exists():
            print("  ⚠ Rhetorical classifier not found. Run training first.")
            self.rhetorical_classifier = None
            return
        
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Load tokenizer and model
        self.rhetorical_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.rhetorical_model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=5
        )
        self.rhetorical_model.load_state_dict(
            torch.load(rhetorical_model_path, map_location=self.device, weights_only=True)
        )
        self.rhetorical_model.to(self.device)
        self.rhetorical_model.eval()
        
        self.rhetorical_classifier = True
        self.rhetorical_v2 = 'v2' in str(rhetorical_model_path)
        print(f"  ✓ Rhetorical classifier loaded ({'V2 improved' if self.rhetorical_v2 else 'V1'})")
    
    def _load_baseline_models(self):
        """Load baseline TF-IDF and sklearn models."""
        print("Loading baseline models...")
        
        # Load TF-IDF vectorizer
        with open(self.models_dir / 'tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf = pickle.load(f)
        
        # Load regression model (for_ratio)
        with open(self.models_dir / 'regression_for_ratio.pkl', 'rb') as f:
            self.baseline_regressor = pickle.load(f)
        
        # Load classification model (stance_category)
        with open(self.models_dir / 'classification_stance_category.pkl', 'rb') as f:
            self.baseline_classifier = pickle.load(f)
        
        # Load label encoder
        with open(self.models_dir / 'label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print("  ✓ Baseline models loaded")
    
    def _load_transformer_models(self):
        """Load transformer models."""
        print("Loading transformer models...")
        
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from src.models.essay_predictor_transformer import EssayTransformerRegressor
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Load regression model
        self.transformer_regressor = EssayTransformerRegressor('distilbert-base-uncased')
        self.transformer_regressor.load_state_dict(
            torch.load(self.models_dir / 'transformer_regression_best.pt', 
                      map_location=self.device, weights_only=True)
        )
        self.transformer_regressor.to(self.device)
        self.transformer_regressor.eval()
        
        # Load classification model
        self.transformer_classifier = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=3
        )
        self.transformer_classifier.load_state_dict(
            torch.load(self.models_dir / 'transformer_classification_best.pt',
                      map_location=self.device, weights_only=True)
        )
        self.transformer_classifier.to(self.device)
        self.transformer_classifier.eval()
        
        print("  ✓ Transformer models loaded")
    
    def get_ground_truth(self, essay):
        """Extract ground truth features from essay annotations."""
        features = self.feature_extractor.extract_features(essay)
        return {
            'for_ratio': features.for_ratio,
            'against_ratio': features.against_ratio,
            'stance_category': features.stance_category,
            'for_count': features.for_count,
            'against_count': features.against_count,
            'total_claims': features.claim_count,
            'total_premises': features.premise_count,
            'evidence_density': features.evidence_density,
            'attack_ratio': features.attack_ratio,
            'support_ratio': features.support_ratio,
            'tree_depth': features.tree_depth,
        }
    
    def predict_baseline(self, essay):
        """Generate predictions using baseline models."""
        # Transform text with TF-IDF
        X = self.tfidf.transform([essay.text])
        
        # Predict for_ratio (regression)
        for_ratio_pred = self.baseline_regressor.predict(X)[0]
        for_ratio_pred = np.clip(for_ratio_pred, 0.0, 1.0)  # Ensure valid range
        
        # Predict stance_category (classification)
        category_idx = self.baseline_classifier.predict(X)[0]
        category_pred = self.label_encoder.inverse_transform([category_idx])[0]
        
        # Get prediction probabilities (if available)
        category_probs = {}
        if hasattr(self.baseline_classifier, 'predict_proba'):
            try:
                probs = self.baseline_classifier.predict_proba(X)[0]
                category_probs = {
                    cls: float(prob) 
                    for cls, prob in zip(self.label_encoder.classes_, probs)
                }
            except AttributeError:
                category_probs = {cls: 1.0 if cls == category_pred else 0.0 
                                 for cls in self.label_encoder.classes_}
        else:
            # SVC without probability - just show prediction
            category_probs = {cls: 1.0 if cls == category_pred else 0.0 
                             for cls in self.label_encoder.classes_}
        
        return {
            'for_ratio': float(for_ratio_pred),
            'stance_category': category_pred,
            'category_probabilities': category_probs
        }
    
    def predict_transformer(self, essay):
        """Generate predictions using transformer models."""
        # Prepare text (include prompt if available)
        if essay.prompt:
            text = f"{essay.prompt} [SEP] {essay.text}"
        else:
            text = essay.text
        
        # Tokenize
        encoding = self.tokenizer(
            text, 
            max_length=512, 
            padding='max_length',
            truncation=True, 
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict for_ratio (regression)
        with torch.no_grad():
            for_ratio_pred = self.transformer_regressor(input_ids, attention_mask)
            for_ratio_pred = for_ratio_pred.cpu().numpy()[0]
            for_ratio_pred = np.clip(for_ratio_pred, 0.0, 1.0)
        
        # Predict stance_category (classification)
        with torch.no_grad():
            outputs = self.transformer_classifier(
                input_ids=input_ids, attention_mask=attention_mask
            )
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            category_idx = torch.argmax(logits).cpu().numpy()
        
        # Map category index to name
        category_map = {0: 'balanced', 1: 'mostly_for', 2: 'mostly_against'}
        category_pred = category_map[int(category_idx)]
        
        return {
            'for_ratio': float(for_ratio_pred),
            'stance_category': category_pred,
            'category_probabilities': {
                'balanced': float(probs[0]),
                'mostly_for': float(probs[1]),
                'mostly_against': float(probs[2])
            }
        }
    
    def analyze_sentence_importance(self, essay, method: str = 'attention'):
        """
        Analyze which sentences are most important for the prediction.
        
        Args:
            essay: Essay object
            method: 'attention' or 'gradient'
            
        Returns:
            Attribution analysis result dict
        """
        return self.attribution_analyzer.analyze_essay(
            essay.text, 
            essay.prompt or "", 
            method=method
        )
    
    def predict_rhetorical_role(self, text: str, prompt: str = "", 
                                 component_type: str = "Claim", position: str = "middle"):
        """
        Predict rhetorical role for a component text.
        
        Args:
            text: Component text
            prompt: Essay prompt for context
            component_type: MajorClaim, Claim, or Premise (for V2 model)
            position: beginning, middle, or end (for V2 model)
        
        Returns:
            Tuple of (role_name, confidence, all_probabilities)
        """
        if not self.rhetorical_classifier:
            return None, 0.0, {}
        
        # Prepare input based on model version
        if hasattr(self, 'rhetorical_v2') and self.rhetorical_v2:
            # V2 enhanced format
            base_text = f"{prompt} [SEP] {text}" if prompt else text
            full_text = f"[TYPE: {component_type}] [POSITION: {position}] {base_text}"
            max_len = 160
        else:
            # V1 format
            full_text = f"{prompt} [SEP] {text}" if prompt else text
            max_len = 128
        
        encoding = self.rhetorical_tokenizer(
            full_text,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.rhetorical_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
        
        role_names = RhetoricalRole.names()
        predicted_role = role_names[pred_idx]
        confidence = probs[pred_idx]
        all_probs = dict(zip(role_names, probs.tolist()))
        
        return predicted_role, confidence, all_probs
    
    def analyze_component_roles(self, essay):
        """
        Analyze rhetorical roles for all components in an essay.
        
        Returns:
            List of component analysis results
        """
        if not self.rhetorical_classifier:
            return []
        
        results = []
        essay_length = len(essay.text)
        
        for comp_id, component in essay.components.items():
            # Compute position in essay
            relative_pos = component.start / essay_length if essay_length > 0 else 0.5
            if relative_pos < 0.33:
                position = "beginning"
            elif relative_pos < 0.67:
                position = "middle"
            else:
                position = "end"
            
            role, confidence, probs = self.predict_rhetorical_role(
                component.text, 
                essay.prompt or "",
                component_type=component.type,
                position=position
            )
            
            # Determine ground truth from annotations
            if component.type == 'MajorClaim':
                gt_role = 'THESIS'
            elif component.type == 'Premise':
                gt_role = 'EVIDENCE'
            elif component.stance == 'Against':
                gt_role = 'COUNTER_ARGUMENT'
            else:
                gt_role = 'MAIN_ARGUMENT'
            
            results.append({
                'id': comp_id,
                'text': component.text[:80] + "..." if len(component.text) > 80 else component.text,
                'type': component.type,
                'stance': component.stance,
                'position': position,
                'ground_truth': gt_role,
                'predicted': role,
                'confidence': confidence,
                'correct': role == gt_role
            })
        
        return results
    
    def analyze_essay(self, essay, show_text: bool = False, show_attribution: bool = False, 
                      show_rhetorical: bool = False):
        """Run complete analysis on a single essay."""
        print(f"\n{'='*70}")
        print(f"ESSAY ANALYSIS: {essay.essay_id}")
        print(f"{'='*70}")
        
        # Basic info
        print(f"\n📝 ESSAY INFO")
        print(f"   Prompt: {essay.prompt[:80]}..." if essay.prompt else "   Prompt: N/A")
        print(f"   Text length: {len(essay.text)} characters, ~{len(essay.text.split())} words")
        print(f"   Split: {essay.split}")
        
        if show_text:
            print(f"\n   First 300 chars: {essay.text[:300]}...")
        
        # Ground truth (from annotations)
        gt = self.get_ground_truth(essay)
        print(f"\n📊 GROUND TRUTH (from BRAT annotations)")
        print(f"   For Claims: {gt['for_count']}")
        print(f"   Against Claims: {gt['against_count']}")
        print(f"   Total Claims: {gt['total_claims']}")
        print(f"   For Ratio: {gt['for_ratio']:.3f}")
        print(f"   Stance Category: {gt['stance_category']}")
        print(f"   Evidence Density: {gt['evidence_density']:.2f} premises/claim")
        print(f"   Attack Ratio: {gt['attack_ratio']:.3f}")
        print(f"   Argument Tree Depth: {gt['tree_depth']}")
        
        # Baseline predictions
        baseline_pred = self.predict_baseline(essay)
        baseline_for_error = abs(baseline_pred['for_ratio'] - gt['for_ratio'])
        baseline_cat_match = '✓' if baseline_pred['stance_category'] == gt['stance_category'] else '✗'
        
        print(f"\n🤖 BASELINE MODEL PREDICTIONS (TF-IDF + SVC/SVR)")
        print(f"   For Ratio: {baseline_pred['for_ratio']:.3f} (error: {baseline_for_error:.3f})")
        print(f"   Stance Category: {baseline_pred['stance_category']} {baseline_cat_match}")
        print(f"   Category Probabilities: {baseline_pred['category_probabilities']}")
        
        # Transformer predictions
        transformer_pred = self.predict_transformer(essay)
        transformer_for_error = abs(transformer_pred['for_ratio'] - gt['for_ratio'])
        transformer_cat_match = '✓' if transformer_pred['stance_category'] == gt['stance_category'] else '✗'
        
        print(f"\n🧠 TRANSFORMER PREDICTIONS (DistilBERT)")
        print(f"   For Ratio: {transformer_pred['for_ratio']:.3f} (error: {transformer_for_error:.3f})")
        print(f"   Stance Category: {transformer_pred['stance_category']} {transformer_cat_match}")
        print(f"   Category Probabilities: {transformer_pred['category_probabilities']}")
        
        # Comparison
        print(f"\n📈 MODEL COMPARISON")
        print(f"   For Ratio Error - Baseline: {baseline_for_error:.3f}, Transformer: {transformer_for_error:.3f}")
        better_regressor = "Baseline" if baseline_for_error < transformer_for_error else "Transformer"
        print(f"   Better for regression: {better_regressor}")
        
        # Sentence Attribution Analysis (optional)
        attribution_result = None
        if show_attribution:
            print(f"\n🔍 SENTENCE IMPORTANCE ANALYSIS")
            print(f"   (Which sentences drive the transformer's prediction?)")
            
            attribution_result = self.analyze_sentence_importance(essay)
            
            print(f"\n   Top 5 Most Important Sentences:")
            for item in attribution_result['most_important']:
                score = item['score']
                sent = item['sentence']
                idx = item['index'] + 1
                # Truncate long sentences
                display_sent = sent[:70] + "..." if len(sent) > 70 else sent
                print(f"   {idx}. [{score:.3f}] \"{display_sent}\"")
            
            print(f"\n   Least Important Sentences:")
            for item in attribution_result['least_important']:
                score = item['score']
                sent = item['sentence']
                idx = item['index'] + 1
                display_sent = sent[:70] + "..." if len(sent) > 70 else sent
                print(f"   {idx}. [{score:.3f}] \"{display_sent}\"")
            
            # Save HTML visualization
            html = self.attribution_analyzer.generate_html_visualization(
                essay.text, attribution_result['sentences']
            )
            output_path = Path(f'reports/attribution_{essay.essay_id}.html')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(f"""
                <html>
                <head>
                    <title>Sentence Attribution: {essay.essay_id}</title>
                    <style>
                        body {{ font-family: Georgia, serif; max-width: 900px; margin: 40px auto; padding: 20px; }}
                        h1 {{ color: #1a365d; }}
                        .stats {{ background: #f7fafc; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                        .legend {{ display: flex; align-items: center; gap: 20px; margin: 15px 0; }}
                        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
                        .legend-box {{ width: 20px; height: 20px; border-radius: 3px; }}
                    </style>
                </head>
                <body>
                    <h1>📊 Sentence Attribution Analysis: {essay.essay_id}</h1>
                    
                    <div class="stats">
                        <p><strong>Predicted For Ratio:</strong> {attribution_result['prediction']:.3f}</p>
                        <p><strong>Actual For Ratio:</strong> {gt['for_ratio']:.3f}</p>
                        <p><strong>Prediction Error:</strong> {abs(attribution_result['prediction'] - gt['for_ratio']):.3f}</p>
                        <p><strong>Total Sentences:</strong> {attribution_result['num_sentences']}</p>
                    </div>
                    
                    <div class="legend">
                        <span><strong>Legend:</strong></span>
                        <div class="legend-item">
                            <div class="legend-box" style="background: rgba(59, 130, 246, 0.7);"></div>
                            <span>High importance</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-box" style="background: rgba(59, 130, 246, 0.3);"></div>
                            <span>Medium importance</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-box" style="background: rgba(59, 130, 246, 0.1);"></div>
                            <span>Low importance</span>
                        </div>
                    </div>
                    
                    <h2>Essay Text (hover for importance scores)</h2>
                    {html}
                    
                    <h2>Prompt</h2>
                    <p><em>{essay.prompt or 'N/A'}</em></p>
                </body>
                </html>
                """)
            
            print(f"\n   ✓ HTML visualization saved to {output_path}")
        
        # Rhetorical Role Analysis (optional)
        rhetorical_result = None
        if show_rhetorical and self.rhetorical_classifier:
            print(f"\n🎭 RHETORICAL ROLE ANALYSIS")
            print(f"   (What role does each component play in the argument?)")
            
            rhetorical_result = self.analyze_component_roles(essay)
            
            # Group by role
            role_counts = {}
            for comp in rhetorical_result:
                pred = comp['predicted']
                role_counts[pred] = role_counts.get(pred, 0) + 1
            
            print(f"\n   Predicted Role Distribution:")
            for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
                print(f"      {role:20s}: {count}")
            
            # Show accuracy
            correct = sum(1 for c in rhetorical_result if c['correct'])
            total = len(rhetorical_result)
            print(f"\n   Prediction Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
            
            # Show sample predictions
            print(f"\n   Sample Component Predictions:")
            shown = {'THESIS': 0, 'MAIN_ARGUMENT': 0, 'COUNTER_ARGUMENT': 0, 'EVIDENCE': 0}
            for comp in rhetorical_result:
                pred = comp['predicted']
                if shown.get(pred, 0) < 1:  # Show 1 of each type
                    match = '✓' if comp['correct'] else '✗'
                    text = comp['text'][:60] + "..." if len(comp['text']) > 60 else comp['text']
                    print(f"      [{pred:18s}] {match} \"{text}\"")
                    shown[pred] = shown.get(pred, 0) + 1
        
        return {
            'essay_id': essay.essay_id,
            'ground_truth': gt,
            'baseline': baseline_pred,
            'transformer': transformer_pred,
            'attribution': attribution_result,
            'rhetorical': rhetorical_result
        }


def main():
    parser = argparse.ArgumentParser(description='Test trained models on essays')
    parser.add_argument('--essay_id', type=str, help='Specific essay ID to test (e.g., essay001)')
    parser.add_argument('--test_sample', type=int, default=0, 
                       help='Number of test essays to sample (0 = all)')
    parser.add_argument('--model', choices=['baseline', 'transformer', 'both'], 
                       default='both', help='Which model to use')
    parser.add_argument('--show_text', action='store_true', 
                       help='Show essay text snippets')
    parser.add_argument('--attribution', action='store_true',
                       help='Show sentence-level importance analysis')
    parser.add_argument('--rhetorical', action='store_true',
                       help='Show rhetorical role analysis for components')
    parser.add_argument('--full', action='store_true',
                       help='Run all analyses (attribution + rhetorical)')
    
    args = parser.parse_args()
    
    # --full enables all analyses
    if args.full:
        args.attribution = True
        args.rhetorical = True
    
    # Load dataset
    print("Loading dataset...")
    dataset_dir = Path('dataset/ArgumentAnnotatedEssays-2.0')
    builder = DatasetBuilder(str(dataset_dir))
    builder.build()
    print(f"✓ Loaded {len(builder.essays)} essays\n")
    
    # Load predictor
    predictor = EssayPredictor()
    
    if args.essay_id:
        # Test specific essay
        essay = builder.essays.get(args.essay_id)
        if not essay:
            print(f"❌ Essay {args.essay_id} not found!")
            print(f"   Available essays: essay001 to essay{len(builder.essays):03d}")
            return
        
        predictor.analyze_essay(essay, show_text=args.show_text, show_attribution=args.attribution,
                               show_rhetorical=args.rhetorical)
        
    elif args.test_sample > 0:
        # Test sample of test essays
        test_essays = [e for e in builder.essays.values() if e.split == 'TEST']
        sample_size = min(args.test_sample, len(test_essays))
        
        print(f"\n{'='*70}")
        print(f"TESTING {sample_size} ESSAYS FROM TEST SPLIT")
        print(f"{'='*70}")
        
        results = []
        for i, essay in enumerate(test_essays[:sample_size]):
            result = predictor.analyze_essay(essay, show_text=args.show_text, show_attribution=args.attribution,
                                            show_rhetorical=args.rhetorical)
            results.append(result)
        
        # Summary statistics
        print(f"\n{'='*70}")
        print(f"SUMMARY ACROSS {sample_size} TEST ESSAYS")
        print(f"{'='*70}")
        
        baseline_errors = [abs(r['baseline']['for_ratio'] - r['ground_truth']['for_ratio']) 
                         for r in results]
        transformer_errors = [abs(r['transformer']['for_ratio'] - r['ground_truth']['for_ratio']) 
                            for r in results]
        
        baseline_correct = sum(1 for r in results 
                              if r['baseline']['stance_category'] == r['ground_truth']['stance_category'])
        transformer_correct = sum(1 for r in results 
                                 if r['transformer']['stance_category'] == r['ground_truth']['stance_category'])
        
        print(f"\nFor Ratio Prediction (MAE):")
        print(f"   Baseline:    {np.mean(baseline_errors):.3f}")
        print(f"   Transformer: {np.mean(transformer_errors):.3f}")
        
        print(f"\nStance Category Accuracy:")
        print(f"   Baseline:    {baseline_correct}/{sample_size} ({100*baseline_correct/sample_size:.1f}%)")
        print(f"   Transformer: {transformer_correct}/{sample_size} ({100*transformer_correct/sample_size:.1f}%)")
        
    else:
        # Default: show one example
        test_essays = [e for e in builder.essays.values() if e.split == 'TEST']
        if test_essays:
            print("No essay specified. Showing first test essay as example...")
            predictor.analyze_essay(test_essays[0], show_text=True, show_attribution=args.attribution,
                                   show_rhetorical=args.rhetorical)
            print(f"\n💡 TIP: Run with --essay_id <id> or --test_sample <n> for more options")
            print(f"   Add --attribution for sentence importance analysis")
        else:
            print("No test essays found!")


if __name__ == "__main__":
    main()

