"""
Sentence Attribution Analysis using Integrated Gradients

Uses the trained DistilBERT model to identify which sentences
contribute most to the predicted stance ratio.

This helps answer: "Which sentences in this essay are most important
for the model's prediction of the stance distribution?"
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SentenceAttributionAnalyzer:
    """
    Analyze which sentences contribute most to model predictions.
    
    Uses gradient-based attribution to identify important tokens,
    then aggregates to sentence-level importance scores.
    """
    
    def __init__(self, model, tokenizer, device=None):
        """
        Initialize the analyzer.
        
        Args:
            model: Trained EssayTransformerRegressor model
            tokenizer: HuggingFace tokenizer
            device: torch device (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        
        if device is None:
            self.device = torch.device(
                'mps' if torch.backends.mps.is_available() else
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
    
    def _compute_gradient_attribution(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> np.ndarray:
        """
        Compute gradient-based attribution scores for each token.
        
        Uses the gradient of the output with respect to input embeddings.
        """
        # Enable gradients for this computation
        self.model.eval()
        
        # Get embeddings
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Get the embedding layer
        embeddings = self.model.encoder.embeddings.word_embeddings(input_ids)
        embeddings.requires_grad_(True)
        
        # Forward pass through encoder using embeddings
        encoder_outputs = self.model.encoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
        
        # Get [CLS] token and pass through regressor
        pooled = encoder_outputs.last_hidden_state[:, 0, :]
        output = self.model.regressor(pooled)
        
        # Backward pass
        output.backward()
        
        # Get gradients
        gradients = embeddings.grad
        
        # Compute attribution: gradient × embedding (element-wise), then sum across embedding dim
        attributions = (gradients * embeddings).sum(dim=-1).squeeze()
        
        # Take absolute value (we care about magnitude, not direction)
        attributions = torch.abs(attributions).detach().cpu().numpy()
        
        return attributions
    
    def _compute_attention_attribution(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> np.ndarray:
        """
        Compute attention-based attribution scores.
        
        Uses attention weights from the last layer as importance proxy.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Get attention from last layer
        # Shape: (batch, num_heads, seq_len, seq_len)
        last_attention = outputs.attentions[-1]
        
        # Average across heads
        avg_attention = last_attention.mean(dim=1).squeeze()
        
        # Get attention TO the [CLS] token (column 0)
        # This shows how much each token contributes to the [CLS] representation
        cls_attention = avg_attention[:, 0].cpu().numpy()
        
        return cls_attention
    
    def _map_tokens_to_sentences(
        self, 
        text: str, 
        token_scores: np.ndarray,
        input_ids: torch.Tensor
    ) -> List[Tuple[str, float]]:
        """
        Map token-level scores to sentence-level scores.
        
        Args:
            text: Original essay text
            token_scores: Per-token importance scores
            input_ids: Tokenized input IDs
            
        Returns:
            List of (sentence, score) tuples
        """
        # Get sentences
        sentences = sent_tokenize(text)
        
        # Get tokens (for debugging/mapping)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        
        # Find which tokens belong to which sentence
        sentence_scores = []
        
        # Simple approach: tokenize each sentence and match lengths
        current_token_idx = 1  # Skip [CLS] token
        
        for sentence in sentences:
            # Tokenize the sentence
            sent_tokens = self.tokenizer.encode(
                sentence, 
                add_special_tokens=False,
                truncation=True,
                max_length=512
            )
            sent_length = len(sent_tokens)
            
            # Get scores for these tokens
            end_idx = min(current_token_idx + sent_length, len(token_scores))
            if current_token_idx < end_idx:
                sent_score = np.sum(token_scores[current_token_idx:end_idx])
            else:
                sent_score = 0.0
            
            sentence_scores.append((sentence, float(sent_score)))
            current_token_idx = end_idx
        
        # Normalize scores to sum to 1
        total_score = sum(abs(s[1]) for s in sentence_scores)
        if total_score > 0:
            sentence_scores = [(s, score / total_score) for s, score in sentence_scores]
        
        return sentence_scores
    
    def analyze_essay(
        self, 
        essay_text: str, 
        prompt: str = "",
        method: str = 'gradient'
    ) -> Dict:
        """
        Compute sentence-level importance scores for an essay.
        
        Args:
            essay_text: The essay text
            prompt: Optional essay prompt for context
            method: Attribution method ('gradient' or 'attention')
            
        Returns:
            Dictionary with:
            - prediction: predicted for_ratio
            - sentences: list of (sentence, importance_score)
            - most_important: top sentences by importance
            - least_important: bottom sentences
        """
        # Prepare input
        if prompt:
            full_text = f"{prompt} [SEP] {essay_text}"
        else:
            full_text = essay_text
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding='max_length'
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # Get prediction
        with torch.no_grad():
            pred = self.model(
                input_ids.to(self.device), 
                attention_mask.to(self.device)
            ).cpu().item()
        
        # Compute attribution
        if method == 'gradient':
            token_scores = self._compute_gradient_attribution(input_ids, attention_mask)
        else:
            token_scores = self._compute_attention_attribution(input_ids, attention_mask)
        
        # Map to sentences (use only essay_text, not the full prompt+text)
        sentence_scores = self._map_tokens_to_sentences(essay_text, token_scores, input_ids)
        
        # Sort by importance
        sorted_sentences = sorted(
            enumerate(sentence_scores),
            key=lambda x: abs(x[1][1]),
            reverse=True
        )
        
        return {
            'prediction': float(np.clip(pred, 0, 1)),
            'sentences': sentence_scores,
            'most_important': [
                {'index': idx, 'sentence': sent, 'score': score}
                for idx, (sent, score) in sorted_sentences[:5]
            ],
            'least_important': [
                {'index': idx, 'sentence': sent, 'score': score}
                for idx, (sent, score) in sorted_sentences[-3:]
            ],
            'num_sentences': len(sentence_scores),
            'method': method
        }
    
    def generate_html_visualization(
        self, 
        essay_text: str, 
        sentence_scores: List[Tuple[str, float]]
    ) -> str:
        """
        Generate HTML visualization with color-coded sentences.
        
        Higher importance = darker blue background.
        """
        html_parts = ['<div style="font-family: Georgia, serif; line-height: 1.8; padding: 20px;">']
        
        # Find max score for normalization
        max_score = max(abs(score) for _, score in sentence_scores) if sentence_scores else 1
        
        for i, (sentence, score) in enumerate(sentence_scores):
            # Normalize score to 0-1 range
            normalized = abs(score) / max_score if max_score > 0 else 0
            
            # Create color (blue with varying intensity)
            opacity = 0.1 + (normalized * 0.6)  # Range 0.1 to 0.7
            color = f"rgba(59, 130, 246, {opacity})"
            
            html_parts.append(
                f'<span style="background-color: {color}; padding: 2px 4px; '
                f'border-radius: 3px; margin: 1px;" '
                f'title="Sentence {i+1}: Importance={score:.3f}">'
                f'{sentence}</span> '
            )
        
        html_parts.append('</div>')
        return ''.join(html_parts)
    
    def generate_text_report(
        self, 
        essay_id: str,
        analysis_result: Dict,
        ground_truth: Dict = None
    ) -> str:
        """
        Generate a text report of the attribution analysis.
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"SENTENCE ATTRIBUTION ANALYSIS: {essay_id}")
        lines.append("=" * 70)
        
        lines.append(f"\n📊 MODEL PREDICTION")
        lines.append(f"   Predicted For Ratio: {analysis_result['prediction']:.3f}")
        if ground_truth:
            lines.append(f"   Actual For Ratio: {ground_truth.get('for_ratio', 'N/A')}")
            error = abs(analysis_result['prediction'] - ground_truth.get('for_ratio', 0))
            lines.append(f"   Prediction Error: {error:.3f}")
        
        lines.append(f"\n📝 ESSAY STRUCTURE")
        lines.append(f"   Total Sentences: {analysis_result['num_sentences']}")
        lines.append(f"   Attribution Method: {analysis_result['method']}")
        
        lines.append(f"\n🔥 TOP 5 MOST IMPORTANT SENTENCES")
        for item in analysis_result['most_important']:
            score = item['score']
            sent = item['sentence']
            idx = item['index'] + 1
            # Truncate long sentences
            display_sent = sent[:80] + "..." if len(sent) > 80 else sent
            lines.append(f"   {idx}. [{score:.3f}] \"{display_sent}\"")
        
        lines.append(f"\n💤 LEAST IMPORTANT SENTENCES")
        for item in analysis_result['least_important']:
            score = item['score']
            sent = item['sentence']
            idx = item['index'] + 1
            display_sent = sent[:80] + "..." if len(sent) > 80 else sent
            lines.append(f"   {idx}. [{score:.3f}] \"{display_sent}\"")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


def main():
    """Test the sentence attribution analyzer."""
    from src.data.dataset_builder import DatasetBuilder
    from src.analysis.essay_features import EssayFeatureExtractor
    from src.models.essay_predictor_transformer import EssayTransformerRegressor
    from transformers import AutoTokenizer
    
    print("Loading dataset...")
    dataset_dir = Path(__file__).parent.parent.parent / "dataset" / "ArgumentAnnotatedEssays-2.0"
    builder = DatasetBuilder(str(dataset_dir))
    builder.build()
    
    print("\nLoading model...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = EssayTransformerRegressor('distilbert-base-uncased')
    model.load_state_dict(
        torch.load('models/transformer_regression_best.pt', 
                  map_location=device, weights_only=True)
    )
    
    # Create analyzer
    analyzer = SentenceAttributionAnalyzer(model, tokenizer, device)
    
    # Get a test essay
    test_essays = builder.get_essays_by_split('test')
    essay = test_essays[0]
    
    print(f"\nAnalyzing essay: {essay.essay_id}")
    
    # Get ground truth
    feature_extractor = EssayFeatureExtractor()
    features = feature_extractor.extract_features(essay)
    ground_truth = {
        'for_ratio': features.for_ratio,
        'stance_category': features.stance_category
    }
    
    # Run attribution analysis
    result = analyzer.analyze_essay(essay.text, essay.prompt, method='attention')
    
    # Print report
    report = analyzer.generate_text_report(essay.essay_id, result, ground_truth)
    print(report)
    
    # Generate HTML (save to file)
    html = analyzer.generate_html_visualization(essay.text, result['sentences'])
    output_path = Path('reports/sentence_attribution_sample.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"""
        <html>
        <head><title>Sentence Attribution: {essay.essay_id}</title></head>
        <body>
        <h1>Sentence Attribution Analysis: {essay.essay_id}</h1>
        <p><strong>Predicted For Ratio:</strong> {result['prediction']:.3f}</p>
        <p><strong>Actual For Ratio:</strong> {ground_truth['for_ratio']:.3f}</p>
        <h2>Essay Text (darker blue = more important)</h2>
        {html}
        </body>
        </html>
        """)
    
    print(f"\n✓ HTML visualization saved to {output_path}")


if __name__ == "__main__":
    main()

