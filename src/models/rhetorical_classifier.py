"""
Rhetorical Role Classifier

Fine-tunes DistilBERT to classify argument components into rhetorical roles:
- THESIS: Main position statement
- MAIN_ARGUMENT: Primary supporting claims
- COUNTER_ARGUMENT: Opposing views
- REBUTTAL: Evidence countering counter-arguments
- EVIDENCE: Supporting premises
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_class_weight

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset_builder import DatasetBuilder
from src.data.rhetorical_labels import RhetoricalLabelGenerator, RhetoricalRole


class RhetoricalDataset(Dataset):
    """PyTorch Dataset for rhetorical role classification."""
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int],
        tokenizer,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class RhetoricalRoleClassifier:
    """
    Fine-tuned DistilBERT for rhetorical role classification.
    """
    
    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        num_labels: int = 5,
        max_length: int = 128,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        epochs: int = 5,
        device: str = None,
        output_dir: str = "models"
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                'mps' if torch.backends.mps.is_available() else
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        ).to(self.device)
        
        # Label names
        self.label_names = RhetoricalRole.names()
        
        # Results storage
        self.results = {}
    
    def prepare_data(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        test_texts: List[str] = None,
        test_labels: List[int] = None
    ) -> Dict[str, DataLoader]:
        """Prepare data loaders."""
        
        # Compute class weights for imbalanced data
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32).to(self.device)
        print(f"Class weights: {dict(zip(self.label_names, self.class_weights.cpu().numpy()))}")
        
        # Create datasets
        train_dataset = RhetoricalDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        val_dataset = RhetoricalDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )
        
        # Create data loaders
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        }
        
        if test_texts and test_labels:
            test_dataset = RhetoricalDataset(
                test_texts, test_labels, self.tokenizer, self.max_length
            )
            dataloaders['test'] = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )
        
        return dataloaders
    
    def train(self, dataloaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """Train the classifier."""
        
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Scheduler
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        best_val_f1 = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss = criterion(outputs.logits, labels)
                train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['macro_f1'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={val_metrics['loss']:.4f}, "
                  f"Val Acc={val_metrics['accuracy']:.4f}, "
                  f"Val Macro-F1={val_metrics['macro_f1']:.4f}")
            
            # Save best model
            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                torch.save(
                    self.model.state_dict(),
                    self.output_dir / 'rhetorical_classifier_best.pt'
                )
                print(f"  → New best model saved (F1={best_val_f1:.4f})")
        
        self.results['training_history'] = history
        self.results['best_val_f1'] = best_val_f1
        
        return history
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on a dataloader."""
        
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss = criterion(outputs.logits, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class_f1': dict(zip(self.label_names, per_class_f1)),
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def evaluate_on_test(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Full evaluation on test set with detailed report."""
        
        # Load best model
        self.model.load_state_dict(
            torch.load(self.output_dir / 'rhetorical_classifier_best.pt',
                      map_location=self.device, weights_only=True)
        )
        
        metrics = self.evaluate(dataloader)
        
        # Generate classification report
        report = classification_report(
            metrics['labels'],
            metrics['predictions'],
            target_names=self.label_names,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(metrics['labels'], metrics['predictions'])
        
        print("\n" + "="*60)
        print("TEST SET RESULTS")
        print("="*60)
        print(f"\nAccuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"\nPer-class F1 scores:")
        for name, f1 in metrics['per_class_f1'].items():
            print(f"  {name:20s}: {f1:.4f}")
        print(f"\nClassification Report:\n{report}")
        
        self.results['test_metrics'] = {
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'per_class_f1': metrics['per_class_f1']
        }
        self.results['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict rhetorical role for a single text.
        
        Args:
            text: Component text (optionally with prompt context)
            
        Returns:
            Tuple of (predicted_role, confidence, all_probabilities)
        """
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
        
        predicted_role = self.label_names[pred_idx]
        confidence = probs[pred_idx]
        all_probs = dict(zip(self.label_names, probs.tolist()))
        
        return predicted_role, confidence, all_probs
    
    def save_results(self, filepath: str = None):
        """Save training results to JSON."""
        if filepath is None:
            filepath = self.output_dir / 'rhetorical_classifier_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                results_json[key] = value.tolist()
            elif isinstance(value, dict):
                results_json[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                results_json[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n✓ Results saved to {filepath}")


def main():
    """Train and evaluate the rhetorical role classifier."""
    
    print("="*60)
    print("RHETORICAL ROLE CLASSIFIER TRAINING")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    dataset_dir = Path(__file__).parent.parent.parent / "dataset" / "ArgumentAnnotatedEssays-2.0"
    builder = DatasetBuilder(str(dataset_dir))
    builder.build()
    
    # Generate labels
    print("\n2. Generating rhetorical labels...")
    generator = RhetoricalLabelGenerator()
    
    # Get essays by split
    train_essays = builder.get_essays_by_split('train')
    val_essays = builder.get_essays_by_split('val')
    test_essays = builder.get_essays_by_split('test')
    
    # Generate datasets for each split
    train_texts, train_labels, _ = generator.generate_dataset(train_essays)
    
    # Reset stats for val/test
    generator.stats.clear()
    val_texts, val_labels, _ = generator.generate_dataset(val_essays)
    
    generator.stats.clear()
    test_texts, test_labels, _ = generator.generate_dataset(test_essays)
    
    print(f"\n   Train: {len(train_texts)} components")
    print(f"   Val:   {len(val_texts)} components")
    print(f"   Test:  {len(test_texts)} components")
    
    # Initialize classifier
    print("\n3. Initializing classifier...")
    classifier = RhetoricalRoleClassifier(
        model_name='distilbert-base-uncased',
        num_labels=5,
        max_length=128,
        batch_size=16,
        learning_rate=2e-5,
        epochs=5
    )
    
    # Prepare data
    print("\n4. Preparing data loaders...")
    dataloaders = classifier.prepare_data(
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels
    )
    
    # Train
    print("\n5. Training...")
    classifier.train(dataloaders)
    
    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    classifier.evaluate_on_test(dataloaders['test'])
    
    # Save results
    classifier.save_results()
    
    # Test prediction on sample
    print("\n7. Sample predictions...")
    samples = [
        "We should prioritize environmental protection over economic growth.",
        "This would lead to job losses in many industries.",
        "Studies show that renewable energy creates more jobs than fossil fuels.",
        "However, some argue that the transition costs are too high.",
        "For example, solar panel installations have increased by 50% last year."
    ]
    
    for sample in samples:
        role, conf, probs = classifier.predict(sample)
        print(f"\n   \"{sample[:60]}...\"")
        print(f"   → {role} ({conf:.2%})")


if __name__ == "__main__":
    main()

