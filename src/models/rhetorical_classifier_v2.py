"""
Improved Rhetorical Role Classifier (V2)

Improvements over V1:
1. Include component type (MajorClaim/Claim/Premise) as input feature
2. Include positional information (beginning/middle/end of essay)
3. Use Focal Loss for better handling of class imbalance
4. Longer training with early stopping
5. Label smoothing for regularization
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset_builder import DatasetBuilder
from src.data.rhetorical_labels import RhetoricalLabelGenerator, RhetoricalRole
from src.data.brat_parser import Essay, Component


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reduces the relative loss for well-classified examples,
    focusing training on hard, misclassified examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class EnhancedRhetoricalDataset(Dataset):
    """
    Enhanced Dataset with component type and position information.
    """
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int],
        component_types: List[str],
        positions: List[str],
        tokenizer,
        max_length: int = 160
    ):
        self.texts = texts
        self.labels = labels
        self.component_types = component_types
        self.positions = positions
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        comp_type = self.component_types[idx]
        position = self.positions[idx]
        
        # Enhanced input format:
        # [TYPE: MajorClaim] [POSITION: beginning] [TEXT: actual text...]
        enhanced_text = f"[TYPE: {comp_type}] [POSITION: {position}] {text}"
        
        encoding = self.tokenizer(
            enhanced_text,
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


class EnhancedLabelGenerator(RhetoricalLabelGenerator):
    """
    Enhanced label generator that includes component type and position.
    """
    
    def generate_enhanced_dataset(
        self, 
        essays: List[Essay]
    ) -> Tuple[List[str], List[int], List[str], List[str], List[str]]:
        """
        Generate training data with enhanced features.
        
        Returns:
            Tuple of (texts, labels, component_types, positions, essay_ids)
        """
        texts = []
        labels = []
        component_types = []
        positions = []
        essay_ids = []
        
        for essay in essays:
            essay_length = len(essay.text)
            labeled_components = self.generate_labels(essay)
            
            for lc in labeled_components:
                # Get text with prompt context
                texts.append(lc.text_with_context)
                labels.append(lc.role.value)
                essay_ids.append(lc.essay_id)
                
                # Component type
                component_types.append(lc.component.type)
                
                # Position in essay (beginning/middle/end)
                comp_start = lc.component.start
                relative_pos = comp_start / essay_length if essay_length > 0 else 0.5
                
                if relative_pos < 0.33:
                    positions.append("beginning")
                elif relative_pos < 0.67:
                    positions.append("middle")
                else:
                    positions.append("end")
        
        return texts, labels, component_types, positions, essay_ids


class ImprovedRhetoricalClassifier:
    """
    Improved Rhetorical Role Classifier with:
    - Component type and position features
    - Focal Loss for class imbalance
    - Early stopping
    - Label smoothing
    """
    
    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        num_labels: int = 5,
        max_length: int = 160,
        batch_size: int = 16,
        learning_rate: float = 3e-5,  # Slightly higher LR
        epochs: int = 10,  # More epochs with early stopping
        patience: int = 3,  # Early stopping patience
        focal_gamma: float = 2.0,  # Focal loss gamma
        label_smoothing: float = 0.1,
        device: str = None,
        output_dir: str = "models"
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
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
        train_data: Tuple,
        val_data: Tuple,
        test_data: Tuple = None
    ) -> Dict[str, DataLoader]:
        """Prepare data loaders with enhanced features."""
        
        train_texts, train_labels, train_types, train_positions, _ = train_data
        val_texts, val_labels, val_types, val_positions, _ = val_data
        
        # Compute class weights
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32).to(self.device)
        
        print(f"Class weights:")
        for name, weight in zip(self.label_names, self.class_weights.cpu().numpy()):
            print(f"  {name:20s}: {weight:.3f}")
        
        # Create datasets
        train_dataset = EnhancedRhetoricalDataset(
            train_texts, train_labels, train_types, train_positions,
            self.tokenizer, self.max_length
        )
        val_dataset = EnhancedRhetoricalDataset(
            val_texts, val_labels, val_types, val_positions,
            self.tokenizer, self.max_length
        )
        
        # Create data loaders
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        }
        
        if test_data:
            test_texts, test_labels, test_types, test_positions, _ = test_data
            test_dataset = EnhancedRhetoricalDataset(
                test_texts, test_labels, test_types, test_positions,
                self.tokenizer, self.max_length
            )
            dataloaders['test'] = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )
        
        return dataloaders
    
    def train(self, dataloaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """Train with Focal Loss and early stopping."""
        
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        # Focal Loss with class weights
        criterion = FocalLoss(alpha=self.class_weights, gamma=self.focal_gamma)
        
        # Optimizer with weight decay
        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Scheduler
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        best_val_f1 = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}
        
        print(f"\nTraining for up to {self.epochs} epochs (early stopping patience: {self.patience})")
        print("-" * 60)
        
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
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_metrics = self.evaluate(val_loader, criterion)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['macro_f1'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={val_metrics['loss']:.4f}, "
                  f"Val Acc={val_metrics['accuracy']:.4f}, "
                  f"Val Macro-F1={val_metrics['macro_f1']:.4f}")
            
            # Early stopping check
            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    self.output_dir / 'rhetorical_classifier_v2_best.pt'
                )
                print(f"  → New best model saved (F1={best_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\n  Early stopping triggered after {epoch+1} epochs")
                    break
        
        self.results['training_history'] = history
        self.results['best_val_f1'] = best_val_f1
        self.results['epochs_trained'] = epoch + 1
        
        return history
    
    def evaluate(self, dataloader: DataLoader, criterion=None) -> Dict[str, Any]:
        """Evaluate model."""
        
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        if criterion is None:
            criterion = FocalLoss(alpha=self.class_weights, gamma=self.focal_gamma)
        
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
        """Full evaluation on test set."""
        
        # Load best model
        self.model.load_state_dict(
            torch.load(self.output_dir / 'rhetorical_classifier_v2_best.pt',
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
        print("TEST SET RESULTS (IMPROVED MODEL V2)")
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
    
    def save_results(self, filepath: str = None):
        """Save training results."""
        if filepath is None:
            filepath = self.output_dir / 'rhetorical_classifier_v2_results.json'
        
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
    """Train improved rhetorical role classifier."""
    
    print("="*60)
    print("IMPROVED RHETORICAL ROLE CLASSIFIER (V2)")
    print("="*60)
    print("\nImprovements:")
    print("  - Component type as input feature")
    print("  - Position in essay (beginning/middle/end)")
    print("  - Focal Loss for class imbalance")
    print("  - Early stopping")
    print("  - Higher learning rate with warmup")
    
    # Load dataset
    print("\n1. Loading dataset...")
    dataset_dir = Path(__file__).parent.parent.parent / "dataset" / "ArgumentAnnotatedEssays-2.0"
    builder = DatasetBuilder(str(dataset_dir))
    builder.build()
    
    # Generate enhanced labels
    print("\n2. Generating enhanced training data...")
    generator = EnhancedLabelGenerator()
    
    # Get essays by split
    train_essays = builder.get_essays_by_split('train')
    val_essays = builder.get_essays_by_split('val')
    test_essays = builder.get_essays_by_split('test')
    
    # Generate enhanced datasets
    train_data = generator.generate_enhanced_dataset(train_essays)
    
    generator.stats.clear()
    val_data = generator.generate_enhanced_dataset(val_essays)
    
    generator.stats.clear()
    test_data = generator.generate_enhanced_dataset(test_essays)
    
    print(f"\n   Train: {len(train_data[0])} components")
    print(f"   Val:   {len(val_data[0])} components")
    print(f"   Test:  {len(test_data[0])} components")
    
    # Show position distribution
    from collections import Counter
    pos_dist = Counter(train_data[3])
    print(f"\n   Position distribution (train):")
    for pos, count in pos_dist.most_common():
        print(f"     {pos}: {count}")
    
    # Initialize improved classifier
    print("\n3. Initializing improved classifier...")
    classifier = ImprovedRhetoricalClassifier(
        model_name='distilbert-base-uncased',
        num_labels=5,
        max_length=160,
        batch_size=16,
        learning_rate=3e-5,
        epochs=10,
        patience=3,
        focal_gamma=2.0
    )
    
    # Prepare data
    print("\n4. Preparing data loaders...")
    dataloaders = classifier.prepare_data(train_data, val_data, test_data)
    
    # Train
    print("\n5. Training...")
    classifier.train(dataloaders)
    
    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    classifier.evaluate_on_test(dataloaders['test'])
    
    # Save results
    classifier.save_results()
    
    # Compare with V1
    print("\n" + "="*60)
    print("COMPARISON WITH V1")
    print("="*60)
    
    v1_results_path = Path("models/rhetorical_classifier_results.json")
    if v1_results_path.exists():
        with open(v1_results_path) as f:
            v1_results = json.load(f)
        
        v1_acc = v1_results.get('test_metrics', {}).get('accuracy', 0)
        v1_f1 = v1_results.get('test_metrics', {}).get('macro_f1', 0)
        v2_acc = classifier.results['test_metrics']['accuracy']
        v2_f1 = classifier.results['test_metrics']['macro_f1']
        
        print(f"\n{'Metric':<20} {'V1':<12} {'V2':<12} {'Change':<12}")
        print("-" * 56)
        print(f"{'Accuracy':<20} {v1_acc:<12.4f} {v2_acc:<12.4f} {v2_acc-v1_acc:+.4f}")
        print(f"{'Macro F1':<20} {v1_f1:<12.4f} {v2_f1:<12.4f} {v2_f1-v1_f1:+.4f}")
        
        print(f"\nPer-class F1 comparison:")
        v1_per_class = v1_results.get('test_metrics', {}).get('per_class_f1', {})
        v2_per_class = classifier.results['test_metrics']['per_class_f1']
        
        for role in RhetoricalRole.names():
            v1_f = v1_per_class.get(role, 0)
            v2_f = v2_per_class.get(role, 0)
            change = v2_f - v1_f
            indicator = "↑" if change > 0 else "↓" if change < 0 else "="
            print(f"  {role:<20} {v1_f:.4f} → {v2_f:.4f} ({change:+.4f}) {indicator}")


if __name__ == "__main__":
    main()

