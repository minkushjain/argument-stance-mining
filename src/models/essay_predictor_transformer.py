"""
Transformer-based Essay-Level Stance Prediction

Fine-tunes RoBERTa/DistilBERT for predicting:
- for_ratio (regression)
- stance_category (classification)

Uses truncation strategies for long essays.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report
)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset_builder import DatasetBuilder
from src.analysis.essay_features import EssayFeatureExtractor


class EssayDataset(Dataset):
    """PyTorch Dataset for essays."""
    
    def __init__(
        self, 
        essays: List,
        features_df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        task: str = 'regression'  # 'regression' or 'classification'
    ):
        self.essays = essays
        self.features_df = features_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        
        # Create essay_id to index mapping
        self.id_to_idx = {e.essay_id: i for i, e in enumerate(essays)}
        
    def __len__(self):
        return len(self.essays)
    
    def __getitem__(self, idx):
        essay = self.essays[idx]
        
        # Get text: combine prompt and essay text
        if essay.prompt:
            text = f"{essay.prompt} [SEP] {essay.text}"
        else:
            text = essay.text
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get features
        essay_features = self.features_df[self.features_df['essay_id'] == essay.essay_id].iloc[0]
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'essay_id': essay.essay_id
        }
        
        if self.task == 'regression':
            item['labels'] = torch.tensor(essay_features['for_ratio'], dtype=torch.float)
        else:  # classification
            # Map category to label
            category = essay_features['stance_category']
            if category == 'balanced':
                item['labels'] = torch.tensor(0, dtype=torch.long)
            elif category == 'mostly_for':
                item['labels'] = torch.tensor(1, dtype=torch.long)
            else:  # mostly_against
                item['labels'] = torch.tensor(2, dtype=torch.long)
        
        return item


class RegressionHead(nn.Module):
    """Custom regression head for transformer."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        return x.squeeze(-1)


class EssayTransformerRegressor(nn.Module):
    """Transformer model for essay regression."""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.regressor = RegressionHead(hidden_size, dropout)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.regressor(pooled)


class EssayPredictorTransformer:
    """
    Transformer-based predictor for essay-level stance features.
    """
    
    def __init__(
        self, 
        model_name: str = 'distilbert-base-uncased',
        max_length: int = 512,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        epochs: int = 5,
        device: str = None,
        output_dir: str = "models"
    ):
        self.model_name = model_name
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
            self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                       'mps' if torch.backends.mps.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Models
        self.regression_model = None
        self.classification_model = None
    
    def create_dataloaders(
        self,
        train_essays, val_essays, test_essays,
        train_features, val_features, test_features,
        task: str = 'regression'
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders for train/val/test."""
        
        train_dataset = EssayDataset(
            train_essays, train_features, self.tokenizer, self.max_length, task
        )
        val_dataset = EssayDataset(
            val_essays, val_features, self.tokenizer, self.max_length, task
        )
        test_dataset = EssayDataset(
            test_essays, test_features, self.tokenizer, self.max_length, task
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader, test_loader
    
    def train_regression_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, List]:
        """Train regression model for for_ratio prediction."""
        
        print("\n" + "="*60)
        print("TRAINING TRANSFORMER REGRESSION MODEL")
        print("="*60)
        
        # Initialize model
        self.regression_model = EssayTransformerRegressor(self.model_name)
        self.regression_model.to(self.device)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.regression_model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Training
            self.regression_model.train()
            train_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.regression_model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.regression_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            self.regression_model.eval()
            val_losses = []
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.regression_model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    val_losses.append(loss.item())
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = np.mean(val_losses)
            val_mae = mean_absolute_error(all_labels, all_preds)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_mae'].append(val_mae)
            
            print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, Val MAE={val_mae:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.regression_model.state_dict(), 
                          self.output_dir / 'transformer_regression_best.pt')
        
        # Load best model
        self.regression_model.load_state_dict(
            torch.load(self.output_dir / 'transformer_regression_best.pt', weights_only=True)
        )
        
        return history
    
    def train_classification_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_labels: int = 3
    ) -> Dict[str, List]:
        """Train classification model for stance_category prediction."""
        
        print("\n" + "="*60)
        print("TRAINING TRANSFORMER CLASSIFICATION MODEL")
        print("="*60)
        
        # Initialize model
        self.classification_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        self.classification_model.to(self.device)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.classification_model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}
        
        best_val_f1 = 0
        
        for epoch in range(self.epochs):
            # Training
            self.classification_model.train()
            train_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.classification_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.classification_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            self.classification_model.eval()
            val_losses = []
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.classification_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_losses.append(outputs.loss.item())
                    preds = torch.argmax(outputs.logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = np.mean(val_losses)
            val_accuracy = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='macro')
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['val_f1'].append(val_f1)
            
            print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.4f}, Val F1={val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.classification_model.state_dict(),
                          self.output_dir / 'transformer_classification_best.pt')
        
        # Load best model
        self.classification_model.load_state_dict(
            torch.load(self.output_dir / 'transformer_classification_best.pt', weights_only=True)
        )
        
        return history
    
    def evaluate_regression(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate regression model on test set."""
        self.regression_model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                outputs = self.regression_model(input_ids, attention_mask)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        results = {
            'mae': mean_absolute_error(all_labels, all_preds),
            'rmse': np.sqrt(mean_squared_error(all_labels, all_preds)),
            'r2': r2_score(all_labels, all_preds),
            'predictions': all_preds,
            'true_values': all_labels
        }
        
        print("\n=== Regression Test Results ===")
        print(f"MAE: {results['mae']:.4f}")
        print(f"RMSE: {results['rmse']:.4f}")
        print(f"R²: {results['r2']:.4f}")
        
        return results
    
    def evaluate_classification(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate classification model on test set."""
        self.classification_model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                outputs = self.classification_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        results = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'macro_f1': f1_score(all_labels, all_preds, average='macro'),
            'predictions': all_preds,
            'true_values': all_labels
        }
        
        print("\n=== Classification Test Results ===")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro-F1: {results['macro_f1']:.4f}")
        
        return results


def main():
    """Train and evaluate transformer models."""
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
    predictor = EssayPredictorTransformer(
        model_name='distilbert-base-uncased',
        max_length=512,
        batch_size=8,
        learning_rate=2e-5,
        epochs=3,  # Reduced for faster training
        output_dir=str(output_dir)
    )
    
    # Create dataloaders for regression
    print("\nCreating dataloaders for regression...")
    train_loader_reg, val_loader_reg, test_loader_reg = predictor.create_dataloaders(
        train_essays, val_essays, test_essays,
        train_features, val_features, test_features,
        task='regression'
    )
    
    # Train regression model
    reg_history = predictor.train_regression_model(train_loader_reg, val_loader_reg)
    reg_results = predictor.evaluate_regression(test_loader_reg)
    
    # Create dataloaders for classification
    print("\nCreating dataloaders for classification...")
    train_loader_cls, val_loader_cls, test_loader_cls = predictor.create_dataloaders(
        train_essays, val_essays, test_essays,
        train_features, val_features, test_features,
        task='classification'
    )
    
    # Train classification model
    cls_history = predictor.train_classification_model(train_loader_cls, val_loader_cls, num_labels=3)
    cls_results = predictor.evaluate_classification(test_loader_cls)
    
    # Save results
    results_summary = {
        'for_ratio_regression': {
            'mae': float(reg_results['mae']),
            'rmse': float(reg_results['rmse']),
            'r2': float(reg_results['r2'])
        },
        'stance_category_classification': {
            'accuracy': float(cls_results['accuracy']),
            'macro_f1': float(cls_results['macro_f1'])
        }
    }
    
    with open(figures_dir.parent / 'transformer_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved transformer_results.json")
    
    print("\n✓ Transformer model training complete!")


if __name__ == "__main__":
    main()

