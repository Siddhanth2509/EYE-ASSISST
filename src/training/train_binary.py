# src/training/train_binary.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from src.data.datamodule import FundusDataModule
from src.metrics.metrics import BinaryClassificationMetrics
from src.utils.seed import set_seed


class BinaryTrainer:
    """
    Trainer for binary DR classification.
    
    Implements training loop with:
    - Class-weighted BCE loss (handles imbalance)
    - Learning rate scheduling
    - Early stopping
    - Metrics tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        data_module: FundusDataModule,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        use_class_weights: bool = True,
        save_dir: str = 'models/binary_dr',
    ):
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_class_weights = use_class_weights
        
        # Loss function will be set up after data_module.setup() is called
        # We'll initialize it in a separate method
        self.criterion = None
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            # verbose=True
        )
        
        # Training state
        self.best_val_sensitivity = 0.0
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_sensitivity': [],
            'val_accuracy': [],
        }
    
    def _setup_criterion(self):
        """Setup loss function after data_module.setup() has been called."""
        if self.criterion is None:
            if self.use_class_weights:
                weights = self.data_module.get_class_weights().to(self.device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1] / weights[0])
                print(f"Using class-weighted BCE loss. Weights: {weights.cpu().tolist()}")
            else:
                self.criterion = nn.BCEWithLogitsLoss()
                print("Using standard BCE loss (no class weights)")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.data_module.train_dataloader(), desc='Training')
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.float().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images).squeeze()
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        return running_loss / num_batches
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        running_loss = 0.0
        num_batches = 0
        metrics = BinaryClassificationMetrics()
        
        with torch.no_grad():
            for images, targets in tqdm(self.data_module.val_dataloader(), desc='Validating'):
                images = images.to(self.device)
                targets = targets.float().to(self.device)
                
                # Forward pass
                logits = self.model(images).squeeze()
                loss = self.criterion(logits, targets)
                
                # Predictions
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                
                # Update metrics
                metrics.update(
                    preds=preds,
                    targets=targets.long(),
                    probs=probs
                )
                
                running_loss += loss.item()
                num_batches += 1
        
        avg_loss = running_loss / num_batches
        computed_metrics = metrics.compute()
        
        return avg_loss, computed_metrics
    
    def train(self, num_epochs: int = 50, patience: int = 10):
        """
        Main training loop.
        
        Args:
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
        """
        # Setup criterion now that datasets are initialized
        self._setup_criterion()
        
        print(f"Training on device: {self.device}")
        print(f"Model: {type(self.model).__name__}")
        print(f"Training samples: {len(self.data_module.train_dataset)}")
        print(f"Validation samples: {len(self.data_module.val_dataset)}")
        
        epochs_without_improvement = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['sensitivity'])
            
            # Save metrics
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_sensitivity'].append(val_metrics['sensitivity'])
            self.train_history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Sensitivity: {val_metrics['sensitivity']:.4f} (PRIMARY METRIC)")
            print(f"Val Specificity: {val_metrics['specificity']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val AUC-ROC: {val_metrics.get('auc_roc', 'N/A')}")
            
            # Save best model
            if val_metrics['sensitivity'] > self.best_val_sensitivity:
                self.best_val_sensitivity = val_metrics['sensitivity']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                epochs_without_improvement = 0
                print(f"✓ New best model saved! (Sensitivity: {val_metrics['sensitivity']:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epochs")
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # Save final history
        self.save_history()
        print(f"\nTraining completed! Best sensitivity: {self.best_val_sensitivity:.4f}")
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_sensitivity': self.best_val_sensitivity,
        }
        
        # Save latest
        torch.save(checkpoint, self.save_dir / 'latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pt')
            torch.save(self.model.state_dict(), self.save_dir / 'best_model.pt')
    
    def save_history(self):
        """Save training history."""
        history_file = self.save_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.train_history, f, indent=2)
