"""
Stage 1 Training: Binary DR Classification

This script trains the multi-task model focusing only on the DR binary head.
Stage 1 is the foundation for multi-disease detection in later stages.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import json
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.multi_task_models import MultiTaskModel
from src.data.datamodule import FundusDataModule
from src.metrics.metrics import BinaryClassificationMetrics
from src.utils.seed import set_seed


class Stage1Trainer:
    """
    Stage 1 Trainer for binary DR classification using MultiTaskModel.
    
    Trains only the DR binary head while keeping the architecture
    ready for future multi-task extensions.
    """
    
    def __init__(
        self,
        model: MultiTaskModel,
        data_module: FundusDataModule,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        use_class_weights: bool = True,
        save_dir: str = 'models/stage1_dr_binary',
    ):
        """
        Initialize Stage 1 trainer.
        
        Args:
            model: MultiTaskModel instance
            data_module: Data module with train/val/test splits
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            use_class_weights: Whether to use class-weighted loss
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_class_weights = use_class_weights
        
        # Loss function (will be set up after data_module.setup())
        self.criterion = None
        
        # Optimizer (Stage 1: Only backbone + DR binary head)
        # CRITICAL: Do not optimize other heads even if they exist
        params = (
            list(self.model.backbone.parameters()) +
            list(self.model.dr_binary_head.parameters())
        )
        self.optimizer = optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Training state
        self.best_val_sensitivity = 0.0
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_sensitivity': [],
            'val_specificity': [],
            'val_accuracy': [],
            'val_auc_roc': [],
        }
    
    def _setup_criterion(self):
        """
        Setup loss function with class weights if enabled.
        
        Note: get_class_weights() returns inverse-frequency weights.
        We compute pos_weight = weights[1] / weights[0] for BCEWithLogitsLoss.
        This gives higher penalty to missing DR cases (false negatives).
        """
        if self.criterion is None:
            if self.use_class_weights:
                # Get inverse-frequency weights from data module
                weights = self.data_module.get_class_weights().to(self.device)
                pos_weight = weights[1] / weights[0]
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                print(f"✓ Using class-weighted BCE loss")
                print(f"  Class weights (inverse-freq): {weights.cpu().tolist()}")
                print(f"  Positive weight (DR): {pos_weight.item():.4f}")
            else:
                self.criterion = nn.BCEWithLogitsLoss()
                print("✓ Using standard BCE loss (no class weights)")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.data_module.train_dataloader(), desc='Training')
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.float().to(self.device)
            
            # Forward pass (get only DR binary output)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            logits = outputs['dr_binary'].squeeze(1)  # (B, 1) -> (B,)
            
            # Compute loss
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
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
                outputs = self.model(images)
                logits = outputs['dr_binary'].squeeze(1)  # (B, 1) -> (B,)
                
                # Compute loss
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
        Main training loop with early stopping.
        
        Args:
            num_epochs: Maximum number of epochs
            patience: Early stopping patience (epochs without improvement)
        """
        # Setup criterion
        self._setup_criterion()
        
        print("\n" + "="*70)
        print("STAGE 1 TRAINING: Binary DR Classification")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Model: {type(self.model).__name__}")
        print(f"Training samples: {len(self.data_module.train_dataset)}")
        print(f"Validation samples: {len(self.data_module.val_dataset)}")
        print(f"Batch size: {self.data_module.batch_size}")
        print(f"Max epochs: {num_epochs}")
        print(f"Early stopping patience: {patience}")
        print(f"Save directory: {self.save_dir}")
        print("="*70 + "\n")
        
        epochs_without_improvement = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Update learning rate based on sensitivity
            self.scheduler.step(val_metrics['sensitivity'])
            
            # Save metrics
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_sensitivity'].append(val_metrics['sensitivity'])
            self.train_history['val_specificity'].append(val_metrics['specificity'])
            self.train_history['val_accuracy'].append(val_metrics['accuracy'])
            self.train_history['val_auc_roc'].append(val_metrics.get('auc_roc', 0.0))
            
            # Print metrics
            print(f"\n📊 Epoch {epoch} Results:")
            print(f"  Train Loss:      {train_loss:.4f}")
            print(f"  Val Loss:        {val_loss:.4f}")
            print(f"  Val Sensitivity: {val_metrics['sensitivity']:.4f} ⭐ (Primary)")
            print(f"  Val Specificity: {val_metrics['specificity']:.4f}")
            print(f"  Val Accuracy:    {val_metrics['accuracy']:.4f}")
            if 'auc_roc' in val_metrics:
                print(f"  Val AUC-ROC:     {val_metrics['auc_roc']:.4f}")
            
            # Save best model
            if val_metrics['sensitivity'] > self.best_val_sensitivity:
                improvement = val_metrics['sensitivity'] - self.best_val_sensitivity
                self.best_val_sensitivity = val_metrics['sensitivity']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                epochs_without_improvement = 0
                print(f"\n✓ New best model! Sensitivity improved by {improvement:.4f}")
                print(f"  Checkpoint saved to: {self.save_dir / 'best.pt'}")
            else:
                epochs_without_improvement += 1
                print(f"\n⚠ No improvement for {epochs_without_improvement} epoch(s)")
                
            # Save latest checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=False)
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\n🛑 Early stopping triggered after {epoch} epochs")
                print(f"   Best sensitivity: {self.best_val_sensitivity:.4f}")
                break
        
        # Save training history
        self.save_history()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Best Validation Sensitivity: {self.best_val_sensitivity:.4f}")
        print(f"Training history saved to: {self.save_dir / 'training_history.json'}")
        print(f"Best model saved to: {self.save_dir / 'best.pt'}")
        print("="*70 + "\n")
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_sensitivity': self.best_val_sensitivity,
            'train_history': self.train_history,
        }
        
        # Save latest
        torch.save(checkpoint, self.save_dir / 'latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pt')
            torch.save(self.model.state_dict(), self.save_dir / 'best_model.pt')
    
    def save_history(self):
        """Save training history to JSON."""
        history_file = self.save_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        print(f"✓ Training history saved to: {history_file}")


def main():
    """Main training script."""
    # Set random seed for reproducibility
    set_seed(42)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Initialize data module
    print("Loading data...")
    data_module = FundusDataModule(
        data_root="Data/splits/fundus",
        image_size=224,
        batch_size=32,
        num_workers=4,
    )
    data_module.setup()
    print("✓ Data loaded\n")
    
    # Initialize multi-task model
    print("Initializing MultiTaskModel...")
    model = MultiTaskModel(backbone_pretrained=True)
    print("✓ Model initialized\n")
    
    # Initialize trainer
    trainer = Stage1Trainer(
        model=model,
        data_module=data_module,
        device=device,
        learning_rate=1e-4,
        weight_decay=1e-5,
        use_class_weights=True,
        save_dir='models/stage1_dr_binary',
    )
    
    # Train
    trainer.train(num_epochs=50, patience=10)


if __name__ == '__main__':
    main()
