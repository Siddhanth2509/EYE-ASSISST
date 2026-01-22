# src/training/evaluate.py

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import json

from src.data.datamodule import FundusDataModule
from src.metrics.metrics import BinaryClassificationMetrics


class BinaryEvaluator:
    """
    Evaluator for binary DR classification.
    
    Evaluates model on test set with comprehensive medical metrics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        data_module: FundusDataModule,
        checkpoint_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        print(f"Loaded model from: {checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"Trained for {checkpoint['epoch']} epochs")
        if 'metrics' in checkpoint:
            print(f"Best validation sensitivity: {checkpoint['metrics'].get('sensitivity', 'N/A')}")
    
    def evaluate(self, split: str = 'test'):
        """
        Evaluate model on specified split.
        
        Args:
            split: 'test' or 'val'
        
        Returns:
            Dictionary of metrics
        """
        # Get appropriate dataloader
        if split == 'test':
            dataloader = self.data_module.test_dataloader()
            dataset_name = 'APTOS (External Test)'
        elif split == 'val':
            dataloader = self.data_module.val_dataloader()
            dataset_name = 'EyePACS (Validation)'
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"\nEvaluating on {dataset_name}...")
        
        metrics = BinaryClassificationMetrics()
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc=f'Evaluating on {split}'):
                images = images.to(self.device)
                targets = targets.float().to(self.device)
                
                # Forward pass
                logits = self.model(images).view(-1)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                
                # Update metrics
                metrics.update(
                    preds=preds,
                    targets=targets.long(),
                    probs=probs
                )
        
        # Compute metrics
        computed_metrics = metrics.compute()
        classification_report = metrics.get_classification_report()
        confusion_matrix = metrics.get_confusion_matrix()
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Evaluation Results - {dataset_name}")
        print(f"{'='*60}")
        print(f"Sensitivity (Recall): {computed_metrics['sensitivity']:.4f} ⭐ (PRIMARY)")
        print(f"Specificity: {computed_metrics['specificity']:.4f}")
        print(f"Accuracy: {computed_metrics['accuracy']:.4f}")
        print(f"Precision: {computed_metrics['precision']:.4f}")
        print(f"F1-Score: {computed_metrics['f1_score']:.4f}")
        if 'auc_roc' in computed_metrics:
            print(f"AUC-ROC: {computed_metrics['auc_roc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              NORMAL    DR")
        print(f"Actual NORMAL   {confusion_matrix[0][0]:5d}  {confusion_matrix[0][1]:5d}")
        print(f"        DR      {confusion_matrix[1][0]:5d}  {confusion_matrix[1][1]:5d}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report)
        
        print(f"\nClinical Interpretation:")
        print(f"- True Positives (DR detected correctly): {computed_metrics['true_positives']}")
        print(f"- False Negatives (DR missed): {computed_metrics['false_negatives']} ⚠️")
        print(f"- False Positives (Normal flagged as DR): {computed_metrics['false_positives']}")
        print(f"- True Negatives (Normal correctly identified): {computed_metrics['true_negatives']}")
        
        return computed_metrics, classification_report, confusion_matrix
    
    def save_results(self, metrics: dict, split: str = 'test', save_path: str = None):
        """Save evaluation results to JSON."""
        if save_path is None:
            save_path = Path('models/binary_dr') / f'evaluation_{split}.json'
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'split': split,
            'metrics': metrics,
            'model_info': {
                'device': str(self.device),
                'model_type': type(self.model).__name__,
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {save_path}")
