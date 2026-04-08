# src/metrics/metrics.py

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple, Optional

class BinaryClassificationMetrics:
    """
    Medical metrics for binary DR classification.
    
    Focus on sensitivity (recall) for DR class as primary metric,
    since false negatives are more dangerous than false positives.
    """
    
    def __init__(self, class_names: Optional[list] = None):
        """
        Args:
            class_names: List of class names, e.g., ['NORMAL', 'DR']
        """
        self.class_names = class_names or ['NORMAL', 'DR']
        self.reset()
    
    def reset(self):
        """Reset all accumulated predictions and targets."""
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, probs: Optional[torch.Tensor] = None):
        """
        Update metrics with batch predictions.
        
        Args:
            preds: Predicted class indices (batch_size,)
            targets: Ground truth class indices (batch_size,)
            probs: Predicted probabilities for positive class (batch_size,)
        """
        self.all_preds.extend(preds.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        if probs is not None:
            self.all_probs.extend(probs.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        # Basic metrics
        accuracy = accuracy_score(targets, preds)
        precision = precision_score(targets, preds, average='binary', zero_division=0)
        recall = recall_score(targets, preds, average='binary', zero_division=0)
        f1 = f1_score(targets, preds, average='binary', zero_division=0)
        
        # Medical terminology
        sensitivity = recall  # True Positive Rate (for DR class)
        # Specificity = TN / (TN + FP) = True Negative Rate
        cm = confusion_matrix(targets, preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            # Edge case: only one class present
            specificity = 0.0
        
        # AUC-ROC if probabilities are available
        auc_roc = None
        if len(self.all_probs) > 0:
            probs = np.array(self.all_probs)
            try:
                auc_roc = roc_auc_score(targets, probs)
            except ValueError:
                # Handle case where only one class present
                auc_roc = 0.0
        
        # Confusion matrix (reuse if already calculated)
        if 'cm' not in locals():
            cm = confusion_matrix(targets, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,  # Primary metric for medical screening
            'specificity': specificity,
            'f1_score': f1,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        }
        
        if auc_roc is not None:
            metrics['auc_roc'] = auc_roc
        
        return metrics
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        return classification_report(
            targets, preds,
            target_names=self.class_names,
            zero_division=0
        )
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        return confusion_matrix(targets, preds)
