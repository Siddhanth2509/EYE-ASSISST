"""
Fundus Classifier Inference Module
Binary classifier to validate if an image is a fundus photograph.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import timm

# Global classifier instance
_fundus_classifier = None


class FundusClassifier:
    """Binary classifier for fundus vs non-fundus images."""
    
    def __init__(self, checkpoint_path: Path, threshold: float = 0.9, device: str = None):
        """
        Initialize the fundus classifier.
        
        Args:
            checkpoint_path: Path to model checkpoint
            threshold: Confidence threshold for fundus classification (default 0.9)
            device: Device to run on (auto-detect if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.model_name = None
        
        # Image transforms (match training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained fundus classifier model."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Fundus classifier not found at {self.checkpoint_path}")
        
        print(f"Loading fundus classifier from {self.checkpoint_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Get model architecture from checkpoint
        self.model_name = checkpoint.get('model_name', 'mobilenetv2_100')
        
        # Create model
        self.model = timm.create_model(self.model_name, pretrained=False, num_classes=2)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get training metrics if available
        val_acc = checkpoint.get('val_accuracy', 0) * 100
        val_f1 = checkpoint.get('val_f1', 0) * 100
        
        print(f"✅ Fundus classifier loaded successfully")
        print(f"   Model: {self.model_name}")
        print(f"   Threshold: {self.threshold * 100:.1f}%")
        print(f"   Val Accuracy: {val_acc:.1f}%, F1: {val_f1:.1f}%")
        print(f"   Device: {self.device}")
    
    def predict(self, image: Image.Image) -> Tuple[bool, float, str]:
        """
        Predict if image is a fundus photograph.
        
        Args:
            image: PIL Image to classify
            
        Returns:
            Tuple of (is_fundus, probability, message)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]
        
        # Class 0 = fundus, Class 1 = non-fundus (based on training)
        prob_fundus = float(probs[0])
        prob_non_fundus = float(probs[1])
        
        print(f"[FundusClassifier] prob_fundus={prob_fundus:.4f}, prob_non_fundus={prob_non_fundus:.4f}, threshold={self.threshold:.2f}")
        
        is_fundus = prob_fundus >= self.threshold
        
        if is_fundus:
            message = f"Valid fundus image ({prob_fundus*100:.1f}% confidence)"
        else:
            message = f"This does not appear to be a fundus photograph. The classifier detected it as non-fundus with {prob_non_fundus*100:.1f}% confidence. Please upload a retinal fundus image."
        
        return is_fundus, prob_fundus, message


def initialize_fundus_classifier(checkpoint_path: Path, threshold: float = 0.9, device: str = None):
    """
    Initialize the global fundus classifier.
    
    Args:
        checkpoint_path: Path to model checkpoint
        threshold: Confidence threshold (default 0.9 = 90%)
        device: Device to run on
    """
    global _fundus_classifier
    _fundus_classifier = FundusClassifier(
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        device=device
    )


def get_fundus_classifier() -> FundusClassifier:
    """Get the global fundus classifier instance."""
    if _fundus_classifier is None:
        raise RuntimeError(
            "Fundus classifier not initialized. "
            "Call initialize_fundus_classifier() first."
        )
    return _fundus_classifier
