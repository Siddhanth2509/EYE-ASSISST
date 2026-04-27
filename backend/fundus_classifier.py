"""
Fundus Classifier Inference Module
Fast binary classification: fundus vs non-fundus
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
from pathlib import Path

class FundusClassifier:
    """
    Binary classifier to gate DR model.
    Outputs probability that image is a fundus photograph.
    """
    
    def __init__(self, checkpoint_path=None, threshold=0.7):
        """
        Args:
            checkpoint_path: Path to trained classifier checkpoint
            threshold: Minimum probability to consider image as fundus (0.0-1.0)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.model_loaded = False
        
        # Image preprocessing (must match training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model if checkpoint provided
        if checkpoint_path:
            self.load_model(checkpoint_path)
    
    def load_model(self, checkpoint_path):
        """Load trained classifier from checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"[WARNING] Fundus classifier not found at {checkpoint_path}")
            print("   System will use fallback validation (DR model confidence)")
            self.model_loaded = False
            return
        
        try:
            print(f"Loading fundus classifier from {checkpoint_path}...")
            
            # Load checkpoint (weights_only=False for compatibility with PyTorch 2.6+)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Create model architecture
            self.model = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=2)
            
            # Handle both checkpoint formats (dict with 'model_state_dict' key OR direct state_dict)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                val_acc = checkpoint.get('val_acc', 0)
                val_f1 = checkpoint.get('val_f1', 0)
            else:
                # Direct state_dict
                state_dict = checkpoint
                val_acc = 0
                val_f1 = 0
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            
            print(f"[OK] Fundus classifier loaded successfully")
            print(f"   Model: mobilenetv2_100")
            print(f"   Threshold: {self.threshold*100:.1f}%")
            if val_acc > 0:
                print(f"   Val Accuracy: {val_acc*100:.1f}%, F1: {val_f1*100:.1f}%")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            # FAIL HARD - don't silently fall back for medical system
            raise RuntimeError(
                f"CRITICAL: Fundus classifier failed to load\n"
                f"   Error: {e}\n"
                f"   Type: {type(e).__name__}\n"
                f"   Path: {checkpoint_path}\n"
                f"   Cannot start backend without fundus validation."
            )
    
    def predict(self, image: Image.Image) -> tuple[bool, float, str]:
        """
        Predict if image is a fundus photograph.
        
        Args:
            image: PIL Image
        
        Returns:
            (is_fundus, probability, message)
            - is_fundus: True if image is fundus (prob >= threshold)
            - probability: Probability that image is fundus (0.0-1.0)
            - message: Explanation message
        """
        if not self.model_loaded:
            # Fallback: assume it's fundus (will be validated by DR model confidence)
            return True, 1.0, "Classifier not loaded, using fallback validation"
        
        try:
            # Preprocess
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0]
            
            # probs[0] = non-fundus, probs[1] = fundus
            prob_non_fundus = float(probs[0])
            prob_fundus = float(probs[1])
            
            # Debug output
            print(f"[FundusClassifier] prob_fundus={prob_fundus:.4f}, prob_non_fundus={prob_non_fundus:.4f}, threshold={self.threshold:.2f}")
            
            # Decision
            is_fundus = prob_fundus >= self.threshold
            
            if is_fundus:
                message = f"Fundus detected (confidence: {prob_fundus*100:.1f}%)"
            else:
                message = (
                    f"This does not appear to be a fundus photograph. "
                    f"The classifier detected it as non-fundus with {prob_non_fundus*100:.1f}% confidence. "
                    f"Please upload a retinal fundus image."
                )
            
            return is_fundus, prob_fundus, message
            
        except Exception as e:
            print(f"Fundus classifier error: {e}")
            # On error, be conservative and reject
            return False, 0.0, "Unable to validate image. Please ensure it is a clear fundus photograph."
    
    def is_fundus_image(self, image: Image.Image) -> tuple[bool, str]:
        """
        Check if image is fundus (compatible with existing validation interface).
        
        Returns:
            (is_valid, error_message)
        """
        is_fundus, prob, message = self.predict(image)
        
        if is_fundus:
            return True, "Valid fundus image"
        else:
            return False, message

# ============================================================================
# Singleton instance
# ============================================================================

# This will be imported by main.py
fundus_classifier = None

def initialize_fundus_classifier(checkpoint_path=None, threshold=0.9):
    """Initialize global fundus classifier instance"""
    global fundus_classifier
    fundus_classifier = FundusClassifier(checkpoint_path, threshold)
    return fundus_classifier

def get_fundus_classifier():
    """Get global fundus classifier instance"""
    global fundus_classifier
    if fundus_classifier is None:
        # Auto-initialize with default path (use absolute path)
        BASE_DIR = Path(__file__).resolve().parent.parent
        default_path = BASE_DIR / "phase0_fundus_classifier" / "models" / "fundus_classifier" / "best_fundus_classifier.pt"
        initialize_fundus_classifier(default_path, threshold=0.9)
    return fundus_classifier
