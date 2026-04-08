# Phase 3: Multi-Disease Detection

This phase implements multi-label classification for detecting multiple eye diseases from fundus images.

## Diseases Detected

1. **Diabetic Retinopathy (DR)** - Damage to retinal blood vessels due to diabetes
2. **Glaucoma** - Optic nerve damage, often due to increased eye pressure
3. **Age-Related Macular Degeneration (AMD)** - Deterioration of central retina
4. **Cataracts** - Clouding of the eye's lens
5. **Hypertensive Retinopathy** - Damage from high blood pressure
6. **Myopic Macular Degeneration** - Retinal changes from severe myopia

## Dataset Format

The training script expects CSV files with the following format:

```csv
image_path,dr,glaucoma,amd,cataract,hypertensive,myopic
Dataset/train/img1.jpg,1,0,0,0,0,0
Dataset/train/img2.jpg,1,1,0,0,0,0
Dataset/train/img3.jpg,0,0,1,0,1,0
```

Each disease column should be:
- `1` if the disease is present
- `0` if the disease is absent

Note: Multiple diseases can be present in a single image (multi-label).

## Directory Structure

```
phase3_multi_disease/
├── train.py              # Training script
├── README.md             # This file
├── data/                 # CSV files
│   ├── train.csv         # Training labels
│   ├── val.csv           # Validation labels
│   └── test.csv          # Test labels (optional)
├── configs/              # Configuration files
├── models/               # Saved model architectures
└── checkpoints/          # Training checkpoints
    └── multidisease_resnet50_TIMESTAMP/
        ├── best_model.pt
        ├── config.json
        └── history.json
```

## Usage

### 1. Prepare Your Data

Create CSV files with image paths and disease labels:

```bash
# Example train.csv location
phase3_multi_disease/data/train.csv
phase3_multi_disease/data/val.csv
```

### 2. Train the Model

Basic training:
```bash
python phase3_multi_disease/train.py \
    --data_root Dataset/ \
    --train_csv phase3_multi_disease/data/train.csv \
    --val_csv phase3_multi_disease/data/val.csv \
    --epochs 50
```

Advanced training with custom settings:
```bash
python phase3_multi_disease/train.py \
    --data_root Dataset/ \
    --train_csv phase3_multi_disease/data/train.csv \
    --val_csv phase3_multi_disease/data/val.csv \
    --model resnet50 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --image_size 224 \
    --num_workers 4 \
    --patience 15
```

### 3. Monitor Training

The script will:
- Print per-epoch metrics (loss, F1, AUC)
- Save best model based on F1 macro score
- Save training history to JSON
- Implement early stopping if no improvement

Example output:
```
Epoch 10/50
------------------------------------------------------------
Training: 100%|████████| 245/245 [02:15<00:00]
Train Loss: 0.2341
Train F1 (micro/macro): 0.8234 / 0.7891

Validation: 100%|████████| 62/62 [00:32<00:00]
Val Loss: 0.2567
Val F1 (micro/macro): 0.8012 / 0.7645
Val AUC (macro): 0.8823

Per-class F1 scores:
  DR             : 0.8912
  Glaucoma       : 0.7234
  AMD            : 0.7891
  Cataract       : 0.8456
  Hypertensive   : 0.6789
  Myopic         : 0.7012

✓ Saved best model (F1 macro: 0.7645)
```

## Model Architecture

The model uses:
- **Backbone**: ResNet50, ResNet18, or EfficientNet-B0 (pretrained on ImageNet)
- **Classifier**: Multi-layer perceptron with dropout
- **Output**: 6 sigmoid outputs (one per disease)
- **Loss**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `Dataset/` | Root directory containing images |
| `--train_csv` | `phase3_multi_disease/data/train.csv` | Training labels CSV |
| `--val_csv` | `phase3_multi_disease/data/val.csv` | Validation labels CSV |
| `--model` | `resnet50` | Backbone architecture (resnet18, resnet50, efficientnet_b0) |
| `--num_diseases` | `6` | Number of disease classes |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--weight_decay` | `1e-5` | Weight decay for AdamW optimizer |
| `--image_size` | `224` | Input image size |
| `--num_workers` | `4` | Number of dataloader workers |
| `--patience` | `10` | Early stopping patience |
| `--no_pretrained` | False | Train from scratch without ImageNet weights |
| `--seed` | `42` | Random seed for reproducibility |

## Data Augmentation

Training uses extensive augmentation:
- Random crop after resize
- Horizontal and vertical flips
- Random rotation (±20°)
- Color jitter (brightness, contrast, saturation)

Validation uses only center crop and normalization.

## Evaluation Metrics

The script computes:
- **F1 Score** (micro and macro averaged)
- **Precision** (per-class)
- **Recall** (per-class)
- **AUC-ROC** (per-class and macro averaged)

Best model is selected based on **F1 macro** score.

## Loading a Trained Model

```python
import torch
from phase3_multi_disease.train import MultiDiseaseModel

# Load checkpoint
checkpoint = torch.load('phase3_multi_disease/checkpoints/.../best_model.pt')

# Create model
model = MultiDiseaseModel(backbone='resnet50', num_diseases=6)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
logits = model(images)
probabilities = torch.sigmoid(logits)
predictions = (probabilities > 0.5).int()
```

## Integration with Backend

After training, the model can be integrated into the backend API:

1. Copy trained model to backend models directory:
```bash
cp phase3_multi_disease/checkpoints/.../best_model.pt backend/models/multi_disease.pt
```

2. Update `backend/main.py` to load and use the multi-disease model

3. Add endpoint for multi-disease prediction

## Dataset Sources

For training this model, you'll need a multi-label fundus dataset. Potential sources:

- **ODIR-5K**: Multi-disease dataset with 8 classes
- **RFMiD**: Retinal Fundus Multi-disease Image Dataset (46 conditions)
- **APTOS + Messidor** (for DR) combined with:
  - **ACRIMA** or **REFUGE** (for Glaucoma)
  - **ADAM** (for AMD)

## Performance Expectations

With proper training on sufficient data:
- **DR**: F1 > 0.85 (abundant training data)
- **Glaucoma**: F1 > 0.75 (moderate data availability)
- **AMD**: F1 > 0.70 (less common in datasets)
- **Cataracts**: F1 > 0.80 (distinctive features)

## Tips for Good Performance

1. **Balance your dataset**: Use class weights or oversampling for rare diseases
2. **Use pretrained weights**: Start with ImageNet-pretrained backbones
3. **Monitor per-class metrics**: Some diseases are harder to detect than others
4. **Increase image size**: Try 384x384 or 512x512 for subtle features
5. **Ensemble models**: Combine predictions from multiple models
6. **Adjust threshold**: Default 0.5 may not be optimal for all diseases

## Future Improvements

- [ ] Add test.py for evaluation on test set
- [ ] Implement class weights for imbalanced data
- [ ] Add TensorBoard logging
- [ ] Implement Grad-CAM for multi-disease visualization
- [ ] Add model ensembling
- [ ] Support for additional diseases
- [ ] Cross-validation support

## Citation

If you use this implementation, please cite:

```
@software{eye_assist_phase3,
  title={EYE-ASSISST Phase 3: Multi-Disease Detection},
  year={2026},
  url={https://github.com/yourusername/eye-assist}
}
```
