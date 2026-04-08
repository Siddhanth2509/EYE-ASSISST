#!/usr/bin/env python3
"""
Phase 3: Prepare Multi-Label Training Data

Creates train/val CSVs with multi-label annotations from existing datasets.
Currently uses DR (Diabetic Retinopathy) severity as proxy for multi-disease labels.

This is a BOOTSTRAP setup - you should replace these synthetic labels with 
real multi-label annotations when available.

Usage:
    python phase3_multi_disease/prepare_data.py --dataset dr_unified_v2
    python phase3_multi_disease/prepare_data.py --dataset odir
    python phase3_multi_disease/prepare_data.py --dataset catract
"""

import argparse
import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

# Disease labels (6 diseases)
DISEASE_LABELS = ['dr', 'glaucoma', 'amd', 'cataract', 'hypertensive', 'myopic']

def create_bootstrap_labels(image_path, dataset_type):
    """
    Create synthetic multi-label annotations.
    
    IMPORTANT: These are BOOTSTRAP labels for testing the framework.
    In production, use real clinical annotations!
    
    For now, we use heuristics:
    - dr_unified_v2: Images often have DR, rarely have other conditions
    - odir: Mixed conditions, more realistic distribution
    - catract: High prevalence of cataracts
    """
    
    # Seed randomness per image for consistency
    np.random.seed(hash(str(image_path)) % (2**32))
    
    labels = [0] * len(DISEASE_LABELS)  # [dr, glaucoma, amd, cataract, hypertensive, myopic]
    
    if dataset_type == 'dr_unified_v2':
        # These images are primarily DR dataset
        # High DR rate, low other conditions
        labels[0] = random.choice([0, 0, 0, 0, 1])  # ~20% DR
        labels[1] = random.choice([0, 0, 0, 0, 0, 0, 0, 1])  # ~12% Glaucoma
        labels[2] = random.choice([0, 0, 0, 0, 0, 0, 0, 1])  # ~12% AMD
        labels[3] = random.choice([0, 0, 0, 0, 0])  # ~0% Cataract
        labels[4] = random.choice([0, 0, 0, 0, 0])  # ~0% Hypertensive
        labels[5] = random.choice([0, 0, 0, 0, 0])  # ~0% Myopic
        
    elif dataset_type == 'odir':
        # ODIR has mixed conditions
        # More realistic multi-disease distribution
        labels[0] = random.choice([0, 0, 1])  # ~33% DR
        labels[1] = random.choice([0, 0, 0, 1])  # ~25% Glaucoma
        labels[2] = random.choice([0, 0, 0, 1])  # ~25% AMD
        labels[3] = random.choice([0, 0, 0, 1])  # ~25% Cataract
        labels[4] = random.choice([0, 0, 0, 1])  # ~25% Hypertensive
        labels[5] = random.choice([0, 0, 0, 1])  # ~25% Myopic
        
    elif dataset_type == 'catract':
        # CATRACT focused on cataracts
        labels[3] = 1  # High cataract rate
        labels[0] = random.choice([0, 0, 1])  # ~33% also has DR
        labels[1] = random.choice([0, 0, 0, 1])  # ~25% Glaucoma
        labels[2] = random.choice([0, 0, 0, 1])  # ~25% AMD
    
    return labels


def prepare_dataset(dataset_type, data_root='Dataset', output_dir='phase3_multi_disease/data'):
    """Prepare train/val CSVs for a dataset."""
    
    print(f"\n{'='*70}")
    print(f"Preparing Phase 3 data from: {dataset_type}")
    print(f"{'='*70}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map dataset types to paths
    dataset_paths = {
        'dr_unified_v2': 'Dataset/dr_unified_v2/dr_unified_v2',
        'odir': 'Dataset/ODIR/Training Set',
        'catract': 'Dataset/CATRACT/dataset'
    }
    
    if dataset_type not in dataset_paths:
        print(f"❌ Unknown dataset type: {dataset_type}")
        print(f"   Available: {', '.join(dataset_paths.keys())}")
        return False
    
    dataset_path = Path(dataset_paths[dataset_type])
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    # Collect all images
    print(f"📂 Collecting images from {dataset_path}...")
    image_files = list(dataset_path.glob('**/*.jpg')) + list(dataset_path.glob('**/*.png')) + list(dataset_path.glob('**/*.jpeg'))
    
    print(f"   Found {len(image_files)} images")
    
    if len(image_files) < 50:
        print(f"❌ Not enough images ({len(image_files)} < 50)")
        return False
    
    # Create multi-label annotations
    print(f"\n🏷️  Creating multi-label annotations...")
    samples = []
    
    for img_path in image_files:
        # Get relative path for CSV
        try:
            rel_path = img_path.relative_to(Path.cwd())
        except ValueError:
            rel_path = img_path
        
        # Generate bootstrap labels
        labels = create_bootstrap_labels(rel_path, dataset_type)
        
        samples.append({
            'image_path': str(rel_path),
            'dr': labels[0],
            'glaucoma': labels[1],
            'amd': labels[2],
            'cataract': labels[3],
            'hypertensive': labels[4],
            'myopic': labels[5]
        })
    
    # Print label distribution
    print(f"\n📊 Label distribution (bootstrap):")
    for i, disease in enumerate(DISEASE_LABELS):
        pos_count = sum(s[disease] for s in samples)
        pct = 100 * pos_count / len(samples)
        print(f"   {disease:15s}: {pos_count:6d}/{len(samples)} ({pct:5.1f}%)")
    
    # Split into train/val (80/20)
    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)
    
    # Write CSVs
    print(f"\n💾 Writing CSV files...")
    
    train_csv = output_dir / f'train_{dataset_type}.csv'
    val_csv = output_dir / f'val_{dataset_type}.csv'
    
    # Write training CSV
    with open(train_csv, 'w') as f:
        f.write('image_path,dr,glaucoma,amd,cataract,hypertensive,myopic\n')
        for sample in train_samples:
            line = f"{sample['image_path']},{sample['dr']},{sample['glaucoma']},{sample['amd']},{sample['cataract']},{sample['hypertensive']},{sample['myopic']}\n"
            f.write(line)
    
    # Write validation CSV
    with open(val_csv, 'w') as f:
        f.write('image_path,dr,glaucoma,amd,cataract,hypertensive,myopic\n')
        for sample in val_samples:
            line = f"{sample['image_path']},{sample['dr']},{sample['glaucoma']},{sample['amd']},{sample['cataract']},{sample['hypertensive']},{sample['myopic']}\n"
            f.write(line)
    
    print(f"   ✅ Train CSV: {train_csv} ({len(train_samples)} samples)")
    print(f"   ✅ Val CSV: {val_csv} ({len(val_samples)} samples)")
    
    # Print command to start training
    print(f"\n🚀 Ready to train! Run:")
    print(f"\n   python phase3_multi_disease/train.py \\")
    print(f"     --train_csv {train_csv} \\")
    print(f"     --val_csv {val_csv} \\")
    print(f"     --epochs 50 \\")
    print(f"     --batch_size 32 \\")
    print(f"     --model resnet50")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare Phase 3 multi-label training data')
    parser.add_argument('--dataset', type=str, default='dr_unified_v2',
                        choices=['dr_unified_v2', 'odir', 'catract'],
                        help='Which dataset to use')
    parser.add_argument('--data_root', type=str, default='Dataset',
                        help='Root directory for datasets')
    parser.add_argument('--output_dir', type=str, default='phase3_multi_disease/data',
                        help='Output directory for CSVs')
    
    args = parser.parse_args()
    
    success = prepare_dataset(args.dataset, args.data_root, args.output_dir)
    
    if not success:
        print(f"\n❌ Failed to prepare dataset")
        exit(1)
    
    print(f"\n{'='*70}")
    print(f"✅ Data preparation complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
