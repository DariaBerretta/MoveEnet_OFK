#!/usr/bin/env python3
"""
Simplified script to extract predictions from a trained GraphEnet-v2 model.
This script focuses on the core functionality of loading a model and extracting predictions.

Usage:
    python simple_predict.py --ckpt_path /path/to/checkpoint.ckpt
"""

import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

# Import required modules
from graph_enet.data.scarfDataset_splineConv import scarfDataset_splineConv
from graph_enet.hpe_gnn.model.hpegnn import hpeGnn_splineConv, hpeGnn_splineConv_single_weight
from graph_enet.hpe_gnn.utils.dataset_utils import dataset_split

def simple_predict_and_visualize(ckpt_path, data_path, arch='two_weights'):
    """
    Simple function to extract predictions from a trained model.
    
    Args:
        ckpt_path: Path to model checkpoint
        data_path: Path to dataset
        arch: Architecture type ('single_weight' or 'two_weights')
    """
    print(f"Loading model from: {ckpt_path}")
    
    # Load model from checkpoint
    if arch == 'single_weight':
        model = hpeGnn_splineConv_single_weight.load_from_checkpoint(ckpt_path, map_location='cpu')
    else:
        model = hpeGnn_splineConv.load_from_checkpoint(ckpt_path, map_location='cpu')
    
    model.eval()
    print("Model loaded successfully!")
    
    # Load dataset
    print(f"Loading dataset from: {data_path}")
    dataset = scarfDataset_splineConv(
        data_path,
        rf_size=14, alpha=1.0, C=0.3, res=(640, 480)
    )
    
    # Use a small fraction for testing
    _, val_dataset = dataset_split(dataset, style='dev', fraction=0.01, dataset_label='scarfDataset_splineConv')
    
    # Create dataloader
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"Dataset loaded. Test samples: {len(val_dataset)}")
    
    # Extract predictions
    predictions = []
    ground_truths = []
    
    print("Extracting predictions...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Forward pass
            pred, _ = model.forward(batch.x, batch.edge_index, batch.edge_attr, batch.pos, batch.batch)
            
            # Store results
            predictions.append(pred.cpu().numpy())
            ground_truths.append(batch.y.cpu().numpy())
            
            # Show progress
            if i < 5:  # Only process first 5 samples for demo
                print(f"Sample {i+1}:")
                print(f"  Prediction shape: {pred.shape}")
                print(f"  Ground truth shape: {batch.y.shape}")
                print(f"  Prediction (first 6 values): {pred.flatten()[:6].cpu().numpy()}")
                print(f"  Ground truth (first 6 values): {batch.y.flatten()[:6].cpu().numpy()}")
                print()
            
            if i >= 4:  # Stop after 5 samples for demo
                break
    
    print("Prediction extraction complete!")
    print(f"Successfully extracted predictions for {len(predictions)} samples")
    
    return predictions, ground_truths

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple prediction extraction")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='/home/dberretta-iit.local/data/new_scarfGNN', help='Path to dataset')
    parser.add_argument('--arch', type=str, default='two_weights', choices=['single_weight', 'two_weights'], help='Model architecture')
    
    args = parser.parse_args()
    
    # Run prediction extraction
    predictions, ground_truths = simple_predict_and_visualize(args.ckpt_path, args.data_path, args.arch)
