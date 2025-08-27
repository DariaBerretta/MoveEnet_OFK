#!/usr/bin/env python3
"""
Diagnostic script to analyze model predictions and identify potential issues.
This script focuses on loading a model, running predictions on a small dataset,
and printing detailed statistics about the predictions and ground truth.
It helps in diagnosing issues like low prediction variance or incorrect model behavior.
"""

import torch
import numpy as np
from torch_geometric.loader import DataLoader
import warnings
warnings.filterwarnings("ignore")

from graph_enet.data.scarfDataset_splineConv import scarfDataset_splineConv
from graph_enet.hpe_gnn.model.hpegnn import hpeGnn_splineConv
from graph_enet.hpe_gnn.utils.dataset_utils import dataset_split, new_dataset_split

def analyze_model_predictions(ckpt_path, data_path):
    """Analyze model predictions to diagnose issues."""
    
    print("="*60)
    print("MODEL PREDICTION ANALYSIS")
    print("="*60)
    
    # Load model
    model = hpeGnn_splineConv.load_from_checkpoint(ckpt_path, map_location='cpu')
    model.eval()
    
    # Load dataset
    dataset = scarfDataset_splineConv(data_path, rf_size=14, alpha=1.0, C=0.3, res=(640, 480))
    # _, val_dataset = dataset_split(dataset, style='dev', fraction=0.01, dataset_label='scarfDataset_splineConv')
    _, _, test_dataset = new_dataset_split(dataset, style='dev', fraction=0.01, dataset_label='scarfDataset_splineConv')
    # test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # print(f"Dataset loaded: {len(val_dataset)} samples")
    print(f"Dataset loaded: {len(test_dataset)} samples")
    
    # Analyze first few samples
    predictions_all = []
    ground_truths_all = []
    node_features_all = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 10:  # Analyze first 10 samples
                break
                
            # Forward pass
            pred, node_features = model.forward(batch.x, batch.edge_index, batch.edge_attr, batch.pos, batch.batch)
            
            predictions_all.append(pred.cpu().numpy())
            ground_truths_all.append(batch.y.cpu().numpy())
            node_features_all.append(node_features.cpu().numpy())
            
            print(f"\nSample {i+1}:")
            print(f"  Input graph: {batch.x.shape[0]} nodes, {batch.edge_index.shape[1]} edges")
            print(f"  Node positions range: X[{batch.pos[:, 0].min():.1f}, {batch.pos[:, 0].max():.1f}], Y[{batch.pos[:, 1].min():.1f}, {batch.pos[:, 1].max():.1f}]")
            
            # Reshape ground truth and prediction for analysis
            y_reshaped = batch.y.reshape(-1)
            pred_reshaped = pred.reshape(-1)
            
            print(f"  Ground truth shape: {batch.y.shape}, pred shape: {pred.shape}")
            print(f"  Ground truth range: [{y_reshaped.min():.1f}, {y_reshaped.max():.1f}]")
            print(f"  Prediction range: [{pred_reshaped.min():.1f}, {pred_reshaped.max():.1f}]")
            print(f"  Prediction std: {pred_reshaped.std():.2f}")
    
    # Overall analysis
    predictions_all = np.array(predictions_all)
    ground_truths_all = np.array(ground_truths_all)
    
    print("\n" + "="*60)
    print("OVERALL ANALYSIS")
    print("="*60)
    
    # Prediction statistics
    pred_x = predictions_all[:, :, ::2]  # X coordinates
    pred_y = predictions_all[:, :, 1::2]  # Y coordinates
    
    gt_x = ground_truths_all[:, :, ::2]
    gt_y = ground_truths_all[:, :, 1::2]
    

    print(f"Predictions - X range: [{pred_x.min():.1f}, {pred_x.max():.1f}], std: {pred_x.std():.2f}")
    print(f"Predictions - Y range: [{pred_y.min():.1f}, {pred_y.max():.1f}], std: {pred_y.std():.2f}")
    print(f"Ground Truth - X range: [{gt_x.min():.1f}, {gt_x.max():.1f}], std: {gt_x.std():.2f}")
    print(f"Ground Truth - Y range: [{gt_y.min():.1f}, {gt_y.max():.1f}], std: {gt_y.std():.2f}")
    
    # Check if model is predicting same values
    # a low prediction variance indicates the model might be predicting similar values for all samples
    # indeed the prediction variance reflects how much the model's predictions vary across different samples
    pred_var = np.var(predictions_all, axis=0)
    print(f"Prediction variance across samples: mean={pred_var.mean():.2f}, min={pred_var.min():.2f}, max={pred_var.max():.2f}")

    if pred_var.mean() < 10: # 10 is an arbitrary threshold, it's unit measure is pixels
        print("⚠️  WARNING: Very low prediction variance - model might be predicting similar values for all samples!")
    
    # Check model parameters
    print(f"\nModel architecture: {type(model).__name__}")
    print(f"Model joints: {model.joints}")
    print(f"Model image size: {model.image_size}")
    
    # Analyze training configuration
    if hasattr(model, 'hparams'):
        print(f"Training hyperparameters:")
        for key, value in model.hparams.items():
            print(f"  {key}: {value}")
    
    return predictions_all, ground_truths_all, model

if __name__ == '__main__':
    ckpt_path = "/home/dberretta-iit.local/Documents/Repos/GraphEnet-v2/lightning_logs/scarf_dataset/version_6/checkpoints/epoch=8-step=9.ckpt"
    data_path = "/home/dberretta-iit.local/data/new_scarfGNN"
    
    predictions, ground_truths, model = analyze_model_predictions(ckpt_path, data_path)
