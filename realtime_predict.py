#!/usr/bin/env python3
"""
Real-time prediction demo script.
This script shows how to use a trained model to make predictions on individual samples.
It includes a class for making predictions and visualizing results.


Usage:
    python realtime_predict.py --ckpt_path /path/to/checkpoint.ckpt
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

from graph_enet.data.scarfDataset_splineConv import scarfDataset_splineConv
from graph_enet.hpe_gnn.model.hpegnn import hpeGnn_splineConv, hpeGnn_splineConv_single_weight
from graph_enet.hpe_gnn.utils.dataset_utils import dataset_split

class PosePredictor:
    """
    A wrapper class for making pose predictions with a trained model.
    """
    
    def __init__(self, checkpoint_path, arch='two_weights', device='cpu'):
        """
        Initialize the pose predictor.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            arch: Model architecture ('single_weight' or 'two_weights')
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.arch = arch
        
        print(f"Loading model from: {checkpoint_path}")
        
        # Load model
        if arch == 'single_weight':
            self.model = hpeGnn_splineConv_single_weight.load_from_checkpoint(
                checkpoint_path, map_location=device
            )
        else:
            self.model = hpeGnn_splineConv.load_from_checkpoint(
                checkpoint_path, map_location=device
            )
        
        self.model.eval()
        self.model = self.model.to(device)
        
        print(f"Model loaded successfully on {device}")
    
    def predict_single_sample(self, sample_data):
        """
        Make a prediction on a single graph sample.
        
        Args:
            sample_data: PyTorch Geometric Data object
            
        Returns:
            Predicted pose coordinates as numpy array
        """
        with torch.no_grad():
            # Move data to device
            sample_data = sample_data.to(self.device)
            
            # Forward pass
            prediction, _ = self.model.forward(
                sample_data.x, 
                sample_data.edge_index, 
                sample_data.edge_attr, 
                sample_data.pos, 
                sample_data.batch
            )
            
            return prediction.cpu().numpy()
    
    def predict_batch(self, data_loader):
        """
        Make predictions on a batch of samples.
        
        Args:
            data_loader: PyTorch DataLoader
            
        Returns:
            List of predictions
        """
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                pred, _ = self.model.forward(
                    batch.x, batch.edge_index, batch.edge_attr, batch.pos, batch.batch
                )
                predictions.append(pred.cpu().numpy())
        
        return predictions
    
    def visualize_prediction(self, prediction, ground_truth=None, save_path=None, show=True):
        """
        Visualize a pose prediction.
        
        Args:
            prediction: Predicted pose coordinates (26 values for 13 joints)
            ground_truth: Ground truth pose coordinates (optional)
            save_path: Path to save visualization (optional)
            show: Whether to display the plot
        """
        # Reshape to (13, 2) for 13 joints with x,y coordinates
        pred_joints = prediction.reshape(-1, 2)
        
        plt.figure(figsize=(10, 6))
        
        if ground_truth is not None:
            gt_joints = ground_truth.reshape(-1, 2)
            plt.subplot(1, 2, 1)
            plt.scatter(gt_joints[:, 0], gt_joints[:, 1], c='green', s=50, label='Ground Truth')
            plt.plot(gt_joints[:, 0], gt_joints[:, 1], 'g-', alpha=0.5)
            plt.title('Ground Truth Pose')
            plt.xlim(0, 640)
            plt.ylim(0, 480)
            plt.gca().invert_yaxis()
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
        
        plt.scatter(pred_joints[:, 0], pred_joints[:, 1], c='red', s=50, label='Prediction')
        plt.plot(pred_joints[:, 0], pred_joints[:, 1], 'r-', alpha=0.5)
        
        # Add joint numbers
        for i, (x, y) in enumerate(pred_joints):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title('Predicted Pose')
        plt.xlim(0, 640)
        plt.ylim(0, 480)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

def demo_realtime_prediction(checkpoint_path, data_path, arch='two_weights'):
    """
    Demonstrate real-time prediction capabilities.
    """
    print("="*50)
    print("Real-time Pose Prediction Demo")
    print("="*50)
    
    # Initialize predictor
    predictor = PosePredictor(checkpoint_path, arch)
    
    # Load test data
    print(f"Loading test data from: {data_path}")
    dataset = scarfDataset_splineConv(
        data_path, rf_size=14, alpha=1.0, C=0.3, res=(640, 480)
    )
    
    # Get a small test set
    _, val_dataset = dataset_split(dataset, style='dev', fraction=0.01, dataset_label='test')
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    print(f"Loaded {len(val_dataset)} test samples")
    
    # Demo single sample prediction
    print("\n--- Single Sample Prediction Demo ---")
    sample = next(iter(test_loader))
    
    print(f"Input graph has {sample.num_nodes} nodes and {sample.num_edges} edges")
    print(f"Ground truth shape: {sample.y.shape}")
    
    # Make prediction
    prediction = predictor.predict_single_sample(sample)
    ground_truth = sample.y.cpu().numpy()
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction (first 6 coordinates): {prediction.flatten()[:6]}")
    print(f"Ground truth (first 6 coordinates): {ground_truth.flatten()[:6]}")
    
    # Visualize results
    predictor.visualize_prediction(
        prediction[0], 
        ground_truth[0], 
        save_path='realtime_prediction_demo.png',
        show=False
    )
    
    # Demo batch prediction
    print("\n--- Batch Prediction Demo ---")
    batch_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    batch_predictions = predictor.predict_batch(batch_loader)
    
    print(f"Made predictions on {len(batch_predictions)} batches")
    print(f"First batch shape: {batch_predictions[0].shape}")
    
    print("\n--- Demo Complete ---")
    print("Check 'realtime_prediction_demo.png' for visualization")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time prediction demo")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='/home/dberretta-iit.local/data/new_scarfGNN', help='Path to dataset')
    parser.add_argument('--arch', type=str, default='two_weights', choices=['single_weight', 'two_weights'], help='Model architecture')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device for inference')
    
    args = parser.parse_args()
    
    # Run demo
    demo_realtime_prediction(args.ckpt_path, args.data_path, args.arch)
