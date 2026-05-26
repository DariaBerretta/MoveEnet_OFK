#!/usr/bin/env python3
"""
This script loads a trained checkpoint, runs inference on test data, and creates
visualizations of the predicted poses.

The script automatically detects which dataset to use based on the checkpoint path:
- ledge_dataset checkpoints → eh36m_spline_gamer dataset
- Improved_scarf_dataset checkpoints → scarfDataset_splineConv dataset

Usage:
    python extract_predictions_visualize.py --ckpt_path /path/to/checkpoint.ckpt --num_samples 10
    python extract_predictions_visualize.py --ckpt_path /home/dberretta-iit.local/Documents/Repos/GraphEnet-v2/lightning_logs/ledge_dataset/version_0/checkpoints/epoch=47-step=240.ckpt --num_samples 10
    python extract_predictions_visualize.py --ckpt_path /home/dberretta-iit.local/Documents/Repos/GraphEnet-v2/lightning_logs/ledge_dataset/version_1/checkpoints/epoch=47-step=11712.ckpt --num_samples 10
    python extract_predictions_visualize.py --ckpt_path /home/dberretta-iit.local/Documents/Repos/GraphEnet-v2/lightning_logs/Improved_scarf_dataset/version_4/checkpoints/epoch=37-step=18337.ckpt --num_samples 10
    
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import warnings
from datetime import datetime

# # Suppress warnings
# warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

# Import fixed metrics first
from graph_enet.utils.fixed_metrics import pck_error, mpjpe_error
# Monkey patch the metrics module before importing the model
import graph_enet.hpe_gnn.utils.metrics
graph_enet.hpe_gnn.utils.metrics.pck_error = pck_error
graph_enet.hpe_gnn.utils.metrics.mpjpe_error = mpjpe_error

# These metrics are not working so i don't import them
# from graph_enet.hpe_gnn.utils.metrics import pck_error, mpjpe_error

from graph_enet.data.scarfDataset_splineConv import scarfDataset_splineConv
from graph_enet.hpe_gnn.data.customDatasets import eh36m_spline_gamer, eh36m_spline_ledge
from graph_enet.hpe_gnn.model.hpegnn import hpeGnn_splineConv, hpeGnn_splineConv_single_weight
from graph_enet.hpe_gnn.utils.model_utils import GraphVisualization
from graph_enet.hpe_gnn.utils.dataset_utils import new_dataset_split, dataset_split, hpe_filter, schema_spline
import graph_enet.hpe_gnn.data.transforms as my_transforms
from graph_enet.hpe_gnn.scripts.config import cfg
import graph_enet.hpe_gnn.utils.training_utils as maps


def load_model_from_checkpoint(ckpt_path, arch='two_weights'):
    """
    Load a trained model from checkpoint.
    Args:
        ckpt_path: Path to the checkpoint file
        arch: Architecture type ('single_weight' or 'two_weights')
    Returns:
        Loaded PyTorch Lightning model
    """
    print(f"Loading model from checkpoint: {ckpt_path}")
    
    if arch == 'single_weight':
        model = hpeGnn_splineConv_single_weight.load_from_checkpoint(
            ckpt_path,
            map_location='cuda'  # Load on GPU
        )
    else:  # two_weights
        model = hpeGnn_splineConv.load_from_checkpoint(
            ckpt_path,
            map_location='cuda'
        )
    
    model.eval()  # Set to evaluation mode
    print(f"Model loaded successfully. Architecture: {arch}")
    return model


def extract_predictions(model, dataloader, device='cuda', max_samples=None):
    """
    Extract predictions from the model on a dataset.
    Args:
        model: Trained PyTorch Lightning model
        dataloader: DataLoader with test data
        device: Device to run inference on
        max_samples: Maximum number of samples to process (None for all)
    Returns:
        Dictionary with predictions, ground truths, and metadata
    """
    model = model.to(device)
    predictions = []
    ground_truths = []
    pck_scores = []
    mpjpe_scores = []
    samples_data = []
    
    print(f"Extracting predictions on {device}...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_samples and batch_idx >= max_samples:
                break
                
            # Move data to device
            batch = batch.to(device)
            
            # Forward pass
            pred, node_features = model.forward(
                batch.x, 
                batch.edge_index, 
                batch.edge_attr, 
                batch.pos, 
                batch.batch
            )
            
            # Reshape ground truth
            y_true = torch.reshape(batch.y, (batch.num_graphs, -1))
            
            # Calculate metrics
            pck = pck_error(y_true, pred, batch.th_pck, model.pck_multiplier)
            mpjpe = mpjpe_error(y_true, pred)
            
            # Store results
            predictions.append(pred.cpu().numpy()) # .cpu() is used to move tensor to CPU before converting to numpy (fixes bug)
            ground_truths.append(y_true.cpu().numpy())
            pck_scores.append(pck.cpu().numpy())
            mpjpe_scores.append(mpjpe.cpu().numpy())
            
            # Store sample data for visualization
            samples_data.append({
                'x': batch.x.cpu(),
                'pos': batch.pos.cpu(),
                'edge_index': batch.edge_index.cpu(),
                'edge_attr': batch.edge_attr.cpu(),
                'batch': batch.batch.cpu(),
                'y': batch.y.cpu(),
                'th_pck': batch.th_pck.cpu()
            })
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")
    
    results = {
        'predictions': np.concatenate(predictions, axis=0),
        'ground_truths': np.concatenate(ground_truths, axis=0),
        'pck_scores': np.array(pck_scores),
        'mpjpe_scores': np.array(mpjpe_scores),
        'samples_data': samples_data
    }
    
    print(f"Extraction complete. Processed {len(results['predictions'])} samples.")
    print(f"Average PCK: {np.mean(results['pck_scores']):.4f}")
    print(f"Average MPJPE: {np.mean(results['mpjpe_scores']):.4f}")
    
    return results


def visualize_predictions(results, save_dir, num_visualizations=5, timestamp=None):
    """
    Create visualizations of predictions vs ground truth.
    Args:
        results: Dictionary with predictions and ground truths
        save_dir: Directory to save visualizations
        num_visualizations: Number of samples to visualize
        timestamp: Optional timestamp string for unique filenames
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    predictions = results['predictions']
    ground_truths = results['ground_truths']
    samples_data = results['samples_data']
    
    num_visualizations = min(num_visualizations, len(predictions), len(samples_data))
    
    # Define skeleton connections based on human body structure
    # KEYPOINTS_MAP = {'head': 0, 'shoulder_right': 1, 'shoulder_left': 2, 'elbow_right': 3, 'elbow_left': 4,
    #                  'hip_left': 5, 'hip_right': 6, 'wrist_right': 7, 'wrist_left': 8, 'knee_right': 9, 'knee_left': 10,
    #                  'ankle_right': 11, 'ankle_left': 12}
    skeleton_connections = [
        (1, 2),   # shoulder_right to shoulder_left
        (1, 3),   # shoulder_right to elbow_right
        (1, 6),   # shoulder_right to hip_right
        (2, 4),   # shoulder_left to elbow_left
        (2, 5),   # shoulder_left to hip_left
        (3, 7),   # elbow_right to wrist_right
        (4, 8),   # elbow_left to wrist_left
        (5, 6),   # hip_left to hip_right
        (5, 10),  # hip_left to knee_left
        (6, 9),   # hip_right to knee_right
        (9, 11),  # knee_right to ankle_right
        (10, 12)  # knee_left to ankle_left
    ]
    
    print(f"Creating {num_visualizations} visualizations...")
    
    for i in range(num_visualizations):
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Sample data
        sample_data = samples_data[i]
        pred = predictions[i]
        gt = ground_truths[i]
        
        # Reshape predictions and ground truth to (num_joints, 2)
        pred_joints = pred.reshape(-1, 2)
        gt_joints = gt.reshape(-1, 2)

        
        
        # Plot 1: Ground truth pose
        axes[0].scatter(gt_joints[:, 0], gt_joints[:, 1], c='green', s=50, label='Ground Truth')
        # Draw skeleton connections for ground truth
        for connection in skeleton_connections:
            start_idx, end_idx = connection
            if start_idx < len(gt_joints) and end_idx < len(gt_joints):
                axes[0].plot([gt_joints[start_idx, 0], gt_joints[end_idx, 0]], 
                           [gt_joints[start_idx, 1], gt_joints[end_idx, 1]], 
                           'g-', alpha=0.5, linewidth=2)
        axes[0].set_title('Ground Truth Pose')
        axes[0].set_xlim(0, 640)
        axes[0].set_ylim(0, 480)
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Predicted pose
        axes[1].scatter(pred_joints[:, 0], pred_joints[:, 1], c='red', s=50, label='Prediction')
        # Draw skeleton connections for prediction
        for connection in skeleton_connections:
            start_idx, end_idx = connection
            if start_idx < len(pred_joints) and end_idx < len(pred_joints):
                axes[1].plot([pred_joints[start_idx, 0], pred_joints[end_idx, 0]], 
                           [pred_joints[start_idx, 1], pred_joints[end_idx, 1]], 
                           'r-', alpha=0.5, linewidth=2)
        axes[1].set_title('Predicted Pose')
        axes[1].set_xlim(0, 640)
        axes[1].set_ylim(0, 480)
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Plot 3: Overlay comparison
        axes[2].scatter(gt_joints[:, 0], gt_joints[:, 1], c='green', s=50, label='Ground Truth', alpha=0.7)
        axes[2].scatter(pred_joints[:, 0], pred_joints[:, 1], c='red', s=50, label='Prediction', alpha=0.7)
        # Draw skeleton connections for both
        for connection in skeleton_connections:
            start_idx, end_idx = connection
            if start_idx < len(gt_joints) and end_idx < len(gt_joints):
                axes[2].plot([gt_joints[start_idx, 0], gt_joints[end_idx, 0]], 
                           [gt_joints[start_idx, 1], gt_joints[end_idx, 1]], 
                           'g-', alpha=0.5, linewidth=2)
            if start_idx < len(pred_joints) and end_idx < len(pred_joints):
                axes[2].plot([pred_joints[start_idx, 0], pred_joints[end_idx, 0]], 
                           [pred_joints[start_idx, 1], pred_joints[end_idx, 1]], 
                           'r-', alpha=0.5, linewidth=2)
        
        # Draw error lines
        for j in range(len(gt_joints)):
            axes[2].plot([gt_joints[j, 0], pred_joints[j, 0]], 
                        [gt_joints[j, 1], pred_joints[j, 1]], 
                        'k--', alpha=0.3, linewidth=1)
        
        axes[2].set_title(f'Comparison (PCK: {results["pck_scores"][i]:.3f}, MPJPE: {results["mpjpe_scores"][i]:.1f})')
        axes[2].set_xlim(0, 640)
        axes[2].set_ylim(0, 480)
        axes[2].invert_yaxis()
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        
        # Save visualization with unique filename
        save_path = os.path.join(save_dir, f'prediction_visualization_{timestamp}_{i+1:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization {i+1}/{num_visualizations}: {save_path}")
    
    print(f"All visualizations saved with timestamp prefix: {timestamp}")
    return timestamp


def create_advanced_visualization(results, save_dir, sample_idx=0, timestamp=None):
    """
    Create advanced visualization using the GraphVisualization class.
    
    Args:
        results: Dictionary with predictions and ground truths
        save_dir: Directory to save visualizations
        sample_idx: Index of sample to visualize
        timestamp: Optional timestamp string for unique filenames
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get sample data
    sample_data = results['samples_data'][sample_idx]
    pred = results['predictions'][sample_idx]
    
    # Create a mock data object for GraphVisualization
    class MockData:
        def __init__(self, sample_data, pred):
            self.x = sample_data['x']
            self.pos = sample_data['pos']
            self.edge_index = sample_data['edge_index'] 
            self.edge_attr = sample_data['edge_attr']
            self.y = sample_data['y']
            self.th_pck = sample_data['th_pck']
            self.batch = sample_data['batch']
    
    mock_data = MockData(sample_data, pred)
    pred_tensor = torch.tensor(pred.reshape(1, -1), dtype=torch.float32)
    
    # Create GraphVisualization instance
    G = GraphVisualization(mock_data, pred=pred_tensor, res=[640, 480])
    
    # Create visualization
    image = G.create_image(show_image=False, show_gt=True, show_pred=True, joints=13)
    
    # Save image with unique filename
    save_path = os.path.join(save_dir, f'advanced_visualization_{timestamp}_sample_{sample_idx}.png')
    cv2.imwrite(save_path, image)
    
    print(f"Saved advanced visualization: {save_path}")
    return timestamp


def save_results_summary(results, save_path, timestamp=None, dataset_type=None):
    """
    Save a summary of the results to a JSON file.
    
    Args:
        results: Dictionary with predictions and metrics
        save_path: Path to save the summary file
        timestamp: Optional timestamp string for unique filenames
        dataset_type: Optional dataset type for metadata
    """
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create unique filename
    save_dir = os.path.dirname(save_path)
    filename = f'results_summary_{timestamp}.json'
    unique_save_path = os.path.join(save_dir, filename)
    
    summary = {
        'timestamp': timestamp,
        'dataset_type': dataset_type,
        'num_samples': len(results['predictions']),
        'mean_pck': float(np.mean(results['pck_scores'])),
        'std_pck': float(np.std(results['pck_scores'])),
        'mean_mpjpe': float(np.mean(results['mpjpe_scores'])),
        'std_mpjpe': float(np.std(results['mpjpe_scores'])),
        'min_pck': float(np.min(results['pck_scores'])),
        'max_pck': float(np.max(results['pck_scores'])),
        'min_mpjpe': float(np.min(results['mpjpe_scores'])),
        'max_mpjpe': float(np.max(results['mpjpe_scores']))
    }
    
    with open(unique_save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results summary saved to: {unique_save_path}")
    return summary, timestamp


def detect_dataset_type(ckpt_path):
    """
    Detect which dataset type to use based on the checkpoint path.
    
    Args:
        ckpt_path: Path to the checkpoint file
    
    Returns:
        tuple: (dataset_type, data_path)
            dataset_type: 'ledge_dataset' or 'scarf_dataset'
            data_path: corresponding data path
    """
    if 'ledge_dataset' in ckpt_path:
        return 'ledge_dataset', '/home/dberretta-iit.local/data/LEDGE_eh36m_val'
    elif 'Improved_scarf_dataset' in ckpt_path:
        return 'scarf_dataset', '/home/dberretta-iit.local/data/new_scarfGNN_val'
    else:
        # Default to scarf_dataset if unclear
        print("Warning: Could not detect dataset type from checkpoint path. Defaulting to scarf_dataset.")
        return 'scarf_dataset', '/home/dberretta-iit.local/data/new_scarfGNN_val'


def main():
    parser = argparse.ArgumentParser(description="Extract predictions and visualize results")
    
    parser.add_argument('--ckpt_path', 
                        type=str, 
                        required=True,
                        help='Path to the model checkpoint'
                        )
    parser.add_argument('--data_path', 
                        type=str, 
                        default=None,
                        help='Path to the dataset (auto-detected if not provided)')   
    parser.add_argument('--arch', 
                        type=str, 
                        default='single_weight', 
                        choices=['single_weight', 'two_weights'],
                        help='Model architecture')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=256,
                        help='Batch size for inference')
    parser.add_argument('--num_samples', 
                        type=int, 
                        default=20,
                        help='Maximum number of samples to process')
    parser.add_argument('--num_visualizations', 
                        type=int, 
                        default=20,
                        help='Number of visualizations to create')
    parser.add_argument('--save_dir', 
                        type=str, 
                        default='./prediction_results',
                        help='Directory to save results')
    parser.add_argument('--data_fraction', 
                        type=float, 
                        default=0.8,
                        help='Fraction of dataset to use for testing')
    parser.add_argument('--device', 
                        type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for inference')
    
    args = parser.parse_args()

    # Auto-detect dataset type and data path
    dataset_type, auto_data_path = detect_dataset_type(args.ckpt_path)
    data_path = args.data_path if args.data_path else auto_data_path
    
    print(f"Detected dataset type: {dataset_type}")
    print(f"Using data path: {data_path}")
    
    # Validate checkpoint path
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*50)
    print("GraphEnet-v2 Prediction Extraction & Visualization")
    print("="*50)
    
    # Load model
    model = load_model_from_checkpoint(args.ckpt_path, args.arch)
    
    # Load dataset based on detected type
    print(f"Loading dataset from: {data_path}")
    
    if dataset_type == 'scarf_dataset':
        # Use scarfDataset_splineConv for Improved_scarf_dataset checkpoints
        val_dataset = scarfDataset_splineConv(
            data_path,
            transform=None,
            pre_transform=None, 
            pre_filter=None,
            rf_size=14, 
            alpha=1.0, 
            C=0.3,
            res=(640, 480)
        )
        dataset_label = 'scarfDataset_splineConv'
    
    elif dataset_type == 'ledge_dataset':
        # Use eh36m_spline_gamer for ledge_dataset checkpoints
        transforms_current = [my_transforms.check_x_size]
        if cfg['node_feature'] != None:
            transforms_current.append(maps.node_feature(cfg['node_feature']))
        if cfg['connectivity'] > 0:
            transforms_current.append(maps.connectivity_map(cfg['connectivity']))
        if cfg['task'] != 'all':
            transforms_current.append(maps.task_map(cfg['task']))

        transforms_current = my_transforms.chain_transforms(transforms_current)

        # dataset = eh36m_spline_gamer(
        #     data_path,
        #     transform=transforms_current, 
        #     pre_filter=hpe_filter, 
        #     schema=schema_spline
        # )
        # dataset_label = 'eh36m_spline_gamer'

        val_dataset = eh36m_spline_ledge(
            data_path, 
            transform=transforms_current, 
            pre_filter=hpe_filter, 
            schema=schema_spline
        )
        dataset_label = 'eh36m_spline_ledge'

    #put a seed for reproducibility
    val_dataset = val_dataset.shuffle()

    print(f'dataset length: {len(val_dataset)} samples')
    
    # Split dataset (we'll use validation split for testing)
    # train_dataset, val_dataset, _= new_dataset_split(
    #     dataset,
    #     style='dev', 
    #     fraction=args.data_fraction, 
    #     dataset_label=dataset_label
    # )
    
    # Create dataloader for test data
    test_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    print(f"Dataset loaded. Test samples: {len(val_dataset)}")
    
    # Extract predictions
    results = extract_predictions(
        model, 
        test_loader, 
        device=args.device,
        max_samples=args.num_samples
    )
    
    # Generate timestamp with dataset type for unique filenames
    timestamp_base = datetime.now().strftime("%Y%m%d_%H%M")
    dataset_prefix = "scarf" if dataset_type == 'scarf_dataset' else "ledge"
    timestamp = f"{dataset_prefix}_{timestamp_base}"
    
    print(f"Using timestamp with dataset prefix: {timestamp}")
    
    # Save results summary
    summary_path = os.path.join(args.save_dir, 'results_summary.json')
    summary, timestamp = save_results_summary(results, summary_path, timestamp, dataset_type)
    
    # Create visualizations
    vis_dir = os.path.join(args.save_dir, 'visualizations')
    visualize_predictions(results, vis_dir, args.num_visualizations, timestamp)
    
    # Create advanced visualization for first sample
    adv_vis_dir = os.path.join(args.save_dir, 'advanced_visualizations')
    try:
        create_advanced_visualization(results, adv_vis_dir, sample_idx=0, timestamp=timestamp)
    except Exception as e:
        print(f"Warning: Could not create advanced visualization: {e}")
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Samples processed: {summary['num_samples']}")
    print(f"Average PCK: {summary['mean_pck']:.4f} ± {summary['std_pck']:.4f}")
    print(f"Average MPJPE: {summary['mean_mpjpe']:.2f} ± {summary['std_mpjpe']:.2f} pixels")
    print(f"PCK range: [{summary['min_pck']:.4f}, {summary['max_pck']:.4f}]")
    print(f"MPJPE range: [{summary['min_mpjpe']:.2f}, {summary['max_mpjpe']:.2f}] pixels")
    print(f"\nResults saved to: {args.save_dir}")
    print("="*50)


if __name__ == '__main__':
    main()
