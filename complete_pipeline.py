#!/usr/bin/env python3
"""
Complete GraphEnet-v2 Training, Validation, and Testing Pipeline with Video Vi    # Model configuration based on architecture
    if cfg['arch'] == 'single_weight':
        model = hpeGnn_splineConv_single_weight(
            in_channels=10,  # SCARF node features
            hidden_channels=cfg['hidden'],
            out_channels=13,  # Number of joints (not joints*2)
            learning_rate=cfg['learning_rate'],
            batch_size=cfg['batch_size'],
            pck_multiplier=0.6
        )
    else:  # two_weights
        model = hpeGnn_splineConv(
            in_channels=10,
            hidden_channels=cfg['hidden'],
            out_channels=13,  # Number of joints (not joints*2)
            learning_rate=cfg['learning_rate'],
            batch_size=cfg['batch_size'],
            pck_multiplier=0.6
        )===============================================================================

This script provides a unified workflow that:
1. Creates and trains a GraphEnet-v2 model
2. Validates the model performance
3. Tests on new data with real-time visualization
4. Creates video output showing ground truth vs predicted poses

Usage:
    # Full pipeline (train + validate + test with video)
    python complete_pipeline.py --data_path /path/to/data --video_data_path /path/to/test/data
    
    # Quick test (skip training, use existing checkpoint)
    python complete_pipeline.py --skip_training --ckpt_path /path/to/checkpoint.ckpt --video_data_path /path/to/test/data
    
    # Training only
    python complete_pipeline.py --data_path /path/to/data --training_only

Example:
    python complete_pipeline.py --data_path /home/dberretta-iit.local/data/new_scarfGNN --video_data_path /home/dberretta-iit.local/data/cam2_S1_Directions
"""

import os
import json
import time
import argparse
import numpy as np
import cv2
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Cartesian
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

# Import fixed metrics first
from graph_enet.test_scripts.fixed_metrics import pck_error, mpjpe_error

# Monkey patch the metrics module before importing the model
import graph_enet.hpe_gnn.utils.metrics
graph_enet.hpe_gnn.utils.metrics.pck_error = pck_error
graph_enet.hpe_gnn.utils.metrics.mpjpe_error = mpjpe_error

# GraphEnet-v2 imports
from graph_enet.data.scarfDataset_splineConv import scarfDataset_splineConv
from graph_enet.hpe_gnn.model.hpegnn import hpeGnn_splineConv, hpeGnn_splineConv_single_weight
from graph_enet.hpe_gnn.utils.dataset_utils import new_dataset_split
from graph_enet.hpe_gnn.utils.library_utils import MyProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# SCARF and visualization imports
from graph_enet.pyScarf.scarf.scarf_class import SCARF
from graph_enet.utils.log_loader import load_events_from_log, load_skeleton_from_log
from graph_enet.data.graph_builder_splineConv import build_scarf_graph_splineConv
from graph_enet.pyScarf.utils.slt_ppr_filter import SpatialFilter


# =====================================================
# CONFIGURATION AND SETUP
# =====================================================

def create_config():
    """Create default training configuration following project guidelines."""
    return {
        'arch': 'single_weight',  # More stable architecture
        'epochs': 30,
        'batch_size': 64,
        'learning_rate': 0.01,
        'data_fraction': 0.8,
        'dataset_split': 'dev',
        'hidden': [32, 64, 128, 64, 32],
        'node_loss_weight': 0.1,
        'target_loss_weight': 1.0,
        'patience': 10,
        'scarf_params': {
            'rf_size': 14,
            'alpha': 1.0,
            'C': 0.3,
            'res': (640, 480),
            'dt': 0.01
        },
        'video_params': {
            'fps': 30,
            'duration_sec': 10.0,
            'show_graph': True,
            'show_skeleton': True
        }
    }


# =====================================================
# TRAINING PIPELINE
# =====================================================

def setup_model_and_trainer(cfg, dataset, experiment_name="complete_pipeline"):
    """Setup model and PyTorch Lightning trainer."""
    
    num_joints = 13
    
    # Model configuration based on architecture
    if cfg['arch'] == 'single_weight':
        model = hpeGnn_splineConv_single_weight(
            dataset.num_features,
            cfg['hidden'],
            num_joints,
            learning_rate=cfg['learning_rate'],
            batch_size=cfg['batch_size'], 
            data_fraction=cfg['data_fraction'], 
            label=experiment_name,
            task='all', 
            transforms=None, 
            node_loss_weight=[cfg['target_loss_weight'], cfg['node_loss_weight']],
            pck_multiplier=0.6
        )
    else:  # two_weights
        model = hpeGnn_splineConv(
            dataset.num_features,
            cfg['hidden'],
            num_joints,
            learning_rate=cfg['learning_rate'],
            batch_size=cfg['batch_size'], 
            data_fraction=cfg['data_fraction'], 
            label=experiment_name,
            task='all', 
            transforms=None, 
            node_loss_weight=[cfg['target_loss_weight'], cfg['node_loss_weight']],
            pck_multiplier=0.6
        )
    
    # Setup callbacks
    early_stop_callback = EarlyStopping(
        monitor='pck/val',
        patience=cfg['patience'],
        verbose=True,
        mode='max'  # PCK should be maximized
    )
    
    progress_bar = MyProgressBar()
    
    # Setup logger
    logger = TensorBoardLogger(
        "lightning_logs", 
        name=experiment_name,
        version=None
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        callbacks=[early_stop_callback, progress_bar],
        logger=logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None
    )
    
    return model, trainer


def train_model(cfg, data_path, experiment_name="complete_pipeline"):
    """Train the GraphEnet-v2 model."""
    
    print("=" * 60)
    print("STEP 1: TRAINING GRAPHENET-V2 MODEL")
    print("=" * 60)
    
    # Load dataset
    print(f"Loading dataset from: {data_path}")
    dataset = scarfDataset_splineConv(
        data_path,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        rf_size=cfg['scarf_params']['rf_size'],
        alpha=cfg['scarf_params']['alpha'],
        C=cfg['scarf_params']['C'],
        res=cfg['scarf_params']['res']
    )
    
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    
    # Split dataset
    train_dataset, val_dataset = new_dataset_split(
        dataset,
        style=cfg['dataset_split'],
        fraction=cfg['data_fraction'],
        dataset_label='scarfDataset_splineConv'
    )
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )
    
    # Setup model and trainer
    model, trainer = setup_model_and_trainer(cfg, dataset, experiment_name)
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    training_time = time.time() - start_time
    
    # Get best checkpoint path
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    
    print(f"Training completed in {training_time/60:.1f} minutes")
    print(f"Best checkpoint: {best_ckpt_path}")
    
    return model, best_ckpt_path, trainer


# =====================================================
# VALIDATION PIPELINE
# =====================================================

def validate_model(model, trainer, data_path, cfg):
    """Validate the trained model."""
    
    print("\n" + "=" * 60)
    print("STEP 2: MODEL VALIDATION")
    print("=" * 60)
    
    # Load validation dataset (same as training data)
    dataset = scarfDataset_splineConv(
        data_path,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        rf_size=cfg['scarf_params']['rf_size'],
        alpha=cfg['scarf_params']['alpha'],
        C=cfg['scarf_params']['C'],
        res=cfg['scarf_params']['res']
    )
    
    # Use a separate validation split
    _, val_dataset = new_dataset_split(
        dataset,
        style=cfg['dataset_split'],
        fraction=cfg['data_fraction'],
        dataset_label='scarfDataset_splineConv'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Run validation
    print("Running validation...")
    val_results = trainer.validate(model, val_loader)
    
    # Extract metrics
    val_metrics = val_results[0] if val_results else {}
    
    print("Validation Results:")
    for key, value in val_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return val_metrics


# =====================================================
# VIDEO TESTING PIPELINE
# =====================================================

def load_test_data(video_data_path, cfg):
    """Load test data for video generation."""
    
    # Load events
    events_path = os.path.join(video_data_path, "ch0dvs")
    skeleton_path = os.path.join(video_data_path, "ch0GT50Hzskeleton") 
    
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Events data not found: {events_path}")
    if not os.path.exists(skeleton_path):
        raise FileNotFoundError(f"Skeleton data not found: {skeleton_path}")
    
    print(f"Loading events from: {events_path}")
    events = load_events_from_log(events_path)
    
    print(f"Loading skeleton data from: {skeleton_path}")
    skeleton_data = load_skeleton_from_log(skeleton_path)
    
    print(f"Loaded {len(events)} events, duration: {events['ts'][-1]:.2f}s")
    print(f"Loaded {len(skeleton_data)} skeleton frames")
    
    return events, skeleton_data


def create_pose_video(model, video_data_path, cfg, output_path="pose_prediction_video.mp4"):
    """Create video showing ground truth vs predicted poses."""
    
    print("\n" + "=" * 60)  
    print("STEP 3: CREATING POSE PREDICTION VIDEO")
    print("=" * 60)
    
    # Load test data
    events, skeleton_data = load_test_data(video_data_path, cfg)
    
    # Initialize SCARF
    scarf_params = cfg['scarf_params']
    scarf = SCARF(
        scarf_params['res'], 
        scarf_params['rf_size'], 
        scarf_params['alpha'], 
        scarf_params['C']
    )
    
    # Initialize noise filter
    filter = SpatialFilter()
    filter.initialise(
        scarf_params['res'][1], 
        scarf_params['res'][0], 
        period=0.1, 
        spatial_range=1
    )
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_path, 
        fourcc, 
        cfg['video_params']['fps'], 
        scarf_params['res']
    )
    
    # Skeleton connections for visualization
    skeleton_connections = [
        (1, 2), (1, 3), (1, 6), (2, 4), (2, 5), (3, 7), (4, 8), 
        (5, 6), (5, 10), (6, 9), (9, 11), (10, 12)
    ]
    
    # Video generation loop
    model.eval()
    timer = 0.0
    event_idx = 0
    skeleton_idx = 0
    frame_count = 0
    max_duration = cfg['video_params']['duration_sec']
    
    print(f"Generating {max_duration}s video at {cfg['video_params']['fps']} fps...")
    
    with torch.no_grad():
        while timer < min(events['ts'][-1], max_duration):
            # Process events in current time window
            start_idx = event_idx
            while event_idx < len(events) and events['ts'][event_idx] <= timer:
                ev = events[event_idx]
                if filter.check(ev['x'], ev['y'], ev['pol'], ev['ts']):
                    scarf.update(ev['x'], ev['y'], ev['pol'])
                event_idx += 1
            
            # Get current ground truth skeleton
            current_gt = None
            if skeleton_idx < len(skeleton_data):
                if skeleton_data['ts'][skeleton_idx] <= timer:
                    current_gt = skeleton_data['data'][skeleton_idx]
                    if skeleton_idx < len(skeleton_data) - 1:
                        skeleton_idx += 1
                else:
                    # Use previous skeleton if available
                    if skeleton_idx > 0:
                        current_gt = skeleton_data['data'][skeleton_idx - 1]
            
            # Create base visualization
            img32 = scarf.get_surface()
            img8U = (img32 * 255).clip(0, 255).astype('uint8')
            inverted = cv2.bitwise_not(img8U)
            colored = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
            
            prediction = None
            if current_gt is not None:
                # Build graph from current SCARF state
                try:
                    graph_data = build_scarf_graph_splineConv(
                        scarf, 
                        current_gt.flatten(),
                        k_neighbour=4,
                        active_ratio=0.15,
                        radius=25
                    )
                    
                    if graph_data is not None:
                        # Move to correct device
                        device = next(model.parameters()).device
                        graph_data = graph_data.to(device)
                        
                        # Get model prediction
                        pred, _ = model.forward(
                            graph_data.x.unsqueeze(0) if graph_data.x.dim() == 1 else graph_data.x,
                            graph_data.edge_index,
                            graph_data.edge_attr,
                            graph_data.pos,
                            torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
                        )
                        
                        prediction = pred.cpu().numpy().reshape(-1, 2)
                        
                        # Draw graph if enabled
                        if cfg['video_params']['show_graph']:
                            G = to_networkx(graph_data.cpu(), to_undirected=True, remove_self_loops=True)
                            pos_dict = {
                                i: (int(graph_data.pos[i][0].item()), int(graph_data.pos[i][1].item()))
                                for i in range(graph_data.num_nodes)
                            }
                            
                            # Draw edges
                            for u, v in G.edges():
                                if u in pos_dict and v in pos_dict:
                                    p1 = pos_dict[u]
                                    p2 = pos_dict[v]
                                    cv2.line(colored, p1, p2, (100, 100, 255), 1)
                            
                            # Draw nodes
                            for node, (x, y) in pos_dict.items():
                                cv2.circle(colored, (x, y), 3, (255, 255, 100), -1)
                    
                except Exception as e:
                    print(f"Warning: Could not process frame at t={timer:.3f}: {e}")
            
            # Draw poses if enabled
            if cfg['video_params']['show_skeleton'] and current_gt is not None:
                gt_joints = current_gt.astype(int)
                
                # Draw ground truth skeleton (green)
                for connection in skeleton_connections:
                    start_idx, end_idx = connection
                    if start_idx < len(gt_joints) and end_idx < len(gt_joints):
                        start_point = tuple(gt_joints[start_idx])
                        end_point = tuple(gt_joints[end_idx])
                        cv2.line(colored, start_point, end_point, (0, 255, 0), 2)
                
                for joint in gt_joints:
                    cv2.circle(colored, tuple(joint), 5, (0, 255, 0), -1)
                
                # Draw predicted skeleton (red)  
                if prediction is not None:
                    pred_joints = prediction.astype(int)
                    for connection in skeleton_connections:
                        start_idx, end_idx = connection
                        if start_idx < len(pred_joints) and end_idx < len(pred_joints):
                            start_point = tuple(pred_joints[start_idx])
                            end_point = tuple(pred_joints[end_idx])
                            cv2.line(colored, start_point, end_point, (0, 0, 255), 2)
                    
                    for joint in pred_joints:
                        cv2.circle(colored, tuple(joint), 5, (0, 0, 255), -1)
                
                # Add legend
                cv2.putText(colored, "GT (Green) / Pred (Red)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(colored, f"t={timer:.2f}s", (10, 460), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to video
            video_writer.write(colored)
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (timer / max_duration) * 100
                print(f"Progress: {progress:.1f}% (t={timer:.2f}s)")
            
            timer += 1.0 / cfg['video_params']['fps']
            frame_count += 1
    
    # Cleanup
    video_writer.release()
    print(f"Video saved to: {output_path}")
    print(f"Total frames generated: {frame_count}")


# =====================================================
# EVALUATION AND METRICS
# =====================================================

def evaluate_model_performance(model, ckpt_path, data_path, cfg):
    """Evaluate model performance on test data."""
    
    print("\n" + "=" * 60)
    print("STEP 4: MODEL PERFORMANCE EVALUATION") 
    print("=" * 60)
    
    # Load model from checkpoint
    if cfg['arch'] == 'single_weight':
        model = hpeGnn_splineConv_single_weight.load_from_checkpoint(ckpt_path)
    else:
        model = hpeGnn_splineConv.load_from_checkpoint(ckpt_path)
    
    model.eval()
    
    # Load test dataset
    dataset = scarfDataset_splineConv(
        data_path,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        rf_size=cfg['scarf_params']['rf_size'],
        alpha=cfg['scarf_params']['alpha'],
        C=cfg['scarf_params']['C'],
        res=cfg['scarf_params']['res']
    )
    
    # Use a separate test split
    _, test_dataset = new_dataset_split(
        dataset,
        style='test',
        fraction=0.2,
        dataset_label='scarfDataset_splineConv'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    # Evaluation
    pck_scores = []
    mpjpe_scores = []
    
    print(f"Evaluating on {len(test_dataset)} test samples...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            
            # Forward pass
            pred, _ = model.forward(
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
            
            pck_scores.append(pck.cpu().item())
            mpjpe_scores.append(mpjpe.cpu().item())
            
            if batch_idx % 20 == 0:
                print(f"Evaluated {batch_idx + 1}/{len(test_loader)} samples...")
    
    # Calculate statistics
    mean_pck = np.mean(pck_scores)
    std_pck = np.std(pck_scores)
    mean_mpjpe = np.mean(mpjpe_scores)
    std_mpjpe = np.std(mpjpe_scores)
    
    print("\nEvaluation Results:")
    print(f"  PCK: {mean_pck:.4f} ± {std_pck:.4f}")
    print(f"  MPJPE: {mean_mpjpe:.2f} ± {std_mpjpe:.2f} pixels")
    
    return {
        'pck_mean': mean_pck,
        'pck_std': std_pck,
        'mpjpe_mean': mean_mpjpe,
        'mpjpe_std': std_mpjpe,
        'num_samples': len(pck_scores)
    }


# =====================================================
# MAIN PIPELINE
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete GraphEnet-v2 Pipeline: Train, Validate, and Test with Video"
    )
    
    # Data paths
    parser.add_argument('--data_path', type=str,
                        default='/home/dberretta-iit.local/data/new_scarfGNN',
                        help='Path to training dataset')
    parser.add_argument('--video_data_path', type=str,
                        default='/home/dberretta-iit.local/data/cam2_S1_Directions',
                        help='Path to test data for video generation')
    
    # Pipeline control
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and use existing checkpoint')
    parser.add_argument('--training_only', action='store_true',
                        help='Only perform training, skip testing')
    parser.add_argument('--ckpt_path', type=str,
                        help='Path to existing checkpoint (if skip_training)')
    
    # Model configuration
    parser.add_argument('--arch', type=str, default='single_weight',
                        choices=['single_weight', 'two_weights'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    
    # Video configuration
    parser.add_argument('--video_duration', type=float, default=10.0,
                        help='Duration of output video (seconds)')
    parser.add_argument('--video_fps', type=int, default=30,
                        help='Video frame rate')
    parser.add_argument('--output_video', type=str, default='pose_prediction_video.mp4',
                        help='Output video filename')
    
    # Other options
    parser.add_argument('--experiment_name', type=str, default='complete_pipeline',
                        help='Experiment name for logging')
    
    args = parser.parse_args()
    
    # Create configuration
    cfg = create_config()
    
    # Override config with command line arguments
    cfg['arch'] = args.arch
    cfg['epochs'] = args.epochs
    cfg['batch_size'] = args.batch_size
    cfg['learning_rate'] = args.learning_rate
    cfg['video_params']['duration_sec'] = args.video_duration
    cfg['video_params']['fps'] = args.video_fps
    
    print("=" * 80)
    print("GRAPHENET-V2 COMPLETE PIPELINE")
    print("=" * 80)
    print(f"Training data: {args.data_path}")
    print(f"Video data: {args.video_data_path}")
    print(f"Architecture: {cfg['arch']}")
    print(f"Skip training: {args.skip_training}")
    print(f"Training only: {args.training_only}")
    print("=" * 80)
    
    # Validate paths
    if not args.skip_training and not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Training data path not found: {args.data_path}")
    
    if not args.training_only and not os.path.exists(args.video_data_path):
        raise FileNotFoundError(f"Video data path not found: {args.video_data_path}")
    
    if args.skip_training and not args.ckpt_path:
        raise ValueError("Must provide --ckpt_path when using --skip_training")
    
    # Pipeline execution
    model = None
    ckpt_path = args.ckpt_path
    trainer = None
    
    try:
        # Step 1: Training (if not skipped)
        if not args.skip_training:
            model, ckpt_path, trainer = train_model(cfg, args.data_path, args.experiment_name)
        
        # Step 2: Validation (if model was trained)
        if model is not None and trainer is not None:
            val_metrics = validate_model(model, trainer, args.data_path, cfg)
        
        # Step 3: Performance evaluation
        if not args.training_only:
            eval_results = evaluate_model_performance(model, ckpt_path, args.data_path, cfg)
        
        # Step 4: Video generation (if not training only)
        if not args.training_only:
            # Load model for video generation if needed
            if model is None:
                if cfg['arch'] == 'single_weight':
                    model = hpeGnn_splineConv_single_weight.load_from_checkpoint(ckpt_path)
                else:
                    model = hpeGnn_splineConv.load_from_checkpoint(ckpt_path)
            
            create_pose_video(model, args.video_data_path, cfg, args.output_video)
        
        # Final summary
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        if not args.skip_training:
            print(f"✓ Model trained and saved to: {ckpt_path}")
        if not args.training_only:
            print(f"✓ Video generated: {args.output_video}")
            print(f"✓ Evaluation completed")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
