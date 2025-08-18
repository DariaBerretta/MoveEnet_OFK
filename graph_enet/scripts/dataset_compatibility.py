#!/usr/bin/env python3
"""
Dataset Compatibility Utilities for SCARF SplineConv
====================================================

This module provides utilities to ensure compatibility between the SCARF SplineConv
dataset and the existing HPE GNN models.

these utilities include:
- Validating dataset compatibility with specific model types
- Creating custom train/val/test splits
- Comparing dataset statistics between SCARF and custom datasets
- Printing detailed compatibility reports

Usage:
    - Import the functions from this module in your training scripts.
    - Use `validate_dataset_compatibility` to check if your dataset is ready for training.
    - Use `create_custom_dataset_split` to create splits for training, validation, and testing.
    - Use `compare_dataset_statistics` to compare SCARF dataset with custom datasets.
    - Use `print_compatibility_report` to print a detailed compatibility report for the dataset.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from graph_enet.hpe_gnn.utils.dataset_utils import dataset_split
import graph_enet.hpe_gnn.data.h36m_utils as h36m


def validate_dataset_compatibility(dataset, model_type='splineconv'):
    """
    Validate that the dataset is compatible with the specified model type.
    
    Args:
        dataset: The dataset to validate
        model_type: Type of model ('splineconv', 'gcn', etc.)
    
    Returns:
        bool: True if compatible, False otherwise
        dict: Compatibility report
    """
    
    # Initialize compatibility report
    report = {
        'compatible': True,
        'issues': [],
        'dataset_info': {},
        'recommendations': []
    }
    
    if len(dataset) == 0:
        report['compatible'] = False
        report['issues'].append("Dataset is empty")
        return report['compatible'], report
    
    # Get sample data
    sample = dataset[0]
    
    # Check basic structure
    required_attrs = ['x', 'edge_index', 'y']
    for attr in required_attrs:
        if not hasattr(sample, attr):
            report['compatible'] = False
            report['issues'].append(f"Missing required attribute: {attr}")
    
    if not report['compatible']:
        return report['compatible'], report
    
    # Record dataset information
    report['dataset_info'] = {
        'num_samples': len(dataset),
        'num_node_features': sample.x.shape[1] if hasattr(sample, 'x') else None,
        'num_nodes_sample': sample.x.shape[0] if hasattr(sample, 'x') else None,
        'num_edges_sample': sample.edge_index.shape[1] if hasattr(sample, 'edge_index') else None,
        'target_shape': sample.y.shape if hasattr(sample, 'y') else None,
        'has_edge_attr': hasattr(sample, 'edge_attr'),
        'has_pos': hasattr(sample, 'pos'),
        'has_th_pck': hasattr(sample, 'th_pck')
    }
    
    # SplineConv specific checks
    if model_type == 'splineconv':
        # Check for edge attributes (required for SplineConv)
        if not hasattr(sample, 'edge_attr'):
            report['compatible'] = False
            report['issues'].append("SplineConv requires edge attributes (edge_attr)")
            report['recommendations'].append("Use Cartesian transform to add edge attributes")
        elif sample.edge_attr.shape[1] != 2:
            report['compatible'] = False
            report['issues'].append(f"SplineConv expects 2D edge attributes, got {sample.edge_attr.shape[1]}D")
        
        # Check node features (your dataset should have 10 features)
        expected_features = 10
        if sample.x.shape[1] != expected_features:
            report['issues'].append(f"Expected {expected_features} node features, got {sample.x.shape[1]}")
            report['recommendations'].append("Verify SCARF feature extraction produces 10D features")
        
        # Check target format
        if len(sample.y.shape) != 2:
            report['issues'].append(f"Expected 2D target tensor, got shape {sample.y.shape}")
        
        # Check for th_pck (required for some evaluations)
        if not hasattr(sample, 'th_pck'):
            report['recommendations'].append("Consider adding th_pck attribute for PCK evaluation")
    
    # Data type checks
    if sample.x.dtype != torch.float32:
        report['recommendations'].append(f"Node features are {sample.x.dtype}, consider using float32")
    
    if sample.y.dtype != torch.float32:
        report['recommendations'].append(f"Targets are {sample.y.dtype}, consider using float32")
    
    return report['compatible'], report


def create_custom_dataset_split(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Create custom train/val/test split for SCARF dataset.
    
    Args:
        dataset: The dataset to split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation  
        test_ratio: Fraction for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    
    # Shuffle dataset
    dataset = dataset.shuffle()
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_dataset)} samples ({train_ratio*100:.1f}%)")
    print(f"  Val: {len(val_dataset)} samples ({val_ratio*100:.1f}%)")
    print(f"  Test: {len(test_dataset)} samples ({test_ratio*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset


def compare_dataset_statistics(scarf_dataset, custom_dataset):
    """
    Compare statistics between SCARF and custom datasets.
    
    Args:
        scarf_dataset: Your SCARF dataset
        custom_dataset: Existing custom dataset
    
    Returns:
        dict: Comparison statistics
    """
    
    stats = {
        'scarf': {},
        'custom': {},
        'comparison': {}
    }
    
    # SCARF dataset stats
    if len(scarf_dataset) > 0:
        scarf_sample = scarf_dataset[0]
        stats['scarf'] = {
            'num_samples': len(scarf_dataset),
            'num_node_features': scarf_sample.x.shape[1],
            'avg_nodes_per_graph': np.mean([data.x.shape[0] for data in scarf_dataset[:min(100, len(scarf_dataset))]]),
            'avg_edges_per_graph': np.mean([data.edge_index.shape[1] for data in scarf_dataset[:min(100, len(scarf_dataset))]]),
            'target_shape': scarf_sample.y.shape,
            'has_edge_attr': hasattr(scarf_sample, 'edge_attr')
        }
    
    # Custom dataset stats
    if len(custom_dataset) > 0:
        custom_sample = custom_dataset[0]
        stats['custom'] = {
            'num_samples': len(custom_dataset),
            'num_node_features': custom_dataset.num_features,
            'avg_nodes_per_graph': np.mean([data.x.shape[0] for data in custom_dataset[:min(100, len(custom_dataset))]]),
            'avg_edges_per_graph': np.mean([data.edge_index.shape[1] for data in custom_dataset[:min(100, len(custom_dataset))]]),
            'target_shape': custom_sample.y.shape,
            'has_edge_attr': hasattr(custom_sample, 'edge_attr')
        }
    
    # Comparison
    if stats['scarf'] and stats['custom']:
        stats['comparison'] = {
            'feature_dim_match': stats['scarf']['num_node_features'] == stats['custom']['num_node_features'],
            'target_shape_match': stats['scarf']['target_shape'] == stats['custom']['target_shape'],
            'edge_attr_match': stats['scarf']['has_edge_attr'] == stats['custom']['has_edge_attr'],
            'avg_graph_size_ratio': stats['scarf']['avg_nodes_per_graph'] / stats['custom']['avg_nodes_per_graph'],
            'dataset_size_ratio': stats['scarf']['num_samples'] / stats['custom']['num_samples']
        }
    
    return stats


def print_compatibility_report(dataset, model_type='splineconv'):
    """Print a detailed compatibility report for the dataset."""
    
    compatible, report = validate_dataset_compatibility(dataset, model_type)
    
    print(f"\nDataset Compatibility Report ({model_type.upper()})")
    print("=" * 50)
    print(f"Compatible: {'✓' if compatible else '✗'}")
    print()
    
    print("Dataset Information:")
    for key, value in report['dataset_info'].items():
        print(f"  {key}: {value}")
    print()
    
    if report['issues']:
        print("Issues Found:")
        for issue in report['issues']:
            print(f"  ✗ {issue}")
        print()
    
    if report['recommendations']:
        print("Recommendations:")
        for rec in report['recommendations']:
            print(f"  → {rec}")
        print()
    
    if compatible:
        print("✓ Dataset is ready for training!")
    else:
        print("✗ Please fix the issues above before training.")


def main():
    """Test compatibility utilities."""
    
    # Example usage
    from graph_enet.data.scarfDataset_splineConv import scarfDataset_splineConv
    
    # Load your dataset
    data_path = "/home/dberretta-iit.local/data/new_scarfGNN"  # Update this path
    
    try:
        dataset = scarfDataset_splineConv(root=data_path)
        print_compatibility_report(dataset, 'splineconv')
        
        # Create splits
        train_dataset, val_dataset, test_dataset = create_custom_dataset_split(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42)

    except Exception as e:
        print(f"Could not load dataset: {e}")
        print("Update the data_path variable in this script to test.")


if __name__ == "__main__":
    main()
