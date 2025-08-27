#!/usr/bin/env python3
"""
Quick test script to validate the complete_pipeline.py functionality
This script performs a minimal test to ensure all components work together.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required imports work."""
    try:
        from graph_enet.data.scarfDataset_splineConv import scarfDataset_splineConv
        from graph_enet.hpe_gnn.model.hpegnn import hpeGnn_splineConv_single_weight
        from graph_enet.pyScarf.scarf.scarf_class import SCARF
        from graph_enet.utils.log_loader import load_events_from_log, load_skeleton_from_log
        from graph_enet.data.graph_builder_splineConv import build_scarf_graph_splineConv
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    try:
        from graph_enet.hpe_gnn.model.hpegnn import hpeGnn_splineConv_single_weight
        
        model = hpeGnn_splineConv_single_weight(
            in_channels=10,
            hidden_channels=[32, 64, 32],
            out_channels=13,  # Number of joints
            learning_rate=0.01,
            batch_size=1,
            data_fraction=0.1,
            label='test',
            task='all',
            transforms=None,
            node_loss_weight=[1.0, 0.1],
            pck_multiplier=0.6
        )
        
        # Test forward pass with dummy data
        x = torch.randn(10, 10)  # 10 nodes, 10 features
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_attr = torch.randn(3, 2)  # 3 edges, 2 attributes each
        pos = torch.randn(10, 2)  # 10 nodes, 2D positions
        batch = torch.zeros(10, dtype=torch.long)
        
        with torch.no_grad():
            output, _ = model.forward(x, edge_index, edge_attr, pos, batch)
        
        assert output.shape == (1, 26), f"Expected output shape (1, 26), got {output.shape}"
        print("✓ Model creation and forward pass successful")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_scarf():
    """Test SCARF functionality."""
    try:
        from graph_enet.pyScarf.scarf.scarf_class import SCARF
        
        scarf = SCARF((640, 480), 14, 1.0, 0.3)
        
        # Test some updates
        scarf.update(100, 100, 1)
        scarf.update(200, 200, 0)
        scarf.update(300, 300, 1)
        
        # Get surface
        surface = scarf.get_surface()
        assert surface.shape == (480, 640), f"Expected surface shape (480, 640), got {surface.shape}"
        
        print("✓ SCARF functionality test successful")
        return True
    except Exception as e:
        print(f"❌ SCARF test failed: {e}")
        return False

def test_data_paths():
    """Test if data paths exist."""
    default_train_path = "/home/dberretta-iit.local/data/new_scarfGNN"
    default_video_path = "/home/dberretta-iit.local/data/cam2_S1_Directions"
    
    train_exists = os.path.exists(default_train_path)
    video_exists = os.path.exists(default_video_path)
    
    print(f"Training data path exists: {train_exists} ({default_train_path})")
    print(f"Video data path exists: {video_exists} ({default_video_path})")
    
    if not train_exists:
        print("⚠️  Training data not found at default location")
    if not video_exists:
        print("⚠️  Video data not found at default location")
    
    return True

def main():
    print("="*60)
    print("COMPLETE PIPELINE - COMPONENT TESTS")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation), 
        ("SCARF Functionality", test_scarf),
        ("Data Paths", test_data_paths)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n✅ All tests passed! complete_pipeline.py should work correctly.")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please check the issues above.")
    
    print("\nTo run the complete pipeline:")
    print("python complete_pipeline.py --data_path /your/training/data --video_data_path /your/test/data")

if __name__ == '__main__':
    main()
