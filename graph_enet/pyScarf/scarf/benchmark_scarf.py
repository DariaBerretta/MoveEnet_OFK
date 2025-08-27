#!/usr/bin/env python3
"""
Performance benchmark comparing original SCARF vs optimized versions.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from graph_enet.pyScarf.scarf.scarf_class import SCARF as SCARF_Original
from graph_enet.pyScarf.scarf.scarf_class_optimized import SCARF_Optimized, SCARF_Ultra

def generate_test_events(n_events=50000, res=(640, 480)):
    """Generate synthetic event data for benchmarking."""
    events = np.zeros(n_events, dtype=[('x', np.int32), ('y', np.int32), ('pol', np.int32)])
    events['x'] = np.random.randint(0, res[0], n_events)
    events['y'] = np.random.randint(0, res[1], n_events)
    events['pol'] = np.random.randint(0, 2, n_events)
    return events

def benchmark_scarf_performance():
    """Benchmark different SCARF implementations."""
    
    # Test parameters (typical values from project)
    res = (640, 480)
    rf_size = 14
    alpha = 1.0
    C = 0.3
    n_events = 50000
    
    print(f"Benchmarking SCARF Performance")
    print(f"Resolution: {res}, RF Size: {rf_size}, Events: {n_events}")
    print("-" * 60)
    
    # Generate test data
    events = generate_test_events(n_events, res)
    
    # Test Original SCARF
    print("Testing Original SCARF...")
    scarf_orig = SCARF_Original(res, rf_size, alpha, C)
    
    start_time = time.time()
    for event in events:
        scarf_orig.update(event['x'], event['y'], event['pol'])
    orig_time = time.time() - start_time
    
    # Get active RFs for comparison
    start_time = time.time()
    orig_active_rfs = scarf_orig.get_active_RF()
    orig_active_time = time.time() - start_time
    
    print(f"Original SCARF - Update Time: {orig_time:.3f}s, Active RF Time: {orig_active_time:.3f}s")
    print(f"Original SCARF - Active RFs: {len(orig_active_rfs)}")
    
    # Test Optimized SCARF
    print("Testing Optimized SCARF...")
    scarf_opt = SCARF_Optimized(res, rf_size, alpha, C)
    
    start_time = time.time()
    for event in events:
        scarf_opt.update(event['x'], event['y'], event['pol'])
    opt_time = time.time() - start_time
    
    # Get active RFs for comparison  
    start_time = time.time()
    opt_active_rfs = scarf_opt.get_active_RF()
    opt_active_time = time.time() - start_time
    
    print(f"Optimized SCARF - Update Time: {opt_time:.3f}s, Active RF Time: {opt_active_time:.3f}s")
    print(f"Optimized SCARF - Active RFs: {len(opt_active_rfs)}")
    
    # Test batch processing
    print("Testing Batch Processing...")
    scarf_batch = SCARF_Optimized(res, rf_size, alpha, C)
    
    start_time = time.time()
    scarf_batch.update_batch(events)
    batch_time = time.time() - start_time
    
    print(f"Batch SCARF - Update Time: {batch_time:.3f}s")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Update Performance:")
    print(f"  Original:  {orig_time:.3f}s (baseline)")
    print(f"  Optimized: {opt_time:.3f}s ({orig_time/opt_time:.1f}x faster)")
    print(f"  Batch:     {batch_time:.3f}s ({orig_time/batch_time:.1f}x faster)")
    
    print(f"\nActive RF Detection:")
    print(f"  Original:  {orig_active_time:.3f}s (baseline)")
    print(f"  Optimized: {opt_active_time:.3f}s ({orig_active_time/opt_active_time:.1f}x faster)")
    
    print(f"\nTotal Processing:")
    orig_total = orig_time + orig_active_time
    opt_total = opt_time + opt_active_time
    print(f"  Original:  {orig_total:.3f}s")
    print(f"  Optimized: {opt_total:.3f}s ({orig_total/opt_total:.1f}x faster)")
    
    # Memory usage comparison (approximate)
    print(f"\nMemory Usage Improvements:")
    print(f"  - Replaced object arrays with int32 arrays (~4x memory reduction)")
    print(f"  - Added caching for active events (reduces repeated computation)")
    print(f"  - Used float32 instead of float64 for images (2x memory reduction)")

def benchmark_initialization():
    """Benchmark initialization performance."""
    
    print("\n" + "=" * 60)  
    print("INITIALIZATION BENCHMARK")
    print("=" * 60)
    
    res = (640, 480)
    rf_size = 14
    alpha = 1.0
    C = 0.3
    
    # Original initialization
    start_time = time.time()
    scarf_orig = SCARF_Original(res, rf_size, alpha, C)
    orig_init_time = time.time() - start_time
    
    # Optimized initialization
    start_time = time.time()
    scarf_opt = SCARF_Optimized(res, rf_size, alpha, C)
    opt_init_time = time.time() - start_time
    
    print(f"Original Initialization:  {orig_init_time:.3f}s")
    print(f"Optimized Initialization: {opt_init_time:.3f}s ({orig_init_time/opt_init_time:.1f}x faster)")

if __name__ == "__main__":
    benchmark_initialization()
    benchmark_scarf_performance()
