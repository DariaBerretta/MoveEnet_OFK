#!/usr/bin/env python3
"""
Realistic performance benchmark for SCARF optimization analysis.
Tests various event batch sizes to find optimal performance characteristics.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from graph_enet.pyScarf.scarf.scarf_class import SCARF as SCARF_Original
from graph_enet.pyScarf.scarf.scarf_class_optimized import SCARF_Optimized

def generate_test_events(n_events=50000, res=(640, 480)):
    """Generate synthetic event data for benchmarking."""
    events = np.zeros(n_events, dtype=[('x', np.int32), ('y', np.int32), ('pol', np.int32)])
    events['x'] = np.random.randint(0, res[0], n_events)
    events['y'] = np.random.randint(0, res[1], n_events)
    events['pol'] = np.random.randint(0, 2, n_events)
    return events

def benchmark_batch_sizes():
    """Test different batch sizes to find optimal performance."""
    
    res = (640, 480)
    rf_size = 14
    alpha = 1.0
    C = 0.3
    
    # Test different batch sizes
    batch_sizes = [100, 500, 1000, 5000, 10000, 25000, 50000]
    
    print("=" * 70)
    print("BATCH SIZE PERFORMANCE ANALYSIS")
    print("=" * 70)
    print(f"{'Batch Size':<12} {'Original':<12} {'Optimized':<12} {'Batch Update':<12} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size in batch_sizes:
        events = generate_test_events(batch_size, res)
        
        # Original SCARF
        scarf_orig = SCARF_Original(res, rf_size, alpha, C)
        start_time = time.time()
        for event in events:
            scarf_orig.update(event['x'], event['y'], event['pol'])
        orig_time = time.time() - start_time
        
        # Optimized SCARF individual updates
        scarf_opt = SCARF_Optimized(res, rf_size, alpha, C)
        start_time = time.time()
        for event in events:
            scarf_opt.update(event['x'], event['y'], event['pol'])
        opt_time = time.time() - start_time
        
        # Optimized SCARF batch updates
        scarf_batch = SCARF_Optimized(res, rf_size, alpha, C)
        start_time = time.time()
        scarf_batch.update_batch(events)
        batch_time = time.time() - start_time
        
        speedup = orig_time / opt_time if opt_time > 0 else 0
        batch_speedup = orig_time / batch_time if batch_time > 0 else 0
        
        print(f"{batch_size:<12} {orig_time:<12.4f} {opt_time:<12.4f} {batch_time:<12.4f} {speedup:<10.2f}x")
    
    return batch_sizes

def benchmark_realistic_workload():
    """Benchmark with realistic GraphEnet-v2 workload patterns."""
    
    print("\n" + "=" * 70)
    print("REALISTIC WORKLOAD BENCHMARK") 
    print("=" * 70)
    
    res = (640, 480)
    rf_size = 14
    alpha = 1.0
    C = 0.3
    
    # Simulate typical training batch processing
    n_samples = 10
    events_per_sample = 5000  # Typical events per time window
    
    print(f"Simulating {n_samples} samples with {events_per_sample} events each")
    print(f"Total events: {n_samples * events_per_sample}")
    
    # Test original implementation
    print("\nTesting Original SCARF...")
    total_orig_time = 0
    total_active_rfs = 0
    
    for i in range(n_samples):
        events = generate_test_events(events_per_sample, res)
        scarf_orig = SCARF_Original(res, rf_size, alpha, C)
        
        start_time = time.time()
        for event in events:
            scarf_orig.update(event['x'], event['y'], event['pol'])
        
        active_rfs = scarf_orig.get_active_RF(0.15)
        update_time = time.time() - start_time
        
        total_orig_time += update_time
        total_active_rfs += len(active_rfs)
        
        if i == 0:
            print(f"  Sample 1: {update_time:.3f}s, {len(active_rfs)} active RFs")
    
    avg_orig_time = total_orig_time / n_samples
    avg_active_rfs = total_active_rfs / n_samples
    
    print(f"  Average per sample: {avg_orig_time:.3f}s, {avg_active_rfs:.0f} active RFs")
    print(f"  Total time: {total_orig_time:.3f}s")
    
    # Test optimized implementation
    print("\nTesting Optimized SCARF...")
    total_opt_time = 0
    
    for i in range(n_samples):
        events = generate_test_events(events_per_sample, res)
        scarf_opt = SCARF_Optimized(res, rf_size, alpha, C)
        
        start_time = time.time()
        for event in events:
            scarf_opt.update(event['x'], event['y'], event['pol'])
        
        active_rfs = scarf_opt.get_active_RF(0.15)
        update_time = time.time() - start_time
        
        total_opt_time += update_time
        
        if i == 0:
            print(f"  Sample 1: {update_time:.3f}s, {len(active_rfs)} active RFs")
    
    avg_opt_time = total_opt_time / n_samples
    
    print(f"  Average per sample: {avg_opt_time:.3f}s")
    print(f"  Total time: {total_opt_time:.3f}s")
    
    # Performance summary
    print("\n" + "-" * 50)
    print("REALISTIC WORKLOAD SUMMARY")
    print("-" * 50)
    print(f"Original implementation:  {total_orig_time:.3f}s total")
    print(f"Optimized implementation: {total_opt_time:.3f}s total")
    print(f"Performance improvement:  {total_orig_time/total_opt_time:.2f}x faster")
    print(f"Time saved per sample:    {avg_orig_time - avg_opt_time:.4f}s")
    print(f"Time saved for full run:  {total_orig_time - total_opt_time:.3f}s")

def analyze_memory_usage():
    """Analyze memory usage improvements."""
    
    print("\n" + "=" * 70)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 70)
    
    res = (640, 480)
    rf_size = 14
    alpha = 1.0
    C = 0.3
    
    # Calculate theoretical memory usage
    width, height = res
    rf_res = (int((res[0] // rf_size) - 1), (res[1] // rf_size) - 1)
    rf_w, rf_h = rf_res
    
    dims_w = width // (rf_w + 1)
    dims_h = height // (rf_h + 1)
    dims_w -= dims_w % 2
    dims_h -= dims_h % 2
    N = int(dims_w * dims_h * alpha * 0.5)
    
    n_rfs = rf_w * rf_h
    
    print(f"Configuration: {res} resolution, {rf_size}x{rf_size} RF size")
    print(f"Number of RFs: {n_rfs}")
    print(f"Buffer size per RF: {N}")
    print(f"Connection map size: {height} x {width} x 4")
    
    # Original memory usage
    orig_cons_map = height * width * 4 * 8  # object references (64-bit)
    orig_image = height * width * 8  # float64
    orig_rf_buffers = n_rfs * N * 4 * 4  # int32
    orig_total = orig_cons_map + orig_image + orig_rf_buffers
    
    # Optimized memory usage  
    opt_cons_map = height * width * 4 * 4  # int32 indices
    opt_cons_active = height * width * 4 * 1  # bool
    opt_image = height * width * 4  # float32
    opt_rf_buffers = orig_rf_buffers  # same
    opt_total = opt_cons_map + opt_cons_active + opt_image + opt_rf_buffers
    
    print(f"\nMemory Usage Breakdown:")
    print(f"  Original:")
    print(f"    Connection map: {orig_cons_map / 1024 / 1024:.1f} MB (object array)")
    print(f"    Image surface:  {orig_image / 1024 / 1024:.1f} MB (float64)")
    print(f"    RF buffers:     {orig_rf_buffers / 1024 / 1024:.1f} MB")
    print(f"    Total:          {orig_total / 1024 / 1024:.1f} MB")
    
    print(f"  Optimized:")
    print(f"    Connection map: {opt_cons_map / 1024 / 1024:.1f} MB (int32)")
    print(f"    Connection active: {opt_cons_active / 1024 / 1024:.1f} MB (bool)")
    print(f"    Image surface:  {opt_image / 1024 / 1024:.1f} MB (float32)")
    print(f"    RF buffers:     {opt_rf_buffers / 1024 / 1024:.1f} MB")
    print(f"    Total:          {opt_total / 1024 / 1024:.1f} MB")
    
    memory_reduction = (orig_total - opt_total) / orig_total * 100
    print(f"\n  Memory reduction: {memory_reduction:.1f}% ({orig_total/opt_total:.1f}x less)")

if __name__ == "__main__":
    benchmark_batch_sizes()
    benchmark_realistic_workload()
    analyze_memory_usage()
