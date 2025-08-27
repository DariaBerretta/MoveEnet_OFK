#!/usr/bin/env python3
"""
Quick validation test to ensure SCARF optimizations work correctly.
Run this to verify both original and optimized versions produce identical results.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from graph_enet.pyScarf.scarf.scarf_class import SCARF as SCARF_Original
from graph_enet.pyScarf.scarf.scarf_class_optimized import SCARF_Optimized

def test_identical_behavior():
    """Test that optimized version produces identical results to original."""
    
    print("🧪 Testing SCARF optimization correctness...")
    
    # Test parameters
    res = (640, 480)
    rf_size = 14
    alpha = 1.0
    C = 0.3
    
    # Generate deterministic test events
    np.random.seed(42)
    n_events = 1000
    events = []
    for _ in range(n_events):
        events.append((
            np.random.randint(0, res[0]),
            np.random.randint(0, res[1]), 
            np.random.randint(0, 2)
        ))
    
    # Test original SCARF
    scarf_orig = SCARF_Original(res, rf_size, alpha, C)
    for u, v, p in events:
        scarf_orig.update(u, v, p)
    
    orig_surface = scarf_orig.get_surface().copy()
    orig_active_rfs = scarf_orig.get_active_RF(0.1)
    
    # Test optimized SCARF with same seed
    np.random.seed(42)
    scarf_opt = SCARF_Optimized(res, rf_size, alpha, C)
    for u, v, p in events:
        scarf_opt.update(u, v, p)
    
    opt_surface = scarf_opt.get_surface().copy()
    opt_active_rfs = scarf_opt.get_active_RF(0.1)
    
    # Verify identical results
    surface_identical = np.allclose(orig_surface, opt_surface, rtol=1e-6)
    active_rf_count_identical = len(orig_active_rfs) == len(opt_active_rfs)
    
    print(f"✅ Surface images identical: {surface_identical}")
    print(f"✅ Active RF counts identical: {active_rf_count_identical} ({len(orig_active_rfs)} vs {len(opt_active_rfs)})")
    
    if surface_identical and active_rf_count_identical:
        print("🎉 All tests passed! Optimized version produces identical results.")
        return True
    else:
        print("❌ Tests failed! Results differ between versions.")
        return False

def test_performance_hint():
    """Show a quick performance comparison."""
    
    print("\n⚡ Quick performance check...")
    
    import time
    
    res = (640, 480) 
    rf_size = 14
    
    # Initialization speed
    start = time.time()
    scarf_orig = SCARF_Original(res, rf_size)
    orig_init = time.time() - start
    
    start = time.time()  
    scarf_opt = SCARF_Optimized(res, rf_size)
    opt_init = time.time() - start
    
    print(f"📊 Initialization: {orig_init:.3f}s → {opt_init:.3f}s ({orig_init/opt_init:.1f}x faster)")
    
    # Update speed (small sample)
    events = [(100, 100, 1), (200, 200, 0), (300, 300, 1)] * 100
    
    start = time.time()
    for u, v, p in events:
        scarf_orig.update(u, v, p)
    orig_update = time.time() - start
    
    start = time.time()
    for u, v, p in events:
        scarf_opt.update(u, v, p)
    opt_update = time.time() - start
    
    if opt_update > 0:
        print(f"📊 Event processing: {orig_update:.4f}s → {opt_update:.4f}s ({orig_update/opt_update:.1f}x faster)")
    else:
        print(f"📊 Event processing: {orig_update:.4f}s → {opt_update:.4f}s (too fast to measure accurately)")

if __name__ == "__main__":
    success = test_identical_behavior()
    test_performance_hint()
    
    if success:
        print("\n✨ Ready to use optimized SCARF class!")
        print("💡 To use optimizations:")
        print("   from graph_enet.pyScarf.scarf.scarf_class_optimized import SCARF_Optimized as SCARF")
    else:
        print("\n⚠️  Please check optimization implementation.")
    
    exit(0 if success else 1)
