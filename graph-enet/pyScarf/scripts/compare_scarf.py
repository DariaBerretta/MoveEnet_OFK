#!/usr/bin/env python3
import os
import cv2
import numpy as np
import imageio.v3 as iio

# ════════════════════════════════════════════════════════════════════
# 1) UPDATE THESE TWO PATHS to match where your EXR files live:
#    - cpp_dir should point to folder containing scarf_cpp_t01.exr … scarf_cpp_t60.exr
#    - py_dir  should point to folder containing scarf_py_t01.exr  … scarf_py_t60.exr
# ════════════════════════════════════════════════════════════════════
cpp_dir = "/data/scarf_snapshots_cpp"
py_dir  = "/data/scarf_snapshots_py"

# Number of snapshots you expect to compare
num_frames = 5449 -1

def load_exr_as_float32(path):
    """
    Use OpenCV to load a float32 EXR (IMREAD_UNCHANGED preserves the float32 values).
    Returns a NumPy array of dtype float32, or None on failure.
    """
    # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # if img is None:
    #     return None
    # If img comes in as (H, W, C) in some builds, collapse to single‐channel:
    # if img.ndim == 3 and img.shape[2] == 1:
    #    img = img[:, :, 0]

    """
    Use imageio to load .exr files in float32 format.
    """
    img = iio.imread(path)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    
    return img.astype(np.float32)

if __name__ == "__main__":
    
    print(f"Comparing C++ vs Python SCARF snapshots in\n  C++: {cpp_dir}\n  Py:  {py_dir}\n")
    
    for i in range(1, num_frames + 1):
        
        fname_cpp = f"scarf_cpp_t{i:04d}.exr"
        fname_py  = f"scarf_py_t{i:04d}.exr"
        path_cpp  = os.path.join(cpp_dir, fname_cpp)
        path_py   = os.path.join(py_dir,  fname_py)

        if not os.path.isfile(path_cpp):
            print(f"[t={i:04d}s]  ✕ Missing C++ file: {path_cpp}")
            continue
        if not os.path.isfile(path_py):
            print(f"[t={i:04d}s]  ✕ Missing Python file: {path_py}")
            continue

        img_cpp = load_exr_as_float32(path_cpp)
        img_py  = load_exr_as_float32(path_py)

        if img_cpp is None:
            print(f"[t={i:04d}s]  ✕ Failed to load C++ EXR: {path_cpp}")
            continue
        if img_py is None:
            print(f"[t={i:04d}s]  ✕ Failed to load Python EXR: {path_py}")
            continue

        if img_cpp.shape != img_py.shape:
            print(f"[t={i:04d}s]  ✕ Shape mismatch → C++: {img_cpp.shape}, Py: {img_py.shape}")
            continue

        diff = img_cpp - img_py
        abs_diff = np.abs(diff)
        max_err = float(abs_diff.max())
        mean_err = float(abs_diff.mean())

        print(f"[t={i:04d}s]  Max abs err : {max_err:.6f}  |  Mean abs diff: {mean_err:.6f}")