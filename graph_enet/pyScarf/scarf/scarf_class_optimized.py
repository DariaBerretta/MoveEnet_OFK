# Copyright (c) 2026 Istituto Italiano di Tecnologia
# Author: Daria Berretta

# Permission is granted to use, copy, modify, and distribute this software 
# for non-commercial purposes only. Commercial use is strictly prohibited 
# without prior written permission from the author.

# To request a commercial license, contact: arren.glover@iit.it
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

import numpy as np

class CARF_Optimized:
    """
    Optimized version of CARF with reduced memory allocations and faster operations.
    """
    
    def __init__(self, N, img, C):
        """
        Create a single CARF (Center Active Receptive Field) data structure.
        """
        self.N = N
        self.points = np.zeros((N, 4), dtype=np.int32)  # (u, v, p, a)
        self.i = 0
        self.img = img
        self.C = C
        # Pre-compute C values for faster conditional updates
        self.C_float = np.float32(C)
        
        # Cache for active events to avoid repeated filtering
        self._active_mask = np.zeros(N, dtype=np.bool_)
        self._cache_valid = False

    def add(self, u, v, p, a):
        """
        Add a point to the CARF buffer and update the SCARF image.
        Optimized to avoid tuple creation/unpacking.
        """
        self.i = (self.i + 1) % self.N
        
        # Get old values directly from array
        old_u, old_v, old_p, old_a = self.points[self.i]

        # Update image with vectorized operations when possible
        if old_a:
            self.img[old_v, old_u] -= self.C_float
        if a:
            self.img[v, u] += self.C_float

        # Update point directly
        self.points[self.i, 0] = u
        self.points[self.i, 1] = v
        self.points[self.i, 2] = p
        self.points[self.i, 3] = a
        
        # Invalidate cache
        self._cache_valid = False

    def get_active_events(self):
        """
        Get from each RF (CARF) only the active events.
        Uses caching to avoid repeated numpy operations.
        """
        if not self._cache_valid:
            self._active_mask = (self.points[:, 3] == 1)
            self._cache_valid = True
        
        return self.points[self._active_mask]

    def invalidate_cache(self):
        """Force cache invalidation for active events."""
        self._cache_valid = False


class SCARF_Optimized:
    """
    Highly optimized version of SCARF with significant performance improvements:
    - Uses integer arrays instead of object arrays for connections
    - Precomputes connection maps more efficiently  
    - Reduces memory allocations and function call overhead
    - Implements vectorized operations where possible
    """

    def __init__(self, res, rf_size, alpha=1.0, C=0.3):
        """
        Entry-point to initialize SCARF using receptive field size.
        """
        self.res = res
        if rf_size % 2 != 0:
            rf_size += 1
        rf_res = (int((res[0] // rf_size) - 1), (res[1] // rf_size) - 1)
        self._init_internal_optimized(rf_res, alpha, C)

    def _init_internal_optimized(self, rf_res, alpha, C):
        """
        Optimized initialization with reduced memory footprint and faster setup.
        """
        width, height = self.res
        rf_w, rf_h = rf_res
        
        # Use float32 for image to reduce memory usage
        self.img = np.zeros((height, width), dtype=np.float32)
        self.count = rf_res
        self.rf_w, self.rf_h = rf_w, rf_h

        dims_w = width // (rf_w + 1)
        dims_h = height // (rf_h + 1)
        dims_w -= dims_w % 2
        dims_h -= dims_h % 2
        self.dims = (dims_w, dims_h)
        self.N = int(dims_w * dims_h * alpha * 0.5)

        # Create RFs with optimized CARF
        self.rfs = [CARF_Optimized(self.N, self.img, C) for _ in range(rf_w * rf_h)]
        
        # Use integer arrays instead of object arrays for much faster indexing
        # Store RF indices (-1 for None connections)
        self.cons_map = np.full((height, width, 4), -1, dtype=np.int32)
        self.cons_active = np.zeros((height, width, 4), dtype=np.bool_)  # Track which connections are active
        
        # Precompute offsets for faster lookup
        self.dims_w_half = dims_w // 2
        self.dims_h_half = dims_h // 2
        
        # Vectorized computation of connection maps
        self._compute_connections_vectorized(width, height, rf_w, rf_h, dims_w, dims_h)

    def _compute_connections_vectorized(self, width, height, rf_w, rf_h, dims_w, dims_h):
        """
        Vectorized computation of connection maps for much faster initialization.
        """
        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
        
        # Compute RF coordinates for all pixels at once
        xm = x_coords - self.dims_w_half
        ym = y_coords - self.dims_h_half
        rfx = xm // dims_w
        rfy = ym // dims_h
        
        # Valid RF mask
        valid_rf = (rfx >= 0) & (rfx < rf_w) & (rfy >= 0) & (rfy < rf_h)
        
        # Set primary connections
        primary_rf_idx = rfy * rf_w + rfx
        self.cons_map[y_coords, x_coords, 0] = np.where(valid_rf, primary_rf_idx, -1)
        self.cons_active[y_coords, x_coords, 0] = valid_rf
        
        # Compute neighbor offsets more efficiently
        kx = (dims_w + (xm % dims_w)) % dims_w
        ky = (dims_h + (ym % dims_h)) % dims_h
        
        # Boolean masks for positions
        top = ky < self.dims_h_half
        bot = ky >= self.dims_h_half
        lef = kx < self.dims_w_half
        rig = kx >= self.dims_w_half
        
        # Define neighbor offsets as arrays for vectorized operations
        neighbor_offsets = [
            (0, -1),  # top
            (0, 1),   # bottom  
            (-1, 0),  # left
            (1, 0),   # right
            (-1, -1), # top-left
            (1, -1),  # top-right
            (-1, 1),  # bottom-left
            (1, 1)    # bottom-right
        ]
        
        neighbor_conditions = [
            top, bot, lef, rig, top & lef, top & rig, bot & lef, bot & rig
        ]
        
        # Fill neighbor connections
        conn_idx = 1
        for (dx, dy), condition in zip(neighbor_offsets, neighbor_conditions):
            if conn_idx >= 4:
                break
                
            neighbor_rfx = rfx + dx
            neighbor_rfy = rfy + dy
            
            neighbor_valid = (condition & 
                            (neighbor_rfx >= 0) & (neighbor_rfx < rf_w) &
                            (neighbor_rfy >= 0) & (neighbor_rfy < rf_h))
            
            neighbor_rf_idx = neighbor_rfy * rf_w + neighbor_rfx
            
            mask = neighbor_valid & (self.cons_map[y_coords, x_coords, conn_idx] == -1)
            self.cons_map[y_coords, x_coords, conn_idx] = np.where(mask, neighbor_rf_idx, -1)
            self.cons_active[y_coords, x_coords, conn_idx] = mask
            
            if np.any(mask):
                conn_idx += 1

    def update(self, u, v, p):
        """
        Optimized update method with reduced function call overhead.
        """
        # Bounds check first (most common early exit)
        if not (0 <= u < self.res[0] and 0 <= v < self.res[1]):
            return
            
        # Get connections for this pixel (much faster than object array indexing)
        connections = self.cons_map[v, u]
        active_mask = self.cons_active[v, u]
        
        # Process only active connections
        for i in range(4):
            if active_mask[i]:
                rf_idx = connections[i]
                self.rfs[rf_idx].add(u, v, p, int(i == 0))

    def update_batch(self, events):
        """
        Batch update method for processing multiple events efficiently.
        Optimized to reduce function call overhead.
        """
        # Pre-filter valid events in batch
        if hasattr(events, 'dtype') and events.dtype.names:
            # Structured array format
            x_vals = events['x']
            y_vals = events['y'] 
            p_vals = events['pol']
        else:
            # Regular array format
            x_vals = events[:, 0]
            y_vals = events[:, 1]
            p_vals = events[:, 2]
        
        # Vectorized bounds checking
        valid_mask = ((x_vals >= 0) & (x_vals < self.res[0]) & 
                     (y_vals >= 0) & (y_vals < self.res[1]))
        
        # Process only valid events
        valid_x = x_vals[valid_mask]
        valid_y = y_vals[valid_mask]
        valid_p = p_vals[valid_mask]
        
        # Bulk update - still needs individual processing for connection logic
        for x, y, p in zip(valid_x, valid_y, valid_p):
            # Inline the update logic to reduce function call overhead
            connections = self.cons_map[y, x]
            active_mask = self.cons_active[y, x]
            
            # Process only active connections
            for i in range(4):
                if active_mask[i]:
                    rf_idx = connections[i]
                    self.rfs[rf_idx].add(x, y, p, int(i == 0))

    def get_surface(self):
        return self.img
    
    def get_active_RF(self, threshold_ratio: float = 0.15):
        """
        Optimized active RF detection with early exit conditions.
        """
        active_rfs = []
        threshold = self.N * threshold_ratio
        
        for idx, carf in enumerate(self.rfs):
            events = carf.get_active_events()
            
            # Quick length check first (fastest)
            if len(events) <= threshold:
                continue
                
            # Check position uniqueness (expensive operation last)
            if len(events) > 0 and np.unique(events[:, :2], axis=0).shape[0] == 1:
                continue

            active_rfs.append((idx, carf, events))
        
        return active_rfs

    def reset(self):
        """
        Reset all receptive fields and the surface image.
        """
        self.img.fill(0.0)
        for carf in self.rfs:
            carf.points.fill(0)
            carf.i = 0
            carf.invalidate_cache()


def _update_image_optimized(img, old_u, old_v, old_a, u, v, a, C):
    """
    Optimized image update helper function.
    """
    if old_a:
        img[old_v, old_u] -= C
    if a:
        img[v, u] += C


class SCARF_Ultra:
    """
    Ultra-high performance version - currently an alias for SCARF_Optimized.
    Can be extended with additional optimizations like Numba JIT compilation
    when compatible NumPy versions are available.
    """
    
    def __init__(self, res, rf_size, alpha=1.0, C=0.3):
        # Initialize with optimized version 
        self.scarf_opt = SCARF_Optimized(res, rf_size, alpha, C)
        
    def update(self, u, v, p):
        """Delegates to optimized implementation."""
        return self.scarf_opt.update(u, v, p)
    
    def __getattr__(self, name):
        """Delegate other methods to optimized implementation."""
        return getattr(self.scarf_opt, name)
