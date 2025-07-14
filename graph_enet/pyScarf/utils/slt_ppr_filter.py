import numpy as np

class VNoiseFilter:
    """
    Full salt-and-pepper filter with spatial and temporal filtering.
    """
    def __init__(self):
        self.x_sfilter = False
        self.x_tfilter = False

        self.t_sfilter = 0.0
        self.s_sfilter = 1
        self.t_tfilter = 0.0

        self.initialised = False
        self.SAE = None       # Surface of Active Events
        self.POL = None       # Last polarity seen
        self.height = None
        self.width = None

    def initialise(self, width, height):
        self.width = width
        self.height = height
        self.SAE = np.zeros((height, width), dtype=np.float64)
        self.POL = np.ones((height, width), dtype=np.uint8) * 255  # 255 = invalid polarity
        self.initialised = True

    def active(self):
        return self.initialised

    def use_temporal_filter(self, t_param):
        self.x_tfilter = True
        self.t_tfilter = t_param

    def use_spatial_filter(self, t_param, s_param):
        self.x_sfilter = True
        self.t_sfilter = t_param
        self.s_sfilter = s_param

    def check(self, x, y, p, t):
        """
        Returns True if the event is valid (passes the filter),
        False if it's considered noise.
        """
        # if not self.initialised:
        #    raise RuntimeError("VNoiseFilter not initialized. Call initialise() first.")

        add = True

        # === TEMPORAL FILTER ===
        if self.x_tfilter:
            if p == self.POL[y, x]:
                if t - self.SAE[y, x] < self.t_tfilter:
                    self.SAE[y, x] = t
                    return False  # reject

        # === SPATIAL FILTER ===
        if self.x_sfilter:
            add = False
            xl = max(x - self.s_sfilter, 0)
            xh = min(x + self.s_sfilter + 1, self.width)
            yl = max(y - self.s_sfilter, 0)
            yh = min(y + self.s_sfilter + 1, self.height)

            for xi in range(xl, xh):
                for yi in range(yl, yh):
                    if t - self.SAE[yi, xi] < self.t_sfilter:
                        add = True
                        break
                #if add:
                    #break

        # === UPDATE SAE and POL ===
        self.SAE[y, x] = t
        self.POL[y, x] = p

        return add


class SpatialFilter:
    """
    Lightweight spatial-only salt-and-pepper filter.
    Maintains two SAE matrices (one per polarity).
    """
    def __init__(self):
        self.sae = [None, None]  # One SAE per polarity (0,1)
        self.period = 0.0
        self.range = 1
        self.height = None
        self.width = None

    def initialise(self, height, width, period=0.1, spatial_range=1):
        self.height = height
        self.width = width
        self.period = period
        self.range = spatial_range
        h_pad = height + (2 * spatial_range)
        w_pad = width + (2 * spatial_range)
        self.sae[0] = np.zeros((h_pad, w_pad), dtype=np.float64)
        self.sae[1] = np.zeros((h_pad, w_pad), dtype=np.float64)

    def check(self, x, y, p, ts):
        """
        Returns True if the event is NOT noise, False if it is noise.
        """
        
        fr = 2 * self.range + 1
        x_pad = x + self.range
        y_pad = y + self.range
        sae_p = self.sae[p]

         # --- Check if event is too isolated (central pixel is too old)
        if ts - self.period > sae_p[y_pad, x_pad]:
            pass_event = False
        else:
            pass_event = True

         # --- Update the patch: same as OpenCV ROI: Rect(x, y, fr, fr)
        sae_p[y_pad : y_pad + fr, x_pad : x_pad + fr] = ts

        return pass_event
