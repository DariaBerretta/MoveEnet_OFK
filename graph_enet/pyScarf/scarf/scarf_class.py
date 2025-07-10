import numpy as np

class CARF:

    def __init__(self , N, img, C):
        """
        Create a single CARF (Center Active Receptive Field) data structure.
        """
        self.N = N      # Lenght of the Center Active Receptive Field's buffer
        self.points = np.zeros((N, 4), dtype=np.int32)  # (u, v, p, a)
        self.i = 0      # Current index in the circular buffer. 
        self.img = img
        self.C = C

    def add(self, pnt):
        """
        Add a point to the CARF buffer and update the SCARF image.
        """
        u, v, p, a = pnt
        self.i = (self.i + 1) % self.N
        old_u, old_v, old_p, old_a = self.points[self.i]

        if old_a:
            self.img[old_v, old_u] -= self.C 
        if a:
            self.img[v, u] += self.C

        self.points[self.i] = pnt

    def get_active_events(self):
        """
        Get from each RF (CARF) only the active events
        """
        return self.points[self.points[:, 3] == 1]     # return datatype 



class SCARF:

    def __init__(self, res, rf_size, alpha=1.0, C=0.3):
        """
        Entry-point to initialize SCARF using receptive field size (instead of RF resolution).
        """
        self.res = res
        if rf_size % 2 != 0:
            rf_size += 1
        rf_res = (int((res[0] // rf_size) - 1), (res[1] // rf_size) - 1)
        self._init_internal(rf_res, alpha, C)

    def _init_internal(self, rf_res, alpha, C):
        """
        Initialize the SCARF structure with receptive fields and connection maps.
        """
        width, height = self.res
        rf_w, rf_h = rf_res
        self.img = np.zeros((height, width), dtype=np.float32)
        self.count = rf_res

        dims_w = width // (rf_w + 1)
        dims_h = height // (rf_h + 1)
        dims_w -= dims_w % 2
        dims_h -= dims_h % 2
        self.dims = (dims_w, dims_h)
        self.N = int(dims_w * dims_h * alpha * 0.5)

        self.rfs = [CARF(self.N, self.img, C) for _ in range(rf_w * rf_h)]
        self.cons_map = np.empty((height, width, 4), dtype=object)

        for y in range(height):
            for x in range(width):
                xm = x - dims_w // 2
                ym = y - dims_h // 2
                rfx = xm // dims_w
                rfy = ym // dims_h

                conns = [None] * 4
                i = 0

                if 0 <= rfx < rf_w and 0 <= rfy < rf_h:
                    conns[i] = self.rfs[rfy * rf_w + rfx]
                i += 1

                kx = (dims_w + (xm % dims_w)) % dims_w
                ky = (dims_h + (ym % dims_h)) % dims_h
                top, bot = ky < dims_h // 2, ky >= dims_h // 2
                lef, rig = kx < dims_w // 2, kx >= dims_w // 2

                neighbors = []
                if top: neighbors.append((rfx, rfy - 1))
                if bot: neighbors.append((rfx, rfy + 1))
                if lef: neighbors.append((rfx - 1, rfy))
                if rig: neighbors.append((rfx + 1, rfy))
                if top and lef: neighbors.append((rfx - 1, rfy - 1))
                if top and rig: neighbors.append((rfx + 1, rfy - 1))
                if bot and lef: neighbors.append((rfx - 1, rfy + 1))
                if bot and rig: neighbors.append((rfx + 1, rfy + 1))

                for rx, ry in neighbors:
                    if i >= 4:
                        break
                    if 0 <= rx < rf_w and 0 <= ry < rf_h:
                        conns[i] = self.rfs[ry * rf_w + rx]
                        i += 1

                self.cons_map[y, x] = conns
    
    def update(self, u, v, p):
        """
        Update SCARF with a single event.
        """
        if not (0 <= u < self.res[0] and 0 <= v < self.res[1]):
            return
        for i, carf in enumerate(self.cons_map[v, u]):
            if carf is not None:
                carf.add((u, v, p, int(i == 0)))

    def get_surface(self):
        return self.img
    
    def get_active_RF(self, threshold_ratio=0.3):
        """
        Return a list of (RF index, carf, active_events) for each RF that has more than (threshold_ratio * N) active events
        """
        active_rfs = []
        for idx, carf in enumerate(self.rfs):
            events = carf.get_active_events()

            #print(f"[INFO] Buffer dimension: {carf.N}")
            #print(f"[INFO] ratio of active events: {len(events)}/{len(carf.points)}")

            # If all the active events are all in the same position the RF is discarded
            if np.unique(events[:, :2], axis=0).shape[0] == 1:
                continue

            if len(events) > carf.N * threshold_ratio:
                active_rfs.append((idx, carf, events))
        return active_rfs

