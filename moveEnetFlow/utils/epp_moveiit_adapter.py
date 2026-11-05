"""Event-only geometry adapter for ATIS move-iit-hpe -> DHP19 MeanLabel PointNet.

The original EventPointPose PointNet, RasEPC and event FIFO are not modified.
The adapter's input is a [N,4] local, zero-based ATIS array [x,y,t,p].
The returned [N,4] array uses the original EPP MeanLabel coordinate format,
which RasEventCloud.convert() indexes using integer(x)-1 and integer(y)-1.

IMPORTANT: the EPP MeanLabel DHP19 input conversion itself reverses the Y
axis compared with the hpe-core data.log coordinate convention. We reproduce
this same virtual-log conversion here (including original boundary indexing),
not an additional, unvalidated orientation fix for the new dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Geometry:
    source_width: int = 1280
    source_height: int = 720
    target_width: int = 346
    target_height: int = 260
    mode: str = "stretch"

    def __post_init__(self):
        if min(self.source_width, self.source_height, self.target_width, self.target_height) < 2:
            raise ValueError("All dimensions must be at least 2")
        if self.mode not in ("stretch", "letterbox"):
            raise ValueError("mode must be 'stretch' or 'letterbox'")

    @property
    def coefficients(self) -> Tuple[float, float, float, float]:
        # Align both corner pixel centers, so an inverse is defined precisely.
        sx = (self.target_width - 1.0) / (self.source_width - 1.0)
        sy = (self.target_height - 1.0) / (self.source_height - 1.0)
        if self.mode == "stretch":
            return sx, sy, 0.0, 0.0
        s = min(sx, sy)
        padx = ((self.target_width - 1.0) - (self.source_width - 1.0) * s) / 2.0
        pady = ((self.target_height - 1.0) - (self.source_height - 1.0) * s) / 2.0
        return s, s, padx, pady

    def to_virtual_log(self, xy: np.ndarray) -> np.ndarray:
        """Map native ATIS image coordinates to virtual DHP19 hpe-core log XY."""
        xy = np.asarray(xy, dtype=np.float64)
        if xy.shape[-1] != 2:
            raise ValueError("xy must end with a dimension of size two")
        sx, sy, ox, oy = self.coefficients
        virtual = np.empty_like(xy, dtype=np.float64)
        virtual[..., 0] = np.floor(xy[..., 0] * sx + ox + 0.5)
        virtual[..., 1] = np.floor(xy[..., 1] * sy + oy + 0.5)
        return virtual

    def to_native_pose(self, xy: np.ndarray) -> np.ndarray:
        """Map native PointNet XY logits/argmax coordinate indices to ATIS pixels.

        Does not apply any extra flip: original EPP output head coordinates are
        treated as predicted image-plane (x,y), as in our DHP19 deployment.
        Output is intentionally not clamped; a letterbox prediction can fall in
        the padded region, and such predictions should be visible in diagnostics.
        """
        xy = np.asarray(xy, dtype=np.float64)
        if xy.shape[-1] != 2:
            raise ValueError("xy must end with a dimension of size two")
        sx, sy, ox, oy = self.coefficients
        native = np.empty_like(xy, dtype=np.float64)
        native[..., 0] = (xy[..., 0] - ox) / sx
        native[..., 1] = (xy[..., 1] - oy) / sy
        return native

    def preprocess(self, events: np.ndarray) -> Tuple[np.ndarray, int]:
        """Return accepted [x_epp,y_epp,t,p] float64 and invalid-event count.

        No DHP19-specific hot-pixel, background, or IR masks are applied to ATIS.
        No event count, timestamp, or event order is changed except validation.
        """
        events = np.asarray(events, dtype=np.float64)
        if events.ndim != 2 or events.shape[1] != 4:
            raise ValueError("Expected [N,4] events")
        if events.shape[0] == 0:
            return np.empty((0, 4), dtype=np.float64), 0

        x, y, t, p = events.T
        valid = (
            np.isfinite(events).all(axis=1)
            & (x >= 0) & (x < self.source_width)
            & (y >= 0) & (y < self.source_height)
            & (x == np.floor(x)) & (y == np.floor(y))
            & ((p == 0) | (p == 1))
        )
        count_invalid = int(np.count_nonzero(~valid))
        good = events[valid].copy()
        if good.shape[0] == 0:
            return good, count_invalid

        virtual_log = self.to_virtual_log(good[:, :2])
        if (virtual_log[:, 0] < 0).any() or (virtual_log[:, 0] >= self.target_width).any() or (
            virtual_log[:, 1] < 0
        ).any() or (virtual_log[:, 1] >= self.target_height).any():
            raise AssertionError("Internal geometry mapping produced out-of-bounds pixel")

        # The *same* conversion as in OnlineEventPreprocessor.already_filtered:
        # x_epp = x_log+1; y_epp = H-1-y_log; RasEPC indexes int()-1.
        good[:, 0] = virtual_log[:, 0] + 1
        good[:, 1] = (self.target_height - 1) - virtual_log[:, 1]
        return good, count_invalid
