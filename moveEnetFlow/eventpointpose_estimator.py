#!/usr/bin/env python3
"""EventPointPose adapter for the moveEnetFlow offline pipeline.

The official EventPointPose repository predicts the same 13 body joints used by
this project. The output order is kept unchanged and must remain:

0 head, 1 shoulderR, 2 shoulderL, 3 elbowR, 4 elbowL, 5 hipL, 6 hipR,
7 handR, 8 handL, 9 kneeR, 10 kneeL, 11 footR, 12 footL.

Coordinates returned by EventPointPoseEstimator.predict_events are 2D pixel
coordinates in the model sensor frame, by default DHP19's 346x260 frame.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch


KEYPOINT_ORDER: Tuple[str, ...] = (
    "head",
    "shoulderR",
    "shoulderL",
    "elbowR",
    "elbowL",
    "hipL",
    "hipR",
    "handR",
    "handL",
    "kneeR",
    "kneeL",
    "footR",
    "footL",
)

SKELETON_PARENT_IDS: Tuple[int, ...] = (0, 0, 0, 1, 2, 1, 2, 3, 4, 5, 6, 9, 10)

MODEL_NAME_MAP = {
    "pointnet": "PointNet",
    "dgcnn": "DGCNN",
    "pointtransformer": "PointTrans",
    "pointtrans": "PointTrans",
}


@dataclass
class EventPointPoseConfig:
    model_type: str = "pointnet"
    checkpoint: Optional[Path] = None
    eventpointpose_repo: Path = Path("/home/EventPointPose")
    num_points: int = 2048
    sensor_width: int = 346
    sensor_height: int = 260
    num_joints: int = 13
    label: str = "mean"
    cuda_num: int = 0
    device: Optional[str] = None
    seed: int = 1
    strict_checkpoint: bool = True


def normalise_model_type(model_type: str) -> str:
    key = model_type.lower().replace("_", "").replace("-", "")
    if key not in MODEL_NAME_MAP:
        valid = ", ".join(sorted(MODEL_NAME_MAP))
        raise ValueError(f"Unknown EventPointPose model_type '{model_type}'. Valid choices: {valid}")
    return MODEL_NAME_MAP[key]


def _repo_imports(repo_path: Path):
    repo_path = repo_path.expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(
            f"EventPointPose repository not found at {repo_path}. "
            "Clone https://github.com/MasterHow/EventPointPose or pass --eventpointpose_repo."
        )
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from dataset.rasterized import RasEventCloud
    from dataset.sample import random_sample_point
    from models import Pose_DGCNN, Pose_PointNet, Pose_PointTransformer

    return RasEventCloud, random_sample_point, Pose_PointNet, Pose_DGCNN, Pose_PointTransformer


def _strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint_obj


def _patch_dgcnn_graph_feature_for_device(repo_path: Path) -> None:
    """Keep the official DGCNN architecture but make its helper device-agnostic.

    The released helper creates index tensors on cuda:0/cuda:1 directly. That
    works for the authors' evaluation scripts, but it breaks CPU inference and
    can also break non-default CUDA device placement. This patch only changes
    tensor placement to follow the input tensor device.
    """
    if str(repo_path.resolve()) not in sys.path:
        sys.path.insert(0, str(repo_path.resolve()))
    import models.DGCNN as dgcnn_module

    def get_graph_feature(x, k=20, idx=None, cudanum=0):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = dgcnn_module.knn(x, k=k)

        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    dgcnn_module.get_graph_feature = get_graph_feature


class EventPointPoseEstimator:
    """Reusable EventPointPose model wrapper.

    The wrapper imports the official model classes and RasEPC conversion code
    from the EventPointPose repository. It only adapts checkpoint loading, point
    sampling, tensor layout, output decoding, and coordinate scaling.
    """

    def __init__(self, config: EventPointPoseConfig):
        self.config = config
        self.model_name = normalise_model_type(config.model_type)
        self.repo_path = config.eventpointpose_repo.expanduser().resolve()
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        if self.model_name == "DGCNN" and config.num_points < 20:
            raise ValueError("DGCNN requires at least 20 sampled points because k=20 in the official model.")
        if self.model_name == "PointTrans" and config.num_points < 256:
            raise ValueError("Point Transformer requires at least 256 sampled points for its downsampling stages.")

        (
            self._RasEventCloud,
            self._random_sample_point,
            Pose_PointNet,
            Pose_DGCNN,
            Pose_PointTransformer,
        ) = _repo_imports(self.repo_path)
        _patch_dgcnn_graph_feature_for_device(self.repo_path)

        self.device = self._select_device(config.device, config.cuda_num)
        args = argparse.Namespace(
            model=self.model_name,
            num_points=config.num_points,
            label=config.label,
            cuda_num=config.cuda_num,
            sensor_sizeH=config.sensor_height,
            sensor_sizeW=config.sensor_width,
            num_joints=config.num_joints,
        )
        if self.model_name == "PointNet":
            self.model = Pose_PointNet(args)
        elif self.model_name == "DGCNN":
            self.model = Pose_DGCNN(args)
        elif self.model_name == "PointTrans":
            self.model = Pose_PointTransformer(args)
        else:
            raise ValueError(f"Unsupported EventPointPose model: {self.model_name}")

        self.model.to(self.device)
        if config.checkpoint is None:
            raise ValueError("A checkpoint path is required for EventPointPose inference.")
        self.load_checkpoint(config.checkpoint)
        self.model.eval()
        self.rasterizer = self._RasEventCloud(
            input_size=(4, config.sensor_height, config.sensor_width)
        )
        # runtime flags to avoid spamming repeated warnings
        self._warned_few_events = False
        self._warned_repo_missing = False

    @staticmethod
    def _select_device(device: Optional[str], cuda_num: int) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device(f"cuda:{cuda_num}")
        return torch.device("cpu")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_path = checkpoint_path.expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"EventPointPose checkpoint not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
        self.model.load_state_dict(state_dict, strict=self.config.strict_checkpoint)

    def preprocess_events(self, events: np.ndarray) -> np.ndarray:
        """Convert raw events [x, y, t, p] into official RasEPC sampled points.

        Input coordinates must already be in the EventPointPose sensor frame and
        must use the official one-based coordinate convention: x in [1, W],
        y in [1, H]. Polarity is expected as 0/1.
        """
        num_sample = self.config.num_points
        if events.size == 0:
            data = np.zeros((1, 5), dtype=np.float32)
            data_sample, _ = self._random_sample_point(data, num_sample)
            return data_sample.astype(np.float32)

        data = np.asarray(events[:, 0:4], dtype=np.float32).copy()
        data[:, 0] = np.clip(data[:, 0], 1, self.config.sensor_width)
        data[:, 1] = np.clip(data[:, 1], 1, self.config.sensor_height)
        data[:, 3] = (data[:, 3] > 0).astype(np.float32)

        # Warn if a window contains fewer raw events than the number of points
        # the model expects. In that case the official sampler will duplicate
        # or upsample events to reach `num_sample`, which can degrade results.
        try:
            raw_count = int(data.shape[0])
        except Exception:
            raw_count = 0
        if raw_count < num_sample and not self._warned_few_events:
            print(
                f"Warning: event window has only {raw_count} events but model expects {num_sample} points. "
                "Sampler will duplicate/upsample events — consider increasing --net_period or reducing --num_points.",
                file=sys.stderr,
            )
            self._warned_few_events = True

        t_min = float(np.min(data[:, 2]))
        t_max = float(np.max(data[:, 2]))
        if len(data) == 1:
            duplicate = data.copy()
            data = np.concatenate([data, duplicate], axis=0)
            data[0, 2] = 0.0
            data[1, 2] = 1e-6
        elif t_max <= t_min:
            data[:, 2] = np.linspace(0.0, 1e-6, num=len(data), dtype=np.float32)

        data = self.rasterizer.convert(data).numpy()[:, 1:]
        data_sample, _ = self._random_sample_point(data, num_sample)
        return data_sample.astype(np.float32)

    def predict_events(self, events: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run inference and return (keypoints_xy, latency_seconds)."""
        point_cloud = self.preprocess_events(events)
        tensor = torch.from_numpy(point_cloud).unsqueeze(0).float().to(self.device)
        if self.model_name != "PointTrans":
            tensor = tensor.permute(0, 2, 1)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            output_x, output_y = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency = time.perf_counter() - t0

        pred_x = torch.argmax(output_x, dim=2).squeeze(0).detach().cpu().numpy()
        pred_y = torch.argmax(output_y, dim=2).squeeze(0).detach().cpu().numpy()
        keypoints_xy = np.stack([pred_x, pred_y], axis=1).astype(np.float32)
        return keypoints_xy, latency


def rescale_events_to_sensor(
    events: np.ndarray,
    input_width: int,
    input_height: int,
    sensor_width: int = 346,
    sensor_height: int = 260,
    input_zero_based: bool = True,
    flip_y: bool = False,
) -> np.ndarray:
    """Scale [x, y, t, p] events to DHP19/EventPointPose one-based pixels.

    Event-driven raw AE logs usually expose top-left image coordinates. Official
    DHP19EPC point-cloud generation stores event y as ``sensor_height - y``.
    Set flip_y=True when converting raw AE logs to match that convention.
    """
    if events.size == 0:
        return events.reshape(0, 4).astype(np.float32)

    scaled = np.asarray(events[:, 0:4], dtype=np.float32).copy()
    if not input_zero_based:
        scaled[:, 0] -= 1.0
        scaled[:, 1] -= 1.0
    scaled[:, 0] = scaled[:, 0] * (float(sensor_width) / float(input_width))
    scaled[:, 1] = scaled[:, 1] * (float(sensor_height) / float(input_height))
    if flip_y:
        scaled[:, 1] = float(sensor_height - 1) - scaled[:, 1]
    scaled[:, 0] = np.clip(np.floor(scaled[:, 0]), 0, sensor_width - 1) + 1.0
    scaled[:, 1] = np.clip(np.floor(scaled[:, 1]), 0, sensor_height - 1) + 1.0
    return scaled


def rescale_keypoints(
    keypoints_xy: np.ndarray,
    from_width: int,
    from_height: int,
    to_width: Optional[int],
    to_height: Optional[int],
) -> np.ndarray:
    if to_width is None or to_height is None:
        return keypoints_xy
    scaled = keypoints_xy.astype(np.float32).copy()
    scaled[:, 0] *= float(to_width) / float(from_width)
    scaled[:, 1] *= float(to_height) / float(from_height)
    return scaled


def _unescape_yarp_string(raw: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(raw):
        c = raw[i]
        if c != 92 or i + 1 >= len(raw):
            out.append(c)
            i += 1
            continue

        nxt = raw[i + 1]
        if nxt == ord("0"):
            out.append(0)
            i += 2
        elif nxt == ord("n"):
            out.append(10)
            i += 2
        elif nxt == ord("r"):
            out.append(13)
            i += 2
        elif nxt == ord("t"):
            out.append(9)
            i += 2
        elif nxt == ord('"'):
            out.append(ord('"'))
            i += 2
        elif nxt == ord("\\"):
            out.append(ord("\\"))
            i += 2
        elif nxt == ord("x") and i + 3 < len(raw):
            try:
                out.append(int(raw[i + 2 : i + 4], 16))
                i += 4
            except ValueError:
                out.append(nxt)
                i += 2
        else:
            out.append(nxt)
            i += 2
    return bytes(out)


def decode_yarp_ae_payload(payload: bytes) -> np.ndarray:
    payload = _unescape_yarp_string(payload)
    usable = len(payload) - (len(payload) % 4)
    if usable <= 0:
        return np.empty((0, 3), dtype=np.float32)
    words = np.frombuffer(payload[:usable], dtype="<u4")
    polarity = (words & 0x1).astype(np.float32)
    x = ((words >> 1) & 0x7FF).astype(np.float32)
    y = ((words >> 12) & 0x3FF).astype(np.float32)
    return np.stack([x, y, polarity], axis=1)


def read_yarp_ae_log(path: Path) -> np.ndarray:
    """Read event-driven/yarp `AE` data.log files into [x, y, t, p]."""
    events: List[np.ndarray] = []
    path = path.expanduser()
    with path.open("rb") as f:
        for line in f:
            if b" AE " not in line:
                continue
            first_quote = line.find(b'"')
            last_quote = line.rfind(b'"')
            if first_quote < 0 or last_quote <= first_quote:
                continue
            header = line[:first_quote].decode("latin1", errors="ignore").strip().split()
            if len(header) < 4:
                continue
            timestamp = float(header[1])
            payload = line[first_quote + 1 : last_quote]
            decoded = decode_yarp_ae_payload(payload)
            if decoded.size == 0:
                continue
            t = np.full((decoded.shape[0], 1), timestamp, dtype=np.float32)
            events.append(np.concatenate([decoded[:, 0:2], t, decoded[:, 2:3]], axis=1))
    if not events:
        return np.empty((0, 4), dtype=np.float32)
    return np.concatenate(events, axis=0).astype(np.float32)


def parse_skeleton_log(path: Path) -> List[Tuple[float, np.ndarray]]:
    """Parse SKLT label logs in the user's 13-joint order.

    Returns a list of (timestamp, keypoints_xy). This is intended for labels or
    visual overlays; it is not used as EventPointPose model input.
    """
    rows: List[Tuple[float, np.ndarray]] = []
    pattern = re.compile(r"SKLT\s*\((.*?)\)")
    with path.expanduser().open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "SKLT" not in line:
                continue
            parts = line.split(maxsplit=3)
            if len(parts) < 3:
                continue
            timestamp = float(parts[1])
            match = pattern.search(line)
            if not match:
                continue
            values = [float(v) for v in match.group(1).split()]
            if len(values) != 26:
                raise ValueError(f"Expected 26 SKLT coordinate values in {path}, got {len(values)}")
            rows.append((timestamp, np.asarray(values, dtype=np.float32).reshape(13, 2)))
    return rows


def write_pose_csv_header(writer: csv.writer) -> None:
    header = ["timestamp", "latency"]
    for joint_idx in range(13):
        header.extend([f"joint{joint_idx}_x", f"joint{joint_idx}_y"])
    writer.writerow(header)


def write_pose_csv_row(writer: csv.writer, timestamp: float, latency: float, keypoints_xy: np.ndarray) -> None:
    row: List[float] = [timestamp, latency]
    for joint_idx in range(13):
        row.extend([float(keypoints_xy[joint_idx, 0]), float(keypoints_xy[joint_idx, 1])])
    writer.writerow([f"{row[0]:.6f}", f"{row[1]:.6f}", *[f"{v:.6f}" for v in row[2:]]])


def iter_time_windows(
    events: np.ndarray,
    frequency: float,
    window_seconds: Optional[float] = None,
) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield sliding time windows ending at the requested inference frequency."""
    if events.size == 0:
        return
    if frequency <= 0:
        raise ValueError("frequency must be > 0")
    stride = 1.0 / frequency
    width = window_seconds if window_seconds is not None else stride
    if width <= 0:
        raise ValueError("window_seconds must be > 0")

    order = np.argsort(events[:, 2])
    events = events[order]
    start_time = float(events[0, 2])
    end_time = float(events[-1, 2])
    t = start_time + stride
    left = 0
    right = 0
    while t <= end_time + 1e-9:
        while left < len(events) and events[left, 2] < t - width:
            left += 1
        while right < len(events) and events[right, 2] <= t:
            right += 1
        yield t, events[left:right]
        t += stride


def iter_last_n_at_fixed_steps(
    events: np.ndarray,
    step_sec: float,
    num_points: int,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield event slices containing the last `num_points` events at fixed steps.

    At each timestamp t (starting from `start_time` or first_event_time + step_sec)
    yield the slice of events with timestamps <= t consisting of the last
    `num_points` events (by event order). If fewer than `num_points` events are
    available, the slice will contain all events up to t (the estimator will
    handle sampling/duplication as needed).
    """
    if events.size == 0:
        return
    if step_sec <= 0:
        raise ValueError("step_sec must be > 0")

    order = np.argsort(events[:, 2])
    events = events[order]
    first_ts = float(events[0, 2])
    last_ts = float(events[-1, 2])
    if start_time is None:
        t = first_ts + step_sec
    else:
        t = start_time
    if end_time is None:
        end_time = last_ts

    # Use searchsorted for efficient index lookup
    times = events[:, 2]
    while t <= end_time + 1e-9:
        right = int(np.searchsorted(times, t, side="right"))
        if right > 0:
            left = max(0, right - int(num_points))
            yield t, events[left:right]
        t += step_sec


def _timestamp_to_seconds(timestamp_value: float, origin: float) -> float:
    value = timestamp_value - origin
    if abs(value) > 1000.0:
        return value * 1e-6
    return value


def iter_official_npy_frames(
    data_files: Sequence[Path],
    label: str,
    fallback_frequency: float,
) -> Iterator[Tuple[float, np.ndarray, Path]]:
    origin: Optional[float] = None
    for frame_idx, data_file in enumerate(data_files):
        data = np.load(str(data_file))
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] < 4:
            continue
        if label == "last":
            data = data[:, [0, 2, 1, 3]]
        if origin is None:
            origin = float(np.min(data[:, 2])) if data.size else 0.0
        if data.size:
            ts = _timestamp_to_seconds(float(np.max(data[:, 2])), origin)
        else:
            ts = frame_idx / max(fallback_frequency, 1.0)
        yield ts, data[:, 0:4], data_file


def _normalise_h5_dvs(dvs: np.ndarray) -> np.ndarray:
    if dvs.ndim != 3:
        raise ValueError(f"Expected DHP19EPC /DVS to be 3D, got shape {dvs.shape}")
    if dvs.shape[-1] == 4:
        return dvs
    if dvs.shape[0] == 4:
        return np.moveaxis(dvs, 0, -1)
    raise ValueError(f"Cannot infer DHP19EPC /DVS axis order from shape {dvs.shape}")


def iter_official_h5_frames(
    h5_files: Sequence[Path],
    label: str,
    camera: Optional[int],
    fallback_frequency: float,
) -> Iterator[Tuple[float, np.ndarray, Path]]:
    import h5py

    origin: Optional[float] = None
    global_frame_idx = 0
    for h5_file in h5_files:
        with h5py.File(str(h5_file), "r") as h5:
            if "DVS" not in h5:
                continue
            dvs = _normalise_h5_dvs(h5["DVS"][...])
            cam_counts = h5["CamPointNum"][...] if "CamPointNum" in h5 else None
            if cam_counts is not None and cam_counts.shape[-1] == dvs.shape[0]:
                cam_counts = cam_counts.T
            for frame_idx in range(dvs.shape[0]):
                frame = dvs[frame_idx]
                if cam_counts is not None and camera is not None:
                    counts = cam_counts[frame_idx].astype(int)
                    start = int(np.sum(counts[:camera]))
                    stop = start + int(counts[camera])
                    frame = frame[start:stop]
                frame = np.asarray(frame, dtype=np.float32)
                if label == "last":
                    frame = frame[:, [0, 2, 1, 3]]
                if origin is None:
                    origin = float(np.min(frame[:, 2])) if frame.size else 0.0
                if frame.size:
                    ts = _timestamp_to_seconds(float(np.max(frame[:, 2])), origin)
                else:
                    ts = global_frame_idx / max(fallback_frequency, 1.0)
                global_frame_idx += 1
                yield ts, frame[:, 0:4], h5_file


def discover_event_inputs(
    dataset_path: Path,
    sequence: Optional[str] = None,
    camera: Optional[int] = None,
) -> Tuple[str, List[Path]]:
    """Discover supported event inputs.

    Returns (kind, paths), where kind is one of: raw_log, npy, h5, skeleton_log.
    """
    dataset_path = dataset_path.expanduser()
    if dataset_path.is_file():
        suffix = dataset_path.suffix.lower()
        if suffix == ".npy":
            return "npy", [dataset_path]
        if suffix in (".h5", ".hdf5"):
            return "h5", [dataset_path]
        if dataset_path.name == "data.log":
            with dataset_path.open("rb") as f:
                sample = f.read(4096).decode("latin1", errors="ignore")
            if " SKLT " in sample:
                return "skeleton_log", [dataset_path]
            return "raw_log", [dataset_path]
        raise ValueError(f"Unsupported dataset file: {dataset_path}")

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"dataset_path does not exist: {dataset_path}")

    if (dataset_path / "data.log").exists():
        return discover_event_inputs(dataset_path / "data.log", sequence=sequence, camera=camera)

    data_dir = dataset_path / "data"
    search_root = data_dir if data_dir.exists() else dataset_path
    pattern = f"*{sequence}*" if sequence else "*"

    npy_files = sorted(search_root.glob(f"{pattern}.npy"))
    if camera is not None:
        npy_files = [p for p in npy_files if f"_cam{camera}" in p.name]
    if npy_files:
        return "npy", npy_files

    h5_files = sorted(search_root.glob(f"{pattern}.h5")) + sorted(search_root.glob(f"{pattern}.hdf5"))
    if camera is not None:
        h5_files = [p for p in h5_files if (f"_cam{camera}" in p.name or "_cam" not in p.name)]
    if h5_files:
        return "h5", h5_files

    raw_logs = sorted(dataset_path.glob("**/data.log"))
    raw_logs = [p for p in raw_logs if "skeleton" not in p.parent.name.lower()]
    if sequence:
        raw_logs = [p for p in raw_logs if sequence in str(p)]
    if raw_logs:
        return "raw_log", raw_logs

    skeleton_logs = sorted(dataset_path.glob("**/data.log"))
    skeleton_logs = [p for p in skeleton_logs if "skeleton" in p.parent.name.lower()]
    if sequence:
        skeleton_logs = [p for p in skeleton_logs if sequence in str(p)]
    if skeleton_logs:
        return "skeleton_log", skeleton_logs

    raise FileNotFoundError(f"No supported EventPointPose inputs found under {dataset_path}")
