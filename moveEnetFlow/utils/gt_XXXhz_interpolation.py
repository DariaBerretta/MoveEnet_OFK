"""
Interpolate skeleton logs from 50 Hz to 200 Hz and mirror the folder structure.

Usage
-----
python gt_XXXhz_interpolation.py \
	--dataset-root /path/to/raw \
	--source-hz 50 \
	--target-hz 200 \
	[--overwrite]

What it does
------------
- Scans all subfolders under --dataset-root for a folder named "ch0GT50Hzskeleton".
- Reads its data.log lines formatted as: `id ts SKLT (<26 floats>) head_size torso_size`.
- Interpolates timestamps and values to the target frequency.
- Writes a sibling folder named "ch0GT{target}Hzskeleton" with an updated data.log
  (and copies info.log if present).

Assumptions
-----------
- Timestamps in the log are seconds and strictly increasing.
- Skeletons have 13 joints (26 flattened coordinates), matching the rest of the codebase.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import re


LOG_PATTERN = re.compile(r"(\d+)\s+([\d\.e\-]+)\s+SKLT\s+\((.*?)\)\s+([\-\d\.]+)\s+([\-\d\.]+)")


@dataclass
class SkeletonSample:
	idx: int
	ts: float
	keypoints: np.ndarray  # shape (26,)
	head_size: float
	torso_size: float


def parse_log(log_path: Path) -> List[SkeletonSample]:
	if not log_path.exists():
		raise FileNotFoundError(f"Missing log: {log_path}")

	samples: List[SkeletonSample] = []
	with log_path.open("r") as f:
		for line in f:
			match = LOG_PATTERN.match(line.strip())
			if not match:
				continue
			idx_str, ts_str, pts_str, head_str, torso_str = match.groups()
			pts = np.array([float(p) for p in pts_str.split() if p], dtype=np.float32)
			samples.append(
				SkeletonSample(
					idx=int(idx_str),
					ts=float(ts_str),
					keypoints=pts,
					head_size=float(head_str),
					torso_size=float(torso_str),
				)
			)

	if not samples:
		raise ValueError(f"No valid lines parsed from {log_path}")

	return samples


def interpolate_samples(
	samples: List[SkeletonSample], target_hz: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	# shapes: ts -> (N,), keypoints -> (N, 26)
	ts_src = np.array([s.ts for s in samples], dtype=np.float64)
	kps_src = np.stack([s.keypoints for s in samples]).astype(np.float64)
	head_src = np.array([s.head_size for s in samples], dtype=np.float64)
	torso_src = np.array([s.torso_size for s in samples], dtype=np.float64)

	if ts_src.ndim != 1 or np.any(np.diff(ts_src) <= 0):
		raise ValueError("Timestamps must be strictly increasing for interpolation")

	dt = 1.0 / target_hz
	t_start, t_end = ts_src[0], ts_src[-1]
	ts_tgt = np.arange(t_start, t_end + 1e-9, dt)

	kps_tgt = np.vstack(
		[np.interp(ts_tgt, ts_src, kps_src[:, i]) for i in range(kps_src.shape[1])]
	).T
	head_tgt = np.interp(ts_tgt, ts_src, head_src)
	torso_tgt = np.interp(ts_tgt, ts_src, torso_src)

	return ts_tgt, kps_tgt, head_tgt, torso_tgt


def write_log(
	out_path: Path,
	ts: np.ndarray,
	kps: np.ndarray,
	head: np.ndarray,
	torso: np.ndarray,
	start_idx: int,
) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w") as f:
		for i, (t, kp_row, h, to) in enumerate(zip(ts, kps, head, torso)):
			coords = " ".join(str(int(round(v))) for v in kp_row)
			line = f"{start_idx + i} {t:.6f} SKLT ({coords}) {h:.6f} {to:.6f}\n"
			f.write(line)


def process_sample(
	sklt_folder: Path, target_hz: float, overwrite: bool, source_tag: str, target_tag: str
) -> None:
	log_path = sklt_folder / "data.log"
	samples = parse_log(log_path)

	ts, kps, head, torso = interpolate_samples(samples, target_hz=target_hz)

	out_folder = sklt_folder.parent / target_tag
	out_log = out_folder / "data.log"

	if out_log.exists() and not overwrite:
		print(f"Skip existing: {out_log}")
		return

	write_log(out_log, ts, kps, head, torso, start_idx=samples[0].idx)

	info_src = sklt_folder / "info.log"
	info_dst = out_folder / "info.log"
	if info_src.exists():
		shutil.copy2(info_src, info_dst)

	print(f"Wrote {out_log} with {len(ts)} frames from {source_tag} -> {target_tag}")


def find_gt50_folders(dataset_root: Path, source_tag: str) -> Iterable[Path]:
	return dataset_root.rglob(source_tag)


def main(argv: List[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description="Interpolate skeleton GT logs to a higher frequency.")
	parser.add_argument("--dataset-root", required=True, type=Path, help="Root folder containing raw samples")
	parser.add_argument("--source-hz", type=float, default=50.0, help="Input GT frequency")
	parser.add_argument("--target-hz", type=float, default=200.0, help="Output GT frequency")
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target logs")

	args = parser.parse_args(argv)

	# Source tag name if DATASET is eh36m 
	source_tag = f"ch0GT{int(args.source_hz)}Hzskeleton"
	target_tag = f"ch0GT{int(args.target_hz)}Hzskeleton"

	# Source tag name if DATASET is dhp19
	# source_tag = f"ch3skeleton"
	# target_tag = f"ch3GT{int(args.target_hz)}Hzskeleton"

	folders = sorted(find_gt50_folders(args.dataset_root, source_tag))
	if not folders:
		print(f"No folders named {source_tag} found under {args.dataset_root}")
		sys.exit(1)

	for sklt_folder in folders:
		try:
			process_sample(
				sklt_folder=sklt_folder,
				target_hz=args.target_hz,
				overwrite=args.overwrite,
				source_tag=source_tag,
				target_tag=target_tag,
			)
		except Exception as exc:  # keep processing the rest
			print(f"Failed on {sklt_folder}: {exc}")


if __name__ == "__main__":
	main()
