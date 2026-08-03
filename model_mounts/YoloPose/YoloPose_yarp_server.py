#!/usr/bin/env python3
"""
YARP sidecar for Ultralytics YOLO pose models.

Input:  YARP ImageMono on /YoloPose/img:i
Output: YARP Bottle on /YoloPose/sklt:o

The Bottle format matches hpe-core::extractSkeletonFromYARP():
  bottle[0] = timestamp placeholder
  bottle[1] = list of 39 floats:
      26 floats: 13 joints as x0,y0,x1,y1,...,x12,y12
      13 floats: confidence for each joint

Internal output joint order follows hpe-core::skeleton13:
  0 head/nose
  1 right_shoulder
  2 left_shoulder
  3 right_elbow
  4 left_elbow
  5 left_hip
  6 right_hip
  7 right_wrist
  8 left_wrist
  9 right_knee
  10 left_knee
  11 right_ankle
  12 left_ankle
"""

import argparse
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import yarp
except ImportError as exc:
    raise SystemExit(
        "Could not import yarp. Install/enable the YARP Python bindings in this environment."
    ) from exc

# Map hpe-core skeleton13 order -> YOLO/COCO17 keypoint index.
# YOLO/COCO17: 0 nose, 5 L shoulder, 6 R shoulder, 7 L elbow, 8 R elbow,
# 9 L wrist, 10 R wrist, 11 L hip, 12 R hip, 13 L knee, 14 R knee,
# 15 L ankle, 16 R ankle.
HPE_TO_YOLO = [
    0,   # head/nose
    6,   # right_shoulder
    5,   # left_shoulder
    8,   # right_elbow
    7,   # left_elbow
    11,  # left_hip
    12,  # right_hip
    10,  # right_wrist
    9,   # left_wrist
    14,  # right_knee
    13,  # left_knee
    16,  # right_ankle
    15,  # left_ankle
]


def choose_person(result):
    """Return index of the person to use. Prefer the largest bounding box."""
    if result.keypoints is None or result.keypoints.data is None:
        return None

    kpts = result.keypoints.data
    if len(kpts) == 0:
        return None

    if result.boxes is not None and result.boxes.xyxy is not None and len(result.boxes.xyxy) == len(kpts):
        xyxy = result.boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        return int(np.argmax(areas))

    return 0


def make_empty_pose():
    xy = np.zeros((13, 2), dtype=np.float32)
    conf = np.zeros((13,), dtype=np.float32)
    return xy, conf


def extract_hpe_pose(result):
    """Extract one person and convert YOLO COCO17 keypoints to hpe-core skeleton13 order."""
    person_idx = choose_person(result)
    if person_idx is None:
        return make_empty_pose()

    kpts = result.keypoints.data[person_idx].cpu().numpy()  # shape: [17, 3] = x, y, conf/visibility
    xy = np.zeros((13, 2), dtype=np.float32)
    conf = np.zeros((13,), dtype=np.float32)

    for hpe_i, yolo_i in enumerate(HPE_TO_YOLO):
        xy[hpe_i, 0] = kpts[yolo_i, 0]
        xy[hpe_i, 1] = kpts[yolo_i, 1]
        conf[hpe_i] = kpts[yolo_i, 2] if kpts.shape[1] >= 3 else 1.0

    return xy, conf


def write_pose(out_port, xy, conf):
    bottle = out_port.prepare()
    bottle.clear()
    bottle.addFloat64(time.time())

    payload = bottle.addList()
    for i in range(13):
        payload.addFloat64(float(xy[i, 0]))
        payload.addFloat64(float(xy[i, 1]))
    for i in range(13):
        payload.addFloat64(float(conf[i]))

    out_port.write()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to yolo26n-pose.pt")
    parser.add_argument("--w", type=int, default=640, help="Input image width")
    parser.add_argument("--h", type=int, default=480, help="Input image height")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--device", default="", help="Ultralytics device, e.g. '0', 'cpu', or empty for auto")
    parser.add_argument("--img_port", default="/YoloPose/img:i")
    parser.add_argument("--sklt_port", default="/YoloPose/sklt:o")
    args = parser.parse_args()

    yarp.Network.init()
    if not yarp.Network.checkNetwork(2.0):
        raise SystemExit("YARP server is not available. Start yarpserver first.")

    model = YOLO(args.model)

    in_port = yarp.Port()
    out_port = yarp.BufferedPortBottle()

    if not in_port.open(args.img_port):
        raise SystemExit(f"Could not open input image port: {args.img_port}")
    if not out_port.open(args.sklt_port):
        in_port.close()
        raise SystemExit(f"Could not open output skeleton port: {args.sklt_port}")

    img_array = np.zeros((args.h, args.w), dtype=np.uint8)
    yarp_img = yarp.ImageMono()
    yarp_img.resize(args.w, args.h)
    yarp_img.setExternal(img_array.data, args.w, args.h)

    predict_kwargs = {"imgsz": args.imgsz, "conf": args.conf, "verbose": False}
    if args.device:
        predict_kwargs["device"] = args.device

    try:
        while True:
            ok = in_port.read(yarp_img)
            if not ok:
                continue

            gray = img_array.copy()
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            result = model.predict(bgr, **predict_kwargs)[0]
            xy, kconf = extract_hpe_pose(result)
            write_pose(out_port, xy, kconf)

    except KeyboardInterrupt:
        pass
    finally:
        in_port.close()
        out_port.close()
        yarp.Network.fini()


if __name__ == "__main__":
    sys.exit(main())
