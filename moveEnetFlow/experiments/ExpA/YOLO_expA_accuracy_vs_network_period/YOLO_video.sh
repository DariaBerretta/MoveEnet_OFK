#!/usr/bin/env bash
exec env POSE_METHOD=yolo "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/shared/run_rgb_pose_video.sh" "$@"
