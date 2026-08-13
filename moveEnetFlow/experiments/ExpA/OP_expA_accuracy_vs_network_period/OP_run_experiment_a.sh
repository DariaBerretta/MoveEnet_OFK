#!/usr/bin/env bash
exec env POSE_METHOD=openpose "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/shared/run_rgb_pose_experiment.sh" "$@"
