#!/usr/bin/env bash
exec env MOVENET_PROFILE=expB "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/shared/run_movenet_grid.sh" "$@"
