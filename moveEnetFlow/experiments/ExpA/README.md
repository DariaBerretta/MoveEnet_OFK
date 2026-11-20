# Experiment A — Accuracy as a function of network inference frequency

This directory contains the **new** unified Experiment A implementation. It does
not overwrite or delete historical scripts, notebooks, or existing results. All
new results are stored outside Git under `HPE_RESULTS_ROOT` (default:
`/data/MoveEnet_OFK_results/experiment_cleanup`).

## Fixed scientific protocol

| Dataset | Methods | KF `pu, muD, muV` | ROI | Effective `of_gain` |
| --- | --- | --- | --- | --- |
| eH36M | MoveEnetOFK, MoveEnet-only, YOLOPose, OpenPose | 0.77, 0.06, 0.97 | 20 | 1.0 |
| DHP19 | MoveEnetOFK, MoveEnet-only, EventPointPose | 0.36, 0.01, 0.01 | **10** | 1.0 |
| Move-IIT-HPE | MoveEnetOFK, MoveEnet-only, EventPointPose | 0.77, 0.06, 0.97 | 35 | 1.0 |

- Output target is 200 Hz for all methods. MoveEnetOFK optical flow: 200 Hz.
- All event-based methods use network rates 2, 5, 10, 20, 50, 100, 200 Hz.
- RGB OpenPose/YOLOPose use network rates 2, 5, 10, 20, 50 Hz.
- `use_lc=true` for the MoveEnetOFK Kalman filter.
- Execute every GT-paired, non-excluded testing sequence. Missing GT and known
  invalid eH36M recordings are logged to `excluded.csv`, not silently skipped.
- No automatic temporal shift of GT is applied. Move-IIT-HPE marker mapping and
  time offset still require independent validation before paper claims.

## Installation and prerequisites

The Python runner needs Python 3.8+ and the existing compiled offline binaries.
The analysis requires pandas and numpy and imports the existing canonical
`moveEnetFlow/analysis/paper_figures/raw_metrics.py` implementation. The paper
figures require Matplotlib and use `analysis/paper_figures/style.json`.

The new runner passes `--of_gain 1.0` to `moveEnetOFK_offline`; **rebuild this C++
binary after installing the accompanying source patch**. The binary previously
cancelled the hpe-core 1.4 internal multiplier with `v/=1.4`. The patched version
continues cancelling it and then applies the explicit `of_gain` (1.0 by default).
Do not use an older compiled binary that ignores `--of_gain`.

Ensure the correct Docker volume mount, YARP server, checkpoints, model sidecars,
and sources are available. Runtime environment and actual completed neural rate
must be checked experimentally; the rate flags are *requested* rates.

The `.yaml` files contain JSON-compatible YAML syntax and are read using the
Python standard library, avoiding an additional PyYAML dependency.

## Launch from the repository / Docker workspace

```bash
cd /workspace/moveEnetFlow
python3 experiments/ExpA/scripts/run_experiment.py --dataset eh36m --dry-run
python3 experiments/ExpA/scripts/run_experiment.py --dataset dhp19 --dry-run
python3 experiments/ExpA/scripts/run_experiment.py --dataset moveiit --dry-run
```

First run **one recording** to verify model startup, sensor geometry, time origin,
and raw CSV completeness; do not launch the entire sweep blindly:

```bash
python3 experiments/ExpA/scripts/run_experiment.py \
  --dataset dhp19 --method moveenetofk --network-hz 5 --limit 1 \
  --run-id smoke_dhp19_ofk_5hz
```

Then run a full dataset, with a different run ID:

```bash
python3 experiments/ExpA/scripts/run_experiment.py \
  --dataset dhp19 --run-id expa_dhp19_v1 --keep-going
```

Resume only after inspecting failures, without recomputing verified completed
trials:

```bash
python3 experiments/ExpA/scripts/run_experiment.py \
  --dataset dhp19 --run-id expa_dhp19_v1 --keep-going --resume
```

**Avoid concurrent runs** of the MoveEnet binaries unless YARP and server process
management have been verified. Existing MoveEnet code uses process-wide cleanup
that is not concurrency-safe.

## Output tree

```text
/data/MoveEnet_OFK_results/experiment_cleanup/
  ExpA/
    expa_dhp19_v1/
      config.yaml              # exact resolved dataset config and selected methods/rates
      metadata.json            # git SHA, model/binary SHA256, device and runtime metadata
      manifest.csv             # one row per recording / method / rate, command and status
      excluded.csv             # input sequences without evaluable GT and reasons
      audit.jsonl              # append-only per-trial status changes
      predictions/dhp19/moveenetofk/nn_5Hz/<sequence>__<hash>.csv
      logs/dhp19/moveenetofk/nn_5Hz/<sequence>__<hash>.log
      telemetry/...
      metrics/summary.csv
      metrics/per_sequence.csv
      metrics/rejected.csv
      metrics/protocol.json
      figures/...
```

Output filenames contain both a human-readable input path and a short hash to
avoid collisions. Nothing in the new experiment writes under legacy result roots.

## Validate, score, plot

```bash
python3 experiments/ExpA/scripts/validate_results.py \
  --run-dir /data/MoveEnet_OFK_results/experiment_cleanup/ExpA/expa_dhp19_v1
python3 experiments/ExpA/analysis/accuracy_metrics.py \
  --run-dir /data/MoveEnet_OFK_results/experiment_cleanup/ExpA/expa_dhp19_v1
python3 experiments/ExpA/visualization/plot_accuracy.py \
  --run-dir /data/MoveEnet_OFK_results/experiment_cleanup/ExpA/expa_dhp19_v1
```

Primary metrics: MPJPE (pixels), and PCK@0.2 normalized by GT joints 1 and 5.
GT is interpolated only within valid consecutive observations with gap <=50ms;
no extrapolation or implicit GT offset. Aggregation uses pooled valid joint
observations, while the inter-recording standard deviation is kept separately.
Predictions are never interpolated to a fictitious network timestamp.

**Important:** the analysis is not permitted to certify Move-IIT-HPE absolute
accuracy until GT marker mapping and time synchronization are independently
validated. It records provisional provenance and can generate exploratory plots.

## Status / limitations

- Dataset-specific input and GT paths are based on existing experiment conventions;
  verify real mount layouts using `--dry-run` before full runs.
- Existing offline binaries, YARP services, hpe-core, and GPU environment cannot be
  executed by the installer. End-to-end runtime tests must be run on the workstation.
- `output_hz=200` is the requested sampling cadence. Actual output timestamp
  regularity and achieved inference frequency require measurement from the logs.
- Older ExpA notebooks remain in place for comparison; this code does not remove
  them or migrate older results.
- The `--limit` option is for development and should not be used for paper figures.

The installer also updates `.gitignore` to explicitly track these three YAML
configuration files and this README. The original `.gitignore` otherwise ignores
all experiment files except shell scripts, Python, and notebooks.
