# Experiments

Experiment A is now maintained under [`ExpA/`](ExpA/README.md), with one
runner, three checked configurations, a shared scorer and common plotting.
Do not use the deleted single-method ExpA wrapper scripts.

The default results root is `/data/MoveEnet_OFK_results/experiment_cleanup`
(overridable with `HPE_RESULTS_ROOT`). Existing outputs in historical
directories are untouched and must not be mixed into new-run results.

## Other experiments

- **ExpB:** `expB_accuracy_vs_flow_period/run_experiment_b.sh`
- **Latency, energy, final benchmarks:** retain their existing runners pending dedicated audits.

Shared `run_movenet_grid.sh` is retained for ExpB; do not delete it.
