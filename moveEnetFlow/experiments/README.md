# Experiments

The public scripts stay beside each experiment; they are compatibility wrappers. Shared implementations are in [`shared/`](shared/): do not copy a runner to start a new grid.

| Experiment | Run | Complete notebook |
| --- | --- | --- |
| ExpA, MoveNet + OFK | `ExpA/expA_accuracy_vs_network_period/run_experiment_a.sh` | `ExpA/expA_accuracy_vs_network_period/notebooks/ExperimentA_MPJPE_PCK_vs_network_period.ipynb` |
| ExpA, OpenPose | `ExpA/OP_expA_accuracy_vs_network_period/OP_run_experiment_a.sh` | `ExpA/OP_expA_accuracy_vs_network_period/notebooks/OP_ExperimentA_MPJPE_PCK_vs_network_period.ipynb` |
| ExpA, YOLO Pose | `ExpA/YOLO_expA_accuracy_vs_network_period/YOLO_run_experiment_a.sh` | `ExpA/YOLO_expA_accuracy_vs_network_period/notebooks/YOLOPose_MPJPE_PCK_vs_network_period.ipynb` |
| ExpA, EventPointPose | `ExpA/EPP_expA_accuracy_vs_network_period/EPP_run_expA.sh` | `ExpA/EPP_expA_accuracy_vs_network_period/Notebooks/EPP_MPJPE_PCK_vs_network_period.ipynb` |
| ExpB, MoveNet + OFK | `expB_accuracy_vs_flow_period/run_experiment_b.sh` | `expB_accuracy_vs_flow_period/notebooks/experimentB_flow_period_analysis_SHARED_HEATMAPS.ipynb` |
| ExpA, method comparison | — | `ExpA/Comparison/ExperimentA_Method_Comparison_SELECTABLE_MODELS.ipynb` |

`ExperimentA_Joint_MPJPE_Comparison.ipynb` is retained because it adds joint-level metrics; it is not a duplicate of the selectable method comparison.

Each retained notebook is the complete, current analysis for its experiment. Historical and single-metric copies were removed.
All experiment runners write their CSV files below
`/data/MoveEnet_OFK_results` by default, using the dataset folders expected by the analysis notebooks
layout. Set `MOVENET_RESULTS_ROOT` to change the common root, or use a runner's
`--raw_dir`, `--log_dir`, `--output_dir`, or `--mask_root` option for a
one-off override.

For EventPointPose, `EPP_video.sh` is the current rolling-FIFO video workflow. `EPP_video_no_rolling.sh` is intentionally separate for the no-rolling ablation; `EPP_video_v2.sh` is a compatibility alias.
