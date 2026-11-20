# Reference validation notebooks

These four specialist notebooks are retained as **diagnostics**, not as the
canonical Experiment A runner or paper-metric computation. They may contain
historical hard-coded paths and were developed against older experimental
protocols. Use `../accuracy_metrics.py` for new raw-CSV evaluation and the
common `../../visualization/plot_accuracy.py` for paper figures.

- `ExperimentA_Method_Comparison_SELECTABLE_MODELS.ipynb`: legacy method-level checks.
- `ExperimentA_Joint_MPJPE_Comparison.ipynb`: joint-level diagnostics to preserve.
- `MoveIIT_ExpA_2DGT_Analysis.ipynb`: invalid-GT masking regression and diagnosis.
- `MoveIIT_ExpA_EventPointPose_3Methods_Analysis.ipynb`: three-method native timestamp pairing and diagnostics.

**Before publishing Move-IIT-HPE accuracy**, validate the anatomical joint
mapping and GT-prediction time offset. These remain provisional.
