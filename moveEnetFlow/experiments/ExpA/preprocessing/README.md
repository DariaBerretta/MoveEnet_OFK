# Optional preprocessing utilities

`eventpointpose_generate_hpm.sh` is retained for DHP19 hot-pixel mask
preparation/diagnostics, but the current GT-paired DHP19 EventPointPose pipeline
uses inputs labelled `already_filtered`. Do not apply a second round of
hot-pixel/background filtering to those inputs.

Its generated files go to the common `HPE_RESULTS_ROOT` tree by default.
