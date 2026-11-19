"""Add genuine between-recording standard-deviation error bars to existing paper plots.

The existing paper_figures renderer recognizes <metric>_low/high interval endpoints.
These endpoints represent mean +/- 1 sample SD (ddof=1), NOT a confidence interval.
For pooled accuracy the center is the pooled joint-weighted estimate, while the
sample SD is computed across independent recording-level estimates.
"""
import numpy as np
import pandas as pd
import warnings

STD_COLUMNS = {
    'mpjpe_px': 'mpjpe_sequence_sd_px',
    'pck_percent': 'pck_sequence_sd_percent',
    'energy_j_per_sequence': 'energy_sequence_sd_j',
    'energy_j_per_input_s': 'energy_sequence_sd_j_per_s',
    'rtf': 'rtf_sequence_sd',
}


def with_standard_deviation(frame, enabled=True, metrics=None, log_metrics=()):
    """Return a copy with interval columns used by paper_figures._bars.

    Only independent recording-level sample SD is used in plotted bars.
    Within-sequence energy-repeat SD is retained in source tables, never pooled
    together with between-sequence variation.
    """
    result = frame.copy()
    if not enabled:
        return result
    metrics = tuple(metrics or STD_COLUMNS.keys())
    log_metrics = set(log_metrics)
    for metric in metrics:
        sd_name = STD_COLUMNS[metric]
        if metric not in result or sd_name not in result:
            raise KeyError('Cannot show SD: missing {} or {}'.format(metric, sd_name))
        center = pd.to_numeric(result[metric], errors='raise').astype(float)
        sd = pd.to_numeric(result[sd_name], errors='coerce').astype(float)
        if (sd.dropna() < 0).any():
            raise ValueError('Negative standard deviation in {}'.format(sd_name))
        if 'n_sequences' in result:
            sd = sd.where(pd.to_numeric(result['n_sequences']) >= 2)
        result[metric + '_low'] = center - sd
        result[metric + '_high'] = center + sd
        # Negative or zero lower endpoints are not representable on a logarithmic axis.
        # Do not move or clip endpoints: omit only that SD bar, retaining the center.
        if metric in log_metrics:
            invalid_log = sd.notna() & (result[metric + '_low'] <= 0)
            if invalid_log.any():
                warnings.warn(
                    '{} of {} SD intervals for {} cross zero; these error bars are '
                    'omitted on the logarithmic axis, while central values are unchanged.'
                    .format(int(invalid_log.sum()), len(result), metric), RuntimeWarning)
                result.loc[invalid_log, [metric + '_low', metric + '_high']] = np.nan
        # Missing SD (for a single recording) means no bar, not an artificial zero.
        result.loc[sd.isna(), [metric + '_low', metric + '_high']] = np.nan
    return result


def check_log_scale_compatibility(frame, metrics, axis_names):
    """Require a linear axis when a mean-SD endpoint is zero or negative."""
    for metric, scale in zip(metrics, axis_names):
        if scale != 'log':
            continue
        name = metric + '_low'
        if name in frame and (pd.to_numeric(frame[name], errors='coerce').dropna() <= 0).any():
            raise ValueError('The SD interval for {} extends to zero/negative values. '
                             'Change that axis to linear; never clip the SD bar.'.format(metric))
