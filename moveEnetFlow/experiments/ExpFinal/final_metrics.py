"""Accuracy semantics ported from ExperimentA_Method_Comparison_SELECTABLE_MODELS.

Predictions are NOT interpolated. GT is linearly interpolated at each actual
prediction timestamp, with the same endpoint padding used by the repository.
PCK@0.2 and PCK@0.4 use GT shoulder_right (1) -> hip_left (5).
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np

from final_common import AuditError, require, sha256_file

JOINT_NAMES = ('head','shoulder_right','shoulder_left','elbow_right','elbow_left',
               'hip_left','hip_right','wrist_right','wrist_left','knee_right',
               'knee_left','ankle_right','ankle_left')
METRIC_PROTOCOL = 'ExpA_GT_to_prediction_linear_endpoint_hold_torso_1_5_finite_rows_v1'

def import_gt_parser(hpecore_root='/usr/local/src/hpe-core'):
    root = Path(hpecore_root).resolve()
    require(root.is_dir(), 'hpe-core non trovato: ' + str(root))
    sys.path.insert(0, str(root))
    from datasets.utils import constants, parsing
    names = tuple(constants.HPECoreSkeleton.KEYPOINTS_MAP.keys())
    require(names == JOINT_NAMES, 'Ordine dei 13 joint diverso dal notebook ExpA: ' + repr(names))
    require(root in Path(parsing.__file__).resolve().parents,
            'Modulo datasets errato, non proviene da hpe-core: ' + parsing.__file__)
    return parsing

def load_gt(gt_path, parser):
    try:
        gt = parser.import_yarp_skeleton_data(Path(gt_path), multi_channel=False)
    except SystemExit as exc:
        raise AuditError('Parser hpe-core ha interrotto la lettura GT: ' + str(gt_path)) from exc
    ts = np.asarray(gt['ts'], dtype=float)
    require(ts.ndim == 1 and len(ts) > 1, 'GT vuota/non valida: ' + str(gt_path))
    require(np.isfinite(ts).all() and np.all(np.diff(ts) >= 0) and ts[0] >= 0,
            'Timestamp GT non ordinati/non finiti/negativi: ' + str(gt_path))
    xy = np.stack([np.asarray(gt[k], dtype=float) for k in JOINT_NAMES], axis=1)
    require(xy.shape == (len(ts), 13, 2), 'Forma GT diversa da N x 13 x 2')
    return ts, xy

def interpolate_gt(ts_gt, joints_gt, ts_pred):
    padded_ts = np.concatenate(([0.0], ts_gt, [ts_gt[-1] + 1.0]))
    padded_xy = np.concatenate((joints_gt[:1], joints_gt, joints_gt[-1:]), axis=0)
    result = np.empty((len(ts_pred), 13, 2), dtype=np.float64)
    for j in range(13):
        for k in range(2):
            result[:, j, k] = np.interp(ts_pred, padded_ts, padded_xy[:, j, k])
    return result

def metric_arrays(ts_pred, joints_pred, ts_gt, joints_gt):
    ts_pred = np.asarray(ts_pred, dtype=float)
    joints_pred = np.asarray(joints_pred, dtype=float)
    require(joints_pred.shape == (len(ts_pred), 13, 2), 'Forma predizioni errata')
    gt_interp = interpolate_gt(ts_gt, joints_gt, ts_pred)
    valid = (np.isfinite(ts_pred) & np.isfinite(joints_pred).all(axis=(1,2))
             & np.isfinite(gt_interp).all(axis=(1,2)))
    require(valid.any(), 'Nessuna riga predizione/GT finita')
    p, g, ts = joints_pred[valid], gt_interp[valid], ts_pred[valid]
    error = np.linalg.norm(p - g, axis=2)
    ref = np.linalg.norm(g[:,5,:] - g[:,1,:], axis=1)
    ref = np.maximum(ref, 1e-6)
    row = {'metric_protocol': METRIC_PROTOCOL,
           'n_output_rows': int(len(ts_pred)), 'n_samples': int(len(ts)),
           'n_nonfinite_rows_excluded': int((~valid).sum()),
           'n_points': int(error.size), 'sum_err_px': float(error.sum()),
           'mpjpe_px': float(error.mean()), 'mpjpe': float(error.mean()),
           'prediction_first_ts_s': float(ts[0]), 'prediction_last_ts_s': float(ts[-1]),
           'n_rows_outside_gt_span': int(((ts < ts_gt[0]) | (ts > ts_gt[-1])).sum()),
           'n_degenerate_torso_rows': int((ref <= 1e-6).sum())}
    for alpha in (.2,.4):
        # float32 matches the repository's calculate_pck_binary().
        hits = (error <= alpha * ref[:,None]).astype(np.float32)
        row['pck_{}'.format(alpha)] = float(hits.mean() * 100.0)
        row['pck_{}_correct_points'.format(alpha)] = int(hits.sum())
    row['pck'] = row['pck_0.2']  # percent, never a fraction in this package
    return row

def evaluate_prediction(csv_path, gt_path, parser, output_period_s=.005, input_span_s=None):
    csv_path, gt_path = Path(csv_path), Path(gt_path)
    require(csv_path.is_file() and csv_path.stat().st_size > 0, 'Predizioni mancanti: ' + str(csv_path))
    with csv_path.open(encoding='utf-8') as f:
        header = f.readline().strip().split(',')
    expected = ['timestamp','latency'] + [x for j in range(13) for x in ('joint{}_x'.format(j),'joint{}_y'.format(j))]
    require(header[:28] == expected, 'Header delle predizioni inatteso: ' + str(csv_path))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        a = np.loadtxt(str(csv_path), delimiter=',', skiprows=1)
    require(a.size > 0, 'CSV header-only: nessuna posa. Non viene escluso silenziosamente: ' + str(csv_path))
    a = np.atleast_2d(a)
    require(a.shape[1] >= 28, 'CSV con meno di 28 colonne')
    ts = a[:,0].astype(float); pred = a[:,2:28].reshape(-1,13,2).astype(float)
    finite_ts = ts[np.isfinite(ts)]
    require(len(finite_ts) > 1 and np.all(np.diff(finite_ts) > 0), 'Timestamp predetti non strettamente crescenti')
    gt_ts, gt_xy = load_gt(gt_path, parser)
    row = metric_arrays(ts, pred, gt_ts, gt_xy)
    d = np.diff(finite_ts)
    row.update(prediction_sha256=sha256_file(csv_path), gt_sha256=sha256_file(gt_path),
               gt_first_ts_s=float(gt_ts[0]), gt_last_ts_s=float(gt_ts[-1]),
               median_output_period_s=float(np.median(d)), max_output_gap_s=float(d.max()),
               observed_output_hz=float((len(finite_ts)-1)/(finite_ts[-1]-finite_ts[0])),
               nominal_output_period_s=float(output_period_s))
    if input_span_s is not None:
        row['prediction_span_fraction_of_input'] = float((finite_ts[-1]-finite_ts[0])/input_span_s)
    # Actual schedules are preserved, including packet-time jitter in MoveEnet.
    # No resampling to a fictitious 200 Hz stream is performed here.
    return row
