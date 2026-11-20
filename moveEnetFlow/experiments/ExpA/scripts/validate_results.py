#!/usr/bin/env python3
"""Validate the ExpA worklist, raw 13-joint CSVs, and output cadence.

CSV timestamps describe *recorded* output cadence; requested neural inference
frequency cannot be independently inferred from a held-pose prediction CSV.
Use model/server logs for the latter.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from expa_common import write_csv
from run_experiment import count_prediction_rows

TIMING_FIELDS = ('dataset', 'method', 'sample_id', 'network_hz', 'output_hz_requested',
                 'n_output_rows', 'median_output_hz_observed', 'min_output_dt_s',
                 'max_output_dt_s', 'cadence_warning')


def inspect_output_cadence(path, expected_hz):
    """Measure the actual saved timestamp spacing without inferring net rate."""
    with path.open(encoding='utf-8', newline='') as f:
        ts = [float(row['timestamp']) for row in csv.DictReader(f)]
    if len(ts) < 2:
        raise ValueError('Need at least two poses to measure output cadence')
    dt = [b - a for a, b in zip(ts[:-1], ts[1:])]
    if any(v <= 0 for v in dt):
        raise ValueError('Nonmonotonic CSV timestamp spacing')
    median_hz = 1.0 / statistics.median(dt)
    # A nominal 200 Hz output is validated from event-packet timestamps; this
    # tolerance is a *warning* threshold, not a fabricated hard real-time claim.
    warning = '' if abs(median_hz / expected_hz - 1.0) <= 0.2 else 'median_output_rate_differs_by_over_20_percent'
    return median_hz, min(dt), max(dt), warning


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--allow-incomplete', action='store_true')
    p.add_argument('--strict-output-rate', action='store_true',
                   help='Treat >20 percent median output-rate deviation as failure')
    args = p.parse_args()
    root = args.run_dir
    snapshot = json.loads((root / 'config.yaml').read_text(encoding='utf-8'))
    config = snapshot['config']
    with (root / 'manifest.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    failures = []
    keys = set()
    timing = []
    for r in rows:
        key = (r['dataset'], r['method'], r['sample_id'], r['network_hz'])
        if key in keys:
            failures.append((key, 'duplicate operating point'))
        keys.add(key)
        if r['method'] not in config['methods'] or float(r['network_hz']) not in config['network_hz'].get(r['method'], []):
            failures.append((key, 'frequency or method outside configuration'))
        if float(r['output_hz']) != 200:
            failures.append((key, 'output rate is not 200 Hz'))
        if r['method'] == 'moveenetofk' and float(r['flow_hz']) != 200:
            failures.append((key, 'OFK flow rate is not 200 Hz'))
        if not Path(r['input_path']).is_file() or not Path(r['gt_path']).is_file():
            failures.append((key, 'input or ground truth missing'))
        if r['status'] == 'complete':
            try:
                path = Path(r['prediction_csv'])
                count = count_prediction_rows(path)
                if count < 2:
                    failures.append((key, 'prediction file has fewer than two rows'))
                else:
                    median_hz, minimum, maximum, warning = inspect_output_cadence(path, float(r['output_hz']))
                    timing.append({'dataset': r['dataset'], 'method': r['method'],
                                   'sample_id': r['sample_id'], 'network_hz': r['network_hz'],
                                   'output_hz_requested': r['output_hz'],
                                   'n_output_rows': count,
                                   'median_output_hz_observed': round(median_hz, 4),
                                   'min_output_dt_s': round(minimum, 7),
                                   'max_output_dt_s': round(maximum, 7),
                                   'cadence_warning': warning})
                    if warning:
                        print('WARNING: {} / {} / {} output median {:.2f} Hz, requested {} Hz'.format(
                            r['dataset'], r['method'], r['sample_id'], median_hz, r['output_hz']))
                        if args.strict_output_rate:
                            failures.append((key, warning))
            except (ValueError, OSError, KeyError) as exc:
                failures.append((key, str(exc)))
        elif not args.allow_incomplete:
            failures.append((key, 'status=' + r['status']))
    # Run-scoped timing QA is inspectable alongside metrics; never modify raw CSVs.
    if timing:
        write_csv(root / 'metrics' / 'output_timing_qa.csv', timing, TIMING_FIELDS)
    print('Dataset:', config['dataset'], '| trials:', len(rows), '| issues:', len(failures),
          '| timing checked:', len(timing))
    for key, problem in failures[:30]:
        print('ERROR', key, problem)
    if len(failures) > 30:
        print('... {} more issues'.format(len(failures) - 30))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
