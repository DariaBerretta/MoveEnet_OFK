#!/usr/bin/env python3
"""Validate the Experiment A worklist and output CSVs before scoring."""
import argparse
import csv
import json
import sys
from pathlib import Path

from run_experiment import count_prediction_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--allow-incomplete', action='store_true')
    args = p.parse_args()
    root = args.run_dir
    snapshot = json.loads((root / 'config.yaml').read_text(encoding='utf-8'))
    config = snapshot['config']
    with (root / 'manifest.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    failures = []
    keys = set()
    for r in rows:
        key = (r['dataset'], r['method'], r['sample_id'], r['network_hz'])
        if key in keys:
            failures.append((key, 'duplicate operating point'))
        keys.add(key)
        if r['method'] not in config['methods'] or float(r['network_hz']) not in config['network_hz'][r['method']]:
            failures.append((key, 'frequency or method outside configuration'))
        if float(r['output_hz']) != 200:
            failures.append((key, 'output rate is not 200 Hz'))
        if r['method'] == 'moveenetofk' and float(r['flow_hz']) != 200:
            failures.append((key, 'OFK flow rate is not 200 Hz'))
        if not Path(r['input_path']).is_file() or not Path(r['gt_path']).is_file():
            failures.append((key, 'input or ground truth missing'))
        if r['status'] == 'complete':
            try:
                if count_prediction_rows(Path(r['prediction_csv'])) < 2:
                    failures.append((key, 'prediction file has fewer than two rows'))
            except (ValueError, OSError) as exc:
                failures.append((key, str(exc)))
        elif not args.allow_incomplete:
            failures.append((key, 'status=' + r['status']))
    print('Dataset:', config['dataset'], '| trials:', len(rows), '| issues:', len(failures))
    for key, problem in failures[:30]:
        print('ERROR', key, problem)
    if len(failures) > 30:
        print('... {} more issues'.format(len(failures) - 30))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
