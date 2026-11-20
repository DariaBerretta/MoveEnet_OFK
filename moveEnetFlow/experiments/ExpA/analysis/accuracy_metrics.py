#!/usr/bin/env python3
"""Score a completed Experiment A run against its original 2D ground truth.

Uses the existing paper_figures.raw_metrics implementation for all datasets.
MPJPE/PCK use valid observed joints only; PCK@0.2 normalises by GT
right-shoulder-to-left-hip (joint 1 to 5). Do not mix separate cohorts.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLOW = HERE.parent.parent.parent
sys.path.insert(0, str(FLOW / 'analysis'))
from paper_figures import raw_metrics  # noqa: E402


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--strict', action='store_true', help='Fail immediately on an invalid prediction')
    p.add_argument('--allow-incomplete', action='store_true', help='Exploratory analysis of partial runs only')
    return p.parse_args()


def main():
    o = cli()
    root = o.run_dir.resolve()
    config = json.loads((root / 'config.yaml').read_text(encoding='utf-8'))['config']
    metadata = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
    with (root / 'manifest.csv').open(encoding='utf-8', newline='') as stream:
        manifest = list(csv.DictReader(stream))
    if not manifest:
        raise RuntimeError('Empty experiment manifest')
    pending = [r for r in manifest if r['status'] != 'complete']
    if pending and not o.allow_incomplete:
        raise RuntimeError('{} runs are not complete; use --allow-incomplete only for diagnostics'.format(len(pending)))
    cache = {}
    scores = []
    failed = []
    for rec in manifest:
        if rec['status'] != 'complete':
            failed.append({'method': rec['method'], 'sample_id': rec['sample_id'],
                           'network_hz': rec['network_hz'], 'reason': 'run_' + rec['status']})
            continue
        try:
            result = raw_metrics.evaluate_one(rec['prediction_csv'], rec['gt_path'],
                                              rec['dataset'], gt_cache=cache,
                                              gt_offset_s=config['gt_offset_s'],
                                              max_gt_gap_s=config['max_gt_gap_s'])
            scores.append({**result, 'dataset': rec['dataset'], 'method': rec['method'],
                           'sample_id': rec['sample_id'], 'network_hz': float(rec['network_hz']),
                           'flow_hz': float(rec['flow_hz']) if rec['flow_hz'] else float('nan'),
                           'output_hz': float(rec['output_hz'])})
        except Exception as exc:
            failed.append({'method': rec['method'], 'sample_id': rec['sample_id'],
                           'network_hz': rec['network_hz'], 'reason': str(exc)})
            if o.strict:
                raise
    output = root / 'metrics'
    output.mkdir(exist_ok=True)
    import pandas as pd
    per_sequence = pd.DataFrame(scores)
    failures = pd.DataFrame(failed, columns=['method', 'sample_id', 'network_hz', 'reason'])
    per_sequence.to_csv(output / 'per_sequence.csv', index=False)
    failures.to_csv(output / 'rejected.csv', index=False)
    if per_sequence.empty:
        raise RuntimeError('No valid scored predictions')
    duplicates = per_sequence.duplicated(['dataset', 'method', 'sample_id', 'network_hz', 'flow_hz'], keep=False)
    if duplicates.any():
        raise ValueError('Duplicate model/recording/rate outputs in manifest')
    summary = raw_metrics.summarize(per_sequence, 'ExpA_2DGT_v1', root.name)
    summary.to_csv(output / 'summary.csv', index=False)
    protocol = {'run_id': root.name, 'git_head': metadata['git_head'],
                'dataset': config['dataset'], 'output_hz': 200,
                'gt_offset_s': config['gt_offset_s'], 'gt_max_gap_s': config['max_gt_gap_s'],
                'gt_protocol_status': config['gt_protocol_status'],
                'pck_reference_joints': [1, 5], 'pck_alpha': 0.2,
                'mpjpe_aggregation': 'pooled_evaluable_joint_observations',
                'pck_aggregation': 'pooled_correct_over_evaluable_joint_observations',
                'n_failed_or_rejected_predictions': len(failures),
                'complete_dataset_evaluation': not pending and not failed}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    print(summary[['dataset', 'method', 'network_hz', 'n_sequences', 'mpjpe_px', 'pck_percent']].to_string(index=False))
    print('Results:', output)
    if failed:
        print('WARNING: {} predictions could not be evaluated; see rejected.csv'.format(len(failed)), file=sys.stderr)
    if config['gt_protocol_status'].startswith('PROVISIONAL'):
        print('WARNING: Move-IIT-HPE GT mapping/time offset remain provisional', file=sys.stderr)
    return 0 if not failed else 2


if __name__ == '__main__':
    sys.exit(main())
