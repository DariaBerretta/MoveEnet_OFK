#!/usr/bin/env python3
"""Sequential real-model Experiment A smoke tests: one recording/rate/method.

This tool requires the user's Docker/YARP/data/GPU environment; it is never
run as part of offline unit tests. It does not execute a full dataset grid.
All output directories are run-scoped under the central HPE_RESULTS_ROOT.
"""
import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

from expa_common import DEFAULT_RESULTS, FLOW, read_config

HERE = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=('dhp19', 'eh36m', 'moveiit', 'all'))
    p.add_argument('--network-hz', type=int, default=5)
    p.add_argument('--results-root', default=os.environ.get('HPE_RESULTS_ROOT', DEFAULT_RESULTS))
    p.add_argument('--binary-dir', default=str(FLOW / 'build'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--run-prefix', default=None, help='Unique safe prefix, default UTC timestamp')
    p.add_argument('--plan', action='store_true', help='Print smoke tests without executing binaries')
    p.add_argument('--keep-going', action='store_true', help='Try other methods if one fails')
    return p


def main():
    args = parser().parse_args()
    datasets = ('dhp19', 'eh36m', 'moveiit') if args.dataset == 'all' else (args.dataset,)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix = args.run_prefix or ('smoke_' + stamp)
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', prefix):
        raise ValueError('Invalid --run-prefix')
    failed = []
    completed = 0
    for dataset in datasets:
        cfg = read_config(dataset)
        for method in cfg['methods']:
            if args.network_hz not in cfg['network_hz'][method]:
                raise ValueError('{} Hz is unsupported for {}/{}'.format(args.network_hz, dataset, method))
            run_id = '{}_{}_{}_{}Hz'.format(prefix, dataset, method, args.network_hz)
            run_dir = Path(args.results_root).expanduser() / 'ExpA' / run_id
            command = [sys.executable, str(HERE / 'run_experiment.py'),
                       '--dataset', dataset, '--method', method,
                       '--network-hz', str(args.network_hz), '--limit', '1',
                       '--run-id', run_id, '--results-root', args.results_root,
                       '--binary-dir', args.binary_dir, '--device', args.device]
            verify = [sys.executable, str(HERE / 'validate_results.py'),
                      '--run-dir', str(run_dir), '--strict-output-rate']
            print('SMOKE', dataset, method, args.network_hz, 'Hz ->', run_dir, flush=True)
            if args.plan:
                print('  RUN:', ' '.join(command))
                print('  CHECK:', ' '.join(verify))
                continue
            result = subprocess.run(command, check=False)
            if result.returncode == 0:
                result = subprocess.run(verify, check=False)
            if result.returncode:
                failed.append((dataset, method, str(run_dir), result.returncode))
                print('FAILED smoke for {}/{}; inspect logs under {}'.format(dataset, method, run_dir), file=sys.stderr)
                if not args.keep_going:
                    break
            else:
                completed += 1
        if failed and not args.keep_going:
            break
    if args.plan:
        print('PLAN ONLY; no inference executed')
        return 0
    print('Smoke tests completed:', completed, '| failed:', len(failed))
    for dataset, method, directory, code in failed:
        print('FAILED', dataset, method, 'exit=', code, directory)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
