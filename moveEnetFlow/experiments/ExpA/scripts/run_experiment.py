#!/usr/bin/env python3
"""Run Experiment A with one immutable configuration, manifest, and output tree.

Requires Python 3.8+. A run never overwrites another run or legacy data. The
runner invokes existing offline binaries sequentially and fails closed on
missing inputs, GT files, output CSVs, or nonzero return codes.
"""
import argparse
import csv
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import re
import time
from pathlib import Path

from expa_common import (DEFAULT_RESULTS, EXPA, FLOW, append_log, git_head,
                         inventory, read_config, safe_stem, sha256, write_csv)

FIELDS = ('dataset', 'method', 'sample_id', 'network_hz', 'flow_hz', 'output_hz',
          'input_path', 'gt_path', 'prediction_csv', 'log_file', 'server_log',
          'gpu_log', 'command_json', 'status', 'return_code', 'elapsed_s',
          'prediction_rows', 'error')
EXCLUDED_FIELDS = ('dataset', 'method', 'sample_id', 'input_path', 'gt_path', 'reason')


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=('eh36m', 'dhp19', 'moveiit'))
    p.add_argument('--method', action='append', help='Repeat to select methods; default all')
    p.add_argument('--network-hz', type=int, action='append', help='Repeat to select rates; default all')
    p.add_argument('--results-root', default=os.environ.get('HPE_RESULTS_ROOT', DEFAULT_RESULTS))
    p.add_argument('--run-id', help='New name under ExpA/, or existing run to resume')
    p.add_argument('--binary-dir', default=str(FLOW / 'build'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, default=0, help='Development only: number of recordings/method')
    p.add_argument('--dry-run', action='store_true', help='Print plan, write metadata and manifest, no inference')
    p.add_argument('--resume', action='store_true', help='Resume only the same run-id and exact config')
    p.add_argument('--keep-going', action='store_true', help='Record failures and continue')
    return p


def command_for(config, method, hz, input_path, pred_path, server_log, gpu_log, binary_dir, device):
    """Use the exact CLI flags exposed by each current C++ executable."""
    dataset = config['dataset']
    p = 1.0 / hz
    output_period = 1.0 / config['output_hz']
    w, h = config['sensor']
    binary = binary_dir / {'movenet': 'moveEnetOFK_offline',
                           'moveenetofk': 'moveEnetOFK_offline',
                           'yolo': 'YoloPose_offline',
                           'openpose': 'OpenPose_offline',
                           'eventpointpose': 'eventPointPose_offline'}[method]
    args = [str(binary), '--data_file', str(input_path), '--net_period', format(p, '.10g'),
            '--output_period', format(output_period, '.10g'), '--device', device]
    if method in ('movenet', 'moveenetofk'):
        kf = config['kf']
        args += ['--flow_period', format(1.0 / config['flow_hz'], '.10g'),
                 '--roi', str(kf['roi']), '--pu', str(kf['pu']), '--muD', str(kf['muD']),
                 '--muV', str(kf['muV']), '--of_gain', str(kf['of_gain']),
                 '--use_lc', str(kf['use_lc']).lower(),
                 '--checkpoint_path', config['movenet_checkpoint'],
                 '--gpu_file', str(gpu_log), '--output_csv_f', str(pred_path), '--no_video']
        if dataset == 'dhp19':
            args += ['--dhp19']
        elif dataset == 'moveiit':
            args += ['--moveiit']
        else:
            args += ['--w', str(w), '--h', str(h)]
        if method == 'movenet':
            args += ['--moveenet_only']
    elif method in ('openpose', 'yolo'):
        args += ['--w', str(w), '--h', str(h), '--no_video',
                 '--gpu_file', str(gpu_log), '--output_csv', str(pred_path)]
    else:
        args += ['--model_path', config['epp_checkpoint'],
                 '--EventPointPose_script', config['epp_script'],
                 '--fifo_size', '7500', '--num_points', '2048', '--seed', '1',
                 '--input_preprocessing', 'already_filtered',
                 '--server_log', str(server_log), '--output_csv', str(pred_path)]
        if dataset == 'dhp19':
            args += ['--camera', input_path.parent.name[2], '--w', str(w), '--h', str(h)]
        elif dataset == 'moveiit':
            args += ['--moveiit', '--moveiit_geometry', config['epp_geometry'],
                     '--camera', '0', '--w', '346', '--h', '260']
        else:
            raise ValueError('EventPointPose not configured for eH36M')
    return args


def count_prediction_rows(path):
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        expected = ['timestamp'] + ['joint{}_{}'.format(j, axis) for j in range(13) for axis in 'xy']
        if not reader.fieldnames or not set(expected).issubset(reader.fieldnames):
            raise ValueError('Missing timestamp or 13-joint CSV columns')
        n = 0
        last = None
        for row in reader:
            ts = float(row['timestamp'])
            if not (ts == ts and abs(ts) < float('inf')) or (last is not None and ts <= last):
                raise ValueError('Nonfinite or nonmonotonic prediction timestamps')
            last = ts
            n += 1
        return n


def main():
    opts = args_parser().parse_args()
    config = read_config(opts.dataset)
    methods = opts.method or list(config['methods'])
    if len(methods) != len(set(methods)) or any(m not in config['methods'] for m in methods):
        raise ValueError('Method selection must be distinct and configured for dataset')
    if opts.limit < 0:
        raise ValueError('--limit cannot be negative')
    if opts.resume and not opts.run_id:
        raise ValueError('--resume requires --run-id')
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = opts.run_id or ('run_' + now + '_' + opts.dataset)
    # Validate the run identifier while allowing underscores, dots and hyphens.
    # The identifier must start with a letter or digit to avoid unsafe paths.
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError(
            '--run-id must start with a letter or digit and contain only '
            'letters, numbers, underscores, dots or hyphens'
        )
    run_dir = Path(opts.results_root).expanduser().resolve() / 'ExpA' / run_id
    if run_dir.exists() and not opts.resume:
        raise FileExistsError('Run already exists (use --resume after checking config): ' + str(run_dir))
    if opts.resume and not run_dir.is_dir():
        raise FileNotFoundError('Cannot resume missing run directory: ' + str(run_dir))
    selection = {'dataset': opts.dataset, 'methods': methods,
                 'network_hz': opts.network_hz, 'device': opts.device,
                 'binary_dir': str(Path(opts.binary_dir).resolve()), 'limit': opts.limit}
    snapshot = {'config': config, 'selection': selection}
    snapshot_path = run_dir / 'config.yaml'
    if opts.resume:
        if json.loads(snapshot_path.read_text(encoding='utf-8')) != snapshot:
            raise ValueError('Resume configuration does not match the original run')
    else:
        run_dir.mkdir(parents=True)
        snapshot_path.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
    binary_dir = Path(opts.binary_dir)
    plan = []
    excluded = []
    binaries = {}
    for method in methods:
        data, rejected = inventory(config, method)
        excluded.extend(rejected)
        if opts.limit:
            data = data[:opts.limit]
        if not data:
            raise RuntimeError('No eligible GT-paired sequences for ' + opts.dataset + '/' + method)
        rates = opts.network_hz or config['network_hz'][method]
        if any(r not in config['network_hz'][method] for r in rates):
            raise ValueError('Unsupported neural frequency in ' + method + ': ' + str(rates))
        if len(rates) != len(set(rates)):
            raise ValueError('Duplicate network frequency selection')
        for source in data:
            for hz in rates:
                tag = safe_stem(source['sample_id'])
                rel = Path(opts.dataset) / method / ('nn_{:g}Hz'.format(hz)) / (tag + '.csv')
                pred = run_dir / 'predictions' / rel
                log = run_dir / 'logs' / rel.with_suffix('.log')
                server_log = run_dir / 'logs' / rel.with_suffix('.server.log')
                gpu_log = run_dir / 'telemetry' / rel.with_suffix('.gpu.csv')
                cmd = command_for(config, method, hz, Path(source['input_path']), pred,
                                  server_log, gpu_log, binary_dir, opts.device)
                if cmd[0] not in binaries:
                    binaries[cmd[0]] = sha256(cmd[0])
                entry = {**source, 'network_hz': hz,
                         'flow_hz': config['flow_hz'] if method == 'moveenetofk' else '',
                         'output_hz': config['output_hz'], 'prediction_csv': str(pred),
                         'log_file': str(log), 'server_log': str(server_log) if method == 'eventpointpose' else '',
                         'gpu_log': str(gpu_log) if method != 'eventpointpose' else '',
                         'command_json': json.dumps(cmd), 'status': 'planned',
                         'return_code': '', 'elapsed_s': '', 'prediction_rows': '', 'error': ''}
                plan.append(entry)
    expected_meta = {'git_head': git_head(), 'configuration': snapshot,
                     'binary_sha256': binaries,
                     'checkpoint_sha256': {k: sha256(config[k]) for k in ('movenet_checkpoint', 'epp_checkpoint') if k in config},
                     'python': sys.version, 'platform': platform.platform(),
                     'protocol': 'ExpA/accuracy/pooled_GT_valid_PCK@0.2_torso_1_5',
                     'gt_protocol_status': config['gt_protocol_status'],
                     'created_utc': now, 'dry_run': opts.dry_run}
    meta_path = run_dir / 'metadata.json'
    if opts.resume:
        old = json.loads(meta_path.read_text(encoding='utf-8'))
        # Exact binaries and source commit must remain constant across resumed runs.
        for key in ('git_head', 'configuration', 'binary_sha256', 'checkpoint_sha256'):
            if old[key] != expected_meta[key]:
                raise ValueError('Cannot resume after changing ' + key)
    else:
        meta_path.write_text(json.dumps(expected_meta, indent=2) + '\n', encoding='utf-8')
    write_csv(run_dir / 'excluded.csv', excluded, EXCLUDED_FIELDS)
    manifest_path = run_dir / 'manifest.csv'
    if opts.resume:
        with manifest_path.open(encoding='utf-8', newline='') as f:
            previous = {row['prediction_csv']: row for row in csv.DictReader(f)}
        if set(previous) != {entry['prediction_csv'] for entry in plan}:
            raise ValueError('Cannot resume: manifest worklist differs from requested worklist')
        for row in plan:
            saved = previous[row['prediction_csv']]
            if any(row[k] != saved[k] for k in ('command_json', 'input_path', 'gt_path')):
                raise ValueError('Cannot resume: run command or inputs changed')
            row.update({k: saved[k] for k in ('status', 'return_code', 'elapsed_s', 'prediction_rows', 'error')})
    write_csv(manifest_path, plan, FIELDS)
    print('Run directory:', run_dir)
    print('Planned trials:', len(plan), '| rejected inputs:', len(excluded))
    if opts.dry_run:
        for row in plan[:8]:
            print('DRY RUN:', row['method'], row['sample_id'], row['network_hz'], 'Hz')
        return 0
    if any(b is None for b in binaries.values()):
        raise FileNotFoundError('Build the offline binaries first: ' + str(binaries))
    failures = 0
    for i, row in enumerate(plan, 1):
        pred = Path(row['prediction_csv'])
        if row['status'] == 'complete' and count_prediction_rows(pred) >= 2:
            print('[{}/{}] SKIP verified complete: {}'.format(i, len(plan), pred))
            continue
        pred.parent.mkdir(parents=True, exist_ok=True)
        log = Path(row['log_file'])
        log.parent.mkdir(parents=True, exist_ok=True)
        if row['server_log']:
            Path(row['server_log']).parent.mkdir(parents=True, exist_ok=True)
        if row['gpu_log']:
            Path(row['gpu_log']).parent.mkdir(parents=True, exist_ok=True)
        if pred.exists():
            pred.unlink()  # Resumed failed trial only; never touch completed predictions.
        row['status'] = 'running'
        row['error'] = ''
        write_csv(manifest_path, plan, FIELDS)
        cmd = json.loads(row['command_json'])
        print('[{}/{}] RUN {} {} {} Hz'.format(i, len(plan), row['method'], row['sample_id'], row['network_hz']), flush=True)
        started = time.monotonic()
        with log.open('w', encoding='utf-8') as f:
            try:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
                code = result.returncode
            except OSError as exc:
                code = 127
                row['error'] = str(exc)
        row['return_code'] = code
        row['elapsed_s'] = round(time.monotonic() - started, 5)
        try:
            count = count_prediction_rows(pred)
            row['prediction_rows'] = count
            if code != 0 or count < 2:
                raise RuntimeError('return_code={} output_rows={}'.format(code, count))
            row['status'] = 'complete'
        except (OSError, ValueError, RuntimeError) as exc:
            row['status'] = 'failed'
            row['error'] = str(exc)
            failures += 1
        append_log(run_dir / 'audit.jsonl', {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                            'sample_id': row['sample_id'], 'method': row['method'],
                                            'network_hz': row['network_hz'], 'status': row['status'],
                                            'error': row['error']})
        write_csv(manifest_path, plan, FIELDS)
        if failures and not opts.keep_going:
            print('FAILED:', row['error'], '| log:', row['log_file'], file=sys.stderr)
            return 1
    print('Complete:', sum(r['status'] == 'complete' for r in plan), '/', len(plan))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
