#!/usr/bin/env python3
"""HOST runner: EventPointPose energy + RTF on the existing 12 Move-IIT samples.

Uses ExpEnergy/measure_energy.py unmodified (RAPL CPU package, NVML GPU device).
Read the exact event input spans from the completed Move-IIT baseline session.
Run on the HOST, outside Docker. Python >=3.10 and pynvml required on host.

Model configurations: original EventPointPose (stretch, unmodified weights), 5/50 Hz
network, 200 Hz nominal output. Five repeats, balanced order and cooldown.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
METER = HERE / 'measure_energy.py'
PROTOCOL = 'moveiit_epp_energy_rtf_v1_reference_exact_packet_span'
SOURCE = 'YARP_data_log_first_last_AE_packet_timestamp'
MODELS = [('eventpointpose_5hz', '0.2'), ('eventpointpose_50hz', '0.02')]
ROOT = Path.home() / 'data/MoveEnet_OFK_results/Energy'
DEFAULT_BASELINE = ROOT / 'moveiit_energy_rtf_roi35_gain1_20261103'
if not DEFAULT_BASELINE.is_dir() and (Path('/data/MoveEnet_OFK_results/Energy') / DEFAULT_BASELINE.name).is_dir():
    DEFAULT_BASELINE = Path('/data/MoveEnet_OFK_results/Energy') / DEFAULT_BASELINE.name
DEFAULT_SESSION = DEFAULT_BASELINE.parent / 'moveiit_eventpointpose_energy_rtf_stretch_v1'
BASE_PATH = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_csv(path):
    require(path.is_file(), 'Missing CSV: {}'.format(path))
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def command(argv, check=True):
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, check=False)
    if check and p.returncode:
        raise RuntimeError('Command failed ({}): {}\n{}'.format(
            p.returncode, shlex.join(argv), p.stderr[-1500:]))
    return p.stdout.strip()


def docker(container, *args):
    return command(['docker', 'exec', container, *map(str, args)])


def samples_from_reference(reference):
    config_path = reference / 'config.json'
    require(config_path.is_file(), 'Reference session not found: {}. Use --reference-session.'.format(reference))
    cfg = json.loads(config_path.read_text())
    require(cfg.get('protocol') == 'moveiit_energy_rtf_expFinal_aligned_v3_exact_packet_span',
            'Reference session does not use the established exact-packet-span protocol')
    rows = read_csv(reference / 'samples_durations.csv')
    require(len(rows) == 12, 'Reference must contain exactly 12 selected samples, found {}'.format(len(rows)))
    require(len({r['sample_id'] for r in rows}) == len(rows), 'Duplicate reference sample ID')
    require({r['sample_id'] for r in rows} == set(cfg['sample_ids']),
            'Reference session and duration manifest sample IDs disagree')
    for r in rows:
        require(r['duration_source'] == SOURCE, 'Incorrect duration source for {}'.format(r['sample_id']))
        require(r['subject'] in ('luna', 'simon'), 'Unknown subject')
        d = float(r['duration_s'])
        lo, hi = float(r['input_start_s']), float(r['input_end_s'])
        require(all(map(math.isfinite, (d, lo, hi))) and d > 0 and abs((hi - lo) - d) < 2e-6,
                'Invalid exact input duration for {}'.format(r['sample_id']))
        expected = '/data/move-iit-it_testing/{}/{}/atis-s/data.log'.format(r['subject'], r['sequence'])
        require(r['event_file'] == expected, 'Unexpected event file for {}: {}'.format(r['sample_id'], r['event_file']))
    rows.sort(key=lambda r: r['sample_id'])
    return cfg, rows


def parse_epp_log(log, expected_events=None):
    """Check full replay, not just a 'valid inference' summary printed on exit."""
    if not log.is_file():
        return False, 'missing log', {}
    data = log.read_text(errors='replace')
    patterns = {
        'loaded_events': r'\b([0-9]+)\s+total events\b',
        'processed_events': r'\bEvents read:\s*([0-9]+)',
        'requests': r'\bPointNet requests:\s*([0-9]+)',
        'valid_inferences': r'\bValid PointNet inferences:\s*([0-9]+)',
        'server_errors': r'\bServer ERROR responses:\s*([0-9]+)',
    }
    got = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, data, re.IGNORECASE)
        if not matches:
            return False, 'missing end-of-replay counter: {}'.format(key), got
        got[key] = int(matches[-1])
    if re.search(r'Termination signal received|request failed|\[ERROR\]', data, re.IGNORECASE):
        return False, 'termination/error in log', got
    if got['loaded_events'] < 1 or got['loaded_events'] != got['processed_events']:
        return False, 'loaded events differ from processed events', got
    if expected_events is not None and got['loaded_events'] != expected_events:
        return False, 'loaded event count changed compared with earlier successful replay', got
    if got['requests'] < 1 or got['valid_inferences'] < 1:
        return False, 'no valid inference', got
    if got['server_errors'] != 0:
        return False, 'server ERROR responses observed', got
    return True, 'OK', got


def build_command(args, sample, model, server_log):
    name, np = model
    executable = [args.binary, '--data_file', sample['event_file'],
                  '--moveiit', '--moveiit_geometry', 'stretch', '--camera', '0',
                  '--w', '346', '--h', '260', '--net_period', np,
                  '--output_period', '0.005', '--input_preprocessing', 'already_filtered',
                  '--fifo_size', '7500', '--num_points', '2048', '--seed', '1',
                  '--model_path', args.model_path, '--EventPointPose_script', args.server_script,
                  '--device', args.device, '--no_csv',
                  '--response_timeout', str(args.response_timeout),
                  '--startup_timeout', str(args.startup_timeout),
                  '--server_log', str(server_log)]
    # C++ launches 'python3' for PointNet. Setting PATH here makes that interpreter
    # deterministic, including for the forked server process.
    env = ['VIRTUAL_ENV=' + args.venv, 'PATH=' + args.venv + '/bin:' + BASE_PATH]
    return ['docker', 'exec', args.container, 'env', *env, *executable]


def completed_keys(energy_file):
    completed = {}
    if not energy_file.is_file():
        return completed
    for r in read_csv(energy_file):
        if r.get('model') not in dict(MODELS):
            continue
        if r.get('status') != 'OK' or r.get('exit_code') not in ('0', '0.0'):
            continue
        ok, reason, counts = parse_epp_log(Path(r['log_path']))
        if not ok:
            continue
        key = (r['sample_id'], r['model'], r['order_id'])
        require(key not in completed, 'More than one completed attempt for {}. Inspect energy CSV.'.format(key))
        completed[key] = counts
    return completed


def attempt_number(energy_file, sample_id, model, order_id):
    if not energy_file.exists():
        return 1
    return 1 + sum(1 for r in read_csv(energy_file)
                   if (r.get('sample_id'), r.get('model'), r.get('order_id')) ==
                   (sample_id, model, order_id))


def preflight(args, rows, reference):
    require(METER.is_file(), 'Original energy meter is missing: {}'.format(METER))
    require(sys.version_info >= (3, 10), 'Host Python >= 3.10 required by original energy meter')
    command([args.python, '-c', 'import pynvml; import sys; assert sys.version_info >= (3,10)'])
    require(Path('/sys/class/powercap/intel-rapl:0/energy_uj').is_file(),
            'Host RAPL package energy counter missing')
    require(os.access('/sys/class/powercap/intel-rapl:0/energy_uj', os.R_OK),
            'Host RAPL counter unreadable')
    int(Path('/sys/class/powercap/intel-rapl:0/energy_uj').read_text().strip())
    # Verify NVML cumulative energy before the first measured run, and that
    # the selected GPU matches the historical MoveEnet session.
    nvml_script = ('import pynvml; pynvml.nvmlInit(); '
                   'h=pynvml.nvmlDeviceGetHandleByIndex({}); '
                   'u=pynvml.nvmlDeviceGetUUID(h); '
                   'print(u.decode() if isinstance(u,bytes) else u); '
                   'print(pynvml.nvmlDeviceGetTotalEnergyConsumption(h)); '
                   'pynvml.nvmlShutdown()').format(args.gpu_index)
    gpu_lines = command([args.python, '-c', nvml_script]).splitlines()
    require(len(gpu_lines) == 2, 'Cannot read host GPU UUID / cumulative-energy counter')
    gpu_uuid = gpu_lines[0]
    base_meter = read_csv(reference/'energy_measurements.csv')
    ref_uuids = {r['gpu_uuid'] for r in base_meter if r.get('status')=='OK'}
    require(ref_uuids == {gpu_uuid},
            'Reference used a different GPU: {} vs {}. Refuse cross-hardware comparison'.format(ref_uuids,gpu_uuid))
    require({str(r['gpu_index']) for r in base_meter if r.get('status')=='OK'} == {str(args.gpu_index)},
            'Reference used a different GPU index')
    require(command(['docker', 'inspect', '-f', '{{.State.Running}}', args.container]) == 'true',
            'Docker container not running')
    for p in (args.binary, args.model_path, args.server_script, args.venv + '/bin/python3'):
        docker(args.container, 'test', '-e', p)
    docker(args.container, 'test', '-x', args.binary)
    # Fail before energy measurements if the PointNet server will use an invalid
    # interpreter environment. Does not import/train/modify the network.
    docker(args.container, 'env', 'PATH=' + args.venv + '/bin:' + BASE_PATH,
           'python3', '-c', 'import torch, numpy, yarp')
    docker(args.container, 'bash', '-lc', 'yarp where >/dev/null 2>&1')
    for r in rows:
        docker(args.container, 'test', '-s', r['event_file'])
    active = docker(args.container, 'bash', '-lc',
                    "pgrep -af '[m]ovenet_online.py|[e]ventPointPose_yarp_server.py|[e]ventPointPose_offline|[m]oveEnetOFK_offline|[o]f_sanity_gui_engine|[Y]oloPose_yarp_server.py' || true")
    require(not active, 'Other HPE job currently running inside container:\n' + active)
    busy = command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                    '--format=csv,noheader'], check=False)
    if busy:
        print('WARNING: GPU compute processes detected; device-wide energy may be contaminated:\n' + busy,
              file=sys.stderr)
    print('PREFLIGHT OK: RAPL, NVML, Docker, checkpoint, model server and dataset.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--container', default='moveEnetOFK')
    ap.add_argument('--reference-session', type=Path,
                    default=Path(os.environ.get('MOVEIIT_ENERGY_SESSION_DIR', str(DEFAULT_BASELINE))))
    ap.add_argument('--session-dir', type=Path, default=DEFAULT_SESSION)
    ap.add_argument('--python', default=sys.executable, help='HOST Python >=3.10 with pynvml')
    ap.add_argument('--venv', default='/opt/venv', help='CONTAINER venv; used by Python PointNet sidecar')
    ap.add_argument('--binary', default='/workspace/moveEnetFlow/build/eventPointPose_offline')
    ap.add_argument('--server-script', default='/workspace/moveEnetFlow/utils/eventPointPose_yarp_server.py')
    ap.add_argument('--model-path', default='/workspace/model_mounts/eventpointpose/PointNet/models/model.pth')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--gpu-index', type=int, default=0)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--cooldown', type=float, default=20.)
    ap.add_argument('--sample', default='', help='M01..M12; only for smoke test')
    ap.add_argument('--response-timeout', type=float, default=60.)
    ap.add_argument('--startup-timeout', type=float, default=90.)
    ap.add_argument('--server-log-dir', default='/data/MoveEnet_OFK_results/Energy/eventpointpose_server_logs')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    require(args.repeats >= 1 and args.cooldown >= 0 and args.gpu_index >= 0,
            'Invalid repeats/cooldown/gpu index')
    require(args.response_timeout > 0 and args.startup_timeout > 0,
            'Timeouts must be positive')
    reference = args.reference_session.expanduser().resolve()
    args.session_dir = args.session_dir.expanduser().resolve()
    baseline_cfg, all_samples = samples_from_reference(reference)
    selected = [r for r in all_samples if not args.sample or r['sample_id'] == args.sample]
    require(selected, 'No matching sample ID')
    print('Move-IIT EventPointPose energy and RTF: {} samples x 2 rates x {} repeats = {} runs'.format(
        len(selected), args.repeats, len(selected) * 2 * args.repeats))
    print('Reference:', reference)
    print('Session:', args.session_dir)
    print('Cases:', MODELS, '| output: 200 Hz | geometry: stretch | no CSV')
    preview = build_command(args, selected[0], MODELS[0], Path(args.server_log_dir) / 'preview.log')
    if args.dry_run:
        print('DRY RUN example:', shlex.join(preview))
        return 0
    preflight(args, selected, reference)
    if args.preflight_only:
        for r in selected:
            print('{} {} / {}: input span {:.6f}s'.format(
                r['sample_id'], r['subject'], r['sequence'], float(r['duration_s'])))
        return 0
    # Capture immutable experiment metadata. Resume cannot silently mix devices,
    # source files, methods, measurement code, or selected recordings.
    container_hash = lambda path: docker(args.container, 'sha256sum', path).split()[0]
    config = dict(protocol=PROTOCOL, reference_session=str(reference),
                  reference_config_sha256=digest(reference/'config.json'),
                  reference_durations_sha256=digest(reference/'samples_durations.csv'),
                  reference_protocol=baseline_cfg['protocol'],
                  sample_ids=[r['sample_id'] for r in selected],
                  models={name: float(np) for name, np in MODELS},
                  repeats=args.repeats, cooldown_s=args.cooldown, gpu_index=args.gpu_index,
                  device=args.device, container=args.container, python_venv=args.venv,
                  binary=args.binary, binary_sha256=container_hash(args.binary),
                  model_path=args.model_path, model_sha256=container_hash(args.model_path),
                  server_script=args.server_script, server_sha256=container_hash(args.server_script),
                  measure_script_sha256=digest(METER),
                  input_span_method=SOURCE, geometry='stretch', fifo_size=7500,
                  num_points=2048, seed=1, output_period_s=.005,
                  input_preprocessing='already_filtered',
                  response_timeout_s=args.response_timeout, startup_timeout_s=args.startup_timeout,
                  server_log_dir=args.server_log_dir)
    session = args.session_dir
    conf_path = session / 'config.json'
    if conf_path.exists():
        require(args.resume, 'Session already exists: {}. Use --resume or another --session-dir'.format(session))
        require(json.loads(conf_path.read_text()) == config,
                'Cannot resume: session configuration has changed')
    else:
        require(not session.exists() or not any(session.iterdir()),
                'Session directory is non-empty; use a new --session-dir')
        session.mkdir(parents=True, exist_ok=True)
        conf_path.write_text(json.dumps(config, indent=2, sort_keys=True)+'\n')
        with (session/'samples_durations.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    run_dir = session / 'runs'
    run_dir.mkdir(exist_ok=True)
    docker(args.container, 'mkdir', '-p', args.server_log_dir)
    energy_csv = session / 'energy_measurements.csv'
    manifest = session / 'run_manifest.csv'
    if not manifest.exists():
        with manifest.open('w', newline='') as f:
            csv.writer(f).writerow(['dataset','sample_id','subject','sequence','model','repetition',
                                    'order_position','net_period','net_hz','output_period',
                                    'status','reason','energy_csv','log_path','server_log'])
    done = completed_keys(energy_csv)
    if args.cooldown:
        time.sleep(args.cooldown)
    for si, r in enumerate(selected):
        for rep in range(1, args.repeats+1):
            # Balance 5/50-Hz order across repetitions and recordings.
            models = MODELS if (si + rep) % 2 == 0 else MODELS[::-1]
            for order_position, model in enumerate(models, 1):
                model_name, period = model
                key = (r['sample_id'], model_name, 'R{}'.format(rep))
                if key in done:
                    print('SKIP {}'.format(key), flush=True)
                    continue
                attempt = attempt_number(energy_csv, *key)
                stem = '{}_{}_r{:02d}_a{:02d}'.format(r['sample_id'], model_name, rep, attempt)
                log = (run_dir/(stem+'.log')).resolve()
                server_log = Path(args.server_log_dir) / (stem+'.server.log')
                argv = build_command(args, r, model, server_log)
                measure = [args.python, str(METER), '--csv', str(energy_csv),
                           '--log', str(log), '--label', stem,
                           '--dataset', 'moveiit', '--sample-id', r['sample_id'],
                           '--model', model_name, '--order-id', key[2],
                           '--order-position', str(order_position), '--net-period', period,
                           '--gpu-index', str(args.gpu_index), '--', *argv]
                print('RUN {} (attempt {})'.format(key, attempt), flush=True)
                result = subprocess.run(measure, check=False)
                good, reason, counts = parse_epp_log(log)
                good = good and result.returncode == 0
                status = 'OK' if good else 'FAILED'
                with manifest.open('a', newline='') as f:
                    csv.writer(f).writerow(['moveiit', r['sample_id'], r['subject'], r['sequence'],
                                            model_name, rep, order_position, period, 1/float(period),
                                            .005, status, reason if result.returncode == 0 else
                                            'meter/command return code {}'.format(result.returncode),
                                            energy_csv, log, server_log])
                if not good:
                    raise RuntimeError('Run invalid: {}. {}. Inspect {} and {}. '
                                       'Relaunch with --resume after resolving the issue.'.format(
                                           key, reason, log, server_log))
                done[key] = counts
                if args.cooldown:
                    time.sleep(args.cooldown)
    print('SUCCESS: all selected runs complete. Energy CSV:', energy_csv)
    print('Analyze: python3 {} --session-dir {} --reference-session {}'.format(
        HERE/'analyze_moveiit_eventpointpose_energy_rtf.py', session, reference))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError) as e:
        print('ERROR:', e, file=sys.stderr)
        sys.exit(1)
