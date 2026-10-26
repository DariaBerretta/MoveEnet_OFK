#!/usr/bin/env python3
"""HOST runner for paired Move-IIT energy and real-time-factor experiments.

Reuses experiments/ExpEnergy/measure_energy.py (RAPL package + NVML device).
Energy runs deliberately disable pose CSV/video output, like the established
final energy protocol. RTF uses the same wall time measured by the energy meter.
The input-span denominator is obtained from the first and last YARP data.log
packet timestamps (microsecond precision). The ExpA offlineLoader range often
prints both Unix timestamps as the same rounded scientific-notation value;
using that rounded line would yield a zero duration. The physical input span
remains the same definition used in ExpFinal, with a higher-precision source.
Run only after ExpA finishes, with the GPU otherwise idle.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import math
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
MEASURE_SCRIPT = HERE / 'measure_energy.py'
SAMPLES_CSV = HERE / 'moveiit_energy_samples.csv'
DATA_ROOT = '/data/move-iit-it_testing'
EXPA = '/data/MoveEnet_OFK_results/Final_ROI_Gain1/Accuracy/moveiit/ExpA'
BINARY = '/workspace/moveEnetFlow/build/moveEnetOFK_offline'
CKPT = '/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth'
VENV = '/opt/venv'
BASE_PATH = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
START_RE = re.compile(r'First event timestamp \(absolute\):\s*([-+\d.eE]+)')
DURATION_SOURCE = 'YARP_data_log_first_last_AE_packet_timestamp'
PROTOCOL_VERSION = 'moveiit_energy_rtf_expFinal_aligned_v3_exact_packet_span'

CASES = (
    ('movenet_50hz', '0.02', False),
    ('movenet_5hz', '0.2', False),
    ('moveenetofk_5hz', '0.2', True),
)


def cmdrun(argv, check=True):
    r = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, check=False)
    if check and r.returncode:
        raise RuntimeError('{}\nSTDERR: {}'.format(shlex.join(argv), r.stderr[-2000:]))
    return r.stdout.strip()


def docker(container, *args):
    return cmdrun(['docker', 'exec', container, *args])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load_samples(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise RuntimeError('Empty or duplicate sample IDs')
    for r in rows:
        if r['subject'] not in ('luna', 'simon') or not re.fullmatch(r'[a-z]+(?:-[a-z]+)?_[fs]\d+', r['sequence']):
            raise RuntimeError('Unexpected subject or sequence: {}'.format(r))
        r['event_file'] = '{}/{}/{}/atis-s/data.log'.format(DATA_ROOT, r['subject'], r['sequence'])
        gt_folder = 'P1_luna' if r['subject'] == 'luna' else 'P11_simon'
        r['gt_file'] = '{}/GT/{}/{}_joint_projections.csv'.format(DATA_ROOT, gt_folder, r['sequence'])
        stem = '{}__{}__atis-s'.format(r['subject'], r['sequence'])
        r['expa_csv'] = '{}/raw/fp_0p005/np_0p2/{}_moveenet_ofk_np_0p2_fp_0p005.csv'.format(EXPA, stem)
        r['expa_log'] = '{}/logs/{}_moveenet_ofk_np_0p2_fp_0p005.log'.format(EXPA, stem)
    return rows


# Read timestamps from ATIS packet headers, not the low-precision summary line.
# This code runs inside the container via Python 3.8 and reads only the
# beginning and (at most) the tail of the potentially large YARP log.
YARP_PACKET_SPAN_SCRIPT = r'''
import json
import os
import re
import sys

path = sys.argv[1]
# Packet headers are text; event payloads may contain encoded/binary data.
header = re.compile(rb'(?:^|\n)([0-9]+)[ \t]+([0-9]{10}\.[0-9]+)[ \t]+AE[ \t]+([0-9]+)[ \t]+')
with open(path, 'rb') as f:
    initial = f.read(8192)
    match = header.search(initial)
    if match is None or match.start() != 0:
        raise RuntimeError('No valid first ATIS packet header: ' + path)
    start_id = int(match.group(1))
    start_ts = match.group(2).decode('ascii')
    size = os.fstat(f.fileno()).st_size
    # The last packet can be longer than the initial read window.
    # Grow the EOF window only if we cannot find a full header yet.
    window = 262144
    last = None
    while True:
        f.seek(max(0, size - window))
        tail = f.read()
        candidates = list(header.finditer(tail))
        if candidates:
            last = candidates[-1]
            if int(last.group(1)) > start_id:
                break
        if window >= size:
            raise RuntimeError('No valid last ATIS packet header: ' + path)
        window *= 2
    end_id = int(last.group(1))
    end_ts = last.group(2).decode('ascii')
print(json.dumps({'first_packet_id': start_id, 'last_packet_id': end_id,
                  'first_timestamp': start_ts, 'last_timestamp': end_ts}))
'''


def get_duration(container, r):
    """First-to-last input packet timestamp, in seconds, without float rounding.

    The same *input-span definition* as ExpFinal is used, but data.log packet
    headers preserve microseconds even when offlineLoader's displayed range
    is rounded from e.g. 1.69815e+09 to 1.69815e+09.
    """
    from decimal import Decimal
    for key in ('event_file', 'gt_file', 'expa_csv', 'expa_log'):
        docker(container, 'test', '-s', r[key])
    log = docker(container, 'cat', r['expa_log'])
    if 'Finished processing event file.' not in log:
        raise RuntimeError('Incomplete ExpA replay log: {}'.format(r['expa_log']))

    raw = docker(container, VENV + '/bin/python', '-c',
                 YARP_PACKET_SPAN_SCRIPT, r['event_file'])
    parsed = json.loads(raw)
    start = Decimal(parsed['first_timestamp'])
    stop = Decimal(parsed['last_timestamp'])
    if not (start.is_finite() and stop.is_finite() and stop > start):
        raise RuntimeError('Invalid exact packet span in {}'.format(r['event_file']))
    # Corroborate against the explicit high-precision first timestamp in ExpA.
    first_event = START_RE.findall(log)
    if first_event and abs(Decimal(first_event[-1]) - start) > Decimal('0.02'):
        raise RuntimeError('ATIS first packet disagrees with ExpA first event timestamp: {}'.format(r['event_file']))
    # Validate packet index consistency where replay logs expose the count.
    packet_count = re.search(r'([0-9]+) packets loaded with', log)
    if packet_count is not None:
        expected_count = int(packet_count.group(1))
        observed_count = parsed['last_packet_id'] - parsed['first_packet_id'] + 1
        if observed_count != expected_count:
            raise RuntimeError('ATIS packet IDs/count mismatch in {}: {} vs {}'.format(
                r['event_file'], observed_count, expected_count))
    first = docker(container, 'head', '-n', '2', r['expa_csv']).splitlines()
    if len(first) != 2 or not first[0].startswith('timestamp,'):
        raise RuntimeError('Invalid completed ExpA CSV: {}'.format(r['expa_csv']))
    return {'duration_s': float(stop - start),
            'input_start_s': float(start), 'input_end_s': float(stop),
            'duration_source': DURATION_SOURCE,
            'source_log': r['expa_log']}


def init_session(path, config, samples, durations, resume):
    path.mkdir(parents=True, exist_ok=True)
    confp = path / 'config.json'
    if confp.exists():
        old = json.loads(confp.read_text())
        if old != config:
            raise RuntimeError('Existing session uses another configuration: {}. Choose another --session-dir.'.format(path))
        if not resume:
            raise RuntimeError('Session already exists. Pass --resume or use another --session-dir.')
    else:
        if (path / 'energy_measurements.csv').exists():
            raise RuntimeError('Untracked energy_measurements.csv exists; choose a new --session-dir')
        confp.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    (path / 'runs').mkdir(exist_ok=True)
    manifest = path / 'run_manifest.csv'
    if not manifest.exists():
        with manifest.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['dataset','sample_id','subject','sequence','camera','motion_class',
                        'model','repetition','order_position','net_period','net_hz',
                        'flow_period','output_period','status','energy_csv','log_path'])
    durations_file = path / 'samples_durations.csv'
    if durations_file.exists():
        with durations_file.open(newline='') as f:
            recorded = {r['sample_id']: (float(r['duration_s']), float(r['input_start_s']), float(r['input_end_s']), r['duration_source']) for r in csv.DictReader(f)}
        updated = {sid: (v['duration_s'], v['input_start_s'], v['input_end_s'], v['duration_source']) for sid, v in durations.items()}
        if set(recorded) != set(updated) or any(not (all(math.isclose(recorded[sid][i], updated[sid][i], rel_tol=0, abs_tol=1e-6) for i in range(3)) and recorded[sid][3] == updated[sid][3]) for sid in updated):
            raise RuntimeError('Event input spans changed since the first run. Use a new --session-dir.')
    with durations_file.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'subject', 'sequence', 'event_file',
                                                'gt_file', 'expa_csv', 'duration_s', 'input_start_s', 'input_end_s', 'duration_source', 'source_log'],
                                extrasaction='ignore')
        writer.writeheader()
        for r in samples:
            writer.writerow({**r, **{k: ('{:.9f}'.format(v) if isinstance(v, float) else v) for k, v in durations[r['sample_id']].items()}})


def completed_keys(csv_file):
    done = set()
    if csv_file.exists():
        with csv_file.open(newline='') as f:
            for row in csv.DictReader(f):
                if row.get('status') == 'OK' and row.get('exit_code') in ('0', '0.0'):
                    log = Path(row.get('log_path', ''))
                    if not log.is_file() or 'Finished processing event file.' not in log.read_text(errors='replace'):
                        raise RuntimeError('OK energy record has incomplete replay; cannot safely resume: {}'.format(log))
                    done.add((row['sample_id'], row['model'], row['order_id']))
    return done


def build_command(container, device, event_file, model):
    name, net, ofk = model
    args = ['docker', 'exec', container, 'env',
            'VIRTUAL_ENV=' + VENV, 'PATH=' + VENV + '/bin:' + BASE_PATH,
            BINARY, '--data_file', event_file, '--moveiit',
            '--net_period', net, '--flow_period', '0.005',
            '--output_period', '0.005', '--roi', '35',
            '--pu', '0.77', '--muD', '0.06', '--muV', '0.97',
            '--use_lc', 'true', '--checkpoint_path', CKPT,
            '--device', device, '--no_csv', '--no_video']
    if not ofk:
        args.append('--moveenet_only')
    return args


def run():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--container', default='moveEnetOFK')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--gpu-index', type=int, default=0)
    ap.add_argument('--python', default=sys.executable, help='Host Python with pynvml, Python >=3.10')
    ap.add_argument('--session-dir', type=Path, default=Path.home() / 'data/MoveEnet_OFK_results/Energy/moveiit_energy_rtf_roi35_gain1_v2')
    ap.add_argument('--samples-csv', type=Path, default=SAMPLES_CSV)
    ap.add_argument('--sample', default='', help='Sample ID, e.g. M01; default all')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--cooldown', type=float, default=20.)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.repeats < 1 or args.cooldown < 0 or args.gpu_index < 0:
        ap.error('invalid repeats/cooldown/gpu-index')
    samples = load_samples(args.samples_csv)
    if args.sample:
        samples = [s for s in samples if s['sample_id'] == args.sample]
        if not samples:
            ap.error('sample ID not found')
    tasks = len(samples) * len(CASES) * args.repeats
    print('move-iit-hpe energy + RTF | {} samples x {} methods x {} repeats = {} runs'.format(
        len(samples), len(CASES), args.repeats, tasks), flush=True)
    print('ROI=35, gain=1, output=200Hz, flow=200Hz, native 1280x720')
    for name, net, ofk in CASES:
        print('  {}: net={:.0f}Hz, OFK={}'.format(name, 1 / float(net), ofk))
    print('Session: {}'.format(args.session_dir))
    if args.dry_run:
        print('EXAMPLE COMMAND:')
        print(shlex.join(build_command(args.container, args.device, samples[0]['event_file'], CASES[0])))
        print('DRY RUN: no jobs or energy measurements executed')
        return 0
    if not MEASURE_SCRIPT.is_file():
        raise RuntimeError('Copy this script next to original measure_energy.py: {}'.format(MEASURE_SCRIPT))
    cmdrun([args.python, '-c', 'import pynvml; import sys; assert sys.version_info >= (3,10)'])
    rapl_path = Path('/sys/class/powercap/intel-rapl:0/energy_uj')
    try:
        int(rapl_path.read_text().strip())
    except (OSError, ValueError):
        raise RuntimeError('Host Intel RAPL energy counter is missing or unreadable')
    if cmdrun(['docker', 'inspect', '-f', '{{.State.Running}}', args.container]) != 'true':
        raise RuntimeError('Docker container is not running: ' + args.container)
    docker(args.container, 'test', '-x', BINARY)
    docker(args.container, 'test', '-f', CKPT)
    docker(args.container, 'test', '-x', VENV + '/bin/python')
    docker(args.container, 'bash', '-lc', 'yarp where >/dev/null 2>&1')
    for s in samples:
        docker(args.container, 'test', '-s', s['event_file'])
        docker(args.container, 'test', '-s', s['gt_file'])

    if args.preflight_only:
        print('PREFLIGHT OK: input files, checkpoint, executable, YARP, host Python/pynvml found.')
        durations = {s['sample_id']: get_duration(args.container, s) for s in samples}
        print('Event-time spans confirmed from exact YARP AE packet headers (input-span definition as ExpFinal):')
        for sid, d in durations.items():
            print('  {}: {:.6f} s'.format(sid, d['duration_s']))
        return 0

    # Never launch an energy benchmark while another job is using its MoveNet sidecar.
    sidecars = docker(args.container, 'bash', '-lc',
                      "pgrep -af '[m]ovenet_online.py|[Y]oloPose_yarp_server.py|[e]ventPointPose_yarp_server.py' || true")
    if sidecars:
        raise RuntimeError('A sidecar is currently running (possibly ExpA). Finish accuracy runs first:\n' + sidecars)
    apps = cmdrun(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], check=False)
    if apps:
        print('WARNING: GPU compute processes detected. Energy may be confounded:\n' + apps, file=sys.stderr)

    durations = {s['sample_id']: get_duration(args.container, s) for s in samples}
    sha_bin = docker(args.container, 'sha256sum', BINARY).split()[0]
    sha_ckpt = docker(args.container, 'sha256sum', CKPT).split()[0]
    config = {'protocol': PROTOCOL_VERSION, 'dataset': 'move-iit-hpe',
              'roi': 35, 'gain': 1.0, 'kf': [0.77, 0.06, 0.97],
              'net_only_50hz': True, 'net_only_5hz': True, 'ofk_5hz': True,
              'flow_period_s': .005, 'output_period_s': .005,
              'repeats': args.repeats, 'cooldown_s': args.cooldown,
              'container': args.container, 'device': args.device,
              'gpu_index': args.gpu_index, 'binary_sha256': sha_bin,
              'checkpoint_sha256': sha_ckpt,
              'sample_manifest_sha256': sha256(args.samples_csv),
              'network_period_s': {m[0]: float(m[1]) for m in CASES},
              'use_latency_compensation': True,
              'input_resolution': [1280, 720],
              'measure_script_sha256': sha256(MEASURE_SCRIPT),
              'sample_ids': [s['sample_id'] for s in samples],
              'event_input_span_method': DURATION_SOURCE, 'bootstrap_replicates': 5000, 'bootstrap_seed': 20260911}
    init_session(args.session_dir, config, samples, durations, args.resume)
    energy_csv = args.session_dir / 'energy_measurements.csv'
    done = completed_keys(energy_csv)
    print('Starting energy measurements; complete-recording wall time also measures RTF numerator.', flush=True)
    if args.cooldown and len(done) < tasks:
        print('Initial cooldown: {} s'.format(args.cooldown), flush=True)
        time.sleep(args.cooldown)
    # Balanced method ordering: repetition + sample-index rotates the methods.
    for si, s in enumerate(samples):
        for rep in range(1, args.repeats + 1):
            rotation = (si + rep - 1) % len(CASES)
            ordered = CASES[rotation:] + CASES[:rotation]
            for order_pos, model in enumerate(ordered, start=1):
                key = (s['sample_id'], model[0], 'R{}'.format(rep))
                if key in done:
                    print('SKIP (already OK): {}'.format(key), flush=True)
                    continue
                log = args.session_dir / 'runs' / '{}_{}_r{:02d}.log'.format(s['sample_id'], model[0], rep)
                cmd = build_command(args.container, args.device, s['event_file'], model)
                # Validate complete processing from the log after every attempt.
                measure = [args.python, str(MEASURE_SCRIPT), '--csv', str(energy_csv),
                           '--log', str(log), '--label', '{}_{}_{}'.format(*key),
                           '--dataset', 'moveiit', '--sample-id', s['sample_id'],
                           '--model', model[0], '--order-id', key[2],
                           '--order-position', str(order_pos),
                           '--net-period', model[1],
                           '--flow-period', '0.005' if model[2] else '',
                           '--gpu-index', str(args.gpu_index), '--', *cmd]
                print('RUN {} {}/{} - {}'.format(s['sample_id'], rep, args.repeats, model[0]), flush=True)
                completed = subprocess.run(measure, check=False)
                log_complete = log.is_file() and 'Finished processing event file.' in log.read_text(errors='replace')
                status = 'OK' if completed.returncode == 0 and log_complete else 'FAILED'
                with (args.session_dir / 'run_manifest.csv').open('a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['moveiit', s['sample_id'], s['subject'], s['sequence'],
                                'atis-s', s['sequence'].rsplit('_', 1)[-1][0], model[0],
                                rep, order_pos, model[1], '{:.6f}'.format(1/float(model[1])),
                                '0.005' if model[2] else '', '0.005', status,
                                str(energy_csv), str(log)])
                if completed.returncode:
                    raise RuntimeError('Energy measurement failed (exit {}). Log: {}'.format(completed.returncode, log))
                if not log_complete:
                    raise RuntimeError('Measurement row exists but replay not completed: {}'.format(log))
                done.add(key)
                if args.cooldown:
                    time.sleep(args.cooldown)
    print('Completed. Energy CSV: {}'.format(energy_csv))
    print('Duration manifest: {}'.format(args.session_dir / 'samples_durations.csv'))
    print('Analyze with: python3 analyze_moveiit_energy_rtf.py --session-dir {}'.format(args.session_dir))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(run())
    except (RuntimeError, OSError, ValueError) as exc:
        print('ERROR: {}'.format(exc), file=sys.stderr)
        sys.exit(1)
