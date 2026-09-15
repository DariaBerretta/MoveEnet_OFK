#!/usr/bin/env python3
"""Container-side preflight, protected replay and metric evaluation.

Invoked by finish_experiment.py. It never measures energy. Every executable and
its sidecars run in a dedicated POSIX process group with a finite wall deadline.
"""
from __future__ import annotations
import argparse
import contextlib
import ctypes
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

from final_common import (AuditError, atomic_json, require, sha256_file, options,
                          completed_log, log_diagnostics, KF_RE, close, hash_obj)
from final_metrics import import_gt_parser, load_gt, evaluate_prediction

ACTIVE_PATTERNS = ('moveEnetOFK_offline', 'moveEnetOFK_offline_mpjpe', 'OpenPose_offline',
                   'YoloPose_offline', 'eventPointPose_offline', 'movenet_online.py',
                   'YoloPose_yarp_server.py', 'eventPointPose_yarp_server.py')

def busy_processes():
    rows = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            argv = p.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            # Inspect exact basenames, not substrings in a shell/checker string.
            if any(Path(a).name in ACTIVE_PATTERNS for a in argv if a):
                rows.append({'pid': int(p.name), 'argv': argv})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return rows

def run_text(argv, timeout=30):
    r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    require(r.returncode == 0, 'Comando fallito: {}\n{}'.format(argv, r.stderr[-3000:]))
    return r.stdout

def readable(path):
    p = Path(path)
    require(p.is_file() and os.access(str(p), os.R_OK), 'File non leggibile: ' + str(p))
    return p

def file_record(path, full_hash=True):
    p = readable(path); stat = p.stat()
    d = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if full_hash:
        d['sha256'] = sha256_file(p)
    return d

def probe_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        require(cap.isOpened(), 'Video non apribile: ' + str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = float(cap.get(cv2.CAP_PROP_FPS))
        require(n > 1 and fps > 0, 'Metadati video non validi: ' + str(path))
        # No model is run; read first and last frames only to check decoding.
        ok, first = cap.read()
        require(ok and first is not None, 'Primo frame video illeggibile')
        first_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        seek_ok = cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
        ok, last = cap.read() if seek_ok else (False, None)
        last_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if ok else 0.0
        span = last_ts - first_ts
        method = 'OpenCV_last_minus_first_frame_timestamp'
        if not span > 0:
            # Explicit fallback; not silently claimed as measured timestamps.
            span = (n - 1) / fps
            method = 'frame_count_minus_one_over_fps_metadata_fallback'
        return {'input_span_s': float(span), 'duration_source': method,
                'video_frames': n, 'video_fps': fps,
                'video_frame_count_over_fps_s': n/fps}
    finally:
        cap.release()

def preflight(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    require(not busy_processes(), 'Altri metodi HPE ancora attivi: ' + repr(busy_processes()))
    import numpy
    import pandas
    import matplotlib
    import torch
    import yarp
    import ultralytics
    parser = import_gt_parser(plan['hpecore_root'])
    yarp.Network.init()
    require(torch.cuda.is_available(), 'CUDA non disponibile nel Python /opt/venv del container')
    require(yarp.Network.checkNetwork(5.0), 'Name server YARP non raggiungibile')
    files = {}; inputs = {}; durations = {}; gts = {}; sources = {}
    for job in plan['jobs']:
        p = readable(job['binary']); require(os.access(str(p), os.X_OK), 'Binario non eseguibile: ' + str(p))
        opts = options(job['base_argv'])
        required = [job['binary']]
        required.extend(opts[f] for f in ('--checkpoint_path','--yolo_model_path','--model_path',
                                         '--YoloPose_script','--EventPointPose_script') if opts.get(f))
        if job['model'] in ('movenet','moveenetofk'):
            required.append(plan['hpecore_root'] + '/example/movenet/movenet_online.py')
        if job['model'] == 'openpose':
            op_root = Path(opts['--op_model_path'])
            required += [str(op_root/'pose/body_25/pose_iter_584000.caffemodel'),
                         str(op_root/'pose/body_25/pose_deploy.prototxt')]
        for f in required:
            if f not in files:
                files[f] = file_record(f)
        # Freeze actual input identity without hashing multi-GB event logs.
        if job['data_file'] not in inputs:
            inputs[job['data_file']] = file_record(job['data_file'], full_hash=False)
        if job['gt_file'] not in gts:
            ts, xy = load_gt(job['gt_file'], parser)
            gts[job['gt_file']] = {'sha256': sha256_file(Path(job['gt_file'])),
                                  'n_gt_rows': len(ts), 'first_ts_s': float(ts[0]),
                                  'last_ts_s': float(ts[-1]), 'joint_shape': list(xy.shape)}
        source = Path(job['binary']).parent.parent / (Path(job['binary']).name + '.cpp')
        if source.is_file() and str(source) not in sources:
            sources[str(source)] = file_record(source)
        if job['data_file'] not in durations:
            if job['input_span_s_from_energy_log'] is not None:
                durations[job['data_file']] = {
                    'input_span_s': job['input_span_s_from_energy_log'],
                    'duration_source': 'energy_log_loader_last_minus_first_packet_timestamp'}
            else:
                require(Path(job['data_file']).suffix.lower() == '.mp4', 'Durata eventi non disponibile nel log')
                durations[job['data_file']] = probe_video(job['data_file'])
    python_sources = [Path(parser.__file__), Path(plan['hpecore_root'])/'datasets/utils/constants.py']
    for p in python_sources:
        sources[str(p)] = file_record(p)
    gpu_text = run_text(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'], 30)
    warnings_ = []
    expected_uuid = plan['audit']['hardware']['gpu_uuid'][0]
    require(expected_uuid in gpu_text, 'GPU dei run energetici non visibile nel container: ' + expected_uuid)
    expected_driver = plan['audit']['hardware']['nvidia_driver'][0]
    if expected_driver not in gpu_text:
        warnings_.append('Driver diverso da quello registrato nei run energia.')
    # Confirm the changed OpenPose scheduler is present in the visible source.
    op_sources = [p for p in sources if p.endswith('/OpenPose_offline.cpp')]
    for p in op_sources:
        txt = Path(p).read_text(errors='replace')
        require('next_output_ts' in txt, 'Sorgente OpenPose non contiene scheduler ZOH separato')
    # This inspection does not rebuild or modify any executable/library.
    linked_libraries = {}
    for binary in sorted({j['binary'] for j in plan['jobs']}):
        linked_libraries[binary] = run_text(['ldd', binary], 30)
        for library in re.findall(r'=>\s+(/\S+)', linked_libraries[binary]):
            low = Path(library).name.lower()
            if any(token in low for token in ('hpe', 'event-driven', 'libyarp')):
                if library not in files:
                    files[library] = file_record(library)
    # Hashes are stable across resumes; timestamps are kept separately for audit.
    identities = {k: v['sha256'] for k,v in files.items()}
    source_identities = {k: v['sha256'] for k,v in sources.items()}
    details = {'files': files, 'source_files': sources, 'input_files': inputs,
               'linked_libraries': linked_libraries,
               'ground_truth': gts, 'durations': durations, 'warnings': warnings_,
               'python': sys.executable, 'numpy': numpy.__version__, 'pandas': pandas.__version__,
               'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
               'matplotlib': matplotlib.__version__, 'gpu': gpu_text,
               'identity_hash': hash_obj({'files': identities, 'sources': source_identities,
                                        'inputs': inputs, 'gts': gts, 'python': sys.executable,
                                        'numpy': numpy.__version__, 'torch': torch.__version__,
                                        'ultralytics': ultralytics.__version__})}
    return details

def stop_group(pgid, grace=4.0):
    """The group is created by this worker, never found with a global pkill."""
    if pgid <= 1 or pgid == os.getpgrp():
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(.1)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)

def unregister_own_ports(log_text):
    # Only after stopping this job's process group, and only names from its log.
    ports = set(re.findall(r'Port (/[^\s|]+) active at', log_text))
    for p in ports:
        if p.startswith(('/YoloPose/', '/YoloPose_offline/', '/movenet/', '/moveEnetOFK_offline/',
                         '/EventPointPose/', '/EventPointPose_offline/')):
            with contextlib.suppress(Exception):
                subprocess.run(['yarp','name','unregister',p], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=5)

def run_job(job_path):
    job = json.loads(Path(job_path).read_text())
    attempt = Path(job['attempt_dir_container']); attempt.mkdir(parents=True, exist_ok=True)
    log = attempt/'run.log'; result_path = attempt/'result.json'
    cancel = attempt/'CANCEL'
    require(not cancel.exists(), 'Tentativo gia annullato: ' + str(attempt))
    require(not busy_processes(), 'Processi HPE gia attivi: ' + repr(busy_processes()))
    for p, expected in job['required_hashes'].items():
        require(sha256_file(Path(p)) == expected, 'File cambiato dopo preflight: ' + p)
    for p, expected in job.get('required_inputs', {}).items():
        actual = file_record(p, full_hash=False)
        require(actual == expected, 'Input cambiato dopo preflight: ' + p)
    interrupted = [False]
    def on_signal(signum, frame):
        interrupted[0] = True
    signal.signal(signal.SIGINT, on_signal); signal.signal(signal.SIGTERM, on_signal)
    try:
        ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)  # Linux PR_SET_CHILD_SUBREAPER
    except Exception:
        pass
    result = {'job_id': job['job_id'], 'status': 'FAILED', 'exit_code': None,
              'config_hash': job['config_hash'], 'identity_hash': job['identity_hash']}
    process = None; reason = ''; start = time.perf_counter()
    try:
        with log.open('w', buffering=1) as output:
            output.write('FINAL_REPLAY_COMMAND ' + json.dumps(job['replay_argv']) + '\n')
            output.flush()
            env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'
            process = subprocess.Popen(job['replay_argv'], stdout=output, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, env=env, start_new_session=True)
            atomic_json(attempt/'process.json', {'pid': process.pid, 'pgid': process.pid,
                                                 'worker_pid': os.getpid(), 'job_id': job['job_id']})
            while process.poll() is None:
                elapsed = time.perf_counter() - start
                if interrupted[0] or cancel.exists():
                    reason = 'CANCELLED'; break
                if elapsed > job['timeout_s']:
                    reason = 'TIMEOUT'; break
                time.sleep(.2)
            if reason:
                stop_group(process.pid); process.wait(timeout=10)
            rc = process.wait(); result['exit_code'] = rc
        result['accuracy_replay_wall_seconds'] = time.perf_counter() - start
        # This duration includes pose logging. It is NOT the runtime metric used
        # in the paper; that metric comes from energy.wall_seconds.
        require(not reason, reason + ': deadline/cancellation, nessuna accuracy valida registrata')
        require(rc == 0, 'Il binario ha restituito exit code ' + str(rc))
        text = log.read_text(errors='replace')
        require(completed_log(text, job['model']), 'Manca completamento normale nel log')
        if job['model'] == 'moveenetofk':
            matches = KF_RE.findall(text)
            require(len(matches) == 1 and all(close(a,b) for a,b in zip(matches[0], job['kf'])),
                    'Parametri KF del replay diversi da quelli misurati')
        diag = log_diagnostics(text, job['model'])
        if job['model'] == 'eventpointpose':
            expected_events = re.findall(r'with\s+(\d+)\s+total events', text)
            read_events = re.findall(r'Events read:\s*(\d+)', text)
            require(expected_events and read_events and expected_events[-1] == read_events[-1],
                    'EventPointPose non ha letto tutti gli eventi')
        parser = import_gt_parser(job['hpecore_root'])
        metrics = evaluate_prediction(job['prediction_container'], job['gt_file'], parser,
                                      job['output_period_s'], job['input_span_s'])
        if job['model'] in ('openpose', 'yolo', 'eventpointpose'):
            require(abs(metrics['median_output_period_s'] - job['output_period_s']) <= 2e-6
                    and metrics['max_output_gap_s'] <= job['output_period_s'] + 2e-6,
                    'CSV non compatibile con la cadenza ZOH richiesta. Controllare binario/scheduler; nessuna riscrittura fittizia.')
        metrics.update({k: job[k] for k in ('job_id','dataset','sample_id','model','config_hash',
                                            'net_period_s','flow_period_s','output_period_s')})
        metrics.update(diagnostics=diag, identity_hash=job['identity_hash'], kf=job['kf'])
        result['metrics'] = metrics; result['status'] = 'OK'
    except Exception as exc:
        result['error'] = '{}: {}'.format(type(exc).__name__, exc)
        result['traceback'] = traceback.format_exc()
        if reason:
            result['status'] = reason
    finally:
        if process is not None:
            stop_group(process.pid, grace=1.0)
            with contextlib.suppress(Exception):
                process.wait(timeout=5)
        # Reap orphaned child processes when subreaper is supported.
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break
        if result['status'] != 'OK' and log.exists():
            unregister_own_ports(log.read_text(errors='replace'))
        atomic_json(result_path, result)
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['preflight','run-job','analysis'])
    p.add_argument('path')
    a = p.parse_args()
    try:
        if a.action == 'preflight':
            out = preflight(a.path)
            print('FINAL_JSON:' + json.dumps(out, allow_nan=False))
            return 0
        if a.action == 'analysis':
            os.environ.setdefault('MPLBACKEND', 'Agg')
            from final_analysis import build_analysis
            build_analysis(Path(a.path)); return 0
        lock_path = Path('/tmp/moveenet_final_accuracy.lock')
        with lock_path.open('a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run_job(a.path)
        print('FINAL_JSON:' + json.dumps({'status': result['status'], 'job_id': result['job_id']}))
        return 0 if result['status'] == 'OK' else 2
    except Exception as exc:
        print('FINAL_ERROR: {}: {}'.format(type(exc).__name__, exc), file=sys.stderr)
        traceback.print_exc()
        return 2

if __name__ == '__main__':
    sys.exit(main())
