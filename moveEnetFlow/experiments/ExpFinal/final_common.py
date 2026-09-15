"""Read-only energy audit and replay-plan construction (Python >= 3.8).

No shell evaluation, no GPU operations, no writes to the original energy session.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
from collections import Counter, defaultdict
from typing import Any, Dict, List

VERSION = '1.0.0'
REVIEWED_COMMIT = 'dbc218d4cd24056280341115163b7cc468a65d89'
MODELS = {'h36m': ('movenet', 'moveenetofk', 'openpose', 'yolo'),
          'dhp19': ('movenet', 'moveenetofk', 'eventpointpose')}
# This is the completed protocol shown in the user's run_manifest, not the
# inconsistent comments/defaults of an older runner.
EXPECTED_NET = {('h36m', 'movenet'): .02, ('h36m', 'moveenetofk'): .2,
                ('h36m', 'openpose'): .02, ('h36m', 'yolo'): .02,
                ('dhp19', 'movenet'): .005, ('dhp19', 'moveenetofk'): .2,
                ('dhp19', 'eventpointpose'): .005}
EXPECTED_KF = {'h36m': (.77, .06, .97), 'dhp19': (.36, .01, .01)}
BINS = {'movenet': 'moveEnetOFK_offline', 'moveenetofk': 'moveEnetOFK_offline',
        'openpose': 'OpenPose_offline', 'yolo': 'YoloPose_offline',
        'eventpointpose': 'eventPointPose_offline'}
OUTPUT_FLAG = {'movenet': '--output_csv_f', 'moveenetofk': '--output_csv_f',
               'openpose': '--output_csv', 'yolo': '--output_csv',
               'eventpointpose': '--output_csv'}
NUM = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
KF_RE = re.compile(r'KF parameters:\s*pu\s*=\s*('+NUM+r')\s+muD\s*=\s*('+NUM+r')\s+muV\s*=\s*('+NUM+r')')
SPAN_RE = re.compile(r'Timestamps range from\s+('+NUM+r')\s+to\s+('+NUM+r')')

class AuditError(RuntimeError):
    pass

def require(condition, message):
    if not condition:
        raise AuditError(message)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def hash_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def atomic_json(path: Path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(str(tmp), str(path))

def write_csv(path: Path, rows: List[dict], fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    os.replace(str(tmp), str(path))

def read_csv(path: Path) -> List[dict]:
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        r = csv.DictReader(f)
        require(r.fieldnames is not None, 'CSV senza header: ' + str(path))
        rows = list(r)
    require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
            'CSV incompleto/malformato: ' + str(path))
    return rows

def number(value, name, positive=False):
    try:
        x = float(value)
    except (ValueError, TypeError):
        raise AuditError('Valore numerico non valido per {}: {!r}'.format(name, value))
    require(math.isfinite(x), '{} non finito'.format(name))
    require(x > 0 if positive else x >= 0, '{} fuori intervallo: {}'.format(name, x))
    return x

def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-9)

def options(argv):
    """Parse binary flags, refusing duplicates. Values remain strings."""
    result = {}; i = 0
    while i < len(argv):
        flag = argv[i]
        require(flag.startswith('--'), 'Argomento inatteso: ' + flag)
        require('=' not in flag, 'Usare opzioni --flag value, non --flag=value: ' + flag)
        require(flag not in result, 'Opzione duplicata: ' + flag)
        if i + 1 < len(argv) and not argv[i + 1].startswith('--'):
            result[flag] = argv[i + 1]; i += 2
        else:
            result[flag] = None; i += 1
    return result

def replace_option(argv, flag, value=None):
    result = []; i = 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1
            if i < len(argv) and not argv[i].startswith('--'):
                i += 1
        else:
            result.append(argv[i]); i += 1
    if value is not False:
        result.append(flag)
        if value is not None:
            result.append(str(value))
    return result

def parse_recorded_command(command: str, model: str, container: str):
    args = shlex.split(command)
    require(len(args) > 4 and args[:2] == ['docker', 'exec'],
            'Comando energetico non supportato (atteso docker exec): ' + command)
    require(args[2] == container, 'Container registrato diverso da quello selezionato: ' + args[2])
    indices = [i for i, a in enumerate(args) if PurePosixPath(a).name == BINS[model]]
    require(len(indices) == 1, 'Binario non identificabile: ' + command)
    idx = indices[0]
    require(idx >= 3, 'Posizione binario non valida')
    prefix = args[3:idx]
    require(prefix and prefix[0] == 'env', 'Atteso prefisso env registrato: ' + command)
    # Restrict prefix to env assignments and the historical -u VIRTUAL_ENV.
    i = 1
    while i < len(prefix):
        token = prefix[i]
        if token == '-u':
            require(i + 1 < len(prefix) and prefix[i + 1] == 'VIRTUAL_ENV', 'env -u non supportato')
            i += 2
        else:
            require(re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', token) is not None,
                    'Prefisso env non supportato: ' + token)
            i += 1
    opt = options(args[idx + 1:])
    for flag in ('--data_file', '--net_period', '--output_period'):
        require(opt.get(flag) is not None, 'Manca ' + flag + ' nel comando energetico')
    require('--no_csv' in opt, 'Il protocollo atteso misura con --no_csv')
    require(model == 'eventpointpose' or '--no_video' in opt, 'Atteso --no_video nel protocollo Energy')
    require('--vis' not in opt, 'La visualizzazione non fa parte del protocollo Energy finale')
    for flag in ('--gpu_file', '--pwrjlr_file', '--dump_dir', '--dump_debug_dir'):
        require(not opt.get(flag), 'Output collaterale non previsto dal protocollo: ' + flag)
    require(('--moveenet_only' in opt) == (model == 'movenet'), 'Flag MoveEnet-only incoerente')
    return {'binary': args[idx], 'prefix': prefix, 'argv': args[idx + 1:], 'options': opt}

def log_file_for(row, session):
    p = Path(row.get('log_path', ''))
    candidate = Path(session) / 'runs' / p.name
    if candidate.is_file():
        return candidate
    require(p.is_file(), 'Log energetico mancante: ' + str(candidate))
    return p

def log_diagnostics(text, model):
    spans = SPAN_RE.findall(text)
    counts = {}
    patterns = {
      'movenet': (r'MoveNet (?:inference requests|inferences completed):\s*(\d+)', r'MoveNet valid (?:predictions|inferences):\s*(\d+)'),
      'moveenetofk': (r'MoveNet (?:inference requests|inferences completed):\s*(\d+)', r'MoveNet valid (?:predictions|inferences):\s*(\d+)'),
      'openpose': (r'OpenPose inferences completed:\s*(\d+)', r'OpenPose valid inferences:\s*(\d+)'),
      'yolo': (r'YOLO inferences completed:\s*(\d+)', r'YOLO valid inferences:\s*(\d+)'),
      'eventpointpose': (r'PointNet requests:\s*(\d+)', r'Valid PointNet inferences:\s*(\d+)')}
    for name, pat in zip(('request_count', 'valid_inference_count'), patterns[model]):
        matches = re.findall(pat, text)
        counts[name] = int(matches[-1]) if matches else None
    if spans:
        counts.update(input_start_s=float(spans[-1][0]), input_end_s=float(spans[-1][1]),
                      input_span_s=float(spans[-1][1])-float(spans[-1][0]))
    frames = re.findall(r'(?:Frames processed:|End of video reached after)\s*(\d+)', text)
    if frames:
        counts['frames_processed'] = int(frames[-1])
    errors = re.findall(r'Server ERROR responses:\s*(\d+)', text)
    if errors:
        counts['server_errors'] = int(errors[-1])
    return counts

def completed_log(text, model):
    if re.search(r'Termination signal received|context canceled|Traceback \(most recent call last\)', text):
        return False
    if model in ('movenet', 'moveenetofk'):
        return 'Finished processing event file.' in text
    if model in ('openpose', 'yolo'):
        return 'End of video reached' in text
    if model == 'eventpointpose':
        return ('Events read:' in text and 'PointNet requests:' in text
                and re.search(r'Server ERROR responses:\s*0(?:\s|$)', text) is not None
                and 'EventPointPose request failed:' not in text)
    return False

def derive_gt(data_file, dataset):
    p = PurePosixPath(data_file)
    if dataset == 'h36m' and p.suffix.lower() == '.mp4':
        require(p.parent.name == 'rgb', 'Directory RGB inattesa: ' + str(p))
        return str(p.parent.parent / 'events' / p.stem / 'ch0GT200Hzskeleton' / 'data.log')
    require(p.name == 'data.log', 'Input eventi inatteso: ' + str(p))
    match = re.fullmatch(r'ch(\d+)dvs', p.parent.name)
    require(match is not None, 'Canale DVS non riconosciuto: ' + str(p))
    ch = match.group(1)
    require((dataset == 'h36m' and ch == '0') or (dataset == 'dhp19' and ch in ('2', '3')),
            'Canale/dataset incompatibili: ' + str(p))
    return str(p.parent.parent / ('ch' + ch + 'GT200Hzskeleton') / 'data.log')

def audit_session(session: Path, container='moveEnetOFK', repeats=5):
    """Strictly validate the existing session. Extra FAILED attempts are retained
    in the audit but never averaged. Exact duplicated CSV rows are collapsed;
    two distinct successful attempts of one repetition are not chosen silently.
    """
    session = Path(session).resolve()
    energy = read_csv(session / 'energy_measurements.csv')
    manifest = read_csv(session / 'run_manifest.csv')
    require(energy and manifest, 'Sessione vuota')
    success = {}; failed = []; duplicate_rows = 0; issues = []
    for i, row in enumerate(energy, 2):
        if row.get('status', '').upper() != 'OK' or row.get('exit_code') not in ('0', '0.0'):
            failed.append({'line': i, 'label': row.get('label'), 'status': row.get('status')}); continue
        ds = row['dataset'].lower(); ds = 'h36m' if ds == 'eh36m' else ds
        model = row['model'].lower(); sample = row['sample_id']
        require(ds in MODELS and model in MODELS[ds], 'Dataset/metodo inatteso nel CSV')
        require(re.fullmatch('[HD]\\d{2}', sample) is not None, 'sample_id non valido: ' + sample)
        require(sample[0] == ('H' if ds == 'h36m' else 'D'), 'sample_id e dataset incoerenti')
        rep_match = re.fullmatch(r'R(\d+)', row.get('order_id', ''))
        require(rep_match is not None, 'order_id di ripetizione non riconosciuto: ' + row.get('order_id', ''))
        rep = int(rep_match.group(1)); key = (ds, sample, model, rep)
        require(1 <= rep <= repeats, 'Ripetizione fuori protocollo: ' + str(key))
        if key in success:
            if row == success[key]['original']:
                duplicate_rows += 1; continue
            raise AuditError('Due misure OK diverse per {}: nessuna scelta automatica.'.format(key))
        t = number(row['wall_seconds'], 'wall_seconds', True)
        cpu = number(row['cpu_package_energy_j'], 'cpu_package_energy_j')
        gpu = number(row['gpu_energy_j'], 'gpu_energy_j')
        total = number(row['measured_compute_energy_j'], 'measured_compute_energy_j', True)
        require(math.isclose(total, cpu + gpu, abs_tol=.0001, rel_tol=1e-6), 'CPU+GPU non coincide con totale: ' + str(key))
        parsed = parse_recorded_command(row['command'], model, container)
        opt = parsed['options']
        np_ = number(opt['--net_period'], '--net_period', True)
        op = number(opt['--output_period'], '--output_period', True)
        require(close(np_, row['net_period_s']), 'Periodo comando/CSV diverso: ' + str(key))
        require(close(np_, EXPECTED_NET[ds, model]) and close(op, .005),
                'Configurazione diversa dal protocollo finale 5Hz OFK: {} net={} out={}'.format(key, np_, op))
        if model in ('movenet', 'moveenetofk'):
            require(('--dhp19' in opt) == (ds == 'dhp19'), 'Flag --dhp19 incoerente: ' + str(key))
        fp = number(opt['--flow_period'], '--flow_period', True) if model == 'moveenetofk' else None
        require(fp is None or close(fp, .005), 'Flow period non 0.005: ' + str(key))
        if model == 'moveenetofk':
            require(close(fp, row['flow_period_s']), 'Flow period comando/CSV diverso')
        logp = log_file_for(row, session); text = logp.read_text(errors='replace')
        require(completed_log(text, model), 'Run OK senza evidenza di completamento o con errore: ' + str(logp))
        diagnostics = log_diagnostics(text, model)
        kf = None
        if model == 'moveenetofk':
            ks = KF_RE.findall(text)
            require(len(ks) == 1, 'Parametri KF non univoci nel log: ' + str(logp))
            kf = [float(x) for x in ks[0]]
            require(all(close(a,b) for a,b in zip(kf, EXPECTED_KF[ds])), 'Tuning KF errato: ' + str(logp))
        fingerprint = hash_obj({'binary': parsed['binary'], 'prefix': parsed['prefix'],
                                'options': opt, 'kf': kf})
        success[key] = {'original': row, 'dataset': ds, 'sample_id': sample, 'model': model,
                        'repetition': rep, 'parsed': parsed, 'net_period_s': np_,
                        'output_period_s': op, 'flow_period_s': fp, 'kf': kf,
                        'config_hash': fingerprint, 'log_path': str(logp),
                        'log_sha256': sha256_file(logp), 'diagnostics': diagnostics}
    required_keys = {(ds, prefix + str(i).zfill(2), model, r)
                     for ds, prefix in [('h36m','H'), ('dhp19','D')]
                     for i in range(1,13) for model in MODELS[ds] for r in range(1,repeats+1)}
    missing = required_keys - set(success)
    require(not missing and set(success) == required_keys,
            'Copertura energia incompleta: {} chiavi mancanti. Esempi: {}'.format(len(missing), sorted(missing)[:10]))
    man = defaultdict(list)
    for row in manifest:
        if row.get('status', '').upper() != 'OK':
            continue
        ds = 'h36m' if row['dataset'] == 'eh36m' else row['dataset']
        k = (ds, row['sample_id'], row['model'], int(row['repetition']))
        man[k].append(row)
    for key, item in success.items():
        require(key in man, 'Energia OK non presente nel manifest: ' + str(key))
        for m in man[key]:
            require(close(m['net_period'], item['net_period_s']) and close(m['output_period'], item['output_period_s']),
                    'Manifest/energia incoerenti: ' + str(key))
    groups = defaultdict(list)
    for key in sorted(success):
        item = success[key]; groups[key[:3]].append(item)
    jobs = []
    for key, items in sorted(groups.items()):
        require(len({x['config_hash'] for x in items}) == 1, 'Configurazione cambiata tra ripetizioni: ' + str(key))
        x = items[0]; opt = x['parsed']['options']
        if key[2] == 'eventpointpose':
            require(opt.get('--camera') in ('2','3'), 'Camera EPP non esplicita')
            require('/ch{}dvs/'.format(opt['--camera']) in opt['--data_file'], 'Camera EPP/input diverso')
        times = [float(i['original']['wall_seconds']) for i in items]
        median = sorted(times)[len(times)//2]
        # Report, never silently delete a successful energy measurement.
        if max(times) > max(60, 3 * median):
            issues.append('Runtime anomalo da esaminare: {} tempi={}'.format(key, times))
        span_values = [i['diagnostics'].get('input_span_s') for i in items]
        span_values = [s for s in span_values if s is not None]
        if span_values:
            require(all(close(s, span_values[0]) for s in span_values) and span_values[0] > 0,
                    'Durata input differente tra ripetizioni: ' + str(key))
        jobs.append({'job_id': '_'.join(key), 'dataset': key[0], 'sample_id': key[1], 'model': key[2],
                     'binary': x['parsed']['binary'], 'prefix': x['parsed']['prefix'],
                     'base_argv': x['parsed']['argv'], 'config_hash': x['config_hash'],
                     'source_command': x['original']['command'], 'kf': x['kf'],
                     'net_period_s': x['net_period_s'], 'flow_period_s': x['flow_period_s'],
                     'output_period_s': x['output_period_s'], 'data_file': opt['--data_file'],
                     'gt_file': derive_gt(opt['--data_file'], key[0]),
                     'input_span_s_from_energy_log': span_values[0] if span_values else None,
                     'timeout_s': max(300.0, 5.0 * max(times) + 120.0),
                     'energy_labels': [i['original']['label'] for i in items]})
    for ds in MODELS:
        require(len({x['gt_file'] for x in jobs if x['dataset'] == ds}) == 12,
                'Attese 12 sequenze-camera distinte per dataset: ' + ds)
        for sample in sorted({x['sample_id'] for x in jobs if x['dataset'] == ds}):
            gt = {x['gt_file'] for x in jobs if x['dataset'] == ds and x['sample_id'] == sample}
            require(len(gt) == 1, 'Input di metodi diversi non corrispondono allo stesso GT: ' + sample)
    hardware_fields = ('gpu_uuid', 'gpu_name', 'nvidia_driver')
    hardware = {f: sorted({i['original'][f] for i in success.values()}) for f in hardware_fields}
    require(all(len(v) == 1 for v in hardware.values()), 'GPU/driver cambiati nella sessione: ' + repr(hardware))
    report = {'version': VERSION, 'reviewed_repository_commit': REVIEWED_COMMIT,
              'successful_unique_runs': len(success), 'sequence_method_pairs': len(jobs),
              'repetitions': repeats, 'ignored_failed_attempts': failed,
              'exact_duplicate_rows_collapsed': duplicate_rows, 'warnings': issues,
              'hardware': hardware,
              'input_hashes': {n: sha256_file(session / n) for n in ('energy_measurements.csv', 'run_manifest.csv')},
              'historical_binary_hashes_available': False,
              'provenance_note': 'Historical energy runner did not record executable hashes. Reuse unchanged binaries; resolved KF is verified from every OFK log.'}
    return jobs, [x for x in success.values()], report

def replay_command(job, csv_path, server_log=None):
    argv = list(job['base_argv'])
    argv = replace_option(argv, '--no_csv', False)
    for flag in ('--output_csv', '--output_csv_f', '--output_file', '--latency_csv'):
        argv = replace_option(argv, flag, False)
    argv = replace_option(argv, OUTPUT_FLAG[job['model']], csv_path)
    if job['model'] == 'moveenetofk':
        for flag, v in zip(('--pu','--muD','--muV'), job['kf']):
            argv = replace_option(argv, flag, format(v, '.12g'))
    if job['model'] == 'eventpointpose' and server_log:
        argv = replace_option(argv, '--server_log', server_log)
    return list(job['prefix']) + [job['binary']] + argv
