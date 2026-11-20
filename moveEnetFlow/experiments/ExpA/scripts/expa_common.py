"""Shared Experiment A configuration, discovery, and execution metadata.

Configurations are JSON syntax (also valid YAML 1.2), so the runner needs no
optional YAML parser. Preserve input-relative paths as recording identifiers.
"""
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPA = HERE.parent
FLOW = EXPA.parent.parent
DEFAULT_RESULTS = '/data/MoveEnet_OFK_results/experiment_cleanup'
ALL_RATES = (2, 5, 10, 20, 50, 100, 200)
VALID = {'eh36m': ('moveenetofk', 'movenet', 'yolo', 'openpose'),
         'dhp19': ('moveenetofk', 'movenet', 'eventpointpose'),
         'moveiit': ('moveenetofk', 'movenet', 'eventpointpose')}


def read_config(dataset, config_dir=None):
    if dataset not in VALID:
        raise ValueError('Unsupported dataset: ' + dataset)
    path = (Path(config_dir) if config_dir else EXPA / 'config') / (dataset + '.yaml')
    config = json.loads(path.read_text(encoding='utf-8'))
    if config['dataset'] != dataset or tuple(config['methods']) != VALID[dataset]:
        raise ValueError('Inconsistent dataset/methods: ' + str(path))
    for method in config['methods']:
        rates = config['network_hz'][method]
        expected = ALL_RATES[:5] if method in ('openpose', 'yolo') else ALL_RATES
        if tuple(rates) != expected:
            raise ValueError('Unexpected network frequencies for ' + method)
    if config['output_hz'] != 200 or config['flow_hz'] != 200:
        raise ValueError('Expected fixed output and flow frequencies of 200 Hz')
    expected_kf = {
        'eh36m': {'pu': 0.77, 'muD': 0.06, 'muV': 0.97, 'roi': 20},
        'dhp19': {'pu': 0.36, 'muD': 0.01, 'muV': 0.01, 'roi': 10},
        'moveiit': {'pu': 0.77, 'muD': 0.06, 'muV': 0.97, 'roi': 35},
    }
    for name, expected in expected_kf[dataset].items():
        if config['kf'][name] != expected:
            raise ValueError('Unexpected {}={} for {}'.format(name, config['kf'][name], dataset))
    if config['kf']['of_gain'] != 1.0 or config['kf']['use_lc'] is not True:
        raise ValueError('ExpA requires of_gain=1.0 and use_lc=true')
    return config


def gt_for_event(dataset, event, config):
    """Resolve the original GT, without silently creating missing entries."""
    event = Path(event)
    if dataset == 'moveiit':
        # Subject keys are inherited from the existing 59-recording protocol.
        subject, recording = event.parts[-4], event.parts[-3]
        subject_tag = {'luna': 'P1_luna', 'simon': 'P11_simon'}.get(subject)
        if subject_tag is None:
            return None
        return Path(config['gt_root']) / 'GT' / subject_tag / (recording + '_joint_projections.csv')
    if dataset == 'dhp19':
        return event.parent.parent / event.parent.name.replace('dvs', 'GT200Hzskeleton') / 'data.log'
    # eH36M: source paths are /events/<cam...>/ch0dvs/data.log.
    return event.parent.parent / 'ch0GT200Hzskeleton' / 'data.log'


def inventory(config, method):
    """Yield eligible input + GT pairs and explicit rejected entries."""
    dataset = config['dataset']
    root_key = 'rgb_root' if method in ('openpose', 'yolo') else 'event_root'
    root = Path(config[root_key]).expanduser()
    if not root.is_dir():
        raise FileNotFoundError('Input directory missing: ' + str(root))
    if method in ('openpose', 'yolo'):
        inputs = sorted(root.rglob('*.mp4'))
    elif dataset == 'eh36m':
        inputs = sorted(root.glob('*/ch0dvs/data.log'))
    elif dataset == 'dhp19':
        inputs = sorted(p for p in root.glob('*/ch*dvs/data.log')
                        if re.fullmatch(r'ch[23]dvs', p.parent.name))
    else:
        inputs = sorted(root.glob('*/**/atis-s/data.log'))
    if not inputs:
        raise FileNotFoundError('No input recordings found for ' + dataset + ' ' + method)
    eligible, excluded = [], []
    seen = set()
    for source in inputs:
        rel = source.relative_to(root)
        if method in ('openpose', 'yolo'):
            sample = rel.with_suffix('').as_posix()
            # Use event-side GT regardless of RGB directory layout.
            gt = Path(config['gt_root']) / rel.with_suffix('') / 'ch0GT200Hzskeleton' / 'data.log'
        else:
            # eH36M event log and its RGB video must have the SAME sample_id.
            # Event input: events/<sequence>/ch0dvs/data.log
            # RGB input:   rgb/<sequence>.mp4
            if dataset == 'eh36m':
                sample = rel.parts[0]
            else:
                sample = rel.parent.as_posix() if rel.name == 'data.log' else rel.as_posix()
            gt = gt_for_event(dataset, source, config)
        name = sample.replace('/', '__')
        if name in seen:
            raise ValueError('Duplicate recording name: ' + name)
        seen.add(name)
        reason = None
        if dataset == 'eh36m' and sample.split('/')[0].lower() in config.get('known_invalid_sequences', []):
            reason = 'known_invalid_eh36m_sequence'
        elif gt is None or not gt.is_file():
            reason = 'ground_truth_missing_or_unmapped'
        if reason:
            excluded.append({'dataset': dataset, 'method': method, 'sample_id': sample,
                             'input_path': str(source), 'gt_path': str(gt or ''), 'reason': reason})
        else:
            eligible.append({'dataset': dataset, 'method': method, 'sample_id': sample,
                             'input_path': str(source), 'gt_path': str(gt)})
    return eligible, excluded



def matched_inventory(config, methods=None):
    """Select only recordings available to every configured comparator.

    Matching by the exact sample_id prevents a model from gaining an accuracy
    advantage through a different sequence cohort. Exclusions are recorded for
    the configured method whose valid data are absent. A selected subset of
    methods still uses the full-dataset common cohort.
    """
    configured = list(config['methods'])
    chosen = list(methods or configured)
    per_method = {}
    excluded = []
    for method in configured:
        available, rejected = inventory(config, method)
        per_method[method] = {rec['sample_id']: rec for rec in available}
        excluded.extend(rejected)
    common = set.intersection(*(set(p) for p in per_method.values()))
    # Some sequences have event input but no corresponding RGB input, or vice versa.
    # Make this mismatch visible in excluded.csv instead of silently comparing
    # different populations across methods.
    full_union = set.union(*(set(p) for p in per_method.values()))
    for sample in sorted(full_union - common):
        for method in configured:
            if sample in per_method[method]:
                rec = per_method[method][sample]
                excluded.append({**rec, 'reason': 'not_available_to_all_comparison_methods'})
    if not common:
        raise RuntimeError('No common GT-paired recordings across {}'.format(configured))
    return {method: [per_method[method][sample] for sample in sorted(common)]
            for method in chosen}, excluded


def safe_stem(sample):
    """Avoid collisions between different recordings and unsafe path chars."""
    stem = re.sub(r'[^A-Za-z0-9_.-]+', '__', sample)
    return stem[:170] + '__' + hashlib.sha256(sample.encode()).hexdigest()[:10]


def sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_head():
    try:
        return subprocess.check_output(['git', '-C', str(FLOW), 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(tmp), str(path))


def append_log(path, record):
    """Append a JSONL audit event, flushed before the next trial is started."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
