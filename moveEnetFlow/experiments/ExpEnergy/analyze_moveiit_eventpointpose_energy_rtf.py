#!/usr/bin/env python3
"""Combine EventPointPose and MoveEnet Move-IIT energy / RTF using paired 12 samples.

Repeat -> sequence means; 12 sequence units -> population estimates & bootstrap.
CPU package and GPU device total energy are not process-isolated measurements.
Baseline measurements come from an earlier session; dates/order are not balanced
across all five methods. Results are descriptive, not a same-session randomized test.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
from run_moveiit_eventpointpose_energy_rtf import PROTOCOL, SOURCE, MODELS, parse_epp_log

BASE = ['movenet_50hz', 'movenet_5hz', 'moveenetofk_5hz']
EPP = [x[0] for x in MODELS]
ORDER = BASE + EPP
DISPLAY = {
    'movenet_50hz': 'MoveEnet only (50 Hz)',
    'movenet_5hz': 'MoveEnet only (5 Hz)',
    'moveenetofk_5hz': 'MoveEnetOFK (5 Hz; OF 200 Hz)',
    'eventpointpose_5hz': 'EventPointPose (5 Hz)',
    'eventpointpose_50hz': 'EventPointPose (50 Hz)',
}
SEED = 20260911
NBOOT = 5000


def require(test, message):
    if not test:
        raise RuntimeError(message)


def load(path):
    require(path.is_file(), 'Missing file: {}'.format(path))
    frame = pd.read_csv(path)
    require(len(frame) > 0, 'Empty file: {}'.format(path))
    return frame


def to_numeric(frame, columns, positive=False):
    for name in columns:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
        require(np.isfinite(frame[name].to_numpy(float)).all(), 'Nonfinite column: {}'.format(name))
        if positive:
            require((frame[name] > 0).all(), 'Nonpositive column: {}'.format(name))


def percentile(a):
    return map(float, np.quantile(a, [.025, .975]))


def run(root, baseline, output=None, strict=True):
    root = Path(root).expanduser().resolve()
    baseline = Path(baseline).expanduser().resolve()
    output = Path(output).expanduser().resolve() if output else root / 'comparison_with_movenet'
    require((root/'config.json').is_file(), 'Missing EventPointPose config.json')
    cfg = json.loads((root/'config.json').read_text())
    require(cfg['protocol'] == PROTOCOL, 'Unexpected EventPointPose protocol')
    require(Path(cfg['reference_session']).resolve() == baseline, 'Different baseline session in EPP config')
    base_cfg = json.loads((baseline/'config.json').read_text())
    require(base_cfg['protocol'] == 'moveiit_energy_rtf_expFinal_aligned_v3_exact_packet_span',
            'Baseline must use the original exact-packet-span protocol')
    require(cfg['reference_config_sha256'] == __import__('hashlib').sha256((baseline/'config.json').read_bytes()).hexdigest(),
            'Baseline config changed after measurement')
    require(cfg['reference_durations_sha256'] == __import__('hashlib').sha256((baseline/'samples_durations.csv').read_bytes()).hexdigest(),
            'Baseline input-duration manifest changed after measurement')

    durations = load(root/'samples_durations.csv')
    reference_durations = load(baseline/'samples_durations.csv')
    require(not durations.sample_id.duplicated().any(), 'Duplicate EPP input sequence')
    require(set(durations.sample_id).issubset(set(reference_durations.sample_id)), 'Unknown EPP samples')
    assert_fields = ['sample_id','duration_s','duration_source']
    for name in assert_fields:
        require(name in durations.columns and name in reference_durations.columns,
                'Missing input-duration column {}'.format(name))
    require(durations.duration_source.eq(SOURCE).all(), 'Unexpected duration source')
    to_numeric(durations, ['duration_s'], positive=True)
    to_numeric(reference_durations, ['duration_s'], positive=True)
    comparison_d = durations[['sample_id','duration_s']].merge(
        reference_durations[['sample_id','duration_s']], on='sample_id',
        validate='one_to_one', suffixes=('_epp','_baseline'))
    require(np.allclose(comparison_d.duration_s_epp, comparison_d.duration_s_baseline,
                        rtol=0, atol=2e-6), 'EPP and MoveEnet input durations disagree')
    if strict:
        require(len(durations) == 12 and len(cfg['sample_ids']) == 12 and
                int(cfg['repeats']) == 5, 'Final protocol requires 12 samples x 5 repeats')

    energy = load(root/'energy_measurements.csv')
    require(set(energy.model) <= set(EPP), 'Other models in EPP meter results')
    require(energy.dataset.eq('moveiit').all(), 'Unexpected dataset')
    to_numeric(energy, ['wall_seconds','cpu_package_energy_j', 'gpu_energy_j',
                        'measured_compute_energy_j'], positive=True)
    require(np.allclose(energy.measured_compute_energy_j,
                        energy.cpu_package_energy_j + energy.gpu_energy_j,
                        atol=.001, rtol=1e-6), 'Energy sum mismatch')
    evidence = []
    for r in energy.itertuples(index=False):
        ok_log, reason, stats = parse_epp_log(Path(r.log_path))
        ok = r.status == 'OK' and int(r.exit_code) == 0 and ok_log
        evidence.append({'sample_id': r.sample_id, 'model': r.model,
                         'order_id': r.order_id, 'log_path': r.log_path,
                         'integrity_ok': ok, 'integrity_reason': reason,
                         'loaded_events': stats.get('loaded_events', np.nan),
                         'processed_events': stats.get('processed_events', np.nan),
                         'valid_inferences': stats.get('valid_inferences', np.nan)})
    evidence = pd.DataFrame(evidence)
    energy = energy.copy()
    energy['integrity_ok'] = evidence.integrity_ok.to_numpy(bool)
    good = energy[energy.integrity_ok].copy()
    require(len(good) > 0, 'No completed EPP measurements')
    require(not good.duplicated(['sample_id', 'model', 'order_id']).any(),
            'Duplicate completed EPP repeats: do not double-count resumes')
    expected = {(sid,m,'R{}'.format(i)) for sid in cfg['sample_ids']
                for m in EPP for i in range(1,int(cfg['repeats'])+1)}
    seen = set(zip(good.sample_id, good.model, good.order_id))
    if strict:
        require(seen == expected, 'Missing/unexpected EPP measurements: missing={} extra={}'.format(
            sorted(expected-seen)[:15], sorted(seen-expected)[:15]))
    else:
        require(not seen-expected, 'Unexpected EPP runs')

    # Hardware check: all 5 methods must use the same physical GPU and index;
    # no process isolation is possible for these counters.
    baseline_meter = load(baseline/'energy_measurements.csv')
    base_valid = baseline_meter[(baseline_meter.status == 'OK') &
                                (pd.to_numeric(baseline_meter.exit_code, errors='coerce') == 0)]
    require(len(base_valid) > 0, 'No original MoveEnet energy records')
    for column in ('gpu_uuid','gpu_index','gpu_name'):
        require(column in good.columns and column in base_valid.columns, 'Missing GPU field {}'.format(column))
        require(good[column].astype(str).nunique() == 1 and
                base_valid[column].astype(str).nunique() == 1 and
                good[column].astype(str).iloc[0] == base_valid[column].astype(str).iloc[0],
                'GPU differs between sessions ({}) - not directly comparable'.format(column))

    baseline_seq = load(baseline/'energy_runtime_by_sample.csv')
    require(set(baseline_seq.model) == set(BASE), 'Original baseline methods differ')
    require(not baseline_seq.duplicated(['sample_id', 'model']).any(), 'Duplicate baseline per-sample rows')
    require(set(baseline_seq.sample_id) == set(reference_durations.sample_id),
            'Baseline analysis does not contain the intended samples')
    if strict:
        require(len(baseline_seq) == 36 and baseline_seq.n_repetitions.eq(5).all(),
                'Baseline summary must contain 12 samples x 3 methods x 5 repeats')

    duration_by_id = durations.set_index('sample_id')
    epp_rows = []
    for (sid, method), g in good.groupby(['sample_id','model'], sort=True):
        require(sid in duration_by_id.index, 'Unknown measured sample: {}'.format(sid))
        if strict:
            require(len(g) == int(cfg['repeats']), 'Missing EPP repeats {} {}'.format(sid, method))
        d = duration_by_id.loc[sid]
        T = g.wall_seconds.to_numpy(float)
        E = g.measured_compute_energy_j.to_numpy(float)
        C = g.cpu_package_energy_j.to_numpy(float)
        G = g.gpu_energy_j.to_numpy(float)
        duration = float(d.duration_s)
        net_period = cfg['models'][method]
        epp_rows.append(dict(dataset='moveiit', sample_id=sid, subject=d.subject,
                             sequence=d.sequence, model=method, method=DISPLAY[method],
                             n_repetitions=len(g), net_period_s=net_period,
                             net_hz=1/net_period, flow_period_s=np.nan,
                             output_period_s=.005, output_hz=200,
                             input_span_s=duration, duration_source=SOURCE,
                             energy_j=E.mean(),
                             energy_repeat_std_j=E.std(ddof=1) if len(E)>1 else np.nan,
                             cpu_energy_j=C.mean(), gpu_energy_j=G.mean(),
                             sequence_wall_seconds=T.mean(),
                             sequence_wall_repeat_std_s=T.std(ddof=1) if len(T)>1 else np.nan,
                             real_time_factor=T.mean()/duration,
                             real_time_factor_repeat_std=T.std(ddof=1)/duration if len(T)>1 else np.nan,
                             speed_vs_realtime=duration/T.mean(),
                             amortized_ms_per_nominal_output_slot=1000*T.mean()/(duration*200),
                             energy_j_per_input_second=E.mean()/duration,
                             energy_j_per_nominal_output_slot=E.mean()/(duration*200),
                             combined_mean_power_w=E.sum()/T.sum(),
                             cpu_mean_power_w=C.sum()/T.sum(),gpu_mean_power_w=G.sum()/T.sum()))
    epp_seq = pd.DataFrame(epp_rows)
    selected = set(cfg['sample_ids'])
    old = baseline_seq[baseline_seq.sample_id.isin(selected)].copy()
    require(not old.empty, 'No baseline samples for selected EPP sequences')
    if strict:
        require(len(old)==3*len(selected), 'Incomplete baseline selected sequences')
    # Subset the baseline with same IDs, preserving original per-sequence statistics.
    shared_columns = sorted(set(old.columns) | set(epp_seq.columns))
    combined = pd.concat([old.reindex(columns=shared_columns),
                          epp_seq.reindex(columns=shared_columns)], ignore_index=True)
    require(not combined.duplicated(['sample_id','model']).any(), 'Duplicate combined sample / model')
    sequence_sets = [set(combined.loc[combined.model == m, 'sample_id']) for m in ORDER]
    require(len({tuple(sorted(s)) for s in sequence_sets}) == 1,
            'Methods use different recording sets; paired analysis not possible')

    estimates = []
    rng = np.random.default_rng(SEED)
    n = len(selected)
    ids = rng.integers(0, n, size=(NBOOT,n))
    for model in ORDER:
        g = combined[combined.model == model].sort_values('sample_id')
        E = g.energy_j.to_numpy(float); T = g.sequence_wall_seconds.to_numpy(float)
        D = g.input_span_s.to_numpy(float)
        C = g.cpu_energy_j.to_numpy(float); G = g.gpu_energy_j.to_numpy(float)
        require(len(g) == n and np.all((E>0)&(T>0)&(D>0)), 'Invalid sample summary '+model)
        metric_boot = {
            'energy': E[ids].mean(axis=1),
            'wall': T[ids].mean(axis=1),
            'rtf': T[ids].sum(axis=1)/D[ids].sum(axis=1),
            'normalized_energy': E[ids].sum(axis=1)/D[ids].sum(axis=1)}
        row = dict(dataset='moveiit',model=model,method=DISPLAY[model],
                   net_period_s=g.net_period_s.iloc[0],net_hz=g.net_hz.iloc[0],
                   output_hz=200.,n_sequences=n, repetitions_per_sequence=g.n_repetitions.iloc[0],
                   energy_mean_j=E.mean(), energy_sequence_sd_j=E.std(ddof=1) if n>1 else np.nan,
                   cpu_energy_mean_j=C.mean(),gpu_energy_mean_j=G.mean(),
                   wall_mean_s=T.mean(), wall_sequence_sd_s=T.std(ddof=1) if n>1 else np.nan,
                   input_span_mean_s=D.mean(),real_time_factor=T.sum()/D.sum(),
                   speed_vs_realtime=D.sum()/T.sum(),energy_j_per_input_second=E.sum()/D.sum(),
                   combined_mean_power_w=E.sum()/T.sum(),
                   cpu_mean_power_w=C.sum()/T.sum(),gpu_mean_power_w=G.sum()/T.sum(),
                   amortized_ms_per_nominal_output_slot=1000*T.sum()/D.sum()/200)
        for key, boot in metric_boot.items():
            row[key+'_ci95_low'],row[key+'_ci95_high']=percentile(boot)
        estimates.append(row)
    summary = pd.DataFrame(estimates)

    pairs = []
    pivot = combined.pivot(index='sample_id', columns='model',
                           values=['energy_j','sequence_wall_seconds','input_span_s']).sort_index()
    comparisons = [('eventpointpose_5hz','movenet_5hz'),
                   ('eventpointpose_5hz','moveenetofk_5hz'),
                   ('eventpointpose_5hz','movenet_50hz'),
                   ('eventpointpose_50hz','movenet_50hz'),
                   ('eventpointpose_50hz','eventpointpose_5hz'),
                   ('moveenetofk_5hz','movenet_50hz')]
    for target, baseline_name in comparisons:
        g = pivot.dropna(subset=[('energy_j',target),('energy_j',baseline_name)])
        if strict:
            require(len(g)==12, 'Unpaired methods: {} vs {}'.format(target, baseline_name))
        Et=g['energy_j'][target].to_numpy(float); Eb=g['energy_j'][baseline_name].to_numpy(float)
        Tt=g['sequence_wall_seconds'][target].to_numpy(float)
        Tb=g['sequence_wall_seconds'][baseline_name].to_numpy(float)
        n2=len(g)
        rng2=np.random.default_rng(SEED)
        draw=rng2.integers(0,n2,size=(NBOOT,n2))
        es=100*(1-Et[draw].mean(axis=1)/Eb[draw].mean(axis=1))
        ts=100*(1-Tt[draw].sum(axis=1)/Tb[draw].sum(axis=1))
        lo,hi=percentile(es); lt,ht=percentile(ts)
        pairs.append(dict(dataset='moveiit',method=target,reference=baseline_name,
                          n_samples=n2, energy_saving_pct=100*(1-Et.mean()/Eb.mean()),
                          energy_saving_ci95_low=lo,energy_saving_ci95_high=hi,
                          runtime_saving_pct=100*(1-Tt.sum()/Tb.sum()),
                          runtime_saving_ci95_low=lt,runtime_saving_ci95_high=ht))
    paired = pd.DataFrame(pairs)

    output.mkdir(parents=True,exist_ok=True)
    evidence.to_csv(output/'epp_replay_integrity_audit.csv',index=False)
    combined.to_csv(output/'energy_runtime_by_sample_5methods.csv',index=False)
    summary.to_csv(output/'energy_runtime_dataset_summary_5methods.csv',index=False)
    paired.to_csv(output/'paired_differences_5methods.csv',index=False)
    protocol = dict(version=PROTOCOL, source_baseline_session=str(baseline),
                    source_eventpointpose_session=str(root),
                    same_gpu_verified=True, same_input_spans_verified=True,
                    energy_measurement='RAPL package + NVML device total; NOT process isolated',
                    runtime='whole measured docker exec command wall time, includes load/startup',
                    real_time_factor='sum sequence mean wall time / sum exact ATIS input span',
                    aggregation='five repeats averaged within each sequence; sequences weighted equally for energy mean',
                    uncertainty='paired bootstrap resampling recordings (N=5000, seed={})'.format(SEED),
                    hardware_caveat='Baseline and EventPointPose energy were recorded in DIFFERENT sessions, not randomized together; differences may include day/thermal/system-load effects.',
                    epp_operating_points=cfg['models'],
                    baseline_operating_points=base_cfg['network_period_s'])
    (output/'analysis_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('Validated EPP attempts: {} success / {} recorded'.format(len(good),len(energy)))
    print('Baseline + EPP: {} samples; same GPU and exact packet-span denominator'.format(n))
    print('CAVEAT: cross-session comparison, different dates/order/thermal conditions.')
    print(summary[['method','energy_mean_j','cpu_energy_mean_j','gpu_energy_mean_j',
                   'wall_mean_s','real_time_factor','energy_j_per_input_second']].round(3).to_string(index=False))
    print('\nPaired differences (positive=saving vs reference):')
    print(paired[['method','reference','energy_saving_pct','runtime_saving_pct']].round(2).to_string(index=False))
    print('Export:', output)
    return combined,summary,paired


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--session-dir',type=Path,required=True)
    ap.add_argument('--reference-session',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=None)
    ap.add_argument('--allow-partial',action='store_true',help='For debugging 1 sample / incomplete runs only')
    args=ap.parse_args()
    run(args.session_dir,args.reference_session,args.out,strict=not args.allow_partial)


if __name__=='__main__':
    try:
        main()
    except (OSError,RuntimeError,ValueError,KeyError) as exc:
        print('ERROR:',exc,file=sys.stderr)
        sys.exit(1)
