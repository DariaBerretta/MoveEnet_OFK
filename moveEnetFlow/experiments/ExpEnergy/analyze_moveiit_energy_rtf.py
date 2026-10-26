#!/usr/bin/env python3
"""Move-IIT energy and RTF analysis aligned with ExpFinal/final_analysis.py.

Five repeats -> per-sequence mean -> dataset-level equal-sequence mean energy,
ratio-of-sums RTF, input-normalized energy; percentile bootstrap resamples the
12 sequence units (5,000 replicates, seed 20260911). No new inference.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import pandas as pd

MODELS = ('movenet_50hz', 'movenet_5hz', 'moveenetofk_5hz')
MODEL_NAMES = {'movenet_50hz':'MoveEnet only (50 Hz)',
               'movenet_5hz':'MoveEnet only (5 Hz)',
               'moveenetofk_5hz':'MoveEnetOFK (5 Hz / OF 200 Hz)'}
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260911
DURATION_SOURCE = 'YARP_data_log_first_last_AE_packet_timestamp'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_csv(path):
    require(path.is_file(), 'Missing file: {}'.format(path))
    df = pd.read_csv(path, keep_default_na=False)
    require(len(df)>0, 'Empty CSV: {}'.format(path))
    return df


def positive_numbers(frame, names, strictly=True):
    for k in names:
        frame[k] = pd.to_numeric(frame[k],errors='raise')
        require(np.isfinite(frame[k]).all(), 'Nonfinite '+k)
        require((frame[k]>0).all() if strictly else (frame[k]>=0).all(), 'Negative/zero '+k)


def interval(a):
    return tuple(map(float, np.quantile(a, [.025,.975])))


def analyze(root, strict=True, expected_repeats=None):
    root=Path(root)
    configp=root/'config.json'
    require(configp.is_file(), 'Missing config.json')
    config=json.loads(configp.read_text())
    require(config.get('protocol')=='moveiit_energy_rtf_expFinal_aligned_v3_exact_packet_span',
            'Wrong energy protocol version. Do not merge with old proxy-span sessions.')
    samples=read_csv(root/'samples_durations.csv')
    energy=read_csv(root/'energy_measurements.csv')
    require(not samples['sample_id'].duplicated().any(), 'Duplicate duration sample IDs')
    require((samples['duration_source']==DURATION_SOURCE).all(), 'Input spans are not exact ATIS packet input spans')
    require(len(samples)==len(config['sample_ids']), 'Sample counts mismatch with config')
    require(set(samples['sample_id'])==set(config['sample_ids']), 'Sample IDs mismatch with config')
    if strict:
        require(len(samples)==12, 'Final energy requires the full 12-sample benchmark')
        require(int(config['repeats'])==5,'Final protocol requires five repeats')
    repeats=int(config['repeats'])
    if expected_repeats is not None:
        require(repeats==expected_repeats,'Unexpected repeat count')
    positive_numbers(samples, ['duration_s'])
    positive_numbers(samples, ['input_start_s','input_end_s'])
    require(np.allclose(samples['input_end_s'] - samples['input_start_s'],samples['duration_s'],rtol=0,atol=2e-6),
            'Duration does not match event endpoints')
    fail=energy[(energy['status']!='OK') | (energy['exit_code'].astype(str).isin(['0','0.0'])==False)]
    ok=energy.drop(fail.index).copy()
    require(len(ok)>0, 'No successful runs')
    require(set(ok['model']).issubset(MODELS), 'Unknown model in energy CSV')
    require(set(ok['sample_id']).issubset(set(samples['sample_id'])),'Unknown sample ID')
    require(ok['dataset'].eq('moveiit').all(),'Unexpected dataset code')
    require(not ok.duplicated(['sample_id','model','order_id']).any(), 'Duplicate OK repetition detected')
    expected_keys={(sid,m,f'R{rep}') for sid in samples['sample_id'] for m in MODELS for rep in range(1,repeats+1)}
    observed_keys=set(zip(ok['sample_id'],ok['model'],ok['order_id']))
    if strict:
        require(observed_keys==expected_keys,'Missing or unexpected measurements: missing={} extra={}'.format(
            len(expected_keys-observed_keys),len(observed_keys-expected_keys)))
    else:
        if observed_keys != expected_keys:
            print('WARNING: partial session; {} missing runs'.format(len(expected_keys-observed_keys)))
    positive_numbers(ok,['wall_seconds','measured_compute_energy_j','cpu_package_energy_j','gpu_energy_j'])
    require(np.allclose(ok['measured_compute_energy_j'],ok['cpu_package_energy_j']+ok['gpu_energy_j'],rtol=1e-6,atol=.001),
            'Total compute energy differs from CPU+GPU sum')
    for r in ok.itertuples(index=False):
        log=Path(r.log_path)
        if not log.is_file():
            log=root/'runs'/log.name
        require(log.is_file() and 'Finished processing event file.' in log.read_text(errors='replace'),
                'Incomplete energy replay: {}'.format(log))
    # 5 measured repeats are not independent observations in the uncertainty model.
    seq=[]
    duration_map=samples.set_index('sample_id')
    for (sid,model),g in ok.groupby(['sample_id','model']):
        if strict:
            require(len(g)==repeats and set(g['order_id'])=={f'R{i}' for i in range(1,repeats+1)},
                    'Missing repeats: {} {}'.format(sid,model))
        dur=duration_map.loc[sid]
        T=g['wall_seconds'].to_numpy(float)
        E=g['measured_compute_energy_j'].to_numpy(float)
        C=g['cpu_package_energy_j'].to_numpy(float)
        G=g['gpu_energy_j'].to_numpy(float)
        d=float(dur.duration_s)
        netp=float(config['network_period_s'][model])
        seq.append(dict(dataset='moveiit', sample_id=sid, subject=dur.subject,
                        sequence=dur.sequence, model=model, method=MODEL_NAMES[model],
                        config_hash=config['binary_sha256'], n_repetitions=len(g),
                        net_period_s=netp,net_hz=1/netp,
                        flow_period_s=0.005 if model=='moveenetofk_5hz' else np.nan,
                        output_period_s=0.005,output_hz=200.,
                        input_span_s=d,duration_source=dur.duration_source,
                        energy_j=E.mean(),energy_repeat_std_j=E.std(ddof=1) if len(E)>1 else np.nan,
                        cpu_energy_j=C.mean(),gpu_energy_j=G.mean(),
                        sequence_wall_seconds=T.mean(),
                        sequence_wall_repeat_std_s=T.std(ddof=1) if len(T)>1 else np.nan,
                        real_time_factor=T.mean()/d,
                        real_time_factor_repeat_std=(T.std(ddof=1)/d) if len(T)>1 else np.nan,
                        speed_vs_realtime=d/T.mean(),
                        amortized_ms_per_nominal_output_slot=1000*T.mean()/(d*200),
                        energy_j_per_input_second=E.mean()/d,
                        energy_j_per_nominal_output_slot=E.mean()/(d*200),
                        combined_mean_power_w=E.sum()/T.sum(),
                        cpu_mean_power_w=C.sum()/T.sum(),gpu_mean_power_w=G.sum()/T.sum()))
    seq=pd.DataFrame(seq).sort_values(['sample_id','model']).reset_index(drop=True)
    summary=[]
    for model in MODELS:
        g=seq[seq.model==model].sort_values('sample_id')
        require(len(g)>0, 'No runs for '+model)
        if strict:
            require(len(g)==12,'Missing samples for '+model)
        E=g.energy_j.to_numpy();T=g.sequence_wall_seconds.to_numpy();D=g.input_span_s.to_numpy()
        C=g.cpu_energy_j.to_numpy();G=g.gpu_energy_j.to_numpy()
        ix=np.random.default_rng(BOOTSTRAP_SEED).integers(0,len(g),size=(BOOTSTRAP_REPLICATES,len(g)))
        row=dict(dataset='moveiit', model=model,method=MODEL_NAMES[model],
                 net_period_s=float(g.net_period_s.iloc[0]),net_hz=float(g.net_hz.iloc[0]),
                 output_hz=200,n_sequences=len(g), repetitions_per_sequence=repeats,
                 energy_mean_j=E.mean(),energy_sequence_sd_j=E.std(ddof=1) if len(E)>1 else np.nan,
                 cpu_energy_mean_j=C.mean(),gpu_energy_mean_j=G.mean(),
                 wall_mean_s=T.mean(),wall_sequence_sd_s=T.std(ddof=1) if len(T)>1 else np.nan,
                 input_span_mean_s=D.mean(),real_time_factor=T.sum()/D.sum(),
                 speed_vs_realtime=D.sum()/T.sum(),
                 energy_j_per_input_second=E.sum()/D.sum(),
                 combined_mean_power_w=E.sum()/T.sum(),
                 cpu_mean_power_w=C.sum()/T.sum(),gpu_mean_power_w=G.sum()/T.sum(),
                 amortized_ms_per_nominal_output_slot=1000*T.sum()/D.sum()/200)
        for name,a in [('energy',E[ix].mean(axis=1)),('wall',T[ix].mean(axis=1)),
                       ('rtf',T[ix].sum(axis=1)/D[ix].sum(axis=1)),
                       ('normalized_energy',E[ix].sum(axis=1)/D[ix].sum(axis=1))]:
            row[name+'_ci95_low'],row[name+'_ci95_high']=interval(a)
        summary.append(row)
    summary=pd.DataFrame(summary)
    paired=[]
    wide=seq.pivot(index='sample_id',columns='model',values=['energy_j','sequence_wall_seconds','input_span_s'])
    for baseline in ('movenet_50hz','movenet_5hz'):
        comparable=wide.dropna(subset=[('energy_j',baseline),('energy_j','moveenetofk_5hz')])
        if strict:
            require(len(comparable)==12,'Paired comparison missing samples')
        Eb=comparable['energy_j'][baseline].to_numpy(); Eo=comparable['energy_j']['moveenetofk_5hz'].to_numpy()
        Tb=comparable['sequence_wall_seconds'][baseline].to_numpy(); To=comparable['sequence_wall_seconds']['moveenetofk_5hz'].to_numpy()
        ix=np.random.default_rng(BOOTSTRAP_SEED).integers(0,len(comparable),size=(BOOTSTRAP_REPLICATES,len(comparable)))
        # Paired bootstrap: the two methods use exactly the same resampled sequences.
        saving=100*(1-Eo[ix].mean(axis=1)/Eb[ix].mean(axis=1))
        tsaving=100*(1-To[ix].sum(axis=1)/Tb[ix].sum(axis=1))
        lo,hi=interval(saving);lt,ht=interval(tsaving)
        paired.append(dict(dataset='moveiit',reference=baseline,method='moveenetofk_5hz',
                           n_samples=len(comparable),
                           energy_saving_pct=100*(1-Eo.mean()/Eb.mean()),
                           energy_saving_ci95_low=lo,energy_saving_ci95_high=hi,
                           runtime_saving_pct=100*(1-To.sum()/Tb.sum()),
                           runtime_saving_ci95_low=lt,runtime_saving_ci95_high=ht))
    pairs=pd.DataFrame(paired)
    seq.to_csv(root/'energy_runtime_by_sample.csv',index=False)
    summary.to_csv(root/'energy_runtime_dataset_summary.csv',index=False)
    pairs.to_csv(root/'paired_differences.csv',index=False)
    # Backward-compatible shorter naming from the first Move-IIT package.
    seq.to_csv(root/'rtf_energy_per_sample.csv',index=False)
    summary.to_csv(root/'rtf_energy_summary.csv',index=False)
    print('Measured OK attempts: {} / {}, excluded failed attempts: {}'.format(len(ok),len(expected_keys),len(fail)))
    print('Duration: first -> last ATIS AE packet timestamp (data.log), NOT prediction output span')
    print('Uncertainty: 5000 bootstrap repetitions over {} sequences (not energy repetitions)'.format(len(samples)))
    print('\nFINAL SUMMARY (same fields/estimator as ExpFinal):')
    print(summary[['method','n_sequences','energy_mean_j','cpu_energy_mean_j','gpu_energy_mean_j',
                   'wall_mean_s','real_time_factor','energy_j_per_input_second',
                   'energy_ci95_low','energy_ci95_high','rtf_ci95_low','rtf_ci95_high']].round(4).to_string(index=False))
    print('\nPAIRED IMPROVEMENTS:')
    print(pairs.round(3).to_string(index=False))
    print('\nSaved results under',root)
    return seq,summary,pairs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session-dir',type=Path,required=True)
    p.add_argument('--strict',action='store_true',help='Require 12 samples x 3 models x 5 repeats (final paper protocol)')
    args=p.parse_args()
    analyze(args.session_dir, strict=args.strict)


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:',e,file=sys.stderr)
        sys.exit(1)
