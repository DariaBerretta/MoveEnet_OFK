"""One analysis implementation used by the CLI and both final notebooks.

Energy/runtime: five repeats -> sequence-camera mean -> dataset summary.
Accuracy: same ExpA point-weighted estimator; sequence-macro result also exported.
Bootstrap units are sequence-camera records, never the five timing repetitions.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from final_common import require, atomic_json

DATASETS = {'h36m':'eH36M', 'dhp19':'DHP19'}
MODEL_NAMES = {'movenet':'MoveEnet only', 'moveenetofk':'MoveEnetOFK',
               'openpose':'OpenPose','yolo':'YOLOPose','eventpointpose':'EventPointPose'}
MODEL_ORDER = {'h36m':['movenet','moveenetofk','openpose','yolo'],
               'dhp19':['movenet','moveenetofk','eventpointpose']}
MARKERS = {'movenet':'o','moveenetofk':'s','openpose':'^','yolo':'D','eventpointpose':'v'}
BOOTSTRAP_REPLICATES=5000
BOOTSTRAP_SEED=20260911


def csv_out(frame, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    frame.to_csv(tmp,index=False); tmp.replace(path)


def energy_sequence_table(e):
    numeric=['wall_seconds','cpu_package_energy_j','gpu_energy_j','measured_compute_energy_j',
             'input_span_s','net_period_s','output_period_s','repetition']
    e=e.copy()
    for col in numeric:
        e[col]=pd.to_numeric(e[col],errors='raise')
    require(np.isfinite(e[numeric].to_numpy(dtype=float)).all(),'Energia/runtime contiene NaN o inf')
    require(not e.duplicated(['dataset','sample_id','model','repetition']).any(),'Ripetizioni energia duplicate')
    rows=[]
    for key, r in e.groupby(['dataset','sample_id','model','config_hash'],sort=True):
        require(len(r)==5 and r['repetition'].nunique()==5,'Attese 5 ripetizioni distinte: '+str(key))
        require(r['input_span_s'].nunique()==1,'Durate non univoche: '+str(key))
        D=float(r['input_span_s'].iloc[0]); output_p=float(r['output_period_s'].iloc[0])
        require(D>0 and output_p>0,'Durata/periodo non positivi')
        T=r['wall_seconds'].to_numpy(float); E=r['measured_compute_energy_j'].to_numpy(float)
        row=dict(zip(['dataset','sample_id','model','config_hash'],key))
        row.update(job_id=r['job_id'].iloc[0], n_repetitions=5,
                   net_period_s=float(r['net_period_s'].iloc[0]), net_hz=1/float(r['net_period_s'].iloc[0]),
                   output_period_s=output_p, output_hz=1/output_p,
                   input_span_s=D, duration_source=r['duration_source'].iloc[0],
                   energy_j=float(E.mean()), energy_repeat_std_j=float(E.std(ddof=1)),
                   cpu_energy_j=float(r['cpu_package_energy_j'].mean()),
                   gpu_energy_j=float(r['gpu_energy_j'].mean()),
                   sequence_wall_seconds=float(T.mean()), sequence_wall_repeat_std_s=float(T.std(ddof=1)),
                   real_time_factor=float(T.mean()/D),
                   real_time_factor_repeat_std=float(T.std(ddof=1)/D),
                   speed_vs_realtime=float(D/T.mean()),
                   amortized_ms_per_nominal_output_slot=float(1000*T.mean()*output_p/D),
                   energy_j_per_input_second=float(E.mean()/D),
                   energy_j_per_nominal_output_slot=float(E.mean()*output_p/D),
                   combined_mean_power_w=float(E.sum()/T.sum()),
                   cpu_mean_power_w=float(r['cpu_package_energy_j'].sum()/T.sum()),
                   gpu_mean_power_w=float(r['gpu_energy_j'].sum()/T.sum()))
        fp=pd.to_numeric(r['flow_period_s'],errors='coerce').dropna() if 'flow_period_s' in r else []
        row['flow_period_s']=float(fp.iloc[0]) if len(fp) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_indices(n):
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0,n,size=(BOOTSTRAP_REPLICATES,n))


def interval(x):
    return tuple(float(y) for y in np.quantile(x,[.025,.975]))


def dataset_summary(s, with_accuracy=False):
    rows=[]
    for (ds,model),r in s.groupby(['dataset','model'],sort=True):
        r=r.sort_values('sample_id').reset_index(drop=True)
        require(r['sample_id'].nunique()==12,'Dataset non completo per '+model)
        ix=bootstrap_indices(len(r))
        E=r['energy_j'].to_numpy(); T=r['sequence_wall_seconds'].to_numpy(); D=r['input_span_s'].to_numpy()
        row={'dataset':ds,'model':model,'method':MODEL_NAMES[model],
             'net_period_s':float(r['net_period_s'].iloc[0]),
             'net_hz':float(r['net_hz'].iloc[0]),'output_hz':float(r['output_hz'].iloc[0]),
             'n_sequences':len(r),'repetitions_per_sequence':5,
             'energy_mean_j':float(E.mean()),'energy_sequence_sd_j':float(E.std(ddof=1)),
             'cpu_energy_mean_j':float(r['cpu_energy_j'].mean()),
             'gpu_energy_mean_j':float(r['gpu_energy_j'].mean()),
             'wall_mean_s':float(T.mean()), 'wall_sequence_sd_s':float(T.std(ddof=1)),
             'input_span_mean_s':float(D.mean()),
             'real_time_factor':float(T.sum()/D.sum()),
             'speed_vs_realtime':float(D.sum()/T.sum()),
             'energy_j_per_input_second':float(E.sum()/D.sum()),
             'combined_mean_power_w':float(E.sum()/T.sum()),
             'cpu_mean_power_w':float(r['cpu_energy_j'].sum()/T.sum()),
             'gpu_mean_power_w':float(r['gpu_energy_j'].sum()/T.sum()),
             'amortized_ms_per_nominal_output_slot':float(1000*T.sum()/D.sum()/r['output_hz'].iloc[0])}
        for name,boot in [('energy',E[ix].mean(axis=1)),('wall',T[ix].mean(axis=1)),
                          ('rtf',T[ix].sum(axis=1)/D[ix].sum(axis=1)),
                          ('normalized_energy',E[ix].sum(axis=1)/D[ix].sum(axis=1))]:
            lo,hi=interval(boot); row[name+'_ci95_low']=lo; row[name+'_ci95_high']=hi
        if with_accuracy:
            n=r['n_points'].to_numpy(float); errors=r['sum_err_px'].to_numpy(float)
            hits=r['pck_0.2_correct_points'].to_numpy(float)
            row.update(pck_weighted_percent=float(100*hits.sum()/n.sum()),
                       mpjpe_weighted_px=float(errors.sum()/n.sum()),
                       pck_sequence_macro_percent=float(r['pck'].mean()),
                       mpjpe_sequence_macro_px=float(r['mpjpe'].mean()),
                       total_evaluated_joint_points=int(n.sum()),
                       accuracy_aggregation='point_weighted_as_ExpA',
                       min_observed_output_hz=float(r['observed_output_hz'].min()),
                       max_observed_output_hz=float(r['observed_output_hz'].max()))
            for name,boot in [('pck',100*hits[ix].sum(axis=1)/n[ix].sum(axis=1)),
                              ('mpjpe',errors[ix].sum(axis=1)/n[ix].sum(axis=1))]:
                lo,hi=interval(boot); row[name+'_ci95_low']=lo; row[name+'_ci95_high']=hi
        rows.append(row)
    return pd.DataFrame(rows)


def model_label(row):
    return '{} ({:g} Hz net)'.format(MODEL_NAMES[row['model']], row['net_hz'])


def errorbar_offsets(mean,lo,hi):
    return np.array([[max(0.,mean-lo)],[max(0.,hi-mean)]])


def plot_tradeoff(summary,dataset,normalized=False):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.9),sharex=True)
    data=summary[summary['dataset']==dataset].set_index('model')
    xname='energy_j_per_input_second' if normalized else 'energy_mean_j'
    cname='normalized_energy' if normalized else 'energy'
    for ax,metric in zip(axes,['pck','mpjpe']):
        yname='pck_weighted_percent' if metric=='pck' else 'mpjpe_weighted_px'
        for model in MODEL_ORDER[dataset]:
            r=data.loc[model].copy(); r['model']=model
            x=float(r[xname]); y=float(r[yname])
            ax.errorbar(x,y,
                        xerr=errorbar_offsets(x,r[cname+'_ci95_low'],r[cname+'_ci95_high']),
                        yerr=errorbar_offsets(y,r[metric+'_ci95_low'],r[metric+'_ci95_high']),
                        fmt=MARKERS[model],markersize=7,capsize=3,linestyle='none',label=model_label(r))
        ax.set_xlabel('Energy [J / input s]' if normalized else 'Energy [J / sequence]')
        ax.set_ylabel('PCK@0.2 [%]' if metric=='pck' else 'MPJPE [px]')
        ax.set_title('(a) PCK@0.2' if metric=='pck' else '(b) MPJPE')
        ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=9)
    fig.suptitle(DATASETS[dataset]+' | CPU-package + GPU energy; nominal output 200 Hz',fontsize=11)
    fig.subplots_adjust(left=.08,right=.98,bottom=.23,top=.82,wspace=.26)
    return fig



def plot_tradeoff_all(summary, normalized=False):
    """Single 2x2 paper figure, alongside the separate per-dataset figures."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 9.7))
    xname = 'energy_j_per_input_second' if normalized else 'energy_mean_j'
    cname = 'normalized_energy' if normalized else 'energy'
    for row, dataset in enumerate(('h36m', 'dhp19')):
        data = summary[summary['dataset'] == dataset].set_index('model')
        for col, metric in enumerate(('pck', 'mpjpe')):
            ax = axes[row, col]
            yname = 'pck_weighted_percent' if metric == 'pck' else 'mpjpe_weighted_px'
            for model in MODEL_ORDER[dataset]:
                r = data.loc[model].copy(); r['model'] = model
                x, y = float(r[xname]), float(r[yname])
                ax.errorbar(x, y,
                    xerr=errorbar_offsets(x, r[cname+'_ci95_low'], r[cname+'_ci95_high']),
                    yerr=errorbar_offsets(y, r[metric+'_ci95_low'], r[metric+'_ci95_high']),
                    fmt=MARKERS[model], markersize=6, capsize=3, linestyle='none', label=model_label(r))
            ax.set_xlabel('Energy [J / input s]' if normalized else 'Energy [J / sequence]')
            ax.set_ylabel('PCK@0.2 [%]' if metric == 'pck' else 'MPJPE [px]')
            ax.set_title('({}) {}'.format('abcd'[2*row+col], DATASETS[dataset]))
            ax.grid(alpha=.2)
        handles, labels = axes[row,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(.53, .49 if row == 0 else .045),
                   ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, bottom=.15, top=.94, hspace=.75, wspace=.26)
    return fig


def plot_runtime(summary,dataset,metric='real_time_factor'):
    import matplotlib.pyplot as plt
    r=summary[summary['dataset']==dataset].set_index('model').loc[MODEL_ORDER[dataset]].reset_index()
    fig,ax=plt.subplots(figsize=(8.2,4.7))
    labels=[model_label(x) for _,x in r.iterrows()]
    values=r[metric].to_numpy(float)
    prefix='rtf' if metric=='real_time_factor' else 'wall'
    err=np.vstack([np.maximum(0,values-r[prefix+'_ci95_low'].to_numpy()),
                   np.maximum(0,r[prefix+'_ci95_high'].to_numpy()-values)])
    ax.bar(np.arange(len(r)),values,yerr=err,capsize=4)
    ax.set_xticks(np.arange(len(r))); ax.set_xticklabels(labels,rotation=15,ha='right')
    if metric=='real_time_factor':
        ax.axhline(1,linestyle='--',linewidth=1,label='RTF = 1')
        ax.set_ylabel('Real-time factor: whole-command time / input span')
        ax.legend(frameon=False)
    else:
        ax.set_ylabel('Whole-sequence wall-clock time [s]')
    ax.set_title(DATASETS[dataset]+' | Existing energy-run timing')
    fig.tight_layout(); return fig


def plot_energy(summary,dataset):
    import matplotlib.pyplot as plt
    r=summary[summary['dataset']==dataset].set_index('model').loc[MODEL_ORDER[dataset]].reset_index()
    fig,ax=plt.subplots(figsize=(8.2,4.7)); x=np.arange(len(r))
    ax.bar(x,r['cpu_energy_mean_j'],label='CPU package')
    ax.bar(x,r['gpu_energy_mean_j'],bottom=r['cpu_energy_mean_j'],label='GPU device')
    ax.set_xticks(x); ax.set_xticklabels([model_label(t) for _,t in r.iterrows()],rotation=15,ha='right')
    ax.set_ylabel('Mean energy [J / sequence]'); ax.set_title(DATASETS[dataset]); ax.legend(frameon=False)
    fig.tight_layout(); return fig


def save_figure(fig,path):
    import matplotlib.pyplot as plt
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    plt.close(fig)


def build_analysis(out:Path):
    out=Path(out); tables=out/'tables'; figures=out/'figures'
    e=pd.read_csv(tables/'energy_validated_runs.csv')
    seq=energy_sequence_table(e)
    require(len(seq)==84,'Attese 84 coppie sequenza-metodo per energia/runtime')
    csv_out(seq,tables/'energy_runtime_by_sample.csv')
    # Keep compatibility with the previously promised runtime analysis.
    runtime_cols=['dataset','sample_id','model','net_hz','output_hz','n_repetitions','input_span_s',
                  'duration_source','sequence_wall_seconds','sequence_wall_repeat_std_s',
                  'real_time_factor','real_time_factor_repeat_std','speed_vs_realtime',
                  'amortized_ms_per_nominal_output_slot']
    csv_out(seq[runtime_cols],tables/'sequence_runtime_by_sample.csv')
    runtime_summary=dataset_summary(seq)
    csv_out(runtime_summary,tables/'energy_runtime_summary.csv')
    normalization=[]
    for _,r in seq.iterrows():
        for f in (5,20,50,100,200):
            normalization.append({'dataset':r['dataset'],'sample_id':r['sample_id'],'model':r['model'],
                                  'measured_net_hz':r['net_hz'],'normalization_frequency_hz':f,
                                  'amortized_ms':1000*r['sequence_wall_seconds']/(r['input_span_s']*f),
                                  'meaning':'fixed_measured_cost_divided_by_nominal_slots_not_new_inference_latency'})
    csv_out(pd.DataFrame(normalization),tables/'amortized_time_normalizations.csv')
    for ds in DATASETS:
        save_figure(plot_runtime(runtime_summary,ds),figures/(ds+'_runtime_rtf'))
        save_figure(plot_runtime(runtime_summary,ds,'wall_mean_s'),figures/(ds+'_runtime_wall_seconds'))
        save_figure(plot_energy(runtime_summary,ds),figures/(ds+'_energy_components'))
    completion=json.loads((out/'completion.json').read_text())
    combined=None
    if completion['accuracy_complete']:
        a=pd.read_csv(tables/'accuracy_by_sample.csv')
        require(len(a)==84,'Accuracy incompleta')
        require(not a.duplicated(['dataset','sample_id','model']).any(),'Accuracy duplicata')
        required=['config_hash','pck','mpjpe','n_points','sum_err_px','pck_0.2_correct_points']
        require(all(c in a for c in required),'Colonne accuracy mancanti')
        require(np.isfinite(a[['pck','mpjpe','n_points','sum_err_px','pck_0.2_correct_points']].to_numpy(float)).all(),
                'Metriche non finite: nessun punto viene scartato silenziosamente')
        # Exact protocol join, no nearest-frequency matching or old summary reuse.
        chosen=['dataset','sample_id','model','config_hash','pck','mpjpe','n_samples','n_points','sum_err_px',
                'pck_0.2_correct_points','pck_0.4','pck_0.4_correct_points','observed_output_hz',
                'median_output_period_s','max_output_gap_s','n_nonfinite_rows_excluded',
                'n_rows_outside_gt_span','n_degenerate_torso_rows','metric_protocol','prediction_csv',
                'prediction_first_ts_s','prediction_last_ts_s','gt_first_ts_s','gt_last_ts_s',
                'kf_pu','kf_muD','kf_muV']
        combined=seq.merge(a[chosen],on=['dataset','sample_id','model','config_hash'],
                           how='outer',validate='one_to_one',indicator=True)
        require((combined['_merge']=='both').all(),'Associazione Energy/accuracy incompleta o configurazione diversa')
        combined=combined.drop(columns='_merge')
        csv_out(combined,tables/'energy_accuracy_by_sample.csv')
        summary=dataset_summary(combined,with_accuracy=True)
        csv_out(summary,tables/'energy_accuracy_summary.csv')
        for ds in DATASETS:
            save_figure(plot_tradeoff(summary,ds),figures/(ds+'_energy_accuracy'))
            save_figure(plot_tradeoff(summary,ds,normalized=True),figures/(ds+'_energy_accuracy_normalized'))
        save_figure(plot_tradeoff_all(summary), figures/'energy_accuracy_both_datasets')
        save_figure(plot_tradeoff_all(summary, normalized=True), figures/'energy_accuracy_both_datasets_normalized')
        # Quality diagnostics are reported, not used to hide inconvenient results.
        coverage=combined[['dataset','sample_id','model','n_samples','observed_output_hz',
                           'median_output_period_s','max_output_gap_s','n_nonfinite_rows_excluded',
                           'n_rows_outside_gt_span','n_degenerate_torso_rows','prediction_first_ts_s',
                           'prediction_last_ts_s','gt_first_ts_s','gt_last_ts_s']]
        csv_out(coverage,tables/'accuracy_coverage_diagnostics.csv')
    readme=(
      'Energy / Accuracy / Runtime analysis\n'
      'Energy runs: 420 successful unique repetitions; failure attempts excluded by status, not by result.\n'
      'Accuracy complete: {complete} ({n}/84).\n'
      'Energy = CPU-package + GPU-device counters, J; not whole-PC energy and not process attribution.\n'
      'Power = energy / wall time, W. J/input-s is energy normalized by dataset time, not wall-time power.\n'
      'Runtime = existing energy wall_seconds, including startup/loading/shutdown; external cooldown excluded.\n'
      'RTF = total sequence-mean wall time / total input span. RTF<1 indicates average offline throughput only.\n'
      'Amortized ms at f Hz = 1000*T/(D*f). Changing f in normalization does NOT simulate a different net rate.\n'
      'Energy means weight sequences equally. Accuracy main summary uses ExpA point weighting; macro also exported.\n'
      '95% percentile bootstrap intervals resample 12 sequence-camera units after averaging energy repeats.\n'
      'These intervals are not uncertainty estimates for independent unseen subjects or accuracy stochasticity.\n'
      'Legacy evaluation: interpolate GT at prediction timestamps; endpoint hold; exclude nonfinite complete rows.\n'
      'PCK reference: GT shoulder_right (joint1) to hip_left (joint5), 13-joint order; percentages [0,100].\n'
      'Output/inference frequencies are nominal schedules; coverage diagnostics report achieved CSV rate.\n'
      'Prediction timestamps/coverage may differ among methods; no fabricated resampling or silent missing-method drop.\n'
      'For the final paper, inspect audit_energy.json warnings and accuracy_coverage_diagnostics.csv.\n'
    ).format(complete=completion['accuracy_complete'],n=completion['accuracy_completed'])
    (out/'ANALYSIS_README.txt').write_text(readme,encoding='utf-8')
    print('ANALYSIS: energy/runtime ready; accuracy {}/84; figures in {}'.format(completion['accuracy_completed'],figures),flush=True)
    return combined

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('results',type=Path)
    args=p.parse_args(); build_analysis(args.results)
