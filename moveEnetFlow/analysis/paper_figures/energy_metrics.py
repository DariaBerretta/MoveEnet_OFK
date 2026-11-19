"""Recompute paired energy benchmark accuracy and runtime from the original artifacts.

Accuracy always comes from prediction CSV + original GT, never from stored PCK/MPJPE.
Energy and wall time come from the original measurement tables and are retained.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from raw_metrics import evaluate_one,read_prediction,summarize


def host_path(path,data_root):
    """Translate a recorded Docker /data mount into the selected host data root."""
    path=Path(str(path))
    if path.is_file(): return path
    if str(path).startswith('/data/'):
        replacement=Path(data_root)/str(path)[len('/data/'):]
        if replacement.is_file(): return replacement
    raise FileNotFoundError('Recorded source not available: {}'.format(path))


def load_final_energy(data_root,valid_subset=True):
    """Join newly aggregated raw energy repetitions to exact original prediction/GT paths."""
    folder=Path(data_root)/'MoveEnet_OFK_results/Final/energy_accuracy_final_5hz_ofk'
    suffix='_valid_subset' if valid_subset else ''
    energy,energy_sources=load_runtime(data_root,valid_subset,include_moveiit=False)
    accuracy_path=folder/'tables'/('accuracy_by_sample'+suffix+'.csv')
    reference=pd.read_csv(accuracy_path)
    required={'dataset','sample_id','model','prediction_csv','gt_file'}
    if not required.issubset(reference.columns):
        raise ValueError('Missing source mappings in {}'.format(accuracy_path))
    reference['dataset']=reference.dataset.replace({'h36m':'eh36m'})
    if reference.duplicated(['dataset','sample_id','model']).any():
        raise ValueError('Duplicate accuracy sample/algorithm mapping')
    if energy.duplicated(['dataset','sample_id','model']).any():
        raise ValueError('Duplicate energy sample/algorithm mapping')
    mapping=reference[['dataset','sample_id','model','prediction_csv','gt_file']]
    paired=energy.merge(mapping,on=['dataset','sample_id','model'],how='left',validate='one_to_one',indicator=True)
    if not paired['_merge'].eq('both').all():
        raise ValueError('Energy cohort has unmapped prediction files; examine mapping')
    return folder,paired.drop(columns='_merge'),energy_sources+[str(accuracy_path)]


def recompute_final_energy(data_root,valid_subset=True,strict=True):
    """Re-evaluate each stored recording, preserving recorded energy and RTF."""
    folder,source,files=load_final_energy(data_root,valid_subset)
    rows,errors,cache=[],[],{}
    for item in source.to_dict('records'):
        try:
            prediction=folder/str(item['prediction_csv'])
            if not prediction.is_file(): raise FileNotFoundError(str(prediction))
            gt=host_path(item['gt_file'],data_root)
            metric=evaluate_one(prediction,gt,'eh36m' if item['dataset']=='h36m' else item['dataset'],cache)
            item.update(metric)
            item['dataset']='eh36m' if item['dataset']=='h36m' else item['dataset']
            item['method']=item['model']
            item['network_hz']=float(item['net_hz'])
            item['flow_hz']=200. if item['model']=='moveenetofk' else np.nan
            item['output_hz']=float(item['output_hz'])
            rows.append(item)
        except Exception as exc:
            errors.append(dict(dataset=item['dataset'],sample_id=item['sample_id'],model=item['model'],error=str(exc)))
            if strict: raise
    result=pd.DataFrame(rows)
    if result.empty: raise ValueError('No energy prediction/GT pairs were evaluable')
    if result.duplicated(['dataset','sample_id','model']).any(): raise ValueError('Duplicate sample-method pairs')
    return result,pd.DataFrame(errors,columns=['dataset','sample_id','model','error']),files


def recompute_moveiit_energy(data_root,strict=True):
    """Use existing matching MoveIIT 2D prediction recordings for measured energy samples.

    Important: energy runs used --no_csv; these accuracy predictions originate from
    matching ExpA/EPP recording replays at the SAME input sequence and network rate.
    They must not be mislabelled as predictions recorded during the energy runs.
    """
    root=Path(data_root)/'MoveEnet_OFK_results'
    folder=root/'Energy/moveiit_200_200_5_final'
    duration=pd.read_csv(folder/'samples_durations.csv')
    raw=pd.read_csv(folder/'rtf_energy_by_run.csv')
    if duration.duplicated('sample_id').any(): raise ValueError('Duplicate MoveIIT sample IDs')
    group=raw.groupby(['dataset','sample_id','model'],as_index=False).agg(
        energy_j=('measured_compute_energy_j','mean'),
        energy_repeat_std_j=('measured_compute_energy_j','std'),
        energy_j_per_input_second=('energy_j_per_input_second','mean'),
        real_time_factor=('real_time_factor','mean'),
        sequence_wall_seconds=('wall_seconds','mean'),
        n_repetitions=('wall_seconds','size'))
    group=group.merge(duration[['sample_id','subject','sequence','gt_file','duration_s']],on='sample_id',validate='many_to_one')
    expa=root/'Final_ROI_Gain1/Accuracy/moveiit/ExpA/raw/fp_0p005'
    epp=root/'Accuracy/moveiit/EPP_200Hz/stretch/raw'
    rows,failures,cache=[],[],{}
    for item in group.to_dict('records'):
        try:
            subject,sequence,model=item['subject'],item['sequence'],item['model']
            if model=='movenet_200hz':
                method='movenet'; period='0p005';suffix='moveenet_only_np_0p005_fp_0p005'
                csv=expa/'np_0p005'/('{}__{}__atis-s_{}.csv'.format(subject,sequence,suffix))
            elif model=='moveenetofk_5hz':
                method='moveenetofk';period='0p2';suffix='moveenet_ofk_np_0p2_fp_0p005'
                csv=expa/'np_0p2'/('{}__{}__atis-s_{}.csv'.format(subject,sequence,suffix))
            elif model=='eventpointpose_200hz':
                method='eventpointpose';period='0p005'
                csv=epp/('{}__{}__atis-s__epp_stretch_np_0p005.csv'.format(subject,sequence))
            else: raise ValueError('Unsupported measured MoveIIT method '+model)
            gt=host_path(item['gt_file'],data_root)
            item.update(evaluate_one(csv,gt,'moveiit',cache))
            item.update(method=method,network_hz=1/float(period.replace('p','.')),
                        flow_hz=200. if method=='moveenetofk' else np.nan,
                        output_hz=200.,dataset='moveiit',
                        source_run_kind='accuracy_from_separate_matching_replay')
            rows.append(item)
        except Exception as exc:
            failures.append(dict(sample_id=item['sample_id'],model=item['model'],error=str(exc)))
            if strict: raise
    result=pd.DataFrame(rows)
    if result.empty: raise ValueError('No MoveIIT accuracy/energy pairs')
    return result,pd.DataFrame(failures,columns=['sample_id','model','error']),[str(folder/'rtf_energy_by_run.csv'),str(folder/'samples_durations.csv')]


def energy_accuracy_summary(rows,protocol_id,cohort_id):
    """Preserve sequence means for energy and pooled joint observations for accuracy."""
    metrics=summarize(rows,protocol_id,cohort_id)
    extra=rows.groupby(['dataset','method','network_hz','flow_hz'],dropna=False,as_index=False).agg(
        energy_j_per_sequence=('energy_j','mean'),
        energy_sequence_sd_j=('energy_j','std'),
        energy_j_per_input_s=('energy_j_per_input_second','mean'),
        energy_sequence_sd_j_per_s=('energy_j_per_input_second','std'),
        rtf=('real_time_factor','mean'),
        rtf_sequence_sd=('real_time_factor','std'),
        wall_mean_s=('sequence_wall_seconds','mean'),
        wall_sequence_sd_s=('sequence_wall_seconds','std'),
        energy_repeat_sd_mean_j=('energy_repeat_std_j','mean'))
    return metrics.merge(extra,on=['dataset','method','network_hz','flow_hz'],validate='one_to_one')


def load_runtime(data_root,valid_subset=True,include_moveiit=True):
    """Reaggregate original energy measurements by repetition and input duration.

    Final benchmark: source_energy/energy_measurements.csv is the measurement
    source. The existing runtime_by_sample table supplies ONLY the original
    verified sequence durations and selected sample cohort, not the metrics.
    MoveIIT: its original rtf_energy_by_run.csv already contains duration and
    wall time per run; aggregate the repetitions by input recording.
    """
    folder=Path(data_root)/'MoveEnet_OFK_results'
    final=folder/'Final/energy_accuracy_final_5hz_ofk'
    suffix='_valid_subset' if valid_subset else ''
    original=final/'source_energy/energy_measurements.csv'
    cohort_file=final/'tables'/('energy_runtime_by_sample'+suffix+'.csv')
    measurements=pd.read_csv(original)
    cohort=pd.read_csv(cohort_file)
    measurements=measurements[measurements.status.astype(str).str.upper().eq('OK')].copy()
    measurements['dataset']=measurements.dataset.replace({'h36m':'eh36m'})
    cohort['dataset']=cohort.dataset.replace({'h36m':'eh36m'})
    keys=['dataset','sample_id','model']
    if cohort.duplicated(keys).any(): raise ValueError('Cohort has duplicate sequence/method')
    needed={'wall_seconds','measured_compute_energy_j'}
    if not needed.issubset(measurements): raise ValueError('Raw energy columns missing')
    for col in needed:
        measurements[col]=pd.to_numeric(measurements[col],errors='coerce')
    measurements=measurements.dropna(subset=list(needed))
    measurements=measurements[(measurements.wall_seconds>0)&(measurements.measured_compute_energy_j>=0)]
    group=measurements.groupby(keys,as_index=False).agg(
        energy_j=('measured_compute_energy_j','mean'),
        sequence_wall_seconds=('wall_seconds','mean'),
        n_repetitions=('wall_seconds','size'),
        energy_repeat_std_j=('measured_compute_energy_j','std'))
    duration=cohort[keys+['input_span_s','net_hz','output_hz']]
    frame=duration.merge(group,on=keys,how='left',validate='one_to_one')
    if frame[['energy_j','sequence_wall_seconds']].isna().any().any():
        raise ValueError('Selected energy cohort lacks raw measurements')
    if (frame.input_span_s<=0).any(): raise ValueError('Invalid recording duration')
    frame['real_time_factor']=frame.sequence_wall_seconds/frame.input_span_s
    frame['energy_j_per_input_second']=frame.energy_j/frame.input_span_s
    frame['method']=frame.model.replace({'yolopose':'yolo','moveenet_only':'movenet'})
    frame['network_hz']=pd.to_numeric(frame.net_hz)
    frame['flow_hz']=np.where(frame.method.eq('moveenetofk'),200.,np.nan)
    frame['output_hz']=pd.to_numeric(frame.output_hz)
    frame['protocol_id']='energy_final_5hz_ofk';frame['cohort_id']='valid_subset' if valid_subset else 'all_samples'
    frames=[frame]; paths=[str(original),str(cohort_file)]
    if include_moveiit:
        p=folder/'Energy/moveiit_200_200_5_final/rtf_energy_by_run.csv'
        m=pd.read_csv(p)
        m['method']=m.model.map({'movenet_200hz':'movenet','moveenetofk_5hz':'moveenetofk',
                                  'eventpointpose_200hz':'eventpointpose'})
        if m.method.isna().any(): raise ValueError('Unknown MoveIIT energy method')
        m['network_hz']=m.model.map({'movenet_200hz':200.,'moveenetofk_5hz':5.,'eventpointpose_200hz':200.})
        m['flow_hz']=np.where(m.method.eq('moveenetofk'),200.,np.nan)
        m['output_hz']=200.
        m['energy_j']=m['measured_compute_energy_j']
        m['protocol_id']='moveiit_200_200_5_final';m['cohort_id']='moveiit_energy'
        m=m.groupby(['dataset','sample_id','model','method','network_hz','flow_hz','output_hz','protocol_id','cohort_id'],dropna=False,as_index=False).agg(
            energy_j=('energy_j','mean'),energy_repeat_std_j=('energy_j','std'),
            real_time_factor=('real_time_factor','mean'),
            energy_j_per_input_second=('energy_j_per_input_second','mean'),
            sequence_wall_seconds=('wall_seconds','mean'),
            input_span_s=('input_span_s','mean'),n_repetitions=('wall_seconds','size'))
        frames.append(m);paths.append(str(p))
    return pd.concat(frames,ignore_index=True),paths


def runtime_summary(per_sample):
    frame=per_sample.copy()
    keys=['dataset','method','network_hz','flow_hz','output_hz','protocol_id','cohort_id']
    summary=frame.groupby(keys,dropna=False,as_index=False).agg(
        n_sequences=('sample_id','nunique'),rtf=('real_time_factor','mean'),
        rtf_sequence_sd=('real_time_factor','std'),
        energy_j_per_input_s=('energy_j_per_input_second','mean'),
        energy_sequence_sd_j_per_s=('energy_j_per_input_second','std'),
        energy_j_per_sequence=('energy_j','mean'),
        wall_mean_s=('sequence_wall_seconds','mean'),
        wall_sequence_sd_s=('sequence_wall_seconds','std'),
        energy_repeat_sd_mean_j=('energy_repeat_std_j','mean'))
    return summary
