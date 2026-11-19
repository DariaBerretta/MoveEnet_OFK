"""Recompute 13-joint 2D HPE accuracy from original prediction CSVs and GT.

Reuses the GT mapping/interpolation conventions in the existing MoveIIT
ExpA_2DGT_Analysis.ipynb and the native SKLT format in Experiment A.
No Vicon 3D files, no image/video inference, no implicit GT time shift.
"""
from pathlib import Path
import re
import json
import numpy as np
import pandas as pd

COORDS = ['joint{}_{}'.format(i, axis) for i in range(13) for axis in 'xy']
RESOLUTIONS = {'eh36m': (640,480), 'dhp19': (346,260), 'moveiit': (1280,720)}
PROJECTION_TO_HPE = [0,1,2,3,4,8,7,5,6,9,10,11,12]
CORRUPT_EH36M = {'cam4_s11_greeting','cam4_s9_eating','cam4_s9_greeting',
                 'cam4_s9_photo','cam4_s9_photo_1','cam4_s9_sittingdown_1',
                 'cam4_s9_waiting_1','cam4_s9_walking'}
METHODS = {'moveenet_only':'movenet','moveenet_ofk':'moveenetofk',
           'movenet':'movenet','yolopose':'yolo','openpose':'openpose',
           'net_flow':'eventpointpose','epp_stretch':'eventpointpose'}


def read_prediction(path):
    """Preserve the saved output timestamps, not the nominal output rate."""
    df = pd.read_csv(path)
    missing = sorted(set(['timestamp']+COORDS)-set(df.columns))
    if missing:
        raise ValueError('Prediction columns missing {} in {}'.format(missing,path))
    ts = pd.to_numeric(df.timestamp,errors='raise').to_numpy(dtype=float)
    xy = df[COORDS].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float).reshape(-1,13,2)
    if len(ts)<2 or not np.isfinite(ts).all() or not np.all(np.diff(ts)>0):
        raise ValueError('Invalid prediction timestamps: {}'.format(path))
    return ts,xy


def read_gt(path,dataset):
    """Read the 2D GT directly: original SKLT log or 2D projection CSV."""
    if dataset=='moveiit':
        df=pd.read_csv(path,encoding='utf-8-sig')
        df.columns=[str(c).strip() for c in df.columns]
        df=df.loc[:,~df.columns.str.match(r'^Unnamed:')]
        tcol='event_timestamp' if 'event_timestamp' in df else 'timestamp'
        cols=[x for x in df if x!=tcol]
        if len(cols)!=26: raise ValueError('Expected 26 GT coordinate columns in {}'.format(path))
        t=pd.to_numeric(df[tcol],errors='raise').to_numpy(dtype=float)
        xy=df[cols].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float).reshape(-1,13,2)
        if cols!=COORDS: xy=xy[:,PROJECTION_TO_HPE,:]
    else:
        ts,poses=[],[]
        pattern=re.compile(r'^\s*\d+\s+(\S+)\s+SKLT\s+\(([^)]*)\)')
        with Path(path).open(encoding='utf-8',errors='replace') as stream:
            for i,line in enumerate(stream,1):
                if not line.strip(): continue
                m=pattern.match(line)
                if m is None: raise ValueError('Malformed SKLT row {} in {}'.format(i,path))
                coords=np.fromstring(m.group(2),sep=' ')
                if coords.size!=26: raise ValueError('Wrong joint count on GT row {}'.format(i))
                ts.append(float(m.group(1))); poses.append(coords.reshape(13,2))
        t=np.asarray(ts,dtype=float);xy=np.asarray(poses,dtype=float)
    if len(t)<2 or not np.isfinite(t).all() or not np.all(np.diff(t)>0):
        raise ValueError('GT timestamp sequence invalid: {}'.format(path))
    width,height=RESOLUTIONS[dataset]
    valid=(np.isfinite(xy).all(axis=2)&(xy[...,0]>=0)&(xy[...,0]<width)&
           (xy[...,1]>=0)&(xy[...,1]<height))
    return t,xy,valid


def interpolate_2d(query,t,xy,valid,max_gap_s=.05,offset_s=0.0):
    """Do not extrapolate, hold endpoints, or interpolate invalid GT markers."""
    query=np.asarray(query,dtype=float)+float(offset_s)
    result=np.full((len(query),13,2),np.nan)
    mask=np.zeros((len(query),13),dtype=bool)
    inside=np.isfinite(query)&(query>=t[0])&(query<=t[-1])
    selected=np.flatnonzero(inside)
    if not len(selected): return result,mask,inside
    q=query[selected]
    high=np.searchsorted(t,q,side='left')
    clipped=np.minimum(high,len(t)-1)
    exact=(high<len(t))&(np.abs(t[clipped]-q)<=1.e-9)
    if exact.any():
        ids,src=selected[exact],clipped[exact]
        mask[ids]=valid[src]
        result[ids]=np.where(valid[src,:,None],xy[src],np.nan)
    between=~exact
    if between.any():
        ids,hi=selected[between],clipped[between]
        lo=hi-1
        delta=t[hi]-t[lo]
        good=(delta>0)&(delta<=max_gap_s)
        interp_valid=valid[lo]&valid[hi]&good[:,None]
        fraction=((q[between]-t[lo])/delta)[:,None,None]
        points=xy[lo]+fraction*(xy[hi]-xy[lo])
        mask[ids]=interp_valid
        result[ids]=np.where(interp_valid[:,:,None],points,np.nan)
    return result,mask,inside


def score(prediction,gt,valid,torso=(1,5),min_torso_px=1.0):
    """Pooled MPJPE and PCK@0.2 with independently recorded denominators."""
    pred_valid=np.isfinite(prediction).all(axis=2)
    shared=valid&pred_valid
    distance=np.linalg.norm(prediction-gt,axis=2)
    torso_length=np.linalg.norm(gt[:,torso[0]]-gt[:,torso[1]],axis=1)
    torso_ok=valid[:,torso[0]]&valid[:,torso[1]]&np.isfinite(torso_length)&(torso_length>=min_torso_px)
    pck_mask=shared&torso_ok[:,None]
    positive=pck_mask&(distance<=.2*torso_length[:,None])
    n=int(shared.sum()); pn=int(pck_mask.sum())
    if not n or not pn: raise ValueError('No joint observations for MPJPE or PCK')
    return {'sum_err_px':float(distance[shared].sum()),'n_points':n,
            'correct_pck02':int(positive.sum()),'n_pck_points':pn,
            'mpjpe_px':float(distance[shared].mean()),'pck_percent':100*positive.sum()/pn,
            'n_prediction_rows':len(prediction),'n_rows_torso_valid':int(torso_ok.sum())}


def evaluate_one(csv_path,gt_path,dataset,gt_cache=None,gt_offset_s=0.0,max_gt_gap_s=.05):
    """Evaluate one recording against the corresponding source GT."""
    gt_cache={} if gt_cache is None else gt_cache
    gt_path=Path(gt_path)
    if gt_path not in gt_cache: gt_cache[gt_path]=read_gt(gt_path,dataset)
    gt_t,gt_xy,gt_valid=gt_cache[gt_path]
    pred_t,pred_xy=read_prediction(csv_path)
    sampled,mask,inside=interpolate_2d(pred_t,gt_t,gt_xy,gt_valid,max_gt_gap_s,gt_offset_s)
    values=score(pred_xy,sampled,mask)
    values.update(prediction_csv=str(csv_path),gt_file=str(gt_path),
                  n_rows_in_gt_span=int(inside.sum()),gt_offset_s=float(gt_offset_s))
    return values


def parse_expa_file(path,dataset,gt_root):
    """Parse the original ExpA filename conventions, including non-MoveEnet models."""
    stem=Path(path).stem.strip()
    p=re.search(r'_np_(\d+p\d+)',stem)
    if p is None: raise ValueError('Missing network period in {}'.format(stem))
    period=float(p.group(1).replace('p','.'))
    if period<=0: raise ValueError('Invalid period')
    flow=re.search(r'_fp_(\d+p\d+)',stem)
    flow_hz=1/float(flow.group(1).replace('p','.')) if flow else np.nan
    method=next((v for k,v in METHODS.items() if re.search(r'(?:^|_)'+k+r'(?:_|$)',stem)),None)
    if method is None: raise ValueError('Unsupported method in {}'.format(stem))
    if dataset=='moveiit':
        m=re.match(r'^([^_]+)__(.+?)__atis-s',stem)
        if not m: raise ValueError('Unsupported MoveIIT sequence naming: {}'.format(stem))
        subject,sequence=m.groups()
        tag={'luna':'P1_luna','simon':'P11_simon'}.get(subject)
        if tag is None: raise ValueError('Unknown MoveIIT subject {}'.format(subject))
        gt=Path(gt_root)/'GT'/tag/(sequence+'_joint_projections.csv')
        sample=subject+'/'+sequence
    elif dataset=='eh36m':
        m=re.match(r'^(cam\d+_S\d+_.+?)(?:__ch\ddvs)?_(?:moveenet_|openpose|yolopose)',stem)
        if not m: raise ValueError('Unsupported eH36M filename {}'.format(stem))
        seq=m.group(1)
        gt=Path(gt_root)/'events'/seq/'ch0GT200Hzskeleton'/'data.log'
        sample=seq
    else:
        m=re.match(r'^(S\d+_\d+_\d+)__(ch\ddvs)_',stem)
        if not m: raise ValueError('Unsupported DHP19 filename {}'.format(stem))
        seq,channel=m.groups()
        gt=Path(gt_root)/seq/channel.replace('dvs','GT200Hzskeleton')/'data.log'
        sample=seq+'__'+channel
    # Optical-flow rate must not create duplicate MoveEnet-only configurations.
    if method!='moveenetofk': flow_hz=np.nan
    return dict(dataset=dataset,method=method,network_hz=1/period,
                network_period_s=period,flow_hz=flow_hz,
                sample_id=sample,gt_file=gt)


def summarize(rows,protocol_id,cohort_id):
    """Aggregate across the existing samples; repetitions are never independent samples."""
    if len(rows)==0: raise ValueError('No evaluated raw predictions')
    g=['dataset','method','network_hz','flow_hz']
    out=rows.groupby(g,dropna=False,as_index=False).agg(
        n_sequences=('sample_id','nunique'),sum_err_px=('sum_err_px','sum'),
        n_points=('n_points','sum'),correct_pck02=('correct_pck02','sum'),
        n_pck_points=('n_pck_points','sum'),
        mpjpe_sequence_sd_px=('mpjpe_px','std'),
        pck_sequence_sd_percent=('pck_percent','std'))
    out['mpjpe_px']=out.sum_err_px/out.n_points.replace(0,np.nan)
    out['pck_percent']=100*out.correct_pck02/out.n_pck_points.replace(0,np.nan)
    out['output_hz']=200.0
    out['protocol_id']=protocol_id;out['cohort_id']=cohort_id
    return out


def evaluate_expa(sources,gt_roots,gt_offsets=None,strict=False,skip_corrupt=True,limit=None):
    """Evaluate all selected source folders with explicit method provenance."""
    rows,failures=[],[]
    cache={};seen=set();gt_offsets=gt_offsets or {}
    for dataset,method,folder in sources:
        folder=Path(folder)
        if not folder.is_dir():
            failures.append(dict(dataset=dataset,method=method,source=str(folder),reason='SOURCE_FOLDER_MISSING'))
            if strict: raise FileNotFoundError(str(folder))
            continue
        paths=sorted(folder.rglob('*.csv'))
        if limit is not None: paths=paths[:limit]
        for path in paths:
            try:
                meta=parse_expa_file(path,dataset,gt_roots[dataset])
                if meta['method']!=method: continue
                if dataset=='eh36m' and skip_corrupt and meta['sample_id'].lower() in CORRUPT_EH36M: continue
                key=(dataset,method,meta['sample_id'],meta['network_hz'],
                     None if method!='moveenetofk' else meta['flow_hz'])
                if key in seen: continue
                if not meta['gt_file'].is_file(): raise FileNotFoundError(str(meta['gt_file']))
                res=evaluate_one(path,meta['gt_file'],dataset,cache,gt_offsets.get(dataset,0.0))
                res.update(meta)
                rows.append(res);seen.add(key)
            except Exception as exc:
                failures.append(dict(dataset=dataset,method=method,source=str(path),reason=str(exc)))
                if strict: raise
    return pd.DataFrame(rows),pd.DataFrame(failures,columns=['dataset','method','source','reason'])


def resolve_data_root(override=None):
    """Locate actual host/container data root using a folder existence check."""
    import os
    choices=[override,os.environ.get('MOVEENET_DATA_ROOT'),'/data',
             str(Path.home()/'data'),'/home/dberretta-iit.local/data']
    for path in choices:
        if path and (Path(path).expanduser()/'MoveEnet_OFK_results').is_dir():
            return Path(path).expanduser()
    raise FileNotFoundError('Could not find dataset root. Set DATA_ROOT_OVERRIDE to the path containing MoveEnet_OFK_results. Checked: {}'.format(choices))


def save_tables(output,**tables):
    """Write new derived tables; never overwrite original experiment results."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,frame in tables.items(): frame.to_csv(output/(name+'.csv'),index=False)
    return output
