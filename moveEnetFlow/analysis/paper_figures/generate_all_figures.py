"""Generate every available, separately exported paper figure without inventing missing results.

Usage:
    python generate_all_figures.py
    python generate_all_figures.py --config sources.json
    python generate_all_figures.py --scan /data/MoveEnet_OFK_results

Plots use original cached notebook results by default. Paths configured in sources.json
replace the cached tables only for their corresponding figure family/dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_figures import (canonical_dataset, load_style, render_accuracy_rate,
                           render_energy_accuracy, render_energy_rtf, export_figure)
from adapters import read_expa_summary, read_energy_summary, read_moveiit_accuracy_summary

ROOT = Path(__file__).resolve().parent
METRICS = ('mpjpe_px', 'pck_percent')


def resolve_path(path):
    if not path: return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def get_df(path, family, dataset, config):
    path = resolve_path(path)
    if path is None: return None, None
    if not path.is_file():
        raise FileNotFoundError('Configured {} source not found: {}'.format(family, path))
    preview = pd.read_csv(path, nrows=2)
    if {'dataset', 'method', 'network_hz', 'protocol_id', 'cohort_id'}.issubset(preview.columns):
        data = pd.read_csv(path)
    elif family == 'accuracy_rate' and dataset != 'moveiit':
        data = read_expa_summary(path, dataset=dataset,
            protocol_id='ExpA_external_PCK_joint_pair_must_be_checked',
            cohort_id='ExpA_external_dataset_summary')
    elif family == 'accuracy_rate' and dataset == 'moveiit':
        data = read_moveiit_accuracy_summary(path,
            protocol_id='MoveIIT_ExpA_external_PCK_1_5_requires_verification',
            cohort_id='MoveIIT_ExpA_external')
    elif family in ('energy_accuracy','energy_rtf'):
        data = read_energy_summary(path, dataset=None if dataset=='all' else dataset,
            protocol_id=family+'_external_review_source',
            cohort_id=family+'_external_review_cohort')
    else:
        raise ValueError('Unsupported source format: {}'.format(path))
    if dataset != 'all':
        data = data[data.dataset.map(canonical_dataset).eq(dataset)].copy()
        if data.empty: raise ValueError('Source {} has no rows for {}'.format(path, dataset))
    if family == 'accuracy_rate':
        data = data[data.method.isin(('movenet','moveenetofk'))].copy()
        if data.empty: raise ValueError('No MoveEnet-only/MoveEnetOFK in {}'.format(path))
        # OFK flow is specified by the verified experiment configuration, not inferred
        # from a filename, and can be overridden when the CSV provides flow_hz.
        expected_flow = config.get('expa_ofk_flow_hz')
        missing = data.method.eq('moveenetofk') & data.flow_hz.isna()
        if missing.any():
            if expected_flow is None:
                raise ValueError('OFK flow rate missing: declare expa_ofk_flow_hz in sources.json')
            data.loc[missing, 'flow_hz'] = float(expected_flow)
        baseline = data.method.eq('movenet')
        if 'source_flow_hz' in data and config.get('baseline_source_flow_hz') is not None:
            copy_match = baseline & np.isclose(data.source_flow_hz.astype(float).fillna(-1), float(config['baseline_source_flow_hz']))
            if copy_match.any():
                data = data[~baseline | copy_match].copy()
    if family == 'energy_rtf':
        data = data.dropna(subset=['energy_j_per_input_s','rtf']).copy()
    return data, path


def scan_sources(root):
    root = Path(root).expanduser()
    if not root.exists():
        print('Results directory does not exist in this environment:', root)
        return
    exact = ('dataset_metrics.csv','comparison_dataset_weighted_metrics.csv',
             'accuracy_summary.csv','accuracy_summary_3methods.csv',
             'energy_accuracy_summary_valid_subset.csv','energy_runtime_summary_valid_subset.csv',
             'energy_runtime_dataset_summary.csv','energy_runtime_dataset_summary_5methods.csv',
             'energy_accuracy_12samples_5methods.csv')
    print('Possible original CSV sources (review cohort/protocol before selecting):')
    count = 0
    for path in root.rglob('*.csv'):
        if path.name in exact or path.name.endswith('_dataset_metrics.csv'):
            print(' ',path)
            count+=1
            if count>=150:
                print('  ... limit 150 candidates'); break
    if not count: print('  No recognized CSV files in this directory.')


def emit(fig, stem, style, data, files, warnings, result):
    files = [p for p in files if p is not None]
    metadata = {
        'source_files': [{'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
        'plotted_rows':data.astype(object).where(pd.notna(data),None).to_dict('records'),
        'warnings':warnings,
        'pck_definition_for_paper':'GT right shoulder (1) to left hip (5); plotter does not recalculate PCK',
        'status':'cached_notebook_values_or_explicit_user_CSV_not_recomputed_from_raw'
    }
    exported=export_figure(fig,stem,style,metadata=metadata)
    plt.close(fig)
    result.append((stem,exported))
    print('OK', stem.relative_to(ROOT))


def render_family(family, datasets, data, style, files, warnings, result, skip_pck=False):
    output = ROOT / 'output' / family
    data = data.copy()
    for dataset in datasets:
        sub = data[data.dataset.map(canonical_dataset).eq(dataset)].copy()
        if sub.empty: continue
        for metric in METRICS if family != 'energy_rtf' else ('energy_j_per_input_s',):
            if family == 'accuracy_rate' and metric=='pck_percent' and skip_pck:
                print('SKIPPED',family,dataset,'PCK: source has not been verified as joint 1/5')
                continue
            if family == 'energy_rtf':
                fig=render_energy_rtf(sub,style,datasets=[dataset])
                stem=output/dataset
            elif family == 'accuracy_rate':
                fig=render_accuracy_rate(sub,style,datasets=[dataset],metrics=[metric])
                stem=output/dataset/('mpjpe' if metric=='mpjpe_px' else 'pck')
            else:
                fig=render_energy_accuracy(sub,style,datasets=[dataset],metrics=[metric])
                stem=output/dataset/('mpjpe' if metric=='mpjpe_px' else 'pck')
            emit(fig,stem,style,sub,files,warnings,result)
        if family != 'energy_rtf' and not skip_pck:
            if family == 'accuracy_rate': fig=render_accuracy_rate(sub,style,datasets=[dataset])
            else: fig=render_energy_accuracy(sub,style,datasets=[dataset])
            emit(fig,output/dataset/'mpjpe_pck_pair',style,sub,files,warnings,result)
    if len(datasets)>1 and family in ('accuracy_rate','energy_accuracy') and not skip_pck:
        available=[d for d in datasets if d in set(data.dataset.map(canonical_dataset))]
        if len(available)>1:
            if family=='accuracy_rate': fig=render_accuracy_rate(data,style,datasets=available)
            else: fig=render_energy_accuracy(data,style,datasets=available)
            emit(fig,output/'all_datasets',style,data,files,warnings,result)
    elif len(datasets)>1 and family=='energy_rtf':
        available=[d for d in datasets if d in set(data.dataset.map(canonical_dataset))]
        if len(available)>1:
            fig=render_energy_rtf(data,style,datasets=available)
            emit(fig,output/'all_datasets',style,data,files,warnings,result)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default='sources.json')
    ap.add_argument('--scan',type=Path,help='Only list possible source CSV paths under this results directory')
    args=ap.parse_args()
    if args.scan:
        scan_sources(args.scan); return
    conf_path=resolve_path(args.config)
    cfg=json.loads(conf_path.read_text())
    style=load_style(ROOT/'style.json')
    result=[]; missing=[]; failures=[]

    # Source 1: original cached notebook summary for energy-vs-accuracy.
    try:
        path=cfg['energy_accuracy'].get('all') or 'data/energy_accuracy_preview.csv'
        summary,source=get_df(path,'energy_accuracy','all',cfg)
        datasets=[d for d in style['dataset_order'] if d in set(summary.dataset.map(canonical_dataset))]
        render_family('energy_accuracy',datasets,summary,style,[source],
            ['Move-IIT GT alignment provisional; no CI in cached table.'], result)
    except Exception as e:
        failures.append('energy_accuracy: '+str(e)); print('ERROR energy_accuracy:',e)

    # Source 2: ablation. Only two methods are included; 59-recording Move-IIT cached
    # table has 1/5 PCK. Original ExpA eH36M/DHP19 have historical 2/6 PCK.
    for dataset in style['dataset_order']:
        path=cfg['accuracy_rate'].get(dataset)
        if not path:
            missing.append('accuracy_rate/{}: CSV not configured'.format(dataset))
            continue
        try:
            rate,source=get_df(path,'accuracy_rate',dataset,cfg)
            skip_pck = dataset in ('eh36m','dhp19') and not cfg.get('expa_pck_1_5_recomputed',False)
            note=['eH36M/DHP19 ExpA source PCK protocol must be independently verified as 1/5'] if skip_pck else []
            if dataset=='moveiit': note=['2D GT temporal offset explicitly provisional in the saved analysis notebook']
            render_family('accuracy_rate',[dataset],rate,style,[source],note,result,skip_pck=skip_pck)
            if skip_pck: missing.append('accuracy_rate/{}/pck: PCK source needs 1/5 recomputation/verification'.format(dataset))
        except Exception as e:
            failures.append('accuracy_rate/{}: {}'.format(dataset,e));print('ERROR accuracy_rate',dataset,e)

    # Source 3: normalized energy vs RTF. eH36M/DHP19 require their original
    # normalized summaries; not safe to derive from aggregate energy means.
    for dataset in style['dataset_order']:
        path=cfg['energy_rtf'].get(dataset)
        if not path:
            missing.append('energy_rtf/{}: CSV not configured'.format(dataset)); continue
        try:
            runtime,source=get_df(path,'energy_rtf',dataset,cfg)
            if runtime.empty:
                raise ValueError('RTF and normalized input-energy columns absent')
            render_family('energy_rtf',[dataset],runtime,style,[source],
                ['J / input second (not average power); original cached RTF and energy denominators'],result)
        except Exception as e:
            failures.append('energy_rtf/{}: {}'.format(dataset,e));print('ERROR energy_rtf',dataset,e)

    report=['# Generated paper figures','',
            'This package exports one **figure per dataset and metric**, plus MPJPE/PCK pairs.',
            'Each figure has **PNG, SVG, PDF** and matching `.meta.json` provenance.','',
            '## Created ({} figure stems; {} exported files)'.format(len(result),len(result)*3),'']
    for stem,_ in result: report.append('- `'+str(stem.relative_to(ROOT))+'`')
    report.extend(['','## Not generated (missing or unverified numerical source)',''])
    report.extend('- '+x for x in missing)
    if failures:
        report.extend(['','## Errors','']+['- '+x for x in failures])
    report.extend(['','## How to enable missing figures','',
                   '1. Find `*_dataset_metrics.csv` for eH36M/DHP19 ExpA, and `energy_runtime_summary_valid_subset.csv` for normalized energy/RTF.','2. Edit `sources.json` and set the appropriate absolute paths.','3. Run `python generate_all_figures.py` again.','',
                   'Do not set `expa_pck_1_5_recomputed=true` unless the eH36M/DHP19 PCK summaries have actually been computed with reference joints 1 and 5.'])
    (ROOT/'output/FIGURE_MANIFEST.md').write_text('\n'.join(report)+'\n')
    print('\nEXPORTED',len(result),'figures x 3 formats =',len(result)*3,'files')
    print('MISSING',len(missing),'| ERRORS',len(failures))
    print('Manifest:',ROOT/'output/FIGURE_MANIFEST.md')
    if failures: raise SystemExit(2)

if __name__=='__main__': main()
