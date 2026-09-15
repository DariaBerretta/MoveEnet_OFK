#!/usr/bin/env python3
"""Finish the Energy--Accuracy--Runtime study without rerunning energy.

HOST entry point, Python >= 3.8, standard library only. No automatic package
installation, git pull, build, or changes to the original energy CSVs.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from final_common import (AuditError, atomic_json, audit_session, hash_obj,
                          log_file_for, replay_command, require, sha256_file, write_csv)

HERE = Path(__file__).resolve().parent
DEFAULT_SESSION = 'energy_accuracy_final_5hz_ofk'

def message(s):
    print(s, flush=True)

def run_checked(argv, timeout=60):
    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=timeout)
    require(result.returncode == 0, '{}\n{}'.format(' '.join(argv[:5]), result.stderr[-5000:]))
    return result.stdout

def docker_info(container):
    data = json.loads(run_checked(['docker','inspect',container],30))[0]
    require(data['State']['Running'], 'Container non in esecuzione: ' + container)
    return data

def container_path(path, mounts):
    path = Path(path).resolve()
    candidates = []
    for mount in mounts:
        src = Path(mount.get('Source','/nonexistent')).resolve()
        try:
            relative = path.relative_to(src)
        except ValueError:
            continue
        candidates.append((len(src.parts), str(Path(mount['Destination'])/relative)))
    require(candidates, 'Percorso host non montato nel Docker: {}. Estrarre ExpFinal nella repository montata.'.format(path))
    return max(candidates)[1]

def host_energy_active():
    names = ('run_energy_accuracy_final.sh','run_energy_experiment.sh','measure_energy.py')
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = p.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            if any(Path(a).name in names for a in args if a):
                found.append(p.name)
        except (FileNotFoundError, PermissionError):
            pass
    return found

def snapshot_files(session, out, items, report):
    dest = out/'source_energy'; dest.mkdir(parents=True,exist_ok=True)
    for name in ('energy_measurements.csv','run_manifest.csv','metadata.txt'):
        p = session/name
        if not p.exists():
            continue
        copy = dest/name
        if copy.exists():
            require(sha256_file(p) == sha256_file(copy),
                    'Sessione energetica cambiata rispetto allo snapshot: ' + str(p))
        else:
            shutil.copy2(str(p),str(copy))
    (dest/'logs').mkdir(exist_ok=True)
    for item in items:
        p = Path(item['log_path']); copy = dest/'logs'/p.name
        if copy.exists():
            require(sha256_file(copy) == item['log_sha256'], 'Log energetico cambiato: ' + str(p))
        else:
            shutil.copy2(str(p),str(copy))
    atomic_json(out/'audit_energy.json',report)


def export_energy_rows(out, items, preflight):
    rows = []
    for item in items:
        r = dict(item['original'])
        r.update(job_id='_'.join((item['dataset'],item['sample_id'],item['model'])),
                 dataset=item['dataset'], repetition=item['repetition'],
                 config_hash=item['config_hash'], output_period_s=item['output_period_s'])
        r.update({k:v for k,v in item['diagnostics'].items() if v is not None})
        d = preflight['durations'][item['parsed']['options']['--data_file']]
        r.update(d)
        r['cpu_gpu_energy_scope'] = 'CPU_package_plus_GPU_device_not_process_or_whole_machine'
        r['runtime_scope'] = 'whole_command_including_startup_and_shutdown_excluding_runner_cooldown'
        rows.append(r)
    write_csv(out/'tables'/'energy_validated_runs.csv',rows)


def make_worker_job(job, attempt, preflight, plan, mounts):
    container_dir = container_path(attempt,mounts)
    prediction = str(Path(container_dir)/'predictions.csv')
    server_log = str(Path(container_dir)/'server.log')
    replay = replay_command(job,prediction,server_log)
    j = dict(job)
    j.update(attempt_dir_container=container_dir, prediction_container=prediction,
             replay_argv=replay, hpecore_root=plan['hpecore_root'],
             input_span_s=preflight['durations'][job['data_file']]['input_span_s'],
             identity_hash=preflight['identity_hash'])
    # The snapshot hashes are checked before launching each job.
    j['required_hashes'] = {p:info['sha256'] for p,info in preflight['files'].items()}
    j['required_hashes'].update({p:info['sha256'] for p,info in preflight['source_files'].items()})
    j['required_hashes'][job['gt_file']] = preflight['ground_truth'][job['gt_file']]['sha256']
    j['required_inputs'] = {job['data_file']: preflight['input_files'][job['data_file']]}
    return j


def completed_accuracy(out, job, identity_hash):
    p = out/'metrics'/ (job['job_id'] + '.json')
    csvp = out/'predictions'/ (job['job_id'] + '.csv')
    if not p.is_file() or not csvp.is_file():
        return False
    m = json.loads(p.read_text())
    require(m['config_hash'] == job['config_hash'] and m['identity_hash'] == identity_hash,
            'Accuracy esistente con configurazione/binari differenti: ' + job['job_id'])
    require(sha256_file(csvp) == m['prediction_sha256'], 'CSV predizioni modificato: ' + str(csvp))
    return True


def write_partial_accuracy(out, jobs):
    rows=[]
    for job in jobs:
        p=out/'metrics'/(job['job_id']+'.json')
        if not p.is_file():
            continue
        m=json.loads(p.read_text()); diagnostics=m.pop('diagnostics',{})
        m['kf_pu'],m['kf_muD'],m['kf_muV'] = m.pop('kf') or ('','','')
        m['prediction_csv'] = 'predictions/'+job['job_id']+'.csv'
        m['gt_file'] = job['gt_file']
        for k,v in diagnostics.items():
            m['replay_'+k]=v
        rows.append(m)
    target='accuracy_by_sample.csv' if len(rows)==len(jobs) else 'accuracy_by_sample_PARTIAL.csv'
    write_csv(out/'tables'/target,rows,
              None if rows else ['dataset','sample_id','model','pck','mpjpe'])
    atomic_json(out/'completion.json', {'accuracy_complete':len(rows)==len(jobs),
                                      'accuracy_completed':len(rows),'accuracy_required':len(jobs)})
    return len(rows)


def execute_job(container, worker, python, jobpath, attempt, timeout_s):
    logpath=attempt/'worker.log'
    with logpath.open('w') as log:
        process=subprocess.Popen(['docker','exec',container,python,worker,'run-job',jobpath],
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.monotonic(); next_msg=start+30
        try:
            while process.poll() is None:
                now=time.monotonic()
                if now>=next_msg:
                    message('    in esecuzione: {:.0f}s / limite {:.0f}s; log {}'.format(now-start,timeout_s,attempt/'run.log'))
                    next_msg=now+30
                # Safety deadline for worker/driver/transport failure. The worker
                # normally applies its own deadline and kills its owned group.
                if now-start > timeout_s+60:
                    (attempt/'CANCEL').write_text('host deadline\n')
                    raise AuditError('Worker non terminato entro la scadenza. Batch fermato: ' + str(attempt))
                time.sleep(.5)
            return process.returncode
        except BaseException:
            (attempt/'CANCEL').write_text('host interrupted\n')
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=15)
            if process.poll() is None:
                # This only terminates the local docker client. The shared
                # CANCEL file remains for the remote watchdog to see.
                process.terminate()
            raise


def snapshot_repository(out):
    repo=HERE.parents[2]
    if (repo/'.git').exists():
        for name,cmd in [('git_head.txt',['rev-parse','HEAD']),
                         ('git_status.txt',['status','--short']),
                         ('git_diff.patch',['diff','--no-ext-diff','--','moveEnetFlow','docker','model_mounts'])]:
            r=subprocess.run(['git','-C',str(repo)]+cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            destination = out/'provenance'/name
            if not destination.exists():
                destination.write_text(r.stdout,encoding='utf-8')
    hashes={p.name:sha256_file(p) for p in HERE.glob('*.py')}
    recorded = out/'provenance'/'analysis_code_hashes.json'
    if recorded.exists():
        require(json.loads(recorded.read_text()) == hashes,
                'Codice ExpFinal cambiato durante una sessione: preservare la versione usata per il replay.')
    else:
        atomic_json(recorded, hashes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',default=DEFAULT_SESSION)
    parser.add_argument('--energy-dir',type=Path)
    parser.add_argument('--results-dir',type=Path)
    parser.add_argument('--container',default='moveEnetOFK')
    parser.add_argument('--container-python',default='/opt/venv/bin/python')
    parser.add_argument('--hpecore-root',default='/usr/local/src/hpe-core')
    parser.add_argument('--prepare-only',action='store_true',help='Audit + runtime/energy analysis, no HPE inference')
    parser.add_argument('--analysis-only',action='store_true',help='Rebuild tables/figures from existing new results, no HPE inference')
    parser.add_argument('--resume',action='store_true',help='Reuse only hash-validated completed accuracy jobs')
    parser.add_argument('--max-attempts',type=int,default=2,help='Attempts per pending accuracy job per invocation (default 2)')
    args=parser.parse_args()
    require(not Path('/.dockerenv').exists(), 'Lanciare finish_experiment.sh sull HOST, non dentro Docker.')
    require(1<=args.max_attempts<=3,'--max-attempts deve essere tra 1 e 3')
    session=(args.energy_dir or Path.home()/'data/MoveEnet_OFK_results/Energy'/args.session).expanduser().resolve()
    require(session.is_dir(),'Directory Energy inesistente: '+str(session))
    out=(args.results_dir or session.parent.parent/'Final'/session.name).expanduser().resolve()
    require(out!=session and session not in out.parents, 'Output separato dalla sessione Energy richiesto.')
    for sub in ('provenance','tables','figures','attempts','metrics','predictions'):
        (out/sub).mkdir(parents=True,exist_ok=True)
    lockfile=Path('/tmp')/('moveenet_final_{}_{}.lock'.format(os.getuid(),args.container))
    with lockfile.open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        info=docker_info(args.container); mounts=info['Mounts']
        worker=container_path(HERE/'container_worker.py',mounts)
        cout=container_path(out,mounts)
        if args.analysis_only:
            subprocess.run(['docker','exec',args.container,args.container_python,worker,'analysis',cout],check=True)
            message('Analisi aggiornata: '+str(out)); return 0
        active=host_energy_active()
        require(not active,'Esperimento energetico ancora attivo sull host (PID): '+repr(active))
        message('[1/4] Audit della sessione energetica (sola lettura): '+str(session))
        jobs,items,audit=audit_session(session,args.container,repeats=5)
        snapshot_files(session,out,items,audit)
        message('      {} misure OK uniche; {} coppie sequenza-metodo; {} tentativi falliti esclusi.'.format(
            audit['successful_unique_runs'],len(jobs),len(audit['ignored_failed_attempts'])))
        for w in audit['warnings']:
            message('      ATTENZIONE: '+w)
        signature=hash_obj({'input_hashes':audit['input_hashes'],'jobs':jobs,
                            'hpecore_root':args.hpecore_root,'container_python':args.container_python})
        plan={'signature':signature,'jobs':jobs,'audit':audit,'hpecore_root':args.hpecore_root,
              'source_energy_dir':str(session),'results_dir':str(out)}
        planfile=out/'plan.json'
        if planfile.exists():
            old=json.loads(planfile.read_text())
            require(old['signature']==signature, 'Piano cambiato: non mescolare protocolli nella stessa directory Final.')
            require(args.resume or args.prepare_only,
                    'Directory Final gia preparata. Usare --resume per NON ripetere accuracy completate.')
        else:
            atomic_json(planfile,plan)
        write_csv(out/'plan.csv',[{k:v for k,v in j.items() if k not in ('prefix','base_argv','energy_labels')}
                                   for j in jobs])
        snapshot_repository(out)
        message('[2/4] Preflight di TUTTI gli input/GT/binari, senza inferenze.')
        result=run_checked(['docker','exec',args.container,args.container_python,worker,'preflight',
                            container_path(planfile,mounts)],timeout=600)
        lines=[x[len('FINAL_JSON:'):] for x in result.splitlines() if x.startswith('FINAL_JSON:')]
        require(len(lines)==1, 'Risposta preflight non valida: '+result[-5000:])
        preflight=json.loads(lines[0]); prefile=out/'preflight.json'
        if prefile.exists():
            previous=json.loads(prefile.read_text())
            require(previous['identity_hash']==preflight['identity_hash'],
                    'Binari/checkpoint/sorgenti/input cambiati dopo il primo preflight. Non mescolare accuracy.')
        else:
            atomic_json(prefile,preflight)
        export_energy_rows(out,items,preflight)
        for j in jobs:
            d=preflight['durations'][j['data_file']]
            j['input_span_s']=d['input_span_s']
        # Store durations separately; the frozen replay-plan signature stays unchanged.
        atomic_json(out/'durations.json',preflight['durations'])
        for job in jobs:
            completed_accuracy(out, job, preflight['identity_hash'])
        n=write_partial_accuracy(out,jobs)
        subprocess.run(['docker','exec',args.container,args.container_python,worker,'analysis',cout],check=True)
        message('      Runtime ed energia gia disponibili in '+str(out/'tables'))
        if args.prepare_only:
            message('PREPARE completato: nessun modello HPE eseguito.'); return 0
        message('[3/4] Accuracy: {} coppie, {} gia completate. NESSUNA nuova misura energetica.'.format(len(jobs),n))
        for index,job in enumerate(jobs,1):
            if completed_accuracy(out,job,preflight['identity_hash']):
                message('[{:02d}/{}] SKIP {}'.format(index,len(jobs),job['job_id'])); continue
            root=out/'attempts'/job['job_id']; root.mkdir(parents=True,exist_ok=True)
            previous_attempts=[int(p.name[7:]) for p in root.glob('attempt*') if p.is_dir() and p.name[7:].isdigit()]
            base=max(previous_attempts,default=0)
            success=False
            for retry in range(1,args.max_attempts+1):
                attempt=root/('attempt{:02d}'.format(base+retry)); attempt.mkdir()
                worker_job=make_worker_job(job,attempt,preflight,plan,mounts)
                jobpath=attempt/'job.json'; atomic_json(jobpath,worker_job)
                message('[{:02d}/{}] {} net={:g} Hz out={:g} Hz; tentativo {}'.format(
                    index,len(jobs),job['job_id'],1/job['net_period_s'],1/job['output_period_s'],base+retry))
                rc=execute_job(args.container,worker,args.container_python,container_path(jobpath,mounts),
                               attempt,job['timeout_s'])
                rp=attempt/'result.json'
                result=json.loads(rp.read_text()) if rp.exists() else {'status':'FAILED','error':'Worker senza result.json'}
                if rc==0 and result['status']=='OK':
                    metrics=result['metrics']
                    src=attempt/'predictions.csv'; dst=out/'predictions'/(job['job_id']+'.csv')
                    require(src.is_file() and sha256_file(src)==metrics['prediction_sha256'], 'Predizioni non integre')
                    shutil.copy2(str(src),str(dst))
                    metrics['accuracy_replay_wall_seconds']=result['accuracy_replay_wall_seconds']
                    metrics['attempt']=str(attempt.relative_to(out))
                    atomic_json(out/'metrics'/(job['job_id']+'.json'),metrics)
                    write_partial_accuracy(out,jobs)
                    message('      OK: {} righe; MPJPE={:.4f} px; PCK@0.2={:.3f}%'.format(
                        metrics['n_samples'],metrics['mpjpe_px'],metrics['pck_0.2']))
                    success=True; break
                message('      {}: {}. Log: {}'.format(result.get('status'),result.get('error',''),attempt))
                if retry<args.max_attempts:
                    message('      Un solo retry tecnico; nessun risultato riuscito viene riselezionato.')
                    time.sleep(2)
            if not success:
                atomic_json(out/'last_failure.json',{'job_id':job['job_id'],'attempt':str(attempt),'result':result})
                raise AuditError('Batch fermato su {}. Le accuracy gia riuscite sono conservate. Dettagli: {}'.format(job['job_id'],out/'last_failure.json'))
        message('[4/4] Tabelle complete e figure finali.')
        require(write_partial_accuracy(out,jobs)==84,'Numero accuracy diverso da 84')
        subprocess.run(['docker','exec',args.container,args.container_python,worker,'analysis',cout],check=True)
        for name,expected in audit['input_hashes'].items():
            require(sha256_file(session/name)==expected,'File Energy cambiato durante la procedura: '+name)
        message('COMPLETATO: 84 accuracy, energia originale immutata, runtime ricavato dai 420 run.')
        message('Risultati: '+str(out))
        return 0

if __name__=='__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nInterrotto: watchdog remoto notificato; usare --resume dopo il controllo dei processi.',file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print('\nERROR: '+str(exc),file=sys.stderr)
        sys.exit(2)
