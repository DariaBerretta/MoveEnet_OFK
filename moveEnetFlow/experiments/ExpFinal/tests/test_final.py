"""Deterministic local tests. No Docker, YARP, model weights or real experiment data.
Run: python3 -m unittest discover -s tests -v
"""
import ast
import copy
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from final_common import (AuditError, MODELS, EXPECTED_NET, BINS, EXPECTED_KF,
                          audit_session, write_csv, sha256_file, replay_command,
                          options, derive_gt, completed_log, log_diagnostics)
from finish_experiment import container_path, snapshot_files, export_energy_rows


def fixture(root):
    session = Path(root)/'Energy'/'energy_accuracy_final_5hz_ofk'
    (session/'runs').mkdir(parents=True, exist_ok=True)
    energy = []; manifest = []
    for ds in MODELS:
        for i in range(1,13):
            sample = ('H' if ds == 'h36m' else 'D') + str(i).zfill(2)
            seq = 'cam2_S9_Action_{}'.format(i) if ds == 'h36m' else 'S13_1_{}'.format(i)
            for model in MODELS[ds]:
                for rep in range(1,6):
                    rgb = model in ('openpose','yolo')
                    if ds=='h36m':
                        data = '/data/eh36m_testing_set_S9S11/' + ('rgb/'+seq+'.mp4' if rgb else 'events/'+seq+'/ch0dvs/data.log')
                    else:
                        data = '/data/dhp19_testing_set_S13toS17/'+seq+'/ch2dvs/data.log'
                    net=EXPECTED_NET[ds,model]
                    prefix=['env','-u','VIRTUAL_ENV','PATH=/usr/bin:/bin'] if model=='openpose' else ['env','VIRTUAL_ENV=/opt/venv','PATH=/opt/venv/bin:/usr/bin:/bin']
                    args=['docker','exec','moveEnetOFK']+prefix+['/workspace/moveEnetFlow/build/'+BINS[model],
                           '--data_file',data,'--net_period',str(net),'--output_period','0.005']
                    if model in ('movenet','moveenetofk'):
                        args+=['--flow_period','0.005','--checkpoint_path','/models/'+ds+'.pth']
                        if ds=='dhp19': args+=['--dhp19']
                    if model=='movenet': args+=['--moveenet_only']
                    if model=='eventpointpose': args+=['--camera','2','--model_path','/models/epp.pth','--response_timeout','0','--EventPointPose_script','/models/eventPointPose_yarp_server.py']
                    if model=='yolo': args+=['--yolo_model_path','/models/yolo.pt','--YoloPose_script','/models/YoloPose_yarp_server.py']
                    if model=='openpose': args+=['--op_model_path','/models/op/']
                    args+=['--device','cuda:0','--no_csv']
                    if model!='eventpointpose': args+=['--no_video']
                    tag='{}_{}_{}_r{:02d}'.format(ds,sample,model,rep)
                    log=session/'runs'/(tag+'.log')
                    text='[INFO] 100 packets loaded with 1000 total events. Timestamps range from 0 to {}\n'.format(10+i)
                    if model=='moveenetofk':
                        text+='[INFO] KF parameters: pu = {} muD = {} muV = {} dataset = {}\n'.format(*EXPECTED_KF[ds],ds)
                    if model in ('movenet','moveenetofk'):
                        text+='[INFO] Finished processing event file. Packets: 100\nMoveNet inference requests: 100\nMoveNet valid predictions: 100\n'
                    elif model=='eventpointpose':
                        text+='Events read: 1000\nPointNet requests: 100\nValid PointNet inferences: 99\nServer ERROR responses: 0\n'
                    else:
                        text+='End of video reached. Frames processed: 600\n{} inferences completed: 600\n{} valid inferences: 600\n'.format('OpenPose' if model=='openpose' else 'YOLO','OpenPose' if model=='openpose' else 'YOLO')
                    log.write_text(text)
                    T=10+i+rep*.1; cpu=T*20; gpu=T*10
                    energy.append({'dataset':ds,'sample_id':sample,'model':model,'order_id':'R'+str(rep),
                                   'label':tag,'status':'OK','exit_code':'0','net_period_s':str(net),
                                   'flow_period_s':'0.005' if model=='moveenetofk' else '',
                                   'wall_seconds':str(T),'cpu_package_energy_j':str(cpu),'gpu_energy_j':str(gpu),
                                   'measured_compute_energy_j':str(cpu+gpu),
                                   'gpu_uuid':'GPU-test','gpu_name':'SYNTHETIC','nvidia_driver':'TEST',
                                   'log_path':str(log),'command':shlex.join(args)})
                    manifest.append({'dataset':ds,'sample_id':sample,'model':model,'repetition':str(rep),
                                     'net_period':str(net),'output_period':'0.005','status':'OK'})
    write_csv(session/'energy_measurements.csv',energy)
    write_csv(session/'run_manifest.csv',manifest)
    (session/'metadata.txt').write_text('SYNTHETIC TEST DATA ONLY\n')
    return session, energy, manifest


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.session,self.energy,self.manifest=fixture(self.root)
    def tearDown(self): self.tmp.cleanup()
    def rewrite(self):
        write_csv(self.session/'energy_measurements.csv',self.energy)
        write_csv(self.session/'run_manifest.csv',self.manifest)
    def test_full420_unique_and84_replays_readonly(self):
        before=sha256_file(self.session/'energy_measurements.csv')
        jobs,items,audit=audit_session(self.session)
        self.assertEqual((len(jobs),len(items)),(84,420))
        self.assertEqual(before,sha256_file(self.session/'energy_measurements.csv'))
        for j in jobs:
            cmd=replay_command(j,'/new/pose.csv','/new/server.log')
            self.assertNotIn('--no_csv',cmd)
            self.assertIn('--output_csv_f' if j['model'] in ('movenet','moveenetofk') else '--output_csv',cmd)
            self.assertEqual(j['net_period_s'],EXPECTED_NET[j['dataset'],j['model']])
            self.assertEqual('--moveenet_only' in cmd,j['model']=='movenet')
            if j['model']=='moveenetofk':
                self.assertEqual(float(cmd[cmd.index('--pu')+1]),EXPECTED_KF[j['dataset']][0])
            if j['model']=='openpose': self.assertEqual(cmd[:3],['env','-u','VIRTUAL_ENV'])
            else: self.assertIn('VIRTUAL_ENV=/opt/venv',cmd)
    def test_failed_attempt_not_averaged(self):
        failed=copy.deepcopy(self.energy[0]); failed.update(status='FAILED',exit_code='143',wall_seconds='20000')
        self.energy.append(failed); self.rewrite()
        jobs,items,audit=audit_session(self.session)
        self.assertEqual(len(items),420); self.assertEqual(len(audit['ignored_failed_attempts']),1)
    def test_conflicting_ok_repetition_fails(self):
        r=copy.deepcopy(self.energy[0]); r['wall_seconds']='999'; self.energy.append(r); self.rewrite()
        with self.assertRaisesRegex(AuditError,'Due misure OK'): audit_session(self.session)
    def test_identical_csv_row_dedup(self):
        self.energy.append(copy.deepcopy(self.energy[0])); self.rewrite()
        self.assertEqual(audit_session(self.session)[2]['exact_duplicate_rows_collapsed'],1)
    def test_missing_key_fails(self):
        self.energy.pop(); self.rewrite()
        with self.assertRaisesRegex(AuditError,'Copertura energia incompleta'): audit_session(self.session)
    def test_mixed_operating_point_fails(self):
        self.energy[0]['command']=self.energy[0]['command'].replace('--net_period 0.02','--net_period 0.05')
        self.rewrite()
        with self.assertRaises(AuditError): audit_session(self.session)
    def test_kf_wrong_fails(self):
        r=next(r for r in self.energy if r['dataset']=='dhp19' and r['model']=='moveenetofk')
        p=Path(r['log_path']); p.write_text(p.read_text().replace('pu = 0.36','pu = 0.77'))
        with self.assertRaisesRegex(AuditError,'Tuning KF errato'): audit_session(self.session)
    def test_no_completion_fails(self):
        Path(self.energy[0]['log_path']).write_text('sidecar connected\n')
        with self.assertRaisesRegex(AuditError,'senza evidenza'): audit_session(self.session)
    def test_epp_server_errors_fail(self):
        r=next(r for r in self.energy if r['model']=='eventpointpose'); p=Path(r['log_path'])
        p.write_text(p.read_text().replace('Server ERROR responses: 0','Server ERROR responses: 4'))
        with self.assertRaises(AuditError): audit_session(self.session)
    def test_sum_inconsistent_fails(self):
        self.energy[0]['gpu_energy_j']='1'; self.rewrite()
        with self.assertRaisesRegex(AuditError,r'CPU\+GPU'): audit_session(self.session)
    def test_snapshot_preserves_energy_and_detects_change(self):
        jobs,items,audit=audit_session(self.session); out=self.root/'Final'; out.mkdir()
        snapshot_files(self.session,out,items,audit)
        before=sha256_file(self.session/'energy_measurements.csv')
        snapshot_files(self.session,out,items,audit)
        self.assertEqual(before,sha256_file(self.session/'energy_measurements.csv'))
        self.energy[0]['label']='CHANGED'; self.rewrite()
        with self.assertRaises(AuditError): snapshot_files(self.session,out,items,audit)
    def test_untrusted_shell_not_evaluated(self):
        self.energy[0]['command']='bash -lc "touch /tmp/DO_NOT_RUN"'; self.rewrite()
        with self.assertRaises(AuditError): audit_session(self.session)


class MetricTests(unittest.TestCase):
    def test_matches_notebook_math(self):
        import numpy as np
        from final_metrics import metric_arrays, interpolate_gt
        gt=np.zeros((3,13,2)); gt[:,5,0]=10; gt[1:,:,1]=5
        t=np.array([0.,1.,2.]); ptime=np.array([.005,.1,.4,1.2,1.6,2.005])
        p=interpolate_gt(t,gt,ptime); p[::2,:,0]+=1; p[1::2,:,0]+=3
        r=metric_arrays(ptime,p,t,gt)
        self.assertAlmostEqual(r['mpjpe'],2.0)
        self.assertAlmostEqual(r['pck'],50.0)
        self.assertEqual(r['n_points'],78)
        self.assertAlmostEqual(r['pck_0.4'],100.0)
    def test_nonfinite_reported_not_hidden(self):
        import numpy as np
        from final_metrics import metric_arrays
        gt=np.zeros((3,13,2)); gt[:,5,0]=10
        p=gt.copy(); p[1,2,0]=np.nan
        r=metric_arrays([0,1,2],p,np.array([0.,1.,2.]),gt)
        self.assertEqual(r['n_nonfinite_rows_excluded'],1); self.assertEqual(r['n_samples'],2)
    def test_time_identity_gt_paths(self):
        self.assertEqual(derive_gt('/data/dhp19_testing_set_S13toS17/S13_1_1/ch3dvs/data.log','dhp19'),
                         '/data/dhp19_testing_set_S13toS17/S13_1_1/ch3GT200Hzskeleton/data.log')
        self.assertEqual(derive_gt('/data/eh36m_testing_set_S9S11/rgb/cam2_S9_Discussion_1.mp4','h36m'),
                         '/data/eh36m_testing_set_S9S11/events/cam2_S9_Discussion_1/ch0GT200Hzskeleton/data.log')
    def test_python38_syntax(self):
        for p in HERE.glob('*.py'):
            ast.parse(p.read_text(),filename=str(p),feature_version=(3,8))
    def test_mount_resolution_longest_prefix(self):
        mounts=[{'Source':'/home/test/repo','Destination':'/workspace'},
                {'Source':'/home/test/data','Destination':'/data'}]
        self.assertEqual(container_path(Path('/home/test/data/x/y'),mounts),'/data/x/y')
        self.assertEqual(container_path(Path('/home/test/repo/a.py'),mounts),'/workspace/a.py')
        with self.assertRaises(AuditError): container_path(Path('/elsewhere'),mounts)


class AnalysisTests(unittest.TestCase):
    def test_repeat_aggregation_uses_12_units_and_original_wall(self):
        import pandas as pd
        import numpy as np
        from final_analysis import energy_sequence_table,dataset_summary
        with tempfile.TemporaryDirectory() as t:
            session,_,_=fixture(t); jobs,items,audit=audit_session(session)
            pre={'durations':{j['data_file']:{'input_span_s':j['input_span_s_from_energy_log'],
                   'duration_source':'SYNTHETIC'} for j in jobs}}
            out=Path(t)/'Final'; out.mkdir(); export_energy_rows(out,items,pre)
            seq=energy_sequence_table(pd.read_csv(out/'tables/energy_validated_runs.csv'))
            self.assertEqual(len(seq),84)
            self.assertTrue((seq['n_repetitions']==5).all())
            one=seq.iloc[0]
            self.assertAlmostEqual(one['amortized_ms_per_nominal_output_slot'],5*one['real_time_factor'])
            self.assertAlmostEqual(one['combined_mean_power_w'],30.)
            summary=dataset_summary(seq)
            self.assertEqual(len(summary),7); self.assertTrue((summary['n_sequences']==12).all())
    def test_point_weighted_accuracy_not_macro(self):
        import pandas as pd
        import numpy as np
        from final_analysis import dataset_summary
        rows=[]
        for i in range(12):
            n=13*(i+1)
            rows.append(dict(dataset='h36m',model='movenet',sample_id='H%02d'%(i+1),
                             energy_j=100.,sequence_wall_seconds=10.,input_span_s=20.,
                             net_period_s=.02,net_hz=50.,output_hz=200.,cpu_energy_j=60.,gpu_energy_j=40.,
                             n_points=n,sum_err_px=n*i,pck_0_2_correct_points=0,
                             pck=i,mpjpe=i,observed_output_hz=200.))
            rows[-1]['pck_0.2_correct_points']=n*(i/100)
        r=dataset_summary(pd.DataFrame(rows),True).iloc[0]
        self.assertNotEqual(r['mpjpe_weighted_px'],r['mpjpe_sequence_macro_px'])
        self.assertAlmostEqual(r['real_time_factor'],.5)


class WatchdogTests(unittest.TestCase):
    def test_timeout_terminates_child_group_and_saves_attempt(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); job=root/'job.json'
            payload=dict(job_id='timeout_test',attempt_dir_container=t,required_hashes={},
                         config_hash='test',identity_hash='test',timeout_s=.2,
                         replay_argv=[sys.executable,'-c','import time; time.sleep(60)'])
            job.write_text(json.dumps(payload))
            p=subprocess.run([sys.executable,str(HERE/'container_worker.py'),'run-job',str(job)],
                             capture_output=True,text=True,timeout=15)
            r=json.loads((root/'result.json').read_text())
            self.assertEqual(r['status'],'TIMEOUT'); self.assertNotEqual(p.returncode,0)
            pid=json.loads((root/'process.json').read_text())['pid']
            self.assertFalse(Path('/proc/'+str(pid)).exists())


class WorkerSuccessTests(unittest.TestCase):
    def test_successful_replay_csv_metrics_and_200hz_guard(self):
        from final_metrics import JOINT_NAMES
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); core=root/'hpecore'; utils=core/'datasets/utils'; utils.mkdir(parents=True)
            (core/'datasets/__init__.py').write_text('')
            (utils/'__init__.py').write_text('')
            (utils/'constants.py').write_text('class HPECoreSkeleton:\n    KEYPOINTS_MAP = '+repr({x:i for i,x in enumerate(JOINT_NAMES)})+'\n')
            (utils/'parsing.py').write_text('''import numpy as np
from .constants import HPECoreSkeleton
def import_yarp_skeleton_data(path, multi_channel=False):
    out={'ts': np.array([0.0, 0.1, 0.2])}
    for j,k in enumerate(HPECoreSkeleton.KEYPOINTS_MAP):
        out[k]=np.tile([10+j, 20+j], (3,1))
    return out
''')
            gt=root/'gt.log'; gt.write_text('SYNTHETIC TEST ONLY\n')
            script=root/'make_predictions.py'
            script.write_text('''import csv,sys
p=sys.argv[1]
dt=float(sys.argv[2])
with open(p,'w') as f:
    w=csv.writer(f)
    w.writerow(['timestamp','latency']+[k for j in range(13) for k in ('joint%d_x'%j,'joint%d_y'%j)])
    for i in range(11): w.writerow([i*dt,0.01]+[k for j in range(13) for k in (10+j,20+j)])
print('End of video reached after 11 frames.')
print('YOLO inferences completed: 11')
print('YOLO valid inferences: 11')
''')
            for dt, expected in [(0.005,'OK'),(0.02,'FAILED')]:
                attempt=root/('attempt_'+str(dt)); attempt.mkdir()
                pred=attempt/'predictions.csv'; job=attempt/'job.json'
                payload=dict(job_id='synthetic_yolo',dataset='h36m',sample_id='H01',model='yolo',
                             attempt_dir_container=str(attempt),required_hashes={},
                             config_hash='test',identity_hash='test',timeout_s=5,
                             replay_argv=[sys.executable,str(script),str(pred),str(dt)],
                             prediction_container=str(pred),gt_file=str(gt),hpecore_root=str(core),
                             output_period_s=.005,input_span_s=.2,net_period_s=.02,flow_period_s=None,kf=None)
                job.write_text(json.dumps(payload))
                proc=subprocess.run([sys.executable,str(HERE/'container_worker.py'),'run-job',str(job)],
                                    capture_output=True,text=True,timeout=15)
                r=json.loads((attempt/'result.json').read_text())
                self.assertEqual(r['status'],expected,proc.stderr+'\n'+repr(r))
                if expected=='OK':
                    self.assertEqual(r['metrics']['mpjpe'],0)
                    self.assertEqual(r['metrics']['pck'],100)
                    self.assertEqual(r['metrics']['n_samples'],11)

if __name__=='__main__': unittest.main()
