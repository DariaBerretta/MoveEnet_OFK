"""Deterministic, dataset-free smoke tests of the actual ExpA batch runner.

These tests create two synthetic DHP19 inputs and a mock executable, then run
one real subprocess end-to-end. They exercise the CLI, output tree, manifest,
resume conditions, and CSV validator without YARP, CUDA, or trained weights.
They do NOT substitute for a live inference smoke test in the user's Docker.
"""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EXPA = Path(__file__).resolve().parents[1]
SCRIPT = EXPA / 'scripts' / 'run_experiment.py'
VALIDATOR = EXPA / 'scripts' / 'validate_results.py'

MOCK_EXECUTABLE = '''#!/usr/bin/env python3
import pathlib,sys
args=sys.argv[1:]
def value(k):return args[args.index(k)+1]
output=pathlib.Path(value('--output_csv_f') if '--output_csv_f' in args else value('--output_csv'))
output.parent.mkdir(parents=True,exist_ok=True)
header='timestamp,latency,'+','.join('joint{}_{}'.format(j,a) for j in range(13) for a in 'xy')
with output.open('w') as f:
 f.write(header+'\\n')
 for i in range(4):
  f.write('{:.3f},0.0,{}\\n'.format(i*.005,','.join('1' for _ in range(26))))
print('SYNTHETIC MOCK ONLY')
'''


class SyntheticPipelineSmokeTests(unittest.TestCase):
    def test_mocked_dhp19_pipeline(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            data = root / 'data'
            for seq in ('S13_1_1', 'S13_1_2'):
                for channel in ('ch2dvs', 'ch2GT200Hzskeleton'):
                    (data / seq / channel).mkdir(parents=True)
                    (data / seq / channel / 'data.log').write_text('sample\\n', encoding='utf-8')
            config_dir = root / 'config'
            config_dir.mkdir()
            config = json.loads((EXPA / 'config' / 'dhp19.yaml').read_text())
            config['event_root'] = str(data)
            config['gt_root'] = str(data)
            weights = root / 'net.pth'
            weights.write_bytes(b'no real model is loaded in this test')
            config['movenet_checkpoint'] = str(weights)
            config['epp_checkpoint'] = str(weights)
            config['epp_script'] = str(root / 'mock_epp_server.py')
            (config_dir / 'dhp19.yaml').write_text(json.dumps(config), encoding='utf-8')
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            binary = bin_dir / 'moveEnetOFK_offline'
            binary.write_text(MOCK_EXECUTABLE, encoding='utf-8')
            binary.chmod(0o755)
            results = root / 'results'
            cmd = [sys.executable, str(SCRIPT), '--dataset', 'dhp19',
                   '--method', 'moveenetofk', '--network-hz', '5',
                   '--limit', '1', '--results-root', str(results),
                   '--config-dir', str(config_dir), '--binary-dir', str(bin_dir),
                   '--run-id', 'smoke_synthetic_dhp19']
            run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 universal_newlines=True)
            self.assertEqual(run.returncode, 0, run.stderr + '\n' + run.stdout)
            run_dir = results / 'ExpA' / 'smoke_synthetic_dhp19'
            with (run_dir / 'manifest.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['status'], 'complete')
            self.assertEqual(int(rows[0]['prediction_rows']), 4)
            self.assertIn('--of_gain', json.loads(rows[0]['command_json']))
            self.assertEqual(float(rows[0]['flow_hz']), 200.0)
            self.assertTrue(Path(rows[0]['prediction_csv']).is_file())
            check = subprocess.run([sys.executable, str(VALIDATOR), '--run-dir', str(run_dir)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True)
            self.assertEqual(check.returncode, 0, check.stderr + '\n' + check.stdout)
            resume = subprocess.run(cmd + ['--resume'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)
            self.assertEqual(resume.returncode, 0, resume.stderr + '\n' + resume.stdout)
            self.assertIn('SKIP verified complete', resume.stdout)


if __name__ == '__main__':
    unittest.main()
