"""No-GPU safety tests for the EventPointPose Move-IIT energy runner.

Run from ExpEnergy: python3 -m unittest discover -s tests -p 'test_eventpointpose_energy.py'
"""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_moveiit_eventpointpose_energy_rtf import (
    parse_epp_log, completed_keys, build_command, MODELS)


class TestEppEnergy(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def log(self, loaded=100, processed=100, valid=5, errors=0, signal=False):
        path = self.folder / 'epp.log'
        path.write_text(
            '[INFO] 10 packets loaded with {} total events.\n'
            '[INFO] Events read: {}\n'
            '[INFO] PointNet requests: 6\n'
            '[INFO] Valid PointNet inferences: {}\n'
            '[INFO] Server ERROR responses: {}\n{}'.format(
                loaded, processed, valid, errors,
                '[INFO] Termination signal received; EventPointPose replay stopped.\n' if signal else ''))
        return path

    def test_complete_replay(self):
        ok, reason, stats = parse_epp_log(self.log())
        self.assertTrue(ok, reason)
        self.assertEqual(stats['processed_events'], 100)

    def test_missing_events_rejected(self):
        ok, reason, _ = parse_epp_log(self.log(processed=50))
        self.assertFalse(ok)
        self.assertIn('differ', reason)

    def test_signal_or_server_error_rejected(self):
        self.assertFalse(parse_epp_log(self.log(signal=True))[0])
        self.assertFalse(parse_epp_log(self.log(errors=1))[0])

    def test_incomplete_meter_row_not_skipped(self):
        log = self.log(processed=20)
        meter = self.folder / 'energy.csv'
        with meter.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_id','model','order_id','status','exit_code','log_path'])
            writer.writeheader()
            writer.writerow(dict(sample_id='M01',model='eventpointpose_5hz',order_id='R1',
                                 status='OK',exit_code='0',log_path=str(log)))
        self.assertEqual(completed_keys(meter), {})
        self.log(processed=100)
        self.assertEqual(len(completed_keys(meter)), 1)

    def test_command_preserves_accuracy_configuration_without_csv(self):
        args = SimpleNamespace(binary='/workspace/moveEnetFlow/build/eventPointPose_offline',
             model_path='/model.pth',server_script='/script.py',device='cuda:0',
             response_timeout=60,startup_timeout=90,venv='/opt/venv',container='moveEnetOFK')
        sample={'event_file':'/data/move-iit-it_testing/luna/box_s1/atis-s/data.log'}
        cmd=build_command(args,sample,MODELS[0],Path('/tmp/pointnet.log'))
        joined=' '.join(cmd)
        for key in ['--no_csv','--moveiit_geometry stretch','--fifo_size 7500',
                    '--num_points 2048','--output_period 0.005','--net_period 0.2',
                    '--input_preprocessing already_filtered','--seed 1',
                    '--response_timeout 60']:
            self.assertIn(key,joined)
        self.assertNotIn('--no_video', joined)  # unsupported EPP flag


if __name__ == '__main__':
    unittest.main()
