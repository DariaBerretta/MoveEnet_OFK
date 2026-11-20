"""Dataset pairing and method-command invariants for Experiment A."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

EXPA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPA / 'scripts'))
from expa_common import matched_inventory, inventory, read_config
from run_experiment import command_for


class CohortTests(unittest.TestCase):
    def test_eh36m_rgb_event_common_population(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            events = root / 'events'
            rgb = root / 'rgb'
            seq = 'cam2_S9_Walking'
            (events / seq / 'ch0dvs').mkdir(parents=True)
            (events / seq / 'ch0GT200Hzskeleton').mkdir(parents=True)
            (events / seq / 'ch0dvs' / 'data.log').write_text('events')
            (events / seq / 'ch0GT200Hzskeleton' / 'data.log').write_text('ground truth')
            rgb.mkdir()
            (rgb / (seq + '.mp4')).write_bytes(b'fake')
            config = read_config('eh36m')
            config.update(event_root=str(events), rgb_root=str(rgb), gt_root=str(events))
            matched, excluded = matched_inventory(config)
            self.assertFalse(excluded)
            for method in config['methods']:
                self.assertEqual([x['sample_id'] for x in matched[method]], [seq])

    def test_missing_rgb_excludes_other_models_for_fair_comparison(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            events = root / 'events'
            rgb = root / 'rgb'
            rgb.mkdir()
            for seq in ('cam2_S9_Walking', 'cam2_S9_Sitting'):
                (events / seq / 'ch0dvs').mkdir(parents=True)
                (events / seq / 'ch0GT200Hzskeleton').mkdir(parents=True)
                (events / seq / 'ch0dvs' / 'data.log').write_text('events')
                (events / seq / 'ch0GT200Hzskeleton' / 'data.log').write_text('gt')
            (rgb / 'cam2_S9_Walking.mp4').write_bytes(b'fake')
            config = read_config('eh36m')
            config.update(event_root=str(events), rgb_root=str(rgb), gt_root=str(events))
            matched, excluded = matched_inventory(config)
            for method in config['methods']:
                self.assertEqual([x['sample_id'] for x in matched[method]], ['cam2_S9_Walking'])
            self.assertEqual({x['reason'] for x in excluded}, {'not_available_to_all_comparison_methods'})
            self.assertEqual({x['sample_id'] for x in excluded}, {'cam2_S9_Sitting'})

    def test_epp_moveiit_geometry_and_yolo_flags(self):
        root = Path('/tmp')
        config = read_config('moveiit')
        cmd = command_for(config, 'eventpointpose', 200, root / 'luna' / 'box' / 'atis-s' / 'data.log',
                          root / 'out.csv', root / 'server.log', root / 'gpu.log', root, 'cuda:0')
        self.assertIn('--moveiit', cmd)
        self.assertEqual(cmd[cmd.index('--net_period') + 1], '0.005')
        self.assertEqual(cmd[cmd.index('--moveiit_geometry') + 1], 'stretch')
        rgb = read_config('eh36m')
        yolo = command_for(rgb, 'yolo', 50, root / 'sample.mp4', root / 'out.csv',
                           root / 'server.log', root / 'gpu.log', root, 'cuda:0')
        self.assertIn('--yolo_model_path', yolo)
        self.assertIn('--YoloPose_script', yolo)
        self.assertNotIn('--flow_period', yolo)


if __name__ == '__main__':
    unittest.main()
