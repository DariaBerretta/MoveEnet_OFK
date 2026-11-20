"""Small dependency-free tests for the ExpA configuration and trial planner."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from expa_common import inventory, read_config, safe_stem
from run_experiment import command_for


class ExperimentATests(unittest.TestCase):
    def test_fixed_configuration(self):
        self.assertEqual(read_config('dhp19')['kf']['roi'], 10)
        self.assertEqual(read_config('moveiit')['kf']['roi'], 35)
        self.assertEqual(read_config('eh36m')['kf']['roi'], 20)
        for dataset in ('eh36m', 'dhp19', 'moveiit'):
            cfg = read_config(dataset)
            self.assertEqual(cfg['output_hz'], 200)
            self.assertEqual(cfg['flow_hz'], 200)
            self.assertEqual(cfg['kf']['of_gain'], 1.0)
            for method, rates in cfg['network_hz'].items():
                self.assertEqual(rates[-1], 50 if method in ('openpose', 'yolo') else 200)

    def test_dhp19_pairing_and_roi(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / 'S13_1_1' / 'ch2dvs').mkdir(parents=True)
            (root / 'S13_1_1' / 'ch2GT200Hzskeleton').mkdir(parents=True)
            sample = root / 'S13_1_1' / 'ch2dvs' / 'data.log'
            sample.write_text('sample', encoding='utf-8')
            gt = root / 'S13_1_1' / 'ch2GT200Hzskeleton' / 'data.log'
            gt.write_text('gt', encoding='utf-8')
            config = read_config('dhp19')
            config['event_root'] = str(root)
            paired, rejected = inventory(config, 'moveenetofk')
            self.assertEqual(len(paired), 1)
            self.assertFalse(rejected)
            self.assertEqual(Path(paired[0]['gt_path']), gt)
            cmd = command_for(config, 'moveenetofk', 5, sample,
                              root / 'out.csv', root / 'server.log', root / 'gpu.csv',
                              root, 'cuda:0')
            self.assertEqual(cmd[cmd.index('--roi') + 1], '10')
            self.assertEqual(cmd[cmd.index('--of_gain') + 1], '1.0')
            self.assertEqual(cmd[cmd.index('--flow_period') + 1], '0.005')
            self.assertEqual(cmd[cmd.index('--output_period') + 1], '0.005')
            self.assertIn('--dhp19', cmd)
            self.assertNotIn('--moveenet_only', cmd)

    def test_all_eventpointpose_rates_and_rgb_limit(self):
        for dataset in ('dhp19', 'moveiit'):
            c = read_config(dataset)
            self.assertEqual(c['network_hz']['eventpointpose'], [2, 5, 10, 20, 50, 100, 200])
        c = read_config('eh36m')
        self.assertEqual(c['network_hz']['yolo'], [2, 5, 10, 20, 50])

    def test_unique_safe_names(self):
        a = safe_stem('S13_1_1/ch2dvs')
        b = safe_stem('S13_1_1__ch2dvs')
        self.assertNotEqual(a, b)
        self.assertTrue(a.endswith(safe_stem('S13_1_1/ch2dvs')[-10:]))


if __name__ == '__main__':
    unittest.main()
