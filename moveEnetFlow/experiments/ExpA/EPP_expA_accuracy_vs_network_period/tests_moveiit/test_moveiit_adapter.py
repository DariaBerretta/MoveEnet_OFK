#!/usr/bin/env python3
"""Dataset geometry + original algorithm signature regressions (numpy only)."""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'utils'))
from epp_moveiit_adapter import Geometry

class AdapterTest(unittest.TestCase):
    def test_stretch_bounds_and_inverse(self):
        g=Geometry(mode='stretch')
        points=np.array([[0,0],[1279,719],[640,360]],dtype=float)
        converted=g.to_virtual_log(points)
        np.testing.assert_array_equal(converted[[0,1]],np.array([[0,0],[345,259]]))
        native=g.to_native_pose(converted)
        sx,sy,_,_=g.coefficients
        self.assertLessEqual(np.max(np.abs(native[:,0]-points[:,0])),.5/sx+1e-9)
        self.assertLessEqual(np.max(np.abs(native[:,1]-points[:,1])),.5/sy+1e-9)
    def test_letterbox_aspect_and_inverse(self):
        g=Geometry(mode='letterbox')
        samples=np.column_stack((np.linspace(0,1279,100), np.linspace(0,719,100)))
        virtual=g.to_virtual_log(samples)
        self.assertTrue(np.all((virtual >= 0)&(virtual<=np.array([345,259]))))
        recovered=g.to_native_pose(virtual)
        s,_,_,_=g.coefficients
        self.assertLessEqual(np.max(np.abs(recovered-samples)),.5/s+1e-9)
    def test_coordinates_polarity_timestamps(self):
        g=Geometry()
        events=np.array([[0,0,12,0],[1279,719,13,1],[150,200,15,0]],dtype=float)
        transformed,invalid=g.preprocess(events)
        self.assertEqual(invalid,0)
        virtual=g.to_virtual_log(events[:,:2])
        np.testing.assert_array_equal(transformed[:,0],virtual[:,0]+1)
        np.testing.assert_array_equal(transformed[:,1],g.target_height-1-virtual[:,1])
        np.testing.assert_array_equal(transformed[:,2:],events[:,2:])
    def test_out_of_bounds_and_noninteger_rejected(self):
        g=Geometry()
        events=np.array([[5,6,0,1],[-1,12,1,1],[1280,0,2,0],[15.5,12,3,1],[1,1,4,3]],dtype=float)
        out,n=g.preprocess(events)
        self.assertEqual(n,4)
        self.assertEqual(len(out),1)
        self.assertEqual(out[0,2],0)
    def test_original_core_source_not_changed(self):
        """Check exact original core source, independent of Python AST version.

        The prior test hashed ast.dump(), which differs across Python 3.8 and 3.13.
        This checks the original source for the 7 unchanged core components.
        """
        expected = {
            'EventFIFO': '8cc9f757569dbfb5c7ebf59728f00cc027fbf03dfa3e18825601e27f638d5612',
            'OnlineEventPreprocessor': '94f88cbc4ec85912a6fe495b71d64319361486363e4f18d6cd6b3e59de8df309',
            'PosePointNet': '75cc9ef1e381b78c184c38b38e5f62d7dd3c2888c5fc155ae2712fc78a4a3b29',
            'RasEventCloud': '56dae3fa3e70e42dfb9218300da37c7ce46cabb8272dc9ac46348d83bc928d33',
            'configure_determinism': 'b65775188a40eed834526672e7e380806407a944fcefe9b48d5052d3bee5fc62',
            'load_checkpoint_strict': '13770dba4ccbe6334d257b7b09e34c63fe4a773d4b716e1ed046ffdf6b68a53b',
            'random_sample_point_exact': '5d527644082f90b5652a3d0b09621d1f08ab0f20e3955d301f38ed05f2219473',
        }
        path = ROOT / 'utils' / 'eventPointPose_yarp_server.py'
        source = path.read_text()
        tree = ast.parse(source)
        found = {}
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in expected:
                segment = ast.get_source_segment(source, node)
                self.assertIsNotNone(segment, 'Unable to extract source for ' + node.name)
                found[node.name] = hashlib.sha256(segment.strip().encode()).hexdigest()
        self.assertEqual(found, expected)

if __name__=='__main__':
    unittest.main(verbosity=2)
