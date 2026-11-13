#!/usr/bin/env python3
"""End-to-end smoke test with random weights (NO accuracy test, NO YARP)."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'utils'))
import eventPointPose_yarp_server as epp


def make_args(model_path, moveiit, geometry='stretch'):
    return SimpleNamespace(device='cpu',seed=1,w=346,h=260,model=str(model_path),
        rasepc_channels=4,fifo_size=7500,num_points=2048,
        input_preprocessing='already_filtered',background_domain='global',
        hotpixel_path='',background_dt_us=70000.0,
        dump_debug_dir='',dump_first_n=0,moveiit=moveiit,moveiit_geometry=geometry)


def main():
    torch.manual_seed(19)
    model=epp.PosePointNet(SimpleNamespace(sensor_sizeH=260,sensor_sizeW=346,num_joints=13))
    with tempfile.TemporaryDirectory() as d:
        ckpt=Path(d)/'random_weights.pth'
        torch.save(model.state_dict(),str(ckpt))
        rng=np.random.RandomState(23)
        events=np.column_stack((rng.randint(0,1280,7600),rng.randint(0,720,7600),
            np.linspace(0,.060,7600),rng.randint(0,2,7600))).astype('float64')
        for geometry in ('stretch','letterbox'):
            engine=epp.EventPointPoseEngine(make_args(ckpt,True,geometry))
            status,pose,confidence,stats=engine.process_request(0,0,events)
            assert status=='OK',status
            assert pose.shape==(13,2) and confidence.shape==(13,)
            assert np.isfinite(pose).all() and stats[0]==7600 and stats[1]==7500
            print('PASS: ATIS',geometry,'-> 7500 FIFO -> RasEPC -> PointNet -> 13 native joints')
        # DHP19 path must stay functional and use its original preprocessor.
        legacy=np.column_stack((rng.randint(0,346,7600),rng.randint(0,260,7600),
            np.linspace(0,.060,7600),rng.randint(0,2,7600))).astype('float64')
        engine=epp.EventPointPoseEngine(make_args(ckpt,False))
        status,pose,conf,stats=engine.process_request(0,2,legacy)
        assert status=='OK' and stats[0]==7600 and stats[1]==7500
        print('PASS: DHP19 default behavior still runs')
    print('NOTE: tests use random PointNet weights, NOT the original trained checkpoint.')

if __name__=='__main__':
    main()
