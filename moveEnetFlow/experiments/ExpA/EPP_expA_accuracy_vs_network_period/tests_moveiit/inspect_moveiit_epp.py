#!/usr/bin/env python3
"""Optional visual QA of a saved EventPointPose Move-IIT inference.

  python3 inspect_moveiit_epp.py --dump_dir DIR --geometry stretch --output IMAGE.png

Requires numpy and matplotlib. Outputs the ATIS sensor event scatter plus the
13-joint model prediction (no ground-truth or timestamp assumptions).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "utils"))
from epp_moveiit_adapter import Geometry

# Order: head, right shoulder, left shoulder, right elbow, left elbow,
# left hip, right hip, right hand, left hand, right knee, left knee,
# right foot, left foot.
EDGES = [(0,1),(0,2),(1,3),(3,7),(2,4),(4,8),(1,6),(2,5),
         (5,6),(6,9),(9,11),(5,10),(10,12)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump_dir',type=Path,required=True)
    ap.add_argument('--geometry',choices=('stretch','letterbox'),default='stretch')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--request_id',type=int,default=None,help='Default latest available debug dump')
    args=ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dumps=sorted(args.dump_dir.glob('request_*_fifo_epp.npy'))
    if not dumps:
        raise SystemExit('No dumps found; rerun experiment with --debug')
    if args.request_id is None:
        stem=dumps[-1].name.split('_fifo_epp.npy')[0]
    else:
        stem='request_{:08d}'.format(args.request_id)
    events=np.load(args.dump_dir/(stem+'_fifo_epp.npy'),allow_pickle=False)
    poses=np.load(args.dump_dir/(stem+'_hpecore_prediction.npy'),allow_pickle=False)
    if events.ndim!=2 or events.shape[1]!=4 or poses.shape!=(13,2):
        raise SystemExit('Unexpected debug dump shape')
    geo=Geometry(mode=args.geometry)
    virtual=np.column_stack((events[:,0]-1,geo.target_height-1-events[:,1]))
    native=geo.to_native_pose(virtual)
    fig,ax=plt.subplots(figsize=(12.8,7.2))
    ax.scatter(native[:,0],native[:,1],s=.9,alpha=.25,linewidths=0,rasterized=True)
    for first,last in EDGES:
        ax.plot(poses[[first,last],0],poses[[first,last],1],linewidth=1.5)
    ax.scatter(poses[:,0],poses[:,1],s=24,zorder=5)
    for i,(x,y) in enumerate(poses):
        ax.text(x+6,y+6,str(i),fontsize=7)
    ax.set_xlim(0,1279)
    ax.set_ylim(719,0)
    ax.set_aspect('equal')
    ax.set_title('EventPointPose move-iit-hpe: {} | {} | {} events'.format(stem,args.geometry,len(events)))
    ax.set_xlabel('ATIS x [px]')
    ax.set_ylabel('ATIS y [px]')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180)
    print('Saved',args.output)

if __name__=='__main__':
    main()
