import argparse
from graph_enet.data.scarfDataset_splineConv import scarfDataset_splineConv
from graph_enet.hpe_gnn.scripts.config import cfg_SCARF as cfg

def main():
    parser = argparse.ArgumentParser(description="Preprocess SCARF dataset into processed .pt files.")
    parser.add_argument('--root', type=str, default=cfg["data_path"], help="Root directory containing raw/ and processed/ folders.")
    parser.add_argument('--rf_size', type=int, default=14, help="Receptive field size (default: 14).")
    parser.add_argument('--alpha', type=float, default=1.0, help="Alpha parameter (default: 1.0).")
    parser.add_argument('--C', type=float, default=0.3, help="C parameter (default: 0.3).")
    parser.add_argument('--res_width', type=int, default=640, help="Resolution width (default: 640).")
    parser.add_argument('--res_height', type=int, default=480, help="Resolution height (default: 480).")

    args = parser.parse_args()

    res = (args.res_width, args.res_height)

    print(f"Starting preprocessing for root: {args.root}")
    print(f"Parameters: rf_size={args.rf_size}, alpha={args.alpha}, C={args.C}, res={res}")

    # Instantiate the dataset, which will trigger process() if processed files don't exist
    dataset = scarfDataset_splineConv(
        root=args.root,
        rf_size=args.rf_size,
        alpha=args.alpha,
        C=args.C,
        res=res
    )

    print(f"Preprocessing complete. Dataset length: {len(dataset)}")

if __name__ == "__main__":
    main()