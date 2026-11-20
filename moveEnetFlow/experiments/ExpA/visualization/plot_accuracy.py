#!/usr/bin/env python3
"""Plot the scored Experiment A summary using the existing common paper style."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLOW = HERE.parent.parent.parent
sys.path.insert(0, str(FLOW / 'analysis'))
from paper_figures.paper_figures import load_style, render_accuracy_rate, export_figure  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True, type=Path)
    options = p.parse_args()
    import pandas as pd
    directory = options.run_dir.resolve()
    data = pd.read_csv(directory / 'metrics' / 'summary.csv')
    meta = json.loads((directory / 'metrics' / 'protocol.json').read_text())
    if not meta['complete_dataset_evaluation']:
        raise ValueError('Refusing to produce final figures from partial/incomplete scoring')
    style = load_style()
    fig = render_accuracy_rate(data, style, datasets=[meta['dataset']])
    output_dir = directory / 'figures'
    output_dir.mkdir(exist_ok=True)
    # Export filenames are run-scoped, so existing experiments are unaffected.
    files = export_figure(fig, output_dir / ('ExpA_' + meta['dataset'] + '_accuracy'), style,
                          metadata=meta)
    print('Figures:', files)


if __name__ == '__main__':
    main()
