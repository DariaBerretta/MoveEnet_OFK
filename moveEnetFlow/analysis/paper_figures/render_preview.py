"""Render ONLY ONE cached energy-vs-accuracy preview (legacy helper). Use generate_all_figures.py for ALL figures."""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd

from paper_figures import load_style, render_energy_accuracy, export_figure


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--style', type=Path, default=root / 'style.json')
    parser.add_argument('--output', type=Path, default=root / 'output/energy_accuracy_preview')
    args = parser.parse_args()
    source = root / 'data/energy_accuracy_preview.csv'
    data = pd.read_csv(source)
    style = load_style(args.style)
    manifest = json.loads((root / 'data/source_manifest.json').read_text())
    figure = render_energy_accuracy(
        data, style, datasets=['eh36m', 'dhp19', 'moveiit'],
        title='Energy–accuracy · cached notebook values (preview)',
    )
    metadata = {
        'status': 'style_preview_from_cached_outputs',
        'input_csv': str(source),
        'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_manifest': manifest,
        'intervals': 'Not available in the cached tables; no error bars drawn.',
        'data': data.astype(object).where(pd.notna(data), None).to_dict(orient='records'),
    }
    paths = export_figure(figure, args.output, style, metadata=metadata)
    print('Preview from stored notebook values; not a recomputed final analysis.')
    print('Original confidence bounds are needed for final error bars.')
    print(paths)


if __name__ == '__main__':
    main()
