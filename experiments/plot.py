"""Plot a measured NOCL field and its reference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads((args.run / 'metrics.json').read_text())
    if metrics['status'] != 'completed' or metrics.get('smoke') or metrics['method'] != 'nocl':
        parser.error('Use a completed full NOCL run')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    problem = metrics['problem']
    if problem not in ('convection', 'rd'):
        parser.error('Only convection and rd are included')
    filename = args.run / 'prediction.npz'
    if not filename.exists():
        parser.error('Run experiments.evaluate first to generate prediction.npz from the checkpoint')
    with np.load(filename, allow_pickle=False) as arrays:
        prediction, reference = arrays['prediction'], arrays['reference']
    if problem == 'rd':
        prediction, reference = prediction.T, reference.T
    extent = (0, 2*np.pi if problem == 'convection' else 1, 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    titles = [f'NOCL\nRelL2={metrics["final_relative_l2"]:.5g}', 'Reference']
    for ax, field, title in zip(axes, (prediction, reference), titles):
        im = ax.imshow(field, origin='lower', extent=extent, aspect='auto',
                       vmin=float(reference.min()), vmax=float(reference.max()), cmap='viridis')
        ax.set(title=title, xlabel='x', ylabel='t')
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
