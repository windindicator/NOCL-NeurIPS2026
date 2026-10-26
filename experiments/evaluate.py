"""Recompute NOCL metrics from portable PINN checkpoints."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--all-levels', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.gpu < 0 else str(args.gpu)
    if args.gpu < 0:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

    import jax
    import jax.numpy as jnp
    import numpy as np
    from nocl_minimal.io import load_params, write_json

    metrics = json.loads((args.run / 'metrics.json').read_text())
    if metrics['status'] != 'completed' or metrics['method'] != 'nocl':
        parser.error('Use a completed NOCL run')
    config = metrics['config']
    if config['problem'] not in ('convection', 'rd'):
        parser.error('Only convection and rd are included')
    if config['problem'] == 'rd':
        from nocl_minimal.rd_rebuttal import PROTOCOL_ID, evaluate_checkpoint
        if config['rd_protocol'] != PROTOCOL_ID:
            parser.error('This evaluator requires the RD rebuttal protocol')
    else:
        from nocl_minimal.models import mlp
        from nocl_minimal.training import coordinate_grid, reference_values
        model = mlp(**config['pinn'])
        predict = jax.jit(model.apply)

    rows = metrics['results'] if args.all_levels else metrics['results'][-1:]
    checks = []
    for row in rows:
        params = load_params(args.run / row['checkpoint'])
        coefficients = jnp.asarray(row['coefficients'], dtype=jnp.float32)
        if config['problem'] == 'rd':
            error, prediction, reference = evaluate_checkpoint(params, row['evaluation_size'])
        else:
            grid = coordinate_grid(config['problem'], row['evaluation_size'])
            points = grid.reshape(-1, grid.shape[-1])
            predictions = [np.asarray(predict(params, points[i:i + 65536]))[:, 0]
                           for i in range(0, len(points), 65536)]
            prediction = jnp.asarray(np.concatenate(predictions)).reshape(grid.shape[:-1])
            reference = reference_values(config['problem'], grid, coefficients)
            error = float(jnp.linalg.norm(prediction - reference) / (jnp.linalg.norm(reference) + 1e-12))
        original = row['relative_l2']
        checks.append({
            'checkpoint': row['checkpoint'], 'coefficients': row['coefficients'],
            'reference_coefficients': row.get('reference_coefficients', row['coefficients']),
            'relative_l2': error, 'training_record_relative_l2': original,
            'absolute_difference': abs(error - original),
            'checkpoint_consistent': bool(np.isclose(error, original, rtol=2e-5, atol=2e-7)),
            'devices': [str(d) for d in jax.devices()],
        })
        if row['checkpoint'] == metrics['final_checkpoint']:
            np.savez_compressed(args.run / 'prediction.npz', prediction=np.asarray(prediction),
                                reference=np.asarray(reference), coefficients=np.asarray(coefficients))
    result = dict(checks[-1])
    if args.all_levels:
        result['level_checks'] = checks
        result['all_levels_consistent'] = all(check['checkpoint_consistent'] for check in checks)
    write_json(args.run / 'reevaluation.json', result)
    print(json.dumps(result, indent=2))
    if not all(check['checkpoint_consistent'] for check in checks):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
