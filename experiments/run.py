"""Train NOCL on Convection or rebuttal Reaction--Diffusion."""
import argparse
import copy
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--problem', choices=('convection', 'rd'))
    selection.add_argument('--config', type=Path)
    parser.add_argument('--gpu', type=int, default=0, help='Physical GPU index; -1 selects CPU')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='Small functionality check, not a reported result')
    parser.add_argument('--save-operator', action='store_true', help='Also save the large FNO weights')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.gpu < 0 else str(args.gpu)
    if args.gpu < 0:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    filename = args.config or Path(__file__).parent / 'configs' / f'{args.problem}.json'
    config = json.loads(filename.read_text())
    if config['problem'] not in ('convection', 'rd'):
        parser.error('Only convection and rd are included')
    if args.seed is not None:
        config['seed'] = args.seed
    if args.smoke:
        config = copy.deepcopy(config)
        config['smoke'] = True
        if config['problem'] != 'rd':
            config['grid_size'] = 8
            config['evaluation_size'] = 16
            config['pinn'] = {'n_layer': 2, 'width': 8, 'output_size': 1}
            config['operator']['architecture'] = {'modes': 2, 'channels': 8, 'depth': 2}
            config['physics']['steps'] = 2
            config['operator']['steps'] = config['warmup']['steps'] = 2
            initial = config['curriculum']['initial']
            if isinstance(initial, list):
                config['curriculum']['initial'] = [initial[0], initial[-1]]
            else:
                initial['count'] = 2
            for stage in config['curriculum']['stages']:
                stage['count'] = 2
    if config['problem'] == 'rd':
        from nocl_minimal.rd_rebuttal import Experiment
    else:
        from nocl_minimal.training import Experiment
    experiment = Experiment(config, args.output, save_operator=args.save_operator)
    try:
        result = experiment.run()
    except BaseException as error:
        experiment.document['status'] = 'failed'
        experiment.document['error'] = f'{type(error).__name__}: {error}'
        experiment.flush()
        raise
    print(f'Completed: {args.output}/metrics.json; RelL2={result["final_relative_l2"]:.9g}')


if __name__ == '__main__':
    main()
