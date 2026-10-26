"""NOCL training for the Convection example."""
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import masking
from .io import environment, save_params, write_json
from .models import mlp, operator
from .transfer_2d import make_steps as make_2d_transfer_steps
from .problems import convection


def coordinate_grid(problem, size):
    if problem != 'convection':
        raise ValueError(problem)
    axes = [jnp.linspace(0, 2 * jnp.pi, size), jnp.linspace(0, 1, size)]
    return jnp.stack(jnp.meshgrid(*axes, indexing='xy'), axis=-1)


def reference_values(problem, grid, coefficients):
    if problem != 'convection':
        raise ValueError(problem)
    return jnp.sin(coefficients[0] * (grid[..., 0] - 30.0 * grid[..., 1]))


def schedule_values(spec, exclude_start=False):
    if isinstance(spec, list):
        return jnp.asarray(spec, dtype=jnp.float32)
    values = jnp.linspace(spec['start'], spec['end'],
                          spec['count'] + int(exclude_start))
    return values[1:] if exclude_start else values


def curriculum(config):
    spec = config['curriculum']
    initial = schedule_values(spec['initial'])
    stages = [schedule_values(stage, exclude_start=True) for stage in spec['stages']]
    return initial, stages


def coefficients_for(config, value):
    return jnp.asarray([value], dtype=jnp.float32)


def adam(settings, steps=None, initial_lr=None):
    transition = settings['transition_steps']
    if settings.get('transition_fraction') is not None:
        transition = max(1, steps // settings['transition_fraction'])
    learning_rate = optax.exponential_decay(
        settings['learning_rate'] if initial_lr is None else initial_lr,
        transition, settings['decay_rate'])
    # Zero weight decay is algebraically Adam; preserve the archived transform.
    return optax.adamw(learning_rate, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0)


class Experiment:
    def __init__(self, config, output, save_operator=False):
        self.config = config
        self.problem = config['problem']
        if self.problem != 'convection':
            raise ValueError('This trainer supports convection only')
        self.output = Path(output)
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f'{self.output}/metrics.json exists; choose a new output directory')
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'checkpoints').mkdir(exist_ok=True)
        self.save_operator = save_operator
        self.start = time.perf_counter()
        self.seed = config['seed']
        self.model = mlp(**config['pinn'])
        self.dimensions = 2
        self.grid = coordinate_grid(self.problem, config['grid_size'])
        self.points = self.grid.reshape(-1, self.dimensions)
        self.archived_2d = config.get('transfer_execution') == 'archived_2d'
        self.predict = jax.jit(self.model.apply)
        self.support_predict = self.model.apply if self.archived_2d else self.predict
        self.fno = operator(self.dimensions, **config['operator']['architecture'])
        self.fno_predict = self.fno.apply if self.archived_2d else jax.jit(self.fno.apply)
        self.fno_params = None
        self.optimizers = {}
        self.records = []
        self.document = {
            'schema_version': 2, 'status': 'running', 'method': 'nocl',
            'problem': self.problem,
            'smoke': config.get('smoke', False), 'config': config,
            'environment': environment(), 'results': self.records,
            'training_counts': {'physics': 0, 'supervised': 0, 'operator': 0},
            'operator_fits': [],
        }
        self.make_supervised_step()
        self.make_operator_step()
        self.archived_warmup_step, self.archived_operator_step = make_2d_transfer_steps(
            self.problem, config['grid_size'])
        self.flush()

    def flush(self):
        self.document['elapsed_seconds'] = time.perf_counter() - self.start
        write_json(self.output / 'metrics.json', self.document)

    def fresh_pinn(self):
        return self.model.init(jax.random.PRNGKey(self.seed),
                               jnp.ones((1, self.dimensions)))

    def physics(self, params, coefficients, steps, guided=False):
        settings = self.config['physics']
        # Preserve the source's Python-float/JAX-scalar distinction.
        alpha = (coefficients[0] if guided and self.config.get('guided_array_coefficient')
                 else float(coefficients[0]))
        initial_lr = settings['learning_rate']
        cache_key = (steps, initial_lr, settings['transition_steps'],
                     settings.get('transition_fraction'))
        if cache_key not in self.optimizers:
            self.optimizers[cache_key] = adam(settings, steps, initial_lr)
        optimizer = self.optimizers[cache_key]
        state = optimizer.init(params)
        for step in range(steps):
            rng = jax.random.PRNGKey(step)
            lb, lr, params, state = convection.training_pinn_step(
                params, self.model, state, optimizer, rng, alpha)
            if (step + 1) % 10000 == 0:
                print(f'  physics {step + 1}/{steps}'
                      f'  boundary={float(lb):.6g} residual={float(lr):.6g}', flush=True)
        jax.block_until_ready(params)
        self.document['training_counts']['physics'] += steps
        return params, {'boundary_loss': float(lb), 'residual_loss': float(lr)}

    def make_supervised_step(self):
        settings = self.config['warmup']
        schedule = optax.exponential_decay(
            settings['learning_rate'], settings['transition_steps'], settings['decay_rate'])
        self.warmup_optimizer = optax.adam(schedule)
        apply, optimizer = self.model.apply, self.warmup_optimizer

        @jax.jit
        def step(params, state, points, values):
            def loss(p):
                return jnp.mean((jnp.squeeze(apply(p, points), axis=-1) - values)**2)
            value, gradient = jax.value_and_grad(loss)(params)
            update, state = optimizer.update(gradient, state, params)
            return optax.apply_updates(params, update), state, value
        self.warmup_step = step

    def warmup(self, coefficients):
        field = jnp.squeeze(self.fno_predict(
            self.fno_params, self.grid[None], coefficients[None]), axis=(0, -1))
        keep_ratio = self.config['mask_keep_ratio']
        values = masking.apply_error_mask(field, self.grid, 30.0, keep_ratio).reshape(-1)
        points = self.points
        params = self.fresh_pinn()
        state = self.warmup_optimizer.init(params)
        steps = self.config['warmup']['steps']
        for _ in range(steps):
            if self.archived_2d:
                loss, params, state = self.archived_warmup_step(
                    params, self.model, state, self.warmup_optimizer, values)
            else:
                params, state, loss = self.warmup_step(params, state, points, values)
        jax.block_until_ready(params)
        self.document['training_counts']['supervised'] += steps
        return params, float(loss)

    def make_operator_step(self):
        optimizer = adam(self.config['operator'], self.config['operator']['steps'])
        self.operator_optimizer = optimizer
        apply, grid = self.fno.apply, self.grid

        @jax.jit
        def step(params, state, fields, coefficients):
            def loss(p):
                grids = jnp.repeat(grid[None], fields.shape[0], axis=0)
                prediction = jnp.squeeze(apply(p, grids, coefficients), axis=-1)
                return jnp.mean((prediction - fields)**2)
            value, gradient = jax.value_and_grad(loss)(params)
            update, state = optimizer.update(gradient, state, params)
            return optax.apply_updates(params, update), state, value
        self.operator_step = step

    def fit_operator(self, fields, coefficients):
        settings = self.config['operator']
        fields = jnp.stack(fields)
        coefficients = jnp.stack(coefficients)
        if self.fno_params is None:
            self.fno_params = self.fno.init(jax.random.PRNGKey(self.seed),
                jnp.ones_like(self.grid[None]), jnp.ones((1, coefficients.shape[-1])))
        state = self.operator_optimizer.init(self.fno_params)
        n = len(fields)
        batch_size = min(n, 16)
        start = time.perf_counter()
        print(f'FNO fit: {n} support fields, batch={batch_size}', flush=True)
        for step in range(settings['steps']):
            if batch_size < n:
                key, _ = jax.random.split(jax.random.PRNGKey(step))
                indices = jax.random.choice(key, n, (batch_size,), replace=False)
                batch_fields, batch_coefficients = fields[indices], coefficients[indices]
            else:
                batch_fields, batch_coefficients = fields, coefficients
            if self.archived_2d:
                values = [float(c[0]) for c in batch_coefficients]
                loss, self.fno_params, state = self.archived_operator_step(
                    self.fno_params, self.fno, state, self.operator_optimizer,
                    batch_fields, values)
            else:
                self.fno_params, state, loss = self.operator_step(
                    self.fno_params, state, batch_fields, batch_coefficients)
            if (step + 1) % 500 == 0:
                print(f'  operator {step + 1}/{settings["steps"]} loss={float(loss):.6g}', flush=True)
        jax.block_until_ready(self.fno_params)
        self.document['training_counts']['operator'] += settings['steps']
        self.document['operator_fits'].append({
            'support_count': n, 'batch_size': batch_size, 'loss': float(loss),
            'seconds': time.perf_counter() - start,
            'parameter_count': sum(leaf.size for leaf in jax.tree_util.tree_leaves(self.fno_params)),
        })
        self.flush()

    def evaluate(self, params, coefficients, final=False):
        # Final convection metrics use the full evaluation grid.
        size = self.config['evaluation_size']
        if not final:
            size = self.config['grid_size']
        grid = coordinate_grid(self.problem, size)
        points = grid.reshape(-1, self.dimensions)
        chunks = [np.asarray(self.predict(params, points[i:i + 65536]))[:, 0]
                  for i in range(0, len(points), 65536)]
        prediction = jnp.asarray(np.concatenate(chunks)).reshape(grid.shape[:-1])
        truth = reference_values(self.problem, grid, coefficients)
        error = float(jnp.linalg.norm(prediction - truth) / (jnp.linalg.norm(truth) + 1e-12))
        if not np.isfinite(error):
            raise FloatingPointError('Non-finite relative L2 error')
        if final:
            np.savez_compressed(self.output / 'prediction.npz',
                                prediction=np.asarray(prediction), reference=np.asarray(truth),
                                coefficients=np.asarray(coefficients))
        return error, size

    def record(self, params, coefficients, phase, losses, final=False):
        relative_l2, size = self.evaluate(params, coefficients, final)
        filename = f'checkpoints/pinn_{len(self.records):02d}.npz'
        save_params(self.output / filename, params)
        row = {'coefficients': np.asarray(coefficients).tolist(), 'phase': phase,
               'relative_l2': relative_l2, 'evaluation_size': size,
               'checkpoint': filename, **losses}
        self.records.append(row)
        print(f'{phase}: coefficients={row["coefficients"]} RelL2={relative_l2:.9g}', flush=True)
        if final:
            self.document['final_relative_l2'] = relative_l2
            self.document['final_checkpoint'] = filename
        self.flush()
        return self.support_predict(params, self.points)[:, 0].reshape(self.grid.shape[:-1])

    def run(self):
        initial, stages = curriculum(self.config)
        steps = self.config['physics']['steps']
        fields, support = [], []
        for value in initial:
            coefficients = coefficients_for(self.config, value)
            params, losses = self.physics(self.fresh_pinn(), coefficients, steps)
            fields.append(self.record(params, coefficients, 'initial', losses))
            support.append(coefficients)
        for stage_index, stage in enumerate(stages):
            self.fit_operator(fields, support)
            for i, value in enumerate(stage):
                coefficients = coefficients_for(self.config, value)
                params, warmup_loss = self.warmup(coefficients)
                params, losses = self.physics(params, coefficients, steps, guided=True)
                losses['warmup_loss'] = warmup_loss
                final = stage_index == len(stages) - 1 and i == len(stage) - 1
                fields.append(self.record(params, coefficients, 'guided', losses, final))
                support.append(coefficients)
        if self.save_operator:
            save_params(self.output / 'operator.npz', self.fno_params)
        self.document['status'] = 'completed'
        self.flush()
        return self.document
