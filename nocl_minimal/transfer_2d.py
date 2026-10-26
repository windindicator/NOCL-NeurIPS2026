"""2D transfer losses retaining the archived grid construction and JIT boundary."""
from functools import partial

import jax
import jax.numpy as jnp
import optax


def make_steps(problem, size):
    indexing = 'xy'

    def grid():
        x = jnp.linspace(0.0, 2.0 * jnp.pi, size)
        t = jnp.linspace(0.0, 1.0, size)
        return jnp.stack(jnp.meshgrid(x, t, indexing=indexing), axis=-1)

    def operator_loss(params, model, targets, values):
        coordinates = jnp.expand_dims(grid(), axis=0)
        coordinates = jnp.repeat(coordinates, len(values), axis=0)
        coefficients = jnp.expand_dims(jnp.array(values), axis=-1)
        output = jnp.squeeze(model.apply(params, coordinates, coefficients), axis=-1)
        return jnp.mean((output - targets)**2)

    @partial(jax.jit, static_argnames=('optimizer', 'model'))
    def operator_step(params, model, state, optimizer, targets, values):
        loss, gradients = jax.value_and_grad(operator_loss)(params, model, targets, values)
        updates, state = optimizer.update(gradients, state, params)
        return loss, optax.apply_updates(params, updates), state

    def supervised_loss(params, model, targets):
        coordinates = grid().reshape(-1, 2)
        output = jnp.squeeze(model.apply(params, coordinates), axis=-1)
        return jnp.mean(jnp.square(output - targets.reshape(-1)))

    @partial(jax.jit, static_argnames=('optimizer', 'model'))
    def supervised_step(params, model, state, optimizer, targets):
        loss, gradients = jax.value_and_grad(supervised_loss)(params, model, targets)
        updates, state = optimizer.update(gradients, state, params)
        return loss, optax.apply_updates(params, updates), state

    return supervised_step, operator_step
