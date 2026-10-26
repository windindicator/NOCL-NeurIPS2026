"""Figure 2 derivative-constrained convection problem; archived loss kernels."""
from functools import partial
import jax
import jax.numpy as jnp
import optax
BETA_FIXED = 30.0

@jax.vmap
def target_function(x, alpha):
    return jnp.sin(alpha * (x[0] - BETA_FIXED * x[1]))

def generate_boundary_conditions(alpha, rng, num_points=100):
    x_bottom = jax.random.uniform(rng, shape=(num_points,), minval=0.0, maxval=2.0 * jnp.pi)
    t_bottom = jnp.zeros_like(x_bottom)
    bottom_points = jnp.stack([x_bottom, t_bottom], axis=1)
    bottom_alpha = jnp.full_like(x_bottom, alpha)
    boundary_points = jnp.concatenate([bottom_points], axis=0)
    alpha_array = jnp.concatenate([bottom_alpha], axis=0)
    boundary_values = target_function(boundary_points, alpha_array)
    return (boundary_points, boundary_values)

def loss_for_b(params, model, alpha, rng):
    boundary_points, boundary_values = generate_boundary_conditions(alpha, rng)
    u_output = model(params, boundary_points)
    u_output = jnp.squeeze(u_output, axis=-1)
    loss = jnp.mean((u_output - boundary_values) ** 2)
    return loss

def target_function_r(x, alpha):
    return jnp.sin(alpha * (x[0] - BETA_FIXED * x[1]))

def loss_for_residual(params, model, alpha, rng):
    key_x, key_t = jax.random.split(rng)
    x_samples = jax.random.uniform(key_x, (256,), minval=0.0, maxval=2.0 * jnp.pi)
    t_samples = jax.random.uniform(key_t, (256,), minval=0.0, maxval=1.0)
    x = jnp.stack([x_samples, t_samples], axis=1)
    grad_fn = jax.grad(target_function_r, argnums=0)
    vmap_grad_fn = jax.vmap(grad_fn, in_axes=(0, None))
    gradients = vmap_grad_fn(x, alpha)
    u_d1 = jax.vmap(jax.grad(lambda x: model(params, x)[0]))(x)
    loss = jnp.mean((u_d1 - gradients) ** 2)
    return loss

@partial(jax.jit, static_argnames=('optimizer', 'mlp'))
def training_pinn_step(params, mlp, opt_state, optimizer, rng, alpha):
    loss_r, grads_r = jax.value_and_grad(loss_for_residual)(params, mlp.apply, alpha, rng)
    loss_b, grads_b = jax.value_and_grad(loss_for_b)(params, mlp.apply, alpha, rng)
    grads = jax.tree_util.tree_map(lambda x, y: x + y, grads_r, grads_b)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    loss = loss_r + loss_b
    return (loss_b, loss_r, params, opt_state)
