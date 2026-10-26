"""Archived residual filtering: discarded values become zero supervision targets."""
import jax.numpy as jnp

def compute_derivatives(coords, f):
    dx = coords[0, 1, 0] - coords[0, 0, 0]
    dt = coords[1, 0, 1] - coords[0, 0, 1]
    du_dt = jnp.gradient(f, dt, axis=0)
    du_dx = jnp.gradient(f, dx, axis=1)
    return (du_dx, du_dt)

def compute_pde_residual(du_dx, du_dt, beta):
    return du_dt + beta * du_dx

def apply_error_mask(f, coords, beta, mask_ratio=0.5):
    du_dx, du_dt = compute_derivatives(coords, f)
    residual = compute_pde_residual(du_dx, du_dt, beta)
    error = jnp.abs(residual)
    threshold = jnp.percentile(error, mask_ratio * 100.0)
    mask = error <= threshold
    return jnp.where(mask, f, 0.0)
