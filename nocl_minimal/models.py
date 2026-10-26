"""Fourier operators from the original NOCL implementation."""
import haiku as hk
import jax
import jax.numpy as jnp


def mlp(n_layer=4, width=64, output_size=1):
    """The same initialization, module names and tanh layers as the archive."""
    def forward(x):
        initializer = hk.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal')
        for _ in range(n_layer):
            x = hk.Linear(width, w_init=initializer)(x)
            x = jax.nn.tanh(x)
        return hk.Linear(output_size, w_init=initializer)(x)
    return hk.without_apply_rng(hk.transform(forward))


def operator(dimensions, modes=16, channels=64, depth=8):
    """Parameter-conditioned FNO; the input is a coordinate grid."""
    if dimensions != 2:
        raise ValueError("Only two-dimensional operators are included")
    def forward(grid, coefficients):
        embedding = hk.Sequential([
            hk.Linear(128), jax.nn.gelu, hk.Linear(32)])(coefficients)
        cls = FNO2D
        return cls(modes=modes, channels=channels, depth=depth,
                   beta_emb_dim=32)(grid, embedding)
    return hk.without_apply_rng(hk.transform(forward))

class FNO2DBlock(hk.Module):

    def __init__(self, modes, channels, name=None):
        super().__init__(name=name)
        self.modes = modes
        self.channels = channels

    def __call__(self, x):
        B, X, Y, C = x.shape
        residual = x
        x_ft = jnp.fft.rfft2(x, axes=(1, 2))
        x_ft_processed = self.fourier_transform(x_ft)
        x = jnp.fft.irfft2(x_ft_processed, s=(X, Y), axes=(1, 2))
        conv = hk.Conv2D(self.channels, kernel_shape=1, with_bias=False)
        return x + conv(residual)

    def fourier_transform(self, x_ft):
        modes_x = min(self.modes, x_ft.shape[1])
        modes_y = min(self.modes, x_ft.shape[2])
        x_ft_cropped = x_ft[:, :modes_x, :modes_y, :]
        params = hk.get_parameter('fourier_weights', (self.modes, self.modes, self.channels, self.channels), init=hk.initializers.TruncatedNormal(0.02))
        params = params[:modes_x, :modes_y, :, :]
        x_ft_processed = jnp.einsum('bmnc,mncd->bmnd', x_ft_cropped, params)
        x_ft_padded = jnp.zeros_like(x_ft)
        x_ft_padded = x_ft_padded.at[:, :modes_x, :modes_y, :].set(x_ft_processed)
        return x_ft_padded

class FNO2D(hk.Module):

    def __init__(self, modes=16, channels=64, depth=8, beta_emb_dim=32, name=None):
        super().__init__(name=name)
        self.modes = modes
        self.channels = channels
        self.depth = depth
        self.beta_emb_dim = beta_emb_dim

    def __call__(self, grid, beta_emb):
        B, X, Y, _ = grid.shape
        x = hk.Linear(self.channels)(grid)
        beta_expanded = jnp.expand_dims(beta_emb, axis=(1, 2))
        beta_expanded = jnp.tile(beta_expanded, (1, X, Y, 1))
        x = jnp.concatenate([x, beta_expanded], axis=-1)
        x = hk.Linear(self.channels)(x)
        for i in range(self.depth):
            x = FNO2DBlock(self.modes, self.channels, name=f'fno_block_{i}')(x)
            x = jax.nn.gelu(x)
        x = hk.Linear(128)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(1)(x)
